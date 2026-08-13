/*
 * fesom_ssh_se.cpp — M12 split-explicit barotropic solver (FESOM_SSH_MODE=se).
 *
 * SM2005 AB3-AM4 dissipative subcycling of the 2-D (η, Ū) system, replacing the
 * semi-implicit CG SSH solve. Spec: ssh_sergey/subcycling.tex; reference Fortran
 * (Demange FB-θ variant only): ssh_sergey/zenodo_se/.../oce_ale_ssh_splitexpl_subcycl.F90.
 * Plan + the G0-G4 gate ladder: docs/plans/20260813-m12-split-explicit.md.
 *
 * TASK-1 state: knob machinery, state allocation, startup CFL check, step stub.
 * The operators (T2), forcing (T3), subcycle loop (T4) and trim (T5) land next.
 */

#include "fesom_ssh_se.h"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_dyn.h"        /* uv / uv_rhs device Fields (T3 forcing)      */
#include "fesom_forcing.h"    /* water_flux (T4 η subcycle)                  */
#include "fesom_ale.h"        /* fesom_ale_is_zstar(): the se=>zstar guard   */
#include "fesom_constants.h"  /* FESOM_G, FESOM_PHASE1_DT                    */
#include "fesom_field.hpp"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   /* fesom_halo_field: the per-substep 2-D exchanges */

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

/*===========================================================================
 * Knobs (proper options: unrecognised values ABORT — the FESOM_ALE/wsplit
 * house rule; a silent fallback selects a different free-surface formulation)
 *===========================================================================*/

#define FESOM_SE_M_DEFAULT 50   /* notes: M=30-50; paper's global case ran 50 */

int fesom_se_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_SSH_MODE");
        if (!e || !e[0] || strcmp(e, "si") == 0) {
            cached = 0;
        } else if (strcmp(e, "se") == 0) {
            cached = 1;
        } else {
            fprintf(stderr, "FESOM_SSH_MODE=%s not supported (si|se) — refusing to guess, "
                            "because guessing silently selects a different free-surface "
                            "formulation\n", e);
            exit(1);
        }
    }
    return cached;
}

int fesom_se_m(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_SE_M");
        if (!e || !e[0]) {
            cached = FESOM_SE_M_DEFAULT;
        } else {
            /* Full-string parse + abort, not atoi: atoi("5o")=5 is a DIFFERENT model. */
            char *end = NULL;
            const long v = strtol(e, &end, 10);
            if (end == e || (end && *end) || v < 1 || v > 100000) {
                fprintf(stderr, "FESOM_SE_M=%s not supported (need an integer in [1,100000]; "
                                "default %d)\n", e, FESOM_SE_M_DEFAULT);
                exit(1);
            }
            cached = (int)v;
        }
    }
    return cached;
}

static int fesom_se_m_force(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_SE_M_FORCE");
        if (!e || !e[0] || strcmp(e, "0") == 0)      cached = 0;
        else if (strcmp(e, "1") == 0)                cached = 1;
        else {
            fprintf(stderr, "FESOM_SE_M_FORCE=%s not supported (0|1)\n", e);
            exit(1);
        }
    }
    return cached;
}

/* SM2005 dissipation χ. Default 0.1 — MEASURED, not guessed: χ=0.05 is
 * UNSTABLE on CORE2 dt1800 real forcing (Antarctic-shelf coastal setdown
 * grows ~2%/step, vertical CFL blows, all-NaN by step ~300; jobs 26935549,
 * 26935626-31), while χ=0.1 saturates at the SI η class (~2.1 m) through
 * 1000 steps (jobs 26935806/07; 4×γ₁ adds nothing beyond χ=0.1). The notes'
 * suggested range was 0.05-0.1 "to be determined experimentally" — this is
 * that experiment's answer for the reference mesh. */
double fesom_se_chi(void)
{
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *e = getenv("FESOM_SE_CHI");
        if (!e || !e[0]) {
            cached = 0.1;
        } else {
            char *end = NULL;
            const double v = strtod(e, &end);
            if (end == e || (end && *end) || !(v > 0.0) || !(v < 0.5)) {
                fprintf(stderr, "FESOM_SE_CHI=%s not supported (need a number in (0,0.5); "
                                "notes suggest 0.05-0.1; default 0.1)\n", e);
                exit(1);
            }
            cached = v;
        }
    }
    return cached;
}

static int fesom_se_visc_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_SE_VISC");
        if (!e || !e[0] || strcmp(e, "1") == 0)      cached = 1;
        else if (strcmp(e, "0") == 0)                cached = 0;
        else { fprintf(stderr, "FESOM_SE_VISC=%s not supported (0|1)\n", e); exit(1); }
    }
    return cached;
}

static double se_visc_gamma(const char *envname, double deflt)
{
    const char *e = getenv(envname);
    if (!e || !e[0]) return deflt;
    char *end = NULL;
    const double v = strtod(e, &end);
    if (end == e || (end && *end) || v < 0.0) {
        fprintf(stderr, "%s=%s not supported (need a number >= 0; Zenodo SE default %g)\n",
                envname, e, deflt);
        exit(1);
    }
    return v;
}

void fesom_se_mode_init(void)
{
    if (!fesom_se_on()) return;

    /* se => zstar: the SE thickness handling IS z* (plan D2). Under linfs the
     * elevation is decoupled from layer thicknesses — a different model. */
    if (!fesom_ale_is_zstar()) {
        fprintf(stderr, "FESOM_SSH_MODE=se REQUIRES FESOM_ALE=zstar (the SE thickness law "
                        "is z*; linfs pins the layers and decouples them from η)\n");
        exit(1);
    }

    /* SI-block introspection diagnostics read ssh_rhs/d_eta, which se never
     * writes — stale zeros presented as model state. Abort, don't confuse
     * (the fesom_step.cpp:130-149 incompat pattern). */
    static const char *const kDiag[] = { "FESOM_DIAG_SSHSLV", "FESOM_DIAG_SPREAD" };
    for (size_t i = 0; i < sizeof(kDiag)/sizeof(kDiag[0]); ++i) {
        if (getenv(kDiag[i])) {
            fprintf(stderr, "FESOM_SSH_MODE=se is INCOMPATIBLE with %s: that diagnostic "
                            "reads the SI-block fields (ssh_rhs/d_eta), which se never "
                            "writes\n", kDiag[i]);
            exit(1);
        }
    }
    const char *v = getenv("FESOM_KK_VERIFY");
    if (v && strstr(v, "ssh")) {   /* no substring collision: ssh ⊄ any other key */
        fprintf(stderr, "FESOM_SSH_MODE=se is INCOMPATIBLE with FESOM_KK_VERIFY=ssh: the "
                        "verify twins re-run the SI block (RHS/CG/update_vel/hbar), which "
                        "se replaces\n");
        exit(1);
    }
}

/*===========================================================================
 * SE state — device-resident Fields, module-owned (the CGPIPE pattern).
 * All 2-D: rings [Nn] / [2*Ne]; memory is trivial next to the 3-D state.
 *===========================================================================*/

struct fesom_se_state {
    int M = 0;              /* substeps per baroclinic step (validated knob)  */
    /* Rings: slot 0 = current level m, 1 = m-1, 2 = m-2 (η also keeps slot 3
     * as the write target for m+1 — AM4 reads FOUR levels simultaneously, so
     * the η ring is size 4; AB3 reads three, so Ū stays 3 and the new level
     * overwrites the retiring m-2 buffer after AB3 consumed it). Rotation is
     * a host-side Field-handle swap; kernels capture raw pointers (L109). */
    fesom::Field eta[4];        /* η ring                      [Nn]           */
    fesom::Field Ubt[3];        /* Ū ring, interleaved (u,v)   [2*Ne]         */
    fesom::Field Ubt_mean;      /* ⟨⟨Ū⟩⟩ accumulator           [2*Ne]         */
    fesom::Field Rbar;          /* Σ_k helem·uv_rhs / τ        [2*Ne]         */
    fesom::Field Fbt;           /* constant subcycle forcing   [2*Ne]         */
    fesom::Field Ubt_n;         /* Ū at m=0 (live-viscosity anchor, plan D3) [2*Ne] */
    fesom::Field Ubt_ab2;       /* AB2 transport sum (exact f-cancel, plan D4) [2*Ne] */
    fesom::Field Ubt_now;       /* Σₖ hⁿ·uv^{n−1/2} this step               [2*Ne] */
    fesom::Field Ubt_prev;      /* last step's Ubt_now (the D4 old term)     [2*Ne] */
    fesom::Field H0e;           /* resting column depth at elements [Ne]      */

    /* T2 — deterministic 2-D operators.
     *
     * div-gather CSR over OWNED nodes: T_n = Σ_k c[2k..2k+1]·Ū[2·elem_k..+1]
     * with the coefficients assembled from the SAME per-edge cross-segment
     * fluxes as the SI edge scatter (compute_hbar/compute_ssh_rhs pattern:
     * c1 = (v·dx1 − u·dy1), c2 = −(v·dx2 − u·dy2), node1 += c1+c2, node2 −=),
     * summed per (node, elem) pair and PRE-DIVIDED by areasvol[top], so
     * T = +∂η/∂t convention (hbar += T·dt exactly as compute_hbar).
     * Entries of each row are sorted by GLOBAL element id — the summation
     * order is then partition-invariant, which is what makes the T7
     * "MPI-invariance exact 0.0" claim provable. No atomics anywhere.
     * Cavity rows (ulevels_nod2D>1) are left empty (T=0, the hbar guard). */
    fesom::IntField div_off;    /* [myDim_nod2D + 1]                          */
    fesom::IntField div_elem;   /* [nnz]  local element index                 */
    fesom::Field    div_c;      /* [2*nnz] (cx, cy) interleaved               */

    /* T7 check state (host-side, FESOM_SE_CHECK only). */
    std::vector<double> sumh0;  /* Σₖ hnode at first SE step, per owned node  */
    double eta_area0 = 0.0;     /* ∫η dA at first step (post-finalize)        */
    double wf_area_acc = 0.0;   /* accumulated τ·∫W dA                        */
    int    chk_started = 0;

    fesom::Field Tchk;          /* FESOM_SE_CHECK scratch: T(⟨⟨Ū⟩⟩) [Nown] —
                                 * module-owned so it dies BEFORE Kokkos::finalize
                                 * (a function-local static Field aborts at exit). */

    /* Owned-element -> 3 edges + 3 edge-neighbour elements (T4 live harmonic
     * viscosity gather; nb = -1 on genuine boundary edges). Assembled from
     * edge_tri inversion over the myDim+eDim edge range — complete for OWNED
     * elements by the partition construction. */
    fesom::IntField elem_edges; /* [3*myDim_elem2D]                           */
    fesom::IntField elem_nb;    /* [3*myDim_elem2D]                           */
};

static fesom_se_state s_se;
static int             s_se_alloc = 0;

/* T2 (defined below, called from fesom_se_startup) */
static void se_build_operators(const struct fesom_mesh *mesh, struct fesom_partit *p);
static void se_selftest(const struct fesom_mesh *mesh, struct fesom_partit *p);

void fesom_se_startup(const struct fesom_mesh *mesh, struct fesom_partit *p)
{
    if (!fesom_se_on()) {
        /* L80/L102 dead-knob note: a sub-knob set while the mode is off does nothing. */
        if (p->mype == 0 && (getenv("FESOM_SE_M") || getenv("FESOM_SE_M_FORCE")))
            fprintf(stderr, "[ssh_se] NOTE: FESOM_SE_* is set but FESOM_SSH_MODE is not 'se' "
                            "— the knobs have no effect (the SE solver never runs).\n");
        return;
    }

    /* Barotropic gravity-wave CFL over OWNED, non-cavity surface nodes:
     *   dtbt_lim(n) = 0.5 · resolution(n) / sqrt(g·H(n)),  H = -zbar[nlev-1]
     * (0.5 = conservative safety for the FB-family stability bound; the exact
     * admissible Courant number is a G3 calibration item, not startup logic).
     * mesh_resolution is the Voronoi diameter; mesh->depth is input metadata
     * only (fesom_mesh.h:79-80), so depth comes from the level grid. */
    double dtbt_lim = 1e30;
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        if (mesh->ulevels_nod2D[n] != 1) continue;               /* cavity column */
        const int nlev = mesh->nlevels_nod2D[n];
        if (nlev < 2) continue;
        const double H = -(double)mesh->zbar[nlev - 1];          /* zbar<0 down */
        if (!(H > 0.0)) continue;
        const double c   = sqrt((double)FESOM_G * H);
        const double lim = 0.5 * (double)mesh->mesh_resolution[n] / c;
        if (lim < dtbt_lim) dtbt_lim = lim;
    }
    MPI_Allreduce(MPI_IN_PLACE, &dtbt_lim, 1, MPI_DOUBLE, MPI_MIN, p->MPI_COMM_FESOM);

    const double tau   = (double)FESOM_PHASE1_DT;
    const int    M     = fesom_se_m();
    const double dtbt  = tau / (double)M;
    const int    M_min = (dtbt_lim > 0.0 && dtbt_lim < 1e29)
                       ? (int)ceil(tau / dtbt_lim) : 1;

    if (p->mype == 0) {
        fprintf(stderr, "[ssh_se] FESOM_SSH_MODE = se (SM2005 AB3-AM4 barotropic subcycling)\n");
        fprintf(stderr, "[ssh_se]   M = %d -> dtbt = %.2f s | mesh limit dtbt <= %.2f s "
                        "(0.5·res/sqrt(gH), min over mesh) -> M_min = %d\n",
                M, dtbt, dtbt_lim, M_min);
    }
    if (dtbt > dtbt_lim) {
        if (fesom_se_m_force()) {
            if (p->mype == 0)
                fprintf(stderr, "[ssh_se]   WARNING: dtbt exceeds the mesh limit "
                                "(FESOM_SE_M_FORCE=1 — running anyway; expect trouble).\n");
        } else {
            if (p->mype == 0)
                fprintf(stderr, "[ssh_se]   ABORT: dtbt %.2f s > limit %.2f s. Set "
                                "FESOM_SE_M >= %d (or FESOM_SE_M_FORCE=1 to override).\n",
                        dtbt, dtbt_lim, M_min);
            MPI_Barrier(p->MPI_COMM_FESOM);
            exit(1);
        }
    }

    /* State allocation. Element fields cover owned + eDim ONLY (the ELEM2D
     * exchange fills eDim; eXDim slots of element fields are never filled —
     * plan "owned+eDim" rule). Zero-init == the cold-start flat rings (η⁰=0,
     * Ū⁰=0 — the port's IC memsets eta/uv to zero; a nonzero-η IC would need
     * an explicit ring init here, notes p.95). */
    const size_t Nn = (size_t)(mesh->myDim_nod2D  + mesh->eDim_nod2D);
    const size_t Ne = (size_t)(mesh->myDim_elem2D + mesh->eDim_elem2D);
    s_se.eta[0].alloc("se.eta.m0",  Nn);
    s_se.eta[1].alloc("se.eta.m1",  Nn);
    s_se.eta[2].alloc("se.eta.m2",  Nn);
    s_se.eta[3].alloc("se.eta.new", Nn);
    s_se.Ubt[0].alloc("se.Ubt.m0",  2 * Ne);
    s_se.Ubt[1].alloc("se.Ubt.m1",  2 * Ne);
    s_se.Ubt[2].alloc("se.Ubt.m2",  2 * Ne);
    s_se.Ubt_mean.alloc("se.Ubt.mean", 2 * Ne);
    s_se.Rbar    .alloc("se.Rbar",     2 * Ne);
    s_se.Fbt     .alloc("se.Fbt",      2 * Ne);
    s_se.Ubt_n   .alloc("se.Ubt.n",    2 * Ne);
    s_se.Ubt_ab2 .alloc("se.Ubt.ab2",  2 * Ne);
    s_se.Ubt_now .alloc("se.Ubt.now",  2 * Ne);
    s_se.Ubt_prev.alloc("se.Ubt.prev", 2 * Ne);
    s_se.H0e     .alloc("se.H0e",      Ne);
    s_se.M       = M;
    s_se_alloc   = 1;

    /* T2: the deterministic operators (div CSR + viscosity adjacency), then
     * the one-shot operator self-test under FESOM_SE_CHECK. */
    se_build_operators(mesh, p);
    if (fesom_se_check_on())
        se_selftest(mesh, p);
}

/*===========================================================================
 * T2 — deterministic 2-D operators: build + self-test.
 *===========================================================================*/

/* FESOM_SE_CHECK: diagnostic family knob (bare on/off like FESOM_DIAG_*;
 * "0"/unset = off). Serial + a violated invariant => abort (the
 * fesom_ale_verify_report_ pattern). */
int fesom_se_check_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_SE_CHECK");
        cached = (e && e[0] && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return cached;
}

/* Host+device CSR gather: T_n = Σ_k c·Ū over the node's row, in CSR order
 * (identical order on both backends => same-backend determinism; global-id
 * sorted rows => partition invariance). Ubt is interleaved [2*Ne]. */
static void se_div_gather_host(const fesom_se_state *S, const real_t *Ubt,
                               real_t *T, int Nown)
{
    const int    *off = S->div_off.h();
    const int    *el  = S->div_elem.h();
    const real_t *c   = S->div_c.h();
    for (int n = 0; n < Nown; ++n) {
        real_t s = 0.0;
        for (int k = off[n]; k < off[n + 1]; ++k)
            s += c[2*k + 0] * Ubt[2*el[k] + 0]
               + c[2*k + 1] * Ubt[2*el[k] + 1];
        T[n] = s;
    }
}

static void se_div_gather_device(const fesom_se_state *S,
                                 const fesom::Field &Ubt_fld,
                                 fesom::Field &T_fld, int Nown)
{
    /* L109: capture raw pointers, not Field objects (closure stays lean). */
    const int    *off = S->div_off.d().data();
    const int    *el  = S->div_elem.d().data();
    const real_t *c   = S->div_c.d().data();
    const real_t *U   = Ubt_fld.d().data();
    real_t       *T   = T_fld.d().data();
    Kokkos::parallel_for("se_div_gather", Kokkos::RangePolicy<>(0, Nown),
        KOKKOS_LAMBDA(const int n) {
            real_t s = 0.0;
            for (int k = off[n]; k < off[n + 1]; ++k)
                s += c[2*k + 0] * U[2*el[k] + 0]
                   + c[2*k + 1] * U[2*el[k] + 1];
            T[n] = s;
        });
}

/* One-time host assembly of the div CSR + the owned-elem edge/neighbour
 * adjacency; pushed to the device once. Mirrors the SI edge scatter over
 * myDim_edge2D (cross-rank edges are replicated into both owners' myDim
 * ranges, so an OWNED node's stencil is complete from its own rank; el<0 on
 * a myDim edge is a genuine boundary — the "missing halo elem" case of
 * fesom_mesh.h:36-41 only occurs for eDim edges, which we do not iterate). */
static void se_build_operators(const struct fesom_mesh *mesh,
                               struct fesom_partit     *p)
{
    const int Nown = mesh->myDim_nod2D;
    const int nl   = mesh->nl;

    /* --- div CSR ------------------------------------------------------- */
    struct Ent { long gid; int el; double cx, cy; };
    std::vector<std::vector<Ent>> rows((size_t)Nown);

    auto add = [&](int n, int el, double cx, double cy) {
        if (n >= Nown || el < 0) return;                  /* halo row / boundary */
        auto &r = rows[(size_t)n];
        for (auto &e : r)                                  /* ≤ ~8 entries: linear scan */
            if (e.el == el) { e.cx += cx; e.cy += cy; return; }
        r.push_back(Ent{ (long)p->myList_elem2D[el], el, cx, cy });
    };

    for (int ed = 0; ed < mesh->myDim_edge2D; ++ed) {
        const int n1  = mesh->edges[2*ed + 0];
        const int n2  = mesh->edges[2*ed + 1];
        const int el1 = mesh->edge_tri[2*ed + 0];
        const int el2 = mesh->edge_tri[2*ed + 1];
        const double dx1 = (double)mesh->edge_cross_dxdy[4*ed + 0];
        const double dy1 = (double)mesh->edge_cross_dxdy[4*ed + 1];
        const double dx2 = (double)mesh->edge_cross_dxdy[4*ed + 2];
        const double dy2 = (double)mesh->edge_cross_dxdy[4*ed + 3];
        /* c1 = Ū_el1·(−dy1, dx1); c2 = Ū_el2·(dy2, −dx2); n1 += c1+c2, n2 −= */
        add(n1, el1, -dy1,  dx1);   add(n2, el1,  dy1, -dx1);
        add(n1, el2,  dy2, -dx2);   add(n2, el2, -dy2,  dx2);
    }

    size_t nnz = 0;
    for (int n = 0; n < Nown; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) { rows[(size_t)n].clear(); continue; } /* cavity */
        std::sort(rows[(size_t)n].begin(), rows[(size_t)n].end(),
                  [](const Ent &a, const Ent &b) { return a.gid < b.gid; });
        nnz += rows[(size_t)n].size();
    }

    s_se.div_off .alloc("se.div.off",  (size_t)Nown + 1);
    s_se.div_elem.alloc("se.div.elem", nnz ? nnz : 1);
    s_se.div_c   .alloc("se.div.c",    2 * (nnz ? nnz : 1));
    int    *off = s_se.div_off.h();
    int    *elv = s_se.div_elem.h();
    real_t *cv  = s_se.div_c.h();
    size_t k = 0;
    for (int n = 0; n < Nown; ++n) {
        off[n] = (int)k;
        const double inv_a = rows[(size_t)n].empty() ? 0.0
            : 1.0 / (double)mesh->areasvol[FESOM_NODE3D(n, mesh->ulevels_nod2D[n] - 1, nl)];
        for (const auto &e : rows[(size_t)n]) {
            elv[k]      = e.el;
            cv[2*k + 0] = (real_t)(e.cx * inv_a);   /* pre-divided: T = +∂η/∂t */
            cv[2*k + 1] = (real_t)(e.cy * inv_a);
            ++k;
        }
    }
    off[Nown] = (int)k;
    s_se.div_off .modify_host(); s_se.div_off .sync_device();
    s_se.div_elem.modify_host(); s_se.div_elem.sync_device();
    s_se.div_c   .modify_host(); s_se.div_c   .sync_device();

    /* --- owned-elem -> 3 edges / 3 neighbours (T4 viscosity) ------------ */
    const int Eown = mesh->myDim_elem2D;
    s_se.elem_edges.alloc("se.elem.edges", 3 * (size_t)Eown);
    s_se.elem_nb   .alloc("se.elem.nb",    3 * (size_t)Eown);
    int *ee = s_se.elem_edges.h();
    int *en = s_se.elem_nb.h();
    for (size_t i = 0; i < 3 * (size_t)Eown; ++i) { ee[i] = -1; en[i] = -1; }
    std::vector<int> cnt((size_t)Eown, 0);
    for (int ed = 0; ed < mesh->myDim_edge2D + mesh->eDim_edge2D; ++ed) {
        const int el1 = mesh->edge_tri[2*ed + 0];
        const int el2 = mesh->edge_tri[2*ed + 1];
        if (el1 >= 0 && el1 < Eown && cnt[(size_t)el1] < 3) {
            ee[3*el1 + cnt[(size_t)el1]] = ed;
            en[3*el1 + cnt[(size_t)el1]] = el2;            /* -1 = boundary */
            ++cnt[(size_t)el1];
        }
        if (el2 >= 0 && el2 < Eown && cnt[(size_t)el2] < 3) {
            ee[3*el2 + cnt[(size_t)el2]] = ed;
            en[3*el2 + cnt[(size_t)el2]] = el1;
            ++cnt[(size_t)el2];
        }
    }
    int bad = 0;
    for (int e = 0; e < Eown; ++e) if (cnt[(size_t)e] != 3) ++bad;
    if (bad) {
        fprintf(stderr, "[ssh_se] r%d OPERATOR BUILD FAIL: %d owned elements without 3 local "
                        "edges (partition assumption broken)\n", p->mype, bad);
        MPI_Abort(p->MPI_COMM_FESOM, 12);
    }
    s_se.elem_edges.modify_host(); s_se.elem_edges.sync_device();
    s_se.elem_nb   .modify_host(); s_se.elem_nb   .sync_device();
}

/* Integer hash -> [-1,1], a pure function of the GLOBAL element id: the test
 * field is partition-independent and backend-independent by construction. */
static inline double se_hash11(unsigned long long x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return 2.0 * ((double)(x >> 11) / 9007199254740992.0) - 1.0;
}

/* One-shot operator self-test (FESOM_SE_CHECK): random global-id-keyed element
 * field -> (a) literal host edge-scatter reference (the compute_hbar loop with
 * h=1, area-normalised), (b) host CSR gather, (c) device CSR gather. Require
 * max|b-a| and max|c-a| ≤ 1e-14·scale; abort on Serial if violated. */
static void se_selftest(const struct fesom_mesh *mesh, struct fesom_partit *p)
{
    const int Nown = mesh->myDim_nod2D;
    const int Ne   = mesh->myDim_elem2D + mesh->eDim_elem2D;
    const int nl   = mesh->nl;

    fesom::Field Ut;  Ut.alloc("se.test.U", 2 * (size_t)Ne);
    fesom::Field Tt;  Tt.alloc("se.test.T", (size_t)(Nown ? Nown : 1));
    real_t *U = Ut.h();
    for (int e = 0; e < Ne; ++e) {
        const unsigned long long gid = (unsigned long long)p->myList_elem2D[e];
        U[2*e + 0] = (real_t)se_hash11(2ULL * gid);
        U[2*e + 1] = (real_t)se_hash11(2ULL * gid + 1ULL);
    }
    Ut.modify_host(); Ut.sync_device();

    /* (a) scatter reference — literal SI edge loop, h=1, then /areasvol. */
    std::vector<double> ref((size_t)Nown, 0.0);
    for (int ed = 0; ed < mesh->myDim_edge2D; ++ed) {
        const int n1  = mesh->edges[2*ed + 0];
        const int n2  = mesh->edges[2*ed + 1];
        const int el1 = mesh->edge_tri[2*ed + 0];
        const int el2 = mesh->edge_tri[2*ed + 1];
        double c1 = 0.0, c2 = 0.0;
        if (el1 >= 0)
            c1 =  ((double)U[2*el1+1] * (double)mesh->edge_cross_dxdy[4*ed+0]
                 - (double)U[2*el1+0] * (double)mesh->edge_cross_dxdy[4*ed+1]);
        if (el2 >= 0)
            c2 = -((double)U[2*el2+1] * (double)mesh->edge_cross_dxdy[4*ed+2]
                 - (double)U[2*el2+0] * (double)mesh->edge_cross_dxdy[4*ed+3]);
        if (n1 < Nown) ref[(size_t)n1] += c1 + c2;
        if (n2 < Nown) ref[(size_t)n2] -= c1 + c2;
    }
    double scale = 0.0;
    for (int n = 0; n < Nown; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) { ref[(size_t)n] = 0.0; continue; }
        ref[(size_t)n] /= (double)mesh->areasvol[FESOM_NODE3D(n, mesh->ulevels_nod2D[n]-1, nl)];
        if (fabs(ref[(size_t)n]) > scale) scale = fabs(ref[(size_t)n]);
    }

    /* (b) host CSR gather. */
    std::vector<real_t> Th((size_t)(Nown ? Nown : 1), 0.0);
    se_div_gather_host(&s_se, Ut.h(), Th.data(), Nown);

    /* (c) device CSR gather. */
    se_div_gather_device(&s_se, Ut, Tt, Nown);
    Kokkos::fence();
    Tt.modify_device(); Tt.sync_host();

    double dmax_hs = 0.0, dmax_ds = 0.0;
    for (int n = 0; n < Nown; ++n) {
        const double dh = fabs((double)Th[(size_t)n] - ref[(size_t)n]);
        const double dd = fabs((double)Tt.h()[n]     - ref[(size_t)n]);
        if (dh > dmax_hs) dmax_hs = dh;
        if (dd > dmax_ds) dmax_ds = dd;
    }
    double glob[3] = { dmax_hs, dmax_ds, scale };
    MPI_Allreduce(MPI_IN_PLACE, glob, 3, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);

    const double tol = 1e-14 * (glob[2] > 0.0 ? glob[2] : 1.0);
    if (p->mype == 0)
        fprintf(stderr, "[ssh_se] SE_CHECK operator self-test: max|csr_host-scatter|=%.3e "
                        "max|csr_dev-scatter|=%.3e scale=%.3e tol=%.3e -> %s\n",
                glob[0], glob[1], glob[2], tol,
                (glob[0] <= tol && glob[1] <= tol) ? "PASS" : "FAIL");
    if ((glob[0] > tol || glob[1] > tol)
        && strcmp(Kokkos::DefaultExecutionSpace::name(), "Serial") == 0) {
        fprintf(stderr, "[ssh_se] SE_CHECK FAIL on Serial — operator bug by construction. "
                        "Aborting.\n");
        MPI_Barrier(p->MPI_COMM_FESOM);
        abort();
    }
}

/*===========================================================================
 * T3 — forcing assembly: R̄, H_e/H0_e, Ū_AB2, F. Runs once per baroclinic
 * step at SE-block entry, on owned+eDim elements (uv_rhs/uv/helem/η are all
 * eDim-valid there), no communication.
 *
 * F = R̄ − r,  r = −(f e_z×Ū_AB2 + g·H_e·∇ηⁿ):
 *   F_x = R̄_x − f·V_AB2 + g·H_e·∂xηⁿ
 *   F_y = R̄_y + f·U_AB2 + g·H_e·∂yηⁿ
 * Exactness (verified against fesom_momentum.cpp:505-561): the Coriolis inside
 * uv_rhs is +f·(ab1·v_old-couplet + ff_step·v_new) with ab1=−(0.5+ε),
 * ff_step=+(1.5+ε) (1.0 at step 1), ε=0.1 — Ū_AB2 uses the SAME weights over
 * the 2-D transport sums (h-lag on the old term = second-order, plan D4); and
 * the η term inside uv_rhs is exactly dt·(−g·Σ gsᵢ·ηᵢ) (area factors cancel),
 * so g·H_e·∇ηⁿ with H_e = Σₖhelemₖ and the SAME gradient_sca cancels it
 * exactly. No viscosity here: it is live in the substep kernel (plan D3).
 *===========================================================================*/

static void se_forcing(int step_n, const struct fesom_mesh *mesh,
                       const struct fesom_dyn *dyn)
{
    const int    Ne   = mesh->myDim_elem2D + mesh->eDim_elem2D;
    const int    nl   = mesh->nl;
    const real_t tau  = (real_t)FESOM_PHASE1_DT;
    const real_t eps  = 0.1;                       /* fesom_momentum.cpp:76 */
    const real_t wold = (step_n == 1) ? 0.0 : -(0.5 + eps);
    const real_t wnew = (step_n == 1) ? 1.0 :  (1.5 + eps);
    const real_t g    = (real_t)FESOM_G;

    /* L109: raw pointers only. */
    const real_t *uv     = dyn->uv_fld.d().data();
    const real_t *uvrhs  = dyn->uv_rhs_fld.d().data();
    const real_t *helem  = mesh->helem_fld.d().data();
    const int    *ulev   = mesh->ulevels_fld.d().data();
    const int    *nlev   = mesh->nlevels_fld.d().data();
    const int    *elnod  = mesh->elem_nodes_fld.d().data();
    const real_t *gs     = mesh->gradient_sca_fld.d().data();
    const real_t *fcor   = mesh->coriolis_fld.d().data();
    const real_t *eta0   = s_se.eta[0].d().data();       /* ηⁿ, eDim-valid  */
    real_t *Rb   = s_se.Rbar.d().data();
    real_t *Unow = s_se.Ubt_now.d().data();
    real_t *Uprev= s_se.Ubt_prev.d().data();
    real_t *Uab2 = s_se.Ubt_ab2.d().data();
    real_t *F    = s_se.Fbt.d().data();
    real_t *H0e  = s_se.H0e.d().data();

    Kokkos::parallel_for("se_forcing", Kokkos::RangePolicy<>(0, Ne),
        KOKKOS_LAMBDA(const int e) {
            const int nzmin = ulev[e] - 1;
            const int nzmax = nlev[e] - 1;
            real_t rx = 0.0, ry = 0.0, ux = 0.0, uy = 0.0, He = 0.0;
            if (ulev[e] == 1) {                       /* cavity guard (hbar pattern) */
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    const real_t h = helem[FESOM_ELEM3D(e, nz, nl)];
                    const size_t k = FESOM_ELEMVEC(e, nz, nl);
                    rx += h * uvrhs[k + 0];
                    ry += h * uvrhs[k + 1];
                    ux += h * uv[k + 0];
                    uy += h * uv[k + 1];
                    He += h;
                }
            }
            Rb[2*e + 0] = rx / tau;
            Rb[2*e + 1] = ry / tau;
            Unow[2*e + 0] = ux;
            Unow[2*e + 1] = uy;
            const real_t uab = wnew * ux + wold * Uprev[2*e + 0];
            const real_t vab = wnew * uy + wold * Uprev[2*e + 1];
            Uab2[2*e + 0] = uab;
            Uab2[2*e + 1] = vab;

            const int n0 = elnod[3*e + 0];
            const int n1 = elnod[3*e + 1];
            const int n2 = elnod[3*e + 2];
            const real_t e0 = eta0[n0], e1 = eta0[n1], e2 = eta0[n2];
            const real_t gx = gs[6*e+0]*e0 + gs[6*e+1]*e1 + gs[6*e+2]*e2;
            const real_t gy = gs[6*e+3]*e0 + gs[6*e+4]*e1 + gs[6*e+5]*e2;
            const real_t f  = fcor[e];
            F[2*e + 0] = Rb[2*e + 0] - f * vab + g * He * gx;
            F[2*e + 1] = Rb[2*e + 1] + f * uab + g * He * gy;
            H0e[e] = He - (e0 + e1 + e2) / 3.0;
        });
    s_se.Rbar.modify_device();     s_se.Ubt_now.modify_device();
    s_se.Ubt_ab2.modify_device();  s_se.Fbt.modify_device();
    s_se.H0e.modify_device();
}

/* FESOM_SE_CHECK: per-step forcing norms (max over owned elements, global
 * reduction). Diagnostic path — host copies of small 2-D fields are fine. */
static void se_check_forcing(int step_n, const struct fesom_mesh *mesh,
                             struct fesom_partit *p)
{
    const int Eown = mesh->myDim_elem2D;
    s_se.Rbar.sync_host(); s_se.Fbt.sync_host(); s_se.Ubt_ab2.sync_host();
    const real_t *Rb = s_se.Rbar.h();
    const real_t *F  = s_se.Fbt.h();
    double mR = 0.0, mF = 0.0, mr = 0.0;
    for (int e = 0; e < 2 * Eown; ++e) {
        if (fabs((double)Rb[e]) > mR) mR = fabs((double)Rb[e]);
        if (fabs((double)F[e])  > mF) mF = fabs((double)F[e]);
        if (fabs((double)(F[e] - Rb[e])) > mr) mr = fabs((double)(F[e] - Rb[e]));
    }
    double glob[3] = { mR, mF, mr };
    MPI_Allreduce(MPI_IN_PLACE, glob, 3, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);
    if (p->mype == 0)
        fprintf(stderr, "[ssh_se] SE_CHECK step %d forcing: |Rbar|=%.4e |F|=%.4e "
                        "|F-Rbar|(=|r|)=%.4e m2/s2\n", step_n, glob[0], glob[1], glob[2]);
}

/*===========================================================================
 * T4 — the SM2005 AB3-AM4 subcycle loop (subcycling.tex eq. 4_7/4_8/4_12).
 *
 * Per substep m -> m+1: TWO kernels + TWO tiny 2-D exchanges.
 *   k1 (owned+eDim elems): Ū^{AB3} = (3/2+β)Ūᵐ −(1/2+2β)Ū^{m−1} +βŪ^{m−2};
 *                          ⟨⟨Ū⟩⟩ += Ū^{AB3}/M.   (local — history halo-valid)
 *   k2 (OWNED nodes):      η^{m+1} = ηᵐ + (τ/M)(T(Ū^{AB3}) − W), T = the
 *                          deterministic div-gather CSR (T2); cavity frozen.
 *   exchange η^{m+1} (NOD2D scalar).
 *   k3 (OWNED elems):      Ū^{m+1} = Ūᵐ + (τ/M)(−f e_z×Ū^{AB3}
 *                          − g·H^{AM4}∇η^{AM4} + F + visc);
 *                          η^{AM4} = δη^{m+1} +(1−δ−γ−ζ)ηᵐ +γη^{m−1} +ζη^{m−2}
 *                          inline at the 3 vertices; H^{AM4} = H0e + mean(η^{AM4});
 *                          visc = V[Ūᵐ] − V[Ūⁿ] live two-term (plan D3, Zenodo
 *                          closure, coefficients from each term's own state).
 *   exchange Ū^{m+1} (ELEM2D pair, one fused message — the mEVP pattern).
 * Ring rotation is a host-side Field-handle swap; kernels capture raw
 * pointers per substep (L109). No reductions anywhere in the loop.
 *
 * Finalize: η^{n+1}=η^M; BOTH rings carry to the next step (flat-ring reset is
 * cold-start only); hbar_old := hbar, hbar := η^{n+1} (full extent — η is
 * halo-valid) + the MANDATORY host sync of {hbar, hbar_old} (the L86 coherence
 * contract: substep 11 derives eta_n = hbar on host, the :614/ice:646 pushes
 * become identity copies). ⟨⟨Ū⟩⟩ needs no exchange (consumed owned-only by
 * the T5 trim). Returns M (the "iteration count" for the driver).
 *===========================================================================*/

int fesom_se_step(int step_n, struct fesom_mesh *mesh,
                  struct fesom_dyn *dyn,
                  const struct fesom_forcing *forcing,
                  struct fesom_partit *p)
{
    const int    Nown = mesh->myDim_nod2D;
    const int    Ne   = mesh->myDim_elem2D + mesh->eDim_elem2D;
    const int    Eown = mesh->myDim_elem2D;
    const int    M    = s_se.M;
    const real_t tau  = (real_t)FESOM_PHASE1_DT;
    const real_t tauM = tau / (real_t)M;
    const real_t g    = (real_t)FESOM_G;

    /* SM2005 weights (subcycling.tex eq. 4_7): the χ-family. */
    const double chi   = fesom_se_chi();
    const real_t wbeta = 0.281105;
    const real_t wzeta = (real_t)(0.00976186 - 0.13451357 * chi);
    const real_t wgam  = (real_t)(0.083445   - 0.513584   * chi);
    const real_t wdelt = (real_t)(0.5 + 2.0 * (double)wzeta + (double)wgam + 2.0 * chi);
    /* AB3: a0·Ūᵐ + a1·Ū^{m−1} + a2·Ū^{m−2} */
    const real_t a0 = (real_t)(1.5 + (double)wbeta);
    const real_t a1 = (real_t)(-(0.5 + 2.0 * (double)wbeta));
    const real_t a2 = wbeta;
    /* AM4: d3·η^{m+1} + d0·ηᵐ + d1·η^{m−1} + d2·η^{m−2} */
    const real_t d3 = wdelt;
    const real_t d0 = (real_t)(1.0 - (double)wdelt - (double)wgam - (double)wzeta);
    const real_t d1 = wgam;
    const real_t d2 = wzeta;

    const int    visc_on = fesom_se_visc_on();
    const real_t vg0 = (real_t)se_visc_gamma("FESOM_SE_VISC_GAMMA0", 10.0);
    const real_t vg1 = (real_t)se_visc_gamma("FESOM_SE_VISC_GAMMA1", 2750.0);

    static int announced = 0;
    if (!announced) {
        announced = 1;
        if (p->mype == 0)
            fprintf(stderr, "[ssh_se] subcycling live: M=%d chi=%.3f (beta=%.6f zeta=%.6f "
                            "gamma=%.6f delta=%.6f) visc=%d (g0=%.3g g1=%.3g)\n",
                    M, chi, (double)wbeta, (double)wzeta, (double)wgam, (double)wdelt,
                    visc_on, (double)vg0, (double)vg1);
    }

    /* Once per step: forcing (T3) + the m=0 viscosity anchor + zero the mean. */
    se_forcing(step_n, mesh, dyn);
    if (fesom_se_check_on())
        se_check_forcing(step_n, mesh, p);
    Kokkos::deep_copy(s_se.Ubt_n.d(), s_se.Ubt[0].d());   /* Ūⁿ, halo-valid */
    s_se.Ubt_n.modify_device();
    Kokkos::deep_copy(s_se.Ubt_mean.d(), 0.0);
    s_se.Ubt_mean.modify_device();

    /* Mesh/forcing pointers reused across substeps (L109 lean captures). */
    const int    *ulev_n = mesh->ulevels_nod2D_fld.d().data();
    const int    *doff   = s_se.div_off.d().data();
    const int    *delem  = s_se.div_elem.d().data();
    const real_t *dc     = s_se.div_c.d().data();
    const real_t *wf     = forcing->water_flux_fld.d().data();
    const int    *elnod  = mesh->elem_nodes_fld.d().data();
    const real_t *gs     = mesh->gradient_sca_fld.d().data();
    const real_t *fcor   = mesh->coriolis_fld.d().data();
    const real_t *earea  = mesh->elem_area_fld.d().data();
    const int    *enb    = s_se.elem_nb.d().data();
    const real_t *F      = s_se.Fbt.d().data();
    const real_t *H0e    = s_se.H0e.d().data();
    const real_t *Un     = s_se.Ubt_n.d().data();
    real_t       *Umean  = s_se.Ubt_mean.d().data();
    real_t       *Uab3   = s_se.Ubt_ab2.d().data();   /* scratch: Ubt_ab2 was
                                                       * consumed into F by
                                                       * se_forcing — free now */

    for (int m = 0; m < M; ++m) {
        /* k1: AB3 extrapolation + mean accumulation (owned+eDim). */
        {
            const real_t *U0 = s_se.Ubt[0].d().data();
            const real_t *U1 = s_se.Ubt[1].d().data();
            const real_t *U2 = s_se.Ubt[2].d().data();
            const real_t invM = 1.0 / (real_t)M;
            Kokkos::parallel_for("se_ab3", Kokkos::RangePolicy<>(0, 2 * Ne),
                KOKKOS_LAMBDA(const int i) {
                    const real_t uab = a0 * U0[i] + a1 * U1[i] + a2 * U2[i];
                    Uab3[i] = uab;
                    Umean[i] += uab * invM;
                });
        }

        /* k2: η^{m+1} on OWNED nodes into the free ring slot eta[3]. */
        {
            const real_t *e0 = s_se.eta[0].d().data();
            real_t       *en = s_se.eta[3].d().data();
            Kokkos::parallel_for("se_eta", Kokkos::RangePolicy<>(0, Nown),
                KOKKOS_LAMBDA(const int n) {
                    if (ulev_n[n] != 1) { en[n] = e0[n]; return; }   /* cavity frozen */
                    real_t T = 0.0;
                    for (int k = doff[n]; k < doff[n + 1]; ++k)
                        T += dc[2*k + 0] * Uab3[2*delem[k] + 0]
                           + dc[2*k + 1] * Uab3[2*delem[k] + 1];
                    en[n] = e0[n] + tauM * (T - wf[n]);
                });
            s_se.eta[3].modify_device();
        }

        /* η ring rotate: (new, m, m−1, m−2) <- (eta3, eta0, eta1, eta2);
         * then exchange the new current level (replace semantics fills halo). */
        {
            fesom::Field tmp = s_se.eta[3];
            s_se.eta[3] = s_se.eta[2];
            s_se.eta[2] = s_se.eta[1];
            s_se.eta[1] = s_se.eta[0];
            s_se.eta[0] = tmp;
        }
        fesom_halo_field(s_se.eta[0], FESOM_HALO_NOD2D, 1, 1, p);

        /* k3: Ū^{m+1} on OWNED elems into the retiring Ū buffer (Ubt[2] — AB3
         * already consumed it this substep). AM4 inline at the 3 vertices. */
        {
            const real_t *ep1 = s_se.eta[0].d().data();   /* η^{m+1} */
            const real_t *em0 = s_se.eta[1].d().data();   /* ηᵐ      */
            const real_t *em1 = s_se.eta[2].d().data();   /* η^{m−1} */
            const real_t *em2 = s_se.eta[3].d().data();   /* η^{m−2} */
            const real_t *U0  = s_se.Ubt[0].d().data();   /* Ūᵐ (halo-valid) */
            real_t       *Unew = s_se.Ubt[2].d().data();
            /* Viscosity scaling: the whole bracket below is multiplied by
             * τ/M, so vi carries NO dt of its own — this reproduces the
             * reference's per-substep increment BT_inv·(dt·√(…)·ΔŪ/A)
             * exactly ((τ/M)·√(…)·ΔŪ/A). */
            Kokkos::parallel_for("se_ubt", Kokkos::RangePolicy<>(0, Eown),
                KOKKOS_LAMBDA(const int e) {
                    const int n0 = elnod[3*e + 0];
                    const int n1 = elnod[3*e + 1];
                    const int n2 = elnod[3*e + 2];
                    const real_t h0 = d3*ep1[n0] + d0*em0[n0] + d1*em1[n0] + d2*em2[n0];
                    const real_t h1 = d3*ep1[n1] + d0*em0[n1] + d1*em1[n1] + d2*em2[n1];
                    const real_t h2 = d3*ep1[n2] + d0*em0[n2] + d1*em1[n2] + d2*em2[n2];
                    const real_t gx = gs[6*e+0]*h0 + gs[6*e+1]*h1 + gs[6*e+2]*h2;
                    const real_t gy = gs[6*e+3]*h0 + gs[6*e+4]*h1 + gs[6*e+5]*h2;
                    const real_t Ham = H0e[e] + (h0 + h1 + h2) / 3.0;
                    const real_t f   = fcor[e];
                    real_t vx = 0.0, vy = 0.0;
                    if (visc_on) {
                        /* live two-term harmonic viscosity, Zenodo closure:
                         * V[Ūᵐ] − V[Ūⁿ], each coefficient from its own state. */
                        const real_t He_e = H0e[e];   /* ≈ H at n (η part small) */
                        for (int s = 0; s < 3; ++s) {
                            const int nb = enb[3*e + s];
                            if (nb < 0) continue;              /* boundary edge */
                            const real_t len  = Kokkos::sqrt(earea[e] + earea[nb]);
                            const real_t hh   = 0.5 * (He_e + H0e[nb]);
                            const real_t ihh  = 1.0 / hh;
                            /* term at Ūᵐ */
                            real_t dux = (U0[2*e+0] - U0[2*nb+0]);
                            real_t duy = (U0[2*e+1] - U0[2*nb+1]);
                            real_t spd = Kokkos::sqrt(dux*dux + duy*duy) * ihh;
                            real_t vi  = Kokkos::sqrt(
                                             Kokkos::fmax(vg0, vg1 * spd) * len);
                            vx -= dux * vi;  vy -= duy * vi;
                            /* minus the frozen term at Ūⁿ */
                            dux = (Un[2*e+0] - Un[2*nb+0]);
                            duy = (Un[2*e+1] - Un[2*nb+1]);
                            spd = Kokkos::sqrt(dux*dux + duy*duy) * ihh;
                            vi  = Kokkos::sqrt(
                                      Kokkos::fmax(vg0, vg1 * spd) * len);
                            vx += dux * vi;  vy += duy * vi;
                        }
                        const real_t ia = 1.0 / earea[e];
                        vx *= ia;  vy *= ia;
                    }
                    Unew[2*e + 0] = U0[2*e + 0] + tauM * (  f * Uab3[2*e + 1]
                                     - g * Ham * gx + F[2*e + 0] + vx);
                    Unew[2*e + 1] = U0[2*e + 1] + tauM * ( -f * Uab3[2*e + 0]
                                     - g * Ham * gy + F[2*e + 1] + vy);
                });
            s_se.Ubt[2].modify_device();
        }

        /* Ū ring rotate, then exchange the new current level (fused pair). */
        {
            fesom::Field tmp = s_se.Ubt[2];
            s_se.Ubt[2] = s_se.Ubt[1];
            s_se.Ubt[1] = s_se.Ubt[0];
            s_se.Ubt[0] = tmp;
        }
        fesom_halo_field(s_se.Ubt[0], FESOM_HALO_ELEM2D, 1, 2, p);
    }

    /* Finalize: hbar_old := hbar; hbar := η^{n+1} (full extent — η halo-valid,
     * cavity values frozen equal by k2). Device write + the MANDATORY host sync. */
    {
        const int Nn = mesh->myDim_nod2D + mesh->eDim_nod2D;
        const real_t *ef  = s_se.eta[0].d().data();
        real_t *hb  = mesh->hbar_fld.d().data();
        real_t *hbo = mesh->hbar_old_fld.d().data();
        Kokkos::parallel_for("se_hbar", Kokkos::RangePolicy<>(0, Nn),
            KOKKOS_LAMBDA(const int n) {
                hbo[n] = hb[n];
                hb[n]  = ef[n];
            });
        mesh->hbar_fld.modify_device();     mesh->hbar_old_fld.modify_device();
        mesh->hbar_fld.sync_host();         mesh->hbar_old_fld.sync_host();
    }

    /* T5 — corrector trim (subcycling.tex eq. 4_13, hⁿ weights per D1),
     * replacing the SI update_vel: with hⁿ trim weights the correction is
     * DEPTH-UNIFORM in velocity units,
     *   corr = (Σₖ hₖ·(uv+uv_rhs)ₖ − ⟨⟨Ū⟩⟩) / Σₖhₖ,
     *   uvₖ := (uv+uv_rhs)ₖ − corr,
     * which makes Σₖ uvₖ·helemₖ = ⟨⟨Ū⟩⟩ exact to rounding (G1b) and keeps
     * the tracer fluxes uv·helem consistent with the η update. Cavity
     * columns get the plain predictor (no trim — the hbar pattern). OWNED
     * elements; the ELEM3D halo below fills eDim (the substep-9 slot). */
    {
        const real_t *uvrhs = dyn->uv_rhs_fld.d().data();
        const real_t *helem = mesh->helem_fld.d().data();
        const int    *ulev  = mesh->ulevels_fld.d().data();
        const int    *nlev  = mesh->nlevels_fld.d().data();
        const real_t *Um    = s_se.Ubt_mean.d().data();
        real_t       *uv    = dyn->uv_fld.d().data();
        const int     nl    = mesh->nl;
        Kokkos::parallel_for("se_trim", Kokkos::RangePolicy<>(0, Eown),
            KOKKOS_LAMBDA(const int e) {
                const int nzmin = ulev[e] - 1;
                const int nzmax = nlev[e] - 1;
                real_t sx = 0.0, sy = 0.0, He = 0.0;
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    const real_t h = helem[FESOM_ELEM3D(e, nz, nl)];
                    const size_t k = FESOM_ELEMVEC(e, nz, nl);
                    sx += h * (uv[k + 0] + uvrhs[k + 0]);
                    sy += h * (uv[k + 1] + uvrhs[k + 1]);
                    He += h;
                }
                real_t cx = 0.0, cy = 0.0;
                if (ulev[e] == 1 && He > 0.0) {
                    cx = (sx - Um[2*e + 0]) / He;
                    cy = (sy - Um[2*e + 1]) / He;
                }
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    const size_t k = FESOM_ELEMVEC(e, nz, nl);
                    uv[k + 0] = uv[k + 0] + uvrhs[k + 0] - cx;
                    uv[k + 1] = uv[k + 1] + uvrhs[k + 1] - cy;
                }
            });
        dyn->uv_fld.modify_device();
    }
    fesom_halo_field(dyn->uv_fld, FESOM_HALO_ELEM3D, mesh->nl, 2, p);

    /* G1b (FESOM_SE_CHECK): ‖Σₖ uvₖ·helemₖ − ⟨⟨Ū⟩⟩‖∞ after the trim — exact
     * to rounding by the trim algebra. Check-only reduction (the production
     * path stays reduction-free). Provisional Serial abort 1e-9 m²/s. */
    if (fesom_se_check_on()) {
        const real_t *uvv   = dyn->uv_fld.d().data();
        const real_t *helem = mesh->helem_fld.d().data();
        const int    *ulev  = mesh->ulevels_fld.d().data();
        const int    *nlev  = mesh->nlevels_fld.d().data();
        const real_t *Um    = s_se.Ubt_mean.d().data();
        const int     nl    = mesh->nl;
        double dmax = 0.0;
        Kokkos::parallel_reduce("se_g1b", Kokkos::RangePolicy<>(0, Eown),
            KOKKOS_LAMBDA(const int e, double &mx) {
                if (ulev[e] != 1) return;
                const int nzmin = ulev[e] - 1;
                const int nzmax = nlev[e] - 1;
                real_t sx = 0.0, sy = 0.0;
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    const real_t h = helem[FESOM_ELEM3D(e, nz, nl)];
                    const size_t k = FESOM_ELEMVEC(e, nz, nl);
                    sx += h * uvv[k + 0];
                    sy += h * uvv[k + 1];
                }
                const double rx = fabs((double)(sx - Um[2*e + 0]));
                const double ry = fabs((double)(sy - Um[2*e + 1]));
                if (rx > mx) mx = rx;
                if (ry > mx) mx = ry;
            }, Kokkos::Max<double>(dmax));
        MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);
        if (p->mype == 0)
            fprintf(stderr, "[ssh_se] SE_CHECK step %d G1b trim-consistency max|r|=%.3e m2/s\n",
                    step_n, dmax);
        if (dmax > 1e-9
            && strcmp(Kokkos::DefaultExecutionSpace::name(), "Serial") == 0) {
            fprintf(stderr, "[ssh_se] SE_CHECK G1b FAIL on Serial (provisional 1e-9). "
                            "Aborting.\n");
            MPI_Barrier(p->MPI_COMM_FESOM);
            abort();
        }
    }

    /* D4 history carry. */
    std::swap(s_se.Ubt_prev, s_se.Ubt_now);

    /* G1a (FESOM_SE_CHECK): the η-compatibility invariant
     *   ‖η^{n+1} − ηⁿ − τ·(T(⟨⟨Ū⟩⟩) − W)‖∞  over owned non-cavity nodes.
     * Machine-precision class by construction (linearity of T; only FP
     * reassociation between the per-substep sum and the mean). Provisional
     * abort threshold 1e-9 m on Serial until first measurements freeze it. */
    if (fesom_se_check_on()) {
        if (!s_se.Tchk.allocated())
            s_se.Tchk.alloc("se.check.T", (size_t)(Nown ? Nown : 1));
        fesom::Field &Tchk = s_se.Tchk;
        se_div_gather_device(&s_se, s_se.Ubt_mean, Tchk, Nown);
        Kokkos::fence();
        Tchk.modify_device(); Tchk.sync_host();
        /* const_cast precedent: fesom_step.cpp:702 (stress_surf push). water_flux
         * host is current on the default rails; the sync is a no-op guard. */
        const_cast<struct fesom_forcing *>(forcing)->water_flux_fld.sync_host();
        const real_t *hb  = mesh->hbar_fld.h();      /* η^{n+1} (just synced) */
        const real_t *hbo = mesh->hbar_old_fld.h();  /* ηⁿ                    */
        const real_t *wfh = forcing->water_flux_fld.h();
        double dmax = 0.0;
        long   nan_n = 0;                    /* NaN-LOUD: max-reductions are NaN-blind
                                              * (comparisons with NaN are false) — count
                                              * NaNs explicitly (t6 lesson: the CORE2 smoke
                                              * died all-NaN under a '0.000 PASS' banner). */
        for (int n = 0; n < Nown; ++n) {
            if (mesh->ulevels_nod2D[n] != 1) continue;
            const double r = ((double)hb[n] - (double)hbo[n])
                           - (double)tau * ((double)Tchk.h()[n] - (double)wfh[n]);
            if (r != r || hb[n] != hb[n]) { ++nan_n; continue; }
            if (fabs(r) > dmax) dmax = fabs(r);
        }
        double glob2[2] = { dmax, (double)nan_n };
        MPI_Allreduce(MPI_IN_PLACE, glob2, 2, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);
        dmax = glob2[0];
        if (p->mype == 0)
            fprintf(stderr, "[ssh_se] SE_CHECK step %d G1a eta-compat max|r|=%.3e m%s\n",
                    step_n, dmax, glob2[1] > 0.0 ? "  *** NaN IN ETA ***" : "");
        if (glob2[1] > 0.0) {
            if (p->mype == 0)
                fprintf(stderr, "[ssh_se] SE_CHECK step %d: NaN detected in the barotropic "
                                "state — aborting loudly instead of running dead.\n", step_n);
            MPI_Barrier(p->MPI_COMM_FESOM);
            abort();
        }
        if (dmax > 1e-9
            && strcmp(Kokkos::DefaultExecutionSpace::name(), "Serial") == 0) {
            fprintf(stderr, "[ssh_se] SE_CHECK G1a FAIL on Serial (provisional 1e-9 m). "
                            "Aborting.\n");
            MPI_Barrier(p->MPI_COMM_FESOM);
            abort();
        }
    }
    /* T7 (FESOM_SE_CHECK): (a) cumulative layer-sum identity — the incremental
     * z* law gives Σₖδh = Δη exactly per step, so Σₖhnode − Σₖhnode⁰ ≡ η
     * cumulatively; (b) volume conservation — Σ_edges cancels pairwise, so
     * ∫η dA − ∫η⁰ dA ≡ −Σsteps τ·∫W dA at rounding. Host path, owned nodes.
     * NOTE: checked at SE-block ENTRY next step (hnode commits in substep 14,
     * AFTER this block) — hence the one-step offset bookkeeping below. */
    if (fesom_se_check_on()) {
        mesh->hnode_fld.sync_host();
        const real_t *hn  = mesh->hnode_fld.h();
        const real_t *hb  = mesh->hbar_fld.h();      /* η^{n+1}, synced above */
        const real_t *hbo = mesh->hbar_old_fld.h();  /* ηⁿ                    */
        const real_t *wfh = forcing->water_flux_fld.h();  /* synced in G1a    */
        const int nl = mesh->nl;
        if (!s_se.chk_started) {
            s_se.chk_started = 1;
            s_se.sumh0.assign((size_t)Nown, 0.0);
            double ea = 0.0;
            for (int n = 0; n < Nown; ++n) {
                if (mesh->ulevels_nod2D[n] != 1) continue;
                double sh = 0.0;
                for (int k = mesh->ulevels_nod2D[n]-1; k < mesh->nlevels_nod2D[n]-1; ++k)
                    sh += (double)hn[FESOM_NODE3D(n, k, nl)];
                /* hnode is PRE-step here; hbar_old = ηⁿ of this first step */
                s_se.sumh0[(size_t)n] = sh - (double)hbo[n];
                ea += (double)hbo[n]
                    * (double)mesh->areasvol[FESOM_NODE3D(n, mesh->ulevels_nod2D[n]-1, nl)];
            }
            MPI_Allreduce(MPI_IN_PLACE, &ea, 1, MPI_DOUBLE, MPI_SUM, p->MPI_COMM_FESOM);
            s_se.eta_area0 = ea;
        } else {
            double dmax = 0.0;
            for (int n = 0; n < Nown; ++n) {
                if (mesh->ulevels_nod2D[n] != 1) continue;
                double sh = 0.0;
                for (int k = mesh->ulevels_nod2D[n]-1; k < mesh->nlevels_nod2D[n]-1; ++k)
                    sh += (double)hn[FESOM_NODE3D(n, k, nl)];
                /* hnode currently holds the LAST committed state = step n's
                 * thicknesses; the matching elevation is hbar_old (ηⁿ). */
                const double r = fabs(sh - s_se.sumh0[(size_t)n] - (double)hbo[n]);
                if (r > dmax) dmax = r;
            }
            MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);
            if (p->mype == 0)
                fprintf(stderr, "[ssh_se] SE_CHECK step %d G1c layer-sum max|Σh-Σh0-eta|=%.3e m\n",
                        step_n, dmax);
        }
        /* Volume window: compare ∫η^{n+1} dA against ∫η⁰ dA − Σ τ·∫W dA. */
        double loc[2] = { 0.0, 0.0 };
        for (int n = 0; n < Nown; ++n) {
            if (mesh->ulevels_nod2D[n] != 1) continue;
            const double A = (double)mesh->areasvol[FESOM_NODE3D(n, mesh->ulevels_nod2D[n]-1, nl)];
            loc[0] += (double)hb[n] * A;
            loc[1] += (double)wfh[n] * A;
        }
        MPI_Allreduce(MPI_IN_PLACE, loc, 2, MPI_DOUBLE, MPI_SUM, p->MPI_COMM_FESOM);
        s_se.wf_area_acc += (double)tau * loc[1];
        const double drift = (loc[0] - s_se.eta_area0) + s_se.wf_area_acc;
        if (p->mype == 0)
            fprintf(stderr, "[ssh_se] SE_CHECK step %d G1d volume drift=%.4e m3 (eta-int %.4e, "
                            "Wacc %.4e)\n", step_n, drift, loc[0] - s_se.eta_area0,
                    s_se.wf_area_acc);
    }

    /* T8 (FESOM_SE_DUMP=<prefix>): per-rank raw dump of the barotropic state
     * (η^{n+1} owned + Ū^{n+1} owned) every step, overwriting — byte-compare
     * across runs/backends for the G2 determinism gates. Check-path only. */
    {
        static const char *dump = getenv("FESOM_SE_DUMP");
        if (dump && dump[0]) {
            s_se.eta[0].sync_host();
            s_se.Ubt[0].sync_host();
            char fn[512];
            snprintf(fn, sizeof fn, "%s.r%04d.bin", dump, p->mype);
            FILE *f = fopen(fn, "wb");
            if (f) {
                fwrite(&step_n, sizeof(int), 1, f);
                fwrite(s_se.eta[0].h(), sizeof(real_t), (size_t)Nown, f);
                fwrite(s_se.Ubt[0].h(), sizeof(real_t), 2 * (size_t)Eown, f);
                fclose(f);
            }
        }
    }

    (void)dyn;
    return M;
}

void fesom_se_free(void)
{
    /* Value-reinit releases the Fields (the D13/L13 fesom_solverinfo_free
     * pattern); must run before Kokkos::finalize(). */
    s_se       = fesom_se_state{};
    s_se_alloc = 0;
}
