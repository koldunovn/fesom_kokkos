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
#include "fesom_ale.h"        /* fesom_ale_is_zstar(): the se=>zstar guard   */
#include "fesom_constants.h"  /* FESOM_G, FESOM_PHASE1_DT                    */
#include "fesom_field.hpp"

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
    /* Rings: slot 0 = current level m, 1 = m-1, 2 = m-2. Rotation is a host-
     * side Field-handle swap; kernels capture raw pointers per substep (L109). */
    fesom::Field eta[3];        /* η ring                      [Nn]           */
    fesom::Field Ubt[3];        /* Ū ring, interleaved (u,v)   [2*Ne]         */
    fesom::Field Ubt_mean;      /* ⟨⟨Ū⟩⟩ accumulator           [2*Ne]         */
    fesom::Field Rbar;          /* Σ_k helem·uv_rhs / τ        [2*Ne]         */
    fesom::Field Fbt;           /* constant subcycle forcing   [2*Ne]         */
    fesom::Field Ubt_n;         /* Ū at m=0 (live-viscosity anchor, plan D3) [2*Ne] */
    fesom::Field Ubt_ab2;       /* AB2 transport sum (exact f-cancel, plan D4) [2*Ne] */
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
    s_se.Ubt[0].alloc("se.Ubt.m0",  2 * Ne);
    s_se.Ubt[1].alloc("se.Ubt.m1",  2 * Ne);
    s_se.Ubt[2].alloc("se.Ubt.m2",  2 * Ne);
    s_se.Ubt_mean.alloc("se.Ubt.mean", 2 * Ne);
    s_se.Rbar    .alloc("se.Rbar",     2 * Ne);
    s_se.Fbt     .alloc("se.Fbt",      2 * Ne);
    s_se.Ubt_n   .alloc("se.Ubt.n",    2 * Ne);
    s_se.Ubt_ab2 .alloc("se.Ubt.ab2",  2 * Ne);
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

void fesom_se_step_stub(int step_n, const struct fesom_mesh *mesh,
                        struct fesom_dyn *dyn,
                        const struct fesom_forcing *forcing,
                        struct fesom_partit *p)
{
    (void)mesh; (void)dyn; (void)forcing;
    static int announced = 0;
    if (!announced) {
        announced = 1;
        if (p->mype == 0)
            fprintf(stderr, "[ssh_se] TASK-1 STUB active (step %d): barotropic state FROZEN "
                            "(hbar/eta_n/uv unchanged by the SSH block) — the subcycling "
                            "lands in T3-T5. Not a model.\n", step_n);
    }
}

void fesom_se_free(void)
{
    /* Value-reinit releases the Fields (the D13/L13 fesom_solverinfo_free
     * pattern); must run before Kokkos::finalize(). */
    s_se       = fesom_se_state{};
    s_se_alloc = 0;
}
