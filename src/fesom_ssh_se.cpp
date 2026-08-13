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
    /* T2 adds: div-gather CSR (node -> adjacent-elem coefficient pairs) and
     * the owned-elem -> 3-edges adjacency for the live harmonic viscosity. */
};

static fesom_se_state s_se;
static int             s_se_alloc = 0;

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
