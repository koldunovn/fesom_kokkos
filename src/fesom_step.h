#ifndef FESOM_STEP_H
#define FESOM_STEP_H

#include "fesom_types.h"
#include "fesom_ssh.h"
#include "fesom_tracer_adv.h"

struct fesom_mesh;
struct fesom_aux;
struct fesom_dyn;
struct fesom_tracers;
struct fesom_forcing;

/*
 * Phase 1 timestep driver. Mirrors the sequence in oce_timestep_ale
 * (oce_ale.F90:3339-3943) restricted to the Phase 1 subset:
 *
 *  1. compute_pressure_bv     (EOS, density, hpressure, bvfreq)
 *  2. compute_pressure_force  (PGF — linfs + full cells)
 *  3. mixing                  (compute_vel_nodes → pp_mixing → mo_convect)
 *  4. compute_vel_rhs         (Coriolis AB2 + SSH grad + PGF)
 *  5. impl_vert_visc          (TDMA: implicit visc + wind stress + bottom drag)
 *  6. compute_ssh_rhs         (linfs branch)
 *  7. ssh_solve (CG)          (preconditioned conjugate gradient)
 *  8. update_vel              (uv += uv_rhs - g·θ·dt·∇d_eta)
 *  9. compute_hbar            (transport divergence → ssh_rhs_old, hbar)
 * 10. eta_n update inline     (eta_n = α·hbar + (1-α)·hbar_old)
 * 11. ALE step (linfs)        (hnode_new = hnode; vert_vel from divergence)
 * 12. solve_tracers           (upwind T, then upwind S; ALE reconstruction)
 * 13. commit thickness        (hnode = hnode_new; helem = mean of vertex hnode)
 *
 * Deferred (off in Phase 1, gated by namelist in Fortran):
 *   - sw_alpha_beta (only needed by KPP/GM)
 *   - sigma_xy / neutral_slope (GM/Redi)
 *   - momentum_advection (momadv_opt=2)
 *   - viscosity_filter (no horizontal viscosity)
 *   - GM/Redi bolus velocity update
 *   - update_stiff_mat_ale (linfs branch skips it)
 */
struct fesom_partit;
struct fesom_ice;
struct fesom_jra55;
struct fesom_sss_runoff;
struct fesom_gm;
struct fesom_kpp;
struct fesom_tke;

typedef struct fesom_step_ctx {
    fesom_ssh_stiff           *stiff;
    fesom_solverinfo          *solver;
    fesom_tracer_adv_scratch  *tra_sc;
    struct fesom_partit       *partit;     /* needed for halo exchanges */
    struct fesom_ice          *ice;        /* sea-ice state; NULL = no ice */
    struct fesom_gm           *gm;         /* GM/Redi state; NULL = stub-only path */
    struct fesom_kpp          *kpp;        /* KPP mixing state; used when FESOM_MIX_SCHEME=KPP */
    struct fesom_tke          *tke;        /* CVMix-TKE state; NON-NULL only when FESOM_MIX_SCHEME=TKE
                                            * (allocated on selection, mirroring the C and the
                                            * Fortran's mix_scheme_nmb==5-gated init) */
    /* Optional pointers for the sea-ice thermodynamics path. NULL is fine
     * if the FESOM_NO_ICE_THERMO env knob is set. */
    const struct fesom_jra55      *jra;
    const struct fesom_sss_runoff *sr;
} fesom_step_ctx;

/*
 * step_n: the current step index, 1-based to mirror Fortran. The very first
 * call must use step_n = 1 — compute_vel_rhs uses lfirst-equivalent gating.
 *
 * Returns the CG iteration count for diagnostics.
 */
int fesom_timestep(int                          step_n,
                   const fesom_step_ctx        *ctx,
                   struct fesom_mesh           *mesh,
                   struct fesom_aux            *aux,
                   struct fesom_dyn            *dyn,
                   struct fesom_tracers        *tracers,
                   const struct fesom_forcing  *forcing);

/* M7 H.9 SSHRAILS — resolves + guards once (requires IOACC; aborts on the per-step host
 * readers of the SSH class). Under it, ssh_rhs/d_eta/ssh_rhs_old/hbar/hbar_old go
 * device-authoritative and eta_n is device-written; callers outside fesom_step.cpp
 * (fesom_ice.cpp:634) gate their hbar push on it. */
bool fesom_sshrails_on(void);

#endif /* FESOM_STEP_H */

/* M8 forensic instrument (FESOM_MP_NANSCAN=1) — see fesom_step.cpp.
 * s4: non-finite (NaN|Inf); _elem/_node variants decode culprit position + geo coords. */
int  fesom_mp_nanscan_enabled(void);
void fesom_mp_nanscan(const char *phase, const real_t *a, size_t n, int step_n);
void fesom_mp_nanscan_elem(const char *phase, const real_t *a, size_t n, int step_n,
                           const struct fesom_mesh *mesh);
void fesom_mp_nanscan_node(const char *phase, const real_t *a, size_t n, int step_n,
                           const struct fesom_mesh *mesh, int nlev);

/* M8 Gate-3b conservation diagnostic (FESOM_MP_CONSERV=N) — see fesom_step.cpp.
 * Collective (Allreduce): when enabled, call on ALL ranks each diagnostic step. */
struct fesom_tracers;
int  fesom_mp_conserv_every(void);
void fesom_mp_conserv(int step_n, struct fesom_mesh *mesh,
                      struct fesom_tracers *tracers, struct fesom_partit *partit);
