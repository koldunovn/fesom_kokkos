#ifndef FESOM_GM_H
#define FESOM_GM_H

#include "fesom_types.h"
#include "fesom_field.hpp"   // M2.5b: DualView-backed storage for the GM scratch (deferred from M1)

struct fesom_mesh;
struct fesom_partit;
struct fesom_aux;
struct fesom_tracers;
struct fesom_dyn;

/*
 * GM / Redi state. Holds the auxiliary fields the parameterisation needs.
 * Mirror of the relevant globals in Fortran's o_ARRAYS module
 * (oce_modules.F90:296+) plus per-step scratch.
 *
 * Memory layout (flat, row-major; Fortran is column-major (comp, nz, n)):
 *   sigma_xy        [N * nl * 2]      (node, level, comp)
 *   neutral_slope   [N * (nl-1) * 3]  (node, level, {x, y, |s|})
 *   slope_tapered   [N * (nl-1) * 3]  (same as above)
 *   fer_tapfac      [N * nl]          (node, level)
 *   fer_gamma       [N * nl * 2]      (node, level, comp)
 *   fer_K           [N * nl]          (node, level) — GM thickness diffusivity
 *   Ki              [N * nl]          (node, level) — Redi diffusivity
 *   fer_C           [N]
 *   fer_scal        [N]
 *
 * All per-node arrays are sized myDim+eDim. Halo coverage is required
 * because downstream readers (init_Redi_GM, fer_solve_Gamma, fer_gamma2vel,
 * tracer-side Redi terms) all loop into the halo.
 *
 * fer_K is initialised to K_GM_max=1000 m²/s at allocation, mirroring
 * Fortran oce_setup_step.F90:934.
 */
typedef struct fesom_gm {
    real_t *sigma_xy;
    real_t *neutral_slope;
    real_t *slope_tapered;
    real_t *fer_tapfac;
    real_t *fer_gamma;
    real_t *fer_K;
    real_t *Ki;
    real_t *fer_C;
    real_t *fer_scal;
    /* G7 scratch — per-element horizontal gradient of the active tracer
     * (Fortran tr_xy from oce_tracer_mod.F90:178-181). One tracer at a
     * time; fesom_diff_ver_part_redi_expl rebuilds and halo-exchanges.
     * Layout: [E_full * (nl-1) * 2], comp innermost. */
    real_t *tr_xy;
    /* G7b scratch — per-node 3D vertical gradient of the active tracer
     * (Fortran tr_z from oce_tracer_mod.F90:222-227). Sized [N * nl]; nz=0
     * (top) and nz=nlevels-1 (bottom) are 0 boundaries. */
    real_t *tr_z;

    /* M2.5b: each persistent array above is OWNED by the matching fesom::Field
     * below; the raw pointer is a NON-OWNING alias re-pointed at field.h() right
     * after field.alloc() in fesom_gm_alloc (the M1.2/D12 / M2.3a KPP pattern).
     * The GM scratch is recomputed in place every step through the raw alias (no
     * buffer swap), so the alias stays valid for the whole run. The device _kk
     * kernels (M2.5b) read/write the .d() views; the host C twins read the .h()
     * alias. fer_K is host-initialised to K_GM_max=1000 at alloc (see below) —
     * overwritten every step by init_Redi_GM before any reader. */
    fesom::Field sigma_xy_fld, neutral_slope_fld, slope_tapered_fld, fer_tapfac_fld;
    fesom::Field fer_gamma_fld, fer_K_fld, Ki_fld, fer_C_fld, fer_scal_fld;
    fesom::Field tr_xy_fld, tr_z_fld;
} fesom_gm;

void fesom_gm_alloc(fesom_gm *g, const struct fesom_mesh *mesh);
void fesom_gm_free (fesom_gm *g);

/* --- Phase G2b: density gradients + neutral slope (per step) ----------- */
/* Forward declarations — implementations land in Phase G2b. */
void fesom_compute_sigma_xy      (struct fesom_aux         *aux,
                                  const struct fesom_tracers *tracers,
                                  const struct fesom_mesh  *mesh,
                                  fesom_gm                 *gm,
                                  struct fesom_partit      *partit);

void fesom_compute_neutral_slope (struct fesom_aux         *aux,
                                  const struct fesom_mesh  *mesh,
                                  fesom_gm                 *gm,
                                  struct fesom_partit      *partit);

/* --- Phase G3: per-step GM/Redi coefficient builder -------------------- */
void fesom_init_redi_gm          (struct fesom_aux         *aux,
                                  const struct fesom_mesh  *mesh,
                                  fesom_gm                 *gm,
                                  struct fesom_partit      *partit);

/* --- Phase G4: streamfunction solve and bolus velocity reconstruction -- */
void fesom_fer_solve_gamma       (const struct fesom_aux   *aux,
                                  const struct fesom_mesh  *mesh,
                                  fesom_gm                 *gm,
                                  struct fesom_partit      *partit);

void fesom_fer_gamma2vel         (struct fesom_dyn         *dyn,
                                  const struct fesom_mesh  *mesh,
                                  const fesom_gm           *gm,
                                  struct fesom_partit      *partit);

/* --- M2.5b: DEVICE (Kokkos) twins of the substep-1b GM chain ----------------
 * Pure compute + modify_device(); the step driver owns the IN rail, the per-kernel
 * sync_host + halo, and the fer_gamma re-push (L30). The host C twins above stay
 * in-tree as the FESOM_KK_VERIFY=gm oracle until M2 closes. Inputs must be
 * device-current (the substep-1b IN rail supplies that, L28). */
void fesom_compute_sigma_xy_kk     (struct fesom_aux           *aux,
                                    const struct fesom_tracers *tracers,
                                    const struct fesom_mesh    *mesh,
                                    fesom_gm                   *gm);
void fesom_compute_neutral_slope_kk(struct fesom_aux           *aux,
                                    const struct fesom_mesh    *mesh,
                                    fesom_gm                   *gm);
void fesom_init_redi_gm_kk         (struct fesom_aux           *aux,
                                    const struct fesom_mesh    *mesh,
                                    fesom_gm                   *gm);
void fesom_fer_solve_gamma_kk      (const struct fesom_aux     *aux,
                                    const struct fesom_mesh    *mesh,
                                    fesom_gm                   *gm);
void fesom_fer_gamma2vel_kk        (struct fesom_dyn           *dyn,
                                    const struct fesom_mesh    *mesh,
                                    const fesom_gm             *gm);

/* --- M2.5b: FESOM_KK_VERIFY=gm per-kernel gates (Serial max|Δ|==0) ---------- */
void fesom_gm_sigma_xy_verify     (struct fesom_aux *aux, const struct fesom_tracers *tracers,
                                   const struct fesom_mesh *mesh, fesom_gm *gm,
                                   struct fesom_partit *partit, int step_n);
void fesom_gm_neutral_slope_verify(struct fesom_aux *aux, const struct fesom_mesh *mesh,
                                   fesom_gm *gm, struct fesom_partit *partit, int step_n);
void fesom_gm_init_redi_verify    (struct fesom_aux *aux, const struct fesom_mesh *mesh,
                                   fesom_gm *gm, struct fesom_partit *partit, int step_n);
void fesom_gm_solve_gamma_verify  (const struct fesom_aux *aux, const struct fesom_mesh *mesh,
                                   fesom_gm *gm, struct fesom_partit *partit, int step_n);
void fesom_gm_gamma2vel_verify    (struct fesom_dyn *dyn, const struct fesom_mesh *mesh,
                                   const fesom_gm *gm, struct fesom_partit *partit, int step_n);

/* --- M2.5b-c: DEVICE twins of the substep-13 Redi diffusion (own their internal
 * tr_xy/tr_z halo D21 brackets; diff_hor is an edge→node atomic_add scatter D22). */
void fesom_diff_ver_part_redi_expl_kk(int                      tr_idx,
                                      fesom_gm                *gm,
                                      const struct fesom_mesh *mesh,
                                      struct fesom_tracers    *tracers,
                                      struct fesom_partit     *partit);
void fesom_diff_part_hor_redi_kk     (int                      tr_idx,
                                      fesom_gm                *gm,
                                      const struct fesom_mesh *mesh,
                                      struct fesom_tracers    *tracers,
                                      struct fesom_partit     *partit);

/* L26 capture-before gate for the combined Redi (driver passes the pre-Redi values). */
#ifdef __cplusplus
#include <vector>
void fesom_gm_redi_verify(int tr_idx, fesom_gm *gm, const struct fesom_aux *aux,
                          const struct fesom_mesh *mesh, struct fesom_tracers *tracers,
                          struct fesom_partit *partit, int step_n,
                          const std::vector<real_t> &pre_redi);
#endif

/* --- Phase G7a: vertical-explicit Redi (oce_ale_tracer.F90:1086-1169)
 *
 * Adds the off-diagonal Redi tensor's vertical projection to T values
 * for the given tracer. Reads valuesold (= pre-step T, matches Fortran
 * `values` at tr_xy build time). Builds gm->tr_xy as scratch and
 * halo-exchanges it. No-op if gm == NULL. */
void fesom_diff_ver_part_redi_expl(int                          tr_idx,
                                   fesom_gm                    *gm,
                                   const struct fesom_aux      *aux,
                                   const struct fesom_mesh     *mesh,
                                   struct fesom_tracers        *tracers,
                                   struct fesom_partit         *partit);

/* --- Phase G7b: horizontal Redi (oce_ale_tracer.F90:1173-1336)
 *
 * Edge-based horizontal Redi/eddy diffusivity flux with 5 partial-cell
 * branches A/B/C/D/E for level mismatches between the two adjacent
 * elements of an edge. Builds gm->tr_z (per-node vertical gradient)
 * and reuses gm->tr_xy from a preceding fesom_diff_ver_part_redi_expl
 * call (same tracer). No-op if gm == NULL.
 *
 * Must be called AFTER fesom_diff_ver_part_redi_expl in the same
 * per-tracer iteration so gm->tr_xy is populated. */
void fesom_diff_part_hor_redi    (int                          tr_idx,
                                  fesom_gm                    *gm,
                                  const struct fesom_aux      *aux,
                                  const struct fesom_mesh     *mesh,
                                  struct fesom_tracers        *tracers,
                                  struct fesom_partit         *partit);

#endif /* FESOM_GM_H */
