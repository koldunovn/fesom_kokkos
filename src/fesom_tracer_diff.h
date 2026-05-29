#ifndef FESOM_TRACER_DIFF_H
#define FESOM_TRACER_DIFF_H

#include "fesom_types.h"
#include <vector>            // M2.7: FESOM_KK_VERIFY capture-before buffers

struct fesom_mesh;
struct fesom_aux;
struct fesom_tracers;
struct fesom_dyn;
struct fesom_forcing;
struct fesom_gm;

/*
 * Implicit vertical diffusion of tracers + surface heat/water flux BC.
 * Literal port of diff_ver_part_impl_ale (oce_ale_tracer.F90:562-1082)
 * for the FCT path (do_wimpl=false), no KPP nonlocal.
 *
 * Phase G5 plumbed the Redi K33 augmentation: when `gm != NULL`,
 * Ki and slope_tapered are read to extend the diagonal vertical
 * diffusivity (Fortran `Ty / Ty1` projection of horizontal-Redi onto
 * the vertical axis). With gm=NULL the routine reduces to its pre-G5
 * behaviour exactly (Ki and st² terms drop to zero).
 *
 * For each tracer (T, S):
 *   1. Per-node TDMA on the column with K_v from PP mixing (+ Redi K33
 *      if gm is non-NULL).
 *   2. Surface BC via bc_surface (oce_ale_tracer.F90:1517-1524):
 *        T:   bc_surface = -dt*(heat_flux/vcpw)        (linfs → is_nonlinfs=0)
 *        S:   bc_surface =  dt*(virtual_salt + relax_salt)   (zero in step 24)
 *   3. tracer += dT (the TDMA solution).
 *
 * Without this stage, surface heat_flux/water_flux are never applied — T/S
 * only respond to advection. Phase 3 step 24 needs this to make JRA55 visible.
 */
void fesom_impl_vert_diff_tracers(const struct fesom_mesh    *mesh,
                                  const struct fesom_aux     *aux,
                                  const struct fesom_forcing *forcing,
                                  struct fesom_tracers       *tracers,
                                  const struct fesom_gm      *gm);

/*
 * M2.7: DEVICE (Kokkos) twin of fesom_impl_vert_diff_tracers — the per-node
 * implicit vertical tracer-diffusion TDMA (T then S) on the device. The Thomas
 * sweep runs sequentially in level inside the per-node lambda over [NL_MAX]
 * scratch (the impl_vert_visc/fer_solve_gamma shape, L31) → race-free, NO scatter
 * → Serial AND OpenMP bit-identical. The DRIVER (substep 13b) owns the IN/OUT rails.
 */
void fesom_impl_vert_diff_tracers_kk(const struct fesom_mesh    *mesh,
                                     const struct fesom_aux     *aux,
                                     const struct fesom_forcing *forcing,
                                     struct fesom_tracers       *tracers,
                                     const struct fesom_gm      *gm);

/*
 * M5.14: DEVICE salinity floor — S := max(S, 0.5) over myDim+eDim × column.
 * Elementwise idempotent clamp (each node writes only its own column → race-free,
 * NO scatter/reduction → bit-identical Serial, OpenMP AND CUDA). Replaces the host
 * floor loop once S is device-resident (the trdiff OUT sync_host is removed). Must be
 * called AFTER the post-trdiff device halo so the owned+halo clamp matches the
 * exchange-then-floor order of the prior host path. Marks values modify_device().
 */
void fesom_salinity_floor_kk(const struct fesom_mesh *mesh,
                             struct fesom_tracers    *tracers);

/*
 * FESOM_KK_VERIFY=trdiff gate: capture-before (L26) on `values` for T and S (the
 * diffusion read-modifies both). Runs the C twin on the restored inputs and asserts
 * the final `values` is bit-identical on Serial; non-intrusive (restores KK state).
 */
void fesom_impl_vert_diff_tracers_verify(const struct fesom_mesh    *mesh,
                                         const struct fesom_aux     *aux,
                                         const struct fesom_forcing *forcing,
                                         struct fesom_tracers       *tracers,
                                         const struct fesom_gm      *gm,
                                         int step_n,
                                         const std::vector<real_t>  &pre_T,
                                         const std::vector<real_t>  &pre_S);

#endif /* FESOM_TRACER_DIFF_H */
