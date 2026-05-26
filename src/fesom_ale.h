#ifndef FESOM_ALE_H
#define FESOM_ALE_H

#include "fesom_types.h"

struct fesom_mesh;
struct fesom_dyn;

/*
 * ALE (Arbitrary Lagrangian-Eulerian) step. Phase 1 implements only the
 * `which_ALE = 'linfs'` branch:
 *
 *   - hnode_new = hnode (no thickness motion)
 *   - w computed by accumulating horizontal edge fluxes over the two
 *     adjacent cells (Gauss-law div integration), then cumsum vertically
 *     from bottom up, then division by area(nz, n) to get m/s.
 *
 * Mirror of the linfs path of vert_vel_ale (oce_ale.F90:2132+) plus
 * update_thickness_ale (oce_ale.F90:1035+).
 *
 * Phase G6a: when `gm_on != 0`, the same edge loop also accumulates
 *   dyn->fer_w from dyn->fer_uv. The dyn->w computation path is
 *   bit-identical to the gm_on=0 case, by construction (the fer_w
 *   block adds reads/writes only on dyn->fer_w / dyn->fer_uv — never
 *   touches dyn->w or its inputs). This is the bit-identity off-switch
 *   guarantee tested at G9.
 *
 * Future: zlevel surface-flux correction, zstar layer scaling,
 * partial cells.
 */
void fesom_ale_vert_vel_linfs(const struct fesom_mesh *mesh,
                              struct fesom_dyn        *dyn,
                              int                      gm_on);

/* hnode_new := hnode (linfs invariant). Called at the start of the ALE step. */
void fesom_ale_thickness_linfs(struct fesom_mesh *mesh);

/* Commit at end of timestep: hnode := hnode_new and helem := mean of 3 vertex hnode. */
void fesom_ale_commit_thickness(struct fesom_mesh *mesh);

/*
 * Vertical CFL (compute_CFLz, oce_ale.F90:2935-3022):
 *   CFL_z[nz, n] gets contributions from W at the two interfaces above and
 *   below layer nz, scaled by the appropriate layer thickness.
 *
 * wsplit (compute_Wvel_split, oce_ale.F90:3026-3074):
 *   For each interface where CFL_z > wsplit_maxcfl, split W into explicit
 *   (w_e, used by tracer advection) and implicit (w_i, used by impl_vert_visc)
 *   parts so the tracer scheme stays CFL-bounded. Default behaviour
 *   (CFL ≤ maxcfl) is w_e = w, w_i = 0.
 */
void fesom_ale_compute_cflz(const struct fesom_mesh *mesh,
                            struct fesom_dyn        *dyn);

void fesom_ale_compute_wvel_split(const struct fesom_mesh *mesh,
                                  struct fesom_dyn        *dyn);

/*===========================================================================
 * M2.5 — DEVICE (Kokkos) twins of the five ALE kernels (substeps 12 + 14).
 *
 * Each `_kk` is a verbatim parallel_for port of its C twin above (D19); the C
 * twins stay untouched as the bit-identity oracle. Shapes (derived from the C
 * BODY, L33):
 *   - thickness/commit/cflz/wvel_split = race-free maps (each entity writes only
 *     its own slot/column) → bit-identical on Serial AND OpenMP.
 *   - vert_vel = an EDGE→NODE SCATTER (Kokkos::atomic_add, D22) + a per-node
 *     sequential level cumsum → Serial bit-identical, OpenMP/CUDA climate-close.
 * None has an internal halo (every fesom_exchange_* is a driver halo) → no D21
 * bracket. The `_verify` fns run the C twin beside the device result on the live
 * state and assert max|Δ|==0 on Serial (FESOM_KK_VERIFY=ale).
 *===========================================================================*/
void fesom_ale_thickness_linfs_kk(struct fesom_mesh *mesh);
void fesom_ale_commit_thickness_kk(struct fesom_mesh *mesh);
void fesom_ale_vert_vel_linfs_kk(const struct fesom_mesh *mesh,
                                 struct fesom_dyn        *dyn,
                                 int                      gm_on);
void fesom_ale_compute_cflz_kk(const struct fesom_mesh *mesh,
                               struct fesom_dyn        *dyn);
void fesom_ale_compute_wvel_split_kk(const struct fesom_mesh *mesh,
                                     struct fesom_dyn        *dyn);

void fesom_ale_thickness_verify(struct fesom_mesh *mesh, int step_n);
void fesom_ale_commit_verify(struct fesom_mesh *mesh, int step_n);
void fesom_ale_vert_vel_verify(const struct fesom_mesh *mesh,
                               struct fesom_dyn *dyn, int gm_on, int step_n);
void fesom_ale_compute_cflz_verify(const struct fesom_mesh *mesh,
                                   struct fesom_dyn *dyn, int step_n);
void fesom_ale_compute_wvel_split_verify(const struct fesom_mesh *mesh,
                                         struct fesom_dyn *dyn, int step_n);

#endif /* FESOM_ALE_H */
