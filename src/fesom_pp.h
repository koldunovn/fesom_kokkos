#ifndef FESOM_PP_H
#define FESOM_PP_H

#include "fesom_types.h"

struct fesom_mesh;
struct fesom_aux;
struct fesom_dyn;

/*
 * Pacanowski-Philander vertical mixing + convective adjustment.
 * Literal ports of:
 *   compute_vel_nodes  (oce_dyn.F90:177-226)
 *   oce_mixing_pp      (oce_ale_mixing_pp.F90:2-93)
 *   mo_convect         (oce_mo_conv.F90:78-120, use_instabmix branch only)
 *
 * Phase 1 deviations from Fortran defaults:
 *   - use_momix = false   (Fortran default true; momix needs ice + forcing)
 *   - use_windmix = false (matches default)
 *   - Kv0_const = true    (matches default; we don't port the lat/depth Kv0)
 *
 * compute_vel_nodes interpolates the cell-centred UV onto nodes via an
 * area-weighted mean over surrounding elements. PP uses the resulting
 * UVnode to compute vertical shear at scalar locations.
 */
void fesom_compute_vel_nodes(const struct fesom_mesh *mesh,
                             struct fesom_dyn        *dyn);

void fesom_pp_mixing(const struct fesom_mesh *mesh,
                     const struct fesom_dyn  *dyn,
                     struct fesom_aux        *aux);

/* Convective adjustment: where N² < 0, raise Kv (and Av at adjacent cells)
   to instabmix_kv. Mirror of the use_instabmix branch in mo_convect. */
void fesom_mo_convect(const struct fesom_mesh *mesh,
                      struct fesom_aux        *aux);

/*
 * M2.2 DEVICE kernel twins (Kokkos parallel_for; the M2.1 EOS template, D19).
 * Production path from M2.2. Each requires its inputs device-current (the step
 * driver's substep-3 sync rail supplies that — see docs/SYNC_MAP.md §2 row 3) and
 * marks its outputs modify_device(). The host C twins above stay in-tree untouched
 * as the FESOM_KK_VERIFY=pp oracle until M2 closes.
 *   - compute_vel_nodes_kk: in dyn->uv (per-step sync) → out dyn->uvnode
 *   - pp_mixing_kk:         in dyn->uvnode, aux->bvfreq → out aux->Kv (node), Av (elem)
 *   - mo_convect_kk:        in aux->bvfreq,Kv,Av        → out aux->Kv, Av
 * ⚠️ mo_convect_kk is the first device reader of bvfreq after the host smooth_nod3D
 *    (substep 1) → its driver rail uses modify_host()+sync_device() on bvfreq, not
 *    a bare sync_device() (L14). pp_mixing keeps the loop-2-before-loop-3 ordering.
 */
void fesom_compute_vel_nodes_kk(const struct fesom_mesh *mesh,
                                struct fesom_dyn        *dyn);

void fesom_pp_mixing_kk(const struct fesom_mesh *mesh,
                        const struct fesom_dyn  *dyn,
                        struct fesom_aux        *aux);

void fesom_mo_convect_kk(const struct fesom_mesh *mesh,
                         struct fesom_aux        *aux);

/*
 * FESOM_KK_VERIFY=pp in-binary per-kernel gates: run the host C twin alongside the
 * Kokkos production result and assert max|Δ|==0 on the Serial backend; non-intrusive
 * (each restores the Kokkos result). Call AFTER the *_kk kernel + its output
 * sync_host(). step_n is for the log. compute_vel_nodes/pp_mixing are EOS-style;
 * mo_convect modifies its inputs in place, so the driver passes the PRE-kernel
 * Kv/Av (Kv_in/Av_in) as the C-twin oracle's input. See docs/SYNC_MAP.md §9.5.
 */
void fesom_compute_vel_nodes_verify(const struct fesom_mesh *mesh,
                                    struct fesom_dyn        *dyn,
                                    int step_n);

void fesom_pp_mixing_verify(const struct fesom_mesh *mesh,
                            const struct fesom_dyn  *dyn,
                            struct fesom_aux        *aux,
                            int step_n);

void fesom_mo_convect_verify(const struct fesom_mesh *mesh,
                             struct fesom_aux        *aux,
                             int step_n,
                             const real_t *Kv_in,
                             const real_t *Av_in);

#endif /* FESOM_PP_H */
