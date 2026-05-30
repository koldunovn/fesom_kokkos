#ifndef FESOM_BULK_H
#define FESOM_BULK_H

#include "fesom_types.h"

struct fesom_mesh;
struct fesom_jra55;
struct fesom_dyn;
struct fesom_tracers;
struct fesom_forcing;
struct fesom_ice;
struct fesom_partit;

/*
 * NCAR bulk formulae (Large & Yeager 2004/2009) and open-water flux assembly.
 * Literal port of:
 *   ncar_ocean_fluxes_mode  (gen_bulk_formulae.F90:126-341)
 *   obudget                 (ice_thermo_oce.F90:777-868)
 *   wind-stress assembly    (gen_forcing_couple.F90:736-765)
 *
 * Without sea ice (Phase 3), surface fluxes are entirely open-water:
 *   stress_atmoce_x = ρ_air * cd * |U_air - U_oce| * (u_air - u_oce)
 *   qsr (downward shortwave to ocean) = (1 - albw) * fsh
 *   qns (non-solar surface heat) = hflwrow + hflwrdout + hfsenow + hflatow
 *   emp (E - P, m/s) = evap - prec_rain - prec_snow
 *
 * Outputs are written into the fesom_forcing struct fields:
 *   forcing->stress_node_surf[2*n+0,1]   (N/m²)
 *   forcing->heat_flux[n]                (W/m², +ve = ocean loses)
 *   forcing->water_flux[n]               (m/s, +ve = ocean loses)
 *
 * Per Fortran convention (line 348): the ice-side code stores qsr/qns with
 * positive=down and adds the minus sign in oce_fluxes(). Our heat_flux is
 * "positive up = ocean loses heat", matching the analytical-forcing path.
 *
 * Note: stress is computed at NODES here (Fortran does this too via atmdata),
 * then interpolated to elements with the existing
 * fesom_forcing_node_to_elem_stress() helper or inline.
 */
void fesom_bulk_compute(const struct fesom_jra55  *jra,
                        const struct fesom_mesh   *mesh,
                        const struct fesom_dyn    *dyn,
                        const struct fesom_tracers *tracers,
                        struct fesom_forcing       *forcing,
                        struct fesom_ice           *ice,
                        struct fesom_partit        *partit);

/* M5.16: device (Kokkos) twin of fesom_bulk_compute — the L&Y09 bulk formulae as a per-surface-node
 * map (bit-identical Serial; gate-only on CUDA). Drop-in replacement: leaves the same host-authoritative
 * forcing/ice state the C twin does (Phase A). Reads SST=T[surface] and uvnode on the DEVICE (so the
 * per-step T/uvnode DtoH in fesom_step.cpp can be removed — the M5.16 residency unlock). FORCED-ONLY. */
void fesom_bulk_compute_kk(const struct fesom_jra55  *jra,
                           const struct fesom_mesh   *mesh,
                           const struct fesom_dyn    *dyn,
                           const struct fesom_tracers *tracers,
                           struct fesom_forcing       *forcing,
                           struct fesom_ice           *ice,
                           struct fesom_partit        *partit);

/* FESOM_KK_VERIFY=bulk — Serial max|Δ|==0 vs the C twin. CORE2-only (pi never calls bulk). The caller
 * must sync_host T + uvnode first so the C twin's host SST/current reads are current (no-op on Serial). */
void fesom_bulk_compute_verify(const struct fesom_jra55  *jra,
                               const struct fesom_mesh   *mesh,
                               const struct fesom_dyn    *dyn,
                               const struct fesom_tracers *tracers,
                               struct fesom_forcing       *forcing,
                               struct fesom_ice           *ice,
                               struct fesom_partit        *partit,
                               int                         step_n);

/* Shortwave penetration (oce_shortwave_pene.F90): build forcing->sw_3d through
 * the column and remove the visible band from forcing->heat_flux. Call after
 * ice->ocean coupling, before the temperature tracer equation. */
void fesom_cal_shortwave_rad(const struct fesom_mesh  *mesh,
                             const struct fesom_jra55 *jra,
                             const struct fesom_ice   *ice,
                             struct fesom_forcing     *forcing);

/* M5.20: DEVICE twin of the sw_3d penetration profile (KPP/tracer-diff read sw_3d device-resident). */
void fesom_cal_shortwave_rad_kk(const struct fesom_mesh  *mesh,
                                const struct fesom_jra55 *jra,
                                const struct fesom_ice   *ice,
                                struct fesom_forcing     *forcing);

#endif /* FESOM_BULK_H */
