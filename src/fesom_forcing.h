#ifndef FESOM_FORCING_H
#define FESOM_FORCING_H

#include "fesom_types.h"
#include "fesom_field.hpp"   // M1.4: DualView-backed storage for the persistent forcing arrays

struct fesom_mesh;

/*
 * Surface forcing (FRESH_START.md §9). Phase 1 starts with zero forcing;
 * later steps wire JRA55-do bulk formulae and SSS restoring.
 *
 *   heat_flux         [nod2D]      net surface heat flux (W/m², +ve = ocean loses)
 *   water_flux        [nod2D]      E - P virtual freshwater (m/s, +ve = ocean loses)
 *   stress_node_surf  [nod2D * 2]  wind stress at nodes (N/m²)
 *   stress_surf       [elem2D * 2] wind stress at elements (N/m²)
 *
 * stress_node_surf and stress_surf are kept in sync by interpolation in the
 * Fortran code. We mirror both because impl_vert_visc reads stress at nodes
 * while compute_vel_rhs uses stress at elements.
 */
typedef struct fesom_forcing {
    real_t *heat_flux;
    real_t *water_flux;
    real_t *stress_node_surf;
    real_t *stress_surf;

    /* Phase 3 step 25 — SSS restoring + CORE2 runoff (o_ARRAYS in Fortran). */
    real_t *runoff;          /* [nod2D]  freshwater input from rivers, m/s   */
    real_t *Ssurf;           /* [nod2D]  monthly SSS climatology, PSU       */
    real_t *virtual_salt;    /* [nod2D]  rsss·water_flux (linfs only)        */
    real_t *relax_salt;      /* [nod2D]  surf_relax_S·(Ssurf - S_top)        */

    /* Per-node bulk transfer coefficients for the open-water (atm-ocean) path.
       Computed by fesom_bulk_compute via L&Y09; consumed by sea-ice
       thermodynamics (fesom_ice_thermodynamics). Mirrors Fortran
       Ch_atm_oce_arr / Ce_atm_oce_arr in g_forcing_arrays. */
    real_t *Ch_atm_oce;      /* [nod2D]  sensible-heat transfer coeff (open water) */
    real_t *Ce_atm_oce;      /* [nod2D]  evaporation transfer coeff   (open water) */

    /* Shortwave penetration (use_sw_pene). Mirrors Fortran g_forcing_arrays
       chl / sw_3d. chl: chlorophyll [mg/m³] (const or Sweeney monthly climatology).
       sw_3d: visible-band shortwave TEMPERATURE flux [K m/s] at level interfaces,
       built by fesom_cal_shortwave_rad (oce_shortwave_pene.F90), consumed by the
       temperature tracer equation (oce_ale_tracer.F90:990). */
    real_t *chl;             /* [nod2D]        chlorophyll concentration            */
    real_t *sw_3d;           /* [nod2D * nl]   shortwave T-flux at interfaces        */

    /* ===================== M1.4 · DualView data layer =========================
     * Each persistent array above is OWNED by the matching fesom::Field below; the
     * raw pointer is a NON-OWNING alias re-pointed at field.h() right after
     * field.alloc() in fesom_forcing_alloc (the same pattern proven on the 28 mesh
     * arrays in M1.2 + 37 dyn/aux/tracers arrays in M1.3, D12/D15). Sizes follow the
     * halo-sized local extent (node = myDim+eDim, elem = myDim+eDim+eXDim) exactly
     * as the original calloc (feedback_array_size_vs_reader_loop). These arrays are
     * written each step through the raw alias (bulk/runoff/analytical/ice fluxes) by
     * in-place value writes/memset — never re-pointed by a buffer swap — so the alias
     * stays valid for the whole run. LayoutRight + host mirror == the C flat layout,
     * so legacy forcing->X[...] keeps working unchanged → Serial stays bit-identical.
     * No device sync at M1: nothing reads these on device yet (the step-driver sync
     * discipline for this evolving state lands in M1.5/M2, D15). */
    fesom::Field heat_flux_fld, water_flux_fld;
    fesom::Field stress_node_surf_fld, stress_surf_fld;
    fesom::Field runoff_fld, Ssurf_fld, virtual_salt_fld, relax_salt_fld;
    fesom::Field Ch_atm_oce_fld, Ce_atm_oce_fld;
    fesom::Field chl_fld, sw_3d_fld;
} fesom_forcing;

void fesom_forcing_alloc(fesom_forcing *f, const struct fesom_mesh *mesh);
void fesom_forcing_free (fesom_forcing *f);

#endif /* FESOM_FORCING_H */
