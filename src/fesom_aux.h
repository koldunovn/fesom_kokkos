#ifndef FESOM_AUX_H
#define FESOM_AUX_H

#include "fesom_types.h"
#include "fesom_field.hpp"   // M1.3: DualView-backed storage for the persistent aux arrays

struct fesom_mesh;

/*
 * Phase 1 subset of the Fortran o_ARRAYS module globals (oce_modules.F90).
 *
 * Per-node 3D fields:
 *   density_m_rho0  [nod2D * nl]   in-situ density minus density_ref (kg/m³)
 *   hpressure       [nod2D * nl]   hydrostatic pressure (Pa) — written by
 *                                  pressure_bv (linfs) or pressure_force (zstar)
 *   bvfreq          [nod2D * nl]   N² (s⁻²)
 *   sw_alpha        [nod2D * nl]   thermal expansion coeff (1/K)
 *   sw_beta         [nod2D * nl]   haline contraction coeff (1/(g/kg))
 *   Kv              [nod2D * nl]   vertical tracer diffusivity (m²/s)
 *   dbsfc           [nod2D * nl]   buoyancy difference surface→level (m/s²),
 *                                  = g·(ρ_surf(z) − ρ(z))/ρ(z). Written by the
 *                                  EOS (fesom_eos.c, gated on KPP) and read by
 *                                  KPP bldepth. Fortran's dbsfc is the
 *                                  o_mixing_KPP_mod member filled from
 *                                  oce_ale_pressure_bv.F90. Zero/unused under PP.
 *
 * Per-element 3D fields:
 *   Av              [elem2D * nl]  vertical momentum viscosity (m²/s)
 *   pgf_x           [elem2D * nl]  pressure-gradient force, x-component (m/s²)
 *   pgf_y           [elem2D * nl]  pressure-gradient force, y-component (m/s²)
 *
 * KPP module state (per-tracer diffK, viscA, blmc, ghats, dkm1, hbl, kbl,
 * Bo, bfsfc, stable, caseA, dVsq, ustar, lookup tables) lives in struct
 * fesom_kpp (fesom_kpp.h), mirroring the o_mixing_KPP_mod module. Only dbsfc
 * lives here — it is written by the EOS, which carries aux but not kpp.
 *
 * GM/Redi additions (Phase G2a):
 *   MLD1_ind        [nod2D]        mixed-layer-depth index per node, computed
 *                                  by fesom_pressure_bv. Used by init_Redi_GM
 *                                  with K_GM_bvref=1 (default) to pick the
 *                                  reference buoyancy at the bottom of the ML.
 *                                  Mirror of o_ARRAYS::MLD1_ind.
 */
typedef struct fesom_aux {
    real_t *density_m_rho0;
    real_t *hpressure;
    real_t *bvfreq;
    real_t *sw_alpha;
    real_t *sw_beta;
    real_t *Kv;
    real_t *dbsfc;          /* [nod2D * nl] — KPP buoyancy re surface (EOS-written) */

    real_t *Av;
    real_t *pgf_x;
    real_t *pgf_y;

    int    *MLD1_ind;       /* [nod2D] — populated by fesom_pressure_bv */

    /* ===================== M1.3 · DualView data layer =========================
     * Each persistent array above is OWNED by the matching fesom::Field/IntField
     * below; the raw pointer is a NON-OWNING alias re-pointed at field.h() right
     * after field.alloc() in fesom_aux_alloc (the M1.2 mesh pattern, D12). These
     * are recomputed in place every step through the raw alias (no buffer swap),
     * so the alias stays valid for the whole run. LayoutRight host mirror == the C
     * flat layout → legacy aux->X[...] + FESOM_*3D macros unchanged → Serial
     * bit-identical. No device sync at M1 (nothing reads these on device yet;
     * step-driver sync is M1.5/M2, D15). */
    fesom::Field    density_m_rho0_fld, hpressure_fld, bvfreq_fld;
    fesom::Field    sw_alpha_fld, sw_beta_fld, Kv_fld, dbsfc_fld;
    fesom::Field    Av_fld, pgf_x_fld, pgf_y_fld;
    fesom::IntField MLD1_ind_fld;
} fesom_aux;

void fesom_aux_alloc(fesom_aux *a, const struct fesom_mesh *mesh);
void fesom_aux_free (fesom_aux *a);

#endif /* FESOM_AUX_H */
