/*
 * NCAR L&Y09 bulk formulae + open-water flux assembly + wind-stress assembly.
 * No sea ice — Phase 3 ocean-only.
 */
#include "fesom_bulk.h"
#include "fesom_speed.hpp"   // M7 Task 1.2: FESOM_SPEED_SWSKIP
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_forcing.h"
#include "fesom_halo.h"
#include "fesom_ice_types.h"
#include "fesom_jra55.h"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_tracers.h"

#include <Kokkos_Core.hpp>          // M5.16: device per-surface-node bulk map
#include "fesom_halo_device.hpp"    // M5.16: fesom_halo_field dispatch

#include <math.h>
#include <string.h>
#include <string>                   // M5.16: verify backend name
#include <vector>                   // M5.16: verify snapshot/restore
#include <algorithm>                // M5.16: std::copy in verify
#include <cstdio>

/* MOD_ICE.F90 ice%thermo defaults (lines 53-79). */
#define BULK_RHOAIR      1.3
#define BULK_INV_RHOAIR  (1.0 / 1.3)
#define BULK_CPAIR       1005.0
#define BULK_CLHW        2.501e6     /* J/kg, water → vapor */
#define BULK_TMELT       273.15
#define BULK_BOLTZMANN   5.67e-8
#define BULK_EMISS_WAT   0.97
#define BULK_ALBW        0.1         /* open-water albedo — CORE2 namelist.ice albw=0.1
                                        (open_water_albedo=0, constant). NOT the LY2004
                                        0.066: the reference run overrides it to 0.1, and
                                        the C used 0.066 -> ~3.8% excess SW absorption ->
                                        tropics-max SST warm bias. Keep == fesom_ice.c th->albw. */
#define BULK_INV_RHOWAT  (1.0 / 1000.0)
#define BULK_GRAV        9.80
#define BULK_VONKARM     0.40
#define BULK_Q1          640380.0
#define BULK_Q2          (-5107.4)
#define BULK_U10MIN      0.3         /* m/s wind floor */

#define BULK_N_ITTS      5
#define BULK_INC_RATIO   1.0e-4

/* core_coeff_2z helper not needed — we use the L&Y09 path only. */

/* ------------------------------------------------------------------------ *
 * NCAR L&Y04/09 ocean exchange coefficients (cd, ce, ch).                  *
 * Literal port of ncar_ocean_fluxes_mode (gen_bulk_formulae.F90:126-341).  *
 * Returns by reference; only called once per node per step.                *
 * ------------------------------------------------------------------------ */
static void ncar_ocean_fluxes_mode(real_t tair_C,    /* °C */
                                   real_t shum,      /* kg/kg */
                                   real_t u_wind,    /* m/s */
                                   real_t v_wind,    /* m/s */
                                   real_t T_oc_C,    /* °C */
                                   real_t u_w,       /* m/s */
                                   real_t v_w,       /* m/s */
                                   real_t z_wind,
                                   real_t z_tair,
                                   real_t z_shum,
                                   real_t *cd_out,
                                   real_t *ce_out,
                                   real_t *ch_out)
{
    /* Convert to Kelvin (Fortran lines 184-185). */
    real_t t  = tair_C  + BULK_TMELT;
    real_t ts = T_oc_C  + BULK_TMELT;
    real_t q  = shum;
    real_t qs = 0.98 * BULK_Q1 * BULK_INV_RHOAIR * exp(BULK_Q2 / ts); /* L-Y eqn 5 */
    real_t tv = t * (1.0 + 0.608 * q);

    real_t dux = u_wind - u_w;
    real_t dvy = v_wind - v_w;
    real_t u   = sqrt(dux*dux + dvy*dvy);
    if (u < BULK_U10MIN) u = BULK_U10MIN;

    real_t u10 = u, t10 = t, q10 = q;

    /* Initial cd_n10 — LY2009 eqn 11a/b. */
    real_t hl1 = (2.7/u10 + 0.142 + 0.0764*u10 - 3.14807e-10 * pow(u10, 6)) / 1.0e3;
    real_t cd_n10 = (0.5 - copysign(0.5, u10 - 33.0)) * hl1
                  + (0.5 + copysign(0.5, u10 - 33.0)) * 2.34e-3;
    real_t cd_n10_rt = sqrt(cd_n10);
    real_t ce_n10    = 34.6 * cd_n10_rt * 1.0e-3;
    real_t stab      = 0.5 + copysign(0.5, t - ts);
    real_t ch_n10    = (18.0*stab + 32.7*(1.0 - stab)) * cd_n10_rt * 1.0e-3;

    real_t cd = cd_n10, ch = ch_n10, ce = ce_n10;
    real_t cd_prev = cd;

    for (int it = 0; it < BULK_N_ITTS; ++it) {
        real_t cd_rt = sqrt(cd);
        real_t ustar = cd_rt * u;                     /* L-Y eqn 7a */
        real_t tstar = (ch / cd_rt) * (t10 - ts);     /* L-Y eqn 7b */
        real_t qstar = (ce / cd_rt) * (q10 - qs);     /* L-Y eqn 7c */
        real_t bstar = BULK_GRAV * (tstar/tv + qstar / (q10 + 1.0/0.608));

        /* zeta_u */
        real_t zeta_u = BULK_VONKARM * bstar * z_wind / (ustar * ustar);
        if (fabs(zeta_u) > 10.0) zeta_u = copysign(10.0, zeta_u);
        real_t x2 = sqrt(fabs(1.0 - 16.0 * zeta_u));
        if (x2 < 1.0) x2 = 1.0;
        real_t x  = sqrt(x2);
        real_t psi_m_u, psi_h_u;
        if (zeta_u > 0.0) {
            psi_m_u = -5.0 * zeta_u;
            psi_h_u = -5.0 * zeta_u;
        } else {
            psi_m_u = log((1.0 + 2.0*x + x2) * (1.0 + x2) / 8.0)
                    - 2.0 * (atan(x) - atan(1.0));
            psi_h_u = 2.0 * log((1.0 + x2) / 2.0);
        }

        /* zeta_t */
        real_t zeta_t = BULK_VONKARM * bstar * z_tair / (ustar * ustar);
        if (fabs(zeta_t) > 10.0) zeta_t = copysign(10.0, zeta_t);
        x2 = sqrt(fabs(1.0 - 16.0 * zeta_t));
        if (x2 < 1.0) x2 = 1.0;
        x = sqrt(x2);
        real_t psi_m_t, psi_h_t;
        (void)psi_m_t;
        if (zeta_t > 0.0) {
            psi_m_t = -5.0 * zeta_t;
            psi_h_t = -5.0 * zeta_t;
        } else {
            psi_m_t = log((1.0 + 2.0*x + x2) * (1.0 + x2) / 8.0)
                    - 2.0 * (atan(x) - atan(1.0));
            psi_h_t = 2.0 * log((1.0 + x2) / 2.0);
        }

        /* zeta_q */
        real_t zeta_q = BULK_VONKARM * bstar * z_shum / (ustar * ustar);
        if (fabs(zeta_q) > 10.0) zeta_q = copysign(10.0, zeta_q);
        x2 = sqrt(fabs(1.0 - 16.0 * zeta_q));
        if (x2 < 1.0) x2 = 1.0;
        x = sqrt(x2);
        real_t psi_m_q, psi_h_q;
        (void)psi_m_q;
        if (zeta_q > 0.0) {
            psi_m_q = -5.0 * zeta_q;
            psi_h_q = -5.0 * zeta_q;
        } else {
            psi_m_q = log((1.0 + 2.0*x + x2) * (1.0 + x2) / 8.0)
                    - 2.0 * (atan(x) - atan(1.0));
            psi_h_q = 2.0 * log((1.0 + x2) / 2.0);
        }

        /* Shift wind/temperature/humidity to 10 m and reference levels. */
        u10 = u / (1.0 + cd_n10_rt * (log(z_wind / 10.0) - psi_m_u) / BULK_VONKARM);
        if (u10 < BULK_U10MIN) u10 = BULK_U10MIN;
        t10 = t - tstar / BULK_VONKARM * (log(z_tair / z_wind) + psi_h_u - psi_h_t);
        q10 = q - qstar / BULK_VONKARM * (log(z_shum / z_wind) + psi_h_u - psi_h_q);
        tv  = t10 * (1.0 + 0.608 * q10);

        /* Update neutral 10 m coefs (LY2009 eqn 11a/b again). */
        hl1 = (2.7/u10 + 0.142 + 0.0764*u10 - 3.14807e-10 * pow(u10, 6)) / 1.0e3;
        cd_n10 = (0.5 - copysign(0.5, u10 - 33.0)) * hl1
               + (0.5 + copysign(0.5, u10 - 33.0)) * 2.34e-3;
        cd_n10_rt = sqrt(cd_n10);
        ce_n10    = 34.6 * cd_n10_rt * 1.0e-3;
        stab      = 0.5 + copysign(0.5, zeta_u);
        ch_n10    = (18.0*stab + 32.7*(1.0 - stab)) * cd_n10_rt * 1.0e-3;

        /* Shift cd/ch/ce back to measurement height. */
        real_t xx = (log(z_wind / 10.0) - psi_m_u) / BULK_VONKARM;
        cd = cd_n10 / pow(1.0 + cd_n10_rt * xx, 2);
        xx = (log(z_wind / 10.0) - psi_h_u) / BULK_VONKARM;
        ch = ch_n10 / (1.0 + ch_n10 * xx / cd_n10_rt) * sqrt(cd / cd_n10);
        ce = ce_n10 / (1.0 + ce_n10 * xx / cd_n10_rt) * sqrt(cd / cd_n10);

        real_t test = fabs(cd - cd_prev) / (cd + 1.0e-8);
        cd_prev = cd;
        if (test < BULK_INC_RATIO) break;
    }

    *cd_out = cd;
    *ce_out = ce;
    *ch_out = ch;
}

/* ------------------------------------------------------------------------ *
 * Open-water heat & freshwater fluxes.                                     *
 * Literal port of obudget (ice_thermo_oce.F90:777-868).                    *
 * Standard saturation_shum_formula = .true. branch.                        *
 * Returns: qsr (downward shortwave to ocean, +ve down)                     *
 *          qns (non-solar surface heat, +ve up = ocean loses)              *
 *          evap (kg/m²/s, +ve = ocean loses)                               *
 * ------------------------------------------------------------------------ */
static void obudget(real_t qa,        /* shum [kg/kg] */
                    real_t fsh,       /* shortwave down [W/m²] */
                    real_t flo,       /* longwave  down [W/m²] */
                    real_t t,         /* surface ocean T [°C] */
                    real_t ug,        /* wind speed scalar [m/s] */
                    real_t ta,        /* air temp [°C] */
                    real_t ch,
                    real_t ce,
                    real_t *qsr_out,  /* downward shortwave (W/m², +ve down) */
                    real_t *qns_out,  /* non-solar surface heat (W/m², +ve up) */
                    real_t *evap_out) /* m/s, +ve up */
{
    /* Standard saturation specific humidity formula (Fortran 840-844). */
    const real_t c1 = 3.8e-3;
    const real_t c4 = 17.27;
    const real_t c5 = 237.3;
    real_t b = c1 * exp(c4 * t / (t + c5));

    real_t hfswrow  = (1.0 - BULK_ALBW) * fsh;
    real_t hflwrow  = flo;
    real_t hflwrdout= -BULK_EMISS_WAT * BULK_BOLTZMANN * pow(t + BULK_TMELT, 4);
    real_t hfsenow  = BULK_RHOAIR * BULK_CPAIR * ch * ug * (ta - t);
    real_t evap     = BULK_RHOAIR * ce * ug * (qa - b);   /* kg/m²/s */
    real_t hflatow  = BULK_CLHW * evap;

    /* qsr: downward shortwave to ocean. qns: non-solar surface heat (positive
     * up = ocean loses). The Fortran ice path collects all heat into ehf
     * (positive down) then negates in oce_fluxes(). For our forcing struct
     * convention (positive up = ocean loses), we put qsr separately and
     * make qns = -(LW_in + LW_out + sensible + latent). */
    real_t qns = -(hflwrow + hflwrdout + hfsenow + hflatow);
    *qsr_out  = hfswrow;
    *qns_out  = qns;
    *evap_out = evap * BULK_INV_RHOWAT;   /* m/s, +ve up */
}

/* ------------------------------------------------------------------------ *
 * Drive bulk formulae over all nodes; assemble forcing arrays.             *
 * ------------------------------------------------------------------------ */
void fesom_bulk_compute(const struct fesom_jra55  *jra,
                        const struct fesom_mesh   *mesh,
                        const struct fesom_dyn    *dyn,
                        const struct fesom_tracers *tracers,
                        struct fesom_forcing       *forcing,
                        struct fesom_ice           *ice,
                        struct fesom_partit        *partit)
{
    /* Loop bound is myDim+eDim — matches Fortran ncar_ocean_fluxes_mode
     * (gen_bulk_formulae.F90:181) and the wind-stress loop
     * (gen_forcing_couple.F90:738), both `do i=1,myDim_nod2D+eDim_nod2D`.
     * Computing the exchange coefficients (Cd/Ch/Ce) and fluxes at eDim from
     * the (already halo-consistent) JRA + ocean-surface inputs keeps
     * Ch_atm_oce/Ce_atm_oce consistent at eDim. The sea-ice thermodynamics
     * reads Ch/Ce directly over myDim+eDim; computing them on myDim only left
     * eDim stale -> divergent a_ice across ranks -> divergent stress_surf on
     * replicated boundary elements -> non-conservative SSH RHS -> dt blow-up. */
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int E = mesh->myDim_elem2D;

    /* For each node compute cd/ce/ch then qsr/qns/emp. */
    for (int n = 0; n < N; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) {
            /* Cavity node — fluxes are 0. */
            forcing->stress_node_surf[2*n + 0] = 0.0;
            forcing->stress_node_surf[2*n + 1] = 0.0;
            forcing->heat_flux [n] = 0.0;
            forcing->water_flux[n] = 0.0;
            ice->stress_atmice_x[n] = 0.0;
            ice->stress_atmice_y[n] = 0.0;
            continue;
        }

        real_t T_oc = tracers->data[FESOM_TRACER_T].values[FESOM_NODE3D(n, 0, mesh->nl)];
        /* Surface ocean current at NODE — use uvnode's surface layer. */
        real_t u_w = dyn->uvnode[(size_t)(2 * (FESOM_NODE3D(n, 0, mesh->nl))) + 0];
        real_t v_w = dyn->uvnode[(size_t)(2 * (FESOM_NODE3D(n, 0, mesh->nl))) + 1];

        real_t ua = jra->u_wind[n];
        real_t va = jra->v_wind[n];
        real_t ta = jra->Tair[n];        /* °C */
        real_t qa = jra->shum[n];        /* kg/kg */
        real_t fsh= jra->shortwave[n];
        real_t flo= jra->longwave[n];
        real_t pr = jra->prec_rain[n];   /* m/s */
        real_t ps = jra->prec_snow[n];   /* m/s */

        real_t cd, ce, ch;
        ncar_ocean_fluxes_mode(ta, qa, ua, va, T_oc, u_w, v_w,
                               jra->z_wind, jra->z_tair, jra->z_shum,
                               &cd, &ce, &ch);
        /* save per-node ch/ce for the sea-ice thermodynamics path
         * (Fortran: Ch_atm_oce_arr / Ce_atm_oce_arr in g_forcing_arrays) */
        forcing->Ch_atm_oce[n] = ch;
        forcing->Ce_atm_oce[n] = ce;

        /* Wind speed for obudget — Fortran ice_thermo_oce uses sqrt(u_wind² + v_wind²). */
        real_t ug = sqrt(ua*ua + va*va);
        real_t qsr, qns, evap;
        obudget(qa, fsh, flo, T_oc, ug, ta, ch, ce, &qsr, &qns, &evap);

        /* Heat flux (positive up = ocean loses). qsr is downward → enters as
         * -qsr in ocean's surface heat budget. We sum into a single
         * heat_flux per analytical-forcing convention; if/when shortwave
         * penetration is added we'll separate. */
        forcing->heat_flux[n]  = qns - qsr;
        /* Freshwater (E - P). evap is m/s positive up (ocean loses);
         * prec is m/s positive down (ocean gains). */
        forcing->water_flux[n] = evap - pr - ps;

        /* Wind stress at node — Fortran gen_forcing_couple.F90 lines 749-757
         * (Swind=0 default in our namelist). */
        real_t dux = ua - u_w;
        real_t dvy = va - v_w;
        real_t mag = sqrt(dux*dux + dvy*dvy) * BULK_RHOAIR;
        forcing->stress_node_surf[2*n + 0] = cd * mag * dux;
        forcing->stress_node_surf[2*n + 1] = cd * mag * dvy;

        /* Atmosphere-ICE momentum stress — Fortran gen_forcing_couple.F90:
         * 759-763. Same bulk form but wind RELATIVE TO ICE and the constant
         * atm-ice drag Cd_atm_ice (AOMIP_drag_coeff=.false. on CORE2). This
         * is the EVP's direct wind forcing (inv_mass·stress_atmice in
         * fesom_ice_evp.c); without it the sea ice never feels the wind. */
        real_t dux_i = ua - ice->uice[n];
        real_t dvy_i = va - ice->vice[n];
        real_t mag_i = sqrt(dux_i*dux_i + dvy_i*dvy_i) * BULK_RHOAIR;
        ice->stress_atmice_x[n] = FESOM_CD_ATM_ICE * mag_i * dux_i;
        ice->stress_atmice_y[n] = FESOM_CD_ATM_ICE * mag_i * dvy_i;
    }

    /* Halo-exchange node-side fields written above so the element
     * interpolation below sees correct values at halo vertices. Without this,
     * partition-boundary element stress is interpolated against stale or
     * uninitialised halo entries → spurious wind torque on boundary triangles
     * → blow-up during momentum step. heat_flux/water_flux are read by
     * tracer_diff for myDim only, but exchange them too for any future kernel
     * that touches them at halo. */
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(forcing->stress_node_surf, FESOM_HALO_NOD2D, 1, 2, partit);
        fesom_halo_exchange(forcing->heat_flux,        FESOM_HALO_NOD2D, 1, 1, partit);
        fesom_halo_exchange(forcing->water_flux,       FESOM_HALO_NOD2D, 1, 1, partit);
    }

    /* Interpolate node stress → element stress (mean of 3 vertices). The
     * existing analytical-forcing path writes stress_surf directly at elements;
     * the bulk formulae assemble at nodes (matches Fortran), then average. */
    memset(forcing->stress_surf, 0, (size_t)E * 2 * sizeof(real_t));
    for (int e = 0; e < E; ++e) {
        real_t sx = 0.0, sy = 0.0;
        for (int k = 0; k < 3; ++k) {
            int v = mesh->elem_nodes[3*e + k];
            sx += forcing->stress_node_surf[2*v + 0];
            sy += forcing->stress_node_surf[2*v + 1];
        }
        forcing->stress_surf[2*e + 0] = sx / 3.0;
        forcing->stress_surf[2*e + 1] = sy / 3.0;
    }
}

/* ======================================================================== *
 *  M5.16 — DEVICE (Kokkos) twin of fesom_bulk_compute. The L&Y09 air-sea     *
 *  bulk formulae are a per-surface-node MAP: each node's exchange coeffs +    *
 *  fluxes depend only on its own JRA/SST/wind/ice state — NO halo, NO scatter,*
 *  NO reduction → bit-identical Serial AND OpenMP, gate-only on CUDA (the     *
 *  EOS/KPP/ice-thermo device-map class, L39/L45). This replaces the single-   *
 *  threaded host loop that the GPU build ran on the Kokkos-Serial host (the    *
 *  blmc/L49 trap: ~16% of the NG5 step, invisible to nsys). ncar_ocean_-      *
 *  fluxes_mode / obudget become KOKKOS_INLINE_FUNCTION device twins; the       *
 *  scalar BULK_* / FESOM_CD_ATM_ICE are #defines (device-safe), the runtime    *
 *  z_wind/z_tair/z_shum are captured by value. The C originals stay the verify *
 *  oracle (D19). ⚠️ FORCED-ONLY → meaningful only on CORE2 (pi never calls it). *
 * ======================================================================== */

/* Device twin of ncar_ocean_fluxes_mode (verbatim port; std math → Kokkos::). */
KOKKOS_INLINE_FUNCTION
void ncar_ocean_fluxes_mode_kk(real_t tair_C, real_t shum, real_t u_wind, real_t v_wind,
                               real_t T_oc_C, real_t u_w, real_t v_w,
                               real_t z_wind, real_t z_tair, real_t z_shum,
                               real_t *cd_out, real_t *ce_out, real_t *ch_out)
{
    real_t t  = tair_C + BULK_TMELT;
    real_t ts = T_oc_C + BULK_TMELT;
    real_t q  = shum;
    real_t qs = 0.98 * BULK_Q1 * BULK_INV_RHOAIR * Kokkos::exp(BULK_Q2 / ts);
    real_t tv = t * (1.0 + 0.608 * q);

    real_t dux = u_wind - u_w;
    real_t dvy = v_wind - v_w;
    real_t u   = Kokkos::sqrt(dux*dux + dvy*dvy);
    if (u < BULK_U10MIN) u = BULK_U10MIN;

    real_t u10 = u, t10 = t, q10 = q;

    real_t hl1 = (2.7/u10 + 0.142 + 0.0764*u10 - 3.14807e-10 * Kokkos::pow(u10, 6.0)) / 1.0e3;
    real_t cd_n10 = (0.5 - Kokkos::copysign(0.5, u10 - 33.0)) * hl1
                  + (0.5 + Kokkos::copysign(0.5, u10 - 33.0)) * 2.34e-3;
    real_t cd_n10_rt = Kokkos::sqrt(cd_n10);
    real_t ce_n10    = 34.6 * cd_n10_rt * 1.0e-3;
    real_t stab      = 0.5 + Kokkos::copysign(0.5, t - ts);
    real_t ch_n10    = (18.0*stab + 32.7*(1.0 - stab)) * cd_n10_rt * 1.0e-3;

    real_t cd = cd_n10, ch = ch_n10, ce = ce_n10;
    real_t cd_prev = cd;

    for (int it = 0; it < BULK_N_ITTS; ++it) {
        real_t cd_rt = Kokkos::sqrt(cd);
        real_t ustar = cd_rt * u;
        real_t tstar = (ch / cd_rt) * (t10 - ts);
        real_t qstar = (ce / cd_rt) * (q10 - qs);
        real_t bstar = BULK_GRAV * (tstar/tv + qstar / (q10 + 1.0/0.608));

        real_t zeta_u = BULK_VONKARM * bstar * z_wind / (ustar * ustar);
        if (Kokkos::fabs(zeta_u) > 10.0) zeta_u = Kokkos::copysign(10.0, zeta_u);
        real_t x2 = Kokkos::sqrt(Kokkos::fabs(1.0 - 16.0 * zeta_u));
        if (x2 < 1.0) x2 = 1.0;
        real_t x  = Kokkos::sqrt(x2);
        real_t psi_m_u, psi_h_u;
        if (zeta_u > 0.0) {
            psi_m_u = -5.0 * zeta_u;
            psi_h_u = -5.0 * zeta_u;
        } else {
            psi_m_u = Kokkos::log((1.0 + 2.0*x + x2) * (1.0 + x2) / 8.0)
                    - 2.0 * (Kokkos::atan(x) - Kokkos::atan(1.0));
            psi_h_u = 2.0 * Kokkos::log((1.0 + x2) / 2.0);
        }

        real_t zeta_t = BULK_VONKARM * bstar * z_tair / (ustar * ustar);
        if (Kokkos::fabs(zeta_t) > 10.0) zeta_t = Kokkos::copysign(10.0, zeta_t);
        x2 = Kokkos::sqrt(Kokkos::fabs(1.0 - 16.0 * zeta_t));
        if (x2 < 1.0) x2 = 1.0;
        x = Kokkos::sqrt(x2);
        real_t psi_m_t, psi_h_t;
        (void)psi_m_t;
        if (zeta_t > 0.0) {
            psi_m_t = -5.0 * zeta_t;
            psi_h_t = -5.0 * zeta_t;
        } else {
            psi_m_t = Kokkos::log((1.0 + 2.0*x + x2) * (1.0 + x2) / 8.0)
                    - 2.0 * (Kokkos::atan(x) - Kokkos::atan(1.0));
            psi_h_t = 2.0 * Kokkos::log((1.0 + x2) / 2.0);
        }

        real_t zeta_q = BULK_VONKARM * bstar * z_shum / (ustar * ustar);
        if (Kokkos::fabs(zeta_q) > 10.0) zeta_q = Kokkos::copysign(10.0, zeta_q);
        x2 = Kokkos::sqrt(Kokkos::fabs(1.0 - 16.0 * zeta_q));
        if (x2 < 1.0) x2 = 1.0;
        x = Kokkos::sqrt(x2);
        real_t psi_m_q, psi_h_q;
        (void)psi_m_q;
        if (zeta_q > 0.0) {
            psi_m_q = -5.0 * zeta_q;
            psi_h_q = -5.0 * zeta_q;
        } else {
            psi_m_q = Kokkos::log((1.0 + 2.0*x + x2) * (1.0 + x2) / 8.0)
                    - 2.0 * (Kokkos::atan(x) - Kokkos::atan(1.0));
            psi_h_q = 2.0 * Kokkos::log((1.0 + x2) / 2.0);
        }

        u10 = u / (1.0 + cd_n10_rt * (Kokkos::log(z_wind / 10.0) - psi_m_u) / BULK_VONKARM);
        if (u10 < BULK_U10MIN) u10 = BULK_U10MIN;
        t10 = t - tstar / BULK_VONKARM * (Kokkos::log(z_tair / z_wind) + psi_h_u - psi_h_t);
        q10 = q - qstar / BULK_VONKARM * (Kokkos::log(z_shum / z_wind) + psi_h_u - psi_h_q);
        tv  = t10 * (1.0 + 0.608 * q10);

        hl1 = (2.7/u10 + 0.142 + 0.0764*u10 - 3.14807e-10 * Kokkos::pow(u10, 6.0)) / 1.0e3;
        cd_n10 = (0.5 - Kokkos::copysign(0.5, u10 - 33.0)) * hl1
               + (0.5 + Kokkos::copysign(0.5, u10 - 33.0)) * 2.34e-3;
        cd_n10_rt = Kokkos::sqrt(cd_n10);
        ce_n10    = 34.6 * cd_n10_rt * 1.0e-3;
        stab      = 0.5 + Kokkos::copysign(0.5, zeta_u);
        ch_n10    = (18.0*stab + 32.7*(1.0 - stab)) * cd_n10_rt * 1.0e-3;

        real_t xx = (Kokkos::log(z_wind / 10.0) - psi_m_u) / BULK_VONKARM;
        cd = cd_n10 / Kokkos::pow(1.0 + cd_n10_rt * xx, 2.0);
        xx = (Kokkos::log(z_wind / 10.0) - psi_h_u) / BULK_VONKARM;
        ch = ch_n10 / (1.0 + ch_n10 * xx / cd_n10_rt) * Kokkos::sqrt(cd / cd_n10);
        ce = ce_n10 / (1.0 + ce_n10 * xx / cd_n10_rt) * Kokkos::sqrt(cd / cd_n10);

        real_t test = Kokkos::fabs(cd - cd_prev) / (cd + 1.0e-8);
        cd_prev = cd;
        if (test < BULK_INC_RATIO) break;
    }

    *cd_out = cd;
    *ce_out = ce;
    *ch_out = ch;
}

/* Device twin of the open-water obudget (verbatim port; std math → Kokkos::). */
KOKKOS_INLINE_FUNCTION
void obudget_oce_kk(real_t qa, real_t fsh, real_t flo, real_t t, real_t ug, real_t ta,
                    real_t ch, real_t ce, real_t *qsr_out, real_t *qns_out, real_t *evap_out)
{
    const real_t c1 = 3.8e-3;
    const real_t c4 = 17.27;
    const real_t c5 = 237.3;
    real_t b = c1 * Kokkos::exp(c4 * t / (t + c5));

    real_t hfswrow  = (1.0 - BULK_ALBW) * fsh;
    real_t hflwrow  = flo;
    real_t hflwrdout= -BULK_EMISS_WAT * BULK_BOLTZMANN * Kokkos::pow(t + BULK_TMELT, 4.0);
    real_t hfsenow  = BULK_RHOAIR * BULK_CPAIR * ch * ug * (ta - t);
    real_t evap     = BULK_RHOAIR * ce * ug * (qa - b);
    real_t hflatow  = BULK_CLHW * evap;

    real_t qns = -(hflwrow + hflwrdout + hfsenow + hflatow);
    *qsr_out  = hfswrow;
    *qns_out  = qns;
    *evap_out = evap * BULK_INV_RHOWAT;
}

void fesom_bulk_compute_kk(const struct fesom_jra55  *jra,
                           const struct fesom_mesh   *mesh,
                           const struct fesom_dyn    *dyn,
                           const struct fesom_tracers *tracers,
                           struct fesom_forcing       *forcing,
                           struct fesom_ice           *ice,
                           struct fesom_partit        *partit)
{
    const int    N      = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    E      = mesh->myDim_elem2D;
    const int    nl     = mesh->nl;
    const real_t z_wind = jra->z_wind;
    const real_t z_tair = jra->z_tair;
    const real_t z_shum = jra->z_shum;

    /* IN rail (L28/L14): the 8 JRA physics arrays are freshly time-interpolated on the host each
     * step via the raw alias (fesom_jra55_step_cal), invisible to the DualView → push host→device.
     * (jra const → const_cast for this coherence-only sync, like the ice-thermo IN rail.) */
    struct fesom_jra55 *j = const_cast<struct fesom_jra55 *>(jra);
    j->u_wind_fld.modify_host();    j->u_wind_fld.sync_device();
    j->v_wind_fld.modify_host();    j->v_wind_fld.sync_device();
    j->shum_fld.modify_host();      j->shum_fld.sync_device();
    j->shortwave_fld.modify_host(); j->shortwave_fld.sync_device();
    j->longwave_fld.modify_host();  j->longwave_fld.sync_device();
    j->Tair_fld.modify_host();      j->Tair_fld.sync_device();
    j->prec_rain_fld.modify_host(); j->prec_rain_fld.sync_device();
    j->prec_snow_fld.modify_host(); j->prec_snow_fld.sync_device();

    auto uw    = jra->u_wind_fld.d();    auto vw    = jra->v_wind_fld.d();
    auto qair  = jra->shum_fld.d();      auto swr   = jra->shortwave_fld.d();
    auto lwr   = jra->longwave_fld.d();  auto tair  = jra->Tair_fld.d();
    auto prain = jra->prec_rain_fld.d(); auto psnow = jra->prec_snow_fld.d();
    auto Tval  = tracers->data[FESOM_TRACER_T].values_fld.d();   /* SST = T surface (device-resident, M5.16 reads it on device) */
    auto uvn   = dyn->uvnode_fld.d();                            /* surface ocean current (device-resident, M5.16) */
    auto uice  = ice->uice_fld.d();      auto vice  = ice->vice_fld.d();
    auto sns   = forcing->stress_node_surf_fld.d();
    auto hf    = forcing->heat_flux_fld.d();
    auto wf    = forcing->water_flux_fld.d();
    auto chao  = forcing->Ch_atm_oce_fld.d();
    auto ceao  = forcing->Ce_atm_oce_fld.d();
    auto sax   = ice->stress_atmice_x_fld.d();
    auto say   = ice->stress_atmice_y_fld.d();
    auto ulev  = mesh->ulevels_nod2D_fld.d();

    /* Per-surface-node MAP over [0,N) (halo included — the C bound; eDim computed from halo-
     * consistent inputs keeps Ch/Ce consistent there for the ice thermo). Race-free, no scatter. */
    Kokkos::parallel_for("fesom_bulk_compute", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            if (ulev(n) > 1) {                          /* cavity — fluxes 0 (Ch/Ce left untouched, as C) */
                sns(2*n + 0) = 0.0; sns(2*n + 1) = 0.0;
                hf(n) = 0.0;        wf(n) = 0.0;
                sax(n) = 0.0;       say(n) = 0.0;
                return;
            }
            const size_t s = FESOM_NODE3D(n, 0, nl);    /* surface row */
            const real_t T_oc = Tval(s);
            const real_t u_w  = uvn(2*s + 0);
            const real_t v_w  = uvn(2*s + 1);

            const real_t ua = uw(n), va = vw(n), ta = tair(n), qa = qair(n);
            const real_t fsh = swr(n), flo = lwr(n), pr = prain(n), ps = psnow(n);

            real_t cd, ce, ch;
            ncar_ocean_fluxes_mode_kk(ta, qa, ua, va, T_oc, u_w, v_w,
                                      z_wind, z_tair, z_shum, &cd, &ce, &ch);
            chao(n) = ch; ceao(n) = ce;

            const real_t ug = Kokkos::sqrt(ua*ua + va*va);
            real_t qsr, qns, evap;
            obudget_oce_kk(qa, fsh, flo, T_oc, ug, ta, ch, ce, &qsr, &qns, &evap);

            hf(n) = qns - qsr;
            wf(n) = evap - pr - ps;

            const real_t dux = ua - u_w, dvy = va - v_w;
            const real_t mag = Kokkos::sqrt(dux*dux + dvy*dvy) * BULK_RHOAIR;
            sns(2*n + 0) = cd * mag * dux;
            sns(2*n + 1) = cd * mag * dvy;

            const real_t dux_i = ua - uice(n), dvy_i = va - vice(n);
            const real_t mag_i = Kokkos::sqrt(dux_i*dux_i + dvy_i*dvy_i) * BULK_RHOAIR;
            sax(n) = FESOM_CD_ATM_ICE * mag_i * dux_i;
            say(n) = FESOM_CD_ATM_ICE * mag_i * dvy_i;
        });

    /* Mark the device writes; halo the 3 the C halos. (fesom_halo_field marks those 3 device-dirty
     * itself; the 4 non-halo'd need an explicit modify_device before the sync_host below.) */
    forcing->Ch_atm_oce_fld.modify_device();   forcing->Ce_atm_oce_fld.modify_device();
    ice->stress_atmice_x_fld.modify_device();  ice->stress_atmice_y_fld.modify_device();

    fesom_halo_field(forcing->stress_node_surf_fld, FESOM_HALO_NOD2D, 1, 2, partit);
    /* M5.23 (L3): heat_flux+water_flux are same-kind (NOD2D nc=1) and adjacent (the bulk kernel
     * wrote both; nothing between) → one FUSED message/neighbour. (stress_node_surf is nc=2 → not
     * batchable with these.) Bit-identical (co-pack only). */
    fesom_halo_field2(forcing->heat_flux_fld, forcing->water_flux_fld, FESOM_HALO_NOD2D, 1, 1, partit);

    /* DROP-IN (M5.16 Phase A): sync the FULL output set to host so the downstream — oce_fluxes_mom
     * [host] reads stress_node_surf, the ice-step IN rails (Ch/Ce → thermo, stress_atmice → EVP),
     * the host element-interp + the ocean-step re-pushes — sees byte-for-byte the host-authoritative
     * state the C fesom_bulk_compute left. (heat_flux/water_flux are overwritten by oce_fluxes; sync'd
     * for safety + the verify.) The win is the device COMPUTE + the removed T/uvnode DtoH, not these
     * small nod2D round-trips; making forcing fully device-resident (drop these + the re-pushes) is a
     * measured follow-on. */
    forcing->stress_node_surf_fld.sync_host();
    forcing->heat_flux_fld.sync_host();
    forcing->water_flux_fld.sync_host();
    forcing->Ch_atm_oce_fld.sync_host();
    forcing->Ce_atm_oce_fld.sync_host();
    ice->stress_atmice_x_fld.sync_host();
    ice->stress_atmice_y_fld.sync_host();

    /* Node→element stress interpolation on the host (reads host stress_node_surf, now current) —
     * verbatim from the C path; oce_fluxes_mom overwrites stress_surf, so this serves the verify +
     * the analytical-path parity. */
    memset(forcing->stress_surf, 0, (size_t)E * 2 * sizeof(real_t));
    for (int e = 0; e < E; ++e) {
        real_t sx = 0.0, sy = 0.0;
        for (int k = 0; k < 3; ++k) {
            int v = mesh->elem_nodes[3*e + k];
            sx += forcing->stress_node_surf[2*v + 0];
            sy += forcing->stress_node_surf[2*v + 1];
        }
        forcing->stress_surf[2*e + 0] = sx / 3.0;
        forcing->stress_surf[2*e + 1] = sy / 3.0;
    }
}

/* FESOM_KK_VERIFY=bulk — bulk is a full per-node OVERWRITE from intact inputs (jra/SST/uvnode/ice,
 * none of which it mutates), so this is the EOS/iceflux-style verify (no capture-before): snapshot
 * the KK outputs (8 arrays, host-current after the kk sync_host), run the C twin (recomputes from the
 * intact inputs; its halo exchanges are collective → run on ALL ranks), diff, restore KK. Race-free
 * map → max|Δ|==0 on Serial AND OpenMP. ⚠️ FORCED-ONLY: pi uses analytical stress and NEVER calls
 * bulk (jra55_year<=0) → this verify is meaningful only on CORE2 with JRA55 active (L42). The caller
 * sync_host's T + uvnode first so the C twin's host SST/current reads are current (no-op on Serial). */
void fesom_bulk_compute_verify(const struct fesom_jra55 *jra, const struct fesom_mesh *mesh,
                               const struct fesom_dyn *dyn, const struct fesom_tracers *tracers,
                               struct fesom_forcing *forcing, struct fesom_ice *ice,
                               struct fesom_partit *partit, int step_n)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int E = mesh->myDim_elem2D;
    std::vector<real_t> ksns(forcing->stress_node_surf, forcing->stress_node_surf + 2*N);
    std::vector<real_t> kss (forcing->stress_surf,      forcing->stress_surf      + 2*E);
    std::vector<real_t> khf (forcing->heat_flux,        forcing->heat_flux        + N);
    std::vector<real_t> kwf (forcing->water_flux,       forcing->water_flux       + N);
    std::vector<real_t> kch (forcing->Ch_atm_oce,       forcing->Ch_atm_oce       + N);
    std::vector<real_t> kce (forcing->Ce_atm_oce,       forcing->Ce_atm_oce       + N);
    std::vector<real_t> ksax(ice->stress_atmice_x,      ice->stress_atmice_x      + N);
    std::vector<real_t> ksay(ice->stress_atmice_y,      ice->stress_atmice_y      + N);

    fesom_bulk_compute(jra, mesh, dyn, tracers, forcing, ice, partit);   /* C twin (overwrites host) */

    auto mx = [&](const std::vector<real_t> &kk, const real_t *c, int len) {
        double d = 0.0; for (int i = 0; i < len; ++i) { double x = std::fabs((double)kk[i]-(double)c[i]); if (x>d) d=x; } return d; };
    double dsns = mx(ksns, forcing->stress_node_surf, 2*N);
    double dss  = mx(kss,  forcing->stress_surf,      2*E);
    double dhf  = mx(khf,  forcing->heat_flux,        N);
    double dwf  = mx(kwf,  forcing->water_flux,       N);
    double dch  = mx(kch,  forcing->Ch_atm_oce,       N);
    double dce  = mx(kce,  forcing->Ce_atm_oce,       N);
    double dsax = mx(ksax, ice->stress_atmice_x,      N);
    double dsay = mx(ksay, ice->stress_atmice_y,      N);

    std::copy(ksns.begin(),ksns.end(),forcing->stress_node_surf);
    std::copy(kss.begin(), kss.end(), forcing->stress_surf);
    std::copy(khf.begin(), khf.end(), forcing->heat_flux);
    std::copy(kwf.begin(), kwf.end(), forcing->water_flux);
    std::copy(kch.begin(), kch.end(), forcing->Ch_atm_oce);
    std::copy(kce.begin(), kce.end(), forcing->Ce_atm_oce);
    std::copy(ksax.begin(),ksax.end(),ice->stress_atmice_x);
    std::copy(ksay.begin(),ksay.end(),ice->stress_atmice_y);

    double dmax = dsns; for (double x : {dss,dhf,dwf,dch,dce,dsax,dsay}) if (x > dmax) dmax = x;
    const std::string be = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=bulk] step %d backend=%s  max|Δ|: sns=%.3e ss=%.3e hf=%.3e wf=%.3e "
                "Ch=%.3e Ce=%.3e satmx=%.3e satmy=%.3e\n",
                step_n, be.c_str(), dsns, dss, dhf, dwf, dch, dce, dsax, dsay);
    std::fflush(stdout);
    if (be == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=bulk] FAIL step %d: bulk Serial must be bit-identical "
                             "(max|Δ|=%.3e)\n", step_n, dmax); std::abort();
    }
}

/*===========================================================================
 * cal_shortwave_rad — literal port of oce_shortwave_pene.F90.
 *
 * Shortwave penetration into the ocean (Morel & Antoine 1994, Sweeney 2005).
 * Builds the visible-band shortwave TEMPERATURE flux sw_3d through the column
 * (two-band exponential with chl-dependent coefficients) and removes the
 * visible fraction (0.54) from heat_flux. No penetration under cavity or sea
 * ice. MUST be called AFTER ice→ocean coupling and BEFORE the temperature
 * tracer equation consumes heat_flux + sw_3d (oce_ale_tracer.F90:990).
 *===========================================================================*/
void fesom_cal_shortwave_rad(const struct fesom_mesh  *mesh,
                             const struct fesom_jra55 *jra,
                             const struct fesom_ice   *ice,
                             struct fesom_forcing     *forcing)
{
    if (!FESOM_PHASE1_USE_SW_PENE) return;

    const int    N    = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    nl   = mesh->nl;
    const real_t albw = (real_t)BULK_ALBW;     /* open-water albedo (CORE2 namelist.ice albw=0.1) */
    const real_t vcpw = (real_t)FESOM_VCPW;
    const real_t *a_ice = ice->data[FESOM_ICE_AICE].values;
    real_t *sw_3d = forcing->sw_3d;
    real_t *chl   = forcing->chl;

    /* ==================================================================== *
     * M7 Task 1.2 — FESOM_SPEED_SWSKIP: the sw_3d half of this function is DEAD WORK.
     *
     * M5.20 moved the sw_3d penetration profile to the device (fesom_cal_shortwave_rad_kk,
     * below) and removed its HtoD push — but left THIS host computation running. Both are
     * called back-to-back every step (fesom_main.cpp:1214-1215), and the device twin STARTS
     * BY ZEROING THE WHOLE ARRAY (the fesom_sw3d_zero kernel) and then rewrites every entry.
     * So everything this function writes into sw_3d is overwritten microseconds later, on
     * BOTH backends:
     *   CUDA   — sw_3d_fld is device-authoritative; nothing reads the host copy.
     *   Serial — sw_3d_fld.d() IS this same host array, and the _kk twin overwrites it.
     * The only unique output of this function is the nod2D `heat_flux += swsurf` side effect
     * (the _kk kernel deliberately does NOT do it — see its :794 comment).
     *
     * The dead half is not cheap. It is a nod3D memset (N*nl doubles — 1.04 GB/rank/step at
     * NG5@4N) plus a per-column exp() walk, single-threaded, on the critical path of every
     * step. It is the bulk of the ~25% host segment the M7 stall budget found — NOT
     * ice_oce_fluxes_mom, which the Task-1.0 A/B measured at only 0.72%.
     *
     * Skipping it is BIT-IDENTICAL by construction (the device twin recomputes sw_3d in full
     * from the same inputs with the same arithmetic), and provable with the FORCE_SERIAL byte
     * proof.
     *
     * ⚠️ The ONE host reader of sw_3d is kpp_bldepth (fesom_kpp.cpp:474), which is static and
     * runs ONLY under -DFESOM_KK_VERIFY (fesom_step.cpp:215). Do not combine this knob with
     * FESOM_KK_VERIFY: the verify twin would read a stale sw_3d.
     * ==================================================================== */
    static int s_swskip = -1;
    const bool skip_sw3d = fesom_speed_on("SWSKIP", &s_swskip);

    /* zero sw_3d over all local nodes/levels (Fortran 39-43) */
    if (!skip_sw3d)
        memset(sw_3d, 0, (size_t)N * (size_t)nl * sizeof(real_t));

    for (int n2 = 0; n2 < N; ++n2) {
        if (mesh->ulevels_nod2D[n2] > 1) continue;   /* cavity: no penetration (F:51) */
        if (a_ice[n2] > 0.0)             continue;   /* under ice: none      (F:52) */

        /* visible shortwave into ocean [W/m²]; '+'-up: add back to heat_flux (F:56-60) */
        real_t swsurf = (1.0 - albw) * jra->shortwave[n2];   /* = qsr */
        swsurf *= 0.54;                                      /* visible part (300-750nm) */
        forcing->heat_flux[n2] += swsurf;   /* THE unique output — always computed */

        if (skip_sw3d) continue;            /* the device twin rebuilds sw_3d in full */

        /* Sweeney 2005 (Appendix A) two-band coefficients from chl (F:66-79) */
        real_t cc = chl[n2];
        if (cc < 0.02) cc = 0.02;                            /* limit from below */
        real_t c  = log10(cc);
        real_t c2 = c*c, c3 = c2*c, c4 = c3*c, c5 = c4*c;
        real_t v1  = 0.008*c + 0.132*c2 + 0.038*c3 - 0.017*c4 - 0.007*c5;
        real_t v2  = 0.679 - v1;
        v1         = 0.321 + v1;
        real_t sc1 = 1.54  - 0.197*c + 0.166*c2 - 0.252*c3 - 0.055*c4 + 0.042*c5;
        real_t sc2 = 7.925 - 6.644*c + 3.662*c2 - 1.815*c3 - 0.218*c4 + 0.502*c5;

        swsurf /= vcpw;                                      /* W/m² → K m/s (F:81) */

        const int nzmin = mesh->ulevels_nod2D[n2] - 1;       /* 0-based top interface */
        const int nzmax = mesh->nlevels_nod2D[n2] - 1;       /* 0-based bottom interface */
        sw_3d[FESOM_NODE3D(n2, nzmin, nl)] = swsurf;         /* Fortran nzmin (F:85) */
        for (int k = nzmin + 1; k <= nzmax; ++k) {           /* F:86-93 */
            real_t z   = mesh->zbar_3d_n[FESOM_NODE3D(n2, k, nl)];
            real_t aux = v1 * exp(z / sc1) + v2 * exp(z / sc2);
            sw_3d[FESOM_NODE3D(n2, k, nl)] = swsurf * aux;
            if (aux < 1.0e-5 || k == nzmax) {
                sw_3d[FESOM_NODE3D(n2, k, nl)] = 0.0;
                break;
            }
        }
    }
}

/*--- fesom_cal_shortwave_rad_kk — DEVICE twin of the sw_3d penetration profile (M5.20) -------
 * Per-surface-node map writing sw_3d (the 3-D shortwave penetration) on the DEVICE, so KPP
 * (substep 3) and tracer-diff (substep 13b) read it device-resident → eliminates the 519 MB/step
 * HtoD the host computation + the substep-3/13b re-pushes cost (the 2nd-biggest PCIe driver, M5.20
 * SYNC_LOG attribution). The `heat_flux += swsurf` SIDE EFFECT stays in the host cal_shortwave_rad
 * (a small nod2D op, unchanged) — this kernel only recomputes swsurf for the profile and writes
 * sw_3d. Race-free (each surface node owns its column). Mirrors fesom_cal_shortwave_rad arithmetic
 * op-for-op; exp/log10 run on the device → Serial bit-identical to the host twin (same libm on the
 * Serial CPU backend), CUDA climate-close (last-ULP transcendental divergence, the EOS class).
 * CONTRACT: jra->shortwave + a_ice device-current on entry, chl device-current (pushed in the
 * forcing phase when it updates); marks sw_3d modify_device(). */
void fesom_cal_shortwave_rad_kk(const struct fesom_mesh  *mesh,
                                const struct fesom_jra55 *jra,
                                const struct fesom_ice   *ice,
                                struct fesom_forcing     *forcing)
{
    if (!FESOM_PHASE1_USE_SW_PENE) return;
    const int    N    = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    nl   = mesh->nl;
    const real_t albw = (real_t)BULK_ALBW;
    const real_t vcpw = (real_t)FESOM_VCPW;

    auto a_ice = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto swr   = jra->shortwave_fld.d();
    auto chl   = forcing->chl_fld.d();
    auto sw_3d = forcing->sw_3d_fld.d();
    auto zbar3 = mesh->zbar_3d_n_fld.d();
    auto uln   = mesh->ulevels_nod2D_fld.d();
    auto nln   = mesh->nlevels_nod2D_fld.d();

    /* zero sw_3d over all local nodes/levels (mirrors the host memset, Fortran 39-43) */
    Kokkos::parallel_for("fesom_sw3d_zero", Kokkos::RangePolicy<>(0, (size_t)N * (size_t)nl),
        KOKKOS_LAMBDA(const std::size_t i) { sw_3d(i) = 0.0; });

    Kokkos::parallel_for("fesom_cal_shortwave_rad", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n2) {
            if (uln(n2) > 1)     return;        /* cavity: no penetration (F:51) */
            if (a_ice(n2) > 0.0) return;        /* under ice: none (F:52) */

            real_t swsurf = (1.0 - albw) * swr(n2);   /* = qsr */
            swsurf *= 0.54;                            /* visible part (300-750nm) */
            /* heat_flux[n2] += swsurf — done on the HOST in cal_shortwave_rad, NOT here. */

            real_t cc = chl(n2);
            if (cc < 0.02) cc = 0.02;
            real_t c  = Kokkos::log10(cc);
            real_t c2 = c*c, c3 = c2*c, c4 = c3*c, c5 = c4*c;
            real_t v1  = 0.008*c + 0.132*c2 + 0.038*c3 - 0.017*c4 - 0.007*c5;
            real_t v2  = 0.679 - v1;
            v1         = 0.321 + v1;
            real_t sc1 = 1.54  - 0.197*c + 0.166*c2 - 0.252*c3 - 0.055*c4 + 0.042*c5;
            real_t sc2 = 7.925 - 6.644*c + 3.662*c2 - 1.815*c3 - 0.218*c4 + 0.502*c5;

            swsurf /= vcpw;                            /* W/m² → K m/s (F:81) */

            const int nzmin = uln(n2) - 1;
            const int nzmax = nln(n2) - 1;
            sw_3d(FESOM_NODE3D(n2, nzmin, nl)) = swsurf;            /* F:85 */
            for (int k = nzmin + 1; k <= nzmax; ++k) {             /* F:86-93 */
                real_t z   = zbar3(FESOM_NODE3D(n2, k, nl));
                real_t aux = v1 * Kokkos::exp(z / sc1) + v2 * Kokkos::exp(z / sc2);
                sw_3d(FESOM_NODE3D(n2, k, nl)) = swsurf * aux;
                if (aux < 1.0e-5 || k == nzmax) {
                    sw_3d(FESOM_NODE3D(n2, k, nl)) = 0.0;
                    break;
                }
            }
        });

    forcing->sw_3d_fld.modify_device();
}
