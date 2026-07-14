#include "fesom_ice_thermo.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_forcing.h"
#include "fesom_halo.h"
#include "fesom_jra55.h"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_sss_runoff.h"
#include "fesom_tracers.h"
#include "fesom_types.h"

#include <Kokkos_Core.hpp>   // M4.3a: device cut_off clamp kernel
#include <string>            // M4.3a: FESOM_KK_VERIFY backend name
#include <algorithm>         // M4.3a: verify snapshot/restore copies
#include <vector>
#include <math.h>

/*
 * Mirror of Fortran TFrez(S) at ice_thermo_oce.F90:899.
 *   TFrez = -0.0575*S + 1.7105e-3 * sqrt(S**3) - 2.155e-4 * S*S
 * Millero (1978) / UNESCO; see Gill (1982).
 */
real_t fesom_ice_tfrez(real_t salinity)
{
    real_t s = salinity;
    return -0.0575 * s + 1.7105e-3 * sqrt(s * s * s) - 2.155e-4 * s * s;
}

/*
 * Mirror of Fortran flooding at ice_thermo_oce.F90:872.
 * Archimedes freeboard test; converts snow to ice when freeboard < 0.
 */
void fesom_ice_flooding(const fesom_ice_thermo *th,
                        real_t                 *h,
                        real_t                 *hsn)
{
    /* hdraft: displaced water (m) = (rhosno*hsn + h*rhoice) / rhowat */
    real_t hdraft = (th->rhosno * (*hsn) + (*h) * th->rhoice) * th->inv_rhowat;
    /* hflood: increase in ice thickness due to flooding */
    real_t hmin = (hdraft < *h) ? hdraft : *h;
    real_t hflood = hdraft - hmin;
    /* convert: add flooded volume to ice, remove mass-equivalent from snow */
    *h   += hflood;
    *hsn -= hflood * th->rhoice * th->inv_rhosno;
}

/*
 * Mirror of Fortran cut_off at ice_thermo_oce.F90:70.
 * Fortran loop bound: 1..myDim_nod2D+eDim_nod2D — halo included.
 */
void fesom_ice_cut_off(fesom_ice            *ice,
                       struct fesom_partit  *partit,
                       struct fesom_mesh    *mesh)
{
    (void)partit;
    real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;
    real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    real_t *m_snow = ice->data[FESOM_ICE_MSNOW].values;

    /* matches ice_thermo_oce.F90:102 — loop bound is myDim+eDim, halo included */
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    for (int n = 0; n < N; ++n) {
        /* upper cutoff on a_ice */
        if (a_ice[n] > 1.0) a_ice[n] = 1.0;
        /* lower cutoff on a_ice: zero all three if near-zero */
        if (a_ice[n] < 1.0e-9) {
            a_ice [n] = 0.0;
            m_ice [n] = 0.0;
            m_snow[n] = 0.0;
        }
        /* lower cutoff on m_ice: zero all three if near-zero */
        if (m_ice[n] < 1.0e-9) {
            m_ice [n] = 0.0;
            m_snow[n] = 0.0;
            a_ice [n] = 0.0;
        }
    }
}

/* ====================================================================== *
 *  M4.3a — DEVICE (Kokkos) twin of fesom_ice_cut_off. Per-node clamp over  *
 *  [0,N) (halo included, like the C); each node read-modify-writes only its *
 *  own a_ice/m_ice/m_snow → race-free, NO scatter → Serial AND OpenMP       *
 *  bit-identical. Marks the three ice-tracer values modify_device.          *
 * ====================================================================== */
void fesom_ice_cut_off_kk(fesom_ice            *ice,
                          struct fesom_partit  *partit,
                          struct fesom_mesh    *mesh)
{
    (void)partit;
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    auto a_ice  = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto m_ice  = ice->data[FESOM_ICE_MICE].values_fld.d();
    auto m_snow = ice->data[FESOM_ICE_MSNOW].values_fld.d();

    Kokkos::parallel_for("fesom_ice_cut_off", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            real_t a = a_ice(n), m = m_ice(n), s = m_snow(n);
            if (a > 1.0)      a = 1.0;
            if (a < 1.0e-9) { a = 0.0; m = 0.0; s = 0.0; }
            if (m < 1.0e-9) { m = 0.0; s = 0.0; a = 0.0; }
            a_ice(n) = a; m_ice(n) = m; m_snow(n) = s;
        });

    ice->data[FESOM_ICE_AICE].values_fld.modify_device();
    ice->data[FESOM_ICE_MICE].values_fld.modify_device();
    ice->data[FESOM_ICE_MSNOW].values_fld.modify_device();
}

/* FESOM_KK_VERIFY=icemap — cut_off READ-MODIFY-WRITES a_ice/m_ice/m_snow (clamps in place),
 * so this is the L26 capture-before: the driver snapshots the pre-clamp values; here snapshot
 * the KK result, restore the pre values, run the C twin, diff, restore KK. max|Δ|==0 on Serial. */
void fesom_ice_cut_off_verify(fesom_ice *ice, struct fesom_partit *partit, struct fesom_mesh *mesh,
                              int step_n, const std::vector<real_t> &pre_a,
                              const std::vector<real_t> &pre_m, const std::vector<real_t> &pre_s)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    real_t *a = ice->data[FESOM_ICE_AICE].values;
    real_t *m = ice->data[FESOM_ICE_MICE].values;
    real_t *s = ice->data[FESOM_ICE_MSNOW].values;
    std::vector<real_t> ka(a, a + N), km(m, m + N), ks(s, s + N);   /* KK result */
    std::copy(pre_a.begin(), pre_a.end(), a);                       /* restore pre-clamp inputs */
    std::copy(pre_m.begin(), pre_m.end(), m);
    std::copy(pre_s.begin(), pre_s.end(), s);
    fesom_ice_cut_off(ice, partit, mesh);                          /* C twin */
    double da = 0.0, dm = 0.0, ds = 0.0;
    for (int i = 0; i < N; ++i) {
        double x;
        x = std::fabs((double)ka[i]-(double)a[i]); if (x>da) da=x;
        x = std::fabs((double)km[i]-(double)m[i]); if (x>dm) dm=x;
        x = std::fabs((double)ks[i]-(double)s[i]); if (x>ds) ds=x;
    }
    std::copy(ka.begin(), ka.end(), a); std::copy(km.begin(), km.end(), m); std::copy(ks.begin(), ks.end(), s);
    double dmax = da; if (dm>dmax) dmax=dm; if (ds>dmax) dmax=ds;
    const std::string be = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=icemap] step %d backend=%s  max|Δ|: cut_off a_ice=%.3e m_ice=%.3e m_snow=%.3e\n",
                step_n, be.c_str(), da, dm, ds);
    std::fflush(stdout);
    if (be == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=icemap] FAIL step %d: cut_off Serial must be "
                             "bit-identical (max|Δ|=%.3e)\n", step_n, dmax); std::abort();
    }
}

/*
 * Mirror of Fortran obudget at ice_thermo_oce.F90:777.
 *
 * Flow:
 *   hfswrow  = (1 - albw) * fsh                          net shortwave
 *   hflwrow  = flo                                       incoming longwave
 *   hflwrdout= -emiss_wat * sigma * (t + tmelt)^4        outgoing longwave
 *   hfradow  = sum of the above                          total radiation
 *   hfsenow  = rhoair * cpair * ch * ug * (ta - t)       sensible heat
 *   evap_kgm2s = rhoair * ce * ug * (qa - b)             evaporation mass flux
 *   hflatow  = clhw * evap_kgm2s                         latent heat
 *   fh       = -(hfradow + hfsenow + hflatow) / cl       growth rate in m ice/s
 *   evap     = evap_kgm2s * inv_rhowat                   convert to m water/s
 *
 * where b is the saturated surface specific humidity
 *   b = c1 * exp(c4 * t / (t + c5))       (standard formula; Fortran default)
 * with c1=3.8e-3, c4=17.27, c5=237.3.
 *
 * The `open_water_albedo > 0` branches (time-dependent Taylor / Briegleb
 * albedo) are out of scope for this port — CORE2 default is 0.
 */
void fesom_ice_obudget(const fesom_ice_thermo *th,
                       real_t qa, real_t fsh, real_t flo,
                       real_t t,  real_t ug,  real_t ta,
                       real_t ch, real_t ce,
                       real_t *fh_out,      real_t *evap_out,
                       real_t *hflatow,     real_t *hfsenow,
                       real_t *hflwrdout,   real_t *hfswrow,
                       real_t *hflwrow,     real_t *hfradow)
{
    FESOM_CHECK(th->open_water_albedo == 0,
                "fesom_ice_obudget: open_water_albedo=%d not supported "
                "(only 0 = fixed albw is ported)", th->open_water_albedo);

    /* saturated surface specific humidity — standard formula branch */
    const real_t c1 = 3.8e-3;
    const real_t c4 = 17.27;
    const real_t c5 = 237.3;
    real_t b = c1 * exp(c4 * t / (t + c5));

    /* radiation heat fluxes, W/m² */
    real_t _hfswr   = (1.0 - th->albw) * fsh;
    real_t _hflwr   = flo;
    real_t _hflwrd  = -th->emiss_wat * th->boltzmann * pow(t + th->tmelt, 4.0);
    real_t _hfrad   = _hfswr + _hflwr + _hflwrd;

    /* sensible heat, W/m² */
    real_t _hfsen = th->rhoair * th->cpair * ch * ug * (ta - t);

    /* evaporation in kg/(m²·s), then latent heat */
    real_t _evap_kgm2s = th->rhoair * ce * ug * (qa - b);
    real_t _hflat = th->clhw * _evap_kgm2s;

    real_t _hftot = _hfrad + _hfsen + _hflat;

    /* outputs */
    *fh_out   = -_hftot / th->cl;          /* m ice / s */
    *evap_out = _evap_kgm2s * th->inv_rhowat;  /* m water / s, negative up */

    if (hfswrow)   *hfswrow   = _hfswr;
    if (hflwrow)   *hflwrow   = _hflwr;
    if (hflwrdout) *hflwrdout = _hflwrd;
    if (hfradow)   *hfradow   = _hfrad;
    if (hfsenow)   *hfsenow   = _hfsen;
    if (hflatow)   *hflatow   = _hflat;
}

/*
 * Mirror of Fortran budget at ice_thermo_oce.F90:657.
 *
 * Albedo selection (line 718-730):
 *   t < 0:  freezing → albsn (snow present) | albi  (no snow)
 *   t ≥ 0:  melting  → albsnm                | albim
 *
 * 5-iteration Newton-Raphson on t (line 741-752):
 *   Residual A1+A2+C, derivative A3, update t += residual/derivative.
 *   B = q1/rhoair * exp(q2/(t+tmelt)) is the saturated specific humidity over ice.
 *   q1 = 11637800.0; q2 = -5897.8 (these are different from obudget's q1,q2
 *   because they parameterise saturation over ice, not water).
 *
 * Final fluxes (line 757-770) match obudget structure but use ice albedo and
 * ice latent heat (clhi instead of clhw) and write `subli` instead of `evap`.
 */
void fesom_ice_budget(const fesom_ice_thermo *th,
                      real_t hice, real_t hsn,
                      real_t *t,
                      real_t ta, real_t qa,
                      real_t fsh, real_t flo,
                      real_t ug, real_t S_oc,
                      real_t ch_i, real_t ce_i,
                      real_t *fh, real_t *subli)
{
    /* albedo selection */
    real_t alb;
    if (*t < 0.0) {
        alb = (hsn > 0.0) ? th->albsn  : th->albi;
    } else {
        alb = (hsn > 0.0) ? th->albsnm : th->albim;
    }

    const real_t q1 = 11637800.0;
    const real_t q2 = -5897.8;
    const int    imax = 5;

    real_t d1 = th->rhoair * th->cpair * ch_i;
    real_t d2 = th->rhoair * ce_i;
    real_t d3 = d2 * th->clhi;

    /* total incoming atmospheric heat flux (constant across iterations) */
    real_t A1 = (1.0 - alb) * fsh + flo + d1 * ug * ta + d3 * ug * qa;

    /* Newton-Raphson on t */
    real_t B = 0.0;  /* outlives loop for the final flux pass */
    for (int iter = 0; iter < imax; ++iter) {
        real_t tk = *t + th->tmelt;
        B = q1 * th->inv_rhoair * exp(q2 / tk);
        real_t A2 = -d1 * ug * (*t)
                    -d3 * ug * B
                    -th->emiss_ice * th->boltzmann * (tk * tk * tk * tk);
        real_t A3 = -d3 * ug * B * q2 / (tk * tk);
        real_t C  = th->con / hice;
        A3 += C + d1 * ug
              + 4.0 * th->emiss_ice * th->boltzmann * (tk * tk * tk);
        C *= (fesom_ice_tfrez(S_oc) - *t);
        *t += (A1 + A2 + C) / A3;
    }
    if (*t > 0.0) *t = 0.0;

    /* final fluxes (W/m²) using the converged t and last-iteration B */
    real_t tk = *t + th->tmelt;
    real_t hfrad = (1.0 - alb) * fsh
                   + flo
                   - th->emiss_ice * th->boltzmann * (tk * tk * tk * tk);
    real_t hfsen = d1 * ug * (ta - *t);
    real_t _subli = d2 * ug * (qa - B);
    real_t hflat  = th->clhi * _subli;
    real_t hftot  = hfrad + hfsen + hflat;

    *fh    = -hftot / th->cl;
    *subli = _subli * th->inv_rhowat;   /* negative upward */
}

/*
 * Mirror of Fortran therm_ice at ice_thermo_oce.F90:367.
 *
 * Phase B4a scope (per .h):
 *   - Skip flooding integration (lines 637-649)         → iflice = 0
 *   - Only use_virt_salt = false branch (lines 614-617)
 *   - Skip new_iclasses=true branch (no hpdf in struct)
 *   - Always assume snowdist = true (Fortran default)
 *
 * The body follows the Fortran line-by-line. Variable names match Fortran
 * to keep diff visible; obvious in/out semantics use C pointer style.
 */
static inline real_t fmin_w(real_t a, real_t b) { return a < b ? a : b; }
static inline real_t fmax_w(real_t a, real_t b) { return a > b ? a : b; }

void fesom_therm_ice(const fesom_ice_thermo *th,
                     int use_virt_salt,
                     real_t *h, real_t *hsn, real_t *A,
                     real_t fsh, real_t flo, real_t Ta, real_t qa,
                     real_t rain, real_t snow, real_t runo, real_t rsss,
                     real_t ug, real_t ustar, real_t T_oc, real_t S_oc,
                     real_t H_ML, real_t *t, real_t ice_dt,
                     real_t ch, real_t ce, real_t ch_i, real_t ce_i,
                     real_t evap_in,
                     real_t *fw, real_t *fwice, real_t *fwsnw,
                     real_t *ehf, real_t *evap, real_t *rsf,
                     real_t *dhgrowth, real_t *dhsngrowth, real_t *dAgrowth,
                     real_t *iflice,
                     real_t *hflatow, real_t *hfsenow, real_t *hflwrdout,
                     real_t *hfswrow, real_t *hflwrow, real_t *hfradow,
                     real_t lid_clo, real_t geolon, real_t geolat,
                     real_t *subli)
{
    (void)evap_in;   /* Fortran ignores the value too — see line 472 */

    /* B4a: assert snowdist=true (Fortran default for CORE2) */
    FESOM_CHECK(th->snowdist == 1, "fesom_therm_ice: snowdist=false not ported");

    /* dhgrowth holds the pre-step ice mass for end-of-step diff */
    real_t _dhgrowth = *h;

    /* Effective snow thickness (Semtner 1976 0-layer); always added under snowdist=true */
    real_t snthick = (*hsn) * (th->con / th->consn) / fmax_w(*A, th->armin);
    real_t thick   = (*h) / fmax_w(*A, th->armin);
    thick = snthick + thick;

    /* --- Open-ocean growth rate (single obudget call) --- */
    real_t rhow = 0.0, _evap = 0.0;
    real_t _hflatow = 0.0, _hfsenow = 0.0, _hflwrdout = 0.0;
    real_t _hfswrow = 0.0, _hflwrow = 0.0, _hfradow   = 0.0;
    fesom_ice_obudget(th, qa, fsh, flo, T_oc, ug, Ta, ch, ce,
                      &rhow, &_evap,
                      &_hflatow, &_hfsenow, &_hflwrdout,
                      &_hfswrow, &_hflwrow, &_hfradow);
    /* Scale radiative components by open-water fraction */
    real_t one_minus_A = 1.0 - (*A);
    _hflatow   *= one_minus_A;
    _hfsenow   *= one_minus_A;
    _hfradow   *= one_minus_A;
    _hfswrow   *= one_minus_A;
    _hflwrow   *= one_minus_A;
    _hflwrdout *= one_minus_A;
    if (hflatow)   *hflatow   = _hflatow;
    if (hfsenow)   *hfsenow   = _hfsenow;
    if (hflwrdout) *hflwrdout = _hflwrdout;
    if (hfswrow)   *hfswrow   = _hfswrow;
    if (hflwrow)   *hflwrow   = _hflwrow;
    if (hfradow)   *hfradow   = _hfradow;

    /* --- Ice-covered growth rate (iclasses loop, Hibler 1984) --- */
    real_t rhice = 0.0;
    real_t _subli = 0.0;
    if (thick > th->hmin) {
        for (int k = 1; k <= th->iclasses; ++k) {
            real_t thact = (real_t)(2*k - 1) * thick / (real_t)th->iclasses;
            /* snowdist=true → no per-class snthick adjustment */
            real_t shice = 0.0, subli_i = 0.0;
            fesom_ice_budget(th, thact, *hsn, t, Ta, qa, fsh, flo,
                             ug, S_oc, ch_i, ce_i, &shice, &subli_i);
            rhice  += shice;
            _subli += subli_i;
        }
        rhice  /= (real_t)th->iclasses;
        _subli /= (real_t)th->iclasses;
    }

    /* Convert growth rates [m/s] -> per-step [m/DT] */
    rhow  *= ice_dt;
    rhice *= ice_dt;

    /* Areal weighting */
    real_t show  = rhow  * one_minus_A;
    real_t shice = rhice * (*A);
    real_t sh    = show + shice;

    /* Atmospheric heat flux averaged over grid cell */
    real_t ahf = -th->cl * sh / ice_dt;

    /* Precipitation into ocean (m water/s); includes runoff */
    real_t prec = rain + runo + snow * one_minus_A;

    /* Snow fall above ice — accumulate */
    *hsn += snow * ice_dt * (*A) * 1000.0 * th->inv_rhosno;
    real_t _dhsngrowth = *hsn;

    _evap   *= one_minus_A;          /* open-water evap weighted */
    _subli  *= (*A);                  /* ice sublimation weighted */

    /* Snow melt by negative atmospheric heat flux (sh<0 → melting) */
    real_t hsntmp = -fmin_w(sh, 0.0) * th->rhoice * th->inv_rhosno;
    hsntmp = fmin_w(hsntmp, *hsn);
    *hsn -= hsntmp;

    /* Negative atmospheric heat flux remaining after snow melt */
    real_t rh = sh + hsntmp * th->rhosno / th->rhoice;
    *h = fmax_w(*h, 0.0);

    /* Ocean-to-ice heat flux (ustar parameterisation) */
    real_t Tfrez_S = fesom_ice_tfrez(S_oc);
    real_t o2ihf = (T_oc - Tfrez_S) * 0.006 * ustar * th->cc * (*A)
                 + (T_oc - Tfrez_S) * H_ML / ice_dt * th->cc * one_minus_A;
    rh -= o2ihf * ice_dt / th->cl;
    real_t qhst = *h + rh;

    /* Melt remaining snow if ML heat content left over */
    real_t sn = *hsn + fmin_w(qhst, 0.0) * th->rhoice * th->inv_rhosno;
    sn = fmax_w(sn, 0.0);
    *hsn = sn;
    *h   = fmax_w(qhst, 0.0);
    if (*h < 1.0e-6) *h = 0.0;

    /* Compute changes per second */
    _dhgrowth   = (*h   - _dhgrowth)   / ice_dt;
    _dhsngrowth = (*hsn - _dhsngrowth) / ice_dt;
    if (dhgrowth)   *dhgrowth   = _dhgrowth;
    if (dhsngrowth) *dhsngrowth = _dhsngrowth;

    /* Net surface heat flux to ocean */
    *ehf = ahf + th->cl * (_dhgrowth + (th->rhosno / th->rhoice) * _dhsngrowth);

    /* Freshwater fluxes — Fortran lines 612-623 */
    real_t _fwice, _fwsnw, _fw, _rsf;
    if (!use_virt_salt) {
        /* real-salt path (use_virt_salt = false): salt leaves with melted ice */
        _fwice = -_dhgrowth   * th->rhoice * th->inv_rhowat;
        _fwsnw = -_dhsngrowth * th->rhosno * th->inv_rhowat;
        _fw    = prec + _evap + _fwice + _fwsnw;
        _rsf   = _fwice * th->Sice;
    } else {
        /* virtual-salt path (linfs default): scale fwice by (rsss-Sice)/rsss */
        _fwice = -_dhgrowth   * th->rhoice * th->inv_rhowat * (rsss - th->Sice) / rsss;
        _fwsnw = -_dhsngrowth * th->rhosno * th->inv_rhowat;
        _fw    = prec + _evap + _fwice + _fwsnw;
        _rsf   = 0.0;
    }
    if (fwice) *fwice = _fwice;
    if (fwsnw) *fwsnw = _fwsnw;

    /* Compactness change (Hibler 1979 eq. 16) */
    rh = -fmin_w(*h, -rh);
    real_t rA  = rhow - o2ihf * ice_dt / th->cl;
    real_t Aold = *A;
    *A = *A + th->c_melt * fmin_w(rh, 0.0) * (*A) / fmax_w(*h, th->hmin)
            + fmax_w(rA, 0.0) * (1.0 - *A) / lid_clo;
    *A = fmin_w(*A, *h * 1.0e6);
    *A = fmin_w(fmax_w(*A, 0.0), 1.0);
    if (dAgrowth) *dAgrowth = (*A - Aold) / ice_dt;

    /* Flooding (snow-to-ice conversion) — Fortran lines 638-649 */
    real_t _iflice = *h;
    fesom_ice_flooding(th, h, hsn);
    _iflice = (*h - _iflice) / ice_dt;
    if (iflice) *iflice = _iflice;

    /*
     * Salt conservation correction for the flooding step.
     * Snow-derived ice carries no salt; without correction the system would
     * "produce" salt as snow becomes ice. The correction adjusts rsf (real
     * salt path) or fw (virtual salt path) to compensate.
     */
    if (!use_virt_salt) {
        _rsf -= _iflice * th->rhoice * th->inv_rhowat * th->Sice;
    } else {
        _fw  += _iflice * th->rhoice * th->inv_rhowat * th->Sice / rsss;
    }
    if (rsf) *rsf = _rsf;
    if (fw)  *fw  = _fw;

    if (evap)  *evap  = _evap + _subli;
    if (subli) *subli = _subli;

    (void)geolon; (void)geolat;  /* reserved for Taylor/Briegleb branch (out of scope) */
}

/*
 * Per-node sea-ice thermodynamics driver.
 * Mirror of Fortran thermodynamics at ice_thermo_oce.F90:148.
 *
 * Cavity skip uses ulevels_nod2D > 1.
 * Phase B: l_snow=true assumed (CORE2 has prec_snow). The l_snow=false branch
 * (synth snow from Tair) lands when JRA55 doesn't provide snow.
 */
void fesom_ice_thermodynamics(fesom_ice                     *ice,
                              struct fesom_partit           *partit,
                              struct fesom_mesh             *mesh,
                              struct fesom_forcing          *forcing,   /* M6.3: now a PRODUCER
                                    (real_salt_flux / evaporation / ice_sublimation) */
                              const struct fesom_jra55      *jra,
                              const struct fesom_sss_runoff *sr)
{
    int nl = mesh->nl;
    real_t  rsss_default = sr->ref_sss;
    int     ref_sss_local = sr->ref_sss_local;
    real_t  h_ml = ice->thermo.h_ml;

    /* ---- Loop 1 (myDim only): friction velocity ---- */
    /* Fortran ice_thermo_oce.F90:230-235 — myDim only, then exchange */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) {
            ice->thermo.ustar[n] = 0.0;
            continue;
        }
        real_t du = ice->uice[n] - ice->srfoce_u[n];
        real_t dv = ice->vice[n] - ice->srfoce_v[n];
        ice->thermo.ustar[n] = sqrt((du*du + dv*dv) * ice->cd_oce_ice);
    }
    /* Halo exchange — Fortran line 238 (call exchange_nod) */
    fesom_exchange_nod2D(ice->thermo.ustar, partit);

    /* ---- Loop 2 (myDim+eDim, halo included): per-node therm_ice ---- */
    /* Fortran line 244 — loop bound includes halo, Fortran writes halo entries */
    /* M7 D.1 (FORCEDEV): this host C twin reads all 8 JRA arrays raw. Verify-only
     * (FESOM_KK_VERIFY=icethermo), but with the device producer live the host mirrors are
     * stale — without this the twin would compare the _kk output against garbage. */
    if (fesom_forcing_dev_on()) {
        struct fesom_jra55 *j = const_cast<struct fesom_jra55 *>(jra);
        j->u_wind_fld.sync_host();    j->v_wind_fld.sync_host();
        j->shum_fld.sync_host();      j->shortwave_fld.sync_host();
        j->longwave_fld.sync_host();  j->Tair_fld.sync_host();
        j->prec_rain_fld.sync_host(); j->prec_snow_fld.sync_host();
    }

    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    for (int n = 0; n < N; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;   /* cavity skip */

        /* Inputs */
        real_t h    = ice->data[FESOM_ICE_MICE].values[n];
        real_t hsn  = ice->data[FESOM_ICE_MSNOW].values[n];
        real_t A    = ice->data[FESOM_ICE_AICE].values[n];
        real_t fsh  = jra->shortwave[n];
        real_t flo  = jra->longwave[n];
        real_t Ta   = jra->Tair[n];
        real_t qa   = jra->shum[n];
        real_t rain = jra->prec_rain[n];
        real_t snow = jra->prec_snow[n];
        real_t runo = forcing->runoff[n];
        real_t ug   = sqrt(jra->u_wind[n]*jra->u_wind[n]
                         + jra->v_wind[n]*jra->v_wind[n]);
        real_t ustar = ice->thermo.ustar[n];
        real_t T_oc  = ice->srfoce_temp[n];
        real_t S_oc  = ice->srfoce_salt[n];
        real_t rsss  = ref_sss_local ? S_oc : rsss_default;
        real_t t     = ice->thermo.t_skin[n];
        real_t ch    = forcing->Ch_atm_oce[n];
        real_t ce    = forcing->Ce_atm_oce[n];
        real_t lid_clo = (mesh->geo_coord_nod2D[2*n + 1] > 0.0)
                            ? ice->thermo.h0
                            : ice->thermo.h0_s;
        real_t geolon = mesh->geo_coord_nod2D[2*n + 0];
        real_t geolat = mesh->geo_coord_nod2D[2*n + 1];

        real_t fw = 0, fwice = 0, fwsnw = 0, ehf = 0, evap = 0, rsf = 0;
        real_t ithdgr = 0, ithdgrsn = 0, ithdgra = 0, iflice = 0;
        real_t hflatow = 0, hfsenow = 0, hflwrdout = 0;
        real_t hfswrow = 0, hflwrow = 0, hfradow = 0, subli = 0;

        fesom_therm_ice(&ice->thermo,
                        sr->use_virt_salt,
                        &h, &hsn, &A,
                        fsh, flo, Ta, qa, rain, snow, runo, rsss,
                        ug, ustar, T_oc, S_oc, h_ml, &t, ice->ice_dt,
                        ch, ce, FESOM_CH_ATM_ICE, FESOM_CE_ATM_ICE,
                        /*evap_in=*/0.0,
                        &fw, &fwice, &fwsnw, &ehf, &evap, &rsf,
                        &ithdgr, &ithdgrsn, &ithdgra, &iflice,
                        &hflatow, &hfsenow, &hflwrdout,
                        &hfswrow, &hflwrow, &hfradow,
                        lid_clo, geolon, geolat, &subli);

        /* Backup of old values (Fortran lines 311-314) */
        ice->data[FESOM_ICE_MICE].values_old [n] = ice->data[FESOM_ICE_MICE].values [n];
        ice->data[FESOM_ICE_MSNOW].values_old[n] = ice->data[FESOM_ICE_MSNOW].values[n];
        ice->data[FESOM_ICE_AICE].values_old [n] = ice->data[FESOM_ICE_AICE].values [n];
        ice->thermo.thdgr_old[n]                 = ice->thermo.thdgr[n];

        /* New values */
        ice->data[FESOM_ICE_MICE].values [n] = h;
        ice->data[FESOM_ICE_MSNOW].values[n] = hsn;
        ice->data[FESOM_ICE_AICE].values [n] = A;

        ice->thermo.t_skin[n]   = t;
        ice->flx_fw[n]          = fw;        /* + down (will be negated in oce_fluxes) */
        ice->flx_h [n]          = ehf;       /* + down */
        ice->thermo.thdgr  [n]  = ithdgr;
        ice->thermo.thdgrsn[n]  = ithdgrsn;
        ice->thermo.thdgra [n]  = ithdgra;

        /* M6.3 (zstar): the three per-node flux stores the port used to THROW AWAY.
         *   real_salt_flux — ice_thermo_oce.F90:352, assigned UNCONDITIONALLY. The VALUE is
         *     use_virt_salt-branched inside therm_ice (0 under linfs; fwice·Sice −
         *     iflice·ρice/ρwat·Sice under zstar), so storing it unconditionally is safe AND
         *     faithful: under linfs it stores a hard 0.
         *   evaporation / ice_sublimation — :324-325; they feed the zstar freshwater
         *     global-balancing assembly in oce_fluxes. Dead stores under linfs.
         * All three are therefore INVISIBLE to the linfs path (the byte gate proves it). */
        forcing->real_salt_flux [n] = rsf;
        forcing->evaporation    [n] = evap;
        forcing->ice_sublimation[n] = subli;

        /* Remaining diagnostics (fw_ice/fw_snw/hf_Q*) still unstored — no arrays, no consumer. */
        (void)iflice;
        (void)fwice; (void)fwsnw;
        (void)hflatow; (void)hfsenow; (void)hflwrdout;
        (void)hfswrow; (void)hflwrow; (void)hfradow;
    }

    (void)nl;  /* unused for now; reserved if surface-level extraction is moved here */
}

/* ======================================================================== *
 *  M4.3d-a — DEVICE (Kokkos) twin of the sea-ice thermodynamics. The column  *
 *  physics is per-node and race-free (each node read-modify-writes only its   *
 *  own a/m/ms + t_skin + flx_* + thdgr*), so it is a single map over [0,N)    *
 *  (the M2.7 bc_surface_kk / per-node-TDMA shape) — bit-identical Serial AND   *
 *  OpenMP, NO scatter. therm_ice + obudget/budget/flooding/tfrez become        *
 *  KOKKOS_INLINE_FUNCTION device twins (the kpp_wscale_kk precedent), taking    *
 *  a POD `IceThermC` (the scalar mirror of fesom_ice_thermo — its Field         *
 *  members can't cross to the device, so the scalars are copied into a POD      *
 *  captured by value). The C originals stay the verify oracle (D19). ⚠️         *
 *  FORCED-ONLY → meaningful only on CORE2 (L42).                                *
 * ======================================================================== */

/* POD scalar mirror of fesom_ice_thermo (drop the Field members; capture by value). */
struct IceThermC {
    real_t rhoair, inv_rhoair, rhowat, inv_rhowat, rhoice, inv_rhoice, rhosno, inv_rhosno;
    real_t cpair, clhw, clhi, cc, cl, tmelt, boltzmann;
    real_t hmin, armin, con, consn, Sice;
    real_t emiss_ice, emiss_wat, albsn, albsnm, albi, albim, albw, c_melt;
    int    iclasses, snowdist, open_water_albedo;
};
static IceThermC make_therm_c(const fesom_ice_thermo *t)
{
    IceThermC c;
    c.rhoair=t->rhoair; c.inv_rhoair=t->inv_rhoair; c.rhowat=t->rhowat; c.inv_rhowat=t->inv_rhowat;
    c.rhoice=t->rhoice; c.inv_rhoice=t->inv_rhoice; c.rhosno=t->rhosno; c.inv_rhosno=t->inv_rhosno;
    c.cpair=t->cpair; c.clhw=t->clhw; c.clhi=t->clhi; c.cc=t->cc; c.cl=t->cl;
    c.tmelt=t->tmelt; c.boltzmann=t->boltzmann;
    c.hmin=t->hmin; c.armin=t->armin; c.con=t->con; c.consn=t->consn; c.Sice=t->Sice;
    c.emiss_ice=t->emiss_ice; c.emiss_wat=t->emiss_wat;
    c.albsn=t->albsn; c.albsnm=t->albsnm; c.albi=t->albi; c.albim=t->albim; c.albw=t->albw;
    c.c_melt=t->c_melt;
    c.iclasses=t->iclasses; c.snowdist=t->snowdist; c.open_water_albedo=t->open_water_albedo;
    return c;
}

KOKKOS_INLINE_FUNCTION real_t tfrez_kk(real_t s)
{ return -0.0575*s + 1.7105e-3*Kokkos::sqrt(s*s*s) - 2.155e-4*s*s; }

KOKKOS_INLINE_FUNCTION real_t fmin_kk(real_t a, real_t b) { return a < b ? a : b; }
KOKKOS_INLINE_FUNCTION real_t fmax_kk(real_t a, real_t b) { return a > b ? a : b; }

/* device twin of fesom_ice_flooding */
KOKKOS_INLINE_FUNCTION
void flooding_kk(const IceThermC &th, real_t *h, real_t *hsn)
{
    real_t hdraft = (th.rhosno*(*hsn) + (*h)*th.rhoice) * th.inv_rhowat;
    real_t hmin   = (hdraft < *h) ? hdraft : *h;
    real_t hflood = hdraft - hmin;
    *h   += hflood;
    *hsn -= hflood * th.rhoice * th.inv_rhosno;
}

/* device twin of fesom_ice_obudget (open_water_albedo==0 branch only). */
KOKKOS_INLINE_FUNCTION
void obudget_kk(const IceThermC &th, real_t qa, real_t fsh, real_t flo,
                real_t t, real_t ug, real_t ta, real_t ch, real_t ce,
                real_t *fh_out, real_t *evap_out,
                real_t *hflatow, real_t *hfsenow, real_t *hflwrdout,
                real_t *hfswrow, real_t *hflwrow, real_t *hfradow)
{
    const real_t c1 = 3.8e-3, c4 = 17.27, c5 = 237.3;
    real_t b = c1 * Kokkos::exp(c4 * t / (t + c5));
    real_t _hfswr  = (1.0 - th.albw) * fsh;
    real_t _hflwr  = flo;
    real_t _hflwrd = -th.emiss_wat * th.boltzmann * Kokkos::pow(t + th.tmelt, 4.0);
    real_t _hfrad  = _hfswr + _hflwr + _hflwrd;
    real_t _hfsen  = th.rhoair * th.cpair * ch * ug * (ta - t);
    real_t _evap_kgm2s = th.rhoair * ce * ug * (qa - b);
    real_t _hflat  = th.clhw * _evap_kgm2s;
    real_t _hftot  = _hfrad + _hfsen + _hflat;
    *fh_out   = -_hftot / th.cl;
    *evap_out = _evap_kgm2s * th.inv_rhowat;
    *hfswrow = _hfswr; *hflwrow = _hflwr; *hflwrdout = _hflwrd; *hfradow = _hfrad;
    *hfsenow = _hfsen; *hflatow = _hflat;
}

/* device twin of fesom_ice_budget (5-iteration Newton-Raphson on t). */
KOKKOS_INLINE_FUNCTION
void budget_kk(const IceThermC &th, real_t hice, real_t hsn, real_t *t,
               real_t ta, real_t qa, real_t fsh, real_t flo,
               real_t ug, real_t S_oc, real_t ch_i, real_t ce_i,
               real_t *fh, real_t *subli)
{
    real_t alb;
    if (*t < 0.0) alb = (hsn > 0.0) ? th.albsn  : th.albi;
    else          alb = (hsn > 0.0) ? th.albsnm : th.albim;

    const real_t q1 = 11637800.0, q2 = -5897.8;
    const int imax = 5;
    real_t d1 = th.rhoair * th.cpair * ch_i;
    real_t d2 = th.rhoair * ce_i;
    real_t d3 = d2 * th.clhi;
    real_t A1 = (1.0 - alb) * fsh + flo + d1 * ug * ta + d3 * ug * qa;

    real_t B = 0.0;
    for (int iter = 0; iter < imax; ++iter) {
        real_t tk = *t + th.tmelt;
        B = q1 * th.inv_rhoair * Kokkos::exp(q2 / tk);
        real_t A2 = -d1*ug*(*t) - d3*ug*B - th.emiss_ice*th.boltzmann*(tk*tk*tk*tk);
        real_t A3 = -d3*ug*B*q2/(tk*tk);
        real_t C  = th.con / hice;
        A3 += C + d1*ug + 4.0*th.emiss_ice*th.boltzmann*(tk*tk*tk);
        C *= (tfrez_kk(S_oc) - *t);
        *t += (A1 + A2 + C) / A3;
    }
    if (*t > 0.0) *t = 0.0;

    real_t tk = *t + th.tmelt;
    real_t hfrad = (1.0 - alb)*fsh + flo - th.emiss_ice*th.boltzmann*(tk*tk*tk*tk);
    real_t hfsen = d1 * ug * (ta - *t);
    real_t _subli = d2 * ug * (qa - B);
    real_t hflat  = th.clhi * _subli;
    real_t hftot  = hfrad + hfsen + hflat;
    *fh    = -hftot / th.cl;
    *subli = _subli * th.inv_rhowat;
}

/* device twin of fesom_therm_ice (snowdist==1, the CORE2 path). use_virt_salt passed in. */
KOKKOS_INLINE_FUNCTION
void therm_ice_kk(const IceThermC &th, int use_virt_salt,
                  real_t *h, real_t *hsn, real_t *A,
                  real_t fsh, real_t flo, real_t Ta, real_t qa,
                  real_t rain, real_t snow, real_t runo, real_t rsss,
                  real_t ug, real_t ustar, real_t T_oc, real_t S_oc,
                  real_t H_ML, real_t *t, real_t ice_dt,
                  real_t ch, real_t ce, real_t ch_i, real_t ce_i,
                  real_t *fw, real_t *fwice, real_t *fwsnw,
                  real_t *ehf, real_t *evap, real_t *rsf,
                  real_t *dhgrowth, real_t *dhsngrowth, real_t *dAgrowth,
                  real_t *iflice, real_t lid_clo,
                  real_t *subli)   /* M6.3: the device twin computed _subli and dropped it; the
                                    * HOST twin has always returned it. Now both do. */
{
    real_t _dhgrowth = *h;
    real_t snthick = (*hsn) * (th.con / th.consn) / fmax_kk(*A, th.armin);
    real_t thick   = (*h) / fmax_kk(*A, th.armin);
    thick = snthick + thick;

    real_t rhow = 0.0, _evap = 0.0;
    real_t d_a=0.0,d_b=0.0,d_c=0.0,d_d=0.0,d_e=0.0,d_f=0.0;   /* obudget diagnostics (unused here) */
    obudget_kk(th, qa, fsh, flo, T_oc, ug, Ta, ch, ce,
               &rhow, &_evap, &d_a, &d_b, &d_c, &d_d, &d_e, &d_f);
    real_t one_minus_A = 1.0 - (*A);

    real_t rhice = 0.0, _subli = 0.0;
    if (thick > th.hmin) {
        for (int k = 1; k <= th.iclasses; ++k) {
            real_t thact = (real_t)(2*k - 1) * thick / (real_t)th.iclasses;
            real_t shice = 0.0, subli_i = 0.0;
            budget_kk(th, thact, *hsn, t, Ta, qa, fsh, flo, ug, S_oc, ch_i, ce_i, &shice, &subli_i);
            rhice  += shice;
            _subli += subli_i;
        }
        rhice  /= (real_t)th.iclasses;
        _subli /= (real_t)th.iclasses;
    }

    rhow  *= ice_dt;
    rhice *= ice_dt;
    real_t show  = rhow  * one_minus_A;
    real_t shice = rhice * (*A);
    real_t sh    = show + shice;
    real_t ahf   = -th.cl * sh / ice_dt;
    real_t prec  = rain + runo + snow * one_minus_A;

    *hsn += snow * ice_dt * (*A) * 1000.0 * th.inv_rhosno;
    real_t _dhsngrowth = *hsn;

    _evap  *= one_minus_A;
    _subli *= (*A);

    real_t hsntmp = -fmin_kk(sh, 0.0) * th.rhoice * th.inv_rhosno;
    hsntmp = fmin_kk(hsntmp, *hsn);
    *hsn -= hsntmp;

    real_t rh = sh + hsntmp * th.rhosno / th.rhoice;
    *h = fmax_kk(*h, 0.0);

    real_t Tfrez_S = tfrez_kk(S_oc);
    real_t o2ihf = (T_oc - Tfrez_S) * 0.006 * ustar * th.cc * (*A)
                 + (T_oc - Tfrez_S) * H_ML / ice_dt * th.cc * one_minus_A;
    rh -= o2ihf * ice_dt / th.cl;
    real_t qhst = *h + rh;

    real_t sn = *hsn + fmin_kk(qhst, 0.0) * th.rhoice * th.inv_rhosno;
    sn = fmax_kk(sn, 0.0);
    *hsn = sn;
    *h   = fmax_kk(qhst, 0.0);
    if (*h < 1.0e-6) *h = 0.0;

    _dhgrowth   = (*h   - _dhgrowth)   / ice_dt;
    _dhsngrowth = (*hsn - _dhsngrowth) / ice_dt;
    *dhgrowth   = _dhgrowth;
    *dhsngrowth = _dhsngrowth;

    *ehf = ahf + th.cl * (_dhgrowth + (th.rhosno / th.rhoice) * _dhsngrowth);

    real_t _fwice, _fwsnw, _fw, _rsf;
    if (!use_virt_salt) {
        _fwice = -_dhgrowth   * th.rhoice * th.inv_rhowat;
        _fwsnw = -_dhsngrowth * th.rhosno * th.inv_rhowat;
        _fw    = prec + _evap + _fwice + _fwsnw;
        _rsf   = _fwice * th.Sice;
    } else {
        _fwice = -_dhgrowth   * th.rhoice * th.inv_rhowat * (rsss - th.Sice) / rsss;
        _fwsnw = -_dhsngrowth * th.rhosno * th.inv_rhowat;
        _fw    = prec + _evap + _fwice + _fwsnw;
        _rsf   = 0.0;
    }
    *fwice = _fwice; *fwsnw = _fwsnw;

    rh = -fmin_kk(*h, -rh);
    real_t rA   = rhow - o2ihf * ice_dt / th.cl;
    real_t Aold = *A;
    *A = *A + th.c_melt * fmin_kk(rh, 0.0) * (*A) / fmax_kk(*h, th.hmin)
            + fmax_kk(rA, 0.0) * (1.0 - *A) / lid_clo;
    *A = fmin_kk(*A, *h * 1.0e6);
    *A = fmin_kk(fmax_kk(*A, 0.0), 1.0);
    *dAgrowth = (*A - Aold) / ice_dt;

    real_t _iflice = *h;
    flooding_kk(th, h, hsn);
    _iflice = (*h - _iflice) / ice_dt;
    *iflice = _iflice;

    if (!use_virt_salt) _rsf -= _iflice * th.rhoice * th.inv_rhowat * th.Sice;
    else                _fw  += _iflice * th.rhoice * th.inv_rhowat * th.Sice / rsss;
    *rsf = _rsf;
    *fw  = _fw;
    *evap = _evap + _subli;
    if (subli) *subli = _subli;   /* M6.3 (zstar): feeds the freshwater balancing assembly */
}

/* D21 internal-halo bracket on a nod2D Field (the FCT/EVP idiom): modify_device() always so the
 * driver OUT sync_host copies the result at np==1; the host round-trip only at np>1. */
static inline void therm_halo_nod2D(fesom::Field &f, struct fesom_partit *partit)
{
    f.modify_device();
    if (partit && partit->npes > 1) {
        f.sync_host();
        fesom_exchange_nod2D(f.h_checked(), partit);
        f.modify_host();
        f.sync_device();
    }
}

#include "fesom_jra55.h"   /* M4.3d-a: the Field-backed jra physics arrays (device Views) */

void fesom_ice_thermodynamics_kk(fesom_ice                     *ice,
                                 struct fesom_partit           *partit,
                                 struct fesom_mesh             *mesh,
                                 struct fesom_forcing          *forcing,  /* M6.3: PRODUCER */
                                 const struct fesom_jra55      *jra,
                                 const struct fesom_sss_runoff *sr)
{
    const int    myDim = mesh->myDim_nod2D;
    const int    N     = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const real_t cd    = ice->cd_oce_ice;
    const real_t ice_dt = ice->ice_dt;
    const real_t h_ml  = ice->thermo.h_ml;
    const real_t h0    = ice->thermo.h0;
    const real_t h0_s  = ice->thermo.h0_s;
    const int    uvs   = sr->use_virt_salt;
    const int    ref_sss_local = sr->ref_sss_local;
    const real_t rsss_default  = sr->ref_sss;
    const real_t ch_i  = (real_t)FESOM_CH_ATM_ICE;
    const real_t ce_i  = (real_t)FESOM_CE_ATM_ICE;
    const IceThermC thc = make_therm_c(&ice->thermo);

    auto uice  = ice->uice_fld.d();      auto vice  = ice->vice_fld.d();
    auto sru   = ice->srfoce_u_fld.d();  auto srv   = ice->srfoce_v_fld.d();
    auto srt   = ice->srfoce_temp_fld.d(); auto srs = ice->srfoce_salt_fld.d();
    auto a_ice = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto m_ice = ice->data[FESOM_ICE_MICE].values_fld.d();
    auto m_snw = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    auto vold_a = ice->data[FESOM_ICE_AICE].values_old_fld.d();
    auto vold_m = ice->data[FESOM_ICE_MICE].values_old_fld.d();
    auto vold_s = ice->data[FESOM_ICE_MSNOW].values_old_fld.d();
    auto ustarv = ice->thermo.ustar_fld.d();
    auto tskin  = ice->thermo.t_skin_fld.d();
    auto thdgr  = ice->thermo.thdgr_fld.d();
    auto thdgrsn = ice->thermo.thdgrsn_fld.d();
    auto thdgra  = ice->thermo.thdgra_fld.d();
    auto thdgr_old = ice->thermo.thdgr_old_fld.d();
    auto flxfw  = ice->flx_fw_fld.d();   auto flxh = ice->flx_h_fld.d();
    auto sw     = jra->shortwave_fld.d(); auto lw  = jra->longwave_fld.d();
    auto tair   = jra->Tair_fld.d();      auto qair = jra->shum_fld.d();
    auto prain  = jra->prec_rain_fld.d(); auto psnow = jra->prec_snow_fld.d();
    auto uw     = jra->u_wind_fld.d();    auto vw  = jra->v_wind_fld.d();
    auto runoff = forcing->runoff_fld.d();
    auto chatm  = forcing->Ch_atm_oce_fld.d(); auto ceatm = forcing->Ce_atm_oce_fld.d();
    /* M6.3 (zstar): the three flux OUTPUTS therm_ice produces and the port used to drop. */
    auto rsf_o  = forcing->real_salt_flux_fld.d();
    auto evap_o = forcing->evaporation_fld.d();
    auto subli_o= forcing->ice_sublimation_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto geo    = mesh->geo_coord_nod2D_fld.d();

    /* Loop 1: friction velocity over OWNED, then the ustar halo (Loop 2 reads ustar at halo n). */
    Kokkos::parallel_for("ice_therm_ustar", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) { ustarv(n) = 0.0; return; }
            const real_t du = uice(n) - sru(n);
            const real_t dv = vice(n) - srv(n);
            ustarv(n) = Kokkos::sqrt((du*du + dv*dv) * cd);
        });
    therm_halo_nod2D(ice->thermo.ustar_fld, partit);

    /* Loop 2: per-node column physics over [0,N) (halo included — the C bound). Race-free map. */
    Kokkos::parallel_for("ice_therm_col", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) return;                      /* cavity skip */
            real_t h    = m_ice(n);
            real_t hsn  = m_snw(n);
            real_t A    = a_ice(n);
            const real_t fsh  = sw(n);
            const real_t flo  = lw(n);
            const real_t Ta   = tair(n);
            const real_t qa   = qair(n);
            const real_t rain = prain(n);
            const real_t snow = psnow(n);
            const real_t runo = runoff(n);
            const real_t ug   = Kokkos::sqrt(uw(n)*uw(n) + vw(n)*vw(n));
            const real_t ustar = ustarv(n);
            const real_t T_oc  = srt(n);
            const real_t S_oc  = srs(n);
            const real_t rsss  = ref_sss_local ? S_oc : rsss_default;
            real_t t           = tskin(n);
            const real_t ch    = chatm(n);
            const real_t ce    = ceatm(n);
            const real_t lid_clo = (geo(2*n + 1) > 0.0) ? h0 : h0_s;

            real_t fw=0, fwice=0, fwsnw=0, ehf=0, evap=0, rsf=0, subli=0;
            real_t ithdgr=0, ithdgrsn=0, ithdgra=0, iflice=0;
            therm_ice_kk(thc, uvs, &h, &hsn, &A, fsh, flo, Ta, qa, rain, snow, runo, rsss,
                         ug, ustar, T_oc, S_oc, h_ml, &t, ice_dt, ch, ce, ch_i, ce_i,
                         &fw, &fwice, &fwsnw, &ehf, &evap, &rsf,
                         &ithdgr, &ithdgrsn, &ithdgra, &iflice, lid_clo, &subli);

            /* backup old (Fortran lines 311-314) */
            vold_m(n) = m_ice(n); vold_s(n) = m_snw(n); vold_a(n) = a_ice(n);
            thdgr_old(n) = thdgr(n);
            /* new values */
            m_ice(n) = h; m_snw(n) = hsn; a_ice(n) = A;
            tskin(n) = t; flxfw(n) = fw; flxh(n) = ehf;
            thdgr(n) = ithdgr; thdgrsn(n) = ithdgrsn; thdgra(n) = ithdgra;
            /* M6.3 (zstar): the three flux stores the port used to throw away. rsf's VALUE is
             * use_virt_salt-branched INSIDE therm_ice (a hard 0 under linfs), so storing it
             * unconditionally is both safe and faithful — invisible to the linfs path. */
            rsf_o(n) = rsf; evap_o(n) = evap; subli_o(n) = subli;
        });

    ice->data[FESOM_ICE_AICE].values_fld.modify_device();
    ice->data[FESOM_ICE_MICE].values_fld.modify_device();
    ice->data[FESOM_ICE_MSNOW].values_fld.modify_device();
    ice->data[FESOM_ICE_AICE].values_old_fld.modify_device();
    ice->data[FESOM_ICE_MICE].values_old_fld.modify_device();
    ice->data[FESOM_ICE_MSNOW].values_old_fld.modify_device();
    ice->thermo.t_skin_fld.modify_device();
    ice->flx_fw_fld.modify_device(); ice->flx_h_fld.modify_device();
    ice->thermo.thdgr_fld.modify_device();   ice->thermo.thdgrsn_fld.modify_device();
    ice->thermo.thdgra_fld.modify_device();  ice->thermo.thdgr_old_fld.modify_device();
    /* M6.3 (zstar): the three new flux outputs. Consumed on the DEVICE by the S surface BC
     * (real_salt_flux) and, under zstar, by the freshwater balancing assembly in oce_fluxes
     * (evaporation / ice_sublimation) — so they stay device-resident, no sync_host here. */
    forcing->real_salt_flux_fld.modify_device();
    forcing->evaporation_fld.modify_device();
    forcing->ice_sublimation_fld.modify_device();
}

/* FESOM_KK_VERIFY=icethermo — thermo read-modify-writes m_ice/m_snow/a_ice + t_skin + thdgr
 * (read as inputs, then overwritten). L26 capture-before those 5 (the driver snapshots PRE-thermo);
 * here snapshot the KK result of the 9 consumed outputs (the 3 values + t_skin + flx_h/flx_fw +
 * thdgr/thdgrsn/thdgra), restore the 5 inputs, run the C twin (recomputes ustar from intact
 * uice/srfoce + the column physics; its internal ustar halo is collective → run on ALL ranks),
 * diff the 9, restore KK. The inputs jra/forcing/srfoce are intact (thermo doesn't modify them).
 * Race-free per-node → max|Δ|==0 on Serial AND OpenMP. ⚠️ Meaningful only with ACTIVE ice (CORE2). */
void fesom_ice_thermodynamics_verify(fesom_ice *ice, struct fesom_partit *partit,
                                     struct fesom_mesh *mesh, struct fesom_forcing *forcing,
                                     const struct fesom_jra55 *jra, const struct fesom_sss_runoff *sr,
                                     int step_n,
                                     const std::vector<real_t> &pre_m, const std::vector<real_t> &pre_s,
                                     const std::vector<real_t> &pre_a, const std::vector<real_t> &pre_t,
                                     const std::vector<real_t> &pre_thdgr)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    real_t *m  = ice->data[FESOM_ICE_MICE].values;
    real_t *s  = ice->data[FESOM_ICE_MSNOW].values;
    real_t *a  = ice->data[FESOM_ICE_AICE].values;
    real_t *tk = ice->thermo.t_skin;
    real_t *fh = ice->flx_h;
    real_t *ff = ice->flx_fw;
    real_t *g  = ice->thermo.thdgr;
    real_t *gs = ice->thermo.thdgrsn;
    real_t *ga = ice->thermo.thdgra;
    /* snapshot the 9 consumed KK outputs */
    std::vector<real_t> km(m,m+N), ks(s,s+N), ka(a,a+N), kt(tk,tk+N), kfh(fh,fh+N),
                        kff(ff,ff+N), kg(g,g+N), kgs(gs,gs+N), kga(ga,ga+N);
    /* restore the 5 RMW inputs to PRE-thermo (the C-twin reads these) */
    std::copy(pre_m.begin(),pre_m.end(),m); std::copy(pre_s.begin(),pre_s.end(),s);
    std::copy(pre_a.begin(),pre_a.end(),a); std::copy(pre_t.begin(),pre_t.end(),tk);
    std::copy(pre_thdgr.begin(),pre_thdgr.end(),g);
    fesom_ice_thermodynamics(ice, partit, mesh, forcing, jra, sr);   /* C twin */
    auto mx=[&](const std::vector<real_t>&kk,const real_t*c){ double d=0.0;
        for(int i=0;i<N;++i){double x=std::fabs((double)kk[i]-(double)c[i]); if(x>d)d=x;} return d; };
    double dm=mx(km,m), ds=mx(ks,s), da=mx(ka,a), dt=mx(kt,tk), dfh=mx(kfh,fh),
           dff=mx(kff,ff), dg=mx(kg,g), dgs=mx(kgs,gs), dga=mx(kga,ga);
    /* restore the 9 KK outputs (production state) */
    std::copy(km.begin(),km.end(),m); std::copy(ks.begin(),ks.end(),s); std::copy(ka.begin(),ka.end(),a);
    std::copy(kt.begin(),kt.end(),tk); std::copy(kfh.begin(),kfh.end(),fh); std::copy(kff.begin(),kff.end(),ff);
    std::copy(kg.begin(),kg.end(),g); std::copy(kgs.begin(),kgs.end(),gs); std::copy(kga.begin(),kga.end(),ga);
    double dmax=dm; for(double x:{ds,da,dt,dfh,dff,dg,dgs,dga}) if(x>dmax) dmax=x;
    const std::string be = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=icethermo] step %d backend=%s  max|Δ|: m_ice=%.3e a_ice=%.3e m_snow=%.3e "
                "t_skin=%.3e flx_h=%.3e flx_fw=%.3e thdgr=%.3e\n",
                step_n, be.c_str(), dm, da, ds, dt, dfh, dff, dg);
    std::fflush(stdout);
    if (be == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=icethermo] FAIL step %d: ice thermo Serial must be "
                             "bit-identical (max|Δ|=%.3e)\n", step_n, dmax); std::abort();
    }
}
