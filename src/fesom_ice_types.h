#ifndef FESOM_ICE_TYPES_H
#define FESOM_ICE_TYPES_H

#include "fesom_types.h"
#include "fesom_field.hpp"   // M1.4: DualView-backed storage for the persistent sea-ice arrays

/*
 * M1.4 · DualView data layer (sea ice). Every persistent pointer array in the four
 * structs below is OWNED by a matching fesom::Field/IntField member; the legacy raw
 * pointer is kept as a NON-OWNING alias re-pointed at field.h() right after
 * field.alloc() in fesom_ice_init (and fesom_ice_mass_matrix_fill for fct_massmatrix)
 * — the same pattern proven on 28 mesh (M1.2, D12) + 37 dyn/aux/tracers (M1.3, D15)
 * arrays. These structs are EMBEDDED BY VALUE in the stack-allocated `fesom_ice ice`
 * (fesom_main.cpp:361), so default-construction runs every nested Field ctor and
 * `*ice = fesom_ice{}` resets/releases them all (D13). The arrays are written each
 * step through the raw alias by in-place value writes/memset — never re-pointed by a
 * buffer swap (audited: all time-history updates such as values_old=values are
 * element-wise copies) — so the aliases stay valid for the whole run. LayoutRight +
 * host mirror == the C flat layout → legacy ice->X[...] and the FESOM_* macros keep
 * working unchanged → Serial stays bit-identical. No device sync at M1 (nothing reads
 * these on device yet; the EVP/thermo/FCT kernels move to device in M4.3, D15).
 *
 * Mirror of Fortran T_ICE / T_ICE_DATA / T_ICE_WORK / T_ICE_THERMO from
 * MOD_ICE.F90. Scope cut for the C port (per docs/plans/20260425-sea-ice-port.md):
 *
 *   - whichEVP = 0 only (standard EVP). uice_aux/vice_aux/alpha_evp_array/
 *     beta_evp_array NOT mirrored — Fortran allocates them only when whichEVP != 0.
 *   - non-ICEPACK build. No __oasis/__oifs/__yac/__ifsinterface fields.
 *   - num_itracers = 3 (a_ice, m_ice, m_snow). No ice_temp tracer (oifs only).
 *   - No icebergs (uice_ib/vice_ib).
 *   - No melt-pond fields beyond apnd/hpnd/ipnd defaults (use_meltponds = false).
 *   - No CMIP6 dyngr/dyngrsn/dyngra diagnostics (output-only, no physics impact).
 *   - No coupled-atmosphere atmcoupl substruct.
 */

/* Tracer indices into ice->data[]. Mirrors Fortran ice%data(1..3). */
#define FESOM_NUM_ICE_TRACERS 3
#define FESOM_ICE_AICE  0     /* ice area fraction      a_ice  */
#define FESOM_ICE_MICE  1     /* ice volume per area    m_ice  */
#define FESOM_ICE_MSNOW 2     /* snow volume per area   m_snow */

/* Mirror of T_ICE_DATA (MOD_ICE.F90:14-23). All sized [myDim+eDim] (nodes). */
typedef struct fesom_ice_data {
    real_t *values;          /* current value          */
    real_t *values_old;      /* previous timestep      */
    real_t *values_rhs;      /* high-order FCT RHS     */
    real_t *values_div_rhs;  /* divergence-form RHS    */
    real_t *dvalues;         /* increment / scratch    */
    real_t *valuesl;         /* low-order FCT solution */
    int     id;              /* 0..2, set at init      */

    /* M1.4: Field owners; the raw ptrs above are non-owning aliases = field.h() (D12). */
    fesom::Field values_fld, values_old_fld, values_rhs_fld;
    fesom::Field values_div_rhs_fld, dvalues_fld, valuesl_fld;
} fesom_ice_data;

/*
 * Mirror of T_ICE_WORK (MOD_ICE.F90:28-41). Mix of node and element arrays.
 *
 *   fct_tmax/tmin/plus/minus  [myDim+eDim]                  (nodes)
 *   fct_fluxes                [elem_size * 3]               (3 edge fluxes per element)
 *   fct_massmatrix            [sum(nn_num(1:myDim_nod2D))]  (CSR-style; sized at alloc)
 *   sigma11/12/22, eps11/12/22, ice_strength
 *                             [elem_size]                    (elements)
 *   inv_areamass, inv_mass    [myDim+eDim]                  (nodes)
 */
typedef struct fesom_ice_work {
    real_t *fct_tmax;
    real_t *fct_tmin;
    real_t *fct_plus;
    real_t *fct_minus;
    real_t *fct_fluxes;
    real_t *fct_massmatrix;
    real_t *sigma11, *sigma12, *sigma22;
    real_t *eps11,   *eps12,   *eps22;
    real_t *ice_strength;
    real_t *inv_areamass;
    real_t *inv_mass;

    /* --- mEVP-only per-call scratch (M6.2; the Fortran's automatic arrays in EVPdynamics_m,
     * heap-persistent here exactly as in the C). ALLOCATED ONLY WHEN whichEVP==1.
     *   mevp_inv_thickness [myDim_nod2D]   1/max(ice thickness, 9.0)   (:635 limiter)
     *   mevp_mass          [myDim_nod2D]   M / ((1+M²)·area)  — the verbatim regularisation
     *   mevp_ice_nod       [myDim_nod2D]   node mask, a_ice >= 0.01
     *   mevp_pressure_fac  [myDim_elem2D]  det2·pstar·msum·exp(-c_pressure·(1-asum))
     *                                      ⚠️ NO 0.5 here — the 0.5 lives in the sigma11/22
     *                                      updates only (mEVP trap 2). Do NOT normalise this
     *                                      against std-EVP's ice_strength.
     *   mevp_ice_el        [myDim_elem2D]  element mask, mean(m_ice) > 0.01  (trap 4)
     * These are NOT shared with the std-EVP path (which uses ice_strength / inv_areamass /
     * inv_mass with different definitions) — keeping them separate is what stops the two
     * rheologies' asymmetries from being "tidied" into each other. */
    real_t *mevp_inv_thickness;
    real_t *mevp_mass;
    int    *mevp_ice_nod;
    real_t *mevp_pressure_fac;
    int    *mevp_ice_el;

    /* --- M9 (FESOM_SPEED_MEVPDIV): the divergence form's carried state.
     *   mevp_Ru/Rv       [myDim+eDim]  R = div(sigma) at NODES, carried across subcycles AND
     *                                  across ocean steps exactly as sigma11/12/22 are in the
     *                                  classic form. Stored UNSCALED (pure divergence; `mass` is
     *                                  applied at use) — deliberately unlike Sergey's F90, which
     *                                  folds inv_mass into R and thereby makes the carried state
     *                                  remember a PREVIOUS step's mass scaling.
     *                                  Sized myDim+eDim: only owned entries are read at K=1, but
     *                                  the halo room is free and avoids a resize when the wide
     *                                  halo lands (it then grows by fesom_evpwide_next()).
     *   mevp_Rchk_u/v    [myDim+eDim]  DIAGNOSTIC ONLY — allocated only when
     *                                  FESOM_MEVPDIV_SELFCHECK is set: div(sigma) reassembled
     *                                  from the carried sigma, so R can be compared against it.
     * mEVP-only, and here rather than shared for the same reason as the block above: so the two
     * rheologies' state cannot be tidied into each other. */
    real_t *mevp_Ru,     *mevp_Rv;
    real_t *mevp_Rchk_u, *mevp_Rchk_v;

    /* M1.4: Field owners; the raw ptrs above are non-owning aliases = field.h() (D12).
     * fct_massmatrix is alloc'd lazily in fesom_ice_mass_matrix_fill (sized stiff->nnz),
     * the others in fesom_ice_init. */
    fesom::Field fct_tmax_fld, fct_tmin_fld, fct_plus_fld, fct_minus_fld;
    fesom::Field fct_fluxes_fld, fct_massmatrix_fld;
    fesom::Field sigma11_fld, sigma12_fld, sigma22_fld;
    fesom::Field eps11_fld, eps12_fld, eps22_fld;
    fesom::Field ice_strength_fld, inv_areamass_fld, inv_mass_fld;
    fesom::Field mevp_inv_thickness_fld, mevp_mass_fld, mevp_pressure_fac_fld;
    fesom::IntField mevp_ice_nod_fld, mevp_ice_el_fld;
    fesom::Field mevp_Ru_fld, mevp_Rv_fld;               /* M9 divergence form */
    fesom::Field mevp_Rchk_u_fld, mevp_Rchk_v_fld;       /* M9 selfcheck only */
} fesom_ice_work;

/*
 * Mirror of T_ICE_THERMO (MOD_ICE.F90:46-101). Per-node arrays + scalar
 * physical constants and namelist parameters. Defaults match the Fortran
 * type defaults (T_ICE_THERMO lines 53-97), which match work_core/namelist.ice.
 *
 * Skipped:
 *   - dyngr/dyngrsn/dyngra (CMIP6 output-only, no physics impact — see Out-of-scope)
 *   - hpdf[15], h_cutoff, new_iclasses (active only when use_meltponds or new_iclasses=true)
 */
typedef struct fesom_ice_thermo {
    /* per-node arrays */
    real_t *t_skin;
    real_t *thdgr, *thdgrsn, *thdgra;
    real_t *thdgr_old;
    real_t *ustar;
    real_t *apnd, *hpnd, *ipnd;          /* meltpond fields, zeroed; use_meltponds=false */

    /* M1.4: Field owners for the per-node arrays above; raw ptrs = field.h() aliases (D12). */
    fesom::Field t_skin_fld, thdgr_fld, thdgrsn_fld, thdgra_fld, thdgr_old_fld;
    fesom::Field ustar_fld, apnd_fld, hpnd_fld, ipnd_fld;

    /* density and inverse (Fortran T_ICE_THERMO:53-57) */
    real_t rhoair,  inv_rhoair;
    real_t rhowat,  inv_rhowat;
    real_t rhofwt,  inv_rhofwt;
    real_t rhoice,  inv_rhoice;
    real_t rhosno,  inv_rhosno;

    /* specific heats J/(kg*K) */
    real_t cpair, cpice, cpsno;

    /* derived (computed in init): cc = rhowat*4190, cl = rhoice*3.34e5 */
    real_t cc, cl;

    /* latent heat (J/kg) */
    real_t clhw;          /* water -> water vapor */
    real_t clhi;          /* sea ice -> water vapor */

    real_t tmelt;         /* 273.15 K */
    real_t boltzmann;     /* sigma * emissivity */

    int    iclasses;      /* number of ice thickness gradations (default 7) */
    real_t hmin;          /* cutoff ice thickness, m */
    real_t armin;         /* minimum ice concentration */

    /* namelist /ice_therm/ */
    real_t con, consn;    /* thermal conductivities ice, snow (W/m/K) */
    real_t Sice;          /* ice salinity (ppt) */
    real_t h0, h0_s;      /* lead closing N hemi / S hemi (m) */
    real_t emiss_ice, emiss_wat;
    real_t albsn, albsnm; /* snow albedo: frozen, melting */
    real_t albi,  albim;  /* ice  albedo: frozen, melting */
    real_t albw;          /* open water albedo (LY2004) */
    real_t h_ml;          /* upper-layer thickness for heat available */

    /* additional namelist params */
    int    snowdist;
    int    open_water_albedo;   /* 0=standard 1=taylor 2=briegleb */
    real_t c_melt;              /* concentration eq. coefficient on melt */

    int    use_meltponds;       /* 0 in CORE2 */
} fesom_ice_thermo;

/*
 * Mirror of T_ICE (MOD_ICE.F90:128-222) — the top-level sea-ice state struct.
 *
 * Sizing convention (per fesom_dyn.c):
 *   node arrays:    myDim_nod2D  + eDim_nod2D
 *   element arrays: myDim_elem2D + eDim_elem2D + eXDim_elem2D
 */
typedef struct fesom_ice {
    /* tracers — a_ice, m_ice, m_snow (indices FESOM_ICE_AICE/MICE/MSNOW) */
    fesom_ice_data data[FESOM_NUM_ICE_TRACERS];

    /* zonal & meridional ice velocity (nodes) */
    real_t *uice, *uice_rhs, *uice_old;
    real_t *vice, *vice_rhs, *vice_old;

    /* surface stresses atm<->ice and oce<->ice (nodes) */
    real_t *stress_atmice_x, *stress_iceoce_x;
    real_t *stress_atmice_y, *stress_iceoce_y;

    /* surface ocean state seen by ice (populated by ocean2ice; nodes) */
    real_t *srfoce_temp, *srfoce_salt, *srfoce_ssh;
    real_t *srfoce_u,    *srfoce_v;

    /* fluxes from thermodynamics (nodes)
       flx_fw is fresh_wa_flux (m/s, includes runoff per non-ICEPACK contract)
       flx_h  is net_heat_flux (W/m^2)                                          */
    real_t *flx_fw;
    real_t *flx_h;

    /* diagnostic ice/snow thicknesses h = m / max(a, 1e-3) (nodes) */
    real_t *h_ice, *h_snow;

    /* M1.4: Field owners for the top-level node arrays above; raw ptrs = field.h() (D12). */
    fesom::Field uice_fld, uice_rhs_fld, uice_old_fld;
    fesom::Field vice_fld, vice_rhs_fld, vice_old_fld;
    fesom::Field stress_atmice_x_fld, stress_iceoce_x_fld;
    fesom::Field stress_atmice_y_fld, stress_iceoce_y_fld;
    fesom::Field srfoce_temp_fld, srfoce_salt_fld, srfoce_ssh_fld;
    fesom::Field srfoce_u_fld, srfoce_v_fld;
    fesom::Field flx_fw_fld, flx_h_fld;
    fesom::Field h_ice_fld, h_snow_fld;

    /* embedded substructs */
    fesom_ice_work   work;
    fesom_ice_thermo thermo;

    /* --- RHEOLOGY scalar parameters (defaults from T_ICE:187-202 = namelist.ice) --- */
    real_t pstar;            /* 30000.0 N/m^2 */
    real_t ellipse;          /* 2.0           */
    real_t c_pressure;       /* 20.0          */
    real_t delta_min;        /* 1.0e-11 s^-1  */
    real_t Clim_evp;         /* 615 kg/m^2    */
    real_t zeta_min;         /* 4.0e+8 kg/s   */

    int    evp_rheol_steps;  /* 120  EVP subcycles per ice timestep */
    real_t ice_gamma_fct;    /* 0.5 smoothing in ice FCT advection (CORE2 namelist) */
    real_t ice_diff;         /* 10.0 stabilising diffusion          */
    real_t theta_io;         /* 0.0  ice-ocean rotation angle       */
    real_t cd_oce_ice;       /* 5.5e-3 ocean-ice drag coefficient   */
    int    ice_free_slip;    /* 0     */

    /* --- EVP flavour (M6.2). FESOM_WHICH_EVP: unset|0 -> standard EVP (default),
     *     1 -> mEVP (fesom_ice_maevp.cpp). Anything else aborts: aEVP (2) is not ported. --- */
    int    whichEVP;         /* 0 = std EVP (default), 1 = mEVP */
    real_t alpha_evp;        /* 250.0 — mEVP stability constant. Set UNCONDITIONALLY, as the C
                              * does (fesom_ice.c:91): it is a MOD_ICE.F90 module default and the
                              * reference namelist.ice repeats it, so there is no over-default trap. */
    real_t beta_evp;         /* 250.0 — ditto (fesom_ice.c:92) */

    /* mEVP auxiliary velocity (ice_maEVP.F90 u_ice_aux/v_ice_aux), [myDim+eDim].
     * ALLOCATED ONLY WHEN whichEVP==1 — the Fortran allocates them only for whichEVP != 0, and
     * the C mirrors that. The mEVP iteration writes these and copies them back into uice/vice at
     * the very end (trap 8: the final copy spans myDim+eDim with NO extra exchange, because the
     * aux halo is already current from the last substep's exchange). */
    real_t *uice_aux;        /* [myDim+eDim] */
    real_t *vice_aux;        /* [myDim+eDim] */
    fesom::Field uice_aux_fld, vice_aux_fld;

    /* --- timestep --- */
    int    ice_ave_steps;    /* 1 — ice timestep = ice_ave_steps * ocean step */
    real_t ice_dt;           /* set in fesom_ice_setup from ocean dt          */
    real_t Tevp_inv;         /* set in fesom_ice_setup = evp_rheol_steps/ice_dt */

    /* misc state */
    int    ice_steps_since_upd;  /* 0 */
    int    ice_update;           /* 1 */
} fesom_ice;

#endif /* FESOM_ICE_TYPES_H */
