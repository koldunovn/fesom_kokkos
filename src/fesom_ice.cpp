#include "fesom_ice.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_ice_coupling.h"
#include "fesom_ice_evp.h"
#include "fesom_ice_fct.h"
#include "fesom_ice_thermo.h"
#include "fesom_halo.h"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_tracers.h"

#include <Kokkos_Core.hpp>   // M4.3a: device h_ice/h_snow diag kernel + ocean2ice/cut_off islands
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>            // M4.3a: FESOM_KK_VERIFY backend name
#include <algorithm>         // M4.3a: verify snapshot/restore copies
#include <vector>

#ifdef FESOM_KK_SYNCCHECK
/* M1.5 plumbing proof (docs/SYNC_MAP.md §4). Bounce a representative set of the sea-ice step's
 * evolving-state Fields host->device->host after the host EVP/FCT/thermo kernels have written them
 * via the raw alias. modify_host() first (the writes bypass the DualView flags, L14); modify_device()
 * models the M4.3 device ice kernel. No-op on Serial/OpenMP, bitwise-exact deep_copy on CUDA -> the
 * run stays bit-identical. Compiled out when the macro is off. */
#define FESOM_KK_BOUNCE(f) do { (f).modify_host(); (f).sync_device(); \
                                (f).modify_device(); (f).sync_host(); } while (0)
static void ice_synccheck_roundtrip(fesom_ice *ice)
{
    FESOM_KK_BOUNCE(ice->uice_fld);
    FESOM_KK_BOUNCE(ice->vice_fld);
    FESOM_KK_BOUNCE(ice->h_ice_fld);
    FESOM_KK_BOUNCE(ice->h_snow_fld);
    FESOM_KK_BOUNCE(ice->stress_iceoce_x_fld);
    FESOM_KK_BOUNCE(ice->stress_iceoce_y_fld);
    for (int k = 0; k < FESOM_NUM_ICE_TRACERS; ++k)
        FESOM_KK_BOUNCE(ice->data[k].values_fld);
}
#undef FESOM_KK_BOUNCE
#endif

/*
 * fesom_ice_init — mirror of Fortran ice_init (MOD_ICE.F90:572).
 *
 * Hardcodes the CORE2 namelist.ice values (work_core/namelist.ice) as
 * defaults rather than parsing a namelist file (consistent with
 * FRESH_START.md §1: namelist parsing is not in scope until later).
 *
 * Allocations match the Fortran field-by-field layout (lines 712-905):
 *   node_size = myDim_nod2D + eDim_nod2D
 *   elem_size = myDim_elem2D + eDim_elem2D + eXDim_elem2D
 *
 * Skipped (per docs/plans/20260425-sea-ice-port.md Out-of-scope):
 *   - whichEVP != 0 aux arrays (uice_aux/vice_aux/alpha_evp/beta_evp).
 *   - data[].values_old etc. for tracer index 4 (oifs/coupled only).
 *   - dyngr/dyngrsn/dyngra (CMIP6 output-only).
 *   - atmcoupl substruct.
 *
 * Phase A: fct_massmatrix left NULL — alloc'd lazily by fesom_ice_mass_matrix_fill.
 */
void fesom_ice_init(fesom_ice           *ice,
                    struct fesom_partit *partit,
                    struct fesom_mesh   *mesh)
{
    (void)partit; /* sizes come from mesh; partit kept in signature to mirror Fortran */

    // The struct (and its embedded data[]/work/thermo) holds fesom::Field members (DualView,
    // non-trivial): a raw memset is UB (L13). Value-initialise instead — zeros every POD and
    // leaves each nested Field an empty DualView (D13).
    *ice = fesom_ice{};

    /* --- scalar defaults: rheology + dynamics (T_ICE:187-204; namelist.ice) --- */
    ice->pstar           = 30000.0;
    ice->ellipse         = 2.0;
    ice->c_pressure      = 20.0;
    ice->delta_min       = 1.0e-11;
    ice->Clim_evp        = 615.0;
    ice->zeta_min        = 4.0e+8;
    ice->evp_rheol_steps = 120;
    ice->ice_gamma_fct   = 0.5;   /* CORE2 reference NAMELIST value (work_core /
                                   * work_kpp_dump namelist.ice:44); NOT the 0.25 module
                                   * default (MOD_ICE.F90:194). feedback_namelist_over_codedefault. */
    ice->ice_diff        = 10.0;
    ice->theta_io        = 0.0;
    ice->cd_oce_ice      = 5.5e-3;
    ice->ice_free_slip   = 0;
    ice->whichEVP        = 0;       /* hard 0; dispatcher (Phase D4) aborts otherwise */
    ice->ice_ave_steps   = 1;
    ice->ice_dt          = 0.0;     /* set in fesom_ice_setup */
    ice->Tevp_inv        = 0.0;     /* set in fesom_ice_setup */
    ice->ice_steps_since_upd = 0;
    ice->ice_update      = 1;

    /* --- thermo scalar defaults (T_ICE_THERMO:53-97) --- */
    fesom_ice_thermo *th = &ice->thermo;
    th->rhoair  = 1.3;     th->inv_rhoair  = 1.0/1.3;
    th->rhowat  = 1025.0;  th->inv_rhowat  = 1.0/1025.0;
    th->rhofwt  = 1000.0;  th->inv_rhofwt  = 1.0/1000.0;
    th->rhoice  = 910.0;   th->inv_rhoice  = 1.0/910.0;
    th->rhosno  = 290.0;   th->inv_rhosno  = 1.0/290.0;
    th->cpair   = 1005.0;
    th->cpice   = 2106.0;
    th->cpsno   = 2090.0;
    th->cc      = th->rhowat * 4190.0;     /* matches Fortran ice_init line 708 */
    th->cl      = th->rhoice * 3.34e5;     /* matches Fortran ice_init line 709 */
    th->clhw    = 2.501e6;
    th->clhi    = 2.835e6;
    th->tmelt   = 273.15;
    th->boltzmann = 5.67e-8;
    th->iclasses  = 7;
    th->hmin    = 0.01;
    th->armin   = 0.01;
    th->con     = 2.1656;
    th->consn   = 0.31;
    th->Sice    = 4.0;
    th->h0      = 0.5;
    th->h0_s    = 0.5;
    th->emiss_ice = 0.97;
    th->emiss_wat = 0.97;
    th->albsn   = 0.81;
    th->albsnm  = 0.77;
    th->albi    = 0.70;
    th->albim   = 0.68;
    th->albw    = 0.1;     /* CORE2 namelist.ice albw=0.1 (was 0.066=LY2004; the reference
                              overrides it). Keep == fesom_bulk.c BULK_ALBW. */
    th->h_ml    = 2.5;
    th->snowdist = 1;
    th->open_water_albedo = 0;
    th->c_melt  = 0.5;
    th->use_meltponds = 0;

    /* --- sizes --- */
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int E = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    size_t n  = (size_t)N;
    size_t e  = (size_t)E;
    size_t e3 = e * 3;

    // M1.4: each Field owns its storage; the raw pointer is a non-owning alias = field.h()
    // (D12). .alloc(label, count) takes the count in ELEMENTS (== the old xcalloc count) and
    // zero-inits like calloc → Serial bit-identical. Halo-sized node (n) / element (e, e3) extents
    // kept verbatim.
    /* --- velocities (T_ICE:132-133, ice_init:719-740) --- */
    ice->uice_fld.alloc("ice.uice", n);                       ice->uice            = ice->uice_fld.h();
    ice->uice_rhs_fld.alloc("ice.uice_rhs", n);               ice->uice_rhs        = ice->uice_rhs_fld.h();
    ice->uice_old_fld.alloc("ice.uice_old", n);               ice->uice_old        = ice->uice_old_fld.h();
    ice->vice_fld.alloc("ice.vice", n);                       ice->vice            = ice->vice_fld.h();
    ice->vice_rhs_fld.alloc("ice.vice_rhs", n);               ice->vice_rhs        = ice->vice_rhs_fld.h();
    ice->vice_old_fld.alloc("ice.vice_old", n);               ice->vice_old        = ice->vice_old_fld.h();

    /* --- stresses (T_ICE:136-137) --- */
    ice->stress_atmice_x_fld.alloc("ice.stress_atmice_x", n); ice->stress_atmice_x = ice->stress_atmice_x_fld.h();
    ice->stress_iceoce_x_fld.alloc("ice.stress_iceoce_x", n); ice->stress_iceoce_x = ice->stress_iceoce_x_fld.h();
    ice->stress_atmice_y_fld.alloc("ice.stress_atmice_y", n); ice->stress_atmice_y = ice->stress_atmice_y_fld.h();
    ice->stress_iceoce_y_fld.alloc("ice.stress_iceoce_y", n); ice->stress_iceoce_y = ice->stress_iceoce_y_fld.h();

    /* --- diagnostic h_ice/h_snow (T_ICE:150) --- */
    ice->h_ice_fld.alloc("ice.h_ice", n);                     ice->h_ice           = ice->h_ice_fld.h();
    ice->h_snow_fld.alloc("ice.h_snow", n);                   ice->h_snow          = ice->h_snow_fld.h();

    /* whichEVP != 0 aux fields intentionally NOT allocated (see header). */

    /* --- surface ocean state (T_ICE:140-142, ice_init:758-767) --- */
    ice->srfoce_u_fld.alloc("ice.srfoce_u", n);               ice->srfoce_u    = ice->srfoce_u_fld.h();
    ice->srfoce_v_fld.alloc("ice.srfoce_v", n);               ice->srfoce_v    = ice->srfoce_v_fld.h();
    ice->srfoce_temp_fld.alloc("ice.srfoce_temp", n);         ice->srfoce_temp = ice->srfoce_temp_fld.h();
    ice->srfoce_salt_fld.alloc("ice.srfoce_salt", n);         ice->srfoce_salt = ice->srfoce_salt_fld.h();
    ice->srfoce_ssh_fld.alloc("ice.srfoce_ssh", n);           ice->srfoce_ssh  = ice->srfoce_ssh_fld.h();

    /* --- fluxes from thermodynamics (T_ICE:145, ice_init:769-772) --- */
    ice->flx_fw_fld.alloc("ice.flx_fw", n);                   ice->flx_fw      = ice->flx_fw_fld.h();
    ice->flx_h_fld.alloc("ice.flx_h", n);                     ice->flx_h       = ice->flx_h_fld.h();

    /* --- ice tracers data[0..2] (ice_init:778-794) --- */
    for (int k = 0; k < FESOM_NUM_ICE_TRACERS; ++k) {
        fesom_ice_data *d = &ice->data[k];
        d->id = k;
        char lbl[48];
        snprintf(lbl, sizeof lbl, "ice.data%d.values", k);         d->values_fld.alloc(lbl, n);          d->values         = d->values_fld.h();
        snprintf(lbl, sizeof lbl, "ice.data%d.values_old", k);     d->values_old_fld.alloc(lbl, n);      d->values_old     = d->values_old_fld.h();
        snprintf(lbl, sizeof lbl, "ice.data%d.values_rhs", k);     d->values_rhs_fld.alloc(lbl, n);      d->values_rhs     = d->values_rhs_fld.h();
        snprintf(lbl, sizeof lbl, "ice.data%d.values_div_rhs", k); d->values_div_rhs_fld.alloc(lbl, n);  d->values_div_rhs = d->values_div_rhs_fld.h();
        snprintf(lbl, sizeof lbl, "ice.data%d.dvalues", k);        d->dvalues_fld.alloc(lbl, n);         d->dvalues        = d->dvalues_fld.h();
        snprintf(lbl, sizeof lbl, "ice.data%d.valuesl", k);        d->valuesl_fld.alloc(lbl, n);         d->valuesl        = d->valuesl_fld.h();
    }

    /* --- work arrays (T_ICE_WORK; ice_init:798-830) --- */
    fesom_ice_work *w = &ice->work;
    w->fct_tmax_fld.alloc("ice.work.fct_tmax", n);            w->fct_tmax        = w->fct_tmax_fld.h();
    w->fct_tmin_fld.alloc("ice.work.fct_tmin", n);            w->fct_tmin        = w->fct_tmin_fld.h();
    w->fct_plus_fld.alloc("ice.work.fct_plus", n);            w->fct_plus        = w->fct_plus_fld.h();
    w->fct_minus_fld.alloc("ice.work.fct_minus", n);          w->fct_minus       = w->fct_minus_fld.h();
    w->fct_fluxes_fld.alloc("ice.work.fct_fluxes", e3);       w->fct_fluxes      = w->fct_fluxes_fld.h();  /* (elem_size,3) → flat e*3 */
    w->fct_massmatrix  = NULL;                   /* Phase E1: fesom_ice_mass_matrix_fill allocs the Field */
    w->sigma11_fld.alloc("ice.work.sigma11", e);              w->sigma11         = w->sigma11_fld.h();
    w->sigma12_fld.alloc("ice.work.sigma12", e);              w->sigma12         = w->sigma12_fld.h();
    w->sigma22_fld.alloc("ice.work.sigma22", e);              w->sigma22         = w->sigma22_fld.h();
    w->eps11_fld.alloc("ice.work.eps11", e);                  w->eps11           = w->eps11_fld.h();
    w->eps12_fld.alloc("ice.work.eps12", e);                  w->eps12           = w->eps12_fld.h();
    w->eps22_fld.alloc("ice.work.eps22", e);                  w->eps22           = w->eps22_fld.h();
    w->ice_strength_fld.alloc("ice.work.ice_strength", e);    w->ice_strength    = w->ice_strength_fld.h();
    w->inv_areamass_fld.alloc("ice.work.inv_areamass", n);    w->inv_areamass    = w->inv_areamass_fld.h();  /* populated by EVPdynamics each step */
    w->inv_mass_fld.alloc("ice.work.inv_mass", n);            w->inv_mass        = w->inv_mass_fld.h();      /* populated by EVPdynamics each step */

    /* --- thermo per-node arrays (ice_init:834-853) --- */
    th->ustar_fld.alloc("ice.thermo.ustar", n);               th->ustar          = th->ustar_fld.h();
    th->t_skin_fld.alloc("ice.thermo.t_skin", n);             th->t_skin         = th->t_skin_fld.h();
    th->thdgr_fld.alloc("ice.thermo.thdgr", n);               th->thdgr          = th->thdgr_fld.h();
    th->thdgrsn_fld.alloc("ice.thermo.thdgrsn", n);           th->thdgrsn        = th->thdgrsn_fld.h();
    th->thdgra_fld.alloc("ice.thermo.thdgra", n);             th->thdgra         = th->thdgra_fld.h();
    th->thdgr_old_fld.alloc("ice.thermo.thdgr_old", n);       th->thdgr_old      = th->thdgr_old_fld.h();
    th->apnd_fld.alloc("ice.thermo.apnd", n);                 th->apnd           = th->apnd_fld.h();
    th->hpnd_fld.alloc("ice.thermo.hpnd", n);                 th->hpnd           = th->hpnd_fld.h();
    th->ipnd_fld.alloc("ice.thermo.ipnd", n);                 th->ipnd           = th->ipnd_fld.h();

    /*
     * --- Mesh boundary mask (mirrors MOD_ICE.F90:889-895) ---
     * Fortran loops local edges, looks up global edge id in myList_edge2D and
     * compares to mesh%edge2D_in. C port uses the equivalent boundary marker
     * already in the mesh: edge_tri[edge*2 + 1] == -1 marks a boundary edge.
     * Both schemes produce the same set of boundary endpoints.
     */
    if (mesh->bc_index_nod2D == NULL) {
        // M1.2 Wave 3: Field-owned (released by fesom_mesh_free's *m = fesom_mesh{}); raw alias.
        // .alloc zero-inits like xcalloc; the loop below then sets the interior mask to 1.0.
        mesh->bc_index_nod2D_fld.alloc("bc_index_nod2D", n);
        mesh->bc_index_nod2D = mesh->bc_index_nod2D_fld.h();
        for (size_t i = 0; i < n; ++i) mesh->bc_index_nod2D[i] = 1.0;
        for (int ed = 0; ed < mesh->myDim_edge2D; ++ed) {
            if (mesh->edge_tri[ed * 2 + 1] >= 0) continue;   /* interior edge */
            int n1 = mesh->edges[ed * 2 + 0];
            int n2 = mesh->edges[ed * 2 + 1];
            mesh->bc_index_nod2D[n1] = 0.0;
            mesh->bc_index_nod2D[n2] = 0.0;
        }
    }
}

/*
 * fesom_ice_setup — mirror of Fortran ice_setup (ice_setup_step.F90:51-91).
 *
 * Fortran lines 74-78 set the three timing constants from the ocean dt.
 * After Phase E1 lands, the call to fesom_ice_mass_matrix_fill below
 * replaces the stub.
 *
 * Skipped vs. Fortran:
 *   - call ice_init(...)         (already done in fesom_ice_init)
 *   - call ice_mass_matrix_fill  (Phase E1)
 *   - call ice_initial_state     (covered by calloc-zero + snapshot;
 *                                 see plan Out-of-scope)
 */
void fesom_ice_setup(fesom_ice           *ice,
                     struct fesom_partit *partit,
                     struct fesom_mesh   *mesh)
{
    (void)partit; (void)mesh;
    /* Fortran: ice_dt = real(ice_ave_steps, WP) * dt */
    ice->ice_dt   = (real_t)ice->ice_ave_steps * fesom_phase1_dt;
    /* Fortran: Tevp_inv = 3.0 / ice_dt  (NOT evp_rheol_steps/ice_dt) */
    ice->Tevp_inv = 3.0 / ice->ice_dt;
    /* Fortran: Clim_evp = Clim_evp * (evp_rheol_steps/ice_dt)^2 / Tevp_inv */
    real_t r = (real_t)ice->evp_rheol_steps / ice->ice_dt;
    ice->Clim_evp = ice->Clim_evp * (r * r) / ice->Tevp_inv;

    /* TODO Phase E1: fesom_ice_mass_matrix_fill(ice, partit, mesh); */
}

/*
 * Mirror of Fortran ice_initial_state at ice_setup_step.F90:500-521
 * (cold-start branch only — `ini_ice_from_file=false`, the default).
 * Loop bound is myDim+eDim (matches Fortran line 503).
 */
void fesom_ice_initial_state(fesom_ice                  *ice,
                             const struct fesom_tracers *tracers,
                             struct fesom_partit        *partit,
                             struct fesom_mesh          *mesh)
{
    (void)partit;
    int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int nl = mesh->nl;
    real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;
    real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    real_t *m_snow = ice->data[FESOM_ICE_MSNOW].values;

    /* Fortran zeroes m_snow first then re-fills only ice-covered nodes — we
     * are coming straight from calloc so already zero. */
    for (int n = 0; n < N; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;   /* cavity skip */
        real_t sst = tracers->data[FESOM_TRACER_T].values[n * nl + 0];
        if (sst < 0.0) {
            real_t lat = mesh->geo_coord_nod2D[2*n + 1];
            if (lat > 0.0) {           /* Northern hemisphere */
                m_ice [n] = 1.0;
                m_snow[n] = 0.1;
            } else {                   /* Southern hemisphere */
                m_ice [n] = 2.0;
                m_snow[n] = 0.5;
            }
            a_ice[n] = 0.9;
            ice->uice[n] = 0.0;
            ice->vice[n] = 0.0;
        }
    }
}

/*
 * fesom_ice_step — mirror of Fortran ice_timestep (ice_setup_step.F90:96).
 *
 * Phase A: env-knob-gated stubs for the three subsystems, plus the post-step
 * h_ice/h_snow diagnostic. With all knobs off (default) this writes the
 * diagnostic only; with any knob set the corresponding subsystem is skipped
 * (which is a no-op in Phase A since all three subsystems are stubbed anyway).
 *
 * Env knobs are read once and cached (matches FESOM_NO_TRADV style elsewhere).
 */
static int s_ice_env_loaded = 0;
static int s_no_ice_dyn     = 0;
static int s_no_ice_adv     = 0;
static int s_no_ice_thermo  = 0;

static void load_ice_env_once(void)
{
    if (s_ice_env_loaded) return;
    s_no_ice_dyn    = (getenv("FESOM_NO_ICE_DYN")    != NULL);
    s_no_ice_adv    = (getenv("FESOM_NO_ICE_ADV")    != NULL);
    s_no_ice_thermo = (getenv("FESOM_NO_ICE_THERMO") != NULL);
    s_ice_env_loaded = 1;
}

/* M4.3a FESOM_KK_VERIFY=icemap gate (ocean2ice + cut_off + h_ice/h_snow diag). Cached. */
static int s_ice_verify_loaded = 0;
static int s_verify_icemap     = 0;
static void load_ice_verify_once(void)
{
    if (s_ice_verify_loaded) return;
    const char *e = getenv("FESOM_KK_VERIFY");
    s_verify_icemap = (e && strstr(e, "icemap")) ? 1 : 0;   /* collision-free token (L25) */
    s_ice_verify_loaded = 1;
}

/* ====================================================================== *
 *  M4.3a — h_ice/h_snow diagnostic (Fortran ice_setup_step.F90:319-330).  *
 *  h = m / max(a, 1e-3) over [0,N) (halo included). Full overwrite from    *
 *  a_ice/m_ice/m_snow → race-free map; the C twin is the verify oracle.    *
 * ====================================================================== */
static void fesom_ice_h_diag(fesom_ice *ice, struct fesom_mesh *mesh)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;
    const real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    const real_t *m_snow = ice->data[FESOM_ICE_MSNOW].values;
    for (int i = 0; i < N; ++i) {
        real_t denom = a_ice[i] > 1e-3 ? a_ice[i] : 1e-3;
        ice->h_ice [i] = m_ice [i] / denom;
        ice->h_snow[i] = m_snow[i] / denom;
    }
}

static void fesom_ice_h_diag_kk(fesom_ice *ice, struct fesom_mesh *mesh)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    auto a_ice  = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto m_ice  = ice->data[FESOM_ICE_MICE].values_fld.d();
    auto m_snow = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    auto h_ice  = ice->h_ice_fld.d();
    auto h_snow = ice->h_snow_fld.d();
    Kokkos::parallel_for("fesom_ice_h_diag", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int i) {
            real_t denom = a_ice(i) > 1e-3 ? a_ice(i) : 1e-3;
            h_ice(i)  = m_ice(i)  / denom;
            h_snow(i) = m_snow(i) / denom;
        });
    ice->h_ice_fld.modify_device();
    ice->h_snow_fld.modify_device();
}

/* EOS-style verify (h_ice/h_snow are a full overwrite from intact a/m/ms). */
static void fesom_ice_h_diag_verify(fesom_ice *ice, struct fesom_mesh *mesh, int step_n)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    std::vector<real_t> ki(ice->h_ice, ice->h_ice + N), ks(ice->h_snow, ice->h_snow + N);
    fesom_ice_h_diag(ice, mesh);   /* C twin recomputes h_ice/h_snow */
    double di = 0.0, ds = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = std::fabs((double)ki[i]-(double)ice->h_ice[i]);  if (x>di) di=x;
        x        = std::fabs((double)ks[i]-(double)ice->h_snow[i]); if (x>ds) ds=x;
    }
    std::copy(ki.begin(), ki.end(), ice->h_ice); std::copy(ks.begin(), ks.end(), ice->h_snow);
    double dmax = di > ds ? di : ds;
    const std::string be = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=icemap] step %d backend=%s  max|Δ|: h_ice=%.3e h_snow=%.3e\n",
                step_n, be.c_str(), di, ds);
    std::fflush(stdout);
    if (be == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=icemap] FAIL step %d: h_diag Serial must be "
                             "bit-identical (max|Δ|=%.3e)\n", step_n, dmax); std::abort();
    }
}

void fesom_ice_step(int                            step,
                    fesom_ice                     *ice,
                    struct fesom_partit           *partit,
                    struct fesom_mesh             *mesh,
                    const struct fesom_dyn        *dyn,
                    const struct fesom_tracers    *tracers,
                    struct fesom_forcing          *forcing,
                    const struct fesom_jra55      *jra,
                    const struct fesom_sss_runoff *sr,
                    const struct fesom_ssh_stiff  *stiff)
{
    load_ice_env_once();
    load_ice_verify_once();

    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;

    /* Phase C1: ocean2ice — populate ice->srfoce_* from ocean state. M4.3a: ON THE DEVICE
     * (a device island within the still-host ice step — EVP/FCT/thermo move in M4.3b-d).
     * IN rail (L28): push the ocean state it reads — tracers T/S (surface), mesh hbar, dyn uv
     * (host-current from the previous ocean step; const → localized const_cast, the forcing
     * pattern). OUT: sync_host(srfoce_*) for the host EVP + the u_w/v_w nod2D halo (driver,
     * the ALE pattern). Falls back to no-op if dyn or tracers are NULL. */
    if (dyn && tracers) {
        struct fesom_dyn     *d  = const_cast<struct fesom_dyn *>(dyn);
        struct fesom_tracers *tr = const_cast<struct fesom_tracers *>(tracers);
        tr->data[FESOM_TRACER_T].values_fld.modify_host(); tr->data[FESOM_TRACER_T].values_fld.sync_device();
        tr->data[FESOM_TRACER_S].values_fld.modify_host(); tr->data[FESOM_TRACER_S].values_fld.sync_device();
        mesh->hbar_fld.modify_host(); mesh->hbar_fld.sync_device();
        d->uv_fld.modify_host();      d->uv_fld.sync_device();
        fesom_ocean2ice_kk(ice, dyn, tracers, partit, mesh);
        ice->srfoce_temp_fld.sync_host(); ice->srfoce_salt_fld.sync_host(); ice->srfoce_ssh_fld.sync_host();
        ice->srfoce_u_fld.sync_host();    ice->srfoce_v_fld.sync_host();
        fesom_exchange_nod2D(ice->srfoce_u_fld.h_checked(), partit);   /* Fortran ocean2ice line 244 */
        fesom_exchange_nod2D(ice->srfoce_v_fld.h_checked(), partit);
        /* Verify AFTER the halo so the device srfoce_u/v halo (driver-exchanged) matches the
         * C twin's own halo at np>1 — the kernel leaves srfoce_u/v halo=0 pre-exchange, so a
         * pre-halo diff would false-positive on the halo nodes (the C twin halos internally). */
        if (s_verify_icemap) fesom_ocean2ice_verify(ice, dyn, tracers, partit, mesh, step);
    }

    if (!s_no_ice_dyn) {
        /* whichEVP=0 (standard EVP) only; mEVP/aEVP are out of scope. */
        if (ice->whichEVP == 0) {
            fesom_ice_evp_dynamics(ice, partit, mesh);
        } else {
            fprintf(stderr, "fesom_ice: whichEVP=%d not supported (only standard EVP=0)\n",
                    ice->whichEVP);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    /* Phase E — FCT advection. Mirrors Fortran ice_setup_step.F90:258-261:
     * call ice_TG_rhs; call ice_fct_solve. cut_off (line 295) follows
     * unconditionally below. */
    if (!s_no_ice_adv && stiff) {
        fesom_ice_tg_rhs   (ice,       partit, mesh);
        fesom_ice_fct_solve(ice, stiff, partit, mesh);
    }
    /* cut_off (Fortran line 295): runs after FCT and BEFORE thermodynamics regardless of
     * NO_ICE_ADV. M4.3a: ON THE DEVICE. IN: push the ice tracer values (host EVP/FCT wrote
     * them via the raw alias, L14). capture-before (RMW clamp) if verify. OUT: sync_host for
     * the host thermo. No halo (cut_off over [0,N) covers halo directly). */
    {
        std::vector<real_t> ca, cm, cs;
        if (s_verify_icemap) {
            ca.assign(ice->data[FESOM_ICE_AICE].values,  ice->data[FESOM_ICE_AICE].values  + N);
            cm.assign(ice->data[FESOM_ICE_MICE].values,  ice->data[FESOM_ICE_MICE].values  + N);
            cs.assign(ice->data[FESOM_ICE_MSNOW].values, ice->data[FESOM_ICE_MSNOW].values + N);
        }
        ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
        fesom_ice_cut_off_kk(ice, partit, mesh);
        ice->data[FESOM_ICE_AICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MSNOW].values_fld.sync_host();
        if (s_verify_icemap) fesom_ice_cut_off_verify(ice, partit, mesh, step, ca, cm, cs);
    }

    /* Thermodynamics + ice→ocean flux update. Both need forcing+jra+sr; without
     * them (e.g. pi-mesh smoke test) the step is silently a no-op. */
    if (!s_no_ice_thermo && forcing && jra && sr) {
        fesom_ice_thermodynamics(ice, partit, mesh, forcing, jra, sr);
        /* Phase C2/C3: oce_fluxes overwrites heat_flux/water_flux with the
         * ice-mediated flx_h/flx_fw and computes virtual_salt + relax_salt. */
        fesom_ice_oce_fluxes(ice, partit, mesh, tracers, forcing, sr);
    }

    /* Post-step diagnostic h_ice/h_snow (Fortran ice_setup_step.F90:319-330). M4.3a: ON THE
     * DEVICE. IN: push a_ice/m_ice/m_snow (the host thermo, if it ran, wrote them via the raw
     * alias). OUT: sync_host(h_ice/h_snow) for I/O + the FESOM_DIAG_MICE block below. */
    real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;     /* kept for the DEBUG diag blocks */
    real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    (void)a_ice;
    ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
    ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
    ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
    fesom_ice_h_diag_kk(ice, mesh);
    ice->h_ice_fld.sync_host();
    ice->h_snow_fld.sync_host();
    if (s_verify_icemap) fesom_ice_h_diag_verify(ice, mesh, step);

    /* DEBUG (FESOM_DIAG_MICE): per-step global max m_ice + max ice speed with
       their global node ids. m_ice should stay < ~10 m physically; MAXLOC
       pinpoints the FIRST runaway node/step (trigger), and whether the ice
       SPEED blows up first (EVP-driven) or the MASS (advection/thermo). */
    if (getenv("FESOM_DIAG_MICE")) {
        struct { double v; int g; } lm = {0.0, -1}, lu = {0.0, -1};
        for (int n = 0; n < mesh->myDim_nod2D; ++n) {
            if ((double)m_ice[n] > lm.v) { lm.v = m_ice[n]; lm.g = partit->myList_nod2D[n]; }
            double sp2 = (double)ice->uice[n]*ice->uice[n] + (double)ice->vice[n]*ice->vice[n];
            if (sp2 > lu.v) { lu.v = sp2; lu.g = partit->myList_nod2D[n]; }
        }
        struct { double v; int g; } gm, gu;
        MPI_Allreduce(&lm, &gm, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        MPI_Allreduce(&lu, &gu, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (partit->mype == 0)
            fprintf(stderr, "[mice-mon] step %d max_mice=%.4e gid=%d  max_uice=%.4e gid=%d\n",
                    step, gm.v, gm.g, sqrt(gu.v), gu.g);
    }

    /* DEBUG (FESOM_DIAG_GID=<global node id>): per-step EVP forcing breakdown
       at one node + (step 1) geometry of its surrounding elements — to see
       WHICH forcing/geometry term drives the single-node velocity runaway. */
    if (getenv("FESOM_DIAG_GID")) {
        int tgid = atoi(getenv("FESOM_DIAG_GID"));
        for (int nn = 0; nn < mesh->myDim_nod2D; ++nn) {
            if (partit->myList_nod2D[nn] != tgid) continue;
            int beg = mesh->nod_in_elem2D_offsets[nn];
            int end = mesh->nod_in_elem2D_offsets[nn + 1];
            real_t istr = 0.0, amin = 1e30;
            for (int k = beg; k < end; ++k) {
                int el = mesh->nod_in_elem2D[k];
                if (ice->work.ice_strength[el] > istr) istr = ice->work.ice_strength[el];
                if (mesh->elem_area[el] < amin) amin = mesh->elem_area[el];
            }
            real_t sp = sqrt(ice->uice[nn]*ice->uice[nn] + ice->vice[nn]*ice->vice[nn]);
            fprintf(stderr,
              "[gid %d step %d r%d] |uice|=%.4e urhs=(%.3e,%.3e) iam=%.4e im=%.4e "
              "rhs_a=%.3e rhs_m=%.3e satmice=(%.3e,%.3e) uw=(%.3e,%.3e) istr=%.4e "
              "amin=%.4e mice=%.4e aice=%.4f\n",
              tgid, step, partit->mype, (double)sp,
              (double)ice->uice_rhs[nn], (double)ice->vice_rhs[nn],
              (double)ice->work.inv_areamass[nn], (double)ice->work.inv_mass[nn],
              (double)ice->data[FESOM_ICE_AICE].values_rhs[nn],
              (double)ice->data[FESOM_ICE_MICE].values_rhs[nn],
              (double)ice->stress_atmice_x[nn], (double)ice->stress_atmice_y[nn],
              (double)ice->srfoce_u[nn], (double)ice->srfoce_v[nn],
              (double)istr, (double)amin, (double)m_ice[nn], (double)a_ice[nn]);
            /* SSH structure at this node: the elevation (srfoce_ssh) value, the
               1-ring neighbour spread (a growing min/max gap = grid-scale 2dx
               oscillation), and the max per-element elevation gradient — the
               REAL driver of the EVP sea-surface-tilt rhs_a (recomputed here,
               since data[AICE].values_rhs is clobbered by the FCT advection). */
            {
                real_t *ssh = ice->srfoce_ssh;
                real_t smin = ssh[nn], smax = ssh[nn], gmax = 0.0;
                for (int k = beg; k < end; ++k) {
                    int el = mesh->nod_in_elem2D[k];
                    real_t *gs = &mesh->gradient_sca[6*el];
                    int v0 = mesh->elem_nodes[3*el+0];
                    int v1 = mesh->elem_nodes[3*el+1];
                    int v2 = mesh->elem_nodes[3*el+2];
                    real_t edx = gs[0]*ssh[v0]+gs[1]*ssh[v1]+gs[2]*ssh[v2];
                    real_t edy = gs[3]*ssh[v0]+gs[4]*ssh[v1]+gs[5]*ssh[v2];
                    real_t g = sqrt(edx*edx + edy*edy);
                    if (g > gmax) gmax = g;
                    real_t sv[3] = { ssh[v0], ssh[v1], ssh[v2] };
                    for (int j = 0; j < 3; ++j) {
                        if (sv[j] < smin) smin = sv[j];
                        if (sv[j] > smax) smax = sv[j];
                    }
                }
                fprintf(stderr, "[gid %d SSH step %d r%d] ssh=%.6e ring[min=%.6e max=%.6e spread=%.4e] max_grad=%.6e\n",
                    tgid, step, partit->mype, (double)ssh[nn],
                    (double)smin, (double)smax, (double)(smax-smin), (double)gmax);
            }
            if (step == 1) {
                for (int k = beg; k < end; ++k) {
                    int el = mesh->nod_in_elem2D[k];
                    real_t *gs = &mesh->gradient_sca[6*el];
                    real_t gn = 0.0; for (int j = 0; j < 6; ++j) gn += gs[j]*gs[j];
                    fprintf(stderr, "[gid %d GEOM r%d] el_g=%d area=%.4e metric=%.4e gradnorm=%.4e\n",
                      tgid, partit->mype, partit->myList_elem2D[el]-1,
                      (double)mesh->elem_area[el], (double)mesh->metric_factor[el], (double)sqrt(gn));
                }
            }
            break;
        }
    }

#ifdef FESOM_KK_SYNCCHECK
    /* M1.5: exercise the host<->device rails on this step's ice state (no-op in production). */
    ice_synccheck_roundtrip(ice);
#endif
}

void fesom_ice_free(fesom_ice *ice)
{
    // Every array (incl. the embedded data[]/work/thermo sub-structs and the lazily-alloc'd
    // work.fct_massmatrix) is now OWNED by a fesom::Field; the raw pointers are non-owning
    // aliases, so they must NOT be free()d. `*ice = fesom_ice{}` releases every DualView (Kokkos
    // refcounting) and zeros the PODs — the assignment IS the release (D13). Mirrors fesom_dyn_free.
    // mesh->bc_index_nod2D is owned by mesh; freed by fesom_mesh_free.
    *ice = fesom_ice{};
}
