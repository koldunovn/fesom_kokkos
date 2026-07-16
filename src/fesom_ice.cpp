#include "fesom_ice.h"
#include "fesom_profile.hpp"   // M5.6: per-phase ice-step timing (FESOM_STEP_PROFILE)
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_ice_coupling.h"
#include "fesom_ice_evp.h"
#include "fesom_ice_maevp.h"
#include "fesom_ice_evpwide.h"   /* M7 E.EVP1: extended-slot tails for the wide-halo EVP */
#include "fesom_ice_fct.h"
#include "fesom_ice_thermo.h"
#include "fesom_forcing.h"   // M4.3d-a: forcing->{runoff,Ch_atm_oce,Ce_atm_oce}_fld IN-rail push
#include "fesom_jra55.h"     // M4.3d-a: the 8 jra physics Fields IN-rail push
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   // M7 H.2 ICERAILS: the srfoce_u/v host halo -> ONE device halo
#include "fesom_speed.hpp"         // M7 H.2 ICERAILS
#include "fesom_bulk.h"            // M7 H.3 BULKTAIL: fesom_bulktail_on()
#include "fesom_io.h"              // M7 H.8 LAZYSNAP: fesom_lazysnap_on()
#include "fesom_step.h"            // M7 H.9 SSHRAILS: fesom_sshrails_on() (the :634 hbar push gate)
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

    /* M6.2 — EVP flavour knob, transcribed from the C (fesom_ice.c:73-90). Numeric env
     * (the FESOM_DIAG_GID atoi precedent), parsed at init because it gates the conditional
     * allocations below: unset/"0" -> standard EVP (default), "1" -> mEVP; anything else
     * aborts — aEVP (2) and garbage are NOT ported. strtol with a full-string check, so a
     * typo fails loudly instead of silently selecting std EVP. */
    {
        const char *we = getenv("FESOM_WHICH_EVP");
        int which = 0;
        if (we && *we) {
            char *end = nullptr;
            long v = strtol(we, &end, 10);
            int ok = (end != we && *end == '\0' && (v == 0 || v == 1));
            if (!ok) {
                fprintf(stderr, "fesom_ice: FESOM_WHICH_EVP='%s' -> whichEVP not ported "
                                "(0=EVP default, 1=mEVP; aEVP/2 unported)\n", we);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            which = (int)v;
        }
        ice->whichEVP = which;
    }
    ice->alpha_evp       = 250.0;   /* mEVP stability constants — set unconditionally, as the C */
    ice->beta_evp        = 250.0;   /* does (namelist.ice == MOD_ICE.F90:198 module default).    */
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
    /* M7 E.EVP1 (opt-in FESOM_SPEED_EVPWIDE): ring 2..R ghost values live in extended slots
     * appended at [N, N+nwx) on EXACTLY the fields the wide-EVP ghost kernels index. Every
     * existing loop uses [0,N) and never sees the tail; knob off => nwx == 0 (identical allocs). */
    size_t nwx = (size_t)fesom_evpwide_next();
    size_t nw  = n + nwx;

    // M1.4: each Field owns its storage; the raw pointer is a non-owning alias = field.h()
    // (D12). .alloc(label, count) takes the count in ELEMENTS (== the old xcalloc count) and
    // zero-inits like calloc → Serial bit-identical. Halo-sized node (n) / element (e, e3) extents
    // kept verbatim.
    /* --- velocities (T_ICE:132-133, ice_init:719-740) --- */
    ice->uice_fld.alloc("ice.uice", nw);                      ice->uice            = ice->uice_fld.h();
    ice->uice_rhs_fld.alloc("ice.uice_rhs", n);               ice->uice_rhs        = ice->uice_rhs_fld.h();
    ice->uice_old_fld.alloc("ice.uice_old", n);               ice->uice_old        = ice->uice_old_fld.h();
    ice->vice_fld.alloc("ice.vice", nw);                      ice->vice            = ice->vice_fld.h();
    ice->vice_rhs_fld.alloc("ice.vice_rhs", n);               ice->vice_rhs        = ice->vice_rhs_fld.h();
    ice->vice_old_fld.alloc("ice.vice_old", n);               ice->vice_old        = ice->vice_old_fld.h();

    /* --- stresses (T_ICE:136-137) --- */
    ice->stress_atmice_x_fld.alloc("ice.stress_atmice_x", nw); ice->stress_atmice_x = ice->stress_atmice_x_fld.h();
    ice->stress_iceoce_x_fld.alloc("ice.stress_iceoce_x", n); ice->stress_iceoce_x = ice->stress_iceoce_x_fld.h();
    ice->stress_atmice_y_fld.alloc("ice.stress_atmice_y", nw); ice->stress_atmice_y = ice->stress_atmice_y_fld.h();
    ice->stress_iceoce_y_fld.alloc("ice.stress_iceoce_y", n); ice->stress_iceoce_y = ice->stress_iceoce_y_fld.h();

    /* --- diagnostic h_ice/h_snow (T_ICE:150) --- */
    ice->h_ice_fld.alloc("ice.h_ice", n);                     ice->h_ice           = ice->h_ice_fld.h();
    ice->h_snow_fld.alloc("ice.h_snow", n);                   ice->h_snow          = ice->h_snow_fld.h();

    /* --- mEVP (whichEVP==1) aux + scratch. Allocated ONLY when selected, mirroring the C
     * (fesom_ice.c) and the Fortran (which allocates these only for whichEVP != 0). Under the
     * default (std EVP) nothing here is touched, so the knob-OFF byte gate is unaffected. */
    if (ice->whichEVP == 1) {
        const size_t nn = (size_t)mesh->myDim_nod2D;
        const size_t ne = (size_t)mesh->myDim_elem2D;
        ice->uice_aux_fld.alloc("ice.uice_aux", n);  ice->uice_aux = ice->uice_aux_fld.h();
        ice->vice_aux_fld.alloc("ice.vice_aux", n);  ice->vice_aux = ice->vice_aux_fld.h();
        ice->work.mevp_inv_thickness_fld.alloc("ice.mevp_inv_thickness", nn);
        ice->work.mevp_mass_fld.alloc          ("ice.mevp_mass",          nn);
        ice->work.mevp_ice_nod_fld.alloc       ("ice.mevp_ice_nod",       nn);
        ice->work.mevp_pressure_fac_fld.alloc  ("ice.mevp_pressure_fac",  ne);
        ice->work.mevp_ice_el_fld.alloc        ("ice.mevp_ice_el",        ne);
        ice->work.mevp_inv_thickness = ice->work.mevp_inv_thickness_fld.h();
        ice->work.mevp_mass          = ice->work.mevp_mass_fld.h();
        ice->work.mevp_ice_nod       = ice->work.mevp_ice_nod_fld.h();
        ice->work.mevp_pressure_fac  = ice->work.mevp_pressure_fac_fld.h();
        ice->work.mevp_ice_el        = ice->work.mevp_ice_el_fld.h();
    }

    /* --- surface ocean state (T_ICE:140-142, ice_init:758-767) --- */
    ice->srfoce_u_fld.alloc("ice.srfoce_u", nw);              ice->srfoce_u    = ice->srfoce_u_fld.h();
    ice->srfoce_v_fld.alloc("ice.srfoce_v", nw);              ice->srfoce_v    = ice->srfoce_v_fld.h();
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
        snprintf(lbl, sizeof lbl, "ice.data%d.values", k);         d->values_fld.alloc(lbl, nw);         d->values         = d->values_fld.h();
        snprintf(lbl, sizeof lbl, "ice.data%d.values_old", k);     d->values_old_fld.alloc(lbl, n);      d->values_old     = d->values_old_fld.h();
        snprintf(lbl, sizeof lbl, "ice.data%d.values_rhs", k);     d->values_rhs_fld.alloc(lbl, nw);     d->values_rhs     = d->values_rhs_fld.h();
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
    w->inv_areamass_fld.alloc("ice.work.inv_areamass", nw);   w->inv_areamass    = w->inv_areamass_fld.h();  /* populated by EVPdynamics each step */
    w->inv_mass_fld.alloc("ice.work.inv_mass", nw);           w->inv_mass        = w->inv_mass_fld.h();      /* populated by EVPdynamics each step */

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

        /* M6.2 — push to the DEVICE. Until now bc_index_nod2D was host-only (std EVP uses
         * `coastal` instead), but the mEVP node solve multiplies its determinant by it
         * (:814, trap 9), on the device.
         *
         * ⚠️ IT MUST BE modify_host() THEN sync_device(). A bare sync_device() here would be
         * a SILENT NO-OP: Field::alloc tags the field Auth::Synced with BOTH spaces zeroed,
         * and the mask above is written through the RAW HOST POINTER, which never sets the
         * dirty bit (L14). So the DualView still believes host and device agree — it would
         * copy nothing, and the device would keep an ALL-ZERO bc_index. Under CUDA that
         * zeroes the mEVP determinant at every node (catastrophic); under Serial host==device
         * so it is completely invisible. This is the one line where Serial cannot protect us. */
        mesh->bc_index_nod2D_fld.modify_host();
        mesh->bc_index_nod2D_fld.sync_device();
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

/* FESOM_KK_VERIFY ice gates: icemap (M4.3a ocean2ice+cut_off+diag), evp (M4.3b),
 * icefct (M4.3c FCT advection). Cached. */
static int s_ice_verify_loaded = 0;
static int s_verify_icemap     = 0;
static int s_verify_evp        = 0;
static int s_verify_icefct     = 0;
static int s_verify_icethermo  = 0;
static int s_verify_iceflux    = 0;
static void load_ice_verify_once(void)
{
    if (s_ice_verify_loaded) return;
    const char *e = getenv("FESOM_KK_VERIFY");
    s_verify_icemap    = (e && strstr(e, "icemap"))    ? 1 : 0;   /* collision-free token (L25) */
    s_verify_evp       = (e && strstr(e, "evp"))       ? 1 : 0;   /* M4.3b */
    s_verify_icefct    = (e && strstr(e, "icefct"))    ? 1 : 0;   /* M4.3c (collision-free, L25) */
    s_verify_icethermo = (e && strstr(e, "icethermo")) ? 1 : 0;   /* M4.3d-a (collision-free, L25) */
    s_verify_iceflux   = (e && strstr(e, "iceflux"))   ? 1 : 0;   /* M4.3d-b (collision-free, L25) */
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

    /* M5.6: per-phase ice timing (FESOM_STEP_PROFILE; host+device). PMARK_ICE closes the current
     * phase + opens the next; final toc + #undef before the synccheck. No-op when profiling off. */
#define PMARK_ICE(nm) do { fesom_prof::toc((nm), _it); _it = fesom_prof::tic(); } while (0)
    double _it = fesom_prof::tic();

    /* Phase C1: ocean2ice — populate ice->srfoce_* from ocean state. M4.3a: ON THE DEVICE
     * (a device island within the still-host ice step — EVP/FCT/thermo move in M4.3b-d).
     * IN rail (L28): push the ocean state it reads — tracers T/S (surface), mesh hbar, dyn uv
     * (host-current from the previous ocean step; const → localized const_cast, the forcing
     * pattern). OUT: sync_host(srfoce_*) for the host EVP + the u_w/v_w nod2D halo (driver,
     * the ALE pattern). Falls back to no-op if dyn or tracers are NULL. */
    /* ==================== M7 H.2 — FESOM_SPEED_ICERAILS (package H, the payoff) ==============
     * The ENTIRE ice step — ocean2ice, EVP, FCT, cut_off, thermo, oce_fluxes, h_diag — is device
     * kernels. Yet it carries 40 H2D + 27 D2H FULL-nod2D-field rails per step (~250-300 MB at
     * NG5@4N), because it still shuttles its own state host-side between consecutive DEVICE
     * kernels. The comments justifying those rails are STALE: they say "sync_host … for the host
     * EVP / the host cut_off / the host thermo" (see :521, :586, :609, :635 below) — every one of
     * those is a `_kk` device kernel now. Concretely:
     *   - srfoce_temp/salt go D2H here and are re-pushed H2D 165 lines later with NO host reader
     *     in between;
     *   - a_ice/m_ice/m_snow bounce D2H->H2D FOUR TIMES in one step (FCT->cut_off->thermo->h_diag);
     *   - sigma11/12/22 go D2H every step and are read only by the NEXT step's H2D.
     *
     * MEASURED, three independent ways (docs/plans/20260714-m7-PACKAGE-H-rails.md):
     *   memcpy accounting  : the model's own full-field rails are 77.6 ms/step (81% of the memcpy
     *                        pool the plan had labelled "MPI staging");
     *   source audit       : 40 H2D + 27 D2H in this function;
     *   GPU-IDLE GAP census: the GPU sits idle 22.7 ms before ice_thermodynamics_kk, 19.5 ms before
     *                        ice_evp_dynamics_kk and 16.5 ms before ocean2ice_kk = 58.7 ms/step of
     *                        the GPU doing NOTHING while PCIe shuffles full fields.
     *
     * WHAT PINNED IT: only three things ever needed the host, and H.1 (FLUXDEV) already killed two
     * (the 4 forcing host halos, and the host fesom_cal_shortwave_rad reading a_ice). The last one
     * is the 2 srfoce_u/v HOST halos immediately below — so this lever starts by converting them to
     * ONE device fesom_halo_field2, and the whole stack falls.
     *
     * WHAT STAYS: hbar / runoff / Ssurf are genuinely HOST-authoritative (the ocean's host eta_n
     * loop writes hbar; sss_runoff writes runoff) — their H2D is real work, not a round trip.
     * And at the END of the ice step we sync_host() exactly the fields a HOST reader can still
     * touch: a/m/ms + uice/vice + srfoce_u/v (fesom_ice_oce_fluxes_mom's host twin reads srfoce_u/v
     * and uice/vice — fesom_ice_coupling.cpp:62 — and it runs unless ICEFLUXDEV is on) and
     * h_ice/h_snow (the FESOM_DIAG_MICE block). 9 D2H at the end instead of 67 rails throughout.
     *
     * 🔴 EVERY OTHER host writer of this state is a VERIFY TWIN (checked by grep: the host EVP
     * fesom_ice_evp.cpp:83-131, the host thermo fesom_ice_thermo.cpp:551-593, the host oce_fluxes
     * fesom_ice_coupling.cpp:264). Those twins CAPTURE-BEFORE from the raw HOST alias, which under
     * this knob is deliberately stale between kernels — so the two are INCOHERENT BY CONSTRUCTION.
     * We abort loudly rather than silently compare against garbage. (SWSKIP sets the precedent.)
     *
     * ⚠️ THE D.1 TRAP, FOR THE LAST TIME: on Serial .d() and .h() are the SAME MEMORY, so every
     * rail deleted here is a NO-OP there and the FORCE_SERIAL byte proof passes WHETHER OR NOT THIS
     * IS RIGHT. NO SERIAL GATE CAN VALIDATE A COHERENCE INVARIANT (L86). The CUDA fidelity gate is
     * the ONLY gate for this lever. Run it standalone before believing anything.
     *
     * Cross-step safety: sigma11/12/22 stay DEVICE-authoritative across steps. fesom_ic.cpp does
     * NOT memset them (verified), so Field::alloc()'s zero-init of BOTH spaces (Auth::Synced) is
     * what step 1 reads. The end-of-step sync_host() leaves a/m/ms + uice/vice Synced, so the next
     * step's kernels read a current device view with no push.
     * ======================================================================================== */
    static int s_icerails = -1;
    const bool icerails = fesom_speed_on("ICERAILS", &s_icerails);

    /* 🔴🔴 THE ONE-SHOT IC PUSH — and the story of how it was found. DO NOT DELETE.
     *
     * The ice INITIAL CONDITION (a/m/ms, uice/vice, sigma11/12/22, t_skin, thdgr) is written ON THE
     * HOST at init through the raw alias. `Field::alloc()` zero-inits BOTH spaces and marks them
     * Synced (fesom_field.hpp:64) — it never learns the host was written afterwards. And the ONLY
     * thing that ever carried that IC to the device was the per-step IN rail this lever deletes.
     *
     * So the first version of ICERAILS left the DEVICE ice state at ZERO for step 1, and the sea ice
     * simply never existed. It is the Z7 trap wearing a new face: a field that "was always pushed
     * anyway" turns out to have had exactly one producer, and it was the rail you just removed.
     *
     * ⚠️ AND EVERY SERIAL GATE CERTIFIED IT AS CORRECT. On Serial .d() IS .h(), so all 55 deleted
     * rails are no-ops and the IC is trivially "on the device". The knob-OFF byte gate passed
     * (26255200) and — the sharp one — the FORCE_SERIAL BYTE PROOF passed BIT-IDENTICALLY against
     * the certified baseline (26255201, diff_snap rc=0), on the very binary whose CUDA run had NO
     * SEA ICE (26255202: a_ice max|Δ| = 9.83e-01, i.e. the entire ice concentration; 15 fields over
     * ceiling COHERENTLY). The project's STRONGEST gate blessed a catastrophically broken lever.
     * That is L86 stated as loudly as it can be: NO SERIAL GATE CAN EVER VALIDATE A COHERENCE
     * INVARIANT. The standalone CUDA fidelity gate is the only thing standing between this lever
     * and a silently ice-free ocean.
     *
     * (Restart note: this is a cold-start-only fix. If a restart path is ever added that reads the
     * ice state into the HOST arrays mid-run, it must re-push — a `static bool` will not fire.) */
    static bool s_ice_ic_pushed = false;
    if (icerails && !s_ice_ic_pushed) {
        s_ice_ic_pushed = true;
        ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
        ice->uice_fld.modify_host();            ice->uice_fld.sync_device();
        ice->vice_fld.modify_host();            ice->vice_fld.sync_device();
        ice->work.sigma11_fld.modify_host();    ice->work.sigma11_fld.sync_device();
        ice->work.sigma12_fld.modify_host();    ice->work.sigma12_fld.sync_device();
        ice->work.sigma22_fld.modify_host();    ice->work.sigma22_fld.sync_device();
        ice->thermo.t_skin_fld.modify_host();   ice->thermo.t_skin_fld.sync_device();
        ice->thermo.thdgr_fld.modify_host();    ice->thermo.thdgr_fld.sync_device();
        /* stress_atmice_x/y and Ch/Ce are re-written by fesom_bulk_compute_kk on the DEVICE in the
         * forcing phase of every step (including step 1, which runs before this one) — they need no
         * IC push. Pushed anyway: it is once, and an IC bug in this class costs a whole gate cycle.
         *
         * 🔴 M7 H.3 — AND UNDER BULKTAIL THAT BELT-AND-BRACES BECOMES THE BUG IT WAS INSURING AGAINST.
         * Today bulk's host tail sync_host()s stress_atmice, so the mirror is current and this push is
         * a harmless self-assignment. BULKTAIL deletes that tail, leaving stress_atmice DEVICE-
         * authoritative with a host mirror that is still Field::alloc()'s ZEROS. modify_host() then
         * sets host_count > dev_count and sync_device() DEEP-COPIES THE ZEROS OVER THE FRESH DEVICE
         * STRESS — killing the wind forcing on the ice, ON STEP 1 ONLY. That is the Z7 signature
         * (bitwise-equal at cold start is what makes step 1 the ONLY wrong step), and step 1 being
         * wrong is the whole run. The comment above says it best: they NEED NO IC PUSH. Under
         * BULKTAIL, don't do one. */
        if (!fesom_bulktail_on()) {
        ice->stress_atmice_x_fld.modify_host(); ice->stress_atmice_x_fld.sync_device();
        ice->stress_atmice_y_fld.modify_host(); ice->stress_atmice_y_fld.sync_device();
        }
    }

    /* M7 H.3 — BULKTAIL once refused to run with whichEVP != 0, because the mEVP branch below pushed
     * stress_atmice from the host UNCONDITIONALLY and would have clobbered the device copy that this
     * lever makes authoritative. That push is now gated on `!icerails` (the same fix that repaired the
     * PRE-EXISTING ICERAILS+mEVP clobber — see the mEVP branch), and mEVP reads `stress_atmice` ONLY
     * from `.d()` (fesom_ice_maevp.cpp:120-121) and touches none of BULKTAIL's other seven arrays.
     * So the two are compatible, and the refusal is gone.
     *
     * ⚠️ THE POINT OF THE OPTIONS MATRIX (L91) IS THAT THIS SENTENCE IS NOT EVIDENCE. It is gated:
     * CUDA fidelity, FESOM_SPEED=1 x FESOM_WHICH_EVP=1, against mEVP's OWN M6 Serial oracle. If you
     * add a lever that changes who owns a field, THAT GATE IS NOW STALE AND YOU MUST RE-RUN IT. */

    if (icerails && (s_verify_icemap || s_verify_evp || s_verify_icefct ||
                     s_verify_icethermo || s_verify_iceflux)) {
        if (partit->mype == 0)
            fprintf(stderr, "[fesom_speed] FESOM_SPEED_ICERAILS is INCOMPATIBLE with FESOM_KK_VERIFY="
                            "<icemap|evp|icefct|icethermo|iceflux>: the verify twins capture-before "
                            "from the raw HOST alias, which this lever deliberately leaves stale "
                            "between the ice kernels. Refusing to compare against garbage.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (dyn && tracers) {
        /* M5.13g1-T: T values device-resident across the boundary - no re-push; ocean2ice reads SST on device.
         * M5.14 (S flip): S values device-resident too (floor → device) - no re-push; ocean2ice reads srfoce_salt = S on device.
         * M5.13g1: uv device-resident across the step boundary (ocean update_vel fesom_halo_field) -
         * no re-push; ocean2ice reads uv surface (nz=0) on device. So T/S/uv all device-resident here now. */
        /* M7 H.9 SSHRAILS flips the ownership this comment asserts: under the knob hbar is
         * DEVICE-authoritative (device halo in the SSH block) and the host mirror is STALE — this
         * push would clobber the fresh device hbar that ocean2ice_kk is about to read (the
         * BULKTAIL-IC / Z7 signature, PROMPT §3.3 trap 1). Legacy: unchanged. */
        if (!fesom_sshrails_on()) {
            mesh->hbar_fld.modify_host(); mesh->hbar_fld.sync_device();   /* hbar IS host-authoritative (legacy) — keep */
        }
        fesom_ocean2ice_kk(ice, dyn, tracers, partit, mesh);
        if (icerails) {
            /* THE UNPIN. srfoce_* stay DEVICE-authoritative; the 2 host-staged halos (each a
             * full-field D2H + host MPI + full-field H2D) become ONE co-packed device exchange.
             * srfoce_temp/salt/ssh need no halo at all — ocean2ice_kk writes [0,N), halo included;
             * only u/v need one (the elem->node interpolation leaves the halo short). */
            fesom_halo_field2(ice->srfoce_u_fld, ice->srfoce_v_fld,
                              FESOM_HALO_NOD2D, 1, 1, partit);
        } else {
            ice->srfoce_temp_fld.sync_host(); ice->srfoce_salt_fld.sync_host(); ice->srfoce_ssh_fld.sync_host();
            ice->srfoce_u_fld.sync_host();    ice->srfoce_v_fld.sync_host();
            fesom_exchange_nod2D(ice->srfoce_u_fld.h_checked(), partit);   /* Fortran ocean2ice line 244 */
            fesom_exchange_nod2D(ice->srfoce_v_fld.h_checked(), partit);
        }
        /* Verify AFTER the halo so the device srfoce_u/v halo (driver-exchanged) matches the
         * C twin's own halo at np>1 — the kernel leaves srfoce_u/v halo=0 pre-exchange, so a
         * pre-halo diff would false-positive on the halo nodes (the C twin halos internally). */
        if (s_verify_icemap) fesom_ocean2ice_verify(ice, dyn, tracers, partit, mesh, step);
    }

    if (!s_no_ice_dyn) {
        /* whichEVP=0 (standard EVP) only; mEVP/aEVP are out of scope. M4.3b: ON THE DEVICE.
         * IN rail (L28/L14): push the EVP inputs the host producers/prev step wrote via raw
         * alias — a/m/ms (prev step's cut_off/thermo), srfoce_u/v/ssh (this step's ocean2ice,
         * host-current after its halo — L30 re-push), stress_atmice_x/y (forcing producers),
         * uice/vice + sigma11/12/22 (the RMW rheology state from the prev step's EVP). OUT:
         * sync_host(uice/vice/sigma) so the host FCT (uice/vice) + next step's IN rail see them. */
        if (ice->whichEVP == 0) {
            const int Nn = mesh->myDim_nod2D + mesh->eDim_nod2D;
            const int Eo = mesh->myDim_elem2D;
            std::vector<real_t> eu, ev, e11, e12, e22;
            if (s_verify_evp) {
                eu.assign (ice->uice, ice->uice + Nn);  ev.assign (ice->vice, ice->vice + Nn);
                e11.assign(ice->work.sigma11, ice->work.sigma11 + Eo);
                e12.assign(ice->work.sigma12, ice->work.sigma12 + Eo);
                e22.assign(ice->work.sigma22, ice->work.sigma22 + Eo);
            }
            /* M7 H.2 ICERAILS — 13 H2D + 5 D2H, ALL of them round trips between device kernels:
             * a/m/ms are device-current from the previous step (left Synced by the end-of-step
             * sync); srfoce_u/v/ssh from ocean2ice_kk above (device-halo'd); stress_atmice_x/y from
             * fesom_bulk_compute_kk (device) via its host tail's sync_host, i.e. Synced; uice/vice +
             * sigma11/12/22 are the EVP's own RMW state from last step's kernel. Nothing HOST wrote
             * any of them (the only host writers are the verify twins, which this knob forbids). */
            if (!icerails) {
            ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
            ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
            ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
            ice->srfoce_u_fld.modify_host();   ice->srfoce_u_fld.sync_device();
            ice->srfoce_v_fld.modify_host();   ice->srfoce_v_fld.sync_device();
            ice->srfoce_ssh_fld.modify_host(); ice->srfoce_ssh_fld.sync_device();
            ice->stress_atmice_x_fld.modify_host(); ice->stress_atmice_x_fld.sync_device();
            ice->stress_atmice_y_fld.modify_host(); ice->stress_atmice_y_fld.sync_device();
            ice->uice_fld.modify_host();    ice->uice_fld.sync_device();
            ice->vice_fld.modify_host();    ice->vice_fld.sync_device();
            ice->work.sigma11_fld.modify_host(); ice->work.sigma11_fld.sync_device();
            ice->work.sigma12_fld.modify_host(); ice->work.sigma12_fld.sync_device();
            ice->work.sigma22_fld.modify_host(); ice->work.sigma22_fld.sync_device();
            }
            fesom_ice_evp_dynamics_kk(ice, partit, mesh);
            if (!icerails) {
            ice->uice_fld.sync_host(); ice->vice_fld.sync_host();
            ice->work.sigma11_fld.sync_host(); ice->work.sigma12_fld.sync_host(); ice->work.sigma22_fld.sync_host();
            }
            if (s_verify_evp) fesom_ice_evp_verify(ice, partit, mesh, step, eu, ev, e11, e12, e22);
        } else if (ice->whichEVP == 1) {
            /* mEVP (M6.2, FESOM_WHICH_EVP=1). Same IN/OUT rail as std EVP — it reads and writes
             * exactly the same host-produced state (a/m/ms, srfoce_*, stress_atmice_*, uice/vice,
             * sigma11/12/22) — but the rheology itself is a different routine with deliberately
             * different constants and branches (see fesom_ice_maevp.cpp's trap list; do NOT
             * normalise it against the std-EVP kernel above). sigma11/12/22 PERSIST across calls
             * in mEVP (never zeroed on entry, trap 12), which is why they are on the IN rail.
             *
             * 🔴🔴 BUG FIX (M7 H.3 session). These rails were NOT gated on `icerails` while the
             * std-EVP branch 40 lines up WAS — so `FESOM_SPEED_ICERAILS=1` + `FESOM_WHICH_EVP=1`
             * SILENTLY CLOBBERED THE ICE DYNAMICS. Under ICERAILS, ocean2ice's five sync_host()
             * calls are replaced by a device halo (:611-623) and on CUDA fesom_halo_field2 RETURNS
             * EARLY (fesom_halo_device.hpp:129) leaving Auth::Device — so srfoce_u/v/ssh are
             * device-authoritative with a STALE host mirror. modify_host() then sets
             * host_count > dev_count and sync_device() DEEP-COPIES THAT STALE MIRROR BACK OVER THE
             * FRESH DEVICE DATA. mEVP ran on last step's surface currents, and on ZEROS at step 1.
             * Same story for stress_atmice (bulk's device output) and a/m/ms.
             *
             * WHY NOTHING CAUGHT IT — and it is the same answer as always (L86): the default is
             * whichEVP=0, so no default gate touches this branch; and M6 certified mEVP with a
             * SERIAL bit-identity test, where .d() IS .h() and every one of these rails is a no-op.
             * A COHERENCE INVARIANT IS NOT VALIDATED BY A SERIAL GATE. It never was.
             * The gate for this fix is CUDA + FESOM_WHICH_EVP=1 — which did not exist until now. */
            if (!icerails) {
            ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
            ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
            ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
            ice->srfoce_u_fld.modify_host();   ice->srfoce_u_fld.sync_device();
            ice->srfoce_v_fld.modify_host();   ice->srfoce_v_fld.sync_device();
            ice->srfoce_ssh_fld.modify_host(); ice->srfoce_ssh_fld.sync_device();
            ice->stress_atmice_x_fld.modify_host(); ice->stress_atmice_x_fld.sync_device();
            ice->stress_atmice_y_fld.modify_host(); ice->stress_atmice_y_fld.sync_device();
            ice->uice_fld.modify_host();    ice->uice_fld.sync_device();
            ice->vice_fld.modify_host();    ice->vice_fld.sync_device();
            ice->work.sigma11_fld.modify_host(); ice->work.sigma11_fld.sync_device();
            ice->work.sigma12_fld.modify_host(); ice->work.sigma12_fld.sync_device();
            ice->work.sigma22_fld.modify_host(); ice->work.sigma22_fld.sync_device();
            }
            /* M7 E.EVP1 + L80: EVPWIDE's resolve lives in the std-EVP path, which this branch
             * never reaches — without this line, FESOM_SPEED_EVPWIDE + mEVP would be a SILENT
             * no-op (correct but mute; the dead-knob trap). Announce loudly, once. */
            {
                static int s_warned = 0;
                if (!s_warned && fesom_evpwide_env_K() > 0) {
                    s_warned = 1;
                    if (partit->mype == 0) {
                        fprintf(stderr, "[fesom_speed] !! FESOM_SPEED_EVPWIDE requested but "
                                        "whichEVP=1 (mEVP has its own subcycle exchange) — the "
                                        "lever is NOT running.\n");
                        fflush(stderr);
                    }
                }
            }
            fesom_ice_evp_dynamics_m_kk(ice, partit, mesh);
            if (!icerails) {
            ice->uice_fld.sync_host(); ice->vice_fld.sync_host();
            ice->work.sigma11_fld.sync_host(); ice->work.sigma12_fld.sync_host(); ice->work.sigma22_fld.sync_host();
            }
        } else {
            fprintf(stderr, "fesom_ice: whichEVP=%d not supported (0=EVP, 1=mEVP)\n",
                    ice->whichEVP);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    PMARK_ICE("ice_dyn(o2i+EVP)");
    /* Phase E — FCT advection. Mirrors Fortran ice_setup_step.F90:258-261:
     * call ice_TG_rhs; call ice_fct_solve. cut_off (line 295) follows
     * unconditionally below. M4.3c: ON THE DEVICE. tg_rhs (element→node SCATTER into values_rhs)
     * + fct_solve (high/low-order CSR-gather mass-matrix solves [high_order = host-loop iter +
     * per-iter dvalues halo] + 3× Zalesak fem_fct [scatters + icepplus/icepminus/values halos]).
     * IN rail (L28): push uice/vice (EVP output) + data[*].values (the FCT reads then RMWs them);
     * the set-once fct_massmatrix + the ssh_stiff CSR are device-current from their one-shot pushes.
     * OUT: sync_host(data[*].values) for the host cut_off + thermo. capture-before (L26) the 3
     * values. The FCT-internal values_rhs/valuesl/dvalues/fct_* are fully recomputed → not pushed. */
    if (!s_no_ice_adv && stiff) {
        std::vector<real_t> fa, fm, fms;
        if (s_verify_icefct) {
            fa.assign (ice->data[FESOM_ICE_AICE].values,  ice->data[FESOM_ICE_AICE].values  + N);
            fm.assign (ice->data[FESOM_ICE_MICE].values,  ice->data[FESOM_ICE_MICE].values  + N);
            fms.assign(ice->data[FESOM_ICE_MSNOW].values, ice->data[FESOM_ICE_MSNOW].values + N);
        }
        /* M7 H.2 ICERAILS — 5 H2D + 3 D2H, all round trips: uice/vice come straight from the EVP
         * kernel above, a/m/ms from the previous step. The FCT is `_kk`; the "host FCT" the OUT
         * rail's comment cites has not existed since M4.3c. */
        if (!icerails) {
        ice->uice_fld.modify_host(); ice->uice_fld.sync_device();
        ice->vice_fld.modify_host(); ice->vice_fld.sync_device();
        ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
        }
        fesom_ice_tg_rhs_kk   (ice,       partit, mesh);
        fesom_ice_fct_solve_kk(ice, stiff, partit, mesh);
        if (!icerails) {
        ice->data[FESOM_ICE_AICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MSNOW].values_fld.sync_host();
        }
        if (s_verify_icefct) fesom_ice_fct_verify(ice, stiff, partit, mesh, step, fa, fm, fms);
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
        /* M7 H.2 ICERAILS — 3 H2D + 3 D2H, a pure D2H->H2D bounce of a/m/ms between two device
         * kernels (FCT above, thermo below). The "host cut_off" and "host thermo" in the comments
         * are `_kk` kernels since M4.3a/M4.3d. */
        if (!icerails) {
        ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
        }
        fesom_ice_cut_off_kk(ice, partit, mesh);
        if (!icerails) {
        ice->data[FESOM_ICE_AICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MSNOW].values_fld.sync_host();
        }
        if (s_verify_icemap) fesom_ice_cut_off_verify(ice, partit, mesh, step, ca, cm, cs);
    }

    PMARK_ICE("ice_fct");
    /* Thermodynamics + ice→ocean flux update. Both need forcing+jra+sr; without
     * them (e.g. pi-mesh smoke test) the step is silently a no-op. M4.3d-a: thermodynamics ON
     * THE DEVICE — per-node column physics over [0,N) (race-free → Serial AND OpenMP bit-identical,
     * NO scatter; therm_ice/obudget/budget/flooding device twins) + the ustar map + its halo. IN
     * rail (L28): push every input the body reads — the 8 jra physics arrays (jra const→const_cast),
     * forcing runoff/Ch/Ce, ice uice/vice/srfoce/values/t_skin/thdgr. OUT: sync_host the 9 consumed
     * outputs (the 3 values + t_skin + flx_h/flx_fw + thdgr/thdgrsn/thdgra) for the host oce_fluxes
     * (flx_*) + h_diag (values) + next-step thermo. capture-before (L26) the 5 RMW inputs. oce_fluxes
     * STAYS HOST (M4.3d-b). */
    if (!s_no_ice_thermo && forcing && jra && sr) {
        std::vector<real_t> tm, ts, ta, tt, tg;
        if (s_verify_icethermo) {
            tm.assign(ice->data[FESOM_ICE_MICE].values,  ice->data[FESOM_ICE_MICE].values  + N);
            ts.assign(ice->data[FESOM_ICE_MSNOW].values, ice->data[FESOM_ICE_MSNOW].values + N);
            ta.assign(ice->data[FESOM_ICE_AICE].values,  ice->data[FESOM_ICE_AICE].values  + N);
            tt.assign(ice->thermo.t_skin, ice->thermo.t_skin + N);
            tg.assign(ice->thermo.thdgr,  ice->thermo.thdgr  + N);
        }
        struct fesom_jra55 *j = const_cast<struct fesom_jra55 *>(jra);
        /* IN rail — jra (8): freshly time-interpolated on the host each step via the raw alias (L14).
         * 🔴 M7 D.1 (FORCEDEV): skipped when the producer is a device kernel — otherwise this
         * pushes the stale host mirror over the fresh device data. (Invisible on Serial: the
         * views are the same memory there. The CUDA fidelity gate is the gate.) */
        if (!fesom_forcing_dev_on()) {
        j->u_wind_fld.modify_host();    j->u_wind_fld.sync_device();
        j->v_wind_fld.modify_host();    j->v_wind_fld.sync_device();
        j->shum_fld.modify_host();      j->shum_fld.sync_device();
        j->shortwave_fld.modify_host(); j->shortwave_fld.sync_device();
        j->longwave_fld.modify_host();  j->longwave_fld.sync_device();
        j->Tair_fld.modify_host();      j->Tair_fld.sync_device();
        j->prec_rain_fld.modify_host(); j->prec_rain_fld.sync_device();
        j->prec_snow_fld.modify_host(); j->prec_snow_fld.sync_device();
        }
        /* forcing (3) + ice inputs (uice/vice/srfoce/values/t_skin/thdgr).
         * M7 H.2 ICERAILS: runoff is genuinely HOST-authoritative (sss_runoff writes it) — KEEP.
         * Ch/Ce are produced by fesom_bulk_compute_kk on the DEVICE and only round-tripped through
         * its host tail (fesom_bulk.cpp:629-635), so they are Synced — skip. All 11 ice inputs are
         * this step's own device output (uice/vice from EVP, srfoce_* from ocean2ice, a/m/ms from
         * cut_off, t_skin/thdgr the thermo's own RMW state from last step). */
        forcing->runoff_fld.modify_host();     forcing->runoff_fld.sync_device();
        if (!icerails) {
        forcing->Ch_atm_oce_fld.modify_host(); forcing->Ch_atm_oce_fld.sync_device();
        forcing->Ce_atm_oce_fld.modify_host(); forcing->Ce_atm_oce_fld.sync_device();
        ice->uice_fld.modify_host();        ice->uice_fld.sync_device();
        ice->vice_fld.modify_host();        ice->vice_fld.sync_device();
        ice->srfoce_u_fld.modify_host();    ice->srfoce_u_fld.sync_device();
        ice->srfoce_v_fld.modify_host();    ice->srfoce_v_fld.sync_device();
        ice->srfoce_temp_fld.modify_host(); ice->srfoce_temp_fld.sync_device();
        ice->srfoce_salt_fld.modify_host(); ice->srfoce_salt_fld.sync_device();
        ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
        ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
        ice->thermo.t_skin_fld.modify_host(); ice->thermo.t_skin_fld.sync_device();
        ice->thermo.thdgr_fld.modify_host();  ice->thermo.thdgr_fld.sync_device();
        }

        fesom_ice_thermodynamics_kk(ice, partit, mesh, forcing, jra, sr);

        /* OUT rail — the 9 consumed outputs to the host (oce_fluxes reads flx_*, h_diag reads values).
         * M7 H.2 ICERAILS: NONE of those 9 has a production host reader any more. flx_h/flx_fw are
         * read by fesom_ice_oce_fluxes_kk on the DEVICE (device->device, its own comment says so);
         * t_skin/thdgr/thdgrsn/thdgra are the thermo's cross-step RMW state, re-read on the device;
         * a/m/ms go to h_diag_kk on the device. The end-of-ice-step sync below covers every host
         * reader that actually exists. */
        if (!icerails) {
        ice->data[FESOM_ICE_AICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MSNOW].values_fld.sync_host();
        ice->thermo.t_skin_fld.sync_host();
        ice->flx_fw_fld.sync_host();  ice->flx_h_fld.sync_host();
        ice->thermo.thdgr_fld.sync_host();   ice->thermo.thdgrsn_fld.sync_host();
        ice->thermo.thdgra_fld.sync_host();
        }
        if (s_verify_icethermo)
            fesom_ice_thermodynamics_verify(ice, partit, mesh, forcing, jra, sr, step, tm, ts, ta, tt, tg);

        /* Phase C2/C3: oce_fluxes overwrites heat_flux/water_flux with the ice-mediated flx_h/flx_fw
         * and computes virtual_salt + relax_salt. M4.3d-b: ON THE DEVICE. flx_h/flx_fw are device-
         * current from the thermo above (device→device, no push). IN: push tracers S (the surface S
         * it reads; tracers const→const_cast) + forcing Ssurf. The kernel owns the 2 integrate_nod_2D
         * reductions + the 4 forcing halos (sync_host → host exchange); the ocean step's IN rail
         * re-pushes the host forcing → device (the "→host(forcing)" handoff). EOS-style verify
         * (oce_fluxes is a full overwrite from intact inputs → no capture-before). */
        /* M5.14 (S flip): S values device-resident (floor → device) - no re-push; oce_fluxes reads surface S on device. */
        forcing->Ssurf_fld.modify_host(); forcing->Ssurf_fld.sync_device();
        fesom_ice_oce_fluxes_kk(ice, partit, mesh, tracers, forcing, jra, sr);
        if (s_verify_iceflux) fesom_ice_oce_fluxes_verify(ice, partit, mesh, tracers, forcing, sr, step);
    }

    PMARK_ICE("ice_thermo+flux");
    /* Post-step diagnostic h_ice/h_snow (Fortran ice_setup_step.F90:319-330). M4.3a: ON THE
     * DEVICE. IN: push a_ice/m_ice/m_snow (the host thermo, if it ran, wrote them via the raw
     * alias). OUT: sync_host(h_ice/h_snow) for I/O + the FESOM_DIAG_MICE block below. */
    real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;     /* kept for the DEBUG diag blocks */
    real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    (void)a_ice;
    if (!icerails) {   /* M7 H.2: a/m/ms come straight from cut_off/thermo_kk — a 4th bounce */
    ice->data[FESOM_ICE_AICE].values_fld.modify_host();  ice->data[FESOM_ICE_AICE].values_fld.sync_device();
    ice->data[FESOM_ICE_MICE].values_fld.modify_host();  ice->data[FESOM_ICE_MICE].values_fld.sync_device();
    ice->data[FESOM_ICE_MSNOW].values_fld.modify_host(); ice->data[FESOM_ICE_MSNOW].values_fld.sync_device();
    }
    fesom_ice_h_diag_kk(ice, mesh);
    if (!icerails) {
    ice->h_ice_fld.sync_host();
    ice->h_snow_fld.sync_host();
    }
    if (s_verify_icemap) fesom_ice_h_diag_verify(ice, mesh, step);

    /* ============ M7 H.2 ICERAILS — THE ONE OUT RAIL THAT REPLACES ALL 67 ====================
     * Everything above kept the ice state DEVICE-authoritative through the whole chain. Here we
     * pay for exactly the host readers that actually exist downstream — no more:
     *   a/m/ms      — the I/O snapshot gather (fesom_io.cpp:370, .h_checked()), the host
     *                 fesom_cal_shortwave_rad's a_ice read when FLUXDEV is OFF, FESOM_DIAG_MICE
     *   uice/vice   — the I/O snapshot; and the HOST fesom_ice_oce_fluxes_mom, which reads them
     *                 (fesom_ice_coupling.cpp:62) whenever ICEFLUXDEV is OFF
     *   srfoce_u/v  — same host fesom_ice_oce_fluxes_mom
     *   h_ice/h_snow— FESOM_DIAG_MICE (no production I/O reader, but 2 D2H is not worth the risk)
     * 9 D2H, ~5.6 ms — against the 40 H2D + 27 D2H (~250-300 MB, ~41 ms) this lever deletes.
     *
     * These also leave a/m/ms + uice/vice SYNCED, which is exactly what the NEXT step's EVP kernel
     * needs to read on the device with no push. That is the cross-step invariant. Do not remove
     * them without re-deriving it.
     *
     * M7 H.8 LAZYSNAP — re-derived (the full derivation is above fesom_lazysnap_on() in
     * fesom_io.cpp). The EVP kernel reads a/m/ms + uice/vice on the DEVICE, and Device-
     * authoritative serves that read just as well as Synced — the invariant needed the device copy
     * current, not the host one. The host mirrors' only snapshot-cadence consumer (the I/O gather)
     * now pulls them itself; every step-cadence host reader is dead under the lever's required set
     * (ICEFLUXDEV/FLUXDEV+SWSKIP/IOACC — enforced by abort), and the census shows this whole block
     * is 7.3 ms/step of GPU idle. So under LAZYSNAP: skip all 9. */
    if (icerails && !fesom_lazysnap_on()) {
        ice->data[FESOM_ICE_AICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MICE].values_fld.sync_host();
        ice->data[FESOM_ICE_MSNOW].values_fld.sync_host();
        ice->uice_fld.sync_host();       ice->vice_fld.sync_host();
        ice->srfoce_u_fld.sync_host();   ice->srfoce_v_fld.sync_host();
        ice->h_ice_fld.sync_host();      ice->h_snow_fld.sync_host();
    }

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

    PMARK_ICE("ice_hdiag");   /* M5.6: close the last ice phase bucket */
#undef PMARK_ICE

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
