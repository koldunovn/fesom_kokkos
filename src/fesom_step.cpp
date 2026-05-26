/*
 * Phase 1 timestep driver — wires the substeps in FRESH_START.md §5 order.
 * For Phase 4 (MPI) every kernel that writes a field other ranks read is
 * followed by a halo exchange. Cheat sheet documented inline.
 */
#include <cmath>   /* sqrt used in the FESOM_DIAG_SPREAD block */
#include "fesom_step.h"
#include "fesom_ale.h"
#include "fesom_aux.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_eos.h"
#include "fesom_forcing.h"
#include "fesom_gm.h"
#include "fesom_halo.h"
#include "fesom_ice.h"
#include "fesom_kpp.h"
#include "fesom_mesh.h"
#include "fesom_momentum.h"
#include "fesom_partit.h"
#include "fesom_pp.h"
#include "fesom_ssh.h"
#include "fesom_tracer_adv.h"
#include "fesom_tracer_diff.h"
#include "fesom_tracers.h"

#include <stdlib.h>
#include <cstring>   /* strstr — FESOM_KK_VERIFY substring match (M2.1) */
#include <vector>    /* M2.2: mo_convect verify pre-kernel Kv/Av capture (host-only diagnostic) */

#ifdef FESOM_KK_SYNCCHECK
/* M1.5 plumbing proof (docs/SYNC_MAP.md §"Proving the rails"). Bounce a representative set of the
 * ocean step's evolving-state Fields through a host->device->host round-trip at the END of the
 * step, after every host kernel has written them via the raw alias. The host writes are invisible
 * to the DualView modify-flags (L14), so modify_host() FIRST; modify_device() then models the M2
 * device kernel that will write each field. On Serial/OpenMP host==device so all four calls are
 * no-ops; on CUDA each sync is a bitwise-exact deep_copy of double, so the host bytes are unchanged
 * and the run stays bit-identical (the SYNCCHECK build is gated bit-for-bit vs the golden). After
 * the round-trip every Field is back in the Synced state, so the next step's halo/I/O h_checked()
 * reads never see a Device-authoritative field at M1 — exactly the invariant the guards assert.
 * Compiled out entirely when the macro is off. */
#define FESOM_KK_BOUNCE(f) do { (f).modify_host(); (f).sync_device(); \
                                (f).modify_device(); (f).sync_host(); } while (0)
static void ocean_synccheck_roundtrip(struct fesom_dyn     *dyn,
                                      struct fesom_tracers *tracers,
                                      struct fesom_aux     *aux)
{
    /* dyn — element-vector (uv), node-3D (w/w_e/w_i), node-2D (eta/ssh) size classes */
    FESOM_KK_BOUNCE(dyn->uv_fld);
    FESOM_KK_BOUNCE(dyn->w_fld);
    FESOM_KK_BOUNCE(dyn->w_e_fld);
    FESOM_KK_BOUNCE(dyn->w_i_fld);
    FESOM_KK_BOUNCE(dyn->uvnode_fld);
    FESOM_KK_BOUNCE(dyn->eta_n_fld);
    FESOM_KK_BOUNCE(dyn->d_eta_fld);
    FESOM_KK_BOUNCE(dyn->ssh_rhs_fld);
    FESOM_KK_BOUNCE(dyn->ssh_rhs_old_fld);
    /* tracers — the active T and S channels (stride nl) */
    FESOM_KK_BOUNCE(tracers->data[FESOM_TRACER_T].values_fld);
    FESOM_KK_BOUNCE(tracers->data[FESOM_TRACER_S].values_fld);
    /* aux — EOS + mixing outputs (node-3D) and PGF (element-3D) */
    FESOM_KK_BOUNCE(aux->density_m_rho0_fld);
    FESOM_KK_BOUNCE(aux->hpressure_fld);
    FESOM_KK_BOUNCE(aux->bvfreq_fld);
    FESOM_KK_BOUNCE(aux->sw_alpha_fld);
    FESOM_KK_BOUNCE(aux->sw_beta_fld);
    FESOM_KK_BOUNCE(aux->Kv_fld);
    FESOM_KK_BOUNCE(aux->Av_fld);
    FESOM_KK_BOUNCE(aux->pgf_x_fld);
    FESOM_KK_BOUNCE(aux->pgf_y_fld);
}
#undef FESOM_KK_BOUNCE
#endif

int fesom_timestep(int                          step_n,
                   const fesom_step_ctx        *ctx,
                   struct fesom_mesh           *mesh,
                   struct fesom_aux            *aux,
                   struct fesom_dyn            *dyn,
                   struct fesom_tracers        *tracers,
                   const struct fesom_forcing  *forcing)
{
    fesom_partit *p = ctx->partit;
    int nl = mesh->nl;

    /* Phase G8 — env-gated master switch for GM/Redi.
     *   FESOM_NO_GMREDI=1 → treat ctx->gm as NULL for the rest of this step.
     *                      Skips compute_sigma_xy, compute_neutral_slope,
     *                      init_Redi_GM, fer_solve_Gamma, fer_gamma2vel,
     *                      Redi K33 in tracer diff, bolus add/sub, and
     *                      G7a/G7b. Used by G9 bit-identity off-switch test.
     * Read once and cache (matches FESOM_NO_TRADV style elsewhere). */
    static int s_gmredi_env_loaded = 0;
    static int s_no_gmredi         = 0;
    if (!s_gmredi_env_loaded) {
        const char *e = getenv("FESOM_NO_GMREDI");
        s_no_gmredi = (e && atoi(e));
        s_gmredi_env_loaded = 1;
    }
    /* Local alias so the original ctx->gm isn't mutated. */
    struct fesom_gm *gm = s_no_gmredi ? NULL : ctx->gm;

    /* Vertical-mixing scheme dispatch (mirror oce_ale.F90:3515 mix_scheme_nmb).
     *   FESOM_MIX_SCHEME=KPP (DEFAULT) → fesom_kpp_mixing (K-Profile) — the CORE2
     *                                    production scheme (mix_scheme='KPP'),
     *                                    validated end-to-end (K0-K10).
     *   FESOM_MIX_SCHEME=PP            → fesom_pp_mixing (Pacanowski-Philander), opt-out
     * Env knob mirroring the FESOM_NO_GMREDI pattern. mo_convect runs after either
     * scheme (Fortran calls it in both branches, :3524 / :3531). */
    static int s_mix_env_loaded = 0;
    static int s_use_kpp        = 1;
    if (!s_mix_env_loaded) {
        const char *e = getenv("FESOM_MIX_SCHEME");
        s_use_kpp = !(e && (e[0] == 'P' || e[0] == 'p'));   /* default KPP; =PP to opt out */
        s_mix_env_loaded = 1;
    }

    /* M2.1 per-kernel gate: FESOM_KK_VERIFY=eos runs the host C twin alongside the Kokkos EOS
     * kernels and asserts max|Δ|==0 on Serial (the in-binary exp1_compare_bidiff). Substring match
     * so FESOM_KK_VERIFY=eos,pp,... works. Cached like the other env knobs. */
    static int s_verify_loaded = 0;
    static int s_verify_eos    = 0;
    static int s_verify_pp     = 0;
    static int s_verify_kpp    = 0;
    static int s_verify_pgf    = 0;   /* M2.4 substep 2 */
    static int s_verify_ivisc  = 0;   /* M2.4 substep 6 */
    static int s_verify_vfilt  = 0;   /* M2.4 substep 5 */
    static int s_verify_vrhs   = 0;   /* M2.4 substep 4 */
    static int s_verify_ale    = 0;   /* M2.5 substeps 12/14 */
    static int s_verify_gm     = 0;   /* M2.5b substep 1b */
    static int s_verify_tradv  = 0;   /* M2.6 substep 13 (FCT) */
    static int s_verify_trdiff = 0;   /* M2.7 substep 13b (tracer diffusion) */
    if (!s_verify_loaded) {
        const char *e = getenv("FESOM_KK_VERIFY");
        s_verify_eos = (e && strstr(e, "eos")) ? 1 : 0;
        s_verify_kpp = (e && strstr(e, "kpp")) ? 1 : 0;   /* safe: kpp is not a substring of pp */
        s_verify_pgf = (e && strstr(e, "pgf")) ? 1 : 0;   /* M2.4: no substring collision */
        s_verify_ivisc = (e && strstr(e, "ivisc")) ? 1 : 0;   /* M2.4: distinct from vfilt */
        s_verify_vfilt = (e && strstr(e, "vfilt")) ? 1 : 0;   /* M2.4: distinct from ivisc */
        s_verify_vrhs = (e && strstr(e, "vrhs")) ? 1 : 0;     /* M2.4: no substring collision */
        s_verify_ale = (e && strstr(e, "ale")) ? 1 : 0;       /* M2.5: no substring collision */
        s_verify_gm  = (e && strstr(e, "gm"))  ? 1 : 0;       /* M2.5b: no substring collision (gm ⊄ any key) */
        s_verify_tradv = (e && strstr(e, "tradv")) ? 1 : 0;   /* M2.6: distinct token (no collision either way, L25) */
        s_verify_trdiff = (e && strstr(e, "trdiff")) ? 1 : 0; /* M2.7: distinct token — ⊄ tradv and tradv ⊄ trdiff (L25) */
        /* Match the "pp" token but NOT the "pp" inside "kpp" (sibling gate key): scan for
         * "pp" not immediately preceded by 'k'. (eos/kpp use a plain strstr; pp needs the
         * guard because kpp is a substring superset — lesson L25.) */
        s_verify_pp = 0;
        if (e) {
            for (const char *q = strstr(e, "pp"); q; q = strstr(q + 2, "pp")) {
                if (q == e || q[-1] != 'k') { s_verify_pp = 1; break; }
            }
        }
        s_verify_loaded = 1;
    }

    /*  1. EOS + hydrostatic pressure + N² — M2.1: the FIRST device kernels.
     *  SYNC_MAP §1 INPUT rail: tracers T/S and mesh hnode are host-written via the raw alias each
     *  step, which the DualView cannot see (L14), so mark them host-dirty and push to the device
     *  before the kernels. Mesh Z/ulevels/nlevels are set-once + already device-current
     *  (mesh_sync_geometry_device). On Serial/OpenMP host==device so every sync below is a no-op
     *  (run stays bit-identical); the rail only moves bytes on CUDA. */
    {
        auto &tT = tracers->data[FESOM_TRACER_T].values_fld;
        auto &tS = tracers->data[FESOM_TRACER_S].values_fld;
        tT.modify_host(); tT.sync_device();
        tS.modify_host(); tS.sync_device();
        mesh->hnode_fld.modify_host(); mesh->hnode_fld.sync_device();
    }
    fesom_pressure_bv_kk(tracers, mesh, aux);   /* device: density_m_rho0/hpressure/bvfreq/dbsfc/MLD1 */
    /* sw_alpha / sw_beta — McDougall (1987). Needed by GM/Redi (and KPP).
     * Mirror of Fortran oce_ale.F90:3475 sw_alpha_beta. */
    fesom_compute_sw_alpha_beta_kk(tracers, mesh, aux);   /* device: sw_alpha/sw_beta */
    /* SYNC_MAP §1 OUTPUT rail: the kernels marked these modify_device(); pull them to the host
     * before the host halo exchanges + smooth_nod3D + the GM/PP/KPP consumers that read via the raw
     * alias. dbsfc has no halo but KPP bldepth reads it on host; MLD1_ind has no halo but GM
     * init_Redi_GM reads it on host — sync both here too. (No-ops on Serial/OpenMP.) */
    aux->density_m_rho0_fld.sync_host();
    aux->hpressure_fld.sync_host();
    aux->bvfreq_fld.sync_host();
    aux->sw_alpha_fld.sync_host();
    aux->sw_beta_fld.sync_host();
    aux->dbsfc_fld.sync_host();
    aux->MLD1_ind_fld.sync_host();
    /* In-binary per-kernel gate (Serial max|Δ|==0); non-intrusive (restores the Kokkos result).
     * Reads aux host-current (just synced above). */
    if (s_verify_eos) fesom_eos_verify(tracers, mesh, aux, step_n);
    /* exchange the per-node 3D outputs (Fortran oce_ale_pressure_bv.F90:2844-).
     * All five EOS-output halos now read through Field::h_checked(): pointer-identical to the raw
     * alias (production unchanged), but under -DFESOM_KK_SYNCCHECK each aborts if the field is still
     * device-authoritative — i.e. if the output sync_host() above were forgotten now that EOS runs
     * on the device (docs/SYNC_MAP.md §1). */
    fesom_exchange_nod3D(aux->density_m_rho0_fld.h_checked(), nl, p);
    fesom_exchange_nod3D(aux->hpressure_fld.h_checked(),      nl, p);
    fesom_exchange_nod3D(aux->bvfreq_fld.h_checked(),         nl, p);
    fesom_exchange_nod3D(aux->sw_alpha_fld.h_checked(),       nl, p);
    fesom_exchange_nod3D(aux->sw_beta_fld.h_checked(),        nl, p);
    /* horizontal N² smoothing — N2smth_h=.true., N2smth_hidx=1 (Fortran
     * oce_ale_pressure_bv.F90:499 smooth_nod3D(bvfreq,1)). The exchange above
     * populated the halo bvfreq the sweep reads (Fortran fills bvfreq on
     * myDim+eDim then smooths); fesom_smooth_nod3D re-exchanges after. Must precede
     * every bvfreq consumer (GM, PP/KPP, mo_convect). */
    fesom_smooth_nod3D(aux->bvfreq_fld.h_checked(), nl, 1, mesh, p);

    /*  1b. GM/Redi prerequisites + per-step coefficient builder + streamfunction
     *      solve + bolus velocity reconstruction — M2.5b: device kernels.
     *      SYNC_MAP §2 row 1b. ⚠️ GM is ON by default in the pi smoke (s_no_gmredi=0,
     *      L34), so this whole chain runs every step on the golden path. Outputs
     *      sigma_xy / neutral_slope / slope_tapered / fer_tapfac / fer_K / fer_C /
     *      Ki / fer_gamma / dyn->fer_uv. fer_uv feeds the DEVICE vert_vel (substep
     *      12b) + the bolus add (13a); slope_tapered/Ki feed the Redi (substep 13).
     *
     *      The five kernels flow DEVICE→DEVICE (each reads its upstream's OWNED
     *      output on the device); the C twins' internal halos move to the driver
     *      (the ALE/M2.5 pattern — none of these is a D21 internal bracket). Only
     *      fer_gamma is re-pushed to the device (fer_gamma2vel reads it at HALO
     *      vertices, L30). All five are race-free maps/gathers/per-node TDMAs →
     *      Serial AND OpenMP bit-identical (no scatter). On Serial/OpenMP host==
     *      device so every sync is a no-op. */
    if (gm) {
        const int nl1 = nl - 1;
        /* IN rail (L28 — sync every input the chain reads that a host op last touched):
         * bvfreq = the L27 device→host(smooth_nod3D)→device hand-off (substep 1);
         * sw_alpha/sw_beta were halo'd on the host (substep 1); T/S were pushed by the
         * substep-1 EOS rail + unchanged, re-pushed here for self-containment; hnode_new
         * (last step's 12a) + helem (last step's 14) are evolving mesh, host-current. */
        aux->bvfreq_fld.modify_host();   aux->bvfreq_fld.sync_device();
        aux->sw_alpha_fld.modify_host(); aux->sw_alpha_fld.sync_device();
        aux->sw_beta_fld.modify_host();  aux->sw_beta_fld.sync_device();
        {
            auto &tT = tracers->data[FESOM_TRACER_T].values_fld;
            auto &tS = tracers->data[FESOM_TRACER_S].values_fld;
            tT.modify_host(); tT.sync_device();
            tS.modify_host(); tS.sync_device();
        }
        mesh->hnode_new_fld.modify_host(); mesh->hnode_new_fld.sync_device();
        mesh->helem_fld.modify_host();     mesh->helem_fld.sync_device();

        /* (G2b) density gradient on neutral surfaces. */
        fesom_compute_sigma_xy_kk(aux, tracers, mesh, gm);
        gm->sigma_xy_fld.sync_host();
        if (s_verify_gm) fesom_gm_sigma_xy_verify(aux, tracers, mesh, gm, p, step_n);
        fesom_halo_exchange(gm->sigma_xy_fld.h_checked(), FESOM_HALO_NOD2D, nl, 2, p);

        /* (G2b) neutral slope + ODM95 tapering. */
        fesom_compute_neutral_slope_kk(aux, mesh, gm);
        gm->neutral_slope_fld.sync_host();
        gm->slope_tapered_fld.sync_host();
        gm->fer_tapfac_fld.sync_host();   /* read by init_redi (owned) + the verify; not halo'd */
        if (s_verify_gm) fesom_gm_neutral_slope_verify(aux, mesh, gm, p, step_n);
        fesom_halo_exchange(gm->neutral_slope_fld.h_checked(), FESOM_HALO_NOD2D, nl1, 3, p);
        fesom_halo_exchange(gm->slope_tapered_fld.h_checked(), FESOM_HALO_NOD2D, nl1, 3, p);

        /* (G3) per-step GM/Redi coefficient builder. */
        fesom_init_redi_gm_kk(aux, mesh, gm);
        gm->fer_scal_fld.sync_host();
        gm->fer_K_fld.sync_host();
        gm->fer_C_fld.sync_host();
        gm->Ki_fld.sync_host();
        if (s_verify_gm) fesom_gm_init_redi_verify(aux, mesh, gm, p, step_n);
        fesom_exchange_nod2D(gm->fer_C_fld.h_checked(), p);
        fesom_halo_exchange(gm->fer_K_fld.h_checked(), FESOM_HALO_NOD2D, nl, 1, p);
        fesom_halo_exchange(gm->Ki_fld.h_checked(),    FESOM_HALO_NOD2D, nl, 1, p);

        /* (G4) streamfunction solve (per-node TDMA). */
        fesom_fer_solve_gamma_kk(aux, mesh, gm);
        gm->fer_gamma_fld.sync_host();
        if (s_verify_gm) fesom_gm_solve_gamma_verify(aux, mesh, gm, p, step_n);
        fesom_halo_exchange(gm->fer_gamma_fld.h_checked(), FESOM_HALO_NOD2D, nl, 2, p);
        /* fer_gamma2vel reads fer_gamma at HALO vertices → re-push the halo'd host
         * values to the device (L30, the bvfreq/hpressure cross-op pattern). */
        gm->fer_gamma_fld.modify_host(); gm->fer_gamma_fld.sync_device();

        /* (G4) bolus velocity reconstruction (vertex→element). */
        fesom_fer_gamma2vel_kk(dyn, mesh, gm);
        dyn->fer_uv_fld.sync_host();
        if (s_verify_gm) fesom_gm_gamma2vel_verify(dyn, mesh, gm, p, step_n);
        fesom_halo_exchange(dyn->fer_uv_fld.h_checked(), FESOM_HALO_ELEM2D_FULL, nl, 2, p);
    }

    /*  2. PGF (linfs + full cells) — M2.4: device kernel.
     *  SYNC_MAP §2 row 2. INPUT rail: hpressure was produced on the device (substep 1),
     *  then sync_host'd (L163) + halo-exchanged on the host (L178) — the halo write is
     *  invisible to the DualView (L14/L27), so push the now-halo-current host hpressure
     *  back to the device with modify_host()+sync_device() (NOT a bare sync_device, which
     *  would feed the kernel the pre-halo device bytes). gradient_sca/elem_nodes/ulevels/
     *  nlevels are set-once device-current (mesh_sync_geometry_device). No-op on Serial/OpenMP. */
    aux->hpressure_fld.modify_host(); aux->hpressure_fld.sync_device();
    fesom_pressure_force_linfs_fullcell_kk(mesh, aux);   /* device: pgf_x, pgf_y (elem) */
    /* OUTPUT rail: pull pgf to host before the elem3D halos (read via h_checked). */
    aux->pgf_x_fld.sync_host();
    aux->pgf_y_fld.sync_host();
    if (s_verify_pgf) fesom_pressure_force_verify(mesh, aux, step_n);
    fesom_exchange_elem3D(aux->pgf_x_fld.h_checked(), nl, p);
    fesom_exchange_elem3D(aux->pgf_y_fld.h_checked(), nl, p);

    /*  3. mixing: UVnode → (PP or KPP) → convective adjustment — M2.2: device kernels.
     *     Dispatch mirrors oce_ale.F90:3515-3531 (mix_scheme_nmb 1=KPP / 2=PP);
     *     mo_convect is shared (called after either scheme). compute_vel_nodes +
     *     mo_convect always run on device; pp_mixing only in the PP branch (KPP is the
     *     default). KPP itself stays a HOST kernel until M2.3 — so on the default path
     *     uvnode round-trips device→host (kernel→halo+KPP), and Kv/Av round-trip
     *     host→device→host (KPP→mo_convect→halos). See docs/SYNC_MAP.md §2 row 3. */

    /* INPUT rail (compute_vel_nodes): dyn->uv is host-written by last step's update_vel
     * + bolus (raw alias, invisible to the DualView, L14) → modify_host()+sync_device().
     * Mesh CSR / elem_area / levels are set-once device-current. (No-op on Serial/OpenMP.) */
    dyn->uv_fld.modify_host(); dyn->uv_fld.sync_device();
    fesom_compute_vel_nodes_kk(mesh, dyn);              /* device: writes dyn->uvnode (owned) */
    dyn->uvnode_fld.sync_host();                        /* OUT rail: before the halo + host KPP/PP */
    if (s_verify_pp) fesom_compute_vel_nodes_verify(mesh, dyn, step_n);
    fesom_halo_exchange(dyn->uvnode_fld.h_checked(), FESOM_HALO_NOD3D, nl, 2, p);

    if (s_use_kpp) {
        /* KPP on device (M2.3). It writes aux->Av (elements) + the single aux->Kv (T-channel,
         * oce_ale.F90:3518-3522) and does its OWN internal halo exchanges (bracketed inside
         * fesom_kpp_mixing_kk). INPUT rail: push every host-authoritative input KPP reads to
         * the device — the Serial gate can't catch a missing sync_device (host==device), so we
         * sync them all explicitly. bvfreq is the L27 device→host(smooth)→device hand-off;
         * forcing is host-produced; sw_alpha, sw_beta, dbsfc, uvnode, S, hnode are device-current
         * from substep 1 but re-synced for robustness. (No-op on Serial/OpenMP.) */
        aux->bvfreq_fld.modify_host();   aux->bvfreq_fld.sync_device();
        aux->sw_alpha_fld.modify_host(); aux->sw_alpha_fld.sync_device();
        aux->sw_beta_fld.modify_host();  aux->sw_beta_fld.sync_device();
        aux->dbsfc_fld.modify_host();    aux->dbsfc_fld.sync_device();
        dyn->uvnode_fld.modify_host();   dyn->uvnode_fld.sync_device();
        tracers->data[FESOM_TRACER_S].values_fld.modify_host();
        tracers->data[FESOM_TRACER_S].values_fld.sync_device();
        mesh->hnode_fld.modify_host();   mesh->hnode_fld.sync_device();
        /* forcing is a const input to the step; the sync_device is a pure coherence op (moves
         * host→device, no logical mutation) → const_cast is safe and localized here. */
        auto *fnc = const_cast<struct fesom_forcing *>(forcing);
        fnc->stress_node_surf_fld.modify_host(); fnc->stress_node_surf_fld.sync_device();
        fnc->heat_flux_fld.modify_host();        fnc->heat_flux_fld.sync_device();
        fnc->water_flux_fld.modify_host();       fnc->water_flux_fld.sync_device();
        fnc->sw_3d_fld.modify_host();            fnc->sw_3d_fld.sync_device();

        fesom_kpp_mixing_kk(ctx->kpp, aux, tracers, forcing, dyn, mesh, p);

        /* OUT rail: pull Av/Kv to host — mo_convect's rail (below) + the L212/213 halos read
         * them host-authoritative (uniform with the PP branch), and the verify needs them host. */
        aux->Av_fld.sync_host();
        aux->Kv_fld.sync_host();
        if (s_verify_kpp) fesom_kpp_verify(ctx->kpp, aux, tracers, forcing, dyn, mesh, p, step_n);
    } else {
        /* PP branch (opt-in; FESOM_MIX_SCHEME=PP). INPUT rail: uvnode (host-halo'd above)
         * + bvfreq (host-written by smooth_nod3D, substep 1) → device. */
        dyn->uvnode_fld.modify_host(); dyn->uvnode_fld.sync_device();
        aux->bvfreq_fld.modify_host(); aux->bvfreq_fld.sync_device();
        fesom_pp_mixing_kk(mesh, dyn, aux);             /* device: Kv (node), Av (elem) */
        aux->Kv_fld.sync_host();                        /* OUT rail: before the Kv/Av halos */
        aux->Av_fld.sync_host();
        if (s_verify_pp) fesom_pp_mixing_verify(mesh, dyn, aux, step_n);
        fesom_exchange_nod3D (aux->Kv_fld.h_checked(), nl, p);
        fesom_exchange_elem3D(aux->Av_fld.h_checked(), nl, p);
    }

    /* mo_convect (always-on, after either scheme). INPUT rail: bvfreq is host-written by
     * smooth_nod3D (substep 1) and mo_convect is its FIRST device reader after that — so
     * modify_host()+sync_device(), NOT a bare sync_device() (L14). Kv/Av are host-
     * authoritative too (KPP wrote them on host on the default path; the PP branch
     * sync_host()'d + halo'd them above) → same. Capture the PRE-kernel Kv/Av for the
     * in-place-modify verify oracle (mo_convect raises Kv/Av in place). */
    std::vector<real_t> mc_Kv_in, mc_Av_in;
    if (s_verify_pp) {
        const size_t nKv = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
        const size_t nAv = (size_t)mesh->myDim_elem2D * (size_t)nl;
        mc_Kv_in.assign(aux->Kv, aux->Kv + nKv);
        mc_Av_in.assign(aux->Av, aux->Av + nAv);
    }
    aux->bvfreq_fld.modify_host(); aux->bvfreq_fld.sync_device();
    aux->Kv_fld.modify_host();     aux->Kv_fld.sync_device();
    aux->Av_fld.modify_host();     aux->Av_fld.sync_device();
    fesom_mo_convect_kk(mesh, aux);                     /* device: Kv, Av */
    aux->Kv_fld.sync_host();                            /* OUT rail: before the Kv/Av halos */
    aux->Av_fld.sync_host();
    if (s_verify_pp) fesom_mo_convect_verify(mesh, aux, step_n, mc_Kv_in.data(), mc_Av_in.data());
    fesom_exchange_nod3D (aux->Kv_fld.h_checked(), nl, p);
    fesom_exchange_elem3D(aux->Av_fld.h_checked(), nl, p);

    /*  4. momentum RHS (Coriolis AB2 + SSH gradient + PGF) — M2.4: device kernel.
     *  ⚠️ AB2 eps=0.1 preserved in the kernel (PORTING_LESSONS §1, the dt=1800 trap).
     *  compute_vel_rhs embeds momentum_adv_scalar: an edge→node SCATTER (atomic_add, D22)
     *  + an INTERNAL uvnode_rhs nod3D halo (D21, owned by the kernel). SYNC_MAP §2 row 4 + §6.
     *  IN rail (L28): uv, uv_rhsAB (part i reads it; the read-modify-write target), eta_n (SSH
     *  grad, read at 3 vertices incl halo), pgf_x/pgf_y (substep-2 device output sync_host'd +
     *  halo'd), w_e (vert adv), the evolving mesh hnode (vert adv). uvnode_rhs is device scratch;
     *  set-once mesh (gradient_sca/coriolis/elem_area/areasvol/edges/edge_tri/edge_cross/
     *  nod_in_elem2D/elem_nodes/levels) device-current. */
    std::vector<real_t> vrhs_uv_rhsAB_in;
    if (s_verify_vrhs) {
        const size_t nvec = (size_t)mesh->myDim_elem2D * (size_t)nl * 2;
        vrhs_uv_rhsAB_in.assign(dyn->uv_rhsAB, dyn->uv_rhsAB + nvec);   /* L26 capture-before (part i reads it) */
    }
    dyn->uv_fld.modify_host();       dyn->uv_fld.sync_device();
    dyn->uv_rhsAB_fld.modify_host(); dyn->uv_rhsAB_fld.sync_device();
    dyn->eta_n_fld.modify_host();    dyn->eta_n_fld.sync_device();
    dyn->w_e_fld.modify_host();      dyn->w_e_fld.sync_device();
    aux->pgf_x_fld.modify_host();    aux->pgf_x_fld.sync_device();
    aux->pgf_y_fld.modify_host();    aux->pgf_y_fld.sync_device();
    mesh->hnode_fld.modify_host();   mesh->hnode_fld.sync_device();
    fesom_compute_vel_rhs_kk(mesh, aux, dyn, /*is_first_step=*/(step_n == 1), p);
    dyn->uv_rhs_fld.sync_host();     /* OUT rail: before the elem3D halos */
    dyn->uv_rhsAB_fld.sync_host();
    if (s_verify_vrhs) fesom_compute_vel_rhs_verify(mesh, aux, dyn, (step_n == 1), p, step_n,
                                                    vrhs_uv_rhsAB_in.data());
    /* uv_rhs needed by visc_filt_bidiff on halo elements */
    fesom_halo_exchange(dyn->uv_rhs_fld.h_checked(),   FESOM_HALO_ELEM3D, nl, 2, p);
    fesom_halo_exchange(dyn->uv_rhsAB_fld.h_checked(), FESOM_HALO_ELEM3D, nl, 2, p);

    /*  5. horizontal viscosity (biharmonic ∇⁴, opt_visc=7) — M2.4: device kernel.
     *     CORE2 runs opt_visc=7; the ∇⁴ damps grid-scale 2Δx modes that opt_visc=5 lets
     *     grow (required for dt=1800 stability). FIRST SCATTER kernel: edge→element via
     *     Kokkos::atomic_add (SCATTER_STRATEGY.md, D22) → Serial bit-identical (single-thread
     *     ordered), OpenMP/CUDA climate-close (≲1e-12, the D5 ladder). SYNC_MAP §2 row 5 + §6.
     *     IN rail: uv (last step's update_vel + halo + bolus) and uv_rhs (compute_vel_rhs
     *     output + halo this step; the read-modify-write target) → device (L28); u_b/v_b
     *     scratch zeroed on device in the kernel; the kernel owns the internal Uc/Vc halo (D21). */
    std::vector<real_t> vfb_uv_rhs_in;
    if (s_verify_vfilt) {
        const size_t nvec = (size_t)(mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D)
                          * (size_t)nl * 2;
        vfb_uv_rhs_in.assign(dyn->uv_rhs, dyn->uv_rhs + nvec);   /* L26 capture-before (full extent) */
    }
    dyn->uv_fld.modify_host();     dyn->uv_fld.sync_device();
    dyn->uv_rhs_fld.modify_host(); dyn->uv_rhs_fld.sync_device();
    fesom_visc_filt_bidiff_kk(mesh, dyn, p);   /* device: uv_rhs += biharmonic; internal Uc/Vc halo (D21) */
    dyn->uv_rhs_fld.sync_host();               /* OUT rail: before the elem3D halo */
    if (s_verify_vfilt) fesom_visc_filt_bidiff_verify(mesh, dyn, p, step_n, vfb_uv_rhs_in.data());
    /* uv_rhs is the final output; needed at halo for impl_vert_visc neighbour
     * reads through the TDMA SpMV. */
    fesom_halo_exchange(dyn->uv_rhs_fld.h_checked(), FESOM_HALO_ELEM3D, nl, 2, p);

    /*  6. implicit vertical viscosity TDMA — M2.4: device kernel (per-element TDMA).
     *  SYNC_MAP §2 row 6. IN rail: sync ALL inputs explicitly (L28) — uv_rhs (the
     *  read-modify-write target; host-written by visc_filt_bidiff + halo, substep 5),
     *  uv (last step's update_vel + halo + bolus), Av (KPP/PP device output, sync_host'd
     *  + halo'd substep 3), helem (evolving mesh, last step's commit + halo — NOT in the
     *  one-shot push), w_i (read at the 3 vertices incl. HALO nodes — w_e/w_i halo'd
     *  substep 12 of the prev step), forcing stress_surf (host-produced in main; const →
     *  localized const_cast). zbar + elem_nodes/ulevels/nlevels are set-once device-current. */
    std::vector<real_t> ivv_uv_rhs_in;
    if (s_verify_ivisc) {
        const size_t nvec = (size_t)mesh->myDim_elem2D * (size_t)nl * 2;
        ivv_uv_rhs_in.assign(dyn->uv_rhs, dyn->uv_rhs + nvec);   /* L26 capture-before */
    }
    dyn->uv_rhs_fld.modify_host(); dyn->uv_rhs_fld.sync_device();
    dyn->uv_fld.modify_host();     dyn->uv_fld.sync_device();
    dyn->w_i_fld.modify_host();    dyn->w_i_fld.sync_device();
    aux->Av_fld.modify_host();     aux->Av_fld.sync_device();
    mesh->helem_fld.modify_host(); mesh->helem_fld.sync_device();
    {   auto *fnc = const_cast<struct fesom_forcing *>(forcing);
        fnc->stress_surf_fld.modify_host(); fnc->stress_surf_fld.sync_device();   }
    fesom_impl_vert_visc_kk(mesh, aux, forcing, dyn);   /* device: uv_rhs */
    dyn->uv_rhs_fld.sync_host();                        /* OUT rail: before the elem3D halo */
    if (s_verify_ivisc) fesom_impl_vert_visc_verify(mesh, aux, forcing, dyn, step_n, ivv_uv_rhs_in.data());
    fesom_halo_exchange(dyn->uv_rhs_fld.h_checked(), FESOM_HALO_ELEM3D, nl, 2, p);

    /*  7. SSH RHS (linfs)  */
    fesom_compute_ssh_rhs_linfs(mesh, dyn);
    fesom_exchange_nod2D(dyn->ssh_rhs, p);   /* Fortran oce_ale.F90:1954 */

    /*  8. CG SSH solve  */
    int cg_iters = fesom_ssh_solve_cg(ctx->stiff, ctx->solver, mesh, dyn);
    /* fesom_ssh_solve_cg already exchanges its own X (= d_eta) at exit
     * (Fortran solver.F90:279). For now we keep the explicit exchange here
     * because the parallel CG modifications are slice 30e, not 30d. */
    fesom_exchange_nod2D(dyn->d_eta, p);

    /*  9. velocity update with SSH-gradient correction  */
    fesom_update_vel(mesh, dyn);
    fesom_halo_exchange(dyn->uv_fld.h_checked(), FESOM_HALO_ELEM3D, nl, 2, p);   /* M1.5 guarded (see §1) */

    /* 10. transport-divergence → ssh_rhs_old, then hbar update  */
    fesom_compute_hbar(mesh, dyn);
    fesom_exchange_nod2D(dyn->ssh_rhs_old, p);   /* Fortran oce_ale.F90:2078 */
    fesom_exchange_nod2D(mesh->hbar,       p);   /* Fortran oce_ale.F90:2102 */

    /* DEBUG (FESOM_DIAG_SSHSLV=<gid>): dump ssh_rhs / d_eta / hbar at one node,
       matching the Fortran [FSSH] format, to compare the implicit free-surface
       DAMPING STRENGTH (|d_eta|/|ssh_rhs|) C-vs-Fortran. A systematically
       larger C d_eta = the stiffness operator under-damps the 2dx mode. */
    if (getenv("FESOM_DIAG_SSHSLV")) {
        int tgid = atoi(getenv("FESOM_DIAG_SSHSLV"));
        for (int row = 0; row < mesh->myDim_nod2D; ++row) {
            if (p->myList_nod2D[row] != tgid) continue;
            fprintf(stderr, "[sshslv r%d] gid %d step %d ssh_rhs=%.7e d_eta=%.7e hbar=%.7e\n",
                    p->mype, tgid, step_n, (double)dyn->ssh_rhs[row],
                    (double)dyn->d_eta[row], (double)mesh->hbar[row]);
            break;
        }
        fflush(stderr);
    }

    /* DEBUG (FESOM_DIAG_SPREAD=<gid>): per-step 1-ring spread of the fields that
       could seed the dt=1800 central-Arctic 2dx blow-up, so we can see WHICH one
       erupts first (SSH-solve side vs element-velocity side) and WHEN. For the
       target node: hbar/d_eta/ssh_rhs spread over the 1-ring (node + edge
       neighbours via surrounding elements), plus the surface |uv| min/max over
       the surrounding ELEMENTS (a checkerboard in element velocity = the 2dx). */
    if (getenv("FESOM_DIAG_SPREAD")) {
        int tgid = atoi(getenv("FESOM_DIAG_SPREAD"));
        for (int row = 0; row < mesh->myDim_nod2D; ++row) {
            if (p->myList_nod2D[row] != tgid) continue;
            double hmin = mesh->hbar[row],  hmax = hmin;
            double emin = dyn->d_eta[row],  emax = emin;
            double rmin = dyn->ssh_rhs[row], rmax = rmin;
            double vmin = 1e30, vmax = -1e30;   /* surface |uv| over surrounding elems */
            double pmin = 1e30, pmax = -1e30;   /* surface |PGF| over surrounding elems */
            double qmin = 1e30, qmax = -1e30;   /* surface |uv_rhs| over surrounding elems */
            int o0 = mesh->nod_in_elem2D_offsets[row];
            int o1 = mesh->nod_in_elem2D_offsets[row + 1];
            for (int k = o0; k < o1; ++k) {
                int e = mesh->nod_in_elem2D[k];
                double u = dyn->uv[FESOM_ELEMVEC(e, 0, nl) + 0];
                double v = dyn->uv[FESOM_ELEMVEC(e, 0, nl) + 1];
                double spd = sqrt(u*u + v*v);
                if (spd < vmin) vmin = spd;
                if (spd > vmax) vmax = spd;
                double px = aux->pgf_x[FESOM_ELEM3D(e, 0, nl)];
                double py = aux->pgf_y[FESOM_ELEM3D(e, 0, nl)];
                double pmag = sqrt(px*px + py*py);
                if (pmag < pmin) pmin = pmag;
                if (pmag > pmax) pmax = pmag;
                double qx = dyn->uv_rhs[FESOM_ELEMVEC(e, 0, nl) + 0];
                double qy = dyn->uv_rhs[FESOM_ELEMVEC(e, 0, nl) + 1];
                double qmag = sqrt(qx*qx + qy*qy);
                if (qmag < qmin) qmin = qmag;
                if (qmag > qmax) qmax = qmag;
                for (int j = 0; j < 3; ++j) {
                    int nd = mesh->elem_nodes[3*e + j];
                    if (mesh->hbar[nd]  < hmin) hmin = mesh->hbar[nd];
                    if (mesh->hbar[nd]  > hmax) hmax = mesh->hbar[nd];
                    if (dyn->d_eta[nd]  < emin) emin = dyn->d_eta[nd];
                    if (dyn->d_eta[nd]  > emax) emax = dyn->d_eta[nd];
                    if (dyn->ssh_rhs[nd] < rmin) rmin = dyn->ssh_rhs[nd];
                    if (dyn->ssh_rhs[nd] > rmax) rmax = dyn->ssh_rhs[nd];
                }
            }
            fprintf(stderr, "[spread r%d] gid %d step %d hbar_sprd=%.4e deta_sprd=%.4e "
                    "sshrhs_sprd=%.4e velsurf[%.3e,%.3e] velsprd=%.3e pgf_sprd=%.4e "
                    "uvrhs_sprd=%.4e\n",
                    p->mype, tgid, step_n, hmax - hmin, emax - emin,
                    rmax - rmin, vmin, vmax, vmax - vmin, pmax - pmin, qmax - qmin);
            break;
        }
        fflush(stderr);
    }

    /* 11. eta_n inline (oce_ale.F90:3771-3775).
          eta_n = α·hbar + (1-α)·hbar_old, but ONLY where ulevels_nod2D == 1.  */
    {
        const real_t alpha = (real_t)FESOM_PHASE1_ALPHA;
        int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
        for (int n = 0; n < N; ++n) {
            if (mesh->ulevels_nod2D[n] == 1) {
                dyn->eta_n[n] = alpha * mesh->hbar[n]
                              + (1.0 - alpha) * mesh->hbar_old[n];
            }
        }
    }
    /* eta_n already covers myDim+eDim because hbar/hbar_old are exchanged. */

    /* 12. ALE step (linfs) — M2.5: device kernels. SYNC_MAP §2 row 12. Each kernel: IN rail
     *  (modify_host()+sync_device() the host-written/halo'd inputs, L28), device kernel
     *  (mod_dev outputs), OUT rail (sync_host before the host halo via h_checked). The host
     *  halo between successive kernels makes the data bounce device→host(halo)→device — the
     *  substep-1/3 rail pattern. None of these has an INTERNAL halo (every exchange_nod is a
     *  driver halo) → no D21 bracket. Default golden path runs gm=0 → the fer_* branches are
     *  dead (preserved verbatim). On Serial/OpenMP host==device so every sync is a no-op. */

    /* 12a. thickness: hnode_new = hnode. IN: hnode (evolving mesh, host-written/halo'd by last
     *  step's commit). OUT: sync_host(hnode_new) — it is read on the HOST by the tracer
     *  advection/diffusion (substeps 13/13b) and stays Synced for the device cflz/commit. */
    mesh->hnode_fld.modify_host(); mesh->hnode_fld.sync_device();
    fesom_ale_thickness_linfs_kk(mesh);
    mesh->hnode_new_fld.sync_host();
    if (s_verify_ale) fesom_ale_thickness_verify(mesh, step_n);
    /* hnode_new = hnode (no exchange needed; both already cover halo). */

    /* 12b. vertical velocity. IN: uv (update_vel+halo, substep 9), helem (evolving mesh),
     *  fer_uv (GM only). EDGE→NODE SCATTER (atomic_add, D22) + per-node level cumsum.
     *  OUT: sync_host(w[,fer_w]) before the halo. */
    dyn->uv_fld.modify_host();     dyn->uv_fld.sync_device();
    mesh->helem_fld.modify_host(); mesh->helem_fld.sync_device();
    if (gm) { dyn->fer_uv_fld.modify_host(); dyn->fer_uv_fld.sync_device(); }
    fesom_ale_vert_vel_linfs_kk(mesh, dyn, gm ? 1 : 0);
    dyn->w_fld.sync_host();
    if (gm) dyn->fer_w_fld.sync_host();
    if (s_verify_ale) fesom_ale_vert_vel_verify(mesh, dyn, gm ? 1 : 0, step_n);
    fesom_exchange_nod3D(dyn->w_fld.h_checked(), nl, p);     /* Fortran oce_ale.F90:2679 */
    if (gm) {
        /* Mirror Fortran oce_ale.F90:2681 — exchange_nod(fer_Wvel). */
        fesom_exchange_nod3D(dyn->fer_w_fld.h_checked(), nl, p);
    }

    /* 12c. vertical CFL. IN: w (just halo'd → host-current), hnode_new (Synced from 12a).
     *  Per-node accumulation into the node's OWN column → race-free (NOT a scatter).
     *  OUT: sync_host(cfl_z) before the halo. */
    dyn->w_fld.modify_host(); dyn->w_fld.sync_device();
    mesh->hnode_new_fld.sync_device();   /* no-op: Synced from 12a; documents the dependency */
    fesom_ale_compute_cflz_kk(mesh, dyn);
    dyn->cfl_z_fld.sync_host();
    if (s_verify_ale) fesom_ale_compute_cflz_verify(mesh, dyn, step_n);
    fesom_exchange_nod3D(dyn->cfl_z_fld.h_checked(), nl, p);

    /* 12d. w-split (⚠️ use_wsplit=.false. preserved → w_e=w, w_i=0). IN: cfl_z (just halo'd),
     *  w (Synced from 12c IN; cflz did not write w). Pure per-(n,nz) map. OUT: sync_host(w_e,w_i). */
    dyn->cfl_z_fld.modify_host(); dyn->cfl_z_fld.sync_device();
    dyn->w_fld.sync_device();   /* no-op: Synced from 12c IN; w unchanged by cflz */
    fesom_ale_compute_wvel_split_kk(mesh, dyn);
    dyn->w_e_fld.sync_host();
    dyn->w_i_fld.sync_host();
    if (s_verify_ale) fesom_ale_compute_wvel_split_verify(mesh, dyn, step_n);
    fesom_exchange_nod3D(dyn->w_e_fld.h_checked(), nl, p);
    fesom_exchange_nod3D(dyn->w_i_fld.h_checked(), nl, p);

    /* 13a. Phase G6b — bolus velocity add (Fortran oce_ale_tracer.F90:199-211).
     * Wraps BOTH advection (13) and diffusion (13b) so each tracer sees the
     * bolus-augmented velocity field. Subtracted back after diffusion (13c).
     * M2.6-c: ON THE DEVICE (uv/w/w_e += fer_uv/fer_w) — now that the device FCT
     * (substep 13) reads uv/w_e on the device, the bolus is its device producer (L36).
     * IN rail (L28): push uv/w/w_e (host-current from update_vel + the ALE w/w_e) +
     * fer_uv/fer_w (re-push — produced on device in 1b/12b then sync_host'd + halo'd on
     * the host, L30). OUT: modify_device then sync_host(uv/w/w_e) so the host mirrors
     * the augmented velocity — the FESOM_KK_VERIFY=tradv C twin reads host uv, and uv/
     * w/w_e then stay device-current (Synced, augmented) through the whole FCT region
     * until 13c restores them (the FCT only READS uv/w_e; Redi/tracer-diff don't touch
     * them). On Serial/OpenMP host==device so the kernel is the C loop, bit-identical. */
    if (gm) {
        dyn->uv_fld.modify_host();     dyn->uv_fld.sync_device();
        dyn->w_fld.modify_host();      dyn->w_fld.sync_device();
        dyn->w_e_fld.modify_host();    dyn->w_e_fld.sync_device();
        dyn->fer_uv_fld.modify_host(); dyn->fer_uv_fld.sync_device();
        dyn->fer_w_fld.modify_host();  dyn->fer_w_fld.sync_device();
        fesom_gm_bolus_apply_kk(dyn, mesh, (real_t)1.0);
        dyn->uv_fld.modify_device();   dyn->w_fld.modify_device();   dyn->w_e_fld.modify_device();
        dyn->uv_fld.sync_host();       dyn->w_fld.sync_host();       dyn->w_e_fld.sync_host();
    }

    /* 13. tracer advection: T then S.
     * FESOM_NO_TRADV=1 → skip advection (diagnostic for MPI drift hunt). */
    static int nt_adv_checked = 0, nt_adv_skip = 0;
    if (!nt_adv_checked) {
        const char *e = getenv("FESOM_NO_TRADV");
        nt_adv_skip = (e && atoi(e));
        nt_adv_checked = 1;
    }
    if (!nt_adv_skip) {
        /* M2.6-b: the FCT tracer advection (T then S) runs on the DEVICE
         * (fesom_tracer_advect_one_fct_kk) — one device island per tracer owning ~24
         * launches + its 3 internal-exchange D21 brackets (fct_LO/tr_xy/fct_plus+minus)
         * and 3 edge→node atomic_add scatters (D22). The M2.5b-c GM/Redi diffusion
         * (diff_ver + diff_hor) still runs on the DEVICE right after each FCT, += onto
         * the FCT-advected `values`. `values`/`valuesold` are read-modify-write → both
         * verifies are L26 capture-before. SYNC_MAP §2 row 13.
         *
         * FCT IN rail (unconditional, L28): the device FCT reads the bolus-augmented
         * dyn->uv / dyn->w_e (13a wrote them on host this step) and the evolving mesh
         * hnode / hnode_new / helem; push all five once (shared across T and S — the FCT
         * and Redi only READ them). The GM/Redi shared rail then adds slope_tapered/Ki
         * (re-pushed: diff_hor reads them at HALO edge-endpoints, L30). Per-tracer
         * values/valuesold are pushed right before each FCT. On Serial/OpenMP host==
         * device so every sync is a no-op (the run stays bit-identical). */
        const int N_redi = mesh->myDim_nod2D + mesh->eDim_nod2D;
        dyn->uv_fld.modify_host();          dyn->uv_fld.sync_device();
        dyn->w_e_fld.modify_host();         dyn->w_e_fld.sync_device();
        mesh->hnode_fld.modify_host();      mesh->hnode_fld.sync_device();
        mesh->hnode_new_fld.modify_host();  mesh->hnode_new_fld.sync_device();
        mesh->helem_fld.modify_host();      mesh->helem_fld.sync_device();
        if (gm) {
            gm->slope_tapered_fld.modify_host(); gm->slope_tapered_fld.sync_device();
            gm->Ki_fld.modify_host();            gm->Ki_fld.sync_device();
        }

        /* ---- T ---- */
        {
            auto &vT  = tracers->data[FESOM_TRACER_T].values_fld;
            auto &voT = tracers->data[FESOM_TRACER_T].valuesold_fld;
            vT.modify_host();  vT.sync_device();          /* FCT per-tracer IN rail */
            voT.modify_host(); voT.sync_device();
            std::vector<real_t> fct_pre_v, fct_pre_vo;    /* L26 capture-before (pre-FCT inputs) */
            if (s_verify_tradv) {
                fct_pre_v.assign (tracers->data[FESOM_TRACER_T].values,
                                  tracers->data[FESOM_TRACER_T].values    + (size_t)N_redi * nl);
                fct_pre_vo.assign(tracers->data[FESOM_TRACER_T].valuesold,
                                  tracers->data[FESOM_TRACER_T].valuesold + (size_t)N_redi * nl);
            }
            fesom_tracer_advect_one_fct_kk(ctx->tra_sc, FESOM_TRACER_T, mesh, dyn, tracers, p);
            vT.sync_host();  voT.sync_host();             /* OUT rail: Redi rail + halo + next step read them */
            if (s_verify_tradv) fesom_tracer_fct_verify(ctx->tra_sc, FESOM_TRACER_T, mesh, dyn,
                                                        tracers, p, step_n, fct_pre_v, fct_pre_vo);
        }
        if (gm) {
            auto &vT  = tracers->data[FESOM_TRACER_T].values_fld;
            auto &voT = tracers->data[FESOM_TRACER_T].valuesold_fld;
            vT.modify_host();  vT.sync_device();          /* Redi IN: post-FCT values (host-current) */
            voT.modify_host(); voT.sync_device();
            std::vector<real_t> redi_pre;     /* L26 capture-before (post-FCT, pre-Redi) */
            if (s_verify_gm) redi_pre.assign(tracers->data[FESOM_TRACER_T].values,
                                             tracers->data[FESOM_TRACER_T].values + (size_t)N_redi * nl);
            fesom_diff_ver_part_redi_expl_kk(FESOM_TRACER_T, gm, mesh, tracers, p);
            fesom_diff_part_hor_redi_kk     (FESOM_TRACER_T, gm, mesh, tracers, p);
            vT.sync_host();                   /* OUT rail: before the host halo */
            if (s_verify_gm) fesom_gm_redi_verify(FESOM_TRACER_T, gm, aux, mesh, tracers, p, step_n, redi_pre);
        }
        fesom_exchange_nod3D(tracers->data[FESOM_TRACER_T].values_fld.h_checked(), nl, p);   /* M1.5 guarded */

        /* ---- S ---- */
        {
            auto &vS  = tracers->data[FESOM_TRACER_S].values_fld;
            auto &voS = tracers->data[FESOM_TRACER_S].valuesold_fld;
            vS.modify_host();  vS.sync_device();
            voS.modify_host(); voS.sync_device();
            std::vector<real_t> fct_pre_v, fct_pre_vo;
            if (s_verify_tradv) {
                fct_pre_v.assign (tracers->data[FESOM_TRACER_S].values,
                                  tracers->data[FESOM_TRACER_S].values    + (size_t)N_redi * nl);
                fct_pre_vo.assign(tracers->data[FESOM_TRACER_S].valuesold,
                                  tracers->data[FESOM_TRACER_S].valuesold + (size_t)N_redi * nl);
            }
            fesom_tracer_advect_one_fct_kk(ctx->tra_sc, FESOM_TRACER_S, mesh, dyn, tracers, p);
            vS.sync_host();  voS.sync_host();
            if (s_verify_tradv) fesom_tracer_fct_verify(ctx->tra_sc, FESOM_TRACER_S, mesh, dyn,
                                                        tracers, p, step_n, fct_pre_v, fct_pre_vo);
        }
        if (gm) {
            auto &vS  = tracers->data[FESOM_TRACER_S].values_fld;
            auto &voS = tracers->data[FESOM_TRACER_S].valuesold_fld;
            vS.modify_host();  vS.sync_device();
            voS.modify_host(); voS.sync_device();
            std::vector<real_t> redi_pre;
            if (s_verify_gm) redi_pre.assign(tracers->data[FESOM_TRACER_S].values,
                                             tracers->data[FESOM_TRACER_S].values + (size_t)N_redi * nl);
            fesom_diff_ver_part_redi_expl_kk(FESOM_TRACER_S, gm, mesh, tracers, p);
            fesom_diff_part_hor_redi_kk     (FESOM_TRACER_S, gm, mesh, tracers, p);
            vS.sync_host();                   /* OUT rail: before the host halo */
            if (s_verify_gm) fesom_gm_redi_verify(FESOM_TRACER_S, gm, aux, mesh, tracers, p, step_n, redi_pre);
        }
        fesom_exchange_nod3D(tracers->data[FESOM_TRACER_S].values_fld.h_checked(), nl, p);
    }

    /* 13b. implicit vertical diffusion of tracers + surface heat/water flux BC.
     * M2.7: ON THE DEVICE (fesom_impl_vert_diff_tracers_kk) — the per-node implicit
     * vertical TDMA (T then S), the LAST host ocean compute in substep 13. The Thomas
     * sweep runs sequentially in level inside the per-node lambda over [NL_MAX] scratch
     * (the impl_vert_visc/fer_solve_gamma shape, L31): each node solves only its own
     * column of `values` → race-free, NO scatter → Serial AND OpenMP bit-identical.
     * SYNC_MAP §2 row 13b.
     *
     * IN rail (L28 — sync EVERY input the body reads): aux Kv (KPP/PP device output,
     * sync_host'd + halo'd substep 3 — re-push so the device owned Kv matches the host),
     * forcing heat_flux/water_flux/virtual_salt/relax_salt/sw_3d (host-produced; const →
     * localized const_cast), per-tracer values (host-current: post-Redi sync_host + the
     * two halos above), slope_tapered/Ki re-pushed if gm (the K33 augmentation, L30).
     * mesh hnode_new (Synced since 12a) + set-once area/areasvol/zbar/levels are already
     * device-current (no push). OUT: the kernel marks values modify_device; sync_host both
     * before the two nod3D halos. `values` is read-modify-write → FESOM_KK_VERIFY=trdiff
     * is L26 capture-before (both T and S). On Serial/OpenMP host==device → every sync a
     * no-op (run stays bit-identical). FESOM_NO_TRDIFF=1 → skip diffusion. */
    static int nt_dif_checked = 0, nt_dif_skip = 0;
    if (!nt_dif_checked) {
        const char *e = getenv("FESOM_NO_TRDIFF");
        nt_dif_skip = (e && atoi(e));
        nt_dif_checked = 1;
    }
    if (!nt_dif_skip) {
        std::vector<real_t> trd_pre_T, trd_pre_S;     /* L26 capture-before (pre-diffusion values) */
        if (s_verify_trdiff) {
            const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
            trd_pre_T.assign(tracers->data[FESOM_TRACER_T].values,
                             tracers->data[FESOM_TRACER_T].values + total);
            trd_pre_S.assign(tracers->data[FESOM_TRACER_S].values,
                             tracers->data[FESOM_TRACER_S].values + total);
        }
        /* IN rail (L28: sync every input the body reads). */
        aux->Kv_fld.modify_host(); aux->Kv_fld.sync_device();
        mesh->hnode_new_fld.sync_device();   /* no-op: Synced since 12a; documents the dependency */
        {   auto *fnc = const_cast<struct fesom_forcing *>(forcing);
            fnc->heat_flux_fld.modify_host();    fnc->heat_flux_fld.sync_device();
            fnc->water_flux_fld.modify_host();   fnc->water_flux_fld.sync_device();
            fnc->virtual_salt_fld.modify_host(); fnc->virtual_salt_fld.sync_device();
            fnc->relax_salt_fld.modify_host();   fnc->relax_salt_fld.sync_device();
            fnc->sw_3d_fld.modify_host();        fnc->sw_3d_fld.sync_device();   }
        tracers->data[FESOM_TRACER_T].values_fld.modify_host();
        tracers->data[FESOM_TRACER_T].values_fld.sync_device();
        tracers->data[FESOM_TRACER_S].values_fld.modify_host();
        tracers->data[FESOM_TRACER_S].values_fld.sync_device();
        if (gm) {
            gm->slope_tapered_fld.modify_host(); gm->slope_tapered_fld.sync_device();
            gm->Ki_fld.modify_host();            gm->Ki_fld.sync_device();
        }
        fesom_impl_vert_diff_tracers_kk(mesh, aux, forcing, tracers, gm);   /* device: values (T,S) */
        tracers->data[FESOM_TRACER_T].values_fld.sync_host();              /* OUT rail: before the halos */
        tracers->data[FESOM_TRACER_S].values_fld.sync_host();
        if (s_verify_trdiff) fesom_impl_vert_diff_tracers_verify(mesh, aux, forcing, tracers, gm,
                                                                 step_n, trd_pre_T, trd_pre_S);
        fesom_exchange_nod3D(tracers->data[FESOM_TRACER_T].values_fld.h_checked(), nl, p);
        fesom_exchange_nod3D(tracers->data[FESOM_TRACER_S].values_fld.h_checked(), nl, p);
    }

    /* Salinity floor — port of the *intent* behind Fortran's tracer_cutoff
     * namelist (declared in oce_modules.F90:172-176 but never actually
     * enforced in the Fortran source either; declared dead-code).
     *
     * Why we need it: confined brackish seas (Baltic, Hudson Bay, etc.) can
     * receive enough spring meltwater + runoff + precipitation to drive
     * surface salinity to near-zero locally. Once S < ~0.2 PSU, the JM
     * equation of state produces NaN density, NaN propagates into the SSH
     * CG solver (pp·App = NaN), and the model aborts. The 2-yr CORE2 run
     * hit this in spring 1959 at Gulf-of-Bothnia / south-Baltic nodes.
     *
     * The clamp is deterministic over myDim+eDim and idempotent, so it's
     * safe to apply without an extra halo exchange. Conservative floor of
     * 0.5 PSU — well below natural Baltic dynamics (typical surface SSS
     * 3-8 PSU there) so it only fires when the model is clearly broken.
     *
     * Net non-conservation introduced is <0.1% of global salt mass over
     * a multi-year run (verified post-hoc on the 2-yr CORE2 case).
     * Suppress with FESOM_NO_SFLOOR=1 to bisect. */
    {
        static int sf_checked = 0, sf_skip = 0;
        if (!sf_checked) {
            const char *e = getenv("FESOM_NO_SFLOOR");
            sf_skip = (e && atoi(e));
            sf_checked = 1;
        }
        if (!sf_skip) {
            const real_t S_FLOOR = (real_t)0.5;
            real_t *S = tracers->data[FESOM_TRACER_S].values;
            const int N_full = mesh->myDim_nod2D + mesh->eDim_nod2D;
            for (int n = 0; n < N_full; ++n) {
                const int nzmax = mesh->nlevels_nod2D[n] - 1;
                for (int nz = 0; nz < nzmax; ++nz) {
                    const size_t i = FESOM_NODE3D(n, nz, mesh->nl);
                    if (S[i] < S_FLOOR) S[i] = S_FLOOR;
                }
            }
        }
    }

    /* 13c. Phase G6b — bolus velocity sub (Fortran oce_ale_tracer.F90:284-295).
     * Restore dyn->uv / dyn->w / dyn->w_e to their pre-add values so the
     * remainder of the timestep (and the next step) sees the original
     * velocity field. M2.6-c: ON THE DEVICE (mirror of 13a). uv/w/w_e are
     * device-current (augmented, Synced) from 13a through the whole FCT region —
     * the FCT only READS uv/w_e and the Redi/tracer-diff/sfloor never touch them —
     * and fer_uv/fer_w stay device-current from 13a, so no IN push is needed. OUT:
     * modify_device then sync_host(uv/w/w_e) so the next step's host substep-3
     * readers (update_vel/compute_vel_nodes) see the restored velocity. */
    if (gm) {
        fesom_gm_bolus_apply_kk(dyn, mesh, (real_t)-1.0);
        dyn->uv_fld.modify_device();   dyn->w_fld.modify_device();   dyn->w_e_fld.modify_device();
        dyn->uv_fld.sync_host();       dyn->w_fld.sync_host();       dyn->w_e_fld.sync_host();
    }

    /* 14. commit thickness — M2.5: device kernel. SYNC_MAP §2 row 14. hnode := hnode_new
     *  (flat copy), helem := vertex mean (owned elems; halo via the exchanges below). IN:
     *  hnode_new is Synced since 12a — the host tracer adv/diff (substep 13) only READ it, so the
     *  device copy is still current (sync_device is a no-op). OUT: sync_host(hnode, helem) before
     *  the halos; both EVOLVING → feed next step's substep-1 EOS + substep-6 TDMA. */
    mesh->hnode_new_fld.sync_device();   /* no-op: Synced since 12a; documents the dependency */
    fesom_ale_commit_thickness_kk(mesh);
    mesh->hnode_fld.sync_host();
    mesh->helem_fld.sync_host();
    if (s_verify_ale) fesom_ale_commit_verify(mesh, step_n);
    fesom_exchange_nod3D (mesh->hnode_fld.h_checked(), nl, p);   /* both already same — but be explicit */
    fesom_exchange_elem3D(mesh->helem_fld.h_checked(), nl, p);   /* Fortran oce_ale.F90:1027,1249 */

    /* Sea-ice step is now called from fesom_main BEFORE the ocean step
     * (ice writes heat_flux/water_flux that the ocean step consumes). */

#ifdef FESOM_KK_SYNCCHECK
    /* M1.5: exercise the host<->device rails on this step's ocean state (no-op in production). */
    ocean_synccheck_roundtrip(dyn, tracers, aux);
#endif

    return cg_iters;
}
