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
#include "fesom_halo_device.hpp"   // M5.4: flip ocean halos to GPU-aware-MPI (fesom_halo_field)
#include "fesom_profile.hpp"       // M5.6: per-substep host+device timing (FESOM_STEP_PROFILE)
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
    static int s_verify_ssh    = 0;   /* M4.2 substeps 7-11 (SSH RHS + CG + update_vel + hbar) */
    static int s_verify_smooth = 0;   /* M5.18 device smoother (bvfreq here; blmc in fesom_kpp) */
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
        s_verify_ssh = (e && strstr(e, "ssh")) ? 1 : 0;       /* M4.2: no substring collision (ssh ⊄ any key) */
        s_verify_smooth = (e && strstr(e, "smooth")) ? 1 : 0; /* M5.18: collision-free token */
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

    /* M5.6: per-substep timing (FESOM_STEP_PROFILE; host+device, fence-bounded — finds hidden
     * host costs like the blmc smoother that nsys can't see). PMARK closes the current phase and
     * opens the next; the final toc + #undef are after substep 14. No-op when profiling is off. */
#define PMARK(nm) do { fesom_prof::toc((nm), _tp); _tp = fesom_prof::tic(); } while (0)
    double _tp = fesom_prof::tic();

    /*  1. EOS + hydrostatic pressure + N² — M2.1: the FIRST device kernels.
     *  SYNC_MAP §1 INPUT rail: tracers T/S and mesh hnode are host-written via the raw alias each
     *  step, which the DualView cannot see (L14), so mark them host-dirty and push to the device
     *  before the kernels. Mesh Z/ulevels/nlevels are set-once + already device-current
     *  (mesh_sync_geometry_device). On Serial/OpenMP host==device so every sync below is a no-op
     *  (run stays bit-identical); the rail only moves bytes on CUDA. */
    {
        /* M5.13g1-T: T values device-resident across the step - no EOS re-push (EOS reads it on device). */
        /* M5.14 (S flip): S values device-resident too - no EOS re-push (EOS reads it on device). */
        /* M5.13f: hnode device-resident from last step's commit (fesom_halo_field) - no re-push; EOS reads it on device. */
    }
    fesom_pressure_bv_kk(tracers, mesh, aux);   /* device: density_m_rho0/hpressure/bvfreq/dbsfc/MLD1 */
    /* sw_alpha / sw_beta — McDougall (1987). Needed by GM/Redi (and KPP).
     * Mirror of Fortran oce_ale.F90:3475 sw_alpha_beta. */
    fesom_compute_sw_alpha_beta_kk(tracers, mesh, aux);   /* device: sw_alpha/sw_beta */
    /* SYNC_MAP §1 OUTPUT rail: the kernels marked these modify_device(); pull them to the host
     * before the host halo exchanges + smooth_nod3D + the GM/PP/KPP consumers that read via the raw
     * alias. dbsfc has no halo but KPP bldepth reads it on host; MLD1_ind has no halo but GM
     * init_Redi_GM reads it on host — sync both here too. (No-ops on Serial/OpenMP.) */
    /* M5.14 (density flip): density_m_rho0 is device-halo'd below + stays device-resident. It has NO
     * on-device consumer (GM recomputes its own ρ-gradient from T/S) and NO per-step host reader except
     * I/O — so the OUT-rail sync_host is removed; the monthly mean reads it on device (resolve_density_dev)
     * and the snapshot via the pre-I/O sync_host (L48). The gated eos verify reads it host-current on Serial. */
    /* M5.13b: hpressure/sw_alpha/sw_beta now device-halo'd below (fesom_halo_field) + stay
     * device-resident - their OUT-rail sync_host removed (PGF/GM/KPP read them on device). */
    /* M5.15 T3: bvfreq + dbsfc OUT sync_host REMOVED — both are VERIFY-ONLY host reads (the
     * host C-twins eos_verify / fesom_kpp_verify[kpp_bldepth] run only under FESOM_KK_VERIFY,
     * where Serial host==device so no sync is needed anyway). Production reads them on DEVICE:
     * bvfreq via eos/gm/pp/kpp .d() (+ the device smoother re-dirties it; snapshot covered by
     * the pre-I/O sync fesom_main.cpp:1309), dbsfc via kpp_bldepth_kk .d(). bvfreq's cosmetic
     * min/max console range goes stale on CUDA (accepted class, like S/density/T post-M5.13/14).
     * MLD1_ind stays (GM init_Redi host read, tiny nod2D index). */
    aux->MLD1_ind_fld.sync_host();
    /* In-binary per-kernel gate (Serial max|Δ|==0); non-intrusive (restores the Kokkos result).
     * Reads aux host-current (just synced above). */
    if (s_verify_eos) fesom_eos_verify(tracers, mesh, aux, step_n);
    /* exchange the per-node 3D outputs (Fortran oce_ale_pressure_bv.F90:2844-).
     * All five EOS-output halos now read through Field::h_checked(): pointer-identical to the raw
     * alias (production unchanged), but under -DFESOM_KK_SYNCCHECK each aborts if the field is still
     * device-authoritative — i.e. if the output sync_host() above were forgotten now that EOS runs
     * on the device (docs/SYNC_MAP.md §1). */
    fesom_halo_field(aux->density_m_rho0_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.14 (density flip) device-halo */
    /* M5.13b: hpressure device-halo (PGF reads it at element vertices on device, substep 2). */
    fesom_halo_field(aux->hpressure_fld, FESOM_HALO_NOD3D, nl, 1, p);
    /* M5.5 (B): bvfreq stays device-resident for the device smoother below → device-halo. */
    fesom_halo_field(aux->bvfreq_fld, FESOM_HALO_NOD3D, nl, 1, p);
    /* M5.9-pin (session 20): the M5.9 blanket sync_host here was a PLACEBO (the leave-one-out's 0.4
     * for bvfreq was sync-FENCE chaos sensitivity, not a stale read). The NaN-poison discriminator —
     * keep the sync, then NaN the host copy so only a HOST reader is affected — proved bvfreq has NO
     * model-feedback host reader on the device path (poisoned: model byte-for-byte the clean run).
     * KPP / mo_convect / GM read it on the DEVICE; the device smoother below re-dirties it anyway.
     * Its only host readers are the read-only min/max print + the netCDF snapshot, both covered by
     * the snapshot-gated pre-I/O sync_host in fesom_main.cpp (L48). So: no per-step sync. */
    /* M5.13b: sw_alpha/sw_beta device-halo (GM substep-1b + KPP substep-3 read them on device). */
    fesom_halo_field(aux->sw_alpha_fld, FESOM_HALO_NOD3D, nl, 1, p);
    fesom_halo_field(aux->sw_beta_fld,  FESOM_HALO_NOD3D, nl, 1, p);
    /* horizontal N² smoothing — N2smth_h=.true., N2smth_hidx=1 (Fortran
     * oce_ale_pressure_bv.F90:499 smooth_nod3D(bvfreq,1)). M5.5 (B): DEVICE smoother
     * (no host round-trip) — bvfreq stays device-resident + halo'd through to its
     * consumers (GM, PP/KPP, mo_convect), whose bvfreq IN re-pushes are now removed. */
    if (s_verify_smooth)   /* M5.18 isolated gate: device smoother vs the host C twin (Serial max|Δ|==0) */
        fesom_smooth_nod3D_kk_verify(aux->bvfreq_fld, 1, mesh, p, 0, 1, 0, "bvfreq", step_n);
    else
        fesom_smooth_nod3D_kk(aux->bvfreq_fld, 1, mesh, p);

    PMARK("1_eos");
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
        /* M5.5 (B): bvfreq is device-resident (device smoother, substep 1) — no re-push. */
        /* M5.13b: sw_alpha/sw_beta device-resident with their halo (substep-1 fesom_halo_field) - no re-push; GM sigma_xy/init_redi read them on device. */
        /* M5.13g1-T: T values device-resident - no GM re-push (sigma_xy reads T on device). */
        /* M5.14 (S flip): S values device-resident too - no GM re-push (sigma_xy reads S on device). */
        mesh->hnode_new_fld.modify_host(); mesh->hnode_new_fld.sync_device();
        /* M5.13f: helem device-resident from last step's commit - no re-push; GM reads it on device. */

        /* (G2b) density gradient on neutral surfaces. */
        fesom_compute_sigma_xy_kk(aux, tracers, mesh, gm);
        if (s_verify_gm) { gm->sigma_xy_fld.sync_host(); fesom_gm_sigma_xy_verify(aux, tracers, mesh, gm, p, step_n); }
        fesom_halo_field(gm->sigma_xy_fld, FESOM_HALO_NOD2D, nl, 2, p);   /* M5.15 T2: device-halo (compute_neutral_slope/init_redi read sigma_xy on device); host rail now verify-only */

        /* (G2b) neutral slope + ODM95 tapering. */
        fesom_compute_neutral_slope_kk(aux, mesh, gm);
        /* M5.13c: slope_tapered device-halo'd below (Redi diff_hor + trdiff K33 read it on device). */
        if (s_verify_gm) { gm->neutral_slope_fld.sync_host(); gm->fer_tapfac_fld.sync_host(); fesom_gm_neutral_slope_verify(aux, mesh, gm, p, step_n); }   /* M5.15 T1/T2: host rails VERIFY-ONLY — init_redi reads neutral_slope/fer_tapfac on device (.d() fesom_gm.cpp:1187/1286) */
        fesom_halo_field(gm->neutral_slope_fld, FESOM_HALO_NOD2D, nl1, 3, p);   /* M5.15 T2: device-halo */
        fesom_halo_field(gm->slope_tapered_fld, FESOM_HALO_NOD2D, nl1, 3, p);   /* M5.13c device-halo */

        /* (G3) per-step GM/Redi coefficient builder. */
        fesom_init_redi_gm_kk(aux, mesh, gm);
        gm->fer_C_fld.sync_host();   /* fer_C kept host-exchanged below (small nod2D, not a PCIe target) */
        /* M5.13c: Ki device-halo'd below (Redi + trdiff read it on device). */
        if (s_verify_gm) { gm->fer_scal_fld.sync_host(); gm->fer_K_fld.sync_host(); fesom_gm_init_redi_verify(aux, mesh, gm, p, step_n); }   /* M5.15 T1/T2: fer_scal/fer_K host rails VERIFY-ONLY (device computes them; fer_K read on device by Redi/trdiff) */
        fesom_exchange_nod2D(gm->fer_C_fld.h_checked(), p);
        fesom_halo_field(gm->fer_K_fld, FESOM_HALO_NOD2D, nl, 1, p);   /* M5.15 T2: device-halo (was host fesom_halo_exchange) */
        fesom_halo_field(gm->Ki_fld, FESOM_HALO_NOD2D, nl, 1, p);   /* M5.13c device-halo */

        /* (G4) streamfunction solve (per-node TDMA). */
        fesom_fer_solve_gamma_kk(aux, mesh, gm);
        if (s_verify_gm) fesom_gm_solve_gamma_verify(aux, mesh, gm, p, step_n);
        /* M5.13c: fer_gamma device-halo; fer_gamma2vel reads it at HALO vertices directly on device
         * (the L30 host re-push round-trip is removed — the device halo leaves fer_gamma owned+halo current). */
        fesom_halo_field(gm->fer_gamma_fld, FESOM_HALO_NOD2D, nl, 2, p);

        /* (G4) bolus velocity reconstruction (vertex→element). */
        fesom_fer_gamma2vel_kk(dyn, mesh, gm);
        if (s_verify_gm) fesom_gm_gamma2vel_verify(dyn, mesh, gm, p, step_n);
        /* M5.13c: fer_uv device-halo; ALE vert_vel (12b) + bolus add (13a) read it on device (re-pushes removed). */
        fesom_halo_field(dyn->fer_uv_fld, FESOM_HALO_ELEM2D_FULL, nl, 2, p);
    }

    PMARK("1b_gm");
    /*  2. PGF (linfs + full cells) — M2.4: device kernel.
     *  SYNC_MAP §2 row 2. INPUT rail: hpressure was produced on the device (substep 1),
     *  then sync_host'd (L163) + halo-exchanged on the host (L178) — the halo write is
     *  invisible to the DualView (L14/L27), so push the now-halo-current host hpressure
     *  back to the device with modify_host()+sync_device() (NOT a bare sync_device, which
     *  would feed the kernel the pre-halo device bytes). gradient_sca/elem_nodes/ulevels/
     *  nlevels are set-once device-current (mesh_sync_geometry_device). No-op on Serial/OpenMP. */
    /* M5.13b: hpressure is device-resident with its halo (substep-1 fesom_halo_field) - the IN
     * re-push is GONE; PGF reads it at the element's 3 vertices (incl. HALO) directly on device. */
    fesom_pressure_force_linfs_fullcell_kk(mesh, aux);   /* device: pgf_x, pgf_y (elem) */
    if (s_verify_pgf) fesom_pressure_force_verify(mesh, aux, step_n);
    /* M5.4: pgf device-halo (GPU-aware MPI on CUDA, host-staged on Serial). The OUT-rail
     * sync_host + the vel_rhs IN re-push (substep 4) are gone — pgf stays device-resident
     * with its halo (vel_rhs reads it on-device). */
    fesom_halo_field(aux->pgf_x_fld, FESOM_HALO_ELEM3D, nl, 1, p);
    fesom_halo_field(aux->pgf_y_fld, FESOM_HALO_ELEM3D, nl, 1, p);
    /* M5.9-pin (session 20): placebo sync dropped — compute_vel_rhs (substep 4) reads pgf_x/pgf_y on
     * the DEVICE (device-resident with its halo); the NaN-poison discriminator proved no model-feedback
     * host reader. The only host readers are the diagnostic min/max print + netCDF snapshot, both
     * covered by the pre-I/O sync_host in fesom_main.cpp (L48). */

    PMARK("2_pgf");
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
    /* M5.13g1: uv device-resident across the step (update_vel fesom_halo_field) - no re-push. */
    fesom_compute_vel_nodes_kk(mesh, dyn);              /* device: writes dyn->uvnode (owned) */
    if (s_verify_pp) fesom_compute_vel_nodes_verify(mesh, dyn, step_n);
    /* M5.4: uvnode device-halo (GPU-aware MPI on CUDA, host-staged on Serial); the OUT sync_host
     * + the KPP/PP IN re-pushes are gone — uvnode stays device-resident with its halo. */
    fesom_halo_field(dyn->uvnode_fld, FESOM_HALO_NOD3D, nl, 2, p);
    /* M5.16: the M5.9-pin uvnode sync_host is GONE — fesom_bulk_compute_kk now reads uvnode's surface
     * on the DEVICE (the L&Y09 wind stress, wind relative to ocean current). uvnode was the SOLE real
     * host reader (proven by the NaN-poison discriminator, M5.9-pin); with bulk on the device, uvnode
     * stays device-resident across the step → the ~nod3D×2 DtoH/step is gone (the residency unlock).
     * KPP already read uvnode on the device via its device-resident halo. (uvnode is not a snapshot
     * output; the verify-only host read is gated in fesom_main.cpp.) */

    if (s_use_kpp) {
        /* KPP on device (M2.3). It writes aux->Av (elements) + the single aux->Kv (T-channel,
         * oce_ale.F90:3518-3522) and does its OWN internal halo exchanges (bracketed inside
         * fesom_kpp_mixing_kk). INPUT rail: push every host-authoritative input KPP reads to
         * the device — the Serial gate can't catch a missing sync_device (host==device), so we
         * sync them all explicitly. bvfreq is the L27 device→host(smooth)→device hand-off;
         * forcing is host-produced; sw_alpha, sw_beta, dbsfc, uvnode, S, hnode are device-current
         * from substep 1 but re-synced for robustness. (No-op on Serial/OpenMP.) */
        /* M5.5 (B): bvfreq is device-resident (device smoother, substep 1) — no re-push. */
        /* M5.13b: sw_alpha/sw_beta device-resident with their halo (substep-1 fesom_halo_field) - no re-push. */
        /* M5.15 T3: dbsfc IN re-push REMOVED — dbsfc is device-current from substep 1 (eos_kk
         * writes it on device) through KPP substep 3 (kpp_bldepth_kk reads it .d()); no host op
         * mutates it in production (the host twin is verify-gated). The "robustness" re-push was
         * a redundant HtoD. (Pairs with the T3 OUT-sync removal at substep 1.) */
        /* M5.4: uvnode device-resident with its halo (substep 3) — no re-push. */
        /* M5.14 (S flip): S values device-resident - no KPP re-push (bldepth reads S on device). */
        /* M5.13f: hnode device-resident from last step's commit - no re-push; KPP reads it on device. */
        /* forcing is a const input to the step; the sync_device is a pure coherence op (moves
         * host→device, no logical mutation) → const_cast is safe and localized here. */
        auto *fnc = const_cast<struct fesom_forcing *>(forcing);
        fnc->stress_node_surf_fld.modify_host(); fnc->stress_node_surf_fld.sync_device();
        fnc->heat_flux_fld.modify_host();        fnc->heat_flux_fld.sync_device();
        fnc->water_flux_fld.modify_host();       fnc->water_flux_fld.sync_device();
        fnc->sw_3d_fld.modify_host();            fnc->sw_3d_fld.sync_device();

        fesom_kpp_mixing_kk(ctx->kpp, aux, tracers, forcing, dyn, mesh, p);

        /* M5.7b: Av/Kv stay DEVICE-resident from KPP → mo_convect → the device-halo (below) →
         * ivisc (Av, substep 6) / trdiff (Kv, substep 13b) — the OUT-rail sync_host + mo_convect
         * IN re-push round trip is gone. KPP marked them modify_device(); mo_convect reads them on
         * device. The verify reads the raw alias (= device on Serial, the gate) → sync only when
         * verifying. (Kv/Av are also snapshot outputs → pre-I/O sync_host in fesom_main.cpp, L48.) */
        if (s_verify_kpp) {
            aux->Av_fld.sync_host();
            aux->Kv_fld.sync_host();
            fesom_kpp_verify(ctx->kpp, aux, tracers, forcing, dyn, mesh, p, step_n);
        }
    } else {
        /* PP branch (opt-in; FESOM_MIX_SCHEME=PP). INPUT rail: bvfreq (host-written by
         * smooth_nod3D, substep 1) → device. M5.4: uvnode is device-resident (substep 3, no re-push). */
        /* M5.5 (B): bvfreq is device-resident (device smoother) — no re-push. */
        fesom_pp_mixing_kk(mesh, dyn, aux);             /* device: Kv (node), Av (elem) */
        if (s_verify_pp) {                              /* verify reads them host (Serial == device) */
            aux->Kv_fld.sync_host();
            aux->Av_fld.sync_host();
            fesom_pp_mixing_verify(mesh, dyn, aux, step_n);
        }
        /* M5.7b: Kv/Av device-halo (was sync_host + host exchange) — device-resident into mo_convect. */
        fesom_halo_field(aux->Kv_fld, FESOM_HALO_NOD3D,  nl, 1, p);
        fesom_halo_field(aux->Av_fld, FESOM_HALO_ELEM3D, nl, 1, p);
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
    /* M5.5 (B): bvfreq is device-resident (device smoother, substep 1) — no re-push. */
    /* M5.7b: Kv/Av are device-resident (KPP modify_device / PP device-halo above) — no IN re-push. */
    fesom_mo_convect_kk(mesh, aux);                     /* device: Kv, Av */
    if (s_verify_pp) {                                  /* verify reads them host (Serial == device) */
        aux->Kv_fld.sync_host();
        aux->Av_fld.sync_host();
        fesom_mo_convect_verify(mesh, aux, step_n, mc_Kv_in.data(), mc_Av_in.data());
    }
    /* M5.7b: Kv/Av device-halo (was sync_host + host exchange) — device-resident into ivisc (Av,
     * substep 6) + trdiff (Kv, substep 13b); their IN re-pushes are removed too. */
    fesom_halo_field(aux->Kv_fld, FESOM_HALO_NOD3D,  nl, 1, p);
    fesom_halo_field(aux->Av_fld, FESOM_HALO_ELEM3D, nl, 1, p);

    PMARK("3_mixing");
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
    /* M5.13g1: uv device-resident - no re-push (compute_vel_rhs reads it on device). */
    /* M5.13d: uv_rhsAB device-resident with its halo (cross-step AB2 history) - no IN re-push;
     * compute_vel_rhs part i reads it on device (last step's fesom_halo_field left it owned+halo). */
    dyn->eta_n_fld.modify_host();    dyn->eta_n_fld.sync_device();
    /* M5.13e: w_e device-resident with its halo (12d fesom_halo_field) - no re-push; compute_vel_rhs reads it on device. */
    /* M5.4: pgf_x/pgf_y are device-resident with their halo from substep 2 — no re-push. */
    /* M5.13f: hnode device-resident from last step's commit - no re-push; compute_vel_rhs reads it on device. */
    fesom_compute_vel_rhs_kk(mesh, aux, dyn, /*is_first_step=*/(step_n == 1), p);
    /* M5.13d: uv_rhsAB OUT sync_host removed - device-halo'd below; AB2 history read on device, no host reader. */
    if (s_verify_vrhs) fesom_compute_vel_rhs_verify(mesh, aux, dyn, (step_n == 1), p, step_n,
                                                    vrhs_uv_rhsAB_in.data());
    /* M5.4: uv_rhs needed by visc_filt_bidiff on HALO elements → device-halo (GPU-aware MPI on
     * CUDA, host-staged on Serial). The old OUT sync_host + the visc IN re-push (below) are gone:
     * uv_rhs now stays device-resident with its halo across substeps 4-6. */
    fesom_halo_field(dyn->uv_rhs_fld, FESOM_HALO_ELEM3D, nl, 2, p);
    fesom_halo_field(dyn->uv_rhsAB_fld, FESOM_HALO_ELEM3D, nl, 2, p);   /* M5.13d device-halo */

    PMARK("4_velrhs");
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
    /* M5.13g1: uv device-resident - no re-push (visc_filt reads it on device). */
    /* M5.4: uv_rhs is device-resident with its halo from substep 4 — no IN re-push. */
    fesom_visc_filt_bidiff_kk(mesh, dyn, p);   /* device: uv_rhs += biharmonic; internal Uc/Vc halo (D21) */
    if (s_verify_vfilt) fesom_visc_filt_bidiff_verify(mesh, dyn, p, step_n, vfb_uv_rhs_in.data());
    /* uv_rhs (final output) needed at halo for impl_vert_visc neighbour reads (TDMA SpMV) →
     * device-halo (GPU-aware MPI on CUDA, host-staged on Serial). */
    fesom_halo_field(dyn->uv_rhs_fld, FESOM_HALO_ELEM3D, nl, 2, p);

    PMARK("5_viscfilt");
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
    /* M5.4: uv_rhs is device-resident with its halo from substep 5 — no IN re-push. */
    /* M5.13g1: uv device-resident - no re-push (impl_vert_visc reads it on device). */
    /* M5.14 (w_i flip): w_i device-resident across the step boundary (12d fesom_halo_field) - no re-push.
     * Cross-step (produced 12d, consumed here at substep 6). At step 1 substep 6 reads device w_i before any
     * 12d has run → the zero-init device View, which equals the runtime value (use_wsplit=false → w_i≡0
     * always), so NO L57 init push is needed (verified: ale.cpp sets w_i=0 in the use_wsplit=false branch). */
    /* M5.7b: Av is device-resident with its halo from substep 3 (mo_convect device-halo) — no re-push. */
    /* M5.13f: helem device-resident from last step's commit - no re-push; impl_vert_visc reads it on device. */
    {   auto *fnc = const_cast<struct fesom_forcing *>(forcing);
        fnc->stress_surf_fld.modify_host(); fnc->stress_surf_fld.sync_device();   }
    fesom_impl_vert_visc_kk(mesh, aux, forcing, dyn);   /* device: uv_rhs */
    if (s_verify_ivisc) fesom_impl_vert_visc_verify(mesh, aux, forcing, dyn, step_n, ivv_uv_rhs_in.data());
    fesom_halo_field(dyn->uv_rhs_fld, FESOM_HALO_ELEM3D, nl, 2, p);   /* device-halo (GPU-aware MPI) */
    /* M5.9-pin (session 20): placebo sync dropped — uv_rhs is read by impl_vert_visc (substep 6) and
     * compute_ssh_rhs (substep 7) on the DEVICE (device-resident with its halo), and is NOT a snapshot
     * or diagnostic-print field, so it has NO host reader at all. The NaN-poison discriminator confirmed
     * zero model effect. */

    /*  7-10. The §5 SSH block — M4.2: ON THE DEVICE (closes the SYNC_MAP §5 mid-step
     *  host CG round-trip; substeps 1-14 now flow on device except the trivial host eta_n
     *  map + the ice step + the salinity floor). compute_ssh_rhs_kk (edge→node SCATTER) →
     *  fesom_ssh_solve_cg_kk (per-row CSR-gather SpMV + dot-product parallel_reduce + the
     *  unchanged scalar MPI_Allreduce, owning its pp/rr/X halo brackets) → update_vel_kk
     *  (per-element map) → compute_hbar_kk (edge→node SCATTER + maps).
     *
     *  IN rail (L28 — sync EVERY input the §5 block reads that a host kernel/halo last
     *  wrote): uv (Synced from substep 6; pushed for self-containment), uv_rhs (substep-6
     *  sync_host + ELEM3D halo, L30), d_eta (last step's CG + halo — the CG warm start),
     *  ssh_rhs_old (last step's compute_hbar + halo — the (1-α) term), helem (evolving mesh),
     *  hbar (last step's compute_hbar + halo — compute_hbar reads it at [0,N_alloc) to save
     *  hbar_old). The set-once stiffness CSR is device-current (pushed once in the
     *  preconditioner) and the set-once mesh geometry from mesh_sync_geometry_device. On
     *  Serial/OpenMP host==device so every sync is a no-op (run stays bit-identical). */
    const int    Nn_ssh  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const size_t Nuv_ssh = (size_t)(mesh->myDim_elem2D + mesh->eDim_elem2D
                                   + mesh->eXDim_elem2D) * (size_t)nl * 2;
    std::vector<real_t> ssh_pre_rhs, ssh_pre_deta, ssh_pre_uv, ssh_pre_rhsold,
                        ssh_pre_hbar, ssh_pre_hbarold, ssh_pre_etan;
    if (s_verify_ssh) {   /* L26 capture-before: the seven read-modify-write outputs */
        ssh_pre_rhs.assign   (dyn->ssh_rhs,     dyn->ssh_rhs     + Nn_ssh);
        ssh_pre_deta.assign  (dyn->d_eta,       dyn->d_eta       + Nn_ssh);
        ssh_pre_uv.assign    (dyn->uv,          dyn->uv          + Nuv_ssh);
        ssh_pre_rhsold.assign(dyn->ssh_rhs_old, dyn->ssh_rhs_old + Nn_ssh);
        ssh_pre_hbar.assign  (mesh->hbar,       mesh->hbar       + Nn_ssh);
        ssh_pre_hbarold.assign(mesh->hbar_old,  mesh->hbar_old   + Nn_ssh);
        ssh_pre_etan.assign  (dyn->eta_n,       dyn->eta_n       + Nn_ssh);
    }
    /* M5.13g1: uv device-resident - no re-push (ssh_rhs/CG read it on device). */
    /* M5.4: uv_rhs is device-resident with its halo from substep 6 — no re-push. */
    dyn->d_eta_fld.modify_host();       dyn->d_eta_fld.sync_device();
    dyn->ssh_rhs_old_fld.modify_host(); dyn->ssh_rhs_old_fld.sync_device();
    /* M5.13f: helem device-resident from last step's commit - no re-push; ssh_rhs/CG read it on device. */
    mesh->hbar_fld.modify_host();       mesh->hbar_fld.sync_device();

    PMARK("6_ivisc");
    /*  7. SSH RHS (linfs) — device. CG reads ssh_rhs at OWNED rows only, so no re-push
     *     after the halo (the device owned ssh_rhs from the kernel stays current). */
    fesom_compute_ssh_rhs_linfs_kk(mesh, dyn);
    dyn->ssh_rhs_fld.sync_host();                          /* OUT: before the nod2D halo */
    fesom_exchange_nod2D(dyn->ssh_rhs_fld.h_checked(), p); /* Fortran oce_ale.F90:1954 */

    /*  8. CG SSH solve — device (host loop control + device vector kernels + CG-owned
     *     pp/rr/X halo brackets). The exit EXCH(X) is the driver's exchange below. */
    int cg_iters = fesom_ssh_solve_cg_kk(ctx->stiff, ctx->solver, mesh, dyn);
    dyn->d_eta_fld.sync_host();                            /* OUT: before the nod2D halo */
    fesom_exchange_nod2D(dyn->d_eta_fld.h_checked(), p);   /* Fortran solver.F90:279 */

    /*  9. velocity update — device. update_vel reads d_eta at the 3 element vertices
     *     (incl. HALO), so re-push d_eta after its halo (L30 cross-op re-push). */
    dyn->d_eta_fld.modify_host(); dyn->d_eta_fld.sync_device();
    fesom_update_vel_kk(mesh, dyn);
    /* M5.13g1: uv device-halo (GPU-aware MPI). uv stays device-resident across the whole step +
     * the next step's substeps 3-7 + the ice-step ocean2ice (ALL uv re-pushes removed). snap-out
     * (u/v element output) → pre-I/O sync in fesom_main.cpp; one-time init push bootstraps step 1. */
    fesom_halo_field(dyn->uv_fld, FESOM_HALO_ELEM3D, nl, 2, p);

    /* 10. transport-divergence → ssh_rhs_old, then hbar update — device. compute_hbar reads
     *     uv at edge_tri (interior elements), but uv was just halo'd on the host (line above),
     *     so re-push it (L30) to keep the device copy coherent with the host. */
    /* M5.13g1: uv device-resident (update_vel fesom_halo_field) - no re-push; compute_hbar reads it on device. */
    fesom_compute_hbar_kk(mesh, dyn);
    dyn->ssh_rhs_old_fld.sync_host();                      /* OUT (3 fields) before the halos */
    mesh->hbar_fld.sync_host();
    mesh->hbar_old_fld.sync_host();
    fesom_exchange_nod2D(dyn->ssh_rhs_old_fld.h_checked(), p);   /* Fortran oce_ale.F90:2078 */
    fesom_exchange_nod2D(mesh->hbar_fld.h_checked(),       p);   /* Fortran oce_ale.F90:2102 */

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

    /* M4.2 FESOM_KK_VERIFY=ssh — gate the whole §5 block (substeps 7-11) against the C
     * twins. Runs AFTER eta_n so the host mirrors hold the full KK result; capture-before
     * was taken at the top of substep 7. Serial max|Δ|==0 (the CG is deterministic; on
     * Serial the device kernels run on the same memory with identical FP ops/order). */
    if (s_verify_ssh)
        fesom_ssh_block_verify(ctx->stiff, ctx->solver, mesh, dyn, step_n,
                               ssh_pre_rhs, ssh_pre_deta, ssh_pre_uv, ssh_pre_rhsold,
                               ssh_pre_hbar, ssh_pre_hbarold, ssh_pre_etan);

    PMARK("7_ssh");
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
    /* M5.13f: hnode device-resident from last step's commit - no re-push; thickness reads it on device. */
    fesom_ale_thickness_linfs_kk(mesh);
    mesh->hnode_new_fld.sync_host();
    if (s_verify_ale) fesom_ale_thickness_verify(mesh, step_n);
    /* hnode_new = hnode (no exchange needed; both already cover halo). */

    /* 12b. vertical velocity. IN: uv (update_vel+halo, substep 9), helem (evolving mesh),
     *  fer_uv (GM only). EDGE→NODE SCATTER (atomic_add, D22) + per-node level cumsum.
     *  OUT: sync_host(w[,fer_w]) before the halo. */
    /* M5.13g1: uv device-resident - no re-push; vert_vel reads it on device. */
    /* M5.13f: helem device-resident from last step's commit - no re-push; vert_vel reads it on device. */
    /* M5.13c: fer_uv device-resident with its halo (substep 1b) - no re-push (vert_vel reads it on device). */
    fesom_ale_vert_vel_linfs_kk(mesh, dyn, gm ? 1 : 0);
    /* M5.14 (fer_w flip): fer_w device-resident - no OUT sync_host (bolus 13a reads it on device, same step;
     * not an output field → no mean entanglement). The gated ale verify reads owned fer_w host-current on Serial. */
    if (s_verify_ale) fesom_ale_vert_vel_verify(mesh, dyn, gm ? 1 : 0, step_n);
    fesom_halo_field(dyn->w_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.13e device-halo (w; snap-out -> pre-I/O sync) */
    if (gm) {
        /* Mirror Fortran oce_ale.F90:2681 — exchange_nod(fer_Wvel). M5.14: device-halo (fer_w stays device-resident). */
        fesom_halo_field(dyn->fer_w_fld, FESOM_HALO_NOD3D, nl, 1, p);
    }

    /* 12c. vertical CFL. IN: w (just halo'd → host-current), hnode_new (Synced from 12a).
     *  Per-node accumulation into the node's OWN column → race-free (NOT a scatter).
     *  OUT: sync_host(cfl_z) before the halo. */
    /* M5.13e: w device-resident with its halo (12b fesom_halo_field) - no re-push (cflz reads it on device). */
    mesh->hnode_new_fld.sync_device();   /* no-op: Synced from 12a; documents the dependency */
    fesom_ale_compute_cflz_kk(mesh, dyn);
    if (s_verify_ale) fesom_ale_compute_cflz_verify(mesh, dyn, step_n);
    /* M5.13a: cfl_z device-halo (GPU-aware MPI on CUDA, host-staged on Serial). The OUT-rail
     * sync_host + the wvel_split IN re-push (below) are gone — cfl_z stays device-resident with
     * its halo; compute_wvel_split (12d) reads it on-device. NOT snap-out. */
    fesom_halo_field(dyn->cfl_z_fld, FESOM_HALO_NOD3D, nl, 1, p);

    /* 12d. w-split (⚠️ use_wsplit=.false. preserved → w_e=w, w_i=0). IN: cfl_z (device-resident,
     *  halo'd above), w (Synced from 12c IN; cflz did not write w). Pure per-(n,nz) map. OUT: sync_host(w_e,w_i). */
    dyn->w_fld.sync_device();   /* no-op: Synced from 12c IN; w unchanged by cflz */
    fesom_ale_compute_wvel_split_kk(mesh, dyn);
    /* M5.14 (w_i flip): w_i device-resident - no OUT sync_host (substep 6 of the NEXT step reads it on device). */
    if (s_verify_ale) fesom_ale_compute_wvel_split_verify(mesh, dyn, step_n);
    fesom_halo_field(dyn->w_e_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.13e device-halo (w_e) */
    fesom_halo_field(dyn->w_i_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.14 (w_i flip) device-halo */

    PMARK("12_ale");
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
        /* M5.13g1: uv device-resident - no bolus IN re-push. */
        /* M5.13e: w/w_e device-resident (12b/12d fesom_halo_field) - no bolus IN re-push. */
        /* M5.13c: fer_uv device-resident with its halo (substep 1b) - no re-push. */
        /* M5.14 (fer_w flip): fer_w device-resident (12b fesom_halo_field) - no bolus IN re-push. */
        fesom_gm_bolus_apply_kk(dyn, mesh, (real_t)1.0);
        dyn->uv_fld.modify_device();   dyn->w_fld.modify_device();   dyn->w_e_fld.modify_device();
        /* M5.13g1: uv now device-resident too (bolus-augmented on device) - no OUT sync_host; the tradv
         * C-twin sees the augmented uv on Serial (host==device); CUDA FCT reads it on device. */
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
        /* M5.13g1: uv device-resident (augmented) - no re-push; FCT reads it on device. */
        /* M5.13e: w_e device-resident (augmented by bolus 13a on device) - no re-push; FCT reads it on device. */
        /* M5.13f: hnode/helem device-resident from last step's commit - no re-push; FCT reads them on device. */
        mesh->hnode_new_fld.modify_host();  mesh->hnode_new_fld.sync_device();
        /* M5.13c: slope_tapered/Ki device-resident with their halo (substep 1b) - no re-push (Redi reads them on device). */

        /* ---- T ---- */
        {
            /* M5.13g1-T (FIX): T values AND valuesold both device-resident - no FCT IN re-push. The MFCT
             * valuesAB couples them, so they must be a COHERENT device pair (the earlier split — values
             * device, valuesold host-staged — diverged T 5e-2 on CUDA; the gate caught it). */
            std::vector<real_t> fct_pre_v, fct_pre_vo;    /* L26 capture-before (pre-FCT inputs) */
            if (s_verify_tradv) {
                fct_pre_v.assign (tracers->data[FESOM_TRACER_T].values,
                                  tracers->data[FESOM_TRACER_T].values    + (size_t)N_redi * nl);
                fct_pre_vo.assign(tracers->data[FESOM_TRACER_T].valuesold,
                                  tracers->data[FESOM_TRACER_T].valuesold + (size_t)N_redi * nl);
            }
            fesom_tracer_advect_one_fct_kk(ctx->tra_sc, FESOM_TRACER_T, mesh, dyn, tracers, p);
            /* M5.13g1-T (FIX): T values + valuesold both device-resident - no FCT OUT sync_host. */
            if (s_verify_tradv) fesom_tracer_fct_verify(ctx->tra_sc, FESOM_TRACER_T, mesh, dyn,
                                                        tracers, p, step_n, fct_pre_v, fct_pre_vo);
        }
        if (gm) {
            /* M5.13g1-T (FIX): T values + valuesold both device-resident from FCT - no Redi IN re-push. */
            std::vector<real_t> redi_pre;     /* L26 capture-before (post-FCT, pre-Redi) */
            if (s_verify_gm) redi_pre.assign(tracers->data[FESOM_TRACER_T].values,
                                             tracers->data[FESOM_TRACER_T].values + (size_t)N_redi * nl);
            fesom_diff_ver_part_redi_expl_kk(FESOM_TRACER_T, gm, mesh, tracers, p);
            fesom_diff_part_hor_redi_kk     (FESOM_TRACER_T, gm, mesh, tracers, p);
            /* M5.13g1-T: T values device-resident - no Redi OUT sync_host (device-halo'd below). */
            if (s_verify_gm) fesom_gm_redi_verify(FESOM_TRACER_T, gm, aux, mesh, tracers, p, step_n, redi_pre);
        }
        fesom_halo_field(tracers->data[FESOM_TRACER_T].values_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.13g1-T device-halo (T values) */

        /* ---- S ---- */
        {
            /* M5.14 (S flip): S values AND valuesold both device-resident - no FCT IN re-push.
             * Mirror of g1-T: the MFCT valuesAB couples them, so they must be a COHERENT device
             * pair (a split — values device, valuesold host-staged — diverged T on CUDA, g1-T). */
            std::vector<real_t> fct_pre_v, fct_pre_vo;    /* L26 capture-before (Serial host==device) */
            if (s_verify_tradv) {
                fct_pre_v.assign (tracers->data[FESOM_TRACER_S].values,
                                  tracers->data[FESOM_TRACER_S].values    + (size_t)N_redi * nl);
                fct_pre_vo.assign(tracers->data[FESOM_TRACER_S].valuesold,
                                  tracers->data[FESOM_TRACER_S].valuesold + (size_t)N_redi * nl);
            }
            fesom_tracer_advect_one_fct_kk(ctx->tra_sc, FESOM_TRACER_S, mesh, dyn, tracers, p);
            /* M5.14 (S flip): S values + valuesold both device-resident - no FCT OUT sync_host. */
            if (s_verify_tradv) fesom_tracer_fct_verify(ctx->tra_sc, FESOM_TRACER_S, mesh, dyn,
                                                        tracers, p, step_n, fct_pre_v, fct_pre_vo);
        }
        if (gm) {
            /* M5.14 (S flip): S values + valuesold both device-resident from FCT - no Redi IN re-push. */
            std::vector<real_t> redi_pre;     /* L26 capture-before (post-FCT, pre-Redi; Serial host==device) */
            if (s_verify_gm) redi_pre.assign(tracers->data[FESOM_TRACER_S].values,
                                             tracers->data[FESOM_TRACER_S].values + (size_t)N_redi * nl);
            fesom_diff_ver_part_redi_expl_kk(FESOM_TRACER_S, gm, mesh, tracers, p);
            fesom_diff_part_hor_redi_kk     (FESOM_TRACER_S, gm, mesh, tracers, p);
            /* M5.14 (S flip): S values device-resident - no Redi OUT sync_host (device-halo'd below). */
            if (s_verify_gm) fesom_gm_redi_verify(FESOM_TRACER_S, gm, aux, mesh, tracers, p, step_n, redi_pre);
        }
        fesom_halo_field(tracers->data[FESOM_TRACER_S].values_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.14 (S flip) device-halo (S values) */
    }

    PMARK("13_fct");
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
        /* M5.7b: Kv is device-resident with its halo from substep 3 (mo_convect device-halo) — no re-push. */
        mesh->hnode_new_fld.sync_device();   /* no-op: Synced since 12a; documents the dependency */
        {   auto *fnc = const_cast<struct fesom_forcing *>(forcing);
            fnc->heat_flux_fld.modify_host();    fnc->heat_flux_fld.sync_device();
            fnc->water_flux_fld.modify_host();   fnc->water_flux_fld.sync_device();
            fnc->virtual_salt_fld.modify_host(); fnc->virtual_salt_fld.sync_device();
            fnc->relax_salt_fld.modify_host();   fnc->relax_salt_fld.sync_device();
            fnc->sw_3d_fld.modify_host();        fnc->sw_3d_fld.sync_device();   }
        /* M5.13g1-T: T values device-resident - no trdiff IN re-push (reads it on device). */
        /* M5.14 (S flip): S values device-resident too - no trdiff IN re-push (reads it on device). */
        /* M5.13c: slope_tapered/Ki device-resident with their halo (substep 1b) - no re-push (trdiff K33 reads them on device). */
        fesom_impl_vert_diff_tracers_kk(mesh, aux, forcing, tracers, gm);   /* device: values (T,S) */
        /* M5.13g1-T: T values device-resident - no trdiff OUT sync_host (device-halo'd below). */
        /* M5.14 (S flip): S values device-resident too - no trdiff OUT sync_host (device-halo'd below). */
        if (s_verify_trdiff) fesom_impl_vert_diff_tracers_verify(mesh, aux, forcing, tracers, gm,
                                                                 step_n, trd_pre_T, trd_pre_S);
        fesom_halo_field(tracers->data[FESOM_TRACER_T].values_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.13g1-T device-halo (T values) */
        /* M5.16: the M5.13g1-T (FIX2, L50) T sync_host is GONE — fesom_bulk_compute_kk now reads
         * SST = T[surface] on the DEVICE (the JRA55 air-sea heat flux). bulk was the SOLE host reader of
         * T (EOS/GM/FCT/Redi/trdiff/ocean2ice all read it on device); with bulk on the device, T stays
         * device-resident across the step → the ~nod3D DtoH/step is gone (the residency unlock, paired
         * with uvnode at substep 3). T's snapshot output is covered by the pre-I/O sync_host in
         * fesom_main.cpp (L48); the verify-only host read is gated there too. */
        fesom_halo_field(tracers->data[FESOM_TRACER_S].values_fld, FESOM_HALO_NOD3D, nl, 1, p);   /* M5.14 (S flip) device-halo (S values) */
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
        /* M5.14 (S flip): the clamp is now a DEVICE kernel (S is device-resident; its host alias is
         * stale after the device trdiff — the OUT sync_host was removed). Placed AFTER the post-trdiff
         * device halo above so the owned+halo clamp reproduces the prior host path's exchange-then-floor
         * order (bit-identical, race-free per-node column, no scatter). FESOM_NO_SFLOOR=1 still bisects. */
        if (!sf_skip) fesom_salinity_floor_kk(mesh, tracers);
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
        /* M5.13g1: uv device-resident (bolus-restored on device) - no OUT sync_host; next-step substep-3 reads it on device. */
    }

    PMARK("13b_trdiff");
    /* 14. commit thickness — M2.5: device kernel. SYNC_MAP §2 row 14. hnode := hnode_new
     *  (flat copy), helem := vertex mean (owned elems; halo via the exchanges below). IN:
     *  hnode_new is Synced since 12a — the host tracer adv/diff (substep 13) only READ it, so the
     *  device copy is still current (sync_device is a no-op). OUT: sync_host(hnode, helem) before
     *  the halos; both EVOLVING → feed next step's substep-1 EOS + substep-6 TDMA. */
    mesh->hnode_new_fld.sync_device();   /* no-op: Synced since 12a; documents the dependency */
    fesom_ale_commit_thickness_kk(mesh);
    if (s_verify_ale) fesom_ale_commit_verify(mesh, step_n);
    /* M5.13f: hnode/helem device-halo'd below (evolving mesh stays device-resident across the step
     * boundary); the ~10 next-step IN re-pushes are removed. NOT snap-out. */
    fesom_halo_field(mesh->hnode_fld, FESOM_HALO_NOD3D,  nl, 1, p);   /* M5.13f device-halo (hnode) */
    fesom_halo_field(mesh->helem_fld, FESOM_HALO_ELEM3D, nl, 1, p);   /* M5.13f device-halo (helem) */

    /* Sea-ice step is now called from fesom_main BEFORE the ocean step
     * (ice writes heat_flux/water_flux that the ocean step consumes). */

    PMARK("13c_bolus+14_commit");   /* M5.6: close the last substep bucket */
#undef PMARK

#ifdef FESOM_KK_SYNCCHECK
    /* M1.5: exercise the host<->device rails on this step's ocean state (no-op in production). */
    ocean_synccheck_roundtrip(dyn, tracers, aux);
#endif

    return cg_iters;
}
