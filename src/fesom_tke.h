#ifndef FESOM_TKE_H
#define FESOM_TKE_H

#include "fesom_types.h"
#include "fesom_field.hpp"   /* DualView-backed storage; device-resident from day one */

struct fesom_mesh;
struct fesom_partit;
struct fesom_aux;
struct fesom_forcing;
struct fesom_dyn;

/*
 * CVMix classical-TKE vertical mixing (Gaspar et al. 1990; Blanke & Delecluse 1993 mixing
 * length) — strictly faithful port of the C oracle's fesom_tke.{c,h}, which is itself a
 * faithful port of the ICON/Brueggemann CVMix fork FESOM2 vendors in src/cvmix_driver/:
 *
 *   fesom_tke.{h,cpp}    = gen_modules_cvmix_tke.F90 (module g_cvmix_tke): state arrays,
 *                          namelist params, the per-node driver calc_cvmix_tke, Av/Kv
 *                          wiring, MPI exchanges.
 *   fesom_cvmix_tke.hpp  = cvmix_tke.F90 (module cvmix_tke): the pure column math.
 *
 * Oracle: /home/a/a270088/port2/fesom2_port_zstar @ df8b9a8 (certified bit-identical to
 * this port at the default config, M6 Task 0.1). Plan:
 * docs/plans/20260712-m6-options-tke-mevp-zstar.md. Reference namelists (all 10):
 * jobs/m6_namelists/tke/ (mix_scheme='cvmix_TKE', dt=1800, &param_tke with tke_cd=3.75).
 *
 * Selected at runtime via FESOM_MIX_SCHEME=TKE (alias cvmix_TKE); KPP stays the default,
 * PP stays available. The TKE branch mirrors oce_ale.F90:3749-52 (calc_cvmix_tke ->
 * mo_convect -> the shared Kv/Av exchanges in fesom_step.cpp).
 *
 * STATE (mirrors the g_cvmix_tke module allocatables, gen:130-273). N = myDim+eDim nodes,
 * stride nl, so each slab is a contiguous [N*nl] array directly passable to the halo layer.
 *
 *   Always allocated, [N*nl]:
 *     tke          PROGNOSTIC TKE at interfaces (init 0 -> floored to tke_min by the first
 *                  call; the per-column tke_old is read every step). The Fortran comment
 *                  calls it a "diagnostic" — it is NOT.
 *     tke_Av/tke_Kv  node viscosity/diffusivity. Global (not per-column scratch) only
 *                  because exchange_nod + the full Kv copy need halo storage. Prior-step
 *                  values are NEVER read — no old-value blending; the C verified
 *                  handle_old_vals and the old_KappaM/old_KappaH args DEAD.
 *   Always allocated, [N]:
 *     forc_normstress  |stress_node_surf| / rho0
 *     forc_botfrict    zeroed every call and passed (no function yet — as the Fortran does)
 *     forc_rhosurf     zeroed every call and passed (ditto)
 *
 *   Gated by diag_on: the 10 budget slabs + 3 cvmix dummies, each [N*nl]. The column core
 *   ALWAYS computes them into local scratch (there are internal read-after-write deps);
 *   diag_on gates only whether they are STORED globally. Model state is byte-identical
 *   either way (the C's T3 gate proved it).
 *
 *   NOT allocated (zero/dead by construction under only_tke=T, dolangmuir=F):
 *   tke_in3d_iwe/iwdis/iwealphac, tke_langmuir, langmuir_* — the column call passes a
 *   shared per-column zero array instead (value-identical; iw_diss IS read unconditionally
 *   into tke_Tiwf, so it must be a REAL zero array, not a skipped argument). The
 *   module-level 1D tke_Av_old/tke_Kv_old/tke_old allocatables + tke_diss are shadowed dead
 *   code in the Fortran and are not ported.
 */
typedef struct fesom_tke {
    int n_nod;          /* myDim+eDim nodes */
    int nl;             /* layer interfaces */
    int diag_on;        /* 1 -> the 13 diag slabs below are allocated + stored */

    /* always allocated [N*nl] */
    real_t *tke;        /* prognostic TKE at interfaces */
    real_t *tke_Av;     /* node viscosity  (KappaM) */
    real_t *tke_Kv;     /* node diffusivity (KappaH) */

    /* always allocated [N] */
    real_t *forc_normstress;  /* |stress_node_surf| / rho0 */
    real_t *forc_botfrict;    /* zeroed every call */
    real_t *forc_rhosurf;     /* zeroed every call */

    /* diag slabs [N*nl] — unallocated unless diag_on */
    real_t *Tbpr;       /* buoyancy production */
    real_t *Tspr;       /* shear production */
    real_t *Tdif;       /* vertical diffusion of TKE */
    real_t *Tdis;       /* dissipation */
    real_t *Twin;       /* wind forcing */
    real_t *Tiwf;       /* internal-wave forcing (identically 0 under only_tke) */
    real_t *Tbck;       /* background restoring (the tke_min reset) */
    real_t *Ttot;       /* total tendency */
    real_t *Lmix;       /* mixing length */
    real_t *Pr;         /* Prandtl number */
    real_t *dummy1;     /* cvmix_dummy_1 = KappaH_out */
    real_t *dummy2;     /* cvmix_dummy_2 = KappaM_out */
    real_t *dummy3;     /* cvmix_dummy_3 = Nsqr */

    /* Each persistent array above is OWNED by the matching fesom::Field; the raw pointer is
     * a NON-OWNING alias re-pointed at field.h() right after field.alloc() (the M1.2/D12
     * pattern, as in fesom_kpp.h). The slabs are recomputed in place every step — never
     * pointer-swapped — so the alias stays valid for the whole run. The device kernels read
     * and write the .d() views. */
    fesom::Field tke_fld, tke_Av_fld, tke_Kv_fld;
    fesom::Field forc_normstress_fld, forc_botfrict_fld, forc_rhosurf_fld;
    fesom::Field Tbpr_fld, Tspr_fld, Tdif_fld, Tdis_fld, Twin_fld, Tiwf_fld, Tbck_fld,
                 Ttot_fld, Lmix_fld, Pr_fld, dummy1_fld, dummy2_fld, dummy3_fld;

    /* Port of the C's file-static `s_zero_col` (fesom_tke.c:48): a [TKE_NL_MAX+1] array of
     * zeros passed as alpha_c / E_iw / iw_diss / tke_plc. The first three are gate-only, but
     * ⚠️ **iw_diss is read UNCONDITIONALLY** by the core (:898, tke_Tiwf = iw_diss), so this
     * must be a REAL zero array, never a null pointer. It is never written — so, unlike
     * bc_index_nod2D, its zeroed device mirror from Field::alloc is exactly right and needs
     * no modify_host()/sync_device(). */
    fesom::Field zero_col_fld;
} fesom_tke;

/*
 * Allocate + zero the state (mirror of init_cvmix_tke's allocate block, gen:151-207, and of
 * the C's fesom_tke_alloc). diag_on additionally allocates the 13 diag slabs. Also runs the
 * gate-only guard: the port covers exactly the reference configuration, so it fails loudly
 * rather than silently running un-ported physics (the constants themselves are checked at
 * COMPILE time in fesom_cvmix_tke.hpp).
 *
 * Called ONLY when FESOM_MIX_SCHEME selects TKE — matching the C (fesom_main.c:357-377) and
 * the Fortran (oce_setup_step.F90:185-187 runs the init only for mix_scheme_nmb==5).
 */
void fesom_tke_alloc(fesom_tke *t, const struct fesom_mesh *mesh, int diag_on);
void fesom_tke_free (fesom_tke *t);

/*
 * TKE driver — mirror of calc_cvmix_tke (gen_modules_cvmix_tke.F90:279-507).
 * Per OWNED node (gen:296,311 node_size = myDim_nod2D — do NOT extend to the halo): build
 * normstress, vshear2 (UVnode/Z_3d_n), bvfreq2, dz_trr and the per-column tke_old copy; call
 * the column core over nun:nln+1; zero tke_Av/tke_Kv at nun and nln+1. Then
 * exchange(tke_Kv) -> full aux->Kv copy; exchange(tke_Av) -> node->elem mean into aux->Av
 * over OWNED elements (gen:500), interior levels only. NO element exchange here — the
 * element-Av halo comes from the shared post-mo_convect ELEM3D exchange in fesom_step.cpp,
 * exactly as for KPP/PP. `tke` itself is NEVER exchanged (no reader needs its halo).
 *
 * Task 1.3 wires the body. Until then this aborts, so selecting TKE fails loudly rather
 * than silently running with zero mixing (the C staged it the same way).
 */
void fesom_tke_mixing_kk(fesom_tke                  *t,
                         struct fesom_aux           *aux,
                         const struct fesom_forcing *forcing,
                         const struct fesom_dyn     *dyn,
                         const struct fesom_mesh    *mesh,
                         struct fesom_partit        *partit);

/* diag_on inputs, mirroring the C (fesom_main.c:366-370):
 *   FESOM_TKE_DIAG=1              -> store the 13 diag slabs
 *   FESOM_TKE_DUMP_DIR=<path> set -> also implies diag (the C's dump reads the stored slabs)
 * NOTE: the port stores the slabs but does NOT write the C's dump files — bisection against
 * the archived /work/.../tke/{cdump,cdump_v2,replay} rails is done with the C oracle binary.
 * fesom_tke_alloc says so once, out loud, if FESOM_TKE_DUMP_DIR is set. */
const char *fesom_tke_dump_dir(void);

#endif /* FESOM_TKE_H */
