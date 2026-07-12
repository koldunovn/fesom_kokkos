#ifndef FESOM_CVMIX_TKE_HPP
#define FESOM_CVMIX_TKE_HPP

#include "fesom_types.h"

/*
 * CVMix classical-TKE column core — port of the C oracle's fesom_cvmix_tke.{c,h}
 * (itself a strictly faithful port of cvmix_tke.F90, module cvmix_tke: init_tke +
 * integrate_tke, the pure column math with no MPI/mesh).
 *
 * M6.1 staging (mirrors the C's own file split):
 *   Task 1.1 (this)  — the parameter block + the gate-only guard.
 *   Task 1.2 (next)  — integrate_tke as a KOKKOS_INLINE_FUNCTION column core.
 *
 * The C reads these 15 values from namelist.cvmix &param_tke via gen:258-272 and
 * hardcodes them at its single call site (fesom_tke.c:227-241). They are compile-time
 * constants in practice, so here they are `constexpr` — which additionally makes them
 * usable inside a device lambda with no __constant__ memory or params-struct copy.
 *
 * ⚠️ tke_cd = 3.75 is the NAMELIST value. The Fortran MODULE default is 1.0 and it LOSES
 * (feedback_namelist_over_codedefault). A port that silently inherited the module default
 * would run different surface-TKE forcing and still look plausible.
 *
 * Values re-verified 2026-07-12 against jobs/m6_namelists/tke/namelist.cvmix (the archived
 * namelist of the run that produced /work/.../tke/fortran_linfs_tke) — all 15 match the C.
 */

/* Per-thread column-scratch cap (C: fesom_cvmix_tke.c:28). CORE2 nl=47, NG5 nl=70. */
enum { TKE_NL_MAX = 128 };

struct fesom_cvmix_tke_params {
    real_t c_k;                   /* {0.1}   KappaM = c_k·L·√e                            */
    real_t c_eps;                 /* {0.7}   dissipation coefficient                      */
    real_t cd;                    /* {3.75}  surface-TKE coefficient — the NAMELIST value;
                                             the executed NEUMANN BC uses it (:793). The
                                             "3.75 = Dirichlet" comment in the Fortran is
                                             advisory only.                                */
    real_t alpha_tke;             /* {30.0}  TKE diffusivity multiplier + L bound          */
    real_t clc;                   /* {0.3}   Langmuir factor (gate-only)                   */
    real_t mxl_min;               /* {1e-8}  mixing-length floor                           */
    real_t kappaM_min;            /* {0.0}   read into a local, never applied (the
                                             commented-out :663 clamp) — kept for parity    */
    real_t kappaM_max;            /* {100.0} viscosity clamp                                */
    real_t tke_surf_min;          /* {1e-4}  min surface TKE                                */
    real_t tke_min;               /* {1e-6}  min interior TKE (the only_tke floor)          */
    int    tke_mxl_choice;        /* {2}     Blanke-Delecluse (the only implemented option) */
    int    only_tke;              /* {1}                                                     */
    int    l_lc;                  /* {0}     Langmuir gate                                   */
    int    use_ubound_dirichlet;  /* {0}                                                     */
    int    use_lbound_dirichlet;  /* {0}                                                     */
};

/* The reference configuration (C: fesom_tke.c:227-241, arg-for-arg). */
inline constexpr fesom_cvmix_tke_params FESOM_TKE_PARAMS = {
    /* c_k                  */ 0.1,
    /* c_eps                */ 0.7,
    /* cd                   */ 3.75,
    /* alpha_tke            */ 30.0,
    /* clc                  */ 0.3,
    /* mxl_min              */ 1.0e-8,
    /* kappaM_min           */ 0.0,
    /* kappaM_max           */ 100.0,
    /* tke_surf_min         */ 1.0e-4,
    /* tke_min              */ 1.0e-6,
    /* tke_mxl_choice       */ 2,
    /* only_tke             */ 1,
    /* l_lc                 */ 0,
    /* use_ubound_dirichlet */ 0,
    /* use_lbound_dirichlet */ 0,
};

/* Gate-only guard (C: fesom_tke.c:246-253 — a runtime abort). The port covers exactly the
 * reference configuration; IDEMIX / Langmuir / Dirichlet BCs / mxl_choice!=2 are NOT ported.
 * Because the params are constexpr here, the C's runtime abort becomes a compile-time
 * assertion — strictly stronger, and it cannot be skipped by a build that never selects TKE.
 * fesom_tke_alloc keeps a runtime check too, so an edit fails loudly either way. */
static_assert(FESOM_TKE_PARAMS.only_tke == 1,
              "fesom_cvmix_tke: only_tke=F (IDEMIX coupling) is not ported");
static_assert(FESOM_TKE_PARAMS.l_lc == 0,
              "fesom_cvmix_tke: Langmuir turbulence (tke_dolangmuir=T) is not ported");
static_assert(FESOM_TKE_PARAMS.use_ubound_dirichlet == 0 &&
              FESOM_TKE_PARAMS.use_lbound_dirichlet == 0,
              "fesom_cvmix_tke: Dirichlet TKE boundary conditions are not ported");
static_assert(FESOM_TKE_PARAMS.tke_mxl_choice == 2,
              "fesom_cvmix_tke: only tke_mxl_choice=2 (Blanke-Delecluse) is ported");

#endif /* FESOM_CVMIX_TKE_HPP */
