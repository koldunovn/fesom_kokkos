#ifndef FESOM_CVMIX_TKE_HPP
#define FESOM_CVMIX_TKE_HPP

#include "fesom_types.h"

#include <Kokkos_Core.hpp>
#include <type_traits>   /* M8: common_type in tke_min2/max2 */

/*
 * CVMix classical-TKE column core — port of the C oracle's fesom_cvmix_tke.{c,h}
 * (itself a strictly faithful port of cvmix_tke.F90, module cvmix_tke: init_tke +
 * integrate_tke, the pure column math with no MPI/mesh). Every formula, loop bound,
 * branch and evaluation order mirrors the C, which mirrors the Fortran (line refs inline
 * are the FORTRAN's, carried over from the C so the three trees stay cross-referenceable).
 *
 * ── BIT-FIDELITY LANDMINES (ported verbatim from the C; do NOT "clean up") ───────────
 *  - The Intel reference build compiles with **-r8** (src/CMakeLists.txt:335), so
 *    default-real literals like `6.6` (:717) and `0.5` are DOUBLE. Port them as plain C++
 *    double literals. The C empirically pinned this: `(double)6.6f` produced prandtl diffs
 *    of exactly (6.6 − (double)6.6f)·Rinum at 27k entries. TKE_C66 below is a plain 6.6.
 *  - `forc_tke_surf**(3./2.)` (:793, :886): 3./2. is exactly 1.5, and Intel under
 *    -fp-model precise emits a pow call → `pow(x, 1.5)`, NOT `x*sqrt(x)`. If a dump-diff
 *    ever shows last-bit noise exactly here, the compiler used x*sqrt(x) — switch
 *    tke_pow32 accordingly.
 *  - solve_tridiag (cvmix_utils_addon.F90:116-149) divides via RECIPROCAL
 *    (`fxa = 1D0/m; cp = c*fxa`) — ported as-is, NOT as `cp = c/m`. The two differ in the
 *    last bit.
 *  - Fortran max/min → compare-select ternaries (matching maxsd/minsd operand semantics
 *    for finite values), the codebase convention.
 *
 * ── DEAD ARGUMENTS (verified dead by the C's own review; kept for call-site parity) ──
 *  - old_KappaM / old_KappaH: intent(in) at :474-475, never read. There is NO old-value
 *    blending — KappaM_out (:704) and KappaH_out (:720) are FRESH overwrites.
 *  - handle_old_vals, max_nlev, i/j/tstep_count, tke_userdef_constants: all dead
 *    (see the C header's verified list).
 *  - `bottom_fric` is KEPT in the signature although the executed body never reads it
 *    (the Fortran driver passes tke_forc2d_botfrict(node) ≡ 0).
 *
 * ── GATE-ONLY BRANCHES (not ported; the constants are static_asserted below) ──────────
 *  - IDEMIX coupling (.not.only_tke): the Rinum modification (:710-712) and the iw_diss
 *    forcing (:742-744). alpha_c/E_iw are read ONLY under that gate.
 *  - Langmuir (l_lc): forc += tke_plc (:737-739).
 *  - Dirichlet upper/lower BCs (:780-790, :799-811, :841-849, :881-883, :890-892). The
 *    executed NEUMANN branches are ported.
 *  ⚠️ `iw_diss` is READ UNCONDITIONALLY at :898 (tke_Tiwf = iw_diss) → the caller MUST
 *    pass a REAL zero array of length nlev+1, never a null pointer.
 *
 * M6.1 staging (mirrors the C's own file split):
 *   Task 1.1 — the parameter block + the gate-only guard.
 *   Task 1.2 — integrate_tke as a KOKKOS_INLINE_FUNCTION column core (below).
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

/*===========================================================================================
 * Landmine helpers — see the header comment. These ARE the C's macros, expressed as
 * templates so the arguments are evaluated once (the C's TKE_MIN2/TKE_MAX2 macros evaluate
 * theirs twice; every call site passes simple expressions, so the codegen is identical, but
 * the template form cannot be mis-called). Compare-select, NOT fmin/fmax: the Fortran
 * min/max map to maxsd/minsd operand semantics for finite values.
 *===========================================================================================*/
/* M8: two-type form with common_type promotion — mixed (double literal, real_t)
 * calls compute in double exactly like the Fortran r8 literals demand (see the
 * 6.6 note below); at FP64 both types are double, bit-identical to the old form. */
template <class A, class B>
KOKKOS_INLINE_FUNCTION constexpr typename std::common_type<A, B>::type
tke_min2(A a, B b)
{
    using T = typename std::common_type<A, B>::type;
    return ((T)a < (T)b) ? (T)a : (T)b;
}
template <class A, class B>
KOKKOS_INLINE_FUNCTION constexpr typename std::common_type<A, B>::type
tke_max2(A a, B b)
{
    using T = typename std::common_type<A, B>::type;
    return ((T)a > (T)b) ? (T)a : (T)b;
}

/* The 6.6 literal at :717 — plain double (the -r8 rule). */
inline constexpr real_t TKE_C66 = 6.6;

/* x**(3./2.) at :793 / :886 — a pow call, not x*sqrt(x). Kokkos::pow is std::pow on the
 * host backend (so Serial is bit-identical to the C oracle) and libdevice pow on CUDA. */
KOKKOS_INLINE_FUNCTION real_t tke_pow32(real_t x) { return Kokkos::pow(x, 1.5); }

/*
 * Budget-diagnostic copy-out targets — each pointer is a [nlev+1] column slice, or nullptr
 * to discard. integrate_tke ALWAYS computes the full set into local column scratch in
 * Fortran order and copies each non-null target at the END of the column only; diag writes
 * NEVER go through these pointers mid-computation.
 */
struct fesom_cvmix_tke_diag {
    real_t *Tbpr;   /* -P_diss_v          buoyancy production                */
    real_t *Tspr;   /*  K_diss_v          shear production                   */
    real_t *Tdif;   /* tridiag residual   vertical diffusion                 */
    real_t *Tdis;   /* -c_eps/L·√e·e_new  dissipation                        */
    real_t *Twin;   /* surface wind forcing (Neumann: cd·f^1.5/dzt1)         */
    real_t *Tiwf;   /* = iw_diss (identically 0 under only_tke)              */
    real_t *Tbck;   /* (tke_new - tke_unrest)/dt   the tke_min reset         */
    real_t *Ttot;   /* (tke_new - tke_old)/dt                                */
    real_t *Lmix;   /* mixing length mxl                                     */
    real_t *Pr;     /* Prandtl number                                        */
    real_t *int1;   /* cvmix_int_1 = KappaH_out                              */
    real_t *int2;   /* cvmix_int_2 = KappaM_out                              */
    real_t *int3;   /* cvmix_int_3 = Nsqr                                    */
};

/*
 * solve_tridiag (cvmix_utils_addon.F90:116-149) — Thomas with the exact reciprocal-multiply
 * form. n = nlev+1 equations, 0-based. ⚠️ `fxa = 1.0/m; cp = c*fxa` — NOT `cp = c/m`.
 */
KOKKOS_INLINE_FUNCTION
void tke_solve_tridiag(const real_t *a, const real_t *b, const real_t *c, const real_t *d,
                       real_t *x, int n)
{
    real_t cp[TKE_NL_MAX + 1], dp[TKE_NL_MAX + 1];
    cp[0] = c[0] / b[0];                                         /* :134 */
    dp[0] = d[0] / b[0];                                         /* :135 */
    for (int i = 1; i < n; ++i) {                                /* :137 */
        real_t m   = b[i] - cp[i - 1] * a[i];
        real_t fxa = 1.0 / m;                                    /* :139  RECIPROCAL */
        cp[i] = c[i] * fxa;
        dp[i] = (d[i] - dp[i - 1] * a[i]) * fxa;
    }
    x[n - 1] = dp[n - 1];                                        /* :144 */
    for (int i = n - 2; i >= 0; --i)                             /* :146 */
        x[i] = dp[i] - cp[i] * x[i + 1];
}

/*===========================================================================================
 * integrate_tke (:415-987). 0-based k <-> Fortran k+1; arrays span k = 0..nlev (interfaces)
 * except dzw (0..nlev-1, layers). Executed path: tke_mxl_choice=2, Neumann BCs both ends,
 * only_tke (the tke_min floor applies), l_lc off.
 *
 * Outputs: tke_new (solved + floored), KappaM_out = min(KappaM_max, c_k·mxl·√tke) (a FRESH
 * overwrite), KappaH_out = KappaM_out/prandtl.
 *
 * WITH_DIAG is a template parameter rather than a runtime `if (diag)`: the 13 d_* budget
 * arrays are pure OUTPUTS (nothing in tke_new / KappaM_out / KappaH_out reads them back),
 * so with WITH_DIAG=false the compiler proves them dead and eliminates all 13 [129]-double
 * locals — dropping the per-thread frame from ~33 KB to ~20 KB on CUDA. The COMPUTATION is
 * still written unconditionally, exactly as the C writes it: this is dead-code elimination
 * of unused outputs, not a reordering, so the numerics and their evaluation order are
 * untouched. (The C's own T3 gate proved diag-on and diag-off model state byte-identical.)
 *===========================================================================================*/
template <bool WITH_DIAG>
KOKKOS_INLINE_FUNCTION
void fesom_cvmix_integrate_tke(int           nlev,
                               real_t        dtime,
                               real_t        rho_ref,
                               real_t        grav,
                               const real_t *dzw,        /* [nlev]   */
                               const real_t *dzt,        /* [nlev+1] */
                               const real_t *tke_old,    /* [nlev+1] */
                               const real_t *Ssqr,       /* [nlev+1] */
                               const real_t *Nsqr,       /* [nlev+1] */
                               const real_t *alpha_c,    /* gated (.not.only_tke) — unread */
                               const real_t *E_iw,       /* gated (.not.only_tke) — unread */
                               const real_t *iw_diss,    /* [nlev+1] READ UNCONDITIONALLY  */
                               const real_t *tke_plc,    /* gated (l_lc) — unread          */
                               real_t        forc_tke_surf,
                               real_t        forc_rho_surf,
                               real_t        bottom_fric, /* unread in the executed body   */
                               real_t       *tke_new,    /* out [nlev+1] */
                               real_t       *KappaM_out, /* out [nlev+1] */
                               real_t       *KappaH_out, /* out [nlev+1] */
                               const fesom_cvmix_tke_diag *diag)
{
    (void)alpha_c;      /* read only under .not.only_tke (:711) — gate-only */
    (void)E_iw;         /* read only under .not.only_tke (:711) — gate-only */
    (void)tke_plc;      /* read only under l_lc (:737-739) — gate-only */
    (void)bottom_fric;  /* never read in the executed body */

    /* local column scratch (:535-581). The 13 diag terms are ALWAYS computed here in
     * Fortran order and copied to the nullable targets at the END — never written through
     * `diag` mid-computation. */
    real_t tke_unrest[TKE_NL_MAX + 1], tke_upd[TKE_NL_MAX + 1];
    real_t mxl[TKE_NL_MAX + 1], sqrttke[TKE_NL_MAX + 1];
    real_t prandtl[TKE_NL_MAX + 1], Rinum[TKE_NL_MAX + 1];
    real_t K_diss_v[TKE_NL_MAX + 1], P_diss_v[TKE_NL_MAX + 1];
    real_t forc[TKE_NL_MAX + 1];
    real_t a_dif[TKE_NL_MAX + 1], b_dif[TKE_NL_MAX + 1], c_dif[TKE_NL_MAX + 1];
    real_t a_tri[TKE_NL_MAX + 1], b_tri[TKE_NL_MAX + 1], c_tri[TKE_NL_MAX + 1];
    real_t d_tri[TKE_NL_MAX + 1], ke[TKE_NL_MAX + 1];
    real_t d_Tbpr[TKE_NL_MAX + 1], d_Tspr[TKE_NL_MAX + 1];
    real_t d_Tdif[TKE_NL_MAX + 1], d_Tdis[TKE_NL_MAX + 1];
    real_t d_Twin[TKE_NL_MAX + 1], d_Tiwf[TKE_NL_MAX + 1];
    real_t d_Tbck[TKE_NL_MAX + 1], d_Ttot[TKE_NL_MAX + 1];
    real_t d_Lmix[TKE_NL_MAX + 1], d_Pr[TKE_NL_MAX + 1];
    real_t d_int1[TKE_NL_MAX + 1], d_int2[TKE_NL_MAX + 1], d_int3[TKE_NL_MAX + 1];
    real_t tke_surf, diff_surf_forc, diff_bott_forc;
    int    k;

    /* :590 tke_constants_saved; :629-642 the local aliases. */
    constexpr real_t alpha_tke  = FESOM_TKE_PARAMS.alpha_tke;
    constexpr real_t c_eps      = FESOM_TKE_PARAMS.c_eps;
    constexpr real_t cd         = FESOM_TKE_PARAMS.cd;
    constexpr real_t KappaM_max = FESOM_TKE_PARAMS.kappaM_max;
    constexpr real_t mxl_min    = FESOM_TKE_PARAMS.mxl_min;
    constexpr real_t c_k        = FESOM_TKE_PARAMS.c_k;
    constexpr real_t tke_min    = FESOM_TKE_PARAMS.tke_min;
    constexpr int    only_tke   = FESOM_TKE_PARAMS.only_tke;
    constexpr int    l_lc       = FESOM_TKE_PARAMS.l_lc;
    /* tke_surf_min (:636) is used only in the gated Dirichlet branch; kappaM_min = 0.0
     * (:662) with the :663 clamp commented out in the Fortran. */

    if (nlev + 1 > TKE_NL_MAX)
        Kokkos::abort("fesom_cvmix_integrate_tke: nlev exceeds TKE_NL_MAX");

    /* initialise diagnostics + work arrays (:602-623). Entries not later overwritten keep 0. */
    for (k = 0; k <= nlev; ++k) {
        d_Tbpr[k] = 0.0; d_Tspr[k] = 0.0; d_Tdif[k] = 0.0; d_Tdis[k] = 0.0;
        d_Twin[k] = 0.0; d_Tiwf[k] = 0.0; d_Tbck[k] = 0.0; d_Ttot[k] = 0.0;
        d_int1[k] = 0.0; d_int2[k] = 0.0; d_int3[k] = 0.0;
        tke_new[k] = 0.0;
        tke_upd[k] = 0.0;
        a_dif[k] = 0.0; b_dif[k] = 0.0; c_dif[k] = 0.0;
        a_tri[k] = 0.0; b_tri[k] = 0.0; c_tri[k] = 0.0;
    }
    tke_surf = 0.0;
    (void)tke_surf;   /* consumed only by the gated Dirichlet branch (:783) */

    /*-------------------------------------------------------------------------------------
     * Part 1: mixing length scale (:666-698)
     *-----------------------------------------------------------------------------------*/
    for (k = 0; k <= nlev; ++k)                                  /* :668 */
        sqrttke[k] = Kokkos::sqrt(tke_max2(0.0, tke_old[k]));
    for (k = 0; k <= nlev; ++k)                                  /* :671 */
        mxl[k] = Kokkos::sqrt(2.0) * sqrttke[k] / Kokkos::sqrt(tke_max2(1.0e-12, Nsqr[k]));

    /* executed: tke_mxl_choice==2 (:674-685); choice 3 not ported (the reference pins 2;
     * the static_asserts above enforce it, mirroring the Fortran's :695-697 stop). */
    mxl[0]    = 0.0;                                             /* :676 */
    mxl[nlev] = 0.0;                                             /* :677 */
    for (k = 1; k <= nlev - 1; ++k)                              /* :678 */
        mxl[k] = tke_min2(mxl[k], mxl[k - 1] + dzw[k - 1]);
    mxl[nlev - 1] = tke_min2(mxl[nlev - 1], mxl_min + dzw[nlev - 1]);   /* :681 */
    for (k = nlev - 2; k >= 1; --k)                              /* :682 */
        mxl[k] = tke_min2(mxl[k], mxl[k + 1] + dzw[k]);
    for (k = 0; k <= nlev; ++k)                                  /* :685 */
        mxl[k] = tke_max2(mxl[k], mxl_min);

    /*-------------------------------------------------------------------------------------
     * Part 2: diffusivities (:700-720)
     *-----------------------------------------------------------------------------------*/
    for (k = 0; k <= nlev; ++k)                                  /* :704  FRESH overwrite */
        KappaM_out[k] = tke_min2(KappaM_max, c_k * mxl[k] * sqrttke[k]);
    for (k = 0; k <= nlev; ++k)                                  /* :705 */
        Rinum[k] = Nsqr[k] / tke_max2(Ssqr[k], 1.0e-12);

    /* :710-712 — IDEMIX Ri modification: gate-only (.not.only_tke never executes;
     * alpha_c/E_iw unread). */

    for (k = 0; k <= nlev; ++k)                                  /* :717 */
        prandtl[k] = tke_max2(1.0, tke_min2(10.0, TKE_C66 * Rinum[k]));
    for (k = 0; k <= nlev; ++k)                                  /* :720 */
        KappaH_out[k] = KappaM_out[k] / prandtl[k];

    /*-------------------------------------------------------------------------------------
     * Part 3: tke forcing (:723-744)
     *-----------------------------------------------------------------------------------*/
    for (k = 0; k <= nlev; ++k)                                  /* :726 */
        forc[k] = 0.0;
    for (k = 0; k <= nlev; ++k)                                  /* :729-730 */
        K_diss_v[k] = Ssqr[k] * KappaM_out[k];
    for (k = 0; k <= nlev; ++k)
        P_diss_v[k] = Nsqr[k] * KappaH_out[k];
    P_diss_v[0] = -forc_rho_surf * grav / rho_ref;               /* :733 */
    for (k = 0; k <= nlev; ++k)                                  /* :734 */
        forc[k] = forc[k] + K_diss_v[k] - P_diss_v[k];

    if (l_lc) {                                                  /* :737-739  gate-only */
        for (k = 0; k <= nlev; ++k)
            forc[k] = forc[k] + tke_plc[k];
    }

    /* :742-744 — iw_diss forcing: gate-only (.not.only_tke). */

    /*-------------------------------------------------------------------------------------
     * Part 4: implicit vertical diffusion + dissipation (:746-827)
     *-----------------------------------------------------------------------------------*/
    for (k = 0; k <= nlev; ++k)                                  /* :749 */
        ke[k] = 0.0;
    for (k = 0; k <= nlev - 1; ++k) {                            /* :750-754 */
        /* Fortran k_F = k+1; kp1 = min(k_F+1, nlev); kk = max(k_F, 2) — 0-based: */
        int kp1 = tke_min2(k + 1, nlev - 1);
        int kk  = tke_max2(k, 1);
        ke[k] = alpha_tke * 0.5 * (KappaM_out[kp1] + KappaM_out[kk]);
    }

    for (k = 0; k <= nlev - 1; ++k)                              /* :757-760 */
        c_dif[k] = ke[k] / (dzt[k] * dzw[k]);
    c_dif[nlev] = 0.0;                                           /* :761 */

    for (k = 1; k <= nlev - 1; ++k)                              /* :764-767 */
        b_dif[k] = ke[k - 1] / (dzt[k] * dzw[k - 1]) + ke[k] / (dzt[k] * dzw[k]);

    for (k = 1; k <= nlev; ++k)                                  /* :770-773 */
        a_dif[k] = ke[k - 1] / (dzt[k] * dzw[k - 1]);
    a_dif[0] = 0.0;                                              /* :774 */

    for (k = 0; k <= nlev; ++k)                                  /* :777 */
        tke_upd[k] = tke_old[k];

    /* upper boundary: the executed NEUMANN branch (:791-796); the Dirichlet body
     * (:780-790) is NOT ported (the static_asserts forbid enabling it). */
    forc[0] = forc[0] + (cd * tke_pow32(forc_tke_surf)) / dzt[0];   /* :793 */
    b_dif[0] = ke[0] / (dzt[0] * dzw[0]);                           /* :794 */
    diff_surf_forc = 0.0;                                           /* :795 */

    /* lower boundary: the executed NEUMANN branch (:812-815); Dirichlet (:799-811) not ported. */
    b_dif[nlev] = ke[nlev - 1] / (dzt[nlev] * dzw[nlev - 1]);       /* :813 */
    diff_bott_forc = 0.0;                                           /* :814 */

    for (k = 0; k <= nlev; ++k) {                                /* :818-821 */
        a_tri[k] = -dtime * a_dif[k];
        b_tri[k] = 1.0 + dtime * b_dif[k];
        c_tri[k] = -dtime * c_dif[k];
    }
    for (k = 1; k <= nlev - 1; ++k)                              /* :820 */
        b_tri[k] = b_tri[k] + dtime * c_eps * sqrttke[k] / mxl[k];

    for (k = 0; k <= nlev; ++k)                                  /* :824 */
        d_tri[k] = tke_upd[k] + dtime * forc[k];

    tke_solve_tridiag(a_tri, b_tri, c_tri, d_tri, tke_new, nlev + 1);  /* :827 */

    /* implicit-tendency diagnostics (:829-849) — computed from the PRE-floor tke_new. */
    for (k = 1; k <= nlev - 1; ++k)                              /* :831-833 */
        d_Tdif[k] = a_dif[k] * tke_new[k - 1] - b_dif[k] * tke_new[k]
                  + c_dif[k] * tke_new[k + 1];
    d_Tdif[0]    = -b_dif[0] * tke_new[0] + c_dif[0] * tke_new[1];                  /* :834 */
    d_Tdif[nlev] = a_dif[nlev] * tke_new[nlev - 1] - b_dif[nlev] * tke_new[nlev];   /* :835 */
    d_Tdif[1]        = d_Tdif[1] + diff_surf_forc;                                  /* :836 */
    d_Tdif[nlev - 1] = d_Tdif[nlev - 1] + diff_bott_forc;                           /* :837 */
    /* :841-849 — Dirichlet Tdif overrides: gate-only. */

    /* dissipation of TKE (:852-853) */
    for (k = 0; k <= nlev; ++k)
        d_Tdis[k] = 0.0;
    for (k = 1; k <= nlev - 1; ++k)
        d_Tdis[k] = -c_eps / mxl[k] * sqrttke[k] * tke_new[k];

    /*-------------------------------------------------------------------------------------
     * Part 5: reset tke to bounding values (:856-869)
     *-----------------------------------------------------------------------------------*/
    for (k = 0; k <= nlev; ++k)                                  /* :860 */
        tke_unrest[k] = tke_new[k];

    if (only_tke) {                                              /* :867-869 */
        for (k = 0; k <= nlev; ++k)
            tke_new[k] = tke_max2(tke_new[k], tke_min);
    }

    /*-------------------------------------------------------------------------------------
     * Part 6: assign diagnostic variables (:871-904)
     *-----------------------------------------------------------------------------------*/
    for (k = 0; k <= nlev; ++k)                                  /* :876-877 */
        d_Tbpr[k] = -P_diss_v[k];
    for (k = 0; k <= nlev; ++k)
        d_Tspr[k] = K_diss_v[k];
    for (k = 0; k <= nlev; ++k)                                  /* :880 */
        d_Tbck[k] = (tke_new[k] - tke_unrest[k]) / dtime;
    /* surface: the executed Neumann else-branch (:885-886); Dirichlet (:882-883) gate-only. */
    d_Twin[0] = (cd * tke_pow32(forc_tke_surf)) / dzt[0];        /* :886 */
    /* bottom: the executed Neumann else-branch (:893-895). */
    d_Twin[nlev] = 0.0;                                          /* :895 */

    for (k = 0; k <= nlev; ++k)                                  /* :898 unconditional read */
        d_Tiwf[k] = iw_diss[k];   /* zero column under only_tke — must be a REAL array */
    for (k = 0; k <= nlev; ++k)                                  /* :899 */
        d_Ttot[k] = (tke_new[k] - tke_old[k]) / dtime;
    /* :901-904 — tke_Lmix(nlev+1:)=0 / tke_Pr(nlev+1:)=0 touch only element nlev of the
     * slice, then the full-range assignment overwrites it: */
    for (k = 0; k <= nlev; ++k)
        d_Lmix[k] = mxl[k];
    for (k = 0; k <= nlev; ++k)
        d_Pr[k] = prandtl[k];

    /* debugging copies (:909-911) */
    for (k = 0; k <= nlev; ++k) {
        d_int1[k] = KappaH_out[k];
        d_int2[k] = KappaM_out[k];
        d_int3[k] = Nsqr[k];
    }
    /* :916-986 — the if(.false.) debug print block: dead. */

    /* End-of-column diag copy-out — the ONLY writes through `diag`. (The C memcpy's; a loop
     * is the same value copy and is device-callable.) */
    if constexpr (WITH_DIAG) {
      if (diag) {                                  /* the C's outer `if (diag)` guard */
        const int n = nlev + 1;
        if (diag->Tbpr) for (k = 0; k < n; ++k) diag->Tbpr[k] = d_Tbpr[k];
        if (diag->Tspr) for (k = 0; k < n; ++k) diag->Tspr[k] = d_Tspr[k];
        if (diag->Tdif) for (k = 0; k < n; ++k) diag->Tdif[k] = d_Tdif[k];
        if (diag->Tdis) for (k = 0; k < n; ++k) diag->Tdis[k] = d_Tdis[k];
        if (diag->Twin) for (k = 0; k < n; ++k) diag->Twin[k] = d_Twin[k];
        if (diag->Tiwf) for (k = 0; k < n; ++k) diag->Tiwf[k] = d_Tiwf[k];
        if (diag->Tbck) for (k = 0; k < n; ++k) diag->Tbck[k] = d_Tbck[k];
        if (diag->Ttot) for (k = 0; k < n; ++k) diag->Ttot[k] = d_Ttot[k];
        if (diag->Lmix) for (k = 0; k < n; ++k) diag->Lmix[k] = d_Lmix[k];
        if (diag->Pr)   for (k = 0; k < n; ++k) diag->Pr[k]   = d_Pr[k];
        if (diag->int1) for (k = 0; k < n; ++k) diag->int1[k] = d_int1[k];
        if (diag->int2) for (k = 0; k < n; ++k) diag->int2[k] = d_int2[k];
        if (diag->int3) for (k = 0; k < n; ++k) diag->int3[k] = d_int3[k];
      }
    } else {
        (void)diag;
    }
}

#endif /* FESOM_CVMIX_TKE_HPP */
