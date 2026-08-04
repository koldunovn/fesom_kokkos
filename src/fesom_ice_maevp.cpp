/*
 * mEVP sea-ice rheology (whichEVP=1) — device kernels.
 * Literal port of the C oracle's fesom_ice_maevp.c (fesom2_port_zstar @ df8b9a8), itself a
 * literal port of EVPdynamics_m (ice_maEVP.F90:429-882, the NR-optimised routine with
 * ssh2rhs / stress_tensor_m / stress2rhs_m inlined). Fortran line refs are carried over from
 * the C so all three trees stay cross-referenceable.
 *
 * ⚠️ READ fesom_ice_maevp.h FIRST. It carries the 13-point list of mEVP-vs-std-EVP asymmetries
 * that are ported DELIBERATELY. Every one of them looks like a bug to a reader who knows the
 * std-EVP kernel in fesom_ice_evp.cpp. Do not "fix" any of them. The trap numbers below refer
 * to that list.
 *
 * Device shape: the structure mirrors the std-EVP device kernel (fesom_ice_evp.cpp:627-716) —
 * element kernel → node solve → BC → fused halo, per substep — but every VALUE and BRANCH comes
 * from the C mEVP, never from the std-EVP template.
 *
 * Element→node accumulations use element-order `Kokkos::atomic_add` (D22): on Serial the range
 * is sequential, so the accumulation order is exactly the C loop's => bit-identical. On CUDA the
 * order is nondeterministic; that is the documented climate-close class the fidelity gate covers.
 */
#include "fesom_ice_maevp.h"
#include "fesom_ice_types.h"
#include "fesom_types.h"
#include "fesom_constants.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_speed.hpp"

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>

/* ══════════════════════════════════════════════════════════════════════════════════════════
 *  M9 — divergence-subcycled mEVP.  docs/plans/20260804-m9-mevp-divergence.md
 *
 *  S. Danilov's reformulation (ice_sergey/evp_div.tex): applying div to the mEVP stress
 *  recursion turns  sigma^{p+1} = det1*sigma^p + det2*sigma~  into
 *  d^{p+1} = det1*d^p + det2*div(sigma~)  EXACTLY — div is linear and the discrete divergence
 *  operator is time-invariant. The carried state therefore moves from 3 ELEMENT arrays
 *  (sigma11/12/22) to 2 NODE arrays (R_u/R_v). Possible only for mEVP, where all sigma
 *  components share one alpha; standard EVP subcycles them differently.
 *
 *  All knobs are default-off and announce once on rank 0. The announce is not cosmetic:
 *  FESOM_SPEED_SWSKIP and _IOACC once shipped SILENTLY DEAD on CUDA and passed EVERY
 *  correctness gate while doing so (fesom_speed.hpp:93-102) — knob-OFF bytes are exactly what
 *  a dead knob gives you. L102 (wsplit, 2026-08-04) sharpened it: a knob can be alive and
 *  announcing while its code path is dead. So the M9 gates grep for evidence that the PATH
 *  ran, and this file emits one [m9] provenance line naming the active cell.
 * ══════════════════════════════════════════════════════════════════════════════════════════ */

/* Cell ③④⑤b — carry R_u/R_v at nodes instead of sigma at elements. Mathematically identical,
 * NOT bitwise (det1*sum(sigma) vs sum(det1*sigma) rounds differently, and the ice_el mask
 * changes membership across ocean steps) => solver class, the CGPOLY precedent. */
static bool maevp_div_on(void)   { static int c = -1; return fesom_speed_on_exp("MEVPDIV",   &c); }

/* Confound control. eps11/12/22 are written by the element kernel every subcycle and read
 * NOWHERE in the mEVP path (only std EVP reads those arrays, fesom_ice_evp.cpp:117-129), and
 * they cost about what the reformulation saves. Measured on its OWN leg so the reformulation
 * and the dead-write removal are never reported as one number. */
static bool maevp_noeps_on(void) { static int c = -1; return fesom_speed_on_exp("MEVPNOEPS", &c); }

/* Diagnostic: carry sigma AND R and compare. Perturbs the trajectory (it re-baselines R at
 * each step entry), so it is NEVER a timing or trajectory leg. See the plan §L2. */
static int maevp_div_selfcheck(void)
{
    static int c = -1;
    if (c < 0) { const char *e = getenv("FESOM_MEVPDIV_SELFCHECK"); c = e ? atoi(e) : 0; }
    return c;
}

/* Cell ⑤ — exchange (u_aux,v_aux) every K-th subcycle; ring 1 goes up to K-1 subcycles stale.
 * An explicit APPROXIMATION, never byte-neutral for K>1.
 *
 * fesom_speed_int is the SILENT helper (only the boolean resolvers announce), so this one owns
 * its announce, its guard, and its own asked-but-off shout — on Serial a value knob silently
 * returns the default unless FESOM_SPEED_FORCE_SERIAL=1, which would make an L1 null rung pass
 * because the knob is DEAD rather than because the lever is neutral. */
static int maevp_lag_K(const struct fesom_ice *ice, const struct fesom_partit *partit)
{
    static int cache = -2;
    static int announced = 0;
    const int K = fesom_speed_int("MEVPLAG", 1, &cache);
    if (announced) return K;
    announced = 1;

    if (K > 1) {
        /* The LAST subcycle must exchange. Otherwise the final u_ice = u_aux copy spanning
         * myDim+eDim (trap 8) publishes a STALE HALO into the ocean coupling — a different and
         * far worse error than a stale subcycle read. */
        FESOM_CHECK(ice->evp_rheol_steps % K == 0,
                    "MEVPLAG: K=%d must divide evp_rheol_steps=%d so the last subcycle exchanges",
                    K, ice->evp_rheol_steps);
        if (partit->mype == 0) {
            fprintf(stderr, "[fesom_speed] FESOM_SPEED_MEVPLAG = %d (LAGGED single halo: exchange "
                            "every %d-th subcycle, ring 1 up to %d subcycles stale; mEVP exchanges/"
                            "step %d -> %d) — APPROXIMATION, not byte-neutral\n",
                    K, K, K - 1, ice->evp_rheol_steps, ice->evp_rheol_steps / K);
            fflush(stderr);
        }
    } else {
        const char *e = getenv("FESOM_SPEED_MEVPLAG");
        if (e && atoi(e) > 1 && partit->mype == 0) {
            fprintf(stderr, "[fesom_speed] !! FESOM_SPEED_MEVPLAG=%s WAS REQUESTED BUT RESOLVES TO "
                            "1 — the lever is NOT running. (Serial backend without "
                            "FESOM_SPEED_FORCE_SERIAL=1?)\n", e);
            fflush(stderr);
        } else if (e && atoi(e) == 1 && partit->mype == 0) {
            /* K=1 explicitly requested = the NULL RUNG. It must announce, or the byte gate that
             * proves it neutral cannot tell "neutral" from "knob never resolved" — the whole
             * point of the rung. (The lever is genuinely inert at K=1: `shortstep % 1 == 0` is
             * always true, so the exchange fires every subcycle exactly as it does by default.) */
            fprintf(stderr, "[fesom_speed] FESOM_SPEED_MEVPLAG = 1 (NULL RUNG: exchange every "
                            "subcycle = default behaviour; must be byte-identical)\n");
            fflush(stderr);
        }
    }
    return K;
}

/* One provenance line naming the active cell of the 2x3 study matrix. Every M9 gate and every
 * A/B leg greps this; it is the single place that says what actually ran. */
static void maevp_announce_cell(const struct fesom_partit *partit,
                                bool div_on, bool noeps_on, int lagK, int selfcheck)
{
    static int announced = 0;
    if (announced || partit->mype != 0) return;
    announced = 1;
    fprintf(stderr, "[m9] mEVP cell: form=%s halo=%s eps=%s%s\n",
            div_on   ? "divergence(R_u,R_v @nodes)" : "classic(sigma @elements)",
            lagK > 1 ? "lagged"                     : "every-subcycle",
            noeps_on ? "dropped"                    : "kept",
            selfcheck ? "  [SELFCHECK ON — diagnostic mode, not a timing leg]" : "");
    if (lagK > 1) fprintf(stderr, "[m9] mEVP cell: lag K=%d\n", lagK);
    fflush(stderr);
}

/*
 * Coastal-node mask for the edge BC (:823-857). The C zeros u_ice_aux/v_ice_aux at the two
 * endpoints of every OWNED open-boundary edge (`myList_edge2D[ed] > edge2D_in` — the
 * rank-boundary-safe criterion, NOT edge_tri<0). `myList_edge2D` is host-only, so precompute a
 * per-node 0/1 device mask once; the BC kernel then zeros every marked node.
 *
 * This is the SAME construction (and therefore the same node set) as `evp_coastal_mask` in
 * fesom_ice_evp.cpp:493 — deliberately duplicated rather than shared, so that the two rheology
 * TUs stay independent and neither one's assumptions can leak into the other (the whole point
 * of the trap list). All writes are 0 => idempotent => race-free, no scatter => Serial AND CUDA
 * see the same result as the C's edge loop. Note the mask spans myDim+eDim: the C's edge loop
 * can also touch HALO endpoints, and it must (their values are overwritten by the exchange that
 * follows, so it is harmless either way — but we port it as written).
 */
static Kokkos::View<int*> g_maevp_coastal_mask;

/* ⚠️ A file-scope Kokkos::View is destructed at static-destruction time, i.e. AFTER
 * Kokkos::finalize() — which makes Kokkos abort ("View destructed after finalize"). It must be
 * released explicitly while Kokkos is still live. fesom_main.cpp calls this next to
 * fesom_ice_evp_free(), which exists for exactly the same reason (M5.8). Symptom if you forget:
 * the run completes, writes every snapshot, prints its timing — and THEN exits 134. */
void fesom_ice_maevp_free(void) { g_maevp_coastal_mask = Kokkos::View<int*>(); }

/* See the contract in fesom_ice_maevp.h. */
int fesom_ice_maevp_div_active(void) { return maevp_div_on() ? 1 : 0; }

static Kokkos::View<int*> maevp_coastal_mask(struct fesom_partit *partit, struct fesom_mesh *mesh)
{
    if (g_maevp_coastal_mask.extent(0) != 0) return g_maevp_coastal_mask;
    const int Nn = mesh->myDim_nod2D + mesh->eDim_nod2D;
    Kokkos::View<int*, Kokkos::HostSpace> h("maevp_coastal_h", Nn);
    for (int n = 0; n < Nn; ++n) h(n) = 0;
    for (int ed = 0; ed < mesh->myDim_edge2D; ++ed) {
        if (partit->myList_edge2D[ed] <= mesh->edge2D_in) continue;   /* interior edge */
        h(mesh->edges[ed * 2 + 0]) = 1;
        h(mesh->edges[ed * 2 + 1]) = 1;
    }
    g_maevp_coastal_mask = Kokkos::View<int*>("maevp_coastal_mask", Nn);
    Kokkos::deep_copy(g_maevp_coastal_mask, h);
    return g_maevp_coastal_mask;
}

void fesom_ice_evp_dynamics_m_kk(struct fesom_ice    *ice,
                                 struct fesom_partit *partit,
                                 struct fesom_mesh   *mesh)
{
    /* M9 knobs — resolved and announced at the TOP, before any kernel output, so the gate greps
     * find the lines in a known place. Resolution is cached; the announce fires once. */
    const bool m9_div       = maevp_div_on();
    const bool m9_noeps     = maevp_noeps_on();
    const int  m9_selfcheck = maevp_div_selfcheck();
    const int  m9_lagK      = maevp_lag_K(ice, partit);
    maevp_announce_cell(partit, m9_div, m9_noeps, m9_lagK, m9_selfcheck);

    const int N     = mesh->myDim_nod2D + mesh->eDim_nod2D;   /* node_size */
    const int myDim = mesh->myDim_nod2D;
    const int Eo    = mesh->myDim_elem2D;
    const int nl    = mesh->nl;

    /* setup (:513-518). ⚠️ TRAP 1: rdt is the FULL ice step — std EVP divides by
     * evp_rheol_steps, mEVP does NOT. */
    const real_t val3  = 1.0 / 3.0;
    const real_t vale  = 1.0 / (ice->ellipse * ice->ellipse);
    const real_t det2  = 1.0 / (1.0 + ice->alpha_evp);
    const real_t det1  = ice->alpha_evp * det2;
    const real_t rdt   = ice->ice_dt;
    const int    steps = ice->evp_rheol_steps;

    const real_t pstar     = ice->pstar;
    const real_t c_press   = ice->c_pressure;
    const real_t delta_min = ice->delta_min;
    const real_t beta_evp  = ice->beta_evp;
    const real_t cd        = ice->cd_oce_ice;
    const real_t rhoice    = ice->thermo.rhoice;
    const real_t rhosno    = ice->thermo.rhosno;
    const real_t rho0      = (real_t)FESOM_DENSITY_0;
    const real_t gravity   = (real_t)FESOM_G;

    /* ice state */
    auto u_ice  = ice->uice_fld.d();
    auto v_ice  = ice->vice_fld.d();
    auto u_aux  = ice->uice_aux_fld.d();
    auto v_aux  = ice->vice_aux_fld.d();
    auto a_ice  = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto m_ice  = ice->data[FESOM_ICE_MICE].values_fld.d();
    auto m_snow = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    auto eps11  = ice->work.eps11_fld.d();
    auto eps12  = ice->work.eps12_fld.d();
    auto eps22  = ice->work.eps22_fld.d();
    auto s11    = ice->work.sigma11_fld.d();
    auto s12    = ice->work.sigma12_fld.d();
    auto s22    = ice->work.sigma22_fld.d();
    auto u_rhs  = ice->uice_rhs_fld.d();
    auto v_rhs  = ice->vice_rhs_fld.d();
    /* ssh-gradient scratch — the Fortran reuses EXACTLY these two rhs arrays (:481-510) */
    auto rhs_a  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    auto rhs_m  = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    auto u_w    = ice->srfoce_u_fld.d();
    auto v_w    = ice->srfoce_v_fld.d();
    auto elev   = ice->srfoce_ssh_fld.d();
    auto sax    = ice->stress_atmice_x_fld.d();
    auto say    = ice->stress_atmice_y_fld.d();
    /* mEVP-only scratch */
    auto inv_th  = ice->work.mevp_inv_thickness_fld.d();
    auto mass    = ice->work.mevp_mass_fld.d();
    auto ice_nod = ice->work.mevp_ice_nod_fld.d();
    auto pfac    = ice->work.mevp_pressure_fac_fld.d();
    auto ice_el  = ice->work.mevp_ice_el_fld.d();
    /* M9: the divergence form's carried node state (and the selfcheck's reference pair). */
    auto R_u     = ice->work.mevp_Ru_fld.d();
    auto R_v     = ice->work.mevp_Rv_fld.d();
    auto Rchk_u  = m9_selfcheck ? ice->work.mevp_Rchk_u_fld.d() : R_u;   /* alias when unused */
    auto Rchk_v  = m9_selfcheck ? ice->work.mevp_Rchk_v_fld.d() : R_v;
    /* mesh */
    auto en       = mesh->elem_nodes_fld.d();
    auto ea       = mesh->elem_area_fld.d();
    auto gsca     = mesh->gradient_sca_fld.d();
    auto mfac     = mesh->metric_factor_fld.d();
    auto ulev_e   = mesh->ulevels_fld.d();
    auto ulev_n   = mesh->ulevels_nod2D_fld.d();
    auto area     = mesh->area_fld.d();
    auto cor      = mesh->coriolis_node_fld.d();
    auto bc_index = mesh->bc_index_nod2D_fld.d();   /* pushed to device in fesom_ice.cpp (M6.2) */
    auto coastal  = maevp_coastal_mask(partit, mesh);

    /* u_ice_aux = u_ice over the FULL allocation extent (:521-522) — a whole-array Fortran
     * assignment, node_size = myDim+eDim. ⚠️ TRAP 8. */
    Kokkos::parallel_for("maevp_aux_init", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            u_aux(row) = u_ice(row);
            v_aux(row) = v_ice(row);
        });

    /* --- inlined ssh2rhs, levitating branch (:535-543, 589-621) ---
     * ⚠️ TRAP 7: rhs_a/rhs_m zeroed over myDim ONLY. */
    Kokkos::parallel_for("maevp_rhs_am_zero", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int row) { rhs_a(row) = 0.0; rhs_m(row) = 0.0; });

    /* (the use_floatice branch :547-587 is not ported — use_floatice=false; this else-branch is
     * the executed path for linfs AND zstar.)
     * ⚠️ TRAP 6: the element assembly writes ALL THREE nodes UNGUARDED (:604-617) — halo entries
     * are written and never read; the arrays are node_size, so it is in-bounds. This is the
     * ASYMMETRY vs the stress2rhs assembly below, which IS guarded. Port both as written. */
    Kokkos::parallel_for("maevp_ssh2rhs", Kokkos::RangePolicy<>(0, Eo),
        KOKKOS_LAMBDA(const int el) {
            if (ulev_e(el) > 1) return;                     /* skip cavity elements (:596) */
            const int n0 = en(3 * el + 0);
            const int n1 = en(3 * el + 1);
            const int n2 = en(3 * el + 2);
            const real_t vol = ea(el);
            const size_t g = (size_t)6 * el;                /* dx = g+0..2, dy = g+3..5 */
            real_t bb = gravity * val3 * vol;
            const real_t aa = bb * (gsca(g+0)*elev(n0) + gsca(g+1)*elev(n1) + gsca(g+2)*elev(n2));
            bb              = bb * (gsca(g+3)*elev(n0) + gsca(g+4)*elev(n1) + gsca(g+5)*elev(n2));
            Kokkos::atomic_add(&rhs_a(n0), -aa);  Kokkos::atomic_add(&rhs_m(n0), -bb);
            Kokkos::atomic_add(&rhs_a(n1), -aa);  Kokkos::atomic_add(&rhs_m(n1), -bb);
            Kokkos::atomic_add(&rhs_a(n2), -aa);  Kokkos::atomic_add(&rhs_m(n2), -bb);
        });

    /* --- node precompute (:624-647, myDim): masks + mass regularisation.
     * Zero-then-cycle order as in the Fortran; the rhs scaling happens INSIDE the ice branch
     * only (:641-642, ⚠️ TRAP 7); mass = M/((1+M²)·area) verbatim (⚠️ TRAP 5); max(·, 9.0)
     * limiter (:635). Reads rhs_a/rhs_m — the atomic scatter above is complete (Kokkos fences
     * between parallel_for calls on the same execution space). */
    Kokkos::parallel_for("maevp_node_pre", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int i) {
            inv_th(i)  = 0.0;
            mass(i)    = 0.0;
            ice_nod(i) = 0;
            if (ulev_n(i) > 1) return;                      /* skip cavity nodes (:631) */
            if (a_ice(i) >= 0.01) {
                const real_t it = (rhoice * m_ice(i) + rhosno * m_snow(i)) / a_ice(i);
                inv_th(i) = 1.0 / ((it > 9.0) ? it : 9.0);  /* limit the mass */

                const real_t ms = m_ice(i) * rhoice + m_snow(i) * rhosno;
                const real_t ar = area((size_t)i * nl + 0);
                mass(i) = ms / ((1.0 + ms * ms) * ar);

                rhs_a(i) = rhs_a(i) / ar;                   /* scale rhs_a, rhs_m too */
                rhs_m(i) = rhs_m(i) / ar;

                ice_nod(i) = 1;
            }
        });

    /* --- element precompute (:650-667): the mean-msum element mask (⚠️ TRAP 4) and
     * pressure_fac WITH det2 and ⚠️ NO 0.5 (TRAP 2 — std-EVP's ice_strength 0.5 does NOT belong
     * here; it sits in the sigma11/22 update). */
    Kokkos::parallel_for("maevp_elem_pre", Kokkos::RangePolicy<>(0, Eo),
        KOKKOS_LAMBDA(const int el) {
            pfac(el)   = 0.0;
            ice_el(el) = 0;
            if (ulev_e(el) > 1) return;                     /* skip cavity elements (:658) */
            const int n0 = en(3*el + 0), n1 = en(3*el + 1), n2 = en(3*el + 2);
            const real_t msum = (m_ice(n0) + m_ice(n1) + m_ice(n2)) * val3;
            if (msum > 0.01) {
                ice_el(el) = 1;
                const real_t asum = (a_ice(n0) + a_ice(n1) + a_ice(n2)) * val3;
                pfac(el) = det2 * pstar * msum * Kokkos::exp(-c_press * (1.0 - asum));
            }
        });

    /* zero u_rhs/v_rhs over myDim (:668-673) */
    Kokkos::parallel_for("maevp_rhs_uv_zero", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int row) { u_rhs(row) = 0.0; v_rhs(row) = 0.0; });

    /* ── M9 SELFCHECK, step entry (plan §L2) ────────────────────────────────────────────────
     * The naive check "R == div(sigma) for the whole run" CANNOT pass, and the reason is the
     * scheme's own honest non-equivalence: the masks are recomputed every ocean step (above,
     * outside this loop) and the classic assembly is restricted to ice_el, so div(sigma) drops a
     * departing element's contribution INSTANTLY while R retains it decaying at det1 per
     * subcycle. WITHIN a step the masks are fixed and the two forms are algebraically identical.
     * So: measure the step-entry gap (that IS the mask-flip non-equivalence, a result to be
     * reported), then re-baseline, then gate the INTRA-step drift at roundoff. */
    double m9_jump = 0.0, m9_scale = 0.0, m9_drift = 0.0;
    if (m9_div && m9_selfcheck) {
        Kokkos::parallel_for("maevp_div_chk_zero", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int i) { Rchk_u(i) = 0.0; Rchk_v(i) = 0.0; });
        /* div(sigma) under THIS step's masks, from the sigma carried out of the last step */
        Kokkos::parallel_for("maevp_div_chk_asm", Kokkos::RangePolicy<>(0, Eo),
            KOKKOS_LAMBDA(const int el) {
                if (ulev_e(el) > 1) return;
                if (!ice_el(el))    return;
                const int eln[3] = { en(3*el + 0), en(3*el + 1), en(3*el + 2) };
                const size_t g = (size_t)6 * el;
                const real_t meancos = val3 * mfac(el);
                const real_t S11 = s11(el), S12 = s12(el), S22 = s22(el);
                const real_t a = ea(el);
                for (int k = 0; k < 3; ++k) {
                    const int n = eln[k];
                    if (n < myDim) {
                        Kokkos::atomic_add(&Rchk_u(n), -(a * (S11*gsca(g+k) + S12*gsca(g+k+3)
                                                              + S12*meancos)));
                        Kokkos::atomic_add(&Rchk_v(n), -(a * (S12*gsca(g+k) + S22*gsca(g+k+3)
                                                              - S11*meancos)));
                    }
                }
            });
        double jmax = 0.0, smax = 0.0;
        Kokkos::parallel_reduce("maevp_div_chk_jump", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int i, double &acc) {
                const double du = Kokkos::fabs((double)R_u(i) - (double)Rchk_u(i));
                const double dv = Kokkos::fabs((double)R_v(i) - (double)Rchk_v(i));
                const double d  = du > dv ? du : dv;
                if (d > acc) acc = d;
            }, Kokkos::Max<double>(jmax));
        Kokkos::parallel_reduce("maevp_div_chk_scale", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int i, double &acc) {
                const double au = Kokkos::fabs((double)Rchk_u(i));
                const double av = Kokkos::fabs((double)Rchk_v(i));
                const double a  = au > av ? au : av;
                if (a > acc) acc = a;
            }, Kokkos::Max<double>(smax));
        m9_jump = jmax; m9_scale = smax;
        /* re-baseline: from here the two forms start from the same state, so any gap that
         * appears during the step is roundoff or a bug — nothing else. */
        Kokkos::parallel_for("maevp_div_chk_rebase", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int i) { R_u(i) = Rchk_u(i); R_v(i) = Rchk_v(i); });
        /* Rchk is an ACCUMULATOR: it must be empty before the first subcycle assembles into it,
         * or subcycle 1 would pile div(sigma^1) on top of this baseline. The in-loop zero (after
         * each comparison) maintains the invariant from there on. */
        Kokkos::parallel_for("maevp_div_chk_zero2", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int i) { Rchk_u(i) = 0.0; Rchk_v(i) = 0.0; });
    }

    /* ================= mEVP pseudo-time iteration (:680-875) ================= */
    for (int shortstep = 1; shortstep <= steps; ++shortstep) {

      if (!m9_div) {
        /* inlined stress_tensor_m + stress2rhs_m: element loop (:689-792) */
        Kokkos::parallel_for("maevp_stress", Kokkos::RangePolicy<>(0, Eo),
            KOKKOS_LAMBDA(const int el) {
                if (ulev_e(el) > 1) return;                 /* (:690) */
                if (!ice_el(el))    return;                 /* (:693) */

                const int eln[3] = { en(3*el + 0), en(3*el + 1), en(3*el + 2) };
                const size_t g = (size_t)6 * el;            /* dx = g+0..2, dy = g+3..5 */
                const real_t meancos = val3 * mfac(el);     /* METRICS (:699) */
                const real_t U0 = u_aux(eln[0]), U1 = u_aux(eln[1]), U2 = u_aux(eln[2]);
                const real_t V0 = v_aux(eln[0]), V1 = v_aux(eln[1]), V2 = v_aux(eln[2]);

                /* deformation rate tensor (:702-705) */
                const real_t E11 = gsca(g+0)*U0 + gsca(g+1)*U1 + gsca(g+2)*U2
                                 - (V0 + V1 + V2) * meancos;                    /* metrics */
                const real_t E22 = gsca(g+3)*V0 + gsca(g+4)*V1 + gsca(g+5)*V2;
                const real_t E12 = 0.5 * ( gsca(g+3)*U0 + gsca(g+4)*U1 + gsca(g+5)*U2
                                         + gsca(g+0)*V0 + gsca(g+1)*V1 + gsca(g+2)*V2
                                         + (U0 + U1 + U2) * meancos );          /* metrics */
                /* M9 confound control (FESOM_SPEED_MEVPNOEPS): these three element stores are
                 * never read in the mEVP path — only std EVP reads those arrays
                 * (fesom_ice_evp.cpp:117-129) — and they cost about what the reformulation
                 * saves, so they are measured on their OWN leg. Branching around pure STORES
                 * cannot change any computed value: E11/E22/E12 are already in registers and
                 * nothing downstream reads the arrays. */
                if (!m9_noeps) { eps11(el) = E11; eps22(el) = E22; eps12(el) = E12; }

                /* switch to eps1, eps2 (:708-709) */
                const real_t eps1 = E11 + E22;
                const real_t eps2 = E11 - E22;

                /* moduli (:712); ⚠️ TRAP 10: delta_min is ADDITIVE (:714) */
                const real_t delta = Kokkos::sqrt(eps1*eps1 + vale*(eps2*eps2 + 4.0*E12*E12));
                const real_t pressure = pfac(el) / (delta + delta_min);

                /* implicit sigma update (:721-723). ⚠️ TRAP 2: the 0.5 lives in the sigma11/22
                 * lines, NOT in sigma12 and NOT in pressure_fac. Each line reads the OLD sigma. */
                const real_t S12 = det1*s12(el) +     pressure * E12 * vale;
                const real_t S11 = det1*s11(el) + 0.5*pressure * (eps1 - delta + eps2*vale);
                const real_t S22 = det1*s22(el) + 0.5*pressure * (eps1 - delta - eps2*vale);
                s12(el) = S12; s11(el) = S11; s22(el) = S22;

                /* inlined stress2rhs_m: the stress-divergence assembly, ⚠️ TRAP 6 — GUARDED to
                 * OWNED nodes (:740-790), unlike the ssh2rhs assembly above. Uses the NEW sigma. */
                const real_t a = ea(el);
                for (int k = 0; k < 3; ++k) {
                    const int n = eln[k];
                    if (n < myDim) {
                        Kokkos::atomic_add(&u_rhs(n), -(a * (S11*gsca(g+k) + S12*gsca(g+k+3)
                                                             + S12*meancos)));   /* metrics */
                        Kokkos::atomic_add(&v_rhs(n), -(a * (S12*gsca(g+k) + S22*gsca(g+k+3)
                                                             - S11*meancos)));   /* metrics */
                    }
                }
            });
      } else {
        /* ══ M9 DIVERGENCE FORM — element kernel ═════════════════════════════════════════════
         * TWIN OF maevp_stress ABOVE — EDIT BOTH OR NEITHER. Everything through `pressure` is
         * character-for-character the classic kernel; the difference is confined to the three
         * stress lines and the absence of the sigma stores.
         *
         * Why a separate kernel rather than a branch inside the classic one: the classic path
         * must stay BYTE-IDENTICAL with the knob off, and folding `det1*s11(el)` behind a
         * ternary would let the compiler contract the FMA differently. Duplication is the
         * cheaper of the two risks here.
         *
         * pfac already carries det2 (:213), so what this assembles into u_rhs/v_rhs is exactly
         * det2*div(sigma~) — the increment the node recursion needs. No carried term, no store:
         * 3 element loads and 3 element stores per element per subcycle disappear. */
        Kokkos::parallel_for("maevp_stress_div", Kokkos::RangePolicy<>(0, Eo),
            KOKKOS_LAMBDA(const int el) {
                if (ulev_e(el) > 1) return;
                if (!ice_el(el))    return;

                const int eln[3] = { en(3*el + 0), en(3*el + 1), en(3*el + 2) };
                const size_t g = (size_t)6 * el;
                const real_t meancos = val3 * mfac(el);
                const real_t U0 = u_aux(eln[0]), U1 = u_aux(eln[1]), U2 = u_aux(eln[2]);
                const real_t V0 = v_aux(eln[0]), V1 = v_aux(eln[1]), V2 = v_aux(eln[2]);

                const real_t E11 = gsca(g+0)*U0 + gsca(g+1)*U1 + gsca(g+2)*U2
                                 - (V0 + V1 + V2) * meancos;
                const real_t E22 = gsca(g+3)*V0 + gsca(g+4)*V1 + gsca(g+5)*V2;
                const real_t E12 = 0.5 * ( gsca(g+3)*U0 + gsca(g+4)*U1 + gsca(g+5)*U2
                                         + gsca(g+0)*V0 + gsca(g+1)*V1 + gsca(g+2)*V2
                                         + (U0 + U1 + U2) * meancos );
                if (!m9_noeps) { eps11(el) = E11; eps22(el) = E22; eps12(el) = E12; }

                const real_t eps1 = E11 + E22;
                const real_t eps2 = E11 - E22;

                const real_t delta = Kokkos::sqrt(eps1*eps1 + vale*(eps2*eps2 + 4.0*E12*E12));
                const real_t pressure = pfac(el) / (delta + delta_min);

                /* the VP stress itself — trap 2 still holds: the 0.5 is on 11/22, not on 12 */
                const real_t S12 =     pressure * E12 * vale;
                const real_t S11 = 0.5*pressure * (eps1 - delta + eps2*vale);
                const real_t S22 = 0.5*pressure * (eps1 - delta - eps2*vale);

                if (m9_selfcheck) {
                    /* diagnostic only: advance the classic sigma alongside and assemble
                     * div(sigma^{p+1}) into Rchk, so R can be compared against it. */
                    const real_t C12 = det1*s12(el) + S12;
                    const real_t C11 = det1*s11(el) + S11;
                    const real_t C22 = det1*s22(el) + S22;
                    s12(el) = C12; s11(el) = C11; s22(el) = C22;
                    const real_t ac = ea(el);
                    for (int k = 0; k < 3; ++k) {
                        const int n = eln[k];
                        if (n < myDim) {
                            Kokkos::atomic_add(&Rchk_u(n), -(ac * (C11*gsca(g+k) + C12*gsca(g+k+3)
                                                                   + C12*meancos)));
                            Kokkos::atomic_add(&Rchk_v(n), -(ac * (C12*gsca(g+k) + C22*gsca(g+k+3)
                                                                   - C11*meancos)));
                        }
                    }
                }

                /* same assembly as the classic kernel, same owned-node guard (trap 6) */
                const real_t a = ea(el);
                for (int k = 0; k < 3; ++k) {
                    const int n = eln[k];
                    if (n < myDim) {
                        Kokkos::atomic_add(&u_rhs(n), -(a * (S11*gsca(g+k) + S12*gsca(g+k+3)
                                                             + S12*meancos)));
                        Kokkos::atomic_add(&v_rhs(n), -(a * (S12*gsca(g+k) + S22*gsca(g+k+3)
                                                             - S11*meancos)));
                    }
                }
            });
      }

        /* node solve (:795-819). ⚠️ TRAP 13: non-ice nodes are SKIPPED — there is NO else branch;
         * a non-ice node KEEPS its u_ice_aux value (std EVP zeroes here; mEVP must not).
         * ⚠️ TRAP 9: bc_index_nod2D multiplies det (:814).
         * ⚠️ TRAP 1: the drag CARRIES rdt; drag·u_w sits OUTSIDE the rdt·(…) group;
         *            +beta·u_aux closes the rhs (:806-811).
         * ⚠️ TRAP 3: NO theta_io rotation anywhere. */
      if (!m9_div) {
        Kokkos::parallel_for("maevp_node_solve", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int i) {
                if (ulev_n(i) > 1) return;                  /* (:797) */
                if (!ice_nod(i))   return;                  /* skip if ice is absent — NO else */

                u_rhs(i) = u_rhs(i) * mass(i) + rhs_a(i);
                v_rhs(i) = v_rhs(i) * mass(i) + rhs_m(i);

                const real_t du   = u_aux(i) - u_w(i);
                const real_t dv   = v_aux(i) - v_w(i);
                const real_t umod = Kokkos::sqrt(du*du + dv*dv);
                const real_t drag = rdt * cd * umod * rho0 * inv_th(i);

                /* rhs for water stress, air stress, u_rhs_ice (internal stress + ssh) */
                const real_t rhsu = u_ice(i) + drag*u_w(i)
                                  + rdt*(inv_th(i)*sax(i) + u_rhs(i))
                                  + beta_evp*u_aux(i);
                const real_t rhsv = v_ice(i) + drag*v_w(i)
                                  + rdt*(inv_th(i)*say(i) + v_rhs(i))
                                  + beta_evp*v_aux(i);

                /* solve (Coriolis and water stress treated implicitly) */
                const real_t obd = 1.0 + beta_evp + drag;
                const real_t rf  = rdt * cor(i);
                const real_t det = bc_index(i) / (obd*obd + rf*rf);

                u_aux(i) = det * (obd*rhsu + rf*rhsv);
                v_aux(i) = det * (obd*rhsv - rf*rhsu);
            });
      } else {
        /* ══ M9 DIVERGENCE FORM — node solve ═════════════════════════════════════════════════
         * TWIN OF maevp_node_solve ABOVE — EDIT BOTH OR NEITHER. Identical from the drag line
         * down; the two differences are the R recursion and where it sits.
         *
         * 🔴🔴 THE M9 TRAP. The R recursion runs for EVERY owned node, OUTSIDE the ice_nod
         * guard. The element mask is mean(m_ice) > 0.01 (trap 4, :210) and the node mask is
         * a_ice >= 0.01 (:185) — DIFFERENT CRITERIA — so sigma genuinely evolves on elements
         * whose vertices are not ice nodes. Put the recursion inside the guard and R freezes at
         * those nodes, then hands the solve a stale divergence the moment they ice over. The
         * classic form has no equivalent bug because sigma lives on the elements, which have
         * their own mask. Only the VELOCITY solve stays masked, exactly as in the twin.
         * (Cavity nodes are likewise updated: div(sigma) there is the sum over incident
         * non-cavity ice elements, which is precisely what u_rhs already holds.) */
        Kokkos::parallel_for("maevp_node_solve_div", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int i) {
                const real_t Ru = det1*R_u(i) + u_rhs(i);
                const real_t Rv = det1*R_v(i) + v_rhs(i);
                R_u(i) = Ru; R_v(i) = Rv;

                if (ulev_n(i) > 1) return;                  /* (:797) */
                if (!ice_nod(i))   return;                  /* skip if ice is absent — NO else */

                /* the classic kernel writes the scaled value back into u_rhs; here u_rhs stays
                 * the pure per-subcycle accumulator and R carries the state, so the scaling
                 * lands in a local. `mass` is applied AT USE — R itself is stored unscaled. */
                const real_t urhs = Ru * mass(i) + rhs_a(i);
                const real_t vrhs = Rv * mass(i) + rhs_m(i);

                const real_t du   = u_aux(i) - u_w(i);
                const real_t dv   = v_aux(i) - v_w(i);
                const real_t umod = Kokkos::sqrt(du*du + dv*dv);
                const real_t drag = rdt * cd * umod * rho0 * inv_th(i);

                const real_t rhsu = u_ice(i) + drag*u_w(i)
                                  + rdt*(inv_th(i)*sax(i) + urhs)
                                  + beta_evp*u_aux(i);
                const real_t rhsv = v_ice(i) + drag*v_w(i)
                                  + rdt*(inv_th(i)*say(i) + vrhs)
                                  + beta_evp*v_aux(i);

                const real_t obd = 1.0 + beta_evp + drag;
                const real_t rf  = rdt * cor(i);
                const real_t det = bc_index(i) / (obd*obd + rf*rf);

                u_aux(i) = det * (obd*rhsu + rf*rhsv);
                v_aux(i) = det * (obd*rhsv - rf*rhsu);
            });

        if (m9_selfcheck) {
            /* intra-step drift: with both forms started from the same state at step entry, any
             * gap here is roundoff or a bug. Reduced every subcycle, reported once per step. */
            double dmax = 0.0;
            Kokkos::parallel_reduce("maevp_div_chk_drift", Kokkos::RangePolicy<>(0, myDim),
                KOKKOS_LAMBDA(const int i, double &acc) {
                    const double du = Kokkos::fabs((double)R_u(i) - (double)Rchk_u(i));
                    const double dv = Kokkos::fabs((double)R_v(i) - (double)Rchk_v(i));
                    const double d  = du > dv ? du : dv;
                    if (d > acc) acc = d;
                }, Kokkos::Max<double>(dmax));
            if (dmax > m9_drift) m9_drift = dmax;
            /* Scale for the RELATIVE criterion, measured over the step rather than only at its
             * entry: at step 1 the carried sigma is zero, so an entry-derived scale is 0 and the
             * relative number degenerates into an absolute one exactly where the check matters
             * most (the cold start is the one place the gap must be identically zero). */
            double smax2 = 0.0;
            Kokkos::parallel_reduce("maevp_div_chk_scale2", Kokkos::RangePolicy<>(0, myDim),
                KOKKOS_LAMBDA(const int i, double &acc) {
                    const double au = Kokkos::fabs((double)Rchk_u(i));
                    const double av = Kokkos::fabs((double)Rchk_v(i));
                    const double a  = au > av ? au : av;
                    if (a > acc) acc = a;
                }, Kokkos::Max<double>(smax2));
            if (smax2 > m9_scale) m9_scale = smax2;
            Kokkos::parallel_for("maevp_div_chk_rezero", Kokkos::RangePolicy<>(0, myDim),
                KOKKOS_LAMBDA(const int i) { Rchk_u(i) = 0.0; Rchk_v(i) = 0.0; });
        }
      }

        /* edge BC (:823-857): coastal boundary edges -> zero the aux velocity at both endpoints.
         * Expressed as the equivalent per-node mask (identical node set; all writes are 0 =>
         * idempotent => race-free). See maevp_coastal_mask above. */
        Kokkos::parallel_for("maevp_edge_bc", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int n) {
                if (coastal(n)) { u_aux(n) = 0.0; v_aux(n) = 0.0; }
            });

        /* halo exchange of (u_ice_aux, v_ice_aux). The Fortran overlaps exchange_nod_begin/end
         * around the rhs zeroing (:861-872); two blocking exchanges plus the zero loop are
         * result-identical (the zero loop touches only u_rhs/v_rhs, never the exchanged fields).
         * Same-kind + adjacent => ONE fused message per neighbour (M5.23 L1, bit-id-neutral).
         *
         * ── M9 CELL ⑤ (FESOM_SPEED_MEVPLAG=K): the LAGGED single halo ──────────────────────
         * Exchange only every K-th subcycle. Ghosts are never updated locally (the node solve
         * runs over [0,myDim) and only this exchange and the coastal-BC kernel touch halo
         * entries), so ring 1 simply goes up to K-1 subcycles stale and the owned elements that
         * share it read slightly old velocities. That is the whole approximation — no rings, no
         * ghost kernels, no extra fields — and it is form-agnostic, so it composes with MEVPDIV
         * to give cells ⑤a and ⑤b.
         *
         * K must divide evp_rheol_steps (guarded at resolve): with shortstep running 1..steps,
         * `shortstep % K == 0` then always fires on the LAST subcycle, which is what keeps the
         * final u_ice = u_aux copy over myDim+eDim (trap 8) from publishing a stale halo into
         * the ocean coupling. K=1 makes the condition identically true => byte-identical.
         *
         * ⚠️ This is an approximation by construction. It never gets a byte gate above K=1; it
         *    gets the L3 trajectory comparison, and a measurable ice difference here is the
         *    RESULT (it is what brackets the exact wide halo of cell ④). */
        if (shortstep % m9_lagK == 0)
            fesom_halo_field2(ice->uice_aux_fld, ice->vice_aux_fld, FESOM_HALO_NOD2D, 1, 1, partit);

        Kokkos::parallel_for("maevp_rhs_uv_rezero", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int row) { u_rhs(row) = 0.0; v_rhs(row) = 0.0; });
    }

    /* ── M9 SELFCHECK report (one line per ocean step, rank 0) ──────────────────────────────
     * Two numbers with completely different meanings, which is exactly why they are reported
     * separately rather than as one "error":
     *   maskjump — |R - div(sigma)| at step ENTRY, relative. This IS non-equivalence (ii): the
     *              ice_el mask changed between steps, so div(sigma) dropped a departing
     *              element's contribution instantly while R decayed it over ~251 subcycles (and
     *              on re-entry the classic form resumes from its FROZEN stale sigma, trap 12,
     *              while R has decayed). A RESULT to be reported, not a failure.
     *   drift    — max over this step's subcycles of |R - div(sigma)| AFTER the re-baseline,
     *              relative. The masks are fixed within a step, so the two forms are
     *              algebraically identical here: anything above roundoff is a BUG. This is the
     *              number the L2 gate reads. Serial must stay ~1e-16..1e-13 relative. */
    if (m9_div && m9_selfcheck) {
        double loc[3] = { m9_jump, m9_drift, m9_scale }, glb[3] = { 0.0, 0.0, 0.0 };
        MPI_Reduce(loc, glb, 3, MPI_DOUBLE, MPI_MAX, 0, partit->MPI_COMM_FESOM);
        if (partit->mype == 0) {
            const double s = glb[2] > 0.0 ? glb[2] : 1.0;
            fprintf(stderr, "[m9-selfcheck] maskjump=%.3e drift=%.3e  (relative: %.3e / %.3e, "
                            "scale=%.3e)\n",
                    glb[0], glb[1], glb[0] / s, glb[1] / s, glb[2]);
            fflush(stderr);
        }
    }

    /* final copy over myDim+eDim (:876-881, ⚠️ TRAP 8) — NO extra exchange: the aux halo is
     * already current from the last substep's exchange. */
    Kokkos::parallel_for("maevp_final_copy", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            u_ice(row) = u_aux(row);
            v_ice(row) = v_aux(row);
        });

    ice->uice_fld.modify_device();      ice->vice_fld.modify_device();
    ice->uice_aux_fld.modify_device();  ice->vice_aux_fld.modify_device();
    ice->work.eps11_fld.modify_device();   ice->work.eps12_fld.modify_device();
    ice->work.eps22_fld.modify_device();
    ice->work.sigma11_fld.modify_device(); ice->work.sigma12_fld.modify_device();
    ice->work.sigma22_fld.modify_device();
    /* M9: R_u/R_v are device-authoritative carried state — the same rail class as sigma, but
     * with no host consumer, so they are marked dirty and never synced back. (When a restart
     * path is eventually added, this is where it plugs in; the divergence form's restart set is
     * these two node arrays instead of three element arrays, and R = div(sigma) is computable
     * from a restored sigma but NOT the reverse.) */
    if (m9_div) { ice->work.mevp_Ru_fld.modify_device(); ice->work.mevp_Rv_fld.modify_device(); }
    if (m9_div && m9_selfcheck) {
        ice->work.mevp_Rchk_u_fld.modify_device(); ice->work.mevp_Rchk_v_fld.modify_device();
    }
    ice->uice_rhs_fld.modify_device();  ice->vice_rhs_fld.modify_device();
    ice->data[FESOM_ICE_AICE].values_rhs_fld.modify_device();
    ice->data[FESOM_ICE_MICE].values_rhs_fld.modify_device();
    ice->work.mevp_inv_thickness_fld.modify_device();
    ice->work.mevp_mass_fld.modify_device();
    ice->work.mevp_ice_nod_fld.modify_device();
    ice->work.mevp_pressure_fac_fld.modify_device();
    ice->work.mevp_ice_el_fld.modify_device();
}
