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
#include "fesom_constants.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"
#include "fesom_mesh.h"
#include "fesom_partit.h"

#include <Kokkos_Core.hpp>

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

    /* ================= mEVP pseudo-time iteration (:680-875) ================= */
    for (int shortstep = 1; shortstep <= steps; ++shortstep) {

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
                eps11(el) = E11; eps22(el) = E22; eps12(el) = E12;

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

        /* node solve (:795-819). ⚠️ TRAP 13: non-ice nodes are SKIPPED — there is NO else branch;
         * a non-ice node KEEPS its u_ice_aux value (std EVP zeroes here; mEVP must not).
         * ⚠️ TRAP 9: bc_index_nod2D multiplies det (:814).
         * ⚠️ TRAP 1: the drag CARRIES rdt; drag·u_w sits OUTSIDE the rdt·(…) group;
         *            +beta·u_aux closes the rhs (:806-811).
         * ⚠️ TRAP 3: NO theta_io rotation anywhere. */
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
         * Same-kind + adjacent => ONE fused message per neighbour (M5.23 L1, bit-id-neutral). */
        fesom_halo_field2(ice->uice_aux_fld, ice->vice_aux_fld, FESOM_HALO_NOD2D, 1, 1, partit);

        Kokkos::parallel_for("maevp_rhs_uv_rezero", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int row) { u_rhs(row) = 0.0; v_rhs(row) = 0.0; });
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
    ice->uice_rhs_fld.modify_device();  ice->vice_rhs_fld.modify_device();
    ice->data[FESOM_ICE_AICE].values_rhs_fld.modify_device();
    ice->data[FESOM_ICE_MICE].values_rhs_fld.modify_device();
    ice->work.mevp_inv_thickness_fld.modify_device();
    ice->work.mevp_mass_fld.modify_device();
    ice->work.mevp_ice_nod_fld.modify_device();
    ice->work.mevp_pressure_fac_fld.modify_device();
    ice->work.mevp_ice_el_fld.modify_device();
}
