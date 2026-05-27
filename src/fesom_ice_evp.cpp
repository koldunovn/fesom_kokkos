#include "fesom_ice_evp.h"
#include "fesom_constants.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   // M5.8: GPU-aware-MPI on-device halo for the EVP subcycle
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_types.h"

#include <Kokkos_Core.hpp>   // M4.3b: device EVP kernels (parallel_for + atomic_add scatters)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>            // M4.3b: FESOM_KK_VERIFY backend name
#include <algorithm>         // M4.3b: verify snapshot/restore copies
#include <vector>

/* --- debug-only dumps for multi-rank EVP divergence diagnosis ---
 * Gated by FESOM_EVP_DUMP_DIR env var. When unset, every dump call is one
 * NULL check + one int increment. Fires only on the first call to
 * fesom_ice_evp_dynamics (== ocean step 1) and only inside iter 0 of the
 * subcycle. See docs/NEXT_SESSION_PROMPT.md.
 */
static int         s_evp_call_step  = 0;
static const char *s_evp_dump_dir   = NULL;
static int         s_evp_dump_loaded = 0;

static void evp_dump_load_env(void)
{
    if (s_evp_dump_loaded) return;
    s_evp_dump_dir = getenv("FESOM_EVP_DUMP_DIR");
    s_evp_dump_loaded = 1;
}

static void evp_dump(int step, const char *point, const char *cls,
                     const real_t * const *vals, int nvals,
                     int N, const int *gids, int rank)
{
    if (!s_evp_dump_dir) return;
    char path[1024];
    snprintf(path, sizeof(path),
             "%s/evp_dump_s%d_%s_%s_rank%d.txt",
             s_evp_dump_dir, step, point, cls, rank);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "evp_dump: cannot open %s\n", path); return; }
    fprintf(f, "# step=%d point=%s array=%s rank=%d N=%d\n", step, point, cls, rank, N);
    for (int n = 0; n < N; ++n) {
        fprintf(f, "%d", gids[n]);
        for (int k = 0; k < nvals; ++k) fprintf(f, " %.17g", vals[k][n]);
        fputc('\n', f);
    }
    fclose(f);
}

/*
 * Mirror of Fortran stress_tensor at ice_EVP.F90:46.
 * Element loop bound: myDim_elem2D (matches Fortran line 94).
 *
 * Three pieces:
 *   1. eps11/22/12 from velocity gradients + metric correction
 *   2. delta = √((eps11²+eps22²)(1+vale) + 4·vale·eps12² + 2(1-vale)·eps11·eps22)
 *      vale = 1/ellipse²
 *   3. EVP rheology stress update via the Hunke det1/det2 form:
 *        si1 = det1·(σ11 + σ22 + dte·r1)
 *        si2 = det2·(σ11 - σ22 + dte·r2)
 *        σ12 = det2·(σ12 + dte·r3)
 *      with r1 = ζ(eps11+eps22) - p·Tevp_inv
 *           r2 = ζ(eps11-eps22)·vale
 *           r3 = ζ·eps12·vale
 *
 * Cavity skip: ulevels[el] > 1.
 * Ice-free skip: ice_strength[el] <= 0 (set by EVPdynamics precompute).
 */
void fesom_ice_stress_tensor(fesom_ice            *ice,
                             struct fesom_partit  *partit,
                             struct fesom_mesh    *mesh)
{
    (void)partit;
    real_t *u_ice = ice->uice;
    real_t *v_ice = ice->vice;
    real_t *eps11 = ice->work.eps11;
    real_t *eps12 = ice->work.eps12;
    real_t *eps22 = ice->work.eps22;
    real_t *sigma11 = ice->work.sigma11;
    real_t *sigma12 = ice->work.sigma12;
    real_t *sigma22 = ice->work.sigma22;
    real_t *ice_strength = ice->work.ice_strength;

    real_t vale = 1.0 / (ice->ellipse * ice->ellipse);
    real_t dte  = ice->ice_dt / (real_t)ice->evp_rheol_steps;
    real_t det1 = 1.0 / (1.0 + 0.5 * ice->Tevp_inv * dte);
    real_t det2 = det1;   /* same expression in Fortran (line 87) */

    int E = mesh->myDim_elem2D;
    for (int el = 0; el < E; ++el) {
        if (mesh->ulevels[el] > 1) continue;          /* cavity skip */
        if (ice_strength[el] <= 0.0) continue;        /* ice-free skip */

        int n0 = mesh->elem_nodes[3*el + 0];
        int n1 = mesh->elem_nodes[3*el + 1];
        int n2 = mesh->elem_nodes[3*el + 2];
        real_t *gs = &mesh->gradient_sca[6*el];
        real_t mfac = mesh->metric_factor[el];
        real_t U[3] = { u_ice[n0], u_ice[n1], u_ice[n2] };
        real_t V[3] = { v_ice[n0], v_ice[n1], v_ice[n2] };

        /* Strain rates (Fortran lines 105-113) */
        eps11[el] = gs[0]*U[0] + gs[1]*U[1] + gs[2]*U[2]
                    - mfac * (V[0] + V[1] + V[2]) / 3.0;
        eps22[el] = gs[3]*V[0] + gs[4]*V[1] + gs[5]*V[2];
        eps12[el] = 0.5 * ( gs[3]*U[0] + gs[4]*U[1] + gs[5]*U[2]
                          + gs[0]*V[0] + gs[1]*V[1] + gs[2]*V[2]
                          + mfac * (U[0] + U[1] + U[2]) / 3.0);

        /* delta (Fortran line 115-116) */
        real_t delta = sqrt(
            (eps11[el]*eps11[el] + eps22[el]*eps22[el]) * (1.0 + vale)
            + 4.0 * vale * eps12[el] * eps12[el]
            + 2.0 * eps11[el] * eps22[el] * (1.0 - vale));

        /* zeta (Fortran lines 132-133, 146) */
        real_t delta_min = (delta > ice->delta_min) ? delta : ice->delta_min;
        real_t zeta = ice_strength[el] / delta_min;
        zeta *= ice->Tevp_inv;

        /* r1, r2, r3 (Fortran 148-150) */
        real_t r1 = zeta * (eps11[el] + eps22[el]) - ice_strength[el] * ice->Tevp_inv;
        real_t r2 = zeta * (eps11[el] - eps22[el]) * vale;
        real_t r3 = zeta * eps12[el] * vale;

        /* sigma update (Fortran 152-157) */
        real_t si1 = det1 * (sigma11[el] + sigma22[el] + dte * r1);
        real_t si2 = det2 * (sigma11[el] - sigma22[el] + dte * r2);
        sigma12[el] = det2 * (sigma12[el] + dte * r3);
        sigma11[el] = 0.5 * (si1 + si2);
        sigma22[el] = 0.5 * (si1 - si2);
    }
}

/*
 * Mirror of Fortran stress2rhs at ice_EVP.F90:177.
 *
 * Three loops:
 *   1. Zero u_rhs/v_rhs over myDim (Fortran 218-221)
 *   2. Element scatter into u_rhs/v_rhs (myDim_elem2D, lines 239-286)
 *   3. Per-node finalisation: divide by inv_areamass and add rhs_a/rhs_m
 *      (myDim, lines 304-317)
 */
void fesom_ice_stress2rhs(fesom_ice            *ice,
                          struct fesom_partit  *partit,
                          struct fesom_mesh    *mesh)
{
    (void)partit;
    real_t *u_rhs = ice->uice_rhs;
    real_t *v_rhs = ice->vice_rhs;
    real_t *sigma11 = ice->work.sigma11;
    real_t *sigma12 = ice->work.sigma12;
    real_t *sigma22 = ice->work.sigma22;
    real_t *rhs_a = ice->data[FESOM_ICE_AICE].values_rhs;
    real_t *rhs_m = ice->data[FESOM_ICE_MICE].values_rhs;
    real_t *inv_areamass = ice->work.inv_areamass;
    real_t *ice_strength = ice->work.ice_strength;
    const real_t val3 = 1.0 / 3.0;

    /* Loop 1: zero rhs over myDim only (matches Fortran line 218) */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        u_rhs[n] = 0.0;
        v_rhs[n] = 0.0;
    }

    /* Loop 2: element scatter (matches Fortran line 239) */
    for (int el = 0; el < mesh->myDim_elem2D; ++el) {
        if (mesh->ulevels[el] > 1) continue;
        if (ice_strength[el] <= 0.0) continue;
        real_t a = mesh->elem_area[el];
        real_t mfac = mesh->metric_factor[el];
        real_t s11 = sigma11[el], s12 = sigma12[el], s22 = sigma22[el];
        real_t *gs = &mesh->gradient_sca[6*el];
        for (int k = 0; k < 3; ++k) {
            int n = mesh->elem_nodes[3*el + k];
            u_rhs[n] -= a * (s11*gs[k]   + s12*gs[k+3]
                             + s12 * val3 * mfac);
            v_rhs[n] -= a * (s12*gs[k]   + s22*gs[k+3]
                             - s11 * val3 * mfac);
        }
    }

    /* Loop 3: per-node finalisation (matches Fortran line 304) */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;
        if (inv_areamass[n] > 0.0) {
            u_rhs[n] = u_rhs[n] * inv_areamass[n] + rhs_a[n];
            v_rhs[n] = v_rhs[n] * inv_areamass[n] + rhs_m[n];
        } else {
            u_rhs[n] = 0.0;
            v_rhs[n] = 0.0;
        }
    }
}

/*
 * Mirror of Fortran EVPdynamics at ice_EVP.F90:330.
 *
 * Sequence:
 *   1. Zero inv_areamass/inv_mass/rhs_a/rhs_m over myDim+eDim (lines 434-439)
 *   2. Compute inv_areamass and inv_mass over myDim (lines 451-473)
 *   3. linfs branch: assemble rhs_a/rhs_m from elevation gradient + ice_strength
 *      (lines 576-618; we are linfs, so the use_floatice branch at lines 482-554
 *      is not taken)
 *   4. Divide rhs_a/rhs_m by area at surface (lines 636-640)
 *   5. EVP subcycle: stress_tensor → stress2rhs → velocity update → BC → halo
 *
 * For Phase D, we mirror only:
 *   - linfs branch (which_ALE='linfs')
 *   - non-cavity boundary conditions only (use_cavity=false)
 *   - non-ICEPACK (no rdg_conv_elem etc.)
 */
void fesom_ice_evp_dynamics(fesom_ice            *ice,
                            struct fesom_partit  *partit,
                            struct fesom_mesh    *mesh)
{
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int nl = mesh->nl;
    real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;
    real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    real_t *m_snow = ice->data[FESOM_ICE_MSNOW].values;
    real_t *u_ice  = ice->uice;
    real_t *v_ice  = ice->vice;
    real_t *u_old  = ice->uice_old;
    real_t *v_old  = ice->vice_old;
    real_t *u_rhs  = ice->uice_rhs;
    real_t *v_rhs  = ice->vice_rhs;
    real_t *u_w    = ice->srfoce_u;
    real_t *v_w    = ice->srfoce_v;
    real_t *elevation       = ice->srfoce_ssh;
    real_t *stress_atmice_x = ice->stress_atmice_x;
    real_t *stress_atmice_y = ice->stress_atmice_y;
    real_t *rhs_a = ice->data[FESOM_ICE_AICE].values_rhs;
    real_t *rhs_m = ice->data[FESOM_ICE_MICE].values_rhs;
    real_t *inv_areamass = ice->work.inv_areamass;
    real_t *inv_mass     = ice->work.inv_mass;
    real_t *ice_strength = ice->work.ice_strength;
    real_t  rhoice = ice->thermo.rhoice;
    real_t  rhosno = ice->thermo.rhosno;

    real_t rdt = ice->ice_dt / (real_t)ice->evp_rheol_steps;
    real_t ax = cos(ice->theta_io);
    real_t ay = sin(ice->theta_io);

    evp_dump_load_env();
    ++s_evp_call_step;
    /* Dump on step 1 AND step 2 — step 2 is where 8/16-rank crash. Tag the
     * dump filenames with the step so 1-rank vs N-rank diff stays clean. */
    int dump_now = (s_evp_dump_dir != NULL)
                && (s_evp_call_step <= 16);

    if (dump_now) {
        /* Q: raw inputs to EVPdynamics at this step. */
        const real_t *qn[5] = { a_ice, m_ice, m_snow, elevation,
                                ice->srfoce_temp };
        evp_dump(s_evp_call_step, "Q", "node", qn, 5,
                 mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);

        const real_t *uv0[2] = { u_ice, v_ice };
        evp_dump(s_evp_call_step, "U0", "node", uv0, 2,
                 mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);
    }

    /* Step 1: zero inv_areamass/inv_mass/rhs_a/rhs_m on myDim+eDim
     * (Fortran line 434, halo included intentionally) */
    for (int n = 0; n < N; ++n) {
        inv_areamass[n] = 0.0;
        inv_mass    [n] = 0.0;
        rhs_a       [n] = 0.0;
        rhs_m       [n] = 0.0;
    }

    /* Step 2: per-node mass and inverse-area-mass (Fortran line 451 — myDim only) */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;
        real_t mass_per_area = rhoice * m_ice[n] + rhosno * m_snow[n];
        if (mass_per_area > 1.0e-3) {
            inv_areamass[n] = 1.0 / (mesh->area[n*nl + 0] * mass_per_area);
        } else {
            inv_areamass[n] = 0.0;
        }
        if (a_ice[n] < 0.01) {
            inv_mass[n] = 0.0;
        } else {
            real_t m = mass_per_area / a_ice[n];
            if (m < 9.0) m = 9.0;     /* limit small mass per Fortran line 468 */
            inv_mass[n] = 1.0 / m;
        }
        /* rhs_a/rhs_m already zero from Step 1; per Fortran lines 471-472 */
    }

    /* Step 3 (linfs branch): per-element ice_strength + elevation-gradient rhs.
     * Fortran lines 576-618 (use_floatice/full-FS branch at 482-554 skipped). */
    for (int el = 0; el < mesh->myDim_elem2D; ++el) {
        ice_strength[el] = 0.0;
        if (mesh->ulevels[el] > 1) continue;
        int n0 = mesh->elem_nodes[3*el + 0];
        int n1 = mesh->elem_nodes[3*el + 1];
        int n2 = mesh->elem_nodes[3*el + 2];
        if (m_ice[n0] <= 0.0 || m_ice[n1] <= 0.0 || m_ice[n2] <= 0.0
         || a_ice[n0] <= 0.0 || a_ice[n1] <= 0.0 || a_ice[n2] <= 0.0) {
            continue;       /* ice_strength stays 0 */
        }
        real_t msum = (m_ice[n0] + m_ice[n1] + m_ice[n2]) / 3.0;
        real_t asum = (a_ice[n0] + a_ice[n1] + a_ice[n2]) / 3.0;
        /* Hunke & Dukowicz strength — Fortran ice_EVP.F90:597-599. The 0.5 is
         * applied UNCONDITIONALLY by the Fortran (line 599) to every ice-covered
         * element; it is NOT spurious. Dropping it (a misguided attempt to add
         * grid-scale EVP damping for a dt=1800 2dx mode that is actually OCEAN-
         * driven — NO_ICE_DYN blows up regardless) doubled ice_strength → 2x zeta
         * → over-stiff pack: the central Arctic ice locks up, no drift, no
         * convergence ridging against Greenland/CAA (ice/ocean speed ratio 0.37
         * vs Fortran 1.11). Restored to keep the port one-to-one. */
        ice_strength[el] = 0.5 * ice->pstar * msum * exp(-ice->c_pressure * (1.0 - asum));

        /* Diagnostic: catch the "exploding ice_strength" pattern. */
        if (ice_strength[el] > 1.0e7) {
            int gn0 = partit->myList_nod2D[n0];
            int gn1 = partit->myList_nod2D[n1];
            int gn2 = partit->myList_nod2D[n2];
            int gel = partit->myList_elem2D[el];
            int my  = mesh->myDim_nod2D;
            fprintf(stderr,
              "[rank %d step %d] HUGE ice_strength=%.3e at el local=%d gid=%d "
              "vertices: (n=%d gid=%d %s a=%.4f m=%.4f) "
              "(n=%d gid=%d %s a=%.4f m=%.4f) "
              "(n=%d gid=%d %s a=%.4f m=%.4f)\n",
              partit->mype, s_evp_call_step, ice_strength[el], el, gel,
              n0, gn0, n0<my?"OWNED":"HALO ", a_ice[n0], m_ice[n0],
              n1, gn1, n1<my?"OWNED":"HALO ", a_ice[n1], m_ice[n1],
              n2, gn2, n2<my?"OWNED":"HALO ", a_ice[n2], m_ice[n2]);
        }

        real_t aa = 9.81 * mesh->elem_area[el] / 3.0;
        real_t *gs = &mesh->gradient_sca[6*el];
        real_t e0 = elevation[n0], e1 = elevation[n1], e2 = elevation[n2];
        real_t edx = gs[0]*e0 + gs[1]*e1 + gs[2]*e2;
        real_t edy = gs[3]*e0 + gs[4]*e1 + gs[5]*e2;
        for (int k = 0; k < 3; ++k) {
            int n = mesh->elem_nodes[3*el + k];
            rhs_a[n] -= aa * edx;
            rhs_m[n] -= aa * edy;
        }
    }

    /* Step 4: divide rhs_a/rhs_m by surface area (Fortran 636-640, myDim only) */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;
        rhs_a[n] /= mesh->area[n*nl + 0];
        rhs_m[n] /= mesh->area[n*nl + 0];
    }

    if (dump_now) {
        const real_t *nv[4] = { inv_mass, inv_areamass, rhs_a, rhs_m };
        evp_dump(s_evp_call_step, "P", "node", nv, 4,
                 mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);
        const real_t *ev[1] = { ice_strength };
        evp_dump(s_evp_call_step, "P", "elem", ev, 1,
                 mesh->myDim_elem2D, partit->myList_elem2D, partit->mype);
        /* F: EVP forcing inputs (wind-on-ice stress + ocean-surface velocity).
         * At step 1 the velocity update reads only these (plus Q-derived
         * inv_mass/u_rhs and mesh coriolis), since u_ice=u_w=0 at iter-0.
         * stress_atmice is the prime suspect for the polar step-1 u_ice diff
         * (rotated wind, feedback_wind_rotation_g2r); u_w confirms srfoce=0. */
        const real_t *fv[4] = { stress_atmice_x, stress_atmice_y, u_w, v_w };
        evp_dump(s_evp_call_step, "F", "node", fv, 4,
                 mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);
    }

    /* Step 5: EVP subcycle (Fortran lines 652-832).
     * 120 iterations of stress_tensor + stress2rhs + velocity update +
     * boundary conditions + halo exchange of (uice, vice). */
    for (int sub = 0; sub < ice->evp_rheol_steps; ++sub) {
        fesom_ice_stress_tensor(ice, partit, mesh);
        if (dump_now && sub == 0) {
            const real_t *sv[3] = {
                ice->work.sigma11, ice->work.sigma12, ice->work.sigma22
            };
            evp_dump(s_evp_call_step, "1", "elem", sv, 3,
                     mesh->myDim_elem2D, partit->myList_elem2D, partit->mype);
        }
        fesom_ice_stress2rhs   (ice, partit, mesh);
        if (dump_now && sub == 0) {
            const real_t *rv[2] = { u_rhs, v_rhs };
            evp_dump(s_evp_call_step, "2", "node", rv, 2,
                     mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);
        }

        /* Save old velocity (myDim+eDim — Fortran line 671) */
        for (int n = 0; n < N; ++n) {
            u_old[n] = u_ice[n];
            v_old[n] = v_ice[n];
        }

        /* Velocity update (myDim only — Fortran line 685) */
        for (int n = 0; n < mesh->myDim_nod2D; ++n) {
            if (mesh->ulevels_nod2D[n] > 1) continue;
            if (a_ice[n] >= 0.01) {
                real_t du = u_ice[n] - u_w[n];
                real_t dv = v_ice[n] - v_w[n];
                real_t umod = sqrt(du*du + dv*dv);
                real_t drag = ice->cd_oce_ice * umod * FESOM_DENSITY_0 * inv_mass[n];
                real_t rhsu = u_ice[n] + rdt * (drag * (ax*u_w[n] - ay*v_w[n])
                                              + inv_mass[n] * stress_atmice_x[n]
                                              + u_rhs[n]);
                real_t rhsv = v_ice[n] + rdt * (drag * (ax*v_w[n] + ay*u_w[n])
                                              + inv_mass[n] * stress_atmice_y[n]
                                              + v_rhs[n]);
                real_t r_a  = 1.0 + ax * drag * rdt;
                real_t r_b  = rdt * (mesh->coriolis_node[n] + ay * drag);
                real_t det  = 1.0 / (r_a * r_a + r_b * r_b);
                u_ice[n] = det * (r_a * rhsu + r_b * rhsv);
                v_ice[n] = det * (r_a * rhsv - r_b * rhsu);
            } else {
                u_ice[n] = 0.0;
                v_ice[n] = 0.0;
            }
        }

        if (dump_now && sub == 0) {
            const real_t *uv[2] = { u_ice, v_ice };
            evp_dump(s_evp_call_step, "3", "node", uv, 2,
                     mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);
        }

        /* Coastal boundary condition: zero velocity at open-boundary edge endpoints.
         * Mirror of Fortran ice_EVP.F90:727 — `if (myList_edge2D(ed) > edge2D_in)`.
         * Using global edge ID + edge2D_in (not local edge_tri[2nd] < 0) avoids
         * misclassifying interior edges whose halo element is missing as boundary,
         * which corrupts ice velocity at multi-rank inter-rank boundaries. */
        for (int ed = 0; ed < mesh->myDim_edge2D; ++ed) {
            int gid_1based = partit->myList_edge2D[ed];
            if (gid_1based <= mesh->edge2D_in) continue;       /* interior edge */
            int e0 = mesh->edges[ed*2 + 0];
            int e1 = mesh->edges[ed*2 + 1];
            u_ice[e0] = 0.0; v_ice[e0] = 0.0;
            u_ice[e1] = 0.0; v_ice[e1] = 0.0;
        }

        if (dump_now && sub == 0) {
            const real_t *uv[2] = { u_ice, v_ice };
            evp_dump(s_evp_call_step, "4", "node", uv, 2,
                     mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);
        }

        /* Halo exchange of u_ice, v_ice (Fortran line 827) */
        fesom_exchange_nod2D(u_ice, partit);
        fesom_exchange_nod2D(v_ice, partit);
    }

    if (dump_now) {
        const real_t *uv_end[2] = { u_ice, v_ice };
        evp_dump(s_evp_call_step, "END", "node", uv_end, 2,
                 mesh->myDim_nod2D, partit->myList_nod2D, partit->mype);
    }

    (void)u_old; (void)v_old;  /* saved per-iter for downstream diagnostics; not used here */
}

/* ======================================================================== *
 *  M4.3b — DEVICE (Kokkos) twin of the EVP dynamics. The 120-subcycle        *
 *  rheology island is the CG/M4.2 pattern: HOST loop control + DEVICE        *
 *  per-subcycle kernels (stress_tensor / stress2rhs / save-old / velocity-   *
 *  update) + the per-subcycle uice/vice halo bracket (host-staged, D21).     *
 *  Setup Steps 1-4 are 4 device kernels. Step 3 + stress2rhs Loop 2 are      *
 *  element→node SCATTERS (atomic_add, D22 → Serial bit-identical, OpenMP/    *
 *  CUDA climate-close); the rest are race-free maps/per-element. The coastal *
 *  BC stays the verbatim C edge loop on the HOST, folded into the halo       *
 *  bracket's host phase (it needs partit->myList_edge2D which is not device- *
 *  backed; at np>1 the halo round-trips uice/vice anyway → free. A device    *
 *  boundary-node mask is an M5 perf note for np=1 single-GPU).               *
 * ======================================================================== */

void fesom_ice_stress_tensor_kk(fesom_ice *ice, struct fesom_mesh *mesh)
{
    const int    E       = mesh->myDim_elem2D;
    const real_t vale    = 1.0 / (ice->ellipse * ice->ellipse);
    const real_t dte     = ice->ice_dt / (real_t)ice->evp_rheol_steps;
    const real_t det1    = 1.0 / (1.0 + 0.5 * ice->Tevp_inv * dte);
    const real_t det2    = det1;
    const real_t dmin_p  = ice->delta_min;
    const real_t Tevp    = ice->Tevp_inv;
    auto u_ice = ice->uice_fld.d(); auto v_ice = ice->vice_fld.d();
    auto eps11 = ice->work.eps11_fld.d(); auto eps12 = ice->work.eps12_fld.d(); auto eps22 = ice->work.eps22_fld.d();
    auto s11   = ice->work.sigma11_fld.d(); auto s12 = ice->work.sigma12_fld.d(); auto s22 = ice->work.sigma22_fld.d();
    auto istr  = ice->work.ice_strength_fld.d();
    auto en    = mesh->elem_nodes_fld.d(); auto gs = mesh->gradient_sca_fld.d();
    auto mf    = mesh->metric_factor_fld.d(); auto ulev = mesh->ulevels_fld.d();

    Kokkos::parallel_for("ice_stress_tensor", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int el) {
            if (ulev(el) > 1)       return;     /* cavity */
            if (istr(el) <= 0.0)    return;     /* ice-free */
            const int n0 = en(3*el+0), n1 = en(3*el+1), n2 = en(3*el+2);
            const int g = 6*el;
            const real_t mfac = mf(el);
            const real_t U0=u_ice(n0),U1=u_ice(n1),U2=u_ice(n2);
            const real_t V0=v_ice(n0),V1=v_ice(n1),V2=v_ice(n2);
            const real_t e11 = gs(g+0)*U0 + gs(g+1)*U1 + gs(g+2)*U2 - mfac*(V0+V1+V2)/3.0;
            const real_t e22 = gs(g+3)*V0 + gs(g+4)*V1 + gs(g+5)*V2;
            const real_t e12 = 0.5*( gs(g+3)*U0+gs(g+4)*U1+gs(g+5)*U2
                                   + gs(g+0)*V0+gs(g+1)*V1+gs(g+2)*V2
                                   + mfac*(U0+U1+U2)/3.0);
            eps11(el)=e11; eps22(el)=e22; eps12(el)=e12;
            const real_t delta = Kokkos::sqrt((e11*e11+e22*e22)*(1.0+vale)
                                             + 4.0*vale*e12*e12 + 2.0*e11*e22*(1.0-vale));
            const real_t dmin = (delta > dmin_p) ? delta : dmin_p;
            real_t zeta = istr(el)/dmin; zeta *= Tevp;
            const real_t r1 = zeta*(e11+e22) - istr(el)*Tevp;
            const real_t r2 = zeta*(e11-e22)*vale;
            const real_t r3 = zeta*e12*vale;
            const real_t si1 = det1*(s11(el)+s22(el)+dte*r1);
            const real_t si2 = det2*(s11(el)-s22(el)+dte*r2);
            s12(el) = det2*(s12(el)+dte*r3);
            s11(el) = 0.5*(si1+si2);
            s22(el) = 0.5*(si1-si2);
        });
    ice->work.eps11_fld.modify_device(); ice->work.eps12_fld.modify_device(); ice->work.eps22_fld.modify_device();
    ice->work.sigma11_fld.modify_device(); ice->work.sigma12_fld.modify_device(); ice->work.sigma22_fld.modify_device();
}

void fesom_ice_stress2rhs_kk(fesom_ice *ice, struct fesom_mesh *mesh)
{
    const int    myDim = mesh->myDim_nod2D;
    const int    E     = mesh->myDim_elem2D;
    const real_t val3  = 1.0/3.0;
    auto u_rhs = ice->uice_rhs_fld.d(); auto v_rhs = ice->vice_rhs_fld.d();
    auto s11 = ice->work.sigma11_fld.d(); auto s12 = ice->work.sigma12_fld.d(); auto s22 = ice->work.sigma22_fld.d();
    auto rhs_a = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    auto rhs_m = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    auto inv_am = ice->work.inv_areamass_fld.d(); auto istr = ice->work.ice_strength_fld.d();
    auto en = mesh->elem_nodes_fld.d(); auto ea = mesh->elem_area_fld.d();
    auto mf = mesh->metric_factor_fld.d(); auto gs = mesh->gradient_sca_fld.d();
    auto ulev = mesh->ulevels_fld.d(); auto ulev_n = mesh->ulevels_nod2D_fld.d();

    /* Loop 1: zero rhs over OWNED only (the C bound; halo rhs is scattered-but-unused). */
    Kokkos::parallel_for("ice_s2rhs_zero", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) { u_rhs(n)=0.0; v_rhs(n)=0.0; });
    /* Loop 2: element→node SCATTER (atomic_add, D22). */
    Kokkos::parallel_for("ice_s2rhs_scatter", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int el) {
            if (ulev(el) > 1)    return;
            if (istr(el) <= 0.0) return;
            const real_t a = ea(el), mfac = mf(el);
            const real_t S11=s11(el), S12=s12(el), S22=s22(el);
            const int g = 6*el;
            for (int k = 0; k < 3; ++k) {
                const int n = en(3*el+k);
                Kokkos::atomic_add(&u_rhs(n), -(a*(S11*gs(g+k) + S12*gs(g+k+3) + S12*val3*mfac)));
                Kokkos::atomic_add(&v_rhs(n), -(a*(S12*gs(g+k) + S22*gs(g+k+3) - S11*val3*mfac)));
            }
        });
    /* Loop 3: per-node finalisation over OWNED. */
    Kokkos::parallel_for("ice_s2rhs_final", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) return;
            if (inv_am(n) > 0.0) {
                u_rhs(n) = u_rhs(n)*inv_am(n) + rhs_a(n);
                v_rhs(n) = v_rhs(n)*inv_am(n) + rhs_m(n);
            } else { u_rhs(n)=0.0; v_rhs(n)=0.0; }
        });
    ice->uice_rhs_fld.modify_device(); ice->vice_rhs_fld.modify_device();
}

/* M5.8: coastal-node mask for the ON-DEVICE EVP boundary condition. The host/Fortran BC zeros
 * uice/vice at the endpoints of OWNED open-boundary edges (gid = myList_edge2D > edge2D_in — the
 * rank-boundary-safe criterion, not edge_tri<0, see line ~430). myList_edge2D is host-only, so we
 * precompute ONCE a per-node 0/1 device mask (built from the host myList_edge2D + edges + the
 * edge2D_in threshold); the device BC kernel then zeros every marked node. All writes are 0
 * (idempotent) → race-free, NO scatter → Serial AND OpenMP bit-identical. Cached for the run
 * (the partition is fixed). */
static Kokkos::View<int*> g_evp_coastal_mask;   /* file scope → released by fesom_ice_evp_free() */

static Kokkos::View<int*> evp_coastal_mask(struct fesom_partit *partit, struct fesom_mesh *mesh)
{
    if (g_evp_coastal_mask.extent(0) != 0) return g_evp_coastal_mask;
    const int Nn = mesh->myDim_nod2D + mesh->eDim_nod2D;
    Kokkos::View<int*, Kokkos::HostSpace> h("evp_coastal_h", Nn);
    for (int n = 0; n < Nn; ++n) h(n) = 0;
    for (int ed = 0; ed < mesh->myDim_edge2D; ++ed) {
        if (partit->myList_edge2D[ed] <= mesh->edge2D_in) continue;   /* interior edge */
        h(mesh->edges[ed*2 + 0]) = 1;
        h(mesh->edges[ed*2 + 1]) = 1;
    }
    g_evp_coastal_mask = Kokkos::View<int*>("evp_coastal_mask", Nn);
    Kokkos::deep_copy(g_evp_coastal_mask, h);
    return g_evp_coastal_mask;
}

/* Release the cached mask View before Kokkos::finalize() (a static View destructed during teardown,
 * after finalize, aborts — same rule as fesom_halo_device_free). No-op if never built. */
void fesom_ice_evp_free(void) { g_evp_coastal_mask = Kokkos::View<int*>(); }

void fesom_ice_evp_dynamics_kk(fesom_ice            *ice,
                               struct fesom_partit  *partit,
                               struct fesom_mesh    *mesh)
{
    const int    N      = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    myDim  = mesh->myDim_nod2D;
    const int    E      = mesh->myDim_elem2D;
    const int    nl     = mesh->nl;
    const real_t rdt    = ice->ice_dt / (real_t)ice->evp_rheol_steps;
    const real_t ax     = cos(ice->theta_io);
    const real_t ay     = sin(ice->theta_io);
    const real_t rhoice = ice->thermo.rhoice;
    const real_t rhosno = ice->thermo.rhosno;
    const real_t pstar  = ice->pstar;
    const real_t c_pres = ice->c_pressure;
    const real_t cd     = ice->cd_oce_ice;
    const real_t rho0   = (real_t)FESOM_DENSITY_0;

    auto a_ice = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto m_ice = ice->data[FESOM_ICE_MICE].values_fld.d();
    auto m_snw = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    auto rhs_a = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    auto rhs_m = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    auto u_ice = ice->uice_fld.d();   auto v_ice = ice->vice_fld.d();
    auto u_w   = ice->srfoce_u_fld.d(); auto v_w = ice->srfoce_v_fld.d();
    auto elev  = ice->srfoce_ssh_fld.d();
    auto sax   = ice->stress_atmice_x_fld.d(); auto say = ice->stress_atmice_y_fld.d();
    auto inv_am = ice->work.inv_areamass_fld.d(); auto inv_m = ice->work.inv_mass_fld.d();
    auto istr   = ice->work.ice_strength_fld.d();
    auto u_rhs  = ice->uice_rhs_fld.d(); auto v_rhs = ice->vice_rhs_fld.d();
    auto en   = mesh->elem_nodes_fld.d(); auto gs = mesh->gradient_sca_fld.d();
    auto ea   = mesh->elem_area_fld.d();  auto area = mesh->area_fld.d();
    auto cor  = mesh->coriolis_node_fld.d();
    auto ulev = mesh->ulevels_fld.d();    auto ulev_n = mesh->ulevels_nod2D_fld.d();

    /* Step 1: zero inv_areamass/inv_mass/rhs_a/rhs_m over [0,N) (halo included). */
    Kokkos::parallel_for("ice_evp_step1", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) { inv_am(n)=0.0; inv_m(n)=0.0; rhs_a(n)=0.0; rhs_m(n)=0.0; });

    /* Step 2: per-node mass + inverse-area-mass over OWNED. */
    Kokkos::parallel_for("ice_evp_step2", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) return;
            const real_t mpa = rhoice*m_ice(n) + rhosno*m_snw(n);
            if (mpa > 1.0e-3) inv_am(n) = 1.0 / (area(n*nl + 0) * mpa);
            else              inv_am(n) = 0.0;
            if (a_ice(n) < 0.01) inv_m(n) = 0.0;
            else {
                real_t m = mpa / a_ice(n);
                if (m < 9.0) m = 9.0;
                inv_m(n) = 1.0 / m;
            }
        });

    /* Step 3: per-element ice_strength + the elevation-gradient rhs SCATTER (atomic_add, D22). */
    Kokkos::parallel_for("ice_evp_step3", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int el) {
            istr(el) = 0.0;
            if (ulev(el) > 1) return;
            const int n0 = en(3*el+0), n1 = en(3*el+1), n2 = en(3*el+2);
            if (m_ice(n0)<=0.0||m_ice(n1)<=0.0||m_ice(n2)<=0.0
             || a_ice(n0)<=0.0||a_ice(n1)<=0.0||a_ice(n2)<=0.0) return;   /* ice_strength stays 0 */
            const real_t msum = (m_ice(n0)+m_ice(n1)+m_ice(n2))/3.0;
            const real_t asum = (a_ice(n0)+a_ice(n1)+a_ice(n2))/3.0;
            istr(el) = 0.5 * pstar * msum * Kokkos::exp(-c_pres*(1.0 - asum));
            const real_t aa = 9.81 * ea(el) / 3.0;
            const int g = 6*el;
            const real_t e0=elev(n0), e1=elev(n1), e2=elev(n2);
            const real_t edx = gs(g+0)*e0 + gs(g+1)*e1 + gs(g+2)*e2;
            const real_t edy = gs(g+3)*e0 + gs(g+4)*e1 + gs(g+5)*e2;
            for (int k = 0; k < 3; ++k) {
                const int n = en(3*el+k);
                Kokkos::atomic_add(&rhs_a(n), -(aa*edx));
                Kokkos::atomic_add(&rhs_m(n), -(aa*edy));
            }
        });

    /* Step 4: divide rhs_a/rhs_m by surface area over OWNED. */
    Kokkos::parallel_for("ice_evp_step4", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) return;
            rhs_a(n) /= area(n*nl + 0);
            rhs_m(n) /= area(n*nl + 0);
        });
    ice->work.inv_areamass_fld.modify_device(); ice->work.inv_mass_fld.modify_device();
    ice->work.ice_strength_fld.modify_device();
    ice->data[FESOM_ICE_AICE].values_rhs_fld.modify_device();
    ice->data[FESOM_ICE_MICE].values_rhs_fld.modify_device();

    /* M5.8: the coastal-node mask (built once) — captured by the on-device subcycle BC kernel. */
    auto coastal = evp_coastal_mask(partit, mesh);

    /* Step 5: the EVP subcycle (host loop + device kernels + per-subcycle ON-DEVICE BC + halo). */
    for (int sub = 0; sub < ice->evp_rheol_steps; ++sub) {
        fesom_ice_stress_tensor_kk(ice, mesh);
        fesom_ice_stress2rhs_kk(ice, mesh);

        /* save old velocity (Fortran line 671; u_old/v_old unused downstream but ported faithfully). */
        { auto u_old=ice->uice_old_fld.d(); auto v_old=ice->vice_old_fld.d();
          auto ui=u_ice; auto vi=v_ice;
          Kokkos::parallel_for("ice_evp_saveold", Kokkos::RangePolicy<>(0, N),
              KOKKOS_LAMBDA(const int n){ u_old(n)=ui(n); v_old(n)=vi(n); });
          ice->uice_old_fld.modify_device(); ice->vice_old_fld.modify_device(); }

        /* velocity update over OWNED (per-node implicit drag/Coriolis solve). */
        Kokkos::parallel_for("ice_evp_velupd", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int n) {
                if (ulev_n(n) > 1) return;
                if (a_ice(n) >= 0.01) {
                    const real_t du=u_ice(n)-u_w(n), dv=v_ice(n)-v_w(n);
                    const real_t umod = Kokkos::sqrt(du*du+dv*dv);
                    const real_t drag = cd*umod*rho0*inv_m(n);
                    const real_t rhsu = u_ice(n)+rdt*(drag*(ax*u_w(n)-ay*v_w(n))+inv_m(n)*sax(n)+u_rhs(n));
                    const real_t rhsv = v_ice(n)+rdt*(drag*(ax*v_w(n)+ay*u_w(n))+inv_m(n)*say(n)+v_rhs(n));
                    const real_t r_a = 1.0 + ax*drag*rdt;
                    const real_t r_b = rdt*(cor(n)+ay*drag);
                    const real_t det = 1.0/(r_a*r_a+r_b*r_b);
                    u_ice(n) = det*(r_a*rhsu+r_b*rhsv);
                    v_ice(n) = det*(r_a*rhsv-r_b*rhsu);
                } else { u_ice(n)=0.0; v_ice(n)=0.0; }
            });
        ice->uice_fld.modify_device(); ice->vice_fld.modify_device();

        /* M5.8: coastal BC + halo ON THE DEVICE (was: sync_host → host BC + host exchange →
         * re-push = 4 PCIe copies × 120 subcycles, the §M5.6 finding #2). The BC zeros uice/vice
         * at the precomputed coastal-node mask — idempotent 0-writes per node → race-free (no
         * scatter), Serial AND OpenMP bit-identical. Halo = the GPU-aware device-halo (exact host
         * bracket on Serial; no-op at npes==1). uice/vice stay DEVICE-resident across the subcycle;
         * the post-loop sync_host (fesom_ice.cpp) pulls them once for FCT / I/O / the next step. */
        Kokkos::parallel_for("ice_evp_coastal_bc", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int n) {
                if (coastal(n)) { u_ice(n) = 0.0; v_ice(n) = 0.0; }
            });
        ice->uice_fld.modify_device(); ice->vice_fld.modify_device();
        fesom_halo_field(ice->uice_fld, FESOM_HALO_NOD2D, 1, 1, partit);
        fesom_halo_field(ice->vice_fld, FESOM_HALO_NOD2D, 1, 1, partit);
    }
}

/* FESOM_KK_VERIFY=evp — the EVP read-modify-writes uice/vice + sigma11/12/22 (the rheology state
 * carried across subcycles AND ocean steps), so capture-before those 5. The driver snapshots them
 * PRE-EVP; here snapshot the KK result, restore the pre-values, run the C twin (the whole
 * fesom_ice_evp_dynamics: setup + 120 subcycles), diff uice/vice (+sigma), restore KK. The inputs
 * (a/m/ms, srfoce_*, stress_atmice_*) are intact (EVP doesn't modify them). On Serial the atomic_add
 * scatters are ordered → max|Δ|==0; OpenMP/CUDA climate-close (D22). ⚠️ Meaningful only with ACTIVE
 * ice (CORE2). */
void fesom_ice_evp_verify(fesom_ice *ice, struct fesom_partit *partit, struct fesom_mesh *mesh,
                          int step_n, const std::vector<real_t> &pre_u, const std::vector<real_t> &pre_v,
                          const std::vector<real_t> &pre_s11, const std::vector<real_t> &pre_s12,
                          const std::vector<real_t> &pre_s22)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int Eo = mesh->myDim_elem2D;
    real_t *u=ice->uice, *v=ice->vice;
    real_t *s11=ice->work.sigma11, *s12=ice->work.sigma12, *s22=ice->work.sigma22;
    std::vector<real_t> ku(u,u+N), kv(v,v+N), k11(s11,s11+Eo), k12(s12,s12+Eo), k22(s22,s22+Eo);
    std::copy(pre_u.begin(),pre_u.end(),u);  std::copy(pre_v.begin(),pre_v.end(),v);
    std::copy(pre_s11.begin(),pre_s11.end(),s11); std::copy(pre_s12.begin(),pre_s12.end(),s12);
    std::copy(pre_s22.begin(),pre_s22.end(),s22);
    fesom_ice_evp_dynamics(ice, partit, mesh);   /* C twin */
    auto mx=[](const std::vector<real_t>&kk,const real_t*c,int n){ double d=0.0;
        for(int i=0;i<n;++i){double x=std::fabs((double)kk[i]-(double)c[i]); if(x>d)d=x;} return d; };
    double du=mx(ku,u,N), dv=mx(kv,v,N), d11=mx(k11,s11,Eo), d12=mx(k12,s12,Eo), d22=mx(k22,s22,Eo);
    std::copy(ku.begin(),ku.end(),u); std::copy(kv.begin(),kv.end(),v);
    std::copy(k11.begin(),k11.end(),s11); std::copy(k12.begin(),k12.end(),s12); std::copy(k22.begin(),k22.end(),s22);
    double dmax=du; for(double x:{dv,d11,d12,d22}) if(x>dmax) dmax=x;
    const std::string be = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=evp] step %d backend=%s  max|Δ|: uice=%.3e vice=%.3e sig11=%.3e sig12=%.3e sig22=%.3e\n",
                step_n, be.c_str(), du, dv, d11, d12, d22);
    std::fflush(stdout);
    if (be == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=evp] FAIL step %d: EVP Serial must be bit-identical "
                             "(max|Δ|=%.3e)\n", step_n, dmax); std::abort();
    }
}
