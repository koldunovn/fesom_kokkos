/*
 * GM / Redi state lifecycle and per-step physics.
 *
 * Phase G1 — state lifecycle (fesom_gm_alloc / fesom_gm_free).
 * Phase G2b — fesom_compute_sigma_xy / fesom_compute_neutral_slope.
 * Phase G3 — fesom_init_redi_gm  (deferred; abort stub).
 * Phase G4 — fesom_fer_solve_gamma / fesom_fer_gamma2vel (deferred).
 */
#include "fesom_gm.h"
#include "fesom_aux.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   // M5.1: GPU-aware-MPI on-device halo (fesom_halo_field)
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_tracers.h"

#include <Kokkos_Core.hpp>   // M2.5b: device kernels (parallel_for) + Kokkos:: math + atomic
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>            // M2.5b: FESOM_KK_VERIFY snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>
#include <cstdio>

enum { NL_MAX = FESOM_MAX_LEVELS };          /* matches the cap used in fesom_eos / momentum */

#define GM_K_GM_MAX 1000.0   /* matches namelist.oce K_GM_max default
                              * and oce_setup_step.F90:934 fer_K init   */

#define GM_CHECK(p, what) do {                                       \
    if (!(p)) {                                                      \
        fprintf(stderr, "fesom_gm_alloc: out of memory (%s)\n",      \
                (what));                                             \
        abort();                                                     \
    }                                                                \
} while (0)

void fesom_gm_alloc(fesom_gm *g, const struct fesom_mesh *mesh)
{
    /* Field members (DualView) make a raw memset UB (L13); value-initialise instead (D13). */
    *g = fesom_gm{};

    int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int nl = mesh->nl;

    size_t n_nl   = (size_t)N * (size_t)nl;
    size_t n_nlx2 = n_nl * 2;
    size_t n_nlm1 = (size_t)N * (size_t)(nl - 1);
    size_t n_nlm1x3 = n_nlm1 * 3;

    int E_full = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    size_t e_nl1x2 = (size_t)E_full * (size_t)(nl - 1) * 2;

    /* M2.5b: each array is OWNED by a fesom::Field; the raw pointer is a non-owning
     * alias to field.h() (the M1.2/D12 / M2.3a KPP data-layer pattern). Field::alloc
     * zero-inits both host and device mirrors (== the calloc semantics here). */
    g->sigma_xy_fld.alloc("gm.sigma_xy", n_nlx2);              g->sigma_xy      = g->sigma_xy_fld.h();
    g->neutral_slope_fld.alloc("gm.neutral_slope", n_nlm1x3);  g->neutral_slope = g->neutral_slope_fld.h();
    g->slope_tapered_fld.alloc("gm.slope_tapered", n_nlm1x3);  g->slope_tapered = g->slope_tapered_fld.h();
    g->fer_tapfac_fld.alloc("gm.fer_tapfac", n_nl);            g->fer_tapfac    = g->fer_tapfac_fld.h();
    g->fer_gamma_fld.alloc("gm.fer_gamma", n_nlx2);            g->fer_gamma     = g->fer_gamma_fld.h();
    g->fer_K_fld.alloc("gm.fer_K", n_nl);                      g->fer_K         = g->fer_K_fld.h();
    g->Ki_fld.alloc("gm.Ki", n_nl);                            g->Ki            = g->Ki_fld.h();
    g->fer_C_fld.alloc("gm.fer_C", (size_t)N);                 g->fer_C         = g->fer_C_fld.h();
    g->fer_scal_fld.alloc("gm.fer_scal", (size_t)N);           g->fer_scal      = g->fer_scal_fld.h();
    g->tr_xy_fld.alloc("gm.tr_xy", e_nl1x2);                   g->tr_xy         = g->tr_xy_fld.h();
    g->tr_z_fld.alloc("gm.tr_z", n_nl);                        g->tr_z          = g->tr_z_fld.h();

    /* Initial value matches Fortran oce_setup_step.F90:934.
     * init_Redi_GM (G3) overwrites this every step, but pre-G3 readers
     * (none yet) should see the namelist default rather than 0. */
    for (size_t i = 0; i < n_nl; ++i) {
        g->fer_K[i] = GM_K_GM_MAX;
    }
}

void fesom_gm_free(fesom_gm *g)
{
    /* *g = fesom_gm{} releases every Field (assigns an empty DualView → drops the refcount/frees)
     * and zeros the raw aliases (D13/L13); no per-array free (the raw ptrs are non-owning). */
    *g = fesom_gm{};
}

/* Physics-function stubs land in later phases. Each stub aborts so that
 * a premature wire-up is caught loudly rather than silently producing
 * wrong physics. */

/* ============================================================ */
/* G2b — compute_sigma_xy (oce_ale_pressure_bv.F90:2851)        */
/* ============================================================ */
/*
 * Density gradient on neutral surfaces — per node, per level, both
 * horizontal components (x, y).
 *
 * Recipe (per node n, per level nz):
 *   sigma_xy(comp, nz, n) = (-sw_alpha(nz,n) * dT/d{x,y}
 *                            +  sw_beta(nz,n) * dS/d{x,y}) * density_0
 *
 * Horizontal gradients of T and S are area-weighted means of the per-
 * element ∇T and ∇S over surrounding elements (gradient_sca is the
 * per-element ∂N_i/∂{x,y}).
 *
 * Halo audit: Fortran writes only for `n=1, myDim_nod2D` then halo-
 * exchanges sigma_xy (one component at a time, lines 2937-2942). We
 * mirror.
 *
 * Layout: sigma_xy[n * nl * 2 + nz * 2 + c]  (c = 0 → x, 1 → y).
 */
void fesom_compute_sigma_xy(struct fesom_aux *aux,
                            const struct fesom_tracers *tracers,
                            const struct fesom_mesh *mesh,
                            fesom_gm *gm,
                            struct fesom_partit *partit)
{
    const int  nl       = mesh->nl;
    const int  myDim    = mesh->myDim_nod2D;
    const real_t rho_ref = (real_t)FESOM_DENSITY_0;

    const real_t *T = tracers->data[FESOM_TRACER_T].values;   /* [N * nl] stride */
    const real_t *S = tracers->data[FESOM_TRACER_S].values;
    const real_t *alpha = aux->sw_alpha;     /* [N * nl] */
    const real_t *beta  = aux->sw_beta;      /* [N * nl] */
    real_t       *sxy   = gm->sigma_xy;      /* [N * nl * 2] */

    /* Stack scratch — Fortran allocates per-thread `tx[nl-1]` etc. */
    FESOM_CHECK(nl <= NL_MAX, "sigma_xy: nl %d > NL_MAX", nl);

    for (int n = 0; n < myDim; ++n) {
        int nle_node = mesh->nlevels_nod2D[n] - 1;     /* exclusive */
        int ule_node = mesh->ulevels_nod2D[n] - 1;     /* 0-based   */
        if (nle_node <= ule_node) continue;            /* dry */

        real_t tx[NL_MAX], ty[NL_MAX], sx[NL_MAX], sy[NL_MAX], vol[NL_MAX];
        for (int nz = ule_node; nz < nle_node; ++nz) {
            tx[nz] = ty[nz] = sx[nz] = sy[nz] = vol[nz] = 0.0;
        }

        int o0 = mesh->nod_in_elem2D_offsets[n];
        int o1 = mesh->nod_in_elem2D_offsets[n + 1];
        for (int k = o0; k < o1; ++k) {
            int el = mesh->nod_in_elem2D[k];
            int nle = mesh->nlevels[el] - 1;            /* exclusive  */
            int ule = mesh->ulevels[el] - 1;            /* 0-based    */
            const real_t *g = &mesh->gradient_sca[6 * el];   /* dN/dx, dN/dy */
            int v0 = mesh->elem_nodes[3*el + 0];
            int v1 = mesh->elem_nodes[3*el + 1];
            int v2 = mesh->elem_nodes[3*el + 2];
            real_t a = mesh->elem_area[el];

            for (int nz = ule; nz < nle; ++nz) {
                /* Tracer T/S are stored with stride nl (per fesom_tracers_alloc) — NOT nl-1. */
                size_t i0 = (size_t)v0 * nl + nz;
                size_t i1 = (size_t)v1 * nl + nz;
                size_t i2 = (size_t)v2 * nl + nz;
                /* Note: Fortran loop bound is `ule_el..nle_el-1` per
                 * element; per-node accumulator only valid in
                 * [ule_node, nle_node-1] but that's a superset — for
                 * partial cells the per-element bound is correct. */
                if (nz < ule_node || nz >= nle_node) continue;

                vol[nz] += a;
                tx[nz] += (g[0]*T[i0] + g[1]*T[i1] + g[2]*T[i2]) * a;
                ty[nz] += (g[3]*T[i0] + g[4]*T[i1] + g[5]*T[i2]) * a;
                sx[nz] += (g[0]*S[i0] + g[1]*S[i1] + g[2]*S[i2]) * a;
                sy[nz] += (g[3]*S[i0] + g[4]*S[i1] + g[5]*S[i2]) * a;
            }
        }

        for (int nz = ule_node; nz < nle_node; ++nz) {
            real_t inv_vol = (vol[nz] > 0.0) ? 1.0 / vol[nz] : 0.0;
            real_t a = alpha[(size_t)n * nl + nz];
            real_t b = beta [(size_t)n * nl + nz];
            sxy[(size_t)n * nl * 2 + nz * 2 + 0] =
                (-a * tx[nz] + b * sx[nz]) * inv_vol * rho_ref;
            sxy[(size_t)n * nl * 2 + nz * 2 + 1] =
                (-a * ty[nz] + b * sy[nz]) * inv_vol * rho_ref;
        }
    }

    /* Halo exchange — Fortran lines 2937-2942 do it component-by-
     * component through an aux array. With our [n, nz, c] layout the
     * two components are interleaved, so a single 2-stride exchange
     * covers both. */
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(sxy, FESOM_HALO_NOD2D, nl, 2, partit);
    }
}

/* ============================================================ */
/* G2b — compute_neutral_slope (oce_ale_pressure_bv.F90:2949)   */
/* ============================================================ */
/*
 * Active config:
 *   scaling_ODM95 = .true.  (c1 hyperbolic tanh tapering)
 *   scaling_LDD97 = .false. (c2 ≡ 1)
 *   Fer_GM = .true., Redi = .true., Redi_Ktaper = .true.
 * → fer_tapfac = c1 * c2 = c1;
 *   slope_tapered = neutral_slope * sqrt(c1*c2) = neutral_slope * sqrt(c1).
 *
 * Layout reminders:
 *   neutral_slope[n * (nl-1) * 3 + nz * 3 + comp]   (comp 0=x, 1=y, 2=|s|)
 *   slope_tapered[same]
 *   fer_tapfac[n * nl + nz]
 *
 * Halo: Fortran exchange_nod on neutral_slope and slope_tapered (lines
 * 3065-3066). We mirror.
 */
void fesom_compute_neutral_slope(struct fesom_aux *aux,
                                 const struct fesom_mesh *mesh,
                                 fesom_gm *gm,
                                 struct fesom_partit *partit)
{
    const int    nl       = mesh->nl;
    const int    nl1      = nl - 1;
    const int    myDim    = mesh->myDim_nod2D;
    const real_t g        = (real_t)FESOM_G;
    const real_t rho_ref  = (real_t)FESOM_DENSITY_0;
    const real_t eps      = 5.0e-6;
    const real_t eps_sq   = eps * eps;

    /* Active namelist params (active branches only). */
    const real_t ODM95_Scr = 0.2e-2;
    const real_t ODM95_Sd  = 1.0e-3;

    real_t *ns  = gm->neutral_slope;     /* [N * nl1 * 3] */
    real_t *st  = gm->slope_tapered;     /* [N * nl1 * 3] */
    real_t *tf  = gm->fer_tapfac;        /* [N * nl     ] */

    for (int n = 0; n < myDim; ++n) {
        int nle = mesh->nlevels_nod2D[n] - 1;       /* exclusive index for slopes */
        int ule = mesh->ulevels_nod2D[n] - 1;       /* 0-based */

        /* Fortran zeroes slope_tapered(:, :, n) = 0 for all nz at top of
         * the loop (line 2979). Mirror: zero the full nz range so dry
         * cells / gaps stay clean. */
        for (int nz = 0; nz < nl1; ++nz) {
            st[(size_t)n * nl1 * 3 + nz * 3 + 0] = 0.0;
            st[(size_t)n * nl1 * 3 + nz * 3 + 1] = 0.0;
            st[(size_t)n * nl1 * 3 + nz * 3 + 2] = 0.0;
        }

        if (nle <= ule) continue;

        /* Pass 1: neutral_slope from sigma_xy / N². Fortran lines 2997-3005. */
        for (int nz = ule; nz < nle; ++nz) {
            real_t bv_sum = aux->bvfreq[(size_t)n * nl + nz]
                          + aux->bvfreq[(size_t)n * nl + (nz + 1)];
            real_t denom  = (bv_sum > eps_sq) ? bv_sum : eps_sq;
            real_t ro_z_inv = 2.0 * g / rho_ref / denom;

            real_t sx = gm->sigma_xy[(size_t)n * nl * 2 + nz * 2 + 0] * ro_z_inv;
            real_t sy = gm->sigma_xy[(size_t)n * nl * 2 + nz * 2 + 1] * ro_z_inv;
            real_t sm = sqrt(sx*sx + sy*sy);
            ns[(size_t)n * nl1 * 3 + nz * 3 + 0] = sx;
            ns[(size_t)n * nl1 * 3 + nz * 3 + 1] = sy;
            ns[(size_t)n * nl1 * 3 + nz * 3 + 2] = sm;
        }

        /* Pass 2: ODM95 tapering c1. Fortran lines 3010-3016. */
        real_t c1[NL_MAX];
        for (int nz = 0; nz < nl1; ++nz) c1[nz] = 1.0;
        for (int nz = ule; nz < nle; ++nz) {
            real_t sm = ns[(size_t)n * nl1 * 3 + nz * 3 + 2];
            real_t v = 0.5 * (1.0 + tanh((ODM95_Scr - sm) / ODM95_Sd));
            real_t bv0 = aux->bvfreq[(size_t)n * nl + nz];
            real_t bv1 = aux->bvfreq[(size_t)n * nl + (nz + 1)];
            if (bv0 <= 0.0 || bv1 <= 0.0) v = 0.0;
            c1[nz] = v;
        }

        /* Pass 3: combine c1*c2 with c2≡1 in our config. Active branch
         * is `Fer_GM .and. Redi .and. Redi_Ktaper` (all true) → use
         * sqrt(c1*c2) for slope_tapered and store fer_tapfac=c1*c2.
         * Fortran lines 3049-3054. */
        for (int nz = ule; nz < nle; ++nz) {
            real_t cc = c1[nz];                          /* c1 * c2 = c1 */
            tf[(size_t)n * nl + nz] = cc;
            real_t s = sqrt(cc);
            real_t sx = ns[(size_t)n * nl1 * 3 + nz * 3 + 0];
            real_t sy = ns[(size_t)n * nl1 * 3 + nz * 3 + 1];
            real_t sm = ns[(size_t)n * nl1 * 3 + nz * 3 + 2];
            st[(size_t)n * nl1 * 3 + nz * 3 + 0] = sx * s;
            st[(size_t)n * nl1 * 3 + nz * 3 + 1] = sy * s;
            st[(size_t)n * nl1 * 3 + nz * 3 + 2] = sm * s;
        }
    }

    /* Halo exchange — Fortran lines 3065-3066. 3 components per (n,nz). */
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(ns, FESOM_HALO_NOD2D, nl1, 3, partit);
        fesom_halo_exchange(st, FESOM_HALO_NOD2D, nl1, 3, partit);
        /* fer_tapfac is consumed in init_Redi_GM at myDim only — no
         * exchange needed (matches Fortran which doesn't exchange it). */
    }
}

/* ============================================================ */
/* G3 — init_Redi_GM (oce_fer_gm.F90:215-506)                   */
/* ============================================================ */
/*
 * Per-step coefficient builder for GM (fer_K, fer_C, fer_scal) and
 * Redi (Ki). Three logical phases:
 *
 *   Pass 1 (F1 = horizontal scaling) — per-node loop using the
 *     conservative level extrema nlevels_nod2D_min / ulevels_nod2D_max
 *     (which are exactly what Fortran derives inline at lines 252-257).
 *     Computes fer_scal, fer_K[nzmin,n], fer_C[n] (all GM); Ki[nzmin,n]
 *     is filled here too but immediately overwritten in the next pass
 *     since Fer_GM ∧ Redi is true in our config.
 *
 *   Pass 2 (Redi=GM sync) — Ki[nzmin,n] = max(fer_scal·Redi_Kmax, K_GM_min)
 *     (Fortran lines 351-359; the FESOM 1.4 carry-over).
 *
 *   Pass 3 (F2 = vertical scaling) — per-node loop using regular
 *     per-node nlevels_nod2D / ulevels_nod2D. Applies the
 *     scaling_GMzexp depth-exp scaling to both fer_K and Ki. Redi
 *     also gets the Redi_Ktaper sqrt(fer_tapfac) tapering.
 *
 * Active sub-features only (default namelist.oce):
 *   Fer_GM=T, Redi=T, K_GM_resscalorder=2 (real, not int → 1/2=0.5),
 *   scaling_resolution=T, scaling_GMzexp=T, Redi_Ktaper=T,
 *   K_GM_max=1000, K_GM_min=2, Redi_Kmin=100, K_GM_cmin=0.1, K_GM_cm=3,
 *   GMzexp_zref=500, GMzexp_smin=0.6, refscalresol=100km.
 *   Skipped: scaling_Rossby, FESOM14, GINsea, Ferreira (and so the
 *   K_GM_bvref=1 MLD-reference branch never runs); also K_GM_Ktaper,
 *   K_GM_rampmax/min, scaling_LDD97. Redi_Kmax=0 auto-syncs to K_GM_max.
 *
 * Halo: Fortran exchanges fer_c, fer_K, Ki (lines 503-505). Mirror.
 */
void fesom_init_redi_gm(struct fesom_aux *aux,
                        const struct fesom_mesh *mesh,
                        fesom_gm *gm,
                        struct fesom_partit *partit)
{
    (void)aux;   /* unused in active branches (would be needed for Ferreira/MLD ref) */

    const int    nl       = mesh->nl;
    const int    myDim    = mesh->myDim_nod2D;
    const real_t pi       = 3.14159265358979323846;

    /* Active-config namelist constants. */
    const real_t K_GM_max     = 1000.0;
    const real_t K_GM_min     = 2.0;
    const real_t K_GM_cmin    = 0.1;
    const real_t K_GM_cm      = 3.0;
    const real_t Redi_Kmin    = 100.0;
    const real_t Redi_Kmax    = K_GM_max;     /* auto-sync (Fortran 240-242) */
    const real_t GMzexp_zref  = 500.0;
    const real_t GMzexp_smin  = 0.6;
    const real_t refscalresol = 100000.0;     /* 100 km */
    const real_t inv_refscalresol_sq = 1.0 / (refscalresol * refscalresol);

    /* --- Pass 1: F1(x,y) per-node scalar ------------------------- */
    /* Bounds nzmin/nzmax mirror Fortran's inline min(nlevels)/max(ulevels)
     * over surrounding elements (= our G0 mesh prereqs). 0-based. */
    for (int n = 0; n < myDim; ++n) {
        int nzmax = mesh->nlevels_nod2D_min[n] - 1;   /* exclusive */
        int nzmin = mesh->ulevels_nod2D_max[n] - 1;   /* 0-based */

        /* baroclinic wave speed cm — depth integral of |N|. */
        real_t cm_sum = 0.0;
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t bv0 = aux->bvfreq[(size_t)n * nl + nz];
            real_t bv1 = aux->bvfreq[(size_t)n * nl + (nz + 1)];
            real_t v0 = (bv0 > 0.0) ? sqrt(bv0) : 0.0;
            real_t v1 = (bv1 > 0.0) ? sqrt(bv1) : 0.0;
            cm_sum += mesh->hnode_new[(size_t)n * nl + nz] * 0.5 * (v0 + v1);
        }
        /* Limited from below by K_GM_cmin, and divided by π·K_GM_cm
         * (Fortran 268-269). c1 itself is only used by scaling_Rossby
         * which is off — we still compute cm. */
        real_t cm = cm_sum / pi / K_GM_cm;
        if (cm < K_GM_cmin) cm = K_GM_cmin;

        /* scaling — start at 1, no Rossby-radius cutoff, area-based
         * resolution scaling for K_GM_resscalorder=2. */
        real_t area_n = mesh->area[(size_t)n * nl + 0];   /* surface CV area */
        real_t scaling = sqrt(area_n * inv_refscalresol_sq * 2.0);
        if (scaling > 1.0) scaling = 1.0;          /* Fortran: fer_scal=min(scaling,1) */

        gm->fer_scal[n] = scaling;

        /* fer_K[nzmin,n] = max(fer_scal*K_GM_max, K_GM_min). Fortran 328-330.
         * Stored 0-based — `nzmin` is the 0-based level index. */
        real_t k_top = scaling * K_GM_max;
        if (k_top < K_GM_min) k_top = K_GM_min;
        gm->fer_K[(size_t)n * nl + nzmin] = k_top;
        gm->fer_C[n] = cm * cm;

        /* Redi placeholder (always overwritten in pass 2 below) — the
         * Fortran K_hor**x branch from lines 340-344 is dead in our
         * config because Fer_GM ∧ Redi triggers the override below. */

        /* --- Pass 2 (inline): Redi = GM sync (Fortran 351-359) ---
         * Ki[nzmin,n] = max(fer_scal*Redi_Kmax, K_GM_min). */
        real_t ki_top = scaling * Redi_Kmax;
        if (ki_top < K_GM_min) ki_top = K_GM_min;
        gm->Ki[(size_t)n * nl + nzmin] = ki_top;
    }

    /* --- Pass 3: F2(z) per-node, per-level vertical scaling ------ */
    real_t zscaling[NL_MAX];
    for (int n = 0; n < myDim; ++n) {
        int nzmax = mesh->nlevels_nod2D[n] - 1;       /* exclusive */
        int nzmin = mesh->ulevels_nod2D[n] - 1;       /* 0-based */

        /* zscaling: GMzexp_smin + (1-GMzexp_smin)*exp(-|z|/zref);
         * clamp to [GMzexp_smin, 1]. Fortran 437-441. */
        for (int nz = 0; nz < nl; ++nz) zscaling[nz] = 1.0;
        for (int nz = nzmin; nz <= nzmax; ++nz) {
            real_t z = mesh->zbar_3d_n[(size_t)n * nl + nz];
            real_t v = GMzexp_smin
                     + (1.0 - GMzexp_smin) * exp(-fabs(z) / GMzexp_zref);
            if (v > 1.0)          v = 1.0;
            if (v < GMzexp_smin)  v = GMzexp_smin;
            zscaling[nz] = v;
        }

        /* Apply F2 to fer_K (Fer_GM). Fortran 459-464. */
        real_t k_top = gm->fer_K[(size_t)n * nl + nzmin];
        for (int nz = nzmin + 1; nz <= nzmax; ++nz) {
            gm->fer_K[(size_t)n * nl + nz] = k_top * zscaling[nz];
        }
        gm->fer_K[(size_t)n * nl + nzmin] = k_top * zscaling[nzmin];
        /* K_GM_Ktaper = false → skip the GM tapering branch. */

        /* Apply F2 to Ki (Redi). Fortran 482-487. */
        real_t ki_top = gm->Ki[(size_t)n * nl + nzmin];
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            gm->Ki[(size_t)n * nl + nz] =
                ki_top * 0.5 * (zscaling[nz] + zscaling[nz + 1]);
        }
        gm->Ki[(size_t)n * nl + nzmin] =
            ki_top * 0.5 * (zscaling[nzmin] + zscaling[nzmin + 1]);

        /* Redi_Ktaper = true (Fortran 490-497):
         *   Ki[nz,n] = Ki[nz,n] * sqrt(tapfac) + Redi_Kmin * |sqrt(tapfac)-1|
         */
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t tf = gm->fer_tapfac[(size_t)n * nl + nz];
            real_t s  = sqrt(tf);
            real_t k  = gm->Ki[(size_t)n * nl + nz];
            gm->Ki[(size_t)n * nl + nz] = k * s + Redi_Kmin * fabs(s - 1.0);
        }
    }

    /* --- Halo exchange (Fortran lines 503-505) -------------------- */
    if (partit && partit->npes > 1) {
        fesom_exchange_nod2D(gm->fer_C, partit);
        fesom_halo_exchange(gm->fer_K, FESOM_HALO_NOD2D, nl, 1, partit);
        fesom_halo_exchange(gm->Ki,    FESOM_HALO_NOD2D, nl, 1, partit);
    }
}

/* ============================================================ */
/* G4 — fer_solve_Gamma (oce_fer_gm.F90:40-163)                 */
/* ============================================================ */
/*
 * Per-node 1D TDMA solving the GM streamfunction equation.
 *
 *   ∂z(C·∂z Γ) − N²·Γ = (g/ρ₀)·∇σ·K_GM(z)
 *
 * The two horizontal components (x, y) share the same matrix and are
 * solved together (Fortran does it with a (2, nl) RHS; we use a
 * 2-component sweep per node).
 *
 * Note the two distinct level bounds:
 *   - outer (zbar_n / Z_n setup):  nlevels_nod2D / ulevels_nod2D
 *   - inner (TDMA solve):          nlevels_nod2D_min / ulevels_nod2D_max
 *                                  (= conservative bounds)
 * In our linfs / no-cavity / no-partial-cells config
 * nzmin_outer == nzmin_inner = 0; nzmax_outer ≥ nzmax_inner.
 *
 * Halo: exchange_nod(fer_gamma) — Fortran line 162. We use stride 2
 * to cover both components in a single exchange.
 */
void fesom_fer_solve_gamma(const struct fesom_aux *aux,
                           const struct fesom_mesh *mesh,
                           fesom_gm *gm,
                           struct fesom_partit *partit)
{
    const int    nl       = mesh->nl;
    const int    myDim    = mesh->myDim_nod2D;
    const real_t g        = (real_t)FESOM_G;
    const real_t rho_ref  = (real_t)FESOM_DENSITY_0;

    real_t zbar_n[NL_MAX], Z_n[NL_MAX];
    real_t a[NL_MAX], b[NL_MAX], c[NL_MAX];
    real_t cp[NL_MAX], tp_x[NL_MAX], tp_y[NL_MAX];
    real_t tr_x[NL_MAX], tr_y[NL_MAX];
    FESOM_CHECK(nl <= NL_MAX, "fer_solve_gamma: nl %d > NL_MAX", nl);

    for (int n = 0; n < myDim; ++n) {
        /* Outer bounds (Fortran lines 71-72). 0-based throughout. */
        int nzmax_o = mesh->nlevels_nod2D[n] - 1;     /* index of bottom interface */
        int nzmin_o = mesh->ulevels_nod2D[n] - 1;     /* = 0 in our config       */
        if (nzmax_o <= nzmin_o + 1) continue;          /* degenerate column */

        /* Build zbar_n, Z_n on [nzmin_o, nzmax_o] (Fortran 76-86).
         * For linfs/no-cavity: zbar_n[nz] == mesh->zbar[nz]. */
        for (int nz = 0; nz < nl; ++nz) { zbar_n[nz] = 0.0; Z_n[nz] = 0.0; }
        zbar_n[nzmax_o] = mesh->zbar[nzmax_o];
        Z_n[nzmax_o - 1] = zbar_n[nzmax_o]
                         + mesh->hnode_new[(size_t)n * nl + (nzmax_o - 1)] * 0.5;
        for (int nz = nzmax_o - 1; nz >= nzmin_o + 1; --nz) {
            zbar_n[nz]    = zbar_n[nz + 1]
                          + mesh->hnode_new[(size_t)n * nl + nz];
            Z_n[nz - 1]   = zbar_n[nz]
                          + mesh->hnode_new[(size_t)n * nl + (nz - 1)] * 0.5;
        }
        zbar_n[nzmin_o] = zbar_n[nzmin_o + 1]
                        + mesh->hnode_new[(size_t)n * nl + nzmin_o];

        /* Inner bounds for TDMA (Fortran 92-93). */
        int nzmax = mesh->nlevels_nod2D_min[n] - 1;
        int nzmin = mesh->ulevels_nod2D_max[n] - 1;
        if (nzmax <= nzmin + 1) {
            /* No interior layers — Γ stays zero everywhere. */
            for (int nz = 0; nz < nl; ++nz) {
                gm->fer_gamma[(size_t)n * nl * 2 + nz * 2 + 0] = 0.0;
                gm->fer_gamma[(size_t)n * nl * 2 + nz * 2 + 1] = 0.0;
            }
            continue;
        }

        /* Top boundary (Dirichlet): a=c=0, b=1. */
        a[nzmin] = 0.0;
        c[nzmin] = 0.0;
        b[nzmin] = 1.0;

        /* zinv2 init for body (Fortran 104). */
        real_t zinv2 = 1.0 / (zbar_n[nzmin] - zbar_n[nzmin + 1]);

        /* Body: nz = nzmin+1 .. nzmax-1 (Fortran 107-114). */
        const real_t fc = gm->fer_C[n];
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t zinv1 = zinv2;
            zinv2 = 1.0 / (zbar_n[nz] - zbar_n[nz + 1]);
            real_t zinv  = 1.0 / (Z_n[nz - 1] - Z_n[nz]);
            a[nz] = fc * zinv1 * zinv;
            c[nz] = fc * zinv2 * zinv;
            real_t bv = aux->bvfreq[(size_t)n * nl + nz];
            if (bv < 1.0e-8) bv = 1.0e-8;
            b[nz] = -a[nz] - c[nz] - bv;
        }

        /* Bottom boundary (Dirichlet). */
        a[nzmax] = 0.0;
        c[nzmax] = 0.0;
        b[nzmax] = 1.0;

        /* RHS (Fortran 124-132). r = g/ρ₀; sigma_xy avg over two
         * adjacent mid-layers (nz-1, nz) at the level interface. */
        const real_t r = g / rho_ref;
        for (int nz = 0; nz < nl; ++nz) { tr_x[nz] = 0.0; tr_y[nz] = 0.0; }
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t sx_up = gm->sigma_xy[(size_t)n * nl * 2 + (nz - 1) * 2 + 0];
            real_t sx_dn = gm->sigma_xy[(size_t)n * nl * 2 + nz       * 2 + 0];
            real_t sy_up = gm->sigma_xy[(size_t)n * nl * 2 + (nz - 1) * 2 + 1];
            real_t sy_dn = gm->sigma_xy[(size_t)n * nl * 2 + nz       * 2 + 1];
            real_t k = gm->fer_K[(size_t)n * nl + nz];
            tr_x[nz] = r * 0.5 * (sx_up + sx_dn) * k;
            tr_y[nz] = r * 0.5 * (sy_up + sy_dn) * k;
        }

        /* Thomas sweep (Fortran 139-148). */
        cp[nzmin]   = c[nzmin]   / b[nzmin];
        tp_x[nzmin] = tr_x[nzmin] / b[nzmin];
        tp_y[nzmin] = tr_y[nzmin] / b[nzmin];
        for (int nz = nzmin + 1; nz <= nzmax; ++nz) {
            real_t m = b[nz] - cp[nz - 1] * a[nz];
            cp  [nz] = c[nz] / m;
            tp_x[nz] = (tr_x[nz] - tp_x[nz - 1] * a[nz]) / m;
            tp_y[nz] = (tr_y[nz] - tp_y[nz - 1] * a[nz]) / m;
        }

        /* Back-substitution into fer_gamma (Fortran 151-157). Zero
         * outside [nzmin, nzmax]. */
        for (int nz = 0; nz < nl; ++nz) {
            gm->fer_gamma[(size_t)n * nl * 2 + nz * 2 + 0] = 0.0;
            gm->fer_gamma[(size_t)n * nl * 2 + nz * 2 + 1] = 0.0;
        }
        gm->fer_gamma[(size_t)n * nl * 2 + nzmax * 2 + 0] = tp_x[nzmax];
        gm->fer_gamma[(size_t)n * nl * 2 + nzmax * 2 + 1] = tp_y[nzmax];
        for (int nz = nzmax - 1; nz >= nzmin; --nz) {
            real_t gx_below = gm->fer_gamma[(size_t)n * nl * 2 + (nz + 1) * 2 + 0];
            real_t gy_below = gm->fer_gamma[(size_t)n * nl * 2 + (nz + 1) * 2 + 1];
            gm->fer_gamma[(size_t)n * nl * 2 + nz * 2 + 0] = tp_x[nz] - cp[nz] * gx_below;
            gm->fer_gamma[(size_t)n * nl * 2 + nz * 2 + 1] = tp_y[nz] - cp[nz] * gy_below;
        }
    }

    /* Halo exchange fer_gamma (Fortran 162). 2 components per (n, nz). */
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(gm->fer_gamma, FESOM_HALO_NOD2D, nl, 2, partit);
    }
}

/* ============================================================ */
/* G7a — diff_ver_part_redi_expl (oce_ale_tracer.F90:1086-1169)  */
/* ============================================================ */
/*
 * Vertical-explicit Redi: adds the off-diagonal Redi tensor's
 * vertical projection of horizontal tracer gradients to del_ttf
 * for the given tracer.
 *
 * Per-tracer flow:
 *   1. Compute gm->tr_xy per element from valuesAB (matches
 *      Fortran ttf which is the AB-extrapolated tracer field).
 *      Halo-exchange (Fortran exchange_elem at oce_tracer_mod.F90:143).
 *   2. Per-node, area-weighted average tr_xy → tr_xynodes
 *      (Fortran 1117-1135). NO halo exchange of tr_xynodes per
 *      Fortran comment "no halo exchange of tr_xynodes is needed".
 *   3. Build zbar_n / z_n on the OLD vertical mesh (uses hnode,
 *      NOT hnode_new — Fortran 1148-1154).
 *   4. vd_flux at interfaces, then accumulate divergence into
 *      del_ttf (Fortran 1157-1166).
 *
 * Layouts:
 *   tr_xy[el * (nl-1) * 2 + nz * 2 + comp]
 *   slope_tapered[n * (nl-1) * 3 + nz * 3 + comp]   (comp 0=x, 1=y, 2=|s|)
 *   Ki[n * nl + nz]
 *   del_ttf[n * nl + nz]   (stride nl per fesom_tracers_alloc)
 *
 * tr_xynodes is a per-node temporary; for our nl≤48 we keep it on
 * the stack as a 2D (2 × nl) buffer per node and consume it in pass 2.
 * The Fortran allocates it for ALL nodes simultaneously
 * (`tr_xynodes(2, nl-1, N)`); we reuse a per-node stack version
 * because pass 2 reads only the current node.
 */
void fesom_diff_ver_part_redi_expl(int                          tr_idx,
                                   fesom_gm                    *gm,
                                   const struct fesom_aux      *aux,
                                   const struct fesom_mesh     *mesh,
                                   struct fesom_tracers        *tracers,
                                   struct fesom_partit         *partit)
{
    (void)aux;   /* not used in this routine — Ki and slope_tapered come from gm */
    if (!gm) return;

    const int    nl     = mesh->nl;
    const int    nl1    = nl - 1;
    const int    myDim  = mesh->myDim_nod2D;
    const int    Etot   = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    const real_t dt     = (real_t)FESOM_PHASE1_DT;

    /* Fortran's `values` at the time tr_xy is built (oce_tracer_mod.F90:127)
     * is the PRE-STEP T (advection hasn't run yet). Our C port runs
     * ale_reconstruct INSIDE fesom_tracer_advect_one_fct, so by the time
     * we get here `values` holds the post-advection T. The pre-step T is
     * preserved in `valuesold` (set in init_tracers_AB_one,
     * fesom_tracer_adv.c:198). Use it to match Fortran's tr_xy semantics. */
    real_t *valsold = tracers->data[tr_idx].valuesold;     /* [N * nl] */
    /* Fortran writes Redi flux into del_ttf BEFORE ale_reconstruct
     * (oce_ale_tracer.F90:394 → 468-471). The C port has reconstruct
     * already done — apply the Redi contribution directly to `values`
     * with the same `/hnode_new` factor that reconstruct would have. */
    real_t *vals    = tracers->data[tr_idx].values;        /* [N * nl] */

    /* ---- Step 1: tr_xy = ∇_h(valsAB) per element ------------ */
    /* Bounds: Fortran allocates and fills tr_xy for full element halo
     * (myDim+eDim+eXDim) so all readers (including diff_part_hor_redi)
     * have valid data. We fill myDim_elem2D then halo-exchange. */
    real_t *txy = gm->tr_xy;
    /* Zero the full extent so halo entries (eDim+eXDim) start clean. */
    memset(txy, 0, (size_t)Etot * (size_t)nl1 * 2 * sizeof(real_t));

    for (int el = 0; el < mesh->myDim_elem2D; ++el) {
        int v0 = mesh->elem_nodes[3*el + 0];
        int v1 = mesh->elem_nodes[3*el + 1];
        int v2 = mesh->elem_nodes[3*el + 2];
        const real_t *g = &mesh->gradient_sca[6 * el];
        int nle = mesh->nlevels[el] - 1;        /* exclusive */
        int ule = mesh->ulevels[el] - 1;        /* 0-based */
        for (int nz = ule; nz < nle; ++nz) {
            real_t T0 = valsold[(size_t)v0 * nl + nz];
            real_t T1 = valsold[(size_t)v1 * nl + nz];
            real_t T2 = valsold[(size_t)v2 * nl + nz];
            txy[(size_t)el * nl1 * 2 + nz * 2 + 0] = g[0]*T0 + g[1]*T1 + g[2]*T2;
            txy[(size_t)el * nl1 * 2 + nz * 2 + 1] = g[3]*T0 + g[4]*T1 + g[5]*T2;
        }
    }
    if (partit && partit->npes > 1) {
        /* Fortran exchange_elem(tr_xy) — full element halo, both components. */
        fesom_halo_exchange(txy, FESOM_HALO_ELEM2D_FULL, nl1, 2, partit);
    }

    /* ---- Step 2: per-node Redi vertical-explicit flux ------ */
    real_t txn[NL_MAX], tyn[NL_MAX];
    real_t zbar_n[NL_MAX], z_n[NL_MAX];
    real_t vd_flux[NL_MAX];
    FESOM_CHECK(nl <= NL_MAX, "diff_ver_part_redi_expl: nl %d > NL_MAX", nl);

    const real_t *st = gm->slope_tapered;
    const real_t *Ki = gm->Ki;

    for (int n = 0; n < myDim; ++n) {
        int nle = mesh->nlevels_nod2D[n] - 1;     /* exclusive */
        int ule = mesh->ulevels_nod2D[n] - 1;     /* 0-based */
        if (nle <= ule) continue;

        /* 2a. tr_xynodes = (1/3) · Σ tr_xy * elem_area / areasvol[nz,n].
         * Fortran 1117-1135. Iterate the per-node level range. */
        for (int nz = 0; nz < nl1; ++nz) { txn[nz] = 0.0; tyn[nz] = 0.0; }
        for (int nz = ule; nz < nle; ++nz) {
            real_t Tx = 0.0, Ty = 0.0;
            int o0 = mesh->nod_in_elem2D_offsets[n];
            int o1 = mesh->nod_in_elem2D_offsets[n + 1];
            for (int k = o0; k < o1; ++k) {
                int el = mesh->nod_in_elem2D[k];
                int el_nle = mesh->nlevels[el] - 1;     /* exclusive */
                int el_ule = mesh->ulevels[el] - 1;
                if (nz >= el_ule && nz < el_nle) {
                    real_t a = mesh->elem_area[el];
                    Tx += txy[(size_t)el * nl1 * 2 + nz * 2 + 0] * a;
                    Ty += txy[(size_t)el * nl1 * 2 + nz * 2 + 1] * a;
                }
            }
            real_t inv = 1.0 / (3.0 * mesh->areasvol[(size_t)n * nl + nz]);
            txn[nz] = Tx * inv;
            tyn[nz] = Ty * inv;
        }

        /* 2b. Build zbar_n, z_n on OLD mesh (Fortran 1146-1154 uses hnode
         * NOT hnode_new). For full cells / no cavity, the bottom interface
         * sits at mesh->zbar[nle]. */
        for (int nz = 0; nz < nl; ++nz) zbar_n[nz] = 0.0;
        for (int nz = 0; nz < nl1; ++nz) z_n[nz] = 0.0;
        zbar_n[nle] = mesh->zbar[nle];
        z_n[nle - 1] = zbar_n[nle] + mesh->hnode[(size_t)n * nl + (nle - 1)] * 0.5;
        for (int nz = nle - 1; nz >= ule + 1; --nz) {
            zbar_n[nz]    = zbar_n[nz + 1]
                          + mesh->hnode[(size_t)n * nl + nz];
            z_n[nz - 1]   = zbar_n[nz]
                          + mesh->hnode[(size_t)n * nl + (nz - 1)] * 0.5;
        }
        zbar_n[ule] = zbar_n[ule + 1]
                    + mesh->hnode[(size_t)n * nl + ule];

        /* 2c. vd_flux at interfaces (Fortran 1157-1162). vd_flux[ule] and
         * vd_flux[nle] act as zero boundaries (we never write them, and
         * the divergence loop below reads them as 0 since we memset). */
        for (int nz = 0; nz < nl; ++nz) vd_flux[nz] = 0.0;
        for (int nz = ule + 1; nz < nle; ++nz) {
            real_t st_x_up = st[(size_t)n * nl1 * 3 + (nz - 1) * 3 + 0];
            real_t st_y_up = st[(size_t)n * nl1 * 3 + (nz - 1) * 3 + 1];
            real_t st_x_dn = st[(size_t)n * nl1 * 3 + nz       * 3 + 0];
            real_t st_y_dn = st[(size_t)n * nl1 * 3 + nz       * 3 + 1];
            real_t Ki_up   = Ki[(size_t)n * nl + (nz - 1)];
            real_t Ki_dn   = Ki[(size_t)n * nl + nz];

            real_t up = (z_n[nz - 1] - zbar_n[nz])
                      * (st_x_up * txn[nz - 1] + st_y_up * tyn[nz - 1])
                      * Ki_up;
            real_t dn = (zbar_n[nz] - z_n[nz])
                      * (st_x_dn * txn[nz]     + st_y_dn * tyn[nz])
                      * Ki_dn;
            real_t mid = z_n[nz - 1] - z_n[nz];
            real_t a_int = mesh->area[(size_t)n * nl + nz];
            vd_flux[nz] = (up + dn) / mid * a_int;
        }

        /* 2d. T += (vd_flux[nz] - vd_flux[nz+1]) * dt / (areasvol*hnode_new)
         * Fortran 1163-1165 writes to del_ttf; reconstruct (468-471) then
         * does T += del_ttf/hnode_new. Composed: same as the line below. */
        for (int nz = ule; nz < nle; ++nz) {
            real_t av = mesh->areasvol[(size_t)n * nl + nz];
            real_t hn = mesh->hnode_new[(size_t)n * nl + nz];
            if (av > 0.0 && hn > 0.0) {
                vals[(size_t)n * nl + nz] +=
                    (vd_flux[nz] - vd_flux[nz + 1]) * dt / (av * hn);
            }
        }
    }
}

/* ============================================================ */
/* G7b — diff_part_hor_redi (oce_ale_tracer.F90:1173-1336)       */
/* ============================================================ */
/*
 * Horizontal Redi flux on edges: builds gm->tr_z (per-node vertical
 * tracer gradient) once per call, then iterates myDim_edge2D applying
 * the Redi flux contribution at the two endpoints. Five partial-cell
 * branches handle level mismatches between the two adjacent elements:
 *   (A) levels in el(1) but not el(2) (el(2) starts deeper)
 *   (B) levels in el(2) but not el(1) (el(1) starts deeper)
 *   (C) levels common to both — averaged
 *   (D) levels in el(1) but not el(2) (el(1) extends deeper)
 *   (E) levels in el(2) but not el(1) (el(2) extends deeper)
 *
 * Reuses gm->tr_xy populated by a preceding fesom_diff_ver_part_redi_expl
 * call. Reads valuesold (= pre-step T) for the tr_z build, matching the
 * Fortran convention that tr_z = grad_z(values) at the time tr_xy is built.
 *
 * Like G7a, applies the Redi flux directly to `values` with the
 * `/hnode_new` factor (composed from Fortran's del_ttf accumulation
 * and the subsequent ale_reconstruct). For our linfs config that's the
 * identical computation as Fortran produces.
 *
 * Halo audit (E7-style — same code class as ocean FCT bug 2):
 *   tr_z is computed per-node for myDim then halo-exchanged before the
 *   edge loop reads it.
 *   tr_xy is per-element with halo exchange in G7a.
 *   slope_tapered, Ki, areasvol — already halo-coherent.
 *   The edge loop bound is myDim_edge2D (matches Fortran 1208).
 *   Per-edge writes only target the two endpoint nodes — for myDim
 *   edges, both endpoints are valid local nodes (myDim or eDim).
 */
void fesom_diff_part_hor_redi(int                          tr_idx,
                              fesom_gm                    *gm,
                              const struct fesom_aux      *aux,
                              const struct fesom_mesh     *mesh,
                              struct fesom_tracers        *tracers,
                              struct fesom_partit         *partit)
{
    (void)aux;
    if (!gm) return;

    const int    nl     = mesh->nl;
    const int    nl1    = nl - 1;
    const int    Nfull  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    EG     = mesh->myDim_edge2D;
    const real_t dt     = (real_t)FESOM_PHASE1_DT;
    const real_t isredi = 1.0;

    real_t *valsold = tracers->data[tr_idx].valuesold;
    real_t *vals    = tracers->data[tr_idx].values;
    real_t *txy     = gm->tr_xy;            /* [E_full * (nl-1) * 2] — built by G7a */
    real_t *trz     = gm->tr_z;             /* [N * nl] */
    const real_t *st = gm->slope_tapered;   /* [N * (nl-1) * 3] */
    const real_t *Ki = gm->Ki;              /* [N * nl] */

    /* ---- Step 1: tr_z = ∂ valuesold/∂z at interfaces (Fortran 219-228) -------
     * Fortran loops myDim+eDim and exchanges; we mirror by looping myDim then
     * halo-exchange. tr_z[nzmin] and tr_z[nzmax] are 0 boundaries. */
    /* Zero halo + boundary slots first. */
    memset(trz, 0, (size_t)Nfull * (size_t)nl * sizeof(real_t));
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        int nzmax = mesh->nlevels_nod2D[n] - 1;     /* exclusive: layer count */
        int nzmin = mesh->ulevels_nod2D[n] - 1;     /* 0-based */
        if (nzmax <= nzmin + 1) continue;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t T_up = valsold[(size_t)n * nl + (nz - 1)];
            real_t T_dn = valsold[(size_t)n * nl + nz];
            real_t h_up = mesh->hnode[(size_t)n * nl + (nz - 1)];
            real_t h_dn = mesh->hnode[(size_t)n * nl + nz];
            real_t dz   = 0.5 * (h_up + h_dn);
            trz[(size_t)n * nl + nz] = (T_up - T_dn) / dz;
        }
        /* nzmin and nzmax stay 0 (zeroed above). */
    }
    if (partit && partit->npes > 1) {
        fesom_exchange_nod3D(trz, nl, partit);
    }

    /* ---- Step 2: edge loop ---------------------------------- */
    real_t rhs1[NL_MAX], rhs2[NL_MAX];
    FESOM_CHECK(nl <= NL_MAX, "diff_part_hor_redi: nl %d > NL_MAX", nl);

    for (int ed = 0; ed < EG; ++ed) {
        int e1 = mesh->edges[2*ed + 0];
        int e2 = mesh->edges[2*ed + 1];
        int el1 = mesh->edge_tri[2*ed + 0];
        int el2 = mesh->edge_tri[2*ed + 1];
        if (el1 < 0) continue;          /* malformed edge — skip */

        real_t dxL = mesh->edge_cross_dxdy[4*ed + 0];
        real_t dyL = mesh->edge_cross_dxdy[4*ed + 1];
        real_t dxR = 0.0, dyR = 0.0;

        int nl1_e = mesh->nlevels[el1] - 1;
        int ul1_e = mesh->ulevels[el1] - 1;
        int nl2_e = -1;                   /* sentinel for "no el(2)" */
        int ul2_e = -1;
        if (el2 >= 0) {
            nl2_e = mesh->nlevels[el2] - 1;
            ul2_e = mesh->ulevels[el2] - 1;
            dxR = mesh->edge_cross_dxdy[4*ed + 2];
            dyR = mesh->edge_cross_dxdy[4*ed + 3];
        }

        /* Common range. Fortran: nl12=min(nl1,nl2), ul12=max(ul1,ul2). */
        int nl12 = (el2 >= 0) ? ((nl1_e < nl2_e) ? nl1_e : nl2_e) : nl1_e;
        int ul12 = (el2 >= 0) ? ((ul1_e > ul2_e) ? ul1_e : ul2_e) : ul1_e;

        /* Zero rhs. Touch only the level range we'll write to + boundaries. */
        for (int nz = 0; nz < nl1; ++nz) { rhs1[nz] = 0.0; rhs2[nz] = 0.0; }

        /* Helper: compute Kh, Tz, SxTz, SyTz for a given level given enodes (e1, e2). */
        #define COMPUTE_KH_TZ_S(NZ)                                                                   \
            real_t Kh   = 0.5 * (Ki[(size_t)e1 * nl + (NZ)] + Ki[(size_t)e2 * nl + (NZ)]);            \
            real_t Tz1  = 0.5 * (trz[(size_t)e1 * nl + (NZ)] + trz[(size_t)e1 * nl + ((NZ)+1)]);      \
            real_t Tz2  = 0.5 * (trz[(size_t)e2 * nl + (NZ)] + trz[(size_t)e2 * nl + ((NZ)+1)]);      \
            real_t st_x_1 = st[(size_t)e1 * nl1 * 3 + (NZ) * 3 + 0];                                  \
            real_t st_y_1 = st[(size_t)e1 * nl1 * 3 + (NZ) * 3 + 1];                                  \
            real_t st_x_2 = st[(size_t)e2 * nl1 * 3 + (NZ) * 3 + 0];                                  \
            real_t st_y_2 = st[(size_t)e2 * nl1 * 3 + (NZ) * 3 + 1];                                  \
            real_t SxTz = 0.5 * (Tz1 * st_x_1 + Tz2 * st_x_2);                                        \
            real_t SyTz = 0.5 * (Tz1 * st_y_1 + Tz2 * st_y_2);

        /* (A) ul1..ul12-1 — el(1) only (Fortran 1233-1246) */
        for (int nz = ul1_e; nz < ul12; ++nz) {
            COMPUTE_KH_TZ_S(nz)
            real_t dz = mesh->helem[(size_t)el1 * nl + nz];
            real_t Tx = txy[(size_t)el1 * nl1 * 2 + nz * 2 + 0];
            real_t Ty = txy[(size_t)el1 * nl1 * 2 + nz * 2 + 1];
            real_t Fx = Kh * (Tx + SxTz * isredi);
            real_t Fy = Kh * (Ty + SyTz * isredi);
            real_t c  = (-dxL * Fy + dyL * Fx) * dz;
            rhs1[nz] += c;
            rhs2[nz] -= c;
        }

        /* (B) ul2..ul12-1 — el(2) only (Fortran 1249-1264) */
        if (el2 >= 0) {
            for (int nz = ul2_e; nz < ul12; ++nz) {
                COMPUTE_KH_TZ_S(nz)
                real_t dz = mesh->helem[(size_t)el2 * nl + nz];
                real_t Tx = txy[(size_t)el2 * nl1 * 2 + nz * 2 + 0];
                real_t Ty = txy[(size_t)el2 * nl1 * 2 + nz * 2 + 1];
                real_t Fx = Kh * (Tx + SxTz * isredi);
                real_t Fy = Kh * (Ty + SyTz * isredi);
                real_t c  = (dxR * Fy - dyR * Fx) * dz;
                rhs1[nz] += c;
                rhs2[nz] -= c;
            }
        }

        /* (C) ul12..nl12 — both elements (Fortran 1267-1280) */
        for (int nz = ul12; nz < nl12; ++nz) {
            COMPUTE_KH_TZ_S(nz)
            real_t dz = (el2 >= 0)
                      ? 0.5 * (mesh->helem[(size_t)el1 * nl + nz]
                             + mesh->helem[(size_t)el2 * nl + nz])
                      : mesh->helem[(size_t)el1 * nl + nz];
            real_t Tx, Ty;
            if (el2 >= 0) {
                Tx = 0.5 * (txy[(size_t)el1 * nl1 * 2 + nz * 2 + 0]
                          + txy[(size_t)el2 * nl1 * 2 + nz * 2 + 0]);
                Ty = 0.5 * (txy[(size_t)el1 * nl1 * 2 + nz * 2 + 1]
                          + txy[(size_t)el2 * nl1 * 2 + nz * 2 + 1]);
            } else {
                Tx = txy[(size_t)el1 * nl1 * 2 + nz * 2 + 0];
                Ty = txy[(size_t)el1 * nl1 * 2 + nz * 2 + 1];
            }
            real_t Fx = Kh * (Tx + SxTz * isredi);
            real_t Fy = Kh * (Ty + SyTz * isredi);
            real_t c  = ((dxR - dxL) * Fy - (dyR - dyL) * Fx) * dz;
            rhs1[nz] += c;
            rhs2[nz] -= c;
        }

        /* (D) nl12+1..nl1 — el(1) only (Fortran 1283-1296) */
        for (int nz = nl12; nz < nl1_e; ++nz) {
            COMPUTE_KH_TZ_S(nz)
            real_t dz = mesh->helem[(size_t)el1 * nl + nz];
            real_t Tx = txy[(size_t)el1 * nl1 * 2 + nz * 2 + 0];
            real_t Ty = txy[(size_t)el1 * nl1 * 2 + nz * 2 + 1];
            real_t Fx = Kh * (Tx + SxTz * isredi);
            real_t Fy = Kh * (Ty + SyTz * isredi);
            real_t c  = (-dxL * Fy + dyL * Fx) * dz;
            rhs1[nz] += c;
            rhs2[nz] -= c;
        }

        /* (E) nl12+1..nl2 — el(2) only (Fortran 1299-1312) */
        if (el2 >= 0) {
            for (int nz = nl12; nz < nl2_e; ++nz) {
                COMPUTE_KH_TZ_S(nz)
                real_t dz = mesh->helem[(size_t)el2 * nl + nz];
                real_t Tx = txy[(size_t)el2 * nl1 * 2 + nz * 2 + 0];
                real_t Ty = txy[(size_t)el2 * nl1 * 2 + nz * 2 + 1];
                real_t Fx = Kh * (Tx + SxTz * isredi);
                real_t Fy = Kh * (Ty + SyTz * isredi);
                real_t c  = (dxR * Fy - dyR * Fx) * dz;
                rhs1[nz] += c;
                rhs2[nz] -= c;
            }
        }

        #undef COMPUTE_KH_TZ_S

        /* Apply rhs to T values at the two endpoints. Fortran 1314-1331:
         * ul12 = ul1 (default); if (ul2 > 0) ul12 = min(ul1, ul2)
         * nl12 := max(nl1, nl2)
         * del_ttf[ul12..nl12, n] += rhs * dt / areasvol
         * Then ale_reconstruct does T += del_ttf/hnode_new. We compose:
         *   T += rhs * dt / (areasvol * hnode_new)
         */
        int ul_apply = ul1_e;
        if (el2 >= 0 && ul2_e < ul_apply) ul_apply = ul2_e;
        int nl_apply = (el2 >= 0 && nl2_e > nl1_e) ? nl2_e : nl1_e;

        for (int nz = ul_apply; nz < nl_apply; ++nz) {
            real_t hn1 = mesh->hnode_new[(size_t)e1 * nl + nz];
            real_t av1 = mesh->areasvol [(size_t)e1 * nl + nz];
            if (av1 > 0.0 && hn1 > 0.0) {
                vals[(size_t)e1 * nl + nz] += rhs1[nz] * dt / (av1 * hn1);
            }
            real_t hn2 = mesh->hnode_new[(size_t)e2 * nl + nz];
            real_t av2 = mesh->areasvol [(size_t)e2 * nl + nz];
            if (av2 > 0.0 && hn2 > 0.0) {
                vals[(size_t)e2 * nl + nz] += rhs2[nz] * dt / (av2 * hn2);
            }
        }
    }
}

/* ============================================================ */
/* G4 — fer_gamma2vel (oce_fer_gm.F90:168-210)                  */
/* ============================================================ */
/*
 * Element-loop reconstruction of bolus velocity from the streamfunction
 * differences across vertical interfaces:
 *   fer_uv(comp, nz, el) = (1/3) · Σ_v (Γ(comp, nz, v) − Γ(comp, nz+1, v))
 *                          / helem(nz, el)
 *
 * Halo: exchange_elem(fer_uv) — Fortran 208. Stride 2.
 */
void fesom_fer_gamma2vel(struct fesom_dyn *dyn,
                         const struct fesom_mesh *mesh,
                         const fesom_gm *gm,
                         struct fesom_partit *partit)
{
    const int  nl      = mesh->nl;
    const int  myDim_e = mesh->myDim_elem2D;
    const real_t onethird = 1.0 / 3.0;

    for (int el = 0; el < myDim_e; ++el) {
        int v0 = mesh->elem_nodes[3*el + 0];
        int v1 = mesh->elem_nodes[3*el + 1];
        int v2 = mesh->elem_nodes[3*el + 2];
        int nzmax = mesh->nlevels[el] - 1;       /* nl1 = layer count */
        int nzmin = mesh->ulevels[el] - 1;       /* 0-based */

        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t h = mesh->helem[(size_t)el * nl + nz];
            if (!(h > 0.0)) continue;            /* dry / sentinel layer */
            real_t zinv = onethird / h;

            real_t gx_top = gm->fer_gamma[(size_t)v0 * nl * 2 + nz       * 2 + 0]
                          + gm->fer_gamma[(size_t)v1 * nl * 2 + nz       * 2 + 0]
                          + gm->fer_gamma[(size_t)v2 * nl * 2 + nz       * 2 + 0];
            real_t gx_bot = gm->fer_gamma[(size_t)v0 * nl * 2 + (nz + 1) * 2 + 0]
                          + gm->fer_gamma[(size_t)v1 * nl * 2 + (nz + 1) * 2 + 0]
                          + gm->fer_gamma[(size_t)v2 * nl * 2 + (nz + 1) * 2 + 0];
            real_t gy_top = gm->fer_gamma[(size_t)v0 * nl * 2 + nz       * 2 + 1]
                          + gm->fer_gamma[(size_t)v1 * nl * 2 + nz       * 2 + 1]
                          + gm->fer_gamma[(size_t)v2 * nl * 2 + nz       * 2 + 1];
            real_t gy_bot = gm->fer_gamma[(size_t)v0 * nl * 2 + (nz + 1) * 2 + 1]
                          + gm->fer_gamma[(size_t)v1 * nl * 2 + (nz + 1) * 2 + 1]
                          + gm->fer_gamma[(size_t)v2 * nl * 2 + (nz + 1) * 2 + 1];

            dyn->fer_uv[(size_t)el * nl * 2 + nz * 2 + 0] = (gx_top - gx_bot) * zinv;
            dyn->fer_uv[(size_t)el * nl * 2 + nz * 2 + 1] = (gy_top - gy_bot) * zinv;
        }
    }

    if (partit && partit->npes > 1) {
        fesom_halo_exchange(dyn->fer_uv, FESOM_HALO_ELEM2D_FULL, nl, 2, partit);
    }
}

/*===========================================================================
 * M2.5b — DEVICE (Kokkos) twins of the substep-1b GM chain. Each is a verbatim
 * parallel_for port of its C twin above; every constant / loop bound / ODM95 +
 * GMzexp tapering association is copied EXACTLY (no-simplification rule, the
 * GM/Redi numerics knobs in docs/PORTING_LESSONS). The kernels do PURE compute +
 * modify_device(); the DRIVER (fesom_step.cpp substep 1b) owns the IN rail
 * (sync_device all inputs, L28), the per-kernel sync_host + halo (the C twins'
 * internal exchanges move to the driver, the ALE/M2.5 pattern — none of these is
 * a D21 internal bracket), and the one re-push (fer_gamma → fer_gamma2vel, which
 * reads it at HALO vertices, L30). All five are race-free MAPS/GATHERS/per-node
 * TDMAs (each entity owns its output column) → Serial AND OpenMP bit-identical
 * (no scatter, no reduction — unlike vert_vel/visc_filt). NL_MAX stack scratch is
 * lambda-local (slow device-local memory, slow-first accepted, the EOS/L31 shape).
 *===========================================================================*/

/*--- compute_sigma_xy — DEVICE (substep 1b) ---------------------------------
 * Per-node gather: density gradient on neutral surfaces, area-weighted over the
 * surrounding elements. Each node owns its sigma_xy[n,:,:] slab → race-free. */
void fesom_compute_sigma_xy_kk(struct fesom_aux           *aux,
                               const struct fesom_tracers *tracers,
                               const struct fesom_mesh    *mesh,
                               fesom_gm                   *gm)
{
    const int  nl       = mesh->nl;
    const int  myDim    = mesh->myDim_nod2D;
    const real_t rho_ref = (real_t)FESOM_DENSITY_0;
    FESOM_CHECK(nl <= NL_MAX, "sigma_xy_kk: nl %d > NL_MAX", nl);

    auto T          = tracers->data[FESOM_TRACER_T].values_fld.d();
    auto S          = tracers->data[FESOM_TRACER_S].values_fld.d();
    auto alpha      = aux->sw_alpha_fld.d();
    auto beta       = aux->sw_beta_fld.d();
    auto sxy        = gm->sigma_xy_fld.d();
    auto nlev_n     = mesh->nlevels_nod2D_fld.d();
    auto ulev_n     = mesh->ulevels_nod2D_fld.d();
    auto nlev_e     = mesh->nlevels_fld.d();
    auto ulev_e     = mesh->ulevels_fld.d();
    auto off        = mesh->nod_in_elem2D_offsets_fld.d();
    auto nie        = mesh->nod_in_elem2D_fld.d();
    auto grad       = mesh->gradient_sca_fld.d();
    auto elnod      = mesh->elem_nodes_fld.d();
    auto earea      = mesh->elem_area_fld.d();

    /* M5.19 (bucket-A coalescing lever): ONE THREAD PER (node, LEVEL) — flat
     * RangePolicy(0, myDim*nl) decoded to (n,nz). The old per-node column scratch
     * tx/ty/sx/sy/vol[NL_MAX] held PER-LEVEL accumulators (no cross-level reduction),
     * so each (n,nz) thread re-loops the node's surrounding elements (k=o0..o1, the
     * SAME order) and accumulates ONLY its level nz in registers → the scratch is
     * eliminated and the per-(n,nz) arithmetic is IEEE op-for-op identical → byte-
     * identical (Serial AND OpenMP; race-free). Coalescing: sxy is node-major (stride
     * 2/level) and the T/S gather reads i{0,1,2}=v*nl+nz are level-contiguous, so a warp
     * of 32 consecutive nz (same node) reads/writes contiguously; the per-element scalars
     * (grad/elnod/earea) are warp-broadcast (all 32 share node n's elements). The old
     * one-thread-per-node strided every sxy store by nl*2 (uncoalesced). The element loop
     * is re-walked per level but the TOTAL arithmetic is unchanged (the element/level
     * loops just swap order). See docs/GPU_FIDELITY.md §M5.19. */
    const int myDimNL = myDim * nl;
    Kokkos::parallel_for("fesom_gm_sigma_xy", Kokkos::RangePolicy<>(0, myDimNL),
        KOKKOS_LAMBDA(const int idx) {
            const int n  = idx / nl;
            const int nz = idx - n * nl;
            int nle_node = nlev_n(n) - 1;       /* exclusive */
            int ule_node = ulev_n(n) - 1;       /* 0-based   */
            if (nz < ule_node || nz >= nle_node) return;   /* outside the node's column (dry → empty range) */

            real_t tx = 0.0, ty = 0.0, sx = 0.0, sy = 0.0, vol = 0.0;
            int o0 = off(n);
            int o1 = off(n + 1);
            for (int k = o0; k < o1; ++k) {
                int el = nie(k);
                int nle = nlev_e(el) - 1;       /* exclusive  */
                int ule = ulev_e(el) - 1;       /* 0-based    */
                if (nz < ule || nz >= nle) continue;        /* element doesn't reach this level */
                real_t g0 = grad(6*el + 0), g1 = grad(6*el + 1), g2 = grad(6*el + 2);
                real_t g3 = grad(6*el + 3), g4 = grad(6*el + 4), g5 = grad(6*el + 5);
                int v0 = elnod(3*el + 0);
                int v1 = elnod(3*el + 1);
                int v2 = elnod(3*el + 2);
                real_t a = earea(el);
                /* Tracer T/S stored with stride nl (NOT nl-1). */
                size_t i0 = (size_t)v0 * nl + nz;
                size_t i1 = (size_t)v1 * nl + nz;
                size_t i2 = (size_t)v2 * nl + nz;
                vol += a;
                tx += (g0*T(i0) + g1*T(i1) + g2*T(i2)) * a;
                ty += (g3*T(i0) + g4*T(i1) + g5*T(i2)) * a;
                sx += (g0*S(i0) + g1*S(i1) + g2*S(i2)) * a;
                sy += (g3*S(i0) + g4*S(i1) + g5*S(i2)) * a;
            }

            real_t inv_vol = (vol > 0.0) ? 1.0 / vol : 0.0;
            real_t a = alpha((size_t)n * nl + nz);
            real_t b = beta ((size_t)n * nl + nz);
            sxy((size_t)n * nl * 2 + nz * 2 + 0) =
                (-a * tx + b * sx) * inv_vol * rho_ref;
            sxy((size_t)n * nl * 2 + nz * 2 + 1) =
                (-a * ty + b * sy) * inv_vol * rho_ref;
        });

    gm->sigma_xy_fld.modify_device();
}

/*--- compute_neutral_slope — DEVICE (substep 1b) ----------------------------
 * Per-node: neutral_slope = sigma_xy/N², ODM95 c1 tapering, slope_tapered.
 * Active config: scaling_ODM95=.true. (c1), scaling_LDD97=.false. (c2≡1),
 * Fer_GM ∧ Redi ∧ Redi_Ktaper → fer_tapfac=c1, slope_tapered=ns*sqrt(c1). */
void fesom_compute_neutral_slope_kk(struct fesom_aux        *aux,
                                    const struct fesom_mesh *mesh,
                                    fesom_gm                *gm)
{
    const int    nl       = mesh->nl;
    const int    nl1      = nl - 1;
    const int    myDim    = mesh->myDim_nod2D;
    const real_t g        = (real_t)FESOM_G;
    const real_t rho_ref  = (real_t)FESOM_DENSITY_0;
    const real_t eps      = 5.0e-6;
    const real_t eps_sq   = eps * eps;
    const real_t ODM95_Scr = 0.2e-2;
    const real_t ODM95_Sd  = 1.0e-3;
    FESOM_CHECK(nl <= NL_MAX, "neutral_slope_kk: nl %d > NL_MAX", nl);

    auto bvfreq = aux->bvfreq_fld.d();
    auto sxy    = gm->sigma_xy_fld.d();
    auto ns     = gm->neutral_slope_fld.d();
    auto st     = gm->slope_tapered_fld.d();
    auto tf     = gm->fer_tapfac_fld.d();
    auto nlev_n = mesh->nlevels_nod2D_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();

    /* M5.19 (bucket-A coalescing lever): ONE THREAD PER (node, LEVEL) — flat
     * RangePolicy(0, myDim*nl1) decoded to (n,nz). The old per-node kernel ran 3 passes
     * with a c1[NL_MAX] column scratch, but every level is INDEPENDENT (Pass 2's c1[nz]
     * depends only on level nz via bvfreq[nz]/bvfreq[nz+1] — an INPUT read, not a written
     * neighbor; Pass 3 re-reads ns[nz] which Pass 1 just wrote = the same bits), so the
     * three passes FUSE into one per-(n,nz) computation with c1 register-local → the
     * scratch is eliminated and the arithmetic is IEEE op-for-op identical → byte-identical
     * (Serial AND OpenMP; race-free). Coalescing: ns/st are node-major (stride 3/level),
     * tf/bvfreq stride 1, sxy stride 2 → a warp of 32 consecutive nz (same node) reads/
     * writes contiguously; the old one-thread-per-node strided every store by nl1*3 / nl
     * (uncoalesced). slope_tapered is zeroed for EVERY nz∈[0,nl1) exactly as the C zero
     * loop (then overwritten on active levels); ns/tf are written only on active levels,
     * matching the C. See docs/GPU_FIDELITY.md §M5.19. */
    const int myDimNL1 = myDim * nl1;
    Kokkos::parallel_for("fesom_gm_neutral_slope", Kokkos::RangePolicy<>(0, myDimNL1),
        KOKKOS_LAMBDA(const int idx) {
            const int n  = idx / nl1;
            const int nz = idx - n * nl1;

            /* Zero slope_tapered for every nz (Fortran line 2979) — overwritten below on active levels. */
            st((size_t)n * nl1 * 3 + nz * 3 + 0) = 0.0;
            st((size_t)n * nl1 * 3 + nz * 3 + 1) = 0.0;
            st((size_t)n * nl1 * 3 + nz * 3 + 2) = 0.0;

            int nle = nlev_n(n) - 1;       /* exclusive index for slopes */
            int ule = ulev_n(n) - 1;       /* 0-based */
            if (nle <= ule) return;        /* dry node */
            if (nz < ule || nz >= nle) return;   /* inactive level — st stays zeroed; ns/tf untouched */

            /* Pass 1: neutral_slope from sigma_xy / N². */
            real_t bv_sum = bvfreq((size_t)n * nl + nz)
                          + bvfreq((size_t)n * nl + (nz + 1));
            real_t denom  = (bv_sum > eps_sq) ? bv_sum : eps_sq;
            real_t ro_z_inv = 2.0 * g / rho_ref / denom;
            real_t sx = sxy((size_t)n * nl * 2 + nz * 2 + 0) * ro_z_inv;
            real_t sy = sxy((size_t)n * nl * 2 + nz * 2 + 1) * ro_z_inv;
            real_t sm = Kokkos::sqrt(sx*sx + sy*sy);
            ns((size_t)n * nl1 * 3 + nz * 3 + 0) = sx;
            ns((size_t)n * nl1 * 3 + nz * 3 + 1) = sy;
            ns((size_t)n * nl1 * 3 + nz * 3 + 2) = sm;

            /* Pass 2: ODM95 tapering c1 (register-local; reads ns[nz]==sm just written). */
            real_t v = 0.5 * (1.0 + Kokkos::tanh((ODM95_Scr - sm) / ODM95_Sd));
            real_t bv0 = bvfreq((size_t)n * nl + nz);
            real_t bv1 = bvfreq((size_t)n * nl + (nz + 1));
            if (bv0 <= 0.0 || bv1 <= 0.0) v = 0.0;
            real_t cc = v;                               /* c1 * c2 = c1 (c2≡1) */

            /* Pass 3: fer_tapfac=c1, slope_tapered=ns*sqrt(c1) (sx/sy/sm == ns[nz]). */
            tf((size_t)n * nl + nz) = cc;
            real_t s = Kokkos::sqrt(cc);
            st((size_t)n * nl1 * 3 + nz * 3 + 0) = sx * s;
            st((size_t)n * nl1 * 3 + nz * 3 + 1) = sy * s;
            st((size_t)n * nl1 * 3 + nz * 3 + 2) = sm * s;
        });

    gm->neutral_slope_fld.modify_device();
    gm->slope_tapered_fld.modify_device();
    gm->fer_tapfac_fld.modify_device();
}

/*--- init_redi_gm — DEVICE (substep 1b) -------------------------------------
 * Two per-node parallel_fors (D20 launch barrier orders pass-3's read of the
 * pass-1 fer_K[nzmin]). Active sub-features only (default namelist.oce):
 * Fer_GM=T, Redi=T, scaling_resolution=T(order 2), scaling_GMzexp=T,
 * Redi_Ktaper=T. ⚠️ every constant + the GMzexp depth-exp + the ODM95 sqrt-taper
 * copied verbatim. Each node owns its fer_K/Ki/fer_C/fer_scal column → race-free. */
void fesom_init_redi_gm_kk(struct fesom_aux        *aux,
                           const struct fesom_mesh *mesh,
                           fesom_gm                *gm)
{
    const int    nl       = mesh->nl;
    const int    myDim    = mesh->myDim_nod2D;
    const real_t pi       = 3.14159265358979323846;

    /* Active-config namelist constants. */
    const real_t K_GM_max     = 1000.0;
    const real_t K_GM_min     = 2.0;
    const real_t K_GM_cmin    = 0.1;
    const real_t K_GM_cm      = 3.0;
    const real_t Redi_Kmin    = 100.0;
    const real_t Redi_Kmax    = K_GM_max;     /* auto-sync */
    const real_t GMzexp_zref  = 500.0;
    const real_t GMzexp_smin  = 0.6;
    const real_t refscalresol = 100000.0;     /* 100 km */
    const real_t inv_refscalresol_sq = 1.0 / (refscalresol * refscalresol);
    FESOM_CHECK(nl <= NL_MAX, "init_redi_gm_kk: nl %d > NL_MAX", nl);

    auto bvfreq    = aux->bvfreq_fld.d();
    auto hnode_new = mesh->hnode_new_fld.d();
    auto area      = mesh->area_fld.d();
    auto zbar_3d_n = mesh->zbar_3d_n_fld.d();
    auto fer_scal  = gm->fer_scal_fld.d();
    auto fer_K     = gm->fer_K_fld.d();
    auto fer_C     = gm->fer_C_fld.d();
    auto Ki        = gm->Ki_fld.d();
    auto tf        = gm->fer_tapfac_fld.d();
    auto nlev_min  = mesh->nlevels_nod2D_min_fld.d();
    auto ulev_max  = mesh->ulevels_nod2D_max_fld.d();
    auto nlev_n    = mesh->nlevels_nod2D_fld.d();
    auto ulev_n    = mesh->ulevels_nod2D_fld.d();

    /* --- Pass 1: F1(x,y) per-node scalar (+ inline Redi=GM sync). --------- */
    Kokkos::parallel_for("fesom_gm_init_redi_p1", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            int nzmax = nlev_min(n) - 1;   /* exclusive */
            int nzmin = ulev_max(n) - 1;   /* 0-based */

            real_t cm_sum = 0.0;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t bv0 = bvfreq((size_t)n * nl + nz);
                real_t bv1 = bvfreq((size_t)n * nl + (nz + 1));
                real_t v0 = (bv0 > 0.0) ? Kokkos::sqrt(bv0) : 0.0;
                real_t v1 = (bv1 > 0.0) ? Kokkos::sqrt(bv1) : 0.0;
                cm_sum += hnode_new((size_t)n * nl + nz) * 0.5 * (v0 + v1);
            }
            real_t cm = cm_sum / pi / K_GM_cm;
            if (cm < K_GM_cmin) cm = K_GM_cmin;

            real_t area_n = area((size_t)n * nl + 0);   /* surface CV area */
            real_t scaling = Kokkos::sqrt(area_n * inv_refscalresol_sq * 2.0);
            if (scaling > 1.0) scaling = 1.0;

            fer_scal(n) = scaling;

            real_t k_top = scaling * K_GM_max;
            if (k_top < K_GM_min) k_top = K_GM_min;
            fer_K((size_t)n * nl + nzmin) = k_top;
            fer_C(n) = cm * cm;

            /* Pass 2 (inline): Redi = GM sync. */
            real_t ki_top = scaling * Redi_Kmax;
            if (ki_top < K_GM_min) ki_top = K_GM_min;
            Ki((size_t)n * nl + nzmin) = ki_top;
        });

    /* --- Pass 3: F2(z) per-node, per-level vertical scaling. -------------- */
    Kokkos::parallel_for("fesom_gm_init_redi_p3", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            int nzmax = nlev_n(n) - 1;       /* exclusive */
            int nzmin = ulev_n(n) - 1;       /* 0-based */

            real_t zscaling[NL_MAX];
            for (int nz = 0; nz < nl; ++nz) zscaling[nz] = 1.0;
            for (int nz = nzmin; nz <= nzmax; ++nz) {
                real_t z = zbar_3d_n((size_t)n * nl + nz);
                real_t v = GMzexp_smin
                         + (1.0 - GMzexp_smin) * Kokkos::exp(-Kokkos::fabs(z) / GMzexp_zref);
                if (v > 1.0)          v = 1.0;
                if (v < GMzexp_smin)  v = GMzexp_smin;
                zscaling[nz] = v;
            }

            /* Apply F2 to fer_K (Fer_GM). */
            real_t k_top = fer_K((size_t)n * nl + nzmin);
            for (int nz = nzmin + 1; nz <= nzmax; ++nz) {
                fer_K((size_t)n * nl + nz) = k_top * zscaling[nz];
            }
            fer_K((size_t)n * nl + nzmin) = k_top * zscaling[nzmin];

            /* Apply F2 to Ki (Redi). */
            real_t ki_top = Ki((size_t)n * nl + nzmin);
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                Ki((size_t)n * nl + nz) =
                    ki_top * 0.5 * (zscaling[nz] + zscaling[nz + 1]);
            }
            Ki((size_t)n * nl + nzmin) =
                ki_top * 0.5 * (zscaling[nzmin] + zscaling[nzmin + 1]);

            /* Redi_Ktaper = true. */
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t tff = tf((size_t)n * nl + nz);
                real_t s  = Kokkos::sqrt(tff);
                real_t k  = Ki((size_t)n * nl + nz);
                Ki((size_t)n * nl + nz) = k * s + Redi_Kmin * Kokkos::fabs(s - 1.0);
            }
        });

    gm->fer_scal_fld.modify_device();
    gm->fer_K_fld.modify_device();
    gm->fer_C_fld.modify_device();
    gm->Ki_fld.modify_device();
}

/*--- fer_solve_gamma — DEVICE (substep 1b): per-node 1D TDMA ----------------
 * Solves ∂z(C·∂z Γ) − N²·Γ = (g/ρ₀)·∇σ·K_GM(z) for the two horizontal
 * components together. The whole Thomas sweep runs SEQUENTIALLY in level INSIDE
 * the per-node lambda over NL_MAX stack scratch (the L31/impl_vert_visc shape) →
 * race-free, Serial AND OpenMP bit-identical. Verbatim copy of fesom_fer_solve_gamma. */
void fesom_fer_solve_gamma_kk(const struct fesom_aux  *aux,
                              const struct fesom_mesh *mesh,
                              fesom_gm                *gm)
{
    const int    nl       = mesh->nl;
    const int    myDim    = mesh->myDim_nod2D;
    const real_t g        = (real_t)FESOM_G;
    const real_t rho_ref  = (real_t)FESOM_DENSITY_0;
    FESOM_CHECK(nl <= NL_MAX, "fer_solve_gamma_kk: nl %d > NL_MAX", nl);

    auto bvfreq    = aux->bvfreq_fld.d();
    auto zbar      = mesh->zbar_fld.d();
    auto hnode_new = mesh->hnode_new_fld.d();
    auto fer_C     = gm->fer_C_fld.d();
    auto fer_K     = gm->fer_K_fld.d();
    auto sxy       = gm->sigma_xy_fld.d();
    auto fer_gamma = gm->fer_gamma_fld.d();
    auto nlev_min  = mesh->nlevels_nod2D_min_fld.d();
    auto ulev_max  = mesh->ulevels_nod2D_max_fld.d();
    auto nlev_n    = mesh->nlevels_nod2D_fld.d();
    auto ulev_n    = mesh->ulevels_nod2D_fld.d();

    Kokkos::parallel_for("fesom_gm_fer_solve_gamma", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            real_t zbar_n[NL_MAX], Z_n[NL_MAX];
            real_t a[NL_MAX], b[NL_MAX], c[NL_MAX];
            real_t cp[NL_MAX], tp_x[NL_MAX], tp_y[NL_MAX];
            real_t tr_x[NL_MAX], tr_y[NL_MAX];

            /* Outer bounds. 0-based throughout. */
            int nzmax_o = nlev_n(n) - 1;     /* index of bottom interface */
            int nzmin_o = ulev_n(n) - 1;     /* = 0 in our config       */
            if (nzmax_o <= nzmin_o + 1) return;   /* degenerate column */

            /* Build zbar_n, Z_n on [nzmin_o, nzmax_o]. */
            for (int nz = 0; nz < nl; ++nz) { zbar_n[nz] = 0.0; Z_n[nz] = 0.0; }
            zbar_n[nzmax_o] = zbar(nzmax_o);
            Z_n[nzmax_o - 1] = zbar_n[nzmax_o]
                             + hnode_new((size_t)n * nl + (nzmax_o - 1)) * 0.5;
            for (int nz = nzmax_o - 1; nz >= nzmin_o + 1; --nz) {
                zbar_n[nz]    = zbar_n[nz + 1]
                              + hnode_new((size_t)n * nl + nz);
                Z_n[nz - 1]   = zbar_n[nz]
                              + hnode_new((size_t)n * nl + (nz - 1)) * 0.5;
            }
            zbar_n[nzmin_o] = zbar_n[nzmin_o + 1]
                            + hnode_new((size_t)n * nl + nzmin_o);

            /* Inner bounds for TDMA. */
            int nzmax = nlev_min(n) - 1;
            int nzmin = ulev_max(n) - 1;
            if (nzmax <= nzmin + 1) {
                for (int nz = 0; nz < nl; ++nz) {
                    fer_gamma((size_t)n * nl * 2 + nz * 2 + 0) = 0.0;
                    fer_gamma((size_t)n * nl * 2 + nz * 2 + 1) = 0.0;
                }
                return;
            }

            /* Top boundary (Dirichlet). */
            a[nzmin] = 0.0; c[nzmin] = 0.0; b[nzmin] = 1.0;

            real_t zinv2 = 1.0 / (zbar_n[nzmin] - zbar_n[nzmin + 1]);

            /* Body. */
            const real_t fc = fer_C(n);
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t zinv1 = zinv2;
                zinv2 = 1.0 / (zbar_n[nz] - zbar_n[nz + 1]);
                real_t zinv  = 1.0 / (Z_n[nz - 1] - Z_n[nz]);
                a[nz] = fc * zinv1 * zinv;
                c[nz] = fc * zinv2 * zinv;
                real_t bv = bvfreq((size_t)n * nl + nz);
                if (bv < 1.0e-8) bv = 1.0e-8;
                b[nz] = -a[nz] - c[nz] - bv;
            }

            /* Bottom boundary (Dirichlet). */
            a[nzmax] = 0.0; c[nzmax] = 0.0; b[nzmax] = 1.0;

            /* RHS. */
            const real_t r = g / rho_ref;
            for (int nz = 0; nz < nl; ++nz) { tr_x[nz] = 0.0; tr_y[nz] = 0.0; }
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t sx_up = sxy((size_t)n * nl * 2 + (nz - 1) * 2 + 0);
                real_t sx_dn = sxy((size_t)n * nl * 2 + nz       * 2 + 0);
                real_t sy_up = sxy((size_t)n * nl * 2 + (nz - 1) * 2 + 1);
                real_t sy_dn = sxy((size_t)n * nl * 2 + nz       * 2 + 1);
                real_t k = fer_K((size_t)n * nl + nz);
                tr_x[nz] = r * 0.5 * (sx_up + sx_dn) * k;
                tr_y[nz] = r * 0.5 * (sy_up + sy_dn) * k;
            }

            /* Thomas sweep. */
            cp[nzmin]   = c[nzmin]   / b[nzmin];
            tp_x[nzmin] = tr_x[nzmin] / b[nzmin];
            tp_y[nzmin] = tr_y[nzmin] / b[nzmin];
            for (int nz = nzmin + 1; nz <= nzmax; ++nz) {
                real_t m = b[nz] - cp[nz - 1] * a[nz];
                cp  [nz] = c[nz] / m;
                tp_x[nz] = (tr_x[nz] - tp_x[nz - 1] * a[nz]) / m;
                tp_y[nz] = (tr_y[nz] - tp_y[nz - 1] * a[nz]) / m;
            }

            /* Back-substitution into fer_gamma. Zero outside [nzmin, nzmax]. */
            for (int nz = 0; nz < nl; ++nz) {
                fer_gamma((size_t)n * nl * 2 + nz * 2 + 0) = 0.0;
                fer_gamma((size_t)n * nl * 2 + nz * 2 + 1) = 0.0;
            }
            fer_gamma((size_t)n * nl * 2 + nzmax * 2 + 0) = tp_x[nzmax];
            fer_gamma((size_t)n * nl * 2 + nzmax * 2 + 1) = tp_y[nzmax];
            for (int nz = nzmax - 1; nz >= nzmin; --nz) {
                real_t gx_below = fer_gamma((size_t)n * nl * 2 + (nz + 1) * 2 + 0);
                real_t gy_below = fer_gamma((size_t)n * nl * 2 + (nz + 1) * 2 + 1);
                fer_gamma((size_t)n * nl * 2 + nz * 2 + 0) = tp_x[nz] - cp[nz] * gx_below;
                fer_gamma((size_t)n * nl * 2 + nz * 2 + 1) = tp_y[nz] - cp[nz] * gy_below;
            }
        });

    gm->fer_gamma_fld.modify_device();
}

/*--- fer_gamma2vel — DEVICE (substep 1b) ------------------------------------
 * Per-element reconstruction of the bolus velocity from the streamfunction
 * differences across vertical interfaces. Reads fer_gamma at the 3 vertices
 * (which can be HALO nodes → the driver re-pushes the halo'd fer_gamma to the
 * device before this kernel, L30). Each element owns its fer_uv slot → race-free. */
void fesom_fer_gamma2vel_kk(struct fesom_dyn        *dyn,
                            const struct fesom_mesh *mesh,
                            const fesom_gm          *gm)
{
    const int  nl      = mesh->nl;
    const int  myDim_e = mesh->myDim_elem2D;
    const real_t onethird = 1.0 / 3.0;

    auto fer_uv    = dyn->fer_uv_fld.d();
    auto fer_gamma = gm->fer_gamma_fld.d();
    auto helem     = mesh->helem_fld.d();
    auto elnod     = mesh->elem_nodes_fld.d();
    auto nlev_e    = mesh->nlevels_fld.d();
    auto ulev_e    = mesh->ulevels_fld.d();

    Kokkos::parallel_for("fesom_gm_fer_gamma2vel", Kokkos::RangePolicy<>(0, myDim_e),
        KOKKOS_LAMBDA(const int el) {
            int v0 = elnod(3*el + 0);
            int v1 = elnod(3*el + 1);
            int v2 = elnod(3*el + 2);
            int nzmax = nlev_e(el) - 1;       /* nl1 = layer count */
            int nzmin = ulev_e(el) - 1;       /* 0-based */

            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t h = helem((size_t)el * nl + nz);
                if (!(h > 0.0)) continue;            /* dry / sentinel layer */
                real_t zinv = onethird / h;

                real_t gx_top = fer_gamma((size_t)v0 * nl * 2 + nz       * 2 + 0)
                              + fer_gamma((size_t)v1 * nl * 2 + nz       * 2 + 0)
                              + fer_gamma((size_t)v2 * nl * 2 + nz       * 2 + 0);
                real_t gx_bot = fer_gamma((size_t)v0 * nl * 2 + (nz + 1) * 2 + 0)
                              + fer_gamma((size_t)v1 * nl * 2 + (nz + 1) * 2 + 0)
                              + fer_gamma((size_t)v2 * nl * 2 + (nz + 1) * 2 + 0);
                real_t gy_top = fer_gamma((size_t)v0 * nl * 2 + nz       * 2 + 1)
                              + fer_gamma((size_t)v1 * nl * 2 + nz       * 2 + 1)
                              + fer_gamma((size_t)v2 * nl * 2 + nz       * 2 + 1);
                real_t gy_bot = fer_gamma((size_t)v0 * nl * 2 + (nz + 1) * 2 + 1)
                              + fer_gamma((size_t)v1 * nl * 2 + (nz + 1) * 2 + 1)
                              + fer_gamma((size_t)v2 * nl * 2 + (nz + 1) * 2 + 1);

                fer_uv((size_t)el * nl * 2 + nz * 2 + 0) = (gx_top - gx_bot) * zinv;
                fer_uv((size_t)el * nl * 2 + nz * 2 + 1) = (gy_top - gy_bot) * zinv;
            }
        });

    dyn->fer_uv_fld.modify_device();
}

/*--- M2.6-c — GM bolus velocity add/sub on device (substeps 13a / 13c) -------
 * uv += sgn*fer_uv (element, both comps), w += sgn*fer_w, w_e += sgn*fer_w (node),
 * over OWNED+HALO (myDim+eDim) — the C loops in fesom_step.cpp:13a/13c. sgn=+1 adds,
 * sgn=-1 restores. `1.0*x==x` and `-1.0*x==-x` are IEEE-exact and `a+(-b)==a-b`, so
 * the sign-parameterised map is bit-identical to the C `+=`/`-=` on Serial. Pure
 * elementwise map (no scatter/reduce) → bit-identical on Serial AND OpenMP. The
 * driver owns the IN/OUT rails. */
void fesom_gm_bolus_apply_kk(struct fesom_dyn *dyn, const struct fesom_mesh *mesh, real_t sgn)
{
    const int nl     = mesh->nl;
    const int E_loop = mesh->myDim_elem2D + mesh->eDim_elem2D;
    const int N_loop = mesh->myDim_nod2D + mesh->eDim_nod2D;

    auto uv     = dyn->uv_fld.d();
    auto w      = dyn->w_fld.d();
    auto w_e    = dyn->w_e_fld.d();
    auto fer_uv = dyn->fer_uv_fld.d();
    auto fer_w  = dyn->fer_w_fld.d();

    /* uv (element, both comps) — flat over [0, E_loop*nl*2). */
    Kokkos::parallel_for("fesom_gm_bolus_uv", Kokkos::RangePolicy<>(0, (size_t)E_loop * nl * 2),
        KOKKOS_LAMBDA(const size_t i) { uv(i) += sgn * fer_uv(i); });
    /* w, w_e (node) — flat over [0, N_loop*nl). */
    Kokkos::parallel_for("fesom_gm_bolus_w", Kokkos::RangePolicy<>(0, (size_t)N_loop * nl),
        KOKKOS_LAMBDA(const size_t i) { w(i) += sgn * fer_w(i); w_e(i) += sgn * fer_w(i); });
}

/*===========================================================================
 * M2.5b — FESOM_KK_VERIFY=gm gates. Each runs the untouched C twin beside the
 * device result on the same live state, reports per-field max|Δ|, and asserts
 * max|Δ|==0 on Serial (the bit-identity oracle). Non-intrusive: snapshots the KK
 * result (raw alias, host-current after the driver per-kernel sync_host), runs
 * the C twin (overwriting the raw alias — incl. its own internal halo, a no-op at
 * np=1), diffs, restores KK. No L26 capture-before is needed for ANY substep-1b
 * kernel — every output is a FULL overwrite derived from inputs the kernel does
 * not modify, so the C twin recomputes from the same intact state (EOS-style gate).
 * The C twins are run in the SAME chain order as the kernels (each reads the
 * KK-produced upstream state, which is bit-identical to the C twin's), so verifying
 * each kernel independently is sound.
 *===========================================================================*/

static double fesom_gm_maxdiff_(const std::vector<real_t> &kk, const real_t *c, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)kk[i] - (double)c[i]);
        if (d > m) m = d;
    }
    return m;
}

static void fesom_gm_verify_report_(int step_n, const char *what, double dmax)
{
    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=gm] FAIL (%s) step %d: Serial must be bit-identical "
                             "to the C twin (max|Δ|=%.3e)\n", what, step_n, dmax);
        std::abort();
    }
}

void fesom_gm_sigma_xy_verify(struct fesom_aux *aux, const struct fesom_tracers *tracers,
                              const struct fesom_mesh *mesh, fesom_gm *gm,
                              struct fesom_partit *partit, int step_n)
{
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl * 2;
    std::vector<real_t> kk(gm->sigma_xy, gm->sigma_xy + total);
    fesom_compute_sigma_xy(aux, tracers, mesh, gm, partit);             /* C twin */
    double d = fesom_gm_maxdiff_(kk, gm->sigma_xy, total);
    std::copy(kk.begin(), kk.end(), gm->sigma_xy);                      /* restore KK */
    std::printf("[FESOM_KK_VERIFY=gm] step %d backend=%s  max|Δ|: sigma_xy=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(), d);
    fesom_gm_verify_report_(step_n, "sigma_xy", d);
}

void fesom_gm_neutral_slope_verify(struct fesom_aux *aux, const struct fesom_mesh *mesh,
                                   fesom_gm *gm, struct fesom_partit *partit, int step_n)
{
    const int nl = mesh->nl;
    const int nl1 = nl - 1;
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const size_t t3 = (size_t)N * (size_t)nl1 * 3;
    const size_t tnl = (size_t)N * (size_t)nl;
    std::vector<real_t> kk_ns(gm->neutral_slope, gm->neutral_slope + t3);
    std::vector<real_t> kk_st(gm->slope_tapered, gm->slope_tapered + t3);
    std::vector<real_t> kk_tf(gm->fer_tapfac,    gm->fer_tapfac    + tnl);
    fesom_compute_neutral_slope(aux, mesh, gm, partit);                 /* C twin */
    double d_ns = fesom_gm_maxdiff_(kk_ns, gm->neutral_slope, t3);
    double d_st = fesom_gm_maxdiff_(kk_st, gm->slope_tapered, t3);
    double d_tf = fesom_gm_maxdiff_(kk_tf, gm->fer_tapfac,    tnl);
    std::copy(kk_ns.begin(), kk_ns.end(), gm->neutral_slope);           /* restore KK */
    std::copy(kk_st.begin(), kk_st.end(), gm->slope_tapered);
    std::copy(kk_tf.begin(), kk_tf.end(), gm->fer_tapfac);
    std::printf("[FESOM_KK_VERIFY=gm] step %d backend=%s  max|Δ|: neutral_slope=%.3e "
                "slope_tapered=%.3e fer_tapfac=%.3e\n", step_n,
                std::string(Kokkos::DefaultExecutionSpace::name()).c_str(), d_ns, d_st, d_tf);
    fesom_gm_verify_report_(step_n, "neutral_slope", std::max(d_ns, std::max(d_st, d_tf)));
}

void fesom_gm_init_redi_verify(struct fesom_aux *aux, const struct fesom_mesh *mesh,
                               fesom_gm *gm, struct fesom_partit *partit, int step_n)
{
    const int nl = mesh->nl;
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const size_t tnl = (size_t)N * (size_t)nl;
    const size_t tn  = (size_t)N;
    std::vector<real_t> kk_scal(gm->fer_scal, gm->fer_scal + tn);
    std::vector<real_t> kk_K(gm->fer_K, gm->fer_K + tnl);
    std::vector<real_t> kk_C(gm->fer_C, gm->fer_C + tn);
    std::vector<real_t> kk_Ki(gm->Ki, gm->Ki + tnl);
    fesom_init_redi_gm(aux, mesh, gm, partit);                          /* C twin */
    double d_scal = fesom_gm_maxdiff_(kk_scal, gm->fer_scal, tn);
    double d_K    = fesom_gm_maxdiff_(kk_K,    gm->fer_K,    tnl);
    double d_C    = fesom_gm_maxdiff_(kk_C,    gm->fer_C,    tn);
    double d_Ki   = fesom_gm_maxdiff_(kk_Ki,   gm->Ki,       tnl);
    std::copy(kk_scal.begin(), kk_scal.end(), gm->fer_scal);            /* restore KK */
    std::copy(kk_K.begin(),    kk_K.end(),    gm->fer_K);
    std::copy(kk_C.begin(),    kk_C.end(),    gm->fer_C);
    std::copy(kk_Ki.begin(),   kk_Ki.end(),   gm->Ki);
    std::printf("[FESOM_KK_VERIFY=gm] step %d backend=%s  max|Δ|: fer_scal=%.3e fer_K=%.3e "
                "fer_C=%.3e Ki=%.3e\n", step_n,
                std::string(Kokkos::DefaultExecutionSpace::name()).c_str(), d_scal, d_K, d_C, d_Ki);
    fesom_gm_verify_report_(step_n, "init_redi_gm",
                            std::max(std::max(d_scal, d_K), std::max(d_C, d_Ki)));
}

void fesom_gm_solve_gamma_verify(const struct fesom_aux *aux, const struct fesom_mesh *mesh,
                                 fesom_gm *gm, struct fesom_partit *partit, int step_n)
{
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl * 2;
    std::vector<real_t> kk(gm->fer_gamma, gm->fer_gamma + total);
    fesom_fer_solve_gamma(aux, mesh, gm, partit);                       /* C twin */
    double d = fesom_gm_maxdiff_(kk, gm->fer_gamma, total);
    std::copy(kk.begin(), kk.end(), gm->fer_gamma);                     /* restore KK */
    std::printf("[FESOM_KK_VERIFY=gm] step %d backend=%s  max|Δ|: fer_gamma=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(), d);
    fesom_gm_verify_report_(step_n, "fer_solve_gamma", d);
}

void fesom_gm_gamma2vel_verify(struct fesom_dyn *dyn, const struct fesom_mesh *mesh,
                               const fesom_gm *gm, struct fesom_partit *partit, int step_n)
{
    const int nl = mesh->nl;
    /* fer_gamma2vel writes OWNED elements [0, myDim_elem2D); the halo is exchange-
     * filled (no-op at np=1). Diff the owned compute extent. */
    const size_t total = (size_t)mesh->myDim_elem2D * (size_t)nl * 2;
    std::vector<real_t> kk(dyn->fer_uv, dyn->fer_uv + total);
    fesom_fer_gamma2vel(dyn, mesh, gm, partit);                         /* C twin */
    double d = fesom_gm_maxdiff_(kk, dyn->fer_uv, total);
    std::copy(kk.begin(), kk.end(), dyn->fer_uv);                       /* restore KK */
    std::printf("[FESOM_KK_VERIFY=gm] step %d backend=%s  max|Δ|: fer_uv=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(), d);
    fesom_gm_verify_report_(step_n, "fer_gamma2vel", d);
}

/*===========================================================================
 * M2.5b-c — DEVICE (Kokkos) twins of the substep-13 Redi diffusion. Run on the
 * device between the HOST FCT advection calls (a device island with a host
 * round-trip on `values`, the M2.2/§5 accepted pattern); the bolus add/sub
 * (13a/13c) STAY ON HOST — the bolus-augmented uv/w/w_e have NO device consumer
 * until M2.6 ports the FCT (verified: FCT/tracer-diff only READ them), so a
 * device bolus would be a pure device→host round-trip; it moves with the FCT.
 *
 * Each Redi kernel OWNS its internal halo (D21 bracket — like KPP/momentum_adv,
 * because the per-node/edge compute reads the just-exchanged scratch at HALO
 * entries): diff_ver exchanges `tr_xy`, diff_hor exchanges `tr_z`. diff_ver adds
 * to `values` per-node (each node its OWN column → race-free map); diff_hor is an
 * edge→node SCATTER (`atomic_add`, D22 — Serial bit-identical, OpenMP/CUDA
 * climate-close). `values` is read-modify-write (+=) → the verify is the L26
 * capture-before. `tr_xy` flows diff_ver→diff_hor on the device (the D21 bracket
 * left it device-current incl. halo). ⚠️ honours feedback_bolus_divergence_balance
 * by copying the C verbatim (no per-cell clamp anywhere).
 *===========================================================================*/

/*--- diff_ver_part_redi_expl — DEVICE (substep 13) --------------------------
 * Vertical-explicit Redi: tr_xy=∇_h(valuesold) per element (+ internal halo),
 * then per-node area-weighted gather + vd_flux divergence into `values`. */
void fesom_diff_ver_part_redi_expl_kk(int                      tr_idx,
                                      fesom_gm                *gm,
                                      const struct fesom_mesh *mesh,
                                      struct fesom_tracers    *tracers,
                                      struct fesom_partit     *partit)
{
    if (!gm) return;
    const int    nl     = mesh->nl;
    const int    nl1    = nl - 1;
    const int    myDim  = mesh->myDim_nod2D;
    const int    myDim_e= mesh->myDim_elem2D;
    const int    Etot   = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    const real_t dt     = (real_t)FESOM_PHASE1_DT;
    FESOM_CHECK(nl <= NL_MAX, "diff_ver_part_redi_expl_kk: nl %d > NL_MAX", nl);

    auto valsold  = tracers->data[tr_idx].valuesold_fld.d();
    auto vals     = tracers->data[tr_idx].values_fld.d();
    auto txy      = gm->tr_xy_fld.d();
    auto st       = gm->slope_tapered_fld.d();
    auto Ki       = gm->Ki_fld.d();
    auto elnod    = mesh->elem_nodes_fld.d();
    auto grad     = mesh->gradient_sca_fld.d();
    auto nlev_e   = mesh->nlevels_fld.d();
    auto ulev_e   = mesh->ulevels_fld.d();
    auto nlev_n   = mesh->nlevels_nod2D_fld.d();
    auto ulev_n   = mesh->ulevels_nod2D_fld.d();
    auto off      = mesh->nod_in_elem2D_offsets_fld.d();
    auto nie      = mesh->nod_in_elem2D_fld.d();
    auto earea    = mesh->elem_area_fld.d();
    auto areasvol = mesh->areasvol_fld.d();
    auto hnode    = mesh->hnode_fld.d();
    auto hnode_new= mesh->hnode_new_fld.d();
    auto zbar     = mesh->zbar_fld.d();
    auto area     = mesh->area_fld.d();

    /* Step 1a: zero tr_xy over the full extent (device twin of the memset). */
    Kokkos::parallel_for("fesom_gm_redi_ver_zero", Kokkos::RangePolicy<>(0, Etot * nl1 * 2),
        KOKKOS_LAMBDA(const int i) { txy(i) = 0.0; });

    /* Step 1b: tr_xy = ∇_h(valuesold) per owned element. */
    Kokkos::parallel_for("fesom_gm_redi_ver_trxy", Kokkos::RangePolicy<>(0, myDim_e),
        KOKKOS_LAMBDA(const int el) {
            int v0 = elnod(3*el + 0), v1 = elnod(3*el + 1), v2 = elnod(3*el + 2);
            real_t g0 = grad(6*el+0), g1 = grad(6*el+1), g2 = grad(6*el+2);
            real_t g3 = grad(6*el+3), g4 = grad(6*el+4), g5 = grad(6*el+5);
            int nle = nlev_e(el) - 1, ule = ulev_e(el) - 1;
            for (int nz = ule; nz < nle; ++nz) {
                real_t T0 = valsold((size_t)v0 * nl + nz);
                real_t T1 = valsold((size_t)v1 * nl + nz);
                real_t T2 = valsold((size_t)v2 * nl + nz);
                txy((size_t)el * nl1 * 2 + nz * 2 + 0) = g0*T0 + g1*T1 + g2*T2;
                txy((size_t)el * nl1 * 2 + nz * 2 + 1) = g3*T0 + g4*T1 + g5*T2;
            }
        });
    /* Internal halo (D21): exchange_elem(tr_xy) — full element halo, both comps.
     * The per-node gather below reads tr_xy at the surrounding elements, which can
     * be HALO elements. M5.1: GPU-aware-MPI on-device under CUDA, host-staged else. */
    fesom_halo_field(gm->tr_xy_fld, FESOM_HALO_ELEM2D_FULL, nl1, 2, partit);

    /* Step 2: per-node Redi vertical-explicit flux → += values (each node owns
     * its column → race-free map). */
    Kokkos::parallel_for("fesom_gm_redi_ver_node", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            real_t txn[NL_MAX], tyn[NL_MAX], zbar_n[NL_MAX], z_n[NL_MAX], vd_flux[NL_MAX];
            int nle = nlev_n(n) - 1, ule = ulev_n(n) - 1;
            if (nle <= ule) return;

            for (int nz = 0; nz < nl1; ++nz) { txn[nz] = 0.0; tyn[nz] = 0.0; }
            for (int nz = ule; nz < nle; ++nz) {
                real_t Tx = 0.0, Ty = 0.0;
                int o0 = off(n), o1 = off(n + 1);
                for (int k = o0; k < o1; ++k) {
                    int el = nie(k);
                    int el_nle = nlev_e(el) - 1, el_ule = ulev_e(el) - 1;
                    if (nz >= el_ule && nz < el_nle) {
                        real_t a = earea(el);
                        Tx += txy((size_t)el * nl1 * 2 + nz * 2 + 0) * a;
                        Ty += txy((size_t)el * nl1 * 2 + nz * 2 + 1) * a;
                    }
                }
                real_t inv = 1.0 / (3.0 * areasvol((size_t)n * nl + nz));
                txn[nz] = Tx * inv;
                tyn[nz] = Ty * inv;
            }

            for (int nz = 0; nz < nl; ++nz) zbar_n[nz] = 0.0;
            for (int nz = 0; nz < nl1; ++nz) z_n[nz] = 0.0;
            zbar_n[nle] = zbar(nle);
            z_n[nle - 1] = zbar_n[nle] + hnode((size_t)n * nl + (nle - 1)) * 0.5;
            for (int nz = nle - 1; nz >= ule + 1; --nz) {
                zbar_n[nz]  = zbar_n[nz + 1] + hnode((size_t)n * nl + nz);
                z_n[nz - 1] = zbar_n[nz]     + hnode((size_t)n * nl + (nz - 1)) * 0.5;
            }
            zbar_n[ule] = zbar_n[ule + 1] + hnode((size_t)n * nl + ule);

            for (int nz = 0; nz < nl; ++nz) vd_flux[nz] = 0.0;
            for (int nz = ule + 1; nz < nle; ++nz) {
                real_t st_x_up = st((size_t)n * nl1 * 3 + (nz - 1) * 3 + 0);
                real_t st_y_up = st((size_t)n * nl1 * 3 + (nz - 1) * 3 + 1);
                real_t st_x_dn = st((size_t)n * nl1 * 3 + nz       * 3 + 0);
                real_t st_y_dn = st((size_t)n * nl1 * 3 + nz       * 3 + 1);
                real_t Ki_up   = Ki((size_t)n * nl + (nz - 1));
                real_t Ki_dn   = Ki((size_t)n * nl + nz);
                real_t up = (z_n[nz - 1] - zbar_n[nz])
                          * (st_x_up * txn[nz - 1] + st_y_up * tyn[nz - 1]) * Ki_up;
                real_t dn = (zbar_n[nz] - z_n[nz])
                          * (st_x_dn * txn[nz]     + st_y_dn * tyn[nz])     * Ki_dn;
                real_t mid = z_n[nz - 1] - z_n[nz];
                real_t a_int = area((size_t)n * nl + nz);
                vd_flux[nz] = (up + dn) / mid * a_int;
            }

            for (int nz = ule; nz < nle; ++nz) {
                real_t av = areasvol((size_t)n * nl + nz);
                real_t hn = hnode_new((size_t)n * nl + nz);
                if (av > 0.0 && hn > 0.0) {
                    vals((size_t)n * nl + nz) +=
                        (vd_flux[nz] - vd_flux[nz + 1]) * dt / (av * hn);
                }
            }
        });

    tracers->data[tr_idx].values_fld.modify_device();
}

/*--- diff_part_hor_redi — DEVICE (substep 13) -------------------------------
 * Horizontal Redi flux on edges: builds tr_z per node (+ internal halo), then an
 * edge→node SCATTER (atomic_add, D22) with 5 partial-cell branches A–E. Reuses
 * tr_xy built by diff_ver_part_redi_expl_kk (device-current). */
void fesom_diff_part_hor_redi_kk(int                      tr_idx,
                                 fesom_gm                *gm,
                                 const struct fesom_mesh *mesh,
                                 struct fesom_tracers    *tracers,
                                 struct fesom_partit     *partit)
{
    if (!gm) return;
    const int    nl     = mesh->nl;
    const int    nl1    = nl - 1;
    const int    Nfull  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    myDim  = mesh->myDim_nod2D;
    const int    EG     = mesh->myDim_edge2D;
    const real_t dt     = (real_t)FESOM_PHASE1_DT;
    const real_t isredi = 1.0;
    FESOM_CHECK(nl <= NL_MAX, "diff_part_hor_redi_kk: nl %d > NL_MAX", nl);

    auto valsold  = tracers->data[tr_idx].valuesold_fld.d();
    auto vals     = tracers->data[tr_idx].values_fld.d();
    auto txy      = gm->tr_xy_fld.d();
    auto trz      = gm->tr_z_fld.d();
    auto st       = gm->slope_tapered_fld.d();
    auto Ki       = gm->Ki_fld.d();
    auto edges    = mesh->edges_fld.d();
    auto edge_tri = mesh->edge_tri_fld.d();
    auto edge_cross = mesh->edge_cross_dxdy_fld.d();
    auto helem    = mesh->helem_fld.d();
    auto hnode    = mesh->hnode_fld.d();
    auto hnode_new= mesh->hnode_new_fld.d();
    auto areasvol = mesh->areasvol_fld.d();
    auto nlev_e   = mesh->nlevels_fld.d();
    auto ulev_e   = mesh->ulevels_fld.d();
    auto nlev_n   = mesh->nlevels_nod2D_fld.d();
    auto ulev_n   = mesh->ulevels_nod2D_fld.d();

    /* Step 1a: zero tr_z (full local extent). */
    Kokkos::parallel_for("fesom_gm_redi_hor_zero", Kokkos::RangePolicy<>(0, Nfull * nl),
        KOKKOS_LAMBDA(const int i) { trz(i) = 0.0; });

    /* Step 1b: tr_z = ∂ valuesold/∂z at interfaces (per owned node). */
    Kokkos::parallel_for("fesom_gm_redi_hor_trz", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            int nzmax = nlev_n(n) - 1, nzmin = ulev_n(n) - 1;
            if (nzmax <= nzmin + 1) return;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t T_up = valsold((size_t)n * nl + (nz - 1));
                real_t T_dn = valsold((size_t)n * nl + nz);
                real_t h_up = hnode((size_t)n * nl + (nz - 1));
                real_t h_dn = hnode((size_t)n * nl + nz);
                real_t dz   = 0.5 * (h_up + h_dn);
                trz((size_t)n * nl + nz) = (T_up - T_dn) / dz;
            }
        });
    /* Internal halo (D21): exchange_nod3D(tr_z). The edge loop reads tr_z at the
     * two endpoints, which can be HALO nodes. M5.1: GPU-aware-MPI on-device. */
    fesom_halo_field(gm->tr_z_fld, FESOM_HALO_NOD3D, nl, 1, partit);

    /* Step 2: edge loop → edge→node SCATTER into `values` (atomic_add, D22). */
    Kokkos::parallel_for("fesom_gm_redi_hor_edge", Kokkos::RangePolicy<>(0, EG),
        KOKKOS_LAMBDA(const int ed) {
            int e1 = edges(2*ed + 0);
            int e2 = edges(2*ed + 1);
            int el1 = edge_tri(2*ed + 0);
            int el2 = edge_tri(2*ed + 1);
            if (el1 < 0) return;          /* malformed edge — skip */

            real_t dxL = edge_cross(4*ed + 0);
            real_t dyL = edge_cross(4*ed + 1);
            real_t dxR = 0.0, dyR = 0.0;

            int nl1_e = nlev_e(el1) - 1;
            int ul1_e = ulev_e(el1) - 1;
            int nl2_e = -1;
            int ul2_e = -1;
            if (el2 >= 0) {
                nl2_e = nlev_e(el2) - 1;
                ul2_e = ulev_e(el2) - 1;
                dxR = edge_cross(4*ed + 2);
                dyR = edge_cross(4*ed + 3);
            }

            int nl12 = (el2 >= 0) ? ((nl1_e < nl2_e) ? nl1_e : nl2_e) : nl1_e;
            int ul12 = (el2 >= 0) ? ((ul1_e > ul2_e) ? ul1_e : ul2_e) : ul1_e;

            real_t rhs1[NL_MAX], rhs2[NL_MAX];
            for (int nz = 0; nz < nl1; ++nz) { rhs1[nz] = 0.0; rhs2[nz] = 0.0; }

            #define COMPUTE_KH_TZ_S(NZ)                                                              \
                real_t Kh   = 0.5 * (Ki((size_t)e1 * nl + (NZ)) + Ki((size_t)e2 * nl + (NZ)));       \
                real_t Tz1  = 0.5 * (trz((size_t)e1 * nl + (NZ)) + trz((size_t)e1 * nl + ((NZ)+1))); \
                real_t Tz2  = 0.5 * (trz((size_t)e2 * nl + (NZ)) + trz((size_t)e2 * nl + ((NZ)+1))); \
                real_t st_x_1 = st((size_t)e1 * nl1 * 3 + (NZ) * 3 + 0);                             \
                real_t st_y_1 = st((size_t)e1 * nl1 * 3 + (NZ) * 3 + 1);                             \
                real_t st_x_2 = st((size_t)e2 * nl1 * 3 + (NZ) * 3 + 0);                             \
                real_t st_y_2 = st((size_t)e2 * nl1 * 3 + (NZ) * 3 + 1);                             \
                real_t SxTz = 0.5 * (Tz1 * st_x_1 + Tz2 * st_x_2);                                   \
                real_t SyTz = 0.5 * (Tz1 * st_y_1 + Tz2 * st_y_2);

            /* (A) ul1..ul12-1 — el(1) only */
            for (int nz = ul1_e; nz < ul12; ++nz) {
                COMPUTE_KH_TZ_S(nz)
                real_t dz = helem((size_t)el1 * nl + nz);
                real_t Tx = txy((size_t)el1 * nl1 * 2 + nz * 2 + 0);
                real_t Ty = txy((size_t)el1 * nl1 * 2 + nz * 2 + 1);
                real_t Fx = Kh * (Tx + SxTz * isredi);
                real_t Fy = Kh * (Ty + SyTz * isredi);
                real_t c  = (-dxL * Fy + dyL * Fx) * dz;
                rhs1[nz] += c;
                rhs2[nz] -= c;
            }

            /* (B) ul2..ul12-1 — el(2) only */
            if (el2 >= 0) {
                for (int nz = ul2_e; nz < ul12; ++nz) {
                    COMPUTE_KH_TZ_S(nz)
                    real_t dz = helem((size_t)el2 * nl + nz);
                    real_t Tx = txy((size_t)el2 * nl1 * 2 + nz * 2 + 0);
                    real_t Ty = txy((size_t)el2 * nl1 * 2 + nz * 2 + 1);
                    real_t Fx = Kh * (Tx + SxTz * isredi);
                    real_t Fy = Kh * (Ty + SyTz * isredi);
                    real_t c  = (dxR * Fy - dyR * Fx) * dz;
                    rhs1[nz] += c;
                    rhs2[nz] -= c;
                }
            }

            /* (C) ul12..nl12 — both elements */
            for (int nz = ul12; nz < nl12; ++nz) {
                COMPUTE_KH_TZ_S(nz)
                real_t dz = (el2 >= 0)
                          ? 0.5 * (helem((size_t)el1 * nl + nz) + helem((size_t)el2 * nl + nz))
                          : helem((size_t)el1 * nl + nz);
                real_t Tx, Ty;
                if (el2 >= 0) {
                    Tx = 0.5 * (txy((size_t)el1 * nl1 * 2 + nz * 2 + 0)
                              + txy((size_t)el2 * nl1 * 2 + nz * 2 + 0));
                    Ty = 0.5 * (txy((size_t)el1 * nl1 * 2 + nz * 2 + 1)
                              + txy((size_t)el2 * nl1 * 2 + nz * 2 + 1));
                } else {
                    Tx = txy((size_t)el1 * nl1 * 2 + nz * 2 + 0);
                    Ty = txy((size_t)el1 * nl1 * 2 + nz * 2 + 1);
                }
                real_t Fx = Kh * (Tx + SxTz * isredi);
                real_t Fy = Kh * (Ty + SyTz * isredi);
                real_t c  = ((dxR - dxL) * Fy - (dyR - dyL) * Fx) * dz;
                rhs1[nz] += c;
                rhs2[nz] -= c;
            }

            /* (D) nl12+1..nl1 — el(1) only */
            for (int nz = nl12; nz < nl1_e; ++nz) {
                COMPUTE_KH_TZ_S(nz)
                real_t dz = helem((size_t)el1 * nl + nz);
                real_t Tx = txy((size_t)el1 * nl1 * 2 + nz * 2 + 0);
                real_t Ty = txy((size_t)el1 * nl1 * 2 + nz * 2 + 1);
                real_t Fx = Kh * (Tx + SxTz * isredi);
                real_t Fy = Kh * (Ty + SyTz * isredi);
                real_t c  = (-dxL * Fy + dyL * Fx) * dz;
                rhs1[nz] += c;
                rhs2[nz] -= c;
            }

            /* (E) nl12+1..nl2 — el(2) only */
            if (el2 >= 0) {
                for (int nz = nl12; nz < nl2_e; ++nz) {
                    COMPUTE_KH_TZ_S(nz)
                    real_t dz = helem((size_t)el2 * nl + nz);
                    real_t Tx = txy((size_t)el2 * nl1 * 2 + nz * 2 + 0);
                    real_t Ty = txy((size_t)el2 * nl1 * 2 + nz * 2 + 1);
                    real_t Fx = Kh * (Tx + SxTz * isredi);
                    real_t Fy = Kh * (Ty + SyTz * isredi);
                    real_t c  = (dxR * Fy - dyR * Fx) * dz;
                    rhs1[nz] += c;
                    rhs2[nz] -= c;
                }
            }

            #undef COMPUTE_KH_TZ_S

            /* Apply rhs to `values` at the two endpoints — EDGE→NODE SCATTER. The
             * n2 path subtracts rhs2 (already the negation accumulated above); the
             * atomic_add of the SAME composed value the C `+=` writes → Serial
             * single-thread ordered → bit-identical (D22). */
            int ul_apply = ul1_e;
            if (el2 >= 0 && ul2_e < ul_apply) ul_apply = ul2_e;
            int nl_apply = (el2 >= 0 && nl2_e > nl1_e) ? nl2_e : nl1_e;

            for (int nz = ul_apply; nz < nl_apply; ++nz) {
                real_t hn1 = hnode_new((size_t)e1 * nl + nz);
                real_t av1 = areasvol ((size_t)e1 * nl + nz);
                if (av1 > 0.0 && hn1 > 0.0) {
                    Kokkos::atomic_add(&vals((size_t)e1 * nl + nz), rhs1[nz] * dt / (av1 * hn1));
                }
                real_t hn2 = hnode_new((size_t)e2 * nl + nz);
                real_t av2 = areasvol ((size_t)e2 * nl + nz);
                if (av2 > 0.0 && hn2 > 0.0) {
                    Kokkos::atomic_add(&vals((size_t)e2 * nl + nz), rhs2[nz] * dt / (av2 * hn2));
                }
            }
        });

    tracers->data[tr_idx].values_fld.modify_device();
}

/*--- FESOM_KK_VERIFY=gm gate for the combined Redi (diff_ver + diff_hor) -----
 * `values` is read-modify-write (the Redi += onto the post-FCT field), so this is
 * the L26 capture-before: the DRIVER snapshots the post-FCT `values` (pre-Redi)
 * and passes it here. Restore it, run BOTH C twins in order (diff_ver rebuilds
 * tr_xy, diff_hor reads it), diff vs the KK result, restore KK. Non-intrusive. */
void fesom_gm_redi_verify(int tr_idx, fesom_gm *gm, const struct fesom_aux *aux,
                          const struct fesom_mesh *mesh, struct fesom_tracers *tracers,
                          struct fesom_partit *partit, int step_n,
                          const std::vector<real_t> &pre_redi)
{
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    real_t *vals = tracers->data[tr_idx].values;
    std::vector<real_t> kk(vals, vals + total);                 /* KK Redi result */
    std::copy(pre_redi.begin(), pre_redi.end(), vals);          /* restore post-FCT (pre-Redi) */
    fesom_diff_ver_part_redi_expl(tr_idx, gm, aux, mesh, tracers, partit);   /* C twin */
    fesom_diff_part_hor_redi    (tr_idx, gm, aux, mesh, tracers, partit);    /* C twin */
    double d = fesom_gm_maxdiff_(kk, vals, total);
    std::copy(kk.begin(), kk.end(), vals);                      /* restore KK */
    std::printf("[FESOM_KK_VERIFY=gm] step %d backend=%s  max|Δ|: redi(tr%d) values=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(), tr_idx, d);
    fesom_gm_verify_report_(step_n, "redi", d);
}
