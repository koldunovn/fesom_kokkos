/*
 * Phase 1 simple-upwind tracer advection. Literal port — see header for
 * Fortran source line references.
 *
 * Fortran's upwind flux convention (oce_adv_tra_hor.F90:174-177):
 *
 *     vflux  = (-v*dx + u*dy) * helem               (dx,dy = el → edge_mid)
 *     flux  := -0.5 * [ T(n1) * (vflux + |vflux|)
 *                     + T(n2) * (vflux - |vflux|) ] - flux
 *
 * "- flux" in the writeback means the routine subtracts the previous content
 * (so calling with l_init_zero=true gives a fresh write; l_init_zero=false
 * gives an antidiffusive HO-LO difference). Here we are always called with
 * effective l_init_zero=true (UPW1 path) — we explicitly zero the flux array
 * up-front and the writeback is then `flux := -0.5*(...) - 0`.
 *
 * Sign convention sanity (FRESH_START.md §14.5 — "edge_vflux sign was reversed
 * in a previous port"). The vertical flux formula is:
 *
 *     flux(nz, n) := -0.5 * [ T(nz)   * (W(nz) + |W(nz)|)
 *                           + T(nz-1) * (W(nz) - |W(nz)|) ] * area(nz, n) - flux
 *
 * Note: vertical W is positive UPWARD. So at an interface where W>0 (upward
 * flow), the upwind-source is the cell BELOW the interface (T(nz)) — the
 * (W+|W|) factor doubles to 2W in that case, picking T(nz). Get this wrong
 * and tracer advection runs in reverse. Verified against the Fortran code.
 *
 * Surface vertical flux uses T(nz=top) directly (line 301):
 *     flux(top, n) := -W(top, n) * T(top, n) * area(top, n) - flux
 * Bottom vertical flux is set to 0 (line 306).
 *
 * STORAGE: all node-column arrays use stride `nl` and FESOM_NODE3D indexing —
 * matches mesh->hnode and tracers->del_ttf (allocated as N*nl). The unused
 * slot at level (nl-1) stays 0; this trades a few bytes for consistency.
 */
#include "fesom_tracer_adv.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   // M5.4c: FCT internal halos on device (fesom_halo_field)
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_tracers.h"

#include <Kokkos_Core.hpp>   // M2.6-b: device kernels (parallel_for) + Kokkos:: math + atomic
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>            // M2.6-b: FESOM_KK_VERIFY snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>
#include <cstdio>

void fesom_tracer_adv_init(fesom_tracer_adv_scratch *sc,
                           const struct fesom_mesh  *mesh)
{
    /* M2.6-a: the scratch now holds fesom::Field (DualView) members, so a raw memset
       over the struct is UB (D13/L13). Value-initialise instead — this runs every
       nested Field's default ctor and zeros every POD (incl. `partit`, which the
       driver re-sets after init). */
    *sc = fesom_tracer_adv_scratch{};
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int E = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    int EG = mesh->myDim_edge2D + mesh->eDim_edge2D;
    size_t e_full = (size_t)EG * (size_t)mesh->nl;
    size_t n_full = (size_t)N  * (size_t)mesh->nl;
    size_t e2_full = (size_t)E * (size_t)mesh->nl * 2;
    size_t eud_full = (size_t)mesh->myDim_edge2D * (size_t)mesh->nl * 4;

    /* M2.6-a: each array is OWNED by a fesom::Field; the raw pointer is a non-owning
       alias to field.h() (the M1.2/D12 / M2.3a / M2.5b-a data-layer pattern). Field::alloc
       zero-inits both host and device mirrors (== the calloc semantics here). Counts are
       in ELEMENTS, not bytes (D12). */
    sc->adv_flux_hor_fld.alloc("tradv.adv_flux_hor", e_full);             sc->adv_flux_hor     = sc->adv_flux_hor_fld.h();
    sc->adv_flux_ver_fld.alloc("tradv.adv_flux_ver", n_full);             sc->adv_flux_ver     = sc->adv_flux_ver_fld.h();
    sc->del_ttf_advhoriz_fld.alloc("tradv.del_ttf_advhoriz", n_full);     sc->del_ttf_advhoriz = sc->del_ttf_advhoriz_fld.h();
    sc->del_ttf_advvert_fld.alloc("tradv.del_ttf_advvert", n_full);       sc->del_ttf_advvert  = sc->del_ttf_advvert_fld.h();
    sc->fct_LO_fld.alloc("tradv.fct_LO", n_full);                         sc->fct_LO           = sc->fct_LO_fld.h();
    sc->fct_ttf_min_fld.alloc("tradv.fct_ttf_min", n_full);               sc->fct_ttf_min      = sc->fct_ttf_min_fld.h();
    sc->fct_ttf_max_fld.alloc("tradv.fct_ttf_max", n_full);               sc->fct_ttf_max      = sc->fct_ttf_max_fld.h();
    sc->fct_plus_fld.alloc("tradv.fct_plus", n_full);                     sc->fct_plus         = sc->fct_plus_fld.h();
    sc->fct_minus_fld.alloc("tradv.fct_minus", n_full);                   sc->fct_minus        = sc->fct_minus_fld.h();
    sc->fct_aux_fld.alloc("tradv.fct_aux", e2_full);                      sc->fct_aux          = sc->fct_aux_fld.h();
    sc->tr_xy_fld.alloc("tradv.tr_xy", e2_full);                          sc->tr_xy            = sc->tr_xy_fld.h();
    sc->edge_up_dn_grad_fld.alloc("tradv.edge_up_dn_grad", eud_full);     sc->edge_up_dn_grad  = sc->edge_up_dn_grad_fld.h();
    FESOM_CHECK(sc->adv_flux_hor && sc->adv_flux_ver
             && sc->del_ttf_advhoriz && sc->del_ttf_advvert
             && sc->fct_LO && sc->fct_ttf_min && sc->fct_ttf_max
             && sc->fct_plus && sc->fct_minus && sc->fct_aux
             && sc->tr_xy && sc->edge_up_dn_grad,
             "tracer adv scratch: out of memory");
}

void fesom_tracer_adv_free(fesom_tracer_adv_scratch *sc)
{
    /* *sc = fesom_tracer_adv_scratch{} releases every Field (assigns an empty DualView →
       drops the refcount/frees) and zeros the raw aliases (D13/L13); no per-array free
       (the raw ptrs are non-owning). */
    *sc = fesom_tracer_adv_scratch{};
}

/*--- compute_fct_LO --------------------------------------------------------
 * Mirror of oce_adv_tra_driver.F90:118-236 (FCT branch).
 * Pre: adv_flux_hor / adv_flux_ver hold UPWIND fluxes for the chosen tracer
 *      (i.e., adv_tra_hor_upw1 + adv_tra_ver_upw1 just ran with init_zero=true).
 *
 * Step 1: zero fct_LO, then accumulate horizontal flux divergence per node:
 *           fct_LO[nz, n1] += adv_flux_hor[nz, e]
 *           fct_LO[nz, n2] -= adv_flux_hor[nz, e]
 * Step 2: per-node finalisation:
 *           fct_LO[nz, n] = (T*hnode + (fct_LO + (vflux[nz] - vflux[nz+1]))
 *                                    * dt/areasvol) / hnode_new
 */
void fesom_tracer_compute_fct_LO(fesom_tracer_adv_scratch *sc,
                                 int                       tr_idx,
                                 const struct fesom_mesh  *mesh,
                                 const struct fesom_tracers *tracers)
{
    const int N    = mesh->myDim_nod2D;
    const int E    = mesh->myDim_edge2D;   /* Fortran adv loops are myDim only */
    const int nl   = mesh->nl;
    const real_t dt = (real_t)FESOM_PHASE1_DT;
    const real_t *T = tracers->data[tr_idx].values;

    /* Step 1a: zero fct_LO */
    memset(sc->fct_LO, 0, (size_t)N * (size_t)nl * sizeof(real_t));

    /* Step 1b: accumulate horizontal flux divergence per node.
       Range matches oce_adv_tra_driver.F90:170 — nu12 to nl12. */
    for (int e = 0; e < E; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];

        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;

        int nl12 = (nl1 > nl2) ? nl1 : nl2;
        int nu12 = nu1;
        if (nu2 > nu12) nu12 = nu2;

        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f = sc->adv_flux_hor[FESOM_NODE3D(e, nz, nl)];
            sc->fct_LO[FESOM_NODE3D(n1, nz, nl)] += f;
            sc->fct_LO[FESOM_NODE3D(n2, nz, nl)] -= f;
        }
    }

    /* Step 2: per-node finalisation */
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nu1; nz < nl1; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            real_t a = mesh->areasvol[k];
            real_t hnod_old = mesh->hnode    [k];
            real_t hnod_new = mesh->hnode_new[k];
            if (hnod_new <= 0.0 || a <= 0.0) {
                sc->fct_LO[k] = 0.0;
                continue;
            }
            real_t f_top = sc->adv_flux_ver[FESOM_NODE3D(n, nz,     nl)];
            real_t f_bot = sc->adv_flux_ver[FESOM_NODE3D(n, nz + 1, nl)];
            real_t numer = T[k] * hnod_old
                         + (sc->fct_LO[k] + (f_top - f_bot)) * dt / a;
            sc->fct_LO[k] = numer / hnod_new;
        }
    }
}

/*--- init_tracers_AB (oce_tracer_mod.F90:13-145, AB2 path) ----------------
 * All loops cover myDim_nod2D + eDim_nod2D (literal Fortran port).
 * Using myDim alone leaves halo entries of valuesAB / valuesold stale, which
 * makes the high-order horizontal advection read wrong tracer values on
 * halo endpoints of boundary edges → partition-dependent tracer drift. */
static void init_tracers_AB_one(int tr_idx,
                                const struct fesom_mesh *mesh,
                                struct fesom_tracers    *tracers,
                                fesom_tracer_adv_scratch *sc)
{
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int nl = mesh->nl;
    /* AB2 stabilization offset (same as momentum, fesom_momentum.c): Fortran
       o_PARAM epsilon=0.1 (oce_modules.F90:92), used in the tracer AB2 at
       oce_tracer_mod.F90:53 as -(0.5+ε)*old+(1.5+ε)*new. The prior 1e-9 was the
       same mis-port corrected in the momentum RHS. */
    const real_t eps = 0.1;
    const real_t c_old = -(0.5 + eps);
    const real_t c_new =  (1.5 + eps);

    /* Zero del_ttf and the partial accumulators (lines 32-39). */
    size_t n_full = (size_t)N * (size_t)nl * sizeof(real_t);
    memset(tracers->del_ttf,        0, n_full);
    memset(sc->del_ttf_advhoriz,    0, n_full);
    memset(sc->del_ttf_advvert,     0, n_full);

    /* AB2 valuesAB = -(0.5+ε)*valuesold + (1.5+ε)*values  (lines 47-53). */
    real_t *vals    = tracers->data[tr_idx].values;
    real_t *valsAB  = tracers->data[tr_idx].valuesAB;
    real_t *valsold = tracers->data[tr_idx].valuesold;
    for (int n = 0; n < N; ++n) {
        for (int nz = 0; nz < nl; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            valsAB[k] = c_old * valsold[k] + c_new * vals[k];
        }
    }

    /* Save valuesold = values (lines 100-102, AB2). Runs after the valuesAB
       computation (Fortran sequence at lines 94-107). */
    memcpy(valsold, vals, n_full);
}

/*--- adv_tra_hor_upw1 (oce_adv_tra_hor.F90:64-257) ------------------------*/
static void adv_tra_hor_upw1(const struct fesom_mesh *mesh,
                             const struct fesom_dyn  *dyn,
                             const real_t            *ttf,
                             real_t                  *flux)
{
    const int E    = mesh->myDim_edge2D;   /* Fortran adv loops are myDim only */
    const int nl   = mesh->nl;

    /* Zero the flux array (l_init_zero=.true.) — lines 91-107 */
    memset(flux, 0, (size_t)E * (size_t)nl * sizeof(real_t));

    for (int e = 0; e < E; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];

        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;

        real_t dx1 = (el1 >= 0) ? mesh->edge_cross_dxdy[4*e + 0] : 0.0;
        real_t dy1 = (el1 >= 0) ? mesh->edge_cross_dxdy[4*e + 1] : 0.0;
        real_t dx2 = (el2 >= 0) ? mesh->edge_cross_dxdy[4*e + 2] : 0.0;
        real_t dy2 = (el2 >= 0) ? mesh->edge_cross_dxdy[4*e + 3] : 0.0;

        int nl12 = (el2 >= 0) ? ((nl1 < nl2) ? nl1 : nl2) : 0;
        int nu12 = nu1;
        if (nu2 > nu12) nu12 = nu2;

        /* (A) el1-only zone above shared layers (Fortran lines 167-178) */
        for (int nz = nu1; nz < nu12; ++nz) {
            real_t u  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
            real_t v  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
            real_t h  = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
            real_t vflux = (-v*dx1 + u*dy1) * h;
            real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
            real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
            flux[FESOM_NODE3D(e, nz, nl)] = -0.5 * (T1*(vflux + fabs(vflux))
                                                  + T2*(vflux - fabs(vflux)));
        }

        /* (B) el2-only zone above shared layers (lines 185-199) */
        if (el2 >= 0) {
            for (int nz = nu2; nz < nu12; ++nz) {
                real_t u  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t v  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t h  = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
                real_t vflux = (v*dx2 - u*dy2) * h;
                real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
                real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
                flux[FESOM_NODE3D(e, nz, nl)] = -0.5 * (T1*(vflux + fabs(vflux))
                                                      + T2*(vflux - fabs(vflux)));
            }
        }

        /* (C) Both segments (lines 207-217) */
        for (int nz = nu12; nz < nl12; ++nz) {
            real_t u1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
            real_t v1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
            real_t h1 = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
            real_t u2 = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
            real_t v2 = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
            real_t h2 = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
            real_t vflux = (-v1*dx1 + u1*dy1) * h1
                         + ( v2*dx2 - u2*dy2) * h2;
            real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
            real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
            flux[FESOM_NODE3D(e, nz, nl)] = -0.5 * (T1*(vflux + fabs(vflux))
                                                  + T2*(vflux - fabs(vflux)));
        }

        /* (D) el1-only remaining (lines 223-233) */
        for (int nz = nl12; nz < nl1; ++nz) {
            real_t u  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
            real_t v  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
            real_t h  = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
            real_t vflux = (-v*dx1 + u*dy1) * h;
            real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
            real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
            flux[FESOM_NODE3D(e, nz, nl)] = -0.5 * (T1*(vflux + fabs(vflux))
                                                  + T2*(vflux - fabs(vflux)));
        }

        /* (E) el2-only remaining (lines 239-248) */
        if (el2 >= 0) {
            for (int nz = nl12; nz < nl2; ++nz) {
                real_t u  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t v  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t h  = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
                real_t vflux = (v*dx2 - u*dy2) * h;
                real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
                real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
                flux[FESOM_NODE3D(e, nz, nl)] = -0.5 * (T1*(vflux + fabs(vflux))
                                                      + T2*(vflux - fabs(vflux)));
            }
        }
    }
}

/*--- adv_tra_hor_central (interim Phase 2 high-order) --------------------
 * Central-difference horizontal flux. Two-sided face value Tmean = 0.5(T1+T2).
 * Equivalent to FESOM's MFCT (oce_adv_tra_hor.F90:546-834) with the
 * edge_up_dn_grad correction zeroed out — i.e., zero-gradient interpretation.
 *
 * Phase 2 interim choice: full MFCT pulls in tracer_gradient_elements +
 * fill_up_dn_grad + a per-edge upwind/downwind triangle index that we
 * haven't ported yet. Central is 2nd-order vs MFCT's 3rd-order with
 * upwind blend, but it is a valid HO scheme for FCT — the antidiffusive
 * flux (HO − LO) it produces is correctly limited by Zalesak.
 *
 * When init_zero is FALSE, this writes (HO − flux_existing); for FCT
 * use, callers run upwind with init_zero=true first, then central with
 * init_zero=false to obtain the antidiffusive horizontal flux.
 */
/* Retained on purpose (currently unused): the 2nd-order central HO flux is kept
 * available as an alternate FCT high-order scheme now that MFCT is the default.
 * Do NOT remove. __attribute__((unused)) silences the no-caller warning. */
__attribute__((unused))
static void adv_tra_hor_central(const struct fesom_mesh *mesh,
                                const struct fesom_dyn  *dyn,
                                const real_t            *ttf,
                                int                      init_zero,
                                real_t                  *flux)
{
    const int E    = mesh->myDim_edge2D;   /* Fortran adv loops are myDim only */
    const int nl   = mesh->nl;

    if (init_zero) {
        memset(flux, 0, (size_t)E * (size_t)nl * sizeof(real_t));
    }

    for (int e = 0; e < E; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];

        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;

        real_t dx1 = (el1 >= 0) ? mesh->edge_cross_dxdy[4*e + 0] : 0.0;
        real_t dy1 = (el1 >= 0) ? mesh->edge_cross_dxdy[4*e + 1] : 0.0;
        real_t dx2 = (el2 >= 0) ? mesh->edge_cross_dxdy[4*e + 2] : 0.0;
        real_t dy2 = (el2 >= 0) ? mesh->edge_cross_dxdy[4*e + 3] : 0.0;

        int nl12 = (el2 >= 0) ? ((nl1 < nl2) ? nl1 : nl2) : 0;
        int nu12 = (nu1 > nu2) ? nu1 : nu2;

        /* Same 5-zone structure as upwind (oce_adv_tra_hor.F90:167-249).
         * Only the per-layer flux formula differs:
         *   upwind   : flux = -0.5 [T1·(v+|v|) + T2·(v-|v|)]
         *   central  : flux = -v · 0.5(T1+T2)
         * The "− flux" writeback for init_zero=false yields the antidiffusive
         * flux when called after the upwind pass. */

        /* (A) el1-only zone above shared layers */
        for (int nz = nu1; nz < nu12; ++nz) {
            real_t u  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
            real_t v  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
            real_t h  = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
            real_t vflux = (-v*dx1 + u*dy1) * h;
            real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
            real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
            real_t hi = -vflux * 0.5 * (T1 + T2);
            flux[FESOM_NODE3D(e, nz, nl)] = hi - flux[FESOM_NODE3D(e, nz, nl)];
        }
        /* (B) el2-only zone above shared */
        if (el2 >= 0) {
            for (int nz = nu2; nz < nu12; ++nz) {
                real_t u  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t v  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t h  = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
                real_t vflux = (v*dx2 - u*dy2) * h;
                real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
                real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
                real_t hi = -vflux * 0.5 * (T1 + T2);
                flux[FESOM_NODE3D(e, nz, nl)] = hi - flux[FESOM_NODE3D(e, nz, nl)];
            }
        }
        /* (C) Both segments — interior */
        for (int nz = nu12; nz < nl12; ++nz) {
            real_t u1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
            real_t v1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
            real_t h1 = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
            real_t u2 = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
            real_t v2 = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
            real_t h2 = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
            real_t vflux = (-v1*dx1 + u1*dy1) * h1
                         + ( v2*dx2 - u2*dy2) * h2;
            real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
            real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
            real_t hi = -vflux * 0.5 * (T1 + T2);
            flux[FESOM_NODE3D(e, nz, nl)] = hi - flux[FESOM_NODE3D(e, nz, nl)];
        }
        /* (D) el1-only remaining */
        for (int nz = nl12; nz < nl1; ++nz) {
            real_t u  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
            real_t v  = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
            real_t h  = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
            real_t vflux = (-v*dx1 + u*dy1) * h;
            real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
            real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
            real_t hi = -vflux * 0.5 * (T1 + T2);
            flux[FESOM_NODE3D(e, nz, nl)] = hi - flux[FESOM_NODE3D(e, nz, nl)];
        }
        /* (E) el2-only remaining */
        if (el2 >= 0) {
            for (int nz = nl12; nz < nl2; ++nz) {
                real_t u  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t v  = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t h  = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
                real_t vflux = (v*dx2 - u*dy2) * h;
                real_t T1 = ttf[FESOM_NODE3D(n1, nz, nl)];
                real_t T2 = ttf[FESOM_NODE3D(n2, nz, nl)];
                real_t hi = -vflux * 0.5 * (T1 + T2);
                flux[FESOM_NODE3D(e, nz, nl)] = hi - flux[FESOM_NODE3D(e, nz, nl)];
            }
        }
    }
}

/*===========================================================================
 * MFCT 3rd-order horizontal tracer advection (replaces adv_tra_hor_central).
 *   tracer_gradient_elements (oce_tracer_mod.F90:175-185)
 *   fill_up_dn_grad          (oce_muscl_adv.F90:356-525)
 *   adv_tra_hor_mfct         (oce_adv_tra_hor.F90:546-834)
 *===========================================================================*/

/* tr_xy = gradient_sca·ttf per element per layer; on myDim then wide-exchanged. */
static void tracer_gradient_elements(const struct fesom_mesh *mesh,
                                     const real_t *ttf, real_t *tr_xy,
                                     struct fesom_partit *partit)
{
    const int nl = mesh->nl;
    const int E  = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    memset(tr_xy, 0, (size_t)E * (size_t)nl * 2 * sizeof(real_t));
    for (int el = 0; el < mesh->myDim_elem2D; ++el) {
        int n0 = mesh->elem_nodes[3*el + 0];
        int n1 = mesh->elem_nodes[3*el + 1];
        int n2 = mesh->elem_nodes[3*el + 2];
        const real_t *gs = &mesh->gradient_sca[6*el];
        int nu = mesh->ulevels[el] - 1, nle = mesh->nlevels[el] - 1;
        for (int nz = nu; nz < nle; ++nz) {
            real_t t0 = ttf[FESOM_NODE3D(n0, nz, nl)];
            real_t t1 = ttf[FESOM_NODE3D(n1, nz, nl)];
            real_t t2 = ttf[FESOM_NODE3D(n2, nz, nl)];
            tr_xy[FESOM_ELEMVEC(el, nz, nl) + 0] = gs[0]*t0 + gs[1]*t1 + gs[2]*t2;
            tr_xy[FESOM_ELEMVEC(el, nz, nl) + 1] = gs[3]*t0 + gs[4]*t1 + gs[5]*t2;
        }
    }
    if (partit && partit->npes > 1)
        fesom_halo_exchange(tr_xy, FESOM_HALO_ELEM2D_FULL, nl, 2, partit);
}

/* area-weighted mean of tr_xy over node's surrounding elements that own layer nz,
 * written into g[nz*4+cx], g[nz*4+cy], for nz in [nz0, nz1). (F:448-457 etc.) */
static void node_avg_grad(const struct fesom_mesh *mesh, const real_t *tr_xy,
                          int node, int nz0, int nz1, real_t *g, int cx, int cy)
{
    const int nl = mesh->nl;
    int b0 = mesh->nod_in_elem2D_offsets[node];
    int b1 = mesh->nod_in_elem2D_offsets[node + 1];
    for (int nz = nz0; nz < nz1; ++nz) {
        real_t tvol = 0.0, tx = 0.0, ty = 0.0;
        for (int k = b0; k < b1; ++k) {
            int el = mesh->nod_in_elem2D[k];
            if (nz >= mesh->nlevels[el] - 1 || nz < mesh->ulevels[el] - 1) continue;
            real_t a = mesh->elem_area[el];
            tvol += a;
            tx   += tr_xy[FESOM_ELEMVEC(el, nz, nl) + 0] * a;
            ty   += tr_xy[FESOM_ELEMVEC(el, nz, nl) + 1] * a;
        }
        if (tvol > 0.0) { g[nz*4 + cx] = tx / tvol; g[nz*4 + cy] = ty / tvol; }
    }
}

/* edge_up_dn_grad[4 per (edge,level)]: shared levels use the up/down-triangle
 * gradient; non-shared (cavity/bathy) + boundary edges use node-averaged. */
static void fill_up_dn_grad(const struct fesom_mesh *mesh, const real_t *tr_xy,
                            real_t *eud)
{
    const int nl  = mesh->nl;
    const int Eme = mesh->myDim_edge2D;
    memset(eud, 0, (size_t)Eme * (size_t)nl * 4 * sizeof(real_t));
    for (int ed = 0; ed < Eme; ++ed) {
        int n1 = mesh->edges[2*ed + 0];
        int n2 = mesh->edges[2*ed + 1];
        int up = mesh->edge_up_dn_tri[2*ed + 0];
        int dn = mesh->edge_up_dn_tri[2*ed + 1];
        real_t *g = &eud[(size_t)ed * nl * 4];
        if (up >= 0 && dn >= 0) {
            int a1 = mesh->ulevels_nod2D_max[n1], a2 = mesh->ulevels_nod2D_max[n2];
            int nzmin = ((a1 > a2) ? a1 : a2) - 1;                   /* 0-based */
            int b1 = mesh->nlevels_nod2D_min[n1], b2 = mesh->nlevels_nod2D_min[n2];
            int nzmax = ((b1 < b2) ? b1 : b2) - 1;
            node_avg_grad(mesh, tr_xy, n1, mesh->ulevels_nod2D[n1]-1, nzmin, g, 0, 2);
            node_avg_grad(mesh, tr_xy, n2, mesh->ulevels_nod2D[n2]-1, nzmin, g, 1, 3);
            for (int nz = nzmin; nz < nzmax; ++nz) {
                g[nz*4 + 0] = tr_xy[FESOM_ELEMVEC(up, nz, nl) + 0];
                g[nz*4 + 1] = tr_xy[FESOM_ELEMVEC(dn, nz, nl) + 0];
                g[nz*4 + 2] = tr_xy[FESOM_ELEMVEC(up, nz, nl) + 1];
                g[nz*4 + 3] = tr_xy[FESOM_ELEMVEC(dn, nz, nl) + 1];
            }
            node_avg_grad(mesh, tr_xy, n1, nzmax, mesh->nlevels_nod2D[n1]-1, g, 0, 2);
            node_avg_grad(mesh, tr_xy, n2, nzmax, mesh->nlevels_nod2D[n2]-1, g, 1, 3);
        } else {
            node_avg_grad(mesh, tr_xy, n1, mesh->ulevels_nod2D[n1]-1,
                          mesh->nlevels_nod2D[n1]-1, g, 0, 2);
            node_avg_grad(mesh, tr_xy, n2, mesh->ulevels_nod2D[n2]-1,
                          mesh->nlevels_nod2D[n2]-1, g, 1, 3);
        }
    }
}

/* 3rd-order reconstructed antidiffusive flux at one (edge, level). */
static inline void mfct_edge_flux(const real_t *ttf, real_t *flux, const real_t *eud,
                                  int n1, int n2, int e, int nz, int nl, real_t vflux,
                                  real_t a, real_t edx, real_t edy, real_t RE, real_t num_ord)
{
    real_t T1   = ttf[FESOM_NODE3D(n1, nz, nl)];
    real_t T2   = ttf[FESOM_NODE3D(n2, nz, nl)];
    real_t diff = T2 - T1;
    const real_t *g = &eud[((size_t)e * nl + nz) * 4];
    real_t Tmean1 = T1 + (2.0*diff + edx*a*g[0] + edy*RE*g[2]) / 6.0;
    real_t Tmean2 = T2 - (2.0*diff + edx*a*g[1] + edy*RE*g[3]) / 6.0;
    real_t av  = fabs(vflux);
    real_t cHO = (vflux + av)*Tmean1 + (vflux - av)*Tmean2;
    real_t hi  = -0.5*(1.0 - num_ord)*cHO - vflux*num_ord*0.5*(Tmean1 + Tmean2);
    flux[FESOM_NODE3D(e, nz, nl)] = hi - flux[FESOM_NODE3D(e, nz, nl)];
}

static void adv_tra_hor_mfct(const struct fesom_mesh *mesh, const struct fesom_dyn *dyn,
                             const real_t *ttf, const real_t *eud, real_t num_ord,
                             int init_zero, real_t *flux)
{
    const int E  = mesh->myDim_edge2D;
    const int nl = mesh->nl;
    const real_t RE = (real_t)FESOM_R_EARTH;
    if (init_zero) memset(flux, 0, (size_t)E * (size_t)nl * sizeof(real_t));

    for (int e = 0; e < E; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];
        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;
        real_t dx1 = (el1 >= 0) ? mesh->edge_cross_dxdy[4*e + 0] : 0.0;
        real_t dy1 = (el1 >= 0) ? mesh->edge_cross_dxdy[4*e + 1] : 0.0;
        real_t dx2 = (el2 >= 0) ? mesh->edge_cross_dxdy[4*e + 2] : 0.0;
        real_t dy2 = (el2 >= 0) ? mesh->edge_cross_dxdy[4*e + 3] : 0.0;
        int nl12 = (el2 >= 0) ? ((nl1 < nl2) ? nl1 : nl2) : 0;
        int nu12 = (nu1 > nu2) ? nu1 : nu2;
        real_t edx = mesh->edge_dxdy[2*e + 0];
        real_t edy = mesh->edge_dxdy[2*e + 1];
        real_t a = (el1 >= 0) ? RE * mesh->elem_cos[el1] : 0.0;
        if (el2 >= 0) a = 0.5 * (a + RE * mesh->elem_cos[el2]);

        for (int nz = nu1; nz < nu12; ++nz) {                /* (A) el1-only above */
            real_t u = dyn->uv[FESOM_ELEMVEC(el1,nz,nl)+0], v = dyn->uv[FESOM_ELEMVEC(el1,nz,nl)+1];
            real_t vflux = (-v*dx1 + u*dy1) * mesh->helem[FESOM_ELEM3D(el1,nz,nl)];
            mfct_edge_flux(ttf, flux, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, num_ord);
        }
        if (el2 >= 0) for (int nz = nu2; nz < nu12; ++nz) {  /* (B) el2-only above */
            real_t u = dyn->uv[FESOM_ELEMVEC(el2,nz,nl)+0], v = dyn->uv[FESOM_ELEMVEC(el2,nz,nl)+1];
            real_t vflux = (v*dx2 - u*dy2) * mesh->helem[FESOM_ELEM3D(el2,nz,nl)];
            mfct_edge_flux(ttf, flux, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, num_ord);
        }
        for (int nz = nu12; nz < nl12; ++nz) {               /* (C) both */
            real_t u1 = dyn->uv[FESOM_ELEMVEC(el1,nz,nl)+0], v1 = dyn->uv[FESOM_ELEMVEC(el1,nz,nl)+1];
            real_t u2 = dyn->uv[FESOM_ELEMVEC(el2,nz,nl)+0], v2 = dyn->uv[FESOM_ELEMVEC(el2,nz,nl)+1];
            real_t vflux = (-v1*dx1 + u1*dy1) * mesh->helem[FESOM_ELEM3D(el1,nz,nl)]
                         + ( v2*dx2 - u2*dy2) * mesh->helem[FESOM_ELEM3D(el2,nz,nl)];
            mfct_edge_flux(ttf, flux, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, num_ord);
        }
        for (int nz = nl12; nz < nl1; ++nz) {                /* (D) el1-only below */
            real_t u = dyn->uv[FESOM_ELEMVEC(el1,nz,nl)+0], v = dyn->uv[FESOM_ELEMVEC(el1,nz,nl)+1];
            real_t vflux = (-v*dx1 + u*dy1) * mesh->helem[FESOM_ELEM3D(el1,nz,nl)];
            mfct_edge_flux(ttf, flux, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, num_ord);
        }
        if (el2 >= 0) for (int nz = nl12; nz < nl2; ++nz) {  /* (E) el2-only below */
            real_t u = dyn->uv[FESOM_ELEMVEC(el2,nz,nl)+0], v = dyn->uv[FESOM_ELEMVEC(el2,nz,nl)+1];
            real_t vflux = (v*dx2 - u*dy2) * mesh->helem[FESOM_ELEM3D(el2,nz,nl)];
            mfct_edge_flux(ttf, flux, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, num_ord);
        }
    }
}

/*--- adv_tra_ver_qr4c (oce_adv_tra_ver.F90:332-434) -----------------------
 * Vertical 4th-order quadratic reconstruction with upwind blend.
 *   num_ord ∈ [0, 1]: fraction of pure 4th-order (1.0 = full 4th-order,
 *                     0.0 = full upwind-blend). Pi config sets it to 1.
 *
 * For Phase 1 (no partial cells, no cavity), Z_3d_n[nz, n] = mesh->Z[nz] and
 * zbar_3d_n[nz, n] = mesh->zbar[nz] for all n.
 *
 * Sign convention (cf. FRESH_START.md §14.5): vertical W is positive UPWARD.
 * Surface flux = -ttf·W·area  (W>0 advects ttf out of the surface layer).
 *
 * When init_zero is FALSE, this writes (HO_flux - flux_existing); for FCT
 * use, callers run upwind first with init_zero=true, then QR4C with
 * init_zero=false to obtain the antidiffusive vertical flux.
 */
static void adv_tra_ver_qr4c(const struct fesom_mesh *mesh,
                             const struct fesom_dyn  *dyn,
                             const real_t            *ttf,
                             real_t                   num_ord,
                             int                      init_zero,
                             real_t                  *flux)
{
    const int N  = mesh->myDim_nod2D;
    const int nl = mesh->nl;
    /* Tracer advection uses the EXPLICIT part of w (w_e). When wsplit is
       not triggered (CFL ≤ maxcfl) compute_Wvel_split sets w_e = w. */
    const real_t *W = dyn->w_e;

    if (init_zero) {
        memset(flux, 0, (size_t)N * (size_t)nl * sizeof(real_t));
    }

    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        if (nzmax <= nzmin) continue;

        /* Surface (Fortran nz=nzmin): -ttf·W·area - flux_existing */
        {
            int nz = nzmin;
            size_t k = FESOM_NODE3D(n, nz, nl);
            flux[k] = -ttf[k] * W[k] * mesh->area[k] - flux[k];
        }
        /* 2nd layer (nzmin+1): centred differences. Guard for shallow cols. */
        if (nzmax - nzmin >= 2) {
            int nz = nzmin + 1;
            real_t Tup = ttf[FESOM_NODE3D(n, nz - 1, nl)];
            real_t Tdn = ttf[FESOM_NODE3D(n, nz,     nl)];
            size_t k = FESOM_NODE3D(n, nz, nl);
            flux[k] = -0.5 * (Tup + Tdn) * W[k] * mesh->area[k] - flux[k];
        }
        /* Bottom - 1 (nzmax-1): centred differences. Same as second-layer. */
        if (nzmax - nzmin >= 2) {
            int nz = nzmax - 1;
            real_t Tup = ttf[FESOM_NODE3D(n, nz - 1, nl)];
            real_t Tdn = ttf[FESOM_NODE3D(n, nz,     nl)];
            size_t k = FESOM_NODE3D(n, nz, nl);
            flux[k] = -0.5 * (Tup + Tdn) * W[k] * mesh->area[k] - flux[k];
        }
        /* Bottom (nzmax): zero flux, less existing. */
        {
            int nz = nzmax;
            size_t k = FESOM_NODE3D(n, nz, nl);
            flux[k] = 0.0 - flux[k];
        }
        /* Interior (nzmin+2 .. nzmax-2 inclusive): 4th-order quadratic. */
        for (int nz = nzmin + 2; nz <= nzmax - 2; ++nz) {
            real_t Z_um1 = mesh->Z[nz - 1];
            real_t Z_u   = mesh->Z[nz];
            real_t Z_dn  = mesh->Z[nz + 1];
            real_t Z_um2 = mesh->Z[nz - 2];
            real_t Tum1 = ttf[FESOM_NODE3D(n, nz - 1, nl)];
            real_t Tu   = ttf[FESOM_NODE3D(n, nz,     nl)];
            real_t Tdn  = ttf[FESOM_NODE3D(n, nz + 1, nl)];
            real_t Tum2 = ttf[FESOM_NODE3D(n, nz - 2, nl)];

            real_t qc = (Tum1 - Tu)  / (Z_um1 - Z_u);
            real_t qu = (Tu   - Tdn) / (Z_u   - Z_dn);
            real_t qd = (Tum2 - Tum1)/ (Z_um2 - Z_um1);

            real_t zb = mesh->zbar[nz];
            real_t Tmean1 = Tu  + (2.0*qc + qu) * (zb - Z_u  ) / 3.0;
            real_t Tmean2 = Tum1 + (2.0*qc + qd) * (zb - Z_um1) / 3.0;
            real_t w_iface = W[FESOM_NODE3D(n, nz, nl)];
            real_t Tmean = (w_iface + fabs(w_iface)) * Tmean1
                         + (w_iface - fabs(w_iface)) * Tmean2;
            real_t a = mesh->area[FESOM_NODE3D(n, nz, nl)];
            real_t hi = (-0.5 * (1.0 - num_ord) * Tmean
                       - num_ord * 0.5 * (Tmean1 + Tmean2) * w_iface) * a;
            flux[FESOM_NODE3D(n, nz, nl)] = hi - flux[FESOM_NODE3D(n, nz, nl)];
        }
    }
}

/*--- adv_tra_ver_upw1 (oce_adv_tra_ver.F90:244-328) -----------------------*/
static void adv_tra_ver_upw1(const struct fesom_mesh *mesh,
                             const struct fesom_dyn  *dyn,
                             const real_t            *ttf,
                             real_t                  *flux)
{
    const int N  = mesh->myDim_nod2D;
    const int nl = mesh->nl;
    /* Use explicit-W (w_e). compute_Wvel_split sets w_e = w when CFL ≤ maxcfl. */
    const real_t *W = dyn->w_e;

    memset(flux, 0, (size_t)N * (size_t)nl * sizeof(real_t));

    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        if (nzmax <= nzmin) continue;

        /* Surface flux (line 301) */
        flux[FESOM_NODE3D(n, nzmin, nl)] =
            -W[FESOM_NODE3D(n, nzmin, nl)]
            * ttf[FESOM_NODE3D(n, nzmin, nl)]
            * mesh->area[FESOM_NODE3D(n, nzmin, nl)];

        /* Bottom flux = 0 (line 306) — already zeroed by memset. */

        /* Interior interfaces (lines 315-319) */
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t w_iface = W[FESOM_NODE3D(n, nz, nl)];
            real_t T_below = ttf[FESOM_NODE3D(n, nz,     nl)];   /* layer index nz */
            real_t T_above = ttf[FESOM_NODE3D(n, nz - 1, nl)];   /* layer index nz-1 */
            real_t a       = mesh->area[FESOM_NODE3D(n, nz, nl)];
            flux[FESOM_NODE3D(n, nz, nl)] =
                -0.5 * (T_below * (w_iface + fabs(w_iface))
                      + T_above * (w_iface - fabs(w_iface))) * a;
        }
    }
}

/*--- oce_tra_adv_flux2dtracer (oce_adv_tra_driver.F90:423-575, non-FCT) ---*/
static void flux2dtracer_upwind(const struct fesom_mesh *mesh,
                                const real_t            *flux_h,
                                const real_t            *flux_v,
                                real_t                  *dttf_h,
                                real_t                  *dttf_v)
{
    const int N  = mesh->myDim_nod2D;
    const int E  = mesh->myDim_edge2D;   /* Fortran adv loops are myDim only */
    const int nl = mesh->nl;
    const real_t dt = (real_t)FESOM_PHASE1_DT;

    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nu1; nz < nl1; ++nz) {
            real_t a = mesh->areasvol[FESOM_NODE3D(n, nz, nl)];
            if (a <= 0.0) continue;
            real_t f_top = flux_v[FESOM_NODE3D(n, nz,     nl)];
            real_t f_bot = flux_v[FESOM_NODE3D(n, nz + 1, nl)];
            dttf_v[FESOM_NODE3D(n, nz, nl)] += (f_top - f_bot) * dt / a;
        }
    }

    for (int e = 0; e < E; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];

        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;

        int nl12 = (nl1 > nl2) ? nl1 : nl2;
        int nu12 = nu1;
        if (el2 >= 0 && nu2 > 0 && nu2 < nu12) nu12 = nu2;

        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f  = flux_h[FESOM_NODE3D(e, nz, nl)];
            real_t a1 = mesh->areasvol[FESOM_NODE3D(n1, nz, nl)];
            real_t a2 = mesh->areasvol[FESOM_NODE3D(n2, nz, nl)];
            if (a1 > 0.0) dttf_h[FESOM_NODE3D(n1, nz, nl)] += f * dt / a1;
            if (a2 > 0.0) dttf_h[FESOM_NODE3D(n2, nz, nl)] -= f * dt / a2;
        }
    }
}

/*--- ALE reconstruction (diff_tracers_ale lines 481-484) ------------------
 *   del_ttf += T * (hnode - hnode_new)
 *   T       += del_ttf / hnode_new
 */
static void ale_reconstruct(const struct fesom_mesh *mesh,
                            real_t                  *T,
                            real_t                  *del_ttf)
{
    const int N  = mesh->myDim_nod2D;
    const int nl = mesh->nl;

    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin; nz < nzmax; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            real_t hnode_old = mesh->hnode    [k];
            real_t hnode_new = mesh->hnode_new[k];
            del_ttf[k] += T[k] * (hnode_old - hnode_new);
            if (hnode_new > 0.0) {
                T[k] += del_ttf[k] / hnode_new;
            }
        }
    }
}

/*--- Zalesak limiter (oce_adv_tra_fct.F90:72-500) -------------------------
 * Inputs:
 *   ttf:    tracer at step n (=values BEFORE advection)
 *   lo:     fct_LO (low-order upwind solution, advanced one step)
 *   adf_h:  HO−LO antidiffusive horizontal flux (in/out — in: anti-diff;
 *                                                       out: limited)
 *   adf_v:  HO−LO antidiffusive vertical flux   (in/out)
 *
 * Workspace (in scratch):
 *   fct_ttf_min/max  (writes per-node admissible-increment bounds)
 *   fct_plus/minus   (limiter factors)
 *   fct_aux          (per-element max/min, reused as Fortran AUX(:,:,:))
 *
 * Algorithm:
 *   a1. fct_ttf_max = max(LO, ttf), fct_ttf_min = min(LO, ttf)   per node
 *   a2. AUX[1] = max over 3 vertices of fct_ttf_max,
 *       AUX[2] = min                                              per element
 *   a3. tvert_max = max over surrounding cells of AUX[1],
 *       tvert_min = min                                            per node
 *   a4. fct_ttf_max[surface] = tvert_max[surface] − LO,
 *       fct_ttf_max[middle ] = max over (nz-1, nz, nz+1) − LO,
 *       fct_ttf_max[bottom ] = tvert_max[bottom] − LO              (and min)
 *   b1. fct_plus  = Σ (max(0, +adf_v[nz]) + max(0, −adf_v[nz+1]))  per node
 *                 + Σ over edges of max(0, ±adf_h)
 *       fct_minus = same with min instead of max
 *   b2. flux  = fct_plus  · dt / areasvol / hnode_new + flux_eps
 *       fct_plus  = min(1, fct_ttf_max / flux)
 *       (similar for fct_minus, with −flux_eps)
 *   b3. Limiting (vertical):
 *         surface: ae = (adf_v ≥ 0 ? fct_plus[nz] : fct_minus[nz])
 *         interior: ae = adf_v ≥ 0 ? min(fct_minus[nz-1], fct_plus[nz])
 *                                   : min(fct_plus[nz-1],  fct_minus[nz])
 *         adf_v *= ae
 *       Limiting (horizontal): adf_h ≥ 0 ? min(fct_plus[n1],fct_minus[n2])
 *                                         : min(fct_minus[n1],fct_plus[n2])
 *         adf_h *= ae
 */
static void oce_tra_adv_fct(fesom_tracer_adv_scratch *sc,
                            const struct fesom_mesh  *mesh,
                            const real_t             *ttf,
                            real_t                   *adf_h,
                            real_t                   *adf_v)
{
    const int N    = mesh->myDim_nod2D;
    const int E    = mesh->myDim_elem2D;
    const int Eedg = mesh->myDim_edge2D;   /* Fortran FCT loops are myDim only */
    const int nl   = mesh->nl;
    const real_t dt        = (real_t)FESOM_PHASE1_DT;
    const real_t flux_eps  = 1e-16;
    const real_t bignumber = 1e3;

    real_t *fct_ttf_min = sc->fct_ttf_min;
    real_t *fct_ttf_max = sc->fct_ttf_max;
    real_t *fct_plus    = sc->fct_plus;
    real_t *fct_minus   = sc->fct_minus;
    real_t *AUX         = sc->fct_aux;     /* [E * nl * 2] : (max=0, min=1) */
    const real_t *LO    = sc->fct_LO;

    /* a1. Per-node max/min between LO and ttf.
     * Fortran oce_adv_tra_fct.F90:124 loops myDim+eDim so the halo entries
     * of fct_ttf_{min,max} match the owner — step a2 reads these values at
     * halo vertices of owned elements. Using myDim alone leaves halo entries
     * stale (partition-dependent tracer drift; this is bug 2 behind the
     * init_tracers_AB fix). */
    const int N_full_a1 = mesh->myDim_nod2D + mesh->eDim_nod2D;
    for (int n = 0; n < N_full_a1; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nu1; nz < nl1; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            real_t a = LO[k], b = ttf[k];
            fct_ttf_max[k] = (a > b) ? a : b;
            fct_ttf_min[k] = (a < b) ? a : b;
        }
    }

    /* a2. Per-element max/min over 3 vertices. Pad below cell's nlevels with
       ±bignumber so the per-node tvert step never picks a stale layer. */
    for (int e = 0; e < E; ++e) {
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];
        int nu1 = mesh->ulevels[e] - 1;
        int nle = mesh->nlevels[e] - 1;
        for (int nz = nu1; nz < nle; ++nz) {
            real_t a0 = fct_ttf_max[FESOM_NODE3D(n0, nz, nl)];
            real_t a1 = fct_ttf_max[FESOM_NODE3D(n1, nz, nl)];
            real_t a2 = fct_ttf_max[FESOM_NODE3D(n2, nz, nl)];
            real_t mx = a0; if (a1 > mx) mx = a1; if (a2 > mx) mx = a2;
            real_t b0 = fct_ttf_min[FESOM_NODE3D(n0, nz, nl)];
            real_t b1 = fct_ttf_min[FESOM_NODE3D(n1, nz, nl)];
            real_t b2 = fct_ttf_min[FESOM_NODE3D(n2, nz, nl)];
            real_t mn = b0; if (b1 < mn) mn = b1; if (b2 < mn) mn = b2;
            AUX[FESOM_ELEM3D(e, nz, nl)*2 + 0] = mx;
            AUX[FESOM_ELEM3D(e, nz, nl)*2 + 1] = mn;
        }
        for (int nz = nle; nz < nl - 1; ++nz) {
            AUX[FESOM_ELEM3D(e, nz, nl)*2 + 0] = -bignumber;
            AUX[FESOM_ELEM3D(e, nz, nl)*2 + 1] = +bignumber;
        }
    }

    /* a3. Per-node max/min over surrounding cells (cluster). Use a per-node
       small buffer for tvert_max/min — allocated once, reused per node. */
    real_t *tvert_max = (real_t *)malloc((size_t)N * (size_t)nl * sizeof(real_t));
    real_t *tvert_min = (real_t *)malloc((size_t)N * (size_t)nl * sizeof(real_t));
    FESOM_CHECK(tvert_max && tvert_min, "FCT: oom (tvert)");
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        int o0 = mesh->nod_in_elem2D_offsets[n];
        int o1 = mesh->nod_in_elem2D_offsets[n + 1];
        for (int nz = nu1; nz < nl1; ++nz) {
            int e0 = mesh->nod_in_elem2D[o0];
            real_t mx = AUX[FESOM_ELEM3D(e0, nz, nl)*2 + 0];
            real_t mn = AUX[FESOM_ELEM3D(e0, nz, nl)*2 + 1];
            for (int k = o0 + 1; k < o1; ++k) {
                int e = mesh->nod_in_elem2D[k];
                real_t a = AUX[FESOM_ELEM3D(e, nz, nl)*2 + 0];
                real_t b = AUX[FESOM_ELEM3D(e, nz, nl)*2 + 1];
                if (a > mx) mx = a;
                if (b < mn) mn = b;
            }
            tvert_max[FESOM_NODE3D(n, nz, nl)] = mx;
            tvert_min[FESOM_NODE3D(n, nz, nl)] = mn;
        }
    }

    /* a4. Per-node admissible increment relative to LO.
          Includes vertical 3-layer cluster max/min (nz-1, nz, nz+1). */
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        if (nu1 >= nl1) continue;

        /* Surface */
        {
            int nz = nu1;
            size_t k = FESOM_NODE3D(n, nz, nl);
            fct_ttf_max[k] = tvert_max[k] - LO[k];
            fct_ttf_min[k] = tvert_min[k] - LO[k];
        }
        /* Interior */
        for (int nz = nu1 + 1; nz < nl1 - 1; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            real_t mxA = tvert_max[FESOM_NODE3D(n, nz - 1, nl)];
            real_t mxB = tvert_max[FESOM_NODE3D(n, nz,     nl)];
            real_t mxC = tvert_max[FESOM_NODE3D(n, nz + 1, nl)];
            real_t mnA = tvert_min[FESOM_NODE3D(n, nz - 1, nl)];
            real_t mnB = tvert_min[FESOM_NODE3D(n, nz,     nl)];
            real_t mnC = tvert_min[FESOM_NODE3D(n, nz + 1, nl)];
            real_t mx = mxA; if (mxB > mx) mx = mxB; if (mxC > mx) mx = mxC;
            real_t mn = mnA; if (mnB < mn) mn = mnB; if (mnC < mn) mn = mnC;
            fct_ttf_max[k] = mx - LO[k];
            fct_ttf_min[k] = mn - LO[k];
        }
        /* Bottom layer (nl1 - 1) */
        if (nl1 - 1 > nu1) {
            int nz = nl1 - 1;
            size_t k = FESOM_NODE3D(n, nz, nl);
            fct_ttf_max[k] = tvert_max[k] - LO[k];
            fct_ttf_min[k] = tvert_min[k] - LO[k];
        }
    }
    free(tvert_max);
    free(tvert_min);

    /* b1. Sum positive/negative antidiffusive contributions per node.
       Vertical: from adf_v[nz] (in) and adf_v[nz+1] (out — flipped sign). */
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nu1; nz < nl1; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            fct_plus [k] = 0.0;
            fct_minus[k] = 0.0;
        }
    }
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nu1; nz < nl1; ++nz) {
            real_t fv_top  = adf_v[FESOM_NODE3D(n, nz,     nl)];
            real_t fv_bot  = adf_v[FESOM_NODE3D(n, nz + 1, nl)];
            real_t pos = (fv_top > 0.0 ? fv_top : 0.0)
                       + (-fv_bot > 0.0 ? -fv_bot : 0.0);
            real_t neg = (fv_top < 0.0 ? fv_top : 0.0)
                       + (-fv_bot < 0.0 ? -fv_bot : 0.0);
            size_t k = FESOM_NODE3D(n, nz, nl);
            fct_plus [k] += pos;
            fct_minus[k] += neg;
        }
    }
    /* Horizontal — per edge */
    for (int e = 0; e < Eedg; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];

        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;

        int nl12 = (nl1 > nl2) ? nl1 : nl2;
        int nu12 = nu1;
        if (el2 >= 0 && nu2 < nu12) nu12 = nu2;

        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f = adf_h[FESOM_NODE3D(e, nz, nl)];
            size_t k1 = FESOM_NODE3D(n1, nz, nl);
            size_t k2 = FESOM_NODE3D(n2, nz, nl);
            fct_plus [k1] += (f > 0.0 ? f : 0.0);
            fct_minus[k1] += (f < 0.0 ? f : 0.0);
            fct_plus [k2] += (-f > 0.0 ? -f : 0.0);
            fct_minus[k2] += (-f < 0.0 ? -f : 0.0);
        }
    }

    /* b2. Per-node limiter factors. */
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nu1; nz < nl1; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            real_t a = mesh->areasvol[k];
            real_t hnod_new = mesh->hnode_new[k];
            if (a <= 0.0 || hnod_new <= 0.0) {
                fct_plus[k]  = 1.0;
                fct_minus[k] = 1.0;
                continue;
            }
            real_t flux_pos = fct_plus[k]  * dt / a / hnod_new + flux_eps;
            real_t ratio_pos = fct_ttf_max[k] / flux_pos;
            fct_plus[k] = (ratio_pos < 1.0 ? ratio_pos : 1.0);

            real_t flux_neg = fct_minus[k] * dt / a / hnod_new - flux_eps;
            real_t ratio_neg = fct_ttf_min[k] / flux_neg;
            fct_minus[k] = (ratio_neg < 1.0 ? ratio_neg : 1.0);
        }
    }
    /* MPI: halo exchange of fct_plus / fct_minus so the limiter ratios are
     * consistent across rank boundaries. Mirrors Fortran
     *   call exchange_nod(fct_plus, fct_minus, partit, luse_g2g=.true.)
     * (oce_adv_tra_fct.F90:401). Without this, the per-edge horizontal
     * limiter at b3 picks an inconsistent (fct_plus, fct_minus) pair on
     * each side of a partition boundary, breaking flux limiting and causing
     * tracer overshoot → density gradient → momentum blowup within a few
     * steps. Two single-field calls (no 2-field variant in our halo). */
    if (sc->partit && sc->partit->npes > 1) {
        fesom_exchange_nod3D(fct_plus,  nl, sc->partit);
        fesom_exchange_nod3D(fct_minus, nl, sc->partit);
    }

    /* b3. Apply limits — vertical first, then horizontal. */
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        if (nu1 >= nl1) continue;
        /* surface interface (nz = nu1) */
        {
            int nz = nu1;
            real_t f = adf_v[FESOM_NODE3D(n, nz, nl)];
            size_t k = FESOM_NODE3D(n, nz, nl);
            real_t ae = (f >= 0.0) ? fct_plus[k] : fct_minus[k];
            adf_v[FESOM_NODE3D(n, nz, nl)] = ae * f;
        }
        /* interior interfaces */
        for (int nz = nu1 + 1; nz < nl1; ++nz) {
            real_t f = adf_v[FESOM_NODE3D(n, nz, nl)];
            real_t ae;
            if (f >= 0.0) {
                real_t a = fct_minus[FESOM_NODE3D(n, nz - 1, nl)];
                real_t b = fct_plus [FESOM_NODE3D(n, nz,     nl)];
                ae = (a < b) ? a : b;
            } else {
                real_t a = fct_plus [FESOM_NODE3D(n, nz - 1, nl)];
                real_t b = fct_minus[FESOM_NODE3D(n, nz,     nl)];
                ae = (a < b) ? a : b;
            }
            adf_v[FESOM_NODE3D(n, nz, nl)] = ae * f;
        }
        /* bottom flux is 0 by construction */
    }
    /* Horizontal */
    for (int e = 0; e < Eedg; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];

        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;

        int nl12 = (nl1 > nl2) ? nl1 : nl2;
        int nu12 = nu1;
        if (el2 >= 0 && nu2 < nu12) nu12 = nu2;

        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f = adf_h[FESOM_NODE3D(e, nz, nl)];
            real_t ae;
            if (f >= 0.0) {
                real_t a = fct_plus [FESOM_NODE3D(n1, nz, nl)];
                real_t b = fct_minus[FESOM_NODE3D(n2, nz, nl)];
                ae = (a < b) ? a : b;
            } else {
                real_t a = fct_minus[FESOM_NODE3D(n1, nz, nl)];
                real_t b = fct_plus [FESOM_NODE3D(n2, nz, nl)];
                ae = (a < b) ? a : b;
            }
            adf_h[FESOM_NODE3D(e, nz, nl)] = ae * f;
        }
    }
}

/*--- flux2dtracer with use_lo=true (oce_adv_tra_driver.F90:451-572) -------
 * Adds the LO contribution to dttf_v then accumulates the (now-limited)
 * antidiffusive fluxes into dttf_h and dttf_v.
 *
 *   dttf_v[nz, n] += −ttf·hnode + LO·hnode_new          (LO transition)
 *   dttf_v[nz, n] += (flux_v[nz] − flux_v[nz+1]) · dt / areasvol
 *   dttf_h[nz, n1] += flux_h · dt / areasvol[n1]
 *   dttf_h[nz, n2] −= flux_h · dt / areasvol[n2]
 */
static void flux2dtracer_fct(const struct fesom_mesh *mesh,
                             const real_t            *ttf,
                             const real_t            *lo,
                             const real_t            *flux_h,
                             const real_t            *flux_v,
                             real_t                  *dttf_h,
                             real_t                  *dttf_v)
{
    const int N  = mesh->myDim_nod2D;
    const int E  = mesh->myDim_edge2D;   /* Fortran adv loops are myDim only */
    const int nl = mesh->nl;
    const real_t dt = (real_t)FESOM_PHASE1_DT;

    /* Vertical: LO transition + antidiffusive accumulation. */
    for (int n = 0; n < N; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nu1; nz < nl1; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            dttf_v[k] += -ttf[k] * mesh->hnode    [k]
                       +  lo[k] * mesh->hnode_new[k];
        }
        for (int nz = nu1; nz < nl1; ++nz) {
            real_t a = mesh->areasvol[FESOM_NODE3D(n, nz, nl)];
            if (a <= 0.0) continue;
            real_t f_top = flux_v[FESOM_NODE3D(n, nz,     nl)];
            real_t f_bot = flux_v[FESOM_NODE3D(n, nz + 1, nl)];
            dttf_v[FESOM_NODE3D(n, nz, nl)] += (f_top - f_bot) * dt / a;
        }
    }

    /* Horizontal */
    for (int e = 0; e < E; ++e) {
        int n1  = mesh->edges[2*e + 0];
        int n2  = mesh->edges[2*e + 1];
        int el1 = mesh->edge_tri[2*e + 0];
        int el2 = mesh->edge_tri[2*e + 1];

        int nl1 = (el1 >= 0) ? mesh->nlevels[el1] - 1 : 0;
        int nu1 = (el1 >= 0) ? mesh->ulevels[el1] - 1 : 0;
        int nl2 = (el2 >= 0) ? mesh->nlevels[el2] - 1 : 0;
        int nu2 = (el2 >= 0) ? mesh->ulevels[el2] - 1 : 0;

        int nl12 = (nl1 > nl2) ? nl1 : nl2;
        int nu12 = nu1;
        if (el2 >= 0 && nu2 > 0 && nu2 < nu12) nu12 = nu2;

        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f  = flux_h[FESOM_NODE3D(e, nz, nl)];
            real_t a1 = mesh->areasvol[FESOM_NODE3D(n1, nz, nl)];
            real_t a2 = mesh->areasvol[FESOM_NODE3D(n2, nz, nl)];
            if (a1 > 0.0) dttf_h[FESOM_NODE3D(n1, nz, nl)] += f * dt / a1;
            if (a2 > 0.0) dttf_h[FESOM_NODE3D(n2, nz, nl)] -= f * dt / a2;
        }
    }
}

/*--- Public FCT entry: full FCT step for one tracer ------------------------*/
void fesom_tracer_advect_one_fct(fesom_tracer_adv_scratch *sc,
                                 int                       tr_idx,
                                 const struct fesom_mesh  *mesh,
                                 const struct fesom_dyn   *dyn,
                                 struct fesom_tracers     *tracers)
{
    const int N    = mesh->myDim_nod2D;
    const int nl   = mesh->nl;

    /* 1. init_tracers_AB (zero del_ttf*, fill valuesAB, save valuesold) */
    init_tracers_AB_one(tr_idx, mesh, tracers, sc);

    real_t *vals  = tracers->data[tr_idx].values;
    real_t *ttfAB = tracers->data[tr_idx].valuesAB;

    /* 2. LO upwind fluxes from current `values` (NOT valuesAB).
       Match Fortran do_oce_adv_tra:115 which calls hor_upw1 with `ttf`. */
    adv_tra_hor_upw1(mesh, dyn, vals, sc->adv_flux_hor);
    adv_tra_ver_upw1(mesh, dyn, vals, sc->adv_flux_ver);

    /* 3. fct_LO from upwind (advance one step of upwind solution). */
    fesom_tracer_compute_fct_LO(sc, tr_idx, mesh, tracers);

    /* Halo-exchange fct_LO — Fortran oce_adv_tra_driver.F90:294
     *   call exchange_nod(fct_LO, partit, luse_g2g=.true.)
     * Step 5 (oce_tra_adv_fct) reads LO at halo nodes when computing
     * T_max/T_min for the FCT bounds; without this exchange, halo LO is
     * stale → wrong fct_plus/fct_minus → boundary tracer drift. */
    if (sc->partit && sc->partit->npes > 1) {
        fesom_exchange_nod3D(sc->fct_LO, mesh->nl, sc->partit);
    }

    /* 4. High-order flux on top of LO (init_zero=false → adv_flux := HO − LO).
       Horizontal: MFCT 3rd-order (num_ord=0, CORE2 namelist.tra).
       FIDELITY: the MFCT flux mixes two fields exactly as the Fortran does —
         * node values + central diff  → valuesAB (ttfAB), passed to adv_tra_hor_mfct
           (oce_adv_tra_driver.F90:307 calls it with ttfAB).
         * up/down gradient correction → edge_up_dn_grad, built ONCE in
           init_tracers_AB from `values` (current step), NOT valuesAB — the
           valuesAB variant at oce_tracer_mod.F90:126 is commented out, :127 uses
           values. So tr_xy/edge_up_dn_grad must be built from vals, not ttfAB. */
    tracer_gradient_elements(mesh, vals, sc->tr_xy, sc->partit);
    fill_up_dn_grad         (mesh, sc->tr_xy, sc->edge_up_dn_grad);
    adv_tra_hor_mfct(mesh, dyn, ttfAB, sc->edge_up_dn_grad, /*num_ord=*/0.0,
                     /*init_zero=*/0, sc->adv_flux_hor);
    adv_tra_ver_qr4c   (mesh, dyn, ttfAB, /*num_ord=*/1.0, /*init_zero=*/0, sc->adv_flux_ver);

    /* 5. Zalesak limit the antidiffusive fluxes. */
    oce_tra_adv_fct(sc, mesh, vals, sc->adv_flux_hor, sc->adv_flux_ver);

    /* 6. flux2dtracer with LO (include the LO transition + limited HO). */
    flux2dtracer_fct(mesh, vals, sc->fct_LO,
                     sc->adv_flux_hor, sc->adv_flux_ver,
                     sc->del_ttf_advhoriz, sc->del_ttf_advvert);

    /* 7. del_ttf := del_ttf_advhoriz + del_ttf_advvert */
    for (int n = 0; n < N; ++n) {
        for (int nz = 0; nz < nl; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            tracers->del_ttf[k] = sc->del_ttf_advhoriz[k]
                                + sc->del_ttf_advvert [k];
        }
    }

    /* 8. ALE reconstruction. With FCT use_lo=true the math reduces to
       T_new = LO + limited_antidiff/areasvol/hnode_new — see fesom_tracer_adv.h */
    ale_reconstruct(mesh, vals, tracers->del_ttf);
}

/*--- Public entry: one full upwind step for one tracer --------------------*/
void fesom_tracer_advect_one(fesom_tracer_adv_scratch *sc,
                             int                       tr_idx,
                             const struct fesom_mesh  *mesh,
                             const struct fesom_dyn   *dyn,
                             struct fesom_tracers     *tracers)
{
    const int N    = mesh->myDim_nod2D;
    const int nl   = mesh->nl;

    /* 1. init_tracers_AB */
    init_tracers_AB_one(tr_idx, mesh, tracers, sc);

    /* 2-3. compute upwind fluxes from valuesAB (Fortran calls UPW1 with ttfAB) */
    real_t *ttfAB = tracers->data[tr_idx].valuesAB;
    adv_tra_hor_upw1(mesh, dyn, ttfAB, sc->adv_flux_hor);
    adv_tra_ver_upw1(mesh, dyn, ttfAB, sc->adv_flux_ver);

    /* 4. accumulate fluxes into del_ttf_advhoriz, del_ttf_advvert */
    flux2dtracer_upwind(mesh, sc->adv_flux_hor, sc->adv_flux_ver,
                        sc->del_ttf_advhoriz, sc->del_ttf_advvert);

    /* 5. del_ttf := del_ttf_advhoriz + del_ttf_advvert (lines 239-242) */
    for (int n = 0; n < N; ++n) {
        for (int nz = 0; nz < nl; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            tracers->del_ttf[k] = sc->del_ttf_advhoriz[k]
                                + sc->del_ttf_advvert [k];
        }
    }

    /* 6. ALE reconstruction */
    ale_reconstruct(mesh, tracers->data[tr_idx].values, tracers->del_ttf);
}

/*===========================================================================
 * M2.6-b — DEVICE (Kokkos) twin of the FCT pipeline (fesom_tracer_advect_one_fct).
 * Runs the whole MFCT 3rd-order H + QR4C V flux-corrected transport on the device
 * as one function owning ~24 parallel_for launches + its THREE internal-exchange
 * D21 brackets (fct_LO nod3D, tr_xy elem2D-full, fct_plus/fct_minus nod3D — the C
 * twin does these inside the pipeline). The DRIVER (fesom_step.cpp substep 13) owns
 * the IN rail (sync values/valuesold/uv/w_e/hnode/hnode_new/helem → device, L28) and
 * the OUT rail (sync_host values+valuesold before the host halo + Redi).
 *
 * THREE edge→node SCATTERS use Kokkos::atomic_add (D22 — Serial bit-identical in the
 * natural edge order, OpenMP/CUDA climate-close): compute_fct_LO step-1 divergence,
 * the Zalesak fct_plus/fct_minus assembly (b1), and flux2dtracer's horizontal
 * accumulation. Everything else is a per-node/per-edge/per-element race-free map or
 * private gather. ⚠️ AB2 eps=0.1 preserved (init_AB); ⚠️ the MFCT element gradient is
 * built from `values` (feedback_mfct_gradient_from_values) while the MFCT flux uses
 * `valuesAB` — exactly as the C twin (oce_tracer_mod.F90:127 / oce_adv_tra_driver.F90:307).
 *
 * Layout matches the C twin (L33: derive every bound from the body): adv_flux_hor is
 * zeroed/written over myDim_edge2D only; adv_flux_ver/fct_LO over myDim; init_AB +
 * Zalesak a1 over myDim+eDim; AUX written for owned elements only (halo AUX stays 0 —
 * the per-node a3 gather reads ±bignumber padding / 0, matching the C). The fct_plus/
 * fct_minus and fct_LO halo scatter garbage is overwritten by the D21 exchange before
 * any consumer reads it (b2/a1 read owned only; b3/MFCT read post-exchange).
 *===========================================================================*/

enum { NL_MAX = FESOM_MAX_LEVELS };   /* matches the cap used in fesom_eos / momentum / gm */

/* Device twin of node_avg_grad: area-weighted mean of tr_xy over the node's
 * surrounding elements that own layer nz, written into eud[gbase + nz*4 + cx/cy]. */
template <class DV, class IV>
KOKKOS_INLINE_FUNCTION
void node_avg_grad_kk(const DV &txy, const DV &earea, const IV &nlev, const IV &ulev,
                      const IV &off, const IV &nie, int nl,
                      int node, int nz0, int nz1,
                      const DV &eud, size_t gbase, int cx, int cy)
{
    int b0 = off(node), b1 = off(node + 1);
    for (int nz = nz0; nz < nz1; ++nz) {
        real_t tvol = 0.0, tx = 0.0, ty = 0.0;
        for (int k = b0; k < b1; ++k) {
            int el = nie(k);
            if (nz >= nlev(el) - 1 || nz < ulev(el) - 1) continue;
            real_t a = earea(el);
            tvol += a;
            tx   += txy((size_t)el * nl * 2 + nz * 2 + 0) * a;
            ty   += txy((size_t)el * nl * 2 + nz * 2 + 1) * a;
        }
        if (tvol > 0.0) { eud(gbase + nz*4 + cx) = tx / tvol; eud(gbase + nz*4 + cy) = ty / tvol; }
    }
}

/* Device twin of mfct_edge_flux: 3rd-order reconstructed antidiffusive flux at one
 * (edge, level), written as (HO − existing) since the LO upwind is already in `flux`. */
template <class DV>
KOKKOS_INLINE_FUNCTION
void mfct_edge_flux_kk(const DV &ttf, const DV &flux, const DV &eud,
                       int n1, int n2, int e, int nz, int nl,
                       real_t vflux, real_t a, real_t edx, real_t edy,
                       real_t RE, real_t num_ord)
{
    real_t T1   = ttf((size_t)n1 * nl + nz);
    real_t T2   = ttf((size_t)n2 * nl + nz);
    real_t diff = T2 - T1;
    size_t gb = ((size_t)e * nl + nz) * 4;
    real_t Tmean1 = T1 + (2.0*diff + edx*a*eud(gb+0) + edy*RE*eud(gb+2)) / 6.0;
    real_t Tmean2 = T2 - (2.0*diff + edx*a*eud(gb+1) + edy*RE*eud(gb+3)) / 6.0;
    real_t av  = Kokkos::fabs(vflux);
    real_t cHO = (vflux + av)*Tmean1 + (vflux - av)*Tmean2;
    real_t hi  = -0.5*(1.0 - num_ord)*cHO - vflux*num_ord*0.5*(Tmean1 + Tmean2);
    flux((size_t)e * nl + nz) = hi - flux((size_t)e * nl + nz);
}

void fesom_tracer_advect_one_fct_kk(fesom_tracer_adv_scratch *sc,
                                    int                       tr_idx,
                                    const struct fesom_mesh  *mesh,
                                    const struct fesom_dyn   *dyn,
                                    struct fesom_tracers     *tracers,
                                    struct fesom_partit      *partit)
{
    const int    nl       = mesh->nl;
    const int    myDim    = mesh->myDim_nod2D;
    const int    N_full   = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    myDim_e  = mesh->myDim_elem2D;
    const int    E_full   = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    const int    Ee       = mesh->myDim_edge2D;       /* Fortran adv/FCT loops are myDim only */
    const real_t dt       = (real_t)FESOM_PHASE1_DT;
    const real_t RE       = (real_t)FESOM_R_EARTH;
    const real_t eps      = 0.1;                       /* ⚠️ AB2 offset (oce_tracer_mod.F90:53) */
    const real_t c_old    = -(0.5 + eps);
    const real_t c_new    =  (1.5 + eps);
    const real_t flux_eps = 1e-16;
    const real_t bignumber= 1e3;
    FESOM_CHECK(nl <= NL_MAX, "tracer_advect_one_fct_kk: nl %d > NL_MAX", nl);

    /* device views */
    auto vals     = tracers->data[tr_idx].values_fld.d();
    auto valsAB   = tracers->data[tr_idx].valuesAB_fld.d();
    auto valsold  = tracers->data[tr_idx].valuesold_fld.d();
    auto delttf   = tracers->del_ttf_fld.d();
    auto aflux_h  = sc->adv_flux_hor_fld.d();
    auto aflux_v  = sc->adv_flux_ver_fld.d();
    auto dth      = sc->del_ttf_advhoriz_fld.d();
    auto dtv      = sc->del_ttf_advvert_fld.d();
    auto fctLO    = sc->fct_LO_fld.d();
    auto fmin     = sc->fct_ttf_min_fld.d();
    auto fmax     = sc->fct_ttf_max_fld.d();
    auto fplus    = sc->fct_plus_fld.d();
    auto fminus   = sc->fct_minus_fld.d();
    auto AUX      = sc->fct_aux_fld.d();
    auto trxy     = sc->tr_xy_fld.d();
    auto eud      = sc->edge_up_dn_grad_fld.d();
    auto uv       = dyn->uv_fld.d();
    auto W        = dyn->w_e_fld.d();
    auto edges    = mesh->edges_fld.d();
    auto edge_tri = mesh->edge_tri_fld.d();
    auto ecross   = mesh->edge_cross_dxdy_fld.d();
    auto edxdy    = mesh->edge_dxdy_fld.d();
    auto elnod    = mesh->elem_nodes_fld.d();
    auto ecos     = mesh->elem_cos_fld.d();
    auto grad     = mesh->gradient_sca_fld.d();
    auto earea    = mesh->elem_area_fld.d();
    auto helem    = mesh->helem_fld.d();
    auto hnode    = mesh->hnode_fld.d();
    auto hnode_new= mesh->hnode_new_fld.d();
    auto area     = mesh->area_fld.d();
    auto areasvol = mesh->areasvol_fld.d();
    auto zbar     = mesh->zbar_fld.d();
    auto Zc       = mesh->Z_fld.d();
    auto nlev_e   = mesh->nlevels_fld.d();
    auto ulev_e   = mesh->ulevels_fld.d();
    auto nlev_n   = mesh->nlevels_nod2D_fld.d();
    auto ulev_n   = mesh->ulevels_nod2D_fld.d();
    auto nlevn_min= mesh->nlevels_nod2D_min_fld.d();
    auto uleln_max= mesh->ulevels_nod2D_max_fld.d();
    auto eudt     = mesh->edge_up_dn_tri_fld.d();
    auto nie      = mesh->nod_in_elem2D_fld.d();
    auto off      = mesh->nod_in_elem2D_offsets_fld.d();

    using RP = Kokkos::RangePolicy<>;

    /* ===== 1. init_tracers_AB (⚠️ AB2 eps=0.1) ===== */
    Kokkos::parallel_for("fct_init_zero", RP(0, (size_t)N_full * nl),
        KOKKOS_LAMBDA(const size_t i) { delttf(i) = 0.0; dth(i) = 0.0; dtv(i) = 0.0; });
    Kokkos::parallel_for("fct_init_AB", RP(0, (size_t)N_full * nl),
        KOKKOS_LAMBDA(const size_t i) { valsAB(i) = c_old * valsold(i) + c_new * vals(i); });
    Kokkos::parallel_for("fct_init_save", RP(0, (size_t)N_full * nl),   /* valuesold = values (after AB) */
        KOKKOS_LAMBDA(const size_t i) { valsold(i) = vals(i); });

    /* ===== 2. LO upwind horizontal flux from `values` (adv_tra_hor_upw1) ===== */
    Kokkos::parallel_for("fct_upw1h_zero", RP(0, (size_t)Ee * nl),
        KOKKOS_LAMBDA(const size_t i) { aflux_h(i) = 0.0; });
    Kokkos::parallel_for("fct_upw1h", RP(0, Ee), KOKKOS_LAMBDA(const int e) {
        int n1 = edges(2*e+0), n2 = edges(2*e+1);
        int el1 = edge_tri(2*e+0), el2 = edge_tri(2*e+1);
        int nu1 = (el1>=0)? ulev_e(el1)-1 : 0, nl1 = (el1>=0)? nlev_e(el1)-1 : 0;
        int nu2 = (el2>=0)? ulev_e(el2)-1 : 0, nl2 = (el2>=0)? nlev_e(el2)-1 : 0;
        real_t dx1 = (el1>=0)? ecross(4*e+0) : 0.0, dy1 = (el1>=0)? ecross(4*e+1) : 0.0;
        real_t dx2 = (el2>=0)? ecross(4*e+2) : 0.0, dy2 = (el2>=0)? ecross(4*e+3) : 0.0;
        int nl12 = (el2>=0)? ((nl1<nl2)?nl1:nl2) : 0;
        int nu12 = nu1; if (nu2 > nu12) nu12 = nu2;
        for (int nz = nu1; nz < nu12; ++nz) {                 /* (A) el1-only above */
            real_t u = uv((size_t)el1*nl*2+nz*2+0), v = uv((size_t)el1*nl*2+nz*2+1);
            real_t vflux = (-v*dx1 + u*dy1) * helem((size_t)el1*nl+nz);
            real_t T1 = vals((size_t)n1*nl+nz), T2 = vals((size_t)n2*nl+nz), av = Kokkos::fabs(vflux);
            aflux_h((size_t)e*nl+nz) = -0.5*(T1*(vflux+av) + T2*(vflux-av));
        }
        if (el2>=0) for (int nz = nu2; nz < nu12; ++nz) {     /* (B) el2-only above */
            real_t u = uv((size_t)el2*nl*2+nz*2+0), v = uv((size_t)el2*nl*2+nz*2+1);
            real_t vflux = (v*dx2 - u*dy2) * helem((size_t)el2*nl+nz);
            real_t T1 = vals((size_t)n1*nl+nz), T2 = vals((size_t)n2*nl+nz), av = Kokkos::fabs(vflux);
            aflux_h((size_t)e*nl+nz) = -0.5*(T1*(vflux+av) + T2*(vflux-av));
        }
        for (int nz = nu12; nz < nl12; ++nz) {                /* (C) both */
            real_t u1 = uv((size_t)el1*nl*2+nz*2+0), v1 = uv((size_t)el1*nl*2+nz*2+1);
            real_t u2 = uv((size_t)el2*nl*2+nz*2+0), v2 = uv((size_t)el2*nl*2+nz*2+1);
            real_t vflux = (-v1*dx1 + u1*dy1) * helem((size_t)el1*nl+nz)
                         + ( v2*dx2 - u2*dy2) * helem((size_t)el2*nl+nz);
            real_t T1 = vals((size_t)n1*nl+nz), T2 = vals((size_t)n2*nl+nz), av = Kokkos::fabs(vflux);
            aflux_h((size_t)e*nl+nz) = -0.5*(T1*(vflux+av) + T2*(vflux-av));
        }
        for (int nz = nl12; nz < nl1; ++nz) {                 /* (D) el1-only below */
            real_t u = uv((size_t)el1*nl*2+nz*2+0), v = uv((size_t)el1*nl*2+nz*2+1);
            real_t vflux = (-v*dx1 + u*dy1) * helem((size_t)el1*nl+nz);
            real_t T1 = vals((size_t)n1*nl+nz), T2 = vals((size_t)n2*nl+nz), av = Kokkos::fabs(vflux);
            aflux_h((size_t)e*nl+nz) = -0.5*(T1*(vflux+av) + T2*(vflux-av));
        }
        if (el2>=0) for (int nz = nl12; nz < nl2; ++nz) {     /* (E) el2-only below */
            real_t u = uv((size_t)el2*nl*2+nz*2+0), v = uv((size_t)el2*nl*2+nz*2+1);
            real_t vflux = (v*dx2 - u*dy2) * helem((size_t)el2*nl+nz);
            real_t T1 = vals((size_t)n1*nl+nz), T2 = vals((size_t)n2*nl+nz), av = Kokkos::fabs(vflux);
            aflux_h((size_t)e*nl+nz) = -0.5*(T1*(vflux+av) + T2*(vflux-av));
        }
    });

    /* ===== 3. LO upwind vertical flux from `values` (adv_tra_ver_upw1) ===== */
    Kokkos::parallel_for("fct_upw1v_zero", RP(0, (size_t)myDim * nl),
        KOKKOS_LAMBDA(const size_t i) { aflux_v(i) = 0.0; });
    /* M5.21 flat lever: one thread per (node,LEVEL) — RP(0,myDim*nl), decode (n,nz). The
     * field is node-major (aflux_v[n*nl+nz]) so the LEVEL is contiguous → coalesced. The
     * vertical read vals[nz-1] is an INPUT (vals unmodified here) → each (n,nz) independent;
     * each writes only its own level → no scatter, Serial AND OpenMP bit-identical. */
    Kokkos::parallel_for("fct_upw1v", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nzmin = ulev_n(n)-1, nzmax = nlev_n(n)-1;
        if (nz < nzmin || nz >= nzmax) return;     /* writes [nzmin,nzmax); nzmax<=nzmin ⇒ all masked */
        if (nz == nzmin) {
            aflux_v((size_t)n*nl+nzmin) = -W((size_t)n*nl+nzmin) * vals((size_t)n*nl+nzmin)
                                         * area((size_t)n*nl+nzmin);
        } else {
            real_t w_iface = W((size_t)n*nl+nz);
            real_t T_below = vals((size_t)n*nl+nz), T_above = vals((size_t)n*nl+(nz-1));
            real_t a = area((size_t)n*nl+nz), aw = Kokkos::fabs(w_iface);
            aflux_v((size_t)n*nl+nz) = -0.5*(T_below*(w_iface+aw) + T_above*(w_iface-aw)) * a;
        }
    });

    /* ===== 4. fct_LO from upwind (compute_fct_LO): edge→node SCATTER + per-node finalise ===== */
    Kokkos::parallel_for("fct_LO_zero", RP(0, (size_t)myDim * nl),
        KOKKOS_LAMBDA(const size_t i) { fctLO(i) = 0.0; });
    Kokkos::parallel_for("fct_LO_scatter", RP(0, Ee), KOKKOS_LAMBDA(const int e) {
        int n1 = edges(2*e+0), n2 = edges(2*e+1);
        int el1 = edge_tri(2*e+0), el2 = edge_tri(2*e+1);
        int nl1 = (el1>=0)? nlev_e(el1)-1 : 0, nu1 = (el1>=0)? ulev_e(el1)-1 : 0;
        int nl2 = (el2>=0)? nlev_e(el2)-1 : 0, nu2 = (el2>=0)? ulev_e(el2)-1 : 0;
        int nl12 = (nl1>nl2)? nl1 : nl2;
        int nu12 = nu1; if (nu2 > nu12) nu12 = nu2;
        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f = aflux_h((size_t)e*nl+nz);
            Kokkos::atomic_add(&fctLO((size_t)n1*nl+nz),  f);
            Kokkos::atomic_add(&fctLO((size_t)n2*nl+nz), -f);
        }
    });
    /* M5.21 flat lever: one thread per (node,LEVEL). f_bot = aflux_v[nz+1] is an INPUT
     * (aflux_v built in upw1v, not modified here); fctLO[k] is read-then-written at the
     * SAME level (no cross-level dep) → each (n,nz) independent, bit-identical. */
    Kokkos::parallel_for("fct_LO_final", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nu1 = ulev_n(n)-1, nl1 = nlev_n(n)-1;
        if (nz < nu1 || nz >= nl1) return;
        size_t k = (size_t)n*nl+nz;
        real_t a = areasvol(k), hnod_old = hnode(k), hnod_new = hnode_new(k);
        if (hnod_new <= 0.0 || a <= 0.0) { fctLO(k) = 0.0; return; }
        real_t f_top = aflux_v((size_t)n*nl+nz), f_bot = aflux_v((size_t)n*nl+(nz+1));
        real_t numer = vals(k)*hnod_old + (fctLO(k) + (f_top - f_bot)) * dt / a;
        fctLO(k) = numer / hnod_new;
    });
    /* D21 internal halo: exchange fct_LO (Zalesak a1 reads LO at halo nodes).
     * M5.4c: device-halo (GPU-aware MPI on CUDA, exact host bracket on Serial). */
    fesom_halo_field(sc->fct_LO_fld, FESOM_HALO_NOD3D, nl, 1, partit);

    /* ===== 5. HO horizontal (MFCT): element gradient FROM `values`, exchange, fill, flux FROM valuesAB ===== */
    Kokkos::parallel_for("fct_grad_zero", RP(0, (size_t)E_full * nl * 2),
        KOKKOS_LAMBDA(const size_t i) { trxy(i) = 0.0; });
    /* M5.21 flat lever: one thread per (elem,LEVEL) — RP(0,myDim_e*nl), decode (el,nz).
     * trxy is elem-major×2-comp (trxy[(el*nl+nz)*2+c]); one-thread-per-(el,nz) writes the
     * 2 adjacent comps → a warp writes 64 contiguous doubles (coalesced); the 3 same-level
     * vertex reads vals[nv*nl+nz] also coalesce across consecutive nz. Per-level map, b-i. */
    Kokkos::parallel_for("fct_grad_elem", RP(0, (size_t)myDim_e * nl), KOKKOS_LAMBDA(const size_t i) {
        const int el = (int)(i / nl), nz = (int)(i - (size_t)el*nl);
        int nu = ulev_e(el)-1, nle = nlev_e(el)-1;
        if (nz < nu || nz >= nle) return;
        int n0 = elnod(3*el+0), n1 = elnod(3*el+1), n2 = elnod(3*el+2);
        real_t g0=grad(6*el+0), g1=grad(6*el+1), g2=grad(6*el+2);
        real_t g3=grad(6*el+3), g4=grad(6*el+4), g5=grad(6*el+5);
        real_t t0 = vals((size_t)n0*nl+nz), t1 = vals((size_t)n1*nl+nz), t2 = vals((size_t)n2*nl+nz);
        trxy((size_t)el*nl*2+nz*2+0) = g0*t0 + g1*t1 + g2*t2;
        trxy((size_t)el*nl*2+nz*2+1) = g3*t0 + g4*t1 + g5*t2;
    });
    /* D21 internal halo: exchange tr_xy (full element halo, 2-comp; fill_up_dn_grad +
     * the node gather read it at HALO elements). M5.4c: device-halo (GPU-aware MPI). */
    fesom_halo_field(sc->tr_xy_fld, FESOM_HALO_ELEM2D_FULL, nl, 2, partit);

    /* fill_up_dn_grad: per-edge up/down + node-averaged gradients. */
    Kokkos::parallel_for("fct_eud_zero", RP(0, (size_t)Ee * nl * 4),
        KOKKOS_LAMBDA(const size_t i) { eud(i) = 0.0; });
    Kokkos::parallel_for("fct_eud_fill", RP(0, Ee), KOKKOS_LAMBDA(const int ed) {
        int n1 = edges(2*ed+0), n2 = edges(2*ed+1);
        int up = eudt(2*ed+0), dn = eudt(2*ed+1);
        size_t gb = (size_t)ed * nl * 4;
        if (up >= 0 && dn >= 0) {
            int a1 = uleln_max(n1), a2 = uleln_max(n2);
            int nzmin = ((a1>a2)?a1:a2) - 1;
            int b1 = nlevn_min(n1), b2 = nlevn_min(n2);
            int nzmax = ((b1<b2)?b1:b2) - 1;
            node_avg_grad_kk(trxy, earea, nlev_e, ulev_e, off, nie, nl, n1, ulev_n(n1)-1, nzmin, eud, gb, 0, 2);
            node_avg_grad_kk(trxy, earea, nlev_e, ulev_e, off, nie, nl, n2, ulev_n(n2)-1, nzmin, eud, gb, 1, 3);
            for (int nz = nzmin; nz < nzmax; ++nz) {
                eud(gb+nz*4+0) = trxy((size_t)up*nl*2+nz*2+0);
                eud(gb+nz*4+1) = trxy((size_t)dn*nl*2+nz*2+0);
                eud(gb+nz*4+2) = trxy((size_t)up*nl*2+nz*2+1);
                eud(gb+nz*4+3) = trxy((size_t)dn*nl*2+nz*2+1);
            }
            node_avg_grad_kk(trxy, earea, nlev_e, ulev_e, off, nie, nl, n1, nzmax, nlev_n(n1)-1, eud, gb, 0, 2);
            node_avg_grad_kk(trxy, earea, nlev_e, ulev_e, off, nie, nl, n2, nzmax, nlev_n(n2)-1, eud, gb, 1, 3);
        } else {
            node_avg_grad_kk(trxy, earea, nlev_e, ulev_e, off, nie, nl, n1, ulev_n(n1)-1, nlev_n(n1)-1, eud, gb, 0, 2);
            node_avg_grad_kk(trxy, earea, nlev_e, ulev_e, off, nie, nl, n2, ulev_n(n2)-1, nlev_n(n2)-1, eud, gb, 1, 3);
        }
    });

    /* adv_tra_hor_mfct (num_ord=0, init_zero=0 → += antidiffusive HO−LO), ttf = valuesAB. */
    Kokkos::parallel_for("fct_mfct_h", RP(0, Ee), KOKKOS_LAMBDA(const int e) {
        int n1 = edges(2*e+0), n2 = edges(2*e+1);
        int el1 = edge_tri(2*e+0), el2 = edge_tri(2*e+1);
        int nu1 = (el1>=0)? ulev_e(el1)-1 : 0, nl1 = (el1>=0)? nlev_e(el1)-1 : 0;
        int nu2 = (el2>=0)? ulev_e(el2)-1 : 0, nl2 = (el2>=0)? nlev_e(el2)-1 : 0;
        real_t dx1 = (el1>=0)? ecross(4*e+0) : 0.0, dy1 = (el1>=0)? ecross(4*e+1) : 0.0;
        real_t dx2 = (el2>=0)? ecross(4*e+2) : 0.0, dy2 = (el2>=0)? ecross(4*e+3) : 0.0;
        int nl12 = (el2>=0)? ((nl1<nl2)?nl1:nl2) : 0;
        int nu12 = (nu1>nu2)? nu1 : nu2;
        real_t edx = edxdy(2*e+0), edy = edxdy(2*e+1);
        real_t a = (el1>=0)? RE*ecos(el1) : 0.0;
        if (el2>=0) a = 0.5*(a + RE*ecos(el2));
        for (int nz = nu1; nz < nu12; ++nz) {                 /* (A) el1-only above */
            real_t u = uv((size_t)el1*nl*2+nz*2+0), v = uv((size_t)el1*nl*2+nz*2+1);
            real_t vflux = (-v*dx1 + u*dy1) * helem((size_t)el1*nl+nz);
            mfct_edge_flux_kk(valsAB, aflux_h, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, 0.0);
        }
        if (el2>=0) for (int nz = nu2; nz < nu12; ++nz) {     /* (B) el2-only above */
            real_t u = uv((size_t)el2*nl*2+nz*2+0), v = uv((size_t)el2*nl*2+nz*2+1);
            real_t vflux = (v*dx2 - u*dy2) * helem((size_t)el2*nl+nz);
            mfct_edge_flux_kk(valsAB, aflux_h, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, 0.0);
        }
        for (int nz = nu12; nz < nl12; ++nz) {                /* (C) both */
            real_t u1 = uv((size_t)el1*nl*2+nz*2+0), v1 = uv((size_t)el1*nl*2+nz*2+1);
            real_t u2 = uv((size_t)el2*nl*2+nz*2+0), v2 = uv((size_t)el2*nl*2+nz*2+1);
            real_t vflux = (-v1*dx1 + u1*dy1) * helem((size_t)el1*nl+nz)
                         + ( v2*dx2 - u2*dy2) * helem((size_t)el2*nl+nz);
            mfct_edge_flux_kk(valsAB, aflux_h, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, 0.0);
        }
        for (int nz = nl12; nz < nl1; ++nz) {                 /* (D) el1-only below */
            real_t u = uv((size_t)el1*nl*2+nz*2+0), v = uv((size_t)el1*nl*2+nz*2+1);
            real_t vflux = (-v*dx1 + u*dy1) * helem((size_t)el1*nl+nz);
            mfct_edge_flux_kk(valsAB, aflux_h, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, 0.0);
        }
        if (el2>=0) for (int nz = nl12; nz < nl2; ++nz) {     /* (E) el2-only below */
            real_t u = uv((size_t)el2*nl*2+nz*2+0), v = uv((size_t)el2*nl*2+nz*2+1);
            real_t vflux = (v*dx2 - u*dy2) * helem((size_t)el2*nl+nz);
            mfct_edge_flux_kk(valsAB, aflux_h, eud, n1, n2, e, nz, nl, vflux, a, edx, edy, RE, 0.0);
        }
    });

    /* ===== 6. HO vertical (adv_tra_ver_qr4c, num_ord=1, init_zero=0 → += HO−LO), ttf=valuesAB ===== */
    /* M5.21 flat lever: one thread per (node,LEVEL) — RP(0,myDim*nl), decode (n,nz). qr4c_v
     * does aflux_v[k] = HO(k) - aflux_v[k] read-then-write at the SAME level; all vertical
     * reads (valsAB[nz±1,±2], Zc[nz±2]) are INPUTS → per-level map. The branch dispatch
     * replicates the original's surface / 2nd-layer / bottom-1 / bottom / interior cases.
     * ⚠️ The ONE within-column write-dep: when nzmax-nzmin==2 the 2nd-layer and bottom-1
     * writes BOTH hit level nzmin+1 (the 2nd reading the 1st's result with the SAME
     * -0.5(Tup+Tdn)W·area term) → they cancel to a net no-op; reproduced by leaving that
     * level untouched. Every other level is written exactly once → Serial bit-identical. */
    Kokkos::parallel_for("fct_qr4c_v", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nzmin = ulev_n(n)-1, nzmax = nlev_n(n)-1;
        if (nzmax <= nzmin) return;
        if (nz < nzmin || nz > nzmax) return;                   /* qr4c_v writes [nzmin,nzmax] inclusive */
        const real_t num_ord = 1.0;
        size_t k = (size_t)n*nl+nz;
        if (nzmax - nzmin == 2 && nz == nzmin+1) {
            return;                                             /* 2nd-layer ∘ bottom-1 cancel → net no-op */
        } else if (nz == nzmin) {                               /* surface */
            aflux_v(k) = -valsAB(k) * W(k) * area(k) - aflux_v(k);
        } else if (nz == nzmax) {                               /* bottom */
            aflux_v(k) = 0.0 - aflux_v(k);
        } else if (nz == nzmin+1) {                             /* 2nd layer (nzmax-nzmin>=3) */
            real_t Tup = valsAB((size_t)n*nl+(nz-1)), Tdn = valsAB((size_t)n*nl+nz);
            aflux_v(k) = -0.5*(Tup+Tdn) * W(k) * area(k) - aflux_v(k);
        } else if (nz == nzmax-1) {                             /* bottom-1 (nzmax-nzmin>=3) */
            real_t Tup = valsAB((size_t)n*nl+(nz-1)), Tdn = valsAB((size_t)n*nl+nz);
            aflux_v(k) = -0.5*(Tup+Tdn) * W(k) * area(k) - aflux_v(k);
        } else {                                                /* interior 4th-order quadratic */
            real_t Z_um1 = Zc(nz-1), Z_u = Zc(nz), Z_dn = Zc(nz+1), Z_um2 = Zc(nz-2);
            real_t Tum1 = valsAB((size_t)n*nl+(nz-1)), Tu = valsAB((size_t)n*nl+nz);
            real_t Tdn  = valsAB((size_t)n*nl+(nz+1)), Tum2 = valsAB((size_t)n*nl+(nz-2));
            real_t qc = (Tum1 - Tu) / (Z_um1 - Z_u);
            real_t qu = (Tu - Tdn) / (Z_u - Z_dn);
            real_t qd = (Tum2 - Tum1) / (Z_um2 - Z_um1);
            real_t zb = zbar(nz);
            real_t Tmean1 = Tu   + (2.0*qc + qu) * (zb - Z_u  ) / 3.0;
            real_t Tmean2 = Tum1 + (2.0*qc + qd) * (zb - Z_um1) / 3.0;
            real_t w_iface = W((size_t)n*nl+nz), aw = Kokkos::fabs(w_iface);
            real_t Tmean = (w_iface + aw)*Tmean1 + (w_iface - aw)*Tmean2;
            real_t a = area((size_t)n*nl+nz);
            real_t hi = (-0.5*(1.0 - num_ord)*Tmean - num_ord*0.5*(Tmean1 + Tmean2)*w_iface) * a;
            aflux_v((size_t)n*nl+nz) = hi - aflux_v((size_t)n*nl+nz);
        }
    });

    /* ===== 7. Zalesak limiter (oce_tra_adv_fct), ttf = values, LO = fct_LO ===== */
    /* a1: per-node max/min(LO, ttf) over OWNED+HALO (a2 reads at halo vertices). */
    /* M5.21 flat lever: one thread per (node,LEVEL) over OWNED+HALO (N_full). Per-level
     * map (fmax/fmin from same-level fctLO/vals) → coalesced, bit-identical. */
    Kokkos::parallel_for("fct_zal_a1", RP(0, (size_t)N_full * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nu1 = ulev_n(n)-1, nl1 = nlev_n(n)-1;
        if (nz < nu1 || nz >= nl1) return;
        size_t k = (size_t)n*nl+nz; real_t a = fctLO(k), b = vals(k);
        fmax(k) = (a>b)?a:b; fmin(k) = (a<b)?a:b;
    });
    /* a2: per-element max/min over 3 vertices + ±bignumber pad below nlevels. */
    /* M5.21 flat lever: one thread per (elem,LEVEL). AUX is elem-major×2 → coalesced.
     * Two regimes per column: [nu1,nle) = max/min over the 3 vertices (same-level reads);
     * [nle,nl-1) = ±bignumber pad (ulevels≤nlevels ⇒ nu1≤nle, so the regimes don't gap).
     * Per-level map, bit-identical. */
    Kokkos::parallel_for("fct_zal_a2", RP(0, (size_t)myDim_e * nl), KOKKOS_LAMBDA(const size_t i) {
        const int e = (int)(i / nl), nz = (int)(i - (size_t)e*nl);
        int nu1 = ulev_e(e)-1, nle = nlev_e(e)-1;
        if (nz < nu1 || nz >= nl-1) return;
        if (nz < nle) {
            int n0 = elnod(3*e+0), n1 = elnod(3*e+1), n2 = elnod(3*e+2);
            real_t a0=fmax((size_t)n0*nl+nz), a1=fmax((size_t)n1*nl+nz), a2=fmax((size_t)n2*nl+nz);
            real_t mx=a0; if(a1>mx)mx=a1; if(a2>mx)mx=a2;
            real_t b0=fmin((size_t)n0*nl+nz), b1=fmin((size_t)n1*nl+nz), b2=fmin((size_t)n2*nl+nz);
            real_t mn=b0; if(b1<mn)mn=b1; if(b2<mn)mn=b2;
            AUX(((size_t)e*nl+nz)*2+0) = mx;
            AUX(((size_t)e*nl+nz)*2+1) = mn;
        } else {
            AUX(((size_t)e*nl+nz)*2+0) = -bignumber;
            AUX(((size_t)e*nl+nz)*2+1) =  bignumber;
        }
    });
    /* a3+a4 fused (per OWNED node, column-local tvert): cluster max/min over surrounding
     * cells (a3) then admissible increment vs LO with vertical 3-layer cluster (a4). */
    Kokkos::parallel_for("fct_zal_a34", RP(0, myDim), KOKKOS_LAMBDA(const int n) {
        int nu1 = ulev_n(n)-1, nl1 = nlev_n(n)-1;
        if (nu1 >= nl1) return;
        real_t tvmax[NL_MAX], tvmin[NL_MAX];
        int o0 = off(n), o1 = off(n+1);
        for (int nz = nu1; nz < nl1; ++nz) {                  /* a3: cluster over cells */
            int e0 = nie(o0);
            real_t mx = AUX(((size_t)e0*nl+nz)*2+0), mn = AUX(((size_t)e0*nl+nz)*2+1);
            for (int k = o0+1; k < o1; ++k) {
                int e = nie(k);
                real_t a = AUX(((size_t)e*nl+nz)*2+0), b = AUX(((size_t)e*nl+nz)*2+1);
                if (a > mx) mx = a;
                if (b < mn) mn = b;
            }
            tvmax[nz] = mx; tvmin[nz] = mn;
        }
        { int nz = nu1; size_t k = (size_t)n*nl+nz;           /* a4 surface */
          fmax(k) = tvmax[nz] - fctLO(k); fmin(k) = tvmin[nz] - fctLO(k); }
        for (int nz = nu1+1; nz < nl1-1; ++nz) {              /* a4 interior 3-layer cluster */
            size_t k = (size_t)n*nl+nz;
            real_t mx = tvmax[nz-1]; if (tvmax[nz]>mx) mx=tvmax[nz]; if (tvmax[nz+1]>mx) mx=tvmax[nz+1];
            real_t mn = tvmin[nz-1]; if (tvmin[nz]<mn) mn=tvmin[nz]; if (tvmin[nz+1]<mn) mn=tvmin[nz+1];
            fmax(k) = mx - fctLO(k); fmin(k) = mn - fctLO(k);
        }
        if (nl1-1 > nu1) { int nz = nl1-1; size_t k = (size_t)n*nl+nz;   /* a4 bottom */
          fmax(k) = tvmax[nz] - fctLO(k); fmin(k) = tvmin[nz] - fctLO(k); }
    });
    /* b1: per-node vertical antidiffusive sums (zero+vertical fused → assign). */
    /* M5.22 flat lever: one thread per (node,LEVEL) — RP(0,myDim*nl), decode (n,nz). Per-(n,nz)
     * map: writes fplus/fminus[k] from aflux_v[nz] and the INPUT aflux_v[nz+1] (qr4c_v output,
     * not written here) → level contiguous → coalesced, bit-identical. */
    Kokkos::parallel_for("fct_zal_b1v", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nu1 = ulev_n(n)-1, nl1 = nlev_n(n)-1;
        if (nz < nu1 || nz >= nl1) return;
        real_t fv_top = aflux_v((size_t)n*nl+nz), fv_bot = aflux_v((size_t)n*nl+(nz+1));
        real_t pos = (fv_top>0.0?fv_top:0.0) + (-fv_bot>0.0?-fv_bot:0.0);
        real_t neg = (fv_top<0.0?fv_top:0.0) + (-fv_bot<0.0?-fv_bot:0.0);
        size_t k = (size_t)n*nl+nz; fplus(k) = pos; fminus(k) = neg;
    });
    /* b1 horizontal: edge→node SCATTER (atomic_add) into fct_plus/fct_minus. */
    Kokkos::parallel_for("fct_zal_b1h", RP(0, Ee), KOKKOS_LAMBDA(const int e) {
        int n1 = edges(2*e+0), n2 = edges(2*e+1);
        int el1 = edge_tri(2*e+0), el2 = edge_tri(2*e+1);
        int nl1 = (el1>=0)? nlev_e(el1)-1 : 0, nu1 = (el1>=0)? ulev_e(el1)-1 : 0;
        int nl2 = (el2>=0)? nlev_e(el2)-1 : 0, nu2 = (el2>=0)? ulev_e(el2)-1 : 0;
        int nl12 = (nl1>nl2)? nl1 : nl2;
        int nu12 = nu1; if (el2>=0 && nu2 < nu12) nu12 = nu2;
        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f = aflux_h((size_t)e*nl+nz);
            Kokkos::atomic_add(&fplus ((size_t)n1*nl+nz), (f>0.0? f:0.0));
            Kokkos::atomic_add(&fminus((size_t)n1*nl+nz), (f<0.0? f:0.0));
            Kokkos::atomic_add(&fplus ((size_t)n2*nl+nz), (-f>0.0? -f:0.0));
            Kokkos::atomic_add(&fminus((size_t)n2*nl+nz), (-f<0.0? -f:0.0));
        }
    });
    /* b2: per-node limiter factors. */
    /* M5.22 flat lever: one thread per (node,LEVEL). Pure per-level map (every read/write at the
     * same k: areasvol/hnode_new/fplus/fminus/fmax/fmin) → coalesced, bit-identical (continue→return). */
    Kokkos::parallel_for("fct_zal_b2", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nu1 = ulev_n(n)-1, nl1 = nlev_n(n)-1;
        if (nz < nu1 || nz >= nl1) return;
        size_t k = (size_t)n*nl+nz; real_t a = areasvol(k), hn = hnode_new(k);
        if (a <= 0.0 || hn <= 0.0) { fplus(k) = 1.0; fminus(k) = 1.0; return; }
        real_t flux_pos = fplus(k)*dt/a/hn + flux_eps;
        real_t r_pos = fmax(k) / flux_pos; fplus(k) = (r_pos<1.0)?r_pos:1.0;
        real_t flux_neg = fminus(k)*dt/a/hn - flux_eps;
        real_t r_neg = fmin(k) / flux_neg; fminus(k) = (r_neg<1.0)?r_neg:1.0;
    });
    /* D21 internal halo: exchange fct_plus/fct_minus (b3 horizontal reads at edge
     * endpoints, which can be HALO nodes). M5.4c: device-halo (GPU-aware MPI).
     * M5.23 (L3): fct_plus+fct_minus are same-kind (NOD3D nc=1) and adjacent (b2 wrote both;
     * nothing reads/writes them between) → one FUSED message/neighbour instead of two. Runs
     * 2×/step (T+S → the highest-freq L3 pair). Bit-identical (field2 = co-pack only). */
    fesom_halo_field2(sc->fct_plus_fld, sc->fct_minus_fld, FESOM_HALO_NOD3D, nl, 1, partit);
    /* b3 vertical: limit adf_v (per-node, own column). */
    /* M5.22 flat lever: one thread per (node,LEVEL). Each (n,nz) RMWs only its OWN aflux_v[k];
     * the nz-1 reads are fminus/fplus (b2 output = INPUTS here, NOT aflux_v) → no written-value
     * recurrence → per-level map, coalesced, bit-identical. Surface level nz==nu1 folds into the mask. */
    Kokkos::parallel_for("fct_zal_b3v", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nu1 = ulev_n(n)-1, nl1 = nlev_n(n)-1;
        if (nz < nu1 || nz >= nl1) return;
        size_t k = (size_t)n*nl+nz; real_t f = aflux_v(k); real_t ae;
        if (nz == nu1)     { ae = (f>=0.0)? fplus(k) : fminus(k); }
        else if (f >= 0.0) { real_t a = fminus((size_t)n*nl+(nz-1)), b = fplus(k);  ae=(a<b)?a:b; }
        else               { real_t a = fplus ((size_t)n*nl+(nz-1)), b = fminus(k); ae=(a<b)?a:b; }
        aflux_v(k) = ae*f;
    });
    /* b3 horizontal: limit adf_h (per-edge, own slot; reads fct_plus/minus at endpoints). */
    Kokkos::parallel_for("fct_zal_b3h", RP(0, Ee), KOKKOS_LAMBDA(const int e) {
        int n1 = edges(2*e+0), n2 = edges(2*e+1);
        int el1 = edge_tri(2*e+0), el2 = edge_tri(2*e+1);
        int nl1 = (el1>=0)? nlev_e(el1)-1 : 0, nu1 = (el1>=0)? ulev_e(el1)-1 : 0;
        int nl2 = (el2>=0)? nlev_e(el2)-1 : 0, nu2 = (el2>=0)? ulev_e(el2)-1 : 0;
        int nl12 = (nl1>nl2)? nl1 : nl2;
        int nu12 = nu1; if (el2>=0 && nu2 < nu12) nu12 = nu2;
        for (int nz = nu12; nz < nl12; ++nz) {
            size_t k = (size_t)e*nl+nz; real_t f = aflux_h(k); real_t ae;
            if (f >= 0.0) { real_t a = fplus ((size_t)n1*nl+nz), b = fminus((size_t)n2*nl+nz); ae=(a<b)?a:b; }
            else          { real_t a = fminus((size_t)n1*nl+nz), b = fplus ((size_t)n2*nl+nz); ae=(a<b)?a:b; }
            aflux_h(k) = ae*f;
        }
    });

    /* ===== 8. flux2dtracer_fct (LO transition + limited HO into del_ttf_adv{vert,horiz}) ===== */
    /* M5.21 flat lever: one thread per (node,LEVEL). Both original loops accumulate into
     * dtv[k] at the SAME level → fuse per (n,nz), preserving the two-step += order (LO
     * transition then antidiffusive divergence; f_bot=aflux_v[nz+1] is an INPUT). The a<=0
     * guard skips only the 2nd term (the 1st += is unconditional, as in the C twin). b-i. */
    Kokkos::parallel_for("fct_f2d_v", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nu1 = ulev_n(n)-1, nl1 = nlev_n(n)-1;
        if (nz < nu1 || nz >= nl1) return;
        size_t k = (size_t)n*nl+nz;
        dtv(k) += -vals(k)*hnode(k) + fctLO(k)*hnode_new(k);      /* LO transition */
        real_t a = areasvol(k);                                  /* antidiffusive vertical divergence */
        if (a <= 0.0) return;
        real_t f_top = aflux_v((size_t)n*nl+nz), f_bot = aflux_v((size_t)n*nl+(nz+1));
        dtv(k) += (f_top - f_bot) * dt / a;
    });
    Kokkos::parallel_for("fct_f2d_h", RP(0, Ee), KOKKOS_LAMBDA(const int e) {   /* edge→node SCATTER */
        int n1 = edges(2*e+0), n2 = edges(2*e+1);
        int el1 = edge_tri(2*e+0), el2 = edge_tri(2*e+1);
        int nl1 = (el1>=0)? nlev_e(el1)-1 : 0, nu1 = (el1>=0)? ulev_e(el1)-1 : 0;
        int nl2 = (el2>=0)? nlev_e(el2)-1 : 0, nu2 = (el2>=0)? ulev_e(el2)-1 : 0;
        int nl12 = (nl1>nl2)? nl1 : nl2;
        int nu12 = nu1; if (el2>=0 && nu2 > 0 && nu2 < nu12) nu12 = nu2;
        for (int nz = nu12; nz < nl12; ++nz) {
            real_t f = aflux_h((size_t)e*nl+nz);
            real_t a1 = areasvol((size_t)n1*nl+nz), a2 = areasvol((size_t)n2*nl+nz);
            if (a1 > 0.0) Kokkos::atomic_add(&dth((size_t)n1*nl+nz),   f*dt/a1);
            if (a2 > 0.0) Kokkos::atomic_add(&dth((size_t)n2*nl+nz), -(f*dt/a2));
        }
    });

    /* ===== 9. del_ttf = del_ttf_advhoriz + del_ttf_advvert (per-node, full nl) ===== */
    Kokkos::parallel_for("fct_delttf_sum", RP(0, (size_t)myDim * nl),
        KOKKOS_LAMBDA(const size_t i) { delttf(i) = dth(i) + dtv(i); });

    /* ===== 10. ALE reconstruction (T_new = LO + limited antidiff) ===== */
    /* M5.21 flat lever: one thread per (node,LEVEL). All reads/writes at the SAME level
     * (delttf[k], vals[k]); the vals write uses the just-updated delttf[k] → each (n,nz)
     * independent, bit-identical. */
    Kokkos::parallel_for("fct_ale_recon", RP(0, (size_t)myDim * nl), KOKKOS_LAMBDA(const size_t i) {
        const int n = (int)(i / nl), nz = (int)(i - (size_t)n*nl);
        int nzmin = ulev_n(n)-1, nzmax = nlev_n(n)-1;
        if (nz < nzmin || nz >= nzmax) return;
        size_t k = (size_t)n*nl+nz;
        real_t hnode_old = hnode(k), hn_new = hnode_new(k);
        delttf(k) += vals(k) * (hnode_old - hn_new);
        if (hn_new > 0.0) vals(k) += delttf(k) / hn_new;
    });

    tracers->data[tr_idx].values_fld.modify_device();
    tracers->data[tr_idx].valuesold_fld.modify_device();
}

/*--- FESOM_KK_VERIFY=tradv gate ---------------------------------------------
 * The FCT pipeline reads-and-modifies `values` (and overwrites `valuesold = values`
 * inside init_AB, after reading the original valuesold for the AB2 combination), so
 * this is the L26 capture-before with TWO inputs: the DRIVER snapshots the pre-FCT
 * `values` and `valuesold` and passes them. Restore both, run the C twin (recomputes
 * everything incl. its own internal halos — a no-op at np=1), diff the final `values`,
 * restore the KK production state (values + valuesold). del_ttf/valuesAB are FCT-
 * internal scratch (C-overwritten == KK on Serial; no host reader after the FCT). */
void fesom_tracer_fct_verify(fesom_tracer_adv_scratch *sc, int tr_idx,
                             const struct fesom_mesh *mesh, const struct fesom_dyn *dyn,
                             struct fesom_tracers *tracers, struct fesom_partit *partit,
                             int step_n, const std::vector<real_t> &pre_values,
                             const std::vector<real_t> &pre_valuesold)
{
    (void)partit;
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    real_t *vals = tracers->data[tr_idx].values;
    real_t *vold = tracers->data[tr_idx].valuesold;
    std::vector<real_t> kk_v(vals, vals + total);          /* KK final values */
    std::vector<real_t> kk_vo(vold, vold + total);         /* KK valuesold (= pre_values) */
    std::copy(pre_values.begin(),    pre_values.end(),    vals);   /* restore FCT inputs */
    std::copy(pre_valuesold.begin(), pre_valuesold.end(), vold);
    fesom_tracer_advect_one_fct(sc, tr_idx, mesh, dyn, tracers);    /* C twin (uses sc->partit) */
    double d = 0.0;
    for (size_t i = 0; i < total; ++i) {
        double dd = std::fabs((double)kk_v[i] - (double)vals[i]);
        if (dd > d) d = dd;
    }
    std::copy(kk_v.begin(),  kk_v.end(),  vals);           /* restore KK production state */
    std::copy(kk_vo.begin(), kk_vo.end(), vold);
    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=tradv] step %d backend=%s  max|Δ|: fct(tr%d) values=%.3e\n",
                step_n, backend.c_str(), tr_idx, d);
    std::fflush(stdout);
    if (backend == "Serial" && d != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=tradv] FAIL (fct tr%d) step %d: Serial must be "
                             "bit-identical to the C twin (max|Δ|=%.3e)\n", tr_idx, step_n, d);
        std::abort();
    }
}
