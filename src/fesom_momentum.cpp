/*
 * Phase 1 momentum: literal port of compute_vel_rhs, impl_vert_visc_ale,
 * update_vel, compute_hbar_ale. Each routine follows the Fortran source
 * line-by-line; deferred branches are explicitly listed in fesom_momentum.h.
 */
#include "fesom_momentum.h"
#include "fesom_aux.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_forcing.h"
#include "fesom_ale.h"   // M6.3: fesom_ale_is_zstar()
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   // M5.1: GPU-aware-MPI on-device halo (fesom_halo_field)
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_speed.hpp"   // M7 C.3a: FESOM_SPEED_VISCNOINIT

#include <Kokkos_Core.hpp>   // M2.4: device kernels (parallel_for) + Kokkos:: math + atomic
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>            // M2.4: FESOM_KK_VERIFY snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>
#include <cstdio>

/*===========================================================================
 * compute_vel_rhs (oce_ale_vel_rhs.F90:35-328)
 *
 * Three sub-loops (we collapse the second AB-shift loop into the first since
 * Phase 1 uses AB_order=2, single-slot history):
 *
 * Per-element loop:
 *   ab1 = -(0.5 + epsilon),  ab2 = (1.5 + epsilon)        (lines 97-100)
 *   For each layer:
 *       UV_rhs[u] = ab1 * UV_rhsAB[u]      ← previous step's Coriolis
 *       UV_rhs[v] = ab1 * UV_rhsAB[v]
 *
 *   ff = coriolis(elem) * elem_area(elem)                  (line 157)
 *   p_eta = g * eta_n[3 vertices]                          (line 155)
 *   pre   = -p_eta                                         (Phase 1: no ice/air/tides)
 *   Fx, Fy = sum( gradient_sca[1..3 / 4..6] * pre )        (lines 188-189)
 *
 *   For each layer:
 *       UV_rhs[u] += (Fx - pgf_x) * elem_area              (lines 197-198)
 *       UV_rhs[v] += (Fy - pgf_y) * elem_area
 *       UV_rhsAB[u] =  v * ff      ← new Coriolis stored for AB next step
 *       UV_rhsAB[v] = -u * ff                              (lines 203-204)
 *
 * Final assembly loop (after the per-element loop completes):
 *   ff_step = ab2     (1.0 on the very first step, to match Fortran's lfirst)
 *   For each layer:
 *       UV_rhs[u] = dt * (UV_rhs[u] + UV_rhsAB[u] * ff_step) / elem_area
 *       UV_rhs[v] = dt * (UV_rhs[v] + UV_rhsAB[v] * ff_step) / elem_area
 *===========================================================================*/

void fesom_compute_vel_rhs(const struct fesom_mesh *mesh,
                           const struct fesom_aux  *aux,
                           struct fesom_dyn        *dyn,
                           int                      is_first_step,
                           struct fesom_partit     *partit)
{
    const int E       = mesh->myDim_elem2D;     /* interior only — halo gets exchange */
    const int nl      = mesh->nl;
    const real_t g    = (real_t)FESOM_G;
    const real_t dt   = (real_t)FESOM_PHASE1_DT;

    /* Adams-Bashforth coefficients (oce_ale_vel_rhs.F90:97-100). `epsilon` is
       FESOM's AB2 *stabilization offset* (o_PARAM, oce_modules.F90:92:
       epsilon=0.1_WP "AB2 offset"; no namelist override), NOT a tiny first-step
       epsilon. With ε=0.1 the AB2 of the Coriolis+advection is ab=(−0.6,1.6)
       instead of pure-AB2 (−0.5,1.5); the offset damps the oscillatory (Coriolis)
       mode, which pure AB2 leaves marginally unstable at high f·dt — the dt=1800
       central-Arctic (highest-latitude) velocity-2Δx blow-up (negligible at
       dt=500). Same offset is used in the tracer AB2 (fesom_tracer_adv.c). */
    const real_t eps = 0.1;
    const real_t ab1 = -(0.5 + eps);
    const real_t ab2 =  (1.5 + eps);

    /* Per-element computation: (i) shift AB history into UV_rhs,
       (ii) add SSH-gradient + PGF, (iii) overwrite UV_rhsAB with this step's
       Coriolis term. */
    for (int e = 0; e < E; ++e) {
        int nzmin = mesh->ulevels[e]   - 1;       /* 1-based → 0-based */
        int nzmax = mesh->nlevels[e]   - 1;
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];

        /* (i) AB history shift */
        for (int nz = nzmin; nz < nzmax; ++nz) {
            size_t k = FESOM_ELEMVEC(e, nz, nl);
            dyn->uv_rhs[k + 0] = ab1 * dyn->uv_rhsAB[k + 0];
            dyn->uv_rhs[k + 1] = ab1 * dyn->uv_rhsAB[k + 1];
        }

        /* SSH gradient (Phase 1: no ice / air / tides — pre = -p_eta) */
        real_t pre0 = -g * dyn->eta_n[n0];
        real_t pre1 = -g * dyn->eta_n[n1];
        real_t pre2 = -g * dyn->eta_n[n2];
        const real_t *gs = &mesh->gradient_sca[6 * e];
        real_t Fx = gs[0]*pre0 + gs[1]*pre1 + gs[2]*pre2;
        real_t Fy = gs[3]*pre0 + gs[4]*pre1 + gs[5]*pre2;

        real_t area = mesh->elem_area[e];
        real_t ff   = mesh->coriolis[e] * area;       /* Coriolis * area */

        /* (ii) and (iii) */
        for (int nz = nzmin; nz < nzmax; ++nz) {
            size_t k    = FESOM_ELEMVEC(e, nz, nl);
            size_t kel  = FESOM_ELEM3D(e, nz, nl);
            real_t pgfx = aux->pgf_x[kel];
            real_t pgfy = aux->pgf_y[kel];
            dyn->uv_rhs[k + 0] += (Fx - pgfx) * area;
            dyn->uv_rhs[k + 1] += (Fy - pgfy) * area;
            real_t u = dyn->uv[k + 0];
            real_t v = dyn->uv[k + 1];
            dyn->uv_rhsAB[k + 0] =  v * ff;
            dyn->uv_rhsAB[k + 1] = -u * ff;
        }
    }

    /* Momentum advection (momadv_opt=2): add ke_adv into the current AB slot
       (uv_rhsAB), exactly as Fortran does between the Coriolis loop and the AB2
       assembly (oce_ale_vel_rhs.F90:271-273). uv (incl. halo) and w_e are from
       the previous step, matching Fortran's lagged use. */
    fesom_momentum_adv_scalar(mesh, dyn, partit);

    /* Final assembly with ff_step. Fortran tracks lfirst as a `save` flag —
       caller passes is_first_step here. */
    real_t ff_step = is_first_step ? 1.0 : ab2;
    for (int e = 0; e < E; ++e) {
        int nzmin = mesh->ulevels[e]   - 1;
        int nzmax = mesh->nlevels[e]   - 1;
        real_t inv_area = 1.0 / mesh->elem_area[e];
        for (int nz = nzmin; nz < nzmax; ++nz) {
            size_t k = FESOM_ELEMVEC(e, nz, nl);
            dyn->uv_rhs[k + 0] = dt * (dyn->uv_rhs[k + 0]
                                     + dyn->uv_rhsAB[k + 0] * ff_step) * inv_area;
            dyn->uv_rhs[k + 1] = dt * (dyn->uv_rhs[k + 1]
                                     + dyn->uv_rhsAB[k + 1] * ff_step) * inv_area;
        }
    }
}

/*===========================================================================
 * momentum_adv_scalar (oce_ale_vel_rhs.F90:335-589) — momadv_opt=2.
 *
 * Literal port. Computes the momentum advection on scalar (vertex) control
 * volumes and ADDS it into uv_rhsAB (the current AB slot, which already holds
 * the Coriolis term), exactly as Fortran does between the Coriolis loop and
 * the AB2 assembly in compute_vel_rhs.
 *
 *   1. vertical advection  w·du/dz, w·dv/dz  → uvnode_rhs (per vertex)
 *   2. horizontal advection (u·du/dx + v·du/dy) via the edge loop → uvnode_rhs
 *   3. divide by scalar control-volume area (areasvol)
 *   4. exchange uvnode_rhs
 *   5. vertex → element: uv_rhsAB += elem_area·mean(3 vertices)
 *
 * Levels (0-based C): for node/elem x, top layer = ulevels[x]-1, bottom layer =
 * nlevels[x]-2; interfaces share the same index (interface nz = top of layer nz).
 * w_e is the vertical velocity at interfaces (lagged one step, as in Fortran).
 *===========================================================================*/
void fesom_momentum_adv_scalar(const struct fesom_mesh *mesh,
                               struct fesom_dyn        *dyn,
                               struct fesom_partit     *partit)
{
    const int nl = mesh->nl;
    const real_t *uv  = dyn->uv;
    real_t       *un  = dyn->uvnode_rhs;        /* [2*FESOM_NODE3D(n,nz,nl)+c] */
    const real_t *we  = dyn->w_e;
    enum { NL_MAX = FESOM_MAX_LEVELS };

    /* 1. vertical advection: w·du/dz at vertices (Fortran 369-420) */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        int ul = mesh->ulevels_nod2D[n] - 1;     /* 0-based top layer */
        int bl = mesh->nlevels_nod2D[n] - 2;     /* 0-based bottom layer */
        real_t wu[NL_MAX], wv[NL_MAX];
        for (int j = 0; j <= bl + 1; ++j) { wu[j] = 0.0; wv[j] = 0.0; }  /* interfaces */
        int b = mesh->nod_in_elem2D_offsets[n], e = mesh->nod_in_elem2D_offsets[n + 1];
        for (int k = b; k < e; ++k) {
            int el  = mesh->nod_in_elem2D[k];
            int ule = mesh->ulevels[el] - 1;     /* 0-based top layer of el */
            int ble = mesh->nlevels[el] - 2;     /* 0-based bottom layer of el */
            real_t a = mesh->elem_area[el];
            if (ule == 0) {                      /* non-cavity surface interface */
                wu[0] += uv[FESOM_ELEMVEC(el, 0, nl) + 0] * a;
                wv[0] += uv[FESOM_ELEMVEC(el, 0, nl) + 1] * a;
            }
            for (int j = ule + 1; j <= ble; ++j) {   /* interior interfaces */
                wu[j] += 0.5 * (uv[FESOM_ELEMVEC(el, j, nl) + 0] + uv[FESOM_ELEMVEC(el, j - 1, nl) + 0]) * a;
                wv[j] += 0.5 * (uv[FESOM_ELEMVEC(el, j, nl) + 1] + uv[FESOM_ELEMVEC(el, j - 1, nl) + 1]) * a;
            }
        }
        for (int j = ul; j <= bl; ++j) {         /* × vertical velocity at interfaces */
            real_t w = we[FESOM_NODE3D(n, j, nl)];
            wu[j] *= w; wv[j] *= w;
        }
        for (int nz = 0; nz < nl - 1; ++nz) {    /* init full column to 0 */
            un[FESOM_ELEMVEC(n, nz, nl) + 0] = 0.0;
            un[FESOM_ELEMVEC(n, nz, nl) + 1] = 0.0;
        }
        for (int nz = ul; nz <= bl; ++nz) {      /* w·du/dz; 1/3 from vertex area share */
            real_t h3 = 3.0 * mesh->hnode[FESOM_NODE3D(n, nz, nl)];
            un[FESOM_ELEMVEC(n, nz, nl) + 0] = -(wu[nz] - wu[nz + 1]) / h3;
            un[FESOM_ELEMVEC(n, nz, nl) + 1] = -(wv[nz] - wv[nz + 1]) / h3;
        }
    }

    /* 2. horizontal advection via edge loop (Fortran 427-544) */
    for (int ed = 0; ed < mesh->myDim_edge2D; ++ed) {
        int n1 = mesh->edges[2*ed + 0], n2 = mesh->edges[2*ed + 1];
        int el1 = mesh->edge_tri[2*ed + 0], el2 = mesh->edge_tri[2*ed + 1];
        if (el1 < 0) continue;
        int ul1 = mesh->ulevels[el1] - 1, bl1 = mesh->nlevels[el1] - 2;
        real_t dx1 = mesh->edge_cross_dxdy[4*ed + 0], dy1 = mesh->edge_cross_dxdy[4*ed + 1];
        real_t un1[NL_MAX], un2[NL_MAX];
        int my = mesh->myDim_nod2D;
        if (el2 >= 0) {
            int ul2 = mesh->ulevels[el2] - 1, bl2 = mesh->nlevels[el2] - 2;
            real_t dx2 = mesh->edge_cross_dxdy[4*ed + 2], dy2 = mesh->edge_cross_dxdy[4*ed + 3];
            int lo = ul1 < ul2 ? ul1 : ul2, hi = bl1 > bl2 ? bl1 : bl2;
            for (int nz = lo; nz <= hi; ++nz) { un1[nz] = 0.0; un2[nz] = 0.0; }
            for (int nz = ul1; nz <= bl1; ++nz)
                un1[nz] =  uv[FESOM_ELEMVEC(el1,nz,nl)+1]*dx1 - uv[FESOM_ELEMVEC(el1,nz,nl)+0]*dy1;
            for (int nz = ul2; nz <= bl2; ++nz)
                un2[nz] = -uv[FESOM_ELEMVEC(el2,nz,nl)+1]*dx2 + uv[FESOM_ELEMVEC(el2,nz,nl)+0]*dy2;
            if (n1 < my)
                for (int nz = lo; nz <= hi; ++nz) {
                    un[FESOM_ELEMVEC(n1,nz,nl)+0] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+0];
                    un[FESOM_ELEMVEC(n1,nz,nl)+1] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+1];
                }
            if (n2 < my)
                for (int nz = lo; nz <= hi; ++nz) {
                    un[FESOM_ELEMVEC(n2,nz,nl)+0] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+0];
                    un[FESOM_ELEMVEC(n2,nz,nl)+1] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+1];
                }
        } else {     /* boundary edge — only el1 contributes */
            for (int nz = ul1; nz <= bl1; ++nz)
                un1[nz] = uv[FESOM_ELEMVEC(el1,nz,nl)+1]*dx1 - uv[FESOM_ELEMVEC(el1,nz,nl)+0]*dy1;
            if (n1 < my)
                for (int nz = ul1; nz <= bl1; ++nz) {
                    un[FESOM_ELEMVEC(n1,nz,nl)+0] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0];
                    un[FESOM_ELEMVEC(n1,nz,nl)+1] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1];
                }
            if (n2 < my)
                for (int nz = ul1; nz <= bl1; ++nz) {
                    un[FESOM_ELEMVEC(n2,nz,nl)+0] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0];
                    un[FESOM_ELEMVEC(n2,nz,nl)+1] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1];
                }
        }
    }

    /* 3. divide by scalar control-volume area (Fortran 550-555) */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        int ul = mesh->ulevels_nod2D[n] - 1, bl = mesh->nlevels_nod2D[n] - 2;
        for (int nz = ul; nz <= bl; ++nz) {
            real_t inv = 1.0 / mesh->areasvol[FESOM_NODE3D(n, nz, nl)];
            un[FESOM_ELEMVEC(n,nz,nl)+0] *= inv;
            un[FESOM_ELEMVEC(n,nz,nl)+1] *= inv;
        }
    }

    /* 4. exchange uvnode_rhs (Fortran 559) */
    fesom_halo_exchange(un, FESOM_HALO_NOD3D, nl, 2, partit);

    /* 5. vertex → element: uv_rhsAB += elem_area·mean(3 vertices) (Fortran 565-573) */
    for (int el = 0; el < mesh->myDim_elem2D; ++el) {
        int ul = mesh->ulevels[el] - 1, bl = mesh->nlevels[el] - 2;
        int v0 = mesh->elem_nodes[3*el+0], v1 = mesh->elem_nodes[3*el+1], v2 = mesh->elem_nodes[3*el+2];
        real_t a = mesh->elem_area[el];
        for (int nz = ul; nz <= bl; ++nz) {
            dyn->uv_rhsAB[FESOM_ELEMVEC(el,nz,nl)+0] += a * (un[FESOM_ELEMVEC(v0,nz,nl)+0]
                + un[FESOM_ELEMVEC(v1,nz,nl)+0] + un[FESOM_ELEMVEC(v2,nz,nl)+0]) / 3.0;
            dyn->uv_rhsAB[FESOM_ELEMVEC(el,nz,nl)+1] += a * (un[FESOM_ELEMVEC(v0,nz,nl)+1]
                + un[FESOM_ELEMVEC(v1,nz,nl)+1] + un[FESOM_ELEMVEC(v2,nz,nl)+1]) / 3.0;
        }
    }
}

/*--- momentum_adv_scalar — DEVICE kernel (M2.4) ------------------------------
 * Five stages mirroring the C twin (momadv_opt=2), each a parallel_for; the launch
 * barriers give the inter-stage ordering (D20). Stages 1/3/5 are race-free (each
 * entity writes only its own slot — a private gather); stage 2 is an EDGE→NODE
 * SCATTER ported with Kokkos::atomic_add (D22 — Serial bit-identical in edge order,
 * OpenMP/CUDA climate-close). Stage 4 is the INTERNAL uvnode_rhs nod3D halo bracket
 * (D21, owned here). Per-stage scratch (wu/wv, un1/un2, each [64]) is lambda-local.
 * Every bound/constant/association is a verbatim copy of fesom_momentum_adv_scalar.
 */
void fesom_momentum_adv_scalar_kk(const struct fesom_mesh *mesh,
                                  struct fesom_dyn        *dyn,
                                  struct fesom_partit     *partit)
{
    const int nl       = mesh->nl;
    const int Nmy      = mesh->myDim_nod2D;
    const int Emy_edge = mesh->myDim_edge2D;
    const int Emy      = mesh->myDim_elem2D;
    const int my       = mesh->myDim_nod2D;

    auto uv         = dyn->uv_fld.d();
    auto un         = dyn->uvnode_rhs_fld.d();
    auto we         = dyn->w_e_fld.d();
    auto uv_rhsAB   = dyn->uv_rhsAB_fld.d();
    auto ulev_n     = mesh->ulevels_nod2D_fld.d();
    auto nlev_n     = mesh->nlevels_nod2D_fld.d();
    auto ulev_e     = mesh->ulevels_fld.d();
    auto nlev_e     = mesh->nlevels_fld.d();
    auto area       = mesh->elem_area_fld.d();
    auto nie        = mesh->nod_in_elem2D_fld.d();
    auto nie_off    = mesh->nod_in_elem2D_offsets_fld.d();
    auto hnode      = mesh->hnode_fld.d();
    auto edges      = mesh->edges_fld.d();
    auto edge_tri   = mesh->edge_tri_fld.d();
    auto edge_cross = mesh->edge_cross_dxdy_fld.d();
    auto areasvol   = mesh->areasvol_fld.d();
    auto elnod      = mesh->elem_nodes_fld.d();

    /* 1. vertical advection: w·du/dz at vertices (Fortran 369-420). Private gather. */
    Kokkos::parallel_for("fesom_momadv_vert", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int ul = ulev_n(n) - 1;
            int bl = nlev_n(n) - 2;
            real_t wu[FESOM_MAX_LEVELS], wv[FESOM_MAX_LEVELS];
            for (int j = 0; j <= bl + 1; ++j) { wu[j] = 0.0; wv[j] = 0.0; }
            int b = nie_off(n), e = nie_off(n + 1);
            for (int k = b; k < e; ++k) {
                int el  = nie(k);
                int ule = ulev_e(el) - 1;
                int ble = nlev_e(el) - 2;
                real_t a = area(el);
                if (ule == 0) {
                    wu[0] += uv(FESOM_ELEMVEC(el, 0, nl) + 0) * a;
                    wv[0] += uv(FESOM_ELEMVEC(el, 0, nl) + 1) * a;
                }
                for (int j = ule + 1; j <= ble; ++j) {
                    wu[j] += 0.5 * (uv(FESOM_ELEMVEC(el, j, nl) + 0) + uv(FESOM_ELEMVEC(el, j - 1, nl) + 0)) * a;
                    wv[j] += 0.5 * (uv(FESOM_ELEMVEC(el, j, nl) + 1) + uv(FESOM_ELEMVEC(el, j - 1, nl) + 1)) * a;
                }
            }
            for (int j = ul; j <= bl; ++j) {
                real_t w = we(FESOM_NODE3D(n, j, nl));
                wu[j] *= w; wv[j] *= w;
            }
            for (int nz = 0; nz < nl - 1; ++nz) {
                un(FESOM_ELEMVEC(n, nz, nl) + 0) = 0.0;
                un(FESOM_ELEMVEC(n, nz, nl) + 1) = 0.0;
            }
            for (int nz = ul; nz <= bl; ++nz) {
                real_t h3 = 3.0 * hnode(FESOM_NODE3D(n, nz, nl));
                un(FESOM_ELEMVEC(n, nz, nl) + 0) = -(wu[nz] - wu[nz + 1]) / h3;
                un(FESOM_ELEMVEC(n, nz, nl) + 1) = -(wv[nz] - wv[nz + 1]) / h3;
            }
        });

    /* 2. horizontal advection via edge loop (Fortran 427-544) — EDGE→NODE SCATTER (atomic_add).
       The C re-evaluates the same `un1*uv[el1] + un2*uv[el2]` for the n1 (+=) and n2 (-=) writes;
       the n2 atomic adds its negation (a - X == a + (-X), IEEE-exact). */
    Kokkos::parallel_for("fesom_momadv_horiz", Kokkos::RangePolicy<>(0, Emy_edge),
        KOKKOS_LAMBDA(const int ed) {
            int n1 = edges(2*ed + 0), n2 = edges(2*ed + 1);
            int el1 = edge_tri(2*ed + 0), el2 = edge_tri(2*ed + 1);
            if (el1 < 0) return;
            int ul1 = ulev_e(el1) - 1, bl1 = nlev_e(el1) - 2;
            real_t dx1 = edge_cross(4*ed + 0), dy1 = edge_cross(4*ed + 1);
            real_t un1[FESOM_MAX_LEVELS], un2[FESOM_MAX_LEVELS];
            if (el2 >= 0) {
                int ul2 = ulev_e(el2) - 1, bl2 = nlev_e(el2) - 2;
                real_t dx2 = edge_cross(4*ed + 2), dy2 = edge_cross(4*ed + 3);
                int lo = ul1 < ul2 ? ul1 : ul2, hi = bl1 > bl2 ? bl1 : bl2;
                for (int nz = lo; nz <= hi; ++nz) { un1[nz] = 0.0; un2[nz] = 0.0; }
                for (int nz = ul1; nz <= bl1; ++nz)
                    un1[nz] =  uv(FESOM_ELEMVEC(el1,nz,nl)+1)*dx1 - uv(FESOM_ELEMVEC(el1,nz,nl)+0)*dy1;
                for (int nz = ul2; nz <= bl2; ++nz)
                    un2[nz] = -uv(FESOM_ELEMVEC(el2,nz,nl)+1)*dx2 + uv(FESOM_ELEMVEC(el2,nz,nl)+0)*dy2;
                if (n1 < my)
                    for (int nz = lo; nz <= hi; ++nz) {
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n1,nz,nl)+0),
                            un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+0) + un2[nz]*uv(FESOM_ELEMVEC(el2,nz,nl)+0));
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n1,nz,nl)+1),
                            un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+1) + un2[nz]*uv(FESOM_ELEMVEC(el2,nz,nl)+1));
                    }
                if (n2 < my)
                    for (int nz = lo; nz <= hi; ++nz) {
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n2,nz,nl)+0),
                            -(un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+0) + un2[nz]*uv(FESOM_ELEMVEC(el2,nz,nl)+0)));
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n2,nz,nl)+1),
                            -(un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+1) + un2[nz]*uv(FESOM_ELEMVEC(el2,nz,nl)+1)));
                    }
            } else {     /* boundary edge — only el1 contributes */
                for (int nz = ul1; nz <= bl1; ++nz)
                    un1[nz] = uv(FESOM_ELEMVEC(el1,nz,nl)+1)*dx1 - uv(FESOM_ELEMVEC(el1,nz,nl)+0)*dy1;
                if (n1 < my)
                    for (int nz = ul1; nz <= bl1; ++nz) {
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n1,nz,nl)+0), un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+0));
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n1,nz,nl)+1), un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+1));
                    }
                if (n2 < my)
                    for (int nz = ul1; nz <= bl1; ++nz) {
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n2,nz,nl)+0), -(un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+0)));
                        Kokkos::atomic_add(&un(FESOM_ELEMVEC(n2,nz,nl)+1), -(un1[nz]*uv(FESOM_ELEMVEC(el1,nz,nl)+1)));
                    }
            }
        });

    /* 3. divide by scalar control-volume area (Fortran 550-555). Per-node, race-free.
     * M5.19 bucket-A coalescing flip: one thread per (node, LEVEL) — un is node-major
     * (FESOM_ELEMVEC(n,nz)=n*nl*2+nz*2) so the level is the contiguous inner dim → the
     * per-(n,nz) warp coalesces (vs the old one-thread-per-node strided-by-nl*2 store).
     * Pure per-level map (no scratch/coupling) → byte-identical. */
    const int NmyNL_a = Nmy * nl;
    Kokkos::parallel_for("fesom_momadv_area", Kokkos::RangePolicy<>(0, NmyNL_a),
        KOKKOS_LAMBDA(const int idx) {
            const int n  = idx / nl;
            const int nz = idx - n * nl;
            int ul = ulev_n(n) - 1, bl = nlev_n(n) - 2;
            if (nz < ul || nz > bl) return;            /* [ul, bl] inclusive — matches the C loop */
            real_t inv = 1.0 / areasvol(FESOM_NODE3D(n, nz, nl));
            un(FESOM_ELEMVEC(n,nz,nl)+0) *= inv;
            un(FESOM_ELEMVEC(n,nz,nl)+1) *= inv;
        });

    /* 4. exchange uvnode_rhs (Fortran 559) — INTERNAL nod3D halo bracket (D21).
       Device-compute → halo → device-compute. M5.1: fesom_halo_field dispatches to
       GPU-aware-MPI on-device exchange under CUDA; host-staged on Serial/OpenMP. */
    fesom_halo_field(dyn->uvnode_rhs_fld, FESOM_HALO_NOD3D, nl, 2, partit);

    /* 5. vertex → element: uv_rhsAB += elem_area·mean(3 vertices) (Fortran 565-573).
       Per-element gather (each element writes only its own uv_rhsAB) → race-free, no atomic. */
    /* M5.19 bucket-A coalescing flip: one thread per (elem, LEVEL) — uv_rhsAB store and the
     * 3 vertex un reads are level-contiguous per element → coalesced; `a` is a per-element
     * scalar (warp-broadcast). Each (el,nz) is touched by one thread → byte-identical. */
    const int EmyNL = Emy * nl;
    Kokkos::parallel_for("fesom_momadv_v2e", Kokkos::RangePolicy<>(0, EmyNL),
        KOKKOS_LAMBDA(const int idx) {
            const int el = idx / nl;
            const int nz = idx - el * nl;
            int ul = ulev_e(el) - 1, bl = nlev_e(el) - 2;
            if (nz < ul || nz > bl) return;
            int v0 = elnod(3*el+0), v1 = elnod(3*el+1), v2 = elnod(3*el+2);
            real_t a = area(el);
            uv_rhsAB(FESOM_ELEMVEC(el,nz,nl)+0) += a * (un(FESOM_ELEMVEC(v0,nz,nl)+0)
                + un(FESOM_ELEMVEC(v1,nz,nl)+0) + un(FESOM_ELEMVEC(v2,nz,nl)+0)) / 3.0;
            uv_rhsAB(FESOM_ELEMVEC(el,nz,nl)+1) += a * (un(FESOM_ELEMVEC(v0,nz,nl)+1)
                + un(FESOM_ELEMVEC(v1,nz,nl)+1) + un(FESOM_ELEMVEC(v2,nz,nl)+1)) / 3.0;
        });
}

/*--- compute_vel_rhs — DEVICE kernel (M2.4) ----------------------------------
 * ⚠️ AB2 eps=0.1 (PORTING_LESSONS §1, the dt=1800 stabilization offset) preserved
 * verbatim: ab1=-(0.5+eps), ab2=(1.5+eps). Two race-free per-element parallel_fors
 * (the Coriolis/SSH-grad/PGF body, then the AB2 assembly) bracket the device
 * momentum-advection call (fesom_momentum_adv_scalar_kk, which holds the edge scatter
 * + the internal uvnode_rhs halo). uv_rhs/uv_rhsAB stay device-resident across the
 * three pieces (launch barriers order them). Every constant/bound/association is a
 * verbatim copy of fesom_compute_vel_rhs. SYNC: INPUTS device-current (driver IN rail);
 * marks uv_rhs/uv_rhsAB modify_device().
 */
void fesom_compute_vel_rhs_kk(const struct fesom_mesh *mesh,
                              const struct fesom_aux  *aux,
                              struct fesom_dyn        *dyn,
                              int                      is_first_step,
                              struct fesom_partit     *partit)
{
    const int E    = mesh->myDim_elem2D;
    const int nl   = mesh->nl;
    const real_t g  = (real_t)FESOM_G;
    const real_t dt = (real_t)FESOM_PHASE1_DT;
    const real_t eps = 0.1;                  /* AB2 stabilization offset (PORTING_LESSONS §1) */
    const real_t ab1 = -(0.5 + eps);
    const real_t ab2 =  (1.5 + eps);

    auto ulev     = mesh->ulevels_fld.d();
    auto nlev     = mesh->nlevels_fld.d();
    auto elnod    = mesh->elem_nodes_fld.d();
    auto gradsca  = mesh->gradient_sca_fld.d();
    auto area     = mesh->elem_area_fld.d();
    auto coriolis = mesh->coriolis_fld.d();
    auto uv       = dyn->uv_fld.d();
    auto uv_rhs   = dyn->uv_rhs_fld.d();
    auto uv_rhsAB = dyn->uv_rhsAB_fld.d();
    auto eta_n    = dyn->eta_n_fld.d();
    auto pgf_x    = aux->pgf_x_fld.d();
    auto pgf_y    = aux->pgf_y_fld.d();

    /* M5.19 (bucket-A coalescing lever, the M5.18 flat-lever recipe): ONE THREAD PER
     * (elem, LEVEL) — flat RangePolicy(0, E*nl) decoded to (e,nz). uv_rhs/uv_rhsAB are
     * element-major (FESOM_ELEMVEC = e*nl*2 + nz*2) and pgf_x/pgf_y are FESOM_ELEM3D =
     * e*nl + nz, so the LEVEL is the contiguous inner dim: a warp of 32 consecutive flat
     * idx = e*nl + nz spans 32 consecutive LEVELS of the same element → every uv_rhs/
     * uv_rhsAB store (stride-2 component = 64 contiguous doubles/warp) and every pgf load
     * is CONTIGUOUS → COALESCED, and the per-(elem,depth) divergence vanishes (one level/
     * thread). The old M2.4 mapping was one-thread-per-ELEMENT with an internal nz loop →
     * consecutive threads = consecutive ELEMENTS → every store strided by nl*2=140 =
     * uncoalesced (the same disease the M5.18 smoother had at 25.7% of GPU compute). The
     * per-element scalars (pre0..2, Fx, Fy, areav, ff) recompute per thread; within a warp
     * all threads share ONE element so eta_n/gradsca/area/coriolis are read once
     * (warp-broadcast) → cheap (a few FLOPs, not nl× the memory). The per-(e,nz) arithmetic
     * runs in the SAME statement order as the C twin (AB-shift store, THEN += PGF), IEEE
     * op-for-op, and each thread reads-then-writes only its OWN k (no cross-level/-thread
     * coupling) → byte-identical → Serial AND OpenMP bit-identical (no scatter); CUDA
     * climate-close. Wasted shallow-column tail (avg depth < nl) is acceptable for a
     * memory-bound kernel — coalescing + occupancy dominate. See docs/GPU_FIDELITY.md §M5.19. */
    const int Eonl = E * nl;        /* E*nl < INT_MAX for any real rank */
    Kokkos::parallel_for("fesom_vel_rhs_elem", Kokkos::RangePolicy<>(0, Eonl),
        KOKKOS_LAMBDA(const int idx) {
            const int e  = idx / nl;
            const int nz = idx - e * nl;
            const int nzmin = ulev(e) - 1;
            const int nzmax = nlev(e) - 1;
            if (nz < nzmin || nz >= nzmax) return;   /* [nzmin, nzmax) — matches the C loop bound */

            int n0 = elnod(3*e + 0);
            int n1 = elnod(3*e + 1);
            int n2 = elnod(3*e + 2);

            size_t k = FESOM_ELEMVEC(e, nz, nl);
            uv_rhs(k + 0) = ab1 * uv_rhsAB(k + 0);              /* (i) AB history shift */
            uv_rhs(k + 1) = ab1 * uv_rhsAB(k + 1);

            real_t pre0 = -g * eta_n(n0);
            real_t pre1 = -g * eta_n(n1);
            real_t pre2 = -g * eta_n(n2);
            real_t gs0 = gradsca(6*e+0), gs1 = gradsca(6*e+1), gs2 = gradsca(6*e+2);
            real_t gs3 = gradsca(6*e+3), gs4 = gradsca(6*e+4), gs5 = gradsca(6*e+5);
            real_t Fx = gs0*pre0 + gs1*pre1 + gs2*pre2;
            real_t Fy = gs3*pre0 + gs4*pre1 + gs5*pre2;
            real_t areav = area(e);
            real_t ff    = coriolis(e) * areav;

            size_t kel = FESOM_ELEM3D(e, nz, nl);               /* (ii) and (iii) */
            real_t pgfx = pgf_x(kel);
            real_t pgfy = pgf_y(kel);
            uv_rhs(k + 0) += (Fx - pgfx) * areav;
            uv_rhs(k + 1) += (Fy - pgfy) * areav;
            real_t u = uv(k + 0);
            real_t v = uv(k + 1);
            uv_rhsAB(k + 0) =  v * ff;
            uv_rhsAB(k + 1) = -u * ff;
        });

    /* Momentum advection (momadv_opt=2): adds ke_adv into uv_rhsAB (device; edge scatter
       + internal uvnode_rhs halo, D21). uv_rhsAB stays device-resident across this. */
    fesom_momentum_adv_scalar_kk(mesh, dyn, partit);

    /* Final assembly with ff_step (1.0 on the first step to match Fortran's lfirst). Race-free. */
    const real_t ff_step = is_first_step ? 1.0 : ab2;
    /* M5.19 bucket-A: same flat (elem,LEVEL) coalescing flip as fesom_vel_rhs_elem above —
     * pure per-level map (inv_area is the only per-element scalar, one division/thread,
     * warp-broadcast inputs). Same expression, op-for-op → byte-identical. */
    Kokkos::parallel_for("fesom_vel_rhs_assembly", Kokkos::RangePolicy<>(0, Eonl),
        KOKKOS_LAMBDA(const int idx) {
            const int e  = idx / nl;
            const int nz = idx - e * nl;
            const int nzmin = ulev(e) - 1;
            const int nzmax = nlev(e) - 1;
            if (nz < nzmin || nz >= nzmax) return;
            real_t inv_area = 1.0 / area(e);
            size_t k = FESOM_ELEMVEC(e, nz, nl);
            uv_rhs(k + 0) = dt * (uv_rhs(k + 0) + uv_rhsAB(k + 0) * ff_step) * inv_area;
            uv_rhs(k + 1) = dt * (uv_rhs(k + 1) + uv_rhsAB(k + 1) * ff_step) * inv_area;
        });

    dyn->uv_rhs_fld.modify_device();
    dyn->uv_rhsAB_fld.modify_device();
}

/*--- FESOM_KK_VERIFY=vrhs (M2.4) ---------------------------------------------
 * compute_vel_rhs reads uv_rhsAB (part i) then overwrites it (part iii + momentum_adv
 * +=), so the L26 capture-before is on uv_rhsAB only — uv_rhs is fully recomputed from
 * uv_rhsAB + intact inputs (never read as a pre-existing input), so it needs no restore.
 * The driver captures the PRE-kernel uv_rhsAB (uv_rhsAB_in, owned). Here: snapshot the
 * Kokkos uv_rhs + uv_rhsAB, restore uv_rhsAB_in, run the C twin (recomputes both from the
 * restored uv_rhsAB + intact uv/eta_n/pgf/w_e/hnode), diff both over the owned region,
 * restore Kokkos. Non-intrusive; max|Δ|==0 on Serial.
 */
void fesom_compute_vel_rhs_verify(const struct fesom_mesh *mesh,
                                  const struct fesom_aux  *aux,
                                  struct fesom_dyn        *dyn,
                                  int is_first_step, struct fesom_partit *partit,
                                  int step_n, const real_t *uv_rhsAB_in)
{
    const int nl = mesh->nl;
    const size_t nvec = (size_t)mesh->myDim_elem2D * (size_t)nl * 2;   /* owned elements */

    std::vector<real_t> kk_uv_rhs  (dyn->uv_rhs,   dyn->uv_rhs   + nvec);
    std::vector<real_t> kk_uv_rhsAB(dyn->uv_rhsAB, dyn->uv_rhsAB + nvec);
    std::copy(uv_rhsAB_in, uv_rhsAB_in + nvec, dyn->uv_rhsAB);          /* restore the input part i reads */
    fesom_compute_vel_rhs(mesh, aux, dyn, is_first_step, partit);       /* C twin */

    auto maxdiff = [](const std::vector<real_t> &kk, const real_t *c, size_t n) -> double {
        double m = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs((double)kk[i] - (double)c[i]);
            if (d > m) m = d;
        }
        return m;
    };
    double d_rhs = maxdiff(kk_uv_rhs,   dyn->uv_rhs,   nvec);
    double d_AB  = maxdiff(kk_uv_rhsAB, dyn->uv_rhsAB, nvec);
    std::copy(kk_uv_rhs.begin(),   kk_uv_rhs.end(),   dyn->uv_rhs);     /* restore KK */
    std::copy(kk_uv_rhsAB.begin(), kk_uv_rhsAB.end(), dyn->uv_rhsAB);

    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    const double dmax = std::max(d_rhs, d_AB);
    std::printf("[FESOM_KK_VERIFY=vrhs] step %d backend=%s  max|Δ|: uv_rhs=%.3e uv_rhsAB=%.3e\n",
                step_n, backend.c_str(), d_rhs, d_AB);
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=vrhs] FAIL: Serial must be bit-identical to the C twin "
                             "(max|Δ|=%.3e)\n", dmax);
        std::abort();
    }
}

/*===========================================================================
 * impl_vert_visc_ale (oce_ale.F90:3124-3335)
 *
 * Per-element TDMA on (u, v) columns:
 *   - Build tridiagonal coefficients a, b, c using Av and helem-derived dz
 *   - Add advective contribution from Wvel_i (zero in Phase 1; see fesom_dyn.h)
 *   - Surface row: add wind stress / density_0 * zinv to RHS
 *   - Bottom row: add bottom drag friction * UV * zinv to RHS
 *   - Convert RHS from "u_new" formulation to "du = u_new - u_old"
 *   - Thomas (sweep) algorithm
 *   - Write result back into UV_rhs
 *
 * For Phase 1 (no partial cells), zbar_n[nz] = zbar[nz] for all elements,
 * Z_n[nz] = Z[nz] = 0.5*(zbar[nz]+zbar[nz+1]). But Fortran rebuilds zbar_n /
 * Z_n per element from helem (so partial-cell-aware). We do the same to keep
 * the code structurally identical.
 *===========================================================================*/

void fesom_impl_vert_visc(const struct fesom_mesh    *mesh,
                          const struct fesom_aux     *aux,
                          const struct fesom_forcing *forcing,
                          struct fesom_dyn           *dyn)
{
    const int E      = mesh->myDim_elem2D;
    const int nl     = mesh->nl;
    const real_t dt  = (real_t)FESOM_PHASE1_DT;
    const real_t Cd  = (real_t)FESOM_PHASE1_C_D;
    const real_t inv_density_0 = 1.0 / (real_t)FESOM_DENSITY_0;

    /* Per-column scratch — nl ≤ 48 so static cap of 64 is safe. */
    enum { NL_MAX = FESOM_MAX_LEVELS };

    for (int e = 0; e < E; ++e) {
        int nzmin = mesh->ulevels[e]   - 1;
        int nzmax = mesh->nlevels[e]   - 1;
        if (nzmax - nzmin < 1) continue;
        FESOM_CHECK(nzmax + 1 < NL_MAX, "impl_vert_visc: nzmax %d > NL_MAX", nzmax);

        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];

        real_t zbar_n[NL_MAX], Z_n[NL_MAX];
        real_t a[NL_MAX], b[NL_MAX], c[NL_MAX];
        real_t ur[NL_MAX], vr[NL_MAX];
        real_t cp[NL_MAX], up[NL_MAX], vp[NL_MAX];

        /* Build zbar_n and Z_n from helem upward — mirror of lines 3167-3179.
           In 0-based C: nzmax is the bottom-interface index (Fortran nlevels-1).
           helem is layer thickness; zbar_n[nz] is the interface above layer nz. */
        for (int k = 0; k <= nzmax; ++k) zbar_n[k] = 0.0;
        for (int k = 0; k < nzmax; ++k)  Z_n   [k] = 0.0;
        zbar_n[nzmax]   = mesh->zbar[nzmax];   /* in Phase 1 = zbar_e_bot[e] */
        Z_n   [nzmax-1] = zbar_n[nzmax]
                        + mesh->helem[FESOM_ELEM3D(e, nzmax - 1, nl)] / 2.0;
        for (int nz = nzmax - 1; nz >= nzmin + 1; --nz) {
            zbar_n[nz]   = zbar_n[nz + 1] + mesh->helem[FESOM_ELEM3D(e, nz,     nl)];
            Z_n   [nz-1] = zbar_n[nz]     + mesh->helem[FESOM_ELEM3D(e, nz - 1, nl)] / 2.0;
        }
        zbar_n[nzmin] = zbar_n[nzmin + 1] + mesh->helem[FESOM_ELEM3D(e, nzmin, nl)];

        /* Tridiagonal coefficients: regular interior (lines 3185-3198) */
        for (int nz = nzmin + 1; nz <= nzmax - 2; ++nz) {
            real_t zinv = dt / (zbar_n[nz] - zbar_n[nz + 1]);
            a[nz] = -aux->Av[FESOM_ELEM3D(e, nz,     nl)] / (Z_n[nz - 1] - Z_n[nz]) * zinv;
            c[nz] = -aux->Av[FESOM_ELEM3D(e, nz + 1, nl)] / (Z_n[nz] - Z_n[nz + 1]) * zinv;
            b[nz] = -a[nz] - c[nz] + 1.0;
            /* Vertical advection update from Wvel_i (zero in Phase 1) */
            real_t wu = (dyn->w_i[FESOM_NODE3D(n0, nz,     nl)]
                       + dyn->w_i[FESOM_NODE3D(n1, nz,     nl)]
                       + dyn->w_i[FESOM_NODE3D(n2, nz,     nl)]) / 3.0;
            real_t wd = (dyn->w_i[FESOM_NODE3D(n0, nz + 1, nl)]
                       + dyn->w_i[FESOM_NODE3D(n1, nz + 1, nl)]
                       + dyn->w_i[FESOM_NODE3D(n2, nz + 1, nl)]) / 3.0;
            a[nz] = a[nz] + (wu < 0.0 ? wu : 0.0) * zinv;
            b[nz] = b[nz] + (wu > 0.0 ? wu : 0.0) * zinv;
            b[nz] = b[nz] - (wd < 0.0 ? wd : 0.0) * zinv;
            c[nz] = c[nz] - (wd > 0.0 ? wd : 0.0) * zinv;
        }

        /* Bottom row (lines 3200-3208) */
        {
            int nz = nzmax - 1;
            real_t zinv = dt / (zbar_n[nz] - zbar_n[nz + 1]);
            a[nz] = -aux->Av[FESOM_ELEM3D(e, nz, nl)] / (Z_n[nz - 1] - Z_n[nz]) * zinv;
            b[nz] = -a[nz] + 1.0;
            c[nz] = 0.0;
            real_t wu = (dyn->w_i[FESOM_NODE3D(n0, nz, nl)]
                       + dyn->w_i[FESOM_NODE3D(n1, nz, nl)]
                       + dyn->w_i[FESOM_NODE3D(n2, nz, nl)]) / 3.0;
            a[nz] = a[nz] + (wu < 0.0 ? wu : 0.0) * zinv;
            b[nz] = b[nz] + (wu > 0.0 ? wu : 0.0) * zinv;
        }

        /* Surface row (lines 3215-3231); zinv kept for the wind-stress RHS use */
        real_t zinv_top = dt / (zbar_n[nzmin] - zbar_n[nzmin + 1]);
        {
            int nz = nzmin;
            c[nz] = -aux->Av[FESOM_ELEM3D(e, nz + 1, nl)] / (Z_n[nz] - Z_n[nz + 1]) * zinv_top;
            a[nz] = 0.0;
            b[nz] = -c[nz] + 1.0;
            real_t wu = (dyn->w_i[FESOM_NODE3D(n0, nz,     nl)]
                       + dyn->w_i[FESOM_NODE3D(n1, nz,     nl)]
                       + dyn->w_i[FESOM_NODE3D(n2, nz,     nl)]) / 3.0;
            real_t wd = (dyn->w_i[FESOM_NODE3D(n0, nz + 1, nl)]
                       + dyn->w_i[FESOM_NODE3D(n1, nz + 1, nl)]
                       + dyn->w_i[FESOM_NODE3D(n2, nz + 1, nl)]) / 3.0;
            b[nz] = b[nz] + wu * zinv_top;
            b[nz] = b[nz] - (wd < 0.0 ? wd : 0.0) * zinv_top;
            c[nz] = c[nz] - (wd > 0.0 ? wd : 0.0) * zinv_top;
        }

        /* Build RHS from current uv_rhs (lines 3238-3245) */
        for (int nz = nzmin; nz <= nzmax - 1; ++nz) {
            ur[nz] = dyn->uv_rhs[FESOM_ELEMVEC(e, nz, nl) + 0];
            vr[nz] = dyn->uv_rhs[FESOM_ELEMVEC(e, nz, nl) + 1];
        }

        /* Surface forcing (wind stress); element wind stress at index 2*e */
        ur[nzmin] += zinv_top * forcing->stress_surf[2*e + 0] * inv_density_0;
        vr[nzmin] += zinv_top * forcing->stress_surf[2*e + 1] * inv_density_0;

        /* Bottom drag (lines 3267-3272). Phase 1: not toy_ocean, so quadratic
           drag from the bottom-layer velocity magnitude. */
        real_t zinv_bot = dt / (zbar_n[nzmax - 1] - zbar_n[nzmax]);
        {
            int nz = nzmax - 1;
            real_t u_bot = dyn->uv[FESOM_ELEMVEC(e, nz, nl) + 0];
            real_t v_bot = dyn->uv[FESOM_ELEMVEC(e, nz, nl) + 1];
            real_t spd   = sqrt(u_bot * u_bot + v_bot * v_bot);
            real_t friction = -Cd * spd;
            ur[nz] += zinv_bot * friction * u_bot;
            vr[nz] += zinv_bot * friction * v_bot;
        }

        /* Convert "solve for u_new" to "solve for du = u_new - u_old"
           (lines 3282-3292). */
        for (int nz = nzmin + 1; nz <= nzmax - 2; ++nz) {
            real_t u_up = dyn->uv[FESOM_ELEMVEC(e, nz - 1, nl) + 0];
            real_t v_up = dyn->uv[FESOM_ELEMVEC(e, nz - 1, nl) + 1];
            real_t u_th = dyn->uv[FESOM_ELEMVEC(e, nz,     nl) + 0];
            real_t v_th = dyn->uv[FESOM_ELEMVEC(e, nz,     nl) + 1];
            real_t u_dn = dyn->uv[FESOM_ELEMVEC(e, nz + 1, nl) + 0];
            real_t v_dn = dyn->uv[FESOM_ELEMVEC(e, nz + 1, nl) + 1];
            ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th + c[nz]*u_dn;
            vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th + c[nz]*v_dn;
        }
        {
            int nz = nzmin;
            real_t u_th = dyn->uv[FESOM_ELEMVEC(e, nz,     nl) + 0];
            real_t v_th = dyn->uv[FESOM_ELEMVEC(e, nz,     nl) + 1];
            real_t u_dn = dyn->uv[FESOM_ELEMVEC(e, nz + 1, nl) + 0];
            real_t v_dn = dyn->uv[FESOM_ELEMVEC(e, nz + 1, nl) + 1];
            ur[nz] -= (b[nz] - 1.0)*u_th + c[nz]*u_dn;
            vr[nz] -= (b[nz] - 1.0)*v_th + c[nz]*v_dn;
        }
        {
            int nz = nzmax - 1;
            real_t u_up = dyn->uv[FESOM_ELEMVEC(e, nz - 1, nl) + 0];
            real_t v_up = dyn->uv[FESOM_ELEMVEC(e, nz - 1, nl) + 1];
            real_t u_th = dyn->uv[FESOM_ELEMVEC(e, nz,     nl) + 0];
            real_t v_th = dyn->uv[FESOM_ELEMVEC(e, nz,     nl) + 1];
            ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th;
            vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th;
        }

        /* Thomas algorithm forward sweep (lines 3301-3312) */
        cp[nzmin] = c[nzmin] / b[nzmin];
        up[nzmin] = ur[nzmin] / b[nzmin];
        vp[nzmin] = vr[nzmin] / b[nzmin];
        for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz) {
            real_t m = b[nz] - cp[nz - 1] * a[nz];
            cp[nz] = c[nz] / m;
            up[nz] = (ur[nz] - up[nz - 1] * a[nz]) / m;
            vp[nz] = (vr[nz] - vp[nz - 1] * a[nz]) / m;
        }
        /* Back substitution (lines 3314-3322) */
        ur[nzmax - 1] = up[nzmax - 1];
        vr[nzmax - 1] = vp[nzmax - 1];
        for (int nz = nzmax - 2; nz >= nzmin; --nz) {
            ur[nz] = up[nz] - cp[nz] * ur[nz + 1];
            vr[nz] = vp[nz] - cp[nz] * vr[nz + 1];
        }

        /* Write back into uv_rhs (lines 3328-3331) */
        for (int nz = nzmin; nz <= nzmax - 1; ++nz) {
            dyn->uv_rhs[FESOM_ELEMVEC(e, nz, nl) + 0] = ur[nz];
            dyn->uv_rhs[FESOM_ELEMVEC(e, nz, nl) + 1] = vr[nz];
        }
    }
}

/*--- impl_vert_visc — DEVICE kernel (M2.4): per-element TDMA -------------------
 * Kokkos parallel_for over owned elements; the tridiagonal coefficient build, the
 * "solve-for-du" conversion, and the Thomas forward-elim + back-sub all run
 * SEQUENTIALLY IN LEVEL inside the lambda. Each element solves only its own (u,v)
 * columns → NO cross-element race, so the Serial range == the C-twin loop AND
 * OpenMP is bit-identical (no scatter/reduction). The per-column scratch
 * (zbar_n/Z_n/a/b/c/ur/vr/cp/up/vp, each [64]) is lambda-local stack — per-thread
 * local memory on device (≈ 5 KB/thread, slow-first accepted), exactly the EOS
 * bulk_*[64] pattern. nl ≤ 48 in Phase 1 so the [64] cap is safe (the C twin's
 * FESOM_CHECK is host-only; dropped here as in the EOS kernel). `sqrt`→`Kokkos::sqrt`
 * (host backend == libm IEEE sqrt → Serial bit-identical; CUDA libdevice). Every
 * constant/bound/association is a verbatim copy of fesom_impl_vert_visc above.
 *
 * SYNC contract (SYNC_MAP §2 row 6): INPUTS uv_rhs (read-modify-write), uv, Av (the
 * KPP/PP device output, sync_host'd+halo'd in substep 3), helem (evolving mesh — NOT
 * set-once), w_i (read at the 3 vertices incl. halo), forcing stress_surf are pushed
 * device-current by the driver IN rail (all of them, L28); zbar + elem_nodes/ulevels/
 * nlevels are set-once device-current. OUTPUT uv_rhs marked modify_device(); the driver
 * sync_host()s it before the elem3D halo.
 */
void fesom_impl_vert_visc_kk(const struct fesom_mesh    *mesh,
                             const struct fesom_aux     *aux,
                             const struct fesom_forcing *forcing,
                             struct fesom_dyn           *dyn)
{
    const int    E              = mesh->myDim_elem2D;
    const int    nl             = mesh->nl;
    const real_t dt             = (real_t)FESOM_PHASE1_DT;
    const real_t Cd             = (real_t)FESOM_PHASE1_C_D;
    const real_t inv_density_0  = 1.0 / (real_t)FESOM_DENSITY_0;

    auto ulev   = mesh->ulevels_fld.d();
    auto nlev   = mesh->nlevels_fld.d();
    auto elnod  = mesh->elem_nodes_fld.d();
    auto zbar   = mesh->zbar_fld.d();
    auto helem  = mesh->helem_fld.d();
    auto Av     = aux->Av_fld.d();
    auto uv_rhs = dyn->uv_rhs_fld.d();
    auto uv     = dyn->uv_fld.d();
    auto w_i    = dyn->w_i_fld.d();
    auto stress = forcing->stress_surf_fld.d();

    /* M7 C.3a — FESOM_SPEED_VISCNOINIT (opt-in _exp until its A/B lands; rule 0.13).
     * Strict store-deletion only (L97 / rule 0.12) — see the banner at the init site. */
    static int s_viscnoinit = -1;
    const int viscnoinit = fesom_speed_on_exp("VISCNOINIT", &s_viscnoinit) ? 1 : 0;

    Kokkos::parallel_for("fesom_impl_vert_visc", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int e) {
            int nzmin = ulev(e) - 1;
            int nzmax = nlev(e) - 1;
            if (nzmax - nzmin < 1) return;          /* C twin: continue */

            int n0 = elnod(3*e + 0);
            int n1 = elnod(3*e + 1);
            int n2 = elnod(3*e + 2);

            real_t zbar_n[FESOM_MAX_LEVELS], Z_n[FESOM_MAX_LEVELS];
            real_t a[FESOM_MAX_LEVELS], b[FESOM_MAX_LEVELS], c[FESOM_MAX_LEVELS];
            real_t ur[FESOM_MAX_LEVELS], vr[FESOM_MAX_LEVELS];
            /* In-place Thomas (M5.24 footprint-shrink): cp reuses c (modified-c),
             * up/vp reuse ur/vr (forward-eliminated RHS). Each c/ur/vr[nz] is read
             * then overwritten in level order → bit-identical; per-thread local
             * frame drops 10->7 arrays. This kernel is local-memory-traffic bound
             * (ncu: ~7% compute, STACK:10240, REG:80). */
            real_t *cp = c, *up = ur, *vp = vr;

            /* Build zbar_n and Z_n from helem upward (lines 3167-3179).
             * VISCNOINIT: drop the (already depth-bounded) zero-init. Byte-identical for
             * MULTI-LAYER columns (nzmax-nzmin >= 2): every read of zbar_n is in
             * [nzmin, nzmax] and of Z_n in [nzmin, nzmax-1] — exactly the ranges the
             * build writes (per-element ulev/nlev bounds; no cross-field relation
             * involved). TWO-LAYER columns (nzmax-nzmin == 1) KEEP the legacy init:
             * their surface row reads Z_n[nzmax] and the (later overwritten) bottom row
             * reads Z_n[nzmin-1] — slots the legacy init loops do not write either, and
             * the branch keeps those columns bit-exactly on the legacy path so the
             * question never arises (the TDMANOINIT per-column-branch precedent). */
            if (!viscnoinit || nzmax - nzmin == 1) {
                for (int k = 0; k <= nzmax; ++k) zbar_n[k] = 0.0;
                for (int k = 0; k < nzmax; ++k)  Z_n[k]   = 0.0;
            }
            zbar_n[nzmax]   = zbar(nzmax);
            Z_n[nzmax - 1]  = zbar_n[nzmax]
                            + helem(FESOM_ELEM3D(e, nzmax - 1, nl)) / 2.0;
            for (int nz = nzmax - 1; nz >= nzmin + 1; --nz) {
                zbar_n[nz]   = zbar_n[nz + 1] + helem(FESOM_ELEM3D(e, nz,     nl));
                Z_n[nz - 1]  = zbar_n[nz]     + helem(FESOM_ELEM3D(e, nz - 1, nl)) / 2.0;
            }
            zbar_n[nzmin] = zbar_n[nzmin + 1] + helem(FESOM_ELEM3D(e, nzmin, nl));

            /* Tridiagonal coefficients: regular interior (lines 3185-3198). */
            for (int nz = nzmin + 1; nz <= nzmax - 2; ++nz) {
                real_t zinv = dt / (zbar_n[nz] - zbar_n[nz + 1]);
                a[nz] = -Av(FESOM_ELEM3D(e, nz,     nl)) / (Z_n[nz - 1] - Z_n[nz]) * zinv;
                c[nz] = -Av(FESOM_ELEM3D(e, nz + 1, nl)) / (Z_n[nz] - Z_n[nz + 1]) * zinv;
                b[nz] = -a[nz] - c[nz] + 1.0;
                real_t wu = (w_i(FESOM_NODE3D(n0, nz,     nl))
                           + w_i(FESOM_NODE3D(n1, nz,     nl))
                           + w_i(FESOM_NODE3D(n2, nz,     nl))) / 3.0;
                real_t wd = (w_i(FESOM_NODE3D(n0, nz + 1, nl))
                           + w_i(FESOM_NODE3D(n1, nz + 1, nl))
                           + w_i(FESOM_NODE3D(n2, nz + 1, nl))) / 3.0;
                a[nz] = a[nz] + (wu < 0.0 ? wu : 0.0) * zinv;
                b[nz] = b[nz] + (wu > 0.0 ? wu : 0.0) * zinv;
                b[nz] = b[nz] - (wd < 0.0 ? wd : 0.0) * zinv;
                c[nz] = c[nz] - (wd > 0.0 ? wd : 0.0) * zinv;
            }

            /* Bottom row (lines 3200-3208). */
            {
                int nz = nzmax - 1;
                real_t zinv = dt / (zbar_n[nz] - zbar_n[nz + 1]);
                a[nz] = -Av(FESOM_ELEM3D(e, nz, nl)) / (Z_n[nz - 1] - Z_n[nz]) * zinv;
                b[nz] = -a[nz] + 1.0;
                c[nz] = 0.0;
                real_t wu = (w_i(FESOM_NODE3D(n0, nz, nl))
                           + w_i(FESOM_NODE3D(n1, nz, nl))
                           + w_i(FESOM_NODE3D(n2, nz, nl))) / 3.0;
                a[nz] = a[nz] + (wu < 0.0 ? wu : 0.0) * zinv;
                b[nz] = b[nz] + (wu > 0.0 ? wu : 0.0) * zinv;
            }

            /* Surface row (lines 3215-3231); zinv_top kept for the wind-stress RHS use. */
            real_t zinv_top = dt / (zbar_n[nzmin] - zbar_n[nzmin + 1]);
            {
                int nz = nzmin;
                c[nz] = -Av(FESOM_ELEM3D(e, nz + 1, nl)) / (Z_n[nz] - Z_n[nz + 1]) * zinv_top;
                a[nz] = 0.0;
                b[nz] = -c[nz] + 1.0;
                real_t wu = (w_i(FESOM_NODE3D(n0, nz,     nl))
                           + w_i(FESOM_NODE3D(n1, nz,     nl))
                           + w_i(FESOM_NODE3D(n2, nz,     nl))) / 3.0;
                real_t wd = (w_i(FESOM_NODE3D(n0, nz + 1, nl))
                           + w_i(FESOM_NODE3D(n1, nz + 1, nl))
                           + w_i(FESOM_NODE3D(n2, nz + 1, nl))) / 3.0;
                b[nz] = b[nz] + wu * zinv_top;
                b[nz] = b[nz] - (wd < 0.0 ? wd : 0.0) * zinv_top;
                c[nz] = c[nz] - (wd > 0.0 ? wd : 0.0) * zinv_top;
            }

            /* Build RHS from current uv_rhs (lines 3238-3245). */
            for (int nz = nzmin; nz <= nzmax - 1; ++nz) {
                ur[nz] = uv_rhs(FESOM_ELEMVEC(e, nz, nl) + 0);
                vr[nz] = uv_rhs(FESOM_ELEMVEC(e, nz, nl) + 1);
            }

            /* Surface forcing (wind stress); element wind stress at index 2*e. */
            ur[nzmin] += zinv_top * stress(2*e + 0) * inv_density_0;
            vr[nzmin] += zinv_top * stress(2*e + 1) * inv_density_0;

            /* Bottom drag (lines 3267-3272). */
            real_t zinv_bot = dt / (zbar_n[nzmax - 1] - zbar_n[nzmax]);
            {
                int nz = nzmax - 1;
                real_t u_bot = uv(FESOM_ELEMVEC(e, nz, nl) + 0);
                real_t v_bot = uv(FESOM_ELEMVEC(e, nz, nl) + 1);
                real_t spd   = Kokkos::sqrt(u_bot * u_bot + v_bot * v_bot);
                real_t friction = -Cd * spd;
                ur[nz] += zinv_bot * friction * u_bot;
                vr[nz] += zinv_bot * friction * v_bot;
            }

            /* Convert "solve for u_new" to "solve for du = u_new - u_old"
               (lines 3282-3292). */
            for (int nz = nzmin + 1; nz <= nzmax - 2; ++nz) {
                real_t u_up = uv(FESOM_ELEMVEC(e, nz - 1, nl) + 0);
                real_t v_up = uv(FESOM_ELEMVEC(e, nz - 1, nl) + 1);
                real_t u_th = uv(FESOM_ELEMVEC(e, nz,     nl) + 0);
                real_t v_th = uv(FESOM_ELEMVEC(e, nz,     nl) + 1);
                real_t u_dn = uv(FESOM_ELEMVEC(e, nz + 1, nl) + 0);
                real_t v_dn = uv(FESOM_ELEMVEC(e, nz + 1, nl) + 1);
                ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th + c[nz]*u_dn;
                vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th + c[nz]*v_dn;
            }
            {
                int nz = nzmin;
                real_t u_th = uv(FESOM_ELEMVEC(e, nz,     nl) + 0);
                real_t v_th = uv(FESOM_ELEMVEC(e, nz,     nl) + 1);
                real_t u_dn = uv(FESOM_ELEMVEC(e, nz + 1, nl) + 0);
                real_t v_dn = uv(FESOM_ELEMVEC(e, nz + 1, nl) + 1);
                ur[nz] -= (b[nz] - 1.0)*u_th + c[nz]*u_dn;
                vr[nz] -= (b[nz] - 1.0)*v_th + c[nz]*v_dn;
            }
            {
                int nz = nzmax - 1;
                real_t u_up = uv(FESOM_ELEMVEC(e, nz - 1, nl) + 0);
                real_t v_up = uv(FESOM_ELEMVEC(e, nz - 1, nl) + 1);
                real_t u_th = uv(FESOM_ELEMVEC(e, nz,     nl) + 0);
                real_t v_th = uv(FESOM_ELEMVEC(e, nz,     nl) + 1);
                ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th;
                vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th;
            }

            /* Thomas algorithm forward sweep (lines 3301-3312). */
            cp[nzmin] = c[nzmin] / b[nzmin];
            up[nzmin] = ur[nzmin] / b[nzmin];
            vp[nzmin] = vr[nzmin] / b[nzmin];
            for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz) {
                real_t m = b[nz] - cp[nz - 1] * a[nz];
                cp[nz] = c[nz] / m;
                up[nz] = (ur[nz] - up[nz - 1] * a[nz]) / m;
                vp[nz] = (vr[nz] - vp[nz - 1] * a[nz]) / m;
            }
            /* Back substitution (lines 3314-3322). */
            ur[nzmax - 1] = up[nzmax - 1];
            vr[nzmax - 1] = vp[nzmax - 1];
            for (int nz = nzmax - 2; nz >= nzmin; --nz) {
                ur[nz] = up[nz] - cp[nz] * ur[nz + 1];
                vr[nz] = vp[nz] - cp[nz] * vr[nz + 1];
            }

            /* Write back into uv_rhs (lines 3328-3331). */
            for (int nz = nzmin; nz <= nzmax - 1; ++nz) {
                uv_rhs(FESOM_ELEMVEC(e, nz, nl) + 0) = ur[nz];
                uv_rhs(FESOM_ELEMVEC(e, nz, nl) + 1) = vr[nz];
            }
        });

    dyn->uv_rhs_fld.modify_device();   /* driver sync_host()s before the elem3D halo */
}

/*--- FESOM_KK_VERIFY=ivisc (M2.4) --------------------------------------------
 * impl_vert_visc READ-MODIFY-WRITES uv_rhs (reads it to build the RHS, overwrites
 * with the TDMA solution), so the C-twin oracle needs the PRE-kernel uv_rhs —
 * the L26 capture-before pattern (cf. mo_convect). The driver captures uv_rhs_in
 * before the IN rail; here: snapshot the KK result, restore uv_rhs_in, run the C
 * twin (reads the restored uv_rhs + the intact uv/Av/stress/w_i/helem), diff,
 * restore KK. The diff covers the owned uv_rhs [0, myDim_elem2D*nl*2); levels the
 * kernel leaves untouched match by construction (both keep uv_rhs_in there).
 * Non-intrusive; asserts max|Δ|==0 on Serial.
 */
void fesom_impl_vert_visc_verify(const struct fesom_mesh    *mesh,
                                 const struct fesom_aux     *aux,
                                 const struct fesom_forcing *forcing,
                                 struct fesom_dyn           *dyn,
                                 int step_n, const real_t *uv_rhs_in)
{
    const int nl = mesh->nl;
    const size_t nvec = (size_t)mesh->myDim_elem2D * (size_t)nl * 2;     /* owned uv_rhs */

    std::vector<real_t> kk_uv_rhs(dyn->uv_rhs, dyn->uv_rhs + nvec);      /* KK result (host-synced) */
    std::copy(uv_rhs_in, uv_rhs_in + nvec, dyn->uv_rhs);                 /* restore the pre-kernel input */
    fesom_impl_vert_visc(mesh, aux, forcing, dyn);                       /* C twin runs on the input */
    double dmax = 0.0;
    for (size_t i = 0; i < nvec; ++i) {
        double d = std::fabs((double)kk_uv_rhs[i] - (double)dyn->uv_rhs[i]);
        if (d > dmax) dmax = d;
    }
    std::copy(kk_uv_rhs.begin(), kk_uv_rhs.end(), dyn->uv_rhs);          /* restore KK */

    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=ivisc] step %d backend=%s  max|Δ|: uv_rhs=%.3e\n",
                step_n, backend.c_str(), dmax);
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=ivisc] FAIL: Serial must be bit-identical to the C twin "
                             "(max|Δ|=%.3e)\n", dmax);
        std::abort();
    }
}

/*===========================================================================
 * update_vel (oce_dyn.F90:73-173)
 *
 *   eta = -g * theta * dt * d_eta[3 vertices]
 *   Fx = sum( gradient_sca[1..3] * eta )
 *   Fy = sum( gradient_sca[4..6] * eta )
 *   uv += uv_rhs + (Fx, Fy)
 *===========================================================================*/

void fesom_update_vel(const struct fesom_mesh *mesh,
                      struct fesom_dyn        *dyn)
{
    const int E = mesh->myDim_elem2D;       /* interior — exchange UV after */
    const int nl = mesh->nl;
    const real_t coef = -(real_t)FESOM_G * (real_t)FESOM_PHASE1_THETA
                       * (real_t)FESOM_PHASE1_DT;

    for (int e = 0; e < E; ++e) {
        int nzmin = mesh->ulevels[e]   - 1;
        int nzmax = mesh->nlevels[e]   - 1;
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];

        real_t e0 = coef * dyn->d_eta[n0];
        real_t e1 = coef * dyn->d_eta[n1];
        real_t e2 = coef * dyn->d_eta[n2];
        const real_t *g = &mesh->gradient_sca[6 * e];
        real_t Fx = g[0]*e0 + g[1]*e1 + g[2]*e2;
        real_t Fy = g[3]*e0 + g[4]*e1 + g[5]*e2;

        for (int nz = nzmin; nz < nzmax; ++nz) {
            size_t k = FESOM_ELEMVEC(e, nz, nl);
            dyn->uv[k + 0] += dyn->uv_rhs[k + 0] + Fx;
            dyn->uv[k + 1] += dyn->uv_rhs[k + 1] + Fy;
        }
    }
}

/*===========================================================================
 * fesom_update_vel_kk — M4.2-b DEVICE twin (substep 9). Race-free per-element
 * map: each element writes only its own uv slots; reads d_eta at the 3 vertices
 * (which can be HALO nodes → the driver re-pushes d_eta after its halo, L30),
 * uv_rhs, gradient_sca. No scatter/reduce → Serial AND OpenMP bit-identical.
 *===========================================================================*/
void fesom_update_vel_kk(const struct fesom_mesh *mesh,
                         struct fesom_dyn        *dyn)
{
    const int E  = mesh->myDim_elem2D;
    const int nl = mesh->nl;
    const real_t coef = -(real_t)FESOM_G * (real_t)FESOM_PHASE1_THETA
                       * (real_t)FESOM_PHASE1_DT;

    auto ulev   = mesh->ulevels_fld.d();
    auto nlev   = mesh->nlevels_fld.d();
    auto en     = mesh->elem_nodes_fld.d();
    auto grad   = mesh->gradient_sca_fld.d();
    auto d_eta  = dyn->d_eta_fld.d();
    auto uv     = dyn->uv_fld.d();
    auto uv_rhs = dyn->uv_rhs_fld.d();

    /* M5.21 flat lever: one thread per (elem,LEVEL) — RP(0,E*nl), decode (e,nz). uv is
     * elem-major×2 (FESOM_ELEMVEC = (e*nl+nz)*2) so one-thread-per-(e,nz) writes the 2
     * adjacent comps → coalesced warp store (vs the old per-element stride of nl*2). Fx/Fy
     * (level-independent) are recomputed per level — cheap, and d_eta/grad reads broadcast
     * across a same-e warp. Each (e,nz) writes its own uv slots → bit-identical. */
    Kokkos::parallel_for("fesom_update_vel", Kokkos::RangePolicy<>(0, (size_t)E * nl),
        KOKKOS_LAMBDA(const size_t i) {
            const int e = (int)(i / nl), nz = (int)(i - (size_t)e*nl);
            int nzmin = ulev(e) - 1;
            int nzmax = nlev(e) - 1;
            if (nz < nzmin || nz >= nzmax) return;
            int n0 = en(3*e + 0);
            int n1 = en(3*e + 1);
            int n2 = en(3*e + 2);
            real_t e0 = coef * d_eta(n0);
            real_t e1 = coef * d_eta(n1);
            real_t e2 = coef * d_eta(n2);
            real_t Fx = grad(6*e + 0)*e0 + grad(6*e + 1)*e1 + grad(6*e + 2)*e2;
            real_t Fy = grad(6*e + 3)*e0 + grad(6*e + 4)*e1 + grad(6*e + 5)*e2;
            size_t k = FESOM_ELEMVEC(e, nz, nl);
            uv(k + 0) += uv_rhs(k + 0) + Fx;
            uv(k + 1) += uv_rhs(k + 1) + Fy;
        });

    dyn->uv_fld.modify_device();   /* driver sync_host()s before the elem3D halo */
}

/*===========================================================================
 * visc_filt_bcksct (oce_dyn.F90:278-426) — opt_visc=5 with easy backscatter
 *===========================================================================*/

void fesom_visc_filt_bcksct(const struct fesom_mesh *mesh,
                            struct fesom_dyn        *dyn,
                            struct fesom_partit     *partit)
{
    const int E    = mesh->myDim_elem2D;
    const int Eedg = mesh->myDim_edge2D + mesh->eDim_edge2D;
    const int nl   = mesh->nl;
    /* Allocations sized for full local extent — touch only the elements we
     * compute on (exchange is done in fesom_step after this returns). */
    int E_alloc = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    int N_alloc = mesh->myDim_nod2D + mesh->eDim_nod2D;
    (void)E_alloc; (void)N_alloc;
    const real_t dt = (real_t)FESOM_PHASE1_DT;
    const real_t g0 = (real_t)FESOM_PHASE1_VISC_GAMMA0;
    const real_t g1 = (real_t)FESOM_PHASE1_VISC_GAMMA1;
    const real_t g2 = (real_t)FESOM_PHASE1_VISC_GAMMA2;
    const real_t bsret = (real_t)FESOM_PHASE1_VISC_EASYBSRETURN;

    /* Zero the four work arrays (lines 316-322) over full local extent. */
    memset(dyn->u_b, 0, (size_t)E_alloc * (size_t)nl * sizeof(real_t));
    memset(dyn->v_b, 0, (size_t)E_alloc * (size_t)nl * sizeof(real_t));
    memset(dyn->u_c, 0, (size_t)N_alloc * (size_t)nl * sizeof(real_t));
    memset(dyn->v_c, 0, (size_t)N_alloc * (size_t)nl * sizeof(real_t));

    /* Edge loop — accumulate per-element viscous updates (lines 327-361).
       Skip boundary edges (Fortran: myList_edge2D > edge2D_in; serial: el2 < 0). */
    for (int ed = 0; ed < Eedg; ++ed) {
        int el1 = mesh->edge_tri[2*ed + 0];
        int el2 = mesh->edge_tri[2*ed + 1];
        if (el1 < 0 || el2 < 0) continue;       /* boundary edge */

        real_t a1 = mesh->elem_area[el1];
        real_t a2 = mesh->elem_area[el2];
        real_t len = sqrt(a1 + a2);

        int nu1 = mesh->ulevels[el1] - 1;
        int nu2 = mesh->ulevels[el2] - 1;
        int nzmin = (nu1 > nu2) ? nu1 : nu2;       /* maxval(ulevels) */
        int nl1   = mesh->nlevels[el1] - 1;
        int nl2   = mesh->nlevels[el2] - 1;
        int nzmax = (nl1 < nl2) ? nl1 : nl2;       /* minval(nlevels) - 1 */

        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t u1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0]
                       - dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
            real_t v1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1]
                       - dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
            real_t mag2 = u1*u1 + v1*v1;
            real_t mag  = sqrt(mag2);
            real_t vi_inner = (g1 * mag > g2 * mag2) ? g1 * mag : g2 * mag2;
            real_t vi = dt * (g0 > vi_inner ? g0 : vi_inner) * len;
            real_t du = u1 * vi;
            real_t dv = v1 * vi;
            dyn->u_b[FESOM_ELEM3D(el1, nz, nl)] -= du / a1;
            dyn->v_b[FESOM_ELEM3D(el1, nz, nl)] -= dv / a1;
            dyn->u_b[FESOM_ELEM3D(el2, nz, nl)] += du / a2;
            dyn->v_b[FESOM_ELEM3D(el2, nz, nl)] += dv / a2;
        }
    }

    /* Halo-exchange U_b, V_b at elements — Fortran oce_dyn.F90:367-369.
     * Without this, the U_c smoothing at boundary nodes reads stale halo
     * U_b → wrong U_c → wrong backscatter at boundary elements → no
     * viscous damping at partition boundaries → uv grows unchecked. */
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(dyn->u_b, FESOM_HALO_ELEM3D, nl, 1, partit);
        fesom_halo_exchange(dyn->v_b, FESOM_HALO_ELEM3D, nl, 1, partit);
    }

    /* Smooth U_b → U_c at nodes (area-weighted mean of surrounding cells).
       Interior nodes only; halo gets values via exchange below.
       Same neighbour traversal as compute_vel_nodes. */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        int nu1 = mesh->ulevels_nod2D[n] - 1;
        int nl1 = mesh->nlevels_nod2D[n] - 1;
        int o0 = mesh->nod_in_elem2D_offsets[n];
        int o1 = mesh->nod_in_elem2D_offsets[n + 1];
        for (int nz = nu1; nz < nl1; ++nz) {
            real_t tot = 0.0, tu = 0.0, tv = 0.0;
            for (int k = o0; k < o1; ++k) {
                int e = mesh->nod_in_elem2D[k];
                real_t a = mesh->elem_area[e];
                tot += a;
                tu  += dyn->u_b[FESOM_ELEM3D(e, nz, nl)] * a;
                tv  += dyn->v_b[FESOM_ELEM3D(e, nz, nl)] * a;
            }
            if (tot > 0.0) {
                dyn->u_c[FESOM_NODE3D(n, nz, nl)] = tu / tot;
                dyn->v_c[FESOM_NODE3D(n, nz, nl)] = tv / tot;
            }
        }
    }

    /* Halo-exchange U_c, V_c — Fortran oce_dyn.F90:391-393. The Apply step
     * below reads U_c[vertex] for vertices that may be halo nodes. */
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(dyn->u_c, FESOM_HALO_NOD3D, nl, 1, partit);
        fesom_halo_exchange(dyn->v_c, FESOM_HALO_NOD3D, nl, 1, partit);
    }

    /* Apply: uv_rhs += u_b − easybsreturn · mean(u_c at 3 vertices)
       (lines 418-422 — non-split-explicit branch, what we use in Phase 2). */
    for (int e = 0; e < E; ++e) {
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];
        int nu1 = mesh->ulevels[e] - 1;
        int nle = mesh->nlevels[e] - 1;
        for (int nz = nu1; nz < nle; ++nz) {
            real_t uc_mean = (dyn->u_c[FESOM_NODE3D(n0, nz, nl)]
                            + dyn->u_c[FESOM_NODE3D(n1, nz, nl)]
                            + dyn->u_c[FESOM_NODE3D(n2, nz, nl)]) / 3.0;
            real_t vc_mean = (dyn->v_c[FESOM_NODE3D(n0, nz, nl)]
                            + dyn->v_c[FESOM_NODE3D(n1, nz, nl)]
                            + dyn->v_c[FESOM_NODE3D(n2, nz, nl)]) / 3.0;
            dyn->uv_rhs[FESOM_ELEMVEC(e, nz, nl) + 0]
                += dyn->u_b[FESOM_ELEM3D(e, nz, nl)] - bsret * uc_mean;
            dyn->uv_rhs[FESOM_ELEMVEC(e, nz, nl) + 1]
                += dyn->v_b[FESOM_ELEM3D(e, nz, nl)] - bsret * vc_mean;
        }
    }
}

/*===========================================================================
 * visc_filt_bidiff (oce_dyn.F90:592-744, opt_visc=7) — biharmonic, flow-aware.
 *
 * This is the scheme CORE2 actually runs (work_core/namelist.dyn opt_visc=7).
 * Strictly energy-dissipative, momentum-conserving biharmonic operator: the
 * ∇⁴ damps grid-scale (2Δx) modes as k⁴, far more strongly than the harmonic
 * opt_visc=5 (visc_filt_bcksct) above — which is why opt_visc=7 is required for
 * stability at dt=1800.
 *
 *   Stage 1 (edge loop): build the flow-aware Laplacian U_c,V_c at elements:
 *     on each interior edge, vi = sqrt(max(γ0, max(γ1·|du|, γ2·|du|²))·len),
 *     U_c[el1] -= du·vi; U_c[el2] += du·vi   (du = uv[el1]-uv[el2]).
 *   exchange_elem(U_c,V_c).
 *   Stage 2 (edge loop): apply the second Laplacian into uv_rhs:
 *     vi = -dt·sqrt(...·len),  viLapl = dt·max(γ0h, γ1h·|du|)·len  (=0 for CORE2),
 *     update = vi·(U_c[el1]-U_c[el2]) + viLapl·du,
 *     uv_rhs[el1] -= update/area1;  uv_rhs[el2] += update/area2.
 *
 * Reuses dyn->u_b/v_b (element scratch) as the stage-1 Laplacian U_c/V_c —
 * opt_visc=5 and opt_visc=7 never run together. Edge loop covers owned+halo
 * edges so owned elements get all edge contributions; halo elements are fixed
 * by the uv_rhs exchange in fesom_step after this returns.
 */
void fesom_visc_filt_bidiff(const struct fesom_mesh *mesh,
                            struct fesom_dyn        *dyn,
                            struct fesom_partit     *partit)
{
    const int Eedg = mesh->myDim_edge2D + mesh->eDim_edge2D;
    const int nl   = mesh->nl;
    const int E_alloc = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    const real_t dt  = (real_t)FESOM_PHASE1_DT;
    /* DIAG ONLY (FESOM_VISC_MULT, default 1.0): scale the flow-aware viscosity
       coefficients to probe whether the residual 2dx is viscosity-dampable.
       Net biharmonic effect scales ~vmult (vi scales by sqrt in each stage). */
    real_t vmult = 1.0;
    { const char *e = getenv("FESOM_VISC_MULT"); if (e) vmult = (real_t)atof(e); }
    const real_t g0  = (real_t)FESOM_PHASE1_VISC_GAMMA0 * vmult;
    const real_t g1  = (real_t)FESOM_PHASE1_VISC_GAMMA1 * vmult;
    const real_t g2  = (real_t)FESOM_PHASE1_VISC_GAMMA2 * vmult;
    const real_t g0h = (real_t)FESOM_PHASE1_VISC_GAMMA0_H;   /* 0 for CORE2 */
    const real_t g1h = (real_t)FESOM_PHASE1_VISC_GAMMA1_H;   /* 0 for CORE2 */

    real_t *Uc = dyn->u_b;     /* stage-1 Laplacian at elements (scratch) */
    real_t *Vc = dyn->v_b;

    /* DIAG (FESOM_DIAG_VISCEDGE): one-time count of edges this loop SKIPS via
       el<0 that are actually INTERIOR (myList_edge2D <= edge2D_in). Fortran
       skips only TRUE boundary edges; any interior skip here is a viscous-
       coupling gap at a rank boundary (2nd triangle beyond the local halo). */
    if (getenv("FESOM_DIAG_VISCEDGE")) {
        static int viscedge_done = 0;
        if (!viscedge_done) {
            viscedge_done = 1;
            int n_int = 0, n_bnd = 0;
            for (int ed = 0; ed < Eedg; ++ed) {
                if (mesh->edge_tri[2*ed+0] >= 0 && mesh->edge_tri[2*ed+1] >= 0) continue;
                if (partit->myList_edge2D[ed] <= mesh->edge2D_in) n_int++; else n_bnd++;
            }
            if (n_int > 0)
                fprintf(stderr, "[viscedge r%d] INTERIOR edges skipped by el<0: %d "
                        "(true-boundary skipped: %d, local edges: %d)\n",
                        partit->mype, n_int, n_bnd, Eedg);
        }
    }

    /* Stage 1: U_c = V_c = 0 over full local extent (Fortran 621-625). */
    memset(Uc, 0, (size_t)E_alloc * (size_t)nl * sizeof(real_t));
    memset(Vc, 0, (size_t)E_alloc * (size_t)nl * sizeof(real_t));

    /* Stage 1 edge loop (Fortran 631-665). Skip boundary edges (Fortran:
       myList_edge2D > edge2D_in; serial/owned: el2 < 0). */
    for (int ed = 0; ed < Eedg; ++ed) {
        int el1 = mesh->edge_tri[2*ed + 0];
        int el2 = mesh->edge_tri[2*ed + 1];
        if (el1 < 0 || el2 < 0) continue;
        real_t len = sqrt(mesh->elem_area[el1] + mesh->elem_area[el2]);
        int nu1 = mesh->ulevels[el1] - 1, nu2 = mesh->ulevels[el2] - 1;
        int nzmin = (nu1 > nu2) ? nu1 : nu2;            /* maxval(ulevels) */
        int nl1 = mesh->nlevels[el1] - 1, nl2 = mesh->nlevels[el2] - 1;
        int nzmax = (nl1 < nl2) ? nl1 : nl2;            /* minval(nlevels)-1 */
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t u1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0]
                       - dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
            real_t v1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1]
                       - dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
            real_t vi = u1*u1 + v1*v1;
            real_t inner = (g1*sqrt(vi) > g2*vi) ? g1*sqrt(vi) : g2*vi;
            vi = sqrt((g0 > inner ? g0 : inner) * len);
            real_t du = u1 * vi, dv = v1 * vi;
            Uc[FESOM_ELEM3D(el1, nz, nl)] -= du;
            Vc[FESOM_ELEM3D(el1, nz, nl)] -= dv;
            Uc[FESOM_ELEM3D(el2, nz, nl)] += du;
            Vc[FESOM_ELEM3D(el2, nz, nl)] += dv;
        }
    }

    /* exchange_elem(U_c, V_c) — Fortran 670-672. */
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(Uc, FESOM_HALO_ELEM3D, nl, 1, partit);
        fesom_halo_exchange(Vc, FESOM_HALO_ELEM3D, nl, 1, partit);
    }

    /* Stage 2 edge loop (Fortran 677-742, non-subcycl branch). */
    for (int ed = 0; ed < Eedg; ++ed) {
        int el1 = mesh->edge_tri[2*ed + 0];
        int el2 = mesh->edge_tri[2*ed + 1];
        if (el1 < 0 || el2 < 0) continue;
        real_t a1 = mesh->elem_area[el1], a2 = mesh->elem_area[el2];
        real_t len = sqrt(a1 + a2);
        int nu1 = mesh->ulevels[el1] - 1, nu2 = mesh->ulevels[el2] - 1;
        int nzmin = (nu1 > nu2) ? nu1 : nu2;
        int nl1 = mesh->nlevels[el1] - 1, nl2 = mesh->nlevels[el2] - 1;
        int nzmax = (nl1 < nl2) ? nl1 : nl2;
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t u1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0]
                       - dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
            real_t v1 = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1]
                       - dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
            real_t vi = u1*u1 + v1*v1;
            real_t inner = (g1*sqrt(vi) > g2*vi) ? g1*sqrt(vi) : g2*vi;
            vi = -dt * sqrt((g0 > inner ? g0 : inner) * len);
            real_t mag = sqrt(u1*u1 + v1*v1);
            real_t viLapl = dt * ((g0h > g1h*mag) ? g0h : g1h*mag) * len;  /* 0 for CORE2 */
            real_t du = vi * (Uc[FESOM_ELEM3D(el1, nz, nl)] - Uc[FESOM_ELEM3D(el2, nz, nl)]) + viLapl*u1;
            real_t dv = vi * (Vc[FESOM_ELEM3D(el1, nz, nl)] - Vc[FESOM_ELEM3D(el2, nz, nl)]) + viLapl*v1;
            dyn->uv_rhs[FESOM_ELEMVEC(el1, nz, nl) + 0] -= du / a1;
            dyn->uv_rhs[FESOM_ELEMVEC(el1, nz, nl) + 1] -= dv / a1;
            dyn->uv_rhs[FESOM_ELEMVEC(el2, nz, nl) + 0] += du / a2;
            dyn->uv_rhs[FESOM_ELEMVEC(el2, nz, nl) + 1] += dv / a2;
        }
    }
}

/*--- visc_filt_bidiff — DEVICE kernel (M2.4): the first edge→element SCATTER ---
 * Biharmonic ∇⁴ (opt_visc=7) as two edge-loop stages, each an EDGE→ELEMENT scatter
 * (neighbouring edges accumulate into the same element), ported with
 * Kokkos::atomic_add (docs/SCATTER_STRATEGY.md, D22). On Serial a single-thread
 * atomic_add executes in the loop's edge order, identical to the C twin's `+=` →
 * Serial bit-identical (FESOM_KK_VERIFY=vfilt max|Δ|==0); on OpenMP/CUDA the atomic
 * accumulation order is non-deterministic → climate-close (≲1e-12), by the D5 ladder
 * (these are the first kernels for which OpenMP is NOT bit-identical to the golden).
 *
 *   zero Uc/Vc (= dyn u_b/v_b scratch) on device
 *   stage 1 (edge scatter): Uc/Vc = the flow-aware Laplacian
 *   ---- INTERNAL elem3D halo of Uc/Vc (D21 bracket, the KPP smooth_blmc analogue) ----
 *   stage 2 (edge scatter): uv_rhs += the biharmonic update
 *
 * Every constant/bound/association (incl. the double `g1*sqrt(vi)` evaluation and
 * the `g0_h`/`g1_h`=0 Laplacian add-on) is a verbatim copy of the C twin above;
 * `sqrt`→`Kokkos::sqrt` (host == libm IEEE → Serial bit-identical). FESOM_VISC_MULT
 * (real physics knob, default 1.0) + the one-time FESOM_DIAG_VISCEDGE diagnostic stay
 * host code in the preamble. SYNC: INPUTS uv/uv_rhs device-current (driver IN rail);
 * the function owns the internal Uc/Vc bracket; OUTPUT uv_rhs modify_device().
 */
void fesom_visc_filt_bidiff_kk(const struct fesom_mesh *mesh,
                               struct fesom_dyn        *dyn,
                               struct fesom_partit     *partit)
{
    const int Eedg    = mesh->myDim_edge2D + mesh->eDim_edge2D;
    const int nl      = mesh->nl;
    const int E_alloc = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    const real_t dt   = (real_t)FESOM_PHASE1_DT;
    real_t vmult = 1.0;
    { const char *e = getenv("FESOM_VISC_MULT"); if (e) vmult = (real_t)atof(e); }
    const real_t g0  = (real_t)FESOM_PHASE1_VISC_GAMMA0 * vmult;
    const real_t g1  = (real_t)FESOM_PHASE1_VISC_GAMMA1 * vmult;
    const real_t g2  = (real_t)FESOM_PHASE1_VISC_GAMMA2 * vmult;
    const real_t g0h = (real_t)FESOM_PHASE1_VISC_GAMMA0_H;   /* 0 for CORE2 */
    const real_t g1h = (real_t)FESOM_PHASE1_VISC_GAMMA1_H;   /* 0 for CORE2 */

    /* DIAG (FESOM_DIAG_VISCEDGE): one-time host count of INTERIOR edges this loop
       SKIPS via el<0 (a viscous-coupling gap at a rank boundary). Verbatim host. */
    if (getenv("FESOM_DIAG_VISCEDGE")) {
        static int viscedge_done = 0;
        if (!viscedge_done) {
            viscedge_done = 1;
            int n_int = 0, n_bnd = 0;
            for (int ed = 0; ed < Eedg; ++ed) {
                if (mesh->edge_tri[2*ed+0] >= 0 && mesh->edge_tri[2*ed+1] >= 0) continue;
                if (partit->myList_edge2D[ed] <= mesh->edge2D_in) n_int++; else n_bnd++;
            }
            if (n_int > 0)
                fprintf(stderr, "[viscedge r%d] INTERIOR edges skipped by el<0: %d "
                        "(true-boundary skipped: %d, local edges: %d)\n",
                        partit->mype, n_int, n_bnd, Eedg);
        }
    }

    auto edge_tri = mesh->edge_tri_fld.d();
    auto area     = mesh->elem_area_fld.d();
    auto ulev     = mesh->ulevels_fld.d();
    auto nlev     = mesh->nlevels_fld.d();
    auto uv       = dyn->uv_fld.d();
    auto Uc       = dyn->u_b_fld.d();
    auto Vc       = dyn->v_b_fld.d();
    auto uv_rhs   = dyn->uv_rhs_fld.d();

    /* Stage 1: U_c = V_c = 0 over full local extent (Fortran 621-625). */
    const int UcN = E_alloc * nl;
    Kokkos::parallel_for("fesom_visc_bidiff_zero", Kokkos::RangePolicy<>(0, UcN),
        KOKKOS_LAMBDA(const int i) { Uc(i) = 0.0; Vc(i) = 0.0; });

    /* Stage 1 edge loop (Fortran 631-665) — EDGE→ELEMENT scatter (atomic_add). */
    Kokkos::parallel_for("fesom_visc_bidiff_stage1", Kokkos::RangePolicy<>(0, Eedg),
        KOKKOS_LAMBDA(const int ed) {
            int el1 = edge_tri(2*ed + 0);
            int el2 = edge_tri(2*ed + 1);
            if (el1 < 0 || el2 < 0) return;
            real_t len = Kokkos::sqrt(area(el1) + area(el2));
            int nu1 = ulev(el1) - 1, nu2 = ulev(el2) - 1;
            int nzmin = (nu1 > nu2) ? nu1 : nu2;            /* maxval(ulevels) */
            int nl1 = nlev(el1) - 1, nl2 = nlev(el2) - 1;
            int nzmax = (nl1 < nl2) ? nl1 : nl2;            /* minval(nlevels)-1 */
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t u1 = uv(FESOM_ELEMVEC(el1, nz, nl) + 0)
                          - uv(FESOM_ELEMVEC(el2, nz, nl) + 0);
                real_t v1 = uv(FESOM_ELEMVEC(el1, nz, nl) + 1)
                          - uv(FESOM_ELEMVEC(el2, nz, nl) + 1);
                real_t vi = u1*u1 + v1*v1;
                real_t inner = (g1*Kokkos::sqrt(vi) > g2*vi) ? g1*Kokkos::sqrt(vi) : g2*vi;
                vi = Kokkos::sqrt((g0 > inner ? g0 : inner) * len);
                real_t du = u1 * vi, dv = v1 * vi;
                Kokkos::atomic_add(&Uc(FESOM_ELEM3D(el1, nz, nl)), -du);
                Kokkos::atomic_add(&Vc(FESOM_ELEM3D(el1, nz, nl)), -dv);
                Kokkos::atomic_add(&Uc(FESOM_ELEM3D(el2, nz, nl)),  du);
                Kokkos::atomic_add(&Vc(FESOM_ELEM3D(el2, nz, nl)),  dv);
            }
        });

    /* ---- INTERNAL HALO BRACKET: exchange_elem(U_c, V_c) (Fortran 670-672) ----
     * Stage 1 wrote Uc/Vc on device → halo → stage 2 reads halo-current Uc/Vc on
     * device (D21). M5.1: fesom_halo_field dispatches to GPU-aware-MPI on-device
     * under CUDA; host-staged on Serial/OpenMP (a no-op round-trip at np=1).
     * M5.23 (L3): u_b+v_b are same-kind (ELEM3D nc=1) and adjacent (stage 1 wrote both;
     * nothing between) → one FUSED message/neighbour. Bit-identical (co-pack only). */
    fesom_halo_field2(dyn->u_b_fld, dyn->v_b_fld, FESOM_HALO_ELEM3D, nl, 1, partit);

    /* Stage 2 edge loop (Fortran 677-742, non-subcycl) — EDGE→ELEMENT scatter into uv_rhs. */
    Kokkos::parallel_for("fesom_visc_bidiff_stage2", Kokkos::RangePolicy<>(0, Eedg),
        KOKKOS_LAMBDA(const int ed) {
            int el1 = edge_tri(2*ed + 0);
            int el2 = edge_tri(2*ed + 1);
            if (el1 < 0 || el2 < 0) return;
            real_t a1 = area(el1), a2 = area(el2);
            real_t len = Kokkos::sqrt(a1 + a2);
            int nu1 = ulev(el1) - 1, nu2 = ulev(el2) - 1;
            int nzmin = (nu1 > nu2) ? nu1 : nu2;
            int nl1 = nlev(el1) - 1, nl2 = nlev(el2) - 1;
            int nzmax = (nl1 < nl2) ? nl1 : nl2;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t u1 = uv(FESOM_ELEMVEC(el1, nz, nl) + 0)
                          - uv(FESOM_ELEMVEC(el2, nz, nl) + 0);
                real_t v1 = uv(FESOM_ELEMVEC(el1, nz, nl) + 1)
                          - uv(FESOM_ELEMVEC(el2, nz, nl) + 1);
                real_t vi = u1*u1 + v1*v1;
                real_t inner = (g1*Kokkos::sqrt(vi) > g2*vi) ? g1*Kokkos::sqrt(vi) : g2*vi;
                vi = -dt * Kokkos::sqrt((g0 > inner ? g0 : inner) * len);
                real_t mag = Kokkos::sqrt(u1*u1 + v1*v1);
                real_t viLapl = dt * ((g0h > g1h*mag) ? g0h : g1h*mag) * len;  /* 0 for CORE2 */
                real_t du = vi * (Uc(FESOM_ELEM3D(el1, nz, nl)) - Uc(FESOM_ELEM3D(el2, nz, nl))) + viLapl*u1;
                real_t dv = vi * (Vc(FESOM_ELEM3D(el1, nz, nl)) - Vc(FESOM_ELEM3D(el2, nz, nl))) + viLapl*v1;
                Kokkos::atomic_add(&uv_rhs(FESOM_ELEMVEC(el1, nz, nl) + 0), -(du / a1));
                Kokkos::atomic_add(&uv_rhs(FESOM_ELEMVEC(el1, nz, nl) + 1), -(dv / a1));
                Kokkos::atomic_add(&uv_rhs(FESOM_ELEMVEC(el2, nz, nl) + 0),  (du / a2));
                Kokkos::atomic_add(&uv_rhs(FESOM_ELEMVEC(el2, nz, nl) + 1),  (dv / a2));
            }
        });

    dyn->uv_rhs_fld.modify_device();   /* driver sync_host()s before the elem3D halo */
}

/*--- FESOM_KK_VERIFY=vfilt (M2.4) --------------------------------------------
 * visc_filt_bidiff READ-MODIFY-WRITES uv_rhs (stage-2 `+=`), so the L26
 * capture-before pattern: the driver snapshots the PRE-kernel uv_rhs (uv_rhs_in)
 * over the FULL local extent (the scatter writes halo elements too) before the IN
 * rail. Here: snapshot the KK result, restore uv_rhs_in, run the C twin (which
 * re-zeros + recomputes its own u_b/v_b scratch and re-does the stage-2 +=), diff,
 * restore KK. The Serial single-thread atomic order == the C `+=` order → max|Δ|==0.
 * Non-intrusive.
 */
void fesom_visc_filt_bidiff_verify(const struct fesom_mesh *mesh,
                                   struct fesom_dyn        *dyn,
                                   struct fesom_partit     *partit,
                                   int step_n, const real_t *uv_rhs_in)
{
    const int nl = mesh->nl;
    const int E_alloc = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    const size_t nvec = (size_t)E_alloc * (size_t)nl * 2;     /* full local uv_rhs */

    std::vector<real_t> kk_uv_rhs(dyn->uv_rhs, dyn->uv_rhs + nvec);   /* KK result (host-synced) */
    std::copy(uv_rhs_in, uv_rhs_in + nvec, dyn->uv_rhs);             /* restore pre-kernel input */
    fesom_visc_filt_bidiff(mesh, dyn, partit);                       /* C twin runs on the input */
    double dmax = 0.0;
    for (size_t i = 0; i < nvec; ++i) {
        double d = std::fabs((double)kk_uv_rhs[i] - (double)dyn->uv_rhs[i]);
        if (d > dmax) dmax = d;
    }
    std::copy(kk_uv_rhs.begin(), kk_uv_rhs.end(), dyn->uv_rhs);      /* restore KK */

    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=vfilt] step %d backend=%s  max|Δ|: uv_rhs=%.3e\n",
                step_n, backend.c_str(), dmax);
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=vfilt] FAIL: Serial must be bit-identical to the C twin "
                             "(max|Δ|=%.3e)\n", dmax);
        std::abort();
    }
}

/*===========================================================================
 * compute_hbar_ale (oce_ale.F90:1974-2116)
 *
 *   ssh_rhs_old = 0
 *   for each edge:
 *       c1 = Σ_nz (v[el1]*dx1 - u[el1]*dy1) * helem[el1, nz]
 *       c2 = -Σ_nz (v[el2]*dx2 - u[el2]*dy2) * helem[el2, nz]   (if el2 ≥ 0)
 *       ssh_rhs_old[n1] += c1 + c2
 *       ssh_rhs_old[n2] -= c1 + c2
 *   (linfs branch skips the water_flux subtraction at lines 2071-2080.)
 *
 *   hbar_old = hbar
 *   hbar = hbar_old + ssh_rhs_old * dt / areasvol[ulevels_nod2D[n], n]
 *===========================================================================*/

void fesom_compute_hbar(const struct fesom_mesh *mesh,
                        struct fesom_dyn        *dyn)
{
    const int N  = mesh->myDim_nod2D;
    const int N_alloc = mesh->myDim_nod2D + mesh->eDim_nod2D;
    /* Fortran oce_ale.F90:2014 uses `do ed=1, myDim_edge2D` only.
     * Iterating eDim_edge2D too would double-count contributions to interior
     * boundary nodes (the same physical edge is processed by the owner's
     * myDim loop too) → small per-step drift → CG NaN after ~85-95 steps. */
    const int E  = mesh->myDim_edge2D;
    const int nl = mesh->nl;
    const real_t dt = (real_t)FESOM_PHASE1_DT;

    /* Reset ssh_rhs_old (lines 2005-2009) over full local extent */
    memset(dyn->ssh_rhs_old, 0, (size_t)N_alloc * sizeof(real_t));

    /* Edge loop — accumulate transport divergence (lines 2014-2065) */
    for (int ed = 0; ed < E; ++ed) {
        int n1 = mesh->edges[2*ed + 0];
        int n2 = mesh->edges[2*ed + 1];
        int el1 = mesh->edge_tri[2*ed + 0];
        int el2 = mesh->edge_tri[2*ed + 1];

        real_t c1 = 0.0;
        if (el1 >= 0) {
            real_t dx1 = mesh->edge_cross_dxdy[4*ed + 0];
            real_t dy1 = mesh->edge_cross_dxdy[4*ed + 1];
            int nzmin = mesh->ulevels[el1] - 1;
            int nzmax = mesh->nlevels[el1] - 1;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t u = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
                real_t v = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
                real_t h = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
                c1 += (v * dx1 - u * dy1) * h;
            }
        }
        real_t c2 = 0.0;
        if (el2 >= 0) {
            real_t dx2 = mesh->edge_cross_dxdy[4*ed + 2];
            real_t dy2 = mesh->edge_cross_dxdy[4*ed + 3];
            int nzmin = mesh->ulevels[el2] - 1;
            int nzmax = mesh->nlevels[el2] - 1;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t u = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t v = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t h = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
                c2 -= (v * dx2 - u * dy2) * h;
            }
        }
        dyn->ssh_rhs_old[n1] += (c1 + c2);
        dyn->ssh_rhs_old[n2] -= (c1 + c2);
    }

    /* hbar update (lines 2090-2102): linfs skips the water_flux term.
     * Fortran oce_ale.F90:2092-2096 saves hbar_old over myDim+eDim_nod2D —
     * we must mirror that so halo hbar_old isn't stale across steps. */
    for (int n = 0; n < N_alloc; ++n) {
        mesh->hbar_old[n] = mesh->hbar[n];
    }
    for (int n = 0; n < N; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;        /* cavity guard */
        int top_layer = mesh->ulevels_nod2D[n] - 1;       /* = 0 in Phase 1 */
        real_t area = mesh->areasvol[FESOM_NODE3D(n, top_layer, nl)];
        mesh->hbar[n] = mesh->hbar_old[n]
                       + dyn->ssh_rhs_old[n] * dt / area;
    }
    /* hbar then halo-exchanged in fesom_step.c after this returns. */
}

/*===========================================================================
 * fesom_compute_hbar_kk — M4.2-b DEVICE twin (substep 10). Three launches
 * (barrier-ordered, D20): (1) zero ssh_rhs_old + save hbar_old=hbar over the
 * full local extent (the C memset + hbar_old loop, fused); (2) the edge→node
 * SCATTER of the transport divergence into ssh_rhs_old (atomic_add, D22 — the
 * per-edge level sum is sequential, the scatter in natural edge order → Serial
 * bit-identical, OpenMP/CUDA climate-close); (3) the owned hbar update reading
 * the just-saved hbar_old + just-scattered ssh_rhs_old. uv is read at edge_tri
 * (interior elements) so its owned device copy from update_vel_kk is current —
 * the driver re-pushes it (L30) since the line-472 halo wrote uv on the host.
 *===========================================================================*/
void fesom_compute_hbar_kk(struct fesom_mesh          *mesh,
                           struct fesom_dyn           *dyn,
                           const struct fesom_forcing *forcing)  /* M6.3: water_flux for the
                                 zstar elevation term; unused under linfs */
{
    const int    N       = mesh->myDim_nod2D;
    const int    N_alloc = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    E       = mesh->myDim_edge2D;
    const int    nl      = mesh->nl;
    const real_t dt      = (real_t)FESOM_PHASE1_DT;

    auto edges    = mesh->edges_fld.d();
    auto edge_tri = mesh->edge_tri_fld.d();
    auto ecd      = mesh->edge_cross_dxdy_fld.d();
    auto ulev     = mesh->ulevels_fld.d();
    auto nlev     = mesh->nlevels_fld.d();
    auto ulev_n   = mesh->ulevels_nod2D_fld.d();
    auto areasvol = mesh->areasvol_fld.d();
    auto uv       = dyn->uv_fld.d();
    auto helem    = mesh->helem_fld.d();
    auto ssh_rhs_old = dyn->ssh_rhs_old_fld.d();
    auto hbar     = mesh->hbar_fld.d();
    auto hbar_old = mesh->hbar_old_fld.d();

    /* (1) reset ssh_rhs_old + hbar_old=hbar over the full local extent. */
    Kokkos::parallel_for("fesom_hbar_pre", Kokkos::RangePolicy<>(0, N_alloc),
        KOKKOS_LAMBDA(const int n) { ssh_rhs_old(n) = 0.0; hbar_old(n) = hbar(n); });

    /* (2) edge loop — EDGE→NODE SCATTER (atomic_add, D22). */
    Kokkos::parallel_for("fesom_hbar_edge", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int ed) {
            int n1  = edges(2*ed + 0);
            int n2  = edges(2*ed + 1);
            int el1 = edge_tri(2*ed + 0);
            int el2 = edge_tri(2*ed + 1);
            real_t c1 = 0.0;
            if (el1 >= 0) {
                real_t dx1 = ecd(4*ed + 0);
                real_t dy1 = ecd(4*ed + 1);
                int nzmin = ulev(el1) - 1;
                int nzmax = nlev(el1) - 1;
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    real_t u = uv(FESOM_ELEMVEC(el1, nz, nl) + 0);
                    real_t v = uv(FESOM_ELEMVEC(el1, nz, nl) + 1);
                    real_t h = helem(FESOM_ELEM3D(el1, nz, nl));
                    c1 += (v * dx1 - u * dy1) * h;
                }
            }
            real_t c2 = 0.0;
            if (el2 >= 0) {
                real_t dx2 = ecd(4*ed + 2);
                real_t dy2 = ecd(4*ed + 3);
                int nzmin = ulev(el2) - 1;
                int nzmax = nlev(el2) - 1;
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    real_t u = uv(FESOM_ELEMVEC(el2, nz, nl) + 0);
                    real_t v = uv(FESOM_ELEMVEC(el2, nz, nl) + 1);
                    real_t h = helem(FESOM_ELEM3D(el2, nz, nl));
                    c2 -= (v * dx2 - u * dy2) * h;
                }
            }
            Kokkos::atomic_add(&ssh_rhs_old(n1),  (c1 + c2));
            Kokkos::atomic_add(&ssh_rhs_old(n2), -(c1 + c2));   /* L32 */
        });

    /* (2b) M6.3 (zstar) — the freshwater VOLUME flux enters the elevation equation
     * (oce_ale.F90:2262-2271): ssh_rhs_old(n) -= water_flux(n) * areasvol(nzmin, n).
     * OWNED nodes, cavity skip. The matching exchange_nod(ssh_rhs_old) that the Fortran emits
     * ONLY in this non-linfs branch is already covered by the always-on exchange the step does
     * right after this returns (v1.0 kept it unconditional). linfs: the term is skipped. */
    if (fesom_ale_is_zstar()) {
        auto wf = forcing->water_flux_fld.d();
        Kokkos::parallel_for("fesom_hbar_zstar_wf", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int n) {
                const int nzmin_f = ulev_n(n);             /* 1-based */
                if (nzmin_f > 1) return;                   /* cavity guard */
                ssh_rhs_old(n) -= wf(n) * areasvol(FESOM_NODE3D(n, nzmin_f - 1, nl));
            });
    }

    /* (3) hbar update over OWNED nodes (the C `for n in [0,N)`). */
    Kokkos::parallel_for("fesom_hbar_upd", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) return;                     /* cavity guard (C continue) */
            int top = ulev_n(n) - 1;                        /* = 0 in Phase 1 */
            real_t area = areasvol(FESOM_NODE3D(n, top, nl));
            hbar(n) = hbar_old(n) + ssh_rhs_old(n) * dt / area;
        });

    dyn->ssh_rhs_old_fld.modify_device();
    mesh->hbar_fld.modify_device();
    mesh->hbar_old_fld.modify_device();
}
