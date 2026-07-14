#include "fesom_ale.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_mesh.h"
#include "fesom_forcing.h"   // M6.3: water_flux for the zstar vert_vel BC
#include "fesom_partit.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"
#include "fesom_speed.hpp"   // M7 Task A.1: FESOM_SPEED_FLAT

#include <Kokkos_Core.hpp>   // M2.5: device kernels (parallel_for) + Kokkos:: math + atomic
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <vector>            // M2.5: FESOM_KK_VERIFY snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>
#include <cstdio>

/* M7 Task A.1 — FESOM_SPEED_FLAT, the two flattenable ALE column-loops (vvel divide,
 * wvel_split). One helper per TU so the lever announces itself exactly once here. */
static bool m7_ale_flat_on()
{
    static int c = -1;
    return fesom_speed_on("FLAT", &c);
}

void fesom_ale_thickness_linfs(struct fesom_mesh *mesh)
{
    /* hnode_new = hnode in linfs. Memcpy is the literal port. */
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    size_t bytes = (size_t)N * (size_t)mesh->nl * sizeof(real_t);
    memcpy(mesh->hnode_new, mesh->hnode, bytes);
}

void fesom_ale_commit_thickness(struct fesom_mesh *mesh)
{
    const int nl = mesh->nl;
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    /* helem on interior elements; halo elements get values via the
     * exchange in fesom_step. */
    int Ei = mesh->myDim_elem2D;
    /* hnode := hnode_new */
    size_t bytes = (size_t)N * (size_t)nl * sizeof(real_t);
    memcpy(mesh->hnode, mesh->hnode_new, bytes);

    for (int e = 0; e < Ei; ++e) {
        int nzmin = mesh->ulevels[e]   - 1;
        int nzmax = mesh->nlevels[e]   - 1;
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];
        for (int nz = nzmin; nz < nzmax; ++nz) {
            mesh->helem[FESOM_ELEM3D(e, nz, nl)] =
                (mesh->hnode[FESOM_NODE3D(n0, nz, nl)] +
                 mesh->hnode[FESOM_NODE3D(n1, nz, nl)] +
                 mesh->hnode[FESOM_NODE3D(n2, nz, nl)]) / 3.0;
        }
    }
}

/*
 * Vertical velocity from horizontal divergence — linfs branch only.
 *
 * Algorithm (vert_vel_ale lines 2189-2346, linfs path):
 *
 * 1. Zero w at all nodes.
 *
 * 2. For each edge ed:
 *      let n1, n2     = edges[ed]
 *      let el1, el2   = edge_tri[ed]   (el2 may be -1 for boundary)
 *      let dxL, dyL   = edge_cross_dxdy[ed][0..1]   (centroid(el1) − edge_mid)
 *      let dxR, dyR   = edge_cross_dxdy[ed][2..3]
 *
 *      For el1 (always present):
 *        c1[nz] = (uv[el1].v * dxL − uv[el1].u * dyL) * helem[el1][nz]
 *        w[n1][nz] += c1[nz]
 *        w[n2][nz] −= c1[nz]
 *
 *      For el2 if present (note negation of dxR,dyR sign):
 *        c1[nz] = -(uv[el2].v * dxR − uv[el2].u * dyR) * helem[el2][nz]
 *        w[n1][nz] += c1[nz]
 *        w[n2][nz] −= c1[nz]
 *
 * 3. Cumulative sum upward (bottom → top):
 *      for nz from nzmax-1 down to nzmin:
 *          w[n][nz] += w[n][nz + 1]
 *
 * 4. Divide by area(nz, n) → m/s.
 *
 * For Phase 1 the upper layer surface-flux correction (zlevel/zstar branches)
 * is omitted — linfs has dh/dt = 0, so w is the physical vertical velocity
 * directly.
 */
void fesom_ale_vert_vel_linfs(const struct fesom_mesh *mesh,
                              struct fesom_dyn        *dyn,
                              int                      gm_on)
{
    const int nl = mesh->nl;

    /* Step 1: zero w everywhere (all local nodes); fer_w too if GM on. */
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    /* Fortran vert_vel_ale (oce_ale.F90:2200) iterates `do ed=1, myDim_edge2D`
     * only. The earlier "myDim+eDim" was wrong — iterating halo edges adds
     * contributions to interior boundary nodes that the owner's loop also
     * adds → small per-step error → CG NaN after ~85 steps. */
    int EG = mesh->myDim_edge2D;
    memset(dyn->w, 0, (size_t)N * (size_t)nl * sizeof(real_t));
    if (gm_on) {
        memset(dyn->fer_w, 0, (size_t)N * (size_t)nl * sizeof(real_t));
    }

    /* Step 2: edge-flux accumulation. Each interior boundary node receives
     * contributions from all its incident edges via this rank's myDim_edge2D
     * (the partition replicates cross-rank edges); exchange_nod(w) afterward
     * refreshes halo entries with their owners' values.
     *
     * Phase G6a: when gm_on, the same edge iteration also accumulates fer_w
     * from dyn->fer_uv. The c1 (regular) and c2 (GM) contributions write to
     * dyn->w and dyn->fer_w respectively — disjoint targets, so the dyn->w
     * accumulator is bit-identical to the gm_on=0 path. */
    for (int ed = 0; ed < EG; ++ed) {
        int n1 = mesh->edges[2*ed + 0];
        int n2 = mesh->edges[2*ed + 1];
        int el1 = mesh->edge_tri[2*ed + 0];
        int el2 = mesh->edge_tri[2*ed + 1];

        /* Left cell — always present in a well-formed mesh */
        if (el1 >= 0) {
            real_t dxL = mesh->edge_cross_dxdy[4*ed + 0];
            real_t dyL = mesh->edge_cross_dxdy[4*ed + 1];
            int nzmin = mesh->ulevels[el1] - 1;
            int nzmax = mesh->nlevels[el1] - 1;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                /* Mirror of c1=(UV(2)*dx - UV(1)*dy)*helem; UV index 1=u, 2=v. */
                real_t u = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
                real_t v = dyn->uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
                real_t h = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
                real_t c1 = (v * dxL - u * dyL) * h;
                dyn->w[FESOM_NODE3D(n1, nz, nl)] += c1;
                dyn->w[FESOM_NODE3D(n2, nz, nl)] -= c1;
                if (gm_on) {
                    real_t fu = dyn->fer_uv[FESOM_ELEMVEC(el1, nz, nl) + 0];
                    real_t fv = dyn->fer_uv[FESOM_ELEMVEC(el1, nz, nl) + 1];
                    real_t c2 = (fv * dxL - fu * dyL) * h;
                    dyn->fer_w[FESOM_NODE3D(n1, nz, nl)] += c2;
                    dyn->fer_w[FESOM_NODE3D(n2, nz, nl)] -= c2;
                }
            }
        }
        /* Right cell — only on interior edges */
        if (el2 >= 0) {
            real_t dxR = mesh->edge_cross_dxdy[4*ed + 2];
            real_t dyR = mesh->edge_cross_dxdy[4*ed + 3];
            int nzmin = mesh->ulevels[el2] - 1;
            int nzmax = mesh->nlevels[el2] - 1;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t u = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t v = dyn->uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t h = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
                real_t c1 = -(v * dxR - u * dyR) * h;
                dyn->w[FESOM_NODE3D(n1, nz, nl)] += c1;
                dyn->w[FESOM_NODE3D(n2, nz, nl)] -= c1;
                if (gm_on) {
                    real_t fu = dyn->fer_uv[FESOM_ELEMVEC(el2, nz, nl) + 0];
                    real_t fv = dyn->fer_uv[FESOM_ELEMVEC(el2, nz, nl) + 1];
                    real_t c2 = -(fv * dxR - fu * dyR) * h;
                    dyn->fer_w[FESOM_NODE3D(n1, nz, nl)] += c2;
                    dyn->fer_w[FESOM_NODE3D(n2, nz, nl)] -= c2;
                }
            }
        }
    }

    /* Step 3: cumulative sum bottom → top (all local nodes; halo nodes will
     * be re-synced by exchange_nod(w) in the timestep wrapper). Mirror for
     * fer_w when gm_on. */
    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmax - 1; nz >= nzmin; --nz) {
            dyn->w[FESOM_NODE3D(n, nz, nl)] +=
                dyn->w[FESOM_NODE3D(n, nz + 1, nl)];
            if (gm_on) {
                dyn->fer_w[FESOM_NODE3D(n, nz, nl)] +=
                    dyn->fer_w[FESOM_NODE3D(n, nz + 1, nl)];
            }
        }
    }

    /* Step 4: divide by area(nz, n) → m/s */
    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t a = mesh->area[FESOM_NODE3D(n, nz, nl)];
            if (a > 0.0) {
                dyn->w[FESOM_NODE3D(n, nz, nl)] /= a;
                if (gm_on) {
                    dyn->fer_w[FESOM_NODE3D(n, nz, nl)] /= a;
                }
            }
        }
    }
}

/*===========================================================================
 * compute_CFLz (oce_ale.F90:2935-3022)
 *
 * At each interface nz the vertical CFL accumulates contributions from
 * both adjacent layers:
 *   iter nz=k: CFL[k]   += |W[k]|·dt / h[k]
 *              CFL[k+1]  = |W[k+1]|·dt / h[k]       (overwrite — first touch)
 *   iter nz=k+1: CFL[k+1] += |W[k+1]|·dt / h[k+1]
 * Net: CFL[interior k] = |W[k]|·dt · (1/h[k-1] + 1/h[k])
 *      CFL[surface]   = |W[nzmin]|·dt / h[nzmin]
 *      CFL[bottom]    = |W[nzmax]|·dt / h[nzmax-1]
 * We zero cfl_z first and use += throughout (mathematically equivalent).
 *===========================================================================*/

void fesom_ale_compute_cflz(const struct fesom_mesh *mesh,
                            struct fesom_dyn        *dyn)
{
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int nl = mesh->nl;
    const real_t dt = (real_t)FESOM_PHASE1_DT;

    memset(dyn->cfl_z, 0, (size_t)N * (size_t)nl * sizeof(real_t));

    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t h = mesh->hnode_new[FESOM_NODE3D(n, nz, nl)];
            if (h <= 0.0) continue;
            real_t w_top = dyn->w[FESOM_NODE3D(n, nz,     nl)];
            real_t w_bot = dyn->w[FESOM_NODE3D(n, nz + 1, nl)];
            real_t c1 = fabs(w_top) * dt / h;
            real_t c2 = fabs(w_bot) * dt / h;
            dyn->cfl_z[FESOM_NODE3D(n, nz,     nl)] += c1;
            dyn->cfl_z[FESOM_NODE3D(n, nz + 1, nl)] += c2;
        }
    }
}

/*===========================================================================
 * compute_Wvel_split (oce_ale.F90:3026-3074)
 *
 *   w_e, w_i : explicit and implicit parts of w
 *   For each (n, nz):
 *     w_e[n, nz] = w[n, nz]
 *     w_i[n, nz] = 0
 *     if (use_wsplit && cfl_z > maxcfl):
 *       dd     = max(cfl_z − maxcfl, 0) / max(maxcfl, 1e-12)
 *       w_e   *= 1 / (1 + dd)
 *       w_i    = dd / (1 + dd) · w
 *===========================================================================*/

void fesom_ale_compute_wvel_split(const struct fesom_mesh *mesh,
                                  struct fesom_dyn        *dyn)
{
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int nl = mesh->nl;
    const real_t maxcfl     = (real_t)FESOM_PHASE1_WSPLIT_MAXCFL;
    const real_t use_wsplit = (real_t)FESOM_PHASE1_USE_WSPLIT;
    const real_t inv_maxcfl = 1.0 / ((maxcfl > 1e-12) ? maxcfl : 1e-12);

    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        /* Loop over interfaces: nzmin..nzmax inclusive */
        for (int nz = nzmin; nz <= nzmax; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            real_t w   = dyn->w[k];
            real_t cfl = dyn->cfl_z[k];
            if (use_wsplit > 0.5 && cfl > maxcfl) {
                real_t excess = cfl - maxcfl;
                real_t dd = (excess > 0.0 ? excess : 0.0) * inv_maxcfl;
                real_t inv_1_dd = 1.0 / (1.0 + dd);
                dyn->w_e[k] = w * inv_1_dd;
                dyn->w_i[k] = w * dd * inv_1_dd;
            } else {
                dyn->w_e[k] = w;
                dyn->w_i[k] = 0.0;
            }
        }
    }
}

/*===========================================================================
 * M2.5 — DEVICE (Kokkos) twins. Each is a verbatim parallel_for port of its C
 * twin above; every constant / loop bound / association is copied exactly. The
 * `FESOM_NODE3D/ELEM3D/ELEMVEC(entity, lev, nl)` macros and `real_t` carry into
 * the device lambdas with the captured `nl`. INPUT views are device-current via
 * the driver IN rail (SYNC_MAP §2 row 12/14, L28); the kernels mark their outputs
 * modify_device(); the driver sync_host()s before the host halos.
 *===========================================================================*/

/*--- thickness_linfs — DEVICE (substep 12): hnode_new = hnode ---------------
 * Race-free flat copy over [0, N*nl) — the device twin of the C memcpy.
 * Bit-identical on every backend (assignment, no arithmetic). */
void fesom_ale_thickness_linfs_kk(struct fesom_mesh *mesh)
{
    const int nl    = mesh->nl;
    const int N     = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int total = N * nl;
    auto hnode     = mesh->hnode_fld.d();
    auto hnode_new = mesh->hnode_new_fld.d();

    Kokkos::parallel_for("fesom_ale_thickness", Kokkos::RangePolicy<>(0, total),
        KOKKOS_LAMBDA(const int i) { hnode_new(i) = hnode(i); });

    mesh->hnode_new_fld.modify_device();
}

/*--- commit_thickness — DEVICE (substep 14) ---------------------------------
 * hnode := hnode_new (flat copy over [0, N*nl)), then helem := mean of the 3
 * vertex hnode on OWNED elements (halo helem via the driver exchange). Two
 * parallel_fors: the launch barrier (D20) orders the helem read of the just-
 * copied hnode. Both stages race-free (each entity writes only its own slot)
 * → bit-identical on every backend. Verbatim copy of fesom_ale_commit_thickness. */
void fesom_ale_commit_thickness_kk(struct fesom_mesh *mesh)
{
    const int nl    = mesh->nl;
    const int N     = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int Ei    = mesh->myDim_elem2D;
    const int total = N * nl;
    auto hnode     = mesh->hnode_fld.d();
    auto hnode_new = mesh->hnode_new_fld.d();
    auto helem     = mesh->helem_fld.d();
    auto elnod     = mesh->elem_nodes_fld.d();
    auto ulev      = mesh->ulevels_fld.d();
    auto nlev      = mesh->nlevels_fld.d();

    Kokkos::parallel_for("fesom_ale_commit_hnode", Kokkos::RangePolicy<>(0, total),
        KOKKOS_LAMBDA(const int i) { hnode(i) = hnode_new(i); });

    Kokkos::parallel_for("fesom_ale_commit_helem", Kokkos::RangePolicy<>(0, Ei),
        KOKKOS_LAMBDA(const int e) {
            int nzmin = ulev(e) - 1;
            int nzmax = nlev(e) - 1;
            int n0 = elnod(3*e + 0);
            int n1 = elnod(3*e + 1);
            int n2 = elnod(3*e + 2);
            for (int nz = nzmin; nz < nzmax; ++nz) {
                helem(FESOM_ELEM3D(e, nz, nl)) =
                    (hnode(FESOM_NODE3D(n0, nz, nl)) +
                     hnode(FESOM_NODE3D(n1, nz, nl)) +
                     hnode(FESOM_NODE3D(n2, nz, nl))) / 3.0;
            }
        });

    mesh->hnode_fld.modify_device();
    mesh->helem_fld.modify_device();
}

/*--- vert_vel_linfs — DEVICE (substep 12): continuity integral --------------
 * Four parallel_fors mirroring the C body 1:1 (the launch barriers give the
 * inter-step ordering, D20):
 *   1. zero w (and fer_w if GM) over [0, N*nl) — device twin of the memset.
 *   2. edge-flux accumulation — an EDGE→NODE SCATTER ported with Kokkos::atomic_add
 *      (D22): Serial single-thread ordered += → bit-identical to the C edge loop;
 *      OpenMP/CUDA reorder the atomics → climate-close (≲1e-12, the D5 ladder).
 *      The n2 atomic adds the exact negation (-c1), IEEE-equal to the C `-=`.
 *   3. cumulative sum bottom→top: per-node, sequential in level INSIDE the lambda;
 *      each node owns its own column → race-free across nodes (a private scan, not
 *      a scatter) → bit-identical on Serial AND OpenMP.
 *   4. divide by area(nz,n): per-node map, race-free.
 * gm_on is captured by value; when 0 (the golden path) the fer_* branches are dead
 * but PRESERVED verbatim (no-simplification rule). No internal halo — exchange_nod(w)
 * is the driver halo. Verbatim copy of fesom_ale_vert_vel_linfs. */
void fesom_ale_vert_vel_linfs_kk(const struct fesom_mesh *mesh,
                                 struct fesom_dyn        *dyn,
                                 int                      gm_on)
{
    const int nl    = mesh->nl;
    const int N     = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int EG    = mesh->myDim_edge2D;
    const int total = N * nl;

    auto w          = dyn->w_fld.d();
    auto fer_w      = dyn->fer_w_fld.d();
    auto uv         = dyn->uv_fld.d();
    auto fer_uv     = dyn->fer_uv_fld.d();
    auto helem      = mesh->helem_fld.d();
    auto edges      = mesh->edges_fld.d();
    auto edge_tri   = mesh->edge_tri_fld.d();
    auto edge_cross = mesh->edge_cross_dxdy_fld.d();
    auto ulev_e     = mesh->ulevels_fld.d();
    auto nlev_e     = mesh->nlevels_fld.d();
    auto ulev_n     = mesh->ulevels_nod2D_fld.d();
    auto nlev_n     = mesh->nlevels_nod2D_fld.d();
    auto area       = mesh->area_fld.d();

    /* Step 1: zero w everywhere (all local nodes); fer_w too if GM on. */
    Kokkos::parallel_for("fesom_ale_vvel_zero", Kokkos::RangePolicy<>(0, total),
        KOKKOS_LAMBDA(const int i) { w(i) = 0.0; if (gm_on) fer_w(i) = 0.0; });

    /* Step 2: edge-flux accumulation — EDGE→NODE SCATTER (atomic_add, D22). */
    Kokkos::parallel_for("fesom_ale_vvel_scatter", Kokkos::RangePolicy<>(0, EG),
        KOKKOS_LAMBDA(const int ed) {
            int n1  = edges(2*ed + 0);
            int n2  = edges(2*ed + 1);
            int el1 = edge_tri(2*ed + 0);
            int el2 = edge_tri(2*ed + 1);

            if (el1 >= 0) {
                real_t dxL = edge_cross(4*ed + 0);
                real_t dyL = edge_cross(4*ed + 1);
                int nzmin = ulev_e(el1) - 1;
                int nzmax = nlev_e(el1) - 1;
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    real_t u = uv(FESOM_ELEMVEC(el1, nz, nl) + 0);
                    real_t v = uv(FESOM_ELEMVEC(el1, nz, nl) + 1);
                    real_t h = helem(FESOM_ELEM3D(el1, nz, nl));
                    real_t c1 = (v * dxL - u * dyL) * h;
                    Kokkos::atomic_add(&w(FESOM_NODE3D(n1, nz, nl)),  c1);
                    Kokkos::atomic_add(&w(FESOM_NODE3D(n2, nz, nl)), -c1);
                    if (gm_on) {
                        real_t fu = fer_uv(FESOM_ELEMVEC(el1, nz, nl) + 0);
                        real_t fv = fer_uv(FESOM_ELEMVEC(el1, nz, nl) + 1);
                        real_t c2 = (fv * dxL - fu * dyL) * h;
                        Kokkos::atomic_add(&fer_w(FESOM_NODE3D(n1, nz, nl)),  c2);
                        Kokkos::atomic_add(&fer_w(FESOM_NODE3D(n2, nz, nl)), -c2);
                    }
                }
            }
            if (el2 >= 0) {
                real_t dxR = edge_cross(4*ed + 2);
                real_t dyR = edge_cross(4*ed + 3);
                int nzmin = ulev_e(el2) - 1;
                int nzmax = nlev_e(el2) - 1;
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    real_t u = uv(FESOM_ELEMVEC(el2, nz, nl) + 0);
                    real_t v = uv(FESOM_ELEMVEC(el2, nz, nl) + 1);
                    real_t h = helem(FESOM_ELEM3D(el2, nz, nl));
                    real_t c1 = -(v * dxR - u * dyR) * h;
                    Kokkos::atomic_add(&w(FESOM_NODE3D(n1, nz, nl)),  c1);
                    Kokkos::atomic_add(&w(FESOM_NODE3D(n2, nz, nl)), -c1);
                    if (gm_on) {
                        real_t fu = fer_uv(FESOM_ELEMVEC(el2, nz, nl) + 0);
                        real_t fv = fer_uv(FESOM_ELEMVEC(el2, nz, nl) + 1);
                        real_t c2 = -(fv * dxR - fu * dyR) * h;
                        Kokkos::atomic_add(&fer_w(FESOM_NODE3D(n1, nz, nl)),  c2);
                        Kokkos::atomic_add(&fer_w(FESOM_NODE3D(n2, nz, nl)), -c2);
                    }
                }
            }
        });

    /* Step 3: cumulative sum bottom → top (per-node sequential scan). */
    Kokkos::parallel_for("fesom_ale_vvel_cumsum", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmax - 1; nz >= nzmin; --nz) {
                w(FESOM_NODE3D(n, nz, nl)) += w(FESOM_NODE3D(n, nz + 1, nl));
                if (gm_on)
                    fer_w(FESOM_NODE3D(n, nz, nl)) += fer_w(FESOM_NODE3D(n, nz + 1, nl));
            }
        });

    /* Step 4: divide by area(nz, n) → m/s.
     * M7 Task A.1 (FESOM_SPEED_FLAT): pure per-(n,nz) map — each slot reads, divides and
     * writes ONLY itself — so one thread per slot is bit-identical to one thread per column.
     * (Steps 2 and 3 above are NOT flattenable: the scatter's per-column atomic order and the
     * cumsum's sequential bottom→top scan ARE the bit-identity. They are packages B.3 / C.2.) */
    if (m7_ale_flat_on()) {
        Kokkos::parallel_for("fesom_ale_vvel_divide_flat", Kokkos::RangePolicy<>(0, total),
            KOKKOS_LAMBDA(const int i) {
                const int n  = i / nl;
                const int nz = i - n * nl;
                const int nzmin = ulev_n(n) - 1;
                const int nzmax = nlev_n(n) - 1;
                if (nz < nzmin || nz >= nzmax) return;   /* legacy: nzmin .. nzmax-1 */
                real_t a = area((size_t)i);              /* FESOM_NODE3D(n,nz,nl) == i */
                if (a > 0.0) {
                    w((size_t)i) /= a;
                    if (gm_on)
                        fer_w((size_t)i) /= a;
                }
            });
    } else {
    Kokkos::parallel_for("fesom_ale_vvel_divide", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t a = area(FESOM_NODE3D(n, nz, nl));
                if (a > 0.0) {
                    w(FESOM_NODE3D(n, nz, nl)) /= a;
                    if (gm_on)
                        fer_w(FESOM_NODE3D(n, nz, nl)) /= a;
                }
            }
        });
    }

    dyn->w_fld.modify_device();
    if (gm_on) dyn->fer_w_fld.modify_device();
}

/*--- compute_cflz — DEVICE (substep 12) -------------------------------------
 * Per-node parallel_for. Each node first zeros its OWN [0,nl) column (the device
 * twin of the C memset, restricted to the node's disjoint slice), then accumulates
 * cfl_z[nz] / cfl_z[nz+1] over its active levels. The same column is touched only
 * by this node's thread, sequentially → race-free, NOT a cross-thread scatter →
 * bit-identical on Serial AND OpenMP. `fabs`→`Kokkos::fabs` (sign-bit clear, exact
 * on every backend). Verbatim copy of fesom_ale_compute_cflz. */
void fesom_ale_compute_cflz_kk(const struct fesom_mesh *mesh,
                               struct fesom_dyn        *dyn)
{
    const int nl    = mesh->nl;
    const int N     = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const real_t dt = (real_t)FESOM_PHASE1_DT;
    auto cfl_z     = dyn->cfl_z_fld.d();
    auto w         = dyn->w_fld.d();
    auto hnode_new = mesh->hnode_new_fld.d();
    auto ulev_n    = mesh->ulevels_nod2D_fld.d();
    auto nlev_n    = mesh->nlevels_nod2D_fld.d();

    Kokkos::parallel_for("fesom_ale_cflz", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            for (int nz = 0; nz < nl; ++nz) cfl_z(FESOM_NODE3D(n, nz, nl)) = 0.0;
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t h = hnode_new(FESOM_NODE3D(n, nz, nl));
                if (h <= 0.0) continue;
                real_t w_top = w(FESOM_NODE3D(n, nz,     nl));
                real_t w_bot = w(FESOM_NODE3D(n, nz + 1, nl));
                real_t c1 = Kokkos::fabs(w_top) * dt / h;
                real_t c2 = Kokkos::fabs(w_bot) * dt / h;
                cfl_z(FESOM_NODE3D(n, nz,     nl)) += c1;
                cfl_z(FESOM_NODE3D(n, nz + 1, nl)) += c2;
            }
        });

    dyn->cfl_z_fld.modify_device();
}

/*--- compute_wvel_split — DEVICE (substep 12) -------------------------------
 * ⚠️ use_wsplit=.false. (FESOM_PHASE1_USE_WSPLIT=0) preserved verbatim: the full
 * use_wsplit branch is kept; with the flag 0 every interface takes the else
 * (w_e=w, w_i=0). Pure per-(n,nz) map, each slot written once → race-free,
 * bit-identical on every backend. Verbatim copy of fesom_ale_compute_wvel_split. */
void fesom_ale_compute_wvel_split_kk(const struct fesom_mesh *mesh,
                                     struct fesom_dyn        *dyn)
{
    const int nl = mesh->nl;
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const real_t maxcfl     = (real_t)FESOM_PHASE1_WSPLIT_MAXCFL;
    const real_t use_wsplit = (real_t)FESOM_PHASE1_USE_WSPLIT;
    const real_t inv_maxcfl = 1.0 / ((maxcfl > 1e-12) ? maxcfl : 1e-12);

    auto w      = dyn->w_fld.d();
    auto cfl_z  = dyn->cfl_z_fld.d();
    auto w_e    = dyn->w_e_fld.d();
    auto w_i    = dyn->w_i_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto nlev_n = mesh->nlevels_nod2D_fld.d();

    /* M7 Task A.1 (FESOM_SPEED_FLAT): pure per-(n,nz) map, every slot written once →
     * flat is bit-identical. ⚠️ The legacy level bound is INCLUSIVE (nz <= nzmax), so the
     * flat guard must be `nz > nzmax`, not `nz >= nzmax`. */
    if (m7_ale_flat_on()) {
        Kokkos::parallel_for("fesom_ale_wvel_split_flat", Kokkos::RangePolicy<>(0, N * nl),
            KOKKOS_LAMBDA(const int i) {
                const int n  = i / nl;
                const int nz = i - n * nl;
                const int nzmin = ulev_n(n) - 1;
                const int nzmax = nlev_n(n) - 1;
                if (nz < nzmin || nz > nzmax) return;    /* legacy: nzmin .. nzmax INCLUSIVE */
                size_t k = (size_t)i;                    /* FESOM_NODE3D(n,nz,nl) == i */
                real_t wv  = w(k);
                real_t cfl = cfl_z(k);
                if (use_wsplit > 0.5 && cfl > maxcfl) {
                    real_t excess = cfl - maxcfl;
                    real_t dd = (excess > 0.0 ? excess : 0.0) * inv_maxcfl;
                    real_t inv_1_dd = 1.0 / (1.0 + dd);
                    w_e(k) = wv * inv_1_dd;
                    w_i(k) = wv * dd * inv_1_dd;
                } else {
                    w_e(k) = wv;
                    w_i(k) = 0.0;
                }
            });
    } else {
    Kokkos::parallel_for("fesom_ale_wvel_split", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin; nz <= nzmax; ++nz) {
                size_t k = FESOM_NODE3D(n, nz, nl);
                real_t wv  = w(k);
                real_t cfl = cfl_z(k);
                if (use_wsplit > 0.5 && cfl > maxcfl) {
                    real_t excess = cfl - maxcfl;
                    real_t dd = (excess > 0.0 ? excess : 0.0) * inv_maxcfl;
                    real_t inv_1_dd = 1.0 / (1.0 + dd);
                    w_e(k) = wv * inv_1_dd;
                    w_i(k) = wv * dd * inv_1_dd;
                } else {
                    w_e(k) = wv;
                    w_i(k) = 0.0;
                }
            }
        });
    }

    dyn->w_e_fld.modify_device();
    dyn->w_i_fld.modify_device();
}

/*===========================================================================
 * M2.5 — FESOM_KK_VERIFY=ale gates. Each runs the untouched C twin beside the
 * device result on the same live state, reports per-field max|Δ|, and asserts
 * max|Δ|==0 on Serial (the bit-identity oracle). Non-intrusive: snapshots the KK
 * result (read from the raw alias, host-current after the driver OUT-rail
 * sync_host), runs the C twin (overwriting the raw alias), diffs, restores KK.
 * No L26 capture-before is needed for ANY ALE kernel — every output is a FULL
 * overwrite derived from inputs the kernel does not modify (thickness/commit read
 * hnode/hnode_new; vert_vel/cflz/wvel_split zero-then-fill from intact uv/helem/
 * w/cfl_z), so the C twin recomputes them from the same intact inputs.
 *===========================================================================*/

static double fesom_ale_maxdiff_(const std::vector<real_t> &kk, const real_t *c, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)kk[i] - (double)c[i]);
        if (d > m) m = d;
    }
    return m;
}

static void fesom_ale_verify_report_(int step_n, const char *what, double dmax)
{
    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=ale] FAIL (%s) step %d: Serial must be bit-identical "
                             "to the C twin (max|Δ|=%.3e)\n", what, step_n, dmax);
        std::abort();
    }
}

void fesom_ale_thickness_verify(struct fesom_mesh *mesh, int step_n)
{
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    std::vector<real_t> kk(mesh->hnode_new, mesh->hnode_new + total);   /* KK result (host-synced) */
    fesom_ale_thickness_linfs(mesh);                                    /* C twin: hnode_new = hnode */
    double d = fesom_ale_maxdiff_(kk, mesh->hnode_new, total);
    std::copy(kk.begin(), kk.end(), mesh->hnode_new);                   /* restore KK */
    std::printf("[FESOM_KK_VERIFY=ale] step %d backend=%s  max|Δ|: hnode_new=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(),d);
    fesom_ale_verify_report_(step_n, "thickness", d);
}

void fesom_ale_vert_vel_verify(const struct fesom_mesh *mesh,
                               struct fesom_dyn *dyn, int gm_on, int step_n)
{
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    std::vector<real_t> kk_w(dyn->w, dyn->w + total);
    std::vector<real_t> kk_fer_w;
    if (gm_on) kk_fer_w.assign(dyn->fer_w, dyn->fer_w + total);
    fesom_ale_vert_vel_linfs(mesh, dyn, gm_on);                         /* C twin recomputes w[,fer_w] */
    double d_w   = fesom_ale_maxdiff_(kk_w, dyn->w, total);
    double d_fer = gm_on ? fesom_ale_maxdiff_(kk_fer_w, dyn->fer_w, total) : 0.0;
    std::copy(kk_w.begin(), kk_w.end(), dyn->w);                        /* restore KK */
    if (gm_on) std::copy(kk_fer_w.begin(), kk_fer_w.end(), dyn->fer_w);
    std::printf("[FESOM_KK_VERIFY=ale] step %d backend=%s  max|Δ|: w=%.3e fer_w=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(),d_w, d_fer);
    fesom_ale_verify_report_(step_n, "vert_vel", std::max(d_w, d_fer));
}

void fesom_ale_compute_cflz_verify(const struct fesom_mesh *mesh,
                                   struct fesom_dyn *dyn, int step_n)
{
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    std::vector<real_t> kk(dyn->cfl_z, dyn->cfl_z + total);
    fesom_ale_compute_cflz(mesh, dyn);                                  /* C twin */
    double d = fesom_ale_maxdiff_(kk, dyn->cfl_z, total);
    std::copy(kk.begin(), kk.end(), dyn->cfl_z);                        /* restore KK */
    std::printf("[FESOM_KK_VERIFY=ale] step %d backend=%s  max|Δ|: cfl_z=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(),d);
    fesom_ale_verify_report_(step_n, "cflz", d);
}

void fesom_ale_compute_wvel_split_verify(const struct fesom_mesh *mesh,
                                         struct fesom_dyn *dyn, int step_n)
{
    const int nl = mesh->nl;
    const size_t total = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    std::vector<real_t> kk_we(dyn->w_e, dyn->w_e + total);
    std::vector<real_t> kk_wi(dyn->w_i, dyn->w_i + total);
    fesom_ale_compute_wvel_split(mesh, dyn);                            /* C twin */
    double d_we = fesom_ale_maxdiff_(kk_we, dyn->w_e, total);
    double d_wi = fesom_ale_maxdiff_(kk_wi, dyn->w_i, total);
    std::copy(kk_we.begin(), kk_we.end(), dyn->w_e);                    /* restore KK */
    std::copy(kk_wi.begin(), kk_wi.end(), dyn->w_i);
    std::printf("[FESOM_KK_VERIFY=ale] step %d backend=%s  max|Δ|: w_e=%.3e w_i=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(),d_we, d_wi);
    fesom_ale_verify_report_(step_n, "wvel_split", std::max(d_we, d_wi));
}

void fesom_ale_commit_verify(struct fesom_mesh *mesh, int step_n)
{
    const int nl = mesh->nl;
    const size_t total_n = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    const size_t total_e = (size_t)mesh->myDim_elem2D * (size_t)nl;     /* owned-elem helem */
    std::vector<real_t> kk_hnode(mesh->hnode, mesh->hnode + total_n);
    std::vector<real_t> kk_helem(mesh->helem, mesh->helem + total_e);
    fesom_ale_commit_thickness(mesh);                                   /* C twin */
    double d_hnode = fesom_ale_maxdiff_(kk_hnode, mesh->hnode, total_n);
    double d_helem = fesom_ale_maxdiff_(kk_helem, mesh->helem, total_e);
    std::copy(kk_hnode.begin(), kk_hnode.end(), mesh->hnode);           /* restore KK */
    std::copy(kk_helem.begin(), kk_helem.end(), mesh->helem);
    std::printf("[FESOM_KK_VERIFY=ale] step %d backend=%s  max|Δ|: hnode=%.3e helem=%.3e\n",
                step_n, std::string(Kokkos::DefaultExecutionSpace::name()).c_str(),d_hnode, d_helem);
    fesom_ale_verify_report_(step_n, "commit", std::max(d_hnode, d_helem));
}

/*===========================================================================================
 * M6.3 — zstar vertical coordinate. Everything below is inert under the default (linfs).
 *===========================================================================================*/

/* Read-once mode switch — literal port of the C's fesom_ale_mode_init (fesom_ale.c:18-33). */
static int    s_ale_zstar     = 0;
static int    s_use_virt_salt = 1;
static real_t s_is_nonlinfs   = 0.0;

void fesom_ale_mode_init(void)
{
    const char *e = getenv("FESOM_ALE");
    if (!e || !e[0] || strcmp(e, "linfs") == 0) {
        s_ale_zstar = 0;
    } else if (strcmp(e, "zstar") == 0) {
        s_ale_zstar = 1;
    } else {
        /* zlevel + the local-zstar fallback are NOT ported (no reference run exercises them) —
         * a gate-only abort, exactly as the C does. */
        fprintf(stderr, "FESOM_ALE=%s not supported (linfs|zstar)\n", e);
        exit(1);
    }
    /* DERIVED in the Fortran (oce_setup_step.F90) from which_ALE — not independent knobs:
     * zstar => use_virt_salt = .false. and is_nonlinfs = 1.0. Under linfs is_nonlinfs = 0.0,
     * which is exactly what makes every non-linfs term drop out identically. */
    s_use_virt_salt = s_ale_zstar ? 0 : 1;
    s_is_nonlinfs   = s_ale_zstar ? 1.0 : 0.0;

    /* (The temporary "zstar incomplete" abort that lived here from Task 3.1 is GONE: the chain
     * closed at Task 3.6 -- geometry + thickness (3.1), the forcing flip (3.2), the SSH
     * stiffness update + rhs/hbar tails (3.3), the vert_vel branch that writes hnode_new (3.4),
     * the Shchepetkin PGF + hpressure gating (3.5), and the geometry re-points (3.6).) */
}

int    fesom_ale_is_zstar(void)      { return s_ale_zstar; }
int    fesom_ale_use_virt_salt(void) { return s_use_virt_salt; }
real_t fesom_is_nonlinfs(void)       { return s_is_nonlinfs; }

/*--- zstar thickness INIT (oce_ale.F90:1134-1218) -------------------------------------------
 * Host-side: it runs ONCE at startup, before any device kernel, and its inputs (zbar, hbar,
 * nlevels*) are all host-current there. The Fields it writes (hnode, hnode_new, helem, dhe) are
 * pushed to the device by the caller's existing geometry sync.
 */
void fesom_ale_init_thickness_zstar(struct fesom_mesh   *mesh,
                                    struct fesom_dyn    *dyn,
                                    struct fesom_partit *partit)
{
    const int    nl    = mesh->nl;
    const int    N     = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const real_t alpha = (real_t)FESOM_PHASE1_ALPHA;
    const real_t dt    = (real_t)FESOM_PHASE1_DT;

    /* Mode-shared pre-block (oce_ale.F90:1002-1007). All zeros at cold start (hbar = hbar_old
     * = 0); ported literally for shape fidelity (it is the same code on a restart). */
    for (int n = 0; n < N; ++n) {
        int ul = mesh->ulevels_nod2D[n] - 1;
        dyn->ssh_rhs_old[n] = (mesh->hbar[n] - mesh->hbar_old[n])
                            * mesh->areasvol[FESOM_NODE3D(n, ul, nl)] / dt;
    }
    for (int n = 0; n < N; ++n)
        dyn->eta_n[n] = alpha * mesh->hbar_old[n] + (1.0 - alpha) * mesh->hbar[n];

    /* zstar node thicknesses (:1134-1172). 1-based Fortran bounds annotated. */
    for (int n = 0; n < N; ++n) {
        int nzmin_f = mesh->ulevels_nod2D[n];        /* 1-based top level    */
        int nzmax_f = mesh->nlevels_nod2D[n] - 1;    /* 1-based bottom layer */
        int min_f   = mesh->nlevels_nod2D_min[n];    /* K_v-, 1-based        */

        if (nzmin_f == 1) {
            /* depth anomaly down to the last level whose scalar prism does NOT intersect the
             * bottom: dd = zbar(1) - zbar(min-1)  (invariant 3: bottom protection) */
            const real_t dd = mesh->zbar[0] - mesh->zbar[min_f - 2];

            /* distribute hbar linearly: Fortran nz = 1 .. min-2 */
            for (int nz = 0; nz <= min_f - 3; ++nz)
                mesh->hnode[FESOM_NODE3D(n, nz, nl)] =
                    (mesh->zbar[nz] - mesh->zbar[nz + 1]) * (1.0 + mesh->hbar[n] / dd);
            /* bottom-intersecting levels keep NOMINAL spacing: Fortran nz = min-1 .. nzmax-1 */
            for (int nz = min_f - 2; nz <= nzmax_f - 2; ++nz)
                mesh->hnode[FESOM_NODE3D(n, nz, nl)] = mesh->zbar[nz] - mesh->zbar[nz + 1];
        } else {
            /* cavity arm (unreachable in our meshes): fixed geometry */
            for (int nz = nzmin_f - 1; nz <= nzmax_f - 2; ++nz)
                mesh->hnode[FESOM_NODE3D(n, nz, nl)] =
                      mesh->zbar_3d_n[FESOM_NODE3D(n, nz,     nl)]
                    - mesh->zbar_3d_n[FESOM_NODE3D(n, nz + 1, nl)];
        }
        /* bottom layer: Fortran hnode(nzmax) = bottom_node_thickness */
        mesh->hnode[FESOM_NODE3D(n, nzmax_f - 1, nl)] = mesh->bottom_node_thickness[n];
    }

    /* zstar element thicknesses + dhe (:1176-1206). Owned elements only, then exchange.
     * Reads hnode at halo nodes (filled above, full local extent). */
    for (int e = 0; e < mesh->myDim_elem2D; ++e) {
        int nzmin_f = mesh->ulevels[e];
        int nzmax_f = mesh->nlevels[e] - 1;
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];

        if (nzmin_f == 1) {
            mesh->dhe[e] = (mesh->hbar[n0] + mesh->hbar[n1] + mesh->hbar[n2]) / 3.0;
            mesh->helem[FESOM_ELEM3D(e, 0, nl)] =
                (mesh->hnode[FESOM_NODE3D(n0, 0, nl)] +
                 mesh->hnode[FESOM_NODE3D(n1, 0, nl)] +
                 mesh->hnode[FESOM_NODE3D(n2, 0, nl)]) / 3.0;
        } else {
            /* cavity arm (unreachable): the Fortran zeroes the WHOLE dhe array here and pins the
             * top helem to the cavity surface depth. No cavities here — keep the guard, make the
             * body fatal if it is ever hit (exactly as the C does). */
            fprintf(stderr, "init_thickness_zstar: cavity element %d (ulevels=%d) not supported\n",
                    e, nzmin_f);
            exit(1);
        }
        /* Fortran nz = nzmin+1 .. nzmax-1  (2..nlevels-2) */
        for (int nz = 1; nz <= nzmax_f - 2; ++nz)
            mesh->helem[FESOM_ELEM3D(e, nz, nl)] =
                (mesh->hnode[FESOM_NODE3D(n0, nz, nl)] +
                 mesh->hnode[FESOM_NODE3D(n1, nz, nl)] +
                 mesh->hnode[FESOM_NODE3D(n2, nz, nl)]) / 3.0;
        /* bottom layer: Fortran helem(nzmax) = bottom_elem_thickness */
        mesh->helem[FESOM_ELEM3D(e, nzmax_f - 1, nl)] = mesh->bottom_elem_thickness[e];
    }

    /* hnode_new = hnode (:1217) — only the variable part is updated later. */
    memcpy(mesh->hnode_new, mesh->hnode, (size_t)N * (size_t)nl * sizeof(real_t));

    /* exchange_elem(helem) (:1218) */
    if (partit && partit->npes > 1)
        fesom_exchange_elem3D(mesh->helem, nl, partit);
}

/*--- zstar thickness COMMIT — DEVICE (oce_ale.F90:1382-1440) --------------------------------
 * This is where the geometry goes LIVE: hnode := hnode_new, and zbar_3d_n / Z_3d_n are rebuilt
 * from it bottom->top. Runs over myDim+eDim and READS the hnode_new halo, which vert_vel_ale
 * exchanged (invariant 4 — this is the M5.20 hnode_new rail, restored under zstar).
 *
 * Per-node: a strictly sequential bottom->top recurrence within each column (zbar_3d_n(nz)
 * depends on zbar_3d_n(nz+1)), so one thread per NODE owning its whole column — the D19
 * entity-outer/level-inner shape. No cross-entity write => Serial == the C loop order.
 * The element helem mean is a per-element GATHER => race-free, no atomics.
 */
void fesom_ale_update_thickness_zstar_kk(struct fesom_mesh   *mesh,
                                         struct fesom_partit *partit)
{
    const int nl = mesh->nl;
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int Eo = mesh->myDim_elem2D;

    auto hnode     = mesh->hnode_fld.d();
    auto hnode_new = mesh->hnode_new_fld.d();
    auto helem     = mesh->helem_fld.d();
    auto zbar3d    = mesh->zbar_3d_n_fld.d();
    auto Z3d       = mesh->Z_3d_n_fld.d();
    auto ulev_n    = mesh->ulevels_nod2D_fld.d();
    auto nlev_min  = mesh->nlevels_nod2D_min_fld.d();
    auto ulev_e    = mesh->ulevels_fld.d();
    auto nlev_e    = mesh->nlevels_fld.d();
    auto elnod     = mesh->elem_nodes_fld.d();

    /* node commit, bottom->top (:1382-1416), myDim+eDim */
    Kokkos::parallel_for("ale_zstar_commit_node", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            const int nzmin_f = ulev_n(n);
            if (nzmin_f > 1) return;                       /* cavity guard */
            const int top_f = nlev_min(n) - 2;             /* 1-based commit top */
            for (int nzf = top_f; nzf >= nzmin_f; --nzf) {
                const int nz = nzf - 1;                    /* 0-based layer */
                const size_t k  = FESOM_NODE3D(n, nz,     nl);
                const size_t k1 = FESOM_NODE3D(n, nz + 1, nl);
                hnode(k)  = hnode_new(k);
                zbar3d(k) = zbar3d(k1) + hnode_new(k);
                Z3d(k)    = zbar3d(k1) + hnode_new(k) / 2.0;
            }
        });
    mesh->hnode_fld.modify_device();
    mesh->zbar_3d_n_fld.modify_device();
    mesh->Z_3d_n_fld.modify_device();

    /* element mean thickness (:1421-1434): Fortran nz = 1 .. nlevels(elem)-2; the bottom helem
     * keeps bottom_elem_thickness (never rewritten here). */
    Kokkos::parallel_for("ale_zstar_commit_elem", Kokkos::RangePolicy<>(0, Eo),
        KOKKOS_LAMBDA(const int e) {
            const int nzmin_f = ulev_e(e);
            if (nzmin_f > 1) return;                       /* cavity guard */
            const int nzmax_f = nlev_e(e) - 1;
            const int n0 = elnod(3*e + 0);
            const int n1 = elnod(3*e + 1);
            const int n2 = elnod(3*e + 2);
            for (int nzf = nzmin_f; nzf <= nzmax_f - 1; ++nzf) {
                const int nz = nzf - 1;
                helem(FESOM_ELEM3D(e, nz, nl)) =
                    (hnode(FESOM_NODE3D(n0, nz, nl)) +
                     hnode(FESOM_NODE3D(n1, nz, nl)) +
                     hnode(FESOM_NODE3D(n2, nz, nl))) / 3.0;
            }
        });
    mesh->helem_fld.modify_device();

    /* exchange_elem(helem) (:1440) */
    fesom_halo_field(mesh->helem_fld, FESOM_HALO_ELEM3D, nl, 1, partit);
}

/*--- M6.3 (zstar): vert_vel_ale zstar branch — DEVICE (oce_ale.F90:2755-2827) ----------------
 * Distributes the per-step SSH change (hbar − hbar_old) proportionally over the STRETCHED part
 * of each column, on top of the shared divergence-built Wvel:
 *
 *   dd1  = zbar_3d_n(nzmax, n),  nzmax = nlevels_nod2D_min − 1     (LIVE geometry)
 *   dd   = (hbar − hbar_old) / (zbar_3d_n(1, n) − dd1);   dddt = dd/dt
 *   nz = 1..nzmax−1:
 *          Wvel(nz)      −= (zbar_3d_n(nz, n) − dd1) · dddt
 *          hnode_new(nz)  = hnode(nz) + (zbar_3d_n(nz) − zbar_3d_n(nz+1)) · dd
 *   Wvel(1) −= water_flux(n)                    (the real volume flux as the surface BC)
 *
 * ⚠️ INVARIANT 5: the Wvel correction is the vertically-INTEGRATED form
 * −(zbar_3d_n(nz) − dd1)·dddt — it is the bottom-to-top SUM of the layer dh/dt, NOT a per-layer
 * h·dd/dt. Port it as written, and using the CURRENT (pre-update) zbar_3d_n.
 * ⚠️ INVARIANT 3: the loop stops at nlevels_nod2D_min − 1, so bottom-intersecting levels are
 * never stretched.
 *
 * OWNED nodes only; the caller exchanges BOTH Wvel and hnode_new (:2870-2871) — hnode_new's halo
 * is what the Z1 commit (myDim+eDim) reads. Cavity columns (nzmin>1) are left untouched.
 *
 * The Fortran's NaN / negative-hnode_new checks (:2812-2868) are write-and-continue warnings.
 * They are NOT ported into the device kernel: a per-thread fprintf is not device-callable, and a
 * Kokkos::abort would turn a Fortran WARNING into a hard failure — changing behaviour. The Z-ladder
 * gates (bit-id vs the C oracle, then the CUDA gate, then the 1-yr climate) are what actually
 * catch a broken column here, and they catch it harder than a printf would.
 */
void fesom_ale_vert_vel_zstar_kk(struct fesom_mesh          *mesh,
                                 struct fesom_dyn           *dyn,
                                 const struct fesom_forcing *forcing)
{
    const int    nl = mesh->nl;
    const int    N  = mesh->myDim_nod2D;
    const real_t dt = (real_t)FESOM_PHASE1_DT;

    auto w         = dyn->w_fld.d();
    auto hnode     = mesh->hnode_fld.d();
    auto hnode_new = mesh->hnode_new_fld.d();
    auto zbar3d    = mesh->zbar_3d_n_fld.d();
    auto hbar      = mesh->hbar_fld.d();
    auto hbar_old  = mesh->hbar_old_fld.d();
    auto ulev_n    = mesh->ulevels_nod2D_fld.d();
    auto nlev_min  = mesh->nlevels_nod2D_min_fld.d();
    auto wf        = forcing->water_flux_fld.d();

    Kokkos::parallel_for("fesom_ale_vert_vel_zstar", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            const int nzmin_f = ulev_n(n);                 /* 1-based */
            const int nzmax_f = nlev_min(n) - 1;           /* 1-based */
            if (nzmin_f != 1) return;                      /* cavity -> linfs-like */

            const real_t dd1 = zbar3d(FESOM_NODE3D(n, nzmax_f - 1, nl));
            real_t dd = zbar3d(FESOM_NODE3D(n, 0, nl)) - dd1;
            dd = (hbar(n) - hbar_old(n)) / dd;
            const real_t dddt = dd / dt;

            for (int nzf = nzmin_f; nzf <= nzmax_f - 1; ++nzf) {
                const int    nz = nzf - 1;                 /* 0-based layer */
                const size_t k  = FESOM_NODE3D(n, nz,     nl);
                const size_t k1 = FESOM_NODE3D(n, nz + 1, nl);
                w(k)         -= (zbar3d(k) - dd1) * dddt;
                hnode_new(k)  = hnode(k) + (zbar3d(k) - zbar3d(k1)) * dd;
            }

            /* surface freshwater flux as the upper continuity BC (:2809) */
            w(FESOM_NODE3D(n, nzmin_f - 1, nl)) -= wf(n);
        });

    dyn->w_fld.modify_device();
    mesh->hnode_new_fld.modify_device();
}
