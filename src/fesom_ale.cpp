#include "fesom_ale.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_mesh.h"

#include <Kokkos_Core.hpp>   // M2.5: device kernels (parallel_for) + Kokkos:: math + atomic
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>            // M2.5: FESOM_KK_VERIFY snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>
#include <cstdio>

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

    /* Step 4: divide by area(nz, n) → m/s. */
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
