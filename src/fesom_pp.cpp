/*
 * PP mixing + convective adjustment. Literal ports — see header for
 * Fortran source line references.
 */
#include "fesom_pp.h"
#include "fesom_aux.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_mesh.h"

#include <Kokkos_Core.hpp>   // M2.2: device kernels (parallel_for) for PP mixing + convection
#include <math.h>
#include <stdlib.h>
#include <vector>            // M2.2: FESOM_KK_VERIFY=pp snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>
#include <cstdio>

/*===========================================================================
 * compute_vel_nodes (oce_dyn.F90:177-226)
 *
 * For each node n and each layer nz:
 *   tx, ty, tvol = 0
 *   for each surrounding element el (via nod_in_elem2D):
 *       if nz outside (ulevels(el) .. nlevels(el)-1): skip
 *       tvol += elem_area(el)
 *       tx   += UV(1, nz, el) * elem_area(el)
 *       ty   += UV(2, nz, el) * elem_area(el)
 *   UVnode(1, nz, n) = tx / tvol
 *   UVnode(2, nz, n) = ty / tvol
 *===========================================================================*/

void fesom_compute_vel_nodes(const struct fesom_mesh *mesh,
                             struct fesom_dyn        *dyn)
{
    const int N    = mesh->myDim_nod2D;       /* interior only — halo gets exchange */
    const int nl   = mesh->nl;

    for (int n = 0; n < N; ++n) {
        int uln = mesh->ulevels_nod2D[n] - 1;     /* 1-based → 0-based */
        int nln = mesh->nlevels_nod2D[n] - 1;     /* exclusive bound  */
        for (int nz = uln; nz < nln; ++nz) {
            real_t tvol = 0.0, tx = 0.0, ty = 0.0;
            int o0 = mesh->nod_in_elem2D_offsets[n];
            int o1 = mesh->nod_in_elem2D_offsets[n + 1];
            for (int k = o0; k < o1; ++k) {
                int e   = mesh->nod_in_elem2D[k];
                int ule = mesh->ulevels[e] - 1;
                int nle = mesh->nlevels[e] - 1;
                if (nz < ule || nz >= nle) continue;
                real_t a = mesh->elem_area[e];
                tvol += a;
                tx   += dyn->uv[FESOM_ELEMVEC(e, nz, nl) + 0] * a;
                ty   += dyn->uv[FESOM_ELEMVEC(e, nz, nl) + 1] * a;
            }
            if (tvol > 0.0) {
                dyn->uvnode[FESOM_ELEMVEC(n, nz, nl) + 0] = tx / tvol;
                dyn->uvnode[FESOM_ELEMVEC(n, nz, nl) + 1] = ty / tvol;
            } else {
                dyn->uvnode[FESOM_ELEMVEC(n, nz, nl) + 0] = 0.0;
                dyn->uvnode[FESOM_ELEMVEC(n, nz, nl) + 1] = 0.0;
            }
        }
    }
}

/*--- compute_vel_nodes — DEVICE kernel (M2.2) --------------------------------
 * Kokkos parallel_for over owned nodes (the M2.1 EOS template, D19). The whole
 * per-node body — the level loop AND the inner gather over the surrounding
 * elements (the nod_in_elem2D CSR) — runs INSIDE the lambda, accumulating into
 * the lambda-local tx/ty/tvol and writing only this node's own uvnode slots. So
 * there is NO cross-node write race, and the per-node accumulation order is the
 * verbatim C-twin order (sequential over the CSR range) → Serial == the C twin
 * AND OpenMP is bit-identical (each node's gather is a private reduction, not a
 * cross-thread one). Every bound/constant is a verbatim copy of fesom_compute_vel_nodes.
 *
 * SYNC contract (SYNC_MAP §2 row-3): INPUT dyn->uv is host-written (update_vel +
 * bolus, previous step) so the driver's rail does modify_host()+sync_device() on it;
 * the mesh CSR / elem_area / ulevels / nlevels / ulevels_nod2D / nlevels_nod2D are
 * set-once and already device-current (mesh_sync_geometry_device). OUTPUT uvnode is
 * marked modify_device() here; the driver sync_host()s it before the halo exchange
 * (and the host KPP/PP readers).
 */
void fesom_compute_vel_nodes_kk(const struct fesom_mesh *mesh,
                                struct fesom_dyn        *dyn)
{
    const int N  = mesh->myDim_nod2D;       /* interior only — halo gets exchange */
    const int nl = mesh->nl;

    auto uv      = dyn->uv_fld.d();
    auto uvnode  = dyn->uvnode_fld.d();
    auto ulev_n  = mesh->ulevels_nod2D_fld.d();
    auto nlev_n  = mesh->nlevels_nod2D_fld.d();
    auto ulev_e  = mesh->ulevels_fld.d();
    auto nlev_e  = mesh->nlevels_fld.d();
    auto area    = mesh->elem_area_fld.d();
    auto nie     = mesh->nod_in_elem2D_fld.d();
    auto nie_off = mesh->nod_in_elem2D_offsets_fld.d();

    Kokkos::parallel_for("fesom_compute_vel_nodes", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            int uln = ulev_n(n) - 1;            /* 1-based → 0-based */
            int nln = nlev_n(n) - 1;            /* exclusive bound  */
            for (int nz = uln; nz < nln; ++nz) {
                real_t tvol = 0.0, tx = 0.0, ty = 0.0;
                int o0 = nie_off(n);
                int o1 = nie_off(n + 1);
                for (int k = o0; k < o1; ++k) {
                    int e   = nie(k);
                    int ule = ulev_e(e) - 1;
                    int nle = nlev_e(e) - 1;
                    if (nz < ule || nz >= nle) continue;
                    real_t a = area(e);
                    tvol += a;
                    tx   += uv(FESOM_ELEMVEC(e, nz, nl) + 0) * a;
                    ty   += uv(FESOM_ELEMVEC(e, nz, nl) + 1) * a;
                }
                if (tvol > 0.0) {
                    uvnode(FESOM_ELEMVEC(n, nz, nl) + 0) = tx / tvol;
                    uvnode(FESOM_ELEMVEC(n, nz, nl) + 1) = ty / tvol;
                } else {
                    uvnode(FESOM_ELEMVEC(n, nz, nl) + 0) = 0.0;
                    uvnode(FESOM_ELEMVEC(n, nz, nl) + 1) = 0.0;
                }
            }
        });

    dyn->uvnode_fld.modify_device();   /* driver sync_host()s before the halo + readers */
}

/*===========================================================================
 * oce_mixing_pp (oce_ale_mixing_pp.F90:2-93)
 *
 * Three loops:
 *   1. Per node, per interior interface nz ∈ [nzmin+1, nzmax-1):
 *        dz_inv = 1 / (Z(nz-1) - Z(nz))             ; > 0 because Z negative-down
 *        shear  = (Δu)² + (Δv)²                     ; node values, not element
 *        shear *= dz_inv²
 *        Kv(nz, node) = shear / (shear + 5*max(N², 0) + 1e-14)
 *      ← THIS IS THE "factor" — overwrites Kv with the dimensionless ratio.
 *
 *   2. Per element, per interior interface nz:
 *        Av(nz, elem) = mix_coeff_PP * Σ Kv(nz, elnodes)² / 3 + A_ver
 *      ← Reads `Kv` BEFORE it has been raised to power 3 and offset by K_ver.
 *
 *   3. Per node again (Kv0_const branch):
 *        Kv(nz, node) = mix_coeff_PP * Kv(nz, node)³ + K_ver
 *      ← Order of (2) and (3) matters: Av is built from the (factor)² of Kv,
 *        before Kv itself is raised to (factor)³ and offset.
 *
 * For Phase 1 with no partial cells, Z_3d_n[nz][n] = Z[nz] (1D mid-level depth).
 *===========================================================================*/

void fesom_pp_mixing(const struct fesom_mesh *mesh,
                     const struct fesom_dyn  *dyn,
                     struct fesom_aux        *aux)
{
    /* Fortran oce_ale_mixing_pp.F90:41,72 iterates `myDim+eDim_nod2D` for the
     * Kv loops (compute Kv at all local nodes, including halo). The element
     * Av loop reads Kv at all 3 vertices — if halo Kv is stale (only myDim
     * computed locally), Av on boundary elements is wrong → wrong vertical
     * mixing → tracer divergence at partition boundaries → S explosion. */
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int E  = mesh->myDim_elem2D;
    const int nl = mesh->nl;
    const real_t mix_coeff = (real_t)FESOM_PHASE1_MIX_COEFF_PP;
    const real_t K_bg      = (real_t)FESOM_PHASE1_K_VER;
    const real_t A_bg      = (real_t)FESOM_PHASE1_A_VER;

    /* Loop 1: factor = shear² / (shear² + 5N² + ε) → write into Kv */
    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t dz = mesh->Z[nz - 1] - mesh->Z[nz];      /* > 0 */
            real_t dz_inv = 1.0 / dz;
            real_t du = dyn->uvnode[FESOM_ELEMVEC(n, nz - 1, nl) + 0]
                      - dyn->uvnode[FESOM_ELEMVEC(n, nz,     nl) + 0];
            real_t dv = dyn->uvnode[FESOM_ELEMVEC(n, nz - 1, nl) + 1]
                      - dyn->uvnode[FESOM_ELEMVEC(n, nz,     nl) + 1];
            real_t shear = (du * du + dv * dv) * dz_inv * dz_inv;
            real_t Nsq   = aux->bvfreq[FESOM_NODE3D(n, nz, nl)];
            real_t Nsq_pos = Nsq > 0.0 ? Nsq : 0.0;
            aux->Kv[FESOM_NODE3D(n, nz, nl)]
                = shear / (shear + 5.0 * Nsq_pos + 1.0e-14);
        }
    }

    /* Loop 2: Av at elements from (Kv-as-factor)² averaged over 3 vertices */
    for (int e = 0; e < E; ++e) {
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];
        int nzmin = mesh->ulevels[e] - 1;
        int nzmax = mesh->nlevels[e] - 1;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t k0 = aux->Kv[FESOM_NODE3D(n0, nz, nl)];
            real_t k1 = aux->Kv[FESOM_NODE3D(n1, nz, nl)];
            real_t k2 = aux->Kv[FESOM_NODE3D(n2, nz, nl)];
            aux->Av[FESOM_ELEM3D(e, nz, nl)]
                = mix_coeff * (k0*k0 + k1*k1 + k2*k2) / 3.0 + A_bg;
        }
    }

    /* Loop 3: Kv = mix_coeff * factor³ + K_ver  (Kv0_const branch) */
    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t f = aux->Kv[FESOM_NODE3D(n, nz, nl)];
            aux->Kv[FESOM_NODE3D(n, nz, nl)] = mix_coeff * f * f * f + K_bg;
        }
    }
}

/*--- oce_mixing_pp — DEVICE kernel (M2.2) ------------------------------------
 * The three pp_mixing loops as three SEPARATE Kokkos parallel_for launches — and
 * the launch boundary is load-bearing, not cosmetic: each parallel_for is a full
 * barrier, so it preserves the ⚠️ loop-2-before-loop-3 dependency (Loop 2 reads Kv
 * while it still holds the dimensionless "factor" from Loop 1; Loop 3 then cubes
 * Kv in place — if Loop 3 ran before/interleaved with Loop 2, Av would be built
 * from factor³+K_ver instead of factor², a silent physics change). Do NOT fuse the
 * two node loops. Every bound, range (Loop 1 & 3 over myDim+eDim incl. halo so
 * boundary elements in Loop 2 read fresh vertex Kv; Loop 2 over owned elems), and
 * constant is a verbatim copy of fesom_pp_mixing.
 *
 * Race-free per D19: Loop 1 each node writes its own Kv; Loop 2 each element writes
 * its own Av (Kv read-only); Loop 3 each node read-modify-writes its own Kv. No
 * cross-entity write → Serial == the C twin AND OpenMP bit-identical (pure maps).
 *
 * SYNC contract (SYNC_MAP §2 row-3, PP branch): INPUTS dyn->uvnode (host-halo'd
 * just above) and aux->bvfreq (host-written by smooth_nod3D in substep 1) are pushed
 * device-current by the driver rail (modify_host()+sync_device()); mesh Z/levels/
 * elem_nodes are set-once device-current. OUTPUTS Kv (node) + Av (elem) marked
 * modify_device(); the driver sync_host()s them before the Kv/Av halos.
 */
void fesom_pp_mixing_kk(const struct fesom_mesh *mesh,
                        const struct fesom_dyn  *dyn,
                        struct fesom_aux        *aux)
{
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int E  = mesh->myDim_elem2D;
    const int nl = mesh->nl;
    const real_t mix_coeff = (real_t)FESOM_PHASE1_MIX_COEFF_PP;
    const real_t K_bg      = (real_t)FESOM_PHASE1_K_VER;
    const real_t A_bg      = (real_t)FESOM_PHASE1_A_VER;

    auto Z      = mesh->Z_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto nlev_n = mesh->nlevels_nod2D_fld.d();
    auto ulev_e = mesh->ulevels_fld.d();
    auto nlev_e = mesh->nlevels_fld.d();
    auto elnod  = mesh->elem_nodes_fld.d();
    auto uvnode = dyn->uvnode_fld.d();
    auto bvfreq = aux->bvfreq_fld.d();
    auto Kv     = aux->Kv_fld.d();
    auto Av     = aux->Av_fld.d();

    /* Loop 1: factor = shear² / (shear² + 5N² + ε) → write into Kv */
    Kokkos::parallel_for("fesom_pp_mixing_kv_factor", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t dz = Z(nz - 1) - Z(nz);              /* > 0 */
                real_t dz_inv = 1.0 / dz;
                real_t du = uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 0)
                          - uvnode(FESOM_ELEMVEC(n, nz,     nl) + 0);
                real_t dv = uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 1)
                          - uvnode(FESOM_ELEMVEC(n, nz,     nl) + 1);
                real_t shear = (du * du + dv * dv) * dz_inv * dz_inv;
                real_t Nsq   = bvfreq(FESOM_NODE3D(n, nz, nl));
                real_t Nsq_pos = Nsq > 0.0 ? Nsq : 0.0;
                Kv(FESOM_NODE3D(n, nz, nl))
                    = shear / (shear + 5.0 * Nsq_pos + 1.0e-14);
            }
        });

    /* Loop 2: Av at elements from (Kv-as-factor)² averaged over 3 vertices.
       Reads Kv BEFORE Loop 3 raises it to the cube — the parallel_for barrier
       above guarantees that ordering. */
    Kokkos::parallel_for("fesom_pp_mixing_av", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int e) {
            int n0 = elnod(3*e + 0);
            int n1 = elnod(3*e + 1);
            int n2 = elnod(3*e + 2);
            int nzmin = ulev_e(e) - 1;
            int nzmax = nlev_e(e) - 1;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t k0 = Kv(FESOM_NODE3D(n0, nz, nl));
                real_t k1 = Kv(FESOM_NODE3D(n1, nz, nl));
                real_t k2 = Kv(FESOM_NODE3D(n2, nz, nl));
                Av(FESOM_ELEM3D(e, nz, nl))
                    = mix_coeff * (k0*k0 + k1*k1 + k2*k2) / 3.0 + A_bg;
            }
        });

    /* Loop 3: Kv = mix_coeff * factor³ + K_ver  (Kv0_const branch) */
    Kokkos::parallel_for("fesom_pp_mixing_kv_final", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t f = Kv(FESOM_NODE3D(n, nz, nl));
                Kv(FESOM_NODE3D(n, nz, nl)) = mix_coeff * f * f * f + K_bg;
            }
        });

    aux->Kv_fld.modify_device();
    aux->Av_fld.modify_device();
}

/*===========================================================================
 * mo_convect — use_instabmix branch only (oce_mo_conv.F90:79-94 for Kv,
 * 99-120 for Av). Phase 1 skips momix and windmix entirely.
 *===========================================================================*/

void fesom_mo_convect(const struct fesom_mesh *mesh,
                      struct fesom_aux        *aux)
{
    if (!FESOM_PHASE1_USE_INSTABMIX) return;

    /* Fortran oce_mo_conv.F90:36,79 iterates `myDim+eDim_nod2D` for the Kv
     * loop. The Av loop reads Kv at element vertices, so halo Kv must be
     * locally fresh — same reasoning as pp_mixing. */
    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int E  = mesh->myDim_elem2D;
    const int nl = mesh->nl;
    const real_t imix_kv = (real_t)FESOM_PHASE1_INSTABMIX_KV;

    /* Kv: where bvfreq < 0, set Kv = max(Kv, instabmix_kv).
       Mirror of oce_mo_conv.F90:85. */
    for (int n = 0; n < N; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            size_t k = FESOM_NODE3D(n, nz, nl);
            if (aux->bvfreq[k] < 0.0) {
                if (aux->Kv[k] < imix_kv) aux->Kv[k] = imix_kv;
            }
        }
    }

    /* Av: where ANY of the three vertex bvfreq < 0, set Av = max(Av, imix_kv).
       Mirror of oce_mo_conv.F90:108. */
    for (int e = 0; e < E; ++e) {
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];
        int nzmin = mesh->ulevels[e] - 1;
        int nzmax = mesh->nlevels[e] - 1;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t b0 = aux->bvfreq[FESOM_NODE3D(n0, nz, nl)];
            real_t b1 = aux->bvfreq[FESOM_NODE3D(n1, nz, nl)];
            real_t b2 = aux->bvfreq[FESOM_NODE3D(n2, nz, nl)];
            if (b0 < 0.0 || b1 < 0.0 || b2 < 0.0) {
                size_t ke = FESOM_ELEM3D(e, nz, nl);
                if (aux->Av[ke] < imix_kv) aux->Av[ke] = imix_kv;
            }
        }
    }
}

/*--- mo_convect — DEVICE kernel (M2.2) ---------------------------------------
 * Two parallel_for launches (Kv over myDim+eDim nodes, Av over owned elements),
 * the verbatim convective-adjustment maxes of fesom_mo_convect. Each entity
 * read-modify-writes only its own slot (a conditional `max` raise) → race-free,
 * Serial == the C twin AND OpenMP bit-identical. The two loops are independent
 * (Kv↔node, Av↔elem; the Av loop reads only bvfreq, never Kv) so their relative
 * order is irrelevant — separate launches are for clarity, not a dependency.
 *
 * SYNC contract (SYNC_MAP §2 row-3): mo_convect is the FIRST device reader of
 * aux->bvfreq AFTER the host smooth_nod3D in substep 1 (which wrote bvfreq via the
 * raw alias — invisible to the DualView, L14), so the driver rail MUST
 * modify_host()+sync_device() bvfreq, not just sync_device(). Kv/Av are host-
 * authoritative on entry too (KPP wrote them on host on the default path; or the PP
 * branch sync_host()'d + halo'd them just above) → same modify_host()+sync_device().
 * OUTPUTS Kv/Av marked modify_device(); the driver sync_host()s them before the halos.
 */
void fesom_mo_convect_kk(const struct fesom_mesh *mesh,
                         struct fesom_aux        *aux)
{
    if (!FESOM_PHASE1_USE_INSTABMIX) return;

    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int E  = mesh->myDim_elem2D;
    const int nl = mesh->nl;
    const real_t imix_kv = (real_t)FESOM_PHASE1_INSTABMIX_KV;

    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto nlev_n = mesh->nlevels_nod2D_fld.d();
    auto ulev_e = mesh->ulevels_fld.d();
    auto nlev_e = mesh->nlevels_fld.d();
    auto elnod  = mesh->elem_nodes_fld.d();
    auto bvfreq = aux->bvfreq_fld.d();
    auto Kv     = aux->Kv_fld.d();
    auto Av     = aux->Av_fld.d();

    /* Kv: where bvfreq < 0, set Kv = max(Kv, instabmix_kv). (oce_mo_conv.F90:85) */
    Kokkos::parallel_for("fesom_mo_convect_kv", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                size_t k = FESOM_NODE3D(n, nz, nl);
                if (bvfreq(k) < 0.0) {
                    if (Kv(k) < imix_kv) Kv(k) = imix_kv;
                }
            }
        });

    /* Av: where ANY of the three vertex bvfreq < 0, set Av = max(Av, imix_kv).
       (oce_mo_conv.F90:108) */
    Kokkos::parallel_for("fesom_mo_convect_av", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int e) {
            int n0 = elnod(3*e + 0);
            int n1 = elnod(3*e + 1);
            int n2 = elnod(3*e + 2);
            int nzmin = ulev_e(e) - 1;
            int nzmax = nlev_e(e) - 1;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t b0 = bvfreq(FESOM_NODE3D(n0, nz, nl));
                real_t b1 = bvfreq(FESOM_NODE3D(n1, nz, nl));
                real_t b2 = bvfreq(FESOM_NODE3D(n2, nz, nl));
                if (b0 < 0.0 || b1 < 0.0 || b2 < 0.0) {
                    size_t ke = FESOM_ELEM3D(e, nz, nl);
                    if (Av(ke) < imix_kv) Av(ke) = imix_kv;
                }
            }
        });

    aux->Kv_fld.modify_device();
    aux->Av_fld.modify_device();
}

/*============================================================================
 * FESOM_KK_VERIFY=pp — in-binary per-kernel gates (M2.2)
 *
 * The in-binary analogue of scripts/exp1_compare_bidiff.py for the three PP
 * substep-3 kernels (the M2.1 fesom_eos_verify shape, D19): run the HOST C twin
 * and the DEVICE Kokkos kernel on the SAME live state, report max|Δ| over the
 * kernel's owned write region, assert max|Δ|==0 on the Serial backend (the
 * bit-identity oracle). All three are non-intrusive — each RESTORES the Kokkos
 * production result into aux/dyn before returning, so a whole run can carry the
 * verify on and still produce the KK output (== golden on Serial; the production
 * path proceeds on the device, preserving host==device coherence on CUDA). The
 * snapshot buffers are plain host vectors (never touched by a device kernel → not
 * "kernel scratch", no Field wrapping, cf. SYNC_MAP §9.5). Diagnostic only.
 *
 * NOTE the asymmetry vs EOS: compute_vel_nodes and pp_mixing fully OVERWRITE their
 * outputs from inputs the kernel does not modify (uv; uvnode+bvfreq), so the C twin
 * recomputes from intact state — pure EOS-style. mo_convect MODIFIES its inputs
 * Kv/Av IN PLACE (a conditional max), so its C-twin oracle needs the PRE-kernel
 * Kv/Av; the driver captures those and passes them in (Kv_in/Av_in).
 *==========================================================================*/

static double pp_maxdiff(const std::vector<real_t> &kk, const real_t *c, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)kk[i] - (double)c[i]);
        if (d > m) m = d;
    }
    return m;
}

static void pp_verify_report(const char *kernel, int step_n,
                             double dmax, const char *detail)
{
    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=pp] step %d backend=%s  %s max|Δ|: %s\n",
                step_n, backend.c_str(), kernel, detail);
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=pp] FAIL: %s Serial must be bit-identical to the "
                             "C twin (max|Δ|=%.3e)\n", kernel, dmax);
        std::abort();
    }
}

/* compute_vel_nodes: EOS-style (input dyn->uv is not modified by the kernel). aux
 * holds the KK uvnode (driver sync_host()'d it); diff the owned region [0,myDim). */
void fesom_compute_vel_nodes_verify(const struct fesom_mesh *mesh,
                                    struct fesom_dyn        *dyn,
                                    int step_n)
{
    const int nl    = mesh->nl;
    const int myDim = mesh->myDim_nod2D;
    const size_t nvec = (size_t)myDim * (size_t)nl * 2;     /* uvnode owned, 2-comp */

    std::vector<real_t> kk_uvnode(dyn->uvnode, dyn->uvnode + nvec);
    fesom_compute_vel_nodes(mesh, dyn);                     /* C twin overwrites via raw alias */
    double dmax = pp_maxdiff(kk_uvnode, dyn->uvnode, nvec);
    std::copy(kk_uvnode.begin(), kk_uvnode.end(), dyn->uvnode);   /* restore KK */

    char detail[64];
    std::snprintf(detail, sizeof detail, "uvnode=%.3e", dmax);
    pp_verify_report("compute_vel_nodes", step_n, dmax, detail);
}

/* pp_mixing: EOS-style (inputs uvnode/bvfreq are not modified; the kernel fully
 * overwrites Kv [0,myDim+eDim) and Av [0,myDim_elem) in range — Loop 1/2 are
 * assignments, so the result is independent of the prior Kv/Av). */
void fesom_pp_mixing_verify(const struct fesom_mesh *mesh,
                            const struct fesom_dyn  *dyn,
                            struct fesom_aux        *aux,
                            int step_n)
{
    const int nl = mesh->nl;
    const size_t nKv = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    const size_t nAv = (size_t)mesh->myDim_elem2D * (size_t)nl;

    std::vector<real_t> kk_Kv(aux->Kv, aux->Kv + nKv);
    std::vector<real_t> kk_Av(aux->Av, aux->Av + nAv);
    fesom_pp_mixing(mesh, dyn, aux);                        /* C twin overwrites via raw alias */
    double d_Kv = pp_maxdiff(kk_Kv, aux->Kv, nKv);
    double d_Av = pp_maxdiff(kk_Av, aux->Av, nAv);
    std::copy(kk_Kv.begin(), kk_Kv.end(), aux->Kv);         /* restore KK */
    std::copy(kk_Av.begin(), kk_Av.end(), aux->Av);

    char detail[96];
    std::snprintf(detail, sizeof detail, "Kv=%.3e Av=%.3e", d_Kv, d_Av);
    pp_verify_report("pp_mixing", step_n, std::max(d_Kv, d_Av), detail);
}

/* mo_convect: in-place modify → the C-twin oracle needs the PRE-kernel Kv/Av,
 * which the driver captured (Kv_in/Av_in) before fesom_mo_convect_kk overwrote
 * them on the device. Restore the input, run the C twin, diff, restore KK. */
void fesom_mo_convect_verify(const struct fesom_mesh *mesh,
                             struct fesom_aux        *aux,
                             int step_n,
                             const real_t *Kv_in,
                             const real_t *Av_in)
{
    const int nl = mesh->nl;
    const size_t nKv = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)nl;
    const size_t nAv = (size_t)mesh->myDim_elem2D * (size_t)nl;

    std::vector<real_t> kk_Kv(aux->Kv, aux->Kv + nKv);      /* KK output (synced to host) */
    std::vector<real_t> kk_Av(aux->Av, aux->Av + nAv);
    std::copy(Kv_in, Kv_in + nKv, aux->Kv);                 /* restore the pre-kernel input */
    std::copy(Av_in, Av_in + nAv, aux->Av);
    fesom_mo_convect(mesh, aux);                            /* C twin runs on the input */
    double d_Kv = pp_maxdiff(kk_Kv, aux->Kv, nKv);
    double d_Av = pp_maxdiff(kk_Av, aux->Av, nAv);
    std::copy(kk_Kv.begin(), kk_Kv.end(), aux->Kv);         /* restore KK */
    std::copy(kk_Av.begin(), kk_Av.end(), aux->Av);

    char detail[96];
    std::snprintf(detail, sizeof detail, "Kv=%.3e Av=%.3e", d_Kv, d_Av);
    pp_verify_report("mo_convect", step_n, std::max(d_Kv, d_Av), detail);
}
