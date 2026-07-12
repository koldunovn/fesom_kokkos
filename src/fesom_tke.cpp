/*
 * CVMix classical-TKE vertical mixing — state, allocation, driver.
 * Faithful port of the C oracle's fesom_tke.c (fesom2_port_zstar @ df8b9a8).
 * See fesom_tke.h for the state map and fesom_cvmix_tke.hpp for the column core.
 *
 * M6.1 staging: Task 1.1 = state + alloc + dispatch entry (driver aborts);
 *               Task 1.2 = the column core; Task 1.3 = the driver kernel + halos.
 */
#include "fesom_tke.h"
#include "fesom_cvmix_tke.hpp"
#include "fesom_mesh.h"
#include "fesom_aux.h"
#include "fesom_dyn.h"
#include "fesom_forcing.h"
#include "fesom_constants.h"
#include "fesom_partit.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>

/*===========================================================================
 * Allocation — mirror of init_cvmix_tke's allocate block (gen:151-254) with the
 * reference &param_tke values, plus the gate-only guard. Every slab zero-init'd
 * (the Fortran inits each to 0; Field::alloc zero-inits both spaces like calloc).
 *===========================================================================*/
void fesom_tke_alloc(fesom_tke *t, const struct fesom_mesh *mesh, int diag_on)
{
    /* Field members (DualView) make a raw memset UB (L13); value-initialise instead (D13). */
    *t = fesom_tke{};

    const int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;   /* gen:153 node_size */
    const int nl = mesh->nl;
    t->n_nod   = N;
    t->nl      = nl;
    t->diag_on = diag_on;

    /* The column core's per-thread scratch is a fixed-size stack frame (device lambdas
     * cannot hold VLAs), so nl must fit. C: fesom_tke.c:194-197. */
    if (nl + 1 > TKE_NL_MAX) {
        std::fprintf(stderr, "fesom_tke_alloc: nl=%d exceeds TKE_NL_MAX=%d\n",
                     nl, (int)TKE_NL_MAX);
        std::abort();
    }

    const size_t n_nl = (size_t)N * (size_t)nl;

    /* Field owns the storage; the raw ptr is a non-owning alias = field.h() set once (D12).
     * .alloc takes an ELEMENT count (== the C's calloc count) and zero-inits → bit-identical. */
    t->tke_fld.alloc   ("tke.tke",    n_nl); t->tke    = t->tke_fld.h();
    t->tke_Av_fld.alloc("tke.tke_Av", n_nl); t->tke_Av = t->tke_Av_fld.h();
    t->tke_Kv_fld.alloc("tke.tke_Kv", n_nl); t->tke_Kv = t->tke_Kv_fld.h();

    t->forc_normstress_fld.alloc("tke.forc_normstress", (size_t)N);
    t->forc_botfrict_fld.alloc  ("tke.forc_botfrict",   (size_t)N);
    t->forc_rhosurf_fld.alloc   ("tke.forc_rhosurf",    (size_t)N);
    t->forc_normstress = t->forc_normstress_fld.h();
    t->forc_botfrict   = t->forc_botfrict_fld.h();
    t->forc_rhosurf    = t->forc_rhosurf_fld.h();

    /* the C's s_zero_col (fesom_tke.c:48) — zeros on BOTH spaces from Field::alloc, never
     * written, so no modify/sync rail is needed. iw_diss reads it unconditionally. */
    t->zero_col_fld.alloc("tke.zero_col", (size_t)TKE_NL_MAX + 1);

    if (diag_on) {
        struct { fesom::Field *f; real_t **raw; const char *label; } slabs[13] = {
            { &t->Tbpr_fld,   &t->Tbpr,   "tke.Tbpr"   },
            { &t->Tspr_fld,   &t->Tspr,   "tke.Tspr"   },
            { &t->Tdif_fld,   &t->Tdif,   "tke.Tdif"   },
            { &t->Tdis_fld,   &t->Tdis,   "tke.Tdis"   },
            { &t->Twin_fld,   &t->Twin,   "tke.Twin"   },
            { &t->Tiwf_fld,   &t->Tiwf,   "tke.Tiwf"   },
            { &t->Tbck_fld,   &t->Tbck,   "tke.Tbck"   },
            { &t->Ttot_fld,   &t->Ttot,   "tke.Ttot"   },
            { &t->Lmix_fld,   &t->Lmix,   "tke.Lmix"   },
            { &t->Pr_fld,     &t->Pr,     "tke.Pr"     },
            { &t->dummy1_fld, &t->dummy1, "tke.dummy1" },
            { &t->dummy2_fld, &t->dummy2, "tke.dummy2" },
            { &t->dummy3_fld, &t->dummy3, "tke.dummy3" },
        };
        for (auto &s : slabs) { s.f->alloc(s.label, n_nl); *s.raw = s.f->h(); }
    }

    /* Gate-only guard, mirroring the C's runtime abort (fesom_tke.c:246-253). The constants
     * are constexpr, so fesom_cvmix_tke.hpp already static_asserts these — this keeps the
     * loud runtime failure the C has, in case the params ever become runtime-loaded. */
    const fesom_cvmix_tke_params &p = FESOM_TKE_PARAMS;
    if (!p.only_tke || p.l_lc || p.use_ubound_dirichlet ||
        p.use_lbound_dirichlet || p.tke_mxl_choice != 2) {
        std::fprintf(stderr, "fesom_tke_alloc: configuration outside the ported path "
                             "(IDEMIX/Langmuir/Dirichlet/mxl_choice!=2 are gate-only, "
                             "not ported)\n");
        std::abort();
    }
}

void fesom_tke_free(fesom_tke *t)
{
    /* *t = fesom_tke{} releases every Field (assigning an empty DualView drops the refcount
     * → frees) and zeros the PODs (D13/L13). The raw ptrs are non-owning aliases — no free. */
    *t = fesom_tke{};
}

/*===========================================================================
 * diag_on / dump-rail env knobs (C: fesom_main.c:366-370, fesom_tke.c dump harness).
 *===========================================================================*/
const char *fesom_tke_dump_dir(void)
{
    static int         loaded = 0;
    static const char *dir    = nullptr;
    if (!loaded) {
        const char *e = std::getenv("FESOM_TKE_DUMP_DIR");
        dir    = (e && e[0]) ? e : nullptr;
        loaded = 1;
    }
    return dir;
}

/*===========================================================================================
 * calc_cvmix_tke — the per-node column loop (C: fesom_tke.c:332-450, gen:279-507).
 *
 * One thread per OWNED node (gen:296,311 node_size = myDim_nod2D — do NOT extend to the
 * halo; the halo values arrive via the exchange below). Each thread owns its column, so
 * there is no cross-entity write: the Serial range is sequential == the C loop => bit-
 * identical, and CUDA is race-free (the D19 entity-outer/level-inner shape).
 *
 * Templated on WITH_DIAG so the core's 13 budget slabs are dead-code-eliminated when the
 * diag knob is off (see fesom_cvmix_tke.hpp).
 *===========================================================================================*/
template <bool WITH_DIAG>
static void tke_column_loop(fesom_tke                  *t,
                            const struct fesom_aux     *aux,
                            const struct fesom_forcing *forcing,
                            const struct fesom_dyn     *dyn,
                            const struct fesom_mesh    *mesh)
{
    const int N_own = mesh->myDim_nod2D;                   /* gen:296 node_size */
    const int nl    = mesh->nl;

    /* Device-lambda captures must be values, not host globals. */
    const real_t dt    = (real_t)FESOM_PHASE1_DT;          /* gen:444 */
    const real_t rho0  = (real_t)FESOM_DENSITY_0;          /* gen:445 */
    const real_t grav  = (real_t)FESOM_G;                  /* gen:446 */

    auto Z       = mesh->Z_fld.d();
    auto hnode   = mesh->hnode_fld.d();
    auto ulev_n  = mesh->ulevels_nod2D_fld.d();
    auto nlev_n  = mesh->nlevels_nod2D_fld.d();
    auto uvnode  = dyn->uvnode_fld.d();
    auto bvfreq  = aux->bvfreq_fld.d();
    auto stress  = forcing->stress_node_surf_fld.d();
    auto tke     = t->tke_fld.d();
    auto tke_Av  = t->tke_Av_fld.d();
    auto tke_Kv  = t->tke_Kv_fld.d();
    auto normstr = t->forc_normstress_fld.d();
    auto botfr   = t->forc_botfrict_fld.d();
    auto rhosurf = t->forc_rhosurf_fld.d();
    auto zerocol = t->zero_col_fld.d();

    auto Tbpr = t->Tbpr_fld.d(); auto Tspr = t->Tspr_fld.d(); auto Tdif = t->Tdif_fld.d();
    auto Tdis = t->Tdis_fld.d(); auto Twin = t->Twin_fld.d(); auto Tiwf = t->Tiwf_fld.d();
    auto Tbck = t->Tbck_fld.d(); auto Ttot = t->Ttot_fld.d(); auto Lmix = t->Lmix_fld.d();
    auto Pr   = t->Pr_fld.d();   auto dm1  = t->dummy1_fld.d();
    auto dm2  = t->dummy2_fld.d(); auto dm3 = t->dummy3_fld.d();

    Kokkos::parallel_for("fesom_tke_columns", Kokkos::RangePolicy<>(0, N_own),
        KOKKOS_LAMBDA(const int node) {
            /* gen:314-315 — nln = nlevels-1 (last wet layer), nun = ulevels. 0-based:
             * uln0 = first wet interface, nln0 = last wet layer. */
            const int nln0 = nlev_n(node) - 2;
            const int uln0 = ulev_n(node) - 1;
            const int nlev = nln0 - uln0 + 1;              /* layer count at the call */

            real_t dz_trr[TKE_NL_MAX], bvfreq2[TKE_NL_MAX], vshear2[TKE_NL_MAX];
            real_t tke_old[TKE_NL_MAX];                    /* gen:287 column copy */

            /* gen:330 — surface momentum forcing from the NODAL wind stress (the
             * element-interpolation variant is commented out in the Fortran). */
            {
                real_t sx = stress(2 * node + 0);
                real_t sy = stress(2 * node + 1);
                normstr(node) = Kokkos::sqrt(sx * sx + sy * sy) / rho0;
            }

            /* gen:335-340 — vertical velocity shear² at interfaces nun+1..nln.
             * ⚠️ The C reads mesh->Z_3d_n here. Under linfs Z_3d_n[nz][n] == Z[nz] (the C
             * says so at fesom_eos.c:118; only zstar's fesom_ale.c:239 makes it live), and
             * this port already uses that identity throughout (fesom_eos.cpp:184,
             * fesom_pp.cpp:151). **M6.3 Task 3.6 must re-point these two Z reads to the
             * live Z_3d_n** — until then TKE is correct for linfs only. */
            for (int k = 0; k < nl; ++k) vshear2[k] = 0.0;
            for (int nz = uln0 + 1; nz <= nln0; ++nz) {
                real_t du = uvnode(FESOM_ELEMVEC(node, nz - 1, nl) + 0)
                          - uvnode(FESOM_ELEMVEC(node, nz,     nl) + 0);
                real_t dv = uvnode(FESOM_ELEMVEC(node, nz - 1, nl) + 1)
                          - uvnode(FESOM_ELEMVEC(node, nz,     nl) + 1);
                real_t dz = Z(nz - 1) - Z(nz);             /* == Z_3d_n(nz-1) - Z_3d_n(nz) */
                vshear2[nz] = (du * du + dv * dv) / (dz * dz);
            }

            /* gen:352-354 — N² (bvfreq already holds the SQUARED frequency, horizontally
             * smoothed upstream by smooth_nod3D). */
            for (int k = 0; k < nl; ++k) bvfreq2[k] = 0.0;
            for (int nz = uln0 + 1; nz <= nln0; ++nz)
                bvfreq2[nz] = bvfreq(FESOM_NODE3D(node, nz, nl));

            /* gen:359-362 — dz_trr: distance between tracer points; the surface and bottom
             * entries are HALF the layer thickness. (Z again — see the Task 3.6 note above.) */
            for (int k = 0; k < nl; ++k) dz_trr[k] = 0.0;
            for (int nz = uln0 + 1; nz <= nln0; ++nz)
                dz_trr[nz] = Kokkos::fabs(Z(nz - 1) - Z(nz));
            dz_trr[uln0]     = hnode(FESOM_NODE3D(node, uln0, nl)) / 2.0;
            dz_trr[nln0 + 1] = hnode(FESOM_NODE3D(node, nln0, nl)) / 2.0;

            /* gen:367-429 — Langmuir block: gate-only (tke_dolangmuir=F). */

            /* gen:433-435 — per-column copy of tke BEFORE the call (the ONLY persistent
             * input; tke_Av_old/tke_Kv_old fed verified-dead args and are NOT ported). */
            for (int k = 0; k < nl; ++k)
                tke_old[k] = tke(FESOM_NODE3D(node, k, nl));

            /* gen:437-482 — the cvmix call with nun:nln(+1) slices. Outputs write STRAIGHT
             * into the global slabs (tke_new/KappaM_out/KappaH_out are slices of
             * tke/tke_Av/tke_Kv), exactly as the C does. */
            const size_t off = FESOM_NODE3D(node, uln0, nl);

            /* ⚠️ A plain `if`, NOT `if constexpr`: nvcc rejects an extended
             * __host__ __device__ lambda that FIRST-captures a variable inside a
             * constexpr-if context, and these 13 views appear nowhere else in the lambda.
             * WITH_DIAG is a compile-time constant, so the branch still folds away entirely
             * (and with it the 13 dead .data() reads) — same codegen, nvcc-legal capture. */
            fesom_cvmix_tke_diag  dcols;
            fesom_cvmix_tke_diag *dptr = nullptr;
            if (WITH_DIAG) {
                dcols.Tbpr = Tbpr.data() + off;  dcols.Tspr = Tspr.data() + off;
                dcols.Tdif = Tdif.data() + off;  dcols.Tdis = Tdis.data() + off;
                dcols.Twin = Twin.data() + off;  dcols.Tiwf = Tiwf.data() + off;
                dcols.Tbck = Tbck.data() + off;  dcols.Ttot = Ttot.data() + off;
                dcols.Lmix = Lmix.data() + off;  dcols.Pr   = Pr.data()   + off;
                dcols.int1 = dm1.data()  + off;
                dcols.int2 = dm2.data()  + off;
                dcols.int3 = dm3.data()  + off;
                dptr = &dcols;
            }

            fesom_cvmix_integrate_tke<WITH_DIAG>(
                nlev,
                dt, rho0, grav,
                hnode.data() + off,        /* dzw = hnode(nun:nln)                    */
                &dz_trr[uln0],             /* dzt(nun:nln+1)                          */
                &tke_old[uln0],            /* tke_old                                 */
                &vshear2[uln0],            /* Ssqr                                    */
                &bvfreq2[uln0],            /* Nsqr                                    */
                zerocol.data(),            /* alpha_c (gated)                         */
                zerocol.data(),            /* E_iw    (gated)                         */
                zerocol.data(),            /* iw_diss — REAL zeros, read UNCONDITIONALLY */
                zerocol.data(),            /* tke_plc (gated)                         */
                normstr(node),             /* forc_tke_surf                           */
                rhosurf(node),             /* forc_rho_surf (zeroed every call)       */
                botfr(node),               /* bottom_fric   (zeroed every call)       */
                tke.data()    + off,       /* tke_new                                 */
                tke_Av.data() + off,       /* KappaM_out                              */
                tke_Kv.data() + off,       /* KappaH_out                              */
                dptr);

            /* gen:484-487 — zero visc/diff at the below-bottom and surface interfaces of the
             * column. ORDER AS THE C WRITES IT (bottom first, then surface): if a degenerate
             * column ever had uln0 == nln0+1 the two would collide, and the C's order wins. */
            tke_Av(FESOM_NODE3D(node, nln0 + 1, nl)) = 0.0;
            tke_Kv(FESOM_NODE3D(node, nln0 + 1, nl)) = 0.0;
            tke_Av(FESOM_NODE3D(node, uln0,     nl)) = 0.0;
            tke_Kv(FESOM_NODE3D(node, uln0,     nl)) = 0.0;
        });
}

/*===========================================================================================
 * The driver (C: fesom_tke.c:270-534). Zero the 2D forcing arrays, run the column loop,
 * then wire the outputs into aux->Kv / aux->Av.
 *
 * NOT ported (C-side debug rails, deliberately): the FESOM_TKE_DUMP_DIR /
 * FESOM_TKE_REPLAY_DIR dump+replay machinery and the FESOM_TKE_HALO_PROBE gate. Bisection
 * against /work/.../tke/{cdump,cdump_v2,replay} is done with the C oracle binary; the
 * column MATH is covered independently by tests/tke_core_twin/.
 *===========================================================================================*/
void fesom_tke_mixing_kk(fesom_tke                  *t,
                         struct fesom_aux           *aux,
                         const struct fesom_forcing *forcing,
                         const struct fesom_dyn     *dyn,
                         const struct fesom_mesh    *mesh,
                         struct fesom_partit        *partit)
{
    const int N_full = mesh->myDim_nod2D  + mesh->eDim_nod2D;
    const int E_own  = mesh->myDim_elem2D;
    const int nl     = mesh->nl;

    /* gen:299-301 — zero the 2D forcing arrays every call, over the FULL extent. normstress
     * is then overwritten at the owned nodes by the column loop; botfrict and rhosurf stay
     * zero (the Fortran has no function for them yet, and passes them anyway). */
    Kokkos::deep_copy(t->forc_normstress_fld.d(), 0.0);
    Kokkos::deep_copy(t->forc_botfrict_fld.d(),   0.0);
    Kokkos::deep_copy(t->forc_rhosurf_fld.d(),    0.0);

    /* gen:304-308 — IDEMIX input load: gate-only (only_tke pinned true). */

    if (t->diag_on) tke_column_loop<true> (t, aux, forcing, dyn, mesh);
    else            tke_column_loop<false>(t, aux, forcing, dyn, mesh);

    t->tke_fld.modify_device();
    t->forc_normstress_fld.modify_device();
    t->forc_botfrict_fld.modify_device();
    t->forc_rhosurf_fld.modify_device();
    if (t->diag_on) {
        t->Tbpr_fld.modify_device(); t->Tspr_fld.modify_device(); t->Tdif_fld.modify_device();
        t->Tdis_fld.modify_device(); t->Twin_fld.modify_device(); t->Tiwf_fld.modify_device();
        t->Tbck_fld.modify_device(); t->Ttot_fld.modify_device(); t->Lmix_fld.modify_device();
        t->Pr_fld.modify_device();   t->dummy1_fld.modify_device();
        t->dummy2_fld.modify_device(); t->dummy3_fld.modify_device();
    }

    /*---------------------------------------------------------------------------------------
     * gen:491-505 — wire the outputs.
     *
     * The C does: exchange(tke_Kv) -> copy Kv -> exchange(tke_Av) -> zero Av -> elem mean.
     * We FUSE the two node exchanges into one fesom_halo_field2 (M5.23 L1, proven
     * bit-id-neutral). That is safe because nothing between the C's two exchanges touches
     * tke_Av: the only statement in between is the Kv copy. Both fields are NOD3D / nl / 1.
     * `tke` itself is NEVER exchanged — no reader needs its halo.
     *-------------------------------------------------------------------------------------*/
    fesom_halo_field2(t->tke_Kv_fld, t->tke_Av_fld, FESOM_HALO_NOD3D, nl, 1, partit);

    /* gen:494 — Kv = tke_Kv over the FULL extent (halo included; it is owner-fresh from the
     * exchange above). Device-to-device copy; deep_copy does not touch the Field's dirty
     * tag, so mark it explicitly. */
    Kokkos::deep_copy(aux->Kv_fld.d(), t->tke_Kv_fld.d());
    aux->Kv_fld.modify_device();

    /* gen:499 — Av = 0 over its FULL extent, then the node->elem mean on OWNED elements over
     * INTERIOR levels only. The halo Av is restored by the shared post-mo_convect element
     * exchange in fesom_step.cpp — exactly as for KPP/PP. */
    Kokkos::deep_copy(aux->Av_fld.d(), 0.0);

    auto Av      = aux->Av_fld.d();
    auto tke_Av  = t->tke_Av_fld.d();
    auto elnod   = mesh->elem_nodes_fld.d();
    auto ulev_e  = mesh->ulevels_fld.d();
    auto nlev_e  = mesh->nlevels_fld.d();

    /* Per-element GATHER (each element writes only its own rows, reads its 3 vertices) —
     * race-free, no atomics: bit-identical on Serial AND CUDA (the L41 SpMV shape, not the
     * D22 scatter shape). */
    Kokkos::parallel_for("fesom_tke_av_node2elem", Kokkos::RangePolicy<>(0, E_own),
        KOKKOS_LAMBDA(const int elem) {                    /* gen:500 */
            const int n0 = elnod(3 * elem + 0);
            const int n1 = elnod(3 * elem + 1);
            const int n2 = elnod(3 * elem + 2);
            const int ule0 = ulev_e(elem) - 1;
            const int nle0 = nlev_e(elem) - 1;
            for (int nz = ule0 + 1; nz <= nle0 - 1; ++nz) {          /* gen:502 */
                Av(FESOM_ELEM3D(elem, nz, nl)) =
                    (tke_Av(FESOM_NODE3D(n0, nz, nl))
                   + tke_Av(FESOM_NODE3D(n1, nz, nl))
                   + tke_Av(FESOM_NODE3D(n2, nz, nl))) / 3.0;
            }
        });
    aux->Av_fld.modify_device();

    (void)N_full;
}
