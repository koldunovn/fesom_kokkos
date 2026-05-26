#include "fesom_eos.h"
#include "fesom_aux.h"
#include "fesom_constants.h"
#include "fesom_mesh.h"
#include "fesom_tracers.h"
#include "fesom_halo.h"

#include <Kokkos_Core.hpp>   // M2.1: first device kernels (parallel_for) + Kokkos:: math
#include <math.h>
#include <stdlib.h>
#include <vector>            // M2.1: FESOM_KK_VERIFY snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>
#include <cstdio>

/*--- JM-EOS components -----------------------------------------------------
 * Literal port of densityJM_components (oce_ale_pressure_bv.F90:2605-2669).
 * All constants reproduced verbatim — do NOT round, fold, or "simplify."
 */
void fesom_eos_jm_components(real_t t, real_t s,
                             real_t *bulk_0, real_t *bulk_pz, real_t *bulk_pz2,
                             real_t *rhopot)
{
    static const real_t a0    = 19092.56,    at   = 209.8925;
    static const real_t at2   = -3.041638,   at3  = -1.852732e-3;
    static const real_t at4   = -1.361629e-5;
    static const real_t as    = 104.4077,    ast  = -6.500517;
    static const real_t ast2  = 0.1553190,   ast3 = 2.326469e-4;
    static const real_t ass   = -5.587545,   asst = 0.7390729;
    static const real_t asst2 = -1.909078e-2;
    static const real_t ap    = -4.721788e-1, apt  = -1.028859e-2;
    static const real_t apt2  = 2.512549e-4,  apt3 = 5.939910e-7;
    static const real_t aps   = 1.571896e-2,  apst = 2.598241e-4;
    static const real_t apst2 = -7.267926e-6, apss = -2.042967e-3;
    static const real_t ap2   = 1.045941e-5,  ap2t = -5.782165e-10;
    static const real_t ap2t2 = 1.296821e-7;
    static const real_t ap2s  = -2.595994e-7, ap2st = -1.248266e-9;
    static const real_t ap2st2= -3.508914e-9;

    static const real_t b0  = 999.842594,   bt   = 6.793952e-2;
    static const real_t bt2 = -9.095290e-3, bt3  = 1.001685e-4;
    static const real_t bt4 = -1.120083e-6, bt5  = 6.536332e-9;
    static const real_t bs  = 0.824493,     bst  = -4.08990e-3;
    static const real_t bst2 = 7.64380e-5,  bst3 = -8.24670e-7;
    static const real_t bst4 = 5.38750e-9;
    static const real_t bss  = -5.72466e-3, bsst = 1.02270e-4;
    static const real_t bsst2 = -1.65460e-6, bss2 = 4.8314e-4;

    real_t s_sqrt = sqrt(s);

    *bulk_0 =  a0      + t*(at   + t*(at2  + t*(at3 + t*at4)))
             + s* (as  + t*(ast  + t*(ast2 + t*ast3))
                  + s_sqrt*(ass  + t*(asst + t*asst2)));

    *bulk_pz =  ap  + t*(apt  + t*(apt2 + t*apt3))
                    + s*(aps + t*(apst + t*apst2) + s_sqrt*apss);

    *bulk_pz2 = ap2 + t*(ap2t + t*ap2t2)
                   + s *(ap2s + t*(ap2st + t*ap2st2));

    *rhopot =  b0 + t*(bt + t*(bt2 + t*(bt3  + t*(bt4  + t*bt5))))
                  + s*(bs + t*(bst + t*(bst2 + t*(bst3 + t*bst4))))
                  + s*s_sqrt*(bss + t*(bsst + t*bsst2))
                  + s*s* bss2;
    /* Note: Fortran wrote s*(... + s_sqrt*(...) + s*bss2) which is identical
       to s*(...) + s*s_sqrt*(...) + s*s*bss2 — expanded here for clarity. */
}

/*--- JM-EOS components, DEVICE copy (M2.1) ----------------------------------
 * KOKKOS_INLINE_FUNCTION twin of fesom_eos_jm_components above, callable from a
 * device parallel_for. The ONLY changes vs the host twin are mechanical and
 * value-preserving: `static const` → `constexpr` (same IEEE-754 literals) and
 * `sqrt` → `Kokkos::sqrt` (on the host backend this IS libm sqrt — IEEE
 * correctly-rounded, so bit-identical; on CUDA it is libdevice sqrt). The
 * polynomial text is byte-for-byte the host twin. This is a deliberate
 * DUPLICATE while the C twin is in-tree (M2): the FESOM_KK_VERIFY=eos gate
 * cross-checks the two at max|Δ|==0 on Serial, so a copy typo is caught; when
 * the C twin is removed at M2-close this becomes the single definition.
 * Do NOT round, fold, or "simplify" — see PORTING_LESSONS §1/§2.
 */
KOKKOS_INLINE_FUNCTION
void fesom_eos_jm_components_kk(real_t t, real_t s,
                                real_t *bulk_0, real_t *bulk_pz, real_t *bulk_pz2,
                                real_t *rhopot)
{
    constexpr real_t a0    = 19092.56,    at   = 209.8925;
    constexpr real_t at2   = -3.041638,   at3  = -1.852732e-3;
    constexpr real_t at4   = -1.361629e-5;
    constexpr real_t as    = 104.4077,    ast  = -6.500517;
    constexpr real_t ast2  = 0.1553190,   ast3 = 2.326469e-4;
    constexpr real_t ass   = -5.587545,   asst = 0.7390729;
    constexpr real_t asst2 = -1.909078e-2;
    constexpr real_t ap    = -4.721788e-1, apt  = -1.028859e-2;
    constexpr real_t apt2  = 2.512549e-4,  apt3 = 5.939910e-7;
    constexpr real_t aps   = 1.571896e-2,  apst = 2.598241e-4;
    constexpr real_t apst2 = -7.267926e-6, apss = -2.042967e-3;
    constexpr real_t ap2   = 1.045941e-5,  ap2t = -5.782165e-10;
    constexpr real_t ap2t2 = 1.296821e-7;
    constexpr real_t ap2s  = -2.595994e-7, ap2st = -1.248266e-9;
    constexpr real_t ap2st2= -3.508914e-9;

    constexpr real_t b0  = 999.842594,   bt   = 6.793952e-2;
    constexpr real_t bt2 = -9.095290e-3, bt3  = 1.001685e-4;
    constexpr real_t bt4 = -1.120083e-6, bt5  = 6.536332e-9;
    constexpr real_t bs  = 0.824493,     bst  = -4.08990e-3;
    constexpr real_t bst2 = 7.64380e-5,  bst3 = -8.24670e-7;
    constexpr real_t bst4 = 5.38750e-9;
    constexpr real_t bss  = -5.72466e-3, bsst = 1.02270e-4;
    constexpr real_t bsst2 = -1.65460e-6, bss2 = 4.8314e-4;

    real_t s_sqrt = Kokkos::sqrt(s);

    *bulk_0 =  a0      + t*(at   + t*(at2  + t*(at3 + t*at4)))
             + s* (as  + t*(ast  + t*(ast2 + t*ast3))
                  + s_sqrt*(ass  + t*(asst + t*asst2)));

    *bulk_pz =  ap  + t*(apt  + t*(apt2 + t*apt3))
                    + s*(aps + t*(apst + t*apst2) + s_sqrt*apss);

    *bulk_pz2 = ap2 + t*(ap2t + t*ap2t2)
                   + s *(ap2s + t*(ap2st + t*ap2st2));

    *rhopot =  b0 + t*(bt + t*(bt2 + t*(bt3  + t*(bt4  + t*bt5))))
                  + s*(bs + t*(bst + t*(bst2 + t*(bst3 + t*bst4))))
                  + s*s_sqrt*(bss + t*(bsst + t*bsst2))
                  + s*s* bss2;
}

/*--- pressure_bv (Phase 1 subset) -------------------------------------------
 * Mirror of pressure_bv (oce_ale_pressure_bv.F90:194-501) restricted to:
 *   - state_equation = 1 (JM-EOS)
 *   - which_ALE = 'linfs' (uses the hpressure linfs branch)
 *   - no cavity (nzmin = 0 in C / 1 in Fortran, ulevels_nod2D = 1)
 *   - use_density_ref = .false.  → density_ref ≡ density_0
 *   - mix_scheme ≠ KPP (skip dbsfc)
 *   - MLD1 (Large et al. 1997 / FESOM 1.4) is computed (Phase G2a);
 *     MLD2 / MLD3 (Levitus, Griffies) still deferred.
 *   - no N² smoothing
 *
 * Per-node temporaries are stack-allocated VLAs (one column at a time).
 * For Phase 1 nl ≤ 48 — fine on the stack.
 */
void fesom_pressure_bv(const struct fesom_tracers *tracers,
                       const struct fesom_mesh    *mesh,
                       struct fesom_aux           *aux)
{
    const int nl = mesh->nl;
    const real_t g       = (real_t)FESOM_G;
    const real_t rho_ref = (real_t)FESOM_DENSITY_0;
    /* state_equation_int = 1 in Phase 1 (JM-EOS).
       Multiplied through `0.1 * z * state_eq_int` exactly as Fortran does. */
    const real_t state_eq_int = 1.0;

    const real_t *T = tracers->data[FESOM_TRACER_T].values;
    const real_t *S = tracers->data[FESOM_TRACER_S].values;

    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;     /* 1-based → 0-based */
        int nzmax = mesh->nlevels_nod2D[n] - 1;     /* exclusive bound; layers 0..nzmax-1 */

        /* Initialise MLD1_ind (G2a). Fortran sets MLD1_ind=nzmin+1 (1-based),
         * which in C 0-based is just nzmin+1. Stored 0-based throughout —
         * G3 readers add +1 to mirror Fortran's `bvfreq(MLD1_ind+1, n)`. */
        if (aux->MLD1_ind) aux->MLD1_ind[n] = nzmin + 1;

        /* Skip nodes with no wet layers (shouldn't happen in Phase 1 but guard). */
        if (nzmax <= nzmin) continue;

        real_t bulk_0[64], bulk_pz[64], bulk_pz2[64], rhopot[64], rho[64];
        /* nl≤48 in Phase 1; static cap of 64 leaves headroom and keeps the
           compiler from generating dynamic-stack code. */

        /* Pass 1: JM-EOS components per layer. */
        for (int nz = nzmin; nz < nzmax; ++nz) {
            size_t i = FESOM_NODE3D(n, nz, nl);
            fesom_eos_jm_components(T[i], S[i],
                                    &bulk_0[nz], &bulk_pz[nz], &bulk_pz2[nz],
                                    &rhopot[nz]);
        }

        /* Pass 2: in-situ density at mid-layer depth Z[nz], dbsfc1, db_max.
           Phase 1 has no partial cells / no cavity → Z_3d_n[nz][n] = Z[nz].
           db_max accumulates the max buoyancy-gradient over the full column,
           used by MLD1 (Large et al. 1997). Fortran lines 314-335. */
        real_t db_max = 0.0;
        real_t z_nzmin = mesh->Z[nzmin];
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t z = mesh->Z[nz];
            real_t bulk = bulk_0[nz] + z*(bulk_pz[nz] + z*bulk_pz2[nz]);
            real_t r = bulk * rhopot[nz] / (bulk + 0.1 * z * state_eq_int) - rho_ref;
            rho[nz] = r;
            aux->density_m_rho0[FESOM_NODE3D(n, nz, nl)] = r;

            /* Surface-density adiabatically brought to depth z (Fortran 326-328) */
            real_t bulk_surf = bulk_0[nzmin] + z*(bulk_pz[nzmin] + z*bulk_pz2[nzmin]);
            real_t rho_surf = bulk_surf * rhopot[nzmin]
                            / (bulk_surf + 0.1 * z * state_eq_int);

            /* dbsfc1 (Fortran line 332): use_density_ref=false in our config
             * so density_ref(nz, n) = density_0 = rho_ref. So
             * rho(nz)+density_ref = (r) + rho_ref = rho_in_situ_full. */
            real_t r_full = r + rho_ref;
            real_t dbsfc1 = -g * (rho_surf - r_full) / r_full;

            /* Store dbsfc for KPP bldepth (Fortran oce_ale_pressure_bv.F90:332,339).
             * Written unconditionally: PP never reads aux->dbsfc, so the PP result is
             * unaffected (Fortran's `if (mixing_kpp)` gate is a write-skip with no
             * numerical consequence). Bottom interface filled after the loop (:337). */
            aux->dbsfc[FESOM_NODE3D(n, nz, nl)] = dbsfc1;

            /* db_max accumulator (Fortran line 334). At nz==nzmin the
             * divisor is the |Z[nzmin] - Z[nzmin+1]| placeholder — a
             * non-zero divisor; at that level dbsfc1 is 0 anyway since
             * rho_surf == r_full, so the term contributes nothing. */
            int nz_eff = (nz > nzmin) ? nz : (nzmin + 1);
            real_t denom = fabs(z_nzmin - mesh->Z[nz_eff]);
            real_t cand = dbsfc1 / denom;
            if (cand > db_max) db_max = cand;
        }
        /* dbsfc bottom fill: dbsfc(nzmax)=dbsfc(nzmax-1) (Fortran :337). nzmax is the
         * deepest interface (0-based nlevels-1); the loop filled nzmin..nzmax-1. */
        aux->dbsfc[FESOM_NODE3D(n, nzmax, nl)] =
            aux->dbsfc[FESOM_NODE3D(n, nzmax - 1, nl)];

        /* hpressure — linfs branch, no cavity (nzmin == 0 in C).
           Mirror of oce_ale_pressure_bv.F90 lines 369-403. Surface boundary:
             hpressure[nzmin] = -Z[nzmin] * rho[nzmin] * g
           Then accumulate downward by 0.5 g * (rho_above*h_above + rho*h). */
        aux->hpressure[FESOM_NODE3D(n, nzmin, nl)] =
            -mesh->Z[nzmin] * rho[nzmin] * g;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t h_up   = mesh->hnode[FESOM_NODE3D(n, nz - 1, nl)];
            real_t h_this = mesh->hnode[FESOM_NODE3D(n, nz,     nl)];
            real_t a = 0.5 * g * (rho[nz - 1] * h_up + rho[nz] * h_this);
            aux->hpressure[FESOM_NODE3D(n, nz, nl)] =
                aux->hpressure[FESOM_NODE3D(n, nz - 1, nl)] + a;
        }

        /* bvfreq — N² between layers nzmin+1 and nzmax-1, then padded.
           Mirror of lines 427-475. Both rho_up and rho_dn are evaluated at the
           *same depth* zmean to cancel compressibility. Also locates MLD1
           (Phase G2a): the shallowest level where N² > db_max. */
        int mld1_done = 0;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            real_t zmean   = 0.5 * (mesh->Z[nz - 1] + mesh->Z[nz]);
            real_t bulk_up = bulk_0[nz - 1] + zmean*(bulk_pz[nz - 1] + zmean*bulk_pz2[nz - 1]);
            real_t bulk_dn = bulk_0[nz    ] + zmean*(bulk_pz[nz    ] + zmean*bulk_pz2[nz    ]);
            real_t rho_up  = bulk_up * rhopot[nz - 1] / (bulk_up + 0.1 * zmean * state_eq_int);
            real_t rho_dn  = bulk_dn * rhopot[nz    ] / (bulk_dn + 0.1 * zmean * state_eq_int);
            real_t dz_inv  = 1.0 / (mesh->Z[nz - 1] - mesh->Z[nz]);
            real_t bv = -g * dz_inv * (rho_up - rho_dn) / rho_ref;
            aux->bvfreq[FESOM_NODE3D(n, nz, nl)] = bv;

            /* MLD1: Large et al. (1997). Fortran lines 447-451.
             * Stored 0-based (G3 readers add +1, mirroring Fortran's
             * `bvfreq(MLD1_ind+1, n)`). */
            if (!mld1_done && bv > db_max && aux->MLD1_ind) {
                aux->MLD1_ind[n] = nz;
                mld1_done = 1;
            }
        }
        /* Pad surface and bottom — Fortran lines 474-475:
         *   bvfreq(nzmin) = bvfreq(nzmin+1)        [surface]
         *   bvfreq(nzmax) = bvfreq(nzmax-1)        [bottom interface]
         * nzmax is the deepest interface (0-based nlevels-1); the loop filled the
         * interior nzmin+1..nzmax-1. The horizontal N² smoothing (N2smth_h=.true.,
         * Fortran pressure_bv:499) is applied by fesom_smooth_nod3D from fesom_step
         * after the bvfreq halo exchange. */
        if (nzmin + 1 < nzmax) {
            aux->bvfreq[FESOM_NODE3D(n, nzmin, nl)] =
                aux->bvfreq[FESOM_NODE3D(n, nzmin + 1, nl)];
            aux->bvfreq[FESOM_NODE3D(n, nzmax, nl)] =
                aux->bvfreq[FESOM_NODE3D(n, nzmax - 1, nl)];
        }
    }
}

/*--- pressure_bv — DEVICE kernel (M2.1, the first one) -----------------------
 * Kokkos parallel_for over owned nodes; the whole per-column body (the level
 * loops) runs inside the lambda, so each node owns its column and there is NO
 * cross-node data race → the Serial range is sequential == the C twin loop, and
 * even OpenMP is bit-identical here (a pure map, no reduction/atomic). Every
 * constant, bound, and accumulation order is a verbatim copy of the host twin
 * fesom_pressure_bv above (re-read it side-by-side to review). The per-column
 * temporaries (bulk_0, bulk_pz, bulk_pz2, rhopot, rho — each [64]) are lambda-local
 * (per-thread local memory on device; nl<=48 in Phase 1, cap 64) — uncoalesced and
 * slow-first is accepted (correctness over speed this phase).
 *
 * SYNC contract (SYNC_MAP §1): INPUTS (tracers T/S `values`, mesh `hnode`) must
 * be device-current — the step driver's EOS input rail does modify_host()+
 * sync_device() on them (they are host-written via the raw alias each step, L14).
 * Mesh Z/ulevels/nlevels are set-once and already device-current (one-shot push,
 * mesh_sync_geometry_device). OUTPUTS are marked modify_device() at the end; the
 * driver sync_host()s them before the halo exchanges + smooth_nod3D + consumers.
 */
void fesom_pressure_bv_kk(const struct fesom_tracers *tracers,
                          const struct fesom_mesh    *mesh,
                          struct fesom_aux           *aux)
{
    const int    nl          = mesh->nl;
    const int    myDim       = mesh->myDim_nod2D;
    const real_t g           = (real_t)FESOM_G;
    const real_t rho_ref     = (real_t)FESOM_DENSITY_0;
    const real_t state_eq_int = 1.0;                       /* JM-EOS in Phase 1 */
    const bool   has_mld1    = aux->MLD1_ind_fld.allocated();

    /* Device views (LayoutRight; host-current inputs pushed by the driver rail). */
    auto T    = tracers->data[FESOM_TRACER_T].values_fld.d();
    auto S    = tracers->data[FESOM_TRACER_S].values_fld.d();
    auto Z    = mesh->Z_fld.d();
    auto hnod = mesh->hnode_fld.d();
    auto ulev = mesh->ulevels_nod2D_fld.d();
    auto nlev = mesh->nlevels_nod2D_fld.d();
    auto density   = aux->density_m_rho0_fld.d();
    auto hpressure = aux->hpressure_fld.d();
    auto bvfreq    = aux->bvfreq_fld.d();
    auto dbsfc     = aux->dbsfc_fld.d();
    auto mld1      = aux->MLD1_ind_fld.d();

    Kokkos::parallel_for("fesom_pressure_bv", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev(n) - 1;                /* 1-based → 0-based */
            int nzmax = nlev(n) - 1;                /* exclusive bound; layers 0..nzmax-1 */

            if (has_mld1) mld1(n) = nzmin + 1;      /* G2a init (0-based) */

            if (nzmax <= nzmin) return;             /* no wet layers (C twin: continue) */

            real_t bulk_0[64], bulk_pz[64], bulk_pz2[64], rhopot[64], rho[64];

            /* Pass 1: JM-EOS components per layer. */
            for (int nz = nzmin; nz < nzmax; ++nz) {
                size_t i = FESOM_NODE3D(n, nz, nl);
                fesom_eos_jm_components_kk(T(i), S(i),
                                           &bulk_0[nz], &bulk_pz[nz], &bulk_pz2[nz],
                                           &rhopot[nz]);
            }

            /* Pass 2: in-situ density at mid-layer depth Z[nz], dbsfc1, db_max. */
            real_t db_max = 0.0;
            real_t z_nzmin = Z(nzmin);
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t z = Z(nz);
                real_t bulk = bulk_0[nz] + z*(bulk_pz[nz] + z*bulk_pz2[nz]);
                real_t r = bulk * rhopot[nz] / (bulk + 0.1 * z * state_eq_int) - rho_ref;
                rho[nz] = r;
                density(FESOM_NODE3D(n, nz, nl)) = r;

                real_t bulk_surf = bulk_0[nzmin] + z*(bulk_pz[nzmin] + z*bulk_pz2[nzmin]);
                real_t rho_surf = bulk_surf * rhopot[nzmin]
                                / (bulk_surf + 0.1 * z * state_eq_int);

                real_t r_full = r + rho_ref;
                real_t dbsfc1 = -g * (rho_surf - r_full) / r_full;
                dbsfc(FESOM_NODE3D(n, nz, nl)) = dbsfc1;

                int nz_eff = (nz > nzmin) ? nz : (nzmin + 1);
                real_t denom = Kokkos::fabs(z_nzmin - Z(nz_eff));
                real_t cand = dbsfc1 / denom;
                if (cand > db_max) db_max = cand;
            }
            dbsfc(FESOM_NODE3D(n, nzmax, nl)) =
                dbsfc(FESOM_NODE3D(n, nzmax - 1, nl));

            /* hpressure — linfs branch, no cavity (nzmin == 0 in C). */
            hpressure(FESOM_NODE3D(n, nzmin, nl)) =
                -Z(nzmin) * rho[nzmin] * g;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t h_up   = hnod(FESOM_NODE3D(n, nz - 1, nl));
                real_t h_this = hnod(FESOM_NODE3D(n, nz,     nl));
                real_t a = 0.5 * g * (rho[nz - 1] * h_up + rho[nz] * h_this);
                hpressure(FESOM_NODE3D(n, nz, nl)) =
                    hpressure(FESOM_NODE3D(n, nz - 1, nl)) + a;
            }

            /* bvfreq — N² between layers nzmin+1 and nzmax-1, then padded. */
            int mld1_done = 0;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t zmean   = 0.5 * (Z(nz - 1) + Z(nz));
                real_t bulk_up = bulk_0[nz - 1] + zmean*(bulk_pz[nz - 1] + zmean*bulk_pz2[nz - 1]);
                real_t bulk_dn = bulk_0[nz    ] + zmean*(bulk_pz[nz    ] + zmean*bulk_pz2[nz    ]);
                real_t rho_up  = bulk_up * rhopot[nz - 1] / (bulk_up + 0.1 * zmean * state_eq_int);
                real_t rho_dn  = bulk_dn * rhopot[nz    ] / (bulk_dn + 0.1 * zmean * state_eq_int);
                real_t dz_inv  = 1.0 / (Z(nz - 1) - Z(nz));
                real_t bv = -g * dz_inv * (rho_up - rho_dn) / rho_ref;
                bvfreq(FESOM_NODE3D(n, nz, nl)) = bv;

                if (!mld1_done && bv > db_max && has_mld1) {
                    mld1(n) = nz;
                    mld1_done = 1;
                }
            }
            if (nzmin + 1 < nzmax) {
                bvfreq(FESOM_NODE3D(n, nzmin, nl)) =
                    bvfreq(FESOM_NODE3D(n, nzmin + 1, nl));
                bvfreq(FESOM_NODE3D(n, nzmax, nl)) =
                    bvfreq(FESOM_NODE3D(n, nzmax - 1, nl));
            }
        });

    /* Outputs now device-authoritative (driver sync_host()s them before halos). */
    aux->density_m_rho0_fld.modify_device();
    aux->hpressure_fld.modify_device();
    aux->bvfreq_fld.modify_device();
    aux->dbsfc_fld.modify_device();
    if (has_mld1) aux->MLD1_ind_fld.modify_device();
}

/*--- smooth_nod3D — area-weighted node-patch horizontal smoother --------------
 * Literal port of smooth_nod3D (gen_support.F90:99-198). Each owned node's value
 * at level nz becomes the area-weighted mean of the three vertices of every
 * surrounding element that reaches level nz:
 *     arr(nz,n) = Σ_el area_el·(arr(nz,V1)+arr(nz,V2)+arr(nz,V3)) / (3·Σ_el area_el)
 * Elements that bottom out above nz drop from both sums (so the patch shrinks
 * with depth, matching Fortran's `nle = min(nln, nlevels(el))`). One full sweep
 * per smoothing cycle, each followed by a halo exchange. The inverse patch area
 * `vol` is built on the first sweep and reused by later sweeps, exactly as the
 * Fortran. The caller must have a valid halo `arr` on entry (the sweep reads halo
 * vertices); the bvfreq use provides that via the preceding fesom_exchange_nod3D
 * (Fortran fills bvfreq on myDim+eDim instead, then smooths). Used for N² with
 * N2smth_h=.true., N2smth_hidx=1 (oce_modules.F90:105). */
void fesom_smooth_nod3D(real_t *arr, int nl, int n_smooth,
                        const struct fesom_mesh *mesh, struct fesom_partit *p)
{
    if (n_smooth < 1) return;
    const int Nmy = mesh->myDim_nod2D;
    const size_t sz = (size_t)Nmy * (size_t)nl;
    real_t *vol  = (real_t *)malloc(sz * sizeof(real_t));
    real_t *work = (real_t *)malloc(sz * sizeof(real_t));
    FESOM_CHECK(vol && work, "fesom_smooth_nod3D: out of memory");

    for (int sweep = 0; sweep < n_smooth; ++sweep) {
        for (int n = 0; n < Nmy; ++n) {
            int uln  = mesh->ulevels_nod2D[n] - 1;     /* 0-based first level   */
            int nlnz = mesh->nlevels_nod2D[n] - 1;     /* 0-based deepest level */
            for (int nz = uln; nz <= nlnz; ++nz) {
                if (sweep == 0) vol[(size_t)n*nl + nz] = 0.0;
                work[(size_t)n*nl + nz] = 0.0;
            }
            int o0 = mesh->nod_in_elem2D_offsets[n];
            int o1 = mesh->nod_in_elem2D_offsets[n + 1];
            for (int k = o0; k < o1; ++k) {
                int el  = mesh->nod_in_elem2D[k];
                int ule = mesh->ulevels[el] - 1;  if (ule < uln)  ule = uln;
                int nle = mesh->nlevels[el] - 1;  if (nle > nlnz) nle = nlnz;
                real_t a = mesh->elem_area[el];
                int v0 = mesh->elem_nodes[3*el + 0];
                int v1 = mesh->elem_nodes[3*el + 1];
                int v2 = mesh->elem_nodes[3*el + 2];
                for (int nz = ule; nz <= nle; ++nz) {
                    if (sweep == 0) vol[(size_t)n*nl + nz] += a;
                    work[(size_t)n*nl + nz] += a * ( arr[(size_t)v0*nl + nz]
                                                   + arr[(size_t)v1*nl + nz]
                                                   + arr[(size_t)v2*nl + nz] );
                }
            }
            /* invert the patch area once (1/(3·Σarea)); reused on later sweeps */
            if (sweep == 0)
                for (int nz = uln; nz <= nlnz; ++nz)
                    vol[(size_t)n*nl + nz] = 1.0 / (3.0 * vol[(size_t)n*nl + nz]);
        }
        /* combined scale + copy back to owned nodes (halo overwritten by exchange) */
        for (int n = 0; n < Nmy; ++n) {
            int uln  = mesh->ulevels_nod2D[n] - 1;
            int nlnz = mesh->nlevels_nod2D[n] - 1;
            for (int nz = uln; nz <= nlnz; ++nz)
                arr[(size_t)n*nl + nz] = work[(size_t)n*nl + nz] * vol[(size_t)n*nl + nz];
        }
        fesom_exchange_nod3D(arr, nl, p);
    }
    free(vol);
    free(work);
}

/*--- pressure_force_4_linfs_fullcell ---------------------------------------
 * Mirror of oce_ale_pressure_bv.F90:575-614. Five-line element loop:
 *   pgf_x[e][nz] = Σ_i gradient_sca[1..3] * hpressure[nz][V_i(e)] / density_0
 *   pgf_y[e][nz] = Σ_i gradient_sca[4..6] * hpressure[nz][V_i(e)] / density_0
 * Vertex layout in our gradient_sca: [6e + i] for i = 0..5.
 */
void fesom_pressure_force_linfs_fullcell(const struct fesom_mesh *mesh,
                                         struct fesom_aux        *aux)
{
    const int nl       = mesh->nl;
    const real_t inv_r = 1.0 / (real_t)FESOM_DENSITY_0;

    /* Interior elements only — gradient_sca and elem_nodes are sized
     * myDim_elem2D in the MPI port. The PGF on halo elements arrives via
     * the elem_area exchange path inside fesom_step's per-step exchanges. */
    for (int e = 0; e < mesh->myDim_elem2D; ++e) {
        int nzmin = mesh->ulevels[e]   - 1;     /* 1-based → 0-based */
        int nzmax = mesh->nlevels[e]   - 1;     /* exclusive bound  */
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];

        const real_t *g = &mesh->gradient_sca[6*e];     /* [dN1x..dN3x, dN1y..dN3y] */

        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t hp0 = aux->hpressure[FESOM_NODE3D(n0, nz, nl)];
            real_t hp1 = aux->hpressure[FESOM_NODE3D(n1, nz, nl)];
            real_t hp2 = aux->hpressure[FESOM_NODE3D(n2, nz, nl)];
            aux->pgf_x[FESOM_ELEM3D(e, nz, nl)] =
                (g[0]*hp0 + g[1]*hp1 + g[2]*hp2) * inv_r;
            aux->pgf_y[FESOM_ELEM3D(e, nz, nl)] =
                (g[3]*hp0 + g[4]*hp1 + g[5]*hp2) * inv_r;
        }
    }
}

/*--- pressure_force_linfs_fullcell — DEVICE kernel (M2.4) ---------------------
 * Kokkos parallel_for over owned elements (the M2.1 EOS template, D19). The level
 * loop runs inside the lambda and each element writes only its own pgf_x/pgf_y →
 * NO cross-element write race → the Serial range is sequential == the C twin loop,
 * and even OpenMP is bit-identical (a pure per-element map, no reduction/atomic).
 * Every bound/constant/association is a verbatim copy of the C twin above (the
 * `(g0*hp0 + g1*hp1 + g2*hp2) * inv_r` parenthesisation is preserved exactly).
 *
 * SYNC contract (SYNC_MAP §2 row 2): INPUT aux->hpressure was produced on the device
 * (pressure_bv_kk, substep 1), then sync_host'd + halo-exchanged on the HOST — the halo
 * write is invisible to the DualView (L14/L27), so the driver's substep-2 rail does
 * modify_host()+sync_device() on it (NOT a bare sync_device()). The set-once mesh
 * gradient_sca/elem_nodes/ulevels/nlevels are already device-current (one-shot push).
 * OUTPUTS pgf_x/pgf_y marked modify_device(); the driver sync_host()s them before the
 * elem3D halos.
 */
void fesom_pressure_force_linfs_fullcell_kk(const struct fesom_mesh *mesh,
                                            struct fesom_aux        *aux)
{
    const int    nl    = mesh->nl;
    const int    E     = mesh->myDim_elem2D;
    const real_t inv_r = 1.0 / (real_t)FESOM_DENSITY_0;

    auto hpressure = aux->hpressure_fld.d();
    auto pgf_x     = aux->pgf_x_fld.d();
    auto pgf_y     = aux->pgf_y_fld.d();
    auto ulev      = mesh->ulevels_fld.d();
    auto nlev      = mesh->nlevels_fld.d();
    auto elnod     = mesh->elem_nodes_fld.d();
    auto gradsca   = mesh->gradient_sca_fld.d();

    Kokkos::parallel_for("fesom_pressure_force", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int e) {
            int nzmin = ulev(e) - 1;            /* 1-based → 0-based */
            int nzmax = nlev(e) - 1;            /* exclusive bound  */
            int n0 = elnod(3*e + 0);
            int n1 = elnod(3*e + 1);
            int n2 = elnod(3*e + 2);
            real_t g0 = gradsca(6*e + 0), g1 = gradsca(6*e + 1), g2 = gradsca(6*e + 2);
            real_t g3 = gradsca(6*e + 3), g4 = gradsca(6*e + 4), g5 = gradsca(6*e + 5);
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t hp0 = hpressure(FESOM_NODE3D(n0, nz, nl));
                real_t hp1 = hpressure(FESOM_NODE3D(n1, nz, nl));
                real_t hp2 = hpressure(FESOM_NODE3D(n2, nz, nl));
                pgf_x(FESOM_ELEM3D(e, nz, nl)) = (g0*hp0 + g1*hp1 + g2*hp2) * inv_r;
                pgf_y(FESOM_ELEM3D(e, nz, nl)) = (g3*hp0 + g4*hp1 + g5*hp2) * inv_r;
            }
        });

    aux->pgf_x_fld.modify_device();   /* driver sync_host()s before the elem3D halos */
    aux->pgf_y_fld.modify_device();
}

/*--- FESOM_KK_VERIFY=pgf (M2.4) ----------------------------------------------
 * EOS-style gate (fesom_eos_verify shape, D19): pgf_x/pgf_y are FULL overwrites
 * from inputs the kernel never modifies (hpressure / mesh), so the C twin
 * recomputes from intact state — no capture-before needed (cf. L26). On entry aux
 * holds the Kokkos result (driver sync_host'd it). Snapshot it, run the C twin
 * (overwrites the owned region via the raw alias), diff, RESTORE the Kokkos result.
 * Non-intrusive; asserts max|Δ|==0 on Serial.
 */
void fesom_pressure_force_verify(const struct fesom_mesh *mesh,
                                 struct fesom_aux        *aux,
                                 int step_n)
{
    const int nl = mesh->nl;
    const size_t nE = (size_t)mesh->myDim_elem2D * (size_t)nl;     /* owned elements */

    std::vector<real_t> kk_pgf_x(aux->pgf_x, aux->pgf_x + nE);
    std::vector<real_t> kk_pgf_y(aux->pgf_y, aux->pgf_y + nE);
    fesom_pressure_force_linfs_fullcell(mesh, aux);     /* C twin overwrites via raw alias */

    auto maxdiff = [](const std::vector<real_t> &kk, const real_t *c, size_t n) -> double {
        double m = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs((double)kk[i] - (double)c[i]);
            if (d > m) m = d;
        }
        return m;
    };
    double d_x = maxdiff(kk_pgf_x, aux->pgf_x, nE);
    double d_y = maxdiff(kk_pgf_y, aux->pgf_y, nE);
    std::copy(kk_pgf_x.begin(), kk_pgf_x.end(), aux->pgf_x);       /* restore KK */
    std::copy(kk_pgf_y.begin(), kk_pgf_y.end(), aux->pgf_y);

    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    const double dmax = std::max(d_x, d_y);
    std::printf("[FESOM_KK_VERIFY=pgf] step %d backend=%s  max|Δ|: pgf_x=%.3e pgf_y=%.3e\n",
                step_n, backend.c_str(), d_x, d_y);
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=pgf] FAIL: Serial must be bit-identical to the C twin "
                             "(max|Δ|=%.3e)\n", dmax);
        std::abort();
    }
}

/*--- sw_alpha_beta (oce_ale_pressure_bv.F90:2751-2846) -------------------------
 * McDougall (1987) thermal expansion and saline contraction coefficients per
 * node per level. Inputs: potential T (°C), S (PSU); pressure proxy = |Z[nz]|.
 *
 * For our linfs / no-cavity / no-partial-cells config Z_3d_n[nz, n] = Z[nz].
 *
 * Halo: Fortran exchange_nod on sw_alpha and sw_beta at end. We mirror.
 * ----------------------------------------------------------------------------- */
void fesom_compute_sw_alpha_beta(const struct fesom_tracers *tracers,
                                 const struct fesom_mesh    *mesh,
                                 struct fesom_aux           *aux)
{
    const int nl     = mesh->nl;
    const int myDim  = mesh->myDim_nod2D;
    const real_t *T  = tracers->data[FESOM_TRACER_T].values;
    const real_t *S  = tracers->data[FESOM_TRACER_S].values;

    for (int n = 0; n < myDim; ++n) {
        int nzmax = mesh->nlevels_nod2D[n] - 1;     /* layer count exclusive */
        int nzmin = mesh->ulevels_nod2D[n] - 1;     /* 0-based */
        for (int nz = nzmin; nz < nzmax; ++nz) {
            real_t t1 = T[FESOM_NODE3D(n, nz, nl)] * 1.00024;
            real_t s1 = S[FESOM_NODE3D(n, nz, nl)];
            real_t p1 = fabs(mesh->Z[nz]);

            real_t t1_2 = t1*t1;
            real_t t1_3 = t1_2*t1;
            real_t t1_4 = t1_3*t1;
            real_t p1_2 = p1*p1;
            real_t p1_3 = p1_2*p1;
            real_t s35  = s1 - 35.0;
            real_t s35_2 = s35*s35;

            real_t beta = 0.785567e-3
                        - 0.301985e-5*t1
                        + 0.555579e-7*t1_2
                        - 0.415613e-9*t1_3
                        + s35*(-0.356603e-6 + 0.788212e-8*t1
                              + 0.408195e-10*p1 - 0.602281e-15*p1_2)
                        + s35_2*(0.515032e-8)
                        + p1*(-0.121555e-7 + 0.192867e-9*t1 - 0.213127e-11*t1_2)
                        + p1_2*(0.176621e-12 - 0.175379e-14*t1)
                        + p1_3*(0.121551e-17);

            real_t a_over_b = 0.665157e-1
                            + 0.170907e-1*t1
                            - 0.203814e-3*t1_2
                            + 0.298357e-5*t1_3
                            - 0.255019e-7*t1_4
                            + s35*(0.378110e-2 - 0.846960e-4*t1
                                  - 0.164759e-6*p1 - 0.251520e-11*p1_2)
                            + s35_2*(-0.678662e-5)
                            + p1*(0.380374e-4 - 0.933746e-6*t1 + 0.791325e-8*t1_2)
                            + p1_2*t1_2*(0.512857e-12)
                            - p1_3*(0.302285e-13);

            aux->sw_beta [FESOM_NODE3D(n, nz, nl)] = beta;
            aux->sw_alpha[FESOM_NODE3D(n, nz, nl)] = a_over_b * beta;
        }
    }
}

/*--- sw_alpha_beta — DEVICE kernel (M2.1) ------------------------------------
 * Kokkos parallel_for twin of fesom_compute_sw_alpha_beta above. Pure per-node
 * map (no sqrt, polynomial only), so Serial == the C twin and OpenMP is
 * bit-identical too. Every coefficient is a verbatim copy. INPUTS T/S are
 * pushed device-current by the driver's EOS rail (already synced for
 * pressure_bv this step); Z is set-once device-current. OUTPUTS sw_alpha/
 * sw_beta are marked modify_device(); the driver sync_host()s them before halos.
 */
void fesom_compute_sw_alpha_beta_kk(const struct fesom_tracers *tracers,
                                    const struct fesom_mesh    *mesh,
                                    struct fesom_aux           *aux)
{
    const int nl    = mesh->nl;
    const int myDim = mesh->myDim_nod2D;
    auto T  = tracers->data[FESOM_TRACER_T].values_fld.d();
    auto S  = tracers->data[FESOM_TRACER_S].values_fld.d();
    auto Z  = mesh->Z_fld.d();
    auto ulev = mesh->ulevels_nod2D_fld.d();
    auto nlev = mesh->nlevels_nod2D_fld.d();
    auto sw_alpha = aux->sw_alpha_fld.d();
    auto sw_beta  = aux->sw_beta_fld.d();

    Kokkos::parallel_for("fesom_sw_alpha_beta", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            int nzmax = nlev(n) - 1;            /* layer count exclusive */
            int nzmin = ulev(n) - 1;            /* 0-based */
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t t1 = T(FESOM_NODE3D(n, nz, nl)) * 1.00024;
                real_t s1 = S(FESOM_NODE3D(n, nz, nl));
                real_t p1 = Kokkos::fabs(Z(nz));

                real_t t1_2 = t1*t1;
                real_t t1_3 = t1_2*t1;
                real_t t1_4 = t1_3*t1;
                real_t p1_2 = p1*p1;
                real_t p1_3 = p1_2*p1;
                real_t s35  = s1 - 35.0;
                real_t s35_2 = s35*s35;

                real_t beta = 0.785567e-3
                            - 0.301985e-5*t1
                            + 0.555579e-7*t1_2
                            - 0.415613e-9*t1_3
                            + s35*(-0.356603e-6 + 0.788212e-8*t1
                                  + 0.408195e-10*p1 - 0.602281e-15*p1_2)
                            + s35_2*(0.515032e-8)
                            + p1*(-0.121555e-7 + 0.192867e-9*t1 - 0.213127e-11*t1_2)
                            + p1_2*(0.176621e-12 - 0.175379e-14*t1)
                            + p1_3*(0.121551e-17);

                real_t a_over_b = 0.665157e-1
                                + 0.170907e-1*t1
                                - 0.203814e-3*t1_2
                                + 0.298357e-5*t1_3
                                - 0.255019e-7*t1_4
                                + s35*(0.378110e-2 - 0.846960e-4*t1
                                      - 0.164759e-6*p1 - 0.251520e-11*p1_2)
                                + s35_2*(-0.678662e-5)
                                + p1*(0.380374e-4 - 0.933746e-6*t1 + 0.791325e-8*t1_2)
                                + p1_2*t1_2*(0.512857e-12)
                                - p1_3*(0.302285e-13);

                sw_beta (FESOM_NODE3D(n, nz, nl)) = beta;
                sw_alpha(FESOM_NODE3D(n, nz, nl)) = a_over_b * beta;
            }
        });

    aux->sw_alpha_fld.modify_device();
    aux->sw_beta_fld.modify_device();
}

/*--- FESOM_KK_VERIFY=eos — in-binary per-kernel gate (M2.1) -------------------
 * The in-binary analogue of scripts/exp1_compare_bidiff.py: run the HOST C twin
 * and the DEVICE Kokkos kernel on the SAME live state and report max|Δ| over the
 * owned region for every EOS output, asserting max|Δ|==0 on the Serial backend
 * (the bit-identity oracle). On entry aux already holds the Kokkos production
 * result (the driver ran the *_kk kernels and sync_host()'d the outputs). We:
 *   1. snapshot that Kokkos result,
 *   2. run the C twins (they overwrite aux on the host via the raw alias),
 *   3. diff Kokkos-snapshot vs C-twin (now in aux),
 *   4. RESTORE the Kokkos result into aux (so the step proceeds on the device
 *      path and host==device coherence is preserved on CUDA),
 *   5. report; abort on Serial if not bit-identical.
 * Diagnostic only (env-gated, never on the production hot path); the snapshot
 * buffers are plain host vectors — they are never touched by a device kernel, so
 * they are NOT "kernel scratch" (no Field wrapping needed; cf. SYNC_MAP §9.5).
 */
void fesom_eos_verify(const struct fesom_tracers *tracers,
                      const struct fesom_mesh    *mesh,
                      struct fesom_aux           *aux,
                      int step_n)
{
    const int    nl    = mesh->nl;
    const int    myDim = mesh->myDim_nod2D;
    const size_t n3    = (size_t)myDim * (size_t)nl;

    /* 1. snapshot the Kokkos production result (aux is host-current). */
    std::vector<real_t> kk_density(aux->density_m_rho0, aux->density_m_rho0 + n3);
    std::vector<real_t> kk_hpress (aux->hpressure,      aux->hpressure      + n3);
    std::vector<real_t> kk_bvfreq (aux->bvfreq,         aux->bvfreq         + n3);
    std::vector<real_t> kk_dbsfc  (aux->dbsfc,          aux->dbsfc          + n3);
    std::vector<real_t> kk_alpha  (aux->sw_alpha,       aux->sw_alpha       + n3);
    std::vector<real_t> kk_beta   (aux->sw_beta,        aux->sw_beta        + n3);
    std::vector<int>    kk_mld1;
    if (aux->MLD1_ind) kk_mld1.assign(aux->MLD1_ind, aux->MLD1_ind + myDim);

    /* 2. run the host C twins on the same state (they write aux via the raw alias). */
    fesom_pressure_bv(tracers, mesh, aux);
    fesom_compute_sw_alpha_beta(tracers, mesh, aux);

    /* 3. max|Δ| over the owned region, per field. */
    auto maxdiff = [n3](const std::vector<real_t> &kk, const real_t *c) -> double {
        double m = 0.0;
        for (size_t i = 0; i < n3; ++i) {
            double d = std::fabs((double)kk[i] - (double)c[i]);
            if (d > m) m = d;
        }
        return m;
    };
    double d_density = maxdiff(kk_density, aux->density_m_rho0);
    double d_hpress  = maxdiff(kk_hpress,  aux->hpressure);
    double d_bvfreq  = maxdiff(kk_bvfreq,  aux->bvfreq);
    double d_dbsfc   = maxdiff(kk_dbsfc,   aux->dbsfc);
    double d_alpha   = maxdiff(kk_alpha,   aux->sw_alpha);
    double d_beta    = maxdiff(kk_beta,    aux->sw_beta);
    long   mld1_mismatch = 0;
    if (aux->MLD1_ind)
        for (int n = 0; n < myDim; ++n)
            if (kk_mld1[(size_t)n] != aux->MLD1_ind[n]) ++mld1_mismatch;

    /* 4. restore the Kokkos result into aux (production path proceeds on device). */
    std::copy(kk_density.begin(), kk_density.end(), aux->density_m_rho0);
    std::copy(kk_hpress.begin(),  kk_hpress.end(),  aux->hpressure);
    std::copy(kk_bvfreq.begin(),  kk_bvfreq.end(),  aux->bvfreq);
    std::copy(kk_dbsfc.begin(),   kk_dbsfc.end(),   aux->dbsfc);
    std::copy(kk_alpha.begin(),   kk_alpha.end(),   aux->sw_alpha);
    std::copy(kk_beta.begin(),    kk_beta.end(),    aux->sw_beta);
    if (aux->MLD1_ind)
        std::copy(kk_mld1.begin(), kk_mld1.end(), aux->MLD1_ind);

    /* 5. report + assert. Serial is the bit-identity oracle; OpenMP is a pure map
     *    (no reduction) so it is also bit-identical here; CUDA is climate-close
     *    (fma + libdevice sqrt) and only reported. */
    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    const double dmax = std::max({d_density, d_hpress, d_bvfreq, d_dbsfc, d_alpha, d_beta});
    std::printf("[FESOM_KK_VERIFY=eos] step %d backend=%s  max|Δ|: density=%.3e hpressure=%.3e "
                "bvfreq=%.3e dbsfc=%.3e sw_alpha=%.3e sw_beta=%.3e  MLD1_mismatch=%ld\n",
                step_n, backend.c_str(), d_density, d_hpress, d_bvfreq, d_dbsfc,
                d_alpha, d_beta, mld1_mismatch);
    std::fflush(stdout);
    if (backend == "Serial" && (dmax != 0.0 || mld1_mismatch != 0)) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=eos] FAIL: Serial must be bit-identical to the C twin "
                             "(max|Δ|=%.3e, MLD1 mismatch=%ld)\n", dmax, mld1_mismatch);
        std::abort();
    }
}
