/*
 * KPP vertical mixing — faithful port of oce_ale_mixing_kpp.F90.
 * See fesom_kpp.h for the struct layout and docs/plans/20260524-kpp-vertical-mixing.md
 * for the phased port plan. Phase K0 lands the scaffolding (struct alloc/free,
 * abort-stub driver, dump harness); the routine bodies arrive K1..K8.
 */
#include "fesom_kpp.h"
#include "fesom_aux.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_eos.h"      /* fesom_smooth_nod3D (smooth_blmc) */
#include "fesom_forcing.h"
#include "fesom_halo.h"     /* fesom_exchange_nod3D (combine exchanges) */
#include "fesom_halo_device.hpp"   /* M5.5 (B): blmc device smoother + slab device-halo */
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_speed.hpp"  /* M7 Task A.1: FESOM_SPEED_FLAT */
#include "fesom_tracers.h"

#include <Kokkos_Core.hpp>   // M2.3b: device kernels (parallel_for) + Kokkos:: math
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>            // M2.3b: FESOM_KK_VERIFY=kpp snapshot buffers (host-only diagnostic)
#include <string>
#include <algorithm>

/*===========================================================================
 * KPP constants — verified against oce_ale_mixing_kpp.F90 + the CORE2
 * reference namelists (docs/kpp_reference_namelists/). Module parameters and
 * namelist values that the CORE2/KPP run uses. A_ver/K_ver reuse the shared
 * FESOM_PHASE1_{A,K}_VER (== the CORE2 1e-4 / 1e-5; same constant as PP).
 *===========================================================================*/
/* module parameters (oce_ale_mixing_kpp.F90:50-59) */
/* M8/Task-3 epsilon-FTZ policy: an additive guard MUST be a NORMAL number in the
 * working precision. 1.0e-40 is subnormal in float32; FTZ/DAZ (Intel -O3, CUDA
 * default) flushes it to 0, turning every  x/(y+epsln)  into 0/0 = NaN exactly at
 * cold start when denominators are still zero — FESOM2 PR-940's headline SP bug
 * (their fix commit 48c37328). FP64 keeps 1.0e-40 so Gate 0 is untouched. */
#if defined(FESOM_SINGLE_PRECISION)
#define KPP_EPSLN        1.0e-20    /* epsln  (:50) — FP32-normal, FTZ-safe */
#else
#define KPP_EPSLN        1.0e-40    /* epsln  (:50)  */
#endif
#define KPP_EPSILON      0.1        /* epsilon_kpp (:52) */
#define KPP_VONK         0.4        /* vonk   (:53)  */
#define KPP_CONC1        5.0        /* conc1  (:54)  */
#define KPP_ZMIN        (-4.0e-7)   /* zmin   (:56)  lookup-table zehat min */
#define KPP_ZMAX         0.0        /* zmax   (:57)  */
#define KPP_UMIN         0.0        /* umin   (:58)  lookup-table ustar min */
#define KPP_UMAX         0.04       /* umax   (:59)  */
/* namelist values (work_core; module defaults coincide except A_ver) */
#define KPP_RICR         0.3        /* Ricr   (namelist.oce:70) */
#define KPP_CONCV        1.6        /* concv  (namelist.oce:71) */
#define KPP_VISC_SH_LIMIT 5.0e-3    /* visc_sh_limit (namelist.oce:66) */
#define KPP_DIFF_SH_LIMIT 5.0e-3    /* diff_sh_limit (namelist.tra:97) */
/* double diffusion (ddmix) — CORE2 has double_diffusion=.false. (namelist.tra:105),
 * so the IF(double_diffusion) CALL ddmix gate (driver :420-422) is never taken.
 * Port-what-CORE2-uses: the ddmix body (:1012-1085) is DEFERRED. If a config ever
 * sets this true, port ddmix first — the #error below makes that explicit. */
#define KPP_DOUBLE_DIFFUSION 0
/* in-routine constants used in later phases (K3/K5/K7) */
#define KPP_RIINFTY      0.8        /* ri_iwmix shear-Ri shape limit (:737) */
#define KPP_MINMIX       3.0e-3     /* surface viscAE floor (driver :408)   */
#define KPP_CEKMAN       0.7        /* bldepth Ekman limit (:478)           */
#define KPP_CMONOB       1.0        /* bldepth Monin-Obukhov limit (:479)   */

/* wm/ws lookup-table index: Fortran wmt(0:nni+1, 0:nnj+1), i in [0,nni+1],
 * j in [0,nnj+1]. Row-major with i as the row (nnj+2 columns). */
#define KPP_TBL(i, j)  ((size_t)(i) * (size_t)(FESOM_KPP_NNJ + 2) + (size_t)(j))

/* per-column stack-scratch cap (matches NL_MAX in fesom_gm/eos/momentum). */
enum { KPP_NL_MAX = FESOM_MAX_LEVELS };

/* fesom_kpp_mixing call index (= ocean step); mirrors the Fortran module's
 * kpp_call_count. Used for dump-step gating. */
static int s_kpp_call = 0;

/*===========================================================================
 * Allocation — mirrors oce_mixing_kpp_init's allocate block
 * (oce_ale_mixing_kpp.F90:128-156), sizing every array exactly as Fortran
 * (myDim+eDim nodes; ghats on nl-1, the rest on nl). All zeroed (calloc),
 * matching the Fortran init loop that sets every element to 0.
 *===========================================================================*/
void fesom_kpp_alloc(fesom_kpp *k, const struct fesom_mesh *mesh, int num_tracers)
{
    // Field members (DualView) make a raw memset UB (L13); value-initialise instead (D13).
    *k = fesom_kpp{};
    int N  = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int nl = mesh->nl;
    k->n_nod       = N;
    k->nl          = nl;
    k->num_tracers = num_tracers;

    size_t n_nl   = (size_t)N * (size_t)nl;
    size_t n_nl1  = (size_t)N * (size_t)(nl - 1);
    size_t tbl    = (size_t)(FESOM_KPP_NNI + 2) * (size_t)(FESOM_KPP_NNJ + 2);

    // M2.3: Field owns storage; raw ptr = non-owning alias = field.h() set once (D12). .alloc count
    // is in ELEMENTS (== the old calloc count) and zero-inits like calloc → Serial bit-identical.
    k->diffK_fld.alloc("kpp.diffK", (size_t)num_tracers * n_nl); k->diffK  = k->diffK_fld.h();
    k->viscA_fld.alloc("kpp.viscA", n_nl);                       k->viscA  = k->viscA_fld.h();
    k->blmc_fld.alloc("kpp.blmc", (size_t)3 * n_nl);             k->blmc   = k->blmc_fld.h();
    k->ghats_fld.alloc("kpp.ghats", n_nl1);                      k->ghats  = k->ghats_fld.h();
    k->dVsq_fld.alloc("kpp.dVsq", n_nl);                         k->dVsq   = k->dVsq_fld.h();

    k->dkm1_fld.alloc("kpp.dkm1", (size_t)3 * (size_t)N);        k->dkm1   = k->dkm1_fld.h();
    k->hbl_fld.alloc("kpp.hbl", (size_t)N);                      k->hbl    = k->hbl_fld.h();
    k->bfsfc_fld.alloc("kpp.bfsfc", (size_t)N);                  k->bfsfc  = k->bfsfc_fld.h();
    k->caseA_fld.alloc("kpp.caseA", (size_t)N);                  k->caseA  = k->caseA_fld.h();
    k->stable_fld.alloc("kpp.stable", (size_t)N);                k->stable = k->stable_fld.h();
    k->ustar_fld.alloc("kpp.ustar", (size_t)N);                  k->ustar  = k->ustar_fld.h();
    k->Bo_fld.alloc("kpp.Bo", (size_t)N);                        k->Bo     = k->Bo_fld.h();
    k->kbl_fld.alloc("kpp.kbl", (size_t)N);                      k->kbl    = k->kbl_fld.h();

    k->wmt_fld.alloc("kpp.wmt", tbl);                            k->wmt    = k->wmt_fld.h();
    k->wst_fld.alloc("kpp.wst", tbl);                            k->wst    = k->wst_fld.h();
}

void fesom_kpp_free(fesom_kpp *k)
{
    // *k = fesom_kpp{} releases every Field (assigns an empty DualView → drops the refcount/frees)
    // and zeros the PODs (D13/L13); no per-array free (the raw ptrs are non-owning aliases).
    *k = fesom_kpp{};
}

/*===========================================================================
 * One-time init — Vtc, cg, deltaz/deltau, wm/ws lookup tables.
 * Literal port of oce_mixing_kpp_init (:111-220), constant block + table build.
 * (The Fortran allocate+zero is handled by fesom_kpp_alloc's calloc.)
 *===========================================================================*/
void fesom_kpp_init(fesom_kpp *k)
{
    /* table-build parameters (oce_ale_mixing_kpp.F90:115-123) */
    const real_t cstar = 10.0,    conam = 1.257, concm = 8.380;
    const real_t conc2 = 16.0,    zetam = -0.2;
    const real_t conas = -28.86,  concs = 98.96, conc3 = 16.0, zetas = -1.0;

    /* Vtc (eqn 23, :177): concv * sqrt(0.2/concs/epsilon_kpp) / vonk^2 / Ricr */
    k->Vtc = KPP_CONCV * sqrt(0.2 / concs / KPP_EPSILON)
             / (KPP_VONK * KPP_VONK) / KPP_RICR;

    /* cg (eqn 20, :188): cstar * vonk * (concs * vonk * epsilon_kpp)^(1/3) */
    k->cg = cstar * KPP_VONK * pow(concs * KPP_VONK * KPP_EPSILON, 1.0 / 3.0);

    /* table steps (:194-195): real(nni+1)/real(nnj+1) divisor */
    k->deltaz = (KPP_ZMAX - KPP_ZMIN) / (real_t)(FESOM_KPP_NNI + 1);
    k->deltau = (KPP_UMAX - KPP_UMIN) / (real_t)(FESOM_KPP_NNJ + 1);

    /* wm/ws tables (eqn 13 & B1, :197-219) */
    for (int i = 0; i <= FESOM_KPP_NNI + 1; ++i) {
        real_t zehat = k->deltaz * (real_t)i + KPP_ZMIN;
        for (int j = 0; j <= FESOM_KPP_NNJ + 1; ++j) {
            real_t usta = k->deltau * (real_t)j + KPP_UMIN;
            real_t zeta = zehat / (usta * usta * usta + KPP_EPSLN);
            real_t wm, ws;
            if (zehat >= 0.0) {
                wm = KPP_VONK * usta / (1.0 + KPP_CONC1 * zeta);
                ws = wm;
            } else {
                if (zeta > zetam)
                    wm = KPP_VONK * usta * pow(1.0 - conc2 * zeta, 1.0 / 4.0);
                else
                    wm = KPP_VONK * pow(conam * usta*usta*usta - concm * zehat,
                                        1.0 / 3.0);
                if (zeta > zetas)
                    ws = KPP_VONK * usta * pow(1.0 - conc3 * zeta, 1.0 / 2.0);
                else
                    ws = KPP_VONK * pow(conas * usta*usta*usta - concs * zehat,
                                        1.0 / 3.0);
            }
            k->wmt[KPP_TBL(i, j)] = wm;
            k->wst[KPP_TBL(i, j)] = ws;
        }
    }

    /* M2.3b: wmt/wst are set-once (never change after this) — push them to the device
     * ONCE here (modify_host()+sync_device(), the mesh_sync_geometry_device pattern, L14),
     * so the device bldepth_kk/blmix_kk lookups read them device-current for free. Kokkos
     * is initialised by now (fesom_main.cpp: Kokkos::initialize precedes fesom_kpp_init).
     * No-op on Serial/OpenMP (host==device); one bitwise deep_copy on CUDA. */
    k->wmt_fld.modify_host(); k->wmt_fld.sync_device();
    k->wst_fld.modify_host(); k->wst_fld.sync_device();
}

/*===========================================================================
 * wscale — turbulent velocity scales wm, ws from the 2D lookup table.
 * Literal port of wscale (oce_ale_mixing_kpp.F90:828-877). Lookup for
 * zehat <= zmax (unstable); stable formula otherwise. Private (static).
 *===========================================================================*/
static void kpp_wscale(const fesom_kpp *k, real_t zehat, real_t us,
                       real_t *wm, real_t *ws)
{
    if (zehat <= KPP_ZMAX) {
        real_t zdiff = zehat - KPP_ZMIN;
        int iz = (int)(zdiff / k->deltaz);          /* INT (trunc) */
        if (iz > FESOM_KPP_NNI) iz = FESOM_KPP_NNI;
        if (iz < 0)             iz = 0;
        int izp1 = iz + 1;

        real_t udiff = us - KPP_UMIN;
        real_t uq = udiff / k->deltau;               /* MIN(.,nnj) then INT */
        if (uq > (real_t)FESOM_KPP_NNJ) uq = (real_t)FESOM_KPP_NNJ;
        int ju = (int)uq;
        if (ju < 0) ju = 0;
        int jup1 = ju + 1;

        real_t zfrac = zdiff / k->deltaz - (real_t)iz;
        real_t ufrac = udiff / k->deltau - (real_t)ju;
        real_t fzfrac = 1.0 - zfrac;

        real_t wam = fzfrac * k->wmt[KPP_TBL(iz, jup1)]
                   + zfrac  * k->wmt[KPP_TBL(izp1, jup1)];
        real_t wbm = fzfrac * k->wmt[KPP_TBL(iz, ju)]
                   + zfrac  * k->wmt[KPP_TBL(izp1, ju)];
        *wm = (1.0 - ufrac) * wbm + ufrac * wam;

        real_t was = fzfrac * k->wst[KPP_TBL(iz, jup1)]
                   + zfrac  * k->wst[KPP_TBL(izp1, jup1)];
        real_t wbs = fzfrac * k->wst[KPP_TBL(iz, ju)]
                   + zfrac  * k->wst[KPP_TBL(izp1, ju)];
        *ws = (1.0 - ufrac) * wbs + ufrac * was;
    } else {
        real_t u3 = us * us * us;
        *wm = KPP_VONK * us * u3 / (u3 + KPP_CONC1 * zehat + KPP_EPSLN);
        *ws = *wm;
    }
}

/*--- wscale — DEVICE inline twin (M2.3b) -------------------------------------
 * KOKKOS_INLINE_FUNCTION callable from device parallel_for (bldepth/blmix). The
 * wmt/wst lookup tables come in as device Views (templated WV) and the table-step
 * scalars deltaz/deltau by value; the body is a verbatim copy of kpp_wscale above
 * (`*wm/*ws` → `wm/ws` refs, `k->wmt[KPP_TBL(...)]` → wmt(KPP_TBL(...))). Pure
 * lookup + arithmetic (no transcendentals), so Serial == the C twin bit-identical.
 */
template <class WV>
KOKKOS_INLINE_FUNCTION
void kpp_wscale_kk(const WV &wmt, const WV &wst, real_t deltaz, real_t deltau,
                   real_t zehat, real_t us, real_t &wm, real_t &ws)
{
    if (zehat <= KPP_ZMAX) {
        real_t zdiff = zehat - KPP_ZMIN;
        int iz = (int)(zdiff / deltaz);             /* INT (trunc) */
        if (iz > FESOM_KPP_NNI) iz = FESOM_KPP_NNI;
        if (iz < 0)             iz = 0;
        int izp1 = iz + 1;

        real_t udiff = us - KPP_UMIN;
        real_t uq = udiff / deltau;                 /* MIN(.,nnj) then INT */
        if (uq > (real_t)FESOM_KPP_NNJ) uq = (real_t)FESOM_KPP_NNJ;
        int ju = (int)uq;
        if (ju < 0) ju = 0;
        int jup1 = ju + 1;

        real_t zfrac = zdiff / deltaz - (real_t)iz;
        real_t ufrac = udiff / deltau - (real_t)ju;
        real_t fzfrac = 1.0 - zfrac;

        real_t wam = fzfrac * wmt(KPP_TBL(iz, jup1))
                   + zfrac  * wmt(KPP_TBL(izp1, jup1));
        real_t wbm = fzfrac * wmt(KPP_TBL(iz, ju))
                   + zfrac  * wmt(KPP_TBL(izp1, ju));
        wm = (1.0 - ufrac) * wbm + ufrac * wam;

        real_t was = fzfrac * wst(KPP_TBL(iz, jup1))
                   + zfrac  * wst(KPP_TBL(izp1, jup1));
        real_t wbs = fzfrac * wst(KPP_TBL(iz, ju))
                   + zfrac  * wst(KPP_TBL(izp1, ju));
        ws = (1.0 - ufrac) * wbs + ufrac * was;
    } else {
        real_t u3 = us * us * us;
        wm = KPP_VONK * us * u3 / (u3 + KPP_CONC1 * zehat + KPP_EPSLN);
        ws = wm;
    }
}

/*===========================================================================
 * ri_iwmix — interior viscosity/diffusivity: shear instability (local Ri) +
 * constant background (A_ver/K_ver) + static instability. Literal port of
 * ri_iwmix (oce_ale_mixing_kpp.F90:890-999). Loops myDim only (the driver
 * exchanges viscA/diffK afterward). Ri is stored in diffK ch0 as scratch.
 * smooth_Ri_ver / smooth_Ri_hor are .false. (CORE2) → omitted (no-op gates).
 *===========================================================================*/
static void kpp_ri_iwmix(fesom_kpp *k, const struct fesom_aux *aux,
                         const struct fesom_dyn *dyn, const struct fesom_mesh *mesh)
{
    const int nl   = k->nl;
    const int Nmy  = mesh->myDim_nod2D;
    const size_t slab = (size_t)k->n_nod * (size_t)nl;
    real_t *viscA  = k->viscA;
    real_t *diffKt = k->diffK + 0 * slab;   /* channel 0 = T (diffK(:,:,1)) */
    real_t *diffKs = k->diffK + 1 * slab;   /* channel 1 = S (diffK(:,:,2)) */
    const real_t A_bg = (real_t)FESOM_PHASE1_A_VER;
    const real_t K_bg = (real_t)FESOM_PHASE1_K_VER;

    /* Loop 1: Richardson number Ri = MAX(N²,0)/(shear²+epsln), stored in
     * diffKt as scratch (Fortran :917-946). shear at Z (mid-layer). */
    for (int n = 0; n < Nmy; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            /* M6.3 (Z7): the C reads Z_3d_n, LIVE under zstar (fesom_kpp.c:238). */
            real_t dz_inv = 1.0 / (mesh->Z_3d_n[FESOM_NODE3D(n, nz - 1, nl)]
                                 - mesh->Z_3d_n[FESOM_NODE3D(n, nz,     nl)]);   /* > 0 */
            real_t du = dyn->uvnode[FESOM_ELEMVEC(n, nz - 1, nl) + 0]
                      - dyn->uvnode[FESOM_ELEMVEC(n, nz,     nl) + 0];
            real_t dv = dyn->uvnode[FESOM_ELEMVEC(n, nz - 1, nl) + 1]
                      - dyn->uvnode[FESOM_ELEMVEC(n, nz,     nl) + 1];
            real_t shear = (du * du + dv * dv) * dz_inv * dz_inv;
            real_t Nsq = aux->bvfreq[FESOM_NODE3D(n, nz, nl)];
            real_t Nsq_pos = Nsq > 0.0 ? Nsq : 0.0;
            diffKt[FESOM_NODE3D(n, nz, nl)] = Nsq_pos / (shear + KPP_EPSLN);
        }
        /* edge copies of the Ri scratch (:932-933) */
        diffKt[FESOM_NODE3D(n, nzmin, nl)] = diffKt[FESOM_NODE3D(n, nzmin + 1, nl)];
        diffKt[FESOM_NODE3D(n, nzmax, nl)] = diffKt[FESOM_NODE3D(n, nzmax - 1, nl)];
    }

    /* Loop 2: viscA / diffK from the shear-Ri shape function (:954-997). */
    for (int n = 0; n < Nmy; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {
            size_t i = FESOM_NODE3D(n, nz, nl);
            real_t Rigg = diffKt[i] > 0.0 ? diffKt[i] : 0.0;        /* AMAX1(Ri,0) */
            real_t ratio = Rigg / KPP_RIINFTY;
            if (ratio > 1.0) ratio = 1.0;                          /* AMIN1(.,1) */
            real_t frit = 1.0 - ratio * ratio;
            frit = frit * frit * frit;
            viscA[i]  = KPP_VISC_SH_LIMIT * frit + A_bg;
            real_t dK = KPP_DIFF_SH_LIMIT * frit + K_bg;           /* Kv0_const branch */
            diffKt[i] = dK;
            diffKs[i] = dK;                                        /* diffK(2)=diffK(1) */
        }
        /* edge copies (:989-995) */
        size_t a0 = FESOM_NODE3D(n, nzmin, nl),     a1 = FESOM_NODE3D(n, nzmin + 1, nl);
        size_t b0 = FESOM_NODE3D(n, nzmax, nl),     b1 = FESOM_NODE3D(n, nzmax - 1, nl);
        viscA[a0]  = viscA[a1];  diffKt[a0] = diffKt[a1]; diffKs[a0] = diffKs[a1];
        viscA[b0]  = viscA[b1];  diffKt[b0] = diffKt[b1]; diffKs[b0] = diffKs[b1];
    }
}

/*--- ri_iwmix — DEVICE twin (M2.3b) ------------------------------------------
 * Two parallel_for launches over owned nodes (the level loops + edge copies
 * inside each per-node lambda → race-free, the C accumulation order). The two
 * launches are load-bearing (D20): Loop 1 fills diffKt with the Richardson-number
 * SCRATCH (all nodes, incl. edge copies); Loop 2 reads that Ri scratch and
 * OVERWRITES diffKt with the shear-Ri diffusivity — fusing them or running Loop 2
 * before Loop 1 completes would feed Loop 2 corrupted Ri. Verbatim arithmetic.
 */
static bool m7_kpp_flat_on()
{
    static int c = -1;
    return fesom_speed_on("FLAT", &c);
}

static void kpp_ri_iwmix_kk(fesom_kpp *k, const struct fesom_aux *aux,
                            const struct fesom_dyn *dyn, const struct fesom_mesh *mesh)
{
    const int nl   = k->nl;
    const int Nmy  = mesh->myDim_nod2D;
    const size_t slab = (size_t)k->n_nod * (size_t)nl;
    const real_t A_bg = (real_t)FESOM_PHASE1_A_VER;
    const real_t K_bg = (real_t)FESOM_PHASE1_K_VER;

    auto viscA  = k->viscA_fld.d();
    auto diffK  = k->diffK_fld.d();
    auto bvfreq = aux->bvfreq_fld.d();
    auto uvnode = dyn->uvnode_fld.d();
    auto Z      = mesh->Z_fld.d();
    auto Z3d    = mesh->Z_3d_n_fld.d();   /* M6.3 (Z7): the C reads Z_3d_n (fesom_kpp.c:238) */
    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto nlev_n = mesh->nlevels_nod2D_fld.d();

    /* M7 Task A.1 — FESOM_SPEED_FLAT: one thread per (node, level) instead of per
     * node-with-an-inner-column-loop. Measured 17.0 ms/step at NG5@4N with SM 2.8%
     * and 23.6 sectors/request — the column loop makes every load uncoalesced.
     *
     * The two launches become FOUR: each legacy loop splits into its INTERIOR (a pure
     * per-(n,nz) map) and its EDGE COPIES (per node, still sequential inside one
     * lambda). Order 1→2→3→4 is load-bearing exactly as D20's 1→2 was; stream order
     * provides the dependency.
     *
     * BIT-IDENTICAL: (a) every interior slot is written once, from operands no other
     * slot writes, so slot order is irrelevant; (b) a node's edge copies only ever
     * touch that node's own slots, and in the legacy code they already ran after that
     * node's interior loop and concurrently with every OTHER node's — so hoisting all
     * interiors ahead of all edges changes nothing; (c) the edge lambdas keep the six
     * assignments in the legacy ORDER, which is what makes the degenerate
     * (nzmax == nzmin+1, empty interior) column reproduce its stale-carry chain
     * verbatim — the interior kernel never touched those slots either. */
    if (m7_kpp_flat_on()) {
        const size_t tot = (size_t)Nmy * (size_t)nl;

        /* 1: Ri interior — legacy loop-1 body, verbatim. */
        Kokkos::parallel_for("kpp_ri_iwmix_Ri_flat", tot, KOKKOS_LAMBDA(const size_t i){
            const int n  = (int)(i / (size_t)nl);
            const int nz = (int)(i % (size_t)nl);
            const int nzmin = ulev_n(n) - 1;
            const int nzmax = nlev_n(n) - 1;
            if (nz <= nzmin || nz >= nzmax) return;      /* legacy: nzmin+1 .. nzmax-1 */
            real_t dz_inv = 1.0 / (Z3d(FESOM_NODE3D(n, nz - 1, nl))
                                 - Z3d(FESOM_NODE3D(n, nz,     nl)));   /* > 0 */
            real_t du = uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 0)
                      - uvnode(FESOM_ELEMVEC(n, nz,     nl) + 0);
            real_t dv = uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 1)
                      - uvnode(FESOM_ELEMVEC(n, nz,     nl) + 1);
            real_t shear = (du * du + dv * dv) * dz_inv * dz_inv;
            real_t Nsq = bvfreq(i);                      /* FESOM_NODE3D(n,nz,nl) == i */
            real_t Nsq_pos = Nsq > 0.0 ? Nsq : 0.0;
            diffK(0 * slab + i) = Nsq_pos / (shear + KPP_EPSLN);
        });

        /* 2: Ri edge copies — legacy loop-1 tail, verbatim, after the FULL interior. */
        Kokkos::parallel_for("kpp_ri_iwmix_Ri_edge", Kokkos::RangePolicy<>(0, Nmy),
            KOKKOS_LAMBDA(const int n){
                int nzmin = ulev_n(n) - 1;
                int nzmax = nlev_n(n) - 1;
                diffK(0 * slab + FESOM_NODE3D(n, nzmin, nl)) =
                    diffK(0 * slab + FESOM_NODE3D(n, nzmin + 1, nl));
                diffK(0 * slab + FESOM_NODE3D(n, nzmax, nl)) =
                    diffK(0 * slab + FESOM_NODE3D(n, nzmax - 1, nl));
            });

        /* 3: shape interior — legacy loop-2 body, verbatim. */
        Kokkos::parallel_for("kpp_ri_iwmix_shape_flat", tot, KOKKOS_LAMBDA(const size_t i){
            const int n  = (int)(i / (size_t)nl);
            const int nz = (int)(i % (size_t)nl);
            const int nzmin = ulev_n(n) - 1;
            const int nzmax = nlev_n(n) - 1;
            if (nz <= nzmin || nz >= nzmax) return;
            real_t Rii = diffK(0 * slab + i);
            real_t Rigg = Rii > 0.0 ? Rii : 0.0;            /* AMAX1(Ri,0) */
            real_t ratio = Rigg / KPP_RIINFTY;
            if (ratio > 1.0) ratio = 1.0;                   /* AMIN1(.,1) */
            real_t frit = 1.0 - ratio * ratio;
            frit = frit * frit * frit;
            viscA(i) = KPP_VISC_SH_LIMIT * frit + A_bg;
            real_t dK = KPP_DIFF_SH_LIMIT * frit + K_bg;    /* Kv0_const branch */
            diffK(0 * slab + i) = dK;
            diffK(1 * slab + i) = dK;                       /* diffK(2)=diffK(1) */
        });

        /* 4: shape edge copies — legacy loop-2 tail, verbatim, after the FULL interior. */
        Kokkos::parallel_for("kpp_ri_iwmix_shape_edge", Kokkos::RangePolicy<>(0, Nmy),
            KOKKOS_LAMBDA(const int n){
                int nzmin = ulev_n(n) - 1;
                int nzmax = nlev_n(n) - 1;
                size_t a0 = FESOM_NODE3D(n, nzmin, nl),     a1 = FESOM_NODE3D(n, nzmin + 1, nl);
                size_t b0 = FESOM_NODE3D(n, nzmax, nl),     b1 = FESOM_NODE3D(n, nzmax - 1, nl);
                viscA(a0) = viscA(a1);
                diffK(0*slab + a0) = diffK(0*slab + a1); diffK(1*slab + a0) = diffK(1*slab + a1);
                viscA(b0) = viscA(b1);
                diffK(0*slab + b0) = diffK(0*slab + b1); diffK(1*slab + b0) = diffK(1*slab + b1);
            });
        return;
    }

    /* Loop 1: Ri = MAX(N²,0)/(shear²+epsln) → diffKt (channel 0) scratch. */
    Kokkos::parallel_for("kpp_ri_iwmix_Ri", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t dz_inv = 1.0 / (Z3d(FESOM_NODE3D(n, nz - 1, nl))
                                     - Z3d(FESOM_NODE3D(n, nz,     nl)));   /* > 0 */
                real_t du = uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 0)
                          - uvnode(FESOM_ELEMVEC(n, nz,     nl) + 0);
                real_t dv = uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 1)
                          - uvnode(FESOM_ELEMVEC(n, nz,     nl) + 1);
                real_t shear = (du * du + dv * dv) * dz_inv * dz_inv;
                real_t Nsq = bvfreq(FESOM_NODE3D(n, nz, nl));
                real_t Nsq_pos = Nsq > 0.0 ? Nsq : 0.0;
                diffK(0 * slab + FESOM_NODE3D(n, nz, nl)) = Nsq_pos / (shear + KPP_EPSLN);
            }
            diffK(0 * slab + FESOM_NODE3D(n, nzmin, nl)) =
                diffK(0 * slab + FESOM_NODE3D(n, nzmin + 1, nl));
            diffK(0 * slab + FESOM_NODE3D(n, nzmax, nl)) =
                diffK(0 * slab + FESOM_NODE3D(n, nzmax - 1, nl));
        });

    /* Loop 2: viscA / diffK from the shear-Ri shape function (reads Loop-1 Ri). */
    Kokkos::parallel_for("kpp_ri_iwmix_shape", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                size_t i = FESOM_NODE3D(n, nz, nl);
                real_t Rii = diffK(0 * slab + i);
                real_t Rigg = Rii > 0.0 ? Rii : 0.0;            /* AMAX1(Ri,0) */
                real_t ratio = Rigg / KPP_RIINFTY;
                if (ratio > 1.0) ratio = 1.0;                   /* AMIN1(.,1) */
                real_t frit = 1.0 - ratio * ratio;
                frit = frit * frit * frit;
                viscA(i) = KPP_VISC_SH_LIMIT * frit + A_bg;
                real_t dK = KPP_DIFF_SH_LIMIT * frit + K_bg;    /* Kv0_const branch */
                diffK(0 * slab + i) = dK;
                diffK(1 * slab + i) = dK;                       /* diffK(2)=diffK(1) */
            }
            size_t a0 = FESOM_NODE3D(n, nzmin, nl),     a1 = FESOM_NODE3D(n, nzmin + 1, nl);
            size_t b0 = FESOM_NODE3D(n, nzmax, nl),     b1 = FESOM_NODE3D(n, nzmax - 1, nl);
            viscA(a0) = viscA(a1);
            diffK(0*slab + a0) = diffK(0*slab + a1); diffK(1*slab + a0) = diffK(1*slab + a1);
            viscA(b0) = viscA(b1);
            diffK(0*slab + b0) = diffK(0*slab + b1); diffK(1*slab + b0) = diffK(1*slab + b1);
        });
}

/* generic per-node column getter for fesom_kpp_dump_nodes: arr is [N*nl]
 * node-major; comp == layer interface nz. */
typedef struct { const real_t *arr; int nl; } kpp_col_src;
static double kpp_get_col(int node, int comp, void *user)
{
    const kpp_col_src *s = (const kpp_col_src *)user;
    return (double)s->arr[(size_t)node * (size_t)s->nl + (size_t)comp];
}

/* K3 dump: raw ri_iwmix node outputs (viscA, diffK T/S) as full columns, plus
 * bvfreq (the input whose sign drives the step-1 background/static-instability
 * branch) so the validation can confirm any C-vs-Fortran viscA disagreement is
 * exactly a bvfreq sign flip (cross-code IC noise), not an algebra bug. */
static void kpp_dump_riiwmix(const fesom_kpp *k, const struct fesom_aux *aux,
                             const struct fesom_mesh *mesh, struct fesom_partit *partit)
{
    size_t slab = (size_t)k->n_nod * (size_t)k->nl;
    kpp_col_src sv = { k->viscA,           k->nl };
    kpp_col_src st = { k->diffK + 0 * slab, k->nl };
    kpp_col_src ss = { k->diffK + 1 * slab, k->nl };
    kpp_col_src sb = { aux->bvfreq,         k->nl };
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "ri_viscA",  k->nl, kpp_get_col, &sv);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "ri_diffKt", k->nl, kpp_get_col, &st);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "ri_diffKs", k->nl, kpp_get_col, &ss);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "ri_bvfreq", k->nl, kpp_get_col, &sb);
}

/*===========================================================================
 * bldepth — oceanic boundary-layer depth hbl + kbl, bfsfc/stable/caseA.
 * Literal port of bldepth (oce_ale_mixing_kpp.F90:674-854). HIGHEST RISK
 * (historical OBL bug, FRESH_START:489). Loops myDim only (Fortran :697/708/813
 * are myDim with the +eDim commented out). Uses zbar (interface depths, NOT Z),
 * the static kpp_wscale (K2), and the driver pre-step ustar/Bo/dVsq + the
 * EOS-filled dbsfc/bvfreq/sw_alpha + forcing sw_3d.
 *
 * CORE2 use_sw_pene = .true. (Key invariant #6) — the sw-penetration branches
 * are always taken, so they are ported inline (port-what-CORE2-uses); the
 * else-branch (bfsfc=Bo constant) is not reachable in CORE2. smooth_hbl=.false.
 * (skip). kbl is stored 0-based (the dump emits kbl+1 to match the Fortran's
 * 1-based real(kbl,WP)).
 *===========================================================================*/
static void kpp_bldepth(fesom_kpp *k, const struct fesom_aux *aux,
                        const struct fesom_forcing *forcing,
                        const struct fesom_mesh *mesh)
{
    const int nl  = k->nl;
    const int Nmy = mesh->myDim_nod2D;
    const real_t Vtc = k->Vtc;

    /* init hbl/kbl to bottomed-out values (:696-702) */
    for (int n = 0; n < Nmy; ++n) {
        int nzmax = mesh->nlevels_nod2D[n] - 1;          /* 0-based deepest interface */
        k->kbl[n] = nzmax;
        k->hbl[n] = fabs(mesh->zbar_3d_n[FESOM_NODE3D(n, nzmax, nl)]);
    }

    /* bulk-Richardson search for hbl (:708-802) */
    for (int n = 0; n < Nmy; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        real_t Bo = k->Bo[n];
        real_t us = k->ustar[n];
        real_t coeff_sw = (real_t)FESOM_G * aux->sw_alpha[FESOM_NODE3D(n, nzmin, nl)]; /* :713 */
        real_t sw_surf  = forcing->sw_3d[FESOM_NODE3D(n, nzmin, nl)];
        real_t Rib_km1  = 0.0;
        k->bfsfc[n] = Bo;                                /* :717 */

        for (int nz = nzmin + 1; nz <= nzmax; ++nz) {    /* :719 (nz=nzmin+1..nzmax) */
            real_t zk   = fabs(mesh->zbar_3d_n[FESOM_NODE3D(n, nz,     nl)]);
            real_t zkm1 = fabs(mesh->zbar_3d_n[FESOM_NODE3D(n, nz - 1, nl)]);

            /* bfsfc = Bo + sw contribution (:725-727) */
            k->bfsfc[n] = Bo + coeff_sw * (sw_surf - forcing->sw_3d[FESOM_NODE3D(n, nz, nl)]);

            real_t bfs    = k->bfsfc[n];
            real_t stable = 0.5 + copysign(0.5, bfs);                  /* :729 */
            k->stable[n]  = stable;
            real_t sigma  = stable + (1.0 - stable) * KPP_EPSILON;     /* :730 */

            real_t zehat = KPP_VONK * sigma * zk * bfs;                /* :736 */
            real_t wm, ws;
            kpp_wscale(k, zehat, us, &wm, &ws);                        /* :737 */

            real_t bvsq = aux->bvfreq[FESOM_NODE3D(n, nz, nl)];        /* :747 */
            real_t Vtsq = zk * ws * sqrt(fabs(bvsq)) * Vtc;            /* :750 */

            real_t Ritop = zk * aux->dbsfc[FESOM_NODE3D(n, nz, nl)];   /* :757 */
            real_t Rib_k = Ritop / (k->dVsq[FESOM_NODE3D(n, nz, nl)] + Vtsq + KPP_EPSLN); /* :758 */
            real_t dzup  = zk - zkm1;                                  /* :759 */

            if (Rib_k > KPP_RICR) {                                    /* :761 */
                k->hbl[n] = zkm1 + dzup * (KPP_RICR - Rib_km1)
                                       / (Rib_k - Rib_km1 + KPP_EPSLN);/* :763 */
                k->kbl[n] = nz;                                        /* :764 */
                break;                                                 /* :765 (EXIT) */
            } else {
                Rib_km1 = Rib_k;                                       /* :767 */
            }

            /* linear interp of sw_3d to hbl, refine bfsfc (:774-785) */
            {
                real_t sw_km1 = forcing->sw_3d[FESOM_NODE3D(n, nz - 1, nl)];
                real_t sw_k   = forcing->sw_3d[FESOM_NODE3D(n, nz,     nl)];
                k->bfsfc[n] = Bo + coeff_sw *
                    ( sw_surf - ( sw_km1 + (sw_k - sw_km1) * (k->hbl[n] - zkm1) / dzup ) );
                real_t st = 0.5 + copysign(0.5, k->bfsfc[n]);          /* :783 */
                k->stable[n] = st;
                k->bfsfc[n]  = k->bfsfc[n] + st * KPP_EPSLN;           /* :784 */
            }
        }

        /* hekman / hmonob limits, gated bfsfc>0 .and. nzmin==1 (:792-801).
         * Fortran nzmin==1 (no cavity) → C nzmin==0. */
        if (k->bfsfc[n] > 0.0 && nzmin == 0) {
            real_t hekman = KPP_CEKMAN * us / fmax(fabs(mesh->coriolis_node[n]), KPP_EPSLN); /* :795 */
            real_t hmonob = KPP_CMONOB * us * us * us
                            / KPP_VONK / (k->bfsfc[n] + KPP_EPSLN);    /* :796-797 */
            real_t hlimit = k->stable[n] * fmin(hekman, hmonob);       /* :798 */
            k->hbl[n] = fmin(k->hbl[n], hlimit);                       /* :799 */
            k->hbl[n] = fmax(k->hbl[n],
                             fabs(mesh->zbar_3d_n[FESOM_NODE3D(n, 1, nl)])); /* :800 (zbar(2)) */
        }
    }

    /* smooth_hbl = .false. → skip (:806-809) */

    /* find new kbl from final hbl, refine bfsfc, set caseA (:812-852) */
    for (int n = 0; n < Nmy; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;

        k->kbl[n] = nzmax;                                            /* :820 */
        for (int nz = nzmin + 1; nz <= nzmax; ++nz) {                 /* :822 */
            if (fabs(mesh->zbar_3d_n[FESOM_NODE3D(n, nz, nl)]) > k->hbl[n]) {
                k->kbl[n] = nz;                                       /* :824 */
                break;                                               /* :825 (EXIT) */
            }
        }

        /* final bfsfc: linear interp of sw_3d to hbl using kbl (:832-844) */
        int kbl = k->kbl[n];
        real_t coeff_sw = (real_t)FESOM_G * aux->sw_alpha[FESOM_NODE3D(n, nzmin, nl)];
        real_t sw_surf  = forcing->sw_3d[FESOM_NODE3D(n, nzmin,  nl)];
        real_t sw_km1   = forcing->sw_3d[FESOM_NODE3D(n, kbl - 1, nl)];
        real_t sw_k     = forcing->sw_3d[FESOM_NODE3D(n, kbl,     nl)];
        real_t zbar_km1 = mesh->zbar_3d_n[FESOM_NODE3D(n, kbl - 1, nl)];
        real_t zbar_k   = mesh->zbar_3d_n[FESOM_NODE3D(n, kbl,     nl)];
        k->bfsfc[n] = k->Bo[n] + coeff_sw *
            ( sw_surf - ( sw_km1 + (sw_k - sw_km1)
                                   * (k->hbl[n] + zbar_km1) / (zbar_km1 - zbar_k) ) );
        real_t st = 0.5 + copysign(0.5, k->bfsfc[n]);                 /* :842 */
        k->stable[n] = st;
        k->bfsfc[n]  = k->bfsfc[n] + st * KPP_EPSLN;                  /* :843 */

        /* caseA: =1 if hbl above mid-point of level kbl, else 0 (:850-851) */
        real_t dzup = zbar_km1 - zbar_k;
        real_t arg  = fabs(zbar_k) - 0.5 * dzup - k->hbl[n];
        k->caseA[n] = 0.5 + copysign(0.5, arg);
    }
}

/*--- bldepth — DEVICE twin (M2.3b) -------------------------------------------
 * Three parallel_for launches over owned nodes (D20 ordering): Loop 1 inits
 * hbl/kbl to the bottomed-out fallback (which Loop 2 keeps if its bulk-Ri search
 * never crosses Ricr); Loop 2 is the bulk-Ri search (per-node inner nz loop with
 * an early `break` — fine in a per-node lambda — + the hekman/hmonob limits);
 * Loop 3 re-finds kbl from the final hbl + the final bfsfc/stable/caseA (reads
 * Loop-2's hbl). Each node owns its hbl/kbl/bfsfc/stable/caseA slots → race-free.
 * wscale via the device inline twin (wmt/wst views). Verbatim arithmetic.
 */
static void kpp_bldepth_kk(fesom_kpp *k, const struct fesom_aux *aux,
                           const struct fesom_forcing *forcing,
                           const struct fesom_mesh *mesh)
{
    const int nl  = k->nl;
    const int Nmy = mesh->myDim_nod2D;
    const real_t Vtc = k->Vtc, deltaz = k->deltaz, deltau = k->deltau;
    const real_t G = (real_t)FESOM_G;

    auto hbl     = k->hbl_fld.d();
    auto kbl     = k->kbl_fld.d();
    auto bfsfc   = k->bfsfc_fld.d();
    auto stableV = k->stable_fld.d();
    auto caseA   = k->caseA_fld.d();
    auto Bo      = k->Bo_fld.d();
    auto ustar   = k->ustar_fld.d();
    auto dVsq    = k->dVsq_fld.d();
    auto wmt     = k->wmt_fld.d();
    auto wst     = k->wst_fld.d();
    auto sw_alpha = aux->sw_alpha_fld.d();
    auto dbsfc    = aux->dbsfc_fld.d();
    auto bvfreq   = aux->bvfreq_fld.d();
    auto sw_3d    = forcing->sw_3d_fld.d();
    auto zbar3    = mesh->zbar_3d_n_fld.d();
    auto coriolis_node = mesh->coriolis_node_fld.d();
    auto ulev_n  = mesh->ulevels_nod2D_fld.d();
    auto nlev_n  = mesh->nlevels_nod2D_fld.d();

    /* Loop 1: init hbl/kbl to bottomed-out values (:696-702) */
    Kokkos::parallel_for("kpp_bldepth_init", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmax = nlev_n(n) - 1;
            kbl(n) = nzmax;
            hbl(n) = Kokkos::fabs(zbar3(FESOM_NODE3D(n, nzmax, nl)));
        });

    /* Loop 2: bulk-Richardson search for hbl (:708-802) */
    Kokkos::parallel_for("kpp_bldepth_search", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            real_t Bo_n = Bo(n);
            real_t us   = ustar(n);
            real_t coeff_sw = G * sw_alpha(FESOM_NODE3D(n, nzmin, nl));
            real_t sw_surf  = sw_3d(FESOM_NODE3D(n, nzmin, nl));
            real_t Rib_km1  = 0.0;
            bfsfc(n) = Bo_n;

            for (int nz = nzmin + 1; nz <= nzmax; ++nz) {
                real_t zk   = Kokkos::fabs(zbar3(FESOM_NODE3D(n, nz,     nl)));
                real_t zkm1 = Kokkos::fabs(zbar3(FESOM_NODE3D(n, nz - 1, nl)));

                bfsfc(n) = Bo_n + coeff_sw * (sw_surf - sw_3d(FESOM_NODE3D(n, nz, nl)));
                real_t bfs    = bfsfc(n);
                real_t stbl   = 0.5 + Kokkos::copysign(0.5, bfs);
                stableV(n)    = stbl;
                real_t sigma  = stbl + (1.0 - stbl) * KPP_EPSILON;

                real_t zehat = KPP_VONK * sigma * zk * bfs;
                real_t wm, ws;
                kpp_wscale_kk(wmt, wst, deltaz, deltau, zehat, us, wm, ws);

                real_t bvsq = bvfreq(FESOM_NODE3D(n, nz, nl));
                real_t Vtsq = zk * ws * Kokkos::sqrt(Kokkos::fabs(bvsq)) * Vtc;

                real_t Ritop = zk * dbsfc(FESOM_NODE3D(n, nz, nl));
                real_t Rib_k = Ritop / (dVsq(FESOM_NODE3D(n, nz, nl)) + Vtsq + KPP_EPSLN);
                real_t dzup  = zk - zkm1;

                if (Rib_k > KPP_RICR) {
                    hbl(n) = zkm1 + dzup * (KPP_RICR - Rib_km1)
                                        / (Rib_k - Rib_km1 + KPP_EPSLN);
                    kbl(n) = nz;
                    break;
                } else {
                    Rib_km1 = Rib_k;
                }

                {
                    real_t sw_km1 = sw_3d(FESOM_NODE3D(n, nz - 1, nl));
                    real_t sw_k   = sw_3d(FESOM_NODE3D(n, nz,     nl));
                    bfsfc(n) = Bo_n + coeff_sw *
                        ( sw_surf - ( sw_km1 + (sw_k - sw_km1) * (hbl(n) - zkm1) / dzup ) );
                    real_t st = 0.5 + Kokkos::copysign(0.5, bfsfc(n));
                    stableV(n) = st;
                    bfsfc(n)   = bfsfc(n) + st * KPP_EPSLN;
                }
            }

            if (bfsfc(n) > 0.0 && nzmin == 0) {
                real_t hekman = KPP_CEKMAN * us / Kokkos::fmax(Kokkos::fabs(coriolis_node(n)), KPP_EPSLN);
                real_t hmonob = KPP_CMONOB * us * us * us
                                / KPP_VONK / (bfsfc(n) + KPP_EPSLN);
                real_t hlimit = stableV(n) * Kokkos::fmin(hekman, hmonob);
                hbl(n) = Kokkos::fmin(hbl(n), hlimit);
                hbl(n) = Kokkos::fmax(hbl(n), Kokkos::fabs(zbar3(FESOM_NODE3D(n, 1, nl))));
            }
        });

    /* Loop 3: find new kbl from final hbl, refine bfsfc, set caseA (:812-852) */
    Kokkos::parallel_for("kpp_bldepth_kbl", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;

            kbl(n) = nzmax;
            for (int nz = nzmin + 1; nz <= nzmax; ++nz) {
                if (Kokkos::fabs(zbar3(FESOM_NODE3D(n, nz, nl))) > hbl(n)) {
                    kbl(n) = nz;
                    break;
                }
            }

            int kb = kbl(n);
            real_t coeff_sw = G * sw_alpha(FESOM_NODE3D(n, nzmin, nl));
            real_t sw_surf  = sw_3d(FESOM_NODE3D(n, nzmin,  nl));
            real_t sw_km1   = sw_3d(FESOM_NODE3D(n, kb - 1, nl));
            real_t sw_k     = sw_3d(FESOM_NODE3D(n, kb,     nl));
            real_t zbar_km1 = zbar3(FESOM_NODE3D(n, kb - 1, nl));
            real_t zbar_k   = zbar3(FESOM_NODE3D(n, kb,     nl));
            bfsfc(n) = Bo(n) + coeff_sw *
                ( sw_surf - ( sw_km1 + (sw_k - sw_km1)
                                       * (hbl(n) + zbar_km1) / (zbar_km1 - zbar_k) ) );
            real_t st = 0.5 + Kokkos::copysign(0.5, bfsfc(n));
            stableV(n) = st;
            bfsfc(n)   = bfsfc(n) + st * KPP_EPSLN;

            real_t dzup = zbar_km1 - zbar_k;
            real_t arg  = Kokkos::fabs(zbar_k) - 0.5 * dzup - hbl(n);
            caseA(n) = 0.5 + Kokkos::copysign(0.5, arg);
        });
}

/*===========================================================================
 * blmix_kpp — boundary-layer mixing coeffs blmc[3] + dkm1[3] + ghats.
 * Literal port of blmix_kpp (oce_ale_mixing_kpp.F90:1155-1327). Matches the
 * interior diffusivities to the surface-layer scaling at hbl (eqns 10/11/18)
 * via the caseA-selected level kn, the shape function G(σ), and wscale.
 * Channels (Fortran 1/2/3 = momentum/T/S): blmc/diff_col/dkm1 C-slabs are
 * 0=momentum(viscA), 1=T(diffK ch1), 2=S(diffK ch2). NOTE the cross-wiring:
 * difsp/difsh/Gs (salinity) read diff_col ch3 (=C slab 2); diftp/difth/Gt
 * (temperature) read diff_col ch2 (=C slab 1). `ghats` is computed even though
 * CORE2 (use_kpp_nonlclflx=.false.) never consumes it. Loops myDim. Uses Z
 * (mid-layer) for the interface σ but zbar (interface) for the kbl-1 dkm1 σ.
 *===========================================================================*/
static void kpp_blmix(fesom_kpp *k, const struct fesom_mesh *mesh)
{
    const int nl  = k->nl;
    const int N   = k->n_nod;
    const int Nmy = mesh->myDim_nod2D;
    const size_t slab = (size_t)N * (size_t)nl;
    FESOM_CHECK(nl <= KPP_NL_MAX, "kpp_blmix: nl %d > KPP_NL_MAX", nl);

    real_t *viscA  = k->viscA;
    real_t *diffKt = k->diffK + 0 * slab;   /* T (diffK(:,:,1)) */
    real_t *diffKs = k->diffK + 1 * slab;   /* S (diffK(:,:,2)) */
    real_t *blmcM  = k->blmc  + 0 * slab;   /* blmc(:,:,1) momentum */
    real_t *blmcT  = k->blmc  + 1 * slab;   /* blmc(:,:,2) temperature */
    real_t *blmcS  = k->blmc  + 2 * slab;   /* blmc(:,:,3) salinity */

    /* zero blmc over myDim+eDim (:1177-1179) */
    memset(k->blmc, 0, (size_t)3 * slab * sizeof(real_t));

    for (int n = 0; n < Nmy; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;          /* 0-based deepest interface */
        /* temporary-solution skips (:1191-1192), in terms of the 1-based counts */
        if (mesh->nlevels_nod2D[n] < 3) continue;
        if (mesh->nlevels_nod2D[n] - mesh->ulevels_nod2D[n] < 2) continue;

        real_t hbl    = k->hbl[n];
        real_t bfsfc  = k->bfsfc[n];
        real_t stable = k->stable[n];
        real_t us     = k->ustar[n];
        int    kbl    = k->kbl[n];

        /* dthick (interface thicknesses, :1194-1196) + diff_col (:1198-1200) */
        real_t dthick[KPP_NL_MAX];
        real_t dcol[KPP_NL_MAX][3];
        for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz)
            dthick[nz] = 0.5 * ( mesh->hnode[FESOM_NODE3D(n, nz - 1, nl)]
                               + mesh->hnode[FESOM_NODE3D(n, nz,     nl)] );
        dthick[nzmin] = mesh->hnode[FESOM_NODE3D(n, nzmin,     nl)] * 0.5;
        dthick[nzmax] = mesh->hnode[FESOM_NODE3D(n, nzmax - 1, nl)] * 0.5;
        for (int nz = nzmin; nz <= nzmax - 1; ++nz) {
            size_t i = FESOM_NODE3D(n, nz, nl);
            dcol[nz][0] = viscA [i];
            dcol[nz][1] = diffKt[i];
            dcol[nz][2] = diffKs[i];
        }
        dcol[nzmax][0] = dcol[nzmax - 1][0];
        dcol[nzmax][1] = dcol[nzmax - 1][1];
        dcol[nzmax][2] = dcol[nzmax - 1][2];

        /* velocity scales at hbl (:1206-1208) */
        real_t sigma = stable * 1.0 + (1.0 - stable) * KPP_EPSILON;
        real_t zehat = KPP_VONK * sigma * hbl * bfsfc;
        real_t wm, ws;
        kpp_wscale(k, zehat, us, &wm, &ws);

        /* caseA-selected matching level kn (:1210-1214). caseA∈{0,1}:
         * INT(caseA+epsln) → kn = kbl-caseA, capped to the interior. */
        int ca   = (int)(k->caseA[n] + KPP_EPSLN);
        int kn   = ca * (kbl - 1) + (1 - ca) * kbl;
        if (kn   > nzmax - 1) kn   = nzmax - 1;          /* MIN(kn, nl1-1) */
        int knm1 = kn - 1; if (knm1 < nzmin) knm1 = nzmin;   /* MAX(kn-1, nu1) */
        int knp1 = kn + 1; if (knp1 > nzmax) knp1 = nzmax;   /* MIN(kn+1, nl1) */

        /* interior viscosities + one-sided derivatives at hbl (eqn 18, :1220-1242) */
        /* M6.3 (Z7): the C reads Z_3d_n, LIVE under zstar (fesom_kpp.c:516). */
        real_t delhat = fabs(mesh->Z_3d_n[FESOM_NODE3D(n, kn, nl)]) - hbl;
        real_t R      = 1.0 - delhat / dthick[kn];
        real_t dvdzup, dvdzdn;
        dvdzup = (dcol[knm1][0] - dcol[kn][0]) / dthick[kn];
        dvdzdn = (dcol[kn][0] - dcol[knp1][0]) / dthick[knp1];
        real_t viscp = 0.5 * ( (1.0 - R) * (dvdzup + fabs(dvdzup))
                             + R         * (dvdzdn + fabs(dvdzdn)) );
        dvdzup = (dcol[knm1][2] - dcol[kn][2]) / dthick[kn];
        dvdzdn = (dcol[kn][2] - dcol[knp1][2]) / dthick[knp1];
        real_t difsp = 0.5 * ( (1.0 - R) * (dvdzup + fabs(dvdzup))
                             + R         * (dvdzdn + fabs(dvdzdn)) );
        dvdzup = (dcol[knm1][1] - dcol[kn][1]) / dthick[kn];
        dvdzdn = (dcol[kn][1] - dcol[knp1][1]) / dthick[knp1];
        real_t diftp = 0.5 * ( (1.0 - R) * (dvdzup + fabs(dvdzup))
                             + R         * (dvdzdn + fabs(dvdzdn)) );

        real_t visch = dcol[kn][0] + viscp * delhat;
        real_t difsh = dcol[kn][2] + difsp * delhat;
        real_t difth = dcol[kn][1] + diftp * delhat;

        real_t f1 = stable * KPP_CONC1 * bfsfc / (us*us*us*us + KPP_EPSLN);

        real_t gat1m = visch / (hbl + KPP_EPSLN) / (wm + KPP_EPSLN);
        real_t dat1m = -viscp / (wm + KPP_EPSLN) + f1 * visch;
        if (dat1m > 0.0) dat1m = 0.0;
        real_t gat1s = difsh / (hbl + KPP_EPSLN) / (ws + KPP_EPSLN);
        real_t dat1s = -difsp / (ws + KPP_EPSLN) + f1 * difsh;
        if (dat1s > 0.0) dat1s = 0.0;
        real_t gat1t = difth / (hbl + KPP_EPSLN) / (ws + KPP_EPSLN);
        real_t dat1t = -diftp / (ws + KPP_EPSLN) + f1 * difth;
        if (dat1t > 0.0) dat1t = 0.0;

        /* shape functions + BL coeffs at interfaces (eqns 10/11, :1259-1300) */
        real_t sig, a1, a2, a3, Gm, Gs, Gt;
        for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz) {
            if (nz >= kbl) break;
            /* M6.3 (Z7): the C reads Z_3d_n, LIVE under zstar (fesom_kpp.c:553). */
            sig   = fabs(mesh->Z_3d_n[FESOM_NODE3D(n, nz, nl)]) / (hbl + KPP_EPSLN);
            sigma = stable * sig + (1.0 - stable) * fmin(sig, KPP_EPSILON);
            zehat = KPP_VONK * sigma * hbl * bfsfc;
            kpp_wscale(k, zehat, us, &wm, &ws);
            a1 = sig - 2.0; a2 = 3.0 - 2.0 * sig; a3 = sig - 1.0;
            Gm = a1 + a2 * gat1m + a3 * dat1m;
            Gs = a1 + a2 * gat1s + a3 * dat1s;
            Gt = a1 + a2 * gat1t + a3 * dat1t;
            size_t i = FESOM_NODE3D(n, nz, nl);
            blmcM[i] = hbl * wm * sig * (1.0 + sig * Gm);
            blmcT[i] = hbl * ws * sig * (1.0 + sig * Gt);
            blmcS[i] = hbl * ws * sig * (1.0 + sig * Gs);
            k->ghats[(size_t)n * (nl - 1) + nz] =
                (1.0 - stable) * k->cg / (ws * hbl + KPP_EPSLN);
        }

        /* diffusivities at the kbl-1 grid level → dkm1 (:1307-1323).
         * NOTE: σ here uses zbar (interface depth), not Z. */
        sig   = fabs(mesh->zbar_3d_n[FESOM_NODE3D(n, kbl - 1, nl)]) / (hbl + KPP_EPSLN);
        sigma = stable * sig + (1.0 - stable) * fmin(sig, KPP_EPSILON);
        zehat = KPP_VONK * sigma * hbl * bfsfc;
        kpp_wscale(k, zehat, us, &wm, &ws);
        a1 = sig - 2.0; a2 = 3.0 - 2.0 * sig; a3 = sig - 1.0;
        Gm = a1 + a2 * gat1m + a3 * dat1m;
        Gs = a1 + a2 * gat1s + a3 * dat1s;
        Gt = a1 + a2 * gat1t + a3 * dat1t;
        k->dkm1[0 * N + n] = hbl * wm * sig * (1.0 + sig * Gm);
        k->dkm1[1 * N + n] = hbl * ws * sig * (1.0 + sig * Gt);
        k->dkm1[2 * N + n] = hbl * ws * sig * (1.0 + sig * Gs);
    }
}

/*--- blmix_kpp — DEVICE twin (M2.3b) -----------------------------------------
 * One parallel_for over owned nodes; the whole per-node body (per-column dthick/
 * dcol scratch, the wscale calls, the shape-function interface loop, the kbl-1
 * dkm1) runs inside the lambda → race-free (each node owns its blmc/dkm1/ghats
 * column). The blmc zero-fill (C memset over 3·slab) is a preceding deep_copy. The
 * `continue` temporary-solution skips become `return` (a per-node lambda has no
 * outer loop). dthick/dcol are lambda-local (per-thread; nl≤KPP_NL_MAX=64, checked
 * once on the host before the launch). Verbatim arithmetic; wscale via the inline twin.
 */
static void kpp_blmix_kk(fesom_kpp *k, const struct fesom_mesh *mesh)
{
    const int nl  = k->nl;
    const int N   = k->n_nod;
    const int Nmy = mesh->myDim_nod2D;
    const size_t slab = (size_t)N * (size_t)nl;
    const real_t deltaz = k->deltaz, deltau = k->deltau, cg = k->cg;
    FESOM_CHECK(nl <= KPP_NL_MAX, "kpp_blmix_kk: nl %d > KPP_NL_MAX", nl);

    auto viscA = k->viscA_fld.d();
    auto diffK = k->diffK_fld.d();
    auto blmc  = k->blmc_fld.d();
    auto dkm1  = k->dkm1_fld.d();
    auto ghats = k->ghats_fld.d();
    auto hblV  = k->hbl_fld.d();
    auto bfsfcV = k->bfsfc_fld.d();
    auto stableV = k->stable_fld.d();
    auto ustarV  = k->ustar_fld.d();
    auto caseAV  = k->caseA_fld.d();
    auto kblV    = k->kbl_fld.d();
    auto wmt = k->wmt_fld.d();
    auto wst = k->wst_fld.d();
    auto hnode = mesh->hnode_fld.d();
    auto Z     = mesh->Z_fld.d();
    auto Z3d   = mesh->Z_3d_n_fld.d();   /* M6.3 (Z7): the C reads Z_3d_n (fesom_kpp.c:553) */
    auto zbar3 = mesh->zbar_3d_n_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();   /* 1-based counts (skip checks) */
    auto nlev_n = mesh->nlevels_nod2D_fld.d();

    Kokkos::deep_copy(blmc, 0.0);   /* zero blmc over 3·slab (C memset, :1177-1179) */

    Kokkos::parallel_for("kpp_blmix", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;                       /* 0-based deepest interface */
            if (nlev_n(n) < 3) return;                       /* temporary-solution skips */
            if (nlev_n(n) - ulev_n(n) < 2) return;

            real_t hbl    = hblV(n);
            real_t bfsfc  = bfsfcV(n);
            real_t stable = stableV(n);
            real_t us     = ustarV(n);
            int    kbl    = kblV(n);

            real_t dthick[KPP_NL_MAX];
            real_t dcol[KPP_NL_MAX][3];
            for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz)
                dthick[nz] = 0.5 * ( hnode(FESOM_NODE3D(n, nz - 1, nl))
                                   + hnode(FESOM_NODE3D(n, nz,     nl)) );
            dthick[nzmin] = hnode(FESOM_NODE3D(n, nzmin,     nl)) * 0.5;
            dthick[nzmax] = hnode(FESOM_NODE3D(n, nzmax - 1, nl)) * 0.5;
            for (int nz = nzmin; nz <= nzmax - 1; ++nz) {
                size_t i = FESOM_NODE3D(n, nz, nl);
                dcol[nz][0] = viscA(i);
                dcol[nz][1] = diffK(0 * slab + i);
                dcol[nz][2] = diffK(1 * slab + i);
            }
            dcol[nzmax][0] = dcol[nzmax - 1][0];
            dcol[nzmax][1] = dcol[nzmax - 1][1];
            dcol[nzmax][2] = dcol[nzmax - 1][2];

            real_t sigma = stable * 1.0 + (1.0 - stable) * KPP_EPSILON;
            real_t zehat = KPP_VONK * sigma * hbl * bfsfc;
            real_t wm, ws;
            kpp_wscale_kk(wmt, wst, deltaz, deltau, zehat, us, wm, ws);

            int ca   = (int)(caseAV(n) + KPP_EPSLN);
            int kn   = ca * (kbl - 1) + (1 - ca) * kbl;
            if (kn   > nzmax - 1) kn   = nzmax - 1;
            int knm1 = kn - 1; if (knm1 < nzmin) knm1 = nzmin;
            int knp1 = kn + 1; if (knp1 > nzmax) knp1 = nzmax;

            /* M6.3 (Z7) — ⚠️ THE zstar BIT-ID BUG. The C reads Z_3d_n (fesom_kpp.c:516), which is
             * LIVE under zstar. This kernel read the STATIC Z. Invisible under linfs (Z_3d_n == Z
             * always) AND at cold start (hbar==0 => Z_3d_n == Z BITWISE), so it survived every gate
             * and only bit from step 2 on: a wrong delhat moves the KPP boundary-layer shape
             * function => Kv/Av shift ~9e-5 => u/v via impl_vert_visc => ssh_rhs => eta_n. */
            real_t delhat = Kokkos::fabs(Z3d(FESOM_NODE3D(n, kn, nl))) - hbl;
            real_t R      = 1.0 - delhat / dthick[kn];
            real_t dvdzup, dvdzdn;
            dvdzup = (dcol[knm1][0] - dcol[kn][0]) / dthick[kn];
            dvdzdn = (dcol[kn][0] - dcol[knp1][0]) / dthick[knp1];
            real_t viscp = 0.5 * ( (1.0 - R) * (dvdzup + Kokkos::fabs(dvdzup))
                                 + R         * (dvdzdn + Kokkos::fabs(dvdzdn)) );
            dvdzup = (dcol[knm1][2] - dcol[kn][2]) / dthick[kn];
            dvdzdn = (dcol[kn][2] - dcol[knp1][2]) / dthick[knp1];
            real_t difsp = 0.5 * ( (1.0 - R) * (dvdzup + Kokkos::fabs(dvdzup))
                                 + R         * (dvdzdn + Kokkos::fabs(dvdzdn)) );
            dvdzup = (dcol[knm1][1] - dcol[kn][1]) / dthick[kn];
            dvdzdn = (dcol[kn][1] - dcol[knp1][1]) / dthick[knp1];
            real_t diftp = 0.5 * ( (1.0 - R) * (dvdzup + Kokkos::fabs(dvdzup))
                                 + R         * (dvdzdn + Kokkos::fabs(dvdzdn)) );

            real_t visch = dcol[kn][0] + viscp * delhat;
            real_t difsh = dcol[kn][2] + difsp * delhat;
            real_t difth = dcol[kn][1] + diftp * delhat;

            real_t f1 = stable * KPP_CONC1 * bfsfc / (us*us*us*us + KPP_EPSLN);

            real_t gat1m = visch / (hbl + KPP_EPSLN) / (wm + KPP_EPSLN);
            real_t dat1m = -viscp / (wm + KPP_EPSLN) + f1 * visch;
            if (dat1m > 0.0) dat1m = 0.0;
            real_t gat1s = difsh / (hbl + KPP_EPSLN) / (ws + KPP_EPSLN);
            real_t dat1s = -difsp / (ws + KPP_EPSLN) + f1 * difsh;
            if (dat1s > 0.0) dat1s = 0.0;
            real_t gat1t = difth / (hbl + KPP_EPSLN) / (ws + KPP_EPSLN);
            real_t dat1t = -diftp / (ws + KPP_EPSLN) + f1 * difth;
            if (dat1t > 0.0) dat1t = 0.0;

            real_t sig, a1, a2, a3, Gm, Gs, Gt;
            for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz) {
                if (nz >= kbl) break;
                sig   = Kokkos::fabs(Z3d(FESOM_NODE3D(n, nz, nl))) / (hbl + KPP_EPSLN);
                sigma = stable * sig + (1.0 - stable) * Kokkos::fmin(sig, (real_t)KPP_EPSILON);
                zehat = KPP_VONK * sigma * hbl * bfsfc;
                kpp_wscale_kk(wmt, wst, deltaz, deltau, zehat, us, wm, ws);
                a1 = sig - 2.0; a2 = 3.0 - 2.0 * sig; a3 = sig - 1.0;
                Gm = a1 + a2 * gat1m + a3 * dat1m;
                Gs = a1 + a2 * gat1s + a3 * dat1s;
                Gt = a1 + a2 * gat1t + a3 * dat1t;
                size_t i = FESOM_NODE3D(n, nz, nl);
                blmc(0 * slab + i) = hbl * wm * sig * (1.0 + sig * Gm);
                blmc(1 * slab + i) = hbl * ws * sig * (1.0 + sig * Gt);
                blmc(2 * slab + i) = hbl * ws * sig * (1.0 + sig * Gs);
                ghats((size_t)n * (nl - 1) + nz) =
                    (1.0 - stable) * cg / (ws * hbl + KPP_EPSLN);
            }

            sig   = Kokkos::fabs(zbar3(FESOM_NODE3D(n, kbl - 1, nl))) / (hbl + KPP_EPSLN);
            sigma = stable * sig + (1.0 - stable) * Kokkos::fmin(sig, (real_t)KPP_EPSILON);
            zehat = KPP_VONK * sigma * hbl * bfsfc;
            kpp_wscale_kk(wmt, wst, deltaz, deltau, zehat, us, wm, ws);
            a1 = sig - 2.0; a2 = 3.0 - 2.0 * sig; a3 = sig - 1.0;
            Gm = a1 + a2 * gat1m + a3 * dat1m;
            Gs = a1 + a2 * gat1s + a3 * dat1s;
            Gt = a1 + a2 * gat1t + a3 * dat1t;
            dkm1(0 * N + n) = hbl * wm * sig * (1.0 + sig * Gm);
            dkm1(1 * N + n) = hbl * ws * sig * (1.0 + sig * Gt);
            dkm1(2 * N + n) = hbl * ws * sig * (1.0 + sig * Gs);
        });
}

/*===========================================================================
 * enhance — enhancement of the BL coeffs at the kbl-1 interface using dkm1.
 * Literal port of enhance (oce_ale_mixing_kpp.F90:1346-1390). Blends the
 * interior (caseA) coefficient, the BL coefficient, and dkm1 at k=kbl-1 with
 * the fractional position delta. Channels 0/1/2 = momentum/T/S (T uses diffK
 * ch1, S uses diffK ch2). Also scales ghats(kbl-1) by (1-caseA). Loops myDim.
 *===========================================================================*/
static void kpp_enhance(fesom_kpp *k, const struct fesom_mesh *mesh)
{
    const int nl = k->nl, N = k->n_nod, Nmy = mesh->myDim_nod2D;
    const size_t slab = (size_t)N * (size_t)nl;
    real_t *viscA  = k->viscA;
    real_t *diffKt = k->diffK + 0 * slab, *diffKs = k->diffK + 1 * slab;
    real_t *blmcM  = k->blmc  + 0 * slab, *blmcT  = k->blmc  + 1 * slab,
           *blmcS  = k->blmc  + 2 * slab;

    for (int n = 0; n < Nmy; ++n) {
        int kk = k->kbl[n] - 1;                              /* k = kbl-1 (:1361) */
        real_t caseA = k->caseA[n];
        real_t zk  = mesh->zbar_3d_n[FESOM_NODE3D(n, kk,     nl)];
        real_t zk1 = mesh->zbar_3d_n[FESOM_NODE3D(n, kk + 1, nl)];
        real_t delta = (k->hbl[n] + zk) / (zk - zk1);        /* :1362 */
        real_t om = 1.0 - delta, om2 = om * om, d2 = delta * delta;
        size_t ik = FESOM_NODE3D(n, kk, nl);
        real_t dkmp5, dstar;
        /* momentum (:1365-1369) */
        dkmp5 = caseA * viscA[ik] + (1.0 - caseA) * blmcM[ik];
        dstar = om2 * k->dkm1[0 * N + n] + d2 * dkmp5;
        blmcM[ik] = om * viscA[ik] + delta * dstar;
        /* temperature (:1372-1376): diffK ch1 → blmc ch2 */
        dkmp5 = caseA * diffKt[ik] + (1.0 - caseA) * blmcT[ik];
        dstar = om2 * k->dkm1[1 * N + n] + d2 * dkmp5;
        blmcT[ik] = om * diffKt[ik] + delta * dstar;
        /* salinity (:1379-1383): diffK ch2 → blmc ch3 */
        dkmp5 = caseA * diffKs[ik] + (1.0 - caseA) * blmcS[ik];
        dstar = om2 * k->dkm1[2 * N + n] + d2 * dkmp5;
        blmcS[ik] = om * diffKs[ik] + delta * dstar;
        /* ghats (:1385) */
        k->ghats[(size_t)n * (nl - 1) + kk] *= (1.0 - caseA);
    }
}

/*--- enhance — DEVICE twin (M2.3b) -------------------------------------------
 * Single parallel_for over owned nodes; per-node enhancement at the kbl-1
 * interface (each node owns its blmc/ghats slot → race-free). Verbatim arithmetic.
 */
static void kpp_enhance_kk(fesom_kpp *k, const struct fesom_mesh *mesh)
{
    const int nl = k->nl, N = k->n_nod, Nmy = mesh->myDim_nod2D;
    const size_t slab = (size_t)N * (size_t)nl;
    auto viscA = k->viscA_fld.d();
    auto diffK = k->diffK_fld.d();
    auto blmc  = k->blmc_fld.d();
    auto dkm1  = k->dkm1_fld.d();
    auto ghats = k->ghats_fld.d();
    auto hblV  = k->hbl_fld.d();
    auto caseAV = k->caseA_fld.d();
    auto kblV  = k->kbl_fld.d();
    auto zbar3 = mesh->zbar_3d_n_fld.d();

    Kokkos::parallel_for("kpp_enhance", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int kk = kblV(n) - 1;                            /* k = kbl-1 */
            real_t caseA = caseAV(n);
            real_t zk  = zbar3(FESOM_NODE3D(n, kk,     nl));
            real_t zk1 = zbar3(FESOM_NODE3D(n, kk + 1, nl));
            real_t delta = (hblV(n) + zk) / (zk - zk1);
            real_t om = 1.0 - delta, om2 = om * om, d2 = delta * delta;
            size_t ik = FESOM_NODE3D(n, kk, nl);
            real_t dkmp5, dstar;
            /* momentum */
            dkmp5 = caseA * viscA(ik) + (1.0 - caseA) * blmc(0 * slab + ik);
            dstar = om2 * dkm1(0 * N + n) + d2 * dkmp5;
            blmc(0 * slab + ik) = om * viscA(ik) + delta * dstar;
            /* temperature: diffK ch0 → blmc ch1 */
            dkmp5 = caseA * diffK(0 * slab + ik) + (1.0 - caseA) * blmc(1 * slab + ik);
            dstar = om2 * dkm1(1 * N + n) + d2 * dkmp5;
            blmc(1 * slab + ik) = om * diffK(0 * slab + ik) + delta * dstar;
            /* salinity: diffK ch1 → blmc ch2 */
            dkmp5 = caseA * diffK(1 * slab + ik) + (1.0 - caseA) * blmc(2 * slab + ik);
            dstar = om2 * dkm1(2 * N + n) + d2 * dkmp5;
            blmc(2 * slab + ik) = om * diffK(1 * slab + ik) + delta * dstar;
            /* ghats */
            ghats((size_t)n * (nl - 1) + kk) *= (1.0 - caseA);
        });
}

/* K5 dump: driver pre-step inputs to bldepth — prestep(ustar,Bo) + dVsq + dbsfc
 * columns. Mirrors the Fortran kpp_dump_prestep (oce_ale_mixing_kpp.F90:554-578). */
static double kpp_get_prestep(int node, int comp, void *user)
{
    const fesom_kpp *k = *(const fesom_kpp *const *)user;
    return comp == 0 ? (double)k->ustar[node] : (double)k->Bo[node];
}
static void kpp_dump_prestep(const fesom_kpp *k, const struct fesom_aux *aux,
                             const struct fesom_mesh *mesh, struct fesom_partit *partit)
{
    const fesom_kpp *kp = k;
    kpp_col_src dv = { k->dVsq,    k->nl };
    kpp_col_src db = { aux->dbsfc, k->nl };
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "prestep", 2, kpp_get_prestep, &kp);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "dVsq",  k->nl, kpp_get_col, &dv);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "dbsfc", k->nl, kpp_get_col, &db);
}

/* K5 dump: bldepth outputs — hbl, kbl(+1 → Fortran 1-based), bfsfc, stable, caseA.
 * Mirrors the Fortran kpp_dump_final's 'bldepth' tag (:596-604). */
static double kpp_get_bldepth(int node, int comp, void *user)
{
    const fesom_kpp *k = *(const fesom_kpp *const *)user;
    switch (comp) {
        case 0:  return (double)k->hbl[node];
        case 1:  return (double)(k->kbl[node] + 1);   /* 0-based → Fortran 1-based */
        case 2:  return (double)k->bfsfc[node];
        case 3:  return (double)k->stable[node];
        default: return (double)k->caseA[node];
    }
}
static void kpp_dump_bldepth(const fesom_kpp *k, const struct fesom_mesh *mesh,
                             struct fesom_partit *partit)
{
    const fesom_kpp *kp = k;
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "bldepth", 5, kpp_get_bldepth, &kp);
}

/* K6 dump: blmix outputs — blmc[3] (full columns), ghats (nl-1 cols), dkm1[3]
 * (per node). Dumped right after blmix, BEFORE enhance/combine, to match the
 * incremental C driver and the Fortran kpp_dump_blmix point (after CALL blmix_kpp,
 * before enhance — enhance modifies blmc/ghats at kbl-1). */
static double kpp_get_dkm1(int node, int comp, void *user)
{
    const fesom_kpp *k = *(const fesom_kpp *const *)user;
    return (double)k->dkm1[(size_t)comp * (size_t)k->n_nod + node];   /* j-major [3*N] */
}
static void kpp_dump_blmix(const fesom_kpp *k, const struct fesom_mesh *mesh,
                           struct fesom_partit *partit)
{
    size_t slab = (size_t)k->n_nod * (size_t)k->nl;
    const fesom_kpp *kp = k;
    kpp_col_src bm = { k->blmc + 0 * slab, k->nl };
    kpp_col_src bt = { k->blmc + 1 * slab, k->nl };
    kpp_col_src bs = { k->blmc + 2 * slab, k->nl };
    kpp_col_src gh = { k->ghats,           k->nl - 1 };   /* ghats stride is nl-1 */
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "blmc_m",   k->nl,     kpp_get_col, &bm);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "blmc_t",   k->nl,     kpp_get_col, &bt);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "blmc_s",   k->nl,     kpp_get_col, &bs);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "bl_ghats", k->nl - 1, kpp_get_col, &gh);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "dkm1",     3,         kpp_get_dkm1, &kp);
}

/* K7 dump: final module-gate fields — post-combine node viscA/diffKt/diffKs/ghats
 * + the element viscAE (=aux->Av). Mirrors the Fortran kpp_dump_final (node tags)
 * plus the added viscAE element dump. kpp_get_col indexes arr[id*stride+comp] for
 * both node and element arrays. */
static void kpp_dump_final(const fesom_kpp *k, const struct fesom_aux *aux,
                           const struct fesom_mesh *mesh, struct fesom_partit *partit)
{
    size_t slab = (size_t)k->n_nod * (size_t)k->nl;
    kpp_col_src sv = { k->viscA,          k->nl };
    kpp_col_src st = { k->diffK + 0*slab, k->nl };
    kpp_col_src ss = { k->diffK + 1*slab, k->nl };
    kpp_col_src sg = { k->ghats,          k->nl - 1 };
    kpp_col_src sa = { aux->Av,           k->nl };
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "viscA",  k->nl,     kpp_get_col, &sv);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "diffKt", k->nl,     kpp_get_col, &st);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "diffKs", k->nl,     kpp_get_col, &ss);
    fesom_kpp_dump_nodes(mesh, partit, s_kpp_call, "ghats",  k->nl - 1, kpp_get_col, &sg);
    fesom_kpp_dump_elems(mesh, partit, s_kpp_call, "viscAE", k->nl,     kpp_get_col, &sa);
}

/* ---- controlled-input replay (FESOM_KPP_REPLAY_DIR) ----------------------
 * Validation aid for K6+: overwrite the C's bldepth/prestep/ri_iwmix outputs
 * with the FORTRAN-dumped values, so the subsequent blmix (/enhance/combine)
 * diff is PURE algebra — isolated from the step-1 forcing difference
 * (project_forcing_step1_diff) that perturbs bfsfc/ustar at ~every node and
 * (via f1∝bfsfc/ustar⁴, wscale) swamps a live-run blmc comparison. The Fortran
 * rank-R dump lists rank R's owned nodes in myList order (same 16r partition as
 * the C), so file line i == local node i (asserted against the gid). */
static void kpp_replay_file(const char *dir, const char *tag, struct fesom_partit *partit,
                            int Nmy, int ncomp, double *out)
{
    char path[1024], line[16384];
    snprintf(path, sizeof(path), "%s/kpp_dump_s1_%s_rank%d.txt", dir, tag, partit->mype);
    FILE *f = fopen(path, "r");
    FESOM_CHECK(f, "kpp_replay: cannot open %s", path);
    if (!fgets(line, sizeof(line), f)) FESOM_DIE("kpp_replay: empty %s", path);   /* header */
    for (int i = 0; i < Nmy; ++i) {
        if (!fgets(line, sizeof(line), f)) FESOM_DIE("kpp_replay: short %s at %d", path, i);
        char *p = line, *e;
        long gid = strtol(p, &e, 10); p = e;
        FESOM_CHECK(gid == partit->myList_nod2D[i],
                    "kpp_replay %s: gid/order mismatch at line %d (%ld vs %d)",
                    tag, i, gid, partit->myList_nod2D[i]);
        for (int c = 0; c < ncomp; ++c) { out[(size_t)i * ncomp + c] = strtod(p, &e); p = e; }
    }
    fclose(f);
}
static void kpp_replay_inputs(fesom_kpp *k, const struct fesom_mesh *mesh,
                              struct fesom_partit *partit, const char *dir)
{
    const int nl = k->nl, Nmy = mesh->myDim_nod2D;
    const size_t slab = (size_t)k->n_nod * (size_t)nl;
    double *buf = (double *)malloc((size_t)Nmy * (size_t)nl * sizeof(double));
    double *p5  = (double *)malloc((size_t)Nmy * 5 * sizeof(double));
    FESOM_CHECK(buf && p5, "kpp_replay: out of memory");
    /* bldepth (5): hbl, kbl(dump is +1 → back to 0-based), bfsfc, stable, caseA */
    kpp_replay_file(dir, "bldepth", partit, Nmy, 5, p5);
    for (int i = 0; i < Nmy; ++i) {
        k->hbl[i]    = p5[i*5+0];  k->kbl[i]   = (int)lround(p5[i*5+1]) - 1;
        k->bfsfc[i]  = p5[i*5+2];  k->stable[i]= p5[i*5+3];  k->caseA[i] = p5[i*5+4];
    }
    kpp_replay_file(dir, "prestep", partit, Nmy, 2, p5);      /* ustar, Bo */
    for (int i = 0; i < Nmy; ++i) { k->ustar[i] = p5[i*2+0]; k->Bo[i] = p5[i*2+1]; }
    kpp_replay_file(dir, "ri_viscA", partit, Nmy, nl, buf);
    for (int i = 0; i < Nmy; ++i) for (int nz = 0; nz < nl; ++nz)
        k->viscA[(size_t)i*nl + nz] = buf[(size_t)i*nl + nz];
    kpp_replay_file(dir, "ri_diffKt", partit, Nmy, nl, buf);
    for (int i = 0; i < Nmy; ++i) for (int nz = 0; nz < nl; ++nz)
        k->diffK[0*slab + (size_t)i*nl + nz] = buf[(size_t)i*nl + nz];
    kpp_replay_file(dir, "ri_diffKs", partit, Nmy, nl, buf);
    for (int i = 0; i < Nmy; ++i) for (int nz = 0; nz < nl; ++nz)
        k->diffK[1*slab + (size_t)i*nl + nz] = buf[(size_t)i*nl + nz];
    free(buf); free(p5);
    if (partit->mype == 0)
        fprintf(stderr, "[kpp_replay] injected Fortran inputs from %s (blmix diff = pure algebra)\n", dir);
}

/*===========================================================================
 * KPP driver — mirror of oce_mixing_KPP (oce_ale_mixing_kpp.F90:249-494),
 * built incrementally. Through K5: zero viscA + pre-step (dVsq/ustar/Bo) +
 * ri_iwmix + bldepth (+ dumps). Downstream (blmix_kpp/enhance/combine/
 * single-Kv) land K6..K8 — guarded so only a FESOM_KPP_DUMP_DIR dump run
 * (1 step) may proceed until the driver is complete.
 *===========================================================================*/
void fesom_kpp_mixing(fesom_kpp                  *k,
                      struct fesom_aux           *aux,
                      const struct fesom_tracers *tracers,
                      const struct fesom_forcing *forcing,
                      const struct fesom_dyn     *dyn,
                      const struct fesom_mesh    *mesh,
                      struct fesom_partit        *partit)
{
    ++s_kpp_call;                            /* = ocean step (mirrors Fortran kpp_call_count) */
    const int nl  = k->nl;
    const int Nmy = mesh->myDim_nod2D;

    /* zero viscA over myDim+eDim (driver :340-343) */
    memset(k->viscA, 0, (size_t)k->n_nod * (size_t)nl * sizeof(real_t));

    /* ---- driver pre-step (oce_ale_mixing_kpp.F90:344-411) -------------------
     * dVsq (eqn 21: shear of UVnode re the surface), ustar (eqn 2), Bo (A2c/A2d).
     * dbsfc is filled by the EOS (fesom_eos.c); here only its surface level is
     * zeroed, matching the Fortran driver (:367). */
    const real_t *S = tracers->data[FESOM_TRACER_S].values;
    const real_t density_0_r = 1.0 / (real_t)FESOM_DENSITY_0;

    for (int n = 0; n < Nmy; ++n) {
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        int nzmax = mesh->nlevels_nod2D[n] - 1;
        k->dVsq[FESOM_NODE3D(n, nzmin, nl)]    = 0.0;     /* :366 */
        aux->dbsfc[FESOM_NODE3D(n, nzmin, nl)] = 0.0;     /* :367 */
        real_t usurf = dyn->uvnode[FESOM_ELEMVEC(n, nzmin, nl) + 0];   /* :370 */
        real_t vsurf = dyn->uvnode[FESOM_ELEMVEC(n, nzmin, nl) + 1];   /* :371 */
        for (int nz = nzmin + 1; nz < nzmax; ++nz) {      /* :372 (nz=nzmin+1..nzmax-1) */
            real_t u_loc = 0.5 * ( dyn->uvnode[FESOM_ELEMVEC(n, nz - 1, nl) + 0]
                                 + dyn->uvnode[FESOM_ELEMVEC(n, nz,     nl) + 0] );
            real_t v_loc = 0.5 * ( dyn->uvnode[FESOM_ELEMVEC(n, nz - 1, nl) + 1]
                                 + dyn->uvnode[FESOM_ELEMVEC(n, nz,     nl) + 1] );
            real_t du = usurf - u_loc, dv = vsurf - v_loc;
            k->dVsq[FESOM_NODE3D(n, nz, nl)] = du * du + dv * dv;       /* :377 */
        }
        k->dVsq[FESOM_NODE3D(n, nzmax, nl)] =
            k->dVsq[FESOM_NODE3D(n, nzmax - 1, nl)];                    /* :380 */
    }

    for (int n = 0; n < Nmy; ++n) {                       /* :401-411 */
        int nzmin = mesh->ulevels_nod2D[n] - 1;
        real_t sx = forcing->stress_node_surf[2*n + 0];
        real_t sy = forcing->stress_node_surf[2*n + 1];
        k->ustar[n] = sqrt( sqrt(sx*sx + sy*sy) * density_0_r );        /* eqn 2 (:404) */
        k->Bo[n] = -(real_t)FESOM_G
                   * ( aux->sw_alpha[FESOM_NODE3D(n, nzmin, nl)]
                         * forcing->heat_flux[n] / (real_t)FESOM_VCPW
                     + aux->sw_beta[FESOM_NODE3D(n, nzmin, nl)]
                         * forcing->water_flux[n] * S[FESOM_NODE3D(n, nzmin, nl)] ); /* :409-410 */
    }

    /* interior mixing (ri_iwmix, :419) */
    kpp_ri_iwmix(k, aux, dyn, mesh);

    /* K4: double diffusion gate (driver :423-425). CORE2 double_diffusion=.false.
     * → ddmix not called (body deferred, port-what-CORE2-uses). */
#if KPP_DOUBLE_DIFFUSION
#error "KPP double_diffusion=.true. unsupported: port ddmix (oce_ale_mixing_kpp.F90:1012-1085)"
    /* CALL ddmix(diffK, ...) */
#endif

    /* OBL depth (bldepth, :428) — K5 */
    kpp_bldepth(k, aux, forcing, mesh);

    /* validation: optionally replace bldepth/prestep/ri inputs with the Fortran
     * dump so the blmix diff is pure algebra (FESOM_KPP_REPLAY_DIR, K6+ only). */
    {
        const char *replay = getenv("FESOM_KPP_REPLAY_DIR");
        if (replay && fesom_kpp_dump_this_step(s_kpp_call))
            kpp_replay_inputs(k, mesh, partit, replay);
    }

    /* boundary-layer mixing coeffs (blmix_kpp, :430) — K6 */
    kpp_blmix(k, mesh);

    /* K6 dumps (BEFORE enhance — blmix blmc/ghats are modified at kbl-1 by enhance) */
    if (fesom_kpp_dump_this_step(s_kpp_call)) {
        kpp_dump_prestep(k, aux, mesh, partit);
        kpp_dump_riiwmix(k, aux, mesh, partit);
        kpp_dump_bldepth(k, mesh, partit);
        kpp_dump_blmix(k, mesh, partit);
    }

    /* enhance the BL coeffs at kbl-1 (enhance, :435) — K7 */
    kpp_enhance(k, mesh);

    const size_t slab = (size_t)k->n_nod * (size_t)nl;

    /* smooth_blmc (:437-447): exchange each blmc channel then 3-sweep smooth_nod3D.
     * smooth_blmc=.true. (hardcoded module logical). Every rank replays/computes its
     * own owned blmc, so the pre-smooth exchange fills halos with owning-rank values. */
    for (int j = 0; j < 3; ++j) fesom_exchange_nod3D(k->blmc + (size_t)j * slab, nl, partit);
    for (int j = 0; j < 3; ++j) fesom_smooth_nod3D(k->blmc + (size_t)j * slab, nl, 3, mesh, partit);

    /* combine blmc into viscA/diffK within the BL; zero ghats outside (:451-463) */
    {
        real_t *diffKt = k->diffK + 0 * slab, *diffKs = k->diffK + 1 * slab;
        real_t *blmcM  = k->blmc  + 0 * slab, *blmcT  = k->blmc  + 1 * slab,
               *blmcS  = k->blmc  + 2 * slab;
        for (int n = 0; n < Nmy; ++n) {
            int nzmin = mesh->ulevels_nod2D[n] - 1;
            int nzmax = mesh->nlevels_nod2D[n] - 1;
            int kbl   = k->kbl[n];
            for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz) {
                size_t i = FESOM_NODE3D(n, nz, nl);
                if (nz < kbl) {                                  /* within the BL */
                    k->viscA[i] = fmax(k->viscA[i], blmcM[i]);
                    diffKt[i]   = fmax(diffKt[i],   blmcT[i]);
                    diffKs[i]   = fmax(diffKs[i],   blmcS[i]);
                } else {
                    k->ghats[(size_t)n * (nl - 1) + nz] = 0.0;
                }
            }
        }
    }

    /* exchanges before the node→elem average (:468-473) */
    fesom_exchange_nod3D(k->diffK + 0 * slab, nl, partit);
    fesom_exchange_nod3D(k->diffK + 1 * slab, nl, partit);
    fesom_exchange_nod3D(k->ghats, nl - 1, partit);
    fesom_exchange_nod3D(k->viscA, nl, partit);   /* needed before averaging on elements */

    /* node→elem viscAE = Σ(viscA over 3 vertices)/3 + bottom fill + minmix floor
     * (:475-490). viscAE is the element viscosity = aux->Av. */
    for (int e = 0; e < mesh->myDim_elem2D; ++e) {
        int n0 = mesh->elem_nodes[3*e + 0];
        int n1 = mesh->elem_nodes[3*e + 1];
        int n2 = mesh->elem_nodes[3*e + 2];
        int nzmin = mesh->ulevels[e] - 1;
        int nzmax = mesh->nlevels[e] - 1;
        for (int nz = nzmin; nz < nzmax; ++nz)
            aux->Av[FESOM_ELEM3D(e, nz, nl)] =
                ( k->viscA[FESOM_NODE3D(n0, nz, nl)]
                + k->viscA[FESOM_NODE3D(n1, nz, nl)]
                + k->viscA[FESOM_NODE3D(n2, nz, nl)] ) / 3.0;
        aux->Av[FESOM_ELEM3D(e, nzmax, nl)] = aux->Av[FESOM_ELEM3D(e, nzmax - 1, nl)];
        if (aux->Av[FESOM_ELEM3D(e, nzmin, nl)] < KPP_MINMIX)
            aux->Av[FESOM_ELEM3D(e, nzmin, nl)] = KPP_MINMIX;
    }

    /* K8: single Kv = diffK(:,:,1) (T-channel) over myDim+eDim, used for BOTH T
     * and S tracer diffusion (oce_ale.F90:3518-3522). diffK ch0 (= slab 0) is
     * halo-valid from the combine exchange; fesom_tracer_diff reads this single
     * aux->Kv UNCHANGED (the S-channel diffK(:,:,2) is NOT routed into salinity in
     * CORE2 — it only feeds the gated-off nonlocal flux). aux->Kv and k->diffK
     * ch0 are both [n_nod*nl], so this is a full-slab copy. */
    memcpy(aux->Kv, k->diffK, (size_t)k->n_nod * (size_t)nl * sizeof(real_t));

    /* dump: final module-gate fields (post-combine viscA/diffKt/diffKs/ghats +
     * viscAE); the post-copy aux->Kv == diffKt (the array the TDMA actually reads). */
    if (fesom_kpp_dump_this_step(s_kpp_call))
        kpp_dump_final(k, aux, mesh, partit);
}

/*===========================================================================
 * KPP driver — DEVICE twin (M2.3b). Mirror of fesom_kpp_mixing above, with every
 * stage a Kokkos parallel_for over owned nodes/elements (the D19/D20 template).
 * Stage flow is device→device on owned nodes with NO host round-trip until the two
 * internal halo points — smooth_blmc (after enhance) and the pre-elem-average
 * exchanges (after combine) — which stay HOST (MPI pack/unpack + the host
 * smoother) and are bracketed device→host→halo→host→device (SYNC_MAP §6). The IN
 * rail (sync the host-authoritative inputs to device) and OUT rail (sync Av/Kv to
 * host) live in the driver (fesom_step.cpp) where the input structs are non-const;
 * this function owns the INTERNAL brackets (k->* and aux are non-const here) and
 * marks the Av/Kv outputs modify_device() at the end. Verbatim arithmetic — the
 * FESOM_KK_VERIFY=kpp gate asserts Serial max|Δ|==0 vs the C twin above.
 *===========================================================================*/
void fesom_kpp_mixing_kk(fesom_kpp                  *k,
                         struct fesom_aux           *aux,
                         const struct fesom_tracers *tracers,
                         const struct fesom_forcing *forcing,
                         const struct fesom_dyn     *dyn,
                         const struct fesom_mesh    *mesh,
                         struct fesom_partit        *partit)
{
    ++s_kpp_call;                            /* = ocean step (mirrors Fortran kpp_call_count) */
    const int nl  = k->nl;
    const int Nmy = mesh->myDim_nod2D;
    const size_t slab = (size_t)k->n_nod * (size_t)nl;
    const real_t G = (real_t)FESOM_G;
    const real_t VCPW = (real_t)FESOM_VCPW;
    const real_t density_0_r = 1.0 / (real_t)FESOM_DENSITY_0;

    /* device views for the inline stages (prestep / combine / viscAE / Kv copy) */
    auto viscA  = k->viscA_fld.d();
    auto diffK  = k->diffK_fld.d();
    auto blmc   = k->blmc_fld.d();
    auto ghats  = k->ghats_fld.d();
    auto dVsq   = k->dVsq_fld.d();
    auto ustarV = k->ustar_fld.d();
    auto BoV    = k->Bo_fld.d();
    auto kblV   = k->kbl_fld.d();
    auto uvnode = dyn->uvnode_fld.d();
    auto sw_alpha = aux->sw_alpha_fld.d();
    auto sw_beta  = aux->sw_beta_fld.d();
    auto dbsfc    = aux->dbsfc_fld.d();
    auto Av       = aux->Av_fld.d();
    auto Kv       = aux->Kv_fld.d();
    auto stress_n = forcing->stress_node_surf_fld.d();
    auto heat_flux  = forcing->heat_flux_fld.d();
    auto water_flux = forcing->water_flux_fld.d();
    auto Sval     = tracers->data[FESOM_TRACER_S].values_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto nlev_n = mesh->nlevels_nod2D_fld.d();
    auto ulev_e = mesh->ulevels_fld.d();
    auto nlev_e = mesh->nlevels_fld.d();
    auto elnod  = mesh->elem_nodes_fld.d();

    /* zero viscA over N (driver :340-343) */
    Kokkos::deep_copy(viscA, 0.0);

    /* ---- driver pre-step: dVsq (eqn 21) + surface dbsfc zero (:344-380) ---- */
    Kokkos::parallel_for("kpp_prestep_dVsq", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            dVsq(FESOM_NODE3D(n, nzmin, nl))  = 0.0;
            dbsfc(FESOM_NODE3D(n, nzmin, nl)) = 0.0;
            real_t usurf = uvnode(FESOM_ELEMVEC(n, nzmin, nl) + 0);
            real_t vsurf = uvnode(FESOM_ELEMVEC(n, nzmin, nl) + 1);
            for (int nz = nzmin + 1; nz < nzmax; ++nz) {
                real_t u_loc = 0.5 * ( uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 0)
                                     + uvnode(FESOM_ELEMVEC(n, nz,     nl) + 0) );
                real_t v_loc = 0.5 * ( uvnode(FESOM_ELEMVEC(n, nz - 1, nl) + 1)
                                     + uvnode(FESOM_ELEMVEC(n, nz,     nl) + 1) );
                real_t du = usurf - u_loc, dv = vsurf - v_loc;
                dVsq(FESOM_NODE3D(n, nz, nl)) = du * du + dv * dv;
            }
            dVsq(FESOM_NODE3D(n, nzmax, nl)) = dVsq(FESOM_NODE3D(n, nzmax - 1, nl));
        });

    /* ustar (eqn 2) + Bo (A2c/A2d) (:401-411) */
    Kokkos::parallel_for("kpp_prestep_ustar_Bo", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            real_t sx = stress_n(2*n + 0);
            real_t sy = stress_n(2*n + 1);
            ustarV(n) = Kokkos::sqrt( Kokkos::sqrt(sx*sx + sy*sy) * density_0_r );
            BoV(n) = -G * ( sw_alpha(FESOM_NODE3D(n, nzmin, nl)) * heat_flux(n) / VCPW
                          + sw_beta(FESOM_NODE3D(n, nzmin, nl))
                              * water_flux(n) * Sval(FESOM_NODE3D(n, nzmin, nl)) );
        });

    /* interior mixing (ri_iwmix, :419) */
    kpp_ri_iwmix_kk(k, aux, dyn, mesh);

#if KPP_DOUBLE_DIFFUSION
#error "KPP double_diffusion=.true. unsupported: port ddmix (oce_ale_mixing_kpp.F90:1012-1085)"
#endif

    /* OBL depth (bldepth, :428) */
    kpp_bldepth_kk(k, aux, forcing, mesh);

    /* optional Fortran-input replay (FESOM_KPP_REPLAY_DIR; off by default): inject host
     * values then push them to the device so blmix_kk reads the replayed inputs. */
    {
        const char *replay = getenv("FESOM_KPP_REPLAY_DIR");
        if (replay && fesom_kpp_dump_this_step(s_kpp_call)) {
            kpp_replay_inputs(k, mesh, partit, replay);
            k->hbl_fld.modify_host();    k->hbl_fld.sync_device();
            k->kbl_fld.modify_host();    k->kbl_fld.sync_device();
            k->bfsfc_fld.modify_host();  k->bfsfc_fld.sync_device();
            k->stable_fld.modify_host(); k->stable_fld.sync_device();
            k->caseA_fld.modify_host();  k->caseA_fld.sync_device();
            k->ustar_fld.modify_host();  k->ustar_fld.sync_device();
            k->Bo_fld.modify_host();     k->Bo_fld.sync_device();
            k->viscA_fld.modify_host();  k->viscA_fld.sync_device();
            k->diffK_fld.modify_host();  k->diffK_fld.sync_device();
        }
    }

    /* boundary-layer mixing coeffs (blmix_kpp, :430) */
    kpp_blmix_kk(k, mesh);

    /* K6 dumps (BEFORE enhance). Off by default; sync the dumped scratch to host first
     * (the dump readers use the raw host alias). On Serial host==device → no-op. */
    if (fesom_kpp_dump_this_step(s_kpp_call)) {
        k->dVsq_fld.modify_device();  k->dVsq_fld.sync_host();
        k->ustar_fld.modify_device(); k->ustar_fld.sync_host();
        k->Bo_fld.modify_device();    k->Bo_fld.sync_host();
        k->viscA_fld.modify_device(); k->viscA_fld.sync_host();
        k->diffK_fld.modify_device(); k->diffK_fld.sync_host();
        aux->dbsfc_fld.modify_device(); aux->dbsfc_fld.sync_host();
        k->hbl_fld.modify_device();   k->hbl_fld.sync_host();
        k->kbl_fld.modify_device();   k->kbl_fld.sync_host();
        k->bfsfc_fld.modify_device(); k->bfsfc_fld.sync_host();
        k->stable_fld.modify_device(); k->stable_fld.sync_host();
        k->caseA_fld.modify_device(); k->caseA_fld.sync_host();
        k->blmc_fld.modify_device();  k->blmc_fld.sync_host();
        k->ghats_fld.modify_device(); k->ghats_fld.sync_host();
        kpp_dump_prestep(k, aux, mesh, partit);
        kpp_dump_riiwmix(k, aux, mesh, partit);
        kpp_dump_bldepth(k, mesh, partit);
        kpp_dump_blmix(k, mesh, partit);
        /* re-push to device for the rest of the pipeline (no-op on Serial). */
        k->blmc_fld.modify_host(); k->blmc_fld.sync_device();
        k->diffK_fld.modify_host(); k->diffK_fld.sync_device();
        k->viscA_fld.modify_host(); k->viscA_fld.sync_device();
    }

    /* enhance the BL coeffs at kbl-1 (enhance, :435) */
    kpp_enhance_kk(k, mesh);

    /* ---- INTERNAL HALO BRACKET 1: smooth_blmc (:437-447) ---------------------
     * smooth_blmc=.true.: exchange each blmc channel then 3-sweep smooth_nod3D (HOST).
     * Device-compute → sync_host → halo+smooth (h_checked) → sync_device. */
    k->blmc_fld.modify_device();   /* blmix wrote blmc (device) */
    /* M5.5 (B): DEVICE smoother — no host round-trip. M5.12d: the 3 channels are smoothed in
     * ONE call (all 3 slabs per gather/scale kernel, decoded inside) instead of 3 separate calls
     * — 9 gather + 9 scale launches → 3 + 3. Channels are independent (each reads only its own
     * slab) so the result is bit-identical. Initial per-channel device-halo first (the smoother's
     * entry contract), then the 3-sweep combined smooth (each sweep re-halos every channel).
     * blmc stays device-resident; the combine below reads it at OWNED nodes. */
    for (int j = 0; j < 3; ++j)
        fesom_halo_field(k->blmc_fld, FESOM_HALO_NOD3D, nl, 1, partit, (std::size_t)j * slab);
    /* M5.18: FESOM_KK_VERIFY=smooth isolates the blmc smoother vs the host C twin (the kpp gate
     * only sees it transitively through the max(viscA/diffK, blmc) combine). Cached env read. */
    static int s_verify_smooth = -1;
    if (s_verify_smooth < 0) {
        const char *e = getenv("FESOM_KK_VERIFY");
        s_verify_smooth = (e && strstr(e, "smooth")) ? 1 : 0;
    }
    if (s_verify_smooth)
        fesom_smooth_nod3D_kk_verify(k->blmc_fld, 3, mesh, partit, /*base=*/0, /*nslab=*/3,
                                     /*slab_stride=*/slab, "blmc", s_kpp_call);
    else
        fesom_smooth_nod3D_kk(k->blmc_fld, 3, mesh, partit, /*base=*/0, /*nslab=*/3, /*slab_stride=*/slab);

    /* combine blmc into viscA/diffK within the BL; zero ghats outside (:451-463) */
    Kokkos::parallel_for("kpp_combine", Kokkos::RangePolicy<>(0, Nmy),
        KOKKOS_LAMBDA(const int n) {
            int nzmin = ulev_n(n) - 1;
            int nzmax = nlev_n(n) - 1;
            int kbl   = kblV(n);
            for (int nz = nzmin + 1; nz <= nzmax - 1; ++nz) {
                size_t i = FESOM_NODE3D(n, nz, nl);
                if (nz < kbl) {                                  /* within the BL */
                    viscA(i)            = Kokkos::fmax(viscA(i),            blmc(0*slab + i));
                    diffK(0*slab + i)   = Kokkos::fmax(diffK(0*slab + i),   blmc(1*slab + i));
                    diffK(1*slab + i)   = Kokkos::fmax(diffK(1*slab + i),   blmc(2*slab + i));
                } else {
                    ghats((size_t)n * (nl - 1) + nz) = 0.0;
                }
            }
        });

    /* ---- INTERNAL HALO BRACKET 2: pre-elem-average exchanges (:468-473) ------
     * combine wrote viscA/diffK/ghats on device → sync_host → halo (h_checked) →
     * sync_device viscA+diffK (needed on device for the elem average + Kv copy).
     * ghats is gated off in CORE2 (not read on device after) → left host. */
    k->diffK_fld.modify_device();
    k->viscA_fld.modify_device();
    k->ghats_fld.modify_device();
    /* M5.7a: diffK (2 slabs) + viscA via the device-halo (GPU-aware MPI on CUDA, exact host
     * bracket on Serial) — removes the sync_host → host-exchange → re-push round trip. Both are
     * read at HALO right below: viscA by the node→elem average (element vertices can be halo
     * nodes), diffK slab-0 by the single-Kv deep_copy. The slab offset is the M5.5b blmc recipe
     * (base_off = j*slab). */
    fesom_halo_field(k->diffK_fld, FESOM_HALO_NOD3D, nl, 1, partit, 0 * slab);
    fesom_halo_field(k->diffK_fld, FESOM_HALO_NOD3D, nl, 1, partit, 1 * slab);
    fesom_halo_field(k->viscA_fld, FESOM_HALO_NOD3D, nl, 1, partit);
    /* M5.21 (Lever 2a, L66): ghats (the KPP non-local transport term) is UNCONSUMED — the port has
     * use_kpp_nonlclflx=.false., so the tracer-diffusion non-local flux is skipped
     * (fesom_tracer_diff.cpp:19,291) and NOTHING reads ghats on host OR device after combine
     * (the kpp verify diffs only Kv/Av; the sole host reader is the FESOM_KPP_DUMP_DIR diagnostic).
     * Its per-step sync_host (255.8 MB/step D2H on NG5 — the #1 per-field PCIe driver, a blocking
     * fence in the mixing substep) + host halo exchange were a pure PLACEBO. Skipped in production;
     * kept ONLY when the dump diagnostic will read host ghats this step, so the dump stays exact.
     * Removing dead data changes NO computed field → BIT-IDENTICAL (Serial AND CUDA). ⚠️ RESTORE the
     * unconditional sync_host + exchange if the KPP non-local flux is ever ported/enabled — a real
     * ghats consumer would then exist (the L36/L39 "device consumer" rule). ghats stays
     * device-authoritative (modify_device above); the dump path syncs it on demand. */
    if (fesom_kpp_dump_this_step(s_kpp_call)) {
        k->ghats_fld.sync_host();
        fesom_exchange_nod3D(k->ghats_fld.h_checked(), nl - 1, partit);
    }

    /* node→elem viscAE = Σ(viscA over 3 vertices)/3 + bottom fill + minmix (:475-490) */
    Kokkos::parallel_for("kpp_viscAE", Kokkos::RangePolicy<>(0, mesh->myDim_elem2D),
        KOKKOS_LAMBDA(const int e) {
            int n0 = elnod(3*e + 0);
            int n1 = elnod(3*e + 1);
            int n2 = elnod(3*e + 2);
            int nzmin = ulev_e(e) - 1;
            int nzmax = nlev_e(e) - 1;
            for (int nz = nzmin; nz < nzmax; ++nz)
                Av(FESOM_ELEM3D(e, nz, nl)) =
                    ( viscA(FESOM_NODE3D(n0, nz, nl))
                    + viscA(FESOM_NODE3D(n1, nz, nl))
                    + viscA(FESOM_NODE3D(n2, nz, nl)) ) / 3.0;
            Av(FESOM_ELEM3D(e, nzmax, nl)) = Av(FESOM_ELEM3D(e, nzmax - 1, nl));
            if (Av(FESOM_ELEM3D(e, nzmin, nl)) < KPP_MINMIX)
                Av(FESOM_ELEM3D(e, nzmin, nl)) = KPP_MINMIX;
        });

    /* single Kv = diffK(:,:,1) (T-channel) over N — the array the TDMA reads (:512-514) */
    Kokkos::deep_copy(Kv, Kokkos::subview(diffK, Kokkos::make_pair((size_t)0, slab)));

    /* outputs now device-authoritative (the driver OUT rail sync_host()s them) */
    aux->Av_fld.modify_device();
    aux->Kv_fld.modify_device();

    if (fesom_kpp_dump_this_step(s_kpp_call)) {
        aux->Av_fld.sync_host(); aux->Kv_fld.sync_host();
        kpp_dump_final(k, aux, mesh, partit);
        aux->Av_fld.modify_host(); aux->Av_fld.sync_device();
        aux->Kv_fld.modify_host(); aux->Kv_fld.sync_device();
    }
}

/*--- FESOM_KK_VERIFY=kpp — in-binary per-kernel gate (M2.3b) ------------------
 * The EOS/PP verify shape (D19): aux holds the Kokkos production Av/Kv (the driver
 * sync_host()'d them). Snapshot, run the HOST C twin (fesom_kpp_mixing — recomputes
 * everything from the host-current inputs, which KPP does not modify except the
 * dbsfc surface it re-zeros identically; it clobbers k->scratch + aux->Av/Kv via the
 * raw alias, all recomputed next step), diff over the owned region, RESTORE the
 * Kokkos Av/Kv (non-intrusive), report + assert Serial max|Δ|==0. s_kpp_call is
 * saved/restored so the C twin's ++ does not double-count the step (dump gating).
 */
void fesom_kpp_verify(fesom_kpp                  *k,
                      struct fesom_aux           *aux,
                      const struct fesom_tracers *tracers,
                      const struct fesom_forcing *forcing,
                      const struct fesom_dyn     *dyn,
                      const struct fesom_mesh    *mesh,
                      struct fesom_partit        *partit,
                      int                         step_n)
{
    const int nl = mesh->nl;
    const size_t nKv = (size_t)k->n_nod * (size_t)nl;            /* aux.Kv == N*nl */
    const size_t nAv = (size_t)mesh->myDim_elem2D * (size_t)nl;  /* KPP writes owned elems */

    std::vector<real_t> kk_Kv(aux->Kv, aux->Kv + nKv);
    std::vector<real_t> kk_Av(aux->Av, aux->Av + nAv);

    int saved_call = s_kpp_call;
    fesom_kpp_mixing(k, aux, tracers, forcing, dyn, mesh, partit);   /* C twin via raw alias */
    s_kpp_call = saved_call;

    auto maxdiff = [](const std::vector<real_t> &kk, const real_t *c, size_t n) -> double {
        double m = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs((double)kk[i] - (double)c[i]);
            if (d > m) m = d;
        }
        return m;
    };
    double d_Kv = maxdiff(kk_Kv, aux->Kv, nKv);
    double d_Av = maxdiff(kk_Av, aux->Av, nAv);

    std::copy(kk_Kv.begin(), kk_Kv.end(), aux->Kv);   /* restore KK production result */
    std::copy(kk_Av.begin(), kk_Av.end(), aux->Av);

    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    const double dmax = std::max(d_Kv, d_Av);
    std::printf("[FESOM_KK_VERIFY=kpp] step %d backend=%s  max|Δ|: Kv=%.3e Av=%.3e\n",
                step_n, backend.c_str(), d_Kv, d_Av);
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=kpp] FAIL: Serial must be bit-identical to the "
                             "C twin (max|Δ|=%.3e)\n", dmax);
        std::abort();
    }
}

/*===========================================================================
 * Dump harness — FESOM_KPP_DUMP_DIR-gated, mirrors the EVP dump diagnostic
 * (reference_evp_dump_diagnostic). Off by default (one NULL check / run).
 *===========================================================================*/
static const char *s_kpp_dump_dir    = NULL;
static int         s_kpp_dump_loaded  = 0;
static int         s_kpp_dump_step    = 1;

static void kpp_dump_load_env(void)
{
    if (s_kpp_dump_loaded) return;
    s_kpp_dump_dir = getenv("FESOM_KPP_DUMP_DIR");
    const char *e = getenv("FESOM_KPP_DUMP_STEP");
    if (e) { int v = atoi(e); if (v > 0) s_kpp_dump_step = v; }
    s_kpp_dump_loaded = 1;
}

const char *fesom_kpp_dump_dir(void)
{
    kpp_dump_load_env();
    return s_kpp_dump_dir;
}

int fesom_kpp_dump_this_step(int step_n)
{
    kpp_dump_load_env();
    return s_kpp_dump_dir && (step_n == s_kpp_dump_step);
}

void fesom_kpp_dump_init(const fesom_kpp *k, int rank)
{
    kpp_dump_load_env();
    if (!s_kpp_dump_dir || rank != 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/kpp_init_rank0.txt", s_kpp_dump_dir);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "kpp_dump_init: cannot open %s\n", path); return; }
    fprintf(f, "# Vtc %.17g\n# cg %.17g\n# deltaz %.17g\n# deltau %.17g\n",
            k->Vtc, k->cg, k->deltaz, k->deltau);
    fprintf(f, "# i j wmt wst  (nni=%d nnj=%d)\n", FESOM_KPP_NNI, FESOM_KPP_NNJ);
    for (int i = 0; i <= FESOM_KPP_NNI + 1; ++i)
        for (int j = 0; j <= FESOM_KPP_NNJ + 1; ++j)
            fprintf(f, "%d %d %.17g %.17g\n", i, j,
                    k->wmt[KPP_TBL(i, j)], k->wst[KPP_TBL(i, j)]);
    fclose(f);
}

/* K2 wscale sweep dump: rank 0 evaluates wscale over a fixed (zehat, ustar)
 * grid spanning unstable-table / stable-branch / clamp regions; writes
 * <dir>/kpp_wscale_rank0.txt (`i j zehat ustar wm ws`). The Fortran gate uses
 * the identical grid; both compared vs scripts/kpp_wscale_reference.py. */
#define KPP_SWEEP_NZ 201
#define KPP_SWEEP_NU 101
void fesom_kpp_dump_wscale_sweep(const fesom_kpp *k, int rank)
{
    kpp_dump_load_env();
    if (!s_kpp_dump_dir || rank != 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/kpp_wscale_rank0.txt", s_kpp_dump_dir);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "kpp_dump_wscale: cannot open %s\n", path); return; }
    fprintf(f, "# i j zehat ustar wm ws  (sweep %dx%d)\n",
            KPP_SWEEP_NZ, KPP_SWEEP_NU);
    for (int i = 0; i < KPP_SWEEP_NZ; ++i) {
        real_t zehat = -1.0e-6 + (real_t)i * (2.0e-6 / (real_t)(KPP_SWEEP_NZ - 1));
        for (int j = 0; j < KPP_SWEEP_NU; ++j) {
            real_t ustar = (real_t)j * (0.05 / (real_t)(KPP_SWEEP_NU - 1));
            real_t wm, ws;
            kpp_wscale(k, zehat, ustar, &wm, &ws);
            fprintf(f, "%d %d %.17g %.17g %.17g %.17g\n", i, j,
                    zehat, ustar, wm, ws);
        }
    }
    fclose(f);
}

void fesom_kpp_dump_nodes(const struct fesom_mesh *mesh,
                          struct fesom_partit     *partit,
                          int step_n, const char *tag, int ncomp,
                          double (*get)(int node, int comp, void *user),
                          void *user)
{
    kpp_dump_load_env();
    if (!s_kpp_dump_dir) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/kpp_dump_s%d_%s_rank%d.txt",
             s_kpp_dump_dir, step_n, tag, partit->mype);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "kpp_dump: cannot open %s\n", path); return; }
    fprintf(f, "# step=%d tag=%s rank=%d N=%d ncomp=%d\n",
            step_n, tag, partit->mype, mesh->myDim_nod2D, ncomp);
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        fprintf(f, "%d", partit->myList_nod2D[n]);   /* 1-based gid */
        for (int c = 0; c < ncomp; ++c) fprintf(f, " %.17g", get(n, c, user));
        fputc('\n', f);
    }
    fclose(f);
}

void fesom_kpp_dump_elems(const struct fesom_mesh *mesh,
                          struct fesom_partit     *partit,
                          int step_n, const char *tag, int ncomp,
                          double (*get)(int elem, int comp, void *user),
                          void *user)
{
    kpp_dump_load_env();
    if (!s_kpp_dump_dir) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/kpp_dump_s%d_%s_rank%d.txt",
             s_kpp_dump_dir, step_n, tag, partit->mype);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "kpp_dump: cannot open %s\n", path); return; }
    fprintf(f, "# step=%d tag=%s rank=%d N=%d ncomp=%d\n",
            step_n, tag, partit->mype, mesh->myDim_elem2D, ncomp);
    for (int e = 0; e < mesh->myDim_elem2D; ++e) {
        fprintf(f, "%d", partit->myList_elem2D[e]);   /* 1-based element gid */
        for (int c = 0; c < ncomp; ++c) fprintf(f, " %.17g", get(e, c, user));
        fputc('\n', f);
    }
    fclose(f);
}
