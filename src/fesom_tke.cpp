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

/*===========================================================================
 * Task 1.2 scaffold: force instantiation of BOTH column-core specialisations so the
 * template body is fully type-checked at this commit. Never called; Task 1.3's driver
 * instantiates the core for real (and only a device call site exercises the nvcc DEVICE
 * compile — a host call like this one does not). Delete when the driver lands.
 *===========================================================================*/
[[maybe_unused]] static void fesom_tke_core_typecheck(void)
{
    real_t col[TKE_NL_MAX + 1] = {0.0};
    fesom_cvmix_tke_diag d{};
    fesom_cvmix_integrate_tke<false>(1, 1800.0, 1030.0, 9.81, col, col, col, col, col,
                                     col, col, col, col, 0.0, 0.0, 0.0,
                                     col, col, col, nullptr);
    fesom_cvmix_integrate_tke<true> (1, 1800.0, 1030.0, 9.81, col, col, col, col, col,
                                     col, col, col, col, 0.0, 0.0, 0.0,
                                     col, col, col, &d);
}

/*===========================================================================
 * Driver — Task 1.3 wires the body. Abort until then, so selecting TKE fails
 * loudly instead of silently running with no vertical mixing (the C staged it
 * the same way: "Abort stub until Phase T2 wires the body").
 *===========================================================================*/
void fesom_tke_mixing_kk(fesom_tke                  *t,
                         struct fesom_aux           *aux,
                         const struct fesom_forcing *forcing,
                         const struct fesom_dyn     *dyn,
                         const struct fesom_mesh    *mesh,
                         struct fesom_partit        *partit)
{
    (void)t; (void)aux; (void)forcing; (void)dyn; (void)mesh; (void)partit;
    std::fprintf(stderr, "fesom_tke_mixing_kk: not implemented yet (M6.1 Task 1.3). "
                         "FESOM_MIX_SCHEME=TKE is not usable at this commit.\n");
    std::abort();
}
