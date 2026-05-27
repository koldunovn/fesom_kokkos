/*
 * Sea-ice FCT advection. Literal port of ice_fct.F90.
 *
 * The Fortran nn_pos(k,row) / nn_num(row) lookup is reused via the SSH
 * stiffness CSR (stiff->rowptr / stiff->colind):
 *
 *   nn_pos(k, row)   ↔  stiff->colind[stiff->rowptr[row] + (k-1)]   (Fortran 1-based)
 *   nn_num(row)      ↔  stiff->rowptr[row+1] - stiff->rowptr[row]
 *   ssh_stiff%rowptr(row) - ssh_stiff%rowptr(1) + 1   ↔  stiff->rowptr[row]   (0-based start)
 *
 * fct_massmatrix shares the ssh_stiff sparsity pattern; allocated to
 * stiff->nnz here.
 *
 * Halo audit (per feedback_write_loops_halo.md and
 * feedback_array_size_vs_reader_loop.md). Each write-loop annotates its
 * Fortran source line and bound. The bound is preserved literally: do NOT
 * "fix" a Fortran myDim_nod2D loop into myDim+eDim — Fortran's downstream
 * readers stay at myDim too, and changing the bound would diverge.
 */

#include "fesom_ice_fct.h"
#include "fesom_constants.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   // M5.1: GPU-aware-MPI on-device halo (fesom_halo_field)
#include "fesom_ice.h"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_ssh.h"
#include "fesom_types.h"

#include <Kokkos_Core.hpp>   // M4.3c: device FCT kernels (parallel_for + atomic_add scatters)
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>            // M4.3c: FESOM_KK_VERIFY backend name
#include <algorithm>         // M4.3c: verify snapshot/restore copies
#include <vector>

#ifndef FESOM_ICE_SCALE_AREA
#define FESOM_ICE_SCALE_AREA 2.0e8   /* oce_modules.F90:29 — global, used by ice_diff scaling */
#endif

/* Logical tracer index used by ice_fem_fct — mirrors Fortran tr_array_id
 * but 0-based here. The Fortran/C mapping:
 *   logical 0 (Fortran 1) → m_ice  (data[FESOM_ICE_MICE])
 *   logical 1 (Fortran 2) → a_ice  (data[FESOM_ICE_AICE])
 *   logical 2 (Fortran 3) → m_snow (data[FESOM_ICE_MSNOW])
 * Caller (fct_solve) calls in this exact order — same as Fortran. */
enum { FCT_LOG_M = 0, FCT_LOG_A = 1, FCT_LOG_MS = 2 };

static int data_idx_for_logical(int log_id)
{
    switch (log_id) {
        case FCT_LOG_M:  return FESOM_ICE_MICE;
        case FCT_LOG_A:  return FESOM_ICE_AICE;
        case FCT_LOG_MS: return FESOM_ICE_MSNOW;
        default:
            fprintf(stderr, "fesom_ice_fct: bad logical tracer id %d\n", log_id);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
    }
}

/* ============================================================ */
/* E1 — ice_mass_matrix_fill (ice_fct.F90:1145)                 */
/* ============================================================ */

void fesom_ice_mass_matrix_fill(fesom_ice                    *ice,
                                const struct fesom_ssh_stiff *stiff,
                                struct fesom_partit          *partit,
                                const struct fesom_mesh      *mesh)
{
    (void)partit;

    /* (Re-)allocate to ssh_stiff->nnz, zero-initialised. Fortran allocates
     * with the same shape via mass_matrix=0 then accumulates. */
    if (ice->work.fct_massmatrix == NULL) {
        // M1.4: Field owns the storage; raw ptr is a non-owning alias = field.h() (D12). .alloc
        // zero-inits like calloc and aborts on OOM. The Field is released by fesom_ice_free's
        // *ice = fesom_ice{}. The == NULL guard still works: the raw alias is NULL until alloc'd.
        ice->work.fct_massmatrix_fld.alloc("ice.work.fct_massmatrix", (size_t)stiff->nnz);
        ice->work.fct_massmatrix = ice->work.fct_massmatrix_fld.h();
    } else {
        memset(ice->work.fct_massmatrix, 0, (size_t)stiff->nnz * sizeof(real_t));
    }

    real_t *mm = ice->work.fct_massmatrix;
    const int N_own = mesh->myDim_nod2D;

    /* --- a) accumulate (ice_fct.F90:1170-1210) ------------------------------
     * Fortran outer DO elem=1, myDim_elem2D; inner DO n=1,3 (vertices);
     * skip if row > myDim_nod2D (halo); inner DO q=1,3 with cavity skip.
     * Bound: myDim_elem2D — matches Fortran exactly.                       */
    for (int elem = 0; elem < mesh->myDim_elem2D; ++elem) {
        const int en[3] = { mesh->elem_nodes[3*elem + 0],
                            mesh->elem_nodes[3*elem + 1],
                            mesh->elem_nodes[3*elem + 2] };
        for (int nl = 0; nl < 3; ++nl) {
            int row = en[nl];
            if (row >= N_own) continue;     /* skip halo row (Fortran line 1177) */
            int rstart = stiff->rowptr[row];
            int rend   = stiff->rowptr[row + 1];

            for (int q = 0; q < 3; ++q) {
                if (mesh->ulevels[elem] > 1) continue;   /* cavity (line 1186) */

                /* find ipos: position of en[q] in row's sparsity pattern */
                int ipos = -1;
                for (int k = rstart; k < rend; ++k) {
                    if (stiff->colind[k] == en[q]) { ipos = k; break; }
                }
                if (ipos < 0) {
                    fprintf(stderr,
                            "fesom_ice_mass_matrix_fill: FATAL — row=%d en[q=%d]=%d not in CSR\n",
                            row, q, en[q]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                mm[ipos] += mesh->elem_area[elem] / 12.0;
                if (q == nl) {
                    mm[ipos] += mesh->elem_area[elem] / 12.0;
                }
            }
        }
    }

    /* --- b) consistency check (ice_fct.F90:1213-1248) -----------------------
     * sum of row entries = area(ulevels_nod2D(q), q). Print warning if any
     * row mismatches by > 0.1. Fortran prints rank/index then continues. */
    for (int q = 0; q < N_own; ++q) {
        int ul = mesh->ulevels_nod2D[q];   /* 1-based level */
        if (ul > 1) continue;              /* cavity */
        int rstart = stiff->rowptr[q];
        int rend   = stiff->rowptr[q + 1];
        real_t aa = 0.0;
        for (int k = rstart; k < rend; ++k) aa += mm[k];
        real_t a_node = mesh->area[q * mesh->nl + (ul - 1)];   /* area(ul, q) */
        if (fabs(a_node - aa) > 0.1) {
            fprintf(stderr,
                    "#### MASS MATRIX PROBLEM rank=%d q=%d aa=%g area=%g ul=%d\n",
                    partit ? partit->mype : 0, q, (double)aa, (double)a_node, ul);
        }
    }

    /* M4.3c: one-shot push of the set-once mass matrix to the device (the M4.2-a CSR pattern,
     * fesom_ssh.cpp:280). fct_massmatrix shares the ssh_stiff sparsity and is never updated after
     * this fill, so the device copy stays valid for the whole run. The ssh_stiff rowptr/colind/
     * values are already device-current from fesom_ssh_preconditioner (called before this in
     * fesom_main). The Field is filled above via the raw host alias (L14) → modify_host() first. */
    ice->work.fct_massmatrix_fld.modify_host();
    ice->work.fct_massmatrix_fld.sync_device();
}

/* ============================================================ */
/* E2 — ice_TG_rhs (ice_fct.F90:91)                             */
/* ============================================================ */

void fesom_ice_tg_rhs(fesom_ice                *ice,
                      struct fesom_partit      *partit,
                      const struct fesom_mesh  *mesh)
{
    (void)partit;

    real_t *u_ice  = ice->uice;
    real_t *v_ice  = ice->vice;
    real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;
    real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    real_t *m_snow = ice->data[FESOM_ICE_MSNOW].values;
    real_t *rhs_a  = ice->data[FESOM_ICE_AICE].values_rhs;
    real_t *rhs_m  = ice->data[FESOM_ICE_MICE].values_rhs;
    real_t *rhs_ms = ice->data[FESOM_ICE_MSNOW].values_rhs;

    const real_t dt   = ice->ice_dt;
    const real_t idiff = ice->ice_diff;
    const real_t scale = (real_t)FESOM_ICE_SCALE_AREA;

    /* zero rhs (ice_fct.F90:137-144) — bound: myDim_nod2D only.
     * Halo rhs entries are intentionally NOT zeroed — Fortran does the same;
     * downstream consumers (solve_low/high_order) read rhs only at myDim. */
    for (int row = 0; row < mesh->myDim_nod2D; ++row) {
        rhs_m [row] = 0.0;
        rhs_a [row] = 0.0;
        rhs_ms[row] = 0.0;
    }

    /* assemble (ice_fct.F90:159-198) — bound: myDim_elem2D; cavity skip.
     * Writes to rhs_*[row] where row is an elnode (can be halo); halo
     * accumulations are stale-summed but harmless (consumers read myDim). */
    for (int elem = 0; elem < mesh->myDim_elem2D; ++elem) {
        if (mesh->ulevels[elem] > 1) continue;          /* cavity (line 163) */

        const int en[3] = { mesh->elem_nodes[3*elem + 0],
                            mesh->elem_nodes[3*elem + 1],
                            mesh->elem_nodes[3*elem + 2] };
        const real_t *g  = &mesh->gradient_sca[6 * elem];
        const real_t dx[3] = { g[0], g[1], g[2] };
        const real_t dy[3] = { g[3], g[4], g[5] };
        const real_t vol  = mesh->elem_area[elem];
        /* Fortran: um = sum(U_ice(elnodes)) — NOT divided by 3 (line 171) */
        const real_t um = u_ice[en[0]] + u_ice[en[1]] + u_ice[en[2]];
        const real_t vm = v_ice[en[0]] + v_ice[en[1]] + v_ice[en[2]];
        const real_t diff = idiff * sqrt(vol / scale);

        for (int n = 0; n < 3; ++n) {
            int row = en[n];
            real_t entries[3];
            for (int q = 0; q < 3; ++q) {
                /* Fortran line 184-187 */
                real_t a = (dx[n] * (um + u_ice[en[q]])
                          + dy[n] * (vm + v_ice[en[q]])) / 12.0;
                real_t b = diff * (dx[n] * dx[q] + dy[n] * dy[q]);
                real_t c = 0.5 * dt
                         * (um * dx[n] + vm * dy[n])
                         * (um * dx[q] + vm * dy[q]) / 9.0;
                entries[q] = vol * dt * (a - b - c);
            }
            real_t sm  = entries[0] * m_ice [en[0]]
                       + entries[1] * m_ice [en[1]]
                       + entries[2] * m_ice [en[2]];
            real_t sa  = entries[0] * a_ice [en[0]]
                       + entries[1] * a_ice [en[1]]
                       + entries[2] * a_ice [en[2]];
            real_t sms = entries[0] * m_snow[en[0]]
                       + entries[1] * m_snow[en[1]]
                       + entries[2] * m_snow[en[2]];
            rhs_m [row] += sm;
            rhs_a [row] += sa;
            rhs_ms[row] += sms;
        }
    }
}

/* ============================================================ */
/* E3 — ice_solve_low_order (ice_fct.F90:243)                   */
/* ============================================================ */

static void ice_solve_low_order(fesom_ice                    *ice,
                                const struct fesom_ssh_stiff *stiff,
                                struct fesom_partit          *partit,
                                const struct fesom_mesh      *mesh)
{
    real_t *a_ice  = ice->data[FESOM_ICE_AICE].values;
    real_t *m_ice  = ice->data[FESOM_ICE_MICE].values;
    real_t *m_snow = ice->data[FESOM_ICE_MSNOW].values;
    real_t *rhs_a  = ice->data[FESOM_ICE_AICE].values_rhs;
    real_t *rhs_m  = ice->data[FESOM_ICE_MICE].values_rhs;
    real_t *rhs_ms = ice->data[FESOM_ICE_MSNOW].values_rhs;
    real_t *a_l    = ice->data[FESOM_ICE_AICE].valuesl;
    real_t *m_l    = ice->data[FESOM_ICE_MICE].valuesl;
    real_t *ms_l   = ice->data[FESOM_ICE_MSNOW].valuesl;
    const real_t  *mm = ice->work.fct_massmatrix;

    const real_t gamma = ice->ice_gamma_fct;
    const int    nl    = mesh->nl;

    /* Fortran line 304: do row=1, myDim_nod2D — myDim only. */
    for (int row = 0; row < mesh->myDim_nod2D; ++row) {
        if (mesh->ulevels_nod2D[row] > 1) continue;        /* cavity (line 307) */

        int rs = stiff->rowptr[row];
        int re = stiff->rowptr[row + 1];

        real_t sm = 0.0, sa = 0.0, sms = 0.0;
        for (int k = rs; k < re; ++k) {
            int j = stiff->colind[k];
            sm  += mm[k] * m_ice [j];
            sa  += mm[k] * a_ice [j];
            sms += mm[k] * m_snow[j];
        }
        /* area(1, row) — Fortran is (1, row) i.e. surface level (1-based);
         * our area[] is flat [n*nl + level], 0-based. Surface = level 0. */
        real_t inv_area = 1.0 / mesh->area[row * nl + 0];
        m_l [row] = (rhs_m [row] + gamma * sm ) * inv_area + (1.0 - gamma) * m_ice [row];
        a_l [row] = (rhs_a [row] + gamma * sa ) * inv_area + (1.0 - gamma) * a_ice [row];
        ms_l[row] = (rhs_ms[row] + gamma * sms) * inv_area + (1.0 - gamma) * m_snow[row];
    }

    /* Fortran line 335: exchange_nod(m_icel, a_icel, m_snowl) */
    fesom_exchange_nod2D(m_l,  partit);
    fesom_exchange_nod2D(a_l,  partit);
    fesom_exchange_nod2D(ms_l, partit);
}

/* ============================================================ */
/* E3 — ice_solve_high_order (ice_fct.F90:347)                  */
/* ============================================================ */

static void ice_solve_high_order(fesom_ice                    *ice,
                                 const struct fesom_ssh_stiff *stiff,
                                 struct fesom_partit          *partit,
                                 const struct fesom_mesh      *mesh)
{
    real_t *rhs_a  = ice->data[FESOM_ICE_AICE].values_rhs;
    real_t *rhs_m  = ice->data[FESOM_ICE_MICE].values_rhs;
    real_t *rhs_ms = ice->data[FESOM_ICE_MSNOW].values_rhs;
    real_t *a_l    = ice->data[FESOM_ICE_AICE].valuesl;
    real_t *m_l    = ice->data[FESOM_ICE_MICE].valuesl;
    real_t *ms_l   = ice->data[FESOM_ICE_MSNOW].valuesl;
    real_t *da     = ice->data[FESOM_ICE_AICE].dvalues;
    real_t *dm     = ice->data[FESOM_ICE_MICE].dvalues;
    real_t *dms    = ice->data[FESOM_ICE_MSNOW].dvalues;
    const real_t  *mm = ice->work.fct_massmatrix;

    const int nl = mesh->nl;
    const int num_iter_solve = 3;   /* Fortran line 362 */

    /* First approximation (lines 400-410): myDim_nod2D bound */
    for (int row = 0; row < mesh->myDim_nod2D; ++row) {
        if (mesh->ulevels_nod2D[row] > 1) continue;
        real_t inv_area = 1.0 / mesh->area[row * nl + 0];
        dm [row] = rhs_m [row] * inv_area;
        da [row] = rhs_a [row] * inv_area;
        dms[row] = rhs_ms[row] * inv_area;
    }
    fesom_exchange_nod2D(dm,  partit);
    fesom_exchange_nod2D(da,  partit);
    fesom_exchange_nod2D(dms, partit);

    /* Iterate (lines 426-488): num_iter_solve - 1 = 2 passes */
    for (int it = 0; it < num_iter_solve - 1; ++it) {
        /* (a) update valuesl  (lines 433-452) — myDim_nod2D bound */
        for (int row = 0; row < mesh->myDim_nod2D; ++row) {
            if (mesh->ulevels_nod2D[row] > 1) continue;
            int rs = stiff->rowptr[row];
            int re = stiff->rowptr[row + 1];

            real_t sm = 0.0, sa = 0.0, sms = 0.0;
            for (int k = rs; k < re; ++k) {
                int j = stiff->colind[k];
                sm  += mm[k] * dm [j];
                sa  += mm[k] * da [j];
                sms += mm[k] * dms[j];
            }
            real_t inv_area = 1.0 / mesh->area[row * nl + 0];
            real_t rhs_new = rhs_m[row] - sm;
            m_l [row] = dm[row] + rhs_new * inv_area;
            rhs_new = rhs_a[row] - sa;
            a_l [row] = da[row] + rhs_new * inv_area;
            rhs_new = rhs_ms[row] - sms;
            ms_l[row] = dms[row] + rhs_new * inv_area;
        }

        /* (b) copy valuesl back to dvalues  (lines 464-473) — myDim_nod2D */
        for (int row = 0; row < mesh->myDim_nod2D; ++row) {
            if (mesh->ulevels_nod2D[row] > 1) continue;
            dm [row] = m_l [row];
            da [row] = a_l [row];
            dms[row] = ms_l[row];
        }

        /* (c) exchange (line 481) */
        fesom_exchange_nod2D(dm,  partit);
        fesom_exchange_nod2D(da,  partit);
        fesom_exchange_nod2D(dms, partit);
    }
}

/* ============================================================ */
/* E4 — ice_fem_fct (ice_fct.F90:498) — Zalesak limiter         */
/* ============================================================ */

static void ice_fem_fct(int                           log_id,
                        fesom_ice                    *ice,
                        const struct fesom_ssh_stiff *stiff,
                        struct fesom_partit          *partit,
                        const struct fesom_mesh      *mesh)
{
    /* Resolve the active tracer triple (values, valuesl, dvalues).
     * Fortran uses tr_array_id ∈ {1,2,3} with the mapping
     *   1 → m_ice, 2 → a_ice, 3 → m_snow
     * (NB: this is *not* the data() array index in Fortran either; it's a
     * logical tracer id). We mirror with FCT_LOG_M / FCT_LOG_A / FCT_LOG_MS. */
    const int  d_active  = data_idx_for_logical(log_id);
    real_t    *vals      = ice->data[d_active].values;     /* m_ice / a_ice / m_snow */
    real_t    *vals_l    = ice->data[d_active].valuesl;    /* low-order */
    real_t    *dvals     = ice->data[d_active].dvalues;    /* high-order increment */

    real_t    *icefluxes = ice->work.fct_fluxes;           /* [elem*3] */
    real_t    *icepplus  = ice->work.fct_plus;
    real_t    *icepminus = ice->work.fct_minus;
    real_t    *tmax      = ice->work.fct_tmax;
    real_t    *tmin      = ice->work.fct_tmin;

    const real_t gamma = ice->ice_gamma_fct;
    const int N_full = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int nl     = mesh->nl;

    /* Init tmax/tmin (lines 562-565) — myDim+eDim. */
    for (int n = 0; n < N_full; ++n) { tmax[n] = 0.0; tmin[n] = 0.0; }

    /* icoef(3,3) Fortran: 1 everywhere except diag = -2 (lines 575-582). */
    static const real_t icoef[3][3] = {
        { -2.0,  1.0,  1.0 },
        {  1.0, -2.0,  1.0 },
        {  1.0,  1.0, -2.0 },
    };

    /* Antidiffusive fluxes per element (lines 594-633) — myDim_elem2D, cavity skip.
     * icefluxes is sized myDim+eDim+eXDim; we only write myDim_elem2D, halo
     * entries left untouched (matches Fortran). */
    for (int elem = 0; elem < mesh->myDim_elem2D; ++elem) {
        if (mesh->ulevels[elem] > 1) continue;
        const int en[3] = { mesh->elem_nodes[3*elem + 0],
                            mesh->elem_nodes[3*elem + 1],
                            mesh->elem_nodes[3*elem + 2] };
        const real_t vol = mesh->elem_area[elem];

        /* For each q (vertex): flux_q = -Σ_n icoef(n,q) * (γ·val + dval)(en[n])
         *                              * vol / area(1, en[q]) / 12 */
        for (int q = 0; q < 3; ++q) {
            real_t s = 0.0;
            for (int n = 0; n < 3; ++n) {
                real_t v = gamma * vals[en[n]] + dvals[en[n]];
                s += icoef[n][q] * v;
            }
            real_t inv_area_q = 1.0 / mesh->area[en[q] * nl + 0];
            icefluxes[elem * 3 + q] = -s * (vol * inv_area_q) / 12.0;
        }
    }

    /* Cluster min/max (lines 654-718) — myDim_nod2D, cavity skip.
     * Only the active tracer is touched (Fortran has 3 separate if-blocks;
     * our log_id selects). */
    for (int row = 0; row < mesh->myDim_nod2D; ++row) {
        if (mesh->ulevels_nod2D[row] > 1) continue;
        int rs = stiff->rowptr[row];
        int re = stiff->rowptr[row + 1];

        real_t lo_l = +INFINITY, hi_l = -INFINITY;
        real_t lo_v = +INFINITY, hi_v = -INFINITY;
        for (int k = rs; k < re; ++k) {
            int j = stiff->colind[k];
            real_t a = vals_l[j], b = vals[j];
            if (a > hi_l) hi_l = a;
            if (a < lo_l) lo_l = a;
            if (b > hi_v) hi_v = b;
            if (b < lo_v) lo_v = b;
        }
        real_t hi = hi_l > hi_v ? hi_l : hi_v;
        real_t lo = lo_l < lo_v ? lo_l : lo_v;
        tmax[row] = hi - vals_l[row];
        tmin[row] = lo - vals_l[row];
    }

    /* Init icepplus/icepminus (lines 751-754) — myDim+eDim */
    for (int n = 0; n < N_full; ++n) { icepplus[n] = 0.0; icepminus[n] = 0.0; }

    /* Sum positive/negative fluxes per node (lines 770-805) — myDim_elem2D,
     * cavity skip, scatter to elnodes (which can be halo). icepplus/icepminus
     * have halo coverage and the exchange below sweeps owner sums. */
    for (int elem = 0; elem < mesh->myDim_elem2D; ++elem) {
        if (mesh->ulevels[elem] > 1) continue;
        const int en[3] = { mesh->elem_nodes[3*elem + 0],
                            mesh->elem_nodes[3*elem + 1],
                            mesh->elem_nodes[3*elem + 2] };
        for (int q = 0; q < 3; ++q) {
            int n = en[q];
            real_t flux = icefluxes[elem * 3 + q];
            if (flux > 0.0) icepplus [n] += flux;
            else            icepminus[n] += flux;
        }
    }

    /* Correction factors (lines 824-841) — myDim_nod2D, cavity skip */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;
        real_t flux = icepplus[n];
        if (fabs(flux) > 0.0) {
            real_t denom = flux > 1e-12 ? flux : 1e-12;
            real_t v = tmax[n] / denom;
            icepplus[n] = v < 1.0 ? v : 1.0;
        } else {
            icepplus[n] = 0.0;
        }
        flux = icepminus[n];
        if (fabs(flux) > 0.0) {
            real_t denom = flux < -1e-12 ? flux : -1e-12;
            real_t v = tmin[n] / denom;
            icepminus[n] = v < 1.0 ? v : 1.0;
        } else {
            icepminus[n] = 0.0;
        }
    }
    /* Fortran line 853: exchange_nod(icepminus, icepplus) — halo filled. */
    fesom_exchange_nod2D(icepplus,  partit);
    fesom_exchange_nod2D(icepminus, partit);

    /* Limit element fluxes (lines 865-879) — myDim_elem2D, cavity skip.
     * Reads icepplus/icepminus at halo elnodes, which are valid post-exchange. */
    for (int elem = 0; elem < mesh->myDim_elem2D; ++elem) {
        if (mesh->ulevels[elem] > 1) continue;
        const int en[3] = { mesh->elem_nodes[3*elem + 0],
                            mesh->elem_nodes[3*elem + 1],
                            mesh->elem_nodes[3*elem + 2] };
        real_t ae = 1.0;
        for (int q = 0; q < 3; ++q) {
            int n = en[q];
            real_t flux = icefluxes[elem * 3 + q];
            real_t cand = (flux >= 0.0) ? icepplus[n] : icepminus[n];
            if (cand < ae) ae = cand;
        }
        icefluxes[elem * 3 + 0] *= ae;
        icefluxes[elem * 3 + 1] *= ae;
        icefluxes[elem * 3 + 2] *= ae;
    }

    /* Apply update (lines 893-947, plus the analogous blocks for tr_array_id 2,3).
     * Two-phase: first overwrite values <- valuesl on myDim, then accumulate
     * limited fluxes from elements (myDim_elem2D), then exchange owners. */
    for (int n = 0; n < mesh->myDim_nod2D; ++n) {
        if (mesh->ulevels_nod2D[n] > 1) continue;
        vals[n] = vals_l[n];
    }
    for (int elem = 0; elem < mesh->myDim_elem2D; ++elem) {
        if (mesh->ulevels[elem] > 1) continue;
        const int en[3] = { mesh->elem_nodes[3*elem + 0],
                            mesh->elem_nodes[3*elem + 1],
                            mesh->elem_nodes[3*elem + 2] };
        for (int q = 0; q < 3; ++q) {
            vals[en[q]] += icefluxes[elem * 3 + q];
        }
    }
    /* Note: Fortran's outer exchange_nod(m_ice, a_ice, m_snow) at line 1130
     * happens after ALL three fem_fct calls. We emit the per-tracer exchange
     * here too because each tracer's `vals` is independent and the next
     * fem_fct call may read a different tracer's halo (it doesn't, but the
     * post-driver exchange in Fortran covers all three regardless). */
    fesom_exchange_nod2D(vals, partit);
}

/* ============================================================ */
/* E5 — ice_fct_solve driver (ice_fct.F90:210)                  */
/* ============================================================ */

void fesom_ice_fct_solve(fesom_ice                    *ice,
                         const struct fesom_ssh_stiff *stiff,
                         struct fesom_partit          *partit,
                         const struct fesom_mesh      *mesh)
{
    /* Fortran order:
     *   1. ice_solve_high_order  (uses valuesl as scratch — must precede low_order)
     *   2. ice_solve_low_order
     *   3. ice_fem_fct(1)  ! m_ice
     *   4. ice_fem_fct(2)  ! a_ice
     *   5. ice_fem_fct(3)  ! m_snow
     */
    ice_solve_high_order(ice, stiff, partit, mesh);
    ice_solve_low_order (ice, stiff, partit, mesh);

    ice_fem_fct(FCT_LOG_M,  ice, stiff, partit, mesh);
    ice_fem_fct(FCT_LOG_A,  ice, stiff, partit, mesh);
    ice_fem_fct(FCT_LOG_MS, ice, stiff, partit, mesh);
}

/* ======================================================================== *
 *  M4.3c — DEVICE (Kokkos) twins of the sea-ice FCT advection. The M2.6     *
 *  ocean-FCT analogue, but 2-D (single surface layer) so much simpler:      *
 *    - tg_rhs           = zero-map + element→node SCATTER into values_rhs    *
 *    - solve_low_order  = per-row CSR gather (ssh_stiff sparsity + the set-  *
 *                         once fct_massmatrix) + map, then 3 valuesl halos    *
 *    - solve_high_order = first-approx map + a HOST loop over 2 device       *
 *                         sweeps (CSR gather + map + copy) with per-iter      *
 *                         dvalues halos — the CG/EVP host-loop pattern        *
 *    - fem_fct (×3)     = the Zalesak limiter: init maps, per-elem antidiff   *
 *                         fluxes (race-free), cluster min/max (CSR gather),    *
 *                         icepplus/icepminus SCATTER, correction map,          *
 *                         icepplus/icepminus halos, limit map, apply           *
 *                         (overwrite-map + vals SCATTER), vals halo            *
 *  The 3 element→node SCATTERS (tg_rhs assemble, fem_fct's +/- sum, fem_fct's *
 *  final vals update) use Kokkos::atomic_add in natural element order (D22 →  *
 *  Serial bit-identical, OpenMP/CUDA climate-close). All halos are D21         *
 *  brackets OWNED by the FCT (the §6 pattern). The ssh_stiff CSR rowptr/colind *
 *  is device-current from fesom_ssh_preconditioner; fct_massmatrix from its    *
 *  one-shot push in fesom_ice_mass_matrix_fill. ⚠️ FORCED-ONLY → the verify    *
 *  is meaningful only on CORE2 (L42/L43).                                      *
 * ======================================================================== */

using DV  = fesom::Field::dev_view_t;
using IDV = fesom::IntField::dev_view_t;

/* D21 internal-halo bracket on a nod2D Field: a device kernel wrote f's OWNED rows;
 * make the halo current for the next device reader. M5.1: fesom_halo_field dispatches
 * to GPU-aware-MPI on-device exchange under CUDA (no full-field PCIe sync); host-staged
 * on Serial/OpenMP. modify_device() always (the driver's OUT sync_host copies the result
 * at np==1 too); the halo only at np>1 (eDim_nod2D==0 at np==1 so there is no halo). */
static inline void fct_halo_nod2D(fesom::Field &f, struct fesom_partit *partit)
{
    fesom_halo_field(f, FESOM_HALO_NOD2D, 1, 1, partit);
}

/* ---- E2 device twin: ice_TG_rhs (ice_fct.F90:91) ---------------------------- */
void fesom_ice_tg_rhs_kk(fesom_ice                *ice,
                         struct fesom_partit      *partit,
                         const struct fesom_mesh  *mesh)
{
    (void)partit;
    const int    myDim = mesh->myDim_nod2D;
    const int    E     = mesh->myDim_elem2D;
    const real_t dt    = ice->ice_dt;
    const real_t idiff = ice->ice_diff;
    const real_t scale = (real_t)FESOM_ICE_SCALE_AREA;

    auto u_ice  = ice->uice_fld.d();
    auto v_ice  = ice->vice_fld.d();
    auto a_ice  = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto m_ice  = ice->data[FESOM_ICE_MICE].values_fld.d();
    auto m_snow = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    auto rhs_a  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    auto rhs_m  = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    auto rhs_ms = ice->data[FESOM_ICE_MSNOW].values_rhs_fld.d();
    auto en     = mesh->elem_nodes_fld.d();
    auto gs     = mesh->gradient_sca_fld.d();
    auto ea     = mesh->elem_area_fld.d();
    auto ulev   = mesh->ulevels_fld.d();

    /* zero rhs over OWNED only (C bound myDim; halo rhs is scattered-but-unused, L37). */
    Kokkos::parallel_for("ice_tg_rhs_zero", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int row) { rhs_m(row)=0.0; rhs_a(row)=0.0; rhs_ms(row)=0.0; });

    /* assemble: element→node SCATTER (atomic_add, D22), cavity skip. */
    Kokkos::parallel_for("ice_tg_rhs_assemble", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int elem) {
            if (ulev(elem) > 1) return;                       /* cavity (C line 178) */
            const int    nn[3] = { en(3*elem+0), en(3*elem+1), en(3*elem+2) };
            const int    g  = 6*elem;
            const real_t dx[3] = { gs(g+0), gs(g+1), gs(g+2) };
            const real_t dy[3] = { gs(g+3), gs(g+4), gs(g+5) };
            const real_t vol = ea(elem);
            const real_t uq[3]  = { u_ice(nn[0]),  u_ice(nn[1]),  u_ice(nn[2])  };
            const real_t vq[3]  = { v_ice(nn[0]),  v_ice(nn[1]),  v_ice(nn[2])  };
            const real_t mq[3]  = { m_ice(nn[0]),  m_ice(nn[1]),  m_ice(nn[2])  };
            const real_t aq[3]  = { a_ice(nn[0]),  a_ice(nn[1]),  a_ice(nn[2])  };
            const real_t msq[3] = { m_snow(nn[0]), m_snow(nn[1]), m_snow(nn[2]) };
            const real_t um = uq[0] + uq[1] + uq[2];          /* sum, NOT /3 (C line 188) */
            const real_t vm = vq[0] + vq[1] + vq[2];
            const real_t diff = idiff * Kokkos::sqrt(vol / scale);
            for (int n = 0; n < 3; ++n) {
                real_t entries[3];
                for (int q = 0; q < 3; ++q) {
                    real_t a = (dx[n]*(um + uq[q]) + dy[n]*(vm + vq[q])) / 12.0;
                    real_t b = diff * (dx[n]*dx[q] + dy[n]*dy[q]);
                    real_t c = 0.5 * dt * (um*dx[n] + vm*dy[n]) * (um*dx[q] + vm*dy[q]) / 9.0;
                    entries[q] = vol * dt * (a - b - c);
                }
                real_t sm  = entries[0]*mq[0]  + entries[1]*mq[1]  + entries[2]*mq[2];
                real_t sa  = entries[0]*aq[0]  + entries[1]*aq[1]  + entries[2]*aq[2];
                real_t sms = entries[0]*msq[0] + entries[1]*msq[1] + entries[2]*msq[2];
                Kokkos::atomic_add(&rhs_m (nn[n]), sm);
                Kokkos::atomic_add(&rhs_a (nn[n]), sa);
                Kokkos::atomic_add(&rhs_ms(nn[n]), sms);
            }
        });
    ice->data[FESOM_ICE_AICE].values_rhs_fld.modify_device();
    ice->data[FESOM_ICE_MICE].values_rhs_fld.modify_device();
    ice->data[FESOM_ICE_MSNOW].values_rhs_fld.modify_device();
    /* No halo: the C ice_TG_rhs does not exchange rhs (consumers read OWNED only). */
}

/* ---- E3 device twin: ice_solve_low_order (ice_fct.F90:243) ------------------ */
static void ice_solve_low_order_kk(fesom_ice                    *ice,
                                   const struct fesom_ssh_stiff *stiff,
                                   struct fesom_partit          *partit,
                                   const struct fesom_mesh      *mesh)
{
    const int    myDim = mesh->myDim_nod2D;
    const int    nl    = mesh->nl;
    const real_t gamma = ice->ice_gamma_fct;

    auto rowptr = stiff->rowptr_fld.d();
    auto colind = stiff->colind_fld.d();
    auto mm     = ice->work.fct_massmatrix_fld.d();
    auto a_ice  = ice->data[FESOM_ICE_AICE].values_fld.d();
    auto m_ice  = ice->data[FESOM_ICE_MICE].values_fld.d();
    auto m_snow = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    auto rhs_a  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    auto rhs_m  = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    auto rhs_ms = ice->data[FESOM_ICE_MSNOW].values_rhs_fld.d();
    auto a_l    = ice->data[FESOM_ICE_AICE].valuesl_fld.d();
    auto m_l    = ice->data[FESOM_ICE_MICE].valuesl_fld.d();
    auto ms_l   = ice->data[FESOM_ICE_MSNOW].valuesl_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto area   = mesh->area_fld.d();

    Kokkos::parallel_for("ice_lo_solve", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int row) {
            if (ulev_n(row) > 1) return;                      /* cavity (C line 246) */
            const int rs = rowptr(row), re = rowptr(row + 1);
            real_t sm = 0.0, sa = 0.0, sms = 0.0;
            for (int k = rs; k < re; ++k) {
                const int j = colind(k);
                sm  += mm(k) * m_ice (j);
                sa  += mm(k) * a_ice (j);
                sms += mm(k) * m_snow(j);
            }
            const real_t inv_area = 1.0 / area(row*nl + 0);
            m_l (row) = (rhs_m (row) + gamma*sm ) * inv_area + (1.0 - gamma) * m_ice (row);
            a_l (row) = (rhs_a (row) + gamma*sa ) * inv_area + (1.0 - gamma) * a_ice (row);
            ms_l(row) = (rhs_ms(row) + gamma*sms) * inv_area + (1.0 - gamma) * m_snow(row);
        });

    /* D21 halo: fem_fct's cluster min/max reads valuesl at CSR-neighbour (halo) nodes. */
    fct_halo_nod2D(ice->data[FESOM_ICE_MICE].valuesl_fld,  partit);
    fct_halo_nod2D(ice->data[FESOM_ICE_AICE].valuesl_fld,  partit);
    fct_halo_nod2D(ice->data[FESOM_ICE_MSNOW].valuesl_fld, partit);
}

/* ---- E3 device twin: ice_solve_high_order (ice_fct.F90:347) ----------------- *
 * The consistent-mass-matrix iteration: first-approx map then a HOST loop over    *
 * num_iter_solve-1 device sweeps, each a CSR-gather correction + map + copy-back   *
 * + the per-iter dvalues halo (the M4.2 CG / M4.3b EVP host-loop-control pattern). *
 * Writes the high-order increment into dvalues (consumed by fem_fct); valuesl is   *
 * scratch here (overwritten by ice_solve_low_order afterwards, per the C order).   */
static void ice_solve_high_order_kk(fesom_ice                    *ice,
                                    const struct fesom_ssh_stiff *stiff,
                                    struct fesom_partit          *partit,
                                    const struct fesom_mesh      *mesh)
{
    const int myDim          = mesh->myDim_nod2D;
    const int nl             = mesh->nl;
    const int num_iter_solve = 3;                              /* C line 293 */

    auto rowptr = stiff->rowptr_fld.d();
    auto colind = stiff->colind_fld.d();
    auto mm     = ice->work.fct_massmatrix_fld.d();
    auto rhs_a  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    auto rhs_m  = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    auto rhs_ms = ice->data[FESOM_ICE_MSNOW].values_rhs_fld.d();
    auto a_l    = ice->data[FESOM_ICE_AICE].valuesl_fld.d();
    auto m_l    = ice->data[FESOM_ICE_MICE].valuesl_fld.d();
    auto ms_l   = ice->data[FESOM_ICE_MSNOW].valuesl_fld.d();
    auto da     = ice->data[FESOM_ICE_AICE].dvalues_fld.d();
    auto dm     = ice->data[FESOM_ICE_MICE].dvalues_fld.d();
    auto dms    = ice->data[FESOM_ICE_MSNOW].dvalues_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();
    auto area   = mesh->area_fld.d();

    /* First approximation: dvalues = rhs / area (C lines 296-302). */
    Kokkos::parallel_for("ice_ho_first", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int row) {
            if (ulev_n(row) > 1) return;
            const real_t inv_area = 1.0 / area(row*nl + 0);
            dm (row) = rhs_m (row) * inv_area;
            da (row) = rhs_a (row) * inv_area;
            dms(row) = rhs_ms(row) * inv_area;
        });
    fct_halo_nod2D(ice->data[FESOM_ICE_MICE].dvalues_fld,  partit);
    fct_halo_nod2D(ice->data[FESOM_ICE_AICE].dvalues_fld,  partit);
    fct_halo_nod2D(ice->data[FESOM_ICE_MSNOW].dvalues_fld, partit);

    for (int it = 0; it < num_iter_solve - 1; ++it) {         /* 2 passes (C line 308) */
        /* (a) update valuesl from the residual rhs - M·dvalues (C lines 310-329). */
        Kokkos::parallel_for("ice_ho_update", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int row) {
                if (ulev_n(row) > 1) return;
                const int rs = rowptr(row), re = rowptr(row + 1);
                real_t sm = 0.0, sa = 0.0, sms = 0.0;
                for (int k = rs; k < re; ++k) {
                    const int j = colind(k);
                    sm  += mm(k) * dm (j);
                    sa  += mm(k) * da (j);
                    sms += mm(k) * dms(j);
                }
                const real_t inv_area = 1.0 / area(row*nl + 0);
                m_l (row) = dm (row) + (rhs_m (row) - sm ) * inv_area;
                a_l (row) = da (row) + (rhs_a (row) - sa ) * inv_area;
                ms_l(row) = dms(row) + (rhs_ms(row) - sms) * inv_area;
            });
        /* (b) copy valuesl back to dvalues (C lines 332-337). */
        Kokkos::parallel_for("ice_ho_copy", Kokkos::RangePolicy<>(0, myDim),
            KOKKOS_LAMBDA(const int row) {
                if (ulev_n(row) > 1) return;
                dm (row) = m_l (row);
                da (row) = a_l (row);
                dms(row) = ms_l(row);
            });
        /* (c) per-iter dvalues halo (C line 340; read at CSR neighbours next pass / by fem_fct). */
        fct_halo_nod2D(ice->data[FESOM_ICE_MICE].dvalues_fld,  partit);
        fct_halo_nod2D(ice->data[FESOM_ICE_AICE].dvalues_fld,  partit);
        fct_halo_nod2D(ice->data[FESOM_ICE_MSNOW].dvalues_fld, partit);
    }
}

/* ---- E4 device twin: ice_fem_fct (ice_fct.F90:498) — Zalesak limiter -------- */
static void ice_fem_fct_kk(int                           log_id,
                           fesom_ice                    *ice,
                           const struct fesom_ssh_stiff *stiff,
                           struct fesom_partit          *partit,
                           const struct fesom_mesh      *mesh)
{
    const int    d_active = data_idx_for_logical(log_id);
    const int    myDim    = mesh->myDim_nod2D;
    const int    N_full   = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    E        = mesh->myDim_elem2D;
    const int    nl       = mesh->nl;
    const real_t gamma    = ice->ice_gamma_fct;

    auto vals   = ice->data[d_active].values_fld.d();
    auto vals_l = ice->data[d_active].valuesl_fld.d();
    auto dvals  = ice->data[d_active].dvalues_fld.d();
    auto icefl  = ice->work.fct_fluxes_fld.d();
    auto icepp  = ice->work.fct_plus_fld.d();
    auto icepm  = ice->work.fct_minus_fld.d();
    auto tmax   = ice->work.fct_tmax_fld.d();
    auto tmin   = ice->work.fct_tmin_fld.d();
    auto rowptr = stiff->rowptr_fld.d();
    auto colind = stiff->colind_fld.d();
    auto en     = mesh->elem_nodes_fld.d();
    auto ea     = mesh->elem_area_fld.d();
    auto area   = mesh->area_fld.d();
    auto ulev   = mesh->ulevels_fld.d();
    auto ulev_n = mesh->ulevels_nod2D_fld.d();

    /* init tmax/tmin over [0,N) (C lines 376-377). */
    Kokkos::parallel_for("ice_fct_tmm_init", Kokkos::RangePolicy<>(0, N_full),
        KOKKOS_LAMBDA(const int n) { tmax(n) = 0.0; tmin(n) = 0.0; });

    /* antidiffusive fluxes per element — each elem writes its OWN 3 slots → race-free
     * (C lines 389-407). icoef(n,q) = -2 on the diagonal, 1 off-diagonal. */
    Kokkos::parallel_for("ice_fct_aflux", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int elem) {
            if (ulev(elem) > 1) return;
            const int    nn[3] = { en(3*elem+0), en(3*elem+1), en(3*elem+2) };
            const real_t vol = ea(elem);
            for (int q = 0; q < 3; ++q) {
                real_t s = 0.0;
                for (int n = 0; n < 3; ++n) {
                    real_t v = gamma * vals(nn[n]) + dvals(nn[n]);
                    real_t icoef = (n == q) ? -2.0 : 1.0;
                    s += icoef * v;
                }
                const real_t inv_area_q = 1.0 / area(nn[q]*nl + 0);
                icefl(elem*3 + q) = -s * (vol * inv_area_q) / 12.0;
            }
        });

    /* cluster min/max over the CSR neighbourhood (gather), OWNED rows (C lines 412-431). */
    Kokkos::parallel_for("ice_fct_cluster", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int row) {
            if (ulev_n(row) > 1) return;
            const int rs = rowptr(row), re = rowptr(row + 1);
            real_t lo_l = INFINITY, hi_l = -INFINITY, lo_v = INFINITY, hi_v = -INFINITY;
            for (int k = rs; k < re; ++k) {
                const int j = colind(k);
                const real_t a = vals_l(j), b = vals(j);
                if (a > hi_l) hi_l = a;
                if (a < lo_l) lo_l = a;
                if (b > hi_v) hi_v = b;
                if (b < lo_v) lo_v = b;
            }
            const real_t hi = hi_l > hi_v ? hi_l : hi_v;
            const real_t lo = lo_l < lo_v ? lo_l : lo_v;
            tmax(row) = hi - vals_l(row);
            tmin(row) = lo - vals_l(row);
        });

    /* init icepplus/icepminus over [0,N) (C line 434). */
    Kokkos::parallel_for("ice_fct_pm_init", Kokkos::RangePolicy<>(0, N_full),
        KOKKOS_LAMBDA(const int n) { icepp(n) = 0.0; icepm(n) = 0.0; });

    /* sum positive/negative fluxes per node — element→node SCATTER (atomic_add, D22)
     * (C lines 439-450). */
    Kokkos::parallel_for("ice_fct_pm_sum", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int elem) {
            if (ulev(elem) > 1) return;
            const int nn[3] = { en(3*elem+0), en(3*elem+1), en(3*elem+2) };
            for (int q = 0; q < 3; ++q) {
                const int    n    = nn[q];
                const real_t flux = icefl(elem*3 + q);
                if (flux > 0.0) Kokkos::atomic_add(&icepp(n), flux);
                else            Kokkos::atomic_add(&icepm(n), flux);
            }
        });

    /* correction factors, OWNED rows (C lines 453-471). */
    Kokkos::parallel_for("ice_fct_corr", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) return;
            real_t flux = icepp(n);
            if (Kokkos::fabs(flux) > 0.0) {
                const real_t denom = flux > 1e-12 ? flux : 1e-12;
                const real_t v = tmax(n) / denom;
                icepp(n) = v < 1.0 ? v : 1.0;
            } else icepp(n) = 0.0;
            flux = icepm(n);
            if (Kokkos::fabs(flux) > 0.0) {
                const real_t denom = flux < -1e-12 ? flux : -1e-12;
                const real_t v = tmin(n) / denom;
                icepm(n) = v < 1.0 ? v : 1.0;
            } else icepm(n) = 0.0;
        });
    /* D21 halo: limit reads icepplus/icepminus at halo elnodes (C line 472-474). */
    fct_halo_nod2D(ice->work.fct_plus_fld,  partit);
    fct_halo_nod2D(ice->work.fct_minus_fld, partit);

    /* limit element fluxes — each elem reads its 3 vertices, scales its OWN slots → race-free
     * (C lines 478-493). */
    Kokkos::parallel_for("ice_fct_limit", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int elem) {
            if (ulev(elem) > 1) return;
            const int nn[3] = { en(3*elem+0), en(3*elem+1), en(3*elem+2) };
            real_t ae = 1.0;
            for (int q = 0; q < 3; ++q) {
                const real_t flux = icefl(elem*3 + q);
                const real_t cand = (flux >= 0.0) ? icepp(nn[q]) : icepm(nn[q]);
                if (cand < ae) ae = cand;
            }
            icefl(elem*3 + 0) *= ae;
            icefl(elem*3 + 1) *= ae;
            icefl(elem*3 + 2) *= ae;
        });

    /* apply: phase 1 overwrite vals=valuesl over OWNED (C lines 498-501); phase 2 element→node
     * SCATTER vals += limited fluxes (atomic_add, D22; C lines 502-510). */
    Kokkos::parallel_for("ice_fct_apply1", Kokkos::RangePolicy<>(0, myDim),
        KOKKOS_LAMBDA(const int n) {
            if (ulev_n(n) > 1) return;
            vals(n) = vals_l(n);
        });
    Kokkos::parallel_for("ice_fct_apply2", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int elem) {
            if (ulev(elem) > 1) return;
            const int nn[3] = { en(3*elem+0), en(3*elem+1), en(3*elem+2) };
            for (int q = 0; q < 3; ++q)
                Kokkos::atomic_add(&vals(nn[q]), icefl(elem*3 + q));
        });
    /* D21 halo: exchange the per-tracer values (C line 516). */
    fct_halo_nod2D(ice->data[d_active].values_fld, partit);
}

/* ---- E5 device driver: ice_fct_solve (ice_fct.F90:210) ---------------------- */
void fesom_ice_fct_solve_kk(fesom_ice                    *ice,
                            const struct fesom_ssh_stiff *stiff,
                            struct fesom_partit          *partit,
                            const struct fesom_mesh      *mesh)
{
    /* Same order as the C: high_order (writes dvalues; valuesl scratch) → low_order
     * (writes valuesl) → 3× fem_fct (m_ice, a_ice, m_snow). */
    ice_solve_high_order_kk(ice, stiff, partit, mesh);
    ice_solve_low_order_kk (ice, stiff, partit, mesh);

    ice_fem_fct_kk(FCT_LOG_M,  ice, stiff, partit, mesh);
    ice_fem_fct_kk(FCT_LOG_A,  ice, stiff, partit, mesh);
    ice_fem_fct_kk(FCT_LOG_MS, ice, stiff, partit, mesh);
}

/* FESOM_KK_VERIFY=icefct — the FCT read-modify-writes data[*].values (each fem_fct overwrites
 * its tracer's values with valuesl then accumulates the limited fluxes; the LO/HO solves + the
 * antidiffusive fluxes all read the ORIGINAL values), so this is L26 capture-before over the 3
 * values. The driver snapshots them PRE-FCT; here snapshot the KK result, restore the pre-values,
 * run the C twins (fesom_ice_tg_rhs + fesom_ice_fct_solve — they recompute every FCT-internal
 * scratch: values_rhs/valuesl/dvalues/fct_* from scratch, and exchange their own halos — collective,
 * so run on ALL ranks), diff the 3 values, restore KK. The inputs uice/vice/mm/CSR are intact.
 * On Serial the atomic_add scatters are ordered → max|Δ|==0; OpenMP/CUDA climate-close (D22).
 * ⚠️ Meaningful only with ACTIVE ice (CORE2). */
void fesom_ice_fct_verify(fesom_ice                    *ice,
                          const struct fesom_ssh_stiff *stiff,
                          struct fesom_partit          *partit,
                          const struct fesom_mesh      *mesh,
                          int                           step_n,
                          const std::vector<real_t>    &pre_a,
                          const std::vector<real_t>    &pre_m,
                          const std::vector<real_t>    &pre_ms)
{
    const int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    real_t *a  = ice->data[FESOM_ICE_AICE].values;
    real_t *m  = ice->data[FESOM_ICE_MICE].values;
    real_t *ms = ice->data[FESOM_ICE_MSNOW].values;
    std::vector<real_t> ka(a, a + N), km(m, m + N), kms(ms, ms + N);   /* KK result */
    std::copy(pre_a.begin(),  pre_a.end(),  a);                        /* restore FCT inputs */
    std::copy(pre_m.begin(),  pre_m.end(),  m);
    std::copy(pre_ms.begin(), pre_ms.end(), ms);
    fesom_ice_tg_rhs   (ice,       partit, mesh);                      /* C twins */
    fesom_ice_fct_solve(ice, stiff, partit, mesh);
    auto mx = [](const std::vector<real_t> &kk, const real_t *c, int n) {
        double d = 0.0;
        for (int i = 0; i < n; ++i) { double x = std::fabs((double)kk[i] - (double)c[i]); if (x > d) d = x; }
        return d; };
    double da = mx(ka, a, N), dm = mx(km, m, N), dms = mx(kms, ms, N);
    std::copy(ka.begin(),  ka.end(),  a);                             /* restore KK production state */
    std::copy(km.begin(),  km.end(),  m);
    std::copy(kms.begin(), kms.end(), ms);
    double dmax = da; if (dm > dmax) dmax = dm; if (dms > dmax) dmax = dms;
    const std::string be = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=icefct] step %d backend=%s  max|Δ|: a_ice=%.3e m_ice=%.3e m_snow=%.3e\n",
                step_n, be.c_str(), da, dm, dms);
    std::fflush(stdout);
    if (be == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=icefct] FAIL step %d: ice FCT Serial must be "
                             "bit-identical (max|Δ|=%.3e)\n", step_n, dmax);
        std::abort();
    }
}
