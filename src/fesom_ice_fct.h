#ifndef FESOM_ICE_FCT_H
#define FESOM_ICE_FCT_H

#include "fesom_ice_types.h"

#include <vector>            // M4.3c: FESOM_KK_VERIFY=icefct capture-before buffers

struct fesom_mesh;
struct fesom_partit;
struct fesom_ssh_stiff;

/*
 * Sea-ice FCT advection. Literal port of ice_fct.F90.
 *
 * The ssh_stiff CSR (rowptr, colind, nnz) is used as the node-neighbour
 * lookup that Fortran exposes as nn_pos / nn_num. ice%work%fct_massmatrix
 * shares the ssh_stiff sparsity pattern (allocated to ssh_stiff->nnz here
 * — mirror of Fortran's mass_matrix(ssh_stiff%rowptr_dim_nz)).
 *
 * Skipped routines (out of scope for our config):
 *   ice_TG_rhs_div    — feeds ice_update_for_div which is commented at the
 *                       call site (ice_setup_step.F90:264). Dead code.
 *   ice_update_for_div — same.
 *   __oifs / __ifsinterface tracer-temp branches — not configured.
 */

/*
 * One-time setup. Allocates ice->work->fct_massmatrix to size stiff->nnz
 * and fills it per ice_fct.F90:1145 ice_mass_matrix_fill.
 *
 * Must be called AFTER fesom_ssh_stiff_alloc_and_build (so stiff->rowptr
 * and stiff->colind are populated).
 */
void fesom_ice_mass_matrix_fill(fesom_ice                    *ice,
                                const struct fesom_ssh_stiff *stiff,
                                struct fesom_partit          *partit,
                                const struct fesom_mesh      *mesh);

/*
 * Taylor-Galerkin RHS — mirror of ice_fct.F90:91 ice_TG_rhs.
 * Writes ice->data[k].values_rhs for k in {0..2} (a, m, m_snow).
 */
void fesom_ice_tg_rhs(fesom_ice                *ice,
                      struct fesom_partit      *partit,
                      const struct fesom_mesh  *mesh);

/*
 * Driver — mirror of ice_fct.F90:210 ice_fct_solve.
 * Sequences: solve_high_order → solve_low_order → 3× fem_fct.
 */
void fesom_ice_fct_solve(fesom_ice                    *ice,
                         const struct fesom_ssh_stiff *stiff,
                         struct fesom_partit          *partit,
                         const struct fesom_mesh      *mesh);

/*===========================================================================
 * M4.3c — DEVICE (Kokkos) twins of the sea-ice FCT advection. The M2.6
 * ocean-FCT analogue (2-D surface). tg_rhs = element→node SCATTER; fct_solve
 * = high/low-order CSR-gather mass-matrix solves (high_order is a host loop
 * over device sweeps + per-iter dvalues halo) + 3× Zalesak fem_fct (scatters
 * + icepplus/icepminus/values halos). The ssh_stiff CSR + fct_massmatrix are
 * device-current from their one-shot pushes. Driver: IN push uice/vice +
 * data[*].values; OUT sync_host(data[*].values).
 *===========================================================================*/

/* Taylor-Galerkin RHS on device. Marks data[*].values_rhs modify_device(). */
void fesom_ice_tg_rhs_kk(fesom_ice                *ice,
                         struct fesom_partit      *partit,
                         const struct fesom_mesh  *mesh);

/* FCT driver on device: solve_high_order → solve_low_order → 3× fem_fct. Each
 * fem_fct ends with the per-tracer data[*].values nod2D halo (modify_device()). */
void fesom_ice_fct_solve_kk(fesom_ice                    *ice,
                            const struct fesom_ssh_stiff *stiff,
                            struct fesom_partit          *partit,
                            const struct fesom_mesh      *mesh);

/* FESOM_KK_VERIFY=icefct gate: L26 capture-before over the 3 read-modify-write
 * data[*].values. The driver snapshots the PRE-FCT values; this snapshots the KK
 * result, restores the pre-values, runs the C twins (tg_rhs + fct_solve), diffs,
 * restores KK. Asserts max|Δ|==0 on Serial. ⚠️ Meaningful only on CORE2 (L42). */
void fesom_ice_fct_verify(fesom_ice                    *ice,
                          const struct fesom_ssh_stiff *stiff,
                          struct fesom_partit          *partit,
                          const struct fesom_mesh      *mesh,
                          int                           step_n,
                          const std::vector<real_t>    &pre_a,
                          const std::vector<real_t>    &pre_m,
                          const std::vector<real_t>    &pre_ms);

#endif /* FESOM_ICE_FCT_H */
