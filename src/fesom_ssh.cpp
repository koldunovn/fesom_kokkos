/*
 * SSH stiffness matrix + diagonal-symmetric preconditioner + CG iteration.
 * Literal port of Fortran:
 *   - init_stiff_mat_ale       (oce_ale.F90:1393-1684)
 *   - ssh_solve_preconditioner (solver.F90:31-95)
 *   - compute_ssh_rhs_ale      (oce_ale.F90:1821-1956, linfs branch)
 *   - ssh_solve_cg             (solver.F90:98-281)
 *
 * Phase 1 simplifications (literal port — these branches are explicitly the
 * Fortran behaviour when the corresponding flags are off):
 *   - linfs ALE: update_stiff_mat_ale is NOT called (gated at oce_ale.F90:3722).
 *     The stiffness matrix is built once and reused every step.
 *   - No partial cells: zbar_e_srf[e] = zbar[0] = 0, zbar_e_bot[e] = zbar[nlevels[e]-1].
 *   - No cavity: ulevels_nod2D = ulevels = 1 → 0-based 0 → diagonal mass term
 *     is added at every node.
 *   - Serial (1 MPI rank): exchange_nod and MPI_Allreduce are no-ops.
 */
#include "fesom_ssh.h"
#include "fesom_ale.h"       // M6.3: fesom_ale_is_zstar()
#include "fesom_forcing.h"   // M6.3: water_flux for the zstar ssh_rhs tail
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_mesh.h"
#include "fesom_momentum.h"   // M4.2-b: C twins update_vel/compute_hbar for the §5 block verify

#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include <Kokkos_Core.hpp>    // M4.2-b: device CG kernels (parallel_for / parallel_reduce / atomic)
#include <string>             // M4.2-b: FESOM_KK_VERIFY backend name
#include <algorithm>          // M4.2-b: verify snapshot/restore copies (host-only diagnostic)

#include "fesom_halo.h"
#include "fesom_halo_device.hpp"   // M5.1: on-device (GPU-aware-MPI) halo exchange
#include "fesom_partit.h"
#include "fesom_speed.hpp"         // M7 E.CG1: FESOM_SPEED_CGPIPE (opt-in _exp lever)
#include <unordered_map>           // M7 E.CG1: one-time gid->local maps (setup only)

/*===========================================================================
 * Stiffness matrix: build CSR + fill values
 *===========================================================================*/

void fesom_ssh_stiff_alloc_and_build(fesom_ssh_stiff       *S,
                                     const struct fesom_mesh *mesh)
{
    /* Stiffness matrix has myDim rows (interior nodes only); column indices
     * reach into the halo (myDim..myDim+eDim) for SpMV. Mirrors Fortran
     * where ssh_stiff is sized myDim_nod2D. */
    const int N         = mesh->myDim_nod2D;
    const int N_alloc   = mesh->myDim_nod2D + mesh->eDim_nod2D;
    /* Loop over INTERIOR edges only — Fortran oce_ale.F90:1517 has
     * `do ed=1,myDim_edge2D` with the comment "!! Attention". Edges in eDim
     * belong to a neighbour rank and are accumulated by the owner. The owner's
     * interior loop adds to its myDim row; the same row is in our eDim_nod2D
     * (halo), where we don't have a matrix row. So skipping halo edges here
     * exactly matches Fortran. */
    const int E         = mesh->myDim_edge2D;
    /* M4.2-a: S now holds fesom::Field members → a raw memset over it is UB (D13/L13).
     * Value-initialise (runs each Field's default ctor, releases any prior storage, zeros
     * every POD) — the fesom_tracer_adv_init pattern. */
    *S = fesom_ssh_stiff{};
    S->dim = N;
    (void)N_alloc;

    /* ---- 1. Per-row neighbour count and local-id list ---------------------
     * Fortran lines 1435-1463. n_pos(1, n) = n is set first so the diagonal
     * is the first entry in each row's neighbour list. */
    /* n_num: counter per row in pass 1, and node-id → sparse-position lookup
     * in pass 2 (used as a sparse hash, so it must be sized for ALL local
     * nodes — interior + halo — since column indices reach into halo). */
    int *n_num = (int *)calloc((size_t)N_alloc, sizeof(int));
    int  max_neighbors = 12;     /* Fortran uses 12 — same hard cap */
    int *n_pos = (int *)calloc((size_t)N * (size_t)max_neighbors, sizeof(int));
    FESOM_CHECK(n_num && n_pos, "ssh_stiff: out of memory (neighbour build)");

    for (int n = 0; n < N; ++n) {
        n_num[n] = 1;
        n_pos[n * max_neighbors + 0] = n;
    }
    for (int ed = 0; ed < E; ++ed) {
        int n1 = mesh->edges[2*ed + 0];
        int n2 = mesh->edges[2*ed + 1];
        /* Only add edges where the row endpoint is INTERIOR (the matrix has
         * no halo-row entries); the column endpoint can be interior or halo. */
        if (n1 < N) {
            FESOM_CHECK(n_num[n1] < max_neighbors,
                        "ssh_stiff: node %d has > %d neighbours", n1, max_neighbors);
            n_pos[n1 * max_neighbors + n_num[n1]++] = n2;
        }
        if (n2 < N) {
            FESOM_CHECK(n_num[n2] < max_neighbors,
                        "ssh_stiff: node %d has > %d neighbours", n2, max_neighbors);
            n_pos[n2 * max_neighbors + n_num[n2]++] = n1;
        }
    }

    /* ---- 2. Build CSR rowptr (lines 1467-1477) ----------------------------
     * M4.2-a: rowptr is OWNED by S->rowptr_fld; the raw pointer is a non-owning alias
     * to field.h(). Field::alloc zero-inits (rowptr is then fully written below). */
    S->rowptr_fld.alloc("ssh.rowptr", (size_t)(N + 1));
    S->rowptr = S->rowptr_fld.h();
    FESOM_CHECK(S->rowptr, "ssh_stiff: out of memory (rowptr)");
    S->rowptr[0] = 0;
    for (int n = 0; n < N; ++n) {
        S->rowptr[n + 1] = S->rowptr[n] + n_num[n];
    }
    S->nnz = S->rowptr[N];

    /* ---- 3. Fill colind + zero values + alloc pr_values (lines 1481-1494) -
     * M4.2-a: colind/values/pr_values are OWNED by their Fields (raw ptrs = aliases).
     * Field::alloc zero-inits → matches the original calloc for values/pr_values; colind
     * is fully written below (the original malloc was uninitialised — alloc over-specifies). */
    S->colind_fld.alloc("ssh.colind", (size_t)S->nnz);
    S->values_fld.alloc("ssh.values", (size_t)S->nnz);
    S->pr_values_fld.alloc("ssh.pr_values", (size_t)S->nnz);
    S->colind    = S->colind_fld.h();
    S->values    = S->values_fld.h();
    S->pr_values = S->pr_values_fld.h();
    FESOM_CHECK(S->colind && S->values && S->pr_values,
                "ssh_stiff: out of memory (CSR arrays)");
    for (int n = 0; n < N; ++n) {
        int start = S->rowptr[n];
        for (int k = 0; k < n_num[n]; ++k) {
            S->colind[start + k] = n_pos[n * max_neighbors + k];
        }
    }
    free(n_pos);

    /* ---- 4. Stiffness term per edge × adjacent cell (lines 1503-1571) -----
     * factor = g * dt * alpha * theta
     * For each edge, for each adjacent cell el(i):
     *   fy[k] = (zbar_e_bot[el(i)] - zbar_e_srf[el(i)]) *
     *           ( gradient_sca[1..3](el(i)) * edge_cross_dxdy[2*i  ](ed)
     *           - gradient_sca[4..6](el(i)) * edge_cross_dxdy[2*i-1](ed) )
     *   if i==2: fy = -fy
     *   row = edges(1, ed): values[npos] += fy * factor
     *   row = edges(2, ed): values[npos] -= fy * factor
     * with npos = positions of elnodes(1..3) in row's neighbour list. */
    const real_t factor = (real_t)FESOM_G * (real_t)FESOM_PHASE1_DT
                        * (real_t)FESOM_PHASE1_ALPHA * (real_t)FESOM_PHASE1_THETA;

    /* n_num is reused as a "node-id → sparse-position-in-current-row" lookup
     * (Fortran line 1507 zeros it). Must reset across the FULL local extent. */
    for (int n = 0; n < N_alloc; ++n) n_num[n] = 0;

    for (int ed = 0; ed < E; ++ed) {
        int el[2] = { mesh->edge_tri[2*ed + 0], mesh->edge_tri[2*ed + 1] };
        int e_n[2] = { mesh->edges[2*ed + 0], mesh->edges[2*ed + 1] };
        for (int i = 0; i < 2; ++i) {
            /* For ed in myDim_edge2D, the FESOM partition guarantees both
             * adjacent triangles are in myDim_elem2D — so gradient_sca and
             * elem_nodes (both sized myDim_elem2D in Fortran) are always
             * safe to read. Skip true boundary (-1) and any element id past
             * myDim_elem2D as defensive (would indicate partition oddity). */
            if (el[i] < 0 || el[i] >= mesh->myDim_elem2D) continue;
            int en[3] = { mesh->elem_nodes[3*el[i] + 0],
                          mesh->elem_nodes[3*el[i] + 1],
                          mesh->elem_nodes[3*el[i] + 2] };
            const real_t *g = &mesh->gradient_sca[6 * el[i]];

            /* zbar_e_srf and zbar_e_bot (Phase 1: no partial cells, no cavity) */
            real_t zbar_e_srf = mesh->zbar[0];                            /* = 0 */
            real_t zbar_e_bot = mesh->zbar[mesh->nlevels[el[i]] - 1];     /* < 0 */
            real_t depth_diff = zbar_e_bot - zbar_e_srf;                  /* < 0 */

            /* edge_cross_dxdy[4*ed + 2*i + 0] = dx_i (Fortran 2*i-1)
               edge_cross_dxdy[4*ed + 2*i + 1] = dy_i (Fortran 2*i)
               Recall Fortran 1-based i ∈ {1,2}: 2*i-1 ∈ {1,3} → C 0-based offset {0,2};
                                                  2*i   ∈ {2,4} → C 0-based offset {1,3}. */
            real_t dx_i = mesh->edge_cross_dxdy[4*ed + 2*i + 0];
            real_t dy_i = mesh->edge_cross_dxdy[4*ed + 2*i + 1];

            real_t fy[3];
            for (int k = 0; k < 3; ++k) {
                fy[k] = depth_diff * (g[k] * dy_i - g[3 + k] * dx_i);
            }
            if (i == 1) {  /* Fortran i==2 → C i==1 */
                fy[0] = -fy[0]; fy[1] = -fy[1]; fy[2] = -fy[2];
            }

            /* Row = edges(1, ed) → C e_n[0]. Only build matrix entries for
             * interior rows; halo-row contributions are summed on the
             * owning rank instead. */
            if (e_n[0] < N) {
                int row = e_n[0];
                int rstart = S->rowptr[row];
                int rend   = S->rowptr[row + 1];
                for (int n = rstart; n < rend; ++n) {
                    n_num[S->colind[n]] = n;
                }
                int npos[3] = { n_num[en[0]], n_num[en[1]], n_num[en[2]] };
                for (int k = 0; k < 3; ++k) {
                    S->values[npos[k]] += fy[k] * factor;
                }
            }
            if (e_n[1] < N) {
                int row = e_n[1];
                int rstart = S->rowptr[row];
                int rend   = S->rowptr[row + 1];
                for (int n = rstart; n < rend; ++n) {
                    n_num[S->colind[n]] = n;
                }
                int npos[3] = { n_num[en[0]], n_num[en[1]], n_num[en[2]] };
                for (int k = 0; k < 3; ++k) {
                    S->values[npos[k]] -= fy[k] * factor;
                }
            }
        }
    }

    /* ---- 5. Mass matrix term on diagonal (lines 1575-1582) ---------------
     * For non-cavity nodes (ulevels_nod2D[row] == 1, all of them in Phase 1):
     *   values[diagonal] += areasvol[ulevels_nod2D[row], row] / dt
     * In C 0-based with ulevels_nod2D[row]=1, the surface-layer index is 0. */
    const real_t inv_dt = 1.0 / (real_t)FESOM_PHASE1_DT;
    for (int row = 0; row < N; ++row) {
        if (mesh->ulevels_nod2D[row] > 1) continue;        /* cavity guard */
        int diag = S->rowptr[row];
        int top_layer = mesh->ulevels_nod2D[row] - 1;       /* = 0 in Phase 1 */
        S->values[diag] += mesh->areasvol[FESOM_NODE3D(row, top_layer, mesh->nl)] * inv_dt;
    }

    free(n_num);
}

void fesom_ssh_stiff_free(fesom_ssh_stiff *S)
{
    /* M4.2-a: rowptr/colind/values/pr_values are non-owning aliases to the Field host
     * mirrors — no per-array free (that would double-free). Value-init releases every
     * Field (empty-DualView assign → refcount drop/free) and zeros the POD members
     * (D13/L13; the fesom_tracer_adv_free pattern). Runs before Kokkos::finalize(). */
    *S = fesom_ssh_stiff{};
}

/*===========================================================================
 * Preconditioner: ssh_solve_preconditioner (solver.F90:31-95)
 *===========================================================================*/

void fesom_ssh_preconditioner(fesom_ssh_stiff *S, const struct fesom_mesh *mesh,
                              struct fesom_partit *partit)
{
    const int N       = mesh->myDim_nod2D;
    const int N_alloc = mesh->myDim_nod2D + mesh->eDim_nod2D;

    /* Collect diagonal values. Sized for full local extent because column
     * indices in `S->colind` can refer to halo nodes (n >= N), and the
     * preconditioner formula reads diag_values[node] for every off-diagonal.
     * Halo entries are filled by halo exchange below (solver.F90 mirrors). */
    real_t *diag_values = (real_t *)calloc((size_t)N_alloc, sizeof(real_t));
    FESOM_CHECK(diag_values, "preconditioner: out of memory");
    for (int row = 0; row < N; ++row) {
        diag_values[row] = S->values[S->rowptr[row]];   /* diag at offset 0 */
    }
    if (partit && partit->npes > 1) {
        fesom_halo_exchange(diag_values, FESOM_HALO_NOD2D, 1, 1, partit);
    }

    /* MITgcm-style symmetric preconditioner (solver.F90:77-86):
     *   pr[diag]                 = 1 / diag(row)
     *   pr[row, col=node, n>0]   = -0.5 * (off / diag(row)) / (diag(row) + diag(node))
     */
    for (int row = 0; row < N; ++row) {
        int  start = S->rowptr[row];
        int  nend  = S->rowptr[row + 1] - start;
        real_t diag_row = S->values[start];
        S->pr_values[start] = 1.0 / diag_row;
        for (int n = 1; n < nend; ++n) {
            int node = S->colind[start + n];
            S->pr_values[start + n] = -0.5 * (S->values[start + n] / diag_row)
                                          / (diag_row + diag_values[node]);
        }
    }
    free(diag_values);

    /* M4.2-a: the CSR is now final (build filled rowptr/colind/values; this routine just
     * filled pr_values). It is set-once — linfs never updates the stiffness matrix
     * (oce_ale.F90:3722) and ice_mass_matrix_fill takes it `const`. Push all four arrays to
     * the device a SINGLE time (the mesh_sync_geometry_device pattern, L14) so the M4.2-b
     * device CG SpMV reads them device-current with no per-step sync. The raw host writes
     * above are invisible to the DualView modify flags, so modify_host() first. No-op on
     * Serial/OpenMP (host==device); one bitwise-exact deep_copy each on CUDA. */
    S->rowptr_fld.modify_host();    S->rowptr_fld.sync_device();
    S->colind_fld.modify_host();    S->colind_fld.sync_device();
    S->values_fld.modify_host();    S->values_fld.sync_device();
    S->pr_values_fld.modify_host(); S->pr_values_fld.sync_device();
}

/*===========================================================================
 * compute_ssh_rhs_ale — linfs branch (oce_ale.F90:1821-1956)
 *===========================================================================*/

void fesom_compute_ssh_rhs_linfs(const struct fesom_mesh *mesh,
                                 struct fesom_dyn        *dyn)
{
    const int N  = mesh->myDim_nod2D;
    const int N_alloc = mesh->myDim_nod2D + mesh->eDim_nod2D;
    /* Fortran oce_ale.F90:1862 uses `do ed=1, myDim_edge2D` only. The edge
     * partition replicates each cross-rank edge in BOTH neighbour ranks'
     * myDim_edge2D, so each interior endpoint receives full contributions
     * from its rank alone. exchange_nod at the end of the routine refreshes
     * halo entries (which were only side-effects of the loop) with owner
     * values — same pattern as Fortran. */
    const int E  = mesh->myDim_edge2D;
    const int nl = mesh->nl;
    const real_t alpha = (real_t)FESOM_PHASE1_ALPHA;
    const real_t one_minus_alpha = 1.0 - alpha;
    (void)N;

    /* Zero ssh_rhs over full local extent. */
    memset(dyn->ssh_rhs, 0, (size_t)N_alloc * sizeof(real_t));

    /* Edge loop — accumulate fluxes into ssh_rhs (lines 1862-1919). */
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
                real_t u  = dyn->uv    [FESOM_ELEMVEC(el1, nz, nl) + 0];
                real_t v  = dyn->uv    [FESOM_ELEMVEC(el1, nz, nl) + 1];
                real_t ur = dyn->uv_rhs[FESOM_ELEMVEC(el1, nz, nl) + 0];
                real_t vr = dyn->uv_rhs[FESOM_ELEMVEC(el1, nz, nl) + 1];
                real_t h  = mesh->helem[FESOM_ELEM3D(el1, nz, nl)];
                c1 += alpha * ((v + vr) * dx1 - (u + ur) * dy1) * h;
            }
        }
        real_t c2 = 0.0;
        if (el2 >= 0) {
            real_t dx2 = mesh->edge_cross_dxdy[4*ed + 2];
            real_t dy2 = mesh->edge_cross_dxdy[4*ed + 3];
            int nzmin = mesh->ulevels[el2] - 1;
            int nzmax = mesh->nlevels[el2] - 1;
            for (int nz = nzmin; nz < nzmax; ++nz) {
                real_t u  = dyn->uv    [FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t v  = dyn->uv    [FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t ur = dyn->uv_rhs[FESOM_ELEMVEC(el2, nz, nl) + 0];
                real_t vr = dyn->uv_rhs[FESOM_ELEMVEC(el2, nz, nl) + 1];
                real_t h  = mesh->helem[FESOM_ELEM3D(el2, nz, nl)];
                c2 -= alpha * ((v + vr) * dx2 - (u + ur) * dy2) * h;
            }
        }
        dyn->ssh_rhs[n1] += (c1 + c2);
        dyn->ssh_rhs[n2] -= (c1 + c2);
    }

    /* linfs branch: ssh_rhs += (1 - alpha) * ssh_rhs_old (lines 1947-1951).
       For Phase 1 with no surface forcing yet, water_flux is zero so we skip
       the corresponding term in the non-linfs branch (which we do not exercise). */
    for (int n = 0; n < N; ++n) {
        dyn->ssh_rhs[n] += one_minus_alpha * dyn->ssh_rhs_old[n];
    }
}

/*===========================================================================
 * Solverinfo lifecycle
 *===========================================================================*/

void fesom_solverinfo_alloc(fesom_solverinfo       *si,
                            const struct fesom_mesh *mesh)
{
    /* M4.2-a: si holds fesom::Field members → value-init, not memset (D13/L13). */
    *si = fesom_solverinfo{};
    si->maxiter = FESOM_PHASE1_MAXITER;
    si->soltol  = FESOM_PHASE1_SOLTOL;
    /* Allocate for full local extent — matrix-vector reads pp at column
     * indices that may reach into halo. The Fields OWN the storage; the raw
     * pointers are non-owning aliases (the C twin reads them). */
    size_t n = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D);
    si->rr_fld.alloc("ssh.cg.rr",   n);  si->rr  = si->rr_fld.h();
    si->zz_fld.alloc("ssh.cg.zz",   n);  si->zz  = si->zz_fld.h();
    si->pp_fld.alloc("ssh.cg.pp",   n);  si->pp  = si->pp_fld.h();
    si->App_fld.alloc("ssh.cg.App", n);  si->App = si->App_fld.h();
    FESOM_CHECK(si->rr && si->zz && si->pp && si->App,
                "solverinfo: out of memory");
}

void fesom_solverinfo_free(fesom_solverinfo *si)
{
    /* M4.2-a: rr/zz/pp/App are non-owning aliases — value-init releases the Fields and
     * zeros the POD (no per-array free; the fesom_tracer_adv_free pattern). */
    *si = fesom_solverinfo{};
}

/*===========================================================================
 * CG iteration — ssh_solve_cg (solver.F90:98-281)
 *
 * Solves S * x = rhs where x = dyn->d_eta (in/out), rhs = dyn->ssh_rhs.
 *===========================================================================*/

/* CSR mat-vec: y = S * v. Same loop body as the Fortran sum() expression
   on line 160-161 / 203 / 239 (uses pr_values for the precond step). */
static void csr_matvec(const fesom_ssh_stiff *S,
                       const real_t          *Avals,
                       const real_t          *v,
                       real_t                *y, int dim)
{
    for (int row = 0; row < dim; ++row) {
        real_t s = 0.0;
        int rstart = S->rowptr[row];
        int rend   = S->rowptr[row + 1];
        for (int n = rstart; n < rend; ++n) {
            s += Avals[n] * v[S->colind[n]];
        }
        y[row] = s;
    }
}

int fesom_ssh_solve_cg(const fesom_ssh_stiff *S,
                       fesom_solverinfo      *si,
                       const struct fesom_mesh *mesh,
                       struct fesom_dyn        *dyn)
{
    /* Slice 30e — full parallel CG. Each iteration exchanges pp before SpMV,
     * rr after residual update; reductions use MPI_Allreduce. */
    /* Helper macro: exchange a nod2D field. */
    #define EXCH(field) fesom_halo_exchange((field), FESOM_HALO_NOD2D, 1, 1, si->partit)
    /* All-reduce sum of a single double in-place. */
    #define ALLREDUCE_SUM(var) MPI_Allreduce(MPI_IN_PLACE, &(var), 1, MPI_DOUBLE, MPI_SUM, si->partit->MPI_COMM_FESOM)

    const int    N      = mesh->myDim_nod2D;
    const real_t soltol = si->soltol;
    real_t      *X      = dyn->d_eta;
    const real_t *rhs   = dyn->ssh_rhs;
    real_t       *rr    = si->rr;
    real_t       *zz    = si->zz;
    real_t       *pp    = si->pp;
    real_t       *App   = si->App;

    /* Initial ‖rhs‖² and tolerance (Fortran solver.F90:142-154). */
    real_t s_old = 0.0;
    for (int row = 0; row < N; ++row) s_old += rhs[row] * rhs[row];
    if (si->partit && si->partit->npes > 1) ALLREDUCE_SUM(s_old);
    /* The global problem size for the rtol normalisation must be the GLOBAL
     * row count, not local (Fortran uses nod2D). */
    int N_global = (si->partit && si->partit->npes > 1) ? mesh->nod2D : N;
    real_t rtol = soltol * sqrt(s_old / (real_t)N_global);

    if (s_old == 0.0) {
        memset(X, 0, (size_t)N * sizeof(real_t));
        si->last_iters = 0;
        return 0;
    }

    /* r0 = rhs - A * X. Need X halo exchanged before SpMV. */
    if (si->partit && si->partit->npes > 1) EXCH(X);
    csr_matvec(S, S->values, X, rr, N);
    for (int row = 0; row < N; ++row) rr[row] = rhs[row] - rr[row];
    if (si->partit && si->partit->npes > 1) EXCH(rr);

    /* z0 = M^{-1} r0; pp = z0 */
    csr_matvec(S, S->pr_values, rr, zz, N);
    memcpy(pp, zz, (size_t)N * sizeof(real_t));

    /* s_old = r0·z0 */
    s_old = 0.0;
    for (int row = 0; row < N; ++row) s_old += rr[row] * zz[row];
    if (si->partit && si->partit->npes > 1) ALLREDUCE_SUM(s_old);

    int iter = 0;
    int   verbose = (getenv("FESOM_VERBOSE_CG") != NULL);
    /* Heartbeat from rank 0 every 100 iters regardless of env var, so a
     * hung CG always shows something in the log. */
    int   heartbeat_every = 100;
    for (iter = 1; iter <= si->maxiter; ++iter) {
        /* App = A * pp; need pp halo exchanged. */
        if (si->partit && si->partit->npes > 1) EXCH(pp);
        csr_matvec(S, S->values, pp, App, N);

        /* α = s_old / (pp·App) */
        real_t s_aux = 0.0;
        for (int row = 0; row < N; ++row) s_aux += pp[row] * App[row];
        if (si->partit && si->partit->npes > 1) ALLREDUCE_SUM(s_aux);
        if (s_aux == 0.0 || s_aux != s_aux) {       /* NaN/zero check */
            if (si->partit == NULL || si->partit->mype == 0) {
                fprintf(stderr, "[fesom_ssh] CG abort at iter %d: pp·App = %g (s_old=%g)\n",
                        iter, (double)s_aux, (double)s_old); fflush(stderr);
            }
            FESOM_DIE("CG: pp·App is %g — matrix singular or NaN propagated", s_aux);
        }
        real_t al = s_old / s_aux;

        for (int row = 0; row < N; ++row) {
            X [row] += al * pp [row];
            rr[row] -= al * App[row];
        }
        if (si->partit && si->partit->npes > 1) EXCH(rr);

        csr_matvec(S, S->pr_values, rr, zz, N);

        real_t sp0 = 0.0, sp1 = 0.0;
        for (int row = 0; row < N; ++row) {
            sp0 += rr[row] * zz[row];
            sp1 += rr[row] * rr[row];
        }
        if (si->partit && si->partit->npes > 1) {
            ALLREDUCE_SUM(sp0);
            ALLREDUCE_SUM(sp1);
        }

        real_t residual = sqrt(sp1 / (real_t)N_global);
        if ((si->partit == NULL || si->partit->mype == 0)
            && (verbose ? (iter <= 5 || iter % 50 == 0)
                        : (iter % heartbeat_every == 0))) {
            fprintf(stderr, "[fesom_ssh] CG iter %4d: res=%.4e rtol=%.4e\n",
                    iter, (double)residual, (double)rtol);
            fflush(stderr);
        }
        if (residual < rtol) break;
        if (residual != residual || residual > 1e30) {       /* NaN/divergence */
            if (si->partit == NULL || si->partit->mype == 0) {
                fprintf(stderr,
                    "[fesom_ssh] CG abort at iter %d: residual=%g (NaN or divergence)\n",
                    iter, (double)residual); fflush(stderr);
            }
            FESOM_DIE("CG residual diverged");
        }

        real_t be = sp0 / s_old;
        s_old = sp0;
        for (int row = 0; row < N; ++row) pp[row] = zz[row] + be * pp[row];
    }
    if (iter > si->maxiter) {
        if (si->partit == NULL || si->partit->mype == 0) {
            fprintf(stderr, "[fesom_ssh] CG hit maxiter=%d without converging "
                    "(last residual ~%.4e, rtol=%.4e)\n",
                    si->maxiter, (double)sqrt(s_old/(real_t)N_global), (double)rtol);
            fflush(stderr);
        }
        FESOM_DIE("CG did not converge");
    }
    if (si->partit && si->partit->npes > 1) EXCH(X);
    si->last_iters = iter;
    #undef EXCH
    #undef ALLREDUCE_SUM
    return iter;
}

/* ======================================================================== *
 *  M4.2-b — DEVICE (Kokkos) twins of the §5 SSH block (substeps 7-8).        *
 *                                                                            *
 *  The CG keeps HOST loop control (the convergence scalars come from the     *
 *  per-iteration MPI_Allreduce, which lives on the host) and launches one    *
 *  device kernel per vector op:                                              *
 *    - SpMV  App = A·p   = a per-ROW CSR GATHER (each output row reads its    *
 *      own matrix row + gathers v at colind → race-free, the inner sum        *
 *      sequential per row → Serial AND OpenMP bit-identical, like the M2      *
 *      gathers; it is NOT a scatter).                                         *
 *    - the dot products = the FIRST Kokkos::parallel_reduce in the port.      *
 *      Serial reduces sequentially in row order == the C `for` sum → bit-     *
 *      identical; OpenMP/CUDA use a tree reduction → climate-close (the FP    *
 *      reduction associativity is the GPU non-determinism source here, D22    *
 *      ladder). The scalar MPI_Allreduce on the result is UNCHANGED.          *
 *    - the AXPYs / copies = race-free maps.                                   *
 *  The per-iteration p / r / X halo exchanges stay HOST-STAGED (device→host→  *
 *  MPI→host→device) as D21 brackets OWNED by the CG (the KPP/FCT pattern),    *
 *  no-op at npes==1; GPU-aware MPI is deferred to M5. The driver owns the     *
 *  IN/OUT rails (push ssh_rhs/d_eta in, sync_host d_eta out).                 *
 * ======================================================================== */

using DV  = fesom::Field::dev_view_t;
using IDV = fesom::IntField::dev_view_t;

/* M5.2 perf: cumulative wall spent in the CG iteration loop over the timed window
 * (reset by fesom_main at the loop-timer warmup boundary). Lets the loop timer
 * report the CG's SHARE of the step — the GPU's dominant cost (L47/M5.1b). */
double g_fesom_cg_wall  = 0.0;
long   g_fesom_cg_iters = 0;

/* y = A·v over owned rows [0,N): per-row CSR gather. Each row writes only y(row)
 * and reads its own matrix row [rowptr(row),rowptr(row+1)) + gathers v at colind
 * (which can reach into the halo → v must be halo-current before this call). The
 * inner sum is sequential per row, exactly the C csr_matvec inner loop, so this is
 * bit-identical on Serial AND OpenMP (race-free, no cross-thread reduction). */
static void cg_spmv(IDV rowptr, IDV colind, DV vals, DV v, DV y, int N)
{
    Kokkos::parallel_for("fesom_cg_spmv", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            real_t s = 0.0;
            const int rstart = rowptr(row);
            const int rend   = rowptr(row + 1);
            for (int n = rstart; n < rend; ++n) s += vals(n) * v(colind(n));
            y(row) = s;
        });
}

/* Σ a(i)·b(i) over [0,N). The first parallel_reduce: Serial sums sequentially in
 * index order == the C `for` loop → bit-identical; OpenMP/CUDA climate-close. */
static real_t cg_dot(DV a, DV b, int N)
{
    real_t s = 0.0;
    Kokkos::parallel_reduce("fesom_cg_dot", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int i, real_t &l) { l += a(i) * b(i); }, s);
    return s;
}

/* (cg_dot2 removed in M5.2 — the in-loop sp0/sp1 dot is now fused into the
 * preconditioner SpMV; the only remaining dot is the single-accumulator cg_dot.) */

/* ======================================================================= *
 * M7 E.CG1 — FESOM_SPEED_CGPIPE: single-exchange 2-ring PCG (opt-in _exp).
 *
 * BACKEND-AGNOSTIC ON PURPOSE (pure Kokkos + MPI — no CUDA-specific API): on
 * Serial the Views live in host space and MPI moves host pointers, so the
 * FORCE_SERIAL byte proof (Serial np>1, knob ON vs the certified baseline,
 * diff_snap rc=0) exercises the REAL lever end to end — the strongest gate
 * this lever has, since full-model CUDA runs are not run-to-run byte-stable
 * (atomic scatters elsewhere in the step, D22). Production Serial stays
 * legacy: fesom_speed resolve forces the knob OFF on Serial builds without
 * FESOM_SPEED_FORCE_SERIAL=1.
 *
 * WHY (session-11 E ledger): the CG is the halo pool's #1 site — 2 exchanges
 * per iteration (pp before the SpMV, rr before the preconditioner SpMV) x ~72
 * iters = 146 sync points/step at ~190-260 us each (27/38 ms at 4N/16N).
 * Both operators are sparse and each needs its operand's fresh 1-ring halo, so
 * 2 exchanges/iter is STRUCTURAL for exact PCG on a 1-ring. On a 2-RING it is
 * not:
 *   - exchange rr on ring1+ring2 (ONE fused message per partner);
 *   - zz = M^-1 rr is then computable at owned AND ring1 rows (the ring1
 *     preconditioner rows are shipped VERBATIM from their owners once at
 *     setup — pr_values is set-once, never refreshed, valid under linfs AND
 *     zstar: the zstar increment touches `values` only, and A is only ever
 *     applied at owned rows);
 *   - pp = zz + be*pp then maintains its OWN ring1 halo by recurrence, and
 *     exch(pp) is DELETED. Exchanges/solve: 2+2k -> 2+k.
 *
 * BYTE-IDENTITY (the lever's hard gate, pre-registered in the session-11
 * findings §5): ring1 rr bytes = owner bytes (same com lists as the 1-ring
 * path); zz at a ring1 row = the owner's zz BITWISE (verbatim row: same
 * coefficients, same column ORDER -> same summation order, same loop shape in
 * the same TU -> same FMA contraction; operand bytes equal); pp at ring1 =
 * owner's pp by induction from pp0 = zz0 with globally-identical be. All
 * owned-row kernels are byte-unchanged => d_eta identical every step => the
 * MODEL is byte-identical ON vs OFF (gate: diff_snap rc=0, multi-rank CUDA;
 * corollary: iteration counts + residual prints match exactly).
 *
 * SETUP (one-time, lazy, collective — all ranks enter solve together):
 *   A. every owner ships, to each com_nod2D send-partner, the preconditioner
 *      CSR rows of the nodes in its slist block (row lengths, colind as
 *      1-based GLOBAL ids in row order, pr_values slices) — 3 staged msgs;
 *   B. the receiver translates gids: owned/ring1 via myList_nod2D, anything
 *      else becomes a RING2 slot appended at N+eDim+t (rr_fld is re-alloc'd
 *      to N+eDim+nring2); ring2 owners come from partit->part (global
 *      partition vector, binary search);
 *   C. ring2 want-lists per owner; MPI_Alltoall(counts) handshake (ring2 can
 *      introduce NEW diagonal partner ranks); owners translate the want gids
 *      to their owned indices -> ring2 send lists;
 *   D. flat per-partner send/recv lists = [existing 1-ring block] ++ [ring2
 *      block], pushed to device once. The per-iteration exchange is then the
 *      standard pack -> fence -> Irecv/Isend -> Waitall -> unpack, one message
 *      per partner, tag 2100 (the fesom_halo_device pattern, M5.17 prof hooks
 *      included).
 *
 * NOFENCE2-audit parity (fesom_halo_device.cpp items 1-4): consumers of the
 * unpack are same-stream kernels; there are no mid-step host readers; the
 * UNCONDITIONAL pre-MPI fence in THIS function drains the previous unpack
 * before rbuf is re-posted; the buffers are CG-private and sized ONCE at
 * setup (no realloc after warmup) => no post-unpack fence.
 * ======================================================================= */
namespace {

struct CgPipeState {
    bool built = false;
    int  N = 0, eDim = 0, nring2 = 0;
    int  nsend = 0, nrecv = 0;
    std::vector<int> partner;            /* partner ranks, ascending (message order) */
    std::vector<int> soff, roff;         /* per-partner offsets into sbuf/rbuf [P+1] */
    Kokkos::View<int*>    sidx_d;        /* [nsend] local slots to pack   */
    Kokkos::View<int*>    ridx_d;        /* [nrecv] local slots to unpack */
    Kokkos::View<double*> sbuf_d, rbuf_d;
    std::vector<MPI_Request> reqs;
    /* ring1 preconditioner CSR: row r (= local slot N+r), cols are LOCAL slots
     * into the extended rr; entries in the OWNER's row order (byte-identity). */
    Kokkos::View<int*>    rp2_d;         /* [eDim+1] */
    Kokkos::View<int*>    ci2_d;
    Kokkos::View<double*> pv2_d;
};
CgPipeState g_cgpipe;

void cgpipe_build(const fesom_ssh_stiff *S, fesom_solverinfo *si,
                  const struct fesom_mesh *mesh, fesom_partit *p)
{
    const fesom_com_struct *cs = &p->com_nod2D;
    const int N    = mesh->myDim_nod2D;
    const int eDim = mesh->eDim_nod2D;
    MPI_Comm comm  = p->MPI_COMM_FESOM;
    const int npes = p->npes;

    /* gid -> local for owned + ring1 (myList_nod2D holds 1-based gids, [0, N+eDim)). */
    std::unordered_map<int, int> g2l;
    g2l.reserve((size_t)(N + eDim) * 2);
    for (int l = 0; l < N + eDim; ++l) g2l.emplace(p->myList_nod2D[l], l);

    /* OWNER rank of every LOCAL node: [0,N) = me; ring1 slots = the com_nod2D
     * provider that delivers them (rptr block k -> rPE[k]). Shipped alongside
     * each column gid — ownership must come from the com graph, NOT from
     * partit->part ranges (global ids are NOT contiguous per rank; the range
     * test mis-assigns owners — caught by the SELF-owned FESOM_CHECK, pi np2). */
    std::vector<int> owner_l((size_t)(N + eDim), p->mype);
    for (int k = 0; k < cs->rPEnum; ++k)
        for (int j = cs->rptr[k] - 1; j < cs->rptr[k + 1] - 1; ++j)
            owner_l[(size_t)(cs->rlist[j] - 1)] = cs->rPE[k];

    /* ---- A. ship pr rows: send my slist blocks' rows; receive my rlist blocks' rows. */
    struct Bundle { std::vector<int> ints; std::vector<double> dbls; int hdr[2]; };
    std::vector<Bundle> sb(cs->sPEnum), rb(cs->rPEnum);
    std::vector<MPI_Request> rq;
    rq.reserve((size_t)(cs->sPEnum + cs->rPEnum));

    for (int k = 0; k < cs->rPEnum; ++k) {                    /* headers in */
        rq.push_back(MPI_Request());
        MPI_Irecv(rb[k].hdr, 2, MPI_INT, cs->rPE[k], 2101, comm, &rq.back());
    }
    for (int k = 0; k < cs->sPEnum; ++k) {                    /* build + headers out */
        Bundle &b = sb[k];
        const int j0 = cs->sptr[k] - 1, j1 = cs->sptr[k + 1] - 1;
        const int nrows = j1 - j0;
        b.ints.push_back(nrows);                              /* [nrows, len_0.., gids..] */
        int nent = 0;
        for (int j = j0; j < j1; ++j) {
            const int row = cs->slist[j] - 1;                 /* 1-based -> owned idx */
            const int a = S->rowptr[row], e = S->rowptr[row + 1];
            b.ints.push_back(e - a);
            nent += e - a;
        }
        for (int j = j0; j < j1; ++j) {
            const int row = cs->slist[j] - 1;
            for (int n = S->rowptr[row]; n < S->rowptr[row + 1]; ++n) {
                const int col = S->colind[n];                      /* col < N+eDim always */
                b.ints.push_back(p->myList_nod2D[col]);            /* col gid */
                b.ints.push_back(owner_l[(size_t)col]);            /* col OWNER rank */
                b.dbls.push_back((double)S->pr_values[n]);         /* VERBATIM, row order */
            }
        }
        b.hdr[0] = nrows; b.hdr[1] = nent;
        rq.push_back(MPI_Request());
        MPI_Isend(b.hdr, 2, MPI_INT, cs->sPE[k], 2101, comm, &rq.back());
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    rq.clear();

    for (int k = 0; k < cs->rPEnum; ++k) {                    /* payloads */
        const int nrows_exp = (cs->rptr[k + 1] - 1) - (cs->rptr[k] - 1);
        FESOM_CHECK(rb[k].hdr[0] == nrows_exp,
                    "cgpipe: provider %d shipped %d rows, expected %d",
                    cs->rPE[k], rb[k].hdr[0], nrows_exp);
        rb[k].ints.resize((size_t)1 + rb[k].hdr[0] + 2 * (size_t)rb[k].hdr[1]);
        rb[k].dbls.resize((size_t)rb[k].hdr[1]);
        rq.push_back(MPI_Request());
        MPI_Irecv(rb[k].ints.data(), (int)rb[k].ints.size(), MPI_INT,    cs->rPE[k], 2102, comm, &rq.back());
        rq.push_back(MPI_Request());
        MPI_Irecv(rb[k].dbls.data(), (int)rb[k].dbls.size(), MPI_DOUBLE, cs->rPE[k], 2103, comm, &rq.back());
    }
    for (int k = 0; k < cs->sPEnum; ++k) {
        rq.push_back(MPI_Request());
        MPI_Isend(sb[k].ints.data(), (int)sb[k].ints.size(), MPI_INT,    cs->sPE[k], 2102, comm, &rq.back());
        rq.push_back(MPI_Request());
        MPI_Isend(sb[k].dbls.data(), (int)sb[k].dbls.size(), MPI_DOUBLE, cs->sPE[k], 2103, comm, &rq.back());
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    rq.clear();
    sb.clear();

    /* ---- B. translate rows into per-ring1-slot CSR; discover ring2. */
    std::vector<std::vector<int>>    row_ci((size_t)eDim);
    std::vector<std::vector<double>> row_pv((size_t)eDim);
    std::unordered_map<int, int> g2r2;                        /* gid -> ring2 ordinal */
    std::vector<int> r2gid, r2owner;
    for (int k = 0; k < cs->rPEnum; ++k) {
        const Bundle &b = rb[k];
        const int nrows = b.hdr[0];
        size_t ip = (size_t)1 + nrows;                        /* (gid,owner) stream cursor */
        size_t dp = 0;
        for (int j = 0; j < nrows; ++j) {
            const int slot = cs->rlist[(cs->rptr[k] - 1) + j] - 1;   /* local, [N, N+eDim) */
            FESOM_CHECK(slot >= N && slot < N + eDim,
                        "cgpipe: rlist slot %d outside ring1", slot);
            const int r   = slot - N;
            const int len = b.ints[(size_t)1 + j];
            row_ci[r].reserve(len); row_pv[r].reserve(len);
            for (int q = 0; q < len; ++q) {
                const int gid = b.ints[ip++];
                const int own = b.ints[ip++];
                int loc;
                auto it = g2l.find(gid);
                if (it != g2l.end()) { loc = it->second; }
                else {
                    FESOM_CHECK(own != p->mype,
                                "cgpipe: shipped gid %d claims MY ownership but is not local", gid);
                    auto r2 = g2r2.emplace(gid, (int)r2gid.size());
                    if (r2.second) { r2gid.push_back(gid); r2owner.push_back(own); }
                    else FESOM_CHECK(r2owner[(size_t)r2.first->second] == own,
                                     "cgpipe: gid %d shipped with conflicting owners %d/%d",
                                     gid, r2owner[(size_t)r2.first->second], own);
                    loc = N + eDim + r2.first->second;
                }
                row_ci[r].push_back(loc);
                row_pv[r].push_back(b.dbls[dp++]);
            }
        }
        FESOM_CHECK(dp == b.dbls.size() && ip == b.ints.size(),
                    "cgpipe: bundle from %d not fully consumed", cs->rPE[k]);
    }
    rb.clear();
    const int nring2 = (int)r2gid.size();

    /* ---- C. ring2 want-lists by owner; Alltoall handshake (new partners possible). */
    std::vector<std::vector<int>> want((size_t)npes);
    for (int t = 0; t < nring2; ++t)
        want[(size_t)r2owner[(size_t)t]].push_back(r2gid[(size_t)t]);   /* slot order within owner */
    std::vector<int> scnt((size_t)npes, 0), rcnt((size_t)npes, 0);
    for (int T = 0; T < npes; ++T) scnt[(size_t)T] = (int)want[(size_t)T].size();
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, comm);
    std::vector<std::vector<int>> wantin((size_t)npes);
    for (int Q = 0; Q < npes; ++Q) {
        if (rcnt[(size_t)Q] > 0) {
            wantin[(size_t)Q].resize((size_t)rcnt[(size_t)Q]);
            rq.push_back(MPI_Request());
            MPI_Irecv(wantin[(size_t)Q].data(), rcnt[(size_t)Q], MPI_INT, Q, 2104, comm, &rq.back());
        }
        if (scnt[(size_t)Q] > 0) {
            rq.push_back(MPI_Request());
            MPI_Isend(want[(size_t)Q].data(), scnt[(size_t)Q], MPI_INT, Q, 2104, comm, &rq.back());
        }
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    rq.clear();
    std::vector<std::vector<int>> slist2((size_t)npes);       /* ring2 SEND lists (owned idx) */
    for (int Q = 0; Q < npes; ++Q) {
        slist2[(size_t)Q].reserve(wantin[(size_t)Q].size());
        for (int gid : wantin[(size_t)Q]) {
            auto it = g2l.find(gid);
            FESOM_CHECK(it != g2l.end() && it->second < N,
                        "cgpipe: rank %d wants gid %d that is not my owned node", Q, gid);
            slist2[(size_t)Q].push_back(it->second);
        }
    }

    /* ---- D. flat per-partner lists = [1-ring block] ++ [ring2 block]. */
    std::vector<int> s1k((size_t)npes, -1), r1k((size_t)npes, -1);
    for (int k = 0; k < cs->sPEnum; ++k) s1k[(size_t)cs->sPE[k]] = k;
    for (int k = 0; k < cs->rPEnum; ++k) r1k[(size_t)cs->rPE[k]] = k;
    /* ring2 recv blocks grouped by owner, slot order = want order. */
    std::vector<std::vector<int>> rslot2((size_t)npes);
    for (int t = 0; t < nring2; ++t)
        rslot2[(size_t)r2owner[(size_t)t]].push_back(N + eDim + t);

    CgPipeState &s = g_cgpipe;
    s.N = N; s.eDim = eDim; s.nring2 = nring2;
    s.partner.clear(); s.soff.assign(1, 0); s.roff.assign(1, 0);
    std::vector<int> sidx, ridx;
    for (int P = 0; P < npes; ++P) {
        const bool has = s1k[(size_t)P] >= 0 || r1k[(size_t)P] >= 0
                      || !slist2[(size_t)P].empty() || !rslot2[(size_t)P].empty();
        if (!has) continue;
        s.partner.push_back(P);
        if (s1k[(size_t)P] >= 0) {
            const int k = s1k[(size_t)P];
            for (int j = cs->sptr[k] - 1; j < cs->sptr[k + 1] - 1; ++j)
                sidx.push_back(cs->slist[j] - 1);
        }
        for (int l : slist2[(size_t)P]) sidx.push_back(l);
        if (r1k[(size_t)P] >= 0) {
            const int k = r1k[(size_t)P];
            for (int j = cs->rptr[k] - 1; j < cs->rptr[k + 1] - 1; ++j)
                ridx.push_back(cs->rlist[j] - 1);
        }
        for (int slot : rslot2[(size_t)P]) ridx.push_back(slot);
        s.soff.push_back((int)sidx.size());
        s.roff.push_back((int)ridx.size());
    }
    s.nsend = (int)sidx.size();
    s.nrecv = (int)ridx.size();
    s.reqs.assign(2 * s.partner.size(), MPI_Request());

    /* ---- E. device pushes + the rr extension. */
    auto push_i = [](const char *lbl, const std::vector<int> &v) {
        /* std::string(lbl): Kokkos' is_view_label rejects a const char* VARIABLE
         * (literals work only because they are char[N]) — the grow() idiom. */
        Kokkos::View<int*> d(std::string(lbl), v.size());
        auto h = Kokkos::create_mirror_view(d);
        for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
        Kokkos::deep_copy(d, h);
        return d;
    };
    s.sidx_d = push_i("cgpipe.sidx", sidx);
    s.ridx_d = push_i("cgpipe.ridx", ridx);
    s.sbuf_d = Kokkos::View<double*>("cgpipe.sbuf", (size_t)s.nsend);
    s.rbuf_d = Kokkos::View<double*>("cgpipe.rbuf", (size_t)s.nrecv);

    std::vector<int> rp2((size_t)eDim + 1, 0), ci2;
    std::vector<double> pv2;
    for (int r = 0; r < eDim; ++r) {
        rp2[(size_t)r + 1] = rp2[(size_t)r] + (int)row_ci[(size_t)r].size();
        ci2.insert(ci2.end(), row_ci[(size_t)r].begin(), row_ci[(size_t)r].end());
        pv2.insert(pv2.end(), row_pv[(size_t)r].begin(), row_pv[(size_t)r].end());
    }
    s.rp2_d = push_i("cgpipe.rp2", rp2);
    s.ci2_d = push_i("cgpipe.ci2", ci2);
    {
        Kokkos::View<double*> d("cgpipe.pv2", pv2.size());
        auto h = Kokkos::create_mirror_view(d);
        for (size_t i = 0; i < pv2.size(); ++i) h(i) = pv2[i];
        Kokkos::deep_copy(d, h);
        s.pv2_d = d;
    }

    /* rr gains the ring2 tail: [0,N) owned | [N,N+eDim) ring1 | [.., +nring2) ring2.
     * Fresh zeroed alloc is safe: every solve writes owned rr before the first
     * exchange, and ring1/ring2 are filled by the exchange before any read. */
    si->rr_fld.alloc("ssh.cg.rr", (size_t)(N + eDim + nring2));
    si->rr = si->rr_fld.h();

    long loc[4] = { (long)nring2, (long)s.partner.size(),
                    (long)(cs->sPEnum > cs->rPEnum ? cs->sPEnum : cs->rPEnum),
                    (long)rp2[(size_t)eDim] };
    long mx[4];
    MPI_Reduce(loc, mx, 4, MPI_LONG, MPI_MAX, 0, comm);
    if (p->mype == 0)
        fprintf(stderr, "[cgpipe] built: ring2(max)=%ld partners(max)=%ld ring1-partners(max)=%ld "
                        "shipped-nnz(max)=%ld — 2-ring single-exchange CG ACTIVE\n",
                mx[0], mx[1], mx[2], mx[3]);
    s.built = true;
}

/* The per-iteration fused 2-ring rr exchange. Pattern + fence discipline =
 * fesom_halo_exchange_device (see the banner above for the no-post-fence audit). */
void cgpipe_exchange_rr(fesom::Field &rr_fld, fesom_partit *p)
{
    CgPipeState &s = g_cgpipe;
    fesom_halo_prof_barrier(p);                       /* M5.17 split instrumentation parity */
    auto rr = rr_fld.d();
    {
        auto sidx = s.sidx_d; auto sbuf = s.sbuf_d;
        if (s.nsend > 0)
            Kokkos::parallel_for("fesom_cgpipe_pack", Kokkos::RangePolicy<>(0, s.nsend),
                KOKKOS_LAMBDA(const int i) { sbuf(i) = rr(sidx(i)); });
    }
    Kokkos::fence();   /* MANDATORY pre-MPI: MPI reads sbuf_d + re-posts rbuf_d (drains prev unpack) */

    int nreq = 0;
    double *sp = s.sbuf_d.data();
    double *rp = s.rbuf_d.data();
    for (size_t q = 0; q < s.partner.size(); ++q) {
        const int rc = s.roff[q + 1] - s.roff[q];
        if (rc > 0)
            MPI_Irecv(rp + s.roff[q], rc, MPI_DOUBLE, s.partner[q], 2100,
                      p->MPI_COMM_FESOM, &s.reqs[(size_t)nreq++]);
    }
    for (size_t q = 0; q < s.partner.size(); ++q) {
        const int sc = s.soff[q + 1] - s.soff[q];
        if (sc > 0)
            MPI_Isend(sp + s.soff[q], sc, MPI_DOUBLE, s.partner[q], 2100,
                      p->MPI_COMM_FESOM, &s.reqs[(size_t)nreq++]);
    }
    fesom_halo_prof_bytes(8.0 * (double)(s.nsend + s.nrecv));
    fesom_halo_prof_waitall(nreq, s.reqs.data());

    {
        auto ridx = s.ridx_d; auto rbuf = s.rbuf_d;
        if (s.nrecv > 0)
            Kokkos::parallel_for("fesom_cgpipe_unpack", Kokkos::RangePolicy<>(0, s.nrecv),
                KOKKOS_LAMBDA(const int i) { rr(ridx(i)) = rbuf(i); });
    }
    /* no post-unpack fence — see the NOFENCE2-audit-parity note in the banner. */
    rr_fld.modify_device();
}

/* zz at ring1 rows from the shipped preconditioner rows. Loop body shape is
 * IDENTICAL to cg_spmv / the fused psolve (same TU) so the FMA contraction —
 * and therefore the bits — match the owner's owned-row computation. */
void cgpipe_zz_ring1(DV rr, DV zz)
{
    CgPipeState &s = g_cgpipe;
    if (s.eDim <= 0) return;
    const int N = s.N;
    auto rp2 = s.rp2_d; auto ci2 = s.ci2_d; auto pv2 = s.pv2_d;
    Kokkos::parallel_for("fesom_cgpipe_zz_ring1", Kokkos::RangePolicy<>(0, s.eDim),
        KOKKOS_LAMBDA(const int r) {
            real_t sacc = 0.0;
            const int a = rp2(r), e = rp2(r + 1);
            for (int n = a; n < e; ++n) sacc += pv2(n) * rr(ci2(n));
            zz(N + r) = sacc;
        });
}

/* FESOM_CGPIPE_SELFCHECK=1 — bring-up validator: after each pp update, diff the
 * RECURRED pp ring1 against a reference host 1-ring exchange of the same owned
 * data. Byte-identity claim => max|Δ| MUST print 0.000e+00. Diagnostic only. */
bool cgpipe_selfcheck_on()
{
    static int c = -1;
    if (c < 0) { const char *e = getenv("FESOM_CGPIPE_SELFCHECK"); c = (e && e[0] == '1') ? 1 : 0; }
    return c != 0;
}

void cgpipe_selfcheck_pp(fesom_solverinfo *si, fesom_partit *p, int iter)
{
    const int N = g_cgpipe.N, eDim = g_cgpipe.eDim;
    si->pp_fld.modify_device();
    si->pp_fld.sync_host();
    const real_t *pph = si->pp;
    std::vector<real_t> ref(pph, pph + (size_t)(N + eDim));
    fesom_halo_exchange(ref.data(), FESOM_HALO_NOD2D, 1, 1, p);   /* reference ring1 */
    double maxd = 0.0;
    for (int l = N; l < N + eDim; ++l) {
        const double d = fabs((double)pph[l] - (double)ref[(size_t)l]);
        if (d > maxd) maxd = d;
    }
    double gmax = 0.0;
    MPI_Allreduce(&maxd, &gmax, 1, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);
    if (p->mype == 0)
        fprintf(stderr, "[cgpipe-selfcheck] iter %4d: max|recurred pp - exchanged pp| = %.3e%s\n",
                iter, gmax, gmax == 0.0 ? "" : "  <-- MUST BE 0: BYTE-IDENTITY BROKEN");
}

} // namespace

/* free the persistent CGPIPE Views BEFORE Kokkos::finalize() (fesom_main.cpp
 * teardown, next to fesom_halo_device_free — same static-destruction hazard). */
void fesom_ssh_cgpipe_free(void)
{
    CgPipeState &s = g_cgpipe;
    s.sidx_d = Kokkos::View<int*>();
    s.ridx_d = Kokkos::View<int*>();
    s.sbuf_d = Kokkos::View<double*>();
    s.rbuf_d = Kokkos::View<double*>();
    s.rp2_d  = Kokkos::View<int*>();
    s.ci2_d  = Kokkos::View<int*>();
    s.pv2_d  = Kokkos::View<double*>();
    s.reqs.clear();
    s.partner.clear(); s.soff.clear(); s.roff.clear();
    s.built = false;
}

void fesom_compute_ssh_rhs_linfs_kk(const struct fesom_mesh    *mesh,
                                    struct fesom_dyn           *dyn,
                                    const struct fesom_forcing *forcing)  /* M6.3: water_flux
                                          feeds the zstar tail; unused under linfs */
{
    const int    N       = mesh->myDim_nod2D;
    const int    N_alloc = mesh->myDim_nod2D + mesh->eDim_nod2D;
    const int    E       = mesh->myDim_edge2D;
    const int    nl      = mesh->nl;
    const real_t alpha           = (real_t)FESOM_PHASE1_ALPHA;
    const real_t one_minus_alpha = 1.0 - alpha;

    auto edges    = mesh->edges_fld.d();
    auto edge_tri = mesh->edge_tri_fld.d();
    auto ecd      = mesh->edge_cross_dxdy_fld.d();
    auto ulev     = mesh->ulevels_fld.d();
    auto nlev     = mesh->nlevels_fld.d();
    auto uv       = dyn->uv_fld.d();
    auto uv_rhs   = dyn->uv_rhs_fld.d();
    auto helem    = mesh->helem_fld.d();
    auto ssh_rhs     = dyn->ssh_rhs_fld.d();
    auto ssh_rhs_old = dyn->ssh_rhs_old_fld.d();

    /* zero ssh_rhs over the full local extent (the C memset over N_alloc). */
    Kokkos::parallel_for("fesom_ssh_rhs_zero", Kokkos::RangePolicy<>(0, N_alloc),
        KOKKOS_LAMBDA(const int n) { ssh_rhs(n) = 0.0; });

    /* edge loop — EDGE→NODE SCATTER (atomic_add, D22). The per-edge level sums for c1/c2
     * are sequential → deterministic; the scatter into ssh_rhs(n1)/(n2) uses atomic_add in
     * natural edge order → Serial bit-identical, OpenMP/CUDA climate-close. */
    Kokkos::parallel_for("fesom_ssh_rhs_edge", Kokkos::RangePolicy<>(0, E),
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
                    real_t u  = uv    (FESOM_ELEMVEC(el1, nz, nl) + 0);
                    real_t v  = uv    (FESOM_ELEMVEC(el1, nz, nl) + 1);
                    real_t ur = uv_rhs(FESOM_ELEMVEC(el1, nz, nl) + 0);
                    real_t vr = uv_rhs(FESOM_ELEMVEC(el1, nz, nl) + 1);
                    real_t h  = helem (FESOM_ELEM3D(el1, nz, nl));
                    c1 += alpha * ((v + vr) * dx1 - (u + ur) * dy1) * h;
                }
            }
            real_t c2 = 0.0;
            if (el2 >= 0) {
                real_t dx2 = ecd(4*ed + 2);
                real_t dy2 = ecd(4*ed + 3);
                int nzmin = ulev(el2) - 1;
                int nzmax = nlev(el2) - 1;
                for (int nz = nzmin; nz < nzmax; ++nz) {
                    real_t u  = uv    (FESOM_ELEMVEC(el2, nz, nl) + 0);
                    real_t v  = uv    (FESOM_ELEMVEC(el2, nz, nl) + 1);
                    real_t ur = uv_rhs(FESOM_ELEMVEC(el2, nz, nl) + 0);
                    real_t vr = uv_rhs(FESOM_ELEMVEC(el2, nz, nl) + 1);
                    real_t h  = helem (FESOM_ELEM3D(el2, nz, nl));
                    c2 -= alpha * ((v + vr) * dx2 - (u + ur) * dy2) * h;
                }
            }
            Kokkos::atomic_add(&ssh_rhs(n1),  (c1 + c2));
            Kokkos::atomic_add(&ssh_rhs(n2), -(c1 + c2));    /* L32: parenthesise the negation */
        });

    /* Water-flux tail (oce_ale.F90:2122-2143).
     *  zstar (non-linfs), no-cavity arm (:2133):
     *      ssh_rhs(n) += -alpha*water_flux(n)*areasvol(nzmin,n) + (1-alpha)*ssh_rhs_old(n)
     *  linfs (:2138-2142):
     *      ssh_rhs(n) += (1-alpha)*ssh_rhs_old(n)                        <- the v1.0 path
     * (The cavity arm's use_cavity_fw2press gate is unreachable -- no cavities in our meshes.) */
    if (fesom_ale_is_zstar()) {
        auto wf       = forcing->water_flux_fld.d();
        auto areasvol = mesh->areasvol_fld.d();
        auto ulev_n   = mesh->ulevels_nod2D_fld.d();
        Kokkos::parallel_for("fesom_ssh_rhs_zstar", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int n) {
                const int nzmin_f = ulev_n(n);                 /* 1-based */
                if (nzmin_f > 1) return;                       /* cavity arm: not ported */
                ssh_rhs(n) += -alpha * wf(n)
                                * areasvol(FESOM_NODE3D(n, nzmin_f - 1, nl))
                            + one_minus_alpha * ssh_rhs_old(n);
            });
    } else {
        Kokkos::parallel_for("fesom_ssh_rhs_linfs", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int n) { ssh_rhs(n) += one_minus_alpha * ssh_rhs_old(n); });
    }

    dyn->ssh_rhs_fld.modify_device();   /* driver sync_host()s before the nod2D halo */
}

/*===========================================================================================
 * M6.3 (zstar) — update_stiff_mat_ale (oce_ale.F90:1892-2001), DEVICE kernel.
 *
 * Per-step CUMULATIVE increment of the SSH stiffness:
 *   factor = g*dt*alpha*theta
 *   per OWNED edge, per adjacent element i in {1,2}, per row j in {1,2} (halo rows skipped):
 *     fy(k) = -dhe(elem) * ( dN_k/dx * dy_i - dN_k/dy * dx_i ),   k = 1..3
 *     i==2 -> fy = -fy ;  the j==2 row takes -fy
 *     values(npos(k)) += fy(k) * factor
 * i.e. exactly the BASE-matrix edge assembly with the static column depth replaced by the
 * per-step elemental delta-hbar = dhe (filled by the PREVIOUS step's compute_hbar).
 *
 * ⚠️ INVARIANT 1: the base matrix is NEVER rebuilt -- this INCREMENTS the same CSR object, so
 * sum(dhe) over steps == hbar - hbar(init). At cold start dhe == 0, so the step-1 update is a
 * no-op by construction.
 *
 * ⚠️ PRECONDITIONER: the Fortran builds it ONCE at the first solve (oce_ale.F90:3306,
 * `if (lfirst) call ssh_solve_preconditioner`) and NEVER refreshes it as the matrix evolves.
 * The C verified this and does nothing here. So do we: pr_values is NOT touched. Refreshing it
 * would be "fixing" the reference.
 *
 * Device shape: the C uses a per-row `spos` scatter table (nodeid -> sparse position). That is
 * a full [myDim+eDim] int array -- untenable per-thread. Instead each (row, column) does a
 * LINEAR SEARCH over the row's colind range. CSR rows have UNIQUE column indices, so the match
 * is unique and the result is identical to the spos lookup (rows are ~7 entries wide here).
 * Accumulation is edge-order Kokkos::atomic_add (D22): on Serial the range is sequential, so
 * the accumulation order is exactly the C's loop order => bit-identical.
 *===========================================================================================*/
void fesom_update_stiff_mat_ale_kk(fesom_ssh_stiff *S, const struct fesom_mesh *mesh)
{
    const int N = mesh->myDim_nod2D;
    const int E = mesh->myDim_edge2D;
    const int Eo = mesh->myDim_elem2D;
    const real_t factor = (real_t)FESOM_G * (real_t)FESOM_PHASE1_DT
                        * (real_t)FESOM_PHASE1_ALPHA * (real_t)FESOM_PHASE1_THETA;

    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    auto values = S->values_fld.d();
    auto edge_tri = mesh->edge_tri_fld.d();
    auto edges    = mesh->edges_fld.d();
    auto elnod    = mesh->elem_nodes_fld.d();
    auto gsca     = mesh->gradient_sca_fld.d();
    auto ecdxdy   = mesh->edge_cross_dxdy_fld.d();
    auto dhe      = mesh->dhe_fld.d();

    Kokkos::parallel_for("fesom_update_stiff_mat_ale", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int ed) {
            const int el[2]  = { edge_tri(2*ed + 0), edge_tri(2*ed + 1) };
            const int e_n[2] = { edges(2*ed + 0),    edges(2*ed + 1)    };
            for (int i = 0; i < 2; ++i) {
                /* Fortran: if (elem < 1) cycle. Same defensive bound as the base builder:
                 * adjacent elements of OWNED edges are owned. */
                if (el[i] < 0 || el[i] >= Eo) continue;
                const int en[3] = { elnod(3*el[i] + 0), elnod(3*el[i] + 1), elnod(3*el[i] + 2) };
                const size_t g  = (size_t)6 * el[i];
                const real_t dx_i = ecdxdy(4*ed + 2*i + 0);
                const real_t dy_i = ecdxdy(4*ed + 2*i + 1);

                real_t fy[3];
                for (int k = 0; k < 3; ++k)
                    fy[k] = -dhe(el[i]) * (gsca(g + k) * dy_i - gsca(g + 3 + k) * dx_i);
                if (i == 1) { fy[0] = -fy[0]; fy[1] = -fy[1]; fy[2] = -fy[2]; }   /* Fortran i==2 */

                /* j = 1 row (+fy) and j = 2 row (-fy); halo rows skipped
                 * (Fortran: if (row > myDim_nod2D) cycle). */
                for (int j = 0; j < 2; ++j) {
                    const int row = e_n[j];
                    if (row >= N) continue;
                    const real_t sgn = (j == 0) ? 1.0 : -1.0;
                    for (int k = 0; k < 3; ++k) {
                        /* linear search for the sparse position of column en[k] in this row --
                         * identical to the C's spos[] lookup (CSR columns are unique per row). */
                        for (int q = rowptr(row); q < rowptr(row + 1); ++q) {
                            if (colind(q) == en[k]) {
                                Kokkos::atomic_add(&values(q), sgn * fy[k] * factor);
                                break;
                            }
                        }
                    }
                }
            }
        });
    S->values_fld.modify_device();
    /* pr_values deliberately NOT refreshed -- see the banner above. */
}

int fesom_ssh_solve_cg_kk(const fesom_ssh_stiff *S,
                          fesom_solverinfo      *si,
                          const struct fesom_mesh *mesh,
                          struct fesom_dyn        *dyn)
{
    const int     N        = mesh->myDim_nod2D;
    const real_t  soltol   = si->soltol;
    fesom_partit *partit   = si->partit;
    const int     parallel = (partit && partit->npes > 1);
    const int     N_global = parallel ? mesh->nod2D : N;

    /* M7 E.CG1 — FESOM_SPEED_CGPIPE (opt-in _exp; see the banner above cg_dot).
     * Resolve FIRST (announces itself, L80), then the activity conjunction; a
     * requested-but-inactive knob warns loudly instead of dying silent. The
     * one-time setup re-allocs rr_fld (ring2 tail) so it MUST run before the
     * device views are taken below. */
    static int s_cgpipe = -1;
    const bool cgpipe_env = fesom_speed_on_exp("CGPIPE", &s_cgpipe);
#ifdef KOKKOS_ENABLE_CUDA
    const bool transport_ok = fesom_halo_device_active();   /* keep the debug toggle coherent */
#else
    const bool transport_ok = true;   /* Serial: host Views + host MPI (the FORCE_SERIAL proof) */
#endif
    const bool cgpipe = cgpipe_env && parallel && transport_ok;
    if (cgpipe_env && !cgpipe) {
        static bool warned = false;
        if (!warned && (!partit || partit->mype == 0)) {
            fprintf(stderr, "[cgpipe] !! FESOM_SPEED_CGPIPE requested but INACTIVE "
                            "(npes==1 or FESOM_HOST_HALO=1) — running the 2-exchange CG\n");
            fflush(stderr);
        }
        warned = true;
    }
    if (cgpipe && !g_cgpipe.built) cgpipe_build(S, si, mesh, partit);

    /* device views (set-once CSR + the warm-start X / rhs / scratch vectors). */
    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    auto vals   = S->values_fld.d();
    auto prvals = S->pr_values_fld.d();
    auto X   = dyn->d_eta_fld.d();
    auto rhs = dyn->ssh_rhs_fld.d();
    auto rr  = si->rr_fld.d();
    auto zz  = si->zz_fld.d();
    auto pp  = si->pp_fld.d();
    auto App = si->App_fld.d();

    /* CG-owned internal halo bracket (device→host→MPI→host→device); no-op at npes==1.
     * The leading modify_device() captures the device-kernel write so sync_host() copies
     * it; sync_device() at the end re-pushes the halo'd field for the next SpMV (D21). */
    auto exch = [&](fesom::Field &f) {
        if (!parallel) return;
#ifdef KOKKOS_ENABLE_CUDA
        if (fesom_halo_device_active()) {            /* M5.1: device pack -> GPU-aware MPI -> device unpack */
            fesom_halo_exchange_device(f, FESOM_HALO_NOD2D, 1, 1, partit);
            return;
        }
#endif
        f.modify_device(); f.sync_host();            /* legacy host-staged (Serial/OpenMP; FESOM_HOST_HALO=1) */
        fesom_halo_exchange(f.h_checked(), FESOM_HALO_NOD2D, 1, 1, partit);
        f.modify_host();   f.sync_device();
    };
    #define ALLREDUCE_SUM(var) do { if (parallel) \
        MPI_Allreduce(MPI_IN_PLACE, &(var), 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); \
        } while (0)

    /* Initial ‖rhs‖² + tolerance (solver.F90:142-154). rhs is read at OWNED rows only. */
    real_t s_old = cg_dot(rhs, rhs, N);
    ALLREDUCE_SUM(s_old);
    real_t rtol = soltol * sqrt(s_old / (real_t)N_global);

    if (s_old == 0.0) {
        Kokkos::parallel_for("fesom_cg_zeroX", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) { X(row) = 0.0; });
        dyn->d_eta_fld.modify_device();
        si->last_iters = 0;
        return 0;
    }

    /* r0 = rhs - A·X. X must be halo-current before the SpMV gathers it at colind. */
    exch(dyn->d_eta_fld);                            /* solver.F90:421 EXCH(X) */
    cg_spmv(rowptr, colind, vals, X, rr, N);
    Kokkos::parallel_for("fesom_cg_r0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) { rr(row) = rhs(row) - rr(row); });
    if (cgpipe) cgpipe_exchange_rr(si->rr_fld, partit);   /* E.CG1: ONE fused 2-ring exchange */
    else        exch(si->rr_fld);                    /* solver.F90:424 EXCH(rr) */

    /* z0 = M⁻¹ r0 ; pp = z0. E.CG1: zz + pp additionally at ring1 (owned rows
     * byte-unchanged; ring1 rows from the shipped preconditioner CSR). */
    cg_spmv(rowptr, colind, prvals, rr, zz, N);
    if (cgpipe) cgpipe_zz_ring1(rr, zz);
    const int Next = cgpipe ? N + mesh->eDim_nod2D : N;
    Kokkos::parallel_for("fesom_cg_pp0", Kokkos::RangePolicy<>(0, Next),
        KOKKOS_LAMBDA(const int row) { pp(row) = zz(row); });

    /* s_old = r0·z0 */
    s_old = cg_dot(rr, zz, N);
    ALLREDUCE_SUM(s_old);

    int iter = 0;
    int verbose         = (getenv("FESOM_VERBOSE_CG") != NULL);
    int heartbeat_every = 100;
    static const bool cg_prof = (getenv("FESOM_CG_PROFILE") != NULL);
    double _cg_t0 = 0.0;
    if (cg_prof) { Kokkos::fence(); _cg_t0 = MPI_Wtime(); }   /* M5.2: opt-in CG-share timer (fences cost ~2-3%) */
    for (iter = 1; iter <= si->maxiter; ++iter) {
        if (!cgpipe)                                 /* E.CG1: pp ring1 is maintained by the
                                                      * recurrence — the exchange is DELETED */
            exch(si->pp_fld);                        /* solver.F90:442 EXCH(pp) */
        /* M5.2: fuse SpMV (App=A·pp) with the dot (s_aux=Σ pp·App) into ONE
         * parallel_reduce — App(row) is computed and pp(row)·App(row) accumulated
         * in row order, identical to the separate cg_spmv+cg_dot (Serial bit-id),
         * saving a kernel launch + a full device read of App per iteration. */
        real_t s_aux = 0.0;
        Kokkos::parallel_reduce("fesom_cg_spmv_dot", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l) {
                real_t s = 0.0;
                const int rstart = rowptr(row), rend = rowptr(row + 1);
                for (int n = rstart; n < rend; ++n) s += vals(n) * pp(colind(n));
                App(row) = s;
                l += pp(row) * s;
            }, s_aux);
        ALLREDUCE_SUM(s_aux);
        if (s_aux == 0.0 || s_aux != s_aux) {        /* NaN/zero check */
            if (partit == NULL || partit->mype == 0) {
                fprintf(stderr, "[fesom_ssh] CG_kk abort at iter %d: pp·App = %g (s_old=%g)\n",
                        iter, (double)s_aux, (double)s_old); fflush(stderr);
            }
            FESOM_DIE("CG_kk: pp·App is %g — matrix singular or NaN propagated", s_aux);
        }
        const real_t al = s_old / s_aux;

        Kokkos::parallel_for("fesom_cg_axpy", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                X (row) += al * pp (row);
                rr(row) -= al * App(row);
            });
        if (cgpipe) cgpipe_exchange_rr(si->rr_fld, partit);   /* E.CG1: the iteration's ONLY exchange */
        else        exch(si->rr_fld);                /* solver.F90:462 EXCH(rr) */

        /* M5.2: fuse the preconditioner SpMV (zz=M⁻¹·rr) with cg_dot2 (sp0=Σrr·zz,
         * sp1=Σrr·rr) into ONE parallel_reduce (row order → Serial bit-id), then ONE
         * 2-element MPI_Allreduce instead of two (same SUM per component, fewer
         * blocking collectives → the per-iter sync count the GPU is latency-bound on). */
        real_t sp0 = 0.0, sp1 = 0.0;
        Kokkos::parallel_reduce("fesom_cg_psolve_dot2", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l0, real_t &l1) {
                real_t s = 0.0;
                const int rstart = rowptr(row), rend = rowptr(row + 1);
                for (int n = rstart; n < rend; ++n) s += prvals(n) * rr(colind(n));
                zz(row) = s;
                l0 += rr(row) * s;
                l1 += rr(row) * rr(row);
            }, sp0, sp1);
        if (cgpipe) cgpipe_zz_ring1(rr, zz);         /* E.CG1: ring1 rows (dots stay owned-only) */
        if (parallel) {
            double sbuf[2] = { (double)sp0, (double)sp1 };
            MPI_Allreduce(MPI_IN_PLACE, sbuf, 2, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM);
            sp0 = sbuf[0]; sp1 = sbuf[1];
        }

        real_t residual = sqrt(sp1 / (real_t)N_global);
        if ((partit == NULL || partit->mype == 0)
            && (verbose ? (iter <= 5 || iter % 50 == 0)
                        : (iter % heartbeat_every == 0))) {
            fprintf(stderr, "[fesom_ssh] CG_kk iter %4d: res=%.4e rtol=%.4e\n",
                    iter, (double)residual, (double)rtol);
            fflush(stderr);
        }
        if (residual < rtol) break;
        if (residual != residual || residual > 1e30) {
            if (partit == NULL || partit->mype == 0) {
                fprintf(stderr, "[fesom_ssh] CG_kk abort at iter %d: residual=%g\n",
                        iter, (double)residual); fflush(stderr);
            }
            FESOM_DIE("CG_kk residual diverged");
        }

        const real_t be = sp0 / s_old;
        s_old = sp0;
        Kokkos::parallel_for("fesom_cg_pp", Kokkos::RangePolicy<>(0, Next),
            KOKKOS_LAMBDA(const int row) { pp(row) = zz(row) + be * pp(row); });
        if (cgpipe && cgpipe_selfcheck_on())         /* bring-up: recurred ring1 MUST equal exchanged */
            cgpipe_selfcheck_pp(si, partit, iter);
    }
    if (cg_prof) {
        Kokkos::fence();
        g_fesom_cg_wall  += MPI_Wtime() - _cg_t0;
        g_fesom_cg_iters += (iter <= si->maxiter) ? iter : si->maxiter;
    }
    if (iter > si->maxiter) {
        if (partit == NULL || partit->mype == 0) {
            fprintf(stderr, "[fesom_ssh] CG_kk hit maxiter=%d without converging "
                    "(last residual ~%.4e, rtol=%.4e)\n",
                    si->maxiter, (double)sqrt(s_old/(real_t)N_global), (double)rtol);
            fflush(stderr);
        }
        FESOM_DIE("CG_kk did not converge");
    }
    /* The C twin's exit EXCH(X) (solver.F90:507) is dropped: the driver does
     * exchange_nod2D(d_eta) immediately after this returns (the same unchanged X), so the
     * two are idempotent → bit-identical. X owned is device-current here; modify_device()
     * lets the driver sync_host() it before that halo. */
    dyn->d_eta_fld.modify_device();
    si->last_iters = iter;
    #undef ALLREDUCE_SUM
    return iter;
}

/* FESOM_KK_VERIFY=ssh — capture-before (L26) over the seven §5 read-modify-write fields.
 * The driver snapshots the pre-block host values; here we snapshot the KK result, restore
 * the pre-values, run the C twins in order (compute_ssh_rhs → CG → update_vel → compute_hbar
 * → eta_n), diff, then restore the KK production state. On Serial the device kernels run on
 * the SAME memory as the C twins (host==device) with identical FP ops/order → max|Δ|==0;
 * non-intrusive. (The CG scratch rr/zz/pp/App is overwritten by the C CG, then re-derived by
 * the next step's device CG before being read — no production state depends on it here.) */
void fesom_ssh_block_verify(const fesom_ssh_stiff   *S,
                            fesom_solverinfo        *si,
                            const struct fesom_mesh *mesh,
                            struct fesom_dyn        *dyn,
                            int step_n,
                            const std::vector<real_t> &pre_ssh_rhs,
                            const std::vector<real_t> &pre_d_eta,
                            const std::vector<real_t> &pre_uv,
                            const std::vector<real_t> &pre_ssh_rhs_old,
                            const std::vector<real_t> &pre_hbar,
                            const std::vector<real_t> &pre_hbar_old,
                            const std::vector<real_t> &pre_eta_n)
{
    const int    nl      = mesh->nl;
    const size_t Nn      = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D);
    const size_t Nuv     = (size_t)(mesh->myDim_elem2D + mesh->eDim_elem2D
                                   + mesh->eXDim_elem2D) * (size_t)nl * 2;
    /* snapshot the KK results (host mirrors hold them — the driver sync_host'd all seven). */
    std::vector<real_t> kk_ssh_rhs    (dyn->ssh_rhs,     dyn->ssh_rhs     + Nn);
    std::vector<real_t> kk_d_eta      (dyn->d_eta,       dyn->d_eta       + Nn);
    std::vector<real_t> kk_uv         (dyn->uv,          dyn->uv          + Nuv);
    std::vector<real_t> kk_ssh_rhs_old(dyn->ssh_rhs_old, dyn->ssh_rhs_old + Nn);
    std::vector<real_t> kk_hbar       (mesh->hbar,       mesh->hbar       + Nn);
    std::vector<real_t> kk_hbar_old   (mesh->hbar_old,   mesh->hbar_old   + Nn);
    std::vector<real_t> kk_eta_n      (dyn->eta_n,       dyn->eta_n       + Nn);

    /* restore the pre-block inputs the C twins read */
    std::copy(pre_ssh_rhs.begin(),     pre_ssh_rhs.end(),     dyn->ssh_rhs);
    std::copy(pre_d_eta.begin(),       pre_d_eta.end(),       dyn->d_eta);
    std::copy(pre_uv.begin(),          pre_uv.end(),          dyn->uv);
    std::copy(pre_ssh_rhs_old.begin(), pre_ssh_rhs_old.end(), dyn->ssh_rhs_old);
    std::copy(pre_hbar.begin(),        pre_hbar.end(),        mesh->hbar);
    std::copy(pre_hbar_old.begin(),    pre_hbar_old.end(),    mesh->hbar_old);
    std::copy(pre_eta_n.begin(),       pre_eta_n.end(),       dyn->eta_n);

    /* run the C twins (substeps 7-11). On Serial npes==1 → the internal halos are no-ops.
     * (compute_hbar writes mesh->hbar/hbar_old through the non-const pointer members; the
     * const on `mesh` applies to the pointer storage, not the pointee, so no cast needed.) */
    fesom_compute_ssh_rhs_linfs(mesh, dyn);
    fesom_ssh_solve_cg(S, si, mesh, dyn);
    fesom_update_vel(mesh, dyn);
    fesom_compute_hbar(mesh, dyn);
    {   /* substep 11 eta_n inline (mirror fesom_step.cpp). */
        const real_t alpha = (real_t)FESOM_PHASE1_ALPHA;
        for (int n = 0; n < (int)Nn; ++n)
            if (mesh->ulevels_nod2D[n] == 1)
                dyn->eta_n[n] = alpha * mesh->hbar[n] + (1.0 - alpha) * mesh->hbar_old[n];
    }

    auto mx = [](const std::vector<real_t> &a, const real_t *b) {
        double d = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double x = std::fabs((double)a[i] - (double)b[i]);
            if (x > d) d = x;
        }
        return d;
    };
    double d_rhs = mx(kk_ssh_rhs,     dyn->ssh_rhs);
    double d_de  = mx(kk_d_eta,       dyn->d_eta);
    double d_uv  = mx(kk_uv,          dyn->uv);
    double d_ro  = mx(kk_ssh_rhs_old, dyn->ssh_rhs_old);
    double d_hb  = mx(kk_hbar,        mesh->hbar);
    double d_hbo = mx(kk_hbar_old,    mesh->hbar_old);
    double d_en  = mx(kk_eta_n,       dyn->eta_n);

    /* restore the KK production state */
    std::copy(kk_ssh_rhs.begin(),     kk_ssh_rhs.end(),     dyn->ssh_rhs);
    std::copy(kk_d_eta.begin(),       kk_d_eta.end(),       dyn->d_eta);
    std::copy(kk_uv.begin(),          kk_uv.end(),          dyn->uv);
    std::copy(kk_ssh_rhs_old.begin(), kk_ssh_rhs_old.end(), dyn->ssh_rhs_old);
    std::copy(kk_hbar.begin(),        kk_hbar.end(),        mesh->hbar);
    std::copy(kk_hbar_old.begin(),    kk_hbar_old.end(),    mesh->hbar_old);
    std::copy(kk_eta_n.begin(),       kk_eta_n.end(),       dyn->eta_n);

    double dmax = d_rhs;
    for (double v : {d_de, d_uv, d_ro, d_hb, d_hbo, d_en}) if (v > dmax) dmax = v;

    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    std::printf("[FESOM_KK_VERIFY=ssh] step %d backend=%s  max|Δ|: ssh_rhs=%.3e d_eta=%.3e "
                "uv=%.3e ssh_rhs_old=%.3e hbar=%.3e hbar_old=%.3e eta_n=%.3e\n",
                step_n, backend.c_str(), d_rhs, d_de, d_uv, d_ro, d_hb, d_hbo, d_en);
    std::fflush(stdout);
    if (backend == "Serial" && dmax != 0.0) {
        std::fprintf(stderr, "[FESOM_KK_VERIFY=ssh] FAIL step %d: Serial must be bit-identical "
                             "to the C twin (max|Δ|=%.3e)\n", step_n, dmax);
        std::abort();
    }
}
