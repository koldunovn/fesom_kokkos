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
#include "fesom_ssh_dump.h"        // M10 T3: solver-lab dump format
#include <sys/stat.h>              // M10 T3: mkdir for FESOM_SSH_DUMP
#include <errno.h>

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
    Kokkos::View<double*, fesom_halo_pinned_space> sbuf_h, rbuf_h;  /* FESOM_HALO_STAGE mirrors */
    std::vector<MPI_Request> reqs;
    /* ring1 preconditioner CSR: row r (= local slot N+r), cols are LOCAL slots
     * into the extended rr; entries in the OWNER's row order (byte-identity). */
    Kokkos::View<int*>    rp2_d;         /* [eDim+1] */
    Kokkos::View<int*>    ci2_d;
    Kokkos::View<double*> pv2_d;
};
CgPipeState g_cgpipe;

/* M10: when non-NULL, cgpipe_build ships THESE preconditioner values for the ring1 rows
 * instead of S->pr_values (the symmetrised M̃⁻¹ — derivations §0.5). NULL for every M7 path,
 * so the certified CGPIPE behaviour is untouched. Set before cgpipe_build, never after
 * (the rows ship once). */
const real_t *cgpipe_ship_pr = NULL;

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
                /* VERBATIM, row order. M10: when a variant runs on the SYMMETRISED
                 * preconditioner, the ring1 rows must be the symmetrised ones too — mixing
                 * the two would make the recurred ring1 `u` differ from the owner's, which
                 * is exactly the byte-identity property this shipping exists to provide. */
                b.dbls.push_back((double)(cgpipe_ship_pr ? cgpipe_ship_pr[n] : S->pr_values[n]));
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
    if (fesom_halo_stage_on()) {   /* M7.5: pinned mirrors for the staged MPI leg */
        s.sbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>("cgpipe.sbuf_h", (size_t)s.nsend);
        s.rbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>("cgpipe.rbuf_h", (size_t)s.nrecv);
    }

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
    const bool staged = fesom_halo_stage_on();   /* M7.5: MPI on pinned mirrors (no GPUDirect) */
    if (staged && s.nsend > 0) Kokkos::deep_copy(s.sbuf_h, s.sbuf_d);
    double *sp = staged ? s.sbuf_h.data() : s.sbuf_d.data();
    double *rp = staged ? s.rbuf_h.data() : s.rbuf_d.data();
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
    if (staged && s.nrecv > 0) Kokkos::deep_copy(s.rbuf_d, s.rbuf_h);   /* M7.5 */

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

/* ======================================================================= *
 * M7 E.CG2 — FESOM_SPEED_CGPOLY=<d>: degree-d Chebyshev polynomial
 * preconditioner on an R=(d+1)-ring single-exchange PCG (opt-in value knob;
 * pre-registration: docs/plans/20260721-m7-session13-FINDINGS.md §3).
 *
 * WHY: post-CGPIPE the CG still pays ~72 iterations/step (1 fused exchange +
 * 2 Allreduces each = 74+146 sync events, 18/24 ms halo + Allreduce pool at
 * 4N/16N). CGPOLY attacks the ITERATION COUNT itself: the MITgcm M⁻¹ is
 * replaced by M⁻¹ = p_d(D̃⁻¹Ã)·D̃⁻¹ — d Chebyshev semi-iterations (Saad
 * Alg. 12.1), NO dot products — and the JAX port measured iters 127→55→42
 * at d=2/3 (2.31×/3.02×) with this exact recurrence at equal
 * unpreconditioned tolerance (port_jax 00f6e3c, its Task-E.3 clone).
 *
 * FROZEN OPERATOR (the ship-once property): Ã = the stiffness SNAPSHOT taken
 * at build (device-authoritative deep_copy), D̃ = diag(Ã). Under linfs Ã ≡ A
 * forever; under zstar A drifts from Ã exactly as it already drifts from the
 * frozen pr_values — the SAME Fortran precedent (oce_ale.F90:3306 lfirst),
 * so NOTHING is ever refreshed: ring rows/diagonals ship ONCE. The owned-row
 * cheb SpMVs also read Ã (a mixed live/frozen M would be non-symmetric).
 * The actual solve operator A·pp stays live at owned rows, untouched.
 *
 * RING THEOREM (the CGPIPE composition): producing zz on owned+ring1 (which
 * keeps the pp recurrence alive and exch(pp) deleted) with d inner SpMVs
 * needs rr on R = d+1 rings, Ã rows on rings 1..d, and D̃⁻¹ on all d+1
 * rings; each semi-iteration's row extent shrinks by one ring (the EVPWIDE
 * frontier argument applied to CG). d=1 degenerates to exactly the CGPIPE
 * 2-ring graph. Discovery generalizes cgpipe round-by-round: every shipped
 * CSR row's colind (gid, owner) pairs reveal the NEXT ring — BFS by
 * row-shipping, no scatter_mesh hook needed. All ring data is shipped owner
 * bytes VERBATIM in the owner's row order (rule 0.28: recompute nothing).
 *
 * BYTE CLASS: knob OFF = byte-identical (this code never runs). Knob ON is
 * NOT byte-identical to the baseline — a different Krylov trajectory
 * converging under the SAME unpreconditioned tolerance (the JAX class,
 * iterates agree ~1e-5 rel). The FORCE_SERIAL structural gate is the
 * SELFCHECK: the ring-extent apply must equal a reference apply that
 * re-exchanges between semi-iterations, BITWISE (same kernel instantiation
 * both paths + shipped owner bytes ⇒ max|Δ| MUST print 0.000e+00).
 *
 * Rule 0.27 byte-growth: the R-ring fused message grows ~×2 over CGPIPE's
 * 2-ring while the event count drops ~×3 (the opposite exposure of the
 * EVPWIDE wrong-high). The build prints worst-partner bytes; the
 * pre-registered watch: if d2/d3 regress while d1 pays, suspect eager→rndv
 * and run the UCX_RNDV_THRESH env-leg rescue.
 * ======================================================================= */

using RDV  = Kokkos::View<double*, Kokkos::LayoutRight>;   /* == Field::dev_view_t */
using RIV  = Kokkos::View<int*,    Kokkos::LayoutRight>;   /* == IntField::dev_view_t */

struct CgPolyState {
    bool built = false;
    int  d = 0, R = 0;                   /* degree, rings exchanged (= d+1) */
    int  N = 0;
    std::vector<int> nring;              /* [R]: nring[0]=ring1(=eDim), .. nring[R-1] */
    int  next_total = 0;                 /* N + Σ nring — the exchanged extent */
    std::vector<int> ext;                /* [d+1]: ext[0]=next_total, ext[j]=semi-j row extent, ext[d]=N+ring1 */
    /* fused R-ring exchange (tag 2110; pattern = cgpipe_exchange_rr) */
    int  nsend = 0, nrecv = 0;
    std::vector<int> partner, soff, roff;
    RIV sidx_d, ridx_d;
    RDV sbuf_d, rbuf_d;
    Kokkos::View<double*, fesom_halo_pinned_space> sbuf_h, rbuf_h;  /* FESOM_HALO_STAGE mirrors */
    std::vector<MPI_Request> reqs;
    /* frozen Ã: owned rows = av0 over S's CSR; ring rows 1..d = shipped CSR
     * (ci = LOCAL slots, entries in the OWNER's row order — byte-identity). */
    RDV av0_d;                           /* [nnz] frozen owned values */
    int nring_rows = 0;                  /* ring rows = Σ nring[0..d-1] */
    RIV rp_d, ci_d;
    RDV av_d;
    RDV invd_d, isq_d;                   /* [next_total] 1/diag(Ã), 1/sqrt(diag) */
    double lam_min = 0.0, lam_max = 0.0, kappa = 30.0;
    RDV f_d, za_d, zb_d, dd_d;           /* cheb scratch [next_total] */
    RDV rf_d, rza_d, rzb_d, rdd_d;       /* selfcheck reference scratch (lazy) */
    long solves = 0;
};
CgPolyState g_cgpoly;

bool cgpoly_selfcheck_on()
{
    static int c = -1;
    if (c < 0) { const char *e = getenv("FESOM_CGPOLY_SELFCHECK"); c = (e && e[0] == '1') ? 1 : 0; }
    return c != 0;
}

/* The per-iteration fused R-ring exchange on a raw device view (rr in the
 * solve; the power-iteration vector and the selfcheck reference reuse it).
 * Fence discipline = cgpipe_exchange_rr (see its banner). */
void cgpoly_exchange(RDV v, fesom_partit *p)
{
    CgPolyState &s = g_cgpoly;
    if (!p || p->npes <= 1) return;
    fesom_halo_prof_barrier(p);
    {
        auto sidx = s.sidx_d; auto sbuf = s.sbuf_d;
        if (s.nsend > 0)
            Kokkos::parallel_for("fesom_cgpoly_pack", Kokkos::RangePolicy<>(0, s.nsend),
                KOKKOS_LAMBDA(const int i) { sbuf(i) = v(sidx(i)); });
    }
    Kokkos::fence();   /* MANDATORY pre-MPI: MPI reads sbuf_d + re-posts rbuf_d */

    int nreq = 0;
    const bool staged = fesom_halo_stage_on();   /* M7.5: MPI on pinned mirrors (no GPUDirect) */
    if (staged && s.nsend > 0) Kokkos::deep_copy(s.sbuf_h, s.sbuf_d);
    double *sp = staged ? s.sbuf_h.data() : s.sbuf_d.data();
    double *rp = staged ? s.rbuf_h.data() : s.rbuf_d.data();
    for (size_t q = 0; q < s.partner.size(); ++q) {
        const int rc = s.roff[q + 1] - s.roff[q];
        if (rc > 0)
            MPI_Irecv(rp + s.roff[q], rc, MPI_DOUBLE, s.partner[q], 2110,
                      p->MPI_COMM_FESOM, &s.reqs[(size_t)nreq++]);
    }
    for (size_t q = 0; q < s.partner.size(); ++q) {
        const int sc = s.soff[q + 1] - s.soff[q];
        if (sc > 0)
            MPI_Isend(sp + s.soff[q], sc, MPI_DOUBLE, s.partner[q], 2110,
                      p->MPI_COMM_FESOM, &s.reqs[(size_t)nreq++]);
    }
    fesom_halo_prof_bytes(8.0 * (double)(s.nsend + s.nrecv));
    fesom_halo_prof_waitall(nreq, s.reqs.data());
    if (staged && s.nrecv > 0) Kokkos::deep_copy(s.rbuf_d, s.rbuf_h);   /* M7.5 */

    {
        auto ridx = s.ridx_d; auto rbuf = s.rbuf_d;
        if (s.nrecv > 0)
            Kokkos::parallel_for("fesom_cgpoly_unpack", Kokkos::RangePolicy<>(0, s.nrecv),
                KOKKOS_LAMBDA(const int i) { v(ridx(i)) = rbuf(i); });
    }
    /* no post-unpack fence — same NOFENCE2-audit parity as cgpipe. */
}

/* f = D̃⁻¹·rr ; z₀ = dd₀ = f/θ over [0, ext). ONE helper for the main apply
 * and the selfcheck reference — same instantiation ⇒ same codegen ⇒ the
 * bitwise-equality claim is about DATA (shipped owner bytes), not compiler
 * whims. Division (not mul-by-inverse) for Saad/JAX faithfulness. */
static void cgpoly_f0(RDV invd, RDV rr, RDV f, RDV z, RDV dd, double theta, int ext)
{
    const real_t th = (real_t)theta;
    Kokkos::parallel_for("fesom_cgpoly_f0", Kokkos::RangePolicy<>(0, ext),
        KOKKOS_LAMBDA(const int i) {
            const real_t fi = invd(i) * rr(i);
            f(i) = fi;
            const real_t z0 = fi / th;
            z(i) = z0; dd(i) = z0;
        });
}

/* One Chebyshev semi-iteration over [0, ext): per row, s = (Ã·zin)(row) —
 * owned rows read S's CSR + frozen av0, ring rows read the shipped CSR —
 * then res = f − D̃⁻¹s ; dd = c1·dd + c2·res ; zout = zin + dd. zin/zout are
 * double-buffered (the SpMV gathers zin cross-row while zout is written).
 * The last semi-iteration writes zz directly (rows < N+ring1 by extent).
 * dd is read/written only at its own row → in-place safe. Race-free gather,
 * sequential inner sum → Serial bit-identical (the cg_spmv argument). */
static void cgpoly_semi(IDV rowptr, IDV colind, RDV av0,
                        RIV rp, RIV ci, RDV av,
                        RDV invd, RDV f, RDV zin, RDV zout, RDV dd,
                        DV zz, bool write_zz,
                        double c1_, double c2_, int N, int ext)
{
    const real_t c1 = (real_t)c1_, c2 = (real_t)c2_;
    Kokkos::parallel_for("fesom_cgpoly_semi", Kokkos::RangePolicy<>(0, ext),
        KOKKOS_LAMBDA(const int i) {
            real_t sacc = 0.0;
            if (i < N) {
                const int a = rowptr(i), e = rowptr(i + 1);
                for (int n = a; n < e; ++n) sacc += av0(n) * zin(colind(n));
            } else {
                const int q = i - N;
                const int a = rp(q), e = rp(q + 1);
                for (int n = a; n < e; ++n) sacc += av(n) * zin(ci(n));
            }
            const real_t res = f(i) - invd(i) * sacc;
            const real_t dn  = c1 * dd(i) + c2 * res;
            const real_t zn  = zin(i) + dn;
            dd(i) = dn;
            if (write_zz) zz(i) = zn; else zout(i) = zn;
        });
}

/* zz = M⁻¹·rr — the degree-d Chebyshev apply. rr must be R-ring-current
 * (cgpoly_exchange). Output CONTRACT: zz valid on owned+ring1 (same as the
 * cgpipe psolve+zz_ring1 pair it replaces); intermediates live on shrinking
 * ring extents. Host recurrence scalars only — NO dot products, NO
 * collectives (the point of the lever). */
void cgpoly_apply(const fesom_ssh_stiff *S, DV rr, DV zz)
{
    CgPolyState &s = g_cgpoly;
    const double theta  = 0.5 * (s.lam_max + s.lam_min);
    const double delta  = 0.5 * (s.lam_max - s.lam_min);
    const double sigma1 = theta / delta;
    double rho = 1.0 / sigma1;

    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    cgpoly_f0(s.invd_d, rr, s.f_d, s.za_d, s.dd_d, theta, s.ext[0]);
    RDV zin = s.za_d, zout = s.zb_d;
    for (int j = 1; j <= s.d; ++j) {
        const double rho_new = 1.0 / (2.0 * sigma1 - rho);
        const bool   last    = (j == s.d);
        cgpoly_semi(rowptr, colind, s.av0_d, s.rp_d, s.ci_d, s.av_d,
                    s.invd_d, s.f_d, zin, zout, s.dd_d, zz, last,
                    rho_new * rho, 2.0 * rho_new / delta, s.N, s.ext[(size_t)j]);
        rho = rho_new;
        std::swap(zin, zout);
    }
}

/* FESOM_CGPOLY_SELFCHECK=1 — the lever's structural byte gate. Reference
 * path: SAME f0/semi kernels (same instantiation), but rows computed on
 * OWNED extent only and the R-ring exchange re-run before every
 * semi-iteration, so every ring value consumed is FRESH OWNER BYTES instead
 * of the locally replayed ones. Claim under test: replayed ring rows ≡
 * shipped owner computation, BITWISE, through all d semi-iterations
 * (operands = shipped bytes; expression = same kernel; order = owner's row
 * order) ⇒ max|zz − ref| over owned+ring1 MUST print 0.000e+00 on every
 * backend (the exchange and the gather are deterministic — no atomics). */
void cgpoly_selfcheck(const fesom_ssh_stiff *S, DV rr, DV zz,
                      fesom_partit *p, int iter)
{
    CgPolyState &s = g_cgpoly;
    if (s.rf_d.extent(0) == 0) {
        s.rf_d  = RDV("cgpoly.rf",  (size_t)s.next_total);
        s.rza_d = RDV("cgpoly.rza", (size_t)s.next_total);
        s.rzb_d = RDV("cgpoly.rzb", (size_t)s.next_total);
        s.rdd_d = RDV("cgpoly.rdd", (size_t)s.next_total);
    }
    const double theta  = 0.5 * (s.lam_max + s.lam_min);
    const double delta  = 0.5 * (s.lam_max - s.lam_min);
    const double sigma1 = theta / delta;
    double rho = 1.0 / sigma1;

    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    cgpoly_f0(s.invd_d, rr, s.rf_d, s.rza_d, s.rdd_d, theta, s.ext[0]);
    RDV zin = s.rza_d, zout = s.rzb_d;
    for (int j = 1; j <= s.d; ++j) {
        const double rho_new = 1.0 / (2.0 * sigma1 - rho);
        cgpoly_exchange(zin, p);              /* rings ← owner bytes, every step */
        cgpoly_semi(rowptr, colind, s.av0_d, s.rp_d, s.ci_d, s.av_d,
                    s.invd_d, s.rf_d, zin, zout, s.rdd_d, zz, false,
                    rho_new * rho, 2.0 * rho_new / delta, s.N, s.N /* OWNED only */);
        rho = rho_new;
        std::swap(zin, zout);
    }
    cgpoly_exchange(zin, p);                  /* ring1 of the final ref ← owner bytes */

    const int cmp = s.ext[(size_t)s.d];       /* owned + ring1 */
    double maxd = 0.0;
    {
        RDV ref = zin;
        Kokkos::parallel_reduce("fesom_cgpoly_selfcheck", Kokkos::RangePolicy<>(0, cmp),
            KOKKOS_LAMBDA(const int i, double &l) {
                const double dd_ = fabs((double)zz(i) - (double)ref(i));
                if (dd_ > l) l = dd_;
            }, Kokkos::Max<double>(maxd));
    }
    double gmax = maxd;
    if (p && p->npes > 1)
        MPI_Allreduce(MPI_IN_PLACE, &gmax, 1, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);
    if (!p || p->mype == 0)
        fprintf(stderr, "[cgpoly-selfcheck] iter %4d: max|apply - exchanged ref| = %.3e%s\n",
                iter, gmax, gmax == 0.0 ? "" : "  <-- MUST BE 0: RING REPLAY BROKEN");
}

/* ---- the R-ring build: generalized cgpipe rounds -----------------------
 * Round 1 (com-driven, the cgpipe part-A/B pattern): owners ship the FROZEN
 * Ã CSR rows of their slist blocks; colind travels as (gid, owner) pairs.
 * Rounds r = 2..d (want-driven): request the rows of ring-r nodes from
 * their owners; every translate of an unknown gid registers a ring-(r+1)
 * slot (BFS completeness: rings 1..r are fully known after round r-1, so an
 * unknown neighbour of a ring-r node is ring r+1 exactly).
 * Round R = d+1 (want-driven, diag only): ring-R nodes need only D̃.
 * Then flat per-partner lists [ring1 | ring2 | ... | ringR] both sides,
 * device pushes, the λmax power iteration, and the L80 announce. */
void cgpoly_build(const fesom_ssh_stiff *S, fesom_solverinfo *si,
                  const struct fesom_mesh *mesh, fesom_partit *p,
                  int degree, double kappa)
{
    CgPolyState &s = g_cgpoly;
    const int N    = mesh->myDim_nod2D;
    const int eDim = mesh->eDim_nod2D;
    const bool parallel = (p && p->npes > 1);
    const int npes = parallel ? p->npes : 1;
    MPI_Comm comm  = parallel ? p->MPI_COMM_FESOM : MPI_COMM_NULL;
    s.d = degree; s.R = degree + 1; s.N = N; s.kappa = kappa;

    /* Frozen Ã snapshot — DEVICE-authoritative (under zstar the host mirror
     * of `values` can be stale; the device is always current). av0_h is the
     * host copy the row shipping + invd build read. */
    const int nnz = S->nnz;
    s.av0_d = RDV("cgpoly.av0", (size_t)nnz);
    Kokkos::deep_copy(s.av0_d, S->values_fld.d());
    std::vector<double> av0_h((size_t)nnz);
    {
        auto m = Kokkos::create_mirror_view(s.av0_d);
        Kokkos::deep_copy(m, s.av0_d);
        for (int i = 0; i < nnz; ++i) av0_h[(size_t)i] = m(i);
    }

    /* slot registry: [0,N) owned | [N,N+eDim) ring1 (existing com order) |
     * appended rings 2..R (registration order ⇒ contiguous ring blocks). */
    std::unordered_map<int, int> g2l;
    g2l.reserve((size_t)(N + eDim) * 2);
    std::vector<int> slot_gid, slot_owner;                 /* slots >= N+eDim */
    std::vector<std::vector<int>> ring_slots((size_t)s.R + 1);   /* [2..R] used */
    std::vector<std::vector<int>> row_ci; std::vector<std::vector<double>> row_pv;
    row_ci.resize((size_t)eDim); row_pv.resize((size_t)eDim);

    const fesom_com_struct *cs = parallel ? &p->com_nod2D : NULL;
    std::vector<int> owner_l((size_t)(N + eDim), parallel ? p->mype : 0);
    if (parallel) {
        for (int l = 0; l < N + eDim; ++l) g2l.emplace(p->myList_nod2D[l], l);
        for (int k = 0; k < cs->rPEnum; ++k)
            for (int j = cs->rptr[k] - 1; j < cs->rptr[k + 1] - 1; ++j)
                owner_l[(size_t)(cs->rlist[j] - 1)] = cs->rPE[k];
    }

    auto reg = [&](int gid, int own, int ring) -> int {
        auto it = g2l.find(gid);
        if (it != g2l.end()) {
            if (it->second >= N + eDim)
                FESOM_CHECK(slot_owner[(size_t)(it->second - N - eDim)] == own,
                            "cgpoly: gid %d shipped with conflicting owners %d/%d",
                            gid, slot_owner[(size_t)(it->second - N - eDim)], own);
            return it->second;
        }
        FESOM_CHECK(own != p->mype,
                    "cgpoly: shipped gid %d claims MY ownership but is not local", gid);
        FESOM_CHECK(ring >= 2 && ring <= s.R,
                    "cgpoly: gid %d discovered outside the ring window (ring %d)", gid, ring);
        const int slot = N + eDim + (int)slot_gid.size();
        slot_gid.push_back(gid); slot_owner.push_back(own);
        ring_slots[(size_t)ring].push_back(slot);
        g2l.emplace(gid, slot);
        return slot;
    };

    /* per-round send lists (owned idx, wantin order) for the flat build. */
    std::vector<std::vector<std::vector<int>>> slist_r((size_t)s.R + 1,
        std::vector<std::vector<int>>((size_t)npes));
    /* per-round want gid lists per owner (my recv order per (partner,ring)). */
    std::vector<std::vector<std::vector<int>>> want_r((size_t)s.R + 1,
        std::vector<std::vector<int>>((size_t)npes));

    std::vector<MPI_Request> rq;

    if (parallel) {
        /* ---- Round 1: com-driven Ã-row ship (cgpipe part A/B, values payload). */
        struct Bundle { std::vector<int> ints; std::vector<double> dbls; int hdr[2]; };
        std::vector<Bundle> sb((size_t)cs->sPEnum), rb((size_t)cs->rPEnum);
        rq.reserve((size_t)(cs->sPEnum + cs->rPEnum) * 2);
        for (int k = 0; k < cs->rPEnum; ++k) {
            rq.push_back(MPI_Request());
            MPI_Irecv(rb[(size_t)k].hdr, 2, MPI_INT, cs->rPE[k], 2121, comm, &rq.back());
        }
        for (int k = 0; k < cs->sPEnum; ++k) {
            Bundle &b = sb[(size_t)k];
            const int j0 = cs->sptr[k] - 1, j1 = cs->sptr[k + 1] - 1;
            int nent = 0;
            b.ints.push_back(j1 - j0);
            for (int j = j0; j < j1; ++j) {
                const int row = cs->slist[j] - 1;
                b.ints.push_back(S->rowptr[row + 1] - S->rowptr[row]);
                nent += S->rowptr[row + 1] - S->rowptr[row];
            }
            for (int j = j0; j < j1; ++j) {
                const int row = cs->slist[j] - 1;
                for (int n = S->rowptr[row]; n < S->rowptr[row + 1]; ++n) {
                    const int col = S->colind[n];
                    b.ints.push_back(p->myList_nod2D[col]);
                    b.ints.push_back(owner_l[(size_t)col]);
                    b.dbls.push_back(av0_h[(size_t)n]);          /* FROZEN Ã, row order */
                }
            }
            b.hdr[0] = j1 - j0; b.hdr[1] = nent;
            rq.push_back(MPI_Request());
            MPI_Isend(b.hdr, 2, MPI_INT, cs->sPE[k], 2121, comm, &rq.back());
        }
        MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
        rq.clear();
        for (int k = 0; k < cs->rPEnum; ++k) {
            const int nrows_exp = (cs->rptr[k + 1] - 1) - (cs->rptr[k] - 1);
            FESOM_CHECK(rb[(size_t)k].hdr[0] == nrows_exp,
                        "cgpoly: provider %d shipped %d rows, expected %d",
                        cs->rPE[k], rb[(size_t)k].hdr[0], nrows_exp);
            rb[(size_t)k].ints.resize((size_t)1 + rb[(size_t)k].hdr[0] + 2 * (size_t)rb[(size_t)k].hdr[1]);
            rb[(size_t)k].dbls.resize((size_t)rb[(size_t)k].hdr[1]);
            rq.push_back(MPI_Request());
            MPI_Irecv(rb[(size_t)k].ints.data(), (int)rb[(size_t)k].ints.size(), MPI_INT,    cs->rPE[k], 2122, comm, &rq.back());
            rq.push_back(MPI_Request());
            MPI_Irecv(rb[(size_t)k].dbls.data(), (int)rb[(size_t)k].dbls.size(), MPI_DOUBLE, cs->rPE[k], 2123, comm, &rq.back());
        }
        for (int k = 0; k < cs->sPEnum; ++k) {
            rq.push_back(MPI_Request());
            MPI_Isend(sb[(size_t)k].ints.data(), (int)sb[(size_t)k].ints.size(), MPI_INT,    cs->sPE[k], 2122, comm, &rq.back());
            rq.push_back(MPI_Request());
            MPI_Isend(sb[(size_t)k].dbls.data(), (int)sb[(size_t)k].dbls.size(), MPI_DOUBLE, cs->sPE[k], 2123, comm, &rq.back());
        }
        MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
        rq.clear();
        sb.clear();
        for (int k = 0; k < cs->rPEnum; ++k) {
            const Bundle &b = rb[(size_t)k];
            const int nrows = b.hdr[0];
            size_t ip = (size_t)1 + nrows, dp = 0;
            for (int j = 0; j < nrows; ++j) {
                const int slot = cs->rlist[(cs->rptr[k] - 1) + j] - 1;
                FESOM_CHECK(slot >= N && slot < N + eDim, "cgpoly: rlist slot %d outside ring1", slot);
                const int r   = slot - N;
                const int len = b.ints[(size_t)1 + j];
                row_ci[(size_t)r].reserve((size_t)len); row_pv[(size_t)r].reserve((size_t)len);
                for (int q = 0; q < len; ++q) {
                    const int gid = b.ints[ip++];
                    const int own = b.ints[ip++];
                    row_ci[(size_t)r].push_back(reg(gid, own, 2));
                    row_pv[(size_t)r].push_back(b.dbls[dp++]);
                }
            }
            FESOM_CHECK(dp == b.dbls.size() && ip == b.ints.size(),
                        "cgpoly: round-1 bundle from %d not fully consumed", cs->rPE[k]);
        }
        rb.clear();

        /* ---- Rounds r = 2..d: want-driven Ã-row ship. */
        for (int r = 2; r <= s.d; ++r) {
            for (int slot : ring_slots[(size_t)r])
                want_r[(size_t)r][(size_t)slot_owner[(size_t)(slot - N - eDim)]]
                    .push_back(slot_gid[(size_t)(slot - N - eDim)]);
            std::vector<int> scnt((size_t)npes, 0), rcnt((size_t)npes, 0);
            for (int P = 0; P < npes; ++P) scnt[(size_t)P] = (int)want_r[(size_t)r][(size_t)P].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, comm);
            std::vector<std::vector<int>> wantin((size_t)npes);
            for (int Q = 0; Q < npes; ++Q) {
                if (rcnt[(size_t)Q] > 0) {
                    wantin[(size_t)Q].resize((size_t)rcnt[(size_t)Q]);
                    rq.push_back(MPI_Request());
                    MPI_Irecv(wantin[(size_t)Q].data(), rcnt[(size_t)Q], MPI_INT, Q, 2180 + r, comm, &rq.back());
                }
                if (scnt[(size_t)Q] > 0) {
                    rq.push_back(MPI_Request());
                    MPI_Isend(want_r[(size_t)r][(size_t)Q].data(), scnt[(size_t)Q], MPI_INT, Q, 2180 + r, comm, &rq.back());
                }
            }
            MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
            rq.clear();

            /* replies: [len per row][(gid,owner)…] + [vals…]; header carries nent. */
            std::vector<Bundle> rsb((size_t)npes), rrb((size_t)npes);
            for (int Q = 0; Q < npes; ++Q) {
                if (scnt[(size_t)Q] > 0) {                    /* I asked Q → header in */
                    rq.push_back(MPI_Request());
                    MPI_Irecv(rrb[(size_t)Q].hdr, 2, MPI_INT, Q, 2130 + r, comm, &rq.back());
                }
                if (rcnt[(size_t)Q] > 0) {                    /* Q asked me → build + header out */
                    Bundle &b = rsb[(size_t)Q];
                    int nent = 0;
                    slist_r[(size_t)r][(size_t)Q].reserve(wantin[(size_t)Q].size());
                    for (int gid : wantin[(size_t)Q]) {
                        auto it = g2l.find(gid);
                        FESOM_CHECK(it != g2l.end() && it->second < N,
                                    "cgpoly: rank %d wants gid %d that is not my owned node", Q, gid);
                        slist_r[(size_t)r][(size_t)Q].push_back(it->second);
                        b.ints.push_back(S->rowptr[it->second + 1] - S->rowptr[it->second]);
                        nent += S->rowptr[it->second + 1] - S->rowptr[it->second];
                    }
                    for (int l : slist_r[(size_t)r][(size_t)Q])
                        for (int n = S->rowptr[l]; n < S->rowptr[l + 1]; ++n) {
                            b.ints.push_back(p->myList_nod2D[S->colind[n]]);
                            b.ints.push_back(owner_l[(size_t)S->colind[n]]);
                            b.dbls.push_back(av0_h[(size_t)n]);
                        }
                    b.hdr[0] = (int)wantin[(size_t)Q].size(); b.hdr[1] = nent;
                    rq.push_back(MPI_Request());
                    MPI_Isend(b.hdr, 2, MPI_INT, Q, 2130 + r, comm, &rq.back());
                }
            }
            MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
            rq.clear();
            for (int Q = 0; Q < npes; ++Q) {
                if (scnt[(size_t)Q] > 0) {
                    Bundle &b = rrb[(size_t)Q];
                    FESOM_CHECK(b.hdr[0] == scnt[(size_t)Q],
                                "cgpoly: round-%d reply from %d has %d rows, expected %d",
                                r, Q, b.hdr[0], scnt[(size_t)Q]);
                    b.ints.resize((size_t)b.hdr[0] + 2 * (size_t)b.hdr[1]);
                    b.dbls.resize((size_t)b.hdr[1]);
                    rq.push_back(MPI_Request());
                    MPI_Irecv(b.ints.data(), (int)b.ints.size(), MPI_INT,    Q, 2140 + r, comm, &rq.back());
                    rq.push_back(MPI_Request());
                    MPI_Irecv(b.dbls.data(), (int)b.dbls.size(), MPI_DOUBLE, Q, 2150 + r, comm, &rq.back());
                }
                if (rcnt[(size_t)Q] > 0) {
                    rq.push_back(MPI_Request());
                    MPI_Isend(rsb[(size_t)Q].ints.data(), (int)rsb[(size_t)Q].ints.size(), MPI_INT,    Q, 2140 + r, comm, &rq.back());
                    rq.push_back(MPI_Request());
                    MPI_Isend(rsb[(size_t)Q].dbls.data(), (int)rsb[(size_t)Q].dbls.size(), MPI_DOUBLE, Q, 2150 + r, comm, &rq.back());
                }
            }
            MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
            rq.clear();

            row_ci.resize(row_ci.size() + ring_slots[(size_t)r].size());
            row_pv.resize(row_pv.size() + ring_slots[(size_t)r].size());
            for (int Q = 0; Q < npes; ++Q) {
                if (scnt[(size_t)Q] == 0) continue;
                const Bundle &b = rrb[(size_t)Q];
                size_t ip = (size_t)b.hdr[0], dp = 0, wi = 0;
                for (int gid : want_r[(size_t)r][(size_t)Q]) {
                    const int slot = g2l.find(gid)->second;
                    const int rr_  = slot - N;               /* ring-row index (rings are appended in order) */
                    const int len  = b.ints[wi++];
                    row_ci[(size_t)rr_].reserve((size_t)len); row_pv[(size_t)rr_].reserve((size_t)len);
                    for (int q = 0; q < len; ++q) {
                        const int cgid = b.ints[ip++];
                        const int cown = b.ints[ip++];
                        row_ci[(size_t)rr_].push_back(reg(cgid, cown, r + 1));
                        row_pv[(size_t)rr_].push_back(b.dbls[dp++]);
                    }
                }
                FESOM_CHECK(dp == b.dbls.size() && ip == b.ints.size(),
                            "cgpoly: round-%d bundle from %d not fully consumed", r, Q);
            }
        }

        /* ---- Round R = d+1: diag-only for the outermost ring. */
        {
            const int r = s.R;
            for (int slot : ring_slots[(size_t)r])
                want_r[(size_t)r][(size_t)slot_owner[(size_t)(slot - N - eDim)]]
                    .push_back(slot_gid[(size_t)(slot - N - eDim)]);
            std::vector<int> scnt((size_t)npes, 0), rcnt((size_t)npes, 0);
            for (int P = 0; P < npes; ++P) scnt[(size_t)P] = (int)want_r[(size_t)r][(size_t)P].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, comm);
            std::vector<std::vector<int>>    wantin((size_t)npes);
            std::vector<std::vector<double>> dout((size_t)npes), din((size_t)npes);
            for (int Q = 0; Q < npes; ++Q) {
                if (rcnt[(size_t)Q] > 0) {
                    wantin[(size_t)Q].resize((size_t)rcnt[(size_t)Q]);
                    rq.push_back(MPI_Request());
                    MPI_Irecv(wantin[(size_t)Q].data(), rcnt[(size_t)Q], MPI_INT, Q, 2180 + r, comm, &rq.back());
                }
                if (scnt[(size_t)Q] > 0) {
                    rq.push_back(MPI_Request());
                    MPI_Isend(want_r[(size_t)r][(size_t)Q].data(), scnt[(size_t)Q], MPI_INT, Q, 2180 + r, comm, &rq.back());
                }
            }
            MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
            rq.clear();
            for (int Q = 0; Q < npes; ++Q) {
                if (scnt[(size_t)Q] > 0) {
                    din[(size_t)Q].resize((size_t)scnt[(size_t)Q]);
                    rq.push_back(MPI_Request());
                    MPI_Irecv(din[(size_t)Q].data(), scnt[(size_t)Q], MPI_DOUBLE, Q, 2160, comm, &rq.back());
                }
                if (rcnt[(size_t)Q] > 0) {
                    dout[(size_t)Q].reserve(wantin[(size_t)Q].size());
                    slist_r[(size_t)r][(size_t)Q].reserve(wantin[(size_t)Q].size());
                    for (int gid : wantin[(size_t)Q]) {
                        auto it = g2l.find(gid);
                        FESOM_CHECK(it != g2l.end() && it->second < N,
                                    "cgpoly: rank %d wants gid %d (diag) that is not my owned node", Q, gid);
                        slist_r[(size_t)r][(size_t)Q].push_back(it->second);
                        dout[(size_t)Q].push_back(av0_h[(size_t)S->rowptr[it->second]]);   /* diag = row's first entry */
                    }
                    rq.push_back(MPI_Request());
                    MPI_Isend(dout[(size_t)Q].data(), rcnt[(size_t)Q], MPI_DOUBLE, Q, 2160, comm, &rq.back());
                }
            }
            MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
            rq.clear();
            /* ---- ring sizes + extents (din consumed for invd just below). */
            s.nring.assign((size_t)s.R, 0);
            s.nring[0] = eDim;
            for (int r2 = 2; r2 <= s.R; ++r2) s.nring[(size_t)(r2 - 1)] = (int)ring_slots[(size_t)r2].size();
            s.next_total = N; for (int q = 0; q < s.R; ++q) s.next_total += s.nring[(size_t)q];
            s.ext.assign((size_t)s.d + 1, N);
            s.ext[0] = s.next_total;
            for (int j = 1; j <= s.d; ++j) {
                int e = N;
                for (int q = 0; q < s.R - j; ++q) e += s.nring[(size_t)q];
                s.ext[(size_t)j] = e;
            }
            s.nring_rows = 0; for (int q = 0; q < s.d; ++q) s.nring_rows += s.nring[(size_t)q];

            /* ---- invd/isq over [0, next_total): owned + ring rows (their shipped
             * diag = first entry, the CSR builder puts the diagonal first) + ring-R
             * shipped diagonals. Division of identical bytes is deterministic, so
             * local 1/diag == the owner's 1/diag bitwise. */
            std::vector<double> invd((size_t)s.next_total, 1.0);
            for (int row = 0; row < N; ++row) {
                FESOM_CHECK(av0_h[(size_t)S->rowptr[row]] > 0.0,
                            "cgpoly: non-positive diagonal at owned row %d", row);
                invd[(size_t)row] = 1.0 / av0_h[(size_t)S->rowptr[row]];
            }
            for (int rr_ = 0; rr_ < s.nring_rows; ++rr_) {
                FESOM_CHECK(!row_pv[(size_t)rr_].empty() && row_pv[(size_t)rr_][0] > 0.0,
                            "cgpoly: ring row %d has no positive diagonal", rr_);
                invd[(size_t)(N + rr_)] = 1.0 / row_pv[(size_t)rr_][0];
            }
            {   /* ring-R diagonals arrive per owner in want order = slot order. */
                std::vector<size_t> pos((size_t)npes, 0);
                for (int slot : ring_slots[(size_t)s.R]) {
                    const int own = slot_owner[(size_t)(slot - N - eDim)];
                    const double dg = din[(size_t)own][pos[(size_t)own]++];
                    FESOM_CHECK(dg > 0.0, "cgpoly: non-positive shipped diag for slot %d", slot);
                    invd[(size_t)slot] = 1.0 / dg;
                }
            }
            std::vector<double> isq((size_t)s.next_total);
            for (int i = 0; i < s.next_total; ++i) isq[(size_t)i] = sqrt(invd[(size_t)i]);
            auto push_d = [](const char *lbl, const std::vector<double> &v) {
                RDV dv(std::string(lbl), v.size());
                auto h = Kokkos::create_mirror_view(dv);
                for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
                Kokkos::deep_copy(dv, h);
                return dv;
            };
            s.invd_d = push_d("cgpoly.invd", invd);
            s.isq_d  = push_d("cgpoly.isq",  isq);
        }
    } else {
        /* npes == 1: no rings, no exchange — the cheb math alone. */
        s.nring.assign((size_t)s.R, 0);
        s.next_total = N;
        s.ext.assign((size_t)s.d + 1, N);
        s.nring_rows = 0;
        std::vector<double> invd((size_t)N), isq((size_t)N);
        for (int row = 0; row < N; ++row) {
            FESOM_CHECK(av0_h[(size_t)S->rowptr[row]] > 0.0,
                        "cgpoly: non-positive diagonal at owned row %d", row);
            invd[(size_t)row] = 1.0 / av0_h[(size_t)S->rowptr[row]];
            isq[(size_t)row]  = sqrt(invd[(size_t)row]);
        }
        RDV dv("cgpoly.invd", (size_t)N), sv("cgpoly.isq", (size_t)N);
        auto h1 = Kokkos::create_mirror_view(dv), h2 = Kokkos::create_mirror_view(sv);
        for (int i = 0; i < N; ++i) { h1(i) = invd[(size_t)i]; h2(i) = isq[(size_t)i]; }
        Kokkos::deep_copy(dv, h1); Kokkos::deep_copy(sv, h2);
        s.invd_d = dv; s.isq_d = sv;
    }

    /* ---- ring CSR device push (rows for rings 1..d, LOCAL ci slots). */
    {
        std::vector<int> rp((size_t)s.nring_rows + 1, 0), ci;
        std::vector<double> av;
        for (int r = 0; r < s.nring_rows; ++r) {
            rp[(size_t)r + 1] = rp[(size_t)r] + (int)row_ci[(size_t)r].size();
            ci.insert(ci.end(), row_ci[(size_t)r].begin(), row_ci[(size_t)r].end());
            av.insert(av.end(), row_pv[(size_t)r].begin(), row_pv[(size_t)r].end());
        }
        auto push_i = [](const char *lbl, const std::vector<int> &v) {
            RIV d(std::string(lbl), v.size());
            auto h = Kokkos::create_mirror_view(d);
            for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
            Kokkos::deep_copy(d, h);
            return d;
        };
        s.rp_d = push_i("cgpoly.rp", rp);
        s.ci_d = push_i("cgpoly.ci", ci);
        RDV avd("cgpoly.av", av.size());
        auto h = Kokkos::create_mirror_view(avd);
        for (size_t i = 0; i < av.size(); ++i) h(i) = av[i];
        Kokkos::deep_copy(avd, h);
        s.av_d = avd;
    }

    /* ---- flat per-partner lists: [ring1 | ring2 | … | ringR] both sides. */
    long worst_partner_bytes = 0;
    if (parallel) {
        std::vector<int> s1k((size_t)npes, -1), r1k((size_t)npes, -1);
        for (int k = 0; k < cs->sPEnum; ++k) s1k[(size_t)cs->sPE[k]] = k;
        for (int k = 0; k < cs->rPEnum; ++k) r1k[(size_t)cs->rPE[k]] = k;
        s.partner.clear(); s.soff.assign(1, 0); s.roff.assign(1, 0);
        std::vector<int> sidx, ridx;
        /* per-ring recv slots grouped by owner, want order (== slot order). */
        std::vector<std::vector<std::vector<int>>> rslot_r((size_t)s.R + 1,
            std::vector<std::vector<int>>((size_t)npes));
        for (int r = 2; r <= s.R; ++r)
            for (int slot : ring_slots[(size_t)r])
                rslot_r[(size_t)r][(size_t)slot_owner[(size_t)(slot - N - eDim)]].push_back(slot);
        for (int P = 0; P < npes; ++P) {
            bool has = s1k[(size_t)P] >= 0 || r1k[(size_t)P] >= 0;
            for (int r = 2; r <= s.R && !has; ++r)
                has = !slist_r[(size_t)r][(size_t)P].empty() || !rslot_r[(size_t)r][(size_t)P].empty();
            if (!has) continue;
            s.partner.push_back(P);
            if (s1k[(size_t)P] >= 0) {
                const int k = s1k[(size_t)P];
                for (int j = cs->sptr[k] - 1; j < cs->sptr[k + 1] - 1; ++j)
                    sidx.push_back(cs->slist[j] - 1);
            }
            for (int r = 2; r <= s.R; ++r)
                for (int l : slist_r[(size_t)r][(size_t)P]) sidx.push_back(l);
            if (r1k[(size_t)P] >= 0) {
                const int k = r1k[(size_t)P];
                for (int j = cs->rptr[k] - 1; j < cs->rptr[k + 1] - 1; ++j)
                    ridx.push_back(cs->rlist[j] - 1);
            }
            for (int r = 2; r <= s.R; ++r)
                for (int slot : rslot_r[(size_t)r][(size_t)P]) ridx.push_back(slot);
            s.soff.push_back((int)sidx.size());
            s.roff.push_back((int)ridx.size());
            const long pb = 8L * ((long)(s.soff.back() - s.soff[s.soff.size() - 2])
                                + (long)(s.roff.back() - s.roff[s.roff.size() - 2]));
            if (pb > worst_partner_bytes) worst_partner_bytes = pb;
        }
        s.nsend = (int)sidx.size();
        s.nrecv = (int)ridx.size();
        s.reqs.assign(2 * s.partner.size(), MPI_Request());
        auto push_i = [](const char *lbl, const std::vector<int> &v) {
            RIV d(std::string(lbl), v.size());
            auto h = Kokkos::create_mirror_view(d);
            for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
            Kokkos::deep_copy(d, h);
            return d;
        };
        s.sidx_d = push_i("cgpoly.sidx", sidx);
        s.ridx_d = push_i("cgpoly.ridx", ridx);
        s.sbuf_d = RDV("cgpoly.sbuf", (size_t)s.nsend);
        s.rbuf_d = RDV("cgpoly.rbuf", (size_t)s.nrecv);
        if (fesom_halo_stage_on()) {   /* M7.5: pinned mirrors for the staged MPI leg */
            s.sbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>("cgpoly.sbuf_h", (size_t)s.nsend);
            s.rbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>("cgpoly.rbuf_h", (size_t)s.nrecv);
        }
    }

    /* rr gains the ring tail; every solve writes owned rr before the first
     * exchange and rings are exchange-filled before any read (cgpipe rule). */
    si->rr_fld.alloc("ssh.cg.rr", (size_t)s.next_total);
    si->rr = si->rr_fld.h();

    /* cheb scratch. */
    s.f_d  = RDV("cgpoly.f",  (size_t)s.next_total);
    s.za_d = RDV("cgpoly.za", (size_t)s.next_total);
    s.zb_d = RDV("cgpoly.zb", (size_t)s.next_total);
    s.dd_d = RDV("cgpoly.dd", (size_t)s.next_total);

    /* ---- λmax of D̃^{-1/2}·Ã·D̃^{-1/2} by 100-iteration power iteration on
     * the device (the JAX enable_cheb_precond clone, distributed). Start
     * vector = splitmix64 of the 1-based GLOBAL id → decomposition-
     * independent floats; norms Allreduce'd. λmax ×1.05 (power iteration
     * converges from below); λmin = λmax/κ — κ tunes EFFICIENCY only, the
     * polynomial stays SPD for any spectrum ≤ λmax. */
    {
        auto rrv = si->rr_fld.d();
        auto h   = Kokkos::create_mirror_view(rrv);
        for (int i = 0; i < s.next_total; ++i) h(i) = 0.0;
        for (int i = 0; i < N; ++i) {
            unsigned long long x = (unsigned long long)(parallel ? p->myList_nod2D[i] : i + 1);
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x =  x ^ (x >> 31);
            h(i) = (double)(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
        }
        Kokkos::deep_copy(rrv, h);
        auto rowptr = S->rowptr_fld.d();
        auto colind = S->colind_fld.d();
        auto av0 = s.av0_d; auto isq = s.isq_d; auto w = s.za_d;
        double lam = 1.0;
        for (int it = 0; it < 100; ++it) {
            cgpoly_exchange(rrv, p);
            Kokkos::parallel_for("fesom_cgpoly_pow", Kokkos::RangePolicy<>(0, N),
                KOKKOS_LAMBDA(const int row) {
                    real_t sacc = 0.0;
                    const int a = rowptr(row), e = rowptr(row + 1);
                    for (int n = a; n < e; ++n) sacc += av0(n) * (isq(colind(n)) * rrv(colind(n)));
                    w(row) = isq(row) * sacc;
                });
            real_t nn = 0.0;
            Kokkos::parallel_reduce("fesom_cgpoly_nrm2", Kokkos::RangePolicy<>(0, N),
                KOKKOS_LAMBDA(const int i, real_t &l) { l += w(i) * w(i); }, nn);
            double nrm2 = (double)nn;
            if (parallel) MPI_Allreduce(MPI_IN_PLACE, &nrm2, 1, MPI_DOUBLE, MPI_SUM, comm);
            lam = sqrt(nrm2);
            if (lam == 0.0) { lam = 1.0; break; }        /* degenerate — keep λ sane */
            const real_t inv = (real_t)(1.0 / lam);
            Kokkos::parallel_for("fesom_cgpoly_pscale", Kokkos::RangePolicy<>(0, N),
                KOKKOS_LAMBDA(const int i) { rrv(i) = w(i) * inv; });
        }
        s.lam_max = lam * 1.05;
        s.lam_min = s.lam_max / kappa;
    }

    /* ---- the L80 announce (fesom_speed_int is silent — say what runs). */
    {
        long loc[4] = { worst_partner_bytes, (long)s.partner.size(),
                        (long)(s.next_total - N), (long)s.nring_rows };
        long mx[4] = { loc[0], loc[1], loc[2], loc[3] };
        if (parallel) MPI_Reduce(loc, mx, 4, MPI_LONG, MPI_MAX, 0, comm);
        if (!parallel || p->mype == 0) {
            fprintf(stderr, "[cgpoly] ACTIVE degree=%d kappa=%.4g lam=[%.6g, %.6g] rings=%d "
                            "ring-slots(max)=%ld ring-rows(max)=%ld partners(max)=%ld "
                            "worst-partner-KB(max)=%.1f — Chebyshev PCG, frozen-A ship-once "
                            "(supersedes the CGPIPE path)\n",
                    s.d, s.kappa, s.lam_min, s.lam_max, s.R,
                    mx[2], mx[3], mx[1], (double)mx[0] / 1024.0);
            fflush(stderr);
        }
    }
    s.built = true;
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
    s.sbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>();   /* M7.5 staged mirrors */
    s.rbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>();
    s.rp2_d  = Kokkos::View<int*>();
    s.ci2_d  = Kokkos::View<int*>();
    s.pv2_d  = Kokkos::View<double*>();
    s.reqs.clear();
    s.partner.clear(); s.soff.clear(); s.roff.clear();
    s.built = false;
}

/* free the persistent CGPOLY Views BEFORE Kokkos::finalize() (same
 * static-destruction hazard as CGPIPE above). */
void fesom_ssh_cgpoly_free(void)
{
    CgPolyState &s = g_cgpoly;
    s.sidx_d = RIV(); s.ridx_d = RIV();
    s.sbuf_d = RDV(); s.rbuf_d = RDV();
    s.sbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>();   /* M7.5 staged mirrors */
    s.rbuf_h = Kokkos::View<double*, fesom_halo_pinned_space>();
    s.av0_d  = RDV(); s.rp_d = RIV(); s.ci_d = RIV(); s.av_d = RDV();
    s.invd_d = RDV(); s.isq_d = RDV();
    s.f_d = RDV(); s.za_d = RDV(); s.zb_d = RDV(); s.dd_d = RDV();
    s.rf_d = RDV(); s.rza_d = RDV(); s.rzb_d = RDV(); s.rdd_d = RDV();
    s.reqs.clear();
    s.partner.clear(); s.soff.clear(); s.roff.clear();
    s.nring.clear(); s.ext.clear();
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

/* ======================================================================= *
 * M10 T2 — [ssh-wire] instrumentation + FESOM_SSH_VERIFY
 * (plan: docs/plans/20260805-m10-ssh-solvers.md; ledger: docs/SSH_SOLVERS_M10.md).
 *
 * FESOM_SSH_STATS=1  — per-solve wire counters (iters, halo-exchange events,
 *   blocking allreduces, iallreduces, solver-body kernel launches, fallbacks,
 *   final recurrence residual) printed on rank 0, plus a finalize aggregate +
 *   (CUDA) a launch-overhead micro-probe (fesom_ssh_wire_report).
 * FESOM_SSH_VERIFY=1 — post-solve TRUE residual ‖b−A·x‖ printed vs the
 *   recurrence residual (catches recurrence drift, the known pipelined-CG
 *   failure mode; armed in every M10 gate). Byte-transparent: touches App
 *   (scratch, re-derived by the next solve before any read) and d_eta HALO
 *   rows (rewritten with the same owner values by the driver's post-solve
 *   exchange); owned model state untouched. Its comm/launches are NOT
 *   wire-counted, so a census with VERIFY off and on agree.
 *
 * The M10 knob family (FESOM_SSH_* and FESOM_PCSI_*) are SOLVER options, not
 * FESOM_SPEED levers: valid on every backend (the serial solution-class
 * gates and the login testbed run them), never implied by the FESOM_SPEED
 * master. House rules kept: default OFF, unrecognised values die loudly,
 * rank-0 announce when ON (L80).
 *
 * Counting rules (census of record reads these): an "exch" is one fused
 * exchange EVENT (1-ring bracket, cgpipe 2-ring, or cgpoly R-ring — each
 * internally pack+MPI+unpack, +2 staged copies under FESOM_HALO_STAGE);
 * "ar_blk"/"ar_i" count MPI_Allreduce/MPI_Iallreduce calls when npes>1;
 * "body-launches" count Kokkos kernel launches issued by the solve itself
 * (cgpoly_apply = d+1, cgpipe_zz_ring1 = 1) EXCLUDING exchange internals
 * and selfcheck/verify diagnostics. Counters increment unconditionally
 * (an integer ++ touches no model state — knob-OFF stays byte-identical;
 * gate of record in SSH_SOLVERS_M10.md).
 * ======================================================================= */

static int fesom_ssh_env01(const char *var)
{
    const char *e = getenv(var);
    if (!e || !e[0]) return 0;
    if (strcmp(e, "0") == 0) return 0;
    if (strcmp(e, "1") == 0) return 1;
    fprintf(stderr, "[fesom_ssh] unrecognised %s='%s' (valid values: 0, 1)\n", var, e);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 0;
}

static bool ssh_stats_on()
{
    static int c = -1;
    if (c < 0) {
        c = fesom_ssh_env01("FESOM_SSH_STATS");
        int rank = 0, ini = 0;
        MPI_Initialized(&ini);
        if (ini) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (c && rank == 0) { fprintf(stderr, "[ssh-wire] FESOM_SSH_STATS = ON\n"); fflush(stderr); }
    }
    return c != 0;
}

static bool ssh_verify_on()
{
    static int c = -1;
    if (c < 0) {
        c = fesom_ssh_env01("FESOM_SSH_VERIFY");
        int rank = 0, ini = 0;
        MPI_Initialized(&ini);
        if (ini) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (c && rank == 0) { fprintf(stderr, "[ssh-verify] FESOM_SSH_VERIFY = ON "
                                              "(post-solve true residual; byte-transparent)\n"); fflush(stderr); }
    }
    return c != 0;
}

struct SshWireState {
    long solves = 0;
    long t_iters = 0, t_exch = 0, t_arb = 0, t_ari = 0, t_launch = 0, t_fallback = 0;
    int  s_exch = 0, s_arb = 0, s_ari = 0, s_launch = 0;   /* per-solve; reset at entry */
    double v_maxtrue = 0.0, v_maxgap = 0.0;                /* FESOM_SSH_VERIFY aggregates */
    long   v_fail = 0;
};
static SshWireState g_sshwire;
#define SSH_WIRE_LAUNCH(n) (g_sshwire.s_launch += (n))

/* Fold the per-solve counters into the totals + the rank-0 per-solve line.
 * Every rank counts identically (exchanges/reductions are collective events). */
static void ssh_wire_close_solve(int iters, double res, double rtol, fesom_partit *partit)
{
    SshWireState &w = g_sshwire;
    ++w.solves;
    w.t_iters += iters; w.t_exch += w.s_exch; w.t_arb += w.s_arb;
    w.t_ari   += w.s_ari; w.t_launch += w.s_launch;
    if (ssh_stats_on() && (partit == NULL || partit->mype == 0)) {
        fprintf(stderr, "[ssh-wire] solve %ld: iters=%d exch=%d ar_blk=%d ar_i=%d "
                        "body-launches=%d fb=%ld res=%.4e rtol=%.4e\n",
                w.solves, iters, w.s_exch, w.s_arb, w.s_ari, w.s_launch,
                w.t_fallback, res, rtol);
        fflush(stderr);
    }
}

/* ---- M10 T3: FESOM_SSH_TRACE — per-iteration α/β/residual on rank 0 ------
 * The Layer-0 comparator's instrument: cg2/pipecg/oati must reproduce the
 * reference PCG's α/β sequences to rounding on real dumps. %.17g = exact
 * double round-trip. Diagnostic prints only — byte-transparent. */
static bool ssh_trace_on()
{
    static int c = -1;
    if (c < 0) c = fesom_ssh_env01("FESOM_SSH_TRACE");
    return c != 0;
}

/* ---- M10 T3: FESOM_SSH_DUMP=<csv 1-based solves> + FESOM_SSH_DUMP_DIR ----
 * Per-rank raw dumps of a solve (CSR + pr + b + x0 + x_final + iters) for
 * the offline lab (tools/fesom_ssh_lab.cpp). Format: src/fesom_ssh_dump.h.
 * The solve index is the [ssh-wire] solve counter (== model step for the
 * 1-solve/step main loop; init-context solves use the C solver, not this). */
static const char *ssh_dump_dir()
{
    static int init = 0;
    static const char *dir = NULL;
    if (!init) {
        init = 1;
        const char *steps = getenv("FESOM_SSH_DUMP");
        if (steps && steps[0]) {
            dir = getenv("FESOM_SSH_DUMP_DIR");
            if (!dir || !dir[0])
                FESOM_DIE("FESOM_SSH_DUMP is set but FESOM_SSH_DUMP_DIR is missing");
        }
    }
    return dir;
}

static bool ssh_dump_wanted(long solve)
{
    static int init = 0;
    static std::vector<long> steps;
    if (!init) {
        init = 1;
        const char *e = getenv("FESOM_SSH_DUMP");
        if (e && e[0]) {
            char *dup = strdup(e), *save = NULL;
            for (char *tok = strtok_r(dup, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
                char *end = NULL;
                long v = strtol(tok, &end, 10);
                if (!end || *end != '\0' || v <= 0)
                    FESOM_DIE("FESOM_SSH_DUMP: bad step '%s' (want a csv of positive ints)", tok);
                steps.push_back(v);
            }
            free(dup);
        }
    }
    for (long s : steps) if (s == solve) return true;
    return false;
}

static void ssh_dump_write(const fesom_ssh_stiff *S, const fesom_solverinfo *si,
                           const struct fesom_mesh *mesh, fesom_partit *partit,
                           long solve,
                           const std::vector<real_t> &av, const std::vector<real_t> &pr,
                           const std::vector<real_t> &b,  const std::vector<real_t> &x0,
                           const real_t *xf, int iters, double final_res, double rtol)
{
    static_assert(sizeof(real_t) == 8, "ssh-dump format is f64; SP builds are out of scope");
    const char *dir = ssh_dump_dir();
    const int mype = partit ? partit->mype : 0;
    const int npes = partit ? partit->npes : 1;
    const int N    = mesh->myDim_nod2D;
    char stepdir[1024], path[1200];
    snprintf(stepdir, sizeof stepdir, "%s/step%04ld", dir, solve);
    if (mype == 0) {
        mkdir(dir, 0755);                                   /* EEXIST is fine */
        if (mkdir(stepdir, 0755) != 0 && errno != EEXIST)
            FESOM_DIE("ssh-dump: mkdir %s failed (errno %d)", stepdir, errno);
    }
    if (partit && npes > 1) MPI_Barrier(partit->MPI_COMM_FESOM);
    snprintf(path, sizeof path, "%s/rank%05d.bin", stepdir, mype);
    FILE *f = fopen(path, "wb");
    FESOM_CHECK(f, "ssh-dump: cannot open %s", path);
    uint64_t u64; int32_t i32; double f64;
    #define WSCAL(x) FESOM_CHECK(fesom_sshdump_wr(f, &(x), sizeof(x)) == 0, "ssh-dump: write failed %s", path)
    #define WARR(p, n) FESOM_CHECK(fesom_sshdump_wr_arr(f, (p), (n)) == 0, "ssh-dump: array write failed %s", path)
    u64 = FESOM_SSHDUMP_MAGIC; WSCAL(u64);
    i32 = FESOM_SSHDUMP_VERSION; WSCAL(i32);
    i32 = (int32_t)solve; WSCAL(i32);
    i32 = npes; WSCAL(i32);   i32 = mype; WSCAL(i32);
    i32 = N; WSCAL(i32);      i32 = mesh->eDim_nod2D; WSCAL(i32);
    i32 = S->nnz; WSCAL(i32); i32 = mesh->nod2D; WSCAL(i32);
    f64 = (double)FESOM_PHASE1_DT; WSCAL(f64);
    f64 = (double)si->soltol; WSCAL(f64);
    i32 = si->maxiter; WSCAL(i32);
    u64 = partit ? fesom_sshdump_fnv1a(partit->myList_nod2D,
                       (size_t)(N + mesh->eDim_nod2D) * sizeof(int)) : 0;
    WSCAL(u64);
    WARR(S->rowptr, (size_t)(N + 1) * sizeof(int));
    WARR(S->colind, (size_t)S->nnz * sizeof(int));
    WARR(av.data(), av.size() * sizeof(real_t));
    WARR(pr.data(), pr.size() * sizeof(real_t));
    WARR(b.data(),  b.size()  * sizeof(real_t));
    WARR(x0.data(), x0.size() * sizeof(real_t));
    WARR(xf,        (size_t)N * sizeof(real_t));
    i32 = iters; WSCAL(i32);
    f64 = final_res; WSCAL(f64);
    f64 = rtol; WSCAL(f64);
    u64 = FESOM_SSHDUMP_MAGIC; WSCAL(u64);
    #undef WSCAL
    #undef WARR
    fclose(f);
    if (mype == 0) {
        fprintf(stderr, "[ssh-dump] solve %ld -> %s (np%d, nnz=%d, iters=%d)\n",
                solve, stepdir, npes, S->nnz, iters);
        fflush(stderr);
    }
}

/* Finalize aggregate (fesom_main, before Kokkos::finalize). No-op unless
 * FESOM_SSH_STATS=1. The CUDA micro-probe prices a kernel launch two ways:
 * back-to-back enqueue (async, one fence at the end) and fully-fenced (the
 * latency-bound shape a blocking allreduce forces every iteration) — the
 * numbers the pipecg/oati pre-registrations weigh +launches against
 * −allreduces with (plan T2). Rank-0 only; not collective. */
void fesom_ssh_wire_report(void)
{
    if (!ssh_stats_on()) return;
    SshWireState &w = g_sshwire;
    int rank = 0, ini = 0;
    MPI_Initialized(&ini);
    if (ini) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;
    if (w.solves > 0) {
        const double s = (double)w.solves;
        fprintf(stderr,
            "[ssh-wire] AGGREGATE over %ld solves: iters/solve=%.2f exch/solve=%.2f "
            "ar_blk/solve=%.2f ar_i/solve=%.2f body-launches/solve=%.2f fallbacks=%ld\n",
            w.solves, (double)w.t_iters / s, (double)w.t_exch / s, (double)w.t_arb / s,
            (double)w.t_ari / s, (double)w.t_launch / s, w.t_fallback);
        if (ssh_verify_on())
            fprintf(stderr, "[ssh-verify] AGGREGATE: max true-res=%.6e  max |true-rec| gap=%.3e  "
                            "true>rtol events=%ld\n", w.v_maxtrue, w.v_maxgap, w.v_fail);
        fflush(stderr);
    }
#ifdef KOKKOS_ENABLE_CUDA
    {
        const int NL = 10000, NF = 1000;
        Kokkos::fence();
        double t0 = MPI_Wtime();
        for (int i = 0; i < NL; ++i)
            Kokkos::parallel_for("fesom_sshwire_probe", Kokkos::RangePolicy<>(0, 1),
                KOKKOS_LAMBDA(const int) {});
        Kokkos::fence();
        const double async_us = (MPI_Wtime() - t0) * 1e6 / (double)NL;
        t0 = MPI_Wtime();
        for (int i = 0; i < NF; ++i) {
            Kokkos::parallel_for("fesom_sshwire_probe", Kokkos::RangePolicy<>(0, 1),
                KOKKOS_LAMBDA(const int) {});
            Kokkos::fence();
        }
        const double fenced_us = (MPI_Wtime() - t0) * 1e6 / (double)NF;
        fprintf(stderr, "[ssh-wire] launch-overhead probe (rank 0): async %.2f us/launch, "
                        "fenced %.2f us/launch (n=%d/%d)\n", async_us, fenced_us, NL, NF);
        fflush(stderr);
    }
#endif
}

/* ======================================================================= *
 * M10 T5a — FESOM_SSH_SOLVER dispatch + shared guard infrastructure
 * (plan docs/plans/20260805-m10-ssh-solvers.md; math
 *  docs/plans/20260805-m10-ssh-derivations.md).
 *
 * The M10 solvers are SOLVER OPTIONS, not FESOM_SPEED levers: they are valid
 * on every backend (the serial solution-class gates and the login testbed run
 * them) and the FESOM_SPEED master never implies one. `cg` = the existing
 * path, byte-untouched: with FESOM_SSH_SOLVER unset or `cg`, nothing below
 * this banner executes and the binary is bit-identical to HEAD.
 *
 * ⭐ FESOM_SSH_SYMPRE — the derivations' headline (§0.4/§0.5). The built
 * preconditioner is M⁻¹ = D⁻¹C with C symmetric and D = diag(A); the D⁻¹ on
 * the LEFT makes M⁻¹ non-symmetric (measured defect ratio 0.638). Every
 * CG-CG-family solver replaces PCG's explicit (p,Ap) with the recurrence
 * σ_i = δ_i − β_i²σ_{i-1}, whose derivation needs (r_i, M⁻ᵀ r_{i-1}) = 0 —
 * i.e. a SYMMETRIC M⁻¹. On CORE2 the as-built preconditioner drives that
 * recurrence 21.8 % away from the truth by iteration 60 (lab --sigma-drift),
 * so α is wrong by the same amount. M̃⁻¹ = D^{-1/2} C D^{-1/2} is symmetric,
 * SIMILAR to M⁻¹ (identical spectrum: D^{1/2}(D⁻¹C)D^{-1/2} = D^{-1/2}CD^{-1/2}),
 * same sparsity, and costs one scaling at setup. Default ON for every non-cg
 * solver; `=0` reproduces the literal MITgcm form (Sergey's configuration) for
 * the falsification leg. Baseline `cg` NEVER consults it.
 * ======================================================================= */

enum fesom_ssh_solver_kind {
    FESOM_SSHSOLV_CG = 0, FESOM_SSHSOLV_CG2, FESOM_SSHSOLV_PIPECG,
    FESOM_SSHSOLV_OATI, FESOM_SSHSOLV_PCSI
};

static const char *ssh_solver_name(int k)
{
    switch (k) {
        case FESOM_SSHSOLV_CG:     return "cg";
        case FESOM_SSHSOLV_CG2:    return "cg2";
        case FESOM_SSHSOLV_PIPECG: return "pipecg";
        case FESOM_SSHSOLV_OATI:   return "oati";
        case FESOM_SSHSOLV_PCSI:   return "pcsi";
    }
    return "?";
}

/* Resolve once; abort loudly on an unrecognised value (house idiom), announce on rank 0
 * so a dead knob is impossible to miss (L80). */
static int ssh_solver_kind()
{
    static int k = -1;
    if (k >= 0) return k;
    const char *e = getenv("FESOM_SSH_SOLVER");
    if      (!e || !e[0] || strcmp(e, "cg") == 0) k = FESOM_SSHSOLV_CG;
    else if (strcmp(e, "cg2")    == 0)            k = FESOM_SSHSOLV_CG2;
    else if (strcmp(e, "pipecg") == 0)            k = FESOM_SSHSOLV_PIPECG;
    else if (strcmp(e, "oati")   == 0)            k = FESOM_SSHSOLV_OATI;
    else if (strcmp(e, "pcsi")   == 0)            k = FESOM_SSHSOLV_PCSI;
    else {
        fprintf(stderr, "[fesom_ssh] unrecognised FESOM_SSH_SOLVER='%s' "
                        "(valid: cg, cg2, pipecg, oati, pcsi)\n", e);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ini = 0;
    MPI_Initialized(&ini);
    if (ini) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && k != FESOM_SSHSOLV_CG) {
        fprintf(stderr, "[ssh-solver] FESOM_SSH_SOLVER = %s (M10)\n", ssh_solver_name(k));
        fflush(stderr);
    }
    return k;
}

/* FESOM_SSH_RING=0 — literal multi-exchange forms. Bring-up/debug and the npes==1
 * degradation ONLY; never a gated or recommended configuration. */
static bool ssh_ring_on()
{
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("FESOM_SSH_RING");
        c = (e && e[0]) ? fesom_ssh_env01("FESOM_SSH_RING") : 1;
    }
    return c != 0;
}

/* FESOM_SSH_FALLBACK (default 1 = armed). 0 exists only for lab/probe experiments. */
static bool ssh_fallback_armed()
{
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("FESOM_SSH_FALLBACK");
        c = (e && e[0]) ? fesom_ssh_env01("FESOM_SSH_FALLBACK") : 1;
        int rank = 0, ini = 0;
        MPI_Initialized(&ini);
        if (ini) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (!c && rank == 0) {
            fprintf(stderr, "[ssh-solver] !! FESOM_SSH_FALLBACK=0 — the auto-fallback guard is "
                            "DISARMED (experiments only; never in a certified run)\n");
            fflush(stderr);
        }
    }
    return c != 0;
}

/* FESOM_SSH_SYMPRE (default 1 for non-cg solvers). */
static bool ssh_sympre_on()
{
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("FESOM_SSH_SYMPRE");
        c = (e && e[0]) ? fesom_ssh_env01("FESOM_SSH_SYMPRE") : 1;
    }
    return c != 0;
}

/* The symmetrised preconditioner M̃⁻¹ = D^{-1/2} C D^{-1/2}, i.e.
 *     p̃r[i,j] = pr[i,j] · sqrt(d_i/d_j),   p̃r[i,i] = pr[i,i] = 1/d_i.
 * Built ONCE (pr_values is frozen — the Fortran precedent), device-resident, same CSR
 * sparsity as pr_values so the cgpipe ring-shipping machinery is untouched. The diagonal
 * of A is needed on owned+ring1 rows, which is exactly what the existing host halo
 * exchange provides at setup (this runs once, off the hot path).
 * NOTE the ring1 rows shipped by cgpipe_build carry the ORIGINAL pr_values; a solver that
 * uses SYMPRE must therefore not mix them — T5b ships the symmetrised rows instead. */
struct SshSymPreState {
    bool built = false;
    Kokkos::View<double*> pr_d;                       /* [nnz] symmetrised */
    std::vector<real_t>   pr_h;                       /* host copy (ring shipping, lab) */
    std::vector<real_t>   diag_h;                     /* [N+eDim] diag(A), halo-current */
};
static SshSymPreState g_sympre;

static void ssh_sympre_build(const fesom_ssh_stiff *S, const struct fesom_mesh *mesh,
                             fesom_partit *partit)
{
    SshSymPreState &s = g_sympre;
    if (s.built) return;
    const int N    = mesh->myDim_nod2D;
    const int Next = N + mesh->eDim_nod2D;

    /* diag(A) on owned+ring1. Read the DEVICE-authoritative values (under zstar the host
     * mirror can be stale — the cgpoly frozen-Ã precedent). */
    std::vector<real_t> vals_h((size_t)S->nnz);
    {
        auto m = Kokkos::create_mirror_view(S->values_fld.d());
        Kokkos::deep_copy(m, S->values_fld.d());
        for (int i = 0; i < S->nnz; ++i) vals_h[(size_t)i] = m(i);
    }
    s.diag_h.assign((size_t)Next, 0.0);
    for (int r = 0; r < N; ++r) s.diag_h[(size_t)r] = vals_h[(size_t)S->rowptr[r]];
    if (partit && partit->npes > 1)
        fesom_halo_exchange(s.diag_h.data(), FESOM_HALO_NOD2D, 1, 1, partit);

    std::vector<real_t> pr_h((size_t)S->nnz);
    {
        auto m = Kokkos::create_mirror_view(S->pr_values_fld.d());
        Kokkos::deep_copy(m, S->pr_values_fld.d());
        for (int i = 0; i < S->nnz; ++i) pr_h[(size_t)i] = m(i);
    }
    long bad = 0;
    for (int r = 0; r < N; ++r)
        for (int n = S->rowptr[r]; n < S->rowptr[r + 1]; ++n) {
            const int c = S->colind[n];
            if (c == r) continue;
            const real_t di = s.diag_h[(size_t)r], dj = s.diag_h[(size_t)c];
            if (!(di > 0.0) || !(dj > 0.0)) { ++bad; continue; }
            pr_h[(size_t)n] *= sqrt(di / dj);
        }
    long gbad = bad;
    if (partit && partit->npes > 1)
        MPI_Allreduce(MPI_IN_PLACE, &gbad, 1, MPI_LONG, MPI_SUM, partit->MPI_COMM_FESOM);
    FESOM_CHECK(gbad == 0, "ssh-sympre: %ld non-positive diagonal(s) — D^{-1/2} undefined", gbad);

    s.pr_h = pr_h;
    s.pr_d = Kokkos::View<double*>("ssh.sympre.pr", (size_t)S->nnz);
    {
        auto h = Kokkos::create_mirror_view(s.pr_d);
        for (int i = 0; i < S->nnz; ++i) h(i) = pr_h[(size_t)i];
        Kokkos::deep_copy(s.pr_d, h);
    }
    s.built = true;

    /* the L80 observable: print the defect BEFORE and AFTER on owned pairs, so a log proves
     * the knob fired and by how much (derivations §0.2 numbers reproduce here). */
    double d_before = 0.0, d_after = 0.0, mx = 0.0;
    for (int r = 0; r < N; ++r)
        for (int n = S->rowptr[r]; n < S->rowptr[r + 1]; ++n) {
            const int c = S->colind[n];
            if (c == r || c >= N) continue;
            for (int m2 = S->rowptr[c]; m2 < S->rowptr[c + 1]; ++m2)
                if (S->colind[m2] == r) {
                    d_before = fmax(d_before, fabs((double)pr_h[(size_t)n] * sqrt(s.diag_h[(size_t)c] / s.diag_h[(size_t)r])
                                                 - (double)pr_h[(size_t)m2] * sqrt(s.diag_h[(size_t)r] / s.diag_h[(size_t)c])));
                    d_after  = fmax(d_after,  fabs((double)pr_h[(size_t)n] - (double)pr_h[(size_t)m2]));
                    mx       = fmax(mx, fabs((double)pr_h[(size_t)n]));
                    break;
                }
        }
    double red[3] = { d_before, d_after, mx };
    if (partit && partit->npes > 1)
        MPI_Allreduce(MPI_IN_PLACE, red, 3, MPI_DOUBLE, MPI_MAX, partit->MPI_COMM_FESOM);
    if (!partit || partit->mype == 0) {
        fprintf(stderr, "[ssh-sympre] BUILT: M~ = D^-1/2 C D^-1/2 — symmetry defect ratio "
                        "%.3e -> %.3e (max|M~_ij| = %.3e)\n",
                red[2] > 0 ? red[0] / red[2] : 0.0, red[2] > 0 ? red[1] / red[2] : 0.0, red[2]);
        fflush(stderr);
    }
}

void fesom_ssh_m10_free(void);   /* defined after the per-solver states (T5b+) */

/* Interaction matrix (plan §Technical Details) — one cell, one behaviour. Evaluated once,
 * on the first solve, with the solver kind already resolved. */
static void ssh_solver_check_interactions(int kind, int npes)
{
    if (kind == FESOM_SSHSOLV_CG) return;
    static bool done = false;
    if (done) return;
    done = true;

    const char *v = getenv("FESOM_KK_VERIFY");
    FESOM_CHECK(!(v && strcmp(v, "ssh") == 0),
                "FESOM_SSH_SOLVER=%s is incompatible with FESOM_KK_VERIFY=ssh "
                "(the C twin runs the legacy solver; the iterates differ within tolerance)",
                ssh_solver_name(kind));

    /* ⚠️ L80/dead-knob: `FESOM_SPEED_CGPOLY` selects a DIFFERENT preconditioner (a Chebyshev
     * polynomial in A) that the M10 variants do not yet consult — they apply `pr_values` or
     * its symmetrised twin. Letting the combination run would measure the poly knob as doing
     * nothing, which is indistinguishable from a lever that does not pay. So it DIES until
     * T9 wires `cgpoly_apply` in as the M-slot. `pcsi` and `oati` die permanently
     * (nested Chebyshev / no poly composition for the unrolled recurrences). */
    static int s_cgpoly = -2;
    const int cgpoly_d = fesom_speed_int("CGPOLY", 0, &s_cgpoly);
    FESOM_CHECK(!(cgpoly_d >= 1 && (kind == FESOM_SSHSOLV_PCSI || kind == FESOM_SSHSOLV_OATI)),
                "FESOM_SSH_SOLVER=%s is incompatible with FESOM_SPEED_CGPOLY "
                "(pcsi: nested Chebyshev; oati: the unrolled recurrences have no poly composition)",
                ssh_solver_name(kind));
    FESOM_CHECK(!(cgpoly_d >= 1 && (kind == FESOM_SSHSOLV_CG2 || kind == FESOM_SSHSOLV_PIPECG)),
                "FESOM_SSH_SOLVER=%s + FESOM_SPEED_CGPOLY=%d is NOT YET IMPLEMENTED (M10 Task 9 "
                "slots cgpoly_apply in as the preconditioner). Refusing to run rather than "
                "silently ignoring the poly knob — a dead knob is indistinguishable from a "
                "lever that does not pay (L80).", ssh_solver_name(kind), cgpoly_d);

    FESOM_CHECK(!(kind == FESOM_SSHSOLV_PCSI && !ssh_sympre_on()),
                "FESOM_SSH_SOLVER=pcsi requires FESOM_SSH_SYMPRE=1 — the Chebyshev/Lanczos "
                "theory needs a self-adjoint preconditioner (derivations sec 4.4)");

    const bool host_halo = getenv("FESOM_HOST_HALO") != NULL;
    int rank = 0, ini = 0;
    MPI_Initialized(&ini);
    if (ini) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if ((npes == 1 || host_halo) && ssh_ring_on() && rank == 0) {
        fprintf(stderr, "[ssh-solver] !! ring composition unavailable (%s) — %s degrades to "
                        "the FESOM_SSH_RING=0 literal form (bring-up path, not a gated config)\n",
                npes == 1 ? "npes==1" : "FESOM_HOST_HALO=1", ssh_solver_name(kind));
        fflush(stderr);
    }
}

/* Shared fallback guard. Every trigger derives from an ALLREDUCED scalar, so the decision is
 * collective by construction (R6 — a rank-local branch would deadlock; the farc hang shows
 * the fleet cost). The caller snapshots X0 at solve entry and restores it before redoing the
 * solve with baseline cg. */
enum { SSH_FB_NONE = 0, SSH_FB_NAN, SSH_FB_STALL, SSH_FB_INDEF, SSH_FB_MAXITER };

static const char *ssh_fb_reason(int r)
{
    switch (r) {
        case SSH_FB_NAN:     return "NaN/Inf in a reduced scalar";
        case SSH_FB_STALL:   return "residual stalled or grew over the watch window";
        case SSH_FB_INDEF:   return "r.u <= 0 (indefinite preconditioner)";
        case SSH_FB_MAXITER: return "maxiter exhausted without convergence";
    }
    return "none";
}

static void ssh_fb_announce(int reason, int iters, double res, fesom_partit *partit, long solve)
{
    ++g_sshwire.t_fallback;
    if (!partit || partit->mype == 0) {
        fprintf(stderr, "[ssh-solver] !! FALLBACK on solve %ld after %d iters (res=%.4e): %s "
                        "— restoring X0 and redoing this solve with baseline cg\n",
                solve, iters, res, ssh_fb_reason(reason));
        fflush(stderr);
    }
}

/* FESOM_SSH_STALL_WINDOW — the guard's plateau tolerance, in the units each solver counts
 * (iterations for cg2/pipecg, CHECK events for pcsi, PAIRS for oati). Unset keeps the
 * per-solver value that has always been compiled in, so knob-off is byte-identical.
 *
 * It exists because the stall test is a HEURISTIC, not a convergence proof: it fires after
 * N consecutive steps without a 0.1 % residual drop, and an ill-conditioned system plateaus
 * by nature. Baseline `cg` carries NO such guard (its body only dies on NaN or maxiter), so
 * `fallbacks=0` on the cg path is structurally guaranteed rather than earned — the variants
 * are the only monitored path. Widening this window is how "would the variant have converged
 * if the guard had not aborted it?" gets measured (M10 open item 3). */
static int ssh_stall_window(int dflt)
{
    static int cached = -1;                          /* 0 = unset, >0 = the override */
    if (cached < 0) {
        const char *e = getenv("FESOM_SSH_STALL_WINDOW");
        if (e && e[0]) {
            cached = atoi(e);
            FESOM_CHECK(cached >= 1, "FESOM_SSH_STALL_WINDOW='%s' must be a positive integer", e);
            int rank = 0, ini = 0;
            MPI_Initialized(&ini);
            if (ini) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {                         /* L80: a knob must announce that it fired */
                fprintf(stderr, "[ssh-solver] FESOM_SSH_STALL_WINDOW=%d (compiled defaults: "
                                "20 cg2/pipecg, 10 pcsi/oati) — the fallback stall heuristic "
                                "is WIDENED; this is a diagnostic, not a production setting\n",
                        cached);
                fflush(stderr);
            }
        } else {
            cached = 0;
        }
    }
    return cached > 0 ? cached : dflt;
}

/* The variant entry point. Returns the iteration count, or -1 to mean "guard tripped, X0
 * restored, redo this solve with baseline cg". Implemented per solver in T5b–T8b. */
static int ssh_solve_variant(int kind, const fesom_ssh_stiff *S, fesom_solverinfo *si,
                             const struct fesom_mesh *mesh, struct fesom_dyn *dyn);

/* ======================================================================= *
 * M10 T5b — `cg2`: Chronopoulos–Gear preconditioned CG
 * (derivations §1; sources: own derivation, Sergey solvers.F90:3,
 *  Cools–Vanroose Alg. 4, and [P] App. B2 ChronGear as a 4th cross-check).
 *
 *   p_i = u_i + β_i p_{i-1}          s_i = w_i + β_i s_{i-1}
 *   x_{i+1} = x_i + α_i p_i          r_{i+1} = r_i − α_i s_i
 *   u_{i+1} = M⁻¹ r_{i+1}            w_{i+1} = A u_{i+1}
 *   ONE fused 3-element Allreduce: γ=(r,u), δ=(w,u), ρ=(r,r)
 *   β_{i+1} = γ_{i+1}/γ_i            α_{i+1} = (δ/γ_{i+1} − β_{i+1}/α_i)⁻¹
 *
 * vs baseline PCG: 2 blocking Allreduce/iter → 1. The α recurrence replaces PCG's explicit
 * (p,Ap); ⚠️ that substitution is valid ONLY for a symmetric M⁻¹ (derivations §0.4 — with
 * the as-built pr_values it is 21.8 % wrong on CORE2), hence FESOM_SSH_SYMPRE, default ON.
 *
 * RING COMPOSITION (ours; no source does this): exchange r on rings 1+2 in ONE fused
 * message ⇒ u = M⁻¹r is computable on owned AND ring1 (ring1 preconditioner rows are
 * shipped verbatim at setup), so w = Au at owned rows needs no second message; p and s
 * keep their ring1 values by recurrence. ⇒ 1 exchange + 1 Allreduce per iteration.
 * FESOM_SSH_RING=0 gives the literal 2-exchange form (bring-up/npes==1 only).
 * ======================================================================= */

struct SshCg2State {
    fesom::Field uu, ww, pp2, ss;                     /* ring-extent scratch */
    int ext = 0;
};
static SshCg2State g_cg2;

/* All four vectors are sized owned+ring1 regardless of composition: `uu` is gathered at
 * halo columns by the w = A·u SpMV in BOTH forms (in the literal form it is halo-exchanged
 * there, which writes [N, N+eDim)), and pp/ss carry ring1 values by recurrence in the ring
 * form. Sizing `uu` owned-only overflows the halo write — caught by the np2 literal-form
 * smoke, which aborted. The recurrence EXTENT still follows the active composition. */
static void ssh_cg2_alloc(const struct fesom_mesh *mesh, bool /*ring*/)
{
    const int want = mesh->myDim_nod2D + mesh->eDim_nod2D;
    if (g_cg2.ext == want) return;
    g_cg2.uu .alloc("ssh.cg2.uu",  (size_t)want);
    g_cg2.ww .alloc("ssh.cg2.ww",  (size_t)want);
    g_cg2.pp2.alloc("ssh.cg2.pp",  (size_t)want);
    g_cg2.ss .alloc("ssh.cg2.ss",  (size_t)want);
    g_cg2.ext = want;
}

static int ssh_solve_cg2(const fesom_ssh_stiff *S, fesom_solverinfo *si,
                         const struct fesom_mesh *mesh, struct fesom_dyn *dyn)
{
    const int     N        = mesh->myDim_nod2D;
    fesom_partit *partit   = si->partit;
    const int     parallel = (partit && partit->npes > 1);
    const int     N_global = parallel ? mesh->nod2D : N;
    const long    solve_id = g_sshwire.solves + 1;

    /* ring composition needs the cgpipe 2-ring graph; it is unavailable at npes==1 and under
     * FESOM_HOST_HALO (announced in the interaction check). */
#ifdef KOKKOS_ENABLE_CUDA
    const bool transport_ok = fesom_halo_device_active();
#else
    const bool transport_ok = true;
#endif
    const bool ring = ssh_ring_on() && parallel && transport_ok;
    /* ⚠️ ORDER IS LOAD-BEARING: the symmetrised preconditioner must exist BEFORE
     * cgpipe_build, because the ring1 rows are shipped once and must be the same values the
     * owner uses on its owned rows. Mixing them silently corrupts every ring1 `u`. */
    if (ssh_sympre_on()) {
        ssh_sympre_build(S, mesh, partit);
        FESOM_CHECK(!g_cgpipe.built || cgpipe_ship_pr == g_sympre.pr_h.data(),
                    "ssh-sympre: the CGPIPE ring1 rows were shipped with a DIFFERENT "
                    "preconditioner (FESOM_SPEED_CGPIPE ran first?) — refusing to mix");
        cgpipe_ship_pr = g_sympre.pr_h.data();
    }
    if (ring && !g_cgpipe.built) cgpipe_build(S, si, mesh, partit);
    ssh_cg2_alloc(mesh, ring);

    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    auto vals   = S->values_fld.d();
    auto prvals = ssh_sympre_on() ? Kokkos::View<const double*>(g_sympre.pr_d)
                                  : Kokkos::View<const double*>(S->pr_values_fld.d());
    auto X   = dyn->d_eta_fld.d();
    auto rhs = dyn->ssh_rhs_fld.d();
    auto rr  = si->rr_fld.d();
    auto uu  = g_cg2.uu.d();
    auto ww  = g_cg2.ww.d();
    auto pp  = g_cg2.pp2.d();
    auto ss  = g_cg2.ss.d();
    const int E = ring ? N + mesh->eDim_nod2D : N;

    auto exch = [&](fesom::Field &f) {
        if (!parallel) return;
        ++g_sshwire.s_exch;
#ifdef KOKKOS_ENABLE_CUDA
        if (fesom_halo_device_active()) { fesom_halo_exchange_device(f, FESOM_HALO_NOD2D, 1, 1, partit); return; }
#endif
        f.modify_device(); f.sync_host();
        fesom_halo_exchange(f.h_checked(), FESOM_HALO_NOD2D, 1, 1, partit);
        f.modify_host();   f.sync_device();
    };
    /* the residual-class exchange: ring form = ONE fused 2-ring message, else the 1-ring bracket */
    auto exch_r = [&]() {
        if (!parallel) return;
        if (ring) { ++g_sshwire.s_exch; cgpipe_exchange_rr(si->rr_fld, partit); }
        else      exch(si->rr_fld);
    };
    /* u = M⁻¹ r on owned rows (+ ring1 by shipped rows in the ring form) */
    auto apply_M = [&]() {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_cg2_psolve", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n = a; n < e; ++n) s += prvals(n) * rr(colind(n));
                uu(row) = s;
            });
        if (ring) { SSH_WIRE_LAUNCH(1); cgpipe_zz_ring1(rr, uu); }
        else      { g_cg2.uu.modify_device(); exch(g_cg2.uu); }
    };

    /* ---- initialisation: r₀ = b − Ax₀, u₀ = M⁻¹r₀, w₀ = Au₀ ---- */
    SSH_WIRE_LAUNCH(1);
    real_t s0 = cg_dot(rhs, rhs, N);
    if (parallel) { MPI_Allreduce(MPI_IN_PLACE, &s0, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }
    const real_t rtol = si->soltol * sqrt(s0 / (real_t)N_global);
    if (s0 == 0.0) {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_cg2_zeroX", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) { X(row) = 0.0; });
        dyn->d_eta_fld.modify_device();
        si->last_iters = 0;
        ssh_wire_close_solve(0, 0.0, (double)rtol, partit);
        return 0;
    }

    exch(dyn->d_eta_fld);
    SSH_WIRE_LAUNCH(2);
    cg_spmv(rowptr, colind, vals, X, rr, N);
    Kokkos::parallel_for("fesom_cg2_r0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) { rr(row) = rhs(row) - rr(row); });
    si->rr_fld.modify_device();
    exch_r();
    apply_M();
    SSH_WIRE_LAUNCH(1);
    Kokkos::parallel_for("fesom_cg2_w0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            real_t s = 0.0;
            const int a = rowptr(row), e = rowptr(row + 1);
            for (int n = a; n < e; ++n) s += vals(n) * uu(colind(n));
            ww(row) = s;
        });
    /* p₋₁ = s₋₁ = 0 over the ACTIVE extent (β₀ = 0 makes the first update p₀ = u₀) */
    SSH_WIRE_LAUNCH(1);
    Kokkos::parallel_for("fesom_cg2_zero_ps", Kokkos::RangePolicy<>(0, E),
        KOKKOS_LAMBDA(const int row) { pp(row) = 0.0; ss(row) = 0.0; });

    real_t g3[3] = { 0.0, 0.0, 0.0 };
    SSH_WIRE_LAUNCH(1);
    Kokkos::parallel_reduce("fesom_cg2_dots0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int i, real_t &l0, real_t &l1, real_t &l2) {
            l0 += rr(i) * uu(i); l1 += ww(i) * uu(i); l2 += rr(i) * rr(i);
        }, g3[0], g3[1], g3[2]);
    if (parallel) { MPI_Allreduce(MPI_IN_PLACE, g3, 3, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }

    double gamma = (double)g3[0], delta = (double)g3[1];
    double alpha = 0.0, beta = 0.0, resid = sqrt((double)g3[2] / (double)N_global);
    int fb = SSH_FB_NONE;
    if (!(gamma == gamma) || !(delta == delta)) fb = SSH_FB_NAN;
    else if (gamma <= 0.0)                       fb = SSH_FB_INDEF;   /* r·M⁻¹r ≤ 0 */
    else if (delta == 0.0)                       fb = SSH_FB_NAN;
    else alpha = gamma / delta;

    /* ---- iterations ---- */
    int iter = 0;
    double best = resid;
    int stall = 0;
    const int STALL_WINDOW = ssh_stall_window(20);
    if (fb == SSH_FB_NONE && resid >= rtol)
    for (iter = 1; iter <= si->maxiter; ++iter) {
        const real_t al = (real_t)alpha, be = (real_t)beta;
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_cg2_update", Kokkos::RangePolicy<>(0, E),
            KOKKOS_LAMBDA(const int row) {
                const real_t p = uu(row) + be * pp(row);
                const real_t s = ww(row) + be * ss(row);
                pp(row) = p; ss(row) = s;
                if (row < 0) return;                   /* (no-op; keeps the lambda uniform) */
            });
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_cg2_xr", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                X (row) += al * pp(row);
                rr(row) -= al * ss(row);
            });
        si->rr_fld.modify_device();
        exch_r();
        apply_M();
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_reduce("fesom_cg2_spmv_dots", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l0, real_t &l1, real_t &l2) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n = a; n < e; ++n) s += vals(n) * uu(colind(n));
                ww(row) = s;
                l0 += rr(row) * uu(row);
                l1 += s       * uu(row);
                l2 += rr(row) * rr(row);
            }, g3[0], g3[1], g3[2]);
        if (parallel) { MPI_Allreduce(MPI_IN_PLACE, g3, 3, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }

        const double gnew = (double)g3[0], dnew = (double)g3[1];
        resid = sqrt((double)g3[2] / (double)N_global);
        if (ssh_trace_on() && (partit == NULL || partit->mype == 0))
            fprintf(stderr, "[ssh-trace] it=%d al=%.17g be=%.17g res=%.17g\n",
                    iter, alpha, beta, resid);

        if (!(gnew == gnew) || !(dnew == dnew) || !(resid == resid)) { fb = SSH_FB_NAN;   break; }
        if (resid < rtol) break;
        if (gnew <= 0.0)                                             { fb = SSH_FB_INDEF; break; }
        if (resid < best * 0.999) { best = resid; stall = 0; }
        else if (++stall >= STALL_WINDOW || resid > 1e3 * best)      { fb = SSH_FB_STALL; break; }

        beta  = gnew / gamma;
        const double inv = dnew / gnew - beta / alpha;
        if (!(inv == inv) || inv == 0.0)                             { fb = SSH_FB_NAN;   break; }
        alpha = 1.0 / inv;
        gamma = gnew; delta = dnew;
    }
    if (fb == SSH_FB_NONE && iter > si->maxiter) fb = SSH_FB_MAXITER;

    if (fb != SSH_FB_NONE) {
        ssh_fb_announce(fb, iter, resid, partit, solve_id);
        return -1;                                   /* caller restores X0 and redoes with cg */
    }

    if (ssh_verify_on()) {
        exch(dyn->d_eta_fld);
        --g_sshwire.s_exch;                          /* verify comm is not wire-counted */
        real_t tn = 0.0;
        Kokkos::parallel_reduce("fesom_cg2_verify", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n = a; n < e; ++n) s += vals(n) * X(colind(n));
                const real_t d = rhs(row) - s;
                l += d * d;
            }, tn);
        if (parallel) MPI_Allreduce(MPI_IN_PLACE, &tn, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM);
        const double tr = sqrt((double)tn / (double)N_global);
        SshWireState &w = g_sshwire;
        if (tr > w.v_maxtrue) w.v_maxtrue = tr;
        const double gap = fabs(tr - resid);
        if (gap > w.v_maxgap) w.v_maxgap = gap;
        if (tr > (double)rtol) ++w.v_fail;
        if (partit == NULL || partit->mype == 0) {
            fprintf(stderr, "[ssh-verify] solve %ld: true=%.6e rec=%.6e rtol=%.6e gap=%.3e%s\n",
                    solve_id, tr, resid, (double)rtol, gap,
                    tr > (double)rtol ? "  <-- TRUE RESIDUAL ABOVE rtol" : "");
            fflush(stderr);
        }
    }

    dyn->d_eta_fld.modify_device();
    si->last_iters = iter;
    ssh_wire_close_solve(iter, resid, (double)rtol, partit);
    return iter;
}

/* ======================================================================= *
 * M10 T6 — `pipecg`: Ghysels / Cools–Vanroose pipelined PCG (derivations §2).
 *
 *   [post Iallreduce on γ=(r,u), δ=(w,u), ρ=(r,r)]
 *       exchange w ; m = M⁻¹w ; n = A m            ← the overlap payload
 *   [Wait]
 *   β,α as in cg2 (first iteration: β=0, α=γ/δ)
 *   z = n + βz ; q = m + βq ; s = w + βs ; p = u + βp
 *   x += αp ; r -= αs ; u -= αq ; w -= αz
 *
 * Only `w` is exchanged (m needs w's halo; n needs m's ring1, which the shipped pr rows
 * supply) — everything else is owned-only, so this is 1 exchange + 1 REDUCTION per
 * iteration, same counts as cg2 but with the reduction nonblocking.
 *
 * ⚠️ R2 — the overlap is structurally unavailable on this stack. T2's probe (26722815 /
 * 26722816) measured NO async `Iallreduce` progression under MPI_THREAD_SINGLE on either
 * production MPI, and a SURCHARGE over blocking Allreduce (+8 µs on openmpi/4.1.2,
 * +1.6–1.8 µs on 4.1.5-nvhpc). Pre-registered attribution: a null-or-negative
 * pipecg-vs-cg2 delta is STACK, not algorithm. The method is implemented faithfully anyway
 * (user decision: all four) — the honest negative is the result.
 * ======================================================================= */

struct SshPipeState {
    fesom::Field uu, ww, mm, nn, zz, qq, ss, pp2;
    int ext = 0;
};
static SshPipeState g_pipe;

static void ssh_pipe_alloc(const struct fesom_mesh *mesh, int ring_tail)
{
    const int N = mesh->myDim_nod2D, E = N + mesh->eDim_nod2D;
    const int want = E + ring_tail;                /* ww carries the cgpipe ring2 tail */
    if (g_pipe.ext == want) return;
    g_pipe.ww .alloc("ssh.pipe.ww", (size_t)want);
    g_pipe.uu .alloc("ssh.pipe.uu", (size_t)E);
    g_pipe.mm .alloc("ssh.pipe.mm", (size_t)E);
    g_pipe.nn .alloc("ssh.pipe.nn", (size_t)E);
    g_pipe.zz .alloc("ssh.pipe.zz", (size_t)E);
    g_pipe.qq .alloc("ssh.pipe.qq", (size_t)E);
    g_pipe.ss .alloc("ssh.pipe.ss", (size_t)E);
    g_pipe.pp2.alloc("ssh.pipe.pp", (size_t)E);
    g_pipe.ext = want;
}

static int ssh_solve_pipecg(const fesom_ssh_stiff *S, fesom_solverinfo *si,
                            const struct fesom_mesh *mesh, struct fesom_dyn *dyn)
{
    const int     N        = mesh->myDim_nod2D;
    fesom_partit *partit   = si->partit;
    const int     parallel = (partit && partit->npes > 1);
    const int     N_global = parallel ? mesh->nod2D : N;
    const long    solve_id = g_sshwire.solves + 1;

#ifdef KOKKOS_ENABLE_CUDA
    const bool transport_ok = fesom_halo_device_active();
#else
    const bool transport_ok = true;
#endif
    const bool ring = ssh_ring_on() && parallel && transport_ok;
    if (ssh_sympre_on()) {
        ssh_sympre_build(S, mesh, partit);
        FESOM_CHECK(!g_cgpipe.built || cgpipe_ship_pr == g_sympre.pr_h.data(),
                    "ssh-sympre: CGPIPE ring1 rows shipped with a different preconditioner");
        cgpipe_ship_pr = g_sympre.pr_h.data();
    }
    if (ring && !g_cgpipe.built) cgpipe_build(S, si, mesh, partit);
    ssh_pipe_alloc(mesh, ring ? g_cgpipe.nring2 : 0);

    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    auto vals   = S->values_fld.d();
    auto prvals = ssh_sympre_on() ? Kokkos::View<const double*>(g_sympre.pr_d)
                                  : Kokkos::View<const double*>(S->pr_values_fld.d());
    auto X   = dyn->d_eta_fld.d();
    auto rhs = dyn->ssh_rhs_fld.d();
    auto rr  = si->rr_fld.d();
    auto uu  = g_pipe.uu.d();  auto ww = g_pipe.ww.d();
    auto mm  = g_pipe.mm.d();  auto nn = g_pipe.nn.d();
    auto zz  = g_pipe.zz.d();  auto qq = g_pipe.qq.d();
    auto ss  = g_pipe.ss.d();  auto pp = g_pipe.pp2.d();

    auto exch = [&](fesom::Field &f) {
        if (!parallel) return;
        ++g_sshwire.s_exch;
#ifdef KOKKOS_ENABLE_CUDA
        if (fesom_halo_device_active()) { fesom_halo_exchange_device(f, FESOM_HALO_NOD2D, 1, 1, partit); return; }
#endif
        f.modify_device(); f.sync_host();
        fesom_halo_exchange(f.h_checked(), FESOM_HALO_NOD2D, 1, 1, partit);
        f.modify_host();   f.sync_device();
    };
    /* w's exchange + m = M⁻¹w on owned (+ring1 in the ring form) + n = A m on owned */
    auto payload = [&]() {
        if (parallel) {
            if (ring) { ++g_sshwire.s_exch; cgpipe_exchange_rr(g_pipe.ww, partit); }
            else      exch(g_pipe.ww);
        }
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pipe_psolve", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n2 = a; n2 < e; ++n2) s += prvals(n2) * ww(colind(n2));
                mm(row) = s;
            });
        if (ring) { SSH_WIRE_LAUNCH(1); cgpipe_zz_ring1(ww, mm); }
        else      { g_pipe.mm.modify_device(); exch(g_pipe.mm); }
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pipe_spmv", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n2 = a; n2 < e; ++n2) s += vals(n2) * mm(colind(n2));
                nn(row) = s;
            });
    };

    /* ---- initialisation ---- */
    SSH_WIRE_LAUNCH(1);
    real_t s0 = cg_dot(rhs, rhs, N);
    if (parallel) { MPI_Allreduce(MPI_IN_PLACE, &s0, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }
    const real_t rtol = si->soltol * sqrt(s0 / (real_t)N_global);
    if (s0 == 0.0) {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pipe_zeroX", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) { X(row) = 0.0; });
        dyn->d_eta_fld.modify_device();
        si->last_iters = 0;
        ssh_wire_close_solve(0, 0.0, (double)rtol, partit);
        return 0;
    }
    exch(dyn->d_eta_fld);
    SSH_WIRE_LAUNCH(2);
    cg_spmv(rowptr, colind, vals, X, rr, N);
    Kokkos::parallel_for("fesom_pipe_r0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) { rr(row) = rhs(row) - rr(row); });
    si->rr_fld.modify_device();
    exch(si->rr_fld);                                 /* u₀ = M⁻¹r₀ needs r's halo */
    SSH_WIRE_LAUNCH(1);
    Kokkos::parallel_for("fesom_pipe_u0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            real_t s = 0.0;
            const int a = rowptr(row), e = rowptr(row + 1);
            for (int n2 = a; n2 < e; ++n2) s += prvals(n2) * rr(colind(n2));
            uu(row) = s;
        });
    g_pipe.uu.modify_device();
    exch(g_pipe.uu);                                  /* w₀ = A u₀ needs u's halo */
    SSH_WIRE_LAUNCH(2);
    Kokkos::parallel_for("fesom_pipe_w0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            real_t s = 0.0;
            const int a = rowptr(row), e = rowptr(row + 1);
            for (int n2 = a; n2 < e; ++n2) s += vals(n2) * uu(colind(n2));
            ww(row) = s;
        });
    Kokkos::parallel_for("fesom_pipe_zero", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) { zz(row) = 0.0; qq(row) = 0.0; ss(row) = 0.0; pp(row) = 0.0; });
    g_pipe.ww.modify_device();

    double gamma_prev = 0.0, alpha_prev = 0.0, resid = 0.0;
    int fb = SSH_FB_NONE, iter = 0;
    double best = 1e300; int stall = 0;
    const int STALL_WINDOW = ssh_stall_window(20);

    for (iter = 1; iter <= si->maxiter; ++iter) {
        real_t d3[3] = { 0.0, 0.0, 0.0 };
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_reduce("fesom_pipe_dots", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int i, real_t &l0, real_t &l1, real_t &l2) {
                l0 += rr(i) * uu(i); l1 += ww(i) * uu(i); l2 += rr(i) * rr(i);
            }, d3[0], d3[1], d3[2]);

        MPI_Request req = MPI_REQUEST_NULL;
        if (parallel) {
            MPI_Iallreduce(MPI_IN_PLACE, d3, 3, MPI_DOUBLE, MPI_SUM,
                           partit->MPI_COMM_FESOM, &req);
            ++g_sshwire.s_ari;
        }
        payload();                                    /* the overlap window (R2: no progression) */
        if (parallel) MPI_Wait(&req, MPI_STATUS_IGNORE);

        const double gamma = (double)d3[0], delta = (double)d3[1];
        resid = sqrt((double)d3[2] / (double)N_global);
        if (!(gamma == gamma) || !(delta == delta) || !(resid == resid)) { fb = SSH_FB_NAN; break; }
        if (resid < rtol) break;
        if (gamma <= 0.0)                                                { fb = SSH_FB_INDEF; break; }
        if (resid < best * 0.999) { best = resid; stall = 0; }
        else if (++stall >= STALL_WINDOW || resid > 1e3 * best)          { fb = SSH_FB_STALL; break; }

        double beta = 0.0, alpha = 0.0;
        if (iter > 1) {
            beta = gamma / gamma_prev;
            const double inv = delta / gamma - beta / alpha_prev;
            if (!(inv == inv) || inv == 0.0)                             { fb = SSH_FB_NAN; break; }
            alpha = 1.0 / inv;
        } else {
            if (delta == 0.0)                                            { fb = SSH_FB_NAN; break; }
            alpha = gamma / delta;
        }
        if (ssh_trace_on() && (partit == NULL || partit->mype == 0))
            fprintf(stderr, "[ssh-trace] it=%d al=%.17g be=%.17g res=%.17g\n",
                    iter, alpha, beta, resid);

        const real_t al = (real_t)alpha, be = (real_t)beta;
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pipe_update", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int i) {
                const real_t z = nn(i) + be * zz(i);
                const real_t q = mm(i) + be * qq(i);
                const real_t s = ww(i) + be * ss(i);
                const real_t p = uu(i) + be * pp(i);
                zz(i) = z; qq(i) = q; ss(i) = s; pp(i) = p;
                X (i) += al * p;
                rr(i) -= al * s;
                uu(i) -= al * q;
                ww(i) -= al * z;
            });
        g_pipe.ww.modify_device();
        gamma_prev = gamma; alpha_prev = alpha;
    }
    if (fb == SSH_FB_NONE && iter > si->maxiter) fb = SSH_FB_MAXITER;
    if (fb != SSH_FB_NONE) { ssh_fb_announce(fb, iter, resid, partit, solve_id); return -1; }

    if (ssh_verify_on()) {
        exch(dyn->d_eta_fld);
        --g_sshwire.s_exch;
        real_t tn = 0.0;
        Kokkos::parallel_reduce("fesom_pipe_verify", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n2 = a; n2 < e; ++n2) s += vals(n2) * X(colind(n2));
                const real_t d = rhs(row) - s;
                l += d * d;
            }, tn);
        if (parallel) MPI_Allreduce(MPI_IN_PLACE, &tn, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM);
        const double tr = sqrt((double)tn / (double)N_global);
        SshWireState &w = g_sshwire;
        if (tr > w.v_maxtrue) w.v_maxtrue = tr;
        const double gap = fabs(tr - resid);
        if (gap > w.v_maxgap) w.v_maxgap = gap;
        if (tr > (double)rtol) ++w.v_fail;
        if (partit == NULL || partit->mype == 0)
            fprintf(stderr, "[ssh-verify] solve %ld: true=%.6e rec=%.6e rtol=%.6e gap=%.3e%s\n",
                    solve_id, tr, resid, (double)rtol, gap,
                    tr > (double)rtol ? "  <-- TRUE RESIDUAL ABOVE rtol" : "");
    }
    dyn->d_eta_fld.modify_device();
    si->last_iters = iter;
    ssh_wire_close_solve(iter, resid, (double)rtol, partit);
    return iter;
}

/* ======================================================================= *
 * M10 T8a — Lanczos eigen-estimator for pcsi (derivations §4.4; [P] App. C with the
 * T-5 correction). Tridiagonalises M̃⁻¹A in the M̃⁻¹ inner product; the extreme Ritz
 * values come from Sturm bisection on T_m.
 *
 * ⚠️ T-5 (measured, testbed §7): [P] prints `q₁ = r₀/(r₀ᵀs₀)`. The missing SQUARE ROOT
 * corrupts α₁ and plants a spurious tiny eigenvalue in T — θmin came out **2270× too
 * small** on the fixture, at every m. We use `q₁ = r₀/sqrt(r₀ᵀs₀)`.
 *
 * Ritz values converge OUTWARD-short (θmax ≤ λmax, θmin ≥ λmin — verified in the testbed
 * by monotonicity in m), which is exactly why the safe margins are deflate-ν / inflate-µ.
 * That guarantee holds only for a self-adjoint operator ⇒ pcsi requires SYMPRE=1 (enforced
 * in the interaction matrix).
 *
 * No reorthogonalisation: at m ≈ 30 the loss of orthogonality mainly duplicates interior
 * Ritz values, and duplicates do not move the extremes (Ritz values always lie inside
 * [λmin, λmax] by Rayleigh–Ritz). Runs ONCE, at the first pcsi solve.
 * ======================================================================= */

/* eigenvalues of T strictly below x (Sturm sequence) */
static int ssh_sturm_count(const std::vector<double> &a, const std::vector<double> &b, double x)
{
    const int m = (int)a.size();
    int cnt = 0;
    double d = a[0] - x;
    if (d < 0.0) ++cnt;
    for (int i = 1; i < m; ++i) {
        if (fabs(d) < 1e-300) d = 1e-300;
        d = (a[(size_t)i] - x) - b[(size_t)i - 1] * b[(size_t)i - 1] / d;
        if (d < 0.0) ++cnt;
    }
    return cnt;
}

static void ssh_tridiag_extremes(const std::vector<double> &a, const std::vector<double> &b,
                                 double *lo, double *hi)
{
    const int m = (int)a.size();
    double g0 = a[0], g1 = a[0];
    for (int i = 0; i < m; ++i) {
        const double bl = (i > 0)     ? b[(size_t)i - 1] : 0.0;
        const double br = (i < m - 1) ? b[(size_t)i]     : 0.0;
        g0 = fmin(g0, a[(size_t)i] - bl - br);
        g1 = fmax(g1, a[(size_t)i] + bl + br);
    }
    auto bisect = [&](int want) {
        double x0 = g0, x1 = g1;
        for (int it = 0; it < 200; ++it) {
            const double xm = 0.5 * (x0 + x1);
            if (ssh_sturm_count(a, b, xm) >= want) x1 = xm; else x0 = xm;
        }
        return 0.5 * (x0 + x1);
    };
    *lo = bisect(1);
    *hi = bisect(m);
}

struct SshPcsiState {
    bool  built = false;
    double nu = 0.0, mu = 0.0;
    fesom::Field rp, dx, tmp;
    int ext = 0;
};
static SshPcsiState g_pcsi;

static void ssh_pcsi_eig(const fesom_ssh_stiff *S, fesom_solverinfo *si,
                         const struct fesom_mesh *mesh, fesom_partit *partit)
{
    SshPcsiState &s = g_pcsi;
    if (s.built) return;
    const int N        = mesh->myDim_nod2D;
    const int Next     = N + mesh->eDim_nod2D;
    const int parallel = (partit && partit->npes > 1);

    /* explicit override wins (FESOM_PCSI_EIG="nu,mu") */
    if (const char *e = getenv("FESOM_PCSI_EIG")) {
        double a = 0.0, b = 0.0;
        FESOM_CHECK(sscanf(e, "%lf,%lf", &a, &b) == 2 && a > 0.0 && b > a,
                    "FESOM_PCSI_EIG='%s' — expected \"nu,mu\" with 0 < nu < mu", e);
        s.nu = a; s.mu = b; s.built = true;
        if (!partit || partit->mype == 0)
            fprintf(stderr, "[pcsi] eigenbounds from FESOM_PCSI_EIG: [%.6e, %.6e]\n", a, b);
        return;
    }

    /* ⚠️ DEFAULT RAISED 30 -> 120 (2026-08-06, measured). m=30 UNDER-ESTIMATES the spectrum:
     * sweeping m on real matrices gave theta_min 2.027e-03 -> 3.482e-04 on farc (kappa
     * 843 -> 4911, a 5.8x error) and 3.446e-03 -> 2.513e-03 on CORE2 (489 -> 670).
     * The farc under-estimate produced a mis-tuned Chebyshev polynomial that STALLED and
     * fired the fallback guard on ~2 % of solves; at m=120 the same configuration runs with
     * ZERO fallbacks. m=120 costs a few extra setup SpMVs, once per run.
     * Note the trade: an under-estimated kappa yields a MORE aggressive polynomial that is
     * slightly faster when it happens to converge (CORE2 -7.2 % at m=30 vs -5.4 % at m=120),
     * so the old default was not merely wrong — it was faster-but-unsafe. Correctness wins. */
    int m = 120;
    if (const char *e = getenv("FESOM_PCSI_LANCZOS")) {
        m = atoi(e);
        FESOM_CHECK(m >= 4 && m <= 500, "FESOM_PCSI_LANCZOS=%d out of range [4,500]", m);
    }
    double mg[2] = { 0.10, 0.05 };
    if (const char *e = getenv("FESOM_PCSI_EIGMARGIN"))
        FESOM_CHECK(sscanf(e, "%lf,%lf", &mg[0], &mg[1]) == 2 && mg[0] >= 0.0 && mg[1] >= 0.0,
                    "FESOM_PCSI_EIGMARGIN='%s' — expected \"deflate_nu,inflate_mu\"", e);

    /* host-side Lanczos on the ORIGINAL (host) arrays — runs once, off the hot path. */
    std::vector<real_t> vals_h((size_t)S->nnz);
    {
        auto mv = Kokkos::create_mirror_view(S->values_fld.d());
        Kokkos::deep_copy(mv, S->values_fld.d());
        for (int i = 0; i < S->nnz; ++i) vals_h[(size_t)i] = mv(i);
    }
    const real_t *PR = g_sympre.pr_h.data();          /* SYMPRE is mandatory for pcsi */
    auto spmv = [&](const real_t *A_, std::vector<real_t> &v, std::vector<real_t> &y) {
        if (parallel) fesom_halo_exchange(v.data(), FESOM_HALO_NOD2D, 1, 1, partit);
        for (int r = 0; r < N; ++r) {
            real_t acc = 0.0;
            for (int n = S->rowptr[r]; n < S->rowptr[r + 1]; ++n)
                acc += A_[n] * v[(size_t)S->colind[n]];
            y[(size_t)r] = acc;
        }
    };
    auto dot = [&](const std::vector<real_t> &x, const std::vector<real_t> &y) {
        double d = 0.0;
        for (int i = 0; i < N; ++i) d += (double)x[(size_t)i] * (double)y[(size_t)i];
        if (parallel) MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM);
        return d;
    };

    std::vector<real_t> q((size_t)Next, 0.0), qp((size_t)Next, 0.0), r((size_t)Next, 0.0),
                        p((size_t)Next, 0.0), t((size_t)Next, 0.0);
    /* deterministic, rank-independent start vector: a function of the GLOBAL node id, so
     * every partitioning produces the SAME Krylov space (bit-reproducible bounds). */
    for (int i = 0; i < N; ++i) {
        const int gid = partit ? partit->myList_nod2D[i] : i + 1;
        r[(size_t)i] = 1.0 + 0.37 * sin(0.7 * (double)gid);
    }
    spmv(PR, r, p);
    const double n0 = dot(r, p);
    FESOM_CHECK(n0 > 0.0, "pcsi Lanczos: r0'M^-1 r0 = %g <= 0 — preconditioner not SPD", n0);
    const double scale = sqrt(n0);                     /* ⭐ T-5: the SQUARE ROOT */
    for (int i = 0; i < N; ++i) q[(size_t)i] = r[(size_t)i] / scale;

    std::vector<double> al, be;
    double beta_prev = 0.0;
    for (int j = 0; j < m; ++j) {
        spmv(PR, q, p);                                /* p = M⁻¹ q   */
        spmv(vals_h.data(), p, r);                     /* r = A M⁻¹ q */
        for (int i = 0; i < N; ++i) r[(size_t)i] -= (real_t)beta_prev * qp[(size_t)i];
        const double a = dot(p, r);
        al.push_back(a);
        for (int i = 0; i < N; ++i) r[(size_t)i] -= (real_t)a * q[(size_t)i];
        spmv(PR, r, t);
        const double bb = dot(r, t);
        if (!(bb > 0.0)) break;
        const double b = sqrt(bb);
        be.push_back(b);
        qp = q;
        for (int i = 0; i < N; ++i) q[(size_t)i] = r[(size_t)i] / (real_t)b;
        beta_prev = b;
    }
    FESOM_CHECK(al.size() >= 2, "pcsi Lanczos broke down after %zu steps", al.size());
    if (be.size() >= al.size()) be.resize(al.size() - 1);

    double th_lo = 0.0, th_hi = 0.0;
    ssh_tridiag_extremes(al, be, &th_lo, &th_hi);
    FESOM_CHECK(th_lo > 0.0 && th_hi > th_lo,
                "pcsi Lanczos: nonsensical Ritz interval [%g, %g]", th_lo, th_hi);
    /* SAFE directions (Ritz converge outward-short): deflate ν, inflate µ */
    s.nu = th_lo * (1.0 - mg[0]);
    s.mu = th_hi * (1.0 + mg[1]);

    /* R6-class rank-agreement assertion: the bounds come from allreduced dots, so they must
     * already be bitwise identical on every rank. Prove it (MIN vs MAX) rather than assume —
     * a silent per-rank ω divergence would corrupt the solve invisibly. */
    if (parallel) {
        double lo2[2] = { s.nu, s.mu }, hi2[2] = { s.nu, s.mu };
        MPI_Allreduce(MPI_IN_PLACE, lo2, 2, MPI_DOUBLE, MPI_MIN, partit->MPI_COMM_FESOM);
        MPI_Allreduce(MPI_IN_PLACE, hi2, 2, MPI_DOUBLE, MPI_MAX, partit->MPI_COMM_FESOM);
        FESOM_CHECK(lo2[0] == hi2[0] && lo2[1] == hi2[1],
                    "pcsi: eigenbounds DIFFER across ranks (nu %.17g..%.17g, mu %.17g..%.17g) "
                    "— every rank must compute the same omega sequence", lo2[0], hi2[0], lo2[1], hi2[1]);
    }
    s.built = true;
    /* Suitability check. Chebyshev needs ~0.5*sqrt(kappa)*ln(2/tol) iterations; when that
     * exceeds what the baseline CG would take, pcsi is the wrong tool for this system and
     * the user should know BEFORE spending a run on it. Measured: farc kappa 4911 => ~428
     * predicted vs CG's 212 (pcsi indeed lost); CORE2 kappa 670 => ~158 vs CG's 106 (pcsi
     * won on wall-clock because its iterations are far cheaper). So this is a WARNING, not
     * an abort — cheap iterations can still win at a moderate iteration penalty. */
    const double pred = 0.5 * sqrt(s.mu / s.nu) * log(2.0 / (double)si->soltol);
    if (!partit || partit->mype == 0) {
        fprintf(stderr, "[pcsi] Lanczos m=%zu on M~^-1 A: theta = [%.6e, %.6e] -> "
                        "[nu,mu] = [%.6e, %.6e] (margins %.2f/%.2f), kappa = %.1f\n",
                al.size(), th_lo, th_hi, s.nu, s.mu, mg[0], mg[1], s.mu / s.nu);
        fprintf(stderr, "[pcsi] predicted Chebyshev iterations ~%.0f (maxiter %d)%s\n",
                pred, si->maxiter,
                pred > 0.6 * (double)si->maxiter
                    ? "  <-- ILL-CONDITIONED: pcsi is likely a poor choice here; expect many "
                      "iterations and check for fallback firings" : "");
        fflush(stderr);
    }
}

/* ======================================================================= *
 * M10 T8b — `pcsi`: preconditioned Chebyshev (classical Stiefel) iteration.
 *
 *   γ = (µ+ν)/2 ; α = 2/(µ−ν) ; ω₀ = 2/γ
 *   Δx₀ = (1/γ)M⁻¹r₀ ; x₁ = x₀ + Δx₀ ; r₁ = b − Ax₁
 *   ω_k = 1/(γ − ω_{k-1}/(4α²))            ← T-6 RESOLVED coefficient
 *   Δx_k = ω_k M⁻¹r_k + (γω_k − 1)Δx_{k-1}
 *   x_{k+1} = x_k + Δx_k ; r_{k+1} = b − Ax_{k+1}   ← TRUE residual, self-correcting
 *
 * 1 exchange, ZERO reductions per iteration; one reduction every FESOM_PCSI_CHECK
 * iterations for the convergence test — and because the residual is the true one by
 * construction, that test reads a residual that needs no verification (the pipelined
 * attainable-accuracy failure mode does not exist here).
 *
 * Ring composition: exchange r on 2 rings ⇒ M⁻¹r on owned+ring1 ⇒ Δx and x carry ring1 by
 * recurrence ⇒ Ax at owned rows sees a current ring1 x with no second message.
 * ======================================================================= */

static void ssh_pcsi_alloc(const struct fesom_mesh *mesh)
{
    const int E = mesh->myDim_nod2D + mesh->eDim_nod2D;
    if (g_pcsi.ext == E) return;
    g_pcsi.rp .alloc("ssh.pcsi.rp",  (size_t)E);
    g_pcsi.dx .alloc("ssh.pcsi.dx",  (size_t)E);
    g_pcsi.tmp.alloc("ssh.pcsi.tmp", (size_t)E);
    g_pcsi.ext = E;
}

static int ssh_solve_pcsi(const fesom_ssh_stiff *S, fesom_solverinfo *si,
                          const struct fesom_mesh *mesh, struct fesom_dyn *dyn)
{
    const int     N        = mesh->myDim_nod2D;
    fesom_partit *partit   = si->partit;
    const int     parallel = (partit && partit->npes > 1);
    const int     N_global = parallel ? mesh->nod2D : N;
    const long    solve_id = g_sshwire.solves + 1;

#ifdef KOKKOS_ENABLE_CUDA
    const bool transport_ok = fesom_halo_device_active();
#else
    const bool transport_ok = true;
#endif
    const bool ring = ssh_ring_on() && parallel && transport_ok;
    ssh_sympre_build(S, mesh, partit);                 /* mandatory for pcsi (checked in T5a) */
    FESOM_CHECK(!g_cgpipe.built || cgpipe_ship_pr == g_sympre.pr_h.data(),
                "ssh-sympre: CGPIPE ring1 rows shipped with a different preconditioner");
    cgpipe_ship_pr = g_sympre.pr_h.data();
    if (ring && !g_cgpipe.built) cgpipe_build(S, si, mesh, partit);
    ssh_pcsi_alloc(mesh);
    ssh_pcsi_eig(S, si, mesh, partit);

    int K = 5;
    if (const char *e = getenv("FESOM_PCSI_CHECK")) {
        K = atoi(e);
        FESOM_CHECK(K >= 1 && K <= 1000, "FESOM_PCSI_CHECK=%d out of range [1,1000]", K);
    }
    int maxit = si->maxiter;
    if (const char *e = getenv("FESOM_PCSI_MAXITER")) {
        maxit = atoi(e);
        FESOM_CHECK(maxit >= 1, "FESOM_PCSI_MAXITER=%d must be positive", maxit);
    }

    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    auto vals   = S->values_fld.d();
    auto prvals = Kokkos::View<const double*>(g_sympre.pr_d);
    auto X   = dyn->d_eta_fld.d();
    auto rhs = dyn->ssh_rhs_fld.d();
    auto rr  = si->rr_fld.d();
    auto rp  = g_pcsi.rp.d();
    auto dx  = g_pcsi.dx.d();

    auto exch = [&](fesom::Field &f) {
        if (!parallel) return;
        ++g_sshwire.s_exch;
#ifdef KOKKOS_ENABLE_CUDA
        if (fesom_halo_device_active()) { fesom_halo_exchange_device(f, FESOM_HALO_NOD2D, 1, 1, partit); return; }
#endif
        f.modify_device(); f.sync_host();
        fesom_halo_exchange(f.h_checked(), FESOM_HALO_NOD2D, 1, 1, partit);
        f.modify_host();   f.sync_device();
    };
    auto exch_r = [&]() {
        if (!parallel) return;
        if (ring) { ++g_sshwire.s_exch; cgpipe_exchange_rr(si->rr_fld, partit); }
        else      exch(si->rr_fld);
    };
    /* rp = M̃⁻¹ r on owned (+ring1 in the ring form) */
    auto apply_M = [&]() {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pcsi_psolve", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n = a; n < e; ++n) s += prvals(n) * rr(colind(n));
                rp(row) = s;
            });
        if (ring) { SSH_WIRE_LAUNCH(1); cgpipe_zz_ring1(rr, rp); }
        else      { g_pcsi.rp.modify_device(); exch(g_pcsi.rp); }
    };
    /* r = b − A x over owned rows; x must be halo-current */
    auto true_resid = [&]() {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pcsi_resid", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n = a; n < e; ++n) s += vals(n) * X(colind(n));
                rr(row) = rhs(row) - s;
            });
        si->rr_fld.modify_device();
    };
    auto resid_norm = [&]() {
        real_t t = 0.0;
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_reduce("fesom_pcsi_rnorm", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int i, real_t &l) { l += rr(i) * rr(i); }, t);
        if (parallel) { MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }
        return sqrt((double)t / (double)N_global);
    };

    SSH_WIRE_LAUNCH(1);
    real_t s0 = cg_dot(rhs, rhs, N);
    if (parallel) { MPI_Allreduce(MPI_IN_PLACE, &s0, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }
    const real_t rtol = si->soltol * sqrt(s0 / (real_t)N_global);
    if (s0 == 0.0) {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pcsi_zeroX", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) { X(row) = 0.0; });
        dyn->d_eta_fld.modify_device();
        si->last_iters = 0;
        ssh_wire_close_solve(0, 0.0, (double)rtol, partit);
        return 0;
    }

    const double gmm = 0.5 * (g_pcsi.mu + g_pcsi.nu);
    const double alp = 2.0 / (g_pcsi.mu - g_pcsi.nu);
    const double c4  = 1.0 / (4.0 * alp * alp);        /* ⭐ T-6: 1/(4α²), NOT (1/4)α² */
    double omega = 2.0 / gmm;

    exch(dyn->d_eta_fld);
    true_resid();
    exch_r();
    apply_M();
    {   /* Δx₀ = (1/γ)M⁻¹r₀ ; x₁ = x₀ + Δx₀ — over owned+ring1 so x keeps its halo */
        const real_t ig = (real_t)(1.0 / gmm);
        const int E = ring ? N + mesh->eDim_nod2D : N;
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_pcsi_dx0", Kokkos::RangePolicy<>(0, E),
            KOKKOS_LAMBDA(const int i) { const real_t d = ig * rp(i); dx(i) = d; X(i) += d; });
        if (!ring) { dyn->d_eta_fld.modify_device(); exch(dyn->d_eta_fld); }
    }
    true_resid();
    double resid = resid_norm();

    int fb = SSH_FB_NONE, iter = 0;
    double best = resid; int stall = 0;
    const int STALL_WINDOW = ssh_stall_window(10);      /* in CHECK events, not iterations */
    if (resid >= rtol)
    for (iter = 1; iter <= maxit; ++iter) {
        omega = 1.0 / (gmm - c4 * omega);
        exch_r();
        apply_M();
        {
            const real_t w = (real_t)omega, g = (real_t)(gmm * omega - 1.0);
            const int E = ring ? N + mesh->eDim_nod2D : N;
            SSH_WIRE_LAUNCH(1);
            Kokkos::parallel_for("fesom_pcsi_step", Kokkos::RangePolicy<>(0, E),
                KOKKOS_LAMBDA(const int i) { const real_t d = w * rp(i) + g * dx(i);
                                             dx(i) = d; X(i) += d; });
            if (!ring) { dyn->d_eta_fld.modify_device(); exch(dyn->d_eta_fld); }
        }
        true_resid();
        if (iter % K == 0 || iter == maxit) {
            resid = resid_norm();
            if (ssh_trace_on() && (partit == NULL || partit->mype == 0))
                fprintf(stderr, "[ssh-trace] it=%d omega=%.17g res=%.17g\n", iter, omega, resid);
            if (!(resid == resid))                        { fb = SSH_FB_NAN;   break; }
            if (resid < rtol) break;
            if (resid < best * 0.999) { best = resid; stall = 0; }
            else if (++stall >= STALL_WINDOW || resid > 1e3 * best) { fb = SSH_FB_STALL; break; }
        }
    }
    if (fb == SSH_FB_NONE && iter > maxit) fb = SSH_FB_MAXITER;
    if (fb != SSH_FB_NONE) { ssh_fb_announce(fb, iter, resid, partit, solve_id); return -1; }

    if (ssh_verify_on() && (partit == NULL || partit->mype == 0))
        fprintf(stderr, "[ssh-verify] solve %ld: true=%.6e rec=%.6e rtol=%.6e gap=%.3e "
                        "(pcsi recurs the TRUE residual — gap is 0 by construction)\n",
                solve_id, resid, resid, (double)rtol, 0.0);
    dyn->d_eta_fld.modify_device();
    si->last_iters = iter;
    ssh_wire_close_solve(iter, resid, (double)rtol, partit);
    return iter;
}

/* ======================================================================= *
 * M10 T7 — `oati`: one allreduce per TWO iterations (derivations §3, incl. §3.2b).
 *
 * We implement the SHALLOW form, not [T]'s deep `n→g→h→e→f` chain. [T]'s chain exists to
 * overlap one Iallreduce with two full iterations of operator work; R2 measured that this
 * stack has NO async progression, so that overlap buys nothing here while costing 4 chained
 * operator applications = 4 rings (our M⁻¹ is sparse; [T] used Jacobi, which needs no halo).
 * The shallow form keeps OATI's actual prize — half the syncs — at cg2's operator and halo
 * cost, and needs no new ring machinery.
 *
 * Per PAIR (j, j+1): 2 exchanges, 2 M⁻¹, 2 A, ONE 10-element reduction.
 *   γ_{j+1}, δ_{j+1} come from recurrences in scalars already reduced, so the odd iteration
 *   needs no communication at all:
 *     γ_{j+1} = γ_j − 2α_j[δ_j + β_jΛsu] + α_j²[Λwm + 2β_jΛwq + β_j²Λsq]
 *     δ_{j+1} = δ_j − 2α_j[Λwm + β_jΛwq]  + α_j²[Λnm + 2β_jΛnq + β_j²Λzq]
 *   ⚠️ Both fold pairs of DIFFERENT inner products that are equal only for a symmetric M⁻¹
 *   ((r,q)=(s,u) and (z,u)=(w,q)) — T-1/T-3. FESOM_SSH_SYMPRE=1 is therefore not optional
 *   for oati either.
 *
 * The 7 Λ's + γ,δ,ρ = 10 reduced scalars — independently the same width as [T]'s λ0…λ9.
 * Convergence is tested once per pair (on the reduced ρ), which costs at most one extra
 * iteration and avoids five further recurrence dots.
 * ======================================================================= */

static int ssh_solve_oati(const fesom_ssh_stiff *S, fesom_solverinfo *si,
                          const struct fesom_mesh *mesh, struct fesom_dyn *dyn)
{
    const int     N        = mesh->myDim_nod2D;
    fesom_partit *partit   = si->partit;
    const int     parallel = (partit && partit->npes > 1);
    const int     N_global = parallel ? mesh->nod2D : N;
    const long    solve_id = g_sshwire.solves + 1;

#ifdef KOKKOS_ENABLE_CUDA
    const bool transport_ok = fesom_halo_device_active();
#else
    const bool transport_ok = true;
#endif
    const bool ring = ssh_ring_on() && parallel && transport_ok;
    if (ssh_sympre_on()) {
        ssh_sympre_build(S, mesh, partit);
        FESOM_CHECK(!g_cgpipe.built || cgpipe_ship_pr == g_sympre.pr_h.data(),
                    "ssh-sympre: CGPIPE ring1 rows shipped with a different preconditioner");
        cgpipe_ship_pr = g_sympre.pr_h.data();
    }
    if (ring && !g_cgpipe.built) cgpipe_build(S, si, mesh, partit);
    ssh_pipe_alloc(mesh, ring ? g_cgpipe.nring2 : 0);   /* same vector set as pipecg */

    auto rowptr = S->rowptr_fld.d();
    auto colind = S->colind_fld.d();
    auto vals   = S->values_fld.d();
    auto prvals = ssh_sympre_on() ? Kokkos::View<const double*>(g_sympre.pr_d)
                                  : Kokkos::View<const double*>(S->pr_values_fld.d());
    auto X   = dyn->d_eta_fld.d();
    auto rhs = dyn->ssh_rhs_fld.d();
    auto rr  = si->rr_fld.d();
    auto uu  = g_pipe.uu.d();  auto ww = g_pipe.ww.d();
    auto mm  = g_pipe.mm.d();  auto nn = g_pipe.nn.d();
    auto zz  = g_pipe.zz.d();  auto qq = g_pipe.qq.d();
    auto ss  = g_pipe.ss.d();  auto pp = g_pipe.pp2.d();

    auto exch = [&](fesom::Field &f) {
        if (!parallel) return;
        ++g_sshwire.s_exch;
#ifdef KOKKOS_ENABLE_CUDA
        if (fesom_halo_device_active()) { fesom_halo_exchange_device(f, FESOM_HALO_NOD2D, 1, 1, partit); return; }
#endif
        f.modify_device(); f.sync_host();
        fesom_halo_exchange(f.h_checked(), FESOM_HALO_NOD2D, 1, 1, partit);
        f.modify_host();   f.sync_device();
    };
    auto payload = [&]() {                 /* exchange w; m = M⁻¹w; n = A m */
        if (parallel) {
            if (ring) { ++g_sshwire.s_exch; cgpipe_exchange_rr(g_pipe.ww, partit); }
            else      exch(g_pipe.ww);
        }
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_oati_psolve", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n2 = a; n2 < e; ++n2) s += prvals(n2) * ww(colind(n2));
                mm(row) = s;
            });
        if (ring) { SSH_WIRE_LAUNCH(1); cgpipe_zz_ring1(ww, mm); }
        else      { g_pipe.mm.modify_device(); exch(g_pipe.mm); }
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_oati_spmv", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n2 = a; n2 < e; ++n2) s += vals(n2) * mm(colind(n2));
                nn(row) = s;
            });
    };
    /* the ONE reduction: γ, δ, ρ and the seven Λ's, all from the CURRENT vector state */
    double L[10];
    auto reduce10 = [&]() {
        real_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0,a8=0,a9=0;
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_reduce("fesom_oati_dots", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int i, real_t &g_, real_t &d_, real_t &r_,
                          real_t &su, real_t &wm, real_t &wq, real_t &sq,
                          real_t &nm, real_t &nq, real_t &zq) {
                const real_t r_i = rr(i), u_i = uu(i), w_i = ww(i);
                const real_t m_i = mm(i), n_i = nn(i);
                const real_t s_p = ss(i), q_p = qq(i), z_p = zz(i);   /* previous index */
                g_ += r_i * u_i;  d_ += w_i * u_i;  r_ += r_i * r_i;
                su += s_p * u_i;  wm += w_i * m_i;  wq += w_i * q_p;  sq += s_p * q_p;
                nm += n_i * m_i;  nq += n_i * q_p;  zq += z_p * q_p;
            }, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9);
        L[0]=a0; L[1]=a1; L[2]=a2; L[3]=a3; L[4]=a4;
        L[5]=a5; L[6]=a6; L[7]=a7; L[8]=a8; L[9]=a9;
        if (parallel) { MPI_Allreduce(MPI_IN_PLACE, L, 10, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }
    };
    /* one half-iteration: recurrences with (α,β) then the four state updates */
    auto half_step = [&](double alpha, double beta) {
        const real_t al = (real_t)alpha, be = (real_t)beta;
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_oati_step", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int i) {
                const real_t z = nn(i) + be * zz(i);
                const real_t q = mm(i) + be * qq(i);
                const real_t s = ww(i) + be * ss(i);
                const real_t p = uu(i) + be * pp(i);
                zz(i)=z; qq(i)=q; ss(i)=s; pp(i)=p;
                X (i) += al * p;
                rr(i) -= al * s;
                uu(i) -= al * q;
                ww(i) -= al * z;
            });
        g_pipe.ww.modify_device();
    };

    SSH_WIRE_LAUNCH(1);
    real_t s0 = cg_dot(rhs, rhs, N);
    if (parallel) { MPI_Allreduce(MPI_IN_PLACE, &s0, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); ++g_sshwire.s_arb; }
    const real_t rtol = si->soltol * sqrt(s0 / (real_t)N_global);
    if (s0 == 0.0) {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_oati_zeroX", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) { X(row) = 0.0; });
        dyn->d_eta_fld.modify_device();
        si->last_iters = 0;
        ssh_wire_close_solve(0, 0.0, (double)rtol, partit);
        return 0;
    }

    /* r₀ = b − Ax₀ ; u₀ = M⁻¹r₀ ; w₀ = Au₀ ; z,q,s,p = 0 (so the index-0 Λ's vanish) */
    exch(dyn->d_eta_fld);
    SSH_WIRE_LAUNCH(2);
    cg_spmv(rowptr, colind, vals, X, rr, N);
    Kokkos::parallel_for("fesom_oati_r0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) { rr(row) = rhs(row) - rr(row); });
    si->rr_fld.modify_device();
    exch(si->rr_fld);
    SSH_WIRE_LAUNCH(1);
    Kokkos::parallel_for("fesom_oati_u0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            real_t s = 0.0;
            const int a = rowptr(row), e = rowptr(row + 1);
            for (int n2 = a; n2 < e; ++n2) s += prvals(n2) * rr(colind(n2));
            uu(row) = s;
        });
    g_pipe.uu.modify_device();
    exch(g_pipe.uu);
    SSH_WIRE_LAUNCH(2);
    Kokkos::parallel_for("fesom_oati_w0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) {
            real_t s = 0.0;
            const int a = rowptr(row), e = rowptr(row + 1);
            for (int n2 = a; n2 < e; ++n2) s += vals(n2) * uu(colind(n2));
            ww(row) = s;
        });
    Kokkos::parallel_for("fesom_oati_zero", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int i) { zz(i)=0.0; qq(i)=0.0; ss(i)=0.0; pp(i)=0.0; });
    g_pipe.ww.modify_device();
    payload();                                  /* m₀, n₀ */
    reduce10();

    double gamma = L[0], delta = L[1], resid = sqrt(L[2] / (double)N_global);
    double gamma_prev = 0.0, alpha_prev = 0.0;
    int fb = SSH_FB_NONE, iter = 0;
    double best = resid; int stall = 0;
    const int STALL_WINDOW = ssh_stall_window(10);  /* in PAIRS */

    if (resid >= rtol)
    while (iter < si->maxiter) {
        /* ---- even half: (α_j, β_j) from the reduced γ_j, δ_j ---- */
        double beta = 0.0, alpha = 0.0;
        if (iter > 0) {
            beta = gamma / gamma_prev;
            const double inv = delta / gamma - beta / alpha_prev;
            if (!(inv == inv) || inv == 0.0) { fb = SSH_FB_NAN; break; }
            alpha = 1.0 / inv;
        } else {
            if (delta == 0.0) { fb = SSH_FB_NAN; break; }
            alpha = gamma / delta;
        }
        if (ssh_trace_on() && (partit == NULL || partit->mype == 0))
            fprintf(stderr, "[ssh-trace] it=%d al=%.17g be=%.17g res=%.17g\n",
                    iter + 1, alpha, beta, resid);
        /* γ_{j+1}, δ_{j+1} by recurrence — NO communication (the point of the method) */
        const double su = L[3], wm = L[4], wq = L[5], sq = L[6],
                     nm = L[7], nq = L[8], zq = L[9];
        const double s_u = delta + beta * su;                       /* (s_j, u_j) */
        const double w_q = wm + beta * wq;                          /* (w_j, q_j) */
        const double s_q = wm + 2.0 * beta * wq + beta * beta * sq; /* (s_j, q_j) */
        const double z_q = nm + 2.0 * beta * nq + beta * beta * zq; /* (z_j, q_j) */
        const double gamma1 = gamma - 2.0 * alpha * s_u + alpha * alpha * s_q;
        const double delta1 = delta - 2.0 * alpha * w_q + alpha * alpha * z_q;

        half_step(alpha, beta);                 /* → index j+1 */
        ++iter;
        if (!(gamma1 == gamma1) || !(delta1 == delta1) || gamma1 <= 0.0) { fb = SSH_FB_INDEF; break; }

        /* ---- odd half: (α_{j+1}, β_{j+1}) from the RECURRED scalars ---- */
        const double beta1 = gamma1 / gamma;
        const double inv1  = delta1 / gamma1 - beta1 / alpha;
        if (!(inv1 == inv1) || inv1 == 0.0) { fb = SSH_FB_NAN; break; }
        const double alpha1 = 1.0 / inv1;
        payload();                              /* m_{j+1}, n_{j+1} — 1 exchange */
        if (ssh_trace_on() && (partit == NULL || partit->mype == 0))
            fprintf(stderr, "[ssh-trace] it=%d al=%.17g be=%.17g res=(recurred)\n",
                    iter + 1, alpha1, beta1);
        half_step(alpha1, beta1);               /* → index j+2 */
        ++iter;
        payload();                              /* m_{j+2}, n_{j+2} — 1 exchange */
        reduce10();                             /* the pair's ONLY reduction */

        gamma_prev = gamma1; alpha_prev = alpha1;
        gamma = L[0]; delta = L[1];
        resid = sqrt(L[2] / (double)N_global);
        if (!(resid == resid))                                 { fb = SSH_FB_NAN;   break; }
        if (resid < rtol) break;
        if (gamma <= 0.0)                                      { fb = SSH_FB_INDEF; break; }
        if (resid < best * 0.999) { best = resid; stall = 0; }
        else if (++stall >= STALL_WINDOW || resid > 1e3 * best) { fb = SSH_FB_STALL; break; }
    }
    if (fb == SSH_FB_NONE && iter >= si->maxiter && resid >= rtol) fb = SSH_FB_MAXITER;
    if (fb != SSH_FB_NONE) { ssh_fb_announce(fb, iter, resid, partit, solve_id); return -1; }

    if (ssh_verify_on()) {
        exch(dyn->d_eta_fld);
        --g_sshwire.s_exch;
        real_t tn = 0.0;
        Kokkos::parallel_reduce("fesom_oati_verify", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l) {
                real_t s = 0.0;
                const int a = rowptr(row), e = rowptr(row + 1);
                for (int n2 = a; n2 < e; ++n2) s += vals(n2) * X(colind(n2));
                const real_t d = rhs(row) - s;
                l += d * d;
            }, tn);
        if (parallel) MPI_Allreduce(MPI_IN_PLACE, &tn, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM);
        const double tr = sqrt((double)tn / (double)N_global);
        SshWireState &w = g_sshwire;
        if (tr > w.v_maxtrue) w.v_maxtrue = tr;
        const double gap = fabs(tr - resid);
        if (gap > w.v_maxgap) w.v_maxgap = gap;
        if (tr > (double)rtol) ++w.v_fail;
        if (partit == NULL || partit->mype == 0)
            fprintf(stderr, "[ssh-verify] solve %ld: true=%.6e rec=%.6e rtol=%.6e gap=%.3e%s\n",
                    solve_id, tr, resid, (double)rtol, gap,
                    tr > (double)rtol ? "  <-- TRUE RESIDUAL ABOVE rtol" : "");
    }
    dyn->d_eta_fld.modify_device();
    si->last_iters = iter;
    ssh_wire_close_solve(iter, resid, (double)rtol, partit);
    return iter;
}

static int ssh_solve_variant(int kind, const fesom_ssh_stiff *S, fesom_solverinfo *si,
                             const struct fesom_mesh *mesh, struct fesom_dyn *dyn)
{
    const int  N        = mesh->myDim_nod2D;
    const bool armed    = ssh_fallback_armed();
    /* X0 snapshot at solve entry (R6): the failed solver has already written X, and a NaN X
     * would poison the baseline restart. Host-side, one vector, only for non-cg solvers. */
    std::vector<real_t> X0;
    if (armed) {
        dyn->d_eta_fld.sync_host();
        X0.assign(dyn->d_eta, dyn->d_eta + N);
    }

    int rc = -1;
    switch (kind) {
        case FESOM_SSHSOLV_CG2:    rc = ssh_solve_cg2(S, si, mesh, dyn);    break;
        case FESOM_SSHSOLV_PIPECG: rc = ssh_solve_pipecg(S, si, mesh, dyn); break;
        case FESOM_SSHSOLV_PCSI:   rc = ssh_solve_pcsi(S, si, mesh, dyn);   break;
        case FESOM_SSHSOLV_OATI:   rc = ssh_solve_oati(S, si, mesh, dyn);   break;
        default:
            FESOM_DIE("FESOM_SSH_SOLVER=%s is not implemented yet (M10 tasks T6-T8b)",
                      ssh_solver_name(kind));
    }
    if (rc >= 0) return rc;

    FESOM_CHECK(armed, "FESOM_SSH_SOLVER=%s tripped its guard with FESOM_SSH_FALLBACK=0 — "
                       "no X0 snapshot exists, so the solve cannot be redone safely",
                ssh_solver_name(kind));
    /* restore X0 and let the caller run baseline cg on the SAME inputs */
    memcpy(dyn->d_eta, X0.data(), (size_t)N * sizeof(real_t));
    dyn->d_eta_fld.modify_host();
    dyn->d_eta_fld.sync_device();
    return -1;
}

/* Release every M10 persistent View before Kokkos::finalize (the cgpipe_free pattern). */
void fesom_ssh_m10_free(void)
{
    g_sympre.pr_d = Kokkos::View<double*>();
    g_sympre.pr_h.clear();
    g_sympre.diag_h.clear();
    g_sympre.built = false;
    g_cg2  = SshCg2State{};
    g_pipe = SshPipeState{};
    g_pcsi = SshPcsiState{};
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

    /* M10 [ssh-wire]: per-solve counter reset (fold + print in ssh_wire_close_solve). */
    g_sshwire.s_exch = 0; g_sshwire.s_arb = 0; g_sshwire.s_ari = 0; g_sshwire.s_launch = 0;

    /* M10 T5a — dispatch. `cg` (default/unset) falls straight through to the certified body
     * below: one integer compare, nothing else changes, so knob-OFF stays byte-identical.
     * A variant that trips its guard restores the solve-entry X0 itself and returns -1; we
     * then fall through and redo THIS solve with baseline cg (the armed fallback, R6). */
    {
        const int kind = ssh_solver_kind();
        if (kind != FESOM_SSHSOLV_CG) {
            ssh_solver_check_interactions(kind, parallel ? partit->npes : 1);
            /* FESOM_CG_PROFILE must cover the VARIANTS too, otherwise the SSH-phase share is
             * reported only for baseline cg and a variant run looks like it has no solver
             * time at all. Same shape as the baseline timer below (2 fences per SOLVE, not
             * per iteration — ~18 µs against a ms-scale solve, so it does not bias the A/B).
             * The baseline path keeps its own timer; this one covers only the variant branch,
             * so nothing is double-counted. */
            static const bool var_prof = (getenv("FESOM_CG_PROFILE") != NULL);
            double _v_t0 = 0.0;
            if (var_prof) { Kokkos::fence(); _v_t0 = MPI_Wtime(); }
            const int rc = ssh_solve_variant(kind, S, si, mesh, dyn);
            if (rc >= 0) {
                if (var_prof) {
                    Kokkos::fence();
                    g_fesom_cg_wall  += MPI_Wtime() - _v_t0;
                    g_fesom_cg_iters += rc;
                }
                return rc;
            }
            /* guard tripped: the variant's partial time still belongs to the SSH phase, and
             * the baseline retry below adds its own — the sum is the honest cost of the
             * fallback, which is what a run with firings should report. */
            if (var_prof) { Kokkos::fence(); g_fesom_cg_wall += MPI_Wtime() - _v_t0; }
            g_sshwire.s_exch = 0; g_sshwire.s_arb = 0;
            g_sshwire.s_ari = 0;  g_sshwire.s_launch = 0;
        }
    }

    /* M7 E.CG1 — FESOM_SPEED_CGPIPE (see the banner above cg_dot). ADOPTED into
     * the FESOM_SPEED=1 blessed set 2026-07-16 (user decision) after the full
     * ladder: FORCE_SERIAL byte proof (bit-identical to the certified baseline),
     * 2x2627 selfchecks all 0.000e+00 (Serial AND CUDA), options x3, A/B
     * -1.41% @4N / -8.07% @16N (jobs 26288442/43). Per-lever override
     * FESOM_SPEED_CGPIPE=0 still forces it off under the master.
     * Resolve FIRST (announces itself, L80), then the activity conjunction; a
     * requested-but-inactive knob warns loudly instead of dying silent. The
     * one-time setup re-allocs rr_fld (ring2 tail) so it MUST run before the
     * device views are taken below. */
    static int s_cgpipe = -1;
    const bool cgpipe_env = fesom_speed_on("CGPIPE", &s_cgpipe);
#ifdef KOKKOS_ENABLE_CUDA
    const bool transport_ok = fesom_halo_device_active();   /* keep the debug toggle coherent */
#else
    const bool transport_ok = true;   /* Serial: host Views + host MPI (the FORCE_SERIAL proof) */
#endif
    /* M7 E.CG2 — FESOM_SPEED_CGPOLY=<d> (opt-in value knob; the master never
     * implies it; fesom_speed_int is SILENT so cgpoly_build announces, L80).
     * When active it SUPERSEDES the CGPIPE path (its R-ring machinery is the
     * cgpipe graph generalized; pr_values are not used at all). npes==1 is
     * allowed (rings empty — login smokes exercise the cheb math). */
    static int s_cgpoly = -2;
    const int  cgpoly_d = fesom_speed_int("CGPOLY", 0, &s_cgpoly);
    const bool cgpoly   = cgpoly_d >= 1 && transport_ok;
    if (cgpoly_d >= 1 && !cgpoly) {
        static bool warned_poly = false;
        if (!warned_poly && (!partit || partit->mype == 0)) {
            fprintf(stderr, "[cgpoly] !! FESOM_SPEED_CGPOLY requested but INACTIVE "
                            "(FESOM_HOST_HALO=1?) — running the MITgcm preconditioner\n");
            fflush(stderr);
        }
        warned_poly = true;
    }
    if (cgpoly) {
        const char *v = getenv("FESOM_KK_VERIFY");
        FESOM_CHECK(!(v && strcmp(v, "ssh") == 0),
                    "cgpoly: FESOM_KK_VERIFY=ssh is incompatible with FESOM_SPEED_CGPOLY "
                    "(the C twin runs the legacy solver; the iterates differ within tolerance)");
    }
    const bool cgpipe = cgpipe_env && parallel && transport_ok && !cgpoly;
    if (cgpipe_env && !cgpipe && !cgpoly) {
        static bool warned = false;
        if (!warned && (!partit || partit->mype == 0)) {
            fprintf(stderr, "[cgpipe] !! FESOM_SPEED_CGPIPE requested but INACTIVE "
                            "(npes==1 or FESOM_HOST_HALO=1) — running the 2-exchange CG\n");
            fflush(stderr);
        }
        warned = true;
    }
    if (cgpoly && !g_cgpoly.built) {
        static double kap = -1.0;
        if (kap < 0.0) {
            const char *e = getenv("FESOM_CGPOLY_KAPPA");
            kap = e ? atof(e) : 30.0;
            if (kap <= 1.0) kap = 30.0;                 /* λmin < λmax required */
        }
        cgpoly_build(S, si, mesh, partit, cgpoly_d, kap);
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
    auto exch = [&](fesom::Field &f, bool wire_count = true) {
        if (!parallel) return;
        if (wire_count) ++g_sshwire.s_exch;          /* M10 [ssh-wire] */
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
    #define ALLREDUCE_SUM(var) do { if (parallel) { \
        MPI_Allreduce(MPI_IN_PLACE, &(var), 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM); \
        ++g_sshwire.s_arb; } } while (0)             /* M10 [ssh-wire] */

    /* Initial ‖rhs‖² + tolerance (solver.F90:142-154). rhs is read at OWNED rows only. */
    SSH_WIRE_LAUNCH(1);
    real_t s_old = cg_dot(rhs, rhs, N);
    ALLREDUCE_SUM(s_old);
    real_t rtol = soltol * sqrt(s_old / (real_t)N_global);

    if (s_old == 0.0) {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_cg_zeroX", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) { X(row) = 0.0; });
        dyn->d_eta_fld.modify_device();
        si->last_iters = 0;
        ssh_wire_close_solve(0, 0.0, (double)rtol, partit);
        return 0;
    }

    /* M10 T3 — FESOM_SSH_DUMP: snapshot the solve inputs for the offline lab.
     * sync_host on read-only mirrors is value-neutral; the file is written at
     * solve exit (x_final + iters land in the footer). The dump decision is
     * uniform across ranks (env + solve counter), so the barrier inside
     * ssh_dump_write is collective by construction. */
    const long dump_solve = g_sshwire.solves + 1;
    const bool dumping = ssh_dump_dir() != NULL && ssh_dump_wanted(dump_solve);
    std::vector<real_t> dmp_av, dmp_pr, dmp_b, dmp_x0;
    if (dumping) {
        dyn->d_eta_fld.sync_host();
        dyn->ssh_rhs_fld.sync_host();
        /* const_cast: sync_host mutates only the DualView bookkeeping/mirror,
         * never the model values — the matrix stays what the device computed. */
        const_cast<fesom_ssh_stiff *>(S)->values_fld.sync_host();
        const_cast<fesom_ssh_stiff *>(S)->pr_values_fld.sync_host();
        dmp_x0.assign(dyn->d_eta,   dyn->d_eta   + N);
        dmp_b .assign(dyn->ssh_rhs, dyn->ssh_rhs + N);
        dmp_av.assign(S->values,    S->values    + S->nnz);
        dmp_pr.assign(S->pr_values, S->pr_values + S->nnz);
    }

    /* r0 = rhs - A·X. X must be halo-current before the SpMV gathers it at colind. */
    exch(dyn->d_eta_fld);                            /* solver.F90:421 EXCH(X) */
    SSH_WIRE_LAUNCH(2);                              /* r0 spmv + map */
    cg_spmv(rowptr, colind, vals, X, rr, N);
    Kokkos::parallel_for("fesom_cg_r0", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int row) { rr(row) = rhs(row) - rr(row); });
    if      (cgpoly) { if (parallel) ++g_sshwire.s_exch;
                       cgpoly_exchange(rr, partit); si->rr_fld.modify_device(); }  /* E.CG2: R-ring */
    else if (cgpipe) { ++g_sshwire.s_exch;
                       cgpipe_exchange_rr(si->rr_fld, partit); }  /* E.CG1: ONE fused 2-ring exchange */
    else             exch(si->rr_fld);               /* solver.F90:424 EXCH(rr) */

    /* z0 = M⁻¹ r0 ; pp = z0. E.CG1: zz + pp additionally at ring1 (owned rows
     * byte-unchanged; ring1 rows from the shipped preconditioner CSR).
     * E.CG2: the Chebyshev apply produces zz on owned+ring1 directly. */
    if (cgpoly) {
        SSH_WIRE_LAUNCH(g_cgpoly.d + 1);             /* f0 + d semi-iterations */
        cgpoly_apply(S, rr, zz);
        if (cgpoly_selfcheck_on()) cgpoly_selfcheck(S, rr, zz, partit, 0);
    } else {
        SSH_WIRE_LAUNCH(1);
        cg_spmv(rowptr, colind, prvals, rr, zz, N);
        if (cgpipe) { SSH_WIRE_LAUNCH(1); cgpipe_zz_ring1(rr, zz); }
    }
    const int Next = (cgpipe || cgpoly) ? N + mesh->eDim_nod2D : N;
    SSH_WIRE_LAUNCH(1);
    Kokkos::parallel_for("fesom_cg_pp0", Kokkos::RangePolicy<>(0, Next),
        KOKKOS_LAMBDA(const int row) { pp(row) = zz(row); });

    /* s_old = r0·z0 */
    SSH_WIRE_LAUNCH(1);
    s_old = cg_dot(rr, zz, N);
    ALLREDUCE_SUM(s_old);

    int iter = 0;
    real_t last_res = 0.0;                           /* M10: final recurrence residual */
    int verbose         = (getenv("FESOM_VERBOSE_CG") != NULL);
    int heartbeat_every = 100;
    static const bool cg_prof = (getenv("FESOM_CG_PROFILE") != NULL);
    double _cg_t0 = 0.0;
    if (cg_prof) { Kokkos::fence(); _cg_t0 = MPI_Wtime(); }   /* M5.2: opt-in CG-share timer (fences cost ~2-3%) */
    for (iter = 1; iter <= si->maxiter; ++iter) {
        if (!cgpipe && !cgpoly)                      /* E.CG1/E.CG2: pp ring1 is maintained by
                                                      * the recurrence — the exchange is DELETED */
            exch(si->pp_fld);                        /* solver.F90:442 EXCH(pp) */
        /* M5.2: fuse SpMV (App=A·pp) with the dot (s_aux=Σ pp·App) into ONE
         * parallel_reduce — App(row) is computed and pp(row)·App(row) accumulated
         * in row order, identical to the separate cg_spmv+cg_dot (Serial bit-id),
         * saving a kernel launch + a full device read of App per iteration. */
        real_t s_aux = 0.0;
        SSH_WIRE_LAUNCH(1);
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

        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_for("fesom_cg_axpy", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row) {
                X (row) += al * pp (row);
                rr(row) -= al * App(row);
            });
        if      (cgpoly) { if (parallel) ++g_sshwire.s_exch;
                           cgpoly_exchange(rr, partit); si->rr_fld.modify_device(); }  /* E.CG2 */
        else if (cgpipe) { ++g_sshwire.s_exch;
                           cgpipe_exchange_rr(si->rr_fld, partit); }  /* E.CG1: the iteration's ONLY exchange */
        else             exch(si->rr_fld);           /* solver.F90:462 EXCH(rr) */

        /* M5.2: fuse the preconditioner SpMV (zz=M⁻¹·rr) with cg_dot2 (sp0=Σrr·zz,
         * sp1=Σrr·rr) into ONE parallel_reduce (row order → Serial bit-id), then ONE
         * 2-element MPI_Allreduce instead of two (same SUM per component, fewer
         * blocking collectives → the per-iter sync count the GPU is latency-bound on).
         * E.CG2: the Chebyshev apply replaces the psolve; the dots run separately
         * (same row order, still ONE 2-element Allreduce). */
        real_t sp0 = 0.0, sp1 = 0.0;
        if (cgpoly) {
            SSH_WIRE_LAUNCH(g_cgpoly.d + 2);         /* apply (f0 + d semis) + dot2 */
            cgpoly_apply(S, rr, zz);
            if (cgpoly_selfcheck_on()) cgpoly_selfcheck(S, rr, zz, partit, iter);
            Kokkos::parallel_reduce("fesom_cg_dot2", Kokkos::RangePolicy<>(0, N),
                KOKKOS_LAMBDA(const int row, real_t &l0, real_t &l1) {
                    l0 += rr(row) * zz(row);
                    l1 += rr(row) * rr(row);
                }, sp0, sp1);
        } else {
        SSH_WIRE_LAUNCH(1);
        Kokkos::parallel_reduce("fesom_cg_psolve_dot2", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l0, real_t &l1) {
                real_t s = 0.0;
                const int rstart = rowptr(row), rend = rowptr(row + 1);
                for (int n = rstart; n < rend; ++n) s += prvals(n) * rr(colind(n));
                zz(row) = s;
                l0 += rr(row) * s;
                l1 += rr(row) * rr(row);
            }, sp0, sp1);
        if (cgpipe) { SSH_WIRE_LAUNCH(1); cgpipe_zz_ring1(rr, zz); }  /* E.CG1: ring1 rows (dots stay owned-only) */
        }
        if (parallel) {
            double sbuf[2] = { (double)sp0, (double)sp1 };
            MPI_Allreduce(MPI_IN_PLACE, sbuf, 2, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM);
            sp0 = sbuf[0]; sp1 = sbuf[1];
            ++g_sshwire.s_arb;                       /* M10 [ssh-wire] */
        }
        if (cgpoly && sp0 < 0.0) {                   /* SPD M ⇒ rr·M⁻¹rr ≥ 0 always */
            if (partit == NULL || partit->mype == 0) {
                fprintf(stderr, "[cgpoly] ABORT at iter %d: rr·M⁻¹rr = %g < 0 — the Chebyshev "
                        "preconditioner is indefinite (λmax bound broken?)\n", iter, (double)sp0);
                fflush(stderr);
            }
            FESOM_DIE("cgpoly: preconditioner indefinite (rr·M⁻¹rr = %g)", (double)sp0);
        }

        real_t residual = sqrt(sp1 / (real_t)N_global);
        last_res = residual;                         /* M10: exported to wire/verify */
        if (ssh_trace_on() && (partit == NULL || partit->mype == 0)) {
            fprintf(stderr, "[ssh-trace] it=%d al=%.17g res=%.17g\n",
                    iter, (double)al, (double)residual);
        }
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
        if (ssh_trace_on() && (partit == NULL || partit->mype == 0))
            fprintf(stderr, "[ssh-trace] it=%d be=%.17g\n", iter, (double)be);
        SSH_WIRE_LAUNCH(1);
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
    /* M10 FESOM_SSH_VERIFY: post-solve TRUE residual ‖b−A·x‖ vs the recurrence
     * residual (see the T2 banner: byte-transparent, comm/launches not counted). */
    if (ssh_verify_on()) {
        exch(dyn->d_eta_fld, false);                 /* halo-current X for the gather */
        real_t tn = 0.0;
        Kokkos::parallel_reduce("fesom_ssh_verify_true_res", Kokkos::RangePolicy<>(0, N),
            KOKKOS_LAMBDA(const int row, real_t &l) {
                real_t s = 0.0;
                const int rstart = rowptr(row), rend = rowptr(row + 1);
                for (int n = rstart; n < rend; ++n) s += vals(n) * X(colind(n));
                const real_t d = rhs(row) - s;
                l += d * d;
            }, tn);
        if (parallel)
            MPI_Allreduce(MPI_IN_PLACE, &tn, 1, MPI_DOUBLE, MPI_SUM, partit->MPI_COMM_FESOM);
        const double true_res = sqrt((double)tn / (double)N_global);
        const double gap      = fabs(true_res - (double)last_res);
        SshWireState &w = g_sshwire;
        if (true_res > w.v_maxtrue) w.v_maxtrue = true_res;
        if (gap > w.v_maxgap)       w.v_maxgap  = gap;
        const bool vfail = true_res > (double)rtol;
        if (vfail) ++w.v_fail;
        if (partit == NULL || partit->mype == 0) {
            fprintf(stderr, "[ssh-verify] solve %ld: true=%.6e rec=%.6e rtol=%.6e gap=%.3e%s\n",
                    w.solves + 1, true_res, (double)last_res, (double)rtol, gap,
                    vfail ? "  <-- TRUE RESIDUAL ABOVE rtol" : "");
            fflush(stderr);
        }
    }
    /* The C twin's exit EXCH(X) (solver.F90:507) is dropped: the driver does
     * exchange_nod2D(d_eta) immediately after this returns (the same unchanged X), so the
     * two are idempotent → bit-identical. X owned is device-current here; modify_device()
     * lets the driver sync_host() it before that halo. */
    dyn->d_eta_fld.modify_device();
    if (cgpoly) {                                    /* the cert ladder's iteration-count log */
        ++g_cgpoly.solves;
        if ((partit == NULL || partit->mype == 0)
            && (g_cgpoly.solves <= 3 || g_cgpoly.solves % 100 == 0)) {
            fprintf(stderr, "[cgpoly] solve %ld: iters=%d\n", g_cgpoly.solves, iter);
            fflush(stderr);
        }
    }
    if (dumping) {                                   /* M10 T3: x_final + footer */
        dyn->d_eta_fld.sync_host();
        ssh_dump_write(S, si, mesh, partit, dump_solve, dmp_av, dmp_pr, dmp_b, dmp_x0,
                       dyn->d_eta, iter, (double)last_res, (double)rtol);
    }
    si->last_iters = iter;
    ssh_wire_close_solve(iter, (double)last_res, (double)rtol, partit);
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
