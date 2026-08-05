/* M10 T3 — the SSH solver lab: offline matrix-replay driver on the REAL model objects
 * (plan docs/plans/20260805-m10-ssh-solvers.md; format src/fesom_ssh_dump.h).
 *
 * WHAT IT IS: the model's own init path (mesh read → partitioning → halo machinery →
 * CSR build) followed by the model's own solver (fesom_ssh_solve_cg_kk) on a dumped
 * right-hand side/matrix — so a lab replay exercises the EXACT production code with zero
 * reimplementation. The dump's rowptr/colind are asserted BITWISE equal to the freshly
 * built CSR: that proves the lab reconstructed the same partitioning the model ran.
 *
 * RULES (doc of record): lab numbers are NEVER performance numbers of record; a lab-tuned
 * constant becomes a default only after an in-model 20-step gate reproduces the lab's
 * iters/solve within ±10 % (R4).
 *
 * Usage:  [mpirun -np N]  fesom_ssh_lab <mesh_dir> <step_dir> [options]
 *   --solver <s>     setenv FESOM_SSH_SOLVER=<s> (dispatch lands in T5a; default = cg)
 *   --tol <t>        override soltol for the replay (default: the dump's)
 *   --maxiter <n>    override maxiter (default: the dump's)
 *   --reps <r>       solve repetitions (default 1; each rep restarts from the dumped x0)
 *   --trace          FESOM_SSH_TRACE=1 (per-iteration α/β/res lines, rank 0)
 *   --sym-check      R1 symmetry-defect measurement on pr_values (+ A as control), then exit
 *   --knob K=V       arbitrary env knob, repeatable (e.g. --knob FESOM_SPEED_CGPOLY=3)
 * N must equal the dump's npes. Exit 0 = certification criteria met (np1: bitwise x_final
 * + same iters; np>1: same iters), 1 = mismatch, 2 = usage/load error.
 */
#include "fesom_ale.h"
#include "fesom_constants.h"
#include "fesom_dyn.h"
#include "fesom_halo.h"
#include "fesom_halo_device.hpp"
#include "fesom_mesh.h"
#include "fesom_mpi.h"
#include "fesom_partit.h"
#include "fesom_ssh.h"
#include "fesom_ssh_dump.h"
#include "fesom_types.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

static void die_usage(int rank)
{
    if (rank == 0)
        fprintf(stderr, "usage: fesom_ssh_lab <mesh_dir> <step_dir> [--solver s] [--tol t]\n"
                        "       [--maxiter n] [--reps r] [--trace] [--sym-check] [--knob K=V]\n");
    MPI_Finalize();
    exit(2);
}

int main(int argc, char **argv)
{
    fesom_mpi mpi;
    /* argv[1] = mesh dir — fesom_mpi_init (MPI_Init inside) reads dist_<np> exactly like
     * the model; the usage check runs after it so MPI is initialised exactly once. */
    const char *mesh_dir = (argc >= 2) ? argv[1] : ".";
    fesom_mpi_init(&mpi, mesh_dir, argc, argv);
    if (argc < 3) die_usage(mpi.mype);
    const char *step_dir = argv[2];

    /* options */
    double opt_tol = -1.0; int opt_maxiter = -1, reps = 1, sym_check = 0;
    for (int a = 3; a < argc; ++a) {
        if      (!strcmp(argv[a], "--solver")  && a + 1 < argc) setenv("FESOM_SSH_SOLVER", argv[++a], 1);
        else if (!strcmp(argv[a], "--tol")     && a + 1 < argc) opt_tol = atof(argv[++a]);
        else if (!strcmp(argv[a], "--maxiter") && a + 1 < argc) opt_maxiter = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--reps")    && a + 1 < argc) reps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--trace"))                    setenv("FESOM_SSH_TRACE", "1", 1);
        else if (!strcmp(argv[a], "--sym-check"))                sym_check = 1;
        else if (!strcmp(argv[a], "--knob")    && a + 1 < argc) {
            char *kv = strdup(argv[++a]); char *eq = strchr(kv, '=');
            if (!eq) die_usage(mpi.mype);
            *eq = 0; setenv(kv, eq + 1, 1); free(kv);
        }
        else die_usage(mpi.mype);
    }
    setenv("FESOM_SSH_STATS", "1", 1);               /* the lab always reports wire lines */

    /* Kokkos with per-node-local-rank device binding (the fesom_main M3.1 block). */
    int local_rank = 0;
    {
        MPI_Comm shm = MPI_COMM_NULL;
        if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm)
                == MPI_SUCCESS && shm != MPI_COMM_NULL) {
            MPI_Comm_rank(shm, &local_rank);
            MPI_Comm_free(&shm);
        } else if (getenv("SLURM_LOCALID")) local_rank = atoi(getenv("SLURM_LOCALID"));
    }
    Kokkos::InitializationSettings ks;
    ks.set_device_id(local_rank);
    Kokkos::initialize(ks);
    {
    /* ---- model init path (fesom_main order, minus IC/forcing/stepping) ---- */
    fesom_mesh mesh;
    fesom_mesh_init(&mesh);
    fesom_ale_mode_init();
    fesom_mesh_read(&mesh, mesh_dir, &mpi);
    if (mpi.npes == 1)
        fesom_partit_set_global_counts_serial(&mpi, mesh.nod2D, mesh.elem2D, mesh.edge2D);
    fesom_mesh_compute_metrics(&mesh, &mpi);
    fesom_mesh_alloc_state(&mesh);
    fesom_halo_identity_test(&mpi);

    fesom_dyn dyn;
    fesom_dyn_alloc(&dyn, &mesh);

    fesom_ssh_stiff  stiff;
    fesom_solverinfo solver;
    fesom_ssh_stiff_alloc_and_build(&stiff, &mesh);
    fesom_ssh_preconditioner(&stiff, &mesh, &mpi);
    fesom_solverinfo_alloc(&solver, &mesh);
    solver.partit = &mpi;

    const int N = mesh.myDim_nod2D;

    /* ---- load the dump ---------------------------------------------------- */
    char path[1200];
    snprintf(path, sizeof path, "%s/rank%05d.bin", step_dir, mpi.mype);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[lab %d] cannot open %s\n", mpi.mype, path); MPI_Abort(MPI_COMM_WORLD, 2); }
    uint64_t u64 = 0; int32_t i32 = 0; double f64 = 0.0;
    #define RD(x)  do { if (fesom_sshdump_rd(f, &(x), sizeof(x))) { \
        fprintf(stderr, "[lab %d] short read %s\n", mpi.mype, path); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
    #define EXPECT(cond, what) do { if (!(cond)) { \
        fprintf(stderr, "[lab %d] dump mismatch: %s\n", mpi.mype, what); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
    RD(u64); EXPECT(u64 == FESOM_SSHDUMP_MAGIC, "magic");
    RD(i32); EXPECT(i32 == FESOM_SSHDUMP_VERSION, "version");
    RD(i32); const int d_step = i32;
    RD(i32); EXPECT(i32 == mpi.npes, "npes (run the lab at the dump's np)");
    RD(i32); EXPECT(i32 == mpi.mype, "mype");
    RD(i32); EXPECT(i32 == N, "myDim");
    RD(i32); EXPECT(i32 == mesh.eDim_nod2D, "eDim");
    RD(i32); EXPECT(i32 == stiff.nnz, "nnz");
    RD(i32); EXPECT(i32 == mesh.nod2D, "nod2D");
    RD(f64); const double d_dt = f64;
    RD(f64); const double d_soltol = f64;
    RD(i32); const int d_maxiter = i32;
    RD(u64);
    {
        uint64_t mh = fesom_sshdump_fnv1a(mpi.myList_nod2D,
                          (size_t)(N + mesh.eDim_nod2D) * sizeof(int));
        EXPECT(u64 == mh, "meshhash (partition fingerprint)");
    }
    /* Adopt the dump's dt. The freshly built matrix/preconditioner values are about to be
     * overwritten from the dump, and nothing else downstream reads dt — the assignment
     * only keeps any dt-derived diagnostics honest. */
    fesom_phase1_dt = (real_t)d_dt;

    std::vector<int>    d_rowptr((size_t)N + 1), d_colind((size_t)stiff.nnz);
    std::vector<real_t> d_av((size_t)stiff.nnz), d_pr((size_t)stiff.nnz);
    std::vector<real_t> d_b((size_t)N), d_x0((size_t)N), d_xf((size_t)N);
    #define RDA(vec, bytes, what) do { int rc_ = fesom_sshdump_rd_arr(f, (vec), (bytes)); \
        if (rc_) { fprintf(stderr, "[lab %d] array %s: %s\n", mpi.mype, what, \
                           rc_ == -2 ? "CHECKSUM MISMATCH" : "short read"); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
    RDA(d_rowptr.data(), d_rowptr.size() * sizeof(int),    "rowptr");
    RDA(d_colind.data(), d_colind.size() * sizeof(int),    "colind");
    RDA(d_av.data(),     d_av.size()     * sizeof(real_t), "values");
    RDA(d_pr.data(),     d_pr.size()     * sizeof(real_t), "pr_values");
    RDA(d_b.data(),      d_b.size()      * sizeof(real_t), "b");
    RDA(d_x0.data(),     d_x0.size()     * sizeof(real_t), "x0");
    RDA(d_xf.data(),     d_xf.size()     * sizeof(real_t), "x_final");
    RD(i32); const int    ref_iters = i32;
    RD(f64); const double ref_res   = f64;
    RD(f64); const double ref_rtol  = f64;
    RD(u64); EXPECT(u64 == FESOM_SSHDUMP_MAGIC, "trailer magic (truncated file?)");
    fclose(f);

    /* The partitioning proof: freshly built CSR must equal the dumped one BITWISE. */
    EXPECT(memcmp(stiff.rowptr, d_rowptr.data(), d_rowptr.size() * sizeof(int)) == 0,
           "rowptr != freshly built (different partitioning?!)");
    EXPECT(memcmp(stiff.colind, d_colind.data(), d_colind.size() * sizeof(int)) == 0,
           "colind != freshly built (different partitioning?!)");

    /* Install the dumped matrix/preconditioner (live zstar values travel with the dump). */
    memcpy(stiff.values,    d_av.data(), d_av.size() * sizeof(real_t));
    memcpy(stiff.pr_values, d_pr.data(), d_pr.size() * sizeof(real_t));
    stiff.values_fld.modify_host();    stiff.values_fld.sync_device();
    stiff.pr_values_fld.modify_host(); stiff.pr_values_fld.sync_device();

    if (mpi.mype == 0)
        printf("[lab] dump step %d: np%d nnz=%d ref iters=%d res=%.6e rtol=%.6e "
               "(dump soltol=%g maxiter=%d)\n",
               d_step, mpi.npes, stiff.nnz, ref_iters, ref_res, ref_rtol, d_soltol, d_maxiter);

    /* ---- R1: symmetry-defect measurement ---------------------------------- */
    if (sym_check) {
        /* Owned-pair scan (both endpoints owned rows). Halo-column pairs need the
         * neighbour's row — reported as uncovered fraction; run at np1 for 100 %
         * coverage (the CORE2 case). A-matrix defect is the control (~0 expected). */
        double mx_pr_def = 0.0, mx_pr = 0.0, mx_av_def = 0.0, mx_av = 0.0;
        long pairs = 0, uncovered = 0;
        for (int row = 0; row < N; ++row) {
            for (int n = stiff.rowptr[row]; n < stiff.rowptr[row + 1]; ++n) {
                const int col = stiff.colind[n];
                if (col == row) continue;
                if (col >= N) { ++uncovered; continue; }   /* halo column: reverse row not local */
                if (col < row) continue;                   /* count each pair once */
                ++pairs;
                double rev_pr = 0.0, rev_av = 0.0; int found = 0;
                for (int m = stiff.rowptr[col]; m < stiff.rowptr[col + 1]; ++m)
                    if (stiff.colind[m] == row) { rev_pr = stiff.pr_values[m]; rev_av = stiff.values[m]; found = 1; break; }
                if (!found) { ++uncovered; continue; }
                const double dpr = fabs((double)stiff.pr_values[n] - rev_pr);
                const double dav = fabs((double)stiff.values[n]    - rev_av);
                if (dpr > mx_pr_def) mx_pr_def = dpr;
                if (dav > mx_av_def) mx_av_def = dav;
                const double apr = fabs((double)stiff.pr_values[n]);
                const double aav = fabs((double)stiff.values[n]);
                if (apr > mx_pr) mx_pr = apr;
                if (aav > mx_av) mx_av = aav;
            }
        }
        double loc[4] = { mx_pr_def, mx_pr, mx_av_def, mx_av }, glob[4];
        long   lcnt[2] = { pairs, uncovered }, gcnt[2];
        MPI_Allreduce(loc, glob, 4, MPI_DOUBLE, MPI_MAX, mpi.MPI_COMM_FESOM);
        MPI_Allreduce(lcnt, gcnt, 2, MPI_LONG, MPI_SUM, mpi.MPI_COMM_FESOM);
        if (mpi.mype == 0) {
            printf("[lab-sym] pr_values : max|M_ij-M_ji| = %.6e  max|M_ij| = %.6e  "
                   "defect ratio = %.6e\n", glob[0], glob[1], glob[0] / glob[1]);
            printf("[lab-sym] A (control): max|A_ij-A_ji| = %.6e  max|A_ij| = %.6e  "
                   "defect ratio = %.6e\n", glob[2], glob[3], glob[2] / glob[3]);
            printf("[lab-sym] off-diag pairs scanned = %ld, uncovered (halo-crossing) = %ld "
                   "(%.1f %% coverage)\n", gcnt[0], gcnt[1],
                   100.0 * (double)gcnt[0] / (double)(gcnt[0] + gcnt[1]));
        }
    } else {

    /* ---- replay ------------------------------------------------------------ */
    solver.soltol  = (opt_tol     > 0.0) ? (real_t)opt_tol : (real_t)d_soltol;
    solver.maxiter = (opt_maxiter > 0)   ? opt_maxiter     : d_maxiter;

    int worst_rc = 0;
    for (int rep = 1; rep <= reps; ++rep) {
        /* restore inputs: X = x0, rhs = b (owned; halos are refreshed inside the solve) */
        memset(dyn.d_eta,   0, (size_t)(N + mesh.eDim_nod2D) * sizeof(real_t));
        memset(dyn.ssh_rhs, 0, (size_t)(N + mesh.eDim_nod2D) * sizeof(real_t));
        memcpy(dyn.d_eta,   d_x0.data(), (size_t)N * sizeof(real_t));
        memcpy(dyn.ssh_rhs, d_b.data(),  (size_t)N * sizeof(real_t));
        dyn.d_eta_fld.modify_host();   dyn.d_eta_fld.sync_device();
        dyn.ssh_rhs_fld.modify_host(); dyn.ssh_rhs_fld.sync_device();

        const int iters = fesom_ssh_solve_cg_kk(&stiff, &solver, &mesh, &dyn);

        dyn.d_eta_fld.sync_host();
        double mxd = 0.0; long nbit = 0;
        for (int i = 0; i < N; ++i) {
            const double d = fabs((double)dyn.d_eta[i] - (double)d_xf[i]);
            if (d > mxd) mxd = d;
            if (dyn.d_eta[i] == d_xf[i]) ++nbit;
        }
        double gmx = 0.0; long gbit = 0, gN = 0, lN = N;
        MPI_Allreduce(&mxd,  &gmx,  1, MPI_DOUBLE, MPI_MAX, mpi.MPI_COMM_FESOM);
        MPI_Allreduce(&nbit, &gbit, 1, MPI_LONG,   MPI_SUM, mpi.MPI_COMM_FESOM);
        MPI_Allreduce(&lN,   &gN,   1, MPI_LONG,   MPI_SUM, mpi.MPI_COMM_FESOM);
        const int bitwise = (gbit == gN);
        const int iters_ok = (iters == ref_iters);
        /* certification: np1 → bitwise + iters; np>1 → iters (bitwise reported) */
        const int rc = mpi.npes == 1 ? !(bitwise && iters_ok) : !iters_ok;
        if (rc > worst_rc) worst_rc = rc;
        if (mpi.mype == 0)
            printf("[lab] rep %d: iters=%d (dump %d %s)  max|X-x_final|=%.3e  "
                   "bitwise %ld/%ld %s\n",
                   rep, iters, ref_iters, iters_ok ? "OK" : "MISMATCH",
                   gmx, gbit, gN, bitwise ? "(EXACT)" : "");
    }
    if (mpi.mype == 0)
        printf("[lab] CERT %s (criteria: %s)\n", worst_rc == 0 ? "PASS" : "FAIL",
               mpi.npes == 1 ? "bitwise x_final + iters" : "iters (np>1)");
    if (worst_rc) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    /* ---- teardown (fesom_main order: Views die before Kokkos::finalize) ---- */
    fesom_ssh_wire_report();
    fesom_solverinfo_free(&solver);
    fesom_ssh_stiff_free(&stiff);
    fesom_dyn_free(&dyn);
    fesom_mesh_free(&mesh);
    fesom_halo_device_free();
    fesom_ssh_cgpipe_free();
    fesom_ssh_cgpoly_free();
    }
    Kokkos::finalize();
    fesom_mpi_finalize(&mpi);
    return 0;
}
