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
 *   --sigma-drift    Layer-0 falsification experiment (derivations §0.4): run reference PCG
 *                    and compare the TRUE (p,Ap) against the CG-CG recurrence
 *                    σ_i = δ_i − β_i²σ_{i-1} that cg2/pipecg/oati rely on. Repeat with the
 *                    symmetrised preconditioner D^{-1/2}CD^{-1/2}. Then exit.
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
    double opt_tol = -1.0; int opt_maxiter = -1, reps = 1, sym_check = 0, sigma_drift = 0;
    for (int a = 3; a < argc; ++a) {
        if      (!strcmp(argv[a], "--solver")  && a + 1 < argc) setenv("FESOM_SSH_SOLVER", argv[++a], 1);
        else if (!strcmp(argv[a], "--tol")     && a + 1 < argc) opt_tol = atof(argv[++a]);
        else if (!strcmp(argv[a], "--maxiter") && a + 1 < argc) opt_maxiter = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--reps")    && a + 1 < argc) reps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--trace"))                    setenv("FESOM_SSH_TRACE", "1", 1);
        else if (!strcmp(argv[a], "--sym-check"))                sym_check = 1;
        else if (!strcmp(argv[a], "--sigma-drift"))              sigma_drift = 1;
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
    /* ---- Layer-0 falsification experiment: σ-recurrence drift (derivations §0.4) ----
     * Reference PCG on the dumped system, host-side (a diagnostic, NOT the production
     * solver), tracking per iteration:
     *    σ_true = (p_i, A p_i)          — what baseline PCG computes explicitly
     *    σ_rec  = δ_i − β_i² σ_rec,i-1  — what cg2/pipecg/oati recur instead
     * The identity σ_true == σ_rec holds iff M⁻¹ is symmetric. Run twice: with the
     * built preconditioner and with M̃⁻¹ = D^{-1/2} C D^{-1/2} (same spectrum, symmetric).
     * Also reports the orthogonality residual (u_i, r_{i-1}) — the term that must vanish. */
    if (sigma_drift) {
        const int Nx = N + mesh.eDim_nod2D;
        std::vector<real_t> pr_sym((size_t)stiff.nnz);
        {   /* p̃r[i,j] = pr[i,j]·sqrt(d_i/d_j); needs d over owned+halo → exchange the diag */
            std::vector<real_t> dg((size_t)Nx, 0.0);
            for (int i = 0; i < N; ++i) dg[(size_t)i] = stiff.values[stiff.rowptr[i]];
            if (mpi.npes > 1) fesom_halo_exchange(dg.data(), FESOM_HALO_NOD2D, 1, 1, &mpi);
            for (int i = 0; i < N; ++i)
                for (int n = stiff.rowptr[i]; n < stiff.rowptr[i + 1]; ++n) {
                    const int j = stiff.colind[n];
                    pr_sym[(size_t)n] = (j == i) ? stiff.pr_values[n]
                        : stiff.pr_values[n] * sqrt(dg[(size_t)i] / dg[(size_t)j]);
                }
        }
        std::vector<real_t> pr_orig(stiff.pr_values, stiff.pr_values + stiff.nnz);

        auto dot = [&](const std::vector<real_t> &a, const std::vector<real_t> &b) {
            double s = 0.0;
            for (int i = 0; i < N; ++i) s += (double)a[(size_t)i] * (double)b[(size_t)i];
            if (mpi.npes > 1) MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, mpi.MPI_COMM_FESOM);
            return s;
        };
        auto spmv = [&](const real_t *vals, std::vector<real_t> &v, std::vector<real_t> &y) {
            if (mpi.npes > 1) fesom_halo_exchange(v.data(), FESOM_HALO_NOD2D, 1, 1, &mpi);
            for (int i = 0; i < N; ++i) {
                real_t s = 0.0;
                for (int n = stiff.rowptr[i]; n < stiff.rowptr[i + 1]; ++n)
                    s += vals[n] * v[(size_t)stiff.colind[n]];
                y[(size_t)i] = s;
            }
        };

        for (int leg = 0; leg < 2; ++leg) {
            const real_t *PR = leg == 0 ? pr_orig.data() : pr_sym.data();
            const char *name = leg == 0 ? "pr_values (as built, NON-symmetric)"
                                        : "D^-1/2 C D^-1/2 (symmetrised)";
            std::vector<real_t> x(Nx,0.0), r(Nx,0.0), u(Nx,0.0), p(Nx,0.0), ap(Nx,0.0),
                                r_prev(Nx,0.0), tmp(Nx,0.0);
            for (int i = 0; i < N; ++i) x[(size_t)i] = d_x0[(size_t)i];
            spmv(stiff.values, x, tmp);
            for (int i = 0; i < N; ++i) r[(size_t)i] = d_b[(size_t)i] - tmp[(size_t)i];
            spmv(PR, r, u);
            for (int i = 0; i < N; ++i) p[(size_t)i] = u[(size_t)i];
            double gamma = dot(r, u), gamma_prev = 0.0, sigma_rec = 0.0, beta = 0.0;
            const double rtol_ = (opt_tol > 0.0 ? opt_tol : d_soltol)
                               * sqrt(dot(d_b, d_b) / (double)mesh.nod2D);
            double worst_rel = 0.0, worst_orth = 0.0;
            int it = 0;
            if (mpi.mype == 0)
                printf("[lab-sigma] === leg %d: %s ===\n"
                       "[lab-sigma] %5s %16s %16s %12s %14s\n", leg, name,
                       "iter", "sigma_true", "sigma_recurred", "rel.drift", "(u_i,r_i-1)/gamma");
            const int SD_MAX = 500;   /* run to CONVERGENCE, not a fixed window: the
                                       * iteration count each preconditioner needs is the
                                       * answer to "does symmetrising help plain PCG too?" */
            for (it = 1; it <= SD_MAX; ++it) {
                spmv(stiff.values, p, ap);
                const double sig_true = dot(p, ap);
                if (it == 1) sigma_rec = sig_true;   /* seed: p₀=u₀ ⇒ σ₀ = δ₀ exactly */
                const double alpha = gamma / sig_true;
                const double rel = fabs(sig_true - sigma_rec) / fabs(sig_true);
                if (it > 1 && rel > worst_rel) worst_rel = rel;
                if (mpi.mype == 0 && (it <= 8 || it % 10 == 0))
                    printf("[lab-sigma] %5d %16.9e %16.9e %12.3e %14.3e\n",
                           it, sig_true, sigma_rec, rel,
                           it > 1 ? dot(u, r_prev) / gamma : 0.0);
                for (int i = 0; i < N; ++i) r_prev[(size_t)i] = r[(size_t)i];
                for (int i = 0; i < N; ++i) {
                    x[(size_t)i] += alpha * p[(size_t)i];
                    r[(size_t)i] -= alpha * ap[(size_t)i];
                }
                spmv(PR, r, u);
                const double gamma_new = dot(r, u);
                const double rr = dot(r, r);
                /* the orthogonality term that (★) needs to vanish */
                const double orth = fabs(dot(u, r_prev)) / fabs(gamma_new);
                if (it > 1 && orth > worst_orth) worst_orth = orth;
                gamma_prev = gamma; gamma = gamma_new;
                beta = gamma / gamma_prev;
                /* CG-CG recurrence for the NEXT iteration's σ: needs δ = (w,u), w = A u */
                spmv(stiff.values, u, tmp);
                const double delta = dot(tmp, u);
                sigma_rec = delta - beta * beta * sigma_rec;
                for (int i = 0; i < N; ++i) p[(size_t)i] = u[(size_t)i] + beta * p[(size_t)i];
                if (sqrt(rr / (double)mesh.nod2D) < rtol_) break;
            }
            if (mpi.mype == 0)
                printf("[lab-sigma] leg %d SUMMARY: PLAIN-PCG iters to converge = %d%s   "
                       "WORST |sigma_true-sigma_rec|/|sigma_true| = %.3e   WORST |(u_i,r_i-1)|/gamma = %.3e\n",
                       leg, it, it > SD_MAX ? " (NOT CONVERGED)" : "", worst_rel, worst_orth);
        }
        if (mpi.mype == 0)
            printf("[lab-sigma] VERDICT: the CG-CG family recurs sigma; leg 0 drift is the error"
                   " cg2/pipecg/oati inherit from a non-symmetric M (derivations sec 0.4).\n");
    }
    else if (sym_check) {
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
        /* Certification compares against the DUMP, which was produced by baseline `cg`.
         * That comparison is a certification only when the replay runs the same solver;
         * for a variant it is a REPORT (a different solver is expected to take a different
         * number of iterations and land on a different — equally valid — iterate). */
        const char *sv = getenv("FESOM_SSH_SOLVER");
        const bool is_baseline = !sv || !sv[0] || strcmp(sv, "cg") == 0;
        const int rc = !is_baseline ? 0
                     : (mpi.npes == 1 ? !(bitwise && iters_ok) : !iters_ok);
        if (rc > worst_rc) worst_rc = rc;
        if (mpi.mype == 0)
            printf("[lab] rep %d: solver=%s iters=%d (dump/cg %d, %s)  max|X-x_final|=%.3e  "
                   "bitwise %ld/%ld %s\n",
                   rep, is_baseline ? "cg" : sv, iters, ref_iters,
                   iters_ok ? "same" : "differs", gmx, gbit, gN, bitwise ? "(EXACT)" : "");
    }
    if (mpi.mype == 0) {
        const char *sv = getenv("FESOM_SSH_SOLVER");
        const bool is_baseline = !sv || !sv[0] || strcmp(sv, "cg") == 0;
        if (is_baseline)
            printf("[lab] CERT %s (criteria: %s)\n", worst_rc == 0 ? "PASS" : "FAIL",
                   mpi.npes == 1 ? "bitwise x_final + iters" : "iters (np>1)");
        else
            printf("[lab] REPORT only — solver '%s' != the dump's baseline cg; iterate and "
                   "iteration-count differences above are expected, not failures\n", sv);
    }
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
    fesom_ssh_m10_free();      /* M10 solver scratch — else Kokkos aborts at static destruction */
    }
    Kokkos::finalize();
    fesom_mpi_finalize(&mpi);
    return 0;
}
