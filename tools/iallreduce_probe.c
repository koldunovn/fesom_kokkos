/* M10 T2 (R2) — MPI_Iallreduce progression probe (plan docs/plans/20260805-m10-ssh-solvers.md).
 *
 * QUESTION: with MPI_THREAD_SINGLE (the port's init mode, fesom_partit.cpp:225) and no
 * progress thread, does an MPI_Iallreduce COMPLETE while the host does unrelated busy-work,
 * or does all progress happen inside MPI_Wait? pipecg's overlap claim rests on the answer
 * (R2: openmpi 4.1 is expected to have NO async progress by default — this probe MEASURES
 * it instead of assuming; a null pipecg-vs-cg2 delta with a null probe is then attributed
 * "stack", not "algorithm").
 *
 * METHOD (all ranks in lockstep):
 *   1. t_ar = median of REPS timed blocking MPI_Allreduce(vec, VECLEN doubles).
 *   2. for each busy factor f in {0, 0.5, 1, 2, 4}:
 *        barrier; t0; MPI_Iallreduce(...); host_busy(f * t_ar); MPI_Wait; t1
 *      medians of t_total and t_wait over REPS.
 *   hidden(f)  = t_ar + T_busy - t_total(f)   (AR latency that vanished under busy work)
 *   overlap%   = 100 * hidden / t_ar          (~100 full async progression, ~0 none;
 *                                              t_wait ≈ t_ar at every f is the no-progress
 *                                              signature)
 *
 * The busy loop polls MPI_Wtime around a volatile FLOP accumulator — no MPI calls inside
 * (that would BE manual progression and defeat the question).
 *
 * Build: mpicc -O2 -o iallreduce_probe iallreduce_probe.c -lm
 * Run:   srun -n <N> ./iallreduce_probe [veclen_doubles]   (default 3 — the cg2/pipecg
 *        fused-allreduce element count; run with ranks spread over >= 2 nodes so the
 *        reduction crosses the real fabric)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPS   400
#define WARM   50
#define NFACT  5
static const double FACT[NFACT] = { 0.0, 0.5, 1.0, 2.0, 4.0 };

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n)
{
    qsort(v, n, sizeof(double), cmp_d);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static volatile double g_sink = 0.0;
/* Spin for `dur` seconds of pure host ALU work; returns loop count (anti-DCE). */
static long host_busy(double dur)
{
    if (dur <= 0.0) return 0;
    const double t0 = MPI_Wtime();
    long   n = 0;
    double a = 1.000000001;
    while (MPI_Wtime() - t0 < dur) {
        for (int i = 0; i < 64; ++i) a = a * 1.000000001 + 1e-12;
        ++n;
    }
    g_sink += a;
    return n;
}

int main(int argc, char **argv)
{
    int provided = 0;
    /* SAME init mode as the model — the whole point. */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    const int veclen = (argc > 1) ? atoi(argv[1]) : 3;

    double *snd = calloc((size_t)veclen, sizeof(double));
    double *rcv = calloc((size_t)veclen, sizeof(double));
    for (int i = 0; i < veclen; ++i) snd[i] = 1.0 + rank + i;

    static double tv[REPS], tw[REPS];

    /* --- 1. blocking Allreduce latency ------------------------------------ */
    for (int r = 0; r < WARM; ++r)
        MPI_Allreduce(snd, rcv, veclen, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int r = 0; r < REPS; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        MPI_Allreduce(snd, rcv, veclen, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tv[r] = MPI_Wtime() - t0;
    }
    double t_ar = median(tv, REPS);
    /* every rank uses rank 0's t_ar so the busy windows agree */
    MPI_Bcast(&t_ar, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char ver[MPI_MAX_LIBRARY_VERSION_STRING]; int vl = 0;
        MPI_Get_library_version(ver, &vl);
        char *nl = strchr(ver, '\n'); if (nl) *nl = 0;
        printf("[iallreduce-probe] npes=%d veclen=%d thread=MPI_THREAD_SINGLE(provided=%d)\n",
               npes, veclen, provided);
        printf("[iallreduce-probe] MPI: %s\n", ver);
        printf("[iallreduce-probe] blocking Allreduce median: %.2f us\n", t_ar * 1e6);
        printf("[iallreduce-probe] %6s %12s %12s %12s %12s %9s\n",
               "factor", "busy_us", "total_us", "wait_us", "hidden_us", "overlap%");
    }

    /* --- 2. Iallreduce + busy + Wait -------------------------------------- */
    for (int fi = 0; fi < NFACT; ++fi) {
        const double T = FACT[fi] * t_ar;
        for (int r = 0; r < WARM; ++r) {
            MPI_Request rq;
            MPI_Iallreduce(snd, rcv, veclen, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rq);
            host_busy(T);
            MPI_Wait(&rq, MPI_STATUS_IGNORE);
        }
        for (int r = 0; r < REPS; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            MPI_Request rq;
            MPI_Iallreduce(snd, rcv, veclen, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rq);
            host_busy(T);
            const double tb = MPI_Wtime();
            MPI_Wait(&rq, MPI_STATUS_IGNORE);
            const double t1 = MPI_Wtime();
            tv[r] = t1 - t0;
            tw[r] = t1 - tb;
        }
        const double t_tot  = median(tv, REPS);
        const double t_wait = median(tw, REPS);
        /* medians are computed per-rank on rank 0's timings — reduce MAX so the
         * printed number is the slowest rank's view (the model's effective cost) */
        double red[2] = { t_tot, t_wait };
        MPI_Allreduce(MPI_IN_PLACE, red, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            const double hidden = t_ar + T - red[0];
            printf("[iallreduce-probe] %6.1f %12.2f %12.2f %12.2f %12.2f %9.1f\n",
                   FACT[fi], T * 1e6, red[0] * 1e6, red[1] * 1e6, hidden * 1e6,
                   100.0 * hidden / t_ar);
            fflush(stdout);
        }
    }

    if (rank == 0)
        printf("[iallreduce-probe] no-progress signature: wait_us ~= blocking latency at "
               "every factor; full-progress: wait_us -> ~0 and overlap%% -> ~100 for "
               "factor >= 1.\n");
    free(snd); free(rcv);
    MPI_Finalize();
    return 0;
}
