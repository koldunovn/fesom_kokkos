/* M7 E.IMB.0 — FESOM_SPEED_PHASESTATS implementation. See fesom_phasestats.h
 * for the design constraints (no fences, PMPI wait attribution, opt-in only).
 *
 * The PMPI interposition at the bottom is the standard MPI profiling shim:
 * OpenMPI exports every MPI_* as a weak alias of PMPI_*, so defining the
 * strong symbol here reroutes EVERY call in the process — ours, netcdf's,
 * hdf5's — through the timing hook. Nothing is changed about the call
 * itself (straight PMPI_ pass-through), so knob-ON stays bit-identical:
 * this file computes no model arithmetic, only reads the clock. */

#include "fesom_phasestats.h"
#include "fesom_speed.hpp"   // fesom_speed_on_exp (includes Kokkos_Macros.hpp first — the include-order trap)

#include <mpi.h>
#include <cstdio>
#include <cstdlib>

/* ------------------------------------------------------------------ state */

static int    s_knob = -1;                    /* fesom_speed_on_exp cache    */
static bool   s_armed = false;                /* true only inside the timed window */
static int    s_cur   = FESOM_PH_OTHER;       /* phase the wall clock is charging  */
static double s_last  = 0.0;                  /* last mark timestamp         */
static double s_wall [FESOM_PH_N] = {0};      /* per-phase wall (s), this rank     */
static double s_wait [FESOM_PH_N] = {0};      /* per-phase wall INSIDE MPI (s)     */
static double s_calls[FESOM_PH_N] = {0};      /* per-phase timed MPI calls         */

bool fesom_phasestats_enabled(void)
{
    return fesom_speed_on_exp("PHASESTATS", &s_knob);
}

void fesom_phasestats_open(void)
{
    if (!fesom_phasestats_enabled()) return;
    for (int i = 0; i < FESOM_PH_N; ++i) s_wall[i] = s_wait[i] = s_calls[i] = 0.0;
    s_cur   = FESOM_PH_OTHER;
    s_last  = MPI_Wtime();
    s_armed = true;
}

void fesom_phasestats_mark(int phase)
{
    if (!s_armed) return;
    const double t = MPI_Wtime();
    s_wall[s_cur] += t - s_last;
    s_last = t;
    s_cur  = phase;
}

void fesom_phasestats_close(void)
{
    if (!s_armed) return;
    const double t = MPI_Wtime();
    s_wall[s_cur] += t - s_last;
    s_last  = t;
    s_armed = false;
}

/* ---------------------------------------------------------------- report */

void fesom_phasestats_report(int timed_steps, fesom_partit *p)
{
    if (!p || timed_steps <= 0 || !fesom_phasestats_enabled()) return;  /* ALL ranks evaluate identically */

    const int npes = p->npes, me = p->mype;
    double loc[3 * FESOM_PH_N];
    for (int i = 0; i < FESOM_PH_N; ++i) {
        loc[i]                 = s_wall[i];
        loc[FESOM_PH_N + i]    = s_wait[i];
        loc[2*FESOM_PH_N + i]  = s_calls[i];
    }
    double *all = NULL;
    if (me == 0) all = (double *)malloc((size_t)npes * 3 * FESOM_PH_N * sizeof(double));
    MPI_Gather(loc, 3 * FESOM_PH_N, MPI_DOUBLE, all, 3 * FESOM_PH_N, MPI_DOUBLE,
               0, p->MPI_COMM_FESOM);
    if (me != 0) return;

    static const char *nm[FESOM_PH_N] = {"force", "ice", "icedyn", "iceadv",
                                         "coupl", "ocean", "cg", "bt", "other"};
    const double ms = 1e3 / (double)timed_steps;   /* s over window -> ms/step */

    printf("[phasestats] steps=%d ranks=%d  (ms/step; busy = wall - MPI-wait; @r = argmax rank)\n",
           timed_steps, npes);
    printf("[phasestats]  %-6s | %28s | %28s | %s\n",
           "phase", "busy   min /  mean /   max @r", "wait   min /  mean /   max @r", "mpi/step");
    for (int ph = 0; ph <= FESOM_PH_N; ++ph) {     /* ph == FESOM_PH_N -> TOTAL row */
        double bmin = 1e30, bmax = -1e30, bsum = 0.0, wmin = 1e30, wmax = -1e30, wsum = 0.0, csum = 0.0;
        int barg = 0, warg = 0;
        for (int r = 0; r < npes; ++r) {
            const double *row = all + (size_t)r * 3 * FESOM_PH_N;
            double w = 0.0, mw = 0.0, c = 0.0;
            if (ph < FESOM_PH_N) { w = row[ph]; mw = row[FESOM_PH_N + ph]; c = row[2*FESOM_PH_N + ph]; }
            else for (int i = 0; i < FESOM_PH_N; ++i) {   /* TOTAL = sum over phases */
                w += row[i]; mw += row[FESOM_PH_N + i]; c += row[2*FESOM_PH_N + i];
            }
            const double b = w - mw;
            if (b > bmax) { bmax = b; barg = r; }
            if (b < bmin) bmin = b;
            if (mw > wmax) { wmax = mw; warg = r; }
            if (mw < wmin) wmin = mw;
            bsum += b; wsum += mw; csum += c;
        }
        printf("[phasestats]  %-6s | %7.1f /%7.1f /%7.1f @%-3d | %7.1f /%7.1f /%7.1f @%-3d | %.1f\n",
               ph < FESOM_PH_N ? nm[ph] : "TOTAL",
               bmin*ms, bsum/npes*ms, bmax*ms, barg,
               wmin*ms, wsum/npes*ms, wmax*ms, warg,
               csum / npes / (double)timed_steps);
    }
    printf("[phasestats] (TOTAL wall/step mean = loop s/step minus the final-barrier tail; "
           "busy spread names the straggler phase, wait shows who absorbs it)\n");

    /* Full per-rank table — the polar-fraction correlation needs rank identity. */
    printf("[phasestats-rank]  rk |   busy: force    ice icedyn iceadv  coupl  ocean     cg  other"
           " |   wait: force    ice icedyn iceadv  coupl  ocean     cg  other\n");
    for (int r = 0; r < npes; ++r) {
        const double *row = all + (size_t)r * 3 * FESOM_PH_N;
        printf("[phasestats-rank] %3d |       ", r);
        for (int i = 0; i < FESOM_PH_N; ++i) printf("%7.1f", (row[i] - row[FESOM_PH_N + i]) * ms);
        printf(" |       ");
        for (int i = 0; i < FESOM_PH_N; ++i) printf("%7.1f", row[FESOM_PH_N + i] * ms);
        printf("\n");
    }
    fflush(stdout);
    free(all);
}

/* ------------------------------------------------- PMPI wait interposition
 *
 * Only the calls that can BLOCK on another rank are wrapped (waits +
 * collectives). Posting calls (Isend/Irecv) never absorb skew. When the
 * window is not armed each wrapper is a single bool load + tail call.  */

static inline void ps_add(double dt) { s_wait[s_cur] += dt; s_calls[s_cur] += 1.0; }

#define PS_WRAP(call)                              \
    do {                                           \
        if (!s_armed) return P##call;              \
        const double _t0 = MPI_Wtime();            \
        const int _rc = P##call;                   \
        ps_add(MPI_Wtime() - _t0);                 \
        return _rc;                                \
    } while (0)

extern "C" {

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{ PS_WRAP(MPI_Wait(request, status)); }

int MPI_Waitall(int count, MPI_Request reqs[], MPI_Status stats[])
{ PS_WRAP(MPI_Waitall(count, reqs, stats)); }

int MPI_Barrier(MPI_Comm comm)
{ PS_WRAP(MPI_Barrier(comm)); }

int MPI_Allreduce(const void *sb, void *rb, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm)
{ PS_WRAP(MPI_Allreduce(sb, rb, count, dt, op, comm)); }

int MPI_Reduce(const void *sb, void *rb, int count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm)
{ PS_WRAP(MPI_Reduce(sb, rb, count, dt, op, root, comm)); }

int MPI_Bcast(void *buf, int count, MPI_Datatype dt, int root, MPI_Comm comm)
{ PS_WRAP(MPI_Bcast(buf, count, dt, root, comm)); }

int MPI_Alltoall(const void *sb, int sc, MPI_Datatype st, void *rb, int rc, MPI_Datatype rt, MPI_Comm comm)
{ PS_WRAP(MPI_Alltoall(sb, sc, st, rb, rc, rt, comm)); }

int MPI_Alltoallv(const void *sb, const int sc[], const int sd[], MPI_Datatype st,
                  void *rb, const int rc[], const int rd[], MPI_Datatype rt, MPI_Comm comm)
{ PS_WRAP(MPI_Alltoallv(sb, sc, sd, st, rb, rc, rd, rt, comm)); }

int MPI_Gather(const void *sb, int sc, MPI_Datatype st, void *rb, int rc, MPI_Datatype rt,
               int root, MPI_Comm comm)
{ PS_WRAP(MPI_Gather(sb, sc, st, rb, rc, rt, root, comm)); }

int MPI_Gatherv(const void *sb, int sc, MPI_Datatype st, void *rb, const int rc[],
                const int rd[], MPI_Datatype rt, int root, MPI_Comm comm)
{ PS_WRAP(MPI_Gatherv(sb, sc, st, rb, rc, rd, rt, root, comm)); }

int MPI_Scatterv(const void *sb, const int sc[], const int sd[], MPI_Datatype st,
                 void *rb, int rc, MPI_Datatype rt, int root, MPI_Comm comm)
{ PS_WRAP(MPI_Scatterv(sb, sc, sd, st, rb, rc, rt, root, comm)); }

int MPI_Allgather(const void *sb, int sc, MPI_Datatype st, void *rb, int rc, MPI_Datatype rt, MPI_Comm comm)
{ PS_WRAP(MPI_Allgather(sb, sc, st, rb, rc, rt, comm)); }

int MPI_Allgatherv(const void *sb, int sc, MPI_Datatype st, void *rb, const int rc[],
                   const int rd[], MPI_Datatype rt, MPI_Comm comm)
{ PS_WRAP(MPI_Allgatherv(sb, sc, st, rb, rc, rd, rt, comm)); }

} /* extern "C" */
