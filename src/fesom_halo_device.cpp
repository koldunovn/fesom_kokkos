/*
 * fesom_halo_device.cpp — on-device halo exchange (GPU-aware MPI), M5.1.
 *
 * See fesom_halo_device.hpp for the why. This is the CUDA-only device path:
 *   PACK   (device gather): send[g*stride+c] = field[(slist[g]-1)*stride + c]
 *   MPI    (device ptrs):   Irecv/Isend on send/recv DEVICE buffers (same tag,
 *                           PE order, MPI_DOUBLE counts as the host path)
 *   UNPACK (device scatter):field[(rlist[g]-1)*stride+c] = recv[g*stride+c]
 *
 * The buffer-offset collapse: send_buf is packed in slist order and (file
 * convention) sptr[0]==rptr[0]==1, so the per-PE MPI offset (sptr[k]-sptr[0])
 * *stride equals the running list index g*stride. Hence PACK/UNPACK are flat
 * gathers/scatters indexed by the global list position g — no per-PE bookkeeping
 * in the kernel. UNPACK is race-free (each halo node appears once in rlist), so
 * no atomics — Serial would be bit-identical too (but Serial never takes this
 * path; it is CUDA-only).
 */
#include "fesom_halo_device.hpp"
#include "fesom_speed.hpp"   // M7: FESOM_SPEED_* levers (NOFENCE2, SYNCSTATS)

#include <cstdlib>   // getenv
#include <cstdio>    // M5.17 halo-mpi-prof report
#include <mpi.h>     // M5.17 barrier-isolation (MPI_Barrier/Waitall/Reduce/Wtime)

bool fesom_halo_device_active()
{
#ifdef KOKKOS_ENABLE_CUDA
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_HOST_HALO");
        cached = (e && e[0] == '1') ? 0 : 1;   // default ON; =1 forces legacy host path
    }
    return cached != 0;
#else
    return false;
#endif
}

/* ======================================================================= *
 * M5.17 halo MPI profiler — barrier-isolation split of the per-step halo
 * wait into LOAD-IMBALANCE (rank arrival skew) vs COMM (messages in flight).
 *
 *   FESOM_HALO_BARRIER=1  inserts an MPI_Barrier before EVERY halo exchange,
 *     so the per-rank arrival skew is absorbed into the barrier and the
 *     following MPI_Waitall measures pure comm. (Off in production.)
 *   FESOM_HALO_MPI_PROF=1 (implied by BARRIER) times Barrier + Waitall and
 *     reports across-rank min/mean/max at the end of the timed loop.
 *
 * Both gated + cached -> zero production cost. Defined here (always compiled)
 * and used by BOTH the device path (fesom_halo_exchange_device, below) and the
 * host path (fesom_halo_exchange, fesom_halo.cpp) so the split covers all
 * per-step comm and matches the nsys MPI_Waitall total.
 * ======================================================================= */
static int    s_halo_barrier = -1;   // FESOM_HALO_BARRIER cached
static int    s_halo_mpiprof = -1;   // FESOM_HALO_MPI_PROF | BARRIER cached
static double s_barrier_s    = 0.0;  // this-rank sum of MPI_Barrier wall (s)
static double s_waitall_s    = 0.0;  // this-rank sum of MPI_Waitall wall (s)
static long   s_waitall_n    = 0;    // this-rank Waitall call count
static double s_halo_bytes   = 0.0;  // this-rank send+recv bytes

static inline bool halo_barrier_on()
{
    if (s_halo_barrier < 0) { const char *e = getenv("FESOM_HALO_BARRIER"); s_halo_barrier = (e && e[0]=='1') ? 1 : 0; }
    return s_halo_barrier != 0;
}
static inline bool halo_mpiprof_on()
{
    if (s_halo_mpiprof < 0) { const char *e = getenv("FESOM_HALO_MPI_PROF"); s_halo_mpiprof = ((e && e[0]=='1') || halo_barrier_on()) ? 1 : 0; }
    return s_halo_mpiprof != 0;
}

void fesom_halo_prof_barrier(fesom_partit *p)
{
    if (!p || p->npes <= 1 || !halo_barrier_on()) return;
    double t0 = MPI_Wtime();
    MPI_Barrier(p->MPI_COMM_FESOM);
    s_barrier_s += MPI_Wtime() - t0;
}

void fesom_halo_prof_waitall(int n_reqs, MPI_Request *reqs)
{
    if (!halo_mpiprof_on()) { MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE); return; }
    double t0 = MPI_Wtime();
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    s_waitall_s += MPI_Wtime() - t0;
    s_waitall_n += 1;
}

void fesom_halo_prof_bytes(double bytes)
{
    if (halo_mpiprof_on()) s_halo_bytes += bytes;
}

void fesom_halo_mpi_report(int timed_steps, fesom_partit *p)
{
    if (!p || !halo_mpiprof_on() || timed_steps <= 0) return;   // ALL ranks evaluate identically
    double loc[4] = { s_barrier_s, s_waitall_s, s_halo_bytes, (double)s_waitall_n };
    double vmin[4], vmax[4], vsum[4];
    MPI_Reduce(loc, vmin, 4, MPI_DOUBLE, MPI_MIN, 0, p->MPI_COMM_FESOM);
    MPI_Reduce(loc, vmax, 4, MPI_DOUBLE, MPI_MAX, 0, p->MPI_COMM_FESOM);
    MPI_Reduce(loc, vsum, 4, MPI_DOUBLE, MPI_SUM, 0, p->MPI_COMM_FESOM);
    if (p->mype != 0) return;
    const double ts = (double)timed_steps;
    const double inv_np = 1.0 / (double)p->npes;
    printf("[halo-mpi-prof] steps=%d ranks=%d  barrier=%s  (per-step s, across-rank min/mean/max)\n",
           timed_steps, p->npes, halo_barrier_on() ? "ON" : "off");
    printf("[halo-mpi-prof]   MPI_Waitall : %.4f / %.4f / %.4f   (calls/step %.0f, halo %.1f MB/step/rank)\n",
           vmin[1]/ts, vsum[1]*inv_np/ts, vmax[1]/ts, vsum[3]*inv_np/ts, vsum[2]*inv_np/ts/1e6);
    if (halo_barrier_on()) {
        printf("[halo-mpi-prof]   MPI_Barrier : %.4f / %.4f / %.4f   (= absorbed load-imbalance)\n",
               vmin[0]/ts, vsum[0]*inv_np/ts, vmax[0]/ts);
        const double imb = vsum[0]*inv_np/ts, comm = vsum[1]*inv_np/ts, tot = imb + comm + 1e-30;
        printf("[halo-mpi-prof]   SPLIT (of barrier+waitall): imbalance %.4f s/step (%.0f%%)  |  comm %.4f s/step (%.0f%%)\n",
               imb, 100.0*imb/tot, comm, 100.0*comm/tot);
    } else {
        printf("[halo-mpi-prof]   (re-run with FESOM_HALO_BARRIER=1 to split Waitall into imbalance vs comm)\n");
    }
    fflush(stdout);
}

/* ======================================================================= *
 * M7 Task 0.3/1.1 — FESOM_SPEED_SYNCSTATS: per-step device-sync counters.
 *
 * Counts what the nsys stall budget measures, from inside the model, so a lever's
 * effect on the SYNC COUNT is visible without re-tracing: exchanges + fields moved
 * by kind, pack/unpack launches, and the three fence classes (pre-MPI pack fence,
 * post-unpack fence executed, post-unpack fence SKIPPED by NOFENCE2, realloc guard).
 *
 * Gated + cached -> zero production cost. Defined in the always-compiled section so
 * the report links on Serial builds too (where the device halo never runs and every
 * counter stays 0).
 * ======================================================================= */
static long s_ss_exch[5]     = {0, 0, 0, 0, 0};   // exchange calls, by kind
static long s_ss_fields[5]   = {0, 0, 0, 0, 0};   // fields moved, by kind (deviceN adds nf)
static long s_ss_fence_pre   = 0;                 // pre-MPI pack fences (MANDATORY)
static long s_ss_fence_post  = 0;                 // post-unpack fences EXECUTED
static long s_ss_fence_skip  = 0;                 // post-unpack fences SKIPPED (NOFENCE2)
static long s_ss_fence_grow  = 0;                 // realloc-guard fences (warmup only)
static long s_ss_pack        = 0;                 // pack kernel launches
static long s_ss_unpack      = 0;                 // unpack kernel launches

/* Opt-in ONLY (fesom_speed_on_exp): a diagnostic counter must not ride the
 * FESOM_SPEED master switch — asking for the blessed perf set is not asking for
 * a per-step report. Set FESOM_SPEED_SYNCSTATS=1 explicitly. */
static inline bool syncstats_on()
{
    static int c = -1;
    return fesom_speed_on_exp("SYNCSTATS", &c);
}

void fesom_halo_syncstats_report(int timed_steps, fesom_partit *p)
{
    if (!p || timed_steps <= 0 || !syncstats_on()) return;   // ALL ranks evaluate identically
    long exch = 0, flds = 0;
    for (int k = 0; k < 5; ++k) { exch += s_ss_exch[k]; flds += s_ss_fields[k]; }
    double loc[8] = { (double)exch, (double)flds, (double)s_ss_pack, (double)s_ss_unpack,
                      (double)s_ss_fence_pre, (double)s_ss_fence_post,
                      (double)s_ss_fence_skip, (double)s_ss_fence_grow };
    double vsum[8], vmax[8];
    MPI_Reduce(loc, vsum, 8, MPI_DOUBLE, MPI_SUM, 0, p->MPI_COMM_FESOM);
    MPI_Reduce(loc, vmax, 8, MPI_DOUBLE, MPI_MAX, 0, p->MPI_COMM_FESOM);
    if (p->mype != 0) return;
    /* Counters run from program start, so normalise by the WHOLE run, not the timed window:
     * per-step here means "per step of the whole loop", close enough to rank the levers. */
    const double inv = 1.0 / ((double)p->npes * (double)timed_steps);
    printf("[halo-syncstats] steps=%d ranks=%d (per step, rank-mean)\n", timed_steps, p->npes);
    printf("[halo-syncstats]   exchanges %.1f  fields %.1f  pack %.1f  unpack %.1f\n",
           vsum[0]*inv, vsum[1]*inv, vsum[2]*inv, vsum[3]*inv);
    printf("[halo-syncstats]   FENCES: pre-MPI %.1f (mandatory)  post-unpack %.1f  "
           "SKIPPED %.1f (NOFENCE2)  realloc-guard %.1f\n",
           vsum[4]*inv, vsum[5]*inv, vsum[6]*inv, vsum[7]*inv);
    fflush(stdout);
}

#ifndef KOKKOS_ENABLE_CUDA
void fesom_halo_device_free() { /* no device Views on host backends */ }
#else  // ====================== CUDA device path ===========================

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <string>
#include <vector>
#include <cmath>     // FESOM_HALO_SELFCHECK: std::fabs
#include <cstdio>    // FESOM_HALO_SELFCHECK: std::printf

#include "fesom_types.h"

namespace {

// Per-kind device-resident comm lists + reusable send/recv buffers + MPI reqs.
struct DevHaloScratch {
    Kokkos::View<int*>    slist_d;   // device copy of cs->slist (1-based local idx)
    Kokkos::View<int*>    rlist_d;   // device copy of cs->rlist (1-based local idx)
    Kokkos::View<double*> send_d;    // device send buffer (grown on demand)
    Kokkos::View<double*> recv_d;    // device recv buffer
    bool                  built = false;
    std::vector<MPI_Request> reqs;
};
DevHaloScratch g_dev[5];   // indexed by fesom_halo_kind (5 entries)

fesom_com_struct *dev_get_com(fesom_partit *p, fesom_halo_kind kind)
{
    switch (kind) {
    case FESOM_HALO_NOD2D:
    case FESOM_HALO_NOD3D:        return &p->com_nod2D;
    case FESOM_HALO_ELEM2D:
    case FESOM_HALO_ELEM3D:       return &p->com_elem2D;
    case FESOM_HALO_ELEM2D_FULL:  return &p->com_elem2D_full;
    default:                      return nullptr;
    }
}

// One-shot: copy cs->slist / cs->rlist to device (the M1.4 / M4.2-a push idiom).
void build_lists(DevHaloScratch &s, fesom_com_struct *cs)
{
    if (s.built) return;
    // File convention: cumulative offsets are 1-based and start at 1. The flat
    // buffer-offset collapse depends on it — assert loudly if a mesh violates it.
    FESOM_CHECK(cs->sptr[0] == 1 && cs->rptr[0] == 1,
                "fesom_halo_device: expected sptr[0]==rptr[0]==1 (got %d,%d) — "
                "buffer-offset collapse invalid", cs->sptr[0], cs->rptr[0]);

    int send_count = cs->sptr[cs->sPEnum] - cs->sptr[0];
    int recv_count = cs->rptr[cs->rPEnum] - cs->rptr[0];
    if (send_count < 0) send_count = 0;
    if (recv_count < 0) recv_count = 0;

    s.slist_d = Kokkos::View<int*>("halo.slist_d", (size_t)send_count);
    s.rlist_d = Kokkos::View<int*>("halo.rlist_d", (size_t)recv_count);
    auto h_s = Kokkos::create_mirror_view(s.slist_d);
    auto h_r = Kokkos::create_mirror_view(s.rlist_d);
    for (int i = 0; i < send_count; ++i) h_s(i) = cs->slist[i];
    for (int i = 0; i < recv_count; ++i) h_r(i) = cs->rlist[i];
    Kokkos::deep_copy(s.slist_d, h_s);
    Kokkos::deep_copy(s.rlist_d, h_r);
    s.built = true;
}

/* ======================================================================= *
 * M7 Task 1.1 — FESOM_SPEED_NOFENCE2: remove the POST-UNPACK halo fence.
 *
 * The exchange is  PACK -> fence[A] -> MPI -> UNPACK -> fence[B] -> modify_device().
 * fence[A] (:251/:344/:439) MUST STAY: MPI reads the device send buffer from a
 * host-posted call, which is not ordered against the Kokkos stream.
 * fence[B] (:286/:377/:475) is what this lever drops. Audit, 2026-07-14, all three
 * exchange paths (device / device2 / deviceN):
 *
 *  1. CONSUMERS. Every consumer of the unpacked halo is a Kokkos kernel on the DEFAULT
 *     execution space = one CUDA stream. Kernels on a stream run in issue order, so the
 *     unpack is already ordered before every downstream device reader. fence[B] only ever
 *     ordered device work against a HOST reader.
 *
 *  2. HOST READERS. There are none mid-step. Every host read of a Field goes through
 *     Field::sync_host() -> DualView::sync_host() -> Kokkos::deep_copy, and a deep_copy
 *     with no execution-space argument fences the default space itself. A raw host read of
 *     a device-authoritative Field is trapped by -DFESOM_KK_SYNCCHECK. src/ contains no
 *     cudaMemcpyAsync. f.modify_device() is host-side bookkeeping only.
 *
 *  3. BUFFER REUSE — the one that actually bites. A LATER exchange's MPI_Irecv writes
 *     s.recv_d, the same buffer THIS exchange's unpack kernel is reading, and MPI's device
 *     writes are not ordered against the Kokkos stream. What saves us is that fence[A] is
 *     UNCONDITIONAL: it sits OUTSIDE the `if (send_count > 0)` guard in all three paths, so
 *     every MPI_Irecv into recv_d is preceded, in its own function, by a full device fence
 *     that drains the previous unpack. The early-return paths post no MPI, so they cannot
 *     break it either. (The next exchange's PACK also overwrites s.send_d, but a pack is a
 *     kernel on the same stream as the unpack — stream order already covers that.)
 *
 *     INVARIANT — keep it true or this lever becomes a data race:
 *       every MPI_Irecv on a device buffer is preceded by an unconditional Kokkos::fence()
 *       in the same function.
 *
 *  4. REALLOC — see grow() below.
 *
 *  ⚠️ Premise 1 is FALSE the instant anything runs on a second stream. Task 4.1 (ICELAG)
 *  does exactly that: re-audit and re-run compute-sanitizer racecheck before landing it.
 * ======================================================================= */
inline bool halo_nofence2()
{
    static int c = -1;
    return fesom_speed_on("NOFENCE2", &c);
}

inline void halo_fence_post_unpack()
{
    if (halo_nofence2()) { if (syncstats_on()) ++s_ss_fence_skip; return; }
    Kokkos::fence();
    if (syncstats_on()) ++s_ss_fence_post;
}

inline void halo_fence_pre_mpi()
{
    Kokkos::fence();   // MANDATORY — see (3) above. Never gate this.
    if (syncstats_on()) ++s_ss_fence_pre;
}

inline void grow(Kokkos::View<double*> &v, size_t need, const char *label)
{
    if (v.extent(0) < need) {
        /* Audit item (4). With fence[B] gone, a PREVIOUS exchange's unpack kernel may still
         * be reading this very buffer when the assignment below drops its last reference and
         * frees it. Whether that is safe depends on which allocator Kokkos picked: plain
         * cudaFree synchronises the device, but this build uses the ASYNC pool
         * (cudaMallocAsync/cudaFreeAsync both appear in the nsys trace), whose free is only
         * stream-ordered. That happens to be safe today — same stream — but it is exactly the
         * "true today, silent tomorrow" reasoning L67 warns about. Don't depend on allocator
         * internals: fence explicitly. Realloc branch only, and the buffers hit their
         * high-water mark during warmup (379 reallocs in a whole 35-step NG5 run), so this
         * costs nothing in steady state. */
        if (halo_nofence2()) { Kokkos::fence(); if (syncstats_on()) ++s_ss_fence_grow; }
        v = Kokkos::View<double*>(std::string(label), need);
    }
}

} // namespace

void fesom_halo_device_free()
{
    for (int i = 0; i < 5; ++i) {
        g_dev[i].slist_d = Kokkos::View<int*>();
        g_dev[i].rlist_d = Kokkos::View<int*>();
        g_dev[i].send_d  = Kokkos::View<double*>();
        g_dev[i].recv_d  = Kokkos::View<double*>();
        g_dev[i].reqs.clear();
        g_dev[i].built = false;
    }
}

void fesom_halo_exchange_device(fesom::Field   &f,
                                fesom_halo_kind kind,
                                int             n_levels,
                                int             n_components,
                                fesom_partit   *p,
                                std::size_t     base_off)
{
    if (!p || p->npes == 1) { f.modify_device(); return; }
    // M5.17: barrier BEFORE the per-rank empty-comm early-return — all npes>1 ranks
    // call this exchange in lockstep (fesom_halo_field is rank-uniform), so the
    // collective is uniform; placing it after the rPEnum==0 return could deadlock a
    // rank with empty comm for this kind. Gated FESOM_HALO_BARRIER; absorbs arrival skew.
    fesom_halo_prof_barrier(p);
    fesom_com_struct *cs = dev_get_com(p, kind);
    if (!cs) FESOM_DIE("fesom_halo_device: unknown kind %d", (int)kind);
    if (cs->rPEnum == 0 && cs->sPEnum == 0) { f.modify_device(); return; }

    DevHaloScratch &s = g_dev[(int)kind];
    build_lists(s, cs);

    const int    stride     = n_levels * n_components;
    if (syncstats_on()) { ++s_ss_exch[(int)kind]; s_ss_fields[(int)kind] += 1; }
    const int    send_count = cs->sptr[cs->sPEnum] - cs->sptr[0];
    const int    recv_count = cs->rptr[cs->rPEnum] - cs->rptr[0];
    const size_t send_total = (size_t)send_count * stride;
    const size_t recv_total = (size_t)recv_count * stride;
    fesom_halo_prof_bytes((double)(send_total + recv_total) * sizeof(double));   // M5.17

    grow(s.send_d, send_total, "halo.send_d");
    grow(s.recv_d, recv_total, "halo.recv_d");

    auto field = f.d();          // RANK-1 flat LayoutRight device view of the field
    auto send  = s.send_d;
    auto recv  = s.recv_d;
    auto slist = s.slist_d;
    auto rlist = s.rlist_d;
    const int  st = stride;
    const long bo = (long)base_off;   // slab offset into the field (0 for whole-field)

    // PACK: gather slist-indexed entries into the send buffer (slist order).
    if (send_count > 0) {
        if (syncstats_on()) ++s_ss_pack;
        Kokkos::parallel_for("fesom_halo_pack", Kokkos::RangePolicy<>(0, send_count),
            KOKKOS_LAMBDA(const int g) {
                const long src = bo + (long)(slist(g) - 1) * st;
                const long dst = (long)g * st;
                for (int c = 0; c < st; ++c) send(dst + c) = field(src + c);
            });
    }
    halo_fence_pre_mpi();   // MANDATORY: PACK must complete before MPI reads the device send buffer

    // MPI — identical loop structure to the host fesom_halo_exchange, but the
    // buffers are DEVICE pointers (GPU-aware MPI). Same tag / PE order / counts.
    const int tag    = 2000 + (int)kind;
    const int needed = cs->rPEnum + cs->sPEnum;
    if ((int)s.reqs.size() < needed) s.reqs.resize(needed);
    int     nreq     = 0;
    double *send_ptr = send.data();
    double *recv_ptr = recv.data();

    for (int k = 0; k < cs->rPEnum; ++k) {
        const int    nseg  = cs->rptr[k + 1] - cs->rptr[k];
        const size_t off   = (size_t)(cs->rptr[k] - cs->rptr[0]) * stride;
        MPI_Irecv(recv_ptr + off, nseg * stride, MPI_DOUBLE, cs->rPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    for (int k = 0; k < cs->sPEnum; ++k) {
        const int    nseg  = cs->sptr[k + 1] - cs->sptr[k];
        const size_t off   = (size_t)(cs->sptr[k] - cs->sptr[0]) * stride;
        MPI_Isend(send_ptr + off, nseg * stride, MPI_DOUBLE, cs->sPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    fesom_halo_prof_waitall(nreq, s.reqs.data());   // M5.17: timed (gated FESOM_HALO_MPI_PROF)

    // UNPACK: scatter the recv buffer into the halo slots (race-free — each
    // rlist entry is a distinct halo node, so no atomics).
    if (recv_count > 0) {
        if (syncstats_on()) ++s_ss_unpack;
        Kokkos::parallel_for("fesom_halo_unpack", Kokkos::RangePolicy<>(0, recv_count),
            KOKKOS_LAMBDA(const int g) {
                const long dst = bo + (long)(rlist(g) - 1) * st;
                const long src = (long)g * st;
                for (int c = 0; c < st; ++c) field(dst + c) = recv(src + c);
            });
    }
    halo_fence_post_unpack();   // M7 NOFENCE2 target

    f.modify_device();   // device authoritative: owned (unchanged) + halo (new)
}

// M5.23 (L1): two-field fused exchange. Co-packs f0,f1 into ONE message/neighbour with per-halo-node
// stride 2*stride ([f0(stride) f1(stride)]). The flat buffer-offset collapse is unchanged (every
// per-entry block is just 2x wider), so PACK/UNPACK stay flat gathers/scatters indexed by g. The
// bytes that land in each field's halo slots are identical to two separate fesom_halo_exchange_device
// calls — message-count reduction only (no new arithmetic, no atomics; UNPACK race-free as before).
void fesom_halo_exchange_device2(fesom::Field   &f0,
                                 fesom::Field   &f1,
                                 fesom_halo_kind kind,
                                 int             n_levels,
                                 int             n_components,
                                 fesom_partit   *p,
                                 std::size_t     base_off)
{
    if (!p || p->npes == 1) { f0.modify_device(); f1.modify_device(); return; }
    fesom_halo_prof_barrier(p);   // M5.17: same placement as the single-field path (before early-return)
    fesom_com_struct *cs = dev_get_com(p, kind);
    if (!cs) FESOM_DIE("fesom_halo_device2: unknown kind %d", (int)kind);
    if (cs->rPEnum == 0 && cs->sPEnum == 0) { f0.modify_device(); f1.modify_device(); return; }

    DevHaloScratch &s = g_dev[(int)kind];
    build_lists(s, cs);

    const int    fstride    = n_levels * n_components;   // per-field per-node block
    const int    stride     = 2 * fstride;               // co-packed: [f0 | f1] per halo node
    if (syncstats_on()) { ++s_ss_exch[(int)kind]; s_ss_fields[(int)kind] += 2; }
    const int    send_count = cs->sptr[cs->sPEnum] - cs->sptr[0];
    const int    recv_count = cs->rptr[cs->rPEnum] - cs->rptr[0];
    const size_t send_total = (size_t)send_count * stride;
    const size_t recv_total = (size_t)recv_count * stride;
    fesom_halo_prof_bytes((double)(send_total + recv_total) * sizeof(double));   // M5.17

    grow(s.send_d, send_total, "halo.send_d");
    grow(s.recv_d, recv_total, "halo.recv_d");

    auto field0 = f0.d();
    auto field1 = f1.d();
    auto send   = s.send_d;
    auto recv   = s.recv_d;
    auto slist  = s.slist_d;
    auto rlist  = s.rlist_d;
    const int  fs = fstride;
    const int  st = stride;
    const long bo = (long)base_off;

    // PACK: gather both fields' owned entries into one interleaved-per-node send buffer.
    if (send_count > 0) {
        if (syncstats_on()) ++s_ss_pack;
        Kokkos::parallel_for("fesom_halo_pack2", Kokkos::RangePolicy<>(0, send_count),
            KOKKOS_LAMBDA(const int g) {
                const long src = bo + (long)(slist(g) - 1) * fs;   // per-field stride into each field
                const long dst = (long)g * st;                     // 2*fs-wide block in the buffer
                for (int c = 0; c < fs; ++c) send(dst + c)      = field0(src + c);
                for (int c = 0; c < fs; ++c) send(dst + fs + c) = field1(src + c);
            });
    }
    halo_fence_pre_mpi();   // MANDATORY (see the NOFENCE2 audit, item 3)

    const int tag    = 2000 + (int)kind;   // same tag family; a fused msg is symmetric on both ranks
    const int needed = cs->rPEnum + cs->sPEnum;
    if ((int)s.reqs.size() < needed) s.reqs.resize(needed);
    int     nreq     = 0;
    double *send_ptr = send.data();
    double *recv_ptr = recv.data();

    for (int k = 0; k < cs->rPEnum; ++k) {
        const int    nseg = cs->rptr[k + 1] - cs->rptr[k];
        const size_t off  = (size_t)(cs->rptr[k] - cs->rptr[0]) * stride;
        MPI_Irecv(recv_ptr + off, nseg * stride, MPI_DOUBLE, cs->rPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    for (int k = 0; k < cs->sPEnum; ++k) {
        const int    nseg = cs->sptr[k + 1] - cs->sptr[k];
        const size_t off  = (size_t)(cs->sptr[k] - cs->sptr[0]) * stride;
        MPI_Isend(send_ptr + off, nseg * stride, MPI_DOUBLE, cs->sPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    fesom_halo_prof_waitall(nreq, s.reqs.data());

    // UNPACK: scatter each field's half of the per-node block back (race-free, no atomics).
    if (recv_count > 0) {
        if (syncstats_on()) ++s_ss_unpack;
        Kokkos::parallel_for("fesom_halo_unpack2", Kokkos::RangePolicy<>(0, recv_count),
            KOKKOS_LAMBDA(const int g) {
                const long dst = bo + (long)(rlist(g) - 1) * fs;
                const long src = (long)g * st;
                for (int c = 0; c < fs; ++c) field0(dst + c) = recv(src + c);
                for (int c = 0; c < fs; ++c) field1(dst + c) = recv(src + fs + c);
            });
    }
    halo_fence_post_unpack();   // M7 NOFENCE2 target

    f0.modify_device();
    f1.modify_device();
}

// M5.23 (fieldN): N-field fused exchange — generalizes device2 to any N same-kind/same-stride
// fields in ONE message per neighbour. The per-halo-node block is [f0(fs) f1(fs) … f(N-1)(fs)],
// stride N*fs. To sidestep the device-array-of-Views problem (a KOKKOS_LAMBDA can't deref a HOST
// Field**), the N pack/unpack launches loop on the HOST, each capturing ONE field's view
// fi=fields[i]->d(). The N pack kernels write DISJOINT buffer slots (field i → slot i*fs), so they
// need NO fence between them — one fence before MPI; likewise N disjoint unpack kernels, one fence
// after. The bytes landing in each field's halo are byte-identical to N separate
// fesom_halo_exchange_device calls — pure message-count reduction (no new arithmetic, no atomics;
// unpack race-free). nf==2 reproduces device2's layout exactly. All N fields MUST share
// kind / n_levels / n_components / base_off.
void fesom_halo_exchange_deviceN(fesom::Field *const *fields, int nf,
                                 fesom_halo_kind kind, int n_levels, int n_components,
                                 fesom_partit *p, std::size_t base_off)
{
    if (nf <= 0) return;
    if (!p || p->npes == 1) { for (int i = 0; i < nf; ++i) fields[i]->modify_device(); return; }
    fesom_halo_prof_barrier(p);   // same placement as the single/two-field path (before early-return)
    fesom_com_struct *cs = dev_get_com(p, kind);
    if (!cs) FESOM_DIE("fesom_halo_deviceN: unknown kind %d", (int)kind);
    if (cs->rPEnum == 0 && cs->sPEnum == 0) { for (int i = 0; i < nf; ++i) fields[i]->modify_device(); return; }

    DevHaloScratch &s = g_dev[(int)kind];
    build_lists(s, cs);

    const int    fstride    = n_levels * n_components;   // per-field per-node block
    const int    stride     = nf * fstride;              // co-packed: [f0|f1|…|f(nf-1)] per halo node
    if (syncstats_on()) { ++s_ss_exch[(int)kind]; s_ss_fields[(int)kind] += nf; }
    const int    send_count = cs->sptr[cs->sPEnum] - cs->sptr[0];
    const int    recv_count = cs->rptr[cs->rPEnum] - cs->rptr[0];
    const size_t send_total = (size_t)send_count * stride;
    const size_t recv_total = (size_t)recv_count * stride;
    fesom_halo_prof_bytes((double)(send_total + recv_total) * sizeof(double));

    grow(s.send_d, send_total, "halo.send_d");
    grow(s.recv_d, recv_total, "halo.recv_d");

    auto send  = s.send_d;
    auto recv  = s.recv_d;
    auto slist = s.slist_d;
    auto rlist = s.rlist_d;
    const int  fs = fstride;
    const int  st = stride;
    const long bo = (long)base_off;

    // PACK: N disjoint pack kernels (field i → buffer slot i*fs). Disjoint writes → no fence between.
    if (send_count > 0) {
        for (int i = 0; i < nf; ++i) {
            auto      fi   = fields[i]->d();
            const int slot = i * fs;
            if (syncstats_on()) ++s_ss_pack;
        Kokkos::parallel_for("fesom_halo_packN", Kokkos::RangePolicy<>(0, send_count),
                KOKKOS_LAMBDA(const int g) {
                    const long src = bo + (long)(slist(g) - 1) * fs;
                    const long dst = (long)g * st + slot;
                    for (int c = 0; c < fs; ++c) send(dst + c) = fi(src + c);
                });
        }
    }
    halo_fence_pre_mpi();   // MANDATORY (see the NOFENCE2 audit, item 3)

    const int tag    = 2000 + (int)kind;
    const int needed = cs->rPEnum + cs->sPEnum;
    if ((int)s.reqs.size() < needed) s.reqs.resize(needed);
    int     nreq     = 0;
    double *send_ptr = send.data();
    double *recv_ptr = recv.data();

    for (int k = 0; k < cs->rPEnum; ++k) {
        const int    nseg = cs->rptr[k + 1] - cs->rptr[k];
        const size_t off  = (size_t)(cs->rptr[k] - cs->rptr[0]) * stride;
        MPI_Irecv(recv_ptr + off, nseg * stride, MPI_DOUBLE, cs->rPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    for (int k = 0; k < cs->sPEnum; ++k) {
        const int    nseg = cs->sptr[k + 1] - cs->sptr[k];
        const size_t off  = (size_t)(cs->sptr[k] - cs->sptr[0]) * stride;
        MPI_Isend(send_ptr + off, nseg * stride, MPI_DOUBLE, cs->sPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    fesom_halo_prof_waitall(nreq, s.reqs.data());

    // UNPACK: N disjoint unpack kernels (field i reads slot i*fs → its own halo). Race-free, no atomics.
    if (recv_count > 0) {
        for (int i = 0; i < nf; ++i) {
            auto      fi   = fields[i]->d();
            const int slot = i * fs;
            if (syncstats_on()) ++s_ss_unpack;
        Kokkos::parallel_for("fesom_halo_unpackN", Kokkos::RangePolicy<>(0, recv_count),
                KOKKOS_LAMBDA(const int g) {
                    const long dst = bo + (long)(rlist(g) - 1) * fs;
                    const long src = (long)g * st + slot;
                    for (int c = 0; c < fs; ++c) fi(dst + c) = recv(src + c);
                });
        }
    }
    halo_fence_post_unpack();   // M7 NOFENCE2 target

    for (int i = 0; i < nf; ++i) fields[i]->modify_device();
}

// FESOM_HALO_SELFCHECK: run the device exchange AND the host exchange on the SAME owned data,
// compare the halo. Pinpoints a device-transport bug (kind + magnitude) isolated from the physics.
// (Owned data is identical to the host path — the dispatcher's modify_device() then our sync_host()
// pull the device-current owned to host; we host-exchange a COPY, device-exchange the Field, diff.)
void fesom_halo_device_selfcheck(fesom::Field &f, fesom_halo_kind kind,
                                 int n_levels, int n_components,
                                 fesom_partit *p, std::size_t base_off)
{
    const int stride = n_levels * n_components;
    const std::size_t Nf = f.size();

    f.sync_host();                                   // device owned -> host (auth Device->Synced)
    std::vector<double> ref(f.h(), f.h() + Nf);      // copy of owned (+ stale halo)
    fesom_halo_exchange(ref.data() + base_off, kind, n_levels, n_components, p);  // HOST reference

    f.modify_device();                               // device owned is still current
    fesom_halo_exchange_device(f, kind, n_levels, n_components, p, base_off);     // DEVICE under test
    f.sync_host();

    fesom_com_struct *cs = dev_get_com(p, kind);
    const int rcount = cs->rptr[cs->rPEnum] - cs->rptr[0];
    int    fails = 0;
    double maxd  = 0.0;
    for (int g = 0; g < rcount; ++g) {
        const long idx = (long)base_off + (long)(cs->rlist[g] - 1) * stride;
        for (int c = 0; c < stride; ++c) {
            double d = std::fabs(f.h()[idx + c] - ref[idx + c]);
            if (d > maxd) maxd = d;
            if (d > 1e-9) ++fails;
        }
    }
    long gfails = 0, lf = fails;
    MPI_Allreduce(&lf, &gfails, 1, MPI_LONG, MPI_SUM, p->MPI_COMM_FESOM);
    double gmax = 0.0;
    MPI_Allreduce(&maxd, &gmax, 1, MPI_DOUBLE, MPI_MAX, p->MPI_COMM_FESOM);
    static const char *knames[5] = {"NOD2D","NOD3D","ELEM2D","ELEM3D","ELEM2D_FULL"};
    if (p->mype == 0 && (gfails > 0 || gmax > 0.0))
        std::printf("[devhalo-selfcheck] kind=%s nl=%d nc=%d base=%zu  rPEnum=%d  "
                    "halo mismatches=%ld  max|dev-host|=%.3e\n",
                    knames[(int)kind], n_levels, n_components, base_off,
                    cs->rPEnum, gfails, gmax);
}

#endif // KOKKOS_ENABLE_CUDA
