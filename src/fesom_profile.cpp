/* fesom_profile.cpp — see fesom_profile.hpp.
 *
 * M5.6: fence-bounded host+device phase timer (existing).
 * M5.11: Kokkos profiling-tools callbacks for per-kernel timing,
 *        deep_copy traffic counter, NVTX bridge. NO source-level wrapping
 *        needed — every Kokkos::parallel_{for,reduce,scan} that has a
 *        string label gets a per-kernel bucket automatically. */
#include "fesom_profile.hpp"

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Profiling.hpp>             // pushRegion/popRegion + tools API
#include <impl/Kokkos_Profiling_C_Interface.h>   // SpaceHandle + callback typedefs
#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace fesom_prof {

namespace {

/* Phase-bucket totals (M5.6). Keyed by name; values include the per-kernel
 * Kokkos labels added in M5.11 (e.g. "fct_init_zero") AND the existing phase
 * names from PMARK chains in fesom_step.cpp / fesom_ice.cpp.
 *
 * NB: when both phase markers and per-kernel callbacks are active, the per-
 * kernel buckets sum INSIDE the phase buckets (they're not disjoint); the
 * report shows both separately, sorted by share. Don't add them up. */
std::map<std::string, double> g_secs;
std::map<std::string, long>   g_calls;

/* M5.11 deep_copy counters. Atomic because Kokkos may fire the callback from
 * any thread; volumes are tiny so contention is irrelevant. The counters hold
 * no Kokkos state, so no cleanup-before-finalize is needed (unlike the M5.8
 * EVP static Kokkos::View trap). */
std::atomic<std::uint64_t> g_dc_count{0};
std::atomic<std::uint64_t> g_dc_bytes{0};

/* M5.11 per-kernel timing stack — populated by parallel_{for,reduce,scan}
 * begin/end callbacks. Single host thread per MPI rank in our codebase, so
 * thread_local is enough; we still take the LIFO discipline (kernels are
 * strictly nested per-thread by Kokkos). */
struct kernel_frame {
    std::string name;
    double      t0;
};
thread_local std::vector<kernel_frame> g_kstack;

/* parallel_for/reduce/scan begin: Kokkos passes a (name, devID, &kID); we
 * fence (drain prior async work), pushRegion (NVTX), stash the name+t0, and
 * return our own kID as the stack depth (Kokkos ignores its meaning). */
void kernel_begin_cb(const char *name, std::uint32_t /*devid*/,
                     std::uint64_t *kid)
{
    Kokkos::fence();
    const char *nm = (name && name[0]) ? name : "anon_kernel";
    Kokkos::Profiling::pushRegion(nm);
    g_kstack.push_back({std::string(nm), MPI_Wtime()});
    *kid = static_cast<std::uint64_t>(g_kstack.size() - 1);
}

/* parallel_for/reduce/scan end: fence (drain THIS kernel), accumulate
 * elapsed into the bucket, popRegion. Without the fence we'd measure
 * launch overhead, not the kernel's work time. */
void kernel_end_cb(std::uint64_t /*kid*/)
{
    Kokkos::fence();
    if (g_kstack.empty()) return;   // defensive — should be impossible
    const kernel_frame &f = g_kstack.back();
    g_secs [f.name] += MPI_Wtime() - f.t0;
    g_calls[f.name] += 1;
    Kokkos::Profiling::popRegion();
    g_kstack.pop_back();
}

/* deep_copy begin: count + bytes. Fires for ALL deep_copies (host<->device,
 * device<->device, host<->host); not filtered by direction in this rev. */
void dc_begin_cb(Kokkos_Profiling_SpaceHandle /*dst*/,
                 const char* /*dst_label*/, const void* /*dst_ptr*/,
                 Kokkos_Profiling_SpaceHandle /*src*/,
                 const char* /*src_label*/, const void* /*src_ptr*/,
                 std::uint64_t size)
{
    g_dc_count.fetch_add(1, std::memory_order_relaxed);
    g_dc_bytes.fetch_add(size, std::memory_order_relaxed);
}

} // anonymous namespace

bool enabled()
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_STEP_PROFILE");
        cached = (e && e[0]) ? 1 : 0;
    }
    return cached != 0;
}

double tic()
{
    if (!enabled()) return 0.0;
    Kokkos::fence();
    return MPI_Wtime();
}

void toc(const char *name, double t0)
{
    if (!enabled()) return;
    Kokkos::fence();
    g_secs [name] += MPI_Wtime() - t0;
    g_calls[name] += 1;
}

void push_region(const char *name)
{
    if (!enabled()) return;
    Kokkos::Profiling::pushRegion(name);
}

void pop_region()
{
    if (!enabled()) return;
    Kokkos::Profiling::popRegion();
}

void install_callbacks()
{
    if (!enabled()) return;
    namespace KTE = Kokkos::Tools::Experimental;
    KTE::set_begin_parallel_for_callback   (&kernel_begin_cb);
    KTE::set_end_parallel_for_callback     (&kernel_end_cb);
    KTE::set_begin_parallel_reduce_callback(&kernel_begin_cb);
    KTE::set_end_parallel_reduce_callback  (&kernel_end_cb);
    KTE::set_begin_parallel_scan_callback  (&kernel_begin_cb);
    KTE::set_end_parallel_scan_callback    (&kernel_end_cb);
    KTE::set_begin_deep_copy_callback      (&dc_begin_cb);
}

void reset()
{
    g_secs.clear();
    g_calls.clear();
    g_dc_count.store(0, std::memory_order_relaxed);
    g_dc_bytes.store(0, std::memory_order_relaxed);
}

void report(double loop_s, long timed_steps, int mype)
{
    if (!enabled() || mype != 0 || loop_s <= 0.0 || timed_steps <= 0) return;

    std::vector<std::pair<std::string, double>> v(g_secs.begin(), g_secs.end());
    std::sort(v.begin(), v.end(),
              [](const std::pair<std::string,double>&a, const std::pair<std::string,double>&b) {
                  return a.second > b.second;
              });

    std::printf("[fesom_prof] phase + kernel breakdown (rank0, sorted by %% of loop):\n");
    std::printf("[fesom_prof]   %-32s  %7s  %12s  %9s\n",
                "name", "%loop", "s/step", "calls/step");
    for (const auto &p : v) {
        std::printf("[fesom_prof]   %-32s  %6.2f%%  %10.4f s  %9.1f\n",
                    p.first.c_str(), 100.0 * p.second / loop_s,
                    p.second / (double)timed_steps,
                    (double)g_calls[p.first] / (double)timed_steps);
    }

    /* M5.11 deep_copy traffic — fires for ALL Kokkos::deep_copy calls. */
    const std::uint64_t dc_count = g_dc_count.load(std::memory_order_relaxed);
    const std::uint64_t dc_bytes = g_dc_bytes.load(std::memory_order_relaxed);
    if (dc_count > 0) {
        const double per_step_count = (double)dc_count / (double)timed_steps;
        const double per_step_mb    = (double)dc_bytes / (double)timed_steps / (1024.0 * 1024.0);
        std::printf("[fesom_prof]   deep_copy: %10.1f calls/step  %12.2f MB/step\n",
                    per_step_count, per_step_mb);
    }
    std::fflush(stdout);
}

} // namespace fesom_prof
