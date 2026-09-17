// kp_ktime.cpp — minimal Kokkos Tools connector: per-kernel device time.
//
// INVESTIGATION ONLY. Not part of the xmach deliverable, never loaded by a measurement run.
//
// Why this exists: MN5's ACC nodes refuse CUPTI-based profiling (nsys fails to attach and the
// rank-0-only wrapper then deadlocks the other ranks), so the nsys-vs-rocprof comparison the
// LUMI profile was meant to be diffed against cannot be produced. This connector uses the
// Kokkos profiling hooks instead, which are plain callbacks in user space: no driver
// privileges, no vendor profiler, and — the real point — the SAME code path on HIP and CUDA,
// so the two machines are measured by one instrument rather than two different ones.
//
// METHOD / CAVEAT: Kokkos kernel launches are asynchronous, so the begin/end callbacks by
// themselves would time the launch, not the kernel. This tool therefore fences the device on
// both sides of every kernel. That makes each kernel's time real and comparable, but it
// serialises the run: total wall time is inflated and any compute/halo overlap is destroyed.
// Use these numbers as a RANKING and a SHARE of device time, never as an s/step figure.
//
// Build (per machine, against that machine's backend):
//   CUDA: nvcc -O2 -shared -Xcompiler -fPIC -DKP_CUDA -o kp_ktime.so kp_ktime.cpp
//   HIP : hipcc -O2 -shared -fPIC -DKP_HIP  -o kp_ktime.so kp_ktime.cpp
// Use:
//   export KOKKOS_TOOLS_LIBS=/path/kp_ktime.so
//   export KP_KTIME_OUT=/path/prefix      # <prefix>.rank<N>.csv, rank 0 unless KP_KTIME_ALL=1
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <chrono>

#if defined(KP_CUDA)
#include <cuda_runtime.h>
static inline void kp_sync() { cudaDeviceSynchronize(); }
static const char* kp_backend = "CUDA";
#elif defined(KP_HIP)
#include <hip/hip_runtime.h>
static inline void kp_sync() { hipDeviceSynchronize(); }
static const char* kp_backend = "HIP";
#else
static inline void kp_sync() {}
static const char* kp_backend = "HOST";
#endif

namespace {

using clk = std::chrono::steady_clock;

struct Acc { uint64_t calls = 0; double ns = 0.0; };

struct Frame { std::string name; clk::time_point t0; };

std::map<std::string, Acc> g_acc;
std::vector<Frame>         g_stack;
double                     g_total_ns = 0.0;
int                        g_rank     = 0;
bool                       g_dump     = true;

int env_rank() {
    const char* v;
    if ((v = std::getenv("OMPI_COMM_WORLD_RANK"))) return std::atoi(v);
    if ((v = std::getenv("PMI_RANK")))             return std::atoi(v);
    if ((v = std::getenv("SLURM_PROCID")))         return std::atoi(v);
    return 0;
}

void begin(const char* name, uint64_t* kID) {
    kp_sync();                                   // retire everything queued before this kernel
    *kID = static_cast<uint64_t>(g_stack.size());
    g_stack.push_back(Frame{name ? name : "(anonymous)", clk::now()});
}

void end(uint64_t kID) {
    kp_sync();                                   // wait for THIS kernel to actually finish
    if (g_stack.empty() || kID >= g_stack.size()) return;
    const clk::time_point t1 = clk::now();
    Frame& f = g_stack[static_cast<size_t>(kID)];
    const double ns =
        std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(t1 - f.t0).count();
    Acc& a = g_acc[f.name];
    a.calls += 1;
    a.ns    += ns;
    g_total_ns += ns;
    g_stack.resize(static_cast<size_t>(kID));    // pop this frame and anything nested below it
}

}  // namespace

extern "C" {

void kokkosp_init_library(int /*loadSeq*/, uint64_t /*interfaceVer*/, uint32_t /*ndev*/,
                          void* /*devInfo*/) {
    g_rank = env_rank();
    g_dump = (g_rank == 0) || (std::getenv("KP_KTIME_ALL") != nullptr);
}

void kokkosp_finalize_library() {
    if (!g_dump) return;
    const char* prefix = std::getenv("KP_KTIME_OUT");
    char path[4096];
    std::snprintf(path, sizeof(path), "%s.rank%d.csv", prefix ? prefix : "kp_ktime", g_rank);
    std::FILE* fp = std::fopen(path, "w");
    if (!fp) { std::fprintf(stderr, "[kp_ktime] cannot write %s\n", path); return; }

    std::vector<std::pair<std::string, Acc>> v(g_acc.begin(), g_acc.end());
    std::sort(v.begin(), v.end(),
              [](const std::pair<std::string, Acc>& a, const std::pair<std::string, Acc>& b) {
                  return a.second.ns > b.second.ns;
              });

    std::fprintf(fp, "# kp_ktime backend=%s rank=%d kernels=%zu total_ms=%.3f\n", kp_backend,
                 g_rank, v.size(), g_total_ns / 1e6);
    std::fprintf(fp, "# fenced both sides of every kernel: times are real, overlap is destroyed\n");
    std::fprintf(fp, "pct,calls,total_ms,avg_us,kernel\n");
    for (size_t i = 0; i < v.size(); ++i) {
        const Acc& a = v[i].second;
        std::string n = v[i].first;
        for (size_t p = 0; p < n.size(); ++p)
            if (n[p] == ',' || n[p] == '\n') n[p] = ' ';   // keep the CSV parseable
        std::fprintf(fp, "%.3f,%llu,%.4f,%.3f,\"%s\"\n",
                     g_total_ns > 0 ? 100.0 * a.ns / g_total_ns : 0.0,
                     static_cast<unsigned long long>(a.calls), a.ns / 1e6,
                     a.calls ? a.ns / 1e3 / static_cast<double>(a.calls) : 0.0, n.c_str());
    }
    std::fclose(fp);
    std::fprintf(stderr, "[kp_ktime] wrote %s (%zu kernels, %.1f ms fenced device time)\n", path,
                 v.size(), g_total_ns / 1e6);
}

void kokkosp_begin_parallel_for   (const char* n, uint32_t, uint64_t* k) { begin(n, k); }
void kokkosp_end_parallel_for     (uint64_t k)                          { end(k);      }
void kokkosp_begin_parallel_reduce(const char* n, uint32_t, uint64_t* k) { begin(n, k); }
void kokkosp_end_parallel_reduce  (uint64_t k)                          { end(k);      }
void kokkosp_begin_parallel_scan  (const char* n, uint32_t, uint64_t* k) { begin(n, k); }
void kokkosp_end_parallel_scan    (uint64_t k)                          { end(k);      }

}  // extern "C"
