// bwtest.cpp — achieved fp64 memory bandwidth, raw HIP / raw CUDA, single GPU.
//
// INVESTIGATION ONLY. Nothing here touches the model or a measurement row.
//
// Purpose: the FESOM kernel profile shows LUMI (MI250X GCD) running 2.63x slower than MN5
// (H100) across kernels that have no atomics and no scratch spill. Vendor peak says the two
// should differ by ~1.25x on bandwidth. This decides which number is right:
//
//   * if the STREAM ratio comes out near 1.25x, then ~2x of the FESOM gap is software
//     (codegen / occupancy / access pattern) and the optimisation ceiling is far above the
//     8-kernel atomic fix;
//   * if the STREAM ratio is itself near 2.5x, LUMI simply does not deliver its paper
//     bandwidth and the 8-kernel fix is close to all that is available.
//
// The gather sweep is the second half of the question. FESOM is an unstructured-mesh code:
// it reads neighbours through an index array, not contiguously. MI250X GCD has 8 MB of L2,
// H100 has 50 MB. Sweeping the gather window from 8 KB to the full array walks the working
// set across both cache sizes, so the two machines' curves separate exactly where cache
// capacity starts to matter -- if that is what is happening.
//
// Build:  hipcc -O3 -DUSE_HIP  --offload-arch=gfx90a -o bwtest bwtest.cpp
//         nvcc  -O3 -DUSE_CUDA -arch=sm_90          -o bwtest bwtest.cpp
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#define gpuMalloc          hipMalloc
#define gpuFree            hipFree
#define gpuMemcpy          hipMemcpy
#define gpuMemcpyHtoD      hipMemcpyHostToDevice
#define gpuDeviceSync      hipDeviceSynchronize
#define gpuEvent_t         hipEvent_t
#define gpuEventCreate     hipEventCreate
#define gpuEventRecord     hipEventRecord
#define gpuEventSync       hipEventSynchronize
#define gpuEventElapsed    hipEventElapsedTime
#define gpuGetDeviceProps  hipGetDeviceProperties
#define gpuDeviceProp      hipDeviceProp_t
static const char* BACKEND = "HIP";
#else
#include <cuda_runtime.h>
#define gpuMalloc          cudaMalloc
#define gpuFree            cudaFree
#define gpuMemcpy          cudaMemcpy
#define gpuMemcpyHtoD      cudaMemcpyHostToDevice
#define gpuDeviceSync      cudaDeviceSynchronize
#define gpuEvent_t         cudaEvent_t
#define gpuEventCreate     cudaEventCreate
#define gpuEventRecord     cudaEventRecord
#define gpuEventSync       cudaEventSynchronize
#define gpuEventElapsed    cudaEventElapsedTime
#define gpuGetDeviceProps  cudaGetDeviceProperties
#define gpuDeviceProp      cudaDeviceProp
static const char* BACKEND = "CUDA";
#endif

typedef double real_t;                                   // same precision as the model

__global__ void k_copy(real_t* __restrict__ dst, const real_t* __restrict__ src, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

__global__ void k_triad(real_t* __restrict__ a, const real_t* __restrict__ b,
                        const real_t* __restrict__ c, real_t s, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        a[i] = b[i] + s * c[i];
}

// The unstructured-mesh access pattern: one indirect read per output element.
__global__ void k_gather(real_t* __restrict__ dst, const real_t* __restrict__ src,
                         const int* __restrict__ idx, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[idx[i]];
}



// FESOM's ACTUAL pattern: node-major storage, one thread per node, inner loop over levels.
// Each thread reads a whole contiguous column of B doubles; a wavefront therefore issues B
// sequential reads from 64 scattered-but-contiguous blocks, not 64 scattered scalars.
__global__ void k_gather_block(real_t* __restrict__ dst, const real_t* __restrict__ src,
                               const int* __restrict__ idx, size_t nblk, int B) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nblk;
         i += (size_t)gridDim.x * blockDim.x) {
        const real_t* p = src + (size_t)idx[i] * B;
        real_t s = 0;
        for (int j = 0; j < B; ++j) s += p[j];
        dst[i] = s;
    }
}


// The SAME work, but LEVEL-MAJOR: src[lev*nblk + node] instead of src[node*B + lev].
// Consecutive threads now read consecutive addresses at every level, so each wavefront
// issues one coalesced request per level instead of 64 separate column walks. If this
// recovers LUMI's contiguous bandwidth, the layout macro is the fix, not the numbering.
__global__ void k_gather_lvlmaj(real_t* __restrict__ dst, const real_t* __restrict__ src,
                                const int* __restrict__ idx, size_t nblk, int B) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nblk;
         i += (size_t)gridDim.x * blockDim.x) {
        const int n = idx[i];
        real_t s = 0;
        for (int j = 0; j < B; ++j) s += src[(size_t)j * nblk + n];
        dst[i] = s;
    }
}

int main(int argc, char** argv) {
    const size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : (1ull << 27);   // 134M doubles ~ 1.07 GB
    const int reps = (argc > 2) ? atoi(argv[2]) : 10;

    gpuDeviceProp prop{};
    gpuGetDeviceProps(&prop, 0);
    printf("# bwtest backend=%s device=\"%s\" N=%zu (%.2f GB/array) reps=%d\n", BACKEND, prop.name,
           N, N * sizeof(real_t) / 1e9, reps);

    real_t *a, *b, *c;
    int* idx;
    if (gpuMalloc(&a, N * sizeof(real_t)) || gpuMalloc(&b, N * sizeof(real_t)) ||
        gpuMalloc(&c, N * sizeof(real_t)) || gpuMalloc(&idx, N * sizeof(int))) {
        printf("ALLOC FAILED\n");
        return 1;
    }
    {
        std::vector<real_t> h(N, 1.0);
        gpuMemcpy(a, h.data(), N * sizeof(real_t), gpuMemcpyHtoD);
        gpuMemcpy(b, h.data(), N * sizeof(real_t), gpuMemcpyHtoD);
        gpuMemcpy(c, h.data(), N * sizeof(real_t), gpuMemcpyHtoD);
    }

    const int block = 256;
    const int grid = (int)std::min<size_t>((N + block - 1) / block, 65535 * 4);

    gpuEvent_t e0, e1;
    gpuEventCreate(&e0);
    gpuEventCreate(&e1);

#define TIME(stmt)                                                     \
    ({                                                                 \
        stmt;                                                          \
        gpuDeviceSync();                                               \
        float best = 1e30f;                                            \
        for (int r = 0; r < reps; ++r) {                               \
            gpuEventRecord(e0, 0);                                     \
            stmt;                                                      \
            gpuEventRecord(e1, 0);                                     \
            gpuEventSync(e1);                                          \
            float ms = 0;                                              \
            gpuEventElapsed(&ms, e0, e1);                              \
            if (ms < best) best = ms;                                  \
        }                                                              \
        best;                                                          \
    })

    printf("# --- contiguous STREAM (the hardware bandwidth floor) ---\n");
    printf("%-10s %12s %12s\n", "test", "ms", "GB/s");

    float ms = TIME((k_copy<<<grid, block>>>(c, a, N)));
    printf("%-10s %12.3f %12.1f\n", "copy", ms, 2.0 * N * sizeof(real_t) / (ms * 1e6));

    ms = TIME((k_triad<<<grid, block>>>(a, b, c, 2.0, N)));
    printf("%-10s %12.3f %12.1f\n", "triad", ms, 3.0 * N * sizeof(real_t) / (ms * 1e6));

    // Gather sweep: idx[i] lands randomly inside a window of W elements around i. Small W fits
    // in cache and isolates cache behaviour; W = N is a full random scatter over the array.
    printf("# --- indirect gather, window sweep (the unstructured-mesh pattern) ---\n");
    printf("%-14s %10s %12s %12s\n", "window", "MB", "ms", "eff GB/s");
    std::mt19937 rng(12345);
    std::vector<int> h_idx(N);
    const size_t windows[] = {1024, 8192, 65536, 524288, 4194304, 16777216, 67108864, 0};
    for (int w = 0; windows[w] != 0 || w == 7; ++w) {
        size_t W = windows[w] ? std::min(windows[w], N) : N;
        std::uniform_int_distribution<long long> d(0, (long long)W - 1);
        for (size_t i = 0; i < N; ++i) {
            long long base = (long long)(i / W) * (long long)W;
            long long j = base + d(rng);
            h_idx[i] = (int)std::min<long long>(j, (long long)N - 1);
        }
        gpuMemcpy(idx, h_idx.data(), N * sizeof(int), gpuMemcpyHtoD);
        ms = TIME((k_gather<<<grid, block>>>(c, a, idx, N)));
        // bytes actually requested: one 8-byte read + 4-byte index + 8-byte write per element
        printf("%-14zu %10.2f %12.3f %12.1f\n", W, W * sizeof(real_t) / 1e6, ms,
               (double)N * 20.0 / (ms * 1e6));
        if (windows[w] == 0) break;
    }


    // --- block gather: the granularity FESOM actually uses (nl levels per node) ---
    {
        const int B = (argc > 3) ? atoi(argv[3]) : 70;        // NG5 level count
        const size_t nblk = N / (size_t)B;
        printf("# --- BLOCK gather, %d doubles (%d B) per node-column ---\n", B, B * 8);
        printf("%-14s %10s %12s %12s\n", "window MB", "blocks", "ms", "eff GB/s");
        const size_t wins[] = {64, 1024, 8192, 65536, 524288, 0};
        for (int w = 0; ; ++w) {
            size_t W = wins[w] ? std::min(wins[w], nblk) : nblk;
            std::uniform_int_distribution<long long> d(0, (long long)W - 1);
            for (size_t i = 0; i < nblk; ++i) {
                long long base = (long long)(i / W) * (long long)W;
                h_idx[i] = (int)std::min<long long>(base + d(rng), (long long)nblk - 1);
            }
            gpuMemcpy(idx, h_idx.data(), nblk * sizeof(int), gpuMemcpyHtoD);
            const int g2 = (int)std::min<size_t>((nblk + block - 1) / block, 65535 * 4);
            float bms = TIME((k_gather_block<<<g2, block>>>(c, a, idx, nblk, B)));
            printf("%-14.2f %10zu %12.3f %12.1f\n", (double)W * B * 8 / 1e6, W, bms,
                   (double)nblk * B * 8 / (bms * 1e6));
            if (!wins[w]) break;
        }

        printf("# --- same work, LEVEL-MAJOR layout (coalesced across threads) ---\n");
        printf("%-14s %10s %12s %12s\n", "window MB", "blocks", "ms", "eff GB/s");
        for (int w = 0; ; ++w) {
            size_t W = wins[w] ? std::min(wins[w], nblk) : nblk;
            std::uniform_int_distribution<long long> d(0, (long long)W - 1);
            for (size_t i = 0; i < nblk; ++i) {
                long long base = (long long)(i / W) * (long long)W;
                h_idx[i] = (int)std::min<long long>(base + d(rng), (long long)nblk - 1);
            }
            gpuMemcpy(idx, h_idx.data(), nblk * sizeof(int), gpuMemcpyHtoD);
            const int g3 = (int)std::min<size_t>((nblk + block - 1) / block, 65535 * 4);
            float lms = TIME((k_gather_lvlmaj<<<g3, block>>>(c, a, idx, nblk, B)));
            printf("%-14.2f %10zu %12.3f %12.1f\n", (double)W * B * 8 / 1e6, W, lms,
                   (double)nblk * B * 8 / (lms * 1e6));
            if (!wins[w]) break;
        }
    }

    gpuFree(a); gpuFree(b); gpuFree(c); gpuFree(idx);
    printf("# done\n");
    return 0;
}
