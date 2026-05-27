/*
 * mpi_cuda_smoke.cpp — decisive GPU-aware-MPI probe for the M5.1 halo work.
 *
 * Question: can MPI send/recv a *device* pointer on Levante's
 * openmpi/4.1.2 (built --without-cuda) + UCX 1.12 (CUDA transports)?
 *
 * Test: each of 2 ranks binds its node-local GPU, cudaMalloc's a device
 * buffer (NOT managed — a true device-only pointer, the same memory class
 * a Kokkos CudaSpace View uses), fills it with rank-distinct values, then
 * MPI_Sendrecv's the DEVICE pointers with its partner. A rank must receive
 * the partner's values. If GPU-aware MPI is absent, this segfaults, errors,
 * or returns garbage (host-memcpy of a device ptr).
 *
 * Build (login node, nvhpc + openmpi loaded):
 *   mpicxx -O2 tools/mpi_cuda_smoke.cpp -o <out> -I$CUDA_HOME/include \
 *          -L$CUDA_HOME/lib64 -lcudart
 * Run (gpu-devel, 1 node, 2 ranks / 2 GPUs, CUDA-aware UCX env): see
 *   jobs/job_mpi_cuda_smoke.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        std::fprintf(stderr, "rank %d CUDA error %s:%d: %s\n",             \
                     rank, __FILE__, __LINE__, cudaGetErrorString(_e));    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                      \
    }                                                                      \
} while (0)

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* node-local rank -> device id (the M3.1 mapping, RUN_GPU.md §2). */
    MPI_Comm shmcomm; int lrank = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &lrank);
    MPI_Comm_free(&shmcomm);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int dev = (ndev > 0) ? (lrank % ndev) : 0;
    CUDA_CHECK(cudaSetDevice(dev));

    if (rank == 0) {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        std::printf("[smoke] compiled with MPIX_CUDA_AWARE_SUPPORT=1; "
                    "MPIX_Query_cuda_support()=%d\n", MPIX_Query_cuda_support());
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
        std::printf("[smoke] MPIX_CUDA_AWARE_SUPPORT defined=0 "
                    "(OMPI self-reports NO cuda; UCX may still carry device bufs)\n");
#else
        std::printf("[smoke] MPIX_CUDA_AWARE_SUPPORT not defined "
                    "(OMPI built --without-cuda; UCX may still carry device bufs)\n");
#endif
        std::fflush(stdout);
    }

    if (size < 2) {
        if (rank == 0) std::printf("[smoke] need >=2 ranks; got %d\n", size);
        MPI_Finalize();
        return 0;
    }

    const int N = 1024;                 /* a bit bigger so a real transfer happens */
    const int partner = rank ^ 1;       /* 0<->1 */
    const bool active = (rank == 0 || rank == 1);

    double *d_send = nullptr, *d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recv, N * sizeof(double)));

    /* fill send with rank-distinct values; recv with a sentinel. */
    double *h = (double *)std::malloc(N * sizeof(double));
    for (int i = 0; i < N; ++i) h[i] = (double)rank * 1.0e6 + (double)i;
    CUDA_CHECK(cudaMemcpy(d_send, h, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_recv, 0, N * sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());

    int fail = 0;
    if (active) {
        /* THE test: device pointers straight into MPI. */
        int ec = MPI_Sendrecv(d_send, N, MPI_DOUBLE, partner, 100,
                              d_recv, N, MPI_DOUBLE, partner, 100,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (ec != MPI_SUCCESS) {
            char s[MPI_MAX_ERROR_STRING]; int n = 0;
            MPI_Error_string(ec, s, &n);
            std::fprintf(stderr, "rank %d MPI_Sendrecv error: %s\n", rank, s);
            fail = 1;
        }
        CUDA_CHECK(cudaMemcpy(h, d_recv, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < N; ++i) {
            double expect = (double)partner * 1.0e6 + (double)i;
            if (h[i] != expect) { if (fail < 100) ++fail; }
        }
        std::printf("[smoke] rank %d (lrank %d -> dev %d): recv[0]=%.1f recv[%d]=%.1f "
                    "expect_first=%.1f  %s\n",
                    rank, lrank, dev, h[0], N - 1, h[N - 1],
                    (double)partner * 1.0e6, fail == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    int global_fail = 0;
    MPI_Allreduce(&fail, &global_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[smoke] =========== %s ===========\n",
                    global_fail == 0 ? "GPU-AWARE MPI WORKS (device-ptr Sendrecv OK)"
                                     : "GPU-AWARE MPI FAILED");
        std::fflush(stdout);
    }

    std::free(h);
    cudaFree(d_send);
    cudaFree(d_recv);
    MPI_Finalize();
    return global_fail == 0 ? 0 : 1;
}
