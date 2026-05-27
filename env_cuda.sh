# env_cuda.sh — CUDA-aware build/run environment for build-cuda (M5.1 GPU-aware MPI).
#
# WHY a separate MPI from env.sh: the spack openmpi/4.1.2-gcc-11.2.0 is built
# --without-cuda and its UCX 1.12 has NO cuda transports, so a device-pointer
# MPI call SEGFAULTS (proven: jobs/job_mpi_cuda_smoke, smoke 25166379). The
# NVIDIA-built openmpi/4.1.5-nvhpc-24.7 IS CUDA-aware (UCX cuda_copy/cuda_ipc,
# smcuda BTL, --with-cuda) and device-ptr MPI WORKS (smoke 25166531). This is
# the prerequisite for on-device halo exchange (no full-field PCIe sync).
#
# Serial/OpenMP (build-serial, build-omp) keep env.sh's openmpi/4.1.2 — the
# bit-identity oracle's proven toolchain is untouched (Approach B).
#
# Source this in the SAME command as the build / mpirun (modules don't persist
# across shells). gdr_copy (GPUDirect-RDMA kernel mod) is absent on Levante GPU
# nodes, so inter-node GPU->GPU stages the (small) PACKED halo through pinned
# host memory via cuda_copy — still a win vs the full-field DualView sync.
source /sw/etc/profile.levante
module --force purge
module unload netcdf-c cdo ncview git 2>/dev/null   # sticky-module fix (L17)
module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0
module unload openmpi 2>/dev/null                   # drop the default (conflict w/ 4.1.5-nvhpc)
module load openmpi/4.1.5-nvhpc-24.7                # CUDA-aware MPI (the M5.1 unlock)
module load netcdf-c/main-openmpi-4.1.5-nvhpc-24.7  # netcdf matched to the nvhpc MPI
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
