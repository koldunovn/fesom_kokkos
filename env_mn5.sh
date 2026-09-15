#!/usr/bin/env bash
# env_mn5.sh — MareNostrum 5 ACC partition (H100 / Hopper, sm_90).
#
# Mirrors env_lumi.sh (HIP) / env_cuda.sh (Levante CUDA). MN5's GPU stack lives
# under /apps/ACC/modulefiles/{environment,libs,tools}. The login nodes are GPP
# (glogin*) but ACC modulefiles are reachable from there via `module use`; the
# binary runs on ACC compute nodes (partition `acc`).
#
# Toolchain choice:
#   - nvidia-hpc-sdk/24.7  → nvcc 12.x (CUDA 12) targeting H100 (compute_90).
#     Bundles its own HPC-X OpenMPI built --with-cuda. That is the MN5
#     equivalent of Levante's openmpi/4.1.5-nvhpc-24.7 (the M5.1 CUDA-aware
#     unlock — see KOKKOS_PORTING_LESSONS L47).
#   - hdf5/1.14.1-2-nvidia-nvhpcx → matched HDF5 (load order: hdf5 before netcdf).
#     The newer 1.14.4.2 build does NOT pair with the nvhpcx netcdf module
#     ("Cannot load … without hdf5/1.14.1-2-nvidia-nvhpcx") — stick with 1.14.1.
#   - netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1-nvidia-nvhpcx → serial netcdf-c on
#     the nvhpc/HPC-X stack (the C port gathers to rank 0 for I/O).
#   - cmake/3.30.5  → CMake new enough for Kokkos 4.x.
#
# Runtime CAVEAT (printed by `module load nvidia-hpc-sdk`):
#   "NVHPC's HPC-X has no srun integration. Use `mpirun --bind-to none` and
#    export SLURM_CPU_BIND=none." Jobs must invoke `mpirun`, NOT `srun`.

# MN5 GPP logins auto-load bsc/intel/impi/mkl/ucx/oneapi at login. We need a
# clean slate so the HPC-X mpicc (from nvhpc) wins over Intel impi on PATH.
module --force purge 2>/dev/null || true

# ACC software tree (separate MODULEPATH from GPP).
module use /apps/ACC/modulefiles/environment
module use /apps/ACC/modulefiles/libs
module use /apps/ACC/modulefiles/tools

# CUDA 12 + HPC-X OpenMPI (CUDA-aware) — load order matters: hdf5 before netcdf.
module load nvidia-hpc-sdk/24.7
module load hdf5/1.14.1-2-nvidia-nvhpcx
module load pnetcdf/1.12.3-nvidia-nvhpcx
module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1-nvidia-nvhpcx
module load cmake/3.30.5

# nvcc_wrapper forwards to g++ for host code (matches env_cuda.sh on Levante).
# Keeps -ffp-contract=off effective on the host portion (D18 / KOKKOS_PORTING_LESSONS).
export NVCC_WRAPPER_DEFAULT_COMPILER=g++

# Compiler wrappers — HPC-X mpicc/mpicxx land in PATH after the nvhpc load.
export FC=mpif90 CC=mpicc CXX=mpicxx

# GPU-aware MPI: HPC-X / UCX with CUDA transports. With nvhpc 24.7 these are
# baked into the bundled OpenMPI, but explicitly select the cuda-capable PML
# and pin the UCX TLS so HPC-X doesn't fall back to host-staged paths.
export OMPI_MCA_pml=ucx
export UCX_TLS=cuda_copy,cuda_ipc,rc,ud,sm,self

# Stack / core-dump defaults. Login nodes deny `ulimit -s unlimited`; compute
# nodes allow it. Tolerate the failure on logins so sourcing env_mn5.sh stays
# clean (`|| true` mirrors env.sh's "best effort" pattern).
ulimit -s unlimited 2>/dev/null || true
ulimit -c 0          2>/dev/null || true

# NetCDF locking off — common GPFS / Lustre footgun (HDF5_USE_FILE_LOCKING).
export HDF5_USE_FILE_LOCKING=FALSE

# Sanity print so callers can confirm the chain stuck.
echo "[env_mn5.sh] loaded modules:"
module list 2>&1 | sed -n '1,30p'
echo "[env_mn5.sh] which nvcc:        $(command -v nvcc)"
echo "[env_mn5.sh] which mpicxx:      $(command -v mpicxx)"
echo "[env_mn5.sh] which nc-config:   $(command -v nc-config)"
echo "[env_mn5.sh] which cmake:       $(command -v cmake)"
echo "[env_mn5.sh] NVCC_WRAPPER_DEFAULT_COMPILER=$NVCC_WRAPPER_DEFAULT_COMPILER"
