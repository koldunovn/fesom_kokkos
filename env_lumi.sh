#!/usr/bin/env bash
# env_lumi.sh — LUMI MI250X (AMD gfx90a) build/run environment.
#
# Mirrors the role of env.sh (Levante GCC serial/OpenMP) and env_cuda.sh
# (Levante NVHPC CUDA-aware MPI) for the LUMI-G HIP backend.
#
# Toolchain choice:
#   - PrgEnv-amd  → Cray CC wrapper drives AMD's amdclang++ (ROCm 6.3.4),
#                   which is what hipcc forwards to underneath. Using CC
#                   (instead of hipcc directly) lets the Cray wrapper add
#                   cray-mpich -I/-L and, with craype-accel-amd-gfx90a, the
#                   GTL library for GPU-aware MPI.
#   - rocm/6.3.4  → /opt/rocm-6.3.4 with hipcc + hip headers
#   - craype-accel-amd-gfx90a → tells CC to compile HIP (-x hip), sets
#                               PE_MPICH_GTL_LIBS_amd_gfx90a so MPI sees
#                               device pointers.
#   - cray-hdf5 + cray-netcdf/4.9.0.17 → serial netcdf-c (the C port
#                                         gathers to rank 0 for I/O).
#   - LUMI/25.03 partition/G + buildtools/25.03 → CMake/Ninja.
#
# Source me in the SAME shell as configure / build / run; LUMI modules
# don't persist across login shells.

# LUMI login nodes have Lmod auto-loading a few sticky modules; that's
# fine. Don't `module purge` — sticky modules (LUMI, partition/G,
# init-lumi, lumi-tools) need to remain to keep the EasyBuild stack
# discoverable.

module load LUMI/25.03 partition/G
module load PrgEnv-amd
module load rocm/6.3.4
module load craype-accel-amd-gfx90a
module load cray-hdf5
module load cray-netcdf
module load buildtools/25.03   # CMake, Ninja, etc.

# This C port doesn't use BLAS/LAPACK — but if cray-libsci stays loaded, the
# Cray CC wrapper auto-links libsci_amd_mpi, which drags in a runtime dep on
# AMD-flang's libflang.so. Drop it: the binary stays a pure C/Kokkos/MPI/NetCDF
# build, and we don't need to chase the flang runtime path.
module unload cray-libsci 2>/dev/null || true

# Cray wrappers — keep the same FC/CC/CXX convention as env.sh.
export FC=ftn CC=cc CXX=CC

# GPU-aware MPI. The cray-mpich library checks this at runtime to
# enable GPU-direct paths through the GTL library.
export MPICH_GPU_SUPPORT_ENABLED=1

# Cray PE puts its libs on CRAY_LD_LIBRARY_PATH instead of LD_LIBRARY_PATH —
# without this prepend, the binary can't find libmpi_amd / libnetcdf_amd /
# libhdf5_amd at runtime. The AMD variants of these Cray libs themselves
# DT_NEED libflang.so (they were compiled with AMD Flang), so we also need
# /opt/rocm-6.3.4/lib/llvm/lib on the path even though this C port has no
# Fortran of its own.
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/opt/rocm-6.3.4/lib/llvm/lib:${LD_LIBRARY_PATH}

# Stack/core dump defaults (same as env.sh / env_cuda.sh).
ulimit -s unlimited
ulimit -c 0

# Help the user verify the environment took effect.
echo "[env_lumi.sh] modules:"
module list 2>&1 | sed -n '1,40p'
echo "[env_lumi.sh] which CC: $(command -v CC)"
echo "[env_lumi.sh] which hipcc: $(command -v hipcc)"
echo "[env_lumi.sh] which nc-config: $(command -v nc-config)"
echo "[env_lumi.sh] which cmake: $(command -v cmake)"
