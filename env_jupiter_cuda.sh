#!/usr/bin/env bash
#
# Jupiter (JSC) CUDA-aware build/run environment for build-cuda (GH200 / Hopper).
#
# WHY a separate env from env_jupiter.sh: the CUDA build needs nvcc + a CUDA-AWARE MPI
# so on-device halo exchange (device-pointer MPI, FESOM's M5.1 path) works instead of
# segfaulting. On Levante that meant swapping to an nvhpc-built OpenMPI; on JUPITER
# ParaStationMPI is already CUDA-capable (UCX cuda_copy/cuda_ipc, pscom) and the
# `MPI-settings/CUDA` module flips the runtime knobs (PSP_CUDA=1, UCX_* memtype) ON.
#
# Toolchain (Stage 2026, CUDA-13-native):
#   GCC 14.3.0  +  ParaStationMPI 5.13.0-1  +  CUDA 13.0  +  netCDF 4.9.3
#   MPI-settings/CUDA  (CUDA-aware MPI runtime)
#   nvcc_wrapper host compiler = g++  (NVCC_WRAPPER_DEFAULT_COMPILER)
#
# Kokkos: vendored submodule at 4.7.03 (CUDA-13-capable; the JSC Kokkos/4.7.03-CUDA-13
# system module proves this exact Kokkos+CUDA+GCC combo builds & runs on GH200).
#
# Source this in the SAME command as the build / srun (modules do not persist across shells).

export LC_ALL=en_US.UTF-8

# Ensure `module` exists: SLURM batch scripts run as NON-login shells where Lmod may
# be undefined. Source the init explicitly if so (login shells already have it).
if ! command -v module >/dev/null 2>&1; then
    for _i in /e/software/default/lmod/*/init/bash "$MODULESHOME/init/bash" /etc/profile.d/z00_lmod.sh; do
        [ -r "$_i" ] && { source "$_i"; break; }
    done
fi

module --force purge 2>/dev/null
module load Stages/2026
module load GCC/14.3.0
module load ParaStationMPI/5.13.0-1
module load CUDA/13
module load netCDF/4.9.3
module load CMake/3.31.8
module load MPI-settings/CUDA          # CUDA-aware MPI runtime (device-ptr halo)

export NVCC_WRAPPER_DEFAULT_COMPILER=g++

ulimit -s unlimited 2>/dev/null
ulimit -c 0 2>/dev/null
