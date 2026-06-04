#!/usr/bin/env bash
#
# Jupiter (JSC) build/run environment — CPU backends (Serial / OpenMP) + the full
# MPI model. Mirrors the ROLE of env.sh (Levante) but for the JUPITER Booster:
# ARM Grace (Neoverse-V2) host, GH200 nodes, JSC Lmod "Stages/2026" software stack.
#
# Source me before building/running a CPU build, or via `bash -l configure_jupiter.sh`.
#
# Toolchain (Stage 2026):
#   GCC 14.3.0  +  ParaStationMPI 5.13.0-1 (JSC-native, pscom over NDR200 Dragonfly+)
#   netCDF 4.9.3 (parallel build; FESOM uses only the serial nc_* API, gathers to rank 0)
#   CMake 3.31.8
# NOTE: ParaStationMPI on Stage 2026 auto-pulls CUDA/13 + CUDA-aware UCX; harmless for a
# CPU binary. For the CUDA build use env_jupiter_cuda.sh (adds MPI-settings/CUDA + nvcc).

export LC_ALL=en_US.UTF-8

# Ensure `module` exists: SLURM batch scripts run as NON-login shells where Lmod may
# be undefined. Source the init explicitly if so (login shells already have it).
if ! command -v module >/dev/null 2>&1; then
    for _i in /e/software/default/lmod/*/init/bash "$MODULESHOME/init/bash" /etc/profile.d/z00_lmod.sh; do
        [ -r "$_i" ] && { source "$_i"; break; }
    done
fi

# Fully explicit, reproducible module set (independent of login defaults).
module --force purge 2>/dev/null
module load Stages/2026
module load GCC/14.3.0
module load ParaStationMPI/5.13.0-1
module load netCDF/4.9.3
module load CMake/3.31.8

export FC=mpif90 CC=mpicc CXX=mpicxx

ulimit -s unlimited 2>/dev/null
ulimit -c 0 2>/dev/null
