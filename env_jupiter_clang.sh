#!/usr/bin/env bash
# env_jupiter.sh (clang variant) — host-compiler A/B experiment ONLY.
# Identical to the s26 stack (Stages/2026: PSMPI 5.13, CUDA 13, netCDF 4.9.3,
# Kokkos 4.7.03, device transport) with ONE change: the nvcc HOST compiler is
# clang++ 20.1.8 instead of g++ 14.3 (NVCC_WRAPPER_DEFAULT_COMPILER below).
# Device SASS is nvcc/ptxas either way — this measures the host-side share only.
# Trees port_kokkos_clang / port_kokkos_mp_clang; results -> port2/clang_ab.
# NEVER mix these binaries into the s25 twin or s26 fleet tables.

export LC_ALL=en_US.UTF-8

if ! command -v module >/dev/null 2>&1; then
    for _i in /e/software/default/lmod/*/init/bash "$MODULESHOME/init/bash" /etc/profile.d/z00_lmod.sh; do
        [ -r "$_i" ] && { source "$_i"; break; }
    done
fi

module --force purge 2>/dev/null
module load Stages/2026
module load GCC/14.3.0                 # toolchain libs (libstdc++) + MPI wrapper base
module load ParaStationMPI/5.13.0-1
module load CUDA/13
module load netCDF/4.9.3
module load CMake/3.31.8
module load Clang/20.1.8
module load MPI-settings/CUDA          # CUDA-aware MPI runtime (device-ptr halo)

export NVCC_WRAPPER_DEFAULT_COMPILER=clang++

ulimit -s unlimited 2>/dev/null
ulimit -c 0 2>/dev/null
