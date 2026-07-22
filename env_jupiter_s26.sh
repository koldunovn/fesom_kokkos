#!/usr/bin/env bash
# env_jupiter.sh (s26 variant) — Stage-2026 build/run environment for the JUPITER
# Stage-2026 A/B pair (port_kokkos_s26 / port_kokkos_mp_s26).
#
# DIFFERS from the fleet's certified twin env (port_kokkos/env_jupiter.sh =
# Stages/2025, CUDA 12.6, Kokkos 4.4.01 as pinned): this stack is
#   Stages/2026: GCC 14.3.0 + ParaStationMPI 5.13.0-1 + CUDA 13 + netCDF 4.9.3
# and the vendored Kokkos submodule is CHECKED OUT AT 4.7.03 (0305d9f), off the
# branch pin — required because Kokkos 4.4 predates CUDA 13. This combo
# (PSMPI 5.13 + MPI-settings/CUDA + Kokkos 4.7.03 + CUDA 13 on GH200) is the one
# the jupiter-gh200 line ran device-pointer halos on for weeks.
# Purpose: the D6 transport A/B (device vs STAGE on a working fabric) + CUDA
# codegen check (the sp -2%-under-twin question). The FLEET stays on the frozen
# Stage-2025 twin binaries — never mix builds inside one campaign table.

export LC_ALL=en_US.UTF-8

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
