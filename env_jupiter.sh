#!/usr/bin/env bash
# env_jupiter.sh — build/run environment for JUPITER (JSC) GH200 booster nodes.
# Adapted from env_dolpung.sh + the proven env_jupiter_cuda.sh of the jupiter-gh200
# line (see that repo's docs/JUPITER_PORT_AND_SCALING.md for the port history).
#
# JUPITER vs dolpung, the three differences that matter:
#  - logins AND compute are both aarch64 (Grace): no cross-arch PATH hygiene needed.
#    The hygiene equivalent here is `module --force purge` + a pinned module set —
#    modules (not absolute spack paths) are the deterministic idiom at JSC.
#  - the CUDA-aware MPI is ParaStationMPI (pscom/UCX), not OpenMPI. MPI-settings/CUDA
#    flips the runtime knobs (PSP_CUDA=1, UCX_* memtype) ON. The OMPI_MCA_* /
#    UCX_TLS lines from the dolpung job templates are NOT shipped — start from
#    system defaults (plan §2), deviations only after an A/B.
#  - Stage choice: Stages/2025 = CUDA 12.6 + GCC 13.3 → supports the vendored
#    Kokkos 4.4.01 pin unmodified (dolpung twin: gcc14.2/CUDA12.9/Kokkos4.4.01).
#    Stages/2026 is CUDA-13-native and would force a Kokkos bump to >=4.7 —
#    kept as the documented fallback, not the default.
#
# Source this in the SAME command as the build / srun (modules don't persist).

export LC_ALL=en_US.UTF-8

# SLURM batch scripts run as non-login shells where Lmod may be undefined.
if ! command -v module >/dev/null 2>&1; then
    for _i in /e/software/default/lmod/*/init/bash "$MODULESHOME/init/bash" /etc/profile.d/z00_lmod.sh; do
        [ -r "$_i" ] && { source "$_i"; break; }
    done
fi

module --force purge 2>/dev/null
module load Stages/2025 2>/dev/null   # deprecated-stage banner silenced; CUDA/12 lives here
module load GCC/13.3.0
module load ParaStationMPI/5.11.0-1
module load CUDA/12
module load netCDF/4.9.2
module load CMake/3.30.3
module load MPI-settings/CUDA          # CUDA-aware MPI runtime (device-ptr halo)

export NVCC_WRAPPER_DEFAULT_COMPILER=g++

ulimit -s unlimited 2>/dev/null
ulimit -c 0 2>/dev/null
