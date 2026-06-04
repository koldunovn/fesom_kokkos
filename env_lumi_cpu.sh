#!/usr/bin/env bash
# env_lumi_cpu.sh — LUMI-C (AMD EPYC 7763) CPU build/run environment.
#
# Mirrors env_lumi.sh's structure but targets the CPU partition: PrgEnv-gnu,
# Kokkos Serial backend, cray-mpich (no GPU transport), cray-hdf5 + cray-netcdf.
# This is the Levante-equivalent reference build (gcc + openmpi + serial
# netcdf-c → here gcc-native + cray-mpich + cray-netcdf), used to A/B against
# the LUMI HIP perf numbers.

module load LUMI/25.03 partition/C
module load PrgEnv-gnu
module load cray-hdf5
module load cray-netcdf
module load buildtools/25.03

# Drop cray-libsci (we don't use BLAS/LAPACK and it drags additional deps).
module unload cray-libsci 2>/dev/null || true

# Cray wrappers — CC under PrgEnv-gnu forwards to g++ (gcc-native).
export FC=ftn CC=cc CXX=CC

ulimit -s unlimited
ulimit -c 0

# Cray PE puts its libs on CRAY_LD_LIBRARY_PATH; prepend so the binary finds
# libmpi / libnetcdf at runtime.
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}

echo "[env_lumi_cpu.sh] modules:"
module list 2>&1 | sed -n '1,30p'
echo "[env_lumi_cpu.sh] which CC: $(command -v CC)"
echo "[env_lumi_cpu.sh] which nc-config: $(command -v nc-config)"
echo "[env_lumi_cpu.sh] which cmake: $(command -v cmake)"
