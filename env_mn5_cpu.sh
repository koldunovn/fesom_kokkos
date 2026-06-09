#!/usr/bin/env bash
# env_mn5_cpu.sh — MN5 GPP (Intel Xeon Platinum 8480+, Sapphire Rapids, 112c/node).
#
# Mirrors env_lumi_cpu.sh's role: Kokkos Serial backend (one MPI rank per core),
# gcc-native + openmpi + serial netcdf-c. Used as the CPU reference build to
# A/B against env_mn5.sh (Kokkos CUDA on H100).
#
# Toolchain choice:
#   - gcc/12.3.0 → C++17 native, matches the toolchain openmpi/4.1.5-gcc12.3 was built with
#   - openmpi/4.1.5-gcc12.3 → CPU-only MPI (no CUDA transports — that's env_mn5.sh's job)
#   - hdf5/1.14.1-2-gcc-openmpi + netcdf/...-gcc-openmpi → serial netcdf-c
#     (the C port gathers to rank 0 for I/O)

module --force purge 2>/dev/null || true
module use /apps/ACC/modulefiles/environment
module use /apps/ACC/modulefiles/libs
module use /apps/ACC/modulefiles/tools

module load gcc/12.3.0
# Use `openmpi/4.1.5-gcc` (no version suffix) — the netcdf-gcc-openmpi module
# hard-pins this name. `openmpi/4.1.5-gcc12.3` is rejected by netcdf's prereq.
module load openmpi/4.1.5-gcc
module load hdf5/1.14.1-2-gcc-openmpi
module load pnetcdf/1.12.3-gcc-openmpi
module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1_hdf5-1.14.1-2_pnetcdf-1.12.3-gcc-openmpi
module load cmake/3.30.5

# OpenMPI wrappers check OMPI_CC/OMPI_CXX to pick the backend compiler. Without
# these, BSC's profile leaks set I_MPI_* but nothing for OpenMPI → wrapper falls
# back to whatever cc/c++ resolves to (sometimes icpc on MN5 logins). Pin g++.
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran

export FC=mpif90 CC=mpicc CXX=mpicxx

# Login nodes deny `ulimit -s unlimited`; compute allows it.
ulimit -s unlimited 2>/dev/null || true
ulimit -c 0          2>/dev/null || true

export HDF5_USE_FILE_LOCKING=FALSE

echo "[env_mn5_cpu.sh] loaded modules:"
module list 2>&1 | sed -n '1,30p'
echo "[env_mn5_cpu.sh] which mpicxx:    $(command -v mpicxx)"
echo "[env_mn5_cpu.sh] which nc-config: $(command -v nc-config)"
echo "[env_mn5_cpu.sh] which cmake:     $(command -v cmake)"
