#!/usr/bin/env bash
# M14 build — replicates the m7 (h17) reference configuration exactly.
# Kokkos 4.4.01 vendored, Release, BUILD_TESTING=ON, SYNCCHECK/SYNC_LOG OFF.
set -e
cd /home/a/a270088/port_kokkos_int
which="$1"   # serial | cuda

if [ "$which" = serial ]; then
  source ./env.sh
  cmake -S . -B build-m14serial -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_SERIAL=ON -DBUILD_TESTING=ON \
        -DFESOM_KK_SYNCCHECK=OFF -DFESOM_SYNC_LOG=OFF
  cmake --build build-m14serial -j 16
else
  # 🔴 env_cuda.sh, NOT env.sh + nvhpc. env.sh loads openmpi/4.1.2, which is built
  # --without-cuda; env_cuda.sh loads openmpi/4.1.5-nvhpc-24.7, the CUDA-aware MPI
  # ("the M5.1 unlock") plus the netCDF matched to it.
  #
  # This mattered, and cost most of a session. Building with env.sh RPATH-pins the binary to
  # /sw/spack-levante/openmpi-4.1.2-mnmady/lib, so sourcing env_cuda.sh at RUN time cannot
  # override it. The result segfaults at fesom_halo_device.cpp:460 — MPI_Isend on a DEVICE
  # pointer — in step 1 with every knob off, while h17 and the m13 base run the identical code
  # fine because they were linked against 4.1.5. It looked exactly like a merge regression:
  # CUDA-only, default path, in a file no merge touched. It was a build-environment defect.
  source ./env_cuda.sh
  cmake -S . -B build-m14cuda -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON \
        -DBUILD_TESTING=ON -DFESOM_KK_SYNCCHECK=OFF -DFESOM_SYNC_LOG=OFF \
        -DCMAKE_CXX_COMPILER=/home/a/a270088/port_kokkos_int/externals/kokkos/bin/nvcc_wrapper
  cmake --build build-m14cuda -j 16
fi
echo "BUILD OK: $which"
