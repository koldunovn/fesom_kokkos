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
  source ./env.sh
  module load nvhpc/24.7-gcc-11.2.0
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
  cmake -S . -B build-m14cuda -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON \
        -DBUILD_TESTING=ON -DFESOM_KK_SYNCCHECK=OFF -DFESOM_SYNC_LOG=OFF \
        -DCMAKE_CXX_COMPILER=/home/a/a270088/port_kokkos_int/externals/kokkos/bin/nvcc_wrapper
  cmake --build build-m14cuda -j 16
fi
echo "BUILD OK: $which"
