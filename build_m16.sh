#!/usr/bin/env bash
# M16 build — same configuration as build_m14.sh (Kokkos 4.4.01 vendored, Release,
# BUILD_TESTING=ON, SYNCCHECK/SYNC_LOG OFF) plus the precision axis.
#
#   bash build_m16.sh ref0          Serial, the UNMODIFIED d4a9fe0 base -> build-m16-ref0
#                                   (the byte oracle; only ever built from a clean checkout of
#                                   d4a9fe0 — refuse otherwise)
#   bash build_m16.sh serial        Serial FP64 (USE_SINGLE_PRECISION=OFF) -> build-m16serial
#   bash build_m16.sh serial-sp     Serial FP32                             -> build-m16serial-sp
#   bash build_m16.sh cuda          CUDA   FP64                             -> build-m16cuda
#   bash build_m16.sh cuda-sp       CUDA   FP32                             -> build-m16cuda-sp
#
# 🔴 CUDA builds MUST use env_cuda.sh (openmpi 4.1.5-nvhpc, CUDA-aware, RPATH-pinned at link
# time) — see the note in build_m14.sh; ldd every new CUDA binary.
set -e
cd /home/a/a270088/port_kokkos_sp
which="${1:?ref0|serial|serial-sp|cuda|cuda-sp}"
REF0_SHA=d4a9fe0

case "$which" in
  ref0)
    head=$(git rev-parse --short HEAD)
    if [ "$head" != "$REF0_SHA" ] || [ -n "$(git status --porcelain src CMakeLists.txt)" ]; then
      echo "refusing: ref0 must be built from a CLEAN $REF0_SHA (HEAD=$head)"; exit 2
    fi
    source ./env.sh
    cmake -S . -B build-m16-ref0 -DCMAKE_BUILD_TYPE=Release \
          -DKokkos_ENABLE_SERIAL=ON -DBUILD_TESTING=ON \
          -DFESOM_KK_SYNCCHECK=OFF -DFESOM_SYNC_LOG=OFF
    rm -f build-m16-ref0/fesom_port
    cmake --build build-m16-ref0 -j 16
    ;;
  serial|serial-sp)
    sp=OFF; dir=build-m16serial
    [ "$which" = serial-sp ] && { sp=ON; dir=build-m16serial-sp; }
    source ./env.sh
    cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Release \
          -DKokkos_ENABLE_SERIAL=ON -DBUILD_TESTING=ON \
          -DFESOM_KK_SYNCCHECK=OFF -DFESOM_SYNC_LOG=OFF -DUSE_SINGLE_PRECISION=$sp
    rm -f "$dir/fesom_port"            # a failed build must leave NO binary (a stale one passed a gate once)
    cmake --build "$dir" -j 16
    ;;
  cuda|cuda-sp)
    sp=OFF; dir=build-m16cuda
    [ "$which" = cuda-sp ] && { sp=ON; dir=build-m16cuda-sp; }
    source ./env_cuda.sh
    cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Release \
          -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON \
          -DBUILD_TESTING=ON -DFESOM_KK_SYNCCHECK=OFF -DFESOM_SYNC_LOG=OFF \
          -DUSE_SINGLE_PRECISION=$sp \
          -DCMAKE_CXX_COMPILER=/home/a/a270088/port_kokkos_sp/externals/kokkos/bin/nvcc_wrapper
    rm -f "$dir/fesom_port"
    cmake --build "$dir" -j 16
    ldd "$dir/fesom_port" | grep -E 'libmpi\.so' || true
    ;;
  *) echo "unknown target $which"; exit 2;;
esac
echo "BUILD OK: $which"
