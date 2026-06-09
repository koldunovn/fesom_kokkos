#!/usr/bin/env bash
# configure_mn5_cpu.sh — Kokkos Serial build for MN5 GPP (Sapphire Rapids).
# Mirrors configure_lumi_cpu.sh. FESOM_GPU_RESIDENT macro evaluates to 0 here
# → host-staged halo bracket, bit-identical to the LUMI-C Serial build.
#
# Output: build-cpu-mn5/fesom_port

set -e

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"
BUILD_DIR=${BUILD_DIR:-build-cpu-mn5}

CLEAN_BUILD=false
CMAKE_EXTRA=()
for arg in "$@"; do
    if [ "$arg" = "--clean" ]; then
        CLEAN_BUILD=true
    else
        CMAKE_EXTRA+=("$arg")
    fi
done

if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning ${BUILD_DIR}..."
    rm -rf "${BUILD_DIR}"
fi

source "${SOURCE_DIR}/env_mn5_cpu.sh"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SOURCE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    "${CMAKE_EXTRA[@]}"

cmake --build . --parallel "$(nproc --all)"

echo
echo "Built: ${SOURCE_DIR}/${BUILD_DIR}/fesom_port"
