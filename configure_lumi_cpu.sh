#!/usr/bin/env bash
# configure_lumi_cpu.sh — Kokkos Serial build for LUMI-C (AMD EPYC).
# Mirrors configure_lumi.sh but no HIP / no GPU device space — the
# FESOM_GPU_RESIDENT macro evaluates to 0 here → host-staged halo bracket,
# bit-identical to the Levante Serial build by design.
#
# Output: build-cpu/fesom_port

set -e

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"
BUILD_DIR=${BUILD_DIR:-build-cpu}

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

source "${SOURCE_DIR}/env_lumi_cpu.sh"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Use the Cray CC wrapper (= g++ under PrgEnv-gnu) — it correctly adds
# cray-mpich + cray-netcdf flags, and without HIP enabled there's no
# --rocm-path/--offload-arch adjacency for the wrapper to munge.
cmake "${SOURCE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_C_COMPILER=cc \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_HIP=OFF \
    "${CMAKE_EXTRA[@]}"

cmake --build . --parallel "$(nproc --all)"

echo
echo "Built: ${SOURCE_DIR}/${BUILD_DIR}/fesom_port"
