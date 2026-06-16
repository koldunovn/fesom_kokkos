#!/usr/bin/env bash
# configure_lumi.sh — Kokkos HIP build for LUMI-G (MI250X / gfx90a).
# Mirrors configure.sh shape; sources env_lumi.sh and adds HIP flags.
#
# Usage:
#   bash -l configure_lumi.sh           # incremental, build-hip/
#   bash -l configure_lumi.sh --clean   # wipe build-hip/ and start over
#
# Output: build-hip/fesom_port

set -e

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"
BUILD_DIR=${BUILD_DIR:-build-hip}

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

source "${SOURCE_DIR}/env_lumi.sh"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Kokkos HIP backend: enable HIP, target gfx90a (MI250X), and use the
# Cray CC wrapper as the C++ compiler. With PrgEnv-amd +
# craype-accel-amd-gfx90a loaded, CC forwards to amdclang++ and adds
# both the HIP toolchain and cray-mpich (+GTL) flags automatically.
# The Cray CC wrapper munges Kokkos's adjacent `--rocm-path=... --offload-arch=...`
# into one concatenated `--rocm-path=...--offload-arch=...`, which makes
# amdclang++ fail the HIP runtime check. So we use amdclang++ directly as
# CMAKE_CXX_COMPILER, but still let FindMPI introspect the CC wrapper to
# extract cray-mpich + GTL include/link flags (that part works fine).
cmake "${SOURCE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=amdclang++ \
    -DCMAKE_C_COMPILER=amdclang \
    -DMPI_CXX_COMPILER=CC \
    -DMPI_C_COMPILER=cc \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_HIP=ON \
    -DKokkos_ARCH_AMD_GFX90A=ON \
    "${CMAKE_EXTRA[@]}"

cmake --build . --parallel "$(nproc --all)"

echo
echo "Built: ${SOURCE_DIR}/${BUILD_DIR}/fesom_port"
