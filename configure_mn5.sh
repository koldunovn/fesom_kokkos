#!/usr/bin/env bash
# configure_mn5.sh — Kokkos CUDA build for MN5 ACC (H100 / Hopper, sm_90).
# Mirrors configure_lumi.sh / configure.sh shape; sources env_mn5.sh and adds
# CUDA flags. Modeled on the Levante CUDA recipe in docs/BUILD.md but switches
# the Kokkos arch to HOPPER90 (MN5 ACC has H100 64GB / sm_90).
#
# Usage:
#   bash -l configure_mn5.sh           # incremental, build-cuda-mn5/
#   bash -l configure_mn5.sh --clean   # wipe and start over
#
# Output: build-cuda-mn5/fesom_port

set -e

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"
BUILD_DIR=${BUILD_DIR:-build-cuda-mn5}

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

source "${SOURCE_DIR}/env_mn5.sh"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Kokkos CUDA backend: enable CUDA, target H100 (Hopper, sm_90), and use
# nvcc_wrapper as CMAKE_CXX_COMPILER (the Kokkos-recommended driver — nvcc
# rejects most C++ flags it sees, the wrapper rewrites them for g++ host pass).
# MPI_*_COMPILER point at the HPC-X mpicc/mpicxx from nvidia-hpc-sdk so FindMPI
# extracts the cuda_copy / cuda_ipc UCX flags.
cmake "${SOURCE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${SOURCE_DIR}/externals/kokkos/bin/nvcc_wrapper" \
    -DCMAKE_C_COMPILER=mpicc \
    -DMPI_CXX_COMPILER=mpicxx \
    -DMPI_C_COMPILER=mpicc \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_HOPPER90=ON \
    "${CMAKE_EXTRA[@]}"

cmake --build . --parallel "$(nproc --all)"

echo
echo "Built: ${SOURCE_DIR}/${BUILD_DIR}/fesom_port"
