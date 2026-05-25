cmake_minimum_required(VERSION 3.18)
project(kokkos_smoke CXX)

# Build the vendored Kokkos in-tree (ICON-style). Backend selected by the
# -DKokkos_ENABLE_* flags passed on the cmake command line:
#   Serial : (default)            -DKokkos_ENABLE_SERIAL=ON
#   OpenMP :                      -DKokkos_ENABLE_OPENMP=ON
#   Cuda   :                      -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
#            with CMAKE_CXX_COMPILER = externals/kokkos/bin/nvcc_wrapper
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../externals/kokkos
                 ${CMAKE_BINARY_DIR}/kokkos)

add_executable(kokkos_smoke smoke.cpp)
target_link_libraries(kokkos_smoke Kokkos::kokkos)
