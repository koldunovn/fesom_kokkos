cmake_minimum_required(VERSION 3.18)
project(fesom_kokkos CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)   # gnu++17 — eases compiling the C sources as C++ (M0 bridge)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Kokkos (vendored submodule, built in-tree; backend via -DKokkos_ENABLE_*) -------------
# -DFESOM_KOKKOS_DIR=<path> builds against an EXTERNAL Kokkos source tree instead of the
# vendored 4.4.01. Added for toolkit experiments: Kokkos 4.4 cannot compile against CUDA 13
# (cudaGraphAddDependencies / cudaMemLocation API breaks), so measuring a newer toolkit
# requires a newer Kokkos. The submodule stays pinned and every CERTIFIED build is the one
# WITHOUT this option — a binary built with it is experimental by construction.
if(FESOM_KOKKOS_DIR)
  message(STATUS "Kokkos: EXTERNAL tree ${FESOM_KOKKOS_DIR} (vendored 4.4.01 bypassed — experimental build)")
  add_subdirectory(${FESOM_KOKKOS_DIR} ${CMAKE_BINARY_DIR}/kokkos)
else()
  add_subdirectory(externals/kokkos ${CMAKE_BINARY_DIR}/kokkos)
endif()

# --- MPI + netCDF (host serial I/O — FESOM gathers to rank 0) -------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)

find_program(NC_CONFIG nc-config)
if(NOT NC_CONFIG)
  message(FATAL_ERROR "nc-config not found. On Levante: module load netcdf-c/4.8.1-gcc-11.2.0")
endif()
execute_process(COMMAND ${NC_CONFIG} --includedir
                OUTPUT_VARIABLE NC_INCDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${NC_CONFIG} --libs
                OUTPUT_VARIABLE NC_LIBS_RAW OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(NC_LIBS UNIX_COMMAND "${NC_LIBS_RAW}")
execute_process(COMMAND ${NC_CONFIG} --libdir
                OUTPUT_VARIABLE NC_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "netCDF include: ${NC_INCDIR}")
message(STATUS "netCDF link:    ${NC_LIBS}")
set(CMAKE_INSTALL_RPATH "${NC_LIBDIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --- Compile flags --------------------------------------------------------------------------
# -ffp-contract=off is the M2 kernel-gate determinism knob (deferred from M0.3 to "when the
# first device kernel lands" — M2.1). It disables fused multiply-add CONTRACTION on the HOST
# compiler, so a host kernel `a*b+c` and the Kokkos Serial port of the same expression compile
# to the SAME mul+add (no fma fusion in one but not the other) → the per-kernel FESOM_KK_VERIFY
# gate can demand max|Δ|==0 on Serial. It also makes the whole Serial/OpenMP build reproducible.
# The pi golden was RE-CAPTURED at this setting in M2.1 (docs/reference/PROVENANCE.md); do not
# revert without re-capturing. On the CUDA build this reaches the host portion only (nvcc_wrapper
# forwards it to g++); DEVICE fmad stays enabled, which is exactly why CUDA first diverges at M2.1
# (climate-close, not bit-identical — plan §M2 / validation ladder D5).
#
# void*->T* casts: all 305 malloc/calloc/realloc sites now carry explicit casts
# (scripts/cast_alloc_voidstar.py — M0-cuda), so -fpermissive is GONE. nvcc's front-end rejects
# implicit void*->T* and ignores -fpermissive anyway (lesson L1); the casts don't change codegen.
add_compile_options(
  -Wall -Wextra -Wno-unused-parameter -Wno-write-strings
  -Wundef   # M16 G1 lesson: an `#if MACRO` on an undefined macro silently reads 0 (fesom_jra55.cpp, 2026-09-07)
  -fno-omit-frame-pointer
  -ffp-contract=off
  "$<$<CONFIG:Debug>:-O0>"  "$<$<CONFIG:Debug>:-g3>"
  "$<$<CONFIG:Release>:-O3>" "$<$<CONFIG:Release>:-g>"
)

# --- Model core as an OBJECT library (M10 T3) -----------------------------------------------
# Everything except fesom_main.cpp, so the solver-lab / unit-test executables link the EXACT
# model objects (same partitioning, halo machinery, solver code — zero reimplementation).
# Usage requirements are PUBLIC: fesom_field.hpp is included by every consumer, so the
# FESOM_KK_SYNCCHECK / FESOM_SYNC_LOG definitions must reach every TU that links the core
# (ODR: a Field compiled with the guard in core but without it in a tool would be two types).
set(FESOM_CORE_SRC
  src/fesom_mesh.cpp        src/fesom_mpi.cpp
  src/fesom_dyn.cpp         src/fesom_tracers.cpp     src/fesom_aux.cpp
  src/fesom_forcing.cpp     src/fesom_ic.cpp          src/fesom_eos.cpp
  src/fesom_ale.cpp         src/fesom_ale_dump.cpp    src/fesom_ssh.cpp         src/fesom_momentum.cpp
  src/fesom_ssh_se.cpp
  src/fesom_tracer_adv.cpp  src/fesom_pp.cpp          src/fesom_step.cpp
  src/fesom_forcing_analytical.cpp                    src/fesom_io.cpp
  src/fesom_constants.cpp   src/fesom_phc.cpp         src/fesom_jra55.cpp
  src/fesom_bulk.cpp        src/fesom_tracer_diff.cpp src/fesom_sss_runoff.cpp
  src/fesom_partit.cpp      src/fesom_halo.cpp        src/fesom_ice.cpp
  src/fesom_halo_device.cpp src/fesom_profile.cpp   src/fesom_phasestats.cpp
  src/fesom_ice_thermo.cpp  src/fesom_ice_coupling.cpp src/fesom_ice_evp.cpp
  src/fesom_ice_evpwide.cpp
  src/fesom_ice_fct.cpp     src/fesom_gm.cpp          src/fesom_kpp.cpp
  src/fesom_calendar.cpp    src/fesom_io_stream.cpp   src/fesom_io_stream_dispatch.cpp
  src/fesom_tke.cpp         src/fesom_ice_maevp.cpp
  src/fesom_io_config.cpp     src/fesom_io_restart.cpp
)
add_library(fesom_core OBJECT ${FESOM_CORE_SRC})
target_include_directories(fesom_core PUBLIC src ${NC_INCDIR})
target_link_libraries(fesom_core PUBLIC Kokkos::kokkos m ${NC_LIBS} MPI::MPI_CXX)

add_executable(fesom_port src/fesom_main.cpp)
target_link_libraries(fesom_port PRIVATE fesom_core)

# --- M10 T3: solver lab — offline matrix-replay driver on the real model objects ------------
add_executable(fesom_ssh_lab tools/fesom_ssh_lab.cpp)
target_link_libraries(fesom_ssh_lab PRIVATE fesom_core)

# M1.5 diagnostic: per-step host<->device Field coherence round-trip + h_checked() guards at the
# halo/I/O host entry points (see docs/SYNC_MAP.md). When ON, fesom::Field::h_checked() aborts on a
# stale-host read and the step driver bounces the evolving state H->D->H every step to exercise the
# rails before any real device compute exists. Both are compiled out when OFF (default), so the
# production build is unchanged. Build the diagnostic as a SEPARATE -O3 Release dir so it still
# matches the golden bit-for-bit: cmake -S . -B build-synccheck -DKokkos_ENABLE_SERIAL=ON
#   -DCMAKE_BUILD_TYPE=Release -DFESOM_KK_SYNCCHECK=ON   (the guard is NDEBUG-independent — L21).
# M16 mixed precision (docs/plans/20260902-m16-mixed-precision.md; upstream FESOM/fesom2#940):
# the working precision of the model state, upstream's option name. OFF (default) MUST stay
# bit-identical to the pre-M16 base — the define is absent, fesom_types.h resolves as before.
# ON flips real_t->float and FESOM_MPI_REAL->MPI_FLOAT; dbl_t islands (docs/PRECISION_ISLANDS.md)
# stay double. Directory-scoped so EVERY translation unit (fesom_core, fesom_port, the lab tool,
# the unit tests that compile src/*.cpp on their own) sees one real_t — an ODR split between
# TUs would be silent. The binary banners its precision at startup and gate scripts assert it
# (scripts/mp_assert_banner.sh, L80). No alias, no auto mode (plan D6).
option(USE_SINGLE_PRECISION "Build the model state in single precision (real_t=float)" OFF)
if(NOT USE_SINGLE_PRECISION MATCHES "^(ON|OFF|TRUE|FALSE|YES|NO|Y|N|1|0)$")
  message(FATAL_ERROR "USE_SINGLE_PRECISION must be a boolean (ON/OFF), got: '${USE_SINGLE_PRECISION}'")
endif()
if(USE_SINGLE_PRECISION)
  add_compile_definitions(FESOM_SINGLE_PRECISION)
  message(STATUS "USE_SINGLE_PRECISION=ON — real_t=float, FESOM_MPI_REAL=MPI_FLOAT (M16; dbl_t islands stay double)")
else()
  message(STATUS "USE_SINGLE_PRECISION=OFF — real_t=double (the byte-gated default)")
endif()

option(FESOM_KK_SYNCCHECK "Assert Field host/device coherence + per-step round-trip (diagnostic)" OFF)
if(FESOM_KK_SYNCCHECK)
  target_compile_definitions(fesom_core PUBLIC FESOM_KK_SYNCCHECK)   # PUBLIC: fesom_field.hpp is in every consumer (ODR)
  message(STATUS "FESOM_KK_SYNCCHECK ON — Field coherence guards + per-step round-trip enabled")
endif()

# FESOM_SYNC_LOG (M5.20 PCIe attribution): emit one `SYNCLOG D2H/H2D <label> <bytes>` line per ACTUAL
# Field sync (need_sync-gated) so the per-step host<->device traffic can be aggregated per field
# offline. Scoped to the fesom_port target (Kokkos unchanged → no Kokkos rebuild). Diagnostic, OFF by
# default (production build byte-identical). Build a separate dir: -DFESOM_SYNC_LOG=ON.
option(FESOM_SYNC_LOG "Per-field DtoH/HtoD deep_copy attribution to stderr (diagnostic)" OFF)
if(FESOM_SYNC_LOG)
  target_compile_definitions(fesom_core PUBLIC FESOM_SYNC_LOG)   # PUBLIC: same ODR argument as SYNCCHECK
  message(STATUS "FESOM_SYNC_LOG ON — per-field PCIe attribution (SYNCLOG lines on stderr)")
endif()

# --- Tests (compiled as C++ to match the now-C++ model objects' linkage) --------------------
enable_testing()

# These three unit tests have .c source files compiled as C++ (LANGUAGE CXX, see D9). nvcc_wrapper
# (the CUDA build's CXX compiler) keys off the .c extension and rejects it ("nvcc fatal: No input
# files specified"), so they are excluded from CUDA builds — letting `cmake --build build-cuda` stay
# green. They are pure host logic (calendar math, I/O config/stream parsing) with no Kokkos
# dependency and no backend-dependent behaviour, fully covered by the Serial/OpenMP ctest.
# (test_field below links Kokkos and builds on every backend.)
if(NOT Kokkos_ENABLE_CUDA)
  add_executable(test_calendar tests/test_calendar.c src/fesom_calendar.cpp)
  set_source_files_properties(tests/test_calendar.c PROPERTIES LANGUAGE CXX)
  target_include_directories(test_calendar PRIVATE src)
  target_link_libraries(test_calendar PRIVATE m)
  add_test(NAME calendar COMMAND test_calendar)

  add_executable(test_io_stream_unit tests/test_io_stream_unit.c src/fesom_io_stream_dispatch.cpp)
  set_source_files_properties(tests/test_io_stream_unit.c PROPERTIES LANGUAGE CXX)
  target_include_directories(test_io_stream_unit PRIVATE src)
  target_link_libraries(test_io_stream_unit PRIVATE m MPI::MPI_CXX)
  add_test(NAME io_stream_unit COMMAND test_io_stream_unit)

  add_executable(test_io_config tests/test_io_config.c src/fesom_io_config.cpp)
  set_source_files_properties(tests/test_io_config.c PROPERTIES LANGUAGE CXX)
  target_include_directories(test_io_config PRIVATE src)
  target_compile_definitions(test_io_config PRIVATE _POSIX_C_SOURCE=200809L)
  target_link_libraries(test_io_config PRIVATE m)
  add_test(NAME io_config COMMAND test_io_config)
endif()

# M1.1: fesom::Field DualView wrapper round-trip (host<->device sync, deep_copy). Needs Kokkos
# only; links the active backend (Serial/OpenMP/CUDA) so the round trip is exercised per build.
add_executable(test_field tests/test_field.cpp)
target_include_directories(test_field PRIVATE src)
target_link_libraries(test_field PRIVATE Kokkos::kokkos MPI::MPI_CXX)   # M16: fesom_field.hpp includes fesom_types.h (mpi.h) for real_t
add_test(NAME field COMMAND test_field)

# M10 T4: solver testbed — SPD Laplacian fixture with analytic spectrum, reference PCG, the
# alpha/beta comparator, the Chebyshev-rate checker and the derivations-§0.4/§0.5 symmetry
# assertions. Pure host arithmetic (no Kokkos, no MPI) so it runs on the login node in <1 s;
# per-solver assertions are added here as T5b-T8b land.
add_executable(test_ssh_solvers tests/test_ssh_solvers.cpp)
target_include_directories(test_ssh_solvers PRIVATE src ${NC_INCDIR})
target_link_libraries(test_ssh_solvers PRIVATE m MPI::MPI_CXX Kokkos::kokkos)
add_test(NAME ssh_solvers COMMAND test_ssh_solvers)
