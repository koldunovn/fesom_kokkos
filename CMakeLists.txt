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
add_subdirectory(externals/kokkos ${CMAKE_BINARY_DIR}/kokkos)

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
  -fno-omit-frame-pointer
  -ffp-contract=off
  "$<$<CONFIG:Debug>:-O0>"  "$<$<CONFIG:Debug>:-g3>"
  "$<$<CONFIG:Release>:-O3>" "$<$<CONFIG:Release>:-g>"
)

# --- Model executable -----------------------------------------------------------------------
set(FESOM_SRC
  src/fesom_main.cpp        src/fesom_mesh.cpp        src/fesom_mpi.cpp
  src/fesom_dyn.cpp         src/fesom_tracers.cpp     src/fesom_aux.cpp
  src/fesom_forcing.cpp     src/fesom_ic.cpp          src/fesom_eos.cpp
  src/fesom_ale.cpp         src/fesom_ssh.cpp         src/fesom_momentum.cpp
  src/fesom_tracer_adv.cpp  src/fesom_pp.cpp          src/fesom_step.cpp
  src/fesom_forcing_analytical.cpp                    src/fesom_io.cpp
  src/fesom_constants.cpp   src/fesom_phc.cpp         src/fesom_jra55.cpp
  src/fesom_bulk.cpp        src/fesom_tracer_diff.cpp src/fesom_sss_runoff.cpp
  src/fesom_partit.cpp      src/fesom_halo.cpp        src/fesom_ice.cpp
  src/fesom_halo_device.cpp src/fesom_profile.cpp
  src/fesom_ice_thermo.cpp  src/fesom_ice_coupling.cpp src/fesom_ice_evp.cpp
  src/fesom_ice_fct.cpp     src/fesom_gm.cpp          src/fesom_kpp.cpp
  src/fesom_calendar.cpp    src/fesom_io_stream.cpp   src/fesom_io_stream_dispatch.cpp
  src/fesom_io_config.cpp   src/fesom_restart.cpp
)
add_executable(fesom_port ${FESOM_SRC})
target_include_directories(fesom_port PRIVATE src ${NC_INCDIR})
target_link_libraries(fesom_port PRIVATE Kokkos::kokkos m ${NC_LIBS} MPI::MPI_CXX)

# M1.5 diagnostic: per-step host<->device Field coherence round-trip + h_checked() guards at the
# halo/I/O host entry points (see docs/SYNC_MAP.md). When ON, fesom::Field::h_checked() aborts on a
# stale-host read and the step driver bounces the evolving state H->D->H every step to exercise the
# rails before any real device compute exists. Both are compiled out when OFF (default), so the
# production build is unchanged. Build the diagnostic as a SEPARATE -O3 Release dir so it still
# matches the golden bit-for-bit: cmake -S . -B build-synccheck -DKokkos_ENABLE_SERIAL=ON
#   -DCMAKE_BUILD_TYPE=Release -DFESOM_KK_SYNCCHECK=ON   (the guard is NDEBUG-independent — L21).
option(FESOM_KK_SYNCCHECK "Assert Field host/device coherence + per-step round-trip (diagnostic)" OFF)
if(FESOM_KK_SYNCCHECK)
  target_compile_definitions(fesom_port PRIVATE FESOM_KK_SYNCCHECK)
  message(STATUS "FESOM_KK_SYNCCHECK ON — Field coherence guards + per-step round-trip enabled")
endif()

# FESOM_SYNC_LOG (M5.20 PCIe attribution): emit one `SYNCLOG D2H/H2D <label> <bytes>` line per ACTUAL
# Field sync (need_sync-gated) so the per-step host<->device traffic can be aggregated per field
# offline. Scoped to the fesom_port target (Kokkos unchanged → no Kokkos rebuild). Diagnostic, OFF by
# default (production build byte-identical). Build a separate dir: -DFESOM_SYNC_LOG=ON.
option(FESOM_SYNC_LOG "Per-field DtoH/HtoD deep_copy attribution to stderr (diagnostic)" OFF)
if(FESOM_SYNC_LOG)
  target_compile_definitions(fesom_port PRIVATE FESOM_SYNC_LOG)
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
target_link_libraries(test_field PRIVATE Kokkos::kokkos)
add_test(NAME field COMMAND test_field)
