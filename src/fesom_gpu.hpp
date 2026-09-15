#ifndef FESOM_GPU_HPP
#define FESOM_GPU_HPP
//
// fesom_gpu.hpp — ONE backend-neutral gate for "this build has a device with its own
// memory space" (xmach 2026-09-15; generalises the LUMI-branch FESOM_GPU_RESIDENT of
// 2026-06, commit 8f3ea0a, to the whole m14 tree).
//
// Every site that used to read `#ifdef KOKKOS_ENABLE_CUDA` meant one of:
//   * the device-resident halo path exists (pack -> MPI on device/pinned ptrs -> unpack),
//   * the FESOM_SPEED_* levers are LIVE (on the host backends they stay legacy unless
//     FESOM_SPEED_FORCE_SERIAL=1 — the byte-proof mechanism),
//   * a host-only code path must not be selected (FESOM_VISC_OPT=5, evpwide selfcheck 3).
// None of them is CUDA-specific: the implementation behind each is Kokkos + MPI only (no
// cuda*/hip* API call anywhere in src/). On an AMD/HIP build the old CUDA guard compiled
// the device path OUT and the levers OFF — correct, silent, and 2-3x slow (the dead-knob
// trap, docs/PORT_HIP_LUMI.md). So the gate is now the pair of backends with a separate
// device memory space.
//
// <Kokkos_Macros.hpp> is the cheap header whose only job is to define KOKKOS_ENABLE_*;
// including it HERE makes the macro include-order independent (fesom_speed.hpp lesson:
// a guard that fires before the Kokkos config is visible resolves every lever to OFF).
//
// On CUDA and Serial builds this header changes NOTHING: FESOM_GPU_RESIDENT is 1 exactly
// when KOKKOS_ENABLE_CUDA was, and 0 exactly when it was not (binaries byte-identical —
// verified by md5 on Levante, see docs/plans/20260915-XMACH-LUMI-MN5-PACKAGE.md).
#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#  define FESOM_GPU_RESIDENT 1
#else
#  define FESOM_GPU_RESIDENT 0
#endif

#endif // FESOM_GPU_HPP
