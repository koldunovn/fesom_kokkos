#ifndef FESOM_FIELD_HPP
#define FESOM_FIELD_HPP
//
// fesom_field.hpp -- DualView-backed field wrapper for the C++/Kokkos port (Milestone M1).
//
// A `Field` owns a Kokkos::DualView<T*, LayoutRight> and exposes:
//   - a HOST raw pointer  h()  (== view_host().data()) so legacy C-style arr[entity*nl+lev]
//     indexing and the FESOM_NODE3D/ELEM3D/ELEMVEC macros keep working UNCHANGED;
//   - a DEVICE view  d()  for Kokkos kernels (from M2 onward);
//   - a coarse modify/sync discipline (modify_host, modify_device, sync_host, sync_device);
//   - an authoritative-space tag + h_checked(), which traps a stale-host read when built with
//     -DFESOM_KK_SYNCCHECK (the GPU analogue of the C port's stale-halo guard).
//
// WHY LayoutRight during the correctness phase (M1-M4): for a RANK-1 view LayoutRight ==
// LayoutLeft (both contiguous), so the host mirror is byte-for-byte the C flat layout
// [entity*nl + level] (level contiguous). Un-ported host C-kernels and ported Serial kernels
// therefore share the exact same bytes, so the Serial build stays bit-identical to the C golden.
//
// On a host-only backend (Serial/OpenMP) the device memory space == HostSpace, so the host and
// device views alias the SAME allocation and the sync/modify calls are no-ops -- which is exactly
// why Serial/OpenMP cannot diverge at M1. On CUDA the device view lives in CudaSpace and sync does
// a bitwise-exact deep_copy of double/int; no compute kernel runs on device until M2.1.
//
// M5 PERFORMANCE NOTE (this is NOT a one-line layout swap): coalescing the hot interleaved
// 2-component fields (uv, uv_rhs, tr_xy via FESOM_ELEMVEC) and 3-D node fields on GPU is a RANK
// CHANGE (1-D flat to View<double**> or View<double**[2]>) and forces the FESOM_NODE3D/ELEM3D/
// ELEMVEC index macros to become layout-agnostic accessor functions over the View. A flat rank-1
// LayoutLeft == LayoutRight buys no coalescing. See plan section M5 and lesson L6.
//
// real_t is the WORKING precision (M8: double by default, float under FESOM_SINGLE_PRECISION —
// see fesom_types.h); Field == FieldT<real_t> is the model-state field type, IntField == FieldT<int>.
//
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <cstddef>
#include "fesom_types.h"
#ifdef FESOM_KK_SYNCCHECK
#  include <cstdio>
#  include <cstdlib>
#endif
#ifdef FESOM_SYNC_LOG
#  include <cstdio>
// Per-field DtoH/HtoD attribution: emit one line per ACTUAL deep_copy (need_sync gated, so no-op
// syncs are not counted) tagging the View label + bytes. Aggregate the SYNCLOG lines offline to
// see which fields drive the residual per-step PCIe. Compile-guarded: build-cuda (no macro) is
// byte-identical. Diagnostic only — never shipped on.
#endif

namespace fesom {

template <class T>
class FieldT {
public:
    using dualview_t  = Kokkos::DualView<T*, Kokkos::LayoutRight>;
    using host_view_t = typename dualview_t::t_host;
    using dev_view_t   = typename dualview_t::t_dev;

    FieldT() = default;

    // Allocate n elements, zero-initialised (like calloc). The C port's malloc'd arrays are always
    // fully written before being read, so zero-init never changes a result -- it only over-specifies
    // the malloc sites, while exactly matching the calloc sites.
    void alloc(const char* label, std::size_t n) {
        dv_   = dualview_t(label, n);
        n_    = n;
        auth_ = Auth::Synced;   // fresh allocation: both spaces zeroed and current
    }

    bool        allocated() const { return n_ != 0; }
    std::size_t size()      const { return n_; }

    // HOST raw pointer for legacy C-style indexing. Does NOT assert currency -- use h_checked()
    // at host entry points (halo pack/unpack, I/O gather, legacy host kernels).
    T* h() const { return dv_.view_host().data(); }

    // DEVICE view (for Kokkos kernels) and the typed host view.
    dev_view_t  d()  const { return dv_.view_device(); }
    host_view_t hv() const { return dv_.view_host(); }

    // Coarse sync discipline. modify marks a space dirty; sync propagates only if needed.
    void modify_host()   { dv_.modify_host();   auth_ = Auth::Host; }
    void modify_device() { dv_.modify_device(); auth_ = Auth::Device; }
    void sync_host()     {
#ifdef FESOM_SYNC_LOG
        if (dv_.need_sync_host())
            std::fprintf(stderr, "SYNCLOG D2H %s %zu\n", dv_.view_host().label().c_str(), n_ * sizeof(T));
#endif
        dv_.sync_host();   if (auth_ == Auth::Device) auth_ = Auth::Synced; }
    void sync_device()   {
#ifdef FESOM_SYNC_LOG
        if (dv_.need_sync_device())
            std::fprintf(stderr, "SYNCLOG H2D %s %zu\n", dv_.view_host().label().c_str(), n_ * sizeof(T));
#endif
        dv_.sync_device(); if (auth_ == Auth::Host)   auth_ = Auth::Synced; }

    // HOST pointer WITH a currency check under -DFESOM_KK_SYNCCHECK. Routing halo/I/O/legacy host
    // reads through this traps a stale-host access (device authoritative and un-synced) at the
    // access site instead of silently corrupting the field. Zero-cost when the macro is off.
    //
    // The check is a self-contained fprintf+abort, NOT assert(): the SYNCCHECK build must be -O3
    // Release to compare bit-for-bit against the golden, and Release defines NDEBUG, which would
    // silently neuter an assert() exactly in the build that matters (lesson L21). The check then
    // depends only on FESOM_KK_SYNCCHECK, never on NDEBUG.
    T* h_checked() const {
#ifdef FESOM_KK_SYNCCHECK
        if (auth_ == Auth::Device) {
            std::fprintf(stderr, "fesom::Field SYNCCHECK: host read while DEVICE is authoritative "
                                 "-- missing sync_host()\n");
            std::abort();
        }
#endif
        return h();
    }

    void free() { dv_ = dualview_t(); n_ = 0; auth_ = Auth::Synced; }

private:
    enum class Auth { Synced, Host, Device };
    dualview_t  dv_{};
    std::size_t n_    = 0;
    Auth        auth_ = Auth::Synced;
};

using Field    = FieldT<real_t>;   /* M8: the single edit that flips the model state precision */
using IntField = FieldT<int>;

} // namespace fesom

#endif // FESOM_FIELD_HPP
