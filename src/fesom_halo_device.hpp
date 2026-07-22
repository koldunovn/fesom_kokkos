#ifndef FESOM_HALO_DEVICE_HPP
#define FESOM_HALO_DEVICE_HPP
//
// fesom_halo_device.hpp — on-device (GPU-aware-MPI) halo exchange (M5.1).
//
// The perf unlock: the legacy D21 bracket stages a halo as
//   modify_device(); sync_host()       // FULL-FIELD device->host PCIe copy
//   fesom_halo_exchange(host_ptr,...);  // host pack -> MPI -> host unpack
//   modify_host();   sync_device()      // FULL-FIELD host->device PCIe copy
// i.e. it copies the ENTIRE field over PCIe twice per exchange. With the CG
// (~90-127 iters), EVP (120 subcycles) and FCT (~21 brackets) on the device,
// that full-field traffic dominates the GPU timestep (0.731 s/step, halo-bound).
//
// The device path packs only the (small) halo into a device buffer, hands the
// DEVICE pointer to MPI (GPU-aware MPI / UCX cuda transports move it GPU-direct
// or via a pinned bounce of just the packed halo), and unpacks on device — NO
// full-field PCIe sync. Requires a CUDA-aware MPI (openmpi/4.1.5-nvhpc-24.7;
// the spack openmpi/4.1.2 SEGFAULTS on device ptrs — see env_cuda.sh).
//
// APPROACH B (zero risk to the Serial bit-identity oracle): this path is
// compiled + used ONLY under KOKKOS_ENABLE_CUDA. On Serial/OpenMP the call
// sites keep the EXACT legacy host-staged bracket (the #ifdef vanishes), so the
// gate is unchanged by construction. Within the CUDA build the env var
// FESOM_HOST_HALO=1 forces the legacy host path (the A/B regression toggle:
// device-halo result MUST byte-match the host-staged CUDA result — data path
// only, not new arithmetic).
//
#include "fesom_field.hpp"
#include "fesom_halo.h"      // fesom_halo_kind, the host fesom_halo_exchange
#include "fesom_partit.h"
#include <cstdlib>          // getenv (FESOM_HALO_SELFCHECK dispatch)
#include <cstring>          // strstr (FESOM_DBG_SYNC pinpoint)
#include <initializer_list> // M5.23 fesom_halo_fieldN({&f0,&f1,…}) call sites

// Is the on-device halo path active? true only on a CUDA build with the env
// override FESOM_HOST_HALO unset/!=1. Always defined (returns false elsewhere)
// so the dispatch reads cleanly on every backend.
bool fesom_halo_device_active();

// M7.5 (dolpung/GH200) — FESOM_HALO_STAGE=1: keep the device-packed transport
// (pack/unpack kernels, per-partner packed buffers, device_active()==true so the
// CG levers stay live) but run the MPI leg on PINNED-HOST mirrors of the packed
// buffers. For fabrics with no GPUDirect (dolpung: ibv_reg_mr on cuda memory
// fails, gdrcopy can't pin — L102): MPI never sees a device pointer, and only
// the PACKED bytes cross the (coherent, fast) host link — unlike FESOM_HOST_HALO
// which drops to the legacy FULL-FIELD sync bracket and deactivates the module.
// The two knobs are alternatives; HOST_HALO=1 wins if both are set (module off).
// Always defined; false on non-CUDA builds.
bool fesom_halo_stage_on();

// Memory space for the staged pinned-host mirrors (HostSpace on host builds so
// shared structs compile everywhere; only ever allocated when STAGE is on).
#ifdef KOKKOS_ENABLE_CUDA
using fesom_halo_pinned_space = Kokkos::CudaHostPinnedSpace;
#else
using fesom_halo_pinned_space = Kokkos::HostSpace;
#endif

// FESOM_HALO_NOFUSE=1 (debug): make field2/fieldN decompose into N single-field
// device exchanges instead of the fused device2/deviceN message. The fused paths
// have no selfcheck variant (the checker decomposes), so this is the isolation
// A/B for fused-transport bugs — and a temporarily-safe fallback config.
inline bool fesom_halo_nofuse_on()
{
    static int c = -1;
    if (c < 0) { const char *e = getenv("FESOM_HALO_NOFUSE"); c = (e && e[0] == '1') ? 1 : 0; }
    return c != 0;
}

// Free the persistent device comm-lists + send/recv buffers. MUST be called
// before Kokkos::finalize() (the Views must not outlive Kokkos). No-op on
// non-CUDA builds. (fesom_halo_free_buffers() frees the host scratch.)
void fesom_halo_device_free();

#ifdef KOKKOS_ENABLE_CUDA
// On-device halo exchange of a nod/elem Field. CONTRACT: f's DEVICE view holds
// current OWNED data (a device kernel just wrote it) on entry; on exit f is
// DEVICE-authoritative with owned unchanged + halo filled (f.modify_device()
// is set inside). Mirrors fesom_halo_exchange's tag / PE order / MPI_DOUBLE
// counts exactly so the moved bytes are identical to the host path.
// base_off (element offset into f) lets a SLAB of a multi-channel field be
// exchanged on its own (e.g. KPP blmc[3] = 3 slabs at j*slab) — default 0.
void fesom_halo_exchange_device(fesom::Field   &f,
                                fesom_halo_kind kind,
                                int             n_levels,
                                int             n_components,
                                fesom_partit   *p,
                                std::size_t     base_off = 0);

// FESOM_HALO_SELFCHECK debug path: device exchange + host exchange on the same owned data, diff
// the halo (pinpoints a device-transport bug). Same signature as the exchange.
void fesom_halo_device_selfcheck(fesom::Field   &f,
                                 fesom_halo_kind kind,
                                 int             n_levels,
                                 int             n_components,
                                 fesom_partit   *p,
                                 std::size_t     base_off = 0);

// M5.23 (L1): TWO same-kind, same-(n_levels,n_components,base_off) fields exchanged in ONE
// message per neighbour instead of two. The send/recv buffer is laid out [g: f0(stride) f1(stride)]
// per halo node (per-entry stride 2*stride), so the existing flat buffer-offset collapse holds.
// The halo values landing in each field's slots are BYTE-IDENTICAL to two separate
// fesom_halo_exchange_device calls (just co-packed) — a pure message-count reduction (no new
// arithmetic, no atomics). Both fields MUST share kind/n_levels/n_components/base_off.
void fesom_halo_exchange_device2(fesom::Field   &f0,
                                 fesom::Field   &f1,
                                 fesom_halo_kind kind,
                                 int             n_levels,
                                 int             n_components,
                                 fesom_partit   *p,
                                 std::size_t     base_off = 0);

// M5.23 (fieldN): N same-kind, same-(n_levels,n_components,base_off) fields exchanged in ONE message
// per neighbour. Per halo-node block [f0(fs) f1(fs) … f(N-1)(fs)], stride N*fs; byte-identical to N
// separate fesom_halo_exchange_device calls (pure message-count reduction). fields is a HOST array of
// N Field*; the N pack/unpack kernels loop on the host (each captures one f_i.d()). nf==2 == device2.
void fesom_halo_exchange_deviceN(fesom::Field *const *fields, int nf,
                                 fesom_halo_kind kind, int n_levels, int n_components,
                                 fesom_partit *p, std::size_t base_off = 0);
#endif // KOKKOS_ENABLE_CUDA

// The standard D21 device-output halo bracket, with GPU-aware-MPI dispatch.
// Replaces the boilerplate:
//   f.modify_device();
//   if (npes>1) { f.sync_host(); fesom_halo_exchange(f.h_checked(),kind,nl,nc,p);
//                 f.modify_host(); f.sync_device(); }
// CONTRACT: a device kernel just wrote f's OWNED rows. On exit f is
// DEVICE-authoritative (owned + halo current). On CUDA with the device path
// active: pack -> GPU-aware MPI -> unpack on device (NO full-field PCIe sync).
// Else (Serial/OpenMP, or FESOM_HOST_HALO=1): the EXACT legacy host-staged
// bracket. (At npes<=1 it just marks device-dirty — the device already holds
// the data, so it also skips the legacy path's pointless npes==1 round-trip.)
// base_off (element offset into f) exchanges a SLAB of a multi-channel field; default 0.

// M5.9 DEBUG: env-gated per-field sync_host to pinpoint which flipped field's host-staleness drives
// the CORE2 divergence (run at HEAD, which dodges the M5.4 heisenbug crash). FESOM_DBG_SYNC is a
// comma-list of ids; if it contains `id` (or "all"), pull the field to host fresh after its device
// halo. Zero cost when unset. Remove once the culprit field is fixed with a targeted sync.
inline void fesom_dbg_sync(fesom::Field &f, const char *id)
{
    static const char *envv = nullptr; static int loaded = 0;
    if (!loaded) { envv = getenv("FESOM_DBG_SYNC"); loaded = 1; }
    if (!envv) return;
    if (std::strstr(envv, "all") || std::strstr(envv, id)) f.sync_host();
}

inline void fesom_halo_field(fesom::Field &f, fesom_halo_kind kind,
                             int n_levels, int n_components, fesom_partit *p,
                             std::size_t base_off = 0)
{
    f.modify_device();
    if (!p || p->npes <= 1) return;
#ifdef KOKKOS_ENABLE_CUDA
    if (fesom_halo_device_active()) {
        static int selfcheck = -1;
        if (selfcheck < 0) { const char *e = getenv("FESOM_HALO_SELFCHECK"); selfcheck = (e && e[0]=='1') ? 1 : 0; }
        if (selfcheck) fesom_halo_device_selfcheck(f, kind, n_levels, n_components, p, base_off);
        else           fesom_halo_exchange_device (f, kind, n_levels, n_components, p, base_off);
        return;
    }
#endif
    f.sync_host();
    fesom_halo_exchange(f.h_checked() + base_off, kind, n_levels, n_components, p);
    f.modify_host();
    f.sync_device();
}

// M5.23 (L1): exchange TWO same-kind fields in ONE message/neighbour. Semantically identical to
// two back-to-back fesom_halo_field calls — use ONLY when f0,f1 are adjacent (no compute between)
// and share kind/n_levels/n_components/base_off. On CUDA (device path) it co-packs into one message
// (240->120 msg/step in the EVP subcycle). On Serial/OpenMP or FESOM_HOST_HALO=1 it falls back to
// the EXACT two legacy host-staged brackets, so the bit-identical oracle is unchanged by construction.
inline void fesom_halo_field2(fesom::Field &f0, fesom::Field &f1, fesom_halo_kind kind,
                              int n_levels, int n_components, fesom_partit *p,
                              std::size_t base_off = 0)
{
    f0.modify_device();
    f1.modify_device();
    if (!p || p->npes <= 1) return;
#ifdef KOKKOS_ENABLE_CUDA
    if (fesom_halo_device_active()) {
        static int selfcheck = -1;
        if (selfcheck < 0) { const char *e = getenv("FESOM_HALO_SELFCHECK"); selfcheck = (e && e[0]=='1') ? 1 : 0; }
        if (selfcheck) {   // fall back to per-field selfcheck (the diff path is single-field)
            fesom_halo_device_selfcheck(f0, kind, n_levels, n_components, p, base_off);
            fesom_halo_device_selfcheck(f1, kind, n_levels, n_components, p, base_off);
        } else if (fesom_halo_nofuse_on()) {   // isolation A/B: two covered single exchanges
            fesom_halo_exchange_device(f0, kind, n_levels, n_components, p, base_off);
            fesom_halo_exchange_device(f1, kind, n_levels, n_components, p, base_off);
        } else {
            fesom_halo_exchange_device2(f0, f1, kind, n_levels, n_components, p, base_off);
        }
        return;
    }
#endif
    // Serial/OpenMP/host-halo: the EXACT two legacy brackets (identical to two fesom_halo_field).
    f0.sync_host();
    fesom_halo_exchange(f0.h_checked() + base_off, kind, n_levels, n_components, p);
    f0.modify_host();
    f0.sync_device();
    f1.sync_host();
    fesom_halo_exchange(f1.h_checked() + base_off, kind, n_levels, n_components, p);
    f1.modify_host();
    f1.sync_device();
}

// M5.23 (fieldN): exchange N same-kind fields in ONE message/neighbour. Semantically identical to N
// back-to-back fesom_halo_field calls — use ONLY when the fields are adjacent (no compute between)
// and share kind/n_levels/n_components/base_off. On CUDA (device path) it co-packs into one message
// (collapses the EOS 5-block 3->1). On Serial/OpenMP or FESOM_HOST_HALO=1 it falls back to N EXACT
// legacy host-staged brackets, so the bit-identical oracle is unchanged by construction.
inline void fesom_halo_fieldN(std::initializer_list<fesom::Field*> fields, fesom_halo_kind kind,
                              int n_levels, int n_components, fesom_partit *p,
                              std::size_t base_off = 0)
{
    for (fesom::Field *f : fields) f->modify_device();
    if (!p || p->npes <= 1) return;
#ifdef KOKKOS_ENABLE_CUDA
    if (fesom_halo_device_active()) {
        static int selfcheck = -1;
        if (selfcheck < 0) { const char *e = getenv("FESOM_HALO_SELFCHECK"); selfcheck = (e && e[0]=='1') ? 1 : 0; }
        if (selfcheck) {   // fall back to per-field selfcheck (the diff path is single-field)
            for (fesom::Field *f : fields)
                fesom_halo_device_selfcheck(*f, kind, n_levels, n_components, p, base_off);
        } else {
            // initializer_list<Field*>::begin() is Field* const* — matches deviceN's HOST array param.
            fesom_halo_exchange_deviceN(fields.begin(), (int)fields.size(),
                                        kind, n_levels, n_components, p, base_off);
        }
        return;
    }
#endif
    // Serial/OpenMP/host-halo: N EXACT legacy brackets (identical to N fesom_halo_field calls).
    for (fesom::Field *f : fields) {
        f->sync_host();
        fesom_halo_exchange(f->h_checked() + base_off, kind, n_levels, n_components, p);
        f->modify_host();
        f->sync_device();
    }
}

#endif // FESOM_HALO_DEVICE_HPP
