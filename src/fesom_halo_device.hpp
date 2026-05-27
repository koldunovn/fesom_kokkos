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

// Is the on-device halo path active? true only on a CUDA build with the env
// override FESOM_HOST_HALO unset/!=1. Always defined (returns false elsewhere)
// so the dispatch reads cleanly on every backend.
bool fesom_halo_device_active();

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
void fesom_halo_exchange_device(fesom::Field   &f,
                                fesom_halo_kind kind,
                                int             n_levels,
                                int             n_components,
                                fesom_partit   *p);
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
inline void fesom_halo_field(fesom::Field &f, fesom_halo_kind kind,
                             int n_levels, int n_components, fesom_partit *p)
{
    f.modify_device();
    if (!p || p->npes <= 1) return;
#ifdef KOKKOS_ENABLE_CUDA
    if (fesom_halo_device_active()) {
        fesom_halo_exchange_device(f, kind, n_levels, n_components, p);
        return;
    }
#endif
    f.sync_host();
    fesom_halo_exchange(f.h_checked(), kind, n_levels, n_components, p);
    f.modify_host();
    f.sync_device();
}

#endif // FESOM_HALO_DEVICE_HPP
