/*
 * fesom_halo_device.cpp — on-device halo exchange (GPU-aware MPI), M5.1.
 *
 * See fesom_halo_device.hpp for the why. This is the CUDA-only device path:
 *   PACK   (device gather): send[g*stride+c] = field[(slist[g]-1)*stride + c]
 *   MPI    (device ptrs):   Irecv/Isend on send/recv DEVICE buffers (same tag,
 *                           PE order, MPI_DOUBLE counts as the host path)
 *   UNPACK (device scatter):field[(rlist[g]-1)*stride+c] = recv[g*stride+c]
 *
 * The buffer-offset collapse: send_buf is packed in slist order and (file
 * convention) sptr[0]==rptr[0]==1, so the per-PE MPI offset (sptr[k]-sptr[0])
 * *stride equals the running list index g*stride. Hence PACK/UNPACK are flat
 * gathers/scatters indexed by the global list position g — no per-PE bookkeeping
 * in the kernel. UNPACK is race-free (each halo node appears once in rlist), so
 * no atomics — Serial would be bit-identical too (but Serial never takes this
 * path; it is CUDA-only).
 */
#include "fesom_halo_device.hpp"

#include <cstdlib>   // getenv

bool fesom_halo_device_active()
{
#ifdef KOKKOS_ENABLE_CUDA
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_HOST_HALO");
        cached = (e && e[0] == '1') ? 0 : 1;   // default ON; =1 forces legacy host path
    }
    return cached != 0;
#else
    return false;
#endif
}

#ifndef KOKKOS_ENABLE_CUDA
void fesom_halo_device_free() { /* no device Views on host backends */ }
#else  // ====================== CUDA device path ===========================

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <string>
#include <vector>

#include "fesom_types.h"

namespace {

// Per-kind device-resident comm lists + reusable send/recv buffers + MPI reqs.
struct DevHaloScratch {
    Kokkos::View<int*>    slist_d;   // device copy of cs->slist (1-based local idx)
    Kokkos::View<int*>    rlist_d;   // device copy of cs->rlist (1-based local idx)
    Kokkos::View<double*> send_d;    // device send buffer (grown on demand)
    Kokkos::View<double*> recv_d;    // device recv buffer
    bool                  built = false;
    std::vector<MPI_Request> reqs;
};
DevHaloScratch g_dev[5];   // indexed by fesom_halo_kind (5 entries)

fesom_com_struct *dev_get_com(fesom_partit *p, fesom_halo_kind kind)
{
    switch (kind) {
    case FESOM_HALO_NOD2D:
    case FESOM_HALO_NOD3D:        return &p->com_nod2D;
    case FESOM_HALO_ELEM2D:
    case FESOM_HALO_ELEM3D:       return &p->com_elem2D;
    case FESOM_HALO_ELEM2D_FULL:  return &p->com_elem2D_full;
    default:                      return nullptr;
    }
}

// One-shot: copy cs->slist / cs->rlist to device (the M1.4 / M4.2-a push idiom).
void build_lists(DevHaloScratch &s, fesom_com_struct *cs)
{
    if (s.built) return;
    // File convention: cumulative offsets are 1-based and start at 1. The flat
    // buffer-offset collapse depends on it — assert loudly if a mesh violates it.
    FESOM_CHECK(cs->sptr[0] == 1 && cs->rptr[0] == 1,
                "fesom_halo_device: expected sptr[0]==rptr[0]==1 (got %d,%d) — "
                "buffer-offset collapse invalid", cs->sptr[0], cs->rptr[0]);

    int send_count = cs->sptr[cs->sPEnum] - cs->sptr[0];
    int recv_count = cs->rptr[cs->rPEnum] - cs->rptr[0];
    if (send_count < 0) send_count = 0;
    if (recv_count < 0) recv_count = 0;

    s.slist_d = Kokkos::View<int*>("halo.slist_d", (size_t)send_count);
    s.rlist_d = Kokkos::View<int*>("halo.rlist_d", (size_t)recv_count);
    auto h_s = Kokkos::create_mirror_view(s.slist_d);
    auto h_r = Kokkos::create_mirror_view(s.rlist_d);
    for (int i = 0; i < send_count; ++i) h_s(i) = cs->slist[i];
    for (int i = 0; i < recv_count; ++i) h_r(i) = cs->rlist[i];
    Kokkos::deep_copy(s.slist_d, h_s);
    Kokkos::deep_copy(s.rlist_d, h_r);
    s.built = true;
}

inline void grow(Kokkos::View<double*> &v, size_t need, const char *label)
{
    if (v.extent(0) < need)
        v = Kokkos::View<double*>(std::string(label), need);
}

} // namespace

void fesom_halo_device_free()
{
    for (int i = 0; i < 5; ++i) {
        g_dev[i].slist_d = Kokkos::View<int*>();
        g_dev[i].rlist_d = Kokkos::View<int*>();
        g_dev[i].send_d  = Kokkos::View<double*>();
        g_dev[i].recv_d  = Kokkos::View<double*>();
        g_dev[i].reqs.clear();
        g_dev[i].built = false;
    }
}

void fesom_halo_exchange_device(fesom::Field   &f,
                                fesom_halo_kind kind,
                                int             n_levels,
                                int             n_components,
                                fesom_partit   *p)
{
    if (!p || p->npes == 1) { f.modify_device(); return; }
    fesom_com_struct *cs = dev_get_com(p, kind);
    if (!cs) FESOM_DIE("fesom_halo_device: unknown kind %d", (int)kind);
    if (cs->rPEnum == 0 && cs->sPEnum == 0) { f.modify_device(); return; }

    DevHaloScratch &s = g_dev[(int)kind];
    build_lists(s, cs);

    const int    stride     = n_levels * n_components;
    const int    send_count = cs->sptr[cs->sPEnum] - cs->sptr[0];
    const int    recv_count = cs->rptr[cs->rPEnum] - cs->rptr[0];
    const size_t send_total = (size_t)send_count * stride;
    const size_t recv_total = (size_t)recv_count * stride;

    grow(s.send_d, send_total, "halo.send_d");
    grow(s.recv_d, recv_total, "halo.recv_d");

    auto field = f.d();          // RANK-1 flat LayoutRight device view of the field
    auto send  = s.send_d;
    auto recv  = s.recv_d;
    auto slist = s.slist_d;
    auto rlist = s.rlist_d;
    const int st = stride;

    // PACK: gather slist-indexed entries into the send buffer (slist order).
    if (send_count > 0) {
        Kokkos::parallel_for("fesom_halo_pack", Kokkos::RangePolicy<>(0, send_count),
            KOKKOS_LAMBDA(const int g) {
                const long src = (long)(slist(g) - 1) * st;
                const long dst = (long)g * st;
                for (int c = 0; c < st; ++c) send(dst + c) = field(src + c);
            });
    }
    Kokkos::fence();   // PACK must complete before MPI reads the device send buffer

    // MPI — identical loop structure to the host fesom_halo_exchange, but the
    // buffers are DEVICE pointers (GPU-aware MPI). Same tag / PE order / counts.
    const int tag    = 2000 + (int)kind;
    const int needed = cs->rPEnum + cs->sPEnum;
    if ((int)s.reqs.size() < needed) s.reqs.resize(needed);
    int     nreq     = 0;
    double *send_ptr = send.data();
    double *recv_ptr = recv.data();

    for (int k = 0; k < cs->rPEnum; ++k) {
        const int    nseg  = cs->rptr[k + 1] - cs->rptr[k];
        const size_t off   = (size_t)(cs->rptr[k] - cs->rptr[0]) * stride;
        MPI_Irecv(recv_ptr + off, nseg * stride, MPI_DOUBLE, cs->rPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    for (int k = 0; k < cs->sPEnum; ++k) {
        const int    nseg  = cs->sptr[k + 1] - cs->sptr[k];
        const size_t off   = (size_t)(cs->sptr[k] - cs->sptr[0]) * stride;
        MPI_Isend(send_ptr + off, nseg * stride, MPI_DOUBLE, cs->sPE[k], tag,
                  p->MPI_COMM_FESOM, &s.reqs[nreq++]);
    }
    MPI_Waitall(nreq, s.reqs.data(), MPI_STATUSES_IGNORE);

    // UNPACK: scatter the recv buffer into the halo slots (race-free — each
    // rlist entry is a distinct halo node, so no atomics).
    if (recv_count > 0) {
        Kokkos::parallel_for("fesom_halo_unpack", Kokkos::RangePolicy<>(0, recv_count),
            KOKKOS_LAMBDA(const int g) {
                const long dst = (long)(rlist(g) - 1) * st;
                const long src = (long)g * st;
                for (int c = 0; c < st; ++c) field(dst + c) = recv(src + c);
            });
    }
    Kokkos::fence();   // halo writes visible before the next device reader

    f.modify_device();   // device authoritative: owned (unchanged) + halo (new)
}

#endif // KOKKOS_ENABLE_CUDA
