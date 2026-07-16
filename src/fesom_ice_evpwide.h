#ifndef FESOM_ICE_EVPWIDE_H
#define FESOM_ICE_EVPWIDE_H

/*
 * E.EVP1 — FESOM_SPEED_EVPWIDE=K: wide-halo (communication-avoiding) EVP subcycling.
 * OPT-IN value knob (0 = off; the FESOM_SPEED master never implies it — user rule 0.24).
 *
 * Contract (docs/plans/20260720-m7-evpwide-design.md §2): exchange (uice, vice) every K-th
 * subcycle on node rings 1..R with R = 2K-1, velocity-update ghost rings 1..R-1 locally each
 * subcycle (owner-order gather for u_rhs — replays the owner's Serial scatter order), sigma-
 * update ALL local elements (maxring <= R) each subcycle, freeze ring R between refreshes.
 * Per step, after owned Step 4, ONE extended exchange ships owner bytes of 11 fields
 * (uice, vice, a_ice, m_ice, m_snow, u_w, v_w, sax, say, rhs_a, rhs_m) onto the rings.
 * Owned loops/order are untouched => knob-OFF is byte-identical; knob-ON is byte-identical
 * on Serial (FORCE_SERIAL proof) and climate-close on CUDA (owner atomics, as today).
 *
 * Split of responsibilities:
 *   - THIS module: scatter-time discovery (ring BFS on the global mesh, ghost-element list,
 *     owner vector, coastal mask), the lazy first-call build (owner-order adjacency + area0
 *     handshake, runtime exchange lists, device pushes), the nf-field extended exchange, the
 *     selfcheck, announce/guards.
 *   - fesom_ice_evp.cpp: the ghost KERNEL BODIES (Step-2/Step-3 maps, K2 sigma, K3 replay)
 *     — deliberately in the SAME TU as the owned kernels (same FMA contraction, the cgpipe
 *     byte-identity argument).
 */

#include <Kokkos_Core.hpp>
#include "fesom_types.h"

struct fesom_mesh;
struct fesom_partit;
struct fesom_ice;

/* Device-side handles for the ghost kernel bodies in fesom_ice_evp.cpp. */
struct FesomEvpwideDev {
    int K = 0;                     /* exchange period (subcycles) */
    int R = 0;                     /* ring depth = 2K-1 */
    int next = 0;                  /* extended node slots appended at [N, N+next) */
    int Eg = 0;                    /* ghost elements (unified elem idx = E + eg) */
    int nUpd = 0;                  /* ghost-updatable node slots (rings 1..R-1) */
    Kokkos::View<int*>    upd_slots;  /* [nUpd] slot ids in [myDim, N+next) */
    Kokkos::View<int*>    gath_ptr;   /* [N+next+1] CSR row ptr (empty row = not updatable) */
    Kokkos::View<int*>    gath_elem;  /* unified elem idx per entry (<E owned, >=E ghost) */
    Kokkos::View<int*>    gath_k;     /* vertex slot 0..2 per entry */
    Kokkos::View<int*>    en_g;       /* [3Eg] vertex node slots (unified, [0, N+next)) */
    Kokkos::View<real_t*> gs_g;       /* [6Eg] gradient_sca */
    Kokkos::View<real_t*> ea_g;       /* [Eg]  elem_area */
    Kokkos::View<real_t*> mf_g;       /* [Eg]  metric_factor */
    Kokkos::View<real_t*> istr_g;     /* [Eg]  per-step ice_strength (ghost Step 3) */
    Kokkos::View<real_t*> s11_g, s12_g, s22_g;   /* [Eg] carried ghost sigma (init 0) */
    Kokkos::View<real_t*> area0x;     /* [N+next] area(n,0): mesh for <N, owner-shipped for ext */
    Kokkos::View<real_t*> corx;       /* [N+next] coriolis_node (byte-equal recompute for ext) */
    Kokkos::View<int*>    coastx;     /* [N+next] owner-parity (GLOBAL-edge) coastal mask */
};

/* Raw knob (fesom_speed_int semantics: opt-in, Serial needs FORCE_SERIAL). No announce. */
int fesom_evpwide_env_K(void);

/* Extended-slot count for allocation tails (0 when off / no stash / npes==1).
 * Valid after fesom_evpwide_mesh_hook. */
int fesom_evpwide_next(void);

/* Scatter-time discovery. MUST be called from scatter_mesh while the GLOBAL arrays
 * (elem_nodes, coord/geo_coord, edges, counts) are still resident and node_g2l/elem_g2l are
 * alive. Collective (owner-vector Allreduce). No-op when the knob is off or npes==1. */
void fesom_evpwide_mesh_hook(struct fesom_mesh *m, struct fesom_partit *p,
                             const int *node_g2l, const int *elem_g2l);

/* Full runtime resolve: 0 = off. First ON call runs the collective lazy build (handshake,
 * lists, device pushes) and announces. Guards: whichEVP!=0 => announced no-op; stash absent
 * (npes==1 / hook skipped) => announced no-op; evp_rheol_steps % K != 0 => abort;
 * FESOM_KK_VERIFY=evp + ON => abort (the host twin is not wide-aware). */
int fesom_evpwide_K(struct fesom_ice *ice, struct fesom_partit *p, struct fesom_mesh *m);

const FesomEvpwideDev &fesom_evpwide_dev(void);

/* The extended exchanges (collective; cgpipe fence discipline, tag 2200; prof hooks).
 * prestep: the 11-field ship after owned Step 4. subcycle: (uice, vice), + selfcheck when
 * FESOM_EVPWIDE_SELFCHECK=1 (max|refresh - local| over rings <= K-1; MUST be 0.0 on Serial). */
void fesom_evpwide_prestep_exchange(struct fesom_ice *ice, struct fesom_partit *p);
void fesom_evpwide_subcycle_exchange(struct fesom_ice *ice, struct fesom_partit *p);

/* Release device Views before Kokkos::finalize (fesom_main teardown; cgpipe rule). */
void fesom_evpwide_free(void);

#endif /* FESOM_ICE_EVPWIDE_H */
