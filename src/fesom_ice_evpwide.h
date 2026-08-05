#ifndef FESOM_ICE_EVPWIDE_H
#define FESOM_ICE_EVPWIDE_H

/*
 * E.EVP1 — FESOM_SPEED_EVPWIDE=K: wide-halo (communication-avoiding) EVP subcycling.
 * OPT-IN value knob (0 = off; the FESOM_SPEED master never implies it — user rule 0.24).
 *
 * Contract (docs/plans/20260720-m7-evpwide-design.md §2, as CORRECTED in session 12):
 * ring depth is R = K — not 2K-1. Everything carried is refreshed every window, so ring rho is
 * clean for K-rho subcycles, dirt advances one ring per subcycle, and it reaches ring 1 only AT
 * the refresh subcycle, after the owned reads. Exchange (uice, vice) every K-th subcycle on node
 * rings 1..R, velocity-update ghost rings 1..R-1 locally each subcycle (owner-order gather for
 * u_rhs — replays the owner's Serial scatter order), sigma-update ALL local elements
 * (maxring <= R) each subcycle, freeze ring R between refreshes, and re-baseline the carried
 * ghost sigma from the owner every window. Per step, after owned Step 4, ONE extended exchange
 * ships owner bytes of 11 fields (uice, vice, a_ice, m_ice, m_snow, u_w, v_w, sax, say, rhs_a,
 * rhs_m) onto the rings.
 * Owned loops/order are untouched => knob-OFF is byte-identical; knob-ON is byte-identical
 * on Serial (FORCE_SERIAL proof) and climate-close on CUDA (owner atomics, as today).
 *
 * ── M9: the SAME machinery now serves mEVP (whichEVP=1) — cells ② and ④ ────────────────────
 * Under mEVP the wide halo is exactly the fix Koldunov et al. (2019, GMD 12, 3991) proposed and
 * never tested ("wider halo regions to reduce communication frequency"). Two variants share this
 * module and differ only in WHAT rides the window message:
 *   ② classic form   — (u_aux, v_aux) on rings + the carried element sigma on ghost elements
 *                      => 8·Ng doubles, 2 segments unfused / 1 fused.
 *   ④ divergence form — (u_aux, v_aux, R_u, R_v) on rings; nothing element-carried at all
 *                      => 4·Ng doubles, 1 segment. This is what Danilov's reformulation buys:
 *                      it is not faster by itself (H1 falsified) but it halves the wide-halo
 *                      message and removes the element segment entirely.
 * Unlike std EVP, mEVP's ghost zone also needs per-step ELEMENT maps (pressure_fac, ice_el) and
 * the boundary mask bc_index_nod2D, which multiplies the node solve's determinant (trap 9).
 *
 * Split of responsibilities:
 *   - THIS module: scatter-time discovery (ring BFS on the global mesh, ghost-element list,
 *     owner vector, coastal mask), the lazy first-call build (owner-order adjacency + area0
 *     handshake, runtime exchange lists, device pushes), the plan-driven extended exchange, the
 *     selfcheck, announce/guards.
 *   - fesom_ice_evp.cpp:   the std-EVP ghost KERNEL BODIES (Step-2/Step-3 maps, K2 sigma, K3 replay)
 *   - fesom_ice_maevp.cpp: the mEVP ghost KERNEL BODIES (node/elem pre-maps, stress, node solve)
 *     — each deliberately in the SAME TU as its owned kernels (same FMA contraction, the cgpipe
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
    int R = 0;                     /* ring depth = K (session-12 correction; see the banner) */
    int dbg = 0;                   /* FESOM_EVPWIDE_SELFCHECK=2: ship+compare owner u_rhs/v_rhs */
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
    Kokkos::View<real_t*> istr_g;     /* [Eg]  per-step ice_strength (ghost Step 3, std EVP) */
    Kokkos::View<real_t*> s11_g, s12_g, s22_g;   /* [Eg] carried ghost sigma (init 0) */
    Kokkos::View<real_t*> area0x;     /* [N+next] area(n,0): mesh for <N, owner-shipped for ext */
    Kokkos::View<real_t*> corx;       /* [N+next] coriolis_node (owner-shipped, never recomputed) */
    Kokkos::View<int*>    coastx;     /* [N+next] owner-parity (GLOBAL-edge) coastal mask */
    /* ── M9 mEVP additions ─────────────────────────────────────────────────────────────────
     * pfac_g/ice_el_g are the mEVP twins of istr_g: per-step element maps recomputed locally
     * each step by running the owned element pre-map over the ghost range (their inputs —
     * m_ice, a_ice at the ghost vertices — arrive as owner bytes in the prestep ship, so the
     * recompute is a pure map of owner bytes, not a re-derivation).
     * bcx is DIFFERENT in kind: bc_index_nod2D is built from LOCAL edges (fesom_ice.cpp:314),
     * so a ghost slot has no correct local value to compute — and it is TIME-INVARIANT, so it
     * is shipped ONCE at build as owner bytes (widening the area0/coriolis reply to 3 doubles)
     * rather than every step. Recomputing it locally would be the libmvec mistake in a new
     * costume: the owner's value is the only correct one, and it is free to carry. */
    Kokkos::View<real_t*> pfac_g;     /* [Eg]  per-step mEVP pressure_fac (carries det2) */
    Kokkos::View<int*>    ice_el_g;   /* [Eg]  per-step mEVP mean-msum element mask */
    Kokkos::View<real_t*> bcx;        /* [N+next] bc_index_nod2D, owner bytes, time-invariant */
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

/* M9 mEVP variants (cells ②/④). The prestep ship carries the same 11 fields as std EVP's, but
 * rhs_a/rhs_m must be shipped AFTER the owner has scaled them by area (trap 7) — the owner's
 * ssh2rhs scatter is order-sensitive, so its RESULT travels and is never recomputed.
 * The window ship is what the two cells differ by:
 *   div_on = false (cell ②) -> (u_aux, v_aux) + the carried element sigma segment
 *   div_on = true  (cell ④) -> (u_aux, v_aux, R_u, R_v), no element segment at all */
void fesom_evpwide_mevp_prestep_exchange(struct fesom_ice *ice, struct fesom_partit *p);
void fesom_evpwide_mevp_window_exchange(struct fesom_ice *ice, struct fesom_partit *p,
                                        int div_on);

/* Release device Views before Kokkos::finalize (fesom_main teardown; cgpipe rule). */
void fesom_evpwide_free(void);

#endif /* FESOM_ICE_EVPWIDE_H */
