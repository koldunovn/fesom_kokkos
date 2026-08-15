#ifndef FESOM_ICE_MAEVP_H
#define FESOM_ICE_MAEVP_H

struct fesom_ice;
struct fesom_partit;
struct fesom_mesh;

/*
 * mEVP sea-ice rheology (whichEVP=1) — literal port of the C oracle's fesom_ice_maevp.c,
 * itself a literal port of EVPdynamics_m (ice_maEVP.F90:429-882, the NR-optimised routine
 * with ssh2rhs / stress_tensor_m / stress2rhs_m inlined).
 *
 * Selected at runtime by FESOM_WHICH_EVP=1 (see fesom_ice.cpp). Standard EVP stays the
 * default. aEVP (whichEVP=2) is NOT ported and aborts at init.
 *
 * ════════════════════════════════════════════════════════════════════════════════════════
 *  ⚠️  mEVP-vs-std-EVP ASYMMETRIES — PORTED DELIBERATELY. DO NOT "FIX" THESE AGAINST THE
 *      fesom_ice_evp.cpp TEMPLATE. Every one of them is a real difference between the two
 *      rheologies; every one of them would look like a bug to a reader who knows std EVP.
 *      (Numbering follows the C oracle's checklist so the two trees stay cross-referenceable.)
 * ════════════════════════════════════════════════════════════════════════════════════════
 *   1. `rdt` is the FULL `ice_dt` — mEVP does NOT divide by evp_rheol_steps (std EVP does).
 *      The drag term CARRIES rdt; `drag·u_w` sits OUTSIDE the `rdt·(…)` group; `+beta·u_aux`
 *      closes the rhs.
 *   2. `pressure_fac` has NO 0.5. The 0.5 of std-EVP's ice_strength lives ONLY in the
 *      sigma11/sigma22 updates here — and NOT in sigma12.
 *   3. NO `theta_io` rotation anywhere. (theta_io=0.0 is in the namelist but is never read
 *      by this path.)
 *   4. Element mask is `mean(m_ice) > 0.01` (msum), NOT std EVP's criterion.
 *   5. Node mass regularisation is `M / ((1+M²)·area)` — verbatim, do not simplify.
 *   6. ASYMMETRIC assembly guards: the inlined **ssh2rhs** element loop writes ALL THREE
 *      nodes UNGUARDED (halo entries are written and never read — the arrays are node_size,
 *      so it is in-bounds), while the inlined **stress2rhs** loop is GUARDED to owned nodes
 *      (`if (eln[k] < myDim_nod2D)`). Both are ported as written.
 *   7. `rhs_a`/`rhs_m` are zeroed over myDim ONLY, and the rhs scaling by area happens
 *      INSIDE the ice branch of the node precompute.
 *   8. `u_ice_aux = u_ice` on entry and `u_ice = u_ice_aux` on exit both span the FULL
 *      myDim+eDim extent, and the final copy takes NO extra exchange (the aux halo is already
 *      current from the last substep's exchange).
 *   9. `bc_index_nod2D` multiplies the determinant in the node solve — redundant with the
 *      edge-BC loop that follows, but BOTH are ported. (This is why bc_index_nod2D is pushed
 *      to the device in fesom_ice.cpp; it was host-only before M6.2.)
 *  10. `delta_min` is ADDITIVE: `pressure = pressure_fac / (delta + delta_min)`.
 *  12. `sigma11/12/22` are NOT zeroed on entry — they PERSIST across calls (which is why they
 *      are on the IN sync rail).
 *  13. Non-ice nodes are SKIPPED in the node solve — there is NO else branch. A non-ice node
 *      KEEPS its u_ice_aux value. std EVP zeroes here; mEVP must not.
 * ════════════════════════════════════════════════════════════════════════════════════════
 *
 * Reference: jobs/m6_namelists/mevp/ (whichEVP=1 is the single knob vs the linfs+KPP
 * baseline). Bisection rails: /work/ab0995/a270088/port/mevp/{cdump_16r,fdump_16r}.
 */
void fesom_ice_evp_dynamics_m_kk(struct fesom_ice    *ice,
                                 struct fesom_partit *partit,
                                 struct fesom_mesh   *mesh);

/* Release the cached coastal-node mask BEFORE Kokkos::finalize() (a file-scope Kokkos::View is
 * otherwise destructed after finalize -> Kokkos aborts). Mirrors fesom_ice_evp_free(); called
 * next to it in fesom_main.cpp. */
void fesom_ice_maevp_free(void);

/*
 * M9 — nonzero when FESOM_SPEED_MEVPDIV is active (the divergence form carries R_u/R_v at nodes
 * instead of sigma11/12/22 at elements).
 *
 * Exposed for ONE caller: fesom_ice.cpp's mEVP branch, which must then skip the sigma
 * host<->device rails. Two reasons, and the second is the one that would have silently biased
 * the whole study:
 *   1. sigma is DEAD in this form — railing it is pure waste.
 *   2. those rails sit INSIDE the icedyn phasestats bracket (the mark is at fesom_ice.cpp:676,
 *      before the rheology branch), so leaving them in place while adding state would charge the
 *      divergence form extra PCIe inside the very metric it is being measured on.
 * The other fields on that rail (a/m/ms, srfoce_*, stress_atmice_*, uice/vice) are host-produced
 * and MUST still be railed — only the sigma triplet is skipped.
 */
int fesom_ice_maevp_div_active(void);

#endif /* FESOM_ICE_MAEVP_H */
