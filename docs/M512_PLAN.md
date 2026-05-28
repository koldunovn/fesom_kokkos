# M5.12 — Launch density reduction via kernel fusion

Branch: TBD (`m512-fusion` off `profile-m511` after merging C1+C2). Off `master` if M5.11 merges to master first.

## Why this lever (vs everything else)

From `docs/PROFILE_M59.md §7` and `KOKKOS_PORTING_LESSONS.md L52`: the GPU wall
on CORE2 dist_8 is **launch density**, NOT memory-coalescing (Lever C / M5.10b
already failed at the FCT phase — 1.3% step gain), NOT PCIe (M5.1+M5.4+M5.7
already reduced it 12× to 1 GB/step), NOT one fat compute kernel (the top
individual kernel is only 1.51% of step).

**Evidence**:
- 2450+ launches/step at dist_8 from 3 hot regions:
  - halo pack + unpack: 1052/step
  - CG iters: ~560/step
  - EVP per-subcycle: ~840/step
- Top 12 kernels combined < 8% of step share. No fat kernel exists.
- Per-call costs 13–30 µs — close to bare CUDA launch overhead floor (~5–10 µs).

**Lever**: collapse adjacent kernels to amortize launch overhead × per-kernel
fence cost. Tackle the high-launch-count regions first.

## Sub-milestones (each independently shippable)

### M5.12a — halo pack/unpack fusion

**Claim**: of the 526 halo brackets / step at dist_8, many of the `pack` calls
read fields that were JUST WRITTEN by an adjacent compute kernel, and many of
the `unpack` calls feed fields that are immediately consumed by the next
compute kernel. Fusing pack into the producer (a "compute-and-pack" combined
kernel) eliminates ~526 launches/step. Fusing unpack into the consumer does
the same on the receive side.

- **Files**: `src/fesom_halo_device.{hpp,cpp}`, the call sites in `fesom_step.cpp`
  and the high-frequency phases (FCT, CG, EVP, momentum).
- **Approach**: introduce a `pack_into(field, halo_buffer, …)` device functor
  that the producer can call inside its parallel_for body; symmetric `unpack_from`.
  Replace the standalone `fesom_halo_pack`/`unpack` launches at high-frequency
  sites only (don't touch the CG halo which is already tight).
- **Expected payoff**: ~1.5% direct (saves the launch+fence overhead for 526
  brackets × ~15 µs at dist_8), possibly more if natural async overlap is
  recovered.
- **Validation**: per-kernel verify gate (FESOM_KK_VERIFY=tradv,pp,kpp,ssh,evp)
  on Serial still PASSES; the gate against fresh oracle (today_serial)
  PASSES at noise floor; same-day perf baseline before/after on CORE2 dist_8.
- **Risk**: medium. Halo pack/unpack must preserve byte-exact semantics
  (the M5.1a halo path is already validated against host-halo at noise floor;
  changing the call shape could regress).

### M5.12b — EVP per-subcycle fusion

**Claim**: of the 7 EVP per-subcycle kernels (`s2rhs_zero`, `s2rhs_scatter`,
`s2rhs_final`, `evp_saveold`, `vel_update`, `stress_tensor`, `coastal_bc`),
several pairs read+write the same buffers in adjacent launches and can be
fused. Specifically:

- `s2rhs_zero` + `s2rhs_scatter` + `s2rhs_final` → one fused per-element kernel
  (the scatter writes nod3D with `atomic_add`; the final divides by mass — both
  can ride on the same data movement).
- `evp_saveold` + `vel_update` → one fused kernel (saveold reads uice; vel_update
  writes uice — fusion saves one read-write round trip).
- `stress_tensor` + `coastal_bc` → coastal_bc is a final per-node 0-write that
  can be folded into a stress_tensor's mask handling.

7 kernels × 120 subcycles → potentially 3 × 120 = 360 kernels (save 480 launches/step).

- **Files**: `src/fesom_ice_evp.cpp` (the 11 sites identified in
  `docs/plans/20260528-m511-profile-pass.md`).
- **Expected payoff**: ~2% direct (480 launches × ~30 µs fence overhead each).
- **Validation**: `FESOM_KK_VERIFY=evp` on Serial PASSES (the fused kernel
  must produce byte-identical Serial output); CORE2 active-ice gate PASSES.
- **Risk**: medium. The 2 atomic_add scatters (`s2rhs_scatter`, `step3_scatter`)
  reorder writes if fused with their producer — climate-close should hold but
  bit-identity on Serial requires the scatter remain in its own parallel_for OR
  the fused kernel preserves the scatter's iteration order.

### M5.12c — CG inner-loop fusion

**Claim**: the CG has 5 high-frequency kernels per iter (`psolve_dot2`,
`spmv_dot`, `axpy`, `pp`, plus the existing M5.2 SpMV+dot fusion). Some of
these have data dependencies that prevent fusion, but the AXPY chain after
SpMV can likely fuse one more level (`axpy(p)` + `axpy(x)` + `axpy(r)`).
~112 iters × 1 fused launch saved = 112 launches/step.

- **Files**: `src/fesom_ssh.cpp` (CG implementation).
- **Expected payoff**: low (CG is only ~5.7% of step at dist_8; fusing ~100
  launches saves ~3 ms/step = ~0.6%).
- **Validation**: bit-identity check via the existing CG-determinism gate
  (M5.2 SpMV+dot is bit-identical on Serial; the new fusion must be too).
- **Risk**: low (CG is well-localized; existing M5.2 fusion provides the
  template).

### M5.12d — smoother call count audit

**Claim**: `fesom_smooth_nod3D_kk` is called 10 times/step at dist_8 (1 bvfreq +
9 blmc-slab). The 9 blmc-slab calls run the same 9-sweep smoother on 3 different
2D slabs (a 3-channel blend via a `base` element offset, per `KOKKOS_HANDOFF.md`).
The 9 calls might collapse to 1 call that does all 3 slabs in one sweep with a
different inner loop structure. Per-call cost is **350 µs** — saving 6 calls
saves ~2 ms/step.

- **Files**: `src/fesom_kpp.cpp` (the 9-sweep smoother invocation), the
  smoother itself in `src/fesom_aux.cpp` or wherever
  `fesom_smooth_nod3D_kk` lives.
- **Expected payoff**: medium (saves ~0.4% but high signal-to-noise per call —
  the smoother's per-call cost is the biggest non-halo per-call kernel).
- **Validation**: `FESOM_KK_VERIFY=kpp` Serial bit-identical; gate PASSES.
- **Risk**: medium. Restructuring the smoother's slab loop can shift the
  arithmetic order; bit-identity is the gate.

## Cumulative target

Conservative: 4–7% step time reduction on CORE2 dist_8 (0.4777 → ~0.45 s/step).
Optimistic: 10% if all sub-milestones land + natural async overlap is recovered.

This is **modest but compounding**:
- M5.4 ocean halo flips: 25%
- M5.5 device smoother: +13% (hidden host cost)
- M5.7 KPP halo: +1%
- M5.8 EVP coastal BC: +0.5%
- M5.9-pin: +3.1%
- **M5.12 (this plan): +4–7%**

Cumulative from M5.0 baseline (0.86 s/step at M3.1) to projected M5.12 (~0.45 s/step):
**~48% improvement** on CORE2 dist_8.

## Validation must-haves (every sub-milestone)

1. **`FESOM_KK_VERIFY=<key>` on Serial PASSES** for every key the sub-milestone
   touches. Serial bit-identity is the bedrock.
2. **`scripts/gpu_fidelity_gate.sh` PASSES** at noise floor against the
   2026-05-28-refreshed oracle (or a freshly-built one if the sub-milestone
   touches anything in `fesom_step.cpp`).
3. **Same-day perf baseline** on CORE2 dist_8 with and without the change —
   apples-to-apples (no comparison to memory-recorded numbers from another day
   per L40).
4. **M3.2 climate fidelity must not regress** — the user's bottom line:
   "we have to be very, very close to Fortran and C port". A short-run gate
   PASS is necessary but not sufficient; a M3.2-style 1-yr CUDA spot-check
   is needed after each sub-milestone unless the per-kernel verify is
   bit-identical (in which case the M3.2 numerics can't shift).

## Why NOT these (discarded levers, kept for the record)

- **Rank-1 → rank-2 LayoutLeft memory coalescing** (Lever C, M5.10b): tried,
  failed. 1.3% on the dominant FCT phase. The atomic scatters are
  layout-agnostic and FCT's Zalesak limiter has branch divergence that
  coalescing can't help. See `docs/LEVER_C_PLAN.md` on the `leverC-coalesce`
  branch (its §RESULT section).
- **Bigger mesh to fill the A100s**: tried, doesn't flip the GPU↔CPU verdict
  (farc still 7× slower than CPU node-for-node). See `docs/SCALING_FARC.md`.
- **More halo flips**: diminishing returns. PCIe is 1 GB/step at dist_8;
  remaining ocean halos are small nod2D or one-per-step.
- **Mixed precision (FP32)**: would risk climate fidelity. Each FP32 kernel
  needs its own M3.2 re-validation. Not a fast win.
