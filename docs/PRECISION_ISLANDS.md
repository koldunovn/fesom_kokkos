# FESOM2-Kokkos precision islands registry (M8)

The authoritative list of everything deliberately kept FP64 (`dbl_t`) while working precision
(`real_t`) is FP32. **Rules:** (1) after the M8 sweep, every `dbl_t` in the tree corresponds to a
row here; raw `double` in state/kernel code is a sweep bug. (2) A row is added/changed only with
evidence: either *pre-registered* (design rationale, cited) or *promoted* (the failing gate
signature that demanded it). (3) Every promotion re-measures the standard pinned pair and records
the prize give-back. The history of this file **is** the FESOM2 precision-sensitivity map.

Status legend: `pre-registered` (design-time, never flipped to FP32) · `suspect-fp32` (runs FP32
until a gate objects) · `promoted` (was FP32, failed a gate, now dbl_t — signature required) ·
`cleared` (suspect that passed all gates in FP32).

## Islands (never FP32)

| Island | Scope | Reason | Evidence | Status |
|---|---|---|---|---|
| Calendar / time (`fesom_calendar.*`) | module | ulp(FP32 s-in-day) ≈ 8 ms; 63-yr accumulation; already a separate double struct | design; PR-940 went FP32 here = latent sub-second-dt trap we refuse | pre-registered |
| All `parallel_reduce` accumulators (23 sites incl. headers) | scalars | FP32 sum over 1e6–1e9 elements loses ~n·2⁻²⁴; free in Kokkos (double acc over float views) | design; NEMO keeps sums DP; PR-940 does NOT (their untested risk) | pre-registered |
| MPI reduction scalars (Allreduce dots/norms/sums) | scalars | same as above; scalar MPI_DOUBLE costs nothing | design | pre-registered |
| CG scalar chain: residual, rtol, α, β (`fesom_ssh.cpp`) | scalars | stopping criterion + recurrence stability; vectors/SpMV stay FP32 | textbook mixed-precision CG | pre-registered |
| CGPIPE/CGPOLY eigen-bounds (λmax power iteration, Chebyshev coefficients) | scalars | pipelined CG is rounding-fragile; wrong λ bounds ⇒ divergent polynomial | design | pre-registered |
| Mesh metrics precompute (areas, gradient coeffs) | init path | compute FP64 from FP64 coords, then store `real_t`; never derive geometry from FP32 coords | design | pre-registered |
| Output / monthly-mean accumulators | buffers | FP64 sums of FP32 samples over ~1e4 steps | PR-940 lesson (their `local_values_r8`) | pre-registered |
| PHC climatology / init path (`fesom_phc.cpp`) | init path | runs once; DP costs nothing; removes init from suspect space | design | pre-registered |
| On-disk schema: restarts + means stay `NC_DOUBLE` | I/O | double holds every float exactly ⇒ SP→disk→SP bit-exact + SP-restart-into-DP exact (Task-6 gated, not assumed); DP→SP truncates by design (free only in the FP32→FP64 rescue direction) | design + PR-940 lesson 4 | pre-registered |
| Fill/missing-value handling | I/O | cast fill value to working precision BEFORE comparison (same rounding as data) | PR-940 bug class | pre-registered |
| Guard epsilons (additive `+eps` denominators) | constants | FP32 guard must be NORMAL under FTZ (≥ ~1e-20); `1e-40` flushes to 0 ⇒ 0/0 NaN at cold start (KPP class); FP64 values unchanged | **PR-940 headline bug** (their commit 48c37328); CUDA FTZ default | pre-registered |

## Suspects (FP32 until a gate objects — promotion order on failure)

| Suspect | Scope | Why suspect | Status |
|---|---|---|---|
| EOS + pressure-gradient chain (`fesom_eos.cpp`) | module | PGF cancellation; anomaly form (`density_m_rho0`) helps but unproven in FP32; PR-940 ran EOS in WP without incident (1-yr only) | suspect-fp32 (promote FIRST) |
| FCT tracer advection (`fesom_tracer_adv.cpp`, ice FCT) | module | conservation under limiter arithmetic over decades | suspect-fp32 |
| mEVP stress iteration (`fesom_ice_maevp.cpp`) | module | stiff subcycled relaxation; PR-940's ice was WP-disciplined and survived their (short) tests | suspect-fp32 |
| KPP stability functions | kernels | Richardson-number cancellations, exp/sqrt chains; PR-940's SP NaN lived here (guard class, fixed by epsilon policy) | suspect-fp32 |
| cvmix_TKE (ported, `fesom_cvmix_tke.hpp`) | module | PR-940 islanded ALL of CVMix as fixed-r8 — FP32 CVMix-TKE is untested anywhere; our port owns the code so it IS testable; the 63A/63B endgame config runs it | suspect-fp32 (**highest-probability promotion**) |

## Accumulation ledger (built during the Task-2 sweep; documentation only)

Every prognostic `state += dt·tendency` / running-sum site, with its typical increment/state
scale — the per-step low-bit-loss map for FP32 and the stagnation map for any future fp16 work
(increments below the state's ulp vanish). Target list for compensated summation / anomaly
variables (parked M9 track; user proposal 2026-07-19).

| Site | Quantity | typical Δ/state per step | Notes |
|---|---|---|---|
| *(filled slice-by-slice during Task 2)* | | | |

## Promotion log

*(empty — populated by gate failures: date, gate + signature, island added, pinned-pair give-back)*
