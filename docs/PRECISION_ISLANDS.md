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
| `fesom_momentum.cpp:1106/1154` | uv (horiz. velocity) | O(dt·rhs) | `uv += uv_rhs + F` (dt-scaled tendency + ssh gradient); C/KK twins |
| `fesom_momentum.cpp:1676/1779` + `fesom_ale.cpp:788` | hbar / eta_n (SSH) | O(dt·rhs/area) | `hbar = hbar_old + ssh_rhs_old·dt/area`; time-filter blend |
| `fesom_tracer_adv.cpp:812/1888` | T,S advection | O(dt) | `del_ttf += T·(hnode_old−hnode_new); T += del_ttf/hnode_new` |
| `fesom_tracer_diff.cpp:344/648` (+304/614 BC, 320/627 SW) | T,S vert-diff | O(dt) | TDMA increment apply; shortwave `+= (top−bot·ar)·dt` |
| `fesom_ice_evp.cpp:824/853` | uice,vice | O(rdt) | EVP subcycle implicit-drag update |
| `fesom_ice_fct.cpp:513,521/918+` | m_ice,a_ice,m_snow | O(flux) | FCT low-order + limited-flux apply |
| `fesom_ice_thermo.cpp:270/722, 390/784, 44/667, 367-368/769-770, 471/845` | ice thermo | mixed | Newton `t += res/deriv`; `hsn += snowfall·dt`; flood; growth sums; fw flux |
| `fesom_cvmix_tke.hpp:326,330,363` | TKE energy | O(dt) | forc[] assembly + implicit TDMA tke_new |
| `fesom_gm.cpp:775/2067` | GM tracer apply | O(dt) | `values += skew-flux div · dt/(av·hn)` |
| `fesom_eos.cpp:250/406` | hpressure | depth recurrence | deep-column running sum — classic FP32 hazard class (not time-accumulation) |
| `fesom_io.cpp:713-819/834-997` + `fesom_io_stream.cpp:367` | output means | O(1/nsamples) | `out[i] += src[i]` into real_t accumulators then `*= inv` — **candidate dbl_t promotion if Gate-4 means look degraded** |
| `fesom_mesh.cpp` compute_node_areas | ocean_area local sum | init-once | local area sum accumulates in real_t before the (now dbl_t) Allreduce — flag: promote local sum to dbl_t during Task 6 if inspection confirms |

## Promotion log

- **2026-07-19 — RULE ESTABLISHED (the first SP-only bug): every `MPI_DOUBLE` reduce buffer MUST
  be `dbl_t`, never `real_t`.** Found by the Gate-2 fleet: `fesom_main.cpp` step-diag Allreduce
  staged `real_t buf_max[16]` under `16, MPI_DOUBLE` ⇒ 2× over-read AND over-write = stack smash
  at npes>1 ⇒ step-1-clean/step-2-NaN (Z7 shape), partition-layout-dependent (np2 survived by
  luck, 128/256 ranks died). One bug explained all three f32 leg failures. Fix `7e90742`;
  tree-wide re-audit clean; `mp32-0` retired → `mp32-1`. Grep-enforceable invariant:
  `MPI_DOUBLE` may only touch `dbl_t`/`double` storage; `FESOM_MPI_REAL` only `real_t` storage.

- **2026-07-19 — PROMOTED: JRA forcing-time machinery → calendar island (first `FieldT<dbl_t>`
  use).** Signature: deterministic global T/S NaN at ~9.4 model-hours (dt1800 step 19 / dt900
  step 37), scheme-independent, rank-independent. Root cause: absolute-Julian-day time axis
  (~2.44e6 d; float ulp = 6 h > 3-h records) ⇒ record-time collision ⇒ delta_t=0 ⇒ Inf coefs ⇒
  NaN forcing everywhere (SP_PORTING_LESSONS.md SP2). Promoted: rdate/timenew/nc_time/delta_t +
  coef_a/coef_b (+ faithful dbl_t binarysearch twin). Give-back: ~none (8 nod2D coef pairs
  double = KBs; interp math per node unchanged count). Fix `7247412`; FP64 bit-identity re-proven.
  ⚠️ Same design in Fortran `gen_surface_forcing.F90` (PR-940 SP branch demoted nc_time to WP —
  latent) and JAX `jra55.py` (safe only via forced x64) — report upstream.

*(further entries: date, gate + signature, island added, pinned-pair give-back)*
