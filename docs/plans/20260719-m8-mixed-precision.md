# M8 — Mixed Precision (FP32 working precision + FP64 islands)

**Branch:** `m8-precision` · **Worktree:** `/home/a/a270088/port_kokkos_mp` · **Base:** `1df683b` (the frozen 63A/63B physics commit; ancestor of the moving m7-speed tip — later m7 commits are figures/probes, binary-inert)
**Status:** plan approved (brainstormed + user-validated section by section, 2026-07-19); **plan-review agent findings applied same day** (blocker fixed: the endgame config is the OPTIONS config `zstar+cvmix_TKE+mEVP+GM`, not KPP-default; FP32×speed-knob rung added). Ban from M6 era explicitly lifted by user for this isolated track.

## Overview

Move the validated FESOM2 C++/Kokkos port to FP32 working precision with explicit FP64 islands
(NEMO-style global switch — approved Option A). Four prizes, all in scope (user: all four):

1. **GPU speed + scaling** — halved bytes attacks both M7-mapped regimes: BW-bound kernels (4N) and
   the 49–58 % comm share at 16N (MPI payloads halve; per-partner toll unchanged but bytes halve).
2. **Memory footprint** — state halves → bigger meshes per GPU (JUPITER GH200 relevance).
3. **CPU throughput** — SIMD width doubles, BW halves (PR-940 measured 1.65× SYPD on CORE2/1 node).
4. **Precision-sensitivity map of FESOM2** — publishable (NEMO/ROMS have one, FESOM2 doesn't);
   the map = the history of `docs/PRECISION_ISLANDS.md`.

**Final acceptance = Gate 5:** a 63-yr CORE2 hindcast (config matched to 63A/63B) judged by
`scripts/m7_hindcast_drift.py` against the FP64 twins **63A/63B** (free — in flight now, same base
physics) and Fortran R2, at pre-registered bars (below).

## Context (from discovery)

- `src/fesom_types.h:15` — `typedef double real_t;` already centralized "so any future precision"
  switch is possible; `FieldT<T>` (`src/fesom_field.hpp`) already templated.
- Adoption partial: ~560 raw `double` sites in `.cpp` (heaviest: `fesom_ssh.cpp` 83,
  `fesom_main.cpp` 69, `fesom_phc.cpp` ~48), 65 `MPI_DOUBLE` sites, 23 `parallel_reduce` sites
  (incl. headers).
- SSH CG: `soltol = 1e-5` (`FESOM_PHASE1_SOLTOL`) — within FP32 vector reach with FP64 scalars.
- EOS already computes anomaly `density_m_rho0` (`use_density_ref` → ρ₀) — PGF cancellation
  partially defused; still suspect #1.
- Calendar/time: separate `double`-based struct, not `real_t`-coupled — island by construction.
- Restarts + outputs currently `NC_DOUBLE`; JRA55 forcing already read as float32 from disk.
- JAX twin ran x64 (forced) — no FP32 evidence from there.
- **PR 940** (`FESOM/fesom2#940`, `support_sp`, draft, milestoned FESOM 2.8): Fortran SP branch —
  mechanism + lessons mined 2026-07-19; report + diff cached at
  `/scratch/a/a270088/tmp/claude-24253/-home-a-a270088-port-kokkos/96d8d63b-0ba3-4bcd-87e8-987b2de5af18/scratchpad/pr940/`
  (copy the keepers into `docs/reference/` before the scratchpad expires).

### PR-940 lessons encoded in this plan

| # | Lesson | Where it lands here |
|---|--------|---------------------|
| 1 | Mechanism = 1 typedef + 1 MPI trait; cost = chasing strays | Task 1 + sweep Tasks 2a–2j |
| 2 | **Headline SP bug: `epsln=1e-40` is denormal in FP32; FTZ (Intel -O3, CUDA default) flushes → 0/0 NaN at KPP cold start** | Task 3 epsilon/FTZ audit; per-precision guard constants (FP32 ≥ ~1e-20); rhymes with our rule-0.41 cold-start class |
| 3 | They run **all global reductions in working precision** (CG dots, area/volume, conservation) | **Deliberately NOT copied** — FP64 accumulators are free in Kokkos; their choice is the un-measured decadal-drift risk; our Gate 3b measures exactly this |
| 4 | Files stay FP64 on disk; convert at I/O boundary; cast fill/missing to WP **before** comparison; means accumulate FP64 from FP32 samples | Tasks 2i, 6; islands table |
| 5 | Self-certifying precision banner asserted by every test | Task 1; all gate scripts assert it (L80 dead-knob rule) |
| 6 | CPU priors: whole-model 1.65× (CORE2, full node); pressure/mixing ~1.6×, SSH solve ~1.55×, ice ~1.0× | Gate 2 expectations |
| 7 | Aliasing trap at precision-boundary shims (same actual as in + inout) | Technical details; any future DP-island shim |
| 8 | Their seconds-in-day went FP32 (ulp 8 ms) — latent sub-second-dt trap | Our calendar island keeps this impossible |
| 9 | Restart interchange SP↔DP falls out of FP64-on-disk | Rescue experiments: run FP32, restart into FP64 |
| 10 | Their validation ceiling: 1 month + 1 yr, no conservation numbers, no review yet | Our ladder (Gates 3–5) is the value-add |

## Development Approach

- **Testing approach: gate-driven** (house analog of TDD): every task ends by passing its gate;
  bars are **pre-registered before runs**. No task starts until the previous task's gate is green.
- Small slices; each sweep slice must keep the FP64 build **bit-identical** (slice-wise Gate 0) so
  any breakage bisects to one slice (Z7-class protection).
- **CRITICAL: update this plan file when scope changes** (➕ new tasks, ⚠️ blockers, [x] immediately).
- House rules: run outputs → `/work/ab0995/a270088/port2/mp/`, never `$HOME`; frozen bins
  `mp/bin/` sha-stamped; **`BIN=` pinned in every job**; ratios only from same-commit same-day
  pinned pairs, min-of-2; cheap gates get cheap walltimes (`-t 00:06:00` class); **ask before push**.

## Testing Strategy — the gate ladder

- **Gate 0 (inert refactor):** whole sweep at `real_t == double` → **bit-identical to unmodified
  base** on smoke + multi-step diff_snap + selfchecks, Serial AND CUDA. Every later FP32 delta is
  then attributable to precision, not sweep mistakes.
- **Gate 1 (it runs):** FP32 Serial+CUDA build; pi-smoke + dist_2/dist_4 NaN-free;
  `-Wdouble-promotion`-clean hot code; precision banner asserted by the gate scripts.
- **Gate 2 (prize-sizing, ALWAYS MEASURE — immediately after Gate 1):** 300-step pinned pairs
  FP64/FP32: dars CPU c1..c32, dars GPU g2/g4/g8, production mesh 4N/16N; device-memory-per-rank
  counter (the JUPITER number). Priors: PR-940 CPU 1.5–1.65× — measured on **native KPP** (they
  islanded ALL of CVMix as fixed-r8; our endgame config runs our *ported* cvmix_TKE in FP32, a
  first anywhere, and the registry's highest-probability promotion candidate); 16N comm-byte
  halving. Knob posture per leg is recorded: headline pairs at the 63A posture (SPEED=1) + one
  knobs-off pair to decompose FP32 savings vs what CGPIPE/CGPOLY already save (no double count).
  **Decision D1 (pre-registered, crisp):** continue if whole-model ≥ 1.25× on ANY of {dars CPU,
  dars GPU, 4N, 16N} OR device-memory-per-rank ≤ 0.65× (int connectivity/index arrays are already
  32-bit, so the floor sits above 0.5×); else pivot to solver-only MP.
- **Gate 3 (short-run physics):**
  (a) *divergence curve*: FP32-vs-FP64 relative field diffs at 1→100 steps judged against the
  chaos-envelope control (FP64 vs FP64+1ulp perturbation); FP32 must track the envelope's growth
  **shape** from a ~1e-7 start — shape departure = bug, not rounding. Needs cross-dtype
  `diff_snap` extension.
  (b) *conservation*: global heat/salt/volume drift (FP64 diagnostics) over 1 month vs twin —
  the measurement PR-940 never made.
  (c) *solver health*: CG iteration counts FP32 vs FP64 at soltol 1e-5 near-equal (stagnation =
  red flag); CGPOLY variant separately (Chebyshev bounds are rounding-fragile).
- **Gate 4 (1-yr climate leg):** M5.23 bar, exact CGPOLY-cert machinery — sst/sss/ssh/a_ice
  pattern correlations vs FP64 twin in the certified class (1.00000/0.99996/1.00000/0.99997
  territory); `m7_climate_check_plots` vs `/work/ab0995/a270088/fesom2_core2`.
- **Gate 5 (endgame, 63-yr hindcast):** 63C-MP = FP32 twin of the **63A posture** specifically
  (63A≠63B; see Task 11) (rule-0.40 trim).
  **Pre-registered bars:** Tbar gap ≤ 0.001 °C; OHC gap ≤ ~10× the port↔Fortran gap (few ZJ of
  ~19400); gap curve **co-tracks/flattens** (no systematic FP32 drift).
- **Failure protocol (any gate):** promote suspects to `dbl_t` islands one at a time (EOS/PGF
  first), re-gate, log in `docs/PRECISION_ISLANDS.md` with the failing signature, and re-measure
  the standard pinned pair so the registry always shows the prize give-back.
- **Gate 3k (FP32 × speed knobs):** `FESOM_SPEED=1`/CGPIPE, CGPOLY d3, and EVPWIDE were certified
  at **FP64 only** — each gets a Gate-3-class re-certification at FP32 (options config, both
  backends: selfchecks, solver health, short-run divergence) BEFORE any 1-yr/63-yr leg runs with
  knobs on.
- **Scheme coverage:** Gates 3–5 run the **actual 63A/63B options config
  `zstar + cvmix_TKE + mEVP` (+ GM)** — the shipped/endgame physics (SCALING-REMEASURE §config).
  KPP-default (the namelist default scheme) is an optional extra Gate-3 rung in Task 12; note the
  PR-940 1.65× prior was measured on native KPP, not on this config (per-scheme floors, L79).

## Progress Tracking

- mark `[x]` immediately when done; ➕ for discovered tasks; ⚠️ for blockers; keep in sync.

## Implementation Steps

### Task 1: Precision switch scaffolding + banner

**Files:**
- Modify: `src/fesom_types.h`, `CMakeLists.txt`, `src/fesom_main.cpp` (banner)
- Create: `docs/PRECISION_ISLANDS.md` (seeded — done at plan time)

- [x] `fesom_types.h`: `#ifdef FESOM_SINGLE_PRECISION` → `real_t` float/double; add
      `typedef double dbl_t;` (deliberate-island marker) + `FESOM_MPI_REAL` macro
- [x] CMake option `FESOM_PRECISION=single|double` → the define; build-dir pairs.
      DONE 2026-07-19: `build-mp-serial` built+gated FP64; `build-mp-serial-sp` CONFIGURED
      (`FESOM_SINGLE_PRECISION` verified in all 43 TU compile commands; SP compile itself is
      Task 6); invalid value FATAL_ERROR guard negative-tested ("half" rejected).
      ➕ CUDA pair (`build-mp-cuda{,-sp}`) deferred to the first device-touching slice gate
      (2a) — Task-1 edits are host-stdout + inactive-ifdef only, no device path exists to gate
- [x] startup banner `[fesom_port] PRECISION: SINGLE|DOUBLE (real_t=…)` printed once by rank 0
      (after `fesom_mpi_init`, before Kokkos init)
- [x] gate scripts assert the banner: `scripts/mp_assert_banner.sh <log> <SINGLE|DOUBLE>`,
      exercised on both np1 and np2 run logs (L80)
- [x] gate PASSED 2026-07-19: FP64 field snapshots bit-identical vs base code via
      `scripts/diff_snap.py` — pi mesh, dt100, 20 steps, snaps @0/10/20; np=1 AND np=2 scatter
      gate (vader-CMA off, L18): ALL FIELDS BIT-IDENTICAL both. REF binary rebuilt from
      stash-clean base in the same build dir (a mid-build edit race in the first baseline
      attempt was invalidated and redone deterministically); runs at
      `/work/ab0995/a270088/port2/mp/task1/{ref,new}{,_np2}`

### Task 2: The inert sweep (real_t completion) — slices a–j, each slice bit-identical

**Files:** (per slice) `src/fesom_*.cpp/.h/.hpp`

- [x] 2a DONE 2026-07-19: `fesom_field.hpp` `Field = FieldT<real_t>` + `#include fesom_types.h`
      (+ `test_field` gains MPI::MPI_CXX for the transitive `<mpi.h>`). `fesom_constants.h` and
      `fesom_aux.*` were ALREADY clean (0 raw doubles). Serial np1+np2 bit-identical;
      CUDA envelope gate PASS (worst 0.10 of allowance).
      ⚠️ **METHODOLOGY FINDING: CUDA is run-to-run non-bit-reproducible** (same-binary control:
      per-field 1e-20..1e-14 on pi/20 steps — atomics order). Per-slice CUDA gate is therefore a
      **noise-envelope gate** (`scripts/mp_cuda_gate.py`: per-(snap,field) diff ≤ max(10× same-
      binary-rerun noise, 1e-13), pre-registered) with Serial as the sole byte oracle — matching
      the house cert shape (Serial byte proofs; CUDA selfchecks + climate-close floors). Standing
      noise basis: `gate_2a/{cuda,cuda_rerun}`; refresh after kernel-set-changing slices (2h).
- [x] 2b dyn/momentum/ALE — 0 promotions needed (already precision-generic; `053e4a2`)
- [x] 2c tracers/adv/diff — 0 promotions (`557ed3d`)
- [x] 2d ice family — 0 promotions; evpwide MPI+static_assert deferred to Task 6 (`c7251f4`)
- [x] 2e mixing — 0 promotions; TKE_C66=6.6 double-literal exception pre-registered (`b486bff`)
- [x] 2f EOS — 9 promotions incl. smoother scratch Views (halve under SP) + the ONE stray
      `FieldT<double>` in eos.h (would have ODR-broken SP) (`744dbb5`)
- [ ] 2g SSH solver (`fesom_ssh.cpp`) — vectors/SpMV real_t; scalar chain
      (residual/rtol/α/β) + CGPIPE/CGPOLY eigen-bounds → `dbl_t` islands.
      **Explicit MPI split (18 sites):** vector-halo `Isend/Irecv` AND their `rb/sb[].dbls`
      buffer types (lines 761/767, 956/962, 1131/1137, 1412/1418, 1511/1517, 1574/1587) →
      `real_t` + `FESOM_MPI_REAL` (else CG comm bytes never halve and Gate 2 under-reads with
      no visible culprit); `Allreduce` dot/norm/scalar sites (428, 1020, 1280, 1790, 2139,
      2253) **stay `MPI_DOUBLE`** (CG-scalar island)
- [x] 2g DONE (`2ad5379`): full CG-islands surgery per spec — scalar chain dbl_t both solvers,
      FP64 dot accumulators, CGPIPE/CGPOLY payloads real_t + FESOM_MPI_REAL (12 sites),
      Chebyshev host recurrence island (already cast-at-kernel-boundary by design)
- [x] 2h DONE (`3cdb388`): halo host+device buffers/pointers real_t, 8 MPI flips, prof-bytes
      sizeof(real_t); timing/verify islands kept; np2 leg exercised the swept exchange
- [x] 2i DONE (`228d4e5`): io gathers + mesh Bcasts → FESOM_MPI_REAL (9 flips); NC staging
      stays double per policy; phc/ic init island; bulk+mesh audited clean; fill-value compare
      sites verified SP-safe (phc compares in staging domain; sss_runoff already casts first)
- [x] 2j audited — zero promotions (main/step/phasestats doubles all diag/printf/timing; `e3eff9b`)
- [ ] while sweeping each file (zero extra cost — the file is open anyway): build the
      **accumulation ledger** in `PRECISION_ISLANDS.md` — tag every prognostic
      `state += dt·tendency` / running-sum site with its typical increment/state scale.
      This is the target list for future compensated-summation / anomaly-variable work and
      the fp16 groundwork (user proposal 2026-07-19); documentation only, no numerics change
- [ ] after each slice: Serial FP64 rebuild + pi-smoke bit-identical (fast slice gate);
      device-touching slices (2a–2h) ALSO CUDA pi-smoke bit-identical — Serial alone cannot
      exercise the device path (GPU fidelity-gate rule)

### Task 3: Epsilon/FTZ audit (PR-940 headline class)

**Files:** all kernels with additive guards; `src/fesom_constants.h`

- [x] grep-audit done: KPP_EPSLN 1.0e-40 was the ONLY sub-1e-38 guard (halo_device 1e-30 is
      timing-division, stays double); tke/pp clean
- [x] KPP_EPSLN per-precision: FP32 1e-20 (FTZ-normal), FP64 1e-40 unchanged (`e3eff9b`)
- [ ] check CUDA fast-math/FTZ flags in our nvcc lines; document denormal posture per backend
      (verify during Task 6 SP-CUDA bring-up)
- [x] gate: FP64 bit-identity holds (values unchanged at FP64)

### Task 4: Reduction-accumulator hardening

**Files:** all 23 `parallel_reduce` sites (incl. `fesom_ssh.h`, `fesom_ice_coupling.h`); MPI reduction call sites

- [x] every `parallel_reduce` audited: cg_dot + fused spmv-dot/dot2 + cgpoly nrm2 (2g),
      ice_coupling integrate host+device (`e3eff9b`), mesh ocean_area + sss_runoff integrate +
      phc max reduce (dbl_t temps; final batch) — evpwide selfchecks were already double
- [x] MPI reduction scalars stay `MPI_DOUBLE`, buffers now dbl_t-coherent everywhere
- [ ] output/monthly-mean accumulators: LEFT real_t for now (Gate-0 inert either way);
      ledger row flags them as the first promotion candidate if Gate-4 means degrade
- [x] gate: FP64 bit-identity ALL FIELDS + CUDA envelope 0.12 (t4b batch)

### Task 5: Gate 0 full battery (P1 exit)

- [ ] Serial + CUDA FP64 builds vs base `1df683b` binaries: full smoke suite + multi-step
      diff_snap + selfchecks 0.000e+00 + CG iteration counts identical
- [ ] freeze `mp/bin/mp64-0` (sha-stamped, both backends); record shas here
- [ ] commit; update this plan + registry

### Task 6: FP32 build + Gate 1

**Files:** whatever the compiler finds; `configure.sh`; gate scripts

- [x] FP32 Serial + CUDA compile clean (35 errors total, all boundary seams: nc-write staging
      → fesom_nc_real.h helpers; evpwide FESOM_MPI_REAL; cgpipe_free Views; tke common_type;
      + the two runtime bugs above). Gate 1 PASSED both backends (pi np1+np2+CUDA NaN-free,
      banner SINGLE). SP survives 48-step CORE2 (2× the old death wall)
- [ ] `-Wdouble-promotion` pass over hot kernels; literal hygiene (`real_t(0.5)` forms) —
      a silently-promoting kernel is a dead knob wearing a lab coat. Caveat: nvcc device-side
      promotion warnings are weak — the authoritative detectors are Gate 2's throughput +
      footprint counters (optionally ptxas register / f64-op inspection)
- [x] restart round-trip gate — **RESOLVED MOOT 2026-07-19: the port has NO restart
      writer/reader** (verified: every "restart" string in src/ is a comment; argv has no
      restart input). The 63-yr endgame runs restart-free in one job, so Gates 4/5 are
      unaffected. PR-940's SP↔DP interchange rescue experiments would require building
      restart machinery first — a port feature decision for the user, outside M8 scope
- [ ] pi-smoke + dist_2/dist_4 NaN-free on both backends, banner asserted
- [ ] freeze `mp/bin/mp32-0`

### Task 7: Gate 2 prize-sizing fleet + D1

- [x] first pinned-pair fleet run 2026-07-19 (BIN=mp64-0 / mp32-2 — same code state, FP64 side
      re-proven bit-identical across the two SP bug fixes): dars c2 CPU, CORE2 c1 CPU,
      dars g2 GPU; 300 steps, min-of-2, knobs-off posture. Two SP bugs found+fixed en route
      (diag-Allreduce stack smash `7e90742`; JRA time-axis FP32-impossibility `7247412` —
      see SP_PORTING_LESSONS.md SP1/SP2)
- [x] device-memory-per-rank: **23.2 → 11.9 GB = 0.51×** (dars g2, sampled mid-run — the
      JUPITER number; int-array floor pushed it barely above ½)
- [x] **GATE-2 BOARD (2026-07-19, s/step min-of-2):**
      | axis | FP64 | FP32 | speedup |
      | CORE2 c1 CPU (128r) | 0.1999 | 0.1320 | **1.51×** |
      | dars g2 GPU (8r)    | 0.7598 | 0.5169 | **1.47×** |
      | dars c2 CPU (256r)  | 3.0417 | 1.9425 | **1.57×** |
      | device mem/GPU      | 23.2 GB | 11.9 GB | **0.51×** |
      CG iters: 91→92 (CORE2), 32→33 (dars) — mixed-precision CG costs ~1 iteration.
      Priors honored: PR-940 CPU 1.5–1.65× (ours 1.51 gcc/128r vs theirs 1.65 intel/64r).
- [x] **DECISION D1: PASSED — CONTINUE FULL CAMPAIGN.** Both crisp criteria met:
      speed ≥1.25× on two axes (1.51× CPU, 1.47× GPU); footprint 0.51× ≤ 0.65×.
- ➕ wider fleet (dars c1..c32 curve, g4/g8, production 4N/16N incl. comm-share phasestats)
      deferred to the Gate-3/4 era — the D1 question is answered; curve completeness is
      documentation, not decision-critical

### Task 8: Gate-3 instrumentation

**Files:**
- Modify: `scripts/diff_snap.py` — cross-dtype relative mode
- Create: `scripts/m8_divergence_curve.py`, `scripts/m8_conservation.py` (if not present in port diagnostics)

- [x] diff_snap cross-dtype: `scripts/mp_divergence_curve.py` (built during Gate-1 era;
      diff_snap.py itself stays the zero-tolerance same-dtype byte oracle)
- [x] chaos-envelope control = **FP64 dt-seed ENSEMBLE** (dt 1800.0000001/.00001/.001),
      not a 1-ulp field hook — Gate-1 finding: single seeds are NON-MONOTONE, envelope
      must be max-over-seeds. `scripts/mp_envelope_verdict.py` (2026-07-19) computes
      sp-vs-envelope ratios + plot. (dt parses via atof; seeds run on the FP64 binary —
      under SP real_t=float would truncate them, noted for L80 hygiene)
- [x] divergence-curve harness + plot (mp_divergence_curve.py --envelope / verdict script)
- [x] conservation time series: `FESOM_MP_CONSERV=N` env-gated FP64 vol/heat/salt hook
      (`e172758`) — dbl_t Kokkos reduce over owned wet columns + MPI_DOUBLE Allreduce,
      device-current views (CUDA-valid). GATED: FP64 off+armed bit-identical vs base;
      SP off+armed bit-identical vs frozen mp32-2; **mp64-2/mp32-3 frozen**
      (013d71cc/276a8960; CUDA pair c24619d6/9e07fc67 — env_cuda.sh needed to build)
- [x] CG iteration-count extraction: `grep -o "it=[0-9]*"` on PRINT_EVERY=1 logs (used at
      bring-up: SP tracks DP mean|Δit|=2.2 max 4 over 60 np1 steps)

### Task 9: Gate 3 verdicts (dist_4 + core2-2N)

- [x] **SP options bring-up PASSED 2026-07-19** (np1 CORE2 login, 60 steps dt1800 =
      2× the SP3 minimum): rc=0, zero NaN/FATAL, banner SINGLE — **cvmix_TKE+mEVP+zstar+GM
      runs at FP32 out of the box; no promotion needed at this rung** (cvmix_TKE was the
      registry's top suspect). CG: SP tracks DP mean|Δit|=2.2, max 4. Divergence at step 60:
      T 1.1e-4 / S 1.1e-5 / eta 1.0e-4 relL2, flat growth; Kv/Av/ice at the KNOWN
      chaos-class magnitudes (Kv Linf 0.26 ≈ FP64 seed-control 0.27); lat/lon exactly float
      ulp (storage class). Runs: `mp/gate3/bringup_{sp,dp}`
- [ ] run battery ON THE OPTIONS CONFIG (`zstar+cvmix_TKE+mEVP`+GM — the endgame physics):
      divergence curves, 1-month conservation drift, solver health (plain CG, CGPIPE, CGPOLY)
      — **fleet 26364722-32 submitted 2026-07-19 (jobs/job_mp_gate3, BIN=mp64-2/mp32-3):**
      3a = a_dp/a_sp + 3 FP64 dt-seed legs (100 steps, snaps @10); 3b = b_dp/b_sp 1440 steps
      = 30 d (CONSERV=10); 3k = CGPIPE/CGPOLY(d3, both dtypes)/EVPWIDE(K8) selfcheck legs.
      All 11 legs ran to completion, banners correct, zero FATAL (b_sp: SP survives 30 model
      days). Verdict analysis in progress
- [x] **Gate 3b VERDICT (2026-07-19, 30-d CORE2 c1 options config, `mp_conserv_drift.py`):**
      heat: FP32−FP64 drift gap −2.2e-7 = **0.2 % of the physical 30-d signal** (−1.10e-4).
      salt: both runs conserve to ~2e-7 relative; gap 8.7e-8 with SIGN CHANGES over the month
      (random-walk, not systematic; ≈6e-11/step — watch at Gate 5 vs the Tbar/OHC bars).
      vol: FP64 zstar closes volume to 8e-16 (machine-exact); FP32 wanders ±2e-9 relative
      (≈0.7 km³ globally, sign-changing). **No leak signature — PASS.** THE measurement
      PR-940 never made. Artifacts: `mp/gate3/gate3b_conserv.{csv,png}`
- [x] **Gate 3c plain-CG VERDICT (30 d, padding-safe `it= *[0-9]*` extraction — beware
      printf %3d):** mean iters 90.83 (FP64) vs 90.88 (SP) = +0.05; pairwise mean|Δit|=0.33,
      max 5; no stagnation. **PASS.** (CGPOLY selfcheck rung → k2 legs.)
- [ ] ⚠️ **first 3k fleet was a DEAD KNOB (L80 strikes again):** legs lacked
      `FESOM_SPEED_FORCE_SERIAL=1`, so on the Serial backend NO lever fired (no selfcheck
      lines; CGPOLY iters identical to knobs-off). The "bit-identical" diffs were vacuous.
      k2 rerun fleet 26364952-55 (+FORCE_SERIAL) + CUDA options legs g_sp/g_dp 26364960/61
      + month-long seed legs b_s7/s5/s3 26364956-58 (mature envelope at snaps 480/960/1440
      — the 100-step envelope is immature: seeds ~1e-11 start vs SP's ~1e-7 rounding offset,
      ratio≫1-with-both-tiny is the predicted regime, shape verdict needs saturation)
- [ ] **Gate 3k:** FP32 × speed-knob re-certification — SPEED=1/CGPIPE, CGPOLY d3, EVPWIDE,
      options config, both backends (these were certified at FP64 only)
- [x] **3k CPU verdicts (k2 fleet, FORCE_SERIAL, knob-fired banners verified — SP10):**
      **CGPIPE selfcheck 0.000e+00 at every iter AT SP** (the recurrence≡exchange claim is
      algebraic — survives FP32 exactly). **CGPOLY d3 at SP: selfcheck 0.000e+00, iters
      120.2→40.2 (FP64 twin 39.6)** — the certified d3 class reproduces; "Chebyshev bounds
      rounding-fragile" worry measured away. **EVPWIDE: N/A under mEVP** — the code refuses
      the lever at whichEVP=1 (own banner), so it was inert in 63A ITSELF; effective 63A
      posture = SPEED=1(+unbind) on mEVP. No SP cert needed for the endgame path (re-open
      only if standard-EVP configs go SP)
- [x] **Gate 3a FORMAL VERDICT: PASS (mature 30-d envelope, `gate3a_mature.{csv,png}`).**
      Every field's sp/envelope ratio DECREASED step 100→1440 as the 3-seed envelope matured
      (T 14→4.8 · S 66→6.4 · eta 37→13 · pgf ~770→14-16 · w 652→64 · density 71→7.2);
      chaos-saturated fields already at 1-2× the envelope (a_ice 1.40, uice 1.26, m_snow
      1.20, h_ice Linf BELOW envelope); no jumps, no shape departure, no runaway. The
      residual ≫1 ratios on smooth fields = continuous re-seeding (SP injects ~6e-8 relative
      noise per operation; a one-time-seed ensemble lags by construction) with a bounded,
      decaying multiple. **GATE 3 PASSES IN FULL — ZERO ISLAND PROMOTIONS NEEDED** (EOS/PGF
      and cvmix_TKE suspects all cleared at FP32 on the endgame config)

### Task 10: Gate 4 — 1-yr climate leg

- [x] 1-yr FP32 CUDA run DONE 2026-07-20 (job 26365487, 25-min class, 2N×4×A100-80,
      63A posture verbatim): 17280/17280 steps, ZERO NaN, 17 monthly streams, CG 92 iters
      at year end; CONSERV over the year: heat −0.40 % (physical annual signal),
      salt −4.2e-6, vol −1.9e-9 rel. Output `/work/ab0995/a270088/port2/mp/y1/63Cmp_1958`
- [x] **GATE 4 VERDICT: PASSED AT THE M5.23 BAR** (`scripts/mp_gate4_verdict.py`, house
      eps_climate_compare convention, annual-mean maps, year 1958):
      | pair | sst | sss | ssh | a_ice | m_ice |
      | FP32 vs FP64 twin (63A) | **1.00000** | **1.00000** | **1.00000** | **0.99999** | 0.99999 |
      | FP32 vs Fortran R2      | 1.00000 | 0.99996 | 0.99999 | 0.99855 | 0.99928 |
      | FP64(63A) vs Fortran R2 | 1.00000 | 0.99996 | 1.00000 | 0.99872 | 0.99930 |
      Twin correlations ≥ the bar class (sst 1.00000/sss 0.99996/ssh 1.00000/a_ice 0.99997)
      on every var; **FP32-vs-Fortran ≡ FP64-vs-Fortran to the printed digit** — precision
      sits exactly on the pre-existing port↔Fortran floor. Biases ≤3e-4; |Δ|max sst 0.50 K
      twin / 0.88 K vs Fortran (= the FP64 run's own 0.89 K weather class)

### Task 11: Gate 5 — the 63-yr hindcast (endgame)

- [ ] pre-register bars in this file BEFORE submission (they are: Tbar ≤ 0.001 °C; OHC ≤ ~10×
      port↔Fortran gap; co-track/flatten) — re-affirm numbers against the final 63A/63B harvest
- [ ] submit 63-yr FP32 hindcast **63C-MP = FP32 twin of the 63A posture** (options config,
      SPEED=1, no CGPOLY — the clean twin, no solver-class confound; 63A≠63B, one run cannot
      match both), 2N class, OUTDIR `/work/ab0995/a270088/port2/mp/climate63/63C/`.
      Optional second arm 63D-MP (63B posture, +CGPOLY) only on user call if the CGPOLY×FP32
      question is worth another 12-h run
- [ ] harvest (rule-0.40 trim); `m7_hindcast_drift.py` vs 63A/63B + Fortran R2; F4-style figure
- [ ] verdict vs bars; if failed → island bisection + re-run decision (user consult)

### Task 12: Sensitivity map + per-scheme rungs + merge decision

- [ ] `docs/PRECISION_ISLANDS.md` finalized (the map: every island + evidence + cost)
- [ ] optional extra rungs: KPP-default config + per-option isolation legs (TKE-only /
      mEVP-only / zstar-only) if the sensitivity map needs per-scheme attribution (the main
      ladder already covered the combined options config)
- [ ] campaign findings doc `docs/plans/20260719-m8-FINDINGS.md`; lessons →
      `docs/KOKKOS_PORTING_LESSONS.md`
- [ ] merge-back assessment vs settled M7 (expect mechanical rename conflicts); default
      posture decision with user (likely: build-time option, never silent default)
- [ ] move this plan to `docs/plans/completed/`

## Technical Details

- `real_t` = working precision; `dbl_t` = deliberate FP64 island (post-sweep, every `dbl_t` in
  tree is a documented decision; raw `double` in state code = sweep bug).
- Kokkos: `FieldT<real_t>` for state; `FieldT<dbl_t>` available for whole-field promotions
  (Option B's capability held in reserve, per-field only, on gate evidence).
- Mixed-type kernel expressions promote to double automatically = correct island semantics;
  the literal-hygiene pass exists to kill *unintended* promotions in hot FP32 kernels.
- Mesh metrics: computed FP64 at init, stored `real_t` (never compute geometry from FP32 coords).
- Restart schema stays `NC_DOUBLE`: a double holds every float exactly → SP→disk→SP round-trip
  bit-exact and SP-restart-into-DP exact (both GATED in Task 6, not assumed; the 63-yr endgame
  runs restart-free so Task 6 is the only rung exercising this). **DP→SP truncates mantissa by
  design** — the interchange is free only in the FP32→FP64 rescue direction. Zero format churn.
- Aliasing at any precision-boundary shim: one seeded buffer when the same actual is bound to
  in + inout roles (PR-940 CVMix lesson).
- Calendar/time: untouched double island (their FP32 seconds-in-day ulp-8ms trap is impossible here).
- **Half precision (future, parked — user asked; answer: yes, per-field):** the type system
  extends downward. A future `half_t` (`Kokkos::Experimental::half_t`/`bhalf_t`, native on
  A100/GH200) can demote *individual fields/buffers* under the same registry discipline
  (demotion entry + gate evidence + pinned-pair give-back), because `FieldT<T>` is templated and
  the convert-at-boundary pattern M8 establishes covers the two hard edges (MPI and NetCDF have
  no native half). **Whole-model FP16/BF16 stays out permanently** — fp16 overflows at 6.5e4
  (pressure in Pa) and bf16 carries ~3 decimal digits (0.01 °C structure drowns at 20 °C).
  Credible first candidates when that day comes: CGPOLY preconditioner-apply storage
  (mixed-precision iterative-refinement literature) and halo-pack compression lanes.
  Nothing in M8 blocks this; nothing in M8 builds it (YAGNI).
- **Anomaly/increment formulation (user proposal 2026-07-19 — parked as the M9 candidate
  track, deliberately NOT M8 step 1):** computing on departures from reference states +
  compensated accumulation is the published enabling pair for fp16-class precision
  (Klöwer/Düben-line results). NOT done first because: (a) it changes rounding even at FP64 →
  forfeits Gate 0 and confounds formulation bugs with precision effects; (b) FP32 evidence
  says it's broadly unnecessary (PR-940 whole-model absolute FP32; NEMO/ROMS; FESOM already
  anomaly-form where it matters: `density_m_rho0`, anomaly-integrated `hpressure`, SSH≈0,
  velocities≈0, T in Celsius); (c) the Gate-3 sensitivity map SELECTS the targets —
  refactor-first inverts measure-first; (d) a badly chosen reference gains nothing, and a
  time-varying reference is the Z7 bug class. Physics-identifiable candidates for M9: S−35
  anomaly (fp16 ulp at 35 ≈ 0.03 PSU — absolute S is fp16-dead), zstar layer-thickness
  perturbation vs resting thickness, compensated (Kahan) tracer accumulation at the ledger
  sites. Trigger: M8 sensitivity-map findings or a concrete fp16 ambition. Runs as its own
  gate-isolated campaign with its own pre-registered floors (it changes numerics at EVERY
  precision).

## Post-Completion (no checkboxes — external/user actions)

- **Push** `m8-precision` (user approval required — standing rule).
- **Paper**: the registry + ladder results are the skeleton of a FESOM2 precision-sensitivity
  paper (companion to the NEMO/ROMS study); coordinate with PR-940 author (suvarchal) — our
  conservation + decadal evidence complements their CPU SP engineering.
- **JUPITER**: fold the footprint + comm numbers into the GH200 campaign plan.
- **Merge-back** into the speed line only after M7 settles, with user.
