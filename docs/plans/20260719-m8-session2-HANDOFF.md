# M8 session-2 handoff (2026-07-19) — GATE 3 PASSED IN FULL (zero promotions) → Gate 4 in flight

**Read first:** `docs/plans/20260719-m8-mixed-precision.md` (checkboxes current through Gate 3) ·
`docs/SP_PORTING_LESSONS.md` (now SP1–SP11; SP10/SP11 earned this session) ·
session-1 handoff `20260719-m8-session1-HANDOFF.md` (worktree layout, base state, recipes).

## Headline verdicts (all on the OPTIONS config = endgame physics, CORE2 c1 unless noted)

1. **SP options bring-up PASSED** — zstar+cvmix_TKE+mEVP+GM runs at FP32 out of the box
   (np1, 60 steps; cvmix_TKE at FP32 = first anywhere; top registry suspect cleared).
2. **Gate 3a PASSED (mature 30-d seed-ensemble envelope):** every field's sp/envelope ratio
   DECREASED as the envelope matured (T 14→4.8, S 66→6.4, pgf ~770→14-16, w 652→64); ice +
   mixing already at 1-2× envelope; no shape departure. Artifacts `mp/gate3/gate3a_mature.*`
   (+ immature-100-step `gate3a_verdict.csv`, `gate3a_envelope.png`).
3. **Gate 3b PASSED — THE PR-940-never-made measurement:** 30-d heat-drift gap FP32−FP64 =
   0.2 % of the physical signal; salt/vol gaps 1e-7/1e-9 class, SIGN-CHANGING (random walk,
   no leak); FP64 zstar closes volume to 8e-16. `mp/gate3/gate3b_conserv.{csv,png}`.
4. **Gate 3c PASSED:** 30-d CG mean iters 90.83 (FP64) vs 90.88 (SP), pairwise mean|Δit|
   0.33, max 5, no stagnation.
5. **Gate 3k PASSED (CPU):** the ENTIRE `FESOM_SPEED=1` 16-lever package is **bit-identical
   ON≡OFF at SP** (diff_snap, 11 snaps); CGPIPE selfcheck 0.000e+00 at SP; **CGPOLY d3 at SP:
   selfcheck 0.000e+00, iters 120.2→40.2 (FP64 twin 39.6)** — Chebyshev is NOT
   rounding-fragile; CGPOLY-vs-plainCG divergence at SP = solver-tolerance class (≤ the
   precision divergence itself). **EVPWIDE = N/A under mEVP** — the code refuses the lever at
   whichEVP=1 (banner), so it was inert in 63A ITSELF; effective 63A posture = SPEED=1
   (+unbind) on mEVP (SP10).
6. **Restart round-trip gate = MOOT:** the port has NO restart writer/reader (all "restart"
   strings are comments; argv has no restart input). Endgame runs restart-free. Building
   restart machinery = user scope decision, not a precision gap.
7. **ZERO island promotions this session.** The registry still holds exactly the two session-1
   islands (diag-Allreduce staging; JRA time axis).

## Binaries (ALWAYS `BIN=`-pin)

**mp64-2 / mp32-3 = current frozen set (conservation hook included), `mp/bin/`:**
serial 013d71cc… / 276a8960… · cuda c24619d6… / 9e07fc67… (`mp64-2_mp32-3.sha256`).
Gate-proven: FP64 off+armed bit-identical vs base; SP off+armed bit-identical vs mp32-2;
CUDA envelope PASS. CUDA builds need `env_cuda.sh` (plain env.sh has no nvcc — session-1 bins
mp64-1/mp32-2 remain valid but lack the hook).

## New instruments (all committed)

- `FESOM_MP_CONSERV=N` — env-gated FP64 global vol/heat/salt integrals every N steps
  (dbl_t Kokkos reduce over owned wet columns, MPI_DOUBLE Allreduce, device-current views —
  valid on CUDA where the host step-diag aliases are stale; collective, rank-0 print).
- `scripts/mp_envelope_verdict.py BASE SP SEED... --csv --plot` — sp-vs-max-over-seeds
  envelope ratios (STATIC coords excluded).
- `scripts/mp_conserv_drift.py DP_LOG SP_LOG --plot --csv` — drift + gap analysis.
- `jobs/job_mp_gate3` (CPU c1 options-config leg) · `jobs/job_mp_gate3_gpu` (1-GPU CORE2) ·
  `jobs/job_mp_gate4` (1-yr 63A-posture twin, output-protected, CONSERV=100 free series).

## IN FLIGHT at handoff (harvest these first)

- **g_sp / g_dp = 26364960/61** (gpu partition, PENDING at write time): SP+FP64 CUDA
  options-config legs, 100 steps, SPEED=1 + CGPIPE selfcheck + CONSERV. Harvest: rc=0,
  banner, `[cgpipe-selfcheck] … 0.000e+00` count, CONSERV lines vs CPU twins (expect
  last-digit-class agreement FP64; SP CONSERV vs CPU-SP differs at CUDA-atomics class).
- **Gate 4 = 26365064**, `--dependency=afterok:26364960` (auto-starts only if g_sp exits 0;
  scancel it if g_sp's selfcheck lines look wrong before it dispatches): 1-yr FP32 CUDA,
  EXACT 63A posture (verbatim KNOBS from 63A PROVENANCE incl. the inert EVPWIDE=8), 2N×4GPU,
  monthly streams only, OUTDIR `/work/ab0995/a270088/port2/mp/y1/63Cmp_1958`.
  **Harvest:** `m7_climate_check_plots.py <outdir> <plotdir> --years 1958` (main-checkout
  script, read-only) vs FP64 twin = **63A year 1958** (`/work/ab0995/a270088/port2/climate63/63A`)
  + Fortran `/work/ab0995/a270088/fesom2_core2`; M5.23 bar = pattern correlations in the
  1.00000/0.99996/1.00000/0.99997 class (sst/sss/ssh/a_ice); + CONSERV series plot.

## Then: Gate 5 (63-yr 63C-MP)

`jobs/job_mp_gate4` generalizes: NSTEPS=1088640, 12-h walltime, fresh OUTDIR
`/work/ab0995/a270088/port2/mp/climate63/63C/`. **Re-affirm the pre-registered bars with the
user first** (plan Task 11: Tbar ≤0.001 °C, OHC ≤~10× port↔Fortran gap, co-track/flatten —
against the FINAL 63A/63B harvest numbers: 41-yr common window |K−F| Tbar 0.0000 °C, OHC
0.1/0.2 ZJ of 19400). 63A posture, no CGPOLY (63A≠63B; optional 63D-MP CGPOLY arm = user call).

## Perf notes banked (indicative, single pairs — Gate-2 board remains the measurement)

Options config c1: 1440-step low-I/O pair 0.1822→0.1127 s/step = **1.62×** (100-step
snap-heavy pair 1.43× — snapshot I/O ~0.1 s/step at SNAP=10 masks compute). np1 login 1.16×.
dt-seed legs cost identical to base (0.182 s/step ×3).

## Traps earned this session

- **SP10 (dead knob in posture):** Serial-backend levers need `FESOM_SPEED_FORCE_SERIAL=1`;
  the effective posture is what the BANNER prints, not what KNOBS lists (EVPWIDE×mEVP).
- **SP11 (format-blind grep):** step-diag is `it=%3d` — grep `"it= *[0-9]*"` + strip, or the
  mean silently averages only ≥100-iter steps.
- One MPI launch per srun step (PMIx): chained `bash -c` runs inside one srun die at
  MPI_Init with "OPAL ERROR: Unreachable"; use separate srun calls.
- fesom_io "NetCDF error: Permission denied" at init = the OUT dir doesn't exist (the port
  does not mkdir; every runner must).
- The a-leg fleet dirs (`mp/gate3/fleet/*`, ~50 GB snapshots) back the Gate-3 verdicts —
  keep until the user signs off on Gate 3, then the snap dirs can go (CSVs/PNGs + logs stay).

## Standing user decisions (unchanged)

Islands extensible on gate evidence; anomaly/increment = M9; half precision parked (per-field
pathway); MP banned on main/m7 line; merge-back only after M7 settles; **ask before push**
(commits `816eded..` + this session all LOCAL).
