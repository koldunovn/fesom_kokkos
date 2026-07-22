# M8 session-4 handoff (2026-07-22) — 1964-storm ROOT-CAUSED + FIXED + PROVEN · Gate-5 RESUBMITTED · scaling campaign COMPLETE (knobs-off + Bp)

**Read first:** this file, then `docs/PRECISION_ISLANDS.md` (the new `ssh-stiff-ale-acc`
entry = the full evidence chain), session-3 handoff for background. Worktree
`/home/a/a270088/port_kokkos_mp`, branch `m8-precision`, ALL COMMITS LOCAL (ask before
push). **Frozen bins `mp/bin/`: mp64-6 / mp32-7 = THE FIXED SET** (serial+cuda, sha in
`mp64-6_mp32-7.sha256`); mp64-2/mp32-3 = the scaling-fleet bins (pre-fix, knobs-off
continuity). ALWAYS `BIN=`-pin.

## THE STORM HUNT — SOLVED (the session's arc, commits d8585c5..7723e7a)

1. **replay58 died step 118503 = 63C's 118504** (3rd trajectory: CPU 512r vs CUDA 8-GPU
   vs cold-start — deterministic, date-locked 1964-10-04/05, ~1.4 h CPU repro).
2. **Nanscan silence explained**: probe ladder was tracer-side only; momentum/ice chain
   unscanned. Probes extended (uv, uvAB, ice fields, stress, uv_rhs per substep, ssh_rhs
   pre-CG) + ARMED banner (L80).
3. **58b verdict**: first NaN in `vel-rhs(uv_rhs)` — but NaN-only scan.
4. **58c (non-finite scan + located reports)**: REAL first corruption = `pressure_bv(rho)`
   = **−inf at node 118958 (1-based), 179.35°E 57.63°N = ALEUTIAN BASIN (3800 m)** — NOT
   Maud Rise (s3 psl attribution was a date coincidence; actual low = 970.7 hPa ON the
   node at JRA rec 2224). Ice exonerated (cell ice-free); pre-storm state SP≡FP64.
5. **Smoking gun**: global max|eta| 1.85–1.97 m entire run → 48.3 m @118500. Node trace
   (58d, `FESOM_MP_TRACE_NODE`): clean EXPONENTIAL, deta doubling ≈20 steps from ≈Sep-22,
   node ssh_rhs −1.2e10 (~10⁶× normal) while local physics sane; eta DECOUPLED from
   thicknesses (hn0 stable); T→−7e7 FINITE at 118502 (runaway-shear advection), EOS −inf
   at 118503.
6. **ROOT CAUSE**: `update_stiff_mat_ale` (fesom_ssh.cpp) incremented the real_t CSR
   `values` open-loop 118k steps. **Float ABSORPTION dropped sub-ulp increments
   asymmetrically per row → SSH operator de-tuned → eigenmode crossed |λ|=1**; crossing
   follows the (SP-identical) seasonal state ⇒ the date-locked death everywhere. FP64
   immune (ulp 2e-16).
7. **FIX (`7723e7a`)**: dbl_t island `values_dbl_fld` — accumulation shadow seeded from
   the base matrix in `fesom_ssh_preconditioner`; increments atomic_add into the shadow;
   real_t working copy refreshed each update (rounded ONCE). CG SpMV stays float-bandwidth.
   Give-back: nnz dbl_t (~7 MB CORE2) + trivial refresh kernel, zstar-only.
8. **CERTS**: FP64-zstar CORE2 byte-IDENTICAL old-vs-new · SP-linfs pi byte-IDENTICAL ·
   **PROOF replay58e (26389907): COMPLETED 120k steps rc=0 ZERO nanscan hits** — rides
   through Oct-4-1964; node trace healthy (rhs O(1e3-1e4)) · CUDA zstar smoke + Serial
   twin: **CG iters exactly 125 both**, eta identical, T/S in envelope (CUDA stale-host
   diag zeros = known print artifact).
9. **⚠️ UPSTREAM (user to report, 2 items)**: Fortran shares the design
   (oce_ale.F90:1892-2001) — PR-940's SP branch dies on any months-long zstar run; plus
   the JRA time-axis (s1). JAX zstar equivalents worth checking too.

## GATE-5 RESUBMITTED (in flight)

**63C2 = 26391115** (`jobs/job_mp_gate5`, 63A posture) · **63D2 = 26391116**
(`jobs/job_mp_gate5_63D`, 63B posture + CGPOLY + E.T1). BIN=mp32-7-cuda, OUTDIRs
`mp/climate63/{63C2,63D2}` (old 63C/63D = the failed arms' partial output, KEPT as
forensic record — the Sep-1964 SP-vs-63A comparisons came from there). 12-h wall,
guillotine+trim harvest, bars unchanged (|Tbar|≤0.001 °C/common-yr, |OHC|≤2 ZJ,
co-track/flatten; `m7_hindcast_drift`-class + `mp_gate4_verdict.py`). Monitor armed
(15-min cadence). Then Task 12 (sensitivity map + findings + merge decision).

## SCALING CAMPAIGN — COMPLETE (knobs-off) + Bp COMPANION (~90 % harvested)

**Knobs-off (final, incl. session fixes):** dars CPU 1.52–1.60× c1..c32 · dars GPU
1.48/1.43/1.36/1.31/(g32 pair pending) · NG5 CPU 1.57/1.58/1.53/**1.61 (c32 = the curve
max; UCX init-fail root-caused → snap_every=-1 mandatory ≥4096 ranks — the step-0
snapshot gather dies in ibv_reg_mr; job SNAP param added)** · NG5 GPU 1.44/1.43/1.38/1.26
· CORE2 1.54/1.39/1.28 · farc 1.50/1.54/1.53/1.39.

**Class-Bp companion (user-commissioned: "SP with all speed optimisations, all meshes"):**
`jobs/job_mp_scale_gpu_bp` (Bp = SPEED=1+EVPWIDE=8+CGPOLY=3+unbind+proto pkg; L80
banner asserts; per-rep retry for the UCX proto-v2 init flake class — bitmap.h/
proto_select.c asserts + init hangs). **HEADLINE: SP and the speed stack OVERLAP — SP
speedup under Bp = 1.11–1.24× (dars), ~1.19× (NG5), 1.03–1.15× (CORE2), ~1.04× (farc
g16) vs 1.26–1.48× knobs-off; the stack itself = 2.0–2.2× at FP64. dp-Bp reproduces the
m7 CSV anchors sub-1 % everywhere.** farc×proto = hang-prone (m7-s16b concurs:
reproducible ≥128 ranks) → farc g32 pair = class B (NOPROTO=1), noted on the figure.
IDs `mp/gate3/bp_fleet.txt` (resubmitted legs annotated). Remaining at handoff: NG5
g16sp/g32 pair, dars g32 Bp + knobs-off pairs, retries (dars g4dp, core2 g8dp, farc
g2dp), farc g32 B pair.

**Figures** `mp/gate3/scaling_figs/mp_scaling{,_bp}.png+pdf` — 4-panel house layout
((a) s/step log-log · (b) SYPD linear CORE2+farc · (c) SYPD linear big meshes · (d)
speedup), shared below-panel legend (user), family colors consistent, **farc prod dt
1200 s (user: 20 min not 15)**, **CG dt-corr applied (m7-s16b measured: dars ×1.0222,
NG5 ×1.0110)** — mirrors m7 DT_CORR so the campaigns' figures agree. Harvest = rerun
`scripts/mp_scaling_fig.py` (parses both fleets, m7-Bp anchors read live from the m7 CSV).

## New instruments (all env-gated, FP64/production inert; certified inert-off by byte gates)

- `FESOM_MP_NANSCAN=1` — now NON-FINITE (NaN|Inf), located reports (elem/node, nz, comp,
  geo, owned/halo), ARMED banner, momentum/ice/ssh probes.
- `FESOM_MP_TRACE_NODE=<1-based global id>` + `FESOM_MP_TRACE_FROM=<step>` — end-of-step
  single-node series (eta/deta/rhs/hn0/T0/S0/rho0/Kv0/w0) from the owner rank.

## Traps earned this session

- **A NaN-only scan is blind to the inf stage of an SP overflow** — scan !isfinite.
- **"finite" ≠ "sane"**: T=−7e7 passed the scan; the EOS turned it into −inf a phase later.
- **Cumulative float accumulators are the SP killer class**: open-loop `float += tiny`
  over 100k steps = absorption → operator drift. Grep audit candidate: any `+=` into
  long-lived real_t state without a feedback loop (the registry's accumulation ledger).
- myList_* global IDs are **1-based** (dist files).
- UCX proto-v2 pkg: intermittent init asserts/hangs (retry in job scripts); farc ≥128
  ranks = reproducible hang (class B fallback; m7-s16b concurs).
- gpu-devel = 1-GPU shape only. CORE2 c1 login init ≈ 3-4 min (2-min Bash timeout trap).
- CUDA step-diag zeros for device-resident fields = stale-host print artifact, NOT a
  model failure (uv-guard blindness class, s3).

## In-flight at handoff

| job | what | monitor |
|---|---|---|
| 26391115/16 | GATE-5 63C2/63D2 (12 h) | b24gg11xb |
| 26386402/03 | dars g32 knobs-off pair | bymybp5l9 |
| bp stragglers | see bp_fleet.txt | bhzrikylx (note: resubmitted ids NOT in its watch set — check sacct at harvest) |

Commits this session (local): `d8585c5` probes · `ef4ec44` SNAP root-cause · `7308b86`/
`396c811`/`66d0cb0`/`3de1a9d`/`44ab2a7` figures · `8244275`/`bca4259` Bp fleet ·
`d929642`/`4d9d4de` verdicts · `7723e7a` THE FIX · this handoff.
