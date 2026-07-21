# M8 session-3 handoff (2026-07-21) — Gates 0-4 PASSED · Gate-5 arms died at the 1964 storm (ACTIVE HUNT) · SP scaling campaign mid-harvest

**Read first:** this file, then `docs/plans/20260719-m8-mixed-precision.md` (plan, checkboxes
current), `docs/SP_PORTING_LESSONS.md` (SP1–SP11), session-1/2 handoffs (worktree layout,
gate recipes). Worktree `/home/a/a270088/port_kokkos_mp`, branch `m8-precision`, ALL COMMITS
LOCAL (ask before push). Frozen bins `mp/bin/`: **mp64-2 / mp32-3** (serial 013d71cc/276a8960,
cuda c24619d6/9e07fc67; CUDA builds need `env_cuda.sh`). ALWAYS `BIN=`-pin.

## Program state

- **Gates 0–4 ALL PASSED, ZERO island promotions.** Gate 4 (1-yr FP32 CUDA, 63A posture):
  twin correlations sst/sss/ssh 1.00000, a_ice 0.99999; FP32-vs-Fortran ≡ FP64-vs-Fortran
  to the printed digit (`scripts/mp_gate4_verdict.py`). Registry still holds exactly two
  islands (diag-Allreduce staging; JRA time axis).
- **Gate 5: BOTH 63-yr arms FAILED — the ACTIVE BUG HUNT (below).** 63C (63A posture,
  26373534) and 63D (63B posture + CGPOLY + E.T1 env, 26373583) both died at the SAME model
  date on divergent trajectories.
- **SP scaling campaign (user-commissioned): mid-harvest**, 4 meshes × CPU/GPU (below).

## 🔴 THE 1964-STORM HUNT (top priority; a replay is IN FLIGHT)

**Evidence chain (all committed, `e8ad691`):**
1. 63C died step 118504 (63D same ±window) = **1964-10-04/05** — identical model DATE on
   trajectories that had diverged for 6.8 yr under different solvers ⇒ external date-locked
   trigger, not chaos and not solver.
2. Death shape: CG iter-1 rhs already NaN (`s_old=nan`); the step before spiked 131 CG iters
   (baseline 76–88). NaN born upstream of the solve, ONE step from healthy to dead.
   The uv-guard was BLIND (CUDA+SPEED=1 reads stale host copies frozen at 0) — no velocity
   telemetry from the dying runs.
3. JRA55-do 1964 scan: **psl's YEAR MINIMUM (922 hPa) at record 2224 = Oct-5 00:00, centred
   62°S / 9.6°E = Maud Rise, Weddell Sea** — the deepest storm of the year sitting on the
   late-winter ice pack (the Weddell-polynya deep-convection region). Winds there ~30-35 m/s
   are NOT the year's max (45-47 m/s day 252 was survived) — it's this storm specifically.
4. FP64 (63A/63B) sailed through the same date/files ⇒ SP-only vulnerability.
5. **Discriminator ANSWERED: cold-start-1964 SP probe (d64_sp, c1 CPU, nanscan armed)
   SURVIVED the storm** (14400/14400 steps, zero nanscan hits, sane physics; FP64 control
   clean too). ⇒ storm alone insufficient — **needs the multi-year SPUN-UP state**
   (multi-year ice / preconditioned pycnocline at Maud Rise are the state suspects).
6. **IN FLIGHT: `mp_replay58` = 26379079** — cold-1958 SP replay on 4 nodes (512 ranks,
   dist_512 private CORE2), NSTEPS=120000 (death ~118504 + margin), `FESOM_MP_NANSCAN=1`,
   CONSERV=1000, PRINTEVERY=100, out `mp/gate3/fleet/replay58_sp/`. ~1–1.5 h compute.
   **Decision tree:** dies at the storm ⇒ nanscan names the producing phase → descend into
   that kernel (SP4 guard class first: denormal-flush epsilons, divisor-floor overflows),
   fix per-precision or promote a dbl_t island (registry entry + failing signature +
   pinned-pair give-back), re-gate (FP64 pi byte gate = 2 min), resubmit BOTH 63-yr arms.
   Survives ⇒ mechanism needs the exact CUDA-trajectory weather → next rung = device-side
   NaN instrumentation (nanscan device twin) + CUDA replay; also worth then testing a
   CPU replay at 8 ranks (the 63C rank count) to rule partition-dependence in.
   NOTE: on ANY source change, rebuild ALL FOUR builds and re-freeze (mp64-3/mp32-4).

## SP scaling campaign (user-commissioned this session)

**Banked so far (same-day pinned pairs, 300 steps min-of-2, knobs-off, per-mesh dt
CORE2 1800 / farc 900 / dars 120 / NG5 180 (c32n dt60); L80 banner asserted per leg):**

| leg | FP64 | FP32 | speedup | m7 anchor |
|---|---|---|---|---|
| dars c1..c32 CPU | 6.005/3.035/1.578/0.842/0.413/0.202 | 3.829/1.942/0.992/0.528/0.264/0.133 | **1.57/1.56/1.59/1.60/1.56/1.52** | Δ≤1% all |
| dars g2 GPU | 0.7675 | 0.5176 | **1.48** | Gate-2 1.47 reproduces |
| NG5 c4/c8/c16 | 4.657/2.350/1.208 | 2.941/1.489/0.789 | **1.58/1.58/1.53** | Δ≤1.7% |
| CORE2 c1/c2/c4 | 0.1991/0.1067/0.0593 | 0.1292/0.0767/0.0462 | **1.54/1.39/1.28** | — |
| farc c1/c2 | 0.9614/0.4882 | 0.6398/0.3173 | **1.50/1.54** | — |

**The emerging law:** SP speedup ≈1.55–1.60× at high per-rank workload, decaying as per-rank
load shrinks (CORE2 c4 = 248 verts/rank → 1.28×; dars c32 dips to 1.52×) — consistent with
the m7-s14 mechanism: SP halves BYTES but not the per-partner TOLL, so the gain saturates
where latency/toll dominates. FP64 continuity vs m7 anchors 0.02–1.7% at every size.

**Still in flight at handoff (harvest → `scripts/mp_scaling_fig.py`, auto-monitor armed):**
farc c1 pair (running), farc c4/c8 pairs (pending, walltimes bumped to 22 min — USER: farc
wants ~20+ min comfortable), dars g4sp/g8 pair, NG5 c32n pair (dt60), NG5 GPU g4..g32 pairs
(user: "they can just stay hanging"), dars c1dp done late. IDs in
`mp/gate3/scaling_fleet.txt`. Fig output `mp/gate3/scaling_figs/mp_scaling.{png,pdf}`.

**🔴 FIGURE CONVENTIONS (user corrected me TWICE — memory `feedback-figure-conventions`):**
copy `m7_scaling_figs.py` helpers: node_axis (log-base-2, PLAIN count labels 1 2 4 8 16 32,
no minors), decimal_log_yaxis (%g labels, no powers of 10), **SYPD = dt_prod/(365·s_step)
at PRODUCTION dt = CORE2 1800 / farc 900 / dars 240 / NG5 240** + the measurement-dt footnote.

## In-flight job table (2026-07-21 ~18:15)

| job | id | what | harvest |
|---|---|---|---|
| mp_replay58 | 26379079 | THE HUNT: cold-1958 SP 4N replay, nanscan | monitor armed; grep `mp-nanscan`/FATAL + last CONSERV |
| mp_sc_* CPU | various | farc/NG5/dars legs | `mp_scaling_fig.py` (auto-monitor rebuilds fig on drain) |
| mp_sc_g*/ng* GPU | 26378507-513, 26378953-960 | dars g4/g8, NG5 GPU curve | same |

Monitors from this session may be ORPHANED if the CLI restarts (it happened once — the
notification names the task ids; the SLURM jobs are unaffected). On resume: `squeue -u
a270088 | grep mp_`, then `sacct -j <id>` for anything missing, harvest per above.

## Other open items

- **Gate-5 resubmission** blocked on the storm fix. Bars stay as affirmed (|Tbar gap| ≤
  0.001 °C/common-yr vs the twin; |OHC gap| ≤ 2 ZJ; co-track/flatten). Jobs
  `jobs/job_mp_gate5` (63C twin) + `jobs/job_mp_gate5_63D` (63B twin) ready to resubmit
  with the fixed BIN.
- **Restart machinery**: port has NONE (all "restart" strings are comments) — Task-6 gate
  closed as moot; building restart I/O = user scope decision. The storm hunt made the cost
  concrete (no way to checkpoint near the death).
- **uv-guard CUDA blindness** (found in the hunt): the blowup guard reads stale host uv
  under SPEED=1 — consider a device-side guard (cheap parallel_reduce at diag cadence) as
  an M8 follow-on; would have caught the 63C death 100 steps earlier with telemetry.
- Task 12 (sensitivity map + findings doc + merge decision) after Gate 5 lands.
- Fleet snapshots (~50 GB) under `mp/gate3/fleet/*` back the Gate-3 verdicts — keep until
  user signs off; CSVs/PNGs/logs are the durable artifacts.

## Traps for the next session (this session's additions)

- SP10 (posture dead-knobs: banner=truth; FORCE_SERIAL on Serial; EVPWIDE inert under mEVP
  — also inert in 63A itself). SP11 (printf-padding-safe greps: `it= *[0-9]*`).
- One MPI launch per srun step (PMIx); fesom_io needs the OUT dir premade.
- Walltime rules: sub-30-min GPU jobs backfill where 45-min waits hours (fairshare 0);
  farc legs want 20+ min; diagnosis runs go WIDE (4N), not c1 (user: wall-time-to-answer
  is the only metric for triage).
- `sacct -X -o State` after every fleet: COMPLETED ≠ success — grep rc + FATAL in logs (SP3).

## Standing user decisions (cumulative)

All four prizes in scope; islands extensible on registry evidence; anomaly/increment = M9;
half precision parked (per-field path); MP banned on main/m7 line; merge-back after M7
settles; ask before push; easybsreturn/visc-5 etc. = m7-line matters, not M8's.
