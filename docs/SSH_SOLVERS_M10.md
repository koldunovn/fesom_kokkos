# M10 — SSH solver track: ledger of record

**Status:** ACTIVE — T1–T8b COMPLETE (all four solvers implemented, gated, solution-classed;
first A/B numbers harvested). T8c/T9–T13 open.
**Plan of record:** `docs/plans/20260805-m10-ssh-solvers.md` (in-tree on this branch)
**Branch:** `m10-ssh-solvers`, worktree `~/port_kokkos_ssh`
**Derivations doc:** `docs/plans/20260805-m10-ssh-derivations.md` (Layer-0 math + typo report T-1…T-9)
**Work area:** `/work/ab0995/a270088/port2/m10/{bin,labdumps,gates,ab,figs}`

Every number in this file carries its SLURM job id (or "login") and the binary md5/sha of
record. Lab numbers are NEVER performance numbers of record (R4/T3 rule). Pre-registrations
are written BEFORE the runs they gate.

## Provenance

- **Base commit:** `f42c453` — *deviation from the plan text (which says `65a1a71`): `f42c453`
  IS `65a1a71` + the plan commit itself (docs/plans/20260805-m10-ssh-solvers.md +
  `.gitignore` `ssh_sergey/`). Branching from it puts the plan of record in-tree, which the
  plan's own Progress Tracking rule requires ("update this plan in the same commit as the
  work"). Same lineage, no source-code delta vs `65a1a71` (`src/` untouched by `f42c453`).*
- **Source material:** `ssh_sergey/` (gitignored, copied from the main checkout 2026-08-05):
  `solvers.F90`, `1612.01395v3.pdf` (Chronopoulos-Gear/pipelined CG), `gmd-9-4209-2016.pdf`
  (P-CSI), `manasi-pcg-jpdc2022.pdf` (PIPECG-OATI).
- **Kokkos submodule:** checked out in-worktree @ `15dc143e` (4.4.01, same as main).
- **`/work` quota at start:** group ab0995 usage 2.533 PB, no hard group cap enforced
  (`lfs quota -g ab0995`); global `/work` free ≈ 14 TB — lab dumps are 2-D-matrix-sized
  (MBs–GBs), fine. Re-check before the NG5 np≥256 dumps land (T3).

### m10-base binaries (provenance only — NOT byte-comparable to the main checkout's:
### `__FILE__`/debug paths differ; behavioural identity is what the base gate proves)

| build | sha256 | md5 | built |
|---|---|---|---|
| `build-m7serial/fesom_port` | `f228d664eb0de144c6f6914212669372b676c9c2c9d9a8672291b601e5784c41` | `54433326fe312057bd0e9fd980de298f` | 2026-08-05 |
| `build-m7cuda/fesom_port` | `e5245fa3290b2247cf6bd115260de61fd6bcb267b413cda7f240b03db7580d2c` | — | 2026-08-05 |

*(Main-checkout `build-m7serial` md5 at the same date: `9743f602aaf8d27dcfbe9baae9b3c977` —
differs from the worktree's, so a gate log showing `54433326…` provably ran THIS tree's
binary: the R9 discriminator.)*

## Gate registry

Every gate/A-B row: date · gate · job id · binary md5 (from the log — R9) · result.

| date | gate | job | binary md5 | result |
|---|---|---|---|---|
| 2026-08-06 | T1 base: knob-free serial 20-step CORE2 byte gate (worktree ROOT, HEAD f42c453) | 26722627 | `54433326…` (log also shows main-checkout `9743f602…` ≠ — R9 armed) | **PASS** diff_snap rc=0 |
| 2026-08-06 | T2 knob-off byte gate on the [ssh-wire]+VERIFY instrumentation commit | 26722771 | `8f2be32b…` ≠ main `9743f602…` (R9) | **PASS** diff_snap rc=0 |
| 2026-08-06 | T3 (R7) knob-off byte gate on the CMake OBJECT split + dump/trace/lab code | 26723005 | (in log) ≠ main (R9) | **PASS** diff_snap rc=0 |
| 2026-08-06 | T5a knob-off byte gate on the dispatch + guard + sympre + cg2 commit | 26723543 | (in log) | **PASS** diff_snap rc=0 |
| 2026-08-06 | T5a explicit `FESOM_SSH_SOLVER=cg` byte gate (proves the default branch is the certified path) | 26723544 | (in log) | **PASS** diff_snap rc=0 |
| 2026-08-06 | T5b CUDA fidelity, `cg2` (CUDA vs Serial, same knobs) | 26723550 | (in log) | **PASS** (identical iters; details below) |
| 2026-08-06 | T5b CUDA fidelity control, `cg` | 26723551 | (in log) | **PASS** (the reference floors) |
| 2026-08-06 | T5b options ×3 (TKE/mEVP/zstar) under `cg2` | 26723560 | (in log) | harvested (see below) |
| 2026-08-06 | T6 knob-off byte gate (pipecg commit) | 26723615 | (in log) | **PASS** |
| 2026-08-06 | T6 options ×3 under `pipecg` | 26723616 | (in log) | harvested — zstar all-pass |
| 2026-08-06 | T8 knob-off byte gate (pcsi commit) | 26723691 | (in log) | **PASS** |
| 2026-08-06 | T8b options ×3 under `pcsi` | 26723757 | (in log) | harvested — zstar all-pass, 0 fallbacks |
| 2026-08-06 | T7 knob-off byte gate (oati commit) | 26723854 | (in log) | **PASS** |
| 2026-08-06 | T7 options ×3 under `oati` | 26723855 | (in log) | harvested — zstar+mEVP all-pass |
| 2026-08-06 | T7 CUDA fidelity, `oati` | 26723856 | (in log) | CUDA ≡ Serial wire counts, 0 fallbacks |
| 2026-08-06 | Dead-knob guard commit (CGPOLY×cg2/pipecg dies) — knob-off byte | 26724330 | (in log) | **PASS** |

### T5b `cg2` — solution-class gate vs baseline `cg` (serial CORE2 np8, dt1800, 20 steps, login)

**Every field inside the pre-registered P-L2 bound. PASS.**

| field | measured max\|Δ\| | P-L2 bound | margin |
|---|--:|--:|--:|
| `eta_n` | 5.238e-05 | 1.5e-04 | 2.9× |
| `u` / `v` | 5.134e-03 / 4.922e-03 | 5e-02 | ~10× |
| `T` | 5.372e-02 | 2e-01 | 3.7× |
| `S` | 2.465e-03 | 2e-02 | 8.1× |
| `Kv` / `Av` | 9.999e-02 / 9.990e-02 | 2e-01 | 2.0× (the known floor) |
| `w` | 2.050e-06 | 5e-05 | 24× |
| `bvfreq` | 2.449e-06 | 2e-05 | 8.2× |
| `pgf_x` / `pgf_y` | 1.073e-08 / 1.032e-08 | 1e-07 | ~9× |
| `density_m_rho0` | 1.931e-03 | 2e-02 | 10× |
| `h_ice` / `h_snow` | 6.546e-03 / 3.358e-04 | 2e-02 | 3.1× / 60× |
| `a_ice` / `m_ice` / `m_snow` | 2.212e-04 / 4.439e-04 / 1.177e-04 | 5e-03 | ≥11× |
| `uice` / `vice` | 6.678e-04 / 6.894e-04 | 5e-03 | 7.3× |

**Wire observable (proves the knob fired, L80):** allreduces/solve **266.70 → 125.35** and
exchanges/solve **266.70 → 125.35** at 123.35 iters — i.e. exactly `2 + k` of each, against
baseline's `2 + 2k`. **⭐ Ring-composition self-check:** `FESOM_SSH_RING=0` (literal
2-exchange form) gives *identical* iteration counts and *identical* verify residuals
(6.964e-11 both) with 249.70 exchanges — so the ring composition is numerically equivalent
to the literal form and only the message count differs.

**⚠️ Iteration parity: pre-registration MISSED (recorded per house style).** P-L2 asked for
cg2 within ±3 iterations of cg; measured **123.35 vs 132.35 = −9.0 (−6.8 %)**. The
pre-registration rested on "same Krylov space, rounding-level drift", which the T4 finding
invalidated: with `SYMPRE=1` cg2 runs on `M̃⁻¹ ≠ M⁻¹`, so it is a *different* preconditioner
and a different Krylov space by construction — fewer iterations is a legitimate outcome, not
a defect. **Revised criterion (registered now, for T6/T7):** iteration parity is asserted
between variants *at equal `SYMPRE`*, and the cg-vs-variant count is reported, not gated.
The "same Krylov space to rounding" claim is instead tested where it actually holds — the
lab α-sequence at `SYMPRE=0`, iteration 1: cg 2.944659607044791 vs cg2 2.944659607044791,
**relative difference 0.000e+00** (bitwise), before the σ drift takes over from iteration 2.

**Lab certification (np1 CORE2 step-20 dump):** baseline `cg` replay — iters 126 = 126 and
x_final **BITWISE** on all 126858 owned nodes, `CERT PASS`.

**CUDA fidelity (job 26723550, CORE2 4 ranks/4 GPUs, CUDA vs Serial at the same knobs):**
CUDA and Serial agree on the iteration count to the last solve (123.30 both), 0 fallbacks,
0 `true>rtol`. Worst fields: `T` 2.334e-03, `Kv` 8.199e-04, `S` 5.737e-04, `uice` 4.393e-04,
`Av` 2.733e-04, `eta_n` 6.856e-06. **⭐ Better than the baseline control** (job 26723551,
`cg`): `Kv` 9.835e-02, `Av` 9.898e-02, `u` 6.530e-03 — i.e. under `cg2` the CUDA/Serial pair
is ~120× closer on the Kv/Av fields. Reading: those fields' ~1e-1 "floor" is a *threshold
switch* (the near-freezing/mixing branch), not a continuous error — under `cg2` both
backends happened to stay on the same side of it. Recorded as an observation, **not** as a
claim that `cg2` improves GPU fidelity; a switch that did not flip is not a smaller error.

### T6 `pipecg` — solution-class gate vs baseline `cg` (serial CORE2 np8, 20 steps, login)

**Every field inside the P-L2 bounds. PASS.** `eta_n` 5.236e-05 · `u`/`v` 5.134e-03/4.922e-03
· `T` 5.372e-02 · `S` 2.465e-03 · `Kv`/`Av` 9.999e-02/9.990e-02 · `w` 2.050e-06 · `bvfreq`
2.449e-06 · `pgf_x/y` ~1.05e-08 · `density` 1.931e-03 · `h_ice`/`h_snow` 5.099e-03/2.652e-04
· `a_ice`/`m_ice`/`m_snow` 2.314e-04/6.219e-04/1.799e-04 · `uice`/`vice` 3.762e-04/6.732e-04.

**Wire observable (the designed signature, and it is unambiguous):** blocking allreduces per
solve **266.70 → 1.00** (only the initial ‖b‖²), nonblocking **0 → 124.35** — exactly one
`MPI_Iallreduce` per iteration and no blocking reduction inside the loop at all. Exchanges
127.35 = 3 + k (ring form) vs 251.70 = 3 + 2k (literal); iteration counts and verify
residuals identical between the two forms, so the ring composition is again verified as
numerically equivalent.

**⚠️ Attainable-accuracy watch (the known pipelined failure mode — MEASURED):** max
\|true − recurrence\| residual over 20 solves is **1.991e-08 for `pipecg`**, against
**6.964e-11 for `cg2`** and **3.030e-11 for `cg`** — a **~300× loss of attainable accuracy**,
exactly the price the literature attributes to propagating the residual through extra
auxiliary recurrences. It is still far inside tolerance here (0 `true>rtol` events in 20
solves at rtol ≈ 4), but it is the quantity to watch on long runs and at tighter `soltol`,
and it is why `FESOM_SSH_VERIFY=1` is armed in every gate. Iterations: 124.35 (`cg2`
123.35, `cg` 132.35).

**T2 instrumentation smoke (login pi np2, dt100, 20 steps, STATS+VERIFY on):** counts EXACT
vs hand count (3-iter solve: exch=8=2+2k, ar_blk=8=2+2k, body-launches=17=6+4k−1(break));
verify true≡rec to 7 digits, **max |true−rec| gap over 20 solves = 1.05e-11** — first data
point for the (deferred) verify gate threshold.

## Pre-registrations

### P-L2 — Layer-2 per-field solver-class bounds (registered 2026-08-06, BEFORE any variant gate)

**Calibration of record (login, free):** the 20-step CORE2 np8 serial gate config was run
off-vs-`FESOM_SPEED=1+CGPOLY=3` (FORCE_SERIAL) — CGPOLY is the CERTIFIED solver-class
precedent, so its 20-step per-field envelope IS the class envelope on exactly the gate
config. Measured step-20 maxima (step-10 similar): `eta_n 4.9e-5 · u 1.4e-2 · v 6.4e-3
(1.4e-2 @10) · T 9.1e-2 · S 3.1e-3 (5.7e-3 @10) · Kv 9.999e-2 · Av 9.990e-2 · w 5.5e-6 ·
bvfreq 2.5e-6 · pgf_x/y 5.7e-9 · density 1.8e-3 (4.8e-3 @10) · h_ice 4.1e-3 (5.4e-3 @10) ·
h_snow 2.1e-3 · a_ice 2.1e-4 · m_ice 6.4e-4 · m_snow 1.2e-4 · uice/vice 9.1e-4`.

**Bounds for every M10 variant gate** (serial np8 CORE2 dt1800 20-step, variant-vs-`cg`
same binary; ~×3 headroom on the measured class, capped near the threshold-switch
amplitude for switch-class fields):

| field(s) | bound (max\|Δ\|) | class rationale |
|---|---|---|
| `eta_n` | **1.5e-4** | the solve's own output; class 5e-5 |
| `u`, `v` | 5e-2 | momentum response, class 1.4e-2 |
| `T` | 2e-1 | isolated near-freezing/threshold cells (class 9e-2; switch-amplitude-capped) |
| `S` | 2e-2 | class 6e-3 |
| `Kv`, `Av` | 2e-1 | mixing-scheme threshold switch, known ~1e-1 floor (L79 family) |
| `w` | 5e-5 | class 7e-6 |
| `bvfreq` | 2e-5 | class 3e-6 |
| `pgf_x`, `pgf_y` | 1e-7 | class 6e-9 |
| `density_m_rho0` | 2e-2 | class 5e-3 |
| `h_ice`, `h_snow` | 2e-2 | class 5e-3 |
| `a_ice`, `m_ice`, `m_snow` | 5e-3 | class 6e-4 |
| `uice`, `vice` | 5e-3 | class 9e-4 |

**Also gated, every variant:** (i) true residual ‖b−Ax‖ ≤ rtol on EVERY solve
(`FESOM_SSH_VERIFY=1` armed in all gates; byte-transparent by construction). (ii) iteration
parity: cg2/pipecg/oati mean iters/solve within ±3 of `cg`'s on the same 20 steps (same
Krylov space, rounding drift only); `pcsi` iters REPORTED, not gated (higher by design).
(iii) 0 fallback firings. **Options ×3 shape:** TKE + zstar under the same bounds — zstar's
Kv/Av standing controls (9.537e-2 / 9.869e-2) are EXPECTED to move in value while holding
~1e-1 magnitude (the CGPOLY zstar precedent); **mEVP: a formal FAIL via near-freezing T
flips at O(tens) of ice-edge cells (~7e-2) is the KNOWN pattern (CGPOLY 26313392)** — if it
appears, the mandatory exoneration probe is pure-Serial off-vs-variant reproducing the same
cells (26313804 shape); the options matrix stays STRICT (user 2026-07-17), the row stands
as formal-FAIL-with-exoneration for user adjudication.

**CUDA fidelity gates (variant CUDA vs variant serial):** the known fidelity floors apply
unchanged — worst-field ~1e-1 = the Kv/Av class, ocean scalars O(1e-4), ice O(1e-3) (D22
scatter class). A variant must not WORSEN these floors.

*(Verify-gap gate threshold: still DEFERRED until the dars/NG5 census distribution is in —
pi np2 20-solve max gap 1.05e-11 is the first point; CORE2 np8 gates will add more.)*

## Baseline census (T2 — the sizing-of-record table)

All rows `FESOM_SSH_STATS=1`; binary `aaa9762` (census jobs print md5 `44afd0c6…`, R9).
"legacy" = all knobs off (2 exch + 2 AR per iter). "speed1" = `FESOM_SPEED=1` (CGPIPE:
1 fused 2-ring exch per iter; AR count UNCHANGED — the axis M10 attacks).

| config | leg | iters/solve | exch/solve | ar_blk/solve | body-launches/solve | s/step | CG busy+wait ms/step | job |
|---|---|--:|--:|--:|--:|--:|--:|---|
| CORE2 np8 serial dt1800, 20 st (cold) | legacy | 132.35 | 266.70 | 266.70 | 534.4 | — | — | login |
| CORE2 np8 serial (FORCE_SERIAL) | speed1 | 132.35 | 134.35 | 266.70 | 667.8 | — | — | login |
| dars 4N g16 dt120, 100 st | legacy | 40.17 | 82.34 | 82.34 | 165.7 | 0.4427 | 11.0+8.3 | 26722817 |
| dars 4N g16 dt120, 100 st | speed1 | 40.18 | 42.18 | 82.36 | 206.9 | 0.2375 | 6.1+4.4 | 26722817 |
| NG5 4N g16 dt180, 100 st | legacy | 83.69 | 169.38 | 169.38 | 339.8 | 1.2318 | 28.8+16.8 | 26722818 |
| NG5 4N g16 dt180, 100 st | speed1 | 83.70 | 85.70 | 169.40 | 424.5 | 0.6462 | 19.0+15.2 | 26722818 |
| NG5 16N g64 dt180, 100 st | legacy | 83.68 | 169.36 | 169.36 | 339.7 | 0.4289 | 22.1+25.7 | 26722819 |
| NG5 16N g64 dt180, 100 st | speed1 | 83.68 | 85.68 | 169.36 | 424.4 | 0.2440 | 10.4+13.0 | 26722819 |

**Census reading (sizing, NOT numbers of record):** at NG5 16N production config the CG
phase is 23.4 ms/step of 244 (9.6 %) and its WAIT (13.0) exceeds its BUSY (10.4) — the
solve is majority-communication at scale, carrying ~169 blocking allreduces + ~86 fused
exchanges per step. That wait pool (~13 ms/step) is the ceiling M10's allreduce levers play
for at 16N; iters/solve 83.7 (NG5 dt180, 100-step window; the ledger's "settled ~72" is the
longer-window value). Census s/step runs HOT vs the certified board (0.2440 vs 0.1995 @16N):
un-pinned bins + STATS/PHASESTATS overhead (per-exchange prof barriers) — expected, and
exactly why perf-of-record only ever comes from pinned 300-step A/B pairs.

**Launch pricing (A100, in-binary probe, rank 0):** async **3.0 µs/launch**, fully-fenced
**8.9 µs/launch** (n=10000/1000; both dars legs agree to 0.1 µs). A blocking allreduce
forces the fenced shape; the +4-AXPY-class launch cost of pipecg/oati is therefore
~12–36 µs/iter async-priced vs each saved/hidden AR's fenced-drain + latency.

**Count reconciliation vs the E-ledger (L100 class):** measured legacy exch = 2+2k and
speed1 exch = 2+k exactly (CORE2: 266.7 = 2+2·132.35, 134.35 = 2+132.35; dars: 82.34/42.18
at k=40.17) — the "146/74 events/step @ ~72 iters" ledger anatomy is the same formula at
NG5's settled k. Iteration counts are IDENTICAL legacy-vs-speed1 to the last solve
(CORE2 132.35/132.35, dars 40.17/40.18 with a single +1-iter solve) — the CGPIPE
byte-identity claim visible at count level on CUDA (allreduce trajectory untouched).

**⚠️ Ledger-divergence note (honest):** the M7 ledger lists dars dt120 iters/solve = 50.1;
this census measures 40.2 (100-step mean incl. cold ramp, settled 37). Different
measurement epoch/state — M10's numbers of record are THIS census (re-measured, same-day,
md5-pinned). The 128.8 CORE2 class reproduces (132.35 cold-20 vs 128.8 settled ✓).

**R2 probe verdict (jobs 26722815 compute / 26722816 GPU nodes):** **NO async progression
on any stack** — `MPI_Iallreduce` + host busy-work + `Wait`: wait time stays = the full AR
latency at every busy factor (overlap ≈ 0, measured as negative because the nonblocking
path itself is DEARER than blocking): openmpi/4.1.2 blocking 0.98 µs vs Iallreduce-path
~8.8 µs total (+8 µs surcharge!); 4.1.5-nvhpc compute 3.17 µs vs ~4.8 (+1.6); 4.1.5 GPU
nodes 3.22 µs vs ~5.0 (+1.8). `MPI_THREAD_SINGLE` confirmed (provided=0). **Pre-registered
attribution rule (binding for T6):** a null-or-negative pipecg-vs-cg2 delta is attributed
to STACK (no progression + Iallreduce surcharge), not algorithm; pipecg proceeds anyway
(user decision: all four) as the honest-negative leg. This also stands as a candidate
explanation for Sergey's null overlap result on the same machine.

## Findings ledger

**F1 (R1) — `pr_values` non-symmetry MEASURED (2026-08-06, lab `--sym-check` on the CORE2
np8 dumps, 99.3 % pair coverage):** `max|M_ij−M_ji| = 8.561e-8` against `max|M_ij| =
1.342e-7` ⇒ **symmetry-defect ratio 0.64** — the MITgcm-style preconditioner is
non-symmetric at ORDER UNITY relative to its own off-diagonal scale (structural cause:
`pr_ij/pr_ji = d_j/d_i`, so the defect is large wherever neighbouring diagonals differ —
area/depth contrasts). Control: the A matrix defect ratio is 1.42e-13 (symmetric to
rounding) — the instrument is sound. Consequences: (a) P-CSI/Lanczos MUST run on a
symmetrised operator (T4 decision; note `M = D⁻¹C` with `C = I − ½·a_ij/(d_i+d_j)`
symmetric ⇒ `D^{1/2} M D^{-1/2}` is symmetric — the cgpoly `isq` machinery is the native
tool); (b) the quantified candidate cause for Sergey's CG² instability (report to user).
Identical at step 1 and step 20 (linfs: matrix static, as designed).

**F1b ⭐⭐ (R1, CONFIRMED 2026-08-06) — the non-symmetry BREAKS the recurrence that every
CG-CG-family solver depends on.** Derived in `docs/plans/20260805-m10-ssh-derivations.md`
§0.4 and measured with `fesom_ssh_lab --sigma-drift` on the CORE2 np1 step-20 dump: the
`σ_i = δ_i − β_i²σ_{i-1}` recurrence — which `cg2`, `pipecg` AND `oati` all use in place of
an explicit `(p,Ap)` — is valid only for symmetric `M⁻¹`. As built it drifts to **21.8 %**
by iteration 60 (the orthogonality term that must vanish reaches **24.1 %** of γ); with the
symmetrised `M̃⁻¹ = D^{−1/2}CD^{−1/2}` it is exact to **1.2e-13**. **α is therefore wrong by
up to 22 % on the production matrix** — a confirmed candidate explanation for Sergey's CG²
instability, and a scope change: the preconditioner-symmetry decision the plan scoped to
P-CSI now governs all four solvers (knob `FESOM_SSH_SYMPRE`, default 1 for non-`cg`).
Baseline `cg` is unaffected — it computes `(p,Ap)` explicitly and recurs nothing.

**F1c ⭐⭐⭐ (T5b, 2026-08-06) — THE INSTABILITY REPRODUCED IN-MODEL AND CURED.** With `cg2`
running on the literal MITgcm preconditioner (`FESOM_SSH_SYMPRE=0` — Sergey's exact
configuration), a 20-step CORE2 np8 serial run fires the armed fallback on **18 of 20
solves**: the residual *grows* to 2.9e+04 against an rtol of ~4 after ~30 iterations. The
same binary with `FESOM_SSH_SYMPRE=1` (the symmetrised `M̃⁻¹`): **0 fallbacks**, 123.35
iters/solve, true residual ≤ rtol on every solve. The lab α-sequence pins the mechanism
quantitatively — at `SYMPRE=0`, `cg2`'s α departs from reference PCG's by **2.017 % at
iteration 2**, matching the independently measured σ-recurrence drift of **1.977 %** at the
same iteration (§F1b) term-for-term, then grows to 6.5 % by iteration 8.

> **This is Sergey's CG² instability, reproduced on demand and removed by a one-line change
> to the preconditioner** — and it was found by re-deriving the algorithm rather than
> transcribing it (the user's Layer-0 directive). Report to Sergey with §0.4/§0.4b of the
> derivations doc.

Corollary — the armed fallback earned its keep on its first outing: it caught all 18
divergences, restored `X0`, redid each solve with baseline `cg`, and the run finished with
0 `true>rtol` events. A silent-divergence failure mode became a logged, recovered event.

**F2 (R2) — no async Iallreduce progression** (see §Baseline census: probe verdict +
binding pipecg attribution rule).

**Lab rules of record:** lab numbers are NEVER performance numbers of record; a lab-tuned
constant becomes a default only after an in-model 20-step gate reproduces the lab
iters/solve within ±10 % (R4). CUDA-trajectory dumps (the zstar pair) are matrix-analysis
material, not bitwise-cert material — cert dumps are serial.

**Lab certification (T3):** CORE2 np8 step-20 replay: **PASS — iters 125=125 AND x_final
BITWISE (126858/126858 nodes)**, i.e. the lab reproduces the in-model solve exactly even
at np8 (login, serial build `aaa9762`+T3). Roundtrip FNV-64 checksums verified on every
array at every load (loader hard-fails on mismatch). np1 bitwise-cert + at-scale dumps:
below.

### T5b options ×3 under `cg2` (job 26723560, serial CORE2 np8, variant vs baseline `cg`)

Two **formal FAILs**, both carrying the threshold-switch signature the P-L2 pre-registration
anticipated, and both requiring the pre-registered exoneration probe before any verdict.
The options matrix stays STRICT (user 2026-07-17) — these rows stand as
formal-FAIL-with-mechanism for user adjudication, exactly as the CGPOLY mEVP row does.

| option | field | step 10 | step 20 | bound | verdict |
|---|---|--:|--:|--:|---|
| **TKE** | `h_ice` | 4.473e-01 | 5.069e-02 | 2e-02 | ❌ formal FAIL |
| **TKE** | `a_ice` | 7.707e-03 | 5.246e-03 | 5e-03 | ❌ formal FAIL (marginal) |
| TKE | `eta_n` · `T` · `Kv` · `uice` | 9.52e-05 · 1.23e-02 · 9.99e-02 · 2.71e-03 | 5.36e-05 · 5.11e-02 · 9.99e-02 · 2.09e-03 | 1.5e-4 · 2e-1 · 2e-1 · 5e-3 | ✅ |
| **mEVP** | `T` | 4.182e-01 | 6.603e-02 | 2e-01 | ❌ formal FAIL |
| mEVP | `a_ice` · `h_ice` · `eta_n` · `uice` | 6.66e-05 · 2.30e-04 · 8.55e-05 · 4.39e-04 | 6.38e-04 · 5.39e-04 · 5.34e-05 · — | — | ✅ |

**⭐ The mEVP fingerprint is numerically identical to the documented CGPOLY one.** The M7
ledger records for CGPOLY-vs-baseline mEVP: "`T` 6.602e-02 … NON-accumulating (max shrinks
**0.42 → 0.066** step 10 → 20)". `cg2` gives **0.4182 → 0.06603**. Same field, same
non-accumulating decay, same magnitudes to two figures — i.e. this is the mEVP scheme's own
per-scheme floor (L79 family) being excited by *any* solver-class change, not something
`cg2` does. Everything here is pure Serial, so the "no CUDA anywhere" half of the CGPOLY
exoneration (probe 26313804) is satisfied by construction; what remains for a full
exoneration is the cell-count/geography characterisation.

**TKE was a NEW pattern — probe run, and it is EXONERATED.** Cell-count/geography probe
(pure Serial both legs, so no CUDA anywhere by construction):

| step | field | cells with \|Δ\| > 1e-3 | of | fraction | where |
|---|---|--:|--:|--:|---|
| 10 | `h_ice` | **3** | 126858 | 0.0024 % | 80.0 °N, −71.8 °S, 80.2 °N |
| 10 | `a_ice` | **1** | 126858 | 0.0008 % | 80.0 °N |
| 20 | `h_ice` | 5 | 126858 | 0.0039 % | 80.0 °N, −71.4 °S, … |
| 20 | `a_ice` | 3 | 126858 | 0.0024 % | 79.8–80.4 °N |

The entire 4.473e-01 `h_ice` excursion is **one cell**, at 80.0 °N, and it is a
**derived-ratio artifact, not a physics difference**: there `a_ice` = 0.0157 (baseline) vs
0.0080 (cg2) — a nearly ice-free cell — and `h_ice = m_ice/a_ice` divides by it. The
CONSERVED quantity barely moves: `m_ice ≈ h_ice·a_ice` = 0.4565·0.0157 = **7.17e-03** vs
0.9038·0.0080 = **7.23e-03**, i.e. **0.8 % on the conserved mass while the derived thickness
differs by 98 %**. `m_ice` itself passes its bound comfortably (6.2e-04 vs 5e-03) in the
default-options gate.

> **Methodological lesson for the bounds (recorded, and it changes P-L2):** bounding a
> *derived ratio* like `h_ice = m_ice/a_ice` at ice-marginal cells measures the denominator,
> not the solver. **Revised rule for the remaining variant gates: `h_ice`/`h_snow` excursions
> are adjudicated on the conserved `m_ice`/`m_snow` at the same cell**; a `h_ice` FAIL whose
> `m_ice` passes is exonerated and recorded as such, with the cell count and the two masses
> quoted. This also retro-explains why `h_snow` (60× margin) never tripped: snow is not
> concentrated at those marginal cells.

### T6 options ×3 under `pipecg` (job 26723616) — and the first use of the revised rule

| option | verdict | evidence (step 20) |
|---|---|---|
| **zstar** | ✅ **PASS, every field** | `eta_n` 8.199e-05 · `T` 3.777e-03 · `S` 9.953e-04 · `h_ice` 2.199e-03 · `m_ice` 1.018e-03 · `a_ice` 4.272e-04 · `u` 3.725e-03 · `Kv`/`Av` 9.999e-02/9.990e-02 |
| **mEVP** | ✅ within bound at step 20 | `T` **6.603e-02** — again the exact CGPOLY figure; all other fields pass |
| **TKE** | ⚠️ `h_ice` 1.445e-01 > 2e-02 → **EXONERATED by the revised rule** | the conserved `m_ice` at the same option is **2.834e-03**, comfortably inside its 5e-03 bound; `S` 1.051e-02 ✓, `a_ice` 1.318e-03 ✓ |

**zstar passing on every field is the load-bearing result here**: zstar is the time-varying
matrix case, so it is what stresses the frozen `pr_values` (and, once `pcsi` is tuned, the
frozen eigenbounds). The standing zstar controls behave exactly as pre-registered for a
solver-class change — `Kv` 9.999e-02 / `Av` 9.990e-02, i.e. moved off the byte-exact
9.537e-02/9.869e-02 in value while holding the ~1e-1 magnitude.

The TKE row is the revised P-L2 rule's first live application, and it worked as intended: the
derived thickness trips, the conserved mass does not, and the row is exonerated on the spot
instead of needing a separate probe.

### T8a/T8b `pcsi` — first results (serial CORE2 np8, 20 steps, login)

**It works, and the trade is much better than the plan assumed.** Lanczos m=30 on `M̃⁻¹A`
returned θ = [3.4455e-03, 1.4440] → with the safe margins (deflate ν 10 %, inflate µ 5 %)
[ν,µ] = [3.1010e-03, 1.5162], **κ = 489**; rank-agreement assertion passed; sympre defect
0.700 → 2.45e-13.

| metric | `cg` | `pcsi` (K=5) | change |
|---|--:|--:|---|
| iters/solve | 132.35 | **138.00** | **+4.3 % only** |
| blocking allreduces/solve | 266.70 | **29.60** | **−88.9 % (9.0×)** |
| exchanges/solve | 266.70 | 140.00 | −47.5 % |
| fallbacks | 0 | 0 | — |

`ar_blk` = 29.60 = `iters/K + 1` exactly, i.e. the ONLY reductions are the every-5th-iteration
convergence checks plus the initial ‖b‖². The plan's expectation was "`pcsi` higher by
design (Chebyshev is CG without the optimality)" — **wrong-high on the iteration penalty**:
+4.3 % is far cheaper than the 9× reduction saving, before any tuning of K or the margins.

**Solution class: 18 of 19 fields inside the P-L2 bounds; `S` 2.749e-02 vs a 2e-02 bound
(1.4×) = formal FAIL.** `S`, `T`, `Kv` and `density` all peak at the **same node
(1564237)** — a single water column. Mechanism note: at equal residual tolerance a Chebyshev
iterate is NOT the A-norm-optimal one CG returns, so a slightly larger solution error at the
same `soltol` is expected behaviour, not a defect — but it is a formal FAIL under the strict
matrix and the remedy (tighter `soltol` for `pcsi`, or margin/K tuning) belongs to T8c.

### T8b options ×3 under `pcsi` (job 26723757) — the frozen-eigenbound question, answered

| option | verdict | evidence (step 20) |
|---|---|---|
| **zstar** | ✅ **PASS, every field** | `eta_n` 4.009e-05 · `T` 3.777e-03 · `S` 2.583e-03 · `h_ice` 9.140e-04 · `m_ice` 4.782e-04 · `a_ice` 1.819e-04 · `u` 3.725e-03 · `Kv` 9.999e-02 |
| **mEVP** | ✅ **PASS, every field** | `T` **5.240e-02** (inside the 2e-01 bound) · `u` 1.893e-02 · all ice fields ≤ 5.4e-04 |
| **TKE** | ⚠️ `h_ice` 1.473e-01 → **EXONERATED** by the revised rule | conserved `m_ice` **2.837e-03** vs a 5e-03 bound ✓; `S` 1.051e-02 ✓; `a_ice` 1.319e-03 ✓ |

**Zero fallback firings across all three options** — and that is the load-bearing sentence
for R3. `pcsi` estimates `[ν,µ]` ONCE, at the first solve, and then never refreshes them;
under zstar the matrix values change every step. Twenty zstar steps with the frozen bounds
produced no divergence, no stall, and every field inside its bound, i.e. **the frozen-bound
bet holds at this horizon** and `FESOM_PCSI_REEIG` remains correctly unbuilt (YAGNI). The
long-horizon check (the two CORE2 dumps 180 days apart, job 26723048) is still T8c's job.

Note the `S` excursion that formally failed at DEFAULT options (2.749e-02) is **not
systematic**: under TKE it is 1.051e-02 and under zstar 2.583e-03, both inside the bound.
That is consistent with the single-water-column reading rather than a scheme-wide error.

### T7 `oati` — implemented as the SHALLOW variant, and it beats the deep one here

Derivations §3.2b: `[T]`'s deep `n→g→h→e→f` chain exists only to overlap one `Iallreduce`
with two iterations of operator work. **R2 measured that this stack has no async
progression**, so that overlap buys nothing — while the chain would cost 4 chained operator
applications ⇒ 4-ring shipping of BOTH A and M (machinery that does not exist: cgpipe is
2-ring and ships M only; cgpoly's R-ring ships A only) ⇒ roughly double the halo bytes on the
axis the census says dominates. The shallow form keeps OATI's actual prize — half the syncs —
at `cg2`'s operator and halo cost. Measured, CORE2 np8, 20 steps:

| | `cg` | `cg2` | `pipecg` | **`oati`** | `pcsi` |
|---|--:|--:|--:|--:|--:|
| iters/solve | 132.35 | 123.35 | 124.35 | **124.00** | 138.00 |
| **blocking allreduces/solve** | 266.70 | 125.35 | 1.00 | **64.00** | **29.60** |
| (nonblocking) | 0 | 0 | 124.35 | 0 | 0 |
| exchanges/solve | 266.70 | 125.35 | 127.35 | **128.00** | 140.00 |
| verify gap (max \|true−rec\|) | 3.03e-11 | 6.96e-11 | 1.99e-08 | 6.93e-09 | 0 by construction |
| fallbacks | 0 | 0 | 0 | 0 | 0 |

`ar_blk` = 64.00 = `iters/2 + 2` — exactly one reduction per two iterations, at the SAME
exchange count as `cg2`. **Correctness:** the α sequence matches `cg2`'s on the same real
CORE2 dump — **it1 bitwise (0.000e+00)**, ~5e-15 at it2–3, ~5e-13 by it6 — i.e. the γ/δ
recurrences reproduce the Krylov space to rounding, which is the Layer-0 acceptance criterion.
Solution class vs `cg`: **all fields inside P-L2** (`eta_n` 5.143e-05 · `u`/`v`
8.932e-03/7.809e-03 · `T` 5.374e-02 · `S` 2.350e-03 · `Kv`/`Av` 9.999e-02/9.990e-02 ·
`h_ice` 6.784e-03 · `m_ice` 4.847e-04 · `uice`/`vice` 5.175e-04/8.671e-04).

### Options ×3 — consolidated across all four solvers (step 20, vs baseline `cg`)

| solver | job | zstar | mEVP | TKE | fallbacks |
|---|---|---|---|---|--:|
| `cg2` | 26723560 | *(not in this job)* | ❌ `T` 0.4182→0.06603 = the CGPOLY fingerprint | ❌ `h_ice` 0.447 → **exonerated** (1 cell) | 0 |
| `pipecg` | 26723616 | ✅ **all fields** | ✅ `T` 6.603e-02 | ⚠️ `h_ice` 1.445e-01 → exonerated (`m_ice` 2.834e-03 ✓) | 0 |
| `oati` | 26723855 | ✅ **all fields** | ✅ `T` 2.422e-02 | ⚠️ `h_ice` 1.476e-01 → exonerated (`m_ice` 2.854e-03 ✓); `a_ice` 5.244e-03 vs 5e-03 = **1.05×, marginal** | 0 |
| `pcsi` | 26723757 | ✅ **all fields** | ✅ `T` 5.240e-02 | ⚠️ `h_ice` 1.473e-01 → exonerated (`m_ice` 2.837e-03 ✓) | 0 |

**The pattern is consistent across all four solvers and is now well characterised**: zstar and
mEVP hold every bound; TKE trips only `h_ice`, always at the same ice-marginal cells, always
with the conserved `m_ice` comfortably inside its bound. **Zero fallback firings anywhere in
the options matrix** — 12 option-runs across four solvers. `oati`'s `a_ice` at 1.05× the bound
is the one genuinely marginal number in the set and is flagged rather than waved through.

### ⭐ R3 ANSWERED — the zstar eigenbound drift check (lab, login, free)

The plan made `FESOM_PCSI_REEIG` conditional: "built ONLY if this check shows the bounds
move". Run on the two CORE2 zstar dumps **180 simulated days apart** (job 26723048, steps 100
and 8740 at dt1800), replaying the real matrices through the production Lanczos estimator:

| dump | θ (Ritz) | [ν,µ] after margins | κ |
|---|---|---|--:|
| step 100 | [3.436277e-03, 1.434050] | [3.092650e-03, 1.505753] | 486.9 |
| step 8740 | [3.433973e-03, 1.433030] | [3.090576e-03, 1.504681] | 486.9 |
| **drift** | **θmin 0.067 % · θmax 0.071 %** | | **identical to 4 s.f.** |

**The preconditioned spectrum moves by less than 0.1 % across 180 simulated days**, against
safety margins of 10 % (deflate ν) and 5 % (inflate µ) — **the margins are ~100× larger than
the drift**. Sergey's premise ("matrix changes are tiny; the eigenvalues are set by domain and
resolution") is confirmed quantitatively on the production matrix.

**⇒ `FESOM_PCSI_REEIG` stays unbuilt, now on evidence rather than YAGNI**, and the frozen-bound
design is vindicated. Combined with the zstar options gate (0 fallbacks, all fields inside
bounds), R3 is closed.

**➕ Discovered while doing it — the symmetrisation is APPROXIMATE under zstar.** The
`[ssh-sympre]` observable reports a post-symmetrisation defect ratio of **6.7e-03 (step 100)
and 7.4e-03 (step 8740)** under zstar, versus **2.5e-13 under linfs**. Cause: `M̃⁻¹` scales the
FROZEN `pr_values` by `sqrt(d_i/d_j)` taken from the CURRENT `diag(A)`, and under zstar `A`
drifts away from the matrix `pr_values` was built from — so the two are mutually stale by the
accumulated drift. Three observations that matter:
- it is still **~100× better than the 0.70 as-built defect**, and 7e-3 ≪ 1, so the σ-recurrence
  argument (§0.4) holds comfortably — consistent with every zstar options gate passing with 0
  fallbacks across all four solvers;
- it **does not grow** between step 100 and step 8740 (6.7e-3 → 7.4e-3), i.e. almost all of it
  is accumulated before step 100, matching the <0.1 % spectral drift above;
- it is nonetheless the one quantity that would degrade on a multi-decade zstar run, so it
  belongs on the T12 climate-leg watch list.

*(Also noted: `pcsi` takes ~16–20 % more iterations than `cg` on these zstar dumps — 130 vs
112 and 105 vs 87 — against +4.3 % on linfs CORE2. Chebyshev's iteration penalty is
mesh/state dependent; T8c should quote it per configuration rather than as one number.)*

## A/B perf legs (submitted 2026-08-06 — HARVEST THESE FIRST next session)

Protocol: `jobs/job_m10_ab` — all legs back-to-back in ONE allocation on the SAME nodes with
the SAME binary, min-of-2 reps, 300 steps, `-C a100_80`, NG5 dt180. Harvest reports s/step,
Δ%, iters/solve and µs/iteration.

| job | rung | legs | purpose |
|---|---|---|---|
| 26723631 | NG5 4N | legacy-cg · cg2 · pipecg | ✅ HARVESTED (below) |
| 26723632 | NG5 16N | legacy-cg · cg2 · pipecg | ✅ HARVESTED (below) |
| 26723692 | NG5 4N | legacy-cg · pcsi · pcsi K=10 | ✅ HARVESTED (below) |
| 26723693 | NG5 16N | legacy-cg · pcsi · pcsi K=10 | ✅ HARVESTED (below) |
| ~~26723783~~ | NG5 4N | *(intended)* SPEED=1 · +cg2 · +pipecg · +pcsi | ❌ **VOID — see the L80 note below** |
| ~~26723784/26723867~~ | NG5 16N | *(intended)* same | ❌ cancelled, same defect |
| **26724474** | **NG5 16N** | **SPEED=1 · +cg2 · +oati · +pcsi** | ✅ **HARVESTED — the numbers of record (below)** |
| **26724475** | **NG5 4N** | **SPEED=1 · +cg2 · +oati · +pcsi** | ✅ **HARVESTED (below)** |

> ### ⚠️ L80 in the wild: a silent A/B truncation, caught by its own output
>
> Job 26723783 reported a clean-looking result — two legs, `+0.11 %` apart. **Both legs were
> the same configuration.** `sbatch --export` uses COMMAS as its own separator, so passing
> `LEGS="FESOM_SPEED=1;FESOM_SPEED=1,FESOM_SSH_SOLVER=cg2;…"` let sbatch eat everything from
> the first comma onward: the job received `LEGS="FESOM_SPEED=1;FESOM_SPEED=1"` and dutifully
> measured the baseline against itself. The earlier jobs (26723631/32/92/93) were unaffected
> only because their legs happened to contain no commas.
>
> This is the dead-knob trap wearing a different hat — a measurement that *looks* like a
> null result but never ran the thing being measured. It was caught because the harvest
> prints the leg name, and two legs printed the same one. **Fixes:** the intra-leg separator
> is now `+`, and every leg now echoes its resolved `FESOM_*` knob set into the log, so two
> legs sharing a knob set is visible at a glance rather than inferable from suspiciously
> equal numbers.

> ⚠️ **Measurement-design correction, made before any number was quoted.** The first four
> jobs clear every knob, so their leg 0 is **legacy `cg`** (2 exchanges + 2 allreduces per
> iteration) — *not* the production configuration. Since the M10 solvers use the CGPIPE ring
> machinery internally, comparing them against legacy `cg` would credit them with the ring
> saving that `FESOM_SPEED=1` already ships, and overstate M10's contribution. Jobs 26723783
> and 26723784 hold `FESOM_SPEED=1` on **every** leg so only the SSH solver differs — those
> are the numbers of record; the first four stand as the plain-PCG reference the plan asks
> for. (Interim leg-0 sanity: NG5 16N legacy `cg` = 0.4228/0.4241 s/step at 76.9 iters/solve.)

### ⭐ FIRST A/B HARVEST (26723631 / 26723632) — vs the plain-PCG reference

NG5, dt180, 300 steps, min-of-2, same allocation, same binary, `-C a100_80`.

| rung | leg | s/step | Δ | iters/solve |
|---|---|--:|--:|--:|
| **NG5 4N** | legacy `cg` | 1.2231 | — | 76.86 |
| | `cg2` | 1.2112 | **−0.97 %** | 76.13 |
| | `pipecg` | 1.2119 | **−0.92 %** | 77.10 |
| **NG5 16N** | legacy `cg` | 0.4228 | — | 76.88 |
| | `cg2` | 0.3985 | **−5.75 %** | 76.13 |
| | `pipecg` | 0.3981 | **−5.84 %** | 77.11 |

**Two results, and the second one is the important one.**

**(1) The gain scales with node count exactly as an allreduce-latency lever should** —
−0.97 % at 4N, **−5.75 % at 16N**. That is the signature the whole track was predicated on:
at 4N the reduction latency is small next to per-rank compute, at 16N it is not (the census
put the CG phase at 10.4 ms busy + 13.0 ms wait per step at that rung). Iteration counts are
flat across legs (76.1–77.1), so this is a *cheaper iteration*, not a different count — the
distinction the crossover claim rests on.

**(2) ⭐ `pipecg` ≡ `cg2` to within noise (−5.84 % vs −5.75 %) — the R2 attribution rule
fires, exactly as pre-registered.** The overlap buys nothing. This was written down BEFORE
the run ("a null-or-negative pipecg-vs-cg2 delta is STACK, not algorithm"), on the strength
of the T2 probe that measured zero `Iallreduce` progression plus a surcharge on both
production MPI stacks. A pipelined solver that pipelines perfectly and gains nothing is the
cleanest possible confirmation that the limitation is the MPI stack, not the method — and it
is the same result Sergey saw on the same machine. **This is paper material as an honest
negative**, and it carries a concrete portability claim: on a stack with real async
progression (or with `MPI_THREAD_MULTIPLE` + a progress thread, out of scope here), `pipecg`
should separate from `cg2`; here it cannot.

### ⭐⭐ pcsi A/B (26723692 / 26723693) — the best solver at scale so far

| rung | leg | s/step | Δ | iters/solve | µs/iteration |
|---|---|--:|--:|--:|--:|
| **NG5 4N** | legacy `cg` | 1.2226 | — | 76.87 | 15904.8 |
| | `pcsi` (K=5) | 1.2102 | **−1.01 %** | 82.07 | 14745.9 |
| | `pcsi` (K=10) | 1.2097 | −1.06 % | 82.07 | 14739.9 |
| **NG5 16N** | legacy `cg` | 0.4234 | — | 76.86 | 5508.7 |
| | `pcsi` (K=5) | 0.3963 | **−6.40 %** | 82.07 | **4828.8** |
| | `pcsi` (K=10) | 0.3963 | −6.40 % | 82.07 | 4828.8 |

**`pcsi` wins at 16N: −6.40 %, ahead of `cg2` (−5.75 %) and `pipecg` (−5.84 %) — and it wins
while doing MORE work.** It takes 82.07 iterations against `cg`'s 76.86 (+6.8 %), so the
per-iteration cost falls by **−12.3 %** (5508.7 → 4828.8 µs/iteration) — a bigger drop than
any other solver, and exactly what a method that removes ~89 % of the blocking reductions
should show. This is the crossover claim's core evidence in its cleanest form: **more
iterations, less time**, because the iterations are cheaper on the axis that dominates at
scale. The Chebyshev wager pays.

**⭐ Tuning result for T8c, already settled: `FESOM_PCSI_CHECK` does not matter.** K=5 and
K=10 are indistinguishable at both rungs (0.3963 vs 0.3963 at 16N; 1.2102 vs 1.2097 at 4N,
inside noise) and give the same iteration count (82.07). At K=5 the residual check is already
off the critical path, so there is nothing to buy by checking less often — **keep the safer
K=5 default**, and T8c's check-interval sweep can be dropped in favour of the margin and
Lanczos-m axes.

⚠️ All six legs above run against LEGACY `cg` (all knobs cleared), while the M10 solvers use
the CGPIPE ring machinery internally — so part of each Δ is the ring saving that
`FESOM_SPEED=1` already ships in production. Jobs 26723783/84/867 isolate the solver's own
contribution and are the numbers of record.

## ⭐⭐⭐ THE NUMBERS OF RECORD (26724474 / 26724475)

`FESOM_SPEED=1` held on **every** leg, so only the SSH solver differs. NG5, dt180, 300 steps,
min-of-2, same allocation, same binary, `-C a100_80`. Each leg's resolved knob set is echoed
in the log (the L80 guard) and all four differ, as required.

| rung | leg | s/step | **Δ vs production** | iters/solve | µs/iteration |
|---|---|--:|--:|--:|--:|
| **NG5 16N** | `FESOM_SPEED=1` (production) | 0.2407 | — | 76.88 | 3130.9 |
| | `+ cg2` | 0.2377 | **−1.25 %** | 76.10 | 3123.5 |
| | `+ oati` | 0.2379 | **−1.16 %** | 76.58 | 3106.6 |
| | **`+ pcsi`** | **0.2353** | **−2.24 %** | 78.33 | **3004.0** |
| **NG5 4N** | `FESOM_SPEED=1` (production) | 0.6403 | — | 76.87 | 8329.6 |
| | `+ cg2` | 0.6375 | −0.44 % | 76.14 | 8372.7 |
| | `+ oati` | 0.6391 | −0.19 % | 76.59 | 8344.4 |
| | **`+ pcsi`** | **0.6346** | **−0.89 %** | 78.37 | 8097.5 |

**`pcsi` is the winner at both rungs, and the gain is scale-dependent: −2.24 % at 16N,
−0.89 % at 4N.** Ranking is stable: pcsi > cg2 ≳ oati.

### ⚠️ The honest headline: M10's own contribution is ~3× smaller than the legacy comparison implied

Against **plain PCG** the same solvers measured −5.75 % … −6.40 % at 16N. Against the
**production** configuration they measure **−1.16 % … −2.24 %**. The difference is the CGPIPE
ring saving that `FESOM_SPEED=1` already ships — the M10 solvers use that machinery
internally, so a legacy baseline credits them with it. **The −2.24 % is M10's contribution;
the −6.40 % is not.** This is exactly why the leg definitions were corrected before any
number was quoted, and it is the single most important caveat in this ledger.

### Diminishing returns on the sync axis — measured, and it reframes the track

Blocking allreduces per solve on this configuration fall 266.7 (plain) → ~125 (cg2) → 64
(oati) → ~30 (pcsi). The time does **not** follow that ordering proportionally:

| step on the axis | AR/solve | Δ vs production |
|---|--:|--:|
| production `cg` (CGPIPE, 2 AR/iter) | 266.7 | — |
| `cg2` — halve the reductions | 125.4 | −1.25 % |
| `oati` — halve them **again** | 64.0 | −1.16 % (**no further gain**) |
| `pcsi` — remove ~89 % of them | 29.6 | −2.24 % |

**`oati` buys nothing beyond `cg2`** despite halving the sync count a second time — a clean,
pre-registered honest negative, and a useful one: it says the allreduce axis is *already
mostly harvested* by the first 2→1 halving, so further sync reduction alone does not pay.
That `pcsi` nonetheless gains more than either suggests its advantage is **not** the reduction
count alone. Leading hypothesis (consistent with T2's launch pricing — fenced 8.9 µs vs async
3.0 µs per launch — but **not yet isolated**): a blocking allreduce forces a device fence and
drain, and `pcsi` fences only every `K`=5 iterations while also running the simplest
per-iteration kernel set (3 AXPY-class ops, no dot products at all between checks). `oati`
saves fences but pays them back in extra vector work. **Testing that properly needs a
fence-count/launch-count attribution run — recorded as an open item for T10/T11, not claimed
here.**

## ⭐⭐⭐ THE SSH-SHARE CAMPAIGN — where the solvers actually pay

**Why this campaign exists.** The first A/B round measured NG5 only, and only whole-step time.
Both are too narrow: NG5 is the LARGEST mesh, where the SSH solve is ~10 % of the step, so any
solver win is diluted ~10×; and a whole-step delta cannot say whether the *method* works. This
round measures **CORE2 / farc / dars / NG5** on **both CPU and GPU**, and reports the SSH-phase
delta and the SSH share alongside the whole-step delta. `FESOM_SPEED=1` on every leg.

`d_SSH` = the solver's own speed-up. `SSH%` = share of the step available to win.
`d_total ≈ d_SSH × SSH%/100`.

### GPU (4 ranks/node, `-C a100_80`)

| mesh · rung | nodes/rank | SSH% | `cg2` d_SSH | `oati` d_SSH | `pcsi` d_SSH | best d_total |
|---|--:|--:|--:|--:|--:|--:|
| **CORE2 g4n** (16 r) | 7929 | 27.4 | −15.76 | **−18.33** | −16.94 | −5.18 (oati) |
| **CORE2 g16n** (64 r) | 1982 | **30.5** | −14.49 | **−20.49** | −19.38 | **−6.73 (oati)** |
| **farc g4n** (16 r) | 39899 | 36.5 | −8.94 | **−12.39** | ⚠️ **+33.30** | −5.12 (oati) |
| **farc g8n** (32 r) | 19950 | 37.6 | −12.21 | **−14.74** | ⚠️ **+29.91** | −6.81 (oati) |
| **dars g8n** (32 r) | 98761 | **5.7** | ⚠️ +4.59 | ⚠️ +11.96 | ⚠️ +21.47 | ⚠️ none — all lose |
| NG5 g16n (64 r) | 115670 | ~10 | −1.25 (total) | −1.16 (total) | −2.24 (total) | −2.24 (pcsi) |

### CPU (128 ranks/node) — CORE2 driven to its MAXIMUM decomposition

CORE2 has no 1024-rank partition; **864 is the largest that exists**, so this ladder is the
full available range for this mesh.

| rung | nodes/rank | SSH% | `cg2` d_SSH | `oati` d_SSH | `pcsi` d_SSH | d_total (pcsi) |
|---|--:|--:|--:|--:|--:|--:|
| CORE2 128 r | 991 | 2.5 | −8.28 | −8.42 | **−16.60** | −0.72 |
| CORE2 256 r | 496 | 4.4 | −23.61 | −25.79 | **−41.92** | −1.71 |
| CORE2 432 r | 293 | 9.4 | −29.15 | −43.49 | **−53.87** | −5.74 |
| CORE2 512 r | 248 | 9.9 | −34.12 | −47.38 | **−60.25** | −6.30 |
| **CORE2 864 r** | **146** | **18.9** | −33.01 | −49.89 | **−58.62** | **−13.10** |

### ⭐⭐⭐ The scaling result: ~1.7 rungs of parallel efficiency recovered

The SSH share climbs monotonically as the mesh thins — 2.5 → 4.4 → 9.4 → 9.9 → **18.9 %** —
i.e. by 864 ranks nearly a fifth of the step is the SSH solve. That is the scaling wall
forming, and it is exactly the wall these solvers attack. Speed-up and parallel efficiency
are relative to the 128-rank rung of each leg:

| ranks | nodes/rank | SSH% | baseline s/step | ×  | eff % | `pcsi` s/step | ×  | eff % | **eff gain** |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 991 | 2.5 | 0.1955 | 1.00 | 100.0 | 0.1941 | 1.00 | 100.0 | — |
| 256 | 495 | 4.4 | 0.1050 | 1.86 | 93.1 | 0.1032 | 1.88 | 94.0 | +0.9 pp |
| 432 | 293 | 9.4 | 0.0662 | 2.95 | 87.5 | 0.0624 | 3.11 | 92.2 | +4.7 pp |
| 512 | 247 | 9.9 | 0.0587 | 3.33 | 83.3 | 0.0550 | 3.53 | 88.2 | +5.0 pp |
| **864** | **146** | **18.9** | 0.0435 | 4.49 | **66.6** | **0.0378** | **5.13** | **76.1** | **+9.5 pp** |

**ALL FOUR legs ran at every rung** — parallel efficiency, each leg against its own 128-rank
point (an earlier draft of this section showed only `baseline` vs `pcsi`, which oversold
`pcsi` and hid how close `oati` runs):

| ranks | baseline | `cg2` | `oati` | `pcsi` |
|--:|--:|--:|--:|--:|
| 128 | 100.0 | 100.0 | 100.0 | 100.0 |
| 256 | 93.1 | 93.9 | 93.6 | 94.0 |
| 432 | 87.5 | 90.4 | 91.8 | 92.2 |
| 512 | 83.3 | 86.5 | 87.5 | 88.2 |
| **864** | **66.6** | **71.8** | **74.9** | **76.1** |

| ranks | `cg2` d_total | `oati` d_total | `pcsi` d_total | `cg2` d_SSH | `oati` d_SSH | `pcsi` d_SSH |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | −0.31 | −0.46 | −0.72 | −8.3 | −8.4 | −16.6 |
| 256 | −1.14 | −1.05 | −1.71 | −23.6 | −25.8 | −41.9 |
| 432 | −3.47 | −5.14 | −5.74 | −29.1 | −43.5 | −53.9 |
| 512 | −4.09 | −5.28 | −6.30 | −34.1 | −47.4 | −60.2 |
| **864** | **−7.59** | **−11.49** | **−13.10** | −33.0 | −49.9 | −58.6 |

Best achievable step time on this mesh: baseline 0.0435 · `cg2` 0.0402 (−7.6 %) · `oati`
0.0385 (−11.5 %) · `pcsi` 0.0378 (−13.1 %).

Four readings:

1. **Every solver recovers efficiency, and they rank consistently at every rung**
   (`pcsi` > `oati` > `cg2` > baseline). At 864: 76.1 / 74.9 / 71.8 / 66.6 % — so even the
   most conservative option, `cg2`, returns 5.2 points.
2. **`oati` is within 1.6 pp of `pcsi` and 1.6 % of it in step time** (0.0385 vs 0.0378), while
   `cg2` gives up about half the benefit. The gap between `oati` and `pcsi` is much smaller
   than the gap between either and `cg2`.
3. **The best achievable step time improves 13.1 %** (0.0435 → 0.0378). The baseline cannot
   reach 0.0378 at *any* rank count available to it — CORE2 has no partition beyond 864 — so
   this is performance the mesh could not previously reach, not the same performance sooner.
4. **`pcsi` at 864 ranks carries the SSH burden the baseline had at 512** (3.40 ms / 9.0 % of
   step vs 5.81 ms / 9.9 %) — roughly **1.7 rungs of scaling headroom returned**.

> **Why this matters for the recommendation, and why it is NOT simply "use pcsi".** `pcsi`
> wins here by a narrow margin over `oati`, but `pcsi` is also the solver that **fails on farc**
> (`d_SSH` +30…+33 %, iterations 212 → 377) and **loses hardest on dars g8n** (+21 %), because
> its iteration count depends on an eigenvalue estimate that can be poor. `oati` is Krylov —
> its iteration count tracks `cg`'s on every mesh measured so far — and it is the **best GPU
> solver** on the small-mesh rungs. On the evidence to date `oati` looks like the better
> default and `pcsi` like the specialist that needs a per-mesh eigenbound check first. That
> judgement is deferred to T11 pending the farc eigenbound investigation.

⚠️ Caveats kept with the number: the 864 rung packs 864 tasks over 7 nodes (~123/node) rather
than filling them, so its *absolute* time is not perfectly comparable to the fully-packed
lower rungs — the A/B deltas are unaffected (all legs share the packing) but the efficiency
column inherits that ragged edge. And the baseline is still *improving* at 864 (0.0587 →
0.0435), so CORE2 has not turned over; it is losing efficiency, not throughput. A turnover
point would need a partition beyond 864, which does not exist for this mesh.

### What these say

1. **The premise is confirmed, strongly.** Where the SSH solve is a limiting factor the win is
   an order of magnitude larger than the NG5 number suggested: `d_SSH` reaches **−20 % on GPU**
   (CORE2 g16n) and **−60 % on CPU** (CORE2 512 ranks) against −2 % whole-step on NG5.
2. **Both trends compound on the CPU ladder, which is the "extends the scaling range" result.**
   As CORE2 thins from 991 → 248 nodes/rank, the SSH share *rises* (2.5 → 4.4 → 9.9 %) **and**
   the solver gain *rises* (−16.6 → −41.9 → −60.3 %). The step time still falls
   (0.1955 → 0.1050 → 0.0587), so CORE2 is still scaling at 512 ranks — the ladder must be
   pushed further (1024+) to find where it stops and show the wall actually moving.
3. **⚠️ There are real losses, and they are systematic.** On **dars g8n** every solver is
   *slower* (`pcsi` +21 %). That rung has the largest per-rank workload (98 761 nodes/rank) and
   the smallest SSH share (5.7 %): the solve is compute-bound, not latency-bound, so the
   variants' extra vector work costs more than the saved synchronisation buys. This is the
   crossover, measured.
4. **⚠️ `pcsi` breaks on farc** — `d_SSH` **+30 to +33 %**, with iterations jumping
   **212 → 377 (+77 %)**. The Chebyshev iteration count is governed by the eigenbound interval;
   on farc the estimate is evidently poor. `cg2`/`oati` are unaffected (they are Krylov, not
   interval-driven). **Open item:** re-run the T8a estimator on a farc dump and compare
   `[ν,µ]`/κ against CORE2's 486.9 before recommending `pcsi` anywhere near farc.
5. **The winner depends on the backend, and this refutes my earlier hypothesis.** GPU favours
   **`oati`**; CPU favours **`pcsi`** by a wide margin. I had proposed that `pcsi`'s edge came
   from fewer device *fences*, which predicted its advantage would **shrink** on CPU. It
   **grew** — so that explanation is wrong. The better one: `pcsi` buys fewer reductions at the
   price of ~15 % more iterations; extra iterations are nearly free on CPU but cost kernel
   launches on GPU (T2: 3.0 µs async / 8.9 µs fenced), so the trade is excellent on CPU and only
   break-even on GPU. `oati` keeps the Krylov iteration count and still halves the syncs, which
   is why it wins where launches are expensive.
6. `oati` --- which measured as a *null* against `cg2` on NG5 --- is the **best GPU solver** on
   the small-mesh rungs (−20.49 % vs cg2's −14.49 %). The NG5 null was a low-SSH-share artefact,
   not a property of the method.

**Still running / to do:** CORE2 g32n, farc g16n, dars g32n, NG5 g16n (GPU); farc + dars CPU;
NG5 CPU **re-submitted at 1024 ranks** (the 256-rank attempt was cancelled during init —
7.4 M nodes is too tight there, job 26733444 void, replacement 26733926).

## Frozen binaries

*(`/work/ab0995/a270088/port2/m10/bin/` + sha256 here; binaries NEVER in git.)*
