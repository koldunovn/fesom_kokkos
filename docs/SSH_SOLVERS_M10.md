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
| 2026-08-06 | `FESOM_SSH_STALL_WINDOW` commit — knob-off byte gate | 26740665 | `0712ee57…` (main `9743f602…` ≠ — R9 armed) | **PASS** diff_snap rc=0 |
| 2026-08-06 | `FESOM_SSH_STALL_WINDOW` commit — explicit `FESOM_SSH_SOLVER=cg` byte gate | 26740666 | `0712ee57…` | **PASS** diff_snap rc=0 |

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

## ⭐⭐⭐ THE CROSSOVER MAP (F5) — where to use these solvers, and where not to

Every completed rung, both backends, `FESOM_SPEED=1` on all legs. `d_SSH` = the solver's own
speed-up, `d_tot` = whole-step. WIN = `d_tot` better than −2 %.

| mesh | backend | ranks | nodes/rank | SSH% | best `d_SSH` | best `d_tot` | |
|---|---|--:|--:|--:|--:|--:|---|
| CORE2 | **GPU** | 4 (1 node) | 31714 | 24.3 | −17.3 | **−5.19** | WIN |
| CORE2 | **GPU** | 8 | 15857 | 25.7 | −19.2 | **−5.65** | WIN |
| CORE2 | **GPU** | 16 | 7928 | 27.4 | −18.3 | **−5.18** | WIN |
| CORE2 | **GPU** | 64 | 1982 | 30.5 | −20.5 | **−6.73** | WIN |
| ~~farc~~ | ~~GPU~~ | ~~16~~ | 39899 | 36.5 | ~~−12.4~~ | ~~−5.12~~ | ❌ **VOID (fallbacks)** |
| ~~farc~~ | ~~GPU~~ | ~~32~~ | 19949 | 37.6 | ~~−14.7~~ | ~~−6.81~~ | ❌ **VOID (fallbacks)** |
| NG5 | **GPU** | 16 / 64 | 462680 / 115670 | 8.5 | −19.6 | **−2.32** | WIN |
| dars | GPU | 32 | 98760 | 5.7 | +4.6 | −0.64 | marginal |
| CORE2 | CPU | 864 | 146 | 18.9 | −58.6 | **−13.10** | WIN |
| **CORE2** | **CPU** | **512** | **247** | 9.9 | **−60.2** | **−6.30** | **WIN (production)** |
| CORE2 | CPU | 432 | 293 | 9.4 | −53.9 | **−5.74** | WIN |
| CORE2 | CPU | 256 | 495 | 4.4 | −41.9 | −1.71 | marginal |
| CORE2 | CPU | 128 | 991 | 2.5 | −16.6 | −0.72 | marginal |
| NG5 | CPU | 2048 | 3614 | 1.6 | +5.8 | −0.42 | ❌ LOSE |
| dars | CPU | 1024 | 3086 | 0.8 | **+149.6** | −0.18 | ❌ LOSE |
| ~~farc~~ | ~~CPU~~ | ~~128~~ | 4987 | 4.3 | — | — | ❌ VOID (fallbacks) |
| dars | CPU | 512 | 6172 | 0.8 | **+77.2** | **+1.25** | ❌ LOSE |
| NG5 | CPU | 1024 | 7229 | 1.1 | +26.8 | −0.38 | ❌ LOSE |
| ~~farc~~ | ~~CPU~~ | ~~64~~ | 9974 | 3.3 | — | — | ❌ VOID (fallbacks) |

### ❌❌ RETRACTION — every farc row is VOID: the solvers do not converge reliably on farc

**All farc A/B legs, on BOTH backends, fired the auto-fallback guard.** My own pre-registered
certification criterion is *"0 fallback firings across the 300-step measurement runs"*, and I
was not harvesting that number — so contaminated legs were reported as wins.

| farc run | `cg2` | `oati` | `pcsi` | status |
|---|--:|--:|--:|---|
| GPU 4 nodes (16 r) | — | **20** | **6** | ❌ void |
| GPU 8 nodes (32 r) | **23** | **21** | **5** | ❌ void |
| CPU 32 r | **20** | **21** | ? | ❌ void |
| CPU 64 r | **20** | **19** | **6** | ❌ void |
| CPU 1024 r | **20** | **22** | **4** | ❌ void |
| CPU 2048 r | **21** | **6** | **6** | ❌ void |

A leg with fallbacks is a **mixture of variant and baseline solves**, so its timing is not an
A/B point at all — and the absurd iteration counts it produced (`pcsi` 12.03/solve where the
baseline needs 212; `oati` 69.70) are the averaging artefact of that mixture, not convergence.

**What the guard was actually reporting:** `maxiter exhausted without convergence` (`pcsi`,
after 501 iterations) and `residual stalled or grew over the watch window` (`cg2`/`oati`, after
~112). About **20 of 300 solves (≈7 %)** on farc, every run.

**Why farc and not CORE2.** farc is the worst-conditioned system measured (κ = 843 vs CORE2's
489) and needs **212 iterations/solve** against CORE2's 106. The CG-CG family replaces PCG's
explicit `(p,Ap)` with a *recurrence*; rounding in that recurrence accumulates with iteration
count even when the preconditioner is symmetrised, so a system needing twice the iterations
gets roughly twice the accumulated drift — and on farc it crosses the stall threshold on ~7 %
of solves. `pcsi` fails differently: its Chebyshev interval is wrong on farc (§eigenbound
diagnosis), so it exhausts `maxiter` outright.

**This is a real robustness limit of the methods, and arguably the most important negative in
the campaign:** the communication-avoiding variants are not simply "slower on some meshes" —
on an ill-conditioned mesh they *fail to converge* on a noticeable fraction of solves, and only
the armed fallback keeps the model correct. Baseline `cg` fired **zero** fallbacks on every
farc run.

**Process fix:** `fallbacks=` is now harvested per leg and printed per rep, with an explicit
`[!! ] this leg is a variant/baseline MIXTURE` warning. **No timing may be quoted without it.**

**Unaffected:** every CORE2 rung (CPU and GPU), every dars rung, and NG5 GPU show
`fallbacks=0` on all legs — the audit is in the git history and those rows stand.

### ⚠️ The CPU rows for dars/NG5 are measured OUTSIDE their scaling range — do not conclude from them yet

FESOM's rule of thumb (user, and the `namelist.config` guidance): a mesh scales until roughly
**300–500 vertices per core**. Against that, the CPU rungs above are nowhere near where these
meshes are actually run:

| mesh | vertices | scaling range (ranks) | = CPU nodes | what I tested | vertices/core tested |
|---|--:|--:|--:|---|--:|
| CORE2 | 126 858 | 253 – 422 | 2 – 3 | 128…864 | 991 … **146** ✅ spans it |
| farc | 638 387 | 1 276 – 2 127 | 9 – 16 | 64, 128 | **9 974, 4 987** ❌ 10–30× too coarse |
| dars | 3 160 340 | 6 320 – 10 534 | 49 – 82 | 512, 1024 | **6 172, 3 086** ❌ |
| NG5 | 7 402 886 | 14 805 – 24 676 | 115 – 192 | 1024, 2048 | **7 229, 3 614** ❌ |

**Only CORE2 was tested inside its operating range** — which is exactly why it is the only mesh
where the CPU numbers looked good. The farc/dars/NG5 "LOSE" rows say the solvers do not help at
3 000–10 000 vertices/core, which is true but uninteresting: nobody runs there, and it is the
regime where the SSH solve is a negligible 0.8–4 % of the step by construction.

**Corrected rungs submitted** (jobs 26738526-31), each inside or as close to the range as the
available partitions allow:

| mesh | ranks | nodes | vertices/core | note |
|---|--:|--:|--:|---|
| farc | 1024 / **2048** | 8 / **16** | 623 / **312** | ✅ reaches the 300–500 range |
| dars | 2048 / **4096** | 16 / **32** | 1543 / **771** | 4096 is the LARGEST partition that exists |
| NG5 | 4096 / **8192** | 32 / **64** | 1807 / **903** | 8192 is the LARGEST partition that exists |

⚠️ **dars and NG5 cannot reach 300–500 vertices/core with the partitions that exist** — dars
would need ~6 300–10 500 ranks (49–82 nodes) and NG5 ~14 800–24 700 (115–192 nodes), against
maxima of 4 096 and 8 192. So even the corrected rungs sit at 771 and 903 vertices/core, still
short of the knee. Reaching it would need new partitions (the `core2_bigpart` recipe applies)
and a 64–192 node allocation. Until then, the CPU verdict for those two meshes is **provisional
at best**, and the honest statement is "not yet measured where it matters".

#### ✅ RESOLVED for dars — the partitions now exist (job 26738726, 2026-08-06)

`/work/ab0995/a270088/port2/mesh/dars_bigpart` carries `dist_6144`, `dist_8192`, `dist_10240`
= **514 / 385 / 308 vertices per core**, i.e. the 300–500 range is now spanned. The md5
integrity gate on `nod2d/elem2d/nlvls/elvls` passed before any partition was written, and the
copy is a full rsync (not a symlink — `/pool` is writable for this account, so a symlink could
carry a stray write back into `/pool`). A/B runs submitted: **26741040** (6144 = 48 nodes) and
**26741041** (8192 = 64 nodes), four legs each (`cg`/`cg2`/`oati`/`pcsi`), binary pinned to
`stallknob_serial`.

🔴 **These rungs form their OWN curve.** The `core2_bigpart` control measured flat METIS
partitions as **7.4 % slower** than the certified mesh's at the same rank count, so
`dars_bigpart` rows must never be merged into a scaling curve with the `/pool` dars rows.
*The variant-vs-baseline A/B is unaffected* — both legs run on the same partition inside one
allocation, so the partition-source bias cancels in the ratio. Only cross-rung and absolute
comparisons are contaminated.

NG5's generator (26738744) is still running: 16384 took >1 h, with 20480 and 24576 to go.

### The rule that falls out (GPU side is solid; CPU side is CORE2-only so far)

**The split is by BACKEND, not by mesh size** (sorting the table by nodes/rank does *not*
separate the wins from the losses; sorting by backend does):

- **GPU — use them essentially everywhere.** Every GPU rung measured wins, from 1 node to 16,
  on three meshes. The reason is visible in the SSH% column: on GPU the solve is **8–38 % of
  the step** because the GPU makes the local compute fast while the solve stays latency-bound.
  Even NG5 at 462 680 nodes/rank wins.
- **CPU — only when the per-rank problem is small.** Wins on CORE2 at ≲1000 nodes/rank; loses
  on farc/dars/NG5, where per-rank data is 3000–10 000 nodes. There the SSH solve is only
  0.8–4 % of the step *and* the variants make the solve itself **slower** (dars 512 ranks:
  +77 %; dars 1024: +150 %).

**Why the CPU losses.** The variants trade reductions for vector work and keep more vectors
live (`cg2` 5, `pipecg`/`oati` 9, vs baseline 4). With ~250 nodes/rank those vectors sit in
cache and only latency matters — the trade is excellent. With 6000+ nodes/rank the extra
passes are memory-bandwidth-bound and cost more than the saved synchronisation. On GPU the
bandwidth is ~an order of magnitude higher, so the extra passes stay cheap and the trade wins
regardless of per-rank size. *(Mechanism inferred from the measurements, not separately
instrumented — a bandwidth/cache counter run would confirm it.)*

⚠️ The CPU losses are mostly small in whole-step terms (−0.4 % … +1.25 %) precisely because
the SSH share is tiny there — but **dars at 512 CPU ranks is a genuine +1.25 % regression**,
so this is a real "do not enable", not just "no benefit".

### ✅ farc at ≥128 ranks does NOT hang with our environment

The inherited caveat ("farc ≥128 ranks = reproducible proto hang") was **over-general**. The
M7 finding was specific to the `UCX_PROTO_ENABLE=y` env-package leg; the standard-env legs ran
clean on the same allocations. Our jobs never set `UCX_PROTO_ENABLE`. Probe job **26735924**
ran farc at **128 CPU ranks** to completion (0.9480 s/step, 231.65 iters/solve, no hang). The
GPU probe at 128 ranks (26735925) is still queued. **Corrected caveat: do not set
`UCX_PROTO_ENABLE=y` on farc at ≥128 ranks — the rank count itself is fine.**

## ⭐ THE PRACTICAL RESULT — CORE2 at 512 CPU ranks (the production configuration)

**This is the number that matters.** CORE2 on 512 CPU ranks is the production setup; everything
below 300 nodes/rank was previously reported here in terms of large percentages at rank counts
nobody runs at, which was a mis-weighting. Measured on the certified mesh, single clean runs,
`FESOM_SPEED=1` on every leg (job 26733441):

| leg | s/step | **whole-step** | SSH ms/step | **solve itself** | SSH% of step |
|---|--:|--:|--:|--:|--:|
| production baseline | 0.0587 | — | 5.81 | — | 9.9 |
| `+ cg2` | 0.0563 | **−4.09 %** | 3.83 | −34.1 % | 6.8 |
| `+ oati` | 0.0556 | **−5.28 %** | 3.06 | −47.4 % | 5.5 |
| **`+ pcsi`** | **0.0550** | **−6.30 %** | 2.31 | **−60.2 %** | 4.2 |

**A 5–6 % whole-step saving on the production configuration, from a solver swap** — no new
hardware, no repartitioning, no change to the science. The solve itself gets 47–60 % faster;
it is 9.9 % of the step, which is what converts that into ~5–6 % overall.

### And it moves the knee

Parallel efficiency on the certified CPU ladder (each leg vs its own 128-rank point):

| ranks | nodes/rank | baseline | `oati` | `pcsi` |
|--:|--:|--:|--:|--:|
| 128 | 991 | 100.0 | 100.0 | 100.0 |
| 256 | 495 | 93.1 | 93.6 | 94.0 |
| 432 | 293 | 87.5 | 91.8 | 92.2 |
| **512** | **247** | **83.3** | **87.5** | **88.2** |
| 864 | 146 | 66.6 | 74.9 | 76.1 |

Production at 512 ranks already sits just past the knee at 83.3 % efficiency; the solvers put
it back to ~88 %. Taking the largest rank count that still holds a given efficiency:

| efficiency floor | baseline | `pcsi` | knee moves | throughput at that point |
|---|--:|--:|--:|--:|
| ≥ 85 % | 479 ranks, 0.0618 s/step | 605 ranks, 0.0504 | **×1.26** | **−18.4 %** |
| ≥ 80 % | 581 ranks, 0.0557 | 750 ranks, 0.0434 | **×1.29** | **−22.2 %** |
| ≥ 75 % | 686 ranks, 0.0512 | 864 ranks, 0.0378 | ×1.26 | −26.1 % |

So the knee moves out by about **1.25–1.3×** in cores, and if you choose to spend those cores
you get **18–26 % more throughput at the same efficiency** you accept today.

### Gain at fixed rank counts — where it is and is not worth doing

| ranks | nodes/rank | `oati` | `pcsi` | verdict |
|--:|--:|--:|--:|---|
| 128 | 991 | −0.46 % | −0.72 % | not worth it — SSH is 2.5 % of the step |
| 256 | 495 | −1.05 % | −1.71 % | marginal |
| 432 | 293 | −5.14 % | −5.74 % | **worthwhile** (FESOM's own ~300 n/rank scaling guidance) |
| **512** | **247** | **−5.28 %** | **−6.30 %** | **production — worthwhile** |
| 864 | 146 | −11.49 % | −13.10 % | past the practical knee; large %, low value |

**Reading:** the benefit switches on around 300 nodes/rank and is already worth having at the
production point. Below that the SSH solve is too small a share for anything to matter.

### On GPU the practical case is stronger, at ordinary node counts

The GPU rungs are not extreme configurations — 4 and 8 nodes — and the SSH share there is
already 27–38 %, because a GPU makes the local compute fast while the solve stays
latency-bound:

| config | nodes | SSH% | best solver | whole-step |
|---|--:|--:|---|--:|
| CORE2 g4n | 4 | 27.4 | `oati` | **−5.18 %** |
| CORE2 g16n | 16 | 30.5 | `oati` | −6.73 % |
| farc g4n | 4 | 36.5 | `oati` | **−5.12 %** |
| farc g8n | 8 | 37.6 | `oati` | **−6.81 %** |

⚠️ The GPU ladder is only two points per mesh, so no knee analysis is possible there yet —
a proper GPU efficiency curve needs 1/2/4/8/16-node rungs per mesh, which is not yet run.

## Past the wall (methodological, not a production recommendation): CORE2 to 2048 ranks

CORE2's shipped maximum is 864 ranks, which was not enough to show a turnover. Larger
partitions were generated with the user's partitioner (METIS 5.1.0, flat single-level,
2D+3D weighted) into a **separate copy** of the mesh,
`/work/ab0995/a270088/port2/mesh/core2_bigpart` — never the certified mesh, per the standing
rule. Integrity was verified before use (`nlvls` 3832750 / `elvls` 7366752 / `nod2d` md5
identical to the certified mesh). New partitions: 864 (control), 1024, 1536, 2048.

### ⚠️ FIRST: the control says the two ladders are NOT comparable

A regenerated **864** was built specifically to test whether these flat partitions time the
same as the certified mesh's existing `dist_864`:

| `dist_864` | baseline s/step |
|---|--:|
| certified mesh (as shipped) | **0.0435** |
| `core2_bigpart` (flat METIS, built here) | **0.0467** |

**7.4 % slower on the same mesh, same rank count, same binary** — so the partitioning strategy
matters, and the certified partitions are better than a naive flat decomposition (most likely
they are hierarchical; the namelist's own default is `n_levels = 2`). **The new rungs
therefore must NOT be appended to the 128–864 certified ladder** — a combined curve would
attribute a partitioner difference to the solvers. The table below is a *separate, internally
consistent* ladder in which every rung uses a flat partition from the copy.

*(This is why the control was built. Without it the 1024–2048 points would have been plotted
straight onto the certified curve and the 7.4 % offset would have been read as physics.)*

### The ladder (all `core2_bigpart`, flat partitions, `FESOM_SPEED=1` on every leg)

| ranks | nodes/rank | SSH% | baseline | eff % vs 864 | `cg2` | `oati` | `pcsi` |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 864 | 146 | 21.3 | 0.0467 | 100.0 | −7.49 | −9.85 | **−11.78** |
| 1024 | 123 | 21.8 | 0.0410 | 96.1 | −6.34 | −10.73 | **−12.68** |
| 1536 | 82 | 26.6 | 0.0408 | 64.4 | −8.09 | −12.25 | **−13.48** |
| **2048** | **61** | **29.9** | 0.0356 | 55.4 | −9.55 | −14.04 | **−16.57** |

*(1536/2048 RE-MEASURED cleanly as jobs 26735610/26735611 after the tag-collision retraction
below; the replacements agree with the contaminated values to ~1–2 pp, so the retracted run
was wrong in provenance rather than wildly wrong in value — but it is the clean pair that is
quoted.)*

> ### ❌ DATA-INTEGRITY RETRACTION (found 2026-08-06, before the numbers were used)
>
> **The 1536 and 2048 rows are withdrawn.** A `sed` formatting error in the submission line
> printed an error but did **not** stop the `sbatch` calls, so each of those two rungs was
> submitted **twice** — jobs 26734774 *and* 26734780 at 1536, 26734776 *and* 26734781 at 2048.
> Both members of each pair carried the same `TAG`, hence the same output directory, and the
> job begins with `rm -rf "$OUT"` — so the concurrent runs wiped and interleaved each other's
> logs while both were writing.
>
> The symptom that exposed it: the consolidated harvest produced **two different results for
> the same 2048 configuration** (baseline 0.0369 s/step with `pcsi` −17.89 %, and 0.0355 with
> `pcsi` missing entirely). One configuration cannot have two answers.
>
> **Unaffected:** 864 control (job 26734773) and 1024 (job 26734671) were each single runs with
> unique tags — those rows stand. The certified-mesh ladder (128–864) is untouched.
>
> **Fix:** `$OUT` now includes `$SLURM_JOB_ID`, so a tag collision cannot share a directory.
> Re-runs submitted as jobs **26735610** (1536) and **26735611** (2048) with unique tags.
>
> ⚠️ **Everything derived from the 2048 row is suspended pending the re-run** — including the
> "`pcsi` at 1024 ranks matches the baseline at 2048" claim and the -14.6 % best-step-time
> figure below. They are left in place, struck, so the correction is visible rather than
> quietly edited away.

### The wall, and what the solvers do to it

⚠️ *The paragraphs below rest on the VOIDED 1536/2048 rows and are suspended pending jobs
26735610/26735611. Retained struck-through rather than deleted, so the correction is visible.*

~~**The baseline stops scaling between 1024 and 1536: 1.5× the cores buys 1.005× the speed.**~~
Parallel efficiency collapses from 96 % to 64 % across that step and reaches 55.5 % at 2048.
The SSH share climbs to **29.6 %** — nearly a third of the step is the solve — which is
exactly why this is where the solvers pay most: their whole-step gain grows monotonically
along the ladder (`pcsi` −11.78 → −12.68 → −13.97 → **−14.65 %**).

**The cleanest statement of the benefit — a 2× core saving:**

| configuration | ranks | s/step |
|---|--:|--:|
| baseline | 2048 | 0.0355 |
| **`pcsi`** | **1024** | **0.0358** |

**`pcsi` at 1024 ranks matches the baseline at 2048 ranks (within 1 %)** — i.e. on this mesh,
in the regime where it has stopped scaling, the solver delivers the same time on **half the
cores**. Best achievable step time overall: baseline 0.0355 → `pcsi` 0.0303 (**−14.6 %**),
`oati` 0.0305, `cg2` 0.0318.

⚠️ Honest limit: `pcsi` also plateaus (1024 → 1536 gives 1.020× for 1.5× cores). The solvers
**lower the curve and buy roughly one doubling of cores; they do not remove the wall.**
Something else becomes limiting beyond ~1500 ranks on this mesh, and identifying it is not
part of M10.

## ⚠️ Why `pcsi` fails on farc — the eigenbound diagnosis

`pcsi` costs +30…+33 % of the SSH phase on farc, with iterations 212 → 377. The model logs the
interval it computed, so the bounds are recoverable from the A/B runs directly:

| mesh | θmin (Lanczos m=30) | θmax | κ after margins | √κ |
|---|--:|--:|--:|--:|
| dars g8n | 1.4548e-02 | 1.3635 | 109.3 | 10.5 |
| CORE2 g16n | 3.4455e-03 | 1.4440 | 489.0 | 22.1 |
| **farc** | **2.0274e-03** | 1.4647 | **842.9** | 29.0 |

farc is worse-conditioned, but only by 1.7× against CORE2 — and Chebyshev cost scales like
**√κ**, so that alone predicts ~1.3× more iterations, not the 3.1× observed. Comparing each
mesh against the Chebyshev bound `½√κ·ln(2/ε)` at `soltol = 1e-5` settles it:

| mesh | Chebyshev prediction | `pcsi` observed | **observed / predicted** |
|---|--:|--:|--:|
| dars | 64 | 48.1 | 0.75 |
| CORE2 | 135 | 121.4 | 0.90 |
| **farc** | **177** | **376.4** | **2.12** |

**A Chebyshev iteration cannot converge more slowly than its own bound if the spectrum lies
inside `[ν,µ]`.** dars and CORE2 come in *under* the bound (0.75, 0.90 — as expected, the bound
is pessimistic). farc runs at **2.12× the bound**, which is only possible if part of the
spectrum is **outside** the interval. Since µ is inflated and the polynomial is well-behaved
above it, the failure must be at the bottom: **Lanczos at m=30 is overestimating λmin on farc**,
so eigenmodes below ν are not damped at all and convergence stalls on them.

This is precisely the failure mode T-5 warns about (a wrong ν inflates the assumed κ and
mis-tunes the polynomial) — except here the cause is not the missing square root, which is
fixed, but simply **too few Lanczos steps for farc's spectrum**. It is also consistent with
`cg2`/`oati` being unaffected: they are Krylov methods whose iteration count adapts to the
actual spectrum, with no interval to get wrong.

### ✅ RESOLVED — the Lanczos sweep (jobs 26734516 farc, 26734517 CORE2 control)

**Prediction 1 confirmed, prediction 2 wrong.** `θmin` versus Lanczos steps:

| mesh | m=30 | m=120 | m=250 | κ: 30 → 120 | fallbacks at m=30 → m=120 |
|---|--:|--:|--:|--:|---|
| **farc** | 2.027e-03 | **3.482e-04** | 3.249e-04 | **843 → 4911 (5.8×)** | **6 → 0** ✅ |
| CORE2 (control) | 3.446e-03 | 2.513e-03 | 2.459e-03 | 489 → 670 (1.4×) | 0 → 0 |

`m=30` **under-estimates the spectrum on both meshes** — badly on farc (5.8×). I predicted
CORE2 would be "essentially flat"; it moved 37 %, so that half of the prediction was wrong.

**The correction fixes CONVERGENCE but not COMPETITIVENESS**, and the theory now matches
measurement exactly:

| farc, `pcsi` | κ used | predicted iters (`½√κ·ln(2/tol)`) | measured iters | fallbacks | vs baseline |
|---|--:|--:|--:|--:|--:|
| m=30 | 843 | 177 | 350 ❌ (2× predicted) | **6** | +11.61 % |
| **m=120** | **4911** | **428** | **376** ✅ | **0** | +9.08 % |
| m=250 | 5263 | 442 | 389 ✅ | 0 | +11.27 % |

At m=30 the measured count was **2× the prediction** — the signature of a spectrum outside
`[ν,µ]`. At m=120 measurement and theory agree and **the fallbacks vanish**. But farc's true
κ ≈ 4900 needs ~400 Chebyshev iterations against CG's 212, so **`pcsi` is simply the wrong
method for a system this ill-conditioned** — correctly converged, and still 9 % slower.

**⚠️ The uncomfortable part: the old default was faster BECAUSE it was wrong.** On CORE2,
m=30 gives −7.20 % and m=120 −5.40 %: under-estimating κ yields a more aggressive polynomial
that wins *when it happens to converge*. On farc that same aggression caused convergence
failures. **Correctness wins — the default is now `FESOM_PCSI_LANCZOS=120`**, accepting ~1.8 pp
of CORE2 speed for a bound that is actually converged.

**Also added:** a setup-time suitability warning. `pcsi` now prints its predicted Chebyshev
iteration count and flags `ILL-CONDITIONED: pcsi is likely a poor choice here` when that
exceeds 60 % of `maxiter` — so the farc case announces itself before a run is spent on it.

*(Historical: the original pre-registration below.)*
**Test in flight** (jobs 26734516 farc, 26734517 CORE2 control): sweep
`FESOM_PCSI_LANCZOS` ∈ {30, 120, 250}. Pre-registered predictions —
1. On **farc**, θmin should keep falling as m grows (the m=30 value is not converged), and the
   iteration count should fall with it. If θmin is instead stable at 2.03e-03 across m, the
   diagnosis is WRONG and the problem is elsewhere (candidates: a genuinely disconnected low
   cluster that Lanczos finds but the margin does not cover, or a farc-specific conditioning
   feature) — that outcome gets reported with equal prominence.
2. On **CORE2** (control), θmin should be essentially flat across m, since its obs/pred of 0.90
   says the m=30 interval is already good there.

**Consequence for the recommendation either way:** `pcsi`'s cost depends on a per-mesh estimate
that can silently be wrong, and the only symptom is slowness — it still converges, and the true
residual still passes. `oati` has no such dependency. This is the substantive argument for
`oati` as the default and `pcsi` as a per-mesh specialist, independent of which mesh happens to
win a given A/B.

## 🔴 Why `cg2`/`oati` fall back on farc — open item 3, and the framing was wrong

**Status: IN PROGRESS.** Two facts are already established and they move the question.

### ❌ Contamination class 4: `fallbacks=0` on baseline `cg` is VACUOUS

The retraction above reads *"Baseline `cg` fired **zero** fallbacks on every farc run"* as
evidence that the communication-avoiding variants are less robust than `cg`. **It is not
evidence. The baseline `cg` body carries no fallback guard at all** — brace-matching
`fesom_ssh_solve_cg_kk` (`src/fesom_ssh.cpp:3897-4298`, 402 lines) gives **0** references to
`SSH_FB`/`STALL`/`ssh_fb_*` anywhere inside it. The guard (`ssh_fb_announce`,
`SSH_FB_STALL`/`NAN`/`INDEF`/`MAXITER`) exists only inside `ssh_solve_cg2` /
`ssh_solve_pipecg` / `ssh_solve_pcsi` / `ssh_solve_oati`. `cg` is the fallback *target*, so
its counter is structurally incapable of incrementing.

What baseline `cg` on farc *does* establish is weaker but real: it neither hit `maxiter=500`
nor produced a NaN (both are `FESOM_DIE`, and no farc run died), so it converged on every
solve. The comparison as stated in the retraction is a **monitored path against an unmonitored
one**, which is a fourth contamination class of the same family as the other three — an
apparent result produced by an asymmetry in the harness rather than by the physics.

*(The retraction itself STANDS: a leg with fallbacks is a variant/baseline mixture and its
timing is not an A/B point. What does not stand is the inference "the variants fail to
converge where `cg` succeeds" — that was never measured.)*

### ⭐⭐⭐ RESOLVED — the farc "breakdown" is a FALSE POSITIVE of our own guard (26740651 + 26741060)

🔴 **This section supersedes both earlier readings, including the one I committed in
`cfafe73` ("the breakdown is REAL"). It was wrong.** The trace below is genuine, but it is
only the first 107 iterations of a solve that converges at 205.

**The decisive run (26741060): the same dumped system, the same binary, the only difference
being `FESOM_SSH_STALL_WINDOW=100000`.**

| solver | iterations | final residual | vs baseline `cg` (211) |
|---|--:|--:|--:|
| `cg2` | **205** | 4.3045e-01 < rtol | **6 FEWER** |
| `oati` | 205 | 4.4407e-01 | 6 fewer |
| `pipecg` | 204 | 4.4580e-01 | 7 fewer |

The lab's residuals reproduce the in-model trace to 5 significant figures (it 87 = 1.5816e+01,
it 95 = 1.8186e+01, it 107 = 1.6213e+01), so it is provably the identical solve. Continuing
past where the model gave up:

```
it 107   1.6213e+01     <- the guard fired HERE
it 110   1.4375e+01     <- the stagnation ENDS, three iterations later
it 130   5.6246e+00
it 205   4.3045e-01  <  rtol 4.3743e-01     CONVERGED, 0 fallbacks
```

**The stagnation ran ~21 iterations (87 → 108) against a `STALL_WINDOW` of 20. The guard
aborted a healthy solve three iterations before it resumed converging.** Residual
non-monotonicity is normal CG behaviour — only the A-norm of the error decreases
monotonically — and on a mesh with κ = 843 a 20-iteration plateau is ordinary, not pathological.

**Consequences, and they are large:**

1. `cg2`/`oati`/`pipecg` **do not break down on farc.** They converge in *fewer* iterations
   than baseline `cg` on the very solve that was reported as a divergence.
2. The robustness defect is in **our fallback heuristic**, not in the solvers or in the
   literature's methods.
3. The retraction's *conclusion* — "on an ill-conditioned mesh they fail to converge on a
   noticeable fraction of solves" — **does not survive**. (Its *timings* stay retracted: a
   fallback still makes a leg a variant/baseline mixture, so those numbers are unusable
   regardless of why the guard fired.)
4. Baseline `cg` "never fails" on farc for the same reason it reports `fallbacks=0`: **it is
   not watched.** It rides through the identical 21-iteration stagnation because nothing is
   checking. The two facts are the same fact.
5. **Every farc A/B row should be re-measured with a corrected window** before farc is called
   a loss for these solvers.

**Why this was so hard to see:** the in-model evidence (a residual that visibly grows and
oscillates for 20 iterations) looks exactly like a genuine breakdown, and it is what talked me
out of the correct hypothesis. Only running past the abort point distinguishes "stalled" from
"stalling *right now*". That is what the `FESOM_SSH_STALL_WINDOW` knob was built for.

#### The airtight control (26741140) — the knob is the ONLY difference

Same system, same binary, same `srun`, run twice:

| `FESOM_SSH_STALL_WINDOW` | `cg2` result |
|---|---|
| unset (=20) | **FALLBACK at 107 iters**, res 1.6213e+01 → redone with `cg`, 211 iters, bitwise exact |
| 100000 | **converged at 205 iters**, res 4.3045e-01, 0 fallbacks |

`oati` on the same system fires **0** fallbacks even at the default window (it converges at
206), which is why the two solvers disagree about which farc solves "fail": the trigger is
sensitive to exactly where each method's plateau falls relative to a fixed counter.

#### 🔴 The threshold is BELOW what baseline `cg` itself needs (26741140-46)

Longest run of consecutive non-improving iterations, guard widened, over the 7 captured farc
systems:

| system | `cg` | `cg2` | `oati` |
|---|--:|--:|--:|
| step0010 | 9 | 8 | 4 |
| step0020 | 11 | 12 | 5 |
| step0030 | 15 | 17 | 8 |
| step0040 | 9 | 5 | 3 |
| step0050 | 11 | 10 | 5 |
| step0060 | 2 | 5 | 2 |
| **step0037** (the "failure") | **21** | **20** | 7 |

**Baseline `cg` — the certified production solver — plateaus for 21 iterations on farc, above
the guard's threshold of 20.** So the guard is not merely strict; it is calibrated below what
the *reference* method requires on this mesh. `cg` escapes only because it is not watched.
`cg2` tracks `cg` closely (8/12/17/5/10/5/20 against 9/11/15/9/11/2/21), exactly as two
implementations of the same Krylov method should.

For contrast, CORE2's longest plateau is **2–3** (measured above) — which is why a window of
20 has never been a problem there, and why the constant survived until a genuinely
ill-conditioned mesh was tested.

**Sizing a safe window:** it must clear the *baseline's* worst plateau with margin. Observed
max is 21 on farc over 7 systems; CORE2 is 3. The `resid > 1e3 * best` divergence trigger is
the one that actually catches the Sergey-class breakdown (α wrong by 21.8 %, residual growing
without bound), and it stays armed independently — so the plateau counter can be loosened a
long way without losing real protection. The re-measurement runs below use **200**.

⚠️ **The default is NOT changed in this commit.** Choosing the shipped value is a judgement
call (how long should the model burn iterations before giving up on a solver?) and belongs
with the adoption decision — see the recommendation section.

#### Recovering the voided farc rows

The retracted farc A/B legs were void because they were variant/baseline mixtures. With the
window corrected the mixture should disappear, so the farc verdict can finally be measured:
**26741203** (farc CPU 1024 = 8 nodes, 623 v/core) and **26741204** (2048 = 16 nodes, 312
v/core — inside the 300–500 scaling range), four legs each,
`FESOM_SSH_STALL_WINDOW=200` on every variant leg, binary pinned. `fallbacks=` is harvested
per leg and must read 0 for these rows to count.

#### ⭐⭐⭐ RECOVERED — farc CPU 2048 is the BEST CPU result in the campaign (26741204)

16 nodes, 2048 ranks, **312 vertices/core — inside the 300–500 scaling range**, dt 900,
300 steps, min-of-2, one allocation. **`fallbacks=0` on all four legs.**

| leg | s/step | d_total | SSH ms/step | d_SSH | SSH% | iters/solve | ms/iter |
|---|--:|--:|--:|--:|--:|--:|--:|
| `cg` | 0.0821 | +0.00 % | 18.72 | +0.00 % | 22.8 | 211.80 | 0.089 |
| `cg2` | 0.0739 | **−9.99 %** | 13.01 | −30.52 % | 17.6 | 210.82 | 0.062 |
| **`oati`** | **0.0712** | **−13.28 %** | **10.18** | **−45.61 %** | 14.3 | 211.27 | 0.048 |
| `pcsi` | 0.0724 | −11.81 % | 11.37 | −39.28 % | 15.7 | 376.43 | 0.030 |

**Iteration parity holds at scale** — 210.82 / 211.27 against `cg`'s 211.80 — which is the
independent confirmation that the guard, not the mathematics, was the farc problem.

🔴 **This falsifies the ledger's own crossover rule.** "CPU — only when the per-rank problem
is small. Wins on CORE2 …; loses on farc/dars/NG5" is **wrong for farc**: at its real
operating point farc is a *bigger* CPU win than CORE2's production configuration
(`oati` −13.3 % vs −5.3 %). **Two independent errors compounded** to produce the old verdict:

1. farc CPU was measured at 64/128 ranks = 9 974/4 987 vertices/core, 10–30× coarser than
   its operating range, where the SSH share is 3–4 % of the step and nothing can matter. At
   2048 ranks the share is **22.8 %**.
2. Every farc row that *was* in range got voided by the guard's false positives.

Neither error was visible from inside the affected rows — the first needed the scaling-range
audit, the second needed running past the abort point.

⚠️ **A ~2.9 pp common-mode gap is NOT yet accounted for (L88).** Whole-step saving vs the
solver's own accounted saving:

| leg | whole-step saved | solver accounts | unaccounted |
|---|--:|--:|--:|
| `cg2` | 8.20 ms | 5.71 ms | **2.49 ms (3.0 pp)** |
| `oati` | 10.90 ms | 8.54 ms | **2.36 ms (2.9 pp)** |
| `pcsi` | 9.70 ms | 7.35 ms | **2.35 ms (2.9 pp)** |

The gap is near-identical across three different solvers, so it is **systematic, not noise**.
Candidates: a first-leg warm-up penalty on the baseline (its rep spread is 1.3 % against the
variants' 0.4 %), or a real secondary benefit (less network contention speeding other phases).
**Control in flight — 26741360 runs the legs in REVERSED order with a repeated `cg` leg last**;
if `cg`-last beats `cg`-first in the same allocation, the gap is an ordering artefact and the
defensible whole-step figures are the solver-accounted ones (`cg2` −7.0 %, `oati` −10.4 %,
`pcsi` −8.9 %). **Until that lands, quote `d_SSH` — which is directly measured — and treat
`d_total` as an upper bound.**

*(`--sym-check` on the same farc system: `pr_values` defect ratio **0.741**, the worst of the
three meshes measured — CORE2 0.638, dars 0.616. Consistent with farc being hardest for the
CG-CG family, though `SYMPRE=1` neutralises it in practice.)*

#### The other three recovered/new in-range CPU rungs — all `fallbacks=0`

**farc 1024** (8 nodes, 623 v/core, guard widened) — 26741203:

| leg | s/step | d_total | d_SSH | SSH% | iters |
|---|--:|--:|--:|--:|--:|
| `cg` | 0.1392 | — | — | 11.7 | 212.06 |
| `cg2` | 0.1333 | −4.24 % | −23.06 % | 9.4 | 211.10 |
| **`oati`** | **0.1314** | **−5.60 %** | −35.46 % | 8.0 | 211.66 |
| `pcsi` | 0.1340 | −3.74 % | −19.37 % | 9.8 | 376.45 |

**dars at its real scaling range for the first time** (`dars_bigpart`, own curve) — 26741040 /
26741041. Note `FESOM_SSH_STALL_WINDOW` was NOT set on these legs and they still fired zero
fallbacks: dars is well-conditioned (36 iters/solve against farc's 212), so the guard was
never close to tripping.

| rung | v/core | leg | s/step | d_total | d_SSH | SSH% |
|---|--:|---|--:|--:|--:|--:|
| **6144** (48 N) | 514 | `cg` | 0.1329 | — | — | 5.6 |
| | | `cg2` | 0.1302 | −2.03 % | −10.78 % | 5.1 |
| | | `oati` | 0.1289 | **−3.01 %** | −27.26 % | 4.2 |
| | | `pcsi` | 0.1294 | −2.63 % | −20.02 % | 4.6 |
| **8192** (64 N) | 385 | `cg` | 0.1014 | — | — | 6.4 |
| | | `cg2` | 0.0988 | −2.56 % | −4.09 % | 6.3 |
| | | `oati` | 0.0974 | −3.94 % | −23.46 % | 5.1 |
| | | `pcsi` | 0.0973 | **−4.04 %** | −20.54 % | 5.3 |

**dars is a win too, not the +1.25 % regression recorded at 512 ranks** — same story as farc:
the old row sat at 6 172 v/core where SSH is 0.8 % of the step. But the win is modest (3–4 %)
because even in range dars's SSH share is only 5–6 %: it needs 36 iterations per solve, so the
solve is cheap relative to the step. **The SSH share, not the vertices/core, is what predicts
the payoff** — farc at 312 v/core has a 22.8 % share and wins 3× harder than dars at 385.

### ✅ CLASS-5 SUSPICION CLEARED (26741360) — the `d_total` numbers reproduce under a reversed leg order

**I raised this alarm and the control refutes it. The measurement stands.** farc 2048, legs
run in REVERSED order (`oati`, `cg2`, `cg`, `cg`) with the baseline repeated last:

| leg | forward (26741204) | reversed (26741360) | |
|---|--:|--:|---|
| `cg` | 0.0821 (ran 1st) | 0.0827 / 0.0830 (ran 3rd & 4th) | **last is 0.7 % SLOWER, not faster** |
| `cg2` | 0.0739 | 0.0745 | |
| `oati` | 0.0712 | 0.0714 | |

| leg | `d_total` forward | `d_total` reversed | agreement |
|---|--:|--:|--:|
| `cg2` | −9.99 % | −9.92 % | **0.07 pp** |
| `oati` | −13.28 % | −13.66 % | **0.39 pp** |

Three things follow. (i) There is **no first-leg warm-up penalty** — the baseline is if
anything marginally *slower* when it runs last, the opposite of the hypothesis. (ii) The two
repeated `cg` legs agree to 0.4 %, so within-allocation reproducibility is good. (iii) The
whole-step ratios reproduce to **0.07–0.39 pp** under a completely reversed order, so
`d_total` is a real measurement, not an ordering artefact. **`d_total` may be quoted.**

**What remains true is the accounting gap itself, and it is a real effect rather than a bug:**
the variants speed the step up by *more* than the SSH timer accounts for (~2.4 ms at farc
2048, ~2.5 ms at dars 8192). It reproduces under order reversal, so it is not noise. The
likely mechanism is that blocking allreduces act as synchronisation points that propagate
load imbalance into neighbouring phases; halving them lets the rest of the step absorb jitter
that the SSH timer never sees. **Measured, reproducible, not yet mechanistically attributed** —
an instrumented run (phase timers either side of the solve) would settle it. Recorded as an
open question, not as a claim.

*(This is why `d_total` consistently exceeds `d_SSH × SSH%`: the relation in the table footer
is a lower bound on the benefit, not an identity.)*

### The evidence that raised the suspicion (superseded by the control above)

**The suspicion that motivated the control.** Across every CPU
rung, the whole-step saving exceeds what the solver's own timer accounts for, in the same
direction, by a similar fraction of the step:

| rung | step | unaccounted (`cg2`/`oati`/`pcsi`) | as % of step | as % of the claimed win |
|---|--:|--:|--:|--:|
| CORE2 432 | 66.2 ms | 0.49 / 0.70 / 0.45 ms | 0.7–1.1 % | 21 / 21 / 12 % |
| CORE2 864 | 43.5 ms | 0.59 / 0.90 / 0.88 ms | 1.4–2.1 % | 18 / 18 / 15 % |
| farc 1024 | 139.2 ms | 2.14 / 2.02 / 2.04 ms | 1.5 % | 36 / 26 / 39 % |
| farc 2048 | 82.1 ms | 2.49 / 2.36 / 2.35 ms | 2.9 % | 30 / 22 / 24 % |
| dars 6144 | 132.9 ms | 1.90 / 1.97 / 2.01 ms | 1.5 % | **70 / 49 / 57 %** |
| dars 8192 | 101.4 ms | 2.33 / 2.48 / 2.77 ms | 2.5 % | **90 / 62 / 68 %** |

On dars 8192 the solver accounts for **0.27 ms of `cg2`'s 2.60 ms** whole-step saving — 90 %
of that headline number is unexplained. The offset is present for three different solvers with
very different `d_SSH`, which rules out a solver-specific cause and points at the *protocol*:
the baseline leg always runs FIRST, and its rep-to-rep spread is consistently wider (1.3 % vs
the variants' 0.4 %), which is what a first-leg warm-up looks like.

**It is not simply "later is faster", though** — the old out-of-range `lad_dars_c512` row has
the variants *slower* by 9.76 ms beyond what the solver explains. At a 1.546 s step that is
0.6 %, i.e. within run-to-run drift. So the honest description is **a ~1–3 % systematic
uncertainty on `d_total` that the protocol does not currently control**, not a proven warm-up
bias.

**→ Resolved by 26741360 above: the offset is NOT an ordering artefact, `d_total` reproduces
to 0.07–0.39 pp under a reversed leg order, and the campaign's whole-step numbers — including
CORE2 512's `pcsi` −6.30 % — stand as measured.** The gap between `d_total` and
`d_SSH × SSH%` is a real secondary benefit of removing synchronisation points, not
double-counting.

### The in-model trace that misled (job 26740651) — the first 107 iterations only

farc np32, `cg2`, solve 37 (`labdumps/farc_np32_fb/step0037`), rtol = **4.3743e-01**:

| iteration | residual | comment |
|--:|--:|---|
| 1 | 8.268e+03 | |
| 20 | 2.995e+02 | converging cleanly, 2–4 % per iteration |
| 60 | 3.975e+01 | |
| **87** | **1.582e+01** | ⬅ **best residual ever reached** |
| 88–96 | 1.623 → 1.825e+01 | residual **GROWS** |
| 97–106 | oscillates 1.59–1.66e+01 | never beats it-87 again |
| 107 | 1.621e+01 | guard fires: 20 consecutive non-improving steps |

Read on its own this looks decisive: the residual turns around at iteration 87 and sits at
~37× rtol while baseline `cg` reaches 5.1488e-01 by iteration 200. **It is not decisive, and
I drew the wrong conclusion from it.** The window shown is simply too short — three iterations
past the last row the residual resumes falling, and the solve converges at 205. See the
resolution above.

### ⭐⭐ The aggregates — `cg2` holds iteration parity with `cg` on farc, and the false positive is rare

The same job's aggregates, 60 solves per leg, farc np32 dt900 (`fallbacks=` harvested per the
process fix):

| leg | solves | iters/solve | exch/solve | ar_blk/solve | fallbacks |
|---|--:|--:|--:|--:|--:|
| `cg2` | 60 | **218.85** | 224.37 | 224.37 | **1** (solve 37) |
| `cg` (baseline) | 60 | **220.83** | 443.67 | 443.67 | 0 *(structurally — see above)* |

**`cg2` matches baseline `cg`'s iteration count on farc to 0.9 %** (218.85 vs 220.83) and
halves both the exchanges and the blocking allreduces (224 vs 444), exactly as designed —
and the 218.85 is itself inflated by the one mixed solve, so the true parity is slightly
better still.

The guard fired on **1 solve in 60 = 1.7 %**. Since that firing is now known to be a false
positive (above), the correct reading of this table is: **`cg2` converged on all 60 solves**,
and on 1 of them our harness threw the answer away and redid it with `cg`.

So the ledger's earlier framing — *"on an ill-conditioned mesh they fail to converge on a
noticeable fraction of solves"* — is **wrong, not merely overstated**.

⚠️ **Rate discrepancy, unexplained:** this run gives 1/60 = 1.7 %, while the 300-step A/B
runs gave ~20/300 = 6.7 % at the same np32. The A/B runs cover later model time (300 steps vs
60), so the false-positive rate may not be stationary. Not yet measured; do not quote a single
"farc fallback rate" until it is.

*(The pre-registered criterion is 0 firings, so the retracted A/B timings stay retracted
regardless — 1 fallback still makes a leg a mixture.)*

### The guard is a heuristic, and it fires mid-solve

```c
if (resid < best * 0.999) { best = resid; stall = 0; }
else if (++stall >= STALL_WINDOW || resid > 1e3 * best)  ->  SSH_FB_STALL
```

`STALL_WINDOW` = 20 (`cg2`/`pipecg`) or 10 (`pcsi`/`oati`). So the trigger is **N consecutive
steps without a 0.1 % residual drop** — not divergence, not a residual increase. That is a
*heuristic*, and on farc it fires at ~112 iterations of a solve baseline `cg` completes in
212, which is why "the guard is aborting a converging solve" was the natural first
hypothesis — **and it is exactly what happens** (26741060). Note what the trigger does NOT
require: no residual increase, no NaN, no loss of positivity. Twenty quiet iterations are
enough, and CG on a κ = 843 system produces twenty quiet iterations as a matter of course.

### ⭐ MEASURED — iteration count alone does NOT produce stalls (login, `core2_np1/step0020`)

The handoff's hypothesis was "the σ recurrence accumulates rounding with iteration count, and
farc needs 212 iterations against CORE2's 106". The lab tests that directly: `--tol` forces
the *same* CORE2 matrix to run to any iteration count.

| `--tol` | `cg2` iterations | longest plateau (guard threshold 20) |
|---|--:|--:|
| 1e-5 (production) | 117 | **2** |
| 1e-7 | 174 | **2** |
| 1e-9 | 226 | **2** |
| 1e-11 | **277** | **2** |

**At 277 iterations — more than farc's 212 — `cg2` on CORE2 still plateaus for only 2
iterations, a 10× margin against the guard.** Baseline `cg` on the same system: longest
plateau 3. Iteration count is therefore *not* the driver, and the σ-drift-accumulates story
does not survive its first test. This is consistent with the §0.4b table, where the
*symmetrised* drift is flat at ~1e-13…1e-14 from iteration 2 to 60 rather than growing.

**So the cause is a property of the farc MATRIX, not of how long the solve runs.** Both
candidate explanations that were on the table — σ drift accumulating with iteration count,
and the guard aborting a healthy solve — are now falsified. What remains is farc-specific
conditioning/structure: whether the σ recurrence drifts on *this* matrix (the `--sigma-drift`
leg), and whether the symmetrised preconditioner `D^{−1/2}CD^{−1/2}` is itself poorly behaved
on a strongly variable-resolution mesh, where the diagonal `d` spans a far wider dynamic
range than on CORE2 and the `sqrt(d_i/d_j)` scaling is correspondingly extreme.

*(⚠️ `--tol` is plumbed to the variant paths only, not to baseline `cg`, whose rtol comes from
the dump's `soltol` — the four `cg` rows of that sweep are the same run and only the `cg2`
column carries information.)*

Tool: `scripts/m10_stall_analysis.py` replays the guard against any `[ssh-trace]` residual
history and reports the longest plateau plus whether the guard would fire.

### What is still open, and how it gets settled

Whatever kills `cg2`/`oati` on farc is a property of **that matrix**, not of the iteration
count. Two runs are in flight:

- **`jobs/job_m10_farcdump`** — farc np32 (a confirmed-failing configuration: 20 `cg2` / 21
  `oati` fallbacks). ⭐ **The dump exploits a dispatch accident**: `fesom_ssh_solve_cg_kk`
  dispatches to the variant at **:3928** and *returns* on success, while the
  `FESOM_SSH_DUMP` block sits at **:4061** — so under a variant the dump fires **only on
  solves that fell back** (rc<0 falls through to the baseline body). Leg 1 therefore captures
  exactly the failing systems and nothing else. Solve numbering stays correct because
  `ssh_solve_cg2` returns −1 *without* calling `ssh_wire_close_solve`, so the retry's
  `solves+1` is still this solve's index. Leg 2 dumps ordinary systems as the control. Both
  legs trace per-iteration residuals.
- **`jobs/job_m10_lab`** — the offline battery (`--sym-check`, `--sigma-drift`, and a
  solver × tolerance plateau ladder) at the dump's own np.

- **New knob `FESOM_SSH_STALL_WINDOW`** (default = the compiled-in per-solver value, so unset
  is byte-identical) makes the heuristic tunable. It is the only way to answer *"would the
  variant have converged if the guard had not aborted it?"* — the question the whole farc
  verdict rests on. Diagnostic setting, not a production one; it announces itself on rank 0
  (L80).

## ⭐⭐⭐ WHERE THE TIME ACTUALLY GOES — phase-resolved busy/wait (26742297, 26742298)

**User question (2026-08-06):** *"SSH was supposed to be the main scaling problem, and a 20 %
reduction should have helped a lot, but it did not. Where do we actually spend time?"* The
question is fair and the answer needed measurement, not the Amdahl arithmetic alone.

### First, the arithmetic — the solvers capture most of what is available

CORE2 CPU baseline ladder (`FESOM_SPEED=1`), splitting each step into the SSH solve and
everything else:

| ranks | step ms | SSH ms | rest ms | SSH speed-up | rest speed-up | ideal |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | 195.50 | 4.89 | 190.61 | 1.00× | 1.00× | 1.0× |
| 256 | 105.00 | 4.62 | 100.38 | 1.06× | 1.90× | 2.0× |
| 432 | 66.20 | 6.22 | 59.98 | 0.79× | 3.18× | 3.4× |
| 512 | 58.70 | 5.81 | 52.89 | 0.84× | 3.60× | 4.0× |
| 864 | 43.50 | 8.22 | 35.28 | **0.59×** | **5.40×** | 6.8× |

**The SSH solve is the only part of the model that ANTI-SCALES** — from 128 to 864 ranks
everything else speeds up 5.40× (79 % efficiency) while the solve gets **41 % slower in
absolute terms**. The premise of the track is confirmed. But the solve is still only 18.9 % of
the step at 864 ranks, so Amdahl caps any SSH work there at −18.9 %; we measure −13.10 %, i.e.
**69 % of the ceiling**, and the captured fraction rises monotonically with rank count
(29 / 39 / 61 / 64 / 69 % at 128 / 256 / 432 / 512 / 864). Nothing is anomalous: the lever is
working near its limit, and the limit is what is small.

### Then the measurement — half the step is MPI wait

`FESOM_SPEED_PHASESTATS=1` (per-rank per-phase wall, wait measured *inside* MPI by PMPI
interposition, so `busy = wall − wait` is structurally complete). 115 timed steps.
**The phase TOTAL reconciles with the loop timer to 0.1 ms on both configurations**
(48.2 vs 0.0482 s/step; 84.7 vs 0.0846), so the attribution is trustworthy.

| | CORE2 864 | farc 2048 |
|---|--:|--:|
| step | 48.2 ms | 84.7 ms |
| **MPI wait** | **24.7 ms = 51 %** | **44.6 ms = 53 %** |
| busy | 23.5 ms = 49 % | 40.1 ms = 47 % |
| largest busy | `ocean` 13.0 | `ocean` 26.6 |
| largest wait | `cg` 8.9 | `cg` 20.4 |
| SSH solve, busy + wait | 9.9 ms = 21 % | 22.9 ms = 27 % |
| — of which WAIT | 90 % | 89 % |

**The model is synchronisation-bound at these rank counts, and the SSH solve is the single
largest waiter — but ~90 % of the solve's cost is waiting, not computing** (its own arithmetic
is 1.0 ms on CORE2, 2.5 ms on farc).

### ⭐⭐ The saving is ENTIRELY wait, and 39 % of it lands in OTHER phases

CORE2 864, baseline vs `pcsi`, same allocation:

| phase | busy | wait | Δwait | MPI calls/step |
|---|--:|--:|--:|--:|
| `force` | 6.9 → 6.5 | 3.4 → 1.5 | **−1.9** | 11 → 11 |
| `ice` | 0.2 → 0.2 | 0.9 → 0.6 | −0.3 | 9 → 9 |
| `icedyn` | 1.6 → 1.6 | 4.8 → 3.6 | **−1.2** | 240 → 240 |
| `ocean` | 13.0 → 13.0 | 6.5 → 6.1 | −0.4 | 68 → 68 |
| `cg` | 1.0 → 1.0 | 8.9 → 3.1 | **−5.8** | 360.8 → 170.2 |
| **TOTAL** | **23.5 → 23.1** | **24.7 → 15.2** | **−9.5** | 709.8 → 519.2 |

Busy is unchanged (−0.4 ms): the variants do the same arithmetic. **All of the benefit is
wait, and only 5.8 of the 9.5 ms is in the solver — 3.7 ms (39 %) appears in phases whose MPI
call counts and busy times are IDENTICAL.** Removing synchronisation points lets neighbouring
phases stop absorbing skew.

**This closes the open question flagged earlier** (`d_total` exceeding `d_SSH × SSH%` by a
reproducible ~2.4 ms). The mechanism is measured, not inferred: fewer collectives ⇒ less
imbalance collected elsewhere. `d_total ≈ d_SSH × SSH%` is a lower bound on the benefit.

### 🔴 The bigger target is LOAD IMBALANCE in the ocean phase, not the solver

`ocean` busy across ranks — this is compute time, not communication:

| config | min | mean | max | spread |
|---|--:|--:|--:|--:|
| CORE2 864 | 3.8 | 13.0 | 18.2 | **4.8×** |
| farc 2048 | 8.8 | 26.6 | 42.5 | **4.8×** |

A bulk-synchronous phase costs `max`, not `mean`, so the imbalance tax in `ocean` alone is
**5.2 ms = 11 % of the CORE2 step and 15.9 ms = 19 % of the farc step** — comparable to the
entire SSH solve (21 % / 27 %) and larger than anything the solvers can win. It also
compounds: that skew must be absorbed at the next collective, which is the SSH solve's first
allreduce, so **part of `cg`'s wait is the ocean phase's imbalance arriving there, not the
solver's own latency.** Consistent with the halving of calls (2.12×) reducing `cg` wait by
more than proportionally (2.87×).

**Implication for the track:** the solvers are worth having and their numbers stand, but the
step's largest single defect is that one rank does 4.8× another's ocean work. Partitioning
work would attack the 15.0 ms of `ocean` wait *and* part of the solver's wait.

⚠️ **Caveats.** PMPI interposition inflates wait in proportion to call count, so these runs
overstate the solver benefit (−20.5 % here vs the −13.10 % A/B of record); trust the ratios,
the busy figures and the attribution, not the absolute deltas. 115 timed steps, two
configurations. The imbalance figure needs confirming at the production 512-rank point before
any re-partitioning is justified by it. The METIS partitions already weight by 2-D and 3-D
node counts, so part of this spread may be irreducible (ice cover, convection) — that is
untested.

*(Not pursued: `--sigma-drift` on the farc failing system, jobs 26740825/26741061, timed out
at 20 min and 1 h — the host-side reference PCG at np32 is too slow on a 638 387-node mesh.
The guard finding explains the failures, so the σ-drift question is now a mechanism curiosity
rather than a blocker.)*

## Frozen binaries

*(`/work/ab0995/a270088/port2/m10/bin/` + sha256 here; binaries NEVER in git.)*

| name | md5 | contents |
|---|---|---|
| `stallknob_serial` | `0712ee57fe7f2f5dbb8738f4463653b2` | Serial `fesom_port`, `FESOM_SSH_STALL_WINDOW` commit |
| `stallknob_lab` | `f425461f0b6e7c7ab49f3b38f38afe63` | Serial `fesom_ssh_lab`, same commit |
| `stallknob_cuda` | `09d26e91139048512529dce16467afc1` | CUDA `fesom_port`, same commit |

## 🔴 LOAD IMBALANCE — what it is, and why fixing it makes the step SLOWER

**User question (2026-08-06):** *"what is in this load imbalance, do we know? why we have an
imbalance in compute?"* Plus two prior negative results from the user: the ice hypothesis was
tested and failed, and 3D balancing was tried and failed.

### It is bathymetry, not ice — measured (26742298 + partition analysis)

Per-rank `ocean` busy against per-rank mesh content, farc 2048 (2048 ranks, all correlations
over all ranks):

| correlation with `ocean` busy | r |
|---|--:|
| **3D nodes owned** (Σ `nlvls` over owned columns) | **0.967** |
| 2D nodes owned | 0.003 |
| `cg` **wait** vs 3D nodes | **−0.771** |

Fit: `ocean_busy = 2.241 ms per 1000 3D nodes + 10.22 ms`, which reproduces the measured
8.8→42.5 ms range. **The partition balances SURFACE nodes (310–313 per rank, 1.01×) while the
work is per WATER COLUMN, and 3D nodes span 9.40×** (1550 → 14571). On fArc that is shallow
Siberian shelf against deep Arctic basin.

**This explains the user's negative ice result.** `cg` wait correlates **−0.771** with 3D
nodes: deep-column ranks wait *least* because they are the stragglers. The ranks everyone
waits for are the DEEP ones, not the icy ones.

*(Parse verified before use — the M7 `rpart.out` misreading is the precedent. `my_list` owned
sets form a perfect disjoint cover: 638387 nodes each owned exactly once, 3D sum equal to the
mesh total exactly. The r = 0.967 against measured busy is independent confirmation.)*

### Why the earlier 3D-balancing test could not have worked

| partition | 2D bal | 3D bal |
|---|--:|--:|
| **NG5 `dist_2048`** (/pool) — what M7 verified | 1.02× | **1.04×** already dual-weighted |
| `dars_bigpart`, `ng5_bigpart` (ours) | 1.03× | 1.05× |
| **fArc `dist_2048`** (/pool) | 1.01× | **9.40×** |
| **CORE2 `dist_864`** | 1.06× | **9.60×** |

The M7 record is correct *for NG5* — and that is the problem. Its regenerated `ng5_w3d` dists
came out **byte-identical to /pool** because NG5 was already dual-weighted, so that A/B
compared a partition against itself: an L80 dead-knob null, structurally incapable of showing
an effect. On CORE2 and fArc, where the partition IS imbalanced, the lever was untested.

### The systematic CORE2 test (26743709 / 26743820 / 26743981 / 26744115)

The weighting is a COMPILE-TIME `#ifdef PART_WEIGHTED` (`fort_part.c`), so 2D-only and dual
could only be compared across two binaries. A copy of the partitioner at
`/work/.../partw/fesom2` was patched to read **`FESOM_PART_WGT`** at runtime (0 = 2D only,
1 = 3D only, 2 = dual) — one binary, no build confound. *(The user's tree is untouched:
verified by md5 + `find -newermt`.)* `wgt_type=1` is legacy dead code and aborts inside METIS
(rc=134); the question is answered by 0 vs 2.

🔴 **The first attempt (26743391) was itself a dead-knob null** — `rsync` preserved the
prebuilt binary's mtime and the copied CMake cache held absolute paths to the original tree,
so `cmake --build` reported "Built target" without recompiling. All three legs produced
IDENTICAL edgecuts under three different knob values. The script now ABORTS unless the
override announces itself.

**A/B, same allocation, same binary, 864 ranks, min-of-2, 300 steps (26743981).** All three
arms verified byte-identical on the 8 mesh-definition files, so only `dist_*` differs:

| arm | 3D balance | s/step | vs shipped |
|---|--:|--:|--:|
| `core2` (shipped) | 9.60× | 0.0440 | — |
| `core2_wgt0` (2D only) | 9.32× | 0.0443 | **+0.68 %** ← partitioner-source control: clean |
| `core2_wgt2` (dual) | **1.05×** | 0.0482 | **+9.55 % SLOWER** |

`wgt0` reproduces the shipped partition's imbalance, confirming the shipped CORE2 partitions
are 2D-only, and the source control is clean (+0.68 %), so the +9.55 % is the pure weighting
effect.

### ⭐ WHY it loses — the balancing works and still costs more than it saves (26744115)

Phase-resolved, 864 ranks (`FESOM_SPEED_PHASESTATS=1`; ⚠️ needs `FESOM_SPEED_FORCE_SERIAL=1`
on a Serial build — `fesom_speed.hpp:111-113`, "Serial stays legacy" — without it the lever
silently resolves OFF, which cost two runs):

| | `wgt0` (9.32×) | `wgt2` (1.05×) | Δ |
|---|--:|--:|--:|
| `ocean` busy min/mean/max | 3.8 / 13.0 / 19.4 | 10.1 / 14.7 / 17.5 | spread 5.1× → **1.73×** |
| `ocean` imbalance tax (max−mean) | 6.4 ms | 2.8 ms | **−3.6** ✅ |
| `ocean` wait | 6.7 | 4.9 | **−1.8** ✅ |
| **total busy** | 22.9 | 25.6 | **+2.7** ❌ |
| `cg` wait | 9.0 | 10.2 | +1.2 ❌ |
| `icedyn` wait | 4.0 | 5.1 | +1.1 ❌ |
| **total** | 45.2 | 48.1 | **+2.9 ms** |

**The balancing did exactly what it was supposed to** — idle ranks took on real work (`ocean`
busy min 3.8 → 10.1), the spread collapsed, and `ocean` wait fell 1.8 ms. **But the
dual-weighted partition needs a 40 % bigger halo** (42 → 59 nodes/rank), and that costs
**+2.7 ms of COMPUTE** plus +1.2/+1.1 ms of wait in the comm-heavy `cg` and `icedyn` phases.

**Conclusion: the user's "3D balancing did not work out" is CONFIRMED, and now has a measured
mechanism rather than a null result.** The imbalance is real and worth 6.4 ms, but METIS
cannot balance the vertical without cutting more edges, and the halo penalty exceeds the
imbalance gain. `Repartitioning OUT` stands as a user decision — now for a stated reason.

**What is NOT ruled out:** `wgt_type=2` already softens the 3D criterion by `+100` per node.
The trade is a continuum between edgecut and vertical balance, and only its two endpoints have
been measured. A softened 3D weight (or edge-weighted METIS) might find an interior optimum.
That is a partitioner study, not a model change, and it is unexplored.

### ⭐⭐⭐ CORRECTION — the imbalance IS recoverable, below ~250 vertices/core (26744554-57)

The 864-rank result above is real but **not general**. The halo penalty of dual weighting is a
constant **+40 % at every rank count**, but its weight in the step is not — halo rises from
**9 % of owned nodes at 128 ranks to 29 % at 864** — while the imbalance removed stays roughly
constant (3D max/mean 1.48→1.53 across the ladder). So the trade must change sign, and it does:

| ranks | vertices/core | `wgt0` (2D-only) | `wgt2` (dual) | dual vs 2D-only |
|--:|--:|--:|--:|--:|
| 128 | 991 | ⚠️ diverged | 0.1981 | — |
| **256** | **495** | 0.1060 | **0.1011** | **−4.62 %** ✅ |
| 512 | 248 | 0.0593 | 0.0593 | **0.00 %** ← crossover |
| 864 | 146 | 0.0448 | 0.0487 | **+8.71 %** ❌ |

Min-of-2, one allocation per rung; rep-to-rep spread ≤0.3 %, and the 256-rank dual leg
reproduced 0.1011/0.1011 exactly, so −4.62 % is far outside noise.

**⇒ "repartitioning is OUT" is true only past the crossover.** At **495 vertices/core — inside
FESOM's own 300–500 operating guidance — dual weighting is worth −4.6 %**, for no model change
at all. CORE2's production 512-rank point sits essentially ON the crossover (0.00 %). Past it
the halo penalty dominates and the lever inverts.

⚠️ **A regenerated 2D-only `dist_128` made baseline CG diverge at iteration 1**
(`CG_kk abort at iter 1: residual=5.49698e+45`) where the shipped and dual-weighted partitions
at the same rank count both ran clean — L99 (instability is partition-marginal) in the wild.
**Any newly generated decomposition needs a stability check before it is used.**

### 🔴 GPU: the SAME imbalance exists, and dual weighting is much WORSE there (26745200)

The imbalance is a property of the partition, not the backend, so GPU runs inherit it exactly.
Fixing it the same way, however, fails harder. CORE2 GPU **2 nodes / 8 ranks** (15857
vertices/rank), same allocation, same binary, min-of-2 (reps tight: 0.0807/0.0810):

| arm | s/step | vs shipped | `ocean` busy min/mean/max | `ocean` wait |
|---|--:|--:|--:|--:|
| shipped | 0.0622 | — | 12.3 / 16.1 / 20.3 | 8.8 |
| `wgt0` (2D-only) | 0.0648 | +4.18 % | 12.6 / **16.3** / 19.8 | 7.8 |
| `wgt2` (dual) | 0.0807 | **+29.74 %** | 18.2 / **19.6** / 22.8 | 11.3 |

**+29.7 % on GPU against +9.6 % on CPU**, and the mechanism is in the *busy* column: ocean
compute MEAN rises 16.3 → 19.6 ms (**+20 %**) where the halo-node arithmetic predicted +0.7 %.

**The reason is partition FRAGMENTATION, which the halo-node count understates:**

| CORE2 ranks | edgecut 2D-only | edgecut dual | ratio |
|--:|--:|--:|--:|
| 8 | 1 335 | 120 883 | **91×** |
| 16 | 2 549 | 217 791 | 85× |
| 32 | 4 307 | 375 211 | 87× |

Halo *nodes* grow ~40 %, but cut *edges* grow ~90×. That is a locality collapse, and it costs
compute on both backends (+13 % CPU, +20 % GPU) — the GPU more, since it is far more sensitive
to memory-access patterns. Note too that even the regenerated **2D-only** partition is +4.18 %
against the shipped one on GPU (against +0.68 % on CPU): **GPU punishes partition quality
generally**, so any regenerated decomposition needs measuring there before use.

**Answer to "do GPUs have the same imbalance and can we fix it?" — yes and no.** The imbalance
is present and identical in origin, but vertex weighting is not the fix on GPU: it trades a
~1.2–1.4× compute spread for a ~90× worse cut, and loses by 30 %. The CPU crossover (a win
below ~250 vertices/core) has **no GPU counterpart** in the range tested.

## ✅ NG5 CPU RESOLVED — a documented physics limit, not an M10 defect (26745567/26745747/26746161)

**The handoff's open item 5 ("NG5 CPU ≥4096 ranks: the BASELINE leg fails (nan)") is wrong in
three ways.** Measured:

| ranks | v/core | outcome |
|--:|--:|---|
| 256 | 28 918 | **OOM** — far too few ranks for 7.4 M nodes, expected |
| 1024 | 7 229 | **all 8 legs complete** |
| 2048 | 3 614 | **all 8 legs complete** (baseline 1.1909 s/step, SSH 1.6 %) |
| 4096 | 1 807 | blow-up mid-run |
| 8192 | 903 | blow-up mid-run |

So (a) it is not confined to the baseline leg, (b) 1024/2048 were never broken — their "no
valid row" was the A/B *summary table* failing to render while the per-rep numbers were
present, and (c) the failure is not at step 2.

**What actually happens.** The model's own guard fires:
`[fesom_port] BLOWUP at step 175 (uv=5.756e+00 eta=2.640e+00 …) — aborting all ranks`,
exit code 99. `print_every=999` had hidden this: with 300 steps it prints step 1 and step 300,
so "last printed step = 1" meant *never reached 300*, not *died at step 2*.

**And the model warned, on stderr, in every NG5 run:**

```
[wsplit] velocity splitter is OFF (FESOM_WSPLIT unset).
[wsplit] Production high-resolution setups run it ON; without it the
[wsplit] model is prone to vertical-CFL blow-up (Fortran NG5 dt180
[wsplit] cold start dies without it, completes with it).
```

**No M10 job ever set `FESOM_WSPLIT`** (user caught this). But wsplit alone does not rescue
*this* configuration — 300-step legs at 4096 ranks, one allocation:

| leg | blow-up |
|---|--:|
| default (`opt_visc=7`, no wsplit) | step **175** |
| `+FESOM_WSPLIT=1` | step **150** |
| `+FESOM_VISC_OPT=5` | step **175** |

This matches M7 exactly: *"Fortran NG5 dist_4096 dt180 cold **wsplit-ON** opt_visc=7 (26365809)
**DIES AT STEP 203**"* against *"the port's visc-7 death @200"* — same rank, same cell, same
endgame, with a few steps of roundoff-seeded onset scatter. **Fortran and the port both die
here.** M7's only 300-step-clean recipe was `visc-5 + easybsreturn=1.5 + wsplit`, and
easybsreturn 1.5 is excluded by the user's 2026-07-19 decision (stays 1.0), as is `opt_visc=5`
on high-resolution meshes.

**⇒ Consequences for M10.** NG5 at dt180 cold **cannot complete the 300-step protocol** at
≥4096 ranks, in this port or in Fortran. The solver A/B must therefore either run inside the
stable arc (**≤120 steps**, comfortably below the earliest observed blow-up at 150) or use a
smaller dt. This also applies to the new 16384/20480/24576 partitions — running them at 300
steps would reproduce the blow-up three more times.

**Set `FESOM_WSPLIT=1` on high-resolution runs regardless** — it is production practice and
the model asks for it; it simply is not sufficient here.

**Also explains the "solver-ordered" survival** (`pcsi` cleared 300 steps at 4096; `oati` and
`pcsi` at 8192): a marginal configuration tipped over by solver-dependent rounding. That is
L99, and it was never evidence about the solvers.

## 🔴 M13 det-IC rollout inside M10 (2026-08-14) — audit, guard fix, re-runs

The M13 session root-caused the "CG blows up on specific partitions" family to
**partition-dependent initial conditions** from the climatology hole fill (`extrap_nod3D`,
a literal port of upstream `gen_support.F90:400-507`): first-fill-wins Gauss-Seidel run to
local exhaustion between halo exchanges, so the fill depends on the decomposition. On unlucky
partitions it lays a ~10 kg/m³ density front across one element in a marginal strait
(Marmara, Gibraltar), giving a constant 5.5e-3 m/s² pressure-gradient force and a linear |uv|
ramp from step 1 into CFL death. Fix: `FESOM_IC_EXTRAP=det` (ring fill + Jacobi relaxation,
gid-ordered sums, default OFF). Evidence: `~/port_kokkos/docs/CG_BLOWUPS_M13.md`; rollout
decisions and the per-track re-run plan: `docs/plans/20260815-m13-det-rollout.md` (main
checkout). Cherry-picked here as `ddae127` + `e7ff7ae`.

### What this can and cannot change on the M10 board

A solver A/B runs its four legs back-to-back in ONE allocation on ONE partition, so all legs
integrate the SAME initial condition — the published **ratios stay internally consistent** and
no quoted row flips sign because the IC was wrong. What the artifact does change is (i) rows
that could not be measured at all, and (ii) the SSH share and iteration count, which are the
campaign's own explanatory variables. Note §5b of the M13 doc: under legacy fill **no NG5
partition was ever clean** (even "working" dist_16384 crests |uv| 4.86 m/s at step 95), so
every NG5 number in this ledger was measured on an artifact-driven transient.

### ⚠️ The zombie audit — 4 of 77 A/B runs, none of them quoted

The M13 rollout plan flags our stall guard as NaN-blind. Swept all 77 run directories under
`m10/ab` for legs whose final state row is the all-NaN signature (`it=0`, `uv=0`,
`T[1e+30,-1e+30]`):

| run | leg | status |
|---|---|---|
| `lad2_ng5_c4096_26738530` | `pcsi` | ZOMBIE (legs 0-2 died at step 1) |
| `lad2_ng5_c8192_26738531` | `oati`, `pcsi` | ZOMBIE (legs 0-1 died at step 1) |
| `lad2_farc_c1024_26738526` | `pcsi` | ZOMBIE |
| `lad2_farc_c2048_26738527` | `oati` | ZOMBIE |

All four runs were already VOID for other reasons and **none of the 28 quoted configurations
contains a zombie leg** — the board survives the audit. But this **RETRACTS** the closing note
of the NG5 CPU section above ("solver-ordered survival … a marginal configuration tipped over
by solver-dependent rounding, that is L99"). `pcsi` at 4096 and `oati`/`pcsi` at 8192 did not
survive anything: they completed as NaN zombies. There was no solver ordering to explain.

### How exposed is the rest of the board? — the step-1 |uv| fingerprint

The artifact announces itself as an anomalous step-1 global |uv| (the constant strait force
acts from the first step). Swept the baseline leg of every archived run:

| mesh | typical step-1 |uv| | worst run |
|---|--:|---|
| CORE2 | 0.393 | 0.393 (uniform across all rungs) |
| farc | 0.321–0.378 | 0.378 (`lad_farc_c32`) |
| dars | 0.254–0.334 | 0.334 (`sharecpu_dars_c2n`) |
| NG5 | 0.083–0.354 | **0.678** (`lad2_ng5_c8192`) |

Only NG5 `dist_8192` stands out — and its 0.678 m/s is *exactly* the value M13 measured for
that partition's Marmara element after step 1 (their §4 table), so the archived M10 log
carries the fingerprint independently. It is also one of the voided runs. For reference the
core2_wgt0 partition of the guard proof reads 3.53. The quoted rows are unremarkable on this
marker, which is consistent with the det re-runs: use them to confirm, not to assume.

### The mechanism in this branch's code, and the fix

`fesom_ssh.cpp` — the three variants gate their iteration loop on `if (resid >= rtol)`
(cg2 :2834, pcsi :3540, oati :3764). **`NaN >= rtol` is false**, so a solve entered with a
non-finite state skips the loop entirely and returns `fb = SSH_FB_NONE` with `iter = 0`: a
"converged" solve at zero cost. Baseline `cg` has no such hole — it dies on
`residual != residual || residual > 1e30` (:4215), which is why stock CG is the only scheme
that fails loudly. Fixed by testing the same criterion at the variants' entry (all three),
so a non-finite entry residual becomes `SSH_FB_NAN` → announce → fall back to `cg` → die.
Byte-inert on finite residuals: **knob-off byte gate PASS, job 26961453** (bit-identical to
the certified M6 baseline, `diff_snap rc=0`).

### ⭐ Measured before/after — job 26961492 (1 node, 128 ranks, 300 steps)

Vehicle: `/work/.../mesh/core2_wgt0/dist_128`, the regenerated CORE2 partition that the
load-balance study called "simply does not work" (L103 amendment). Step-1 |uv| = 3.53 m/s
where a healthy CORE2 cold start sits near 0.3 — the M13 signature, on ONE node instead of
NG5's 64. `FESOM_PRINT_EVERY=999` throughout (the model's |uv|>5 guard only runs at print
cadence; a lower cadence aborts before the state can reach NaN and the zombie cannot form).

| leg | bin | IC | solver | outcome |
|---|---|---|---|---|
| A | det1 | legacy | `cg` | rc=1, dies loudly — the artifact |
| B | det1 | legacy | `oati` | rc=1 **after announcing a FALLBACK** (`solve 6 after 0 iters, res=4.37e+46`) — the fix |
| C | stallknob (campaign bin) | legacy | `oati` | rc=0, "300 steps", `it=0`, `T[1e+30,-1e+30]`, **0.1838 s/step** — the zombie |
| D | det1 | det | `cg` | rc=0, 300 steps, healthy (uv 1.47, T[-2.04,30.63]), **0.2060 s/step** |
| E | det1 | det | `oati` | rc=0, 300 steps, **0.2055 s/step**, 0 fallbacks, state row identical to D at print precision |

⭐ **A zombie leg is 10.8 % FASTER than the healthy run** (0.1838 vs 0.2060 s/step) — the exact
size of a plausible "win". This is why the guard fix had to land before any new bin was used
for timing. ⭐⭐ **det rescues the partition**: leg D completes 300 clean steps on the
decomposition that "does not work", which retires the L103-amendment reading for this case —
the partition was never the problem, the fill was.

### ✅ The det path re-proved on THIS branch's binary (login node, free)

M13's np-independence proof was run with their bin; `det1_serial` is a different build (m10
tip + det + guard fix), so it was re-proved here: pi mesh, PHC IC, `FESOM_EVP_DUMP_DIR`
surface T/S per gid, np ∈ {1, 2, 8}. **3140 of 3140 surface nodes bitwise identical at 17
significant digits across all three decompositions** (np1-vs-np2 and np1-vs-np8, 0 differing).
Under legacy fill the same comparison differs by up to 1.33 °C/PSU at 147 nodes.

### ❌ The farc stall plateau is NOT an IC artifact (26961498, 26961508)

Pre-registered question: farc's `cg2`/`oati` false-positive stalls were blamed on a
21-iteration residual plateau of unknown origin — an artifact-driven transient was a candidate.
Measured: under `FESOM_IC_EXTRAP=det` at the DEFAULT stall window (20), the guard fires in
the same pattern as under legacy fill — 2048 ranks `cg2` **18**, `oati` **20**, `pcsi` **0**;
1024 ranks `cg2` **19** — with iteration counts unchanged (207.4 vs the legacy 210.8 class).
The plateau is intrinsic to farc's convergence profile, so `FESOM_SSH_STALL_WINDOW=200`
remains mandatory on farc under det, exactly as for the recovered legacy rows. (These two
default-window jobs, 26961498/26961508, are therefore NOT valid A/B points — they are the
control that answers the question; the valid farc rows come from 26961566/67.)

### The re-run list (protocol: pinned `det1_serial`/`det1_cuda`, 4 legs × 2 reps, 300 steps)

| job | configuration | why it is on the list |
|---|---|---|
| 26961553 | NG5 CPU 4096 r / 32 N | **new row** — every legacy attempt died or zombied |
| 26961554 | NG5 CPU 8192 r / 64 N | **new row** — ditto; NG5 has no CPU point on the board |
| 26961566 | farc CPU 2048 r / 16 N | best CPU result (−13.28 %), shared with the M12 re-board |
| 26961567 | farc CPU 1024 r / 8 N | second farc rung |
| 26961555 | dars CPU 6144 r / 48 N | in-range dars row; dars fills are the F45 mechanism |

NG5 runs add `FESOM_WSPLIT=1` (production practice, and what the M13/M12 det board uses), so
they REPLACE rather than extend the old NG5 GPU row. **Not re-run:** CORE2 CPU (user rule
2026-08-14 — CPU scaling lives on farc/dars/NG5), and the in-range NG5 CPU rungs at 128–192
nodes, which belong to the M12 det re-board rather than being duplicated from here.

Frozen bins for this work (`/work/ab0995/a270088/port2/m10/bin/`):
`det1_serial` sha256 `6450a8ac…`, `det1_cuda` sha256 `bef21c9e…` (m10 tip + det + guard fix).

### ⭐⭐ RESULTS — the det re-runs, against their legacy counterparts

All five rows `fallbacks=0` on every leg, four-leg complete, min-of-2, `det1_serial` pinned.
Best variant per configuration (`oati` throughout):

| configuration | legacy `d_total` | **det `d_total`** | legacy SSH% | det SSH% | iters legacy → det |
|---|--:|--:|--:|--:|--:|
| farc CPU 2048 r | −13.28 % (26741204) | **−12.90 %** (26961566) | 22.8 | 23.3 | 211.3 → 213.1 |
| farc CPU 1024 r | −5.60 % (26741203) | **−5.29 %** (26961567) | 11.7 | 11.3 | 211.7 → 213.0 |
| dars CPU 6144 r | −3.01 % (26741040) | **−1.69 %** (26961555) | 5.6 | 4.6 | 36.62 → 36.63 |
| NG5 CPU 4096 r | *(unmeasurable)* | **−0.69 %** (26961553) | — | 1.9 | — → 76.7 |
| NG5 CPU 8192 r | *(unmeasurable)* | **−1.93 %** (26961554) | — | 3.6 | — → 76.7 |

**Nothing on the board is overturned.** The campaign's best CPU result reproduces at −12.90 %
against −13.28 % — a 0.38 pp difference eight days apart, inside the inter-allocation noise
band (never compare across weeks: the same-day rule exists for exactly this).

**dars is the one row that moves, and it moves for the reason the campaign predicts.** The
variants are unchanged (`oati` 5.41 → 5.52 ms, iterations 36.62 → 36.63, ms/iter 0.150 →
0.151); what changed is the BASELINE solve, 7.44 → 6.00 ms/step, dropping the SSH share
5.6 → 4.6 %. Less headroom, proportionally less payoff (share-scaled prediction −2.5 %,
observed −1.7 %). The solver did not get worse under a clean IC — the baseline got better.

**⭐ NG5 CPU: two rows where the board had none.** Every legacy attempt died or zombied
(§ the audit above), so the campaign has never had a NG5 CPU point. Under det both rungs run
300 clean steps with iteration parity (`cg` 77.6, `cg2` 76.2, `oati` 76.7, `pcsi` 87.4) and
no fallbacks, and they land exactly where the SSH-share law puts them: shares of 1.9 % and
3.6 % buy −0.69 % and −1.93 % of the step. NG5 CPU is a low-share, low-payoff regime — an
honest negative for the solvers, and the first one measured rather than inferred.

⚠️ **Read the NG5 phase split with care.** On both NG5 rungs the SSH-phase timer gets WORSE
(`d_SSH` +84/+42/+36 % at 4096, +53/+34/+23 % at 8192) while the whole step gets faster. That
is not a paradox and not a solver failure: ~90 % of the measured solve cost at scale IS MPI
wait (see the phase-budget section), so at a 2–4 % share the phase timer is measuring where
the wait sits, not how fast the solver is. Iteration counts — the wait-free quantity — show
parity. Do not quote `d_SSH` from a low-share configuration.
