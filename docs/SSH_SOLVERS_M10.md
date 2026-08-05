# M10 — SSH solver track: ledger of record

**Status:** ACTIVE — Task 1 (worktree + hygiene)
**Plan of record:** `docs/plans/20260805-m10-ssh-solvers.md` (in-tree on this branch)
**Branch:** `m10-ssh-solvers`, worktree `~/port_kokkos_ssh`
**Derivations doc:** `docs/plans/20260805-m10-ssh-derivations.md` (Task 4, to be created)
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
| 2026-08-06 | T5a knob-off byte gate on the dispatch + guard + sympre + cg2 commit | 26723543 | (in log) | *(pending)* |
| 2026-08-06 | T5a explicit `FESOM_SSH_SOLVER=cg` byte gate (proves the default branch is the certified path) | 26723544 | (in log) | *(pending)* |
| 2026-08-06 | T5b CUDA fidelity, `cg2` (CUDA vs Serial, same knobs) | 26723550 | (in log) | *(pending)* |
| 2026-08-06 | T5b CUDA fidelity control, `cg` | 26723551 | (in log) | *(pending)* |
| 2026-08-06 | T5b options ×3 (TKE/mEVP/zstar) under `cg2` | 26723560 | (in log) | *(pending)* |

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

## A/B perf legs (submitted 2026-08-06 — HARVEST THESE FIRST next session)

Protocol: `jobs/job_m10_ab` — all legs back-to-back in ONE allocation on the SAME nodes with
the SAME binary, min-of-2 reps, 300 steps, `-C a100_80`, NG5 dt180. Harvest reports s/step,
Δ%, iters/solve and µs/iteration.

| job | rung | legs | purpose |
|---|---|---|---|
| 26723631 | NG5 4N | legacy-cg · cg2 · pipecg | the paper's "plain PCG" reference point |
| 26723632 | NG5 16N | legacy-cg · cg2 · pipecg | ditto |
| 26723692 | NG5 4N | legacy-cg · pcsi · pcsi K=10 | pcsi + its check-interval sensitivity |
| 26723693 | NG5 16N | legacy-cg · pcsi · pcsi K=10 | ditto |
| **26723783** | **NG5 4N** | **SPEED=1 · +cg2 · +pipecg · +pcsi** | ⭐ **the number of record** |
| **26723784** | **NG5 16N** | **SPEED=1 · +cg2 · +pipecg · +pcsi** | ⭐ **the number of record** |

> ⚠️ **Measurement-design correction, made before any number was quoted.** The first four
> jobs clear every knob, so their leg 0 is **legacy `cg`** (2 exchanges + 2 allreduces per
> iteration) — *not* the production configuration. Since the M10 solvers use the CGPIPE ring
> machinery internally, comparing them against legacy `cg` would credit them with the ring
> saving that `FESOM_SPEED=1` already ships, and overstate M10's contribution. Jobs 26723783
> and 26723784 hold `FESOM_SPEED=1` on **every** leg so only the SSH solver differs — those
> are the numbers of record; the first four stand as the plain-PCG reference the plan asks
> for. (Interim leg-0 sanity: NG5 16N legacy `cg` = 0.4228/0.4241 s/step at 76.9 iters/solve.)

## Frozen binaries

*(`/work/ab0995/a270088/port2/m10/bin/` + sha256 here; binaries NEVER in git.)*
