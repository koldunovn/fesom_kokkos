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

## Frozen binaries

*(`/work/ab0995/a270088/port2/m10/bin/` + sha256 here; binaries NEVER in git.)*
