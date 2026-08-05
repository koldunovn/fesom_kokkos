# M10 — SSH solver track: communication-avoiding CG variants + P-CSI

**Date:** 2026-08-05 · **Status:** IN PROGRESS — T1–T4 complete (2026-08-06); T5a underway.
**Base:** `f42c453` (= `65a1a71` + this plan commit) — deviation documented in `docs/SSH_SOLVERS_M10.md` §Provenance.
**Branch:** `m10-ssh-solvers` in worktree `~/port_kokkos_ssh`, base `65a1a71` (= `m7-jupiter` tip = `main`)
**Source material:** `ssh_sergey/` — Sergey Danilov's `solvers.F90` + three papers (gitignored, third-party)
**Docs of record (to be created):** `docs/SSH_SOLVERS_M10.md` (ledger/findings) · `docs/plans/20260805-m10-ssh-derivations.md` (Layer-0 math + typo report — paper appendix seed)

> **🔴 New-session bootstrap:** a session started in `~/port_kokkos_ssh` gets a DIFFERENT memory
> index (M9 lesson). First actions there: read the main project memory
> `~/.claude/projects/-home-a-a270088-port-kokkos/memory/MEMORY.md`, then THIS plan, then
> `docs/SSH_SOLVERS_M10.md` once it exists.

## Overview

The SSH solve is the #1 communication site of the port (E-ledger, L100): at 16N its halo
exchanges alone cost 38.5 ms/step, and every iteration additionally executes **2 blocking
`MPI_Allreduce`** (a 1-element after the fused SpMV·dot, a 2-element after the fused
precond·dot2) — at 72–120 iters/solve that is **~150–240 blocking global syncs per step**, each
a device-fence + global latency + imbalance-absorption point (the s14 per-partner toll amplifies
them at scale). Our existing levers never touched this axis: CGPIPE cut halo *exchanges*
(2→1/iter), CGPOLY cut *iterations* (129→40 CORE2). Sergey's proposal attacks the *allreduces*.

**Goal:** implement all four communication-avoiding solvers (user decision: all four,
unconditionally), certify them at the solver-class bar (bit-identity NOT required for the new
solvers; knob-OFF stays byte-identical), measure a paper-grade Levante scaling campaign, and
produce robust figures + a documented recommendation. A GH200/JUPITER follow-up ports only the
winners.

**The four solvers** (one knob, `FESOM_SSH_SOLVER`):

| value | algorithm | per-iteration comm (ring-composed) | source of truth |
|---|---|---|---|
| `cg` (default) | current PCG — path byte-untouched | 1 exch + **2 blocking Allreduce** | existing certified code |
| `cg2` | Chronopoulos–Gear CG | 1 exch + **1 blocking 3-elem Allreduce** | Sergey `solvers.F90:3` + arXiv:1612.01395v3 Alg 4 + own derivation |
| `pipecg` | Ghysels/Cools–Vanroose pipelined PCG | 1 exch + **1 `Iallreduce` overlapped** with both SpMVs | Sergey `solvers.F90:118` + Alg 6 + own derivation |
| `oati` | PIPECG-OATI (JPDC 163 (2022) 147–155, Alg 4+5) | **1 Allreduce per 2 iterations** (+12 dots, +21 VMAs local) | own 2-iteration unroll of pipecg; paper = cross-check only |
| `pcsi` | P-CSI Chebyshev/Stiefel iteration (GMD 9, 4209–4225, 2016) | 1 exch + **0 reductions**; 1 Allreduce per `FESOM_PCSI_CHECK` iters | paper §B3/App C + Golub–Varga theory + own derivation |

Expected iteration counts: `cg2`/`pipecg`/`oati` ≈ `cg` (same Krylov space; rounding-level
drift); `pcsi` higher by design (Chebyshev is CG without the optimality — the wager is that
allreduce-free iterations are cheap enough at scale). Where the crossover sits, per mesh × node
count × backend, **is the paper's central figure**. Note `oati`'s halved blocking-sync count
does NOT depend on `Iallreduce` progression (that is `pipecg`'s mechanism) — a null overlap
probe (R2) reduces `pipecg`'s expectation, not `oati`'s.

**Compositions measured, not assumed:** the CGPIPE ring machinery transfers to every variant
(exchange the residual-class vector on 2 rings once/iter; preconditioner rows at ring1 are
already shipped verbatim, frozen — the zstar-safe precedent); CGPOLY slots in as the
preconditioner M inside `cg2`/`pipecg` (fewer iterations × fewer allreduces multiply).
`pcsi`+CGPOLY is excluded by design (nested Chebyshev) and aborts loudly.

**Non-goals:** changing the default solver (stays `cg` until an explicit user decree); changing
production `soltol` (sidebar measures the cost of accuracy; any change is a separate user
decision with its own climate leg); GH200 in the first wave; bit-identity for the new solvers.

## Context (from discovery)

**Files/components involved:**
- `src/fesom_ssh.cpp` (~2400 ln) — everything solver: C reference solver `fesom_ssh_solve_cg`
  (:418, **stays pristine** — it is the verify twin and the init-context solver used from
  `fesom_main.cpp:667/686/735`; dispatch does NOT touch it); production KK solver
  `fesom_ssh_solve_cg_kk` (:2059) — **dispatch goes here**; cgpipe machinery (ring shipping,
  `cgpipe_exchange_rr`, `cgpipe_zz_ring1`); cgpoly machinery (`cgpoly_build/apply/exchange`,
  distributed λmax power iteration ~:1776, symmetric scaling `isq = 1/sqrt(diag)` ~:1103,
  `FESOM_CGPOLY_KAPPA` default 30 = **assumed** λmin); preconditioner `pr_values`
  (MITgcm-class, same CSR sparsity, built once at init by `fesom_ssh_preconditioner` (:242)
  and never refreshed — the Fortran-precedent banner sits at ~:1988).
  **⚠️ `pr_values` is NOT symmetric** (`pr[i,j] = −0.5·(a_ij/d_i)/(d_i+d_j)`; the transpose has
  `d_j`) — reviewer finding; consequences → R1 and Task 3/4.
- `src/fesom_ssh.h` — `fesom_ssh_stiff` CSR struct (rowptr/colind/values/pr_values + Fields).
- `src/fesom_constants.h:105-106` — `FESOM_PHASE1_SOLTOL 1.0e-5`, `FESOM_PHASE1_MAXITER 500`.
- `src/fesom_step.cpp:782` — the per-step call site (1 solve/step). Matrix values refreshed per
  step under zstar via `fesom_update_stiff_mat_ale_kk` (`fesom_ssh.h:124`).
- `src/fesom_partit.cpp:225` — `MPI_Init_thread(…, MPI_THREAD_SINGLE, …)`: **no async progress**
  on this stack by default → R2 (Iallreduce progression must be probed, not assumed).
- `src/fesom_phasestats.h:44` — `FESOM_PH_CG` phase exists (time instrument, phst bins).
- Stopping rule (both solvers): `rtol = soltol * sqrt(‖b‖²/nod2D_global)`, recurrence residual
  `sqrt(‖r‖²/nod2D) < rtol` — Sergey's F90 uses the identical normalisation (his :39/:157 vs
  our :2165). Maxiter exhaustion currently `FESOM_DIE`s (:2314) → fallback trigger instead (T5a).
- `tests/` + `CMakeLists.txt:124-146` — add_executable pattern for host-side unit tests.
  (`tools/mpi_cuda_smoke.cpp` is a standalone mpicxx one-liner — NOT a precedent for the lab,
  which must link model objects → Task 3 CMake surgery + R7.)
- `jobs/job_m7_*` family (gate / ab_env / dtpair) — templates for `job_m10_*`.
  **⚠️ they hardcode `ROOT=/home/a/a270088/port_kokkos`** → R9 (worktree dead-gate trap).
- `scripts/m7_scaling_figs.py` — figure conventions of record (READ FIRST — user rule).

**Known numbers that size the problem** (all previously measured, cited from the M7 ledger):
CORE2 iters/solve 128.8 (poly d2 51.6 / d3 40.1); NG5 settled ~72 (dt240: 119.5); dars dt120:
50.1; CG halo pool 27.3→38.5 ms/step 4N→16N (187→264 µs/event); comm share of step 58 %@4N /
49 %@16N. The **allreduce share has never been isolated** — T2 measures it before any lever
(ALWAYS MEASURE; no invented pre-sizing in this plan).

**House rules that bind this track:** BIN= pinning on every measured run · same-day min-of-2
300-step pinned pairs · cheap gates get cheap walltimes (`-t 00:06:00` class) · `-C a100_80` on
GPU absolutes · private CORE2 mesh `/work/ab0995/a270088/port2/mesh/core2` (never /pool CORE2;
NG5/dars/farc perf runs DO use /pool) · snap_every=-1 at ≥4096 ranks · dead-knob trap L80/L102
(prove the knob FIRED via an observable) · L91 options gates per lever · rule 0.41 dts (dars
dt120, NG5 dt180 ladders / c32 dt60 point, CORE2 dt1800) · farc ≥128-rank proto-hang caveat ·
never commit binaries (frozen bins → `/work/ab0995/a270088/port2/m10/bin/` + sha256 in docs) ·
figure conventions per `m7_scaling_figs.py` (SYPD at production dt 1800/900/240/240).

**Merge surface vs the parallel M9 ice track:** M9 touches EVPWIDE/ice halo files; M10 touches
`fesom_ssh.{cpp,h}` + tools/tests/jobs/scripts additions. Overlap ≈ nil (shared halo-machinery
family headers at most). Both branch from the same lineage; merge order = user's call at the end.

## RISKS (consolidated — detection · mitigation)

- **R1 non-symmetric preconditioner.** `pr_values` ≠ `pr_valuesᵀ` breaks the SPD assumption
  under Lanczos/Chebyshev (P-CSI) and weakens CG-variant theory; it is also a testable
  hypothesis for the CG² instability Sergey observed. *Detect:* T3 symmetry-defect measurement
  `max|M_ij−M_ji|/max|M_ij|` on real dumps. *Mitigate:* P-CSI runs on a symmetric M decided in
  T4 (candidates: `D^{−1/2}AD^{−1/2}` — machinery exists in cgpoly — or a symmetrised MITgcm
  form); finding recorded in derivations doc + paper appendix.
- **R2 `Iallreduce` progression.** `MPI_THREAD_SINGLE` + openmpi 4.1 = no async progress by
  default; `pipecg`'s overlap may be structurally unavailable (and may explain Sergey's null).
  *Detect:* T2 standalone progression probe (both MPI modules, both backends), result → doc.
  *Mitigate:* pre-registered attribution in T6 (a null pipecg-vs-cg2 delta with a null probe is
  "stack", not "algorithm"); a progress thread would need `MPI_THREAD_MULTIPLE` = out of scope,
  documented.
- **R3 frozen eigenbounds under zstar.** Matrix values drift per step; [ν,µ] estimated once.
  *Detect:* lab drift check on the two 6-months-apart CORE2 dumps (T3/T8); in-model every-K
  true-residual check. *Mitigate:* safe-direction margins (deflate ν, inflate µ); armed
  fallback; `FESOM_PCSI_REEIG` built ONLY if the drift check demands it (YAGNI until then).
- **R4 lab→model transfer.** Lab replays one cold solve; the model warm-starts and drifts.
  *Mitigate:* promotion rule — no lab-tuned constant becomes a default until one in-model
  20-step gate reproduces the lab's iters/solve within a pre-registered ±10 % band; misses
  recorded as lab-vs-model divergences.
- **R5 pcsi maxiter.** Chebyshev needs more iterations than CG; `FESOM_PHASE1_MAXITER 500`
  could exhaust and (today) `FESOM_DIE` mid-campaign. *Mitigate:* maxiter exhaustion becomes a
  fallback trigger (T5a); `FESOM_PCSI_MAXITER` knob sized from lab convergence data.
- **R6 fallback correctness.** A fallback needs the pre-solve `X0` (the failed solver already
  wrote X; NaN X poisons the restart) and must be collective (a rank-local branch deadlocks —
  the farc hang shows the fleet cost). *Mitigate:* X0 snapshot at solve entry; design
  invariant: fallback/exit decisions branch ONLY on allreduced scalars (identical on all
  ranks by construction); `r·u ≤ 0` (indefinite-M signal, existing cgpoly guard class) added
  as a trigger for all variants.
- **R7 CMake refactor vs the byte class.** Splitting model objects for the lab touches the
  certified build path. *Detect/mitigate:* knob-off serial byte gate re-run immediately after
  the refactor (T3), before any solver code lands on top.
- **R8 farc ≥128 ranks.** Reproducible proto hang (E.T1 caveat). *Mitigate:* farc is an
  optional rung, class-A env only, documented.
- **R9 worktree dead-gate trap.** `job_m7_*` hardcode the MAIN checkout ROOT — a cloned gate
  submitted from the worktree would test the wrong binary with every gate green (L80 at gate
  scale). *Mitigate:* `job_m10_*` set `ROOT=${M10_ROOT:-$HOME/port_kokkos_ssh}` and print the
  resolved binary path + md5 into every log; T2 first-gate check asserts the md5 differs from
  the main checkout's binary; the T10 harvest asserts md5 == the frozen sha of record.

## Development Approach

- **Gates-before-code is this project's TDD:** every lever's acceptance gate and expected-range
  pre-registration is written in `SSH_SOLVERS_M10.md` BEFORE the A/B runs (house discipline).
  Code-first within a task, but no task closes without its gate set green.
- Each task completed fully (gates green, boxes ticked) before the next; small focused commits
  with the house one-line style; plan updated when scope changes.
- **Every task carries its tests** — here that means: host unit tests (`tests/`), lab
  equivalence checks, and SLURM gate jobs. They are checklist items, not optional. Per-solver
  correctness assertions live IN that solver's task (the testbed scaffold precedes them, T4).
- Backward compatibility is absolute: `FESOM_SSH_SOLVER=cg` (or unset) must be byte-identical
  to today's binary — proven by serial byte gate on the dispatch commit itself (T5a).

## Testing Strategy (the four layers — user-approved)

**Layer 0 — math triple-check (before any model run; papers may contain typos — REPORT them):**
1. **Re-derive, don't transcribe**: cg2 from PCG via the Chronopoulos–Gear substitutions;
   pipecg from cg2 via auxiliary vectors; oati by OUR OWN mechanical 2-iteration unroll of
   pipecg (paper listing = cross-check only); pcsi coefficients from Chebyshev semi-iteration
   (Golub–Varga). All in the derivations doc.
2. **Cross-source**: derivation vs Sergey's `solvers.F90` vs paper listing (vs P-CSI's own
   App B ChronGear listing as a fourth source for cg2). Every discrepancy → derivations doc
   "typo report" section + reported to the user (forwardable to Sergey).
   **Already-flagged ambiguity to resolve first:** P-CSI ω-recurrence extracted as
   `ω_k = 1/(γ − 1 4α2 ω_{k−1})` — `(1/4)α²` vs `1/(4α²)`; Golub–Varga says the coefficient is
   `δ²/4 = 1/(4α²)` with `α=2/(µ−ν)`. Verify against the PDF layout AND numerically (wrong
   reading ⇒ wrong convergence rate, instantly visible in the lab). OATI's dense two-column
   listing is the highest-risk transcription — derivation is authoritative there.
3. **Numerical equivalence on real matrices (decisive):** cg2/pipecg/oati reproduce reference-
   PCG iterates to rounding (α/β sequence compare, early-iteration iterate diffs ~1e-12 rel);
   pcsi converges at the theoretical Chebyshev rate for its [ν,µ]. Runs in the lab + the serial
   unit testbed (`tests/test_ssh_solvers.cpp`, login-runnable) — scaffold in T4, per-solver
   assertions in T5–T8.

**Layer 1 — knob-OFF byte gates** (per implementation commit): serial byte-identity
(diff_snap rc=0) + CUDA fidelity, unchanged house standard. All new solvers opt-in.

**Layer 2 — per-variant solver-class gates** (20-step CORE2, cheap walltime):
`[ssh-wire]` observable proves the knob fired (allreduce/exchange counts CHANGE as designed);
prognostic fields vs baseline within the pre-registered solver-class bounds (written in T2,
citing `docs/REFERENCE_RUNS.md` per-scheme floors + the L79 zstar Kv ~1e-1 control class +
the CGPOLY/M5.23 bar — BEFORE any variant gate is submitted); **true residual ‖b−Ax‖ ≤ rtol
re-verified every solve** (`FESOM_SSH_VERIFY=1` in every gate — catches recurrence-residual
drift, the known pipelined failure mode); options ×3 (TKE / mEVP / **zstar** — the
time-varying-matrix case that stresses frozen pr_values AND frozen eigenbounds); iteration
parity reported (cg2≈cg expected; pcsi higher by design — reported, not gated).

**Stability policy (armed everywhere):** triggers = NaN · residual stall/growth over a window ·
`r·u ≤ 0` · maxiter exhaustion. Action: restore the solve-entry `X0` snapshot, redo the CURRENT
solve with baseline `cg`, count the event, warn on rank 0. All triggers derive from allreduced
scalars ⇒ the decision is collective by construction (R6). Certification requires **0 fallback
firings** across the 300-step measurement runs; the guard stays armed in production.
`FESOM_SSH_FALLBACK=0` exists only for lab/probe experiments.

**Layer 3 — climate:** 1-yr CORE2 leg at the CGPOLY/M5.23 bar (sst 1.00000 / sss 0.99996 /
ssh 1.00000 / a_ice 0.99997 correlation class) for the RECOMMENDED config(s) only.

**Perf protocol:** solver A/Bs and ladders per the M7 standard (300-step pinned pairs,
min-of-2, same-day, BIN=, class-A env held fixed; env-pkg interaction measured once on the
winner as an optional Bp leg). **µs/iteration is a mandatory harvest column** next to s/step
and CG-phase ms/step — it separates "cheaper iteration" from "different iteration count", the
distinction the crossover claim rests on. Numbers of record NEVER from the lab.

## Progress Tracking

- tick `[x]` immediately when done; ➕ prefix for discovered tasks; ⚠️ for blockers
- every gate/A-B records its SLURM job id next to the checkbox (house style)
- update this plan + `SSH_SOLVERS_M10.md` in the same commit as the work

## What Goes Where

- **Implementation Steps** (checkboxes): everything achievable in-repo — code, gates, jobs,
  figures, docs.
- **Post-Completion** (no checkboxes): GH200 campaign execution, Sergey feedback loop, paper
  writing, adoption/tolerance decisions (user), M9 merge-order decision (user).

## Implementation Steps

*Session map (orientation only — the checkboxes are the truth, the map is elastic):
S1 = T1–T3 · S2 = T4–T5 · S3 = T6–T7 · S4 = T8 · S5 = T9–T10 · S6 = T11–T13.*

### Task 1: Worktree + hygiene

**Files:**
- Modify: `.gitignore` (on `m10-ssh-solvers`)
- Create: worktree `~/port_kokkos_ssh` · `/work/ab0995/a270088/port2/m10/{bin,labdumps,gates,ab,figs}`

- [x] worktree created 2026-08-05 — ➕ base `f42c453`, not `65a1a71` (deviation, documented in
      `SSH_SOLVERS_M10.md` §Provenance: `f42c453` = `65a1a71` + the plan commit itself, which puts
      THIS plan + the `ssh_sergey/` gitignore in-tree; `src/` identical to `65a1a71`)
- [x] `.gitignore` `ssh_sergey/` inherited from `f42c453`; `ssh_sergey/` copied into the worktree
- [x] `/work/.../m10/{bin,labdumps,gates,ab,figs}` created; `SSH_SOLVERS_M10.md` skeleton;
      quota checked (group ab0995 no hard cap; global /work ≈14 TB free — re-check before NG5 dumps)
- [x] baseline builds clean (login, -j16): serial `f228d664…`, cuda `e5245fa3…` → doc §m10-base
      (➕ kokkos submodule checked out in-worktree @ `15dc143e`)
- [x] test: knob-free serial 20-step CORE2 gate from the worktree — **PASS** diff_snap rc=0,
      job 26722627 (via ➕ `jobs/job_m10_gate`, the R9-safe job created early: prints worktree
      binary md5 `54433326…` ≠ main-checkout `9743f602…` in-log)

### Task 2: Instrumentation first — `[ssh-wire]`, verify, probes, baseline census

**Files:**
- Modify: `src/fesom_ssh.cpp`, `src/fesom_ssh.h`
- Create: `jobs/job_m10_gate` (cheap-walltime gate template) · `tools/iallreduce_probe.c` (tiny, standalone)

- [x] `[ssh-wire]` counters in `fesom_ssh_solve_cg_kk` (iters, exch events, blocking/i-allreduce,
      solver-body kernel launches ➕, fallback, final recurrence residual); per-solve line +
      finalize aggregate under `FESOM_SSH_STATS=1`; ➕ (CUDA) launch-overhead micro-probe in
      `fesom_ssh_wire_report` (async + fenced µs/launch — the T2 launch pricing)
- [x] `FESOM_SSH_VERIFY=1`: post-solve true residual (fused SpMV-diff-reduce, byte-transparent,
      comm/launches not wire-counted); threshold DEFERRED — first data: pi np2 max gap 1.05e-11
- [x] knobs parse via the house pattern (`fesom_ssh_env01`: unrecognized ⇒ abort listing 0/1;
      rank-0 announce when ON)
- [x] `job_m10_gate` (created in T1): `ROOT=${M10_ROOT:-...}`, binary md5 + main-checkout md5
      printed (R9)
- [x] serial knob-off byte gate on the instrumentation commit: **PASS** job 26722771, diff_snap
      rc=0, log md5 `8f2be32b…` ≠ main `9743f602…` (R9 armed proof)
- [x] **Iallreduce progression probe** (R2): jobs 26722815 (compute, both stacks) + 26722816
      (GPU nodes) — **NO progression on any stack** (wait = full AR latency at every busy
      factor; Iallreduce path carries +8 µs (4.1.2) / +1.6-1.8 µs (4.1.5) surcharge over
      blocking); attribution rule pre-registered in doc §census (null pipecg ⇒ STACK)
- [x] baseline census harvested (doc §census): CORE2 np8 login (132.35 it, legacy+FORCE_SERIAL
      cgpipe legs) · dars 4N 26722817 (40.2 it, cg 11.0+8.3 → 6.1+4.4 ms/step) · NG5 4N
      26722818 (83.7 it) · NG5 16N 26722819 (83.7 it, cg 10.4 busy + **13.0 wait** ms/step —
      majority-comm at scale); launch pricing ➕ IN-BINARY probe: async 3.0 / fenced 8.9
      µs/launch (A100, stable across 4 runs)
- [x] **Layer-2 per-field solver-class bounds pre-registered** (doc §P-L2) — ➕ calibrated by
      MEASUREMENT: off-vs-CGPOLY-d3 FORCE_SERIAL 20-step pair on the exact gate config (the
      certified solver-class precedent) instead of invented numbers; mEVP
      formal-FAIL-with-exoneration shape + strict-matrix rule carried over
- [x] test: counts reconcile EXACTLY — legacy 2+2k / cgpipe 2+k at CORE2 (266.7/134.35 @
      k=132.35) and dars (82.34/42.18 @ k=40.17); iteration counts IDENTICAL legacy-vs-speed1
      (CGPIPE byte-claim at count level, CUDA included)

### Task 3: Solver lab — matrix dump + replay driver

**Files:**
- Create: `tools/fesom_ssh_lab.cpp`
- Modify: `CMakeLists.txt` (new executable linking model objects — OBJECT-library split or
  equivalent; relocates `target_compile_definitions` carefully)
- Modify: `src/fesom_ssh.cpp` (`FESOM_SSH_DUMP=<csv-steps>` — write per-rank CSR
  rowptr/colind/values/pr_values + b + x0 + x_final to `/work/.../m10/labdumps/<mesh>_np<N>/step<NNNN>/`)

- [x] dump knob `FESOM_SSH_DUMP=<csv>` + `FESOM_SSH_DUMP_DIR` + binary format
      (`src/fesom_ssh_dump.h`; FNV-64 per array + trailer magic)
- [x] lab driver `tools/fesom_ssh_lab.cpp` — the model's OWN init path + the model's OWN
      solver (zero reimplementation), `--solver/--tol/--maxiter/--reps/--trace/--knob`;
      ➕ `--sym-check` (R1) and ➕ `--sigma-drift` (the Layer-0 falsification experiment).
      The dumped rowptr/colind are asserted BITWISE vs the freshly built CSR = proof the lab
      rebuilt the same partitioning. `FESOM_SSH_TRACE=1` gives the α/β sequences (%.17g)
- [x] **post-refactor byte gate** (R7): **PASS** job 26723005, diff_snap rc=0, run immediately
      after the CMake OBJECT-library split and before any solver code
- [x] dumps taken: CORE2 np8 steps 1/20/300 · ➕ CORE2 np1 step 20 (100 %% sym coverage +
      bitwise cert) · dars np64 (26723046) · NG5 np256 (26723047) · zstar CORE2 np1 steps
      100/8740 = 180 days apart (26723048)
- [x] **symmetry-defect measurement** (R1): pr_values **0.638**, A control **1.42e-13**
      (99.3 %% pair coverage, identical at steps 1 and 20) → derivations §0.2. ⭐⭐ This grew
      into the track's headline finding (§0.4/§0.4b): the defect BREAKS the σ recurrence that
      cg2/pipecg/oati all share — α wrong by **21.8 %%** on CORE2, 1.2e-13 once symmetrised
- [x] test (lab certification): CORE2 np8 step-20 replay — iters 125=125 AND x_final
      **BITWISE** on all 126858 owned nodes (stronger than the np>1 criterion required)
- [x] test: FNV-64 per array, verified at every load; loader hard-fails on mismatch
- [x] rules recorded in `SSH_SOLVERS_M10.md` §Findings ledger (lab-never-of-record + R4
      promotion rule + CUDA-dumps-are-matrix-material-only)

### Task 4: Layer-0 derivations + typo triple-check + testbed scaffold

**Files:**
- Create: `docs/plans/20260805-m10-ssh-derivations.md`
- Create: `tests/test_ssh_solvers.cpp` · Modify: `CMakeLists.txt` (host-only unit test)

- [x] derived all four in `docs/plans/20260805-m10-ssh-derivations.md` (paper-appendix quality;
      cg2 from PCG via the S/R substitutions, pipecg from cg2 via the aux vectors, oati by our
      own 2-iteration unroll, pcsi from Golub-Varga)
- [x] cross-source tables per algorithm (4 sources for cg2 incl. [P] App B2 — which proves
      ChronGear's `σ_k = δ_k − β_k²σ_{k-1}` is algebraically IDENTICAL to CG-CG's α form);
      **typo report T-1…T-9**; ⭐ P-CSI ω **RESOLVED to `1/(4α²)`** by derivation + [P]'s own
      `ω₀ = 2/γ` seed + Golub-Varga, and the testbed proves the misread coefficient misses the
      Chebyshev bound while the derived one meets it
- [x] **preconditioner DECIDED** — `M̃⁻¹ = D^{−1/2} C D^{−1/2}` (derivations §0.5): symmetric,
      SAME preconditioner spectrum (similar matrix, verified to 10 s.f.), same sparsity, cost =
      one setup-time scaling. ➕ **SCOPE CHANGE: the decision governs ALL FOUR solvers, not just
      pcsi** — §0.4 shows cg2/pipecg/oati share the σ recurrence that needs it. Knob
      `FESOM_SSH_SYMPRE` (T5a), pcsi rejects `=0`. Hypothesis for the CG² instability written up
      AND numerically confirmed (§0.4b)
- [x] Layer-0 findings reported to the user in-session (T-1…T-9 + the §0.4b measurement),
      forwardable to Sergey
- [x] testbed `tests/test_ssh_solvers.cpp` (+ ctest `ssh_solvers`): 32×32 SPD Laplacian in
      `fesom_ssh_stiff` CSR shape (diag at offset 0), reference PCG, α/β comparator (detects an
      injected 1e-9 perturbation ✅), Chebyshev-rate checker measured in the **A-norm of the
      error** (a residual ratio is not what the theory bounds — first draft was wrong and the
      test caught it). ➕ a VARIABLE-diagonal fixture: the uniform Laplacian has a constant
      diagonal, which makes the MITgcm preconditioner accidentally symmetric and would have
      hidden §0.4 entirely. ➕ Lanczos + Sturm bisection (the T8a prototype) proving Ritz values
      converge OUTWARD (justifies deflate-ν/inflate-µ) and that [P]'s un-rooted `q₁` costs
      2270× in θmin
- [x] testbed green on login: **PASS (0 failures)**, 18 assertions, <1 s

### Task 5a: Dispatch + shared guard infrastructure

**Files:**
- Modify: `src/fesom_ssh.cpp`, `src/fesom_ssh.h`

- [ ] `FESOM_SSH_SOLVER` dispatch (default `cg` = the existing path UNTOUCHED; abort on
      unrecognized; rank-0 announce; L80-visible)
- [ ] interaction matrix enforced in code (see Technical Details): `FESOM_KK_VERIFY=ssh` +
      non-`cg` ⇒ `FESOM_CHECK` abort (CGPOLY precedent :2104-2108); npes==1 / `FESOM_HOST_HALO=1`
      ⇒ warn-and-degrade to the `FESOM_SSH_RING=0` literal form (cgpipe precedent :2111-2119);
      `pcsi`×CGPOLY ⇒ die
- [ ] shared fallback infrastructure: solve-entry X0 snapshot; triggers (NaN, stall/growth
      window, `r·u ≤ 0`, maxiter exhaustion — replaces the :2314 die for non-cg solvers);
      collective-by-construction (allreduced scalars only); `[ssh-wire]` fallback counter;
      `FESOM_SSH_FALLBACK=0` escape
- [ ] teardown: `fesom_ssh_m10_free()` for all new persistent Views, registered before
      `Kokkos::finalize` (static-destruction hazard precedent ~:1838)
- [ ] gates: serial knob-off byte on the dispatch commit (job id → doc) · CUDA knob-off fidelity
- [ ] test: testbed still green (dispatch refactor changed no math)

### Task 5b: CG² (host + device)

**Files:**
- Modify: `src/fesom_ssh.cpp`

- [ ] `cg2` ring-composed (2-ring rr exchange reusing cgpipe shipping; uu at ring1 from shipped
      pr rows; pp/ss by recurrence; scratch Views sized to the ACTIVE composition's ring extent,
      not blindly N+eDim) + fused 3-element blocking allreduce; `FESOM_SSH_RING=0` literal
      2-exchange form (bring-up/debug + npes==1 fallback ONLY — never a gated or recommended
      configuration, excluded from the options matrix)
- [ ] testbed: cg2 assertions (converges; iterates match reference PCG to rounding; fallback
      fires on an injected NaN) — green on login
- [ ] lab: cg2 α/β vs cg on the real CORE2 dump (~1e-12 rel, early iters)
- [ ] gates (job ids → doc): serial cg2 vs cg 20-step solution-class (bounds from T2
      pre-registration) + true-residual + iters parity · CUDA cg2 vs serial cg2 · options ×3
      (TKE/mEVP/zstar) under cg2 · wire observable (allreduces/iter 2→1, exchanges unchanged)
- [ ] pre-register the A/B expected range in the doc (from T2 census arithmetic, stated as a
      range with the reasoning), THEN run: NG5 4N + 16N GPU cg2-vs-cg, 300-step pinned pairs,
      min-of-2, same-day; harvest incl. µs/iteration
- [ ] freeze `m10-cg2` bins (serial+CUDA sha256 → doc, binaries → `/work/.../m10/bin/`)

### Task 6: pipecg

**Files:**
- Modify: `src/fesom_ssh.cpp`

- [ ] pipecg recurrences (+4 vectors: zz,qq,mm,nn class, ring-extent sized), `MPI_Iallreduce`
      posted before the overlap window (ring-composed ww exchange + precond SpMV + matrix SpMV),
      `MPI_Wait` after; first-iteration special-case per Sergey's F90 (`IF(iter>1)`, :205)
- [ ] testbed + lab equivalence assertions for pipecg (as 5b)
- [ ] Layer-2 gate set (same shape as 5b) + wire observable (iallreduce=1/iter, blocking=0)
- [ ] attribution pre-registered from the T2 probe (R2): what a null vs cg2 means
- [ ] pre-register + A/B NG5 4N/16N vs cg AND vs cg2 (the "does overlap pay on GPU" question);
      harvest incl. µs/iteration; attainable-accuracy watch: max |true−recurrence| residual
      across the run → ledger
- [ ] test: gates green before Task 7

### Task 7: oati

**Files:**
- Modify: `src/fesom_ssh.cpp`

- [ ] implement from OUR unroll (T4), not the paper listing; single fused allreduce per 2
      iterations (≈12-element buffer); odd-count exit handled; vector-op fusion where row order
      allows (document any intentional order changes — solver-class anyway)
- [ ] testbed + lab equivalence (every-2nd-iterate match vs pipecg)
- [ ] Layer-2 gate set + wire observable (allreduce count ≈ iters/2)
- [ ] pre-register + A/B NG5 4N/16N vs cg2/pipecg — the flop/launch-overhead-vs-sync tradeoff
      is the measurement (T2's launch pricing feeds the pre-registration); an honest negative
      is paper material, not a failure. (Unconditional per user decision — R2's probe outcome
      changes the EXPECTATION, not the scope.)

### Task 8a: Lanczos eigen-estimator

**Files:**
- Modify: `src/fesom_ssh.cpp`, `src/fesom_ssh.h` (+ small symmetric-tridiag eigensolver,
  host-only, init-time)

- [ ] Lanczos on M⁻¹A with the T4-DECIDED symmetric M (`FESOM_PCSI_LANCZOS` steps, default 30)
      at first solve, reusing the solver's SpMV/dot/allreduce primitives; T_m eigenvalues via
      bisection/QL; margins with the SAFE directions — **deflate ν, inflate µ** (spectrum must
      stay ⊂ [ν,µ]); `FESOM_PCSI_EIGMARGIN` (default "0.10,0.05", finalized in the lab),
      `FESOM_PCSI_EIG="ν,µ"` override; rank-0 announce
- [ ] **rank-agreement assertion** (R6-class): [ν,µ] computed from allreduced dots on all ranks
      AND asserted bitwise-identical via MIN/MAX allreduce (abort on mismatch) — converts a
      silent wrong-ω divergence into a loud stop
- [ ] testbed: Lanczos on the Laplacian fixture recovers known extreme eigs within margin
- [ ] test: estimator on the CORE2 lab dump; θ's stable across np1/np8 (job/log → doc)

### Task 8b: pcsi iteration + guards

**Files:**
- Modify: `src/fesom_ssh.cpp`

- [ ] pcsi per the verified recurrence (b−Ax true-residual form — self-correcting, note in
      docs); ring-composed single 2-ring r exchange; convergence check every `FESOM_PCSI_CHECK`
      iters (default 5) = the only allreduce; `FESOM_PCSI_MAXITER` (default sized from lab
      data; exhaustion ⇒ fallback, not die); divergence detect at check points ⇒ fallback
      (+ log); `pcsi`+CGPOLY ⇒ die (already in 5a matrix)
- [ ] testbed: pcsi rate within Chebyshev bound on the fixture; broken-[ν,µ] fallback fires
- [ ] Layer-2 gate set + wire observable (blocking allreduces ≈ iters/K, exchanges 1/iter)
- [ ] test: gates green before 8c

### Task 8c: pcsi tuning + A/B

- [ ] lab tuning campaign: margins × check-interval × Lanczos-m on all three meshes' dumps
      (CORE2 on login; dars/NG5 replays as cheap SLURM jobs); zstar drift check on the two
      6-months-apart CORE2 dumps (R3) — **`FESOM_PCSI_REEIG` is built ONLY if this check shows
      the bounds move**
- [ ] R4 promotion gate: one in-model 20-step gate reproduces lab iters/solve within ±10 %
      before any tuned constant becomes a default
- [ ] pre-register + A/B: NG5 4N/16N + dars g8n vs cg2 (iters↑ vs sync↓ — the wager measured);
      harvest incl. µs/iteration
- [ ] ➕ opportunistic: `FESOM_CGPOLY_KAPPA=auto` — feed measured λmin/λmax into CGPOLY in
      place of the assumed κ=30 (lab first; ships only if it helps iterations)

### Task 9: Compositions + tolerance sidebar

**Files:**
- Modify: `src/fesom_ssh.cpp` (CGPOLY-as-M slot inside cg2/pipecg)

- [ ] `cg2`+CGPOLY d3 and `pipecg`+CGPOLY d3 (M-apply = `cgpoly_apply` — SPD by construction,
      the symmetric scaling is native there; ring extents already handled by the poly
      machinery); zstar gate minimum + solution-class gate each
- [ ] lab equivalence: poly-preconditioned cg2 vs poly-preconditioned reference PCG
- [ ] pre-register + A/B the two compositions at 4N/16N (naive-additivity stated as the prior,
      L93 entanglement expected — that IS the measurement)
- [ ] tolerance sidebar (lab): iters vs soltol ∈ {1e-4 … 1e-7} per mesh; model spot-check at 2
      points on CORE2; documented as "cost of accuracy" — **no production change here**

### Task 10: Campaign fleet + figures

**Files:**
- Create: `jobs/job_m10_ladder` (m7 ab_env family clone, M10_ROOT-aware) · `scripts/m10_ssh_figs.py`

- [ ] **read `scripts/m7_scaling_figs.py` conventions FIRST** (user rule: node-axis ticks,
      decimal y, SYPD at production dt, legend below panels)
- [ ] fleet (per-rung same-day min-of-2, BIN-pinned, class-A env): dars c1→c32 + g2→g32, NG5
      c8→c32 + g4→g32, CORE2 c1→c4 + g1→g4 — legs per rung: {cg, cg2, pipecg, oati, pcsi,
      best-composition}; one legacy-cg (CGPIPE off) reference leg per mesh/scale for the
      paper's "plain PCG" point; farc = optional rung (R8)
- [ ] winner Bp leg (env pkg) once, at 16N
- [ ] harvest → `SSH_SOLVERS_M10.md` tables (s/step · CG-phase ms · **µs/iteration** · iters);
      figures → `/work/.../m10/figs/`: **F1** solver-time vs nodes per method (per mesh,
      CPU/GPU panels, iters annotated) · **F2** total SYPD vs nodes best-vs-baseline · **F3**
      convergence histories (lab, real matrices) · **F4** stacked comm-anatomy bars at 16N/32N
      (compute/halo/allreduce from wire+phasestats) · **F5** crossover map (winning method per
      mesh × scale × backend)
- [ ] test: harvest script cross-checks every plotted point against its job log — md5 vs the
      frozen sha of record (R9), announce lines (L80), scripted

### Task 11: Docs of record + recommendation

**Files:**
- Modify: `docs/SSH_SOLVERS_M10.md`, `README.md` (knob table), `docs/KOKKOS_PORTING_LESSONS.md`
- Modify: `docs/plans/20260805-m10-ssh-derivations.md` (final typo report)

- [ ] complete ledger: every gate job id, every A/B, every pre-registration vs outcome
      (wrong-high/wrong-low noted, house style)
- [ ] crossover analysis + **the recommendation**: which solver per mesh-class × scale ×
      backend, stated as a documented manual knob (default stays `cg`)
- [ ] README options table += `FESOM_SSH_*` + `FESOM_PCSI_*` knobs; lessons appended
- [ ] test: doc claims spot-audited against job logs (no number without a job id)

### Task 12: Climate leg

- [ ] 1-yr CORE2 with the recommended config (reuse the `m7_climate_check_plots.py` pipeline
      + M5.23-bar comparison); job id + verdict → doc
- [ ] pass ⇒ recommendation stands; fail ⇒ ⚠️ documented, recommendation demoted, next-best
      config gets the leg

### Task 13: Acceptance + wrap

- [ ] verify every Overview goal: 4 solvers implemented + gated; compositions measured;
      campaign harvested; figures F1–F5 exist; derivations + typo report delivered; climate
      leg verdict recorded
- [ ] full gate registry green; testbed + lab certification re-run on the final binaries
      (the wsplit "ladder of record on final bins" precedent)
- [ ] GH200 follow-up plan stub (`docs/plans/YYYYMMDD-m10-GH200.md`: winners only, dolpung
      first, `FESOM_HALO_STAGE=1` + no-GPUDirect facts carried over)
- [ ] paper outline doc (intro/methods/verification/results/discussion skeleton with figure
      slots F1–F5 + derivations appendix pointer)
- [ ] move this plan to `docs/plans/completed/`

## Technical Details

**Knob table (all abort-on-unrecognized, rank-0 announce, L80-observable):**

| knob | default | meaning |
|---|---|---|
| `FESOM_SSH_SOLVER` | `cg` | `cg`·`cg2`·`pipecg`·`oati`·`pcsi` |
| `FESOM_SSH_RING` | `1` | 0 = literal 2-exchange forms — bring-up/debug + npes==1 fallback ONLY, never gated/recommended |
| `FESOM_SSH_STATS` | off | `[ssh-wire]` per-solve + aggregate |
| `FESOM_SSH_VERIFY` | off (gates: on) | post-solve true-residual check |
| `FESOM_SSH_FALLBACK` | `1` (armed) | 0 = disable auto-fallback (experiments only) |
| `FESOM_SSH_DUMP` | off | csv step list → lab dumps |
| `FESOM_PCSI_CHECK` | `5` | iterations between residual checks |
| `FESOM_PCSI_MAXITER` | lab-sized | pcsi-specific cap; exhaustion ⇒ fallback |
| `FESOM_PCSI_LANCZOS` | `30` | Lanczos steps for [ν,µ] |
| `FESOM_PCSI_EIGMARGIN` | `0.10,0.05` | deflate-ν, inflate-µ fractions |
| `FESOM_PCSI_EIG` | unset | explicit `ν,µ` override |

(`FESOM_PCSI_REEIG` deliberately absent — built only if the T8c drift check demands it.)

**Interaction matrix (enforced in T5a, one cell = one behavior):**

| condition | × non-`cg` solver ⇒ |
|---|---|
| `FESOM_KK_VERIFY=ssh` | `FESOM_CHECK` abort (C twin runs legacy solver) |
| npes==1 or `FESOM_HOST_HALO=1` | warn + degrade to `FESOM_SSH_RING=0` literal form |
| `FESOM_SPEED_CGPIPE` | ignored (ring machinery is used internally; knob applies to `cg` only) |
| `FESOM_SPEED_CGPOLY` | `cg2`/`pipecg`: preconditioner slot · `pcsi`/`oati`: `FESOM_DIE` |

**Per-iteration anatomy (design targets, ring-composed):**

| solver | SpMV A | SpMV M | AXPY-class | exch | blocking AR | overlapped AR |
|---|---|---|---|---|---|---|
| cg (SPEED=1) | 1 (fused dot) | 1 (fused dot2) | 3 | 1 | 2 | – |
| cg2 | 1 | 1 | 4 | 1 | 1 (3-elem) | – |
| pipecg | 1 | 1 | 8 | 1 | – | 1 (3-elem) |
| oati (per 2 it) | 2 | 2 | ~2×8+21 VMA | 2 | – | 1 (≈12-elem) |
| pcsi | 1 | 1 | 3 | 1 | 1/K | – |

Each AXPY is a kernel launch on GPU — T2's census prices launches so the pipecg/oati
pre-registrations weigh +launches against −allreduces with measured numbers.

**Ring-composition invariant** (why 1 exchange suffices everywhere): exchange the residual-class
vector on rings 1+2 once per iteration; the preconditioner apply at ring1 uses the pr_values
rows already shipped verbatim at setup (frozen — the cgpipe zstar-safe precedent); every other
ring1 quantity is maintained by recurrence; owned-row SpMVs then see current halos. No stiffness
rows are ever shipped (they change per step under zstar — that trap is what the pr-only design
avoids). Scratch-vector extents follow the ACTIVE composition (N+eDim under cgpipe rings vs
N+Σnring under poly — allocation per cell, not one-size).

**P-CSI specifics:** recurrence uses the b−Ax true-residual form (self-correcting — no drift
problem by construction; the check-interval allreduce reads a residual that is already true).
Lanczos Ritz values converge OUTWARD-short: θmax ≤ λmax and θmin ≥ λmin, hence the margin
directions (deflate ν, inflate µ) are both conservative — VALID ONLY for symmetric M, which is
why T4's preconditioner decision precedes all of Task 8 (R1). Frozen bounds under zstar are the
bet (Sergey: matrix changes are tiny; eigenvalues set by domain + resolution) — the lab drift
check + the armed fallback are the nets under it (REEIG only if the check demands it).

**Dump format:** per-rank raw binary, little-endian, header {magic, version, step, dt, npes,
myDim, eDim, nnz, mesh-id hash} then rowptr/colind/values/pr_values/b/x0/x_final. Lab rebuilds
partitioning from the same mesh+dist inputs — byte-exact halo graph, zero serialization of the
com structures.

**Naming note for the paper:** our historical CGPIPE ≠ literature "pipelined CG". In all M10
text: CGPIPE = "single-exchange ring PCG" (halo-side); `pipecg` = the Ghysels-class pipelined
PCG (allreduce-side). The knob names keep the historical spellings; prose must not.

## Post-Completion

**GH200/JUPITER follow-up** (own plan doc, after winners known): winning 2–3 configs on dolpung
ladders; carry the s17 facts (`FESOM_HALO_STAGE=1`, no GPUDirect on that fabric, NEVER
HOST_HALO with CG ring solvers).

**Sergey feedback loop:** send the typo report + the pr_values non-symmetry finding (candidate
explanation for his CG² instability) + lab convergence comparisons + the crossover figure; his
Fortran implementations make a natural cross-model validation section if he runs them at
matching tolerance.

**User decisions parked for the end:** adoption/default question; production soltol change (has
its own climate-leg price); M9-vs-M10 merge order; farc inclusion in the paper.

**Paper:** venue GMD-class; skeleton from Task 13; data/figures publishable via the existing
Zenodo machinery (global CLAUDE.md notes).
