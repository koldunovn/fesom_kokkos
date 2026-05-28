# M5.11 — GPU Profile Pass ("understand the wall")

## Overview

Profile-only milestone. Produce per-kernel timing + a hardware-level bound label for
the top GPU kernels of the port_kokkos build, so the next optimization milestone
(M5.12) is evidence-based rather than guessed.

- **Problem it solves**: The Kokkos GPU build is 4.6–5.3× slower than the Kokkos-CPU
  node and 5.6–8.9× slower than the C port node-for-node on CORE2 (see
  `docs/SCALING_CORE2.md`). The previous attempt (M5.10b, "Lever C": rank-1 → rank-2
  `LayoutLeft` memory coalescing on FCT, the dominant 17.4% phase) was REVERTED — it
  delivered only **1.3% on the step and 0.6% on FCT** itself. (The historical
  `docs/LEVER_C_PLAN.md` lives on the `leverC-coalesce` branch; the revert is master
  commit `b27e4f6`. M5.10b's §RESULT section is preserved there for posterity.)
  Diagnosis: FCT was not memory-coalescing-bound; the three `atomic_add` scatters,
  mixed-layout loads, and Zalesak branch divergence dominate. **The Lever C failure
  proves a structural lesson: without per-kernel timing and hardware-bound labels,
  the next lever is a guess.**
- **Priority**: NO code optimization in scope. M5.11 only produces measurements and
  the M5.12 plan draft.
- **Goal output**:
  - `docs/PROFILE_M59.md` — per-kernel R1 table, halo-cost split (np=1 → dist_4 →
    dist_8), nsys timeline summary, `ncu` bound labels for the top 5 kernels, host
    residual audit, and **one named M5.12 lever** with payoff range and risk.
  - `docs/M512_PLAN.md` — next-session lever plan in the `LEVER_C_PLAN.md` shape.

## Context (from discovery)

- **Branch**: `profile-m511`, forked off `master` (HEAD = `466ea3e` = M3.2 docs
  closure on top of `05182aa` = M5.9-pin; no code delta between them). The
  `leverC-coalesce` branch carries the M5.10a `Field2DT<T>` scaffold + M5.10b-i
  halo support as parked infrastructure — **not** carried into M5.11.
- **Existing profiler** (M5.6, `src/fesom_profile.{hpp,cpp}`): a fence-bounded
  host+device wall-time accumulator, env-gated by `FESOM_STEP_PROFILE`. Provides
  PHASE granularity only (FCT 17.4%, momentum ~15%, GM 9.9%, SSH 9.1%, EVP 7.9%).
  No per-kernel-within-substep breakdown, no PCIe traffic count, no NVTX bridge.
- **Kokkos profiling API** (verified in bundled Kokkos 4.0 at
  `externals/kokkos/core/src/impl/Kokkos_Profiling.{hpp,_C_Interface.h}`):
  `Kokkos::Profiling::pushRegion(const std::string&)` / `popRegion()` are
  always-on (no `#ifdef` guard, bridges to NVTX automatically when nsys is
  attached), and
  `Kokkos::Tools::Experimental::set_begin_deep_copy_callback` accepts a callback
  of type `void(SpaceHandle, const char*, const void*, SpaceHandle, const char*,
  const void*, uint64_t)` — used to count PCIe transfers (count + bytes).
- **~80–100 device launch sites** across the 6 hot-phase files (initial Explore
  agent count: 79; plan-review agent recount: 96 — the delta is in init/setup
  parallel_fors). After the C1 design pivot, **no manual wrapping is needed**:
  the Kokkos `set_begin/end_parallel_{for,reduce,scan}_callback` API auto-
  instruments every kernel by its label, including kernels OUTSIDE the 6 hot
  phases (SSH CG, ALE, ice FCT, etc.). The exact count surfaces in
  `PROFILE_M59.md §1`. ~10 of these are `atomic_add` scatters that hold known
  non-determinism.
- **Canonical job templates**: `jobs/job_gpuaware_prof_core2` (CORE2 dist_8 with
  `env_cuda.sh` + JRA55=1958 for active-ice) and `jobs/job_nsys_core2_np1`.
  `JRA55_YEAR=1958` is the active-ice trigger — EVP/FCT/icethermo only run with
  it. **All M5.11 jobs must use JRA55=1958.**
- **Validation pressure**: low. Instrumentation is additive, env-gated, and
  default-off. The load-bearing gate is `scripts/gpu_fidelity_gate.sh`
  (CORE2-active-ice CUDA-vs-Serial) — pi alone is not enough (the M5.9
  stale-host class only manifests on CORE2-active-ice).
- **Project memory**: see `~/.claude/projects/-home-a-a270088-port-kokkos/memory/`
  — particularly `feedback-gpu-fidelity-gate.md`, `feedback-hpc-run-hygiene.md`,
  `reference-cuda-aware-mpi.md`.
- **Plan format precedent**: `docs/plans/20260525-kokkos-port.md` (the big-picture
  port plan, milestone-structured) and `docs/plans/completed/20260524-kpp-vertical-mixing.md`
  (a focused single-milestone plan). This plan adopts the milestone-structured
  style + a §RESULT-style postmortem at close (a convention introduced informally
  by the now-on-`leverC-coalesce` LEVER_C_PLAN.md).

## Development Approach

This is a **profile-only milestone** of a scientific-model port. The generic
"unit-test-per-function/TDD" model does not fit — there is no new physics code,
only instrumentation. "Tests" here are **validation gates** that the existing
per-kernel-verify + SYNCCHECK + GPU-fidelity-gate infrastructure already
provides.

- **Validation gates** (the "tests"):
  - **Build clean** on `build-serial`, `build-omp`, `build-cuda` (build-cuda uses
    `env_cuda.sh` per [[reference-cuda-aware-mpi]]).
  - **Per-kernel verify** (12 keys, build-serial pi): `max|Δ|==0` on every key.
    The profile markers are env-gated, so default behavior is unchanged.
  - **SYNCCHECK** build clean (no assert) on pi.
  - **GPU fidelity gate** PASSES at noise floor on `scripts/gpu_fidelity_gate.sh`
    (CORE2-active-ice CUDA-vs-Serial). **Load-bearing** per
    [[feedback-gpu-fidelity-gate]] — pi alone misses ice and was the trap that
    let the M5.9 stale-host class survive ~8 commits.
  - **Same-day overhead baseline**: one `job_profile_core2_dist8` run with
    `FESOM_STEP_PROFILE` UNSET vs SET. Overhead < 1% unset; phase totals
    (FCT/momentum/…) match the pre-M5.11 baseline within noise when set.
- **No code optimization in scope.** If R1/R2/R3 surface an obvious quick win
  (e.g. a stray host-staged copy), it is documented in `PROFILE_M59.md` as part
  of the M5.12 lever — **not** patched in M5.11. The point is the data.
- **Maintain `docs/KOKKOS_PORTING_LESSONS.md` as a first-class deliverable** —
  the lesson from M5.11 ("the bound label drives the lever, not intuition")
  gets appended to the log in the same commit as `PROFILE_M59.md`.
- **Same-day apples-to-apples is the only honest perf number** — the M5.10
  reverted-Lever-C postmortem (`docs/LEVER_C_PLAN.md §RESULT`) makes this
  unambiguous. Re-run pre-M5.11 baseline on the same hardware on the day of the
  R1 runs; do not compare to memory-recorded numbers from another week.

## Testing Strategy

Mapped to the FESOM port's existing validation infrastructure (no new test
harness; the gates below already exist):

- **Per-kernel verify (`FESOM_KK_VERIFY=<key>`)**: gates kernel correctness on
  `build-serial` (the bit-identity oracle). Runs after C1 instrumentation.
- **SYNCCHECK** (`build-synccheck`): gates DualView sync discipline. Runs after
  C1.
- **GPU fidelity gate** (`scripts/gpu_fidelity_gate.sh`): the load-bearing CUDA
  correctness gate. Runs after C1.
- **Overhead baseline**: a paired same-binary run with the env unset vs set —
  documents the marker cost.
- **Climate gate is NOT in scope** — M5.11 does not change numerics, so the
  existing M3.2 climate fidelity already covers it.

## Progress Tracking

- Mark completed items with `[x]` immediately when done.
- Add newly discovered tasks with ➕ prefix.
- Document issues/blockers with ⚠️ prefix.
- Update this plan if implementation deviates from original scope (the M5.10
  precedent: when Lever C tanked, `LEVER_C_PLAN.md` was updated with a §RESULT
  section recording why — same convention here).

## What Goes Where

- **Implementation Steps** (`[ ]`): source edits, job scripts, doc files, commits.
- **Post-Completion** (no checkboxes): slurm wall-time-bound runs that produce
  the input data for `PROFILE_M59.md`. The runs themselves are sequenced inside
  the plan (Task C2.1), but their wall-time success is reported, not gated.

## Implementation Steps

## ───────────── C1 (M5.11-a): Instrumentation + slurm jobs ─────────────

> **Design pivot (during implementation, 2026-05-28):** instead of manually
> wrapping ~80–100 `parallel_for` sites with `FPROF_KBEG`/`FPROF_KEND`, we
> register Kokkos profiling-tools callbacks for `parallel_{for,reduce,scan}`
> begin/end and let Kokkos auto-instrument every kernel by its label. This
> drops 6 file-edit tasks (Tasks C1.3–C1.8 in the original plan) in favour of
> a single source change in `fesom_profile.cpp`. The kernel label (first arg
> to `parallel_for`) becomes the bucket name automatically; existing labels
> (`fct_init_zero`, `momentum_adv_horiz`, …) are already descriptive. The
> design trade-off: in profile-on mode, each `parallel_for` now fences at
> begin + end (so end-time = kernel completion, not launch return). Cost on a
> CORE2 dist_8 GPU step at ~80–100 launches/step ≈ 0.1–2 ms = < 0.5% of the
> 477 ms step. Zero cost when env unset.

### Task C1.1: Extend `fesom_profile` with NVTX bridge + per-kernel + deep_copy callbacks

**Files:**
- Modify: `src/fesom_profile.hpp`
- Modify: `src/fesom_profile.cpp`

- [x] add `push_region` / `pop_region` helpers wrapping `Kokkos::Profiling::pushRegion` /
      `popRegion` — bridges to NVTX automatically when nsys is attached with `-t nvtx`.
- [x] add `fesom_prof::install_callbacks()` that registers **6 + 1 callbacks** via
      `Kokkos::Tools::Experimental`:
      `set_begin/end_parallel_for_callback`,
      `set_begin/end_parallel_reduce_callback`,
      `set_begin/end_parallel_scan_callback`,
      `set_begin_deep_copy_callback`. The same `kernel_begin_cb` / `kernel_end_cb`
      handles all three kernel-launch types (same `beginFunction` typedef).
- [x] begin callback: `Kokkos::fence`; `pushRegion(name)`; stash `{name, MPI_Wtime()}` on
      a `thread_local std::vector` stack. End callback: `Kokkos::fence`;
      accumulate elapsed into `g_secs[name]`; `popRegion`; pop the stack.
- [x] deep_copy callback: two `std::atomic<uint64_t>` counters (`g_dc_count`,
      `g_dc_bytes`). Atomics hold no Kokkos state → **no cleanup-before-finalize
      needed**, unlike the M5.8 EVP `static Kokkos::View` trap.
- [x] extend `report()` to print per-step deep_copy stats (`calls/step`, `MB/step`).
- [x] include `<impl/Kokkos_Profiling.hpp>` + `<impl/Kokkos_Profiling_C_Interface.h>`
      (verified always-on in bundled Kokkos 4.0 — no `#ifdef KOKKOS_ENABLE_PROFILING`
      guard needed).
- [x] simplify `fesom_profile.hpp`: keep the existing one-arg `FPROF_BEG(t)` and
      two-arg `FPROF_END(t, name)` phase-scope macros; remove the `FPROF_KBEG/KEND`
      kernel-scope macros from the prior design (no longer needed — auto-instrument).

### Task C1.2: Wire `install_callbacks` into `fesom_main`

**Files:**
- Modify: `src/fesom_main.cpp`

- [x] add `fesom_prof::install_callbacks()` immediately after `Kokkos::initialize`
      (next to the existing M5.8 EVP-free cleanup pattern).

### Task C1.3: Smoke-test auto-instrumentation

- [ ] build `build-cuda/fesom_port` (env: `source env_cuda.sh`).
- [ ] run a short `FESOM_STEP_PROFILE=1` pi smoke (e.g. 20 steps); confirm
      `[fesom_prof]` output shows per-kernel buckets with the existing Kokkos
      labels (`fct_init_zero`, `momentum_adv_horiz`, …) AND the phase buckets
      from the PMARK chains (`ocean.*`, `ice.*`, …) AND a `deep_copy:` line
      with non-zero `calls/step` and `MB/step`.
- [ ] verify the call counts make sense (e.g. `fct_init_zero calls/step` matches
      "called once per FCT advect tracer call" × number of tracers).

### Task C1.4: Build clean on all 3 backends

- [ ] `make -j -C build-serial fesom_port`
- [ ] `make -j -C build-omp fesom_port`
- [ ] `make -j -C build-cuda fesom_port` (env: `source env_cuda.sh` first)
- [ ] inspect compiler warnings — any new ones from the markers must be benign.

### Task C1.5: Validation gates

- [ ] **Per-kernel verify**: loop `k ∈ {eos, pp, kpp, pgf, vrhs, vfilt, ivisc,
      ale, gm, tradv, trdiff, ssh}`; run `FESOM_KK_VERIFY=$k build-serial/fesom_port`
      on pi smoke. All `max|Δ| == 0`. (Markers env-gated → default unchanged.)
- [ ] **SYNCCHECK**: pi run on `build-synccheck/fesom_port`. No SYNCCHECK assert.
- [ ] **GPU fidelity gate**: `bash scripts/gpu_fidelity_gate.sh`. PASS at noise
      floor. **Load-bearing.** The gate script defaults to CORE2-active-ice
      (JRA55=1958) — verify the gate run log shows CORE2 + active-ice, **not**
      pi. The M5.9 stale-host class only manifests on CORE2-active-ice and was
      invisible to pi for ~8 commits (lesson L50, `feedback-gpu-fidelity-gate.md`).
- [ ] **Overhead baseline**: one `job_profile_core2_dist8` run with
      `FESOM_STEP_PROFILE` UNSET, one with SET. Loop s/step overhead < 1% unset;
      with set, phase totals (FCT/momentum/…) match pre-M5.11 baseline within
      noise.

### Task C1.6: Add 6 slurm jobs

**Files:**
- Create: `jobs/job_profile_core2_dist1`
- Create: `jobs/job_profile_core2_dist4`
- Create: `jobs/job_profile_core2_dist8`
- Create: `jobs/job_nsys_core2_np1_v2`
- Create: `jobs/job_nsys_core2_dist4`
- Create: `jobs/job_ncu_top5`

- [ ] template from `jobs/job_gpuaware_prof_core2`: `source env_cuda.sh`,
      UCX/MPI env, JRA55=1958 (active-ice trigger), output under
      `/work/ab0995/a270088/port2/kokkos_profile_m511/<tag>/` per
      [[feedback-hpc-run-hygiene]].
- [ ] `dist1` = np=1 1 A100 (`gpu-devel`), 200 steps, `FESOM_STEP_PROFILE=1`.
- [ ] `dist4` = 4 A100 / 1 GPU node, 200 steps.
- [ ] `dist8` = 8 A100 / 2 GPU nodes, 200 steps (refresh of current baseline).
- [ ] `nsys_np1_v2` = np=1, `nsys profile -t cuda,nvtx,mpi,osrt -o prof
      --force-overwrite=true --stats=true`, ~30 steps (refresh of the older
      `-t cuda`-only job; NVTX picks up our pushRegion names).
- [ ] `nsys_dist4` = dist_4 nsys, ~30 steps (halo MPI visible).
- [ ] `ncu_top5` = np=1, `ncu --set full --kernel-name regex:<top-5-from-R1>
      -o prof --force-overwrite=true`, 3 steps. The regex is filled in C2.1
      after R1 picks the top-5. **Fallback if `--set full` OOMs on FCT** (the
      largest working set): `--set roofline` + specific sections
      (`MemoryWorkloadAnalysis,SchedulerStats,Occupancy,WarpStateStats`).
      Document the fallback choice in `PROFILE_M59.md §1`.

### Task C1.7: Commit C1 (M5.11-a)

- [ ] single commit on `profile-m511`: fesom_profile extension + 79 wrapped
      sites + main wiring + 6 new jobs.
- [ ] commit message follows the M5.x convention (see
      `git log master --oneline -10` for shape).

## ───────────── C2 (M5.11-b): Runs + synthesis doc ─────────────

### Task C2.1: Submit timing runs (R1a/b/c), pick top-5

- [ ] `sbatch jobs/job_profile_core2_dist1` — np=1 reference (no halos).
- [ ] `sbatch jobs/job_profile_core2_dist4` — intra-node halos only.
- [ ] `sbatch jobs/job_profile_core2_dist8` — cross-node halos (current baseline).
- [ ] collect outputs; tabulate per-kernel s/step + %step + launches/step,
      sorted by dist_8 % share.
- [ ] identify top-5 kernels for ncu drill-down; update `job_ncu_top5` regex.

### Task C2.2: Submit nsys + ncu runs (R2 + R3)

- [ ] `sbatch jobs/job_nsys_core2_np1_v2` — timeline with our NVTX markers; refreshes
      the stale "13 GB/step / cudaMemcpy 83% of API time" stat.
- [ ] `sbatch jobs/job_nsys_core2_dist4` — halo MPI visible alongside kernels.
- [ ] `sbatch jobs/job_ncu_top5` — kernel-level metrics
      (`MemoryWorkloadAnalysis`, `SchedulerStats`, `Occupancy`,
      `WarpStateStats`) on the top-5.

### Task C2.3: Host residual audit

- [ ] walk `src/fesom_step.cpp`: count `sync_host()` / `sync_device()` calls per
      step. Anything non-zero outside the documented L48 pre-I/O snapshot guard
      or the M5.9-pinned `uvnode` is a suspect.
- [ ] grep for raw `for (` host loops outside `parallel_for` in the 6 hot-phase
      files: `grep -nE '^\s*for\s*\(' src/fesom_{tracer_adv,momentum,gm,ice_evp,kpp,tracer_diff}.cpp`.
- [ ] `grep -rn 'smooth_nod3D' src/` — confirm M5.5 covered every call site (blmc-class
      catcher).
- [ ] check the M5.4 leftovers flagged as low/uncertain payoff
      (GM chain owned-reads, `ssh_rhs` nod2D, tracer-diff): if any surface in
      R1 as fat, lift them in the M5.12 lever.

### Task C2.4: Compose `docs/PROFILE_M59.md`

**Files:**
- Create: `docs/PROFILE_M59.md`

- [ ] §1 Setup — binary commit hash, configs, jobs, output paths.
- [ ] §2 R1 per-kernel table — columns np=1 / dist_4 / dist_8, sorted by dist_8
      % share.
- [ ] §3 Halo wall — `(R1c − R1b)` and `(R1b − R1a)` per phase →
      label IB-bound / intra-node-bound / compute-bound.
- [ ] §4 Top-5 deep dive — for each kernel: nsys idle-gap %, ncu bound label
      (mem-bw util %, achieved occupancy, warp stall reasons, branch divergence),
      phase share, expected lever.
- [ ] §5 Host audit — findings from C2.3 with verdict per item.
- [ ] §6 THE lever (single M5.12 proposal) — kernels targeted, mechanism
      (fusion / scatter coloring / TeamPolicy / mixed-precision / etc.), payoff
      range (low / med / high), validation strategy, risk.
- [ ] §7 Discarded levers — what the data ruled out + why (the M5.10 §RESULT
      precedent).

### Task C2.5: Update `KOKKOS_PORTING_LESSONS.md`

**Files:**
- Modify: `docs/KOKKOS_PORTING_LESSONS.md`

- [ ] append a single lesson entry (L51 or next free) — "M5.11: the bound label
      drives the lever — without per-kernel timing + ncu, the optimization is a
      guess; Lever C cost a session of work to learn this. The deep_copy
      counter is the blmc-class catcher for PCIe traffic that fence-bounded
      phase timers miss."

### Task C2.6: Commit C2 (M5.11-b)

- [ ] commit `PROFILE_M59.md` + `KOKKOS_PORTING_LESSONS.md` updates.
- [ ] raw slurm logs stay under `/work/ab0995/a270088/port2/kokkos_profile_m511/`
      (NOT committed; the synthesis doc cites concrete numbers from them).

## ───────────── C3 (M5.11-c): M5.12 plan draft ─────────────

### Task C3.1: Compose `docs/M512_PLAN.md`

**Files:**
- Create: `docs/M512_PLAN.md`

- [ ] follow the `docs/LEVER_C_PLAN.md` shape:
      claim → why-this-and-not-others → validation gates → sub-milestones →
      payoff target.
- [ ] reference `PROFILE_M59.md §4` + `§6` as the evidence base.
- [ ] sub-milestones are M5.12a, M5.12b, … each independently shippable, each
      with its own per-kernel verify gate + same-day baseline + payoff target.

### Task C3.2: Commit C3 (M5.11-c) — close milestone

- [ ] commit `M512_PLAN.md`.
- [ ] update task list in this plan with `[x]` for everything completed.
- [ ] move this plan file to `docs/plans/completed/20260528-m511-profile-pass.md`
      (mkdir if needed). Done.

## Technical Details

### Naming convention

Phase-prefixed kernel labels, lowercase, dot-separated: `fct.<name>`,
`momentum.<name>`, `gm.<name>`, `evp.<name>`, `kpp.<name>`, `trdiff.<name>`.
The sorted report (rank-0, `report()` in `fesom_profile.cpp`) groups by phase
when columns are sorted by name, by share when sorted by `%loop` (default).

### Site inventory (initial Explore agent, 2026-05-28 — preliminary, exact count emerges in C1)

| File | Sites | Function(s) | Atomic-add scatters (lines) | Line range |
|:-----|:-----:|:-----------|:----------------------------|:-----------|
| `src/fesom_tracer_adv.cpp` | 28 | `fesom_tracer_advect_one_fct_kk` (FCT incl. LO/MFCT/QR4C/Zalesak/F2D) | 1521, 1737, 1811 | 1449–1831 |
| `src/fesom_momentum.cpp` | 14 | `compute_vel_rhs`, `momentum_adv`, `visc_filt_bidiff`, `impl_vert_visc`, `update_vel`, hbar | 356, 1426, 1460, 1654 | 317–1691 |
| `src/fesom_gm.cpp` | 14 | GM gradients, `fer_solve_gamma` TDMA, bolus apply, Redi gather + scatter | 1921 | 1111–1921 |
| `src/fesom_ice_evp.cpp` | 11 | `stress_tensor`, `stress2rhs`, EVP step 1–4, `vel_update`, coastal-BC | 544, 653 | 494–727 |
| `src/fesom_kpp.cpp` | 11 | Ri/iw mix, bldepth, blmix slab, prestep, combine, viscAE | — | 359–1589 |
| `src/fesom_tracer_diff.cpp` | 1 | `impl_vert_diff_tracers_kk` TDMA | — | 438 |
| **Total** | **79** | | **10 atomic_add scatters** | |

Explicitly skipped — known shape, < 7% phase share:
`src/fesom_ssh.cpp` (9 sites, the CG + SSH RHS; design-bound by CG iteration
count) and `src/fesom_ale.cpp` (9 sites, single-pass per step including 1
element→node scatter).

### Kokkos profiling API choices (verified)

- `<impl/Kokkos_Profiling.hpp>` exposes `Kokkos::Profiling::pushRegion(const
  std::string&)` / `popRegion()` via using-decls from `Kokkos::Tools`. Always
  compiled — no `#ifdef KOKKOS_ENABLE_PROFILING` guard.
- `<impl/Kokkos_Profiling_C_Interface.h>` defines `Kokkos_Profiling_SpaceHandle`
  (just `char name[64]`) and the `beginDeepCopyFunction` callback signature.
- Register the callback via
  `Kokkos::Tools::Experimental::set_begin_deep_copy_callback(&dc_begin_cb)`
  after `Kokkos::initialize` returns.
- Counters: `std::atomic<uint64_t> g_dc_count`, `g_dc_bytes` — atomic because
  Kokkos may fire the callback from any thread; contention is irrelevant
  (the volumes are tiny).
- The callback fires for ALL deep_copies (host↔device, device↔device, host↔host).
  Initial report does NOT filter by direction — total + count is a useful upper
  bound. Filtering by `SpaceHandle.name` is a future refinement.

### EVP bucket aggregation

The 120-subcycle EVP loop reuses 4 kernel labels per subcycle
(`evp.stress_tensor`, `evp.s2rhs_scatter`, `evp.vel_update`,
`evp.coastal_bc`). `fesom_prof` already buckets by name — the per-step report
shows `evp.vel_update s/step=… calls/step=120`. No per-subcycle index suffix
is needed; would inflate the report from ~80 lines to ~600.

### Output paths

All `/work/ab0995/a270088/port2/kokkos_profile_m511/`:
- `dist1/`, `dist4/`, `dist8/` — timing runs (R1).
- `nsys_np1/`, `nsys_dist4/` — `prof.nsys-rep` + `.sqlite`.
- `ncu_top5/` — per-kernel `prof.ncu-rep` reports.

Per [[feedback-hpc-run-hygiene]]: **never $HOME** (60 GB quota; a CORE2 GPU run
is 3.5 GB; nsys traces can reach several GB).

## Post-Completion

*Items requiring slurm wall-time (sequenced inside C2 but the wall-time gates
the docs):*

- **R1a/b/c timing** — ~30 min total slurm queue + execute (3× 200-step runs).
- **R2 nsys × 2** — ~30 min total (each ~30 steps).
- **R3 ncu top-5** — ~2 h (ncu replays each kernel, very slow per kernel —
  the agent's per-kernel time estimate is 20 min × 5 = 100 min).

If `gpu` partition is contended on the day, the np=1 run can drop to `gpu-devel`
(it's a single-A100 run that fits within the devel partition wall-time).

**External system updates**: none — M5.11 has no external dependency change.
