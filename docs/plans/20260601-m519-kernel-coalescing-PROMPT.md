# Next session — M5.19: generalize the coalescing lever to the remaining node/element-major kernels

*Paste this whole file to start. Self-contained. Written 2026-05-30 at the close of M5.18 (smoother coalescing).*

---

## 0. TL;DR — where we are

The whole FESOM2 **C→C++/Kokkos** port is device-resident and validated (M0–M4 Serial-bit-identical; M5.x GPU-perf campaign). **M5.18 just landed the coalescing lever on the #1 GPU kernel** — the horizontal smoother `fesom_smooth_nod3D_kk` was re-parallelized from one-thread-per-(slab,**node**) [internal level loop, uncoalesced node-major stores] to one-thread-per-(slab,node,**level**) [the level is the contiguous dim → coalesced]. Result: kernel **15.5× faster**, **NG5 dist_16 step −14.2 % (2.4823→2.1296 s/step)**, bit-identical (Serial+OpenMP `max|Δ|=0`, pi np1+np2 identical, CUDA gate PASS, 1-yr climate ≡ M5.16). Committed `7277a83` on branch `m517-mpi-comms` (M5.17 instrument = `2292bc0`). Full story: `docs/GPU_FIDELITY.md` §M5.18, lesson **L63**, [[project-m518-smoother-compute]].

**This session: generalize the lever to the next-biggest GPU kernels — BUT they are NOT all the same shape.** The pre-work below (a code-read of each candidate, done at the M5.18 close) found that the smoother was *uniquely simple*: its level loop was a **pure independent per-level map**. Most other node-major kernels are **per-column** (column-local scratch / multi-pass / vertical coupling) or **TDMAs** (sequential level dependency) or **scatters** — each needs a *different* coalescing technique. **The main job this session is to apply the RIGHT technique per kernel class, not the same flat lever everywhere.**

---

## 1. Do we need to re-profile first? NO (but do it as a cheap sanity step 0)

M5.18 touched ONLY the smoother, so every other kernel's **absolute** nsys time is unchanged → the pre-M5.18 ranking *minus the smoother* is already the valid target list. The deciding input is **kernel structure** (code-read), not the profile. Still, run **`jobs/job_ng5_prof`** (per-phase) + **`jobs/job_nsys_ng5`** (kernel ranking) once at the start to (a) confirm the smoother dropped out of the top (sanity-check the M5.18 win at the kernel level), and (b) get exact post-M5.18 shares. Expect: smoother now ~2–3 %; the new top kernels = FCT, `compute_vel_rhs`, GM `compute_sigma`/`neutral_slope`, `diff_ver`/`impl_vert_visc`.

---

## 2. THE KEY CLASSIFICATION (done at M5.18 close — verify, then act)

Every node/element-major kernel falls into one of four buckets. **The bucket dictates the technique.**

| bucket | how to recognize (code) | coalescing technique | examples (verify each) |
|--|--|--|--|
| **A. pure per-level map** | `RangePolicy(0, N or E)` + an internal `for(nz)` where each level is computed from **same-level / fixed-per-column** reads; no column scratch, no vertical scan | **The M5.18 flat lever**: `RangePolicy(0, N*nl)` decode `(n,nz)`, mask out-of-column, one level/thread. Bit-identical (register-accumulate same order). | `compute_vel_rhs` (`fesom_momentum.cpp:473` `vel_rhs_elem` + `:517` `vel_rhs_assembly` — two independent level loops, element scalars `Fx/Fy/ff` recomputed per thread = cheap); `momadv_vert`/`momadv_area`; the already-flat `fct_init_*`, GM `bolus_*` |
| **B. per-column** | internal `for(nz)` with **column-local scratch** (`c1[NL_MAX]`), **multiple passes** that read each other, or **vertical neighbor coupling** that isn't a strict down-column recurrence | **TeamPolicy**: league = node/elem, team threads = levels, `c1[]`-style scratch in **team-shared memory**; OR keep per-node + **Lever C layout** (§5). NOT the flat lever (would break the column scratch). | GM `neutral_slope` (`fesom_gm.cpp:1191` — `c1[NL_MAX]` + Pass1/Pass2/taper); GM `sigma_xy` (verify); FCT Zalesak `fct_zal_a34` (per-column min/max reduction) + `fct_qr4c_v` (vertical reconstruction) |
| **C. TDMA (level-dependent)** | a down-then-up column sweep (forward elimination + back substitution); level `nz` depends on `nz-1` | **Lever C ONLY** (§5): the per-level lever CANNOT apply (sequential dependency). Coalesce by making consecutive nodes' same-level values contiguous (`View<double**>`), keeping one-thread-per-node. | `diff_ver` (`fesom_tracer_diff.cpp:438` `impl_vert_diff_tracers` — the M2.7 per-node TDMA); `impl_vert_visc` (`fesom_momentum.cpp:815`, per-elem TDMA); GM `fer_solve_gamma` (`fesom_gm.cpp:1401`, per-node TDMA) |
| **D. scatter** | `Kokkos::atomic_add` into a node/elem array from an **edge/element loop** | Different axis (edge-coloring, or Lever C layout for the target). Leave for last; the atomics are correctness-load-bearing (D22). | momentum `momadv_horiz`/`visc_bidiff`; FCT `fct_LO_scatter`/`fct_zal_b1h`/`b3h`; GM redi edge scatters |

**Order of attack (best ROI first):** (1) **bucket A** kernels — same proven recipe, bit-identical, low risk. Start with `compute_vel_rhs` (~3 %, confirmed pure-map). (2) **bucket B** via TeamPolicy — `neutral_slope`+`sigma_xy` (~3 %), more involved but big. (3) FCT — heterogeneous: classify each of its ~24 sub-kernels into A/B/D and treat per-bucket (the single biggest fn, ~6 %, but mixed). (4) **bucket C/D** only if A+B plateau → that's really **Lever C** territory (§5).

---

## 3. THE RECIPE per bucket

### Bucket A — the M5.18 flat lever (proven, bit-identical)
Mirror `fesom_smooth_nod3D_kk` exactly (`src/fesom_eos.cpp:488`):
- `RangePolicy(0, N*nl)` (or `E*nl`); decode `n = idx/nl`, `nz = idx - n*nl` (element-major: `e = idx/nl`).
- Mask `nz < uln(n) || nz > nlnz(n)` → `return` (or the element's `ulev/nlev`).
- Compute the level's value into **registers**, write once. The per-element/per-node scalars (e.g. `Fx,Fy,ff` in `vel_rhs`) **recompute per thread** — a few FLOPs, cheap vs the coalescing win; only hoist via TeamPolicy if ncu shows it matters.
- **Bit-identity:** the per-(n,nz) arithmetic must run in the SAME order as the C twin (register accumulator ≡ in-memory, IEEE op-for-op). No scatter → Serial AND OpenMP `max|Δ|=0`.

### Bucket B — TeamPolicy (column scratch / multi-pass)
- `TeamPolicy(league = N, team = Kokkos::AUTO)`; `TeamThreadRange(team, uln, nlnz)` for the per-level work; `team.team_barrier()` between passes; allocate `c1[]`-style column scratch in `team_scratch(0)` (per-team shared). Coalescing comes from the team's threads striding consecutive levels (contiguous in node-major). Keeps the multi-pass / column-reduction structure intact → bit-identical. More complex; measure it beats the per-node baseline before committing.

### Bucket C — defer to Lever C (§5). Do NOT force the per-level lever onto a TDMA.

---

## 4. VALIDATION LADDER (identical discipline to M5.18 — every kernel)

Each kernel touched is arithmetically identical → **Serial bit-identity is the gate**.
1. **Per-kernel verify.** Reuse the existing `FESOM_KK_VERIFY=<key>` for the kernel's owning module (vel_rhs→`vrhs`, GM→`gm`, FCT→`tradv`, tracer-diff→`trdiff`); confirm Serial AND OpenMP `max|Δ|=0`. If the existing key doesn't isolate the specific sub-kernel tightly, add a capture-before hook like **`fesom_smooth_nod3D_kk_verify`** (`fesom_eos.cpp` — the M5.18 template: snapshot input → run device → run host C twin → diff owned).
2. **pi np1 + np2 BIT-IDENTICAL** vs `docs/reference/c_baseline_snapshots/pi` + `/scratch/a/a270088/pi_np2_ref_m13_nocma` (np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`, L18; `diff_snap.py` takes DIRECTORIES, L19). ⚠️ check the kernel actually runs in pi (GM is ON in pi, L34; FCT runs; ice is CORE2-only, L42).
3. **SYNCCHECK** clean (`build-synccheck`).
4. **CUDA gate** `scripts/gpu_fidelity_gate.sh --fresh-oracle` (rebuild oracle after editing a Serial-shared file). PASS = the climate-close floor (M5.18 worst 7.7e-3).
5. **Same-day SAME-NODE A/B** for the s/step delta: clone **`jobs/job_ng5_m518_ab`** (it runs BEFORE+AFTER binaries back-to-back in ONE allocation — the only valid baseline, [[feedback-perf-same-day-baseline]]). The before/after-binary build recipe: save the M5.19 diff as a patch, `git checkout --` the files, build-cuda → cp to `fesom_port_before`, restore the patch, build-cuda → cp to `fesom_port_after`.
6. **ncu before/after** per kernel: clone **`jobs/job_ncu_smooth_ab`** + set `NCU_REGEX` to the kernel's fn name (`ncu_rank0.sh` `NCU_METRICS` already captures the sectors/req coalescing counter that `--set basic` drops). Target: STORE sec/req ↓, occupancy ↑, SM-util ↑ (the M5.18 signature: 29.5→6.9, 52→91 %, 2.2→59 %).
7. **1-yr CORE2 CUDA climate** to close (`jobs/job_m32_cuda_core2` 1-yr; `scripts/m32_climate_compare.py <dir> --years 1958` vs Fortran+C, and `--cref m32_cuda_m518_1yr` for the apples-to-apples zero-cost check). Bit-identical-on-Serial ⇒ expect corr=1.00000 vs the prior binary.

---

## 5. Lever C — the heavyweight layout refactor (the LAST resort + the only path for buckets C/D)

The TDMAs (bucket C) and scatters (bucket D) **cannot** be coalesced by re-parallelizing levels. Their only coalescing path is the global layout change: `fesom_field.hpp` rank-1 (`arr[node*nl+nz]`, node-major) → **`View<double**>` with the LEVEL as the slowest/outer dim** so consecutive *nodes'* same-level values are contiguous → a per-node-thread warp coalesces. This touches **all 126 fields + every kernel** (high risk, separate branch, careful staged validation). **Only escalate here once the local bucket-A/B wins plateau** — the M5.18 result + the bucket-A/B wins this session tell you whether the local approach is enough. Estimate the remaining headroom first (sum the bucket-C/D kernel shares from step-0 nsys); if it's small, Lever C may not be worth the risk.

## 6. The other track — residual PCIe `deep_copy` = 3.41 GB/step (34 % of CUDA API time)

Orthogonal to coalescing. Residency was declared exhausted (M5.15/16) but 3.4 GB/step still moves. Attribute it per-field (`FESOM_SYNC_LOG` rail in `fesom_field.hpp` + the `FESOM_STEP_PROFILE` deep_copy counter): it's the forcing HtoD (8 JRA55 fields) + verify-gated syncs + Phase-A bulk output round-trips. **Phase B** (fully device-resident forcing, deferred from M5.16) reclaims part of it. Cheaper + lower-risk than Lever C; consider interleaving.

## 7. HARD CONSTRAINTS (carry every session)
- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB quota). Big/NG5/CORE2 runs via SLURM, never login.
- ⚠️ **NG5 perf jobs write ~50 GB `*.monthly.nc` even at `snap_every=-1`** — `rm <dir>/*.monthly.nc` after each (the job templates do this).
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; env.sh's 4.1.2 SEGFAULTs on device ptrs, L47). ⚠️ `env_cuda.sh` PURGES `git` — do git ops in a separate shell. CPU builds use `env.sh`. Build dirs `build-cuda`/`build-serial`/`build-synccheck` carry M5.18; `build-omp` was rebuilt M5.18.
- **Same-day same-node perf baselines only** ([[feedback-perf-same-day-baseline]]). The absolute s/step varies ~7 %/day by node mix (M5.18 same-node before was 2.48, vs §M5.17's 2.68 — same binary, different day).
- **Device/kernel changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); pi is insufficient (no ice). **Commit/push only when the user asks.** KPP is the default mix_scheme ([[feedback-kpp-default]]).

## 8. POINTERS
- **Memory:** [[project-m518-smoother-compute]] (the proven lever + the M5.18 commits), [[project-m517-mpi-comms]], [[feedback-perf-same-day-baseline]], [[feedback-gpu-fidelity-gate]], [[reference-cuda-aware-mpi]], [[reference-build-run]].
- **Docs:** `docs/GPU_FIDELITY.md` §M5.18 (the lever + ncu method) + §M5.13–17 (the arc), `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–**L63**), `docs/SCALING_NG5.md`.
- **The M5.18 template (copy this):** `fesom_smooth_nod3D_kk` (`src/fesom_eos.cpp:488`) = the flat-lever kernel; `fesom_smooth_nod3D_kk_verify` (same file) = the capture-before per-kernel verify.
- **Tooling (clone + retarget via `NCU_REGEX`/binary path):** `jobs/job_ng5_m518_ab` (same-node s/step A/B), `jobs/job_ncu_smooth_ab` (ncu A/B w/ sectors/req), `jobs/job_ng5_prof` + `jobs/job_nsys_ng5` (re-profile), `jobs/job_ng5_halo_split` (the M5.17 barrier gate).
- **State:** branch `m517-mpi-comms`, HEAD `7277a83` (M5.18). NOT pushed/merged to master/tagged. `master` is at `m5.16-bulk-port`. The validated M5.18 binary = `build-cuda/fesom_port` (and `fesom_port_m518`); `fesom_port_m516` is the pre-M5.18 binary, kept for re-measurement.

## 9. Bottom line
The coalescing lever is **proven** (M5.18: −14.2 % from one kernel, bit-identical). But the smoother was a pure per-level map; the next kernels split into **A pure-map** (flat lever — start with `compute_vel_rhs`), **B per-column** (TeamPolicy — `neutral_slope`/`sigma_xy`), **C TDMA** (Lever C only — `diff_ver`/`impl_vert_visc`), **D scatter** (different axis). Re-profile once for exact shares, then attack A→B→FCT, holding the Serial-bit-identical gate + same-node A/B + ncu before/after on every kernel. Escalate to the heavyweight Lever C layout only if A/B plateau, and weigh the 3.41 GB/step PCIe track in parallel. Don't claim a win until it's same-day measured + bit-identical + gate PASS + climate-validated.
