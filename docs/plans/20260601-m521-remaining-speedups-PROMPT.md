# Next session — M5.21: the rest of the GPU-speedup levers that DON'T touch the climate

*Paste this whole file to start. Self-contained. Written 2026-05-30 at the close of M5.20 (PCIe residency).*

---

## 0. TL;DR — where we are

The whole FESOM2 **C→C++/Kokkos** port (ocean + sea-ice) is device-resident and validated (M0–M4 Serial-bit-identical, tagged; M5.x = the GPU-perf campaign). **`master` is at annotated tag `m5.20-pcie-residency` (`4b4da79`; NO git remote → local-only).** Recent landings (all committed on master):
- **M5.18** (`7277a83`): the **coalescing lever** — re-parallelized the #1 GPU kernel `fesom_smooth_nod3D_kk` to one-thread-per-(node,**LEVEL**) (the contiguous inner dim → coalesced). −14.2 % NG5 step, byte-identical. L63, §M5.18.
- **M5.19** (`e3454fe`): **generalized** it to `compute_vel_rhs` + GM `sigma_xy`/`neutral_slope` (6 maps). −5.35 %. KEY: GM was flat-lever-able, not TeamPolicy. L64, §M5.19.
- **M5.20** (`047c5e8`): the **PCIe track** — measured per-field (`FESOM_SYNC_LOG`), which **refuted Phase-B forcing**, then made `hnode_new` + `sw_3d` device-resident. deep_copy 3413→2176 MB/step (−36 %), step −17.6 %. L65, §M5.20.

**NG5 dist_16 ≈ 1.77 s/step** now ≈ **2.4× faster than a CPU node** (node-for-node; arc: campaign-start 3.76× SLOWER → now ~2.4× faster). The big single kernels (smoother 25.7 %, vel_rhs+GM ~5 %) and the two fat PCIe round-trippers (hnode_new, sw_3d) are done.

**This session: harvest the REMAINING speedups — every lever below preserves the physics** (Serial bit-identical OR climate-close at the established floor; the CUDA fidelity gate + the 1-yr climate confirm ZERO climate cost, exactly as M5.18/M5.19/M5.20 each did). **No algorithm/physics change — only execution (parallelization, memory layout, residency, partition).**

---

## 1. Step 0 — RE-PROFILE first (the campaign discipline; it has refuted the plan 3×)

M5.20 changed only the forcing/residency path, so the per-kernel ranking is mostly intact, but **re-measure before committing to a lever** (M5.15 re-profile, M5.17 Lever-B, M5.20 Phase-B were ALL overturned by measurement). Run on the current `build-cuda` (= m520):
- **`jobs/job_ng5_prof`** (`--nodes=4 --export=ALL,TAG=ng5prof_m520`) — per-phase + per-kernel %; expect `13_fct` ≈ #1 phase (~24 %), `3_mixing`, `12_ale`, `1b_gm`, then the smaller kernels.
- **`jobs/job_nsys_ng5`** — kernel ranking + the PCIe memops split.
- **`jobs/job_ng5_synclog`** (the `FESOM_SYNC_LOG` per-field PCIe; binary `build-cuda/fesom_port_synclog` OR rebuild it: scoped CMake `-DFESOM_SYNC_LOG=ON`) — confirm the post-M5.20 deep_copy (~2176 MB/step) is now led by **`kpp.ghats` (256) + `tracers.S` (249)** + ice-FCT values + startup.

---

## 2. The levers, prioritized (best ROI / lowest risk first)

### Lever 1 — FINISH the coalescing track (bucket-A FCT + `update_vel`) — bit-identical, proven, lowest risk
The M5.18/M5.19 flat lever, applied to the kernels classified at the M5.19 close (`docs/GPU_FIDELITY.md` §M5.19 "Remaining coalescing headroom"). **`13_fct` is the #1 phase (~24 %)** but is ~24 small mixed-bucket sub-kernels sharing scratch in ONE giant function (`fesom_tracer_adv.cpp` `fesom_tracer_advect_one_fct_kk`) — flip ONLY the **bucket-A** ones (each <0.9 %, but they add up):
- `fct_zal_a1` (per-node map), `fct_zal_a2` (per-elem map), `fct_qr4c_v` (per-node — vertical reads `valsAB[nz±2]` are INPUTS → flat-lever-able), `fct_f2d_v`, `fct_ale_recon`, `fct_grad_elem`, `fct_LO_final`, `fct_upw1v`.
- Also `update_vel` (0.82 %, `fesom_momentum.cpp:1106`, clean per-elem map, **`ssh` verify**).
- **Recipe = the M5.18 flat lever** (`src/fesom_eos.cpp:488` `fesom_smooth_nod3D_kk` is the template): `RangePolicy(0, N*nl)`, decode `(n,nz)`, mask out-of-column, register-write in the SAME order → byte-identical (Serial AND OpenMP). **Verify = `tradv`** (FCT) / `ssh` (update_vel); the shared verify covers the whole chain.
- ⚠️ Do NOT flat-lever the bucket-B (column-reduction `fct_zal_a34`, `momadv_vert`) or bucket-D (scatter `*_h`, `LO_scatter`) ones — those need TeamPolicy or Lever C (below). Re-read each kernel's data flow before flipping (M5.19 lesson: a "column scratch" of per-level accumulators IS flat-lever-able; a true cross-level reduction is NOT).

### Lever 2 — PCIe remainder: `ghats` + `S` residency (the deliberate host-stays)
The post-M5.20 PCIe (2176 MB/step) is led by `kpp.ghats` (256, D2H) + `tracers.S` (249, D2H). Both are *intentional* host-stays:
- **`ghats`** (KPP non-local term) — "stays host" since M5.7. Find its host reader; if it's verify-only/placebo → drop the sync (M5.9-pin / M5.20-hnode_new pattern); if there's a real host consumer → port it to device.
- **`S` (salinity)** — synced D2H for the **salinity floor** (the L36/L39 host clamp). ⚠️ **This is climate-load-bearing — port the CLAMP itself to a device kernel BIT-IDENTICALLY** (it's a simple `max(S, S_floor)` map; do it on device, drop the D2H). Do NOT change the clamp value/logic.
- **Recipe = the M5.20 residency flip.** ⚠️ Carry the two M5.20 traps: (a) the **step-1 IC-seed checklist** (list every device reader's substep vs the producing kernel; if a reader precedes the producer, seed at `step_n==1`; the crash signature is a step-1 `CG_kk pp·App=-nan`); (b) **the gate catches it, pi cannot** (forcing/ice not exercised under pi). Validate sw_3d-style: fresh-CORE2-oracle == saved-prior-oracle.

### Lever 3 — Lever C: the heavyweight layout refactor (the biggest STRUCTURAL remaining win)
`fesom_field.hpp` rank-1 (`arr[node*nl+nz]`, node-major) → **`View<double**>` with the LEVEL as the OUTER (slow) dim**, so consecutive *nodes'* same-level values are contiguous → a per-NODE-thread warp coalesces. **This is the ONLY path for the bucket-C TDMAs** (`impl_vert_diff_tracers` 1.61 %, `impl_vert_visc`, `fer_solve_gamma` — sequential level dependency, the flat lever cannot apply) AND it globally improves the scatters (bucket D) and every node-major kernel at once. **Climate-safe** (a bit-identical memory reorder — the arithmetic is unchanged). **HIGH risk/effort: touches all 126 fields + every kernel + every halo/scatter/index macro → separate branch, staged validation, the full ladder per stage.** Escalate here once Lever 1+2 plateau; estimate the headroom first (sum the bucket-C/D kernel shares from step-0 nsys — FCT scatters + the 3 TDMAs ≈ 5–8 %). This is where FESOM2-GPU eventually has to go for the FCT/TDMA half of the step.

### Lever 4 — Lever D: load imbalance (deployment-side, climate-safe)
M5.17 measured ~9 % load-imbalance in the halo `MPI_Waitall`. A better GPU domain decomposition (re-partition `dist_16` for balanced per-rank work) recovers part of it — **no code change, just the partition** (climate-identical). Cheap to try (regenerate the mesh partition); worth a measurement.

### (Lower priority) bucket-D scatters via edge-coloring
`momadv_horiz`, `visc_bidiff`, FCT `*_h`, `ale_vvel_scatter`, redi edges — `atomic_add` from edge/element loops. Coalesce via a different axis (edge-coloring) or fold into Lever C. Climate-close (atomic ordering). Leave for last.

---

## 3. VALIDATION LADDER (every change — the discipline that has held all campaign) 
1. **Per-kernel `FESOM_KK_VERIFY=<key>` Serial `max|Δ|==0`** (keys: `tradv` FCT, `ssh` update_vel, `kpp` ghats-chain, `trdiff`/`gm` for S, etc.). Add a capture-before isolated hook if the existing key doesn't tighten on the sub-kernel (M5.18 `fesom_smooth_nod3D_kk_verify` is the template).
2. **pi np1 + np2 BIT-IDENTICAL** (`build-serial`, vs golden + `/scratch/a/a270088/pi_np2_ref_m13_nocma`; np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`). ⚠️ pi uses ANALYTICAL forcing → it does NOT exercise sw_3d/ghats/bulk/ice — for forcing/residency changes, the real Serial check is a **fresh CORE2 Serial oracle diffed vs a saved prior oracle** (`cp -r serref_core2 serref_saved` before `gpu_fidelity_gate.sh --fresh-oracle`, then `diff_snap.py`).
3. **SYNCCHECK** clean (`build-synccheck`).
4. **CUDA fidelity gate** `scripts/gpu_fidelity_gate.sh [--fresh-oracle]` (CORE2 dist_8 ice-active) — PASS = the climate-close floor (~1e-3..1e-2; M5.20 was 3.87e-3). **THIS is the gate that catches stale-host/step-1-seed bugs; pi cannot** ([[feedback-gpu-fidelity-gate]]). If the dev leg crashes with only `snap_000000` → a step-1 NaN (see the step-1-seed checklist).
5. **Same-day SAME-NODE A/B** for the s/step delta (clone `jobs/job_ng5_m520_ab`: BEFORE=`fesom_port_m520`, AFTER=your new build, back-to-back in ONE allocation — the only valid baseline, [[feedback-perf-same-day-baseline]]). Also read the `FESOM_STEP_PROFILE` deep_copy line for the PCIe delta. ⚠️ **deep_copy bytes ≠ wall-clock** — prioritize fields/kernels by per-step round-trips/pipeline-stalls, not byte count (L65: 742 MB→−11.4 %, ~8× bandwidth).
6. **ncu before/after** per coalescing kernel (`jobs/job_ncu_m519_ab`, set `NCU_REGEX` to the enclosing fn; `scripts/ncu_coalesce_summary.py` parses it). Target: STORE/LOAD sectors/req ↓, SM-util ↑, occ ↑.
7. **1-yr CORE2 CUDA climate to close** (`jobs/job_m32_cuda_core2`, `M32_NSTEPS=17280 M32_TAG=_<tag>`; `scripts/m32_climate_compare.py <dir> --years 1958 --cref m32_cuda_m520_1yr` for the apples-to-apples zero-cost check + the default cref/fref for the C/Fortran budget). Serial-bit-identical ⇒ expect corr=1.00000 vs m520.

---

## 4. HARD CONSTRAINTS (carry every session)
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; env.sh's 4.1.2 SEGFAULTs on device ptrs, L47). ⚠️ `env_cuda.sh` PURGES `git` — do git ops in a separate shell. CPU builds use `env.sh`. Build dirs `build-cuda`/`build-serial`/`build-synccheck`/`build-omp` carry M5.20; `build-cuda` is configured `FESOM_SYNC_LOG=OFF` (production).
- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB quota). Big/NG5/CORE2 runs via SLURM, never login. ⚠️ NG5 perf jobs write ~50 GB `*.monthly.nc` even at `snap_every=-1` — the job templates `rm` them; do so.
- **Same-day same-node perf baselines only** ([[feedback-perf-same-day-baseline]]); absolute s/step varies ~7–10 %/day by node mix.
- **Device/kernel changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); pi is insufficient. **Commit/push only when the user asks.** KPP is the default mix_scheme ([[feedback-kpp-default]]).
- ⚠️ **`hnode_new` device-residency is LINFS-specific** (it ≡ hnode, a trivial device copy). When **zstar** is added (a future vertical coordinate, no code yet), hnode_new becomes a genuinely-evolving field — **restore its host rail** then (grep `M5.20: hnode_new` in `fesom_step.cpp` for the 3 ⚠️-guarded sites). Same caution for any optimization that exploits "field X is trivial in the current config."

## 5. BINARIES, TAGS, STATE
- `build-cuda/fesom_port` == `fesom_port_m520` (the validated M5.20 binary). `fesom_port_m519` = the pre-M5.20 before. `fesom_port_synclog` = the FESOM_SYNC_LOG diagnostic build. `serref_m519_saved` = the M5.19 Serial oracle (for diff checks).
- `master` @ `m5.20-pcie-residency` (`4b4da79`). Branch `m517-mpi-comms` == master tip (kept). NO remote → nothing pushed; tags/merges are local.
- Memory was condensed 2026-05-30 (full pre-condense version = `MEMORY_v1.md`).

## 6. POINTERS
- **Memory:** [[project-m520-pcie-residency]], [[project-m519-kernel-coalescing]], [[project-m518-smoother-compute]], [[feedback-perf-same-day-baseline]], [[feedback-gpu-fidelity-gate]], [[reference-cuda-aware-mpi]], [[reference-build-run]].
- **Docs:** `docs/GPU_FIDELITY.md` §M5.18 (the coalescing template + ncu method), §M5.19 (the bucket classification + remaining-headroom list), §M5.20 (the PCIe measurement + the deep_copy≠wall-clock insight); `docs/KOKKOS_PORTING_LESSONS.md` L63–L65 (+ D1–D22, L1–L62).
- **Templates:** `src/fesom_eos.cpp:488` (flat-lever kernel + `_verify`); `src/fesom_bulk.cpp` `fesom_cal_shortwave_rad_kk` (the M5.20 forcing→device port); `src/fesom_step.cpp` (the M5.20 residency edits + the `step_n==1` seed + the ⚠️ zstar guards).
- **Tools:** `jobs/job_ng5_prof` + `job_nsys_ng5` (re-profile), `job_ng5_synclog` (per-field PCIe), `job_ng5_m520_ab` (same-node s/step A/B), `job_ncu_m519_ab` (ncu coalescing), `scripts/ncu_coalesce_summary.py`, `scripts/gpu_fidelity_gate.sh` (the gate), `scripts/m32_climate_compare.py` (climate close-out).

## 7. Bottom line
The fat single kernels and the two big PCIe round-trippers are harvested. What's left, all climate-safe: **(1) finish the bucket-A coalescing** (FCT sub-kernels + update_vel — proven, bit-identical, but each small); **(2) the deliberate host-stays `ghats`/`S`** (residency ports, the harder M5.20-style flips, with the step-1-seed + gate discipline); **(3) Lever C** — the layout refactor that unlocks the FCT/TDMA half of the step (the biggest structural win, highest risk, separate branch); **(4) Lever D** — re-partition for the ~9 % imbalance (deployment, free). Re-profile first, attack 1→2, escalate to 3 only when 1+2 plateau, and hold the Serial-bit-identical gate + same-node A/B + 1-yr climate on every change. Don't claim a win until it's same-day measured + gate PASS + climate-validated (corr=1.0 vs m520). NONE of this touches the climate — the gate + climate prove it each time.
