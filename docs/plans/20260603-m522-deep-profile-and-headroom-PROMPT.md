# Next session — M5.22: deep-profile the m522 GPU port + map (and check) the remaining optimization headroom

*Paste this whole file to start. Self-contained. A MEASUREMENT + STRATEGY session, not a single-lever implement: the deliverable is an **analysis** — where the step's time actually goes now, the ranked candidate optimizations with headroom/risk, and the validated result of the 1–2 most promising ones checked. Written 2026-05-31 at the close of M5.21.*

---

## 0. State — where we are

The FESOM2 **C→C++/Kokkos** port (ocean + sea-ice) is device-resident + validated. **`master` @ annotated tag `m5.21-coalescing-ghats-sss` (`4f9ea70`; NO git remote → local-only).** The M5.x GPU-perf campaign (M5.13→M5.21) took **NG5 dist_16 from 16.3 → 1.45 s/step** and, node-for-node, the GPU from **3.8–5.3× SLOWER than a CPU node → ~2.9× FASTER** on the production meshes. Binary `build-cuda/fesom_port` == `fesom_port_m522` (Lever 1 coalescing + Lever 2a ghats + Lever 2b SSS-trim, all bit-identical, climate corr=1.00000 vs m520).

**The m522 scaling state (measured 2026-05-31, `docs/SCALING_M522.md`):** GPU÷CPU node-for-node at 4 GPU nodes — CORE2 2.4× slow, farc parity, **dars 2.84× fast, NG5 2.95× fast**. The crossover is ≈ farc (0.64 M nodes). **⚠️ KEY: the GPU advantage SHRINKS with node count** (NG5 2.95× @4N → 2.82× @8N → **2.38× @16N**) — i.e. at scale the wall shifts from single-node compute/PCIe toward MPI/halo/load-imbalance. That shift is one of the things this session must quantify.

Lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–L67); perf campaign `docs/GPU_FIDELITY.md` (§M5.1–§M5.21). Memory: [[project-m521-coalescing-finish]], [[project-m520-pcie-residency]] (L65 deep_copy≠wall-clock), [[feedback-gpu-fidelity-gate]], [[feedback-perf-same-day-baseline]].

---

## 1. The mission

The campaign has been a sequence of single levers, each found by re-profiling. After M5.13–21 the step's **budget has changed completely** — the old "NG5 is 75 % PCIe" (L56, M5.12) was measured BEFORE the residency campaign removed most of it. **So job #1 is to re-establish the budget from scratch**, then enumerate + rank the remaining headroom, then check the top 1–2. Concretely, answer:

1. **The time budget of the m522 step** — what fraction is GPU-compute vs PCIe (`deep_copy`) vs MPI/halo vs host vs idle? (nsys is the decisive instrument — see §3.) Build it for NG5 @ 4 nodes (the production sweet spot) AND @ 16 nodes (the comm-bound regime) so you can see the shift.
2. **The per-kernel compute ranking** — is `13_fct` still #1? What's the shape now (scatters / TDMAs / the "b" cluster)?
3. **The residual PCIe** — `ghats`+`S` are gone; rebuild the synclog on m522 and find what's left (ice-FCT values? startup? halos?).
4. **The MPI/comm cost** — how much is halo exchange vs the `Waitall` load-imbalance (M5.17 said 79 % imbalance / 21 % comm at CORE2; re-measure at NG5 scale). This is the lever that grows with node count.
5. **The ranked candidate optimizations** — headroom estimate + risk + which technique, for each (§5).
6. **Check the top 1–2** — prototype + validate (the full ladder) and report the measured delta.

Deliver all of this as a written analysis (`docs/PROFILE_M522.md` or similar) + whatever lever(s) you validated.

---

## 2. Which meshes / node-counts make sense

- **NG5 (7.4 M nodes, 70 lvl) — PRIMARY.** The production mesh; where the GPU wins biggest. Profile here.
- **dars (3.16 M, 47 lvl) — secondary cross-check.** GPU also wins (2.84×); confirms findings generalize across nod3D meshes.
- **SKIP CORE2 (0.13 M) and farc (0.64 M) for optimization** — GPU-starved / comm-bound (CORE2 even flips CPU-favored at ≥2 nodes). They're useful only as the "small-mesh" reference in the scaling story, not optimization targets.
- **Node counts:** profile NG5 at **4 nodes (dist_16, the sweet spot)** primarily; add **one 16-node (dist_64) nsys** to characterise the comm-bound regime (cheap to interpret, expensive to schedule — one run). Same-day same-node A/B for any perf delta (`jobs/job_ng5_m521b_ab` clone; [[feedback-perf-same-day-baseline]]). ⚠️ avoid flaky UCX nodes — the M5.21 A/B segfaulted 3× on `l501xx`; `--exclude` them or re-run.

---

## 3. The instrument suite (and the ONE decisive lesson)

⚠️ **L56: the per-phase WALL timer (`FESOM_STEP_PROFILE`) CANNOT separate compute from PCIe/MPI inside a phase** — it measures kernel+sync+MPI together. The decisive instrument for "compute vs data-movement" is an **`nsys` CUDA trace** (kernels vs memcpy vs MPI). Use the right tool for each question:

| question | tool | job |
|--|--|--|
| compute vs PCIe vs MPI vs idle (the BUDGET) | **nsys** `cuda_gpu_kern_sum` + `cuda_gpu_mem_time_sum` + `mpi_event_sum` | `jobs/job_nsys_ng5` |
| per-phase + per-kernel % (ranking) | `FESOM_STEP_PROFILE=1` | `jobs/job_ng5_prof` |
| per-field PCIe (which fields D2H/H2D, how often) | `FESOM_SYNC_LOG` | `jobs/job_ng5_synclog` — ⚠️ **rebuild `fesom_port_synclog` on m522 first** (the committed one is the m521 build from `build-cuda-synclog/`; `cmake --build build-cuda-synclog` to refresh, then `cp` it) |
| per-kernel memory efficiency (sectors/req, SM/mem util, occ) | **ncu** `--set basic` on the top kernels | `jobs/job_ncu_*` (set `NCU_REGEX`); `scripts/ncu_coalesce_summary.py` |
| MPI halo: comm vs imbalance | nsys `mpi_event_sum` + the `[halo-mpi-prof]` per-step timer (`fesom_halo_device.cpp:104`) | nsys + `FESOM_*` halo-prof env |

**Method for the budget:** from one NG5 dist_16 nsys run (rank 0), partition the per-step wall into: Σ kernel time (GPU-compute), Σ memcpy time (PCIe), Σ MPI time (halo+CG-allreduce), and the remainder (host / launch-gaps / idle). Cross-check Σkernel against the `FESOM_STEP_PROFILE` per-kernel sum, and the memcpy against the synclog `deep_copy` MB/step ÷ PCIe bandwidth. (L56 did exactly this at M5.12 → 7 % compute / 75 % PCIe / 18 % MPI-idle; the residency campaign has since gutted the PCIe — re-derive the m522 split.)

---

## 4. Step 1 — re-establish the budget (the measurement matrix to run first)

Run on the current `build-cuda` (= m522):
1. `jobs/job_nsys_ng5` (NG5 dist_16, rank0) → the compute/PCIe/MPI split + the kernel ranking + the memops split. **This is the anchor.**
2. `jobs/job_ng5_prof` (`--nodes=4 --export=ALL,TAG=prof_m522`) → per-phase + per-kernel % (cross-check + the manual-region phase view).
3. `jobs/job_ng5_synclog` (after rebuilding the synclog binary on m522) → the post-ghats/S per-field PCIe.
4. One **NG5 dist_64 (16-node) nsys** → the comm-bound regime (how the MPI fraction grows; whether the per-rank compute shrinks below the halo cost).
5. (optional) the same matrix on dars dist_16 → confirm NG5 findings generalise.

Write the budget table (compute / PCIe / MPI / host-idle, at 4N and 16N) — that table DICTATES which §5 candidates are worth pursuing. **Don't pre-commit to a lever before the budget says where the time is** (the campaign's plan has been refuted by measurement 4–5× — M5.15, M5.17, M5.20-PhaseB, M5.21-S-floor).

---

## 5. Step 2 — the candidate optimizations (enumerate + estimate; the budget ranks them)

The known remaining buckets (from §M5.19/§M5.21 + L56–L67). For each, the headroom is conditional on the §4 budget:

**If the budget is COMPUTE-bound (likely the 4-node regime now):**
- **Lever C — the layout refactor (the biggest STRUCTURAL win).** `fesom_field.hpp` rank-1 `arr[node*nl+nz]` (node-major) → a `View<double**>` with **LEVEL as the OUTER (slow) dim** so consecutive *nodes'* same-level values are contiguous → a per-NODE-thread warp coalesces. **This is the ONLY path for the bucket-C TDMAs** (`impl_vert_diff_tracers` ~1.6 %, `impl_vert_visc`, `fer_solve_gamma` — sequential level dependency, the flat lever can't touch them) AND it globally improves the bucket-D scatters and every node-major kernel. Climate-safe (a bit-identical memory reorder). **HIGH risk/effort — touches all 126 fields + every kernel + every halo/scatter/index macro → separate branch, staged validation, the full ladder per stage.** Estimate the headroom first (sum the bucket-C/D kernel shares from the §4 nsys — the 3 TDMAs + the FCT scatters ≈ 5–8 % of the step). This is where FESOM2-GPU eventually has to go for the FCT/TDMA half of the step.
- **FCT `*_h` edge-scatter coalescing (edge-coloring).** `fct_eud_fill` (~2.2 %), `fct_mfct_h` (~2 %), `fct_zal_b3h`/`b1h`, `fct_f2d_h`, `momadv_horiz`, `visc_bidiff`, `ale_vvel_scatter`, redi edges — all `atomic_add` from edge/element loops (uncoalesced + atomic contention). Coalesce via a different axis (edge-coloring → no atomics) or fold into Lever C. Climate-close (atomic ordering).
- **The "b" cluster (`fct_zal_b2` ~1.4 %, `b1v` ~0.8 %, `b3v` ~0.6 %) — code-read first.** Unclassified at the M5.19 close; some per-node b-kernels may be a clean **bucket-A subset the M5.19 sweep missed** (flat-lever-able, bit-identical, cheap). Read the data flow (L64: a "column scratch" of per-level accumulators IS flat-lever-able; a true cross-level reduction is NOT). Low-risk if any qualify.
- **Kernel fusion / launch overhead.** The step launches ~100+ kernels; if the §4 nsys shows significant launch-gap/host time, fuse adjacent small kernels (the M5.12 fusion pattern) to cut launch latency. Bit-identical if the fused arithmetic is unchanged.

**If the budget is PCIe-bound (residual round-trips):**
- The post-ghats/S synclog leaders (§4.3). After M5.20+M5.21 the per-step D2H is led by ice-FCT `values` (~18 MB/step ×3, 5/step H2D + 3/step D2H — the ice tracer FCT round-trips) + forcing pushes + startup. Apply the M5.20 residency pattern (find the host reader; placebo→drop, real→port) — but FIRST confirm they're per-step round-trippers worth more than their bytes (L65: prioritise by round-trips/substep-boundary, not GB).

**If the budget is MPI/COMM-bound (the 16-node regime, growing with scale):**
- **Halo-exchange cost vs load imbalance.** M5.17 measured the CORE2 halo `Waitall` as 79 % load-imbalance / 21 % comm → overlap ceiling only 2.4 %. **Re-measure at NG5 scale** — the balance may differ (bigger halos). If it's genuine comm: compute/comm overlap (start the halo before the interior kernel finishes), or NCCL/NVLink for intra-node halos. If it's imbalance: **re-partition (Lever D)** — a better GPU domain decomposition for balanced per-rank work (no code change, just the mesh partition; climate-identical). Cheap to try (regenerate the partition); worth a measurement since the scaling shows comm growing with node count.
- **The CG (SSH solve) allreduce** — was DEAD at CORE2 (3.3 %) but check at NG5 scale (more iters, bigger allreduce).

**(Speculative — flag, don't chase unless the budget demands):** mixed precision (FP32 for selected fields — changes the climate, needs its own validation), algorithmic (fewer CG iters via a better preconditioner). Both are big + climate-risky; out of scope unless the budget shows they're the only wall.

---

## 6. Step 3 — check the top 1–2

Pick the 1–2 highest headroom × lowest risk candidates the budget surfaces, **prototype + validate** (the full ladder §7), and report the same-node A/B delta + the gate/climate result. If the top candidate is Lever C (high effort), at minimum **prototype it on ONE TDMA** (`impl_vert_diff_tracers`) on a branch to measure the actual coalescing win + de-risk the layout change, before committing to the full refactor. The b-cluster code-read + any clean bucket-A flips are cheap quick wins worth landing regardless.

---

## 7. The deliverable + the validation discipline

**Deliverable: `docs/PROFILE_M522.md`** — (1) the time budget (compute/PCIe/MPI/idle, at 4N and 16N, NG5 ± dars); (2) the per-kernel ranking + the per-field PCIe; (3) the ranked candidate table (headroom est. + technique + risk); (4) the measured result of whatever you checked; (5) a recommendation for the M5.23 lever. Plus update `docs/GPU_FIDELITY.md` + a lesson (L68).

**Validation ladder (any code change — the discipline that has held all campaign):**
1. Per-kernel `FESOM_KK_VERIFY=<key>` Serial `max|Δ|==0` (add a capture-before hook if needed — `src/fesom_eos.cpp:488` `_verify` is the template).
2. pi np1+np2 BIT-IDENTICAL (`build-serial`, vs golden + `/scratch/a/a270088/pi_np2_ref_m13_nocma`; np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`). ⚠️ pi uses analytical forcing → for forcing/PCIe changes the real Serial check is a **fresh CORE2 Serial oracle vs a saved prior** (`gpu_fidelity_gate.sh --fresh-oracle` + `diff_snap.py`).
3. SYNCCHECK clean (`build-synccheck`).
4. **CUDA fidelity gate** `scripts/gpu_fidelity_gate.sh [--fresh-oracle]` (CORE2 dist_8 ice-active) — PASS = the climate-close floor (~1e-2). THIS catches stale-host/staleness bugs; **pi cannot** ([[feedback-gpu-fidelity-gate]]).
5. **Same-node A/B** for s/step (`jobs/job_ng5_m521b_ab` clone, BEFORE=`fesom_port_m522`; back-to-back in ONE allocation — [[feedback-perf-same-day-baseline]]). Read the `FESOM_STEP_PROFILE` deep_copy line for the PCIe delta. ⚠️ deep_copy bytes ≠ wall-clock (L65).
6. ncu before/after per coalescing kernel.
7. 1-yr CORE2 CUDA climate to close (`jobs/job_m32_cuda_core2`, `M32_NSTEPS=17280 M32_TAG=_<tag>`; `scripts/m32_climate_compare.py <dir> --years 1958 --cref m32_cuda_m522_1yr` — m522 is the new cref; bit-identical ⇒ corr=1.0).

## 8. Hard constraints (carry every session)
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; env.sh's 4.1.2 SEGFAULTs on device ptrs, L47). ⚠️ `env_cuda.sh` PURGES `git` — git ops in a separate shell. CPU builds use `env.sh`. Build dirs `build-cuda`/`build-serial`/`build-synccheck`/`build-omp`/`build-cuda-synclog` carry m522.
- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB quota; ~14 GB free). Big/NG5/CORE2 runs via SLURM, never login. ⚠️ NG5 perf jobs write ~50 GB `*.monthly.nc` even at `snap_every=-1` — the `job_ng5_*_ab`/`job_scaling_gpu`/`job_ng5_synclog` templates `rm` them; `job_ng5_prof` does NOT — clean it. ⚠️ the SLURM account has an `AssocMaxJobsLimit` (~4 concurrent) — batch submissions trickle.
- **Same-day same-node perf baselines only**; absolute s/step varies ~7–10 %/day by node mix. **Device/kernel changes MUST pass `gpu_fidelity_gate.sh` before commit. Commit/push only when the user asks.** KPP is the default mix_scheme.
- ⚠️ Lever C touches the field layout — the `hnode_new` LINFS-residency guard (M5.20) and every `FESOM_NODE3D`/`FESOM_ELEMVEC` index macro must be revisited; do it on a separate branch with staged per-field validation.

## 9. Pointers
- **Tools:** `jobs/job_nsys_ng5` (the budget anchor), `job_ng5_prof`, `job_ng5_synclog`, `job_ncu_*`, `job_scaling_gpu` (the generic scaler), `job_ng5_m521b_ab` (clone for A/B), `scripts/ncu_coalesce_summary.py`, `scripts/gpu_fidelity_gate.sh`, `scripts/m32_climate_compare.py`, `scripts/plot_scaling.py`.
- **Docs:** `docs/GPU_FIDELITY.md` §M5.18 (coalescing template + ncu method), §M5.19 (bucket classification + remaining-headroom list), §M5.20 (PCIe measurement + deep_copy≠wall-clock), §M5.21 (the Lever-2 ghats/SSS findings); `docs/SCALING_M522.md` (the mesh/node matrix + the comm-bound-at-scale finding); `docs/SCALING_{CORE2,FARC,DARS,NG5}.md` (pre-campaign per-mesh detail); `docs/KOKKOS_PORTING_LESSONS.md` L56 (the nsys-is-decisive lesson), L63–L67.
- **Superseded:** `docs/plans/20260601-m521-remaining-speedups-PROMPT.md` (M5.21, DONE) and `docs/plans/20260602-m522-sss-runoff-residency-PROMPT.md` (Lever 2b, DONE — folded into M5.21). This is the live prompt.

## 10. Bottom line
The campaign harvested every single lever found by re-profiling; the step's budget is now unrecognisable from L56's pre-residency "75 % PCIe". **Re-establish the budget first** (nsys, NG5 @4N and @16N), then map the remaining headroom onto the candidates (Lever C layout refactor for the TDMA/scatter half; edge-coloring for the FCT scatters; the b-cluster quick win; re-partition/overlap for the comm regime that grows with scale), estimate each, **check the top 1–2**, and deliver the analysis + a recommendation for the next lever. Measure before building; hold the Serial-bit-identical + gate + same-node A/B + climate on every change. The big structural question this session should answer: **is the remaining win in single-node compute (Lever C) or in the comm/scaling regime (re-partition/overlap)?** — the §4 budget at 4N vs 16N decides it.
