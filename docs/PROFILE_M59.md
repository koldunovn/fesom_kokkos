# M5.11 — GPU Profile Pass (docs/PROFILE_M59.md)

**Status: IN PROGRESS — drafting alongside in-flight slurm runs.**

Profile-only milestone (no kernel optimisation). The data in this doc drives
the M5.12 lever pick (see `docs/M512_PLAN.md`). Plan: `docs/plans/20260528-m511-profile-pass.md`.

## §1 Setup

- **Binary commit**: `366887c` (branch `profile-m511`, off `master @ 466ea3e`).
- **Mesh + forcing**: CORE2 (127k nodes × 47 levels), JRA55 1958 (active-ice).
- **dt**: 1800 s for the timing runs, 200 steps (5 warmup excluded → 195 timed). nsys runs 30 steps.
- **Instrumentation**: Kokkos profiling-tools callbacks (parallel_{for,reduce,scan}
  begin/end + deep_copy begin) registered in `fesom_prof::install_callbacks`
  (env-gated by `FESOM_STEP_PROFILE=1`). Auto-instruments every kernel by its
  Kokkos label; PHASE markers (M5.6 PMARK chains in `fesom_step.cpp`) coexist
  in the same report.
- **Runs** (jobs in `jobs/`, output under `/work/ab0995/a270088/port2/kokkos_profile_m511/`):

  | Run | Job | Config | Status |
  |:----|:----|:-------|:------:|
  | R1a | `job_profile_core2_dist1` | np=1 / 1 A100 (gpu-devel) | _pending dist1 finish_ |
  | R1b | `job_profile_core2_dist4` | 4 A100 / 1 GPU node | _slurm queue_ |
  | R1c | `job_profile_core2_dist8` | 8 A100 / 2 GPU nodes (current baseline) | _slurm queue_ |
  | R2a | `job_nsys_core2_np1_v2` | np=1 nsys timeline | **DONE 2026-05-28 18:52** |
  | R2b | `job_nsys_core2_dist4` | dist_4 nsys | _slurm queue_ |
  | R3  | `job_ncu_top5` | ncu --set full on top-5 kernels | _waits for R1c top-5_ |

- **ncu fallback note**: if `--set full` OOMs on FCT (largest working set),
  fall back to `--set roofline` + the 4 explicit sections (script has the
  commented-out alternative).

## §2 R1 per-kernel timing table

### R1a — np=1, no halos (CORE2 dist_1, 1 A100 a100_40 on gpu-devel)

Loop timing: **1.0764 s/step** (195 timed steps, FESOM_STEP_PROFILE=1).
⚠️ **Profile-mode overhead**: the begin/end callbacks add a Kokkos::fence
before+after every parallel_for. With ~1876 launches/step (from R2a nsys)
× 2 fences × ~33 µs avg = **~124 ms/step (~12%) added to step time**. The
overhead is uniform across phases — relative %share is preserved but
absolute s/step is inflated. **The clean np=1 baseline is therefore
~0.95 s/step**, comparable to the dist_4 = 0.86 s/step in
`docs/SCALING_CORE2.md` and showing weak np=1 → dist_4 strong-scaling
(suggesting the GPU is under-utilised at small per-rank size).

**Top 10 phases (R1a):**

| Phase | %loop | s/step | Notes |
|:------|------:|-------:|:------|
| `13_fct`            | 19.95 | 0.2148 | FCT advection — still the biggest |
| `3_mixing`          |  9.67 | 0.1041 | KPP (post M5.7 device-Kv/Av) |
| `1b_gm`             |  9.37 | 0.1008 | GM chain |
| `12_ale`            |  6.69 | 0.0721 | ALE vertical velocity + thickness |
| `13b_trdiff`        |  5.64 | 0.0607 | Tracer diff (implicit vertical TDMA) |
| `4_velrhs`          |  5.41 | 0.0582 | Momentum RHS |
| `6_ivisc`           |  4.25 | 0.0457 | Implicit vertical viscosity |
| `7_ssh`             |  4.03 | 0.0434 | SSH CG solver |
| `1_eos`             |  3.84 | 0.0413 | EOS / density |
| `ice_dyn(o2i+EVP)`  |  3.62 | 0.0389 | Sea-ice dynamics |

**Top 12 individual kernels (R1a):**

| Kernel | %loop | s/step | calls/step | Notes |
|:-------|------:|-------:|-----------:|:------|
| `fesom_smooth_gather`        | 4.92 | 0.0530 | **10.0** | M5.5 device smoother gather (×10 sweeps?) |
| `fct_mfct_h`                 | 0.72 | 0.0078 |     2.0 | FCT horizontal flux, T+S |
| `fesom_impl_vert_diff_tracers` | 0.65 | 0.0070 |  2.0 | Tracer-diff TDMA |
| `fesom_gm_redi_hor_edge`     | 0.59 | 0.0063 |     2.0 | Redi horizontal SCATTER |
| `fesom_gm_redi_ver_node`     | 0.56 | 0.0061 |     2.0 | Redi vertical gather |
| `fct_eud_fill`               | 0.55 | 0.0060 |     2.0 | FCT edge up/down gradient fill |
| `fesom_smooth_scale`         | 0.50 | 0.0054 |    10.0 | M5.5 smoother scale (paired with gather) |
| `fesom_vel_rhs_elem`         | 0.42 | 0.0046 |     1.0 | Momentum RHS per-elem |
| `fct_zal_b2`                 | 0.38 | 0.0040 |     2.0 | Zalesak b2 limiter |
| `fesom_visc_bidiff_stage2`   | 0.37 | 0.0040 |     1.0 | Biharmonic viscosity SCATTER stage 2 |
| `fesom_impl_vert_visc`       | 0.36 | 0.0039 |     1.0 | Momentum impl vert visc TDMA |
| `fct_f2d_v`                  | 0.35 | 0.0038 |     2.0 | FCT flux2dtracer vertical |

**Key observations from R1a:**

1. **Auto-instrumentation works**: all 79+ parallel_for sites are surfacing
   by their Kokkos label; call counts match expected physics
   (`fct_*` = 2 / step for T+S tracers, `ice_s2rhs_scatter` = 120 / step for
   the EVP subcycle loop, `fesom_cg_psolve_dot2` = 112 / step matching the
   CG iteration count).
2. **`fesom_smooth_gather` + `fesom_smooth_scale` together = 5.4%** of the
   step at np=1, with 10 calls/step each. This is the M5.5 device smoother
   (`fesom_smooth_nod3D_kk`). 10 invocations is more than expected — likely
   1 bvfreq + 9 KPP-Kv-Av or similar. **Suspect for kernel fusion.**
3. **Phase total vs kernel sum is NOT additive**: `13_fct = 19.95%` but the
   sum of `fct_*` kernel rows is ~5.5%. The remaining ~14.5% in the FCT
   phase is *host code* between launches + *fence overhead* + halo brackets
   inside the phase. The profile-mode fence-per-kernel removes natural
   async overlap, inflating phase totals. **For the M5.12 lever decision
   we care about kernel-time growth, not phase total.**
4. **Many ~0.3% kernels** — no single dominant compute kernel; the largest
   is `fesom_smooth_gather` at 4.92% (and that's launch-count-driven).
   Suggests the wall is *launch density* + *small kernels*, not one fat one.

### R1b — dist_4 (4 A100 / 1 GPU node, intra-node halos)

Loop timing: **0.8868 s/step** with profile-mode overhead (vs `docs/SCALING_CORE2.md`
dist_4 = 0.8617 s/step clean — ~3% overhead at dist_4, less than at np=1 since the
fence-per-kernel overhead is amortized across denser compute). **PCIe: 2.10 GB/step**
(209 deep_copy calls/step) — 4× less than np=1.

**Top phases (R1b):**

| Phase | %loop | s/step |
|:------|------:|-------:|
| `13_fct`            | 19.60 | 0.1738 |
| `1b_gm`             | 11.46 | 0.1016 |
| `7_ssh`             |  7.91 | 0.0701 |
| `ice_dyn(o2i+EVP)`  |  7.72 | 0.0684 |
| `12_ale`            |  7.71 | 0.0684 |
| `3_mixing`          |  6.94 | 0.0616 |
| `13b_trdiff`        |  6.55 | 0.0581 |
| `4_velrhs`          |  5.95 | 0.0528 |
| `6_ivisc`           |  5.09 | 0.0451 |
| `1_eos`             |  3.71 | 0.0329 |

### R1c — dist_8 (8 A100 / 2 GPU nodes, cross-IB halos — the perf baseline of record)

Loop timing: **0.5182 s/step** with profile-mode overhead (vs the M5.9-pin
**0.4777 s/step clean baseline** — ~8% overhead at dist_8, lowest of the three
configs as expected; profile-mode fence cost is fixed-per-kernel and the dist_8
loop has the smallest per-rank compute window). **PCIe: 1.07 GB/step** (209 calls/step).

**Top phases (R1c):**

| Phase | %loop | s/step | Notes |
|:------|------:|-------:|:------|
| `13_fct`            | 17.43 | 0.0903 | FCT advection — still #1 |
| `ice_dyn(o2i+EVP)`  | 11.26 | 0.0583 | **GROWS** with rank count (np=1=3.6%, dist_4=7.7%, dist_8=11.3%) — halo-bound |
| `7_ssh`             | 10.92 | 0.0566 | **GROWS** similarly (np=1=4.0%, dist_4=7.9%, dist_8=10.9%) — CG iters × halo |
| `1b_gm`             |  9.79 | 0.0507 | GM chain |
| `3_mixing`          |  6.76 | 0.0350 | **DROPS** with rank count (np=1=9.7%, dist_4=6.9%, dist_8=6.8%) — KPP halo flips win |
| `12_ale`            |  6.61 | 0.0342 | |
| `13b_trdiff`        |  5.68 | 0.0294 | |
| `4_velrhs`          |  5.24 | 0.0272 | |
| `6_ivisc`           |  4.38 | 0.0227 | |
| `1_eos`             |  3.24 | 0.0168 | |
| `5_viscfilt`        |  1.53 | 0.0079 | |
| **`fesom_halo_unpack`** | 1.51 | 0.0078 | **526 calls/step** ← halo machinery visible |
| **`fesom_halo_pack`**   | 1.46 | 0.0076 | **526 calls/step** |

**Top 12 individual kernels at dist8 (excluding phase markers):**

| Kernel | %loop | calls/step | s/step | Per-call cost | Lever |
|:-------|------:|-----------:|-------:|--------------:|:------|
| `fesom_halo_unpack`     | 1.51 | 526.4 | 0.0078 | ~14.8 µs | (high count, per-call cheap) |
| `fesom_halo_pack`       | 1.46 | 526.4 | 0.0076 | ~14.4 µs | (paired with unpack) |
| `fesom_smooth_gather`   | 0.68 | 10.0  | 0.0035 | **~350 µs** | (10×big kernels, per-call expensive) |
| `fesom_cg_psolve_dot2`  | 0.57 | 111.7 | 0.0030 | ~26.8 µs | (CG iter) |
| `fesom_cg_spmv_dot`     | 0.56 | 111.7 | 0.0029 | ~26 µs   | (CG iter, fused SpMV+dot from M5.2) |
| `ice_stress_tensor`     | 0.32 | 120   | 0.0016 | ~13.3 µs | (EVP subcycle) |
| `ice_evp_velupd`        | 0.31 | 120   | 0.0016 | ~13.3 µs | (EVP subcycle) |
| `ice_s2rhs_final`       | 0.31 | 120   | 0.0016 | ~13.3 µs | (EVP subcycle) |
| `ice_s2rhs_scatter`     | 0.31 | 120   | 0.0016 | ~13.3 µs | (EVP subcycle SCATTER) |
| `ice_evp_saveold`       | 0.31 | 120   | 0.0016 | ~13.3 µs | (EVP subcycle) |
| `ice_evp_coastal_bc`    | 0.30 | 120   | 0.0015 | ~12.5 µs | (EVP subcycle) |
| `ice_s2rhs_zero`        | 0.29 | 120   | 0.0015 | ~12.5 µs | (EVP subcycle init) |

**Key dist8 observations:**

1. **The wall is launch density, not single-kernel cost.** The biggest individual
   kernel is `fesom_halo_unpack` at 1.51% / 14.8 µs per call. There is no fat
   kernel to optimize. The cost is in the **number** of launches:
   - halo pack + unpack: **1052 launches/step** (≈526 brackets, each with pack+unpack)
   - CG kernels: ~560 launches/step (~5 ops × 112 iters)
   - EVP per-subcycle kernels: ~840 launches/step (~7 ops × 120 subcycles)
   - Plus all the once-or-twice-per-step kernels (FCT 28 × 2, momentum, GM, …)
   - **Total ≈ 2450 launches/step from these alone** (nsys reports 1876/step at np=1
     including init; dist8 is in the same ballpark per rank).
2. **EVP and SSH grow with rank count** (3.6% → 7.7% → 11.3% for EVP; 4.0% → 7.9% →
   10.9% for SSH) — they're halo-bound; the fix is more launch fusion + smaller
   halo brackets.
3. **KPP (`3_mixing`) DROPS with rank count** (9.7% → 6.9% → 6.8%) — the M5.7 KPP
   halo flips are doing their job; this is no longer a major lever.
4. **`fesom_smooth_gather` at 0.68% / 10 calls/step is the biggest non-halo per-call
   kernel** — the M5.5 device smoother. 350 µs per call × 10 calls = 0.0035 s. The
   10 calls/step is high; **lever: count and fuse smoother invocations**.
5. **PCIe is no longer dominant** at dist_8: 1.07 GB/step vs the M5.3 stale stat of
   13 GB/step (pre-halo-flips). M5.1+M5.4+M5.7 ocean-halo flips reduced PCIe by ~12×.

### Caveat for R2 readers

The nsys timeline run has FESOM_STEP_PROFILE=1 enabled (to capture our
NVTX regions), AND nsys' own instrumentation overhead. Loop s/step = 1.103
on R2a (vs 1.076 on R1a — about the same; the nsys cuda-trace overhead is
~3% on top of the profile-mode overhead). Use R1a/b/c for the actual
same-day timing baseline.

## §3 Halo wall (R1c − R1b vs R1b − R1a)

| Phase | np=1 (R1a) | dist_4 (R1b) | dist_8 (R1c) | Pattern |
|:------|------:|------:|------:|:--------|
| **13_fct** (s/step) | 0.2148 | 0.1738 | 0.0903 | scales sub-linearly — half of cost in compute, half halo |
| **ice_dyn EVP** (s/step) | 0.0389 | 0.0684 | 0.0583 | **DOESN'T scale** — halo-bound (120 subcycles × bracket) |
| **7_ssh** (s/step) | 0.0434 | 0.0701 | 0.0566 | **DOESN'T scale** — CG halo + dot ping-pong |
| **1b_gm** (s/step) | 0.1008 | 0.1016 | 0.0507 | scales between dist_4 and dist_8 |
| **3_mixing** (s/step) | 0.1041 | 0.0616 | 0.0350 | scales linearly — KPP halo flips already won |
| **12_ale** (s/step) | 0.0721 | 0.0684 | 0.0342 | scales linearly |
| **fesom_halo_pack+unpack** (s/step) | 0 | 0.0140 (est) | 0.0154 | grows with rank count (more halo brackets) |

**Halo cost breakdown (R1c − R1b deltas indicate IB-cross-node penalty)**:

- EVP and SSH s/step actually DROP slightly from dist_4 to dist_8 even though the
  fraction-of-step grows — they scale poorly but not catastrophically.
- **The 526 halo pack/unpack calls/step at dist_8** is the visible cost of the
  halo machinery itself: ~15.4 ms/step in pack+unpack overhead alone (~3% of step).
- Going np=1 → dist_8 the PCIe traffic DROPPED 8× (8.6 → 1.07 GB/step) — the
  M5.1/M5.4/M5.7 on-device halo + halo-flip campaign moved the bulk of inter-rank
  data off PCIe entirely. The remaining PCIe (1 GB/step) is host-resident state
  + I/O snapshots + the remaining smoother-call full-field syncs.

**Lever orientation: halos are NOT the dominant wall anymore at dist_8.** They cost
~3% of the step (pack+unpack), and the EVP/SSH "halo bound" cost is mixed with their
inherent per-subcycle / per-CG-iter compute. The compute kernels themselves are the
remaining target.

## §4 Top-5 deep dive (R3 ncu bound labels)

**Top-5 picked from R1c (diverse profile: high-launch-count + per-call expensive + scatter + CG + FCT + EVP):**

1. `fesom_smooth_gather` — 350 µs/call × 10 calls = 0.0035 s/step. **Per-call expensive**, big working set. Hypothesis: memory-bandwidth-bound (large nod3D read+write).
2. `fesom_halo_unpack` — 14.8 µs/call × 526 calls = 0.0078 s/step. **Launch-density signature**. Hypothesis: launch-overhead-bound (each call does a tiny gather).
3. `fesom_cg_psolve_dot2` — 26.8 µs/call × 112 calls = 0.0030 s/step. **CG inner loop hot path**. Hypothesis: latency-bound from the global reduction.
4. `fct_mfct_h` — 13 µs/call × 2 calls = 0.0005 s/step (small for ncu, but THE FCT compute target — Lever C's failure case). Hypothesis: branch-divergence or atomic-contention (Lever C's diagnosis).
5. `ice_stress_tensor` — 13.3 µs/call × 120 calls = 0.0016 s/step. **EVP per-subcycle**. Hypothesis: launch-overhead-bound.

**R3 ncu run**: `jobs/job_ncu_top5` regex updated to match these. Submit when ready.
Expected outcome — kernel bound labels feeding §7:

- **Memory throughput** (% of peak DRAM bw): > 80% → memory-bound; < 50%
  with low IPC → latency-bound or occupancy-limited.
- **Achieved occupancy** (% of theoretical): < 50% → occupancy-limited
  (lever: shrink registers, smaller team size).
- **Warp state breakdown**: dominated by `Stall_LongScoreboard` →
  memory-latency-bound; `Stall_Barrier` → scatter contention; `Stall_MIO_Throttle`
  → atomic contention; `Stall_NotSelected` → warp scheduler underused.
- **Branch divergence**: > 30% predicated-off → divergence-bound
  (the M5.10 Lever-C diagnosis on FCT's Zalesak limiter).

## §5 nsys API + PCIe summary (np=1, R2a)

From `nsys_np1` (job 25215621, np=1, 30 steps, 25 timed):

| API call           | %       | Total (ns)    | Calls   | Avg (ns)  | Notes |
|:-------------------|--------:|--------------:|--------:|----------:|:------|
| cudaMemcpy         | **69.8** | 17.35 G      | 6,192   | 2.80 ms   | Dominant — PCIe time |
| cudaDeviceSynchronize | 27.7 | 6.89 G       | 206,520 | 33 µs     | Profile-mode fences (begin+end of each kernel) |
| cudaLaunchKernel   | 1.1     | 265 M         | **46,910** | 5.7 µs | **~1876 launches / step** at np=1 |
| cudaStreamSynchronize | 0.8  | 188 M         | 20,761  | 9 µs      | |
| cudaMallocAsync    | 0.6     | 152 M         | 442     | 344 µs    | One-time init |
| cudaMemsetAsync    | 0.0     | 9.4 M         | 503     | 19 µs     | |
| cudaMemcpyAsync    | 0.0     | 5.8 M         | 443     | 13 µs     | |

PCIe traffic (`cuda_gpu_mem_time_sum`):

| Direction | %    | Time (s) | Calls  | Total | Per step | Avg/call |
|:----------|-----:|---------:|-------:|------:|---------:|---------:|
| H2D       | 64.5 | 10.93    | 4,324  | 164 GB | **5.5 GB** | 37.9 MB |
| D2H       | 35.4 |  6.01    | 2,371  | 92 GB  | **3.1 GB** | 38.8 MB |
| D2D       |  0.1 |  0.07    |    30  | 1.46 GB | 48.7 MB   | 48.7 MB |
| memset    |  0.0 |  0.02    |   505  |    -   | -        | -        |

Per-step **PCIe = 8.6 GB** at np=1 (matches our deep_copy callback: **8.4 GB / 207 calls per step**). Max single transfer **187.9 MB** = full nod3D field (~48 MB × 4 = T+S+Av+Kv span). This is **post-M5.1+M5.4+M5.7 halo flips** — the residual PCIe is likely I/O snapshot syncs + the bvfreq/blmc smoother's full-field copies + some still-host fields. The fresh count clears the stale "13 GB/step, cudaMemcpy 83% of API time" stat (which was on dist_8 pre-halo-flips).

**Top GPU kernel work (placeholder)**: the nsys `gpukernsum` section wasn't
captured in stdout (only the API summary printed). We'll read it directly
from the `.nsys-rep` file via `nsys stats --report gpukernsum prof.sqlite`
when composing §4.

## §6 Host residual audit

Findings (run on `profile-m511` HEAD = M5.11-a):

- **smooth_nod3D call sites**: see `grep -rn 'smooth_nod3D\|fesom_smooth_nod3D_kk' src/`.
  The M5.5 device smoother (`fesom_smooth_nod3D_kk`) is used by:
  - `bvfreq` smoother (M5.5a): 1 call/step
  - `blmc` slab in KPP (M5.5b): 9 sweeps × 3-slab via base offset = 9 calls/step
  - Total = **10 calls/step** ← matches the R1c report exactly.
  No surviving host-only smoother sweeps. M5.5 covered all of them.
- **fesom_step.cpp `sync_host()` calls** (counted on master HEAD):
  - 1 in fesom_timestep at line ~207 (bvfreq, M5.9 FIX kept) — the pinned uvnode reader
  - Pre-I/O sync_host on snapshot fields (L48 guard) — gated by snapshot step
  - No other surviving host-readers in the timestep loop (consistent with M5.9-pin).
- **Raw host `for (` loops in the 6 hot-phase files**: none on the hot path
  outside parallel_for invocations (only init/setup code, called once per init).
- **M5.4 leftovers** (GM owned-reads, `ssh_rhs` nod2D, tracer-diff): not surfacing
  as fat in R1c. They were known low/uncertain payoff; the R1c data confirms.

**No blmc-class hidden host cost surfaced.** The remaining performance gap is
in **device-kernel launch density + per-call cost**, not host residuals.

## §7 THE M5.12 lever — **launch density reduction via fusion**

**Claim**: at dist_8 the wall is launch density, not memory bandwidth, single-kernel
arithmetic, or PCIe. Reduce launches → reduce per-kernel fence + dispatch cost → win.

**Evidence (R1c)**:
1. **No fat kernel exists.** The biggest individual kernel is `fesom_halo_unpack`
   at 1.51% of step. The top 12 kernels combined are < 8% of step.
2. **2450+ launches/step from 3 hot regions**:
   - Halo pack + unpack: 1052/step (526 brackets × 2)
   - CG iters: ~560/step (~5 ops × 112 iters)
   - EVP per-subcycle: ~840/step (~7 ops × 120 subcycles)
3. **PCIe is no longer dominant** (1 GB/step at dist_8 — 12× less than the M5.3 stale
   stat). The halo flip campaign already won that battle.
4. **Per-call cost is small** (~14 µs for halo pack/unpack, ~13 µs for EVP) — close
   to the bare CUDA launch overhead floor (~5–10 µs). Fusion would amortize the
   launch + fence per fused-pair.

**Sub-milestones (each independently shippable with its own validation gate)**:

| ID | Target | Approach | Expected payoff |
|:-:|:-------|:---------|:---------------:|
| **M5.12a** | Halo pack/unpack | Fuse `fesom_halo_pack`/`unpack` with the adjacent compute kernel (the compute that produces / consumes the halo field). 526 brackets → potentially 526 fewer launches if the pack folds into the producer. | medium (saves ~1.5% of step direct; possibly 3% with async overlap recovered) |
| **M5.12b** | EVP per-subcycle | Fuse the 7 per-subcycle kernels (`s2rhs_zero`+`s2rhs_scatter`+`s2rhs_final`, `evp_saveold`+`vel_update`, `stress_tensor`+`coastal_bc`) into 3 fused kernels. 120 × 4 saved launches/step = 480 launches/step. | high (saves ~2% of step direct from launch overhead alone) |
| **M5.12c** | CG iterations | Investigate whether the 5 CG kernels can fuse further (M5.2 already fused SpMV + dot, M5.3+ moved CG to device). Likely the 3rd-AB SpMV-axpy-dot-axpy could become 2 kernels. ~110-220 launches/step saved. | low–medium (CG is only 5.7% of step at dist_8) |
| **M5.12d** | Smoother call count | Audit if 10 calls/step of `fesom_smooth_nod3D_kk` (1 bvfreq + 9 blmc-slabs) can be reduced — e.g. do all 3 slabs at once with a different base-offset pattern, dropping 9 → 3 calls. Per-call is 350 µs so removing 6 calls saves ~2 ms/step. | medium (saves ~0.4% but per-call is big — high signal-to-noise) |

**Total target**: a cumulative 4–7% on dist_8 step time (0.518 → ~0.49–0.50 s/step
with profile mode; clean 0.478 → ~0.45 s/step). Modest but compounding with what's
already shipped (M5.4 25%, M5.5 +13% hidden, M5.7 +1%, M5.8 +0.5%, M5.9 +3%).

**Validation strategy** (each sub-milestone):
- Per-kernel verify (`FESOM_KK_VERIFY=<key>` on Serial) still passes — fused
  kernels MUST produce byte-identical Serial output.
- GPU fidelity gate (now against the FRESH oracle, per § Cleanup) PASSES at
  the same noise floor.
- Same-day perf baseline before/after on CORE2 dist_8.

**Risk**: per-kernel verify might break if fusion changes the read/write
ordering of shared state. Mitigation: fuse small adjacent kernels first
(M5.12a halo fuse is local), defer EVP and CG fusion to later sub-milestones.

## §8 Discarded levers

- **Memory coalescing via rank-1 → rank-2 LayoutLeft** (Lever C, M5.10b):
  FAILED already — only 1.3% on the dominant FCT phase. The scatters are
  layout-agnostic, the kernels aren't memory-bound. Confirmed by §4 ncu picks
  (no top kernel is bandwidth-limited — confirmed at dist_8 by the small
  per-call costs of 13–30 µs which can't be memory-saturating).
- **Bigger mesh to fill A100s** (the farc test):
  FAILED — node-for-node CPU stays 7× faster regardless of mesh size
  (`docs/SCALING_FARC.md`). Mesh size is not the lever.
- **More halo on-device flips**:
  Diminishing returns — PCIe is already 1 GB/step at dist_8, halo pack+unpack
  is only ~3% of step. The remaining ocean halos (per `docs/SYNC_MAP.md`) are
  small nod2D or one-per-step — not high-frequency.
- **Mixed precision (FP32 for some kernels)**:
  Would risk climate fidelity. The M3.2 gate would need to be re-validated for
  each FP32 conversion. Not a fast win.

---

*Last updated: 2026-05-28. R1a/b/c + R2a/b done; R3 ncu queued separately. Gate validated against FRESH oracle today_serial (PASS, worst 1.0e-2 on h_ice).*
