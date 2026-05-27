# GPU / OpenMP climate fidelity (M3.2)

**Status: SKELETON — fill in when the M3.2 runs complete.** Created at the end of M4 (whole model on
device, tag `m4-full-device`); the per-kernel + 1-yr Serial gates are bit-identical, this doc records the
*climate-close* fidelity of the threaded/GPU backends over a multi-year run.

## Why this doc

Serial is the bit-identity oracle (`max|Δ|==0` vs the C twin, per-kernel + the 1-yr CORE2 acceptance).
OpenMP and CUDA are **climate-close, not bit-identical** (D22): the edge/element→node `atomic_add`
SCATTERS (`visc_filt_bidiff`, `momentum_adv`, `vert_vel`, Redi `diff_hor`, the 3 ocean-FCT scatters, the
2 SSH scatters, the ice EVP ×2, the ice FCT ×3, the ice oce_fluxes net-subtract) and the `parallel_reduce`
REDUCTIONS (the CG dot-products, `integrate_nod_2D`) reassociate across threads/lanes. On pi the per-step
floor is ≲1e-12 (≈ FP), but **pi has zero ice** — the sea-ice scatters/reductions are exercised only on
CORE2 (forced), and they **compound over a multi-year run**. M3.2 measures whether that compounding stays
within the Fortran↔C budget (i.e. the GPU/OpenMP port is climate-faithful).

## Known baselines (from earlier milestones)

- **CUDA pi smoke** (single-step regime, ocean only — `memory/reference-cuda-eos-divergence.md`): density
  Δ≈3.18e-12 stable, Av/Kv≈0.095 isolated threshold-flips, u/v≈3.7e-4/5.9e-5, pgf≈8e-18, + M4.2's
  `eta_n`≈9.4e-11. No new divergence class through M4.2. **(pi = zero ice → says nothing about the ice scatters.)**
- **OpenMP pi** (M4.2 floor): T≈1.8e-15, Av/Kv≈2e-17, u/v/eta≈1e-18 — the ocean scatters/CG-dot. Again zero ice.
- **C ↔ Fortran 2-yr budget** (the absolute target — run `scripts/eps_climate_compare_2yr.py`, C
  `eps_2yr_dt1800` vs `fortran_pp_2yr`): _fill in the corr/bias/RMS per field here as the reference budget._

## The M3.2 runs

| backend | job | partition | run | time est. | output |
|---|---|---|---|---|---|
| CUDA   | `jobs/job_m32_cuda_core2` | gpu (8×A100/2 nodes, dist_8) | 2-yr CORE2 (34560 steps) | **measured 0.731 s/step** (job 25163175, post-M4) → **2-yr ≈ 7.0 h** (1-yr ≈ 3.5 h); fits the 12 h gpu wall | `/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda` |
| OpenMP | `jobs/job_m32_omp_core2`  | compute (2 nodes, 32 ranks × 8 threads) | 2-yr CORE2 | ~1–2 h (CPU; ⚠️ rebuild `build-omp` first — stale since M4.3c) | `/work/ab0995/a270088/port2/m32_omp` |

Both write monthly means `<var>.fesom.{1958,1959}.monthly.nc` (JRA55 1958→1959 rollover — confirm in the log).

**Perf note (post-M4 GPU step, job 25163175):** 0.731 s/step on dist_8 — only ~15% faster than M3.1's
0.86 s/step *despite* moving the CG (M4.2) + the whole ice step (M4.3) onto the device. The per-step time
is now dominated by the **host-staged halos** (device→host→MPI→host→device): the CG's ~90–127 iters/step
× its pp/rr exchanges, the EVP's 120 subcycles × uice/vice, the FCT's ~21 brackets — and moving the CG/EVP
to the device actually *added* PCIe round-trips for those halos (they were host↔host at M3.1). **GPU-aware
MPI to halo on-device is the M5 unlock**, not more kernel porting. (CORE2 is also small/bandwidth-bound,
M3.1 note.) M3.2 is about *fidelity*, not speed — but this is the headline for the M5 prompt.

### M5.1 result (GPU-aware MPI on-device halo) — measured 2026-05-27

GPU-aware MPI was **proven** (`openmpi/4.1.5-nvhpc-24.7`; the old `4.1.2` is `--without-cuda` and SEGFAULTs
on device pointers — [[reference-cuda-aware-mpi]]) and the device-pointer halo path built + validated
(L47). After flipping the CG + momentum + gm + ice-FCT brackets to the on-device exchange, CORE2 dist_8
(startup-free internal loop timer, 2 reps each):

| config (CORE2 dist_8, 8×A100/2 nodes) | s/step | vs host-staged |
|---|---|---|
| host-staged (`FESOM_HOST_HALO=1`, same binary) | 0.780 (0.7818 / 0.7779) | — |
| **device-halo** (GPU-aware MPI) | **0.716** (0.7166 / 0.7156) | **8.2% faster** |

**The payoff is modest (~8%), and that is itself the finding.** The win is the **large nod3D fields**
(~48 MB — the full-field PCIe sync was real); the **small nod2D hot loops (CG/EVP, ~1 MB) barely benefit**
(PCIe-cheap; the device path still fences). The *other* big nod3D halos (kpp `diffK`/`blmc`, eos pressure)
are **host-bound by `fesom_smooth_nod3D`** (a host op forcing the round-trip regardless) → not device-halo
candidates. So ~8% is near the ceiling for this optimization on this config; **the "halo-bound" premise was
only partly right.** Correctness: device-halo == host-staged at the CUDA run-to-run noise floor (no NEW
divergence — data path, not arithmetic). The next real GPU levers: port the kpp/eos **smoothing** to device,
batch/fuse the CG's ~1000 launches/step, or a **bigger mesh** to fill the A100s. (Validated; not bit-identity
— CUDA is non-deterministic run-to-run via the atomic scatters, L47.)

### M5.1b — bigger mesh (farc, 638k nodes × 48 lev ≈ 30 M 3-D pts, ~5× CORE2) — measured 2026-05-27

Tested the "CORE2 is too small / a bigger mesh fills the A100" hypothesis on **farc** (high-res, ~638k
nodes). De-risk: np=1 on one A100 reads + runs + **fits in ~8.3 GB** of 3-D fields (so any `dist_N` fits on
a 40 GB card), stable at dt=900, but **8.66 s/step with ~276 CG iters** (the stiff high-res SSH does ~2×
CORE2's CG iterations). Then host-staged vs device-halo (internal timer, 2 reps, dt=900, snap off):

| farc config | hardware | host-staged s/step | device-halo s/step | device gain |
|---|---|---|---|---|
| `dist_4` | 4 A100 / **1** gpu node | 7.28 | **6.38** | **+12.3%** |
| `dist_8` | 8 A100 / **2** gpu nodes | 3.73 | **3.30** | **+11.5%** |
| `dist_256` | 256 cores / **2** compute nodes (CPU, build-serial) | — | **0.445** | — |

Three findings:
1. **The device-halo gain GREW with the mesh: 8.2% (CORE2 dist_8) → ~12% (farc).** Bigger per-rank nod3D
   fields ⇒ the eliminated full-field PCIe sync moves more bytes. Confirms the L47 mechanism.
2. **The GPU strong-scales near-linearly: dist_4 → dist_8 = 6.38 → 3.30 = 1.93× on 2× GPUs.** The A100s are
   NOT saturated — adding GPUs helps (the CG iter count is partition-independent, so per-iter work halves).
3. **Node-for-node the CPU is still ~7.4× faster** (2 compute nodes 0.445 vs 2 gpu nodes 3.30) — **a bigger
   mesh did NOT flip the GPU↔CPU verdict.** ⚠️ **I first blamed the CG here — that was WRONG (see §M5.2):
   measured, the CG is only 1–5 % of the step.** The "16.5 ms/CG-iter" was a miscalculation (3.3 s ÷ 200 iters
   conflates the CG iterations with the *whole* step; the CG actually runs once/step at ~1.75 ms/iter and is
   a tiny slice). The real bottleneck is the **bulk ocean + sea-ice kernels** (~95–99 % of the step) — still
   to be profiled per-substep. (np=1 farc was 8.66 s/step, but that too is dominated by the bulk kernels, not
   the CG.)

**Conclusion:** chasing bigger meshes or more halo flips will NOT make the GPU competitive node-for-node. The
*why* was nailed down in §M5.2 below — and it is **NOT** the CG (my first guess). Jobs: `jobs/job_farc_gpu_np1`,
`jobs/job_gpuaware_time_farc_dist{4,8}`, `jobs/job_farc_cpu_dist256`.

### M5.2 — the CG is NOT the bottleneck (measured, 2026-05-27)

Added a startup-free **CG-share timer** (env `FESOM_CG_PROFILE=1`; fence+`MPI_Wtime` around the CG iteration
loop, accumulated over the timed window, printed by the loop timer) + applied two **bit-identity-preserving**
CG optimizations (fuse each SpMV with its following dot into one `parallel_reduce` — same row-order reduction
→ Serial np1+np2 still byte-identical to golden; fuse the sp0/sp1 dots into ONE 2-element `MPI_Allreduce`,
3→2 collectives/iter). farc dist_4 (CG-opt build):

| leg | s/step | **CG share** | CG iters/step | CG ms/iter |
|---|---|---|---|---|
| host-staged | 7.44 | **5.2 %** (0.38 s) | 219 | 1.75 |
| device-halo | 6.55 | **1.0 %** (0.067 s) | 219 | 0.305 |

**The CG is only 1–5 % of the step → it is NOT the bottleneck (this corrects §M5.1b).** Consequences:
- The CG fusions are correct + safe but **immaterial to the step** (the CG is tiny). The valuable artifact is
  the CG-share timer, which revealed the truth. (The 2 % apparent slowdown vs the m5.1 baseline is the
  timer's per-step fences — now gated behind `FESOM_CG_PROFILE`.)
- The device-halo makes the **CG itself 5.7× faster** (1.75 → 0.305 ms/iter — the per-iter cost was the 2
  host-staged nod2D halo syncs; at farc's bigger nod2D + 219 iters these are NOT cheap, unlike the CORE2
  finding). But since the CG is tiny, that's only ~⅓ of the device-halo's 0.89 s/step total gain; the other
  ⅔ is the ocean nod3D halos.
- **The real bottleneck is the bulk ocean + sea-ice kernels (~95–99 % of the step), un-profiled.** Likely
  suspects: the per-node/per-element TDMA solves (KPP, `impl_vert_visc`, vertical tracer diffusion, GM
  `fer_gamma`) — serial-within-column, low arithmetic intensity; the many FCT launches; the host-bound
  kpp/eos `fesom_smooth_nod3D`, the salinity floor, and the ice host parts. **Next: a per-substep timer in
  `fesom_step.cpp` (the CG-share pattern, applied to each substep) to find which kernels dominate** — that,
  not the CG, is the path to a GPU win.

### M5.3 — where the ocean's 82% goes: ~half kernels, ~half full-field PCIe (2026-05-27)

Per-phase step profiler (`FESOM_STEP_PROFILE=1`, `jobs/job_gpuaware_prof_core2`) on CORE2 dist_8
(device-halo, 0.738 s/step):

| phase | % of step |
|---|---|
| **ocean** (`fesom_timestep`) | **82.2%** |
| sea-ice | 7.4% |
| forcing | 5.3% |
| CG (within ocean) | 4.6% |
| coupling | 1.5% |

⇒ **the sea ice is NOT the GPU trouble (only 7.4%); the ocean dominates** (the Fortran/CPU "it's the ice"
intuition does not carry over — the ice is 2-D and cheap, the ocean carries the heavy 3-D physics).

`nsys` on np=1 CORE2 (`jobs/job_nsys_core2_np1`, per-kernel + memcpy) split the ocean's 82% ~half/half:
- **3-D physics KERNELS**: FCT tracer advection **~30%** (the biggest — ~20 kernels × 2 tracers), momentum
  ~15%, diffusion (Redi `diff_hor` + vertical TDMA `diff_ver`) ~14%, GM (`sigma_xy` 5.6% alone) ~10%,
  ALE `vert_vel` ~5%, KPP ~1%.
- **FULL-FIELD PCIe**: `cudaMemcpy` = **82.8% of CUDA-API time**, **~13 GB/step** at np=1 (copies up to
  188 MB = full elemvec fields) = the host-staged ocean halos NOT flipped in M5.1 (full-field `sync_host`/
  `sync_device`) + the EOS/KPP `smooth_nod3D` round-trips. **This is the removable cost.**

### M5.4 — lever A: flip the remaining ocean halos to device-halo (in progress)

The host-staged ocean halos use the M1.5 **split-rail** pattern: device kernel → `sync_host` (OUT) → host
`fesom_halo_exchange` → the consumer's `modify_host(); sync_device()` (IN re-push). Flipping each: replace
the OUT-rail + halo with `fesom_halo_field`, and **remove the downstream IN re-push** (the field stays
device-resident with its halo; else the re-push clobbers the device halo). Safe because: the per-kernel
verifies read the **raw alias** (not `h_checked()` → no SYNCCHECK assert), and on Serial every sync is a
no-op (host==device), so removing them is a no-op there. Validate EACH: Serial np1+np2 bit-identical +
CUDA pi dist_2 A/B at the run-to-run noise floor + re-time CORE2 dist_8.

**Progress (CORE2 dist_8, device-halo s/step; host-staged baseline ~0.77):**

| flipped | device s/step | total gain |
|---|---|---|
| M5.1: CG, momentum (`uvnode_rhs`/`u_b`/`v_b`), gm (`tr_xy`/`tr_z`), ice-FCT | 0.716 | 8.2% |
| **M5.4: + `uv_rhs`** (elem3D, 3×/step across momentum substeps 4–6) | **0.658** | **14.3%** |

`uv_rhs` alone added ~8% (0.716→0.658) — confirming the PCIe finding. **Remaining targets** (by payoff):
`pgf_x`/`pgf_y` (elem3D, big), the FCT internal halos (`fesom_tracer_adv.cpp`), `uvnode` (nod3D), `ssh_rhs`
(nod2D, small), tracer-diff; the **GM chain** (8 halos — its device kernels read *owned*, so some halos may
be verify-only → trace before flipping); **NOT** EOS/KPP (smoothing-bound — need `smooth_nod3D` on device).

## The comparison

`scripts/m32_climate_compare.py <backend_dir> --label <CUDA|OpenMP>` — annual-mean surface stats
(corr / bias / RMS / |Δ|max) per field per year + a year-to-year DRIFT check, vs **two** references:
- **vs the C-port** (`eps_2yr_dt1800` == Serial == bit-identical): this diff IS the backend's
  scatter/reduce drift, isolated from the Fortran↔C budget. **This is the headline M3.2 metric.**
- **vs Fortran** (`fortran_pp_2yr`): the absolute climate budget (backend-vs-Fortran should ≈ C-vs-Fortran).

## Verdict criteria (PASS)

- corr ≈ 1.0 for sst/sss/ssh/a_ice/m_ice (and uice if present), both years;
- backend-vs-C bias/RMS **bounded and clearly ≤ the C-vs-Fortran budget** (the scatter drift is below the
  model's own Fortran↔C uncertainty);
- **DRIFT ≈ 0** (year-2 bias ≈ year-1 bias) — the scatter perturbation does not run away over the years;
- no NaN/blow-up (T/S ranges bounded in the run log).

## Results

### CUDA 2-yr — _TODO_
_(paste the `m32_climate_compare.py --label CUDA` table; note the measured s/step + total wall.)_

### OpenMP 2-yr — _TODO_
_(paste the `m32_climate_compare.py --label OpenMP` table.)_

### Verdict — _TODO_
_(PASS/FAIL vs the criteria above; if a field drifts, which scatter/reduction is the suspect and whether
edge-coloring / a deterministic reduction (M5) is warranted.)_
