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
