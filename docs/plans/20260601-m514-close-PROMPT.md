# Next-session prompt — M5.14 close-out: the 1-yr climate verdict + build-dir cleanup

> Copy-paste to start. M5.14 (Lever A) is substantively DONE — all 4 flips committed + gated, node-for-node
> PARITY CROSSED. This session just: (1) read the queued 1-yr climate verdict, (2) clean up the temp build dir,
> (3) decide the next track. Load memory [[project-m514-residual-residency]] first.

## Where we are (M5.14 = ✅ done, climate confirmation queued)

Branch `m514-residency`. **Lever A finished device residency on the last host-staged nod3D fields.** 4 flips, each
the L48 split-rail recipe, each passed Serial bit-identity + the CORE2-active-ice CUDA fidelity gate + device-mean
correctness:
- t4 device-mean-accum `2611936` (enabler + stale-3D-means fix) · **S** = g2 mirror-T `491ccb8` (+ device salinity
  floor `fesom_salinity_floor_kk`) · **density** `84c1d8d` · **fer_w + w_i** `0bc7da9`.

**Headline (NG5 dist_16, SAME-DAY baseline both sides): step 6.14→3.80 s/step (−38%); deep_copy 11.0→6.57 GB/step
(−40%); GPU 3.805 / CPU 4.327 = 0.879× → the GPU is ~14% FASTER than the CPU = node-for-node PARITY CROSSED.** This
overturned L58's "~1.4× asymptote" (S was a hidden PCIe chunk). Regime is now compute-bound (smoother = #1 GPU
kernel, 25.7%). T-trim evaluated + SKIPPED (~4% of deep_copy, DualView-sidestep risk). Lesson L59; docs written
(`SCALING_NG5.md` §M5.14, `GPU_FIDELITY.md` §M5.14).

## ⚠️ TASK 1 — read the 1-yr climate verdict (the only unfinished validation)

The authoritative fidelity check is QUEUED (GPU partition was full + the 2 32-node scaling jobs held the
2026-05-30 08:29/09:00 reservation, so the 2-node climate run estimated start 09:50, finish ~12:15). Job was
`25238839`, run dir `/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_m514_1yr`, binary `build-cuda-m514`,
`M32_NSTEPS=17280` (1 yr).
- `sacct -j 25238839` → confirm COMPLETED rc=0; check the run.log: 1959 rollover, T/S bounded (no NaN), monthly files.
- **If it never ran** (e.g. cancelled / build-cuda-m514 deleted): re-submit `jobs/job_m32_cuda_core2_m514`
  (`--export=ALL,M32_NSTEPS=17280,M32_TAG=_m514_1yr`) — but ONLY if build-cuda-m514 still exists (see Task 2).
- Compare with `scripts/m32_climate_compare.py` vs the M5.13 1-yr (`m32_cuda_m513_1yr`) + Fortran/C refs
  (`fortran_kpp_5yr_fix` + `kpp_5yr_fix`), **snapshot AND `.monthly.nc` means**. EXPECT: statistically identical to
  the M5.13 run on every field (corr~1, bias O(1e-4)) — the M5.14 flips are the same climate-neutral class (L58).
  Pay attention to **salt/sss/density** (the newly device-resident + device-mean fields) — they should match the
  M5.13 run to GPU-noise (D22), NOT drift.
- Fill the result into `docs/GPU_FIDELITY.md` §M5.14 ("[Result to be filled ...]") and flip the memory + index from
  "climate IN FLIGHT" to the verdict.

## ⚠️ TASK 2 — build-dir hygiene (the standing cleanup obligation)

CUDA work this campaign used a **temporary `build-cuda-m514`** (the canonical `build-cuda` was claimed by the
scaling sweep). Once **BOTH** the scaling sweep has landed **AND** the 1-yr climate has finished reading
build-cuda-m514:
- Rebuild canonical `build-cuda`: `source ./env_cuda.sh && cmake --build build-cuda -j16` (it's now stale =
  pre-Lever-A; the m514 branch is HEAD).
- `rm -rf build-cuda-m514`
- Delete the temp job copies: `jobs/job_gpu_fidelity_dev_m514`, `jobs/job_ng5_prof_m514`, `jobs/job_nsys_ng5_m514`,
  `jobs/job_m32_cuda_core2_m514` + `/tmp/*_jid.txt` scratch.
- (build-serial / build-synccheck were rebuilt with the flips — no-ops on Serial, so CPU/Serial timings unchanged.)

## TASK 3 — finalize + next track
- Move `docs/plans/20260531-lever-a-residual-residency.md` → `docs/plans/completed/`.
- Update the big handoff `docs/KOKKOS_HANDOFF.md` status blob (M5.14 done, parity crossed) — deferred from the
  M5.14 session because the climate was queued.
- **Next track = NOT more residency** (PCIe is solved). The step is compute-bound now. Two levers, separate
  session + branch: **B = climate-safe MPI interior/boundary overlap** of the device-halo (`fesom_halo_device`);
  **C = the heavy-kernel coalescing refactor** (`fesom_field.hpp` rank-1 → `View<double**>`) + launch fusion —
  the smoother (#1 kernel, 25.7%) + FCT are the targets. Both attack compute/launch, not data movement.

## Pointers
- Memory: [[project-m514-residual-residency]] (full state), [[project-m513-pcie-campaign]] (base), L58/L59 in
  `docs/KOKKOS_PORTING_LESSONS.md`.
- Perf jobs: `job_ng5_prof_m514` / `job_nsys_ng5_m514` (final binary) + `job_ng5_prof` (build-cuda = pre-Lever-A
  same-day baseline) + `job_ng5_scaling_cpu` (CPU dist_512). Gate: `gpu_fidelity_gate.sh` + `job_gpu_fidelity_dev_m514`.
- Build GPU with `source ./env_cuda.sh`. Output → `/work/ab0995/a270088/port2/...`, never $HOME.

## Bottom line
M5.14 broke through to node-for-node parity (GPU ~14% faster than CPU at NG5 dist_16). Perf + per-flip fidelity
gates are DONE and committed. This session is just the 1-yr climate confirmation (expected PASS) + the temp-build
cleanup, then pick lever B or C.
