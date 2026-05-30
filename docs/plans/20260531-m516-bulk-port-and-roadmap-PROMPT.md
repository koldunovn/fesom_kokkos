# Next session — M5.16 (port `fesom_bulk_compute` to device) + the GPU-perf lever roadmap

*Paste this whole file to start. Self-contained. Written 2026-05-30 at the close of M5.15.*

---

## 0. TL;DR — where we are

The whole FESOM2 **C→C++/Kokkos** port is device-resident (M0–M4: ocean + sea-ice on the GPU, Serial bit-identical to the C twin). We are deep in the **M5.x GPU-performance campaign**. Just finished **M5.15 (GM-chain device residency), climate-validated.** Branch **`m515-gm-residency`**, HEAD **`1c19a3a`**.

**Perf arc, NG5 dist_16 (7.4M nodes, the production mesh):** 16.27 → 6.12 (M5.13) → 3.80 (M5.14) → **3.456 s/step** (M5.15). **Node-for-node GPU/CPU: 3.76× → ~0.80×** (the GPU is now ~25 % *faster* than a CPU node). CORE2 dist_8: 0.731 (post-M4) → **0.1424 s/step** (M5.15).

**The through-line is MEASURE-FIRST.** A clean re-profile this session overturned M5.14's "compute-bound" verdict: **the step is data-movement / halo-latency bound — GPU utilization is only ~30 %** (the smoother, the #1 *kernel*, is ~8 % of *wall*). The CG/EVP collective comms is **dead** (`Allreduce` 0.2 %). So levers are ranked by what the profiler says, not intuition. Do NOT skip the re-profile after each step — the wall moves.

---

## 1. THE IMMEDIATE TASK — M5.16: `fesom_bulk_compute` → device

### Why (measured, decisive)
The forcing-phase split (gated `FPROF` in `fesom_main.cpp:1045`, run via `job_ng5_prof`): **`force:bulk_compute` = 0.5517 s/step = 16.0 % of the step** (NG5 dist_16) — the L&Y09 air-sea bulk formulae run **single-threaded on the GPU build's Kokkos-Serial host** while the GPU idles (the **blmc/L49 trap**: a host loop nsys can't see — it profiles GPU kernels only — but a huge real cost). `force:jra55_read` is only 0.098 s (2.8 %) and stays host.

### Payoff (why this beats Lever B / C)
- **~16 % contained win** (one routine), bigger and more contained than overlap-B.
- **It dissolves the residency wall**: `T` (SST) and `uvnode` (surface velocity) are currently synced **DtoH every step** *only* because the host bulk reads them (~760 MB/step on NG5, the "required" reads that ended M5.15's residency). Once bulk reads them **on the device**, those DtoH disappear — bulk is simultaneously a compute port AND the last residency win.
- **Climate-safe**: bulk is a per-surface-node **MAP** (each node's flux from its own SST/wind/atmospheric state — no halo, no scatter, no reduction) → bit-identical on Serial, gate-only validation (like the EOS/KPP/ice-thermo device maps).

### The work (the port)
- **`fesom_bulk_compute`** is `src/fesom_bulk.cpp:226` — a host loop over surface nodes (L&Y09: saturation vapor pressure, drag/transfer coefficients, sensible/latent/longwave heat flux, wind stress). Port it to a `KOKKOS_LAMBDA` over `nod2D` (myDim+eDim). **Pattern: the `IceThermC` POD-of-constants device twin (L45)** — pack the scalar namelist constants into a POD struct captured by value; or `KOKKOS_INLINE_FUNCTION` helpers like the ice-thermo column physics.
- **Reads** → make device-current:
  - The **8 JRA55 forcing fields** (in `fesom_jra55 jra` — currently host arrays after `fesom_jra55_step_cal` interpolates them): wrap as device Views + **HtoD the 8 interpolated nod2D surface fields each step** (far smaller than the 3-D `T`/`uvnode` DtoH it replaces → net PCIe still improves). The NetCDF *disk read* + time-interp (`fesom_jra55_step_cal`) **STAYS host** — only the bulk math moves.
  - **SST** = `T` surface row (device-resident already) + **`uvnode`** (surface velocity, device-produced by `fesom_compute_vel_nodes`, currently synced to host for bulk). Read both on device.
- **Writes** → `forcing.stress_surf` (element, 2×elem2D), `forcing.stress_node_surf` (node), `forcing.heat_flux`, evap/precip etc. — per-entity maps; make the `forcing` struct device-resident (it's consumed by the ocean momentum substep + the ice step). Check each consumer reads on device.
- **THE RESIDENCY UNLOCK (do after the port validates):** remove the `T` sync_host (`fesom_step.cpp:892`) and the `uvnode` sync_host (`fesom_step.cpp:334`) — the L50 "required host reader" is now on-device. Re-run `job_ng5_prof`: expect `force:bulk_compute` → ~0 **and** the `T`/`uvnode` DtoH gone from `deep_copy`.
- ⚠️ **`fesom_sss_runoff_step`** (`fesom_main.cpp:1054`) also reads `S` surface on host — but it's **gated `use_sr`** (OFF in the gate + climate runs: no SSS-restoring file). Leave it (flagged); or port if you do an SSS-active run.
- ⚠️ Confirm `fesom_compute_vel_nodes` (the `uvnode` producer) is on device (it is — uvnode is device-resident per M5.9-pin); if any piece is host, port it too (it feeds bulk).

### Validation (this is FORCED-ONLY physics — the verify is CORE2, NOT pi!)
- ⚠️⚠️ **pi uses analytical wind stress and NEVER calls `fesom_bulk_compute`** (`jra55_year` ≤ 0). So a new `bulk` verify key (Serial C-twin max|Δ|==0) must run on **CORE2 via SLURM with JRA55 active** — exactly like the ice verify (L42). Pi smoke is INSUFFICIENT here (same lesson as the M5.9 stale-host bug that hid at pi=1e-17, [[feedback-gpu-fidelity-gate]]).
- Full ladder: Serial `bulk` verify max|Δ|==0 (CORE2 SLURM) → Serial CORE2 bit-identical to the oracle → SYNCCHECK → **`gpu_fidelity_gate.sh --fresh-oracle`** (CORE2-active-ice; bulk drives the air-sea fluxes so the gate exercises it heavily) → **1-yr climate** to close (`job_m32_cuda_core2` NSTEPS=17280 + `scripts/m32_climate_compare.py` vs `m32_cuda_m515_1yr` + Fortran/C — expect statistically identical, the D22 4th-5th-sig-fig floor).
- Per-step success signal: `force:bulk_compute` drop (the gated FPROF timer) + the `deep_copy` GB/step drop (T/uvnode gone).

---

## 2. THE LONG-TERM LEVER ROADMAP (evidence-ranked — re-measure between each)

1. **M5.16 — bulk-compute device port** (~16 %, climate-safe, finishes residency). ← THIS prompt.
2. **Lever B — interior/boundary halo overlap** (the ~47 % `MPI_Waitall`): start the device halo exchange, compute interior nodes while it's in flight, finish the boundary-dependent kernel after. **Climate-safe** (numerics-neutral reordering) but **INVASIVE** — split every haloed kernel into interior/boundary node sets + restructure `fesom_halo_device`. ⚠️ a chunk of the 47 % is **load-imbalance idle** (rank waiting on a slower rank), which overlap can't recover → **re-measure the overlappable-vs-imbalance split first** (after M5.16, profile per-rank `MPI_Waitall` — if it's mostly imbalance, B's ceiling is low and it's not worth the invasiveness). Separate session + branch.
3. **Lever C — kernel coalescing / memory-layout refactor** (`fesom_field.hpp` rank-1 `T*` → `View<double**>` + launch fusion): the long-shelved data-layer refactor. **LAST** — the GPU is only ~30–34 % utilized, so faster kernels help only the busy slice; a 2× on the hot kernels (FCT/smoother/momentum, `ncu` showed them memory-leaning ~50–58 % SOL) ≈ a few % of step *today*. Revisit once M5.16 + B raise utilization. Own branch + session; touches the data layer → invasive.
4. **Lever D — deployment guidance** (free, no code): more work/GPU → nearer parity (the data/compute-balance lesson L58; dars dist_8 was closer to parity than dist_16). Don't over-decompose in production — state the per-rank-work operating point (near the A100-80GB ceiling, ~900k nod2D/rank fits).

**DEAD / DONE — do NOT re-pursue:** CG/EVP comm-reduction (`Allreduce` 0.2 %, confirmed M5.2 + this session); more device-residency (exhausted after M5.16 unlocks T/uvnode); the "compute-bound → Lever C first" reading (overturned — GPU is idle-bound, not compute-bound).

---

## 3. VALIDATION LADDER (per device-kernel change — ALL must pass; cheap→costly)

1. **Serial per-kernel verify:** `FESOM_KK_VERIFY=<key> ./build-serial/fesom_port …`; PASS = `grep FESOM_KK_VERIFY= run.log | grep -v 0.000e+00` empty (max|Δ|==0). Keys: eos/pp/kpp/pgf/vrhs/vfilt/ivisc/ale/gm/tradv/trdiff/ssh + ice (evp/icemap/icefct/icethermo/iceflux) + **add `bulk`**.
2. **Serial bit-identity:** pi np1 vs `docs/reference/c_baseline_snapshots/pi`; np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none` vs `/scratch/a/a270088/pi_np2_ref_m13_nocma` (L18). `scripts/diff_snap.py` takes **directories**, zero-tol. **For forced-only physics (bulk/ice), bit-identity must be CORE2** (pi never runs them).
3. **SYNCCHECK:** `build-synccheck` clean exit (catches a dropped sync_host a host reader needs — but ⚠️ raw `.h()` reads BYPASS it; trace the reader's verify-gating in source, L60).
4. **CORE2-active-ice CUDA fidelity gate:** `scripts/gpu_fidelity_gate.sh --fresh-oracle` (`--fresh-oracle` every milestone — each edits `fesom_step.cpp` → ULP drift, L51). PASS = all fields within the climate-close ceilings, no staleness regression.
5. **1-yr CORE2 CUDA climate** to close a campaign (L58: never settle fidelity with the 20-step gate). `scripts/m32_climate_compare.py <dir> --label … --years 1958` — apples-to-apples: re-run the PRIOR binary's 1-yr through the *same* current script (L58 lesson 2).

**Reusable harness:** `jobs/job_m515_serial_val` (1-node compute: verify + np1/np2 bit-id + SYNCCHECK in one — **update its `FESOM_KK_VERIFY=` to include `bulk`**, and note it's pi → for `bulk` you need a CORE2 variant). `jobs/job_ng5_prof` (clean step + `deep_copy` + per-substep + the `force:*` split — submit with `--nodes=4 --ntasks=16 --export=ALL,TAG=…`). `jobs/job_nsys_ng5` (MPI-traced nsys; mine the sqlite for the windowed GPU-util / DtoH-by-copyKind / MPI_Waitall decomposition — the script pattern is in the M5.15 session transcript). `src/fesom_field.hpp` `FESOM_SYNC_LOG` (compile-guarded per-field DtoH/HtoD attribution).

---

## 4. HARD CONSTRAINTS (carry these every session)

- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB home quota; NG5 needs `snap_every=-1`, rank-0 gather OOMs ~66 GB). Big/CORE2 runs via **SLURM compute/gpu nodes, never the login node** (login-node NetCDF4 `nc_create` hits EACCES on Lustre).
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware). The `env.sh` `openmpi/4.1.2` is `--without-cuda` → device-ptr MPI SEGFAULTs ([[reference-cuda-aware-mpi]], L47). CPU builds use `env.sh`.
- **Validation jobs need `source /sw/etc/profile.levante` BEFORE `env.sh`** (init the spack module env, else `module load` picks the wrong HDF5 → `nc_create` EACCES) and **`mkdir -p` the output dir** (`fesom_port` doesn't create it). L60.
- **Same-day perf baselines only** (Levante node mix + contention ~5 % noise; [[feedback-perf-same-day-baseline]]) — rebuild the prior commit + run the same job today for a real attribution. nsys-inflated vs clean (`job_ng5_prof`) step numbers differ — compare like-for-like.
- **Device-halo/sync changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); pi is insufficient (no ice / no bulk).
- **Commit/push only when the user asks.** KPP is the default mix_scheme ([[feedback-kpp-default]]).
- **Build dirs:** `build-cuda` (M5.15, on `m515-gm-residency`), `build-serial`, `build-synccheck`, `build-omp` (STALE — rebuild before any OpenMP leg). All four carry M5.15.

---

## 5. POINTERS

- **Memory:** [[project-m515-gm-residency]] (full M5.15 state + the next-lever measurement), [[project-m514-residual-residency]], [[project-m513-pcie-campaign]], [[reference-build-run]], [[reference-cuda-aware-mpi]], [[feedback-gpu-fidelity-gate]], [[feedback-perf-same-day-baseline]], [[reference-m32-references]].
- **Docs:** `docs/GPU_FIDELITY.md` §M5.15 (the decomposition + per-field attribution + the bulk-compute finding + the climate PASS), `docs/SCALING_NG5.md`, `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–L60 — **L60 = the M5.15 lessons**: GPU-util-is-the-metric, the FESOM_SYNC_LOG tool, verify-only-sync removal + the raw-h() SYNCCHECK-bypass caveat, residency-kills-two-costs, residency-exhausts, the validation-run gotchas).
- **Key commits (branch `m515-gm-residency`):** `baed09c` groundwork+toolkit · `81b8947` T1+T2 (GM halos) · `a39ac70` T3 (bvfreq/dbsfc) · `99bc414` forcing-split (the bulk-compute finding) · `1c19a3a` climate PASS / M5.15 complete.
- **References for the climate compare:** Fortran `/scratch/a/a270088/fortran_kpp_5yr_fix`, C-port `/work/ab0995/a270088/port/kpp_5yr_fix`; prior CUDA 1-yr runs `m32_cuda_m514_1yr`, `m32_cuda_m515_1yr` (all in `/work/ab0995/a270088/port2/kokkos_gpu_runs/`).
- **Templates:** the device-kernel ports `fesom_{kpp,gm,tracer_adv,tracer_diff,ssh,ice_*}.cpp` + `fesom_step.cpp` substep chain. For bulk: the per-node-map + POD-constants pattern of `fesom_ice_thermo.cpp` (`IceThermC`, L45) is the closest analogue.

## 6. Bottom line for the next session
M5.15 is closed (climate-validated, −9 % step). **Start M5.16: port `fesom_bulk_compute` to a device per-surface-node map** — a measured ~16 % win that's climate-safe and finishes residency. Re-profile after to confirm + decide whether Lever-B overlap is worth its invasiveness (re-measure the load-imbalance split). The GPU is idle-bound, not compute-bound — keep measuring before committing to any lever.
