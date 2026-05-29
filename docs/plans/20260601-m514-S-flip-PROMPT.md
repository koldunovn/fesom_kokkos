# Next-session prompt — M5.14 Lever A: the S flip (and the rest of residual residency)

> Copy-paste this whole file to start the session. Self-contained; pointers at the end have detail.
> Flow: read this → load memory ([[project-m514-residual-residency]]) → read the plan
> `docs/plans/20260531-lever-a-residual-residency.md` → execute Task 5 (S) → density → fer_w/w_i → T-trim → measure → climate-validate.

---

## Where we are (read first)

**Branch `m514-residency`** (off `m513-pcie`). The post-M5.13 **re-profile is done** and chose the lever:
NG5 dist_16 step 6.42 s = **PCIe `cudaMemcpy` 2.81 s (44 %)** / MPI-sync 2.42 s (38 %) / kernels 1.19 s
(19 %); `deep_copy` **11 GB/step**. **CG 0.9 %, EVP+ice 2.65 %, launches 0.2 % → pipelined-CG,
EVP-subcycle, Kokkos-Graphs all RULED OUT.** The lever is **Lever A**: finish device residency on the
remaining host-staged nod3D fields (S, density, fer_w, w_i). Figure `docs/figures/m513_reprofile_levers.png`.

**✅ Task 4 DONE + committed (`2611936`): device-side mean accumulation** (`fesom_io.cpp` +
`fesom_io_stream.{h,cpp}`). It both (a) **enables** S/density/T residency (the monthly-mean stream no longer
needs a per-step host copy of them) and (b) **fixes a latent bug**: the mean stream accumulated device-resident
`u/v/w/Kv/Av/bvfreq` from the **stale host alias** every step (host refreshed only on snapshot steps) → the 3-D
means were wrong. Now accumulated on-device (`field.d()` → per-var device buffer, 1 D2H/month at flush).
Validated: Serial device-vs-host **bit-identical** (`cdo` ± controls); CUDA staleness-fix confirmed
(u/v/Kv/Av differ >1e-3 host-vs-device); SYNCCHECK clean. Toggle `FESOM_IO_HOST_ACCUM=1` = host path (A/B).

## ⚠️ FIRST: build-dir hygiene (the user's standing instruction)

`build-cuda` is claimed by **2 pending 32-node scaling jobs** (`kk_ng5_gpu`/`kk_dars_gpu`). CUDA validation this
session used a **temporary `build-cuda-m514`** (CUDA-aware clone, built via `source ./env_cuda.sh`).
- `squeue -u a270088` — if those jobs have **landed**: rebuild canonical `build-cuda`
  (`source ./env_cuda.sh && cmake --build build-cuda -j16`) and **`rm -rf build-cuda-m514`** for a clean setup.
- If still pending: keep using `build-cuda-m514`; do **not** rebuild `build-cuda`/`build-serial` while their jobs queue.
- `build-serial` + `build-synccheck` are free (no pending CPU jobs) — use directly for Serial verify / pi / SYNCCHECK.

## The mandate: execute Task 5 — the S flip (= g2 = "do for S exactly what g1-T did for T")

S is the biggest target (lives in 13_fct, the biggest phase). It's **one atomic flip** (a partial flip
clobbers — L48). The pattern is mechanical: **every S re-push site sits right beside an already-converted
T-site** (look for the `M5.13g1-T: T values device-resident — no re-push` comments in `fesom_step.cpp`).
Mirror them for S. Exact edit list (grep-verified this session):

| piece | sites |
|--|--|
| Remove S `values` re-pushes (`modify_host()/sync_device()` IN, `sync_host()` OUT) | `fesom_step.cpp` **174-176, 248-250, 359-360, 812, 822, 829, 836, 886-887, 891** |
| Remove S `valuesold` re-pushes | `fesom_step.cpp` **813, 822, 830** |
| Remove cross-file ice re-pushes (ocean2ice/oce_fluxes read S on device) | `fesom_ice.cpp` **447, 618** |
| 2 host exchanges → `fesom_halo_field(…S…values_fld, FESOM_HALO_NOD3D, nl, 1, p)` | `fesom_step.cpp` **839, 901** |
| Host salinity floor (`if(S<0.5)S=0.5` over myDim+eDim × nlevels) → **device kernel** `S(i)=Kokkos::max(S(i),0.5)`, placed AFTER the post-trdiff device halo (matches exchange-then-floor order → bit-identical) | `fesom_step.cpp` **923-942** |
| **L57 init push** of S `values` AND `valuesold` before the time loop (mirror T @ `fesom_main.cpp:1024-1026`) | `fesom_main.cpp` |
| **Wire salt/sss into device-mean-accum** — add `resolve_salt_dev`/`resolve_sss_dev` (mirror `resolve_temp_dev`/`resolve_sst_dev`) + flip the `salt`/`sss` table rows' `dev_resolve` from `nullptr`. **COUPLED with residency — same commit** (else the salt mean reads stale device S) | `fesom_io.cpp` |
| Add S `values` to the snapshot-gated pre-I/O `sync_host` block | `fesom_main.cpp:1295` |
| **Gated** surface SSS `sync_host` (defensive, L50) — `fesom_sss_runoff.cpp:407` reads `S[surface]` on host, gated on `sr->sss_path[0]` (OFF in our gate/climate runs → keep but it is NOT gate-validated; flag for a future SSS-active run) | `fesom_step.cpp` |

Keep the `s_verify_*` capture blocks (gated; Serial host==device so they still see correct S). The device clamp
is bit-identical (no reduction/scatter). Note: `valuesold` is not halo-exchanged today — don't add one.

## Then (lower risk, after S):
- **Task 6 density** — `fesom_step.cpp:201` exchange → `fesom_halo_field`; remove re-push; add to pre-I/O sync;
  wire `resolve_density_dev` + table. (density has NO on-device consumer — the flip defers its every-step host-MPI/D2H.)
- **fer_w / w_i** — clean split-rail flips (no mean entanglement; not output fields). `fesom_step.cpp:700` (fer_w),
  `722` (w_i) → `fesom_halo_field`; **w_i also remove the cross-step re-push at `:494`** (consumed substep 6 before produced 12d).
- **T-trim** — now that the mean accumulates T on device, trim the line-900 full-field `sync_host` to surface-only (bulk SST).
- **Measure** (`jobs/job_nsys_ng5` + `job_ng5_prof`) → the new PCIe / deep_copy / GPU-CPU ratio; hunt any residual.

## Validation ladder (per flip)
1. Serial **`FESOM_KK_VERIFY=trdiff`** (+ `tradv`/`gm` for the FCT/Redi sub-blocks) `max|Δ|==0`. The device clamp must be bit-identical.
2. Serial **pi np1 + np2 bit-identical** (`scripts/diff_snap.py`, DIRS; np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`).
3. **SYNCCHECK clean** (`build-synccheck`).
4. **CORE2-active-ice `scripts/gpu_fidelity_gate.sh --fresh-oracle`** (pi INSUFFICIENT — no ice).
5. **Diff the `.monthly.nc` salt/sss means** (the gate is snapshots-only; means are the gate-blind path — `cdo diffn`).
   For S specifically: a CUDA A/B (`jobs/job_m514_io_ab_pi`, point it at the S binary) — salt mean should now be
   correct (device path) vs the host path; and Serial device-vs-host bit-identical.
6. **Final** (after all flips): 1-yr CORE2 CUDA climate compare (`scripts/m32_climate_compare.py`) vs the pre-Lever-A
   binary — snapshot AND mean — corr~1 / bias O(1e-4). Never settle fidelity with the 20-step gate (L58).

## Hard constraints / gotchas (do not relearn)
- Build GPU with **`source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`); env.sh's 4.1.2 SEGFAULTs on device ptrs.
- **`fesom_io_stream.h` must stay Kokkos-FREE** — it's included by a C unit test (`tests/test_io_stream_unit.c`)
  + a no-Kokkos target (`fesom_io_stream_dispatch.cpp`). All Kokkos lives in the `.cpp` (both `fesom_port`-only).
  The device-accum hook is a raw `real_t*` device ptr (`Kokkos::kokkos_malloc`); `delete`→`kokkos_free` before `Kokkos::finalize`.
- **L48** partial flip CLOBBERS — remove EVERY re-push. **L57** cross-step non-zero-init device field needs a one-time
  init push. **L50** surface-row host readers (bulk SST=T, SSS=S, uvnode wind) keep a targeted surface sync.
- Output runs → **`/work/ab0995/a270088/port2/…`, never `$HOME`** (60 GB quota). NG5 needs `snap_every=-1`.
- Same-day perf baseline only (rebuild the prior commit + run the same job).

## Pointers
- Plan + checklist (Task 4 ticked, Task 5+ pending): `docs/plans/20260531-lever-a-residual-residency.md`.
- Memory: [[project-m514-residual-residency]] (+ [[project-m513-pcie-campaign]] for the M5.13 base, [[feedback-gpu-fidelity-gate]]).
- The T-pattern template to mirror for S: `fesom_step.cpp` substeps at **806-901** (T fully device-resident there).
- Re-profile jobs: `jobs/job_nsys_ng5`, `jobs/job_ng5_prof`; the I/O A/B job `jobs/job_m514_io_ab_pi`; pi mesh `/home/a/a270088/port2/fesom2/test/meshes/pi`.
- Lessons to append on completion: L48/L50/L57 restated for g2 + a NEW lesson on the device-mean-accum + the stale-3D-means discovery + the C-safe-header trap.

## Bottom line
Task 4 (device-mean-accum) is the foundation + a real correctness fix, committed + validated. **S is fully
mapped — it's the g2 mirror-T flip, ~20 edits, one atomic change, the #1 blast-radius risk.** Execute it
carefully behind the validation ladder, then the easy ones (density, fer_w/w_i, T-trim), then re-measure +
climate-validate. Don't forget the `build-cuda-m514` cleanup when the sweep lands.
