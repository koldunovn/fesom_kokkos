# Next session — M5.17 (Lever B: attack the MPI communications) — MEASURE FIRST

*Paste this whole file to start. Self-contained. Written 2026-05-30 at the close of M5.16.*

---

## 0. TL;DR — where we are

The whole FESOM2 **C→C++/Kokkos** port is device-resident (M0–M4: ocean + sea-ice on the GPU, Serial bit-identical to the C twin). The **M5.x GPU-performance campaign** has finished device-residency (M5.13→M5.15) and the **M5.16 bulk-compute device port** (the L&Y09 air-sea formulae → a per-surface-node Kokkos map; the T/uvnode residency unlock).

**`master` is now CAUGHT UP** — fast-forwarded to M5.16 (tag **`m5.16-bulk-port`**; the prior whole-model-GPU + M3.2-validated state is tagged **`m5.9-pin`**). The 54-commit M5.11→M5.16 campaign is on `master`.

**Perf arc, NG5 dist_16 (7.4M nodes, the production mesh):** 16.27 → 6.12 (M5.13) → 3.80 (M5.14) → 3.456 (M5.15) → **2.677 s/step (M5.16, −22.5 % same-day same-node)**. **Node-for-node GPU/CPU: the GPU node is now ~1.6× FASTER than a CPU node** (GPU dist_16 2.677 vs CPU dist_512 4.350 s/step, 4 nodes) — up from **3.76× SLOWER** at the campaign start.

**The through-line is MEASURE-FIRST, and it has repeatedly overturned intuition** (M5.14's "compute-bound" verdict was wrong; the bulk port returned 22.5 % not the estimated 16 %; the "uice C↔F budget" was a script artifact). Residency is exhausted; the step is data-movement / halo-latency bound. The biggest remaining bucket is **`MPI_Waitall` (~47 % of the step on the M5.15 binary)** — **BUT a chunk of that is load-imbalance idle, NOT recoverable comm. Do NOT assume the 47 % is winnable. The FIRST task is to MEASURE the split.**

**Branch: `m517-mpi-comms`** (created off `master` @ `6059808` at the M5.16 close; this prompt is committed on it).

---

## 1. THE IMMEDIATE TASK — decompose the `MPI_Waitall`, THEN attack only what's recoverable

### Why MPI comms is the lever now
After M5.16, residency is done (T/uvnode were the last genuinely-required DtoH, L50/L61). The profiler put `MPI_Waitall` at ~47 % of the step (CG/EVP `Allreduce` is DEAD at 0.2 % — that flavor of comm reduction is confirmed not worth it, M5.2/L60). That 47 % is the biggest remaining bucket — but it is a SUM of three components, and only some are recoverable:

1. **Genuine comm latency / bandwidth** — time messages are actually in flight. Recoverable by **OVERLAP** (compute interior nodes while the halo exchanges).
2. **Many-small-message latency** — the halo posts many tiny per-field, per-neighbor messages (M5.13 nsys measured ~4575 transfers/step, H2D-dominant). Recoverable by **AGGREGATION** (pack several fields per neighbor into one message; fuse adjacent exchanges).
3. **Load imbalance** — fast ranks idling at the `Waitall` waiting for slow ranks to reach the exchange. **NOT recoverable by overlap** (it's a synchronization wait, not a comm wait). Only better mesh partitioning (Lever D) or a structural floor.

### STEP 1 — the GATE (do this BEFORE writing any overlap/aggregation code)
Re-profile the **M5.16** binary on NG5 dist_16 and SPLIT the `Waitall`:

- **Re-profile** (`jobs/job_ng5_prof`, M5.16 `build-cuda`, `--nodes=4 --ntasks=16 --export=ALL,TAG=m517_base,NSTEPS=35`): the new `MPI_Waitall` fraction (it grew *relatively* as the step shrank to 2.677). Confirm the clean s/step.
- **Per-rank `Waitall` variance** (`jobs/job_nsys_ng5` → mine the sqlite `mpi_event_sum` PER RANK — the script pattern is in the M5.15 session transcript / `docs/SCALING_NG5.md` §M5.13): **HIGH variance across ranks ⇒ load-imbalance-dominant; LOW variance (all ranks wait ~equally) ⇒ comm-dominant.**
- **Barrier-isolation experiment (the decisive, cheap one — IMPLEMENT THIS FIRST):** add an env-gated `MPI_Barrier(MPI_COMM_FESOM)` immediately *before* each halo exchange (in `fesom_halo_device.hpp`/`fesom_halo.cpp`, gated `FESOM_HALO_BARRIER=1`, off in production). With the barrier, the imbalance is absorbed BEFORE the exchange, so the remaining `Waitall` measures pure comm. Then `step(barrier=1) − step(barrier=0)` ≈ the **imbalance share**, and the residual `Waitall` ≈ the **overlappable comm**. This single experiment tells you whether Lever B has a ceiling worth the invasiveness.
- **Message count + size histogram** (`job_nsys_ng5` MPI trace): messages/step + the size distribution → many-tiny (→ aggregate) vs few-big (→ overlap/bandwidth).

### STEP 2 — pick the sub-lever from the data (do NOT pre-commit)
- **comm-latency-dominant, few big messages → interior/boundary OVERLAP** (§2A). The invasive one.
- **many-small-message-dominant → halo AGGREGATION / fusion** (§2B). Less invasive; attacks the message COUNT.
- **load-imbalance-dominant → STOP.** Overlap can't recover it. Document the ceiling, then either improve the FESOM mesh partition (deployment/Lever D) or move to **Lever C** (kernel coalescing, `fesom_field.hpp` rank-1 → `View<double**>`). Don't build overlap that can't pay.

### ⚠️ The discipline (the user's explicit ask, restated)
**MEASURE BEFORE CLAIMING.** The 47 % is not all recoverable. Run the barrier-isolation + per-rank-variance experiments FIRST and report the overlappable ceiling. Implement only the sub-lever the data supports, and re-measure same-day after each step (rebuild the prior commit on the same nodes — the M5.16 baseline procedure, [[feedback-perf-same-day-baseline]]).

---

## 2. IMPLEMENTATION NOTES (when the measurement says go)

### 2A — interior/boundary overlap
- **Split `fesom_halo_field` into non-blocking halves.** Today (`src/fesom_halo_device.hpp:94`) it is blocking: `modify_device` → (CUDA device-active) device-pack → GPU-aware `MPI_Isend/Irecv` → `Waitall` → device-unpack, all in one call (else the host-staged bracket). Add `fesom_halo_start(f,…)` (pack + post the Isend/Irecv, return immediately) + `fesom_halo_finish(f,…)` (Waitall + unpack). The device pack/unpack kernels + the device-buffer MPI live in `src/fesom_halo_device.cpp`.
- **Split the consuming kernel into interior + boundary node sets.** A halo updates the eDim (halo) rows of field F; the next kernel K reads F at eDim. K's OWNED nodes whose stencil touches eDim are "boundary"; the rest are "interior". Pattern: `halo_start(F)` → run K over INTERIOR owned nodes (no eDim read) → `halo_finish(F)` → run K over BOUNDARY owned nodes. Precompute the interior/boundary node masks ONCE at setup (a node in `[0,myDim)` is boundary iff its CSR/element stencil includes any node ≥ myDim). FESOM's `partit` send/recv lists (`com_nod2D`) already encode the boundary set — derive the mask from there.
- **Start with the 2–3 highest-cost haloed kernels** the Step-1 profile fingers (candidates: T/S `values`, `uvnode`/`uvnode_rhs`, the FCT internal halos, the GM chain, `Kv`/`Av`). Overlap those, MEASURE, then decide whether the rest are worth it. Do NOT split all ~20 haloed kernels blind.

### 2B — halo aggregation / fusion
- Coalesce the per-field halo exchanges into **one packed message per neighbor** carrying multiple fields (a single device-pack over a field LIST → one Isend/neighbor → one unpack). Attacks the ~4575-transfers/step message count directly. Less invasive than 2A (no kernel splitting), and composes with it later.
- Cheap first cut: fuse the obvious adjacent same-shape exchanges (e.g. the substep-1 nod3D set, or T+S `values` which are halo'd back-to-back).

---

## 3. VALIDATION LADDER — Lever B is NUMERICS-NEUTRAL → bit-identity MUST hold

Overlap-reorder and aggregation do NOT change the math — the halo values delivered are identical; only WHEN they arrive / how they're packed changes. So unlike the residency flips (which were Serial no-ops), **the np>1 bit-identity test now actually exercises the change and is the key gate.**

1. **Serial per-kernel verify** `FESOM_KK_VERIFY=<key>` max|Δ|==0 (the kernel split must not change owned-node compute).
2. **np2 bit-identity (THE key test):** pi np1 vs `docs/reference/c_baseline_snapshots/pi`; **np2** (needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`, L18) vs `/scratch/a/a270088/pi_np2_ref_m13_nocma`, zero-tol via `scripts/diff_snap.py` (takes DIRECTORIES, L19). At np2 the halo + the interior/boundary split run real code — a reorder bug shows HERE.
3. **SYNCCHECK** (`build-synccheck`) clean.
4. **CORE2-active-ice CUDA fidelity gate** `scripts/gpu_fidelity_gate.sh --fresh-oracle` (every milestone edits `fesom_step.cpp`/halo → rebuild the oracle, L51). PASS = the climate-close floor, no staleness regression.
5. **1-yr CORE2 CUDA climate** to close (`jobs/job_m32_cuda_core2 --export=ALL,M32_NSTEPS=17280,M32_TAG=_m517_1yr` + `scripts/m32_climate_compare.py <dir> --label … --years 1958` vs `m32_cuda_m516_1yr` + Fortran/C). Numerics-neutral → expect statistically identical (the D22 4th–5th-sig-fig floor). ⚠️ Apples-to-apples: re-run the PRIOR binary's 1-yr through the SAME script (L58).

---

## 4. HARD CONSTRAINTS (carry these every session)

- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB home quota). Big/CORE2/NG5 runs via **SLURM compute/gpu nodes, never the login node**.
- ⚠️ **NEW perf-run gotcha (M5.16):** the NG5 profile/scaling jobs write **~50 GB of monthly-mean `*.monthly.nc` per run EVEN with `snap_every=-1`** (the monthly accumulator is independent of snap_every). Clean them after a perf run (`rm <dir>/*.monthly.nc`, keep the log) or you'll silently fill `/work`. Consider adding a "no means" switch to `job_ng5_prof`.
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; the `env.sh` `openmpi/4.1.2` is `--without-cuda` → device-ptr MPI SEGFAULTs, L47). ⚠️ **`env_cuda.sh` PURGES the `git` module** — do git ops in a separate shell sourcing only `/sw/etc/profile.levante` (not env_cuda). CPU builds use `env.sh`.
- **Validation jobs need `source /sw/etc/profile.levante` BEFORE `env.sh`** + `mkdir -p` the output dir (L60).
- **Same-day perf baselines only** ([[feedback-perf-same-day-baseline]]) — rebuild the prior commit + run the same job today (the M5.16 procedure: save `git diff` as a patch, `git checkout --` the files, build, `cp` the binary to a stable path, restore the patch, rebuild; a running SLURM job is unaffected by clobbering its on-disk binary — the linker rename gives a new inode).
- **Device-halo/sync changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); pi is insufficient (no ice / no bulk).
- **Commit/push only when the user asks.** KPP is the default mix_scheme ([[feedback-kpp-default]]).
- **Build dirs:** `build-cuda`, `build-serial`, `build-synccheck` all carry M5.16; `build-omp` STALE (rebuild for an OpenMP leg). On branch `m517-mpi-comms`.

---

## 5. POINTERS

- **Memory:** [[project-m516-bulk-port]] (M5.16 full state + the node-for-node), [[project-m515-gm-residency]], [[feedback-perf-same-day-baseline]], [[feedback-gpu-fidelity-gate]], [[feedback-ice-mask-averaging]] (the uice/script-artifact lesson), [[reference-cuda-aware-mpi]].
- **Docs:** `docs/GPU_FIDELITY.md` §M5.16 (+ the uice correction note), `docs/SCALING_NG5.md` (the nsys MPI decomposition + the per-rank pattern), `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–**L61**).
- **Tags:** `m5.9-pin` (whole-model GPU + gate + M3.2), `m5.16-bulk-port` (residency exhausted + bulk port, master HEAD). `git log m5.9-pin..m5.16-bulk-port` = the campaign.
- **Tooling:** `jobs/job_ng5_prof` (FESOM_STEP_PROFILE + deep_copy + force-split), `jobs/job_nsys_ng5` (MPI-traced nsys → mpi_event_sum), `jobs/job_bulk_verify_core2` + `jobs/job_bulk_synccheck_core2` (the CORE2 forced-only verify pattern), `src/fesom_field.hpp` `FESOM_SYNC_LOG` (per-field DtoH). `src/fesom_halo_device.{hpp,cpp}` is where the start/finish split + aggregation land.

## 6. Bottom line for the next session
**Measure the `MPI_Waitall` split FIRST** (barrier-isolation experiment + per-rank variance + message-count histogram on the M5.16 binary). If comm-latency-dominant → interior/boundary overlap; if many-small-messages → aggregate; **if load-imbalance-dominant → it's a floor, document it and pivot to Lever C.** The 47 % is a *ceiling*, not a *promise*. Don't claim a win until it's same-day measured + Serial/np2 bit-identical + gate PASS + 1-yr climate-validated. The wall has moved every milestone — keep measuring.
