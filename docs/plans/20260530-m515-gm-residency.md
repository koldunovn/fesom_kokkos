# M5.15 — finish the GM-chain residency (close the residual PCIe + host-halo wait)

**Branch:** `m515-gm-residency` off `m514-residency` (after the M5.14 closeout `e7109e7`).
**Base:** [[project-m514-residual-residency]] (parity crossed, climate PASS), [[project-m513-pcie-campaign]].
**Recipe:** the L48 split-rail flip (proven through M5.13/M5.14). **All tasks are residency/numerics-neutral
→ climate-safe** (no math change; validation = the gate + a campaign-end 1-yr climate, per L58).

---

## The mandate (the measurement that chose this lever — session 2026-05-30)

A clean re-profile of the **M5.14 binary** (NG5 dist_16, `nsys_ng5/ng5.sqlite`, steady-state windowed
decomposition + per-field sync attribution via a `FESOM_SYNC_LOG` instrumented build) overturned the M5.14
"compute-bound" read and pinpointed the residual:

- **GPU utilization is only ~30%** of wall — the step is **NOT compute-bound**. It idles ~70% waiting on
  data movement + halos. → **Lever C (kernel coalescing) is the wrong target** (optimizes the 30% busy slice).
- **Residual PCIe ≈ 34% of wall** (~5.5 GB/step, ~118 full-field deep_copies), DtoH-heavy (4 GB→host).
- **`MPI_Waitall` ≈ 51% of wall** (~5000 tiny exchanges/step, median 2 KB) = halo latency + load-imbalance.
- **CG `Allreduce` = 0.2%** → the "reduce CG/EVP collectives" flavor of Lever B is **DEAD** (matches M5.2).

**Per-field DtoH attribution** (per-step-per-rank, CORE2 dist_8; ranking is mesh-invariant):

| field | MB/step/rank | class |
|---|---|---|
| `gm.neutral_slope` | 18.1 | **GM host-bracket halo → flip** |
| `gm.sigma_xy` | 12.3 | **GM host-bracket halo → flip** |
| `dyn.uvnode` | 12.3 | required (M5.9-pin bulk forcing) — hard, defer |
| `tracers T` / `S` | 6.2 / 6.2 | host reader — investigate |
| `hnode_new` | 6.2 | evolving mesh — investigate |
| `gm.fer_tapfac` | 6.2 | **verify-only → pure waste, guard** |
| `gm.fer_K` | 6.2 | **GM host-bracket halo → flip** |
| `aux.dbsfc` | 6.2 | KPP `bldepth` host reader — investigate |
| `aux.bvfreq` | 6.2 | supposedly flipped (M5.5) — investigate (placebo?) |
| `kpp.ghats` | 6.0 | "stays host" (memory) — investigate |

**The GM chain dominates** (neutral_slope + sigma_xy + fer_K + fer_tapfac ≈ 43 MB/step/rank). And the GM
fields are **host-bracket halos**: `sync_host` (the DtoH) **+** `fesom_halo_exchange` (the `MPI_Waitall`)
**+** re-import (HtoD). **Flipping them to `fesom_halo_field` kills the PCIe AND shrinks the halo-wait — one
fix, both costs.** Ice ruled out (all nod2D). KPP's 13-field block is debug-gated (`fesom_kpp_dump_this_step`,
off in production).

Evidence: `docs/GPU_FIDELITY.md` §M5.15 (to add); the nsys report `nsys_ng5/ng5.sqlite`; the instrumented
run `synclog_core2/run.err`. Profiling toolkit added this session: the `FESOM_SYNC_LOG` rail in
`fesom_field.hpp` (compile-guarded, off by default), `job_nsys_ng5` now extracts `mpi_event_sum`,
`job_synclog_core2`.

---

## The flip recipe (L48 split-rail — stated once, reused by every flip task)

Per field X currently host-bracketed (`X.sync_host()` then `fesom_halo_exchange(X.h_checked(), KIND, nl, nc, p)`):
1. Replace **both** lines with one `fesom_halo_field(X_fld, FESOM_HALO_<KIND>, nl, nc, p)` at the producer
   (the M5.13c pattern — `slope_tapered`/`Ki` were already done this way at `fesom_step.cpp:267`).
2. **Remove** any downstream IN re-push (`X.modify_host(); X.sync_device();`) — X is left device-authoritative;
   later DEVICE kernels read it directly.
3. Keep any genuine HOST-reader sync (and if a reader is host-only, that's a separate "flip the reader to
   device" sub-task). If X is a snapshot output, add it to the pre-I/O `sync_host` block (`fesom_main.cpp`, L48).
4. Use the **NaN-poison discriminator** (L50, `job_poison_dev`) before deleting any sync you suspect is a
   placebo — a leave-one-out is confounded by the lost fence.

---

## Validation ladder (per flip task — cheap→costly, all must pass)

1. **Serial per-kernel verify:** `FESOM_KK_VERIFY=gm ./build-serial/fesom_port <pi> /tmp/out 100 20 10`;
   PASS = `grep FESOM_KK_VERIFY= run.log | grep -v 0.000e+00` empty (max|Δ|==0). (Also `eos`/`trdiff` if T/S touched.)
2. **Serial pi bit-identity np1 AND np2** (`scripts/diff_snap.py`, directories, zero-tol): np1 vs
   `docs/reference/c_baseline_snapshots/pi`; np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none` vs
   `/scratch/a/a270088/pi_np2_ref_m13_nocma`.
3. **SYNCCHECK clean:** `build-synccheck` pi np1/np2, clean exit (catches a dropped sync_host that a host reader needs).
4. **CORE2-active-ice CUDA fidelity gate:** `scripts/gpu_fidelity_gate.sh --fresh-oracle` (build-cuda with
   `source ./env_cuda.sh`). `--fresh-oracle` every milestone (each edits `fesom_step.cpp` → ULP drift, L51).
   Run its CUDA leg with `FESOM_STEP_PROFILE=1` so the same job yields the fidelity verdict AND the deep_copy
   count/MB. **PER-FLIP SUCCESS SIGNAL = the deep_copy transfer-count/GB DROP** (deterministic, mesh-independent).
   Always diff ALL output fields (L48), never a subset.

**NG5 checkpoint** (~2): after T2 (the GM-halo flips) and at acceptance — `job_nsys_ng5` (now MPI-traced) +
`submit_ng5_scaling.sh`. Re-measure GPU util + PCIe% + the node-for-node ratio. Output → `/work/...`, never $HOME.

---

## Resumable status (one line per task — read this, not the chat, for "where are we")

- [x] **T1** — guard GM verify-only syncs (`fer_tapfac`, `fer_scal`) behind `s_verify_gm` (folded into the T2 verify-blocks)
- [x] **T2** — flip GM host-bracket halos to device: `neutral_slope`, `sigma_xy`, `fer_K` (`fer_C` kept host = small nod2D). **Validated: Serial gm-verify max|Δ|==0, pi np1+np2 ALL-FIELDS-BIT-IDENTICAL, SYNCCHECK clean, CORE2 CUDA gate PASS (worst 8.08e-3, no staleness regression).**
- [x] **CP1** — NG5 nsys checkpoint: **clean step 3.80→3.61 s/step (−5%), deep_copy 6.57→4.84 GB/step (−26%), nsys DtoH 4.11→2.35 (−43%), GPU util 29.8→34%, 1b_gm phase ~11-16%→4.63%; node-for-node 0.879→~0.834×.** GPU_FIDELITY §M5.15.
- [x] **T3** — investigated all 6; removed `bvfreq` (`:192`, verify-only + cosmetic-print) + `dbsfc` (`:193` OUT + `:351` IN re-push, verify-only/redundant — host `kpp_bldepth` is in the verify-gated twin). **Kept: `T` (required — bulk SST forcing, T4 class), `S` (already M5.14), `ghats` (deliberate host-keep), `MLD1_ind` (tiny, GM host read).** Validated: eos/kpp/gm verify max|Δ|==0, pi np1+np2 BIT-IDENTICAL, SYNCCHECK clean, CUDA gate PASS (worst 7.47e-3, bvfreq snapshot 3.9e-7). **Result: step 3.61→3.456 s/step (−4.3%), deep_copy 4.84→4.10 GB/step (−742 MB). Cumulative M5.15: 3.80→3.456 (−9%), deep_copy 6.57→4.10 (−38%), node-for-node ~0.80×.**
- [ ] **T4** — `uvnode` surface-only refresh (the hard one; optional/defer)
- [~] **CP2 / ACCEPTANCE** — **1-yr CORE2 CUDA climate IN FLIGHT** (job 25245115, `m32_cuda_m515_1yr`, M5.15 binary; compare vs `m32_cuda_m514_1yr` + Fortran/C with `scripts/m32_climate_compare.py` next session — expected PASS, climate-safe by construction). **L60 written.** Lever B-overlap re-measure = next campaign. T4 (uvnode) deferred (marginal).

---

## Task spine (sites verified on `fesom_step.cpp` @ HEAD; re-verify a line if the file moved)

### T1 — guard the GM verify-only syncs (freebie, lowest risk)
**Files:** `src/fesom_step.cpp`
- `fer_tapfac` (`:264`): comment says "read by init_redi (owned) + the verify". `fesom_init_redi_gm_kk`
  (`:270`) is a DEVICE kernel reading OWNED → the host `sync_host` serves only the gated `s_verify_gm`.
- [ ] Confirm no production host reader of `fer_tapfac`/`fer_scal` (grep `fer_tapfac_fld.h`, `.h_checked`).
- [ ] Wrap `gm->fer_tapfac_fld.sync_host();` (`:264`) and `gm->fer_scal_fld.sync_host();` (`:271`) in
      `if (s_verify_gm) { ... }`. (Pure waste removal — the field stays device-authoritative in production.)
- [ ] Validate: `FESOM_KK_VERIFY=gm` max|Δ|==0 (verify still reads host-current when ON); Serial pi np1+np2
      bit-id; SYNCCHECK; gate A/B + deep_copy drop (expect ~2 nod3D copies/step gone).
- [ ] Commit: `M5.15 T1: guard GM verify-only syncs (fer_tapfac/fer_scal) — pure-waste DtoH removed`.

### T2 — flip the GM host-bracket halos to the device path (the dominant win)
**Files:** `src/fesom_step.cpp`
Each is the L48 recipe (replace `sync_host` + `fesom_halo_exchange` with `fesom_halo_field`; remove re-pushes).
Confirm each field's downstream consumers (`init_redi_gm_kk`, the substep-13 Redi `diff_hor`/`diff_ver`,
`trdiff` K33) read it on the DEVICE before removing the host rail.
- [ ] `sigma_xy` (`:256` sync_host + `:258` `fesom_halo_exchange(..., NOD2D, nl, 2, p)`) → `fesom_halo_field(gm->sigma_xy_fld, FESOM_HALO_NOD2D, nl, 2, p)`.
- [ ] `neutral_slope` (`:262` + `:266` `..., NOD2D, nl1, 3, p`) → `fesom_halo_field(..., FESOM_HALO_NOD2D, nl1, 3, p)`.
- [ ] `fer_K` (`:272` + `:277` `..., NOD2D, nl, 1, p`) → `fesom_halo_field(..., FESOM_HALO_NOD2D, nl, 1, p)`.
- [ ] `fer_C` (`:273` + `:276` `fesom_exchange_nod2D`) → `fesom_halo_field(..., FESOM_HALO_NOD2D, 1, 1, p)` (nod2D, small — flip for consistency / to drop the host read).
- [ ] Remove any now-dead downstream re-pushes of these fields.
- [ ] Validate (full ladder). **This is the biggest flip — expect the largest deep_copy drop AND a Waitall drop**
      (the host `fesom_halo_exchange` calls become device GPU-aware). Watch the gate A/B at the noise floor.
- [ ] Commit per logical group (sigma_xy+neutral_slope together; fer_K+fer_C together) for clean bisection.

### CP1 — NG5 checkpoint
- [ ] `sbatch jobs/job_nsys_ng5` (MPI-traced) + `bash jobs/submit_ng5_scaling.sh`. Re-run the sqlite decomposition
      (GPU util, PCIe%, Waitall%) + the node-for-node ratio. Record in `docs/GPU_FIDELITY.md` §M5.15 + `docs/SCALING_NG5.md`.

### T3 — investigate + flip the nod3D host-readers
**Files:** `src/fesom_step.cpp`, `src/fesom_kpp.cpp`, readers TBD
For each, classify (genuine host reader / placebo / flippable halo) then act:
- [ ] `bvfreq` (`:192` `sync_host`): the M5.9-pin notes call its blanket sync a placebo — its device readers
      (KPP/mo_convect/GM) + the device smoother re-dirty it. **NaN-poison it** (L50): if model byte-identical,
      drop the `:192` sync (snapshot covered by the pre-I/O block). Likely a freebie like the M5.9 pins.
- [ ] `dbsfc` (`:193`): KPP `bldepth` reads it on host → either flip that reader to device, or accept the sync.
- [ ] `ghats` (`fesom_kpp.cpp:1586`): "stays host" — find the host reader; flip it or accept.
- [ ] `T`/`S` (`tracers.data0/1.values`, `fesom_step.cpp:897` for T): determine the per-step host reader
      (salinity floor is now device, M5.14; mean-accum is device). If none in production → drop/guard.
- [ ] `hnode_new` (`:251` HtoD re-push + a DtoH): evolving mesh; check if the GM HtoD re-push (`:251`) is needed
      now that GM reads on device.
- [ ] Validate each independently (full ladder); commit per field.

### T4 — uvnode surface-only refresh (hard; optional)
**Files:** `src/fesom_step.cpp:334`, `src/fesom_forcing*`/`fesom_bulk*`
- `uvnode` (12.3 MB/step) is required by `fesom_bulk_compute` (JRA55 wind stress, the SOLE real host reader,
  M5.9-pin). Only the **surface row** is read. The M5.9-pin surface-only refresh hit 3.6% but its per-step
  Kokkos alloc overhead wasn't worth it.
- [ ] Re-evaluate with a **persistent** pre-allocated surface host buffer (avoid the per-step alloc). If it nets
      > the alloc cost, ship it; else document as irreducible and defer.

### CP2 / ACCEPTANCE
- [ ] NG5 full nsys + scaling sweep: PCIe% down, GPU util up, node-for-node ratio (re-confirm parity / improve).
- [ ] **1-yr CORE2 CUDA climate** via `job_m32_cuda_core2` (NSTEPS=17280) + `scripts/m32_climate_compare.py`
      vs the M5.14 1-yr (`m32_cuda_m514_1yr`) + Fortran/C — statistically identical (L58: never settle fidelity
      with the 20-step gate). The residency flips are climate-safe but the campaign ENDS on the climate gate.
- [ ] **Re-decide Lever B (interior/boundary overlap):** with the host-bracket halos gone, re-measure the
      residual `MPI_Waitall`. If a large overlappable (not load-imbalance) slice remains, scope the overlap
      refactor; if not, the gap is load-imbalance/latency-floor and B is not worth the invasiveness.
- [ ] Update [[project-m514-residual-residency]] / write `project-m515-gm-residency` memory; lesson L60.

---

## What this is NOT (deprioritized by the measurement)
- **NOT Lever C (kernel coalescing / `View<double**>`):** GPU util is ~30% — the step isn't compute-bound; the
  smoother (#1 kernel) is ~8% of wall. Revisit only after residency + overlap raise GPU util.
- **NOT CG/EVP comm-reduction:** `Allreduce` is 0.2% (CG is cheap, M5.2/this session). Dead.
- **Lever B (halo overlap) is deferred to CP2:** finish the cheaper residency first (it removes host-bracket
  halos, which double as PCIe + Waitall), THEN measure whether overlap is still worth a big interior/boundary refactor.

## Post-completion (external/manual)
- Production deployment guidance (Lever D): more work/GPU → nearer parity; don't over-decompose (the L58 balance lesson).
