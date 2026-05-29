# M5.13 — NG5 device-residency / PCIe-reduction campaign (executable task plan)

**Branch:** `m513-pcie` (off `m512-fusion @ 2956ff3`). **Playbook (prose rationale + full per-site
detail):** `docs/plans/20260530-pcie-residency-campaign.md` — this file is the *executable* companion.
**Charter/evidence:** `docs/SCALING_NG5.md` § nsys decomposition + `docs/figures/nsys_ng5_breakdown.png`.
**Lessons:** `docs/KOKKOS_PORTING_LESSONS.md` L36, L47, L48, L50, L51, L56.

> Line numbers below were read on `m513-pcie @ 2956ff3`. **Always re-read the actual code before
> editing** — every milestone edits `fesom_step.cpp`, so later milestones' lines shift. The site
> tables are guidance; the *recipe* (§ Flip recipe) and the *gate* (§ Validation ladder) are invariant.

---

## STATE CHECKLIST (resumability — read this to know "where are we")

- [x] **M5.13a** — `cfl_z` (NOD3D) — DONE ✅ rungs 1-3 (Serial `ale`=0 / pi np1+np2 bit-id / SYNCCHECK clean); gate PASS worst 5.7e-3 (no staleness); CORE2 deep_copy baseline **207.7/step, 1067.6 MB/step**
- [x] **M5.13b** — EOS `hpressure` + `sw_alpha`/`sw_beta` (NOD3D) — DONE ✅ verify eos/pgf/gm/kpp=0; pi np1+np2 bit-id; SYNCCHECK clean; gate PASS worst 2.3e-3; **deep_copy 207.7→199.7/step (−8), 1067.6→1020.4 MB/step**
- [ ] **M5.13c** — GM quartet `fer_gamma`/`slope_tapered`/`Ki`/`fer_uv` — highest aggregate + complexity → **NG5 ckpt 1**
- [ ] **M5.13d** — `uv_rhsAB` (ELEM3D) — cross-step AB2 history
- [ ] **M5.13e** — ALE `w`/`w_e` + bolus round-trips — tangled (`w` snap-out)
- [ ] **M5.13f** — ALE commit `hnode`/`helem` — highest cross-step fan-out → **NG5 ckpt 2**
- [ ] **M5.13g1** — tracer `T` values + `uv`-after-update_vel — highest blast radius (all snap-out)
- [ ] **M5.13g2** — *(CONDITIONAL)* salinity floor → device clamp, unpin `S` → **NG5 ckpt 3 = acceptance**
- [ ] **Acceptance** — NG5 final re-trace + scaling sweep interpreted, ratio recorded
- [ ] **Docs** — lessons/scaling/fidelity/handoff updated, plan → `completed/`

Each milestone is committed separately. A fresh session resumes by finding the first unchecked box.

---

## Overview

**The mandate.** An `nsys` CUDA trace on the production mesh NG5 (7.4 M nodes, 70 levels, dist_16,
job `25227869`, 16.94 s/step, snapshots off) proved the GPU step is **PCIe-data-movement-bound**:
GPU kernels = **7 %** of the step; **~75 %** is full-field host↔device `cudaMemcpy` from host-staged
halos (H2D 48 % + D2H 27 %; `cudaMemcpy` = 90.6 % of all CUDA API time, ~4575 transfers/step). The
GPU is idle ~93 % of every step.

**The lever = DEVICE RESIDENCY.** Flip the remaining host-staged nod3D/elem3D halos to the on-device
GPU-aware-MPI path `fesom_halo_field` (`src/fesom_halo_device.hpp:94`) and eliminate host-op syncs
that have a viable device twin. Each un-flipped 3-D halo is a full-field **~259 MB** sync at NG5
(462 k nodes/rank) — cheap at CORE2 (16 k/rank, why M5.1 left them), dominant at NG5.

**NOT in scope:** Lever C (rank-1 → `View<double**>` coalescing) and launch-fusion — both touch the
**7 %** compute, not the **75 %** PCIe. Shelved. Also out of scope: changing whether/how salinity is
conditioned (the floor is a numerical NaN-guard; any g2 device port is a *byte-identical* relocation).

**Benefit / acceptance.** Reduce NG5 per-step PCIe and re-measure the node-for-node GPU/CPU ratio
(currently **3.8×**, L56 calls it an upper bound on the *current* device-residency, not a hard floor).
Closing half the PCIe could move NG5 toward ~2×.

---

## Context (from discovery)

- **Infra you flip *to*:** `fesom_halo_field(Field&, fesom_halo_kind, n_levels, n_components,
  fesom_partit*, base_off=0)` (`src/fesom_halo_device.hpp:94`). On CUDA → device pack → GPU-aware MPI
  on device ptrs → device unpack (no full-field PCIe). On Serial/OpenMP (or CUDA + `FESOM_HOST_HALO=1`)
  → the exact legacy host bracket. `kind ∈ FESOM_HALO_{NOD2D,NOD3D,ELEM2D,ELEM3D,ELEM2D_FULL}`.
- **The rails** (`src/fesom_field.hpp`): `modify_*`/`sync_*` over a `DualView` + a 3-state `auth_` tag.
  On Serial/OpenMP host==device → all syncs are **no-ops** → **Serial is the strict bit-identity oracle,
  unchanged by any flip (Approach B)**. The risk is **CUDA-only staleness**.
- **Driver:** `src/fesom_step.cpp` (substep driver; `FESOM_KK_VERIFY` dispatch at :137-159; per-substep
  `PMARK` profiler). **Pre-I/O sync gate:** `src/fesom_main.cpp:1283-1296` (snapshot-step-gated). **Free
  before finalize:** `src/fesom_main.cpp:1349-1351`.
- **Worked precedents already in tree:** `uv_rhs` (M5.4a, `:455/:479/:504`), `pgf_x/pgf_y` (M5.4b,
  `:313-314`), `bvfreq`+device-smoother (M5.5, `:206/:220`), KPP `blmc`/`Kv`/`Av` (M5.5/M5.7).

---

## Development Approach

**This is a data-movement campaign on an existing, validated kernel set — not feature development.**
Every kernel already exists and is verified; we are *relocating halo exchanges* and *removing redundant
PCIe syncs*. Therefore:

- **There are no new unit-test files.** The project's "tests" are the **validation ladder** (§ below):
  per-kernel `FESOM_KK_VERIFY` (Serial `max|Δ|==0`), pi bit-identity (np1+np2), SYNCCHECK, and the
  mandatory CORE2-active-ice GPU fidelity gate. **The ladder IS the test and it is mandatory.**
- **Validation-gate-driven (code first, then validate)** — there is nothing to write test-first; the
  edit is a sync/halo relocation, validated immediately after.
- **CRITICAL: every milestone MUST pass the full § Validation ladder before the next milestone starts.**
  No exceptions (this is the analogue of "all tests pass before next task").
- Small, focused changes; one milestone = one commit. Update the STATE CHECKLIST + this plan as you go
  (`[x]`; `➕` new task; `⚠️` blocker).
- **Serial bit-identity is preserved by construction** (Approach B) — but verify it every time anyway
  (rung 2), because a stray edit can reorder host loads (L51 ULP drift).

---

## The flip recipe (L48 split-rail) — invariant, reused by every milestone

For field `X`:
1. **Replace** the OUT-rail `X.sync_host()` + `fesom_exchange_*(X.h_checked(),…)` with **one**
   `fesom_halo_field(X_fld, FESOM_HALO_<KIND>, nl, nc, p)` at the producer.
2. **REMOVE** the downstream IN re-push(es) `X.modify_host(); X.sync_device();` — `X` is left
   **device-authoritative** (owned + halo current); later **device** kernels read it directly.
3. **Keep** any *genuine host-reader* sync (trace the readers; use the NaN-poison tool if unsure).
4. If `X` is a **snapshot output**, add it to the snapshot-gated pre-I/O `sync_host` block at
   `fesom_main.cpp:1283-1296` (the **I/O-staleness trap**, L48).

**Snap-out set** (`fesom_io.cpp:435-457`): `T, S, eta_n, w, u, v` + `density_m_rho0, bvfreq, pgf_x,
pgf_y, Kv, Av` + ice `a_ice, m_ice, m_snow, uice, vice, h_ice, h_snow`. (Pre-I/O block already covers
`pgf_x/pgf_y/bvfreq/Kv/Av`; `density` keeps its OUT rail.)

**NaN-poison discriminator** (`jobs/job_poison_dev`, L50): before deleting a `sync_host` you *think* is
redundant — KEEP the sync, NaN the host copy after it (no `modify_host`). Only a genuine HOST read sees
the NaN. A leave-one-out toggle is confounded (it also drops a fence → reshuffles atomics → false "needed").

---

## Validation ladder (run for EVERY milestone — all rungs must pass)

Build CUDA with **`source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; env.sh's 4.1.2
SEGFAULTs on device ptrs). Output → **`/work/ab0995/a270088/port2/…`**, never `$HOME`.

1. **Serial per-kernel verify** (`build-serial`, key per milestone):
   `FESOM_KK_VERIFY=<key> ./build-serial/fesom_port <pi-mesh> /tmp/out 100 20 10 2>&1 | tee run.log`
   PASS = `grep FESOM_KK_VERIFY= run.log | grep -v 0.000e+00` is **empty** (`max|Δ|==0`).
   (`pp` substring-collides with `kpp` — handled in the dispatch; not a key we use here.)
2. **Serial pi bit-identity, np1 AND np2** — `scripts/diff_snap.py` (takes **DIRECTORIES**, zero-tol):
   - np1 vs `docs/reference/c_baseline_snapshots/pi`.
   - np2: `export OMPI_MCA_btl_vader_single_copy_mechanism=none` first; vs
     `/scratch/a/a270088/pi_np2_ref_m13_nocma` (NOT the CMA-tainted `…_m12`).
3. **SYNCCHECK clean** — `build-synccheck` (Serial + `-DFESOM_KK_SYNCCHECK`), pi np1/np2, clean exit
   (no `h_checked` abort). *Caveat: Serial host==device, so this cannot catch a CUDA-only stale read.*
4. **MANDATORY CORE2-active-ice fidelity gate** — `scripts/gpu_fidelity_gate.sh --fresh-oracle`
   (CUDA-vs-Serial, ice active; **pi is INSUFFICIENT** — no ice → a stale-host bug stays at 1e-17, how
   the M5.9 bug hid ~8 commits). **`--fresh-oracle` every milestone** (each edits `fesom_step.cpp` →
   ULP drift at bathymetry edges, L51). **Run its CUDA leg with `FESOM_STEP_PROFILE=1`** so the one GPU
   job yields BOTH the field-fidelity verdict AND the `deep_copy` transfer count/MB.
   - **PER-FLIP SUCCESS SIGNAL = the `deep_copy` transfer-count DROP** vs the previous milestone
     (mesh-independent, deterministic). CORE2 wall-time may be flat; **the count is the proof**.
   - **Diff ALL output fields, never a subset** (L48 — the I/O-staleness trap is invisible otherwise).
   - CUDA is non-deterministic (atomic-scatter reassociation): the fidelity gate's ceilings already
     encode the host-vs-host noise floor; never expect device==host byte-identity.

---

## NG5 checkpoints (the real acceptance — GPU time is paid-for; the constraint is QUEUE residency)

~3 checkpoints (not every milestone, not too rare). Each: **bundle `jobs/job_nsys_ng5` +
`jobs/submit_ng5_scaling.sh` into ONE background submission**, fire it, and **keep developing the next
milestone while it sits in queue** (never block on the queue). NG5 runs use `snap_every=-1` (rank-0
gather OOMs ~66 GB). Output → `/work/ab0995/a270088/port2/…`.

- **Checkpoint 1 — after M5.13c:** `nsys` re-trace → confirm PCIe % is trending down (the GM-quartet
  nod3D win is the one we most want to *see* land).
- **Checkpoint 2 — after M5.13f:** `nsys` re-trace → PCIe % trend.
- **Checkpoint 3 — after M5.13g (ACCEPTANCE):** full `nsys` + full scaling sweep (dist_4/8/16) →
  PCIe % down **and** node-for-node GPU/CPU ratio down from 3.8× (toward ~2×). Record in
  `docs/SCALING_NG5.md`.

---

## MUST STAY HOST (do NOT flip)

- **Salinity floor** (`:931-950`, L39) — host clamp guarding the EOS-NaN→CG-abort; pins `S`'s OUT-rail
  (unless **g2** triggers, which only *relocates* it byte-identically).
- **`uvnode → fesom_bulk_compute`** (`:338`, L50) — the SOLE genuine host reader (JRA55 bulk wind
  stress, surface row, every CORE2 step). Proven by NaN-poison. **Do NOT remove.**
- **`eta_n` inline map** (`:650-659`) — trivial nod2D host loop; re-synced as a substep-4 IN-push.
- **All `fesom_exchange_*` inside `FESOM_KK_VERIFY` C-twin functions** (the host reference oracle).
- **Setup-time exchanges** (`fesom_mesh.cpp`, `fesom_partit.cpp`, `fesom_phc.cpp`).

**Already flipped (do NOT re-target):** `uv_rhs`, `pgf_x/pgf_y`, `uvnode`, FCT internals, `bvfreq`+smoother,
KPP `blmc`/`diffK`/`viscA`, `Kv`/`Av`, EVP `uice`/`vice`, `momentum_adv`/`visc_filt`, CG `pp/rr/X/d_eta`,
Redi internal `tr_xy`/`tr_z`, ice FCT.

---

## Traps to carry into every milestone

- **I/O-staleness (L48):** snap-out field → pre-I/O block + **diff ALL output fields**.
- **Cached `static Kokkos::View`** a flip introduces MUST be file-static + freed before
  `Kokkos::finalize()` (`fesom_main.cpp:1349`) → else SIGABRT at exit.
- **GPU atomic-scatter run-to-run non-determinism (D22):** validate CUDA at the host-vs-host noise
  floor (the gate's ceilings), never byte-identity.
- **RMW-verify is Serial-safe by construction (L38):** the per-kernel `FESOM_KK_VERIFY` runs on
  `build-serial` (host==device, halo done on host) → flipped RMW fields (`uv`, `values`) still verify
  `max|Δ|==0`. The CUDA gate for those = the **end-to-end** fidelity gate / `diff_snap`, never a
  per-kernel CUDA verify.
- **L36:** never force-flip a field with no *device* consumer this step (adds a pure round-trip).
- **L51:** re-bake the gate oracle (`--fresh-oracle`) every milestone.

---

## Implementation Steps

> Per-milestone task = edits (the flip + re-push removals) → the §Validation ladder (the "tests") →
> commit. The 4 ladder rungs are listed as separate checkboxes (the literal commands are in
> §Validation ladder; the verify key is filled in per task). **All rungs must pass before the next
> milestone.**
>
> ⚠️ **The enumerated re-push sites below are a FLOOR (verified @2956ff3), NOT a complete list.** The
> per-task "grep the field's `_fld` for `.modify_host()`/`.sync_device()`" inventory is AUTHORITATIVE —
> always grep to confirm none is missed (a plan-review found e/f/g1 undercounted; see each task's ⚠️).

### Task M5.13a — flip `cfl_z` to device-halo (lowest risk, prove the loop end-to-end)

**Files:** Modify `src/fesom_step.cpp`. **Verify key:** `ale`. **Snap-out:** no. **Kind:** NOD3D, nl, 1.

- [ ] re-read substep 12c (`~:705-718`): producer `fesom_ale_compute_cflz_kk` → OUT `cfl_z.sync_host()`
      (`:711`) → verify (`:712`) → `fesom_exchange_nod3D(cfl_z…)` (`:713`); IN re-push at `:717`
      (`cfl_z.modify_host(); sync_device();`) feeding `wvel_split` (12d, reads `cfl_z` on device)
- [ ] replace `:711` + `:713` with **one** `fesom_halo_field(dyn->cfl_z_fld, FESOM_HALO_NOD3D, nl, 1, p)`
      placed **after** the `:712` verify (verify reads owned `cfl_z` — Serial-fine)
- [ ] **REMOVE** the IN re-push at `:717`
- [ ] rung 1: Serial `FESOM_KK_VERIFY=ale` → `max|Δ|==0`
- [ ] rung 2: Serial pi bit-identity np1 + np2
- [ ] rung 3: SYNCCHECK clean (np1/np2)
- [ ] rung 4: `gpu_fidelity_gate.sh --fresh-oracle` (CUDA leg `FESOM_STEP_PROFILE=1`) — PASS + record
      `deep_copy` count (baseline for the campaign) + **diff ALL output fields**
- [ ] commit: `M5.13a: flip cfl_z (NOD3D) to device-halo (fesom_halo_field)`

### Task M5.13b — flip EOS `hpressure` + `sw_alpha`/`sw_beta` (clean, real device readers)

**Files:** Modify `src/fesom_step.cpp`. **Verify key:** `eos` (+ `pgf` for the hpressure consumer).
**Snap-out:** no (none of the three). **Kind:** NOD3D, nl, 1 each.

- [ ] re-read substep 1 OUT block (`~:188-220`): the 7 outputs are `sync_host`'d as a **group**
      (`:188-194`) then halo'd separately (`:203-215`); bvfreq is the M5.5 precedent (`:206`/`:220`)
- [ ] **hpressure:** delete its `sync_host` in the `:188-194` group AND replace its
      `fesom_exchange_nod3D(hpressure…)` (`:204`) with `fesom_halo_field(aux->hpressure_fld,
      FESOM_HALO_NOD3D, nl, 1, p)`; **REMOVE** the PGF IN re-push at `:307`
      (`hpressure.modify_host(); sync_device();` — PGF reads hpressure at element vertices on device)
- [ ] **sw_alpha & sw_beta:** delete their `sync_host` in the `:188-194` group AND replace their
      exchanges (`:214`/`:215`) with two `fesom_halo_field(…, FESOM_HALO_NOD3D, nl, 1, p)`; **REMOVE**
      the GM-1b IN re-pushes (`:246`/`:247`) AND the KPP IN re-pushes (near `:359-360`) — both read
      sw_α/β on device
- [ ] ⚠️ resolve the **grouped-sync_host** subtlety: confirm bvfreq's residual `sync_host` (`:190`)
      precedent — decide if hpressure/sw_α/β leave any residual host sync (use NaN-poison if unsure;
      goal = kill the wasted D2H, keep nothing load-bearing)
- [ ] ⚠️ **ordering invariant:** delete a field's group `sync_host` (`:189/:191/:192`) ONLY together with
      replacing that field's `h_checked()` exchange (`:204/:214/:215`) in the SAME edit — they are ~15 lines
      apart; a partial edit leaves a host-staged `fesom_exchange_*(…h_checked()…)` reading stale host on
      CUDA (Serial-invisible → rungs 1-3 pass, only the rung-4 gate catches it)
- [ ] rung 1: Serial `FESOM_KK_VERIFY=eos` AND `=pgf` → `max|Δ|==0`
- [ ] rung 2: pi np1+np2 · rung 3: SYNCCHECK · rung 4: fidelity gate `--fresh-oracle` (profile on) — record `deep_copy` drop, diff ALL fields
- [ ] commit: `M5.13b: flip EOS hpressure + sw_alpha/sw_beta (NOD3D) to device-halo`

### Task M5.13c — flip the GM quartet (highest aggregate; HIGHEST COMPLEXITY — complete re-push inventory)

**Files:** Modify `src/fesom_step.cpp`. **Verify key:** `gm` (consumers also `ale`/`tradv`/`trdiff`).
**Snap-out:** no. All edits are inside `if (gm)` blocks. **GM runs every step (L34).**

- [ ] re-read the GM-1b chain (`~:238-297`) and EVERY consumer re-push site listed below
- [ ] **fer_gamma** (NOD2D, nl, 2): replace OUT `:285` + halo `:287` with `fesom_halo_field(gm->fer_gamma_fld,
      FESOM_HALO_NOD2D, nl, 2, p)`; **REMOVE** IN re-push `:290` (`fer_gamma2vel` reads at halo vertices on device)
- [ ] **slope_tapered** (NOD2D, nl-1, 3): replace OUT `:266` + halo `:270` with `fesom_halo_field(…, nl-1, 3, p)`;
      **REMOVE** re-pushes `:781` (FCT/Redi shared IN) and `:900` (trdiff IN)
- [ ] **Ki** (NOD2D, nl, 1): replace OUT `:277` + halo `:281` with `fesom_halo_field(…, nl, 1, p)`;
      **REMOVE** re-pushes `:782` and `:901`
- [ ] **fer_uv** (ELEM2D_FULL, nl, 2): replace OUT `:294` + halo `:296` with `fesom_halo_field(dyn->fer_uv_fld,
      FESOM_HALO_ELEM2D_FULL, nl, 2, p)`; **REMOVE** re-pushes `:694` (ALE vert_vel IN) and `:743` (bolus 13a IN)
- [ ] ⚠️ when deleting re-pushes inside **shared** IN-rail blocks (FCT `:775-783`, trdiff `:886-902`,
      ALE `:692-694`, bolus `:740-744`) delete ONLY the slope_tapered/Ki/fer_uv lines; **leave the other
      fields' syncs** (uv/w_e/hnode/hnode_new/helem/values/forcing)
- [ ] re-grep `slope_tapered_fld`, `Ki_fld`, `fer_uv_fld`, `fer_gamma_fld` for `.modify_host()` /
      `.sync_device()` to confirm **no re-push was missed** (the complete inventory)
- [ ] rung 1: Serial `FESOM_KK_VERIFY=gm` → `max|Δ|==0` (also spot-check `ale`/`tradv`/`trdiff` pass)
- [ ] rung 2: pi np1+np2 · rung 3: SYNCCHECK · rung 4: fidelity gate `--fresh-oracle` (profile on) — record `deep_copy` drop (should be the biggest yet), diff ALL fields
- [ ] commit: `M5.13c: flip GM quartet (fer_gamma/slope_tapered/Ki/fer_uv) to device-halo`
- [ ] **NG5 CHECKPOINT 1:** submit bundled `job_nsys_ng5` + `submit_ng5_scaling.sh` to background; note PCIe-% trend when it returns (keep going — do not block)

### Task M5.13d — flip `uv_rhsAB` (cross-step AB2 history)

**Files:** Modify `src/fesom_step.cpp`. **Verify key:** `vrhs`. **Snap-out:** no. **Kind:** ELEM3D, nl, 2.

- [ ] re-read substep 4 (`~:437-456`): OUT `uv_rhsAB.sync_host()` (`:449`) + halo `:456`; the next-step
      substep-4 IN re-push at `:443` (`uv_rhsAB.modify_host(); sync_device();`)
- [ ] replace `:449` + `:456` with `fesom_halo_field(dyn->uv_rhsAB_fld, FESOM_HALO_ELEM3D, nl, 2, p)`
- [ ] **REMOVE** the IN re-push `:443` (`compute_vel_rhs` part i reads `uv_rhsAB` on device next step)
- [ ] confirm the `vrhs` verify's L26 capture-before (`:440`) still reads host `uv_rhsAB` (Serial-safe — host-current via the host bracket)
- [ ] rung 1: Serial `FESOM_KK_VERIFY=vrhs` → `max|Δ|==0`
- [ ] rung 2: pi np1+np2 · rung 3: SYNCCHECK · rung 4: fidelity gate `--fresh-oracle` (profile on) — record `deep_copy` drop, diff ALL fields
- [ ] commit: `M5.13d: flip uv_rhsAB (ELEM3D) to device-halo`

### Task M5.13e — flip ALE `w`/`w_e` + collapse the bolus round-trips (TANGLED — careful trace)

**Files:** Modify `src/fesom_step.cpp`, `src/fesom_main.cpp` (`w` is snap-out). **Verify key:** `ale`
(+ `tradv` for the FCT consumer). **Kind:** NOD3D, nl, 1 (both `w`, `w_e`).

- [ ] re-read substep 12b (`~:689-703`, `w`), 12c IN re-push (`:708`), 12d (`w_e` `~:715-724`), and the
      bolus blocks 13a (`:739-748`) + 13c (`:961-965`)
- [ ] **w:** replace OUT `:696` + halo `:699` with `fesom_halo_field(dyn->w_fld, FESOM_HALO_NOD3D, nl, 1, p)`;
      **REMOVE** the cflz IN re-push `:708`. ⚠️ `w` is **SNAP-OUT** → add `aux`/`dyn` `w_fld.sync_host()`
      to the pre-I/O block (`fesom_main.cpp:1283-1296`)
- [ ] **w_e:** replace OUT `:720` + halo `:723` with `fesom_halo_field(dyn->w_e_fld, FESOM_HALO_NOD3D, nl, 1, p)`;
      **REMOVE** the FCT IN re-push `:776` (FCT reads `w_e` on device)
- [ ] ⚠️ **also REMOVE the substep-4 `w_e` IN re-push `:445`** (`dyn->w_e_fld.modify_host(); sync_device();`
      in the compute_vel_rhs IN rail — reads prior-step `w_e`). If `w_e` is left device-authoritative at step
      end (bolus `w_e` sync collapsed below) but `:445` stays, next step pushes STALE host `w_e` onto the
      correct device value — a CUDA-only clobber (Serial-invisible, only rung-4 catches it). Removing `:445`
      and collapsing the bolus `w_e` sync are COUPLED — do both or neither
- [ ] grep `w_e_fld` for ALL `.modify_host()`/`.sync_device()` sites and confirm the complete set is handled
      (`:445` substep-4 IN, `:742` bolus-13a IN, `:776` FCT IN; OUTs `:720`/`:747`/`:964`)
- [ ] trace the bolus `uv`/`w`/`w_e` `modify_host`/`sync_host` pulls (`:740-748` 13a, `:961-965` 13c):
      now that `w`/`w_e` are device-resident, eliminate the redundant ones; **KEEP** any sync feeding a
      genuine next-step HOST reader (use NaN-poison to decide). Note `uv` itself is flipped in **g1** —
      coordinate (the 13a/13c `uv` syncs may need to survive until g1)
- [ ] rung 1: Serial `FESOM_KK_VERIFY=ale` (+ `tradv`) → `max|Δ|==0`
- [ ] rung 2: pi np1+np2 · rung 3: SYNCCHECK · rung 4: fidelity gate `--fresh-oracle` (profile on) —
      record `deep_copy` drop, **diff ALL fields (incl. `w` — the snap-out check)**
- [ ] commit: `M5.13e: flip ALE w/w_e to device-halo + collapse bolus round-trips`

### Task M5.13f — flip ALE commit `hnode`/`helem` (HIGHEST cross-step fan-out)

**Files:** Modify `src/fesom_step.cpp`. **Verify key:** `ale`. **Snap-out:** no. **Kind:** `hnode`
NOD3D nl 1, `helem` ELEM3D nl 1.

- [ ] re-read substep 14 (`~:973-979`): OUT `hnode.sync_host()` (`:975`), `helem.sync_host()` (`:976`),
      halos `:978` (hnode), `:979` (helem)
- [ ] replace `:975`+`:978` with `fesom_halo_field(mesh->hnode_fld, FESOM_HALO_NOD3D, nl, 1, p)` and
      `:976`+`:979` with `fesom_halo_field(mesh->helem_fld, FESOM_HALO_ELEM3D, nl, 1, p)`
- [ ] **REMOVE** the next-step IN re-pushes that read `hnode`/`helem` on device (**~10 sites, NOT ~6**):
      `hnode` at EOS `:178` (combined T/S/hnode block — remove ONLY the hnode line), KPP `:365`,
      compute_vel_rhs `:447`, ALE-12a thickness `:683`, FCT `:777`; `helem` at GM-1b `:255`, ivisc `:499`,
      ssh `:543`, vert_vel `:693`, FCT `:779`. (Grep both `_fld` to confirm the full set.)
- [ ] ⚠️ do **NOT** touch `hnode_new` re-pushes (separate field — the ALE working thickness)
- [ ] **CRITICAL trace:** grep `hnode_fld`/`helem_fld` for `.modify_host()`/`.sync_device()` AND for any
      **host** reader (raw alias `mesh->hnode`/`mesh->helem`) in the next step — confirm every consumer is
      a device kernel before removing its re-push (NaN-poison any doubtful one)
- [ ] rung 1: Serial `FESOM_KK_VERIFY=ale` → `max|Δ|==0`
- [ ] rung 2: pi np1+np2 · rung 3: SYNCCHECK · rung 4: fidelity gate `--fresh-oracle` (profile on) — record `deep_copy` drop, diff ALL fields
- [ ] commit: `M5.13f: flip ALE commit hnode/helem to device-halo (remove ~6 next-step re-pushes)`
- [ ] **NG5 CHECKPOINT 2:** submit bundled `job_nsys_ng5` + `submit_ng5_scaling.sh` to background; note PCIe-% trend (keep going)

### Task M5.13g1 — flip tracer `T` values + `uv`-after-update_vel (HIGHEST blast radius — all snap-out)

**Files:** Modify `src/fesom_step.cpp`, `src/fesom_main.cpp` (T, u, v snap-out). **Verify keys:**
`tradv`/`trdiff` (T), `ssh`/`vrhs` (uv). **Kind:** `T` values NOD3D nl 1; `uv` ELEM3D nl 2.

- [ ] re-read the FCT/Redi T section (`~:785-816`), trdiff (`~:877-909`), and update_vel `uv`
      (`~:559-569`)
- [ ] **T values (within-step path is tangled — gm-conditional):** the flow is FCT-OUT `:799`
      (`vT.sync_host()`, always) → Redi-IN re-push `:806-807` (`vT/voT.modify_host(); sync_device()`,
      **gm-only**) → Redi-OUT `:813` (gm-only) → halo `:816` (**outside** `if(gm)`) → trdiff-IN re-push
      `:895-896` → trdiff-OUT `:904` → halo `:908`. To make T device-resident: **REMOVE** the FCT→Redi
      round-trip (`:799` + `:806-807` — Redi reads T device-current from the FCT), flip the Redi-OUT/halo
      (`:813`/`:816`) and the trdiff-OUT/halo (`:904`/`:908`) to `fesom_halo_field(…NOD3D, nl, 1, p)`, and
      **REMOVE** the trdiff-IN re-push `:895-896`. ⚠️ On the gm-OFF path Redi is skipped (`:806-:816` differ)
      — handle both. T flips clean (no host pin)
- [ ] ⚠️ **Leave `S` to g2** — `S` is pinned by the floor's OUT sync (`:905`); here only shed S's
      *within-step* re-pushes if safe, keep the floor pin
- [ ] **uv-after-update_vel:** replace OUT `:563` + halo `:564` with `fesom_halo_field(dyn->uv_fld,
      FESOM_HALO_ELEM3D, nl, 2, p)`; **REMOVE** the compute_hbar IN re-push `:569`
- [ ] ⚠️ `uv` has **HUGE fan-out** — the complete re-push inventory is compute_vel_nodes `:332`,
      compute_vel_rhs `:442`, visc `:473`, ivisc `:496`, ssh `:539`, compute_hbar `:569`, vert_vel `:692`,
      bolus-13a `:740`, FCT `:775` (grep `uv_fld` for `.modify_host()` to confirm). **The clean removal in
      g1 is `:569`** (same-step compute_hbar reads the just-flipped `uv` on device). The NEXT-step re-pushes
      (`:332`/`:442`/…) are KEPT unless you also drop the bolus-13c `uv` `sync_host` (which currently keeps
      host current) — that coordination is optional/deferrable, not required for g1. **KEEP** the bolus
      13a/13c `uv` sync feeding the host `tradv` C-twin (L38) + any genuine host reader; on Serial all these
      are no-ops (RMW-safe)
- [ ] ⚠️ `T`, `u`, `v` are **SNAP-OUT** → add their `sync_host` to the pre-I/O block (`fesom_main.cpp:1283`)
- [ ] rung 1: Serial `FESOM_KK_VERIFY=tradv` + `trdiff` + `ssh` + `vrhs` → `max|Δ|==0`
- [ ] rung 2: pi np1+np2 · rung 3: SYNCCHECK · rung 4: fidelity gate `--fresh-oracle` (profile on) —
      record `deep_copy` drop, **diff ALL fields (T/u/v snap-out check)**
- [ ] commit: `M5.13g1: flip tracer T values + uv-after-update_vel to device-halo`

### Task M5.13g2 — *(CONDITIONAL)* port the salinity floor to a device clamp → unpin `S`

**Run ONLY IF** NG5 checkpoint 2/3 shows `S`'s forced `sync_host` is a material remaining sink **AND**
a clean trace confirms every next-step host reader of `S` is device/surface-covered. Else skip and note why.

**Files:** Modify `src/fesom_step.cpp` (+ a `_kk` clamp, likely `src/fesom_tracer_diff.cpp` or inline).
**Verify key:** `trdiff` + full `diff_snap`. **Scope guard:** BYTE-IDENTICAL relocation of the existing
0.5-PSU clamp — same threshold, same `[0,myDim+eDim)` extent, same outcome. **NOT** a change to whether/how
salinity is conditioned (that is a separate future model-fidelity track — note it, don't do it here).

- [ ] trace ALL next-step host readers of `S.values`: `sss_runoff` relaxation, `ocean2ice` SSS, I/O —
      confirm each is a device kernel or surface-only / pre-I/O-covered (NaN-poison the floor's OUT sync)
- [ ] port `:931-950` to a race-free per-node device clamp (the ice `cut_off` kernel shape — each node
      its own column, no scatter → bit-identical Serial AND OpenMP); `mod_dev(S.values)`
- [ ] flip `S` values halos to `fesom_halo_field`; **REMOVE** the floor's forced OUT `sync_host` (`:905`,
      the trdiff S sync the host floor pins) and the now-redundant next-step EOS `S` IN re-push (`:177`
      group); `S` is snap-out → ensure pre-I/O covers it
- [ ] rung 1: Serial `FESOM_KK_VERIFY=trdiff` → `max|Δ|==0` (the clamp relocation must be bit-identical)
- [ ] rung 2: pi np1+np2 · rung 3: SYNCCHECK · rung 4: fidelity gate `--fresh-oracle` (profile on) —
      record `deep_copy` drop, diff ALL fields
- [ ] commit: `M5.13g2: port salinity floor to device clamp; S fully device-resident`

### Task — Acceptance (NG5 final re-trace + scaling sweep)

- [ ] **NG5 CHECKPOINT 3:** full `jobs/job_nsys_ng5` re-trace + full `jobs/submit_ng5_scaling.sh`
      (dist_4/8/16), output → `/work/ab0995/a270088/port2/…`
- [ ] confirm PCIe `cudaMemcpy` % is **down** from the 75 % baseline (the primary success criterion)
- [ ] re-compute the node-for-node GPU/CPU ratio; compare to the 3.8× baseline (target: moved toward ~2×)
- [ ] sanity: NG5 still stable over the timed steps (T/S in range, no NaN)
- [ ] record the new nsys decomposition + ratio in `docs/SCALING_NG5.md` (+ regenerate the figure)

### Task — Documentation + close-out

- [ ] add lesson **Lnn** to `docs/KOKKOS_PORTING_LESSONS.md` (the campaign's findings: per-flip
      `deep_copy`-count proxy, the queue-overlap NG5 cadence, the salinity g1/g2 split, the new ratio)
- [ ] update `docs/GPU_FIDELITY.md` (§M5.13) and the SYNC_MAP rows for every flipped field
- [ ] update the handoff (`docs/KOKKOS_HANDOFF.md` / the MEMORY pointer) to "M5.13 complete"
- [ ] move this plan to `docs/plans/completed/`

---

## Technical Details

- **`fesom_halo_field` signature:** `(fesom::Field &f, fesom_halo_kind kind, int n_levels,
  int n_components, fesom_partit *p, std::size_t base_off = 0)`. `stride = n_levels*n_components`.
  `base_off` = slab-offset for one channel of a multi-channel field (not needed here — all targets are
  single-channel or whole-field).
- **Kinds used:** NOD3D (cfl_z, hpressure, sw_α/β, w, w_e, hnode, T-values), NOD2D (fer_gamma,
  slope_tapered, Ki), ELEM3D (uv_rhsAB, helem, uv), ELEM2D_FULL (fer_uv). `nl` = number of levels;
  `nc` = components (1 except fer_gamma/fer_uv/uv = 2, slope_tapered = 3).
- **Why CORE2 wall-time is flat but the count isn't:** a flipped halo removes a *fixed number* of
  full-field transfers/step (mesh-independent); at CORE2 each is ~tiny (16 k/rank), at NG5 each is
  ~259 MB (462 k/rank). The `deep_copy` count is the faithful per-flip proxy; multiply by ~259 MB for
  the NG5 bandwidth projection.

## Post-Completion
*Items requiring external systems / interpretation — informational.*

**Measurement (SLURM, GPU partition):**
- NG5 checkpoints run on `--gres=gpu:4 --ntasks-per-node=4 --gpu-bind=none`, partition `gpu`; `dist_N`
  ⇒ N ranks ⇒ N GPUs ⇒ ceil(N/4) nodes. Do NOT use `--gpus-per-task`/`--gpu-bind=single`.
- The acceptance number is a re-*measurement*, subject to ~5 % node/contention noise — same-day,
  same-allocation comparison where possible.

**Future tracks (explicitly OUT of M5.13):**
- Lever C (rank-1 → `View<double**>` coalescing) — touches the 7 % compute; separate session + branch.
- Removing the salinity floor via a better EOS/brackish near-zero-S treatment — a model-fidelity
  question, not data movement.
