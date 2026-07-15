# M7 session 8 — findings (written as the work happens)

*2026-07-15, session 8. Branch `m7-speed` from `44acff8`. Plan: `20260716-m7-session8-PROMPT.md`
(user-approved). All pre-registrations below are committed BEFORE their jobs land.*

---

## 1. THE FOUR BACKGROUND MEASUREMENTS — pre-registrations (BEFORE submission)

All three GPU jobs pinned `BIN=` h10 CUDA `13dbddb4` (PROVENANCE verified at fire time).
Job-script fixes made first: `job_ncu_fctgm_ng5` hardcoded the stale `build-cuda` binary and
ignored `BIN=` (rule 0.7 violation) — now takes BIN/KNOBS/TAG/NCU_METRICS + writes SHA.txt;
`job_m7_tier1_cuda_1yr` OUT dir now TAG-able so the packH gate does not clobber the Tier-1
gate's output (26238055).

### 1.1 gap300_h10 census (job_m7_hostprof, NSYS_TRACE=cuda,mpi, NSYS_SAMPLE=none, 300 steps, a100_80, FESOM_SPEED=1)

Pre-registered (from the h9 census 26258712 minus H.8):
- `ice_h_diag→oce_fluxes_mom` (7.3 ms in h9) **GONE** — that was LAZYSNAP's gap.
- Top of the census: **halo self-gaps ≈ 17 ms** (9.0 + 8.2 in h9) + the **SSH/hbar class
  ≈ 13–14 ms** (h9 rows: 2.9 + 0.9 + 2.4 + 6.0 + 2.4 = 14.6 — some sub-ms fraction may drop
  below the 1 ms threshold).
- Total gaps > 1 ms ≈ **39 ms/step** (46.4 − 7.3); traced step ≈ **671 ms** (678.1 − 7.3).
- Protocol note: h8's census used `-t cuda,mpi`; h9's used `-t cuda` (the outlier). This one
  returns to `cuda,mpi` — gap sizes stay comparable (µs-scale tracing overhead), and the MPI
  column becomes real again (needed to size H.9's four host halos and package E).
- **This census SIZES H.9** — its floor is pre-registered in §3 below only after this lands.

### 1.2 B precondition — FCT tracer-invariant traffic fraction (job_ncu_fctgm_ng5, BIN=h10, FESOM_SPEED=1)

Metrics: `gpu__time_duration.sum, dram__bytes_read.sum, dram__bytes_write.sum, lts__t_bytes.sum,
l1tex__t_bytes.sum` per kernel, default regex (covers `tracer_advect_one_fct` + `redi|fer_` —
the redi rows are a free C cross-check on `diff_ver_part_redi_expl`).

Protocol: from source, classify every array each FCT kernel touches as **per-tracer** (tr fields,
LO solution, antidiffusive fluxes, R± limiters, per-tracer scratch) vs **tracer-invariant**
(mesh geometry/topology, edge tables, areas/volumes, layer thicknesses, velocity/w — shared by
T and S). Compute the invariant fraction *f* of measured DRAM traffic. Model check (L92 — columns
must sum): analytic per-kernel bytes must land within ~30 % of the measured DRAM bytes, else the
split is not trusted.

**🔴 THE GO/NO-GO LINE, pre-registered before the job runs.** FCT pool = 181.8 ms/step kernel-busy
at 4N; T+S batching reads invariant bytes once instead of twice ⇒ ideal saving = *f*/2 × 181.8 ms
on a 666.6 ms step (h10). Calibration from L93's successors: kernel-fusion levers realize well
under the ideal; assume ~60 %.
- ***f* ≥ 0.25 ⇒ GO for B** (ideal ≥ 22.7 ms = 3.4 %; realistic ≈ 2 % — above any remaining
  H-class lever).
- ***f* < 0.15 ⇒ NO-GO** (realistic ≤ ~1.2 % — loses to C and E for far more effort + 3–5 GB/rank
  memory cost).
- **0.15 ≤ *f* < 0.25 ⇒ B ranks BEHIND C**; build only if C's re-derived pool disappoints.

### 1.3 packH 1-yr climate gate (job_m7_tier1_cuda_1yr, BIN=h10, KNOBS="FESOM_SPEED=1", TAG=packh_cuda_1yr)

The per-tier climate gate packH never had (Tier-1 did: 26238055 matched the M5.23 bar EXACTLY).
Pre-registered PASS bar (vs Fortran, same refs, L74 frames):
- **sst ≥ 0.99990 · sss ≥ 0.99986 · ssh ≥ 0.99990 · a_ice ≥ 0.99987** (the M5.23/Tier-1 bar
  minus 1e-4 tolerance), bias/RMS same order as Tier-1's, backend-vs-C ≤ C-vs-Fortran (the
  script's own rule), zero nan/blowup lines.
- Expectation: match the bar to the printed digit, as Tier-1 did — every h-lever passed its
  35-step CUDA fidelity gate at the climate-close floor and none accumulates state.
- Note: with SNAP_EVERY=1440 the LAZYSNAP writer-pull path actually fires 12× here — this is
  also the first long-run exercise of the H.8 sync-before-gather.

### 1.4 C precondition — spill pool re-derived from the h10 binary (login node, free)

`cuobjdump --dump-resource-usage` on h10 `fesom_port_cuda`, cross-ranked by kernel-busy ms/step
from the gap300_h9 sqlite (h9→h10 changed no kernels; refresh against gap300_h10 when it lands).
Pre-registered expectation (from the packageH-PROMPT audit): 14 spilling kernels, pool
≈ 166 ms/step; **rank 1 = `diff_ver_part_redi_expl`** (42.3 ms, 5,120 B/thread, 58 reg) — which
is NOT in package C's stale list; `diff_part_hor_redi` (23.7 ms, 80 reg) and `impl_vert_visc`
(13.4 ms, 82 reg, 7,168 B) follow.

---

## 2. HARVEST (filled as jobs land)

### 2.1 ✅ Background job 1 — the spill pool, re-derived (login node, h10 binary `13dbddb4` × gap300_h9 busy)

**Scored against §1.4:** 14 spilling device functions ✓; `diff_ver_part_redi_expl` at rank 1 of
the spiller list ✓ (42.36 ms, 58 reg, 5,120 B — the audit's numbers to the digit). Two upgrades
on the stale expectation:

**THE POOL = 188.8 ms/step = 34.8 % of kernel-busy (h9 trace, steps 99–295):**

| busy ms/step | reg | STACK B/thr | kernel |
|--:|--:|--:|---|
| 42.36 | 58 | 5,120 | `fesom_diff_ver_part_redi_expl_kk` |
| 32.58 | 66 | 6,144 | `diff_ver_part_impl_ale_kk` — **the TDMA kernel, invisible in every previous list** (see trap below) |
| 24.84 | 56 | 2,048 | `fesom_momentum_adv_scalar_kk` |
| 23.72 | 80 | 2,048 | `fesom_diff_part_hor_redi_kk` |
| 22.45 | 40 | 2,048 | FCT island `lambda(int)#5` — the ONLY spiller of FCT's 28 sub-kernels |
| 13.35 | 82 | 7,168 | `fesom_impl_vert_visc_kk` |
| 11.36 | 62 | 5,120 | `fesom_pressure_bv_kk` |
| 9.37 | 43 | 7,168 | `fesom_fer_solve_gamma_kk` |
| 6.29 | 40 | 1,024 | `fesom_init_redi_gm_kk` |
| 2.48 | 72 | 4,096 | `kpp_blmix_kk` |

(Untimed spillers, config-dead here: `tke_column_loop` **37,120 B** — the TKE options mode;
`fesom_pressure_force_zxxxx_shchepetkin_kk` 2,096 B. The old "~166 ms pool" = this table minus
the FCT row — the audit had missed FCT's spiller; the FCT island's other 27 lambdas are clean,
so B and C overlap in exactly ONE 22.45 ms kernel.)

**🔴 Two tool traps found (and fixed) on the way — both L92-class (a wrong column/token returns
a CONFIDENT wrong answer):**
1. In `cuobjdump --dump-resource-usage`, spills live in **STACK** (ptxas stack frame), NOT in
   `LOCAL` (static `.local` arrays — 0 for every kernel in this binary). Keying on LOCAL returns
   an empty pool that *looks* like "nothing spills".
2. Never name a kernel by "the first `fesom_*` token in its demangled name": for static kernels
   the first token is a PARAMETER TYPE or the internal-linkage module hash.
   `diff_ver_part_impl_ale_kk` (32.6 ms!) hid inside a fake "fesom_mesh" row for exactly this
   reason. Both scripts (`m7_kernel_busy.py`, `m7_spill_pool.py`) now parse the enclosing
   function after `ParallelFor<`, and the busy table cross-sums to the census's 543.2 ms/step.

Caveats for the C decision: STACK>0 includes ABI stack, not pure spill traffic — the *recoverable*
time is the local-memory traffic share, not the whole 188.8 ms (the ncu job's DRAM/L2 bytes will
bound it for the FCT+redi rows). Ranking refresh against gap300_h10 when it lands (h9→h10 changed
no kernels — expect identical).

### 2.2 Background job 2 landed (26264207) — ncu bytes captured; classification PENDING

Job completed in 3:05, 90 steady-state launches (skip 650 ≈ step 10–11 of 12), h10 `13dbddb4` ✓,
knob announce ✓. srun rc=143 is the expected post-profile termination (ncu detach), not a failure.
Caveat: rank 0 sat on `l40366` = a100_40 — byte counts are hardware-independent (what we use);
durations are ~25 % pessimistic vs the a100_80 census (cross-checked: FCT `#3` 21.9 ms/launch here
vs 18.9 on a100_80 — consistent).

**Raw per-kernel table harvested** (aggregate in `/work/.../m7/ncu_fct_h10/sol.csv`). Two facts
already visible, noted BEFORE the invariant classification (so they cannot be post-hoc):
1. `diff_ver_part_redi_expl` per launch: 17.8 GB DRAM, **42.7 GB L2** — the spill amplification
   is visible (L2 ≈ 2.4× DRAM). Good for C.
2. 🔴 **The claim "FCT is bandwidth-bound" is only HALF true.** The island's biggest kernels split:
   `#4` 789 GB/s and `#5` 737 GB/s are near the roof, but **`#3` (37.7 ms/step, the single biggest)
   runs at 347 GB/s** with L2 3.4× DRAM, and `#6`/`#7`/`#8` sit at 300–390 GB/s. For those, B's
   read-once saving realizes at well under the byte ratio — the realization factor in §1.2's model
   (60 %) applies only to the DRAM-bound rows. This nuance is recorded before *f* is computed.

**✅ THE MEASUREMENT (classification done from source, every one of the 28 island lambdas mapped
to its named launch — the mapping was verified by grid sizes and busy-time cross-checks):**

- Total FCT island DRAM traffic: **142.5 GB/step** (71.2 GB per tracer pass ×2), consistent with
  182.6 ms at a ~780 GB/s traffic-weighted rate.
- Tracer-invariant traffic (uv, helem, W, hnode/hnode_new, area/areasvol, Z3d/zbar3d, gradients,
  topology): **17.3 GB per pass ⇒ *f* = 0.242** (sensitivity ±10 pp on the two dominant uncertain
  shares, `mfct_h`/`upw1h`: 0.21–0.27). The invariant traffic is CONCENTRATED: `mfct_h` (0.50 ×
  16.1 GB) + `upw1h` (0.65 × 4.5 GB) carry ~11 GB of it; the whole Zalesak limiter chain is
  per-tracer scratch at *f* ≈ 0.05.

**Scored against the §1.2 pre-registered line: *f* = 0.242 < 0.25 ⇒ B ranks BEHIND C** (the
"0.15–0.25" band, at its top edge). The conversion argument sharpens it: the ideal one-pass
saving is 17.3 GB/step ≈ 3.2 % of step at roofline, but only the ~11 GB sitting in the two
DRAM-bound kernels converts near 1:1 — honest projection **≈ 1.5–2.2 %** for a 28-kernel
rewrite + 3–5 GB/rank scratch + a fused-kernel register-pressure risk on the already-spilling
`zal_a34`. B is PARKED unless C disappoints.

### 2.3 ✅ Background job 3 — gap300_h10 census (26264206, h10 `13dbddb4` ✓, pure a100_80 ✓, FESOM_SPEED=1 announce ✓)

**Scored against §1.1: HIT on every line.**
- `ice_h_diag→oce_fluxes_mom` **GONE** ✓ — H.8's 7.3 ms gap does not exist in the h10 census.
- Gaps > 1 ms: **39.8 ms/step** (pre-reg ≈39 ✓) of a 673.6 ms traced step (pre-reg ≈671 ✓;
  the 0.6771 loop s/step vs the 0.6666 anchor is the usual nsys+mpi tracing overhead).
- Halo self-gaps **≈ 18 ms** (9.6 + 7.1 + 1.4 pair rows) ✓ — and with `-t cuda,mpi` restored,
  they are now provably **MPI-wait** (9.4/7.2/1.3 ms MPI-covered): package E's pool, measured.
- **The SSH/hbar class = 14.8 ms/step** (pre-reg 13–14, slightly richer):
  `compute_hbar→timestep` 6.1 (PCIe 3.9; 17.7 MB) + `halo→compute_ssh_rhs` 3.0 (HtoD 10.6 MB,
  the :630-633 pushes) + `ssh_solve→update_vel` 2.5 (the d_eta bounce) + `timestep→ale_thickness`
  2.2 (the eta_n push) + `compute_ssh_rhs→launch` 1.0 (ssh_rhs DtoH). The four host
  `fesom_exchange_nod2D` calls' MPI time is tiny (~0.2 ms — small 2D messages); the cost is the
  PCIe staging (≈8.6 ms summed) + untraced host pack/unpack/eta_n-loop time.
- Future-lever note (not H.9): the ice-thermo bounce class shows ~4.3 ms
  (`ice_thermodynamics` self-gaps + `ice_cut_off→thermo`), and `resolve_vice→jra55_step` 1.0 ms
  is the daily forcing read.

### 2.4 ✅ Background job 4 — the packH 1-yr climate gate (26264208, h10 `13dbddb4` ✓, FESOM_SPEED=1 announce ✓)

**PASS — the M5.23/Tier-1 bar matched TO THE PRINTED DIGIT** (pre-reg §1.3 cleared on all four):

| field | corr vs Fortran | the bar | bias (Tier-1's) |
|---|---|---|---|
| sst | **1.00000** | 1.00000 | +4.3e-05 (+5.7e-05) |
| sss | **0.99996** | 0.99996 | −5.4e-04 (−5.4e-04) |
| ssh | **1.00000** | 1.00000 | +2.1e-05 (+3.8e-05) |
| a_ice | **0.99997** | 0.99997 | +1.3e-04 (+1.3e-04) |

backend-vs-C ≤ C-vs-Fortran on every field ✓; zero nan/blowup; the full year ran (17280 steps,
0.0695 s/step, 17 monthly streams + 12 snapshot gathers — the first long-run exercise of H.8's
sync-before-gather, which is exactly what the snapshot cadence hit 12 times). **The whole packH
stack (h5→h10) is now climate-certified at 1 year.**

⚠️ Wart, recorded: the job exits FAILED (SIGABRT **at teardown**, after `loop timing` printed and
all output was written) — the SAME pre-existing full-blessed teardown quirk as `run rc=134` on
the knob-ON fidelity gates (h10's 26259165 too). Physics unaffected; the gate's substance is the
compare table above. Worth a cleanup lever eventually (suspect: finalize ordering under
IOACC/NOFENCE2); until then, expect FAILED job states on full-blessed runs whose *content*
completed.

---

## 3. H.9 SSHRAILS — the audit (complete), the sizing, and the PRE-REGISTRATION

### 3.1 The audit — every trap from PROMPT §3.3 worked through (verified in source today)

The five fields go device-authoritative: `ssh_rhs`, `d_eta`, `ssh_rhs_old` (dyn), `hbar`,
`hbar_old` (mesh), plus `eta_n` becomes device-written. Site-by-site:

**Dies under the knob (fesom_step.cpp):** the `:630-633` defensive pushes (d_eta/ssh_rhs_old/hbar
— under sshrails these become Z7 clobbers, not redundancies); `:646` ssh_rhs sync_host; `:652`
d_eta sync_host + `:658` re-push; `:670-672` three sync_hosts; `:681-682` hbar/hbar_old re-pushes;
`:818-819` eta_n push; **`:519` substep-4 eta_n push (under sshrails it is a CLOBBER of the
device eta_n — must be gated, exactly as its `:814` comment predicted)**.

**Becomes device (fesom_step.cpp):** the four host `fesom_exchange_nod2D` (`:647` ssh_rhs, `:653`
d_eta, `:673` ssh_rhs_old, `:674` hbar) → `fesom_halo_field(..., FESOM_HALO_NOD2D, 1, 1, p)`
(ICERAILS-proven infrastructure, fesom_halo_device.cpp:204); the `:776-784` host eta_n loop → a
per-node device kernel (same expression order; mask `ulevels_nod2D==1`; range myDim+eDim —
hbar's device halo completes first, hbar_old is written full-extent by compute_hbar_kk from
pre-update hbar, so both are halo-complete by construction, same as legacy).

**The traps, each answered:**
1. **`fesom_ice.cpp:634`** hbar push before `ocean2ice_kk` (which reads hbar on DEVICE): under
   sshrails the host mirror is stale ⇒ the push is the BULKTAIL-IC-clobber signature. **Gate on
   `!sshrails`.** Rule 0.3 (who writes the IC): `fesom_ic.cpp:54-60` memsets all six fields to 0
   at init, and the **startup self-tests (fesom_main.cpp:655-760) then run the HOST twins + CG
   before the loop**, leaving nonzero host values that legacy step-1 pushes carry to the device ⇒
   the lever keeps a **once-only first-step IC push of the class (ICERAILS `:576-605` pattern)**
   so the device trajectory is value-identical to legacy at step 1.
2. **eta_n's consumers**: `resolve_ssh_dev` (fesom_io.cpp:956) reads eta_n on device ✓ fed by the
   new kernel; the host `resolve_ssh` (:730) reads it per step ⇒ **require IOACC**. Snapshot
   gather (:461) reads `h_checked()` ⇒ **eta_n sync_host added inside `fesom_io_write_snapshot`**
   (H.8's unconditional writer-pull — covers both callsites, no-op when host-authoritative).
   Of the six fields ONLY eta_n is gathered. `compute_vel_rhs_kk` reads eta_n on device ✓ from
   the prev step's kernel (cross-step; step 1 covered by the IC push).
3. **hbar/hbar_old host readers**: `fesom_ale.cpp:784/:788/:804/:829` are the ale INIT (runs once
   at startup, before any device kernel — comment `:767`) — SAFE. `fesom_ale.cpp:969-983`
   (wvel_split) reads on DEVICE ✓. The `fesom_ale_dump_*` bisect rails **self-sync** via
   `ale_sync()` (fesom_ale_dump.cpp:117-137) — safe, no abort needed. `fesom_main.cpp:519/:751`
   are init-time IC checks — SAFE. The print block (`fesom_main.cpp:1244`) reads eta_n for
   `eta_max` at print cadence ⇒ **add print-cadence `eta_n_fld.sync_host()`** (uv/w/T/S in that
   block are ALREADY stale device-resident reads today — pre-existing, out of H.9 scope, noted).
4. **zstar**: `update_stiff_mat_ale_kk` reads dhe (device, from the `:683-701` device dhe_fill
   which reads hbar/hbar_old DEVICE views after the halo) ✓; wvel_split device ✓; no per-step
   host hbar reader exists in any ALE mode. **The zstar options leg remains THE gate** (0.4).
5. **d_eta cross-step RMW** (CG warm start): CG reads device d_eta across the step boundary; the
   device halo at the `:653` position keeps owned+halo current on device — the boundary is
   covered; step 1 by the IC push.
6. **Per-field halo requirements** derived: d_eta halo read by update_vel (3 verts incl halo) ✓
   device halo before it; hbar halo read by dhe_fill ✓ device halo before it; ssh_rhs halo read
   by NOTHING per-step (CG reads owned rows only) but the exchange is KEPT (device) to match the
   Fortran/legacy halo values bit-for-bit; ssh_rhs_old halo read next step by compute_ssh_rhs ✓.
7. **Requirements/abort (L80)**: requires **IOACC** (host ssh stream resolver otherwise reads
   stale eta_n per step). Aborts on `FESOM_DIAG_SSHSLV`, `FESOM_DIAG_SPREAD` (both read the class
   per step from host when set), and `FESOM_KK_VERIFY=ssh|vrhs` (their capture-before/host twins
   read stale mirrors: fesom_step.cpp:619-627, fesom_momentum.cpp:97-99). ICERAILS is NOT
   required (ice reads hbar only on device). **Guard-abort test in the ladder** (H.8 pattern).

### 3.2 Sized from the gap300_h10 census (L89: gap ms, not bytes)

| | ms/step |
|---|--:|
| the five census rows (§2.3) | **14.8** |
| of which PCIe staging (deleted outright) | 8.6 |
| host pack/unpack + eta_n loop (untraced, deleted) | ~4 |
| MPI wire time that STAYS (moves to device halos) | ~0.2–0.5 |
| device-halo replacement cost ADDED (4 NOD2D exchanges + eta_n kernel) | ~1.5–2.5 |
| traced step | 673.6 |

### ⇒ **PRE-REGISTERED (before any gate job is submitted): H.9 SSHRAILS = −1.9 % at NG5@4N (35-step A/B, h11 binary)**
### **floor −1.3 % · ceiling −2.4 %** *(the gap has real host compute in it → L93 entanglement is possible, but the lever also ADDS device-halo work the census cannot see — both priced in)*

Post-census expectation (for the eventual gap300_h11 diff): the five §2.3 rows GONE; the halo
self-gap rows grow by the four new NOD2D exchanges' wait time.

### 3.3 Gate ladder (identical to H.8's 9/9, per PROMPT §3.4) — BUILT + SUBMITTED

Binaries frozen FIRST: `m7/bin/h11`, CUDA **`d74d31b4`** / Serial **`c125b424`** (h10 + SSHRAILS,
both build rc=0). The lever: `fesom_sshrails_on()` guard in fesom_step.cpp (requires IOACC;
aborts on DIAG_SSHSLV/SPREAD + VERIFY=ssh|vrhs); once-only step-1 IC push of the six fields;
4 host exchanges → device NOD2D halos (ssh_rhs_old+hbar co-packed via `fesom_halo_field2`);
eta_n → device kernel; 9 pushes/syncs gated off (`:519`, `:630-633`, `:646`, `:652/:658`,
`:670-672`, `:681-682`, `:818-819`); fesom_ice.cpp:634 hbar push gated; eta_n writer-pulls
added to write_snapshot + the print block.

| job | gate | expectation |
|---|---|---|
| 26265018 | knob-OFF byte (Serial, live = h11) | diff_snap rc=0 |
| 26265019 | FORCE_SERIAL byte proof, SSHRAILS+IOACC | rc=0 (device-halo Serial fallback = exact legacy bracket; eta_n kernel = same expression order) |
| 26265020 | FORCE_SERIAL byte proof, full blessed | rc=0 |
| 26265021 | CUDA fidelity, isolated | PASS at the climate-close floor — THE gate (L86), and it reads the lever's output (L83: snapshot gathers eta_n; T/S/uv sit downstream of the device-halo'd d_eta) |
| 26265022 | CUDA fidelity, full blessed | PASS |
| 26265023/24/25 | options ×3 (TKE / mEVP / zstar vs their own oracles) | PASS ×3 — zstar is THE leg (hbar feeds the ALE chain); the standing control must reproduce: Kv max\|Δ\| = 9.537e-02 (L79) |
| 26265026 | **GUARD**: SSHRAILS without IOACC | must **ABORT** with the REQUIRES-IOACC message (job FAILs — that is the pass) |
| 26265027 | the pre-registered 35-step A/B, NG5@4N, a100_80, h11 both legs (`FESOM_SPEED=1;SSHRAILS=0` vs `FESOM_SPEED=1`) | **−1.9 % (floor −1.3, ceiling −2.4)** |

Then: 300-step h11 anchor (a100_80) → ledger.

### 3.4 ✅ GATE LADDER: NINE FOR NINE (~25 min wall, submission → last gate)

| gate | result |
|---|---|
| 26265018 knob-OFF byte | ✅ diff_snap rc=0 |
| 26265019 FORCE_SERIAL, SSHRAILS+IOACC | ✅ rc=0 — **the byte proof: pure re-execution elimination** (device-halo Serial fallback + the eta_n kernel are bit-identical re-executions) |
| 26265020 FORCE_SERIAL, full blessed | ✅ rc=0 |
| 26265021 CUDA fidelity, isolated | ✅ PASS, worst 5.778e-03 — THE gate (L86) |
| 26265022 CUDA fidelity, full blessed | ✅ PASS, worst 3.228e-03 |
| 26265023 options TKE | ✅ PASS, worst 1.474e-01 (the known TKE floor; isolated-outlier NOTE, not staleness) |
| 26265024 options mEVP | ✅ PASS, worst 7.282e-04 |
| 26265025 options zstar | ✅ PASS — **the standing control holds EXACTLY: worst 9.869e-02, identical to the h10 ladder's zstar leg (L79) ⇒ the lever adds nothing** — THE leg for this lever (hbar feeds the ALE chain) |
| 26265026 GUARD (SSHRAILS w/o IOACC) | ✅ **ABORTED CORRECTLY**: `FESOM_SPEED_SSHRAILS REQUIRES IOACC (have ioacc=0) ... Refusing to run`, rc=1 |

(`run rc=134` on the knob-ON CUDA gates is the pre-existing full-blessed teardown quirk — h10's
26259165 shows the same; the isolated legs exit rc=0 both ladders. Announce lines fired on every
leg, L80.)

**Lever committed as `f5c130a` on ladder-green (the H.8 criterion).** The A/B was resubmitted as
**26265290** with `-t 00:25:00` after 26265027 sat un-backfillable on a 1:30 claim (rule 0.7 —
the SECOND time this lesson has paid this campaign). 300-step h11 anchor queued as **26265348**
(single-leg 300-step job_m7_ab_env, the std300 pattern). **Anchor pre-registration, committed
before either job lands: anchor = 0.6666 × (1 + Δ_A/B), tolerance ±0.5 %** (H.8's formula).

---

## 4. THE B vs C vs E DECISION (PROMPT §4) — decided on the measurements, not priors

**Verdict: C first. B parked. E stays live as the 16N-side complement.**

- **B (FCT2)** — the §1.2 pre-registered line said *f* ≥ 0.25 ⇒ GO; **measured *f* = 0.242**
  (§2.2) ⇒ by the pre-registration B ranks behind C. The conversion analysis makes the margin
  wider than the headline: only ~11 of the 17.3 invariant GB sit in DRAM-bound kernels, so the
  honest payoff is ≈1.5–2.2 % for the most invasive rewrite on the table (28 kernels, 3–5 GB/rank
  scratch, fused-kernel register pressure on an already-spilling limiter). PARKED unless C
  disappoints at implementation.
- **C (TDMA/spills)** — the re-derived pool (§2.1) is **188.8 ms/step busy across 10 spilling
  kernels = 28 % of the step**, each fixable INDEPENDENTLY (measure → restructure column scratch →
  gate → A/B, one kernel at a time — the opposite of B's monolith). The ladder verdict (26248860)
  says kernel levers keep ~56 % of their 4N value at real 16N. Order of attack by
  busy × spill-severity: **C.1 `diff_ver_part_redi_expl`** (42.4 ms, 5,120 B/thr, L2 2.4× DRAM
  measured), **C.2 `diff_ver_part_impl_ale`** (32.6 ms, 6,144 B/thr — the TDMA column, the
  M5.24-era frontier), then `momentum_adv_scalar` (24.8), `diff_part_hor_redi` (23.7, 80 reg —
  occupancy as much as stack), FCT `zal_a34` (22.5 — NOTE: this one kernel is B∩C; fixing its
  column buffers under C removes part of B's remaining case). Even a conservative 15–25 %
  recovery on the top four ≈ **2.8–4.6 % of the step** — bigger than any remaining H lever.
- **E (comm)** — the halo self-gaps are now MEASURED as MPI-wait (18 ms/step at 4N, §2.3) and
  grow with rank count; it is the only pool that gets BIGGER toward the 16N Stage-2 point. It
  stays live, but per the ladder it no longer outranks B/C by default at 4N.
- Remaining H-class tail (for completeness): the ice-thermo bounce class ≈ 4.3 ms (§2.3) — a
  possible H.10, sized below C's first two rungs.

Honest denominator note: all 4N percentages above are against the h10/h11 step (~0.66 s); 16N
projections must use 0.2688 (not packa's 0.3420), per the PROMPT §4 caveat.

### 4.1 C.1 pre-audit (`diff_ver_part_redi_expl`) — scoped in source, session 8

- The 5,120 B/thread is EXACTLY five `NL_MAX=128` column buffers in the per-node kernel
  `fesom_gm_redi_ver_node` (fesom_gm.cpp:1809: `txn, tyn, zbar_n, z_n, vd_flux` = 5×128×8).
  That kernel is 36.8 ms of the group's 42.4 (2 × 18.4 ms/launch measured); the trxy+zero
  sub-kernels are clean.
- **The rewrite is a single bottom-up column sweep with O(1) carried state — no arrays:** the
  zbar_n/z_n recurrence already runs bottom-up (:1835); `vd_flux[nz]` needs only
  zbar_n[nz]/z_n[nz∓1] (computable on the fly from hnode) and txn/tyn at [nz−1, nz] (the
  surrounding-element gather is order-free → fuse it per level, carry a rolling pair); the apply
  loop needs vd_flux at [nz, nz+1] → rolling pair, one-level lag. Every value keeps its exact
  expression and evaluation order ⇒ **Serial bit-identity is achievable** (FORCE_SERIAL
  byte-proof, unlike most kernel levers).
- (Alternative considered and NOT needed: reading the existing `Z_3d_n`/`zbar_3d_n` device fields
  instead of rebuilding depths — value-equivalence vs the local hnode recurrence would need its
  own proof; the O(1)-sweep avoids the question entirely.)
- Sizing prior (NOT a pre-registration yet — ncu the kernel first): spill slots are 461k threads
  × 5,120 B = 2.4 GB backing; the kernel measures 17.8 GB DRAM + 42.7 GB L2 per launch at
  ~950 GB/s — near-roof partly BECAUSE of spill traffic. If spills are 30–50 % of its DRAM bytes,
  the sweep recovers ~30–45 % of 36.8 ms ≈ **11–17 ms/step ≈ 1.7–2.5 % at 4N**, kernel-class
  (holds ~56 % at 16N). Pre-register properly from a targeted ncu
  (`smsp__inst_executed_op_local_ld/st` or `l1tex__t_bytes_pipe_lsu_mem_local`) before building.
