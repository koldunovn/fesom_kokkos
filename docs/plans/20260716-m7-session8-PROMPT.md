# M7 next session (session 8) — PROMPT

*Written 2026-07-15 morning, close of session 7 (autonomous overnight). Branch `m7-speed`,
tag `m7.4-stage2met` @ `2956be0` — **both PUSHED to origin** (user-approved). Working tree clean.
The plan below was reviewed and approved by the user ("looks like a good plan").*

**Read in this order:** this file → `docs/plans/20260716-m7-session7-FINDINGS.md` (the numbers, the
H.8 story, the H.9 pre-audit §2c, the ladder verdict §4) → `docs/KOKKOS_PORTING_LESSONS.md` **L95**.

---

## 0. THE RULES. Compressed; every one has a lesson number and a scar.

- **0.1 "ALWAYS MEASURE, DO NOT GUESS"** *(user directive)*. Session 7 made it 4-for-4: every plan
  estimate re-examined against source was wrong (H.8's "ocean half" did not exist as scoped).
  *Measure → read the code → pre-register. In that order.*
- **0.2 THE GAP CENSUS IS A FLOOR — except when it's EXACT (L93, updated).** Levers whose gap
  contains host compute get an entanglement bonus (SMOOTHSCRATCH −2.8 pre-reg → −4.21). A pure
  PCIe+fence deletion gets NONE: LAZYSNAP's census said 7.3 ms and the A/B returned −7.3 ms TO THE
  DECIMAL. Classify the gap's contents before widening the ceiling.
- **0.3 NO SERIAL GATE VALIDATES A COHERENCE INVARIANT (L86).** The 42-second CUDA fidelity gate is
  THE gate for any rail/sync change. Before deleting a rail, ask **who put the initial value there**.
- **0.4 THE OPTIONS MATRIX IS PER-LEVER (L91)** — and H.9 changes ownership of fields **zstar
  actively reads** (hbar/hbar_old feed the ALE chain). The zstar leg is not a formality this time.
- **0.5 A PROTOCOL IS VALIDATED PER MESH (L95).** dars@8N dies deterministically at step 204 from
  cold PHC (dt180, dist_1024) — dars rows use the **150-step protocol, annotated**. Both reps dying
  at the SAME step = physics, not hardware.
- **0.6 🔴 THE `gpu` PARTITION IS HETEROGENEOUS (L94).** `-C a100_80` on every absolute anchor
  (`job_m7_ab_env`/`job_m7_hostprof` already pin it — never remove). An A/B is immune; an anchor is not.
- **0.7 ALWAYS PIN `BIN=`** (a running multi-`srun` job re-execs per srun). **Backfill trick:** ask
  6–10 min for a 30-second gate or it will not run behind a fat reservation.
- **0.8 THE PER-RANK PROXY IS FOR REGIME FRACTIONS, NEVER FOR LEVER TRIAGE** (26248860: the FLAT/ROT
  ordering FLIPS between dars@8N and real NG5@16N). Rank levers only at the real target point.

**Binaries** `m7/bin/…`: 🔴 `h3` = broken ICERAILS, never use. `h8` = CUDA `7dab6c5a`. `h9` = +H.7
(CUDA `9e1f514b` / Serial `91eeb573`). ✅ **`h10` = CURRENT BEST, FULLY CERTIFIED** (+H.8 LAZYSNAP):
CUDA **`13dbddb4`** / Serial **`7c75afc0`** — 9/9 gates green incl. options ×3 + guard-abort
(26259160-69; `PROVENANCE.txt` in the dir).

---

## 1. WHERE THE CAMPAIGN IS (one paragraph)

**Stage-1 MET: 6.87× at NG5@4N** (h10, 0.6666 / 4.5785, matched 300-step pinned pair).
**Stage-2 MET WITH MARGIN: NG5@16N SYPD@dt240 = 2.37** (0.657/0.2688/1.03; ×1.019 is the
4N-measured CG correction — `scripts/m7_cg_share.py`). Clean standard set: **4N 6.87× / 8N 5.68× /
16N 4.56× / dars@8N 4.15×** (150-step, L95). The step is **~80 % kernels-busy**; total gaps >1 ms
are only ~39 ms/step after H.8. What's left of the charter is the **8× stretch at 4N = another
~14 % off the step** (0.6666 → ~0.57), and the pools big enough are **B/C (kernels) + E (comm)**.
The 26248860 ladder put **B/C back on top** (FLAT holds 56 % of its 4N value at real 16N).

---

## 2. THE SHAPE OF THIS SESSION: measurements in the background, H.9 in the foreground.

### 2.1 Fire these FOUR background jobs FIRST (all cheap; user-approved plan)

| # | what | how | why |
|---|---|---|---|
| 1 | **C precondition: re-derive the spill pool** | LOGIN NODE, free: `cuobjdump --dump-resource-usage build-m7cuda/fesom_port` (or the h10 binary) → per-kernel regs/spill; cross-rank by kernel-busy ms from the gap300_h9 sqlite (or the new h10 one) | package C's list is STALE — `diff_ver_part_redi_expl` (42.3 ms, the biggest spiller) is not in it |
| 2 | **B precondition: FCT's tracer-invariant traffic fraction** | `job_ncu_fctgm_ng5` (BIN=h10) → DRAM/L2 bytes per FCT kernel; then from source, classify each input array as per-tracer vs tracer-invariant (mesh geometry, mass matrix, areas) and compute the invariant fraction of measured traffic | B's payoff = hoisting the invariant traffic across the 2 tracers; **nobody has measured the fraction** — pre-register a go/no-go line BEFORE reading the result |
| 3 | **h10 gap census** | `NSYS_TRACE=cuda,mpi NSYS_SAMPLE=none BIN=<h10cuda> TAG=gap300_h10 sbatch jobs/job_m7_hostprof` then `scripts/m7_gap_census.py` | pre-register: `ice_h_diag→oce_fluxes_mom` GONE; top = halo self-gaps (~17) + the SSH/hbar class (~13-14). This census SIZES H.9 (rule 0.1) |
| 4 | **1-yr climate gate on h10** | `BIN=<h10cuda> sbatch jobs/job_m7_tier1_cuda_1yr` (check the job's tag/ref conventions; refs in `docs/REFERENCE_RUNS.md`) | the per-tier climate gate — packH (h5→h10) has never had one; Tier-1 did (26238055) |

Then go straight to §3. Harvest as they land.

### 2.2 Standing queue discipline
Nothing else is in flight. All session-7 jobs are harvested and written up (findings doc + ledger).

---

## 3. 🔴 THE FOREGROUND LEVER — **H.9 "SSHRAILS"**: the SSH/hbar host-staged nod2D bounce class

**~13–14 ms/step of gap ≈ −1.9 % at 4N (pre-H.8 census; re-size from gap300_h10 FIRST and
pre-register a floor). HOST-class lever ⇒ per L84(b) + the 26248860 ladder it should HOLD at 16N.**

### 3.1 The inventory (verified in source, session 7 — findings §2c has the census rows)

Per step, five nod2D fields bounce device→host→device to feed **four host MPI exchanges** and
**one host loop**:
- `fesom_step.cpp:630-633` — pushes d_eta + ssh_rhs_old + hbar to device (10.6 MB HtoD)
- `:646-647` — ssh_rhs sync_host → `fesom_exchange_nod2D` (host halo #1)
- `:652-658` — d_eta sync_host → host halo #2 → modify_host+sync_device re-push (L30)
- `:670-674` — ssh_rhs_old/hbar/hbar_old sync_host (10.6 MB DtoH) → host halos #3+#4
- `:681-682` — hbar/hbar_old re-push (7.1 MB HtoD) for the device dhe_fill
- `:776-784` — **the eta_n update `eta_n = α·hbar + (1−α)·hbar_old` on the HOST** (reads
  hbar/hbar_old, writes eta_n) + the `:818` push

### 3.2 Shape of the change (ICERAILS is the template — `fesom_halo_field2 NOD2D` already exists)

Knob `FESOM_SPEED_SSHRAILS`. Under it: the four `fesom_exchange_nod2D` become device NOD2D halos
(`fesom_halo_field`/`field2`); the eta_n update becomes a trivial per-node device kernel (no
scatter); the five fields go device-authoritative across the step; every push/sync in §3.1 dies.

### 3.3 🔴 THE TRAP LIST (each is an ICERAILS-class scar — work through ALL of them)

1. **`fesom_ice.cpp:633`**: `hbar_fld.modify_host(); sync_device()` with the comment *"hbar IS
   host-authoritative — keep"*. Under SSHRAILS that flips: the push becomes the BULKTAIL-IC-push
   clobber (Z7 signature, step-1-only wrongness). Gate it on `!sshrails` — and answer rule 0.3's
   question for hbar (who writes the IC? `fesom_ic.cpp`? verify).
2. **eta_n's other consumers**: the `:511/:519` substep-4 push becomes redundant (its comment
   already says the `:776` loop is the ONLY per-step host writer); `resolve_ssh_dev`
   (`fesom_io.cpp:868`, IOACC accumulator) reads eta_n on DEVICE per step — a device eta_n kernel
   FEEDS it correctly, but re-read the 30,000×-scar comment at `:785-818` before touching anything.
   The snapshot gather reads eta_n via `h_checked()` → add eta_n to the snapshot-cadence sync (the
   H.8 writer-pull pattern; eta_n is NOT in the fesom_main.cpp:1375 block because it used to be
   host-authoritative).
3. **hbar/hbar_old host readers to cadence-check**: `fesom_ale.cpp:804/:829` (which ALE mode? init
   or per-step?), the print/BLOWUP blocks (print cadence — never fires in benchmarks), the
   `fesom_ale_dump_*` bisect rails (env-gated → abort or sync-in-place), `fesom_main.cpp:519/:751`.
4. **zstar reads hbar in the ALE chain** (`update_stiff_mat_ale`, dhe, thickness) — check
   device-vs-host reads per mode. **The zstar options leg is load-bearing for this lever** (0.4).
5. **d_eta is CG's cross-step RMW state** (`:619-627` capture list — the initial guess). The
   coherence derivation must cover the step BOUNDARY, not just the step.
6. **CG reads ssh_rhs at OWNED rows only** (`:643-644` comment) — the halo requirement per field is
   NOT uniform; derive per-field who reads halo rows (d_eta: update_vel at 3 verts incl halo;
   hbar: dhe_fill incl halo).
7. Requirements/abort (L80): derive the dependency set from the audit (ICERAILS? IOACC for
   resolve_ssh_dev? FESOM_DIAG_SSHSLV/SPREAD read these fields per step when set → abort). Add a
   **guard-abort test to the ladder** (H.8's 26259169 pattern).

### 3.4 Gate ladder (identical to H.8's, which went 9/9)

knob-OFF byte → FORCE_SERIAL byte proof (levered Serial vs baseline; snapshots ON in gate config)
→ CUDA fidelity isolated + full → **options ×3 (zstar is THE one)** → guard-abort test → 35-step
A/B (pre-registered) → 300-step h11 anchor (a100_80) → ledger + freeze `m7/bin/h11`.

---

## 4. AFTER THE MEASUREMENTS: pick B vs C vs E for the 8× stretch

- The 26248860 ladder verdict (findings §4): **B/C first**, E stays live. Decide on the numbers
  from §2.1 jobs 1+2, not on priors.
- **B (FCT2)** goes ahead only if the tracer-invariant fraction clears your pre-registered line.
- **C (TDMA/spills)** goes ahead on what `cuobjdump` × kernel-busy actually names (expect
  `diff_ver_part_redi_expl` to enter the pool at rank 1).
- **E (comm)**: the halo self-gaps (~17 ms) are the top of the census either way; if B/C
  preconditions BOTH fail, E is the default.
- Keep the honest denominator: ladder percentages were against `packa` (0.3420 at 16N) — recompute
  any 16N projections against the current 0.2688.

## 5. TOOLING

`scripts/m7_gap_census.py` (census + `--diff`, FENCE column, PCIe pair-table) ·
`scripts/m7_cg_share.py` (NEW: CG region share from any nsys sqlite) · `scripts/m7_stall_budget.py`
(host-side complement) · `jobs/m7_provenance.sh` (SHA.txt in every OUT dir — check md5 FIRST when
harvesting).

## 6. USER PREFERENCES (standing)

- **"Always measure, do not guess."** Pre-register BEFORE the job lands; check the announce line
  (L80) and the `SHA.txt` md5 on every harvest.
- **Ask before pushing or tagging.** (Session 7's push + `m7.4-stage2met` were user-approved.)
  Mixed precision **BANNED** (FP64 only).
- SLURM only for big runs; output under `/work/ab0995/a270088/port2/m7/`; scratch for temporaries.
- CORE2 gates use the **private mesh** `/work/ab0995/a270088/port2/mesh/core2`, never `/pool`
  (L73); NG5/dars perf meshes DO come from /pool.
- dars rows: **150-step protocol only** (L95), annotated in the ledger.
- The user reads these sessions closely — when two of your own numbers disagree, resolve it in the
  open.
