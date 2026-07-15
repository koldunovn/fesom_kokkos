# M7 next session (session 9) — PROMPT

*Written 2026-07-15, close of session 8. Branch `m7-speed`, tag `m7.5-h11` @ `adfa603` — pushed
to origin (user-approved). Working tree clean. The direction below (C first) was decided on the
session-8 measurements per the pre-registered rules and approved by the user.*

**Read in this order:** this file → `docs/plans/20260716-m7-session8-FINDINGS.md` (the four
measurements §2, the H.9 story §3, the B/C/E decision + C.1 pre-audit §4) →
`docs/KOKKOS_PORTING_LESSONS.md` **L96**.

---

## 0. THE RULES. Compressed; every one has a lesson number and a scar.

- **0.1 "ALWAYS MEASURE, DO NOT GUESS"** *(user directive)*. Measure → read the code →
  pre-register. In that order. Session 8 scored 6/6 pre-registrations because the order was kept.
- **0.2 THE CENSUS CALIBRATION IS NOW TWO-SIDED (L93, closed).** A pure PCIe+fence gap deletes
  census-EXACT (H.8: −7.3 ms on a 7.3 ms gap); a gap with HOST COMPUTE in it collects an
  entanglement bonus (H.9: −15.7 ms on a 14.8 ms class). Classify the gap contents, then size.
- **0.3 NO SERIAL GATE VALIDATES A COHERENCE INVARIANT (L86)** — but C.1 is the opposite case: it
  claims BIT-IDENTITY, so the **FORCE_SERIAL byte proof is THE gate** for it (plus CUDA fidelity
  as always).
- **0.4 THE OPTIONS MATRIX IS PER-LEVER (L91)** — cheap, and it runs even for levers that "change
  no ownership". GM/Redi is active in all three legs.
- **0.5 A PROTOCOL IS VALIDATED PER MESH (L95).** dars rows: **150-step protocol, annotated**.
- **0.6 🔴 THE `gpu` PARTITION IS HETEROGENEOUS (L94).** `-C a100_80` on every absolute anchor.
  An A/B is immune; an anchor is not.
- **0.7 ALWAYS PIN `BIN=`; A CHEAP JOB MUST *LOOK* CHEAP.** Paid twice now: the H.9 A/B sat
  un-backfillable on a 1:30 claim and ran instantly at `-t 00:25:00`. Size walltime to the work.
- **0.8 THE PER-RANK PROXY IS NEVER FOR LEVER TRIAGE** (26248860). Rank levers at the real point.
- **0.9 🔴 L96 — a kernel-resource table lies two ways**: cuobjdump spills live in **STACK**, not
  LOCAL; and never name a kernel by its first `fesom_*` token (parameter types / module hashes
  masquerade — the TDMA kernel hid in a fake "fesom_mesh" row through two campaigns). Use
  `scripts/m7_spill_pool.py` / `m7_kernel_busy.py`, which do it right and cross-sum to the census.
- **0.10 Two documented warts — don't rediscover them:** (a) full-blessed runs **SIGABRT at
  TEARDOWN** after all output is written → the JOB state reads FAILED while the content is
  complete; judge runs by their content (loop timing + output files), not sacct. (b) the
  print/BLOWUP block reads uv/w/T/S from STALE mirrors under the blessed set (CUDA blowup
  detection is effectively dead outside Serial; eta_n is synced there since H.9). Both are
  pre-existing, recorded in findings §2.4/§3.1 — cleanup levers someday, not now.

**Binaries** `m7/bin/…`: 🔴 `h3` = broken ICERAILS, never use. `h10` = CUDA `13dbddb4` / Serial
`7c75afc0`. ✅ **`h11` = CURRENT BEST, FULLY CERTIFIED** (h10 + H.9 SSHRAILS): CUDA **`d74d31b4`**
/ Serial **`c125b424`** — 9/9 gates (26265018-26) + A/B −2.29% (26265290) + anchor 0.6503
(26265348); PROVENANCE.txt in the dir. packH climate-certified at 1 yr on h10 (26264208, the
M5.23 bar to the digit).

---

## 1. WHERE THE CAMPAIGN IS (one paragraph)

**⭐ NG5@4N ratio = 7.04×** (h11, 4.5785/0.6503, matched 300-step pinned pair, pure a100_80).
Stage-1 (≥5×) met with 41 % margin; **Stage-2 met** (16N SYPD@dt240 = 2.37, h9-measured — the
16N/8N GPU legs are still h9 and should be refreshed on h11, §2.1). The step is ~80 % kernel-busy;
gaps >1 ms ≈ 39.8 ms minus H.9's 14.8. **The 8× stretch = 0.6503 → ~0.572 = another −12 %**, and
the only pool that big is **package C: 188.8 ms/step of busy time in 10 spilling kernels (28 % of
the step)**. B is PARKED (invariant fraction f = 0.242 < the pre-registered 0.25 line; honest
payoff 1.5–2.2 % for the biggest rewrite on the table). E (halo self-gaps, ~18 ms at 4N, proven
MPI-wait) stays live, esp. toward 16N.

---

## 2. THE SHAPE OF THIS SESSION: measurements in the background, C.1 in the foreground.

### 2.1 Fire these background jobs FIRST

| # | what | how | why |
|---|---|---|---|
| 1 | 🔴 **C.1/C.2 pre-registration input: local-memory traffic per spiller** | `job_ncu_fctgm_ng5` with `BIN=<h11cuda>`, `KNOBS="FESOM_SPEED=1"`, `NCU_REGEX="redi_expl\|impl_ale\|momentum_adv_scalar\|impl_vert_visc\|pressure_bv\|fer_solve_gamma\|tracer_advect_one_fct"`, `NCU_METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"` (maybe `NCU_COUNT=120`) | **BLOCKS the C.1 pre-registration** — the local ld/st bytes are the spill traffic itself; C.1's floor = local-bytes share × kernel time. Also prices C.2/C.3+ in the same job. Harvest BEFORE building. |
| 2 | **gap300_h11 census** | `NSYS_TRACE=cuda,mpi NSYS_SAMPLE=none BIN=<h11cuda> TAG=gap300_h11 NSTEPS=300 KNOBS="FESOM_SPEED=1" sbatch jobs/job_m7_hostprof` | pre-registered in session-8 findings §3.2: the five SSH/hbar rows GONE; halo self-gap rows grow by the four new NOD2D exchanges' wait. Re-sizes E and the H.10 tail (ice-thermo ≈4.3 ms). |
| 3 | **standard-set GPU refresh on h11** | 8N + 16N NG5 GPU (300-step) + dars@8N GPU (150-step, L95), all `BIN=<h11cuda>`, a100_80, min of 2 (the std300 single-leg job_m7_ab_env pattern; CPU legs never move) | H.9 is HOST-class ⇒ per L84(b)+the ROT precedent it should hold ~60 % at 16N. **Pre-register: 16N ≈ 0.2650 (−1.4 ±0.5 %) ⇒ SYPD@dt240 ≈ 2.40; 8N ≈ 0.407 ±0.5 %; dars@8N ≈ 0.201 ±0.7 %.** Below 16N −0.5 % ⇒ H.9 decayed like a kernel lever — worth knowing either way. |
| 4 | **1-yr climate gate on h11** | `BIN=<h11cuda> KNOBS="FESOM_SPEED=1" TAG=packh11_cuda_1yr sbatch jobs/job_m7_tier1_cuda_1yr` | h10's gate predates H.9. Bar: sst 1.00000 · sss 0.99996 · ssh 1.00000 · a_ice 0.99997. ⚠️ Expect the job state FAILED from the teardown wart (0.10a) — judge by the compare table. |

Then go straight to §3. Harvest as they land.

### 2.2 Standing queue discipline
Nothing is in flight. Everything from session 8 is harvested, committed, and pushed.

---

## 3. 🔴 THE FOREGROUND LEVER — **C.1 "REDISWEEP"**: kill `diff_ver_part_redi_expl`'s spills

**Pool row: 42.4 ms/step (36.8 in the per-node kernel), 58 reg, 5,120 B/thread STACK. Sizing
prior 1.7–2.5 % at 4N — but PRE-REGISTER ONLY AFTER job §2.1-1 lands (the local-bytes share is
the floor's denominator). Kernel-class lever ⇒ ~56 % holds at 16N (26248860).**

### 3.1 The anatomy (verified in source, session 8 — findings §4.1)

`fesom_gm_redi_ver_node` (fesom_gm.cpp:1807): per OWNED node, THREE column passes with FIVE
`NL_MAX=128` local arrays — `txn/tyn` (the surrounding-element gather, :1813-1829),
`zbar_n/z_n` (bottom-up depth recurrence from hnode, :1831-1839), `vd_flux` (:1841-1856), then
the apply loop (`vals += (vd_flux[nz]−vd_flux[nz+1])·dt/(av·hn)`, :1858-1865).
5 × 128 × 8 B = **exactly the measured 5,120 B/thread**. The sibling sub-kernels (trxy, zero)
are clean — don't touch them.

### 3.2 Shape of the change (knob `FESOM_SPEED_REDISWEEP`, rides the master)

**A single bottom-up column sweep with O(1) carried state — no local arrays:**
- The zbar_n/z_n recurrence ALREADY runs bottom-up; carry `zbar_cur/z_cur/z_prev` as scalars.
- The txn/tyn gather per level is order-free; fuse it into the sweep, carry a rolling pair
  (`tx_cur/ty_cur` + `tx_up/ty_up` — vd_flux[nz] reads levels nz−1 and nz).
- `vd_flux` consumption is one-level-lagged (apply at nz needs vd_flux[nz] and vd_flux[nz+1]);
  carry a rolling pair, apply one level behind the sweep.
- **Every value keeps its exact expression and evaluation order** (the gather loop order over
  `off/nie` unchanged; the recurrence updates unchanged) ⇒ bit-identical on Serial by
  construction ⇒ FORCE_SERIAL byte-provable (rule 0.3 inverted — this is the rare kernel lever
  with a byte proof).

### 3.3 🔴 THE TRAP LIST

1. **Boundary zeros**: the flux loop writes vd_flux only on [ule+1, nle); the apply loop reads
   vd_flux[ule] (never written → 0 from init) and vd_flux[nle] (0). In the sweep these zeros
   must be EXPLICIT (`flux_below = 0` at the bottom entry; `flux_cur = 0` when the loop exits at
   ule). Off-by-one here is bitwise-visible — the byte proof will catch it, but read it twice.
2. **Sweep direction vs apply direction**: the apply loop originally runs TOP-DOWN (ule→nle) but
   is order-free (each nz writes only vals[nz] with += of values that don't depend on other
   levels' writes). Verify `vals` is not read at other levels inside the same loop — it isn't
   (:1862 reads only vd_flux and areasvol/hnode_new). So a bottom-up apply is bit-identical
   per-element; state that in the commit message, don't just assume it.
3. **The surface special case** (:1833-1839): zbar_n[nle]=zbar(nle) seed, z_n[nle−1] half-step,
   zbar_n[ule] final step OUTSIDE the loop. Keep the exact same sequence of adds in the sweep.
4. **Register pressure is the whole point**: acceptance criterion = **cuobjdump STACK == 0 for
   the new kernel** (free, login node, `scripts/m7_spill_pool.py` names it). If the compiler
   still spills (e.g. it keeps the rolling state in memory), check reg count and consider
   `Kokkos::LaunchBounds`. A rewrite that halves the stack is NOT the lever — 0 is.
5. **Do not fuse away the `if (nle <= ule) return`** cavity guard or the `av>0 && hn>0` apply
   guard — both are branch-order-visible in the byte proof.
6. **The host C twin** (fesom_gm.cpp:635 `fesom_diff_ver_part_redi_expl`) stays untouched — it
   is the knob-OFF/verify reference. The lever only touches the `_kk` node kernel, gated.
7. **T and S both call it** (fesom_step.cpp:1000/1029) — the knob branch lives INSIDE the kernel
   function, not at the call sites, so both tracers flip together.

### 3.4 Gate ladder

knob-OFF byte → **FORCE_SERIAL byte proof (isolated: `FORCE_SERIAL=1;REDISWEEP=1`) — THE gate**
→ FORCE_SERIAL full blessed → cuobjdump **STACK==0** check (login, free, before burning GPU
hours) → CUDA fidelity ×2 (isolated + full) → options ×3 → 35-step A/B (pre-registered from the
§2.1-1 ncu) → 300-step h12 anchor (a100_80) → ledger + freeze `m7/bin/h12`.
(No guard-abort test: REDISWEEP has no dependency set — it requires nothing and leaves no stale
mirror; the knob-OFF byte gate covers the OFF path.)

---

## 4. AFTER C.1 — C.2 and the re-rank

- **C.2 = `diff_ver_part_impl_ale_kk`** (fesom_tracer_diff.cpp:403, the TDMA kernel, 32.6 ms,
  6,144 B/thread = 6 NL_MAX arrays, 66 reg). 🔴 **Honest pre-scope: O(1) state is IMPOSSIBLE for
  Thomas** — backward substitution needs the forward sweep's modified coefficients (O(nl) per
  column, irreducibly). The options, in audit order: (a) count the ACTUAL arrays — 6 now; the
  M5.24 in-place aliasing precedent suggests some are removable (2 is the Thomas minimum:
  c′ and d′); (b) shared-memory staging (budget check: 128 thr/blk × 2 × nl × 8 B ≈ 143 KB > SM
  budget at that block size — smaller blocks change nothing arithmetically, launch config is not
  in the byte contract... but VERIFY the reduction/scan structure first); (c) launch-bounds /
  occupancy tuning (no arithmetic change); (d) PCR/cyclic-reduction = DIFFERENT ARITHMETIC ⇒
  loses the Serial byte proof, needs the fidelity-gate path — legal in M7 but last resort.
  **Audit before pre-registering anything.**
- The **momentum_adv_scalar / diff_part_hor_redi / zal_a34 / impl_vert_visc / pressure_bv /
  fer_solve_gamma** rungs follow the same per-kernel pattern; re-rank them on the §2.1-1 ncu
  local-bytes numbers, not on busy time alone (80-reg kernels like diff_part_hor_redi may be
  occupancy problems, not spill problems).
- **E** re-sizes from the gap300_h11 census; if C.1's realized payoff lands under ~1 %, E's
  ~18 ms MPI-wait pool (and its 16N growth) is the pivot. **H.10 (ice-thermo bounce, ≈4.3 ms)**
  is the small-lever filler if a session needs a low-risk win.
- Keep the honest denominator: 4N percentages against **0.6503**; 16N projections against the
  refreshed 16N leg from §2.1-3 (not 0.2688, once superseded).

## 5. TOOLING

`scripts/m7_gap_census.py` (census + `--diff`) · `scripts/m7_spill_pool.py` (**NEW**: cuobjdump
STACK × kernel-busy, L96-proof) · `scripts/m7_kernel_busy.py` (**NEW**: per-kernel busy from any
census sqlite, correct naming) · `scripts/m7_cg_share.py` · `jobs/m7_provenance.sh` (SHA.txt —
check md5 FIRST when harvesting) · `job_ncu_fctgm_ng5` (now honors BIN/KNOBS/TAG/NCU_METRICS/
NCU_REGEX) · `job_m7_tier1_cuda_1yr` (now TAG-able).

## 6. USER PREFERENCES (standing)

- **"Always measure, do not guess."** Pre-register BEFORE the job lands; check the announce line
  (L80) and the `SHA.txt` md5 on every harvest.
- **Ask before pushing or tagging.** (Session 8's push + `m7.5-h11` were user-approved.)
  Mixed precision **BANNED** (FP64 only).
- SLURM only for big runs; output under `/work/ab0995/a270088/port2/m7/`; scratch for temporaries.
- CORE2 gates use the **private mesh** `/work/ab0995/a270088/port2/mesh/core2`, never /pool (L73);
  NG5/dars perf meshes DO come from /pool.
- dars rows: **150-step protocol only** (L95), annotated in the ledger.
- The user reads these sessions closely — when two of your own numbers disagree, resolve it in
  the open.
