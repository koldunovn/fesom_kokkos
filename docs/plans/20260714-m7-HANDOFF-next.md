# M7 HANDOFF — next session (Tier 1 is DONE; where the remaining headroom actually is)

*Written 2026-07-14 at the close of the Tier-1 session. Milestone tag: **`m7.1-stage1`** (annotated,
`db36776`) — read `git show m7.1-stage1` for the full record. Campaign log + ledger:
`docs/GPU_SPEED_M7.md`. Plan of record: `docs/plans/20260714-m7-speed-fp64.md`.*

---

## 1. Read these three things before touching anything

They each cost real time to learn, and they will bite again.

- **L80 — a dead knob passes EVERY gate.** `docs/KOKKOS_PORTING_LESSONS.md`. `fesom_speed.hpp`'s
  `#ifndef KOKKOS_ENABLE_CUDA` guard fires on a *CUDA* build if the header is included before Kokkos'
  generated config → **every knob in that TU silently resolves to OFF**. The knob-OFF byte gate
  passes, the fidelity gate passes, and **the FORCE_SERIAL byte proof passes *because* FORCE_SERIAL
  bypasses the very guard that is killing the knob**. The A/B faithfully reports 0.00%.
  **Only physics caught it** (a 261 MB memset cannot cost 0.01%).
  *Now mitigated:* the header includes `<Kokkos_Macros.hpp>`, and **every lever announces itself on
  rank 0** (`[fesom_speed] FESOM_SPEED_X = ON`) and shouts if requested-but-OFF. **Check that line in
  the log before believing any A/B.**
- **L81 — a profiler tells you WHERE the time is, never WHICH SOURCE LINE.** `-t cuda,mpi` cannot see
  inside host code by construction. It localises; it does not attribute. Two agreeing measurements of
  the same *window* are not independent confirmation of an *attribution*. **Only a same-allocation A/B
  turns an attribution into a fact — run it before writing the number down.** To name host time:
  `nsys profile --sample=process-tree --backtrace=dwarf -t osrt` (values are
  `process-tree|system-wide|none` — **not** `cpu`), then query `SAMPLING_CALLCHAINS` in the sqlite;
  the `cpu_profile` stats report does not exist in this nsys.
- **Never extrapolate one scale point's factor to another.** The Tier-1 gain tracks **per-rank domain
  size** (−28.5% at ~462k nod2D/rank → −17.5% at ~99k). I extrapolated the 4N factor to 16N, published
  "SYPD 1.99", and the dars@8N A/B caught it. Measured: **1.86**.

---

## 2. Where things stand

**Stage-1 target MET.** Branch `m7-speed`, tag `m7.1-stage1` @ `db36776`. **40 commits, NOTHING PUSHED**
(the push decision is the user's and has not been made).

| | row-0 | Tier-1 | now | **TIER 1** | gain | nod2D/rank |
|---|--:|--:|--:|--:|--:|--:|
| **NG5@4N** | 1.2796 | **0.9145** | 3.60× | **5.03×** ⭐ | −28.5% | ~462k |
| NG5@8N | 0.7381 | **0.5520** | 3.20× | **4.28×** | −25.2% | ~231k |
| NG5@16N | 0.4487 | **0.3432** | 2.72× | **3.55×** (SYPD@dt240 **1.86**) | −23.5% | ~116k |
| dars@8N | 0.3178 | **0.2622** | 2.69× | **3.27×** | −17.5% | ~99k |

**The four levers** (`FESOM_SPEED=1` = all of them; each also has its own knob):

| knob | A/B (NG5@4N) | what |
|---|--:|---|
| `FESOM_SPEED_SWSKIP` | **−26.5%** | skip the **dead** host `sw_3d` (`fesom_bulk.cpp:698`) — M5.20 ported it to the device and left the host copy running: a 261 MB/rank/step `memset` + ~9M `exp()` calls, overwritten microseconds later |
| `FESOM_SPEED_IOACC` | −1.1% | 6 host I/O mean accumulators → device |
| `FESOM_SPEED_NOFENCE2` | −0.8% | drop the post-unpack halo fence |
| `FESOM_SPEED_ICEFLUXDEV` | −0.7% | `ice_oce_fluxes_mom` host loop → device |
| **all four** | **−28.4%** | |

**All four are BIT-IDENTICAL.** Three carry passing FORCE_SERIAL byte proofs; `NOFENCE2` changes no
arithmetic. **The licence to break bit-identity was never spent.** Every gate is green: byte gate,
3× FORCE_SERIAL, fidelity gates, memcheck (0 invalid accesses), racecheck (0 hazards), the **1-yr
CORE2 climate gate** (identical to the un-levered bar to five decimals), and the **options matrix**
(`FESOM_SPEED=1` × TKE / mEVP / zstar).

---

## 3. 🔴 THE STRATEGIC POINT: the bottleneck MOVED, and the plan's tier order is now wrong

Tier 1 deleted the host work. **The step is a different animal now**, and Tiers 2/3 should be
re-targeted accordingly. Derived from the Tier-0 stall budget minus what Tier 1 removed
(**verify with the jobs in §4 — do not trust this table, it is arithmetic, not measurement**):

**NG5@4N, ~915 ms/step (was 1275):**

| component | pre-Tier-1 | **post-Tier-1 (est.)** |
|---|--:|--:|
| **GPU kernels** | 594 ms (46.6%) | **~594 ms (≈65%)** ← now dominant |
| memcpy (mostly MPI staging) | 101 ms (7.9%) | ~101 ms (11%) |
| MPI wait | 106 ms (8.3%) | ~106 ms (12%) |
| launch gap | 38 ms (3.0%) | ~38 ms (4%) |
| host segment | **408 ms (32.0%)** | **~45 ms (5%)** ← gone |
| fence spin | 18 ms (1.4%) | ~8 ms (1%) |

**dars@8N (the 16N-class per-rank proxy), ~262 ms/step (was 318):**

| component | pre-Tier-1 | **post-Tier-1 (est.)** |
|---|--:|--:|
| **MPI wait** | 100 ms (31.2%) | **~100 ms (≈38%)** ← now dominant |
| GPU kernels | 88 ms (27.5%) | ~88 ms (34%) |
| host segment | 79 ms (24.7%) | ~23 ms (9%) |
| launch gap | 29 ms (8.9%) | ~29 ms (11%) |

### What that means for the tier ladder

- **At 4–8N the step is now ~65% GPU-KERNEL-bound → Tier 2 (kernel work) is exactly the right
  lever**, and it is where the Stage-1 *stretch* (8×) has to come from. I told the user at one point
  that Tier 2 looked low-yield; **that was based on the pre-Tier-1 budget and is now wrong.**
- **At 16N the step is ~38% MPI → Tier 3 (CG 1-reduce, Chebyshev precond, wide-halo EVP) is the
  Stage-2 lever.** Host work is no longer the 16N story.
- **The plan's Stage-2 framing needs revisiting:** Tier 1 *raised* the 16N ratio (2.72→3.55×) but the
  *relative* 4N→16N decay **widened** (24%→29%), because a host-work lever pays most where per-rank
  domains are large. Flattening the curve is a comm problem, not a host problem.

### Honest ceiling on the 8× stretch

To reach 8× at NG5@4N the step must go 0.9145 → **0.575 s** (another −37%). Kernels are ~594 ms of
915. Even a **40% cut across all kernels** only gets you to ~677 ms = **6.8×**. **8× at 4N therefore
requires roughly a 55% kernel reduction — that is layout-big-bet territory** (the level-major refactor
in the plan's Post-Completion section), not Tier 2 alone. A realistic Tier-2 outcome is **10–20% →
~5.6–6.3×**. Say this to the user before committing to 8×.

---

## 4. First actions next session (in order)

1. **Harvest the two jobs I left running** — they replace the *estimated* table above with measurement:
   ```
   cat /work/ab0995/a270088/port2/m7/nsys_t1_ng5_4n/stall_budget.txt      # job 26242512
   grep 'STEP PROFILE' /work/ab0995/a270088/port2/m7/stepprof_t1/run.log  # job 26242513
   ```
   Both run `FESOM_SPEED=1` on the frozen Tier-1 binary. **Rebuild the §3 table from these**, then
   pick the tier that owns the biggest remaining slice.

2. **Tier 2, if the user wants the stretch.** The ncu roofline (`docs/GPU_SPEED_M7.md`) already names
   the targets and *rewrites* the plan's Tier-2 reasoning:
   - **`fesom_tracer_advect_one_fct_kk` is the #1 kernel (~20% of the NEW step) and is NOT a
     coalescing problem** — it already runs at **59% of DRAM peak with 7.3 sectors/request**
     (FP64 ideal = 8.0). You cannot speed it up by fixing access patterns. **The only lever is moving
     less traffic → Task 1.3 `FCT2` (T+S batching: hoist and share the geometry/velocity/edge loads
     across both tracers).** This is now the single highest-value kernel item.
   - **Every TDMA is spill-bound AND uncoalesced** (`impl_vert_visc` 7168 B/thread + 23.2 sec/req;
     `fer_solve_gamma` 7168 B + 22.0; `diff_ver_part_impl_ale` 6144 B + 23.6) → **Task 2.3**.
   - ➕ **New target the plan didn't know about: `kpp_ri_iwmix_kk`** — SM 2.8%, **23.6 sec/req**, the
     worst coalescing in the set.
   - `fesom_smooth_nod3D_kk` is the **control group**: 2.8 sec/req, SM 45.7%, occupancy 88.6%,
     "balanced" — that is M5.18's own kernel, proving the coalescing lever works and is *done*.
   - ⚠️ Register spill is **free and exact from the binary**: `cuobjdump --dump-resource-usage`. Don't
     use ncu for it.

3. **Tier 3, if the user wants Stage-2 (16N).** MPI is ~38% of the 16N-class step. The plan's F1/F2/D2
   (CG single-Allreduce, Chebyshev preconditioner, wide-halo EVP) are unchanged and correctly scoped.
   Note from Tier 0: the CG is **351 kernel launches/step for only 12 ms of GPU time** — it is
   *launch-latency* bound, not compute bound, which argues for kernel-count reduction over kernel
   optimisation.

4. ➕ **`FESOM_SPEED=1` also enables `SYNCSTATS`** (a diagnostic), so production runs with the master
   switch print counters. Harmless but untidy — consider excluding diagnostics from the master switch.

---

## 5. Machinery you inherit (all of it works; don't rebuild it)

| thing | what |
|---|---|
| `jobs/job_m7_gpu_gate` | CUDA fidelity gate, **~35 s**. `KNOBS=`, `SREF=` (for options-matrix), `BIN=`. Cheap enough to run per lever. |
| `jobs/job_m7_gate_serial` | knob-OFF byte gate **and** the FORCE_SERIAL byte proof (pass `KNOBS="FESOM_SPEED_FORCE_SERIAL=1;FESOM_SPEED_X=1"`) |
| `jobs/job_m7_ab_gpu` | **same-allocation A/B** — both legs, same nodes, same binary, only the knob differs. The arbiter. |
| `jobs/job_m7_sanitize` | compute-sanitizer, **baseline-differential** (knobs OFF then ON; only the difference is attributable). ⚠️ **`racecheck` is shared-memory-only** — for a fence/sync removal the tool that can actually see the hazard is **memcheck**. |
| `jobs/job_m7_nsys` + `scripts/m7_stall_budget.py` | nsys → sqlite → **named** stall attribution (fence drains / launch gaps / MPI / host). Interval algebra brute-force-tested. |
| `jobs/job_m7_hostprof` | CPU **call-stack** sampling — the only thing that can name host time |
| `jobs/job_m7_ncu_top10` + `scripts/m7_roofline.py` | self-calibrating ncu roofline |
| `jobs/job_m7_tier1_cuda_1yr` | the 1-yr climate close (refs + the bar are baked in) |
| `jobs/job_m7_scale_{gpu,cpu}` | standard set; `KNOBS=`, `BIN=` |
| `/work/.../m7/bin/row0/` | frozen **row-0** (certified-source) binaries |
| `/work/.../m7/bin/tier1/` | frozen **Tier-1** binary (`788844b3`) |

**Measurement traps baked into those scripts (don't re-learn):** CUPTI names the CUDA API with a
version suffix (`cudaStreamSynchronize_v3020`); **not every device sync is yours** —
`cudaStreamSynchronize` tracks the *message* count (it is MPI's), **ours is `cudaDeviceSynchronize`**;
nsys/ncu see **C++ symbols**, never Kokkos runtime labels (a label-based ncu regex matched 2 of 10
kernels **and ncu still exited 0**); `MPI_START_WAIT_EVENTS` has one row **per request**, not per
Waitall call; ncu **normalises units** (derive GB/s from the unit-free `pct_of_peak`).

---

## 6. Standing rules

- **Same-day, same-allocation A/B only** (±10% inter-allocation noise). `jobs/job_m7_ab_gpu`.
- **Knob-OFF byte gate green at every commit.** Bit-id-claimed levers additionally get the
  FORCE_SERIAL byte proof.
- **CORE2 = the private mesh** `/work/ab0995/a270088/port2/mesh/core2` (never `/pool`, L73).
- **Verify the knob FIRED** (`[fesom_speed] … = ON` on rank 0) before believing any A/B.
- **Sanity-check every payoff against a physical floor** (bytes ÷ bandwidth, flops ÷ rate). A result
  below the floor means the plumbing is wrong, not the hardware.
- Output under `/work/ab0995/a270088/port2/m7/` only. **Nothing is pushed unless the user asks.**
- ⚠️ Don't combine `SWSKIP` with a `-DFESOM_KK_VERIFY` build (the verify-only `kpp_bldepth` twin reads
  the host `sw_3d`). `FESOM_SPEED_FORCE_SERIAL` is dev-only.

---

## 7. Open decisions for the user

1. **How far to push?** Stage-1 is met. Tier 2 ≈ **5.6–6.3×** plausible; **8× needs the layout
   big-bet**. Tier 3 is the Stage-2 (16N) lever.
2. **Push `m7-speed` + tags?** 40 commits, 3 tags, nothing pushed.
3. The plan is still in `docs/plans/` (not `completed/`) because Tiers 2/3 are live.
