# M7 — speed beyond bit-identity (pure FP64): campaign log

*Plan: `docs/plans/20260714-m7-speed-fp64.md`. Branch `m7-speed` off `main@69e506d`. Goal: GPU-node/CPU-node ratio ≥5× (stretch 8×) at NG5/dars 4–8N (Stage 1), flatten the decay toward 16N (Stage 2), pure FP64, no repartitioning. Bit-identity replaced by the two-level gate (per-lever fidelity+A/B; per-tier 1-yr climate; knob-OFF byte-identical always).*

---

## ✅ THE ANSWER: the ~25% WAS the dead host `sw_3d`. A dead knob hid it.

**`FESOM_SPEED_SWSKIP` collapses the coupling phase 327 ms → 12 ms and cuts the step 25.5%.**

Phase profile, **same binary** (`788844b3`), knob off vs on — `m7/stepprof_ng5_4n` vs
`m7/stepprof_swskip` (job 26237118):

| phase | knob-OFF | `SWSKIP=1` | Δ |
|---|--:|--:|--:|
| forcing | 0.1007 s | 0.1010 s | — |
| sea-ice | 0.1274 s | 0.1272 s | — |
| **coupling** | **0.3270 s** | **0.0118 s** | **−0.3152 s** |
| ocean | 0.7239 s | 0.7003 s | −0.0236 s |
| **step** | **1.3272 s** | **0.9885 s** | **−25.5%** |

The arithmetic closes exactly: the step drop (−0.3387 s) = the coupling collapse (−0.3152 s) + a
small ocean gain (−0.0236 s, cache pressure from the 261 MB/step `memset` going away).

### ✅ CONFIRMED BY THE SAME-ALLOCATION A/B (job 26237206) — **−26.47%**

Same nodes, same binary, only the knob differs. Both reps agree to 0.01%, and the knob-OFF leg
(1.2788) reproduces row-0 (1.2796) exactly.

| leg | rep a | rep b | min |
|---|--:|--:|--:|
| knob-OFF | 1.2788 | 1.2808 | **1.2788** |
| `SWSKIP=1` | 0.9403 | 0.9404 | **0.9403** |
| | | | **−26.47%** |

**Ratios with SWSKIP** (row-0 GPU × 0.7353, CPU unchanged — the lever is CUDA-path-only):

| | GPU now | GPU +SWSKIP | CPU | ratio now | **ratio new** | SYPD@dt240 |
|---|--:|--:|--:|--:|--:|--:|
| **NG5@4N** | 1.2796 | **0.9409** | 4.6005 | 3.60× | **4.89×** | — |
| NG5@8N | 0.7381 | 0.5427 | 2.3624 | 3.20× | **4.35×** | — |
| **NG5@16N** | 0.4487 | 0.3299 | 1.2188 | 2.72× | **3.69×** | **1.42 → 1.93** |
| dars@8N | 0.3178 | 0.2337 | 0.8563 | 2.69× | **3.66×** | — |

**NG5@4N lands at 4.89× — a hair under the 5.0× Stage-1 target, from ONE bit-identical lever** — and
`NOFENCE2` (~0.8%) + `ICEFLUXDEV` (0.72%) + `IOACC` should carry it over. **NG5@16N reaches
SYPD@dt240 = 1.93**, i.e. the ~2 SYPD this campaign was chartered to find, **in pure FP64**, without
the mixed precision the user banned.

### The lever

`fesom_main.cpp:1214-1215` calls the host **and** device shortwave routines back-to-back every step.
M5.20 moved the `sw_3d` penetration profile to the device and removed its 519 MB/step HtoD push —
**but left the host computation running.** The device twin *starts by zeroing the whole array*
(`fesom_sw3d_zero`, `fesom_bulk.cpp:784`) and rewrites every entry, so the host's work is overwritten
microseconds later on BOTH backends. The host function's only unique output is the cheap nod2D
`heat_flux += swsurf` (the device kernel deliberately does not do it, `:794`). The dead half is a
**261 MB/rank/step `memset`** plus an **`exp()` column walk (~9 M `exp()` calls/rank/step)**,
single-threaded, on the critical path of every step.

Skipping it is **bit-identical by construction** and proven by the FORCE_SERIAL byte proof (26237210).

**Triple-confirmed, three independent ways:**

| method | evidence |
|---|---|
| **phase timer** (job 26237118) | coupling **327 ms → 12 ms**; step **−25.5%**; arithmetic closes exactly |
| **CPU sampling** (job 26237176, `--sample=process-tree --backtrace=dwarf`) | **42.6% of leaf samples** in two adjacent addresses inside nvhpc's **math library** = the `exp()`/`log10()` column walk; `__memset_avx2_unaligned_erms` also present (the 261 MB memset); `fesom_cal_shortwave_rad` has **10× the samples** of `fesom_ice_oce_fluxes_mom` (1917 vs 194) — exactly the −25.5% vs −0.72% ratio |
| **code reading** | the device twin zeroes (`fesom_sw3d_zero`) and rewrites every entry; the host's only unique output is the nod2D `heat_flux += swsurf` (`fesom_bulk.cpp:794`) |

*(The sampling run's absolute percentages are diluted by init — `_IO_vfscanf`/`extrap_nod3D`/`flush_all` are mesh-read and I/O, not the step loop — but the math-library dominance and the shortwave-vs-oce_fluxes ratio are unambiguous.)*

### 🔴 Why this took three attempts — the two lessons (L80, L81)

1. **A dead knob passes EVERY gate.** `fesom_speed.hpp`'s `#ifndef KOKKOS_ENABLE_CUDA` guard fires on
   a *CUDA* build if the header is included before Kokkos' generated config — silently forcing every
   knob in that TU OFF. It killed `SWSKIP` (`fesom_bulk.cpp`) and `IOACC` (`fesom_io.cpp`). The
   knob-OFF byte gate passed (knob-OFF is what a dead knob gives you), the fidelity gate passed (the
   output *is* the legacy output), and **the FORCE_SERIAL byte proof passed *because* `FORCE_SERIAL`
   bypasses the very guard that was killing the knob.** The first SWSKIP A/B faithfully measured the
   legacy path and reported **−0.01%**.
   **Only physics caught it:** removing a 261 MB `memset` cannot cost 0.01%. That is not a
   disappointing lever, it is *arithmetically impossible*. **Trust the arithmetic before the gates.**
   *Fixed:* `fesom_speed.hpp` includes `<Kokkos_Macros.hpp>` (include-order-independent, verified by
   preprocessing all five TUs on both backends), and every lever now **announces itself** on rank 0.
2. **A profiler tells you WHERE the time is, never WHICH SOURCE LINE.** The stall budget localised the
   cost to the microsecond but `-t cuda,mpi` cannot see inside host code; I guessed the function and
   was wrong once (`ice_oce_fluxes_mom`, a real bug, worth a real **−0.72%**). Two agreeing
   measurements of the same *window* are not independent confirmation of an *attribution*.

---

## Tier-1 result

| lever | what it removes | A/B NG5@4N | gates |
|---|---|--:|---|
| **`SWSKIP` (1.2)** | the DEAD host `sw_3d` | **−25.5%** (phase profile; A/B 26237206 pending) | ✅ all pass, FORCE_SERIAL byte proof |
| `ICEFLUXDEV` (1.0) | `ice_oce_fluxes_mom` host loop → device | **−0.72%** | ✅ all pass, FORCE_SERIAL byte proof |
| `NOFENCE2` (1.1) | post-unpack halo fence | ~−0.8% | ✅ all pass, memcheck-clean |
| `IOACC` (1.0b) | 6 host I/O accumulators → device | pending (knob was dead; now live) | ✅ byte gate + FORCE_SERIAL proof |

### Still solid from Task 0.3

- Host segment **437.3 ms/step (34.3%)**, confirmed two independent ways, uniform (stdev 1.5%).
  SWSKIP accounts for ~315 ms of it; the remaining ~120 ms is the I/O accumulators (`IOACC`, 54.7 ms)
  and per-exchange host overhead.
- Trace trustworthy: traced step 1274.6 ms vs untraced 1279.6 (0.4%); kernel share 46.6% reproduces
  PROFILE_M522's 46%.
- **Fences are NOT the problem** (spin 1.4%); launch gaps 3.0%. The plan's Tier-1A premise is dead.

## Gate definitions

- **Knob-OFF byte gate:** `jobs/job_m7_gate_serial` — `build-m7serial` CORE2 dist_8 (private mesh, dt1800, 20 steps, snap 10) `diff_snap.py` vs `/work/ab0995/a270088/port2/m6_baseline_serial` → rc=0 required at every commit.
- **FORCE_SERIAL byte proof** (bit-id-claimed levers): same run with `FESOM_SPEED_FORCE_SERIAL=1` + the lever knob → still rc=0.
- **CUDA fidelity gate:** `jobs/job_m7_gpu_gate` (KNOBS-aware, reusable, ~35 s) — CORE2 dist_8 CUDA vs the certified Serial baseline via `gpu_fidelity_check.py`. An M7 lever does not change the physics, so the reference stays `m6_baseline_serial` for EVERY knob state.
- **Tier climate gate:** 1-yr CORE2 CUDA, tier knobs ON, `m32_climate_compare.py --cref-frame` vs the certified C oracle at the M6/L79 floors.
- **A/B rule:** same-day, same-allocation, both legs in one job (`KNOBS` env on `job_m7_scale_gpu`); 35 steps, 2 reps, min; dt180; ±10% inter-allocation noise makes anything else meaningless.

## Ratio ledger

Baseline anchors are re-measured same-day (row 0), not inherited from `SCALING_M524.md`.
Binaries frozen at `/work/ab0995/a270088/port2/m7/bin/row0/` (md5 `02c8a0d1…` cuda / `267c9a6a…` serial)
so later jobs can be pinned to certified-source code while the build tree moves.

| after | NG5@4N GPU | NG5@4N CPU | **ratio** | NG5@8N GPU | NG5@8N CPU | **ratio** | NG5@16N GPU | NG5@16N CPU | **ratio** | dars@8N GPU | dars@8N CPU | **ratio** | dars@2N GPU |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| M5.24 (ref, 2026-05-31 — historical, NOT the anchor) | 1.273 | 4.599 | 3.61 | 0.810 | 2.356 | 2.91 | 0.492 | 1.237 | 2.51 | 0.344 | — | — | 0.814 |
| **row 0: m7 baseline** ✅ COMPLETE (2026-07-14, min of 2 reps) | **1.2796** | **4.6005** | **3.60** | **0.7381** | **2.3624** | **3.20** | **0.4487** | **1.2188** | **2.72** | **0.3178** | **0.8563** | **2.69** | **0.8177** |

**The gap to close: 3.60× → ≥5.0× at 4N (Stage 1); flatten the 2.72× at 16N (Stage 2).**

NG5@4N reproduces M5.24's 3.61× exactly, so the Δ-anchor is sound. But **8N and 16N both came in
BETTER than the historical numbers** (3.20× vs 2.91×; 2.72× vs 2.51×) — differences well outside
run-to-run noise. That is exactly why the same-day rule exists: **row 0, not `SCALING_M524.md`, is
what every lever is measured against.** The 16N legs are the same-day Stage-2 anchor the plan review
asked for. Harvest: `grep -h 'loop timing' /work/ab0995/a270088/port2/m7/base_*/log_rep_*.txt`.

**Where the levers stand (measured, not projected):**

| lever | class | A/B | status |
|---|---|--:|---|
| `ICEFLUXDEV` (Task 1.0) | bit-id | **−0.72%** (NG5@4N, job 26235600) | landed, gated. Real but small — **NOT the headline** |
| `NOFENCE2` (Task 1.1) | bit-id | queued | floor 1.1%, ceiling 4.1% from the stall budget |
| **`SWSKIP` (Task 1.2)** | **bit-id** | **queued (26236820)** | **the actual ~25% lever — dead host `sw_3d`** |

**PROJECTION for SWSKIP, the A/B is the arbiter:** if it recovers the ~25% host segment,

| | now | −25% | ratio | SYPD@dt240 |
|---|--:|--:|--:|--:|
| NG5@4N | 1.2796 | ~0.96 | **3.60× → ~4.8×** | — |
| NG5@16N | 0.4487 | ~0.34 | **2.72× → ~3.6×** | **1.42 → ~1.9** |

At 4N that would be essentially the whole Stage-1 target from one bit-identical lever; at 16N it
takes SYPD@dt240 from 1.42 to ~1.9, within reach of the ~2 SYPD the campaign was chartered to find,
in pure FP64 without the banned mixed precision. The host segment is 24.7% at dars@8N (the 16N-class
proxy) too, so the lever should pay at both scales. **Do not quote these until 26236820 lands** — the
last time I quoted a projection from this budget, I had the wrong function.

SYPD@dt240 = 0.657 / (s/step at dt180) × (1/1.03 CG correction) for NG5.

## Stall budget — NG5@4N (Task 0.3) ✅

`m7/nsys_ng5_4n`, rank 0, steady-state window = steps 11–34 (23 steps).
Tool: `scripts/m7_stall_budget.py` (nsys → sqlite → attribution).

**Validation of the method:** trace step time **1274.6 ms** vs the untraced measured baseline
**1279.6 ms** (0.4%) → nsys adds no measurable overhead, so every number below is a real cost of the
production step. Kernel share **46.6%** independently reproduces PROFILE_M522's "46%".

| component | ms/step | % of step |
|---|--:|--:|
| **GPU busy** | 694.4 | 54.5 |
| — of which kernels | 593.7 | 46.6 |
| — of which memcpy only (mostly MPI's own staging) | 100.6 | 7.9 |
| **GPU IDLE** | 580.2 | 45.5 |
| — **host segment** (no CUDA call, no MPI call) | **408.2** | **32.0** |
| — MPI wait (halo Waitall + CG Allreduce) | 106.3 | 8.3 |
| — launch gap (host in cudaLaunchKernel/Memcpy) | 38.1 | 3.0 |
| — **fence spin** (GPU already drained) | 18.1 | 1.4 |
| — other CUDA API | 9.6 | 0.8 |

**Where the host segment is** (attributed by the kernel each gap follows):

| host gap follows | ms/step | % step | gaps/step | what it is |
|---|--:|--:|--:|---|
| `fesom_ice_h_diag_kk` | **333.6** | **26.2** | 6 | the coupling-phase host code. **⚠️ This window contains BOTH `fesom_ice_oce_fluxes_mom` AND the host `fesom_cal_shortwave_rad` (`fesom_main.cpp:1214`). The A/B proved the shortwave is the cost (ICEFLUXDEV = −0.72%) → Task 1.2 `SWSKIP`.** |
| `resolve_bvfreq_dev` | 54.7 | 4.3 | 108 | I/O accumulator resolvers — ➕ follow-up (see below) |
| halo exchange (device2/device) | 14.9 | 1.2 | 5667 | per-exchange host overhead |
| everything else | 5.0 | 0.4 | — | |

✅ **CONFIRMED by a second, independent method** (`m7/stepprof_ng5_4n`, job 26235416 — the model's
own `FESOM_STEP_PROFILE` phase timer, which knows nothing about nsys):

> `STEP PROFILE (rank0, % of loop): forcing 7.6% (0.1007 s) · sea-ice 9.6% (0.1274) ·`
> **`coupling 24.6% (0.3270 s/step)`** · `ocean 54.5% (0.7239)`

| method | the coupling-phase host cost |
|---|--:|
| nsys host-gap attribution | 333.6 ms/step (26.2%) |
| `FESOM_STEP_PROFILE` phase timer | **327.0 ms/step (24.6%)** |

**They agree to within 2%.** (The profiled run is 1.3272 s/step vs the 1.2796 baseline — the phase
timer inserts its own fences, ~3.7% overhead — so its share is measured against a slightly larger
denominator.) The coupling phase contains essentially no GPU kernels, and the only per-step code in
it is the host `fesom_cal_shortwave_rad` + `fesom_ice_oce_fluxes_mom`. Payoff bracket for the pair:
**~25–26% of the step** — and the A/B has now shown the split is ~0.7% / ~25%, i.e. it is the
shortwave. The A/B remains the arbiter for the landed number.

➕ **The second host cost, root-caused.** The 54.7 ms/step is the I/O mean accumulators: six output
vars still have `nullptr` device accumulators in `fesom_default_monthly_table`
(`fesom_io.cpp:823`) — **`ssh`, `a_ice`, `m_ice`, `m_snow`, `uice`, `vice`** — so they fall back to
HOST resolvers (`resolve_uice` &c, `fesom_io.cpp:712`). Under `io.config.daily_monthly` each runs at
BOTH cadences: 12 host loops/step over ~1.86 M nodes/rank ≈ 22 M host iterations, which is the
54.7 ms. Fix = extend the M5.14 `resolve_*_dev` pattern to those six (mechanical, bit-identical:
per-element `out[i] += src[i]`, no reduction). ➕ Task 1.0b.

Uniform, not bursty: per-step host time = mean 464.7 ms, **stdev 7.0 ms (1.5%)** across 24 steady
steps. That rules out I/O flushes/forcing reads and confirms a per-step host loop.

### The second scale point — dars@8N (the 16N-class per-rank proxy) ✅

`m7/nsys_dars_8n`. Trace step **319.4 ms** vs measured baseline **318.0 ms** (0.4%) — the method
validates again. Kernel share **27.5%** independently reproduces PROFILE_M522's *"at 16N GPU-compute
falls to ~28%"*.

| component | NG5@4N | dars@8N (16N-class) |
|---|--:|--:|
| GPU kernels | 46.6% | **27.5%** |
| memcpy (mostly MPI staging) | 7.9% | 5.3% |
| **host segment** | **32.0%** | **24.7%** |
| MPI wait | 8.3% | **31.2%** |
| launch gap | 3.0% | 8.9% |
| fence spin | 1.4% | 1.8% |
| post-unpack fences /step | 432.5 | 385.7 |
| our fences /step | 996.3 | 910.7 |

**Reading it.** The regime shifts exactly as M5.22 said it would — MPI goes 8.3% → 31.2% and becomes
the wall — **but the host loop does NOT go away: it is still a quarter of the step.** So Task 1.0
pays at BOTH ends of the scaling curve, while MPI/imbalance (31%) is what Tier 3 (CG1R, CGPOLY,
EVPWIDE) has to attack for Stage 2. Fence spin stays ~1–2% everywhere: NOFENCE2 is real but small,
at any scale.

### Per-step sync counters

| | /step | spin (ms/step) |
|---|--:|--:|
| **OUR fences** (`Kokkos::fence` → `cudaDeviceSynchronize`) | **996.3** | 18.1 |
| — post-unpack halo fence (`:286/:377/:475`) — **Task 1.1 target** | 432.5 | **13.6** |
| — pre-MPI pack fence (`:251/:344/:439`) — **must stay** | 402.7 | 3.4 |
| — other (CG `parallel_reduce`, ice, EOS) | 161.1 | 1.1 |
| halo exchanges (dedup `MPI_Waitall`) | 391.1 | — |
| kernel launches | 1679 | — |
| MPI messages (Isend+Irecv) | 3911 | — |
| *MPI's OWN device syncs (`cudaStreamSynchronize`) — NOT ours* | *3636* | *(in the MPI bucket)* |

⚠️ **Two traps, both of which produced confident wrong answers on the first pass.** (1) CUPTI names the
runtime API with a version suffix (`cudaStreamSynchronize_v3020`), so exact-matching finds nothing and
reports a tidy "0 fences/step". (2) Not every device sync is ours: `cudaStreamSynchronize` (3636/step)
tracks the *message* count, not our fences — it is CUDA-aware MPI's internal per-message sync. Counting
it as a fence credits Task 1.1 with **6× the fences it can actually remove**. Both are documented in
`scripts/m7_stall_budget.py`.

## Roofline (Task 0.4)

**Register spill, from `cuobjdump --dump-resource-usage` on the row-0 binary** — free, no GPU, exact.
14 of 456 kernels carry a stack frame:

| kernel | stack B/thread | REG | note |
|---|--:|--:|---|
| `tke_column_loop<true>` | **37120** | 70 | ⚠️ TKE only (`FESOM_MIX_SCHEME=TKE`) — **off in the default path**, but catastrophic for anyone running TKE on GPU. ➕ new finding |
| `tke_column_loop<false>` | **23712** | 69 | idem |
| `fesom_impl_vert_visc_kk` | 7168 | 82 | ✅ Task 2.3 target (as predicted) |
| `fesom_fer_solve_gamma_kk` | 7168 | 43 | ✅ Task 2.3 target |
| `impl_vert_diff_tracers` (`fesom_tracer_diff.cpp`) | 6144 | 66 | ✅ Task 2.3 primary target |
| **`fesom_pressure_bv_kk`** (EOS) | **5120** | 62 | ➕ **new** — not in the plan, and it is in the top-10 |
| **`fesom_diff_ver_part_redi_expl_kk`** | **5120** | 58 | ➕ **new** — 3.3% of step |
| `fesom_pressure_force_…_shchepetkin_kk` | 2096 | 94 | |
| `fesom_momentum_adv_scalar_kk` (×2) | 2048 | 56/46 | |
| `fesom_tracer_advect_one_fct_kk` | 2048 | 40 | the #1 kernel |
| `fesom_diff_part_hor_redi_kk` | 2048 | 80 | |

The plan's Task-2.3 hypothesis (TDMA `real_t[128]` stack arrays → spill) is **confirmed with hard
data**, and it gains two targets the plan did not know about.

**ncu roofline ✅ COMPLETE** (`m7/ncu_top10`, job 26235287; CORE2 dist_1, np=1, steady-state launches;
A100 peak DRAM 1.94 TB/s; **ideal sectors/request for FP64 = 8.0**, 32.0 = fully scattered):

| kernel | GB/s | %peak | SM% | mem% | occ% | regs | stackB | sec/req | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `fesom_tracer_advect_one_fct_kk` (#1, 14.3%) | **1141** | **59.0** | 19.9 | 67.0 | 63.0 | 80 | 2048 | **7.3** | near the DRAM roofline, WELL coalesced |
| `diff_ver_part_impl_ale_kk` (TDMA) | 1123 | 58.0 | 3.2 | 58.0 | 39.6 | 66 | **6144** | **23.6** | spill + uncoalesced |
| `fesom_momentum_adv_scalar_kk` | 1092 | 56.4 | 17.4 | 64.6 | 63.6 | 56 | 2048 | 13.6 | spill |
| `fesom_visc_filt_bidiff_kk` | 1065 | 55.1 | 9.6 | 63.7 | 54.4 | 60 | 0 | 16.5 | uncoalesced |
| `fesom_impl_vert_visc_kk` (TDMA) | 979 | 50.6 | 9.0 | 56.9 | 27.2 | 82 | **7168** | **23.2** | spill + uncoalesced |
| `fesom_smooth_nod3D_kk` | 941 | 48.6 | **45.7** | 52.3 | **88.6** | 32 | 0 | **2.8** | **balanced** |
| `fesom_diff_part_hor_redi_kk` | 913 | 47.2 | 8.5 | 53.7 | 31.7 | 80 | 2048 | 13.1 | spill |
| `fesom_ale_vert_vel_linfs_kk` | 913 | 47.2 | 7.0 | 55.3 | 58.2 | 48 | 0 | 17.8 | uncoalesced |
| `fesom_diff_ver_part_redi_expl_kk` | 851 | 44.0 | 9.0 | 54.1 | 35.1 | 58 | **5120** | 12.4 | spill |
| `fesom_fer_solve_gamma_kk` (TDMA) | 837 | 43.3 | 3.2 | 48.1 | 49.2 | 43 | **7168** | **22.0** | spill + uncoalesced |
| `fesom_pressure_bv_kk` (EOS) | 665 | 34.3 | 8.6 | 39.8 | 44.7 | 62 | **5120** | **23.4** | spill + uncoalesced |
| `kpp_ri_iwmix_kk` | 600 | 31.0 | **2.8** | 40.0 | 53.6 | 41 | 0 | **23.6** | badly uncoalesced, latency-bound |

**What it says for Tier 2 — and it partly rewrites it:**
- **The #1 kernel is NOT a coalescing problem.** `tracer_advect_one_fct` already runs at **59% of
  DRAM peak with 7.3 sectors/request** — near-perfectly coalesced and close to the memory roofline.
  You cannot make it faster by fixing access patterns; the only lever is **moving less traffic** —
  which is exactly what **Task 1.3 FCT2** does (batch T+S so geometry/velocity/edge loads are read
  ONCE instead of twice). Its value is now measured, not assumed.
- **`smooth_nod3D` is the control group** — 2.8 sec/req, SM 45.7%, occupancy 88.6%, "balanced". That
  is the M5.18 coalescing lever's own kernel, and it proves the lever works and is *done*.
- **Every TDMA is spill-bound AND uncoalesced** (22–24 sec/req, SM 3–9%): Task 2.3 confirmed twice
  over, and they will benefit from the layout change as much as from the spill fix.
- ➕ **`kpp_ri_iwmix_kk` is a new target**: SM 2.8%, 23.6 sec/req — the worst coalescing in the set.

⚠️ **The ncu regex trap.** `PROFILE_M522` names kernels by their **Kokkos runtime label**
(`fct_eud_fill`, `gm_redi_ver_node`, `ale_vvel_scatter`) — the string passed to `parallel_for()`.
ncu and nsys never see that string; they see the **C++ symbol of the enclosing function**
(`fesom_tracer_advect_one_fct_kk`). The plan's label-based top-10 regex matched **2 of 10 kernels**,
and ncu still exited 0 with a near-empty report. The real top-10, measured from the trace:

| # | kernel (C++ symbol) | launches/step | ms/step | % step |
|--:|---|--:|--:|--:|
| 1 | `fesom_tracer_advect_one_fct_kk` | 56.0 | 182.6 | **14.3** |
| 2 | `fesom_ale_vert_vel_linfs_kk` | 4.0 | 44.9 | 3.5 |
| 3 | `fesom_diff_ver_part_redi_expl_kk` | 6.0 | 42.3 | 3.3 |
| 4 | `diff_ver_part_impl_ale_kk` | 2.0 | 32.6 | 2.6 |
| 5 | `fesom_momentum_adv_scalar_kk` | 4.0 | 25.2 | 2.0 |
| 6 | `fesom_diff_part_hor_redi_kk` | 6.0 | 23.7 | 1.9 |
| 7 | `fesom_visc_filt_bidiff_kk` | 3.0 | 19.6 | 1.5 |
| 8 | `fesom_smooth_nod3D_kk` | 8.0 | 17.7 | 1.4 |
| 9 | `kpp_ri_iwmix_kk` | 2.0 | 17.0 | 1.3 |
| 10 | `fesom_impl_vert_visc_kk` | 1.0 | 13.3 | 1.1 |
| 12 | `fesom_ssh_solve_cg_kk` | **351.3** | 12.0 | 0.9 |

The CG is **351 launches/step for 12 ms of GPU time** — ~34 µs/kernel. It is *launch-latency* bound,
not compute bound: exactly what CGSLIM/CG1R attack, and it argues for kernel-count reduction over
kernel optimisation.

## Decision note — Tier-1A re-ranked by measured share (Task 0.3)

The plan ranked Tier-1A as *fence removal → CG slimming → small-kernel fusion*. The data re-ranks it:

| rank | lever | measured payoff at NG5@4N | class | status |
|--:|---|---|---|---|
| **1** | ➕ **Task 1.2 `SWSKIP`** — skip the DEAD host `sw_3d` (`fesom_bulk.cpp:698`) | **the ~25% host segment** | bit-id | **THE lever. All correctness gates PASS; A/B running** |
| ~~1~~ | ~~Task 1.0 `ICEFLUXDEV`~~ — port `ice_oce_fluxes_mom` to Kokkos | **A/B: −0.72%** | bit-id | landed + gated, but **my original attribution was WRONG** — see the headline |
| 2 | Task 1.3 `FCT2` — T+S tracer batching | FCT is 14.3% of step in 56 launches/step; batching halves the launches AND the host issue cost | bit-id | as planned, now better motivated |
| 3 | Task 1.1 `NOFENCE2` — post-unpack fence | **floor 13.6 ms (1.1%)**, ceiling 51.7 ms (4.1%) incl. the launch gap it unblocks | bit-id | ✅ implemented — modest, but cheap and bit-identical |
| 4 | Task 1.2 `CGSLIM` — CG body | 351 launches/step for 12 ms GPU → pure launch overhead | bit-id | as planned |
| 5 | ➕ `resolve_bvfreq_dev` I/O resolvers | 54.7 ms/step (4.3%) | tbd | ➕ follow-up |

**Honest caveat on ranks 3–4:** "fence spin" and "launch gap" are entangled. A fence's true cost is not
only the time the host spins after the GPU drains — it also stops the host running ahead, so the launch
queue empties and the GPU starves at the next kernel. Removing a fence recovers its spin **plus** some
share of the 38.1 ms launch gap. Spin alone (13.6 ms) is the **floor**; spin+gap (51.7 ms) the **ceiling**.

## Knob registry

| knob | lever | class | status |
|---|---|---|---|
| `FESOM_SPEED` | master switch (all blessed levers) | — | scaffolded |
| `FESOM_SPEED_FORCE_SERIAL` | dev-only: allow levers on Serial (byte proofs) | — | scaffolded |
| `FESOM_SPEED_SYNCSTATS` | per-step sync/fence counters (diagnostic) | — | ✅ implemented (0.3/1.1) |
| `FESOM_SPEED_NOFENCE2` | drop post-unpack halo fence | bit-id | ✅ implemented, gates pending (1.1) |
| `FESOM_SPEED_ICEFLUXDEV` | port `ice_oce_fluxes_mom` to device | bit-id | ✅ landed, all gates PASS. **A/B −0.72%** — real but small |
| **`FESOM_SPEED_SWSKIP`** | **skip the DEAD host `sw_3d` (the device twin rebuilds it in full)** | **bit-id** | **implemented (1.2) — THE ~25% lever; gates + A/B queued** |
| `FESOM_SPEED_CGSLIM` | CG iteration-body slimming | bit-id | pending (1.2) |
| `FESOM_SPEED_FCT2` | FCT T+S tracer batching | bit-id | pending (1.3) |
| `FESOM_SPEED_EVPCOMPACT` | EVP active-set compaction | bit-id | pending (1.4) |
| `FESOM_SPEED_SCATTER` | scatter de-atomization (1=coloring, 2=store+gather) | rounding | pending (2.1) |
| `FESOM_SPEED_TDMA` | TDMA spill-kill (1=recompute-in-sweep, 2=PCR) | bit-id / rounding | pending (2.3) |
| `FESOM_SPEED_CG1R` | CG single-Allreduce (Chronopoulos–Gear); **supersedes CGSLIM when both set** | solver | pending (3.1) |
| `FESOM_SPEED_CGPOLY` | Chebyshev polynomial preconditioner (=degree) | solver | pending (3.2) |
| `FESOM_SPEED_EVPWIDE` | comm-avoiding wide-halo EVP (=k rings) | solver | pending (3.3) |
| `FESOM_SPEED_ICELAG` | lagged ice–ocean coupling | physics | reserve (4.1) |
| `FESOM_SPEED_EVPTHIN` | EVP halo thinning (stale ring) | physics | reserve (4.2) |

## Lever log

### Task 1.1 — `FESOM_SPEED_NOFENCE2` (post-unpack halo fence) — implemented, gates pending

Drops the `Kokkos::fence()` after the halo unpack in all three exchange paths
(`fesom_halo_device.cpp` `:286` + the `device2`/`deviceN` twins). The pre-MPI pack fence stays.

Audit (in the code, `fesom_halo_device.cpp`, above `halo_fence_post_unpack`):
1. **Consumers** — all on the default execution space = one CUDA stream → stream order already
   serialises the unpack before every device reader. The fence only ordered device work vs a *host* reader.
2. **Host readers** — none mid-step. Every host read goes through `Field::sync_host()` →
   `DualView::sync_host()` → `Kokkos::deep_copy`, which fences; a raw stale-host read is trapped by
   `-DFESOM_KK_SYNCCHECK`. `src/` contains no `cudaMemcpyAsync`.
3. **Buffer reuse (the one that bites)** — a LATER exchange's `MPI_Irecv` writes the same `recv_d` this
   unpack is reading, and MPI's device writes are not ordered against the Kokkos stream. Safe **because
   the pre-MPI fence is unconditional**: it sits outside the `if (send_count > 0)` guard in all three
   paths, so every `MPI_Irecv` into `recv_d` is preceded, in its own function, by a full device fence
   that drains the previous unpack. **INVARIANT to preserve:** every `MPI_Irecv` on a device buffer is
   preceded by an unconditional `Kokkos::fence()` in the same function.
4. **Realloc (a hazard the plan did not name)** — `grow()` can free `recv_d` while a previous unpack is
   still reading it. Today that is saved only by allocator behaviour, and the nsys trace shows this build
   uses the **async pool** (`cudaMallocAsync`/`cudaFreeAsync` both appear), whose free is merely
   stream-ordered — the "true today, silent tomorrow" reasoning L67 warns about. Now fenced **explicitly**
   on the realloc branch only (buffers hit their high-water mark in warmup: 379 reallocs in a whole
   35-step NG5 run), so it costs nothing in steady state.

⚠️ Premise 1 is FALSE the instant anything runs on a second stream — **Task 4.1 (ICELAG) does exactly
that: re-audit + re-racecheck before landing it.**

Expected payoff (measured): floor **1.1%**, ceiling **4.1%** of the NG5@4N step.
Gates still to run: racecheck, knob-OFF byte gate, FORCE_SERIAL byte proof, CUDA fidelity gate, A/B.

## Tier-1 gate results (2026-07-14)

| # | gate | knobs | verdict |
|---|---|---|---|
| 26235595 | knob-OFF Serial byte gate (the campaign invariant) | none | **PASS** — `diff_snap` rc=0 |
| 26235596 | **FORCE_SERIAL byte proof** | `FORCE_SERIAL=1 ICEFLUXDEV=1` | **PASS — the ICEFLUXDEV bit-identity claim is PROVEN.** The levered kernels run on the Serial backend and reproduce the certified baseline byte-for-byte: pure re-execution, not merely "close". |
| 26235597 | CUDA fidelity gate | `ICEFLUXDEV=1` | **PASS** (deltas in the same band as the un-levered baseline, as a bit-identical lever requires) |
| 26235598 | CUDA fidelity gate | `NOFENCE2=1` | **PASS** |
| 26235599 | CUDA fidelity gate | both | **PASS** |
| 26235643 | **compute-sanitizer memcheck**, baseline-differential | `NOFENCE2=1` | **CLEAN** — 4 errors knob-OFF, 4 knob-ON (identical). All 4 are the benign `CUDA_ERROR_INVALID_CONTEXT` on `cuCtxGetDevice` (UCX/CUDA-aware-MPI probing the context from a non-CUDA thread under the sanitizer). **ZERO Invalid read / Invalid write / use-after-free** — precisely what the `grow()` hazard would have produced had the explicit realloc fence been missing. |
| 26235644 | memcheck, baseline-differential | both | **CLEAN** (same result) |
| 26235645 | racecheck (completeness only — see below) | both | queued |
| 26235600/1/2 | same-alloc A/B (NG5@4N ×2, dars@8N) | — | queued — **the payoff numbers** |

⚠️ **A racecheck lesson, learned the hard way.** The plan said "racecheck the fence removal". The
first attempt (job 26235606) died on an invalid flag and its non-zero exit *looked* exactly like
"hazards found". Two things were wrong: (1) **racecheck is "Shared memory hazard checking"** — it
sees `__shared__` only, so it structurally CANNOT prove or disprove NOFENCE2, whose entire risk
surface is GLOBAL memory ordering. **memcheck** is the tool that can catch the one real hazard
(`grow()` freeing `recv_d` under a running unpack = use-after-free). (2) **No baseline = no
verdict**: Kokkos' reductions use shared memory and the model is full of intentional atomics, so a
bare "N hazards" means nothing. `jobs/job_m7_sanitize` now runs each tool knobs-OFF *then* knobs-ON
and only attributes the difference.

## Tier climate gates

| tier | knobs ON | 1-yr climate vs C oracle | NG5@16N direct | tag |
|---|---|---|---|---|
| 0 | none | n/a — baseline CUDA fidelity gate **PASS** (all 27 fields at the climate-close floor, worst 9.9e-03 `h_ice`; job 26235125) | CPU 1.2188 ✅ / GPU queued | `m7.0-baseline` ✅ `3d00123` |
| 1 | ICEFLUXDEV + NOFENCE2 | **pending** (needs the 1-yr CORE2 run) | pending | `m7.1-bitid` (pending) |
