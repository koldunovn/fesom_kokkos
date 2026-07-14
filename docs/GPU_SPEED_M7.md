# M7 — speed beyond bit-identity (pure FP64): campaign log

*Plan: `docs/plans/20260714-m7-speed-fp64.md`. Branch `m7-speed` off `main@69e506d`. Goal: GPU-node/CPU-node ratio ≥5× (stretch 8×) at NG5/dars 4–8N (Stage 1), flatten the decay toward 16N (Stage 2), pure FP64, no repartitioning. Bit-identity replaced by the two-level gate (per-lever fidelity+A/B; per-tier 1-yr climate; knob-OFF byte-identical always).*

---

## 🔴 THE TASK-0.3 HEADLINE: the "~25–30% remainder" is HOST WORK — and it is DEAD host work

**32% of the NG5@4N step is single-threaded host C++.** The plan's Tier-1A was built on fences and
launch gaps; measured, those are **1.4% and 3.0%**. The fence *count* was right (996/step vs the
~880 estimate) — the fences are just cheap.

| the guess (plan Tier-1A) | the measurement (NG5@4N, rank 0, 23-step steady window) |
|---|---|
| ~880 device fences/step, "blocking-sync stalls" | 996 fences/step ✅ — but their **spin is only 18.1 ms/step (1.4%)** |
| "launch gaps" | **38.1 ms/step (3.0%)** |
| "host" | **408.2 ms/step (32.0%)** ← the whole remainder is here |

**~25 points of it are ONE function — and I first attributed it to the WRONG one. Read this before
trusting any attribution in this doc.**

The nsys host-gap is bounded precisely: it starts after the last sea-ice kernel and ends exactly at
`fesom_cal_shortwave_rad_kk`. I read `fesom_main.cpp:1176–1202`, saw the chlorophyll block was not
per-step, concluded the only remaining per-step code in the window was `fesom_ice_oce_fluxes_mom`
(`fesom_ice_coupling.cpp:512`), and stopped reading. **The answer was 12 lines further down, at
`fesom_main.cpp:1214`.** The A/B settled it: porting `ice_oce_fluxes_mom` to the device
(`ICEFLUXDEV`) is bit-identical, gated, and worth **−0.72%** — not 25%.

**The real culprit: `fesom_cal_shortwave_rad` (host, `fesom_bulk.cpp:698`) is DEAD WORK.**

```c
fesom_main.cpp:1214  fesom_cal_shortwave_rad(&mesh, &jra, &ice, &forcing);     // host
fesom_main.cpp:1215  fesom_cal_shortwave_rad_kk(&mesh, &jra, &ice, &forcing);  // device (M5.20)
```

Both are called, back-to-back, every step. M5.20 moved the `sw_3d` shortwave-penetration profile to
the device and removed its 519 MB/step HtoD push — **but left the host computation running.** The
device twin *starts by zeroing the whole array* (`fesom_sw3d_zero`, `fesom_bulk.cpp:784`) and then
rewrites every entry, so everything the host wrote is overwritten microseconds later, on **both**
backends (on Serial, `sw_3d_fld.d()` *is* that same host array). The host function's only unique
output is the cheap nod2D `heat_flux += swsurf` side effect — which the device kernel deliberately
does not do (`fesom_bulk.cpp:794`).

The dead half is not cheap, and it is worth being precise about WHY (measured from the mesh:
NG5 nod2D = 7,402,886, nl = 70, dist_16 → ~466 k nodes/rank):
- the nod3D `memset`: 466,324 × 70 × 8 B = **261 MB per rank per step** (1.04 GB per 4-GPU node) —
  single-threaded, so only ~30–50 ms of it;
- **the real cost — the `exp()` column walk**: ~373 k open-ocean nodes × ~12 levels × 2 `exp()`
  ≈ **9 M `exp()` calls per rank per step**. At 20–50 ns per double-precision `exp`, that alone is
  200–450 ms — which is where the measured ~333 ms actually comes from.

→ **Task 1.2 `FESOM_SPEED_SWSKIP`**: skip the `sw_3d` half of the host function, keep `heat_flux`.
Bit-identical by construction (the device twin recomputes `sw_3d` in full from the same inputs with
the same arithmetic) and provable with the FORCE_SERIAL byte proof. ⚠️ The one host reader of
`sw_3d` is `kpp_bldepth` (`fesom_kpp.cpp:474`), which is `static` and runs **only** under
`-DFESOM_KK_VERIFY` (`fesom_step.cpp:215`) — do not combine the knob with VERIFY.

**Why it still caps the ratio.** Host code runs on **4 threads on a GPU node** (4 ranks = 4 GPUs)
but on **128 ranks on a CPU node** — a ~32× parallelism deficit the GPU pays and the CPU does not.
That reasoning was right; only the function was wrong.

**The lesson (see `feedback-profiler-traps` in project memory):** the profiler told me *where* the
time was, to the microsecond, and it was right. It cannot tell you *which source line* is in that
window — for that you must read **all** the code between the bracketing kernels, not the first
plausible candidate. Two agreeing measurements of the same *window* are not independent confirmation
of an *attribution*. **The same-alloc A/B is what turns an attribution into a fact.**

✅ **The class is closed.** A device port that leaves the host original running is invisible to every
correctness gate and costs full price, so I swept every `foo()` / `foo_kk()` pair for others.
Comment-stripped, restricted to the per-step loop (`fesom_main.cpp:1072+` and `fesom_step.cpp`):
**`fesom_cal_shortwave_rad` is the ONLY one.** Every other host twin (`bulk_compute`, `pressure_bv`,
`compute_vel_nodes`, `mo_convect`, `pp_mixing`, `ssh_solve_cg`, `impl_vert_visc`, `update_vel`,
`compute_hbar`, `ale_thickness_linfs`, `compute_ssh_rhs_linfs`, `pressure_force_linfs_fullcell`)
is called only from the **initialisation / IC-seeding** section (lines 553–905), which runs once.

### Gate results — Task 1.2 `SWSKIP`

| # | gate | verdict |
|---|---|---|
| 26236816 | knob-OFF Serial byte gate | **PASS** |
| 26236817 | **FORCE_SERIAL byte proof** (`SWSKIP=1`) | **PASS — bit-identity PROVEN** |
| 26236818 | CUDA fidelity gate (`SWSKIP=1`) | **PASS** |
| 26236819 | CUDA fidelity gate (all three knobs) | running |
| 26236820 | **same-alloc A/B, NG5@4N** | **running — THE payoff number** |
| 26236821/2 | A/B all-three (NG5@4N, dars@8N) | running |

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
| **1** | ➕ **Task 1.0 `ICEFLUXDEV`** — port `fesom_ice_oce_fluxes_mom` to Kokkos | **up to 333 ms/step (26.2%)** | bit-id | **NEW — the campaign's biggest single lever** |
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
