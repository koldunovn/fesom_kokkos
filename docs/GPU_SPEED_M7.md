# M7 — speed beyond bit-identity (pure FP64): campaign log

*Plan: `docs/plans/20260714-m7-speed-fp64.md`. Branch `m7-speed` off `main@69e506d`. Goal: GPU-node/CPU-node ratio ≥5× (stretch 8×) at NG5/dars 4–8N (Stage 1), flatten the decay toward 16N (Stage 2), pure FP64, no repartitioning. Bit-identity replaced by the two-level gate (per-lever fidelity+A/B; per-tier 1-yr climate; knob-OFF byte-identical always).*

---

## 🔴🔴 THE PROTOCOL CHANGED (2026-07-15). READ THIS BEFORE QUOTING ANY RATIO BELOW.

**Every ratio row further down was measured on the 35-STEP protocol, and the 35-step protocol was
CONTAMINATED.** `getcoeffld` rebuilt the JRA interpolation coefficients **8× per step, every step**, for
the first 30–60 steps of every run (the model starts before the first JRA record; the "no extrapolation
back in time" clamp never released). **Every benchmark in this campaign was 35 steps.** Fixed
bit-identically in `7f64be1`.

**The artifact is ASYMMETRIC** — `getcoeffld` is host code over `myDim_nod2D`, and the GPU config carries
**463 k nodes/rank** against the CPU's **14.5 k** (4 ranks/node vs 128). It cost the GPU **5.7 %** of its
step and the CPU **0.03 %** of its. **So it under-reported the ratio**, and the older rows are all low.

**⇒ THE PROTOCOL IS NOW 300 STEPS, and it is clean.** Proven, not assumed: two matched nsys traces show
the 300-step run's *reported* loop timing sits within ~3 ms of its own steady state, and the per-step
series settles by ~step 30 (the residual is the CG spin-up, 86 → 72 iters — physics, not a bug).

> **🔴 The "~22 ms of unattributed cold start" that earlier handoffs chased DOES NOT EXIST.** It was
> **model error**: a *measured* 72.6 ms delta minus a *modelled* `getcoeffld` cost, with the remainder
> given a physical story. Two gap censuses show the 25-step and 300-step GPU-idle budgets are
> **identical** (93.6 vs 94.1 ms/step). **RETRACTED.** → L88.

### ⭐⭐ THE RATIO, RE-MEASURED ON MATCHED PAIRS (2026-07-15) — these are the only honest rows

| after | NG5@4N GPU | NG5@4N CPU | **ratio** | binaries | jobs |
|---|--:|--:|--:|---|---|
| through the `getcoeffld` fix | 0.7239 | 4.5785 | **6.32×** | `h5` CUDA `0d39d8a2` / Serial `950ee0f9` | 26255936 / 26256684 |
| + H.3 `BULKTAIL` | 0.7058 | 4.5785 | 6.49× | `h8` CUDA `7dab6c5a` / Serial `ef6bdec4` | 26257716 / 26256684 |
| + H.7 `SMOOTHSCRATCH` (session 7) | 0.6739 | 4.5785 | 6.79× | `h9` CUDA `9e1f514b` / Serial `91eeb573` | 26258582 / 26256684 |
| + H.8 `LAZYSNAP` (session 7) | 0.6666 | 4.5785 | 6.87× | `h10` CUDA `13dbddb4` / Serial `7c75afc0` | 26260292 / 26256684 |
| **⭐ + H.9 `SSHRAILS`** (session 8) | **0.6503** | **4.5785** | **⭐ 7.04×** | **`h11` CUDA `d74d31b4`** / Serial `c125b424` | **26265348** / 26256684 |

**The h9 row is a CONFIRMATION, not just an anchor** (job 26258582, one allocation, a100_80): its
base leg re-ran h8-equivalent (`SMOOTHSCRATCH=0`) and reproduced the 0.7058 anchor at **0.7052**
(−0.09 %), and its scratch leg hit **0.6739 twice with 0.00 % spread**. Δ = −4.44 % (pre-registered
−4.2 % — the fifth census-sized lever to beat its pre-registration, L93).

### ⭐ THE CLEAN STANDARD SET — 300 steps, pinned, `-C a100_80`, min of 2 (session 7)

The re-measure L94 demanded (the old 16N rows ran on mixed hardware). GPU = `h9` `9e1f514b`
except the 4N row (h10 `13dbddb4`); CPU = `h9` Serial `91eeb573` (the CPU column is
lever-independent). *dars@8N is the **150-step** protocol — the 300-step window walks into the
known cold-start blowup at step 204 (deterministic, both reps; L95). Every other row is 300 steps.*

| | GPU s/step | CPU s/step | **ratio** | jobs (GPU/CPU) |
|---|--:|--:|--:|---|
| **⭐ NG5@4N (h16)** | **0.6467** | 4.5785 | **⭐ 7.08×** | 26280027 / 26256684 |
| NG5@4N (h14) | 0.6495 | 4.5785 | 7.05× | 26271441 / 26256684 |
| NG5@4N (h11) | 0.6503 | 4.5785 | 7.04× | 26265348 / 26256684 |
| NG5@4N (h10) | 0.6666 | 4.5785 | 6.87× | 26260292 / 26256684 |
| NG5@4N (h9) | 0.6739 | 4.5785 | 6.79× | 26258582 / 26256684 |
| **NG5@8N (h11)** | **0.4022** | **2.3530** | **5.85×** | 26267148 / 26258754 |
| NG5@8N (h9) | 0.4143 | 2.3530 | 5.68× | 26258752 / 26258754 |
| **NG5@16N (h14)** | **0.2629** | 1.2267 | **4.67×** | 26274345 / 26258753 |
| NG5@16N (h11) | 0.2629 | 1.2267 | 4.67× | 26267149 / 26258753 |
| NG5@16N (h9) | 0.2688 | 1.2267 | 4.56× | 26258751 / 26258753 |
| **dars@8N (h11**, 150-step) | **0.1981** | **0.8464** | **4.27×** | 26267150 / 26259246 |
| dars@8N (h9, 150-step) | 0.2041 | 0.8464 | 4.15× | 26259245 / 26259246 |

*(Session 10: h16 = h14 + FERNOINIT/VISCNOINIT promoted (C.2b/C.3a strict reductions, ladder
9/9, A/B −0.46 % RANGE HIT, ncu to the digit) — anchor pre-reg 0.6465 ±0.5 % → 0.6467 HIT.
NG5@16N h11 refresh 0.2629 = 0.8 % better than pre-reg (third at-scale under-run; H.9 holds
above the 60 % model) ⇒ **Stage-2 SYPD@dt240 = 2.43** (0.657/0.2629/1.03; 2.45 at ×1.019).
Full session-10 record: `docs/plans/20260718-m7-session10-FINDINGS.md`.)*

## ⭐ THE M7 DIVIDEND BY MESH × NODES (session 10, the user-requested cross-mesh survey)

**What the whole M7 stack bought, per mesh × node count**: same-day pinned GPU pairs, row0
`02c8a0d1` (campaign start, knobless) vs h14 `18275c68` (`FESOM_SPEED=1`), both `-C a100_80`,
min of 2, md5+announce-audited per leg. SYPD at each row's own benchmark dt (no production-dt
correction; the dt120 row is NOT SYPD-comparable to dt180 rows). Job ids + per-point notes in
the session-10 findings §3.1.

| point | verts/rank | protocol | row0 → h14 s/step | **Δ** | SYPD row0 → h14 |
|---|--:|---|---|--:|--:|
| dars@2N | 395k | 150 st, **dt120** (dt180 NaNs at step ~10 — partition-marginal cold start, findings §3.2) | 0.7771 → 0.3926 | **−49.5 %** | 0.42 → 0.84 |
| NG5@4N | 462k | 300 st, dt180 | 1.2299 → 0.6497 | **−47.2 %** | 0.40 → 0.76 |
| dars@4N | 198k | 150 st, dt180 | 0.4559 → 0.2541 | **−44.3 %** | 1.08 → 1.94 |
| NG5@8N | 231k | 300 st, dt180 | 0.7085 → 0.4025 | **−43.2 %** | 0.70 → 1.22 |
| farc@2N | 80k | 300 st, dt180 | 0.1678 → 0.1001 | **−40.3 %** | 2.94 → 4.92 |
| dars@8N | 99k | 150 st, dt180 | 0.3017 → 0.1985 | **−34.2 %** | 1.63 → 2.48 |
| farc@4N | 40k | 300 st, dt180 | 0.1260 → 0.0864 | **−31.4 %** | 3.91 → 5.70 |
| core2@1N | 32k | 300 st, dt1800, /pool | 0.1087 → 0.0754 | **−30.6 %** | 45.3 → 65.4 |
| core2@2N | 16k | 300 st, dt1800, /pool | 0.0922 → 0.0705 | **−23.5 %** | 53.4 → 69.9 |
| farc@8N | 20k | 300 st, dt180 | 0.1087 → 0.0861 | **−20.8 %** | 4.53 → 5.72 |
| NG5@16N | 116k | 300 st, dt180 | 0.4267 → 0.2629 | **−38.4 %** | 1.15 → 1.87 |

**The regime read:** the dividend GROWS monotonically with per-rank workload within every
mesh (ordering perfect on all four columns) and the stack ~halves the step wherever per-rank
domains are large. Both NG5 points landed IN their pre-registered bands (the model's
calibration mesh); EVERY off-NG5 point with a prior beat its band — the host-class levers
retain far more value on small/mid meshes than the L84(b)-style retention models assumed.
Practical headline: core2 on ONE GPU node does 65 SYPD; dars@8N clears the 2-SYPD line.

*(Session 9: h11 std-set refresh — 8N and dars both ~0.7 % better than pre-registered; the h11
16N leg (26267149) was still queued at write-time and supersedes the h9 row when it lands.
h14 = h11 + TDMANOINIT (−0.30 % A/B 26269642), anchor pre-reg 0.6483 ±0.5 % → 0.6495 HIT.)*

*(Tier-1, for the decay shape: 5.03× / 4.28× / 3.55×-mixed-hw / 3.27×. The 4N→8N decay is now
6.87→5.68 (−17 %), vs Tier-1's 5.03→4.28 (−15 %) — the host levers hold their share at 8N.)*

**16N scored against the pre-registration (4.5–5.0×; below 4.0 falsifies L84(b)): 4.56× — IN
RANGE, at the low end. L84(b) SURVIVES: the host-class packH levers carried to 16N.**

### ⭐⭐ STAGE-2 IS MET, COMFORTABLY: NG5@16N SYPD@dt240 = 2.37 (pessimistic ×1.03; 2.40 with the 4N-measured ×1.019)

`0.657 / 0.2688 / 1.03 = 2.37` — on clean hardware (pure a100_80), the clean 300-step protocol, a
pinned certified binary (h9), min of 2 reps. The last read was "1.99, right AT the line" on mixed
hardware and the contaminated protocol. The 2-SYPD goal the campaign was chartered for is now past
with an 18 % margin, in pure FP64, with mixed precision still banned and unspent. GPU strong-scaling
efficiency 4N→16N: 62.7 % (CPU: 93.3 %) — the 8× stretch at 4N and the 16N flattening both route
through packages B/C (+E for the halo self-gaps) per the 26248860 ladder verdict.

All legs **300 steps**, min of 2 reps, same day, **all pinned with `BIN=`**. Rep spreads: GPU 0.07 %,
CPU 0.20 %. *(The CPU column is unchanged because every `FESOM_SPEED_*` lever is CUDA-only — the knobs
resolve OFF on a Serial build. `h5` and `h8` Serial are the same model.)*

**`BIN=` pinning proved itself here:** the two `h5` anchors recorded **different git HEADs** at submit time
and still ran the *identical* frozen binaries. **`BIN=` is what runs; the build tree is irrelevant.**

**BULKTAIL's own cross-check — it removes a FIXED cost, and the numbers say so:**

| protocol | baseline | with BULKTAIL | Δ | Δ % |
|---|--:|--:|--:|--:|
| 35-step A/B (26256973) | 0.7437 | 0.7256 | **−18.1 ms** | −2.43 % |
| 300-step anchors | 0.7239 | 0.7058 | **−18.1 ms** | −2.50 % |

**Identical in absolute terms at both protocols** — exactly what a lever that deletes a fixed block of
host work (7 rails + a dead host loop) must do, and *not* what a cold-start artifact would do. Only the
percentage moves, because the denominator shrank. The pre-registration for the 300-step anchor was
**~0.706 s/step**; it came in at **0.7058**.

> **A note on the derivation I refused to make.** I explicitly declined to compute this ratio by
> multiplying the 35-step −2.43 % into the 300-step anchor, and said "measure it." Had I derived it, I
> would have got 0.7063 — **within 0.07 % of the measured 0.7058.** The model was right.
> **That is not a reason to have skipped the measurement; it is what the measurement is FOR.** You do not
> get to know a model is right until you check it, and the check cost four minutes of backfilled GPU time.
> *(The last time this campaign trusted an arithmetic ratio instead of measuring one, it had to be
> retracted — see below.)*

**Superseded, kept so they cannot quietly return:** 5.84× (35-step, contaminated) · 6.17× (35-step,
post-fix) · and a retracted "5.83× / 6.2×" that **mixed a long-run GPU number with a 35-step CPU anchor**.
**You cannot mix protocols and get an honest ratio.**

**CG correction for SYPD — RE-DERIVED (session 7).** The formula is
`correction = 1 + CG_share × (iters@240/iters@180 − 1)`. Measured on the h9 300-step trace
(26258712, steady window, `scripts/m7_cg_share.py`): the **CG region is 43.9 ms/step = 6.5 %** of
the 678.1 ms step at NG5@4N (kernel-busy alone 10.1 ms — the region is Allreduce/halo-dominated,
and all of it scales with iters); settled **iters@dt180 = 71.9** (last-50 mean, reproduced
independently from the fresh 16N CPU log). iters@dt240 is NOT measurable from a cold start; the
original 115/89 = 1.29 ratio is carried over as a stated assumption.
⇒ **correction = 1 + 0.065 × 0.29 = ×1.019 at NG5@4N** (the old ×1.03 was calibrated on the
cold-start 90-iter CG — pessimistic, as suspected). ⚠️ At **16N** the CG share is unmeasured on h9
(comm-bound, plausibly larger): quote 16N SYPD with **×1.03 as the pessimistic bound** and note
×1.02 is the 4N-measured value (~1 % of SYPD between them).

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

`SWSKIP` alone at NG5@4N: **1.2788 → 0.9403 s/step**, i.e. ratio **3.60× → 4.89×**. See the
all-four numbers below for the landed Tier-1 result.

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

> ## 🔴🔴 AND THE **16N** COLUMNS ARE CONTAMINATED A SECOND WAY: **MIXED GPU HARDWARE** (L94).
> The Levante `gpu` partition is **heterogeneous** — `l40xxx` = **a100_40** (HBM2, 1555 GB/s),
> `l50xxx` = **a100_80** (HBM2e, 1935 GB/s, ~25 % more). FESOM's step is 74.6 % memory-bound kernels and
> **the slowest MPI rank sets the pace**, so ONE a100_40 node drags the whole job by a **measured +3.4 %**.
>
> | job | row | nodes | verdict |
> |---|---|---|---|
> | **26235123** | **row-0 NG5@16N** (0.4487) | **2× a100_40** of 16 | ❌ **CONTAMINATED** |
> | **26238086** | **⭐ TIER-1 NG5@16N** (0.3432) | **1× a100_40** of 16 | ❌ **CONTAMINATED** |
> | 26238084 / 26238085 | Tier-1 4N / 8N | pure a100_80 | ✅ clean |
>
> **⇒ Both 16N ratios are UNDERSTATED by ~3 %** (row-0 2.72 → ~2.81; Tier-1 **3.55 → ~3.67**).
> **This does NOT explain the 5.03× → 3.55× decay toward 16N (a 29 % drop) — L84 SURVIVES.** It is a
> correction, not a story change. **But it lands exactly on the Stage-2 claim:** the docs say
> *"SYPD@dt240 = 1.99 at 16N … right AT the line, not comfortably past it."* A +3.4 % correction puts
> that at **≈2.06 — Stage-2 may ALREADY BE MET.**
> 🔴 **RE-MEASURE the 16N rows with `-C a100_80`. Do not adjust them.**
> *(An **A/B is IMMUNE** — all legs share the nodes, so the hardware cancels. Only ABSOLUTE anchors are
> hit. That includes job **26248860**, the pending 16N ladder test: it is a 4-leg A/B, so **its
> percentages are valid even if it lands on mixed hardware.**)*

> ## 🔴 EVERY ROW IN THIS TABLE IS ON THE CONTAMINATED 35-STEP PROTOCOL — SEE THE TOP OF THIS FILE.
> They are kept for provenance (the *marginal* A/B percentages in them are still sound — the
> `getcoeffld` artifact sits in BOTH legs and only inflates the denominator, understating each lever by
> ~5 % of itself). **But the RATIO column is systematically LOW and must not be quoted.**
> **The live ratio is the matched 300-step pair at the top of this file: `6.32×` at NG5@4N.**
> The standard set (NG5@4N/8N/16N + dars@8N, GPU **and** CPU) still needs re-running on the 300-step
> protocol against the current best binary.

Baseline anchors are re-measured same-day (row 0), not inherited from `SCALING_M524.md`.
Binaries frozen at `/work/ab0995/a270088/port2/m7/bin/row0/` (md5 `02c8a0d1…` cuda / `267c9a6a…` serial)
so later jobs can be pinned to certified-source code while the build tree moves.

| after | NG5@4N GPU | NG5@4N CPU | **ratio** | NG5@8N GPU | NG5@8N CPU | **ratio** | NG5@16N GPU | NG5@16N CPU | **ratio** | dars@8N GPU | dars@8N CPU | **ratio** | dars@2N GPU |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| M5.24 (ref, 2026-05-31 — historical, NOT the anchor) | 1.273 | 4.599 | 3.61 | 0.810 | 2.356 | 2.91 | 0.492 | 1.237 | 2.51 | 0.344 | — | — | 0.814 |
| **row 0: m7 baseline** ✅ COMPLETE (2026-07-14, min of 2 reps) | **1.2796** | **4.6005** | **3.60** | **0.7381** | **2.3624** | **3.20** | **0.4487** | **1.2188** | **2.72** | **0.3178** | **0.8563** | **2.69** | **0.8177** |
| **⭐ TIER 1** ✅ COMPLETE (SWSKIP+ICEFLUXDEV+NOFENCE2+IOACC; standard set, jobs 26238084-86) | **0.9145** | 4.6005 | **5.03** | **0.5520** | 2.3624 | **4.28** | **0.3432** | 1.2188 | **3.55** | **0.2622** | 0.8563 | **3.27** | — |

## ⭐⭐ STAGE-1 TARGET MET IN TIER 1 — 5.02× at NG5@4N, SYPD@dt240 = 1.99 at 16N

**All four Tier-1 levers, same-allocation A/B (job 26237207): −28.39%.** knob-OFF 1.2798 → knob-ON
0.9165 s/step; both reps agree to 0.01%; the knob-OFF leg reproduces row-0 (1.2796) exactly.

| lever | A/B NG5@4N |
|---|--:|
| **`SWSKIP`** — skip the DEAD host `sw_3d` | **−26.47%** (job 26237206) |
| `ICEFLUXDEV` — `ice_oce_fluxes_mom` → device | −0.72% |
| `NOFENCE2` — post-unpack halo fence | ~−0.8% |
| `IOACC` — 6 host I/O accumulators → device | ~−1.1% |
| **ALL FOUR, NG5@4N** | **−28.39%** (job 26237207) |
| **ALL FOUR, dars@8N** | **−17.29%** (job 26237208) |

**Every one is bit-identical**, and `SWSKIP` / `ICEFLUXDEV` / `IOACC` each carry a **passing
FORCE_SERIAL byte proof** (identical bytes by re-execution on Serial, not by argument). `NOFENCE2` is
a pure ordering change (no arithmetic) and is memcheck-clean. **Nothing here trades accuracy for
speed** — the campaign's licence to break bit-identity was never even spent.

### ⚠️ The lever pays LESS at scale — do NOT extrapolate the 4N factor

**NG5@4N −28.39% but dars@8N only −17.29%.** That is not noise: the host segment is **32% of the step
at NG5@4N but 24.7% at dars@8N** (the 16N-class per-rank proxy), so a lever that removes host work
necessarily pays less as ranks shrink. **I initially quoted 8N/16N ratios by scaling row-0 with the
4N factor. That was an over-claim, and the dars@8N A/B caught it.**

| | GPU row-0 | CPU | ratio now | **TIER 1** | basis |
|---|--:|--:|--:|--:|---|
| **NG5@4N** | 1.2796 | 4.6005 | 3.60× | **5.02×** ⭐ | **MEASURED** (A/B 26237207) |
| dars@8N | 0.3178 | 0.8563 | 2.69× | **3.26×** | **MEASURED** (A/B 26237208) |
| NG5@8N | 0.7381 | 2.3624 | 3.20× | *[3.87 – 4.47×]* | **NOT measured** — job 26238085 |
| NG5@16N | 0.4487 | 1.2188 | 2.72× | *[3.28 – 3.79×]* | **NOT measured** — job 26238086 |

The brackets are bounded by the two measured factors (dars@8N = pessimistic, NG5@4N = optimistic).
**NG5@16N SYPD@dt240 is therefore in [1.72 – 1.99], not "1.99"** — the dars@8N proxy is the faithful
one for per-rank host work, so expect the LOW end. The standard set is running (26238084-86);
**quote nothing here until it lands.**

**What IS established: the Stage-1 target (≥5.0× at NG5@4N) is MET and MEASURED at 5.02×.**

---

**Original gap: 3.60× → ≥5.0× at 4N (Stage 1); flatten the 2.72× at 16N (Stage 2).**

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

## ✅ Task 5.1 — OPTIONS MATRIX: `FESOM_SPEED=1` × each M6 physics knob

The speed levers must not break the M6 options matrix. CORE2 dist_8 CUDA fidelity gate, master
switch ON, each compared against **that knob's own M6 Serial oracle** (the physics differs, so the
default-config baseline is the wrong reference):

| # | config | reference | verdict |
|---|---|---|---|
| 26238785 | `FESOM_SPEED=1` + `FESOM_MIX_SCHEME=TKE` | `m6/tke_bitid/kk_tke` | ✅ **PASS** |
| 26238786 | `FESOM_SPEED=1` + `FESOM_WHICH_EVP=1` (mEVP) | `m6/mevp_bitid/kk_mevp` | ✅ **PASS** |
| 26238787 | `FESOM_SPEED=1` + `FESOM_ALE=zstar` | `m6/zstar_bitid/kk_zstar` | ✅ **PASS** |

**zstar was the one that could bite**, and it is worth being explicit about why: zstar makes
`zbar_3d_n` **time-varying**, and the device shortwave kernel that `SWSKIP` now relies on reads it —
the exact shape of the Z7 bug (L78). It passes, and the proof is stronger than a green light:

| field | M6's own zstar gate (no speed levers) | M7 zstar × `FESOM_SPEED=1` |
|---|--:|--:|
| Kv | 9.537e-02 | **9.537e-02** |
| S | 5.378e-04 | 5.382e-04 |
| T | 1.419e-03 | 1.367e-03 |

**The Kv delta is IDENTICAL to four significant figures.** That 9.5e-02 (95% of its 1e-01 ceiling)
is **zstar's own CUDA-vs-Serial floor**, not something the levers introduced — check the floor before
judging the number (L79). `FESOM_SPEED=1` adds *nothing* on the zstar path.

*(Aside worth knowing: under zstar the host `sw_3d` may have been computing from a stale host
`zbar_3d_n` all along — and nobody noticed, precisely because its output was dead. `SWSKIP` deletes
the latent stale read along with the wasted work.)*

### ✅ compute-sanitizer on the combined `FESOM_SPEED=1` config (job 26238798)

Baseline-differential (same tool, knobs OFF then ON — only the difference is attributable):

| leg | benign `cuCtxGetDevice` artefact | **real: Invalid read / write / use-after-free** |
|---|--:|--:|
| knobs OFF | 8 | **0** |
| `FESOM_SPEED=1` | 8 | **0** |

**Zero invalid memory accesses in either leg** — which is precisely the class `NOFENCE2`'s `grow()`
hazard would have produced had the explicit realloc fence been missing. The only findings are the
known-benign `CUDA_ERROR_INVALID_CONTEXT` on `cuCtxGetDevice` (UCX/CUDA-aware-MPI probing the context
from a non-CUDA thread under the sanitizer), present identically with the knobs off.

**racecheck** (job 26238799) is also clean — `RACECHECK SUMMARY: 0 hazards displayed (0 errors,
0 warnings)` on both legs. Run for completeness only: **racecheck is shared-memory-only and cannot
speak to `NOFENCE2`'s global-memory ordering either way** (L80/L81 note). **memcheck is the tool that
can, and it says zero.**

## Knob registry

⚠️ **Every knob here is verified LIVE on the CUDA build** (preprocessor check, all five TUs). `SWSKIP`
and `IOACC` once resolved silently to OFF there — see L80 above. Since the fix, each lever also
**announces itself on rank 0** and shouts if requested-but-OFF.

| knob | lever | class | status |
|---|---|---|---|
| `FESOM_SPEED` | master switch (all blessed levers) | — | scaffolded |
| `FESOM_SPEED_FORCE_SERIAL` | dev-only: allow levers on Serial (byte proofs) | — | scaffolded |
| `FESOM_SPEED_SYNCSTATS` | per-step sync/fence counters (diagnostic) | — | ✅ implemented (0.3/1.1). **Opt-in ONLY** (`fesom_speed_on_exp`) since A.1 — a diagnostic must not ride the master switch |
| **`FESOM_SPEED_FLAT`** | **flatten 4 column-loop kernels to one-thread-per-(column,level)** — `io_acc_u/v`, `kpp_ri_iwmix` (2→4 launches), `ale_vvel_divide`, `ale_wvel_split` | **bit-id** | ✅ **LANDED (A.1)** — all gates PASS incl. FORCE_SERIAL byte proof on **both** streams (snapshots + monthly). A/B below |
| **`FESOM_SPEED_SWSKIP`** | **skip the DEAD host `sw_3d` (the device twin rebuilds it in full)** | **bit-id** | ✅ **LANDED — A/B −26.47%.** All gates PASS incl. the FORCE_SERIAL byte proof (1.2) |
| `FESOM_SPEED_ICEFLUXDEV` | port `ice_oce_fluxes_mom` to device | bit-id | ✅ landed, all gates PASS. **A/B −0.72%** — real but small |
| `FESOM_SPEED_NOFENCE2` | drop post-unpack halo fence | bit-id | ✅ landed, all gates PASS + memcheck-clean. **~−0.8%** (1.1) |
| `FESOM_SPEED_IOACC` | 6 host I/O mean accumulators → device (`ssh`,`a_ice`,`m_ice`,`m_snow`,`uice`,`vice`) | bit-id | ✅ implemented, byte gate + FORCE_SERIAL proof PASS. A/B pending (1.0b) |
| **`FESOM_SPEED_ROTCACHE`** | **cache the JRA wind-rotation sin/cos (a MESH CONSTANT recomputed 1.85 M×/rank/step)** | **bit-id** | ⏳ implemented (D.0), gates pending. ⚠️ **host-side lever — speeds the CPU reference up too** |
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

### H.8 — `FESOM_SPEED_LAZYSNAP` (the ice OUT rail → snapshot cadence) — ✅ LANDED `c9f2fee` (session 7, 2026-07-15)

The 9-copy ice OUT rail (`fesom_ice.cpp:934`, ICERAILS' own "ONE OUT rail that replaces all 67")
fired every step to serve a reader that runs at *snapshot* cadence — never in benchmarks
(`snap_every=-1→0`), monthly in production. The lever gates the rail off and makes
`fesom_io_write_snapshot` pull the 7 gathered ice fields itself (unconditionally — no-op when
Synced, loud `h_checked()` abort under SYNCCHECK if a sync is ever missed). `srfoce_u/v` are
gathered by nothing → pure deletions.

- **Sized from the census** (26258712, h9, 300 steps): the whole `ice_h_diag→oce_fluxes_mom` gap,
  **7.3 ms/step** (5.0 PCIe = 9 DtoH × 3.54 MB). Pre-registered **−1.1 % (floor −1.0, ceiling −1.6)**.
- **A/B 26259170 (h10 `13dbddb4`, a100_80): −1.05 % = −7.3 ms/step — the census number TO THE
  DECIMAL.** First census-sized lever with no entanglement bonus (pure PCIe+fence, no host compute
  in the gap): the census was exact, not a floor. L93's calibration note updated accordingly.
- **NOT independent** (BULKTAIL pattern): REQUIRES ICERAILS + ICEFLUXDEV + FLUXDEV + SWSKIP + IOACC
  (each kills a different host reader the rail fed); aborts on any missing (L80) and on
  `FESOM_DIAG_MICE`/`FESOM_DIAG_GID`. Guard test 26259169 proved the abort fires.
- **Gates 9/9** (26259160-69): knob-OFF byte ✓ FORCE_SERIAL ×2 ✓ CUDA fidelity iso+full ✓✓
  options TKE/mEVP/zstar ✓✓✓ (zstar's Kv control identical to h9's floor, L79) guard-abort ✓.
- The handoff's "ocean half" (~2× hope) was **falsified by audit**: those census rows are the SSH
  nod2D host-halo bounces + the host `eta_n` update — step-cadence consumers, scoped separately as
  **H.9 SSHRAILS** (~13–14 ms; see the session-7 findings §2c).
- Binaries: **h10** = h9 + LAZYSNAP, CUDA `13dbddb4` / Serial `7c75afc0`, frozen `m7/bin/h10/`.

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

| tier | knobs ON | 1-yr CORE2 climate | NG5@16N direct | tag |
|---|---|---|---|---|
| 0 | none | n/a — baseline CUDA fidelity gate **PASS** (worst 9.9e-03 `h_ice`; job 26235125) | GPU 0.4487 / CPU 1.2188 ✅ | `m7.0-baseline` ✅ `3d00123` |
| **1** | **SWSKIP + ICEFLUXDEV + NOFENCE2 + IOACC** | ✅ **PASS** (job 26238055) | ✅ **0.3432 → 3.55×** (26238086) | ✅ `m7.1-bitid` |

## ⭐ TIER-1 VERDICT — measured standard set (jobs 26238084-86)

| | row-0 GPU | Tier-1 GPU | CPU | now | **TIER 1** | gain | nod2D/rank |
|---|--:|--:|--:|--:|--:|--:|--:|
| **NG5@4N** | 1.2796 | **0.9145** | 4.6005 | 3.60× | **5.03×** ⭐ | −28.5% | ~462k |
| NG5@8N | 0.7381 | **0.5520** | 2.3624 | 3.20× | **4.28×** | −25.2% | ~231k |
| **NG5@16N** | 0.4487 | **0.3432** | 1.2188 | 2.72× | **3.55×** | −23.5% | ~116k |
| dars@8N | 0.3178 | **0.2622** | 0.8563 | 2.69× | **3.27×** | −17.5% | ~99k |

**Against the campaign's three stated goals:**

| goal (plan Overview) | result |
|---|---|
| **Stage 1: ≥5.0× at NG5/dars 4–8N** (4N firm, 8N stretch) | ✅ **NG5@4N = 5.03× — MET.** NG5@8N = 4.28× (the stretch edge, not reached) |
| **Stage 2: flatten the ratio decay toward 16N** (was 2.72×) | ✅ **NG5@16N = 3.55×**, up from 2.72×. But the *relative* decay 4N→16N widened (24% → 29%) because the lever pays more where per-rank domains are large — honest, and it points Stage-2 work at comm/imbalance (Tier 3), not host code |
| **~2 SYPD @ dt240 on NG5 at 16–32N, pure FP64** | **SYPD@dt240 = 1.86 at 16N** (from 1.42) — **within 7% of the goal**, in pure FP64, with mixed precision banned |

**The gain tracks per-rank domain size** (−28.5% at 462k nod2D/rank → −17.5% at ~99k), exactly as a
host-work lever must. ⚠️ **Never extrapolate one scale point's factor to another** — I did, published
"SYPD 1.99", and the dars@8N A/B caught it. The measured answer is **1.86**.

### ✅ Tier-1 1-yr CORE2 climate gate — PASS (job 26238055)

Full model year, all four speed knobs ON, vs the certified DEFAULT-config references
(`kpp_5yr_fix --cref-frame rotated` + `zstar/fortran_linfs_2yr_b`; vectors rotated to geographic on
both sides, L74). Ran the full 17 280 steps, exit 0, T[−2.01, 31.27] / S[3.98, 41.06] — bounded, no
runaway, no NaN.

| field | vs Fortran | vs C-port | **the bar** (M5.23 CUDA, un-levered) |
|---|--:|--:|--:|
| sst | **1.00000** | 1.00000 | 1.00000 |
| sss | **0.99996** | 0.99996 | 0.99996 |
| ssh | **1.00000** | 1.00000 | 1.00000 |
| a_ice | **0.99997** | 0.99997 | 0.99997 |
| m_ice | 0.99997 | 0.99998 | — |
| uice / vice | 0.99973 / 0.99976 | 0.99974 | — |

**Identical to the un-levered baseline, to five decimal places** — which is what must happen:
`SWSKIP`/`ICEFLUXDEV`/`IOACC` each carry a passing FORCE_SERIAL byte proof and `NOFENCE2` changes no
arithmetic. **The speed came from deleting dead work, not from trading accuracy.**

---

## Post-Tier-1 MEASURED budget (jobs 26242512 + 26242513) → the 2026-07-14 re-scope

Both `FESOM_SPEED=1` on the frozen Tier-1 binary `788844b3`. Knobs verified live three ways
(SYNCSTATS `post-unpack 0.0 / SKIPPED 437.3`; the device I/O resolvers visible in the GPU trace;
step time reproduces the Tier-1 A/B). Raw: `m7/nsys_t1_ng5_4n/stall_budget.txt`, `m7/stepprof_t1/run.log`.

**NG5@4N, 913.5 ms/step.** Phase view (26242513, phase-timer overhead ~5%): **ocean 71.9%
(693.7 ms) · sea-ice 13.4% (129.7 ms) · forcing 10.1% (97.1 ms) · coupling 0.3%** (SWSKIP killed it).

| pool | ms/step | % |
|---|--:|--:|
| GPU kernels | 591.8 | 64.8 |
| memcpy (MPI staging + forcing HtoD) | 89.8 | 9.8 |
| MPI wait + Isend/Irecv call time | 138.1 | 15.1 |
| launch gap + other CUDA API | 46.5 | 5.1 |
| fence spin | 18.1 | 2.0 |
| host segment (unnamed → hostprof 26243196) | 66.9 | 7.3 |

Comm machinery detail: 437.3 exchanges/step → 1955.7 Isends (32.3 ms of CALL time alone), 3636
MPI-internal device syncs, Waitall 72.1 ms. CG = 351 launches + 179 reduces/step for ~22.6 ms GPU.

**Top kernels (ms/step):** FCT pipeline **181.8** (56 launches; two monsters 18.9+17.0 ms ×2/step) ·
`ale_vert_vel` 44.1 (scatter ≈29.5, cumsum 7.9, divide 6.7) · `redi_expl` 40.7 · `impl_ale` TDMA
32.6 · `momentum_adv` ~25 · `hor_redi` 22.9 · **`resolve_u/v_dev` 21.7** (column-loop I/O
accumulators, ~20× the streaming floor) · `visc_filt` 19.6 · `smooth` 17.8 · `kpp_ri_iwmix` 17.0 ·
`impl_vert_visc` TDMA 13.3 · `compute_vel_nodes` 12.8 · `pressure_bv` 11.3 · `fer_solve_gamma` 9.4 ·
`wvel_split` 7.2.

**Consequence — the plan is re-scoped into packages A–F** (see the plan's RE-SCOPE section for the
full ladder + the 8× arithmetic): A flatten column-loops + hostprof + UCX config · B FCT2/Redi
batching + scatter store+gather · C TDMA/spill · D forcing→device · E CG1R/EVPWIDE/CGPOLY ·
F ICELAG **(user-approved as EXPERIMENT 2026-07-14)**. Central estimate without F ≈ 7.0–7.5×, with
F ≈ 8×+. **Layout big-bet DEMOTED** (the #1 kernel is already at 59% of DRAM peak — a flip helps
only the ~130–150 ms column-serial class that A/C fix per-kernel). User decisions: start package A;
ICELAG experiment green-lit; **no push**.

---

# Package A — results (2026-07-14, Opus session)

## Task A.3 — UCX/transport env A/B (job 26243303, frozen Tier-1 binary, NG5@4N, min of 2 reps)

| leg | s/step | vs ref |
|---|--:|--:|
| `ref` (current job env) | 0.9161 | — |
| `UCX_RNDV_SCHEME=get_zcopy` | 1.2380 | **+35.14%** |
| `UCX_RNDV_SCHEME=put_zcopy` | 1.2379 | **+35.13%** |
| `UCX_RNDV_FRAG_MEM_TYPE=cuda` | 0.9056 | −1.15% |

**Verdict: NOTHING ADOPTED — Task A.3 CLOSED.** The adoption bar was >2% *and* a fidelity gate;
the only positive leg (`fragcuda`, −1.15%) misses the bar, and the two rendezvous-scheme overrides
are catastrophic — forcing zcopy defeats UCX's own pipelined staging for GPU buffers. The default
env is already the right one. Transport diag (`UCX_LOG_LEVEL=info`): every rank pair runs
`tag(rc_mlx5/mlx5_0:1)` — RC over the single IB port, as intended.

## Task A.2 — hostprof: WHAT the "unnamed host segment" is (job 26243196)

CPU call-stack sampling (`--sample=process-tree --backtrace=dwarf`), rank 0, 22-step steady window,
~880 Hz, **every sample `Running`** (the host never blocks: MPI polls and CUDA fences spin), so one
sample = 1.136 ms of main-thread wall clock. Traced step 926 ms vs the untraced 913.5 → 1.4%
perturbation; the shape is trustworthy.

**Answer: the host segment is the JRA55 FORCING, and it is 100% serial host code.** The host timer
(not the sampler — see the L81 note below) says it plainly:

| host phase (FPROF bracket, rank 0) | ms/step | calls/step |
|---|--:|--:|
| **`force:jra55_read`** (`fesom_jra55_step_cal`) | **75.2** | **1.0** |
| `force:bulk_compute` | 21.9 | 1.0 |
| → forcing phase total | **97.1** (10.1% of step) | |

Sampling attribution inside it (deepest app frame, ms/step): `getcoeffld` 38.3 · `sincos` 23.5 ·
`fesom_vector_g2r` 13.8 · `fesom_bulk_compute_kk` 5.4 · `fesom_cal_shortwave_rad` 1.5 ·
`read_one_time_slice` 1.2. Also visible: `Kokkos DeepCopyCuda` 26.3 (the staging memcpy's host
cost) and `profile_fence_event` 145.5 (host spinning in `cudaStreamSynchronize` — that is *waiting
for the GPU*, i.e. the 591.8 ms of kernels, not overhead).

### 🔴 The arithmetic that actually scoped package D (and a fresh L81 sighting)

The sampler says `getcoeffld` costs 38.3 ms/step. **It cannot.** `getcoeffld` re-reads and
re-interpolates a JRA time slice only when `rdate` leaves `[nc_time[t_indx], nc_time[t_indx_p1]]`
— 3-hourly data at dt=180 → **once per 60 steps**, and the year-start prefetch covers step 1, so in
a 25-step run its body runs **ZERO times**. (Verified by transcribing the C control flow into Python
and running it against the real `uas.1958.nc` time axis: 3 refreshes per 200 steps, at steps 61 /
121 / 181. The time-axis transform `nc_time/nm_nc_freq + julday(1900,1,1)` makes `nc_time[0]`
exactly equal `julday(1958,1,1)` = 2436205, so the search lands cleanly at `t_indx=1`.) `getcoeffld`
is `static` and gets inlined into `fesom_jra55_step`; with `--backtrace=dwarf` nsys expands DWARF
inline frames, so the caller's per-node loop is being reported under the inlined callee's name.
**L81 again, in a new costume: the profiler localised (the forcing phase) and mis-named (the
function).** The host timer + node arithmetic is what settled it.

**What the 75.2 ms actually is:** the per-node time-interpolation loop in `fesom_jra55_step`
(`:665-701`), which runs over `myDim_nod2D` = **463 k nodes/rank** at NG5@4N and, for every node,
every step, calls `fesom_vector_g2r` → **4 `sincos` per node** (`sin/cos` of `glat, glon, rlat,
rlon`, CSE'd by the compiler). 463k × 4 sincos ≈ 1.85 M transcendental pairs per rank per step
≈ **~44 ms at 2.5 GHz** — which is exactly what the sampler's `sincos`+`g2r` (37 ms) shows. The rest
is 16 coef-array streams in / 8 forcing arrays out (~89 MB/step) plus the host halo exchange of the
8 forcing fields, all inside the same bracket.

### 🔴 The lever this hands package D: **the rotation trig is a CONSTANT**

`fesom_vector_g2r` rotates the wind using only the node's **fixed** coordinates (`geo_coord_nod2D`,
`coord_nod2D`) and a **cached** rotation matrix. The four `sincos` values per node are therefore
**identical on every step, forever** — ~44 ms/step of recomputing a mesh constant. Precomputing them
once into a per-node table (8 doubles/node = 30 MB/rank) leaves the expression tree untouched → the
same operands in the same order → **bit-identical**, and turns 4 transcendentals into 8 loads.

→ **New Task D.0 (`FESOM_SPEED_ROTCACHE`)**, ahead of the D.1 device port: ~20 lines, expected
**−40 ms/step ≈ −4% at NG5@4N** — on its own comparable to all of package A, and the cheapest lever
left in the campaign.

### 🔴 L84 — the rank-count asymmetry (why host code keeps winning)

ROTCACHE is a *host* fix, so it was flagged as a threat to the GPU-vs-CPU ratio ("it speeds the CPU
reference up too"). Right in principle, **wrong by 160×**, and the reason reframes the campaign:

| | ranks/node | **nodes/rank** @ NG5 4N | the rotation trig |
|---|--:|--:|--:|
| **GPU** run | **4** (one per GPU) | **463 k** | 37 ms of a 913.5 ms step = **4.05%** |
| **CPU** run | **128** (one per core) | **14.5 k** | 1.2 ms of a 4599 ms step = **0.025%** |

The forcing loop is **per-rank serial host work**. The CPU run spreads the mesh over 128 processes per
node; the GPU run concentrates it into 4 — so **the GPU config carries ~32× more host work per rank,
for identical code.** (Measured: `jobs/job_m7_ab_cpu`, job 26244994.)

Hence: a host cost invisible in a CPU profile can dominate the GPU step; **this is *why* host code keeps
being the answer in M7** (SWSKIP −26%, IOACC, ICEFLUXDEV, now the forcing) — it is structural, so expect
the next bottleneck to be host code too; and **the fix is always "move it to the device", never "make the
host loop faster."** D.0 (−4%) buys the amplification factor once; **D.1 (−8%) removes it.**

The CPU denominator moves by 0.025%, i.e. **40× below the same-alloc noise floor** — so a host lever does
not inflate the ratio. But that is because of the rank asymmetry, *not* because the lever is GPU-specific.
**Still re-measure the CPU anchor same-day when quoting a ratio** (that rule is about ±5% cluster noise).

#### MEASURED (job 26244994, `jobs/job_m7_ab_cpu`, NG5@4N = 4 nodes × 128 ranks = dist_512)

| leg | rep a | rep b | min | vs base |
|---|--:|--:|--:|--:|
| `base` | 4.5965 | 4.5853 | **4.5853 s/step** | — |
| `rot` (`FORCE_SERIAL=1;ROTCACHE=1`) | 4.6258 | 4.6089 | **4.6089 s/step** | **+0.51%** |

**Verdict: ROTCACHE gives the CPU NOTHING** (predicted +0.025%; measured −0.51%). The announce fired, so
the knob was live — this is not L80. Read it honestly: the within-leg rep spread is **0.24–0.37%**, so a
0.51% gap is at the edge of what 2 reps resolve, and the obvious mechanism does **not** hold up — the
table is ~960 KB/rank × 128 ranks/node ≈ 123 MB/node/step, which at Levante's ~400 GB/s is **0.3 ms**, not
the ~23 ms a real +0.51% would need. **So: no measurable effect, NOT a demonstrated regression.** Do not
claim an L3-pressure story without evidence for it.

**What this settles:** the CPU denominator is untouched → **the GPU-vs-CPU ratio is honest**, and the
existing "levers act on the CUDA path only" rule (knobs resolve OFF on a non-CUDA build) is now the right
default *on performance grounds*, not merely as protection for the Serial debug oracle. **Do not enable
ROTCACHE on the CPU path.**

**Same-day CPU anchor: NG5@4N = 4.5853 s/step** — 0.3% from the ledger's 4.599, so today's cluster is
consistent with the row-0 anchor and the package-A ratio row can be computed against it.

## Task A.1 — `FESOM_SPEED_FLAT` (4 flattened column-loop kernels)

One boolean knob, four sites, all re-parallelized from one-thread-per-column to
one-thread-per-(column, level):

| site | file | measured (Tier-1, NG5@4N) |
|---|---|--:|
| `io_acc_u` / `io_acc_v` | `fesom_io.cpp:765/776` | 21.7 ms (10.9 each; ~20× the streaming floor) |
| `kpp_ri_iwmix` (2 launches → 4) | `fesom_kpp.cpp:343` | 17.0 ms (SM 2.8%, 23.6 sec/req) |
| `ale_vvel_divide` | `fesom_ale.cpp:456` | 6.7 ms |
| `ale_wvel_split` | `fesom_ale.cpp:518` | 7.2 ms |
| | | **pool 52.6 ms** |

`ale_vvel_scatter` (29.5 ms) and `ale_vvel_cumsum` (7.9 ms) are deliberately **NOT** touched — their
per-column order *is* their bit-identity (packages B.3 / C.2).

**Bit-identity argument.** Every flattened site is a pure per-(n, nz) map whose every slot is written
exactly once from operands no other slot writes, so slot order cannot change a byte. `ri_iwmix` is
the only subtle one: each legacy loop splits into an INTERIOR kernel (the map) and an EDGE-COPY
kernel (per node, six assignments kept in the legacy order inside one lambda). That is exact because
a node's edge copies only ever touch that node's own slots and already ran after that node's interior
loop — including for the degenerate `nzmax == nzmin+1` column, whose stale-carry chain the edge
kernel reproduces verbatim (the interior kernel never touched those slots either). Launch order
1→2→3→4 is load-bearing, exactly as D20's 1→2 was.

### A/B — NG5@4N, 4 legs, ONE allocation (job 26244262; min of 2 reps)

| leg | s/step | vs Tier-1 | ms/step |
|---|--:|--:|--:|
| `t1` (Tier-1) | 0.9154 | — | reference (reproduces the 0.9145 Tier-1 anchor to **0.1%**) |
| `flat` (+`FLAT`) | 0.8873 | **−3.07%** | −28.1 |
| `rot` (+`ROTCACHE`) | 0.8957 | **−2.15%** | −19.7 |
| **`both`** | **0.8662** | **−5.37%** | **−49.2** |

Additivity: the two marginals sum to −5.22% vs −5.37% measured → disjoint code paths, as designed
(the small excess is second-order: less host time ⇒ less GPU idle).

**L80 announce armor — both knobs fired, and only where they should:** `t1` 0/0 · `flat` 3/0 ·
`rot` 0/1 · `both` 3/1 (FLAT announces once per TU: io/kpp/ale). Neither is a dead knob.

**Ratio, both sides measured the SAME DAY** (CPU anchor 4.5853, job 26244994 — no cross-day noise):
**5.01× → 5.29× at NG5@4N.**

#### 🔴 Both levers under-shot the predicted payoff. In both cases the PREDICTION was wrong, not the lever.

- **`ROTCACHE` −19.7 ms vs a predicted ~37 ms.** The estimate counted the sampler's `sincos` (23.5 ms)
  **and** `fesom_vector_g2r` self (13.8 ms) as the removable pool. **Only the `sincos` is removable** —
  ROTCACHE keeps the 3×3 matrix arithmetic inside `g2r` and merely feeds it trig from a table. The
  correct pool is `23.5 − ~4` (the new 30 MB/step of table loads) ≈ **−20 ms**. Measured **−19.7 ms**.
  **The arithmetic closes exactly once the attribution is right.**
  → *Rule: when sizing a lever off a profile, count only the work the lever REMOVES, not the whole
  function it lives in. The `g2r self` samples are the arithmetic that survives.*
- **`FLAT` −28.1 ms = 53% of the 52.6 ms pool.** This is the **predicted ceiling, not a shortfall**:
  `uv`/`uvnode` are stride-2 interleaved (`FESOM_ELEMVEC`), so even a perfectly flat access pattern is
  **~50% sector-efficient by construction** (handoff trap #3). Roughly half the pool is unreachable
  without a layout flip — which the RE-SCOPE demoted on measured grounds. 53% is the right answer.

### Gates — ALL PASS

| gate | job | verdict |
|---|---|---|
| knob-OFF byte gate | 26243832 | ✅ `diff_snap` rc=0 — default path untouched |
| FORCE_SERIAL byte proof (`snap_*.nc`) | 26243833 | ✅ rc=0 — covers kpp (`viscA`/`diffK`) + ale (`w`, `w_e`, `w_i`) |
| FORCE_SERIAL byte proof (`*.monthly.nc`) | ↑ same run | ✅ rc=0, **17 files** incl. `u`/`v` — covers the io resolvers |
| CUDA fidelity gate | 26243834 | ✅ PASS (worst 3.745e-03, the climate-close floor) |
| announce (L80 armor) | — | ✅ exactly **3** `FESOM_SPEED_FLAT = ON` lines (io, kpp, ale) — the knob FIRED |

### ⚠️ A gate hole this task found and closed

The two io accumulators feed **only the time-mean stream** (`u/v.fesom.1958.monthly.nc`); they never
touch `snap_*.nc`, which is all `diff_snap.py` globbed. So a FORCE_SERIAL proof of `FLAT` would have
been **vacuous for two of its four sites** — the kernels would run, and nothing would compare their
output. `diff_snap.py` now takes `--pattern` (default `snap_*.nc`, so every existing gate is
unchanged) and the proof runs twice: once on the snapshots, once on `*.monthly.nc`. **Any future
lever touching an io resolver must do the same** — a passing gate that never reads the lever's output
is not evidence.

### dars@8N — PREDICTION, recorded BEFORE the run landed (job 26245783)

Pre-registering this because the campaign has twice been burned by reading a result and then
constructing the story that fits it (L80, L81). dars@8N has **98.5 k nodes/rank** vs NG5@4N's
**463 k** — a factor **0.213**.

| lever | NG5@4N measured | dars@8N predicted | basis |
|---|--:|--:|---|
| `ROTCACHE` | −2.15% (−19.7 ms) | **−1.3%** (−4.2 ms) | pure per-node HOST work ⇒ scales with nodes/rank (**L84**) |
| `FLAT` | −3.07% (−28.1 ms) | −1.9% (−6.0 ms) **or better** | GPU kernel work; launch overhead + occupancy floors mean it should NOT scale down linearly |
| `both` | −5.37% | ~−3.2% naive | |

(The packA handoff independently guessed −2..3% for FLAT at dars@8N.)

🔴 **This is a FALSIFIABLE TEST of L84.** If `ROTCACHE` instead holds near −2% at dars@8N, then host
cost does *not* scale with nodes/rank, the per-rank-amplification story is **wrong**, and the
"host code is structurally amplified 32× on the GPU config" conclusion — which is currently steering
the whole campaign toward D.1 — must be revisited.

### dars@8N — RESULT (job 26245783), scored against the prediction above

| leg | s/step | vs Tier-1 |
|---|--:|--:|
| `t1` | 0.2625 | — |
| `flat` | 0.2598 | **−1.03%** (−2.70 ms) |
| `rot` | 0.2577 | **−1.83%** (−4.80 ms) |
| **`both`** | **0.2549** | **−2.90%** |

Announce armor: `t1` 0/0 · `flat` 3/0 · `rot` 0/1 · `both` 3/1. Both knobs live.

#### ✅ L84 SURVIVED FALSIFICATION — the host-cost scaling law holds

| lever | predicted | measured | verdict |
|---|--:|--:|---|
| `ROTCACHE` (host) | 4.19 ms | **4.80 ms** (1.15×) | **tracks per-rank size** (×0.244 vs the ×0.213 yardstick) |
| `FLAT` (GPU kernel) | 5.98 ms | **2.70 ms** (0.45×) | **decayed 2.2× FASTER than per-rank size** |

The rival hypothesis — "host cost does *not* scale with nodes/rank" — predicted ROTCACHE ≈ 19.7 ms
= **7.5%** of the dars step. Measured: **1.83%**. The test discriminated decisively. **L84 stands:
serial host work scales with nodes/rank, so the GPU config (4 ranks/node) is structurally penalised
vs the CPU config (128 ranks/node).**

#### 🔴 THE UNPREDICTED FINDING, and it outranks both levers

**I predicted FLAT would hold up BETTER than linear. It held up 2.2× WORSE.** And the ordering
*inverts* between the two scale points:

| | NG5@4N (compute-bound) | dars@8N (comm-bound) |
|---|--:|--:|
| `FLAT` (GPU **kernel** lever) | **−3.07%** | −1.03% |
| `ROTCACHE` (**host** lever) | −2.15% | **−1.83%** |

**Mechanism:** in a comm-bound step, GPU-kernel time you free up is partly absorbed by MPI wait the
GPU was doing anyway — but **host** time you free up is on the critical path regardless of what the
GPU is waiting for. So **kernel levers decay in the comm-bound regime; host levers do not.**

**Why this is a campaign-level warning:** dars@8N is **98.5 k nodes/rank ≈ NG5@16N's 115 k** — by the
per-rank-proxy method it *is* the **Stage-2 regime**. So:

- **Packages B (FCT2, 181.8 ms pool) and C (TDMA) are pure GPU-KERNEL levers.** Their NG5@4N payoff
  will **not** carry to 16N — which is exactly where Stage 2 / the 2-SYPD goal lives. Size them at 4N,
  but **do not bank them at 16N without measuring at dars@8N first.**
- **Package D.1 (forcing → device) is a HOST lever** and should hold its value into the Stage-2 regime.
  Combined with L84 (host work is 32× amplified on the GPU config), D.1 is now the best-evidenced lever
  in the ladder.
- **Package E (CG1R/EVPWIDE/CGPOLY) attacks the comm itself** — and the comm is what is absorbing the
  kernel levers' gains at 16N. Its priority goes UP.

⚠️ **Caveat, stated honestly:** dars is a different *mesh*, not just a different rank count, so
"comm-bound" is confounded with mesh geometry. The per-rank-proxy method ([[feedback-per-rank-proxy]])
says dars@8N is faithful for halo/comm behaviour (NOT for CG iteration counts). **Confirm at NG5@16N
before betting the ladder on it** — but the direction is strong enough to re-rank D and E above B/C now.

---

# Task D.1 — `FESOM_SPEED_FORCEDEV` (the forcing loop → device)

Implemented 2026-07-14. The per-node time-interpolation loop becomes a Kokkos device kernel; the 8
forcing arrays turn DEVICE-authoritative; the 8 separate host halo rounds collapse into ONE
`fesom_halo_fieldN` (device pack/exchange/unpack); the two host→device IN rails are skipped.

### 🔴 What made this dangerous, and what a Serial proof cannot tell you

The 8 arrays are DualViews whose raw host pointers were written in place by the host producer,
invisibly to the DualView — so `fesom_bulk.cpp:520` and `fesom_ice.cpp:647` call
`modify_host(); sync_device()` on all 8 **unconditionally, every step**. With a device producer those
rails **deep-copy the stale host mirror over the fresh device data.** **On Serial the host and device
views are the same memory, so the rail is a no-op and the FORCE_SERIAL byte proof passes either way.**
The CUDA fidelity gate is the only gate that can see this.

Three surviving host readers, each given a `sync_host()`:
1. `fesom_cal_shortwave_rad` — reads `jra->shortwave` **every step, in production**. `SWSKIP` does NOT
   skip it (it drops only the dead `sw_3d` half; the `heat_flux += swsurf` accumulation always runs).
2. the host `fesom_bulk_compute` — **not verify-only**: `fesom_main.cpp:905` runs it once at init.
3. the host `fesom_ice_thermodynamics` twin — verify-only.

A fourth trap, found while reviewing my own diff: `fesom_halo_fieldN` is guarded on `partit`, and it is
what sets `modify_device()`. On a single-rank run the halo call is skipped, so the fields would have
stayed flagged *host*-current — the device data invisible to the DualView, and every later
`sync_host()` a silent no-op returning stale values. The producer now calls `modify_device()` itself.

### Acceptance arithmetic — PRE-REGISTERED (before the A/B landed), and it corrects my own error

My first projection (5.78×) **double-counted**: it assumed D.1 removes the full 75.2 ms forcing from
the 866.2 ms step — but ROTCACHE had already taken 19.7 ms of it. Forcing now = **55.5 ms**, not 75.2.

| D.1 removes | ms/step |
|---|--:|
| host loop 55.5 → device kernel ~2 | 53.5 |
| 8 host halo rounds → 1 device halo | 2.5 |
| the two rails' H2D (8 × 3.7 MB/step) | 1.5 |
| **NEW cost:** shortwave D2H for the surviving host reader | −0.2 |
| **total** | **57.3** |

- **NG5@4N: 866.2 → ~809 ms/step = −6.6% marginal → ratio ~5.67×** (NOT 5.78×).
- **NG5@16N: −4.8% → SYPD ≈ 2.01** — **Stage 2 is right at the line, not comfortably past it.**
- 🔴 **~0% = DEAD KNOB (L80), not a null lever.** Check the `FORCEDEV` announce line first.

### D.1 gates — ALL PASS (the CUDA gate is the one that mattered)

| gate | job | verdict |
|---|---|---|
| knob-OFF byte gate | 26249149 | ✅ rc=0 — default path untouched |
| FORCE_SERIAL byte proof (`FORCEDEV=1`) | 26249150 | ✅ rc=0, announce fired — the device kernel ran on Serial and was **bit-identical** |
| **CUDA fidelity, `FORCEDEV=1`** | 26249151 | ✅ **PASS** (worst 9.892e-03) — **the only gate that can see the rail bug** |
| **CUDA fidelity, FLAT+ROTCACHE+FORCEDEV** | 26249152 | ✅ **PASS** (worst 4.456e-03) |

The all-knobs run scoring *lower* than FORCEDEV-alone confirms these are run-to-run atomics noise at
the CUDA floor, not a systematic D.1 effect.

**A/B (job 26249153) still queued at session end — see `docs/plans/20260714-m7-HANDOFF-D1.md` §1.**

---

## h17 — CGPIPE ADOPTED into `FESOM_SPEED=1` (session 12 harvest, 2026-07-16) ⭐ THE ROW OF RECORD

h17 = h16 + the one-line `_exp`→master promotion (user decision 2026-07-16). Binaries
`m7/bin/h17/`: CUDA **`f8384e86`** / Serial `5c3c90fc`. Full cert **4/4**:

| gate / anchor | job | verdict |
|---|---|---|
| knob-OFF byte gate | 26299411 | ✅ rc=0 |
| CUDA fidelity, bare `FESOM_SPEED=1` | 26299412 | ✅ PASS (worst 5.281e-03) **+ `FESOM_SPEED_CGPIPE = ON` fired — the L80 adoption check** |
| **4N anchor (std300, min-of-2)** | 26299413 | **0.6382 s/step** (0.6382/0.6393) — pre-reg 0.6381 HIT +0.02% ⇒ **ratio 4.5785/0.6382 = 7.17×** |
| **16N anchor (std300, min-of-2)** | 26299414 | **0.2413 s/step** (0.2414/0.2413) — pre-reg 0.2414 HIT −0.04% ⇒ **16N ratio ≈ 5.09×, Stage-2 SYPD@dt240 ≈ 2.65** |

**NG5@4N: 7.17× · NG5@16N: 5.09× · SYPD 2.65.** 8× @4N = another −10.3% (0.6382 → 0.5723).

## E.EVP1 — `FESOM_SPEED_EVPWIDE=K` wide-halo EVP (OPT-IN; built + FULLY BYTE-CERTIFIED session 12)

Design + the four-correction history: `docs/plans/20260720-m7-evpwide-design.md` (**read the
correction header**; final architecture: R=K rings + per-window SIGMA refresh + `orient_cw`
replay + **verbatim owner-byte shipping for ALL geometry/coriolis — the libmvec lesson:
gcc -O3 vectorizes the mesh geometry loops, so local recompute of anything transcendental is
never byte-safe**). EVP exchanges/step 120 → 120/K + 1 (each refresh = node msg + sigma msg
per partner, same Waitall). Binaries of record `m7/bin/evpw0/` v3: CUDA **`9c900b4f`** /
Serial `21cea692`.

| gate (v3 regate, all on the frozen pair) | job | verdict |
|---|---|---|
| knob-OFF byte | 26306409 | ✅ rc=0 |
| **K=2 FORCE_SERIAL byte proof** | 26306410 | ✅ **rc=0 — BIT-IDENTICAL** |
| **K=4 FORCE_SERIAL byte proof** | 26306411 | ✅ **rc=0 — BIT-IDENTICAL** |
| **K=8 FORCE_SERIAL byte proof** | 26306413 | ✅ **rc=0 — BIT-IDENTICAL** |
| CUDA fidelity K=4 | 26306414 | ✅ PASS (worst 1.129e-02) |
| options TKE / mEVP / zstar (K=4 set) | 26306416/17/18 | ✅ 3× PASS; zstar `Kv 9.537e-02` = the L79 control, exact both regates; mEVP = loud no-op |
| **A/B K-sweep 16N (off/2/4/8)** | 26306419 | ⏳ queued — pre-reg §6: K=4 ceiling −23.4 ms, central −15 |
| **A/B K-sweep 4N (off/2/4/8)** | 26306420 | ⏳ queued — ceiling −18.8 ms, central −12 |

**⇒ EVPWIDE lands in the CGPIPE certification class (byte proof at every K + fidelity +
options), not merely climate-close.** Stays OPT-IN (rule 0.24): a 1-yr climate leg is still
the user's promotion bar. Debug arsenal: `FESOM_EVPWIDE_SELFCHECK=1/2/3`, `FESOM_EVPWIDE_RINGS`;
the pre-step uv echo MUST print exactly 0.000e+00.

## E.CG2 — `FESOM_SPEED_CGPOLY=<d>` Chebyshev polynomial PCG (OPT-IN; session 13)

The session-13 main lever (user 2026-07-16; the JAX port's CGPOLY results imported as priors —
port_jax `00f6e3c`: iters 127→55/42 at d=2/3, A/Bs −20.7 %/−9.6 %/−3.6 %). Design + pre-reg:
`docs/plans/20260721-m7-session13-FINDINGS.md` §2-3. Mechanism: the MITgcm M⁻¹ is replaced by
`p_d(D̃⁻¹Ã)·D̃⁻¹` — d Chebyshev semi-iterations, ZERO dot products — on an **R=(d+1)-ring
single-exchange PCG** (the cgpipe graph generalized round-by-round; frozen-Ã ship-once, so
zstar-safe by the pr_values freeze precedent; λmax by distributed fixed-seed power iteration,
κ=30 default via `FESOM_CGPOLY_KAPPA`). Per iteration: 1 fused R-ring exchange + 2 Allreduce
calls (unchanged) — the lever cuts the ITERATION COUNT itself, the pool CGPIPE cannot touch.
Fidelity class: **knob-OFF byte-identical; ON = solver-tolerance-equivalent** (same
unpreconditioned tolerance, different Krylov trajectory — expect NOT bit-identical, the JAX
class). Binaries `m7/bin/cgpoly0/`: CUDA **`ee2c4fdd`** / Serial `87392308`.

| gate | job | verdict |
|---|---|---|
| knob-OFF byte (CORE2 np8 Serial) | 26313389/off | ✅ **diff_snap rc=0** |
| ring-replay selfcheck d2+d3 (FORCE_SERIAL np8) | 26313389 | ✅ **1939 applies, ALL 0.000e+00 — BITWISE** |
| **E.3 kill-fast verify** (CORE2 dt1800 settled iters) | 26313389 | ✅ **off 128.8 → d2 51.6 (2.50×) → d3 40.1 (3.21×)** — JAX prior 127→55/42 CONFIRMED; λ=[0.0606,1.8165] ≈ JAX [0.0601,1.803] |
| CUDA fidelity (SPEED=1+CGPOLY=3) | 26313390 | ✅ PASS (worst 9.999e-02 = Kv floor) |
| options TKE / zstar (CGPOLY=3) | 26313391/93 | ✅ PASS ×2 (zstar Kv moved off 9.537e-02 in value, held ~1e-1 magnitude — as pre-registered for solver class) |
| **options mEVP (CGPOLY=3)** | 26313392 | ⚠️ **formal FAIL, LEVER EXONERATED by probe 26313804: pure-Serial off-vs-d3 reproduces T 6.602e-02 @ EXACTLY 43 cells (no CUDA anywhere) — deterministic near-freezing branch flips at ice-edge cells (40/43 poleward of ±50°: Nares, Weddell, N-Svalbard), NON-accumulating (max shrinks 0.42→0.066 step 10→20), fingerprint changes with degree (d2: 20 cells). mEVP's per-scheme floor (L79 family) meeting the campaign's FIRST solver-class lever; PASS-criterion redefinition = user's call at review** |
| CUDA selfcheck leg | 26313454 | ✅ PASS + **856/856 selfcheck = 0.000e+00 on CUDA** (CGPIPE parity: bitwise ring replay Serial AND CUDA) |
| **A/B 16N (off/d1/d2/d3, std300)** | 26313501 | ✅ **off 0.2417 (h17 anchor ✓) · d1 −3.19 % · d2 −3.68 % · d3 −4.26 % → 0.2314** — pre-reg central −8..10 % ⇒ the 2nd wrong-HIGH (marginal-cost decay, findings §9 lesson 0.31) |
| **A/B 4N (off/d1/d2/d3, std300)** | 26313502 | ✅ **off 0.6379 (h17 anchor ✓) · d1 −1.61 % · d2 −2.05 % · d3 −2.60 % → 0.6213** — pre-reg ceiling −2.5 % ⇒ the 6th wrong-LOW |

### E.CG2 RESULT (d\* = 3; monotone sweep both scales — no rendezvous signature despite
### worst-partner 81.1 KB @4N / 41.8 KB @16N; settled iters 23 = the model's 22.4)

**NG5@4N: 0.6213 s/step ⇒ 7.37× · NG5@16N: 0.2314 s/step ⇒ 5.31×, SYPD@dt240 ≈ 2.76**
(×1.03 correction now over-conservative — CGPOLY shrinks the CG share ~3×, true correction
≈ 1.01 ⇒ SYPD ≈ 2.81; re-derive at the next E.5-style close). CGPOLY stays **OPT-IN**
(`FESOM_SPEED_CGPOLY=3`); adoption/promotion + climate leg = user decisions (see the
options-mEVP threshold-flip note above). 8× @4N: another −7.9 % (0.6213 → 0.5723).

### E.EVP1 A/B HARVEST (26306420 4N · 26306419 16N; std300, min-of-2, same-alloc, evpw0 v3)

| leg | 4N s/step | Δ4N | 16N s/step | Δ16N |
|---|--:|--:|--:|--:|
| off | 0.6380 | — | 0.2412 | — |
| K=2 | 0.6506 | **+2.0 %** | 0.2495 | **+3.4 %** |
| K=4 | 0.6468 | **+1.4 %** | 0.2452 | **+1.7 %** |
| **K=8** | **0.6341** | **−0.6 % (−3.9 ms)** | **0.2358** | **−2.2 % (−5.4 ms)** |

(off legs reproduce the h17 anchors 0.6382/0.2413 ✓ — v3 knob-off = h17.)

**Verdict per the pre-registered rule: K\* = 8** (argmax 16N, 4N non-regressing). **The pre-reg
MISSED WRONG-HIGH for the first time after five wrong-LOWs** (K=4 central −15 ms @16N,
measured +4.0): the event-count model silently assumed the payload stays eager/latency-class,
but the wide refresh multiplies bytes ×K (+ the sigma segment, 2 msgs/partner) — widest msg
122 KB @16N K=4, 95 KB @4N K=2 ⇒ rendezvous/pinned-bounce regime, per-refresh cost ate the
count saving. 🔴 LESSON (ledger-grade): **the 2D-latency pool is only fungible against event
count while messages STAY EAGER — size byte growth against UCX_RNDV_THRESH before pricing any
event-reduction lever.**

Implied IF ADOPTED at K=8 (opt-in, user's call): 4N 0.6341 ⇒ **7.22×**; 16N 0.2358 ⇒
**5.21×, Stage-2 SYPD ≈ 2.71**. Pre-registered rescue candidates before any promotion talk:
(a) E.4 `UCX_RNDV_THRESH` bump so the wide messages stay eager (could unlock the modeled
−15/−23 ms at K=4), (b) fuse node+sigma segments into ONE message/partner, (c) E.2
interior/boundary overlap. The knob + certification are DONE either way (byte-proven at every K).
