# M7 scaling RE-MEASUREMENT campaign (the paper-figure update)

*2026-07-18, session 14 tail; user directive: "remeasure everything — from 1 node
(CORE2) to 32 nodes for meshes where it makes sense." Baseline of comparison =
the M5.24 campaign table (`docs/SCALING_M524.md`, 2026-05-31) — that table is the
"before"; this campaign is the "after" with the M7 optimizations.*

## The two figures (user decision, 2026-07-18)

- **Figure A — bit-identical class:** configs whose CPU/serial twin is bit-identical
  to the C reference; GPU runs sit at the SAME documented climate-close floor as the
  unoptimized CUDA build. Config: `FESOM_SPEED=1` + `FESOM_SPEED_EVPWIDE=8`
  (byte-certified) **+ `SLURM_CPU_BIND=none` — GATES GREEN 2026-07-18: serial byte
  gate BIT-IDENTICAL (26351255) + CUDA fidelity PASS (26351256) ⇒ unbind is
  figure-A certified; figure A plots the Au legs.**
- **Figure B — climate-identical class (solver-tolerance):** + `FESOM_SPEED_CGPOLY=3`
  + the env package (`UCX_PROTO_ENABLE=y UCX_IB_GPU_DIRECT_RDMA=yes
  UCX_NET_DEVICES=all`), 1-yr certified at the M5.23 bar EXACTLY (26351019).
  Permanent manual knobs / documented recommendation — never the default set.

## Protocol (rule-1 std300, upgraded from M5.24's 35-step)

300 steps, min-of-2 same-alloc, `BIN=cgpoly0` (`ee2c4fdd`, knob-off ≡ h17 proven),
GPU `-C a100_80`, CPU knob-free `build-m7serial`-class binary (bit-identical certified),
per-mesh dt as M5.24 (CORE2 1800 · farc 900 · **dars → dt120/90, rule 0.41 below** · NG5 180), private CORE2 mesh
(L73), provenance md5 per job, L80 announce checks per leg at harvest.

**GPU points: ONE ab_env job per (mesh, N), 4 same-alloc legs:**
`A` = SPEED=1+EVPWIDE=8 · `Au` = A + unbind · `B` = Au + CGPOLY=3 · `Bp` = B + proto
pkg. (Figure A plots A (or Au if gates pass); figure B plots best(B, Bp) with the
per-scale env noted honestly.)

**CPU points: job_m7_scale_cpu (one config, knobs structurally off, NSTEPS=300).**

## The matrix (ranks = 4×N GPU, 128×N CPU; all dists verified present)

| mesh | GPU N | CPU N | notes |
|---|---|---|---|
| CORE2 (private) | 1 2 4 8 | 1 2 4 | GPU-favored ≤2N historically; 8N = the over-decomp tail |
| farc | 1 2 4 8 16 32 | 1 2 4 8 16 | 32N GPU = documented over-decomposition point (keep for honesty) |
| dars | 2 4 8 16 32 | 1 2 4 8 16 32 | 1N GPU impossible (memory) |
| NG5 | 2 4 8 16 32 | 4 8 16 32 | 1N GPU impossible; CPU <4N not run (M5.24 precedent) |

M5.24 baselines to beat (s/step, GPU): CORE2 0.117/0.095/0.112/0.111 · farc
0.309/0.244/0.210/0.190/0.177/0.256 · dars 0.814/0.475/0.344/0.237/0.211 · NG5
2.335/1.273/0.810/0.492/0.374. Expectation (stated before harvest): class A ≈ ×1.9-2.1
faster than M5.24 at NG5/dars mid-scales (the h17-vs-M5.24 factor), less at CORE2/farc
small-N (compute-bound, the levers are comm-side); class B adds the knob+env gains
measured in session 14 (−4..−17 % depending on scale). SYPD@production-dt re-derived
at harvest per the M5.24 method.

## Also in this fleet

- **unbind classification gates**: serial byte gate (SLURM_CPU_BIND=none, knob-free,
  diff vs m6_baseline_serial must be rc=0) + CUDA fidelity gate (FESOM_SPEED=1 +
  unbind). Clean ⇒ unbind is figure-A eligible.
- The CORE2 GPU 4N point doubles as the **63-yr sizing probe** (year wall-time = 
  s/step × 17280; 1-yr @2N recommended-config measured 13:57 ⇒ 63 yr ≈ 15.4 h @2N).

## 63-yr CORE2 runs (user decision 2026-07-18: TWO runs — bit-identical + all-opts)

**Protocol match VERIFIED against the JAX/Fortran pair (port_jax
`CORE2_FORTRAN_SPEC.md` + `HANDOFF-20260630`):** 1958–2020 (63 yr, 1,088,640 steps),
**dt=1800 constant**, JRA55-do v1.4.0 (files through 2020 ✓ on /pool), PHC IC,
CORE2 mesh (our private copy = the pre-corruption original both refs used, L73 ✓),
monthly streams. **⚠️ The R2/JAX physics is the OPTIONS config, not our default:
`zstar + cvmix_TKE + mEVP` (+ GM ON).** Parameter audit — ALL MATCH our port:
ice_diff=10.0 ✓ · alpha=beta_evp=250 ✓ · evp_rheol_steps=120 ✓ · ice_gamma_fct=0.5 ✓
· GM K_GM_max=1000/resscalorder=2/… ✓ (fesom_gm.cpp block mirrors namelist.oce) ·
GM on by default ✓. Fortran reference ON DISK: `/work/ab0995/a270088/fesom2_core2/`
(819 GB, 1450 files, 22 streams × 63 yr ✓ verified).

| run | config (all + `FESOM_MIX_SCHEME=TKE;FESOM_WHICH_EVP=1;FESOM_ALE=zstar`) | class |
|---|---|---|
| **63A** | `FESOM_SPEED=1` + `EVPWIDE=8` + `SLURM_CPU_BIND=none` (unbind gates GREEN) | bit-identical (M6 all3 combined twin = the serial proof) |
| **63B** | 63A + `CGPOLY=3` + env package | climate-identical; the mEVP×CGPOLY threshold-flip floor (s13 §8) gets its strongest arbiter |

**NODE COUNT: 2 (user decision 2026-07-18** after the core2 fleet points showed 2N
fastest for every config — 4N costs ~7-8 % of the years; the sizing table lives in
the harvest). Launch = `sbatch -N2 --ntasks=8 jobs/job_m7_climate63` with
OUTDIR=/work/ab0995/a270088/port2/climate63/{63A,63B}, BIN=cgpoly0, full NSTEPS;
fires MANUALLY once 26351395/96 (combined gates) show PASS text + 26351397 (2-yr
rollover smoke) is clean — no sbatch-dependency automation (a FAILed gate can still
exit 0; the PASS text is the criterion).

**Sizing rules (user, 2026-07-18):**
- **IO-adjusted sizing**: the fleet's std300 legs never cross a month boundary ⇒ they
  EXCLUDE the monthly-write cost. The anchor of record is the 1-yr leg's loop timing
  — **0.0464 s/step @2N Bp-config INCLUDING all 12 monthly writes of 17 streams**
  (801.5 s/yr; init 36 s). Sizing = this anchor × the fleet's (config, N) RATIOS,
  + 10 % margin.
- **Partial-run policy: NO long-QOS, NO restart I/O — submit the FULL 63 yr
  (NSTEPS=1088640) at 12 h walltime and keep whatever completed.** Per-year monthly
  files close at year rollover, so every finished year is valid on disk; only the
  in-flight year at SIGKILL is lost (the 2-yr smoke also validates the rollover file
  handling). Expected @2N Bp: ~52-53 yr; 63A slower ⇒ fewer.
- **⚠️ Planning constraint: the paper's SST/SSS RMSE window is 1980-2009 ⇒ ≥52
  completed years strongly preferred.** Pick N to MAXIMIZE COMPLETED YEARS (not
  raw s/step) from the fleet's core2 points — if M7's comm gains moved the old
  "2N-is-fastest" verdict to 4N, the full 63 may fit outright.

**Prerequisites before launch (in order):**
1. **Combined-options × speed gates** (never certified TOGETHER; L91): gpu_gate ×2
   with the 63A and 63B knob sets, SREF = the M6 `all3_bitid` combined serial ref.
2. **Sizing from the fleet's CORE2 GPU points**: pick N so ONE job ≤ 12 h (no restart
   I/O exists!). 1-yr @2N recommended-config = 13:57 ⇒ 63 yr ≈ 14.7 h @2N — 4N or 8N
   likely fits; the fleet's core2 g4n/g8n legs give the exact per-config s/step.
   If nothing fits 12 h ⇒ long-QOS request or the restart-I/O work item revives.
3. **2-yr smoke under the exact 63A config** (year-rollover + combined-options +
   speed path de-risk; ~40-60 min) before burning a 12-h job.
4. **Storage check**: R2 = 819 GB ⇒ ours ≈ 0.8 TB × 2 runs — verify /work quota
   headroom BEFORE launch (user visibility).

## ⚠️ RULE 0.40 (found by the USER's eye on the smoke figures, 2026-07-18): the
## 17280-step "year" is 360 DAYS, and a truncated run's final monthly record is a
## PARTIAL-MONTH mean — never compare it against a full-calendar reference

The 2-yr smoke (NSTEPS=2×17280) ended 1959-12-22; its "December 1959" monthly file
averages Dec 1-21. vs Fortran's full December that manufactured a hemispherically
ANTISYMMETRIC SST bias (+0.15 NH/−0.23 SH — winter-cooling/summer-warming trends
half-sampled, ice signs flipping likewise) that dominated the 1959 ANNUAL diff map.
All 23 complete months sit at ±0.002-0.009 K (weather noise); excluding the partial
December: NH −0.0005 / SH +0.0023 K, RMS 0.0137. **The model is exonerated; the
comparison convention was at fault.**
- **HARVEST RULE for 63A/63B (and any walltime-guillotined run): DROP the final
  partial month (and treat the final year as partial) before any climatology or
  comparison.** `m7_climate_check_plots.py --trim-final-month K` implements it.
- Footnote: every historical "1-yr" leg (17280 steps) carries a Dec-1-26-partial
  December — internally consistent across the whole M5-M7 record (all legs + the
  C-port refs share the convention), but vs REAL-calendar references (Fortran R2)
  only complete months compare.
- The 63-yr jobs' NSTEPS (63×17280 = 62.1 real yr) is unreachable behind the 12-h
  walltime — harmless; the harvest rule handles the tail either way.

## ⚠️ RULE 0.41 (2026-07-18 late): a stability verdict is only as long as its
## measurement window — the dars "bug" was the M5.24 cold-start CFLz class, not code

The whole dars fleet (every GPU leg incl. `CGPIPE=0`, AND the knob-free legacy CPU
points) dies with the same `CG_kk abort: pp·App = nan` at dt=180 within 300 steps:
GPU at **step 38** (all 4 legs, both reps, two different allocations), CPU legacy at
**steps 189/194/203** (c1/c2/c8n). Config-independent, backend-independent ⇒ NOT a
lever/port bug but the **cold-start CFL blowup M5.24 already documented** ("dt=240 is
CFL-unstable from the cold PHC start on BOTH dars and NG5"; NG5 at dt180 blew at steps
85–160 on fine partitions, root-caused to genuine CFLz≈3 at Gibraltar which the port
rides less robustly than Fortran — `docs/SCALING_M524.md`). "dt180 stable" was only
ever a **35-step statement**; the std300 protocol re-opened it, and dars fails it.
The onset step is roundoff-seeded (CUDA atomics vs serial order, rank count), which is
why every 35-step M5.24 leg, the 35-step discriminator (26353463 — all 4 legs PASS,
speed 0.4005 / ew 0.3949), and the 3-step serial reproducer sat under the wall.
My interim "CGPIPE ring builder breaks on dars" reading is **retracted** (mid-flight
misread of a half-finished discriminator). Byproduct worth keeping: the serial
reproducer (26353625) ran the CGPIPE bitwise selfcheck on dars dist_8 —
**0.000e+00 across all iterations, 3 steps** — the ring machinery is bitwise-correct
on dars. No recertification is needed anywhere.

**Consequence for the matrix:** dars legs move to a smaller measurement dt (probes in
flight: dt120 GPU g2n 26353796, dt90 g2n 26353797, dt120 CPU c1n 26353798, all 300
steps). **dt=120 is the JAX port's dars timestep (user, 2026-07-18) — the precedent
says it should hold; the probe is the confirmation, not the search.**

**PROBE VERDICT (2026-07-19 早): dt120 GREEN everywhere — GPU 300×4 legs×2 reps 0 aborts
(dt90 also green = margin; s/step dt-independent, 0.374 vs 0.377), CPU c1n 300×2 reps
0 aborts (5.9386/5.9389). Full dars ladders re-run at dt120: GPU 26354132-36
(sc_dars_g{2,4,8,16,32}n) · CPU 26354471-76 (sc_dars_c{1,2,4,8,16,32}n, ALL SIX GREEN:
5.947/3.031/1.583/0.839/0.413/0.201 s/step c1→c32).**

**Rule 0.41 second strike — NG5 c32n (dist_4096, dt180): died step ~242 of 300 —
M5.24's own table predicted this partition blows ~155 (uv-guard) at dt180. All other
NG5 points (c4-c16 = dist_512-2048, GPU g2-g32 = dist_8-128) completed 300 clean and
STAND at dt180.**

**dt120 remeasure (26355103) FAILED rc=99 — and the failure mode is DIAGNOSTIC: both
reps COMPLETED 300 steps but ended at uv=6.39 (guard fires post-loop, before the
timing print). Model-time analysis: the uv≈5 crossing sits at ~8 model HOURS at BOTH
dt180 (step ~155-242) and dt120 (step ~250+) ⇒ the dist_4096 instability is a
COLD-START-ADJUSTMENT phenomenon in model time, NOT a per-step CFL that dt cures —
unlike dars, where dt120 genuinely stabilized the full 10-model-hour window (uv sane
through step 300, rc=0). The dt60 attempt (26355306) COMPLETED as predicted — **0.6183/0.6176 s/step, rc=0,
final state uv=3.25 climbing (mid-ramp, would blow ~step 480) and CG it=23
(dt-flattered)**. The number is banked in sc_ng5_c32n with these caveats; whether to
plot it (footnoted "dt60, first 5 model h; CPU flattered ⇒ 32N speedup conservative")
or end the NG5 speedup curve at 16N is a **figure-review decision for the user**. The
M5.24 "before" c32n number was a 35-step measurement and cannot be mixed in under
rule 1.** On a green probe the full dars ladder resubmits at that dt; `m7_scaling_figs.py`
DT_RUN["dars"] updates, and the CG dt-correction to production dt240 is re-derived from
measured iters (the ×1.03 was 180→240). s/step is ~dt-independent (M5.24, user-confirmed),
so cross-mesh comparability is unaffected; the figure footnote states the per-mesh dt.
Fallback if even dt90 blows: dars at 35-step legs, footnoted (the M5.24 "before" is
35-step too — internally consistent).
