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
(dt-flattered)**. **USER DECISION (2026-07-19): "we need 32 nodes, add it" — the dt60
point IS the 32N NG5 CPU number**, footnoted on the speedup figure (biases run
conservative: dt-flattered CPU ⇒ 32N speedup understated). Sanity: 0.6176 vs 16N's
1.2096 = 2 % off perfect halving — on-curve. The M5.24 "before" c32n number was a
35-step measurement and cannot be mixed in under rule 1.**

## Figure caption notes (user 2026-07-19: caveats go in the CAPTION, not on the figure)

For the speedup figure caption: *"The NG5 32-node CPU point was measured at dt=60 s
(the finest partition is unstable past ~8 h of model time from a cold start at larger
timesteps); the reduced timestep slightly lowers the per-step solver cost of the CPU
run, making the 32-node NG5 speedup a conservative estimate."* For the scaling
figures: *"dars and NG5 throughput (SYPD) is reported at the production timestep
(240 s) from runs measured at 120/180 s; time-per-step is timestep-independent."*

## Work item (pre-JUPITER): fix the cold-start vertical-robustness gap (wsplit)

**⭐ DISCRIMINATOR ANSWERED (2026-07-19 morning, probes 26360443/44 — Fortran fesom.x,
NG5 dist_4096, dt180, cold PHC start, 300 steps):**
- **F0 (use_wsplit=.false.): DIED — rc=1, NaN** after 261 CFLz warnings. Even Fortran
  does not survive the 300-step window without wsplit.
- **F1 (use_wsplit=.true.): COMPLETED — rc=0, zero NaN**, riding 334 CFLz warnings.
**⟹ wsplit IS the cure (matches production practice: large meshes always run it).
And rule 0.41 strikes a THIRD time: M5.24's "port less robust than Fortran" verdict
was a window artifact — Fortran-no-wsplit completed 205 steps then but dies before
300; the port dies ~242. SAME robustness class. The only real port↔Fortran gap is
that the port's wsplit implementation is buggy (M5.24: CG NaN ~step 85 when enabled).
The work item reduces to: debug port-wsplit to bit-match Fortran-wsplit (M6 ladder:
Fortran wsplit reference → C oracle wsplit-on → Kokkos twin → gates).** The Fortran
reference is now one submission away (`job_m524_scale_fortran` + `WSPLIT=true`,
committed `161b25d`); F1's WORK dir holds a complete wsplit-on NG5 run for behavioral
reference. Payoff after the fix: cold-start legs at production dt on every mesh
(dars dt240, NG5 dt240) + JUPITER-scale cold starts.

*User 2026-07-19: "would be nice to diagnose the error and fix it." Not blocking the
paper figures; becomes IMPORTANT before JUPITER (GH200 scale-out ⇒ finer partitions ⇒
this wall everywhere at cold start).*

- **The diagnosis largely EXISTS (M5.24):** cold PHC start drives genuine CFLz≈3 at
  the Gibraltar/Med outflow (glon/glat −4.81/36.01, from ~step 4). Fortran
  (wsplit=.false.) rides it and completes; the port under identical CFLz ramps uv→>5
  and dies; the port's wsplit implementation does NOT help (near-identical ramp then
  CG NaN ~step 85 — the fesom_ale.c:88 signature) and is disabled ("wsplit on
  diverged the C from the stable Fortran path", fesom_constants.h:54). Partition
  fineness is the AMPLIFIER (dist_2048 clean / dist_4096 ~155-242 / dist_8192 ~10),
  not the cause. ⇒ the gap is in the port's implicit vertical advection / wsplit
  path vs Fortran's.
- **Fix shape:** repair the port's wsplit to actually match Fortran's algorithm;
  ship as an OPT-IN knob (OFF = today's bit-identical path untouched — no recert of
  the existing matrix); certify against a Fortran wsplit-ON reference run; then
  cold-start NG5/dars at production dt should complete like Fortran R2 does (its
  205-step dist_4096 proof exists). Full options ×3 ladder for the knob per L91.
- **Iteration is CHEAP (measured 2026-07-19):** 32-node compute jobs at short
  walltime started after 14 s / 11 min / 7 min queue wait; one rep to the dt180
  blowup ≈ 8 min wall (4 min init + 3 min to step ~242). CPU-side reproducible —
  no GPU queue needed. Optional cheaper still: oversubscribe dist_4096 on 8-16
  nodes for correctness-only iterations.

**⭐ ROOT CAUSE FOUND (2026-07-19 evening, code-level — no run needed):** the port's
FCT tracer path is missing Fortran's wsplit machinery (`oce_adv_tra_driver.F90`):
1. FCT **low-order** = `adv_tra_ver_upw1(we)` explicit + **under `use_wsplit` an
   IMPLICIT vertical advection of the low-order field with `wi`:**
   `call adv_tra_vert_impl(dt, wi, fct_LO)` (line 286). **The port has NO
   `adv_tra_vert_impl` at all** — the `w_i` share of tracer transport is silently
   DROPPED at every CFL-limited cell when wsplit is on.
2. FCT **high-order** vertical advection uses `pwvel => w` — the FULL velocity
   (line 315); the port feeds `dyn->w_e` (fesom_tracer_adv.cpp:637/715/1421).
Both errors are exactly zero at wsplit-OFF (`w_e=w, w_i=0`) — invisible to every
certification ever run; at wsplit-ON they give the slow tracer/density error at
Gibraltar-class cells → the CG-NaN-~step-85 signature M5.24 observed. (The non-FCT
`do_wimpl` implicit-diffusion path is irrelevant: `oce_ale_tracer.F90:617` forces it
off for FCT configs.) The momentum-side `w_i` TDMA EXISTS in the port (substep 6)
but is unexercised — verify against Fortran during cert. The M5.14 w_i-residency
assumptions (no-init-push fesom_step.cpp:676; L3 same-kind halo fuse) need re-audit
once `w_i ≠ 0`. Fix = port `adv_tra_vert_impl` + high-order `w_e→w` + runtime knob
`FESOM_WSPLIT` (default off ⇒ byte-identical) + the gate ladder above. On a green probe the full dars ladder resubmits at that dt; `m7_scaling_figs.py`
DT_RUN["dars"] updates, and the CG dt-correction to production dt240 is re-derived from
measured iters (the ×1.03 was 180→240). s/step is ~dt-independent (M5.24, user-confirmed),
so cross-mesh comparability is unaffected; the figure footnote states the per-mesh dt.
Fallback if even dt90 blows: dars at 35-step legs, footnoted (the M5.24 "before" is
35-step too — internally consistent).

---

## CAMPAIGN CLOSED — FINAL TABLE (2026-07-22)

The full fleet landed (last points: dars g16 26354135, dars g32 26354136, NG5 g32
26351276, farc g32 26351266 + retry 26373406). All L80 announce checks pass. Figures:
`/work/ab0995/a270088/port2/m7/scaling_figs/` (regenerated with shared below-panel
legend + measured dt-corrections).

### GPU env-A/B fleet, s/step (min-of-2, same-allocation; A=SPEED+EVPWIDE ref,
### Au=+unbind, B=+CGPOLY3, Bp=+UCX proto trio)

| point | A | Au | B | Bp |
|---|---|---|---|---|
| dars g16n (64 GPU, dt120) | 0.1271 | 0.1213 | 0.1146 | 0.0939 (−26.1 %) |
| dars g32n (128 GPU, dt120) | 0.1164 | 0.1119 | 0.1023 | 0.0807 (−30.7 %) |
| NG5 g32n (128 GPU, dt180) | 0.1978 | 0.1917 | 0.1810 | 0.1435 (−27.5 %) |
| farc g32n (128 GPU, dt900) | 0.1396 | 0.1407 | 0.0945 (−32.3 %) | **HANG (×2)** |

- **Bp gains GROW with scale** (dars: −1..−5 % small counts → −26/−31 % at 16/32N) —
  consistent with the s14 per-partner-toll mechanism: the proto trio attacks exactly
  the comm overhead that dominates at high rank counts.
- ⚠️ **farc-32N proto hang is REPRODUCIBLE**: Bp force-terminated on two independent
  allocations (26351266, 26373406) while the same allocations ran A/B clean (0.1396/
  0.1397). CAVEAT on the E.T1 env recommendation: do not enable `UCX_PROTO_ENABLE=y`
  on farc at ≥128 ranks. Figure B unaffected (min(B,Bp)=B=0.0945).

### CG dt-correction (SYPD footnote) — MEASURED, replaces the old ×1.03 estimate

Same-allocation dt pairs (`jobs/job_m7_dtpair`, frozen cgpoly0, class-A env, g2n):
- **NG5 dt180→240: ×1.0110** (35-step pair 26378115: 1.1967→1.2099; iters 89.4→119.5).
- **dars dt120→240: ×1.0222** (median of three 10-step pairs 26386487/26387609/26387610:
  1.0222/1.0214/1.0277; iters 50.1→96.1 bit-identical across pairs. dt240 cold dies at
  step 12 — rule 0.41 — so 10-step legs are the longest measurable window; the three
  dt240 minima spread 0.15 %).
- Applied in `m7_scaling_figs.py` (`DT_CORR`); footnote updated. Correction measured on
  GPU g2n and applied to all sypd rows (CPU bias second-order; s/step figures raw).

### Ops notes from the close-out
- gpu-partition node-pair l[50190,50193] hung a run at post-init (log frozen 15 min,
  job 26385842); excluded via `-x` on the retries. Second hang class of the night after
  the farc proto hang — both env/fabric, neither touches a banked number.
- (from the M8 track) snapshot gather at 4096 ranks on NG5 blows UCX memory registration
  (`ibv_reg_mr EFAULT`) — the m7 jobs are immune ONLY because they pass snap_every=-1
  (no step-0 write). Keep `-1` in every ≥4096-rank job; carry into the JUPITER plan.
