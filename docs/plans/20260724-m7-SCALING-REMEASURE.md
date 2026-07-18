# M7 scaling RE-MEASUREMENT campaign (the paper-figure update)

*2026-07-18, session 14 tail; user directive: "remeasure everything — from 1 node
(CORE2) to 32 nodes for meshes where it makes sense." Baseline of comparison =
the M5.24 campaign table (`docs/SCALING_M524.md`, 2026-05-31) — that table is the
"before"; this campaign is the "after" with the M7 optimizations.*

## The two figures (user decision, 2026-07-18)

- **Figure A — bit-identical class:** configs whose CPU/serial twin is bit-identical
  to the C reference; GPU runs sit at the SAME documented climate-close floor as the
  unoptimized CUDA build. Config: `FESOM_SPEED=1` + `FESOM_SPEED_EVPWIDE=8`
  (byte-certified). `SLURM_CPU_BIND=none` joins figure A iff its byte+fidelity gates
  come back clean (submitted with this fleet), else it belongs to B.
- **Figure B — climate-identical class (solver-tolerance):** + `FESOM_SPEED_CGPOLY=3`
  + the env package (`UCX_PROTO_ENABLE=y UCX_IB_GPU_DIRECT_RDMA=yes
  UCX_NET_DEVICES=all`), 1-yr certified at the M5.23 bar EXACTLY (26351019).
  Permanent manual knobs / documented recommendation — never the default set.

## Protocol (rule-1 std300, upgraded from M5.24's 35-step)

300 steps, min-of-2 same-alloc, `BIN=cgpoly0` (`ee2c4fdd`, knob-off ≡ h17 proven),
GPU `-C a100_80`, CPU knob-free `build-m7serial`-class binary (bit-identical certified),
per-mesh dt as M5.24 (CORE2 1800 · farc 900 · dars 180 · NG5 180), private CORE2 mesh
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
| **63A** | `FESOM_SPEED=1` + `EVPWIDE=8` (+unbind if gates green) | bit-identical (M6 all3 combined twin = the serial proof) |
| **63B** | 63A + `CGPOLY=3` + env package | climate-identical; the mEVP×CGPOLY threshold-flip floor (s13 §8) gets its strongest arbiter |

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
