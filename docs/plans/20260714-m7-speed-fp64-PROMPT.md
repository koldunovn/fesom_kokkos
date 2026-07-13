# M7 HANDOFF — resume the speed campaign (next session kickoff)

*Written 2026-07-14 at the end of the planning+scaffolding session. The plan of record is
`docs/plans/20260714-m7-speed-fp64.md` — READ IT FIRST; this file only says where things stand
and what to do next. Campaign log + ledger: `docs/GPU_SPEED_M7.md`.*

## What this campaign is (one paragraph)

Raise the GPU-node/CPU-node ratio from ~3.6× to **≥5× (stretch 8×) in pure FP64** (mixed/single
precision banned by user), breaking bit-identity to the C port where profitable. User-validated
decisions: staged targets (NG5/dars @4–8N first, then flatten the decay toward 16N); lever classes
allowed = rounding-level numerics + solver-path + (reserve) physics-visible scheduling;
**repartitioning is OUT** (user tried it, no gain); acceptance = two-level gate (per-lever same-day
fidelity gate + same-alloc A/B; per-tier 1-yr CORE2 climate vs the certified C oracle; tier-4 levers
individual sign-off) with **knob-OFF byte-identical at every commit** and **Serial-backend-always-legacy**.
Tiers: 0 measure → 1 bit-id (fence surgery, FCT T+S batch, EVP compaction) → 2 rounding
(scatter de-atomization, TDMA spill-kill) → 3 solver (CG 1-reduce + Chebyshev precond, wide-halo EVP)
→ 4 physics reserve → layout big-bet only if 8× demanded.

## State at handoff (all on branch `m7-speed`, NOT pushed)

| thing | state |
|---|---|
| Plan | `docs/plans/20260714-m7-speed-fp64.md` (Task 0.1 ✅, 0.2 in flight, 0.3/0.4 next) |
| Builds | `build-m7cuda/` + `build-m7serial/` built from certified source (`-ffp-contract=off`, `sm_80` verified). Certified `build-cuda/`, `build-serial/`, `build-m6oracle/` untouched. |
| Knob helper | `src/fesom_speed.hpp` (not yet #included anywhere — first consumer is Task 1.1/0.3) |
| Run tree | `/work/ab0995/a270088/port2/m7/` |
| Byte gate | **PASS rc=0** (job 26234850, `m7/gate_t01_scaffold`) — fresh build == `m6_baseline_serial` byte-for-byte |
| Baseline jobs | 26234869 NG5-GPU-4N · 26234870 NG5-GPU-8N · 26234871 dars-GPU-2N · 26234872 dars-GPU-8N · 26234873 NG5-CPU-4N · 26234874 NG5-CPU-8N · 26234875 dars-CPU-8N. At session end: CPU running, GPU 4N/2N running, GPU 8N×2 pending. Early numbers match M5.24 (dars CPU@8N 0.856; NG5 CPU@8N 2.362; NG5 GPU@4N 1.280). |
| Jobs added | `jobs/job_m7_gate_serial`, `jobs/job_m7_scale_gpu` (has `KNOBS` env for later A/Bs), `jobs/job_m7_scale_cpu`, `jobs/submit_m7_baseline.sh` |
| Plan review | DONE, verdict "needs light revision", **all findings folded into the plan** (2026-07-14): CG knob precedence (CG1R supersedes CGSLIM; CGPOLY composes), NOFENCE2↔ICELAG stream dependency + combined racecheck in 5.1, fence-removal audit extended (buffer reuse, `:286` comment), racecheck for coloring, same-day 16N anchor added to 0.2 (dars proxy unfaithful for CG iters), EVPCOMPACT order-preserving scan + per-subcycle coastal BC, F1 standalone 1-yr climate, FORCE_SERIAL backend-identical precondition, dt240 correction re-derivation at 3.4, scatter variant 2 built first, FCT2 memory 3–5 GB/rank. |
| User's own jobs | `bench_ng5_halo` / `bench_dars_hal` (26232224/25) were pending in the gpu queue — not ours, don't touch. |

## ⚠️ The one live trap

**Do NOT rebuild `build-m7cuda`/`build-m7serial` while any 262348xx job is PENDING/RUNNING** — queued
jobs execute the binary on disk at start time, and ledger row 0 must be the certified-source binary.
Check `squeue -u a270088` first. The first rebuild is Task 1.1's (bundle the `FESOM_SPEED_SYNCSTATS`
counters into it — see plan Task 0.3 note).

## First actions next session (in order)

1. `git status` / confirm on `m7-speed`; read plan + `docs/GPU_SPEED_M7.md`.
2. Harvest the finished baseline runs: `grep -h 'loop timing' /work/ab0995/a270088/port2/m7/base_*/log_rep_*.txt`
   → complete ledger row 0 (min of 2 reps each; compute the two NG5 ratios).
3. Task 0.2 last checkbox: run the CORE2 CUDA fidelity gate on the m7cuda binary
   (`./scripts/gpu_fidelity_gate.sh`, or the m6-style pair of jobs) — healthy-tip check.
4. Task 0.3: create + submit the two nsys jobs (adapt `jobs/job_nsys_ng5`: keep `-t cuda,mpi,nvtx`,
   add sqlite export for launch-gap analysis; make a dars dist_32 @8N variant). They can run on the
   row-0 binary — no rebuild needed.
5. Task 0.4: create + submit the ncu top-10 job (adapt `jobs/job_ncu_top5`: regex
   `fct_eud_fill|impl_vert_diff_tracers|fct_mfct_h|gm_redi_ver_node|ale_vvel_scatter|fct_zal_a34|fct_zal_b3h|gm_redi_hor_edge|impl_vert_visc|pressure_bv`,
   CORE2 **private mesh** (`/work/ab0995/a270088/port2/mesh/core2`, dist_1 exists), np=1 on gpu-devel
   (30-min limit! → use `--launch-count ~40`, 2 steps, or split into 2–3 jobs / use `-p gpu`),
   binary `build-m7cuda`).
6. Fill the stall-budget + roofline tables in `docs/GPU_SPEED_M7.md`; write the 0.3 decision note
   (rank Tier-1A sub-items by measured stall shares); tag `m7.0-baseline`.
7. Proceed to Task 1.1 (fence surgery + SYNCSTATS counters, one rebuild) per the plan.

## Standing rules (do not relearn these the hard way)

- Same-day, same-allocation A/B only (±10% node noise) — `job_dars_l3_ab` pattern; the new
  `job_m7_scale_gpu` takes `KNOBS="FESOM_SPEED_X=1;..."` for knob-ON legs.
- CORE2 = private mesh always (L73). Climate compares: r2g frame + L79 per-scheme floors.
- Every commit: knob-OFF byte gate green (`M7_TAG=<tag> sbatch jobs/job_m7_gate_serial`).
- Bit-id-claimed levers additionally: `FESOM_SPEED_FORCE_SERIAL=1` byte proof.
- Kill-fast: flat A/B in the lever's payoff regime → revert same day.
- Output under `/work/ab0995/a270088/port2/m7/` only; nothing pushed unless the user asks.
