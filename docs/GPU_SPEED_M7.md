# M7 — speed beyond bit-identity (pure FP64): campaign log

*Plan: `docs/plans/20260714-m7-speed-fp64.md`. Branch `m7-speed` off `main@69e506d`. Goal: GPU-node/CPU-node ratio ≥5× (stretch 8×) at NG5/dars 4–8N (Stage 1), flatten the decay toward 16N (Stage 2), pure FP64, no repartitioning. Bit-identity replaced by the two-level gate (per-lever fidelity+A/B; per-tier 1-yr climate; knob-OFF byte-identical always).*

## Gate definitions

- **Knob-OFF byte gate:** `jobs/job_m7_gate_serial` — `build-m7serial` CORE2 dist_8 (private mesh, dt1800, 20 steps, snap 10) `diff_snap.py` vs `/work/ab0995/a270088/port2/m6_baseline_serial` → rc=0 required at every commit.
- **FORCE_SERIAL byte proof** (bit-id-claimed levers): same run with `FESOM_SPEED_FORCE_SERIAL=1` + the lever knob → still rc=0.
- **CUDA fidelity gate:** `scripts/gpu_fidelity_gate.sh` (CORE2 dist_8 ice-active, established ceilings; climate-close floor ~1e-3).
- **Tier climate gate:** 1-yr CORE2 CUDA, tier knobs ON, `m32_climate_compare.py --cref-frame` vs the certified C oracle at the M6/L79 floors.
- **A/B rule:** same-day, same-allocation, both binaries in one job (`job_dars_l3_ab` pattern); 35 steps, 2 reps, min; dt180; ±10% inter-allocation noise makes anything else meaningless.

## Ratio ledger

Baseline anchors are re-measured same-day (row 0), not inherited from `SCALING_M524.md`.

| after | NG5@4N GPU | NG5@4N CPU | ratio | NG5@8N GPU | NG5@8N CPU | ratio | dars@8N GPU | dars@2N GPU | notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| M5.24 (ref, 2026-05-31) | 1.273 | 4.599 | 3.61 | 0.810 | 2.356 | 2.91 | 0.344 | 0.814 | historical, NOT the Δ-anchor |
| **row 0: m7 baseline** (2026-07-14, jobs 26234869–75, min of 2 reps) | 1.2796 | 4.6005 | **3.60** | *queued* | 2.3624 | — | *queued* | 0.8177 | dars CPU@8N = 0.8563; the two GPU 8N legs (26234870/72) were still PD at handoff — harvest from `m7/base_*/log_rep_*.txt`. All five finished legs reproduce M5.24 within noise. |

SYPD@dt240 = 0.657 / (s/step at dt180) × (1/1.03 CG correction) for NG5.

## Stall budget (Task 0.3 — the ~25–30% remainder, decomposed)

| component | NG5@4N ms/step | dars@8N ms/step | source |
|---|--:|--:|---|
| fence drains (post-unpack) | — | — | pending |
| fence drains (pre-MPI) | — | — | pending |
| kernel-launch gaps | — | — | pending |
| host segments between phases | — | — | pending |
| cudaDeviceSynchronize stalls | — | — | pending |

Per-step sync counters (`FESOM_SPEED_SYNCSTATS=1`): exchanges/step by kind, fences/step, pack/unpack launches/step — pending.

## Roofline, top-10 kernels (Task 0.4)

| kernel | % step | achieved GB/s | % of 1.9 TB/s | SM util | occupancy | local-mem (spill) | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| fct_eud_fill | 2.47 | — | — | — | — | — | pending |
| impl_vert_diff_tracers | 2.29 | — | — | — | — | — | pending |
| fct_mfct_h | 2.26 | — | — | — | — | — | pending |
| gm_redi_ver_node | 2.06 | — | — | — | — | — | pending |
| ale_vvel_scatter | 1.94 | — | — | — | — | — | pending |
| fct_zal_a34 | 1.47 | — | — | — | — | — | pending |
| fct_zal_b3h | 1.16 | — | — | — | — | — | pending |
| gm_redi_hor_edge | 1.04 | — | — | — | — | — | pending |
| impl_vert_visc | 0.98 | — | — | — | — | — | pending |
| pressure_bv (EOS) | — | — | — | — | — | — | pending |

## Knob registry

| knob | lever | class | status |
|---|---|---|---|
| `FESOM_SPEED` | master switch (all blessed levers) | — | scaffolded |
| `FESOM_SPEED_FORCE_SERIAL` | dev-only: allow levers on Serial (byte proofs) | — | scaffolded |
| `FESOM_SPEED_SYNCSTATS` | per-step sync counters (diagnostic) | — | pending (0.3) |
| `FESOM_SPEED_NOFENCE2` | drop post-unpack halo fence | bit-id | pending (1.1) |
| `FESOM_SPEED_CGSLIM` | CG iteration-body slimming | bit-id | pending (1.2) |
| `FESOM_SPEED_FCT2` | FCT T+S tracer batching | bit-id | pending (1.3) |
| `FESOM_SPEED_EVPCOMPACT` | EVP active-set compaction | bit-id | pending (1.4) |
| `FESOM_SPEED_SCATTER` | scatter de-atomization (1=coloring, 2=store+gather) | rounding | pending (2.1) |
| `FESOM_SPEED_TDMA` | TDMA spill-kill (1=recompute-in-sweep, 2=PCR) | bit-id / rounding | pending (2.3) |
| `FESOM_SPEED_CG1R` | CG single-Allreduce (Chronopoulos–Gear); **supersedes CGSLIM when both set** (carries its fusions) | solver | pending (3.1) |
| `FESOM_SPEED_CGPOLY` | Chebyshev polynomial preconditioner (=degree) | solver | pending (3.2) |
| `FESOM_SPEED_EVPWIDE` | comm-avoiding wide-halo EVP (=k rings) | solver | pending (3.3) |
| `FESOM_SPEED_ICELAG` | lagged ice–ocean coupling | physics | reserve (4.1) |
| `FESOM_SPEED_EVPTHIN` | EVP halo thinning (stale ring) | physics | reserve (4.2) |

## Lever log

*(one entry per landed/killed lever: what, gates, same-alloc A/B numbers, verdict)*

## Tier climate gates

| tier | knobs ON | 1-yr climate vs C oracle | NG5@16N direct | tag |
|---|---|---|---|---|
| 0 | none | n/a | — | `m7.0-baseline` (pending) |
