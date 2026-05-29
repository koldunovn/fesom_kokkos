# NG5 strong-scaling — GPU vs CPU (2026-05-29)

NG5 mesh (`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5`, **7,402,886 nodes**, **70 levels**),
dt=180 s, 35 steps (5 warmup excluded → 30 timed), JRA55 1958 forcing, PHC winter IC.
GPU = `build-cuda` (Kokkos CUDA, A100×4/node) at `m512-fusion` HEAD (d+b); CPU = `build-serial`
(Kokkos Serial, 128c/node) same HEAD. Internal loop timer; 2 reps each (rep spread < 0.2%).
Jobs: `jobs/job_ng5_scaling_{gpu,cpu}` + `jobs/submit_ng5_scaling.sh`.

⚠️ **NG5 required two deep-mesh bug fixes first** (commit `328536c`) — see § Bugs below. Without
them the model segfaults in init (nl=70 > the old hardcoded 64-level scratch cap) and the step-0
snapshot OOMs rank 0 (~66 GB global gather). The scaling runs use `snap_every=-1` (no I/O).

⚠️ **Binary note**: NG5 ran on `m512-fusion` (M5.12 d+b); CORE2/farc/dars ran on M5.11. The d+b
GPU delta is < 1.3% (CPU Serial unaffected) — negligible for the factor-level GPU/CPU ratio.

## Per-step time (s/step) — mean of 2 reps

| Nodes | GPU (CUDA, A100×4/node) | nod2D/rank | CPU (Serial, 128c/node) | nod2D/rank |
|:-----:|:----------------------:|:----------:|:-----------------------:|:----------:|
|   2   | 30.39 (dist_8)         | 926k       | _OOM_ (dist_256)¹       | 28.9k      |
|   4   | 16.27 (dist_16)        | 462k       | 4.330 (dist_512)        | 14.5k      |
|   8   |  8.698 (dist_32)       | 231k       | 2.239 (dist_1024)       | 7.2k       |
|  16   |  4.514 (dist_64)       | 116k       | 1.171 (dist_2048)       | 3.6k       |

¹ CPU dist_256 = 128 ranks/node × 28.9k nod2D/rank × 70 levels OOM-killed the 256 GB node
  (`OUT_OF_MEMORY`). The densest CPU point doesn't fit at full 128-core packing — the CPU
  analogue of a per-rank memory wall. (Notably the GPU's dist_8 — 926k nod2D/rank — DID fit
  8×A100 80 GB at 2 nodes; the GPU has the per-node memory edge at the dense end.)

## Per-doubling strong-scaling efficiency

|             | GPU 2→4 | GPU 4→8 | GPU 8→16 | CPU 4→8 | CPU 8→16 |
|:------------|:-------:|:-------:|:--------:|:-------:|:--------:|
| ratio       | 1.87×   | 1.87×   | 1.93×    | 1.93×   | 1.91×    |
| efficiency  | 93%     | 94%     | 96%      | 97%     | 96%      |

GPU 2→16 (8×): **6.7× (84%)**. CPU 4→16 (4×): **3.70× (92%)**. Both strong-scale well
(93–97% per doubling) — like dars, NG5 has enough per-rank work that neither halo nor launch
overhead dominates in the scaling regime.

## Node-for-node ratio (the headline)

| Nodes | GPU/CPU per node                          |
|:-----:|:------------------------------------------|
|   4   | **3.76×** slower (16.27 / 4.330)          |
|   8   | **3.88×** slower (8.698 / 2.239)          |
|  16   | **3.86×** slower (4.514 / 1.171)          |

**NG5 GPU/CPU ≈ 3.8×**, stable across 4/8/16 nodes (3.76–3.88×). The 2-node pair is missing on
the CPU side (dist_256 OOM); extrapolating CPU dist_256 from the ~96% CPU scaling (~8.4 s/step)
gives ~3.6× at 2N — consistent.

## The mesh-size trend (the lever from `docs/SCALING_DARS.md`, now extended)

| Mesh   | nodes | levels | GPU/CPU (node-for-node) | GPU strong-scale |
|:-------|------:|:------:|:-----------------------:|:----------------:|
| CORE2  | 127k  | 47     | **8.9×** (4N)           | 81 %             |
| farc   | 638k  | 48     | **5.5×**                | 91 %             |
| dars   | 3.16M | 47     | **4.1×**                | 97 %             |
| **NG5**| 7.4M  | 70     | **3.8×**                | **93–96 %**      |

**The mesh-size lever is real but ASYMPTOTING.** The GPU/CPU gap keeps shrinking with mesh size,
but the increments collapse: 8.9 → 5.5 (−3.4), 5.5 → 4.1 (−1.4), 4.1 → 3.8 (**−0.3**). Doubling
the mesh from dars (3.16M) to NG5 (7.4M) — and adding 49 % more levels (47→70) — bought only
~0.3× more. **The floor for this FESOM-Kokkos port on Levante A100 is ~3.6–3.8× slower per node
than EPYC**, reached by dars-class meshes. Bigger meshes than NG5 will not meaningfully close it.

## Per-rank vs OMEGA's A100 sweet spot (~250k nod2D/rank)

| Run (GPU)   | nod2D/rank | vs OMEGA sweet spot |
|:------------|-----------:|--------------------:|
| dist_8 (2N) | 926k       | 3.7× above          |
| dist_16 (4N)| 462k       | 1.8× above          |
| dist_32 (8N)| 231k       | ~at the sweet spot  |
| dist_64(16N)| 116k       | 2.2× below          |

The 3.8× ratio holds **constant** from 926k down to 116k nod2D/rank — unlike CORE2 (ratio
degraded as per-rank shrank below the launch-overhead floor). NG5 keeps the GPU fed across the
whole matrix, so the 3.8× is the intrinsic per-node compute ratio, not a feeding artifact.

## Per-watt (Levante: GPU node 2160 W, CPU node 560 W → 3.86× power)

| Mesh   | throughput ratio | per-watt vs CPU |
|:-------|-----------------:|----------------:|
| CORE2  | 1/8.9            | 1/27            |
| dars   | 1/4.1            | 1/15.8          |
| **NG5**| 1/3.8            | **1/14.8**      |

Per-watt continues to improve with mesh size (1/14.8 at NG5) but, like the throughput ratio, is
asymptoting. Per-watt parity remains unreachable on Levante A100 for full-physics ocean+ice.

## Bugs NG5 exposed (fixed, commit `328536c`)

1. **Stack-buffer-overflow** (`nl=70` > hardcoded 64-level per-column scratch): `bulk_0[64]` etc.
   in eos (host+device) + `NL_MAX`/`KPP_NL_MAX=64` enums in momentum/gm/kpp/tracer_adv/tracer_diff.
   Fixed: project-wide `FESOM_MAX_LEVELS=128` (`fesom_types.h`) + load-time `nl <= FESOM_MAX_LEVELS`
   guard. Bit-identity preserved (pi per-kernel verify max|Δ|==0). **Any ≥48-level mesh hit this.**
2. **Rank-0 I/O gather OOM** (~66 GB at NG5): the step-0 initial snapshot. Scaling runs use
   `snap_every=-1` (disable). Production NG5 snapshots need parallel I/O / single-buffer-reuse.

## Memory / stability notes

- **dist_8 GPU runs** (926k nod2D/rank fits 8× A100 80 GB at 2 nodes) — the earlier "dist_8 OOM"
  was the host I/O gather, not device memory.
- **dist_256 CPU OOMs** at 128 ranks/node (28.9k nod2D/rank × 70 lvl × 128 > 256 GB). To get a
  2-node CPU point, run dist_256 at ≤64 ranks/node (≥4 nodes) — but that breaks node-for-node.
- All runnable points stable over 35 steps (T ∈ [0.4, 30.2] °C, S ∈ [30.8, 36.6] PSU); no NaN.

## Per-phase profile — THE OPTIMIZATION REGIME SHIFTS WITH MESH SIZE (2026-05-29)

Re-profiled NG5 GPU with `FESOM_STEP_PROFILE=1` (per-phase + Kokkos per-kernel) at dist_16
(462k nod2D/rank — well above the OMEGA sweet spot). dist_8 (926k/rank) OOM'd the A100 in
profile mode (`kpp.viscA` alloc; the scaling run barely fit → 926k/rank IS the 80 GB device
ceiling). Job `25225499`; loop 16.29 s/step (profile mode, ~fence overhead), ocean **84.6 %**.

| phase            | CORE2 dist_8 %loop | NG5 dist_16 %loop | NG5 abs s/step |
|:-----------------|-------------------:|------------------:|---------------:|
| **13_fct**       | 17.4 %             | **22.8 %**        | 3.708          |
| 1b_gm            | 9.8 %              | 11.2 %            | 1.818          |
| 12_ale           | 6.6 %              | 9.1 %             | 1.483          |
| 3_mixing (KPP)   | 6.8 %              | 8.3 %             | 1.348          |
| 13b_trdiff       | 5.7 %              | 7.2 %             | 1.173          |
| 4_velrhs         | 5.2 %              | 7.2 %             | 1.165          |
| 6_ivisc          | 4.4 %              | 5.9 %             | 0.968          |
| 1_eos            | 3.2 %              | 4.5 %             | 0.733          |
| **7_ssh (CG+halo)**  | **10.9 %**     | **4.5 %**         | 0.730          |
| **ice_dyn (EVP halo)** | **11.3 %**   | **3.2 %**         | 0.522          |
| CG share         | 5.7 %              | **0.3 %**         | —              |

**The regime flips from launch/halo-bound (CORE2) to compute/bandwidth-bound (NG5).** At CORE2
the dominant phases were **SSH (CG+halo, 10.9 %) and ice_dyn (EVP per-subcycle halo, 11.3 %)** —
both halo/launch-overhead-bound, growing with rank count (M5.12's targets). At NG5 those collapse
to **4.5 % and 3.2 %** (CG just 0.3 %), and the **compute-heavy 3-D ocean kernels take over**:
FCT + GM + ALE + KPP + tracer-diff + momentum-RHS + impl-visc ≈ **72 %** of the step.

**Proof it's compute/bandwidth-bound, not overhead:** the FCT phase scaled **41× (0.090 → 3.708
s/step) for 43× more per-rank work** — essentially linear. At CORE2 only ~5.5 % of the step was
in actual FCT kernels (rest = host/fence/halo between launches); at NG5 it's genuinely the GPU
grinding through 32 M nod3D cells/rank.

### Implication — the M5.12 conclusion is REGIME-SCOPED

The M5.12 launch-fusion levers (f reverted, d +0.7 %, b +0.6 %; L53–L55) optimized the **CORE2
development regime**, where launch density + halos dominate. **At production scale (dars/NG5)
they are nearly irrelevant** — the wall is per-kernel compute/bandwidth in FCT/GM/ALE/diffusion/
momentum. **The right production lever is the one M5.10b/Lever C tried and abandoned at 1.3 % —
the rank-1 → `View<double**>` coalesced memory layout + OMEGA's vertical-scratch TeamPolicy —
which failed precisely because CORE2's kernels were NOT bandwidth-bound. At NG5 they are.** A
scoped Lever-C retry on the FCT/GM hot kernels, validated at dars/NG5 (not CORE2), is the
M5.13 candidate. **Next: `ncu` on the top NG5 kernels** (FCT, GM) for the bound labels
(% peak DRAM bandwidth, achieved occupancy, stall reasons) to size the opportunity.

---

*Generated 2026-05-29. Companion: `docs/SCALING_CORE2.md`, `docs/SCALING_FARC.md`,
`docs/SCALING_DARS.md`, `docs/PROFILE_M59.md`. Figures: `docs/figures/scaling_*.png`,
`docs/figures/profile_regime_shift.png` (CORE2-vs-NG5 per-phase composition).*
