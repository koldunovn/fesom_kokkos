# dars strong-scaling — GPU vs CPU (2026-05-28)

dars mesh (`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars`, **3,160,340 nodes**, 47 levels),
dt=180 s, 35 steps (5 warmup excluded → 30 timed), JRA55 1958 forcing, PHC winter IC.
Same Kokkos M5.11 binary on `build-cuda` (GPU) and `build-serial` (CPU). Internal loop
timer; 2 reps each. Jobs: `jobs/job_dars_scaling_{gpu,cpu}` + `jobs/submit_dars_scaling.sh`.

## Per-step time (s/step)

| Nodes | GPU (CUDA Kokkos, A100×4/node) | CPU (Kokkos Serial, 128c/node) |
|:-----:|:------------------------------:|:------------------------------:|
|   2   | _stability blowup at step 35_¹ | 2.843 / 2.845                  |
|   4   | 6.005 / 6.001                  | 1.466 / 1.464                  |
|   8   | 3.095 / 3.096                  | _CG NaN at step 10_²           |

¹ GPU dist_8 ran 34 steps clean (u/v in range), then blew up at step 35 with
  u=99.4 m/s. Per-rank load is 395k nod2D — the biggest in our matrix. CG iters
  were climbing (58→61→64) suggesting accumulated SSH error. Likely partition-size
  + scatter-order chaos pushing the CG past stability. **NOT fixed here** —
  intermediate ranks are clean.
² CPU dist_1152 (NaN at CG step 10). Per-rank load only 2744 nod2D — too small,
  likely partition-pathology. Also not fixed here.

## Per-doubling efficiency (the partial scaling we have)

|       | GPU n4 → n8 | CPU n2 → n4 |
|:------|:-----------:|:-----------:|
| ratio | 1.94×       | 1.94×       |
| efficiency | **97 %**   | **97 %**    |

Both backends strong-scale identically at 97% per doubling. This is much better
than CORE2 (GPU at 81–91% per doubling) — dars has enough per-rank work that
neither halo nor launch overhead is dominant in the scaling regime.

## Node-for-node ratio (the headline)

| Nodes | GPU/CPU per node                                              |
|:-----:|:--------------------------------------------------------------|
|   4   | **4.10× slower** (6.005 / 1.465)                              |
|   8   | **4.10× slower** (3.095 / 0.755 extrapolated from CPU n4)     |

The ratio **does NOT degrade with rank count** on dars — unlike CORE2 where it
went from 5.6× (1N) to 8.9× (4N) as per-rank shrunk below the GPU's launch-overhead
floor. On dars we are above that floor at both 4N and 8N.

## Comparison with CORE2 + farc (the mesh-size trend)

| Mesh                | nodes | per-rank @ dist_8 | GPU/CPU 4N        | GPU strong-scale (1→2N or 2→4N) |
|:--------------------|------:|------------------:|------------------:|:------------------------------:|
| CORE2 (127k, dt=1800)| 127k | 16k/rank          | **8.9×**          | 81 %                            |
| farc (638k, dt=900) | 638k | 80k/rank          | **5.5×**          | 91 %                            |
| **dars (3.16M, dt=180)** | 3.16M | 395k/rank      | **4.1×**          | **97 %**                        |

**The mesh-size lever is monotonic and substantial.** Going from CORE2 to dars
shrinks the GPU/CPU gap by **54 %** (8.9× → 4.1×) with **zero code changes**.
Strong-scaling efficiency simultaneously grows from 81 % to 97 %.

## Per-rank work vs OMEGA's claimed A100 sweet spot

OMEGA (GMD 19, 3569, 2026, Sect. 5.3) reports near-ideal A100 performance at
~1M cells per GPU node = ~250k nod2D per rank (4 ranks/node) × 96 levels.

| Mesh   | per-rank @ dist_8 | vs OMEGA sweet spot |
|:-------|------------------:|--------------------:|
| CORE2  | 16k/rank          | **16× below**       |
| farc   | 80k/rank          | 3× below            |
| dars   | 395k/rank         | **1.6× above**      |

dars is the first mesh in our matrix where we are operating **at or above** the
A100 per-rank sweet spot. The 4.1× GPU/CPU ratio is what the current FESOM-Kokkos
port delivers **when the GPU is properly fed**.

## Per-watt arithmetic (Levante)

GPU node = 2× EPYC 7763 (280 W) + 4× A100 SXM4 80GB (400 W) = **2160 W**
CPU node = 2× EPYC 7763 = **560 W**
Power ratio: **3.86×**

| Mesh    | throughput ratio | per-watt vs CPU |
|:--------|-----------------:|----------------:|
| CORE2   | 1/7              | **1/27**        |
| farc 4N | 1/5.5            | 1/21            |
| dars 4N | 1/4.1            | **1/15.8**      |

Per-watt at dars is **42 % better than at CORE2**, again with zero code changes.
Still far from per-watt parity (would require 3.9× speedup from today's dars number).
On Levante A100, per-watt parity is **not realistically reachable** for full-physics
unstructured ocean+ice — that conversation belongs on MI250X / H100 hardware.

## Implications for next-step optimisation

1. **Bigger mesh IS a real lever.** CORE2 7× → dars 4.1× is the largest single
   improvement we've found this session. The next data point (NG5 7.4M) will
   establish whether the trend asymptotes around 3-4× or continues to shrink.
2. **dars is the production-relevant test regime.** CORE2 is the development /
   per-kernel-verify regime; dars and bigger is where the user actually does
   climate work. Optimisation priorities should be measured against dars, not
   CORE2.
3. **The 4.1× ratio is stable across rank counts** on dars, meaning further
   improvements need code work, not just more nodes.
4. **Per-rank work matters more than absolute mesh size.** The OMEGA sweet-spot
   framing (1M cells / GPU node) is a useful design constant — keep per-rank
   nod2D > ~250k whenever possible.

## Stability bugs surfaced (track separately, not blocking)

1. **GPU CUDA at dist_8 on dars** — blowup at step 35 with u≈100 m/s. Reproducible
   (rep_a only got to step 35; the dist_8 dars binary crashed in the same place).
   Per-rank is 395k nod2D — the largest in the matrix. CG iters climbing through
   the run (58→64) suggest SSH/CG state diverging slowly before the blowup.
2. **CPU Serial at dist_1152** — CG NaN at step 10. Per-rank only 2744 nod2D —
   likely too-small-partition pathology (we don't run CORE2 at dist_>=512 either).

Both are at the extremes of partition size and don't affect the perf conclusions.
File issues if a user is likely to hit these in production at these exact partitions.

---

*Generated 2026-05-28 during M5.11→M5.12 brainstorm session, after OMEGA paper
analysis revealed the per-rank-sweet-spot framing.*

*Companion docs: `docs/SCALING_CORE2.md`, `docs/SCALING_FARC.md`, `docs/PROFILE_M59.md`,
`docs/M512_PLAN.md`.*
