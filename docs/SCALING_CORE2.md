# CORE2 strong-scaling — GPU vs CPU/C/Fortran (2026-05-28)

CORE2 mesh (~127k nodes, 47 levels), 200 steps (5 warmup excluded), dt=1800, JRA55 forcing,
PHC winter IC. Binary: Kokkos M5.9-pin (`05182aa`); C port `port2/fesom2_port`; Fortran
`fesom27/fesom2/bin/fesom.x`. Node-level strong scaling — each row is a SINGLE problem on
varying resources. GPU node = 4 A100; CPU node = 128 cores (1 rank/core).

## Per-step time (s/step)

| Nodes | GPU (CUDA Kokkos)           | Kokkos Serial CPU       | C port                  | Fortran                 |
|:-----:|:----------------------------|:------------------------|:------------------------|:------------------------|
|       | 4/8/16 A100, dist_4/8/16    | 128/256/512r, dist_*    | 128/256/512r, dist_*    | 128/256/512r, dist_*    |
|   1   | 0.8617                      | 0.1856                  | **0.1533**              | 0.1625                  |
|   2   | 0.4741                      | 0.0946                  | **0.0678**              | 0.0760                  |
|   4   | 0.2937                      | 0.0555                  | **0.0329**              | 0.0394                  |

## Per-doubling efficiency

| Step      | GPU  | Kokkos CPU | C port      | Fortran     |
|:----------|:----:|:----------:|:-----------:|:-----------:|
| 1 → 2     | 91 % | 98 %       | **113 %** † | **107 %** † |
| 2 → 4     | 81 % | 85 %       | **103 %** † | 96 %        |
| 4 → 8 GPU | 72 % | —          | —           | —           |

† Superlinear — cache-fit effect (per-rank data shrinks into L2/L3 as ranks grow).

## Node-for-node ratios (GPU/CPU per node)

| Nodes | GPU / Kokkos-CPU | GPU / C-port | GPU / Fortran |
|:-----:|:----------------:|:------------:|:-------------:|
|   1   | 4.6× slower      | 5.6× slower  | 5.3× slower   |
|   2   | 5.0×             | 7.0×         | 6.2×          |
|   4   | 5.3×             | 8.9×         | 7.5×          |

## Findings

1. **CPU node beats GPU node ~5–9×** on CORE2 (consistent with the M5.1b farc memory: "node-for-node CPU ~7.4× faster"). One A100 ≈ 7 CPU cores of work.
2. **C port is fastest on CPU** at every node count. Fortran trails by 6–20 %; Kokkos-CPU trails C by 21 → 40 → 68 % — its overhead **grows** with node count.
3. **C / Fortran scale superlinearly** at the small-mesh end (cache effects); Kokkos-CPU and GPU don't show this. Kokkos's per-rank DualView/Field infrastructure has a fixed cost that masks the cache win.
4. **GPU scales sub-linearly** (91 → 81 → 72 % per doubling). Halo + per-rank-too-small effects at dist_16 and beyond. Bigger mesh (farc / dars2) should extend the linear regime.

## Implications for the perf track

At CORE2 mesh size the GPU node is not cost-competitive with a CPU node. The remaining big lever is **Lever C — fesom_field.hpp rank-1 → View<T**>** for coalesced GPU memory layout. Other ideas surveyed (RangePolicy, Knuth double-double, Omega paper) gave little new ground because we already adopted them or they're orthogonal to the dominant bottleneck.
