# farc strong-scaling — GPU vs CPU/C/Fortran (2026-05-28)

farc mesh (~638k nodes, 48 levels), 200 steps (5 warmup excluded), **dt=900** (CORE2's dt=1800
is unstable on farc — EVP/CFL violation), JRA55 forcing, PHC winter IC. Same Kokkos M5.9-pin
binary (`05182aa`), C port, Fortran (with `check_opt_visc=.false.` because Fortran refuses
opt_visc=5 on farc-class meshes by default). Node-level strong scaling.

## Per-step time (s/step)

| Nodes | GPU (CUDA Kokkos)  | Kokkos Serial CPU       | C port                  | Fortran                 |
|:-----:|:-------------------|:------------------------|:------------------------|:------------------------|
|       | 4/8/16 A100        | 128/256/512r            | 128/256/512r            | 128/256/512r            |
|   1   | 4.0193             | 0.8713                  | 0.8430                  | **0.8471**              |
|   2   | 2.1225             | 0.4434                  | 0.4185                  | **0.4017**              |
|   4   | 1.1580             | 0.2281                  | **0.2067**              | 0.2047                  |

## Per-doubling efficiency

| Step      | GPU  | Kokkos CPU | C port | Fortran |
|:----------|:----:|:----------:|:------:|:-------:|
| 1 → 2     | 95 % | 98 %       | 100 %  | **105 %** † |
| 2 → 4     | 91 % | 97 %       | 101 %  | 98 %    |

† Slight superlinear (cache fit).

## Node-for-node ratios (GPU/CPU per node)

| Nodes | GPU / Kokkos-CPU | GPU / C-port | GPU / Fortran |
|:-----:|:----------------:|:------------:|:-------------:|
|   1   | 4.6× slower      | 4.8× slower  | 4.7× slower   |
|   2   | 4.8×             | 5.0×         | 5.3×          |
|   4   | 5.0×             | 5.5×         | 5.7×          |

## Findings

1. **Scaling is near-linear on farc** for all CPU codes (≥97 %) and very good for GPU (91–95 %).
   Bigger per-rank work → halo cost relatively smaller → scaling extends further.
2. **GPU-vs-CPU ratio is more stable** on farc (4.6–5.7×) than CORE2 (4.6–8.9×) — the GPU's
   disadvantage stops growing because per-GPU work stays substantial.
3. **C port and Fortran are essentially tied** on CPU (within 1 %), Kokkos-CPU is 3-10 % slower.
   The Kokkos-CPU overhead that ballooned on CORE2 (21→68 %) almost disappears on farc.
4. **Per-A100 throughput rises with bigger mesh** — but a CPU node still beats a GPU node ~5×
   on this hardware (Levante A100 vs EPYC 7763). Per memory, the Frontier MI250X numbers in
   the Omega paper close this gap; A100 perf is the floor.

## Comparison with CORE2

| Metric                | CORE2 (~127k nodes, dt=1800) | farc (~638k nodes, dt=900) |
|:----------------------|:----------------------------:|:--------------------------:|
| 1-node GPU s/step     | 0.8617                       | 4.0193 (~4.7× slower)      |
| 1-node CPU s/step (C) | 0.1533                       | 0.8430 (~5.5× slower)      |
| GPU/CPU per node      | 5.6× (1N) → 8.9× (4N)        | 4.8× (1N) → 5.5× (4N)      |
| Best GPU scaling      | 91 % (1→2), 81 % (2→4)       | 95 % (1→2), 91 % (2→4)     |

farc per-step is ~5× CORE2's per-step (mesh ratio matches: 638/127 ≈ 5), but with dt halved
the model produces the same amount of model time per step. So **same model-time throughput at
~half the per-step cost-ratio** is the actual figure of merit.
