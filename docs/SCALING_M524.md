# M5.24 full GPU+CPU strong-scaling + SYPD + step-profile campaign (2026-05-31)

Fresh "where are we now" sweep on the **M5.24** binary (`446caed`, in-place TDMA aliasing) across
CORE2 / farc / dars / NG5, GPU vs CPU, up to 32 GPU nodes. Supersedes `SCALING_M522.md`.

## Setup
- **GPU** = `build-cuda/fesom_port` (M5.24), A100×4/node, 1 rank/GPU, dist=4×nodes.
- **CPU** = `build-serial/fesom_port` (M5.24, bit-identical), EPYC 128 ranks/node (pure MPI), dist=128×nodes.
- Internal loop timer (5 warm-up steps excluded), 2 reps (min taken), `snap_every=-1` (no I/O), JRA55 1958, PHC winter IC.
- Harness `jobs/job_m524_scale_{gpu,cpu}` + `jobs/submit_m524_scaling.sh`; collect `scripts/collect_m524_scaling.py`
  → `docs/m524_scaling_data.json`; plot `scripts/plot_m524_{scaling,profile}.py`.
- **Node-to-node timing noise ≈ ±10%** (different allocations / Levante node mix) — see NG5 16N: 0.492 vs 0.445 on two dt=180 runs.

## Timestep / the dt=240 finding (important)
User asked dars+NG5 to use the **4-min (dt=240) post-spinup step** (vs the conservative 3-min/180).
**Measured: dt=240 is CFL-unstable from the *cold* PHC start on BOTH dars and NG5** — dars blows in
several partitions (T→40 °C, S→68 PSU, NaN→MPI_ABORT); NG5 ran clean to 16N but **tipped at the 32N
partition (T→−10 °C)**. This is the "only valid after spinup" effect: from a cold IC the 4-min step
exceeds CFL on the fine cells.
**Resolution (user-confirmed):** s/step is ~dt-independent (same kernels/work per step; only the CG
count shifts), so **dars+NG5 s/step are measured at the stable dt=180** and **SYPD is reported at the
production dt=240** via `SYPD = dt/(365·s_step)`, with a ×1.03 CG correction (dt=240 runs ~115 SSH-solve
iters vs 89 at 180 — a ~3% step-cost bump, within the node noise). CORE2 (1800) / farc (900) measured at
their production dt directly. (A spun-up dars/NG5 restart would allow a direct dt=240 measurement.)

### Partition-scaling cold-start instability (NG5, dt=180) — probed
Pushing a single run past the 35-step timing window (probe `jobs/job_m524_blowup_serial`, Kokkos-Serial, 300
steps, the port's `uv>5.0||NaN` guard) shows the cold PHC start blows up **faster the finer you decompose**:

| partition | ranks | nod2D/rank | blowup step (uv>5.0) |
|---|--:|--:|---|
| 16N `dist_2048` | 2048 | ~3600 | none seen (35-step timing sane; 205-step C-port completed) |
| 32N `dist_4096` | 4096 | ~1800 | **~155–160** (uv 5.23) |
| 64N `dist_8192` | 8192 | ~900  | **~10** (uv 5.38; T already [−100,120] °C, S→232) |

Monotonic and steep (>15×/doubling). The step-1 field is physical and degrades over the next steps (not a
corrupt-partition garbage-at-step-1 signature). NOT infrastructure, NOT a C-port-specific *bug*: the
bit-identical Serial port reproduces it. The earlier "C-port blew at 205 vs Serial 160" gap was a
**print-cadence artifact** (C-port `print_every=1000` only checks at step 1 and `n==nsteps=205`; Serial PE=10
catches it at the first 10-step check after `uv` crosses 5.0 ~155 — both cross at ~the same step).

**ROOT CAUSE (corrected — wsplit is NOT it):** the NG5 cold start drives genuine high vertical CFL
(**CFLz≈3 at the Gibraltar/Med outflow, glon/glat −4.81/36.01**, from ~step 4) — implementation-independent.
The discriminator is **vertical-scheme robustness, not a namelist switch:**
- **Fortran (wsplit=`.false.`)** rides CFLz≈3 (prints `WARNING CFLz>2.5`) and **completes 205 steps** — robust.
- **Port (wsplit OFF)** under the *same* CFLz≈3 ramps `uv`→>5 and **blows up (code 99) ~step 155** — less robust.
- **Port (wsplit ON, experimental `build_nl128_wsplit`)** does NOT help: near-identical `uv` ramp then a
  **CG NaN (`FESOM_CHECK` abort code 1) ~step 85** — the exact "small-per-step-error → CG NaN after ~85 steps"
  signature documented at `fesom_ale.c:88`. The port's wsplit impl is imperfect (the documented reason it's
  disabled to match CORE2: "wsplit on diverged the C from the stable Fortran path", `fesom_constants.h:54`).

⟹ a **genuine port↔Fortran vertical-stability robustness gap**, unmasked only here because validation was on
CORE2 (CFLz never near 1). Partition decomposition is the *amplifier* (more partition noise → faster onset),
not the cause. **Ways to actually run NG5 cold on the port:** spun-up restart (avoids the CFL spike — what
production does), smaller dt, or hardening the port's vertical advection / fixing its wsplit. **Caveats:**
cold-start transient only; at 64N the blowup (step 10) is *inside* the 35-step timing window → NG5-64N-from-cold
is unmeasurable, which is why the campaign caps big meshes at 32N (blowup ~155 ≫ 35 steps ⟹ all scaling valid).

## Per-step time (s/step), best of 2 reps — FINAL
| mesh | GPU 1N | 2N | 4N | 8N | 16N | 32N | CPU 1N | 2N | 4N | 8N | 12-16N | 32N |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CORE2 (dt1800) | 0.117 | 0.095 | 0.112 | 0.111 | — | — | 0.200 | 0.107 | 0.061 | — | — | — |
| farc (dt900) | 0.309 | 0.244 | 0.210 | 0.190 | 0.177 | **0.256** | 0.959 | 0.492 | 0.258 | 0.142 | 0.108(12N)/0.088(16N) | — |
| dars (dt180) | *n/a* | 0.814 | 0.475 | 0.344 | 0.237 | **0.211** | 5.94 | 3.03 | 1.58 | 0.853 | 0.428(16N) | 0.222 |
| NG5 (dt180) | *n/a* | 2.335 | 1.273 | 0.810 | 0.492 | **0.374** | — | — | 4.599 | 2.356 | 1.237(16N) | 0.631 |

NG5 CPU strong-scales near-ideal: 4→8→16→32N = 1.95×/1.91×/1.96× per doubling (~96–98% eff). NG5 CPU 32N → ~1.0 SYPD@dt240.
**GPU 32N now measured (no longer extrapolated):** dars 0.211, NG5 0.374 keep scaling (16→32N = 1.12× / 1.32× per doubling — the GPU is comm-bound here, sub-ideal). **farc GPU 32N = 0.256 is *slower* than its 16N 0.177** — 638k nodes over 128 GPU subdomains (~5k/rank) is over-decomposed: halo/launch overhead exceeds the per-GPU compute. farc CPU was not run past 16N (156 nodes/rank is below the CPU efficient band).

(GPU 1-node on dars/NG5 = **not possible** — 3.16M/7.4M nodes don't fit 4×A100; both start at 2N. User-confirmed.)

## Node-for-node GPU/CPU (Kokkos CUDA / Serial-128), <1 = GPU faster
| mesh | 1N | 2N | 4N | 8N | 16N | 32N |
|---|--:|--:|--:|--:|--:|--:|
| CORE2 | 0.58 | 0.89 | **1.83** | — | — | — |
| farc | 0.32 | 0.50 | 0.81 | 1.33 | 2.02 | — |
| dars | — | **0.27** | 0.30 | 0.40 | 0.55 | 0.95 |
| NG5 | — | — | **0.28** | 0.34 | 0.40 | 0.59 |

- **GPU 3.3–3.7× faster than CPU node-for-node on dars/NG5 at low node counts**, drifting toward ~1.8–2.5×
  by 16N and to **near-parity at 32N (dars 0.95)** — over-decomposing thins per-GPU work so the ratio rises
  monotonically; NG5 (the largest) still holds a 1.7× GPU edge at 32N (0.59).
- **farc crosses parity at ~5 nodes** (0.81 @4N → 1.33 @8N). **CORE2 (0.13M) is GPU-favored only at 1–2 nodes**
  (0.58/0.89) and flips to CPU-favored at ≥4N (1.83) — too small to fill 16 A100s.
- The campaign holds the M5.22 picture (GPU 2.8–3.0× faster on big meshes) — M5.24's TDMA win is single-GPU compute,
  invisible in the node-for-node ratio at these per-rank sizes.

## SYPD at the production timestep
| mesh | dt | 2N | 4N | 8N | 16N | 32N (measured) |
|---|--:|--:|--:|--:|--:|--:|
| dars GPU | 240 | 0.78 | 1.34 | **1.86** | 2.70 | **3.03** |
| NG5 GPU | 240 | 0.27 | 0.50 | 0.79 | **~1.3–1.4** | **1.71** |
| farc GPU | 900 | 10.1 | 11.8 | 13.0 | 13.9 | 9.6 (over-decomp) |
| CORE2 GPU | 1800 | 51.7 | 44.0 | 44.4 | — | — |

- **NG5 GPU reaches the 1–2 SYPD production band at 16 nodes (~1.3–1.4 SYPD)** and **1.71 SYPD at 32N** (measured,
  *below* the old ~2.4 extrapolation — NG5 is comm-bound past 16N so the strong scaling is sub-ideal there).
  The 4-min step lifts NG5 16N from the old 1.0 SYPD (dt=180) toward ~1.4.
- **dars GPU is in the band already at 4–8 nodes (1.3–1.9 SYPD)**, 2.7 at 16N, **3.0 at 32N**.
- At 32N the **CPU implementations also reach ~1 SYPD on NG5** (Fortran 1.19, Kokkos-CPU 1.01) **and ~2.5–3.7 on dars**
  (Fortran 3.66 even edges past the GPU's 3.03) — CPU strong-scales steeply and catches the GPU on the smaller/cheaper
  meshes at high node counts; on NG5 the GPU keeps a ~1.4–1.7× SYPD lead.

## Step profiling — where step time goes (3 regimes)
`FESOM_STEP_PROFILE`; figure `docs/figures/m524_profile_phases.png`.

| config | s/step | ocean | sea-ice | coupling/halo | forcing | top ocean phases |
|---|--:|--:|--:|--:|--:|---|
| **CORE2@4N CPU** (512c sweet spot) | 0.063 | 71% | 12% | **0.7%** | 15% | FCT 23% · SSH 14% · KPP 12% |
| **farc@4N GPU** (near-parity) | 0.240 | 75% | 19% | **1.5%** | 4% | **SSH(CG) 48%** (steep topo, 224 it/step) |
| **NG5@16N GPU** (comm-bound) | 0.520 | 63% | 14% | **14.4%** | 6% | **SSH(CG+halo) 16.5%** · FCT 14% · ice 10% |

- The **coupling/halo cost is the regime knob: 0.7% (CPU small) → 1.5% (GPU parity) → 14.4% (GPU comm-bound)**.
- At the comm-bound NG5@16N the **SSH solve (CG + its halos) is the #1 phase** — the strong-scaling wall (matches the M5.22 comm analysis). farc is intrinsically CG-bound (steep topography).

## Partitions generated this session (fesom_ini.x → writable /pool dirs)
farc dist_32/64/1024 ✓ · dars dist_1024/2048 ✓ · NG5 dist_4096 = **copied** from the byte-identical
`/work/.../meshes/NG5/` (fesom_ini.x stack-overflows reading NG5's 7.4M mesh at `ulimit -s 102400`; the
copy is valid — nod2d/elem2d/aux3d md5-identical). Jobs `jobs/job_partgen`.

## Status / caveats
- **The 32-node GPU points (farc/dars/NG5) are now MEASURED** (the 32-node-GPU alloc finally landed):
  farc 0.256 (over-decomposed, see above), dars 0.211 (SYPD 3.0), NG5 0.374 (SYPD 1.71). The earlier
  ~1.85×/doubling extrapolation was optimistic — actual 16→32N is 1.12×(dars)/1.32×(NG5), comm-bound.
- **NG5 C-port 32N is not measurable with this wall-diff harness — for C-port/model reasons, NOT infrastructure**
  (same nodes run Kokkos/Fortran fine). Three layered causes, each masking the next, found by peeling back:
  (1) **snapshot I/O doesn't scale**: with the harness's `snap_every=999999`, `fesom_main.c:903` writes a step-0
  snapshot that does `MPI_Gatherv` of all 7.4M nodes → rank 0; at 4096 ranks this aborts (SIGABRT via a UCX
  `rndv.c:485` assertion; forcing eager just moves rank-0 RSS to ~90 GB → segfault in `gather_node`). NG5×2048
  and dars×4096 both gather fine — only NG5×4096 maxes field-size AND peer-count together. (2) with `snap=-1`
  (gather gone, confirmed "snapshot every 0"), the run reaches **step 205 then numerically BLOWS UP**
  (uv=6.5 m/s, the C-port's own guard aborts all ranks) — a cold-start instability at the 4096-way partition
  that **Fortran survives** (real C↔F partition-noise divergence at extreme decomposition). (3) the C-port has
  **no internal loop timer**, so even pre-blowup the wall-diff is swamped by the NG5×4096 init cost, which
  swings ~70–180 s run-to-run (>> the ~110 s step signal). The Kokkos runs dodge all three (snap off, internal
  timer, 35 steps < the step-205 blowup). ⟹ leave NG5 C-port 32N ABSENT; the 4/8/16N curve parallels Fortran
  (extrapolates ~0.55 s/step ≈ Fortran 0.537). To actually obtain it would require adding an internal step
  timer to the C-port. The other four implementations have NG5 32N.
- farc 32N is GPU-only by design (CPU at 32N = 156 nodes/rank, below the efficient band).
- Figures: `docs/figures/m524_{scaling_overview, sypd, profile_phases}.png` + the 5-way comparison
  `m524_compare_4way{,_sypd}.png` + `m524_compare_cpu_rel.png` (all re-rendered with the 32N points).
