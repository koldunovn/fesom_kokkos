# Scaling re-run on m522 (post-M5.21 full campaign) — 2026-05-31

Re-measures GPU strong-scaling on the **m522** binary (M5.18–20 coalescing/PCIe-residency + M5.21
Lever 1 coalescing + Lever 2a ghats + Lever 2b SSS-trim) across the four meshes. **CPU numbers are
REUSED unchanged** from the 2026-05-28/29 runs (the CPU path is `#ifdef CUDA`-gated → identical
binary; see `SCALING_{CORE2,FARC,DARS,NG5}.md`). GPU = `build-cuda/fesom_port` (m522), A100×4/node,
dist_$(4×nodes); internal loop timer, 2 reps (spread < 0.5 %), snap_every=−1. dt: CORE2 1800,
farc 900, dars/NG5 180. Figures: `docs/figures/scaling_{meshsize_trend,overview}.png` (plot_scaling.py).

## Per-step time, GPU m522 (s/step, mean of 2 reps) — CPU reused (Serial, 128c/node)

| Nodes | CORE2 GPU | CORE2 CPU | farc GPU | farc CPU | dars GPU | dars CPU | NG5 GPU | NG5 CPU |
|:--:|--|--|--|--|--|--|--|--|
| 1  | 0.1285 | 0.1856 | 0.3245 | 0.8713 | —      | —     | —      | —     |
| 2  | 0.1129 | 0.0946 | 0.2687 | 0.4434 | 0.8659 | 2.844 | 2.4550 | —     |
| 4  | 0.1345 | 0.0555 | 0.2328 | 0.2281 | 0.5162 | 1.465 | 1.4699 | 4.330 |
| 8  | 0.1354 | —      | —      | —      | 0.3629 | —     | 0.7936 | 2.239 |
| 16 | —      | —      | —      | —      | —      | —     | 0.4921 | 1.171 |

## Node-for-node GPU÷CPU (× ; <1 = GPU faster)

| Nodes | CORE2 | farc | dars | NG5 |
|:--:|--|--|--|--|
| 1  | **0.69** (1.45× fast) | **0.37** (2.7× fast) | — | — |
| 2  | 1.19 (slow) | **0.61** (1.65× fast) | **0.30** (3.3× fast) | — |
| 4  | 2.42 (slow) | 1.02 (parity) | **0.35** (2.84× fast) | **0.34** (2.95× fast) |
| 8  | (no CPU) | — | (no CPU) | **0.35** (2.82× fast) |
| 16 | — | — | — | **0.42** (2.38× fast) |

## The meshsize trend (4 GPU nodes = dist_16) — the headline

| mesh | M nodes | GPU m522 | CPU | ratio | vs pre-campaign | vs post-M5.13 |
|--|--|--|--|--|--|--|
| CORE2 | 0.13 | 0.1345 | 0.0555 | 2.42× slow | 5.3× slow | — |
| farc  | 0.64 | 0.2328 | 0.2281 | **1.02× ≈ parity** | 5.1× slow | — |
| dars  | 3.16 | 0.5162 | 1.465  | **2.84× FAST** | 4.1× slow | 1.60× slow |
| NG5   | 7.40 | 1.4699 | 4.330  | **2.95× FAST** | 3.76× slow | 1.41× slow |

## Findings
- **The full M5.x campaign crosses parity.** GPU went from **3.8–5.3× slower** (pre-campaign, PCIe-bound)
  to **~2.9× FASTER** on the big nod3D meshes (dars, NG5). Crossover ≈ farc (0.64 M nodes).
- **GPU wins most at LOW node counts** (more work per GPU); as nodes increase the CPU strong-scales
  better and the GPU becomes comm-bound, so the GPU advantage shrinks with node count (NG5: 2.95× @4N
  → 2.38× @16N) and the crossover node-count grows with mesh size. **CORE2 even flips back to CPU-favored
  at ≥2 nodes** — it's too small to fill 16 A100s (per-rank work ~8 k nod2D; flat strong scaling).
- **Per-watt:** the GPU node draws 3.86× the power, so per-Wh the CPU still leads — but only **~1.3×** on
  dars/NG5 now (was 14–20× pre-campaign). The energy gap is nearly closed on the production meshes.
- **dars n8 (0.363) has no CPU pair** (prior dars dist_8 CPU hit a CG-NaN); NG5 n16 (0.492) is the
  largest GPU point. All GPU runs T/S-sane (no blowups at dist_16 for any mesh on m522).

Memory [[project-m521-coalescing-finish]]; per-mesh pre-campaign detail in `SCALING_{CORE2,FARC,DARS,NG5}.md`.
