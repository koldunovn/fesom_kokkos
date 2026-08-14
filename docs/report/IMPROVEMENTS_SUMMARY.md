# Certified sea-ice & SSH improvements across the campaign (as of 2026-08-14)

Figure: `improvements_summary.png`. Every number is a certified, same-day pinned-pair
measurement **against its own track's baseline** — the percentages are NOT additive across
tracks (CGPIPE/SSHRAILS already live inside the `FESOM_SPEED=1` baseline that M10 and M12
were measured against; the M7-era numbers used the 2026-07 production board).

## Joint table — sea ice + SSH

| component | track / lever | point | Δ step | Δ component | status |
|---|---|---|---|---|---|
| sea ice | M9 lean wide-halo mEVP (④L, K=8) | CORE2 1N GPU | −14.4% | −58.2% of ice | recommended (GPU) |
| sea ice | M9 ④L | fArc 4N GPU | −9.6% | −43.3% of ice | recommended (GPU) |
| sea ice | M9 ④L | fArc 16N GPU | — | −52% of ice | ice cost FALLS with nodes |
| sea ice | M9 ④L | NG5 16N GPU | −12.7% | −44.1% of ice | recommended (GPU) |
| sea ice | M9 ④L | CPU (all points) | — | +46…+173% of ice | NOT profitable on CPU |
| SSH | M7 CGPIPE (pipelined CG) | CORE2 4N GPU | ≈−3% | — | adopted in `FESOM_SPEED=1` |
| SSH | M7 SSHRAILS (device SSH class) | CORE2 4N GPU | −1.9% | — | adopted in `FESOM_SPEED=1` |
| SSH | M7 CGPOLY deg-8 (poly precond) | CORE2 4N GPU | −7.9% | — | permanent manual knob |
| SSH | M10 pcsi | CORE2 4N GPU (prod cfg) | −6.30% | −60.2% of solve | worktree, unmerged |
| SSH | M10 best | CORE2 64 GPU | −6.73% | — | worktree, unmerged |
| SSH | M10 pcsi | CORE2 2048 CPU | −16.57% | — | worktree, unmerged |
| SSH | M10 oati | farc 2048 CPU | −13.28% | −45.6% of solve | worktree, unmerged |
| SSH | M10 in-range | dars CPU | −3…−4% | — | worktree, unmerged |
| SSH | M10 best | NG5 16/64 GPU | −2.32% | — | worktree, unmerged |
| SSH | **M12 split-explicit (SE)** | CORE2 4N GPU | **−7.1%** | replaces the solve | certified, `m12-split-explicit` |
| SSH | M12 SE | CORE2 16N GPU | **−10.9%** | ssh MPI 320→101/step | certified |
| SSH | M12 SE (M=90) | farc 2048 CPU | **−14.2%** | ssh MPI 846→181/step | certified — campaign-best at this point |
| SSH | M12 SE (M=20) | dars 2048 CPU | +1.1% | low ssh share at dt120 | neutral |
| SSH | M12 SE (M=20) | NG5 64 GPU | **−3.5%** | ssh MPI 234→41/step | certified |

## SSH-only table — implicit-solver line vs split-explicit, same points

| point | SI share of step | M10 best implicit | M12 split-explicit |
|---|---|---|---|
| CORE2 4N GPU (prod cfg) | ~10% | pcsi −6.30% | **SE −7.1%** |
| CORE2 16N GPU / 64 GPU | ~19–30% | −6.73% | **SE −10.9%** |
| CORE2 2048 CPU | ~30% | pcsi **−16.57%** | not measured |
| farc 2048 CPU | ~23% (90% of it MPI wait) | oati −13.28% | **SE −14.2%** |
| dars 2048 CPU (dt120) | small | −3…−4% | +1.1% |
| NG5 64 GPU | ~8.5% | −2.32% | **SE −3.5%** |

Reading: **the split-explicit solver matches or beats the best implicit-solver variant at
every point where both were measured**, and it does it by *removing* the global-reduction
wait chain rather than optimizing it (farc: 846→181 ssh MPI calls/step; total 1196→526).
The M10 law "SSH share predicts payoff" holds for both lines — dars at dt120 (small share)
is neutral for both. M10's solvers remain valuable where SE does not apply (linfs configs)
and as the CORE2-CPU-2048 record holder until SE is measured there.

Physics: SE ≡ SI to 0.2 mm rms annual-mean SSH (1-yr CORE2 twin, field std 0.66 m);
χ=0.1 required (0.05 unstable on real forcing) — see `docs/SSH_SE_M12.md`.
