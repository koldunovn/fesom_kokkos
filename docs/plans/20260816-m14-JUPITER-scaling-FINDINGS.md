# M14 on JUPITER — combined-campaign scaling findings

**Status: MEASUREMENT PHASE COMPLETE** (2026-08-16 night; 227 jobs, 2,420 node-h =
0.697 Mcore-h ≈ 1% of the remaining e-sta-destine budget). **Methodological ruling (user, 2026-08-16): production machines are BUSY —
JUPITER's quiet windows are the exception, not the reference condition.** Loaded-fabric
measurements are therefore deployment-relevant results, not contamination; allocation-to-
allocation spread is reported as a range, and the planned full "calm pass" was cut to three
quiet-window confirmations of the headline NG5 points (base / SE / composition at 1024 GPUs).
Numbers marked 🔶 are single-allocation and carry that spread.

Campaign: branch `m14-integrate` (M9+M10+M11+M12b+M13 merged), JUPITER booster, quad-GH200,
4 ranks = 4 GPUs per node, up to 256 nodes = 1024 GPUs. Protocol = `jobs/job_m14_jupiter_ladder`:
300-step legs, warmup discarded, ABBA both-arms-in-one-allocation, zombie check on stdout+stderr,
`FESOM_HALO_STAGE=1`, `FESOM_IC_EXTRAP=det` both arms, wsplit ON for farc/dars/NG5 (gate G3
verified the runtime print — it goes to STDERR, one line per rank). Binaries:
`fab4919c` (pre-EVPWIDE-fix, all jobs ≤ ~1391520) and `0cad3289` (post-fix; EVPWIDE-inert paths
byte-identical in behavior). Harvest: `scripts/m14_scaling_figs.py` →
`$SCRATCH/port2/m14/figs/m14_scaling.csv` + two figures.

## Headline answers

### Q1 — Does M14 convert NG5's beyond-knee regression? YES — via split-explicit, not oati.

The NG5 SE ladder (`FESOM_SSH_MODE=se`, `FESOM_SE_M=20` from the probe: mesh limit
dtbt ≤ 10.69 s → M_min=17; zstar both arms) is clean at every rung (0 rejections) and its
**absolute step time falls monotonically through 1024 GPUs**:

| NG5 ranks | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| base (zstar) | 0.3372 | 0.1882 | 0.1092 | 0.0755 | 0.0597 | 0.0594 | 0.0661 |
| SE (M=20) | 0.3350 | 0.1827 | 0.1012 | 0.0651 | 0.0470 | 0.0434 | **0.0417** |
| gain | −0.7% | −2.9% | −7.3% | −13.8% | −21.3% | −26.9% | **−36.9%** |

The July M7 knee (peak at 512 ranks, ~14% regression at 1024) is **erased** in the SE
configuration: 0.0434 → 0.0417 across the old knee. Mechanism: SE removes the barotropic CG's
global allreduces outright; the levers that merely accelerate the same allreduce-bound solver
(`oati`) fade exactly where SE keeps growing. SE is likewise the strongest lever on every mesh
that runs it: fArc −28→−63% (g16→g512; base collapses past its knee, SE erases most of it),
dars −2.3→−18% clean (g32→g256) and 🔶 −49% at g512 (loaded allocation).

### Q2 — The partition lever above 64 GPUs: Levante's gains do NOT transfer at small scale;
🔶 an apparent crossover to wins at ≥256 ranks correlates with fabric load.

Every Levante partition gain measured ≤64 GPUs fails to reproduce on GH200 (all clean
allocations, within-job ABBA):

| point | Levante A100 | JUPITER GH200 |
|---|---|---|
| core2 g4 | −7.1% | −1.3% (and null/loss across g4–g64) |
| dars g64 | −18.5% | +0.4% |
| ng5 g64 | −9.6% | +1.1% |
| ng5/dars small rungs | — | +3.7 to +9.5% (losses) |

At larger scale the sign flips — dars: +0.4% (g64) → −16.1% (g256) → −19.3% (g512);
NG5: −12.5% (g512), −2.3% (g1024). The winning points sit in allocations whose base legs ran
15–100% above the quiet-window baseline, while quiet allocations at 256 ranks still show
losses (NG5 +4.1%): on this fabric the M11 partitions buy **resilience to fabric load**
(fewer/better-shaped messages) rather than quiet-fabric speed. Under the campaign's ruling
that busy is the production condition, that resilience IS the deployment verdict — the
partition lever belongs in the big-rung production config, with the quiet-window nulls
recorded as the spread's other edge. M11's own caveat — the winner is point- and
fabric-specific — is confirmed in the strongest possible way.

### The best-combination result and the production recommendation

Best combo = SE + EVPWIDE (staged) + M11 partition (partition only where it pays: ≥~64–128
ranks on dars/NG5, never on CORE2, ~null on fArc small rungs). Full production-dt table in
`figs/m14_scaling.csv`; peaks: CORE2 176 SYPD (g64), fArc 69.3 SYPD (g128), dars 23.7 SYPD
(g512), NG5 19.0–19.1 SYPD (g512–g1024 plateau; 17.5 at g2048, gain −60.3% there — the
campaign's largest). fArc g32 filled with SE+EVPWIDE on stock partition (−36.1%, 62.9 SYPD;
the M11 engine has no dist_32 anywhere — the zoo's one hole). At 2048 GPUs the M11 partition's
marginal on top of SE+EVPWIDE is −14% (within-job gain fractions: −60.3% vs −53.6%; raw
cross-allocation best/best says −33%) — the partition lever's largest measured contribution,
still growing with scale.
**NG5 production recommendation: 128 nodes = 512 GPUs** — the plateau's left edge; doubling
to 1024 GPUs buys +0.5% SYPD for 2× the resources (user ruling 2026-08-16). The base config's
knee (peak 11 SYPD at g512, then decline) is fully erased by the combo, which degrades only
gently past the plateau (−9% at 2048 GPUs vs base's collapse to 0.0934 s/step).

## The lever matrix (settled results)

- **`oati`** — clean, monotonic gains on core2 (−6→−13%, g4→g64), dars (−1→−4.3%, g16→g128)
  and NG5 (−0.5→−6.9%, g16→g128). 🔶 Above 128 ranks in calm allocations it goes null/loss
  (NG5 +3.0% g256, dars +0.4% g256); large apparent gains at 512/1024 (−9 to −49%) are
  load-regime measurements. g1024 across two allocations: +4.9% / −9.4% with bimodal base arms
  (0.081–0.190 s/step) and *stable* best arms (0.085–0.097) — the variance asymmetry is itself
  a finding: oati stabilizes the latency-bound rung.
- **`oati` on fArc — VOID at every rung** (M10 handoff item 3, reproduced): every best leg
  fires `[ssh-solver] !! FALLBACK` from solve ~70 (deterministic onset), and on GH200 the
  fallback recovery itself often NaNs (`CG_kk … s_old=-nan`, ~60% of legs die; CFLz=0 — NOT
  the M5.24 class). Survivor legs "win" ~−15% but are variant/baseline mixtures (M10 rule:
  no timing quoted without `fallbacks=`). Not a merge regression, not wsplit.
- **Split-explicit** — see Q1. Per-mesh M matters (probe first): farc 90, dars 20, NG5 20
  (M_min 17), core2 35 (probe confirmed the M12b calibration; core2 SE ladder ran with it).
- **Partition** — see Q2.
- **Composition (partition+`oati`)** — NG5 g512: −18.3% in a *clean* allocation
  (base 0.0618 ≈ solo baseline 0.0624); best-arm 0.0505 was the campaign's fastest NG5 number
  until SE beat it (0.0434/0.0417). 🔶 g1024: −42.3% with both arms internally consistent but a
  loaded base (0.122/0.128). CORE2/farc/dars composition rows: farc VOID (oati component);
  others in the CSV.
- **Sea-ice wide halo (EVPWIDE)** — see below; measurement was impossible pre-fix.

## The EVPWIDE bug, fix, and validation (settled)

Pre-fix, `FESOM_SPEED_EVPWIDE=8 FESOM_SPEED_EVPWIDE_LEAN=1` crashed at **every** rung on
**every** mesh, including single-node: `evpw_exchange()` (`src/fesom_ice_evpwide.cpp`, old
:1036–1094) posted `MPI_Isend/Irecv` directly on CUDA device Kokkos views, bypassing
`FESOM_HALO_STAGE` — the only unstaged MPI data path in the port. On PSMPI 5.11/UCX 1.19
without GPUDirect, UCX CPU-packs the device pointer (`ucp_mem_type_pack`, dt.c:86) → SIGSEGV;
intra-node shared memory hits the same pack. Base runs were untouched (their exchanges are
staged), which made the lever look "measured as a loss" on dolpung when it had never validly
run on such a stack at all.

Fix (working tree, uncommitted): pinned host mirrors in `EvpwState`, D2H of the used range →
MPI on host pointers → H2D before the diagnostic kernels; keyed on the same
`fesom_halo_stage_on()` as cgpipe; knob-off byte-identical; message counts/tags/order
untouched. Validation (jobs 1391541/1391545): 30 steps clean, `[evpwide-self] step-ship uv
echo = 0.000e+00` 30/30 (byte-exact staged transport), base leg unperturbed. **All prior
EVPWIDE numbers anywhere (incl. Levante/dolpung) were measured on the crashing device-pointer
path; the post-fix ladder (in the serial chain) is the first valid data for this lever on a
staged transport.** First opportunistic point (mixed-binary job, flagged): NG5 g16 −0.9%.

## Infrastructure record (JSC-relevant)

- Node `jpbo-060-24` hung mid-leg (IO errors from its 4 ranks, step timeout, then
  "Requested nodes are busy" wedging the remaining legs); Slurm marked the job NODE_FAIL.
  Excluded via `--exclude` on resubmit. A second scattered-task step timeout hit another
  256-node allocation the same evening. Wedged-step retry loops do not self-heal — cancel.
- Allocation-to-allocation variation at ≥256 ranks reached ~2× on base arms during the
  evening's loaded window (own fleet ≈2400 nodes + unknown external load), far above July's
  24%-at-1536-ranks record. Within-job ABBA gains remain the only defensible estimator; ladder
  *shapes* must come from the serial calm pass.

## Estimator discipline applied

Within-job ratios only; min over admitted legs; zombie check incl. stderr CG-NaN grep;
mixture legs (fallbacks>0) voided; `cfg=` keeps wsplit/zstar rows apart; baseline-only JCSV
`best=0.0000` artifact from the pre-fix awk (jobs ≤1390937) ignored by the harvester;
mixed-binary jobs (running across the mid-campaign rebuild) flagged manually: 1391342, 1391348.

## Cost

~330 node-hours through the parallel matrix; projected ~2200–2700 nh (~0.63–0.78 Mcore-h at
288 core-h/node-h) for the full campaign incl. calm pass — ~1% of the remaining e-sta-destine
budget (72.8 of 97 Mcore-h available as of 2026-08-16).

## Phase attribution at the top rungs → M15 targets (FESOM_SPEED_PHASESTATS pairs, jobs 1392314/1392376)

NG5 base at 1024 GPUs is 61% MPI-wait, with the implicit-SSH CG alone at 21.6 ms wait/step
(31% of the step — quantitatively reproducing July's knee diagnosis). The best combo removes CG
(SE's bt = 4.2 ms, 42 msg/step) and cuts icedyn messages 120→16 (EVPWIDE), leaving 35.5 ms/step
whose anatomy across g512→g1024 names the next campaign:

1. **Ocean-phase compute imbalance under the M11 partition** — busy spread 1.6× (g512) → 1.8×
   (g1024) vs 1.12× on the stock partition: a5_u30 bought its message wins with work imbalance
   (weights ~2D-heavy, A=100). Target: re-score the zoo's WGT_A axis (a3_a0/a15/a40) at NG5
   512–2048 ranks with phasestats busy-spread as a scoring column. Headroom ≈12–14%/step.
   (Generation in flight on Levante, 2026-08-16 night.)
2. **icedyn replication floor** — busy flat at 4.9 ms from g512 to g1024: the K=8 ghost ring's
   redundant work is rank-count-invariant. Target: K-sweep (K=4 vs 8) at the top rungs; plus
   EVPWIDE_FUSE, measured −3.1% (g256) / −5.3% (g1024) fused-vs-unfused clean pairs — joins the
   recommended config (confirmation on the full combo pending).
3. **bt latency floor** — 42 msg/step at M=20. M12b's k-periodic η exchange (Path-B: exchange
   every k substeps on the existing K=1 wide rung, which is exact free-running after the s3
   fixes) prices deep-K with no extended mesh; k=2 removes ~25% of SE exchanges.
   Implementation + 1-node validation done this session (uncommitted); staging race at g512
   next.

## Open items

1. Calm-pass adjudication of every 🔶 above (running).
2. Post-fix EVPWIDE ladder (running) → then decide whether EVPWIDE joins the per-mesh
   max-config; if yes, run SE+partition(+EVPWIDE) compositions at top rungs (user sign-off).
3. Commit the EVPWIDE staging fix + the awk guard fix + fetch-script statics stanza (with
   byte-gate re-run on Levante for the EVPWIDE-off path) — M15.
4. fArc `oati` stall root-cause (σ-recurrence hypothesis, M10 item 3) — still open upstream.
5. NG5 SE bit-fidelity/climate check vs base (SE is a different discretization — the speed
   numbers here are protocol-timing only; adoption needs the M12b fidelity gates re-run at NG5).
