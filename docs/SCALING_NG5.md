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

> **⚠️ Caveat (see § nsys decomposition below):** "3.6–3.8×" is the floor *at the current
> device-residency*, NOT an intrinsic compute floor. nsys shows the NG5 GPU spends ~75 % of each
> step blocked on full-field PCIe `cudaMemcpy` (un-flipped 3-D halos) and only ~7 % computing.
> That PCIe overhead is reducible (device-residency / halo flips); the ratio should be
> re-measured after the NG5 halo-flip campaign, not treated as fixed.
>
> **✅ RESOLVED (M5.13, 2026-05-30):** the campaign re-measured it. The "3.6–3.8× floor" was indeed
> mostly reducible PCIe — device residency dropped NG5 to **1.41×** and dars to **1.60×** (denser dars
> dist_8 → **1.52×**). The asymptote conclusion above is **superseded**: the floor was a *data-movement*
> artifact, not a compute floor. See § M5.13 RESULT and § dars cross-mesh confirmation below; the real
> remaining floor (~1.4×) is per-step launch/MPI overhead, a different lever.

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

## Per-phase profile + nsys decomposition — NG5 IS PCIe-DATA-MOVEMENT-BOUND (2026-05-29)

> **⚠️ CORRECTION (supersedes the first draft of this section).** The first pass used the
> `FESOM_STEP_PROFILE=1` per-phase *wall* timer and concluded NG5 was "compute/bandwidth-bound"
> → Lever C. That was WRONG. The per-phase timer measures *phase wall* = kernel + PCIe-sync +
> MPI *together*; it cannot tell compute from data-movement inside a phase. An `nsys` CUDA
> trace (kernels **vs** memcpy, the decisive instrument) shows the GPU **computes only ~7 % of
> the step** and spends **~75 % blocked on full-field host↔device `cudaMemcpy` (PCIe)**. NG5 is
> **data-movement-bound**, the same class as the M5.3-era CORE2 wall — just bigger. Lever C
> (kernel coalescing) would touch only the 7 % and is NOT the production lever.

**nsys CUDA trace, NG5 dist_16, rank 0, 8 developed steps (job `25227869`, snapshots OFF):**

| step component                         | s/step | % of 16.94 s step |
|:---------------------------------------|-------:|------------------:|
| **PCIe `cudaMemcpy` H2D**              | 8.13   | 48 %              |
| **PCIe `cudaMemcpy` D2H**              | 4.61   | 27 %              |
| MPI / `cudaDeviceSynchronize` / host   | 3.01   | 18 %              |
| **GPU kernels (all compute)**          | 1.19   | **7 %**           |

- `cudaMemcpy` = **90.6 %** of all CUDA API time (102 s / 8 steps, blocking). **~4 575 PCIe
  transfers/step** (2 842 H2D + 1 733 D2H), max single = 327–355 ms (full-field). H2D-dominant
  (63.8 %) ⇒ this is **halo unpack + re-push (`sync_host`→MPI→`sync_device`)**, not the rank-0
  I/O gather (that would be D2H, and I/O was disabled).
- GPU **busy 7 %, idle 93 %.** The idle is PCIe (75 %) + MPI/sync (18 %).

**Within the 7 % GPU compute** (nsys kernel sum, grouped):

| kernel group                | % of GPU compute | ms/step |
|:----------------------------|-----------------:|--------:|
| smoother (bvfreq + KPP blmc)| 30.4 %           | 361     |
| FCT advection               | 28.5 %           | 338     |
| momentum / ALE / KPP-mix    | 16.3 %           | 194     |
| GM/Redi · EOS/PGF · SSH/CG  | ~6 %             | 73      |

Smoother + FCT = 59 % of compute, but compute is only 7 % of the step, so each is ≈ 2 % of wall.
Per `ncu --set basic`, the FCT and GM kernels are **memory-leaning** (SOL Memory ~50–58 %,
Compute ~4–5 %, occupancy ~50–60 %) — real, but irrelevant while they're 7 % of the step.
*(The earlier ncu read reported these as µs; that was a unit misread — they are ms-scale, per
the authoritative in-situ nsys trace. The SOL bound-labels were correct; only the durations were
wrong.)*

### The per-phase wall table (job `25225499`) — still valid as phase composition

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

The dominant phases shift from CORE2's **nod2D-halo** phases (SSH 10.9 %, ice_dyn/EVP 11.3 %) to
NG5's **nod3D** phases (FCT 22.8 %, GM 11.2 %, ALE 9.1 %, KPP 8.3 %). But the *mechanism* is the
same in both: **host↔device data movement**, not compute.

> **The decisive number:** the FCT *phase wall* is 3.708 s/step but FCT *kernels* are only
> 0.338 s/step (nsys) — **11× more wall than compute.** That ~3.3 s gap is the FCT phase's
> full-field PCIe halo syncs (the internal halos not yet device-flipped). The "FCT scaled 41×
> linearly" observation is **equally consistent with PCIe-bound** (a 43× bigger field ⇒ 43×
> bigger memcpy) and is therefore NOT evidence of compute-bound. nsys settles it: PCIe.

So the regime shift is real but it is **nod2D-halo-latency-bound (CORE2) → nod3D-PCIe-bandwidth-
bound (NG5)** — two flavors of the same data-movement wall, *not* "launch-bound → compute-bound."

### Implication — the production lever is DEVICE RESIDENCY, not Lever C

The M5.12 launch-fusion levers (f reverted, d +0.7 %, b +0.6 %; L53–L55) optimized CORE2's
nod2D-halo-latency regime and are nearly irrelevant at NG5. **The production lever is to kill
the ~12.8 s/step of PCIe — i.e. continue the M5.1/M5.4/M5.7 device-residency / halo-flip
campaign into the NG5 3-D regime:** flip the remaining host-staged nod3D halos (FCT internals,
GM chain, ALE, tracer-diff) to `fesom_halo_field` (GPU-aware MPI, no full-field PCIe), and
eliminate the host-op syncs (the L39 salinity floor, L50 uvnode) where a device twin is viable.
Each un-flipped 3-D halo that was PCIe-cheap at CORE2 (16k/rank) is a full-field 259 MB sync at
NG5 (462k/rank); the M5.4 campaign flipped CORE2's *big* halos and left many that balloon here.

**Consequence for the headline ratio:** the **3.8× GPU/CPU gap is mostly reducible PCIe
overhead, not an intrinsic compute floor.** The GPU wastes 75 % of every step on host↔device
copies; closing even half of that would move NG5 toward ~2× (and per-watt accordingly). The
"~3.6–3.8× floor" / "per-watt unreachable" conclusions above are therefore **upper bounds on the
*current* device-residency, not hard limits** — they should be re-measured after the NG5 halo
flips. Lever C (rank-1 → `View<double**>` coalescing) stays shelved: it improves the 7 %, not
the 75 %.

---

## M5.13 RESULT — the device-residency campaign confirms the PCIe was reducible (2026-05-30)

The L56 prediction held. M5.13 flipped the remaining host-staged nod3D/elem3D halos to
`fesom_halo_field` (a `cfl_z`, b EOS `hpressure`/`sw_α`/`sw_β`, c the GM quartet, d `uv_rhsAB`,
e ALE `w`/`w_e`+bolus, f ALE commit `hnode`/`helem`). NG5 dist_16, snapshot binaries:

| metric (NG5 dist_16)        | baseline |  a–f  | +g1-uv | **+g1-T (final)** |
|:----------------------------|---------:|------:|-------:|------------------:|
| **step (s/step), clean**    | **16.27**| 10.88 |  6.97  | **6.12**          |
| **node-for-node GPU/CPU**   | 3.76×    | 2.51× | 1.61×  | **1.41×**         |
| PCIe `cudaMemcpy` (s/step)  | 12.74    | 7.48 (nsys) |  —   | **2.83 (nsys)**   |

(g1-uv = full `uv` residency, the biggest single win; g1-T = full T values+valuesold residency,
fixed with an L50 bulk-SST sync. ✅ **The authoritative 1-year CORE2 CUDA-vs-Fortran+C climate
validation PASSED** — the campaign binary is statistically identical to the pre-campaign M5.9-pin
binary (vs C-port: sst corr 1.00000/bias 1.2e-4, all fields ≥0.99996; zero climate-level change from
the flips). See `docs/GPU_FIDELITY.md` § Climate validation. The 20-step per-milestone gate was only a
*staleness* tripwire, not the climate test.)

g1-uv (full `uv` device-residency — flip update_vel + remove ALL 11 uv re-pushes + the ocean2ice
cross-file push) was the **single biggest win**: 10.88 → 6.97 s/step (−36% more), because `uv`
(ELEM3D×nl×2, the largest field) is read ~11×/step. The CORE2 deep_copy proxy MB halved (641 → 342
MB/step). **The campaign blew past the ~2× target to 1.61×.**

- **Clean a–f GPU dist_16 = 10.88 s/step** (2 reps 10.868/10.894, job `25233271`, no nsys/no I/O)
  vs **CPU dist_512 = 4.330 s/step** (node-for-node, UNCHANGED — the flips are
  `#ifdef KOKKOS_ENABLE_CUDA`, so the Serial/CPU binary is byte-identical and the CPU scaling
  column above still holds). → **node-for-node GPU/CPU 3.76× → 2.51×**, the gap closed ~⅔ of the
  way from 3.8× toward the ~2× target, **from a–f alone**.
- **PCIe `cudaMemcpy` dropped 41 %** (12.74 → 7.48 s/step); GPU compute unchanged (~1.2 s/step);
  PCIe share 75 % → 67 %. The blocking full-field memcpy count fell (213 → 192 /step at a+b+c) as
  the device-halo path replaces each big blocking copy with on-device pack + GPU-aware-MPI async.
  The "3.6–3.8× floor / per-watt unreachable" conclusions were indeed mostly reducible PCIe.
- **g1-uv + g1-T DONE** (full `uv` and tracer-`T` device-residency; g1-T needed the L50 bulk-SST
  sync). **Only g2 (S-floor → device) deferred** — conditional gate not met; S is floor-pinned and
  hits the same L50 SSS host reader. Validation: every flip = Serial verify=0 + pi np1+np2 bit-id +
  SYNCCHECK + the CORE2-active-ice fidelity gate; ✅ the 1-yr CORE2 CUDA-vs-Fortran+C **climate**
  validation PASSED (statistically identical to pre-campaign; corr ~1.0000 / bias O(1e-4) vs C). See
  `docs/plans/20260530-m513-pcie-residency-tasks.md` § Deferred, lesson **L57**, `docs/GPU_FIDELITY.md` § M5.13.

**Figures (M5.13):** `docs/figures/m513_ng5_progression.png` (NG5 step 16.27→6.12 s/step, GPU/CPU
3.76×→1.41×) · `docs/figures/m513_deepcopy_proxy.png` (per-milestone PCIe deep_copy 1068→277 MB/step)
· `docs/figures/m513_ng5_pcie_decomp.png` (PCIe share 75%→44% — the GPU un-starved). Script:
`scripts/plot_m513_progress.py`.

### dars cross-mesh confirmation + the data/compute-balance test (2026-05-30)

The campaign was developed and accepted on NG5, but the flips are mesh-independent (`#ifdef CUDA`,
always active). Re-running the **campaign binary on dars** (3.16M nodes, 47 lvl) confirms the win
generalizes across meshes — and tests the user's hypothesis that *the right data/compute balance
gets the GPU/CPU ratio near 1×*:

| dars (campaign binary)        | GPU s/step | CPU node-for-node | ratio  | nod2D/rank |
|:------------------------------|-----------:|------------------:|-------:|-----------:|
| pre-campaign dist_16 (4N)     | 6.003      | 1.465 (dist_512)  | 4.10×  | 197k       |
| **post dist_16 (4N)**         | **2.342**  | 1.465 (dist_512)  | **1.60×** | 197k    |
| **post dist_8 (2N, denser)**  | **4.311**  | 2.844 (dist_256)  | **1.52×** | **395k** |

- **The campaign cuts dars's gap 4.10× → 1.60×** (−61 %), the same magnitude as NG5 (3.76× → 1.41×) —
  it is not an NG5-specific artifact; both big nod3D meshes were PCIe-bound and both are now released.
- **The data/compute-balance theory holds — in direction.** The denser dist_8 packing (395k nod2D/rank,
  2× the per-rank work of dist_16) lands **closer to parity (1.52×) than dist_16 (1.60×)**. Plotted with
  NG5 dist_16 (462k/rank → 1.41×), the trend is **monotone: more work per GPU → nearer 1×**. The A100 is
  feed-rate-limited at low per-rank work; loading it more closes the gap.
- **But the asymptote on Levante-A100 is ~1.4×, not 1×.** Even at 395–462k/rank the residual gap is the
  irreducible per-step overhead — ~720 kernel launches + MPI/sync (the lever-C / launch-fusion territory,
  *not* PCIe). To approach 1× you'd need both more per-rank work (denser packing — but the CPU OOMs first
  on bigger meshes) **and** the launch/fusion lever. So: device residency took the gap from "GPU is a
  curiosity" (3.8×) to "GPU is within ~1.5× and energy-competitive"; closing the last ~0.4× is a
  different lever. Figure: `docs/figures/scaling_meshsize_trend.png` (the green campaign curve diving
  through the old 3.6–3.8× PCIe floor; the white triangle = the dist_8 denser-packing probe).

### High-node strong-scaling on the campaign binary — 8/16/32 nodes (2026-05-30)

`jobs/submit_m513_hinode_scaling.sh` (campaign binary, 35 steps, 2-rep means). Confirms the campaign
binary strong-scales AND quantifies the per-rank-work drift the data/compute-balance test predicts.

| nodes | NG5 GPU | NG5 CPU | NG5 ratio | nod2D/rank | dars GPU | dars CPU | dars ratio | nod2D/rank |
|------:|--------:|--------:|----------:|-----------:|---------:|---------:|-----------:|-----------:|
|   2   |    —    |    —    |     —     |     —      | 4.311    | 2.844    | **1.52×**  |   395k     |
|   4   | 6.12    | 4.330   | **1.41×** |   462k     | 2.342    | 1.465    | **1.60×**  |   197k     |
|   8   | 3.307   | 2.230   | **1.48×** |   231k     | 1.264    | (no dist_1024) | —    |    99k     |
|  16   | 1.813   | 1.171   | **1.55×** |   116k     | 0.750    | (no dist_2048) | —    |    49k     |
|  32   | *n/sched*¹ | (no dist_4096) | — | 58k | *n/sched*¹ | 0.2051 | — | 25k |

¹ The two 32-node GPU runs (dist_128 = 128 A100) never scheduled over ~5 h of polling (the GPU
partition was contended/reserved; one's nodes were `DOWN/DRAINED`) — **cancelled**. The 32N CPU
(dars dist_4096) did land. Not re-chased: M5.14 (below) superseded the M5.13 binary anyway.

- **GPU strong-scaling is near-ideal**: NG5 1.85× (4→8N), 1.82× (8→16N); dars 1.84/1.85/1.69× (2→4→8→16N)
  — ~91–93 % parallel efficiency per doubling, easing only at 16N where the columns thin out.
- **The node-for-node GPU/CPU ratio drifts UP with node count** (NG5 1.41× → 1.48× → 1.55× as
  nod2D/rank falls 462k → 231k → 116k): adding nodes spreads the work thinner per GPU. The
  **production rule stands: the GPU is most competitive at the FEWEST nodes (highest per-rank work)
  that fit A100-80GB memory** — over-decomposing wastes the GPU's strength.
  (dars 8/16N + NG5 32N lack a node-for-node CPU partition, so those are GPU strong-scaling only.)
- ⚠️ **The "drift toward a launch/MPI floor" reading (and L58's ~1.4× asymptote) was SUPERSEDED by
  M5.14 (next section):** these M5.13-binary points still had **S as a hidden full-field PCIe chunk**;
  flipping it (Lever A) dropped NG5 4N to 0.879× — *below* parity. So the residual was still mostly
  reducible PCIe, not the launch floor. This section is the (valid) M5.13-binary characterization that
  *motivated* M5.14; for the live ratio see § M5.14.
- Figure: `docs/figures/scaling_hinode_m513.png` (`scripts/plot_hinode_scaling.py`).

### M5.14 — Lever A finished the residency: PARITY CROSSED (2026-05-30)

M5.14 flipped the LAST host-staged nod3D fields M5.13 left — **S** (the g2 mirror-T flip, biggest), **density**,
**fer_w**, **w_i** — to `fesom_halo_field` + full device residency (+ the host salinity-floor → a device kernel,
+ device-side monthly-mean accumulation as the enabler). All four committed and gated (`docs/GPU_FIDELITY.md` §M5.14).

**Same-day baseline (the rigorous comparison — rebuilt the pre-Lever-A `build-cuda`, confirmed pre-S [0 salinity-floor
symbols], ran the identical `job_ng5_prof` today):**

| NG5 dist_16 (4 nodes) | pre-Lever-A | M5.14 | change |
|--|--:|--:|--:|
| step time            | 6.136 s | **3.805 s** | **−38.0 %** |
| deep_copy (PCIe/step)| 11.02 GB | **6.57 GB** | **−40.4 %** |
| deep_copy calls/step | 139 | 121 | −18 |

The same-day GPU baseline 6.136 reproduced M5.13's 6.12 to 0.4 % → today's node conditions match M5.13's. A same-day
CPU dist_512 (4 nodes, `build-serial`, dt=180) re-measure gave **4.327 s/step** (matching the M5.13 ref 4.330 to 0.1 %).
**Node-for-node: GPU 3.805 / CPU 4.327 = 0.879× → the GPU is ~12–14 % FASTER than the CPU** — confirmed same-day on
BOTH sides (no node-noise caveat). **This overturns the M5.13 "~1.4× asymptote"
conclusion above** (§ High-node + L58): that nsys was on the a–f binary BEFORE S/density were flipped — S alone was
still doing ~13 full-field 259 MB D2H/step at NG5 (biggest field, in the biggest phase `13_fct`). Removing it dropped
the GPU step below the CPU. The "~1.4× launch/MPI floor" was a *partial-residency* artifact, not fundamental.

**Regime flipped to compute-bound.** Post-M5.14 nsys kernel summary (rank 0, 8 steps): the `fesom_smooth_nod3D`
smoother is now the **#1 GPU kernel (25.7 % of kernel time)**, FCT/momentum/GM/trdiff filling the rest; `cudaMemcpy`
no longer dominates. Per-phase (`FESOM_STEP_PROFILE`): `3_mixing` 19.9 %, `1b_gm` 16.1 %, `13_fct` 13.1 %,
`fesom_smooth_gather` 8.2 %, `1_eos` 5.3 %, `13b_trdiff` 5.3 %. **The next levers are B (MPI interior/boundary
overlap) and C (kernel coalescing + launch fusion) — NOT more residency** (PCIe is no longer the wall). The device
mean-accum (`resolve_*_dev`) and device salinity floor show at ~0.0 % of kernel time (negligible).

Jobs: `job_ng5_prof_m514` (build-cuda-m514) + `job_ng5_prof` (build-cuda = pre-Lever-A, same-day) + `job_nsys_ng5_m514`.
Commits: t4 `2611936` · S `491ccb8` · density `84c1d8d` · fer_w/w_i `0bc7da9`. Lesson **L59**.

---

*Generated 2026-05-29. Companion: `docs/SCALING_CORE2.md`, `docs/SCALING_FARC.md`,
`docs/SCALING_DARS.md`, `docs/PROFILE_M59.md`. Figures: `docs/figures/scaling_*.png`,
`docs/figures/nsys_ng5_breakdown.png` (the corrected step decomposition — PCIe 75 % / GPU 7 %),
`docs/figures/profile_regime_shift.png` (per-phase composition; read as data-movement shift, not
compute).*
