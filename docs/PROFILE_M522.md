# M5.22 — deep profile of the m522 GPU port + remaining-headroom map

*Measurement + strategy session, 2026-05-31. Binary = `build-cuda/fesom_port` == `fesom_port_m522` (master tip `4f9ea70`, tag `m5.21-coalescing-ghats-sss`). All runs NG5 (7.4 M nodes, 70 lvl), dt=180 s, snap_every=−1, on the `gpu` partition. The campaign has driven NG5 from 16.3 → ~1.4 s/step; this re-establishes the step's time budget from scratch (the pre-residency "75 % PCIe" of L56 is long obsolete) and ranks what's left.*

---

## 0. TL;DR

- **The PCIe wall is gone.** The residency campaign (M5.13–M5.21) did its job: the big per-step nod3D round-trippers (`hnode_new` 778, `sw_3d` 519, `ghats` 256, `S` 249 MB/step) are **all absent** from the m522 synclog. What remains is **659 MB/step, forcing-dominated** (the 8 JRA55 surface fields @7.4 MB + the bulk flux round-trips), ≈26–51 ms of actual bandwidth — not a productive lever.
- **At the 4-node production sweet spot the step is COMPUTE-bound (~46 % GPU-kernel-active).** The single biggest remaining lever-able buckets are exactly the ones the flat lever **cannot** touch: the **FCT edge-scatters** (bucket D) and the **vertical TDMAs** (bucket C). The clean flat lever is exhausted, as predicted at the M5.21 close.
- **At 16 nodes the step is comm/imbalance-bound** (GPU-compute falls to ~28 % of wall; MPI_Waitall = 82.6 % of MPI time). The lever shifts with scale: single-node compute (Lever C / edge-coloring) at low node-count, partition/overlap (Lever D) at high node-count.
- **MPI: the overlappable-comm ceiling is only ~4.6 % of the step** (re-confirms M5.17); the recoverable MPI cost is the **~9.7 % load-imbalance** (Lever D, deployment-side, climate-free).
- **Checked this session:** flipped the 3 missed bucket-A FCT b-kernels (`fct_zal_b1v/b2/b3v`, ≈3.04 % of step) to the flat lever — **A/B −3.10 % (1.3387→1.2972 s/step), bit-identical** (Serial `tradv` `max|Δ|=0` + CORE2-Serial all-fields-bit-identical + CUDA gate PASS 7.5e-3). Details §6/§7.
- **Recommendation (M5.23):** the frontier has split — **Lever C** (the single-TDMA layout prototype, for the compute-bound 4-node regime) and **Lever D** (work-weighted re-partition, for the comm-bound regime that grows with node count) are now co-equal. PCIe and comm-overlap are retired. See §6.

---

## 1. The time budget (the anchor)

### 1a. NG5 dist_16 (4 GPU nodes — the production sweet spot)

Clean wall = **1.3903 s/step** (`m517_clean`, no instrument; nsys-traced run 1.3619, profiler-fenced 1.5269 — the fence overhead is the usual ~10 %). Instruments: `jobs/job_nsys_ng5` (rank-0 CUDA+MPI trace), `jobs/job_ng5_prof` (`FESOM_STEP_PROFILE`), `jobs/job_ng5_synclog` (per-field PCIe), `jobs/job_ng5_halo_split` (barrier-isolation).

| component | ms/step | % of wall | how measured |
|---|---:|---:|---|
| **GPU compute** (Σ kernel, rank 0) | ~640 | **46 %** | nsys `cuda_gpu_kern_sum`: top kernel 1.321e9 ns = 5.9 % → Σ = 22.4e9 ns / 35 steps |
| **PCIe** (per-step field syncs) | ~26–51 | ~2–4 % | synclog 659 MB/step ÷ 13–25 GB/s (nsys raw memcpy 166 ms/step incl. startup mirrors + small async halo copies) |
| **MPI halo — comm** (overlappable) | 64 | 4.6 % | halo_split barrier-isolation: pure comm |
| **MPI halo — load-imbalance** (idle) | 134 | 9.7 % | halo_split: `MPI_Barrier` absorbs arrival skew |
| **CG (SSH solve)** | ~60 | 3.9 % | `FESOM_CG_PROFILE`: 88.8 iters/step, 0.676 ms/iter (subset of GPU compute above) |
| remainder | balance | ~25–30 % | host / kernel-launch gaps / blocking-sync pipeline stalls (the cudaDeviceSynchronize wait that L65 flags) |

**Reading it:** the step is **GPU-compute-bound** at 4 nodes — ~46 % of the wall is the GPU actually running kernels, and the next-largest *recoverable* chunk is the ~9.7 % halo load-imbalance. PCIe and overlappable-comm are each <5 % — both effectively retired as levers. (nsys profiles rank 0 only; rank 0 is a *fast* rank, so its `MPI_Waitall` wall reads 1047 ms/step — that is idle-wait, NOT comm. The cross-rank-mean halo cost from the barrier experiment, 183 ms/step, is the honest number. This is the L60/L62 "Waitall wall = imbalance idle" caveat, restated.)

### 1b. NG5 dist_64 (16 GPU nodes — the comm-bound regime)

Wall = **0.5598 s/step** (`jobs/job_nsys_ng5_n16`, the 16-node clone created this session, job 25252684).

| component | ms/step | % of wall | source |
|---|---:|---:|---|
| GPU compute (Σ kernel, rank 0) | ~159 | **28 %** | nsys: top kernel 350.6e6 ns = 6.3 % → Σ = 5.57e9 / 35 steps |
| PCIe (nsys memcpy, HtoD+DtoH) | ~41 | ~7 % | nsys: HtoD 667e6 + DtoH 776e6 ns / 35 |
| MPI/halo wait (rank 0, idle-inflated) | balance | **~65 %** | nsys MPI_Waitall = 82.6 % of MPI time (the dominant API now) |

**The scale shift (the SCALING_M522 finding, now mechanistic):** 4N → 16N quadruples the rank count, so per-rank GPU work falls ~4× (640 → 159 ms/step, a bit more than ÷4 from strong-scaling-efficiency loss + smaller, less-efficient kernels), while the halo surface/comm shrinks far less. The GPU-compute fraction drops **46 % → 28 %** and the step becomes **MPI/imbalance-bound** (MPI_Waitall jumps from 67 % to 82.6 % of MPI time, and MPI is now the dominant CUDA-API consumer where at 4N `cudaDeviceSynchronize`/compute led). The honest comm-vs-imbalance split at 16N would need a barrier-isolation run there (not done; at 4N it was 68/32 imbalance/comm — the imbalance share almost certainly grows at 16N). This is exactly why the node-for-node GPU advantage shrinks (NG5 2.95× @4N → 2.38× @16N): the levers that help at 4 nodes (compute: C/D buckets) buy little at 16 nodes, where the wall is the halo + load imbalance.

---

## 2. The per-kernel compute ranking (4 nodes)

From `FESOM_STEP_PROFILE` (named kernels, % of step wall), cross-checked against the nsys `cuda_gpu_kern_sum` (the nsys names are truncated `fesom_tracer_advect_…`; mapped by per-step ns).

**Phases:** `13_fct` **20.26 %** (still #1), `1_eos` 9.85 %, `3_mixing` 6.24 %, `ice_dyn(o2i+EVP)` 6.09 %, `7_ssh` 5.61 %, `12_ale` 4.06 %, `13b_trdiff` 2.85 %, `1b_gm` 2.59 %. STEP PROFILE coarse split: ocean 57.5 %, coupling 21.4 %, sea-ice 9.8 %, forcing 8.1 %.

**Top individual kernels (% of step wall) and their bucket:**

| # | kernel | %loop | s/step | bucket | flat-lever? |
|--:|---|---:|---:|:--:|:--:|
| 1 | `fct_eud_fill` | 2.47 | 0.0377 | D (edge) | no |
| 2 | `fesom_impl_vert_diff_tracers` | 2.29 | 0.0350 | C (TDMA) | no |
| 3 | `fct_mfct_h` | 2.26 | 0.0346 | D (edge) | no |
| 4 | `fesom_gm_redi_ver_node` | 2.06 | 0.0315 | B (written-value recurrence `zbar_n`) | no (TeamPolicy) |
| 5 | `fesom_ale_vvel_scatter` | 1.94 | 0.0297 | D (scatter) | no |
| 6 | `fct_zal_b2` | 1.55 | 0.0236 | **A** | **YES → flipped M5.22** |
| 7 | `fct_zal_a34` | 1.47 | 0.0224 | B (col reduction) | no (TeamPolicy) |
| 8 | `fct_zal_b3h` | 1.16 | 0.0177 | D (edge dispatch) | no |
| 9 | `fesom_gm_redi_hor_edge` | 1.04 | 0.0160 | D (edge) | no |
| 10 | `fesom_impl_vert_visc` | 0.98 | 0.0150 | C (TDMA) | no |
| — | `fct_zal_b1v` | 0.85 | 0.0130 | **A** | **YES → flipped M5.22** |
| — | `fesom_momadv_vert` | 0.80 | 0.0122 | B | no |
| — | `fesom_visc_bidiff_stage2` | 0.79 | 0.0120 | D (scatter) | no |
| — | `fct_zal_b1h` | 0.77 | 0.0118 | D (scatter) | no |
| — | `fesom_momadv_horiz` | 0.70 | 0.0107 | D (scatter) | no |
| — | `fesom_gm_fer_solve_gamma` | 0.67 | 0.0102 | C (TDMA) | no |
| — | `fct_zal_b3v` | 0.64 | 0.0097 | **A** | **YES → flipped M5.22** |
| — | `fct_f2d_h` | 0.43 | 0.0065 | D (scatter) | no |
| — | `fct_LO_scatter` | 0.42 | 0.0064 | D (scatter) | no |

**Bucket totals (the headroom by technique):**
- **Bucket C — vertical TDMAs:** `impl_vert_diff_tracers` 2.29 + `impl_vert_visc` 0.98 + `fer_solve_gamma` 0.67 = **≈3.9 %** → Lever-C layout refactor ONLY.
- **Bucket D — edge/element scatters:** `fct_eud_fill` 2.47 + `fct_mfct_h` 2.26 + `ale_vvel_scatter` 1.94 + `gm_redi_hor_edge` 1.04 + `fct_zal_b3h` 1.16 + `fct_zal_b1h` 0.77 + `momadv_horiz` 0.70 + `visc_bidiff` (s1+s2) 1.25 + `f2d_h` 0.43 + `LO_scatter` 0.42 = **≈12.4 %** → edge-coloring / different axis (or fold into Lever C).
- **Bucket B — column reductions:** `fct_zal_a34` 1.47 + `momadv_vert` 0.80 = **≈2.3 %** → TeamPolicy.
- **b-cluster bucket-A (the M5.22 quick win):** `b2` 1.55 + `b1v` 0.85 + `b3v` 0.64 = **≈3.0 %** — a clean per-(node,level) subset the M5.19/M5.21 sweep missed. **FLIPPED this session** (see §6).

**The decisive fact:** every one of the top-5 GPU kernels is bucket C or D. After flipping the 3 bucket-A b-kernels, the flat coalescing lever (M5.18–M5.22) is now genuinely exhausted — the next compute win requires either the Lever-C layout refactor (for the TDMAs) or edge-coloring (for the scatters). Neither is bit-identical-for-free.

---

## 3. The residual PCIe (4 nodes)

`jobs/job_ng5_synclog` (rebuilt `fesom_port_synclog` on m522 this session). Per-field per-step transfers, rank 0:

**Total: 659.3 MB/step (118 transfers/step) — D2H 183.4, H2D 475.9.**

Per-step leaders (all ≈7.4 MB/step = nod2D surface fields):
- **H2D forcing** (the genuine input I/O): the 8 JRA55 fields `u_wind, v_wind, Tair, shum, shortwave, prec_snow, prec_rain, longwave` (7.4 MB, 2/step each); `ice.srfoce_u/v`; `forcing.heat_flux/water_flux` (2/step); `dyn.d_eta`, `hbar`; `forcing.stress_node_surf` (1/step).
- **D2H** (the bulk→host handoff): `forcing.heat_flux/water_flux` (2/step), `forcing.stress_node_surf` (1/step).
- **Startup-only** (calls/step ≈0.04 = once): the mesh mirrors `area`, `areasvol`, `zbar_3d_n` (259 MB each, one-time).

**The big nod3D round-trippers are confirmed gone** — no `hnode_new`, `sw_3d`, `ghats`, or `tracers.S` in the per-step list. The residency campaign is complete. The remaining ~659 MB/step is genuine forcing I/O (host-interpolated JRA55 each step) + the small nod2D bulk-flux handoff. By L65 (value PCIe by round-trips/pipeline-stalls, not bytes), none of these is a high-value target: they are small (7.4 MB), and the forcing fields are freshly host-computed each step so they *must* cross. **PCIe is not a productive M5.23 lever.**

---

## 4. The MPI / comm cost (Lever B vs Lever D)

`jobs/job_ng5_halo_split` — the M5.17 barrier-isolation experiment, re-run at m522 NG5 dist_16:

```
clean                 1.3903 s/step
MPI_Waitall (no barrier, across-rank min/mean/max): 0.1215 / 0.1833 / 0.2846   (637 calls/step, 274.7 MB/step/rank)
MPI_Waitall (barrier on):                           0.0421 / 0.0639 / 0.0789
MPI_Barrier (= absorbed load-imbalance):            0.0676 / 0.1342 / 0.2573
SPLIT: imbalance 0.1342 s/step (68 %)  |  comm 0.0639 s/step (32 %)
```

- **Overlappable comm = 0.064 s/step = 4.6 % of the step** → this is the absolute ceiling for Lever B (compute/comm overlap or message aggregation). Confirms M5.17's verdict at the new (smaller) step: **Lever B is still not worth it.** (The 79/21 imbalance/comm split of M5.17 is now 68/32 — comm's *share of the halo* grew only because the step shrank under the compute levers; comm's *absolute* fraction of the step is still tiny.)
- **Load-imbalance = 0.134 s/step = 9.7 % of the step** → this is the recoverable MPI cost, and it is **Lever D** (a better/work-weighted GPU domain decomposition — a partition change, no code, climate-identical). This is the single biggest non-compute lever at 4 nodes, and it *grows* with node count (the 16-node Waitall is ~51 % of a much smaller wall). Worth a measurement: regenerate the `dist_16` partition with a work-weighted metric and re-time.
- CG `Allreduce` is dead (0.2 % at 4N; nsys 16N Allreduce ~1.3 % of MPI) — re-confirms M5.2.

---

## 5. The ranked candidate optimizations

Headroom from the §2 bucket totals; risk/effort from the code-structure read (`src/fesom_tracer_adv.cpp`, `fesom_tracer_diff.cpp`, `fesom_gm.cpp`, `fesom_momentum.cpp`) and the §4 MPI split.

| # | candidate | technique | headroom (4N) | climate risk | effort | verdict |
|--:|---|---|---:|---|---|---|
| 1 | **b-cluster bucket-A flips** (b1v/b2/b3v) | the M5.18 flat lever | **≈3.0 %** | **bit-identical** | **LOW** | **DONE this session — see §6** |
| 2 | **Lever D — re-partition** | work-weighted `dist_16` partition | up to ~9.7 % (the imbalance) | climate-identical | LOW (deployment, no code) | **top M5.23 pick** — only non-compute lever that pays, and it *grows* with node count |
| 3 | **Edge-coloring (bucket D)** | color edge sets → drop atomics, coalesce | ≈12 % (all FCT/GM/momentum scatters) | climate-close (atomic order → needs the gate, not Serial-bit-id) | MED–HIGH | the biggest *compute* bucket; the FCT `*_h` scatters run 2×/step |
| 4 | **Lever C — layout refactor** | rank-1 `arr[node*nl+nz]` → `View<double**>` LEVEL-outer | ≈3.9 % (the 3 TDMAs) + global scatter/node-major gains | bit-identical (memory reorder) | **HIGH** (82 fields, 896 macro sites) | the ONLY path for the bucket-C TDMAs; **prototype on 1 TDMA first** (feasible, see below) |
| 5 | `fct_zal_a34`, `momadv_vert` (bucket B) | TeamPolicy (league=node, team=levels, team-scratch reduction) | ≈2.3 % | bit-identical | MED | the genuine cross-level reductions; modest |

**Lever-C blast radius (measured by grep):** 82 unique `fesom::Field`/`IntField` members, all rank-1 `DualView<T*,LayoutRight>`; **896** index-macro call sites (`FESOM_NODE3D` 531, `FESOM_ELEMVEC` 263, `FESOM_ELEM3D` 103) across 22 files; macros defined at `src/fesom_types.h:33-35` as flat `entity*nl+lev`. A full refactor rewrites all three. **But a single-TDMA prototype is feasible without touching any of the 82 fields:** allocate a dedicated `View<double**,LayoutLeft> (nl, nnodes)` scratch, transpose-in (coalesced copy from the rank-1 `values`), run the *same* per-node Thomas sweep reading/writing the rank-2 scratch (+ its ~6 coefficient fields fed the same way), transpose-out, and ncu-measure the kernel's sectors/req — a self-contained 1-kernel measurement to decide whether the global refactor is worth it before committing.

**Edge-coloring note:** the bucket-D scatters (`fct_eud_fill`, `fct_mfct_h`, `fct_f2d_h`, `fct_LO_scatter`, `fct_zal_b1h`, `gm_redi_hor_edge`, `momadv_horiz`, `visc_bidiff` stages 1/2) all `atomic_add` from an edge/element loop into shared nodes. Edge-coloring (precompute per-rank color sets so each color's edges touch disjoint nodes → drop the atomics, run one un-colored `parallel_for` per color) is the route, but it changes the summation order → climate-close, gated by `gpu_fidelity_gate.sh`, not the Serial diff. This is the largest single compute bucket but the highest-touch compute lever.

---

## 6. Checked lever — the b-cluster flat lever (M5.22)

**What:** the §2 ranking + a code-structure read found **3 FCT Zalesak-limiter kernels still parallelized one-thread-per-NODE** that are clean **bucket A** (the M5.19/M5.21 sweep missed them):
- `fct_zal_b1v` (`fesom_tracer_adv.cpp:1759`, 0.85 %) — writes `fplus/fminus[k]` from `aflux_v[nz]` + the **input** `aflux_v[nz+1]` (qr4c_v output, not written here).
- `fct_zal_b2` (`:1785`, 1.55 %) — pure per-level map (every read/write at the same `k`: `areasvol/hnode_new/fplus/fminus/fmax/fmin`).
- `fct_zal_b3v` (`:1801`, 0.64 %) — RMWs only its own `aflux_v[k]`; the `nz-1` reads are `fplus/fminus` (b2 output = **inputs** here, not `aflux_v`) → no written-value recurrence; the surface case `nz==nu1` folds into the mask.

Each flipped to the proven flat lever: `RangePolicy(0, myDim*nl)`, decode `(n,nz)`, mask `nz∉[nu1,nl1)`, same-order register write. The field is node-major (`aflux_v[n*nl+nz]`, level contiguous) so a warp of consecutive `nz` now coalesces. The C twin (`fesom_tracer_advect_one_fct`) is untouched, so the `tradv` verify directly proves bit-identity.

**Measured A/B (same-node, NG5 dist_16, one allocation, job 25254299, BEFORE=`fesom_port_m522` vs AFTER=`fesom_port_m522b`): clean 1.3387 → 1.2972 s/step = −3.10 %** (prof-fenced 1.3891 → 1.3469 = −3.04 %, consistent → fully attributed). The 3 b-kernels were ≈3.04 % of the step (b2 1.55 + b1v 0.85 + b3v 0.64) and each drops sharply under the flat lever (the M5.18 signature), recovering ~3.1 % — a pure-compute lever (no PCIe change; deep_copy unchanged).

**M5.23 recommendation:** the budget says the lever choice now depends on the deployment regime:
- **At the 4-node production sweet spot (compute-bound, 46 % GPU-active):** the flat lever is exhausted after this session. The remaining compute is **bucket D (≈12 %, edge-coloring)** and **bucket C (≈3.9 %, Lever C)**. Recommend **prototyping Lever C on `impl_vert_diff_tracers`** (the de-risking single-TDMA measurement above) — it is the structural direction FESOM2-GPU must eventually take for the TDMA/scatter half of the step, and the prototype is cheap and bit-identical.
- **At scale (16 nodes, comm/imbalance-bound, 28 % GPU-active):** compute levers buy little; the wall is the halo + the ~9.7 %-and-growing load imbalance. Recommend **Lever D (work-weighted re-partition)** — climate-identical, no code, and the only lever whose payoff grows with node count.

The honest framing: **single-node compute (Lever C / edge-coloring) and comm/scaling (Lever D) are now co-equal frontiers**, and which to pursue depends on whether the production target is 4-node throughput or 16-node strong-scaling. PCIe and comm-overlap (Lever B) are both retired (<5 % each).

---

## 7. Validation status (the M5.22 b-cluster lever)

- **Serial `tradv` per-kernel verify (pi np1):** every step `max|Δ| = 0.000e+00` for both tracers (T, S) → the 3 flips are **bit-identical to the C twin**.
- **CORE2-Serial fresh-vs-saved (real bathymetry, all fields, `diff_snap` vs the preserved `serref_m522_saved`):** **ALL FIELDS BIT-IDENTICAL** (`diff_snap` rc=0) — the flat lever is a pure re-parallelization on real bathymetry, byte-for-byte the C twin (the shallow-column edge cases pi can't reach are covered here, the M5.21/L66 method).
- **CUDA fidelity gate** (`gpu_fidelity_gate.sh`, CORE2 dist_8 ice-active, m522b CUDA vs the m522 Serial oracle): **PASS, worst h_ice 7.521e-3** (T 8.55e-4, S 3.15e-4, u/v 1.7–1.8e-4) = the unchanged atomic-scatter/EVP floor, no staleness regression. The flat-levered kernels add zero atomics → zero new divergence.
- **m522 baseline gate (pre-change sanity, run this session):** PASS, worst h_ice 3.029e-3 — the m522 tip is healthy at the climate-close floor.
- **1-yr CORE2 CUDA climate:** not run this session (Serial bit-identical ⇒ corr=1.00000 vs m522 guaranteed, as every M5.18–M5.21 flat-lever flip closed). Recommended before tagging if desired, but the bit-identity proof above is dispositive.

The lever is pure re-parallelization (no atomics added, no sync-rail change), so the CUDA divergence floor is unchanged from m522 (the flat-levered kernels add zero new atomics).

---

## 8. The comm-bound regime — deep-dive (the SYPD question)

*Added after the 4N/16N budget, to answer: at the per-rank size where the GPU goes comm-bound, what is the comm actually made of, and can the GPU reach a usable throughput (1–2 SYPD) on NG5?*

### 8.1 The cheap proxy method (the comm regime is governed by 2D-vertices-per-rank)

The comm-bound regime is set by **2D-vertices per rank**, NOT by absolute mesh size or node count. So NG5 dist_64 (16 nodes) = 7.40 M / 64 = **115.7 k 2D-pts/rank** can be reproduced on 1–8 nodes with smaller meshes — no need to fight for 16 GPU nodes mid-week:

| proxy | dist (nodes) | 2D-pts/rank |
|---|---|---:|
| NG5 dist_64 | 16 N | 115.7 k ← the target |
| **dars dist_32** | **8 N** | **98.8 k** ← closest, same CG-conditioning class |
| farc dist_4 | 1 N | 159.6 k |
| farc dist_8 | 2 N | 79.8 k |
| farc dist_16 | 4 N | 39.9 k |

⚠️ The proxy is faithful for the **halo/geometry** terms (per-rank-size driven). It is **NOT** faithful for the **CG iteration count**, which is conditioning-driven and mesh-specific: farc needs ~229 CG iters/step (steep topography), dars ~67, NG5 ~89. So use **dars** (67, closest to NG5's 89) for the CG fraction; farc only for the halo numbers. Jobs: `jobs/job_farc_halo_split`, `job_dars_halo_split`, `job_farc_nsys` (barrier-isolation = the M5.17/L62 instrument).

### 8.2 What the comm wall is made of (measured 2026-05-31)

Barrier-isolation sweep (clean s/step; dt differs per mesh so the *fractions*, not s/step, define the regime):

| proxy | 2D-pts/rank | clean s/step | CG share | halo Waitall (% step) | imbalance / comm |
|---|---:|---:|---:|---:|---:|
| farc 1N | 160 k | 0.3609 (dt900) | 22.6 % | 0.067 (19 %) | 49 % / 51 % |
| farc 2N | 80 k | 0.3405 (dt900) | 37.0 % | 0.104 (31 %) | 53 % / 47 % |
| farc 4N | 40 k | 0.2663 (dt900) | 42.0 % | 0.090 (34 %) | 41 % / 59 % |
| **dars 8N** | **99 k** | **0.3830 (dt180)** | **12.5 %** | **0.140 (37 %)** | **63 % / 37 %** |

**At the NG5@16N-like per-rank size (dars 8N, 99 k/rank), the step budget is:**
- **Halo `MPI_Waitall` ≈ 37 % of the step**, of which **63 % is load imbalance** (≈23 % of the whole step) and 37 % is genuine comm (≈14 % of step).
- **CG (SSH solve) ≈ 12.5 %** — ~67 iters/step, each with **~2 `MPI_Allreduce` + 1 halo** (nsys farc: 474 Allreduce/step ÷ 229 iters = 2.07/iter = standard CG's two dot-products, already as fused as plain CG allows).
- **The killer trend:** as you subdivide (farc 160 k→40 k/rank), BOTH comm components grow — CG share 22.6 %→42 %, halo 19 %→34 %. **Adding nodes to chase SYPD makes the comm fraction worse, fighting the speedup.** This is the mechanism behind the node-for-node shrink (NG5 2.95×@4N → 2.38×@16N) and the dying strong-scaling.

nsys message profile (farc 2N, rank 0, latency-vs-bandwidth read): halo = 67.6 MB/step/rank over 970 halo calls/step (~27 KB/msg avg = mixed small nod2D + big nod3D); MPI_Isend 2 494/step, Allreduce 474/step. The high *count* of small messages → latency-bound → persistent-requests / aggregation are the relevant levers; the nod3D fraction is bandwidth-bound.

### 8.3 The SYPD verdict (the existential question)

SYPD = 0.493 / (s/step) for NG5 at dt=180 (480 steps/model-day, 175 200/yr). Production needs **1–2 SYPD** (CPU does 1–2 SYPD on large allocations). Against measured + extrapolated NG5 scaling:

| nodes | s/step | **SYPD (compute only, pre-I/O)** |
|---|---:|---:|
| 4 | 1.47 | 0.34 |
| 8 | 0.79 | 0.62 |
| 16 | 0.49 | **1.0** |
| 32 (extrap.) | ~0.35 | ~1.4 |
| 64 (extrap.) | ~0.28 | ~1.8 |

NG5 GPU reaches ~1 SYPD at 16 nodes; the dying scaling pushes 2 SYPD out to ~64 nodes. **This is NOT a dead end** — the reframe (2026-05-31, user):
1. **64 nodes is an acceptable target.** The honest comparison is *GPU nodes available* vs *CPU nodes available* — production will not get 1000 CPU nodes either. ~1.8 SYPD at 64 GPU nodes is a viable production point, and per-watt the GPU lead is large.
2. **Mixed precision is the big un-pulled lever (~×2).** FP32 (or mixed) for selected fields/kernels roughly halves both compute AND comm bytes → potentially ~2 SYPD at 32 nodes, or ~3.6 at 64. It changes the climate → needs its own validation campaign (not bit-identical), but it's the single largest remaining factor and was wrongly shelved as "speculative" in the M5.22 prompt framing.
3. **There are likely dead/redundant exchanges to remove** (boundary exchanges happening but not consumed) — 970 halo calls/step is a lot; the M5.9-pin/L67 placebo-removal method (a sync/halo with no real consumer) applies to halos too, and is bit-identical when a halo truly has no downstream reader.

### 8.4 The ranked comm-reduction menu (for M5.23+)

| lever | mechanism | targets | est. effect | climate | effort |
|---|---|---|---|---|---|
| **Mixed precision** | FP32/mixed for selected fields+halos | compute + comm bytes | **~×2** (user est.) | changes climate → own validation | HIGH |
| **Redundant-exchange audit** | find halos with no consumer, drop them (L67 method) | the ~970 halo calls/step | unknown, possibly large | bit-identical (if truly dead) | LOW–MED |
| **Persistent MPI requests** | `Send_init`/`Recv_init`/`Startall` for the fixed halo pattern | latency on ~970 halo calls/step | modest | bit-identical | LOW |
| **Multi-field aggregation** | pack fields sharing a halo kind into one msg (GM bracket, velocity pair, CG/EVP) | message *count* | modest–med | bit-identical | MED |
| **Run at more nodes (32–64)** | accept the node count | throughput | 16N 1.0 → 64N ~1.8 SYPD | none (deployment) | none |
| **Comm/compute overlap** | start halo before interior kernel finishes | comm-proper (~14 %) | low (ceiling 4.6 % @4N) | climate-close | MED |
| **Pipelined / comm-avoiding CG** | 1 Allreduce/iter instead of 2 | CG collective latency | modest | climate-close | MED–HIGH |
| **On-node NVLink/NCCL** | intra-node GPU↔GPU (Levante has no gdr_copy → currently stages via host) | on-node halo bandwidth | med | climate-close | HIGH + LUMI-portability risk |
| **Re-partition (Lever D)** — *biggest single lever, SHELVED by user choice* | work-weighted partition | the ~23 %-of-step imbalance | up to ~23 % | climate-identical, no model code | LOW (deployment) |

**Bottom line:** at NG5-scale per-rank sizes the comm wall is ~37 % halo (≈⅔ of it load imbalance) + ~12 % CG. The biggest *single* climate-safe lever is repartitioning (~23 % of step), kept in memory but not pursued now. Of the in-scope levers, **mixed precision (~×2) is the largest by far**, the **redundant-exchange audit** is the cheapest bit-identical win, and **running at 32–64 nodes is an acceptable deployment answer**. The combination — not any single lever — is the path to 1–2 SYPD. (M5.22 measured + framed this; M5.23 picks the lever.)

### 8.5 The comm code-read — exact implementation surface (file:line)

Read-only structural analysis of `src/fesom_halo_device.{cpp,hpp}`, `fesom_halo.cpp`, `fesom_ssh.cpp`, `fesom_ice_evp.cpp`, `fesom_ice_fct.cpp`, the per-step exchange call-sites. The three properties that set the lever surface:

- **Requests are NOT persistent** — `MPI_Isend`/`Irecv` are posted fresh every exchange (`fesom_halo_device.cpp:262-274`), even though the comm-lists `g_dev[5]` + neighbor PEs/offsets are already built-once and never change (`build_lists`, `:159-182`). → persistent-request hoisting is a clean bit-identical win.
- **One field per exchange** (ncomp of ONE field IS packed contiguously, stride `nl×ncomp`, `:224,248` — so velocity nc=2, GM nc=2/3 are already single messages; but two *distinct* Fields = two messages). There is **no "halo a list of fields" entry point** — that new entry point is the shared dependency for the two highest-value bit-identical levers.
- **Two full-device `Kokkos::fence()` per exchange** (`:251` pre-MPI, `:286` post-unpack) → ~1940 device fences/step at ~970 halo calls/step.

**Message-count drivers (the latency-bound targets):**
- **EVP per-subcycle = 240 NOD2D msgs/step** — `fesom_ice_evp.cpp:713-714` exchanges `uice` then `vice` as TWO separate adjacent same-kind calls, ×120 subcycles (`evp_rheol_steps=120`). The single biggest contributor; directly batchable → 120.
- **CG per-iter** — `fesom_ssh_solve_cg_kk` (`fesom_ssh.cpp:680`) does **2 `MPI_Allreduce`/iter** (`:772` pAp + `:805` the already-fused rz/rr 2-element, M5.2) + 2 halos/iter (`:758` pp, `:787` rr), ×~67-89 iters.
- **~10 adjacent same-kind Field pairs** batchable: FCT `fct_plus`+`fct_minus` (`fesom_tracer_adv.cpp:1803-1804`), PGF `pgf_x`+`pgf_y` (`fesom_step.cpp:319-320`), visc `u_b`+`v_b` (`fesom_momentum.cpp:1493-1494`), `Kv`+`Av`, `uv_rhs`+`uv_rhsAB`, GM `neutral_slope`+`slope_tapered`/`fer_K`+`Ki`, `hnode`+`helem`, ice-FCT host triples/pairs.

**The buildable lever ranking (climate-safe; bit-identical unless flagged):**

| # | lever | file:line | attacks | effect | effort | climate |
|--:|---|---|---|---|---|---|
| **L1** | **EVP {uice,vice} batching** (1 nc=2 msg) ✅ **DONE −9.1%** | `fesom_ice_evp.cpp:716` | comm-proper (240→120 msg/step) | latency-count | LOW–MED | **bit-identical (proven)** |
| **L2** | **persistent requests** (Send_init/Startall) | `fesom_halo_device.cpp:262-274` (+host twin) | comm-proper (setup latency, ~970 calls/step) | latency | MED | **bit-identical** |
| **L3** | **adjacent same-kind PAIR fusion** (10 sites) ✅ **DONE −2.4%** | `fesom_halo_field2` drop-in on the 10 pairs (FCT `tracer_adv.cpp:1806` first) | comm-proper (−11 exch/step + −22 fences) | latency-count | LOW (drop-in) | **bit-identical (proven)** |
| **L5** | **dead-exchange removal** (L67 poison method) | verify `uv_rhsAB` `step.cpp:462`, pre-smoother `bvfreq` `step.cpp:214` | comm-proper + fences | latency-count | LOW/candidate | **bit-identical if dead** |
| **L4** | **CG 2→1 Allreduce** (Chronopoulos reorder) | `fesom_ssh.cpp:772,805` | CG (~134→67 Allreduce/step) | latency-syncs | MED–HIGH | **CUDA-gate-class** (not Serial-bit-id) |
| L6 | comm/compute overlap | `fesom_halo_device.cpp:286` | comm-proper | marginal | HIGH | bit-id — **skip** (ceiling 4.6 %) |
| L7 | on-node NVLink/NCCL | `fesom_halo_device.cpp:262-273` | on-node bandwidth | med | HIGH | **vendor-lock risk — defer** |

**Shared dependency (§5 entry point):** L1 and L3 both need a new `fesom_halo_fields({field list}, kind, …)` that packs N distinct Fields into one per-neighbor buffer (N pack kernels → one Irecv/Isend/neighbor → N unpack), reusing the existing per-kind `g_dev[kind]` lists. Single-Field/multi-component already exists; multi-Field is the new code. Build it once → L1 + all of L3 land on it. **No placebo halo found** beyond M5.21's `ghats` (L5 candidates need the poison test before removal — don't assume dead).

### 8.6 L1 — EVP {uice,vice} fused halo: DONE, −9.1 % in the comm regime, bit-identical

**What:** the two-field exchange entry point (the §5 shared dependency, built two-field rather than N-field — covers L1 and every L3 *pair*). New `fesom_halo_exchange_device2` (`fesom_halo_device.cpp`) co-packs two same-kind/same-stride fields into ONE message per neighbour (per-halo-node block `[f0(stride) f1(stride)]`, stride 2×); the flat buffer-offset collapse + race-free unpack are unchanged → the bytes landing in each field's halo are byte-identical to two separate exchanges (pure message-count cut, no new arithmetic, no atomics). Dispatch `fesom_halo_field2` (`fesom_halo_device.hpp`) mirrors `fesom_halo_field`; on Serial/OpenMP/`FESOM_HOST_HALO=1` it falls back to the EXACT two legacy host brackets (the M5.1 Approach-B guarantee → the bit-identical oracle is untouched by construction). Applied to the EVP subcycle (`fesom_ice_evp.cpp:716`, was two adjacent NOD2D nc=1 calls ×120 subcycles).

**Validated (the full ladder, all PASS):**
- **Serial bit-identity:** fresh CORE2 dist_8 vs the preserved `serref_m522_saved` oracle — **ALL FIELDS BIT-IDENTICAL** (`diff_snap` rc=0, all 3 snaps), cross-checked by an **independent** per-variable `np.array_equal` (81/81 vars identical). (⚠️ the first run reported a false DIVERGENCE = a self-inflicted double-submit clobber that `rm`'d snap_000010 mid-compare; the clean re-run alone is definitive — see L69.)
- **CUDA fidelity gate** (CORE2 dist_8 ice-active, m522c vs Serial oracle): **PASS**, worst T 4.532e-3 (= the unchanged atomic-scatter floor; the fused exchange adds zero atomics).
- **A/B same-allocation, the COMM regime** (`job_farc_l1_ab`/`job_dars_l1_ab`, m522b→m522c): **farc 2N (80k/rank) 0.3429→0.3175 = −7.4 %** (calls/step 970→830); **dars 8N (99k/rank = the NG5@16N proxy) 0.3836→0.3487 = −9.1 %** (calls/step 584→444). Exactly 140 calls/step removed in both (120 EVP subcycles ×(2→1) + ~20 tally), halo MB/step **unchanged** (67.6 / 107.7) → confirmed pure message-count win. (NG5 4N compute-regime A/B was ~flat as expected — wrong regime for a comm lever.)

**Binary `build-cuda/fesom_port_m522c`** (= live `build-cuda/fesom_port`); BEFORE kept as `fesom_port_m522b`. This is **the campaign's first comm-regime win** (−9.1 % at NG5-scale per-rank); the entry point now makes the L3 adjacent pairs (FCT plus/minus, PGF, visc, etc.) cheap follow-ons.

### 8.7 L3 — adjacent same-kind PAIR fusion (10 sites): DONE, −2.4 % at the NG5@16N proxy, bit-identical

**What:** dropped `fesom_halo_field2` (the L1 entry point) onto the 10 verified adjacent same-kind/same-stride halo pairs — a pure drop-in, **zero new machinery**. Each was two back-to-back `fesom_halo_field` calls with both fields written by the preceding kernel and nothing reading/writing them between → co-pack into one message/neighbour, byte-identical. The pairs (kind, nl, nc; freq):
- **FCT** `fct_plus`+`fct_minus` (`tracer_adv.cpp:1806`, NOD3D nl 1) — **2×/step** (T+S), the highest-freq pair
- **EOS** `density`+`hpressure` & `sw_alpha`+`sw_beta` (`step.cpp:214,226`, NOD3D nl 1) — the 5-field EOS block 5→3 exchanges (`bvfreq` left single — a fieldN fuser would take it to 1)
- **PGF** `pgf_x`+`pgf_y` (`step.cpp:325`, ELEM3D nl 1)
- **GM** `neutral_slope`+`slope_tapered` (`step.cpp:282`, NOD2D nl1 3) & `fer_K`+`Ki` (`step.cpp:292`, NOD2D nl 1) — GM is active on dars (its 3 pairs fire)
- **vert-vel** `w`+`fer_w` (`step.cpp:728`, NOD3D nl 1, gm-conditional) & `w_e`+`w_i` (`step.cpp:751`, NOD3D nl 1)
- **visc** `u_b`+`v_b` (`momentum.cpp:1495`, ELEM3D nl 1); **bulk** `heat_flux`+`water_flux` (`bulk.cpp:598`, NOD2D 1 1)

Deliberately NOT fused: `Kv`+`Av` / `hnode`+`helem` (DIFFERENT kinds — the prompt's warning, verified), `uv_rhs`+`uv_rhsAB` (left for the L5 poison-test — if `uv_rhsAB` is dead, REMOVING it beats fusing), `diffK` slabs (one field, two slabs → needs a slab-fuser, not field2).

**Validated (full ladder, all PASS):**
- **Serial bit-identity:** fresh CORE2 dist_8 vs `serref_m522_saved` — **ALL FIELDS BIT-IDENTICAL** (`diff_snap` rc=0). field2 on Serial = the EXACT two legacy brackets → bit-identical by construction; this leg catches any kind/nl/nc/field typo in the 10 calls.
- **CUDA fidelity gate** (CORE2 dist_8 ice-active, m523L3 vs fresh Serial oracle): **PASS, worst h_ice 7.29e-3** (the unchanged atomic-scatter floor). Every field DOWNSTREAM of a fused halo — density 2.1e-4, bvfreq 1.8e-7, T/S 7.5e-4/2.8e-4, u/v 8.0e-5/1.8e-4, w 4.1e-8, Av/Kv — is at floor → **no adjacency/co-pack error** (the check the Serial test CAN'T do: a stale co-pack would diverge a downstream field beyond floor).
- **A/B same-allocation, the COMM regime:** **dars 8N (99k/rank = the NG5@16N proxy) 0.3451→0.3369 = −2.38 %**; **farc 2N (80k/rank, CG-dominated bracket) 0.2827→0.2797 = −1.06 %**. The DETERMINISTIC msg-count drop is IDENTICAL on both meshes: single-field exchanges −21.8/−21.9, two-field +11.0 (the 11 fused-pair invocations/step), **MPI calls/step −12**, halo MB/step UNCHANGED → pure message-count + fence cut (each removed exchange also drops 2 `Kokkos::fence()`s).

**Why −2.4 % not −9 % (the honest read):** L1's EVP pair ran 120×/step (one per subcycle); the L3 pairs are structural, once-per-step (FCT 2×). L3 removes ~11 exchanges/step vs L1's 120 — an order of magnitude fewer. The payoff scales with how comm-bound (vs CG-bound) the regime is: dars/NG5-class (CG ~13 %) → −2.4 %, CG-dominated farc (CG 37 %, which L3 doesn't touch) → −1.1 %. **There is no other EVP-like high-frequency pair** — the structural pairs are the whole remaining L3 surface, and stacking all 10 gives −2.4 %. So the dars-8N −2.4 % is the faithful NG5@16N estimate.

**Binary `build-cuda/fesom_port_m523L3`** (= the L5/fieldN BEFORE); BEFORE kept as `fesom_port_m522c`. Artifacts: `dars_l3_{before_c,after_l3}`, `farc_l3_{before_c,after_l3}` (jobs 25259743/25259773); `serref_m523L3_test` (Serial bit-id); `gate_dev`/`serref_core2` (gate). Jobs `jobs/job_dars_l3_ab`, `job_farc_l3_ab`, `job_core2_serial_m523L3`.

### 8.8 L5 + fieldN + L2 — the cheap comm grind finishes (and PLATEAUS)

Three levers off the §8.4 menu, in the prompt's order. Each measured in the comm regime, held to Serial-bit-id + the CUDA gate.

**L5 — dead-halo poison-test (1 free removal).** The L67 NaN-poison method on a HALO: a gated device kernel (`FESOM_POISON_<F>=1`) NaNs the field's halo tail `[myDim·stride, size)` right after the exchange; run the gate; PASS-while-poisoned ⇒ no downstream reader ⇒ dead.
- **`bvfreq` (step.cpp:216) = NEEDED** (the POSITIVE CONTROL): the device smoother `fesom_smooth_nod3D_kk` gathers bvfreq at element vertices incl. halo nodes (`eos.cpp:560`) → poison CRASHED the run (only `snap_000000`). This proves the poison mechanism propagates, so a clean PASS elsewhere is trustworthy.
- **`uv_rhsAB` (step.cpp:467) = DEAD**: `compute_vel_rhs_kk` reads `uv_rhsAB` only at OWNED elements (`E=myDim_elem2D`, momentum.cpp:462/515); momadv scatters INTO it but its flux reads `uvnode_rhs`, not `uv_rhsAB` → nothing reads its halo. Poison PASSED (worst h_ice 8.4e-3, NaN-free, all snapshots) → **removed the exchange = a whole ELEM3D nl=2 halo + 2 fences/step, bit-identical** (Serial all-fields). Job `jobs/job_gpu_poison_l5` (3 legs: clean / bvfreq / uv_rhsAB, in one allocation).

**fieldN — EOS 5-block 3→1 (−0.6 %).** Generalized field2 to N fields: `fesom_halo_exchange_deviceN(Field *const*, nf, …)` + `fesom_halo_fieldN({&f0,…})`. **Design (sidesteps the device-array-of-Views problem):** the N pack/unpack launches loop on the HOST, each capturing ONE `fi=fields[i]->d()` writing the DISJOINT buffer slot `g·stride + i·fs` → no fence between the N packs (same CUDA stream serializes + slots disjoint), ONE fence before MPI, ONE msg/neighbour, N unpacks, ONE fence. `nf==2` reproduces device2 exactly. Folded density+hpressure+bvfreq+sw_alpha+sw_beta (NOD3D nl 1; L3 had them 5→3) at `step.cpp:214` into ONE exchange.
- **Serial ALL-FIELDS-BIT-IDENTICAL** (vs `serref_m522_saved`, `diff_snap` rc=0 — fieldN on Serial = 5 sequential brackets, same order); **CUDA gate PASS** (worst h_ice 5.7e-3, then 6.3e-3 on the post-L2-revert re-gate — no NaN, every EOS-downstream field at floor → no co-pack/adjacency error).
- **A/B (same allocation):** **dars 8N 0.3393→0.3374 = −0.56 %**, **farc 2N 0.2822→0.2800 = −0.78 %**. DETERMINISTIC **−3 MPI calls/step** (−2 EOS fieldN + −1 uv_rhsAB; farc 754→751 exactly; dars 401→397 = −4 incl. ±1 CG-iter jitter), halo **−1…2 MB/step/rank** (the uv_rhsAB ELEM3D removal — fieldN co-pack keeps bytes constant). **Binary `fesom_port_m523fN`** (= live `fesom_port`). Jobs `job_dars_fN_ab`, `job_farc_fN_ab`, `job_core2_serial_m523fN`.

**L2 — persistent MPI requests = MEASURED DEAD END (reverted).** Implemented `MPI_Recv_init`/`Send_init` once per (kind,stride) + `Startall`/`Waitall` per call (one shared helper for device/device2/deviceN; rebuild-on-`grow()` via buffer-base-pointer staleness; env toggle `FESOM_HALO_NOPERSIST` for a same-binary A/B). The reach IS broad — the dominant call generators DO route the device path (CG per-iter `fesom_halo_exchange_device(NOD2D,1,1)` at `ssh.cpp:710`, ~65–220/step; EVP `field2(NOD2D,1,1)` at `evp.cpp:716`, 120/step).
- **Gate PASSED** — device-ptr persistent requests WORK in OpenMPI-4.1.5-nvhpc/UCX and move byte-identical data (worst 1.13e-2, within ceilings).
- **But the same-binary A/B (fresh vs persist) was flat-to-SLOWER:** farc-2N 0.3015→0.3023 (**+0.27 %**), dars-8N 0.3681→0.3683 (**+0.05 %**), identical CG-iter counts. WHY: UCX Irecv/Isend POSTING is already cheap; the real per-exchange cost is the 2 `Kokkos::fence()`s + the transfer + the load-imbalance Waitall — none of which persistent requests touch. And L2 doesn't change calls/step (still 1 Waitall/exchange) → no deterministic proof, only noisy timing. **Reverted** to the inline fresh path. **Worth retrying on LUMI/AMD** (Cray-MPICH — different request-setup cost). Jobs `job_dars_l2_ab`, `job_farc_l2_ab` (toggle A/B). (Echoes §M5.17/L62: a comm idea the regime's real wall makes a no-op.)

**Verdict — the cheap comm grind has PLATEAUED.** L1 (EVP, 120×/step) −9.1 % → L3 (10 structural pairs) −2.4 % → fieldN+L5 −0.6 % → L2 dead. The structural once-per-step halo surface is EXHAUSTED. The two remaining climate-safe comm levers: **L4** (CG 2→1 Allreduce, Chronopoulos/Eijkhout — CG is 14 % dars / 38 % farc of the step, the higher-potential one, but NOT bit-identical → reassociates the reduction → gate-class + 1-yr climate; §8.4 parked it) and **mixed precision (≈×2 — the ONLY lever that reaches the 2-SYPD target, halving BOTH compute and comm bytes).** Per the user's M5.23 deferral, the mixed-precision decision returns now. Lesson L71.

---

## Appendix — artifacts (all under `/work/ab0995/a270088/port2/kokkos_gpu_runs/`)

- 4N nsys: `nsys_ng5/stats.txt`, `nsys_ng5/run.log` (job 25252639)
- 4N per-phase: `ng5prof.25252640.out` (job_ng5_prof, TAG=m522_prof)
- 4N synclog: `synclog.25252641.out` + `ng5_synclog/err.0` (job 25252641)
- 4N halo-split: `ng5hsplit.25252642.out` (job 25252642)
- 16N nsys: `nsys_ng5_n16/stats.txt`, `nsys_ng5_n16/run.log` (job 25252684)
- comm proxy (§8): `farc_n{1,2,4}_{clean,base,barrier}/run.log` (jobs 25256636/637/638, barrier-isolation), `dars_n8_{clean,base,barrier}/run.log` (job 25256642), `farc_nsys_n2/stats.txt` (job 25256639, MPI message profile). Jobs: `jobs/job_farc_halo_split`, `job_dars_halo_split`, `job_farc_nsys`.
