# M11 — Partitioning & mesh-ordering track: research digest

Date: 2026-08-10. Worktree `~/port_kokkos_part`, branch `m11-partition` (from `f42c453`).
Sources: 3 web surveys (software landscape / reordering / peer-model practice) + 1 repo/partitioner
exploration, all claims verified at source (papers fetched, `fort_part.c` read, GitHub API dates
checked). This file is the track's reference; the plan lives in `docs/plans/`.

---

## 0. The problem, measured (M10 baseline)

- fArc CPU 2048r: per-rank ocean busy vs owned **3-D nodes r = 0.967** (vs 2-D nodes r = 0.003);
  busy spread 4.8×; the imbalance tax is **11 % of the CORE2 step, 19 % of the fArc step**; MPI wait
  is 51/53 % of the whole step. (`port_kokkos_ssh/docs/report/m10_load_balance.*`,
  `m10_where_time_goes.*`)
- The dual-constraint experiment (`ncon=2`, balance 2-D and 3-D counts): 3-D balance 9.60×→1.05×,
  halo nodes +40 %, net **+9.55 % slower** CPU 864r, **+29.7 %** GPU 8r (ocean busy itself +20 % —
  fragmentation multiplies *replicated element/edge work*, not just halo bytes).
  ⚠️ **CORRECTED 2026-08-10 (plan-review B1)**: the earlier "edgecut ×87–91" reading
  (1,335 → 120,883) mixes UNITS — `fort_part.c:191-205` sets `adjwgt` only for weighted runs, so
  the METIS print is an unweighted cut COUNT for wgt0 but an nlev-weighted cut SUM for dual arms.
  The true unweighted-cut ratio is unknown (plausibly ~1.5–2×: mean edge weight ~50–60, and
  120,883 raw edges would be ~32 % of CORE2's 371k — irreconcilable with +40 % halo). The
  scorecard (plan Task 2) re-measures both quantities; M10's report tables carry the same mix.
- Sign flips with rank count: dual weighting **wins −4.6 % at 256r** (495 verts/core), 0.0 % at
  512r, loses +8.7 % at 864r. Production CORE2 512 sits on the crossover.
- User-reported additional defects: up to ~140 disconnected stray vertices in a part; objective
  ignores edge/triangle operations; old METIS; mesh numbering has no spatial locality.

## 1. Ground truth: what our pipeline actually does (new findings ★)

Partitioner = `fesom_meshpart` from `/home/a/a270088/fesom_part/fesom2` (bundled **METIS 5.1.0**,
2013, int32). Graph = SSH stiffness sparsity (2-D nodal, quad diagonals count)
(`fvom_init.F90:1609-1674`). Weights: default `PART_WEIGHTED` ⇒ `ncon=2`, constraint2 =
`nlevels+100` (softened); coarsest-level edge weights = sum of endpoint depths (`fort_part.c:152-205`).

- ★ **We call `METIS_PartGraphRecursive`, not Kway** (`fort_part.c:233`; Kway lines commented out).
  Consequences: `METIS_OPTION_CONTIG` is **ignored by Recursive** (and is set 0 anyway), and the
  **`OBJTYPE_VOL` communication-volume objective is unreachable** (Kway-only). So contiguity and
  comm-volume optimization have never been ON in any partition we run. `UFACTOR=1`, `NCUTS=10`,
  seed 35243.
- ★ **Injection seam is one line**: `fvom_init.F90:1792-1793` (`do_partit`). Everything downstream
  consumes only `part[]` + npes. A `FESOM_PART_FILE` env patch (mirror of the proven
  `FESOM_PART_WGT` patch at `/work/ab0995/a270088/port2/partw/fesom2/src/fort_part.c:45-60`) lets
  ANY external tool produce partitions while the battle-tested Fortran code writes the dist files ⇒
  full Fortran + Kokkos compatibility for free. ~15 lines.
- Halo send/recv lists are **stored** in `com_info*.out` (local indices, rlist collapsed to
  identity), not derived at runtime; `my_list` halo-tail order must match. Full format spec in the
  explorer report (records, 1-based, ASCII). Kokkos port reads rpart records 1–2 + my_list +
  com_info only; rpart record 3 (PE-contiguous renumbering) is needed by Fortran FESOM + our
  analysis scripts (`scripts/m7_part_spread.py:18-29` = verified parser).
- `check_partitioning` post-pass (`fvom_init.F90:1809-1977`) already moves nodes with ≤1
  same-partition neighbour — keep it for any injected vector.
- 🔴 Traps: partitioner **mutates `nlvls.out`/`elvls.out`** and refuses to overwrite existing
  edge/level files while using freshly-computed in-memory values (`fvom_init.F90:663-683,996-1017`;
  FESOM/fesom2#852) ⇒ md5-gate mesh files (M10 jobs already do), delete edge files when renumbering.
- Hierarchical partitioning (nodes→cores, ≤10 levels) is fully implemented and namelist-driven; the
  *shipped* default is `n_part=2,128`. All partitions we ever generated are FLAT (`n_levels=1`).
  Possibly explains the M10 mystery that shipped CORE2 `dist_864` is **7.4 % faster** than our flat
  regeneration.
- Partitioning cost: minutes (NG5 7.4M nodes, 2048 parts = 22 min single node). Offline is fine.
- Existing dist inventory: CORE2 private copy has dist_{1..864}; /pool farc→2304, dars→4096,
  ng5→8192.

## 2. Modern partitioning software (verified versions, 2026-08)

| Tool | Ver (date) | License | Why we care |
|---|---|---|---|
| METIS 5.2.1 | 2022-12 (master active 2026) | Apache-2 | Kway: **`OBJTYPE_VOL`** (= exact one-directional halo volume, `vsize` per node), **`MINCONN`** (min max-neighbour count), **`CONTIG`** — all combinable in one call |
| KaMinPar | v3.7.3 (2026-03) | MIT | THE modern large-k partitioner (deep multilevel, k→2^20); strict balance; per-block max weights; `-P strong` flow refinement; METIS format in |
| Mt-KaHyPar | v1.6.2 (2026-07) | MIT | Best quality; **km1 objective = exact total comm volume** (star-expansion hypergraph, ~50-line converter); **Steiner-tree process mapping** onto machine topology; ESA'26 high-quality multi-constraint lives here |
| KaHIP | v3.25 (2026-03) | MIT | ★ **`--connected_blocks`** (v3.25): constructively CONNECTED blocks (strong preset, connected input) — directly kills the stray-vertex defect |
| Scotch 7.0.13 | 2026-08 | CeCILL-C | reference static mapper (arbitrary target graph); orderings; NO multi-constraint |
| Not worth | | | ParMETIS (non-free, 2013, documented balance failures), Jet (self-declared weak on 2-D meshes, no vwgt import), mt-METIS (dormant, infeasible outputs), PaToH (binary-only), Sphynx/Zoltan2 (quality below multilevel), streaming tools (no memory problem) |

Key literature verdicts:
- **Multi-constraint is the documented villain**: refinement must respect every constraint ⇒
  crippled local search. ESA'26 (Maas, arXiv:2605.28333): METIS multi-constraint leaves ≥11 %
  connectivity on the table even when feasible; competitors <90 % feasibility at d>2. Our
  dual-weight net loss is this phenomenon measured in step time (the cut ratio itself awaits
  re-measurement in honest units — see §0 correction).
- **Single scalar weight avoids the penalty entirely** (balance is one constraint; search space
  intact). Field practice: OpenFOAM measured `weightField`, Alya measured element weights with a
  re-measure loop, SeisSol LTS weights, MPAS-Seaice climatological `icePresence`. METIS handles
  `vwgt` (compute) and `vsize` (comm volume) **independently** — model both.
- METIS manual explicitly: the **nodal graph is the correct comm model for vertex-centred codes**
  (dual graph is not) — our graph is already the right one; the objective was not.
- `CONTIG` caveat: **silently ignored if the input graph is disconnected** — the global ocean graph
  IS disconnected (Caspian etc.) ⇒ must pre-connect components (virtual edges) or partition
  components separately, else CONTIG is a no-op again.
- Max-per-rank comm volume objective: only UMPa (dead code) / PuLP (low quality); practical route =
  minimize total volume + post-process outliers (MINCONN pass).

## 3. Mesh renumbering (sorting) — evidence and recipe

Measured precedents (closest analogues to us):
- **CPU**: Hilbert reorder of a production FV code (JCP 2023, 1.5B cells, 12,800 ranks): kernel
  −24 %, cache misses −54 %, **wall −18 %** (range −11…−30 % over 10 cases). SEM code: 9–25 %.
- **GPU**: RCM + derived face order on V100 FV (FGCS 2023): kernels 1.4–1.5×, **whole app
  1.05–1.63×**, L2 transactions −50 %. OP2 studies: thread-block-local clustering raises in-block
  reuse 2→3.6. Atomics BEAT coloring on V100/A100 at all mesh sizes (coloring destroys locality);
  warp-level pre-aggregation only pays where fp64-atomic HW is absent (MI100) — A100: plain atomics.
- **Sphere recipe**: cubed-sphere Hilbert key (Google S2 cell id) or 3-D Hilbert on unit-sphere xyz
  (~100 lines, Skilling). NEVER a 2-D curve on lon/lat (dateline/pole seams). Elements: sort by
  centroid key. Edges: first-touch from element sweep (regeneration by the partitioner preserves the
  `edge2D_in` interior-first convention automatically). Standard workflow (OP2): renumber offline
  once, then partition; Firedrake ships RCM-at-load as DEFAULT.
- **Our specifics temper expectations**: fields are level-contiguous (`FESOM_NODE3D = node*nl+lev`)
  and the L63/L64/L66 campaign already fixed field-access coalescing (#1 kernel at 7.3
  sectors/request, 59 % of A100 DRAM peak). Renumbering headroom on GPU = the **index streams**
  (elem→node gathers, edge gathers, atomic scatters, halo slist/rlist) + TLB/L2 on the node-strided
  accesses; on CPU = classic cache locality (bigger expected win). GPU is demonstrably sensitive to
  partition/order quality: a merely *regenerated* CORE2 partition cost +4.18 % on GPU.
- **Renumbering is transparent to model code** (explorer verified: no restart reader; forcing/IC
  interpolated at runtime by lat/lon; `nod2d.out` id column must be rewritten to identity;
  hard-check `fesom_mesh.cpp:250-251`). Must permute: nod2d/elem2d/aux3d/nlvls/elvls (+depth@
  files if present); must DELETE so they regenerate: edges/edge_tri/edgenum; delete pyfesom2
  caches/griddes; regenerate all dist_*.
- 🔴 Protocol pitfalls: (1) **permute-labels-first** — first A/B must keep the OLD partition
  content (carry part[] through the permutation) so ordering is isolated from repartitioning;
  (2) FP reduction orders change ⇒ archived reference NetCDFs incomparable; the C↔K bit-identity
  oracle survives by running BOTH on the renumbered mesh; (3) SSH iteration counts must be
  IDENTICAL under a pure symmetric permutation — free correctness gate; (4) Z7-class risk: one
  missed node-indexed file = wrong-later-not-at-step-1.
- Literature gap we would settle: SFC vs RCM for a vertex-centred FV code at 30k–120k verts/GPU —
  nobody has published it.

## 4. Peer-model precedents

- **POP/Dennis 2007** (the classic): balance ONE scalar "wet work" via weighted space-filling
  curve; imbalance 0.48→0.05; −21 % step at 29k cores; 4.0→7.9 SYPD. Tolerated DISCONNECTED
  ownership, cost bounded 15–20 % of max comm. CICE productionized weights (`latitude`, `wghtfile`);
  Craig 2015: up to 45 % in CESM. MPAS-Seaice: partitions driven by 50-yr climatological ice
  presence.
- **NEMO**: north-fold ranks deliberately get SMALLER subdomains to offset known extra comm — the
  `tpwgts` idea (asymmetric target sizes) nobody has tried with METIS in ocean codes. PyFR does
  measured per-device tpwgts (3:27:23 CPU:GPU:GPU).
- **ICON GPU**: 1 rank/GPU; below ~100k columns/GPU the limiter is GPU *utilization*, not halo
  bytes; strong scaling poor. Matches our regime — on GPU, partition QUALITY (fragmentation ⇒
  replicated work) and per-rank utilization outrank imbalance.
- **Nobody anywhere** has published: a net-time-positive bathymetry-weighted partition (our M10
  data is already ahead of the literature — publishable); GPU-count-specific partitioning;
  contiguity enforcement on ocean graphs; wait-aware objectives. MPAS-O/SCHISM/FVCOM/ADCIRC/WW3
  are all plain unweighted (Par)METIS.

## 5. Candidate levers (input to brainstorm/plan)

- **A. Modernize the METIS call** (small): METIS 5.2.1 **Kway** + `OBJTYPE_VOL` (`vsize=nlevels`)
  + `MINCONN` + `CONTIG` (with component pre-connect) + **single scalar weight `w = a + b·nlev`**
  (profile-calibrated; sweep a/b and UFACTOR — the interior between wgt0 and wgt2 was never
  explored) + optional `tpwgts` arm. Attacks imbalance, disconnection, and comm volume at once,
  zero new dependencies.
  - Calibration caveat: the M10 fit (2.241 ms/1000 3-D nodes, intercept 10.22 ms/rank) has 2-D
    cost hidden in the per-rank intercept (2-D counts were ~constant); a/b needs its own fit (vary
    both dims or fit per-phase), per backend.
- **B. Engine swap via part-vector injection** (medium): `FESOM_PART_FILE` patch at the one-line
  seam; then KaMinPar (CPU large-k), Mt-KaHyPar km1/highest-quality (GPU small-k, true comm-volume
  objective; element replication modeled), KaHIP `--connected_blocks`. Dist files still written by
  the Fortran tool ⇒ format + Fortran compat preserved.
- **C. Renumbering converter** (medium): Hilbert/S2 vertex sort + derived elem/edge orders +
  permute-labels-first A/B (CPU + GPU, big meshes most promising); optional per-rank local RCM arm;
  halo-list contiguity as a bonus (pack kernels become near-sequential).
- **D. Hierarchy & placement** (small-medium): hierarchical partitioning is ALREADY in the tool and
  unexplored by us (shipped-864 mystery suggests it matters); Levante topology mapping (128-core
  nodes, 4×A100 NVLink) via hierarchy or Mt-KaHyPar Steiner mapping.
- **E. Harness first** (prereq): partition scorecard script — 2-D/3-D/weighted imbalance, edgecut,
  total+max comm volume, max/mean neighbour count, **connected components per part**, halo fraction,
  **element/edge replication factors** (Σ myDim_elem2D / elem2D — the fragmentation currency
  behind the measured +20 % GPU busy) — plus stability gate and phasestats replay. Success currency: net step time,
  matched pairs, min-of-2, same day, both backends.

## 6. Constraints & gates (standing)

- dist-file format unchanged at runtime (injection path guarantees it); mesh copies under
  `/work/ab0995/a270088/port2/mesh/`, never edit /pool or the private CORE2 in place; GPU jobs ≤16
  nodes; same-day pinned baselines; `BIN=` pinned.
- Stability gate MANDATORY for every new partition: regenerated partitions have diverged before
  (wgt0 dist_128 CG blowup at iter 1; NG5 shipped dist_4096 with 13.89× 3-D imbalance hit the
  blow-up guard at step 175 while a 1.05× partition ran clean).
- Renumbered meshes: full validation ladder ON the new mesh (C↔K twin is order-consistent there);
  never compare raw NetCDF across orderings without the permutation.
- METIS weights are int32 — keep scalar weights small (a+b·nlev ≤ ~200).

## 7. Key sources

METIS manual (VOL/CONTIG/MINCONN semantics); Maas ESA'26 arXiv:2605.28333 (multi-constraint
penalty); KaMinPar github.com/KaHIP/KaMinPar; Mt-KaHyPar github.com/kahypar/mt-kahypar; KaHIP
v3.25 release notes (`--connected_blocks`); CSUR 2023 survey arXiv:2205.13202; JCP 2023
112009 (Hilbert FV −18 %); FGCS 2023 (RCM GPU 1.05–1.63×); OP2 JPDC 2019 + MPI dev guide
(renumber-offline workflow, layering); FUN3D SC21 (atomics/warp-agg); Dennis IPDPS 2007 (POP SFC);
Craig 2015 IJHPCA (CICE 45 %); NEMO GMD 2022 (north-fold shrink); ICON GMD 2022/2026 (GPU
utilization-bound); Koldunov 2019 GMD (hierarchical partitioning); FESOM/fesom2#852 (partitioner
mutates level files). Full agent reports in session transcript 2026-08-10.
