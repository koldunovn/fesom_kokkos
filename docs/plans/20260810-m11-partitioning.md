# M11: Partitioning & mesh-ordering campaign

*Rev 2 — hardened per plan-review 2026-08-10 (3 blockers, 10 majors, 9 minors applied; see
session log for the review record).*

## Overview

Attack the four measured/reported defects of FESOM2's domain decomposition — bathymetry-driven
load imbalance (11–19 % of the CPU step at 864–2048 ranks), disconnected partitions (~140 stray
vertices), an objective blind to the code's edge/element operations, and spatially arbitrary node
numbering — by (a) modernizing the METIS call, (b) injecting external state-of-the-art
partitioners, and (c) renumbering meshes with sphere-aware space-filling curves + RCM.
Architecture: **score offline, then race** — a scorecard prunes a large offline candidate zoo to
a shortlist; only the shortlist buys cluster time. Deliverables: certified per-backend default
partitions/orderings + a publishable LaTeX report. The model binary is **untouched** throughout —
every lever is an input-data lever, raced with the frozen certified `h17` binaries.

Full research digest (read first): `docs/PARTITIONING_M11_RESEARCH.md`.

## Context (from discovery)

- Partitioner ground truth: `fesom_part/fesom2/src/fort_part.c` calls `METIS_PartGraphRecursive`
  (METIS 5.1.0, dual-constraint `+100`, `UFACTOR=1`, seed 35243) ⇒ `CONTIG` and `OBJTYPE_VOL`
  have never been active. Injection seam = `fvom_init.F90:1792` (one call; everything downstream
  consumes only `part[]`). Halo lists are STORED in `com_info*` — dist format fully specified in
  the research digest.
- M10 evidence: dual-constraint fixed 3-D balance 9.60×→1.05× but made the model net slower
  (+9.55 % CPU 864r / +29.7 % GPU 8r, with GPU ocean *busy* +20 %). ⚠️ **Units caveat (review
  B1)**: the METIS "edgecut" print is an unweighted count for wgt0 but an nlev-weighted cut SUM
  for weighted arms (`fort_part.c:191-205` sets `adjwgt` only when `wgt_type != 0`), so M10's
  "1,335 → 120,883 (×90)" mixes units; the true unweighted-cut ratio is unknown (plausibly
  ~1.5–2×). The scorecard re-measures fragmentation in honest units (Task 2); element/edge
  replication factors are the fragmentation currency.
- Crossover law: dual weighting WINS −4.62 % at CORE2 256r (495 verts/core), 0.00 % at 512r,
  loses +8.71 % at 864r — the race grid must include a rung on the winning side.
- Verified rpart parser: `scripts/m7_part_spread.py:18-29`. Partition maps: M9 tooling
  (`port_kokkos_ice/scripts/m9_partition_artefact.py`, `m9_artefact_maps.py`).
- This tree's partitioner KEEPS pre-existing `nlvls.out`/`elvls.out`/edge files (write-guards
  `fvom_init.F90:663-683,996-1017`) while partitioning with freshly **recomputed in-memory**
  levels — so balance weights can silently diverge from the on-disk files the model and scorecard
  read (review M5). When the files are ABSENT it writes them into MeshPath (the fesom2#852
  hazard class). Both directions are guarded in Task 1/4.
- Frozen binaries: `h17` (Serial `5c3c90fc` / CUDA `f8384e86`) — pinned via `BIN=` in every job.
- Source meshes (read-only): CORE2 private copy `/work/ab0995/a270088/port2/mesh/core2`;
  `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/{farc,dars,ng5}`.

## Development Approach

- **Testing approach: verification-gated (project-native).** No unit-test framework; instead every
  task ends with executable verification against ground truth (published M10 numbers in corrected
  units, byte-identity null tests, permutation invariants) and no task's outputs are used
  downstream until its verification items pass.
- Complete each task fully before the next; small focused changes; update this plan file when
  scope changes (`[x]` immediately, ➕ new tasks, ⚠️ blockers).
- 🔴 **HARD RULE — production meshes are read-only.** `/pool` meshes AND the existing private
  copies other tracks gate on are never written (adding a `dist_N` counts as writing). All work in
  `/work/ab0995/a270088/port2/mesh_m11/` copies. Guards (Task 1) close the three real write
  paths: symlink write-through (`cp -aL` + no-symlink assert), MeshPath escapes (assert on the
  namelist value re-read after sed, `readlink -f`-resolved), and source-side writes (source md5 +
  `-newermt` sweep after every partitioner invocation). The ONLY sanctioned write outside
  `mesh_m11/` is `m11_promote` (Task 16), which creates new directories only.
- Standing rules in force: BIN= pinned, matched same-day min-of-2 pairs, GPU ≤16 nodes,
  `-C a100_80` on GPU absolutes, outputs to /work never $HOME, cheap gates get cheap walltimes
  (`-t 00:06:00`), L80 every knob announces itself + abort on unrecognized values, rule 0.41
  (stability re-proven at protocol length before adoption).

## Testing Strategy

- **Tool regression targets** (corrected units, review B1/19):
  - `edgecut_unweighted` (wgt0-class only): CORE2 8/16/32r → 1,335 / 2,549 / 4,307.
  - `cutweight_nlev` (dual arms): CORE2 8/16/32r → 120,883 / 217,791 / 375,211.
  - 3-D balance: shipped CORE2 `dist_864` → 9.60×; `core2_wgt2` → 1.05×; fArc `/pool dist_2048`
    → 1.01× (2-D) / 9.40× (3-D).
  - halo nodes/rank: `core2_wgt0` → 42; `core2_wgt2` → 59 (targets belong to the wgt arms, not
    the shipped dist).
- **Byte-identity null tests** (on the 5.1.0 build only, review B2): patched partitioner with
  default knobs ≡ unpatched output; `FESOM_PART_FILE` fed a partition's own vector ≡ that
  partition's dist files (byte-identical, given zero `check_partitioning` moves — see Task 4).
- **Permutation invariants**: the scorecard's permutation-invariant block is identical between
  (mesh, dist) and (renumbered mesh, label-permuted dist); ordering-sensitive block exempt.
- **Model-level gates per raced arm**: 300-step stability SCREEN (no blow-up-guard hits);
  accuracy vs pre-registered partition-class floors measured from control pairs (Task 10);
  protocol-length stability RE-PROOF before adoption (Task 14, rule 0.41); orderings
  additionally: C↔K Serial twin bitwise ON the renumbered mesh + SSH iterations |Δ| ≤ 1 per
  solve, mean shift < 0.5 (systematic shift = bug); CUDA-vs-Serial fidelity check once per
  would-be-adopted arm.
- **Halo/dist correctness gate** (named, review M4): (i) Python side, in the scorecard:
  owned-sets exact disjoint cover + `com_info` reciprocity (every rlist entry has the matching
  slist entry on the partner rank); (ii) model side: the gid-identity halo test
  (`fesom_halo.cpp:212-280` — exchanges global ids, asserts every halo slot) — Task 1 verifies
  this test exists, runs on Serial, and demonstrably FIRES (force one corrupt com_info on a
  scratch copy); if absent/vacuous, a standalone checker is built instead.

## Progress Tracking

- mark `[x]` immediately; ➕ for discovered tasks; ⚠️ for blockers; session log
  `docs/PARTITIONING_M11.md` records every run id, BIN, env, node-hours, and verdict.

## What Goes Where

- **Implementation Steps**: everything achievable in this repo + /work sandbox.
- **Post-Completion**: upstream FESOM PR, paper submission, production repointing in other
  worktrees, M10-doc units-correction note.

## Implementation Steps

### Task 1: Mesh sandbox + guard library + session log

**Files:**
- Create: `scripts/m11_guards.sh`
- Create: `docs/PARTITIONING_M11.md`
- ➕ Create: `scripts/m11_corrupt_com_info.py` (halo-gate negative control)
- ➕ Create: `jobs/m11_gate_halo.sh` (two-leg gate job)
- Create (off-repo): `/work/ab0995/a270088/port2/mesh_m11/{core2_m11,farc_m11}/` + `MD5MANIFEST`

- [x] `m11_guards.sh`: `m11_assert_sandbox <path>` (readlink -f, abort unless under `mesh_m11/`;
      callers pass the MeshPath **re-read from the namelist after sed**, M10 pattern),
      `m11_md5_check`/`m11_md5_write <meshdir>` (five mesh files), `m11_check_sources`
      (md5 over the five mesh files of BOTH source meshes + `find <src> -newermt <job-start>`
      must be empty — run after every partitioner invocation)
      → ⚠️ `readlink -m` (not `-f`): the target of a sandbox assert usually does not exist yet.
      → ⚠️ `/pool/data/...` is a symlink into `/work/pd1284/...`, so the forbidden-root list must
      name the resolved project root too (a `/pool`-prefix test alone never matches).
- [x] copy sources with `cp -aL` (dereference symlinks); assert `find mesh_m11 -type l` empty;
      write MD5MANIFESTs; record source md5s for `m11_check_sources`
- [x] start `docs/PARTITIONING_M11.md` (decisions, run table, env block, node-hour ledger,
      plan-review record)
- [x] verify: guard aborts on a /pool path, on the private CORE2 path, and on a symlinked path;
      passes on sandbox → `bash scripts/m11_guards.sh selftest` 22/22 PASS
- [x] verify: halo/dist correctness gate exists and FIRES (corrupt one com_info entry on a
      scratch dist copy → model-side gid-identity test must abort; scorecard reciprocity check
      must flag it); source-dir `-newermt` sweep clean after the copies
      → job **26850057**: control leg rc=0 with the positive announcement; corrupted leg rc=1
      with exactly the 2 predicted gid mismatches. Scorecard-side reciprocity check lands in
      Task 2 and is re-verified against this same `gate_negctl` dist there.

### Task 2: Scorecard `m11_scorecard.py`

**Files:**
- Create: `scripts/m11_scorecard.py`

- [x] input = mesh dir + (`dist_N` | raw part-vector file); reuse the verified rpart parser by
      import from `scripts/m7_part_spread.py` (never re-implement)
- [x] **permutation-invariant block**: 2-D/3-D/weighted imbalance (max/mean, `w=a+nlev` for given
      a, plus Σw per arm for the int32 ledger), `edgecut_unweighted`, `cutweight_nlev`, total +
      max-per-rank comm volume (`vsize=nlev`), neighbour count max/mean, connected components per
      part + stray-vertex count, halo fraction, element + edge replication factors, wet-graph
      component count; owned-cover + com_info reciprocity gates
      → ➕ `isolated_nodes` (≤1 same-part neighbour = check_partitioning's own criterion) and
      the stray count split into `noncore_vertices` / `singleton_vertices`: a part in two large
      lobes and a part trailing loose vertices are different defects.
- [x] **ordering-sensitive block** (exempt from invariance gates, review M9): mean |Δindex| over
      edges, gather-stream stride histogram for elem→node and edge→node streams
- [x] output: one CSV row per (mesh, N, arm) + human summary; `--compare` mode
- [x] verify (regression): reproduce ALL Testing-Strategy targets in their CORRECT units;
      **recompute the dual arms' unweighted cut** and record the true wgt0→wgt2 cut ratio —
      then correct `PARTITIONING_M11_RESEARCH.md` §0 and the memory file with the measured
      number before it is quoted anywhere else (review B1)
      → **12/12 PASS**; true ratio ×1.44–1.61 unweighted / ×1.22–1.30 weighted; digest §0 and
      both memory files updated.
      → ⚠️ one target needed a mechanism, not a tolerance: the 16r weighted cut is 217,796 on
      disk vs the 217,791 METIS printed, because `check_partitioning` runs AFTER the print.
      ➕ new Task-4 verification item below.
- [x] verify (invariance): invariant block identical under a synthetic label permutation of a
      small dist; ordering block changes as expected
      → 26/26 invariant keys identical under a random relabelling; all 11 ordering keys moved.
- [x] ➕ verify (gate, cross-check with Task 1): the scorecard's reciprocity check FAILS on the
      `gate_negctl` dist that made the model abort, naming the right block
      (`rank 0<-1: 1 of 142 gids differ, first at slot 0`) and exiting 1.

### Task 3: Graph exporter + part-vector importer

**Files:**
- Create: `scripts/m11_graph_export.py`
- Create: `scripts/m11_part_import.py`

- [x] exporter: mesh → METIS .graph (adjacency = share-an-element, matching `stiff_mat_ini`
      semantics), variants: unweighted / `vwgt=a+nlev` / `vsize=nlev` / both; plus star-expansion
      hypergraph (hMETIS format) for Mt-KaHyPar km1 (net per vertex = {v}∪N(v), net weight nlev)
      → also `--weights dual` (ncon=2, `(1, nlev+100)`) so the legacy arm can be reproduced in an
      external engine, and `--edge-weights` (`adjwgt = nlev_i+nlev_j`) mirroring fort_part.c.
      Graph comes from `m11_scorecard.Mesh.graph()` by import — one implementation, not two.
- [x] importer: engine output (one part id per line, 0/1-based autodetect) → `FESOM_PART_FILE`
      format + sanity (k parts nonempty, exact cover); `--from-dist` mode extracting the part
      vector from an existing `dist_N` (the extractor Tasks 4/5/9 use)
      → `--from-dist` additionally cross-checks the derived assignment against rpart's own
      per-rank count block (an independent witness inside the same file).
- [x] verify: exported CORE2 graph symmetric; importer round-trip (dist → vector → import →
      identical vector); edgecut of shipped vector on exported graph == scorecard edgecut
      (consistency only — the authoritative CSR diff happens in Task 4 via the dump knob)
      → CORE2 and fArc: symmetry exact (743,288 / 3,783,940 directed entries), edge multiset
      identical to `Mesh.graph()`, cut from file == cut from graph (1,361 / 124,915), round-trip
      identical both ways, scorecard on the part FILE == scorecard on the dist.
- [x] verify: star-expansion pin counts and net weights spot-checked against node patches
      → net weight = nlev on every net, net size = degree+1 on every net (870,146 pins = n+2m),
      three random patches equal {v}∪N(v), and ➕ **km1 of the written file = 47,620 = METIS
      totalv(vsize=nlev) exactly** — the premise of the Mt-KaHyPar arm verified numerically,
      not just asserted.

### Task 4: Patched partitioner `partm11` — two builds (injection + knobs + METIS 5.2.1 + pre-connect)

**Files:**
- Create (off-repo): `/work/ab0995/a270088/port2/partm11/fesom2/` (copy of `~/fesom_part/fesom2`)
- Modify (off-repo): `partm11/fesom2/src/fort_part.c`, `src/fvom_init.F90`,
  `mesh_part/CMakeLists.txt`, `lib/` (metis-5.2.1 + GKlib alongside bundled 5.1.0)
- Create: `scripts/m11_partgen.sh` (job template sourcing `m11_guards.sh`)

- [x] copy tree; port `FESOM_PART_WGT`; add `FESOM_PART_FILE` (read 0-based vector, SKIP the
      `part[i]--` renumbering decrement, keep `check_partitioning`; assert min=0, max=npes-1,
      every part nonempty, print per-rank histogram — review m14); add `FESOM_PART_GRAPH_DUMP`
      (write the exact `ssh_stiff` CSR handed to METIS — review m16)
      → the dump also carries the **in-memory `nlevels_nod2D`**, which settles review M5 at
      this seam without patching Fortran at all.
- [x] add `FESOM_PART_*` knobs: KWAY, OBJ(cut/vol), MINCONN, CONTIG, UFACTOR, WGT_A (scalar
      `w=a+nlev`; `-1`=legacy dual), TPWGTS_FILE — every knob announces itself, aborts on
      unrecognized values (L80)
      → plus VSIZE, NCUTS, NITER and **SEED** (Task 14 needs seed variants and the seed was
      compile-time). ➕ **OBJ=vol / MINCONN / CONTIG now ABORT unless KWAY=1** — METIS ignores
      them under Recursive without a word, which is exactly how they were silently off for
      every FESOM partition ever generated. ➕ CONTIG counts components first and refuses on a
      disconnected graph rather than pre-connecting behind the user's back.
- [x] **build `partm11-a`** = patches on bundled METIS **5.1.0** → verify the two nulls here
      (review B2): (null-1) default-knob `dist_8` byte-identical to the unpatched binary, same
      seed; (null-2) injection: `--from-dist` vector re-fed → `check_partitioning` moves == 0 AND
      output byte-identical (if moves > 0: diagnose, log as deviation, do not ship — review m15)
      → job **26851187**: null-1 PASS (17 files identical, edgecut 120883 = M10's value),
      null-2 PASS (0 moves, byte-identical).
- [x] **build `partm11-b`** = same patches on **5.2.1** → its default output is NOT expected to
      match; define arm **A0** (5.2.1, Recursive, dual+100, UFACTOR=1, seed 35243) whose delta
      vs 5.1.0 is scored and raced as an explicit lever; record both binaries' md5 in the log
      → job **26851421** leg A0 announces "Metis version 5.2.1"; arm generated at 8/16/32 by
      job 26851516 and scored. ⚠️ three silent traps in the 5.2.1 build, all documented in
      `docs/partm11/README.md` (`i64=0` selects **64-bit**; `cflags=` is dropped by its
      Makefile and its `-fPIC` is GCC-branch-only; GKlib installs to `lib64`, METIS reads `lib`).
- [x] verify: exporter graph (Task 3) diffs EXACTLY against `FESOM_PART_GRAPH_DUMP` CSR on
      CORE2 + fArc; dump partm11's in-memory `nlevels_nod2D`/`nlevels` and diff vs on-disk
      `nlvls.out`/`elvls.out` per mesh copy — log agreement or state the discrepancy in every
      weight-arm row (review M5)
      → CORE2: rowptr identical at all 126,859 entries, colind identical at all 743,288, and
      **in-memory `nlevels_nod2D` == on-disk `nlvls.out` at all 126,858 nodes** — no M5
      divergence on this mesh. fArc via job 26851516.
      → note: Task 2 already showed the on-disk `nlvls.out` and `elvls.out` of
      `core2_wgt0`/`core2_wgt2` are mutually consistent (0 nodes differ from
      max-over-incident-elements), so any in-memory divergence would be a *third* value.
- [x] ➕ verify (from Task 2): reproduce the print-vs-file cut gap. Run wgt2 at CORE2 16r with
      UNFILTERED stdout and confirm `check_partitioning` reports moving gid 125423 from part 9
      to part 8; the printed edgecut must then be 5 BELOW the scorecard's reading of the file.
      Record how often the post-pass moves anything at all across the Task-7 zoo (it moved
      nothing in 5 of the 6 CORE2 arms M10 generated).
      → job **26851421** leg E, exact: `Isolated node 125423 in partition 9` /
      `Neighbouring nodes are in partitions 9 8 8`, with `METIS edgecut 217791` printed against
      the file's 217,796. Predicted from the dist files alone, confirmed in the tool's own
      stdout. Move counts are now logged by `m11_partgen.sh` on every arm.
- [x] verify (knobs): each announces; bogus value aborts; Kway+VOL+CONTIG runs on CORE2 with
      component pre-connect engaging iff the wet graph is disconnected (component count from
      Task 2 decides; never silent)
      → job **26851421**: **26/26 PASS** — 13 accept-and-announce legs, 12 refuse legs, plus a
      short part vector refused rather than padded. CONTIG announces "1 connected component"
      on CORE2 and proceeds, so no pre-connect is engaged (and none is needed there).

### Task 5: Renumbering converter `m11_renumber.py`

**Files:**
- Create: `scripts/m11_renumber.py`

- [ ] orderings: `hilbert-xyz` (Skilling, 21 bits/dim on unit-sphere xyz), `s2` (cubed-sphere
      gnomonic + per-face 2-D Hilbert), `rcm` (scipy reverse_cuthill_mckee on the Task-3 graph —
      review M6); stable sort → P_node (saved old→new + inverse); P_elem = centroid-key (or
      min-new-vertex) sort
- [ ] permutation spec (review m17): P_node → nod2d rows (id column rewritten to identity),
      aux3d DEPTH block (header nl+zbar untouched), nlvls; P_elem → elem2d rows + elvls, with
      elem2d VALUES mapped through P_node (vertex cycles preserved, never rotated); DELETE
      edges/edge_tri/edgenum (regenerate); delete pyfesom2 caches + griddes; MD5MANIFEST +
      provenance README into a NEW `mesh_m11/` copy (guarded)
- [ ] exhaustive-classification pass: every file in the source mesh dir is classified
      (node-indexed / elem-indexed / neither / delete); UNKNOWN file ⇒ abort (Z7-class risk)
- [ ] `--permute-labels` mode: carry an existing dist's part vector (via
      `m11_part_import.py --from-dist`) through P_node → `FESOM_PART_FILE` for the renumbered mesh
- [ ] verify: P∘P⁻¹ = id; invariants (depth/nlvls/coast follow their node; element areas equal
      as multisets); scorecard invariant block: (new mesh, label-permuted vector) == (old mesh,
      old dist) exactly
- [ ] verify (smoke, sequenced): renumbered CORE2 → `partm11-a` regenerates edge files + a
      label-permuted `dist_8` → Serial `h17` short run + halo/dist correctness gate green

### Task 6: Engine builds + wrappers

**Files:**
- Create (off-repo): `/work/ab0995/a270088/port2/partm11/engines/{kaminpar,mtkahypar,kahip}/`
- Create: `scripts/m11_engines.sh` (wrappers: `m11_graph_export.py` → engine → `m11_part_import.py`)

- [ ] build from source with Levante gcc/cmake modules, pinned: KaMinPar v3.7.3, Mt-KaHyPar
      v1.6.2 (TBB+Boost), KaHIP v3.25; exact commits + module list in the session log
- [ ] wrappers: KaMinPar (`-P default|strong`, `-e {0.01,0.03}`), Mt-KaHyPar (`quality`,
      `-o km1`, star-expansion), KaHIP (`--preconfiguration=strong --connected_blocks`,
      pre-connected graph)
- [ ] any engine that fights the toolchain is DROPPED with a session-log note, not fought
- [ ] verify: each engine on CORE2 512 → importer sanity + scorecard row (balance within its ε)
- [ ] verify: one engine vector → `FESOM_PART_FILE` → dist → Serial smoke + halo/dist
      correctness gate green

### Task 7: A-family zoo (METIS) — generate & score

**Files:**
- Create: `scripts/m11_zoo_a.sh` (drives `m11_partgen.sh`; job arrays, skip-if-exists)
- Create (off-repo): per-arm REAL mesh dirs `mesh_m11/zoo/<mesh>/<arm>/` (five mesh files +
  edge files copied, dists generated inside; no symlinks — review B3; budget: CORE2 ~80 MB/arm,
  fArc ~400 MB/arm, ledger in session log)

- [ ] baselines scored: shipped dists (scoring reads production paths read-only), regenerated
      flat dual+100 on **5.1.0** (2 seeds), and **A0** (5.2.1 default null, review B2)
- [ ] grid (review M7/M11): CORE2 {4, 8, 256, 512, 864} + fArc {16, 64, 2048}; arms: A1
      Kway-cut-unweighted; A2 Kway-VOL; A3 VOL + `w=a+nlev`, a ∈ {0,15,40,100,∞}; A4 best-A3 +
      MINCONN + CONTIG; A5 UFACTOR {1,10,30,100} on best; A7 hierarchical — CPU (nodes×128 at
      512/2048) AND GPU (nodes×4 at CORE2 8, fArc 16/64) — incl. shipped-864 mystery probe
      (hierarchical vs flat vs seed)
- [ ] ➕ **A8 rank-relabelling arm** (from Task 2 Finding 4): the shipped `dist_864` and our
      `wgt0` regeneration are indistinguishable on every invariant metric (cut 34,159 vs 34,157,
      halo 42.1 vs 42.1, elem repl 1.473 vs 1.474) yet differ 7.4 % in step time ⇒ the difference
      cannot be partition quality. Relabel the ranks of ONE partition (pure permutation of the
      part vector — identical geometry, identical cut, identical scorecard row) and race it:
      a null result kills the placement hypothesis, a positive one is a free lever.
- [ ] scorecard CSV every arm; `m11_check_sources` after every partitioner invocation
- [ ] Pareto prune on the invariant block + 1–2 diversity picks; **known-bad anchors exempt
      from pruning** (wgt2-style dual at 864 + one deliberately high-cut arm) so the
      predicted-vs-measured regression spans the range (review praise item)
- [ ] session log: zoo table + node-hour ledger; A6 tpwgts explicitly deferred to round 2

### Task 8: B-family zoo (engines) — generate & score

**Files:**
- Create: `scripts/m11_zoo_b.sh` (drives `m11_engines.sh` + `m11_partgen.sh` injection)

- [ ] run the three engines at the Task-7 grid with the A3-best weight variant
- [ ] KaHIP arm: verify parts are actually connected (components == 1 per part)
- [ ] Mt-KaHyPar arm: record km1 (claimed comm volume) vs scorecard-measured comm volume
- [ ] score; merge into the joint Pareto pruning with A-family
- [ ] verify: one injected dist per engine per mesh passes Serial smoke + halo/dist gate

### Task 9: C-family zoo (orderings)

**Files:**
- Create: `scripts/m11_zoo_c.sh` (drives `m11_renumber.py` + `m11_partgen.sh`)
- Create (off-repo): renumbered meshes `mesh_m11/{core2,farc}_m11_{hil,s2,rcm}`

- [ ] generate 6 renumbered mesh copies (2 meshes × 3 orderings); edge files regenerated via
      `partm11-a`; md5-freeze after
- [ ] label-permuted baseline dists (identical partition content) at all race counts — the pure
      ordering A/B inputs
- [ ] combined-arm dists: the SCORECARD-BEST partition arm from Tasks 7/8 (available now,
      review M8) re-generated on each renumbered mesh
- [ ] scorecard: invariant block equality gate on every label-permuted dist (exact); ordering
      block recorded for all
- [ ] pick per mesh: one SFC + RCM to race (by the ordering-block locality proxies); the third
      ordering is a documented fallback

### Task 10: Shortlist + pre-registration + control floors

**Files:**
- Modify: `docs/PARTITIONING_M11.md`

- [ ] assemble WAVE-1 shortlist: ≤4 partition arms per mesh per backend (review M12) + the
      known-bad anchors + 2 ordering arms per mesh; expansion to wave-2 only if wave-1 produces
      a winner or the regression demands range
- [ ] PRE-REGISTER before any race: arms, gates, adoption rule (≥~2 % net, reproduced across two
      same-day pair days, per backend), SSH-iteration bound (|Δ| ≤ 1/solve, mean < 0.5),
      phasestats-second-pass policy, node-hour budget per task
- [ ] measure the **partition-class accuracy floor** from control pairs (shipped-vs-regenerated
      same-count Serial, 300 steps, diff_snap) per mesh; floors recorded BEFORE arm results exist
- [ ] SSH iteration behaviour: gate for pure-ordering arms; diagnostic for partition arms
- [ ] verify: every shortlisted dist has md5-frozen mesh + exact-cover + reciprocity green +
      scorecard row in the log

### Task 11: CPU races (Serial `h17`, lean matrix)

**Files:**
- Create: `jobs/m11_race_cpu.sh`

- [ ] matched 300-step pinned pairs, min-of-2, all arms of a point inside ONE allocation:
      CORE2 **256**/512/864, fArc 2048 (16 CPU nodes); `BIN=h17` Serial `5c3c90fc`; same-day
      baseline re-run each session
- [ ] stability screen on every arm before its timing counts; known-bad anchors raced too
      (regression range)
- [ ] accuracy gate vs Task-10 floors on every arm that beats baseline
- [ ] phasestats pass (busy/wait per phase) on interesting arms only; PMPI caveat noted per run
- [ ] verdicts + node-hours into the run table; losers get one line + scorecard row, no re-runs

### Task 12: GPU races (CUDA `h17`, lean matrix)

**Files:**
- Create: `jobs/m11_race_gpu.sh`

- [ ] CORE2 1N (4r) + **2N (8r — the bridge to M10's only GPU evidence, review M11)**, fArc 4N +
      16N (16/64r); `-C a100_80`; ≤16 nodes; `BIN=h17` CUDA `f8384e86`; matched pairs as Task 11
- [ ] A7-GPU hierarchical arm (nodes×4) raced at CORE2 8 and fArc 16/64
- [ ] stability + accuracy gates as Task 11 (GPU control floors re-measured if Serial floors
      don't transfer)
- [ ] CUDA-vs-Serial fidelity check once per would-be-adopted arm
- [ ] per-backend verdict table; different winners than CPU explicitly allowed

### Task 13: Ordering races (both backends)

**Files:**
- Modify: `jobs/m11_race_cpu.sh`, `jobs/m11_race_gpu.sh` (ordering arms)

- [ ] pure-ordering pairs (label-permuted dists): old vs renumbered mesh, both backends, race
      counts as Tasks 11/12
- [ ] gates: C↔K Serial twin bitwise ON the renumbered mesh; SSH iterations within the
      pre-registered bound (|Δ| ≤ 1/solve, mean < 0.5); violation = stop and diagnose
- [ ] combined arm (new order + scorecard-best partition) raced where pure ordering wins on
      either backend
- [ ] GPU: one ncu profile of the top-3 index-stream kernels (sectors/request, L2 hit) old vs
      new order — mechanism evidence for the report
- [ ] verdicts per backend into the session log

### Task 14: Verdict assembly + finalist hardening

- [ ] finalists re-raced on a second day (the two-pair-day adoption requirement)
- [ ] METIS-seed noise: GENERATE 2–3 seed variants of finalist partition arms via
      `m11_partgen.sh`, gate them (md5 + scorecard + stability screen), then race; spread
      recorded (review M8)
- [ ] **protocol-length stability re-proof (rule 0.41)** for every adopted CORE2/fArc arm at the
      standing protocol dt/length before it is called adopted (review M10)
- [ ] adoption decisions per backend per mesh with the full evidence chain (scorecard row →
      gates → pairs → noise → protocol-length proof)
- [ ] scorecard-vs-measured regression across ALL raced arms incl. anchors — the report's
      headline figure data; lessons → `KOKKOS_PORTING_LESSONS.md`; memory + digest §5 updated

### Task 15: DARS/NG5 confirmation wave (adopted levers only)

**Files:**
- Create (off-repo): `mesh_m11/{dars_m11,ng5_m11}` from
  `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/{dars,ng5}` (+ renumbered variants only if an
  ordering was adopted)

- [ ] ⚠️ ask user before submitting the 32/64-node CPU partition-generation + race jobs
      (DARS 4096 / NG5 8192); GPU stays ≤16N
- [ ] weight-scaling per the int32 ledger (Technical Details): normalize `w` so Σw ≤ 2^30
      (review M13); scorecard records Σw per arm
- [ ] generate adopted-arm partitions at production counts; scorecard + md5 + source-sweep gates
- [ ] one matched pair + full gate set per adopted lever per mesh per backend; rule 0.41 at
      protocol length; `snap_every=-1` at ≥4096 ranks
- [ ] confirm or demote each adoption; verify guard logs clean (no /pool or private-mesh writes)

### Task 16: Adoption packaging

- [ ] `m11_promote` (added to `m11_guards.sh`): the sanctioned exception — creates a NEW
      certified mesh dir (`core2_v2` style) only, never writes into an existing one; copies only
      certified dists; writes MESH_PROVENANCE + md5 manifest; logs the exception (review m18)
- [ ] repoint run configs/job templates in this worktree; document the switch recipe for other
      worktrees (they repoint only after M11 merges)
- [ ] if an ordering is adopted: re-baseline `REFERENCE_RUNS.md` floors on the renumbered mesh
      (C↔K twin reruns per scheme)
- [ ] verify: clean-checkout smoke run against a certified copy works from the documented
      config alone

### Task 17: Report

**Files:**
- Create: `docs/report/m11_partitioning.tex`
- Create: `scripts/m11_figs.py`

- [ ] read `scripts/m7_scaling_figs.py` BEFORE any figure; partition maps via the M9 tooling
- [ ] sections: measured problem (M10 recap WITH the corrected edgecut units and the re-measured
      cut ratio) · pipeline anatomy (Recursive ⇒ CONTIG/VOL never on) — with `fort_part.c`
      diffed against upstream FESOM/fesom2 master and the upstream line cited before any
      "applies to every installation" claim (review m22) · scorecard method + zoo table · race
      results both backends + crossover law (with the 256r rung) · ordering verdict (SFC vs RCM
      for vertex-centred FV — now with an actual RCM arm) · adoption recommendations · ALL nulls
      with scorecard rows
- [ ] headline figure: scorecard-predicted vs measured step time (anchors included)
- [ ] build with the texlive module (M10 Makefile pattern); PDF into docs/report/
- [ ] user read-through pass; revisions

### Task 18: Verify acceptance criteria + close

- [ ] every Overview defect has a measured verdict (fixed, improved, or documented null)
- [ ] all gates green on adopted arms; all checkboxes `[x]` or ➕/⚠️-annotated
- [ ] session log complete (every run id has BIN/env/node-hours/verdict); memory files updated
- [ ] hand user the M10 units-correction note (M10's report tables carry the same edgecut units
      mix — their docs, their call)
- [ ] decide upstream-PR go/no-go with user (injection + knobs + METIS 5.2.1 + pre-connect)
- [ ] move this plan to `docs/plans/completed/`

## Technical Details

- **Weights**: int32 (`idx_t`; 64-bit rebuild is NOT an option — breaks the Fortran `do_partit`
  ABI). `w = a + nlev`, a ∈ {0,…,200}; ledger rule: Σw per constraint ≤ 2^30 — normalize by an
  integer divisor s (`w = round((a+nlev)/s)`) where needed (NG5 7.4M × 270 ≈ 2.0e9 exceeds the
  ledger ⇒ s=4; ratio quantization negligible). `a=0` = pure-3D (5.1.0 aborted here — re-test on
  5.2.1, keep the abort as a finding if it reproduces); `a=∞` = unweighted. `vsize = nlev`
  whenever OBJ=vol (independent array).
- **`FESOM_PART_FILE`**: nod2D whitespace-separated ints, 0-based rank per node in file order;
  the injection path SKIPS `fort_part.c:240`'s `part[i]--`; asserts min=0/max=npes-1/nonempty +
  histogram print.
- **`FESOM_PART_GRAPH_DUMP`**: writes the exact `ssh_stiff` CSR handed to METIS — the
  authoritative reference the exporter must match byte-for-byte (as sorted adjacency).
- **Race protocol constants**: 300-step screen, min-of-2, same allocation, dt per mesh protocol
  (CORE2 1800; others per standing ladders); production env (`FESOM_SPEED=1`) frozen in the
  session log; cheap gates `-t 00:06:00`; protocol-length re-proof per rule 0.41 before adoption.
- **Pre-connect**: virtual edges (weight 1) from each minor wet component to the largest;
  count + sizes logged; applied ONLY when CONTIG=1, never silently.
- **Star-expansion** (Mt-KaHyPar): |V| nets, net_v = {v} ∪ N(v), weight nlev(v); km1 then equals
  METIS totalv exactly.
- **Ordering keys**: hilbert-xyz (Skilling transpose, 21-bit/dim); s2 (argmax-face gnomonic +
  2-D Hilbert per face, faces chained); rcm (scipy, on the exported graph). Elements by centroid
  key; edges regenerated by the partitioner (preserves the `edge2D_in` interior-first convention
  the ice/momentum code relies on).
- **Storage**: per-arm real mesh dirs (no symlinks, no hardlinks — write-through hazard);
  budget ledger in the session log; CORE2 ~80 MB/arm, fArc ~400 MB/arm; NG5 copies only in
  Task 15.
- **Known traps carried in**: partitioner keeps stale on-disk level/edge files while using
  in-memory recomputed levels (M5 divergence check in Task 4) and writes them only when absent
  (delete-then-regenerate for renumbered meshes); CONTIG silently no-ops on disconnected graphs
  (pre-connect + announcement); regenerated partitions have diverged before (stability screen
  before ANY timing); dead-knob trap L80 (announce-or-abort everywhere).

## Post-Completion

**Upstream** (decided in Task 18): PR to FESOM/fesom2 — `FESOM_PART_FILE` injection, knob
family, METIS 5.2.1, component pre-connect; the report is the justification document; include
the upstream `fort_part.c` diff from Task 17.

**Paper**: the report is written to publication standard; the scorecard-vs-measured analysis and
the SFC-vs-RCM GPU verdict stand alone even on nulls.

**M10 correction note**: M10's load-balance report quotes the units-mixed edgecut ratio; hand
the corrected numbers (Task 2 output) to the M10 track — their docs, their edit.

**Other worktrees / production**: M8/M9/M10 and main repoint to certified mesh copies only after
M11 merges to main; renumbered-mesh adoption additionally requires their baselines re-derived
(coordinated switch, not per-track).

**External tooling**: pyfesom2 users of renumbered meshes must drop cached
`pickle_mesh_py3_fesom2` / interpolation caches (deleted in our copies; note for any shared use).
