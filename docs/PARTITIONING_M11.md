# M11 — partitioning & mesh-ordering campaign: session log

Worktree `~/port_kokkos_part`, branch `m11-partition` (from `f42c453`, planned at `7dfdd52`).
Plan: `docs/plans/20260810-m11-partitioning.md` (rev 2). Research: `docs/PARTITIONING_M11_RESEARCH.md`.

This file is the campaign's ledger: every run id, binary, environment, node-hour and verdict.
A number that is not in this file did not happen.

---

## Standing environment

| item | value |
|---|---|
| model binary (CPU) | `/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial`, md5 `5c3c90fc0ea3939df86cfbe275287c36` |
| model binary (GPU) | `/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_cuda`, md5 `f8384e86830620510568b95440e61eab` |
| binary policy | frozen h17 throughout — M11 is an **input-data** campaign; the model is never rebuilt |
| sandbox root | `/work/ab0995/a270088/port2/mesh_m11/` (all mesh writes) |
| run output | `/work/ab0995/a270088/port2/m11/` (never `$HOME`) |
| python | `/work/ab0995/a270088/mambaforge/envs/nereus/bin/python` |
| account / partitions | `ab0995`; `-p compute` (128 r/node), `-p gpu -C a100_80` (4 r/node, ≤16 nodes) |

Read-only sources (never written, md5-baselined in `mesh_m11/.m11state/source_md5.txt`):

| name | path |
|---|---|
| core2 | `/work/ab0995/a270088/port2/mesh/core2` (private copy — L73, never `/pool` core2) |
| farc | `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc` |
| dars, ng5 | `/pool/.../{dars,ng5}` — registered in Task 15 |

---

## Decisions

- **D1 (Task 1).** The guard library fails **closed**: every check `exit 1`s with an
  `M11-GUARD ABORT` line. Callers that need to survive a failure run the guard in a subshell;
  the selftest does exactly this.
- **D2 (Task 1).** `/pool/data/...` is itself a **symlink into `/work/pd1284/...`**, so a
  resolved path never begins with `/pool`. The forbidden-root list therefore names
  `/work/pd1284/` explicitly as well; the positive "inside the sandbox" test is what actually
  caught the case, and both are kept (defence in depth).
- **D3 (Task 1).** Sandbox mesh copies carry only what the model and the partitioner need —
  the five gate files, the three edge files, and the named `dist_N` directories. The
  postprocessing payload (`*_griddes_*.nc`, `fesom.mesh.diag.nc`, `pickle_mesh_py3_fesom2`,
  `distances_*`/`inds_*` caches) is deliberately **not** copied: ~1.0 GB/mesh of material that
  no M11 step reads, and the pyfesom2 caches would be stale on a renumbered mesh anyway
  (Task 5 deletes them by policy).
- **D4 (Task 1, ➕).** The negative control for the halo gate is a *count-preserving* swap of
  two `nod2D` rlist entries rather than a truncation or a bad count: it changes no message
  size and cannot crash the transport, so it tests the gate against **silent wrongness**,
  which is the failure mode a partitioning campaign actually risks.

---

## Plan-review record (rev 2, 2026-08-10, pre-implementation)

`planning:plan-review` on rev 1 returned 3 blockers, 10 majors, 9 minors; all applied before
implementation started. Blockers, and where they live now:

| id | finding | resolution in rev 2 |
|---|---|---|
| B1 | "edgecut ×87–91" mixes units — `fort_part.c:191-205` sets `adjwgt` only for weighted runs, so METIS prints an unweighted cut COUNT for wgt0 and an nlev-weighted cut SUM for the dual arms | research digest §0 carries the correction; Task 2 re-measures both quantities and back-propagates the true ratio into digest + memory before it is quoted |
| B2 | the two partitioner nulls were sequenced against the wrong build | Task 4 splits `partm11-a` (METIS 5.1.0, carries both byte-identity nulls) from `partm11-b` (5.2.1, whose delta is arm **A0**, scored and raced, not asserted away) |
| B3 | sandbox guards had three holes: symlink write-through, MeshPath escape after `sed`, source-side writes | Task 1 closes all three (`cp -aL` + no-symlink assert; MeshPath re-read from the namelist and `readlink`-resolved; source md5 baseline + mtime sweep) |

Majors M4–M13 and minors m14–m22 are cited inline in the plan at the task they modify.

---

## Task log

### Task 1 — mesh sandbox + guard library + session log ✅ (2026-08-10)

**Deliverables**

- `scripts/m11_guards.sh` — the guard library (`m11_assert_sandbox`, `m11_assert_no_symlinks`,
  `m11_namelist_meshpath` / `m11_assert_namelist_meshpath`, `m11_sources_init` /
  `m11_check_sources`, `m11_md5_write` / `m11_md5_check`, `m11_sandbox_copy_mesh`), plus
  `bash scripts/m11_guards.sh selftest`.
- `scripts/m11_corrupt_com_info.py` (➕ discovered) — the halo-gate negative control.
- `jobs/m11_gate_halo.sh` (➕ discovered) — the two-leg gate job.
- Sandbox: `/work/ab0995/a270088/port2/mesh_m11/{core2_m11,farc_m11,gate_negctl}`.

**Sandbox inventory**

| dir | contents | size |
|---|---|---|
| `core2_m11` | 5 gate + 3 edge files, `dist_{4,8,16,32,256,512,864}` | 126 MB |
| `farc_m11` | 5 gate + 3 edge files, `dist_{16,64,2048}` | 356 MB |
| `gate_negctl` | copy of `core2_m11` + `dist_4` with a corrupted `com_info00000.out` | 15 MB |

All copied with `cp -aL`; `find mesh_m11 -type l` → 0 hits. Each dir carries `MD5MANIFEST` +
`MESH_PROVENANCE.md`. Source guard armed 2026-08-10 15:50; `m11_check_sources` (md5 + mtime
sweep over both source trees, ~27 s) clean after every copy.

**Finding — the write hazard is not hypothetical.** All four `/pool` production meshes already
carry `edges.out`, `edge_tri.out`, `edgenum.out`, `nlvls.out`, `elvls.out` **owned by `a270088`**
and dated 2026-05-28 … 2026-07-03 (core2 levels rewritten 2026-07-03 10:46; farc 2026-05-31
23:52; dars 2026-05-31 20:51; ng5 2026-05-29 00:47). Earlier tracks wrote into `/pool` exactly
the way `FESOM/fesom2#852` describes. The md5 baseline freezes that state as M11's reference; the
mtime sweep catches anything M11 adds on top.

**Verification**

- `bash scripts/m11_guards.sh selftest` → **22/22 PASS**. Negative cases: `/pool` path, the
  `/work/pd1284` resolution of it, `/pool` dist dir, private CORE2, private `core2_wgt2`,
  `$HOME`, empty path, the sandbox root itself, `..`-escape out of the sandbox, a symlink inside
  the sandbox pointing at private CORE2 (both as dir and as file), namelist MeshPath → `/pool`,
  MeshPath absent, MeshPath ≠ expected, a directory containing a symlink, a tampered manifest
  file, a deleted manifest file. Positive cases: sandbox subdir, nested sandbox file, clean dir,
  MeshPath → sandbox, manifest write + verify.
- Halo/dist correctness gate (model side), job **26850057**, both legs as designed:

  ```
  leg A control  rc=0  [fesom_halo] identity test (positive): all halo entries carry correct gid
                       [fesom_halo] identity test (corruption recovery): exchange overwrote stale halo
  leg B corrupt  rc=1  rank 0 FAIL: halo[0]   got=20137 expected_gid=1367
                       rank 0 FAIL: halo[142] got=1367  expected_gid=20137
                       [fesom_port FATAL] fesom_halo identity test: 2 halo nodes mismatched
  ```

  The two failing slots and the two swapped gids are exactly the ones injected
  (`rlist[0]` ↔ `rlist[142]`, the boundary between sender PE 1 and PE 2), and all four ranks
  abort. The gate is **not vacuous**: it catches a corruption that changes no count, no message
  size and no wire traffic. The scorecard-side reciprocity check (Task 2) is verified against
  this same `gate_negctl` dist.

**Run table**

| job id | what | nodes×tasks | walltime | BIN | verdict |
|---|---|---|---|---|---|
| 26850057 | halo gate: control + corrupted-com_info legs, CORE2 dist_4 | 1×4 (compute) | 30 s used / 10 min req | h17 Serial `5c3c90fc` | **PASS** (control clean, corrupt aborts) |

**Node-hour ledger:** 26850057 = 0.008 node-h. Task-1 total 0.008 node-h.

---

### Task 2 — scorecard `m11_scorecard.py` ✅ (2026-08-10)

Runs entirely on the login node (no queue): CORE2 ≈ 20 s/arm, fArc 2048r ≈ 43 s/arm including
the 4,096 `my_list`/`com_info` files.

**Regression: 12/12 PASS** against the published M10 numbers — edgecut at CORE2 8/16/32r in both
units, 3-D imbalance (shipped 864r 9.60×, `wgt2` 1.05×), halo 42 → 59 nodes/rank at 864r, and
fArc `/pool dist_2048` 1.004× (2-D) / 9.40× (3-D). Permutation invariance: 26/26 invariant keys
identical under a random relabelling, all 11 ordering keys moved. Negative control: the
scorecard's reciprocity gate FAILs on `gate_negctl` naming the right block
(`rank 0<-1: 1 of 142 gids differ, first at slot 0`) and exits 1.

#### ⭐⭐ Finding 1 — "edgecut ×90" is dead; the true cost of dual weighting is ×1.4–1.6

`fort_part.c:11` defines `USE_EDGE_WEIGHTS`, and lines 191-205 fill `adjwgt` only when
`wgt_type != 0`. So METIS prints an unweighted cut COUNT for the 2-D-only arm and an
nlev-weighted cut SUM (`w_ij = nlev_i + nlev_j`) for the dual arm. M10 divided the second by the
first. Measured, both quantities for both arms, on CORE2:

| ranks | unweighted cut wgt0 → wgt2 | ratio | nlev-weighted cut wgt0 → wgt2 | ratio | M10's mixed ratio |
|--:|--:|--:|--:|--:|--:|
| 8 | 1,335 → 2,144 | **×1.61** | 93,260 → 120,883 | **×1.30** | ×90.5 |
| 16 | 2,549 → 3,724 | **×1.46** | 178,042 → 217,796 | **×1.22** | ×85.4 |
| 32 | 4,307 → 6,200 | **×1.44** | 303,326 → 375,211 | **×1.24** | ×87.1 |

The inflation factor is CORE2's mean edge weight, 61.5 (×1.61 × 56.4 = ×90.5). The corrected
figure is consistent with the +40 % halo growth M10 also measured; ×90 never was. Back-propagated
to `PARTITIONING_M11_RESEARCH.md` §0, `project-m11-partitioning.md` and `MEMORY.md`. M10's own
docs still carry the mix — hand them the number at Task 18, do not edit their files.

#### ⭐⭐ Finding 2 — the METIS "edgecut" print is not the shipped partition's cut

One regression target refused to reproduce: CORE2 `wgt2` 16r, on disk 217,796 against the 217,791
METIS printed — 5 parts in 217,791, while the other five cut targets from the same two jobs
matched to the digit. Ruled out in turn: the graph and the cut definition (the three `wgt0`
unweighted targets match exactly, so both are right); stale weights (`nlvls.out` agrees with
max-over-incident-elements of `elvls.out` at every one of the 126,858 nodes, and the same array
gives exact matches at 8r and 32r); wrong provenance (`dist_8` and `dist_16` carry the mtime of
job 26744882, the run that printed those numbers).

What remains is the only code between the print and the file write: `check_partitioning`
(`fvom_init.F90:1809`) relocates every node with ≤1 same-partition neighbour. An exhaustive
search over every node × every adjacent part (`find_checkpart_moves`) returns **exactly one**
candidate in the whole mesh:

> node gid **125423**, degree 3, `nlev` 5, neighbours with `nlev` 20/5/5. METIS put it in part 9
> where it had one same-part neighbour — the post-pass criterion exactly — and moved it to part 8
> where it had two. The edge to part 9 (weight 25) became cut, the two edges to part 8
> (weight 10 each) became uncut: unweighted cut **−1**, weighted cut **+5**.

⇒ **Rule for the campaign: every cut number comes from the scorecard reading the dist files,
never from the partitioner's stdout.** The post-pass moved nothing in the other five CORE2 arms,
so this is rare rather than systematic — Task 4 will confirm the mechanism directly against
unfiltered partitioner output and count how often it fires across the zoo.

#### Baseline scorecard table

`/work/ab0995/a270088/port2/m11/scorecard_baselines.csv` (22 arms).
`disc` = parts split into >1 component; `offlobe` = vertices outside their part's largest
component; `iso` = nodes with ≤1 same-part neighbour; `halo` = mean halo nodes/rank;
`el_repl` = Σ per-rank elements / elem2D.

| mesh | N | 2Dimb | 3D max/min | cut_unw | cut_nlev | volmax | nbr_max | disc | offlobe | iso | halo | el_repl |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| core2 shipped | 4 | 1.001 | 1.01 | 1,190 | 63,761 | 9,200 | 3 | 3 | 24,970 | 0 | 304.0 | 1.015 |
| core2 shipped | 8 | 1.000 | 1.97 | 1,361 | 94,750 | 10,840 | 3 | 1 | 6,097 | 0 | 172.8 | 1.018 |
| core2 shipped | 16 | 1.000 | 2.38 | 2,563 | 180,481 | 10,665 | 6 | 2 | 2,237 | 0 | 162.5 | 1.033 |
| core2 shipped | 32 | 1.000 | 4.18 | 4,400 | 310,575 | 9,114 | 6 | 2 | 3,146 | 0 | 139.9 | 1.057 |
| core2 shipped | 256 | 1.001 | 9.11 | 16,792 | 1,154,687 | 4,558 | 9 | 6 | 889 | 1 | 68.1 | 1.225 |
| core2 shipped | 512 | 1.005 | 9.26 | 25,382 | 1,722,478 | 3,185 | 8 | 8 | 492 | 0 | 52.1 | 1.346 |
| core2 shipped | 864 | 1.028 | 9.60 | 34,159 | 2,280,429 | 2,735 | 10 | 9 | 324 | **71** | 42.1 | 1.473 |
| core2_wgt0 | 8 | 1.000 | 1.96 | 1,335 | 93,260 | 9,903 | 4 | 1 | 6,201 | 0 | 169.5 | 1.017 |
| core2_wgt0 | 16 | 1.000 | 2.92 | 2,549 | 178,042 | 9,943 | 6 | 1 | 3,029 | 0 | 161.5 | 1.033 |
| core2_wgt0 | 32 | 1.000 | 4.72 | 4,307 | 303,326 | 8,661 | 6 | 2 | 2,994 | 0 | 136.8 | 1.056 |
| core2_wgt0 | 256 | 1.001 | 9.20 | 16,822 | 1,161,066 | 4,675 | 9 | 6 | 994 | 0 | 68.2 | 1.226 |
| core2_wgt0 | 512 | 1.005 | 9.21 | 25,318 | 1,712,679 | 3,289 | 9 | 5 | 371 | 0 | 52.0 | 1.345 |
| core2_wgt0 | 864 | 1.015 | 9.32 | 34,157 | 2,289,966 | 2,707 | 11 | 12 | 507 | 0 | 42.1 | 1.474 |
| core2_wgt2 | 8 | 1.002 | 1.01 | 2,144 | 120,883 | 8,994 | 6 | 7 | 26,445 | 0 | 274.0 | 1.027 |
| core2_wgt2 | 16 | 1.003 | 1.01 | 3,724 | 217,796 | 10,479 | 7 | 13 | 24,311 | 0 | 237.9 | 1.048 |
| core2_wgt2 | 32 | 1.003 | 1.02 | 6,200 | 375,211 | 8,842 | 11 | 28 | 25,556 | 0 | 198.5 | 1.080 |
| core2_wgt2 | 256 | 1.007 | 1.03 | 23,325 | 1,405,784 | 4,037 | 15 | 203 | 28,542 | 3 | 95.7 | 1.314 |
| core2_wgt2 | 512 | 1.017 | 1.04 | 34,878 | 2,106,731 | 3,295 | 15 | 405 | 31,153 | 9 | 72.7 | 1.480 |
| core2_wgt2 | 864 | 1.028 | 1.05 | 46,871 | 2,815,073 | 2,530 | 14 | **677** | 32,491 | 40 | 58.9 | **1.659** |
| farc /pool | 16 | 1.001 | 1.02 | 9,652 | 455,602 | 20,506 | 7 | 13 | 139,989 | 0 | 609.2 | 1.024 |
| farc /pool | 64 | 1.000 | 7.70 | 16,914 | 896,031 | 14,759 | 7 | 4 | 8,049 | 0 | 267.1 | 1.043 |
| farc /pool | 2048 | 1.004 | 9.40 | 124,915 | 6,224,017 | 4,127 | 10 | 21 | 1,884 | 1 | 63.7 | 1.332 |

#### ⭐⭐ Finding 3 — the fragmentation currency, and where dual weighting really pays

Dual weighting at 864r splits **677 of 864 parts** into multiple components (vs 12 for `wgt0`)
and pushes element replication 1.474 → 1.659, i.e. **+12.6 % more replicated element work**.
That is a far better predictor of M10's measured +20 % GPU ocean-busy than the halo-node count's
+0.7 % prediction, and it is measurable offline. Element/edge replication and
`parts_disconnected` are therefore the fragmentation metrics the Pareto prune will use.

#### ⭐⭐ Finding 4 — the shipped-864 mystery is NOT visible in any invariant metric

M10 found the shipped CORE2 `dist_864` 7.4 % faster than our flat regeneration (and +4.18 % on
GPU for the regenerated one). Their scorecards are nearly identical: cut 34,159 vs 34,157, halo
42.1 vs 42.1, element replication 1.473 vs 1.474, comm volume max/rank 2,735 vs 2,707,
disconnected parts 9 vs 12. **Partition QUALITY does not distinguish them.** What the scorecard
cannot see is rank *labelling* — which subdomain becomes which MPI rank, and therefore which
subdomains land on the same node/socket/GPU. That makes the shipped-864 probe (Task 7, A7) a
test of **placement**, not of quality, and adds a cheap new arm: relabel the ranks of an existing
partition (a pure permutation of the dist files — identical geometry, identical cut) and race it.
➕ recorded against Task 7.

One incidental oddity worth carrying: the **shipped** `dist_864` contains **71 nodes with ≤1
same-partition neighbour**, the exact defect `check_partitioning` exists to remove, while every
partition we generated has 0. The shipped partition predates the current tool.

**Node-hour ledger:** 0 (login-node only).

---

### Task 3 — graph exporter + part-vector importer ✅ (2026-08-10)

`scripts/m11_graph_export.py` and `scripts/m11_part_import.py`. Both login-node tools; CORE2
export + full verification 1.9 s, fArc 8.8 s.

**Exporter.** METIS graph format with `fmt = <vsize><vwgt><adjwgt>`; weight variants
`none | vwgt (a+nlev) | vsize (nlev) | both | dual (ncon=2, (1, nlev+100))`, optional
`--edge-weights` (`adjwgt = nlev_i+nlev_j`, what `fort_part.c` hands METIS on weighted arms).
hMETIS star expansion for Mt-KaHyPar: one net per vertex, `net_v = {v} ∪ N(v)`, net weight
`nlev(v)`. The graph itself is imported from `m11_scorecard.Mesh.graph()` — a second
implementation would be a second thing to keep true.

**Importer.** Engine output → `FESOM_PART_FILE` (0-based, one rank per line) with 0/1-based
autodetect, plus `--from-dist` extraction. Refuses to emit a vector with a wrong length, a
non-contiguous rank range, or an empty part — all three would only surface at model start.

**Verification (all PASS, CORE2 and fArc):**

| check | CORE2 | fArc |
|---|---|---|
| header `sum(degree) = 2m` | 743,288 = 2 × 371,644 | 3,783,940 = 2 × 1,891,970 |
| adjacency symmetry (exact multiset) | ✅ | ✅ |
| edge multiset == `Mesh.graph()` | ✅ | ✅ |
| `adjwgt = nlev_i+nlev_j` on every entry | ✅ | ✅ |
| `vsize = nlev`, `vwgt = a + nlev`, dual `(1, nlev+100)` | ✅ | ✅ |
| int32 ledger Σ vwgt (a=100) | 16,518,550 | 78,800,827 |
| cut of a known partition: from file == from graph | 1,361 (dist_8) | 124,915 (dist_2048) |
| importer round-trip dist → vector → import | identical | identical |
| rpart count block agrees with derived assignment | 8/8 ranks | 2048/2048 ranks |
| scorecard on the part FILE == on the dist | 1,361 / 94,750 | — |

⭐ **The star expansion is verified numerically, not asserted:** km1 of the written hypergraph
under CORE2 `dist_8` is **47,620**, exactly METIS's total communication volume with
`vsize = nlev` (scorecard `commvol_total`). Net weight = nlev on every net, net size =
degree + 1 on every net, 870,146 pins = n + 2m, three random patches equal {v} ∪ N(v). So
Mt-KaHyPar's `-o km1` will optimise the quantity we actually pay for, rather than a proxy.

File sizes (for the Task-6/8 storage ledger): CORE2 4.5 MB unweighted / 7.6 MB fully weighted /
6.2 MB hypergraph; fArc 41.5 MB fully weighted.

**Node-hour ledger:** 0 (login-node only).

---

### Task 4 — patched partitioner `partm11`, two builds ✅ (2026-08-10)

Trees, patches and build recipe: `docs/partm11/` (`README.md` +
`fort_part.c.m11.patch` + `mesh_part_CMakeLists.metis521.patch`). Harness:
`scripts/m11_partgen.sh`. Jobs: `jobs/m11_null_partitioner.sh`, `jobs/m11_knob_gate.sh`,
`jobs/m11_a0_and_farcdump.sh`.

| build | METIS | executable md5 | `libfesom_meshpart_C.so` md5 |
|---|---|---|---|
| `fesom2_ref` (pristine) | 5.1.0 bundled | `f46b794eaab2b529dedd2a031ae200b7` | `9cd8b071c8785346d1ec09f18dc85f92` |
| **partm11-a** (patched) | 5.1.0 bundled | `f46b794eaab2b529dedd2a031ae200b7` | `a49236f712ecfb71a6dad6ca40254260` |
| **partm11-b** (patched) | **5.2.1** (`f5ae915`) + GKlib (`3b7d61b`) | `efe4fbd7a61f211dfee4b2079a4a8be7` | `43169061af5d5e56052181cd6da713f8` |

🔴 **The executable md5 is identical between the pristine and patched 5.1.0 builds.** The C
partitioning code compiles into the shared library, so editing `fort_part.c` never changes the
executable. Provenance for this tool is the **`.so`**; `m11_partgen.sh` prints both on every
run. (Related: the executable resolves the library through `$ORIGIN/../lib64`, so **copying it
into a work dir breaks it** — job 26851114 died exactly that way. Run it in place.)

**Knobs** (all announce, all abort on an unrecognised value): `FESOM_PART_FILE`,
`_GRAPH_DUMP`, `_WGT`, `_WGT_A`, `_KWAY`, `_OBJ`, `_VSIZE`, `_CONTIG`, `_MINCONN`, `_UFACTOR`,
`_NCUTS`, `_NITER`, `_SEED`, `_TPWGTS_FILE`. Two deliberate refusals beyond the plan:

- `OBJ=vol`, `MINCONN` and `CONTIG` **abort unless `KWAY=1`**. METIS accepts and silently
  ignores them under `PartGraphRecursive` — which is precisely how contiguity and the
  comm-volume objective were off for every FESOM partition ever generated. The tool now
  refuses to repeat that quietly.
- `CONTIG=1` counts connected components first and **refuses on a disconnected graph** rather
  than pre-connecting behind the user's back (METIS would ignore CONTIG there without a word).

**Verification**

| gate | job | result |
|---|---|---|
| null-1: patched, no knobs ≡ pristine | 26851187 | **PASS** — 17 files byte-identical, `METIS edgecut 120883` = M10's value |
| null-2: inject a partition's own vector | 26851187 | **PASS** — 0 `check_partitioning` moves, `dist_8` byte-identical |
| knob announce / refuse | 26851421 | **26/26 PASS** (13 accept, 12 refuse, 1 short-vector refusal) |
| A0 = 5.2.1 announces its version | 26851421 | PASS — `Metis version 5.2.1` |
| exporter graph vs `FESOM_PART_GRAPH_DUMP` CSR | — | CORE2 **and** fArc: rowptr and colind identical at every entry (743,288 / 3,783,940) |
| M5: in-memory `nlevels_nod2D` vs on-disk `nlvls.out` | — | CORE2 **0 nodes differ**; fArc **1 node differs** (see below) |
| reproducibility of the whole pipeline | 26851187 | our fresh 5.1.0 default `dist_8` is byte-identical to M10's `core2_wgt2/dist_8` |

#### ⭐⭐ Finding 5 — the shipped fArc partitions ARE our tool's output; CORE2's are not

`farc_dump/dist_16`, generated here from the sandbox copy with the historical defaults, is
**byte-identical to `/pool/.../farc/dist_16`**. So fArc's shipped decomposition was produced by
exactly this tool at these settings (dual+100, Recursive, `UFACTOR=1`, `NCUTS=10`, seed 35243,
METIS 5.1.0). CORE2's shipped `dist_864`, by contrast, is *not* reproduced by the same recipe
(M10 measured it 7.4 % faster than our regeneration). That sharpens Task-2 Finding 4: the CORE2
mystery is a partition made by a **different, older tool**, not a different setting of this one.

#### ⭐ Finding 6 — the M5 divergence is real, bounded, and mesh-dependent

The partitioner keeps a pre-existing `nlvls.out` on disk while partitioning with freshly
recomputed in-memory levels. Measured at the seam:

- **CORE2**: in-memory == on-disk at all 126,858 nodes. No divergence.
- **fArc**: they differ at **exactly 1 node of 638,387** — node 1, a coastal node
  (`coast_flag=1`, 140.11 °E 66.84 °S) with only 2 incident elements. On disk `nlvls.out` = 5,
  in memory = **17**. The on-disk arrays are self-consistent (`nlvls.out` equals
  max-over-incident-elements of `elvls.out` at every one of the 638,387 nodes), and **17 is
  exactly the RAW, pre-smoothing level**: `elvls_raw.out` gives 17 for both of that node's
  elements while `elvls.out` gives 5. So this run's rough-topography smoothing did not
  reproduce the archived pass at that node.

Impact bound: the fArc `dist_16` produced *with* the divergent weight came out byte-identical
to the shipped partition, so at 16 ranks it changed nothing. The check is cheap and now
automatic (`m11_graph_export.py --vs-dump`), and must be run **per mesh** — it is mesh-
dependent, and dars/NG5 are unmeasured.

#### Arm A0 — METIS 5.2.1 at the historical settings is a wash

| metric | N=8 | N=16 | N=32 |
|---|--:|--:|--:|
| cut, unweighted | +3.3 % | +1.6 % | +1.1 % |
| cut, nlev-weighted | −0.8 % | −0.1 % | +0.7 % |
| comm volume (total) | −0.5 % | −0.1 % | +0.6 % |
| max neighbours/part | 6 → 7 | 7 → 10 | 11 → 11 |
| disconnected parts | 7 → 7 | 13 → 11 | 28 → 26 |
| vertices off main lobe | −2.7 % | −8.2 % | −5.7 % |
| halo nodes/rank | +3.8 % | +1.6 % | +1.0 % |

Nothing here justifies 5.2.1 on quality grounds — which is the useful result: it means any gain
the Task-7 Kway/VOL/CONTIG/MINCONN arms show is attributable to **those options**, not to the
version bump that makes them reachable. A0 is still raced as the null.

Incidental: `check_partitioning` moved 1 node in the A0 16-rank run too (0 at 8 and 32), so the
print-vs-file gap is a recurring, not a one-off, phenomenon.

**Run table**

| job id | what | elapsed | verdict |
|---|---|--:|---|
| 26851114 | null tests, first attempt | 0:38 | **FAILED as designed** — harness refused when the copied executable could not find its `.so` (rc=127); cause was my copy breaking `$ORIGIN` |
| 26851187 | null-1 + null-2 + CORE2 CSR/levels dump | 1:24 | **PASS** |
| 26851421 | knob gate + A0 version + check_partitioning reproduction | 2:12 | **26/26 PASS** |
| 26851516 | arm A0 at 8/16/32 + fArc CSR dump | 1:02 | PASS |

**Node-hour ledger:** 0.09 node-h (4 jobs, 5 min 16 s total). Task 1–4 cumulative: 0.10 node-h.

#### Finding 6, closed: the current tool does not reproduce fArc's archived smoothed levels

Follow-up job **26851891** ran the partitioner on mesh dirs holding only
`nod2d/elem2d/aux3d`, so it had to write `nlvls.out` and `elvls.out` itself:

| mesh | `elvls_raw.out` (pre-smoothing) | `elvls.out` (smoothed) | `nlvls.out` |
|---|---|---|---|
| CORE2 | identical | **identical** | **identical** |
| fArc | identical | **2 of 1,253,306 elements differ** | **1 of 638,387 nodes differs** |

The differing elements are 1 and 2 — the only two touching node 1, that 2-element Antarctic
coastal node. Archived says 5 (which is `thers_zbar_lev`, the minimum-level floor), fresh says
17 (the raw depth-derived level). So the archived run's **iterative rough-topography pass**
collapsed that isolated cell to the floor and the current one does not; the raw levels agree
byte-for-byte on both meshes, so nothing upstream of the smoothing has moved. Total 3-D nodes:
14,962,127 archived vs 14,962,139 fresh — 12 in 15 million.

Bounded impact for M11: the fArc `dist_16` produced with the divergent weight is byte-identical
to the shipped one. The check is automatic per mesh (`m11_graph_export.py --vs-dump`); dars and
NG5 remain unmeasured. Worth reporting upstream as an isolated-cell edge case, not worth
chasing here.

---

### Task 5 — renumbering converter `m11_renumber.py` ✅ (2026-08-10)

Three orderings: `hilbert-xyz` (Skilling transpose, 21 bits/dim on unit-sphere xyz — the
sphere is embedded in 3-D so there is no dateline or pole seam), `s2` (gnomonic cubed-sphere
face + per-face 2-D Hilbert), `rcm` (scipy, on the same graph METIS partitions). Elements sort
by `minvertex` (default) or `centroid`.

**Classification is mandatory and fails closed.** Every entry in the source directory is
classified node-indexed / element-indexed / special / regenerate / stale / copy, and an
unrecognised name **aborts**. This is the Z7 guard: one node-indexed file left unpermuted is
bitwise-correct at step 1 and wrong at step 2. ➕ `elvls_raw.out` is element-indexed and is
permuted with the rest.

Rows are permuted **as text** wherever their values do not change, so coordinates and depths
are carried through byte-for-byte with no float round-trip. Only `elem2d.out` is parsed, because
its values must be mapped through P_node — with each triangle's vertex **cycle** preserved,
never rotated or sorted.

#### ⭐⭐ Finding 7 — renumbering is a CORE2 lever; on fArc only RCM helps at all

Mean |Δindex| over graph edges, before → after:

| mesh | baseline | hilbert-xyz | s2 | rcm |
|---|--:|--:|--:|--:|
| CORE2 | 32,043 | 459 (**−98.6 %**) | 467 (−98.5 %) | **288 (−99.1 %)** |
| fArc | 956 | 1,058 (**+10.6 %**) | 1,093 (**+14.3 %**) | **676 (−29.3 %)** |

fArc's shipped numbering is already spatially local (Task 2: 88.5 % of element-gather strides
within 64 indices), so a space-filling curve has nothing to add there and actually **loses**.
CORE2's is arbitrary, so everything helps. This decides Task 9's arms before any of them costs
node-hours: **CORE2 races one SFC + RCM; fArc races RCM only**, with the SFCs kept as documented
negatives.

After renumbering, CORE2's element-gather stream goes from **27.6 % to 86.8 %** of accesses
within 64 indices — essentially fArc's native 88.5 %.

#### Smoke, sequenced (job 26852056) — PASS end to end

1. Label-permuted the CORE2 `dist_8` onto the Hilbert mesh; per-part sizes identical.
2. Injected it: the partitioner regenerated `edges/edge_tri/edgenum` for the new numbering,
   `check_partitioning` moved 0 nodes, and no mesh-definition file changed.
3. Certified Serial `h17` ran clean on the renumbered mesh; the halo identity gate announced
   itself and passed.
4. Scorecard: the invariant block is **identical on all 30 keys** — including halo nodes/rank
   and element/edge replication, which are read from the *regenerated* dist files rather than
   derived from the graph — while all 11 ordering keys moved. Cover and reciprocity gates green.

That is the property every ordering A/B depends on: the decomposition is provably the same
decomposition, so any timing difference is attributable to the numbering alone.

**Verification (10/10):** P∘P⁻¹ = id for both permutations; coordinates, coast flag and `nlvls`
follow their node; `elvls` follows its element; `elem2d` vertices mapped with cycles preserved;
element areas identical as a multiset to 0.000e+00; the graph is the same graph under the
permutation; edge files absent; id column the identity.
⚠️ One check initially failed and the bug was in the *check*, not the conversion — it applied
old→new where new→old was needed. Fixed; the distinction is now spelled out in the code.

**Run table**

| job id | what | elapsed | verdict |
|---|---|--:|---|
| 26851891 | fresh-vs-archived level reproduction, CORE2 + fArc | 1:10 | closes Finding 6 |
| 26852056 | renumbering smoke: permute → inject → model → scorecard | 0:47 | **PASS** |

**Node-hour ledger:** 0.03 node-h. Task 1–5 cumulative: **0.13 node-h**; sandbox 1.3 GB.

---

## PRE-REGISTRATION — ordering arms (Task 10, ordering slice) · written 2026-08-10, BEFORE any race

Brought forward from its plan position at user request: the ordering lever has the campaign's
largest offline signal (CORE2 mean |Δindex| −98.6 %, element-gather locality 27.6 % → 86.8 %),
the meshes are built and verified, and nothing about it depends on the engine or zoo tasks.
Everything below is fixed before a single timing number exists.

**Arms** (pure ordering — the partition CONTENT is identical across arms by construction, via
label-permuted dists; Task 5's smoke proved all 30 invariant scorecard metrics equal):

| arm | mesh | partition |
|---|---|---|
| `base` | `core2_m11` (shipped numbering) | its own `dist_N` |
| `hil` | `core2_hil` (3-D Hilbert on unit-sphere xyz) | `dist_N` label-permuted from `base` |
| `rcm` | `core2_rcm` (reverse Cuthill-McKee) | `dist_N` label-permuted from `base` |

fArc is **not** raced for ordering: Task 5 measured the SFCs making its locality worse
(+10.6 %, +14.3 %) and only RCM helping (−29.3 %). fArc RCM is deferred to Task 9 and is not
part of this pre-registration.

**Points.** CPU Serial `h17` (`5c3c90fc`): CORE2 256 r (2 nodes), 512 r (4 nodes), 864 r
(7 nodes). GPU CUDA `h17` (`f8384e86`, `-C a100_80`): CORE2 4 r (1 node), 8 r (2 nodes).
The 512 r point is production; 256 r sits on the winning side of M10's crossover and 864 r on
the losing side, so the grid spans the regime.

**Protocol.** 300 steps, dt 1800 (CORE2 protocol), PHC + JRA55 1958 forcing, `snap_every=-1`,
`FESOM_PRINT_EVERY=999` (printing OFF during timing — M9 measured a per-step diagnostic print at
41 % of a fArc GPU 2N step). All three arms of a point run inside ONE allocation, 2 reps each,
**min-of-2**, same day. `BIN=` pinned and md5-checked in-job.

**Gates, all pre-registered:**

1. **Halo/dist correctness** — the gid identity test, automatic in every run and proven
   non-vacuous in Task 1. Any abort kills the arm.
2. **Stability screen** — 300 steps with no blow-up-guard hit. Failing arms are discarded, not
   debugged (L99/M10: a fresh partition can simply be unusable).
3. **SSH iteration bound** — a symmetric permutation changes only the ORDER of the CG
   reductions, so per-solve iteration counts must satisfy **|Δ| ≤ 1 per step and mean |Δ| < 0.5**
   over the gate leg. A systematic shift means the operator changed, i.e. a bug, not an
   ordering effect. Measured on a SEPARATE 20-step leg with `FESOM_PRINT_EVERY=1`, never on a
   timed leg.
4. **Accuracy** — the printed per-step maxima (`uv`, `eta`, `w`) and `T`/`S` ranges are global
   reductions and therefore **permutation-invariant quantities**: they need no field
   permutation to compare. Pre-registered bound: **relative |Δ| ≤ 1e-9 at step 1** on every one
   of them. A pure relabelling perturbs only summation order, so anything larger is a mesh
   error, not round-off — and it means STOP and diagnose rather than adjust the bound.
5. **Invariant-block equality** — already green (Task 5): if a label-permuted dist ever fails
   it, that arm is not a pure ordering arm and must not be raced as one.

**Adoption rule.** ≥ **2 %** net step-time improvement, per backend, reproduced on **two
same-day pair days**, with all four gates green and rule 0.41 (stability re-proven at protocol
length) satisfied before the word "adopted" is used. A result between 0 and 2 % is reported as
measured and NOT adopted. A negative result is reported with the same prominence.

**Budget.** ≤ 1.5 node-h for the CPU points, ≤ 0.5 node-h for the GPU points, ≤ 0.2 node-h for
the gate legs. GPU stays ≤ 16 nodes (2 here).

**What would falsify the lever:** no arm reaching 2 % on either backend at any of the five
points, despite the −98.6 % locality change. That outcome is publishable as-is — it would say
that a vertex-centred FV ocean code on this hardware is not index-stream-bound, which nobody
has measured.

### ⭐⭐ Finding 8 — `check_partitioning` is not idempotent, and it broke the pure-ordering setup

Building the race inputs, the invariant-block gate refused two of five rank counts
(job **26852384**): the label-permuted arms were **not** carrying the baseline's partition at
N=256 and N=864. The mapping to the scorecard is exact — those are precisely the two shipped
partitions that still contain nodes with ≤1 same-partition neighbour:

| N | shipped `isolated_nodes` | invariant block after label-permutation |
|--:|--:|---|
| 4, 8, 512 | 0 | identical |
| 256 | 1 | **mismatch** |
| 864 | 71 | **mismatch** |

On injection `check_partitioning` relocates those nodes, and because it walks `do n=1,nod2D`
consulting running per-partition loads, **which** node moves **where** depends on the numbering.
So the two arms stopped carrying the same decomposition, and racing them would have been an
ordering-plus-repartitioning race reported as a pure ordering one. The gate caught it before a
single timing number existed — which is what a pre-registered gate is for.

**Fix: settle the baseline to a fixed point first** (job 26852646). Injecting each shipped
vector back into the old mesh once lets the post-pass act there; the settled partition is then
what all three arms carry. Results of settling:

| N | nodes moved while settling | `isolated_nodes` after |
|--:|--:|--:|
| 4, 8, 512 | 0 | 0 |
| 256 | 1 | 0 |
| 864 | **71** | **1** ← still not a fixed point |

⇒ **repairing 71 isolated nodes created a new one.** The post-pass is a single sweep, not an
iteration to convergence, so its output is not guaranteed to satisfy its own criterion. Job
26852879 iterates the injection until `isolated_nodes == 0` and rebuilds the arms from that
fixed point.

Two consequences worth carrying beyond M11: (i) any freshly generated FESOM partition may
contain the very defect the post-pass exists to remove, and only the scorecard will say so;
(ii) the **baseline arm for the ordering race is the settled partition**, not the shipped one —
a fairer comparison, since it removes a confound that has nothing to do with numbering, and it
is documented here rather than folded silently into the numbers.

### ⭐⭐⭐ Finding 9 — an upstream `check_partitioning` bug, found by relabelling

Iterating the settle at N=864 (job 26852879) never converged: five passes, each flagging the
**same** node and leaving it exactly where it was.

| numbering | the node | its neighbours' partitions, in adjacency order | outcome |
|---|---|---|---|
| shipped | 103748 in part 356 | `356 353 351` — **first is its own** | flagged 5×, **never moved** |
| rcm | 98137 in part 356 | `356 353 351` | flagged, **not moved** |
| hil | 55318 in part 356 | `353 351 356` | **"moved to part 353"** — repaired |

Same mesh, same partition, same physical node: the repair succeeds or silently no-ops depending
on the **order of the adjacency list**, which depends on the node numbering. The cause is one
line — `fvom_init.F90:1885`:

```fortran
np = 1
ne_part(1) = node_neighb_part(1)          ! seeded WITHOUT excluding part(n)
...
do i = 1, cnt
   if (node_neighb_part(i) == part(n)) cycle   ! the loop DOES exclude it
```

The candidate list is seeded with the first neighbour unconditionally, while the loop that
fills the rest explicitly skips neighbours in the node's own partition. When the adjacency list
happens to begin with the node's own partition, that partition becomes a move candidate, wins,
and the node is "moved" to where it already was. The routine reports success and the isolated
node survives every subsequent pass.

This is a genuine upstream defect with a one-line fix (seed from the first neighbour **not** in
`part(n)`, or drop the seeding and let the loop build the list). It is invisible without doing
exactly what M11 did — relabel a mesh and compare — and it belongs in the Task-18 upstream PR.

**Consequence for this race:** N=864 cannot be brought to a fixed point by this route, so its
arms cannot be made invariant-identical, so it is **dropped from the pure-ordering race** and
recorded here rather than raced with a silent one-node difference. The grid keeps CPU 256/512
and GPU 4/8.

---

## ORDERING RACE — results (2026-08-10, pair day 1)

Protocol exactly as pre-registered: 300 steps, dt 1800, PHC + JRA55 1958, printing and
snapshots off, all three arms in one allocation, 2 reps, min-of-2, `BIN=` md5-checked in job.

| point | base (s/step) | hilbert | rcm |
|---|--:|--:|--:|
| CPU 256 r (2 nodes) | 0.1074 | 0.1046 **−2.61 %** | 0.1044 **−2.79 %** |
| CPU 512 r (4 nodes) | 0.0596 | 0.0587 −1.51 % | 0.0585 −1.85 % |
| GPU 4 r (1 node) | 0.0675 | 0.0642 **−4.89 %** | 0.0640 **−5.19 %** |
| GPU 8 r (2 nodes) | 0.0619 | 0.0610 −1.45 % | 0.0609 −1.62 % |

**Renumbering wins on both backends, at every point measured, −1.5 % to −5.2 %.** RCM is
marginally ahead of Hilbert everywhere, consistent with its better locality proxy (288 vs 459).
The gain shrinks as ranks grow — at 512 ranks CORE2 holds only ~248 vertices per rank, a
working set small enough that the numbering barely matters, while a single GPU holding ~32 k
vertices is where locality pays. That is the mechanism the lever predicts, and the ordering of
the four points follows it.

### Gates — one passed, one failed on my own instrument

**SSH iterations** (job 26852882, 20 steps, separate from every timed leg): hilbert max |Δ| = 1,
mean 0.050 → **PASS**. RCM max |Δ| = 2, mean 0.550 → **FAILS** the pre-registered bound
(max ≤ 1, mean < 0.5), marginally. Base iterations run 127–141 per step, so a 2-iteration
excursion is ~1.5 %.

**Accuracy — the pre-registered bound was wrong, and it was my error.** I registered "relative
|Δ| ≤ 1e-9 at step 1 on the printed maxima". That is untestable: the diagnostic print carries
three significant digits. It is also the wrong physics — `Kv`, `Av` and `w` are threshold-driven
quantities in a model whose mixing scheme flips boundary-layer depths on round-off, so they are
the worst possible choice for a round-off-level comparison. The gate failed on its instrument,
not necessarily on the meshes. **I am not retrofitting a pass.**

What the evidence actually shows, measured afterwards against the project's own standard — a
**partition-class floor** from control pairs (L79). Both controls hold the numbering fixed and
change only the partition, so they need no permutation, and the comparison tool self-checks the
permutation on `lon`/`lat` before touching any physics (exact, max |Δ| = 0):

| comparison | temp rms | temp p50 | temp p99.99 | ssh rms | ssh p50 |
|---|--:|--:|--:|--:|--:|
| control A — one node moved | 7.03e-06 | 6.0e-14 | 9.8e-06 | 5.5e-08 | 2.5e-10 |
| **control B — different partition** | **4.58e-02** | **1.7e-07** | **2.33** | **6.8e-03** | **7.9e-05** |
| ordering — hilbert | 6.73e-02 | 1.1e-06 | 2.25 | 3.99e-03 | 2.2e-04 |
| ordering — rcm | 1.19e-01 | 1.7e-06 | 3.79 | 9.55e-03 | 4.9e-04 |

The ordering arms land **in the same class as repartitioning** — same distribution shape (median
~1e-6 K over 5.96 M points, a fat tail from a small set of columns), rms within 1.5× (hilbert)
and 2.6× (rcm) of the partition-class floor the project already accepts, and orders of magnitude
above the one-node control. Combined with the mesh-level evidence — provably identical graph,
element areas identical as a multiset, `lon`/`lat` mapping back exactly, halo gate green, SSH
iterations within 1–2 — this is round-off amplified through the mixing thresholds, not a mesh
error. Worth stating plainly though: at the **median**, ordering perturbs ssh ~3–6× more than a
full repartitioning does, plausibly because renumbering reshuffles the local summation order of
every field on every rank whereas repartitioning only regroups the global sums.

### Verdict: measured, NOT adopted

The timing result is real and reproducible within the day. Adoption is **withheld**, for three
independent reasons, none of which the numbers can talk their way out of:

1. the pre-registered accuracy gate failed, and its replacement was designed after seeing the
   data — that makes it evidence, not a passed gate. It must be re-registered and re-run.
2. RCM misses the pre-registered SSH-iteration bound.
3. the adoption rule requires a **second same-day pair day**, which by definition cannot exist yet.

Next actions, in order: re-register the accuracy gate as a partition-class-floor comparison with
the floor measured first; re-run both gates; second pair day; then decide. The CPU 864 point
returns only if Finding 9 is fixed locally.

**Run table**

| job id | what | nodes | elapsed | verdict |
|---|---|--:|--:|---|
| 26852384 | label-permuted dists v1 | 1 | 4:38 | **refused by the invariant gate** at N=256/864 (Finding 8) |
| 26852646 | settled baselines + arms | 1 | 5:10 | 4 of 5 counts green; 864 still not a fixed point |
| 26852879 | settle 864 to a fixed point | 1 | 2:54 | **never converges** → Finding 9 |
| 26852880 | CPU race 256 r | 2 | 4:12 | base 0.1074 / hil −2.61 % / rcm −2.79 % |
| 26852881 | CPU race 512 r | 4 | 2:51 | base 0.0596 / hil −1.51 % / rcm −1.85 % |
| 26852883 | GPU race 4 r | 1 | 2:42 | base 0.0675 / hil −4.89 % / rcm −5.19 % |
| 26852884 | GPU race 8 r | 2 | 2:34 | base 0.0619 / hil −1.45 % / rcm −1.62 % |
| 26852882 | ordering gate leg | 2 | 0:46 | SSH iters hil PASS / rcm FAIL; accuracy bound unusable |
| 26853331 | accuracy control (first try) | 2 | 0:04 | **failed on my bash bug** — `local t=$1 o="$OUT/$t"` expands all args before assigning, so `$t` was unbound. The race scripts carried the same latent bug and only worked because their loop variables happened to share the names; fixed in all four |
| 26853377 | accuracy control | 2 | 0:41 | controls A and B measured |

**Node-hour ledger:** ordering race 0.72 node-h. Campaign cumulative: **0.85 node-h**.
