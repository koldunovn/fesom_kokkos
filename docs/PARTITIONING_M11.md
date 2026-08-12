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

---

# SESSION 2 (2026-08-10, evening)

## ⭐⭐ Finding 10 — the day-1 gate leg and the 256-rank race arm were run against the WRONG baseline

Before re-registering anything, a check of the inputs the day-1 numbers were produced from.
`jobs/m11_race_cpu.sh` and `jobs/m11_ordering_gate.sh` both set the base arm to
`core2_m11` — the **shipped** mesh — while `hil` and `rcm` carry the **settled** partition that
Finding 8 introduced precisely because the shipped one could not be relabelled invariantly.
Measured on the files:

| N | `core2_m11/dist_N/rpart.out` vs `core2_base/dist_N/rpart.out` |
|--:|---|
| 4 | identical |
| 8 | identical |
| **256** | **differs** (the one node `check_partitioning` moved while settling) |
| 512 | identical |

So three of the four raced points were clean, and **CPU 256 r was an ordering-plus-one-node
race** — the point with the largest CPU gain (−2.79 %) and the point every gate leg was run at.
The confound is one node in 126,858 and cannot plausibly explain a 2.8 % step-time difference,
but it does invalidate the words "pure ordering" at that point, and it is a live confound in the
SSH-iteration gate, where a single relocated node reorders the CG reductions on two ranks.

Fixed in both jobs (base = `core2_base`), with a **refusal** added: each job now extracts the
part vector of every arm's `dist_N` and aborts unless all three are the same partition up to the
node permutation. A confound that is one `sed` away from returning should not be prevented by
remembering.

## RE-REGISTRATION — ordering gates, v2 · written 2026-08-10 18:5x, BEFORE the new legs ran

The day-1 accuracy gate failed on its own instrument (three-significant-digit prints of
threshold-driven quantities), and its replacement was designed after seeing the data. This
section replaces it. What follows is fixed before the jobs are submitted; the numbers it will be
judged on do not exist yet.

**Honesty note, recorded rather than hidden.** I have already seen the day-1 evidence table
(hilbert temp rms 1.5×, rcm 2.6× of a *single* control), and the day-1 SSH-iteration numbers
(hil max 1 / mean 0.050, rcm max 2 / mean 0.550) — both against the wrong baseline. Any
multiplier I choose now is therefore informed, not blind. Two things are done about that rather
than claimed away: (i) gate **R1** below is a genuinely blind test — its bound comes from
floating-point arithmetic, not from any measured M11 number, and it discriminates by ten orders
of magnitude, so no choice of constant changes its verdict; (ii) gates **R2**/**R3** are
calibrated against an *ensemble* of three partition-class controls instead of the single control
day 1 used, and every control is reported alongside every arm so a reader can apply their own
multiplier.

### R1 — step-1 field identity (blind, first-principles, the gate that actually tests the mesh)

Each arm runs **1 step** (dt 1800, cold start, monthly output = the state after step 1) and is
compared to the reference through the node permutation, which is itself self-checked on
`lon`/`lat` (exact, max |Δ| = 0) before any physics is read.

**Bound: median |ΔT| ≤ 1e-12 K, median |ΔS| ≤ 1e-12 psu, median |Δssh| ≤ 1e-12 m.**

Derivation, independent of every M11 measurement: the fields are O(1)–O(30) in double
precision, so one arithmetic operation carries ~1e-15 relative round-off; a single time step
applies O(10²) operations per node, so a difference in *summation order alone* cannot move the
median past ~1e-13. A mesh error of the class this campaign risks — any node-indexed array left
unpermuted, a triangle's vertex cycle rotated, `nlvls` following the wrong node — displaces
whole columns and moves the median to O(1e-2)–O(1). The bound sits ten orders below the smallest
such error and three above pure round-off, so it cannot be tuned into either verdict.

The **maximum** is explicitly NOT gated: KPP's boundary-layer depth is a discontinuous
functional of the state, so a handful of columns will differ at O(1) at step 1 in any run of
this class — including the controls. That is the mechanism day 1 mistook for a failure.

The same instrument is run on the controls. If a control (same numbering, different partition)
fails R1, the instrument is measuring something other than round-off and the gate is void.

### R2 — partition-class floor, ensemble (the in-class check)

20 steps, reference = the settled baseline, statistic = **rms over the whole field** for
`temp`, `salt`, `ssh`. Three controls, all holding the numbering fixed and changing only the
partition, so none needs a permutation:

| control | what changes |
|---|---|
| A | one node reassigned (shipped vs settled `dist_256`) |
| B | a wholly different partition (`core2_wgt0` `dist_256`, M10's 2-D-only arm) |
| C | the same recipe at a different METIS seed (`FESOM_PART_SEED`, fresh `dist_256`) |

**Bound: arm rms ≤ 3 × max(control A, B, C) for each of temp, salt, ssh.**
K = 3, fixed here. Rationale for the multiplier rather than the number: three controls sample a
distribution whose spread is not known in advance, and the gate must fail an arm that is out of
class by an order of magnitude while not failing one that is an ordinary draw from it.

### R3 — SSH iterations, re-derived from first principles

The day-1 bound (max |Δ| ≤ 1, mean |Δ| < 0.5) assumed the iteration count is insensitive to
round-off. It is not, and the assumption was wrong for a reason worth writing down: the count is
a **threshold crossing on a residual norm**, and from step 2 onward the two model states
themselves differ — the mixing scheme amplifies round-off to O(1) in individual columns within
one step. So |Δ| has no a-priori bound of 1, on any arm, for any correct mesh.

What the gate exists to catch is the failure it was named for: **the operator changed**. That
shows up as a *systematic* shift — consistently more (or fewer) iterations — not as a symmetric
excursion. Measured over the same 20-step leg, for every control and every arm against the same
reference:

- **R3a (no systematic shift):** |mean signed Δ| ≤ max(0.5, 2 × max over controls |mean signed Δ|)
- **R3b (magnitude in class):** mean |Δ| ≤ 2 × max over controls mean |Δ|, **and**
  max |Δ| ≤ max(2, 2 × max over controls max |Δ|)

K_ssh = 2, fixed here. The per-step signed sequence is reported for every arm so the shift, if
any, is visible rather than summarised.

### Unchanged from the day-1 pre-registration

Arms, points, protocol, the halo/dist correctness gate, the stability screen, the
invariant-block equality gate, the ≥ 2 % adoption threshold, and the requirement of **two
same-day pair days**. Day 1's CPU-256 pair is retired by Finding 10 and re-run; the other three
points stand.

## ORDERING GATES v2 — results (2026-08-10 evening, jobs 26854848 at 256 r and 26854938 at 512 r)

Two rank counts, six legs each (reference + three controls + two arms), R1 at 1 step and R2/R3
at 20 steps, all inside one allocation per rank count.

### The instrument proves itself first

At 512 ranks the shipped and settled `dist_512` are byte-identical, so the `ship` leg is a
bitwise null — and it reports **exactly zero** on every statistic: temp/salt/ssh rms 0.000e+00,
median 0.000e+00, max 0.000e+00, SSH iteration Δ = 0 at all 20 steps. The whole chain (netCDF
read, permutation handling, statistics) returns a hard zero when nothing changed. At 256 the
same leg is the one-node control instead, and it reports temp median exactly 0 with ssh median
1.7e-13.

### ⭐⭐⭐ Finding 11 — the accuracy floor is the SSH SOLVER TOLERANCE, not machine epsilon

R1 was pre-registered with the clause "if a control fails, the instrument is void, not the arm".
**That clause fired**: two of the three controls miss the 1e-12 median bound at both rank
counts. The derivation behind the bound assumed round-off is the only mechanism available in
one time step. It is not, and the reason is measurable:

| leg (256 r) | SSH iteration Δ over 20 steps | temp median at step 1 | ssh median at step 1 |
|---|--:|--:|--:|
| ship (one node moved) | 0 at every step | **0.000e+00** | **1.70e-13** |
| wgt0 (different partition) | 2 steps differ | 7.14e-09 | 1.25e-05 |
| seed (different seed) | 2 steps differ | 9.70e-09 | 1.90e-05 |
| hil | 1 step differs | 6.26e-08 | 9.16e-05 |
| rcm | 9 steps differ | 1.22e-07 | 2.18e-04 |

The SSH solve is a CG iteration stopped on a **relative residual of 1e-5**
(`FESOM_PHASE1_SOLTOL`, `src/fesom_constants.h:105`; `rtol = soltol·√(‖rhs‖²/N)`, break on
`residual < rtol`, `fesom_ssh.cpp:446,518`). Two decompositions that follow the **same**
iteration path land on the **same iterate** and stay at round-off — that is the `ship` leg. Any
decomposition change that alters the path lands on a **different admissible iterate**, and the
set of admissible iterates is as wide as the stopping criterion allows: a residual of 1e-5
relative becomes a solution difference of `1e-5 × κ(A)`. Measured against the step-1 ssh field
(rms 0.0375 m, max 0.35 m):

| leg | median \|Δssh\| | relative to the field rms |
|---|--:|--:|
| ship (same iteration path) | 1.70e-13 m | **4.5e-12** |
| wgt0 | 1.25e-05 m | 3.3e-04 |
| seed | 1.90e-05 m | 5.1e-04 |
| hil | 9.16e-05 m | 2.4e-03 |
| rcm | 2.18e-04 m | 5.8e-03 |

Eight orders of magnitude separate the leg that kept the iteration path from the four that did
not, and the four sit at 3e-4 … 6e-3 relative — consistent with a 1e-5 residual tolerance and an
operator conditioning of O(10²–10³), which is ordinary for a 2-D elliptic SSH operator. The
floor is realised in **step 1** and it is domain-wide (a median over 5.96 M points, not a few
columns). The two ordering arms land 7× and 17× further out than the repartitioning controls —
inside the same admissible set, but further inside it.

Three consequences worth carrying beyond M11:

1. **No accuracy gate on this model can be tighter than 1e-5 relative for any change that
   touches the decomposition or the numbering.** Day 1's "relative |Δ| ≤ 1e-9 at step 1" was
   unreachable for any instrument, not just for a three-digit print.
2. The project's partition-class floor (L79) is not a convention — it is the only bound the
   solver leaves available.
3. The **maximum** is uninformative here for a second, independent reason: KPP's boundary-layer
   depth is a discontinuous functional, so O(1) column differences appear in every leg,
   controls included (temp max 4.7–10.3 K at step 1 across all of them).

R1 is therefore recorded as **void by its own control clause**. The mesh-identity question it
was meant to answer is settled offline instead — 10/10 renumbering invariants, 30/30 scorecard
invariant keys, element areas identical as a multiset, and `lon`/`lat` mapping back exactly in
the model's own output (checked in every leg above).

### R2 — partition-class floor, both arms in class at both rank counts

rms over the whole field at 20 steps, arms compared through the node permutation:

| | temp rms | salt rms | ssh rms | verdict |
|---|--:|--:|--:|---|
| **256 r** control floor (max of A/B/C) | 7.19e-02 | 2.83e-01 | 8.96e-03 | |
| hil | 6.73e-02 (0.94×) | 3.16e-02 (0.11×) | 3.99e-03 (0.45×) | **PASS** |
| rcm | 1.19e-01 (1.66×) | 7.18e-02 (0.25×) | 9.55e-03 (1.07×) | **PASS** |
| **512 r** control floor | 5.56e-02 | 1.70e-01 | 6.42e-03 | |
| hil | 6.83e-02 (1.23×) | 1.99e-01 (1.17×) | 7.32e-03 (1.14×) | **PASS** |
| rcm | 1.14e-01 (2.05×) | 2.81e-01 (1.65×) | 1.62e-02 (2.52×) | **PASS** |

Both arms sit inside the pre-registered 3× band at both counts. Hilbert is consistently the
quieter of the two — at or below the floor at 256, ~1.2× at 512 — while RCM runs 1.7–2.5×.

### ⭐⭐ Finding 12 — RCM fails the re-derived SSH-iteration gate at BOTH rank counts

| | mean signed Δ | mean \|Δ\| | max \|Δ\| | bound (2× control ensemble) | verdict |
|---|--:|--:|--:|---|---|
| 256 r hil | −0.050 | 0.050 | 1 | 0.500 / 0.200 / 2 | **PASS** |
| 256 r rcm | −0.450 | **0.550** | 2 | 0.500 / 0.200 / 2 | **FAIL** mean \|Δ\| |
| 512 r hil | −0.050 | 0.050 | 1 | 0.500 / 0.100 / 2 | **PASS** |
| 512 r rcm | −0.450 | **0.850** | 2 | 0.500 / 0.100 / 2 | **FAIL** mean \|Δ\| |

Not marginal: RCM is 5.5× the bound at 256 and 8.5× at 512, against controls that sit at
0.00–0.10. And the per-step sequence has a shape — at 512 it runs
`[-1,0,1,1,1,1,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2]`: a sustained one-iteration deficit that
sets in after ~10 steps at both counts. Hilbert's sequence is a single −1 in 20 steps, exactly
what the controls do.

The bound was re-derived from first principles **before** these legs ran (see the
re-registration above), and the failure reproduces at two independent rank counts, so this is a
gate result and not a tuned one. Why RCM perturbs the solver ~10× more than a repartitioning
while Hilbert does not is **not explained** here; the mesh is provably the same mesh, so the
difference is in summation order alone, and its magnitude is unaccounted for.

### Verdict on the ordering arms: carry Hilbert, drop RCM

**RCM is dropped.** It fails a pre-registered, first-principles gate at both rank counts, and
the handoff's own condition for keeping it ("either RCM is dropped, or the bound is re-derived
from first principles *before* re-measuring") was met and it still failed.

**Hilbert is carried.** It passes R2 and R3 at both rank counts, sits inside the control
ensemble on every statistic, and gives up almost nothing in speed (below).

## ORDERING RACE — CPU 256 r re-run on the corrected baseline (job 26854862)

Same protocol, same allocation, min-of-2, with `base` = the settled `core2_base`:

| point | base s/step | hilbert | rcm |
|---|--:|--:|--:|
| day 1, shipped base (retired) | 0.1074 | −2.61 % | −2.79 % |
| **corrected, settled base** | **0.1073** | **−2.42 %** | **−2.24 %** |

Two things change. The gain is ~0.2–0.5 pp smaller, and **the day-1 claim that "RCM is
marginally ahead of Hilbert everywhere" does not survive the re-run** — here Hilbert leads. The
arms' own absolute times moved 0.1 % (hil 0.1046 → 0.1047) and 0.5 % (rcm 0.1044 → 0.1049)
between the two same-day runs, so the hil–rcm separation (0.2 pp) was always inside the noise.
The lever is ~2.4 % at this point either way.

The other three points (CPU 512, GPU 4, GPU 8) used dists that are byte-identical between the
shipped and settled meshes, so their day-1 numbers stand unchanged.

## ⭐⭐⭐ Finding 13 — the `check_partitioning` fix, and 864 rejoins the race (job 26854955)

`FESOM_PART_FIXISO=1` (default off) builds the candidate list only from neighbours outside
`part(n)`. Announced by the tool, aborts on any value but 0/1, and **verified to fire** (the
`[M11] FESOM_PART_FIXISO=1` line is present in the ON leg's log and absent in the OFF leg's —
L80, not a dead knob).

**The seed line has two defects, and the second one is what actually fired here.** Besides
admitting the node's own partition as a move candidate, the seed counts the first neighbour
**twice** — once in the seed, once again when the loop finds it already in the list. At CORE2
864 the node Finding 9 named (gid 103748, own partition 356, neighbours in `356 353 351`):

| numbering | candidate list, stock | outcome, stock | outcome, FIXISO=1 |
|---|---|---|---|
| shipped / rcm | `{356:1, 353:1, 351:1}` → kmax count 1 | not moved (`ne_part_num(kmax) <= 1`) | not moved |
| hil (`353 351 356`) | `{353:`**2**`, 351:1}` → kmax count 2 | **"moved to part 353"** | **not moved** |

So under the stock code the repair fires or not depending on whether the double-counted
neighbour happens to be first — which is a property of the numbering, not of the mesh. With the
fix, the node is left in place under **every** numbering, with the honest reason the routine
already has for this case ("no chance — this is probably a boundary node that has only two
neighbours").

**`isolated_nodes == 0` was the wrong convergence criterion, and my leg C used it.** That node
is genuinely irreparable by this heuristic: it has exactly one neighbour in each of two other
partitions, so no move can give it two same-partition neighbours. Iterating the injection with
the fix gives passes 2–5 of `flagged 1, moved 0` — the partition **is** a fixed point; what is
not zero is a metric. The right criterion is idempotence, which leg D tests directly:

| N=864 with FIXISO=1 | result |
|---|---|
| pure-ordering check (part vector under the permutation) | **EXACT** for both arms |
| invariant block, 30 keys, base vs hil | **IDENTICAL** |
| invariant block, 30 keys, base vs rcm | **IDENTICAL** |
| cover / reciprocity gates | ok / ok |

⇒ **CPU 864 is back in the ordering race** (it was dropped in session 1 precisely because these
arms could not be made invariant-identical), and the fix is upstream-PR material with a
two-defect explanation and a reproduction.

The job exits FAIL only because leg C asserts the wrong criterion; the criterion is fixed in the
job for the next run rather than the verdict being reinterpreted after the fact.

Also settled by leg A: **the rebuilt executable with no knobs still reproduces the reference
`dist_8` byte-for-byte** (17 files, `METIS edgecut 120883`). ➕ Correction to Task 4's
provenance note: the pristine and patched 5.1.0 executables are **not** md5-identical
(`f46b794e…` vs `0af460a2…`, and they differ in size by 48 KB — build-path and flag noise; even
`gpmetis` differs). The rule that came out of it is unchanged and now doubly true: record BOTH
md5s, because the C knobs live in `libfesom_meshpart_C.so` and the Fortran fix lives in the
executable. Current partm11-a: executable `29b14f200547eb0d473c47e408470061`, libC
`a49236f712ecfb71a6dad6ca40254260`.

### Run table — session 2

| job id | what | nodes | elapsed | verdict |
|---|---|--:|--:|---|
| 26854807 | control C (seed-variant `dist_256`) | 1 | 1:40 | PASS |
| 26854848 | ordering gate v2, 256 r | 2 | 2:07 | R1 void (controls fail), R2 both PASS, R3 hil PASS / rcm FAIL |
| 26854862 | CPU race 256 r, corrected baseline | 2 | 4:08 | base 0.1073 / hil −2.42 % / rcm −2.24 % |
| 26854917 | control C `dist_512` | 1 | 0:47 | PASS |
| 26854938 | ordering gate v2, 512 r | 4 | 2:05 | same pattern; `ship` leg is a bitwise zero null |
| 26854955 | FIXISO: null re-verify + Finding-9 case + settle + 864 arms | 1 | 4:09 | null-1 PASS, knob fires, **864 arms invariant-identical**; exits FAIL on leg C's wrong criterion |

**Node-hour ledger:** 0.028 + 0.071 + 0.14 + 0.013 + 0.14 + 0.069 = **0.46 node-h**.
Campaign cumulative: **1.31 node-h**.

---

### Task 6 — engine builds + wrappers ✅ (2026-08-10)

`scripts/m11_engines.sh` (one entry point for six engine variants) and `scripts/m11_zoo_b.sh`
(the sweep driver). Trees under `/work/ab0995/a270088/port2/partm11/engines/`.

| engine | tag | build | CORE2 512 wall time |
|---|---|---|--:|
| KaMinPar | v3.7.3 | `-DKAMINPAR_BUILD_DISTRIBUTED=OFF -DKAMINPAR_BUILD_WITH_SPARSEHASH=OFF -DKAMINPAR_BUILD_WITH_MTUNE_NATIVE=OFF` | 0–2 s |
| Mt-KaHyPar | v1.6.2 | `-DKAHYPAR_DOWNLOAD_TBB=On`, then re-configure with `-DCMAKE_EXE_LINKER_FLAGS=-pthread` for the CLI | 3–6 s |
| KaHIP | v3.25 | `-DNOMPI=ON -DPARHIP=OFF -DNONATIVEOPTIMIZATIONS=ON` | 28–29 s |

Toolchain: `gcc/13.4.0-gcc-13.4.0` + `cmake/3.31.11-gcc-13.4.0`, built on the **login node** —
KaMinPar and Mt-KaHyPar fetch TBB over the network at configure time and a compute node has no
external route. None of the three had to be dropped; each cost exactly one flag:

- KaMinPar wants Google Sparsehash, which Levante does not carry (`KAMINPAR_BUILD_WITH_SPARSEHASH`
  defaults ON).
- Mt-KaHyPar's **library** links but its **CLI** does not: `undefined reference to
  pthread_setspecific … DSO missing from command line`. `CMAKE_EXE_LINKER_FLAGS=-pthread` and a
  relink fix it; nothing else needs rebuilding.
- KaHIP's default build wants MPI for ParHIP/kaffpaE; `kaffpa` alone is sequential.
- Mt-KaHyPar v1.6.2 no longer needs Boost (older versions did — the plan's note is out of date).

All three consume the Task-3 exports unchanged and their output passes `m11_part_import.py`'s
refusals (length, contiguous rank range, no empty part) at CORE2 512.

#### ⭐⭐⭐ Finding 14 — every external engine is SINGLE-constraint, and that is the whole story

KaMinPar, Mt-KaHyPar and KaHIP all balance **one** vertex weight. FESOM's legacy arm balances
**two** (`ncon=2`, `(1, nlev+100)`), which is why `core2_wgt2` reaches 2-D imbalance 1.017 **and**
3-D max/min 1.037 at the same time. No external engine can express that, so the shared axis for
the whole zoo is the scalar `w = a + nlev`. Measured at CORE2 512 (`--weights none` = unit
weights), the trade is severe and monotone:

| arm (ε = 3 %) | cut_unw | commvol | **cv_max** | 2-D imb | 3-D max/min | disc | offlobe |
|---|--:|--:|--:|--:|--:|--:|--:|
| shipped (METIS dual, UFACTOR=1) | 25,382 | 902,540 | 3,185 | 1.005 | 9.26 | 8 | 492 |
| `core2_wgt2` (METIS dual) | 34,878 | 1,115,000 | 3,295 | **1.017** | **1.04** | 405 | 31,153 |
| kahip a=0 | **23,656** | 909,267 | **2,657** | 4.40 | 1.66 | **2** | **233** |
| kahip a=15 | 23,962 | 891,692 | 2,796 | 2.26 | 3.06 | 8 | 709 |
| kahip a=100 | 23,897 | 865,277 | 2,895 | 1.275 | 6.91 | 13 | 1,012 |
| kahip unweighted | 23,861 | **847,778** | 3,156 | **1.025** | 10.54 | 7 | 645 |
| mtkahypar a=0 | 24,618 | 913,637 | 2,735 | 5.07 | 1.51 | 10 | 1,294 |
| kaminpar a=100 | 25,686 | 920,917 | 3,100 | 1.251 | 6.74 | 18 | 1,255 |

Reading the table: at their design slack the engines beat METIS's shipped cut by 5–7 % and its
**max-per-rank comm volume by up to 17 %** (2,657 vs 3,185) — but every arm that fixes the 3-D
imbalance pays 4–5× in 2-D node count, which is the currency the ice model and every per-rank
2-D allocation are billed in. The dual constraint buys both balances for +37 % cut; the engines
buy cut and comm volume for one balance or the other. **That trade, not algorithmic strength, is
the real difference between FESOM's partitioner and the state of the art.**

#### ⭐⭐⭐ Finding 15 — at FESOM's OWN slack, METIS beats all three engines on every metric

The comparison above is not matched: the engines ran at ε = 3 %, FESOM's partitioner at
`UFACTOR=1` = **0.1 %**. Re-run at ε = 0.001:

| arm (ε = 0.1 %) | cut_unw | commvol | cv_max | disc | iso |
|---|--:|--:|--:|--:|--:|
| shipped METIS (UFACTOR=1) | **25,382** | **902,540** | **3,185** | **8** | **0** |
| kaminpar unweighted | 26,764 | 947,222 | 3,546 | 96 | 68 |
| kahip unweighted | 27,421 | 965,709 | 3,895 | 35 | 31 |
| mtkahypar unweighted | 31,705 | 1,142,816 | 4,072 | 127 | 279 |
| kaminpar a=0 | 27,056 | 1,037,215 | 3,426 | 123 | 110 |
| mtkahypar a=0 | 31,470 | 1,138,316 | 3,782 | 106 | 303 |
| kahip a=0 | 33,886 | 1,313,085 | 5,258 | 166 | 95 |

**Every engine loses to METIS on cut, on total and max comm volume, on disconnected parts and on
isolated nodes** — by 5 % (KaMinPar) to 33 % (KaHIP a=0) on the cut, and by an order of magnitude
on fragmentation. The engines' 3 %-slack advantage was the slack, not the algorithm; asked for
0.1 % balance they fall apart, which is fair to say only with the caveat that 0.1 % is outside
the envelope they are designed and tuned for (their papers report ε ≥ 1 %).

⇒ **The lever is not "a better partitioner". It is the balance/imbalance trade itself**, and the
matched test in the other direction — give METIS the engines' slack (`UFACTOR` 10/30/100, arm A5)
— is the one that decides whether FESOM should keep `UFACTOR=1` at all. That arm is queued in
zoo A wave 2.

### ➕ `scripts/m11_placement.py` — scoring what the scorecard is blind to

Task-2 Finding 4 left a hole: the shipped CORE2 `dist_864` and our regeneration agree on every
invariant metric yet differ 7.4 % in step time, so the difference has to live in something
invariant metrics cannot see — the rank **labels**, and therefore which subdomains share a node.
This tool measures that directly. With SLURM's block distribution (`--ntasks-per-node=128`, what
every race job uses), rank *r* lives on node *r*//128, so the halo traffic splits into on-node
and off-node parts:

| partition | nodes | off-node 2-D | off-node 3-D | as % of all halo |
|---|--:|--:|--:|--:|
| CORE2 `dist_256` | 2 | 239 | 8,317 | **1.4 %** |
| CORE2 `dist_512` | 4 | 797 | 26,617 | **2.9 %** |
| CORE2 `dist_864` | 7 | 3,126 | 108,760 | **9.0 %** |
| fArc `dist_2048` | 16 | 7,139 | 192,754 | **5.9 %** |

#### ⭐⭐⭐ Finding 16 — a rank count that does not tile the node triples the off-node traffic

fArc at 2048 ranks spreads over **16** nodes and ships 5.9 % of its halo off-node. CORE2 at 864
spreads over **7** and ships 9.0 % — more inter-node traffic across fewer node boundaries, on a
mesh five times smaller. The difference is that 2048 = 16 × 128 exactly while
864 = 6.75 × 128: METIS 5 labels parts in recursive-bisection order, which is spatially
coherent, so consecutive blocks of 128 ranks land on one node **only when the rank count tiles
the node**. 256 = 2 × 128 and 512 = 4 × 128 tile it and sit at 1.4 % and 2.9 %; 864 does not and
pays 3× the trend.

What that is worth, measured as the inter-block halo of the best top-level cut we can build
(which is exactly the off-node volume a hierarchical `n_part = <nodes>,128` would pay):

| point | flat, off-node 3-D | best top-level cut | headroom |
|---|--:|--:|--:|
| CORE2 512 → 4 nodes | 26,617 | 21,836 (Mt-KaHyPar) | −18 % of 2.9 % ⇒ **negligible** |
| CORE2 864 → 7 nodes | 108,760 | **37,917** (Mt-KaHyPar) | **−65 %** |
| fArc 2048 → 16 nodes | 192,754 | 155,493 (Mt-KaHyPar) | −19 % |

⇒ Two concrete recommendations, both testable in one race:

1. **At a ragged rank count, hierarchical partitioning is worth 65 % of the off-node traffic**
   (arm A7). At a tiling rank count it is worth nothing — METIS's natural labelling is already
   there. That is a sharper statement than "try hierarchical", and it says where.
2. **Or simply run 896 ranks instead of 864.** 864 on 7 nodes leaves 32 cores idle *and* pays
   the ragged-labelling penalty; 896 = 7 × 128 fills the nodes and tiles the labelling.

Also measured: a naive greedy re-labelling of an existing partition (agglomerate the
part-communication graph into groups of 128) buys −5.1 % of off-node volume at CORE2 864 and
**loses 67 %** at fArc 2048. METIS's labelling is near a local optimum and should not be
disturbed by hand — which downgrades arm **A8** from "a free lever" to "measured, ~5 % of a
9 % term at the one rank count where it helps at all". The placement hypothesis for the
shipped-864 mystery survives but does not explain it: shipped 109,037 vs regenerated 112,246 is
**2.9 %** of off-node volume, in the right direction but an order of magnitude short of 7.4 % of
step time.

Applied to the B-family arms at CORE2 512 (4 nodes), placement erodes but does not erase their
advantage — all three engines number their blocks coherently enough to keep 96–97 % of the halo
on-node, against METIS's 97.1 %:

| arm | off-node 3-D | vs METIS | total commvol vs METIS |
|---|--:|--:|--:|
| METIS settled | 26,617 | — | — |
| Mt-KaHyPar a=0 | 27,628 | +3.8 % | +1.2 % |
| KaHIP unweighted | 29,413 | **+10.5 %** | **−6.1 %** |
| KaMinPar a=100 | 37,381 | +40.4 % | +2.0 % |

So KaHIP's 6 % saving in *total* comm volume comes with 10 % *more* of it crossing a node
boundary — the currency that costs. Any Pareto prune that ranks on total comm volume alone would
pick it for the wrong reason; the shortlist uses the off-node column too.

#### ⭐⭐ Mt-KaHyPar's reported objective IS FESOM's communication volume — exactly, on every arm

Task 3 verified the star expansion for a single partition (km1 of the written hypergraph =
47,620 = METIS's `totalv` with `vsize = nlev`). The engine's own optimisation target now
confirms it end to end: Mt-KaHyPar's final `km1` and the scorecard's `commvol_total`, computed
independently from the mesh and the part vector, agree **to the digit** on all five weight
variants at CORE2 512:

| arm | km1 (engine) | commvol (scorecard) |
|---|--:|--:|
| a=0 | 913,637 | 913,637 |
| a=15 | 891,062 | 891,062 |
| a=40 | 880,739 | 880,739 |
| a=100 | 875,706 | 875,706 |
| unweighted | 910,438 | 910,438 |

So the hypergraph arm is not optimising a proxy: whatever Mt-KaHyPar improves, FESOM stops
shipping. (Which makes Finding 15 sharper rather than softer — the engine optimises the right
quantity and still loses to METIS once it is held to FESOM's own balance tolerance.)

### Task 10 (partial) — the CORE2 512 shortlist, with the prediction written down first

`scripts/m11_pareto.py`. With five objectives almost nothing is dominated (15 of 16 arms sit on
the front), so a bare Pareto front is not a prune. The prune is **feasibility first, front
second**, and the feasibility filter is stated rather than buried in a weighted score:

> **`n2d_imb ≤ 1.30`.** The ice model and every per-rank 2-D allocation scale with the 2-D node
> count, so an arm's 2-D imbalance multiplies the slowest rank's ice work directly. M9 measured
> the ice at 5–13 % of the GPU step and more on CPU; a 4–5× 2-D imbalance cannot be bought back
> by any communication saving. Ten of sixteen arms fail it — every `a ∈ {0, 15, 40}` variant.

Front after the filter (CORE2 512, 4 nodes):

| arm | cv_max | off-node 3-D | 3-D max/min | 2-D imb | disc |
|---|--:|--:|--:|--:|--:|
| kahip a=100 | **2,895** | 35,900 | 6.91 | 1.275 | 13 |
| mtkahypar a=100 | 2,958 | 31,688 | 7.53 | 1.259 | 24 |
| kaminpar a=100 | 3,100 | 37,381 | 6.74 | 1.251 | 18 |
| kahip unweighted | 3,156 | 29,413 | 10.54 | **1.025** | **7** |
| METIS settled (baseline) | 3,185 | **26,617** | 9.26 | **1.005** | 8 |

**Shortlist to race**, four arms and an anchor, each with the question it answers:

1. `metis_settled` — the baseline.
2. `kahip a=100` — the best feasible max-per-rank comm volume (−9.1 %) at a 1.28 2-D imbalance.
3. `kahip unweighted` — the same balance profile as the baseline (2-D 1.025) with −6.1 % total
   comm volume but **+10.5 % off-node**: it isolates cut quality from balance, and it tests
   whether total or off-node volume is the currency.
4. `mtkahypar a=0` — **deliberately infeasible** (2-D imbalance 5.07) and raced anyway: it has
   the best 3-D balance in the whole zoo (1.51 vs the baseline's 9.26) and off-node volume
   within 4 % of the baseline. It is the test of the feasibility filter itself. If it wins, the
   filter is wrong and there is a large lever behind it; if it loses, the filter is validated by
   measurement instead of assertion.
5. anchor `core2_wgt2` — M10's dual-constraint arm, exempt from pruning so the
   predicted-vs-measured regression spans the range.

**Prediction, registered before the race:** all four engine arms land within ±2 % of the
baseline at CORE2 512. Reasoning: at this point only 2.9 % of the halo leaves the node, and the
off-node volume moves from 26,617 to 27,628–37,381 across the shortlist while total comm volume
moves −6 % … +2 %. If a ±2 % band is what comes out, the honest conclusion is that partition
quality is not the lever at 512 ranks and the campaign's weight should go to the ordering lever
and to the ragged-rank-count effect (Finding 16). A result outside the band falsifies the
off-node-volume model and is the more interesting outcome.

#### The 864 point is where an engine arm has something to win

Repeating the placement analysis at CORE2 864 — the ragged rank count, where 9.0 % of the halo
leaves the node instead of 2.9 % — changes which arms look interesting:

| arm | off-node 3-D | vs METIS | **per-node max** | vs METIS |
|---|--:|--:|--:|--:|
| METIS settled | 108,760 | — | 24,028 | — |
| kaminpar a=100 | 101,664 | **−6.5 %** | **20,062** | **−16.5 %** |
| kahip unweighted | 105,701 | −2.8 % | 20,395 | **−15.1 %** |
| mtkahypar unweighted | 106,209 | −2.3 % | 25,471 | +6.0 % |
| mtkahypar a=0 | 109,899 | +1.0 % | 21,131 | −12.1 % |
| kahip a=0 | 134,039 | +23.2 % | 23,980 | −0.2 % |

Two arms cut the **per-node maximum** off-node volume — the node whose NIC has the most to ship,
which is what the step actually waits on — by 15–17 %, where at 512 ranks the same arms moved it
by nothing worth racing. ⇒ **the B-family shortlist gets a second point: CORE2 864 with
`kaminpar a=100` and `kahip unweighted`.** Both keep 2-D imbalance inside the feasibility filter
(1.25 and 1.03), so no assumption has to be relaxed to race them.

That is consistent with Finding 16 rather than an accident of it: at a rank count that tiles the
node, METIS's labelling already keeps 97 % of the halo local and there is nothing to win; at a
ragged one, 9 % is in play and a partitioner that happens to group differently picks up part of
it.

### The fArc zoo — the strongest feasible engine result in the campaign

fArc's shipped partitions ARE this tool's output (Finding 5), so these deltas are against a
partition the production configuration actually uses, not against a reconstruction. ε = 3 %,
w = a + nlev; deltas vs the shipped METIS partition at the same rank count:

| point | arm | cut | total commvol | **cv_max** | 2-D imb | 3-D max/min |
|---|---|--:|--:|--:|--:|--:|
| **2048** (16 nodes) | mtkahypar a=100 | −3.3 % | −2.4 % | **−18.0 %** | 1.209 | **8.78** |
| | kahip a=100 | −6.4 % | −3.7 % | −14.9 % | 1.209 | 10.76 |
| | kaminpar a=100 | +1.1 % | +3.6 % | −12.2 % | 1.203 | 7.08 |
| | kahip unweighted | −6.3 % | −6.3 % | −9.5 % | 1.027 | 12.19 |
| | mtkahypar a=0 | −7.2 % | +7.0 % | −34.0 % | *4.80* | 1.67 |
| | kahip a=0 | −9.8 % | +6.2 % | −26.1 % | *4.83* | 2.67 |
| **64** | mtkahypar a=100 | +0.5 % | −7.3 % | **−12.1 %** | 1.191 | 7.60 |
| | mtkahypar a=0 | −3.7 % | +1.2 % | −21.4 % | *2.41* | 1.24 |
| **16** | mtkahypar a=100 | −28.0 % | −28.0 % | **−9.6 %** | 1.187 | 3.84 |
| | kahip a=0 | −42.0 % | −26.5 % | +1.8 % | *2.06* | 1.13 |

*italic* = fails the `n2d_imb ≤ 1.30` feasibility filter.

**`Mt-KaHyPar` with `w = 100 + nlev` cuts the max-per-rank communication volume by 10–18 % at
every fArc rank count while keeping the 2-D imbalance under 1.21** — and at 2048 it is the only
feasible arm that also *improves* the 3-D balance (8.78 against the shipped 9.40). KaHIP at the
same weight takes twice as much off the cut (−6.4 %) but gives 3-D balance back (10.76), and
KaHIP unweighted keeps the 2-D balance at 1.027 with 3-D at 12.19. So the three feasible arms
are not variations on one candidate: they trade cut, off-node volume and 3-D balance against
each other, which is what makes racing them informative rather than redundant.

Two cautions to carry into the race:

- At fArc 16 the engines look spectacular (cut −26…−42 %, comm volume −25…−28 %) because the
  shipped 16-rank partition is dual-constrained and pays heavily for its 1.02 3-D balance. That
  is the same trade as Finding 14, not a new lever.
- Every one of these arms is ε = 3 %. Finding 15 says the advantage evaporates at FESOM's own
  0.1 %. The race therefore tests *both* the arm and the slack; if `mtkahypar a=100` wins, the
  next question is whether METIS at `UFACTOR=30` gets there too (arm A5), which is much cheaper
  to adopt than a new dependency.

**Updated shortlist to race** — CORE2 512 (4 arms + anchor, above), CORE2 864 (`kaminpar a=100`,
`kahip unweighted` — the per-node-max placement arms), fArc 2048 (`mtkahypar a=100`).

#### ⚠️ …and the placement column tempers it: at fArc 2048 the engines' advantage is in the CHEAP traffic

The `cv_max` improvements above are max-per-**rank** communication volume. Split that by where
it goes (128 ranks/node, 16 nodes):

| arm | off-node 3-D | vs shipped | per-node max off-node | vs shipped |
|---|--:|--:|--:|--:|
| METIS shipped | 192,754 (5.9 %) | — | 27,962 | — |
| mtkahypar a=100 | 200,026 (6.3 %) | **+3.8 %** | 26,954 | −3.6 % |
| kahip unweighted | 211,686 (7.0 %) | +9.8 % | 30,422 | +8.8 % |
| kahip a=100 | 215,883 (6.9 %) | +12.0 % | 30,721 | +9.9 % |
| kaminpar a=100 | 229,716 (6.8 %) | +19.2 % | 33,462 | +19.7 % |
| mtkahypar a=0 | 218,479 (6.3 %) | +13.3 % | 26,155 | −6.5 % |

**Every engine arm ships more data across node boundaries than the shipped partition does** —
by 4 % to 19 % — while cutting the busiest rank's total volume by 10–18 %. The two currencies
disagree in sign because at 128 ranks per node most of a rank's exchange is with neighbours on
the same node, so `cv_max` is dominated by traffic that never touches the fabric. The saving is
mostly in the cheap kind; the expensive kind gets worse.

⇒ **Revised expectation for the fArc 2048 race, registered before it runs:** `mtkahypar a=100`
lands between −1 % and +1 % of the baseline, not at the −18 % its headline metric suggests. It
stays on the shortlist because it is the only arm that improves the per-node maximum *and* the
3-D balance, and because a two-currency disagreement is precisely the case where a measurement
settles something an offline score cannot. If it wins clearly, `cv_max` is the right currency
and the off-node column is a distraction; if it loses, the off-node column is the one to score
the rest of the campaign on.

### Consequence for fArc: the ordering lever has no adoptable arm there

Task 5 measured the two space-filling curves making fArc's index locality **worse** (+10.6 %,
+14.3 % mean |Δindex| over graph edges) because its shipped numbering is already spatially local
(88.5 % of element-gather strides within 64 indices). Only RCM helped (−29.3 %). RCM has now
failed the SSH-iteration gate at two CORE2 rank counts, and the mechanism the failure points at
— a reordering of every local summation, not of the global ones — is a property of the ordering,
not of the mesh.

⇒ **fArc's ordering arms are: two documented negatives and one arm that fails a gate.** Unless
the RCM verdict is overturned, the ordering lever is a CORE2 lever carried by Hilbert alone.
This is an inference from CORE2 measurements; it should be stated as such in the report, and it
costs one gate leg on fArc to convert into a measurement if the lever ever matters there.

### ➕ Combined arms (Task 9's third bullet) prepared offline

The combined arm — the Hilbert numbering carrying a *partition* arm rather than the baseline
partition — needs no new partitioning: it is the flat arm's part vector read through the same
node permutation the mesh was built with. Written for the three CORE2 512 shortlist partitions
(`core2_hil_{kahip_anone,kahip_a100,mtkahypar_a0}_k512.part` under
`/work/ab0995/a270088/port2/m11/engines/`), each verified to carry per-part sizes identical to
its flat twin. They inject through `jobs/m11_zoo_b_dists.sh` like any other vector, and the
invariant-block gate applies to them unchanged — so a combined arm that fails it is not a
combined arm.

---

## ⭐⭐⭐ Finding 17 — CORE2 is the ONLY mesh that needs renumbering (dars, NG5, FORCA20 surveyed)

`scripts/m11_ordering_survey.py`, read-only against the production mesh directories, no cluster
time. Calibrated against Finding 7 first: it reproduces CORE2 hilbert at 86.8 % element-gather
stride and −98.6 % mean |Δindex|, and fArc's +10.6 %/+14.3 %/−29.3 %, to the digit.

| mesh | nodes | **shipped** mean \|Δidx\| | **shipped** elem stride ≤64 | hilbert-xyz | s2 | rcm |
|---|--:|--:|--:|--:|--:|--:|
| **CORE2** | 126,858 | **32,043** | **27.6 %** | −98.6 % | −98.5 % | −99.1 % |
| fArc | 638,387 | 956 | 88.5 % | +10.6 % | +14.3 % | −29.3 % |
| FORCA20 | 2,127,871 | 1,613 | 88.5 % | +5.4 % | +25.1 % | −34.0 % |
| dars | 3,160,340 | 2,022 | 88.2 % | +2.6 % | +31.5 % | −39.6 % |
| NG5 | 7,402,886 | 2,935 | 88.3 % | +6.0 % | +42.3 % | −29.9 % |

(percentages are the change in mean |Δindex| over graph edges; negative = better locality)

**CORE2 is the outlier, not the rule.** Every other production mesh already ships with a
spatially local numbering — element-gather stride 88.2–88.5 %, against CORE2's 27.6 %, and mean
|Δindex| of 1,000–3,000 on meshes of 2–7 million nodes. There is nothing for a space-filling
curve to fix: on all four, Hilbert is a wash to slightly worse (+2.6 % to +10.6 %) and the
cubed-sphere curve is clearly worse (+14 % to +42 %).

⇒ **dars, NG5 and FORCA20 do not need renumbering, and none is built.** The ordering lever is a
repair for one mesh's arbitrary numbering, not a general optimisation. Task 15's "renumbered
variants only if an ordering was adopted" is answered before it costs anything: for these meshes
the answer is no, on measurement rather than on budget.

### ⭐⭐ …and the survey explains WHY RCM had to be dropped

RCM is the one ordering that improves the bandwidth proxy on every mesh (−29 % to −40 %) — and
it is the only one that **destroys the element-gather stream**: 88 % → 36 % on every large mesh,
27.6 % → 40.1 % on CORE2 where Hilbert reaches 86.8 %. RCM minimises matrix bandwidth, which is
the mean |Δindex| over *graph edges*; the kernels pay for consecutive *gather addresses*, and
those are different objectives. Its p95 |Δindex| is 3–13× worse than the shipped numbering's on
every mesh (2,464 vs 353 on dars) — the long tail RCM's mean hides.

So the two independent verdicts agree, which is worth stating because they were reached
separately: RCM fails the SSH-iteration gate at two CORE2 rank counts (Finding 12), and it is
the wrong objective for this code's access pattern (here). Hilbert is the ordering arm.

## ⭐⭐ ORDERING RACE — CPU 864 r, the point FIXISO restored (job 26855115)

| arm | s/step | vs base |
|---|--:|--:|
| base (settled) | 0.0448 | — |
| **hilbert** | 0.0438 | **−2.23 %** |
| rcm | 0.0440 | −1.79 % |

Pure-ordering precheck EXACT for both arms, so this is the decomposition the baseline carries,
relabelled. The point exists at all because of Finding 13; it was dropped in session 1.

### 🔴 This kills the mechanism the campaign has been quoting

Session 1 explained the ordering gain as a working-set effect — "the gain shrinks as ranks grow;
at 512 ranks CORE2 holds only ~248 vertices per rank, small enough that the numbering barely
matters, while a single GPU holding ~32 k is where locality pays". The full CPU set now reads:

| ranks | vertices/rank | hilbert |
|--:|--:|--:|
| 256 | ~496 | −2.42 % |
| 512 | ~248 | −1.51 % |
| 864 | ~147 | **−2.23 %** |

**Not monotone.** The smallest per-rank working set gives nearly the largest gain, so the
working-set story cannot be the whole mechanism and should not be repeated as if it were. A
plausible replacement — untested, and labelled as such — is that two terms compete: the
element/edge gather inside the kernels, whose share falls as ranks grow, and the halo
pack/unpack gather, whose share rises (halo/owned goes 0.137 at 256 ranks to 0.33 at 864, and at
864 nine per cent of that halo crosses a node boundary, Finding 16). Renumbering makes both
gathers contiguous. Deciding between them needs a phase breakdown, not another race.

What is safe to say without a mechanism: **the ordering lever gives −1.5 % to −2.4 % on CPU at
every rank count measured (256, 512, 864) and −1.5 % to −4.9 % on GPU, and it does not decay
with scale.** That is a better result than the session-1 story implied, and it rests on four CPU
points and two GPU points rather than on the explanation.

## 🔴🔴 Finding 18 — `commvol_max_rank` is an extreme-value statistic, and the seed alone moves it 10–17 %

The zoo-A job carried a control I had not planned to use this way: `legacy` and `seedb` differ
**only** in the METIS seed. At CORE2 512 they read cut 34,878 vs 34,778 (−0.3 %), total comm
volume 1,115,243 vs 1,112,111 (−0.3 %) — and **`cv_max` 3,295 vs 2,981, a 9.5 % move.**

Checked directly on the engine side, three seeds of the *same* Mt-KaHyPar configuration at
fArc 2048:

| seed | cut | total commvol | cv_max |
|---|--:|--:|--:|
| 1 | 120,849 | 3,166,661 | 3,238 |
| 2 | 121,084 | 3,174,100 | 3,745 |
| 3 | 120,763 | 3,167,078 | 3,776 |
| **spread** | **0.3 %** | **0.2 %** | **16.6 %** |

`cv_max` is a maximum over k parts, so it samples the tail of a distribution and moves with the
seed while the sums do not.

**Consequence, and it is a correction to this session's own reporting:** the headline I recorded
for the fArc B-family — `mtkahypar a=100` at **−18.0 % cv_max** — is *inside the noise of its own
engine at a different seed*. It is not a lever as stated. The same applies to every `cv_max`
delta in this log below ~17 %: the CORE2 512 shortlist's "−9.1 %", the 864 "−12.2 %", the fArc
"−9.6 / −12.1 %".

What survives, because the sums are stable to 0.3 %: the **cut**, the **total comm volume**, the
**off-node volume** (a sum), the balances, and the fragmentation counts. The Pareto objective set
is corrected accordingly — `commvol_max_rank` is demoted from an objective to a reported
diagnostic, and any arm ranked on it must be regenerated at 2–3 seeds first (which is what
Task 14 already requires of finalists, one task too late to have caught this).

Re-reading the shortlist under the corrected rule: `kahip unweighted` at CORE2 512 (cut −6.1 %,
total comm volume −6.1 %, both stable metrics) and `mtkahypar a=100` at fArc 2048 (cut −3.3 %,
total comm volume −2.4 %) remain worth racing; the arms whose entire case was `cv_max` do not.

## ⭐⭐⭐ Finding 19 — M10's shipped-864 "7.4 %" does NOT reproduce (job 26855293)

Three partitions of the same mesh (five mesh files verified md5-identical across all three
directories), one allocation, interleaved reps, min-of-2, CORE2 864 ranks:

| arm | s/step | vs shipped |
|---|--:|--:|
| ship (the shipped `dist_864`, an older tool) | 0.0442 | — |
| base (the same partition settled with FIXISO) | 0.0441 | −0.23 % |
| **wgt0 (M10's flat regeneration)** | 0.0448 | **+1.36 %** |

M10 reported this regeneration **7.4 % slower**. Measured here it is **1.36 % slower** — same
sign, five times smaller, and within sight of the run-to-run spread (the reps of an arm differ by
0.4–1.1 % once the first, cold run of the job is excluded; min-of-2 is what the protocol takes).

⇒ **The shipped-864 mystery is retired.** It was never explicable by partition quality (Task 2),
placement (−2.9 % of off-node volume, Finding 16) or per-node aggregate work (identical), and the
reason those three offline explanations all fell short by an order of magnitude is that the
effect they were asked to explain is itself an order of magnitude smaller than reported. The
honest reading: there is a real but small penalty for the flat regeneration at 864 ranks, ~1.4 %,
and no mystery.

Two things this does **not** say. It does not say M10 measured wrong — a different binary,
build or day can carry a different number, and their GPU leg (+4.18 %) is untested here. And it
does not license reading 1.36 % as a finding: it is one pair day at two reps, so it is an upper
bound on the size of whatever is there.

Incidental, and useful: **settling the shipped partition costs nothing** (−0.23 %, i.e. nothing
at this resolution), which is what the ordering race needed to be true of its baseline.

## Task 8 closed — the engine dists exist, run, and are better on the metrics that are stable

Job 26855445 injected the three shortlisted engine vectors, generated real `dist_512`
directories, scored them **with the dist files present** (which is where element replication and
halo counts come from) and smoke-ran one under the halo/dist correctness gate:

| arm | cut | commvol | halo/rank | elem repl | edge repl |
|---|--:|--:|--:|--:|--:|
| METIS settled | 25,382 | 902,491 | 52.1 | 1.3463 | 1.1389 |
| kahip a=100 | 23,896 (−5.9 %) | 865,264 | 49.3 (**−5.5 %**) | 1.3300 (**−1.21 %**) | 1.1310 |
| kahip unweighted | 23,861 (−6.0 %) | 847,778 | 49.1 (**−5.7 %**) | 1.3286 (**−1.31 %**) | 1.1307 |
| mtkahypar a=0 | 24,618 (−3.0 %) | 913,637 | 50.7 (−2.6 %) | 1.3383 (−0.59 %) | 1.1351 |

All three carry **less halo and less element replication than the METIS baseline**, and both are
sums rather than extreme-value statistics, so unlike `cv_max` (Finding 18) they are not seed
noise. Element replication is the fragmentation currency that predicted M10's +20 % GPU ocean
busy where halo counts predicted +0.7 % (Finding 3), so a −1.3 % on it is a small but honest
prediction of a win rather than a wash.

Gates: `owned cover` and `com_info reciprocity` ok on all three; Serial 20-step smoke on
`kahip a=100` clean with the halo identity test announcing and passing.
⚠️ `check_partitioning` moved **128 nodes** when injecting `kahip a=100` (0 for the other two),
so that arm as raced is the engine's partition with 128 nodes relocated — recorded rather than
assumed away, and it is why the move count is printed for every arm.

---

## ⭐⭐⭐ Finding 20 — RACED: a partition arm wins 4.4 % at CORE2 512, and it is NOT the one the cut predicted

Job 26857530. Five partitions of the same mesh (five mesh files md5-identical across all arms),
one allocation, interleaved reps, min-of-2, 300 steps, 512 ranks:

| arm | s/step | vs base | rep spread | offline: cut | total commvol | 2-D imb | 3-D max/min |
|---|--:|--:|--:|--:|--:|--:|--:|
| base (METIS dual, UFACTOR=1) | 0.0595 | — | 1.8 % | 25,382 | 902,491 | 1.005 | 9.26 |
| **kahip a=100** | **0.0569** | **−4.37 %** | 0.4 % | 23,896 | 865,264 | 1.275 | **6.91** |
| wgt2 (M10's dual anchor) | 0.0584 | −1.85 % | 0.7 % | 34,878 | 1,115,243 | 1.017 | **1.04** |
| kahip unweighted | 0.0595 | **+0.00 %** | 0.0 % | 23,861 | 847,778 | 1.025 | 10.54 |
| mtkahypar a=0 | 0.1018 | **+71.09 %** | 0.2 % | 24,618 | 913,637 | **5.07** | 1.51 |

**Three things fall out, and my pre-registered prediction (±2 % for every arm) is falsified —
which is the outcome I said would be the more interesting one.**

1. **The lever is 3-D load balance, not communication.** `kahip unweighted` improves the cut by
   6.0 % and the total communication volume by 6.1 % and buys **exactly nothing** (+0.00 %, with
   a 0.0 % rep spread — the cleanest null in the campaign). `kahip a=100` improves the cut by
   about the same amount and wins 4.4 % — and the only material difference between them is 3-D
   imbalance, 6.91 against 10.54. **Cut quality alone is worth zero here.** Every offline metric
   the campaign has been ranking on except the balances would have picked the wrong arm.
2. **The 2-D feasibility filter is validated, hard.** `mtkahypar a=0` has the best 3-D balance in
   the zoo (1.51) and is **71 % slower** because its 2-D imbalance is 5.07. The filter was a
   stated modelling assumption at `n2d_imb ≤ 1.30`; racing an arm that violates it deliberately
   turned it into a measurement. There is an optimum in `a`, and it is interior.
3. **M10's dual arm is 1.85 % FASTER at 512 ranks**, not slower. That is consistent with their
   own crossover law (dual wins at 256, neutral at 512, loses at 864) and with reading 3-D
   balance as the lever — but it also means the "repartitioning is out" conclusion belongs to
   864 ranks, not to CORE2 generally.

Every delta above except `kahip unweighted`'s exceeds its arm's rep spread, and the three fast
arms have spreads of 0.2–0.7 %, so the ordering of the table is not noise. One pair day, so the
adoption rule still wants a second.

### What this changes about what to measure next

The winning arm is a **scalar weight `w = a + nlev` at 3 % slack**. Two questions now decide
whether FESOM needs an external partitioner at all, and both are one race:

- **Is it the weight or the slack?** METIS with the *same* scalar weight at its own tight
  `UFACTOR=1` (arm `a3_a100`, already generated) against METIS with the same weight at 3 %
  (`a5_u30`) against the KaHIP winner.
- **Where is the optimum in `a`?** The engine sweep spans 2-D imbalance 2.26 / 1.60 / 1.28 and
  3-D 3.06 / 5.31 / 6.91 for a = 15 / 40 / 100. The 71 % catastrophe at a = 0 and the 4.4 % win
  at a = 100 bracket it.

Both are queued (jobs 26857589, 26857590).

## ⭐⭐⭐ Finding 21 — the win is available from METIS ALONE, and it is two knob settings

Job 26857651, CORE2 512 ranks, same protocol. The question was whether the 4.4 % belongs to
KaHIP or to the *formulation* KaHIP was given (a single scalar weight at 3 % slack).

| arm | s/step | vs base | rep spread |
|---|--:|--:|--:|
| base — METIS as shipped (dual constraint, `UFACTOR=1`) | 0.0594 | — | 2.0 % |
| **METIS `w=100+nlev`, `UFACTOR=30` (3 %)** | **0.0569** | **−4.21 %** | 0.2 % |
| **KaHIP `w=100+nlev`, ε=3 %** | **0.0569** | **−4.21 %** | 0.4 % |
| METIS `w=100+nlev`, `UFACTOR=1` (0.1 %) | 0.0579 | −2.53 % | 0.3 % |
| METIS `w=100+nlev`, `UFACTOR=100` (10 %) | 0.0578 | −2.69 % | 0.7 % |
| KaHIP `w=40+nlev` | 0.0572 | −3.70 % | 0.7 % |
| KaHIP `w=15+nlev` | 0.0645 | **+8.59 %** | 0.0 % |

**METIS at 3 % slack lands on KaHIP's number to four digits.** The external engines are not
needed for this lever; what was needed was to stop asking METIS for the wrong thing. Two runtime
knobs, both already in `partm11`:

```
FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=100 FESOM_PART_UFACTOR=30
```

Both settings matter and both have an interior optimum: the same weight at FESOM's own
`UFACTOR=1` gives only −2.53 %, at 10 % slack −2.69 %; and along the weight axis a=15 is
**8.6 % slower**, a=40 −3.70 %, a=100 −4.21 %, unweighted 0.00 %, pure-nlev +71 %.

This also retires Finding 15 as a conclusion about the engines: METIS did beat them at 0.1 %
slack, but the right reading is that **0.1 % was the wrong tolerance**, not that the engines are
weak.

## ⭐⭐⭐ Finding 22 — the two levers stack: 2×2 factorial, CORE2 512 (job 26857849)

Ordering × partition in one allocation, four arms, `MIXED_NUMBERING=1` so the mesh-identity
check groups by numbering instead of refusing:

| arm | numbering | partition | s/step | vs base |
|---|---|---|--:|--:|
| base | shipped | METIS shipped | 0.0596 | — |
| hil | **Hilbert** | METIS shipped | 0.0585 | −1.85 % |
| kahip | shipped | **`w=100+nlev` 3 %** | 0.0572 | −4.03 % |
| **hil_kahip** | **Hilbert** | **`w=100+nlev` 3 %** | **0.0563** | **−5.54 %** |

Sum of the single-lever gains is −5.88 %; measured together, −5.54 %. **The levers are ~94 %
additive** — they act on different things (index locality inside a rank versus how much work each
rank has) and the small negative interaction is the overlap. Rep spreads on the three fast arms
are 0.0–0.2 %, so the ordering of the table is not noise.

## ⭐⭐⭐ Finding 23 — fArc 2048 on 16 nodes: −6.8 %, the largest win in the campaign

Job 26857848, dt 900 (fArc's production step; the first attempt at CORE2's dt 1800 blew up on
**every** arm including the baseline, which identified it as my protocol error rather than a
partition failure — recorded because a partition campaign that reads a blow-up as an arm result
is one wrong dt away from a false verdict).

| arm | s/step | vs base | 2-D imb | 3-D max/min |
|---|--:|--:|--:|--:|
| base — fArc's shipped `dist_2048` | 0.0827 | — | 1.004 | 9.40 |
| **mtkahypar `w=100+nlev`** | **0.0771** | **−6.77 %** | 1.209 | 8.78 |
| kahip unweighted | 0.0845 | +2.18 % | 1.027 | 12.19 |
| mtkahypar pure-nlev | 0.1696 | **+105.08 %** | 4.80 | 1.67 |

The pattern replicates on a production mesh five times CORE2's size and at four times the rank
count: the scalar-weighted arm wins, the cut-optimised unweighted arm **loses** 2.2 %, and the
over-balanced arm doubles the step time.

### The law, stated across four raced points

| point | best arm | gain | the arm that only improved the CUT |
|---|---|--:|--:|
| CORE2 512 | `w=100+nlev`, 3 % | −4.2 % | ±0.00 % |
| CORE2 864 | `w=100+nlev`, 3 % | −4.1 % | −0.23 % |
| fArc 2048 | `w=100+nlev`, 3 % | **−6.8 %** | +2.18 % |
| CORE2 512, + Hilbert | both levers | **−5.5 %** | — |

**Communication quality is worth nothing; 3-D load balance is worth 4–7 %, provided the 2-D
imbalance stays near 1.2–1.3.** Every arm that pushed 3-D balance further by sacrificing 2-D
balance lost catastrophically (+53 %, +71 %, +105 %).

## ⭐⭐⭐ Finding 24 — the partition lever is CPU/high-rank only, and the predictor is the SHIPPED 3-D imbalance

GPU races (CUDA `h17`, `-C a100_80`, `FESOM_SPEED=1`, min-of-3, fresh allocations) plus same-day
replications of the CPU points in fresh allocations. Every number below is min-of-3.

**GPU, CORE2, 1 node / 4 ranks** (job 26858541) — the sensible CORE2 GPU point:

| arm | s/step | vs base | spread |
|---|--:|--:|--:|
| base | 0.0664 | — | 2.3 % |
| **hil (renumbered)** | 0.0631 | **−4.97 %** | 0.6 % |
| kahip `w=100+nlev` | 0.0684 | **+3.01 %** | 1.0 % |
| hil + kahip | 0.0640 | −3.61 % | 0.9 % |
| METIS `w=100+nlev` u30 | 0.0669 | +0.75 % | 0.9 % |

**GPU, fArc, 4 nodes / 16 ranks** (job 26858542):

| arm | s/step | vs base |
|---|--:|--:|
| base | 0.1161 | — |
| mtkahypar `w=100+nlev` | 0.1183 | **+1.89 %** |
| mtkahypar pure-nlev | 0.1640 | +41.26 % |

**The partition lever LOSES at both GPU points**, while the ordering lever gives −4.97 % — and the
day-1 GPU renumbering number (−4.89 %) replicates to within 0.08 pp on fresh nodes.

### The predictor, and it was in the Task-2 baseline table all along

| point | shipped partition's 3-D max/min | best partition arm |
|---|--:|--:|
| CORE2 GPU 4 r | **1.01** | +0.75 % |
| fArc GPU 16 r | **1.02** | +1.89 % |
| CORE2 CPU 512 r | 9.26 | −3.6 … −4.2 % |
| CORE2 CPU 864 r | 9.60 | −4.1 % |
| fArc CPU 2048 r | 9.40 | **−6.8 … −7.6 %** |

**The partition lever pays exactly what the shipped partition wastes.** At 4 or 16 ranks the
existing decomposition is already balanced in 3-D to 1 %, so there is nothing to recover and the
extra 2-D imbalance a weighted arm introduces is a pure cost. Bathymetry-driven imbalance grows
with rank count (CORE2: 1.01 at 4 ranks → 9.26 at 512 → 9.60 at 864), and so does the gain.

⇒ **Rule for adoption: score the shipped partition first. 3-D max/min near 1 ⇒ do not
repartition. Near 9–10 ⇒ repartition, and expect 4–8 %.**

## Same-day replication in fresh allocations (min-of-3)

The two-day adoption rule is replaced, at the user's direction, by replication in a fresh
allocation on the same day — which controls for node-specific effects but not for slow system
drift, and that limitation is stated rather than papered over.

| point | arm | first run | replication | agree to |
|---|---|--:|--:|--:|
| CORE2 CPU 512 | hil | −1.85 % | −1.19 % | 0.7 pp |
| | kahip `w=100+nlev` | −4.03 % | −3.57 % | 0.5 pp |
| | **METIS `w=100+nlev` u30** | −4.21 % | **−3.57 %** | matches KaHIP to the digit in both runs |
| | **hil + kahip (both levers)** | −5.54 % | **−5.09 %** | 0.5 pp |
| fArc CPU 2048 | mtkahypar `w=100+nlev` | −6.77 % | **−7.58 %** | 0.8 pp |
| CORE2 GPU 4 r | hil | −4.89 % (day 1) | −4.97 % | 0.08 pp |

Every sign reproduces and every magnitude agrees within 0.8 pp.

---

## ⭐⭐⭐ Finding 25 — the GPU imbalance is REAL on large meshes, and it is the HALO, not the load

Measured with the frozen PHASESTATS build (`m7/bin/phst1`, commit `0ec70cf`) — a different
binary from the certified `h17`, so its absolute step times are diagnostics only. 100 steps,
`FESOM_SPEED=1`, `-C a100_80`. ⚠️ The first CPU attempt printed
`FESOM_SPEED_PHASESTATS WAS REQUESTED BUT RESOLVES TO OFF` — the Serial backend needs
`FESOM_SPEED_FORCE_SERIAL=1`. The binary refused to pretend, which is the L80 rule working.

| point | busy max/min | owned 2-D max/min | owned 3-D max/min | halo max/min | corr(busy, 3-D) | corr(busy, halo) |
|---|--:|--:|--:|--:|--:|--:|
| CORE2, 4 GPUs | **1.03** | 1.00 | 1.01 | — | — | — |
| fArc, 16 GPUs | **1.30** | 1.00 | 1.02 | **2.89** | 0.18 | **0.47** |
| NG5, 64 GPUs | **1.27** | 1.01 | 1.01 | **2.70** | −0.22 | **0.59** |
| CORE2, 512 CPU ranks | **3.51** | 1.01 | **9.26** | 15.0 | **0.91** | 0.79 |

**Two different machines.** On CPU at high rank counts the busy time tracks owned 3-D work at
r = 0.91 — M10's bathymetry result reproduced (they measured 0.967 at 864 ranks) — and a depth
weight removes part of it, which is the 4–8 % this campaign raced. On GPU the owned work is
already balanced to 1–2 % in **both** currencies at every point measured, and the busy time still
spreads 27–30 %; the only per-rank quantity that tracks it is the **halo**, which spreads 2.7–2.9×.

That is why every weighted partition arm lost on GPU: it optimised a currency that was already
balanced, and paid in 2-D imbalance.

### The ice hypothesis, killed a second time — now by direct measurement

M10 rejected it on CPU by correlation (ocean busy vs 3-D nodes r = 0.967, vs 2-D nodes r = 0.003).
The GPU probe kills it directly. CORE2 on 4 GPUs, ice-covered nodes taken from the model's own
January `a_ice`:

| rank | ice-covered nodes | ice-dyn busy |
|---|--:|--:|
| 2 | 5,038 | 9.9 ms |
| 3 | **17,356** | **10.3 ms** |

**3.45× the ice, 4 % more time.** Fitting the four ranks, the ice-dependent part of the ice
dynamics kernel is 0.16–0.56 ms of ~10 ms; the other ~95 % is a fixed cost, because the kernels
run over all 2-D nodes whether or not there is ice on them. Perfectly balancing the ice would
save ~0.2 ms of a 73 ms step: **0.3 %**. An ice-aware vertex weight is built
(`scripts/m11_ice_weight.py`, mask from a model run rather than a latitude cut) and is **not
worth generating partitions for** on this evidence.

➕ The same measurement is a lead for someone else: the ice phases are **11.9 ms of 60.4 ms busy,
~20 % of the CORE2 GPU step**, and ~95 % of that is spent on ice-free water (29 % of CORE2's
nodes carry ice in January). Masking or compacting the ice kernels to ice-covered nodes is a
code lever, outside M11, and it also explains M9's result that the sea ice does not strong-scale
on GPU — a fixed per-rank cost cannot.

### Can the halo be balanced? The target is the MAX, and nothing we have hits it

| arm | mean halo | **max halo** | max/mean |
|---|--:|--:|--:|
| fArc shipped, 16 GPUs | 609 | **876** | 1.44 |
| fArc mtkahypar `w=100+nlev` | 439 (−28 %) | 810 (−7.5 %) | 1.85 |
| fArc mtkahypar pure-nlev | 406 (−33 %) | 691 (−21 %) | 1.70 |
| NG5 shipped, 64 GPUs | 1,347 | **1,922** | 1.43 |

Every engine arm cuts the **mean** halo substantially and leaves the **max** nearly alone, because
partitioners minimise a SUM (total cut, total comm volume) and the GPU step waits for the
slowest rank. That mismatch of objective is the finding.

Two routes exist and both are now buildable:

1. **Compensate** — `scripts/m11_tpwgts.py`: give the ranks with expensive halos a smaller share
   of owned work via `FESOM_PART_TPWGTS_FILE`, iterate to a fixed point. First-order estimate on
   fArc/16: −0.9 % of max cost; on the NG5 fit, −10.5 % of max busy. **Those two disagree by an
   order of magnitude and the cost model's R² is 0.28–0.35**, with the owned-3-D regressor nearly
   collinear with the intercept because owned work barely varies. The prize is not established,
   and the tool prints that warning itself rather than reporting a number.
2. **Reduce the max directly** — `MINCONN` + `CONTIG` (METIS Kway) shape compact parts, which is
   what a 1.44 max/mean says is missing. Generation queued (arm `a4` at fArc 16).

⚠️ Not measured: dars at 32 and 64 GPUs died with `task Killed` on both attempts while NG5 —
five times larger — ran fine at 64. That is unexplained and recorded as such.

## ⭐⭐ Finding 26 — the max halo is NOT reachable by any partitioner knob we have

Three attempts, all measured on fArc at 16 parts (the GPU point where the halo spreads 2.89× and
the busy time 1.30×). Target: the **maximum** halo per rank, 876 nodes on the shipped partition,
because the GPU step waits for the slowest rank.

| attempt | mean halo | **max halo** | verdict |
|---|--:|--:|---|
| shipped (reference) | 609 | **876** | — |
| `w=100+nlev`, 3 % slack (the CPU winner) | 462 | 887 | mean −24 %, max +1 % |
| Mt-KaHyPar `w=100+nlev` | 439 | 810 | mean −28 %, max −7.5 % |
| **`MINCONN=1 CONTIG=1`** | 543 | **891** | max **+2 %**, and 3-D balance 1.02 → 4.56 |
| **tpwgts loop, from the CPU winner** | 462→394 | 887 → 835 → 954 → 919 → 941 | **diverges** |
| **tpwgts loop, from a balanced start, damping 0.25** | 609→596 | 876 → 885 → 923 → **1033** → 957 → 994 | **monotonically worse** |

**Every route makes the maximum worse or leaves it alone**, while several cut the mean by a
quarter. The reason is structural and it is the same one Finding 25 named: partitioners minimise
a **sum** (total cut, total communication volume) and offer no objective on the **maximum**
boundary. `MINCONN` minimises the maximum number of *neighbouring subdomains*, which is a
different quantity and does not follow.

The compensation route deserves its own note because it failed for two different reasons and I
got the first one wrong. Started from the CPU winner, the cost model sees that arm's deliberate
owned-work imbalance (3-D max/min 4–7) as the thing to correct and drives the target shares to
0.016…0.216 against a uniform 0.0625 — it was correcting the wrong imbalance. Restarted from a
balanced partition with a quarter of the damping, so that the halo is the only imbalance left,
it still degrades monotonically: **an unequal target share does not make a subdomain rounder**,
it just moves the boundary, and unequal shares make shapes more irregular in general. The
hypothesis that shrinking a part shrinks its boundary as √area is true for a disc and false for
what METIS produces under a share constraint.

⇒ **Within the space of partitioner weights and constraints, the GPU max-halo imbalance is not
solvable.** Two routes remain, both outside that space:

1. a custom **min-max boundary refinement** — a local pass that moves nodes to shrink the worst
   part's boundary, which is an objective no library here exposes;
2. attack the **cost** instead of the balance — the ice kernels are ~20 % of the CORE2 GPU step
   with ~95 % of that spent on ice-free water (Finding 25), which is a much larger and much
   better-understood target than the ~5 % the halo imbalance is worth.

Recommendation: stop spending GPU node-hours on repartitioning. The lever is CPU-side, it is
worth 4–8 % there, and it is available from METIS alone.

## ⭐⭐⭐ Finding 27 — MY PREDICTION WAS WRONG: on GPU the currency is MESSAGE COUNT, not volume

I predicted `MINCONN=1 CONTIG=1` would lose on GPU, from its offline profile: worse max halo
(891 vs 876), worse 3-D balance (4.56 vs 1.02), worse 2-D balance (1.274 vs 1.003). Raced at
fArc 16 GPUs, min-of-3 (job 26868366):

| arm | s/step | vs base | spread | comm volume | **max/rank** | **neighbours/rank** | max halo |
|---|--:|--:|--:|--:|--:|--:|--:|
| base (shipped) | 0.1179 | — | 2.2 % | 228,972 | 20,506 | **5.25** | 876 |
| **MINCONN+CONTIG** | **0.1145** | **−2.88 %** | 0.4 % | 203,572 | 25,815 (**+26 %**) | **3.62 (−31 %)** | 891 |
| mtkahypar `w=100+nlev` | 0.1189 | +0.85 % | 0.3 % | **164,751 (−28 %)** | **18,537** | 4.00 | **810** |

**The arm with the worst max-per-rank volume and the worst max halo won; the arm with the best
of both won nothing.** The only metric that orders these three correctly is the **number of
communication partners per rank** — 3.62 against 5.25 and 4.00.

That is a coherent mechanism rather than a coincidence: a GPU rank pays a fixed cost per
*message* (launch, latency, and on this fabric a staging copy), and the step carries 120 MPI
calls in ice dynamics and 640 in the solver. Cutting the partner count by 31 % removes messages;
cutting the volume by 28 % removes bytes, which are not what is scarce.

It also retires my own Finding-26 conclusion in the direction that matters: I wrote that "within
the space of partitioner weights and constraints the GPU imbalance is not solvable" on the
strength of a halo model. The halo model was the wrong model. `MINCONN` — a knob that has never
been active in any FESOM partition ever generated, because `PartGraphRecursive` silently ignores
it — is worth **−2.9 %** on GPU.

### Two different machines want two different partitioners

| | CPU (512–2048 ranks) | GPU (16 ranks) |
|---|---|---|
| what is imbalanced | owned 3-D work, 9.3–9.6× | nothing in the owned work (1.01×) |
| what predicts the win | 3-D load balance | **communication partner count** |
| the setting | `WGT_A=100 UFACTOR=30` | `MINCONN=1 CONTIG=1` |
| measured | −4.1 … −7.6 % | **−2.9 %** |
| the arm that loses | unweighted (best cut) | `w=100+nlev` (best volume) |

This answers the earlier question — *do CPU and GPU want different METIS settings?* — with a
measured **yes**, and the two settings are almost disjoint.

⚠️ One point, one day, min-of-3 with a 0.4 % rep spread against a 2.88 % delta. `MINCONN` and
`CONTIG` are not yet separated (jobs 26869443/26869445 generate each alone, plus the pair with
`UFACTOR=30`, at fArc 16 and CORE2 4/512), and nothing is replicated. It is a lead with one
measurement behind it, not an adoption.

## ⭐⭐⭐ Finding 28 — the knobs separate: slack is the CPU lever, MINCONN is not

CORE2 512 CPU, min-of-3, all five arms in one allocation (job 26872968):

| arm | knobs on top of Kway+VOL+`w=100+nlev` | s/step | vs base | spread |
|---|---|--:|--:|--:|
| base | shipped (dual, Recursive, `UFACTOR=1`) | 0.0591 | — | 2.5 % |
| **`a5_u30`** | **`UFACTOR=30`** | **0.0569** | **−3.72 %** | 0.4 % |
| `a4u30` | `MINCONN=1 CONTIG=1 UFACTOR=30` | 0.0572 | −3.21 % | 0.5 % |
| `a4` | `MINCONN=1 CONTIG=1` (UFACTOR=1) | 0.0590 | −0.17 % | 0.5 % |
| `a4m` | `MINCONN=1` (UFACTOR=1) | 0.0591 | **+0.00 %** | 0.2 % |

**On CPU, `MINCONN`/`CONTIG` are worth nothing** (+0.00 % and −0.17 %, both inside their rep
spreads) and they **cost 0.5 pp when added to the winner** (−3.21 % against −3.72 %). The entire
CPU gain comes from the balance tolerance.

That is the exact mirror of the GPU result, where `MINCONN`+`CONTIG` won −2.88 % and the
tolerance arm won nothing. The two backends are not just tuned differently — **each backend's
lever is the other backend's null.**

CPU recommendation, now measured three times in three separate allocations
(−4.21 %, −3.57 %, −3.72 % at CORE2 512):

```
FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=100 FESOM_PART_UFACTOR=30
```

and **do not add `MINCONN` or `CONTIG` on CPU**.

## ⭐⭐⭐ Finding 29 — the full matrix: the best setting is MESH- AND RANK-DEPENDENT, and two of my earlier claims were wrong

Five points raced with the same arm family, min-of-3, each point in one allocation. Deltas
against that point's shipped partition; **bold = best at that point**.

| arm (all Kway + OBJ=vol + `w=100+nlev`) | CORE2 4 GPU | fArc 16 GPU | CORE2 512 CPU | CORE2 864 CPU | fArc 2048 CPU |
|---|--:|--:|--:|--:|--:|
| `UFACTOR=30` (slack only) | +0.59 % | +1.9 % | **−3.72 %** | −0.23 % | −1.33 % |
| `MINCONN` only | **−7.51 %** | −2.30 % | +0.00 % | −0.68 % | **−4.84 %** |
| `CONTIG` only | — | −1.96 % | — | — | — |
| `MINCONN`+`CONTIG` | −1.18 % | **−2.47 %** | −0.17 % | — | — |
| `MINCONN`+`CONTIG`+`UFACTOR=30` | **−7.66 %** | +0.34 % | −3.21 % | **−2.71 %** | −1.93 % |
| external engine, `w=100+nlev` 3 % | +0.85 % | +0.85 % | −3.57 % | −4.07 % | **−7.26 %** |

Rep spreads are 0.2–1.4 % against deltas of 2–8 %, so the ordering is real. **There is no single
recipe**: the best arm differs at every one of the five points.

### Two corrections to what this log said earlier

1. **"On CPU, `MINCONN` is worth nothing" (Finding 28) is false in general.** It is true at
   CORE2 512 (+0.00 %) and false at fArc 2048, where `MINCONN` alone is **−4.84 %** — the second
   largest CPU gain in the campaign. Finding 28 measured one point and generalised.
2. **"METIS alone gets the whole win" (Finding 21) holds only at CORE2 512.** There METIS
   (−3.72 %) matches the engine (−3.57 %) to within a rep spread. At CORE2 864 the engine gives
   −4.07 % against METIS's best −2.71 %, and at fArc 2048 the engine gives **−7.26 %** against
   METIS's best −4.84 %. The external engines are worth 1.4–2.4 pp at the large points.

### What does survive: the partner count explains the GPU, and it does so exactly

CORE2 on 4 GPUs, where the shipped partition's owned work is balanced to 1 %:

| arm | neighbours/rank | 3-D max/min | result |
|---|--:|--:|--:|
| base | **3.00** | 1.01 | — |
| `MINCONN`+`CONTIG` | 2.50 | 1.46 | −1.18 % |
| `MINCONN` | **2.00** | 1.41 | **−7.51 %** |
| `MINCONN`+`CONTIG`+u30 | **2.00** | 1.67 | **−7.66 %** |
| `UFACTOR=30` | 2.50 | 1.40 | +0.59 % |

Partner count orders the table perfectly and nothing else does — the winning arms have the
**worst** 3-D balance (1.41, 1.67 against the baseline's 1.01). Going from 3 partners to 2 at
four ranks removes a third of the messages and buys 7.5 %.

⚠️ `CONTIG` is not free: at CORE2/4 it costs 6 pp on top of `MINCONN` (−1.18 % against −7.51 %)
because it pushes the partner count back up to 2.50, and only the extra slack recovers it. At
fArc/16 it is worth −1.96 % on its own. It should never be set without measuring.

### Practical recommendation, as measured

| where | setting | gain |
|---|---|--:|
| **GPU, few ranks** (CORE2 1 node) | `KWAY=1 OBJ=vol VSIZE=1 WGT_A=100 MINCONN=1` | **−7.5 %** |
| **GPU, more ranks** (fArc 4 nodes) | add `CONTIG=1` | −2.5 % |
| **CPU, moderate ranks** (CORE2 512) | `… WGT_A=100 UFACTOR=30`, no MINCONN | **−3.7 %** |
| **CPU, many ranks** (fArc 2048) | `… WGT_A=100 MINCONN=1`, or an external engine | −4.8 % / **−7.3 %** |

`MINCONN` helps at four of the five points and has **never been active in any FESOM partition
ever generated**, because `PartGraphRecursive` silently ignores it.

## ⭐⭐ Finding 30 — dars and NG5: `MINCONN` wins again, and a protocol error of mine nearly became a partition finding

**dars, 2048 CPU ranks, dt 240, 150 steps, min-of-2** (job 26884449):

| arm | s/step | vs base | spread |
|---|--:|--:|--:|
| base (shipped) | 0.4190 | — | 0.1 % |
| **`MINCONN`** | **0.4019** | **−4.08 %** | 0.1 % |
| `UFACTOR=30` | **FAILED** | — | — |
| `MINCONN`+`CONTIG`+`UFACTOR=30` | **FAILED** | — | — |
| KaMinPar `w=100+nlev` 3 % | **FAILED** | — | — |

Three of five arms died with `CG_kk residual diverged` / `pp·App is nan`, **reproducibly in both
reps**, and on NG5 the `MINCONN` arm died the same way. That looked like a partition property,
and the pre-registration says failing arms are discarded rather than debugged — so it would have
been recorded as "three of four candidate partitions are unusable on dars".

**It is far more likely to be my timestep.** I ran both meshes at their *production* dt (240),
but the project's cold-start ladder is **dars dt 120 and NG5 dt 180**, and every race here is a
cold start. The scorecards support that reading and give no other candidate: the failing and
surviving arms are indistinguishable on every metric —

| arm | status | 2-D max/mean | 3-D max/min | disconnected | isolated | min part (2-D nodes) |
|---|---|--:|--:|--:|--:|--:|
| base | works | 1.0006 | 11.22 | 20 | 0 | 1,542 |
| `a5_u30` | FAILS | 1.3324 | 7.99 | 59 | 0 | 1,328 |
| `a4m` | works | 1.3142 | 7.56 | 65 | **6** | 1,366 |
| `a4u30` | FAILS | 1.3544 | 8.99 | **0** | 5 | 1,210 |

`a4u30` is **perfectly contiguous** and fails; `a4m` has 65 disconnected parts, 6 isolated nodes
and works. No arm has a degenerate rank. Nothing in the partition explains it, and Finding 11
supplies the mechanism that does: any decomposition change lands on a different admissible
iterate of the SSH solve, and from a marginally-stable cold start the trajectories separate.

Both meshes are re-running at the ladder dts. **Prediction, recorded before they land: the
failures largely disappear at dt 120 / 180.** If they do not, the arms are genuinely unusable and
that is the finding instead — but the cheap explanation has to be excluded first, and this is
rule 0.41 in its original form: *a stability verdict is only as long as its measurement window*.

Either way `MINCONN` at `UFACTOR=1` now wins at a fourth mesh (−4.08 % on dars), and it is the
only arm that survived a protocol I got wrong — which is a weak form of evidence, but it points
the same way as the other four points.

## ⭐⭐⭐ Finding 31 — dars confirms the prediction; NG5 splits the lever open

### dars: the failures WERE my timestep, exactly as predicted

Re-run at the cold-start ladder dt 120 instead of the production dt 240 (job 26885210), 2048 CPU
ranks, min-of-2:

| arm | s/step | vs base | at dt 240 |
|---|--:|--:|---|
| base | 0.4148 | — | works |
| **`MINCONN`** | **0.3960** | **−4.53 %** | works, −4.08 % |
| `UFACTOR=30` | 0.3995 | −3.69 % | **FAILED** |
| `MINCONN`+`CONTIG`+u30 | 0.3999 | −3.59 % | **FAILED** |
| KaMinPar `w=100+nlev` | 0.4000 | −3.57 % | **FAILED** |

**Every arm that died at dt 240 runs clean at dt 120 and wins 3.6–4.5 %.** The prediction was
recorded before the job landed and it held: three "unusable partitions" were a protocol error of
mine, not a property of the decompositions. Rep spreads 0.0–0.2 %.

⇒ A partition arm must be screened at the mesh's **cold-start** ladder dt, not its production dt.
A campaign that screens at the wrong dt will discard good partitions and call it a finding.

### NG5: the same arm gives the campaign's largest win AND an unusable partition

| point | `MINCONN` |
|---|--:|
| **NG5, 64 GPUs** | **−9.96 %** (base 0.2511 → 0.2261, spreads 0.1–0.2 %) |
| NG5, 2048 CPU ranks | **FAILS** — `CG_kk abort at iter 1: pp·App = nan (s_old=nan)` |

Same mesh, same knob, same partitioner. At 64 ranks it is the best result in the campaign; at
2048 it produces a partition the model cannot start on. The failure is **not** a slow divergence:
`s_old` is the initial ‖rhs‖² and it is already NaN at the **first** solve of the **first** step.
The halo identity gate passes immediately before it, so the decomposition transports correctly.

Nothing in the scorecard predicts it:

| NG5 2048 | 2-D max/mean | 3-D max/min | disconnected | off-lobe | isolated |
|---|--:|--:|--:|--:|--:|
| base (works) | 1.008 | 1.04 | **1,720** | **1,182,621** | 3 |
| `a4m` (FAILS) | 1.485 | 8.45 | 45 | 54,683 | **0** |

The failing arm is the *cleaner* partition on every fragmentation metric — 45 disconnected parts
against the baseline's 1,720, and zero isolated nodes against three.

⇒ **Recorded as open.** What it means for adoption is not open: **no partition may be shipped
without a short smoke run at the target rank count**, because neither the scorecard nor the halo
gate catches this class of failure. `jobs/m11_zoo_b_dists.sh` already carries that smoke leg; it
now becomes mandatory rather than a convenience.

### Where the matrix stands after dars and NG5

| point | best arm | gain |
|---|---|--:|
| CORE2 4 GPU | `MINCONN` | −7.5 % |
| **NG5 64 GPU** | `MINCONN` | **−9.96 %** |
| fArc 16 GPU | `MINCONN`+`CONTIG` | −2.5 % |
| CORE2 512 CPU | `UFACTOR=30` | −3.7 % |
| CORE2 864 CPU | `MINCONN`+`CONTIG`+u30 | −2.7 % |
| fArc 2048 CPU | external engine (`MINCONN` −4.8 %) | −7.3 % |
| **dars 2048 CPU** | `MINCONN` | **−4.53 %** |
| NG5 2048 CPU | none — `MINCONN` unusable, slack +0.87 % | — |

`MINCONN` is the best arm at five of eight points and within a point of best at two more. It
fails at exactly one, and that failure is not predicted by any offline metric.

---

## ⭐⭐⭐ Finding 32 — the gates: every winner survives protocol length, and the gains do not decay

Everything up to here measured **speed only**. Nothing in the partition family had been through
the accuracy and stability gates the campaign pre-registered. This closes that gap at three
points, one per (mesh, backend) combination we intend to recommend.

### The stability re-proof (rule 0.41): 3,000 steps, every arm, no failures

| point | job | dt | arms | outcome |
|---|---|--:|---|---|
| CORE2 512 CPU | 26886214 | 1800 | base, `MINCONN`, slack, renumber+engine | 4/4 rc=0 |
| CORE2 4 GPU | 26892875 | 1800 | base, `MINCONN`, `MINCONN`+`CONTIG`(+u30) | 4/4 rc=0 |
| fArc 2048 CPU | 26892880 | 900 | base, `MINCONN`, Mt-KaHyPar | 3/3 rc=0 |

Eleven runs of 3,000 steps, zero NaN/blow-up/divergence lines, and at step 3,000 the arms agree
with their baseline on the field ranges to the printed digits (CORE2 CPU `T[-2.03,31.48]`
`S[4.45,41.08]` for base and both partition arms; the renumbered arm differs in the 3rd decimal,
which is what a different mesh numbering is expected to do).

### The gains do not decay at length — they grow slightly

This is the part worth quoting. Every headline in this campaign was earned on 150–300-step
min-of-3 races. Re-measured over 3,000 steps in a **fresh allocation on a different day**:

| point / lever | short race | 3,000-step re-proof | drift |
|---|--:|--:|--:|
| CORE2 512 CPU, slack `UFACTOR=30` | −3.7 % | **−3.83 %** | −0.1 pp |
| CORE2 512 CPU, renumber + engine | −5.54 % | **−5.83 %** | −0.3 pp |
| CORE2 4 GPU, `MINCONN` | −7.5 % | **−8.06 %** | −0.6 pp |
| fArc 2048 CPU, `MINCONN` | −4.8 % | **−5.49 %** | −0.7 pp |
| fArc 2048 CPU, Mt-KaHyPar | −7.3 / −7.58 % | **−7.52 %** | ±0.2 pp |

Five independent points, every one reproducing within 0.7 pp and every one in the same direction.
The short-race protocol is therefore not flattering the lever, and if anything it *under*-reports
it — plausibly because the fixed start-up cost is amortised over 10–20× more steps.

### Accuracy: the arms sit inside the repartitioning class

The yardstick is the project's standing one (L79): an arm is in class if it sits inside the
spread of ordinary repartitioning CONTROLS of the same mesh. 20 steps, printing on.

**CORE2 4 GPU (job 26892982), rms vs base:**

| leg | temp | salt | ssh |
|---|--:|--:|--:|
| `MINCONN` (arm) | 2.92e−2 | 3.32e−2 | 4.02e−3 |
| `MINCONN`+`CONTIG`+u30 (arm) | 2.31e−2 | 3.15e−2 | 4.05e−3 |
| *control: seed change only* | 3.99e−2 | 3.57e−2 | 4.34e−3 |
| *control: `MINCONN`+`CONTIG`* | 3.45e−2 | 3.20e−2 | 7.24e−3 |

Both arms are **below both controls on all three variables**. PASS.

**CORE2 512 CPU (job 26885353)** was already recorded: arms 5.78e−2 / 5.41e−2 temp against
controls 4.82e−2 / 5.56e−2, ssh *below* both controls. PASS.

**fArc 2048 CPU (job 26892879):** `MINCONN` in class on all three (4.19e−2 / 4.13e−2 / 3.94e−3
against a control at 4.58e−2 / 4.44e−2 / 5.45e−3). The Mt-KaHyPar arm is in class on temp and
ssh but its **salt rms is 8.91e−2, twice the control's 4.44e−2** — re-gated against a second,
seed-only control (job 26893194) before that engine is recommended for fArc.

🔴 **My first GPU gate used a control that was not one.** `core2_m11` (shipped) vs `core2_base`
(settled) at 4 ranks differ by rms temp 5.9e−7 — at that rank count the two partitions coincide,
so the "control" measured round-off and would have made any arm look out of class. A control has
to be *demonstrated* to be a different partition, not assumed to be. Fixed by generating a
seed-only re-roll (`core2_seed/dist_4`, job 26892943) and re-running.

### Side finding: the SSH iteration count is partition-dependent, and that is most of the spread

At CORE2 4 GPU the `MINCONN` and slack arms take **exactly** the baseline's CG iteration count at
every one of 20 steps, while the `MINCONN`+`CONTIG` control takes **4.65 fewer on average**
(max 6) — and it is also the leg with the largest ssh rms (7.24e−3 against ~4e−3 for everything
else). The solver stops on a *relative* residual, so a partition that converges faster in the
norm stops sooner and lands further from the reference. The accuracy spread of a partition arm is
therefore partly a solver-tolerance effect and not a discretisation effect, which is another
reason the control spread — not a guessed bound — has to be the yardstick.

---

## 🔴 Finding 33 — CORRECTS Finding 31: the NG5 partition is not un-startable, and the same dt error hit the GPU races

### The NG5 `MINCONN` partition starts fine

A 5-step smoke of all four NG5 arms at 2048 ranks, dt 180 (job 26892920): **4/4 rc=0**, including
the `a4m` and `a4u30` arms that abort in the 150-step race.

So my Finding 31 reading was wrong. `CG_kk abort at iter 1` names the first CG *iteration* of
whichever step first carries a NaN, not the first step of the run. The race log confirms it in
hindsight: with printing suppressed, step 1 printed **normal** values (`uv=8.92e-02
eta=1.87e-01 T[-2.08,30.17]`) and the run died before the next print at step 150.

⇒ NG5/`MINCONN` at 2048 ranks is a **run that diverges somewhere in steps 6–150**, not a
decomposition the model cannot set up. Job 26893204 traces it step by step, alongside a
second-seed re-roll of the same knobs (`a4m_seedb`, job 26892932) to tell a systematic property
of `MINCONN` from one pathological part.

**The adoption consequence changes with it.** "Smoke it at the target rank count" is necessary
and **not sufficient** — a 5-step smoke passes this partition and so would a 20-step gate. The
bar has to be the protocol-length stability screen. `m11_promote` (below) therefore requires a
`steps=` field of at least 3,000 in the evidence line, not merely a passing run.

### The same protocol error had also invalidated both GPU 64-rank races

| job | mesh | dt used | ladder dt | outcome |
|---|---|--:|--:|---|
| 26884452 | dars, 64 GPU | 240 | **120** | **every arm died, including base** |
| 26884451 | NG5, 64 GPU | 240 | **180** | `MINCONN`+`CONTIG` arms died; base and `MINCONN` ran |

Both logs say `[fesom_port FATAL] CG_kk residual diverged` — the `srun: task N: Killed` lines
underneath are just the teardown after `MPI_Abort`, which is why this first looked like a machine
fault ("dars GPU keeps getting killed") rather than the third instance of one protocol error.
Re-running at the ladder dt: dars 26893037, NG5 26893204.

The NG5 GPU **−9.96 %** headline is a matched pair (base and arm both ran, same dt, same
allocation) so it stands as measured — but it was measured at dt 240 and is being re-confirmed at
the ladder dt 180.

⇒ Rule, third time of asking: **every M11 job must assert the mesh's ladder dt**, because a
partition campaign reads a blow-up as an arm result.

---

## Task 16 — `m11_promote`, the one sanctioned way a partition leaves the campaign

`scripts/m11_guards.sh` gains `m11_promote`, with selftest section [7] (7 cases, all pass):

```
m11_promote --src <sandbox_meshdir> --name core2_v2 --evidence <file> dist_512 dist_4
```

* creates a **new** directory under `mesh_m11_certified/` and refuses to overwrite an existing
  one, so a promotion can never redefine a mesh other runs already point at;
* builds under `.partial.$$` and renames last — an interrupted promotion cannot leave something
  that looks like a usable mesh directory (this was a real bug in the first version, caught by
  the selftest);
* refuses any `dist_N` whose evidence line lacks `run=<jobid> steps=>=3000 rc=0` **at N ranks**
  — Finding 33 made mechanical;
* requires the source to be a sandbox mesh with a verified md5 manifest;
* writes `MESH_PROVENANCE.md` (source, commit, evidence lines, source manifest) and its own
  manifest, and appends to `mesh_m11/.m11state/promotions.log`.

---

## ⭐⭐⭐ Finding 34 — the NG5 failure is ONE PARTITION, not the knob: a re-roll of the same seed fixes it

Job 26893204 traced all four NG5 arms at 2048 ranks, dt 180, **printing every step**. That turns
Finding 33's "it dies somewhere in steps 6–150" into a mechanism.

### It is a local velocity blow-up, and the barotropic solution does not notice

| step | base uv | `a4m` uv | `a4u30` uv | base eta | `a4m` eta | `a4u30` eta |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 6.69e−1 | 6.08e−1 | 6.78e−1 | 5.17e−1 | 5.42e−1 | 5.33e−1 |
| 20 | 1.07e+0 | 1.08e+0 | 1.04e+0 | 8.33e−1 | 8.41e−1 | 8.56e−1 |
| 30 | 1.24e+0 | 1.57e+0 | 1.51e+0 | 1.33e+0 | 1.35e+0 | 1.34e+0 |
| 40 | 1.23e+0 | 2.04e+0 | 1.96e+0 | 1.28e+0 | **1.28e+0** | **1.28e+0** |
| 50 | 1.59e+0 | 2.60e+0 | 2.33e+0 | 1.57e+0 | **1.57e+0** | **1.57e+0** |
| 60 | 1.77e+0 | 2.71e+0 | 2.45e+0 | 1.73e+0 | **1.73e+0** | **1.73e+0** |
| 63 | 1.76e+0 | — | **1.16e+2** | 1.75e+0 | — | 2.15e+2 |
| 71 | 1.66e+0 | **5.67e+1** | — | 1.90e+0 | 8.87e+1 | — |

The runaway starts around step 25 and takes ~40 steps to go superlinear. **Through step 60 the
sea-surface height is identical to the baseline to three significant figures** while max|u| is
already 1.5× larger. So the global solution is unchanged and a *local* velocity maximum is
exploding — this is not the SSH solver, not the barotropic mode, and not a bad decomposition of
the domain. Final states are `T[−21670, 16884]` and `T[−62144, 71609]`: an ordinary blow-up.

### A second METIS seed of the SAME knobs runs clean

`a4m_seedb` — identical arm definition, `FESOM_PART_SEED=424242` instead of the default — tracks
the baseline the whole way (step 60: uv 1.76 vs base 1.77; step 112: uv 1.55, `T[−2.07, 30.…]`).
It is genuinely a different partition (2,045 of 2,048 ranks have different owned counts) with
statistically identical macro-properties (owned max/min 1.61 for both) and a slightly **worse**
edgecut: 28,791,521 against `a4m`'s 28,694,317.

⇒ **`MINCONN` on NG5 is not poisoned. One roll of the dice produced an unusable partition.**

### No scorecard column separates the four arms

| NG5 2048 | 2-D imb | 3-D max/min | nbr max | disconnected | isolated | halo mean | commvol |
|---|--:|--:|--:|--:|--:|--:|--:|
| `a4m` **DIES** | 1.485 | 8.45 | 9 | 45 | **0** | 242.4 | 2.87e7 |
| `a4u30` **DIES** | 1.475 | **14.72** | 8 | 0 | 2 | 231 | 2.74e7 |
| `a5_u30` survives | 1.494 | 8.77 | 9 | 39 | 1 | 231.9 | 2.75e7 |
| `a4` (5-step only) | 1.488 | 8.55 | 8 | 0 | 184 | 250 | 2.94e7 |

Every column puts a dying arm on the *better* side of a surviving one: `a4m` dies with 3-D
imbalance 8.45 where `a5_u30` survives at 8.77, dies with **zero** isolated nodes where the
survivor has one, and dies with a lower edgecut than its own healthy re-roll.

### What this changes

1. **The knob survives.** `MINCONN` stays the campaign's headline lever — it is best or within a
   point of best at seven of nine measured points.
2. **The screen is mandatory and the remedy is cheap:** run the protocol-length stability screen
   at the target rank count; if a partition fails it, **re-roll the seed** and screen again. That
   is a two-line recipe, not a reason to drop the lever.
3. **Do not trust the scorecard as a safety check.** It is a design tool for ranking candidates.
   It has now failed, on four arms of one mesh, to identify a partition that destroys the run.

---

## ⭐⭐⭐ Finding 35 — dars on GPU: −18.6 %, the largest result in the campaign

Job 26893037, dars 64 GPU ranks (16 nodes), **dt 120 = the dars cold-start ladder dt**, min-of-2,
against the shipped partition. The three arms that "all died" in job 26884452 were dying of the
production dt, not of their decompositions (Finding 33):

| arm | s/step | vs base | rep spread |
|---|--:|--:|--:|
| base (shipped) | 0.1400 | — | 0.7 % |
| `MINCONN`+`CONTIG` | 0.1235 | −11.79 % | 0.2 % |
| `MINCONN` | 0.1209 | −13.64 % | 0.2 % |
| **`MINCONN`+`CONTIG`+`UFACTOR=30`** | **0.1139** | **−18.64 %** | **0.0 %** |

Every arm beats the shipped partition, and the best one by nearly a fifth of the step. This is
consistent with the mechanism established in Finding 27: on GPU the currency is the **number of
communication partners**, and `MINCONN` is the only METIS objective that targets it — a knob that
`PartGraphRecursive` silently ignores, so it has never been active in any FESOM partition ever
generated.

Note the ordering flips against CPU: at dars 2048 CPU the winner is `MINCONN` alone (−4.53 %) and
adding `CONTIG`+slack costs a point; on GPU adding them **gains** five points. The backend split
holds all the way to the largest mesh points measured.

## NG5 CPU 2048 is a null point for the partition lever

With the trace job complete (26893204), the full NG5 CPU picture at the ladder dt 180:

| arm | outcome |
|---|---|
| base | 1.2268 s/step |
| `UFACTOR=30` | +0.87 % |
| `MINCONN` re-rolled seed | +0.94 % (150 steps clean) |
| `MINCONN` | **BLOWUP at step 71** (`uv=5.674e+01`) |
| `MINCONN`+`CONTIG`+u30 | **BLOWUP at step 63** (`uv=1.161e+02`) |

No arm wins, one arm re-rolls into a healthy but pointless partition, and two blow up. NG5 at 2048
CPU ranks is where the lever has nothing to offer — recorded as a null, not as a gap.

---

## ⭐⭐ Finding 36 — the NG5 blow-up is a STABILITY-MARGIN failure: halving dt saves the same partition

Job 26893393, NG5 2048 CPU, the **same `a4m` partition that blew up at step 71**, run at dt 90 for
300 steps — the identical span of model time (27,000 s):

| | base | `a4m` |
|---|--:|--:|
| at dt 180, 150 steps | clean, uv 2.11 | **BLOWUP at step 71**, uv 5.67e+01 |
| at dt 90, 300 steps | clean, uv 2.12 | **clean**, uv 2.71 |

So the partition is not carrying a broken operator — the model integrates it happily with a
smaller timestep over exactly the interval in which it exploded. It is a **stability-margin**
failure: this decomposition sits close enough to the edge that the ladder dt tips it over.

The honest caveat: `a4m`'s max|u| stays ~28 % above the baseline's throughout the dt-90 run
(2.71 vs 2.12) while eta agrees to 3.5 %. A domain maximum is an extreme-value statistic and two
valid solutions of a chaotic system may differ that much in it, so this is suggestive of a local
hot spot rather than proof of one. What is not in doubt is the practical conclusion.

### The open question this raises is not about the partition

`base` at NG5/dt 180 has only ever been proven to **150 steps** in this campaign. If a
repartitioning can cross the stability edge there, the margin at that mesh × dt is thinner than
anyone has measured — which is a rule-0.41 question for **every** track that runs NG5, not only
M11. Job 26893909 runs the NG5 baseline for 3,000 steps at dt 180 to settle it.

### Consolidated NG5 remedy

Three independent routes now restore the arm: **re-roll the METIS seed** (Finding 34, free),
**halve the timestep** (this finding, costly), or **accept the null** (NG5 CPU has no winning arm
anyway). For the GPU point, where NG5/`MINCONN` is worth −9.96 %, the seed re-roll is the route
and job 26893649 re-races all arms at the ladder dt 180.

---

## 🔴 Correction — how the `MINCONN` claim must be worded upstream

Three places above say `MINCONN` "has never been active in any FESOM partition ever generated,
because `PartGraphRecursive` silently ignores it" (lines ~1865, ~1971, ~2301). The conclusion is
right; the word **silently** is not, and it would not survive review on the upstream PR.

Read `fort_part.c:326-355` verbatim:

* `MINCONN` is **never set at all** — not set-and-ignored. So "no stock-FESOM partition has ever
  had it active" is true, and true for a simpler reason than we wrote.
* `CONTIG` **is** set, to `0`, with the comment `/* Ignored by METIS_PartGraphRecursive */`.
* `OBJTYPE` is set to `CUT`, with the comment `/* But: _VOL only works with METIS_PartGraphKway*/`.
* The choice of Recursive is documented and reasoned: it "resulted in a far better partition than
  Kway" on `mesh_aguv` with `wgt_type=2`, and "there is no rule which one works best".

So the authors knew exactly which options their call gives up and wrote it down. What has changed
is the hardware: that comparison predates GPU backends, and the objective those options serve —
partner count — is the one that dominates on GPU (Finding 27).

⇒ **Upstream wording:** "`PartGraphRecursive` cannot use `OBJTYPE=VOL`, `CONTIG` or `MINCONN`;
the last is never set in the code. On GPU `MINCONN` is worth up to −18.6 %, so the Recursive-vs-Kway
comparison recorded in the comments is worth re-running on modern hardware." Not "silently ignores".

---

## ⭐⭐⭐ Finding 37 — the regression over ALL raced arms: the GPU claim reproduces at three points, the CPU claim does not survive as a scorecard rule

`scripts/m11_scorecard_regression.py` joins every protocol race (`m11_races.csv`) to its scorecard
row and computes Spearman ρ **within** each (mesh, backend, ranks) group of ≥3 arms — never
across groups, because a cross-point fit would be dominated by mesh size rather than by the
partition. Nine groups qualify, 88 per-group correlations.

ρ > 0 means a larger metric goes with a slower step.

| metric | CPU median ρ (5 groups) | CPU range | GPU median ρ (3 groups) | GPU range |
|---|--:|---|--:|---|
| **`nbr_max`** (partners) | −0.39 | [−0.60, +0.13] | **+0.87** | **[+0.87, +0.87]** |
| `nbr_mean` | +0.21 | [−0.15, +0.32] | **+0.74** | [+0.50, +0.87] |
| `n3d_maxmin` (owned work) | −0.19 | [−0.56, +0.40] | −0.44 | [−0.55, +0.50] |
| `edgecut_unweighted` | −0.16 | [−0.40, +0.67] | **−0.78** | [−0.99, +1.00] |
| `halo_nod_mean` | −0.16 | [−0.40, +0.67] | **−0.78** | [−0.99, +1.00] |
| `commvol_max_rank` | +0.15 | [−1.00, +0.67] | −0.72 | [−0.78, +1.00] |
| `elem_repl` | −0.15 | [−0.40, +0.67] | −0.78 | [−0.99, +1.00] |

### What holds

**On GPU the partner count is the only metric family with a consistent positive sign at every
point, and every volume-like metric has a consistent NEGATIVE one** — arms that ship *more* data
are *faster*. Finding 27 reported that inversion at a single point (fArc/16); it now reproduces
independently at CORE2/4, fArc/16 and dars/64.

🔴 **Do not quote the +0.87 as a constant.** It is identical at all three GPU points because
`nbr_max` is a small integer that takes only two or three distinct values per group, so ρ is
tie arithmetic, not a physical coefficient. At CORE2/4 it is a perfect *binary* split — every
`nbr_max=2` arm lands at −7.5…−8.1 % and every `nbr_max=3` arm at +3.0…−2.5 % — and it says
nothing about ordering *within* a level: at dars/64 the two `nbr_max=7` arms differ by 5 pp, and
at fArc/16 the five `nbr_max=6` arms span 3.2 pp. **Partner count is a threshold, not a ranking.**

### What does NOT hold, and a distinction I had been eliding

**No scorecard column orders the arms on CPU.** Every metric's range straddles zero across the
five CPU groups. In particular `n3d_maxmin` — the owned-3-D-work imbalance — has median ρ −0.19,
range [−0.56, +0.40]: an arm with better global 3-D balance is *not* a faster arm.

That is not a contradiction of Finding 24/29, but it does correct how I have been paraphrasing
them. The measured r = 0.91 was between **per-rank busy time and per-rank owned 3-D nodes inside a
single partition** — a statement about where the time goes during a run. It never implied that a
partition with a better *global* imbalance statistic runs faster, and the regression shows that it
does not. Both are true; only the first is evidence about how to choose a partition.

⇒ **The scorecard's role, stated exactly:** on GPU, use `nbr_max` to reject candidates above the
threshold, then race the survivors. On CPU it cannot rank arms at all — race them. This is the
third independent line of evidence (with Findings 18 and 34) that the scorecard is a design aid
and never a gate.
