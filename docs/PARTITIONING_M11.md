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
