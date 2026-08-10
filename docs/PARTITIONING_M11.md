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
