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
