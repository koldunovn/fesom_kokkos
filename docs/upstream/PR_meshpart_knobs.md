# PR B — meshpart: runtime partitioning knobs, Kway option family, isolated-node fix

**Branch:** `meshpart-runtime-knobs` (exists locally in
`/work/ab0995/a270088/port2/partm11/upstream_pr`, 2 commits on current `origin/main`
`962c3f0e`; compile-checked on Levante; not yet pushed to the fork).

**Title:** `meshpart: runtime FESOM_PART_* knobs; optional PartGraphKway so MINCONN/CONTIG/OBJTYPE=VOL reach METIS`

Independent of PR A (applies cleanly on main without it), but reviews best after it.

---

## Body (ready for `gh pr create --body-file`)

### What

Two commits, both confined to the standalone partitioner:

1. **`fort_part.c`: every partitioning choice becomes a runtime environment knob**, and
   `METIS_PartGraphKway` becomes available next to `PartGraphRecursive`. Unset environment ==
   today's behaviour, byte-for-byte on the dist files (we verified this explicitly). The knob
   family:

   | knob | meaning |
   |---|---|
   | `FESOM_PART_KWAY=1` | use `METIS_PartGraphKway` (default remains Recursive) |
   | `FESOM_PART_MINCONN=1` | minimise the maximum number of neighbouring subdomains |
   | `FESOM_PART_CONTIG=1` | force contiguous parts (checks the graph is connected first) |
   | `FESOM_PART_OBJ=cut\|vol` | edge-cut vs communication-volume objective |
   | `FESOM_PART_VSIZE=1` | pass `vsize = nlev` (comm-volume sizes) |
   | `FESOM_PART_WGT=0\|1\|2` · `FESOM_PART_WGT_A=<a>` | vertex weighting at runtime; `WGT_A` = single scalar weight `w = a + nlev` (avoids the multi-constraint refinement penalty) |
   | `FESOM_PART_UFACTOR/NCUTS/NITER/SEED` | the METIS tuning options, previously compile-time |
   | `FESOM_PART_GRAPH_DUMP=<path>` · `FESOM_PART_FILE=<path>` | dump the exact CSR handed to METIS / inject an externally computed part vector (validated hard) while the tool still writes the dist files — the interface to external partitioners (KaMinPar, Mt-KaHyPar, KaHIP…) |
   | `FESOM_PART_TPWGTS_FILE=<path>` | per-part target fractions |

   Every knob announces itself when set and aborts on an unrecognised value; requesting a
   Kway-only option under Recursive aborts rather than silently doing nothing.

2. **`fvom_init.F90`: `FESOM_PART_FIXISO=1`** repairs the isolated-node relocation in
   `check_partitioning`: the candidate list is seeded with `node_neighb_part(1)`
   unconditionally, so when the adjacency list happens to begin with the node's own partition,
   the node is "moved" to where it already is and the isolated node survives while the routine
   prints success — whether that happens depends on the mesh numbering. The seed also counts
   the first neighbour twice. With the knob set, the loop builds the whole list. Default OFF.

### Why

`METIS_PartGraphRecursive` cannot apply `OBJTYPE=VOL`, `CONTIG` or `MINCONN` — the METIS
manual documents them as `PartGraphKway`-only — and `MINCONN` is not set anywhere in
`fort_part.c`, so no partition produced by this tool has ever had it active. The choice of
Recursive was a documented, measured decision (the comment block in `fort_part.c` records
that it beat Kway on `mesh_aguv` with `wgt_type=2`, and rightly notes there is no universal
rule). What has changed since is the hardware: on GPU backends the number of communication
partners per rank — exactly what `MINCONN` minimises — dominates the halo-exchange cost,
because each neighbour costs a separate message/launch rather than bandwidth.

Measured on the Kokkos port of FESOM2 (A100, Levante), with 3,000-step stability screens and
accuracy gates against seed-control envelopes:

* CORE2, 4 GPUs: `MINCONN` partition −8.1 % wall time (certified; reproduces at −8.9 % on a
  GH200 node).
* Eddy-resolving meshes at 64 GPUs: −9.8 % (NG5) and −14.3 % (DARS) from `MINCONN` alone,
  accuracy review still open on these two points under our (deliberately strict) yardstick.
* CPU at scale, via `FESOM_PART_FILE` + external engines: −7.5 % (fArc, 2048 ranks,
  Mt-KaHyPar) and −4.2 % (DARS, 2048 ranks, KaMinPar), both certified. `UFACTOR=30` alone is
  worth −3.8 % at CORE2/512.

The gain on GPU tracks the fractional reduction of the maximum neighbour count; on CPU no
offline metric predicts the winner and candidates have to be raced.

### A warning that belongs in the PR text

Some repartitioned decompositions make the model diverge after hundreds to thousands of
steps while passing every short test — we caught 8 such partitions across recipes,
**including one produced by today's stock recipe with only a different METIS seed**, and no
offline quality metric separates them from healthy ones. Anyone regenerating a partition
(with or without these knobs) should run a few-thousand-step stability check at the target
rank count before adopting it, and re-roll `FESOM_PART_SEED` if it fails: the fragility is a
property of the individual partition, not of the options.

### Validation

* Null test: unset environment ⇒ dist files byte-identical to the unpatched tool.
* Injection null: injecting a partition's own vector ⇒ zero `check_partitioning` moves,
  `dist_N` byte-identical.
* Knob discipline: 26/26 announce/refuse cases (13 accept, 12 refuse, 1 short-vector refusal).
* `GRAPH_DUMP` CSR verified entry-for-entry against an independent graph exporter on two
  meshes (743,288 and 3,783,940 entries).
* Compile-checked on Levante (gcc/openmpi + bundled METIS 5.1.0) on top of `origin/main`.
* **Re-verified on this exact branch (2026-08-13, rebased on `962c3f0e`)**: stock build vs PR
  build with no knobs ⇒ CORE2 `dist_8` byte-identical (aggregate md5
  `c462881224f157d5cda22e7e75e485f6`); `KWAY=1 MINCONN=1` announces both and completes rc=0;
  `UFACTOR=xyz` refuses with a message, rc=1.

### Not in this PR

* **METIS 5.2.1**: we also built against 5.2.1 + current GKlib; at the historical settings it
  is measurably a wash on CORE2 (cut/volume within ±3 %, neighbour counts mixed), so the
  upgrade is maintenance, not performance, and is better as its own PR if wanted.
* **Changing any default**: Recursive with today's weighting remains the default; every knob
  is opt-in.

---

## Pre-push checklist

1. `cd /work/ab0995/a270088/port2/partm11/fesom2 && git push fork fix_meshpart_reshape_stack_overflow`
   (add `fork` remote first: `git remote add fork https://github.com/koldunovn/fesom2.git`)
2. `cd /work/ab0995/a270088/port2/partm11/upstream_pr && git push fork meshpart-runtime-knobs`
3. `gh pr create --repo FESOM/fesom2 --head koldunovn:fix_meshpart_reshape_stack_overflow --title "..." --body-file <PR A body>`
4. `gh pr create --repo FESOM/fesom2 --head koldunovn:meshpart-runtime-knobs --title "..." --body-file <PR B body>`
5. ~~Extra diligence: re-run the null on the rebased branch~~ **DONE 2026-08-13** — stock and
   PR builds at `962c3f0e`, no knobs, CORE2 `dist_8` byte-identical; announce/refuse spot
   checks pass (see Validation above). Test sandboxes: `mesh_m11/nullpr_{stock,knobs}`;
   builds in the session scratchpad (disposable).
