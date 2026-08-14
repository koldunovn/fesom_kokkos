# CG blow-ups on specific partitions — HANDOFF for the fix session (2026-08-14)

**You are a debugging session running IN THE MAIN CHECKOUT `~/port_kokkos`** (same folder as
the M12 session — coordination protocol at the end). Your memory index is the normal one
(no worktree gotcha). Task from Nikolay: **the stock CG SSH solver blows up on some mesh
partitions — solve it.** This document is everything the M12 campaign knows.

## 1. The phenomenon

The stock semi-implicit SSH solver (`fesom_ssh_solve_cg_kk` — the literal port of
solver.F90:98-281 CG with the diagonal-Jacobi-symmetrised preconditioner,
`fesom_ssh.cpp`/`fesom_ssh.h`) fails on SPECIFIC partitions of NG5 (and historically dars),
while on the same partitions, same binaries, same steps:
- the M12 split-explicit solver (`FESOM_SSH_MODE=se`, no CG at all) runs clean, and
- the M10 `oati` variant (communication-avoiding CG, different preconditioning path,
  binary `/work/ab0995/a270088/port2/m10/bin/stallknob_serial`) runs clean.

Failures are binary-independent (se0 bin AND m10 bin), ALE-independent (zstar AND linfs),
and reproducible (2 of 2 reps every time). All at NG5 ladder dt=180, wsplit=1, 300-step runs.

## 2. Failure inventory (2026-08-14, logs under /work/ab0995/a270088/port2/m12/)

| partition | ranks | outcome | failure mode | evidence |
|---|---|---|---|---|
| ng5 /pool dist_4096 | 4096 (32N) | **FAIL** | `[fesom_port FATAL] CG_kk residual diverged`, early steps (~2 min wall) | g4_ng532_siz{a,b}, g4_ng532_sil{a,b} (jobs 26944983/…87) |
| ng5 /pool dist_8192 | 8192 (64N) | **FAIL** | same | g4_ng564_siz*/sil* |
| ng5 bigpart dist_16384 | 16384 (128N) | works | 300 steps clean, 0.1814 s/step | g4_ng5128_siz* |
| ng5 bigpart dist_20480 | 20480 (160N) | **FAIL** | `CG_kk residual diverged` | g4_ng5160_siz{a,b} |
| ng5 bigpart dist_24576 | 24576 (192N) | works | 0.1478 s/step | g4_ng5192_siz* |
| ng5 bigpart dist_32768 | 32768 (256N) | **FAIL** | `CG_kk abort at iter 1: pp·App = nan (s_old=nan)` — NaN in the FIRST iteration | g4_ng5256_siz{a,b} (26952342/43) |

The dist_32768 case is the most valuable: NaN at iteration 1 of (presumably) step 1 means
the NaN is in the ASSEMBLED system (matrix, preconditioner, or rhs) or the very first SpMV —
no slow-divergence dynamics to chase. That partition was freshly generated on 2026-08-14
(partitioner clone: `/home/a/a270088/fesom_part/fesom2/work_part_m12_ng5`, job 26946830).

## 3. Prior evidence (M11 campaign — read `project-m11-partitioning.md` in memory)

- M11 F45: "a STOCK-recipe control blew up at step 14 ⇒ fragility is NOT a MINCONN property —
  screen every partition, any recipe."
- M11 NG5 re-race: MINCONN reproduces −9.71% (spread 0.0) while the 3 alternate partitions
  ALL diverge in both reps — "NG5 = most partition-fragile mesh."
- M11 dars: "the u30 dist_64 is clean while its dist_2048 sibling blew up at step 25 —
  fragility per-partition in BOTH directions."
So the class predates M12; M12's contribution is the clean solver discriminant (SE and oati
survive where cg dies, same runs).

⚠️ Do NOT confuse with the M10 farc "divergences": those were FALSE POSITIVES of the
variants' stall guard (threshold 20 < baseline cg's own 21-iteration plateau; fixed via
`FESOM_SSH_STALL_WINDOW`). The failures above are the REAL divergence guard and a real NaN.

## 4. Hypotheses, ranked, with the evidence so far

**H1 (prime): the preconditioner.** M10's review found the stock preconditioner
`pr_values` is NOT symmetric (R1 in the M10 plan commit f42c453), and Sergey's CG2
instability was CURED by symmetrising it (`FESOM_SSH_SYMPRE`, M10 worktree
`~/port_kokkos_ssh`, `docs/SSH_SOLVERS_M10.md`). An asymmetric preconditioner breaks the
CG's SPD contract → genuine divergence is possible and can be modulated by partition-induced
assembly/ordering differences. **Discriminator D-A below is cheap and decisive-ish.**

**H2: partition-sensitive assembly/halo defect.** In exact arithmetic CG on the same global
SPD matrix is partition-independent; if the LOCAL matrices differ beyond rounding across
partitions (an ownership/halo corner case in `fesom_ssh_stiff_alloc_and_build`,
`fesom_update_stiff_mat_ale_kk`, or the rhs), that is a real bug. The iter-1 NaN points here
too (something in the assembled system is NaN on dist_32768).

**H3 (ELIMINATED at node level): degenerate/tiny ranks.** rpart.out census, 2026-08-14:

```
ng5 /pool dist_4096  (FAILS)  min=1806 med=1807 max=1808 zeros=0 tiny<10=0
ng5 /pool dist_8192  (FAILS)  min= 901 med= 904 max= 905 zeros=0 tiny<10=0
ng5 bigpart dist_16384 (ok)   min= 444 med= 452 max= 459 zeros=0 tiny<10=0
ng5 bigpart dist_20480 (FAILS)min= 316 med= 361 max= 412 zeros=0 tiny<10=0
ng5 bigpart dist_24576 (ok)   min= 241 med= 301 max= 357 zeros=0 tiny<10=0
ng5 bigpart dist_32768 (FAILS)min= 219 med= 226 max= 232 zeros=0 tiny<10=0
```

No zeros, no tiny parts; no size correlation with failure. NOTE the /pool 4096/8192
partitions are ultra-uniform (spread 2!) — a different recipe than bigpart's METIS — yet
bigpart has failing (20480) between working (16384/24576): recipe alone doesn't decide.
An ELEMENT/EDGE-level census (a rank owning nodes but pathological element/edge sets) is
still open — the my_list/com_info files carry it.

**H4: conditioning + FP-order sensitivity.** The stiffness is mass + gθ²τ²·div(H·grad); at
dt180 it is much better conditioned than at dt1800 — yet NG5 fails at dt180. Pure rounding
reordering should NOT diverge a correct SPD CG; only in combination with H1/H2. dt is still
a useful probe (D-D).

## 5. Discriminating experiments (do these roughly in order)

- **D-A (cheapest decisive): run `cg2` on the failing partitions.** The m10 bin has it:
  `BIN=/work/ab0995/a270088/port2/m10/bin/stallknob_serial FESOM_SSH_SOLVER=cg2
  FESOM_SSH_STALL_WINDOW=200` on ng5 /pool dist_4096 (32N, dies in ~2 min if it dies).
  cg2 = CG with the M10 fixes incl. the symmetrised preconditioner. cg2 survives + cg dies
  ⇒ H1 confirmed to first order; port the SYMPRE fix (or the whole cg2) into the stock path.
  Job template: `m12/job_g4_point` (see §2 job names for the exact env).
- **D-B: NaN census of the assembled system on dist_32768.** Instrument
  `fesom_ssh_preconditioner`/stiff build: after assembly, per-rank NaN/Inf scan of
  `values`, `pr_values`, `ssh_rhs`, and the diagonal (5 lines + an Allreduce); print the
  first offending (rank, row, gid). If the assembly is NaN-free, instrument iter-1 of CG
  (dump pp/App at the first NaN). This localises H2 vs H1 immediately.
- **D-C: 1-rank-vs-N-rank matrix comparison on a SMALL fragile partition** (see §6): dump
  owned rows (gid-keyed) of `values`/`pr_values` on np1 and npN and diff — beyond-rounding
  differences = H2 proven, and the diff names the rows.
- **D-D: dt ladder on a failing partition** (dt 180→60→20): if divergence persists at tiny
  dt (near-identity matrix), it is NOT conditioning — strengthens H1/H2.
- **D-E: element/edge-level census** of failing vs working partitions from the dist files
  (my_list*/com_info*): min owned elements, ranks with owned nodes but no owned elements,
  pathological edge cuts.

## 6. Smaller/cheaper reproduction — meshes and a recipe to MANUFACTURE fragile partitions

The failure is per-partition, so a small repro = a fragile partition on a small mesh.
Nothing small is KNOWN-fragile yet; manufacture candidates:

- **pi mesh** (3k nodes, `/home/a/a270088/port2/fesom2/test/meshes/pi`, dist_2/8 exist,
  runs on a LOGIN NODE in seconds — see `reference-build-run.md` memory for the ob1
  override): generate dist_64/128/256 (12–47 nodes/rank — extreme) and screen the stock CG
  at dt100. If ANY pi partition reproduces, the whole debug loop becomes login-node,
  seconds-per-iteration. ⚠️ pi is a COPY-first mesh if you write dists into it — make
  `/work/ab0995/a270088/port2/mesh/pi_fragile/` (house rule: never write into shared dirs).
- **CORE2** (127k, the certified private copy `/work/ab0995/a270088/port2/mesh/core2` —
  COPY it before adding dists, e.g. `core2_fragile`): generate many partitions at 128–864
  ranks with varied seeds/recipes and SCREEN (the M11 `m11_promote` screening pattern:
  short stock-recipe runs). M11's F45 proves CORE2-class stock blowups exist.
- **dars u30 dist_2048** (M11 worktree meshes, `~/port_kokkos_part` area): known blowup at
  step 25 (M11) — 16 nodes, minutes; the only KNOWN-fragile existing partition smaller than
  NG5. Locate it via the M11 memory/worktree docs.
- **Partitioner recipe**: clone `/home/a/a270088/fesom_part/fesom2/work_part_m12_ng5`
  (already set up; edit `MeshPath` + `n_part` in namelist.config, `sbatch job_ini_levante`;
  pi/CORE2 partitioning takes seconds/minutes). Generating MANY seeds: M11's tooling in
  `~/port_kokkos_part` varied recipes/seeds — reuse it.

## 7. Existing instruments

- `FESOM_DIAG_SSHSLV=<gid>` / `FESOM_DIAG_SPREAD=<gid>` — per-step SI-block introspection
  at one node (fesom_step.cpp:857-922). Works under SI (aborts only under se).
- The m10 bin knobs: `FESOM_SSH_SOLVER=cg|cg2|pipecg|oati|pcsi`, `FESOM_SSH_TRACE=1`,
  `FESOM_SSH_STALL_WINDOW`; the M10 solver-lab dump machinery (worktree docs).
- Startup CG self-tests (fesom_main.cpp:660-762): zero-RHS + unit-perturbation stiffness
  residual + rest-state — **currently np1-only** (`do_sanity = npes==1`, :557). Making them
  run at npes>1 on the failing partition is a 1-line change and might catch the NaN before
  any physics runs (fast D-B variant).
- `scripts/diff_snap.py` byte gate; the frozen bins (`/work/ab0995/a270088/port2/m12/bin/`
  se0 pair, `/work/ab0995/a270088/port2/m10/bin/stallknob_*`) — ALWAYS pin BIN=.

## 8. Constraints, house rules, coordination (same-folder!)

- Branch: create your own off the current `m12-split-explicit` tip (suggest
  `m13-cg-robustness`). The M12 session (still alive) is DOCS-ONLY from here: it may add
  docs/report commits for its final board — docs-only, merge-trivial; it checks the current
  branch before committing and will not touch `src/`. YOU own `src/` and the working tree.
- Builds: rebuild `build-m7serial`/`build-m7cuda` freely — every M12 job pins frozen bins
  (BIN=), nothing in flight reads the live builds.
- Do NOT delete the failing partitions (evidence): ng5 /pool are read-only anyway; bigpart
  dist_20480/32768 keep.
- Any fix must pass the house ladder: knob-off byte gate vs the certified baseline
  (`jobs/job_m7_gate_serial`), CUDA fidelity gate, and a re-run of the failing-partition
  inventory (§2) flipping FAIL→works WITHOUT changing any working case's bytes (if the fix
  is default-on, it changes the solve — then it needs the solver-class gates, see M10's
  P-L2 pattern in `~/port_kokkos_ssh/docs/SSH_SOLVERS_M10.md`).
- Cheap gates look cheap (`-t 00:06:00` class); `-C a100_80` on GPU absolutes; ≤16 GPU
  nodes; no CORE2 CPU scaling (user rule, memory `feedback-no-core2-cpu-scaling.md`);
  binaries → /work + sha only.

## 9. Upstream relevance

Our CG is a literal port of solver.F90's CG. If the Fortran production path runs the same
algorithm/preconditioner on these partitions, the fragility may exist upstream too — once
root-caused, this belongs in the Sergey packet (`docs/report/M12_SERGEY_PACKET.md` has the
context; the M11 upstream-PR machinery shows how we ship fixes upstream).
