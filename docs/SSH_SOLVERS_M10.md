# M10 — SSH solver track: ledger of record

**Status:** ACTIVE — Task 1 (worktree + hygiene)
**Plan of record:** `docs/plans/20260805-m10-ssh-solvers.md` (in-tree on this branch)
**Branch:** `m10-ssh-solvers`, worktree `~/port_kokkos_ssh`
**Derivations doc:** `docs/plans/20260805-m10-ssh-derivations.md` (Task 4, to be created)
**Work area:** `/work/ab0995/a270088/port2/m10/{bin,labdumps,gates,ab,figs}`

Every number in this file carries its SLURM job id (or "login") and the binary md5/sha of
record. Lab numbers are NEVER performance numbers of record (R4/T3 rule). Pre-registrations
are written BEFORE the runs they gate.

## Provenance

- **Base commit:** `f42c453` — *deviation from the plan text (which says `65a1a71`): `f42c453`
  IS `65a1a71` + the plan commit itself (docs/plans/20260805-m10-ssh-solvers.md +
  `.gitignore` `ssh_sergey/`). Branching from it puts the plan of record in-tree, which the
  plan's own Progress Tracking rule requires ("update this plan in the same commit as the
  work"). Same lineage, no source-code delta vs `65a1a71` (`src/` untouched by `f42c453`).*
- **Source material:** `ssh_sergey/` (gitignored, copied from the main checkout 2026-08-05):
  `solvers.F90`, `1612.01395v3.pdf` (Chronopoulos-Gear/pipelined CG), `gmd-9-4209-2016.pdf`
  (P-CSI), `manasi-pcg-jpdc2022.pdf` (PIPECG-OATI).
- **Kokkos submodule:** checked out in-worktree @ `15dc143e` (4.4.01, same as main).
- **`/work` quota at start:** group ab0995 usage 2.533 PB, no hard group cap enforced
  (`lfs quota -g ab0995`); global `/work` free ≈ 14 TB — lab dumps are 2-D-matrix-sized
  (MBs–GBs), fine. Re-check before the NG5 np≥256 dumps land (T3).

### m10-base binaries (provenance only — NOT byte-comparable to the main checkout's:
### `__FILE__`/debug paths differ; behavioural identity is what the base gate proves)

| build | sha256 | md5 | built |
|---|---|---|---|
| `build-m7serial/fesom_port` | `f228d664eb0de144c6f6914212669372b676c9c2c9d9a8672291b601e5784c41` | `54433326fe312057bd0e9fd980de298f` | 2026-08-05 |
| `build-m7cuda/fesom_port` | `e5245fa3290b2247cf6bd115260de61fd6bcb267b413cda7f240b03db7580d2c` | — | 2026-08-05 |

*(Main-checkout `build-m7serial` md5 at the same date: `9743f602aaf8d27dcfbe9baae9b3c977` —
differs from the worktree's, so a gate log showing `54433326…` provably ran THIS tree's
binary: the R9 discriminator.)*

## Gate registry

Every gate/A-B row: date · gate · job id · binary md5 (from the log — R9) · result.

| date | gate | job | binary md5 | result |
|---|---|---|---|---|
| 2026-08-06 | T1 base: knob-free serial 20-step CORE2 byte gate (worktree ROOT, HEAD f42c453) | 26722627 | `54433326…` (log also shows main-checkout `9743f602…` ≠ — R9 armed) | **PASS** diff_snap rc=0 |
| 2026-08-06 | T2 knob-off byte gate on the [ssh-wire]+VERIFY instrumentation commit | 26722771 | `8f2be32b…` ≠ main `9743f602…` (R9) | **PASS** diff_snap rc=0 |

**T2 instrumentation smoke (login pi np2, dt100, 20 steps, STATS+VERIFY on):** counts EXACT
vs hand count (3-iter solve: exch=8=2+2k, ar_blk=8=2+2k, body-launches=17=6+4k−1(break));
verify true≡rec to 7 digits, **max |true−rec| gap over 20 solves = 1.05e-11** — first data
point for the (deferred) verify gate threshold.

## Pre-registrations

*(Written BEFORE the corresponding runs. Layer-2 per-field solver-class bounds land here in
T2, citing `docs/REFERENCE_RUNS.md` per-scheme floors + the L79 zstar Kv ~1e-1 control class
+ the CGPOLY/M5.23 bar. Each A/B's expected range with reasoning lands here before
submission.)*

## Baseline census (T2 — the sizing-of-record table)

*(iters/solve, halo-exchange count, blocking/i-allreduce counts, PHASESTATS `FESOM_PH_CG`
ms/step, CG-phase kernel-launch count + per-launch overhead — CORE2 np8 login, dars g4n,
NG5 g4n/g16n.)*

## Findings ledger

*(R1 symmetry-defect numbers, R2 probe results, lab-vs-model divergences, typo-report
pointers, every pre-registration vs outcome — wrong-high/wrong-low noted, house style.)*

## Frozen binaries

*(`/work/ab0995/a270088/port2/m10/bin/` + sha256 here; binaries NEVER in git.)*
