# M14 — Integration branch + combined strong-scaling campaign

**Date:** 2026-08-15 · **Branch:** `m14-integrate` (new worktree `~/port_kokkos_int`)
**Status:** PLANNED — no work started · **Rev 2** (post plan-review, 2026-08-15)

---

## Overview

Merge the five active campaign branches into a single branch that is mergeable to `main`, under a
hard condition set by the user: **with every new knob unset, the binary must return to the
pre-campaign state that reproduces FESOM-Fortran closely** (the configuration the paper rests on).

Then run, for each mesh, separate CPU and GPU strong-scaling ladders, and produce one figure per
mesh/backend carrying two curves: the state *before* these campaigns, and the state with the best
available combination of new knobs and partitions.

**What this buys.** Five tracks (M9 sea ice, M10 SSH solvers, M11 partitioning, M12b split-explicit
+ wide halo, M13 deterministic IC) currently each hold a handful of single-point wins measured on
their own branch, against their own baseline, on their own day. None has ever been measured
*together*, on one binary, along a ladder. This plan turns scattered point results into one
defensible board, and produces a shippable branch as a by-product.

**Honest framing of the two curves.** The "before" curve is not literally the pre-campaign binary:
`FESOM_IC_EXTRAP=det` is on in both arms (D3), because a det-off baseline has *holes* — four of six
NG5 partitions are fatal for stock CG. The "after" curve changes the free-surface formulation (SE),
the ice rheology (mEVP), the partition and the IC simultaneously. Both facts are stated on the
figure and in the campaign document, not buried.

**Non-goals.** M8 mixed precision is out of scope. No new physics.

---

## Context (from discovery, verified 2026-08-15)

### Branch topology — all five branch off `main = 65a1a719`

| branch | commits ahead | source footprint | product |
|---|---|---|---|
| `m13-cg-robustness` | 29 | `fesom_phc.cpp` +231, `fesom_main.cpp` +91 | det IC hole-fill; **already contains all of `m12-split-explicit`** |
| `m12b-widehalo` | 85 | `fesom_ssh_se.*`, `fesom_mesh.cpp`, `fesom_step.cpp` | forked from m12 at `ab25f6c`; SE file is a **superset** of m12's (+1212/−29) |
| `m9-mevp-double` | 111 | `fesom_ice*.{cpp,h}` +2272 | wide-halo mEVP + lean writing |
| `m10-ssh-solvers` | 89 | `fesom_ssh.{cpp,h}` +2000, `tools/`, `tests/` | cg2 / pipecg / oati / p-csi |
| `m11-partition` | 134 | **none** beyond the shared det patch | partition data + docs + jobs |

Verified relations:
- `git merge-base --is-ancestor m12-split-explicit m13-cg-robustness` → **YES** (m12 ⊂ m13)
- `merge-base(m13, m12b)` = `ab25f6c`, which m13 contains
- the det-IC diff on `fesom_phc.cpp` is **byte-identical** on m9/m10/m11/m13 (sha1 `fbe4a19b3fce`)

### Conflict probe (`git merge-tree --write-tree --messages`, against `m13`)

```
m12b -> 0 conflicts
m10  -> src/fesom_main.cpp  +  docs/KOKKOS_PORTING_LESSONS.md
m9   -> docs/KOKKOS_PORTING_LESSONS.md
m11  -> docs/KOKKOS_PORTING_LESSONS.md
```

The lessons-file conflicts are append-only and resolve by concatenation — but check for **L-number
collisions**, since the branches numbered independently (m12b reached L122).

⚠️ **The `fesom_main.cpp` conflict is a duplicated hunk, not three additive blocks.** m10 already
carries the identical M13 elemprobe block (`port_kokkos_ssh/src/fesom_main.cpp:1440-1494` vs
`port_kokkos/src/fesom_main.cpp:1448-1502`); "keep both" yields two copies of the same probe. m10's
*own* `fesom_main.cpp` delta is teardown/report wiring (`fesom_ssh_wire_report()`,
`fesom_ssh_m10_free()`, `:1628-1631`), not knob parsing.

### Verified source facts that shape the recipe

- **`FESOM_SPEED_EVPWIDE_LEAN` is inert under standard EVP.** `fesom_ice_evpwide.cpp:855` —
  `if (ice->whichEVP == 0)` prints *"the lean lever is NOT running"*. M9's whole ④L result is an
  **mEVP** result, so `FESOM_WHICH_EVP=1` is required, in **both** arms.
- **K is `FESOM_SPEED_EVPWIDE=K`**, via `fesom_evpwide_env_K()` (`:132`). `FESOM_EVPWIDE_RINGS` is a
  debug override that only *widens* R past K (`:166`, `if (v > R) R = v`) — extra communication for
  a drift diagnostic, no gain. It must not appear in a performance recipe.
- **CGPIPE is inside `FESOM_SPEED=1`.** `fesom_ssh.cpp:2070` — *"ADOPTED into the FESOM_SPEED=1
  blessed set 2026-07-16 (user decision)"*. So an abort on "SE + CGPIPE" would abort **every best
  arm leg**, since D7 sets `FESOM_SPEED=1` in both arms.
- **`FESOM_SE_WIDE` accepts K=1 only** (`port_kokkos_wh/src/fesom_ssh_se.cpp:128-152` aborts on
  K≥2). Deep K is not reachable without the extended-mesh build; this plan does not chase it.
- **The precedent for an inert knob is NOTE-and-ignore**, `fesom_step.cpp:112-118` — *not*
  abort-on-pair. `fesom_ale.cpp:268-283` (the wsplit precedent) is abort-on-unrecognised-**value**.

### Partition inventory — verified, and it is the campaign's hard blocker

`/work/ab0995/a270088/port2/mesh_m11_certified/` contains **five** `dist_*` directories, total:

```
core2_v1/dist_4      core2_v1/dist_512     core2hil_v1/dist_512
farc_v1/dist_2048    dars_v1/dist_2048
```

Nothing for NG5. The ladders in C3 need roughly fifty. Worse, **the base arm is blocked too**:
`SSH_SOLVERS_M10.md` states *"CORE2 has no 1024-rank partition; 864 is the largest that exists"* and
*"the baseline is still improving at 864 … CORE2 has not turned over; it is losing efficiency, not
throughput. A turnover point would need a partition beyond 864, which does not exist for this
mesh."* Its efficiency column is normalised against **128** ranks (256 → 93.1, 432 → 87.5,
512 → 83.3, 864 → 66.6), and the 864 rung packs ~123 tasks/node rather than filling nodes.

⚠️ An earlier revision of this plan cited "efficiency vs 864 ranks = 96 % @1024, 64 % @1536,
55 % @2048". **Those numbers do not exist**; they were inherited from a memory note that
mis-attributed them. Corrected at source.

Each certified point also used a **different partitioner** (METIS `MINCONN`, KaHIP `UFACTOR=30`,
Mt-KaHyPar, KaMinPar), chosen at one rank count, and `M11_RECOMMENDATION.md` warns the winner is
point-specific. `core2hil_v1` is a **renumbered mesh** that re-baselines every C↔K floor in
`docs/REFERENCE_RUNS.md` (`:258-263`) — **excluded from M14**, since the acceptance bar here is
bitwise identity to `main`.

### Other artifacts

- `scripts/m7_scaling_figs.py` — **read before writing any plotting code** (standing user rule)
- Frozen bins: `/work/ab0995/a270088/port2/{m7,m9,m10,m11,m12,m12b,m13}/bin` → M14 goes to
  `port2/m14/bin`, sha-named. **Never commit binaries.** The correct m12b pair is serial `73c6cf29`
  / cuda `58ac143b`; `4cc9eda4`/`3a584dc9` predates the lean fix.
- Job precedents on `main`: `jobs/job_{scaling,m7_scale,dars_scaling,ng5_scaling}_{cpu,gpu}`
- Big-partition mesh copies live in `core2_bigpart`, `dars_bigpart`, `ng5_bigpart`

---

## Decisions taken (brainstorm 2026-08-15 — do not relitigate)

| # | decision |
|---|---|
| D1 | Merge scope = M9 + M10 + M11 + M12b + M13. M8 out. M12 subsumed by M12b. |
| D2 | Revert gate = **bitwise vs `main`, both backends** with all new knobs unset. |
| D3 | `FESOM_IC_EXTRAP=det` is **ON in both arms** — a correctness fix, not a speed knob. |
| D4 | Config selection = a-priori seed recipe + a **bounded interaction hunt**, not a full factorial. |
| D5 | Matrix = 4 meshes × 2 backends. The "no CORE2 CPU scaling" rule is **relaxed**. **CPU ladders run past the rollover knee** where partitions permit; GPU stops at the knee or the 16-node cap — ask before 32. |
| D6 | Account shared; 2–3 of the 5 `ab0995_gpu` slots. |
| D7 | Baseline curve = integration binary, new knobs off, `FESOM_SPEED=1`, det on, mEVP on. |
| D8 | Fidelity bar = 3000-step screen per **mesh × backend** per published config + NaN-zombie check on every run, **plus one 1-year CORE2 twin** of the combined config (see E-twin). |

---

## Development approach

This is an HPC measurement campaign. "Tests" means **gates**: byte gates (`diff_snap`), liveness
gates (a knob must change something), and screens (3000 steps, finite diagnostics, non-vacuous
solver trace). Every task ends with one.

Standing rules, non-negotiable, applying to every task:

- **`BIN=` pinned** on every job (multi-srun re-execs at each `srun`)
- ladder `dt` per mesh: CORE2 1800 · fArc 900 · dars 120 · NG5 180
- `-C a100_80` on every GPU absolute (heterogeneous partition — L94)
- **`snap_every=-1` at ≥4096 ranks** (not ">4096" — 4096 itself is in the ibv_reg_mr regime)
- a cheap gate must *look* cheap (`-t 00:06:00` flips `(Priority)` → `(None)`)
- all output to `/work`, never `$HOME`
- **budget the det fill**: ≈7 min at NG5/64 ranks vs 71 s of compute; grows with mesh, falls with
  ranks; has already killed four jobs. It is **setup, and must be outside the timing window.**

---

# Phase 0 — Partition inventory and generation (start day 0, off the critical path)

*Elevated to Phase 0 by review: it blocks most of Phase D, depends on nothing in Phase A, and its
absence would surface only after all the merge and gate work was already spent.*

### Task P1: Audit what exists

**Files:** Create `docs/m14/PARTITIONS.md`

- [ ] enumerate every `dist_N` under `mesh_m11_certified/`, the base mesh trees, and the
      `*_bigpart` copies; record which partitioner produced each certified point
- [ ] mark every (mesh, backend, rank-count) cell of the C3 ladders as **have / must-generate /
      impossible**
- [ ] **exclude `core2hil_v1`** from M14 and record why (renumbered mesh re-baselines every C↔K
      floor, against an acceptance bar of bitwise identity to `main`)
- [ ] gate: the table has an entry for every ladder point, base arm and best arm

### Task P2: Generate the missing partitions

**Files:** Create `jobs/job_m14_partgen`

- [ ] generate base-arm partitions for every ladder point that lacks one — **including CORE2 CPU
      beyond 864**, which does not exist today and without which the CORE2 past-the-knee points
      cannot run at all
- [ ] generate best-arm partitions only where M11 identified a specific winning partitioner for
      that mesh; use `/home/a/a270088/fesom_part/fesom2/work_part`, output to `/work` copies
      (**never modify `/pool`**)
- [ ] **policy for cells with no certified best partition** (most of them): run the *base* partition
      in **both** arms, so the partition lever is simply absent at that point and the figure says
      so. Do **not** substitute a fresh untested draw — a re-roll is +8…+15 % slower at CORE2 864
      and +3…+8 % at fArc.
- [ ] gate: every "must-generate" cell from P1 is either generated and loadable, or reclassified
      as "partition lever absent" in writing

---

# Phase A — Build the integration branch

### Task A1: Create the worktree and branch

- [ ] `git worktree add ~/port_kokkos_int -b m14-integrate m13-cg-robustness`
- [ ] copy the gitignored third-party dirs (`ice_sergey/`, `ssh_sergey/`) into the worktree
- [ ] confirm HEAD = `624e04c` and `src/fesom_ssh_se.cpp` (m12's SE) is present
- [ ] gate: `git status` clean

### Task A2: Merge m12b (wide halo) — expect zero conflicts

**Files:** Modify `src/fesom_ssh_se.{cpp,h}`, `src/fesom_mesh.cpp`,
`src/fesom_forcing_analytical.cpp`, `src/fesom_ice_evpwide.cpp`, `docs/KOKKOS_PORTING_LESSONS.md`

- [ ] `git merge m12b-widehalo` (real merge, no squash — provenance must stay bisectable)
- [ ] verify `fesom_ssh_se.cpp` is m12b's superset (~2320 added lines vs `main`), not m12's 1147
- [ ] verify the m12b **lean fix** is present (device gather/scatter staging, **not** whole-array
      `Fbt.sync_host/sync_device`) — L121, worth 37 % of NG5 16N's busy
- [ ] gate: build Serial + CUDA clean
- [ ] **gate: per-merge Serial byte gate** — knobs-off, pi np1 + CORE2 np8, `diff_snap` vs the
      `main` `h17` reference. Cheap (minutes) and it attributes a byte break to *this* merge instead
      of surfacing it at B1 after all five.
- [ ] gate: `FESOM_SSH_MODE=se FESOM_SE_WIDE=1 FESOM_SE_H0E_XCHG=1 FESOM_SE_WIDE_RECON=1` reproduces
      the bitwise-exact rung (drift 0.0 every step) at np8

### Task A3: Merge m11 (partitions) — docs/jobs/scripts only

- [ ] `git merge m11-partition`
- [ ] resolve `KOKKOS_PORTING_LESSONS.md` by concatenation; **check L-number collisions**
- [ ] gate: `git diff --stat HEAD^1 HEAD -- src/` is empty (m11 carries no source)

### Task A4: Merge m9 (sea ice)

**Files:** Modify `src/fesom_ice*.{cpp,h}`, `docs/KOKKOS_PORTING_LESSONS.md`

- [ ] `git merge m9-mevp-double`; resolve the lessons file
- [ ] check the m12b 5-line touch to `fesom_ice_evpwide.cpp` survived m9's rewrite of that file
- [ ] gate: build Serial + CUDA clean
- [ ] **gate: per-merge Serial byte gate** (as A2)
- [ ] **gate: ice liveness — assert on the `[fesom_speed]` announce line, not the exit code.**
      Run `FESOM_WHICH_EVP=1 FESOM_SPEED_EVPWIDE=8 FESOM_SPEED_EVPWIDE_LEAN=1` and require the
      announce to confirm the wide halo *and* the lean path are running. The naive form
      (`FESOM_SPEED_EVPWIDE=1` + `FESOM_EVPWIDE_RINGS=8`, no `WHICH_EVP`) passes on a warning-only
      no-op — this is precisely the L80 trap.

### Task A5: Merge m10 (SSH solvers) — the one real conflict

**Files:** Modify `src/fesom_main.cpp` (**conflict**), `docs/KOKKOS_PORTING_LESSONS.md`;
add `src/fesom_ssh.{cpp,h}` changes, `src/fesom_ssh_dump.h`, `tools/`, `tests/`

- [ ] `git merge m10-ssh-solvers`
- [ ] resolve `fesom_main.cpp` by **diffing the two hunks**: the M13 elemprobe block is duplicated
      and resolves to **one** copy; then verify m10's own additions (`fesom_ssh_wire_report`,
      `fesom_ssh_m10_free`) survive
- [ ] **verify the M10 NaN-blind stall-guard fix is present** — `resid >= rtol` is false for NaN, so
      the loop was skipped and the run reported "0 iters, converged"; a zombie leg measured 10.8 %
      *faster* than a healthy run
- [ ] gate: build Serial + CUDA clean; `ctest` passes including `test_ssh_solvers`
- [ ] gate: per-merge Serial byte gate (as A2)
- [ ] gate: each of `FESOM_SSH_SOLVER=cg2|pipecg|oati|pcsi` completes at CORE2/np8

### Task A6: Knob hygiene — summary first, aborts narrowly

*Rewritten after review: the original "abort on mutually-exclusive pairs" would have aborted every
best-arm leg, because CGPIPE lives inside `FESOM_SPEED=1`.*

- [ ] emit a **startup summary** of every active new knob — this is the actual deliverable, it makes
      every log self-documenting
- [ ] abort **only** when an explicitly-set *per-lever* knob is inert: e.g. `FESOM_SSH_SOLVER=oati`
      or `FESOM_SPEED_CGPIPE=1` together with `FESOM_SSH_MODE=se`
- [ ] **never abort on a master-implied lever** — `FESOM_SPEED=1` + `se` must run, and warn at most
- [ ] follow the NOTE-and-ignore precedent (`fesom_step.cpp:112-118`) for everything else; do not
      widen this into a general knob-table audit mid-campaign
- [ ] gate: `FESOM_SPEED=1 FESOM_SSH_MODE=se` **runs** (the combination the campaign depends on)
- [ ] gate: `FESOM_SPEED_CGPIPE=1 FESOM_SSH_MODE=se` aborts with rc != 0 and a clear message
- [ ] gate: with no new knob set, the summary reports "no M14 knobs active"

---

# Phase B — The revert gate (the user's stated condition)

### Task B1: G1 — byte gate vs `main`, both backends

**Files:** Create `jobs/job_m14_g1_byte`, `docs/m14/G1_byte_gate.md`

- [ ] build Serial + CUDA with the **same compiler, flags and Kokkos version** as `main`'s `h17`
      (CUDA `f8384e86` / Serial `5c3c90fc`); freeze to `port2/m14/bin`, record shas here
- [ ] **specify the option matrix** (L91 — options ×3; a default-config gate never executes
      `fesom_ice_maevp.cpp`, the 2272-line file m9 rewrote, nor the zstar/SE-adjacent paths m12b
      touched): default (EVP+KPP), **mEVP**, **zstar**, **TKE**, and the combined twin
- [ ] **specify the step count**: long enough to clear Z7/L78 — a mode that makes a constant array
      time-varying is bitwise-equal at cold start and only breaks at step 2. Minimum 20 steps with
      snapshots at 1, 2 and 20; a 1-step gate is worthless here.
- [ ] Serial points: pi np1/np2, CORE2 np8, CORE2 np128
- [ ] CUDA points: **np8 and np16 (2 and 4 nodes)** — *not* np128, which would be 32 GPU nodes and
      breaks the 16-node cap. A byte gate gains nothing from rank count beyond exercising multi-node
      halo.
- [ ] state explicitly that `diff_snap` compares **snapshot files, not logs** — A6's startup summary
      legitimately changes stdout
- [ ] gate: zero differing bytes, both backends, every option in the matrix. Any difference blocks
      Phase C.

### Task B2: G2 — knob-liveness gate (the L80 trap)

- [ ] **grep** the knob list from the merged tree; do not hand-write it
- [ ] switch each on individually at CORE2/np8 and confirm the output **differs** from knobs-off
- [ ] for knobs bit-neutral by design (the m12b wide halo is bitwise-exact), assert on the
      instrumented signal instead — announce line, exchange counts, phase timers, selfcheck — and
      record which knobs are in this class and why
- [ ] include the ice combination explicitly: `WHICH_EVP=1` + `SPEED_EVPWIDE=8` + `LEAN=1`
- [ ] gate: no knob is silently dead; every knob has a documented liveness signal

### Task B3: G3 — null-cost gate

- [ ] time `m14-integrate` knobs-off vs `main` `h17`, same allocation, at CORE2 4 GPU,
      fArc 2048 CPU, dars 64 GPU
- [ ] **both campaign arms use the integration binary regardless of the outcome.** A non-zero null
      cost is a **finding to report and fix**, not a reason to split binaries — splitting would put a
      binary + toolchain difference under every point in the campaign, which is far worse than a
      known small overhead.
- [ ] state the resolution limit honestly: at ~2 % baseline spread this test cannot resolve a 2 %
      effect, so a pass is "no large overhead", not "zero overhead"
- [ ] gate: the number and its uncertainty are recorded before any ladder is submitted

---

# Phase C — Campaign preparation

### Task C1: Resolve the partition decisions

- [ ] confirm with the user whether `dars_gpu_v1` / `ng5_gpu_v1` promote (`M11_RECOMMENDATION.md:280`
      — both already pass `m11_promote`'s stability bar, screens 26895260 / 26908635; `:278` still
      says "not yet packaged")
- [ ] re-measure the NG5 64 GPU alternates under `det` — `20260815-m11-det-rerun.md:129` lists all
      three as "diverged in both reps at ladder dt", which predates the det fix
- [ ] for dars CPU, decide whether the seed re-roll enters the recipe — ⚠️ **race it, never assume**
- [ ] gate: every mesh/backend cell names a specific partition directory, or is marked
      "partition lever absent"

### Task C2: Freeze the config matrix

**Files:** Create `docs/m14/CONFIG_MATRIX.md`, `jobs/m14_config.sh`

**Both arms carry:** `FESOM_SPEED=1`, `FESOM_IC_EXTRAP=det`, `FESOM_WHICH_EVP=1`.
`WHICH_EVP` is a **scheme choice like det**, not a lever — M9's result is an mEVP result, and the
lean path does not exist under standard EVP.

Seed recipe (a starting guess, to be confirmed or refuted by D2):

| mesh / backend | SSH lever | ice lever | partition |
|---|---|---|---|
| CORE2 GPU | SE + wide halo (−9.1 % at 16 N) | `SPEED_EVPWIDE=8` + `LEAN=1` | `core2_v1` where it exists |
| CORE2 CPU | probe | same | `core2_v1/dist_512`; elsewhere absent |
| fArc CPU | SE (−14.2 %) | same | `farc_v1/dist_2048`; elsewhere absent |
| fArc GPU | SE + wide halo | same | pending P1 |
| dars CPU | `oati` (SE is a loss, +1.1 %) | same | `dars_v1/dist_2048`; elsewhere absent |
| dars GPU | probe SE vs implicit | same | pending C1 (`dars_gpu_v1` unpromoted) |
| NG5 CPU | `oati` (−0.7…−1.9 %) | same | **none exists** — lever absent |
| NG5 GPU | SE, **wide halo off** (wash at 16 N) | same | pending C1 |

- [ ] `jobs/m14_config.sh` carries **`MESH=` and `dist_N` per point**, not just arm env — the base
      arm's mesh differs by rank count (`core2_bigpart`, `dars_bigpart`, `ng5_bigpart`)
- [ ] confirm from the M9 docs that ④L is the CPU winner too, not just GPU
- [ ] gate: sourcing the file and echoing each cell reproduces this table, with a real path per cell

### Task C3: Pre-register the ladders, stopping rule, reps and estimator

**Files:** Create `docs/m14/LADDERS.md`

**Stopping rule — stated so it cannot be adjusted after seeing data:**

- metric = **median per-step model time of the BASE arm** (the base arm defines the knee for both,
  so the two curves always span the same points)
- the knee = the first rung where **two consecutive** rungs each return < 10 % time reduction
  *normalised per unit of added resource* (the ladders contain non-doublings — 512→864 is ×1.69,
  864→1024 ×1.18, 4096→6144 ×1.5 — so a raw "per rung" threshold is not well defined)
- **CPU** continues one rung past the knee where a partition exists; **GPU** stops at the knee or
  the 16-node cap, whichever comes first
- if the rule and the table below disagree, **the rule wins** and the table's remaining points are
  dropped

| mesh | CPU ranks | GPU nodes (4× A100-80) |
|---|---|---|
| CORE2 | 128, 256, 432, 512, 864, **[1024, 1536 — require P2 generation]** | 1, 2, 4, 8, 16 |
| fArc | 512, 1024, 2048, 4096, **8192** | 1, 2, 4, 8, 16 |
| dars | 1024, 2048, 4096, 6144, 8192 | 2, 4, 8, 16 |
| NG5 | 2048, 4096, 8192, 16384 | 4, 8, 16 |

⚠️ **CORE2 CPU above 864 does not exist today** and must come from P2. M10 measured efficiency
falling to 66.6 % at 864 (vs 128) while throughput was *still improving* — so CORE2's turnover has
never been observed, only its efficiency decline. Showing the turnover is exactly why the user
relaxed the rule, and it requires new partitions.

**Cut by review, and I agree:** NG5 32768 (256 CPU nodes) and dars 16384 (128 nodes) are dropped as
routine past-the-knee points. They would likely exceed the rest of the CPU campaign combined and
contradict the 2–3-slot budget. If NG5 or dars has not clearly turned over, add **one base-arm-only
single leg** at the next rung and label it a shape indicator, not a measured point.

**Rep and estimator rule:**

- **equal leg count per arm at a point**; the noisier arm sets that common count. Unequal n biases
  a min estimator — with base spread 2.0 % and arm spread 0.1 %, min-of-3 vs min-of-2 shifts the
  ratio by ~1 pp, the size of the effect at the small-gain points.
- 2 legs default, 3 where the base arm is noisy (M11: base 2.0 %, arms 0.1 %)
- estimator = **min over all legs**, chosen by cross-pair reproducibility: M12b's CORE2 16 N GPU
  number moved −11.5 % → −9.1 % because min-of-2 on a bimodal certified arm did not reproduce,
  while min-over-all-legs agreed across two independent pairs to 0.1 %
- if two pairs of the same point disagree by more than the arm spread, **add legs**, do not choose

**Timing window:**

- 300 steps per leg, pinned; discard the first 10 steps as warmup
- the timer is the **model per-step timer**, never total walltime — the det fill is ≈7 min against
  71 s of compute at NG5/64 and differs between partitions, so a walltime-based ratio is garbage
- `m14_collect.py` **refuses** a log lacking that timer

**Allocation discipline:** run both arms **inside one allocation on the same nodes**, alternating
arm order between legs. "Same day" is too weak when the target gain is −0.7 % (NG5 CPU) against a
2 % base spread; `-C a100_80` constrains the GPU model, not node identity. This is free.

- [ ] commit this file **before** the first ladder job is submitted
- [ ] note the NG5 GPU device-memory floor — `dist_8` GPU was at the A100-80 ceiling, so the NG5 GPU
      ladder cannot start below 4 nodes

### Task C4: Resolve the fArc hang before any fArc ladder

*Promoted from a footnote by review: the fArc CPU ladder is 512…8192 and the E.T1 proto hang covers
**every** point in it. A known reproducible hang over 100 % of a ladder is a blocker, not a note.*

- [ ] reproduce the hang, establish whether it still fires on the merged tree
- [ ] either fix it, or document a mitigation that every fArc job applies
- [ ] gate: a 512-rank and a 4096-rank fArc leg both complete on the merged tree

### Task C5: Fidelity screens

- [ ] 3000-step screen for **base and best on each of the 8 mesh×backend cells** (16 screens, not
      8 — backends are not interchangeable, and dars CPU `oati` is a different config from dars GPU SE)
- [ ] screen at the **knee** point, and additionally at any large-rank point where M13's census found
      partition-dependent failures (`dist_20480`, `dist_32768`)
- [ ] SE bitwise-drift check where the wide halo is active (drift 0.0 every step)
- [ ] **run this AFTER D2 freezes the recipe** — D2 can overturn the seed, which would otherwise
      invalidate the screens; budget a re-screen if a cell changes
- [ ] gate: no config appears in a published curve without a passing screen for its own cell

---

# Phase D — Campaign execution

### Task D1: Harness + zombie detector

**Files:** Create `jobs/job_m14_ladder_{cpu,gpu}`, `scripts/m14_collect.py`,
`scripts/m14_zombie_check.py`

- [ ] build both ladder jobs from the `job_m7_scale_{cpu,gpu}` precedent; `BIN=` pinned, mesh `dt`,
      `snap_every=-1` at ≥4096 ranks, `-C a100_80` on GPU, both arms in one allocation
- [ ] `m14_zombie_check.py`: reject any leg reporting "0 iters, converged", non-finite diagnostics,
      or non-finite `uv` max — run on **every** leg before its timing is admitted
- [ ] `m14_collect.py`: tidy CSV (mesh, backend, ranks/nodes, arm, leg, per-step time, SYPD, job id,
      binary sha, full config string, allocation id); refuse logs without the per-step timer
- [ ] gate: the detector flags known-bad M10 run 26961492 and passes a known-good one
- [ ] gate: the collector round-trips one existing M12b log correctly

### Task D2: Interaction hunt — 4 probes, at the knee, multiplicative null

*Revised on all three axes by review.*

- [ ] **null model is multiplicative**: `T_all/T_base = Π (T_lever/T_base)`. Speedups compose by
      products, not sums — three −10 % levers predict −27.1 %, not −30 %. At this campaign's
      magnitudes the additive-vs-multiplicative gap is 1–3 pp, which is the size of the residual
      being hunted **and points in the pre-registered direction**, so an additive null would have
      confirmed "sub-additive on CPU" from arithmetic alone.
- [ ] **probe at the knee, not the cheapest rung.** All three levers are communication levers whose
      value grows with rank count, and the plan's own elasticities differ per point (CPU 0.17 vs
      GPU 0.79; bt is 23 % of CORE2 16N's step but 6.8 % of NG5's). An interaction measured where
      the levers are weakest does not transport to where every headline number lives.
- [ ] **4 probes, not 8**: one CPU and one GPU cell where two independent levers genuinely both
      exist, plus the two cells with the largest single levers. Where no certified partition exists
      (most cells, per P1), `partition-alone` is an empty config and the probe degenerates.
- [ ] configs per probe: `base`, `ssh-alone`, `ice-alone`, `partition-alone` (where it exists),
      `all-on`, at the C3 rep count
- [ ] pre-register the predictions **before running**: sub-additive on CPU (comm levers compete for
      a wait pool that is 70 % imbalance at fArc 2048, elasticity 0.17); closer to additive on GPU
      (staging lands in busy, elasticity 0.79); super-additive candidate = partition × comm lever,
      since a partition drains the imbalance pool and a comm lever the message-floor pool
- [ ] chase only positive residuals exceeding measured noise, at the mesh where they appear
- [ ] update `jobs/m14_config.sh` wherever the hunt overturns the seed
- [ ] gate: `ADDITIVITY.md` records prediction, observation and residual for all 4 probes —
      **including the refutations**

### Task D3: CPU ladders (4 meshes)

- [ ] CORE2, fArc, dars, NG5 — base + best, both arms in one allocation, stop by the C3 rule
- [ ] every leg passes the zombie check before its timing is admitted
- [ ] gate: `m14_results.csv` has both arms at every admitted point, with binary sha and allocation id

### Task D4: GPU ladders (4 meshes, 16-node cap)

- [ ] CORE2 1→16, fArc 1→16, dars 2→16, NG5 4→16 nodes; base + best
- [ ] if NG5 GPU has not rolled over at 16 nodes, **stop and ask the user** before requesting 32
- [ ] gate: as D3

### Task D5: Sequencing under a shared account

- [ ] never hold more than 2–3 of the 5 `ab0995_gpu` slots
- [ ] interleave meshes, not arms, so a stuck job cannot block a whole mesh
- [ ] gate: a running log of job ids → (mesh, backend, ranks, arm, leg, allocation) is maintained

---

# Phase E — Analysis and delivery

### Task E1: Figures

**Files:** Create `scripts/m14_scaling_figs.py`, `docs/figures/m14_scaling_{mesh}_{backend}.png`

- [ ] **read `scripts/m7_scaling_figs.py` first** — standing user rule
- [ ] 8 panels: base vs best, knee marked, config annotated, and **cells where the partition lever
      is absent labelled as such**
- [ ] ⚠️ **SYPD at production dt is not cost-invariant for SE**: the barotropic subcycle count scales
      with the baroclinic dt, so an SE leg timed at dars ladder dt 120 and replotted at production
      dt 240 understates SE's per-step cost. Either state the assumption on the figure, or measure
      one SE point at production dt to calibrate.
- [ ] gate: figures regenerate from `m14_results.csv` with no manual editing

### Task E-twin: One 1-year CORE2 twin of the combined config

*Added by review, and it is cheap against a campaign this size.* The best arm changes free surface,
ice rheology, partition and IC at once; the only long-integration evidence that exists is M12's SE
twin at CORE2 alone. A 3000-step screen cannot see a slow drift.

- [ ] 1-year CORE2 twin, combined best config vs baseline
- [ ] gate: climate-close by the `docs/REFERENCE_RUNS.md` per-scheme floors, or the production
      recommendation in E2 is softened to "measured over 3000 steps" wording

### Task E2: Campaign document

**Files:** Create `docs/M14_INTEGRATION.md`

- [ ] the board, the additivity table, the three gates with numbers including the null cost
- [ ] the recommended production knob package per mesh and backend — with fidelity evidence stated
      alongside each recommendation
- [ ] every config screened, and every config rejected and why
- [ ] state plainly that det is on in both arms, and that the best curve changes four things at once
- [ ] gate: no number in the document lacks a job id

### Task E3: Merge readiness

- [ ] re-run G1/G2/G3 on the final tree (knobs may have moved during D2)
- [ ] lessons file reconciled, no duplicate L-numbers
- [ ] binaries in `port2/m14/bin`, shas in the docs, **none committed**
- [ ] open the merge to `main` for the user's decision — **do not merge unasked**
- [ ] gate: `git diff main...m14-integrate -- src/` reviewed end to end

### Task E4: Verify acceptance criteria

- [ ] every knob unset returns bitwise to `main` on both backends, across the full option matrix
- [ ] every mesh has a CPU and a GPU panel with two curves
- [ ] every published config has a passing screen for its own mesh **and backend**
- [ ] the stopping rule was applied as pre-registered, not adjusted after seeing data
- [ ] move this plan to `docs/plans/completed/`

---

## Technical details

**Knob table: generate it, don't hand-write it.** B2 already grep's the knob list; E2's table must
come from the same grep. The Rev-1 hand-written table was missing `FESOM_SPEED_EVPWIDE` (the primary
ice knob), `FESOM_WHICH_EVP`, `FESOM_EVPWIDE_RINGS`, `FESOM_EVPWIDE_SELFCHECK`, `FESOM_SE_M_FORCE`,
`FESOM_SSH_RING` and `FESOM_SSH_FALLBACK`.

**Mutually exclusive:** `FESOM_SSH_SOLVER` / `FESOM_PCSI_*` / `FESOM_SPEED_CGPOLY` /
`FESOM_SPEED_CGPIPE` are inert under `FESOM_SSH_MODE=se`. A6 aborts only when such a knob is set
**explicitly**; the `FESOM_SPEED=1` master must never abort.

**Known traps carried into this campaign:**

- det fill ≈7 min at NG5/64 ranks vs 71 s compute — **setup, outside the timing window**
- NaN-blind stall guard (fixed on m10; re-verify after merge); a zombie leg measures ~10.8 % *faster*
- `snap_every=-1` at **≥**4096 ranks
- fArc ≥128 ranks proto hang — covers the entire fArc CPU ladder, resolved in C4
- `gpu` partition heterogeneous — `-C a100_80` on every absolute
- a re-rolled partition is not universally better: +8…+15 % slower at CORE2 864, +3…+8 % at fArc
- `core2hil_v1` renumbers the mesh and re-baselines every C↔K floor — excluded from M14

---

## Post-completion

*Requires external action or a user decision.*

**User decisions needed:**
- promote `dars_gpu_v1` / `ng5_gpu_v1`? (C1)
- authorise 32 GPU nodes if NG5 GPU has not rolled over at 16? (D4)
- merge `m14-integrate` to `main` once the gates are green? (E3)

**Coordination:** the M9/M10/M11/M12b worktrees stay live; this campaign shares the account.
`FESOM/fesom2#979` (upstream Fortran det-IC PR) is open and independent.

**Deferred:** M8 mixed-precision integration; deep-K SE (blocked — `FESOM_SE_WIDE` accepts K=1 only
without the extended-mesh build).

---

## Review history

**Rev 2 (2026-08-15)** — plan-review agent, 22 findings. Accepted after independent verification
against source: the ice arm was a no-op without `FESOM_WHICH_EVP=1` and used the wrong K knob; the
A6 abort would have killed every best-arm leg because CGPIPE is inside `FESOM_SPEED=1`; the
certified partitions exist at 5 rank counts against ~50 needed, and CORE2 CPU has none above 864 at
all; the CUDA byte point at np128 broke the 16-node cap; the additivity null was additive where it
must be multiplicative. Also corrected: a CORE2 efficiency table quoted in Rev 1 does not exist —
`SSH_SOLVERS_M10.md` says CORE2 has no 1024-rank partition and has **not** turned over. Cut:
NG5 32768, dars 16384, and half the interaction probes. Added: Phase 0, per-merge byte gates, the
1-year twin, the fArc hang blocker, allocation discipline, and a timing-window definition.
