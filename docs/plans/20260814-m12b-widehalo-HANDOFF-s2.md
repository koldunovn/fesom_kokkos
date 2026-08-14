# M12b wide halo — HANDOFF after session 2 (2026-08-14 evening)

**You are a fresh session in the worktree `~/port_kokkos_wh`, branch `m12b-widehalo`.**
🔴 Your auto-memory index is worktree-scoped — read the MAIN index first:
`/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/MEMORY.md`, then
[[project-m12b-widehalo]].

**Read in this order:** this file → `docs/WIDEHALO_M12B.md` (reference, all numbers) →
`docs/plans/20260814-m12b-widehalo.md` (plan + gate ladder). The **session-1 handoff**
(`…-HANDOFF.md`) and the design sketch in it are **superseded**; where they differ from these
files, these files won.

---

## 1. State of play in one paragraph

The K=1 rung is **built, committed, and works exactly as designed** — it halves the per-substep
exchanges, its ring computation is provably correct, and every structural gate is green. **It is
also numerically unstable, and that is now measured, not suspected.** Removing the η exchange
removes the coupling that keeps neighbouring subdomains' barotropic solutions locked together; the
scheme then amplifies interface perturbations at ~1.2× per substep, and the model dies (farc 2048
by step 30 of drift ⇒ NaN by 300; CORE2 np128 survives to ~2000-2500). **No performance number
from this session survives** — both farc timings came from NaN runs and were withdrawn. The track
needs a design decision (§6) before any further measurement is worth the queue time.

---

## 2. What is CERTIFIED and must not be re-derived

Everything here is measured, with job ids. Do not spend queue time re-establishing it.

**Census over 11 operating points** (`scripts/m12b_ring_census.py`, jobs 26949302 / 26951957;
CORE2 16/64/128, farc 64/2048, dars 64/2048/8192, NG5 64/4096/8192; raw output
`/work/ab0995/a270088/port2/m12b/census.*.out`):

- **Containment 0-missing at all 11 points** — `eDim_elem2D + eXDim_elem2D` *is* element ring 2, so
  `com_elem2D_full` is exactly the list the rung needs.
- **`com_elem2D_full` REPLACES `com_elem2D`'s coverage**, spanning `[myDim, myDim+eDim+eXDim)` —
  both halo rings in one message. That is why messages *halve* rather than grow.
- **Messages/substep ×0.500–0.505 everywhere** (counted via `rPEnum`, not assumed); doubles ×1.34–1.49.
- **Ring-1 redundant compute +1.2 %** (NG5 16N GPU) … **+28.2 %** (dars 8192 CPU).
- **40–44 % of the edges incident to a ring-1 node are NOT in the local edge list** ⇒ the owner's
  div-CSR row *must* be shipped; it cannot be rebuilt locally at any precision.
- **The SE rim shrinks ONE ring per substep, not mEVP's 2K−1** ⇒ K substeps need K rings, and deep
  K is cheapest where the mesh is biggest per rank (K=8 node zone: 0.10 NG5 64 GPU, 0.66 CORE2 64
  GPU, 2.40 farc 2048 CPU, 3.02 dars 8192 CPU).
- `elem_nb` eXDim hazard: fires at **none** of the 11 points (guarded by a collective abort anyway).

**Gates that passed:**

- **W0 knob-off byte null: PASS** at np8 and np64 (26952236 / 26952349) — the full-extent
  allocation and everything else in Task 2 is inert.
- **W3 wire observable:** `exchanges/substep 2 -> 1`, `partners ×0.500`, and (after the fix) the
  per-step line reads `M + 1` exchanges.
- **Ring computation is correct:** the per-substep selfcheck is **exactly 0.0** through all 50
  substeps of step 1, and at np128 with `VISC=0` it stays exact **to one ulp** (2e-19 at step 2 →
  3.9e-16 at step 40). The shipped CSR rows, their owner order, and the `ELEM2D_FULL` transport are
  all right. ⚠️ **This does not certify the rung** — see §4.
- **W5b disturbance report: rc=0**, lever 2200–103000× below the model's own rank-count spread on
  eight fields at two rank counts (26953179). ⚠️ **Caveat: 20 steps only, i.e. before the
  instability manifests.** It says the first 20 steps are benign, nothing more.

---

## 3. 🔴 The verdict: the rung is unstable, and here is the measurement

Free-running drift, farc 2048, M=90 (`FESOM_SE_WIDE_SELFCHECK=2`, job 26954275) — stash what M
substeps of local computation produced, let the per-step exchange deliver the owner's values on
top, report the gap:

| step | 1 | 2 | 3 | 10 | 20 | **30** | 40+ |
|---|---|---|---|---|---|---|---|
| max\|local − owner\| (m) | 0 | 3.0e-8 | 6.9e-7 | 3.2e-6 | 9.3e-6 | **1.15e+02** | NaN |

**The key line is step 2.** After 90 substeps the drift is 3.0e-8, where accumulating ulp-level
per-substep seeds would give ~1e-14 — six orders smaller. So the scheme **amplifies** an interface
perturbation at roughly **1.2× per substep**.

Two ingredients, both measured:

1. **The seed** — `myDim_elem2D` is not a partition. **1341 of 244659 CORE2 dist_8 elements
   (0.55 %)** sit in the `myDim` range of more than one rank, are computed redundantly, and are
   reconciled by no exchange (rlist covers only `[myDim, …)`). Ū across holders: step 1 END
   identical, **step 2 END 529 elements differ, max 5.2e-07**. The certified path hides this
   entirely because η is exchanged and the node owner's copy wins.
2. **The amplification** — the scheme's own, established by the step-2 arithmetic above.

**Partial fix already committed** (`0ab0595`): one η exchange per **step** (not per substep) to
restore *"halo == the owner's bytes"* before η leaves the module — finalize writes `hbar`, substep
11 derives `eta_n`, and the 3-D model consumes η at **halo** nodes for ALE thicknesses, level masks
and wet/dry decisions, which assume a byte-copy of the owner's value. It costs 1 message where the
certified path spends M (so 100 → **51** at M=50, not 100 → 50).

**It delays but does not cure:**

| point | without the per-step fix | with it |
|---|---|---|
| CORE2 np128 | NaN before step 500 | clean to 300 (matches control exactly), **dead by 2500** (healthy at 2000; job 26954122) |
| farc 2048 | NaN by 300 | **still NaN by 300** (job 26954123) |

---

## 4. 🔴 Traps this session paid for — read before designing an experiment

1. **A diagnostic that repairs what it measures certifies nothing. This caught me TWICE.**
   `FESOM_SE_WIDE_SELFCHECK=1` performs the η exchange *as well as* the local compute, so every run
   under it silently runs the certified data path. Every "healthy" wide run for hours was either
   20 steps or selfcheck-instrumented. Then the same trap again one level down: `SELFCHECK=2`
   *also* took the per-substep path until commit `e2a3b34`, so a farc run under it looked healthy
   to step 100. **Before believing any wide-halo diagnostic, check whether it restores the
   exchange.** Current semantics (correct): `1` = per-substep compare (exchange restored, proves
   the ring math), `2` = FREE running + per-step drift report (proves nothing is repaired).
2. **Run the control before using a configuration as a diagnostic.** I used `FESOM_SE_VISC=0` as
   the "provably exact" arm for many jobs before checking whether the *certified* path survives
   without its viscosity. It does not: knob-OFF with `VISC=0` blows up at np128 in the same 200–250
   step range (job 26954124). Those arms said nothing about the rung.
3. **A build-tree path is not a pin.** I rebuilt `build-m7serial/fesom_port` while a gate job was
   running against that path; `srun` re-execs per leg, so its legs could run different binaries.
   Freeze hash-named copies to `/work/…/m12b/bin/` and pin those.
4. **Announce before the guards**, or a guard abort is indistinguishable from a knob that never
   fired — my job script misreported exactly that.
5. **`FESOM_SPEED_PHASESTATS` resolves OFF on the Serial backend** without `FESOM_SPEED_FORCE_SERIAL=1`
   (rule 0.24). Phase attribution needs its own legs; keep timing legs lever-free.
6. **Check the run's own state before quoting its wall-clock.** Both farc timings I reported came
   from all-NaN runs. The signature is `T[1e30, -1e30]` with every norm zero — NaN comparisons
   never displace a min/max reduction's sentinels. `grep " it=" run.log | tail -1` costs nothing.
7. My first disturbance report compared knob-on@np128 against a knob-off@**np64** reference, which
   measures the rank-count change with the lever buried in the last digits. **Pair the lever at one
   rank count**; the controls supply the bar.

---

## 5. Where everything lives

- **Code:** `src/fesom_ssh_se.{h,cpp}`. Knobs `FESOM_SE_WIDE` (0/1; ≥2 aborts),
  `FESOM_SE_WIDE_SELFCHECK` (1 = per-substep compare, 2 = free + per-step drift),
  `FESOM_SE_WIDE_GEOCHK` (min/max-over-holders probes for `elem_area`, stencil size, `Fbt`/`H0e`,
  `Ubt` — the instrument that found the redundancy).
- **Jobs:** `jobs/job_m12b_{census,baseline,wgate,probe,screen,w5b_disturb,w6_cpu,w6_gpu,w0_msweep}`.
- **Scripts:** `scripts/m12b_ring_census.py` (census + partners + local-edge coverage),
  `scripts/m12b_disturbance.py` (paired-at-fixed-rank-count graded report).
- **Docs:** `docs/WIDEHALO_M12B.md` (reference), `docs/plans/20260814-m12b-widehalo.md` (plan),
  `docs/SSH_SE_M12.md` (now carries the rank-redundancy caveat for the M12 module itself).
- **Bins** (`/work/ab0995/a270088/port2/m12b/bin/`, shas in `SHA256.m12b`):
  `fesom_port_serial_2991beb5` = **current** (per-step coherence fix + correct SELFCHECK semantics);
  `fesom_port_cuda_2997d09b` = CUDA at the per-step-fix commit, **built before** the SELFCHECK fix;
  earlier serial bins are the intermediate debug states.
- **Run dirs:** `/work/ab0995/a270088/port2/m12b/`.
- **Baseline oracle:** `base_np8` / `base_np64` (SE, 50 steps, snaps at 0/25/50), made with the
  pre-change binary `5d2fec66` at git `c926ab3` — still valid for W0.

**Queued:** `26952126` / `26952127` (M-sweep on the frozen M12 `se0` bins, GPU, behind the
maintenance window) — these measure the *baseline's* latency structure (does bt wait track exchange
count) and remain useful input for §6. The W6 GPU perf pairs were **cancelled**: they would have
measured an unstable configuration.

---

## 6. 🔴 The decision waiting for you, and the first experiment for each path

The user was asked at the point where the rung was known to be non-bitwise and chose *"accept
rounding-class, measure now"*. That choice was made **before** the instability was known. The
evidence has since moved, so the decision should be put again, framed as:

**Path A — reconcile the redundantly-computed elements** (make Ū globally consistent so the seed is
exactly zero). Deferred earlier as optional cleanup; the evidence makes it a *prerequisite*, since
with no seed the amplification has nothing to act on. It also removes a latent artefact from the
certified M12 module, which has value independent of M12b.
*First experiment:* build the list of multi-claimed elements (the `GEOCHK` probe already finds
them), add an owner-wins reconciliation after k3, and re-run the farc drift measurement. If the
drift stays at 0 for tens of steps, the mechanism is confirmed and the rung becomes viable.
*Risk:* it changes the certified baseline's own numbers and needs its own certification pass.

**Path B — exchange η every k substeps rather than never.** The rung is the k = ∞ limit of that
family; the certified path is k = 1. With the measured growth rate (~1.2×/substep) a small k keeps
the drift at ulp level, and k=2 still removes 25 % of the exchanges.
*First experiment:* make the exchange period a knob and run the drift instrument at k = 2, 3, 5, 10
at farc 2048. The largest viable k is then *derived*, and the census cost model already says what
each k costs.

**Path C — stop, and write up.** The negative result is genuinely valuable: the census cost model,
the rim algebra (K rings, not 2K−1), the redundancy finding, and the measured amplification rate
are all publishable inputs for whoever tries this next, and the redundancy finding is worth sending
to Sergey on its own.

⚠️ Whichever path: **the deep-K design (`docs/WIDEHALO_M12B.md` §4) inherits this instability.**
Deep K removes *more* coupling, not less, so it cannot be attempted until the seed is eliminated.

---

## 7. What NOT to do

- Do not re-run the census — 11 points, all clean, in the repo.
- Do not re-litigate whether the ring math is right; it is exact to a ulp, measured twice.
- Do not quote a timing from a run without checking its final `it=` line.
- Do not use `FESOM_SE_VISC=0` as a "clean" arm — it is unstable in the certified path too.
- Do not run CORE2 CPU scaling (user rule); CORE2 is GPU points + ≤128-rank gates only.
- Do not submit GPU perf pairs until §6 is settled — they would measure a diverging model.
