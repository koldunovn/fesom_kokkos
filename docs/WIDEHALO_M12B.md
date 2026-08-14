# M12b — wide halo for the SE barotropic subcycle (`FESOM_SE_WIDE`)

**Status 2026-08-14 s3: the s2 instability is ROOT-CAUSED and FIXED (§3 — halo H0e was not the
owner's bytes; one per-step exchange, `FESOM_SE_H0E_XCHG=1` default). The rung is exact through
the first steps and its free-running drift plateaus at machine noise (§3e). Stability screens at
np128/farc-2048 and the perf board are the s3 verification wave.** Plan + full gate record:
`docs/plans/20260814-m12b-widehalo.md`. SE reference: `docs/SSH_SE_M12.md`.

## What it is

The SE block does **two halo exchanges per substep** — η over `NOD2D`, Ū over `ELEM2D` — at
M = 20…90 substeps per step, and at CORE2 16N GPU that block is 7.8 ms busy + **11.8 ms MPI wait**
of an 84 ms step. The K=1 rung removes half of them: **η is computed on the ring-1 (eDim) nodes
instead of being received**, and only Ū is exchanged, over the *existing* `com_elem2D_full` list.

Nothing new is allocated in the node index space — ring-1 nodes *are* the eDim halo slots. The
element state grows to the full local extent (`myDim+eDim+eXDim`) because `com_elem2D_full`
delivers element rings 1 **and** 2 (it replaces `com_elem2D`'s coverage rather than extending it,
which is exactly why the rung is a message *replacement*, not an extra message).

Each ring-1 node needs its div-CSR row, and that row **cannot be assembled locally at any
precision**: 40–44 % of the edges incident to a ring-1 node are absent from the local edge list
(an edge joining two ring-1 nodes touches no owned node). So the owner ships its row — the final
post-`inv_a` coefficients, in its own accumulation order — once at startup, over the `com_nod2D`
partner structure, tags 2300–2302.

## Knobs

| knob | default | notes |
|---|---|---|
| `FESOM_SE_WIDE` | 0 | `1` = the rung; **`≥2` aborts** — deep K needs the extended-mesh layer (§4). Requires `FESOM_SSH_MODE=se`; set while SE is off it prints the L80 dead-knob note |
| `FESOM_SE_H0E_XCHG` | **1** | s3 root-cause fix (§3): one per-STEP exchange makes halo `H0e` the owner's bytes (`ELEM2D_FULL` under the rung). `0` = the legacy incoherent halo, diagnostic arms only — it re-arms the seed |
| `FESOM_SE_WIDE_SELFCHECK` | 0 | `1` = per-substep compare (exchange restored — proves the ring math; aborts on nonzero, and a 3-D-born ulp reaches the ring from step ~4, §3e); `2` = FREE running + per-step drift report (the stability instrument) |
| `FESOM_SE_WIDE_GEOCHK` | off | `1` = the s2 probes (holder-vs-holder, x-component); `2` = the s3 ingredient probes: BOTH components, `viscM`/`viscN` separately, halo-copy-vs-owner scope, `He`/`meta` decomposition, offender dump — the instrument that found §3 |

Startup aborts, both from measured assumptions rather than belief: an owned element whose
viscosity neighbour lands in the eXDim tail, and any `com_elem2D_full` rlist that fails to cover
`[myDim, myDim+eDim+eXDim)` exactly once.

## 1. What it costs and what it buys (census, 11 operating points)

`scripts/m12b_ring_census.py`, jobs 26949302 / 26951957; CORE2, farc, dars, NG5; 16–8192 ranks.
Containment holds at **all 11 points** (0 elements incident to a ring-1 node missing locally), and
the eXDim viscosity-neighbour hazard fires at **none** of them.

| point | ranks | ring-1 redundant compute | messages/substep | doubles/substep |
|---|---|---|---|---|
| NG5 16N GPU | 64 | **+1.2 %** | ×0.500 | ×1.34 |
| dars 16N GPU | 64 | +1.7 % | ×0.500 | — |
| farc 16N GPU | 64 | +2.9 % | ×0.500 | ×1.44 |
| CORE2 4N GPU | 16 | +2.0 % | ×0.500 | ×1.45 |
| CORE2 16N GPU | 64 | +6.1 % | ×0.500 | ×1.46 |
| NG5 CPU | 8192 | +13.8 % | ×0.500 | — |
| farc CPU | 2048 | **+19.7 %** | ×0.503 | ×1.49 |
| dars CPU | 8192 | **+28.2 %** | ×0.505 | ×1.41 |

**Messages halve because `com_elem2D_full` reaches the same partner ranks as `com_elem2D`** — more
entities per partner, not more partners. That is a claim about `rPEnum`, so it was counted, not
assumed. The GPU/CPU split is Sergey's "GPU real advantage, CPU just better scaling" as a number.

## 2. Rim algebra (what deep K would cost)

With Ū valid to element ring E and η to node ring D: η^{m+1} is valid wherever its incident
elements are (D′ = E), and Ū^{m+1} loses one ring to the viscosity's edge-neighbour term
(E′ = E−1). **K substeps therefore need K rings, not the 2K−1 of mEVP** — mEVP carries element σ
across substeps, SE's η/Ū alternation does not. Cumulative node zone / owned:

| point | K=2 | K=4 | K=8 |
|---|---|---|---|
| NG5 64 GPU | 0.02 | 0.05 | **0.10** |
| CORE2 64 GPU | 0.13 | 0.30 | 0.66 |
| farc 2048 CPU | 0.44 | 1.01 | 2.40 |
| dars 8192 CPU | 0.59 | 1.30 | 3.02 |

Deep K is cheapest where the mesh is **biggest per rank** — nearly free at NG5/dars GPU, hopeless
at the big-CPU points.

## 3. 🔴 ROOT CAUSE (s3, 2026-08-14): halo `H0e` was not the owner's bytes — FIXED

**The s2 story below ("not bitwise by module property") is superseded.** The seed was found,
measured to the byte, and removed:

1. `se_forcing` computes `H0e` locally over owned+eDim on every rank. At halo elements that
   computation was **η-class WRONG**: an eDim *edge-neighbour* element's far vertex is routinely a
   ring-2 node, absent from the local node list, and the scatter stores `elem_nodes = -1` there —
   so the η-mean read `eta0[-1]`, out of bounds. Measured (GEOCHK=2, jobs 26959682/26959760):
   1334 halo copies differ from the owner's value by up to **1.0e-1 m**, and the decomposition is
   unambiguous — `dHe = 0` for every offender, `dmeta` carries the whole difference. At step 1 the
   error is invisible because H0e is η-independent at cold start (the Z7 shape, for the third time).
2. k3's viscosity reads `H0e[nb]` **at halo slots**. Two holders of a multi-claimed element whose
   shared neighbour resides in the halo on one of them therefore evaluated V with different `hh` —
   and the difference expresses exactly when the two viscosity terms stop cancelling: the first
   nonzero V[Ūᵐ]−V[Ūⁿ] with a nonzero anchor, step 2 substep 1. Probe-verified: at step 2
   substep 0 `viscM` and `viscN` carry **identical** spreads (max 5.0e-10) that cancel exactly —
   Unew stays coherent; from substep 1 they no longer cancel and the copies diverge.
3. The free-running rung then amplified the η-class interface inconsistency (~1.2×/substep) into
   the farc NaN of §3d.

**Fix (`FESOM_SE_H0E_XCHG`, default 1): one per-STEP H0e exchange** — halo H0e becomes the
owner's bytes; the local recompute at halo slots was never legitimate in this data model (its
vertices are not all local). Under the rung the exchange runs over `ELEM2D_FULL`, which also fills
the eXDim tail — precisely the remedy the startup guard (1) had prescribed for a different
trigger. The `eta0[-1]` read is additionally guarded (`-1` vertex → 0.0 contribution; owned
elements are assert-mapped at scatter, so the certified owned rows are untouched).

`Fbt` halo copies are garbage-class (uninitialized `uv_rhs` reads) — **dead state**, k3 reads F
at owned elements only; documented, deliberately not exchanged.

**Fix part 2 (`FESOM_SE_WIDE_RECON`, default 1, rung-only): owner-wins F over the multi-claimed
elements.** The H0e fix alone is NOT sufficient, and the reason is the deepest measurement of the
track: with the seed cut by nine orders (5.2e-17 at step 2 instead of 3.0e-8), the free-running
drift still grew **exponentially — ×5.35 per STEP at farc 2048 (M=90), ×1.056 per step at CORE2
np128 (M=50)** — reaching 49 m by step 30 at farc (jobs 26959819/26959818). The free-running
interface iteration is linearly unstable: *any* rank-inconsistency near the interface is
integrated by the barotropic dynamics, so the seed must be exactly zero, not merely small. The
only remaining inconsistency channel (probe-verified, job 26959826) is `Fbt` at the multi-claimed
elements — the 3-D model's own last-bit redundancy (one element, one ulp, first at step 4). One
tiny per-step wave (tag 2303; discovery = one GE-byte claim-count allreduce + an allgatherv of the
~0.55 % lists at startup) makes F single-valued; with H0e AND F owner-coherent every k3 input is
single-valued, the redundant Ū copies stay **bitwise-locked by induction**, and the locally
computed ring-1 η IS the owner's bytes — the free rung has nothing to amplify, at any rank count.

**What survives of the s2 finding:** `myDim_elem2D` is still not a partition (1341 of 244659
CORE2 dist_8 elements multi-claimed, unreconciled) — but the redundancy was the *site* where the
incoherence became observable, not its origin. With owner-coherent inputs the redundant copies
stay bitwise-locked (measured through step 3), so "no local recomputation can be bitwise" is
**retracted**; what remains is §3e's 3-D-born ulp.

The s2 hypothesis-elimination table, all rows still valid measurements (they cleared everything
except the one array no probe compared halo-vs-owner):

| question | answer |
|---|---|
| shipped CSR row / its order right? | **exactly 0 through all 50 substeps of step 1** |
| the viscosity? | `FESOM_SE_VISC=0` → **0 at every substep of the run** |
| the neighbour summation order? | canonicalised by global id → no change, to the last digit |
| the geometry? | `elem_area` differs for **0 of 244659** elements across holders |
| a dropped neighbour? | stencil size differs for **0 of 244659** |
| the forcing? | `Fbt`/`H0e` identical at steps 1 **and** 2 |
| Ū itself? | step 1 END **0**; step 2 END **529 elements, max 5.2e-07** |

*(The s2 residual curve — 6.1e-10 at step 2 → 2.4e-6 m saturating at step 200, job 26952789 —
was this seed's growth under the per-substep exchange. It is obsolete: the fixed module's residual
is §3e's ulp plateau, ~9 orders smaller.)*

### 3e. The intermediate state (H0e only): a 3-D ulp seed, and the exponential that forced part 2

With H0e fixed and F NOT yet reconciled, the rung was exact — 0.0 — through step 3 everywhere,
and at step 4 a fresh seed entered from OUTSIDE the SE module: `Fbt` differing across holders of
ONE multi-claimed element by 2.2e-19/4.3e-19 (x/y — one ulp of the 3-D vertically-integrated RHS)
**before any SE state had diverged** (job 26959826). What the free rung then did with that ulp is
the decisive measurement of the whole track:

| free-running max\|local − owner\| | steps 1-3 | 10 | 25/30 | 300 | growth law |
|---|---|---|---|---|---|
| np8, M=50 (26959760) | 0.0 | 1.1e-15 | 1.2e-15 | — | flat (plateau) |
| np128, M=50 (26959818) | 0.0-ulp | ~1e-15 | ~2e-15 | **1.7e-10** | **×1.056/step** |
| farc 2048, M=90 (26959819) | 0.0 → 5.2e-17 | 6.1e-13 | **49 m @30 → NaN** | all-NaN | **×5.35/step** |

Nine orders of seed reduction moved the farc NaN by ~15 steps: the growth is a property of the
scheme (rank-dependent η near the interface, integrated by the barotropic dynamics; rate rises
with the interface fraction), so **no seed reduction suffices — the seed must be exactly zero.**
That is what fix part 2 (§3) delivers, and why the s2 estimate "1.2×/substep" is superseded: the
true per-substep factor is 5.35^(1/90) ≈ 1.019 at farc, and what matters is the per-STEP compound.

*For Sergey: two findings worth reporting independently of M12b — (a) halo `H0e` in the SE module
was locally recomputed from unmappable vertices (an out-of-bounds `eta0[-1]` read, η-class wrong;
fixed by the per-step owner exchange); (b) the 3-D model's `myDim_elem2D` redundancy leaves
per-element fields (uv_rhs-derived) different at the last bit across claimants — harmless in the
exchanged path, but it seeds ANY compute-instead-of-communicate transformation, and the free
barotropic interface amplifies whatever it is seeded with, exponentially per step.*

### 3f. s3 verification board — the complete fix (H0e + F-reconcile), bin `90c5c12d`

| gate | point | result | job |
|---|---|---|---|
| inertness null (knobs off, XCHG=0) | np8, 50 steps vs `base_np8` | **diff_snap rc=0** | 26959981 |
| M12 G1 invariants, fixed certified path | np64, 100 steps | G1a 1.6e-14 m · G1b 4.5e-13 m²/s | 26959981 |
| legacy arm (both halves off) | np8 | reproduces the break (abort @2) | 26959980 |
| W1 selfcheck=1, 0.0-abort armed | np8, 25 steps | **EXACT 0.0, every substep** | 26959980 |
| free-running drift | np8 ×25 · np128 ×300 · farc 2048 ×50 | **0.000000e+00 at every step** | 26959980/81/82 |
| W5 3000-step screen, rung ON free | farc 2048, M=90, wsplit | **clean; η=2.05/uv=2.01 ≈ control η=2.05/uv=2.00** | 26959982 |
| W5 3000-step screen | CORE2 np128 | in queue | 26960088 |
| W5b graded disturbance | np64/np128 | in queue | 26960089 |
| W2 CUDA drift gate | CORE2 4N GPU | in queue (partition drained) | 26960090 |

**The rung is bitwise-exact free-running** (its trajectory ≡ the same binary's exchanged path with
the reconcile active); knob-on vs knob-off is a rounding-class pair (the F-reconcile is rung-only
by design, so the certified path stays byte-frozen vs its own history under `XCHG=0`).

## 3b. Disturbance report — the lever against the model's own rank-dependence (W5b, PASS)

Job 26953179, CORE2, 20 steps, `scripts/m12b_disturbance.py`. The residual is a rank-ordering
effect, so it is judged against the spread the model already shows when only the rank count
changes — the same perturbation class, and the standard the project accepts (L79).

🔴 **Paired at the same rank count.** An arm run at a *different* rank count from the reference
differs by the lever AND by the rank count, and the second term dominates by orders of magnitude:
the first version of this report compared on@np128 with off@np64 and called it "outside the
spread" when it was in fact measuring the rank-count change. Each lever row below is knob-off vs
knob-on at ONE rank count; the controls are knob-off runs at other rank counts.

| field | control spread (rms) | **lever** rms @np64 | @np128 | lever vs smallest control |
|---|---|---|---|---|
| `eta_n` | 3.2e-3 … 1.2e-2 | 7.88e-7 | 7.80e-7 | **4087× smaller** |
| `T` | 2.5e-2 … 4.5e-2 | 6.67e-7 | 7.97e-7 | 37368× |
| `S` | 2.3e-2 … 2.6e-1 | 2.22e-7 | 2.92e-7 | 103165× |
| `u` / `v` | 1.4e-3 … 2.4e-3 | 4.7e-7 / 4.4e-7 | 4.4e-7 / 4.6e-7 | ~3000× |
| `w` | 8.4e-6 … 1.7e-5 | 3.99e-10 | 3.79e-10 | 21112× |
| `m_ice` / `a_ice` | 1.7e-2 … 2.4e-2 | 8.4e-6 / 7.2e-6 | 8.5e-6 / 7.3e-6 | ~2200× |

**rc=0 on every field**, and the two rank counts agree with each other to within a few per cent —
the lever's footprint is a property of the transformation, not of the partition it ran on. In the
M11 grading this is comfortably tier 1: the winner is not distinguishable from a seed re-roll,
by three to five orders of magnitude.

## 3c. 🔴 RETRACTED performance number, and the failure it exposed (W5/W6)

**The farc 2048 CPU result of −8.9 % (0.0723 → 0.0659 s/step, job 26953153) is WITHDRAWN: the
knob-on legs ran to all-NaN.** The diagnostic print shows `T[1e30, -1e30]` with every norm at
zero — the signature of an all-NaN field, because NaN comparisons leave a min/max reduction's
sentinels untouched. A NaN run's wall-clock is not a performance measurement. The same failure
appears in the W5 3000-step screen: knob-off reaches step 3000 at η=1.89 (the SE reference),
knob-on is NaN by step 500.

**Why every earlier run looked healthy — the trap worth naming.** Every clean wide run to that
point was either 20 steps, or ran with `FESOM_SE_WIDE_SELFCHECK`, **which performs the η exchange
as well** (stash → exchange → diff). The selfcheck therefore restores the certified data path each
substep, so it validates the ring COMPUTATION while hiding what happens when the rung runs free.
The residual growth curve in §3 was measured in exactly that configuration: it is a faithful
measure of the per-substep seed, and NOT of free-running drift. 🔴 **A diagnostic that repairs the
thing it measures cannot certify it — run the lever free before believing any of its numbers.**

**Free-running behaviour, measured:** CORE2 np8 is healthy past step 260 (η ~3.1, indistinguishable
from knob-off); CORE2 np128 is NaN before step 500; farc 2048 is NaN by step 300. So the failure is
**rank-count dependent**, which fits the interface fraction (ring-1 is 6.1 % of owned at np64,
13.2 % at np128, 19.7 % at farc 2048).

**The hypothesis was wrong, and the measurement said so** (job 26953774): the `VISC=0` arm blew up
too (step ~110 vs ~90 for the viscous arm) while the knob-off control ran clean to step 300. And at
np128 with `VISC=0` the selfcheck showed the ring compute is exact **to one ulp** (2e-19 at step 2,
3.9e-16 by step 40). So the residual was never the cause.

### 🔴 The actual cause, and the fix: "halo == the owner's bytes" is an invariant the model relies on

Inside the subcycle it does not matter that our ring-1 η is the owner's value only to within a ulp
— it feeds the barotropic loop and nothing else. But at the end of the step **η leaves the
module**: finalize writes `hbar` over the full node extent, substep 11 derives `eta_n`, and the 3-D
model then consumes η **at halo nodes** (ALE layer thicknesses, level masks, wet/dry decisions).
Those consumers assume a halo entry is a byte-copy of the owner's value — which an exchange
guarantees and a local recomputation does not. A one-ulp disagreement is enough for two ranks to
reach different *structural* decisions about the same node.

**Fix: one η exchange per STEP** (not per substep), restoring the invariant where it matters.
At M=50 the rung becomes 100 → **51** exchanges rather than 100 → 50, so essentially all of the
saving survives.

**Result at CORE2 np128, 300 steps** (job 26953980): the rung now runs clean and ends at
`η=2.02, uv=1.47` — **identical to the knob-off control at the same step**.

✅ The `VISC=0` control came back (job 26954124): **knob-OFF with `VISC=0` blows up in the same
step range (200–250)**, so that instability belongs to the certified scheme without its viscosity,
not to the rung. Using `VISC=0` as a "provably exact" diagnostic arm without first checking that
the certified path survives it was a mistake — the control should always precede the diagnostic.

### 🔴 3d. [s2 verdict, RESOLVED by §3] The rung was UNSTABLE, and the free-running drift is what showed it

**Status 2026-08-14 s3: the instability below was the §3 H0e seed under amplification, and is
FIXED — the fixed rung's drift plateaus at machine noise (§3e; farc re-measurement in §3f). The
section is kept as the record of the measurement that forced the root-cause hunt.**

The per-step coherence exchange fixed CORE2 np128 (300 steps, matching the control) but **not
farc 2048, which is still NaN by step 300.** The instrument that had been missing all along:
measure the drift of the FREE-RUNNING rung, by stashing what M substeps of local computation
produced and letting the per-step exchange deliver the owner's values on top
(`FESOM_SE_WIDE_SELFCHECK=2`; job 26954275, farc 2048, M=90).

| step | 1 | 2 | 3 | 10 | 20 | **30** | 40+ |
|---|---|---|---|---|---|---|---|
| free-running max\|local − owner\| (m) | 0 | 3.0e-8 | 6.9e-7 | 3.2e-6 | 9.3e-6 | **1.15e+02** | NaN |

**The scheme amplifies.** After the 90 substeps of step 2 the drift is 3.0e-8, where accumulating
ulp-level per-substep seeds would give ~1e-14 — six orders smaller. That is growth of roughly
**1.2× per substep** of an interface perturbation. The seed is irreducible (the §3 redundant
elements), the amplification is the scheme's, and together they reach 115 m by step 30.

**Consequence.** Replacing the η exchange entirely removes the coupling that keeps neighbouring
subdomains' barotropic solutions locked to one another; each side then integrates its own version
of the interface and the difference grows. The rung is therefore **not viable in this form**, and
the two options that follow from the measurement are:

1. **Reconcile the redundantly-computed elements** so Ū is globally consistent and the seed is
   exactly zero — the option deferred earlier as optional cleanup, which the evidence now makes a
   *prerequisite*. With no seed, the amplification has nothing to act on.
2. **Exchange η every k substeps rather than never** — the design space the rung occupies at
   k = ∞. With the measured growth rate ~1.2×/substep, a small k keeps the drift at ulp level;
   k=2 would still remove 25 % of the exchanges. The growth rate is now measurable per point, so
   the largest viable k can be derived rather than guessed.

*(s3 outcome: neither option was needed. The seed was not the redundancy itself but the §3
halo-H0e incoherence feeding it; with that fixed, the redundant copies stay locked on their own
and the k-periodic fallback is unnecessary. The measured amplification rate stands — it is what
turns any systematically re-injected interface error into a blow-up, and it is why the H0e fix is
load-bearing rather than cosmetic.)*

⚠️ Also fixed here: `SELFCHECK=2` previously still took the per-substep compare path, which
re-runs the exchange — so it too repaired what it measured, and a farc run under it looked healthy
to step 100. **That is the same trap twice in one session**: a wide-halo diagnostic must be checked
for whether it restores the exchange before its verdict is believed.

⚠️ `FESOM_SPEED_PHASESTATS` resolves **OFF on the Serial backend** without `FESOM_SPEED_FORCE_SERIAL=1`
(rule 0.24; the guard announces it rather than producing an empty report), so phase attribution
rides its own legs and the timing legs stay lever-free.

## 4. The extended-mesh contract (design of record; not built)

The rule that makes deep K and higher-order advection share one layer: **extended entities are
appended in BFS ring order**, so "everything within ring r" is the contiguous range `[0, N_r)` and
a consumer expresses rim shrink as a `RangePolicy` bound — no indirection, no ghost kernel, no
extra launch (L109; M9's LEAN writing beat its separate-kernel twin by ×1.8–5 for this reason).
Shipped as owner bytes, never recomputed: `elem_nodes` as local ids, `gradient_sca`, `elem_area`,
metric factor, coriolis, H0e/depth, `ulevels`/`nlevels`, and per-node operator rows in owner order.
Build by BFS at scatter time (`fesom_evpwide_mesh_hook` precedent). SE would take ring values +
operator rows + a window scheduler; higher-order advection takes ring values + geometry + vertex
maps once per step, no scheduler. The certified ice EVPWIDE does **not** migrate onto this.

## Open items (s3)

W6 CPU pairs on the fixed bin `90c5c12d` — farc 2048 (26960156) + dars 8192 (26960157), in queue ·
CORE2 np128 3000-step screen (26960088) + W5b disturbance (26960089), in queue · W2 CUDA drift
gate (26960090) + W6 GPU pairs — behind the drained GPU partition · per-step wire is now M+3
(subcycle + η coherence + H0e + F-reconcile) vs the certified 2M · deep-K decision on the W6
numbers — note the s3 induction transfers to deep K only if ring-K inputs are owner-coherent the
same way (H0e over the K-ring extent, F still reconciled; the §4 extended-mesh layer ships owner
bytes by construction, which is compatible).
