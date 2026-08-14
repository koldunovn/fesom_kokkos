# M12b — wide halo for the SE barotropic subcycle (`FESOM_SE_WIDE`)

**Status 2026-08-14 s3 COMPLETE on CPU: the s2 instability is ROOT-CAUSED and FIXED by two
coherence repairs (§3: per-step H0e exchange + owner-wins F over the multi-claimed elements). The
free-running rung is BITWISE-exact (drift ≡ 0.0 — np8 ×25, np128 ×300, farc-2048 ×50), both
3000-step screens are green (farc η=2.05, np128 η=1.89 = the SE references), W5b rc=0, and the W6
CPU board reads farc −1.8 % / dars +0.4 % (§5) — the CPU latency share bounds the payoff, as
Sergey predicted. GPU (W2 gate + W6 pairs) queued behind the maintenance window.**
**s4 (2026-08-14 late): the `elem_nodes(-1)` hygiene item is CLOSED — 45 read sites audited, one
real defect fixed, the containment invariant now measured at startup and green at seven points
(§6) — and the deep-K partner growth is MEASURED (§7): flat in K on GPU-sized subdomains,
+42…74 % at the big-CPU points. The GPU rows did not move (the account's 5 GPU job slots were
held all session).** Plan + full gate record: `docs/plans/20260814-m12b-widehalo.md`. SE
reference: `docs/SSH_SE_M12.md`.

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
| W5 3000-step screen, rung ON free | CORE2 np128 | **clean; η=1.89 both arms (the SE reference), print-precision match** (s2: NaN by 500) | 26960088 |
| W5b graded disturbance | np64/np128 | **rc=0 at BOTH rank counts; lever 2200-2400× below the rank-count spread** | 26960089 |
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

## 5. s3 W6 board (fixed bin `90c5c12d`, same-day pinned pairs, min-of-2, 300 steps, loop-only)

| point | off s/step | on s/step | Δ | mechanism (phasestats) |
|---|---|---|---|---|
| farc 2048 CPU dt900 M=90 wsplit | 0.0731 | 0.0718 | **−1.8 %** | bt mpi/step 182→94 (halved, as designed), bt wait 5.2→4.7 ms; bt is only ~16 % of the 73 ms step here — the latency share bounds the CPU payoff (Sergey's "on CPU it is just better scaling", now as a number). Job 26960156, all four legs healthy (η=3.47, T sane), reps within 0.4 % |
| dars 8192 CPU dt120 M=20 wsplit | 0.0976 | 0.0980 | **+0.4 % (wash)** | bt mpi/step 42→24, bt wait 3.1→2.8 ms — mechanism intact, but bt is ~6 % of a 98 ms step and the +28.2 % ring-1 redundant compute eats the rest, exactly as the census predicted. Job 26960157, legs healthy (η=4.08), off-rep spread 1.5 % > the delta |
| CORE2 16N GPU / NG5 16N GPU | — | — | behind the drained gpu partition | the motivating points: CORE2 16N bt = 7.8 ms busy + **11.8 ms MPI wait** of an 84 ms step |

The s2 farc "−8.9 %" remains RETRACTED (all-NaN legs); −1.8 % is the honest CPU number.

## 6. s4: the `elem_nodes(-1)` audit — one real defect, and the invariant is now checked

The s3 hygiene item is closed. All **203 references / 45 read sites** of `elem_nodes` across 28
files were audited (enclosing loop bound + guard, then every bound resolved by hand). The
result is a rule with one exception:

> Every consumer in the 3-D model, the ice and the IO loops either over **owned elements** or
> over **`nod_in_elem2D` rows of owned nodes**. Both are `-1`-free — provided an element
> incident to an owned node has all three vertices in the local node list.

That proviso is an assumption about the partition, so `fesom_mesh_compute_metrics` now
**measures** it at startup (`audit_elem_nodes_unmappable`): a per-class census of the
unmappable refs, and a hard abort if any owned-node CSR row reaches an element with a `-1`
vertex. One pass over the elements, one over the owned rows.

| point | `-1` refs at eDim | at eXDim | at owned elements | reachable from owned `nod_in_elem2D` rows |
|---|---|---|---|---|
| pi np2 (login) | 35 | 94 | 0 | **0** |
| CORE2 np8 | 1 335 | 3 204 | 0 | **0** |
| CORE2 np128 | 15 483 | 37 802 | 0 | **0** |
| farc np128 | 37 604 | 89 518 | 0 | **0** |
| NG5 np128 | 130 684 | 265 773 | 0 | **0** |
| farc 2048 | 128 585 | 321 078 | 0 | **0** |
| dars 8192 | 858 207 | 1 902 795 | 0 | **0** |

Jobs 26961249 (CORE2/farc/NG5 legs, 1 node), 26961257 (farc 2048, 16 nodes, 0.08 node-hours),
26961258 (dars 8192, 64 nodes). The two np128 legs on the other mesh families died *after* the
census, for reasons that belong to the leg and not to the code: farc at dt900 hit the SE
startup CFL guard (the job carried CORE2's `FESOM_SE_M=50`; the guard wants ≥82 there — L113
doing its job), and NG5 at 128 ranks on ONE node was OOM-killed. Both printed the census first,
which is what those legs were for.

The eDim column is the s3 finding at scale: the unmappable refs are **not** an eXDim-tail
curiosity — at farc 2048 the first halo ring carries 129 k of them, at dars 8192 858 k. The
scatter message that said "(outer eXDim ring)" is corrected.

Same job also re-ran the rung free at CORE2 np128 on the new binary: **drift nonzero-steps = 0**
(25 steps), and the np8 inertness null is `diff_snap rc=0` against `base_np8`.

**The one defect the audit found:** `fesom_forcing_analytical.cpp` looped the FULL element
extent and read `geo_coord_nod2D` at a `-1` vertex — the identical defect class to §3's, in the
analytical-forcing path. It now averages the mappable vertices only; at owned elements the
arithmetic is unchanged bit for bit, and the pi np2 gate (which runs analytical forcing) is
**bit-identical to the frozen s3 binary `90c5c12d`**. A comment was added at the EVPWIDE
adjacency build, whose `n < myDim` filter also admits `-1` and is safe only because its extent
is owned-only. Bin `4cc9eda4`; lessons L114–L117.

## 7. Deep K — the decision arithmetic (s4), and the one measurement it still needs

Two curves decide deep K, and only one of them was known.

**Curve 1 — messages/step. The K=1 halving rests on a partner-count claim, so the K≥2 version
was measured too** (`scripts/m12b_partner_growth.py`, job 26961509: BFS rings from each sampled
rank's owned set, counting the distinct other-rank owners of each cumulative zone — node owner
from the owned block of `my_list`, element owner = lowest-ranked claimant, since `myDim_elem2D`
is not a partition). The answer is the one the rung needed:

| point | ranks | partners, ring 1 | ring 2 | ring 5 | ring 9 |
|---|---|---|---|---|---|
| CORE2 4N GPU | 16 | 3.4 | 3.4 | 3.5 | 3.5 |
| CORE2 16N GPU | 64 | 4.5 | 4.5 | 4.6 | 4.9 |
| farc 16N GPU | 64 | 5.9 | 5.9 | 5.9 | 6.0 |
| NG5 16N GPU | 64 | 6.4 | 6.4 | 6.4 | 6.4 |
| dars 16N GPU | 64 | 6.2 | 6.2 | 6.2 | 6.2 |
| farc 2048 CPU | 2048 | 5.5 | 5.5 | 6.1 | 7.8 |
| dars 8192 CPU | 8192 | 7.6 | 7.9 | 9.1 | 13.2 |

**On GPU-sized subdomains the partner count is flat in K** — a rank's neighbours nine rings out
are the same ranks as one ring out, because the subdomain is large compared with the ring. So
messages/step fall almost as 1/K: at CORE2 16N, ×0.530 (K=1) → ×0.280 (K=2) → ×0.166 (K=4) →
×0.104 (K=8) of the certified 2M. At big-CPU rank counts the zone reaches round the corner and
the 1/K breaks: farc 2048 grows 5.5 → 7.8 partners by ring 9 (+42 %), and **dars 8192 grows
7.6 → 13.2 (+74 %), enough that its messages/step stop falling — ×0.234 at K=4 and back up to
×0.249 at K=8.** On the big-CPU points deep K therefore fails on *both* curves, not only on
redundant compute.

**Curve 2 — redundant compute** is the rim algebra of §2 (cumulative node zone / owned): CORE2
64 → 0.06 / 0.14 / 0.30 / 0.67 at K=1/2/4/8; NG5 64 → 0.01 / 0.02 / 0.05 / 0.10; dars 64 →
0.02 / 0.03 / 0.07 / 0.14; farc 2048 CPU → 0.20 / 0.46 / 1.04 / 2.49.

**The model, and the assumption in it that turned out to be measurable already.** Write
`bt_wait(K) = bt_wait_cert × msg_ratio(K)^ε` and `bt_busy(K) = bt_busy_cert × (1 + zone_K)`.
The first version of this section took ε = 1 — the bt wait is a latency pool, halve the messages
and halve the wait — and the W6 CPU pairs turn out to measure ε directly, because they halve the
message count at unchanged compute. They do not support ε = 1
(`scripts/m12b_wait_anatomy.py`, per-rank phasestats of jobs 26960156/57; the counter is
*exchanges* per step, each fanning out to the ~6 partners of the table above, and at K=1 the
partner count is unchanged so exchanges and messages fall by the same factor):

| point | msg/step | bt busy (ms) | bt wait (ms) | ε = dln(wait)/dln(msg) | corr(busy, wait) | wait left at the busiest rank |
|---|---|---|---|---|---|---|
| farc 2048, certified | 182 | 6.72 | 5.23 | — | −0.59 | 3.28 |
| farc 2048, rung K=1 | 94 | 6.70 (−0.3 %) | 4.69 | **0.166** | −0.60 | 2.46 (−25 %) |
| dars 8192, certified | 42 | 2.55 | 3.09 | — | −0.60 | 2.34 |
| dars 8192, rung K=1 | 24 | 2.55 (−0.3 %) | 2.75 | **0.208** | −0.62 | 1.81 (−23 %) |

Two things fall out, and both change the arithmetic.

1. **A quarter to a half of the bt "wait" is the block's own load imbalance, which no message
   reduction can touch.** A rank's bt wait correlates with its bt busy at −0.6 at both points: the ranks that
   finish the subcycle early are the ones that wait. Regressing wait on the busy deficit gives
   `wait ≈ 2.0 × (busy_max − busy) + floor`, i.e. 24–48 % of the mean wait is imbalance
   absorption. Only the **floor** — what the busiest rank still waits — is message-related, and
   *its* elasticity is **0.44–0.46**, not 1: halving the messages removed a quarter of it. (The
   floor is itself an *upper* bound on the latency part: skew inherited from earlier phases lands
   wherever the next exchange is, and for the SE step that is bt. So the genuinely message-elastic
   component is at most this, and the case for a communication lever is if anything weaker.)
2. **The ring-1 redundant compute is free at K=1.** bt busy is unchanged to 0.3 % at both points
   although the census says the rung adds +19.7 % (farc) and +28.2 % (dars) of ring-1 node work.
   The exchange the rung removes took its pack/unpack out of `busy` with it, and that pays for
   the ring. So the honest compute term is `(1 + zone_K − zone_1)`, calibrated at zero for K=1,
   not `(1 + zone_K)`.

With `bt_wait(K) = bt_wait_cert × [φ + (1−φ)·msg_ratio(K)^ε_f]` — φ the imbalance share, ε_f the
floor elasticity — CORE2 16N GPU (7.8 ms busy + 11.8 ms wait of an 84 ms step) brackets like
this:

| K | msg ratio | zone−zone₁ | Δ step, latency-bound (φ=0, ε_f=1) | Δ step, CPU-like (φ=0.4, ε_f=0.45) |
|---|---|---|---|---|
| 1 | 0.530 | 0.00 | **−6.6 %** | **−2.1 %** |
| 2 | 0.280 | 0.08 | **−9.4 %** | −2.9 % |
| 4 | 0.166 | 0.24 | −9.5 % | −2.4 % |
| 8 | 0.104 | 0.61 | −6.9 % | +0.3 % |

The two columns disagree about whether deep K is worth building at all: latency-bound, K=2–4 is
worth 1.4× the K=1 gain and the extended-mesh layer earns its keep; CPU-like, the whole rung is
a ~2 % lever at CORE2 and deep K is negative by K=8. Both agree that **K=8 is never the answer
at CORE2** and that NG5/dars 16N — zone ≤ 0.14 even at K=8 — are where deep K is cheapest.

**Pre-registered prediction for the pending W6 GPU pair:** if the model holds, CORE2 16N GPU
lands between **−2.1 %** (CPU-like) and **−6.6 %** (latency-bound). A result outside that band
means something the model does not contain — most likely the GPU pack/unpack cost, which on this
fabric stages through pinned host memory.

🔴 **So the W6 GPU pair is not only a performance row: it measures ε and φ on the fabric that
matters.** Run `scripts/m12b_wait_anatomy.py off=<off_ph1_dir> on=<on_ph1_dir>` on its
phasestats legs — the same two numbers, on GPU. The queued M-sweep (26952126/27) is the
independent check on the baseline. **Decision rule, pre-registered:** build the §4 extended-mesh
layer only if the GPU floor elasticity is ≥0.7 *and* the imbalance share φ is ≤0.3 — that is the
regime where the left-hand column is right and K=2–4 buys another ~40 % over K=1. Otherwise the
K=1 rung is where this line stops, and the remaining bt wait is an imbalance problem, i.e. M11's
territory, not the halo's.

## 8. Where the barotropic imbalance comes from (s4) — and why it is worth more than the halo on CPU

§7 measured that a quarter to a half of the bt "wait" is the block's own imbalance. That part is
not a communication problem, so the next question is what it *is* proportional to
(`scripts/m12b_bt_imbalance.py`, the same per-rank phasestats against the partition files):

| point | corr(bt busy, owned **nodes**) | corr(bt busy, owned **elements**) | owned nodes max/min | owned elements max/min | R² of the element fit |
|---|---|---|---|---|---|
| farc 2048 | **+0.015** | **+0.959** | 1.01 | **1.47** | 0.92 |
| dars 8192 | +0.227 | **+0.849** | 1.03 | **1.22** | 0.72 |

**The partitioner balances nodes; the barotropic subcycle costs elements.** Owned node counts are
equal to 1 %, and bt busy is uncorrelated with them. Owned element counts vary by 47 % (farc) and
22 % (dars) — the ratio of elements to nodes is a property of where a subdomain sits, coastline
and boundary shape — and bt busy tracks *that*, at r = 0.96 / 0.85. Ū, the viscosity and the
element-side pack are all per-element; η is the only per-node part and it is one of the two
substep halves.

For contrast, the 3-D `ocean` phase at the same point is only weakly explained by 2-D counts
(r = +0.46, R² = 0.21) — its 1.62× imbalance is the bathymetry effect M10 found. So the two
phases want **different** balance constraints, which is what makes this a multi-constraint
partitioning question rather than a re-weighting.

**Size of the prize.** At farc 2048 the bt wall is busy 6.7 + wait 5.2 = 11.9 ms of a 73 ms step.
With elements balanced, the straggler's busy falls to the mean and the imbalance absorption goes
with it, leaving ≈ 6.7 + 3.3 (the message floor) = 10.0 ms — **≈1.9 ms, or 2.6 % of the step.
That is more than the wide halo's measured −1.8 % at the same point**, and the two are
independent: the halo removes messages, the partition removes the imbalance the messages absorb.

**Owned or halo?** Both — and they are collinear, so the split is indicative rather than
identified. Fitting `bt busy ~ a·owned_elems + b·halo_elems + c` gives 10.66 µs and 5.00 µs at
farc 2048 (R² 0.926; of the 3.60 ms measured busy spread, 2.43 ms is the owned term and 1.03 ms
the halo term) and 4.17 / 0.74 µs at dars 8192 (R² 0.726; 0.68 vs 0.20 ms of a 0.90 ms spread).
Owned and halo element counts correlate at +0.96 / +0.90 with each other, which is why the
coefficients cannot be cleanly separated — and also why balancing one would largely balance the
other.

**What this predicts for the GPU points, before they run.** The same entity imbalance is readable
from the partition files alone (`partition_only()` in the same script):

| point | ranks | owned nodes max/mean | owned **elements** max/mean | elements max/min |
|---|---|---|---|---|
| NG5 16N GPU | 64 | 1.005 | **1.008** | 1.02 |
| dars 16N GPU | 64 | 1.005 | **1.014** | 1.04 |
| farc 16N GPU | 64 | 1.000 | 1.026 | 1.10 |
| CORE2 4N GPU | 16 | 1.000 | 1.040 | 1.11 |
| CORE2 16N GPU | 64 | 1.001 | 1.052 | 1.18 |
| farc 2048 CPU | 2048 | 1.004 | 1.062 | **1.47** |
| dars 8192 CPU | 8192 | 1.011 | 1.057 | 1.22 |

Node balance is within 1.1 % everywhere; the element imbalance is what varies, and it is **least
at NG5 and dars 16N**. Those are also the two points with the smallest deep-K zone (§7). So the
pre-registration is specific: **NG5 and dars at 16N are where φ should be smallest and the
latency-bound column of §7 most nearly right**, and CORE2 16N — 1.052 element imbalance and the
steepest zone — is where the rung has the most imbalance mixed into its "wait" and the least room
for deep K.

**Testing the mechanism on a partition that already exists (job 26961927).** M11's certified farc
partition is the same mesh **byte for byte** (md5 of `nod2d.out`/`elem2d.out` match /pool) with a
different partition — Mt-KaHyPar weighted by `nlev`, i.e. balancing 3-D column work — and in the
2-D currencies it is markedly worse:

| farc 2048 partition | owned nodes max/min | owned elements max/mean | owned elements max/min | halo elems mean |
|---|---|---|---|---|
| stock | 1.010 | 1.062 | 1.468 | 141.5 |
| M11-certified (`MTKAHYPAR_w100+nlev`, −7.52 % CPU) | **1.857** | **1.233** | **2.275** | 137.5 |

That is the nlev weighting doing exactly what it is for: nodes are deliberately unbalanced so the
3-D work balances. It also makes the partition a ready-made test of §8, run at farc 2048 under SE
with these predictions written down first:

* **P1** bt busy max/mean rises from the measured 1.146 toward ~1.23, and the bt *wait* rises with
  it — the §8 mechanism, on a partition chosen for an unrelated reason. **A failure here falsifies
  §8 and the handover with it.**
* **P2** the 3-D `ocean` phase's busy imbalance falls (1.62 max/mean at stock) — what the weighting
  buys, and why M11 measured −7.52 %.
* **P3** net under SE the M11 partition still wins, but by *less* than under the implicit solver,
  because SE moves work out of the solver (which the weighting helps) into bt (which it hurts).
  🔴 M11's −7.52 % was measured on another day with another protocol, so P3 needs **both** solvers
  measured here: job 26961927 runs the SE pair, 26961990 the SI pair, same allocation, same day.

(The 2.6 % above is an **upper bound**: it assumes perfect element balance removes all of the
imbalance absorption, and the element fit leaves 8 % of the busy spread unexplained.)

**The whole-step wait budget, since the same decomposition runs on every phase.** farc 2048 CPU,
certified path, 73 ms/step of which the phasestats TOTAL wait is 28.9 ms (40 %):

| phase | busy mean (ms) | wait mean (ms) | corr(busy, wait) | imbalance part | message floor |
|---|---|---|---|---|---|
| ocean | 26.63 (9.0 … 43.1) | 15.18 | **−0.955** | **12.5 ms (82 %)** | 2.65 |
| bt | 6.72 | 5.23 | −0.591 | 2.0 ms (37 %) | 3.28 |
| icedyn | 5.03 (0.8 … 8.5) | 4.80 | −0.970 | 3.2 ms (68 %) | 1.56 |
| force | 3.58 | 2.73 | −0.736 | 2.7 ms (~100 %) | ~0 |

**About 20 of the 28.9 ms — 70 % of the MPI wait, 27 % of the whole step — is load imbalance being
absorbed at an exchange, against ~7.5 ms that any communication lever could address at all.** The
`ocean` phase alone contributes 12.5 ms of imbalance, with a busy spread of 9.0 → 43.1 ms across
ranks (4.8×) — the bathymetry effect M10 identified, here in wait-milliseconds. This is the
proportion to keep in mind for the whole SSH line: M10's "half the step is MPI wait" is true, and
most of that wait is not communication. (Caveat: each phase's imbalance is measured against its
own busiest rank, and skew propagates between phases — a rank late in `ocean` arrives late at
`bt` — so the per-phase split is a good attribution for the large phases and rougher for the
small ones, where `force`'s slightly-over-100 % shows the fit's edge.)

The same budget at **dars 8192** (98 ms step, TOTAL wait 31.8 ms) splits differently: ocean 15.09
ms wait but only 44 % imbalance (floor 8.37 ms — 65 exchanges/step at 8192 ranks is a real
communication cost, ~129 µs each), icedyn 9.35 ms / 61 %, bt 3.09 ms / 24 %, force 2.40 ms / 45 %
— about **14 ms imbalance against 16 ms floor**. So the mix is point-dependent and moves the way
strong scaling predicts: **the imbalance share of the wait falls as the subdomains shrink**
(farc 2048: ~70 % imbalance; dars 8192: ~45 %). Both are large; which lever pays depends on where
on that curve the configuration sits, and the decomposition costs nothing to run.

🔴 **This is an M11 item, not an M12b one** — and a specific one: a *two-constraint* partition
(balance owned nodes **and** owned elements) rather than the single node constraint in use. M11
has Mt-KaHyPar in its toolbox, which supports multi-constraint partitioning directly; its earlier
dual-weighting experiment weighted by 3-D work and lost to a +40 % halo, which is a different
question from adding an element constraint at equal halo. Not attempted here — the finding is
handed over with its mechanism and its measured size.

## Open items (s4)

W2 CUDA drift gate (26960090) + W6 GPU pairs — still queued behind the account's 5-GPU-job
limit (13 jobs ahead as of s4; the user's call was to wait rather than move accounts) ·
per-step wire is now M+3
(subcycle + η coherence + H0e + F-reconcile) vs the certified 2M · deep-K decision on the W6
numbers — note the s3 induction transfers to deep K only if ring-K inputs are owner-coherent the
same way (H0e over the K-ring extent, F still reconciled; the §4 extended-mesh layer ships owner
bytes by construction, which is compatible).
