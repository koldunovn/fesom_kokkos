# M12b — wide halo for the SE barotropic subcycle (`FESOM_SE_WIDE`)

**Status 2026-08-14 s3 COMPLETE on CPU: the s2 instability is ROOT-CAUSED and FIXED by two
coherence repairs (§3: per-step H0e exchange + owner-wins F over the multi-claimed elements). The
free-running rung is BITWISE-exact (drift ≡ 0.0 — np8 ×25, np128 ×300, farc-2048 ×50), both
3000-step screens are green (farc η=2.05, np128 η=1.89 = the SE references), W5b rc=0, and the W6
CPU board reads farc −1.8 % / dars +0.4 % (§5) — the CPU latency share bounds the payoff, as
Sergey predicted. GPU (W2 gate + W6 pairs) queued behind the maintenance window.**
**🔴 s4 (2026-08-15): THE GPU BOARD IS IN, and it is the track's headline — CORE2 16N GPU
−11.5 %** (0.0841 → 0.0744 s/step, min-of-2, all legs η=2.02), **NG5 16N GPU +0.3 % (wash)**. The
W2 CUDA gate passed first (drift 0.000000e+00 every step on GPU). The mechanism is not the one
this document predicted: on GPU the exchange cost sits in `busy` (~51 µs each at CORE2 16N), so
halving the exchanges cut bt busy 31 % and, because that cost varies with halo size across ranks,
the wait fell 40 % too. **The pre-registered band (−1.5…−2.9 %) was wrong by 4× and §7b says
exactly which term of the model failed; the deep-K "no" of §7 is RETRACTED in §7c.** The board is
a **per-rank-size law**: the rung pays where the subdomain is small (1 982 nodes/rank → −11.5 %)
and is a wash where it is large (115 670 → +0.3 %). s4 also CLOSED the `elem_nodes(-1)` hygiene
item (§6) and measured deep-K partner growth (§7).** Plan + full gate record: `docs/plans/20260814-m12b-widehalo.md`. SE
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
| W2 CUDA drift gate | CORE2 4N GPU, 50 steps | **drift nonzero-steps = 0** — 0.000000e+00 at every step on CUDA | 26960090 |
| W2 CUDA free run | CORE2 4N GPU, 300 steps | clean, **η=2.02** | 26960090 |
| W2 CUDA same-binary control | CORE2 4N GPU, 300 steps | clean, **η=2.02 — identical to the rung arm at every printed digit** (T, S, stress, hf, wf, rs all match) | 26960090 |

**s4: the CUDA gate passes.** The s3 induction does not care what the 3-D model's CUDA atomics do
to `Fbt` — the reconcile makes it single-valued either way — and the measurement agrees: the
first-ever CUDA run of the `ELEM2D_FULL` device-halo path and of the F-reconcile's per-step host
round-trip has **zero free-running drift at every step**, and its 300-step endpoint η=2.02 equals
both the same-binary control and the M12 CUDA SE reference (job 26936711). ⚠️ On CUDA the
diagnostic line prints `uv=0.00e+00 w=0.00e+00 Kv=0 Av=0`: that is **pre-existing** (the M12 CUDA
runs print it too) — device-resident fields the host-side print does not sync, not a dead ocean.
η, T and S are live and match the CPU run digit for digit (η=3.88 at step 50 on both).

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
| **CORE2 4N GPU dt1800 M=50** | 0.0684 | 0.0611 | **−10.7 %** | job 26982690 |
| **CORE2 16N GPU dt1800 M=50** | 0.0815 | 0.0741 | **−9.1 %** | job 26982689 |
| **CORE2 16N GPU dt1800 M=100** | 0.1002 | 0.0855 | **−14.7 %** | job 26969145 — the rung's gain grows with M, because it removes half of a per-substep cost |
| **dars 16N GPU dt120 M=20** | 0.1422 | 0.1394 | **−2.0 %** | job 26970050 |
| **NG5 16N GPU dt180 M=20** | 0.2463 | 0.2447 | **−0.65 %** | job 26970051 |

All GPU rows are on the lean binary `58ac143b` (M=100 on `674a53bc`, where the difference is
0.2 ms — see §7d). Every leg ends at the same physical state per point (CORE2 η=2.02
T[−2.06,30.05] S[5.63,41.12]; NG5 η=3.62), off and on alike.

🔴 **Estimator note, and a correction to this table's first version.** It read −11.5 % for CORE2
16N. That came from `min-of-2` over the two un-instrumented legs, and **the certified arm at this
point is too noisy for two reps to converge**: across the six legs of two independent pairs it
spans 0.0814…0.0854 (**4–5 %**), while the rung arm spans 0.0741…0.0760 (**0.4 % in the lean job**).
Taking the min over all three legs of each arm — the phasestats leg's instrumentation is not
measurable here, 0.0816 against 0.0815 — the two pairs agree at **−8.7 % and −9.1 %**, whereas
min-of-2 gave −11.5 % and −9.1 %. The rows above use min-over-all-legs. **The honest CORE2 16N
number is ≈ −9 %, not −11.5 %.**

That noise asymmetry is itself a result worth keeping: **the rung does not only make the step
faster, it makes it more reproducible** — 102 exchanges per step exposes the certified path to
network variation that 54 exchanges largely removes (spread 4.8 % → 0.4 % in the same allocation).

The s2 farc "−8.9 %" remains RETRACTED (all-NaN legs); −1.8 % is the honest CPU number.

**On GPU the board is a clean per-rank-size law, and with the §7d fix every GPU point is a gain:**

| nodes per rank | point | Δ |
|---|---|---|
| 1 982 | CORE2 16N GPU | **−9.1 %** |
| 7 928 | CORE2 4N GPU | **−10.7 %** |
| 49 380 | dars 16N GPU | **−2.0 %** |
| 115 670 | NG5 16N GPU | **−0.65 %** |

(The two CORE2 points are equal within the certified arm's noise; the law's content is the drop
to −2 % and −0.65 % at the large-subdomain points, which are measured to 0.1 %.)

Monotone in per-rank size, because the barotropic exchange costs a fixed ~50 µs per call
(§7c) while the arithmetic grows with the subdomain: the rung pays where the block is
overhead-dominated, i.e. at the strong-scaling limit where a model runs out of speedup. 🔴 **The
CPU points are NOT on this curve** and should not be read as part of it — farc 2048 CPU has 5 700
nodes per rank and gives −1.8 %, where a GPU point of that size gives about −10 %. On CPU the pack
is a memcpy and there is no staging term at all (§7b), so the CPU mechanism is the much weaker
latency one and its own ordering is flat. The baseline M-sweep (jobs 26952126/27) agrees
independently: one extra substep costs **0.33 ms** of step time at CORE2 16N GPU, of which
~0.20 ms is one of its two exchanges, while at NG5 16N a substep costs **0.455 ms** and its
exchange is worth ~0.

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
   The mechanism is not subtle once stated: **imbalance-driven wait is conserved under any
   communication lever.** The time difference between a fast rank and a slow one has to be
   absorbed somewhere; removing an exchange does not remove it, it relocates it to the next
   synchronisation point. Only the latency component is actually removable, which is why an
   elasticity measured against the *total* wait can never approach 1 in an imbalanced phase.
2. **The ring-1 redundant compute is free at K=1.** bt busy is unchanged to 0.3 % at both points
   although the census says the rung adds +19.7 % (farc) and +28.2 % (dars) of ring-1 node work.
   The exchange the rung removes took its pack/unpack out of `busy` with it, and that pays for
   the ring. So the honest compute term is `(1 + zone_K − zone_1)`, calibrated at zero for K=1,
   not `(1 + zone_K)`.

**φ did not need the W6 pair — it was already on disk.** The M12 board's GPU runs carry per-rank
phasestats, so the imbalance share of the *bt* wait can be read at every GPU point right now
(`scripts/m12b_wait_anatomy.py` on jobs 26936705 / 26936711 / 26938162, certified SE path):

| point | step (ms) | bt busy mean / **max** | bt wait | corr(busy, wait) | **φ** | floor (ms) | bt share of step |
|---|---|---|---|---|---|---|---|
| CORE2 4N GPU | 70.3 | 6.88 / 10.20 | 7.69 | −0.990 | 47 % | 4.06 | 21 % |
| CORE2 16N GPU | 84.3 | 7.79 / **14.50** | 11.82 | −0.993 | **56 %** | 5.17 | 23 % |
| NG5 16N GPU | 246.1 | 11.36 / 12.80 | 5.32 | −0.952 | **31 %** | 3.70 | 6.8 % |

The correlations are −0.95…−0.99: on GPU the bt wait is *more* imbalance-driven than on CPU, not
less. NG5 has the smallest share, as §8's partition table predicted (its element imbalance is
1.008, the most balanced point we have). And note the CORE2 16N line: the widely quoted "7.8 ms
busy" is the **mean**; the critical path is 14.5 ms, and 6.7 of the 11.8 ms wait is the other
ranks waiting for it.

⚠️ On GPU the busy spread is **not** the element-count effect of §8 — at CORE2 16N the entity
imbalance is 1.035 while bt busy max/mean is 1.86, and the correlation runs to the **halo**
(+0.63) rather than to owned elements (+0.38, R² 0.15). With ~4 000 elements per rank the
subcycle kernels are launch- and halo-staging-bound, so §8's prescription is a CPU one.

With `bt_wait(K) = bt_wait_cert·φ + floor·msg_ratio(K)^ε_f` and `bt_busy(K) = bt_busy_cert ×
(1 + zone_K − zone_1)`, the measured φ narrows the prediction sharply (ε_f = 1 is the
latency-bound bound, ε_f = 0.45 the CPU-measured one; exchanges/step 101 → 53 at CORE2 M=50, and
41 → 23 at NG5 M=20):

| point | K | Δ step at ε_f = 1 | Δ step at ε_f = 0.45 |
|---|---|---|---|
| CORE2 16N GPU | 1 | **−2.9 %** | **−1.5 %** |
| CORE2 16N GPU | 4 | −2.9 % | −1.2 % |
| NG5 16N GPU | 1 | **−0.7 %** | **−0.35 %** |
| NG5 16N GPU | 4 | −1.0 % | −0.5 % |

🔴 **The table above is SUPERSEDED — it was the pre-registration, and the measurement refuted it**
(§7b: CORE2 16N came in at −11.5 %, four times the top of its band). It is kept because the
refutation is the finding: the term it got wrong was `bt_busy`, and §7b names why. The deep-K
conclusion that followed from it is retracted in §7c. Read on; do not quote this table.

## 🔴 7b. The prediction was wrong by 4×, and the reason is where the pack cost lives

Pre-registered: CORE2 16N in **−1.5 % … −2.9 %**, NG5 16N in **−0.35 % … −0.7 %**.
Measured: CORE2 16N **−11.5 %**, NG5 16N **+0.3 %**. The NG5 magnitude is right and its sign is
not; the CORE2 number is four times the top of its band. **The model was wrong, and specifically
one term of it was wrong.**

| CORE2 16N GPU | certified | rung K=1 | |
|---|---|---|---|
| exchanges/step | 102 | 54 | |
| bt **busy** mean | 7.84 | **5.37** | **−31.4 %** ← the model said 0 % |
| bt busy spread (max−min) | 9.20 | **5.20** | |
| bt **wait** mean | 11.82 | 7.15 | −39.5 % |
| wait elasticity dln(wait)/dln(exch) | — | **0.790** | the CPU measured 0.166 |
| floor elasticity | — | 0.429 | the CPU measured 0.435 |
| imbalance share φ | 55.9 % | 44.5 % | |

**Where the error was.** The model took `bt_busy(K=1) = bt_busy_cert`, calibrated on the CPU pairs
where busy was flat to 0.3 % because the removed exchange's pack (a memcpy) exactly paid for the
ring compute. On GPU that cancellation does not hold in either direction: the pack/unpack and its
staging are *kernels and copies*, they land in `busy`, and at 1 982 nodes per rank they dominate
it — a two-point fit gives **~51 µs of busy per exchange** against ~2.6 ms of actual arithmetic.
Halving the exchanges removed a third of the block's busy time outright.

**And that is also why the wait fell.** The per-rank staging cost scales with halo size, which
varies 11× across ranks at this point (corr(bt busy, halo elements) = **+0.63**, against +0.38 for
owned elements). So a large part of what §7 classified as "imbalance" was **communication cost
wearing an imbalance costume**: removing exchanges removed the spread too (9.20 → 5.20 ms), and
with it the absorption. Hence the wait elasticity of 0.79 on GPU against 0.17 on CPU.

**What survives.** The conservation argument of §7 is still true of *work* imbalance — but the
decomposition's labels depend on which side of the timer the pack sits, and on GPU it sits in
busy. L118's rule needs that caveat, and the practical instruction is sharper: **regress the
phase's busy on the halo as well as the owned counts before deciding what its wait is made of.**

**🔴 What I first wrote about NG5 here was wrong, and my own defect caused it.** The original text
read: "115 670 nodes per rank, halo/owned ≈ 1.2 %: staging is a small part of an 11.4 ms busy, and
the rung's wider element extent and its two extra per-step waves cost **+7.1 % busy** — more than
it saves. The block is compute-bound, so there is nothing for the lever to take." The lean
re-measurement (§7d) falsifies it: **that +7.1 % was the whole-array `Fbt` round trip, not the ring
compute.**

| NG5 16N GPU, bt busy | off arm | on arm | |
|---|---|---|---|
| fat reconcile (26962397) | 11.41 | **12.22 (+7.1 %)** | the rung appeared to *cost* busy |
| lean reconcile (26970051) | 11.42 | **10.34 (−9.5 %)** | the rung *saves* busy, as everywhere else |

So the rung reduces barotropic busy at **every** GPU point once its own overhead is gone: CORE2
−31 %, dars −20 %, NG5 −9.5 %. And the per-exchange staging cost is consistent across all four
points — **44 µs (CORE2 4N) · 51 µs (CORE2 16N) · 68 µs (dars 16N) · 60 µs (NG5 16N)** — a fixed
per-call cost that rises only slightly with payload. **The lever's saving is therefore roughly the
same everywhere; what differs is the step it is divided by.** NG5's bt block is 6.8 % of a 246 ms
step, so a 1.6 ms saving is 0.65 %; CORE2 16N's is 23 % of an 84 ms step, so 9.7 ms is 11.5 %.
That is a share argument, not a compute-cost argument, and it is the honest form of the law.

### 7b-bis. Pre-registration for the two rows that complete the law

The law says the payoff is set by nodes per rank, through the staging share of the barotropic
busy. Two GPU points fill the gap between 1 982 (CORE2 16N, −11.5 %) and 115 670 (NG5 16N,
+0.3 %), and their predictions are written down before they run:

| point | nodes/rank | bt busy / wait (certified, measured) | **predicted Δ step** | **measured** |
|---|---|---|---|---|
| CORE2 4N GPU (job 26969116) | 7 928 | 6.88 / 7.69 ms of a 70.3 ms step | **−6 % … −9 %** | **−9.7 %** — just outside, the model still slightly conservative on the wait side (it assumed the imbalance term would hold, and the busy spread fell 4.5 → 3.3 ms with the staging) |
| dars 16N GPU (job 26970050, lean bin) | 49 380 | not yet measured | **−1 % … −3 %** | **−2.0 %** — inside the band, so the law survives its own falsification test |

The CORE2 4N reasoning: at 102 exchanges and ~51 µs each, staging is ~5.2 ms of that 6.88 ms
busy, so the rung should take ~2.5 ms of busy plus a proportional slice of the 7.69 ms wait — call
it 5 ms of a 70 ms step. The dars reasoning is weaker (its bt split is unmeasured) and rests only
on its position in the ordering, which is why the band is wide. **If dars lands outside −1…−3 %
the law is not a clean function of per-rank size and needs the staging share measured per point.**

## 🔴 7c. Deep K: the "no" is RETRACTED — and the model behind the new estimate is now validated

§7's verdict was built on the flat-busy model, which §7b falsified. The replacement is a two-term
law — per-exchange staging plus ring compute — and the **M=100 pair (job 26969145) was run to test
its load-bearing assumption: that the saving is LINEAR in the exchanges removed.** It is, on three
counts at once:

| CORE2 16N GPU | M=50 | M=100 |
|---|---|---|
| exchanges removed by the rung | 48 | 98 |
| bt busy saved | 2.47 ms | 5.09 ms |
| ⇒ **staging per exchange** | **51.5 µs** | **51.9 µs** |
| bt wait saved | 4.67 ms | 9.64 ms |
| ⇒ **wait amplification per ms of busy** | **1.89×** | **1.89×** |
| ⇒ inferred arithmetic (busy − staging) | 2.59 ms | **4.81 ms** (≈2×, as M doubles) |
| step | 0.0841 → 0.0744 (**−11.5 %**) | 0.1002 → 0.0855 (**−14.7 %**, predicted −12…−16 ✅) |

The per-exchange cost is the same to 1 % across a doubling of M, the wait amplification is
identical, and the arithmetic the fit backs out doubles when the substep count doubles — which it
must. The barotropic block on GPU is **2.59 ms of arithmetic plus 5.29 ms of exchange staging** at
M=50, and the rung takes half the staging.

**Deep K on those constants** (exchanges/step ≈ ⌈M/K⌉+4; ring compute from §2's zone deltas
against K=1, 0.08 / 0.24 / 0.60, applied to the 2.59 ms arithmetic; the 1.89× amplification
applied to the net busy change):

| K | exchanges | staging saved vs K=1 | ring compute added | net busy | **Δ step vs K=1** |
|---|---|---|---|---|---|
| 1 | 54 | — | — | — | — (measured −11.5 % vs certified) |
| 2 | 29 | 1.29 ms | +0.21 ms | −1.08 ms | **≈ −2.4 %** |
| 4 | 17 | 1.91 ms | +0.62 ms | −1.29 ms | **≈ −2.9 %** |
| 8 | 11 | 2.22 ms | +1.55 ms | −0.67 ms | ≈ −1.5 % |

So **K=2–4 would take CORE2 16N from −11.5 % to about −14 %**, with K=8 giving back most of it —
the same optimum the original §7 table guessed, on constants that are now measured rather than
assumed. ⚠️ Two honest bounds on this: the 1.89× amplification was measured for a *staging*
change and must **saturate** as staging vanishes (at K=4 staging is 0.88 ms of a 4.09 ms busy, so
the residual wait is increasingly the arithmetic's own imbalance), which makes these upper
estimates; and the ring-compute term assumes the K-ring exchange keeps the same partner count
(measured flat in §7) and per-exchange cost.

**Verdict for the decision:** deep K is worth **2–3 points on top of the rung** at CORE2-class
per-rank size, for the cost of building the §4 extended-mesh layer. That is a real but modest
return, and it should be weighed against the same effort spent elsewhere — the rung itself already
delivers −9.7 % to −11.5 % at those points with no new mesh machinery. Note also what the M=100
row is *not*: raising M is not an alternative route to the extra points, because the certified
path pays for the extra substeps too (0.0841 → 0.1002).

## 🔴 7d. The rung was carrying an avoidable per-step cost — M9's lesson, again

Prompted by the M9 recollection (the wide-halo EVP got much better once the recomputation that
was not needed came out), the SE rung's per-step path was audited for the same shape. There was
exactly one offender, and it is in the s3 fix rather than in the rung itself.

`se_wide_reconcile_F` sends the owner's `(Fx, Fy)` to the other claimants of the ~0.55 % of
elements that more than one rank owns — a few hundred values. It bracketed that exchange with:

```
s_se.Fbt.sync_host();     …tiny MPI…     s_se.Fbt.modify_host(); s_se.Fbt.sync_device();
```

`se_forcing` marks `Fbt` device-modified every step, so the dirty flag fires every time: **two
whole-array copies per step, plus two device fences, to patch a few hundred elements.** `Fbt` is
`2 × Ne` doubles, so the traffic scales with the mesh per rank, not with the thing being fixed:

| point | elements/rank | Fbt round trip per step | ≈ cost at ~25 GB/s |
|---|---|---|---|
| CORE2 16N GPU | 4 180 | 134 KB | negligible |
| dars 16N GPU | ~98 000 | 3.1 MB | ~0.13 ms |
| **NG5 16N GPU** | ~234 000 | **7.5 MB** | **~0.30 ms** |

That is about **37 % of the +0.81 ms busy regression** measured at NG5 — the point where the rung
came out a wash. **The defect hid in exactly the place that made the rung look bad**: it is
invisible on CPU (host and device alias, so the syncs are no-ops), and negligible at CORE2 16N,
the small-subdomain point the design was tuned on. It bites in proportion to per-rank size, which
is the axis §5's law is drawn on — so part of what that law attributes to "the ring compute costs
more than the staging saves at large subdomains" was this instead.

**Fix (lean staging):** a device kernel gathers the owned slots into a small buffer, only that
buffer crosses the bus, MPI moves it, and a second kernel scatters the received values back.
`Fbt` never leaves the device and the traffic is proportional to the multi-claimed count. The same
values are moved, so the trajectory must be bitwise unchanged — and it is:

| gate (job 26969407) | result |
|---|---|
| rung, 50 steps, old bin `4cc9eda4` vs lean `73c6cf29` | **ALL FIELDS BIT-IDENTICAL, rc=0** |
| rung free-running drift, np128 × 25 steps, lean | **nonzero-steps = 0** |
| knob-off null vs `base_np8`, lean | **rc=0** |

Bins: serial `73c6cf29`, cuda `58ac143b`. **Re-measurement:** the lean row at dars 16N GPU is job
26970050 and at NG5 16N GPU job 26970051; the NG5 pair also *measures* the fix, since its fat
counterpart (26962397) ran the same arms today. CORE2 is unaffected either way (its round trip is
134 KB at 16N, ~500 KB at 4N), so the queued CORE2 rows keep the fat binary rather than restarting
their queue wait.

**Audit result for the rest of the module.** Every other `sync_host()` on the SE per-step path is
inside an env-gated diagnostic (`GEOCHK`/`SELFCHECK`, and `step_n ≤ 5` at that), and the H0e
coherence exchange uses `fesom_halo_field`, the certified packed-halo path, which stays
device-resident. One live offender.

There is a second whole-array sync per step — `hbar` and `hbar_old` after the η commit (~1.9 MB at
NG5 16N) — but it is a different animal on two counts: it is **required** (the ALE pre-block reads
`hbar`/`hbar_old` on the host to build `ssh_rhs_old`, `eta_n`, `zbar_3d_n` and `dhe`, and the ice
coupling reads `hbar` for `srfoce_ssh`), and it is **paid by both arms**, so it cancels exactly in
every A/B on this board. It is a standing candidate for moving the ALE pre-block to the device —
an M12/ALE item, not an M12b one, and not touched here. The reconcile's copies were rung-only,
which is precisely why they distorted the comparison.

**The same scan across the whole port** (a `sync_host()` on a field followed by that field's
`sync_device()` within 40 lines, comments excluded) finds **19 further sites**: `fesom_ice.cpp` 10,
`fesom_kpp.cpp` 4, `fesom_step.cpp` 3, `fesom_ice_thermo.cpp` 1, `fesom_ssh.cpp` 1. Several are
certainly legitimate — a genuinely host-side algorithm has to bring its data over — but each is a
whole-array round trip whose cost scales with the mesh per rank, and none of them can be seen from
a CPU pair. Listed here for the tracks that own those files rather than touched from M12b; the
scan is two lines of Python and worth running before any GPU per-rank-size claim.

⚠️ **The GPU board rows in §5 and the three queued rows were all measured with the fat version**,
so they are a lower bound wherever the mesh per rank is large. The lean re-measurement at NG5 and
dars is the pair that quantifies it.

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

**Size of the prize — the first estimate was wrong, and the pre-registered test is what caught
it.** The estimate here originally read "≈1.9 ms, 2.6 % of the step, more than the wide halo
itself", derived from the mean-wait decomposition: remove the bt imbalance and the absorption goes
with it. Job 26961927 falsified that reasoning, and the reason is worth more than the number.

**Result of the pre-registered test** (farc 2048, SE, same binary, same allocation, same day;
stock partition vs M11's `nlev`-weighted one, which has 2.3× the element imbalance):

| | stock | M11-certified | |
|---|---|---|---|
| s/step (min-of-2) | 0.0728 | **0.0679** | **−6.7 %** |
| bt busy mean / **max** | 6.7 / **7.7** | 6.8 / **8.3** | P1 ✅ on compute |
| bt busy max/mean | 1.144 | **1.226** | tracks the 2-D entity imbalance (1.042 → 1.222) |
| corr(bt busy, owned elements) | +0.959 | **+0.909** | the §8 mechanism holds on a different partition |
| bt **wait** | 5.3 | **4.3** | P1 ❌ on wait — it *fell* |
| corr(bt busy, bt wait) | −0.605 | **−0.256** | the coupling that the estimate assumed, gone |
| ocean busy mean / **max** | 26.6 / **43.1** | 27.8 / **36.7** | P2 ✅ |
| ocean wait | 15.3 | **10.5** | |
| TOTAL busy mean / **max** | 44.0 / **62.9** | 45.7 / **56.0** | |
| TOTAL wait | 28.9 | **22.0** | |

**P1 is confirmed on compute and refuted on wait.** The bt block's *busy* imbalance behaves exactly
as §8 says — it grew with the element imbalance, on a partition built for an unrelated reason, and
the element correlation survived at +0.91. But the bt *wait* went **down**, and its correlation
with the block's own busy collapsed from −0.61 to −0.26: on this partition the bt wait is set by
the skew the ranks *arrive* with from the 3-D phase, not by bt's own spread. Phase-local wait
decompositions are therefore not additive across phases, and the "remove the imbalance, keep the
absorbed wait" arithmetic that produced the 2.6 % does not hold.

**The currency that does work is the critical path — TOTAL busy max.** It falls 62.9 → 56.0 ms
while the mean rises 44.0 → 45.7, and the step follows the max, not the mean (72.9 → 67.7 ms
predicted, 0.0728 → 0.0679 measured). In that currency the two effects are simply additive and
have the right signs: the element imbalance **adds 0.6 ms** to the critical path (bt busy max
7.7 → 8.3) and the 3-D rebalancing **removes 6.4 ms** (ocean busy max 43.1 → 36.7). So the honest
revised prize for adding an element constraint is **the bt block's own max−mean, ~1.0–1.5 ms
(1.5–2.2 % of the step)** — real, worth having, but *on top of* the −6.7 % the nlev weighting
already delivers, and not larger than the wide halo as this section first claimed.

**P3 and a result worth having on its own** (job 26961990, the same pair under the implicit
solver, same day, same allocation):

| farc 2048 CPU, min-of-2 | stock partition | M11-certified | partition effect |
|---|---|---|---|
| implicit (SI) | 0.0840 | 0.0778 | **−7.4 %** |
| split-explicit (SE) | 0.0728 | 0.0679 | **−6.7 %** |
| solver effect | **−13.3 %** | **−12.7 %** | |

P3 said the partition would win by *less* under SE, because SE moves work out of the solver into
bt. The sign is right — −6.7 % against −7.4 % — but the 0.7-point difference is the same size as
the rep spread (1.2–1.5 % between reps of one arm), so **P3 is consistent, not resolved**; it
would need more reps to call.

The SI legs also say, in one line, why SE wins at all here — and that the elasticity of L118 is a
**per-phase** property, not a machine constant:

| farc 2048, stock partition | busy mean (ms) | wait mean (ms) | MPI calls/step |
|---|---|---|---|
| `cg` (implicit solver) | 2.6 | **20.3** | **845.8** |
| `bt` (SE subcycle) | 6.7 | 5.3 | 182 |

The implicit solver's wait is 20.3 ms of an 85 ms step against 2.6 ms of arithmetic, spread over
846 MPI calls — a latency chain, with almost no imbalance in it (its busy is 1.6–3.6 ms across all
2048 ranks). The SE block replaces 22.9 ms of that with 12.0 ms, which is the measured −13 %
almost exactly. **So `cg` is message-bound and `bt` is imbalance-bound, in the same run**: a
communication lever pays handsomely on the first and barely on the second, and only the per-phase
decomposition tells you which you are looking at.

What is resolved is more useful: **the two levers are independent and compose.** SE is worth
−13 % on either partition, the partition is worth −7 % under either solver, and together
0.0840 → 0.0679 = **−19.2 %** at farc 2048 CPU. The M12 board's farc SE number was earned on the
stock partition; it does not have to choose.

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
