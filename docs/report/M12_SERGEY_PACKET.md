# M12 split-explicit SSH — review packet for Sergey (2026-08-14)

We implemented the split-explicit barotropic solver in the C++/Kokkos port following your
notes (`subcycling.tex`, the Shchepetkin–McWilliams AB3–AM4 variant), behind
`FESOM_SSH_MODE=se`. It is certified (null gates, machine-precision invariants, 3000-step
screens, a 1-year SE-vs-SI twin) and measured (5-point ladder; large-CPU extension running).
This note lists exactly where the implementation follows the notes, where it deviates and
why, what we measured that the notes left open, and the questions we would like your view on.

## 1. What follows the notes directly

- Transports as the working variable with the existing velocity RHS reused: R̄ = Σₖ hₖ·uv_rhsₖ/τ
  after the full predictor (momentum + biharmonic + implicit vertical viscosity), exactly the
  "multiply the rhs with h" simplification. The r-term removes the n-level Coriolis and
  elevation gradient from R̄; both cancellations are exact-by-construction (see §3).
- The subcycling is your eq. (4_8) with the eq. (4_7) weights: η stepped with AB3-extrapolated
  Ū; Ū stepped with AM4-interpolated η *including* η^{m+1}; β=0.281105, ζ/γ/δ from χ.
  No filtering; η^{n+M/M} = η^{n+1}; both rings carry across steps (flat rings at cold start).
- ⟨⟨Ū⟩⟩ accumulated from Ū^{AB3} (eq. 4_12); the corrector trim is eq. (4_13); tracers consume
  the trimmed velocities through the unchanged advection code.
- Extrapolations are computed on owned+halo entities; exactly M elevation + M transport
  exchanges per baroclinic step (plus none for ⟨⟨Ū⟩⟩ — it is consumed owned-only by the trim).
- Harmonic viscosity on the barotropic flow with your "single edge loop" scope; bottom drag
  left out (the Zenodo snapshot ships it disabled as well).

## 2. Deviations from the notes (D1–D4) and one from the reference code

- **D1 — trim thickness hⁿ, not h^{n+1/2}.** Our tracer advection and vertical-velocity
  diagnosis read `uv·helem` with helem still at hⁿ (the thickness commit runs after tracers),
  so hⁿ trim weights are what keeps Σₖ uvₖ·helemₖ = ⟨⟨Ū⟩⟩ exact through the whole chain —
  measured at 3e-13 m²/s over 1000 CORE2 steps. With hⁿ weights the correction is
  depth-uniform in velocity units. The h^{n+1/2} refinement ("remains to be seen") would
  require re-plumbing tracers to true transports; deferred. **Question 1: do you expect the
  centering refinement to matter at dt=1800 on CORE2-class meshes?**
- **D2 — incremental z\* law.** We reuse the validated zstar kernels
  (h^{n+1}=hⁿ(1+Δη/H̃ⁿ)) rather than the notes' closed form h⁰(1+η/H̃⁰); the two are
  algebraically identical (exact telescoping) and the cumulative layer-sum identity is
  measured at 1.5e-11 m after 1000 steps.
- **D3 — viscosity is live, two-term.** Following the Zenodo code rather than a literal
  reading of eq. (3): V[Ūᵐ] − V[Ūⁿ], each term's coefficient from its own state (the
  reference builds `UVBT_harmvisc` inside the substep loop and removes the n-level term from
  the forcing). With the flow-dependent γ₁ coefficient the two-term structure is not
  equivalent to a single V(Ūᵐ−Ūⁿ); with γ₀-only it is.
- **D4 — Coriolis part of r at AB2 consistency.** Our port's uv_rhsAB mixes the Coriolis and
  momentum-advection AB terms, so the reference's `UVBT_4AB` mirror was not directly
  available. We reconstruct Ū_AB2 = (1.5+ε)·Σₖhⁿuv^{n−1/2} − (0.5+ε)·[stored last-step sum],
  second-order-exact (error ∝ Δh·Δu). The elevation-gradient cancellation is exact (§3).
- **Reference-code observation:** the Zenodo FB variant advances Ū with *semi-implicit*
  Coriolis and ∇η at ηᵐ ("I use SI stepping for the Coriolis"), while your notes prescribe
  f×Ū^{AB3} and ∇η^{AM4} for the SM variant. **We implement the notes.** Worth knowing which
  you consider canonical for SM — the AM4 path is what carries the χ-dissipation.

## 3. A small exactness point that reads as a correction to the reference

For the r-term's elevation gradient we use H_e = Σₖ helemₖ (the current element column sum),
not `-zbar_e_bot + mean(η)`. In our port this is *required* for exactness, not cosmetic: the
predictor's rhs carries the η-gradient un-h-weighted per layer, so R̄'s pressure term is
exactly −g·(Σₖhelemₖ)·∇ηⁿ through the same P1 gradient operator — and only H_e = Σhelem
cancels it identically. The Zenodo `hh = -zbar_e_bot + mean(hbar)` is the looser variant of
the same quantity.

## 4. What the notes left experimental — measured answers

- **χ:** 0.05 is *unstable* on CORE2 dt1800 with real 1958 forcing. The failure is physical
  in origin: the winter Antarctic-shelf coastal setdown (Ross, then Bellingshausen), which
  the θ=1 implicit solver damps heavily and SE resolves, grows ~2%/step; the resulting
  vertical velocities (4.3 cm/s) blow the vertical CFL and the tracers die by step ~300.
  **χ=0.1 saturates at the SI equilibrium** (η_max 1.89 vs 1.90 m at 3000 steps); χ=0.15 is
  indistinguishable; doubling M does not help (not a barotropic-CFL issue); the γ₁ viscosity
  helps monotonically but χ is the effective lever. Default set to the measured 0.1.
- **M:** the startup gravity-wave check (dtbt ≤ 0.5·res/√(gH), min over the mesh, abort
  unless forced) gives M_min per mesh: CORE2 dt1800 → 35 (we run 50) · farc dt900 → 82 (we
  run 90) · dars dt120 → 15 (we run 20) · NG5 dt180 → 17 (we run 20). The farc bound was
  confirmed empirically (a forced M=50 run goes `inf` at step 1). Your "M=30–50" phrasing:
  at CORE2 dt1800, M=30 is genuinely inadmissible.
- **Wide halo (your "next step"):** deferred by design; the first board says the payoff
  region is where the bt-phase MPI wait dominates (CORE2 16N GPU: 60% of the bt cost is
  wait). Scoping note in the plan (M12b).

## 5. Certification and physics summary

- Invariants (CORE2, 1000 steps): η-compatibility ‖η^{n+1}−ηⁿ−τ(T(⟨⟨Ū⟩⟩)−W)‖∞ ≤ 2e-14 m ·
  trim consistency ≤ 3.4e-13 m²/s · cumulative layer-sum 1.5e-11 m · volume drift ≡ 18 nm of
  mean sea level. The module has no atomics and no global reductions (SASS-audited);
  serial runs are byte-reproducible.
- **1-year twin (CORE2 dt1800): SE−SI annual-mean SSH rms = 0.2 mm, max 2.9 mm**, against a
  0.66 m field std — the transient difference (SE's honestly-resolved cold-start response,
  peak η ~3 m vs SI ~2 m) does not accumulate.
- Performance (same-day pinned pairs vs the FESOM_SPEED=1 SI baseline): CORE2 4N GPU −7.1% ·
  CORE2 16N GPU −10.9% · farc 2048 CPU −14.2% (best SSH result of the campaign; the cg
  wait-chain — 846 MPI calls/step — replaced by 181) · dars 2048 CPU +1.1% (low share at
  dt120) · NG5 64 GPU −3.5%. Large-CPU extension (dars 32/64 nodes, NG5 32/64/128 nodes,
  head-to-head with the M10 implicit solvers) in progress.

## 6. For the Fortran side

Restart continuity needs the subcycle history in restart files: the η ring (two old levels)
and Ū ring (two old levels) at the step boundary — our port is cold-start-only so we did not
implement this, but any Fortran adoption of the SM variant will need it (your notes' point).

## Questions (collected)

1. h^{n+1/2} vs hⁿ trim weights — worth the transport re-plumbing? (D1)
2. AB3+AM4 vs the reference's SI-Coriolis for the SM variant — which is canonical? (§2)
3. χ=0.1 as default: consistent with your stability analysis expectations for SM?
4. Is a Demange-FB arm worth implementing for a direct stability A/B on the same code base,
   or is the SM evidence (stable at χ=0.1, same equilibrium as SI, 0.2 mm annual mean)
   sufficient?
5. The coastal-setdown transient: SE "hotter than SI by physics" — do you want a dedicated
   comparison against tide-gauge-class observations before recommending SE for production?

## Addendum — Sergey's first feedback (relayed 2026-08-14)

"Надо отрегулировать число шагов, оно может быть разным на разных сетках… 30 может работать
[на равномерных сетках]; Патрик использовал 50, но это уже много. На GPU переход на wide halo
должен принести преимущества, а на CPU это просто лучшее скалирование."

Actions taken: (a) per-mesh M is automated by the startup CFL guard (this packet §4) and the
ladder already runs per-mesh M; (b) following "50 is already a lot", CORE2 M=40 arms (guard
minimum 35) were run at 4N/16N GPU + 2048 CPU; (c) the wide-halo phase (M12b) is scoped
GPU-first, with the CPU variant framed as a scaling play — exactly this guidance.

**Results on (b), and an interaction with (c) that we did not anticipate.** M=40 runs clean at
CORE2 16N GPU and reaches the same 300-step state as M=50 (η=2.02, T[−2.06, 30.05], S[5.63,
41.12] — identical to the printed digits), for **0.0801 s/step against 0.0841 at M=50**. So your
"50 is already a lot" holds, on physics and on time.

But M and the wide halo are **the same lever applied twice**, and the wide halo made that
measurable. The barotropic block's GPU cost is dominated by a **fixed ~50 µs per halo exchange**
(a per-call cost: 44 µs at CORE2 4N, 51 µs at 16N, 68/60 µs on our two large meshes), and both
levers work by removing exchanges — M by having fewer substeps, the rung by halving the exchanges
each substep does. Measured at CORE2 16N GPU, same day, same binary: one extra substep costs
**0.322 ms** of step time on the certified path and only **0.222 ms** with the rung, because the
rung has already taken half of what a substep pays for. Consequences:

* the rung's *relative* gain **grows with M** — −11.5 % at M=50 and −14.7 % at M=100 — so a
  configuration with many substeps is exactly where it pays;
* lowering M shrinks that gain slightly but the two still **compose in the right direction**:
  M=40 with the rung should land near 0.0722 s/step against 0.0841 for M=50 without it, i.e.
  **≈ −14 %** from applying both. We are measuring that pair now rather than quoting the
  extrapolation.

**Question 9: is there a reason to prefer one over the other where they overlap?** Lowering M
changes the barotropic solution (it is a discretisation choice); the wide halo does not (it is
bitwise-exact against the exchanged path). If both buy the same ~4 ms, we would rather take it
from the halo and keep M where the CFL guard and your stability analysis want it — unless you see
an accuracy reason to prefer the smaller M in its own right.

**On "30 may work on uniform meshes":** our startup guard agrees with the spirit of it. It derives
M_min from the mesh (dtbt ≤ 0.5·res/√(gH), min over the mesh) and returns **15 for dars and 17 for
NG5** — where we run 20 — against **35 for CORE2**, whose bound is set by its own
shallow-and-coarse spots rather than by its average resolution. So the meshes closest to uniform
are already running near your number; CORE2 cannot go to 30 without forcing the guard.

## Addendum 2 — M12b wide halo (K=1): what it took to make it exact, and two findings that stand on their own

The wide-halo step you asked for is built and certified on CPU (K=1 rung, `FESOM_SE_WIDE=1`).
Instead of receiving η on the halo, each rank computes it on its ring-1 nodes; only Ū is
exchanged, over the existing `com_elem2D_full` partner structure. Messages per substep are
halved — measured ×0.500 at 11 operating points (CORE2/farc/dars/NG5, 16–8192 ranks), because
the full element list reaches the *same* partner ranks, not more of them. The price is the
ring-1 redundant compute: **+1.2 % (NG5 16N GPU) … +6.1 % (CORE2 16N GPU) … +19.7 % (farc 2048
CPU) … +28.2 % (dars 8192 CPU)** — the GPU/CPU asymmetry in your comment, as a number.

Getting from "it runs" to "it is exact" took two repairs, and both are worth reporting
independently of M12b, because neither is a property of the wide halo.

**(a) In the SE module the halo copy of `H0e` was not the owner's bytes.** `se_forcing`
recomputes `H0e` locally over owned+halo elements. At a halo element that computation reads η
at vertices that are not all in the local node list — an edge-neighbour's far vertex is
routinely a ring-2 node, and our scatter marks it −1 — so the η-mean read `eta0[-1]`, out of
bounds. Measured: 1334 halo copies differ from the owner's value by up to **1.0e-1 m**, and the
decomposition is unambiguous (the depth part is identical for every offender; the η part
carries the whole difference). At step 1 it is invisible, because `H0e` is η-independent at
cold start. In the certified path those halo values are dead state — Ū at halo elements is
received, not computed — so nothing was ever wrong in the results we certified. They become
live the moment anything is computed instead of received, which is exactly what a wide halo
does: k3's viscosity reads `H0e[nb]` at halo slots, two holders of a shared element then
evaluate V with different hh, and the two viscosity terms stop cancelling. Fix: one per-step
owner exchange of `H0e` (plus a −1 guard at the local recompute).

**(b) The 3-D model's element ownership is not a partition, and per-element fields differ at
the last bit across the claimants.** 1341 of 244659 CORE2 dist_8 elements are claimed by more
than one rank, and `Fbt` (the vertically integrated velocity RHS) differs across holders at the
last bit: with everything else made coherent, the first offender appears at step 4 — one
element, 2.2e-19 in x — before any barotropic state has diverged. Harmless wherever the value
is subsequently exchanged. It is not harmless as a seed.

**(c) The measurement that ties them together: the free barotropic interface iteration is
linearly unstable to any rank-inconsistency.** Running the rung free (no η exchange at all)
and reporting the per-step drift of the locally computed η against the owner's value:

| free-running max\|local − owner\| | step 1–3 | 10 | 25/30 | 300 | growth |
|---|---|---|---|---|---|
| CORE2 np8, M=50 | 0.0 | 1.1e-15 | 1.2e-15 | — | flat |
| CORE2 np128, M=50 | 0.0–ulp | ~1e-15 | ~2e-15 | 1.7e-10 | **×1.056 / step** |
| farc 2048, M=90 | 0.0 → 5.2e-17 | 6.1e-13 | **49 m @ 30** | NaN | **×5.35 / step** |

Cutting the seed by nine orders of magnitude moved the farc blow-up by ~15 steps. The rate
rises with the ring-1 fraction of the subdomain (6.1 % of owned at np64, 13.2 % at np128,
19.7 % at farc 2048), so it is a property of the scheme, not of the seed: **no seed reduction
suffices — every input of the substep must be single-valued across ranks, exactly.** With
`H0e` exchanged once per step and `Fbt` reconciled owner-wins over the 0.55 % multi-claimed
elements (one small extra message per step), the redundant copies stay **bitwise** locked by
induction and the free rung's drift is 0.000000e+00 at every step — verified at np8 (25
steps), np128 (300) and farc 2048 (50), with both 3000-step screens landing on the SE
references (farc η=2.05, CORE2 np128 η=1.89). Per-step wire: M+3 messages against the
certified 2M.

**Rim algebra for deeper K.** With Ū valid to element ring E and η to node ring D: η^{m+1} is
valid wherever its incident elements are (D′ = E), and Ū^{m+1} loses one ring to the
viscosity's edge-neighbour term (E′ = E−1). **K substeps therefore need K rings, not the 2K−1
of mEVP** — mEVP carries element σ across substeps, the η/Ū alternation does not. Cumulative
node zone as a fraction of owned at K=8: NG5 64 GPU **0.10** · CORE2 64 GPU 0.66 · farc 2048
CPU 2.40 · dars 8192 CPU 3.02. Deep K is nearly free exactly where the subdomain is largest.

**The numbers, and your prediction was right about GPU-vs-CPU — but the reason is not the one we
expected.** Same-day pinned pairs, 300 steps, min-of-2, loop-only:

| point | nodes per rank | certified | rung K=1 | Δ |
|---|---|---|---|---|
| **CORE2 16 nodes GPU** | 1 982 | 0.0841 | 0.0744 | **−11.5 %** |
| farc 2048 CPU | 5 700 | 0.0731 | 0.0718 | −1.8 % |
| dars 8192 CPU | 5 100 | 0.0976 | 0.0980 | +0.4 % |
| **NG5 16 nodes GPU** | 115 670 | 0.2461 | 0.2469 | +0.3 % |

We expected the GPU gain to come from removing MPI *wait*. It mostly came from removing GPU
*work*: on the device the halo pack/unpack and its staging are kernels and copies that sit in the
block's busy time — about **51 µs per exchange** at CORE2 16N against roughly 2.6 ms of actual
barotropic arithmetic — so halving the exchanges cut the barotropic busy time by **31 %**
(7.84 → 5.37 ms). The wait then fell by 40 % as a consequence, because that staging cost scales
with each rank's halo size (which varies 11× across ranks here), so it was also most of the
block's load imbalance. On CPU the same pack is a memcpy and the effect does not exist, which is
why the CPU rows are small.

The consequence is that **the payoff is governed by the number of nodes per rank, not by the
mesh**: the rung pays where the subdomain is small enough that the barotropic block is dominated
by per-exchange overhead, which is precisely the strong-scaling limit. At NG5 with 115 670 nodes
per rank the wider element extent costs +7.1 % of busy and the rung is a wash. So "GPU advantage,
CPU better scaling" holds — with the refinement that it is really "small-subdomain advantage",
and a big-mesh GPU run at few nodes behaves like the CPU case.

**One more measurement, which qualifies the whole expectation.** The argument for the wide halo
is that the barotropic block's MPI wait is large (60 % of its cost at CORE2 16N GPU). But a wait
is not automatically a latency pool. The CPU pairs halve the exchange count at unchanged compute,
so they measure the elasticity directly: at farc 2048 the bt wait went 5.23 → 4.69 ms for
182 → 94 exchanges (elasticity **0.17**), at dars 8192 3.09 → 2.75 ms for 42 → 24 (**0.21**).
Per-rank statistics say why: a rank's bt wait correlates with its own bt busy at −0.6, i.e. a
quarter to a half of the "wait" is the block's load imbalance being absorbed at the exchange, and
even the floor left at the busiest rank has elasticity only ~0.45. Two consequences worth your
view: (a) on CPU the remaining barotropic wait is a partitioning problem rather than a
communication one; (b) the ring-1 redundant compute turned out to be *free* — bt busy is
unchanged to 0.3 % although the ring adds +20…28 % of node work, because the exchange the rung
removes took its pack/unpack with it. The GPU equivalent of these two numbers is what decides
whether we build the deeper-K layer at all.

**Questions 6–8.**
6. Does the Fortran wide-halo plan intend to keep a coherence exchange every k substeps, or to
   make all substep inputs owner-coherent as we did? Our measurement says the second is
   required if η is never exchanged: at farc-class interface fractions even a 1e-17 seed
   reaches metres in 30 steps.
7. Is the multi-claimed-element redundancy (b) something you would rather fix in the mesh
   layer (one owner per element) than work around per module? Every future
   compute-instead-of-communicate transformation will meet it.
8. For deep K on GPU, is K=8 with a K-ring extended mesh (owner bytes shipped once at startup,
   BFS ring order) the direction you would take, given that its node-zone cost at NG5/dars GPU
   is 10–30 % of owned? **We had talked ourselves into "no" and the measurement
   reopened it.** Our first estimate treated the barotropic wait as mostly load imbalance, which
   no halo change can remove, and concluded K=1 was the end of the line. The K=1 GPU pair then
   came in at −11.5 % against our predicted −1.5…−2.9 %, because the exchange cost is GPU *work*
   rather than wait (above). On the measured cost — ~51 µs of busy per exchange — going from
   ⌈50/1⌉ to ⌈50/4⌉ exchanges would take another ~2 ms out of the barotropic block before the
   ring compute (which grows to ~30 % of the owned node set at K=4 for this configuration)
   claws some back, so **K=2–4 looks worth roughly three more points at CORE2 16N** — an
   extrapolation from two points, not a measurement. **Would you expect the ring compute to
   behave that way at K=4, and is there a reason to prefer exchanging η every k substeps
   (cheap to try, no extra mesh machinery) over building the K-ring extended mesh?**
