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
minimum 35) submitted at 4N/16N GPU + 2048 CPU; (c) the wide-halo phase (M12b) is scoped
GPU-first, with the CPU variant framed as a scaling play — exactly this guidance.
