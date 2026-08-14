# Upstream PR — deterministic initial-condition hole filling (`ic_extrap_det`)

**Branch:** `ic-extrap-deterministic` on top of upstream `main` (`962c3f0e`), prepared in the
worktree `/work/ab0995/a270088/port2/partm11/upstream_pr_ic` (same clone/remote setup as the
M11 meshpart PRs). One commit, three files: `src/gen_support.F90` (+196),
`src/gen_ic3d.F90` (namelist group), `config/namelist.tra` (documented defaults).

**Title:** `initial conditions: optional deterministic, decomposition-independent hole filling (ic_extrap_det)`

---

## Body (ready for `gh pr create --body-file`)

### What

`extrap_nod3D` fills climatology holes (nodes whose surrounding source-grid cells are
land-contaminated, plus levels below the deepest source data) by averaging already-valid
neighbours. As written, the result depends on the domain decomposition and on the node
numbering: the fill is in-place Gauss–Seidel in local-numbering order, each PE runs its fill to
local exhaustion before the next halo exchange, and a filled node is never revisited
(first-fill-wins). The same cold start on two different `dist_N` therefore integrates two
different initial states wherever extrapolation acted.

In marginal seas this is not a rounding-level effect. On a 5 km global mesh (NG5) initialised
from `phc3.0_winter.nc` we measured, at one element in the Sea of Marmara at ~300 m depth,
initial salinities at its three vertices of 18.61/18.61/18.61 with one partitioning (4096 PEs)
and **21.9/25.5/27.9 with another** (8192 PEs) — a ~10 kg/m³ density jump across a single 2 km
element (the fill reaches the basin from the Black Sea side on one decomposition and mixes
Aegean-side water on the other). The resulting pressure-gradient force of 5.5e-3 m/s² drives a
linear velocity ramp of ~0.5 m/s per timestep from step 1 and a CFL blow-up within ~25 steps.
Of six partitionings of that mesh, four blow up this way and two happen to crest below the
instability threshold. The blow-up is scheme-independent (same ramp with the SSH-CG step and
with a split-explicit barotropic step) and reproducible bit-for-bit per partitioning; which
partitionings die looks random in advance — this is the mechanism behind "this dist_N runs and
that one diverges" on high-resolution cold starts.

This PR adds `ic_extrap_det` (namelist `tracer_init3d`, default `.false.` — **the default
behaviour is unchanged bit-for-bit**). When enabled, the fill is computed in two phases per
level, both Jacobi (source values frozen per sweep) with the neighbour sums accumulated in
ascending global-node-id order, halo exchange after every sweep, and global Allreduce
termination — so the filled values are independent of the decomposition and of the numbering:

1. **Ring fill** — every still-dummy node with at least one valid neighbour takes the
   multiplicity-weighted mean of its valid neighbours (same stencil and eligibility tests as
   the existing routine).
2. **Relaxation** — the filled nodes (original valid data held fixed) are relaxed with the same
   neighbour mean until the global max per-sweep change is below `ic_extrap_tol` (default
   1e-3). Phase 1 alone would meet in an equidistant front of maximal contrast wherever two
   water masses fill one basin; the relaxation replaces those fronts with the smooth interior
   interpolation of the valid boundary data.

The vertical fill and the final exchange are unchanged.

### Why it matters beyond stability

Even where nothing blows up, cold starts on different `dist_N` currently simulate measurably
different transients in every extrapolated region, so scaling studies and any
partition-sensitivity work compare different physical experiments. With `ic_extrap_det=.true.`
we verified (in a C++ port of this exact code path, at 4096 vs 8192 PEs) that the filled T/S
columns are bit-identical across decompositions, all six NG5 partitionings run a 300-step cold
start cleanly, and the runs end in state maxima identical to print precision. Restarted runs
are unaffected either way — the defect only enters through `do_ic3d`.

### Validation status

- The algorithm was developed and validated at scale in a C++ port of this code path
  (bitwise decomposition-independence at 1/2/8 and 4096/8192 PEs; the blow-ups above flip to
  clean runs; default path proven bit-identical to the previous behaviour).
- This Fortran implementation mirrors that validated code 1:1 and compile-tests against
  `main`; we have not run the full Fortran model with it. Happy to add a test or adjust
  conventions.
- Cost: one-off at initialisation; on the 5 km mesh at 8192 PEs the C++ twin needs ~10k fill +
  ~25k relaxation layer-sweeps ≈ tens of seconds.

### How to see the defect on main (no code needed)

Run `do_ic3d` twice on the same mesh with two different `dist_N` and diff the resulting T/S:
differences of O(1) °C/PSU appear in marginal seas (Marmara, gulfs, archipelagos) and below the
deepest source level. With `ic_extrap_det=.true.` the two agree bit-for-bit.

---

## Status: OPENED 2026-08-14 as https://github.com/FESOM/fesom2/pull/979
(commit c7283ee5 on koldunovn:ic-extrap-deterministic; full model compile-verified on Levante/Intel before push)

## Pre-push checklist (executed)

1. `cd /work/ab0995/a270088/port2/partm11/upstream_pr_ic`
   `git remote add fork https://github.com/koldunovn/fesom2.git` (if not present)
   `git push fork ic-extrap-deterministic`
2. `gh pr create --repo FESOM/fesom2 --head koldunovn:ic-extrap-deterministic \
      --title "initial conditions: optional deterministic, decomposition-independent hole filling (ic_extrap_det)" \
      --body-file <this file's Body section>`
