# M12 — Split-explicit barotropic solver (`FESOM_SSH_MODE=se`)

**Status 2026-08-14: implemented, gated (G0–G3), first perf board harvested (G4).**
Track plan + full gate record: `docs/plans/20260813-m12-split-explicit.md`.
Spec: Sergey's notes `ssh_sergey/subcycling.tex` (SM2005 AB3–AM4 dissipative subcycling);
Banerjee et al. 2024 (GMD 17, 7051). Module: `src/fesom_ssh_se.{h,cpp}`.

## What it is

Replaces the semi-implicit CG SSH solve (the step's only anti-scaling component, M10) with
M explicit substeps of the 2-D (η, Ū) system per baroclinic step: AB3-extrapolated transport
in the η equation, AM4-interpolated elevation (4 time levels) in the Ū equation, χ-dissipative
SM2005 weights, live two-term harmonic viscosity V[Ūᵐ]−V[Ūⁿ] (Zenodo closure), exact removal
of the n-level Coriolis/η-gradient from the baroclinic forcing (r-term), corrector trim that
pins Σₖuv·helem = ⟨⟨Ū⟩⟩ to rounding. **No global reductions; per substep two tiny 2-D halo
exchanges** (η node-scalar + Ū element-pair — the mEVP pattern). The validated zstar ALE tail
is reused unchanged (hbar/hbar_old carry Δη). Deterministic gather-CSR divergence (no atomics
anywhere in the module — SASS-audited).

## Knobs

| knob | default | notes |
|---|---|---|
| `FESOM_SSH_MODE` | `si` | `se` requires `FESOM_ALE=zstar`; unrecognised values abort |
| `FESOM_SE_M` | 50 | substeps; startup CFL guard prints the mesh minimum and aborts below it (`FESOM_SE_M_FORCE=1` overrides). Measured ladder values: CORE2 dt1800 → 50 (min 35) · farc dt900 → 90 · dars dt120 → 20 · NG5 dt180 → 20 |
| `FESOM_SE_CHI` | **0.1** | **measured, not guessed**: 0.05 dies on CORE2 real forcing by step ~300 (Antarctic-shelf setdown → vertical-CFL tracer death); 0.1 saturates at the SI η class; 0.15 indistinguishable |
| `FESOM_SE_VISC` | 1 | live two-term harmonic viscosity; `FESOM_SE_VISC_GAMMA0/GAMMA1` (Zenodo defaults 10 / 2750) |
| `FESOM_SE_CHECK` | off | invariant suite (η-compatibility, trim consistency, layer-sum, volume window; NaN-loud) |
| `FESOM_SE_DUMP` | off | per-rank barotropic-state dump (determinism gates) |

Interplay: `FESOM_SPEED_SSHRAILS` force-off under `se` (announced); `FESOM_DIAG_SSHSLV/SPREAD`
and `FESOM_KK_VERIFY=ssh` abort; wsplit composes (3000-step screen green).

## Certification summary

- **G0**: knob-off byte-identical to the certified baseline (serial diff_snap rc=0) + CUDA
  fidelity gate PASS — at T1, after T4/T5, and at T6.
- **G1** (machine-precision invariants, CORE2 1000 steps): η-compatibility ≤ 2e-14 m ·
  trim ≤ 3.4e-13 m²/s · layer-sum 1.5e-11 m · volume drift ≡ 18 nm of mean sea level.
- **G2**: serial same-binary byte-determinism (CORE2 dist_8); 0 atomic/RED SASS instructions
  in all SE kernels; CUDA coupled floor recorded in `docs/REFERENCE_RUNS.md` (η ~1e-3 —
  judge SE-CUDA runs against that row, not the SI one).
- **G3**: 3000-step screens pass (headline / wsplit-compose / χ=0.15 / CUDA; equilibrium
  η=1.89 vs SI 1.90); **1-year twin: SE−SI annual-mean SSH rms 0.2 mm, max 2.9 mm, vs field
  std 0.66 m**.
- ⚠️ The scheme is **hotter than SI in transients by physics** (the θ=1 implicit solve damps
  the fast barotropic response; SE resolves it): cold-start peak η ≈ 3 m vs SI ≈ 2 m around
  step 200 on CORE2. χ=0.1 is what keeps that bounded — do not lower it without a screen.

## First perf board (2026-08-14, frozen `se0` bins, pinned same-day pairs, min-of-2, 300 steps, loop-only)

| point | SI s/step | SE s/step | Δ |
|---|---|---|---|
| CORE2 4N GPU dt1800 | 0.0758 | 0.0704 | **−7.1%** |
| CORE2 16N GPU dt1800 | 0.0942 | 0.0839 | **−10.9%** |
| farc 2048 CPU dt900 (M=90) | 0.0860 | 0.0738 | **−14.2%** |
| dars 2048 CPU dt120 (M=20) | 0.4196 | 0.4243 | +1.1% |
| NG5 64 GPU dt180 (M=20, wsplit) | 0.2544 | 0.2456 | **−3.5%** |

farc −14.2% **beats the M10 best implicit-solver result (oati −13.28%)** at the same point.
Mechanism (phasestats): at farc the SI cg phase is wait-dominated (2.6 ms busy + 20.6 ms MPI
wait, 846 MPI calls/step) → SE bt = 11.8 ms, 181 calls; total MPI/step 1196→526. dars at
dt120 has a small SSH share — the M10 "share predicts payoff" law holds for SE too.
Absolute s/step are loop-only (no snapshots) — not comparable to the M7 production board.

Frozen bins: `/work/ab0995/a270088/port2/m12/bin/` `fesom_port_serial_se0` (df959e90…) /
`fesom_port_cuda_se0` (d17fdf38…) @ git 29e443b; shas in `SHA256.se0`.

## Open items

the formal M11-style 20-step disturbance report · wide-halo phase
(M12b) decision on the board · Sergey review packet (deviations D1–D4; the reference's
SI-Coriolis/∇ηᵐ vs the notes' AB3/η^AM4 — we implement the notes; H_e=Σhelem exactness note;
the χ=0.1 stability measurement).
