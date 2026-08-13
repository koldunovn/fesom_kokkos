# M12 — Split-explicit barotropic solver (`FESOM_SSH_MODE=se`)

Branch `m12-split-explicit` in `~/port_kokkos` (in place; main checkout is the M12 track home).
Spec: Sergey's notes `ssh_sergey/subcycling.tex` (2026-08-13) — the Shchepetkin–McWilliams (2005)
AB3–AM4 dissipative subcycling — plus Banerjee et al. 2024 (GMD 17, 7051; doi:10.5194/gmd-17-7051-2024).
Reference Fortran (FB-θ variant only): `ssh_sergey/zenodo_se/fesom2-refactoring_addsshsubcycl/src/oce_ale_ssh_splitexpl_subcycl.F90`
(Zenodo 10.5281/zenodo.10040944; gitignored third-party material).

Plan-review 2026-08-13 (plan-review agent): NEEDS REVISION → all findings applied in this
version. Headline corrections: G2 re-registered to the FMA floor + same-backend determinism
(device fmad stays enabled in certified builds, CMakeLists.txt:47-55 — the M2.1 class);
D3 corrected — the Zenodo viscosity is LIVE per substep, acting on Ūᵐ−Ūⁿ, not frozen;
SSHRAILS gates two pushes OUTSIDE the SI block that must be neutralized under `se`;
Coriolis r-term needs the AB2-consistent transport sum for exact cancellation; dead knobs dropped.

## Overview

Replace the semi-implicit (SI) SSH solve — the CG iteration that M10 proved to be the **only
anti-scaling component** of the step (0.59× while the rest scales 5.40×; the paper finds SE pays
off at high rank counts, below ~a few hundred surface vertices/core — recheck the exact figure
against the paper before it goes into docs) — with explicit barotropic subcycling: M ≈ 30–50
substeps of the 2-D (η, Ū) system per baroclinic step. **No global reductions at all**; per
substep only two tiny 2-D halo exchanges (η node-scalar, Ū element-pair). Behind a proper option
knob; SI path byte-untouched when off.

Success = G0–G3 certified + G4 ladder measured honestly (both currencies: % of step AND SSH share).
Wide-halo (mEVP-style) is explicitly **deferred** to a follow-up phase gated on G4 numbers.
(The notes' "<300 surface vertices/core" figure refers to the CPU doubled-halo regime of that
follow-up phase, not to the SE payoff threshold.)

## Settled decisions (brainstorm 2026-08-13 — do not relitigate)

1. Track: M12, branch `m12-split-explicit`, work in place in `~/port_kokkos`.
2. Scheme: **SM AB3–AM4 only**, weights in a small struct so Demange FB(θ) can drop in later.
3. Architecture: side-by-side module `fesom_ssh_se.{h,cpp}`, dispatch at the §5 block; predictor
   (substeps 1–6) byte-untouched; `uv` stays the stored variable, transports formed at block edges.
4. Validation: G0 null gate, G1 invariants, G2 CUDA↔Serial at the **FMA floor** with
   **same-backend byte-determinism** (revised — see Task 8), G3 SE-vs-SI physics twins,
   G4 standard perf ladder. No Fortran runs — Zenodo is reading material.
5. `se` requires `FESOM_ALE=zstar` (abort under linfs): the SE thickness law *is* z\*.
6. No restart plumbing: the port is cold-start only (io_restart.F90 never ported). Notes' restart
   remark recorded for upstream only.
7. Perf scope: nominal implementation; L109 lean captures (raw pointers) from day one; honest risk
   at 4N GPU — measure, don't argue.

## Deviations from the notes (register — flag to Sergey at review)

- **D1 trim thickness:** corrector trim + `uv` writeback use **hⁿ** (current `helem`), not
  h^{n+1/2}. Reason: tracer advection and vert_vel read `uv·helem` with helem still at hⁿ
  (substep-14 commit runs *after* tracers), so hⁿ is the choice that keeps
  Σₖ(uv·helem) = ⟨⟨Ū⟩⟩ **exact** through the whole chain. The h^{n+1/2} refinement ("remains to
  be seen" in the notes) needs true-transport plumbing in tracers — deferred. Hardwired in v1
  (deliberately NOT a knob — L80/L102 dead-knob rule).
- **D2 thickness law:** reuse the existing incremental zstar kernels
  (h^{n+1}=hⁿ(1+Δη/H̃ⁿ)). Algebraically identical to the notes'/Zenodo closed form
  h⁰(1+η/H̃⁰) (exact telescoping), and it is the validated Fortran-mirroring path.
- **D3 barotropic viscosity is LIVE (corrected by plan review):** the notes put V_h(Ū) in the
  barotropic operator with r removing its value at n. The Zenodo code implements exactly that:
  `UVBT_harmvisc` is rebuilt **inside the substep loop** from current Ū
  (oce_ale_ssh_splitexpl_subcycl.F90:934-1003), while the once-per-step part (:844-898, opposite
  sign `vi=-dt*…`) *removes* V_h(Ūⁿ) from the forcing — net effect: viscosity acts on
  **Ūᵐ−Ūⁿ**, the fast barotropic noise only. We implement the linear-equivalent single form:
  the Ū-update kernel adds V_h(Ūᵐ−Ūⁿ) (harmonic, edge-neighbor gather, Ūⁿ frozen at m=0).
  No F modification, no double-counting against the 3-D biharmonic inside R̄, by construction.
- **D4 first-order barotropic forcing:** F is at time n (notes accept this; "AB extrapolation can
  be tried" = future knob, not v1). Exception: the **Coriolis part of the r-term uses the same
  AB2 weights as the 3-D momentum RHS** — our port's weights (fesom_momentum.cpp:77-78,131):
  old couplet ×−(0.5+ε), new ×+(1.5+ε), first step ×1.0. Implementation detail (refined at T2:
  our `uv_rhsAB` mixes Coriolis with momentum advection, so the Zenodo `UVBT_4AB` mirror is not
  directly available): Ū_AB2 = ab2·Σₖhⁿ·uv^{n−1/2} + ab1·`se_Ubt_prev`, where `se_Ubt_prev` is
  last step's stored 2-D transport sum (Σₖh^{n−1}·uv^{n−3/2}). The h-lag on the old term makes
  the cancellation second-order-exact (error ∝ Δh·Δu) at zero 3-D storage cost.
- Bottom drag: **not implemented in v1** (the Zenodo snapshot ships it disabled too); tidal work
  adds the code and its knob together.

## The scheme as implemented

Predictor (unchanged, substeps 1–6): after `fesom_impl_vert_visc_kk`, `uv_rhs` **is** the complete
velocity increment (fesom_momentum.h:94-98) and is halo-valid (step.cpp:712). All element-field
work below runs on **owned + eDim** (the ELEM3D/ELEM2D exchanges fill eDim only; eXDim slots of
helem/uv_rhs are never filled — do not iterate them).

- R̄_e = Σₖ helemₖ·uv_rhsₖ / τ  — element 2-vector, computable on owned+eDim, **no comm**.
- H_e ≡ Σₖ helemₖ (column sum of current element thicknesses); H0_e := H_e − ηⁿ_e (once per step,
  ηⁿ_e = vertex mean). During subcycles H^{AM4} = H0_e + η^{AM4}_e. Using Σhelem is **required**,
  not merely tidy: the port's `uv_rhs` carries the η-gradient un-h-weighted
  (fesom_momentum.cpp:520-534), so R̄'s pressure term is exactly −g·(Σhelem)·∇η and only
  H_e=Σhelem cancels it exactly. (The Zenodo `-zbar_e_bot+mean(eta)` form is the looser variant —
  note this in the Sergey packet.)
- Ū_AB2 = Σₖ helemₖ·uv^{AB2}ₖ — the AB2-weighted vertical transport sum matching the 3-D
  Coriolis (D4; weights + first-step case resolved from fesom_momentum.cpp:517-561 at T3).
- F_e = R̄_e + (f e_z×Ū_AB2 + g·H_e·∇ηⁿ)_e — constant over substeps. ∇η via `gradient_sca`
  (the same P1 gradient the momentum RHS uses), so the g∇η inside R̄ cancels exactly.
  No viscosity term in F (see D3).
- Subcycle m = 0…M−1 (SM2005 generalized FB; **3 kernels + 2 exchanges**):
  1. per-element (owned+eDim): Ū^{AB3} = (3/2+β)Ūᵐ −(1/2+2β)Ū^{m−1} +βŪ^{m−2};
     ⟨⟨Ū⟩⟩ += Ū^{AB3}/M.
  2. per-node (owned): η^{m+1} = ηᵐ + (τ/M)(−div(Ū^{AB3}) − W); div = **precomputed gather CSR**
     (node → adjacent-element coefficient pairs assembled from `edge_cross_dxdy`), no atomics,
     fixed order → deterministic. W = `forcing->water_flux` (+ = ocean loses; ∂η/∂t = −W).
  3. halo exchange η: `fesom_halo_field(se_eta_cur, NOD2D, 1, 1, p)`.
  4. per-element (owned): Ū^{m+1} = Ūᵐ + (τ/M)(−f e_z×Ū^{AB3} − g·H^{AM4}·∇η^{AM4} + F
     + V_h(Ūᵐ−Ūⁿ)); η^{AM4} = δη^{m+1} +(1−δ−γ−ζ)ηᵐ +γη^{m−1} +ζη^{m−2} evaluated **inline**
     at the 3 vertices; V_h = harmonic edge-neighbor gather (neighbors halo-valid from the
     previous substep's exchange; Ūⁿ = `se_Ubt_n`, frozen copy at m=0).
  5. halo exchange Ū: `fesom_halo_field(se_Ubt_cur, ELEM2D, 1, 2, p)` (one fused message; the
     mEVP per-substep pattern, fesom_ice_maevp.cpp:322).
  6. ring rotate (host-side View-handle swap; kernels capture raw pointers per substep — L109).
- Weights: β=0.281105; ζ=0.00976186−0.13451357χ; γ=0.083445−0.513584χ; δ=½+2ζ+γ+2χ.
  Cold start: both old ring levels := current field (η⁰, Ū⁰).
- Finalize: η^{n+1}=η^M; **both rings (η and Ū) carry to the next step** (the m=0,−1,−2 history
  of step n+1 is the m=M,M−1,M−2 of step n; the flat-ring reset is cold-start only). No ⟨⟨Ū⟩⟩
  exchange needed: it is consumed only by the owned-element trim, and uv's ELEM3D halo follows.
  Then **hbar_old := hbar; hbar := η^{n+1}; eta_n := η^{n+1}** — same timing class as SI, which
  at α=1 also sets eta_n = hbar = the new elevation (step.cpp:942-943).
  **Correctness requirement, not IO convenience:** finalize must `sync_host` {eta_n, hbar,
  hbar_old} every step — two `!fesom_sshrails_on()`-gated host→device pushes OUTSIDE the SI block
  (`fesom_step.cpp:614-616` eta_n before substep 4; `fesom_ice.cpp:646-648` hbar before
  ocean2ice) would otherwise overwrite SE's device-written fields with stale host mirrors (the
  L86 class: no Serial gate can catch a coherence bug — step.cpp:970-975). Those two sites also
  get `&& !fesom_se_on()` so the pushes are structurally dead under `se` (T6).
- Corrector (replaces update_vel): U**ₖ = helemₖ·(uv+uv_rhs)ₖ; per column (owned elements)
  uvₖ := (U**ₖ − (helemₖ/Σhelem)(ΣₖU** − ⟨⟨Ū⟩⟩)) / helemₖ; then the existing
  `fesom_halo_field(uv_fld, ELEM3D, nl, 2, p)` (step.cpp:803 slot). Weights cancel in the vertical
  sum, so Σₖ uvₖ·helemₖ = ⟨⟨Ū⟩⟩ exactly regardless of the weight choice (see D1).
- ALE tail **reused unchanged**: 12b divergence-w from trimmed uv + `fesom_ale_vert_vel_zstar_kk`
  (reads hbar−hbar_old = Δη — exactly what SE wrote) producing corrected w + `hnode_new`;
  substep 14 `fesom_ale_update_thickness_zstar_kk` commits hnode/helem. Tracer advection (13)
  reads uv/helem/w_e as today.
- Skipped under `se`: substep 6b stiffness update, substeps 7–11 **except the dhe fill**
  (step.cpp:833-851 — kept: cheap, still meaningful as the Δη element mean, and
  `fesom_ale_dump_dhe` reads it). Startup stiffness build + CG self-tests stay (harmless, keep
  G0 trivial). Bisect rails: `fesom_ale_dump_hbar` still fires under `se`;
  `fesom_ale_dump_sshsolve` (d_eta/ssh_rhs) is skipped with the block.

## Knobs (proper options — abort on unrecognized; announce once on rank 0)

| Knob | Default | Meaning |
|---|---|---|
| `FESOM_SSH_MODE` | `si` | `si` = current CG path (byte-identical); `se` = this module. Requires `FESOM_ALE=zstar` under `se`. |
| `FESOM_SE_M` | 50 | substeps per baroclinic step; startup CFL check prints the mesh-derived minimum-safe M (from `mesh_resolution` and column depth **−zbar[nlevels_nod2D−1]** — `mesh->depth` is input metadata only, fesom_mesh.h:79-80) and aborts if clearly unstable (override: `FESOM_SE_M_FORCE=1`). |
| `FESOM_SE_CHI` | 0.05 | SM dissipation χ (notes: 0.05–0.1, experimental — G3 scans it). |
| `FESOM_SE_VISC` | 1 | live harmonic barotropic viscosity per substep (D3), two-term structure V[Ūᵐ]−V[Ūⁿ]. Coefficient: the Zenodo formula `vi=dt·√(max(γ₀,γ₁·\|Δu\|)·len)` with the Zenodo SE defaults γ₀=10, γ₁=2750 (γ₂ fixed 0); `FESOM_SE_VISC_GAMMA0`/`FESOM_SE_VISC_GAMMA1` override; calibrated in G3. `0` = off (a G3 arm). |
| `FESOM_SE_CHECK` | unset | G1 diagnostics: per-step invariant norms + one-shot operator self-test; Serial + nonzero ⇒ abort on violation (the `fesom_ale_verify_report_` pattern, fesom_ale.cpp:729). |

Deliberately **not** knobs in v1 (L80/L102 dead-knob rule): trim-h choice (D1, hⁿ hardwired) and
bottom drag (no code yet — knob lands with the feature).

Interplay (enforced at startup under `se`):
- `FESOM_SPEED_SSHRAILS`: force-resolved **OFF with a printed notice** under `se` (it is an
  SI-block lever; `FESOM_SPEED=1` master-on must not abort). Its two out-of-block gated sites are
  additionally `!fesom_se_on()`-guarded (see finalize bullet) — that pairing is what makes
  "ignored" true.
- `FESOM_DIAG_SSHSLV/SPREAD` and `FESOM_KK_VERIFY=ssh` **abort** (they introspect SI-block
  fields; house incompat pattern, fesom_step.cpp:130-149).
- CGPIPE/CGPOLY levers are inert (their init lives inside the CG solve); free hooks are safe no-ops.
- `FESOM_SSH_SOLVER` (M10, worktree — not on this branch): when the tracks merge, `se` must
  abort-or-announce on a non-default setting (dead-knob rule). Recorded here for merge time.

## SE state (all 2-D — memory trivial; device-resident Fields in `fesom_se_state`)

η ring ×3 `[nod2D]`; Ū ring ×3, ⟨⟨Ū⟩⟩, R̄, F, `se_Ubt_n` (m=0 freeze for D3), `se_Ubt_ab2`
(AB2 transport sum for D4) `[elem2D*2]`; H0_e `[elem2D]`; div-gather CSR (`IntField`
offsets/elem-ids + `Field` coefficient pairs); elem→3-edges adjacency for the viscosity gather
(owned elements only — all 3 edges of an owned element are in the myDim+eDim edge range; skip
boundary edges per `myList_edge2D[ed] > edge2D_in`, the Zenodo :853/:958 rule; port convention
fesom_mesh.h:36-41, precedent fesom_ice_evp.cpp:432-438). Alloc when knob on; free hook next to
`fesom_ssh_cgpipe_free()` (fesom_main.cpp:1548-1549 block).

## Development approach

No unit-test framework exists for physics in this repo; per-task "tests" = clean builds of
**both** trees (`build-m7serial`, `build-m7cuda`) + the named gate/self-check runs. ctest targets
must stay green (note: calendar/io_stream/io_config are Serial-build-only, CMakeLists.txt:123-142;
only `field` runs under CUDA). **Null-gate cadence (house rule, fesom_speed.hpp:10-11): every
task that touches `src/fesom_step.cpp` ends with a cheap unset-vs-unset diff_snap re-check** —
not just T1 and T6. Gates are sequential: a task's gate must pass before the next task starts.
Update this file's checkboxes as work lands; ➕ for discovered tasks, ⚠️ for blockers.
House rules bind throughout: `BIN=` pinning on every multi-srun job, same-day pinned perf pairs,
`-C a100_80` on every GPU absolute, cheap gates get `-t 00:06:00`-class walltimes, ≤16 GPU nodes
without asking, binaries to `/work` + sha only (never committed), meshes = private copies.

## Implementation steps

### Task 1: Module skeleton, knob, state, CFL check (no physics)
**Files:** Create `src/fesom_ssh_se.h`, `src/fesom_ssh_se.cpp`; Modify `CMakeLists.txt` (FESOM_SRC
list, :69-86), `src/fesom_main.cpp` (init after `fesom_ale_mode_init()` :347 + free hook :1548),
`src/fesom_step.cpp` (dispatch stub).
- [x] `fesom_se_on()` / knob parsing per the `fesom_wsplit_on` template: DONE. Smoke matrix on
      pi (login, ob1 override): junk value aborts · se-without-zstar aborts · DIAG_SSHSLV combo
      aborts · junk FESOM_SE_M aborts · dead-knob note prints when FESOM_SE_* set with mode off ·
      SSHRAILS force-off early-out added inside `fesom_sshrails_on()` (before the speed resolve,
      so FESOM_SPEED=1 master-on cannot trip the IOACC pairing abort).
- [x] `fesom_se_state` struct + alloc/free (module-static, CGPIPE pattern); free hook wired at
      fesom_main.cpp Kokkos-finalize block. Rings zero-init == cold-start flat rings.
- [x] Startup barotropic-CFL report DONE (0.5·res/√(gH), H=−zbar[nlev−1], one Allreduce; abort
      unless FESOM_SE_M_FORCE=1). pi check: dtbt=2.00 s vs limit 233.65 s, M_min=1.
- [x] Step-driver dispatch stub DONE: `if (!fesom_se_on())` wraps substeps 6b–10 (un-reindented,
      banner comments); cg_iters hoisted; the :752 SI self-containment pushes additionally
      `!fesom_se_on()`-gated; dhe fill + substep 11 + dump_hbar stay shared.
- [x] Build serial+cuda clean; ctest 4/4 (Serial full set).
- [x] **G0a null gate PASS** — refined to the house convention (stronger than the pre-branch
      plan): Serial knob-off byte gate vs the STANDING certified baseline `m6_baseline_serial`
      (jobs/job_m7_gate_serial, job 26935040, diff_snap rc=0) + CUDA fidelity gate
      (jobs/job_m7_gpu_gate, job 26935041, rc=0 — CUDA is never byte-gated: D22 atomic scatter).
      `si`-vs-unset one-shot: bit-identical on pi 10 steps (diff_snap rc=0).

### Task 2: Deterministic 2-D operators (div gather CSR, grad check, viscosity adjacency)
**Files:** Modify `src/fesom_ssh_se.cpp` (+`.h`).
- [x] CSR assembly DONE: per-owned-node rows summed per (node,elem) pair, **sorted by GLOBAL
      element id** (partition-invariant summation order — the T7 exact-0.0 enabler),
      pre-divided by `areasvol[top]` so T = +∂η/∂t (hbar += T·dt exactly as compute_hbar);
      cavity rows empty; myDim_edge2D iteration only (el<0 there = genuine boundary).
- [x] Gradient convention VERIFIED (fesom_momentum.cpp:505-561): `gradient_sca` rows 0-2 =
      ∂N/∂x, 3-5 = ∂N/∂y in `elem_nodes` vertex order; the area factors cancel in the
      assembly, so the η term inside uv_rhs is exactly dt·(−g·Σgsᵢηᵢ) — the r-term
      cancellation partner is exact. Bonus pin for D4: Coriolis AB2 weights are
      old×−(0.5+ε), new×+(1.5+ε) (ff_step, =1.0 on step 1).
- [x] elem→3-edges + 3-neighbours adjacency (owned elements, myDim+eDim edge inversion,
      3-edges-found assert, nb=−1 boundary).
- [x] Host 2-D scatter reference written (the compute_hbar edge loop with h=1, /areasvol).
- [x] **Self-test PASS everywhere** (one-shot under `FESOM_SE_CHECK`; global-id-hashed test
      field ⇒ partition/backend-independent): pi 1r max|Δ|=1.355e-20, pi 2r identical, pi CUDA
      identical (device gather = host), CORE2 128r 8.132e-20 (job 26935186) — all ≈3e-16
      relative = the reassociation floor, well under tol 1e-14·scale.
- [x] Build serial+cuda clean; ctest unchanged.

### Task 3: Forcing assembly (R̄, H0_e, Ū_AB2, F)
**Files:** Modify `src/fesom_ssh_se.cpp`.
- [x] R̄ kernel DONE — one fused owned+eDim column kernel: R̄=Σₖhelem·uv_rhs/τ, Ubt_now=Σₖhelem·uv,
      H_e=Σₖhelem, cavity guard, lean raw-pointer captures.
- [x] Ū_AB2 DONE per D4 (wold=−(0.5+ε), wnew=+(1.5+ε), step-1 = (0, 1.0), ε=0.1;
      `se_Ubt_prev` history swap at block end).
- [x] F kernel DONE with the sign algebra verified in-code against the in-R̄ Coriolis
      (+f·v couplets) and η-gradient (dt·(−g·Σgsᵢηᵢ)): F_x=R̄_x−f·V_AB2+g·H_e·∂xη,
      F_y=R̄_y+f·U_AB2+g·H_e·∂yη. No viscosity in F.
- [x] Viscosity closure RESOLVED (plan amendment): the Zenodo SE defaults are FLOW-DEPENDENT —
      `se_visc_gamma0=10, gamma1=2750, gamma2=0` (MOD_DYN.F90:135-137); form
      vi=dt·√(max(γ₀,γ₁|Δu|)·len), |Δu|=|ΔŪ|/hh, hh=½(H_e1+H_e2), len=√(A₁+A₂), net term
      −Σ_edges(Ū_e−Ū_nb)·vi/A_e. v1 ships THIS closure and defaults (knobs
      FESOM_SE_VISC_GAMMA0/GAMMA1; γ₂ fixed 0). With γ₁≠0 the single-form V(Ūᵐ−Ūⁿ) is NOT
      equivalent — T4 implements the reference's two-term structure V[Ūᵐ]−V[Ūⁿ], each term's
      coefficient from its own state (γ₀-only reduces to the single form exactly).
- [x] **Check** DONE: per-step |R̄|∞,|F|∞,|F−R̄|(=|r|) prints; frozen-stub self-consistency
      confirmed on pi Serial+CUDA (uv=0, η=0 ⇒ r≡0 exactly; |R̄|=9.87e-3 m²/s², identical
      digits both backends).
- [x] Build serial+cuda clean; runs green.

### Task 4: The subcycle loop + finalize
**Files:** Modify `src/fesom_ssh_se.cpp`, `src/fesom_step.cpp` (stub → real block),
`src/fesom_phasestats.h/.cpp` (new phase).
- [ ] SM weight struct (β,ζ,γ,δ from χ; FB slot documented in a comment, not implemented).
- [ ] Kernels 1/2/4 per the scheme + the two per-substep exchanges + ring rotation; ⟨⟨Ū⟩⟩
      accumulation; W term; M from knob; `se_Ubt_n` freeze at m=0; live V_h(Ūᵐ−Ūⁿ) inside
      kernel 4 (`FESOM_SE_VISC=0` bypass).
- [ ] Finalize: both-ring carry; hbar_old/hbar/eta_n writes (device) + the **mandatory** host
      sync of {eta_n, hbar, hbar_old} (coherence requirement — see scheme section).
- [ ] **Phasestats:** add `FESOM_PH_BT` marks around the SE block (the G4 currency; keep the
      CG mark untouched so SI boards stay comparable).
- [ ] **G1a:** `FESOM_SE_CHECK` per-step η-compatibility ‖η^{n+1}−ηⁿ+τ(div⟨⟨Ū⟩⟩+W)‖∞ —
      machine-precision class expected; measure first, then freeze the threshold in code;
      Serial abort on violation.
- [ ] Build serial+cuda; 10-step CORE2 run: G1a passes, η evolves, no NaN. Note: until T5 lands,
      `uv` never receives the step increment (7–11 skipped, trim absent) — the model state is
      unphysical; G1a is internal to the subcycle and valid regardless. Read "no NaN" accordingly.
- [ ] **Null re-check** (step.cpp touched): cheap unset-vs-unset diff_snap, Serial + CUDA.

### Task 5: Corrector trim + velocity writeback
**Files:** Modify `src/fesom_ssh_se.cpp`, `src/fesom_step.cpp`.
- [ ] Trim kernel (per owned-element column, hⁿ weights per D1); uv writeback; the
      substep-9-slot ELEM3D uv exchange retained.
- [ ] **G1b check:** ‖Σₖ uvₖ·helemₖ − ⟨⟨Ū⟩⟩‖∞ machine-precision class after trim (per step
      under FESOM_SE_CHECK; measure→freeze threshold; Serial abort on violation).
- [ ] Build serial+cuda; 10-step run: G1a+G1b green on both backends.
- [ ] **Null re-check** (step.cpp touched): cheap unset-vs-unset diff_snap, Serial + CUDA.

### Task 6: Full dispatch, coherence sites, ALE-tail integration, null gate
**Files:** Modify `src/fesom_step.cpp`, `src/fesom_ice.cpp`, `src/fesom_ssh_se.cpp`,
`src/fesom_main.cpp`.
- [ ] **Neutralize the two out-of-block SSHRAILS-gated pushes under `se`**: add
      `&& !fesom_se_on()` at `fesom_step.cpp:614-616` (eta_n pre-substep-4 push) and
      `fesom_ice.cpp:646-648` (hbar pre-ocean2ice push). These + the finalize sync are the
      coherence contract (L86: no Serial gate catches this class — CUDA run required).
- [ ] Verify-and-wire: 12b vert_vel (divergence-w from trimmed uv) + `vert_vel_zstar_kk` reading
      SE's hbar/hbar_old; substep 14 zstar commit; dhe fill kept; w halos unchanged
      (step.cpp:1055-1065); `fesom_ale_dump_hbar` fires, `fesom_ale_dump_sshsolve` skipped.
- [ ] IO coherence: `ssh` stream resolves eta_n — host default reads the synced host value,
      IOACC device path reads the Field (fesom_io.cpp:962, table :1025); main-loop eta_max print
      (fesom_main.cpp:1271-1281) reads the synced value.
- [ ] Ice coherence: `ocean2ice` reads mesh->hbar (host :71 / device :170). Ice runs BEFORE the
      ocean step each iteration (fesom_main.cpp:1170→1247), so ice sees the previous step's η —
      the same class as SI (whose hbar also updates later, in substep 10); confirm and document.
- [ ] `fesom_timestep` return/print under `se`: return M (or 0) with the main-loop print adjusted
      so campaign logs don't read "0 CG iters" (fesom_main.cpp:1252).
- [ ] **G0 final:** unset-vs-unset vs pre-branch binary, CORE2 128r, Serial + CUDA →
      diff_snap rc=0.
- [ ] **SE-on smoke:** CORE2 128r zstar `se`, 500 steps Serial+CUDA: no NaN, G1 green, η field
      visually sane (agshow map for Nikolay). This is also the first live test of the coherence
      contract on CUDA.
- [ ] Build serial+cuda clean; ctest green.

### Task 7: G1 full invariant suite
**Files:** Modify `src/fesom_ssh_se.cpp` (+ small script `scripts/se_invariants.py` if useful).
- [ ] Layer-sum identity after commit: derive the SE-correct form from the incremental zstar law
      (note: Σₖhnodeₖ ≠ H⁰+η exactly under the incremental law even in SI-zstar — assert the
      identity that IS exact, measured then frozen).
- [ ] w_surf residual print (pre-pin) + conservation window: global ∫η dA drift vs Σ∫W over a
      500-step CORE2 run; thresholds from first measurement, then frozen.
- [ ] MPI-invariance spot check: **2r vs 128r** Serial, 20 steps (1r is confounded: the
      npes==1 startup-sanity block at fesom_main.cpp:557,660-762 mutates uv/hbar/d_eta before
      the loop) — the barotropic path is deterministic gather + exchange, so expect exact 0.0
      on η/Ū; investigate any nonzero.
- [ ] Fix anything the suite catches; freeze all thresholds in code.

### Task 8: G2 cross-backend agreement + determinism (revised registration)
**Files:** gate scripts under `scripts/` (reuse `gpu_fidelity_gate.sh` pattern); `docs/REFERENCE_RUNS.md`.
- [ ] **Same-backend determinism byte-gate (pre-registered: byte-equal):** CUDA run ×2, same
      ranks — η/Ū/⟨⟨Ū⟩⟩ dumps byte-identical. This is what actually tests the no-atomics /
      no-reduction design. A violation here IS a bug by construction.
- [ ] **Cross-backend gate (pre-registered: FMA-floor agreement, NOT bit):** CUDA vs Serial at
      N∈{1,2,10} steps — device fmad stays enabled in certified builds (CMakeLists.txt:47-55;
      the documented M2.1 divergence class), so expect ~1e-16-relative differences with **no
      systematic growth in M or step count**; record the measured floor. Optional true-bit
      check: a dedicated `--fmad=false` diagnostic build dir (never the certified bins).
- [ ] Full-step SE floor: CUDA-vs-Serial diff_snap field maxima at steps {1,10,100} recorded as
      the scheme's floor row in `docs/REFERENCE_RUNS.md` (L79 per-scheme-floor table).
- [ ] Gate wired into a small `scripts/se_gate.sh` (cheap: `-t 00:06:00` class).

### Task 9: G3 physics twins (SLURM campaign)
**Files:** campaign scripts under `m12/` (pattern from m7/m11 campaigns); findings appended here.
- [ ] Arms (CORE2 dt1800, zstar, 128r CPU class + one CUDA rep): SI control ×≥3 demonstrated-
      distinct controls, SE{χ=0.05,M=50}, SE{χ=0.1,M=50}, SE{χ=0.05,M=30}, SE{visc=0},
      SE+wsplit compose arm.
- [ ] 20-step disturbance report vs seed-control spread (M11 graded-tier framework — tiers,
      no binary verdicts) for the headline arm.
- [ ] ≥3000-step stability screen, every arm (rule 0.41: verdicts only at protocol length; dt1800).
- [ ] 1-year SE-vs-SI climate twin (headline arm): mean SSH/T/S/KE maps + the paper's
      difference-class comparison; χ/M sensitivity summary.
- [ ] Verdict block written here + `docs/` note; pick production defaults (χ, M, γ₀).

### Task 10: G4 perf ladder + docs + closure
**Files:** `m12/` runs; `docs/SSH_SE_M12.md` (new); `docs/REFERENCE_RUNS.md`;
`docs/KOKKOS_PORTING_LESSONS.md`; `docs/SYNC_MAP.md`; memory.
- [ ] Freeze candidate bins (`m12/bin/`, sha to /work, never committed).
- [ ] Same-day pinned pairs, min-of-2 (median if base noisy): CORE2 4N GPU, CORE2 16N GPU
      (`-C a100_80`, ladder dt 1800), farc 2048 CPU (dt 900), dars 2048 CPU (dt 120),
      NG5 64 GPU (dt 180) — SI(h17-class stock) vs SE at G3-chosen defaults; `FESOM_PH_BT` vs
      `FESOM_PH_CG` share on the respective sides. Per-mesh M from the CFL check, not a
      global 50.
- [ ] Report BOTH currencies + SYPD at production dt; the 4N-GPU launch-overhead result reported
      honestly whichever way it lands.
- [ ] `docs/SSH_SE_M12.md`: scheme, knobs, gates, numbers, deviations register, wide-halo
      follow-up decision gated on these numbers. `docs/SYNC_MAP.md`: the new SE device-
      authoritative 2-D class + its host consumers and the finalize sync contract.
- [ ] Lessons (KOKKOS_PORTING_LESSONS.md) + memory files updated; plan moved to
      `docs/plans/completed/`.

## Risks

- **SM stability is the experiment** (Sergey: "it is hoped… will add some stability"): χ/M/visc
  arms in G3; FB weights are the documented fallback slot; the live viscosity (D3) is the main
  stability lever and is now correctly specified.
- 4N GPU: ~7 launches/substep with halo pack/unpack ≈ **350+/step at M=50** + 2M exchange
  latencies vs ~30 deterministic-cost CG iterations — SE may lose at low rank counts. Expected,
  measured, reported; wide-halo phase is the remedy if the ladder says so.
- Coherence class (L86): SE makes {eta_n, hbar, hbar_old} device-authoritative with host
  consumers — only the CUDA smoke + G2 catch a violated contract; Serial gates are blind to it.
- Gather-CSR fringe correctness (edge_tri<0 semantics) — T2 self-test cross-checks vs the host
  scatter reference at 128r.
- CORE2 dt1800 ⇒ dtbt=36 s at M=50: comfortable; but dars dt120/NG5 dt180 ⇒ dtbt 2.4–3.6 s —
  M can likely SHRINK there (the CFL check will say); SE cost scales with M.

## Post-completion

- Sergey review packet: deviations register D1–D4, χ/M/γ₀ G3 findings, the compatibility-invariant
  measurements, the H_e=Σhelem finding (reads as a correction to the reference's
  `-zbar_e_bot+mean(eta)` form), wide-halo phase proposal (M12b) with the G4 evidence.
- Upstream note: restart-field list for the Fortran SE (η/Ū rings) — our port is cold-start only.
- M10 cross-track: once both tracks land, a single farc-2048 bake-off (SE vs oati/pcsi) on one
  same-day pinned board; at merge time, add the `FESOM_SSH_MODE`×`FESOM_SSH_SOLVER` interplay
  guard (see Knobs).
