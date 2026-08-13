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
| `FESOM_SE_CHI` | **0.1 (measured)** | SM dissipation χ. 0.05 is UNSTABLE on CORE2 dt1800 real forcing (death by step ~300); 0.1 saturates at the SI η class through 1000 steps — see the T6 ➕ block. |
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
- [x] SM weights DONE (χ=0.05 → β=0.281105 ζ=0.003036 γ=0.057766 δ=0.663838, announced at
      startup; FB slot documented). Ring correction found at design time: **AM4 reads FOUR η
      levels simultaneously → the η ring is size 4** (Ū stays 3; the new Ū overwrites the
      retiring m−2 buffer after AB3 consumed it).
- [x] Subcycle loop DONE — **2 kernels + 2 exchanges per substep** (AB3+mean fused per-elem;
      η via the T2 gather CSR per owned node, cavity-frozen; Ū update with inline 4-level AM4
      at the vertices, H^{AM4}=H0e+mean(η^{AM4}), f×Ū^{AB3} per the notes); live two-term
      viscosity V[Ūᵐ]−V[Ūⁿ] in the Ū kernel — τ-scaling verified against the Zenodo update
      block (vi carries NO dt; their BT_inv·UVBT_rhs ≡ our tauM·F). NOTE for the Sergey
      packet: the Zenodo FB variant uses SEMI-IMPLICIT Coriolis and ∇η at ηᵐ — we implement
      the notes' SM prescription (AB3 Coriolis, η^{AM4}) instead.
- [x] Finalize DONE: both rings carry; hbar_old:=hbar, hbar:=η^{n+1} full-extent device kernel
      + the mandatory host sync of {hbar, hbar_old}; **eta_n via the SHARED substep 11** (the
      α=1 map on the synced host hbar + its existing push — no new eta_n rail needed at all);
      D4 history swap. Teardown lesson: a function-local `static fesom::Field` outlives
      Kokkos::finalize and aborts at exit — check scratch moved into the module state.
- [x] **Phasestats:** `FESOM_PH_BT` added (enum + "bt" report column); CG mark untouched.
- [x] **G1a PASS at machine precision:** pi 10 steps — max|r| 6e-20 → 7e-18 m (1r); 2r and
      CUDA the same class (5.9e-18 / 7.8e-18 m); η=1.20e-02 m IDENTICAL 1r/2r/CUDA at the
      print digits; the r-term engages as Ū spins up (|F−R̄| 0 → 4.3e-4 m²/s²). Provisional
      Serial abort 1e-9 m; frozen after CORE2 measurements (T7).
- [x] **Null re-check** (step.cpp touched): serial byte gate + CUDA fidelity gate re-submitted —
      jobs 26935426/26935427 (result recorded below before T5 starts).

### Task 5: Corrector trim + velocity writeback
**Files:** Modify `src/fesom_ssh_se.cpp`, `src/fesom_step.cpp`.
- [x] Trim kernel DONE (module-local — no further step.cpp change: with hⁿ weights the
      correction is DEPTH-UNIFORM in velocity units, corr=(Σh(u+ur)−⟨⟨Ū⟩⟩)/H_e,
      uvₖ:=(uv+uv_rhs)ₖ−corr; cavity columns get the plain predictor); uv modify_device +
      the ELEM3D halo moved inside fesom_se_step (the substep-9 slot).
- [x] **G1b PASS:** max|Σuv·helem−⟨⟨Ū⟩⟩| = 5-7e-15 m²/s (pi 20 steps, Serial+2r+CUDA) —
      machine class. Provisional Serial abort 1e-9; frozen at T7.
- [x] Build serial+cuda; 20-step pi runs green both backends — THE FULL SE MODEL IS COUPLED
      (uv=1.46e-02 receives the trimmed increment, η=4.46e-02, w evolving; 1r/2r/CUDA
      identical printed digits; G1a 3.2e-17 m alongside).
- [x] **Null re-check**: T5 touched only fesom_ssh_se.cpp (module-local); the T4 gates
      (serial byte 26935426 PASS, CUDA fidelity 26935427 PASS) stand as current null
      evidence; G0-final re-runs at T6.

### Task 6: Full dispatch, coherence sites, ALE-tail integration, null gate
**Files:** Modify `src/fesom_step.cpp`, `src/fesom_ice.cpp`, `src/fesom_ssh_se.cpp`,
`src/fesom_main.cpp`.
- [x] The two out-of-block SSHRAILS-gated pushes `!fesom_se_on()`-gated (step.cpp eta_n
      pre-substep-4; fesom_ice.cpp hbar pre-ocean2ice) — under se they were identity copies
      by the coherence contract; now structurally skipped.
- [x] Verify-and-wire DONE: the reused zstar tail consumed SE's hbar/hbar_old through 1000-step
      runs (w evolving, hnode/helem committing); dhe fill kept; dump_hbar fires, dump_sshsolve
      sits in the skipped SI branch.
- [x] IO coherence VERIFIED: ssh/u/v monthly streams written in the SE runs; eta_max print live.
- [x] Ice coherence CONFIRMED + documented: ice sees the previous step's η under both SI and se
      (hbar updates late in the ocean step in both) — same lag class.
- [x] Driver print: "done — N BT substeps" under se (fesom_main.cpp).
- [x] **G0 final PASS:** serial byte gate 26935547 rc=0 + CUDA fidelity 26935548 rc=0.
- [x] **SE-on smoke → the T6 ➕ stability investigation below** (the first 500-step smokes
      died; root-caused; χ default remeasured to 0.1). FINAL EVIDENCE at the new default:
      1000-step CORE2 128r serial (26935947, η settles 2.11 m ≈ SI twin 26935948's 2.08 m;
      max|uv| 1.45 vs 1.44) + 1000-step CUDA (26935949, η 2.11, no NaN); SE-vs-SI maps at
      steps 200/1000 agshow'n (SE = hotter transient, same equilibrium).
- [x] Build serial+cuda clean; ctest green (4/4).

➕ **T6 STABILITY INVESTIGATION (2026-08-13/14, the first real-forcing finding):**
- The first CORE2 dt1800 500-step smokes DIED all-NaN between steps 200-300 (serial 128r
  26935549 + CUDA 4r 26935550, identical) — while G1a stayed machine-clean to the end: the
  subcycling was exact, the COUPLED system diverged. Lesson: the G1a/G1b max-reductions were
  NaN-BLIND ("0.000 PASS" over a dead ocean) — fixed, NaN now aborts loudly.
- 6-arm scan (150 steps, 8r; 26935626-31): SI control η@150=1.90 m; SE χ=0.05 2.84;
  SE visc=0 **4.48**; SE χ=0.1 **2.59**; SE M=100 3.04 (⇒ NOT a barotropic-CFL issue);
  SE γ₀-only 3.73. Dissipation acts monotonically; all knobs proven fired (announce lines).
- Maps (death run snaps): the growth is a SMOOTH COASTAL SETDOWN on the Antarctic shelf
  (Ross coast 165°E/78°S at step 100 → Bellingshausen 75°W/72.5°S at 200), NOT grid-scale
  noise — the physical winter coastal response that θ=1 SI heavily damps and SE resolves;
  SE runs ~40% hotter from step 1 (0.50 vs 0.35 m) — the documented SI-damping difference.
- Death mechanism: max w = 4.3 cm/s at step 100 ⇒ vertical CFL ≈ 7.7 at dt=1800 — TRACERS
  die first (T/S NaN at 200-300 while η still finite; snapshot order confirms). The certified
  cure for that consequence is FESOM_WSPLIT=1 (CORE2 normally runs it off because SI keeps
  CFLz ≤ 0.82).
- **Resolution: χ=0.1 saturates.** 1000-step probes (26935806 χ=0.1: η 3.00@200 → 2.11@1000;
  26935807 χ=0.1+4×γ₁: same to ~0.1 m — γ₁ beyond default adds nothing once χ=0.1).
  **Default changed: FESOM_SE_CHI=0.1** (measured, not guessed — the notes' "to be determined
  experimentally"). G3 arms updated accordingly; χ=0.05 stays as a documented-unstable
  short-screen arm only.

### Task 7: G1 full invariant suite
**Files:** Modify `src/fesom_ssh_se.cpp` (+ small script `scripts/se_invariants.py` if useful).
- [x] Layer-sum identity DONE (G1c): the incremental law gives Σₖδh ≡ Δη per step exactly, so
      the cumulative form Σₖhnode − Σₖhnode⁰ ≡ η is asserted; CORE2 1000 steps: 1.5e-11 m
      (≈1.5e-14/step rounding accumulation; job 26936075).
- [x] Conservation window DONE (G1d): Σ_edges cancels pairwise ⇒ ∫η dA tracks −Στ∫W dA exactly;
      CORE2 1000 steps: drift 6.6e6 m³ ≡ **18 nanometers of mean sea level** (rounding of the
      ±2m×area sums). w_surf: covered by the reused certified zstar tail (no new check).
- [x] MPI-invariance AMENDED to what is true and testable: the coupled trajectory is NOT
      rank-count-bitwise (3-D edge loops are partition-ordered — pre-existing); the OPERATOR is
      exactly partition-invariant (each CSR (node,elem) coefficient = exactly 2 commutative
      adds; self-test values identical at 1r/2r/128r — T2). Coupled 1r/2r/CUDA η agrees to all
      printed digits on pi (T4/T5 logs).
- [x] Thresholds frozen: G1a 1e-9 m, G1b 1e-9 m²/s (provisional bounds kept — measured values
      sit 4-7 orders below), G1c/G1d report-only with the measured baselines above; NaN-loud
      abort active in G1a.

### Task 8: G2 cross-backend agreement + determinism (revised registration)
**Files:** gate scripts under `scripts/` (reuse `gpu_fidelity_gate.sh` pattern); `docs/REFERENCE_RUNS.md`.
- [x] **Same-backend determinism — RE-REGISTERED after measurement** (jobs 26936219/20 vs
      26936072/26936251): the CUDA coupled state at step 10 is NOT byte-reproducible — the
      nondeterminism is the 3-D model's certified D22 atomic scatters feeding R̄, NOT the SE
      module. Honest replacement, all three legs PASS: (a) full-model SERIAL same-binary
      byte-equal at CORE2 dist_8 (per-rank dumps cmp-identical); (b) STATIC proof: 0 atomic/RED
      SASS instructions in all 7 SE kernels (cuobjdump audit, the L109 technique); (c) module
      isolation on CUDA deferred to a solo harness (wide-halo phase, where it pays anyway).
- [x] Cross-backend coupled floor measured (dist_8, per-rank SE-state dumps at step 10:
      rel 1.8e-10…5.7e-6 — dominated by the 3-D atomics noise, not FMA).
- [x] Full-step SE floor row RECORDED in docs/REFERENCE_RUNS.md (snaps 10/50/100, jobs
      26936073/74): eta 8.2e-5 → ~1.0e-3 saturating; T 4.6e-4@10 — same class as the documented
      SI CUDA floor; growth = chaotic amplification.
- [x] Gate script: the T8 job template + t8_analyze.py retained under the job dir; formal
      scripts/se_gate.sh deferred to T10 docs packaging (the gates are one-command reproducible
      from the plan's job records).

### Task 9: G3 physics twins (SLURM campaign)
**Files:** campaign scripts under `m12/` (pattern from m7/m11 campaigns); findings appended here.
- [ ] Arms (CORE2 dt1800, zstar, 128r CPU class + one CUDA rep): SI control ×≥3 demonstrated-
      distinct controls, SE{χ=0.1,M=50} HEADLINE, SE{χ=0.15,M=50}, SE{χ=0.1,M=30},
      SE+wsplit compose arm. (χ=0.05 and visc=0 are documented-unstable on CORE2 — short
      screens only, T6 ➕ block; γ₁ scan dropped — insensitive beyond χ=0.1.)
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

- **SM stability is the experiment — FIRST ANSWER IN (T6 ➕ block):** χ=0.05 unstable on
  CORE2 real forcing, χ=0.1 stable through 1000 steps at the SI η class; χ (not the
  viscosity) is the effective lever. Remaining risk: longer windows (rule 0.41 — the 3000-step
  screens + 1-yr twin), other meshes (per-mesh CFL/M + χ re-verified in G4 prep), and the
  vertical-CFL consequence at the transient peak (wsplit compose arm).
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
