# M7 session 13 — findings (2026-07-16 evening, live document)

*Branch `m7-speed`, baseline h17 (CUDA `f8384e86` / Serial `5c3c90fc`; 4N 0.6382 ⇒ 7.17×,
16N 0.2413 ⇒ 5.09×, SYPD 2.65). Main lever per the session-13 PROMPT: **E.CG2 CGPOLY**
(user 2026-07-16: "Let's try CGPOLY, also a knob, breaks bit identity I think").*

## 1. E.CG2.0 — the JAX implementation + numbers (fetched FIRST, per the prompt)

Source: `/home/a/a270088/port_jax` commit `00f6e3c` (`fesom_jax/ssh.py:369-446`), findings
`docs/plans/20260716-m7levers-session3-FINDINGS.md` §4, tune log 26312901.

- **What it is:** degree-k **Chebyshev polynomial preconditioner** over the diag-scaled
  operator, replacing the MITgcm M⁻¹ in the SAME PCG (not s-step, not a solver rewrite).
  `M⁻¹ = p_k(D⁻¹S)·D⁻¹ = D^{-1/2}·p_k(D^{-1/2}SD^{-1/2})·D^{-1/2}` — SPD as long as the
  spectrum ≤ λmax (κ mis-guess degrades efficiency, never correctness).
- **Recurrence:** classical Chebyshev semi-iteration (Saad Alg. 12.1) from z₀=0, STATIC host
  coefficients; apply = k SpMV(+halo), **zero dot products** (that is the point: per-iteration
  collective count unchanged, ITERATION count drops).
- **Bounds:** λmax of D^{-1/2}SD^{-1/2} via fixed-seed 100-iter power iteration at setup,
  ×1.05 safety (power iteration converges from below); λmin = λmax/κ, **κ=30 default**.
  CORE2 measured lam=[0.06009, 1.803].
- **Iteration counts** (real CORE2 step-1 rhs, equal UNPRECONDITIONED-residual tolerance):
  MITgcm 127 → k=2: 55 (**2.31×**) → k=3: 42 (**3.02×**).
- **A/Bs (bench-finite clean ×4 each):** CORE2-8 **−20.7 %** (72.26→57.06 ms/step) ·
  NG5-64 **−9.6 %** (515→460) · dars-32 dt120 **−3.6 %** (333.6→321.6). "Pays everywhere."
- **Tune (26312901, core2-8):** off 72.52 · k2 58.18 · **k3 57.38 · k4 57.27 (plateau)** ·
  k3-κ100 62.75 (**κ=100 regresses — κ=30 confirmed**).
- **Fidelity class:** solver-tolerance-equivalent, NOT byte-identical (early-stopped iterates
  agree to 2.3e-5 rel; both valid soltol-1e-5 solutions). Default OFF = byte-identical.
- Their NG5-64 shard = 115k nodes/GPU = **exactly our NG5@16N regime** (per-rank proxy) ⇒
  −9.6 % is the best external prior for our 16N central.

## 2. E.CG2.1 — the audit (fesom_ssh.cpp, measured numbers only)

### 2.1 Our solver = the same animal the JAX lever attacked

- Preconditioner (`fesom_ssh.cpp:261-275`): **the MITgcm-style symmetric M⁻¹** (pr_values:
  diag→1/diag, off→−0.5·(off/diag_r)/(diag_r+diag_c)) — the JAX baseline exactly, so the
  measured iteration RATIOS (2.31×/3.02×) are transferable priors.
- Tolerance: `soltol=1e-5`, residual = √(rr·rr/nod2D) vs rtol=soltol·√(rhs·rhs/nod2D) —
  UNPRECONDITIONED, same class as the JAX equal-tolerance gate. maxiter 500.
- pr_values frozen at first build, never refreshed (zstar increments touch `values` only) —
  the Fortran freeze precedent (oce_ale.F90:3306) that CGPOLY will inherit.

### 2.2 Iterations + per-iteration cost (h17 = CGPIPE adopted)

- **iters/step ≈ 71.9 settled @NG5 dt180** (session-7, h9 300-step trace 26258712 + fresh 16N
  CPU log independently; cold start ~86-90). Iteration count is a mesh+dt property — same at
  4N and 16N.
- Per iteration (post-CGPIPE): **1 fused 2-ring rr exchange** (`cgpipe_exchange_rr`) +
  **2 MPI_Allreduce calls** (1-elem pp·App + 2-elem {rr·zz, rr·rr}) + 2 fused SpMV+dot
  parallel_reduce + zz_ring1 + axpy + pp-update (over N+eDim).
- Measured pools (session-11 E ledger + session-13 prompt §1):
  - pre-CGPIPE CG halo: 146 ev/step = 27.3 @4N / 38.5 ms @16N (187/264 µs/event);
  - CGPIPE deleted 72 ev/step ⇒ measured **−9.0 ms @4N / −21.2 ms @16N** (marginal
    **125 / 295 µs per deleted exchange**);
  - post-CGPIPE CG halo residual **~18 @4N / ~24 ms @16N** (74 ev/step) + **144 Allreduce
    calls/step** + CG GPU ~12 ms in ~351 launches (~34 µs/launch, launch-bound).
  - CG region total 43.9 ms/step = 6.5 % @4N on h9 (pre-CGPIPE; ⇒ ~35 ms on h17).

### 2.3 Composition with CGPIPE — the ring theorem (the audit's core result)

CGPIPE's per-iteration structure survives a polynomial preconditioner **iff the machinery
deepens with the degree**:

- The apply consumes rings: producing zz on owned+ring1 (required for the pp recurrence that
  deletes `exch(pp)`) with d inner SpMVs needs **rr exchanged on R = d+1 rings**, the frozen
  operator rows on **rings 1..d**, and D̃⁻¹ on **all d+1 rings**. Each semi-iteration's row
  extent shrinks by one ring (the EVPWIDE frontier argument, applied to CG).
- **d=1 fits today's 2-ring graph exactly** (null-ish rung). d=2 ⇒ R=3, d=3 ⇒ R=4: the
  cgpipe row-shipping build generalizes round-by-round (each shipped CSR row's colind gids
  reveal the next ring; want-list handshake per round; owners ship rows VERBATIM in row order
  — the byte-replay argument unchanged).
- **Ship-once is PRESERVED by freezing the preconditioner operator:** M⁻¹ =
  p_d(D̃⁻¹Ã)·D̃⁻¹ with **Ã = the stiffness snapshot at preconditioner-build time** (owned
  rows: device copy of `values`; ring rows: shipped once) and D̃ = diag(Ã). Under linfs
  Ã ≡ A forever; under zstar A drifts from Ã exactly as it already drifts from pr_values —
  the SAME Fortran freeze precedent, so nothing is ever refreshed. (Consistency requires the
  OWNED cheb SpMVs to also use Ã, not live A — otherwise M is non-symmetric.) pr_values are
  simply NOT shipped/used when CGPOLY is on; the actual solve operator A·pp stays live-zstar
  at owned rows, untouched.
- Rule 0.28 (libmvec) is satisfied by construction: every ring value is shipped owner bytes;
  nothing is recomputed locally.

### 2.4 Rule 0.27 — byte growth vs UCX_RNDV_THRESH (the wrong-high check)

- Measured 2-ring fused exchange today (cgpipe build prints): NG5@4N ring2(max)=3092,
  7 partners; @16N ring2(max)=1949, 10 partners; ring1 ≈ ring2 size class ⇒ total recv
  ≈ 49 KB @4N / 31 KB @16N per rank, mean/partner ≈ 7 / 3 KB, worst partner ~15-20 / ~8-10 KB.
- R=4 (d=3) ≈ ×1.9-2.1 the 2-ring bytes ⇒ worst partner ≈ **30-40 KB @4N, 16-20 KB @16N**.
- `UCX_RNDV_THRESH` resolves `auto` (intra+inter) on Levante — no fixed number to check
  against, so the build **prints the per-partner max bytes** and the pre-reg carries the
  EVPWIDE-signature watch: if d2/d3 regress while d1 pays, suspect an eager→rndv flip and run
  the env-leg rescue (`UCX_RNDV_THRESH=256k`), exactly the session-12 lesson.
- Unlike EVPWIDE (bytes ×K at CONSTANT event count), CGPOLY grows bytes ~×2 while cutting
  events ×3 — the exposure is structurally smaller.

### 2.5 Event arithmetic (per step, 72 → 72/ratio iters)

| config | exchanges | Allreduce calls | SpMVs | ceiling model Δ @4N / @16N |
|---|--:|--:|--:|---|
| h17 today | 74 (2-ring) | 146 | 144 | — |
| d=1 (R=2) | ~47 | ~92 | ~90 | −7 / −13 ms (iters ÷~1.6, unmeasured — gate will tell) |
| d=2 (R=3) | ~33 | ~64 | ~93 | −12 / −20 ms |
| d=3 (R=4) | ~26 | ~50 | ~96 | **−14..16 / −24..28 ms** |

Model: per-iter marginal = exchange (125/295 µs, +10-30 % for deeper rings) + 2 Allreduces
(~60/120 µs) + kernels/compute (~170-250 µs, 34 µs/launch × ~8 + SpMV growth). Iteration
ratios = JAX-measured 2.31×/3.02× at d=2/3.

## 3. E.CG2 PRE-REGISTRATION (frozen before code, 2026-07-16)

- **Knob:** `FESOM_SPEED_CGPOLY=<d>` (int ≥1; unset/0 = OFF = byte-identical). OPT-IN:
  value knob via `fesom_speed_int` — the master never implies it. `fesom_speed_int` is
  SILENT ⇒ **explicit `[cgpoly]` announce** (degree, κ, λ bounds, rings, worst-partner
  bytes) — the L80/EVPWIDE lesson. `FESOM_CGPOLY_KAPPA` (default 30.0).
  Requires the device-halo transport (same conjunction as CGPIPE) or Serial FORCE_SERIAL;
  npes==1 allowed (rings empty). When active it SUPERSEDES the CGPIPE path (announced);
  CGPIPE alone = certified h17 path, untouched.
- **Mechanism:** frozen-operator Chebyshev (§2.3): R=d+1 ring build (generalized cgpipe
  rounds, lazy at first solve), λmax by 100-iter symmetric power iteration on device
  (deterministic splitmix64(gid) start vector, ×1.05, κ=30), apply = d fused
  SpMV+recurrence kernels over shrinking ring extents with double-buffered z, output zz on
  owned+ring1 ⇒ pp recurrence keeps `exch(pp)` deleted; 1 fused R-ring rr exchange +
  2 Allreduces per iteration.
- **Expected fidelity class: NOT byte-identical when ON** (different Krylov trajectory,
  same unpreconditioned tolerance) — the JAX class (2.3e-5 rel iterate agreement). Knob-OFF
  byte-identical. If iteration counts ever hit maxiter or s_aux ≤ 0 (indefinite M = bounds
  broken), the solver dies loudly.
- **Pre-registered numbers (NG5 std300, h17 anchors 0.6382 / 0.2413):**
  - **16N: ceiling −28 ms (−11.6 %), central −8..10 % ⇒ 0.217-0.222 s/step (ratio
    5.53-5.66, SYPD@dt240 ≈ 2.85-2.95), floor 0.** External prior: JAX NG5-64 −9.6 % at the
    matched 115k/rank shard.
  - **4N: ceiling −16 ms (−2.5 %), central −1.5..2.5 % ⇒ 0.622-0.629, floor 0.**
  - **d\* rule (pre-registered): d\* = argmax 16N saving, subject to 4N not regressing
    > +0.3 %; expected d\*=3 with d2 close behind (JAX k3≈k4 plateau ⇒ no d=4 leg).**
  - dt240 bonus (not banked): CG share shrinks ⇒ the ×1.03 SYPD correction shrinks.
- **Kill-fast criterion (BEFORE the A/Bs):** CORE2 np8 gate iters/step must drop ≥1.8× at
  d=2-3 at equal tolerance (the E.3 verify line). If not, stop — do not spend the A/B.
- **Certification ladder** (CGPIPE-class, adapted for the non-bit class):
  1. login np1 pi + np2 CORE2 smokes (announce fires, converges, iters drop, selfcheck 0).
  2. `job_m7_gate_serial` knob-OFF byte gate — rc=0 (default path untouched).
  3. Serial np8 CORE2 FORCE_SERIAL **structural** gate: `FESOM_CGPOLY_SELFCHECK=1` (the
     ring-replay bitwise check: ring-extent apply ≡ reference exchanged apply, max|Δ| MUST
     print 0.000e+00 — the cgpipe selfcheck argument) + logged iters off/d2/d3 (the ≥1.8×
     verify). NOT a byte-vs-baseline diff (expected ≠, pre-registered).
  4. CUDA fidelity gate: `FESOM_SPEED=1;FESOM_SPEED_CGPOLY=3` vs m6_baseline_serial —
     standard climate-close floors (solver wiggle ≪ CUDA atomics floor).
  5. Options ×3 (TKE / mEVP / zstar) with CGPOLY=3: same floors vs each M6 oracle.
     **Pre-registered expectation: the zstar Kv control lands at the ~9.5e-02 MAGNITUDE but
     is NOT bit-equal to 9.537e-02** — a solver-class lever legitimately breaks the L79
     exactness control; magnitude-class pass is the criterion.
  6. A/Bs 16N + 4N, legs `off / d1 / d2 / d3`, std300 same-alloc, `BIN=` frozen `cgpoly0`
     (freeze AFTER the last edit on BOTH backends, 0.23; md5 in PROVENANCE + harvest).
  7. Climate leg (1-yr) = the user's call (opt-in lever; not scheduled).
- **Watch items:** per-partner byte print vs the EVPWIDE regression signature (§2.4);
  iteration-count creep over long zstar runs (frozen-Ã quality decay — benign, monitor);
  `FESOM_KK_VERIFY=ssh` aborts under CGPOLY (C twin runs the legacy solver — clone the
  EVPWIDE guard); CGPOLY+EVPWIDE composition untested (both opt-in; orthogonal machinery).

## 4. E.CG2.2 — BUILT (both backends compile; login smokes ALL GREEN)

Implementation (all in `fesom_ssh.cpp` after the CGPIPE section; ~750 lines):
`CgPolyState g_cgpoly` + `cgpoly_build` (round-based R-ring discovery: round 1 = com-driven
frozen-Ã row ship, rounds 2..d = want-list row ship, round d+1 = diag-only; flat per-partner
lists `[ring1|ring2|…|ringR]`; frozen Ã snapshot DEVICE-authoritative; distributed λmax power
iteration, splitmix64(gid) start vector) + `cgpoly_exchange` (fused R-ring, tag 2110) +
`cgpoly_f0`/`cgpoly_semi` (shared kernels, double-buffered z, shrinking extents, last semi
writes zz directly) + `cgpoly_apply` + `cgpoly_selfcheck` (reference = SAME kernels, owned
extent + re-exchange per semi ⇒ ring replay must match owner bytes BITWISE) + solver wiring
(supersedes CGPIPE when active; exch(pp) stays deleted; separate dot2 keeps ONE 2-elem
Allreduce; sp0<0 indefinite-M abort; KK_VERIFY=ssh abort guard; per-solve iters log).
`FESOM_CGPOLY_KAPPA` env (default 30). Teardown `fesom_ssh_cgpoly_free` in fesom_main.

**L0 login smokes (Serial build, pi mesh):**
- np1 knob-off: runs, 2 CG iters/step (trivial solve), rest-state ≈0 unchanged.
- np1 CGPOLY=2: announce fires, lam=[0.0379, 1.1376], converges (5-6 iters — MORE than the
  MITgcm 2: expected on a trivially-conditioned system; the κ=30 window is pessimal when the
  true spectrum clusters near 1. The kill-fast metric lives at CORE2 dt=1800.)
- **np2 CGPOLY=3 + SELFCHECK: every apply prints max|apply − exchanged ref| = 0.000e+00** —
  the ring-replay byte-identity holds through all 3 semi-iterations. λ bounds BYTE-IDENTICAL
  np1 vs np2 (decomposition-independent power iteration confirmed).
- **np2 knob-off, 20 steps: diff vs the frozen h17 Serial binary rc=0 — ALL FIELDS
  BIT-IDENTICAL** (the old pi np2 oracle was purged from /scratch; h17-frozen is the
  stronger reference anyway).

**Gate fleet submitted (build-tree binaries; freeze = cgpoly0 after gates if no edits):**
26313389 serial 3-leg gate (knob-off byte diff + selfcheck zeros + the E.3 ≥1.8× iters
verify at CORE2 dt1800) · 26313390 CUDA fidelity (SPEED=1+CGPOLY=3) · 26313391/92/93
options TKE/mEVP/zstar (each vs its own M6 oracle; zstar Kv expected ~9.5e-02 magnitude,
NOT bit-equal — pre-registered §3).
