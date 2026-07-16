# M7 next session (session 13) — PROMPT

*Written 2026-07-16, close of session 12 (Fable). Branch `m7-speed`, sessions 9-12 PUSHED to
origin (`b8d4acc` + this handoff). Read in this order: this file → `docs/GPU_SPEED_M7.md`
(h17 row + the E.EVP1 section) → `docs/plans/20260720-m7-evpwide-design.md` **correction
header** (the four-correction history is the day's lesson set).*

---

## 0. THE RULES (deltas on the session-12 PROMPT §0; everything there still stands)

- **0.27 🔴 THE FIRST WRONG-HIGH PRE-REG (after five wrong-lows): an event-count sizing is
  only valid while the messages STAY EAGER.** EVPWIDE multiplied bytes ×K (+ a 2nd sigma
  msg/partner): widest 122 KB @16N K=4 ⇒ rendezvous/pinned-bounce regime ⇒ K=2/K=4 REGRESSED
  at both scales. Before pricing ANY event-reduction lever, compute the per-partner byte
  growth against `UCX_RNDV_THRESH` and say which side of the threshold each K lands on.
- **0.28 🔴 THE LIBMVEC LESSON: local recompute of ANYTHING transcendental is never
  byte-safe.** gcc -O3 vectorizes mesh loops (SIMD cos/tan differ from scalar in the last
  bit on unlucky inputs). Ship owner bytes for every value a replay consumes (the cgpipe
  rule, no exceptions). A 64-element parity sample proves nothing (CORE2 lucky, NG5 elem 48
  unlucky).
- **0.29 🔴 A scatter-time stash must replay EVERY later mutation of the scattered copies.**
  `orient_cw` runs AFTER `scatter_mesh` (fesom_mesh.cpp:1329) and swaps v1↔v2 of CCW
  triangles (NG5's file is 100 % CCW — it swaps EVERYTHING). EVPWIDE replays it at build.
- **0.30 EVPWIDE state: built, byte-certified at EVERY K (CGPIPE class), K\*=8 modest
  (−2.2 % @16N / −0.6 % @4N), K=2/K=4 regress. STAYS OPT-IN; rescues are STRICTLY OPTIONAL
  (user, 2026-07-16).** No climate leg scheduled unless the user asks.

**Binaries** `m7/bin/…`: ✅ **`h17` = CURRENT BEST, CERT 4/4** (CUDA **`f8384e86`** / Serial
`5c3c90fc`; 4N 0.6382 ⇒ **7.17×**, 16N 0.2413 ⇒ **5.09×**, SYPD **2.65**). `evpw0` v3 =
EVPWIDE lever binary (CUDA `9c900b4f` / Serial `21cea692`; knob-off ≡ h17 — off-legs
reproduced the anchors exactly). 🔴 `h3` broken, never use.

## 1. WHERE THE CAMPAIGN IS

- **8× @4N target (rule 0.21): 0.6382 → 0.5723 = another −10.3 %** (from EVPWIDE-K8's
  0.6341: −9.7 %).
- **Measured pools @4N (16N):** CG residual ~18 (~24) ms — **now the #1 target via CGPOLY**;
  rank imbalance 36.1 (53.0) — recon first; E.1 fuses 7-8 (12-13) ceiling; EVPWIDE rescue
  potential IF the rndv fix works: the modeled −12/−15 minus the realized −3.9/−5.4.
- EVPWIDE K-sweep (std300, evpw0 v3): off 0.6380/0.2412 · k2 +2.0 %/+3.4 % · k4 +1.4 %/+1.7 %
  · **k8 −0.6 %/−2.2 %**. Adoption/climate-leg = user's call, not scheduled.

## 2. SESSION-13 SHAPE

### 2.1 ⭐ THE MAIN LEVER: E.CG2 — CGPOLY (polynomial CG), an OPT-IN knob

**The user (2026-07-16): "the JAX port implemented the CGPOLY and got some good results…
Let's try CGPOLY (also a knob, as it breaks bit identity I think)."**

1. **E.CG2.0 — fetch the JAX implementation FIRST** (`/home/a/a270088/port_jax`, jax_cgpoly
   jobs ran 2026-07-16): which polynomial (Chebyshev/Neumann/Jacobi?), degree, how it is
   applied (preconditioner replacement vs s-step restructuring), iteration counts
   before/after, tolerance/fidelity story, their measured speedup. Get NUMBERS, not vibes
   (feedback-always-measure).
2. **E.CG2.1 — the audit before code (the E.0 discipline)** on `fesom_ssh.cpp`:
   - Count OUR iterations/step (4N + 16N, `FESOM_SPEED_SYNCSTATS` or the solver print) and
     the per-iteration cost split: 1 fused exchange (post-CGPIPE) + how many Allreduces +
     SpMV/precond kernels. What does a degree-d polynomial preconditioner do to each? (Fewer
     iterations ⇒ fewer exchanges AND fewer allreduces — it attacks the pool CGPIPE cannot:
     the iteration count itself. But each precond application may cost d SpMVs ⇒ d halo
     exchanges unless the CGPIPE ring machinery covers the extra reach — audit whether the
     2-ring rr extension suffices for small d or needs deepening. 0.27: byte-growth check!)
   - Composition with CGPIPE: pr_values rows are shipped ONCE (valid under zstar because
     never refreshed) — does the polynomial change what must be shipped? Does it keep the
     ship-once property?
   - Pre-register: expected iters saved × ms/iter (measured), the knob
     (`FESOM_SPEED_CGPOLY=<degree?>`, opt-in `_exp`, master never implies it), floors/
     ceilings, and the certification ladder. **Expect NOT byte-identical** (the Krylov
     trajectory changes ⇒ different converged iterate within tolerance): cert = knob-OFF
     byte + CUDA fidelity + options ×3 + logged iteration counts + A/Bs both scales +
     (user) climate leg. If a FORCE_SERIAL-provable formulation exists (exact same iterate
     sequence, unlikely), take it — but do not force it.
3. **E.CG2.2 — build + ladder + A/B** (the CGPIPE/EVPWIDE per-lever ladder verbatim; freeze
   AFTER the last edit on BOTH backends, 0.23).

### 2.2 STRICTLY OPTIONAL (user's words) — EVPWIDE rescues, only if cheap and convenient
- (a) env-only: `job_m7_ab_env` legs at K=4/K=8 with `UCX_RNDV_THRESH` raised past the wide
  message sizes (95-122 KB ⇒ try 256 KB; check `UCX_RNDV_THRESH` default in the ucxdiag log
  first). No code, no refreeze — BIN=evpw0 v3.
- (b) small code: fuse the node+sigma segments into ONE message/partner (halves msgs per
  refresh). Needs refreeze + the byte-gate ladder again (cheap: 4 serial gates).
- Do NOT let this displace CGPOLY.

### 2.3 Cheap tracks still on the table (unchanged from session-12 §2.4)
E.1a T+S post-trdiff fuse (~30 min, byte-identical, −0.55/−0.91 ms) · E.1b/c KPP+ice-FCT
fieldN fuses (−7-13 ms ceiling) · imbalance recon (36/53 ms pool: per-rank 2D-verts,
ice-active fraction, CG rows, kernel-busy from any census sqlite).

## 3. STANDING MACHINERY (unchanged — see session-12 PROMPT §3)
`scripts/m7_halo_sites.py` (L100) · `m7_gap_census.py --min-gap-ms 0.1` · E.split
instruments · `job_m7_ab_env` (LEG1..4, BIN=, NSTEPS=300) · `job_m7_gate_serial` (KNOBS =
FORCE_SERIAL byte proof) · `job_m7_gpu_gate` (KNOBS+SREF+BIN; options SREFs in its header)
· `jobs/m7_provenance.sh` · EVPWIDE debug: `FESOM_EVPWIDE_SELFCHECK=1/2/3`,
`FESOM_EVPWIDE_RINGS`; the pre-step uv echo MUST print 0.000e+00.

## 4. USER PREFERENCES / DECISIONS (standing + session-12)
- 8× is the target (0.21). CGPIPE adopted (h17). **EVPWIDE opt-in, K\*=8, rescues strictly
  optional. CGPOLY = the next lever, opt-in knob, expected non-bit-identical.**
- Always measure; pre-register BEFORE submission; SHA/md5 on every harvest; same-day anchors
  `-C a100_80`; A/Bs same-alloc; options ×3 in the per-lever ladder (L91).
- Sessions 9-12 pushed (user-approved 2026-07-16). Ask before any future push.
- Mixed precision BANNED. Output under `/work/ab0995/a270088/port2/m7/`. CORE2 gates =
  private mesh (L73).
