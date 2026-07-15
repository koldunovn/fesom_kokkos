# M7 session 10 — findings (written as the work happens)

*Branch `m7-speed` @ `0f7eaee`. Plan: `20260718-m7-session10-PROMPT.md`. User decision standing:
~7× is the 4N endpoint; effort moves to 16N. This session: (1) harvest the queued h11 16N leg
26267149 (est. start 2026-07-15T19:03, pre-reg 0.2650 −1.4 ±0.5 %); (2) fire the gap300_16N
census on h14; (3) the user-requested cross-mesh dividend survey (row0 `02c8a0d1` knobless vs
h14 `18275c68` blessed, same-day pinned pairs).*

---

## 1. PRE-REGISTRATIONS (recorded BEFORE any submission, 2026-07-15 ~14:50)

### 1.1 The dividend survey — design restated

Per point: **two single-leg `job_m7_ab_env` jobs, same day, both `-C a100_80`, min of 2 reps.**
Leg A = `m7/bin/row0/fesom_port_cuda` (`02c8a0d1`, campaign start, KNOBLESS — `LEG1="row0::"`,
the job unsets every `FESOM_SPEED*` in the leg subshell). Leg B = `m7/bin/h14/fesom_port_cuda`
(`18275c68`, `LEG1="h14::FESOM_SPEED=1"`). Same-day pairing makes the Δ immune to cluster
drift (L94); it is NOT a same-alloc A/B, so each point is quoted (row0, h14, Δ %) + both job
ids. Announce audit at harvest, BOTH directions (L80): h14 log must show the speed announce
lines; row0 log must show none. md5 via PROVENANCE.txt on every harvest.

Protocols: NG5 300-step dt180; dars **150-step** dt180 (L95); farc 300-step dt180 (**L95
drill** — no validated M7 protocol; if both legs die deterministically at the same step, the
honest fallback is a shorter matched window); core2 300-step **dt1800** (same drill; /pool
mesh — perf pairs are same-mesh-vs-same-mesh, only CORE2 *gates* need the private mesh, L73).

### 1.2 Pre-registered numbers (bands where anchors exist, models declared)

Model inputs: the 35-step-era row0 rows (NG5@4N 1.2796, NG5@8N 0.7381, NG5@16N 0.4487-mixed-hw,
dars@8N 0.3178) are CONTAMINATED (getcoeffld, asymmetric) and sit on windows that include the
CG spin-up ⇒ on the clean 300-step protocol row0 should read ~4–7 % FASTER than those rows
(artifact dilution ~0.7 % + spin-up window ~4 %). The old dars row is additionally on a
different protocol (dt/window) ⇒ wide band. h14 legs: anchored where measured.

| point | row0 pre-reg (s/step) | h14 pre-reg (s/step) | Δ pre-reg |
|---|--:|--:|--:|
| NG5@4N | 1.21 ±4 % | 0.6495 ±0.5 % (anchor 26271441) | **−46.5 ±2 %** |
| NG5@8N | 0.70 ±4 % | 0.4015 ±0.5 % (26267148 − kernel-share of TDMANOINIT) | **−42.5 ±2.5 %** |
| NG5@16N | 0.42 ±5 % (mixed-hw row −3 % hw, −4 % window) | 0.2646 ±1 % | **−37 ±4 %** |
| dars@8N | 0.27 ±8 % | 0.1975 ±1 % (26267150 scaled) | −27 ±7 % |
| dars@4N | ~0.55 (no prior) | record | −30 ±8 % |
| dars@2N | ~1.1 (no prior; 395k verts/rank ≈ NG5@4N regime by the per-rank proxy) | record | −38 ±8 % |
| farc@2N (80k/rank ≈ dars@8N regime) | record | record | −25 ±8 % |
| farc@4N (40k/rank) | record | record | −20 ±8 % |
| farc@8N (20k/rank, deep comm-bound) | record | record | −15 ±10 % |
| core2@1N (32k/rank, NO inter-node comm) | record | record | wide: −20…−45 % (fixed host costs are a large share of a short step) |
| core2@2N (16k/rank) | record | record | smaller than core2@1N per ordering (a) |

**Ordering pre-registrations (the regime read — these are the real predictions):**
- **(a)** within each mesh, |Δ| DECREASES as node count increases (comm share grows);
- **(b)** at matched node count, dars |Δ| < NG5 |Δ| (dars@8N sits at NG5@16N's per-rank size);
- **(c)** points at matched 2D-verts/rank should show comparable |Δ| across meshes (the
  per-rank proxy, halo-faithful only);
- any point that breaks (a)/(b) is a FINDING (L84 precedent), not noise — chase it.

### 1.3 gap300_16N census on h14 (E's design input)

Same recipe as 26267784 (`NSYS_TRACE=cuda,mpi NSYS_SAMPLE=none NSTEPS=300 KNOBS=FESOM_SPEED=1`)
but 16N (`-N16 --ntasks=64`, ng5 dist_64), BIN=h14. Pre-register: (i) **rc=0** (h14 carries the
teardown fix — rc=134 here would be a NEW bug per rule 0.14); (ii) the E pool (halo self-gaps,
MPI-covered) GROWS from 18 ms/4N — the comm-bound side is exactly where the ratio decays;
(iii) kernel-busy per step SHRINKS ~4× from 543 ms (strong scaling of kernel work), so E's
*share* grows much faster than its absolute size.

### 1.4 The queued h11 16N leg (26267149)

Pre-reg unchanged from session 9: **0.2650 −1.4 ±0.5 %** ⇒ SYPD@dt240 ≈ 2.40. BIN=h11
(`d74d31b4` — verify SHA.txt). The fresh h14@16N survey leg (§1.2) is the h14 ledger 16N row.

## 1.5 C.2b FERNOINIT + C.3a VISCNOINIT — built; ladder pre-registration (BEFORE submission)

Per the PROMPT §2.3: C continues as strict reductions only. Audit of `fer_solve_gamma_kk`
(fesom_gm.cpp) found three deletions (zbar_n/Z_n init never read — inner⊆outer bounds now
FESOM_CHECKed once at mesh setup; tr_x/tr_y init → the 2 load-bearing Dirichlet zeros per
component; fer_gamma zero-fill → complement only). `impl_vert_visc_kk` (fesom_momentum.cpp)
drops its depth-bounded zbar_n/Z_n init for multi-layer columns (2-layer columns keep legacy —
they read Z_n[nzmax]/Z_n[nzmin−1], slots neither init writes; the TDMANOINIT branch precedent).
Both `_exp` (rule 0.13). h15 probe rung frozen: CUDA `ce1e859d` / Serial `70aff24a`; REG
fer 43→45, visc 82=82, STACK 7168 both unchanged.

**Ladder pre-registration:** 9 gates all PASS; knob-OFF byte-identical (also proves the new
unconditional mesh CHECK is silent on the private mesh); FORCE_SERIAL byte proofs rc=0 for
each lever in isolation AND full-blessed+both; announce line present in every knob-ON leg,
absent knob-OFF (L80 both directions); zstar controls Kv 9.537e-02 AND Av 9.869e-02 to the
digit (rule 0.15); every leg run rc=0 (rule 0.14 — teardown wart is dead).
**A/B pre-registration deferred until the ladder is green** (recorded in §4 below before any
A/B submission). Sizing note (not a pre-reg): FERNOINIT deletes ≈2.2 KB/thread of local
stores + ≈0.6 KB/thread of global stores (nl=70, NG5) ≈ 1.0+0.27 GB/step ⇒ TDMANOINIT-scaled
expectation ≈ −0.15…−0.2 %; VISCNOINIT ≈ 0.45 GB/step ≈ −0.07 % — below single-A/B
resolution, will need the ncu counter for attribution.

ncu-prep (measured from the gap300_h11 sqlite, per the 0.9/session-9 SKIP rule): BOTH kernels
launch exactly **1×/step** (299 each over the 299 traced steps) ⇒ a two-kernel regex = 2
launches/step; recompute NCU_SKIP/NCU_COUNT from that before any ncu submission.

**C.4 audit (momentum_adv, the #2 local pool 19.5 GB/step): PARKED with cause.** The spiller
is `fesom_momadv_vert`'s wu/wv pair — their zero-init is the ACCUMULATOR BASE for the
element-gather (+=) and wu[bl+1] is READ as the init zero (ble ≤ bl for every contributing
element ⇒ nothing ever writes it) — the init is load-bearing end to end; no strict deletion
exists. Only the `un` output zero-fill is complement-restrictable (~0.26 GB/step global,
≈ −0.04 % — below any A/B's resolution). The C strict-reduction tail is, as the PROMPT
priced it, ~1-3 ms total: FERNOINIT+VISCNOINIT is most of it.

## 2. SUBMISSIONS (job ids filled at submit time)

All submitted 2026-07-15 ~14:55, all `-C a100_80` (job-script default), all pinned `BIN=`
(row0 `02c8a0d1` / h14 `18275c68`, md5-verified pre-submit). Survey tags:
`div_<mesh>_<N>n_<leg>` under `/work/ab0995/a270088/port2/m7/`.

| job | what | walltime |
|---|---|---|
| (queued, session 9) 26267149 | h11 NG5@16N std300 (est. start 19:03) | 25 min |
| **26274311** | gap300_16n_h14 census (§1.3), `-N16 --ntasks=64` | 25 min |
| 26274324 / 26274325 | core2@1N row0 / h14 (300-step dt1800, L95 drill) | 12 / 12 min |
| 26274326 / 26274327 | core2@2N row0 / h14 | 10 / 10 min |
| 26274328 / 26274329 | dars@2N row0 / h14 (150-step dt180) | 20 / 18 min |
| 26274330 / 26274331 | dars@4N row0 / h14 | 15 / 13 min |
| 26274332 / 26274333 | dars@8N row0 / h14 | 10 / 10 min |
| 26274334 / 26274335 | farc@2N row0 / h14 (300-step dt180, L95 drill) | 15 / 13 min |
| 26274336 / 26274337 | farc@4N row0 / h14 | 12 / 12 min |
| 26274338 / 26274339 | farc@8N row0 / h14 | 10 / 10 min |
| 26274340 / 26274341 | NG5@4N row0 / h14 (300-step dt180) | 30 / 22 min |
| 26274342 / 26274343 | NG5@8N row0 / h14 | 25 / 18 min |
| 26274344 / 26274345 | NG5@16N row0 / h14 (the priority 16-node survey ask; the h14 leg doubles as the h14 ledger 16N row) | 25 / 20 min |
| 26274728 / 26274729 | dars@2N **dt120** rescue pair (§3.2) | 20 / 18 min |
| 26274906 / 07 / 09 / 10 | h15 Serial gates: knob-OFF byte / FORCE_SERIAL FERNOINIT iso / VISCNOINIT iso / full-blessed+both | 20 min each |
| **26278160** | h15 3-leg A/B (§3.0c): ref / +FERNOINIT / +both, NG5@4N 300-step same-alloc | 50 min |
| 26274911 / 13 / 15 / 16 / 17 | h15 CUDA gates: fidelity iso / fidelity full / options TKE / mEVP / zstar | 10 min each |

## 3. HARVEST (filled as jobs land)

### ⭐ 3.-1 Job 26267149 — the h11 16N leg (the session's #1 harvest)

**0.2629 s/step** (h11 `d74d31b4` ✓ SHA checked, 16 × pure a100_80 ✓, min of 2, spread
0.11 %). Pre-reg 0.2650 ±0.5 % → measured 0.8 % BETTER, 0.3 % below the band floor — the
THIRD consecutive at-scale under-run (8N −0.7 %, dars@8N −0.75 %, session 9): H.9's
retention away from 4N systematically beats the L84(b) 60 % model. **Stage-2 SYPD@dt240 =
0.657/0.2629/1.03 = 2.43** (2.45 with the 4N-measured ×1.019), up from h9's 2.37. 16N ratio
vs the h9-era CPU row (1.2267, cross-day caveat): **4.67×** (h9: 4.56×). E's re-size at 16N
waits on the census (26274311).

### 3.0 h15 ladder — Serial half: 4/4 GREEN (~2 min wall each)

| gate | result |
|---|---|
| 26274906 knob-OFF byte | ✅ rc=0, diff_snap rc=0 (also: the new unconditional mesh-invariant FESOM_CHECK is silent on the private mesh) |
| 26274907 FORCE_SERIAL, FERNOINIT iso | ✅ diff_snap rc=0 — all three deletions hold in bytes |
| 26274909 FORCE_SERIAL, VISCNOINIT iso | ✅ diff_snap rc=0 — the multi-layer deletion + 2-layer legacy branch hold in bytes |
| 26274910 FORCE_SERIAL, full blessed + both | ✅ diff_snap rc=0, run rc=0 |

Announce audit (L80, both directions): `FERNOINIT = ON` / `VISCNOINIT = ON` present in every
knob-ON leg, absent in knob-OFF. ⚠️ Audit note: in THIS gate the announces land in `run.err`,
not `run.log` (the serial gate splits streams; job_m7_ab_env merges them) — grep the right
stream before crying L80. CUDA half (26274911/13/15/16/17) pending.

### 3.1 The dividend survey (min of 2; md5 + announce audit both directions + a100_80 checked per row)

First pair's full audit: row0 md5 `02c8a0d1` ✓ / h14 `18275c68` ✓; h14 log announces the full
blessed set incl. TDMANOINIT, REDISWEEP absent ✓; row0 log has ZERO `[fesom_speed]` lines ✓;
all nodes l50xxx ✓. Later rows: same checks run, only exceptions noted.

SYPD column: at each row's OWN benchmark dt (`SYPD = dt / (365.25 × s/step)` — the Stage-2
convention MINUS the production-dt correction; do NOT compare across rows with different dt,
in particular the dt120 rescue row's SYPD is intrinsically ~2/3 of a dt180 number).

| point | row0 s/step | h14 s/step | **Δ** | SYPD row0→h14 | pre-reg | jobs (row0/h14) | note |
|---|--:|--:|--:|--:|--:|---|---|
| core2@1N | 0.1087 | 0.0754 | **−30.6 %** | 45.3 → **65.4** | −20…−45 (record) | 26274324/26274325 | L95 drill PASS (300 @ dt1800, both legs, no blowup) |
| core2@2N | 0.0922 | 0.0705 | **−23.5 %** | 53.4 → **69.9** | < core2@1N per (a) ✓ | 26274326/26274327 | ordering (a) HOLDS |
| dars@2N (dt180) | **DNF** | **DNF** | — | — | −38 ±8 | 26274328/26274329 | 🔴 **CG NaN abort at step ~10** (`fesom_ssh.cpp:888`), all 4 reps, BOTH binaries ⇒ deterministic configuration failure, NOT an M7 regression. The L95 class — see §3.2 |
| dars@2N (**dt120** rescue) | 0.7771 | 0.3926 | **−49.5 %** — the survey MAX | 0.42 → **0.84** (@dt120!) | −38 ±8 — **MISSED HIGH** (5th) | 26274728/26274729 | ✅ dt120 completes (150 steps, both legs) — the dt180 DNF was the marginal cold-start, not data. Row annotated: its OWN protocol (dt120). 395k verts/rank = the survey's largest per-rank workload ⇒ biggest dividend, exactly where the regime model peaks. dars ordering (a) perfect: −49.5/−44.3/−34.2 |
| farc@2N | 0.1678 | 0.1001 | **−40.3 %** | 2.94 → **4.92** | −25 ±8 — **MISSED HIGH by ~2 bands** | 26274334/26274335 | L95 drill PASS (300 @ dt180, both legs). ⚠️ pattern-breaker candidate vs ordering (c): 80k verts/rank ≈ dars@8N regime but the dividend is NG5-class. Hold verdict until the farc column completes — if the whole column sits high, it is a mesh-level effect (shallow Arctic columns ⇒ small kernel time ⇒ host-class savings weigh more), not noise. |
| dars@4N | 0.4559 | 0.2541 | **−44.3 %** | 1.08 → **1.94** | −30 ±8 — **MISSED HIGH** | 26274330/26274331 | row0 PASS at dt180 (150 steps) — the dist_8 DNF is partition-specific. Second high miss: the Δ pre-registrations look systematically conservative off-NG5 |
| dars@8N | 0.3017 | 0.1985 | **−34.2 %** | 1.63 → **2.48** | −27 ±7 — at the band EDGE (3rd high) | 26274332/26274333 | h14 reproduces session-9 h11 (0.1985 vs 0.1981, +0.2 % cross-day); dars ordering (a) holds: −44.3 (4N) → −34.2 (8N) |
| farc@4N | 0.1260 | 0.0864 | **−31.4 %** | 3.91 → **5.70** | −20 ±8 — **MISSED HIGH** (4th) | 26274336/26274337 | ordering (a) within farc holds: −40.3 (2N) → −31.4 (4N) |
| farc@8N | 0.1087 | 0.0861 | **−20.8 %** | 4.53 → **5.72** | −15 ±10 ✓ (first in-band point) | 26274338/26274339 | Strong-scaling note: row0 4N→8N is only −14 % (20k verts/rank — deep comm plateau) |
| NG5@4N | 1.2299 | 0.6497 | **−47.2 %** | 0.40 → **0.76** | row0 1.21 ±4 ✓, Δ −46.5 ±2 ✓ **BOTH IN BAND** | 26274340/26274341 | h14 leg REPRODUCES the anchor 26271441 (0.6497 vs 0.6495, +0.03 %). Protocol cross-check: 35-step-era row0 1.2796 → clean-300 1.2299 = −3.9 %, inside the modelled 4–7 % contamination correction ✓ |
| NG5@8N | 0.7085 | 0.4025 | **−43.2 %** | 0.70 → **1.22** | row0 0.70 ±4 ✓, Δ −42.5 ±2.5 ✓ **BOTH IN BAND** | 26274342/26274343 | h14 leg reproduces session-9 h11 (0.4025 vs 0.4022, +0.07 %) |

### 3.0b h15 ladder — ✅ **9/9 GREEN**

CUDA half: fidelity iso 26274911 PASS rc=0 (announces ×2 ✓) · fidelity full-blessed 26274913
PASS rc=0 (FERNOINIT+VISCNOINIT+TDMANOINIT ON, REDISWEEP absent ✓) · options TKE 26274915 /
mEVP 26274916 / zstar 26274917 all PASS rc=0 — **zstar standing controls Kv 9.537e-02 AND
Av 9.869e-02 TO THE DIGIT** (rule 0.15). With the Serial 4/4 (§3.0): the full ladder is green;
both levers byte-proven, options-safe, teardown-clean. → commit + A/B.

### 3.0c A/B pre-registration (recorded BEFORE submission)

One same-alloc 3-leg job (job_m7_ab_env, NG5@4N, h15 `ce1e859d`, 300 steps, min of 2,
`-C a100_80`): LEG1 ref `FESOM_SPEED=1` · LEG2 +`FERNOINIT` · LEG3 +`FERNOINIT+VISCNOINIT`.
- **LEG2 vs LEG1 (FERNOINIT): point −0.15 %, floor 0.0, ceiling −0.45** (1.0 GB/step
  coalesced local stores at TDMANOINIT's realized ~60 % byte price + 0.27 GB/step DRAM-priced
  global stores).
- **LEG3 vs LEG1 (both): point −0.2 %, floor 0.0, ceiling −0.6.**
- LEG3 vs LEG2 (VISCNOINIT alone ≈ −0.07 %) is BELOW single-A/B resolution — the promotion
  decision for VISCNOINIT rides the ncu counter (locST on `fesom_impl_vert_visc`, 1 launch/step),
  not the A/B delta.
- Promotion rule 0.13: each lever promotes to the master ONLY if its attributed effect lands
  in range; a wrong-sign result flips it to permanent `_exp` (REDISWEEP precedent).

### 3.0d ✅ A/B 26278160 + ncu pair 26279695/96 — BOTH RANGE HITS ⇒ BOTH PROMOTED

A/B (same-alloc, h15 `ce1e859d` ✓, min of 2): ref 0.6482 · fer **0.6470 (−0.19 %** ∈
[0, −0.45] ✓**)** · both **0.6452 (−0.46 %** ∈ [0, −0.6] ✓**)**. Rep spreads 0.08/0.26/0.33 %.
ncu attribution (steps 11-12, reps agree to 3 digits):
| kernel | dur/launch | locST | locLD | dramW |
|---|--:|--:|--:|--:|
| fer_solve_gamma | 9.75→8.10 ms (**−1.65, −17 %**) | 4.697→3.687 (**−1.010 GB = the predicted 1.0 TO THE DIGIT**) | unchanged ✓ | −1.30 GB |
| impl_vert_visc | 13.66→13.33 ms (**−0.33**) | 8.804→7.947 (**−0.857 GB**, 2× the conservative model) | unchanged ✓ | −0.43 GB |

Kernel-sum −2.0 ms/step ≈ −0.31 %; the A/B's −0.46 % includes downstream relief (L93
direction). FERNOINIT promotes on its range-hit A/B + digit-exact counter; VISCNOINIT on its
pre-registered ncu route (real, right-sign, byte-proven). Pricing-rule datapoint #3: coalesced
local stores again converted at ~full byte price ON THE DELETING KERNEL (1.65 ms ≈ 1.0 GB
at ~1600 GB/s effective − the dramW share), consistent with TDMANOINIT.

## 5. h16 CANDIDATE (h15 + both promotions) — cert pre-registration (BEFORE submission)

CUDA `470ead46` / Serial `23d55df3`. Cert = knob-OFF byte + CUDA fidelity full-blessed
(h14-promotion precedent) + fresh 300-step 4N anchor. Pre-registered: knob-OFF bit-identical;
fidelity PASS rc=0 with FERNOINIT+VISCNOINIT announcing under bare `FESOM_SPEED=1` and
REDISWEEP absent; **anchor 0.6465 ±0.5 % (band [0.6433, 0.6497]; = 0.6495 × (1−0.0046))** ⇒
ratio 4.5785/0.6465 = **7.08×** at NG5@4N if it lands on point.

### ✅ 5.1 h16 CERTIFIED — anchor HIT to 0.03 %

26280025 knob-OFF byte PASS (diff_snap rc=0) · 26280026 fidelity full-blessed PASS rc=0
(FERNOINIT+VISCNOINIT announce under bare `FESOM_SPEED=1` ✓, REDISWEEP absent ✓) ·
**anchor 26280027: 0.6467** (min of 2, spread 0.08 %, pure a100_80, md5 `470ead46` ✓) —
pre-reg 0.6465 **HIT** ⇒ **⭐ RATIO 4.5785 / 0.6467 = 7.08× at NG5@4N.** h16 = CURRENT BEST.
(The C strict-reduction tail is now harvested: TDMANOINIT −0.30 % + FERNOINIT/VISCNOINIT
−0.43 % measured anchor-to-anchor. Remaining C candidates are all sub-0.05 % — the package
winds down as priced; E and 16N remain the frontier.)

**farc column complete — the regime read so far:** ordering (a) is perfect on every complete
column (farc −40.3/−31.4/−20.8; dars −44.3/−34.2; core2 −30.6/−23.5). Ordering (c) at matched
per-rank size holds directionally at the small end (farc@8N 20k/rank −20.8 ≈ core2@2N 16k/rank
−23.5) but farc@2N (80k/rank, −40.3) runs ~6 points hotter than dars@8N (99k/rank, −34.2) —
and EVERY absolute pre-registration off-NG5 was conservative. The systematic story: the
post-Tier-1 stack (H.3/H.7/H.8/H.9 host-class + kernel levers) holds its share on mid-size
per-rank workloads far better than the L84(b)-style retention models assumed — consistent
with session 9's 8N/dars refresh both beating THEIR pre-regs by 0.7 %. The NG5 pairs will
anchor the top end.

### 3.2 dars@2N DNF — diagnosis open

The "Killed" tasks were MPI_Abort collateral, NOT OOM (first hypothesis discarded on reading
the log — the FATAL is `CG_kk: pp·App is -nan`). Both binaries, all reps, elapsed ≈1:07/rep
(≈ mesh load + O(few) steps). dist_8 rpart.out structurally sane (8 × ~395k). Discriminator
in flight: dars@4N (dist_16) — if IT completes, suspect the dist_8 point specifically (never
used by any campaign); if it also dies instantly, the dt180 cold-start is unstable at coarse
decompositions (would need dt120 + annotation, or DNF the point). Waiting before resubmitting.

**RESOLVED (same session): it is the cold-start instability class, partition-marginal.** The
log shows the run HEALTHY through step ~9 (CG 69→72 iters, normal ramp) and NaN at step ~10 —
not a data problem. dars@4N (dist_16) completed at dt180 ✓ and dist_32 did in session 9 ⇒
dt180 from cold PHC is MARGINAL on dars and partition-dependent rounding decides which
decomposition trips (dist_8 does at step ~10; note this INVERTS the L95 finer-dies-earlier
observation, which was NG5-only — one more instance of "an instability boundary is a property
of the configuration"). **Rescue: the pair re-ran at dt120** (annotated protocol — the
dividend Δ% is per-step work and stays comparable): jobs 26274728 (row0) / 26274729 (h14),
150 steps, both `-C a100_80`.
