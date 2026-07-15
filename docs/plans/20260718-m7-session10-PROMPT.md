# M7 next session (session 10) — PROMPT

*Written 2026-07-15, close of session 9. Branch `m7-speed` @ `1f2c8a8` (NOT pushed — ask the
user). Working tree clean. Read in this order: this file →
`docs/plans/20260717-m7-session9-FINDINGS.md` (the C.1 falsification §3.3-3.4, the pricing rule
§3.7-3.8, the census §2.1c) → `docs/KOKKOS_PORTING_LESSONS.md` **L97**.*

---

## 0. THE RULES (deltas on top of session-9 PROMPT §0 — those all still stand)

- **0.11 🔴 L97 — a spill pool RANKS, it never PRICES.** C.1 was byte-perfect (STACK 0, byte
  proof rc=0, 8/8 gates) and measured **+1.88 %**. Structural rewrites (fusion, anchor
  direction, phase count) are CACHE-LOCALITY changes; byte deltas that ignore reuse distance
  are models, and the model had the wrong sign. Gates certify VALUES, never SPEED.
- **0.12 The C pricing rule, measured from both sides:** REUSED local traffic ≈ free
  (REDISWEEP +1.88 %); ONCE-PER-STEP local writeback ≈ full DRAM price (TDMANOINIT −0.30 %,
  ncu cross-check to the digit). **Package C continues ONLY as strict reductions** — delete
  stores/arrays while keeping loop structure — each behind `_exp` until its A/B, then promote.
- **0.13 New-lever lifecycle (learned the hard way): build behind `fesom_speed_on_exp`,
  promote to `fesom_speed_on` ONLY after the A/B hits its range.** REDISWEEP rode the master
  into its own regression window; nothing shipped because the A/B caught it same-session.
- **0.14 The 0.10a teardown wart is DEAD** (smooth.work finalize hook, `9bd0aba`). Full-blessed
  runs exit rc=0 now — job states are trustworthy again. If you see rc=134 at teardown, it is
  a NEW bug, not the known wart. (`job_m7_hostprof` additionally runs
  `srun --kill-on-bad-exit=0` so a rank abort can never again cost a whole nsys trace.)
- **0.15 The zstar standing controls are TWO numbers: Kv 9.537e-02 AND Av 9.869e-02** (both to
  the digit on h12 AND h13 ladders; session-8's "Kv 9.869e-02" had mislabeled Av).

**Binaries** `m7/bin/…`: 🔴 `h3` broken, never use. `h11` = CUDA `d74d31b4` (climate-certified
at 1 yr, 26267151). `h12` = C.1 candidate that FAILED its A/B — study only. `h13` = C.2a probe
rung. ✅ **`h14` = CURRENT BEST, CERTIFIED**: CUDA **`18275c68`** / Serial **`27f2eb7d`** —
h13 + TDMANOINIT promoted; knob-OFF byte + full-blessed fidelity (rc=0, announce audit) +
anchor 26271441 **0.6495** (pre-reg 0.6483 ±0.5 % HIT). PROVENANCE.txt in every dir.

---

## 1. WHERE THE CAMPAIGN IS

**⭐ NG5@4N ratio = 7.05×** (4.5785 / 0.6495, matched 300-step pinned pair, pure a100_80).
Stage-1 met (41 % margin). Stage-2 16N SYPD: h11 16N leg **[26267149 — still queued at
session-9 close; harvest it FIRST, pre-reg 0.2650 −1.4 ±0.5 % ⇒ SYPD@dt240 ≈ 2.40]**.
**The 8× stretch = 0.6495 → 0.5723 = another −11.9 %.** The measured pools:

| pool | size at 4N | 16N behaviour | status |
|---|--:|---|---|
| **E: halo MPI-wait** | **18.0 ms (2.8 %)** | GROWS (comm-bound side) | **THE LEADING PACKAGE** — all provably MPI-wait (gap300_h11) |
| C strict reductions | ~1-3 ms each | kernel-class ~56 % | fer_solve_gamma TDMA-shape audit next; impl_vert_visc init small |
| H.10 ice-thermo bounce | 4.9 ms (0.75 %) | host-class ~hold | the low-risk filler (census §2.1c: 2.9 PCIe + host residual) |
| B (FCT2) | parked (f=0.242) | — | unchanged |
| C rewrites | **CLOSED at 4N** | — | L97 — the pool was never a reservoir |

Honest arithmetic: E+C+H.10 ≈ 25-28 ms ≈ −4 % even at FULL conversion — **the −11.9 % stretch
is NOT reachable from the measured 4N pools alone.** Say so if asked; do not manufacture a
path. The kernel-busy 543 ms (83.6 % of the h14 step) is the wall: further 4N gains mean
making the BUSY kernels faster (roofline work: launch bounds, occupancy on the 80-reg pair,
fusion that RESPECTS locality), each priced per L97 before building.

## 2. SESSION-10 SHAPE — **USER-APPROVED (2026-07-15): ~7× IS THE 4N ENDPOINT; the effort
moves to 16N.** While the long 16N jobs queue, run the cross-mesh dividend survey (§2.1).

1. **Harvest 26267149 (16N)** if it ran; else resubmit (`job_m7_ab_env` single-leg pattern,
   BIN=h11cuda or h14cuda — note which; h14 differs from h11 by −0.30 % 4N-kernel-class).
   Refresh Stage-2 SYPD; then re-size E at 16N (its pool grows exactly where the ratio decays).
   Also fire the **gap300_16N census on h14** (background, `-C a100_80`, the fixed
   job_m7_hostprof) — E's design input. Expect BOTH to sit long in queue: 16 a100_80 nodes.

### 2.1 🔴 USER-REQUESTED: the cross-mesh dividend survey (the while-we-wait workstream)

**Question: what did the whole M7 stack actually buy on every mesh × node count we track —
not just NG5@4N?** Nothing outside NG5@{4,8,16}N + dars@8N has ever been measured on a
post-Tier-1 binary.

- **Design: per point, a same-day pinned GPU pair** — leg A `m7/bin/row0/fesom_port_cuda`
  (`02c8a0d1`, campaign start, run KNOBLESS — it predates most knobs), leg B
  `m7/bin/h14/fesom_port_cuda` (`18275c68`, `FESOM_SPEED=1`). Two single-leg `job_m7_ab_env`
  jobs per point, both `-C a100_80`, min of 2, walltime sized (rule 0.7). Same-day pairing
  makes the DELTA immune to cluster drift (L94/same-day rule); it is not a same-alloc A/B, so
  quote each point as (row0, h14, Δ%) with both job ids.
- **The matrix** (4 GPUs/node; partitions verified to exist 2026-07-15):
  | mesh | node points | protocol |
  |---|---|---|
  | core2 (**/pool** — perf runs; only the CORE2 *gates* must use the private mesh, L73) | 1N (dist_4), 2N (dist_8) | ⚠️ no validated M7 long-window protocol — validate per L95 (start 300-step @ dt1800; watch for blowup; a matched-window pair is still honest if shorter) |
  | dars | 2N, 4N, 8N (dist_8/16/32) | **150-step** (L95), dt180 |
  | farc | 2N, 4N, 8N (dist_8/16/32) | ⚠️ no validated M7 protocol — same L95 drill, dt180 |
  | NG5 | 4N, 8N (dist_16/32) | 300-step, dt180 (4N/8N h14 legs also complete the h14 ledger row set) |
  16N points for dars/farc (dist_64 exists) are OPTIONAL — only if the queue is kind; the NG5
  16N pair is the priority 16-node ask.
- **Read the result by REGIME, not just as one number**: host-class levers pay by
  host-work-per-step (≈ mesh-size-independent), kernel-class by kernel time ⇒ the dividend %
  should GROW on smaller per-rank workloads at fixed node count and SHRINK toward comm-bound
  points. A point that breaks that pattern is a finding (L84 precedent), not noise — chase it.
- **CPU/SYPD context**: reuse the existing protocol-matched CPU rows (NG5/dars session-7,
  core2 M5.24) with an explicit cross-day caveat; fire fresh Serial legs ONLY where a headline
  ratio is wanted (CPU legs at scale are expensive). The GPU-vs-GPU dividend needs no CPU leg.
- **Deliverable**: a "M7 dividend by mesh × nodes" table in `docs/GPU_SPEED_M7.md` (row0 →
  h14 s/step, Δ%, regime note), plus the ledger rows for every fresh h14 leg.

2. **E, opened properly** (once the 16N census lands): the 18 ms is MPI-WAIT, not bandwidth —
   the levers are overlap (async halo + compute between post and wait), message coalescing
   (device2/deviceN pairs), and rank-order/topology. Start from the gap300_h11 pair rows
   (§2.1c) + `m7_gap_census.py --diff` between the 4N and 16N censuses.
3. C strict-reduction tail as filler between E measurements: audit fer_solve_gamma's 7-array
   TDMA (same shape as impl_ale: fold gather/coeffs bottom-up? NO — L97 — only DELETIONS;
   check its init pattern + M5.24-style aliasing candidates), each behind `_exp` → A/B → promote.
4. H.10 if a low-risk win is wanted (ice-thermo bounce class, 4.9 ms, PCIe+host mix — same
   SSHRAILS recipe: device halos + gated rails).

## 3. STANDING MACHINERY (unchanged unless noted)

`scripts/m7_gap_census.py` (+`--diff`) · `m7_spill_pool.py` / `m7_kernel_busy.py` (L96-proof) ·
`job_ncu_fctgm_ng5` (BIN/KNOBS/TAG/NCU_METRICS/NCU_REGEX/NCU_SKIP/NCU_COUNT — 🔴 recompute
SKIP/COUNT from the census launch counts for ANY new regex; the 7-family regex = 71
launches/step, session-9 nearly lost the capture to the stale default 650) ·
`job_m7_hostprof` (now `--kill-on-bad-exit=0`) · `job_m7_ab_env` (single-leg = the std300
pattern; A/B = two legs) · `jobs/m7_provenance.sh` (check md5 FIRST on every harvest).

## 4. USER PREFERENCES (standing)

- **"Always measure, do not guess."** Pre-register BEFORE submission; announce-line check (L80)
  BOTH directions (fires when on, absent when off); SHA.txt md5 on every harvest.
- **Ask before pushing or tagging.** Session-9 commits (`ad7d6c9`…`1f2c8a8`) are LOCAL-ONLY.
- Mixed precision BANNED. SLURM for big runs; output under `/work/ab0995/a270088/port2/m7/`.
- CORE2 gates: private mesh (L73). dars: 150-step protocol (L95).
- When two of your own numbers disagree, resolve it in the open (the Kv/Av case is the model).
