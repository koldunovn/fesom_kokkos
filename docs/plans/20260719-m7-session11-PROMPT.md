# M7 next session (session 11) — PROMPT

*Written 2026-07-15, close of session 10. Branch `m7-speed` @ `ab14c1b`, **PUSHED** (everything
through session 10 is on origin). Working tree clean; nothing in flight. Read in this order:
this file → `docs/plans/20260718-m7-session10-FINDINGS.md` **§6** (the E repricing — the
session's load-bearing finding) → `docs/KOKKOS_PORTING_LESSONS.md` **L98, L99**.*

---

## 0. THE RULES (deltas on top of session-10 PROMPT §0 — those all still stand)

- **0.16 🔴 L98 — a threshold is part of the measurement.** The census setting of record for
  ALL package-E work is **`--min-gap-ms 0.1`**. Quote thresholds with pool numbers. The tell
  that a threshold is lying: a pool that shrinks while its share of idle explodes.
- **0.17 🔴 The 8× question is OPEN again and it is the USER'S, not yours.** "~7× is the 4N
  endpoint" was decided on the 18 ms E pool; the honest pool is 97 ms/step (14.8 %) at 4N
  (L98). This was surfaced to the user at session-10 close (they said "push. what are the
  next plans?" and got the finding in the same reply — no verdict yet). **Do not treat 8× as
  a target until the user says so; do not bury that it is now plausibly reachable either.**
  Both roads run through the same E work, so nothing blocks on the answer.
- **0.18 L99 — instability boundaries are partition-marginal.** dars dt180-from-cold: dist_8
  dies at step ~10, dist_16/32 complete — the COARSEST died (inverts the NG5 pattern). Ask
  "does it complete" per mesh × partition × dt. And: read the log before fixing the wrong
  failure (the "Killed" tasks were MPI_Abort collateral, not OOM).
- **0.19 Audit trap: `job_m7_gate_serial` splits streams — the `[fesom_speed]` announces are
  in `run.err`, not `run.log`** (job_m7_ab_env merges). Grep the right stream before crying L80.
- **0.20 The ncu promotion route is precedent.** A strict-reduction lever whose expected A/B
  is below single-A/B resolution can pre-register promotion on the per-kernel ncu counter
  instead (VISCNOINIT: locST −0.857 GB, dur −0.33 ms/launch — promoted). Pre-register the
  route BEFORE the measurement, as always.

**Binaries** `m7/bin/…`: 🔴 `h3` broken, never use. `h11` = climate-certified (1 yr). `h14` =
CUDA `18275c68` / Serial `27f2eb7d` (the survey's blessed leg). ✅ **`h16` = CURRENT BEST,
CERTIFIED**: CUDA **`470ead46`** / Serial **`23d55df3`** — h14 + FERNOINIT + VISCNOINIT riding
the master; ladder 9/9, knob-OFF byte, fidelity full, anchor **0.6467** (pre-reg 0.6465 HIT to
0.03 %). PROVENANCE.txt in every dir. *(h16 has NOT had its own 1-yr climate gate; its diff vs
h14 is entirely byte-proven strict deletions, so this is belt-and-suspenders — fire it as a
background leg if a slot is free, pre-reg: the M5.23 bar to the digit, h11 precedent.)*

---

## 1. WHERE THE CAMPAIGN IS

**⭐ NG5@4N ratio = 7.08×** (4.5785 / 0.6467, h16, matched 300-step pinned pair, pure a100_80).
**⭐ Stage-2 16N SYPD@dt240 = 2.43** (0.657/0.2629/1.03; 16N ratio 4.67×). **The cross-mesh
dividend survey is COMPLETE (11/11)** — table in `GPU_SPEED_M7.md`; dividends −20.8 %…−49.5 %,
monotone in per-rank workload on every mesh, all NG5 points in band, every off-NG5 prior beaten.
**Package C is CLOSED** (TDMANOINIT −0.30 % + FERNOINIT/VISCNOINIT −0.43 % anchor-to-anchor;
momentum_adv parked with cause; everything left audits < 0.05 %).

**THE FRONTIER IS PACKAGE E, and it was repriced 5× by the 16N census (findings §6 / L98):**

| @0.1 ms (the setting of record) | 4N (h11 census) | 16N (h14 census) |
|---|--:|--:|
| step / kernel-busy | 656.8 / 543.0 (82.7 %) | 277.3 / 137.8 (49.7 %) |
| **halo-wait pool** | **97.3 ms = 14.8 %** | **124.8 ms = 45.0 %** |
| — MPI-covered | 88.8 | 116.3 |
| — staging PCIe (~70 KB/copy) | 16.5 | 16.5 |
| halo gap events/step | **358** (216 device + 132 device2 + 4-6 deviceN) | **358** — identical, structural |
| per-event wait (device leg) | ~278 µs | ~386 µs |

Kernels scale ÷3.94; the 358 events' ~300 µs latency does not ⇒ **this IS the 7.08×→4.67×
decay, end to end.** Both censuses' sqlites are on disk: `gap300_h11/hostprof.sqlite` (4N),
`gap300_16n_h14/hostprof.sqlite` (16N).

## 2. SESSION-11 SHAPE — OPEN PACKAGE E ON THE MEASURED NUMBERS

### 2.1 E.0 — the code audit BEFORE any lever (foreground, no GPU needed)

Map the 358 events/step to source. `src/fesom_halo_device.cpp/.hpp` + call sites of
`fesom_halo_exchange_device*`. Questions the census cannot answer:
- How many exchange CALLS per step, over which fields, at which cadences? (216 device-class
  events/step — how many are per-tracer? per-CG-iteration? The CG runs ~70 iters/step and
  `ssh_solve_cg` shows only 0.2 ms of >0.1 ms gaps — so the CG's exchanges are apparently NOT
  in this pool; verify.)
- Per call: how many MPI messages (neighbors × fields), what sizes, and is the payload staged
  host-side (the 16.5 ms PCIe says yes for at least part)? Which path is CUDA-aware-direct
  vs host-staged? (Memory: `reference-cuda-aware-mpi` — device halo needs openmpi/4.1.5-nvhpc,
  which env_cuda.sh loads; a staged path may be a deliberate M5.17-era choice — read the
  history before "fixing" it. M5.17 is marked DEAD END in memory: find out what exactly died.)
- Which exchanges are fusable (same neighbor set, same cadence, adjacent in the step) —
  the event-count/coalescing lever's shopping list, with byte sizes.
- What compute sits BETWEEN a pack and its wait today — the overlap lever's headroom, per
  exchange site.

**Deliverable: a table (exchange site → fields, msgs/step, bytes, staged?, fusable-with,
overlap headroom) in the session-11 findings — the E ledger everything else prices against.**

### 2.2 E lever candidates, in evidence order (each: pre-register → `_exp` knob → ladder → A/B)

1. **E.1 COALESCE / event-count reduction** — 358 latency-bound events at ~70 KB staged is the
   textbook small-message halo wall. Fuse per-field exchanges sharing a neighbor set + cadence
   into per-group exchanges (fewer, bigger messages). Data movement identical ⇒ values
   byte-identical ⇒ FORCE_SERIAL byte proof applies ONLY if the pack/unpack kernel source is
   backend-identical AND the single-rank path exercises it (it may not — L86-adjacent; the
   real gates are knob-OFF byte, multi-rank CUDA fidelity, options ×3 per L91: comm structure
   = ownership-adjacent).
2. **E.2 OVERLAP** — post exchanges early, run independent kernels between post and wait
   (the E.0 headroom column says where). This changes SCHEDULING only, not values.
3. **E.3 GPU-DIRECT** — kill the measured 16.5 ms/step of staging PCIe where the transport
   allows it (UCX + openmpi/4.1.5-nvhpc; A.3's env A/B machinery `job_m7_ab_env` LEG1..LEG4
   is built for exactly this class of experiment — some of E.3 may be pure env, no code).
4. **E.4 per-event latency** — rendezvous scheme (`UCX_RNDV_SCHEME`), rank order/topology.
   Env-class first (free to A/B), code-class only if the env A/B shows headroom.

**Sizing discipline (L93/L98): the pool is 97/125 ms but conversion factors for LATENCY pools
are unknown at this site — pre-register floors of zero and let the first A/B calibrate the
class.** A/Bs at BOTH 4N and 16N for every E lever (the pool's share differs 3×; a lever can
be null at 4N and decisive at 16N — the 16N A/B is the one Stage-2 cares about; 0.7-per-rank
proxy does NOT hold for CG/latency effects, L: per-rank-proxy).

### 2.3 Background / filler (fire early, harvest whenever)

- h16 1-yr climate gate (optional, §0 binaries note) — 4N, `job_m7_tier1_cuda_1yr` pattern,
  pre-reg the M5.23 bar.
- H.10 ice-thermo bounce (4.9 ms at 4N, 1.2 ms of >0.1 gaps at 16N — NOT the frontier;
  only if E stalls in queue).
- OPTIONAL survey extension only if the user asks: dars/farc@16N (dist_64 exists).

## 3. STANDING MACHINERY (unchanged unless noted)

`scripts/m7_gap_census.py` — **always `--min-gap-ms 0.1` for E work (L98)** ·
`m7_spill_pool.py` / `m7_kernel_busy.py` · `job_m7_ab_env` (single-leg std300; A/B up to 4
legs — the env-class E levers use LEG1..LEG4 directly) · `job_m7_hostprof`
(`--kill-on-bad-exit=0`; census mode: `NSYS_TRACE=cuda,mpi NSYS_SAMPLE=none NSTEPS=300`) ·
`job_ncu_fctgm_ng5` (recompute SKIP/COUNT per regex from census launch counts) ·
`jobs/m7_provenance.sh` (md5 FIRST on every harvest) · gates: `job_m7_gate_serial`
(announces in run.err! 0.19) + `job_m7_gpu_gate` (options via KNOBS + SREF).

## 4. USER PREFERENCES (standing)

- **"Always measure, do not guess."** Pre-register BEFORE submission; announce audit both
  directions (right stream, 0.19); SHA/md5 on every harvest; same-day anchors; `-C a100_80`
  on absolutes; A/Bs same-alloc.
- **Ask before pushing or tagging** (session-10 commits are pushed through `ab14c1b`).
- Mixed precision BANNED. SLURM for big runs; output under `/work/ab0995/a270088/port2/m7/`.
- CORE2 gates: private mesh (L73). dars: 150-step dt180 at ≥dist_16; dist_8 needs dt120 (L99).
- The 8× question is the user's call (0.17). When two of your numbers disagree, resolve it in
  the open.
