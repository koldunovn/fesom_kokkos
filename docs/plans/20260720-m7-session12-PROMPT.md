# M7 next session (session 12) — PROMPT

*Written 2026-07-16, close of session 11 (Fable). Branch `m7-speed`, all session-11 work
committed LOCALLY (⚠️ sessions 9-11 are NOT pushed — ask the user for push approval EARLY).
Read in this order: this file → `docs/plans/20260719-m7-session11-FINDINGS.md` **§6 (results)
+ §2 (the E ledger) + §5 (the CGPIPE ladder record)** → `docs/KOKKOS_PORTING_LESSONS.md` L100.*

---

## 0. THE RULES (deltas on the session-11 PROMPT §0; everything there still stands)

- **0.21 🔴🔴 8× IS THE TARGET — the user said so explicitly (2026-07-16: "the 8x is still a
  target, yes").** Rule 0.17 is RESOLVED. At 4N that is 0.6381 → 0.5723 s/step = another
  −10.3 %. The shopping list is MEASURED (§1); no lever on it is speculative.
- **0.22 🔴 L100 — a pair-keyed census collapses same-tag chains.** All halo pack/unpack
  kernels demangle to their enclosing exchange function, so per-site attribution needs the
  TIMELINE walker `scripts/m7_halo_sites.py` (bracketing compute kernels name the site), never
  the pair histogram. This is how "the CG is not in the pool" (session 10) went wrong.
- **0.23 🔴 THE STALE-FREEZE TRAP: rebuild BOTH backends after the LAST source edit, THEN
  freeze.** Session 11 froze a CUDA binary built before the final fix; 3 CUDA gates crashed
  with a DELETED check's message text while Serial passed everything. When a frozen pair
  splits Serial-pass/CUDA-fail, suspect the freeze before the code.
- **0.24 CGPIPE IS ADOPTED (user decision 2026-07-16): it rides `FESOM_SPEED=1` as of h17.**
  E.EVP1 wide-halo (the user's approved option 2) stays OPT-IN when built — it changes
  rounding (climate-close class), and the user's default-safer rule applies to it.
- **0.25 Instrument projections differ — do not "cross-check" numbers from different
  projections.** Census "MPI-covered" includes Allreduces inside GPU-idle gaps; the
  `[halo-mpi-prof]` Waitall timer counts halo Waitalls only but INCLUDING GPU-busy time
  (63/72 ms vs 89/116 ms — both correct). The census halo pool (97.3 / 124.8 ms @0.1 ms) is
  the sizing of record.
- **0.26 Watcher hygiene: `squeue` can transiently return EMPTY for live jobs.** Require ≥3
  consecutive empty polls + an `sacct` state confirm before declaring jobs done (a watcher
  false-fired on this in session 11).

**Binaries** `m7/bin/…`: 🔴 `h3` broken, never use. ✅ **`h17` = CURRENT BEST PENDING CERT
HARVEST** (h16 + CGPIPE adopted into the master): CUDA **`f8384e86`** / Serial `5c3c90fc`,
PROVENANCE.txt in the dir. `h16` = certified fallback (CUDA `470ead46` / Serial `23d55df3`,
7.08×). `cgpipe0` = h16 + the lever as opt-in (CUDA `ef86c3c9` — the A/B binary of record).

---

## 1. WHERE THE CAMPAIGN IS

**⭐ CGPIPE (E.CG1) is built, fully certified, measured, and ADOPTED.** Single-exchange 2-ring
PCG (`fesom_ssh.cpp`; exchanges/iter 2→1, CG events 146→74/step). Certification: FORCE_SERIAL
byte proof — **knob-ON is BIT-IDENTICAL to the certified baseline** (CORE2 np8 ice,
diff_snap rc=0) — plus 2×2627 in-vivo selfchecks all exactly 0.000e+00 (Serial AND CUDA),
gpu fidelity ON/OFF, options ×3, CG iters identical ON-vs-OFF. **A/B (same-alloc, 300 steps):
4N 0.6472→0.6381 = −1.41 % (pre-reg central HIT); 16N 0.2626→0.2414 = −8.07 % (pre-reg
ceiling BEATEN — 5th consecutive wrong-low, L93 entanglement).**

**🔴 HARVEST FIRST — the h17 cert set (submitted 2026-07-16 ~10:00):**
| job | what | expect |
|---|---|---|
| 26299411 | knob-OFF byte gate (serial) | diff_snap rc=0 |
| 26299412 | gpu fidelity, bare `FESOM_SPEED=1`, BIN=h17 | PASS **and** `[fesom_speed] FESOM_SPEED_CGPIPE = ON` in the log — the adoption L80 check: the master must now FIRE the lever |
| 26299413 | 4N anchor (ab_env single-leg, h17, std300) | **pre-reg 0.6381** ⇒ ⭐ ratio 4.5785/0.6381 = **7.18×** |
| 26299414 | 16N anchor | **pre-reg 0.2414** ⇒ 16N ratio ≈ **5.09×**, Stage-2 SYPD@dt240 ≈ **2.65** |

On harvest: md5 `f8384e86` FIRST; anchors outside ±0.8 % of pre-reg → L94 checklist (node
mix in `scontrol show job`/SHA.txt) then re-run — NEVER adjust. Then write the h17 row into
`docs/GPU_SPEED_M7.md` and update MEMORY.

**The measured 8× shopping list (4N ms/step; 16N in parens):**
| pool | size | lever | class |
|---|--:|---|---|
| EVP subcycle | 25.3 (31.6) | **E.EVP1 wide-halo — THIS SESSION's main lever** | climate-close, OPT-IN |
| rank imbalance | 36.1 (53.0) | Lever-D partition/work balance — measure per-rank work FIRST | deployment-side |
| CG residual (74 ev) | ~18 (~24) | E.4 transport (comm share proven 58 %/49 %) | env-first |
| E.1 fuses | 7-8 (12-13) ceiling | KPP 15→5, ice FCT 21→11, T+S post-trdiff 2→1 | byte-identical, FORCE_SERIAL-provable |
| 3D-site transport | part of 39 (48) | E.3/E.4 (staging = UCX pinned bounce, 181 MB/step) | env-first |

**E.split verdict (h16-pinned, findings §6): imbalance 42 % / comm 58 % at 4N; 51 % / 49 % at
16N.** The comm branch of the pre-registered decision rule fired at BOTH scales ⇒ transport
(E.3/E.4) and overlap-class levers are LIVE; **M5.17's 79 %-imbalance dead-end verdict is
STALE** (it was measured on a 4×-slower regime). Barrier-leg overhead +0.37 %/+2.10 % (sane).
Transport note: 4N runs `tag(rc_mlx5)`, **16N runs `tag(dc_mlx5)`** — UCX flips to
dynamic-connect at 64 ranks (different latency profile; an E.4 lead).

## 2. SESSION-12 SHAPE

### 2.1 Harvest h17 (above), ledger + memory rows. Ask about pushing sessions 9-11.

### 2.2 E.EVP0 — the wide-halo EVP code audit BEFORE any code (the E.0 discipline)

`src/fesom_ice_evp.cpp` (K1/K2/K3 kernels, the subcycle loop ~:600-717). Questions:
- Per subcycle, which arrays are READ on the halo and WRITTEN owned? (uice/vice exchanged;
  eps/sigma live on ELEMENTS; u_rhs/v_rhs are element→node atomic scatters; inv_am, rhs_a/m,
  coastal masks static.) What must be valid on a k-deep ghost ring for one subcycle's element
  pass + node update to be computable k more times without exchange?
- Element rings: elements touching ring-k nodes — the partition's eXDim has UNMAPPABLE vertex
  refs (mesh log: "4740 halo-element vertex refs unmappable") ⇒ element-ring data (areas,
  gradients, node ids) must be SHIPPED at setup like CGPIPE's pr-rows. The CGPIPE builder
  (`cgpipe_build`, fesom_ssh.cpp) is the reusable pattern: want-lists via the com graph
  (NEVER partit->part ranges — L: session-11 bug), Alltoall handshake, verbatim shipping.
- mEVP (`fesom_ice_maevp.cpp:322`) has its OWN subcycle exchange — pre-register whether
  EVPWIDE no-ops under mEVP (recommended: std-EVP only first) and gate it in the options leg.
- **Deliverable: the EVP data-flow table + the K-ring state contract, then the pre-reg.**

### 2.3 E.EVP1 — `FESOM_SPEED_EVPWIDE=K` (value knob, 0=off, OPT-IN — never rides the master)

- Mechanism: exchange (uice,vice) every K subcycles on a K-ring; redundantly compute the
  ghost zone, shrinking by 1 ring per subcycle. Events 120 → 120/K.
- Sizing (L93 floors): K=4 ⇒ −~19 (4N) / −~24 ms (16N) CEILING from the event count; floor 0
  (redundant-compute cost + bigger messages eat some; ghost fraction ~2-5 % of rows at 4N).
  **Pre-register a K sweep {2,4,8} at 16N first** (the bigger pool), then 4N.
- Gates: knob-OFF byte; CUDA fidelity vs Serial baseline (climate-close — NO byte proof
  available: the element→node scatter is atomic, redundant ghost compute cannot be bitwise);
  options ×3 (mEVP leg per 2.2); ice-focused fidelity attention (uice floor, L79/L75);
  A/B at both scales. It stays OPT-IN (0.24) — a 1-yr climate leg before any promotion talk.

### 2.4 Parallel cheap tracks (fire early, harvest whenever)
- **E.1a T+S post-trdiff fuse** (`fesom_step.cpp:1249/1256` → field2): ~30-minute change,
  byte-identical class, FORCE_SERIAL-provable, −0.55/−0.91 ms. The warm-up lever.
- **E.1b KPP per-base_off fieldN** (15→5) + **E.1c ice FCT fuses** (21→11): needs a
  `fesom_halo_fieldN` variant taking per-field `base_off` (blmc/diffK slabs differ only in
  offset). Byte-identical class. Ceiling −7-8/−12-13 ms combined.
- **E.4 env legs** (ab_env, both scales): `UCX_RNDV_FRAG_MEM_TYPE=cuda` at 16N (the A.3
  survivor, −1.15 % @4N, never tested where the pool is 3×); `UCX_RNDV_THRESH` sweep; a
  gdr_copy availability probe (`UCX_TLS` diag first); rc-vs-dc at 64 ranks.
- **Imbalance recon** (36/53 ms pool): per-rank work dump (2D verts, ice-active fraction, CG
  row count, measured kernel-busy per rank from any census sqlite by filtering PID/device) —
  identify WHAT makes the slow rank slow before proposing partition work.

## 3. STANDING MACHINERY

`scripts/m7_halo_sites.py` (site attribution — NEW, L100) · `scripts/m7_gap_census.py`
(`--min-gap-ms 0.1` for E work, L98) · E.split instrument: `FESOM_HALO_MPI_PROF=1` /
`FESOM_HALO_BARRIER=1` (+`FESOM_SPEED_SYNCSTATS=1` counters) — reports at loop end, merged
streams in ab_env logs · `job_m7_ab_env` (LEG1..4 arbitrary env; single-leg = anchor; also
carries KNOB legs) · `job_m7_gpu_gate` (KNOBS+SREF+BIN) · `job_m7_gate_serial` (doubles as
the FORCE_SERIAL byte proof via KNOBS; ⚠️ announces in run.err, 0.19) ·
`jobs/m7_provenance.sh` (md5 FIRST) · CGPIPE debug: `FESOM_CGPIPE_SELFCHECK=1` (per-iter
recurred-vs-exchanged pp; must print 0.000e+00), `FESOM_SPEED_CGPIPE=0` per-lever off.

## 4. USER PREFERENCES / DECISIONS (standing + session-11 additions)

- **8× is the target** (0.21). **CGPIPE adopted** (0.24); **EVP-wide stays opt-in**; further
  promotions are the user's call with numbers in hand.
- "Always measure, do not guess" — pre-register BEFORE submission; announce audits in the
  right stream; SHA/md5 on every harvest; same-day anchors; `-C a100_80` on absolutes
  (A/Bs are node-mix immune, but E.split-style IMBALANCE measurements are NOT — keep them
  pinned, session-11 note); A/Bs same-alloc.
- **Ask before pushing or tagging** — ⚠️ sessions 9-11 are LOCAL (through the session-11 close
  commit); raise it at session start.
- Mixed precision BANNED. SLURM for big runs; output under `/work/ab0995/a270088/port2/m7/`.
- CORE2 gates: private mesh (L73). dars: dt180 needs ≥dist_16; dist_8 needs dt120 (L99).
- When two of your numbers disagree, resolve it in the open (0.25 is the worked example).
