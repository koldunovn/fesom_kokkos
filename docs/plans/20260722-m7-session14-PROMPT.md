# M7 next session (session 14) — PROMPT

*Written 2026-07-17 mid-session 13/14 (Fable), at the user's request (context half full).
Branch `m7-speed`. Read in this order: this file → session-13 findings
(`docs/plans/20260721-m7-session13-FINDINGS.md`, §§8-11 are the tail) → the ledger E.CG2
section (`docs/GPU_SPEED_M7.md`) → the literature survey
(`docs/plans/20260717-m7-LITERATURE-gpu-optimization.md`, its §0 ranking + §9 do-not-chase).*

---

## 0. THE RULES (deltas on the session-13 PROMPT §0; everything there still stands)

- **0.31 🔴 A site's marginal cost per deleted event DECAYS as the site shrinks — never
  price a second-round deletion at the first round's marginal.** CGPIPE's measured
  295 µs/event @16N came from deleting the FIRST 72 events; CGPOLY's pre-reg priced its
  deletions at that marginal and came in at HALF the central (the 2nd wrong-high). Even the
  zero-byte-growth d1 leg underdelivered ⇒ the miss was pricing, not rendezvous.
- **0.32 🔴 MESH RULE (user directive 2026-07-17, memory `feedback-mesh-copies-never-pool`):
  any repartitioned/modified mesh is a COPY under /work — never create, modify, or write
  ANYTHING for meshes on /pool.** And NEVER regenerate the private CORE2 `dist_8` — every
  FORCE_SERIAL byte baseline is tied to that exact decomposition.
- **0.33 THE OPTIONS MATRIX STAYS STRICT (user 2026-07-17): no solver-class amendment to
  the PASS criterion ⇒ any lever that breaks bit-identity is a PERMANENT MANUAL KNOB —
  it never joins `FESOM_SPEED=1`.** CGPOLY is the precedent: its options-mEVP row stands
  as a formal FAIL even though the lever was exonerated (see 0.34). The L79 bit-exact
  zstar-Kv control (9.537e-02) holds for byte-class levers only; solver-class legs match
  the MAGNITUDE, not the bits (measured: 9.999e-02).
- **0.34 The mEVP threshold-flip floor (session-13 §8):** under ANY solver-class lever,
  mEVP flips freezing/melting branches at a few dozen bistable ice-edge cells (43 cells,
  T ≤ 6.6e-02, 40/43 polar, NON-accumulating 0.42→0.066 K step 10→20, flip set changes
  with the trajectory). Reproduced bit-for-bit on pure Serial ⇒ deterministic physics, not
  staleness. `jobs/job_m7_cgpoly_mevp_probe` is the discriminator template.
- **0.35 Ring sizes GROW with ring number** (~linearly, 2D perimeter): the session-13 byte
  pre-estimate assumed constant rings and undershot ×2 (worst-partner R=4: 81.1 KB @4N,
  41.8 KB @16N — printed by the `[cgpoly]` announce). Fold into every 0.27 check.
- **0.36 🔴 NEVER build a conclusion on a parsed file format without cross-checking the
  parse against the producer's own accounting of the same quantity.** The session-13 §6
  recon read `rpart.out`'s tail as rank-major gid lists; it is a **gid → new-index
  permutation** — the misread manufactured a phantom "22 %/51 % 3D imbalance" and a whole
  lever (E.PART) on top of it. `check_partitioning`'s own numbers (one grep away) refuted
  it to the digit. `scripts/m7_part_spread.py` now implements the VERIFIED format.

**Binaries** `m7/bin/…`: ✅ **`h17` = the adopted master** (CUDA `f8384e86` / Serial
`5c3c90fc`; 4N 0.6382 ⇒ 7.17×, 16N 0.2413 ⇒ 5.09×, SYPD 2.65). **`cgpoly0`** = CGPOLY
lever (CUDA **`ee2c4fdd`** / Serial `87392308`; knob-off ≡ h17 — off-legs reproduced the
anchors). `evpw0` v3 = EVPWIDE lever (`9c900b4f`/`21cea692`). 🔴 `h3` broken, never use.

## 1. WHERE THE CAMPAIGN IS

- **E.CG2 CGPOLY: DONE end to end in session 13** (audit → pre-reg → build → full ladder →
  A/Bs → user review). `FESOM_SPEED_CGPOLY=d` (+`FESOM_CGPOLY_KAPPA`, default 30):
  degree-d Chebyshev preconditioner on an R=(d+1)-ring single-exchange PCG, frozen-Ã
  ship-once, iters 72→23 at d3 (settled, both scales). **d\*=3: 4N 0.6379→0.6213 (−2.60 %)
  ⇒ 7.37×; 16N 0.2417→0.2314 (−4.26 %) ⇒ 5.31×, SYPD@dt240 ≈ 2.76** (the ×1.03 CG
  correction now over-penalizes — re-derive ⇒ ≈2.81; do it at the next ledger close).
  **PERMANENT MANUAL KNOB per rule 0.33.** Cert: knob-OFF byte rc=0 · selfcheck 0.000e+00
  1939/1939 Serial + 856/856 CUDA · E.3 verify 2.50×/3.21× · fidelity PASS · options
  TKE/zstar PASS, mEVP formal FAIL (0.34, exonerated).
- **8× @4N accounting: from h17 0.6382 the target 0.5723 needs −10.3 %; with the CGPOLY
  knob on, from 0.6213 it needs −7.9 %.** (State which config a ratio refers to — the
  master set no longer equals the best-knobs set.)
- **Measured pools:** rank imbalance 36.1/53.0 ms @4N/16N — 2D verts BALANCED (0.5/1.0 %)
  but **3D Σnlvls spread 22.2 %/51.1 % (worst rank +7.5 %/+15.9 %)** = the 4N pool
  explained + a chunk of 16N's (session-13 findings §6); ice polar fraction 0–90 %/rank =
  the rest. CG residual pool now ~⅓ of its pre-CGPOLY size when the knob is on. E.1 fuse
  ceilings 7-8/12-13 ms unchanged.

## 2. IN FLIGHT AT HANDOFF — HARVEST THESE FIRST (no monitors survive the session)

1. ~~26323301 CGPOLY 1-yr climate leg~~ — **HARVESTED IN-SESSION: ✅ PASS AT THE BAR
   EXACTLY** (sst 1.00000 · sss 0.99996 · ssh 1.00000 · a_ice 0.99997 vs BOTH refs =
   the M5.23 bar to every printed decimal). CGPOLY's record is COMPLETE: built, certified,
   measured (7.37×/5.31×), climate-documented — permanent manual knob per rule 0.33.
   **⇒ session 14 has NO pending harvests; start directly at §3 (E.IMB.0).**
2. ~~26323500/501 NG5 weighted partitions~~ — **HARVESTED + RESOLVED IN-SESSION: the lever
   is DEAD (findings §12).** The generated dists came back BYTE-IDENTICAL to /pool's; the
   §6 "22 %/51 % 3D imbalance" was an rpart.out format misreading (rule 0.36). TRUE /pool
   NG5 balance: 2D 0.50 %/0.96 %, **3D 0.77 %/1.33 %** — already dual-weighted (the user
   made those dists with this partitioner on May 29). The STOP-rule fired as designed;
   no model runs were spent. `ng5_w3d` copy kept (statics + identical dists, README notes
   the resolution) — reusable only if an ice-weighted variant is ever built.

## 3. SESSION-14 SHAPE — A PURE MEASUREMENT SESSION (recommendation, user-endorsed shape
## 2026-07-17: "update handoff with your recommendation")

**Why measurement-first: the two biggest remaining numbers are respectively UNATTRIBUTED
and STALE.** The board (4N / 16N): rank imbalance **36.1 / 53.0 ms** (runtime-real, cause
unknown — every static volume proxy is FLAT; ice concentration is the surviving suspect);
launch gap + fence spin **~33 ms @4N but h8-era** (the step was 0.87 s then, 0.62 now — the
fraction likely GREW); staging/PCIe 16.5 ms + per-event latency (untouched); E.1 fuses
7-8/12-13 ms (known, byte-class); solver remnants (guess extrapolation −1..3 ms @16N).
The 8× gap is −49 ms @4N — reachable if two of the top three pools convert. Do NOT build
until these are measured; the build lever then picks itself (the same discipline that
caught the E.PART phantom at zero GPU cost).

**The fleet (run 1-4 in parallel; only #1 involves code, and it is SYNCSTATS-class):**

1. **E.IMB.0 — per-rank per-phase attribution (the centerpiece).** A cheap opt-in
   diagnostic (`FESOM_SPEED_PHASESTATS=1`-style; follows the SYNCSTATS pattern — a counter
   NEVER rides the master switch, L91/0.19): per-rank busy/wait wall per phase {ice,
   ocean-compute, CG, halo-wait} accumulated over the timed window, reduced to
   rank-min/mean/max (+argmax rank id) and printed once at the end by rank 0. Run
   `job_m7_ab_env`-style single legs at 4N + 16N (std300, BIN=frozen, `-C a100_80`).
   READ: which phase carries the 36/53 ms straggler.
   - Hypotheses it discriminates: (a) ice concentration (polar-fraction/rank 0.2–91 %
     @dist_16, 0–100 % @dist_64; 11/64 ranks >80 % polar, 17/64 ice-free ⇒ the ice phase
     has a built-in ~2.7× straggler) → lever = ice-weighted partition
     (`jobs/job_m7_ab_mesh` + the `ng5_w3d` copy are ready; needs a partitioner source
     change for the third constraint) or ICELAG (Task F.1, user-approved experiment) or
     EVPCOMPACT (Task E.4); (b) comm-topology skew → different levers; (c) neither →
     re-examine.
   - ALSO re-check the original session-11 E.split jobs were `-C a100_80`-pinned — an
     unpinned mixed allocation fakes up to ~3.4 % as "imbalance" (L94).
2. **Launch/fence census refresh** on h17 at both scales (`scripts/m7_gap_census.py
   --min-gap-ms 0.1`, the setting of record L98) — prices lit-#2 (CUDA-graph capture of
   the CG iteration body + EVPWIDE window + KPP sweeps; Kokkos 4.4.01 has
   `Kokkos::Experimental::Graph` on CUDA) honestly before any graph work.
3. **GPUDirect probe chain** (lit-#3, all env-only on the frozen binary, A.3 caution —
   fidelity gate before ANY adoption): (a) on a gpu node: `lsmod | grep -iE
   'nvidia_peermem|nv_peer_mem'` + `ucx_info -d | grep -iE 'gdr|cuda'`; (b) ab_env leg
   `UCX_TLS=rc_x,cuda_copy,gdr_copy` vs baseline at both scales; (c) HPC-X leg (NVHPC
   24.7 `comm_libs/hpcx` module swap, same binary).
4. **Ride-along A/B (one job, no code): CGPOLY=3 + EVPWIDE=8 composition** — do the two
   opt-in knobs stack for knobbed-config users? Naive additivity ⇒ 16N ~0.226, SYPD ~2.87.
   Legs: off / CGPOLY=3 / EVPWIDE=8 / both (BIN=cgpoly0 — it contains BOTH levers).
   Optional 5th probe if queue is friendly: the RNDV env leg for CGPOLY's 81 KB @4N
   messages (`UCX_RNDV_THRESH=256k` leg vs baseline, both with CGPOLY=3).

**Exit criterion for the session: a table naming each pool's cause + the ONE build lever
for session 15, pre-registered.** (If E.IMB.0 lands early and points clearly at ice, the
ICELAG audit (F.1) can start in the same session — it is the user-approved experiment with
the largest prize if the ice phase is the straggler.)

## 4. AFTER THE MEASUREMENT SESSION (build levers, each needs its own E.0 audit + pre-reg)

Pick per the §3 exit table. The bench: **ice-side** (ice-weighted partition / ICELAG F.1 /
EVPCOMPACT E.4 — whichever the attribution names), **#2 CUDA-graph capture** (if the census
refresh confirms a ≥15-20 ms launch/fence pool), **#3 GPU-direct transport** (only if the
probes open the path AND the env legs pay — fidelity gate before adoption), **E.1 fuses**
(byte-class, always available as the safe filler), **initial-guess extrapolation** (small,
composes with CGPOLY). P-CSI pocketed for ≥64N (0.31). The do-not-chase list is lit-doc §9
(split-explicit rewrite, AMG, pipelined-CG, MPI-4 GPU-triggered, FP64 compression,
implicit-VP-GPU, coloring-vs-atomics, cuSPARSE TDMA, Kokkos 5.0, packs, mixed precision
BANNED).

## 5. STANDING MACHINERY (additions this session)

`jobs/job_m7_ab_mesh` (per-leg MESH same-alloc A/B) · `scripts/m7_part_spread.py` (2D/3D
balance probe of record) · `jobs/job_m7_gate_cgpoly` (3-leg serial gate: byte-off +
selfcheck + iters verify; ⚠️ iters live in run.err) · `jobs/job_m7_cgpoly_mevp_probe`
(solver-class-vs-staleness discriminator) · CGPOLY debug: `FESOM_CGPOLY_SELFCHECK=1`
(bitwise ring-replay check, must print 0.000e+00), `FESOM_CGPOLY_KAPPA`, per-solve iters
log `[cgpoly] solve N: iters=…` (solves 1-3 + every 100th), the `[cgpoly] ACTIVE` announce
prints λ bounds + rings + worst-partner-KB. Everything else per session-12/13 prompts
(ab_env, gate_serial, gpu_gate, provenance, halo-sites, gap-census).

## 6. USER PREFERENCES / DECISIONS (standing + session-13/14)

- 8× @4N is the target (0.21). h17 = the adopted master. **Strict options matrix (0.33) ⇒
  CGPOLY and every future non-bit-identical lever = permanent manual knobs.**
- **Mesh rule 0.32** (copies under /work, never /pool; private CORE2 dist_8 untouchable).
- Partitioner setup: `/home/a/a270088/fesom_part/fesom2/work_part` (binary
  `../bin/fesom_meshpart`, METIS 5.1.0, PART_WEIGHTED built in; env
  `../env/levante.dkrz.de/shell`).
- Always measure; pre-register BEFORE submission; SHA/md5 on every harvest; same-day
  anchors `-C a100_80`; A/Bs same-alloc; options ×3 in the per-lever ladder (L91).
- Pushed through `c9f1749` (user-approved 2026-07-17). **Commits after that are LOCAL —
  ask before pushing.** Mixed precision BANNED. Output under
  `/work/ab0995/a270088/port2/m7/`. CORE2 gates = private mesh (L73).
