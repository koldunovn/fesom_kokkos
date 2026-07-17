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

1. **26323301 — CGPOLY 1-yr CUDA climate leg** (user-approved, documentation only; ~3 h
   from 2026-07-17 ~09:50; `KNOBS="FESOM_SPEED=1;FESOM_SPEED_CGPOLY=3"`, BIN=cgpoly0).
   Output `/work/ab0995/a270088/port2/m7/t1_1yr.26323301.out` — the job template computes
   the comparison itself (Task-1.5 pattern); **the bar = M5.23's sst 1.00000 · sss 0.99996
   · ssh 1.00000 · a_ice 0.99997** (refs in the job header; solver-class ⇒ expect the same
   magnitudes, exact repetition NOT required). Write the result into the ledger E.CG2
   section + findings.
2. **26323500 / 26323501 — NG5 weighted partitions dist_16 / dist_64** (single-rank
   compute jobs; FORCA20 took 92 s at 2.1M nodes, NG5 = 7.4M ⇒ expect ≲30 min; outputs
   `/work/ab0995/a270088/port2/m7/epart/gen{16,64}/fesom_meshpart.out` + the dist dirs in
   `/work/ab0995/a270088/port2/mesh/ng5_w3d/`). Harvest = (a) the job's own
   `LOAD BALANCING` block (2D + 3D percent lines), (b) the probe of record:
   `python scripts/m7_part_spread.py /work/ab0995/a270088/port2/mesh/ng5_w3d 16 64`
   (nereus python). **Pre-registered expectation (findings §11): 3D spread 22.2 % → ≲2-3 %
   (dist_16), 51.1 % → ≲3-5 % (dist_64); 2D spread may open a few %; verify
   `Distribution weight: 2D and 3D nodes` printed.** If the spread does NOT collapse,
   STOP and re-audit (wrong binary / wrong weights) before any model run.

## 3. SESSION-14 SHAPE — E.PART model phase (the main lever; lit-survey rank #1)

1. **E.PART.1 — pre-register THE MODEL NUMBERS** (only after the §2.2 probe): the honest
   model from §1 pools with 0.31 humility applied — straggler-time removal is not
   event-deletion, but do NOT assume the full 36/53 ms converts. Suggested pre-reg shape:
   4N ceiling ≈ −30 ms (the 3D-skew share), central −15..25 ms (0.6382 → ~0.613-0.623 on
   the master config); 16N ceiling ≈ −20 ms, central −8..14 ms (0.2413 → ~0.227-0.233);
   floors 0 (multi-constraint partitions worsen edgecut/halo — the announce/partner stats
   will show it). WRITE IT BEFORE SUBMITTING.
2. **E.PART.2 — the mesh A/Bs** (machinery READY: `jobs/job_m7_ab_mesh`, per-leg MESH
   paths, KNOBS applies to all legs):
   ```
   sbatch -N16 --ntasks=64 --export=ALL,DT=180,NSTEPS=300,TAG=epart_ab_16n,\
     BIN=/work/ab0995/a270088/port2/m7/bin/cgpoly0/fesom_port_cuda,KNOBS="FESOM_SPEED=1",\
     LEG1="old::/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5",\
     LEG2="w3d::/work/ab0995/a270088/port2/mesh/ng5_w3d" jobs/job_m7_ab_mesh
   ```
   + the same at `-N4 --ntasks=16` with TAG=epart_ab_4n. Two more legs are free — consider
   LEG3/LEG4 = the same two meshes with `KNOBS` extended… NO: KNOBS is job-global; run a
   SECOND pair with `KNOBS="FESOM_SPEED=1;FESOM_SPEED_CGPOLY=3"` to measure the
   partition×CGPOLY composition (the knobbed config is where 16N's remaining imbalance
   share is largest). Off-leg (old mesh) must reproduce the h17/cgpoly anchors same-day.
3. **E.PART.3 — validation/cert design (decomposition-class, NOT byte-class):** a
   partition change reorders Allreduces/atomics ⇒ same class as changing the rank count
   (routinely done in the scaling campaigns). Gates: the run's own bench-finite health +
   fields sane + the A/B anchors + (if adopted for production benches) a fidelity-style
   NG5 short-run cross-check old-vs-new partition (climate-close floors). NO FORCE_SERIAL
   byte claim exists or is needed. Adoption of `ng5_w3d` as the BENCHMARK mesh = the
   user's call with the A/B numbers (it changes every future anchor's decomposition!).
4. **Watch items:** partner counts + halo bytes on the new partition (cgpipe/cgpoly
   announces print them; the E.split instruments if needed); the ice-concentration axis is
   NOT addressed (0–90 % polar fraction — an ice-mask second constraint would need
   partitioner source changes; second-order, separate lever); dist_16 node counts CHANGED
   per rank ⇒ per-rank memory shifts slightly (NG5@4N was ~41 GiB peak on 40 GB cards in
   the JAX runs — OUR runs fit fine at 4N, just note it).

## 4. AFTER E.PART (the lit-survey ranking, each needs its own E.0 audit + pre-reg)

- **#2 Launch/fence hygiene + CUDA-graph capture** (EVPWIDE K-window, CGPOLY iteration
  body, KPP sweeps): FIRST re-measure the stall budget on h17 at both scales
  (`m7_gap_census.py --min-gap-ms 0.1`) — the 16.8+16.1 ms numbers are h8-era.
- **#3 GPUDirect probe chain** (all S-effort, pre-registered in lit-doc §7): peermem lsmod
  on a gpu node → gdr_copy env leg (`UCX_TLS=rc_x,cuda_copy,gdr_copy`) → HPC-X module leg.
  A.3 precedent: transport swaps can be catastrophic — ab_env legs only, fidelity gate
  before any adoption.
- **Solver leftovers:** initial-guess extrapolation (−1..3 ms @16N, S-effort); P-CSI
  pocketed for ≥64N (0.31). **Optional:** RNDV env-leg for CGPOLY's 81 KB @4N messages;
  CGPOLY+EVPWIDE=8 composition A/B (untested; both knobs opt-in).
- E.1 fuses (7-8/12-13 ms ceilings) remain on the table; the do-not-chase list is
  lit-doc §9 (split-explicit rewrite, AMG, pipelined-CG, MPI-4 GPU-triggered, FP64
  compression, implicit-VP-GPU, coloring-vs-atomics, cuSPARSE TDMA, Kokkos 5.0, packs,
  mixed precision BANNED).

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
