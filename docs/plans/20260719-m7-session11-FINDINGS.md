# M7 session 11 — FINDINGS

*2026-07-15 (Fable session 11). Branch `m7-speed`. Opens package E per the session-11 PROMPT:
E.0 (map the 358 exchange events/step to source) ran FIRST, foreground, zero GPU — everything
below is measured from the two censuses of record already on disk
(`gap300_h11/hostprof.sqlite` 4N, `gap300_16n_h14/hostprof.sqlite` 16N, both re-scored at
`--min-gap-ms 0.1`, L98) plus a source audit. New tool: `scripts/m7_halo_sites.py`.*

---

## 1. 🔴🔴 E.0 headline — the pool has a SHAPE the pair census could not see: the CG solver
## is its single largest site (28–31 %), and session 10's "the CG's exchanges are apparently
## NOT in this pool" was an ATTRIBUTION artifact, now corrected.

The gap census names a gap by its (predecessor → victim) kernel pair — but every pack/unpack
kernel of one exchange class carries the SAME tag (the lambda's enclosing function), so every
same-class MPI wait collapses into one `halo → halo` row. Filtering the census's phase view for
`ssh_solve_cg` therefore showed ~0.2 ms and the session-10 handoff concluded (with a "verify"
flag, to its credit) that the CG's exchanges were not in the pool. **They are the largest single
component of it.** `scripts/m7_halo_sites.py` walks the kernel timeline instead: a maximal run
of halo-class kernels = one exchange block; the bracketing compute kernels name the site.

**Reconciliation is exact** (window = steady 198/199 steps, threshold >0.1 ms):

| | 4N (h11) | 16N (h14) |
|---|--:|--:|
| exchanges/step (d1 + d2 + dN) | **349** = 215 + 131 + 3 | **349** = 215 + 131 + 3 |
| events >0.1 ms | 349 | 349 |
| in-block wait ms/step | 95.8 | 123.5 |
| + approach gaps → census pool | 97.3 ✓ | 124.8 ✓ |

**Every one of the ~349 exchanges/step produces exactly ONE >0.1 ms wait, at BOTH scales.**
The "358" of the census tables = the same population plus boundary/approach events. The static
source census matches to the exchange: 215 singles = 146 CG + 66 step-cadence + 3 boundary rows;
131 device2 = 120 EVP + 11 step-cadence; 3 deviceN (EOS×5f, JRA55×8f, ice-coupling×4f).

## 2. THE E LEDGER (deliverable of E.0)

Site → cadence source, kind, exchanges/step, measured wait (ms/step and µs/exchange):

| site (source) | kind × levels | ex/st | 4N ms | 16N ms | µs/ex 4N→16N | fusable? / overlap headroom |
|---|---|--:|--:|--:|---|---|
| **CG solver** `fesom_ssh.cpp:846-897` — `exch(pp)` + `exch(rr)` per iter (2 + 2×~72 iters; the preconditioner SpMV reads rr at halo columns) | NOD2D×1 | **146** | **27.3** | **38.5** | 187→264 | NOT fusable (pp/rr at different loop points). Overlap = interior/boundary SpMV split (code). Algorithmic (pipelined-CG single-exchange variants, local preconditioner) = USER decision |
| **EVP subcycle** `fesom_ice_evp.cpp:716` — field2(uice,vice) ×120 subcycles | NOD2D×1 | **120** | **25.3** | **31.6** | 211→263 | Already fused (M5.23). Serial dependency across subcycles. Overlap = interior/boundary split; or K-wide halo every K subcycles (big code) |
| KPP block `fesom_kpp.cpp:1652,1664,1696-98` — 3 pre-smooth blmc slabs + 3 sweeps × 3 slabs + diffK×2+viscA | NOD3D×nl | 15 | 6.7 | 13.1 | 340-780→730-970 | **FUSE 15→5** (needs per-field `base_off` fieldN variant — slabs differ only in base_off); sweeps serially dependent, slabs within a sweep independent |
| ice FCT `fesom_ice_fct.cpp:703-928` — 3 valuesl + 3 dvalues×3 iters + (plus,minus,values)×3 fem_fct | NOD2D×1 | 21 | 4.2 | 5.5 | ~196→~280 | **FUSE 21→11** (valuesl 3→1, dvalues 3→1 per iter, plus+minus→field2 ×3); values×3 stay (separated by compute) |
| tracer FCT adv `fesom_tracer_adv.cpp:1562,1584,1813` ×2 tracers | NOD3D + ELEM2D_FULL | 6 | 5.0 | 4.8 | 820-843 | fct_LO / tr_xy different kinds — no fuse; plus/minus already field2 |
| Redi `fesom_gm.cpp:1992,2133` ×2 tracers | ELEM2D_FULL + NOD3D | 4 | 6.6 | 3.3 | 1652→800 | different kinds, dependency-separated — no fuse |
| GM chain `fesom_step.cpp:386-417` (sigma_xy, slope pair, fer_K+Ki, fer_gamma, fer_uv) | NOD2D×nl mixed | 6 | 4.8 | 5.4 | 430-1364 | pairs already fused (M5.23 L3); rest dependency-separated |
| T/S values (post-Redi ×2, post-trdiff ×2) `fesom_step.cpp:1151,1180,1249,1256` | NOD3D×nl | 4 | 1.8 | 3.3 | 340-910 | **post-trdiff T+S ADJACENT → field2 (saves 1)**; post-Redi pair separated by S's Redi |
| momentum (uvnode_rhs, uv_rhs×3, u_b+v_b) | NOD3D/ELEM3D | 6 | 3.5 | ~4.4 | 510-915 | dependency-separated |
| SSH step-cadence (ssh_rhs, d_eta, uv, uvnode, hbar+rhs_old, w_e+w_i, w+fer_w, cfl_z, hnode_new, hnode+helem, Kv+Av) | mixed | ~14 | ~7.5 | ~9.5 | 200-1600 | Kv+Av / hnode+helem = different kinds; rest serial |
| EOS 5-field fieldN + bvfreq smoother exit | NOD3D×nl | 2 | 1.8 | 1.7 | 300-1500 | already max-fused (M5.23 fieldN) |
| forcing (bulk stress + heat/water, jra55 8-field, ice-coupling 4-field, srfoce) | NOD2D | 5 | 1.4 | 2.7 | 190-720 | already fieldN'd |

**Regime split, the load-bearing structure:** the pool is TWO populations —
- **2D-latency class** (CG 146 + EVP 120 + ice FCT 21 + misc ≈ 292 ex/step, payload ~10-20 KB/msg):
  **56.8 ms/step at 4N, 75.6 at 16N (58/61 % of pool)**, per-event 190-280 µs ≈ a per-exchange
  FLOOR (fence + MPI stack + staged pipeline + neighbor skew), scale-invariant count.
- **3D-bandwidth class** (~57 ex/step, NOD3D/ELEM payloads ~1-8 MB/exchange): 39/48 ms, per-event
  0.3-1.7 ms, tracks payload (Redi halves at 16N; KPP doubles — mixed latency+bandwidth).

**Transport reality (answers §2.1 "staged?"):** the device path hands DEVICE pointers to MPI
(`fesom_halo_device.cpp` Irecv/Isend on `send_d/recv_d`) — but UCX pipelines every byte through
a host pinned bounce: the censuses show **symmetric D2H+H2D staging of 181 MB/step (4N) /
162 MB/step (16N)** inside the waits (~3.9k/6.1k copies/step, ~30-50 KB each; 16.5 ms/step of
PCIe at both scales). Transport of record: `tag(rc_mlx5/mlx5_0:1)` (A.3 diag). NOT GPU-direct
RDMA. What actually died in history (the PROMPT asked): **M5.17** = comm OVERLAP, killed by a
79 %-imbalance/21 %-comm barrier split measured on the M5.16 binary (2.68 s/step, GPU ~30 %
busy — a regime 4× slower than today's; the split is STALE); **L67-3** = persistent MPI
requests (flat: posting was never the cost); **A.3** = rndv-scheme FORCING (+35 % catastrophic
— UCX's pipeline is load-bearing; `UCX_RNDV_FRAG_MEM_TYPE=cuda` was the one green leg at
−1.15 %, sub-bar at 4N, NEVER tested at 16N where the pool share is 3×).

## 3. E levers, re-ranked on the ledger (supersedes the PROMPT §2.2 ordering)

1. **E.split (FIRST, measurement not lever, jobs submitted — see §4):** re-run the M5.17
   barrier-isolation on TODAY'S regime at both scales. The 2D-latency class is 58-61 % of the
   pool; whether its ~200-260 µs/event is per-event floor (transport-attackable) or arrival
   skew (only overlap/partition-attackable) decides everything below. The instrument
   (`FESOM_HALO_BARRIER` / `FESOM_HALO_MPI_PROF`) is committed and wired (main.cpp:1424).
2. **E.1 COALESCE — honest ceiling from the ledger: −7-8 ms (4N) / −12-13 ms (16N)**, i.e.
   ~24 events saved (KPP 15→5, ice FCT 21→11, T+S post-trdiff 2→1) × their measured per-event
   waits. Needs a per-field-`base_off` fieldN variant (blmc/diffK slabs). NOT the frontier the
   PROMPT priced it as — the 358 events are NOT dominated by fusable step-cadence sites; the
   two subcycled sites (CG+EVP, 76 % of the 2D class) are structurally unfusable. Pre-register
   floor 0 (L93/L98); gates per PROMPT E.1 (knob-OFF byte, multi-rank CUDA fidelity, options ×3).
3. **E.3/E.4 transport (env-first, A/B at BOTH scales):** legs in evidence order —
   `UCX_RNDV_FRAG_MEM_TYPE=cuda` at 16N (the A.3 survivor, untested where it matters);
   `UCX_RNDV_THRESH` sweep (the 2D msgs sit near the rndv boundary; A.3 forced the SCHEME, never
   moved the THRESHOLD); gdr_copy probe (`UCX_TLS=+gdr_copy`, diag first — is it even present?).
   Ceiling if the split says "floor": some fraction of 57/76 ms. Ceiling if "skew": ~0.
4. **E.2 OVERLAP (code, biggest single prize if the split allows):** interior/boundary split of
   the CG SpMV (hides up to 27/38 ms) and the EVP subcycle (25/32 ms). Gated on E.split — this
   is exactly the lever M5.17 killed, on a stale regime; do not build it before the new split.
5. **E.5 (algorithmic, USER decision required, rule 0.17):** the CG does 2 exchanges + 2
   Allreduces per iteration × ~72 iters. Pipelined/single-exchange CG reformulations or a
   halo-free preconditioner change rounding (same FP64, same tolerance, different trajectory —
   climate-close bar, not bit-id). Park until the user weighs in; the 8× question passes through
   this site (CG+EVP = 63/70 ms of a 97/125 ms pool).

## 4. Pre-registration — E.split (submitted this session)

Two `job_m7_ab_env` jobs, h16-pinned (`BIN=m7/bin/h16/fesom_port_cuda`, md5 `470ead46`),
NG5, DT=180, NSTEPS=300, same-alloc A/B (L94-immune), `-C a100_80`:
- **4N**: `-N4 --ntasks=16`, TAG=`e0split_4n`
- **16N**: `-N16 --ntasks=64`, TAG=`e0split_16n`
- LEG1 `prof::FESOM_SPEED=1;FESOM_HALO_MPI_PROF=1;FESOM_SPEED_SYNCSTATS=1`
- LEG2 `barrier::FESOM_SPEED=1;FESOM_HALO_BARRIER=1`

Pre-registered readouts and rules:
- (i) LEG1 `[halo-mpi-prof]` Waitall/step should land near the census MPI-covered pool
  (**4N ≈ 89 ms, 16N ≈ 116 ms**, ±20 %) — cross-validates the walker. `[halo-syncstats]`
  exchanges/step should read **≈ 349** (counter-vs-census check, the user's directive).
- (ii) LEG2 splits barrier+waitall into imbalance vs comm. **Decision rule: comm ≥ 40 % ⇒
  E.3/E.4 live at full pool share and E.2 is real; imbalance ≥ 75 % (the M5.17 outcome) ⇒
  E.1/E.4-transport yields discount to the comm fraction, and the lever list pivots to
  overlap-tolerant work + partition/topology.** In between: prorate.
- (iii) Overhead sanity: LEG2 wall vs LEG1 ≤ +6 % (349 barriers/step × 20-50 µs at 64 ranks).
- (iv) The legs differ only in DIAGNOSTIC env; no fidelity gate needed (nothing adoptable here).

## 5. E.CG1 `FESOM_SPEED_CGPIPE` — single-exchange 2-ring PCG (pre-registration, design frozen BEFORE implementation)

*User decision 2026-07-15: decisions 1+2 are approved as OPT-IN options — default OFF/safer,
explicit switch to enable, never folded into `FESOM_SPEED=1` without a later explicit promotion.
Implementation order: pipelined CG first. Knob class: `fesom_speed_on_exp` (opt-in only).*

**Mechanism.** The CG does 2 halo exchanges/iteration (`pp` before the SpMV, `rr` before the
preconditioner — both operators are sparse, each needs its operand's fresh 1-ring halo, and the
operands are updated after the other operator: 2 exchanges is structural for exact PCG on a
1-ring). CGPIPE replaces both with **ONE fused 2-ring exchange of `rr`** per iteration:
- `rr` is exchanged on ring1+ring2 (ring2 = neighbors-of-ring1; new comm lists + possibly new
  diagonal partner ranks, built once at setup).
- `zz = M⁻¹·rr` is then computable at owned **and ring1** rows — the ring1 preconditioner rows
  are shipped VERBATIM (values + column order) from their owners once at setup. `pr_values` is
  set-once and never refreshed (fesom_ssh.cpp banner; the zstar path increments `values` only,
  and A is only ever applied at owned rows) ⇒ ship-once is valid under linfs AND zstar.
- `pp = zz + β·pp` is then maintained at owned+ring1 by the recurrence — `exch(pp)` is DELETED.

**The byte-identity claim (the novel HARD gate this lever earns).** Ring1 `rr` bytes = owner
bytes (same com lists as today). `zz` at a ring1 row = the owner's `zz` BITWISE: verbatim row
(same coefficients, same summation order), same operand bytes, same loop shape in the same TU
(same FMA contraction). `pp` at ring1 ≡ owner's `pp` by induction from `pp₀ = zz₀` with
globally-identical β. All owned-row kernels byte-unchanged ⇒ `App`, all dots, all scalars,
`d_eta` identical every step ⇒ **the whole model output is byte-identical ON vs OFF** — not
merely climate-close. Corollary: iteration counts and residual prints must match EXACTLY.

**Event arithmetic.** Exchanges/solve: 2+2k → 2+k (k ≈ 72) ⇒ CG events 146 → 74/step. New
per-event cost slightly higher (bytes ×~2.3 — still ~25-45 KB/msg, latency-dominated; partner
set grows by ring2-only diagonal ranks). **Pre-registered range: 4N −8…−13 ms/step (central
−10 ≈ −1.5 %); 16N −12…−19 ms (central −15 ≈ −5.5 %); floor 0** (L93/L98: latency-pool
conversion factors unknown until the first A/B; E.split may further discount — if the split
says imbalance-dominant, each surviving exchange absorbs more skew and the saving compresses).

**Ladder (all pre-registered):** (i) build serial+cuda; (ii) pi smoke np1+np2; (iii) knob-OFF
byte gate vs h16 (diff_snap rc=0); (iv) **knob-ON byte gate: CUDA multi-rank, CGPIPE=1 vs
CGPIPE=0, diff_snap rc=0** + identical CG iteration counts; (v) `FESOM_CGPIPE_SELFCHECK=1`
bring-up validator (recurred pp ring1 vs a reference host exchange, max|Δ| must be 0.0);
(vi) options matrix ×3 with the knob ON (comm-structure change = L91 ownership-adjacent);
(vii) A/B at BOTH 4N and 16N (same-alloc, h16-derived binary, KNOBS legs); (viii) counter
checks: `[cgpipe]` build announce (ring2 size, partners), CG events/step ≈ 74 in a walker
trace, `[fesom_speed] FESOM_SPEED_CGPIPE = ON` announce present (L80/0.19).

**⚠️ AMENDMENT to gate (iv), made BEFORE any gate ran** (recorded per rule 0.x honesty): a
full-model CUDA ON-vs-OFF byte diff is NOT well-defined — the step contains atomic scatters
(D22), so even same-binary CUDA runs differ run-to-run and diff_snap between two CUDA runs
proves nothing. The lever was therefore made **backend-agnostic on purpose** (pure Kokkos+MPI,
no CUDA API), which unlocks the STRONGER established gate: **(iv-a) the FORCE_SERIAL byte
proof** — Serial np=8 CORE2 (ice active), `FESOM_SPEED_FORCE_SERIAL=1;FESOM_SPEED_CGPIPE=1`,
diff_snap rc=0 vs the certified m6 baseline (job_m7_gate_serial's documented double-duty) —
plus **(iv-b) the in-vivo CUDA selfcheck** (per-iteration recurred-vs-exchanged pp on device,
max|Δ|==0 — this is the CUDA-side FMA-contraction check the static argument can't replace).

**Bring-up (login node, pi, Serial, 2026-07-15): GREEN.** np=2 AND np=8, ON vs OFF
`diff_snap rc=0` (ALL FIELDS BIT-IDENTICAL), every selfcheck line exactly 0.000e+00, identical
iteration counts, clean teardown (after adding `fesom_ssh_cgpipe_free()` — file-static Views
must not outlive Kokkos::finalize, the fesom_halo_device_free class). Two real bugs caught by
the protocol FESOM_CHECKs: (1) `partit->part[]` does NOT encode ownership as contiguous gid
ranges (its header comment is misleading) — owners are now shipped per column from the com
graph, never derived from part[]; (2) Kokkos `is_view_label<const char*>` rejects label
VARIABLES (literals are char[N]) — std::string wrap.

**Frozen binary `m7/bin/cgpipe0/`**: CUDA **`ef86c3c9`** (REFROZEN — see below) / Serial
`851e1ca9` (h16 code + the opt-in lever; PROVENANCE.txt in the dir).

**Gate results, first wave (2026-07-15 night):** serial knob-OFF 26288247 **PASS rc=0** ·
**FORCE_SERIAL ON byte proof 26288248 PASS rc=0** — CGPIPE=ON on Serial np=8 CORE2 (ice
active) is BIT-IDENTICAL to the certified m6 baseline; announce fired (`run.err`, 0.19);
`[cgpipe] built: ring2(max)=325 partners(max)=3`; **2627 selfcheck lines, ALL exactly
0.000e+00** · gpu knob-OFF 26288249 **PASS**.

**🔴 A STALE-BINARY TRAP, caught by the gate ladder in minutes:** gpu ON/self/TKE
(26288250/51/52) CRASHED with the *pre-fix* FATAL text ("ring2 gid owned by SELF") — the
first cgpipe0 CUDA freeze (`60fe548b`) was built BEFORE the owner-shipping fix; only Serial
had been rebuilt after the last source edit. The message text itself was the tell (that
CHECK no longer exists in the source). Rule: **rebuild BOTH backends after the LAST edit,
then freeze — and when a frozen pair splits Serial-pass/CUDA-fail, suspect the freeze before
the code.** Refrozen CUDA `ef86c3c9` (verified: the stale string is absent from the binary).

**Jobs, second wave:** gpu ON 26288437 · self 26288438 · options TKE/mEVP/zstar
26288439/40/41 · **A/B 4N 26288442 · A/B 16N 26288443** (LEG1 `FESOM_SPEED=1` vs LEG2
`FESOM_SPEED=1;FESOM_SPEED_CGPIPE=1`, NSTEPS=300, same-alloc; stale-binary predecessors
26288253-56 scancelled before start).

**Non-goals:** EVP untouched (next lever); host CG and Serial builds untouched (CUDA-only
lever); no promotion into `FESOM_SPEED=1` — explicitly opt-in until the user decides otherwise.

## 6. ⭐⭐ RESULTS (2026-07-16 early) — E.split lands; CGPIPE −1.41 % @4N / −8.07 % @16N

**E.split (h16 `470ead46` ✓ both, same-alloc A/B, 300 steps, pure a100_80):**

| | 4N (26285953) | 16N (26285954) |
|---|---|---|
| prof-leg Waitall (rank min/mean/max ms/step) | 53.2 / **63.0** / 69.2 | 54.3 / **71.8** / 87.0 |
| barrier split of (barrier+waitall) | **imbalance 42 % (36.1 ms) · comm 58 % (50.3 ms)** | **imbalance 51 % (53.0 ms) · comm 49 % (50.2 ms)** |
| barrier-leg overhead (iii) | +0.37 % ✓ (≤6 %) | +2.10 % ✓ |
| syncstats exchanges/step (i) | 361.7 | 361.8 |

- **Decision rule §4(ii) OUTCOME: the comm branch fires at BOTH scales (58 %/49 % ≥ 40 %)** ⇒
  E.3/E.4 transport levers and E.2 overlap are LIVE at roughly half-to-full pool share.
  **M5.17's 79 %-imbalance verdict is CONFIRMED STALE** on today's regime (79 → 42-51 %).
- **A NEW measured pool: rank imbalance 36.1 / 53.0 ms/step** — partition/work-balance
  (Lever-D class) is now a first-class, E-sized target, especially at 16N where it is the
  LARGER half.
- Cross-check (i), recorded honestly: the exchange COUNT confirms the census exactly-in-class
  (361.7 ≈ 349 steady + startup/jra55 extras; Waitall calls/step 366). The ms BAND as
  pre-registered (census MPI-covered 89/116 ±20 %) was mis-specified, not the measurement:
  census "MPI-covered" includes Allreduces inside GPU-idle gaps; the prof timer counts halo
  Waitalls only but INCLUDING time when the GPU is busy — different projections of the same
  object (63/72 measured). The census halo-pool (97/125) remains the sizing of record.
- Transport note: 4N runs `tag(rc_mlx5)`, 16N runs `tag(dc_mlx5)` — UCX switches to
  dynamic-connect at 64 ranks (E.4-relevant: DC has different latency characteristics).

**CGPIPE A/Bs (cgpipe0 `ef86c3c9` ✓ both, `FESOM_SPEED=1` vs `+FESOM_SPEED_CGPIPE=1`,
same-alloc, 300 steps, min of 2):**

| | 4N (26288442) | 16N (26288443) |
|---|---|---|
| off | 0.6472 (h16 anchor 0.6467 ✓ 0.08 %) | 0.2626 (ledger 0.2629 ✓) |
| **cgpipe** | **0.6381** | **0.2414** |
| Δ | **−1.41 % = −9.1 ms** | **−8.07 % = −21.2 ms** |
| vs pre-reg (§5) | −8…−13 ms, central −10: **HIT** | −12…−19 ms, central −15: **BEAT THE CEILING by 2.2 ms** |

- The 16N over-run is the campaign's 5th consecutive wrong-LOW (L93 entanglement: deleting
  72 sync points/step also recovers skew absorption + downstream launch gaps the per-event
  arithmetic can't see).
- CG iteration counts IDENTICAL ON-vs-OFF across all 20 gate steps ✓ (byte-identity corollary).
- **Implied, IF the user adopts CGPIPE into the benchmark config** (it stays OPT-IN per the
  standing decision — no promotion into `FESOM_SPEED=1`): 4N 0.6381 ⇒ **ratio 7.18×**;
  16N 0.2414 ⇒ 16N ratio ≈ **5.09×**, **Stage-2 SYPD@dt240 ≈ 2.65** (from 2.43). Ledger-grade
  numbers require same-day anchors at adoption time (these A/B legs already cross-check the
  ledger to 0.1 %).
- **The 8× arithmetic, updated for the user (rule 0.17):** with CGPIPE counted, 8× at 4N needs
  another −10.3 % (0.6381 → 0.5723). The measured 4N pools still on the table: EVP subcycle
  25.3 ms (E.EVP1 wide-halo, approved as option 2), E.1 coalescing ceiling ~7-8 ms, CG residual
  ~18 ms (74 remaining events — E.4 transport now proven ~58 % comm), 3D-site transport share,
  and the newly-priced 36.1 ms imbalance pool. At 16N the same levers act on larger shares.
  Reachability improved again; the endpoint decision remains the user's.

## 7. Machinery + corrections log

- **NEW `scripts/m7_halo_sites.py`** — site attribution for the halo pool from any nsys sqlite;
  reuses m7_gap_census loaders; window/threshold conventions identical. This is the tool that
  found the CG. Reproduce the ledger: `python3 scripts/m7_halo_sites.py <sqlite>`.
- Correction to session-10 findings §6 wording: "the CG's exchanges are apparently NOT in this
  pool" — refuted here (§1); the ~70 KB staged copies are UCX pipeline chunks, not per-message
  payloads (2D msgs are ~10-20 KB; 3D exchanges are MB-scale).
- The PROMPT §2.1 question "how many are per-CG-iteration?" — answer: **2 per iteration**
  (`pp` before the fused SpMV·dot, `rr` after the axpy, because the Jacobi-ish preconditioner
  SpMV gathers rr at halo columns).
- jra55 8-field fieldN fires EVERY step (measured 1.00 blk/st), not at the 3-h forcing cadence.

## 8. USER DECISIONS (2026-07-16, mid-session)

- **8× IS the target** (user: "the 8x is still a target, yes") — rule 0.17 resolved. With
  CGPIPE counted, 8× at 4N = another −10.3 % (0.6381 → 0.5723); the measured shopping list:
  EVP subcycle 25.3 ms → E.EVP1 wide-halo (approved option 2, next lever) · E.1 fuses ~7-8 ms ·
  CG residual ~18 ms (E.4 transport, comm share proven 58 %) · rank-imbalance pool 36.1 ms
  (partition lever). At 16N the same levers act on larger shares (EVP 31.6, imbalance 53.0).
- **CGPIPE adoption into `FESOM_SPEED=1`**: explained to the user (one-line promotion `_exp`→
  master-riding + re-certification + same-day anchors ⇒ official ratio becomes ~7.18×);
  RECOMMENDED because the lever is byte-identical (the "default-safer" opt-in rationale
  protects against changed results, and CGPIPE provably changes none). AWAITING the user's
  word; until then every official number keeps the non-CGPIPE config.
