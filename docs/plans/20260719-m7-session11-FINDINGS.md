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

## 5. Machinery + corrections log

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
