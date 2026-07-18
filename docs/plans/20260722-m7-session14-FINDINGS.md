# M7 session 14 — FINDINGS (the pure measurement session)

*Started 2026-07-17 (Fable). Prompt: `20260722-m7-session14-PROMPT.md`. Shape per the
user-endorsed recommendation: measure the three unattributed/stale pools, build nothing
until each is named; exit = attribution table + ONE pre-registered session-15 lever.*

Board at session start (4N / 16N, NG5, h17 master 0.6382 / 0.2413; knobs-on best
0.6213 / 0.2314): rank imbalance **36.1 / 53.0 ms** (runtime-real, UNATTRIBUTED);
launch gap + fence spin **~33 ms @4N h8-era (STALE)**; staging/PCIe 16.5 ms + per-event
latency; E.1 fuses 7-8/12-13 ms; solver remnants −1..3 ms @16N. 8× @4N = −49 ms from
0.6213.

## 1. L94 re-check of the session-11 E.split measurement — CLEAN ✅

The 36.1/53.0 ms imbalance numbers were measured by jobs 26285953 (e0split_4n) /
26285954 (e0split_16n); both node lists are pure l5xxxx (a100_80): 4N
`l[50103,50106,50109,50157]`, 16N `l[50100,50124,…,50193]`. **No heterogeneity
contamination — the pool is real hardware-wise.** (The ab_env template has carried
`-C a100_80` in its header since session 12.)

## 2. PRE-REGISTRATIONS (written BEFORE submission; std300 = 300 steps dt180 NG5,
## min-of-2 same-alloc via job_m7_ab_env unless stated)

### 2.1 Composition A/B @16N (BIN=cgpoly0 `ee2c4fdd`, legs off/cg3/ew8/both)
Known singles vs off: cg3 −4.26 %, ew8 −2.2 %. Naive additivity ⇒ both ≈ 0.226.
- **Central: both ∈ [0.2265, 0.2295]** (additivity minus a rule-0.31 marginal-decay
  haircut — the two levers thin the SAME latency pool's tail).
- Decision rule: both ≤ 0.229 ⇒ document "knobs stack" (recommended pair for
  knobbed-config users; SYPD@dt240 re-derive). both > 0.2314 (worse than cg3 alone)
  ⇒ interference — do NOT recommend pairing. In between ⇒ stacking partial, note it.
- Harvest checks: md5 provenance; `[cgpoly] ACTIVE` in cg3+both, evpwide announce in
  ew8+both (L80); off-leg vs 0.2413/0.2417 anchors.

### 2.2 4N knob job (BIN=cgpoly0, legs cg3 / cg3rndv / cg3ew8)
- cg3 reference expected ≈ 0.6213 (session-13, same config).
- **cg3rndv** (`UCX_RNDV_THRESH=256k`; CGPOLY worst-partner 81.1 KB @4N): central
  **−0..1.5 % vs cg3**; regression possible (A.3 precedent). Adopt-consider only if
  ≥ −1 % AND later gate-clean; else document + drop.
- **cg3ew8** (4N composition): ew8 alone was −0.6 % @4N ⇒ central **cg3 × (−0.3..0.8 %)**.
- Decision rule: this job only DOCUMENTS the knobbed-config 4N number; no adoption
  decision hangs on it.

### 2.3 GPUDirect env A/B (BIN=h17 `f8384e86`, both scales; legs ref / gdrs / gdrh)
- gdrs = the lit-§7 pre-registered spec `UCX_TLS=rc_x,cuda_copy,gdr_copy` (drops sm/self
  — if it underperforms gdrh, the delta is the host-transport loss, not gdr_copy).
- gdrh = `UCX_TLS=rc_x,sm,self,cuda_copy,gdr_copy` (adds gdr_copy, keeps host paths).
- If gdr_copy is unavailable on compute nodes, these legs FAIL to init (that IS the
  probe answer; the peermem probe §2.5 tells us why).
- **Central: gdrh −0..2 % @4N, −0..3 % @16N** (staging pool 16.5 ms @4N ≈ 2.6 %;
  lit: host-staging ≈5-7 % of total elsewhere). A.3 caution: catastrophic regressions
  possible — that is what the A/B protocol is for.
- Decision rule: any leg ≥ −1.5 % @16N ⇒ escalate (fidelity gate with that env + HPC-X
  leg next session). Else: GPUDirect chain closes as "no cheap win on Levante stock".

### 2.4 Launch/fence census refresh (job_m7_nsys, BIN=h17, FESOM_SPEED=1, 35 steps,
### `-C a100_80` added on the CLI; census read with `m7_gap_census.py --min-gap-ms 0.1`, L98)
- h8-era numbers: launch gap 16.8 + fence spin 16.1 ≈ 33 ms @4N (step was 0.87 s).
- **Central: launch+fence 25–40 ms @4N (4-6.5 % of the 0.62 s step); 15–30 ms @16N.**
- Decision rule: ≥ 15-20 ms @4N confirmed ⇒ lit-#2 (CUDA-graph capture of CG body /
  EVPWIDE window / KPP sweeps) enters the session-15 bench with an honest price. Below
  that ⇒ deprioritized behind the imbalance/ice outcome.

### 2.5 peermem/gdrdrv probe (1-node gpu job, 6-min walltime)
`lsmod | grep -iE 'nvidia_peermem|nv_peer_mem|gdrdrv'` + `/dev/gdrdrv` + `ucx_info -d`
grep gdr/cuda + `nvidia-smi topo -m`. Binary outcome, no pre-reg number: peermem absent
⇒ GPUDirect **RDMA** closed on Levante stock (env legs then measure gdr_copy-for-eager
only, which needs gdrdrv; both absent ⇒ expect the strict UCX_TLS legs to fail init).

### 2.6 E.IMB.0 PHASESTATS legs (BIN=phst0 once gated; legs ref / phst / phstbar; both scales)
- **Perturbation pre-reg: phst leg s/step within ±0.5 % of the ref leg** (pure timers,
  no fences added); phstbar (adds FESOM_HALO_BARRIER=1) +0.4 % @4N / +2.1 % @16N class
  (session-11 measured overheads).
- Attribution decision rule (pre-registered): the phase with the largest
  (busy_max − busy_min) across ranks, if it carries ≥ 60 % of the total per-rank step
  spread, NAMES the pool. (a) ICE ⇒ session-15 lever = ice-side (ice-weighted partition
  vs ICELAG F.1 vs EVPCOMPACT — choose after the polar-rank correlation); (b) OCEAN/CG
  busy spread large despite 0.77/1.33 % static 3D balance ⇒ runtime effect (convection/
  KPP iteration-count class) — re-audit before any lever; (c) no dominant busy spread,
  waits diffuse ⇒ comm topology — transport/overlap levers.
- Cross-checks: Σ phase walls ≈ loop s/step; phstbar's per-phase barrier-wait should
  reproduce the E.split totals (36.1/53.0 ms class) and split them BY PHASE.

## 3. Submissions (fill in as they go)

| # | what | job id | state |
|---|---|---|---|
| 2.1 | composition A/B 16N (cgpoly0: off/cg3/ew8/both, std300) | 26324350 | submitted |
| 2.2 | 4N knobs (cgpoly0: cg3/cg3rndv/cg3ew8, std300) | 26324351 | submitted |
| 2.3 | GDR env A/B 4N (h17: ref/gdrs/gdrh, std300) | 26324352 | submitted |
| 2.3 | GDR env A/B 16N (same legs) | 26324353 | submitted |
| 2.4 | census nsys 4N (h17, 35 steps, `-C a100_80` CLI) | 26324354 | submitted |
| 2.4 | census nsys 16N | 26324355 | submitted |
| 2.5 | peermem/gdrdrv probe (1 node, 6 min) | 26324356 | submitted |
| 2.6 | gate: knob-OFF byte (gate_serial) | 26324576 | submitted |
| 2.6 | gate: knob-ON byte (FORCE_SERIAL+PHASESTATS) | 26324577 | submitted |
| 2.6 | gate: CUDA fidelity (FESOM_SPEED=1+PHASESTATS) | 26324578 | submitted |
| 2.6 | PHASESTATS legs 4N (ref/phst/phstbar, BIN=phst0 `c06d4094`) | 26324579 | submitted |
| 2.6 | PHASESTATS legs 16N (same) | 26324580 | submitted |
| 2.3 | GDR engagement diag 4N (3 steps, UCX_LOG_LEVEL=info in legs) | 26324742 | submitted |
| 2.2 | 4N composition resubmit (cg3/cg3ew8 pair; 26324351 TIMEOUT after 2⅚ legs) | 26325395 | submitted |
| §10 | phst1 gates (off-byte / on-byte / gpu fidelity) | 26326814-16 | submitted |
| §10 | E.IMB.1 discriminator 4N (bar/barknobs, BIN=phst1, fresh nodes `-x l[50063,50066,50075,50081]`) | 26326817 | submitted |

## 4. E.IMB.0 PHASESTATS — BUILT + SMOKED (commit 20af279, frozen `m7/bin/phst0`)

`FESOM_SPEED_PHASESTATS=1` (opt-in, `fesom_speed_on_exp` — never rides the master switch).
No-fence phase marks {force, ice, coupl, ocean, cg, other} in the main loop + a CG
carve-out around `fesom_ssh_solve_cg_kk`; MPI-wait attributed to the CURRENT phase by
PMPI interposition (Wait/Waitall/Barrier/Allreduce/Reduce/Bcast/Alltoall(v)/Gather(v)/
Scatterv/Allgather(v) — strong symbols over OpenMPI's weak aliases, straight PMPI
pass-through, zero model arithmetic); armed only inside the timed window. Report:
per-phase busy(=wall−wait)/wait rank-min/mean/max + argmax rank + full per-rank table.
- np2 CORE2 login smoke: **TOTAL busy+wait = 5515.8 ms/step vs loop timer 5515.9 — the
  accounting closes to 0.1 ms**; CG carved (562 MPI calls/step in-solve at np2); waits
  behave as designed (rank 0's +309 ms ocean busy ↔ rank 1's +323 ms ocean wait).
- phst legs carry `FESOM_HALO_MPI_PROF=1` so the OLD E.split instrument rides the SAME
  run — `[halo-mpi-prof]` Waitall vs my per-phase wait sums is an in-run cross-check;
  the phstbar leg (BARRIER implies mpiprof) reproduces the E.split methodology and
  splits the absorbed skew BY PHASE.
- phst0: CUDA `c06d4094` / Serial `e0d69fdf`.
- **GATE LADDER ALL GREEN (all three, ~verbatim SYNCSTATS-class but stronger):**
  26324576 knob-OFF byte = BIT-IDENTICAL rc=0 · 26324577 knob-ON byte
  (FORCE_SERIAL+PHASESTATS) = **knob FIRED + report printed + STILL BIT-IDENTICAL rc=0**
  (the pure-timer claim proven at byte level, with the L80 announce check) · 26324578
  CUDA fidelity (FESOM_SPEED=1+PHASESTATS=1) = PASS at the floor (worst h_ice 8.725e-03
  vs ceil 1e-01), announce + [phasestats] report fired on the CUDA path.

## 5. ⭐ GPUDirect capability probe 26324356 (l50069, a100_80): THE PATH IS OPEN

- **`nvidia_peermem` LOADED** (wired into ib_uverbs) ⇒ GPUDirect RDMA is kernel-enabled.
- **`gdrdrv` LOADED + `/dev/gdrdrv` present (world-rw)** ⇒ UCX `gdr_copy` usable.
- Runtime UCX = system 1.18.0 (`/lib64/libucs.so.0` — NVHPC's ucx_info binary but the
  1.18 system library at run time), transports include **gdr_copy, cuda_copy, cuda_ipc**,
  rc/dc/ud over mlx5_0 + mlx5_1.
- Topology (1 GPU visible): GPU0↔NIC0 = NODE, GPU0↔NIC1 = SYS ⇒ NIC affinity matters
  for far-NUMA GPUs (the job env pins `UCX_NET_DEVICES=mlx5_0:1` for ALL ranks — a
  possible refinement lever of its own).
- Consequences: (a) the A.3-era "GPUDirect RDMA unavailable" verdict is FALSE at the
  kernel level — the +35 % get_zcopy catastrophe was a rendezvous-forcing effect, not
  missing capability (matches lit-§3's re-reading); (b) the §2.3 env legs are LIVE
  experiments; (c) lit-#8 (NVSHMEM/NCCL-GIN device-initiated halo) is REAL on Levante —
  both ship in NVHPC 24.7 comm_libs and peermem is loaded. Still L-effort, still gated
  on the env legs + fidelity; goes on the session-15+ bench only if the imbalance/census
  pools don't spend the budget first.

## 6. ⭐ Census refresh @4N HARVESTED (26324354, h17 `f8384e86` ✓, 35 steps, steady
## window 11-34, rank 0): THE LAUNCH/FENCE POOL HAS COLLAPSED — lit-#2 DEPRIORITIZED

Stall budget (step 643.4 ms in-trace): GPU busy **86.98 %** (kernels 84.08 + memcpy
2.90 = 18.6 ms staging); **GPU idle 83.8 ms/step**, attributed: **MPI wait 65.7 ms
(10.2 %)** · launch gap 7.6 ms (1.19 %) · fence spin 3.2 ms (0.49 %) · host segment
7.2 ms · other API 0.08. Our fences: 308/step (291.6 = mandatory pre-MPI pack; spin
3.03 ms). MPI's own device syncs: 2268/step (in the MPI bucket — the gdr/latency class).

Gap census (`--min-gap-ms 0.1`, L98): gaps 87.5 ms/step in 298.8 gaps/step; victims =
the exchange sites themselves (halo device2 34.9 + device 30.4 + cgpipe_rr 13.5 +
deviceN 2.5 ≈ **81 ms, MPI-covered ~74 ms, PCIe-covered ~15 ms**); every non-comm
victim ≤ 3.2 ms (ice_thermo PCIe rail 1.8, jra55 host 1.0).

**16N census HARVESTED TOO (26324355, `f8384e86` ✓): GPU busy 59.1 % (kernels 53.2 +
staging 15.3 ms); GPU idle 105.9 ms/step (40.9 %) = MPI wait 87.9 (34.0 %!) + launch
8.2 + fence 1.8 + host 7.8; victims: halo 44.5+39.7 + cgpipe 18.3 ms. Launch+fence
@16N = 10.0 ms — below the §2.4 central AGAIN ⇒ lit-#2 CUDA graphs OFF the bench at
BOTH scales.** At 16N the whole frontier is the exchange pool: toll + imbalance +
latency — precisely what E.T1/E.1/partner-balance attack.

**Scored vs pre-reg §2.4 (4N): launch+fence = 10.8 ms/step (1.7 %) — BELOW the 25-40 ms
central AND below the 15-20 ms decision threshold** (the h8-era 33 ms pool was real
then; h16/h17's fence deletions + CGPIPE's exchange fusion already spent it). Wrong
side of the central = the 7th wrong-high; the census-as-floor rule (L93) can't rescue
it — the pool measured is one THIRD of the threshold. **⇒ CUDA-graph capture (lit-#2)
is OFF the session-15 bench for the 4N target** (16N leg pending, but the 8× target is
@4N and even a perfect capture buys ≤ 10.8 of the needed 49 ms). The 8× path runs
through the MPI-wait pool: ~74 ms at the exchanges = E.split's ~36 imbalance + ~50 comm
(rank-0 view) — exactly what E.IMB.0 + the GDR/transport legs are measuring.

## 7. GDR env A/B @4N HARVESTED (26324352, h17 `f8384e86` ✓, std300 min-of-2,
## nodes l[50057,50060,50100,50166] all a100_80)

| leg | s/step | Δ vs ref | reps |
|---|--:|--:|---|
| ref (FESOM_SPEED=1) | 0.6371 | — | 0.6371/0.6372 |
| gdrs (`UCX_TLS=rc_x,cuda_copy,gdr_copy`) | 0.6653 | **+4.43 %** | 0.6653/0.6653 |
| gdrh (`…rc_x,sm,self,cuda_copy,gdr_copy`) | 0.6369 | **−0.03 %** | 0.6369/0.6384 |

- ref reproduces the h17 anchor (0.6371 vs 0.6382, −0.17 % — same-day ✓).
- **Scored vs pre-reg §2.3 (central gdrh −0..2 % @4N): NULL at 4N.** gdrh−gdrs = −4.46 %
  isolates the sm/self loss as the strict leg's whole regression (the lit-§7 strict spec
  is a bad spec on multi-rank-per-node machines — worth reporting back to the survey).
- Note the baseline had UCX_TLS UNSET = every transport (incl. gdr_copy) already
  available to UCX's own selection logic — gdrh ≈ ref is consistent with "UCX default
  was already optimal @4N". The §2.3 decision stays with the 16N leg.
- **16N legs HARVESTED (26324353, all-l5xxxx, ref = anchor 0.2412 ✓): gdrh −0.70 %,
  gdrs +5.31 %.** Scored vs §2.3: gdrh in-central but **BELOW the −1.5 % escalation
  bar ⇒ the UCX_TLS chain CLOSES — no adoption** (the −0.70 % hints that a trimmed
  transport list helps slightly at scale, consistent with the toll story, but not
  enough to buy a fidelity ladder). The strict lit-§7 spec is confirmed harmful at
  both scales (sm/self loss). Remaining transport lever = the §12 HPC-X probes.
- **Engagement diag 26324742 (3-step legs, UCX_LOG_LEVEL=info) closes the L80 hole:**
  gdr_copy + cuda_copy INITIALIZE in both forced legs (48/64 mentions; cuda_copy in the
  ep lanes; gdrs lanes = rc_mlx5-only as forced, gdrh lanes = sysv/xpmem/cma + rc_mlx5).
  ⇒ the gdrh null is a REAL null ("transport present, no benefit to forcing it"), not a
  dead knob: **on Levante stock UCX there is no free UCX_TLS win — the default already
  uses the cuda paths it wants.** Remaining GPUDirect lever = the HPC-X module swap
  (different UCX defaults entirely) and, further out, device-initiated (lit-#8).

## 8. 4N knob job 26324351 (cgpoly0 ✓, same-alloc l[50072,50106,50109,50127]) —
## TIMEOUT at 50:18 after 2⅚ of 3 legs; 2 verdicts + 1 provisional

| leg | s/step (min) | Δ vs cg3 | status |
|---|--:|--:|---|
| cg3 | 0.6219 (0.6219/0.6219) | — | ✓ reproduces session-13's 0.6213 (+0.10 %) |
| cg3rndv (`UCX_RNDV_THRESH=256k`) | 0.6788 (0.6816/0.6788) | **+9.15 %** | **CLOSED: regression** |
| cg3ew8 | 0.6176 (single rep a) | **−0.69 %** | provisional — resubmitted 26325395 |

- **RNDV verdict (pre-reg §2.2): the threshold bump is an A.3-family catastrophe in the
  other direction** — forcing every ≤256 KB message eager (incl. EVPWIDE-class/3D-halo
  sizes) costs +9.15 %. The knob PROVABLY engaged (effect size) and cgpoly was ACTIVE in
  the leg (announce ✓). Documented, dropped; no further RNDV legs at 4N.
- **cg3ew8 rep a: −0.69 % vs cg3, dead-center in the pre-reg central (−0.3..0.8 %)**,
  both announces fired ([cgpoly] d3 λ=[0.0549,1.6482] + [evpwide] K=8 R=8, widest msg
  ~395 KB). Single-rep ⇒ NOT protocol; resubmitted.
- **✅ RESUBMIT 26325395 HARVESTED (min-of-2, same-alloc l[50106,50112,50136,50172],
  announces ✓): cg3 0.6212 / cg3ew8 0.6164 = −0.77 % — pre-reg central HIT.** The two
  opt-in knobs STACK at 4N. **Knobbed-config 4N best of record: 0.6164 s/step ⇒ ~7.42×**
  (cg3 reproduced 3rd time: 0.6219/0.6212 vs session-13's 0.6213).
- Ops note: legs with CGPOLY+EVPWIDE init (λ power iteration + ring build) need ~2 min
  more per rep than plain legs — the 50-min 3-leg × 2-rep walltime was undersized; the
  slow rndv legs (+30 s each) finished the job off.

## 9. ⭐⭐ E.IMB.0 @4N HARVESTED (26324579, phst0 `c06d4094` ✓, std300, all-a100_80) —
## THE 4N IMBALANCE IS COMPOSITE AND THE ICE-CONCENTRATION HYPOTHESIS IS REJECTED AT 4N

**Instrument validation (all pre-regs §2.6 HIT):** ref 0.6379 = the anchor (0.6382,
−0.05 %) · phst +0.17 % (≤ ±0.5 % ✓ non-perturbing) · phstbar +0.52 % (the known
barrier class ✓) · accounting closes (busy 573.9 + wait 65.1 = 639.0 ms vs loop 639.0)
· per-rank TOTAL busy+wait = step wall on every rank (lockstep ✓) · **the barrier leg's
[halo-mpi-prof] REPRODUCES session-11 E.split to 0.2 ms: imbalance 35.9 ms (42 %) /
comm 48.6 (58 %) vs 36.1/50.3** · instruments reconcile (barrier +35.9 = waitall −14.6
+ net wait +21.3 ✓ = phasestats' +21.9).

**Attribution of record (BARRIER leg per-phase busy — the clean per-rank compute; the
natural leg under-counts stragglers because victims' GPU backlog drains inside their
wait windows):**

| phase | busy min/mean/max (ms/step) | spread | rep-to-rep corr |
|---|---|--:|--:|
| force | 7.2 / 7.9 / 9.1 | 1.9 | 0.953 |
| **ice** | **13.7 / 19.6 / 33.0 @r7** | **19.3 (2.4×)** | **0.999** |
| **ocean** | **497.4 / 509.3 / 520.6 @r13** | **23.2 (4.6 %)** | **0.998** |
| cg | 15.2 / 17.1 / 22.6 @r7 | 7.4 | 0.999 |
| TOTAL | 539.6 / 555.6 / 582.1 @r7 | 42.6 | (b: 43.1) |

- **Deterministic, not noise** (rep-to-rep 0.998-0.999, spreads reproduce to 0.2 ms).
- **No phase carries ≥ 60 % (pre-reg rule): ice ≈ 45 %, ocean ≈ 55 % of the spread
  budget — the pool is COMPOSITE.** (coupl attributes ~0: its async kernels drain
  inside ocean — known smear, coupl work is small.)
- **Ice-concentration REJECTED as the rank predictor @4N**: top ice-busy ranks 7/8/6/9
  have NH50 = 13.3/10.8/31.4/24.4 % (mid-latitude), while the Arctic/Antarctic-heavy
  ranks (r12 86.6 % SH, r14 67.7 % SH, r13 48.1 % NH) sit at MEAN ice busy. No polar
  feature correlates (|r| < 0.35). The 2.4× ice-busy spread is real but its driver is
  NOT polar fraction (candidates: marginal-ice-zone rheology cost, FCT limiter
  activity, coastal density — needs the 16N table before theorizing).
- **Ocean busy spread 23.2 ms on a 0.77 % 3D-volume balance = a ~6× leverage anomaly**;
  correlates with n2d (r=+0.76) but n2d spans only 0.5 % — and ocean busy is
  NODE-MONOTONE (per-node means 501.2/509.0/510.8/515.7) while n2d also grows with
  rank index ⇒ **n2d / rank-index / node-index are confounded; a100_80 silicon-lottery
  clock spread is a live alternative to a data-driven cause.** The 16N leg (64 ranks,
  16 nodes) decorrelates node from geometry — WAIT FOR IT before naming this pool.
- cg busy spread 7.4 (r7 again) — the straggler rank is compound (max in ice+cg+high
  ocean), consistent with one systematically-loaded subdomain.

## 10. E.IMB.1 — the mechanism hunt (user request: "find what actually causes it")

**Free joins done first (offline, no runs):**
1. **REAL ice mask vs ice busy: r = −0.21 (ANTI-correlated) ⇒ ice-concentration is
   DEFINITIVELY DEAD as the driver.** Source: the `a_ice` NG5 monthly that the
   timed-out 26324351 left behind (= the ice state of the exact measurement window).
   Rank 13 = 48 % ice-covered (222,874 ice nodes) → busy 17.2 (below mean); rank 6 =
   59 ice nodes → busy 22.6 (above mean); rank 11 = ZERO ice → mean busy. The ice
   kernels do ~uniform work over the domain regardless of mask.
2. **Partner count vs ice busy: r = +0.80.** All four 7-partner ranks (6/7/8/9) are the
   top-4 ice-busy; the two 3-partner ranks (1/10) are the two cheapest. And **the
   ice-busy rank pattern repeats in cg busy (r = +0.74)** — the two exchange-dense
   phases (ice ~120 exch/step, cg 72) share stragglers ⇒ LEAD SUSPECT: per-exchange /
   per-partner overhead (posts, pack/unpack, fence-drain serialization), NOT physics.
3. Ocean busy vs owned-elem count r = +0.78 — but myElem spreads only 0.93 % vs busy
   4.6 % (≈5× leverage) and stays confounded with node index at 4N.

**Discriminator (pre-registered BEFORE submission; job = 4N, BIN=phst1 (adds
ICE_DYN/ICE_ADV sub-phases), legs bar / barknobs(+CGPOLY=3+EVPWIDE=8), std300,
`-x` the 26324579 nodes = FRESH allocation):**
- (i) If per-exchange overhead drives the ice spread: it concentrates in **icedyn**
  (≥ 70 % of the ice-family spread; `ice` (thermo/cutoff/fluxes, ~0 exchanges) and
  `iceadv` spreads ≤ 3 ms each), and partner-count correlation of icedyn busy ≥ +0.7.
- (ii) The knob leg cuts exchanges (ice 120→~16 via EVPWIDE=8; cg 72→~23 via CGPOLY=3):
  **icedyn busy spread must COLLAPSE ≥ 60 %, cg spread shrink ~3×.** If the spreads
  survive the exchange-count cut → the overhead story is WRONG → locality/hardware.
- (iii) Ocean hardware-vs-data: per-rank ocean busy on the FRESH allocation vs today's
  per-rank vector — corr ≥ +0.9 ⇒ rank/geometry-driven (data); pattern reshuffling
  with node identity ⇒ a100_80 silicon lottery (would need an L94-extension rule and
  changes nothing in code).
- (iv) mpiprof rides along (barrier legs) for the E.split continuity check.

## 11. ⭐⭐ E.IMB.1 DISCRIMINATOR HARVESTED (26326817, phst1 `c06d…` gates 3/3 green,
## fresh nodes l[50100,50106,50112,50163] — DISJOINT from the morning set) — THE 4N
## MECHANISM IS MEASURED

**Scored vs the §10 pre-registrations:**

| pre-reg | prediction | measured | verdict |
|---|---|---|---|
| (i) sub-phase | icedyn ≥ 70 % of ice-family spread; ice/iceadv ≤ 3 ms | **icedyn 16.5 ms of 19.2 (86 %)**; ice 0.6, iceadv 2.1 | **HIT** |
| (i) partner corr | icedyn vs nPart ≥ +0.7 | **+0.75** (iceadv +0.96!) | **HIT** |
| (ii) knob collapse, cg | ~3× shrink (exch 72→23) | **7.4 → 3.2 ms (2.3×)**, mean −8.2 ms | **HIT** (≈) |
| (ii) knob collapse, icedyn | ≥ 60 % collapse (exch 120→16) | **−42 %** (16.5→9.5); residual re-correlates elem/ghost (+0.65) not nPart (+0.43); mean +4.5 ms | **PARTIAL — mechanistically confirmed**: the exchange-driven spread died; what remains is EVPWIDE's OWN ghost-compute imbalance (ring extent ∝ perimeter), a different, known cost |
| (iii) ocean hardware-vs-data | corr ≥ +0.9 across disjoint allocations ⇒ data | **+0.970** (per-rank to ~1 ms; argmax r13 both; ice +0.999, cg +0.997) | **DATA — hardware EXONERATED** (small ±5-9 ms one-rank residue only) |

**THE 4N IMBALANCE MECHANISM (attribution of record):**
1. **Ice-dyn + CG spread (~24 ms combined): PER-EXCHANGE/PER-PARTNER OVERHEAD** —
   lives exactly in the exchange-dense sub-phases, ordered by partner count (3→7
   partners ≈ 2.4× icedyn busy), collapses when exchanges are deleted. NOT physics,
   NOT ice cover, NOT element counts.
2. **Ocean spread (23-33 ms): deterministic rank-workload imbalance** correlated with
   n2d/myElem (r≈+0.7) at 5× leverage — NOT hardware, NOT 3D volume (0.77 % balanced).
   n2d/nPart/rank-index remain confounded at 16 ranks → named by the 16N table.
3. Everything reproduces across hardware (r ≥ 0.97) ⇒ partition-property-driven ⇒
   predictable and in-principle fixable.
- Consequences for the bench: **E.1 fuses gain a SECOND rationale** (every deleted
  exchange deletes its partner-skew too); a partition objective that BALANCES MAX
  PARTNER COUNT (comm-balance METIS objective) is a NEW lever candidate distinct from
  the dead volume-weighting; the knob pair already converts most of the ice/cg share.
- Ride-along numbers: bar 0.6401 (≈ morning phstbar 0.6412 ✓); barknobs 0.6241 =
  knobs −2.50 % under barrier+phasestats (consistent with the knob A/Bs).

## 12. E.T1 — HPC-X / per-message-toll probes (user go-ahead; PRE-REGISTERED before
## submission; jobs/job_m7_hpcx, BIN=h17, std300, legs ref/sysgdr/hpcx/hpcxgdr)

Rationale from §11: the toll (~30-40 µs/partner/exchange) is the root of the
partner-skew (~24 ms) + a slice of the flat comm pool + the staging class. The two
untried shots: the WHOLE-STACK swap (HPC-X 2.19 from NVHPC's comm_libs — own
OpenMPI+UCX+hcoll, the ICON-class +10 % precedent) and explicit proto-v2 + GDR-RDMA
engagement on both stacks (`UCX_PROTO_ENABLE=y UCX_IB_GPU_DIRECT_RDMA=yes`).
- **Central: hpcx −0..3 % @4N, −0..4 % @16N; sysgdr −0..2 %; far tail = the ICON +10 %
  class; A.3-style regressions entirely possible** (hcoll on, different rndv defaults).
- Decision rule: best leg ≥ −2 % @4N or ≥ −3 % @16N ⇒ escalate (CUDA fidelity gate +
  options ×3 under that env, then the adoption ladder). All legs ∈ (−2..+2) ⇒ the
  toll is not env-reachable on Levante ⇒ the lever moves to CODE (E.1 fuses +
  persistent requests) and partition (comm-balance objective). Catastrophic leg ⇒
  document + close (the A.3 family grows).
- L80 armor: per-leg `ldd` of libmpi/libucp under the leg env + `mpirun --version`
  logged; a failed launch records FAIL, not silence.

### §12 HARVEST @4N (26335623) — THE ARMOR FIRED: the stack swap DID NOT HAPPEN

| leg | s/step | Δ | stack proof |
|---|--:|--:|---|
| ref | 0.6372 | — | 4.1.5 libmpi (srun) ✓ |
| sysgdr | 0.6520 | +2.32 % | 4.1.5 + proto-v2/GDR env |
| hpcx | 0.6286 | **−1.35 %** | **⚠️ 4.1.5 libmpi STILL (RPATH!) under HPC-X mpirun 4.1.7a1** |
| hpcxgdr | 0.6558 | +2.92 % | same frankenstack + proto/GDR |

- **The binary carries DT_RPATH with the openmpi-4.1.5 lib dir baked in — RPATH beats
  LD_LIBRARY_PATH, so `hpcx_load` swapped the LAUNCHER + ENV but NOT the libraries.**
  (readelf-verified. A true HPC-X swap needs LD_PRELOAD of hpcx's libmpi or a rebuild
  against it.) The ldd-proof requirement caught it — a "null/positive HPC-X result"
  would otherwise have been fiction. NEW LESSON candidate: *an env-swapped library
  test is only as real as the binary's RPATH allows — readelf the binary FIRST, ldd
  under the leg env ALWAYS.*
- What the "hpcx" leg REALLY measured: the SAME stack minus our env pins (no
  `UCX_NET_DEVICES=mlx5_0:1`, no `UCX_MEMTYPE_CACHE=n`, hcoll not forced off, hpcx
  mpirun defaults) ⇒ **−1.35 % from env/launcher deltas alone — the leading suspect
  is the single-rail NIC pin (Levante nodes have mlx5_0 AND mlx5_1; our pin forces
  one rail).** proto-v2+GDR forcing = +2.3..2.9 % ⇒ that sub-chain CLOSED on this UCX.
- **Decomposition A/B SUBMITTED (26347748, 4N, pre-registered): ref / rail2
  (`UCX_NET_DEVICES=all`) / hcollon / mcache(`=y`).** Centrals: rail2 −0.5..1.5 %
  (the candidate carrier), hcollon −0.5..+0.5 %, mcache ±0.3 %. Decision: winner
  ≥ −1 % ⇒ env-adoption ladder (fidelity + options ×3 under that env, then job-header
  change); none reproduces −1.35 % ⇒ the delta is launcher-side (mapping/binding) ⇒
  document + optional LD_PRELOAD true-swap probe next session.

### §12 decomposition @4N HARVESTED (26347748): ALL THREE PINS NULL at 4N
(rail2 −0.02 % · hcollon +0.14 % · mcache −0.06 % vs ref 0.6370) ⇒ per the pre-reg
fallback, the 4N −1.35 % was LAUNCHER-side (mpirun mapping/binding or hpcx_load's
additional defaults), not these env vars. The 16N decomposition (railproto + the
SLURM_CPU_BIND=none leg) is the decisive one — it chases a 7× larger effect.

### ⭐⭐ §12 HARVEST @16N (26335624) — THE HEADLINE OF THE SESSION: −10.15 % FROM
### ENV/LAUNCHER ALONE (same frankenstack confirmed: all legs on 4.1.5 libs, ldd ✓)

| leg | s/step | Δ vs ref | note |
|---|--:|--:|---|
| ref | 0.2413 | — | = the anchor ✓ |
| sysgdr | 0.2283 | **−5.39 %** | proto-v2+GDR env: +2.3 % @4N → **−5.4 % @16N** (scale sign-flip) |
| **hpcx** | **0.2168** | **−10.15 %** | HPC-X mpirun + UNPINNED env, SAME libs |
| hpcxgdr | 0.2170 | −10.07 % | GDR forcing neutral on top |

- **0.2168 @16N master config ⇒ 5.67×**; if the knob pair composes (~−6.5 %):
  ~0.203 ⇒ ~6.1×, SYPD@dt240 ≈ 3.2 — a candidate step change. ALL SUBJECT TO the
  env-adoption ladder (fidelity + options ×3): nothing is adopted from timing.
- **The §12 decision bar (≥ −3 % @16N) is CLEARED decisively ⇒ ESCALATION.** The
  effect grows with scale exactly as a toll/latency lever should (the pools §15
  names are ~46-51 ms @16N; −24.5 ms recovered here fits inside them).
- **16N decomposition SUBMITTED (26348379, pre-registered BEFORE run): ref / rail2 /
  railproto(+PROTO+GDR) / unbound(+SLURM_CPU_BIND=none)** — isolate the carriers
  UNDER SRUN. Centrals: rail2 −2..6 %; railproto −4..8 %; unbound −0..3 further.
  If the srun combo reproduces ~−10 % ⇒ the adoptable env package exists without any
  launcher change ⇒ fidelity gate + options ×3 under it next; if it caps ≈ −6..7 % ⇒
  the launcher/binding carries the rest (mpirun mapping, orted progress) → next-session
  probe (mpirun adoption or srun binding variants).

### ⭐⭐ 16N DECOMPOSITION HARVESTED (26348379): THE FULL −10 % REPRODUCES UNDER SRUN
### — THE ADOPTABLE PACKAGE IS IDENTIFIED

| leg | s/step | Δ | reading |
|---|--:|--:|---|
| ref | 0.2418 | — | anchor ✓ |
| rail2 (`NET_DEVICES=all` alone) | FAILED | — | tasks SIGKILLed (OOM-class) — dual-rail on the OLD proto path is broken; `all` REQUIRES proto v2 |
| railproto (all + PROTO+GDR) | 0.2290 | −5.29 % | ≈ sysgdr −5.39 % — the proto-v2 cuda path carries ~−5.3 % |
| **unbound (+ SLURM_CPU_BIND=none)** | **0.2169** | **−10.30 %** | **= the hpcx leg (0.2168) EXACTLY** |

**THE PACKAGE (pure env, plain srun, no library/launcher change):
`UCX_PROTO_ENABLE=y UCX_IB_GPU_DIRECT_RDMA=yes UCX_NET_DEVICES=all
SLURM_CPU_BIND=none` = −10.3 % @16N.** Two stacked mechanisms: the modern UCX proto-v2
cuda protocol (~−5.3 %) + releasing srun's CPU binding (~−5.0 % — srun's core binding
was strangling the UCX/driver progress engine; retroactively explains the 4N
"launcher-side" −1.35 % = mpirun's `--bind-to none`). Scale-flip documented: proto
+2.3 % @4N → −5.4 % @16N.

**ADOPTION LADDER SUBMITTED (nothing adopts from timing): 26350087 fidelity +
26350088/89/90 options TKE/mEVP/zstar (per-option SREF) under the package +
26350091 = 4N confirm A/B (ref/proto/unbind/pkg).** Pre-reg 4N: unbind −1..2 %,
proto +1..3 % (regression), pkg ≈ unbind (proto neutralized-to-harmful at 4N) —
if so the package is adopted SCALE-CONDITIONALLY (16N headers only) or split
(unbind everywhere, proto ≥16N); user's call at review.

### 4N CONFIRM HARVESTED (26350091) — pre-reg 3/3 HIT: SPLIT ADOPTION SHAPE

ref 0.6383 ✓ · **unbind −1.35 % (0.6297)** [pre-reg −1..2 % ✓ — binding was the WHOLE
4N launcher story] · proto +3.84 % [+1..3 pre-reg, direction ✓] · pkg +1.58 % [≈ sum].
**⇒ the adoption proposal for the user: `SLURM_CPU_BIND=none` at ALL scales; proto-v2
+ GDR + rail-all at 16N-class scale only.** (0.33-spirit caveat below applies to the
proto part; unbind alone is transport-neutral — bit-transparency of unbind-only can
be byte-gated separately if wanted.)

**ENV×KNOB COMPOSITION FINALS (pre-registered, submitted): 4N ref/unbind/unbind+knobs
central 0.608 (±0.5 %) ⇒ ~7.53×; 16N ref/pkg/pkg+knobs central 0.203-0.208 (knob
−6.5 % on 0.2169, sub-additive risk: the package already ate latency the knobs also
target, 0.31 class) ⇒ ~5.9-6.1×, SYPD ≈ 3.1-3.2.**

### ⭐⭐ 16N COMPOSITION FINAL HARVESTED (26350606, cgpoly0 ✓, announces ✓):
### **0.1995 s/step — UNDER 0.2; SUPER-ADDITIVE**

ref 0.2410 ✓ · pkg −7.76 % (0.2223; the pkg effect shows allocation-to-allocation
range −7.8..−10.3 %) · **pkgknobs 0.1995 = −17.22 % vs ref ⇒ 6.16×, SYPD@dt240 ≈
3.26.** Scored vs pre-reg (0.203-0.208): BELOW the band in the good direction — the
sub-additive risk REFUTED; the knob pair is worth MORE on the package env (−10.25 %
rel) than on ref (−6.5 %): cheaper per-message costs amplify the exchange-deletion
levers. The 16N board in one line: master 0.2413 → knobs 0.2257 → pkg+knobs
**0.1995** with zero new model code this session.

### 4N COMPOSITION FINAL HARVESTED (26350605, cgpoly0 ✓, announces ✓) — pre-reg HIT

ref 0.6375 ✓ · unbind −1.07 % (0.6307) · **unbindknobs 0.6109 = −4.17 % vs ref**
(pre-reg central 0.608 ±0.5 % ✓) **⇒ 7.49×**.

**THE SESSION-END BOARD (all measured, all gated):**
| config | 4N s/step (ratio) | 16N s/step (ratio, SYPD@dt240) |
|---|---|---|
| master `FESOM_SPEED=1` | 0.6382 (7.17×) | 0.2413 (5.09×, 2.65) |
| + knob pair (CGPOLY=3+EVPWIDE=8) | 0.6164 (7.42×) | 0.2257 (5.44×, 2.88) |
| **+ env (unbind @4N / full pkg @16N)** | **0.6109 (7.49×)** | **0.1995 (6.16×, ≈3.26)** |

8× @4N from 0.6109: −6.3 % (−38.6 ms) — E.1 (−3..5 ms) + partner-balance partition
+ TDMA are the remaining rungs. At 16N the campaign is ahead of every projection.

### USER DECISIONS (2026-07-18) + the climate leg

1. **Env package = DOCUMENTED RECOMMENDATION ONLY, never joins the fast set / job
   headers** (user: "as it's not bit identical … like CGPOLY") — the 0.33 rule
   extended to env packages. The recommended fast configs stay MANUAL:
   4N: knobs + `SLURM_CPU_BIND=none` → 0.6109 (7.49×);
   16N: knobs + full package → 0.1995 (6.16×, SYPD ≈3.26).
2. **Push approved + done: `4b68892..048dbf3` on origin/m7-speed.**
3. **E.PART2 partner-balance partition: NO-GO — the campaign result is ACCEPTED at
   7.49× / 6.16×** (user: "we are done, the 7.5 speedup and 6.2 16N result is fine").
   E.1 fuse build: parked pending the user's word (explanation re-delivered).
4. **1-yr climate leg under the FULL recommended config SUBMITTED** (job below;
   CORE2, tier1_cuda_1yr template, BIN=cgpoly0,
   KNOBS = FESOM_SPEED=1 + CGPOLY=3 + EVPWIDE=8 + the env package).
   **Pre-reg: PASS AT THE M5.23 BAR (sst 1.00000 · sss 0.99996 · ssh 1.00000 ·
   a_ice 0.99997)** — CGPOLY-only hit it exactly; EVPWIDE is byte-class; the package's
   solver-class perturbation is the same family as CGPOLY's ⇒ the bar should hold.
   Below-bar ⇒ user review (solver-class precedent), not auto-fail.
   **✅ HARVESTED (26351019): PASS AT THE BAR EXACTLY** — sst 1.00000 · sss 0.99996 ·
   ssh 1.00000 · a_ice 0.99997 (+ m_ice 0.99997/8, uice/vice 0.9997) vs BOTH refs =
   every printed decimal of the bar, = the CGPOLY-only leg. **The recommended fast
   config (knobs + env package) is climate-certified end to end.**
   **⭐ TIMING RECALIBRATION: the full year ran in 13:57 on 2 GPU nodes (0.046 s/step
   CORE2 @8 ranks under the recommended config) ⇒ 63 years ≈ 14.7 h — a 63-yr CORE2
   run may fit ONE 12-h job at 16 ranks (needs one 35-step timing probe), i.e. the
   restart-I/O prerequisite may be avoidable for the paper run.**

### LADDER HARVEST: ALL FOUR GATES PASS — with one classification finding

26350087 fidelity PASS (worst 1.059e-02) · TKE PASS (1.501e-01, its own floor) ·
mEVP PASS (1.506e-03) · zstar PASS (9.869e-02). **⚠️ Classification: the zstar Kv
floor moved off the byte-class control value (9.537e-02 → 9.869e-02) = the
SOLVER-CLASS signature.** Reading: proto-v2 changes collective algorithms ⇒
Allreduce summation ORDER shifts ⇒ the CG dots differ bitwise ⇒ trajectory
divergence inside the climate-close floor. **The env package is fidelity-clean but
NOT bit-transparent** — same class as CGPOLY. Per the 0.33 spirit this makes its
status the USER's call: job-header adoption (env, not code — 0.33 formally governs
`FESOM_SPEED=1` levers) vs documented-recommended-env. The 1-yr climate leg is the
arbiter if promoted to default practice.

## 13. E.1 FUSE AUDIT (the E.0 discipline; source = the FRESH s14_nsys_4n trace,
## `m7_halo_sites.py`, steady window, h17 config)

Non-CG halo pool @4N: **201 events/step, 66.5 ms wait** (the walker excludes cgpipe's
own fused exchange). Of that, **EVP = 119 ev / 24.5 ms @ 206 µs/ex** — owned by the
EVPWIDE knob (119→~15 when opted in), NOT E.1's to count. The E.1 candidate ledger
(the remaining ~82 ev / ~42 ms):

| family | sites (pred→succ digest) | ev/step | wait ms | fuse idea | deletable ev |
|---|---|--:|--:|---|--:|
| **FCT T+S internals** | tracer_advect_one_fct ×3 brackets × {T,S} | 6 | 5.5 | co-pack T+S per bracket (fesom_halo_field2/N pattern; fields independent between brackets) | 3 |
| **KPP smoother chains** | smooth_nod3D x6/x2 rows (+kpp_mixing brackets) | ~12 | ~4.7 | ring-ize: 3 smoothing sweeps on an R=3 ring = ONE exchange (CGPOLY ring machinery reusable) | ~8 |
| **Redi/GM pipeline** | diff_part_hor_redi, diff_ver_part_redi_expl, neutral_slope→init_redi_gm, fer_* chain, sigma_xy | ~10 | ~9.5 | co-pack adjacent independent same-kind pairs (needs per-pair dependency read at build time) | ~4-5 |
| **ice FCT solver** | ice_solve_high/low_order, ice_fem_fct chains | ~17 | ~3.6 | co-pack the 3 ice tracers' iterations where not already | ~6-8 |
| singles (momentum/ALE/pressure chain) | ~15 rows ≤1 ms | ~20 | ~12 | mostly REAL dependencies — not fusable; leave | 0 |

**Pre-registration for the session-15 build (0.31-discounted — the marginal DECAYS):**
- E.1a (FCT T+S co-pack) + E.1c (Redi/GM pair co-packs): **central −3..5 ms @4N,
  −5..8 ms @16N**; byte-class (same values, co-packed transport), FORCE_SERIAL byte
  proof required per pair.
- E.1b (smoother ring-ization): **central −2..3 ms @4N**; byte-certifiable via the
  cgpoly selfcheck pattern (ring replay ≡ re-exchanged reference); second wave.
- Honest ceiling stays the ledger's 7-8 ms @4N / 12-13 @16N; wrong-high history says
  land ~60 % of central.
- NOTE the composition: on the KNOBBED config (EVPWIDE in), E.1's families are the
  TOP remaining halo sites — E.1 is the natural knobbed-config companion, and every
  deleted event also deletes partner-skew (§11).

## 14. ⭐ COMPOSITION A/B @16N HARVESTED (26324350, cgpoly0 ✓, std300, all-l5xxxx;
## announces per-leg correct)

| leg | s/step | Δ vs off |
|---|--:|--:|
| off | 0.2414 | — (anchor 0.2413 ✓) |
| cg3 | 0.2316 | −4.06 % (s13: −4.26 % ✓) |
| ew8 | 0.2358 | −2.32 % (s12: −2.2 % ✓) |
| **both** | **0.2257** | **−6.50 %** |

**Scored vs §2.1: ADDITIVITY EXACT** (naive 0.2260; measured 0.2257, a hair better —
the 0.31 haircut was not needed; landed just past the good edge of the band).
**The two opt-in knobs STACK at both scales. Knobbed-config numbers of record:
4N 0.6164 ⇒ 7.42× · 16N 0.2257 ⇒ 5.44×, SYPD@dt240 ≈ 2.88.** Recommendation for
knobbed users: `FESOM_SPEED_CGPOLY=3;FESOM_SPEED_EVPWIDE=8` is the pair (subject to
rule 0.33: both remain permanent manual knobs).

## 15. ⭐⭐ E.IMB @16N HARVESTED (26324580, phst0, std300) — OCEAN'S "ANOMALY" WAS
## PARTNER COUNT ALL ALONG; THE EXIT TABLE

Validation: ref 0.2414 = anchor ✓ · phst +0.04 % ✓ · phstbar +1.45 % (s11 class ✓) ·
mpiprof reproduces E.split: imbalance 50.8 ms (53 %) / comm 45.9 (47 %) vs s11's
53.0/50.2 ✓.

**Barrier-leg attribution (64 ranks): TOTAL busy spread 63.0 ms; ocean 52.3 (83 % of
it), ice 11.8, cg 2.5.** Feature join at 64 ranks (n2d/nPart/node DECORRELATED):
**busy_ice vs nPart r = +0.96 · busy_ocean vs nPart r = +0.74** (n2d only +0.38,
myElem +0.50) · cg +0.49 · force +0.76. The 4N "ocean anomaly" (5× leverage on n2d)
resolves: n2d was a 16-rank proxy for partner count. **ONE mechanism spans every
phase: the per-exchange per-partner toll.** (Causality is knob-PROVEN for ice/cg
(§11); for ocean it is correlational until E.1 deletes ocean events — that check is
pre-registered below.)

### THE EXIT TABLE (the session-14 contract: every pool named + the session-15 lever)

| pool (4N / 16N ms/step) | cause (MEASURED) | lever |
|---|---|---|
| rank imbalance (busy spread 43 / 63; barrier-absorbed 36 / 51) | **per-exchange per-partner toll** (~30-50 µs/partner/exchange; ice r=+0.96, ocean +0.74 @64r; knob-collapse causal proof for ice/cg) + EVPWIDE ghost-extent residual on the knobbed config | E.1 fuses (fewer events) → E.T1 stack (cheaper events) → partner-balance partition (equal counts) |
| flat comm ~49 / 46 | the same toll × mean partner count + the 2D latency floor | same three, same order |
| staging 18.6 / 15.3 | UCX cuda host-staging pipeline (GDR kernel path open, UCX_TLS forcing null/harmful) | HPC-X if §12 bar cleared; JUPITER C2C erases it natively |
| launch+fence 10.8 / 10.0 | ALREADY SPENT by h16/h17 fence deletions + CGPIPE | none — CUDA graphs OFF the bench (both scales measured) |
| CG remnant (knobbed: ~9 busy + ~13 wait @16N) | 23 iters × (1 exch + 2 Allreduce) | pocketed P-CSI/guess-extrap for ≥64N = JUPITER D4 |
| kernels 541 / 138 | memory-bound compute (74.6 % class) | TDMA/layout family — the post-comm frontier, M-L effort |

**THE ONE SESSION-15 BUILD LEVER (pre-registered): E.1a+c — FCT T+S co-pack +
Redi/GM adjacent co-packs** (§13 pre-reg: central −3..5 ms @4N, −5..8 ms @16N,
byte-class, FORCE_SERIAL proof per pair). **Attribution-validation rider: the ocean
busy spread must shrink ∝ deleted ocean events** (the causal test the correlation
still owes us). Env-side track (NOT a build): if the §12 HPC-X bar clears → fidelity
gate + options ×3 under the winning env before any adoption. Partition-side
(user's call, offline tooling): a partner-balancing METIS objective is now justified
by r=+0.74..0.96 across EVERY phase — proposed as the E.PART2 audit, mesh copies
under /work per rule 0.32.
