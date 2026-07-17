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
