# dolpung (GH200) scaling campaign — 2026-07-22, session 17

*User directive: "fix the speed issue, then complete scaling for all meshes, fastest
config with speed options, and also SP (fastest); expectation: exceed A100." One day:
first compile on the partition → 21-point dual-precision fleet complete, 168/168 legs
rc=0.*

## Platform + the two GH200 lessons

dolpung = 42 nodes × 4 GH200 (72-core Grace + H100-class 120 GB, CUDA 12.9, aarch64).
Toolchain `env_dolpung.sh` (absolute paths; strips the x86 login-env leak — L102).
Compile on-node: `salloc -t 360 -A mh1571 -p dolpung -n 288`.

1. **Transport (L102)**: default UCX crashes on device buffers (gdrcopy can't pin
   cudaMalloc on coherent GH200). Blanket `^gdr_copy` = 3.6× slow. The recipe:
   `UCX_TLS=self,sm,cuda_copy,cuda_ipc[,dc_mlx5 inter-node]` **+ `FESOM_HOST_HALO=1`**
   (2.5× on top: device-pointer MPI costs ~1 ms/msg on this UCX; host halos ride the
   coherent C2C). M5.1's A100 lesson inverts on this architecture.
2. **Lever transfer is not 1:1**: EVPWIDE=8 measured as a LOSS on CORE2 g1 (0.0973 vs
   0.0845) → dropped from the fleet config; CGPOLY≈neutral through 16N (its A100 win
   is partially subsumed by host-halo).
3. **Probe verdicts (jobs 26403755/56, farc_g4 + dars_g8): the fabric cannot register
   CUDA memory for RDMA at all** — `ibv_reg_mr(... cuda ...) failed: Bad address` on
   mlx5 ⇒ (a) device-halo INTER-node = crash (HOST_HALO=1 is REQUIRED, not merely
   2.5× faster); (b) **EVPWIDE = unusable on GH200 multi-node** (its wide-halo
   exchange passes device pointers regardless of FESOM_HOST_HALO — crashed even in
   host-halo legs; fleet was safe: SPEED=1 does not enable EVPWIDE). This bounds the
   DP-vs-A100-Bp gap: A100's extra levers (EVPWIDE, proto env, GPUDirect halos) all
   ride Levante's working GPUDirect stack, which dolpung's driver/fabric lacks today
   — a platform limitation, not a port one. Revisit when DKRZ ships GPUDirect/dmabuf
   support (then re-run this probe job).

Fleet config: `FESOM_SPEED=1 + FESOM_HOST_HALO=1` (+CGPOLY=3 leg), DP = m7-speed
`893a04b`+env/jobs, SP = m8-precision `0ff8e3b` `-DFESOM_PRECISION=single`.
Protocol: 300 steps, min-of-2 same-alloc, dt 1800/900/120/180 (rule 0.41), snap −1.

## Results — s/step (min-of-2; `jobs/job_dolpung_scale`, logs in
## /work/ab0995/a270088/port2/dolpung/scale/)

| point | dp | dp_cgp | sp | sp_cgp |
|---|---|---|---|---|
| core2 g1 | 0.0787 | 0.0789 | 0.0635 | 0.0634 |
| core2 g2 | 0.0594 | 0.0592 | 0.0544 | 0.0544 |
| core2 g4 | 0.0629 | 0.0641 | 0.0542 | 0.0541 |
| core2 g8 | 0.0585 | 0.0588 | 0.0525 | 0.0523 |
| farc g1 | 0.2149 | 0.2154 | 0.1507 | 0.1509 |
| farc g2 | 0.1544 | 0.1556 | 0.1163 | 0.1163 |
| farc g4 | 0.1226 | 0.1212 | 0.0966 | 0.0977 |
| farc g8 | 0.1024 | 0.1025 | 0.0869 | 0.0875 |
| farc g16 | 0.0938 | 0.0932 | 0.0880 | 0.0880 |
| farc g32 | 0.1248 | 0.1252 | 0.1127 | 0.1130 |
| dars g1 ⭐ | 0.8597 | 0.8624 | 0.5719 | 0.5675 |
| dars g2 | 0.4763 | 0.4725 | 0.3221 | 0.3193 |
| dars g4 | 0.2720 | 0.2711 | 0.1876 | 0.1884 |
| dars g8 | 0.1633 | 0.1629 | 0.1173 | 0.1171 |
| dars g16 | 0.1161 | 0.1150 | 0.0859 | 0.0859 |
| dars g32 | 0.0891 | 0.0886 | 0.0701 | 0.0702 |
| ng5 g2 | 1.3536 | 1.3525 | 0.9261 | 0.9242 |
| ng5 g4 | 0.7414 | 0.7424 | 0.5107 | 0.5124 |
| ng5 g8 | 0.4301 | 0.4290 | 0.2983 | 0.2979 |
| ng5 g16 | 0.2539 | 0.2540 | 0.1806 | 0.1800 |
| ng5 g32 | 0.1784 | 0.1794 | 0.1365 | 0.1356 |

⭐ **dars g1 is a new capability**: 3.16M-node mesh on ONE node (4×120 GB) — impossible
on A100 (4×80 GB). Single GH200 GPU (np1, CORE2): 0.0604 s/step — one chip outruns a
whole M5.24-era A100 node (0.117).

Shape notes: CORE2 floors at ~0.052–0.059 past 2N (same over-decomp tail as A100);
farc peaks at 8–16N and inverse-scales at 32N (A100 did too); dars/NG5 scale
positively to 32N (dars 1.30×, NG5 1.42× per last doubling). SP/DP = 1.24× (CORE2)
→ 1.43–1.5× (big meshes) — the M8 per-rank-workload law holds on GH200.

## SYPD at production dt (CORE2 1800 · farc 1200 [M8 user convention; measured at
## dt900, no CG corr] · dars 240 ×1.0222 · NG5 240 ×1.0110)

| mesh (best point) | DP SYPD | SP SYPD |
|---|---|---|
| CORE2 (g8; g2 within 2 %) | 84.3 | **94.3** |
| farc (g16 dp / g8 sp) | 35.3 | **37.8** |
| dars (g32) | 7.26 | **9.17** |
| NG5 (g32) | 3.65 | **4.80** |

## vs A100 (Levante) — the honest three-anchor ledger

Anchors: (a) M5.24 fleet = knobs-off-era build (`docs/SCALING_M524.md`); (b) A100-best
= M8-Bp fleet (SPEED+EVPWIDE+CGPOLY3+proto env; dp+sp, g1–g16, `port2/mp/gate2/`) and
M7 fleet A/Bp at g32 (DP only).

- **vs M5.24 A100 (same-maturity comparison)**: GH200 DP is 1.4–1.9× faster
  node-for-node on farc/dars/NG5, growing with N; SP doubles it (2.0–2.6×).
- **vs A100-best (Bp)**:
  - **SP: GH200 ≥ A100-best-SP on the big meshes** — dars g16 tied (0.0859 vs
    0.0860), NG5 g2/g4 ahead (0.9242 vs 0.9843 / 0.5107 vs 0.5195), NG5 g16 behind
    (0.1800 vs 0.1629); **at g32 (no A100-SP anchor exists) GH200-SP 0.1356 beats
    even A100's best-ever DP-Bp 0.1435 (NG5) and 0.0701 vs 0.0807 (dars)**.
  - **DP: A100-Bp still leads by ~1.2–1.5×** at like N (its stack alone was worth
    2–2.2×; GH200 currently runs SPEED+HOSTHALO only). Largest gap on farc
    (0.0639 vs 0.0932 @16N) and CORE2 (0.0443 vs 0.0592 @2N) — the small/medium
    meshes are where A100's proto-env/EVPWIDE levers paid most.
- Suspects for closing the DP gap = the pending probes: EVPWIDE on big meshes,
  device-halo at big-mesh message sizes, and a GH200 env-tuning pass (nothing like
  E.T1 has been attempted on this fabric yet).

## FLEET v2 — the STAGE transport (evening 2026-07-22; supersedes the v1 numbers above)

User calibration: JUPITER GH200 ≈ 3× A100, dolpung reportedly ≥ JUPITER ⇒ v1's ~1×
meant a setup problem. Found it: **FESOM_HOST_HALO=1 (v1's fleet config) is a debug
fallback that reverts EVERY exchange to full-field GPU↔host syncs AND silently
deactivates CGPIPE+CGPOLY** (`[cgpipe] INACTIVE` was in every v1 log; the "2.4×
FESOM_SPEED gain" was CGPIPE-less). Falsified en route: 1-core-per-rank binding
(c1 0.0789 vs c72 0.0791 — no effect); UCX cuda-layer/memtype overhead (no effect).
np1-vs-np1 chip truth: **GH200 = 2.0× A100** (0.0574 vs 0.1153, same binary+knobs).

**Fix: `FESOM_HALO_STAGE=1`** (new transport mode, both trees): device pack/unpack
+ per-partner packed buffers unchanged; the MPI leg runs on pinned-host mirrors
(D2H/H2D deep_copy of the USED range). No GPUDirect needed; CG levers stay live.
Five sites: 3 generic exchanges + cgpipe ring (2100) + cgpoly R-ring (2110).
CORE2 g1: HOST_HALO 0.0788 → STAGE 0.0564 → **STAGE+CGPOLY 0.0460** (×2 reps) —
beats A100-best-ever 1-node (m8-Bp dp 0.0558).

**Certification (the achievable level for the CUDA build):** per-exchange halo
selfcheck SILENT (dolpung + A100) and cgpoly selfcheck 0.000e+00 under STAGE;
trajectory-level bitwise gates are INVALID for the CUDA build — measured: the
device transport does not reproduce ITSELF run-to-run (atomics; dev-vs-dev T
3.7e-4 = dev-vs-stage 3.8e-4 after 10 steps, same class). The v1↔v2 "divergence"
scares were this intrinsic noise. `FESOM_HALO_NOFUSE=1` debug knob added
(decompose fused exchanges; the fused paths have no selfcheck variant).

Fleet v2 = same 21-point matrix, legs dp/dp_cgp/sp/sp_cgp all with
SPEED=1+STAGE=1 (+CGPOLY on _cgp), -c72 block:block. Jobs 26405881-26405901.

## Ops notes

- Fresh-association queue lag ~40 min at priority 1 (all sprio factors 0), misleading
  "Nodes DOWN/DRAINED/reserved" reason; later waves started in ~5 min.
- AssocMaxJobsLimit=5 RUNNING per user — idle `--no-shell` allocations count against
  it and throttle a fleet; release them before submitting.
- ~40-min pathfinder discipline paid: core2_g2 validated `dc_mlx5` inter-node before
  the 17 multi-node jobs launched. Zero failed jobs in the campaign.
