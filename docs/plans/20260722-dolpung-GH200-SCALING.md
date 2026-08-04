# dolpung (GH200) scaling campaign — 2026-07-22, session 17

*User directive: "fix the speed issue, then complete scaling for all meshes, fastest
config with speed options, and also SP (fastest); expectation: exceed A100." One day:
first compile on the partition → 21-point dual-precision fleet complete, 168/168 legs
rc=0.*

## Platform + the GH200 lessons

dolpung = 42 nodes x 4 GH200 (72-core Grace + H100-class 120 GB, aarch64).
Toolchain `env_dolpung.sh` (absolute paths; strips the x86 login-env leak - L102).
Compile on-node: `salloc -t 360 -A mh1571 -p dolpung -n 288`.

1. **The fabric cannot register CUDA memory** (probes 26403755/56): intra-node
   `gdr_pin_buffer ret:22` -> `Fatal: failed to register buffer with mem type
   domain cuda`; inter-node `ibv_reg_mr(... cuda ...) Bad address`. No working
   GPUDirect => device-pointer MPI is unusable. Consequences: **EVPWIDE is
   unusable multi-node** (its wide-halo exchange passes device pointers
   regardless of the halo knobs; the fleet was safe because SPEED=1 does not
   enable it), and the halo transport must avoid device pointers entirely.
2. **Transport recipe**: `UCX_TLS=self,sm,cuda_copy,cuda_ipc[,dc_mlx5
   inter-node]` (blanket `^gdr_copy` = 3.6x slower: UCX falls back to
   knem/xpmem/IB-loopback) **+ `FESOM_HALO_STAGE=1`** (section FLEET v2).
   ⚠️ NOT `FESOM_HOST_HALO=1` - that is a debug fallback, see below.
3. **Lever transfer is not 1:1**: EVPWIDE = measured loss AND fabric-incompatible
   (out); CGPOLY=3 = a BIG win under STAGE (-20..-35 %), unlike under the v1
   host-halo config where it was silently inactive.
4. **ICON-dolpung platform tricks A/B'd** (their `sap0006.run`, k203123; job
   26423973 on dars_g2): per-rank NIC pinning `UCX_NET_DEVICES=mlx5_$LOCALID:1`
   = **49 % SLOWER** for our 4-rank/node layout (0.293 vs 0.196) - UCX's
   automatic distance-aware selection wins; numactl membind, `MALLOC_CONF=
   thp:never`, `OMPI_MCA_coll=^hcoll,ml` all neutral. Nothing adopted.
5. **Toolchain ceiling probed** (jobs 26424926/26425016/26425400): the vendored
   Kokkos 4.4.01 CANNOT build against CUDA 13.x (cudaGraphAddDependencies /
   cudaMemLocation API breaks); Kokkos 4.7.04 + CUDA 13.2 DOES build (via the
   new `-DFESOM_KOKKOS_DIR=` escape hatch; `ARCH_NATIVE` breaks on aarch64 with
   `-msve-vector-bits=`, use `ARCH_ARMV9_GRACE`) and is **-4.6 % on the
   launch-bound case (CORE2 g1) but 0 % where per-GPU work is large (dars g2)**.
   Not adopted: does not pay for a certification cycle. Details in the JUPITER
   plan addendum.

## RESULTS — FLEET v2, the campaign of record (complete: 21 points, 168/168 legs rc=0)

s/step, min-of-2, best of the plain / +CGPOLY3 legs. Config: `FESOM_SPEED=1 +
FESOM_HALO_STAGE=1` (+`FESOM_SPEED_CGPOLY=3`), `-c72 block:block`, 300 steps,
snap −1. DP = m7-speed build, SP = m8 `-DFESOM_PRECISION=single`.
Logs `/work/ab0995/a270088/port2/dolpung/scale/`, jobs 26405881-26405901 +
26411467/68. (The v1 HOST_HALO numbers this doc first carried are SUPERSEDED —
see the FLEET v2 section below for why they were wrong by ~2x.)

| nodes (GPUs) | CORE2 dp | CORE2 sp | farc dp | farc sp | dars dp | dars sp | NG5 dp | NG5 sp |
|---|---|---|---|---|---|---|---|---|
| 1 (4)    | 0.0463 | 0.0397 | 0.0930 | 0.0770 | 0.3697 | 0.3175 | — | — |
| 2 (8)    | 0.0328 | 0.0295 | 0.0606 | 0.0524 | 0.1964 | 0.1682 | 0.5859 | 0.5236 |
| 4 (16)   | 0.0328 | 0.0303 | 0.0507 | 0.0446 | 0.1150 | 0.0947 | 0.3152 | 0.2774 |
| 8 (32)   | 0.0325 | 0.0306 | 0.0472 | 0.0427 | 0.0704 | 0.0590 | 0.1776 | 0.1541 |
| 16 (64)  | — | — | 0.0450 | 0.0412 | 0.0517 | 0.0448 | 0.1045 | 0.0886 |
| 32 (128) | — | — | 0.0535 | 0.0521 | 0.0459 | 0.0398 | 0.0753 | 0.0657 |

⭐ **dars g1 is a new capability**: the 3.16M-node mesh runs on ONE node (4×120 GB)
— impossible on A100 (4×80 GB). Single GH200 GPU, CORE2 np1: 0.0574 s/step.

Shape: CORE2 saturates past 2 nodes (floor ~0.030 = the over-decomposition tail,
same as A100); farc peaks at 16 nodes and inverse-scales at 32 (0.0450 → 0.0535
dp) exactly as it does on A100; **dars and NG5 keep scaling to 128 GPUs**
(g16→g32 = 1.13x dars, 1.39x NG5 dp).

**SP speedup (FP64 s/step / FP32 s/step, matched pairs): 1.03–1.21x, median
1.15x** — figures `fig_dolpung_spdp_{small,large}`. Per mesh on GH200: CORE2
1.06–1.17 (falls with node count), farc 1.03–1.21, dars 1.15–1.21, NG5
1.12–1.18. The A100 median is 1.18x (range 0.99–1.29, noisier), so **FP32 buys
slightly LESS on GH200 than on A100** — expected: STAGE removed the byte-volume
bottleneck that FP32 was partly compensating for, and the M8 design keeps FP64
islands (CG scalar chain, zstar ALE accumulation), so the halved-word benefit
never applies to the whole step. Under the v1 host-halo config the same builds
showed 1.24–1.5x — that number measured the full-field sync traffic, not the
model.

## SYPD at production dt (CORE2 1800 · farc 1200 · dars 240 x1.0222 · NG5 240 x1.0110)

| mesh | best DP point | SYPD | best SP point | SYPD |
|---|---|---|---|---|
| CORE2 | g8  | 152 | g2  | **167** |
| farc  | g16 | 73  | g16 | **80** |
| dars  | g32 | 14.0 | g32 | **16.2** |
| NG5   | g32 | 8.6 | g32 | **9.9** |

## vs A100 (Levante) — node-for-node against the A100's BEST tuned config

Anchor = the m8-Bp fleet (SPEED+EVPWIDE+CGPOLY3+proto env, dp+sp legs, ladder
ends at 64 GPUs) and, at 128 GPUs, the m7 DP-Bp fleet.

| point | GH200 dp / A100 dp | GH200 sp / A100 sp |
|---|---|---|
| CORE2 g2  | 0.0328 / 0.0443 = **1.35x** | 0.0295 / 0.0429 = **1.45x** |
| farc g16  | 0.0450 / 0.0639 = **1.42x** | 0.0412 / 0.0614 = **1.49x** |
| dars g16  | 0.0517 / 0.0955 = **1.85x** | 0.0448 / 0.0860 = **1.92x** |
| NG5 g16   | 0.1045 / 0.2003 = **1.92x** | 0.0886 / 0.1629 = **1.84x** |
| dars g32  | 0.0459 / 0.0807 = **1.76x** | (no A100 SP anchor) |
| NG5 g32   | 0.0753 / 0.1435 = **1.91x** | (no A100 SP anchor) |

**Verdict: GH200 beats the A100's best-tuned configuration everywhere, by 1.35–1.5x
on the small meshes and 1.8–1.9x on the multi-million-node meshes** — i.e. the full
chip ratio (measured 2.0x np1-vs-np1, CORE2, same binary+knobs: 0.0574 vs 0.1153)
is delivered at scale despite dolpung having NO working GPUDirect, because STAGE
moves only packed halo bytes over the coherent C2C link. Figures:
`scripts/dolpung_sypd_figs.py` -> `/work/ab0995/a270088/port2/dolpung/figs/`
(sp/dp/both x small/large).


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
