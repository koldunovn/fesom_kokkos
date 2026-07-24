# M7 → JUPITER: scaling experiments with the optimized port — PLAN + instructions

*Written 2026-07-17 (session 14), at the user's request, while the Levante E.IMB fleet
drains. Branch `m7-speed`, prepared through the commit this file lands in. Everything
here is written so a fresh session (or a person) can execute it on JUPITER top-to-bottom.*

**JUPITER = the JSC exascale system: GH200 Grace-Hopper nodes (4× GH200 per node —
72-core Grace ARM + H100-class GPU per superchip, NVLink-C2C 900 GB/s host↔device,
HBM3), NDR InfiniBand.** System facts marked ⚠️VERIFY below are from general knowledge
and MUST be re-checked on the machine (module names, partitions, quotas change).

---

## 0. WHAT WE WANT TO GET (the deliverables, in priority order)

| # | deliverable | why |
|---|---|---|
| D1 | **Strong-scaling curve of the optimized port on NG5** (7.4M surface nodes, nl≤70): s/step + SYPD@dt240 + parallel efficiency at 4→64(+) GH200 nodes, for (a) the `FESOM_SPEED=1` master set and (b) the knobbed set (+`FESOM_SPEED_CGPOLY=3` +`FESOM_SPEED_EVPWIDE=8`) | the headline result; Levante tops out at 16N |
| D2 | **GH200-vs-A100 node-for-node transfer** at matched rank counts (JUPITER 4N/16N vs Levante 4N/16N anchors 0.6382/0.2413 master, 0.6213/0.2314 knobbed) | 74.6 % of the step is memory-bound kernels → HBM3 should pay ~2-3×; NVLink-C2C should ~erase the 18.6 ms/step PCIe staging pool |
| D3 | **Per-phase, per-rank scaling attribution** via `FESOM_SPEED_PHASESTATS=1` at EVERY scale point (busy/wait for force/ice(dyn/adv)/coupl/ocean/cg/other) | extends session-14's E.IMB work to scale: where does the model leave the compute-bound regime; how do the imbalance pools grow with N; whether the partner-count overhead story (findings §10) amplifies |
| D4 | **The ≥64N solver decision with data**: CG share + Allreduce cost vs N | the P-CSI/initial-guess levers are pocketed "for ≥64N" (rule 0.31, lit-#4/#5) — JUPITER produces the numbers that un-pocket or bury them |
| D5 | **A published-peer calibration row** (lit-survey §8 table) at JUPITER scale | paper material; the field's shared frontier (2D latency + imbalance) measured on the newest hardware |
| D6 | (opportunistic) **GPUDirect/transport story on a properly-wired machine** — probe + env legs; GH200 C2C changes the staging economics entirely | Levante session-14 verdict was "path open, no cheap win"; JUPITER's fabric is a different animal |

**Non-goals:** no new physics, no new levers designed on JUPITER (levers are built and
certified on Levante against the oracle chain; JUPITER measures them at scale). Mixed
precision stays BANNED (user directive).

## 1. WHAT TO TRANSFER (from Levante; sizes measured 2026-07-17)

| what | path on Levante | size | notes |
|---|---|---|---|
| repo | `github.com/koldunovn/fesom_kokkos.git` branch `m7-speed` | — | 🔴 needs a PUSH (commits after `c9f1749` are local; ASK THE USER first — standing rule) |
| NG5 mesh statics | `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5/` (`nod2d.out elem2d.out aux3d.out nlvls.out elvls.out edge*` + caches) | ~few GB of the 35 GB total | rsync selectively |
| NG5 dists | same dir, `dist_{16,32,64,128,256,512,1024}` (2048/4096/8192 exist if needed) | small each | partitions ALREADY EXIST for every planned scale — no partitioner needed on JUPITER |
| private CORE2 mesh | `/work/ab0995/a270088/port2/mesh/core2` (+`dist_1,2,8`) | ~1-2 GB | the GATE mesh (L73: never /pool core2) |
| gate reference | `/work/ab0995/a270088/port2/m6_baseline_serial/` (snap_*.nc) | small | for the CROSS-ARCH DIFF (documentation, not a pass bar — §3) |
| forcing 1958 | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/{uas,vas,huss,rsds,rlds,tas,prra,prsn}.1958.nc` + `PHC2_salx.nc` + `CORE2_runoff.nc` | ~12 GB | `FESOM_FORCING_DIR` (new env knob, this commit) points at the copy |
| chl climatology | `/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc` | small | `FESOM_CHL_FILE` env (already existed) |
| PHC IC | `/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc` | 33 MB | run argument |
| provenance | `/work/ab0995/a270088/port2/m7/bin/*/PROVENANCE.txt` | tiny | paper trail; binaries themselves are useless cross-arch |
| M5.23 climate refs | see `docs/REFERENCE_RUNS.md` for the ref-run dirs | ~GB | only needed if a climate leg runs on JUPITER (recommended once, at the end) |

Transfer route: `rsync -avP` Levante→JUPITER over the login nodes (both DKRZ and JSC
allow outbound ssh; JUDAC/`judac.fz-juelich.de` is the JSC transfer host ⚠️VERIFY).
Put data under the project's `$SCRATCH`/`$PROJECT` equivalent; meshes are COPIES by
definition there (rule 0.32 trivially satisfied — but the private-CORE2-dist_8 rule
still holds: NEVER regenerate it, copy it).

## 2. BUILD ON JUPITER

- Toolchain ⚠️VERIFY via `module spider`: an NVHPC (24.x+) or GCC+CUDA 12.x stage with
  a CUDA-AWARE MPI (OpenMPI or ParaStationMPI). The Levante lesson L77 transfers as a
  CLASS: *an MPI built without CUDA support SEGFAULTs on device pointers — prove
  CUDA-awareness before believing any halo crash* (`ompi_info --parsable | grep
  cuda`, or the ParaStation equivalent, plus a 2-rank device-buffer ping-pong).
- CMake (the repo's `configure.sh` idiom, new build dirs):
  - Serial oracle build: `-DCMAKE_BUILD_TYPE=Release` host-only; host compiler = the
    stage's g++ (aarch64). Add `-DKokkos_ARCH_NATIVE=ON` (or `ARMV9_GRACE` if the
    vendored Kokkos 4.4 exposes it ⚠️VERIFY with `cmake -LA | grep ARCH`).
  - CUDA build: vendored `externals/kokkos/bin/nvcc_wrapper` with
    `-DKokkos_ARCH_HOPPER90=ON` (replaces Levante's `AMPERE80`) +
    `NVCC_WRAPPER_DEFAULT_COMPILER=g++` (aarch64 g++).
- netcdf-c linked from the stage (⚠️VERIFY a netcdf-c matching the chosen MPI).
- **Env knobs that do NOT transfer** (Levante-specific, re-derive on JUPITER):
  `UCX_NET_DEVICES=mlx5_0:1` (JUPITER has ~1 NIC per GH200 ⚠️VERIFY — likely NO pin
  needed, or a per-rank pin), `OMPI_MCA_pml=ucx OMPI_MCA_btl=self`,
  `UCX_MEMTYPE_CACHE=n`, hcoll off, romio. Start with the system defaults + the
  smallest set that runs; record every deviation in the job header like the Levante
  templates do.
- `FESOM_FORCING_DIR=<jupiter forcing copy>` in every job.

## 3. GATE LADDER ON JUPITER (redefined for cross-architecture — this is the part
## that is NOT a copy of the Levante ladder)

The x86 bit-identity chain (Serial ≡ C oracle ≡ Fortran refs) does NOT carry across
architecture/compiler (libm, FMA contraction, SIMD order). What carries and what
replaces it:

1. **J-G1 serial smoke** (np1, CORE2 private, 20 steps): runs, no NaN, plausible state.
2. **J-G2 arch-internal byte gates — these DO carry:** np1-vs-np8 `diff_snap` byte
   equality ON JUPITER (scatter/halo proof); knob-OFF vs knob-ON byte proofs
   (`FESOM_SPEED_FORCE_SERIAL=1` ladder) for every byte-class lever — the FORCE_SERIAL
   methodology is architecture-internal and transfers VERBATIM.
3. **J-G3 cross-arch drift DOCUMENTATION (not a pass bar):** JUPITER-serial 20-step
   CORE2 vs the Levante `m6_baseline_serial` snapshots — record max|Δ| per field.
   Expect a small compiler/libm floor; whatever it measures becomes the documented
   JUPITER floor (L79 discipline: every platform has its own floor — measure it, never
   assume it).
4. **J-G4 CUDA fidelity:** JUPITER-CUDA vs JUPITER-serial, 20 steps, the D22
   climate-close criterion (coherence/outlier logic of `gpu_fidelity_check.py`, ceilings
   as on Levante). This is the gate of record for the CUDA build.
5. **J-G5 options ×3** (TKE / mEVP / zstar) on the JUPITER-CUDA build, vs JUPITER-serial
   (L91: per-platform re-cert).
6. **J-G6 (once, at the end):** 1-yr NG5 climate leg vs the M5.23 bar refs (sst/sss/
   ssh/a_ice pattern correlations) — the arbiter that the whole stack on new hardware
   still produces the climate. Bars are correlation-based → arch-robust.

## 4. EXPERIMENT MATRIX (after gates; std300 protocol = 300 steps dt180 NG5,
## min-of-2, same-alloc A/B legs, `BIN=` pinned frozen binaries, pre-registered)

**J0 — anchors + machinery shakedown (4N = 16 ranks, dist_16):**
ref / phst / knobbed legs (one ab_env-style job). Establishes: JUPITER 4N s/step
(master + knobbed), PHASESTATS overhead ≤0.5 % (re-verify on the new stack), the
GH200 staging story (SYNCSTATS + one nsys census leg — pre-reg: the memcpy/staging
class COLLAPSES vs Levante's 18.6 ms; launch/fence class similar).

**J1 — strong scaling, master set:** nodes {4, 8, 16, 32, 64} → ranks {16, 32, 64,
128, 256} (dists exist for all). Per point: 3-leg same-alloc job = ref / phst /
knobbed(+phst). Deliverables D1-D3 fall straight out of the phst tables (per-phase
busy/wait vs N; the imbalance pools vs N; comm-share curve).
- Homogeneity check ONCE (the L94 analog): the SAME 4N job on two disjoint node sets;
  if per-rank busy reshuffles with hardware → JUPITER has its own lottery → pin rule.

**J2 — the knob value curve:** from J1's knobbed legs: Δ(CGPOLY), Δ(EVPWIDE), Δ(both)
vs N. Pre-registration (honest, from Levante structure): both knobs' relative value
GROWS with N (they delete latency-class events; the latency share grows) — if it
SHRINKS, the GH200 fabric latency floor differs and that's a finding.

**J3 — deep scale (conditional on J1 efficiency ≥ ~50 % at 64N):** 128N/256N (dists
512/1024). This is D4's data: CG wall share, per-iteration Allreduce cost, iters vs N.
Decision rule: CG share ≥ 15 % at ≥64N ⇒ un-pocket P-CSI + initial-guess extrapolation
as the next Levante/JUPITER build levers; else they stay buried.

**J4 — transport (opportunistic):** the `job_m7_gdrprobe` equivalent (peermem/gdrdrv/
`ucx_info -d`) + 2-3 env legs ONLY if J0 shows a staging/latency pool worth chasing.
A.3 caution transfers: A/B only, fidelity gate before any adoption.

## 5. PRE-REGISTERED EXPECTATIONS (structure, not numerology — every J-run gets its
## own pre-reg in the JUPITER findings doc before submission)

- **PCIe/staging pool → ~0** on GH200 (C2C replaces PCIe; Levante: 18.6 ms @4N).
- **Kernel share speeds up ×2-3** (HBM3 vs HBM2e on the 74.6 % memory-bound share);
  the comm/latency floors move LESS ⇒ **the step becomes MORE comm/imbalance-bound at
  the same N** — PHASESTATS quantifies exactly this shift (D3).
- The session-14 imbalance structure (partner-count-correlated exchange overhead; the
  ocean busy gradient) reappears on JUPITER if it is code/data-driven; if it was
  Levante-hardware, it vanishes. Either outcome is information.
- SYPD: no number promised before J0 lands (rule: measure, then extrapolate).

## 6. MACHINERY TO CLONE (all in `jobs/`, adapt headers only: partition/account/
## gres/constraint names ⚠️VERIFY, `-C a100_80` has no JUPITER meaning)

`job_m7_ab_env` (the workhorse — same-alloc multi-leg env A/B) · `job_m7_gate_serial`
+ `job_m7_gpu_gate` (gate pair; swap mesh/ref paths) · `job_m7_nsys` +
`scripts/m7_stall_budget.py` + `scripts/m7_gap_census.py` (census; nsys ships with
NVHPC) · `scripts/m7_rank_features.py` + `scripts/m7_phasestats_join.py` (the E.IMB
analysis chain — works from mesh dir + run log alone) · `jobs/m7_provenance.sh`
(md5-pins every run). The walltime honesty rule transfers: a cheap gate must LOOK
cheap to the scheduler.

## 7. RISKS / OPEN ITEMS

- ⚠️ aarch64 + nvcc_wrapper + vendored Kokkos 4.4: expected to work (Kokkos supports
  Grace-Hopper since 4.0) but is the likeliest day-0 friction; fall back to
  `Kokkos_ARCH_NATIVE` host-side and plain `HOPPER90` device-side.
- ⚠️ JUPITER MPI: if only ParaStationMPI is CUDA-aware in the default stage, the OMPI
  MCA knob set in our job templates is inert — strip it, use `PSP_CUDA=1`-class knobs
  instead (⚠️VERIFY names).
- ⚠️ Queue policy: large-node-count windows may be batched (reservation days); plan
  J3 around them.
- The serial oracle on Grace is ~fast enough for gates (np8 CORE2 = minutes) — if not,
  gates move to np8-on-one-node and nothing else changes.
- JRA55 transfer (12 GB) + NG5 (few GB selective) over WAN: hours, do it first.

## 8. DAY-0 CHECKLIST ON JUPITER (concrete)

```
1  module spider nvhpc; module spider openmpi; module spider netcdf   # pick the stage
2  git clone -b m7-speed <repo>; cd port_kokkos
3  serial build (host g++, ARCH_NATIVE) -> np1 CORE2 smoke (J-G1)
4  ompi_info|grep -i cuda (or PSP equivalent) -> 2-rank device ping-pong
5  CUDA build (HOPPER90) -> np8 J-G2 byte gates -> J-G3 drift doc -> J-G4 fidelity
6  rsync data (start FIRST, it runs while 1-5 proceed)
7  clone job headers (partition/account); FESOM_FORCING_DIR everywhere
8  J0 anchors -> write pre-regs -> J1
```

---
*Levante session-14 context feeding this plan: launch/fence pool collapsed (10.8 ms);
the idle pool = MPI wait + imbalance; imbalance is composite (ice: partner-count-
correlated r=+0.80, NOT ice-cover r=−0.21; ocean: node-monotone, hardware-vs-data
pending); GPUDirect kernel-path open on Levante but no cheap UCX_TLS win. JUPITER's
D3 attribution at scale is the continuation of exactly that thread.*

---

# ADDENDUM (2026-07-22 s17): the dolpung GH200 shakedown — what carries to JUPITER

*dolpung (DKRZ, 42× quad-GH200) is JUPITER-class hardware; users report dolpung ≈
JUPITER node-for-node. One day of shakedown (campaign doc
`20260722-dolpung-GH200-SCALING.md`) produced a complete 21-point dual-precision
fleet and rewrote several assumptions in this plan. The JUPITER campaign should
reproduce the dolpung FLEET v2 protocol below, not the original §4 matrix alone.*

## A. The transport decision ladder (REPLACES blind reuse of Levante env pins)

Run these in order on day 0; each is a 10-step CORE2 dist_4 job:

1. **Device transport as-is** (no knobs). If it crashes at step 1 with
   `gdr_pin_buffer failed` / `Fatal: failed to register buffer with mem type
   domain cuda`, or multi-node with `ibv_reg_mr(... cuda ...) Bad address` —
   the fabric has no working GPUDirect (dolpung's state 2026-07).
2. **`FESOM_HALO_STAGE=1`** — the s17 transport (device pack/unpack kept, MPI leg
   on pinned-host mirrors of the PACKED buffers; CGPIPE/CGPOLY stay live). Works
   on ANY fabric. On dolpung this delivered the full 2.0× chip ratio:
   CORE2 g1 0.0463 dp / 0.0397 sp; dars 1.8–1.9× over the A100-best fleet at
   every node count. **Even if GPUDirect works on JUPITER, A/B device-vs-STAGE —
   on coherent GH200 the staged path may win regardless.**
3. **NEVER `FESOM_HOST_HALO=1` for performance.** It is a debug fallback: reverts
   every exchange to FULL-FIELD GPU↔host syncs AND silently deactivates
   CGPIPE+CGPOLY (`[cgpipe] INACTIVE` banner). It cost dolpung v1 a factor ~2-3
   and masqueraded as "GH200 barely beats A100" for half a day.
4. UCX: start from defaults + `UCX_TLS=self,sm,cuda_copy,cuda_ipc[,dc_mlx5
   inter-node]`; with STAGE the MPI buffers are host, so IB device names matter
   less. Do NOT ship `^gdr_copy` (3.6× slow: UCX picks knem/xpmem/loopback).
   Levante's mlx5_0 pin and the E.T1 proto trio remain Levante-only until A/B'd.

## B. Knob transfer measured on GH200 (v2 fleet, STAGE config)

- `FESOM_SPEED=1` (CGPIPE et al.): large, always on.
- `FESOM_SPEED_CGPOLY=3`: **BIG on GH200** — −20 % (CORE2 g1) to −35 % (farc g8),
  everywhere positive under STAGE. Run it as its own leg (dp_cgp/sp_cgp).
- `FESOM_SPEED_EVPWIDE=8`: measured LOSS on GH200 CORE2 (0.0973 vs 0.0845) and
  its wide-halo exchange still passes device pointers (crashes on no-GPUDirect
  fabrics). Leave OUT unless JUPITER GPUDirect works AND an A/B says it pays.
- Host binding: `--cpus-per-task=72`, `srun -c72 --distribution=block:block`
  (rank i → Grace socket i). Measured ~neutral on dolpung but it is the correct
  hygiene and free. 1-cpu-per-task (SLURM default!) was measured harmless too —
  but don't rely on that on JUPITER.
- SP: build the m8-precision branch with `-DFESOM_PRECISION=single`; SP/DP =
  1.24× (CORE2) → 1.5× (big meshes) on GH200 — same per-rank-workload law as A100.

## C. Build lessons (aarch64)

- `env_dolpung.sh` is the template: ABSOLUTE PATHS, no modules; sanitize
  `CMAKE_PREFIX_PATH`/`PKG_CONFIG_PATH`/`PATH` of foreign-arch spack entries
  (srun exports the login environment; an x86 netcdf via CMAKE_PREFIX_PATH and
  an x86 `git` each broke one configure on dolpung). Pin
  `-DNC_CONFIG=<aarch64 nc-config>` explicitly.
- ⚠️ **The vendored Kokkos 4.4.01 CANNOT build against CUDA 13.x** (tested 13.0 and
  13.2 on dolpung, 2026-07-22): CUDA 13 changed API signatures the Kokkos CUDA
  backend uses — `Kokkos_Cuda_Instance.hpp` fails on `cudaGraphAddDependencies`
  (now takes `const cudaGraphEdgeData*`), `cudaMemAdvise`/`cudaMemPrefetchAsync`
  (now take `cudaMemLocation`, not an int device id). **If JUPITER's stack is
  CUDA 13-only, this is a day-0 BLOCKER**: either pick a CUDA 12.x toolkit (12.9
  is proven) or bump the Kokkos submodule to a CUDA-13-capable release (≥4.7)
  — the latter re-opens certification, so treat it as a decision, not a detail.
- **Kokkos 4.7.04 DOES build against CUDA 13.2** (measured 2026-07-22; use
  `-DFESOM_KOKKOS_DIR=<external kokkos tree>` to test without touching the pinned
  submodule). Two caveats: (a) with `-DKokkos_ARCH_NATIVE=ON` the 4.7 build FAILS
  on aarch64 with `g++: error: missing argument to '-msve-vector-bits='` (its SVE
  width auto-detection yields an empty value) — use `-DKokkos_ARCH_ARMV9_GRACE=ON`
  instead; (b) payoff is workload-dependent: **−4.6 % on the launch-overhead-bound
  case (CORE2 g1: 0.0453 vs 0.0475) but 0 % where per-GPU work is large (dars g2:
  0.1960 vs 0.1966)**, same allocation, 2 reps each. Kokkos-version and toolkit
  effects are NOT separated (both moved at once).
- ⚠️ Related, and needed on ANY CUDA-13 machine even after a Kokkos bump:
  `externals/kokkos/bin/nvcc_wrapper` hardcodes `default_arch="sm_70"` and
  appends it to every arch-less invocation (CMake's compiler probe). CUDA 13
  dropped Volta ⇒ `nvcc fatal: Unsupported gpu architecture 'sm_70'` before any
  real source compiles. Fix shipped: `scripts/nvcc_wrapper_sm90` (same wrapper,
  sm_90 default) — pass it as `CMAKE_CXX_COMPILER`.
- CMake: `-DKokkos_ENABLE_CUDA -DKokkos_ARCH_HOPPER90 -DKokkos_ARCH_NATIVE` +
  nvcc_wrapper + gcc≥12 host; Kokkos 4.4.1 + CUDA 12.9 + gcc 14.2 = clean.
  Full CUDA build ~7 min on 64 Grace cores; build ON a compute node via a batch
  job (no interactive needed): `sbatch -N1 -c144 --wrap 'make -C build… -j96'`.

## D. Certification method (REFINES §3 J-G2/J-G4)

**Trajectory-level bitwise gates are INVALID for the CUDA build** — measured on
A100: the device transport does not reproduce ITSELF run-to-run (atomics;
dev-vs-dev = dev-vs-stage noise, T ~4e-4 after 10 steps CORE2). Valid instruments:
- per-exchange halo selfcheck `FESOM_HALO_SELFCHECK=1` (silent = pass),
- `FESOM_CGPOLY_SELFCHECK=1` (prints max|apply−ref|, expect 0.000e+00),
- serial byte gates on-platform (Serial backend is unaffected by all of this),
- climate-bar correlations for anything longer (M5.23 bar).
`FESOM_HALO_NOFUSE=1` decomposes the fused device2/deviceN exchanges (no
selfcheck variant exists for the fused layout) — the isolation lever if a fused
transport bug is ever suspected.

## E. The fleet protocol to reproduce on JUPITER (= dolpung FLEET v2)

Clone `jobs/job_dolpung_scale` + `jobs/submit_dolpung_scaling.sh` (adapt
partition/account/gres + mesh paths after transfer). Per point: 4 legs
(dp, dp_cgp, sp, sp_cgp) × min-of-2, 300 steps, snap −1, dt CORE2 1800 / farc
900 / dars 120 / NG5 180 (rule 0.41), SYPD at production dt with the measured
CG corrections (dars ×1.0222, NG5 ×1.0110; farc reported at dt1200).
Matrix: CORE2 g1-8, farc g1-32, dars g1-32 (g1 fits GH200's 4×120 GB — a
capability A100 lacks), NG5 g2-32(+64 on JUPITER where the dists exist to 8192).
Ops: pathfind ONE 2-node job before releasing the multi-node bulk; walltimes
sized to the v2 speeds (see dolpung numbers) so backfill works; idle `--no-shell`
allocations count against per-user MaxJobs — release them before the fleet.
Figures: `scripts/dolpung_sypd_figs.py` (house style; add a JUPITER series
alongside the GH200/A100 ones).

## F. Calibration for pre-registration (§5), from dolpung v2

Chip-for-chip (np1, CORE2, same knobs): GH200 = 2.0× A100. Node-for-node vs the
A100-best (m8-Bp) fleet: CORE2 1.4-1.5×, farc 1.4-1.5×, dars 1.8-1.9×, growing
with mesh size and node count. JUPITER with working GPUDirect should match or
exceed these; if a JUPITER point lands BELOW its dolpung twin, suspect setup
(transport ladder step A) before anything else — that exact symptom on dolpung
was the HOST_HALO trap.
