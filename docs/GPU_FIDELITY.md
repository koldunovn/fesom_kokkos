# GPU / OpenMP climate fidelity (M3.2)

**Status: SKELETON — fill in when the M3.2 runs complete.** Created at the end of M4 (whole model on
device, tag `m4-full-device`); the per-kernel + 1-yr Serial gates are bit-identical, this doc records the
*climate-close* fidelity of the threaded/GPU backends over a multi-year run.

## Why this doc

Serial is the bit-identity oracle (`max|Δ|==0` vs the C twin, per-kernel + the 1-yr CORE2 acceptance).
OpenMP and CUDA are **climate-close, not bit-identical** (D22): the edge/element→node `atomic_add`
SCATTERS (`visc_filt_bidiff`, `momentum_adv`, `vert_vel`, Redi `diff_hor`, the 3 ocean-FCT scatters, the
2 SSH scatters, the ice EVP ×2, the ice FCT ×3, the ice oce_fluxes net-subtract) and the `parallel_reduce`
REDUCTIONS (the CG dot-products, `integrate_nod_2D`) reassociate across threads/lanes. On pi the per-step
floor is ≲1e-12 (≈ FP), but **pi has zero ice** — the sea-ice scatters/reductions are exercised only on
CORE2 (forced), and they **compound over a multi-year run**. M3.2 measures whether that compounding stays
within the Fortran↔C budget (i.e. the GPU/OpenMP port is climate-faithful).

## Known baselines (from earlier milestones)

- **CUDA pi smoke** (single-step regime, ocean only — `memory/reference-cuda-eos-divergence.md`): density
  Δ≈3.18e-12 stable, Av/Kv≈0.095 isolated threshold-flips, u/v≈3.7e-4/5.9e-5, pgf≈8e-18, + M4.2's
  `eta_n`≈9.4e-11. No new divergence class through M4.2. **(pi = zero ice → says nothing about the ice scatters.)**
- **OpenMP pi** (M4.2 floor): T≈1.8e-15, Av/Kv≈2e-17, u/v/eta≈1e-18 — the ocean scatters/CG-dot. Again zero ice.
- **C ↔ Fortran 2-yr budget** (the absolute target — run `scripts/eps_climate_compare_2yr.py`, C
  `eps_2yr_dt1800` vs `fortran_pp_2yr`): _fill in the corr/bias/RMS per field here as the reference budget._

## The M3.2 runs

| backend | job | partition | run | time est. | output |
|---|---|---|---|---|---|
| CUDA   | `jobs/job_m32_cuda_core2` | gpu (8×A100/2 nodes, dist_8) | 2-yr CORE2 (34560 steps) | **measured 0.731 s/step** (job 25163175, post-M4) → **2-yr ≈ 7.0 h** (1-yr ≈ 3.5 h); fits the 12 h gpu wall | `/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda` |
| OpenMP | `jobs/job_m32_omp_core2`  | compute (2 nodes, 32 ranks × 8 threads) | 2-yr CORE2 | ~1–2 h (CPU; ⚠️ rebuild `build-omp` first — stale since M4.3c) | `/work/ab0995/a270088/port2/m32_omp` |

Both write monthly means `<var>.fesom.{1958,1959}.monthly.nc` (JRA55 1958→1959 rollover — confirm in the log).

**Perf note (post-M4 GPU step, job 25163175):** 0.731 s/step on dist_8 — only ~15% faster than M3.1's
0.86 s/step *despite* moving the CG (M4.2) + the whole ice step (M4.3) onto the device. The per-step time
is now dominated by the **host-staged halos** (device→host→MPI→host→device): the CG's ~90–127 iters/step
× its pp/rr exchanges, the EVP's 120 subcycles × uice/vice, the FCT's ~21 brackets — and moving the CG/EVP
to the device actually *added* PCIe round-trips for those halos (they were host↔host at M3.1). **GPU-aware
MPI to halo on-device is the M5 unlock**, not more kernel porting. (CORE2 is also small/bandwidth-bound,
M3.1 note.) M3.2 is about *fidelity*, not speed — but this is the headline for the M5 prompt.

### M5.1 result (GPU-aware MPI on-device halo) — measured 2026-05-27

GPU-aware MPI was **proven** (`openmpi/4.1.5-nvhpc-24.7`; the old `4.1.2` is `--without-cuda` and SEGFAULTs
on device pointers — [[reference-cuda-aware-mpi]]) and the device-pointer halo path built + validated
(L47). After flipping the CG + momentum + gm + ice-FCT brackets to the on-device exchange, CORE2 dist_8
(startup-free internal loop timer, 2 reps each):

| config (CORE2 dist_8, 8×A100/2 nodes) | s/step | vs host-staged |
|---|---|---|
| host-staged (`FESOM_HOST_HALO=1`, same binary) | 0.780 (0.7818 / 0.7779) | — |
| **device-halo** (GPU-aware MPI) | **0.716** (0.7166 / 0.7156) | **8.2% faster** |

**The payoff is modest (~8%), and that is itself the finding.** The win is the **large nod3D fields**
(~48 MB — the full-field PCIe sync was real); the **small nod2D hot loops (CG/EVP, ~1 MB) barely benefit**
(PCIe-cheap; the device path still fences). The *other* big nod3D halos (kpp `diffK`/`blmc`, eos pressure)
are **host-bound by `fesom_smooth_nod3D`** (a host op forcing the round-trip regardless) → not device-halo
candidates. So ~8% is near the ceiling for this optimization on this config; **the "halo-bound" premise was
only partly right.** Correctness: device-halo == host-staged at the CUDA run-to-run noise floor (no NEW
divergence — data path, not arithmetic). The next real GPU levers: port the kpp/eos **smoothing** to device,
batch/fuse the CG's ~1000 launches/step, or a **bigger mesh** to fill the A100s. (Validated; not bit-identity
— CUDA is non-deterministic run-to-run via the atomic scatters, L47.)

### M5.1b — bigger mesh (farc, 638k nodes × 48 lev ≈ 30 M 3-D pts, ~5× CORE2) — measured 2026-05-27

Tested the "CORE2 is too small / a bigger mesh fills the A100" hypothesis on **farc** (high-res, ~638k
nodes). De-risk: np=1 on one A100 reads + runs + **fits in ~8.3 GB** of 3-D fields (so any `dist_N` fits on
a 40 GB card), stable at dt=900, but **8.66 s/step with ~276 CG iters** (the stiff high-res SSH does ~2×
CORE2's CG iterations). Then host-staged vs device-halo (internal timer, 2 reps, dt=900, snap off):

| farc config | hardware | host-staged s/step | device-halo s/step | device gain |
|---|---|---|---|---|
| `dist_4` | 4 A100 / **1** gpu node | 7.28 | **6.38** | **+12.3%** |
| `dist_8` | 8 A100 / **2** gpu nodes | 3.73 | **3.30** | **+11.5%** |
| `dist_256` | 256 cores / **2** compute nodes (CPU, build-serial) | — | **0.445** | — |

Three findings:
1. **The device-halo gain GREW with the mesh: 8.2% (CORE2 dist_8) → ~12% (farc).** Bigger per-rank nod3D
   fields ⇒ the eliminated full-field PCIe sync moves more bytes. Confirms the L47 mechanism.
2. **The GPU strong-scales near-linearly: dist_4 → dist_8 = 6.38 → 3.30 = 1.93× on 2× GPUs.** The A100s are
   NOT saturated — adding GPUs helps (the CG iter count is partition-independent, so per-iter work halves).
3. **Node-for-node the CPU is still ~7.4× faster** (2 compute nodes 0.445 vs 2 gpu nodes 3.30) — **a bigger
   mesh did NOT flip the GPU↔CPU verdict.** ⚠️ **I first blamed the CG here — that was WRONG (see §M5.2):
   measured, the CG is only 1–5 % of the step.** The "16.5 ms/CG-iter" was a miscalculation (3.3 s ÷ 200 iters
   conflates the CG iterations with the *whole* step; the CG actually runs once/step at ~1.75 ms/iter and is
   a tiny slice). The real bottleneck is the **bulk ocean + sea-ice kernels** (~95–99 % of the step) — still
   to be profiled per-substep. (np=1 farc was 8.66 s/step, but that too is dominated by the bulk kernels, not
   the CG.)

**Conclusion:** chasing bigger meshes or more halo flips will NOT make the GPU competitive node-for-node. The
*why* was nailed down in §M5.2 below — and it is **NOT** the CG (my first guess). Jobs: `jobs/job_farc_gpu_np1`,
`jobs/job_gpuaware_time_farc_dist{4,8}`, `jobs/job_farc_cpu_dist256`.

### M5.2 — the CG is NOT the bottleneck (measured, 2026-05-27)

Added a startup-free **CG-share timer** (env `FESOM_CG_PROFILE=1`; fence+`MPI_Wtime` around the CG iteration
loop, accumulated over the timed window, printed by the loop timer) + applied two **bit-identity-preserving**
CG optimizations (fuse each SpMV with its following dot into one `parallel_reduce` — same row-order reduction
→ Serial np1+np2 still byte-identical to golden; fuse the sp0/sp1 dots into ONE 2-element `MPI_Allreduce`,
3→2 collectives/iter). farc dist_4 (CG-opt build):

| leg | s/step | **CG share** | CG iters/step | CG ms/iter |
|---|---|---|---|---|
| host-staged | 7.44 | **5.2 %** (0.38 s) | 219 | 1.75 |
| device-halo | 6.55 | **1.0 %** (0.067 s) | 219 | 0.305 |

**The CG is only 1–5 % of the step → it is NOT the bottleneck (this corrects §M5.1b).** Consequences:
- The CG fusions are correct + safe but **immaterial to the step** (the CG is tiny). The valuable artifact is
  the CG-share timer, which revealed the truth. (The 2 % apparent slowdown vs the m5.1 baseline is the
  timer's per-step fences — now gated behind `FESOM_CG_PROFILE`.)
- The device-halo makes the **CG itself 5.7× faster** (1.75 → 0.305 ms/iter — the per-iter cost was the 2
  host-staged nod2D halo syncs; at farc's bigger nod2D + 219 iters these are NOT cheap, unlike the CORE2
  finding). But since the CG is tiny, that's only ~⅓ of the device-halo's 0.89 s/step total gain; the other
  ⅔ is the ocean nod3D halos.
- **The real bottleneck is the bulk ocean + sea-ice kernels (~95–99 % of the step), un-profiled.** Likely
  suspects: the per-node/per-element TDMA solves (KPP, `impl_vert_visc`, vertical tracer diffusion, GM
  `fer_gamma`) — serial-within-column, low arithmetic intensity; the many FCT launches; the host-bound
  kpp/eos `fesom_smooth_nod3D`, the salinity floor, and the ice host parts. **Next: a per-substep timer in
  `fesom_step.cpp` (the CG-share pattern, applied to each substep) to find which kernels dominate** — that,
  not the CG, is the path to a GPU win.

### M5.3 — where the ocean's 82% goes: ~half kernels, ~half full-field PCIe (2026-05-27)

Per-phase step profiler (`FESOM_STEP_PROFILE=1`, `jobs/job_gpuaware_prof_core2`) on CORE2 dist_8
(device-halo, 0.738 s/step):

| phase | % of step |
|---|---|
| **ocean** (`fesom_timestep`) | **82.2%** |
| sea-ice | 7.4% |
| forcing | 5.3% |
| CG (within ocean) | 4.6% |
| coupling | 1.5% |

⇒ **the sea ice is NOT the GPU trouble (only 7.4%); the ocean dominates** (the Fortran/CPU "it's the ice"
intuition does not carry over — the ice is 2-D and cheap, the ocean carries the heavy 3-D physics).

`nsys` on np=1 CORE2 (`jobs/job_nsys_core2_np1`, per-kernel + memcpy) split the ocean's 82% ~half/half:
- **3-D physics KERNELS**: FCT tracer advection **~30%** (the biggest — ~20 kernels × 2 tracers), momentum
  ~15%, diffusion (Redi `diff_hor` + vertical TDMA `diff_ver`) ~14%, GM (`sigma_xy` 5.6% alone) ~10%,
  ALE `vert_vel` ~5%, KPP ~1%.
- **FULL-FIELD PCIe**: `cudaMemcpy` = **82.8% of CUDA-API time**, **~13 GB/step** at np=1 (copies up to
  188 MB = full elemvec fields) = the host-staged ocean halos NOT flipped in M5.1 (full-field `sync_host`/
  `sync_device`) + the EOS/KPP `smooth_nod3D` round-trips. **This is the removable cost.**

### M5.4 — lever A: flip the remaining ocean halos to device-halo (in progress)

The host-staged ocean halos use the M1.5 **split-rail** pattern: device kernel → `sync_host` (OUT) → host
`fesom_halo_exchange` → the consumer's `modify_host(); sync_device()` (IN re-push). Flipping each: replace
the OUT-rail + halo with `fesom_halo_field`, and **remove the downstream IN re-push** (the field stays
device-resident with its halo; else the re-push clobbers the device halo). Safe because: the per-kernel
verifies read the **raw alias** (not `h_checked()` → no SYNCCHECK assert), and on Serial every sync is a
no-op (host==device), so removing them is a no-op there. Validate EACH: Serial np1+np2 bit-identical +
CUDA pi dist_2 A/B at the run-to-run noise floor + re-time CORE2 dist_8.

**Progress (CORE2 dist_8, device-halo s/step; host-staged baseline ~0.77):**

| flipped | device s/step | total gain |
|---|---|---|
| M5.1: CG, momentum (`uvnode_rhs`/`u_b`/`v_b`), gm (`tr_xy`/`tr_z`), ice-FCT | 0.716 | 8.2% |
| M5.4a: + `uv_rhs` (elem3D, 3×/step, momentum substeps 4–6) | 0.658 | 14.3% |
| M5.4b: + `pgf_x`/`pgf_y` (elem3D) + `uvnode` (nod3D 2-comp) | 0.643 | 16.7% |
| M5.4c: + FCT internal halos (`fct_LO`/`tr_xy`/`fct_plus`/`fct_minus`, 8 exch/step) | 0.592 | 23.3% |
| M5.5a (lever B): `bvfreq` device smoother (`smooth_nod3D_kk`) + 4 re-pushes removed | 0.577 | ~25% |
| **M5.5b (lever B done): KPP `blmc` device smoother** (3 slabs × 3 sweeps, slab-offset) | **0.503** | **~30–35%** |

**Lever B (M5.5) — `smooth_nod3D` on the device (+ an I/O-staleness fix).** `fesom_smooth_nod3D_kk`
(`fesom_eos.cpp`): the node-patch smoother on-device (2 race-free kernels/sweep — gather then scale,
*separate* so `arr` is read-then-written, no race; + device-halo between sweeps). Serial bit-identical to the
C smoother. Wired `bvfreq`: device-halo + device-smooth + **removed the 4 consumer `bvfreq` IN re-pushes**
(GM/KPP/PP/mo_convect) → device-resident; ~2.5%. **M5.5b finished B with KPP `blmc`** (3 slabs × 3 sweeps;
added a `base` element-offset to `fesom_halo_exchange_device`/`fesom_halo_field`/`smooth_nod3D_kk` so a slab
exchanges/smooths on its own) → **dev 0.577 → 0.503 = ~13%!** The host `smooth_nod3D` ran blmc's **9 sweeps
single-threaded on the host** (the GPU build's Serial host) — a large hidden cost that **`nsys` never saw**
(GPU-kernel-only) but the phase profiler counted inside the ocean's 82%. So **lever B ≈ 15% total** (bvfreq
2.5% + blmc 13%) — much more than the nsys kernel view (KPP ~1%) implied, because the dominant cost was
host *compute*, not a GPU kernel. **Cumulative device-halo + smoother: M5.1 0.716 → 0.503 s/step (~30% on
the device path; ~35% vs the original ~0.78 all-host).** Lever B DONE. ⚠️ **I/O-staleness gotcha it caught (an
M5.4b regression):** flipping a field that is *also a snapshot output* (`pgf`, `bvfreq`) removes its OUT-rail
`sync_host`, so it's DEVICE-authoritative at I/O time and the gather (`h_checked()`) reads the **stale host
copy** on CUDA — the *model* is correct (device-resident), only the *diagnostic output* is wrong. Fix: a
pre-I/O `sync_host` of those fields, gated on the snapshot step (`fesom_main.cpp`). Missed in M5.4b because
the A/B grepped a subset → **always diff ALL output fields** (the Serial gate + SYNCCHECK can't catch it: on
Serial `fesom_halo_field` leaves the field synced).

**host-staged 0.772 → device-halo 0.592 = 23.3%** (the clean self-contained + split-rail flips, each
Serial np1+np2 bit-identical + SYNCCHECK-clean + CUDA A/B at the noise floor). Two flip patterns proven:
**self-contained** brackets (CG, FCT internal — replace wholesale with `fesom_halo_field`) and **split-rail**
(`uv_rhs`/`pgf`/`uvnode` — replace OUT-rail+halo, remove the downstream IN re-push). Diminishing per-flip at
dist_8 (small per-rank fields); the high-frequency fields dominate (`uv_rhs` 3×/step, FCT 8×/step gave ~8%
each; `pgf`+`uvnode` 1×/step gave ~2%). **Remaining** (lower/uncertain payoff): the **GM chain** (8 halos —
its device kernels read *owned*, so most halos are verify-only on the device path → flipping gives no device
benefit unless a later substep [Redi] reads them at halo; needs per-field tracing — only `fer_gamma` is
clearly read-at-halo), `ssh_rhs` (nod2D, small — CG reads it owned), tracer-diff; **NOT** EOS/KPP
(smoothing-bound — need `smooth_nod3D` on device first). On a bigger mesh (farc) each flip pays more (the
M5.1 8%→farc 12% scaling).

### M5.6 — per-substep timing diagnostics through the WHOLE model (2026-05-27)

The §M5.3 phase profiler stopped at 4 coarse phases; §M5.5 then found a hidden +13% host-compute cost
(`blmc`) that *no GPU profiler saw* because it was host *compute*, not a kernel. To find the rest of those,
`fesom_profile.{hpp,cpp}` (`FESOM_STEP_PROFILE=1`) now marks **every ocean substep** (`fesom_step.cpp`,
`PMARK`) **and every ice phase** (`fesom_ice.cpp`, `PMARK_ICE`) — fence-bounded host+device wall, gated so
production/Serial are unchanged. Full breakdown, CORE2 dist_8, device-halo + smoother (**0.511 s/step**):

| phase | % loop | s/step | what |
|---|---|---|---|
| **13_fct** | **17.4%** | 0.0889 | FCT tracer advection — heavy kernels (~24 launches), the single biggest |
| **3_mixing** | **11.0%** | 0.0563 | KPP/PP — **kernels ~1% → ~10% is HIDDEN host-staged** (`diffK`/`viscA`/`ghats` split-rail halos) |
| 1b_gm | 9.9% | 0.0506 | GM chain (`fer_gamma` TDMA + sigma/Redi) |
| 7_ssh | 9.1% | 0.0463 | SSH RHS + CG (CG itself 5.9% per the CG-share timer) |
| **ice_dyn (o2i+EVP)** | **8.8%** | 0.0452 | **the entire ice cost (10.1%) is here** — EVP 120 host-staged subcycle halos + host coastal-BC |
| 12_ale | 6.7% | 0.0344 | ALE thickness + `vert_vel` scatter |
| 13b_trdiff | 6.1% | 0.0312 | implicit vertical tracer diffusion (per-node TDMA) |
| 4_velrhs | 5.3% | 0.0272 | `compute_vel_rhs` + `momentum_adv` |
| 6_ivisc | 5.1% | 0.0261 | `impl_vert_visc` (per-elem TDMA) |
| 1_eos | 3.3% | 0.0167 | EOS + density smoothing |
| 5_viscfilt | 1.5% | 0.0079 | `visc_filt_bidiff` |
| 13c_bolus+14 | 1.2% | 0.0059 | bolus subtract + step commit |
| ice_thermo+flux | 0.7% | 0.0036 | column thermodynamics + `oce_fluxes` |
| ice_fct | 0.5% | 0.0026 | ice FCT (2-D, cheap) |
| 2_pgf | 0.4% | 0.0022 | pressure-gradient force |
| ice_hdiag | 0.04% | 0.0002 | h_ice/h_snow diagnostics |

**Two blmc-like hidden host-time sinks found "through the model"** (the user's ask — where we *unnecessarily*
spend time):
1. **`3_mixing` ~10% host-staged.** The KPP device kernels are ~1% (nsys/§M5.3 confirmed), so the other ~10%
   is the `diffK`/`viscA`/`ghats` split-rail halos still going host. **Flippable now** with the slab-offset
   device-halo (exactly the `blmc`/`bvfreq` recipe) — the highest-payoff remaining lever-B flip.
2. **`ice_dyn` (EVP) 8.8%** — *all* of the ice. The EVP runs **120 subcycles on a host loop**, each staging
   `uice`/`vice` halos host + a host coastal-BC (needs `partit->myList_edge2D`). This is the ice analogue of
   the CG: a host-loop-over-device-kernels with a per-iteration host halo. Flipping the subcycle halo to
   device-halo (and folding the coastal-BC into the bracket) is the ice equivalent of lever A/B.

Everything below `1b_gm` is either a heavy kernel (lever C — FCT/momentum/trdiff/ivisc, memory-layout work,
separate branch) or already small. So the **actionable remaining wins are #1 (KPP host halos, lever B) and
#2 (EVP host halos)** — together ~19% of the loop currently spent host-staging, both removable with the
proven device-halo. `13_fct` (17.4%) is the largest but it is genuine device kernels → lever C.

### M5.7 — flip the KPP/mixing host-staged halos (the §M5.6 ~10% finding) — DONE (2026-05-27)

The §M5.6 profiler found `3_mixing` = 11.0% with KPP *kernels* ~1% → ~10% host-staged. Two halo clusters
carried it, both now on the device-halo:
- **M5.7a — KPP-internal `combine` (`fesom_kpp.cpp`):** `diffK` (2 slabs) + `viscA` were sync_host →
  host-exchange → re-push; flipped to `fesom_halo_field` (slab-offset, the M5.5b `blmc` recipe). They are
  read at HALO right below (the node→elem `viscAE` average reads `viscA` at element vertices that may be
  halo nodes; the single-Kv `deep_copy` reads `diffK` slab-0). `ghats` stays HOST (line 1572: gated off in
  CORE2, not read on device after) → its one sync_host + host exchange remain.
- **M5.7b — driver `Kv`/`Av` (`fesom_step.cpp`):** made `Kv`/`Av` **device-resident across KPP → mo_convect
  → ivisc (substep 6) → trdiff (substep 13b)**. Removed: the KPP OUT sync_host, the mo_convect IN re-push,
  the mo_convect OUT + host exchange (→ device-halo `Kv` NOD3D / `Av` ELEM3D), the ivisc Av re-push, the
  trdiff Kv re-push. The PP branch (opt-in) flipped to match. The verify sync_host calls are now guarded by
  their verify flags (Serial host==device → the gate still reads correct values). **⚠️ Kv/Av are snapshot
  outputs → added the L48 pre-I/O `sync_host` (gated on the snapshot step, `fesom_main.cpp`).**

**Result (CORE2 dist_8): device-halo 0.503 → ~0.476 s/step (~5.4%); `3_mixing` 11.0% → 6.15%** (0.0563 →
0.0296 s/step — nearly half the substep, the host-staged part, gone; the residual 6.15% is the genuine KPP
kernels + the kept `ghats` sync + the IN-syncs). Validation: Serial np1+np2 **bit-identical** + KPP verify
**max|Δ|==0** + SYNCCHECK **0 guard-hits** + CUDA A/B **at the run-to-run noise floor** — host-vs-dev ==
dev-vs-dev' to machine ε (Kv/Av the flipped fields: 1–4e-17 == the dev-vs-dev' floor 0.9–2.7e-17; T/u/w
identical to the floor). The L48 Kv/Av snapshot staleness is ruled out: the all-fields A/B shows Kv/Av at
the noise floor (a staleness bug would be O(field) ≈ 1e-3). `jobs/job_gpuaware_noisefloor_pi` (the dev-vs-dev'
reference) added.

Running totals (CORE2 dist_8 device path): M5.1 0.716 → M5.4c 0.592 → M5.5b 0.503 → **M5.7 ~0.476 s/step**
(~33% vs M5.1's device-halo start; ~39% vs the original ~0.78 all-host). **Next: EVP** (`ice_dyn` 8.8–9.7%,
the §M5.6 finding #2 — 120 host-staged subcycle halos + host coastal-BC).

### M5.8 — flip the EVP subcycle's 120× host-staged halo + coastal BC (finding #2) — DONE (2026-05-27)

`ice_dyn` was 8.8–9.7% = the entire sea-ice cost, dominated by the EVP: a host loop over
`evp_rheol_steps` (120) subcycles, each staging `uice`/`vice` host (`sync_host` → host coastal BC →
host `fesom_exchange_nod2D` → `sync_device`) = **4 PCIe copies × 120 = 480 small copies/step**, the
ice analogue of the CG. Flipped (`fesom_ice_evp.cpp`):
- **Coastal BC → device kernel.** The host BC zeros `uice`/`vice` at the endpoints of OWNED
  open-boundary edges (`gid = myList_edge2D > edge2D_in` — the rank-boundary-safe criterion). Since
  `myList_edge2D` is host-only, precompute ONCE a per-node 0/1 device mask (`evp_coastal_mask`, built
  from the host `myList_edge2D`+`edges`+`edge2D_in`); the BC kernel zeros the marked nodes. All writes
  are 0 → **idempotent, race-free, no scatter → Serial AND OpenMP bit-identical.**
- **Halo → device-halo** (`fesom_halo_field`, NOD2D). `uice`/`vice` stay device-resident across the
  subcycle; the existing post-loop `sync_host` (`fesom_ice.cpp:491`) pulls them once for FCT / I/O /
  the next step — so **nothing downstream changes** (no L48 risk, no cross-step risk).
- **⚠️ Cleanup gotcha:** the cached mask is a `Kokkos::View` → a function-`static` View destructs
  during teardown *after* `Kokkos::finalize()` and **aborts** (SIGABRT — caught it the first run). Fix:
  file-static + `fesom_ice_evp_free()` called before `Kokkos::finalize()` (the `fesom_halo_device_free`
  rule). Same trap for any cached device View.

**Result (CORE2 dist_8): device-halo ~0.476 → ~0.464 s/step; `ice_dyn` 9.7% → 7.90%** (0.0465 →
0.0372 s/step). Smaller than KPP because the remaining 7.9% is the **120 subcycles × ~6 kernel
launches = ~720 launches/step** (launch-latency bound, not PCIe) — that's lever-C (kernel fusion).
Validation: **Serial dist_16 all 5 ice keys bit-identical, `evp` 960 lines max|Δ|==0**, clean exit;
CUDA host-vs-dev **byte-identical to the M5.7 binary** (the EVP flip adds *zero* divergence — proven
by stashing M5.8 and re-running the A/B). `jobs/job_gpuaware_validate_core2` (CORE2 A/B) added.

### ⚠️ M5.8-side discovery: a PRE-EXISTING CORE2 host-vs-dev OCEAN divergence (needs investigation)

The first-ever **CORE2-active-ice** CUDA A/B (host-halo vs device-halo) revealed a large divergence —
but it is **NOT from M5.8** (the M5.7 binary shows it byte-for-byte identical) and **NOT in the ice**
(uice 27× the floor) — it is in the **OCEAN device-halo path**, latent since M5.1:

| field | host-vs-dev @ step10 | dev-vs-dev' floor @ step10 | ratio |
|---|---|---|---|
| T | 1.6e-1 | 2.0e-4 | ~800× |
| Kv | 2.2e-1 | 1.7e-5 | ~13000× |
| u | 5.1e-2 | 1.1e-4 | ~460× |
| uice | 1.6e-2 | 5.9e-4 | ~27× |

It was never caught because **CORE2 was only timing-profiled, never A/B'd** — the M5.1–M5.7 fidelity
gate was **pi dist_2** (1 node, no ice), where host-vs-dev is 1e-17. The divergence is **reproducible
across separate SLURM jobs** (identical argmax/values) → deterministic, not random run-to-run noise.

**Localised (2026-05-27):**
- **NOT inter-node IB.** pi dist_2 forced onto **2 separate nodes** (1 GPU/node → GPU-aware MPI over
  inter-node IB, `jobs/job_gpuaware_validate_pi_2node`) is at the noise floor (T 1.2e-14, Kv 1.1e-17)
  — identical to 1-node pi. So the device-halo's inter-node GPU-aware MPI is byte-correct.
- **NOT the ice / NOT M5.8** (ocean fields dominate; M5.7 binary byte-identical).
- ⇒ **scale/dynamics-dependent**: pi (≈3k nodes, weak idealised dynamics) is clean at any node count;
  CORE2 (≈126k nodes, realistic ice+ocean, strong gradients) diverges. The remaining hypotheses:
  **(b)** the host path's extra `sync_host`/`sync_device` deep-copies perturb **atomic-scatter** thread
  scheduling → different-but-valid scatter orderings, whose seed grows with the number of colliding
  `atomic_add`s (tiny on pi, large on CORE2) and amplifies chaotically — benign; **(c)** a device-halo
  pack/unpack correctness issue that only bites at CORE2 halo size — real. To split (b)/(c): dump a
  flipped field's halo device-vs-host on CORE2 step 1 (if byte-equal → (b); else (c)), and/or build
  with all ocean halos host-staged (`FESOM_HOST_HALO=1` is exactly that) — which is the A/B's host leg.

**The designed arbiter is the M3.2 climate run** (in flight, **also CORE2 dist_8 = 2 nodes**, the same
path): PASS (CUDA-vs-C ≤ C-vs-Fortran, drift≈0) ⇒ benign (b); drift ⇒ real (c). **Orthogonal to the
KPP/EVP halo-flip work — warrants a focused session.**

**ROOT CAUSE (2026-05-27, hypothesis (c) confirmed — it is a real bug, NOT benign):** the device-halo
**exchange is byte-perfect** — a per-call self-check (`FESOM_HALO_SELFCHECK=1`, `fesom_halo_device_selfcheck`:
device exchange + host exchange on the SAME owned data, diff the halo) found **0 mismatches** across all
kinds on CORE2 (multi-neighbour, inter-node). The bug is **stale HOST copies**: the M5.1+ flips removed the
OUT-rail `sync_host`, leaving fields **device-authoritative (host stale)**; a remaining **host** operation
(salinity floor / eta_n map / reductions / coupling — the per-step CPU compute) reads the **raw stale host
copy**, injecting a per-step perturbation that amplifies chaotically. **Proof:** vs the Serial oracle
(bit-identical to C) at step 20 — device-halo **host-stale** (shipped) T=**0.41**; device-halo **host-fresh**
(self-check syncs every call) T=**1.4e-3** == the all-host-halo path. Keeping host fresh collapses 0.41→1e-3.
Invisible everywhere it was tested: pi (1e-17, idealised/no-ice), Serial (host==device), the host path +
self-check (both `sync_host`). `SYNCCHECK` did NOT trip on CUDA → the stale read is a **raw `h()`** (an
unguarded CPU read), not an `h_checked()` halo/IO entry. **Severity:** the shipped GPU path is wrong on
CORE2; the in-flight M3.2 run is on it → re-run after the fix. **FIX (decided): bisect the M5.x flips vs the
Serial oracle on CORE2 to find the first flip that breaks fidelity, then restore a *targeted* `sync_host`
only for the host-read field (preserves the perf).** Tools added: `fesom_halo_device_selfcheck` +
`jobs/job_halo_selfcheck_core2` (exchange byte-check), `job_core2_serial_ref` (the CORE2 Serial oracle),
`job_gpuaware_validate_core2` (3-way A/B), `job_gpuaware_validate_pi_2node` (inter-node localiser),
`job_dbg_{serial,dev,host}_step1` + `job_dbg_selfcheck20` (step-1 onset + the host-fresh proof).

### M5.9 — bisecting the stale-host culprit (IN PROGRESS, GPU-blocked 2026-05-27)

Bisect the M5.x flips vs the CORE2 Serial oracle (`serref_core2`, dist_8, dt=1800, 20 steps; dev-vs-Serial
@ step20: clean ≈1e-3, broken ≈0.1–0.9). Each point: `git checkout <commit>` → build-cuda →
`/work/.../bisect/run_dev_core2.sh` → `diff_snap` vs serref.

| commit | flips added | dev-vs-Serial T@20 | verdict |
|---|---|---|---|
| **M5.1** `d6b0a1f` | CG, momentum(uvnode_rhs/u_b/v_b), gm(tr_xy/tr_z), ice-FCT | **1.2e-3** | ✅ clean |
| M5.4a `4120d02` | uv_rhs (ELEM3D nc=2) | — | ⛔ crash-blocked |
| M5.4b `bfa71ff` | pgf + uvnode | — | ⛔ crash-blocked |
| M5.4c `d3e0726` | FCT internal halos | — | ⛔ crash-blocked |
| **M5.5a** `0b329e3` | bvfreq device smoother | **0.41** | ❌ broken |

⇒ **the stale-host bug entered in M5.4a–M5.5a** (5 candidate flips). ⚠️ **GPU obstacle:** the device-ptr
GPU-aware-MPI path went **~100% segfault** (`invalid permissions for mapped object`, `knem/cma not
available`) for a stretch — **14 consecutive** dist_8 crashes across many nodes (NOT one bad node: `l50103`
ran M5.1 clean then crashed M5.4a). Transient partition flakiness, independent of the (deterministic)
divergence; it blocks the M5.4a/b/c runs. Resume the bisect when the partition settles.

**Ruled out by static analysis (so the culprit is subtle):** (1) no host op reads a flipped field — the ice
(`ocean2ice` reads `dyn->uv`/T/S/`hbar`, none flipped), the eta_n map (`hbar`/`hbar_old`), the salinity floor
(owned `S`, synced by trdiff), the reductions (diagnostic, read-only); (2) no surviving
`modify_host()+sync_device()` re-push of a flipped field at HEAD (the flips removed them); (3) the device
exchange is byte-perfect (self-check). Remaining mechanisms: a flipped field's **halo region** read on host,
or a `sync_device` of a host-written-owned field clobbering the device **halo** with a stale host halo.
**Finish plan:** (a) resume the GPU bisect (M5.4a→clean? then M5.4b/c) when the partition cooperates; once the
commit is known, the culprit is one field → restore its targeted `sync_host` (keeps perf). An instrumented
build (per-field "device-halo'd this step, then host-touched" tracker) would pinpoint it without the flaky
GPU. **Proven-correct fallback: `sync_host` in the `fesom_halo_field` device path → 1e-3** (the
`FESOM_HALO_SELFCHECK` run), at one PCIe copy/halo (still < the original 2-copy host-staging).

### M5.9 — FIXED + a permanent fidelity GATE (`6ba27e9`, 2026-05-27)

The M5.4a/b/c commits hit an intermittent device-ptr GPU-MPI crash that blocked the git-bisect, so the
culprit was localised **at HEAD** (which runs) with a `FESOM_DBG_SYNC` env-gated per-field `sync_host` +
a **leave-one-out** sweep: **all four of `bvfreq`, `pgf_x/y`, `uvnode`, `uv_rhs` must be host-fresh** —
each, when left device-stale, alone decorrelates the chaotic CORE2 run to the same ~0.4 attractor by
step 20 (so the device exchange is fine, the *host data* is read stale; the halo already fences, ruling
out a race). **FIX: `sync_host()` after each of those 4 device halos** (`fesom_step.cpp`; no-op on
Serial → the bit-identity oracle is untouched). Result: **CORE2 dev-vs-Serial @ step20 T 0.41 → 1.1e-3**
(= the host-halo path / the CUDA climate-close floor) + Serial np1 still BIT-IDENTICAL to golden. The
shipped GPU path is now as close to byte-identical as CUDA allows (the 1e-3 residual is the inherent
atomic-scatter + fmad non-determinism, same as the host path).

**Final perf (CORE2 dist_8, clean timing, no I/O, the M5.9-fixed binary):** device-halo **0.493 s/step**
vs host-staged **0.812** = **39.3% faster (1.65×)**; vs pre-M5 (M4 = 0.731) **32.6% faster**. The
correctness fix cost only **+6.2%** (the broken 0.464 → 0.493 = the ~5 `sync_host`/step) — so the M5
device-halo campaign nets a **correct ~39% speedup** on this config. Follow-ups: pin the exact host-reader
(to optimise to a minimal/earlier sync and claw back the 6%) + the M5.4a/b/c bisect verdicts.

**THE GATE (run before committing ANY device-halo / sync-rail / device-residency change):**
`./scripts/gpu_fidelity_gate.sh` → builds the build-serial CORE2 oracle (bit-identical to C),
runs build-cuda CORE2 dist_8 (device-halo, ICE ACTIVE), and `scripts/gpu_fidelity_check.py` asserts
dev-vs-Serial ≤ the per-field CUDA-floor ceiling (PASS ~1e-3, FAIL ~1e-1). **pi is INSUFFICIENT — it has
no ice + idealised dynamics, so this whole class of bug stays at ~1e-17 and is invisible there; CORE2 was
only ever timed, never accuracy-checked, which is how M5.1–M5.8 shipped the regression.** Jobs:
`job_core2_serial_ref` (oracle) + `job_gpu_fidelity_dev` (CUDA leg).

### M5.9-pin — the host reader is `uvnode`→bulk ALONE; the other 3 syncs were placebos (session 20, 2026-05-28)

The M5.9 fix `sync_host`'d **four** device-halo'd fields after their halo (`bvfreq` :207, `pgf_x/y` :311,
`uvnode` :331, `uv_rhs` :488), on a leave-one-out (`FESOM_DBG_SYNC`, toggle each on/off) that found all
four required. **Pinning the actual reader showed only ONE is real.**

**Static trace (CUDA path).** `bvfreq` (KPP/mo_convect/GM + the device smoother), `pgf` (compute_vel_rhs),
`uv_rhs` (impl_vert_visc, compute_ssh_rhs) are read **only by device kernels** (device-resident with their
halos); their only host readers are the read-only min/max print + the netCDF snapshot, both covered by the
snapshot-gated pre-I/O `sync_host` (L48). `uvnode` alone has a per-step **host** reader: `fesom_bulk_compute`
(`fesom_main.cpp:1027`, the JRA55 bulk wind-stress formula — wind relative to ocean surface current, read at
`uvnode[2*(n*nl)]` over owned+halo, **surface only**). That call exists only under `use_jra` (CORE2), never
on pi (analytical stress) — **exactly the pi-invisibility / active-forcing signature.**

**The discriminator (NaN-poison).** A leave-one-out is confounded on a chaotic backend: removing a `sync_host`
removes a device FENCE, which reshuffles atomic/launch ordering enough to slide the CUDA run to the same ~0.4
attractor *without any stale read*. So instead: **keep each `sync_host` (device scheduling byte-identical to the
fixed run) and overwrite the host copy with NaN AFTER it** (no `modify_host` → device stays correct; only a
genuine HOST read sees the NaN). `FESOM_POISON_HOST={bvfreq,pgf,uvnode,uvrhs}`, `jobs/job_poison_dev`. Three
CORE2 dist_8 (ICE active) runs vs the Serial oracle:

| run | host-poisoned | model state @ step 20 | outcome |
|---|---|---|---|
| clean | — | u/v vs oracle ~1.1–1.7e-4 | completes (floor) |
| poison_others | `bvfreq,pgf,uv_rhs` | u 5e-5 / v 1.7e-4 / bvfreq 3.8e-7, **0 NaN** | **completes — model identical** |
| poison_uvnode | `uvnode` | — | **CRASH: `CG_kk: pp·App = nan`** (NaN stress→momentum→CG) |

So NaN in `bvfreq`/`pgf`/`uv_rhs` host copies is inert (the netCDF snapshot was even healed by the pre-I/O
sync; only the cosmetic min/max print showed it) — **they have no model-feedback host reader**. NaN in `uvnode`
blows the model up via bulk → these were the ~0.4 *fence-chaos* placebos, not stale reads.

**Fix (`fesom_step.cpp`):** delete the `bvfreq`/`pgf`/`uv_rhs` per-step `sync_host`; keep `uvnode`'s. `uv_rhs`
has NO host reader at all (not even a snapshot field). `bvfreq`/`pgf` snapshot+print diagnostics stay correct
via the existing L48 pre-I/O sync (the per-step min/max print may read a stale device-resident value — cosmetic,
like `Kv`/`Av` already do).

**Validation.** GATE PASS (drop-3 and a tried surface-`uvnode` variant both): worst field ~4–5e-3, T ~1.1e-3,
no NaN, no crash — at the CUDA climate-close floor. Serial pi np1 + np2 BIT-IDENTICAL to the C golden (the
dropped/kept syncs are no-ops on Serial; `FESOM_KK_VERIFY` unaffected — no kernel changed). **Perf (CORE2
dist_8, clean no-I/O, same nodes l40360/69, 25 timed steps, two runs): all-syncs 0.4931 → fix (drop 3, keep
full `uvnode` sync) 0.4777 s/step = 3.1% recovered** — the measured cost of the 3 placebo PCIe copies (4
fields: `bvfreq`+`pgf_x`+`pgf_y`+`uv_rhs`). That is near the achievable ceiling: `uvnode` genuinely needs a
sync, so the full M5.9 +6.2% was never fully recoverable. A surface-only `uvnode` refresh (bulk reads only
nz=0) reached 0.4752 (3.6%) but the extra ~0.5% did not justify its custom pack kernel + per-step `Kokkos::View`
scratch alloc + host unpack (which nearly cancel the ~94 MB PCIe saving); a persistent-buffer version is a
future micro-opt. Tooling: `jobs/job_poison_dev` (the NaN discriminator), `jobs/job_perf_compare` (same-node
all-syncs-vs-fix timing). See lessons L49/L50. ⚠️ The GPU-partition device-ptr UCX segfault (L47) hit a perf
run mid-campaign — just re-run (it is transient, not a code bug).

## The comparison

`scripts/m32_climate_compare.py <backend_dir> --label <CUDA|OpenMP>` — annual-mean surface stats
(corr / bias / RMS / |Δ|max) per field per year + a year-to-year DRIFT check, vs **two** references:
- **vs the C-port** (`eps_2yr_dt1800` == Serial == bit-identical): this diff IS the backend's
  scatter/reduce drift, isolated from the Fortran↔C budget. **This is the headline M3.2 metric.**
- **vs Fortran** (`fortran_pp_2yr`): the absolute climate budget (backend-vs-Fortran should ≈ C-vs-Fortran).

## Verdict criteria (PASS)

- corr ≈ 1.0 for sst/sss/ssh/a_ice/m_ice (and uice if present), both years;
- backend-vs-C bias/RMS **bounded and clearly ≤ the C-vs-Fortran budget** (the scatter drift is below the
  model's own Fortran↔C uncertainty);
- **DRIFT ≈ 0** (year-2 bias ≈ year-1 bias) — the scatter perturbation does not run away over the years;
- no NaN/blow-up (T/S ranges bounded in the run log).

## Results

### CUDA 1-yr 1958 (2026-05-28, M5.9-pin binary `05182aa`, CORE2 dist_8)

Run dir: `/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_1yr_pin/`
References: canonical KPP (`fortran_kpp_5yr_fix` + `kpp_2yr_rebase`) — see `docs/REFERENCE_RUNS.md`.

```
================= CUDA-KPP vs Fortran-KPP =================
field  year |     corr         bias         RMS     |d|max
sst    1958   |  1.00000  +3.1472e-05  1.4541e-02  3.131e-01
sss    1958   |  0.99996  -5.3230e-04  2.6202e-02  1.976e+00
ssh    1958   |  1.00000  +2.0300e-05  1.0136e-03  2.387e-02
a_ice  1958   |  0.90662  -1.0303e-01  1.7039e-01  6.561e-01
m_ice  1958   |  0.98244  -1.0228e-01  1.6718e-01  8.337e-01
uice   1958   |  0.85019  -1.0131e-03  2.7949e-02  2.275e-01

================= CUDA-KPP vs C-port-KPP =================
field  year |     corr         bias         RMS     |d|max
sst    1958   |  1.00000  +1.0415e-04  1.4065e-02  3.140e-01
sss    1958   |  0.99996  -1.7994e-04  2.6125e-02  1.970e+00
ssh    1958   |  1.00000  -1.4326e-05  9.0499e-04  2.380e-02
a_ice  1958   |  0.99997  +1.6348e-04  2.8583e-03  1.777e-01
m_ice  1958   |  0.99998  -1.5053e-04  3.5701e-03  1.481e-01
uice   1958   |  0.99978  -7.4589e-05  5.3412e-04  1.125e-02
```

CUDA-vs-C bias on the ocean fields is O(1e-4)°C SST / O(1e-3) PSU SSS — the pure GPU scatter/reduce
drift (D22). Ice fields show O(1e-4) bias / corr ~0.99998 — the per-subcycle EVP scatter compounding
stays bounded over 17280 steps.

> **⚠️ Correction (2026-05-30):** the `a_ice/m_ice`-vs-Fortran numbers in the table above (corr ~0.91/0.98,
> bias ~0.1) were a **stale old-script ice-mask-averaging artifact** — they predate the `nan_to_num`-per-month
> fix (`466ea3e`, [[feedback-ice-mask-averaging]]). Re-running the **current** script on this same run dir gives
> a_ice/m_ice-vs-Fortran **0.99997** (see § M5.13 climate validation, the apples-to-apples table). So there is
> **no** genuine C↔Fortran gap on ice *concentration*; the one genuine, irreducible C↔Fortran ice budget is on
> **uice** (ice velocity, corr 0.85 — both scripts, GPU-independent since uice-vs-C = 0.99978).

DRIFT not computed (single-year run). Bias maps: `docs/m32_bias_maps/bias_*_1958_kpp.png`.

### M3.2 paradox postmortem (the original "−0.236 °C" finding)

The first M3.2 comparison reported CUDA-vs-C SST bias of **−0.236 °C**, contradicting the
backend-vs-C ≤ backend-vs-Fortran assumption. Root cause: the `m32_climate_compare.py` defaults
pointed at deprecated references with TWO physics-config deltas vs the CUDA-KPP run (PP vs KPP
**and** ice_gamma_fct 0.25 vs 0.5). After repointing to the canonical KPP refs the bias collapsed
to +1.0e-4 °C — 3 orders of magnitude. Full investigation in `docs/m32_bias_investigation.md`;
reference catalog in `docs/REFERENCE_RUNS.md`.

### OpenMP 2-yr 1958–1959 (2026-05-28, M5.9-pin binary `05182aa`, CORE2 16r×8t × 2 nodes)

Run dir: `/work/ab0995/a270088/port2/m32_omp/` (job 25211500).
References: canonical KPP (`fortran_kpp_5yr_fix` + `kpp_5yr_fix`).

```
================= OpenMP-KPP vs Fortran-KPP =================
sst    1958   |  1.00000  -4.6142e-04  1.2644e-02   sst    1959   |  1.00000  +3.5269e-04  2.5301e-02
sss    1958   |  0.99995  -7.5258e-04  2.7814e-02   sss    1959   |  0.99999  -1.5755e-03  1.1194e-02
ssh    1958   |  0.99999  +3.8482e-04  2.9970e-03   ssh    1959   |  0.99999  +4.7585e-04  2.1700e-03
a_ice  1958   |  0.99997  +6.1640e-05  2.8890e-03   a_ice  1959   |  0.99996  -4.7026e-05  3.2510e-03
m_ice  1958   |  0.99996  -1.5449e-04  5.4855e-03   m_ice  1959   |  0.99995  -8.2470e-04  7.9293e-03
uice   1958   |  0.84948  -7.9896e-04  2.8003e-02   uice   1959   |  0.85156  -3.2935e-03  2.7208e-02

================= OpenMP-KPP vs C-port-KPP =================
sst    1958   |  1.00000  -3.8874e-04  1.1987e-02   sst    1959   |  1.00000  +1.0116e-03  2.1915e-02
sss    1958   |  0.99995  -4.0022e-04  2.7681e-02   sss    1959   |  0.99999  -9.9095e-04  1.0343e-02
ssh    1958   |  0.99999  +3.5020e-04  2.9655e-03   ssh    1959   |  1.00000  +4.0126e-04  2.0170e-03
a_ice  1958   |  0.99997  +9.0591e-05  2.8376e-03   a_ice  1959   |  0.99996  -8.1856e-05  3.1845e-03
m_ice  1958   |  0.99998  +7.9883e-05  3.9429e-03   m_ice  1959   |  0.99998  -5.5307e-04  4.5539e-03
uice   1958   |  0.99995  -4.3172e-06  2.6270e-04   uice   1959   |  0.99949  +3.5984e-05  7.3150e-04
```

DRIFT (year2 bias − year1 bias; bounded ⇒ scatter compounding does not run away):
```
              vs Fortran    vs C-port
  sst         +8.1e-4       +1.4e-3
  sss         -8.2e-4       -5.9e-4
  ssh         +9.1e-5       +5.1e-5
  a_ice       -1.1e-4       -1.7e-4
  m_ice       -6.7e-4       -6.3e-4
  uice        -2.5e-3       +4.0e-5
```

### Verdict (OpenMP leg) — **PASS**
- corr ~1.0000 on every field × year × ref. Same scatter-drift floor as CUDA: O(1e-3 to 1e-4) bias
  on ocean, O(1e-3) on ice.
- All DRIFT deltas O(1e-3) or smaller — the thread-reassociated reductions stay climate-close
  over 34560 steps × 16 ranks × 8 threads; no runaway.
- The `uice` corr ~0.85 vs Fortran mirrors the CUDA leg exactly — it's the C↔Fortran-at-KPP
  physics budget on ice drift (already validated by the C-port at `375f3eb`), not a port artifact.

### M3.2 — both backends PASS
Both the CUDA 1-yr 1958 leg (above) and the OpenMP 2-yr 1958–1959 leg confirm climate-close
fidelity against the canonical KPP references. The Kokkos port, as of M5.9-pin (`05182aa`),
reproduces the C-port-KPP at the scatter-drift floor on every gate we have.

### Verdict (CUDA leg) — **PASS**
- corr ~1.0000 on all ocean fields; corr 0.99997 on ice (vs C-port-KPP).
- backend-vs-C bias (~1e-4 °C SST) ≪ C-vs-Fortran budget (~0.1 on ice fields).
- no NaN/blow-up; T/S bounded across the run.

---

## §M5.13 — the NG5 device-residency campaign (a–f): every flip passed the CORE2-active-ice gate

Each milestone ran the mandatory `scripts/gpu_fidelity_gate.sh --fresh-oracle` (CORE2 dist_8, ICE
active, CUDA device-halo vs the rebuilt Serial oracle) with `FESOM_STEP_PROFILE=1` (→ the deep_copy
PCIe proxy from the same job). All passed at the CUDA climate-close floor (worst field |Δ| well under
its ceiling; ice `h_ice` ceil 1e-1, T ceil 1e-2):

| milestone | flip | gate worst \|Δ\| | deep_copy calls/step | MB/step |
|:---|:---|---:|---:|---:|
| (baseline) | — | — | 207.7 | 1067.6 |
| a `cfl_z` | NOD3D | 5.7e-3 | 207.7 | 1067.6 |
| b EOS hpressure/sw_α/sw_β | NOD3D | 2.3e-3 | 199.7 | 1020.4 |
| c GM quartet | NOD2D/ELEM2D_FULL | 8.9e-3 | 188.7 | 857.2 |
| d `uv_rhsAB` | ELEM3D | 1.9e-3 | 186.7 | 810.7 |
| e ALE w/w_e+bolus | NOD3D | 3.8e-3 | 175.8 | 746.5 |
| f ALE commit hnode/helem | NOD3D/ELEM3D | 4.3e-3 | 163.8 | 641.4 |
| **g1-uv** full uv residency | ELEM3D | 4.7e-3 | 150.9 | 342.3 |
| **g1-T** full T values+valuesold | NOD3D | 1.4e-2¹ | **139.9** | **277.3** |

¹ g1-T's worst is h_ice 1.4e-2 (ceil 1e-1); T itself 1.6e-3 (the floor). ⚠️ g1-T FAILED the gate TWICE first
(deterministic T 5.0e-2): the cause was `fesom_bulk_compute` reading `SST=T[surface]` on the host every step
(`fesom_bulk.cpp:259`, **L50** — the uvnode pattern for SST), NOT the values/valuesold asymmetry (ruled out by
a bit-identical 2nd fail). FIX = one targeted T `sync_host` after trdiff. The gate caught it; Serial/pi could not.

(Serial side of every milestone: per-kernel `FESOM_KK_VERIFY` max|Δ|==0 + pi np1+np2 bit-identical +
SYNCCHECK clean — bit-identity preserved by construction, Approach B.)

### ⚠️ The f gate CAUGHT a real CUDA-only bug pi/Serial could NOT (the gate-is-mandatory reason, restated)
f's first gate run **FAILED**: `CG_kk pp·App = -nan` abort at **step 1, iter 1**, 0 steps completed.
Cause: `hnode`/`helem` are EVOLVING mesh (NOT in the set-once `mesh_sync_geometry_device` push,
SYNC_MAP §8); f removed the per-step IN re-pushes that bootstrapped them onto the device, but **step 1
has no prior substep-14 commit** → device read stale/zero `hnode`/`helem` → NaN density → CG abort.
**Serial/pi could not see this** (host==device → the missing push is a no-op, the stale read can't
happen; pi also has no ice). FIX = a one-time `mesh.hnode/helem.modify_host()+sync_device()` before
the time loop (`fesom_main.cpp`); steps 2+ get them from the commit. Re-gated → PASS (worst 4.3e-3).
**General rule (L57):** a field made device-resident across the step boundary with non-zero initial
values needs a one-time step-1 init push.

### Result (NG5 dist_16) — see `docs/SCALING_NG5.md` § M5.13
Clean step **16.27 → 10.88 (a–f) → 6.97 (+g1-uv) → 6.12 s/step (+g1-T) = −62 %**; node-for-node GPU/CPU
**3.76× → 1.41×**; PCIe `cudaMemcpy` (nsys) **12.74 → 2.83 s/step** (share 75 % → 44 %). g1-uv (full uv residency, `e2ad90e`)
was the biggest single win; g1-T (full T values+valuesold, `eaac63b`) needed the L50 bulk-SST fix. Cross-mesh:
the same flips cut **dars** 4.10× → 1.60× (denser dist_8 → 1.52×, the data/compute-balance probe). g2 deferred.

### Climate validation — 1-yr CORE2 CUDA-vs-Fortran+C on the campaign binary (2026-05-30) ✅ PASS

The authoritative fidelity check (not the per-milestone 20-step staleness gate). 1-yr 1958 CORE2 KPP dist_8 on the
**campaign binary** (a–f+g1-uv+g1-T, `f263151`; run `25236304`, `m32_cuda_m513_1yr`, rc=0, 17280 steps, T∈[−2.0,32.1] °C
S∈[3.95,41.05] — physically sane). Compared to the same M5.9-pin pre-campaign run (`m32_cuda_1yr_pin`) **with the same
current compare script** (apples-to-apples), vs canonical KPP refs (`fortran_kpp_5yr_fix` + `kpp_5yr_fix`):

```
backend-vs-C-port (isolates the GPU scatter/reduce drift; C-twin ≡ Serial ≡ bit-identical):
field |  pre-campaign (M5.9-pin)        |  campaign (M5.13)
sst   |  1.00000  +1.0415e-4  RMS 1.41e-2 | 1.00000  +1.1675e-4  RMS 1.40e-2
sss   |  0.99996  -1.7994e-4  RMS 2.61e-2 | 0.99996  -1.8291e-4  RMS 2.61e-2
ssh   |  1.00000  -1.4326e-5             | 1.00000  -1.4023e-5
a_ice |  0.99997  +1.6348e-4             | 0.99997  +1.6261e-4
m_ice |  0.99998  -1.5053e-4             | 0.99998  -1.4952e-4
uice  |  0.99978  -7.4589e-5             | 0.99978  -7.4579e-5
```

**Statistically identical on every field** — pre vs post differ only in the 4th–5th significant figure, i.e. the
run-to-run GPU non-determinism between two independent CUDA runs (D22), **not** a campaign effect. The device-residency
campaign introduced **zero climate-level change**. vs **Fortran**: sst/sss/ssh/a_ice/m_ice all corr ≥ 0.99996; **uice
0.85019** (identical pre & post) = the pre-existing genuine C↔Fortran ice-velocity scheme-skill budget — uice-vs-C is
0.99978, so the GPU reproduces C's ice drift and the 0.85 is *not* GPU/campaign-induced. PASS by the script's own
criterion (backend-vs-C ≪ the C-vs-Fortran budget). **This directly retires the "is 1.623e-3 acceptable?" worry:** the
20-step gate's worst single-cell |Δ| does NOT accumulate — over a full year the campaign binary tracks C-port to corr
1.00000 / bias O(1e-4), exactly as the pre-campaign binary. ⚠️ The doc's older pre-campaign §"CUDA 1-yr 1958" recorded
a_ice/m_ice-vs-Fortran ~0.91/0.98 — that was a stale **old-script** ice-mask-averaging artifact ([[feedback-ice-mask-averaging]]);
the current script gives 0.99997 for *both* runs (shown above). See plan § Deferred + L57; `docs/SCALING_NG5.md` § M5.13.

## §M5.14 — Lever A: the last host-staged nod3D fields (S, density, fer_w, w_i) → device residency

Continuation of M5.13 into the deferred (`S`=g2) / missed (`density`, `fer_w`, `w_i`) fields. Each flip passed the
full ladder (Serial `FESOM_KK_VERIFY` max|Δ|==0 + Serial pi np1+np2 ALL-FIELDS-BIT-IDENTICAL + SYNCCHECK clean +
the **CORE2-active-ice CUDA fidelity gate** `gpu_fidelity_gate.sh --fresh-oracle`, build-cuda-m514 vs the build-serial
oracle). Gate worst-field per flip (all ≪ ceilings; no staleness regression):

| flip (commit) | gate worst field | S max\|Δ\| | T max\|Δ\| | density max\|Δ\| |
|--|--|--|--|--|
| S = g2 mirror-T (`491ccb8`)         | h_ice 4.5e-3 | 3.66e-4 | 9.95e-4 | — |
| + density (`84c1d8d`)               | h_ice 7.8e-3 | 4.32e-4 | 1.18e-3 | 3.08e-4 |
| + fer_w + w_i (`0bc7da9`)           | h_ice 7.7e-3 | 5.03e-4 | 1.35e-3 | 3.87e-4 |

All at the established CUDA climate-close floor (ceilings: S/T 1e-2, density 1e-1, h_ice 1e-1). The cross-flip drift in
the S/T/density values is **run-to-run CUDA non-determinism** between independent gate runs (D22), not accumulation.

**The gate-blind path — monthly means — validated separately** (the gate diffs snapshots only; the means are the
reason for the device-mean-accum, Task 4 `2611936`):
- Serial device-accum vs host-accum (`FESOM_IO_HOST_ACCUM=1`) `.monthly.nc`: **salt/sss/density bit-identical**
  (`cdo diffn`, +/- controls) → `resolve_{salt,sss,density}_dev` arithmetic is correct.
- CUDA device-accum mean vs the **Serial ground-truth** mean (gate runs' `.monthly.nc`): salt/sss/density
  **0/49 records differ > 1e-3** (max ~8.1e-5 salt, comparable to temp's 2.2e-4) → the staleness-free device mean
  is correct on the GPU.
- The standalone CUDA dev-vs-host A/B job (`job_m514_io_ab_pi`) kept hitting transient `cudaErrorDevicesUnavailable`
  (GPU contention, the documented "just re-run" flakiness); the ground-truth comparison above is stronger anyway.

**Device salinity floor** (`fesom_salinity_floor_kk`, the host floor's device twin): race-free per-node
`max(S,0.5)` over myDim+eDim×column, placed AFTER the post-trdiff device halo → bit-identical Serial AND CUDA
(no scatter/reduction). Verified Serial pi/CORE2 bit-identical; nsys ~0.0 % (0.2 ms/step).

**1-yr CORE2 CUDA climate validation (the authoritative verdict): ✅ PASS** (`m32_cuda_m514_1yr`, rc=0, 17280 steps,
0.1727 s/step, T∈[−2.02, 32.12] °C / S∈[3.95, 41.05] — physical, no runaway). The apples-to-apples test (L58 lesson 2:
re-run the prior binary's 1-yr through the **identical** current `scripts/m32_climate_compare.py` so any delta is the
M5.14 flips, not the script). Surface annual-mean stats, year 1958, vs the C-port KPP reference:

| field | M5.14 corr / bias / RMS | M5.13 corr / bias / RMS |
|--|--|--|
| sst  | 1.00000 / +1.04e-4 / 1.407e-2 | 1.00000 / +1.17e-4 / 1.404e-2 |
| sss  | 0.99996 / −1.76e-4 / 2.613e-2 | 0.99996 / −1.83e-4 / 2.612e-2 |
| ssh  | 1.00000 / −1.41e-5 / 9.06e-4  | 1.00000 / −1.40e-5 / 9.05e-4  |
| a_ice| 0.99997 / +1.63e-4 / 2.858e-3 | 0.99997 / +1.63e-4 / 2.857e-3 |
| m_ice| 0.99998 / −1.50e-4 / 3.570e-3 | 0.99998 / −1.50e-4 / 3.568e-3 |
| uice | 0.99978 / −7.46e-5 / 5.34e-4  | 0.99978 / −7.46e-5 / 5.34e-4  |

The two binaries differ only in the **4th–5th significant figure** — the run-to-run CUDA atomic-scatter floor (D22),
not a systematic shift. vs Fortran: identical picture (sst/ssh corr 1.00000, sss/a_ice/m_ice 0.99996–0.99997); the
`uice`-vs-Fortran **0.85019** is the known genuine C↔Fortran ice-drift budget (`uice`-vs-C is 0.99978, far inside it),
**bit-for-bit the same as M5.13** (0.85019). DRIFT not assessed (single year). **Conclusion: the parity win (NG5 4N
0.879× node-for-node) cost ZERO climate fidelity** — the M5.14 device-residency flips are statistically identical to
the M5.13 campaign binary on every field. Campaign (M5.13 + M5.14) is climate-validated end to end.

## §M5.15 — GM-chain device residency (T1+T2): the dominant residual-PCIe source

A clean re-profile of the M5.14 binary (NG5 dist_16, `nsys_ng5/ng5.sqlite` + a `FESOM_SYNC_LOG`-instrumented per-field
attribution run) **overturned M5.14's "compute-bound" read**: GPU utilization is only **~30 %** of wall (NOT
compute-bound — the smoother, the #1 *kernel* at 25.7 %, is only ~8 % of *wall*); the step idles ~70 % on data
movement + halos. Decomposition (steady-state, % of wall): PCIe DtoH 24.7 % (4.11 GB/step), HtoD 9.7 %,
`MPI_Waitall` 51.3 %, CG `Allreduce` **0.2 %** (the CG-reduction flavor of Lever B is dead — confirms M5.2).

**Per-field DtoH attribution** (the sync-rail logger, ranking mesh-invariant) put the **GM chain** on top:
`neutral_slope` 18.1, `sigma_xy` 12.3, `fer_K` 6.2, `fer_tapfac` 6.2 MB/step/rank — all **host-bracket halos**
(`sync_host` DtoH + host `fesom_halo_exchange` = the `MPI_Waitall` + re-import HtoD) or, for `fer_tapfac`/`fer_scal`,
**verify-only syncs firing in production**. Ice ruled out (all nod2D); KPP's 13-field dump block is debug-gated (off).

**T1+T2 (`81b8947`):** flipped `sigma_xy`/`neutral_slope`/`fer_K` host halos → `fesom_halo_field` (device GPU-aware,
the L48/M5.13c recipe; `fer_C` kept host = small nod2D), and guarded `fer_tapfac`/`fer_scal` behind `s_verify_gm`
(device kernels read them OWNED; only the gated verify needs host). One fix kills both costs (the DtoH AND the host
exchange→device path).

**Validated (full ladder):** Serial gm-verify max|Δ|==0; pi np1+np2 ALL-FIELDS-BIT-IDENTICAL (vs C baseline + nocma);
SYNCCHECK clean (no production host reader broken — confirms the verify-only classification); **CORE2-active-ice CUDA
fidelity gate PASS** (worst 8.08e-3, all within ceilings, no staleness regression). Climate-safe (pure residency).

**Result (NG5 dist_16, same-day vs M5.14's 3.80 s/step):**

| metric | M5.14 | T1+T2 | Δ |
|--|--|--|--|
| clean step (`job_ng5_prof`) | 3.80 s/step | **3.61** | **−5 %** |
| `deep_copy` | 6.57 GB/step (121 calls) | **4.84 (116)** | **−26 %** |
| nsys PCIe DtoH | 4.11 GB/step | 2.35 | **−43 %** |
| GPU utilization | 29.8 % | 34.0 % | +4 pts |
| `1b_gm` phase (% of loop) | ~11–16 % | **4.63 %** | collapsed |
| `MPI_Waitall` | 51.3 % | 47.3 % | −4 pts |
| node-for-node GPU/CPU | 0.879× | **~0.834×** | GPU ~17 % faster |

The 1.73 GB/step `deep_copy` drop matches the GM fields exactly (`neutral_slope` 765 + `sigma_xy` 517 + `fer_K` 259 +
`fer_tapfac` 259 MB ≈ 1.8 GB), and the `1b_gm` phase collapse is the direct fingerprint. The `MPI_Waitall` fell only
4 pts → the bulk of the remaining halo-wait is the *other* exchanges + load-imbalance (the T3 / deferred overlap-B
target). Tooling: `FESOM_SYNC_LOG` rail in `fesom_field.hpp` (compile-guarded), `job_nsys_ng5` (now MPI-traced),
`job_synclog_core2`, `job_m515_serial_val`. Plan: `docs/plans/20260530-m515-gm-residency.md`. Lesson L60.

**T3 (`<commit>`) — the remaining nod3D host-readers, investigated + the verify-only ones removed.** Per-field
classification (the attribution + code trace): `bvfreq` (`:192`) and `dbsfc` (`:193` OUT + `:351` IN re-push) are
**verify-only** — their raw-`h()` readers are the host C-twins `eos_verify`/`fesom_kpp_verify`(`kpp_bldepth`), which
run only under `FESOM_KK_VERIFY` (Serial host==device → no sync needed); production reads them on device
(`bvfreq` eos/gm/pp/kpp `.d()` + the smoother re-dirties it; `dbsfc` `kpp_bldepth_kk` `.d()`). Removed both (the
`dbsfc` re-push was a redundant HtoD; `bvfreq`'s cosmetic min/max console range goes stale on CUDA — accepted class).
**Kept:** `T` (genuinely required — `fesom_bulk.cpp:259` reads SST on host next step, the JRA55 heat flux; the L50
uvnode class), `S` (already device, M5.14), `ghats` (a deliberate host-keep — its post-exchange consumer is gated
off in CORE2), `MLD1_ind` (tiny nod2D index, GM `init_Redi` host read). Validated: eos/kpp/gm verify max|Δ|==0,
pi np1+np2 ALL-FIELDS-BIT-IDENTICAL, SYNCCHECK clean, **CUDA gate PASS (worst 7.47e-3; `bvfreq` snapshot 3.9e-7
→ the pre-I/O sync fully covers it)**.

| metric | M5.14 | T1+T2 | **+T3** |
|--|--|--|--|
| clean step (NG5 dist_16) | 3.80 s | 3.61 | **3.456 (−9% cumulative)** |
| `deep_copy` | 6.57 GB/step (121) | 4.84 (116) | **4.10 (113) — −38% cumulative** |
| node-for-node GPU/CPU | 0.879× | ~0.834× | **~0.80× (GPU ~25% faster)** |

**Residency is now largely exhausted** — the remaining per-step DtoH is genuinely-required host readers
(`T`/`uvnode` for the JRA55 bulk forcing, the L50 class) + small/deliberate fields.

**NEXT-LEVER MEASUREMENT (the forcing-phase split, gated FPROF `fesom_main.cpp:1045`):** `force:bulk_compute`
= **0.5517 s/step = 16.0 % of the step** (the host bulk formulae run **single-threaded** on the GPU build's
Kokkos-Serial host — the blmc/L49 trap, invisible to nsys), vs `force:jra55_read` = 0.098 s (2.8 %, stays host).
**⟹ porting `fesom_bulk_compute` to a device per-surface-node map is the recommended next campaign (M5.16),
ahead of Lever-B overlap:** it's a ~16 % *contained* win, climate-safe (race-free map → bit-identical Serial,
gate-only), AND it dissolves the residency wall — once bulk reads `T`/`uvnode` on-device, their ~760 MB/step DtoH
disappears (the new HtoD for the 8 nod2D JRA55 surface fields is far smaller). Lever B (the ~47 % `MPI_Waitall`,
interior/boundary overlap) is the prize *after* that — bigger but invasive + partly load-imbalance.

**✅ 1-yr CORE2 CUDA climate = PASS (closes M5.15).** `m32_cuda_m515_1yr` (17275 steps, **0.1424 s/step** on CORE2
dist_8 — vs M5.14's 0.1727, **−17.5 %**; T∈[−2.02,32.12]/S∈[3.95,41.05] physical). Apples-to-apples (L58): M5.15 vs
M5.14 1-yr through the identical `m32_climate_compare.py` differ only in the **4th–5th sig-fig** (the D22 atomic-scatter
floor) — vs C-port: sst/ssh corr 1.00000, sss/a_ice/m_ice 0.99996–0.99998, bias O(1e-4), `uice` 0.99978; vs Fortran
identical, `uice`-vs-Fortran 0.85019 = the known C↔F ice-drift budget (bit-for-bit the M5.13/M5.14 number). **The
GM-chain residency cost ZERO climate fidelity** — the whole residency arc (M5.13 → M5.14 → M5.15) is climate-neutral
end to end. **M5.15 COMPLETE.**

## §M5.16 — port `fesom_bulk_compute` to a device per-surface-node map (the forcing-compute lever)

The M5.15 close MEASURED the forcing phase (gated FPROF split, `fesom_main.cpp`): `force:bulk_compute` = **16.0 % of
the NG5 step** (0.55 s/step) — the L&Y09 air-sea bulk formulae ran **single-threaded on the GPU build's Kokkos-Serial
host** (the blmc/L49 trap: a host loop nsys can't see — it profiles GPU kernels only). M5.16 ports it to a
`KOKKOS_LAMBDA` per-surface-node MAP over `[0,N)` (the EOS/KPP/ice-thermo device-map class, L39/L45): `ncar_ocean_-
fluxes_mode` + `obudget` become `KOKKOS_INLINE_FUNCTION` device twins (std math → `Kokkos::`; the scalar `BULK_*`/
`FESOM_CD_ATM_ICE` are `#define`s, `z_wind/z_tair/z_shum` captured by value); the **8 JRA55 surface fields are HtoD'd
at bulk**; **SST=`T`[surface] + `uvnode` are read on the DEVICE.**

**THE RESIDENCY UNLOCK.** `T` and `uvnode` were the LAST genuinely-required per-step DtoH (the L50 class — proven by
the M5.9-pin NaN-poison discriminator that bulk was their SOLE host reader). With bulk reading them on-device, the
per-step `T` `sync_host` (`fesom_step.cpp:900`) and `uvnode` `sync_host` (`:334`) are **GONE** → the ~nod3D + nod3D×2
DtoH/step disappears. (Confirmed I/O-safe: `T` snapshot output is covered by the pre-I/O `sync_host`, `fesom_main.cpp`,
L48; `uvnode` is not a snapshot field; `ocean2ice` reads `T` via `.d()`.)

**Phase A (drop-in).** `fesom_bulk_compute_kk` writes ALL outputs on device, halos `{stress_node_surf,heat_flux,
water_flux}` via `fesom_halo_field`, then **`sync_host`s the full output set** so the downstream is byte-for-byte the
host-authoritative state the C twin left: `oce_fluxes_mom` [host] reads `stress_node_surf`; the ice-step IN rails
(`Ch_atm_oce`/`Ce_atm_oce` → thermo, `stress_atmice` → EVP); the host element-interp → `stress_surf`; the ocean-step
re-pushes. **No downstream edit** → minimal validation surface. The win is the device COMPUTE + the removed `T`/`uvnode`
DtoH, not these small nod2D round-trips; making `forcing` fully device-resident (drop the output `sync_host` + the
downstream re-pushes) is the measured **Phase B follow-on** (it would reclaim the deep_copy that Phase A re-adds).

**Validated (full ladder):**
- **Serial `bulk` verify max|Δ|==0** (`job_bulk_verify_core2`, CORE2 dist_16, JRA active, 60 steps): all 8 outputs
  (`sns/ss/hf/wf/Ch/Ce/satmx/satmy`) bit-identical to the C twin every step on all 16 ranks, **and zero nonzeros across
  `bulk`+the 5 ice keys** (the forcing→ice→ocean chain unchanged — the drop-in is proven). ⚠️ FORCED-ONLY: pi uses
  analytical stress and NEVER calls bulk (`jra55_year≤0`), so this verify is CORE2-only (L42), exactly like ice.
- **SYNCCHECK clean** (`job_bulk_synccheck_core2`, build-synccheck CORE2): no `h_checked()` stale-host abort — the
  residency unlock breaks no production host reader, and the bulk device writes are all `sync_host`'d.
- **CORE2-active-ice CUDA fidelity gate PASS** (`gpu_fidelity_gate.sh --fresh-oracle`): all 27 fields at the climate-
  close floor, **worst `h_ice` 8.53e-3** (ceil 1e-1); **`T` 1.13e-3 (ceil 1e-2) — at the floor, NOT a staleness
  divergence** → the SST device-read + residency unlock are correct on CUDA. "No staleness regression."

**Result (NG5 dist_16, RIGOROUS same-day + SAME-NODE baseline — M5.15 binary rebuilt + run on the same 4 GPU nodes
`l50133/45/48/93` today, jobs 25246378 vs 25246296):**

| metric | M5.15 baseline | **M5.16** | Δ |
|--|--|--|--|
| clean step (`job_ng5_prof`) | 3.4524 s/step | **2.6766** | **−22.5 %** |
| `force:bulk_compute` | 0.5535 s (16.03 %) | **0.0253 (0.95 %)** | **−95 %** |
| `force:jra55_read` | 0.0975 s (2.82 %) | 0.0970 (3.63 %) | unchanged (stays host) |
| `deep_copy` | 4099 MB/step (113) | **3413 (126)** | **−16.7 %** |

The same-day baseline reproduced M5.15's 3.456 exactly (3.4524) → the **−22.5 % is real, not node-mix noise**.
Attribution: the bulk-compute device port (−0.528 s, the `force:bulk_compute` collapse 16.0 %→0.95 % — the single-
threaded host loop is gone) + the `T`/`uvnode` residency unlock (−0.248 s, the `deep_copy` −0.69 GB → less ocean-phase
PCIe). **This exceeds the prompt's ~16 % estimate** (which counted only the bulk compute; the residency unlock added
~7 % more). The `deep_copy` drop (−16.7 %) is smaller than the raw `T`/`uvnode` bytes because Phase A re-adds the small
nod2D output `sync_host` + the 8 JRA HtoD (+13 calls); Phase B reclaims it. ⚠️ M5.16 is a **no-op on the CPU build**
(the OpenMP path's bulk is already cheap per-rank at high decomposition; residency is GPU-only) → node-for-node GPU/CPU
improves purely from the GPU step drop (needs a same-day CPU re-run to pin — Lever D).

**Node-for-node GPU/CPU (NG5, 4 nodes, RIGOROUS same-day):** GPU dist_16 **2.6766** / CPU dist_512 (M5.16 build-serial,
128 c/node, reps 4.378+4.322) **4.350** = **0.615× → the GPU node is 1.63× FASTER than a CPU node** (was ~0.80× at
M5.15). The CPU/Serial build is provably unaffected by M5.16 (the residency flips are no-ops on Serial; the bulk kernel
runs single-threaded there and is tiny at 14.5k nod2D/rank) — confirmed same-day (4.35 ≈ the M5.12 study's 4.33). The
node-for-node arc over the whole M5.x campaign: **3.76× SLOWER (M5.12) → 0.88× (M5.14) → 0.80× (M5.15) → 0.615×
(M5.16) — a ~6× swing on identical node counts.**

**✅ 1-yr CORE2 CUDA climate = PASS (closes M5.16).** `m32_cuda_m516_1yr` (17275 steps, **0.1271 s/step on CORE2
dist_8 — vs M5.15's 0.1424, −10.7 %**; the bulk port helps CORE2 too). Apples-to-apples (L58), M5.16 vs M5.15 1-yr
through the IDENTICAL `m32_climate_compare.py` differ only in the **4th–5th sig-fig of the bias** (the D22 atomic-
scatter floor): vs C-port sst/ssh corr **1.00000**, sss/a_ice/m_ice 0.99996–0.99998, `uice` **0.99978** (M5.15: 0.99978);
vs Fortran sst/ssh 1.00000, sss/a_ice/m_ice 0.99996–0.99997. **The bulk device port + the T/uvnode residency unlock
cost ZERO climate fidelity** (M5.16 ≡ M5.15 to the floor, and the port reproduces the C twin: every backend-vs-C corr
≥ 0.99978). Lesson L61. **M5.16 COMPLETE.**

⚠️ **`uice`-vs-Fortran SCRIPT-ARTIFACT CORRECTION (2026-05-30):** docs §M5.13–§M5.15 reported `uice`-vs-Fortran ≈ 0.850
as "the C↔F ice-drift budget." That number was **artifact-deflated**: `uice` was omitted from the per-month NaN→0
ice-masking that `a_ice`/`m_ice` get ([[feedback-ice-mask-averaging]]) — Fortran masks ice-free water as NaN, the
C/Kokkos port writes 0 (73.5 % of CORE2 nodes), so the Fortran annual `nanmean` dropped ice-free months while the
port's mean kept the zeros → a spurious decorrelation hitting ONLY the Fortran comparison (C/Kokkos share the
0-convention → `uice`-vs-C was always a clean 0.99978). **Fixed** (`uice`/`vice` added to `ICE_FIELDS` in
`m32_climate_compare.py`): `uice`-vs-Fortran = **0.919** (ice-covered-nodes-only = 0.913 — same ballpark; `|d|max`
0.228→0.136). The residual ~0.91 is a **real but modest** C-port-vs-Fortran ice-velocity (EVP) difference — NOT a
Kokkos/GPU artifact (the port matches the C twin at 0.99978). This correction does not change any M5.x perf/fidelity
verdict (those rest on backend-vs-C + M5.16≡M5.15); it just relabels the Fortran `uice` number.

## §M5.17 — Lever B (MPI comms) MEASURED FIRST → the `MPI_Waitall` is 79 % LOAD-IMBALANCE, NOT recoverable comm

The M5.x roadmap nominated **Lever B** (overlap / aggregate the halo comms) as the next prize — the profiler had put
`MPI_Waitall` at ~47 % of the (M5.14/M5.15) step. **The prompt's explicit discipline was MEASURE the split before
building anything: a chunk of that 47 % is load-imbalance idle (fast ranks waiting at the exchange for slow ranks),
which overlap canNOT recover.** A barrier-isolation experiment settled it decisively, and **Lever B is a dead end.**

**Instrument (`src/fesom_halo_device.cpp` + `fesom_halo.cpp`, env-gated, zero prod cost; covers BOTH the device and
host halo paths):** `FESOM_HALO_BARRIER=1` inserts an `MPI_Barrier(MPI_COMM_FESOM)` before every halo exchange (so the
per-rank arrival skew is absorbed into the barrier and the following `MPI_Waitall` measures PURE comm);
`FESOM_HALO_MPI_PROF=1` times `Barrier` + `Waitall` and reports across-rank min/mean/max via an `MPI_Reduce` at loop
end. Validated byte-clean: **pi np1 + np2 BIT-IDENTICAL** (np2 exercises the host-path edit) → the instrument is
measurement-only. Job `jobs/job_ng5_halo_split` runs 3 passes back-to-back on the SAME 4 GPU nodes.

**Result — NG5 dist_16, M5.16 binary, 30 timed steps (job 25246957):**

| run (same 4 nodes) | s/step | `MPI_Waitall` mean (min/max) | `MPI_Barrier` mean (min/max) | split |
|--|--|--|--|--|
| clean (no instrument) | **2.6766** | — | — | (reproduces M5.16 exactly) |
| base (prof on, no barrier) | 2.6750 | **0.2878** (0.147 / 0.565) | — | Waitall = imbalance + comm |
| barrier (prof + barrier) | 2.6957 | **0.0651** (0.041 / 0.082) | **0.2455** (0.094 / 0.529) | **imbalance 79 % \| comm 21 %** |

- **The per-step halo `Waitall` is only 0.288 s/step = 10.8 % of the step** (NOT 47 %; the older nsys 47 % was rank-0
  wall that lumped imbalance idle + comm + setup). Of it, the barrier shows **79 % is load imbalance** (0.2455 s/step)
  and **only 21 % is comm** (0.0651 s/step).
- **The overlappable-comm CEILING is 0.065 s/step = 2.4 % of the step.** Even a *perfect* interior/boundary overlap
  (§2A) — and the GPU is only ~30 % utilized, so there is barely any interior compute to hide comm behind — saves
  ≤ 2.4 %. **Lever B is not worth the invasiveness.**
- **The variance corroborates:** base-`Waitall` spread across ranks is HUGE (min 0.147 → max 0.565, a 0.42 s skew);
  barrier-`Waitall` is TIGHT (0.041 → 0.082, all ranks wait ~equally once arrival is equalized). High variance ⇒
  imbalance-dominant; the tight post-barrier comm ⇒ comm is small and uniform.
- **Wall-time proves the imbalance is intrinsic, not a barrier artifact:** +638 barriers/step changed the wall by
  **+0.7 %** (2.6766 → 2.6957). The imbalance idle was ALREADY being paid as `Waitall` in the natural run; the barrier
  just relabels it. (The per-exchange barrier is an *upper* bound on imbalance — it blocks cross-kernel skew
  amortization — but the +0.7 % wall bounds the over-count to noise.)
- **Aggregation (§2B) won't rescue it either:** nsys shows `MPI_Isend`/`Irecv` 95 680 each, ~87 KB avg message
  (medium, not many-tiny) → the latency win from packing is marginal, and it only attacks the 2.4 % comm anyway.
- **CG/EVP `Allreduce` confirmed DEAD** (nsys 0.4 % of MPI; matches M5.2 / L60).

**Verdict (per the prompt's decision matrix): LOAD-IMBALANCE-DOMINANT → STOP Lever B. Document the ceiling, pivot.**
The real remaining walls on the 2.677 s/step are: **(1) load imbalance ~0.245 s/step (9.2 %)** — fast ranks idling
for slow ranks; recoverable only by a better mesh partition (**Lever D**, deployment-side, no port-code risk) or a
work-weighted decomposition, *if* the skew is static (per-rank dump needed to tell static-partition from
dynamic-physics/GPU-jitter). **(2) the GPU-compute + residual PCIe bulk (~89 %)** — nsys top kernel `fesom_smooth_nod3D`
**25.7 % of GPU compute**, plus `deep_copy` still 3.41 GB/step → **Lever C** (rank-1 → `View<double**>` coalescing /
layout) + any residual-residency. **(3) the comm itself (2.4 %)** — Lever B — not worth it. Lesson **L62**. Tooling:
`jobs/job_ng5_halo_split`, the `[halo-mpi-prof]` rail in `fesom_halo_device.cpp` (env-gated).

## §M5.18 — the coalescing lever: `fesom_smooth_nod3D_kk` re-parallelized to one-thread-per-(node,level)

§M5.17 pivoted to the GPU-compute+PCIe bulk (~89 %), and the fattest single kernel was the area-weighted
horizontal smoother `fesom_smooth_nod3D_kk` (`src/fesom_eos.cpp`; bvfreq N² 1 sweep + KPP blmc 3 sweeps×3 channels)
— **25.7 % of all GPU compute** per nsys. The M5.5 device port mapped **one thread per (slab, owned-node)** with an
internal loop over the node's levels. Since the field is **node-major** (`arr[node*NL+nz]`, `NL=70` contiguous),
consecutive threads = consecutive *nodes* → every `work`/`vol`/`arr` access strided by `NL` → catastrophically
**uncoalesced stores**, plus hard **depth divergence** (adjacent nodes have wildly different column depths).

**The fix (commit-pending, M5.18):** re-parallelize to **one thread per (slab, node, LEVEL)** — flat
`RangePolicy(0, nslab*Nmy*NL)` decoded to `(s,n,nz)`, masking `nz∉[uln,nlnz]`. Now a warp of 32 consecutive flat
idx spans 32 consecutive **levels of the same node** → all element-vertex reads `arr(sb+v*NL+nz0..+31)` and all
`work`/`vol`/`arr` stores are **contiguous = coalesced**, and the divergence vanishes (one level/thread). The
per-`(n,nz)` float sum still runs over the SAME element order `k=o0..o1` and accumulates in the SAME sequence
(register accumulator ≡ the old in-memory one, IEEE op-for-op) → **byte-identical**. No global layout refactor —
this exploits the EXISTING layout's level-contiguity (the local, low-risk cousin of Lever C). The 2-kernel
read-then-write split + per-channel device halo are unchanged.

**ncu before/after (NG5 dist_16, rank0, `jobs/job_ncu_smooth_ab`, explicit `--metrics` so the sectors/req counter
that `--set basic` drops is captured; the `before` reproduces the prompt's baseline — gather SM 2.27→2.23 %, occ
52.7→52.4 %, 48–145 ms — validating the A/B):**

| kernel | metric | BEFORE (m516 per-node) | AFTER (m518 per-(n,nz)) | factor |
|--|--|--|--|--|
| gather | duration med (range) | 64.0 ms (47.7–145.3) | **4.1 ms (1.4–4.3)** | **15.5×**, divergence spread gone |
| gather | STORE sectors/req | 29.5 | **6.9** | 4.3× (≈optimal for f64; masked warps < 8) |
| gather | LOAD sectors/req | 23.8 | **2.9** | 8.3× |
| gather | occupancy / SM util | 52.4 % / 2.2 % | **91.5 % / 58.9 %** | ALUs were 98 % idle on latency → working |
| scale | duration med | 13.9 ms | **1.4 ms** | 9.9×; now DRAM-bound (73 %) — the goal |

**Perf — same-day, SAME-NODE A/B (both binaries back-to-back in ONE 4-node allocation, `jobs/job_ng5_m518_ab`,
job 25247500; the only valid baseline — [[feedback-perf-same-day-baseline]]. NB the absolute 2.48 differs from
§M5.17's 2.68 by node/day variance ~7 %, which is exactly why a memory-recorded number from another day is NOT a
baseline):**

| run (clean, no profiler fences) | s/step |
|--|--|
| BEFORE (m516) | 2.4823 |
| **AFTER (m518)** | **2.1296** |
| | **−14.2 %** |

The `FESOM_STEP_PROFILE` breakdown attributes it cleanly: the **ocean** phase dropped **1.7992 → 1.4488 s/step
(−0.350, −19.5 %)** while forcing (0.2118→0.2126), sea-ice (0.1462→0.1457) and coupling (0.3260→0.3260) are
byte-stable — both smoothers live in the ocean phase (bvfreq in `1_eos`, blmc in `3_mixing`), and the ocean delta
(0.350) ≈ the clean-step delta (0.353). **~14 % of the whole step from one bit-identical kernel re-parallelization
— ~2× the prompt's 5–8 % estimate, because the kernel sped up 10–15× (not the assumed 2–4×).**

### Validation — bit-identical → the M2.x gate, plus a NEW isolated `smooth` verify
A new `FESOM_KK_VERIFY=smooth` hook (`fesom_smooth_nod3D_kk_verify`, `fesom_eos.cpp`) captures the input
before the device kernel (L26), runs it, then runs the host C twin `fesom_smooth_nod3D` per channel and diffs the
owned region. This is the **only tight, isolated** check of the smoother: the `eos` gate runs BEFORE the bvfreq
smoother (unsmoothed bvfreq), and `kpp` covers blmc only transitively through the `max(viscA/diffK, blmc)` combine.
- **Serial + OpenMP** `smooth` (bvfreq nslab=1 + blmc nslab=3): `max|Δ|=0.000e+00` every step (race-free map — no
  scatter → OpenMP is bit-identical too). `eos`/`kpp` also clean.
- **pi np1 + np2** end-to-end (100/20/10, analytical): **ALL FIELDS BIT-IDENTICAL** vs the golden + the np2 oracle
  (np2 exercises real MPI/scatter). pi DOES exercise the bvfreq smoother (the port calls it unconditionally) AND blmc
  (KPP on).
- **SYNCCHECK** clean.
- **CUDA fidelity gate** (`gpu_fidelity_gate.sh --fresh-oracle`, CORE2 dist_8 ICE ACTIVE): **PASS** — all 27 fields at
  the climate-close floor (worst h_ice 7.7e-3 ≤ ceil 1e-1; T 9.8e-4, S 3.4e-4, bvfreq 2.3e-7), the SAME floor as M5.16
  → the rewrite adds ZERO new divergence.
- **1-yr CORE2 CUDA climate PASS** (`m32_cuda_m518_1yr`, 17275 steps, 0.130 s/step): **vs M5.16 apples-to-apples (the
  zero-cost proof) — every field corr = 1.00000, bias O(1e-6), RMS O(1e-4) → M5.18 ≡ M5.16 to the D22 atomic-scatter
  floor, ZERO climate cost.** vs C-port: sst/ssh corr 1.00000, sss/a_ice/m_ice ≥ 0.9999, uice 0.99978. vs Fortran
  (science budget): corr ~1; uice 0.919 = the known, faithfully-reproduced C-vs-Fortran EVP difference (the
  [[feedback-ice-mask-averaging]] number, unchanged from M5.16). **M5.18 COMPLETE.**

Tooling added this session: `jobs/job_ng5_m518_ab` (same-node clean+profile A/B), `jobs/job_ncu_smooth_ab` (ncu A/B
with the sectors/req metric), `ncu_rank0.sh` `NCU_METRICS` override. Lesson **L63**.

## §M5.19 — generalize the coalescing lever to `compute_vel_rhs` + GM `sigma_xy`/`neutral_slope` (the next memory-bound node/element-major kernels)

M5.18 proved the lever on the #1 kernel (the smoother). §M5.19 generalizes it. **Step-0 re-profile** (`jobs/job_ng5_prof`
+ `jobs/job_nsys_ng5` on the M5.18 binary) confirmed the smoother dropped out of the top (`fesom_smooth_gather`
25.7 %→**0.60 %**) and gave the post-M5.18 ranking: `13_fct` 22.8 %, `3_mixing` 10.1 %, `1b_gm` 7.5 %, `12_ale` 6.7 %,
`4_velrhs` 4.4 %; the fattest single coalescing targets = `fesom_vel_rhs_elem` 1.73 %, `fesom_gm_sigma_xy` 1.67 %,
`fesom_vel_rhs_assembly` 0.88 %, `fesom_gm_neutral_slope` 0.80 %.

**The classification (code-read, not profile-driven — the bucket dictates the technique):**

| bucket | recognize | technique | this session |
|--|--|--|--|
| **A. pure per-level map** | internal `for(nz)` from same-level/fixed-per-column reads; no scratch/scan | the M5.18 flat lever: `RangePolicy(0, N*nl)`, decode `(n,nz)`, mask, register-write | `vel_rhs_elem`, `vel_rhs_assembly`, `momadv_area`, `momadv_v2e`, `update_vel`(M5.20) |
| **B. per-column** | column scratch / multi-pass / vertical-neighbor coupling | TeamPolicy, **OR** if the "scratch" is really per-level accumulators / the passes have no cross-level WRITE dep → the flat lever still applies (see GM below) | GM `sigma_xy`+`neutral_slope` (turned out flat-lever-able); `momadv_vert`, FCT `zal_a34`, `qr4c_v` deferred |
| **C. TDMA** | down-then-up column sweep, `nz` depends on `nz-1` | Lever C layout only | `impl_vert_visc`, `fer_solve_gamma`, `impl_vert_diff_tracers` |
| **D. scatter** | `atomic_add` from an edge/element loop | different axis | `momadv_horiz`, `visc_bidiff`, FCT `*_h`/`LO_scatter`, redi edges |

**KEY FINDING — GM `sigma_xy`/`neutral_slope` are flat-lever-able, NOT TeamPolicy (the prompt guessed B→TeamPolicy):**
- `sigma_xy`'s `tx/ty/sx/sy/vol[NL_MAX]` "column scratch" holds **per-level accumulators with no cross-level reduction**.
  A per-`(n,nz)` thread re-walks the node's surrounding elements (`k=o0..o1`, the SAME order) and accumulates ONLY its
  level in registers → scratch eliminated, element/level loops just swap order, total arithmetic unchanged → byte-identical.
- `neutral_slope`'s 3 passes + `c1[NL_MAX]` have **no cross-level WRITE dependency** (Pass 2's `c1[nz]` reads only
  `bvfreq[nz]`/`bvfreq[nz+1]` — INPUTS; Pass 3 re-reads `ns[nz]` = the bits Pass 1 just wrote). The passes **fuse** into one
  per-`(n,nz)` computation with `c1` register-local → byte-identical.
- Lesson: a "column scratch" is only a true bucket-B blocker when it carries a CROSS-LEVEL reduction or a recurrence on
  WRITTEN values. Per-level accumulators and input-only neighbor reads are still bucket A. Always read the scratch's data flow.

The 4 momentum maps (`vel_rhs_elem`/`assembly` `src/fesom_momentum.cpp:491,537`; `momadv_area`/`v2e` `:404,421`) + 2 GM maps
(`fesom_gm_sigma_xy`/`neutral_slope` `src/fesom_gm.cpp`) were all flipped to one-thread-per-(elem/node, LEVEL). Element/node
scalars (`Fx/Fy/ff`, `grad/area`) recompute per thread but are warp-broadcast (all 32 threads in a warp share one element/node).

**Perf — same-day SAME-NODE A/B (`jobs/job_ng5_m519_ab`, job 25249075; m518 vs m519 back-to-back in one allocation):**

| run (clean, no profiler fences) | s/step |
|--|--|
| BEFORE (m518) | 2.1278 |
| **AFTER (m519)** | **2.0139** |
| | **−5.35 %** |

The `FESOM_STEP_PROFILE` breakdown attributes it to exactly the two touched phases: **`4_velrhs` 0.0955→0.0323 s/step
(−66 %)** + **`1b_gm` 0.1638→0.1137 (−31 %)** = **−0.1133 s/step**, ≈ the clean-step delta (−0.1139); every other phase is
byte-stable (`13_fct` 0.4978→0.4976, `3_mixing` 0.2203→0.2196, `12_ale` 0.1460→0.1458, `7_ssh`/`ice`/`eos`/`force` flat).
`fesom_gm_sigma_xy`/`fesom_vel_rhs_elem` fall out of the top-17 kernels entirely after the flip.

**ncu A/B (`jobs/job_ncu_m519_ab`, job 25249076, NG5 dist_16 rank0, `--metrics` w/ the sectors/req coalescing counter):**

| kernel | dur ms (B→A) | SM % (B→A) | occ % (B→A) | LD sec/req | ST sec/req |
|--|--|--|--|--|--|
| `vel_rhs_elem` | 37.8 → 2.8 (**13.4×**) | 2.5 → 18.0 | 70.9 → 71.0 | 30.2 → 3.8 | 30.8 → 13.2 |
| `compute_sigma_xy` | 36.4 → 2.9 (**12.6×**) | 2.0 → 43.8 | 52.6 → 56.8 | 22.4 → 3.0 | 30.6 → 13.2 |
| `vel_rhs_assembly` | 19.0 → 1.6 (**12.0×**) | 1.7 → 22.7 | 93.6 → 92.5 | 29.7 → 7.9 | 30.8 → 13.2 |
| `compute_neutral_slope` | 17.8 → 1.7 (**10.6×**) | 3.1 → 29.9 | 53.1 → 67.7 | 30.0 → 7.4 | 31.1 → 20.0 |

Identical M5.18 signature: before = ~2–3 % SM util (ALUs idle on memory latency), LD ~22–30 sectors/req (uncoalesced
node/element-major stores strided by `nl·2`); after = LD 3–8 sectors/req, SM 18–44 %, **10–13× per-kernel**. The ncu durations
match the per-step profile (e.g. `vel_rhs_elem` 37.8 ms ≈ profile 37.7 ms), and the 4 kernels' savings (~102 ms/step) ≈ the
clean-step delta (114 ms) → **the −5.35 % is fully attributed to the coalescing flips** (the rest = the tiny `momadv_*` maps).

### Validation — bit-identical → the M2.x gate (same discipline as M5.18)
All six flips are race-free maps with the per-`(n,nz)` arithmetic IEEE op-for-op identical to the C twin → **Serial bit-identical**
(which, for a race-free map, ⟹ OpenMP bit-identical; the only OpenMP non-determinism in the momentum chain is the pre-existing,
untouched `momadv_horiz` D22 scatter).
- **`FESOM_KK_VERIFY=vrhs` Serial**: `uv_rhs`=0.000e+00, `uv_rhsAB`=0.000e+00 every step (covers the whole momentum-rhs chain
  incl. `momadv_area`/`v2e`). **`FESOM_KK_VERIFY=gm` Serial**: `sigma_xy`=0, `neutral_slope`/`slope_tapered`/`fer_tapfac`=0, and
  the entire downstream GM chain (`fer_K`/`fer_gamma`/`fer_uv`/`redi`)=0.000e+00.
- **pi np1 + np2** (100/20/10, analytical, build-serial → ordered scatter → full bit-identity): **ALL FIELDS BIT-IDENTICAL** vs the
  golden + the np2 nocma oracle. (GM is ON in pi, so the GM flips are exercised end-to-end.)
- **SYNCCHECK** clean (no sync-rail change — pure kernel re-parallelization).
- **CUDA fidelity gate** (`gpu_fidelity_gate.sh --fresh-oracle`, CORE2 dist_8 ICE ACTIVE): **PASS** — all fields at the
  climate-close floor (worst h_ice 1.718e-3; T 1.7e-4, density 3.3e-4, bvfreq 3.8e-7) → ZERO new divergence vs M5.18.
- **1-yr CORE2 CUDA climate PASS** (`m32_cuda_m519_1yr`, 17280 steps, clean: T[-2.01,31.27] S[3.95,41.05], no nan): **vs
  `m32_cuda_m518_1yr` apples-to-apples (the zero-cost proof) — every field corr=1.00000, bias O(1e-6–1e-7), RMS O(1e-4–1e-5)
  → M5.19 ≡ M5.18 to the D22 atomic-scatter floor, ZERO climate cost.** vs C-port (KPP): sst/ssh corr 1.00000, sss 0.99996,
  a_ice/m_ice 0.99997–0.99998, uice **0.99978** (identical to M5.18). vs Fortran (science budget): corr ~1; uice 0.91872 = the
  known, faithfully-reproduced C-vs-Fortran EVP difference ([[feedback-ice-mask-averaging]]), unchanged from M5.18. **M5.19 COMPLETE.**

### Remaining coalescing headroom (for M5.20) — classified, not yet flipped
- **Bucket A (flat lever, ready):** `fesom_update_vel` (0.82 %, `fesom_momentum.cpp:1106`, clean per-elem map, `ssh` verify);
  FCT `fct_zal_a1` (0.71 %, per-node map), `fct_zal_a2` (0.64 %, per-elem map), `fct_qr4c_v` (0.58 %, per-node — vertical
  reads `valsAB[nz±1,±2]` are INPUTS → flat-lever-able), `fct_grad_elem`, `fct_LO_final`, `fct_f2d_v`, `fct_ale_recon` (each
  0.4–0.9 %). All in `fesom_tracer_adv.cpp`, validated by `tradv` (L37 capture-before on `values`+`valuesold`).
- **Bucket B (true scratch/coupling):** `momadv_vert` (0.54 %, `wu[nz]-wu[nz+1]` coupling + element-gather-into-scratch);
  FCT `fct_zal_a34` (1.03 %, `tvmax[]` cluster with vertical 3-layer dependency on WRITTEN values).
- **Bucket C (TDMA, Lever C only):** `impl_vert_diff_tracers` 1.61 %, `impl_vert_visc` 0.69 %, `fer_solve_gamma` 0.47 %.
- **Bucket D (scatter):** the FCT `*_h` edge scatters (`mfct_h` 1.59 %, `eud_fill` 1.73 % is per-edge+node-gather, B-ish),
  `momadv_horiz`, `visc_bidiff`, `ale_vvel_scatter`, redi edges.
- **The biggest single phase, `13_fct` (22.8 %), is ~24 small mixed-bucket sub-kernels** — the clean bucket-A ones sum to
  ~3–4 % but each is <0.9 % and they share scratch inside one giant function → higher per-kernel risk, lower per-kernel ROI than
  the momentum/GM kernels just done. Attack FCT bucket-A in a focused M5.20 pass with the `tradv` verify on each.

Tooling added: `jobs/job_ng5_m519_ab`, `jobs/job_ncu_m519_ab` (NCU_REGEX = enclosing-fn names `compute_vel_rhs|compute_sigma_xy|compute_neutral_slope`), `scripts/ncu_coalesce_summary.py`. Lesson **L64**.

## §M5.20 — the PCIe track: per-field attribution refutes Phase-B; `hnode_new` + `sw_3d` device-residency = −36 % deep_copy

The orthogonal lever (the residual 3.41 GB/step `deep_copy`). **Measured first** (the campaign discipline). nsys settled that the memops are ~100 % PCIe (HtoD 56 % + DtoH 43 %; device↔device only 0.04 %). Then **`FESOM_SYNC_LOG`** (a scoped CMake option added this session → `SYNCLOG D2H/H2D <label> <bytes>` per actual Field sync; `jobs/job_ng5_synclog`, NG5 dist_16 rank0) attributed the 3413 MB/step per field:

| field | dir | MB/step | what |
|--|--|--|--|
| `hnode_new` | H2D×2 + D2H | **778** | layer thickness, 3× host round-trip/step |
| `forcing.sw_3d` | H2D×2 | **519** | shortwave-penetration 3-D, host-computed + pushed 2× |
| `kpp.ghats` | D2H | 256 | "ghats stays host" (M5.7) |
| `tracers.S` | D2H | 249 | host salinity floor (L36/L39) |

**The 8 JRA55 forcing fields are only ~7 MB/step each** → the prompt's §6 **Phase-B-forcing suggestion was REFUTED by the measurement** — it would have given ~0. The halo'd fields show count/step≈0.04 (startup only) → halos are already device-resident. *Measuring before building is why this session didn't waste effort on the wrong lever.*

### Flip 1 — `hnode_new` device-resident (the M5.9-pin placebo pattern + a step-1 catch)
The 3 syncs (`fesom_step.cpp` 12a-OUT `sync_host` + the substep-1/13 `modify_host`+`sync_device` re-pushes) served only verify-only C-twins + the M5.13 "self-containment" defensive re-push — the device kernels read `hnode_new` on device. **First attempt removed all 3 → crash** (`CG_kk pp·App=-nan` at step 1): substep-1b GM reads `hnode_new` on device *before* the 12a thickness kernel runs, and at step 1 that value is the IC (host-written) → the device copy was alloc-zeros → NaN. **Fix:** keep device-resident (step ≥2 holds last step's 12a value = identical to the old re-push), but **seed the device copy from the IC at step 1 only** (`if (step_n==1)`); the 12a-OUT + substep-13 syncs stay removed. ⚠️ **LINFS-SPECIFIC** (hnode_new ≡ hnode, device-computed) — guarded at all 3 sites: **zstar must revisit `hnode_new`'s rail** when it makes the thickness a genuinely-evolving field (per the user). The gate caught the crash; the fix passes.

### Flip 2 — `sw_3d` device port (`fesom_cal_shortwave_rad_kk`)
`sw_3d` was host-computed (`fesom_cal_shortwave_rad`, the Jerlov penetration profile) + pushed 2× (substep-3 H2D for KPP, substep-13b re-push for tracer-diff — the latter a placebo, sw_3d is forcing set once/step). **Ported the 3-D profile to a per-surface-node device kernel** (`fesom_bulk.cpp`); the `heat_flux += swsurf` side effect stays on host (small nod2D op). `chl` pushed to device on update (const-once / monthly). Both pushes removed → **zero sw_3d sync sites**. exp/log10 on device → **Serial bit-identical** (same libm on the Serial CPU backend), CUDA climate-close (EOS-class last-ULP divergence).

### Validation — all PASS
- **Serial:** pi np1+np2 bit-identical (driver sanity; the seeds/sw_3d are no-ops/unexercised under pi). The decisive sw_3d check = **fresh CORE2 Serial oracle == saved M5.19 oracle, ALL FIELDS BIT-IDENTICAL** → the `sw_3d` device kernel equals the host `cal_shortwave_rad` op-for-op on Serial (and hnode_new is a Serial no-op).
- **CUDA fidelity gate** (`--fresh-oracle`, CORE2 dist_8 ice-active): **PASS**, worst max|Δ|=3.870e-03 (T 1.4e-3) — same floor as M5.16/M5.18, zero staleness regression (the gate caught Flip-1's first-attempt crash).
- **A/B (same-node, NG5 dist_16):** 2.1523 → **1.7741 s/step = −17.6 %** (prof corroborates 2.2099→1.8292). hnode_new alone was a clean **−11.4 %** on its own allocation (2.0138→1.7843).
- **deep_copy: 3413 → 2176 MB/step = −36.3 %** (hnode_new −742, sw_3d −496) — the clean cross-allocation-proof of both flips.
- **1-yr CORE2 CUDA climate PASS** (`m32_cuda_m520_1yr`, 17280 steps, clean T[-2.01,31.27] S[3.95,41.05]): **vs m519 apples-to-apples — every field corr=1.00000, bias O(1e-6–1e-7), RMS O(1e-4–1e-5) → M5.20 ≡ M5.19 to the D22 floor, ZERO climate cost** (the sw_3d device-exp perturbation is climatically invisible). Science budget vs Fortran preserved: sst/ssh 1.00000, sss 0.99996, a_ice/m_ice 0.99997 (same as M5.19). **M5.20 COMPLETE.**

### Remaining PCIe headroom (2176 MB/step) — for M5.21
`kpp.ghats` (256 D2H — the M5.7 "ghats stays host" decision) + `tracers.S` (249 D2H — the L36/L39 host salinity floor) are the next two; both are *deliberate* host-stays that need their host consumer ported/reconsidered (harder than placebo removal). Then ice-FCT values (~18 MB/step ×5) + startup. Tooling: `jobs/job_ng5_synclog` (per-field PCIe), `jobs/job_ng5_m520_ab`, the `FESOM_SYNC_LOG` CMake option. Lesson **L65**.
