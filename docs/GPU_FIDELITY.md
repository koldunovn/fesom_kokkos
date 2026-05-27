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

### CUDA 2-yr — _TODO_
_(paste the `m32_climate_compare.py --label CUDA` table; note the measured s/step + total wall.)_

### OpenMP 2-yr — _TODO_
_(paste the `m32_climate_compare.py --label OpenMP` table.)_

### Verdict — _TODO_
_(PASS/FAIL vs the criteria above; if a field drifts, which scatter/reduction is the suspect and whether
edge-coloring / a deterministic reduction (M5) is warranted.)_
