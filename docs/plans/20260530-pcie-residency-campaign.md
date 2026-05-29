# M5.13 — NG5 device-residency / PCIe-reduction campaign

**Status:** PLANNED (handoff for the next session). Branch off `m512-fusion @ fc156c3`.
**Charter doc:** `docs/SCALING_NG5.md` § "nsys decomposition" + § "Implication — the production
lever is DEVICE RESIDENCY, not Lever C". **Lesson:** `docs/KOKKOS_PORTING_LESSONS.md` L56.
**Generated:** 2026-05-29 from a 6-agent codebase survey (device-halo infra · sync rails ·
flip-target inventory · validation · measurement · lessons). Every file:line below was read
from source on branch `m512-fusion`; re-verify a line if the file moved.

---

## 0. The mandate (why this campaign exists)

An `nsys` CUDA trace on the production mesh **NG5 (7.4 M nodes, 70 levels), dist_16, rank 0,
8 steps** (job `25227869`, snapshots off, 16.94 s/step) settled the bottleneck question that
the per-phase wall timer could not:

| step component | s/step | % of step |
|:---|---:|---:|
| **PCIe `cudaMemcpy` (H2D 8.13 + D2H 4.61)** | **12.74** | **75 %** |
| MPI / `cudaDeviceSynchronize` / host | 3.01 | 18 % |
| **GPU kernels (all compute)** | 1.19 | **7 %** |

`cudaMemcpy` is **90.6 % of all CUDA API time**; **~4 575 transfers/step**, H2D-dominant ⇒ this
is **host-staged halo unpack + re-push** (`sync_host` → MPI → `sync_device`), **not** file I/O
(disabled). **The GPU is PCIe-starved, idle ~93 % of the step.** The lever is **DEVICE
RESIDENCY**: flip the remaining host-staged 3-D halos to the on-device GPU-aware-MPI path
(`fesom_halo_field`) and eliminate host-op syncs that have a viable device twin.

**What this campaign is NOT:** Lever C (rank-1 → `View<double**>` coalescing) and launch-fusion
touch the **7 %** compute, not the **75 %** PCIe — both stay shelved. See L56's two method
sub-lessons: (a) the per-phase wall timer cannot separate compute from PCIe inside a phase — use
nsys kernel-vs-memcpy; (b) profile at the **production** mesh, the dominant phase changes
CORE2 → NG5 (nod2D-halo-latency → nod3D-PCIe-bandwidth — the *same* data-movement wall, different
flavor). Figure: `docs/figures/nsys_ng5_breakdown.png`.

---

## 1. Goal & success criteria

- **Primary:** reduce the NG5 per-step PCIe `cudaMemcpy` (currently 12.74 s/step) by flipping the
  remaining host-staged nod3D/elem3D halos to device residency. Each un-flipped 3-D halo is a
  full-field **~259 MB** sync at NG5 (462 k nodes/rank) — cheap at CORE2 (16 k/rank), dominant at NG5.
- **Headline metric:** the node-for-node **GPU/CPU ratio at NG5** (currently 3.8×). L56 flags this
  as "an upper bound on the *current* device-residency, NOT a hard floor." Re-measure with
  `jobs/submit_ng5_scaling.sh` after the campaign; closing half the PCIe could move NG5 toward ~2×.
- **Fast iteration metric:** CORE2 dist_8 s/step via the internal loop timer (currently ~0.478),
  measured **same-day before/after** (§7). Note: many of these halos are *cheap at CORE2* (that's
  why M5.1 left them), so CORE2 may show small gains while NG5 shows large ones — **the NG5 nsys
  re-trace + the NG5 scaling re-measure are the real acceptance, not CORE2 s/step.**
- **Hard gate (every flip):** see §6. Serial bit-identity is preserved by construction (Approach B);
  the risk is CUDA-only staleness, caught by the CORE2-active-ice fidelity gate + all-fields diff.

---

## 2. The infrastructure (what you flip *to*)

### 2.1 `fesom_halo_field` — the dispatcher (`src/fesom_halo_device.hpp:94-113`)

```cpp
inline void fesom_halo_field(fesom::Field &f, fesom_halo_kind kind,
                             int n_levels, int n_components, fesom_partit *p,
                             std::size_t base_off = 0);
```
- `f.modify_device()` always; `if (!p || p->npes<=1) return;` (device already holds the data).
- `#ifdef KOKKOS_ENABLE_CUDA` + `fesom_halo_device_active()` → device pack → **GPU-aware MPI on
  device pointers** → device unpack (`fesom_halo_exchange_device`, `.cpp:120-199`). No full-field PCIe.
- Else (Serial/OpenMP, or CUDA with `FESOM_HOST_HALO=1`) → the **exact legacy host bracket**
  `sync_host(); fesom_halo_exchange(h_checked()+base_off,…); modify_host(); sync_device();`.
- `kind` ∈ `FESOM_HALO_{NOD2D,NOD3D,ELEM2D,ELEM3D,ELEM2D_FULL}` (`src/fesom_halo.h:31-37`);
  `stride = n_levels*n_components`. `base_off` = the **slab-offset** for one channel of a
  multi-channel field (KPP `blmc[3]`, `diffK[2]`).
- **`fesom_halo_device_active()`** (`.cpp:22-34`): CUDA build → true unless `FESOM_HOST_HALO=1`
  (the A/B regression toggle, same binary). Non-CUDA → hard false (Approach B: the Serial oracle's
  call site is byte-for-byte the legacy bracket, zero risk by construction).
- **Approach B** = device path compiled+used ONLY under `KOKKOS_ENABLE_CUDA`. Serial = unchanged.

### 2.2 The sync rails (`src/fesom_field.hpp`) — what "flipping" means mechanically

A `fesom::Field` wraps a `Kokkos::DualView<T*, LayoutRight>` + a 3-state `auth_` tag
(`Synced/Host/Device`). Rails (`fesom_field.hpp:72-75`): `modify_host()/modify_device()` mark a
space dirty (+ set `auth_`); `sync_host()/sync_device()` lazily `deep_copy` **only if** the
opposite space is newer (else no-op), clearing the tag. **On Serial/OpenMP host==device → all
syncs are no-ops** → Serial is the strict bit-identity oracle, unchanged by any flip.

The legacy **split-rail** (a host-staged halo) is:
```
[producer device kernel] → f.sync_host()            ← OUT rail: full-field D2H PCIe copy
                           fesom_exchange_nod3D(f.h_checked(),…)   ← host pack/MPI/unpack
[before next device reader] f.modify_host(); f.sync_device()      ← IN re-push: full-field H2D PCIe copy
```
**Flipping field X (the L48 recipe):**
1. **Replace** the OUT-rail `sync_host()` + `fesom_exchange_*(X.h_checked()…)` with a single
   `fesom_halo_field(X_fld, kind, nl, nc, p)` right after X's producing device kernel.
2. **REMOVE** the downstream IN re-push (`X.modify_host(); X.sync_device();`) — X is left
   DEVICE-authoritative (owned + halo current); every later **device** kernel reads it directly.
3. Keep any *genuine host reader's* targeted `sync_host` (the M5.9 lesson — e.g. `uvnode:338`).

Worked example already in tree (`uv_rhs`, M5.4a): `fesom_step.cpp:455, 479, 504` — three
`fesom_halo_field(dyn->uv_rhs_fld, FESOM_HALO_ELEM3D, nl, 2, p)` across substeps 4-6, no IN re-push.
Contrast `uv_rhsAB` (`:449,:456`) deliberately left host-staged — that's a target now (§4).

---

## 3. The four flip patterns (decision tree) — L48

1. **Self-contained-replace** — the whole bracket sits in one place (CG `exch`, FCT internals):
   replace wholesale with `fesom_halo_field`. Trivial.
2. **Split-rail replace-OUT + remove-IN-repush** — the common ocean case (§2.2). OUT-rail at the
   producer, IN re-push at the consumer (sometimes substeps later). **You MUST trace every consumer**
   (grep the field's `.modify_host()/.sync_device()` and its device-kernel reads). Remove each IN
   re-push; leave a real host reader's sync.
3. **Smoothing → device-kernel (lever B)** — the halo is host-bound because a host `smooth_nod3D`
   runs between sweeps. Port to `fesom_smooth_nod3D_kk` (`src/fesom_eos.cpp:488`): per-node patch
   gather, 2 race-free kernels (gather then scale — separate to avoid the arr read-write race),
   device-halo between sweeps, remove consumer re-pushes. (Already done for `bvfreq`/`blmc`.)
4. **Slab-offset via `base`** — for a multi-channel field, exchange each slab on its own:
   `fesom_halo_field(…, p, base + (std::size_t)s*stride)` (see `fesom_eos.cpp:559`, `kpp.cpp:1549`).

A host op with **no** device consumer this step should NOT be force-flipped (L36) — that adds a
round-trip serving only the host. Decide per-field by grepping the actual readers.

---

## 4. The ranked target worklist

All sites are in `src/fesom_step.cpp` unless noted. "snap-out" = also a snapshot output ⇒ L48
I/O-staleness trap applies (add a snapshot-gated pre-I/O `sync_host` in `fesom_main.cpp:1283-1295`).
Recommended order is the milestone sequence in §9.

### TIER 1 — clean, high payoff (a proven device halo-reader + removable re-pushes)

| target | site(s) | kind / size | re-pushes to drop | snap-out | notes |
|:---|:---|:---|:---|:---:|:---|
| **ALE `cfl_z`** | `:711`/`:713` (+IN `:717`) | NOD3D | 1 (`:717`) | no | **lowest-risk** clean producer→consumer hop (cflz→wvel_split) |
| **EOS `hpressure`** | `:189`/`:204` (+IN `:307`) | NOD3D | 1 (`:307`, PGF) | no | PGF reads it at halo on device (the `:307` comment proves it) |
| **EOS `sw_alpha`/`sw_beta`** | `:191-192`/`:214-215` | NOD3D | 2 each (`:246-247`,`:359-360`) | no | removes 3 PCIe touches each/step (GM + KPP re-pushes) |
| **GM `fer_gamma`** | `:285`/`:287` (+IN `:290`) | NOD2D×nl×2 | 1 (`:290`) | no | `fer_gamma2vel_kk` reads it at element vertices (halo) on device |
| **GM `slope_tapered`** | `:266`/`:270` | NOD2D×(nl-1)×3 | 2 (`:781`,`:900`) | no | read at halo by Redi `diff_hor` + trdiff K33 |
| **GM `Ki`** | `:277`/`:281` | NOD2D×nl×1 | 2 (`:782`,`:901`) | no | paired with `slope_tapered` |
| **GM `fer_uv`** | `:294`/`:296` | ELEM2D_FULL×nl×2 | 2 (`:694`,`:743`) | no | read by ALE vert_vel + bolus on device; same family as `uv_rhs` |
| **`uv_rhsAB`** | `:449`/`:456` (+IN `:443` next step) | ELEM3D×nl×2 | 1 (`:443`) | no | cross-step AB2 history; sits beside flipped `uv_rhs` |

GM runs **every step** (GM ON by default, L34) → the GM quartet (`fer_gamma`,`slope_tapered`,`Ki`,
`fer_uv`) is the highest *aggregate* (1-3 PCIe touches each).

### TIER 1b — high but tangled (sequence after Tier 1)

| target | site(s) | kind / size | snap-out | entanglement |
|:---|:---|:---|:---:|:---|
| **`uv` after update_vel** | `:563`/`:564` (+IN `:569`) | ELEM3D×nl×2 | **yes** | `compute_hbar` reads at halo; high fan-out (ALE/bolus/FCT readers) |
| **ALE `w`** | `:696`/`:699` (+IN `:708`) | NOD3D | **yes** | bolus 13a/13c augment `w` on device then sync_host (`:746-748`,`:963-964`) |
| **ALE `w_e`** | `:720`/`:723` | NOD3D | no | read on device by FCT; bolus round-trips at `:742`,`:747`,`:776` |
| **ALE bolus uv/w/w_e pulls** | `:740-748`,`:961-965` | mixed | (uv yes) | eliminate the `sync_host` pulls bracketing FCT once underlying fields are resident |
| **ALE commit `hnode`/`helem`** | `:975-976`/`:978-979` | NOD3D / ELEM3D | no | evolving mesh read by ~6 next-step device kernels (EOS `:178`, ivisc `:499`, ssh `:543`, GM `:254-255`, vert_vel `:693`, FCT `:777-779`) that each re-push — highest cross-step fan-out |
| **Tracer `T`/`S` values post-FCT+Redi** | `:816`/`:849` | NOD3D | **yes** | trdiff IN re-push `:895-898`; the Redi kernel owns an *internal* device halo (`gm.cpp:1792,1918`) — trace which the driver one duplicates |
| **Tracer `T`/`S` values post-trdiff** | `:904-905`/`:908-909` | NOD3D | **yes** | cleaner (race-free TDMA); **S is pinned by the host salinity floor** — S can only shed within-step re-pushes, T flips clean |

### TIER 3 — low / uncertain payoff (skip unless an NG5 re-trace flags them)

`GM sigma_xy` (`:259`/`:261`, halo not read on device) · `GM fer_K`/`fer_C` (`:275-281`, read at
owned; `fer_C` is cheap nod2D) · `EOS density_m_rho0` (`:188`/`:203`, unclear device halo-reader,
snap-out, maybe smoother-bound) · `KPP ghats` (`kpp.cpp:1586-1587`, gated off in CORE2 per code
comment) · the **nod2D SSH driver halos** `ssh_rhs`/`d_eta`/`ssh_rhs_old`/`hbar` (`:551,:557,:574,
:575` — PCIe-cheap single-level; the CG itself is already device, M5.1).

---

## 5. What MUST stay host (do NOT flip)

- **Salinity floor (L39, `:931-950`):** host clamp of `S.values < 0.5 PSU` over [0, myDim+eDim),
  guarding the EOS NaN→CG abort. Next-step EOS re-syncs S → propagates for free. **Pins S's
  OUT-rail `sync_host` at `:905`** → S cannot be fully device-resident across the step boundary;
  the S values-halo flips can only drop *within-step* re-pushes. (Porting the floor itself to a
  device twin to kill the sync is a *judgment call*, not a default — trace whether the sync it
  forces is the expensive full-field nod3D one at NG5 before attempting.)
- **`uvnode → fesom_bulk_compute` (L50, `:338`):** the SOLE genuinely-required M5.9 sync.
  `fesom_bulk_compute` (`fesom_bulk.cpp:261-262`, every CORE2 step) reads `uvnode`'s surface row on
  the host for wind stress. Proven the only real host reader by the NaN-poison discriminator. KPP
  reads `uvnode` on device via its resident halo. **Do NOT remove.** (A persistent-buffer
  surface-only refresh is a ~1 % future micro-opt, not this campaign.)
- **`eta_n` inline map (`:650-659`):** trivial nod2D host loop on host-current `hbar`/`hbar_old`;
  re-synced as a substep-4 IN-push (`:444`). Leave alone.
- **All `fesom_exchange_*`/`fesom_halo_exchange` calls inside the `FESOM_KK_VERIFY` C-twin
  functions** (`gm.cpp:189,294-295,453-455,599,689,1064`; `momentum.cpp:263,1165-1197,1321-1322`;
  `tracer_adv.cpp:472,1070-1071,1233`; `ssh.cpp:254,422-537`; the ice twins) — these are the host
  reference oracle run only under verify, NOT the production device path. Do NOT flip them.
- **Setup-time exchanges** (`fesom_mesh.cpp`, `fesom_partit.cpp`, `fesom_phc.cpp` IC load) — not
  per-step. Leave alone.

### Already flipped (do not re-target)
`uv_rhs` ELEM3D (M5.4a) · `pgf_x/pgf_y` ELEM3D + `uvnode` NOD3D (M5.4b) · FCT internals
`fct_LO/tr_xy/fct_plus/fct_minus` (M5.4c, `tracer_adv.cpp:1547,1565,1766-1767`) · `bvfreq` +
device smoother (M5.5, `:206`/`eos.cpp:559`) · KPP `blmc[3]`/`diffK[2]`/`viscA` (M5.5/M5.7,
`kpp.cpp:1549,1583-1585`) · `Kv`/`Av` (M5.7b, `:397-398,424-425`) · EVP `uice`/`vice` (M5.8) ·
`momentum_adv` `uvnode_rhs` + visc_filt `u_b`/`v_b` (M5.1, `momentum.cpp:417,1456-1457`) · CG
`pp/rr/X/d_eta` (M5.1, `ssh.cpp:705-716`) · GM Redi internal `tr_xy`/`tr_z` (`gm.cpp:1792,1918`) ·
ice FCT (`ice_fct.cpp`). And the L48 pre-I/O sync already covers `pgf_x/pgf_y/bvfreq/Kv/Av`
(`fesom_main.cpp:1283-1291`).

---

## 6. The validation gate (every flip must pass)

Serial bit-identity is preserved *by construction* (Approach B). The real risk is a **CUDA-only
stale read**. Ladder, in order:

1. **Serial per-kernel verify `max|Δ|==0`** on `build-serial` for the affected key. Keys →
   kernels: `eos`(`eos.cpp:903`), `pgf`(`eos.cpp:686`), `pp`(`pp.cpp:471`), `kpp`(`kpp.cpp:1667`),
   `vrhs`(`momentum.cpp:571`), `ivisc`(`momentum.cpp:1003`), `vfilt`(`momentum.cpp:1522`),
   `ale`(`ale.cpp:595…661`), `gm`(`gm.cpp:1627…2086`), `tradv`(`tracer_adv.cpp:1877`),
   `trdiff`(`tracer_diff.cpp:654`), `ssh`(`ssh.cpp:940`); ice (CORE2-only):
   `icemap/evp/icefct/icethermo/iceflux`. Run: `FESOM_KK_VERIFY=<key> ./build-serial/fesom_port
   <pi> /tmp/out 100 20 10`; PASS = `grep FESOM_KK_VERIFY= run.log | grep -v 0.000e+00` empty.
   ⚠️ `pp` substring-collides with `kpp` — the dispatch guards it (`fesom_step.cpp:152-157`).
2. **Serial full-run bit-identity** vs golden — `scripts/diff_snap.py` (zero-tolerance,
   takes **DIRECTORIES** not files, L19) on pi **np1 AND np2**. np2 needs
   `export OMPI_MCA_btl_vader_single_copy_mechanism=none` (L18) and diffs against
   `/scratch/a/a270088/pi_np2_ref_m13_nocma` (the `…_m12` oracle is CMA-tainted). np1 golden:
   `docs/reference/c_baseline_snapshots/pi`.
3. **SYNCCHECK clean** — `build-synccheck` (Serial + `-DFESOM_KK_SYNCCHECK`). `Field::h_checked()`
   aborts on a host read while device-authoritative (the exact failure a flip can introduce). Run
   pi np1/np2; clean exit. *Caveat:* Serial host==device, so SYNCCHECK cannot catch a CUDA-only
   stale read — that's step 5.
4. **CUDA A/B at the run-to-run noise floor** — same `build-cuda` binary, toggle `FESOM_HOST_HALO`.
   CUDA is non-deterministic (atomic-scatter reassociation ~1e-13), so the gate is
   **host-vs-dev ≈ host-vs-host** (run host-staged twice for the floor), NOT byte-identity.
5. **`scripts/gpu_fidelity_gate.sh` — the MANDATORY pre-commit gate** (CORE2-active-ice,
   CUDA-vs-Serial). Builds the Serial oracle `serref_core2` (= ground truth, bit-id to the C twin),
   runs the CUDA device-halo leg, diffs field-by-field vs per-field ceilings
   (`gpu_fidelity_check.py`: T/u/v/w/eta 1e-2, Kv/Av/density/bvfreq/pgf 1e-1, ice uice/vice/a_ice/
   m_ice/m_snow 5e-2, h_ice/h_snow 1e-1). **pi is
   INSUFFICIENT** (no ice, analytical forcing → a stale-host bug stays at 1e-17 — exactly how the
   M5.9 bug hid for ~8 commits). Use `--fresh-oracle` after any Serial-side change (and this
   campaign edits `fesom_step.cpp` heavily → **expect to re-bake the oracle**, L51).

### ⚠️ The I/O-staleness trap (L48) — the single most-repeated regression here
Flipping a **snap-out** field removes its OUT-rail `sync_host` → device-authoritative at I/O →
the rank-0 gather reads STALE host (model fine, **only the netCDF diagnostic wrong**). NOT caught
by Serial or SYNCCHECK. **Fix:** add the field to the snapshot-step-gated pre-I/O `sync_host` block
in `fesom_main.cpp:1283-1295`. **Rule: after any flip, diff ALL output fields, never a subset.**
The full snap-out set (`fesom_io.cpp:435-457`): `T,S,eta_n,w,u,v` + `density_m_rho0,bvfreq,pgf_x,
pgf_y,Kv,Av` + ice `a_ice,m_ice,m_snow,uice,vice,h_ice,h_snow`.

---

## 7. Build / run / profile / measure

**Output rule: write to `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB quota; one NG5
field ≈ 8.5 GB). CLI: `fesom_port <mesh> [out] [dt] [nsteps] [snap_every] [phc] [jra55_year]`
(`fesom_main.cpp:269`); `snap_every=-1` disables I/O (mandatory at NG5 — rank-0 gather OOMs ~66 GB);
`jra55_year=1958` = active ice, `0` = off.

**Builds** (dirs already configured; just rebuild the target):
- `build-cuda` — **`source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware — REQUIRED;
  env.sh's 4.1.2 is `--without-cuda` and SEGFAULTs on device ptrs). `cmake --build build-cuda
  --target fesom_port -j 16`. (`Kokkos_ARCH_AMPERE80=ON`, nvcc_wrapper.) ⚠️ `docs/RUN_GPU.md` §5 is
  STALE (shows old 4.1.2 dance) — use env_cuda.sh.
- `build-serial` — `source ./env.sh`; the bit-identity oracle.
- `build-synccheck` — Serial + `-DFESOM_KK_SYNCCHECK=ON`.
- `build-omp` — OpenMP (M3.2 leg; `OMP_NUM_THREADS=1` for bit-id vs Serial).

**GPU runtime env** (from the job scripts): `OMPI_MCA_pml=ucx OMPI_MCA_btl=self
UCX_NET_DEVICES=mlx5_0:1 UCX_MEMTYPE_CACHE=n OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_io=romio321
HDF5_USE_FILE_LOCKING=FALSE`. SLURM: `--gres=gpu:4 --ntasks-per-node=4 --gpu-bind=none` on
partition `gpu`; `dist_N` ⇒ N ranks ⇒ N GPUs ⇒ ceil(N/4) nodes. Do NOT use
`--gpus-per-task`/`--gpu-bind=single` (breaks the rank→device map).

**Profile:**
- `FESOM_STEP_PROFILE=1` (`fesom_profile.{hpp,cpp}`): per-phase wall + per-kernel
  (Kokkos-callback, fence-bounded) + `deep_copy` count/MB (coarse PCIe proxy). ⚠️ phase wall =
  kernel+PCIe+MPI together — cannot isolate PCIe (the L56 mistake).
- **`nsys` = the decisive PCIe instrument** — reports `cuda_gpu_kern_sum` (the 7 % compute),
  `cuda_gpu_mem_time_sum` (H2D/D2H = PCIe), `cuda_api_sum` (blocking `cudaMemcpy` wall). Jobs:
  `jobs/job_nsys_ng5` (the production-mesh trace) + `jobs/job_nsys_core2_np1` (fast kernel
  drill-down). Re-stat an existing rep: `nsys stats --report
  cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum --format column <rep>`.
- `ncu` (`jobs/job_ncu_fctgm_ng5` + `jobs/ncu_rank0.sh`) — kernel SOL only; **GPU-kernel-only,
  CANNOT see the 75 % host-staged PCIe** (read durations in the report's stated unit — an earlier
  read mistook ms for µs). Use only for SOL of the 7 %.
- `FESOM_CG_PROFILE` — CG share (already known ~0.3-5 %, not a target).

**Measure payoff (CORE2 = fast iteration):** the internal startup-free loop timer
(`fesom_main.cpp:1298-1306`) prints `loop timing: N steps … s/step`; set `FESOM_PRINT_EVERY=999`.
**SAME-DAY-BASELINE RULE (L40):** ~5 % node/contention noise — never compare to a number from
another week. Build BEFORE and AFTER binaries, run them back-to-back in the SAME SLURM allocation,
2 reps interleaved (template `jobs/job_m512{b,d,f}_perf_core2_dist8`). The single-binary host-vs-dev
toggle job `jobs/job_gpuaware_time_core2` (`FESOM_HOST_HALO=1`) measures the device-halo gain
directly. For the **real** acceptance: re-run `jobs/job_nsys_ng5` (PCIe ↓?) +
`jobs/submit_ng5_scaling.sh` (GPU/CPU ratio ↓?).

---

## 8. Traps to carry in

- **NaN-poison discriminator (L50, `jobs/job_poison_dev`)** — the tool for "is this `sync_host`
  load-bearing or a placebo?". KEEP the sync (so device scheduling is byte-identical), overwrite the
  host copy with NaN *after* it *without* `modify_host`. Only a genuine HOST read sees the NaN. A
  leave-one-out toggle is CONFOUNDED (it also removes a fence → reshuffles atomics → fake "needed").
  Use this before deleting any sync you suspect is redundant.
- **Cached `static Kokkos::View` destructs AFTER `Kokkos::finalize()` → SIGABRT.** Any new cached
  device View a flip introduces MUST be **file-static + freed before finalize**
  (`fesom_main.cpp:1349-1351`: `fesom_halo_device_free(); fesom_ice_evp_free();` then finalize).
- **GPU atomic-scatter run-to-run non-determinism (D22).** Never expect device==host byte-identity;
  validate at the host-vs-host noise floor. Scatters in flight: `visc_filt_bidiff`+`momentum_adv`,
  `vert_vel`, Redi `diff_hor`, the 3 FCT scatters.
- **"The verify can't catch its own input" (L38).** A verify's C-twin reads the HOST alias — if a
  flip leaves a field device-only-authoritative, the C-twin advects with stale host and the verify
  FALSELY fails (or a RMW field corrupts both kernel and twin identically → false PASS). For a
  flipped RMW field (`values`, `uv`) keep the host mirror current for the verify, or update the
  twin; the real gate for those is the end-to-end Serial `diff_snap`.
- **L36 — don't force-flip a field with no device consumer this step** (adds a round-trip). Grep
  the actual readers first.
- **L51 — re-bake the gate oracle** (`--fresh-oracle`) since the campaign edits `fesom_step.cpp`
  heavily (even no-op sync changes can reorder host loads → ULP drift at a few bathymetry-edge cells).

---

## 9. Recommended campaign sequence (milestones)

Each milestone = one logical flip group, validated by §6, committed separately. Re-trace
`job_nsys_ng5` after every 2-3 milestones to confirm PCIe is actually dropping (CORE2 may be flat).

- **M5.13a — `cfl_z`** (lowest risk: clean single hop, not snap-out). Proves the per-flip loop end
  to end on a trivial target.
- **M5.13b — EOS `hpressure` + `sw_alpha`/`sw_beta`** (clean, real device readers, no L48 trap;
  removes up to 3 PCIe touches each/step).
- **M5.13c — the GM quartet `fer_gamma` + `slope_tapered` + `Ki` + `fer_uv`** (highest aggregate,
  runs every step; 1-3 re-pushes each). The big nod3D win.
- **M5.13d — `uv_rhsAB`** (beside the already-flipped `uv_rhs`, same kind/size).
- **M5.13e — ALE `w`/`w_e` + the bolus uv/w/w_e round-trips together** (tangled;
  needs the bolus output kept device-resident; `w` is snap-out → L48).
- **M5.13f — ALE commit `hnode`/`helem`** (high cross-step fan-out; removes several next-step IN
  re-pushes — verify all ~6 downstream readers).
- **M5.13g — tracer `T`/`S` values halos + `uv` after update_vel** (all snap-out → L48 block; S
  pinned by the salinity floor, so T flips cleaner). Do this last (highest blast radius).
- **Acceptance:** NG5 `job_nsys_ng5` re-trace (PCIe % down) + `submit_ng5_scaling.sh` re-measure
  (GPU/CPU ratio down from 3.8×). Update `docs/SCALING_NG5.md`, `docs/GPU_FIDELITY.md`, L56→Lnn,
  and the handoff.

---

## 10. References

**Charter / evidence**
- `docs/SCALING_NG5.md` — § nsys decomposition + § the DEVICE-RESIDENCY implication (the charter).
- `docs/figures/nsys_ng5_breakdown.png` — PCIe 75 % / GPU 7 % step decomposition.
- nsys ground truth: `/work/ab0995/a270088/port2/kokkos_gpu_runs/nsys_ng5/{ng5.nsys-rep,stats.txt}`.
- `docs/KOKKOS_PORTING_LESSONS.md` — L36, L38, L47, L48, L49, L50, L51, L56; D22 in
  `docs/SCATTER_STRATEGY.md`. `docs/GPU_FIDELITY.md` §M5.1-§M5.9. `docs/SYNC_MAP.md` (the rails).

**Code**
- `src/fesom_halo_device.{hpp,cpp}` (`fesom_halo_field`, dispatch, `fesom_halo_device_free`).
- `src/fesom_field.hpp` (the sync rails, `h_checked`). `src/fesom_step.cpp` (the substep driver,
  the targets, `FESOM_KK_VERIFY` dispatch `:137`). `src/fesom_main.cpp:1283` (pre-I/O sync gate),
  `:1349` (free-before-finalize). `src/fesom_io.cpp:435` (snapshot output list). `src/fesom_eos.cpp:488`
  (`fesom_smooth_nod3D_kk`, lever B). `src/fesom_profile.{hpp,cpp}` (`FESOM_STEP_PROFILE`).

**Gates / jobs**
- `scripts/gpu_fidelity_gate.sh [--fresh-oracle]`, `scripts/gpu_fidelity_check.py`,
  `scripts/diff_snap.py` (DIRS, zero-tol).
- `jobs/job_core2_serial_ref` (→ `serref_core2`), `jobs/job_gpu_fidelity_dev` (→ `gate_dev`),
  `jobs/job_ice_verify_core2` (5 ice keys), `jobs/job_poison_dev` (NaN discriminator),
  `jobs/job_gpuaware_time_core2` (host-vs-dev timer), `jobs/job_m512{b,d,f}_perf_core2_dist8`
  (same-day before/after), `jobs/job_gpuaware_prof_core2` (`FESOM_STEP_PROFILE`+`CG_PROFILE`),
  `jobs/job_nsys_ng5`, `jobs/job_nsys_core2_np1`, `jobs/job_ncu_fctgm_ng5` + `jobs/ncu_rank0.sh`,
  `jobs/job_ng5_scaling_{gpu,cpu}` + `jobs/submit_ng5_scaling.sh`.

**References / oracles**
- pi np1 golden: `docs/reference/c_baseline_snapshots/pi`; pi np2 (nocma):
  `/scratch/a/a270088/pi_np2_ref_m13_nocma`.
- NG5 mesh: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5/` (7.4 M nodes, 70 levels, dt=180).
- Meshes: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/{core2,farc,dars,ng5}`.
