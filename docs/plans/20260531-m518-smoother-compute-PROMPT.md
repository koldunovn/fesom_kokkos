# Next session — M5.18 (attack the COMPUTE+PCIe bulk; spearhead = the smoother `fesom_smooth_nod3D_kk`)

*Paste this whole file to start. Self-contained. Written 2026-05-30 at the close of the M5.17 measure-first gate.*

---

## 0. TL;DR — where we are

The whole FESOM2 **C→C++/Kokkos** port is device-resident (M0–M4: ocean + sea-ice on the GPU, Serial bit-identical to the C twin). The **M5.x GPU-perf campaign** drove NG5 dist_16 (7.4M-node production mesh, 4 GPU nodes) from **16.27 → 2.677 s/step** and the node-for-node ratio from **3.76× SLOWER than a CPU node → 1.63× FASTER**.

**M5.17 just CLOSED Lever B (MPI-comm overlap/aggregation) as a DEAD END — by measurement.** A barrier-isolation experiment (env-gated `MPI_Barrier` before every halo exchange + a per-rank Barrier/Waitall accountant; instrument pi np1+np2 BIT-IDENTICAL) split the per-step halo `MPI_Waitall` (0.288 s/step = 10.8% of step) into **79% load-imbalance idle / 21% comm**. The overlappable-comm **ceiling is 0.065 s/step = 2.4%** — not worth the invasiveness. The old "47% Waitall" was rank-0 wall lumping imbalance + setup, never recoverable comm. See `docs/GPU_FIDELITY.md` §M5.17, lesson L62, [[project-m517-mpi-comms]].

**So the real walls on the 2.677 s/step are now (1) load imbalance ~9% [Lever D, deployment-side] and (2) the GPU-compute + residual-PCIe bulk ~89% [this session].** The user's call: **attack the compute bulk, starting with the single most ridiculous kernel — `fesom_smooth_nod3D_kk`, the #1 GPU kernel at 25.7% of all GPU compute.**

**Branch:** continue on `m517-mpi-comms`, or cut `m518-smoother-compute` off it. ⚠️ The M5.17 halo-MPI-prof instrument + the §M5.17 docs are **UNCOMMITTED working-tree changes** (validated byte-clean; the user has not asked to commit). Decide with the user whether to commit M5.17 first or carry it.

---

## 1. THE TARGET — `fesom_smooth_nod3D_kk` (the area-weighted node-patch horizontal smoother)

**Where:** `src/fesom_eos.cpp:488` (device twin; host original `:425`). **Callers (every step):**
- `fesom_step.cpp:227` — `bvfreq` (N²), `n_smooth=1`, `nslab=1`.
- `fesom_kpp.cpp:1550` — KPP `blmc`, `n_smooth=3`, `nslab=3` (3 channels via `base`/`slab_stride`).

**What it does:** each owned node's value at level `nz` ← area-weighted mean of the 3 vertices of every surrounding element reaching `nz`: `arr(n,nz) = Σ_el area·(v0+v1+v2) / (3·Σ_el area)`. Per sweep = **2 kernels** (`fesom_smooth_gather` → `work`/`vol`, then `fesom_smooth_scale` → `arr = work·vol`; separate to avoid the read-then-write race), then a per-channel device halo. nsys: gather+scale = **140 instances each / 35 steps = 4 + 4 launches/step**; the gather is the heavy one (avg ~78 ms, min 46 / max 143 → huge variance).

### Why it's slow — read straight off the code (`fesom_eos.cpp:518-556`)
The kernel is **`RangePolicy(0, nslab*Nmy)` = ONE THREAD PER (slab, owned-NODE)**, with an INTERNAL loop over that node's levels `nz` (uln..nlnz, ~5–70) and elements (o0..o1, ~6). Three structural pathologies:

1. **UNCOALESCED memory (the big one).** Scratch is indexed `work[idx*NL + nz]` / `vol[idx*NL + nz]` and the field `arr[v*NL + nz]` is **node-major** (`NL=70` is the contiguous inner dim). Consecutive threads = consecutive *nodes* → their accesses stride by `NL=70` (560 B) → **32 separate cache lines per warp, ~zero coalescing.** Every memory transaction wastes ~7/8 of each 32-byte sector.
2. **WARP DIVERGENCE on depth.** Adjacent threads are adjacent nodes with wildly different column depths (shallow shelf ~5 levels vs deep ocean ~70) and element counts → the internal `nz`/element loops diverge hard across the warp. This is the source of the 46→143 ms per-launch spread.
3. **Per-step scratch alloc** (minor): `vol`/`work` Views allocated fresh per call (`nslab*Nmy*NL` doubles = 258 MB for bvfreq, 774 MB for blmc) — BUT Kokkos uses `cudaMallocAsync` (stream-ordered/pooled), measured at only ~0.5%/step. Hoist it for cleanliness, not speed.

### ncu baseline (NG5 dist_16, rank0, `jobs/job_ncu_smooth_ng5`, job 25247192, 2026-05-30) — MEASURED, the diagnosis is confirmed + sharpened
The `gather` kernel: **Compute (SM) 2.27 %** · **Memory 46.1 %** · DRAM 34.2 % · L2 46–47 % · **achieved occupancy 52.7 %** · **duration 48–145 ms (3× spread = the depth divergence)**. ncu's own OPT verdict: *"low compute AND memory-bandwidth utilization (both < 60 % of peak) → latency issues."* So the ALUs are **98 % idle waiting on scattered memory** — it is **latency-bound, not BW-bound**. The smoking gun is the coalescing metric: **global-STORE sectors/request ≈ 52** (catastrophic — the `work`/`vol`/`arr` writes at `idx*NL+nz` with consecutive-thread = consecutive-node stride `NL` → ~every store its own sector) vs **global-LOAD ≈ 2.3** (the `arr` reads are L2-cached by neighbor reuse, L2 throughput 46 %). **The uncoalesced STORES are the killer**, compounded by mediocre 52.7 % occupancy + the divergence. The per-(node,level) rewrite fixes all three at once: stores → coalesced (target ~2 sec/req), ~50× more threads → occupancy, one level/thread → no divergence. The `scale` kernel is lighter (4.3 ms, occ 91 %, same Compute 2.05 %) but also store-uncoalesced. **Re-run this exact job after the rewrite for the before/after.**

---

## 2. THE FIX — re-parallelize to ONE THREAD PER (slab, node, LEVEL); coalesced + divergence-free; BIT-IDENTICAL

**The key insight: the field is ALREADY node-major (`arr[node*NL + level]`), so the LEVEL is the contiguous dimension.** If threads span LEVELS (not nodes), a warp of 32 consecutive `idx = ((s*Nmy)+n)*NL + nz` is 32 consecutive levels *of the same node n* → all read `arr[v0*NL + nz0..nz0+31]` = **contiguous = COALESCED** (v0 is the same element-vertex for the whole warp), and write `arr[n*NL + nz0..+31]` / `work[idx]` contiguous = coalesced. The depth divergence vanishes (each thread does ONE level). **No global layout refactor needed** — this exploits the existing layout's level-contiguity. (This is the *local, low-risk* cousin of the heavyweight Lever-C `View<double**>` refactor; do this first.)

### The minimal rewrite (keep the 2-kernel read-then-write structure → no race, bit-identical)
- `RangePolicy(0, nslab*Nmy*NL)`; decode `s = idx/(Nmy*NL)`, `rem = idx%(Nmy*NL)`, `n = rem/NL`, `nz = rem%NL` (or a `MDRangePolicy<Rank<3>>` / a flat decode — benchmark both).
- **gather thread (s,n,nz):** mask out `nz < uln(n) || nz > nlnz(n)` (return). Else loop elements `k=o0..o1`; for each element with `nz ∈ [ule,nle]` accumulate `a*(arr[v0]+arr[v1]+arr[v2])` → `work[idx]`, and (sweep 0) `vol[idx] += a`; then `vol[idx] = 1/(3·vol[idx])`. **The element loop runs in the SAME `k` order → the per-(n,nz) float sum is byte-identical to the current code → Serial/OpenMP bit-identical.**
- **scale thread (s,n,nz):** `arr[sb + n*NL + nz] = work[idx]*vol[idx]` (masked to valid levels).
- Halo unchanged (per-channel `fesom_halo_field`, M5.17-deadlock-safe).
- **Hoist `vol`/`work`** to a persistent buffer (size `3*Nmy*NL`, allocated once — a file-static or `aux`-owned View, freed before `Kokkos::finalize()` per the M5.8 static-View trap).

### Refinements to benchmark (only if the flat version leaves perf on the table)
- **Warp-straddle:** a flat `idx` lets one warp per ~2.2 straddle two nodes (`70/32`). A `TeamPolicy` (league = `nslab*Nmy`, one team per node, team threads = levels) eliminates straddle and gives clean per-node coalescing + lets `vol`/`work` live in team scratch. More complex; measure whether it beats flat.
- **Fuse gather+scale via ping-pong:** read `arr_in`, write `arr_out = patchsum·vol` in ONE kernel, swap buffers per sweep (needs a 2nd field-sized buffer). Halves launches (8→4/step). Bit-identical (all reads from `arr_in`). Follow-on after the coalescing win is banked.
- **Masked-thread waste:** launching `Nmy*NL` threads wastes the shallow-node tail (avg depth < NL). Acceptable for a memory-bound kernel (coalescing + occupancy dominate); a compacted (n,nz)-pair list is a later micro-opt.

### Expected payoff
The smoother is ~25.7% of GPU compute ≈ **~0.30 s/step ≈ 11% of wall**. A memory-bound uncoalesced→coalesced flip typically wins 2–4× on the kernel → **~5–8% of the step** if it lands. MEASURE same-day (don't claim it).

---

## 3. VALIDATION LADDER — the rewrite is ARITHMETICALLY IDENTICAL → bit-identity MUST hold

The thread-mapping change does NOT change the per-(node,level) float accumulation order → Serial bit-identical is the gate, exactly like the M2.x ports.
1. **Serial per-kernel verify** `FESOM_KK_VERIFY` — bvfreq lands in the `eos` path, blmc in `kpp`. Confirm the post-smoother fields are `max|Δ|==0` on Serial (and OpenMP, since it's race-free — no scatter). Add a dedicated `smooth` verify hook if the existing keys don't isolate it.
2. **pi np1 + np2 BIT-IDENTICAL** vs `docs/reference/c_baseline_snapshots/pi` and `/scratch/a/a270088/pi_np2_ref_m13_nocma` (np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`, L18; `diff_snap.py` takes DIRECTORIES, L19). ⚠️ pi exercises the **bvfreq** smoother only if `N2smth_h` is on for the pi config — verify it actually runs (else the bvfreq path is pi-invisible like bulk/ice; fall back to a CORE2 Serial verify, L42).
3. **SYNCCHECK** (`build-synccheck`) clean.
4. **CORE2-active-ice CUDA fidelity gate** `scripts/gpu_fidelity_gate.sh --fresh-oracle` (edits `fesom_eos.cpp`/`fesom_kpp.cpp` → rebuild the oracle, L51). The smoother feeds N²→PP/KPP→mixing and blmc→KPP, so a bug shows as a mixing/T divergence.
5. **1-yr CORE2 CUDA climate** to close (`jobs/job_m32_cuda_core2`, compare via `scripts/m32_climate_compare.py` vs `m32_cuda_m516_1yr` + Fortran/C; apples-to-apples L58). Bit-identical-on-Serial → expect statistically identical (D22 floor).

---

## 4. AFTER THE SMOOTHER — the rest of the compute+PCIe bulk (the M5.18 arc)

The smoother is the spearhead + the **proof-of-concept for the coalescing lever**. Once it lands, GENERALIZE:
- **Re-profile** (`jobs/job_ng5_prof` + nsys `jobs/job_nsys_ng5`) to re-rank the kernels. The other big ones (nsys, rank0): `fesom_tracer_advect…` (FCT, ~3.1%×2), `fesom_compute_vel_rhs` (3.1%), `fesom_compute_sigma…` (3.0% — GM neutral-slope), `diff_ver_part_impl` (2.9%). **Check each for the same node-major uncoalesced pattern** (one-thread-per-node + internal level loop) and apply the same per-(node,level) re-parallelization where the arithmetic stays identical.
- **Residual PCIe `deep_copy` = 3.41 GB/step (34% of CUDA API time = `cudaMemcpy`).** Residency was declared exhausted (M5.15/16) but 3.4 GB/step still moves. Attribute it per-field (the `FESOM_SYNC_LOG` rail in `fesom_field.hpp` + the `FESOM_STEP_PROFILE` deep_copy counter) — it's now the forcing HtoD (8 JRA55 fields) + any verify-gated syncs + the Phase-A bulk output round-trips. Phase B (fully device-resident forcing, deferred from M5.16) reclaims part of it.
- **Heavyweight Lever C** (`fesom_field.hpp` rank-1 → `View<double**>` global layout) stays the LAST resort — only if the local per-kernel coalescing plateaus. It touches all 126 fields + every kernel; high risk. The smoother result tells you whether the local approach is enough.
- **Load imbalance (~9%, Lever D)** is orthogonal + deployment-side: a per-rank work dump (node/elem/level counts vs the M5.17 per-rank Barrier times) tells static-partition (→ re-partition / work-weighted metis, a clean ~9% with zero port-code risk) from dynamic-physics/GPU-jitter. Worth one cheap diagnostic; not a code change.

---

## 5. HARD CONSTRAINTS (carry every session)
- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB quota). Big/NG5/CORE2 runs via **SLURM gpu nodes, never login**.
- ⚠️ **NG5 perf jobs write ~50 GB `*.monthly.nc` even with `snap_every=-1`** — `rm <dir>/*.monthly.nc` after each (the M5.16 gotcha; `job_ng5_halo_split`/`job_ng5_prof` patterns do this).
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; the env.sh `openmpi/4.1.2` SEGFAULTs on device ptrs, L47). ⚠️ `env_cuda.sh` PURGES `git` — do git ops in a separate shell (`source /sw/etc/profile.levante` only). CPU builds use `env.sh`. Build dirs `build-cuda`/`build-serial`/`build-synccheck` carry M5.17; `build-omp` STALE.
- **Same-day same-node perf baselines only** ([[feedback-perf-same-day-baseline]]): rebuild the prior commit + run the SAME job on the SAME nodes (the M5.16 procedure — save the diff as a patch, `git checkout --` the files, build, `cp` the binary to a stable path, restore the patch, rebuild; a running SLURM job is unaffected by clobbering the on-disk binary, new inode).
- **Device/kernel changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); pi is insufficient (no ice / maybe no bvfreq-smooth). **Commit/push only when the user asks.** KPP is the default mix_scheme ([[feedback-kpp-default]]).

## 6. POINTERS
- **Memory:** [[project-m517-mpi-comms]] (Lever B dead-end + the gate method), [[project-m516-bulk-port]], [[feedback-perf-same-day-baseline]], [[feedback-gpu-fidelity-gate]], [[reference-cuda-aware-mpi]], [[reference-build-run]].
- **Docs:** `docs/GPU_FIDELITY.md` §M5.17 (the gate) + §M5.13–16 (the residency arc), `docs/SCALING_NG5.md` (the per-phase + nsys decomposition), `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–**L62**).
- **Tooling:** `jobs/job_ncu_smooth_ng5` (the smoother ncu, **created this session**; `NCU_REGEX` now parametrizes `ncu_rank0.sh`), `jobs/job_ng5_prof` (per-phase + deep_copy), `jobs/job_nsys_ng5` (kernel + PCIe + MPI), `jobs/job_ng5_halo_split` (the M5.17 barrier gate). The smoother code: `src/fesom_eos.cpp:425` (host) / `:488` (device).
- **Tags:** `m5.16-bulk-port` (master HEAD, residency exhausted + bulk port), `m5.9-pin` (whole-model GPU + climate). M5.17 = uncommitted on `m517-mpi-comms`.

## 7. Bottom line
Lever B is measured-dead (2.4% ceiling). The mass is in **GPU compute + residual PCIe (~89%)**, and the single fattest kernel is the smoother (25.7%), slow because it's an **uncoalesced node-major gather with depth divergence**. The fix is a **local, bit-identical re-parallelization to one-thread-per-(node,level)** (level = the contiguous dim → coalesced, divergence-free) — no global layout refactor. MEASURE the ncu before/after + same-day s/step, hold the Serial-bit-identical gate, then generalize the coalescing lever to the other node-major kernels and attack the 3.4 GB/step residual PCIe. Don't claim a win until it's same-day measured + Serial/np2 bit-identical + gate PASS + climate-validated.
