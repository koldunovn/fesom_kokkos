# Running FESOM2-Kokkos on multiple GPUs (M3.1)

**Status (2026-05-26, M3.1):** the CUDA build runs across multiple A100s, **one MPI rank ⇒ one
GPU**, with each rank bound to a *distinct* device by its node-local rank. Verified on Levante:
pi `dist_2` (2 ranks / 2 GPUs, 1 `gpu-devel` node) and CORE2 `dist_8` (8 ranks / 8 GPUs, 2 `gpu`
nodes). This is **correctness + binding**, not a performance benchmark — see the perf caveat at the
end. The 2-yr/5-yr climate validation is M3.2.

> CUDA is **climate-close**, NOT bit-identical (the D5 validation ladder, see
> `docs/KOKKOS_PORTING_LESSONS.md`). `scripts/diff_snap.py` printing "DIVERGENCE" is the expected
> PASS for a GPU run, not a failure. The Serial backend remains the bit-identity oracle.

---

## 1. TL;DR — launch recipe

```bash
# Build once on a login node (full module dance — §5):
cmake --build build-cuda --target fesom_port -j 16

# Multi-GPU smokes (job scripts carry the full env + the binding contract):
sbatch jobs/job_gpu_multi_pi     # pi dist_2  : 2 ranks / 2 GPUs, 1 gpu-devel node
sbatch jobs/job_gpu_core2        # CORE2 dist_8: 8 ranks / 8 GPUs, 2 gpu nodes

# Confirm each rank bound a DISTINCT GPU (the M3.1 gate):
grep "Kokkos device_id" runs/pi_gpu2.<jobid>.out      # device_id 0 and 1 (NOT both 0)
```

The number of ranks **must** equal the partition count: run `dist_N` ⇒ launch `N` ranks ⇒ use `N`
GPUs. There is no oversubscription path in M3.1 (one rank per GPU is the contract).

---

## 2. The rank → GPU mapping (the core of M3.1)

A CUDA process must call `cudaSetDevice` (here: `Kokkos::InitializationSettings::set_device_id`)
to pick its GPU. The id must be the **node-local rank**, *not* the global MPI rank: every node
exposes GPUs `0..(g-1)`, so global rank 5 (the 2nd rank on node 1 with 4 GPUs/node) must drive
device **1**, not device 5 (which doesn't exist on its node).

`src/fesom_main.cpp` (right after `MPI_Init`, before `Kokkos::initialize`):

```c
int local_rank = 0;
MPI_Comm shmcomm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
MPI_Comm_rank(shmcomm, &local_rank);          /* fallback: getenv("SLURM_LOCALID") */
MPI_Comm_free(&shmcomm);

Kokkos::InitializationSettings s;
s.set_device_id(local_rank);                  /* ignored by Serial/OpenMP backends */
Kokkos::initialize(s);
```

- **`MPI_COMM_TYPE_SHARED`** splits `MPI_COMM_WORLD` into sub-communicators of ranks that can share
  memory — i.e. the ranks on one node. The rank *within* that sub-communicator is the node-local
  rank. This is the authoritative source; it works whether or not the launcher exports
  `*_LOCAL_RANK` env vars.
- The old `Kokkos::initialize(argc, argv)` form set **no** device id, so every rank defaulted to
  GPU 0 — the multi-GPU bug this fixes.
- **No-op on CPU backends.** Kokkos only consults `device_id` from a GPU backend's initialize
  (`Kokkos::Impl::get_gpu()`); Serial/OpenMP never call it, so the id sits unused and the CPU
  bit-identity oracle is unaffected. For `npes==1` the node-local rank is 0 ⇒ device 0, identical
  to the old default.
- **Per-rank proof.** Each rank prints `Kokkos::device_id()` (the *bound* device; -1 on CPU):
  ```
  [fesom_port] rank 0 on vader2: node-local rank 0 -> Kokkos device_id 0
  [fesom_port] rank 1 on vader2: node-local rank 1 -> Kokkos device_id 1
  ```

### Why correctness follows from the mapping *alone* (no kernel changes)

Every halo exchange (`fesom_exchange_*`) runs on **host pointers** (`Field::h_checked()`), and the
M1.5 sync rails already `sync_host` before each halo and `sync_device` after. So a multi-GPU halo
stages **device → host → MPI → host → device** with **regular, non-GPU-aware MPI** — the same MPI
that moves host buffers on CPU. No kernel or exchange code changes for multi-GPU; the device-id
binding is the whole change. (GPU-aware MPI — device-pointer halos, the throughput optimisation —
is explicitly deferred to M5/perf.)

---

## 3. SLURM GPU binding — the visibility contract

`set_device_id(local_rank)` requires that **every GPU on a node is visible to every rank on it**
(then Kokkos' `visible_devices[local_rank]` is the local_rank-th physical GPU → distinct per rank).
The job scripts request this with:

```
#SBATCH --gres=gpu:<g>           # g GPUs for the node (2 on gpu-devel, 4 on gpu)
#SBATCH --ntasks-per-node=<g>    # g ranks on the node — one per GPU
#SBATCH --gpu-bind=none          # all g GPUs visible to all g ranks (do NOT per-task-bind)
```

- **Do NOT** use `--gpus-per-task=1` or `--gpu-bind=single:1`: those set a *per-task*
  `CUDA_VISIBLE_DEVICES` (each rank sees only "its" GPU as device 0), which contradicts
  `set_device_id(local_rank)` and makes rank 1 request a device that isn't visible.
- **Safety net:** if a rank requests a device id ≥ the number of visible GPUs, Kokkos **aborts**
  with `Requested GPU with id 'N' but only M GPU(s) available!` — a loud, immediate signal that the
  binding policy is wrong (rather than a silent double-bind). If you see this, you have per-task GPU
  binding; add `--gpu-bind=none`.
- **Alternative (not used here):** Kokkos has a built-in `settings.set_map_device_id_by("mpi_rank")`
  that reads the local rank from env vars and assigns `visible_devices[local_rank % num_visible]`
  (the modulo tolerates per-task binding *and* oversubscription). We use the explicit
  `MPI_COMM_TYPE_SHARED` + `set_device_id` instead because it is launcher-independent (no reliance
  on `*_LOCAL_RANK` env vars) and the printed mapping is under our control. The modulo form is the
  route to take if M5 ever needs >1 rank per GPU.

---

## 4. Partition choice

Run `dist_N` ⇒ launch `N` ranks ⇒ bind `N` GPUs. Available CORE2 partitions:
`dist_{8,16,32,64,72,144,256,288,432,512,864}` (under
`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/`); pi has `dist_{2,8}`.

Levante GPU partitions (`sinfo -p gpu,gpu-devel`):

| partition   | A100/node | nodes | time   | use                                            |
|-------------|-----------|-------|--------|------------------------------------------------|
| `gpu-devel` | **2**     | 3     | 30 min | quick smokes: pi `dist_2` (1 node), `dist_8` (4 nodes — exceeds 3, use `gpu`) |
| `gpu`       | **4**     | 4×40GB + 59×80GB | 12 h | real runs: CORE2 `dist_8` = 2 nodes, `dist_16` = 4 nodes, … |

So `#GPUs = N`, `#nodes = ceil(N / GPUs-per-node)`, `--ntasks-per-node = GPUs-per-node`.
CORE2 `dist_8` on `gpu` = `--nodes=2 --ntasks-per-node=4 --gres=gpu:4`.

---

## 5. Build (full module dance — the sticky-module fix, L17)

```bash
source /sw/etc/profile.levante
module --force purge
module unload netcdf-c cdo ncview git 2>/dev/null   # --force purge leaves these sticky; the build's
                                                    # netcdf-c/4.8.1 then conflicts and nvhpc never loads
module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0 \
            openmpi/4.1.2-gcc-11.2.0 netcdf-c/4.8.1-gcc-11.2.0
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
which nvcc                                           # must resolve, else the load failed (Error 127)
cmake --build build-cuda --target fesom_port -j 16   # verify "Built target fesom_port" (L17)
```

**MPI transport for multi-rank GPU runs:** use the standard Levante UCX/IB transport (the `env.sh`
knobs: `OMPI_MCA_pml=ucx`, `UCX_NET_DEVICES=mlx5_0:1`, …) — see the job scripts. Do **NOT** use the
`OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` override from `jobs/job_pi_smoke_gpu`; that is a
single-rank / login-node workaround and has no IB path for inter-node halos.

---

## 6. M3.1 validation — smoke results

The M3.1 gate (CUDA, so **no** bit-identity): the run **completes** (exit 0), is **physical** (T/S
sane, no blow-up), **each rank binds a distinct GPU**, and the result is **climate-close**.

### pi `dist_2` — 2 ranks / 2 GPUs, 1 `gpu-devel` node (job 25140844) — PASS

- Exit 0; node `vader2`, 2 GPUs (distinct UUIDs). Binding: `rank 0 → device_id 0`, `rank 1 →
  device_id 1` (distinct — the bug is fixed). T/S bounded `T[10.00,14.99] S[35.00,35.00]` through
  step 20 (no blow-up).
- Climate-close **at the unchanged CUDA budget**. The right comparison is vs the **same-rank-count**
  Serial oracle (isolates the device-fma budget from the partition effect — see §7):

  | comparison (step 20)                          | density   | T        | u        | Av/Kv  | verdict |
  |-----------------------------------------------|-----------|----------|----------|--------|---------|
  | np=2 CUDA vs **np=2 Serial** (`…_m13_nocma`)  | 2.75e-11  | 1.45e-10 | 2.32e-04 | 0.095  | CUDA budget ✓ |
  | np=1 CUDA vs np=1 C golden (single-GPU 25140843) | 3.18e-12 | 1.86e-11 | 1.84e-04 | 0.095  | CUDA budget ✓ |
  | np=2 CUDA vs np=1 C golden (mixed)            | 3.93e-05  | 1.72e-04 | 5.13e-03 | 0.130  | = the np effect, §7 |

  The np=2-CUDA-vs-np=2-Serial budget (density ~1e-11, u/v ~1e-4, pgf ~1e-17, Av/Kv 0.095
  threshold-flips) is the **same** as the single-GPU budget — no new divergence class.

### CORE2 `dist_8` — 8 ranks / 8 GPUs, 2 `gpu` nodes (job 25140967) — PASS

- Real-mesh + **multi-node** smoke (cross-node IB halos), dt=1800, 30 steps, JRA55-1958 + PHC winter IC.
- Exit 0. **8 distinct bindings, 4 per node** — the `MPI_COMM_TYPE_SHARED` split resets the node-local
  rank on the second node:
  ```
  rank 0..3 on node l50109 -> node-local 0,1,2,3 -> device_id 0,1,2,3
  rank 4..7 on node l50121 -> node-local 0,1,2,3 -> device_id 0,1,2,3
  ```
- Physical: T[-2.06, 30.17] S[5.67, 41.11] stable through step 30 (healthy spin-up from PHC under real
  JRA55 forcing, no blow-up / NaN). CG ~113–127 iters/step.

---

## 7. ⚠️ Validating a multi-rank GPU run: use the same-rank-count oracle

A multi-rank run differs from a single-rank run **even on the bit-identity Serial backend** — the
domain decomposition changes the order of edge→node scatters and halo exchanges at partition
boundaries, so np=1 and np=2 are not bit-identical (this is why a separate np=2 Serial oracle,
`/scratch/a/a270088/pi_np2_ref_m13_nocma`, exists). On the pi mesh at step 20 the pure np=1↔np=2
**Serial** difference is already density 3.9e-05 / T 1.7e-04 / u 5.1e-03.

So diffing a **np=2 CUDA** run against the **np=1** C golden folds the (large) partition effect
**and** the (tiny) CUDA device-fma budget together — and the partition effect dominates by ~6 orders
of magnitude. To judge whether the GPU result is climate-close **at the CUDA budget**, diff against
the **same-rank-count Serial oracle** (np=2 CUDA vs np=2 Serial). That isolates the device-fma
budget (here density 2.75e-11, ≪ the 3.9e-05 partition effect) and is the comparison that proves
"no new divergence class". Both the partition effect and the CUDA budget are bounded and
climate-close; the magnitude clarity comes from choosing the right oracle.

For M3.2 (climate validation), this means the GPU run is compared against a same-rank-count CPU
reference at the same partition count, and the GPU↔CPU budget is read off *that* pair.

---

## 8. ⚠️ Performance — measured, and why the GPU is NOT yet faster

**M3.1 is correctness + binding, not a tuned benchmark — and at this stage the CPU is faster
node-for-node.** Measured per-step wall time (CORE2, dt=1800, loop = wall(105 steps) − wall(5 steps),
no I/O; jobs `job_{gpu,cpu,cpu256}_time_core2`):

| config (2 nodes each)                         | decomposition | host threads/rank | s/step | vs GPU |
|-----------------------------------------------|---------------|-------------------|--------|--------|
| **GPU — 8× A100** (2 `gpu` nodes)             | `dist_8`      | 1 (Serial host)   | **0.86** | 1.0× (ref) |
| CPU — 8 cores (same node footprint as 1 rank/GPU) | `dist_8`  | 1 (Serial)        | 1.96   | 2.3× *slower* |
| **CPU — 256 cores** (2 `compute` nodes)       | `dist_256`    | 1 (Serial)        | **0.051** | **17× faster** |

The middle row (8 A100 vs 8 cores) is a **per-rank** comparison and is misleading as a *hardware*
comparison — 8 cores is 1/16 of one CPU node. The fair **node-for-node** comparison (bottom row) fills
the same 2 nodes the way each architecture naturally runs: the GPU gets **8 ranks** (1/A100), the CPU
gets **256 ranks** (1/core). There the **CPU is ~17× faster**. All three runs are physical and produce
identical physics (matching CG `it=` counts and T/S to printed precision).

**Why the GPU loses node-for-node today** (not a port flaw — the next two milestones fix it):
1. **The CG solver and sea ice are still on the host.** With `dist_8` they run on only **8 ranks**
   (each owning 1/8 of the domain) vs the CPU's **256 ranks** (1/256 each) — ~32× less parallel — and
   they dominate the timestep. The CG does ~113–127 iterations/step (host), and `uv_rhs`/solution
   round-trip device↔host every step (`docs/SYNC_MAP.md` §5). Roughly ~0.7 of the GPU's 0.86 s/step is
   host CG + ice + transfers; the GPU ocean kernels are a small slice.
2. **CORE2 is too small to fill an A100** — ~500k wet pts/GPU at `dist_8` (bandwidth/occupancy-bound).
   The M2 compute-bound estimate (~1 A100 ≈ 256 cores) is the *ceiling*, not the current host-bound
   reality.

**The path to a GPU win:** **M4.2** (CG → device, removes the per-step round-trip *and* the serial host
CG) + **M4.3** (sea ice → device) eliminate the host bottleneck so the 8-rank GPU layout is no longer
penalized for poor host parallelism; then **higher-resolution meshes** fill the A100s. Only after M4.2
is a GPU↔CPU wall-time comparison meaningful. Do **not** quote a GPU speedup from M3.1.

---

## 9. Next

- **M3.2** — 2-yr / 5-yr CUDA CORE2 climate validation vs `/scratch/a/a270088/fortran_pp_2yr`
  (+ KPP ref): SST/SSS RMS within the Fortran↔C budget; document the GPU↔CPU budget in
  `docs/GPU_FIDELITY.md`; tag `m3-gpu-climate`. (Compare against a same-rank-count CPU reference, §7.)
- **M4.2** — SSH RHS + CG solver on device (closes the §5 round-trip; the prerequisite for a fair
  GPU benchmark).
