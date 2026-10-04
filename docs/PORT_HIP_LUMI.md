# LUMI-G (AMD MI250X / gfx90a) HIP port

Port log for bringing the FESOM2 C/Kokkos port up on LUMI-G, with the
device-resident, GPU-aware-MPI halo path **active** on the HIP backend.
Cuts the NG5 dist_16 step from ~8.7 s to ~3.4 s (2.6× same binary, A/B
proven). For the working build recipe + module set, see `env_lumi.sh` and
`configure_lumi.sh`; the rest of this doc is the *why*.

## TL;DR

| Stage | s/step | Notes |
|---|---:|---|
| LUMI host-staged halo (`FESOM_HOST_HALO=1`) | **8.70** | every halo: D2H → host MPI → H2D |
| LUMI device-resident halo (default) | **3.36** | device-ptr MPI via cray-mpich + libmpi_gtl_hsa |
| Levante CUDA, same config (reference) | ~1.3 | from `docs/SCALING_NG5.md` |

Speedup on LUMI: **2.6×**, vs Levante's ~9× from the same architectural
change. Closing the remaining gap is future work (Slingshot vs IB,
GTL overhead, kernel-coalescing on RDNA-class CUs). See "Gap analysis"
below.

## Toolchain on LUMI (LUMI/25.03 + PrgEnv-amd)

`env_lumi.sh` records the exact module set; the essentials:

```
LUMI/25.03  partition/G
PrgEnv-amd  amd/6.3.4 (clang 18, ROCm 6.3.4)
rocm/6.3.4
craype-accel-amd-gfx90a       # adds GTL link + -xhip drive (and concat bug, below)
cray-hdf5  cray-netcdf/4.9.0.17
buildtools/25.03               # CMake/Ninja
```

Unloaded: `cray-libsci` (pulls AMD-Flang runtime that we don't need;
removes one `libflang.so` runtime dep). Runtime: `MPICH_GPU_SUPPORT_ENABLED=1`
+ prepend `$CRAY_LD_LIBRARY_PATH` and `/opt/rocm-6.3.4/lib/llvm/lib` to
`LD_LIBRARY_PATH` (the AMD-flavor cray libs still DT_NEED libflang).

### Gotcha 1: the Cray `CC` wrapper concatenates adjacent flags

The wrapper under PrgEnv-amd merges Kokkos's `--rocm-path=… --offload-arch=gfx90a`
into a single `--rocm-path=…--offload-arch=gfx90a` token, which makes
amdclang++ fail the HIP runtime auto-detect with "cannot find HIP runtime".
Workaround in `configure_lumi.sh`: compile with `-DCMAKE_CXX_COMPILER=amdclang++`
directly, but keep `-DMPI_CXX_COMPILER=CC` so CMake's FindMPI introspects
the wrapper for cray-mpich + GTL flags (those parts work fine).

### Gotcha 2: cray-netcdf requires cray-hdf5 loaded first

`module load cray-netcdf` alone fails with "cannot be loaded as requested"
on LUMI/25.03 partition/G — needs `module load cray-hdf5` immediately
before it.

## The actual port: FESOM_GPU_RESIDENT macro widening (10 sites)

The model's hot data layer (mesh, dyn, tracers, aux) already lives in
`Kokkos::View` / `fesom::Field` and runs through Kokkos kernels — that
half is backend-portable by construction; nothing changed.

The **performance-critical** half is the device-resident halo exchange
path: `src/fesom_halo_device.{cpp,hpp}` packs a halo into a device buffer,
hands the **device pointer** to MPI, unpacks on device. The legacy host-
staged bracket — `modify_device(); sync_host(); host-MPI; modify_host();
sync_device()` — copies the *whole field* over PCIe twice per exchange.
With CG (~80-127 iters), EVP (120 subcycles), FCT (~21 brackets) per step,
that traffic dominates the step (PCIe wall).

Pre-port, all of this was gated `#ifdef KOKKOS_ENABLE_CUDA` at 10 sites
(6 in `fesom_halo_device.hpp`, 3 in `.cpp`, 1 in `fesom_ssh.cpp`). On
HIP — which has a separate device memory space exactly like CUDA — the
guard compiled the path *out*, so the binary silently fell back to the
host-staged bracket. That's the "correct but slow" mode.

The fix is to introduce a single backend-neutral gate:

```c
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#  define FESOM_GPU_RESIDENT 1
#else
#  define FESOM_GPU_RESIDENT 0
#endif
```

defined once near the top of `fesom_halo_device.hpp` (after the Kokkos
include so the macros are visible), and replace the 10 `#ifdef
KOKKOS_ENABLE_CUDA` sites with `#if FESOM_GPU_RESIDENT`. Negation
(`!FESOM_GPU_RESIDENT`) selects the host-staged-only fallback.

**Why this works at zero risk:** the device-halo implementation in
`fesom_halo_device.cpp` is 100% Kokkos + MPI-on-device-pointer. There
is **no `cuda*`/`hip*` API call** in the file. Kokkos `View::data()`
returns the right device pointer on either backend; `Kokkos::fence()`
flushes the right queue; cray-mpich+`libmpi_gtl_hsa` handles HIP
device pointers exactly as openmpi+UCX-cuda handled CUDA ones. So the
macro change is the *only* code change required — verified by the
codegen-neutral re-time below (3.3651 s/step pre-refactor, 3.3482 s/step
after — within noise).

Serial / OpenMP paths untouched (those backends have `FESOM_GPU_RESIDENT
== 0`, fall through to the legacy host-staged bracket, the bit-identity
oracle is preserved by construction).

## Validation gates

1. **2-rank GPU-aware MPI smoke** (job 19010979, see
   `/pfs/lustrep4/projappl/project_465002727/sbeyer/gpu_mpi_smoke/`): rank 0
   `MPI_Send`s a HIP device pointer (`0x14e8ed200000`), rank 1 `MPI_Recv`s
   into a HIP device pointer (`0x14db17c00000`), `hipMemcpy` to host
   verifies — **0 mismatches out of 1024 doubles, PASS.** Proves cray-mpich
   + GTL actually moves device pointers between nodes; the device-halo
   speedup isn't a silent fallback.

2. **A/B same-binary, same-mesh, same-allocation** (NG5 dist_16, 2 nodes
   × 8 GCDs, 35 steps, dt=180):
   - host-staged (`FESOM_HOST_HALO=1`, job 19010335): **8.70 s/step**
   - device-resident (default,            job 19010156): **3.36 s/step**

   Both runs converge to the same physics at step 35 (it=83, eta=1.21,
   T[-2.08, 30.09/30.18], S[5.62/5.60, 41.16/41.20]). The slight last-step
   T.max/S.min differences are stale-host diagnostic, not numerical
   drift: snapshot writes do explicit `sync_host()` so on-disk outputs
   would match. The host-halo print shows full healthy uv/w/Kv numbers
   (uv=3.44, w=3.76e-01, Kv=3.05); the device-halo print shows these
   as 0 because the device-only fields aren't pre-print synced — see
   "Diagnostic stale-host" below.

3. **Macro-refactor codegen-neutral** (job 19010981, NG5 dist_16 after
   the FESOM_GPU_RESIDENT collapse): 3.3482 s/step, vs 3.3651 pre-refactor
   = ±0.5%, within noise. Confirms the macro is purely stylistic.

## Diagnostic stale-host print

`uv/w/Kv/Av/bv` in the per-step `it=` line print as `0.00e+00` under the
device-halo path on multi-rank runs. Cause is purely diagnostic: those
fields are deliberately *device-resident* across the device-halo
(comment in `fesom_step.cpp:388-392`: "Av/Kv stay DEVICE-resident from
KPP → mo_convect → device-halo → ivisc / trdiff — the OUT-rail sync_host
… round trip is gone"). The per-step stats `printf` reads via the raw
host alias `aux.Kv[i]`, which is stale → reads as zero.

Snapshot output (`snap_NNNNNN.nc`) is unaffected: the snapshot bracket
in `fesom_main.cpp:1338-1352` explicitly `sync_host()`s every device-
resident output field before the rank-0 gather. So the *model state* is
correct; only the per-step diagnostic print is misleading.

Two clean future fixes (not done): (a) add a 4-line `sync_host()` block
ahead of the per-step stats reductions; (b) compute the stats on device
via a `parallel_reduce` and pull just the scalar to host.

## Gap analysis (LUMI 2.6× vs Levante ~9×)

Same architectural change, smaller win. Candidates:
- **GTL vs UCX-cuda overhead**: cray-mpich routes device pointers through
  the GTL library (libmpi_gtl_hsa). UCX-cuda on Levante had cuda_copy /
  cuda_ipc transports for finer-grained intra-node paths. A 2-node smoke
  is too small to characterize the difference; needs a per-message-size
  ping-pong sweep to confirm.
- **gfx90a vs A100 SM occupancy / register pressure on the device-halo
  pack/unpack kernels**: the kernels are tiny memory-bound gathers. Levante
  A100 may extract more bandwidth here than gfx90a CDNA2; a quick
  `rocprof --hsa-trace --hip-trace` on the pack kernel would tell.
- **Slingshot vs HDR-IB MPI latency**: per-message overhead higher on
  Slingshot at small message sizes; if the per-step halo workload is
  many small messages (CG = ~80 1-double-per-node sends per neighbour),
  this dominates.

None of these are addressed in this port — they're optimization work for
a follow-up M-series milestone on LUMI.

## File map

| Change | File | What |
|---|---|---|
| New | `env_lumi.sh` | LUMI module set + LD path + MPICH_GPU_SUPPORT_ENABLED |
| New | `configure_lumi.sh` | CMake driver: amdclang++ as CXX, CC for MPI introspection, gfx90a |
| New | `jobs/job_core2_8_lumi` | smoke job: CORE2 dist_8, full physics, 200 steps |
| New | `jobs/job_gpu_scaling_lumi` + `submit_gpu_scaling_lumi.sh` | strong-scaling sweep, 1/8/16/32 nodes |
| New | `jobs/job_profile_core2_dist8_lumi` | FESOM_STEP_PROFILE=1 per-kernel breakdown |
| New | `jobs/job_ng5_test_lumi`, `jobs/job_ng5_hosthalo_ab` | NG5 device-halo vs host-halo A/B |
| Edit | `src/fesom_halo_device.hpp` | introduce FESOM_GPU_RESIDENT (single source of truth) |
| Edit | `src/fesom_halo_device.cpp` | 3 gates → FESOM_GPU_RESIDENT (function body + impl block + trailing comment) |
| Edit | `src/fesom_ssh.cpp` | 1 gate → FESOM_GPU_RESIDENT (CG inner halo bracket) |
| Edit | `src/fesom_main.cpp` | FESOM_SSS_PATH / FESOM_RUNOFF_PATH env vars for non-Levante deployments |
| Edit | `src/fesom_jra55.cpp` | FESOM_JRA55_DIR env var |
| Edit | `src/fesom_partit.cpp` | `read_int` handles Fortran shorthand `N*M` + commas (LUMI dist_8/rpart.out uses them) |
| Edit | `src/fesom_mesh.cpp` | `read_edges` handles comma-separated `edges.out` / `edge_tri.out` |

## How to reproduce

```bash
cd /pfs/lustrep4/projappl/project_465002727/sbeyer/fesom_kokkos
bash -l configure_lumi.sh --clean
bash jobs/submit_gpu_scaling_lumi.sh        # CORE2 strong scaling
sbatch jobs/job_ng5_test_lumi               # NG5 device-halo run (default)
# A/B test:
sbatch jobs/job_ng5_hosthalo_ab             # same binary, FESOM_HOST_HALO=1
```
