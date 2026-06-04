# FESOM2-Kokkos on JUPITER (JSC GH200) — port + scalability study (2026-06-04)

First build, run, and strong-scaling study of the FESOM2-Kokkos port on the **JUPITER
Booster** at Jülich (NVIDIA GH200 Grace-Hopper). Project account `hclimrep`.

## TL;DR

- **It builds and runs**, CPU (Kokkos Serial on ARM Grace) and GPU (Kokkos CUDA on
  Hopper), Serial validated end-to-end and CUDA climate-close to Serial (T/S/eta/stress
  match at printed precision).
- **🏆 The GPU now WINS node-for-node** (reversing Levante A100): GH200 is **1.31×** faster
  than the 288 Grace cores on core2 and **2.53×** faster on ng5 (per node). NVLink-C2C
  (900 GB/s) erased the PCIe host↔device wall that made the A100 *slower*; the margin grows
  with mesh size.
- **ng5 (7.4 M nodes) GPU strong-scaling**: clean 85–90 %/doubling to **16 nodes** (64 GPU,
  0.98 → 0.19 s/step), best absolute at **64 nodes / 256 GPU (0.106 s/step, 9.2×)**, then it
  **reverses** — 128 nodes is *slower* (pure-comms regime). Ceiling ≈ 64 nodes for ng5.
- **core2 (127 k nodes) does NOT scale on GPU** — flat at ~0.057 s/step (it is far too
  small to fill even 8 H100s; pure launch/halo floor). Use ng5-class meshes for GPU.
- Cross-node **CUDA-aware MPI works** (ParaStationMPI + `MPI-settings/CUDA`): on-device
  halo exchange across InfiniBand NDR200, no host staging needed.

## The machine (one JUPITER Booster node)

| | |
|---|---|
| Superchips/node | 4× NVIDIA GH200 |
| GPU | Hopper H100, 132 SM, **96 GB HBM3 @ 4 TB/s** (×4) |
| CPU | 72-core ARM **Grace** (Neoverse-V2 @ 3.1 GHz) per superchip → **288 cores/node** |
| Grace↔Hopper | **NVLink-C2C @ 900 GB/s** (vs Levante PCIe ~64 GB/s — ~14×) |
| Grace mem | 120 GB LPDDR5X @ 512 GB/s (×4 → 480 GB/node) |
| Network | 4× InfiniBand NDR200 (ConnectX-7), Dragonfly+ |
| Partition | `booster` (only one; 12 h max, 1 h default), `--account=hclimrep` |

The NVLink-C2C is the architecturally important number: on Levante the GPU port was
host↔device-transfer-bound over PCIe; GH200's 900 GB/s C2C largely removes that wall.

## Build

Toolchain (JSC Stage 2026, **CUDA-13-native**):

```
GCC 14.3.0 + ParaStationMPI 5.13.0-1 + CUDA 13.0 + netCDF 4.9.3 + CMake 3.31.8
MPI-settings/CUDA            # CUDA-aware MPI runtime (device-ptr halo)
```

Env files (this repo): `env_jupiter.sh` (CPU), `env_jupiter_cuda.sh` (CUDA),
`env_jupiter_data.sh` (data paths). They source the Lmod init explicitly so they work in
non-login SLURM batch shells.

**Kokkos bumped 4.4.01 → 4.7.03** (submodule). Stage 2026 only offers CUDA 13, which
predates Kokkos 4.4.01's support; 4.7.03 is CUDA-13-capable and the JSC
`Kokkos/4.7.03-CUDA-13` system module proves the exact combo builds & runs on GH200. The
FESOM source compiled unchanged against 4.7.03 (Serial ctest green).

```bash
# Serial (CPU oracle):
source env_jupiter.sh
cmake -S . -B build-serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-serial -j 16

# CUDA (GH200): Hopper90 device + ARMV9_GRACE host, nvcc_wrapper(host=g++)
source env_jupiter_cuda.sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON -DKokkos_ARCH_ARMV9_GRACE=ON \
  -DCMAKE_CXX_COMPILER=$PWD/externals/kokkos/bin/nvcc_wrapper
cmake --build build-cuda -j 16    # ~1 min wall on the Grace login node
```

## Jupiter-specific changes that were needed

1. **Filesystem topology (the big gotcha).** `/p/scratch/hclimrep` is **login-only — NOT
   mounted on booster compute nodes**. Data staged there (the meshes) is invisible to
   jobs; a job whose SLURM `-o` points there dies instantly (RaisedSignal:53, no output).
   The compute-visible scratch is **`/e/scratch/hclimrep`** (`$SCRATCH_hclimrep`, the `/e`
   Exascale tier). All meshes + forcing were staged `/p/scratch/.../meshes` →
   `/e/scratch/hclimrep/koldunov1/meshes` (~48 GB). HOME `/e/home` is also compute-visible.

2. **GPU binding.** JUPITER's native launch is `--gpus-per-task=1` → per-task GPU
   isolation (each rank sees only its GPU, renumbered to logical 0). The code's
   `set_device_id(local_rank)` (Levante: all-GPUs-visible) would request a non-existent
   device. Fixed in `src/fesom_main.cpp` to `device_id = local_rank % num_visible`
   (num_visible parsed from `CUDA_VISIBLE_DEVICES`) — correct under BOTH regimes, no-op on
   Serial/OpenMP. Verified: 4 distinct GH200 per node, 8 across 2 nodes.

3. **Portable forcing paths.** Hardcoded Levante `/pool/...` forcing paths replaced by env
   overrides: `FESOM_JRA55_DIR` (`src/fesom_jra55.cpp`), `FESOM_SSS_PATH` /
   `FESOM_RUNOFF_PATH` (`src/fesom_main.cpp`); `FESOM_CHL_FILE` already existed. On JUPITER
   the runoff file is `runoff.nc` (not `CORE2_runoff.nc`). All four set in
   `env_jupiter_data.sh`.

## Methodology

- 35 steps, internal loop timer (excludes 5 warmup → **30 timed**), `snap_every` huge
  (no I/O — rank-0 gather OOMs on ng5). JRA55-do 1958 forcing + PHC winter IC (realistic
  ocean+ice). dt = 1800 s (core2), 180 s (ng5, high-res stability). 1 MPI rank = 1 H100.
- Jobs: `jobs/job_jupiter_gpu` (CUDA), `jobs/job_jupiter_cpu` (Serial). Answer line:
  `grep "loop timing" <out>/log`.

## GPU strong scaling — ng5 (7,402,886 nodes, 70 levels), dt=180

s/step is the mean of 2 reps where taken (n16/n32/n64; rep spread < 3 %, so the structure
below is reproducible, not noise):

| nodes | GPUs (H100) | nod2D/GPU | s/step | speedup vs 2N | per-doubling | par. eff. |
|:-----:|:-----------:|:---------:|:------:|:-------------:|:------------:|:---------:|
|   2   |      8      |   926 k   | 0.9806 | 1.00×         |   —          |   —       |
|   4   |     16      |   463 k   | 0.5416 | 1.81×         |  90 %        |  90 %     |
|   8   |     32      |   231 k   | 0.3181 | 3.08×         |  85 %        |  77 %     |
|  16   |     64      |   116 k   | 0.1883 | 5.21×         |  85 %        |  65 %     |
|  32   |    128      |    58 k   | 0.1653 | 5.93×         |  57 % ⟵ dip  |  37 %     |
|  64   |    256      |    29 k   | 0.1061 | 9.24×         |  78 % ⟵ rebound | 14 %   |
| 128   |    512      |    14 k   | 0.1321 | 7.42×         |  **slower**  |  6 %      |

The CG solver runs a **constant ~85 iterations/step at every rank count** (86/83 at dist_8 …
85/83 at dist_512), so the solver and the domain decomposition are NOT degrading with scale —
whatever happens below is **communication**, not solver/partition quality.

Three regimes:
1. **2→16 nodes — clean strong scaling**, 85–90 % per doubling. ng5 keeps the H100s fed down
   to ~116 k nod2D/GPU. This is the regime to run production in.
2. **16→64 — comms-bound, non-monotonic** (per-doubling 57 % then 78 %). Per-step time here is
   tiny (0.11–0.19 s) and dominated by halo exchange + the ~85 CG `MPI_Allreduce`s, all
   latency-bound. The wiggle is most plausibly **inter-node placement/contention on the
   Dragonfly+ fabric** (which nodes the scheduler assigned), NOT partition quality (CG iters
   are flat). Reps were within 3 %, but the exact shape in this regime should not be over-read.
   **Best absolute time is 64 nodes / 256 GPUs: 0.106 s/step, 9.2× over the 2-node base.**
3. **64→128 — strong scaling REVERSES** (128 nodes is *slower* than 64). At ~14 k wet
   points/GPU almost no compute is left per H100; adding nodes only adds communication.
   **Past ~64 nodes (256 GPUs) ng5 does not benefit; it regresses.**

So the strong-scaling ceiling for ng5 on GH200 is **~64 nodes / 256 GPUs**. Larger meshes
would push that ceiling out (more work to amortise the comms); ng5 itself is exhausted there.
The gentler per-doubling vs Levante A100 (93–96 %) is expected — the H100 finishes each rank's
work so fast that the comms-bound regime arrives at a larger per-rank size than on A100.

## GPU "scaling" — core2 (126,858 nodes), dt=1800 — the small-mesh floor

| nodes | GPUs | nod2D/GPU | s/step |
|:-----:|:----:|:---------:|:------:|
|   1   |  4   |  31.7 k   | 0.0655 |
|   2   |  8   |  15.9 k   | 0.0579 |
|   4   | 16   |   7.9 k   | 0.0572 |
|   8   | 32   |   4.0 k   | 0.0563 |

Flat — 1.16× over an 8× node increase. core2 is **far too small** to occupy even four
H100s (≤32 k wet points/GPU); the timestep is entirely kernel-launch + halo + the per-step
MPI Allreduce in the CG solver. Do not use sub-million-node meshes to benchmark GH200.

## GPU vs CPU node-for-node — the headline (GPU WINS on GH200)

Grace has **72 physical cores/superchip, no SMT → 288 cores/node**; the CPU runs Kokkos
Serial, **1 MPI rank = 1 physical core**. So a full CPU node is 288 ranks.

| mesh  | nodes | GPU (4 H100/node) | CPU (Grace, 1 rank/core) | cores used | **GPU speedup** |
|:------|:-----:|:-----------------:|:------------------------:|:----------:|:---------------:|
| core2 |   1   | 0.0655 (dist_4)   | 0.0859 (dist_288) | **288/288 (full)** | **1.31×** |
| ng5   |   2   | 0.9806 (dist_8)   | 2.4785 (dist_512) | 512/576 = 256/node | **2.53×** (≈2.25× core-normalised) |

⚠️ **The ng5 CPU point used 256 of 288 cores/node**, *not* because that's optimal but because
ng5's pre-generated partitions only exist at 256, 512, 1024, … (multiples of 256) — there is no
`dist_288`/`dist_576` (this port reads partitions, it doesn't generate them). So 256 cores ran,
32/node sat idle (89 %). Normalising to a full 288-core node (assuming ideal CPU scaling, which
is optimistic) drops the ng5 win from 2.53× to **~2.25×**. **core2 (1.31×) is the only fully
apples-to-apples full-node point.** A clean ng5 full-node number needs a `dist_288` partition
generated with the FESOM mesh partitioner (future work).

**Either way, the GPU wins — and this reverses the Levante A100 result.** On Levante the same
port was 3.8–8.9× *slower* per node than EPYC at the PCIe-bound floor, ~1.4× slower even after
the M5.13 device-residency campaign. On GH200 the GPU is **faster** node-for-node — NVLink-C2C
(900 GB/s) erases the host↔device wall that sank the A100, and the H100 out-muscles the Grace
cores. Margin grows with mesh size (1.3× core2 → ~2.2–2.5× ng5): bigger meshes feed the H100s.

> Second caveat (also favouring the CPU, so the win is conservative): the CPU is Kokkos
> *Serial* — 1 thread/rank, no host vectorisation tuning. A threaded/vectorised Grace build
> could narrow the gap somewhat.

### The CPU memory wall (ng5 at ≥4 nodes)

ng5 CPU at **256 ranks/node OOMs at ≥4 nodes** (dist_1024 and dist_2048 both SIGKILL'd by
the OOM killer) — the FESOM port reads the global mesh + PHC IC **redundantly on every rank**
(34 MB PHC + iterative `extrap_nod3D` per rank), so per-node memory and init time blow up
with ranks/node. Levante hit the identical wall ("dist_256 OOMs at 128 ranks/node"). The
2-node ng5 CPU point (512 ranks) fit; 4/8-node did not at full packing. This is also why CPU
runs take ~8–10 min just to *initialise* at high rank counts (the GPU's few-rank layout
never triggers it). Fixing it needs rank-0-reads-and-broadcasts init I/O (future work).

## Notes / caveats

- **CUDA per-step stats print shows `uv=w=Kv=Av=0`** — a cosmetic host-side staleness: the
  velocity / vert-velocity / mixing fields are device-resident (M5.13 device-residency) and
  not synced to host for the diagnostic print. The Serial run prints them nonzero; T/S/eta
  (host-synced) match between Serial and CUDA, confirming the compute is correct. Snapshot
  I/O syncs everything.
- CUDA is **climate-close, not bit-identical** to Serial (device fmad). Serial is the
  bit-identity oracle.
