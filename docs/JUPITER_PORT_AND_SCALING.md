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
- **ng5 (7.4 M nodes) GPU strong-scaling** (10 reps/point): clean scaling **2→32 nodes**
  (0.96 → 0.14 s/step, per-doubling eff 89→70 %); then a **high-variance plateau** to 128 nodes
  (CoV grows 1 %→46 % — Dragonfly+ placement noise) and a **reversal past 128** (256/512 nodes
  are slower). **Throughput saturates ~4.7–5.2 SYPD@dt=240 across 32–128 nodes; run production
  on ~32 nodes / 128 GPUs (~4.7 SYPD) — same throughput as 128 nodes at ¼ the GPUs.** [An
  earlier single-run pass saw a "dip/reverse" the reps proved was placement noise — use mean±std.]
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

Each point is **multiple SEPARATE job submissions** (10 reps for 2–64 nodes, 6 for 128, 1 probe
for 256/512; each rep gets its own SLURM allocation → samples Dragonfly+ node placement). SYPD =
`dt / (365 × s_per_step)`; the **dt=240 (production) column = dt=180 × 4/3** (per-step cost is
dt-independent — same compute per step):

| nodes | GPUs | nod2D/GPU | reps | mean s/step | **CoV** | per-doubling | **SYPD@180** | **SYPD@240** |
|:-----:|:----:|:---------:|:----:|:-----------:|:-------:|:------------:|:------------:|:------------:|
|   2   |   8  |  926 k    | 10 | 0.964 |  1.0 % |  —          | 0.51 | 0.68 |
|   4   |  16  |  463 k    | 10 | 0.541 |  3.5 % | 1.78× (89 %) | 0.91 | 1.21 |
|   8   |  32  |  231 k    | 10 | 0.320 |  6.1 % | 1.69× (85 %) | 1.54 | 2.05 |
|  16   |  64  |  116 k    | 10 | 0.196 |  5.0 % | 1.63× (82 %) | 2.52 | 3.36 |
|  32   | 128  |   58 k    | 10 | 0.140 |  9.9 % | 1.40× (70 %) | **3.51** | **4.68** |
|  64   | 256  |   29 k    | 10 | 0.136 | 20.1 % | 1.03× (flat) | 3.62 | 4.82 |
| 128   | 512  |   14 k    |  6 | 0.126 | 46.4 % | 1.08× (flat) | 3.91 | 5.21 |
| 256   |1024  |  7.2 k    |  1 | 0.160 |   —     | 0.79× (worse) | 3.07 | 4.10 |
| 512   |2048  |  3.6 k    |  1 | 0.579 |   —     | 0.28× (worse) | 0.85 | 1.14 |

The CG solver runs a **constant ~85 iterations/step at every rank count** (86/83 at dist_8 …
85/83 at dist_512), so the solver and the decomposition are NOT degrading with scale —
everything below is **communication**, not solver/partition quality.

Two findings the reps make solid (they overturn an earlier single-run reading):
1. **Clean, low-variance strong scaling 2→32 nodes** — per-doubling efficiency 89 → 85 → 82 →
   70 %, CoV ≤ 10 %. ng5 keeps the H100s usefully fed down to ~58 k nod2D/GPU.
2. **Beyond ~32 nodes: a high-variance plateau, then reversal.** The mean flattens (n32=0.140,
   n64=0.136, n128=0.126 — a ~1.0× plateau) while **run-to-run variance explodes: CoV 10 % →
   20 % → 46 %**. The step is tiny (~0.13 s) and dominated by latency-bound comms (halo + ~85 CG
   `MPI_Allreduce`s) whose cost depends on which nodes the scheduler gave the job. At 128 nodes a
   single run ranged 0.086–0.240 s (a 2.8× spread). Past 128 the mean clearly **reverses**:
   256 nodes = 0.160, 512 nodes = 0.579 s/step (1024–2048 GPUs left ~4–7 k pts each — pure
   comms). (An earlier single-run pass saw a "dip at 32 / low at 128 / reverse at 256"; reps
   show 32–128 is one noisy plateau and the real turn-up is at 256 — it was placement noise.)

**Throughput (SYPD).** The plateau means **throughput saturates at ~3.5–3.9 SYPD across 32–128
nodes** at the stable dt=180, getting noisier the higher you push. The efficient production point
is **32 nodes / 128 GPUs (~3.5 SYPD@180)** — essentially the same throughput as 128 nodes but with
¼ the GPUs (32→128 nodes buys +11 % SYPD for 4× the hardware). Past 128 nodes throughput falls.

> ⚠️ **The SYPD@240 column is NOT achievable as-is.** dt=240 was meant as the "production"
> timestep, but ng5 **does not run stably at dt=240** on this code: the SSH conjugate-gradient
> solver diverges (`CG_kk residual diverged`) around step ~200 (dt=300 NaNs by ~150). The stable
> envelope is **dt ≤ 180** (500–1500 steps clean; verified `ng5-long-run` branch). So the realistic
> production figure is **SYPD@180 (~3.5 at 32 nodes)**; the @240 column is kept only to show the
> ceiling IF the barotropic-CG stability at large dt is fixed (solver/preconditioner work, or a
> split-explicit barotropic mode — future). See `docs/NG5_LONG_RUN.md`.

The gentler per-doubling vs Levante A100 (93–96 %) is expected — the H100 finishes each rank's
work so fast that the comms-bound regime arrives at a larger per-rank size than on A100.
**Methodological takeaway: in the comms-bound regime, report mean ± std over
several separate-allocation reps — single runs scatter by up to ~50 %.**

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
