# Cross-machine re-run of the paper's scaling ladders on LUMI and MareNostrum 5 — the package

**Written 2026-09-15 on Levante. Branch `xmach-lumi-mn5`, repository
`github.com/koldunovn/fesom_kokkos`.** This document is written so that a fresh Claude Code
session (or a person) on LUMI or on MN5 can execute it top-to-bottom without access to Levante.
Everything it depends on is named; where a step needs a decision, the decision rule is given.

---

## 0. The one-paragraph version

The FESOM2 Kokkos-port paper (GMD submission, "an ocean model ported by an LLM") quotes LUMI and
MN5 numbers that were measured in June 2026 with a build that is now ~2× slower than the code the
rest of the paper certifies, and with a 35-step protocol that was later shown to be contaminated.
The job is to **re-measure the same strong-scaling ladders on both machines with the paper's code
and the paper's protocol**: branch `xmach-lumi-mn5` (= the paper's `m14-integrate` plus a
HIP-neutral preprocessor gate and this harness), the paper's input bundle, 300 steps, warm-up leg
discarded, two measured legs, the paper's knob set and nothing else (**no single precision, no
SSH-solver / split-explicit / partitioning / sea-ice lever**). The harness in `jobs/xmach/` does
this mechanically once four paths are filled in; the deliverable is one CSV per machine in the
paper's own schema plus the raw logs.

Order of work: **clone + build (§2) → get the inputs (§3) → gates (§4) → pathfinder (§5.1) →
fleet (§5.2) → harvest (§6).** Read §7 (traps) before the first `sbatch`.

## 1. What code, exactly, and why

| | |
|---|---|
| paper's certified branch | `m14-integrate` @ `d4a9fe0` (its `src/` is unchanged since the paper's build sha `a0b474b`, 2026-08-18) |
| this branch | `xmach-lumi-mn5` = `d4a9fe0` + (a) one backend-neutral gate for the device path, (b) the LUMI/MN5 environments, (c) `jobs/xmach/`, (d) this document |
| what (a) changes | **nothing on CUDA or Serial**: every `#ifdef KOKKOS_ENABLE_CUDA` in `src/` became `#if FESOM_GPU_RESIDENT`, defined in the new `src/fesom_gpu.hpp` as `CUDA || HIP`; the pinned-host mirror space becomes `Kokkos::HIPHostPinnedSpace` under HIP. Verified on Levante by rebuilding the pristine and the edited tree in the same directory and comparing every object file (§10). |
| why (a) is needed | on an AMD/HIP build the old CUDA-only guard compiled the device-resident halo path OUT and resolved every `FESOM_SPEED_*` lever to OFF — correct output, 2–3× slow, silent. The June LUMI branch fixed the halo half of this; the levers (which did not exist in June) would have stayed dead. `docs/PORT_HIP_LUMI.md` has the June story. |
| paper's configuration ("base" arm of every Levante row) | `FESOM_SPEED=1` (the certified tier-1 lever set, incl. CGPIPE; inert on the Serial/CPU build) + `FESOM_IC_EXTRAP=det` (deterministic, partition-independent initial-condition fill) + `FESOM_WSPLIT=1` on farc/dars/ng5 (OFF on core2). Default physics (linear free surface, KPP, standard EVP, GM/Redi on), PHC3 initial condition, JRA55-do 1958 forcing. |
| explicitly OUT | `FESOM_SSH_SOLVER`, `FESOM_SSH_MODE=se`, `FESOM_SPEED_EVPWIDE*`, optimised partitions, any `FESOM_PRECISION`/single-precision build (that is branch `m16-precision`, a different paper). The job scrubs every inherited `FESOM_*` knob before setting the three above, so these cannot leak in. |
| protocol | 300 steps, `snap_every=-1`, dt core2 1800 / farc 900 / dars 120 / ng5 180 s, one warm-up leg discarded, two measured legs in the same allocation, min over admitted legs, the model's own `loop timing` line (its first steps are excluded internally) — never walltime. |
| unit of the GPU ladders | one MPI rank per device: LUMI 8 GCDs/node (one MI250X = two GCDs = two devices), MN5 4 H100/node. CPU ladders: one rank per core, whole nodes (LUMI-C 128, MN5 GPP 112). |

The June measurements are superseded, not corrected: they were a different build (pre-M7), a
35-step window, and on MN5 dt = 240 s. Do not merge the two sets.

## 2. Clone and build

```bash
git clone -b xmach-lumi-mn5 https://github.com/koldunovn/fesom_kokkos.git fesom_kokkos_xmach
cd fesom_kokkos_xmach
git submodule update --init --recursive        # vendored Kokkos 4.4.01 — NEVER a different Kokkos
git log --oneline -1                           # record the sha in your notes
```

Build **two** binaries per machine. The configure scripts source their environment themselves
(modules do not persist across shells on either machine — always build in a fresh shell):

| machine | side | command | output | notes |
|---|---|---|---|---|
| LUMI | GPU (HIP, gfx90a) | `bash -l configure_lumi.sh --clean` | `build-hip/fesom_port` | `env_lumi.sh`: LUMI/25.03 partition/G, PrgEnv-amd, rocm/6.3.4, craype-accel-amd-gfx90a, cray-hdf5 + cray-netcdf, `MPICH_GPU_SUPPORT_ENABLED=1`. Compiler is `amdclang++` directly with `-DMPI_CXX_COMPILER=CC` for FindMPI — the Cray `CC` wrapper concatenates `--rocm-path`/`--offload-arch` and breaks the HIP probe (`docs/PORT_HIP_LUMI.md` gotcha 1). |
| LUMI | CPU (Serial) | `bash -l configure_lumi_cpu.sh --clean` | `build-cpu/fesom_port` | `env_lumi_cpu.sh`: partition/C, PrgEnv-gnu. |
| MN5 | GPU (CUDA, sm_90) | `bash -l configure_mn5.sh --clean` | `build-cuda-mn5/fesom_port` | `env_mn5.sh`: nvidia-hpc-sdk/24.7 (CUDA 12 + HPC-X CUDA-aware OpenMPI), hdf5/1.14.1-2-nvidia-nvhpcx, netcdf nvhpcx, `nvcc_wrapper` with `NVCC_WRAPPER_DEFAULT_COMPILER=g++`, `-DKokkos_ARCH_HOPPER90=ON`. |
| MN5 | CPU (Serial) | `bash -l configure_mn5_cpu.sh --clean` | `build-cpu-mn5/fesom_port` | `env_mn5_cpu.sh`: gcc/12.3.0 + openmpi/4.1.5-gcc. |

These four scripts are the June 2026 ones that built and ran on both machines; module names
may have rotated since — if a `module load` fails, `module spider <name>` and pin the nearest
version, and record the change in the script (it ships back with the results). Two things must
hold whatever you change:

- **The MPI must be GPU-aware** and the device build must be linked against it. On LUMI that is
  cray-mpich + `craype-accel-amd-gfx90a` (GTL) with `MPICH_GPU_SUPPORT_ENABLED=1` at run time;
  on MN5 it is the HPC-X OpenMPI bundled with nvidia-hpc-sdk. A non-GPU-aware MPI segfaults in
  the first halo exchange of step 1 on a device pointer (`fesom_halo_device.cpp`, `MPI_Isend`) —
  that is a build-environment defect, never a code regression. If in doubt, run the halo
  selfcheck gate (§4) before believing anything.
- **Kokkos stays the vendored 4.4.01.** Do not point CMake at a system Kokkos.

Login-node smoke (seconds; needs the bundle from §3): `mpirun -np 1 ./build-*/fesom_port
<INPUTS>/core2 /tmp/smoke 1800 5 -1 <INPUTS>/ic/phc3.0_winter.nc 1958` with
`FESOM_FORCING_DIR=<INPUTS>/forcing_1958 FESOM_CHL_FILE=<INPUTS>/ic/Sweeney_2005.nc` exported.
Expect `loop timing` in the log and no NaN. (Use `dist_1` — it is in the private CORE2 copy.)

## 3. Inputs — the bundle

The paper's inputs, assembled on Levante into ONE directory so it can be copied as a unit:

```
/work/ab0995/a270088/port2/xmach/inputs/          (~25 GB; MANIFEST.sha256 lists every file)
  core2/   nod2d elem2d aux3d nlvls elvls edges edge_tri edgenum .out + README.md MESH_PROVENANCE.md
           dist_{4,8,16,32,64,128,256,512,864}  + dist_{112,224,448,896} (MN5 CPU, new)
  ng5/     7 statics (2.1 GB)   dist_{8,16,32,64,128,256,512,1024,2048,4096} (~0.65 GB each)
           + dist_{224,448,896,1792,3584} (MN5 CPU, new)
  dars/    7 statics (0.9 GB)   dist_{4..2048} + dist_{112,224,448,896,1792}
  farc/    7 statics (0.2 GB)   dist_{4..2048} + dist_{112,224,448,896,1792}
  forcing_1958/  {uas,vas,huss,rsds,rlds,tas,prra,prsn}.1958.nc  PHC2_salx.nc  CORE2_runoff.nc   (11 GB)
  ic/            phc3.0_winter.nc  Sweeney_2005.nc
```

Why these and not the copies already on LUMI/MN5 from June: the CORE2 here is the paper's
**private** copy (the shared `/pool` CORE2 had its level files altered on 2026-07-03; the paper's
runs and references are on this bathymetry — `core2/README.md`), and every `dist_N` is the exact
partition the Levante and JUPITER rows were measured on (NG5's were generated 2026-05-28 with
`fesom_meshpart`; the MN5 112-multiples were generated the same way on 2026-09-15). The port
reads only the seven statics per mesh; nothing else from a FESOM mesh directory is needed.

**Transfer.** The bundle is on DKRZ Levante. Whoever holds both logins pulls it from the target
machine (rsync resumes; run it in `tmux`):

```bash
rsync -avP --partial <dkrz_user>@levante.dkrz.de:/work/ab0995/a270088/port2/xmach/inputs/ <INPUTS>/
cd <INPUTS> && sha256sum -c MANIFEST.sha256 --quiet && echo "bundle verified"
```

`scripts/xmach_pull_from_levante.sh` is that pair of lines with an `ONLY=` filter (e.g.
`ONLY=core2,ng5,forcing_1958,ic` first, `dars,farc` later — the harness SKIPs a rung whose
`dist_N` is not there yet, so nothing has to wait for the whole bundle). If the June copies must be
reused instead (no DKRZ access at all), then at least (1) verify each file against
`MANIFEST.sha256` and use only files that match, and (2) know that the June mesh files on LUMI
came in a Fortran list-directed format (`N*M` repeats, commas) that this branch's reader does not
accept; the June parser fix cherry-picks cleanly (`git cherry-pick ad59408` from branch `LUMI`)
but is deliberately not on this branch, so that its Serial/CUDA binaries stay byte-identical to
the paper's.

## 4. Configure the harness and run the gates

Four paths, in `jobs/xmach/machine_lumi.sh` and `jobs/xmach/machine_mn5.sh`: `ACCOUNT`, `ROOT`
(the clone), `INPUTS` (the bundle), `RUNBASE` (where runs and logs go). Everything else — partitions,
ranks per node, launcher (`srun` on LUMI, HPC-X `mpirun --bind-to none` on MN5), the CPU rank
ladders — is already there and documented in the file. Check the partition/QoS names against
`sinfo`/the project's allocation; adjust in the machine file only.

Then the day-0 gates, **CPU first, then GPU** (the GPU job compares against the CPU output):

```bash
bash jobs/xmach/submit_xmach.sh lumi cpu gates     # or: mn5 cpu gates
bash jobs/xmach/submit_xmach.sh lumi gpu gates     # after the cpu gate finished
grep -h VERDICT <RUNBASE>/logs/xgates_*.out
```

`jobs/xmach/job_xmach_gates` runs CORE2 `dist_8`, 20 steps, snapshots at step 0 and 20:

| gate | what | pass bar | if it fails |
|---|---|---|---|
| G-det (cpu) | Serial np8 twice | snapshots BYTE-IDENTICAL (`scripts/diff_snap.py` exit 0) | host stack nondeterminism: wrong MPI/compiler flags. Do not continue. |
| G-fid (gpu) | GPU np8 (device transport, and again with `FESOM_HALO_STAGE=1`) vs the Serial np8 snapshots | `scripts/gpu_fidelity_check.py` PASS (climate-close ceilings; ~1e-3 floor from FMA/atomics/transcendentals) | a real port/transport fault; check G-self first. **Bitwise** comparison against Serial is NOT the bar for a GPU build and is expected to fail. |
| G-self (gpu) | GPU np8 with `FESOM_HALO_SELFCHECK=1` | SILENT (no mismatch line) | the device halo exchange moved wrong bytes: MPI not GPU-aware, or a HIP transport bug. Stop; this invalidates every GPU number. |
| liveness (both) | the log must contain `[fesom_speed] FESOM_SPEED_SWSKIP = ON` on the GPU build and must NOT on the CPU build; `[fesom_phc] FESOM_IC_EXTRAP=det` on both | as stated | no lever announce on GPU = the dead-knob trap (§7.1) — the build is not `FESOM_GPU_RESIDENT`. |
| G-ab (gpu) | 100-step timing, device-pointer MPI vs `FESOM_HALO_STAGE=1` | informational | **this decides `TRANSPORT` for the fleet** (below). |

The compare steps need `python3` with `numpy` + `netCDF4`; if the compute node has no such
python the job prints the two commands to run on a login node (LUMI: `module load cray-python`
then `pip install --user netCDF4`; MN5: `module load python` or a venv).

**Transport decision.** `TRANSPORT` is exported unchanged into every fleet job and stamped into
every row's `cfg`. Two candidates: `""` = MPI on device pointers (LUMI June: worked, 2.6× the
host-halo; Levante A100: the production path) and `FESOM_HALO_STAGE=1` = the MPI leg runs on
pinned-host mirrors of the packed halo (JUPITER: the only working path on Stages/2025 and the
faster one at scale). Rule: take G-ab's winner on CORE2, **re-check once at the NG5 pathfinder
rung** (run it with both; keep the faster), then freeze `TRANSPORT` for the whole fleet on that
machine. Never `FESOM_HOST_HALO=1` for a measurement — it is the debug fallback that reverts
every exchange to full-field host syncs and deactivates the CG levers (`[cgpipe] INACTIVE`).

## 5. Run

```bash
export TRANSPORT=""                                   # or FESOM_HALO_STAGE=1, from §4
bash jobs/xmach/submit_xmach.sh lumi gpu path         # 5.1 pathfinder: core2 @1 node + ng5 @16 devices
bash jobs/xmach/submit_xmach.sh lumi cpu path         #     core2 @1 node + ng5 @256 ranks
# read them (§6) — then
bash jobs/xmach/submit_xmach.sh lumi gpu fleet        # 5.2 everything up to the paper's range
bash jobs/xmach/submit_xmach.sh lumi cpu fleet
bash jobs/xmach/submit_xmach.sh lumi gpu deep         # 5.3 ng5 512 / dars 512 devices — after the fleet shows the knee
bash jobs/xmach/submit_xmach.sh lumi cpu deep         #     ng5 4096 ranks (LUMI) / 3584 (MN5)
```
`DRY=1` previews; `MESH_FILTER='ng5|core2'` restricts; a rung already submitted is skipped
(`FORCE=1` resubmits); a rung whose partition is not in the bundle is SKIPped and listed.

The ladders (ranks = devices on the GPU side):

| mesh | GPU rungs (both machines; a rung is used only if it is a whole number of nodes) | LUMI-C CPU ranks | MN5 GPP CPU ranks |
|---|---|---|---|
| core2 | 4 8 16 32 64 (LUMI starts at 8 = 1 node) | 128 256 512 864 | 112 224 448 896 |
| farc | 4 8 16 32 64 128 256 | 128 … 2048 | 112 … 1792 |
| dars | 8 16 32 64 128 256 512 | 128 … 2048 | 112 … 1792 |
| ng5 | 16 32 64 128 256 512 | 256 … 4096 | 224 … 3584 |

Priority if allocation is short: **NG5 GPU and CORE2 GPU first** (those are the paper's
cross-machine figure), then the CPU ladders of the same two meshes (node-for-node GPU/CPU ratio
per machine), then dars and farc.

One job = warm-up + 2 legs of 300 steps at one rung. Walltimes in the submitter are deliberately
generous (NG5 at 16 devices: 4 h — the `det` fill is startup cost that grows with mesh size and
shrinks with rank count, ≈7 min at 64 ranks on Levante, so at 16 it can be tens of minutes per
leg; it does not enter the timing). After the pathfinder, trim walltimes to ~2× the measured job
length if backfill rewards it. Roughly 30–40 jobs per machine per side; the deep NG5 rungs
dominate the node-hours.

A healthy job log ends with

```
leg 1 base: rc=0 s/step=0.6521 live=yes
leg 2 base: rc=0 s/step=0.6498 live=yes
=== result ===
  legs=2  min=0.6498  mean=0.6510  std=0.0016  spread=0.35%
  SYPD at dt=180: 0.76
XCSV LUMI_MI250X,ng5,GPU,2,16,GCD,180,0.6498,0.0016,2,300,on,lumi:x_ng5_gpu_16.1234567,paper+wsplit1+device,<md5>,1234567,device
```

A leg is REJECTED (its timing not admitted) with the reason in brackets: `no-final-step`,
`nan-in-state`, `cg-iters-0`, `CG-NaN(rule0.41:wsplit?)`, `NO-LEVER-ANNOUNCE(dead-knob)`,
`CGPIPE-INACTIVE(transport)`, `LEVER-ON-ON-CPU`, `det-not-announced`, `wsplit-not-announced`.
Each of those is a configuration or build fault, never noise — fix the cause, resubmit with
`FORCE=1`. A job with no admitted legs prints `NO ADMITTED LEGS` and no XCSV line.

## 6. Harvest and deliver

```bash
python3 scripts/xmach_harvest.py <RUNBASE>/logs -o xmach_lumi.csv     # prints a table too
```

The CSV's first 13 columns are exactly the paper's `data/jupiter_vs_lumi.csv` schema
(`machine,mesh,backend,nodes,units,unit_kind,dt_s,s_per_step,std_s_per_step,reps,steps,gmredi,source`),
followed by `cfg,binary_md5,job,transport,also`. Rows are the min over admitted legs; a rung run
twice keeps the faster job and lists the other under `also`; rows that differ in `cfg` (wsplit
state, transport) are never merged.

**Deliver back** (one directory or tarball per machine):

1. `xmach_<machine>.csv` (the harvest) and `<RUNBASE>/logs/` (every `.out`/`.err`, including the
   gate logs with their `VERDICT` lines);
2. the git sha built, `md5sum` of both binaries, and the two `env_*.sh` / `configure_*.sh` as
   actually used (if modules were re-pinned, the diff is the record);
3. `module list` output from inside one GPU job and one CPU job;
4. the `TRANSPORT` decision and the two G-ab numbers it was based on;
5. anything that did not fit the protocol (a rung that would not run at 300 steps, a partition
   that OOMs) — say so explicitly rather than substituting a shorter run.

Do not compute or "correct" SYPD at production time step; the paper side does that with its own
CG-iteration corrections. Deliver `s_per_step` at the measured dt.

## 7. Traps, in the order they will bite

1. **The dead-knob trap.** A device build on which the levers do not announce `= ON` is measuring
   the legacy path and will look 2–3× slow while passing every correctness check. The job refuses
   such legs. Cause on a HIP build: a source tree without `src/fesom_gpu.hpp` (i.e. not this
   branch), or a build configured without `-DKokkos_ENABLE_HIP=ON`.
2. **wsplit** is ON for farc/dars/ng5 and OFF for core2, resolved from the mesh name. Without it a
   cold start at the protocol dt dies of a vertical-CFL blow-up whose onset step is roundoff-seeded
   (step 4 to step 291), i.e. it looks like a random solver bug (`CG_kk: pp·App is -nan` on
   **stderr**). It is a configuration, not a lever, and the job checks its announce line.
3. **GPU-aware MPI.** LUMI: `MPICH_GPU_SUPPORT_ENABLED=1` must be in the environment of the
   ranks (the machine file exports it). MN5: HPC-X mpirun, not srun; `SLURM_CPU_BIND=none`.
   Symptom of getting it wrong: segfault in step 1 inside `fesom_halo_device.cpp`.
4. **OpenMPI does not forward the environment** to remote ranks by default. The MN5 launcher
   passes every `FESOM_*`/`HDF5_*`/`UCX_*`/`OMPI_MCA_*` variable with `-x`. If you replace the
   launcher, keep that.
5. **Cross-decomposition snapshots are not comparable** (the initial-condition interpolation is
   partition-dependent; only `det` makes the *state* partition-independent, and only from step 1).
   Every gate pair is the same rank count. Never gate np4 vs np8.
6. **A GPU build is not bitwise reproducible against itself** (atomics). Fidelity is
   `gpu_fidelity_check.py`'s ceilings, not `diff_snap.py`.
7. **Memory.** NG5 at 16 devices is ~460 k surface vertices per rank; it ran on 64 GB A100/GCD/H100
   in June. If a rung is killed for memory, do not shrink the mesh or the step count — drop the rung
   and say so.
8. **Queue hygiene.** Submit the pathfinder alone; submit the fleet only after reading it. On LUMI
   `dev-g` is capped at two concurrent jobs and 3 h — the harness uses `standard-g`/`standard`
   (`small` for ≤4 CPU nodes). Idle interactive allocations count against per-user limits.
9. **Snapshots off.** `snap_every=-1` on every timing leg; at ≥4096 ranks it is mandatory anyway
   (the gather blows the IB registration).
10. **Do not "improve" anything.** No knob, no compiler flag, no Kokkos version, no partition other
    than the bundle's, no dt other than the table's. A variant you think is worth measuring is a
    separate, labelled extra row — never a replacement of the protocol row.

## 8. What to expect (for sanity, not for a pass bar)

The paper's Levante A100 rows (base arm, min s/step, 300 steps, same code) and the JUPITER GH200
baseline, all at the protocol dt:

| mesh | Levante A100 (4/node) | Levante EPYC CPU (128/node) | JUPITER GH200 (4/node) |
|---|---|---|---|
| core2 | 4 GPU 0.0652 · 8 0.0586 · 16 0.0688 · 32 0.0743 · 64 0.0868 | 128 r 0.1987 · 256 0.1065 · 512 0.0590 · 1024 0.0431 | — |
| ng5 | 16 GPU 0.6356 · 32 0.3807 | 1024 r 2.390 · 2048 1.204 · 4096 0.610 | 8 GPU 0.623 · 16 0.323 · 32 0.182 |
| dars | 8 GPU 0.3755 · 16 0.2325 · 64 0.1274 | 512 r 1.593 · 1024 0.837 · 2048 0.405 | — |
| farc | 4 GPU 0.1535 · 8 0.1234 · 16 0.1111 | 512 r 0.2515 · 1024 0.1356 · 2048 0.0799 | — |

June 2026 (old build, 35 steps): LUMI NG5 16 GCD 3.35 s/step, 32 → 1.61, 64 → 0.89, 128 → 0.49,
256 → 0.30; MN5 NG5 (dt 240) 16 H100 1.15, 32 → 0.65, 64 → 0.35, 128 → 0.26. The current build
halved Levante's NG5 step at 16 devices (1.27 → 0.64) relative to that epoch, so the new LUMI
numbers should land roughly at half the June ones and MN5 somewhat below Levante's A100 per
device. A point far outside that band is a setup problem (transport, dead knobs, wrong build)
before it is a result.

## 9. Scope notes

- **dars and farc** complete the paper's four-mesh figure; they are cheaper than NG5 and worth
  running once NG5 and CORE2 are in. **MN5 GPP CPU** needs the 112-multiple partitions, which are
  in the bundle; it is the lowest priority.
- **The June LUMI/MN5 branches** (`LUMI`, `jupiter-gh200`) are history, kept for provenance. Do not
  build from them.
- **Single precision**, and the levers of the "three strategies" paper, are out of scope here even
  though the branch carries the lever code (knobs unset = the paper's base configuration, byte-identical
  to `main` on Serial by the m14 merge gates).

## 10. Verification record (Levante, 2026-09-15)

- Baseline (pristine `d4a9fe0`) and edited trees built in the SAME directory, same toolchain, for
  both backends (Serial: gcc 11.2 + openmpi 4.1.2; CUDA: nvhpc 24.7 + openmpi 4.1.5-nvhpc,
  AMPERE80). The edits are line-count neutral in every file so `__LINE__`-bearing messages and
  DWARF line tables cannot move.
- Object-file comparison of every `fesom_core`/`fesom_port` object: see the VERDICT block below,
  appended when the comparison finished.
- HIP itself cannot be compiled on Levante (no ROCm); the HIP build is verified on LUMI by §4's
  gates. The name `Kokkos::HIPHostPinnedSpace` exists in the vendored Kokkos 4.4.01
  (`externals/kokkos/core/src/HIP/Kokkos_HIP_Space.hpp:131`).

**VERDICT (2026-09-15 14:18, Levante login node, builds in `/scratch/a/a270088/xmach_build`):**

| backend | objects compared | byte-identical | identical after `objcopy --strip-debug` | machine code differs | final binary |
|---|---|---|---|---|---|
| Serial (gcc 11.2 / openmpi 4.1.2) | 45 | 45 | — | **0** | **md5 identical** (`032e3d2ace3929750ac79e373ca0f529`, pristine and edited) |
| CUDA (nvhpc 24.7 / openmpi 4.1.5, AMPERE80) | 45 | 22 | 23 (debug info only) | **0** | **identical after strip-debug** (raw md5 differs — nvcc's debug info is not reproducible even between two builds of the same tree: `ddd5e787…` vs `5277373b…`) |

The 23 CUDA objects whose debug sections differ are exactly the TUs that include
`fesom_halo_device.hpp` (nvcc records the new `fesom_gpu.hpp` in the DWARF file table); their
`.text`/`.rodata` are byte-equal. Conclusion: on the two backends the paper certifies, this
branch produces the same machine code as `m14-integrate` @ `d4a9fe0`. The only behavioural
change is on HIP, where the device path and the levers now exist.

