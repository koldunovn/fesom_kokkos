#!/bin/bash
# jobs/xmach/machine_lumi.sh — LUMI (CSC, Finland) site definitions for the xmach harness.
#
# Sourced by submit_xmach.sh on a login node and by job_xmach / job_xmach_gates inside the batch
# job. EDIT THE FOUR PATHS BELOW ONCE; nothing else in the harness carries a site path.
#
#   LUMI-G node : 4x AMD MI250X = 8 GCDs (one MPI rank per GCD -> 8 ranks/node), 64-core Trento
#   LUMI-C node : 2x AMD EPYC 7763 = 128 cores (one rank per core -> 128 ranks/node)
#
# Toolchains: env_lumi.sh (PrgEnv-amd + rocm 6.3.4 + craype-accel-amd-gfx90a, Kokkos HIP) and
# env_lumi_cpu.sh (PrgEnv-gnu, Kokkos Serial). Built by configure_lumi.sh / configure_lumi_cpu.sh
# into build-hip/ and build-cpu/. See docs/PORT_HIP_LUMI.md for the three LUMI gotchas.
MACHINE=lumi
LABEL_GPU=LUMI_MI250X          # the `machine` column of the paper CSV (data/jupiter_vs_lumi.csv)
LABEL_CPU=LUMI_EPYC7763
ACCOUNT=${ACCOUNT:-project_465002727}
ROOT=${ROOT:-/pfs/lustrep4/projappl/project_465002727/sbeyer/fesom_kokkos_xmach}   # clone of branch xmach-lumi-mn5
INPUTS=${INPUTS:-/scratch/project_465002727/sbeyer/xmach_inputs}                  # the Levante bundle (PACKAGE §3)
RUNBASE=${RUNBASE:-/scratch/project_465002727/sbeyer/xmach_runs}

# side_config gpu|cpu -> ENVSH BIN RPN UNIT LABEL BACKEND SBATCH_SIDE RUNTIME_ENV
side_config () {
  case "$1" in
    gpu) ENVSH=$ROOT/env_lumi.sh;     BIN=$ROOT/build-hip/fesom_port; RPN=8;   UNIT=GCD;  LABEL=$LABEL_GPU; BACKEND=GPU
         # standard-g: up to 1024 nodes, 2 days. dev-g is capped (2 jobs/user, 3 h) — use it only
         # for quick tests via SBATCH_EXTRA="--partition=dev-g".
         SBATCH_SIDE="--partition=standard-g --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7 --mem=0"
         RUNTIME_ENV="MPICH_GPU_SUPPORT_ENABLED=1" ;;    # cray-mpich GPU-aware path (GTL); required for device-pointer MPI
    cpu) ENVSH=$ROOT/env_lumi_cpu.sh; BIN=$ROOT/build-cpu/fesom_port; RPN=128; UNIT=rank; LABEL=$LABEL_CPU; BACKEND=CPU
         # `small` takes <= 4 nodes (faster queue); `standard` up to 512. The submitter picks.
         SBATCH_SIDE="--ntasks-per-node=128 --cpus-per-task=1 --hint=nomultithread --mem=0"
         RUNTIME_ENV="" ;;
    *) echo "side_config: bad side '$1'"; return 2 ;;
  esac
}
# cpu_partition <nodes> -> partition name for a CPU job of that size
cpu_partition () { if [ "$1" -le 4 ]; then echo small; else echo standard; fi; }
gpu_partition () { echo standard-g; }

# CPU rank ladder per mesh (multiples of 128 = whole LUMI-C nodes; every rung exists in the bundle).
cpu_ranks () {
  case "$1" in
    core2) echo "128 256 512 864" ;;          # 864 = the largest stock CORE2 partition (7 nodes, 123/node)
    farc)  echo "128 256 512 1024 2048" ;;
    dars)  echo "128 256 512 1024 2048" ;;
    ng5)   echo "256 512 1024 2048 4096" ;;
  esac
}

# launch <ntasks> <binary> <args...>  — the environment is what the batch script exported.
# Kokkos maps rank -> GCD by SLURM_LOCALID (device = local rank % 8), so no gpu-bind is needed.
launch () { local n=$1; shift; srun -n "$n" "$@"; }
