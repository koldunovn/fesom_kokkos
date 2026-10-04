#!/bin/bash
# jobs/xmach/machine_mn5.sh — MareNostrum 5 (BSC, Spain) site definitions for the xmach harness.
#
# Sourced by submit_xmach.sh on a login node and by job_xmach / job_xmach_gates inside the batch
# job. EDIT THE FOUR PATHS BELOW ONCE; nothing else in the harness carries a site path.
#
#   MN5 ACC node : 4x NVIDIA H100 64 GB (PCIe), 2x Intel Sapphire Rapids 8460Y+ (80 cores)
#                  -> one MPI rank per GPU = 4 ranks/node, 20 cores per rank
#   MN5 GPP node : 2x Intel Sapphire Rapids 8480+ = 112 cores -> 112 ranks/node
#
# Toolchains: env_mn5.sh (nvidia-hpc-sdk/24.7 = CUDA 12 + HPC-X CUDA-aware OpenMPI, Kokkos CUDA
# HOPPER90) and env_mn5_cpu.sh (gcc 12 + openmpi 4.1.5, Kokkos Serial). Built by configure_mn5.sh /
# configure_mn5_cpu.sh into build-cuda-mn5/ and build-cpu-mn5/.
#
# 🔴 HPC-X has NO srun integration: GPU jobs launch with `mpirun --bind-to none` and must export
#    SLURM_CPU_BIND=none (the nvidia-hpc-sdk module banner says so). The CPU side (gcc openmpi)
#    launches with mpirun as well so the two sides are symmetric.
MACHINE=mn5
LABEL_GPU=MN5_H100
LABEL_CPU=MN5_SPR8480
ACCOUNT=${ACCOUNT:-ehpc01}
ROOT=${ROOT:-/gpfs/projects/ehpc01/sbeyer/fesom_kokkos_xmach}          # clone of branch xmach-lumi-mn5
INPUTS=${INPUTS:-/gpfs/scratch/ehpc01/sbeyer/xmach_inputs}             # the Levante bundle (PACKAGE §3)
RUNBASE=${RUNBASE:-/gpfs/scratch/ehpc01/sbeyer/xmach_runs}

side_config () {
  case "$1" in
    gpu) ENVSH=$ROOT/env_mn5.sh;     BIN=$ROOT/build-cuda-mn5/fesom_port; RPN=4;   UNIT=GPU;  LABEL=$LABEL_GPU; BACKEND=GPU
         SBATCH_SIDE="--partition=acc --qos=acc_ehpc --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=20"
         RUNTIME_ENV="SLURM_CPU_BIND=none" ;;
    cpu) ENVSH=$ROOT/env_mn5_cpu.sh; BIN=$ROOT/build-cpu-mn5/fesom_port;  RPN=112; UNIT=rank; LABEL=$LABEL_CPU; BACKEND=CPU
         SBATCH_SIDE="--partition=gpp --qos=gp_ehpc --ntasks-per-node=112 --cpus-per-task=1"
         RUNTIME_ENV="" ;;
    *) echo "side_config: bad side '$1'"; return 2 ;;
  esac
}
cpu_partition () { echo gpp; }
gpu_partition () { echo acc; }

# CPU rank ladder per mesh: multiples of 112 = whole GPP nodes. These partitions do NOT exist in
# the stock FESOM sets; they were generated on Levante for this campaign with the same partitioner
# and settings as the stock NG5 set (jobs/xmach/levante_partgen_112.sh) and ship in the bundle.
cpu_ranks () {
  case "$1" in
    core2) echo "112 224 448 896" ;;
    farc)  echo "112 224 448 896 1792" ;;
    dars)  echo "112 224 448 896 1792" ;;
    ng5)   echo "224 448 896 1792 3584" ;;
  esac
}

# launch <ntasks> <binary> <args...>
# OpenMPI does not forward the whole environment to remote ranks; every FESOM_* / HDF5_* / UCX_* /
# OMPI_* variable that is exported here is passed explicitly with -x.
launch () {
  local n=$1; shift
  local xs=(); local v
  for v in $(env | grep -oE '^(FESOM_[A-Za-z0-9_]*|HDF5_USE_FILE_LOCKING|SLURM_CPU_BIND|UCX_[A-Za-z0-9_]*|OMPI_MCA_[A-Za-z0-9_]*)='); do xs+=(-x "${v%=}"); done
  if [ "$SIDE" = gpu ]; then mpirun --bind-to none "${xs[@]}" -np "$n" "$@"
  else                       mpirun --bind-to core "${xs[@]}" -np "$n" "$@"; fi
}
