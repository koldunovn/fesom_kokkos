#!/bin/bash
#SBATCH --job-name=m11gimb
#SBATCH -p gpu
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH -C a100_80
#SBATCH --time=00:25:00
#SBATCH -o /work/ab0995/a270088/port2/m11/gimb.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/gimb.%j.err
#
# M11 — WHERE is a GPU rank waiting at a small rank count?
#
# The partition arms all LOSE at CORE2 GPU 4 ranks, and the reason I gave (the shipped
# partition is already balanced) rests on 2-D and 3-D node counts, which balance to 1 %. But
# the polar node count across those same four parts is 3.35x max/min, and sea ice is the one
# component whose work is spatially concentrated. Counting nodes cannot decide this; the
# per-rank busy/wait breakdown can.
#
# Uses the frozen PHASESTATS build (m7/bin/phst1, commit 0ec70cf) — a DIFFERENT binary from the
# certified h17, so its absolute step time is not comparable to any race number. It is a
# diagnostic, and it is labelled as one.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/env_cuda.sh"
export OMPI_MCA_pml=ucx OMPI_MCA_btl=self UCX_NET_DEVICES=mlx5_0:1
export UCX_MEMTYPE_CACHE=n OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_io=romio321 HDF5_USE_FILE_LOCKING=FALSE
export FESOM_PRINT_EVERY=999
export FESOM_SPEED=1
export FESOM_SPEED_PHASESTATS=1

SB=/work/ab0995/a270088/port2/mesh_m11
BIN=/work/ab0995/a270088/port2/m7/bin/phst1/fesom_port_cuda
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
DT=${DT:-1800}; NSTEPS=${NSTEPS:-100}
MESHES=${MESHES:-"base=$SB/core2_base"}
OUT=/work/ab0995/a270088/port2/m11/gimb.${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "=== M11 GPU imbalance probe  ranks=$SLURM_NTASKS  steps=$NSTEPS  dt=$DT  $(date '+%F %T')"
echo "    BIN=$BIN md5=$(md5sum $BIN | cut -d' ' -f1)  (PHASESTATS build, NOT h17)"

IFS=',' read -ra SPECS <<< "$MESHES"
for spec in "${SPECS[@]}"; do
    n=${spec%%=*}; d=${spec#*=}
    o="$OUT/$n"; mkdir -p "$o"
    echo; echo "--- $n  $d"
    srun "$BIN" "$d" "$o" "$DT" "$NSTEPS" -1 "$PHC" 1958 > "$OUT/log_$n.txt" 2>&1
    echo "    rc=$?"
    sed -n '/PHASESTATS/,$p' "$OUT/log_$n.txt" | head -60
done
echo; echo "=== done $(date '+%F %T')"
