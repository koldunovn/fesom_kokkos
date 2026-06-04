#!/bin/bash
# ng5 GPU strong-scaling REP sweep — multiple SEPARATE jobs per node count so each rep
# gets a fresh SLURM allocation (samples Dragonfly+ placement variability). Report
# mean ± spread per partition to settle whether the 16->64-node non-monotonicity is real
# or placement noise. dt=180, 55 steps (50 timed), JRA55 1958, no I/O.
#
# Usage:  bash jobs/submit_ng5_sweep.sh            # submit
#         DRY=1 bash jobs/submit_ng5_sweep.sh      # preview
# Tags:   ng5sweep_n<NODES>_r<REP>  under /e/scratch/hclimrep/koldunov1/fesom_runs/
set -u
SBATCH=sbatch; [ "${DRY:-0}" = 1 ] && SBATCH="echo DRY:"
# node count -> reps (fewer reps at the largest count to cap the node footprint)
declare -A REPS=( [2]=5 [4]=5 [8]=5 [16]=5 [32]=5 [64]=5 [128]=3 )
for N in 2 4 8 16 32 64 128; do
  R=${REPS[$N]}; NT=$((N*4))
  for r in $(seq 1 "$R"); do
    $SBATCH --nodes="$N" --ntasks="$NT" \
      --export=ALL,MESH=ng5,TAG=ng5sweep_n${N}_r${r},YEAR=1958,DT=180,NSTEPS=55 \
      jobs/job_jupiter_gpu
  done
done
echo "submitted ng5 sweep. collect with: bash jobs/tally_ng5_sweep.sh"
