#!/bin/bash
# Submit the dars (3.16M) strong-scaling matrix: 3 GPU + 3 CPU runs.
# Each TAG → /work/ab0995/a270088/port2/kokkos_gpu_runs/<TAG>/
#
# Pairing (node-for-node):
#   2 nodes:  GPU dist_8  (8 A100)  ↔ CPU dist_256 (256 cores)
#   4 nodes:  GPU dist_16 (16 A100) ↔ CPU dist_512 (512 cores)
#   8 nodes:  GPU dist_32 (32 A100) ↔ CPU dist_1152* (1152 cores on 9 nodes — closest)
#
# dt=180 (dars typically stable at 3 min, 4 min if lucky — per user 2026-05-28).
# 35 steps (5 warmup excluded → 30 timed).
#
# Usage:  bash jobs/submit_dars_scaling.sh
# Or DRY=1 to preview without submitting.

set -u
ROOT=/home/a/a270088/port_kokkos
SBATCH="sbatch"
[ "${DRY:-0}" = "1" ] && SBATCH="echo DRY: sbatch"

DT=${DT:-180}
NSTEPS=${NSTEPS:-35}
EXPORT_BASE="ALL,DT=$DT,NSTEPS=$NSTEPS"

# --- GPU side ---
$SBATCH --nodes=2 --ntasks=8  --export=${EXPORT_BASE},TAG=dars_gpu_n2 "$ROOT/jobs/job_dars_scaling_gpu"
$SBATCH --nodes=4 --ntasks=16 --export=${EXPORT_BASE},TAG=dars_gpu_n4 "$ROOT/jobs/job_dars_scaling_gpu"
$SBATCH --nodes=8 --ntasks=32 --export=${EXPORT_BASE},TAG=dars_gpu_n8 "$ROOT/jobs/job_dars_scaling_gpu"

# --- CPU side ---
$SBATCH --nodes=2 --ntasks=256  --export=${EXPORT_BASE},TAG=dars_cpu_n2 "$ROOT/jobs/job_dars_scaling_cpu"
$SBATCH --nodes=4 --ntasks=512  --export=${EXPORT_BASE},TAG=dars_cpu_n4 "$ROOT/jobs/job_dars_scaling_cpu"
$SBATCH --nodes=9 --ntasks=1152 --export=${EXPORT_BASE},TAG=dars_cpu_n9 "$ROOT/jobs/job_dars_scaling_cpu"

echo "Submitted. Watch with: squeue -u $USER | grep kk_dars"
echo "Tally: grep -h 'loop timing\\|ERROR' /work/ab0995/a270088/port2/kokkos_gpu_runs/dars_{gpu,cpu}_n{2,4,8,9}/log_rep_*.txt 2>/dev/null"
