#!/bin/bash
# M5.13 post-campaign HIGH-NODE strong-scaling: NG5 + dars on the CAMPAIGN binary
# (build-cuda = a-f+g1-uv+g1-T) at 8 / 16 / 32 nodes, with node-for-node CPU pairs
# where the partition exists. 35 steps (5 warmup excluded -> 30 timed), dt=180, 2 reps.
# GPU: 4 A100/node -> nodes = dist/4.  CPU: 128 cores/node -> nodes = dist/128.
# Fresh *_m513 tags; idempotent (skips missing/incomplete partitions). DRY=1 to preview.
#
# Coverage notes (partition availability, checked 2026-05-30):
#   NG5  dist_32/64/128 (8/16/32N) all READY; NG5 has no dist_4096 -> 32N is GPU-only.
#   dars dist_32 (8N) + dist_128 (32N) READY; dist_64 (16N) NOT pre-generated -> dars 16N skipped.
#   dars has no dist_1024 (8N CPU) -> dars 8N is GPU-only; dars 32N CPU = dist_4096.
set -u
ROOT=/home/a/a270088/port_kokkos
DT=${DT:-180}; NSTEPS=${NSTEPS:-35}; TIME=${TIME:-00:30:00}
EXPORT_BASE="ALL,DT=$DT,NSTEPS=$NSTEPS"
SBATCH="sbatch"; [ "${DRY:-0}" = "1" ] && SBATCH="echo DRY: sbatch"
NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars

# meshdir : nodes : dist : tasks : jobfile : tag
ROWS=(
  # --- GPU (campaign binary) ---
  "$NG5:8:32:32:job_ng5_scaling_gpu:ng5_gpu_n8_m513"
  "$NG5:16:64:64:job_ng5_scaling_gpu:ng5_gpu_n16_m513"
  "$NG5:32:128:128:job_ng5_scaling_gpu:ng5_gpu_n32_m513"
  "$DARS:8:32:32:job_dars_scaling_gpu:dars_gpu_n8_m513"
  "$DARS:32:128:128:job_dars_scaling_gpu:dars_gpu_n32_m513"
  # --- CPU node-for-node pairs (binary is campaign-invariant; run same-session for a clean ratio) ---
  "$NG5:8:1024:1024:job_ng5_scaling_cpu:ng5_cpu_n8_m513"
  "$NG5:16:2048:2048:job_ng5_scaling_cpu:ng5_cpu_n16_m513"
  "$DARS:32:4096:4096:job_dars_scaling_cpu:dars_cpu_n32_m513"
)
echo "=== M5.13 high-node scaling sweep (DT=$DT NSTEPS=$NSTEPS TIME=$TIME) ==="
for row in "${ROWS[@]}"; do
  IFS=: read -r meshdir nodes dist tasks jobfile tag <<< "$row"
  d="$meshdir/dist_$dist"; need=$((2*dist+1)); have=$(ls "$d" 2>/dev/null | wc -l)
  if [ ! -d "$d" ] || [ "$have" -lt "$need" ]; then
    echo "SKIP   $tag (dist_$dist: $have/$need files)"; continue
  fi
  jid=$($SBATCH --parsable --time=$TIME --nodes=$nodes --ntasks=$tasks \
        --export=${EXPORT_BASE},TAG=$tag "$ROOT/jobs/$jobfile")
  echo "SUBMIT $tag  dist_$dist  ${nodes}N  job=$jid"
done
echo "--- queue ---"; squeue -u "$USER" -o "%.10i %.16j %.8T %.5D %.11M %R" 2>/dev/null | grep -E "kk_|JOBID" | head -20
