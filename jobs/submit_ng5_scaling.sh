#!/bin/bash
# NG5 (7.4M) strong-scaling matrix — GPU dist_8/16/32/64 + CPU dist_256/512/1024/2048.
# IDEMPOTENT: submits only partitions that (a) exist + complete (file count >= 2N+1) and
# (b) have not been submitted yet (marker in $MARK). Safe to re-run as partitions appear.
#   node-for-node pairs: gpu_n2↔cpu_n2, n4↔n4, n8↔n8, n16↔n16
#   dt=180 (3 min, per user), 35 steps (5 warmup excluded -> 30 timed). HEAD binary (m512-fusion d+b).
# Usage: bash jobs/submit_ng5_scaling.sh        (DRY=1 to preview)
set -u
ROOT=/home/a/a270088/port_kokkos
MESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
MARK=/work/ab0995/a270088/port2/kokkos_gpu_runs/ng5_submitted
mkdir -p "$MARK"
DT=${DT:-180}; NSTEPS=${NSTEPS:-35}
EXPORT_BASE="ALL,DT=$DT,NSTEPS=$NSTEPS"
SBATCH="sbatch"; [ "${DRY:-0}" = "1" ] && SBATCH="echo DRY: sbatch"

#        N : nodes : tasks : side : tag
MATRIX=(
  "8:2:8:gpu:ng5_gpu_n2"
  "16:4:16:gpu:ng5_gpu_n4"
  "32:8:32:gpu:ng5_gpu_n8"
  "64:16:64:gpu:ng5_gpu_n16"
  "256:2:256:cpu:ng5_cpu_n2"
  "512:4:512:cpu:ng5_cpu_n4"
  "1024:8:1024:cpu:ng5_cpu_n8"
  "2048:16:2048:cpu:ng5_cpu_n16"
)
for row in "${MATRIX[@]}"; do
  IFS=: read -r N nodes tasks side tag <<< "$row"
  if [ -f "$MARK/$tag" ]; then echo "done  $tag (already submitted: $(cat "$MARK/$tag"))"; continue; fi
  d="$MESH/dist_$N"; need=$((2*N+1)); have=$(ls "$d" 2>/dev/null | wc -l)
  if [ ! -d "$d" ] || [ "$have" -lt "$need" ]; then
    echo "wait  $tag (dist_$N: $have/$need files)"; continue
  fi
  jid=$($SBATCH --parsable --nodes=$nodes --ntasks=$tasks \
        --export=${EXPORT_BASE},TAG=$tag "$ROOT/jobs/job_ng5_scaling_$side")
  echo "SUBMIT $tag  dist_$N  ${nodes}N  job=$jid"
  [ "${DRY:-0}" = "1" ] || echo "$jid" > "$MARK/$tag"
done
echo "--- queue ---"; squeue -u "$USER" -o "%.10i %.14j %.8T %.5D %R" 2>/dev/null | grep -E "kk_ng5|JOBID" | head -20