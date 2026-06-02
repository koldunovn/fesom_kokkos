#!/bin/bash
# Submit the LUMI GPU strong-scaling sweep on the CORE2 mesh.
# Companion to jobs/job_gpu_scaling_lumi. Run from the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."

# nodes / ranks pairs — each ntasks must match an existing dist_<N> partition.
# 1 node × 8  GCDs = dist_8
# 8 nodes × 8       = dist_64
# 16 nodes × 8      = dist_128
# 32 nodes × 8      = dist_256
declare -a SWEEP=(
    "1  8   n1"
    "8  64  n8"
    "16 128 n16"
    "32 256 n32"
)

echo "=== LUMI GPU strong-scaling sweep (CORE2) ==="
for entry in "${SWEEP[@]}"; do
    read -r NODES NTASKS TAG <<<"$entry"
    JOB=$(sbatch --parsable \
        --nodes="$NODES" --ntasks="$NTASKS" \
        --export=ALL,TAG="$TAG" \
        jobs/job_gpu_scaling_lumi)
    printf "  TAG=%-4s nodes=%2s ntasks=%3s → job %s\n" "$TAG" "$NODES" "$NTASKS" "$JOB"
done
echo
squeue -u "$USER" -o "%i %P %j %T %M %D %R"
