#!/bin/bash
# Submit the LUMI NG5 strong-scaling sweep.
# Companion to jobs/job_ng5_scaling_lumi. Run from the repo root.
#
# Note: dist_16 + dist_240 fit dev-g (<=49 nodes); dist_512+ needs standard-g.
# dev-g has a 2-concurrent per-user cap; submit standard-g jobs first or accept
# they'll queue while dev-g is busy.
set -euo pipefail

cd "$(dirname "$0")/.."

# nodes / ranks / partition / TAG
declare -a SWEEP=(
    "2   16   dev-g       n2"
    "30  240  dev-g       n30"
    "64  512  standard-g  n64"
)

echo "=== LUMI NG5 strong-scaling sweep ==="
for entry in "${SWEEP[@]}"; do
    read -r NODES NTASKS PART TAG <<<"$entry"
    JOB=$(sbatch --parsable \
        --nodes="$NODES" --ntasks="$NTASKS" \
        --partition="$PART" \
        --export=ALL,TAG="$TAG" \
        jobs/job_ng5_scaling_lumi)
    printf "  TAG=%-4s nodes=%3s ntasks=%4s part=%-11s → job %s\n" \
        "$TAG" "$NODES" "$NTASKS" "$PART" "$JOB"
done
echo
squeue -u "$USER" -o "%i %P %j %T %M %D %R"
