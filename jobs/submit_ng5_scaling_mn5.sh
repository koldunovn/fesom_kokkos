#!/bin/bash
# Submit the MN5 NG5 strong-scaling sweep (companion to job_ng5_scaling_mn5).
# Run from the repo root on MN5 (or use rsync to push first, then ssh + sbatch).
#
# Sweep is "match nodes" (option A): 2/4/8/16/32 MN5 nodes = 8/16/32/64/128 H100.
# All required NG5 dists are present under
#   /gpfs/scratch/ehpc01/sbeyer/inproot/fesom/v21/NG5/dist_{8,16,32,64,128}.
#
# acc_debug QOS is capped at 1 concurrent job and 2h walltime; use acc_ehpc for
# the sweep (3-day cap, no concurrency limit relevant at this size).
set -euo pipefail

cd "$(dirname "$0")/.."

# nodes / ranks / TAG  (1 rank per H100, 4 H100/node)
declare -a SWEEP=(
    "2    8    n2"
    "4    16   n4"
    "8    32   n8"
    "16   64   n16"
    "32   128  n32"
)

echo "=== MN5 NG5 strong-scaling sweep ==="
for entry in "${SWEEP[@]}"; do
    read -r NODES NTASKS TAG <<<"$entry"
    JOB=$(sbatch --parsable \
        --nodes="$NODES" --ntasks="$NTASKS" \
        --export=ALL,TAG="$TAG" \
        jobs/job_ng5_scaling_mn5)
    printf "  TAG=%-4s nodes=%3s ntasks=%4s → job %s\n" \
        "$TAG" "$NODES" "$NTASKS" "$JOB"
done
echo
squeue -u "$USER" -o "%i %P %j %T %M %D %R"
