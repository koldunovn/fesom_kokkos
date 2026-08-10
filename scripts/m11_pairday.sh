#!/bin/bash
# M11 ordering race — submit ONE pair day: all five points, both backends, in one command.
#
#   bash scripts/m11_pairday.sh          # submit
#   bash scripts/m11_pairday.sh --dry    # print what would be submitted
#
# The adoption rule requires two same-day pair days (docs/PARTITIONING_M11.md). Day 1 is
# 2026-08-10 — with the CPU-256 point re-run on the corrected baseline (Finding 10) and CPU 864
# added once FESOM_PART_FIXISO made its arms invariant-identical (Finding 13). Day 2 is simply
# this script, run on a different day.
#
# Each job races base/hil/rcm inside ONE allocation, 2 reps, min-of-2, and refuses to start
# unless the three arms carry the same partition under the node permutation. RCM is raced and
# reported but is NOT adoptable — it fails the SSH-iteration gate at 256 and 512 (Finding 12).
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
DRY=${1:-}
sub() {
    echo "  $*"
    [ "$DRY" = "--dry" ] || sbatch "$@" | sed 's/^/    /'
}
echo "=== M11 pair day, $(date '+%F')  (all arms of a point inside one allocation, min-of-2)"
echo "--- CPU, Serial h17 5c3c90fc, CORE2 256 / 512 / 864 ranks"
sub -N 2 --ntasks=256 "$ROOT/jobs/m11_race_cpu.sh"
sub -N 4 --ntasks=512 "$ROOT/jobs/m11_race_cpu.sh"
sub -N 7 --ntasks=864 "$ROOT/jobs/m11_race_cpu.sh"
echo "--- GPU, CUDA h17 f8384e86, -C a100_80, CORE2 4 / 8 ranks (<= 16 nodes, standing rule)"
sub -N 1 --ntasks=4 "$ROOT/jobs/m11_race_gpu.sh"
sub -N 2 --ntasks=8 "$ROOT/jobs/m11_race_gpu.sh"
echo
echo "results: /work/ab0995/a270088/port2/m11/race{cpu,gpu}.<jobid>.out — take the min-of-2 block"
