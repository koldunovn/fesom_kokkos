#!/bin/bash
# jobs/xmach/submit_xmach.sh — submit the cross-machine ladder, one phase at a time.
#
#   bash jobs/xmach/submit_xmach.sh <lumi|mn5> <gpu|cpu> <gates|path|fleet|deep>
#
#   gates : the day-0 gate job (job_xmach_gates) — run cpu FIRST, then gpu
#   path  : pathfinder — core2 at ONE node + ng5 at its smallest rung. Read these before the fleet:
#           they calibrate the walltimes and confirm the machine is in the expected range.
#   fleet : every rung of every mesh up to the paper's range (GPU: core2 1-16 nodes, ng5 to 256
#           devices, dars/farc to 256; CPU: the cpu_ranks ladder of the machine file)
#   deep  : the far rungs (ng5 512 devices, dars 512, ng5 4096 CPU ranks) — submit after the fleet
#           shows where scaling turns over; do not submit blind
#
# Options (environment):
#   TRANSPORT=""                  device-pointer MPI (default)      } set from the gates' A/B;
#   TRANSPORT="FESOM_HALO_STAGE=1" staged pinned-host MPI leg        } stamped into every row's cfg
#   MESH_FILTER='ng5|core2'       regex on mesh names
#   DRY=1                         print the sbatch lines, submit nothing
#   FORCE=1                       resubmit rungs already marked as submitted
#   NSTEPS=300                    the protocol length (do not shorten for real rows)
#   SBATCH_EXTRA="..."            appended verbatim (e.g. --partition=dev-g, --reservation=...)
#
# IDEMPOTENT: a marker in $RUNBASE/submitted/<jobname> stops a rung from being submitted twice;
# a missing dist_N in the bundle SKIPs the rung (so this is safe before the transfer is complete).
set -u
MACHINE=${1:?machine (lumi|mn5)}; SIDE=${2:?side (gpu|cpu)}; PHASE=${3:?phase (gates|path|fleet|deep)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); ROOT=${ROOT:-$(cd "$HERE/../.." && pwd -P)}
source "$HERE/machine_${MACHINE}.sh"; side_config "$SIDE" || exit 2
TRANSPORT=${TRANSPORT:-}; MESH_FILTER=${MESH_FILTER:-.}; NSTEPS=${NSTEPS:-300}; SBATCH_EXTRA=${SBATCH_EXTRA:-}
SB=sbatch; [ "${DRY:-0}" = 1 ] && SB="echo DRY: sbatch"
mkdir -p "$RUNBASE/logs" "$RUNBASE/submitted"
EXPORT="ALL,ROOT=$ROOT,MACHINE=$MACHINE,SIDE=$SIDE,INPUTS=$INPUTS,RUNBASE=$RUNBASE,NSTEPS=$NSTEPS,TRANSPORT=$TRANSPORT"

# GPU rank ladders (ranks = devices); a rung is used only if it is a whole number of nodes here.
gpu_ranks () { case "$1" in core2) echo "4 8 16 32 64";; farc) echo "4 8 16 32 64 128 256";;
                            dars) echo "8 16 32 64 128 256 512";; ng5) echo "16 32 64 128 256 512";; esac; }
deep_rung () { case "${SIDE}_$1_$2" in gpu_ng5_512|gpu_dars_512|cpu_ng5_4096|cpu_ng5_3584) return 0;; *) return 1;; esac; }
path_rung () { local minr; minr=$(ranks_for "$1" | awk '{print $1}')
               [ "$1" = core2 ] && [ "$2" -eq "$RPN" ] && return 0
               [ "$1" = ng5 ] && [ "$2" -eq "$minr" ] && return 0; return 1; }
ranks_for () { local r; for r in $( [ "$SIDE" = gpu ] && gpu_ranks "$1" || cpu_ranks "$1" ); do
                 [ $((r % RPN)) -eq 0 ] && [ "$r" -ge "$RPN" ] && echo "$r"; done; }
# Walltime: warmup + 2 legs x (det fill + NSTEPS steps). Generous by design; the pathfinder rungs
# tell you the real cost — trim afterwards if the queue rewards it (backfill), never below 2x measured.
wall () {  # wall <mesh> <ranks>
  case "${SIDE}_$1" in
    gpu_core2) echo 00:40:00;; gpu_farc) echo 01:00:00;;
    gpu_dars)  [ "$2" -le 16 ] && echo 02:30:00 || echo 01:30:00;;
    gpu_ng5)   [ "$2" -le 16 ] && echo 04:00:00 || { [ "$2" -le 64 ] && echo 02:30:00 || echo 01:30:00; };;
    cpu_core2) echo 00:45:00;; cpu_farc) echo 01:30:00;; cpu_dars) echo 02:30:00;;
    cpu_ng5)   [ "$2" -le 512 ] && echo 04:00:00 || echo 02:30:00;;
  esac; }

if [ "$PHASE" = gates ]; then
  nodes=$(( (8 + RPN - 1) / RPN )); part=$( [ "$SIDE" = gpu ] && gpu_partition || cpu_partition $nodes )
  # the gates want exactly 8 ranks: override the per-node task count from the side config
  side_flags=$(echo "$SBATCH_SIDE" | sed -E 's/--ntasks-per-node=[0-9]+//; s/--partition=[^ ]+//')
  jid=$($SB --parsable --account=$ACCOUNT --partition=$part --job-name=xgates_${MACHINE}_${SIDE} --nodes=$nodes --ntasks=8 \
        --ntasks-per-node=$(( 8 / nodes )) --time=00:45:00 $side_flags $SBATCH_EXTRA \
        --output=$RUNBASE/logs/%x.%j.out --error=$RUNBASE/logs/%x.%j.err --export=$EXPORT "$ROOT/jobs/xmach/job_xmach_gates")
  echo "SUBMIT gates $MACHINE/$SIDE nodes=$nodes job=$jid"; exit 0
fi

sub=0; skip=0
for mesh in core2 farc dars ng5; do
  echo "$mesh" | grep -qE "$MESH_FILTER" || continue
  for r in $(ranks_for "$mesh"); do
    case "$PHASE" in
      path)  path_rung "$mesh" "$r" || continue ;;
      fleet) { deep_rung "$mesh" "$r" || path_rung "$mesh" "$r"; } && continue ;;
      deep)  deep_rung "$mesh" "$r" || continue ;;
      *) echo "bad PHASE=$PHASE"; exit 2 ;;
    esac
    name=x_${mesh}_${SIDE}_${r}; nodes=$(( r / RPN ))
    [ -d "$INPUTS/$mesh/dist_$r" ] || { echo "SKIP   $name (no $INPUTS/$mesh/dist_$r)"; skip=$((skip+1)); continue; }
    if [ -f "$RUNBASE/submitted/$name" ] && [ "${FORCE:-0}" != 1 ]; then echo "done   $name (job $(cat $RUNBASE/submitted/$name); FORCE=1 to resubmit)"; continue; fi
    part=$( [ "$SIDE" = gpu ] && gpu_partition || cpu_partition $nodes )
    side_flags=$(echo "$SBATCH_SIDE" | sed -E 's/--partition=[^ ]+//')
    jid=$($SB --parsable --account=$ACCOUNT --partition=$part --job-name=$name --nodes=$nodes --ntasks=$r --time=$(wall $mesh $r) \
          $side_flags $SBATCH_EXTRA --output=$RUNBASE/logs/%x.%j.out --error=$RUNBASE/logs/%x.%j.err \
          --export=$EXPORT,POINT=$mesh "$ROOT/jobs/xmach/job_xmach")
    echo "SUBMIT $name  dist_$r  ${nodes}N  wall=$(wall $mesh $r)  transport='${TRANSPORT:-device}'  job=$jid"
    [ "${DRY:-0}" = 1 ] || echo "$jid" > "$RUNBASE/submitted/$name"; sub=$((sub+1))
  done
done
echo "submitted: $sub  skipped(no partition): $skip   [$MACHINE/$SIDE/$PHASE]"
