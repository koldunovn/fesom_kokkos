#!/bin/bash
# jobs/xmach/sub_one.sh — submit ONE rung, bypassing the fleet loop.
#
#   bash jobs/xmach/sub_one.sh <machine> <mesh> <side> <ranks> <walltime> [extra sbatch args]
#
# Exists because submit_xmach.sh derives its walltime from wall(), which is right for a
# healthy run but too short for a rung that has to be retried after a machine-side stall.
# Also takes no marker in $RUNBASE/submitted, so it can always be re-issued.
#
#   TRANSPORT=...  as for submit_xmach.sh (default: the GPU production choice, stage)
#   NSTEPS=300     protocol length
set -u
MACHINE=${1:?machine (lumi|mn5)}; MESH=${2:?mesh}; SIDE=${3:?side (gpu|cpu)}
R=${4:?ranks}; WALL=${5:?walltime}; shift 5
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd "$HERE/../.." && pwd -P)}
source "$HERE/machine_${MACHINE}.sh"
side_config "$SIDE" || exit 2
NODES=$(( R / RPN ))
TRANSPORT=${TRANSPORT:-}
NSTEPS=${NSTEPS:-300}
EXPORT="ALL,ROOT=$ROOT,MACHINE=$MACHINE,SIDE=$SIDE,INPUTS=$INPUTS,RUNBASE=$RUNBASE,NSTEPS=$NSTEPS,TRANSPORT=$TRANSPORT,POINT=$MESH"
mkdir -p "$RUNBASE/logs"
jid=$(sbatch --parsable --account=$ACCOUNT --job-name=x_${MESH}_${SIDE}_${R} \
      --nodes=$NODES --ntasks=$R --time=$WALL $SBATCH_SIDE "$@" \
      --output=$RUNBASE/logs/%x.%j.out --error=$RUNBASE/logs/%x.%j.err \
      --export="$EXPORT" "$ROOT/jobs/xmach/job_xmach")
echo "SUBMIT x_${MESH}_${SIDE}_${R} nodes=$NODES ranks=$R wall=$WALL steps=$NSTEPS transport='${TRANSPORT:-device}' job=$jid"
