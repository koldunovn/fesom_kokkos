#!/bin/bash
# Chain N ng5 production segments, each resuming from the shared restart dir (auto-resume).
# Each segment is its own SLURM job, linked by --dependency=afterok, so the chain runs
# unattended and stops cleanly if a segment fails (resume by re-pointing at the checkpoint).
#
# Usage:
#   bash jobs/submit_ng5_chain.sh <RUN> <first_seg> <n_segs> <nsteps_per_seg> <dep|none>
# e.g. extend the existing run by 7 x 90-day segments after job 581923:
#   bash jobs/submit_ng5_chain.sh ng5_1958 4 7 43200 581923
#
# dt=180 -> 480 steps/day; 14400=30d, 43200=90d, 175200=1yr. Stay within the staged JRA
# years (run dies crossing into a year whose <var>.<Y>.nc files are missing).
set -u
ROOT=/e/home/jusers/koldunov1/jupiter/fesom_kokkos
RUN=${1:?RUN name}; FIRST=${2:?first seg #}; NSEG=${3:?n segs}; NSTEPS=${4:?nsteps/seg}; DEP=${5:-none}
# walltime sized to the segment: ~0.12 s/step + init, +50% margin, capped at 12h.
secs=$(awk -v n="$NSTEPS" 'BEGIN{printf "%d", (n*0.12+400)*1.5}'); [ "$secs" -gt 43200 ] && secs=43200
wt=$(printf "%02d:%02d:00" $((secs/3600)) $(((secs%3600)/60)))
echo "chaining $NSEG segments x $NSTEPS steps ($((NSTEPS/480)) days each), walltime $wt, RUN=$RUN"
prev=$DEP
for i in $(seq 0 $((NSEG-1))); do
  seg=$(printf "%02d" $((FIRST+i)))
  dep=""; [ "$prev" != none ] && dep="--dependency=afterok:$prev"
  jid=$(sbatch --parsable $dep --time="$wt" \
        --export=ALL,RUN=$RUN,SEG=$seg,NSTEPS=$NSTEPS "$ROOT/jobs/job_jupiter_ng5_production")
  echo "  seg$seg -> job $jid (after ${prev})"
  prev=$jid
done
echo "tail of chain: job $prev (last segment)"
