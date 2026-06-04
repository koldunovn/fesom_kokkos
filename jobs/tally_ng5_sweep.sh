#!/bin/bash
# Tally the ng5 GPU rep sweep: per node count, mean/std/min/max/CoV of s/step over reps.
set -u
RUNS=/e/scratch/hclimrep/koldunov1/fesom_runs
printf "%-6s %-5s %-9s %-9s %-9s %-9s %-7s  %s\n" nodes reps mean std min max "CoV%" "reps(s/step)"
for N in 2 4 8 16 32 64 128; do
  vals=()
  for d in "$RUNS"/ng5sweep_n${N}_r*; do
    [ -d "$d" ] || continue
    v=$(grep -hE "loop timing" "$d/log" 2>/dev/null | grep -oE "[0-9.]+ s/step" | grep -oE "^[0-9.]+")
    [ -n "$v" ] && vals+=("$v")
  done
  [ ${#vals[@]} -eq 0 ] && { printf "%-6s (no completed reps)\n" "$N"; continue; }
  printf "%s\n" "${vals[@]}" | awk -v n="$N" '
    {x[NR]=$1; s+=$1; if(NR==1||$1<mn)mn=$1; if(NR==1||$1>mx)mx=$1}
    END{ m=s/NR; for(i=1;i<=NR;i++)v+=(x[i]-m)^2; sd=(NR>1)?sqrt(v/(NR-1)):0;
         printf "%-6s %-5d %-9.4f %-9.4f %-9.4f %-9.4f %-7.1f  ", n, NR, m, sd, mn, mx, 100*sd/m;
         for(i=1;i<=NR;i++)printf "%.4f ", x[i]; print "" }'
done
