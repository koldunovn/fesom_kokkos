#!/bin/bash
# JUPITER GH200 strong-scaling fleet = dolpung FLEET v2 twin + beyond-limit rungs.
# PHASE=1 -> single-node points + the core2_g2 multi-node PATHFINDER only.
# PHASE=2 -> the remaining twin multi-node points (after the pathfinder is green).
# PHASE=3 -> beyond-twin scalability probe: ng5 g128/g256, dars g64/g128/g256
#            (submit after PHASE=2 efficiency is seen; rungs past g256 by hand
#            once the g256 numbers exist — no blind 512-node submissions).
# DRY=1 previews. MESH_FILTER=regex restricts meshes (e.g. MESH_FILTER='farc|dars'
# to submit late-arriving meshes without touching the rest). Missing dist dirs
# auto-SKIP, so this is safe to run before farc/dars finish transferring.
# TRANSPORT: set from the day-0 ladder verdict before PHASE=1:
#   TRANSPORT=""                   device transport
#   TRANSPORT="FESOM_HALO_STAGE=1" staged transport (the dolpung config)
# dt rule 0.41: core2 1800, farc 900, dars 120, ng5 180; SYPD later at production
# dt with the measured CG corrections (dars x1.0222, ng5 x1.0110).
set -u
ROOT=/e/home/jusers/koldunov1/jupiter/port_kokkos
SB="sbatch"; [ "${DRY:-0}" = "1" ] && SB="echo DRY:"
PHASE=${PHASE:?PHASE=1, 2 or 3 required}
TRANSPORT=${TRANSPORT-__UNSET__}
[ "$TRANSPORT" = "__UNSET__" ] && { echo "TRANSPORT must be set explicitly (\"\" or FESOM_HALO_STAGE=1) — run the ladder first"; exit 2; }
MESH_FILTER=${MESH_FILTER:-.}
declare -A MESH=(
 [core2]=/e/scratch/hclimrep/koldunov1/meshes/core2
 [ng5]=/e/scratch/hclimrep/koldunov1/meshes/ng5
 [farc]=/e/scratch/e-sta-destine/koldunov1/meshes/farc
 [dars]=/e/scratch/e-sta-destine/koldunov1/meshes/dars )
declare -A DT=( [core2]=1800 [farc]=900 [dars]=120 [ng5]=180 )
# twin matrix (dolpung FLEET v2) + JUPITER extensions: ng5 g64 (Addendum E) and the
# beyond-limit rungs on ng5+dars (user 2026-07-22: find where scalability dies).
declare -A GPU_N=( [core2]="1 2 4 8" [farc]="1 2 4 8 16 32" \
                   [dars]="1 2 4 8 16 32 64 128 256" [ng5]="2 4 8 16 32 64 128 256" )
deep(){ case "$1_$2" in ng5_128|ng5_256|dars_64|dars_128|dars_256) return 0;; *) return 1;; esac; }
wall(){ # mesh N -> walltime (8 runs of NSTEPS + init each; dolpung-v2-sized)
  case "$1_$2" in
    ng5_2) echo 03:30:00;; ng5_4|dars_1|dars_2) echo 02:30:00;;
    ng5_*|dars_*) echo 01:30:00;; farc_1|farc_2) echo 01:30:00;;
    *) echo 01:00:00;;
  esac; }
mkdir -p /e/scratch/e-sta-destine/koldunov1/port2/scale
sub=0
for mesh in core2 farc dars ng5; do
  echo "$mesh" | grep -qE "$MESH_FILTER" || continue
  M=${MESH[$mesh]}; dt=${DT[$mesh]}
  for n in ${GPU_N[$mesh]}; do
    t=$((4*n)); d="$M/dist_$t"; tag=${mesh}_g${n}
    [ -d "$d" ] || { echo "SKIP $tag (no dist_$t)"; continue; }
    case "$PHASE" in
      1) [ "$n" -eq 1 ] || [ "$tag" = core2_g2 ] || continue ;;
      2) { [ "$n" -gt 1 ] && [ "$tag" != core2_g2 ] && ! deep $mesh $n; } || continue ;;
      3) deep $mesh $n || continue ;;
      *) echo "bad PHASE=$PHASE"; exit 2 ;;
    esac
    jid=$($SB --parsable --nodes=$n --time=$(wall $mesh $n) \
          --export=ALL,MESH=$M,DT=$dt,TAG=$tag,TRANSPORT="$TRANSPORT" "$ROOT/jobs/job_jupiter_scale")
    echo "SUBMIT $tag dist_$t ${n}N wall=$(wall $mesh $n) job=$jid"; sub=$((sub+1))
  done
done
echo "submitted: $sub (phase $PHASE)"
