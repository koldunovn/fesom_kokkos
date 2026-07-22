#!/bin/bash
# dolpung GH200 strong-scaling fleet (2026-07-22): DP+SP, fast configs, 300 steps.
# PHASE=1 -> single-node points + the core2 g2 multi-node PATHFINDER only.
# PHASE=2 -> the remaining multi-node points (submit after pathfinder is green:
#            the inter-node UCX_TLS (dc_mlx5) is unproven until core2_g2 passes).
# DRY=1 previews. dt: core2 1800, farc 900, dars 120 (rule 0.41), ng5 180;
# SYPD later derived at production dt with measured CG corrections (x1.0222/x1.0110).
set -u
ROOT=/home/a/a270088/port_kokkos
SB="sbatch"; [ "${DRY:-0}" = "1" ] && SB="echo DRY:"
PHASE=${PHASE:?PHASE=1 or 2 required}
declare -A MESH=(
 [core2]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
 [farc]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
 [dars]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
 [ng5]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5 )
declare -A DT=( [core2]=1800 [farc]=900 [dars]=120 [ng5]=180 )
declare -A GPU_N=( [core2]="1 2 4 8" [farc]="1 2 4 8 16 32" [dars]="1 2 4 8 16 32" [ng5]="2 4 8 16 32" )
wall(){ # mesh N -> walltime (8 runs of NSTEPS + init each)
  case "$1_$2" in
    ng5_2) echo 03:30:00;; ng5_4|dars_1|dars_2) echo 02:30:00;;
    ng5_*|dars_*) echo 01:30:00;; farc_1|farc_2) echo 01:30:00;;
    *) echo 01:00:00;;
  esac; }
mkdir -p /work/ab0995/a270088/port2/dolpung/scale
sub=0
for mesh in core2 farc dars ng5; do
  M=${MESH[$mesh]}; dt=${DT[$mesh]}
  for n in ${GPU_N[$mesh]}; do
    t=$((4*n)); d="$M/dist_$t"; tag=${mesh}_g${n}
    [ -d "$d" ] || { echo "SKIP $tag (no dist_$t)"; continue; }
    if [ "$PHASE" = 1 ]; then
      [ "$n" -eq 1 ] || { [ "$tag" = core2_g2 ] || continue; }
    else
      [ "$n" -gt 1 ] && [ "$tag" != core2_g2 ] || continue
    fi
    jid=$($SB --parsable --nodes=$n --time=$(wall $mesh $n) \
          --export=ALL,MESH=$M,DT=$dt,TAG=$tag "$ROOT/jobs/job_dolpung_scale")
    echo "SUBMIT $tag dist_$t ${n}N wall=$(wall $mesh $n) job=$jid"; sub=$((sub+1))
  done
done
echo "submitted: $sub (phase $PHASE)"
