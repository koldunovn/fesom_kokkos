#!/bin/bash
# JUPITER s26 fleet — remeasure the full matrix on Stage-2026 + DEVICE transport
# (PSMPI 5.13 CUDA-aware, Kokkos 4.7.03, CUDA 13; certified j26ab/j26ab2
# 2026-07-23: DP fidelity PASS np4+np8 both transports, SP noise-envelope PASS
# with the documented single-cell uice waiver, selfchecks silent, device beats
# STAGE by 5-11% at g1/g2). Protocol otherwise identical to the s25 twin fleet
# (4 legs x 2 reps, 300 steps, dt rule 0.41). Results -> port2/scale26 so the
# dolpung-twin table stays untouched — two series, one protocol.
# No PHASE gating: inter-node device transport was pathfound at core2_g2.
# DRY=1 previews; MESH_FILTER=regex restricts; TRANSPORT overridable
# (default "" = device; FESOM_HALO_STAGE=1 for staged spot-checks).
set -u
ROOT=/e/home/jusers/koldunov1/jupiter/port_kokkos_s26
SB="sbatch"; [ "${DRY:-0}" = "1" ] && SB="echo DRY:"
TRANSPORT=${TRANSPORT-}
MESH_FILTER=${MESH_FILTER:-.}
declare -A MESH=(
 [core2]=/e/scratch/hclimrep/koldunov1/meshes/core2
 [ng5]=/e/scratch/hclimrep/koldunov1/meshes/ng5
 [farc]=/e/scratch/e-sta-destine/koldunov1/meshes/farc
 [dars]=/e/scratch/e-sta-destine/koldunov1/meshes/dars )
declare -A DT=( [core2]=1800 [farc]=900 [dars]=120 [ng5]=180 )
# mirror every point measured on s25 (deep rungs g256/g128 join when their s25
# twins land, to keep the ladders in lockstep)
declare -A GPU_N=( [core2]="1 2 4 8" [farc]="1 2 4 8 16 32" \
                   [dars]="1 2 4 8 16 32 64" [ng5]="2 4 8 16 32 64 128" )
wall(){ case "$1_$2" in
    ng5_2) echo 03:30:00;; ng5_4|dars_1|dars_2) echo 02:30:00;;
    ng5_*|dars_*) echo 01:30:00;; farc_1|farc_2) echo 01:30:00;;
    *) echo 01:00:00;; esac; }
mkdir -p /e/scratch/e-sta-destine/koldunov1/port2/scale26
sub=0
for mesh in core2 farc dars ng5; do
  echo "$mesh" | grep -qE "$MESH_FILTER" || continue
  M=${MESH[$mesh]}; dt=${DT[$mesh]}
  for n in ${GPU_N[$mesh]}; do
    t=$((4*n)); d="$M/dist_$t"; tag=${mesh}_g${n}
    [ -d "$d" ] || { echo "SKIP $tag (no dist_$t)"; continue; }
    jid=$($SB --parsable --nodes=$n --time=$(wall $mesh $n) \
          --export=ALL,MESH=$M,DT=$dt,TAG=$tag,TRANSPORT="$TRANSPORT" "$ROOT/jobs/job_jupiter_scale26")
    echo "SUBMIT $tag dist_$t ${n}N wall=$(wall $mesh $n) job=$jid"; sub=$((sub+1))
  done
done
echo "submitted: $sub (s26 device fleet)"
