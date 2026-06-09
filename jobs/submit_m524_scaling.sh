#!/bin/bash
# M5.24 full GPU+CPU strong-scaling sweep across CORE2/farc/dars/NG5 (2026-05-31).
# GPU: build-cuda (M5.24), 4 A100/node, dist=4*nodes.  CPU: build-serial (M5.24), 128 ranks/node, dist=128*nodes.
# dt: core2 1800, farc 900, dars/ng5 240 (4-min post-spinup step, user 2026-05-31).
# Idempotent-ish: skips (mesh,nodes) whose dist_<tasks> partition is missing, and LISTS them.
# DRY=1 previews (no sbatch). NSTEPS default 35 (5 warmup excluded by the loop timer; 2 reps).
set -u
ROOT=/home/a/a270088/port_kokkos
NSTEPS=${NSTEPS:-35}
SB="sbatch"; [ "${DRY:-0}" = "1" ] && SB="echo DRY:"
declare -A MESH=(
 [core2]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
 [farc]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
 [dars]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
 [ng5]=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5 )
declare -A DT=( [core2]=1800 [farc]=900 [dars]=240 [ng5]=240 )
# GPU node ladders (dist=4*nodes). NG5 has no dist_4 (1N too big for 4 A100).
# CORE2 capped at 8N: 127k nodes goes inverse-scaling past ~2N (GPU-starved); 16/32N just waste a big alloc.
declare -A GPU_N=( [core2]="1 2 4 8" [farc]="1 2 4 8 16 32" [dars]="1 2 4 8 16 32" [ng5]="2 4 8 16 32" )
# CPU node ladders (dist=128*nodes). NG5 1N/2N OOM the 256GB node -> start at 4N.
declare -A CPU_N=( [core2]="1 2 4" [farc]="1 2 4 8 12" [dars]="1 2 4 8 16 32" [ng5]="4 8 16 32" )

sub=0; skip=""
for mesh in core2 farc dars ng5; do
  M=${MESH[$mesh]}; dt=${DT[$mesh]}
  for n in ${GPU_N[$mesh]}; do
    t=$((4*n)); d="$M/dist_$t"; need=$((2*t+1)); have=$(ls "$d" 2>/dev/null | wc -l); tag=m524_${mesh}_gpu_n${n}
    if [ ! -d "$d" ] || [ "$have" -lt "$need" ]; then skip+="  GPU $tag  (need dist_$t : $have/$need files)\n"; continue; fi
    jid=$($SB --parsable --nodes=$n --ntasks=$t --time=01:00:00 \
          --export=ALL,MESH=$M,DT=$dt,NSTEPS=$NSTEPS,TAG=$tag "$ROOT/jobs/job_m524_scale_gpu")
    echo "SUBMIT GPU $tag  dist_$t  ${n}N  job=$jid"; sub=$((sub+1))
  done
  for n in ${CPU_N[$mesh]}; do
    t=$((128*n)); d="$M/dist_$t"; need=$((2*t+1)); have=$(ls "$d" 2>/dev/null | wc -l); tag=m524_${mesh}_cpu_n${n}
    if [ ! -d "$d" ] || [ "$have" -lt "$need" ]; then skip+="  CPU $tag  (need dist_$t : $have/$need files)\n"; continue; fi
    jid=$($SB --parsable --nodes=$n --ntasks=$t --time=00:45:00 \
          --export=ALL,MESH=$M,DT=$dt,NSTEPS=$NSTEPS,TAG=$tag "$ROOT/jobs/job_m524_scale_cpu")
    echo "SUBMIT CPU $tag  dist_$t  ${n}N  job=$jid"; sub=$((sub+1))
  done
done
echo "=== submitted=$sub ==="
echo -e "=== SKIPPED (missing partition — candidates for fesom_ini.x) ===\n$skip"
