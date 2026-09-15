#!/bin/bash
#SBATCH --job-name=xpart112
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH -o /work/ab0995/a270088/port2/xmach/partgen/xpart112.%j.out
#SBATCH -e /work/ab0995/a270088/port2/xmach/partgen/xpart112.%j.err
#
# jobs/xmach/levante_partgen_112.sh — LEVANTE-SIDE: generate the 112-multiple partitions that a
# MareNostrum 5 GPP node (112 cores) needs for a whole-node CPU ladder. The stock FESOM sets are
# powers of two / multiples of 128 or 144 only.
#
# Same tool and settings as the stock NG5 partitions of 2026-05-28 (fesom_meshpart from
# ~/fesom_part/fesom2, the np_2048 namelist with only MeshPath and n_part swapped), so the new
# rungs are the same kind of partition as the ones the paper measured on. The partitioner runs on
# a COPY of each mesh (never on /pool, never on the private CORE2 — user rule); the resulting
# dist_N directories are then copied into the transfer bundle.
source /sw/etc/profile.levante
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /home/a/a270088/fesom_part/fesom2/env/levante.dkrz.de/shell   # references unset vars: source BEFORE set -u
set -u
ulimit -s 102400
BIN=/home/a/a270088/fesom_part/fesom2/bin/fesom_meshpart
export LD_LIBRARY_PATH=/home/a/a270088/fesom_part/fesom2/mesh_part/build/lib:$LD_LIBRARY_PATH
NML=/home/a/a270088/fesom_part/fesom2/work_part/ng5_part_runs/np_2048/namelist.config
PG=/work/ab0995/a270088/port2/xmach/partgen
BUNDLE=/work/ab0995/a270088/port2/xmach/inputs
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
declare -A SRC=( [core2]=/work/ab0995/a270088/port2/mesh/core2 [ng5]=$POOL/ng5 [dars]=$POOL/dars [farc]=$POOL/farc )
declare -A NPS=( [core2]="112 224 448 896" [farc]="112 224 448 896 1792" [dars]="112 224 448 896 1792" [ng5]="224 448 896 1792 3584" )
echo "partitioner: $(md5sum $BIN | cut -d' ' -f1)   $(date '+%F %T')"
for mesh in core2 farc dars ng5; do
  M=$PG/$mesh; mkdir -p "$M"
  for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out edges.out edge_tri.out edgenum.out; do
    [ -f "$M/$f" ] || cp -p "${SRC[$mesh]}/$f" "$M/"; done
  for NP in ${NPS[$mesh]}; do
    if [ -f "$BUNDLE/$mesh/dist_$NP/rpart.out" ]; then echo "have  $mesh dist_$NP"; continue; fi
    R=$M/run_$NP; mkdir -p "$R"
    sed -e "s|^MeshPath.*|MeshPath         = '$M/'|" -e "s|^[[:space:]]*n_levels.*|n_levels = 1|" \
        -e "s|^[[:space:]]*n_part.*|n_part   = $NP|" "$NML" > "$R/namelist.config"
    cd "$R"; t0=$SECONDS
    srun -n 1 "$BIN" > "$R/meshpart_$NP.log" 2>&1; rc=$?
    n=$(ls "$M/dist_$NP" 2>/dev/null | wc -l)
    if [ $rc -eq 0 ] && [ -f "$M/dist_$NP/rpart.out" ] && [ "$n" -eq $((2*NP+1)) ]; then
      rsync -a "$M/dist_$NP/" "$BUNDLE/$mesh/dist_$NP/"
      echo "OK    $mesh dist_$NP  rc=$rc  $((SECONDS-t0))s  files=$n  -> bundle"
    else
      echo "FAIL  $mesh dist_$NP  rc=$rc  $((SECONDS-t0))s  files=$n (expected $((2*NP+1)))"; tail -5 "$R/meshpart_$NP.log" | sed 's/^/    | /'
    fi
  done
done
echo "ALL DONE $(date '+%F %T')"
