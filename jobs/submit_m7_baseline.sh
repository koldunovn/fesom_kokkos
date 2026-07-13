#!/bin/bash
# M7 Task 0.2 — same-day re-baseline of HEAD on the standard metric set.
# GPU: NG5@{4,8}N, dars@{2,8}N. CPU: NG5@{4,8}N, dars@8N. dt=180, 35 steps, 2 reps.
# Usage: bash jobs/submit_m7_baseline.sh [DRY=1 to preview]
set -u
MESHROOT=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
NG5=$MESHROOT/ng5; DARS=$MESHROOT/dars
sub(){ # sub <gpu|cpu> <nodes> <ntasks> <mesh> <dt> <tag>
  local kind=$1 nodes=$2 ntasks=$3 mesh=$4 dt=$5 tag=$6
  local job=jobs/job_m7_scale_$kind
  local cmd=(sbatch -N "$nodes" --ntasks="$ntasks" --export=ALL,MESH="$mesh",DT="$dt",TAG="$tag" "$job")
  echo "+ ${cmd[*]}"
  [ "${DRY:-0}" = "1" ] || "${cmd[@]}"
}
# GPU (4 ranks/node)
sub gpu 4  16  "$NG5"  180 base_ng5_gpu_4n
sub gpu 8  32  "$NG5"  180 base_ng5_gpu_8n
sub gpu 2  8   "$DARS" 180 base_dars_gpu_2n
sub gpu 8  32  "$DARS" 180 base_dars_gpu_8n
# CPU (128 ranks/node)
sub cpu 4  512  "$NG5"  180 base_ng5_cpu_4n
sub cpu 8  1024 "$NG5"  180 base_ng5_cpu_8n
sub cpu 8  1024 "$DARS" 180 base_dars_cpu_8n
