#!/bin/bash
# M12 G4 ladder driver — same-day pinned SI-vs-SE pairs, min-of-2 clean reps + 1 phasestats rep.
# Usage: bash m12/submit_g4_ladder.sh <M_farc> <M_dars> [<M_ng5>]
# Points: CORE2 4N GPU · CORE2 16N GPU · farc 2048 CPU · dars 2048 CPU (+ NG5 64 GPU if M_ng5 given).
# House rules: BIN pinned (job default = frozen se0), -C a100_80 on GPU, cheap walltimes, ≤16 nodes.
set -eu
ROOT=/home/a/a270088/port_kokkos
M_FARC=${1:?M for farc}; M_DARS=${2:?M for dars}; M_NG5=${3:-}
CORE2=/work/ab0995/a270088/port2/mesh/core2
FARC=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5

sub() { # sub <name> <partition-args...> -- <exports>
  local name=$1; shift
  local args=(); while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  sbatch -J "$name" "${args[@]}" --export=ALL,"$1" "$ROOT/m12/job_g4_point"
}

pair() { # pair <tag> <sbatch-args...> -- <common-exports-without-ARM>
  local tag=$1; shift
  local args=(); while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  local common=$1
  for arm in si se; do
    for rep in a b; do
      sub "${tag}_${arm}${rep}" "${args[@]}" -- "TAG=${tag}_${arm}${rep},ARM=${arm},PHST=0,${common}"
    done
    sub "${tag}_${arm}p" "${args[@]}" -- "TAG=${tag}_${arm}p,ARM=${arm},PHST=1,${common}"
  done
}

GPU4="-p gpu -C a100_80 -N 4  --ntasks=16 --ntasks-per-node=4 --gres=gpu:4 --gpu-bind=none -t 00:15:00"
GPU16="-p gpu -C a100_80 -N 16 --ntasks=64 --ntasks-per-node=4 --gres=gpu:4 --gpu-bind=none -t 00:15:00"
CPU16="-p compute -N 16 --ntasks=2048 --ntasks-per-node=128 -t 00:25:00"

pair c2_4n  $GPU4  -- "USE_CUDA=1,MESH=$CORE2,DT=1800,KNOBS=FESOM_SPEED=1"
pair c2_16n $GPU16 -- "USE_CUDA=1,MESH=$CORE2,DT=1800,KNOBS=FESOM_SPEED=1"
pair farc   $CPU16 -- "USE_CUDA=0,MESH=$FARC,DT=900,KNOBS=FESOM_WSPLIT=1;FESOM_SE_M=$M_FARC"
pair dars   $CPU16 -- "USE_CUDA=0,MESH=$DARS,DT=120,KNOBS=FESOM_WSPLIT=1;FESOM_SE_M=$M_DARS"
if [ -n "$M_NG5" ]; then
  GPUNG="-p gpu -C a100_80 -N 16 --ntasks=64 --ntasks-per-node=4 --gres=gpu:4 --gpu-bind=none -t 00:25:00"
  pair ng5 $GPUNG -- "USE_CUDA=1,MESH=$NG5,DT=180,KNOBS=FESOM_SPEED=1;FESOM_WSPLIT=1;FESOM_SE_M=$M_NG5"
fi
echo "ladder submitted"
