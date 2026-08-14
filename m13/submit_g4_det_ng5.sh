#!/bin/bash
# M13 → M12 re-board driver: the NG5 CPU ladder under FESOM_IC_EXTRAP=det.
# Same protocol as m12/submit_g4_ladder.sh (same-day pinned pairs, min-of-2 + 1 phasestats
# rep, ladder dt 180, wsplit) — but with deterministic ICs, all six partitions run and the
# legs are cross-partition comparable. Fire on ONE day (same-day baseline rule).
# Usage: bash m13/submit_g4_det_ng5.sh [points...]   (default: 32 64 128 160 192 256)
set -eu
ROOT=/home/a/a270088/port_kokkos
BIN=/work/ab0995/a270088/port2/m13/bin/fesom_port_serial_det1     # m12-tip + det knob, sha 95a4d6a4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
BIG=/work/ab0995/a270088/port2/mesh/ng5_bigpart
M_NG5=20

mesh_for() { case $1 in 32|64) echo $POOL;; *) echo $BIG;; esac; }

sub() { # sub <tag> <nodes> <arm> <phst>
  local tag=$1 nodes=$2 arm=$3 phst=$4
  local knobs="FESOM_WSPLIT=1;FESOM_IC_EXTRAP=det"
  [ "$arm" = se ] && knobs="FESOM_SE_M=$M_NG5;$knobs"
  sbatch -J "$tag" -p compute -N "$nodes" --ntasks=$((nodes*128)) --ntasks-per-node=128 \
    -t 00:20:00 --export=ALL,TAG=$tag,ARM=$arm,MESH=$(mesh_for $nodes),DT=180,NSTEPS=300,PHST=$phst,BIN=$BIN,"KNOBS=$knobs" \
    "$ROOT/m12/job_g4_point"
}

for N in "${@:-32 64 128 160 192 256}"; do
  for arm in si se; do
    sub "ngD${N}_${arm}a" "$N" "$arm" 0
    sub "ngD${N}_${arm}b" "$N" "$arm" 0
    sub "ngD${N}_${arm}p" "$N" "$arm" 1
  done
done
echo "NG5 det ladder submitted (bin $BIN)"
