#!/bin/bash
# M10 — complete the mesh x backend ladders.
#
# WHY: after the SSH-share campaign only CORE2 had a real LADDER (and only on CPU). farc,
# dars and NG5 had 1-2 isolated points, which is enough for a spot delta but NOT enough for
# the analysis that turned out to matter: parallel efficiency, the scaling knee, and where
# the benefit switches on. This fills the gaps on BOTH backends.
#
# Practical framing (user, 2026-08-06): CORE2 @ 512 CPU ranks is the production configuration;
# results far past the scalability limit are not useful. So each ladder brackets the plausible
# production range for its mesh rather than chasing extreme rank counts.
#
# 🔴 farc is capped BELOW 128 ranks — R8/E.T1 reproducible proto hang at >=128.
#
# Usage: bash jobs/m10_submit_ladders.sh
set -u
ROOT=${M10_ROOT:-$HOME/port_kokkos_ssh}
cd "$ROOT"
CORE2=/work/ab0995/a270088/port2/mesh/core2
FARC=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
L="FESOM_SPEED=1;FESOM_SPEED=1+FESOM_SSH_SOLVER=cg2;FESOM_SPEED=1+FESOM_SSH_SOLVER=oati;FESOM_SPEED=1+FESOM_SSH_SOLVER=pcsi"

sub () {  # sub <gpu|cpu> <nodes> <ntasks> <mesh> <dt> <tag>
    local B=$1 N=$2 NT=$3 MESH=$4 DT=$5 TAG=$6
    [ -d "$MESH/dist_$NT" ] || { printf '  SKIP %-26s dist_%s missing\n' "$TAG" "$NT"; return; }
    case "$MESH" in *farc*) [ "$NT" -ge 128 ] && { printf '  REFUSE %-24s farc >=128 ranks = R8 hang\n' "$TAG"; return; };; esac
    local JOB=jobs/job_m10_ab; [ "$B" = cpu ] && JOB=jobs/job_m10_ab_cpu
    local id
    id=$(sbatch -N "$N" --ntasks="$NT" \
         --export=ALL,MESH="$MESH",DT="$DT",TAG="$TAG",NSTEPS=300,LEGS="$L" "$JOB" \
         | grep -o '[0-9]*')
    printf '  %-28s %-3s %5s ranks  job %s\n' "$TAG" "$B" "$NT" "$id"
}

echo "=== GPU ladders (4 ranks/node) ==="
sub gpu  1    4 "$CORE2" 1800 lad_core2_g1n
sub gpu  2    8 "$CORE2" 1800 lad_core2_g2n
sub gpu 16   64 "$DARS"  120  lad_dars_g16n
sub gpu 32  128 "$DARS"  120  lad_dars_g32n
sub gpu 32  128 "$NG5"   180  lad_ng5_g32n

echo "=== CPU ladders (128 ranks/node) ==="
sub cpu  4  512 "$DARS"  120  lad_dars_c512
sub cpu  8 1024 "$DARS"  120  lad_dars_c1024
sub cpu  4  512 "$NG5"   180  lad_ng5_c512
sub cpu 16 2048 "$NG5"   180  lad_ng5_c2048
sub cpu  1   32 "$FARC"  900  lad_farc_c32

echo
squeue -u a270088 -o "%.10i %.14j %.2t" | grep -c m10 | xargs echo "  m10 jobs in queue:"
