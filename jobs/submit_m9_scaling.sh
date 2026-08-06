#!/bin/bash
# M9 P5 — the SCALING fleet: standard vs the two LEAN wide-halo forms, across node counts,
# on both backends, for the report's scaling figures.
#
# Three legs only, which is what the figures need and what keeps each job short enough to
# backfill: standard · 2L (classic form, widened loops) · 4L (divergence form, widened loops).
# The delayed exchange is deliberately ABSENT — it is not a usable scheme and it is being cut
# from the report entirely.
#
# 🔴 16 nodes is the cap (user rule 2026-08-06). Every job here is <= 16 nodes.
# 🔴 Two jobs per point: the step comes from the CLEAN leg and the ice cost from the
#    INSTRUMENTED one, because PHASESTATS perturbs (the A/B job's own protocol).
# 🔴 core2 uses the PRIVATE mesh (L73). farc/dars/ng5 use /pool.
# ⚠️ NG5: the private ng5_w3d copy only carries dist_16 and dist_64, so a curve needs the /pool
#    partition set. That is a different partitioning from the operating-point table's NG5 row.
#
# usage:  submit_m9_scaling.sh gpu|cpu [mesh ...]
set -u
ROOT=/home/a/a270088/port_kokkos_ice
BINDIR=/work/ab0995/a270088/port2/m9/bin/b_94617f1d
L1="standard::"
L2="lean2::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_EVPWIDE_FUSE=1;FESOM_SPEED_EVPWIDE_LEAN=1"
L3="lean4::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_MEVPDIV=1;FESOM_SPEED_EVPWIDE_LEAN=1"

CORE2=/work/ab0995/a270088/port2/mesh/core2
FARC=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5

BACKEND=${1:?gpu or cpu}; shift || true
WANT="${*:-core2 farc dars ng5}"

# mesh  nodes(space-sep)  dt  nsteps  walltime   -- GPU: 4 ranks/node
gpu_rows() { cat <<EOF
core2 $CORE2 1,2,4,8,16 1800 200 00:20:00
farc  $FARC  1,2,4,8,16  900 100 00:20:00
dars  $DARS  2,4,8,16    120  60 00:25:00
ng5   $NG5   2,4,8,16    180  40 00:30:00
EOF
}
# CPU: 128 ranks/node
cpu_rows() { cat <<EOF
core2 $CORE2 1,2,4       1800 100 00:25:00
farc  $FARC  1,2,4,8,16   900 100 00:25:00
dars  $DARS  1,2,4,8,16   120  60 00:35:00
ng5   $NG5   1,2,4,8,16   180  40 00:35:00
EOF
}

if [ "$BACKEND" = gpu ]; then ROWS=$(gpu_rows); RPN=4;  JOB=$ROOT/jobs/job_m9_ab_gpu; BIN=$BINDIR/fesom_port_cuda
else                          ROWS=$(cpu_rows); RPN=128; JOB=$ROOT/jobs/job_m9_ab_cpu; BIN=$BINDIR/fesom_port_serial
fi

n=0
while read -r name mesh nodes dt nsteps wt; do
    [ -z "${name:-}" ] && continue
    case " $WANT " in *" $name "*) ;; *) continue ;; esac
    IFS=',' read -ra NN <<< "$nodes"
    for N in "${NN[@]}"; do
        RANKS=$(( N * RPN ))
        if [ ! -d "$mesh/dist_$RANKS" ]; then
            echo "SKIP $BACKEND $name ${N}N: no dist_$RANKS"; continue
        fi
        for MODE in clean phst; do
            TAG="sc_${BACKEND}_${name}_${N}n"; EXTRA=""
            [ "$MODE" = phst ] && { TAG="${TAG}_phst"; EXTRA=",PHST=1"; }
            if [ "$BACKEND" = gpu ]; then
                sbatch -N"$N" --ntasks="$RANKS" -t "$wt" \
                  --export=ALL,BIN=$BIN,LEG1=$L1,LEG2=$L2,LEG3=$L3,MESH=$mesh,DT=$dt,NSTEPS=$nsteps,TAG=$TAG$EXTRA \
                  "$JOB" >/dev/null && { echo "submitted $TAG"; n=$((n+1)); }
            else
                sbatch -N"$N" --ntasks="$RANKS" -t "$wt" \
                  --export=ALL,BIN=$BIN,LEG1=$L1,LEG2=$L2,LEG3=$L3,MESH=$mesh,DT=$dt,NSTEPS=$nsteps,TAG=$TAG$EXTRA \
                  "$JOB" >/dev/null && { echo "submitted $TAG"; n=$((n+1)); }
            fi
        done
    done
done <<< "$ROWS"
echo "=== $n jobs submitted ($BACKEND) ==="
