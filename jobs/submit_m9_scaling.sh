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
#    (M13 does NOT lift this: the two NG5 columns differ in which partition FILES exist, not in
#    whether the model can run them. Under det they at least start from the same ocean.)
#
# 🔴 M13 (2026-08-14). Every point on a scaling curve uses a DIFFERENT partition, and under the
#    legacy cold-start hole fill the initial condition is a function of the partition — so the
#    original fleet compared points that were integrating measurably different oceans. IC=det
#    (the default here now) removes that: all points share one bitwise-identical IC.
#
# usage:  submit_m9_scaling.sh gpu|cpu [mesh ...]        # det fleet, tags scd_*
#         IC=legacy submit_m9_scaling.sh gpu|cpu [mesh]  # reproduce the old fleet, tags sc_*
set -u
ROOT=/home/a/a270088/port_kokkos_ice
# b_det1 = m9-mevp-double tip (41da460), the first M9 bin carrying FESOM_IC_EXTRAP.
# The original fleet ran on b_94617f1d, which predates the knob; keep that name here if you
# ever need to reproduce the legacy curves.
BINDIR=${BINDIR:-/work/ab0995/a270088/port2/m9/bin/b_det1}
# M13 cold-start hole fill. det = deterministic, partition-INDEPENDENT (every point on a
# scaling curve then integrates the same ocean); legacy = the first-fill-wins Gauss-Seidel
# the original fleet ran under, kept reachable for before/after comparison.
IC=${IC:-det}
PFX=${PFX:-$([ "$IC" = det ] && echo scd || echo sc)}
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
# ⚠️ Walltimes carry headroom for the det fill, which is INIT cost, not step cost: the
#    relaxation sweeps run before step 1 and every job pays them 6 times (3 legs x 2 reps).
#    They do not touch s/step, so the measurement is unaffected — only the wall clock is.
#    Measured on fArc (26961292/3): 11804 fill + 67615/103030 relax sweeps for T and S, a few
#    seconds at 288 and at 1152 ranks. +5 min over the legacy fleet's walltimes is ample.
gpu_rows() { cat <<EOF
core2 $CORE2 1,2,4,8,16 1800 200 00:25:00
farc  $FARC  1,2,4,8,16  900 100 00:25:00
dars  $DARS  2,4,8,16    120  60 00:30:00
ng5   $NG5   2,4,8,16    180  40 00:35:00
EOF
}
# CPU: 128 ranks/node
cpu_rows() { cat <<EOF
core2 $CORE2 1,2,4       1800 100 00:30:00
farc  $FARC  1,2,4,8,16   900 100 00:30:00
dars  $DARS  1,2,4,8,16   120  60 00:40:00
ng5   $NG5   1,2,4,8,16   180  40 00:40:00
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
            # 🔴 The tag prefix separates the det fleet from the legacy one. The A/B job does
            #    `rm -rf $OUT` on entry, so re-using sc_* here would delete the curves the det
            #    fleet is supposed to be compared against.
            TAG="${PFX}_${BACKEND}_${name}_${N}n"; EXTRA=""
            [ "$MODE" = phst ] && { TAG="${TAG}_phst"; EXTRA=",PHST=1"; }
            [ "$IC" = det ] && EXTRA="$EXTRA,FESOM_IC_EXTRAP=det"
            sbatch -N"$N" --ntasks="$RANKS" -t "$wt" \
              --export=ALL,BIN=$BIN,LEG1=$L1,LEG2=$L2,LEG3=$L3,MESH=$mesh,DT=$dt,NSTEPS=$nsteps,TAG=$TAG$EXTRA \
              "$JOB" >/dev/null && { echo "submitted $TAG"; n=$((n+1)); }
        done
    done
done <<< "$ROWS"
echo "=== $n jobs submitted ($BACKEND, IC=$IC, prefix $PFX, bin $BINDIR) ==="
