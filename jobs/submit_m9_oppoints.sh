#!/bin/bash
# M9 — the four OPERATING POINTS (report Sect. 5.3 table and Figure 1): one GPU point per mesh,
# five legs, clean + instrumented. This is the companion to submit_m9_scaling.sh, which does the
# curves; here the partition is held FIXED and only the scheme changes.
#
# Reconstructed from the legacy fleet's own logs (p6_core2/farc/dars/ng5, jobs 26741184-91),
# which were submitted by hand. The knob strings below are exactly what those runs announced.
#
# 🔴 The delayed exchange (the legacy fleet's sixth leg, `lag8`) is NOT here. It was cut from
#    the report — it imprints the domain decomposition on the ice field — so re-running it under
#    det would be measuring a scheme nobody will use.
#
# 🔴 core2 and ng5 use the PRIVATE mesh copies (L73); farc and dars use /pool.
# 🔴 Two jobs per point: the model step comes from the CLEAN leg and the ice cost from the
#    INSTRUMENTED one (the A/B job's own protocol -- PHASESTATS perturbs).
#
# M13 (2026-08-14): IC=det is the default here now. An operating point holds its partition
# fixed, so both sides of every A/B shared one IC even under the legacy fill and the RATIOS
# were largely protected; what was not protected is the absolute ice cost per mesh and the
# cross-mesh ordering claim, which compares four points that ran four different partitions.
#
# usage:  submit_m9_oppoints.sh [mesh ...]        # det, tags op6_*
#         IC=legacy submit_m9_oppoints.sh [mesh]  # legacy, tags p6l_*
set -u
ROOT=/home/a/a270088/port_kokkos_ice
BINDIR=${BINDIR:-/work/ab0995/a270088/port2/m9/bin/b_det1}
BIN=$BINDIR/fesom_port_cuda
IC=${IC:-det}
PFX=${PFX:-$([ "$IC" = det ] && echo op6 || echo p6l)}

L1="standard::"
L2="wide_k8::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_EVPWIDE_FUSE=1"
L3="wide_k8_lean::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_EVPWIDE_FUSE=1;FESOM_SPEED_EVPWIDE_LEAN=1"
L4="widediv_k8::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_MEVPDIV=1"
L5="widediv_k8_lean::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_MEVPDIV=1;FESOM_SPEED_EVPWIDE_LEAN=1"

CORE2=/work/ab0995/a270088/port2/mesh/core2
FARC=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
NG5=/work/ab0995/a270088/port2/mesh/ng5_w3d

WANT="${*:-core2 farc dars ng5}"

# name  mesh  nodes  dt  nsteps  walltime      (GPU, 4 ranks/node)
# Walltimes from the legacy jobs' own elapsed (5:13 / 4:53 / 16:06 / 29:04 for SIX legs),
# minus the leg we dropped, plus headroom for the det fill.
ROWS=$(cat <<EOF
core2 $CORE2  1 1800 300 00:14:00
farc  $FARC   4  900 100 00:15:00
dars  $DARS   8  120 100 00:25:00
ng5   $NG5   16  180  40 00:35:00
EOF
)

n=0
while read -r name mesh N dt nsteps wt; do
    [ -z "${name:-}" ] && continue
    case " $WANT " in *" $name "*) ;; *) continue ;; esac
    RANKS=$(( N * 4 ))
    [ -d "$mesh/dist_$RANKS" ] || { echo "SKIP $name ${N}N: no $mesh/dist_$RANKS"; continue; }
    for MODE in clean phst; do
        TAG="${PFX}_${name}"; EXTRA=""
        [ "$MODE" = phst ] && { TAG="${TAG}_phst"; EXTRA=",PHST=1"; }
        [ "$IC" = det ] && EXTRA="$EXTRA,FESOM_IC_EXTRAP=det"
        sbatch -N"$N" --ntasks="$RANKS" -t "$wt" \
          --export=ALL,BIN=$BIN,LEG1=$L1,LEG2=$L2,LEG3=$L3,LEG4=$L4,LEG5=$L5,MESH=$mesh,DT=$dt,NSTEPS=$nsteps,TAG=$TAG$EXTRA \
          "$ROOT/jobs/job_m9_ab_gpu" >/dev/null && { echo "submitted $TAG"; n=$((n+1)); }
    done
done <<< "$ROWS"
echo "=== $n jobs submitted (IC=$IC, prefix $PFX, bin $BINDIR) ==="
