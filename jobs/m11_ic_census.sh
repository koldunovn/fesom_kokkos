#!/bin/bash
#SBATCH --job-name=m11iccen
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:25:00
#SBATCH -o /work/ab0995/a270088/port2/m11/iccen.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/iccen.%j.err
#
# M11 × M13 — HOW CONTAMINATED WAS THIS MESH?
#
# M13 showed the climatology hole-filler produces a partition-dependent initial condition, and
# that this, not any solver, is what killed partitions (docs/CG_BLOWUPS_M13.md). Before re-running
# the campaign we need the size of the effect PER MESH, because it decides what must be re-run:
# a mesh whose initial condition never depended on the decomposition has uncontaminated results.
#
# The measurement is direct and needs no model integration: write the step-0 snapshot — which IS
# the initial condition after the fill — for several partitions of the same mesh, under the legacy
# fill and under det, and difference them.
#
#   legacy legs differ  -> every legacy result on this mesh compared different problems
#   det legs identical  -> the fix works at this mesh and rank count
#
# snap_000001 is also compared: it separates "the initial state differs" from "and one step has
# already amplified it".
#
#   ARMS="base=/path,arm=/path,ctl=/path" sbatch -N 16 --ntasks=2048 jobs/m11_ic_census.sh
#
# Ladder dt per mesh (CORE2 1800 · fArc 900 · dars 120 · NG5 180) — one step only, but a cold
# start at the wrong dt is how this campaign lost three findings (L110), so assert it anyway.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
export OMPI_MCA_btl_vader_single_copy_mechanism=none    # L18 deterministic gather
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')
unset FESOM_IC_EXTRAP FESOM_IC_EXTRAP_TOL

BIN=${BIN:-/work/ab0995/a270088/port2/m11/bin/det1/fesom_port_serial}
BIN_MD5_EXPECT=${BIN_MD5_EXPECT:-07d982e0757e8e258a479485840d897c}
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
DT=${DT:?DT required — the mesh cold-start ladder dt}
NPES=$SLURM_NTASKS
ARMS=${ARMS:?ARMS="name=meshdir,..." required}
TAG=${TAG:-$SLURM_JOB_ID}
OUT=/work/ab0995/a270088/port2/m11/iccen.${SLURM_JOB_ID}
mkdir -p "$OUT"

md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = "$BIN_MD5_EXPECT" ] || { echo "BIN md5 $md5 not the pinned binary"; exit 2; }
echo "=== M11 IC census  tag=$TAG  ranks=$NPES nodes=$SLURM_NNODES  dt=$DT  $(date '+%F %T')"
echo "    BIN=$BIN md5=$md5"

NAMES=""; declare -A MESH
IFS=',' read -ra SPECS <<< "$ARMS"
for spec in "${SPECS[@]}"; do
    n=${spec%%=*}; d=${spec#*=}
    [ -d "$d/dist_$NPES" ] || { echo "REFUSE: $n has no $d/dist_$NPES"; exit 2; }
    MESH[$n]=$d; NAMES="$NAMES $n"
done
for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out; do
    u=$(for n in $NAMES; do md5sum "${MESH[$n]}/$f" | cut -d' ' -f1; done | sort -u | wc -l)
    [ "$u" = 1 ] || { echo "REFUSE: $f differs across the legs — not one mesh"; exit 2; }
done
echo "    legs:$NAMES   (five mesh files md5-identical)"

# MODES lets a re-run cover only the half that is missing. The det legs cost several minutes of
# fill and relaxation each on dars/NG5, so a job sized for the legacy legs will not hold both.
MODES=${MODES:-"legacy det"}
echo "    modes: $MODES"
for mode in $MODES; do
    for n in $NAMES; do
        o="$OUT/${mode}_$n"; mkdir -p "$o"
        FESOM_IC_EXTRAP=$mode srun "$BIN" "${MESH[$n]}" "$o" "$DT" 1 1 "$PHC" 1958 \
            > "$OUT/log_${mode}_$n.txt" 2>&1
        rc=$?
        ann=$(grep -acq "FESOM_IC_EXTRAP=det" "$OUT/log_${mode}_$n.txt" && echo yes || echo no)
        printf "  %-8s %-16s rc=%-3s det-announced=%s\n" "$mode" "$n" "$rc" "$ann"
    done
done

REF=$(echo $NAMES | awk '{print $1}')
for mode in $MODES; do
    for snap in snap_000000.nc snap_000001.nc; do
        echo
        echo "=== $mode / $snap — every leg vs $REF ($([ "$snap" = snap_000000.nc ] \
             && echo 'the initial condition itself' || echo 'after one step'))"
        for n in $NAMES; do
            [ "$n" = "$REF" ] && continue
            echo "  --- $n vs $REF"
            # rc must be captured off the python, NOT off a pipeline (a `| sed` would hand back
            # sed's exit status and report every comparison as identical)
            d="$OUT/diff_${mode}_${n}_${snap%.nc}.txt"
            "$PY" "$ROOT/scripts/diff_snap.py" --pattern "$snap" \
                  "$OUT/${mode}_$REF" "$OUT/${mode}_$n" > "$d" 2>&1
            rc=$?
            grep -avE "getfattr|^Pre  SHA|^Post SHA|^Comparing" "$d" | sed 's/^/    /'
            echo "    diff_snap rc=$rc (0 = bit-identical)"
        done
    done
done
echo
echo "=== done $(date '+%F %T') ==="
