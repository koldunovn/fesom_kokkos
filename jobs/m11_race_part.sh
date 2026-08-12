#!/bin/bash
#SBATCH --job-name=m11rpart
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:45:00
#SBATCH -o /work/ab0995/a270088/port2/m11/racepart.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/racepart.%j.err
#
# M11 Tasks 11/12 — race PARTITION arms. Same protocol as the ordering race, different axis:
# here the mesh numbering is identical across arms and only the decomposition differs.
#
#   sbatch -N 4 --ntasks=512 --export=ALL,ARMS="base=/path,kahip_anone=/path,..." \
#          jobs/m11_race_part.sh
#
# The job refuses to start unless the five mesh-definition files are md5-identical across every
# arm — otherwise "only the partition differs" is an assumption rather than a fact.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/jobs/m7_provenance.sh"
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')
# Timing runs print once at the start and once at the end — the per-step diagnostic print is
# expensive enough to be a lever of its own (M9 P4b: 41 % of a farc GPU 2N step). Overridable,
# because when an arm dies mid-run the step number is the whole finding.
export FESOM_PRINT_EVERY=${FESOM_PRINT_EVERY:-999}

BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial}
BIN_MD5_EXPECT=5c3c90fc0ea3939df86cfbe275287c36
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
DT=${DT:-1800}; NSTEPS=${NSTEPS:-300}; REPS=${REPS:-2}
YEAR=${YEAR:-1958}
NPES=$SLURM_NTASKS
ARMS=${ARMS:?ARMS="name=meshdir,name=meshdir,..." required}
OUT=/work/ab0995/a270088/port2/m11/racepart.${SLURM_JOB_ID}
mkdir -p "$OUT"

md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = "$BIN_MD5_EXPECT" ] || { echo "BIN md5 $md5 is not certified h17 Serial"; exit 2; }
echo "=== M11 partition race  ranks=$NPES nodes=$SLURM_NNODES  dt=$DT steps=$NSTEPS reps=$REPS"
echo "    BIN=$BIN md5=$md5   $(date '+%F %T')"
m7_provenance "$OUT" "$BIN"

NAMES=""; declare -A MESH
IFS=',' read -ra SPECS <<< "$ARMS"
for spec in "${SPECS[@]}"; do
    n=${spec%%=*}; d=${spec#*=}
    [ -d "$d/dist_$NPES" ] || { echo "REFUSE: $n has no $d/dist_$NPES"; exit 2; }
    MESH[$n]=$d; NAMES="$NAMES $n"
done
echo "    arms:$NAMES"

# Arms must differ ONLY in the partition — unless the race is deliberately a 2x2 over
# numbering AND partition (MIXED_NUMBERING=1), in which case arms are grouped by their mesh
# files and the identity requirement is enforced WITHIN each group. Anything else is a race
# whose arms differ in a way nobody wrote down.
echo "--- mesh-file identity"
declare -A GRP
for n in $NAMES; do
    GRP[$n]=$(for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out; do
                  md5sum "${MESH[$n]}/$f" | cut -d' ' -f1; done | md5sum | cut -c1-8)
done
ngrp=$(for n in $NAMES; do echo "${GRP[$n]}"; done | sort -u | wc -l)
if [ "$ngrp" != 1 ] && [ "${MIXED_NUMBERING:-0}" != 1 ]; then
    echo "REFUSE: the arms carry $ngrp different meshes and MIXED_NUMBERING is not set"
    for n in $NAMES; do printf "    %-16s mesh-group %s\n" "$n" "${GRP[$n]}"; done
    exit 2
fi
echo "    $ngrp mesh group(s) across $(echo $NAMES | wc -w) arms"
for n in $NAMES; do printf "    %-16s mesh-group %s   %s\n" "$n" "${GRP[$n]}" "${MESH[$n]}"; done
for n in $NAMES; do
    printf "    %-14s %s\n" "$n" "${MESH[$n]}"
done

run() {
    local a=$1
    local r=$2
    local o="$OUT/${a}_$r"
    mkdir -p "$o"
    srun "$BIN" "${MESH[$a]}" "$o" "$DT" "$NSTEPS" -1 "$PHC" "$YEAR" > "$OUT/log_${a}_$r.txt" 2>&1
    local rc=$?
    local t
    t=$(grep -a "loop timing" "$OUT/log_${a}_$r.txt" | tail -1 | sed -E 's/.*-> +([0-9.]+) s\/step.*/\1/')
    printf "  %-14s rep%-2s rc=%-3s s/step=%s\n" "$a" "$r" "$rc" "${t:-FAILED}"
    grep -aqE "identity test \(positive\)" "$OUT/log_${a}_$r.txt" || echo "        !! halo gate silent"
    grep -aiE "blow ?up|NaN|FATAL" "$OUT/log_${a}_$r.txt" | head -2 | sed 's/^/        !! /'
    echo "${t:-}" >> "$OUT/times_$a.txt"
}

# interleave, so a drift in node state hits every arm alike
for r in $(seq 1 "$REPS"); do for a in $NAMES; do run "$a" "$r"; done; done

echo
echo "=== min-of-$REPS per arm (s/step), $NPES ranks ==="
python3 - "$OUT" $NAMES <<'EOF'
import sys, os
out, names = sys.argv[1], sys.argv[2:]
res = {}
for a in names:
    p = f"{out}/times_{a}.txt"
    v = [float(x) for x in open(p).read().split() if x] if os.path.exists(p) else []
    res[a] = (min(v), max(v)) if v else (None, None)
b = res[names[0]][0]
print(f"  {'arm':<16}{'min s/step':>12}{'spread':>10}{'vs ' + names[0]:>14}")
for a in names:
    t, mx = res[a]
    if t is None:
        print(f"  {a:<16}   FAILED"); continue
    d = "" if (a == names[0] or not b) else f"{100*(t/b-1):+.2f} %"
    print(f"  {a:<16}{t:>12.4f}{100*(mx/t-1):>9.1f}%{d:>14}")
print("\n  spread = max/min over the reps of that arm; a delta smaller than the spread is noise.")
EOF
echo "=== done $(date '+%F %T') ==="
