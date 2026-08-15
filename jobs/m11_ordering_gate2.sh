#!/bin/bash
#SBATCH --job-name=m11ogat2
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH -N 2
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:25:00
#SBATCH -o /work/ab0995/a270088/port2/m11/ogate2.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/ogate2.%j.err
#
# M11 session 2: the RE-REGISTERED ordering gate leg (R1/R2/R3), with a control ensemble.
#
# What changed against the day-1 leg (job 26852882), and why:
#   * the reference is the SETTLED baseline core2_base, not the shipped core2_m11. At N=256
#     those are different partitions (Finding 10), so day 1 measured ordering PLUS one
#     relocated node and called it pure ordering.
#   * three partition-class controls run through the same instrument as the arms, so every
#     bound is calibrated on measured spread instead of a guess.
#   * two lengths: 1 step (R1, the blind mesh-identity test) and 20 steps (R2 accuracy floor,
#     R3 SSH iterations). R1 is where a renumbering bug would show; R2/R3 are in-class checks.
#
# All twelve legs run inside ONE allocation. Timing is irrelevant here — this leg is never
# used for a step-time number, which is exactly why it may print every step.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')

SB=/work/ab0995/a270088/port2/mesh_m11
BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial}
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
OUT=/work/ab0995/a270088/port2/m11/ogate2.${SLURM_JOB_ID}
mkdir -p "$OUT/s1" "$OUT/s20"

md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = 5c3c90fc0ea3939df86cfbe275287c36 ] || { echo "BIN md5 $md5 not certified h17 Serial"; exit 2; }
echo "=== M11 ordering gate v2  CORE2 $SLURM_NTASKS ranks  BIN md5 $md5  $(date '+%F %T')"

declare -A MESH=(
    [settled]=$SB/core2_base
    [ship]=$SB/core2_m11
    [wgt0]=/work/ab0995/a270088/port2/mesh/core2_wgt0
    [seed]=$SB/core2_seed
    [hil]=$SB/core2_hil
    [rcm]=$SB/core2_rcm
)
LEGS="settled ship wgt0 seed hil rcm"
for t in $LEGS; do
    [ -d "${MESH[$t]}/dist_$SLURM_NTASKS" ] || { echo "REFUSE: ${MESH[$t]}/dist_$SLURM_NTASKS missing"; exit 2; }
done

# --- the arms must carry the reference partition up to the node permutation -----------------
echo
echo "--- pure-ordering precondition: arm partitions == reference partition under the permutation"
$PY "$ROOT/scripts/m11_part_import.py" --from-dist "${MESH[settled]}" --npes "$SLURM_NTASKS" \
    -o "$OUT/vec_settled.txt" --mesh "${MESH[settled]}" >/dev/null || exit 2
for t in hil rcm; do
    $PY "$ROOT/scripts/m11_part_import.py" --from-dist "${MESH[$t]}" --npes "$SLURM_NTASKS" \
        -o "$OUT/vec_$t.txt" --mesh "${MESH[$t]}" >/dev/null || exit 2
done
$PY - "$OUT" "$SB" $LEGS <<'EOF' || { echo "REFUSE: the arms are not pure-ordering arms"; exit 2; }
import sys, numpy as np
out, sb = sys.argv[1], sys.argv[2]
ref = np.loadtxt(f"{out}/vec_settled.txt", dtype=np.int64)
bad = 0
for arm in ("hil", "rcm"):
    p = np.load(f"{sb}/core2_{arm}/m11_perm_node.npy")     # new position -> old index
    v = np.loadtxt(f"{out}/vec_{arm}.txt", dtype=np.int64)
    ok = np.array_equal(v, ref[p])
    print(f"  {arm:<5} part vector under the permutation == reference: {'EXACT' if ok else 'DIFFERS'}")
    bad += 0 if ok else 1
raise SystemExit(1 if bad else 0)
EOF

run() {           # run <tag> <nsteps> <subdir> <printevery>
    local tag=$1
    local nsteps=$2
    local sub=$3
    local pe=$4
    local o="$OUT/$sub/$tag"
    mkdir -p "$o"
    FESOM_PRINT_EVERY=$pe srun "$BIN" "${MESH[$tag]}" "$o" 1800 "$nsteps" -1 "$PHC" 1958 \
        > "$OUT/log_${sub}_${tag}.txt" 2>&1
    local rc=$?
    printf "  %-8s %-4s rc=%-3s %s\n" "$tag" "$sub" "$rc" \
        "$(grep -acE 'identity test \(positive\)' "$OUT/log_${sub}_${tag}.txt" \
           | sed 's/^0$/!! halo gate silent/;s/^[1-9].*/halo gate ok/')"
    grep -aiE "blow ?up|NaN|FATAL" "$OUT/log_${sub}_${tag}.txt" | head -2 | sed 's/^/      !! /'
}

echo
echo "--- R1 legs: 1 step, printing off"
for t in $LEGS; do run "$t" 1 s1 999; done
echo
echo "--- R2/R3 legs: 20 steps, printing every step"
for t in $LEGS; do run "$t" 20 s20 1; done

echo
$PY "$ROOT/scripts/m11_gate2_analyze.py" "$OUT" \
    --ref settled --controls ship,wgt0,seed --arms hil,rcm \
    --perm "hil=$SB/core2_hil,rcm=$SB/core2_rcm"
rc=$?
echo "=== ordering gate v2 rc=$rc   $(date '+%F %T')"
exit $rc
