#!/bin/bash
#SBATCH --job-name=m11tpw
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/tpw.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/tpw.%j.err
#
# M11 — shrink the MAXIMUM halo by giving the expensive ranks less owned work.
#
# The GPU step waits for the slowest rank, and on every large-mesh GPU point measured the owned
# work is balanced to 1-2 % while the halo spreads 2.7-2.9x (Finding 25). Partitioners minimise
# a SUM, so no objective they offer targets the max. `tpwgts` does: it sets a per-part target
# share of the total vertex weight, so a rank whose halo is expensive can be given fewer nodes.
#
# Why this is more than a bookkeeping trick, and why my first-order estimate of it was too
# pessimistic: shrinking a part shrinks its BOUNDARY too, roughly as sqrt(area). So reducing a
# high-halo part's share attacks the halo twice — less owned work AND a shorter boundary — and
# the fixed-point iteration is what captures the second effect. The one-shot estimate in
# `m11_tpwgts.py` holds the halo fixed and therefore reports a lower bound.
#
#   sbatch --export=ALL,SRC=<mesh>,MESHTAG=farc,K=16,KAPPA=41,ITERS=4 jobs/m11_tpwgts_loop.sh
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
SRC=${SRC:-$SB/farc_m11}
MESHTAG=${MESHTAG:-farc}
K=${K:-16}
KAPPA=${KAPPA:-41}
ITERS=${ITERS:-4}
DAMPING=${DAMPING:-0.7}
M=$SB/zoo/$MESHTAG/tpw
OUT=/work/ab0995/a270088/port2/m11/tpw.${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"

echo "=== M11 tpwgts loop  mesh=$SRC  k=$K  kappa=$KAPPA  iters=$ITERS  damping=$DAMPING"
[ -f "$M/nod2d.out" ] || { rm -rf "$M"; m11_sandbox_copy_mesh "$SRC" "zoo/$MESHTAG/tpw" || exit 2; }

halo_stats() {   # halo_stats <meshdir> <k> -> "mean max max/mean"
    $PY - "$1" "$2" <<'EOF'
import sys, numpy as np
sys.path.insert(0, "/home/a/a270088/port_kokkos_part/scripts")
from m11_scorecard import load_dist_files
d = load_dist_files(sys.argv[1], int(sys.argv[2]))
h = np.array([x["eDim_nod2D"] for x in d["my_list"]], float)
print(f"{h.mean():.0f} {h.max():.0f} {h.max()/h.mean():.3f}")
EOF
}

# 🔴 The starting partition matters, and my first run got it wrong. Starting from the CPU
# winner (w=100+nlev at 3 % slack) means starting from a partition whose OWNED work is
# deliberately unbalanced (3-D max/min 4-7). The cost model then sees that owned imbalance as
# the thing to correct, drives the shares to 0.016..0.216 against a uniform 0.0625, and the loop
# diverges (max halo 887 -> 835 -> 954 -> 919 -> 941). For the GPU case the halo is the ONLY
# imbalance, so the loop must start from a partition whose owned work is already balanced.
BASEKNOBS=${BASEKNOBS:-"FESOM_PART_UFACTOR=1"}

# iteration 0: the base arm, with no target shares
rm -rf "$M/dist_$K"
# shellcheck disable=SC2086
env $BASEKNOBS MESH="$M" RANKS="$K" TAG="tpw_it0" ROOT="$ROOT" OUT="$OUT" \
    PARTROOT="$P" BIN="$P/mesh_part/build/bin/fesom_meshpart" \
    bash "$ROOT/scripts/m11_partgen.sh" > "$OUT/gen_it0.log" 2>&1 || { echo "it0 FAILED"; exit 3; }
echo "  it 0 (no tpwgts): halo mean/max/ratio = $(halo_stats "$M" "$K")"
best=$(halo_stats "$M" "$K" | awk '{print $2}')

for it in $(seq 1 "$ITERS"); do
    $PY "$ROOT/scripts/m11_tpwgts.py" "$M" --npes "$K" --kappa "$KAPPA" \
        --damping "$DAMPING" -o "$OUT/tpwgts_$it.txt" | sed 's/^/    /'
    rm -rf "$M/dist_$K"
    # shellcheck disable=SC2086
    env $BASEKNOBS FESOM_PART_TPWGTS_FILE="$OUT/tpwgts_$it.txt" \
        MESH="$M" RANKS="$K" TAG="tpw_it$it" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
        BIN="$P/mesh_part/build/bin/fesom_meshpart" \
        bash "$ROOT/scripts/m11_partgen.sh" > "$OUT/gen_it$it.log" 2>&1 \
        || { echo "  it $it FAILED"; sed -n '/REFUSE\|\[M11\]/p' "$OUT/gen_it$it.log" | head -5; break; }
    s=$(halo_stats "$M" "$K")
    echo "  it $it: halo mean/max/ratio = $s"
    cp -r "$M/dist_$K" "$OUT/dist_${K}_it$it"
done

echo
echo "--- scorecard of the final iterate"
$PY "$ROOT/scripts/m11_scorecard.py" "$M" --dist "$K" --arm "tpw_$K" --csv "$OUT/tpw.csv" \
    2>&1 | sed -n '/balance\|cut  \|comm volume\|halo\/repl\|gates/p' | sed 's/^/    /'
m11_check_sources
echo "=== tpwgts loop done. Iterate-0 max halo was $best; compare above."
