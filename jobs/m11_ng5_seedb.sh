#!/bin/bash
#SBATCH --job-name=m11ng5sb
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH -o /work/ab0995/a270088/port2/m11/ng5seedb.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/ng5seedb.%j.err
#
# M11 — is the NG5/2048 MINCONN failure a property of the KNOB or of one pathological part?
#
# `a4m` (MINCONN) and `a4u30` (MINCONN+CONTIG+slack) both abort at the first CG iteration with a
# NaN right-hand side, while `base` and `a5_u30` (slack, no MINCONN) run. No offline metric we
# have separates them: owned-node imbalance is 1.61/1.66 for the failures and 1.66 for the
# survivor, no rank owns zero nodes, and the failing partition is the CLEANER one on every
# fragmentation count. So the discriminating experiment is a re-roll: same knobs, different seed.
#
#   different seed also fails  -> MINCONN systematically produces something FESOM cannot set up
#   different seed runs        -> one pathological part, i.e. a lottery, which is worse for
#                                 shipping than a systematic failure and makes the target-rank
#                                 smoke run mandatory rather than advisory
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
SRC=$SB/ng5_m11
RANKS=${RANKS:-2048}
SEED=${SEED:-424242}
ARM=${ARM:-a4m_seedb}
M=$SB/zoo/ng5/$ARM
OUT=/work/ab0995/a270088/port2/m11/ng5seedb.${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"

echo "=== M11 NG5 MINCONN re-roll  arm=$ARM seed=$SEED ranks=$RANKS  $(date '+%F %T')"
[ -f "$M/nod2d.out" ] || { rm -rf "$M"; m11_sandbox_copy_mesh "$SRC" "zoo/ng5/$ARM" || exit 2; }

env FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=100 \
    FESOM_PART_MINCONN=1 FESOM_PART_SEED="$SEED" \
    MESH="$M" RANKS="$RANKS" TAG="${ARM}" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
    BIN="$P/mesh_part/build/bin/fesom_meshpart" \
    bash "$ROOT/scripts/m11_partgen.sh" 2>&1 | tail -25

for N in $RANKS; do
    [ -d "$M/dist_$N" ] || { echo "  dist_$N was not produced"; exit 3; }
    echo "--- scorecard $ARM N=$N"
    $PY "$ROOT/scripts/m11_scorecard.py" "$M" --dist "$N" --arm "${ARM}_$N" \
        --csv "$OUT/${ARM}.csv" 2>&1 | tail -30
done
m11_check_sources
echo "=== done $(date '+%F %T').  Smoke this arm at $RANKS ranks next — that is the only test"
echo "    that has separated a shippable partition from an un-startable one on NG5."
