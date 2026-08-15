#!/bin/bash
#SBATCH --job-name=m11bdist
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/bdist.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/bdist.%j.err
#
# M11 Task 8, second half — turn the shortlisted ENGINE part vectors into real `dist_N`
# directories, score them with the dist files present (which is where element replication and
# halo counts come from), and prove one of them runs.
#
#   sbatch --export=ALL,ARMS="kahip_a100 kahip_anone mtkahypar_a0",K=512 jobs/m11_zoo_b_dists.sh
#
# The injection goes through `FESOM_PART_FILE`, whose byte-identity null is verified (Task 4),
# so what the model reads is exactly the engine's partition — modulo `check_partitioning`, whose
# moves are counted and printed for every arm rather than assumed to be zero.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
ENG=/work/ab0995/a270088/port2/m11/engines
SRC=${SRC:-$SB/core2_base}
MESHTAG=${MESHTAG:-core2}
K=${K:-512}
ARMS=${ARMS:-"kahip_a100 kahip_anone mtkahypar_a0"}
SMOKE=${SMOKE:-1}
OUT=/work/ab0995/a270088/port2/m11/bdist.${SLURM_JOB_ID:-manual}
CSV=$OUT/zoo_b_dists_${MESHTAG}_k$K.csv
mkdir -p "$OUT"
fail=0

echo "=== M11 zoo B dists  mesh=$SRC  k=$K  arms='$ARMS'  $(date '+%F %T')"
for arm in $ARMS; do
    V=$ENG/$(basename "$SRC")_${arm}_k$K.part
    [ -s "$V" ] || V=$ENG/$(basename "$SRC")_${arm}_e0.03_k$K.part
    [ -s "$V" ] || { echo "  $arm: no part vector ($V)"; fail=1; continue; }
    M=$SB/zoo/$MESHTAG/b_$arm
    echo
    echo "=================================================================="
    echo "=== $arm   vector $V"
    [ -f "$M/nod2d.out" ] || { rm -rf "$M"; m11_sandbox_copy_mesh "$SRC" "zoo/$MESHTAG/b_$arm" || { fail=1; continue; }; }
    rm -rf "$M/dist_$K"
    env MESH="$M" RANKS="$K" TAG="b_${arm}" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
        BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_FILE="$V" \
        bash "$ROOT/scripts/m11_partgen.sh" > "$OUT/gen_$arm.log" 2>&1
    rc=$?
    L="$OUT/meshpart_b_${arm}_$K.log"
    moved=$(grep -ac 'is moved to part' "$L" 2>/dev/null || echo 0)
    if [ $rc -ne 0 ]; then
        echo "  generation FAILED rc=$rc"; tail -5 "$OUT/gen_$arm.log"; fail=1; continue
    fi
    echo "  dist_$K written; check_partitioning moved $moved node(s)"
    $PY "$ROOT/scripts/m11_scorecard.py" "$M" --dist "$K" --arm "${arm}_$K" --csv "$CSV" \
        2>&1 | sed -n '/balance\|cut  \|comm volume\|contiguity\|halo\/repl\|gates/p' | sed 's/^/    /'
done

if [ "$SMOKE" = 1 ]; then
    first=$(echo $ARMS | awk '{print $1}')
    M=$SB/zoo/$MESHTAG/b_$first
    echo
    echo "=================================================================="
    echo "=== Serial smoke + halo/dist correctness gate on $first"
    set +u; source /sw/etc/profile.levante; source "$ROOT/env.sh"; set -u
    ulimit -s 204800
    while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')
    export FESOM_PRINT_EVERY=999
    BIN=/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial
    PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
    # the smoke runs at the allocation's rank count, so score/generate that count too
    if [ ! -d "$M/dist_$SLURM_NTASKS" ]; then
        V=$ENG/$(basename "$SRC")_${first}_k$SLURM_NTASKS.part
        [ -s "$V" ] || V=$ENG/$(basename "$SRC")_${first}_e0.03_k$SLURM_NTASKS.part
        if [ -s "$V" ]; then
            env MESH="$M" RANKS="$SLURM_NTASKS" TAG="b_${first}_smoke" ROOT="$ROOT" OUT="$OUT" \
                PARTROOT="$P" BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_FILE="$V" \
                bash "$ROOT/scripts/m11_partgen.sh" > "$OUT/gen_smoke.log" 2>&1 || fail=1
        else
            echo "  no $SLURM_NTASKS-rank vector for $first — smoke skipped"
        fi
    fi
    if [ -d "$M/dist_$SLURM_NTASKS" ]; then
        mkdir -p "$OUT/smoke"
        srun "$BIN" "$M" "$OUT/smoke" 1800 20 -1 "$PHC" 1958 > "$OUT/smoke.log" 2>&1
        rc=$?
        echo "  rc=$rc"
        grep -aE "identity test" "$OUT/smoke.log" | head -2 | sed 's/^/    /'
        grep -aiE "blow ?up|NaN|FATAL" "$OUT/smoke.log" | head -3 | sed 's/^/    !! /'
        [ $rc -eq 0 ] || fail=1
    fi
fi

m11_check_sources
echo; echo "=== zoo B dists: $([ $fail = 0 ] && echo PASS || echo FAIL)   csv: $CSV"
exit $fail
