#!/bin/bash
#SBATCH --job-name=m11null
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /work/ab0995/a270088/port2/m11/null.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/null.%j.err
#
# M11 Task 4 — the two byte-identity nulls, both on the METIS 5.1.0 build (review B2).
#
#   null-1  patched binary with NO knobs set  ==  pristine binary, byte-for-byte dist_8.
#           If this fails, every arm the patched tool ever produces is suspect, because the
#           patch itself moved the baseline.
#   null-2  the injection path is transparent: feed a partition's OWN vector back through
#           FESOM_PART_FILE and the dist must come out byte-identical. This is what licenses
#           an external engine to own the decomposition while the Fortran tool still writes
#           the files.
#
# Both legs are METIS 5.1.0. The 5.2.1 build is a separate arm (A0) whose output is NOT
# expected to match and is scored and raced rather than asserted away.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"

SB=/work/ab0995/a270088/port2/mesh_m11
REF=/work/ab0995/a270088/port2/partm11/fesom2_ref
PM11=/work/ab0995/a270088/port2/partm11/fesom2
OUT=/work/ab0995/a270088/port2/m11/null.${SLURM_JOB_ID:-manual}
N=${N:-8}
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
mkdir -p "$OUT"

run_leg() {   # run_leg <tag> <partroot> <meshdir> [env assignments...]
    local tag=$1 partroot=$2 mesh=$3; shift 3
    echo; echo "############ leg $tag"
    env "$@" MESH="$mesh" RANKS="$N" TAG="$tag" PARTROOT="$partroot" \
        BIN="$partroot/mesh_part/build/bin/fesom_meshpart" OUT="$OUT" ROOT="$ROOT" \
        bash "$ROOT/scripts/m11_partgen.sh"
}

fail=0
run_leg ref  "$REF"  "$SB/null_ref" || fail=1
run_leg a    "$PM11" "$SB/null_a"   || fail=1

echo; echo "############ null-1: patched (no knobs) vs pristine, dist_$N byte-for-byte"
if diff -r "$SB/null_ref/dist_$N" "$SB/null_a/dist_$N" > "$OUT/null1.diff" 2>&1; then
    echo "  NULL-1 PASS: $(ls "$SB/null_a/dist_$N" | wc -l) files identical"
else
    echo "  NULL-1 FAIL:"; head -20 "$OUT/null1.diff"; fail=1
fi

echo; echo "############ null-2: re-inject the partition's own vector"
$PY "$ROOT/scripts/m11_part_import.py" --from-dist "$SB/null_a" --npes "$N" \
    -o "$OUT/inject_$N.txt" --mesh "$SB/null_a" || fail=1
run_leg inj "$PM11" "$SB/null_inj" FESOM_PART_FILE="$OUT/inject_$N.txt" || fail=1

moved=$(grep -ac "Isolated node" "$OUT/meshpart_inj_$N.log" || true)
echo "  check_partitioning moves on the injected leg: $moved (must be 0 for byte identity)"
if diff -r "$SB/null_a/dist_$N" "$SB/null_inj/dist_$N" > "$OUT/null2.diff" 2>&1; then
    echo "  NULL-2 PASS: injection is transparent, dist_$N byte-identical"
else
    echo "  NULL-2 FAIL:"; head -20 "$OUT/null2.diff"; fail=1
fi

echo; echo "############ graph + in-memory levels dump (exporter cross-check, review M5)"
env MESH="$SB/null_a" RANKS="$N" TAG=dump PARTROOT="$PM11" OUT="$OUT" ROOT="$ROOT" \
    BIN="$PM11/mesh_part/build/bin/fesom_meshpart" \
    FESOM_PART_GRAPH_DUMP="$OUT/core2_csr.txt" bash "$ROOT/scripts/m11_partgen.sh" \
    > "$OUT/dump_leg.log" 2>&1 || fail=1
grep -aE "^\[M11\]" "$OUT/dump_leg.log" | sed 's/^/  /'

echo; echo "############ M11 partitioner nulls: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
