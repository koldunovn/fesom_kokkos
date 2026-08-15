#!/bin/bash
#SBATCH --job-name=m11fixiso
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/fixiso.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/fixiso.%j.err
#
# M11 Finding 9, fixed: `check_partitioning` seeds its candidate list with the FIRST neighbour
# unconditionally, while the loop that fills the rest of the list skips neighbours in the
# node's own partition. When the adjacency list happens to begin with the node's own partition,
# that partition becomes a move candidate, wins, and the node is "moved" to where it already
# was — silently, and depending only on the NUMBERING. (It also counts the first neighbour
# twice, biasing the choice towards it.)
#
# `FESOM_PART_FIXISO=1` builds the candidate list from neighbours OUTSIDE part(n) only.
# Default off, so the tool still reproduces upstream byte-for-byte with no knobs set.
#
# Four legs:
#   A  null re-verify — the rebuilt binary with NO knobs must still reproduce the reference
#      dist_8 byte-for-byte. The Fortran edit lands in the EXECUTABLE (the C knobs land in
#      libfesom_meshpart_C.so), so this re-establishes provenance for a changed executable.
#   B  reproduce the bug with the knob OFF, then watch it disappear with the knob ON, on the
#      exact case Finding 9 names (CORE2 864, the node in partition 356).
#   C  settle the SHIPPED dist_864 to a fixed point with the fix on.
#   D  rebuild the two ordering arms at 864 and run the invariant-block equality gate — the
#      gate that dropped 864 from the race in the first place.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
SHIP=$SB/core2_m11
BASE=$SB/core2_base
N=${N:-864}
OUT=/work/ab0995/a270088/port2/m11/fixiso.${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"
fail=0

gen() {   # gen <mesh> <tag> <N> <partfile-or-empty>
    local mesh=$1
    local tag=$2
    local n=$3
    local pf=${4:-}
    if [ -n "$pf" ]; then
        env MESH="$mesh" RANKS="$n" TAG="$tag" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
            BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_FILE="$pf" \
            bash "$ROOT/scripts/m11_partgen.sh"
    else
        env MESH="$mesh" RANKS="$n" TAG="$tag" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
            BIN="$P/mesh_part/build/bin/fesom_meshpart" \
            bash "$ROOT/scripts/m11_partgen.sh"
    fi
}
iso() { $PY "$ROOT/scripts/m11_scorecard.py" "$1" --dist "$2" --no-dist-files 2>/dev/null \
        | sed -n 's/.*isolated (<=1 same-part nb) \([0-9]*\).*/\1/p'; }

echo "=================================================================="
echo "=== leg A: null re-verify, rebuilt executable, no knobs"
echo "    executable md5 $(md5sum "$P/mesh_part/build/bin/fesom_meshpart" | cut -d' ' -f1)"
echo "    libC       md5 $(md5sum "$P/mesh_part/build/lib64/libfesom_meshpart_C.so" | cut -d' ' -f1)"
rm -rf "$SB/nullfix"
m11_sandbox_copy_mesh "$SB/null_a" nullfix || exit 2
rm -rf "$SB/nullfix/dist_8"
( unset FESOM_PART_FIXISO; gen "$SB/nullfix" nullfix 8 ) || fail=1
if diff -r "$SB/null_a/dist_8" "$SB/nullfix/dist_8" > "$OUT/nullfix.diff" 2>&1; then
    echo "    null-1 re-verify: dist_8 BYTE-IDENTICAL to the certified reference ($(ls "$SB/null_a/dist_8" | wc -l) files)"
else
    echo "    null-1 re-verify: FAILED"; head -5 "$OUT/nullfix.diff"; fail=1
fi

echo
echo "=================================================================="
echo "=== leg B: the Finding-9 case, knob OFF then ON  (CORE2 dist_$N)"
$PY "$ROOT/scripts/m11_part_import.py" --from-dist "$SHIP" --npes "$N" \
    -o "$OUT/ship_$N.txt" --mesh "$SHIP" >/dev/null || exit 3
for mode in off on; do
    rm -rf "$SB/fixiso_$mode"
    m11_sandbox_copy_mesh "$BASE" "fixiso_$mode" || exit 2
    if [ $mode = on ]; then export FESOM_PART_FIXISO=1; else unset FESOM_PART_FIXISO; fi
    gen "$SB/fixiso_$mode" "iso_$mode" "$N" "$OUT/ship_$N.txt" > "$OUT/gen_iso_$mode.log" 2>&1 \
        || { echo "    generation FAILED ($mode)"; fail=1; continue; }
    L="$OUT/meshpart_iso_${mode}_$N.log"
    echo "--- knob $mode: flagged $(grep -ac 'Isolated node' "$L") | moved $(grep -ac 'is moved to part' "$L")" \
         "| left in place $(grep -ac 'no neighbouring partition' "$L")"
    echo "    residual isolated_nodes in the written dist: $(iso "$SB/fixiso_$mode" "$N")"
    grep -a -A2 'Isolated node.*in partition *356' "$L" | head -6 | sed 's/^/      /'
done
unset FESOM_PART_FIXISO

echo
echo "=================================================================="
echo "=== leg C: settle the shipped dist_$N to a fixed point WITH the fix"
# The convergence criterion is IDEMPOTENCE — a pass that moves nothing — and NOT
# `isolated_nodes == 0`. CORE2 864 contains a node with exactly one neighbour in each of two
# other partitions: no move can give it two same-partition neighbours, so the heuristic
# correctly declines to move it and `isolated_nodes` stays at 1 forever. Asserting 0 there
# fails a partition that is a perfectly good fixed point (my first version of this leg did).
export FESOM_PART_FIXISO=1
rm -rf "$BASE/dist_$N"
moved=1
cp "$OUT/ship_$N.txt" "$OUT/v_0.txt"
for it in 1 2 3 4 5; do
    gen "$BASE" "settle$it" "$N" "$OUT/v_$((it-1)).txt" > "$OUT/gen_settle$it.log" 2>&1 || { fail=1; break; }
    L="$OUT/meshpart_settle${it}_$N.log"
    moved=$(grep -ac 'is moved to part' "$L")
    echo "--- pass $it: flagged $(grep -ac 'Isolated node' "$L"), moved $moved"\
         "-> isolated_nodes now $(iso "$BASE" "$N")"
    [ "$moved" = "0" ] && { echo "    FIXED POINT after $it pass(es) (a pass that moves nothing)"; break; }
    $PY "$ROOT/scripts/m11_part_import.py" --from-dist "$BASE" --npes "$N" \
        -o "$OUT/v_$it.txt" --mesh "$BASE" >/dev/null || { fail=1; break; }
done
[ "$moved" = "0" ] || { echo "    REFUSE: still moving nodes after 5 passes"; fail=1; }

echo
echo "=================================================================="
echo "=== leg D: rebuild the ordering arms at N=$N and gate them"
for arm in hil rcm; do
    $PY "$ROOT/scripts/m11_renumber.py" --permute-labels "$SB/core2_$arm" \
        --from-dist "$BASE" --npes "$N" -o "$OUT/part_${arm}.txt" > "$OUT/perm_$arm.log" 2>&1 \
        || { echo "  $arm: label permutation FAILED"; fail=1; continue; }
    rm -rf "$SB/core2_$arm/dist_$N"
    gen "$SB/core2_$arm" "${arm}_$N" "$N" "$OUT/part_${arm}.txt" > "$OUT/gen_$arm.log" 2>&1 || fail=1
    L="$OUT/meshpart_${arm}_${N}_$N.log"
    [ -f "$L" ] || L="$OUT/meshpart_${arm}_$N.log"
    echo "  $arm: post-pass flagged $(grep -ac 'Isolated node' "$L" 2>/dev/null || echo ?)," \
         "moved $(grep -ac 'is moved to part' "$L" 2>/dev/null || echo ?)"
done
unset FESOM_PART_FIXISO

$PY "$ROOT/scripts/m11_pure_ordering_check.py" --npes "$N" --base "$BASE" \
    --arm "hil=$SB/core2_hil" --arm "rcm=$SB/core2_rcm" || fail=1

CSV=$OUT/fixiso_$N.csv
$PY "$ROOT/scripts/m11_scorecard.py" "$BASE"         --dist "$N" --arm "base_$N" --csv "$CSV" >/dev/null 2>&1
$PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_hil" --dist "$N" --arm "hil_$N"  --csv "$CSV" >/dev/null 2>&1
$PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_rcm" --dist "$N" --arm "rcm_$N"  --csv "$CSV" >/dev/null 2>&1
$PY - "$CSV" <<'EOF' || fail=1
import csv, sys
rows = {r["arm"].rsplit("_", 1)[0]: r for r in csv.DictReader(open(sys.argv[1]))}
INV = ["n2d_max","n2d_imb","n3d_max","n3d_imb","n3d_maxmin","w0_imb","w0_sum","w100_imb",
       "w100_sum","edgecut_unweighted","cutweight_nlev","edges_total","mean_edge_weight",
       "commvol_total","commvol_max_rank","commvol_imb","boundary_nodes","nbr_max","nbr_mean",
       "wet_components","parts_disconnected","components_total","components_max",
       "noncore_vertices","singleton_vertices","isolated_nodes","halo_nod_mean","halo_nod_max",
       "elem_repl","edge_repl"]
b = rows.get("base"); bad = 0
if not b:
    print("  no baseline row"); raise SystemExit(1)
print(f"  base isolated_nodes={b['isolated_nodes']} cut_unw={int(b['edgecut_unweighted']):,} "
      f"halo/rank={float(b['halo_nod_mean']):.1f}")
for a in ("hil", "rcm"):
    r = rows.get(a)
    if not r:
        print(f"  {a}: MISSING"); bad += 1; continue
    diff = [k for k in INV if b[k] != r[k]]
    print(f"  {a:<4} invariant {'IDENTICAL' if not diff else 'MISMATCH ' + str(diff)} | "
          f"gates {r['gate_cover']}/{r['gate_recip']}")
    bad += bool(diff)
raise SystemExit(1 if bad else 0)
EOF
m11_check_sources
echo; echo "=== fixiso: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
