#!/bin/bash
#SBATCH --job-name=m11set864
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/port2/m11/set864.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/set864.%j.err
#
# M11: `check_partitioning` is NOT idempotent in a single pass. Settling the shipped CORE2
# dist_864 relocated 71 isolated nodes and left exactly ONE new one behind — so the partition
# was still not a fixed point, and injecting it into a renumbered mesh moved that node again,
# differently, breaking the invariant-block equality the ordering race depends on.
#
# Iterate the injection until the scorecard reports isolated_nodes == 0, then rebuild the two
# renumbered arms from the fixed point.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
BASE=$SB/core2_base
N=${N:-864}
OUT=/work/ab0995/a270088/port2/m11/set864.${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"

iso() { $PY "$ROOT/scripts/m11_scorecard.py" "$1" --dist "$N" --no-dist-files 2>/dev/null \
        | sed -n 's/.*isolated (<=1 same-part nb) \([0-9]*\).*/\1/p'; }

echo "=== settling CORE2 dist_$N to a fixed point of check_partitioning"
for it in 1 2 3 4 5; do
    cur=$(iso "$BASE")
    echo "--- pass $it: isolated_nodes before = ${cur:-?}"
    [ "${cur:-1}" = "0" ] && { echo "    fixed point reached"; break; }
    $PY "$ROOT/scripts/m11_part_import.py" --from-dist "$BASE" --npes "$N" \
        -o "$OUT/v_$it.txt" --mesh "$BASE" >/dev/null || exit 3
    env MESH="$BASE" RANKS="$N" TAG="settle${it}" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
        BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_FILE="$OUT/v_$it.txt" \
        bash "$ROOT/scripts/m11_partgen.sh" > "$OUT/gen_$it.log" 2>&1 || exit 3
    echo "    post-pass moved $(grep -ac 'Isolated node' "$OUT/meshpart_settle${it}_$N.log" || echo 0) node(s)"
done
final=$(iso "$BASE")
echo "=== baseline isolated_nodes = ${final:-?}"
[ "${final:-1}" = "0" ] || { echo "REFUSE: never reached a fixed point"; exit 3; }

fail=0
for arm in hil rcm; do
    echo; echo "--- rebuild $arm dist_$N from the fixed point"
    $PY "$ROOT/scripts/m11_renumber.py" --permute-labels "$SB/core2_$arm" \
        --from-dist "$BASE" --npes "$N" -o "$OUT/part_${arm}.txt" || { fail=1; continue; }
    env MESH="$SB/core2_$arm" RANKS="$N" TAG="${arm}_$N" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
        BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_FILE="$OUT/part_${arm}.txt" \
        bash "$ROOT/scripts/m11_partgen.sh" > "$OUT/gen_$arm.log" 2>&1 || fail=1
    echo "    post-pass moved $(grep -ac 'Isolated node' "$OUT/meshpart_${arm}_${N}_$N.log" 2>/dev/null || grep -ac 'Isolated node' "$OUT/meshpart_${arm}_$N.log" 2>/dev/null || echo 0) node(s)"
done

echo; echo "=== invariant-block equality gate at N=$N"
CSV=$OUT/settle.csv
$PY "$ROOT/scripts/m11_scorecard.py" "$BASE"         --dist $N --arm "base_$N" --csv "$CSV" >/dev/null 2>&1
$PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_hil" --dist $N --arm "hil_$N"  --csv "$CSV" >/dev/null 2>&1
$PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_rcm" --dist $N --arm "rcm_$N"  --csv "$CSV" >/dev/null 2>&1
$PY - "$CSV" <<'EOF' || fail=1
import csv, sys
rows = {r["arm"].rsplit("_",1)[0]: r for r in csv.DictReader(open(sys.argv[1]))}
INV = ["n2d_max","n2d_imb","n3d_max","n3d_imb","n3d_maxmin","w0_imb","w0_sum","w100_imb",
       "w100_sum","edgecut_unweighted","cutweight_nlev","edges_total","mean_edge_weight",
       "commvol_total","commvol_max_rank","commvol_imb","boundary_nodes","nbr_max","nbr_mean",
       "wet_components","parts_disconnected","components_total","components_max",
       "noncore_vertices","singleton_vertices","isolated_nodes","halo_nod_mean","halo_nod_max",
       "elem_repl","edge_repl"]
b = rows["base"]; bad = 0
print(f"  base isolated_nodes={b['isolated_nodes']} cut_unw={int(b['edgecut_unweighted']):,} "
      f"halo/rank={float(b['halo_nod_mean']):.1f}")
for a in ("hil", "rcm"):
    r = rows[a]
    diff = [k for k in INV if b[k] != r[k]]
    print(f"  {a:<4} {'IDENTICAL' if not diff else 'MISMATCH ' + str(diff)} | "
          f"gates {r['gate_cover']}/{r['gate_recip']}")
    bad += bool(diff)
raise SystemExit(1 if bad else 0)
EOF
m11_check_sources
echo; echo "=== settle $N: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
