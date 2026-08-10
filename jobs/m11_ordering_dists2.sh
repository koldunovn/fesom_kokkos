#!/bin/bash
#SBATCH --job-name=m11odis2
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/odist2.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/odist2.%j.err
#
# M11 ordering race, step 0 (v2): SETTLED baselines, then label-permuted arms.
#
# WHY v2. Job 26852384 label-permuted the SHIPPED partitions straight onto the renumbered
# meshes, and the invariant-block gate refused the result at N=256 and N=864 — exactly the two
# counts whose shipped partition still contains nodes with <=1 same-partition neighbour (1 and
# 71 respectively; the scorecard's `isolated_nodes`). On injection `check_partitioning` moves
# those nodes, and because it walks `do n=1,nod2D` and consults running per-partition loads,
# WHICH node goes WHERE depends on the numbering. So the two arms stopped carrying the same
# partition, and a race between them would have been an ordering-plus-repartitioning race.
#
# FIX: settle first. Inject each shipped vector back into the OLD mesh once, letting the
# post-pass do its work there; the settled partition then has no isolated nodes, so injecting
# it into a renumbered mesh moves nothing and all three arms carry an identical partition.
# The baseline arm becomes the settled partition too, which is the fair comparison — it
# removes a confound that has nothing to do with numbering.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
SHIP=$SB/core2_m11
BASE=$SB/core2_base
OUT=/work/ab0995/a270088/port2/m11/odist2.${SLURM_JOB_ID:-manual}
RANKS=${RANKS:-"4 8 256 512 864"}
mkdir -p "$OUT"
fail=0

gen() {   # gen <mesh> <tag> <N> <partfile>
    env MESH="$1" RANKS="$3" TAG="$2" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
        BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_FILE="$4" \
        bash "$ROOT/scripts/m11_partgen.sh"
}

for N in $RANKS; do
    echo "=================================================================="
    echo "=== settle the shipped dist_$N on the OLD numbering"
    $PY "$ROOT/scripts/m11_part_import.py" --from-dist "$SHIP" --npes "$N" \
        -o "$OUT/ship_$N.txt" --mesh "$SHIP" >/dev/null || { fail=1; continue; }
    gen "$BASE" "base_$N" "$N" "$OUT/ship_$N.txt" || { fail=1; continue; }
    moved=$(grep -ac "Isolated node" "$OUT/meshpart_base_${N}_$N.log" 2>/dev/null || \
            grep -ac "Isolated node" "$OUT/meshpart_base_$N.log" 2>/dev/null || echo "?")
    echo "    post-pass moved: $moved node(s) while settling"

    for arm in hil rcm; do
        echo "--- $arm dist_$N from the SETTLED vector"
        $PY "$ROOT/scripts/m11_renumber.py" --permute-labels "$SB/core2_$arm" \
            --from-dist "$BASE" --npes "$N" -o "$OUT/part_${arm}_$N.txt" || { fail=1; continue; }
        gen "$SB/core2_$arm" "${arm}_$N" "$N" "$OUT/part_${arm}_$N.txt" || fail=1
    done
done

echo; echo "=================================================================="
echo "=== invariant-block equality gate (settled baseline)"
CSV=$OUT/ordering_dists2.csv
for N in $RANKS; do
    $PY "$ROOT/scripts/m11_scorecard.py" "$BASE"          --dist "$N" --arm "base_$N" --csv "$CSV" >/dev/null 2>&1
    $PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_hil"  --dist "$N" --arm "hil_$N"  --csv "$CSV" >/dev/null 2>&1
    $PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_rcm"  --dist "$N" --arm "rcm_$N"  --csv "$CSV" >/dev/null 2>&1
done
$PY - "$CSV" <<'EOF' || fail=1
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
INV = ["n2d_max","n2d_imb","n3d_max","n3d_imb","n3d_maxmin","w0_imb","w0_sum","w100_imb",
       "w100_sum","edgecut_unweighted","cutweight_nlev","edges_total","mean_edge_weight",
       "commvol_total","commvol_max_rank","commvol_imb","boundary_nodes","nbr_max","nbr_mean",
       "wet_components","parts_disconnected","components_total","components_max",
       "noncore_vertices","singleton_vertices","isolated_nodes","halo_nod_mean","halo_nod_max",
       "elem_repl","edge_repl"]
by = collections.defaultdict(dict)
for r in rows:
    arm, n = r["arm"].rsplit("_", 1)
    by[n][arm] = r
bad = 0
for n in sorted(by, key=int):
    base = by[n].get("base")
    if not base:
        print(f"  N={n}: no settled baseline"); bad += 1; continue
    print(f"  base N={n:<4} isolated_nodes={base['isolated_nodes']:<3} "
          f"cut_unw={int(base['edgecut_unweighted']):>7,}  halo/rank={float(base['halo_nod_mean']):.1f}")
    for arm in ("hil", "rcm"):
        r = by[n].get(arm)
        if not r:
            print(f"    {arm}: MISSING"); bad += 1; continue
        diff = [k for k in INV if base[k] != r[k]]
        gates = r["gate_cover"] == "ok" and r["gate_recip"] == "ok"
        print(f"    {arm:<4} invariant {'IDENTICAL' if not diff else 'MISMATCH ' + str(diff)}"
              f" | gates {'ok' if gates else 'FAIL'}")
        if diff or not gates:
            bad += 1
print(f"\n  ordering dists v2: {'ALL GREEN' if not bad else str(bad) + ' FAILED'}")
raise SystemExit(1 if bad else 0)
EOF

m11_check_sources
echo; echo "=== ordering dists v2: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
