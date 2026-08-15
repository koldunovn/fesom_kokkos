#!/bin/bash
#SBATCH --job-name=m11odist
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/odist.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/odist.%j.err
#
# M11 ordering race, step 0: label-permuted dists on the renumbered CORE2 meshes.
#
# The partition CONTENT is carried through the node permutation, so every arm of a point owns
# exactly the same set of nodes per rank — that is what makes the race a PURE ordering A/B
# rather than a repartitioning one. The scorecard's invariant block is then required to be
# identical to the baseline's, and the run refuses to write a dist that fails that check.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
OLD=$SB/core2_m11
OUT=/work/ab0995/a270088/port2/m11/odist.${SLURM_JOB_ID:-manual}
RANKS=${RANKS:-"4 8 256 512 864"}
mkdir -p "$OUT"
fail=0

for arm in hil rcm; do
    NEW=$SB/core2_$arm
    for N in $RANKS; do
        echo "=================================================================="
        echo "=== $arm  dist_$N"
        $PY "$ROOT/scripts/m11_renumber.py" --permute-labels "$NEW" --from-dist "$OLD" \
            --npes "$N" -o "$OUT/part_${arm}_$N.txt" || { fail=1; continue; }
        env MESH="$NEW" RANKS="$N" TAG="${arm}_$N" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
            BIN="$P/mesh_part/build/bin/fesom_meshpart" \
            FESOM_PART_FILE="$OUT/part_${arm}_$N.txt" \
            bash "$ROOT/scripts/m11_partgen.sh" || { fail=1; continue; }
    done
done

echo; echo "=================================================================="
echo "=== invariant-block equality gate: every arm must match the baseline exactly"
CSV=$OUT/ordering_dists.csv
for N in $RANKS; do
    $PY "$ROOT/scripts/m11_scorecard.py" "$OLD"          --dist "$N" --arm "base_$N" --csv "$CSV" >/dev/null 2>&1
    $PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_hil" --dist "$N" --arm "hil_$N"  --csv "$CSV" >/dev/null 2>&1
    $PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_rcm" --dist "$N" --arm "rcm_$N"  --csv "$CSV" >/dev/null 2>&1
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
    for arm in ("hil", "rcm"):
        r = by[n].get(arm)
        if not r:
            print(f"  {arm}_{n}: MISSING"); bad += 1; continue
        diff = [k for k in INV if base[k] != r[k]]
        gates = r["gate_cover"] == "ok" and r["gate_recip"] == "ok"
        print(f"  {arm:<4} N={n:<4} invariant {'IDENTICAL' if not diff else 'MISMATCH '+str(diff)}"
              f" | gates {'ok' if gates else 'FAIL'}"
              f" | ord |di| {float(base['ord_edge_didx_mean']):>7,.0f} -> {float(r['ord_edge_didx_mean']):>6,.0f}"
              f" | strides<=64 {float(base['ord_stride_elem_l64']):.3f} -> {float(r['ord_stride_elem_l64']):.3f}")
        if diff or not gates:
            bad += 1
print(f"\n  ordering dists: {'ALL GREEN' if not bad else str(bad)+' FAILED'}")
raise SystemExit(1 if bad else 0)
EOF

m11_check_sources
echo; echo "=== ordering dists: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
