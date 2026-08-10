#!/bin/bash
#SBATCH --job-name=m11rnum
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:20:00
#SBATCH -o /work/ab0995/a270088/port2/m11/rnum.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/rnum.%j.err
#
# M11 Task 5 smoke, in the order the plan requires (each step's output is the next step's
# input, so a failure cannot be mistaken for a later one):
#
#   1. label-permute the CORE2 dist_8 partition onto the renumbered mesh — SAME partition,
#      new numbering. This is what makes an ordering A/B a pure ordering A/B: without it the
#      arm would also be a repartitioning arm.
#   2. inject that vector; the partitioner regenerates edges.out/edge_tri.out/edgenum.out for
#      the new numbering and writes dist_8.
#   3. run the certified Serial h17 binary on the renumbered mesh — the halo identity test
#      (proven non-vacuous in Task 1) is the correctness gate.
#   4. the scorecard's permutation-invariant block on (new mesh, permuted dist) must be
#      EXACTLY equal to (old mesh, old dist); the ordering block must move.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
OLD=$SB/core2_m11
NEW=${NEW:-$SB/core2_hil}
N=8
OUT=/work/ab0995/a270088/port2/m11/rnum.${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"
fail=0

echo "############ 1. label-permute dist_$N onto $(basename "$NEW")"
$PY "$ROOT/scripts/m11_renumber.py" --permute-labels "$NEW" --from-dist "$OLD" --npes $N \
    -o "$OUT/part_$N.txt" || fail=1

echo; echo "############ 2. inject it; the partitioner regenerates the edge files"
env MESH="$NEW" RANKS="$N" TAG=rnum ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
    BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_FILE="$OUT/part_$N.txt" \
    bash "$ROOT/scripts/m11_partgen.sh" || fail=1
for f in edges.out edge_tri.out edgenum.out; do
    [ -f "$NEW/$f" ] && echo "  regenerated $f" || { echo "  MISSING $f"; fail=1; }
done

echo; echo "############ 3. Serial h17 on the renumbered mesh (halo identity gate)"
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
BIN=/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial
md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = 5c3c90fc0ea3939df86cfbe275287c36 ] || { echo "  BIN md5 $md5 is not certified h17"; exit 2; }
RUN=$OUT/run; mkdir -p "$RUN"
srun -l "$BIN" "$NEW" "$RUN" 1800 2 -1 > "$OUT/model.log" 2>&1
rc=$?
echo "  rc=$rc  binary md5 $md5"
grep -E "identity test|FATAL|mismatched" "$OUT/model.log" | head -5
[ $rc -eq 0 ] || fail=1
grep -q "identity test (positive): all halo entries carry correct gid" "$OUT/model.log" \
    || { echo "  the halo gate did not announce itself"; fail=1; }

echo; echo "############ 4. scorecard: invariant block must be IDENTICAL, ordering block must move"
CSV=$OUT/renumber_compare.csv
$PY "$ROOT/scripts/m11_scorecard.py" "$OLD" --dist $N --arm old_numbering --csv "$CSV" > "$OUT/sc_old.log" 2>&1
$PY "$ROOT/scripts/m11_scorecard.py" "$NEW" --dist $N --arm new_numbering --csv "$CSV" > "$OUT/sc_new.log" 2>&1
$PY - "$CSV" <<'EOF' || fail=1
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
a, b = rows[0], rows[1]
INV = ["n2d_max","n2d_imb","n3d_max","n3d_imb","n3d_maxmin","w0_imb","w0_sum","w100_imb",
       "w100_sum","edgecut_unweighted","cutweight_nlev","edges_total","mean_edge_weight",
       "commvol_total","commvol_max_rank","commvol_imb","boundary_nodes","nbr_max","nbr_mean",
       "wet_components","parts_disconnected","components_total","components_max",
       "noncore_vertices","singleton_vertices","isolated_nodes","halo_nod_mean","halo_nod_max",
       "elem_repl","edge_repl"]
bad = [k for k in INV if a[k] != b[k]]
print(f"  invariant keys compared: {len(INV)} (incl. halo and replication, which come from the"
      f" regenerated dist files, not just the graph)")
for k in bad:
    print(f"  MISMATCH {k}: {a[k]} -> {b[k]}")
print("  invariant block: " + ("IDENTICAL" if not bad else f"{len(bad)} MISMATCHES"))
ORD = [k for k in a if k.startswith("ord_")]
moved = [k for k in ORD if a[k] != b[k]]
print(f"  ordering block: {len(moved)}/{len(ORD)} keys moved  "
      f"(|di| edge mean {float(a['ord_edge_didx_mean']):,.0f} -> {float(b['ord_edge_didx_mean']):,.0f}; "
      f"elem strides <=64 {float(a['ord_stride_elem_l64']):.3f} -> {float(b['ord_stride_elem_l64']):.3f})")
print(f"  gates: cover {b['gate_cover']} | reciprocity {b['gate_recip']}")
raise SystemExit(0 if (not bad and moved and b['gate_cover']=='ok' and b['gate_recip']=='ok') else 1)
EOF

m11_check_sources
echo; echo "############ Task 5 smoke: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
