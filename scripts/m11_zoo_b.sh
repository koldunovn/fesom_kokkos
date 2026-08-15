#!/bin/bash
# M11 Task 8 — the B-family (external engines) zoo. Login-node work: no MPI, no queue, and
# CORE2 finishes in seconds to half a minute per arm.
#
#   bash scripts/m11_zoo_b.sh <mesh_dir> <k> [engines] [weights]
#
# The weight axis is the same one the A-family sweeps: a single scalar w = a + nlev, with
# `none` meaning unit vertex weights. It has to be: KaMinPar, Mt-KaHyPar and KaHIP are all
# SINGLE-constraint partitioners, so FESOM's legacy dual constraint (1, nlev+100) cannot be
# expressed in any of them. That is a structural result for the report, not a limitation of
# this script — the dual arm is reachable only through METIS.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
PY=${PY:-/work/ab0995/a270088/mambaforge/envs/nereus/bin/python}
MESH=${1:?mesh dir}
K=${2:?k}
ENGINES=${3:-"kaminpar mtkahypar kahip"}
WEIGHTS=${4:-"0 15 40 100 none"}
OUT=${OUT:-/work/ab0995/a270088/port2/m11/engines}
CSV=$OUT/zoo_b_$(basename "$MESH")_k${K}_e${M11_EPS:-0.03}.csv
mkdir -p "$OUT"

echo "=== M11 zoo B   mesh=$MESH  k=$K  engines='$ENGINES'  weights='$WEIGHTS'"
for e in $ENGINES; do
    for w in $WEIGHTS; do
        tag="$(basename "$MESH")_${e}_a${w}_e${M11_EPS:-0.03}_k$K"
        if [ "$w" = none ]; then
            export M11_GRAPH=none M11_WGT_A=0
        else
            export M11_GRAPH=vwgt M11_WGT_A=$w
        fi
        export M11_TAG=$tag
        if [ -s "$OUT/$tag.part" ]; then
            echo "  $tag cached"
        else
            bash "$ROOT/scripts/m11_engines.sh" "$e" "$MESH" "$K" "$OUT" 2>&1 | tail -1
        fi
        [ -s "$OUT/$tag.part" ] || { echo "  $tag MISSING — skipped"; continue; }
        $PY "$ROOT/scripts/m11_scorecard.py" "$MESH" --part-file "$OUT/$tag.part" --npes "$K" \
            --arm "${e}_a${w}_e${M11_EPS:-0.03}" --csv "$CSV" --no-dist-files > /dev/null 2>&1 \
            || echo "  $tag scorecard FAILED"
    done
done

echo
$PY - "$CSV" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
K = [("edgecut_unweighted","cut_unw","{:,.0f}"), ("commvol_total","commvol","{:,.0f}"),
     ("commvol_max_rank","cv_max","{:,.0f}"), ("n2d_imb","2Dimb","{:.3f}"),
     ("n3d_maxmin","3Dmax/min","{:.2f}"), ("w100_imb","w100imb","{:.3f}"),
     ("nbr_max","nbr","{:.0f}"), ("parts_disconnected","disc","{:.0f}"),
     ("noncore_vertices","offlobe","{:,.0f}"), ("isolated_nodes","iso","{:.0f}")]
print(f"{'arm':<20}" + "".join(f"{h:>11}" for _, h, _ in K))
for r in rows:
    print(f"{r['arm']:<20}" + "".join(f"{f.format(float(r[k])):>11}" for k, _, f in K))
EOF
echo "  csv: $CSV"
