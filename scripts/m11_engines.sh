#!/bin/bash
# M11 Task 6 — one wrapper per external partitioner: mesh -> graph -> engine -> part vector.
#
#   bash scripts/m11_engines.sh <engine> <mesh_dir> <k> [outdir] [extra engine args...]
#   engines: kaminpar | kaminpar-strong | mtkahypar | mtkahypar-graph | kahip | kahip-conn
#
# Everything the engines read comes from `m11_graph_export.py`, and everything they write goes
# back through `m11_part_import.py`, so the only engine-specific code is the command line and
# the shape of its output file. The exported graph has already been diffed against the
# partitioner's own CSR dump (Task 4), so an engine that reads it is reading exactly the graph
# FESOM hands METIS.
#
# Weighting. Node weight is `a + nlev` with a=100 by default — the scalar equivalent of the
# legacy dual constraint, and the variant Task 7's A3 arm sweeps. Edge weights are deliberately
# NOT exported here: `fort_part.c` sets adjwgt only on weighted arms, and mixing weighted edges
# into an unweighted-cut objective is the units error the campaign already had to correct once.
#
# Mt-KaHyPar reads the STAR EXPANSION (one net per vertex, net_v = {v} u N(v), net weight
# nlev), whose km1 was verified equal to METIS's total communication volume with vsize=nlev
# (Task 3) — so `-o km1` optimises the quantity FESOM actually pays for.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
E=${M11_ENGINE_ROOT:-/work/ab0995/a270088/port2/partm11/engines}
PY=${PY:-/work/ab0995/a270088/mambaforge/envs/nereus/bin/python}
KAMINPAR=$E/kaminpar/build/apps/KaMinPar
MTKAHYPAR=$E/mtkahypar/build/mt-kahypar/application/MtKaHyPar
KAFFPA=$E/kahip/build/kaffpa

ENGINE=${1:?engine}; MESH=${2:?mesh dir}; K=${3:?k}
OUT=${4:-/work/ab0995/a270088/port2/m11/engines}
shift 4 2>/dev/null || shift $#
EPS=${M11_EPS:-0.03}
WGT_A=${M11_WGT_A:-100}
THREADS=${M11_THREADS:-16}
SEED=${M11_SEED:-0}
TAG=${M11_TAG:-$(basename "$MESH")_${ENGINE}_k${K}}
mkdir -p "$OUT"

set +u; source /sw/etc/profile.levante > /dev/null 2>&1
module load gcc/13.4.0-gcc-13.4.0 > /dev/null 2>&1; set -u

graph() {   # graph <fmt: none|vwgt|hmetis> -> path (cached per mesh+variant+a)
    local kind=$1
    local base; base=$(basename "$MESH")
    local p="$OUT/graphs/${base}_${kind}_a${WGT_A}"
    mkdir -p "$OUT/graphs"
    case $kind in
        none)   p="$p.graph"; [ -s "$p" ] || $PY "$ROOT/scripts/m11_graph_export.py" "$MESH" \
                    -o "$p" --weights none  >&2 ;;
        vwgt)   p="$p.graph"; [ -s "$p" ] || $PY "$ROOT/scripts/m11_graph_export.py" "$MESH" \
                    -o "$p" --weights vwgt --wgt-a "$WGT_A" >&2 ;;
        hmetis) p="$p.hgr";   [ -s "$p" ] || $PY "$ROOT/scripts/m11_graph_export.py" "$MESH" \
                    -o "$p" --format hmetis --wgt-a "$WGT_A" >&2 ;;
    esac
    printf '%s' "$p"
}

RAW=$OUT/$TAG.raw
VEC=$OUT/$TAG.part
LOG=$OUT/$TAG.log
t0=$(date +%s)

case $ENGINE in
  kaminpar|kaminpar-strong)
    preset=default; [ "$ENGINE" = kaminpar-strong ] && preset=strong
    G=$(graph "${M11_GRAPH:-vwgt}")
    "$KAMINPAR" -G "$G" -k "$K" -e "$EPS" -t "$THREADS" -P "$preset" -s "$SEED" \
        -o "$RAW" "$@" > "$LOG" 2>&1 || { echo "ENGINE FAILED (see $LOG)"; tail -5 "$LOG"; exit 3; }
    ;;
  mtkahypar|mtkahypar-graph)
    if [ "$ENGINE" = mtkahypar-graph ]; then G=$(graph "${M11_GRAPH:-vwgt}"); FF=metis; else G=$(graph hmetis); FF=hmetis; fi
    # MtKaHyPar names its own output file; run it in a private dir and take whatever appeared.
    W="$OUT/$TAG.work"; rm -rf "$W"; mkdir -p "$W"
    "$MTKAHYPAR" -h "$G" -k "$K" -e "$EPS" -o km1 --preset-type=quality --file-format=$FF \
        -t "$THREADS" --seed "$SEED" -w --partition-output-folder="$W" "$@" > "$LOG" 2>&1 \
        || { echo "ENGINE FAILED (see $LOG)"; tail -5 "$LOG"; exit 3; }
    f=$(ls -S "$W" | head -1)
    [ -n "$f" ] || { echo "no partition file written to $W"; exit 3; }
    cp "$W/$f" "$RAW"
    ;;
  kahip|kahip-conn)
    G=$(graph "${M11_GRAPH:-vwgt}")
    extra=""; [ "$ENGINE" = kahip-conn ] && extra="--connected_blocks"
    W="$OUT/$TAG.work"; rm -rf "$W"; mkdir -p "$W"
    ( cd "$W" && "$KAFFPA" "$G" --k "$K" --imbalance $(awk -v e="$EPS" 'BEGIN{print e*100}') \
        --preconfiguration=strong --seed "$SEED" $extra --output_filename="$RAW" "$@" ) \
        > "$LOG" 2>&1 || { echo "ENGINE FAILED (see $LOG)"; tail -5 "$LOG"; exit 3; }
    ;;
  *) echo "unknown engine '$ENGINE'"; exit 2 ;;
esac
t1=$(date +%s)

[ -s "$RAW" ] || { echo "engine produced no output ($RAW)"; exit 3; }
$PY "$ROOT/scripts/m11_part_import.py" "$RAW" --npes "$K" --mesh "$MESH" -o "$VEC" \
    || { echo "importer REFUSED the engine output"; exit 3; }
echo "  $TAG: $((t1 - t0)) s, vector -> $VEC"
