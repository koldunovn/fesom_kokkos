#!/bin/bash
#SBATCH --job-name=m11g2pre
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH -o /work/ab0995/a270088/port2/m11/g2prep.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/g2prep.%j.err
#
# M11 session 2: build CONTROL C for the re-registered accuracy/SSH gates — the same
# partitioning recipe at a different METIS seed.
#
# Why a third control. Day 1 judged the ordering arms against a SINGLE partition-class control,
# so the reference was one draw from a distribution whose spread nobody had measured. Controls
# A (one node moved) and B (a wholly different partition) bracket the extremes; a seed variant
# is the ordinary case — same objective, same weights, same tool, different local minimum — and
# it is what an unlucky regeneration actually looks like in production.
#
# The mesh files are copied from the SETTLED baseline, so control C differs from the reference
# in the partition and in nothing else.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
OUT=/work/ab0995/a270088/port2/m11/g2prep.${SLURM_JOB_ID:-manual}
SEED=${SEED:-7}
mkdir -p "$OUT"

echo "=== control C: seed-variant partition, CORE2 256 ranks, seed $SEED"
rm -rf "$SB/core2_seed"
m11_sandbox_copy_mesh "$SB/core2_base" core2_seed || exit 2

env MESH="$SB/core2_seed" RANKS=256 TAG="seed$SEED" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
    BIN="$P/mesh_part/build/bin/fesom_meshpart" FESOM_PART_SEED="$SEED" \
    bash "$ROOT/scripts/m11_partgen.sh" || exit 3

echo
echo "=== scorecard: control C against the settled reference"
CSV=$OUT/controlC.csv
$PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_base" --dist 256 --arm settled_256 --csv "$CSV" | tail -25
$PY "$ROOT/scripts/m11_scorecard.py" "$SB/core2_seed" --dist 256 --arm seed_256    --csv "$CSV" | tail -25

$PY - "$CSV" <<'EOF'
import csv, sys
rows = {r["arm"]: r for r in csv.DictReader(open(sys.argv[1]))}
a, b = rows.get("settled_256"), rows.get("seed_256")
if not (a and b):
    print("  MISSING rows"); raise SystemExit(1)
print(f"\n  {'key':<22}{'settled':>14}{'seed':>14}")
for k in ("edgecut_unweighted", "cutweight_nlev", "n3d_maxmin", "commvol_total",
          "halo_nod_mean", "elem_repl", "isolated_nodes", "parts_disconnected"):
    print(f"  {k:<22}{a[k]:>14}{b[k]:>14}")
print(f"\n  gates seed: cover={b['gate_cover']} recip={b['gate_recip']}")
raise SystemExit(0 if b["gate_cover"] == "ok" and b["gate_recip"] == "ok" else 1)
EOF
rc=$?
m11_check_sources
echo; echo "=== gate2 prep rc=$rc"
exit $rc
