#!/bin/bash
#SBATCH --job-name=m11acc
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH -N 2
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:20:00
#SBATCH -o /work/ab0995/a270088/port2/m11/acc.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/acc.%j.err
#
# M11 ordering race: the PARTITION-CLASS ACCURACY FLOOR, which my brought-forward
# pre-registration should have specified and did not.
#
# What went wrong: I pre-registered "relative |delta| <= 1e-9 at step 1 on the printed maxima".
# That bound is untestable — the diagnostic print carries three significant digits — and it is
# the wrong physics: `Kv`, `Av` and `w` are threshold-driven quantities in a model whose
# mixing scheme flips boundary-layer depths on round-off. The gate failed on its own
# instrument, not necessarily on the meshes.
#
# The project's own standard is a per-class FLOOR measured from control pairs (L79), so measure
# it. Both controls keep the numbering fixed and change only the partition, so they need no
# permutation and isolate exactly the perturbation a renumbering also introduces:
#   control A (minimal)   shipped dist_256  vs  settled dist_256   — ONE node assigned differently
#   control B (realistic) shipped dist_256  vs  core2_wgt0 dist_256 — a wholly different partition
# The ordering arms are then judged against these, not against a number I guessed.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')
export FESOM_PRINT_EVERY=999

SB=/work/ab0995/a270088/port2/mesh_m11
BIN=/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
OUT=/work/ab0995/a270088/port2/m11/acc.${SLURM_JOB_ID}
mkdir -p "$OUT"
md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = 5c3c90fc0ea3939df86cfbe275287c36 ] || { echo "BIN md5 $md5 not certified"; exit 2; }
echo "=== M11 partition-class accuracy floor, CORE2 256 ranks, 20 steps dt 1800"

# ship = the shipped partition; settled = one node moved by check_partitioning;
# wgt0 = M10's regenerated 2D-only partition (same mesh files, verified identical md5)
run() {  # run <tag> <mesh>
    local t=$1 m=$2
    local o="$OUT/$t"; mkdir -p "$o"
    srun "$BIN" "$m" "$o" 1800 20 -1 "$PHC" 1958 > "$OUT/log_$t.txt" 2>&1
    echo "  $t rc=$? mesh=$m"
}
run ship    "$SB/core2_m11"
run settled "$SB/core2_base"
run wgt0    /work/ab0995/a270088/port2/mesh/core2_wgt0
echo "=== done; compare with scripts/m11_accuracy_compare.py"
