#!/bin/bash
#SBATCH --job-name=partwgt
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /work/ab0995/a270088/port2/mesh/%x.%j.out
#SBATCH -e /work/ab0995/a270088/port2/mesh/%x.%j.err
#
# M10 load-balance study — generate CORE2 partitions under all three METIS vertex weightings
# from ONE binary, so the comparison isolates the weighting.
#
# WHY: the ocean phase's per-rank COMPUTE spans 4.8x (farc 2048, CORE2 864) and correlates
# with owned 3D nodes at r=0.967, not with 2D nodes (r=0.003). The shipped CORE2 dist_864 and
# /pool farc dist_2048 have a 3D spread of 9.6x / 9.4x, while /pool NG5 and our own
# dars/ng5_bigpart are dual-weighted at 1.04-1.05x.
#
# 🔴 The M7-era conclusion "repartitioning is OUT (user tried)" rests on an NG5 test whose
# regenerated ng5_w3d dists came out BYTE-IDENTICAL to /pool — because NG5 was ALREADY
# dual-weighted. That A/B compared a partition against itself (an L80 dead-knob null) and was
# structurally incapable of showing an effect. On CORE2 the partition IS imbalanced, so the
# lever is untested here.
#
# The weighting was a COMPILE-TIME #ifdef PART_WEIGHTED, so 2D-only and dual could previously
# only be compared across two binaries. The patched copy at $PW reads FESOM_PART_WGT at
# runtime (0=2D only, 1=3D only, 2=dual), which removes that confound entirely.
#
# Usage: sbatch jobs/m10_part_wgt_study.sh          (default ranks 512 864)
#        RANKS="512" sbatch jobs/m10_part_wgt_study.sh
set -u
SRC=/work/ab0995/a270088/port2/mesh/core2          # the PRIVATE CORE2 copy (standing rule L73)
PW=/work/ab0995/a270088/port2/partw/fesom2
BIN=$PW/mesh_part/build/bin/fesom_meshpart
# 🔴 L80: the knob MUST announce itself. The first run of this study (26743391) silently
# used a stale pre-rebuild binary and produced three IDENTICAL partitions (same edgecut)
# under three different FESOM_PART_WGT values. The assertion below makes that unmissable.
RANKS=${RANKS:-"512 864"}
# wgt_type=1 (3D-only) is LEGACY dead code — it hands METIS the caller's Fortran array
# directly and aborts inside METIS in this build (job 26743709, rc=134). The question is
# answered by 0 (2D-only, = how the shipped CORE2/farc partitions look) vs 2 (dual).
WGTS=${WGTS:-"0 2"}
BASE=/work/ab0995/a270088/port2/mesh

[ -x "$BIN" ] || { echo "REFUSE: partitioner $BIN missing"; exit 2; }

set +u
source /sw/etc/profile.levante
source $PW/env/levante.dkrz.de/shell
set -u
ulimit -s unlimited

for WGT in $WGTS; do
    MESH=$BASE/core2_wgt$WGT
    echo "=================================================================="
    echo "=== weighting $WGT -> $MESH"; date
    if [ ! -d "$MESH" ]; then
        mkdir -p "$MESH"
        # mesh definition only; dist_* are generated below. Full copies, never symlinks
        # into /pool or into the certified mesh (a stray write must not travel back).
        for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out; do
            [ -f "$SRC/$f" ] && cp -n "$SRC/$f" "$MESH/"
        done
    fi
    # integrity gate: the copy must match the source on the mesh definition
    for f in nod2d.out elem2d.out nlvls.out elvls.out; do
        a=$(md5sum "$SRC/$f" | cut -d' ' -f1); b=$(md5sum "$MESH/$f" | cut -d' ' -f1)
        [ "$a" = "$b" ] || { echo "REFUSE: $f differs from source"; exit 2; }
    done
    echo "  integrity gate OK"

    W=$MESH/_partwork; mkdir -p "$W"; cd "$W"
    ln -sf "$BIN" fesom_meshpart
    for n in namelist.config namelist.forcing namelist.oce namelist.ice namelist.icepack \
             namelist.cvmix namelist.dyn namelist.tra; do
        [ -f "$PW/config/$n" ] && cp -f "$PW/config/$n" . 2>/dev/null
    done
    for N in $RANKS; do
        sed -i -E "s|^MeshPath *=.*|MeshPath         = '$MESH/'|" namelist.config
        sed -i -E "s|^n_levels *=.*|n_levels = 1|"                namelist.config
        sed -i -E "s|^n_part *=.*|n_part   = $N|"                 namelist.config
        got=$(grep -E "^MeshPath" namelist.config | sed -E "s|.*'(.*)'.*|\1|")
        [ "$got" = "$MESH/" ] || { echo "REFUSE: MeshPath='$got'"; exit 2; }
        echo "--- wgt=$WGT n_part=$N"
        rm -rf "$MESH/dist_$N"
        FESOM_PART_WGT=$WGT srun -l ./fesom_meshpart > "meshpart_w${WGT}_$N.out" 2>&1
        rc=$?; echo "    rc=$rc"
        [ "$rc" -eq 0 ] || { echo "    !! partitioner FAILED rc=$rc (see meshpart_w${WGT}_$N.out)"; exit 3; }
        grep -aE "FESOM_PART_WGT|Distribution weight|edgecut" "meshpart_w${WGT}_$N.out" | sed 's/^/      /'
        # L80 assertion: the override must have announced itself, or the leg is meaningless
        if ! grep -aq "FESOM_PART_WGT=$WGT overrides" "meshpart_w${WGT}_$N.out"; then
            echo "    !! REFUSE: FESOM_PART_WGT=$WGT did NOT fire — stale binary? aborting"; exit 3
        fi
        [ -d "$MESH/dist_$N" ] && echo "    -> dist_$N OK" || echo "    !! dist_$N MISSING"
    done
done
echo "=== done ==="
