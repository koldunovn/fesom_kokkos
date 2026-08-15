#!/bin/bash
#SBATCH --job-name=bigpart
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH -o /work/ab0995/a270088/port2/mesh/%x.%j.out
#SBATCH -e /work/ab0995/a270088/port2/mesh/%x.%j.err
#
# Generate the partitions dars and NG5 need to reach 300-500 vertices/core.
#
# 🔴 STANDING RULE: modified meshes are COPIES under /work, never /pool. /pool IS writable
# for this account, so symlinking the base files back to /pool would NOT be safe — these are
# full rsync copies. Integrity (L73 level sums) is asserted below before anything is written.
#
# WHY these counts (user rule of thumb: a mesh scales to ~300-500 vertices/core):
#   dars 3160340 verts -> 6320..10534 ranks  (49-82 CPU nodes)   existing max: dist_4096
#   NG5  7402886 verts -> 14805..24676 ranks (115-192 CPU nodes) existing max: dist_8192
# so neither mesh can currently be measured where it actually stops scaling.
#
# ⚠️ Generating a partition is cheap; USING it needs a 48-192 node allocation, which may or
# may not be obtainable. These exist so the measurement is possible whenever the machine allows.
#
# Usage: MESHNAME=dars sbatch jobs/m10_make_bigparts.sh
#        MESHNAME=ng5  sbatch jobs/m10_make_bigparts.sh
set -u
MESHNAME=${MESHNAME:?set MESHNAME=dars or ng5}
MESH=/work/ab0995/a270088/port2/mesh/${MESHNAME}_bigpart
# 🔴 CORE2 MUST come from the PRIVATE mesh, never /pool: /pool's core2 has nlvls/elvls
# swapped (L73). Verified 2026-08-16: private nlvls md5 cbcbee86 vs /pool 8e54713c, and
# core2_bigpart matches the PRIVATE one. SRC_OVERRIDE keeps the integrity gate meaningful.
SRC=${SRC_OVERRIDE:-/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/${MESHNAME}}
W=$MESH/_partwork
case "$MESHNAME" in
  dars) TARGETS="6144 8192 10240" ;;
  ng5)  TARGETS="16384 20480 24576" ;;
  # M14 2026-08-15 (user: "generate additional farc partitions"). fArc is 638387 verts, so the
  # 300-500 v/core rule puts its knee near 1280-2130 ranks — and the largest partition that
  # existed anywhere was 2304 (277 v/core), i.e. AT the knee. fArc CPU therefore could not show
  # a turnover at all, which is what the campaign's past-the-knee points are for.
  #   3072 -> 208 v/core | 4096 -> 156 | 8192 -> 78 (deep past the rule of thumb)
  farc) TARGETS="3072 4096 8192" ;;
  # CORE2 push to NEGATIVE scaling: at 2048 it still improves (0.0371). 8192 ranks is
  # 15 verts/rank, far past any sane operating point -- the aim is to SHOW the rollover.
  core2) TARGETS="3072 4096 6144 8192" ;;
  *)    echo "unknown mesh $MESHNAME"; exit 2 ;;
esac
TARGETS=${TARGETS_OVERRIDE:-$TARGETS}

# ---- integrity gate: the copy must match the source bit-for-bit on the mesh definition ----
for f in nod2d.out elem2d.out nlvls.out elvls.out; do
    a=$(md5sum "$SRC/$f" | cut -d' ' -f1); b=$(md5sum "$MESH/$f" | cut -d' ' -f1)
    if [ "$a" != "$b" ]; then echo "REFUSE: $f differs between source and copy"; exit 2; fi
    echo "  ok $f md5 $a"
done

mkdir -p "$W"; cd "$W"
FP=/home/a/a270088/fesom_part/fesom2
ln -sf $FP/bin/fesom_meshpart .
for n in namelist.config namelist.forcing namelist.oce namelist.ice namelist.icepack \
         namelist.cvmix namelist.dyn namelist.tra; do
    [ -f "$FP/config/$n" ] && cp -n "$FP/config/$n" . 2>/dev/null
done

set +u   # the FESOM env script reads LD_LIBRARY_PATH before setting it
source /sw/etc/profile.levante
source $FP/env/levante.dkrz.de/shell
set -u
ulimit -s unlimited

NV=$(head -1 "$MESH/nod2d.out" | tr -d ' ')
for N in $TARGETS; do
    echo "=================================================================="
    echo "=== $MESHNAME -> $N parts   ($((NV / N)) vertices/core)"; date
    sed -i -E "s|^MeshPath *=.*|MeshPath         = '$MESH/'|" namelist.config
    sed -i -E "s|^n_levels *=.*|n_levels = 1|"                namelist.config
    sed -i -E "s|^n_part *=.*|n_part   = $N|"                 namelist.config
    got=$(grep -E "^MeshPath" namelist.config | sed -E "s|.*'(.*)'.*|\1|")
    [ "$got" = "$MESH/" ] || { echo "REFUSE: MeshPath='$got'"; exit 2; }
    srun -l ./fesom_meshpart > "meshpart_$N.out" 2>&1
    echo "    rc=$?"; tail -3 "meshpart_$N.out"
    if [ -d "$MESH/dist_$N" ]; then
        echo "    -> dist_$N OK ($(ls "$MESH/dist_$N" | wc -l) files, $(/usr/bin/du -sh "$MESH/dist_$N" | cut -f1))"
    else
        echo "    !! dist_$N MISSING"
    fi
done
echo "=== $MESHNAME partitions now available:"
ls -d "$MESH"/dist_* 2>/dev/null | sed 's/.*dist_//' | sort -n | tr '\n' ' '; echo
