#!/bin/bash
#SBATCH --job-name=m11pgen
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /work/ab0995/a270088/port2/m11/partgen.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/partgen.%j.err
#
# M11 partition generation harness. One arm = one invocation with FESOM_PART_* set.
#
#   sbatch --export=ALL,MESH=...,RANKS="8 16",TAG=a1,FESOM_PART_KWAY=1 scripts/m11_partgen.sh
#
# Guarantees, in order of how badly their absence would hurt:
#   * MeshPath is re-read FROM the namelist after the sed and asserted to be inside the
#     sandbox — the sed is the thing that can go wrong, so the variable it was built from
#     is not evidence.
#   * the read-only source meshes are md5- and mtime-checked after EVERY partitioner run.
#   * nod2d/elem2d/aux3d must be byte-identical before and after; nlvls/elvls/edge files may
#     legitimately appear if they were absent (the partitioner writes them then), which is
#     announced, never silent.
#   * provenance is the md5 of BOTH the executable AND libfesom_meshpart_C.so — the C
#     partitioning code lives in the shared library, so the executable's md5 alone does not
#     change when fort_part.c does. (M10 lost two runs to a stale binary; this is how that
#     happens.)
#   * every knob that was set is echoed, and the log is kept UNFILTERED: the partitioner's
#     "Isolated node" lines from check_partitioning are the only record that the shipped
#     partition differs from the one METIS scored.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"

PARTROOT=${PARTROOT:-/work/ab0995/a270088/port2/partm11/fesom2}
BIN=${BIN:-$PARTROOT/mesh_part/build/bin/fesom_meshpart}
MESH=${MESH:?MESH required (a mesh dir inside the M11 sandbox)}
RANKS=${RANKS:?RANKS required, e.g. "8 16 32"}
TAG=${TAG:-$(basename "$MESH")}
NLEVELS=${NLEVELS:-1}
NPART_LOW=${NPART_LOW:-}          # optional: hierarchical, e.g. "4" -> n_part = <N>,4
OUT=${OUT:-/work/ab0995/a270088/port2/m11/partgen.${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "REFUSE: partitioner $BIN missing"; exit 2; }
SO=$(ldd "$BIN" 2>/dev/null | sed -n 's/.*=> \(.*libfesom_meshpart_C\.so\).*/\1/p')
echo "=== M11 partgen  TAG=$TAG  mesh=$MESH  ranks=$RANKS  n_levels=$NLEVELS"
echo "  binary : $BIN"
echo "  md5    : $(md5sum "$BIN" | cut -d' ' -f1)"
echo "  libC   : ${SO:-<none>}"
echo "  md5    : $( [ -n "$SO" ] && md5sum "$SO" | cut -d' ' -f1 || echo n/a)"
echo "  knobs  : $(env | grep -E '^FESOM_PART_' | sort | tr '\n' ' ')"
echo "  (unset knobs mean the historical compiled defaults: dual+100, Recursive, UFACTOR=1,"
echo "   NCUTS=10, NITER=15, seed 35243)"

m11_assert_sandbox "$MESH" "partgen MESH" > /dev/null
m11_sources_init

# nod2d/elem2d/aux3d define the mesh and must survive untouched.
declare -A BEFORE
for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out edges.out edge_tri.out edgenum.out; do
    [ -f "$MESH/$f" ] && BEFORE[$f]=$(md5sum "$MESH/$f" | cut -d' ' -f1) || BEFORE[$f]=ABSENT
done

set +u
source /sw/etc/profile.levante
source "$PARTROOT/env/levante.dkrz.de/shell"
set -u
ulimit -s unlimited

W=$OUT/work_$TAG; mkdir -p "$W"; cd "$W"
# 🔴 Run the binary IN PLACE. fesom_meshpart finds libfesom_meshpart_C.so through an
# rpath of $ORIGIN/../lib64, which the dynamic linker resolves from the real path of the
# running image — so copying the executable into a work dir breaks it with a bare
# "cannot open shared object file" (job 26851114). Only the namelists need to be in CWD.
export LD_LIBRARY_PATH="$(dirname "$BIN")/../lib64:$(dirname "$BIN")/../lib:${LD_LIBRARY_PATH:-}"
for n in namelist.config namelist.forcing namelist.oce namelist.ice namelist.icepack \
         namelist.cvmix namelist.dyn namelist.tra; do
    [ -f "$PARTROOT/config/$n" ] && cp -f "$PARTROOT/config/$n" . 2>/dev/null
done
[ -f namelist.config ] || { echo "REFUSE: no namelist.config in $PARTROOT/config"; exit 2; }

rc_all=0
for N in $RANKS; do
    NP="$N"; [ -n "$NPART_LOW" ] && NP="$N,$NPART_LOW"
    sed -i -E "s|^MeshPath *=.*|MeshPath         = '$MESH/'|"  namelist.config
    sed -i -E "s|^n_levels *=.*|n_levels = $NLEVELS|"          namelist.config
    sed -i -E "s|^n_part *=.*|n_part   = $NP|"                 namelist.config
    # re-read from the FILE, resolve, and assert — the sed is what can fail
    m11_assert_namelist_meshpath namelist.config "$MESH" > /dev/null

    echo; echo "--- $TAG n_part=$NP  (dist_$N)"
    LOG="$OUT/meshpart_${TAG}_$N.log"
    rm -rf "$MESH/dist_$N"
    srun -l "$BIN" > "$LOG" 2>&1
    rc=$?
    echo "    rc=$rc  log=$LOG"
    # the source sweep runs whatever happened: a partitioner that died half-way is
    # exactly when a stray write is most likely
    m11_check_sources
    if [ $rc -ne 0 ]; then
        echo "    !! partitioner FAILED"; tail -20 "$LOG"; rc_all=3; continue
    fi
    grep -aE "^\[M11\]|Distribution weight|METIS edgecut|Isolated node|Check for isolated" "$LOG" \
        | sed 's/^/      /'
    iso=$(grep -ac "Isolated node" "$LOG" || true)
    echo "    check_partitioning moved $iso node(s)"
    [ -d "$MESH/dist_$N" ] && echo "    -> dist_$N written" \
                           || { echo "    !! dist_$N MISSING"; rc_all=3; }
done

echo; echo "--- mesh-file integrity after $TAG"
for f in "${!BEFORE[@]}"; do
    now=ABSENT; [ -f "$MESH/$f" ] && now=$(md5sum "$MESH/$f" | cut -d' ' -f1)
    if [ "${BEFORE[$f]}" = "$now" ]; then
        continue
    elif [ "${BEFORE[$f]}" = "ABSENT" ]; then
        echo "  NOTE: $f was ABSENT and has been WRITTEN by the partitioner ($now)"
        case "$f" in
            nod2d.out|elem2d.out|aux3d.out)
                echo "  REFUSE: the partitioner must never create a mesh-definition file"; rc_all=4 ;;
        esac
    else
        echo "  REFUSE: $f CHANGED (${BEFORE[$f]} -> $now)"; rc_all=4
    fi
done
[ $rc_all -eq 0 ] && echo "  all tracked mesh files unchanged"
m11_md5_write "$MESH" > /dev/null
echo; echo "=== M11 partgen $TAG done, rc=$rc_all"
exit $rc_all
