#!/bin/bash
#SBATCH --job-name=m11zooa
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /work/ab0995/a270088/port2/m11/zooa.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/zooa.%j.err
#
# M11 Task 7 — the A-family (METIS) zoo: generate, then score offline. No model runs here;
# the point of the campaign's architecture is that a candidate must earn its node-hours on the
# scorecard before it gets a race.
#
#   sbatch --export=ALL,ARMS="a1 a2 a3_a0 ...",RANKS="256 512" jobs/m11_zoo_a.sh
#
# Every arm gets its OWN mesh directory (real copies, no symlinks — a symlinked mesh would let
# the partitioner write through into a source tree), and every invocation runs the source md5 +
# mtime sweep afterwards.
#
# Arm definitions live in one place, below. Two rules they follow:
#   * `OBJ=vol`, `MINCONN` and `CONTIG` are Kway-only — METIS accepts and silently ignores them
#     under PartGraphRecursive, which is exactly how contiguity and the comm-volume objective
#     were off for every FESOM partition ever generated. partm11 aborts instead.
#   * `WGT_A=a` means w = a + nlev as a single scalar constraint: a=0 is pure 3-D, large a
#     approaches pure 2-D. The legacy arm is the DUAL constraint (WGT=2), which no external
#     engine can express — so this scalar is the axis the whole zoo shares.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
SRC=${SRC:-$SB/core2_base}
MESHTAG=${MESHTAG:-core2}
ZOO=$SB/zoo/$MESHTAG
RANKS=${RANKS:-"256 512"}
ARMS=${ARMS:-"a1 a2 a3_a0 a3_a15 a3_a40 a3_a100"}
OUT=/work/ab0995/a270088/port2/m11/zooa.${SLURM_JOB_ID:-manual}
CSV=$OUT/zoo_a_$MESHTAG.csv
mkdir -p "$OUT" "$ZOO"
fail=0

arm_knobs() {   # arm_knobs <arm> -> "VAR=val VAR=val ..."
    case $1 in
        a1)      echo "FESOM_PART_KWAY=1 FESOM_PART_WGT=0" ;;
        a2)      echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT=0" ;;
        a3_a0)   echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=0" ;;
        a3_a15)  echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=15" ;;
        a3_a40)  echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=40" ;;
        a3_a100) echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=100" ;;
        a4m)     echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=${A4_A:?A4_A required} FESOM_PART_MINCONN=1" ;;
        a4c)     echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=${A4_A:?A4_A required} FESOM_PART_CONTIG=1" ;;
        a4u30)   echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=${A4_A:?A4_A required} FESOM_PART_MINCONN=1 FESOM_PART_CONTIG=1 FESOM_PART_UFACTOR=30" ;;
        a4)      echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=${A4_A:?A4_A required} FESOM_PART_MINCONN=1 FESOM_PART_CONTIG=1" ;;
        a5_u10)  echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=${A5_A:?A5_A required} FESOM_PART_UFACTOR=10" ;;
        a5_u30)  echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=${A5_A:?A5_A required} FESOM_PART_UFACTOR=30" ;;
        a5_u100) echo "FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=${A5_A:?A5_A required} FESOM_PART_UFACTOR=100" ;;
        legacy)  echo "" ;;                       # the compiled defaults: dual+100, Recursive
        seedb)   echo "FESOM_PART_SEED=424242" ;; # second seed of the legacy arm
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

echo "=== M11 zoo A   mesh=$SRC   arms='$ARMS'   ranks='$RANKS'   $(date '+%F %T')"
echo "    partitioner  $(md5sum "$P/mesh_part/build/bin/fesom_meshpart" | cut -d' ' -f1) (exe)"
echo "    libC         $(md5sum "$P/mesh_part/build/lib64/libfesom_meshpart_C.so" | cut -d' ' -f1)"

for arm in $ARMS; do
    knobs=$(arm_knobs "$arm") || { echo "REFUSE: unknown arm '$arm'"; exit 2; }
    [ "$knobs" = UNKNOWN ] && { echo "REFUSE: unknown arm '$arm'"; exit 2; }
    M=$ZOO/$arm
    echo
    echo "=================================================================="
    echo "=== arm $arm   knobs: ${knobs:-<compiled defaults>}"
    if [ ! -f "$M/nod2d.out" ]; then
        rm -rf "$M"
        m11_sandbox_copy_mesh "$SRC" "zoo/$MESHTAG/$arm" || { fail=1; continue; }
    fi
    for N in $RANKS; do
        [ -d "$M/dist_$N" ] && { echo "  dist_$N exists, skipping"; continue; }
        # shellcheck disable=SC2086
        env $knobs MESH="$M" RANKS="$N" TAG="${arm}_$N" ROOT="$ROOT" OUT="$OUT" PARTROOT="$P" \
            BIN="$P/mesh_part/build/bin/fesom_meshpart" \
            bash "$ROOT/scripts/m11_partgen.sh" > "$OUT/gen_${arm}_$N.log" 2>&1
        rc=$?
        moved=$(grep -ac 'is moved to part' "$OUT/meshpart_${arm}_${N}_$N.log" 2>/dev/null || echo 0)
        if [ $rc -ne 0 ]; then
            echo "  N=$N rc=$rc FAILED"; sed -n '/REFUSE\|FAILED\|\[M11\]/p' "$OUT/gen_${arm}_$N.log" | head -5
            fail=1; continue
        fi
        echo "  N=$N ok, check_partitioning moved $moved node(s)"
        $PY "$ROOT/scripts/m11_scorecard.py" "$M" --dist "$N" --arm "${arm}_$N" --csv "$CSV" \
            > "$OUT/score_${arm}_$N.txt" 2>&1 || { echo "  N=$N scorecard FAILED"; fail=1; }
    done
done

echo
echo "=================================================================="
echo "=== scorecard summary"
$PY - "$CSV" <<'EOF'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(dict)
for r in rows:
    arm, n = r["arm"].rsplit("_", 1)
    by[int(n)][arm] = r
K = [("edgecut_unweighted", "cut_unw", "{:,.0f}"), ("commvol_total", "commvol", "{:,.0f}"),
     ("commvol_max_rank", "cv_max", "{:,.0f}"), ("n2d_imb", "2Dimb", "{:.3f}"),
     ("n3d_maxmin", "3Dmax/min", "{:.2f}"), ("nbr_max", "nbr", "{:.0f}"),
     ("parts_disconnected", "disc", "{:.0f}"), ("noncore_vertices", "offlobe", "{:,.0f}"),
     ("isolated_nodes", "iso", "{:.0f}"), ("halo_nod_mean", "halo", "{:.1f}"),
     ("elem_repl", "elrepl", "{:.4f}")]
for n in sorted(by):
    print(f"\n--- N={n}")
    print(f"  {'arm':<10}" + "".join(f"{h:>11}" for _, h, _ in K))
    for arm in sorted(by[n]):
        r = by[n][arm]
        print(f"  {arm:<10}" + "".join(f"{f.format(float(r[k])):>11}" for k, _, f in K))
EOF
m11_check_sources
echo; echo "=== zoo A: $([ $fail = 0 ] && echo PASS || echo "PASS with $fail failed arm(s)")  $(date '+%F %T')"
echo "    csv: $CSV"
exit 0
