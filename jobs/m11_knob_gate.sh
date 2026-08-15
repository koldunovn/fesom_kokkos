#!/bin/bash
#SBATCH --job-name=m11knob
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/port2/m11/knob.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/knob.%j.err
#
# M11 Task 4 — knob gate (L80: a knob that does not announce itself is a knob that can be
# silently dead, and a dead knob produces a null result that looks like a measurement).
#
# Every knob is exercised twice: once with a value it must ACCEPT and announce, once with a
# value it must REFUSE. Plus:
#   * arm A0 — the METIS 5.2.1 build's default output, which is NOT expected to match 5.1.0
#     and is scored and raced as a lever rather than asserted away (review B2);
#   * the check_partitioning reproduction — the +5 cut gap Task 2 predicted from the files
#     alone, now checked against the partitioner's own unfiltered stdout.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/scripts/m11_guards.sh"
SB=/work/ab0995/a270088/port2/mesh_m11
PA=/work/ab0995/a270088/port2/partm11/fesom2         # METIS 5.1.0
PB=/work/ab0995/a270088/port2/partm11/fesom2_b       # METIS 5.2.1
OUT=/work/ab0995/a270088/port2/m11/knob.${SLURM_JOB_ID:-manual}
MESH=$SB/knobgate
mkdir -p "$OUT"

m11_assert_sandbox "$MESH" "knob gate mesh" > /dev/null
m11_sources_init

set +u; source /sw/etc/profile.levante; source "$PA/env/levante.dkrz.de/shell"; set -u
ulimit -s unlimited
W=$OUT/work; mkdir -p "$W"; cd "$W"
for n in namelist.config namelist.forcing namelist.oce namelist.ice namelist.icepack; do
    [ -f "$PA/config/$n" ] && cp -f "$PA/config/$n" .
done

pass=0; fail=0
leg() {   # leg <expect ok|abort> <name> <bin> <ranks> <must-match regex> [ENV=V ...]
    local expect=$1 name=$2 bin=$3 n=$4 want=$5; shift 5
    local log="$OUT/$name.log" rc
    sed -i -E "s|^MeshPath *=.*|MeshPath         = '$MESH/'|" namelist.config
    sed -i -E "s|^n_levels *=.*|n_levels = 1|"                namelist.config
    sed -i -E "s|^n_part *=.*|n_part   = $n|"                 namelist.config
    m11_assert_namelist_meshpath namelist.config "$MESH" > /dev/null
    rm -rf "$MESH/dist_$n"
    export LD_LIBRARY_PATH="$(dirname "$bin")/../lib64:${LD_LIBRARY_PATH:-}"
    env "$@" srun -l "$bin" > "$log" 2>&1
    rc=$?
    local ok=0
    if [ "$expect" = ok ]    && [ $rc -eq 0 ] && grep -aqE "$want" "$log"; then ok=1; fi
    if [ "$expect" = abort ] && [ $rc -ne 0 ] && grep -aqE "$want" "$log"; then ok=1; fi
    if [ $ok -eq 1 ]; then
        pass=$((pass+1)); printf "  PASS  %-34s expect=%-5s rc=%-3s | %s\n" "$name" "$expect" "$rc" \
            "$(grep -aoE "$want" "$log" | head -1)"
    else
        fail=$((fail+1)); printf "  FAIL  %-34s expect=%-5s rc=%-3s\n" "$name" "$expect" "$rc"
        grep -aE "^\[M11\]|METIS|error" "$log" | head -4 | sed 's/^/         /'
    fi
    m11_check_sources > /dev/null
}

BA=$PA/mesh_part/build/bin/fesom_meshpart
BB=$PB/mesh_part/build/bin/fesom_meshpart

echo "=== A. the 5.2.1 build announces its version (arm A0) ==="
leg ok    A0_metis521          "$BB" 8 "Metis version 5\.2\.1"
echo
echo "=== B. every knob announces on a value it accepts ==="
leg ok    kway1                "$BA" 8 "\[M11\] FESOM_PART_KWAY=1"                 FESOM_PART_KWAY=1
leg ok    obj_vol_with_kway    "$BA" 8 "COMMUNICATION VOLUME"                      FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol
leg ok    vsize                "$BA" 8 "vsize = nlev passed to METIS"              FESOM_PART_VSIZE=1
leg ok    minconn              "$BA" 8 "\[M11\] FESOM_PART_MINCONN=1"              FESOM_PART_KWAY=1 FESOM_PART_MINCONN=1
leg ok    contig               "$BA" 8 "connected component"                       FESOM_PART_KWAY=1 FESOM_PART_CONTIG=1
leg ok    ufactor              "$BA" 8 "\[M11\] FESOM_PART_UFACTOR=30"             FESOM_PART_UFACTOR=30
leg ok    ncuts                "$BA" 8 "\[M11\] FESOM_PART_NCUTS=3"                FESOM_PART_NCUTS=3
leg ok    niter                "$BA" 8 "\[M11\] FESOM_PART_NITER=5"                FESOM_PART_NITER=5
leg ok    seed                 "$BA" 8 "\[M11\] FESOM_PART_SEED=999"               FESOM_PART_SEED=999
leg ok    wgt0                 "$BA" 8 "FESOM_PART_WGT=0 overrides"                FESOM_PART_WGT=0
leg ok    wgt_a100             "$BA" 8 "single scalar weight w = 100 \+ nlev"      FESOM_PART_WGT_A=100
leg ok    wgt_a0               "$BA" 8 "single scalar weight w = 0 \+ nlev"        FESOM_PART_WGT_A=0
echo
echo "=== C. every knob refuses a value it does not recognise ==="
leg abort kway_2               "$BA" 8 "FESOM_PART_KWAY=2 out of range"            FESOM_PART_KWAY=2
leg abort kway_junk            "$BA" 8 "is not an integer"                         FESOM_PART_KWAY=yes
leg abort obj_junk             "$BA" 8 "FESOM_PART_OBJ='sideways' unrecognised"    FESOM_PART_OBJ=sideways
leg abort obj_vol_no_kway      "$BA" 8 "Kway-only and are SILENTLY IGNORED"        FESOM_PART_OBJ=vol
leg abort contig_no_kway       "$BA" 8 "Kway-only and are SILENTLY IGNORED"        FESOM_PART_CONTIG=1
leg abort minconn_no_kway      "$BA" 8 "Kway-only and are SILENTLY IGNORED"        FESOM_PART_MINCONN=1
leg abort ufactor_0            "$BA" 8 "FESOM_PART_UFACTOR=0 must be >= 1"         FESOM_PART_UFACTOR=0
leg abort wgt_5                "$BA" 8 "FESOM_PART_WGT=5 out of range"             FESOM_PART_WGT=5
leg abort wgt_a_negative       "$BA" 8 "negative means 'legacy dual'"              FESOM_PART_WGT_A=-1
leg abort partfile_missing     "$BA" 8 "cannot be opened"                          FESOM_PART_FILE=/nonexistent/p.txt
leg abort tpwgts_missing       "$BA" 8 "cannot be opened"                          FESOM_PART_TPWGTS_FILE=/nonexistent/t.txt
echo
echo "=== D. a short part vector must be refused, not padded ==="
head -100 "$OUT/../null.$(ls /work/ab0995/a270088/port2/m11 | grep -o 'null\.[0-9]*' | tail -1 | cut -d. -f2)/inject_8.txt" \
    > "$OUT/short.txt" 2>/dev/null || seq 0 99 > "$OUT/short.txt"
leg abort partfile_short       "$BA" 8 "entries could be read"                     FESOM_PART_FILE="$OUT/short.txt"

echo
echo "=== E. check_partitioning reproduction: CORE2 wgt2 at 16 ranks ==="
echo "    Task 2 predicted from the FILES that the post-pass moves gid 125423 (part 9->8),"
echo "    so METIS should print 217791 while the dist on disk scores 217796."
leg ok    checkpart_wgt2_16    "$BA" 16 "METIS edgecut 217791"                     FESOM_PART_WGT=2
grep -aE "Isolated node|Neighbouring nodes|moved to partition|New partition" \
    "$OUT/checkpart_wgt2_16.log" | head -8 | sed 's/^/    /'

echo
echo "=== knob gate: $pass PASS, $fail FAIL ==="
exit $((fail > 0))
