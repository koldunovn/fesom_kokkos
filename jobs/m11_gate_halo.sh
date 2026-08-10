#!/bin/bash
#SBATCH --job-name=m11halo
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH -o /work/ab0995/a270088/port2/m11/gate_halo.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/gate_halo.%j.err
#
# M11 Task 1 — prove the halo/dist correctness gate is not vacuous.
#
# leg A (control) : pristine sandbox CORE2 dist_4 -> fesom_halo_identity_test PASSES
# leg B (negative): the SAME mesh with two nod2D rlist entries swapped in
#                   com_info00000.out (scripts/m11_corrupt_com_info.py) -> the test
#                   must ABORT. A gate that has never been seen to fail cannot be
#                   quoted as evidence for a generated partition.
#
# The binary is the frozen certified h17 Serial build, pinned by md5 (BIN=).
# Nothing here writes to a production mesh: both legs live in mesh_m11/.
set -u
ROOT=/home/a/a270088/port_kokkos_part          # worktree, NOT the main repo
source "$ROOT/scripts/m11_guards.sh"
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800

BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial}
BIN_MD5_EXPECT=5c3c90fc0ea3939df86cfbe275287c36
MESH_OK=${MESH_OK:-/work/ab0995/a270088/port2/mesh_m11/core2_m11}
MESH_BAD=${MESH_BAD:-/work/ab0995/a270088/port2/mesh_m11/gate_negctl}
OUT=/work/ab0995/a270088/port2/m11/gate_halo.${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"

md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = "$BIN_MD5_EXPECT" ] || { echo "BIN md5 $md5 != certified h17 Serial $BIN_MD5_EXPECT"; exit 2; }
echo "=== M11 halo/dist gate — BIN=$BIN md5=$md5 ntasks=$SLURM_NTASKS ==="

# Both legs must be inside the sandbox, and both must still match their manifests
# (so a stale/half-copied mesh cannot masquerade as a gate result).
m11_assert_sandbox "$MESH_OK"  "control mesh"   > /dev/null
m11_assert_sandbox "$MESH_BAD" "negative mesh"  > /dev/null
m11_md5_check "$MESH_OK"
m11_md5_check "$MESH_BAD"

DT=1800; NSTEPS=2; SNAP=-1
run_leg() {   # run_leg <name> <mesh>
    local name=$1 mesh=$2 rc
    local o="$OUT/$name"; mkdir -p "$o"
    srun -l "$BIN" "$mesh" "$o" "$DT" "$NSTEPS" "$SNAP" > "$OUT/log_$name.txt" 2>&1
    rc=$?
    echo "leg $name: rc=$rc mesh=$mesh"
    grep -E "identity test|FESOM (DIE|ABORT)|halo\[" "$OUT/log_$name.txt" | head -8
    return $rc
}

fail=0
echo; echo "--- leg A: control (pristine dist_4) — expect PASS"
if run_leg control "$MESH_OK"; then
    grep -q "identity test (positive): all halo entries carry correct gid" "$OUT/log_control.txt" \
        || { echo "VERDICT-FAIL: control ran but the identity test never announced itself"; fail=1; }
else
    echo "VERDICT-FAIL: control leg aborted on a healthy dist"; fail=1
fi

echo; echo "--- leg B: negative control (2 nod2D rlist entries swapped) — expect ABORT"
if run_leg corrupt "$MESH_BAD"; then
    echo "VERDICT-FAIL: the corrupted dist RAN CLEAN — the gate is vacuous"; fail=1
else
    if grep -qE "identity test: .* halo nodes mismatched|rank .* FAIL: halo\[" "$OUT/log_corrupt.txt"; then
        echo "VERDICT-OK: gate fired on the corrupted com_info"
    else
        echo "VERDICT-FAIL: corrupt leg died, but NOT via the identity test — check the log"
        tail -20 "$OUT/log_corrupt.txt"; fail=1
    fi
fi

echo; m11_check_sources
echo; echo "=== M11 halo gate verdict: $([ $fail = 0 ] && echo PASS || echo FAIL) (logs in $OUT) ==="
exit $fail
