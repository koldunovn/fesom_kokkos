#!/bin/bash
#SBATCH --job-name=m11detgate
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:15:00
#SBATCH -o /work/ab0995/a270088/port2/m11/detgate.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/detgate.%j.err
#
# M13-on-M11: certify the re-pinned Serial binary before any partition re-run uses it.
#
# Two legs, and BOTH are required, because each one alone can be passed by a broken build:
#
#   A. knob OFF  -> byte-identical to the certified M6/M7 Serial baseline. This is what makes
#      "the only thing that changed is the initial condition" a fact rather than a hope: the
#      new bin is h17-equivalent everywhere except behind the knob.
#   B. knob ON (FESOM_IC_EXTRAP=det) -> MUST differ from that baseline. Without leg B a binary
#      that ignored the environment variable would sail through leg A (the dead-knob trap, L80),
#      and every "det" re-run would silently be a legacy re-run.
#
# Same configuration as jobs/job_m7_gate_serial, which produced the baseline: private CORE2
# mesh, dist_8, 8 ranks, dt 1800, 20 steps, snap every 10, year 1958.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source /sw/etc/profile.levante
source "$ROOT/env.sh"
export OMPI_MCA_btl_vader_single_copy_mechanism=none    # L18 deterministic gather

MESH=/work/ab0995/a270088/port2/mesh/core2                # private mesh, never /pool (L73)
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
BASELINE=/work/ab0995/a270088/port2/m6_baseline_serial
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
BIN=${BIN:-/work/ab0995/a270088/port2/m11/bin/det1/fesom_port_serial}

unset FESOM_ALE FESOM_MIX_SCHEME FESOM_WHICH_EVP
unset FESOM_TKE_DIAG FESOM_TKE_DUMP_DIR FESOM_EVP_DUMP_DIR FESOM_ALE_DUMP_DIR
unset FESOM_IC_EXTRAP FESOM_IC_EXTRAP_TOL
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')

OUT=/work/ab0995/a270088/port2/m11/detgate.${SLURM_JOB_ID}
rm -rf "$OUT"; mkdir -p "$OUT/off" "$OUT/det"
echo "=== M11 det-bin gate  BIN=$BIN"
echo "    md5=$(md5sum "$BIN" | cut -d' ' -f1)   HEAD=$(cd "$ROOT" && git rev-parse --short HEAD)"
echo "    $(date '+%F %T')"

echo
echo "--- leg A: knob OFF, must be BYTE-IDENTICAL to the certified baseline ---"
srun -l "$BIN" "$MESH" "$OUT/off" 1800 20 10 "$PHC" 1958 > "$OUT/off.log" 2> "$OUT/off.err"
echo "run rc=$?"
grep -a "FESOM_IC_EXTRAP" "$OUT/off.log" | head -2
"$PY" "$ROOT/scripts/diff_snap.py" "$BASELINE" "$OUT/off"; RC_OFF=$?
echo "leg A diff_snap rc=$RC_OFF   (0 = identical = PASS)"

echo
echo "--- leg B: FESOM_IC_EXTRAP=det, must DIFFER from the baseline (proof the knob fires) ---"
FESOM_IC_EXTRAP=det srun -l "$BIN" "$MESH" "$OUT/det" 1800 20 10 "$PHC" 1958 \
    > "$OUT/det.log" 2> "$OUT/det.err"
echo "run rc=$?"
grep -a "FESOM_IC_EXTRAP\|deterministic" "$OUT/det.log" | head -3
"$PY" "$ROOT/scripts/diff_snap.py" "$BASELINE" "$OUT/det"; RC_DET=$?
echo "leg B diff_snap rc=$RC_DET   (nonzero = the knob changed the IC = PASS)"

echo
if [ "$RC_OFF" -eq 0 ] && [ "$RC_DET" -ne 0 ]; then
    echo "=== DET-BIN GATE PASS — default path untouched, knob demonstrably live ==="
    exit 0
fi
echo "=== DET-BIN GATE FAIL — legA(want 0)=$RC_OFF legB(want !=0)=$RC_DET ==="
exit 1
