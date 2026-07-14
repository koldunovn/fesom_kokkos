#!/bin/bash
# M7 — wait for the row-0 jobs to drain, rebuild both backends, fire the Tier-1 gate ladder.
#
# The row-0 baseline jobs (26234870/26234872) execute whatever binary is on disk WHEN THEY
# START, so build-m7cuda must not change until they are gone. Everything submitted since then
# is pinned to the frozen snapshot in m7/bin/row0/ and is immune. This script blocks on exactly
# those two, then rebuilds and gates.
#
# Usage: bash jobs/m7_rebuild_and_gate.sh [blocker-jobid ...]
set -u
ROOT=/home/a/a270088/port_kokkos
cd "$ROOT"
BLOCKERS="${*:-26234870 26234872}"
LOG=/work/ab0995/a270088/port2/m7/rebuild_and_gate.log
exec > >(tee -a "$LOG") 2>&1
echo "=========== $(date -Is)  m7_rebuild_and_gate ==========="
echo "waiting on blockers: $BLOCKERS"

while true; do
    live=$(squeue -u a270088 -h -o "%i" 2>/dev/null | tr '\n' ' ')
    still=""
    for j in $BLOCKERS; do [[ " $live " == *" $j "* ]] && still="$still $j"; done
    [ -z "$still" ] && break
    sleep 60
done
echo "blockers cleared at $(date -Is) — rebuilding"

# ⚠️ NEVER source env.sh and env_cuda.sh in the same shell (L77) — separate subshells.
( source "$ROOT/env_cuda.sh" >/dev/null 2>&1 && cmake --build "$ROOT/build-m7cuda"  -j 16 ) \
    > /tmp/m7_build_cuda.log 2>&1
rc_cu=$?
( source "$ROOT/env.sh"      >/dev/null 2>&1 && cmake --build "$ROOT/build-m7serial" -j 16 ) \
    > /tmp/m7_build_ser.log 2>&1
rc_se=$?
echo "build rc: cuda=$rc_cu serial=$rc_se"
if [ $rc_cu -ne 0 ] || [ $rc_se -ne 0 ]; then
    echo "!! BUILD FAILED — not gating. Tail:"
    tail -25 /tmp/m7_build_cuda.log /tmp/m7_build_ser.log
    exit 1
fi
echo "new binaries:"; md5sum build-m7cuda/fesom_port build-m7serial/fesom_port

NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars

echo
echo "--- 1. knob-OFF byte gate (the campaign-long invariant, required at EVERY commit) ---"
sbatch --export=ALL,M7_TAG=t1_knoboff jobs/job_m7_gate_serial

echo "--- 2. FORCE_SERIAL byte proof: ICEFLUXDEV claims bit-identity -> prove it ---"
sbatch --export=ALL,M7_TAG=t10_proof,KNOBS="FESOM_SPEED_FORCE_SERIAL=1;FESOM_SPEED_ICEFLUXDEV=1" \
       jobs/job_m7_gate_serial
#   (NOFENCE2 gets NO byte proof: the device halo path is inside #ifdef KOKKOS_ENABLE_CUDA,
#    so the Serial backend never executes it. It relies on the audit + racecheck + fidelity gate.)

echo "--- 3. CUDA fidelity gates, knob-ON (~35 s each) ---"
sbatch --export=ALL,M7_TAG=t10_icefluxdev,KNOBS="FESOM_SPEED_ICEFLUXDEV=1" jobs/job_m7_gpu_gate
sbatch --export=ALL,M7_TAG=t11_nofence2,KNOBS="FESOM_SPEED_NOFENCE2=1"     jobs/job_m7_gpu_gate
sbatch --export=ALL,M7_TAG=t1_both,KNOBS="FESOM_SPEED_ICEFLUXDEV=1;FESOM_SPEED_NOFENCE2=1" \
       jobs/job_m7_gpu_gate

echo "--- 4. same-allocation A/Bs (the payoff numbers) ---"
sbatch -N 4 --ntasks=16 --export=ALL,MESH=$NG5,DT=180,TAG=ab_icefluxdev_ng5_4n,KNOBS="FESOM_SPEED_ICEFLUXDEV=1" jobs/job_m7_ab_gpu
sbatch -N 4 --ntasks=16 --export=ALL,MESH=$NG5,DT=180,TAG=ab_nofence2_ng5_4n,KNOBS="FESOM_SPEED_NOFENCE2=1"     jobs/job_m7_ab_gpu
sbatch -N 8 --ntasks=32 --export=ALL,MESH=$DARS,DT=180,TAG=ab_both_dars_8n,KNOBS="FESOM_SPEED_ICEFLUXDEV=1;FESOM_SPEED_NOFENCE2=1" jobs/job_m7_ab_gpu

echo
echo "=== submitted. Harvest with: ==="
echo "  grep -h 'GATE\|A/B RESULT' -A6 /work/ab0995/a270088/port2/m7/{gate,gpugate,ab}.*.out"
squeue -u a270088 -o "%.10i %.14j %.2t %R"
