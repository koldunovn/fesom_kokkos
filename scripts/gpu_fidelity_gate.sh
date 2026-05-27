#!/bin/bash
# scripts/gpu_fidelity_gate.sh — the CORE2-active-ice CUDA fidelity GATE (M5.9).
#
# RUN THIS BEFORE COMMITTING any device-halo / sync-rail / device-residency change. The pi A/B gate
# is INSUFFICIENT (pi has no ice + idealised dynamics → a stale-host bug stays ~1e-17 and is invisible;
# that is exactly how the M5.9 CORE2 staleness bug slipped past M5.1–M5.8). This runs CORE2 dist_8 on
# build-cuda (device-halo) + diffs vs the build-serial oracle (bit-identical to the C twin = ground
# truth). PASS = the CUDA climate-close floor (~1e-3); FAIL = a staleness/correctness regression.
#
#   ./scripts/gpu_fidelity_gate.sh                # uses the cached Serial oracle
#   ./scripts/gpu_fidelity_gate.sh --fresh-oracle # rebuild the oracle (after a Serial-code change)
# ⚠️ Build both backends first (build-serial AND build-cuda). ⚠️ If the CUDA run crashes (intermittent
# device-ptr UCX flakiness), just re-run — the check reports FAIL on a missing snapshot.
set -e
ROOT=/home/a/a270088/port_kokkos
SREF=/work/ab0995/a270088/port2/kokkos_gpu_runs/serref_core2
DEV=/work/ab0995/a270088/port2/kokkos_gpu_runs/gate_dev
[ "$1" = "--fresh-oracle" ] && rm -rf "$SREF"
if [ ! -f "$SREF/snap_000020.nc" ]; then
  echo "[gate] building the Serial oracle (build-serial CORE2 dist_8, bit-identical to C)..."
  sbatch --wait "$ROOT/jobs/job_core2_serial_ref"
fi
echo "[gate] running the CUDA device-halo leg..."
sbatch --wait "$ROOT/jobs/job_gpu_fidelity_dev"
python "$ROOT/scripts/gpu_fidelity_check.py" "$SREF" "$DEV"
