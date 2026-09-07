#!/usr/bin/env bash
# M16 Task B6c gate — FP64 restart round-trip on pi np2 (seconds; login node is fine):
#   A  = straight run N+M steps;  D1 = N steps writing a restart;  D2 = M steps from D1's restart.
# D2's final snapshot must be BYTE-IDENTICAL to A's (the m14 restart gate's own claim).
#   scripts/m16_restart_gate.sh <build-dir-name | /abs/bin> [N=10] [M=10]
set -u
SRC=/home/a/a270088/port_kokkos_sp; PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
MESH=/home/a/a270088/port2/fesom2/test/meshes/pi
build=${1:?}; N=${2:-10}; M=${3:-10}
if [ -f "$build" ]; then BIN=$build; tag=$(basename "$(dirname "$build")"); else BIN=$SRC/$build/fesom_port; tag=$build; fi
ROOT=/work/ab0995/a270088/port2/m16/gate0/restart_$tag; rm -rf "$ROOT"; mkdir -p "$ROOT"/{A,D1,D2}
source /sw/etc/profile.levante >/dev/null 2>&1 || true; source "$SRC/env.sh" >/dev/null 2>&1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_[A-Za-z0-9_]*\)=.*/\1/p')
export FESOM_SSH_PRECOND=0 FESOM_ALE=zstar        # zstar: the stiffness matrix is part of the restart
run(){ local o=$1 ns=$2; shift 2; ( for kv in "$@"; do export "$kv"; done; mpirun -np 2 "$BIN" "$MESH" "$o" 100 $ns $ns > "$o/run.log" 2> "$o/run.err"; echo $? > "$o/rc" ); cat "$o/rc"; }
echo "A  rc=$(run $ROOT/A  $((N+M)) FESOM_RESTART_OUT=$ROOT/A FESOM_RESTART_EVERY=0 FESOM_RESTART_AT=$N)"
echo "D1 rc=$(run $ROOT/D1 $N       FESOM_RESTART_OUT=$ROOT/D1 FESOM_RESTART_EVERY=0)"
RST=$(ls "$ROOT"/D1/*.restart.nc 2>/dev/null | head -1); [ -n "$RST" ] || { echo "no restart written by D1"; exit 1; }
echo "D2 rc=$(run $ROOT/D2 $M       FESOM_RESTART_IN=$RST FESOM_RESTART_OUT=$ROOT/D2 FESOM_RESTART_EVERY=0)"
# compare the LAST snapshot of A with the last of D2 (snapshot names differ by step index)
a=$(ls "$ROOT"/A/snap_*.nc | tail -n 1); d=$(ls "$ROOT"/D2/snap_*.nc | tail -n 1); mkdir -p "$ROOT/cmpA" "$ROOT/cmpD"; cp "$a" "$ROOT/cmpA/snap_final.nc"; cp "$d" "$ROOT/cmpD/snap_final.nc"
"$PY" "$SRC/scripts/diff_snap.py" "$ROOT/cmpA" "$ROOT/cmpD" | tail -n 3 && echo "=== RESTART ROUND-TRIP PASS ($tag)" || { echo "=== RESTART ROUND-TRIP FAIL ($tag)"; exit 1; }
