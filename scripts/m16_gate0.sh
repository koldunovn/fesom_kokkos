#!/usr/bin/env bash
# M16 Gate 0 — FP64 byte gate against the ref0 oracle (plan D9/G0, Task P2).
#
#   scripts/m16_gate0.sh <build|/abs/path/to/fesom_port> <config|all> [np]
#
# <build>  a build dir name under ~/port_kokkos_sp (build-m16-ref0, build-m16serial, ...) or an
#          absolute path to a binary (e.g. the frozen i4 Serial bin). ref0 populates the oracle.
# <config> one of the names in CONFIGS below, or "all".
# [np]     ranks, default 1 (pi has dist_2 and dist_8; login node runs np 1/2 via vader).
#
# Runs the config with EVERY FESOM_* knob unset, then FESOM_SSH_PRECOND=0 exported (D3: the M16
# default is variant 1, so the byte gate must pin variant 0), then the config's own knobs, on the
# pi mesh with analytical forcing, 20 steps at dt 100, a snapshot EVERY step (Z7/L78: a constant
# array made time-varying is bitwise at step 1 and breaks at step 2). Then diff_snap.py (zero
# tolerance) vs $ROOT/ref0/<config>_np<N>. Exit != 0 on a non-zero run rc OR any differing byte
# (the M14 gate once passed two segfaults — never trust a diff without the rc).
#
# Output root: /work (standing rule) — override with M16_ROOT for the D10 read-only-/work case.
set -u
SRC=/home/a/a270088/port_kokkos_sp
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
# M16_PRESET=pi (default): pi mesh, analytical forcing, dt 100, 20 steps, snapshot every step.
# M16_PRESET=core2: private CORE2 mesh (NEVER /pool), PHC init + JRA55 1958, dt 1800, 20 steps,
#   snapshots at 10 and 20 (Z7 satisfied; every-step would be ~10 GB per config) — the oracle for
#   the Phase-B slices pi never executes (forcing, bulk, SSS restoring, PHC/det init). Login np1 is
#   ~3.5 min per config; np2 needs dist_2 (present).
PRESET=${M16_PRESET:-pi}
case "$PRESET" in
  pi)    ROOT=${M16_ROOT:-/work/ab0995/a270088/port2/m16/gate0}
         MESH=/home/a/a270088/port2/fesom2/test/meshes/pi; DT=100; NSTEPS=20; SNAP=1; EXTRA=();;
  core2) ROOT=${M16_ROOT:-/work/ab0995/a270088/port2/m16/gate0_core2}
         MESH=/work/ab0995/a270088/port2/mesh/core2; DT=1800; NSTEPS=20; SNAP=10
         EXTRA=(/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc 1958);;
  *) echo "M16_PRESET must be pi|core2"; exit 2;;
esac

# config -> knobs (semicolon separated). Serial speed levers need FESOM_SPEED_FORCE_SERIAL=1.
declare -A CONFIGS=(
  [default]=""
  [mevp]="FESOM_WHICH_EVP=1"
  [zstar]="FESOM_ALE=zstar"
  [tke]="FESOM_MIX_SCHEME=TKE"
  [se]="FESOM_ALE=zstar;FESOM_SSH_MODE=se;FESOM_SE_M=50"
  [sewide]="FESOM_ALE=zstar;FESOM_SSH_MODE=se;FESOM_SE_M=50;FESOM_SE_WIDE=1"
  [evpwlean]="FESOM_WHICH_EVP=1;FESOM_SPEED=1;FESOM_SPEED_FORCE_SERIAL=1;FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_EVPWIDE_LEAN=1"
  [cg2]="FESOM_SSH_SOLVER=cg2"
  [pipecg]="FESOM_SSH_SOLVER=pipecg"
  [oati]="FESOM_SSH_SOLVER=oati"
  [pcsi]="FESOM_SSH_SOLVER=pcsi"
  [det]="FESOM_IC_EXTRAP=det"
)
G0_FIVE="default mevp zstar tke se"
ALL="$G0_FIVE sewide evpwlean cg2 pipecg oati pcsi det"

build=${1:?build dir name or binary path}; cfg=${2:?config or all}; np=${3:-1}
if [ -f "$build" ]; then BIN=$build; tag=$(basename "$(dirname "$build")")_$(basename "$build")
else BIN=$SRC/$build/fesom_port; tag=$build; fi
[ -x "$BIN" ] || { echo "no binary: $BIN"; exit 2; }
is_ref0=0; [ "$tag" = build-m16-ref0 ] && { is_ref0=1; tag=ref0; }

# environment: login node (vader) or inside a SLURM allocation
source /sw/etc/profile.levante >/dev/null 2>&1 || true
source "$SRC/env.sh" >/dev/null 2>&1
export OMPI_MCA_btl_vader_single_copy_mechanism=none            # L18 deterministic gather
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
  unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL \
        HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
  LAUNCH="mpirun -np $np"
else
  LAUNCH="srun -n $np"
fi

run_one() {
  local c=$1 out=$ROOT/$tag/${c}_np$np ref=$ROOT/ref0/${c}_np$np
  [ -n "${CONFIGS[$c]+x}" ] || { echo "unknown config $c"; return 2; }
  if [ "$c" = det ] && [ "$PRESET" = pi ]; then echo "[$tag/$c np$np] skipped: the det IC fill only runs with a PHC init (core2 preset)"; return 0; fi
  if [ "$c" = evpwlean ] && [ "$np" -lt 2 ]; then echo "[$tag/$c np$np] skipped: the wide halo builds no extended zone at np1 (M9 FATAL by design)"; return 0; fi
  rm -rf "$out"; mkdir -p "$out"
  ( # subshell: knob hygiene per config
    while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_[A-Za-z0-9_]*\)=.*/\1/p')
    export FESOM_SSH_PRECOND=0                                    # D3
    if [ -n "${CONFIGS[$c]}" ]; then IFS=';' read -ra KV <<< "${CONFIGS[$c]}"; for kv in "${KV[@]}"; do export "$kv"; done; fi
    { echo "BIN=$BIN"; md5sum "$BIN"; env | grep -E '^FESOM_' | sort; } > "$out/ENV.txt"
    $LAUNCH "$BIN" "$MESH" "$out" $DT $NSTEPS $SNAP "${EXTRA[@]}" > "$out/run.log" 2> "$out/run.err"
    echo $? > "$out/rc"
  )
  local rc; rc=$(cat "$out/rc")
  if [ "$rc" != 0 ]; then echo "[$tag/$c np$np] RUN rc=$rc  ($out/run.err)"; tail -3 "$out/run.err"; return 1; fi
  local nsnap; nsnap=$(ls "$out"/snap_*.nc 2>/dev/null | wc -l)
  [ "$nsnap" -ge 2 ] || { echo "[$tag/$c np$np] only $nsnap snapshots (need >=2, Z7)"; return 1; }
  if [ $is_ref0 = 1 ]; then echo "[$tag/$c np$np] ORACLE written: $nsnap snapshots"; fi
  [ -d "$ref" ] || { echo "[$tag/$c np$np] no oracle $ref"; return 1; }
  "$PY" "$SRC/scripts/diff_snap.py" "$ref" "$out" > "$out/diff.txt" 2>&1
  local drc=$?
  if [ $drc = 0 ]; then echo "[$tag/$c np$np] BYTE-IDENTICAL to ref0 ($nsnap snapshots)"; return 0
  else echo "[$tag/$c np$np] DIFFERS from ref0:"; head -8 "$out/diff.txt"; return 1; fi
}

case "$cfg" in
  all)  list=$ALL;;
  five) list=$G0_FIVE;;
  *)    list=$cfg;;
esac
fail=0
for c in $list; do run_one "$c" || fail=1; done
[ $fail = 0 ] && echo "=== GATE 0 PASS ($tag np$np: $list) ===" || echo "=== GATE 0 FAIL ($tag np$np) ==="
exit $fail
