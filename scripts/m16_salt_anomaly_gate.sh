#!/usr/bin/env bash
# M16 Phase D gate — S_ref invariance + restart detection for FESOM_SALT_ANOMALY (pi np2, seconds).
#   For each of {default (linfs), zstar}: three runs off / =1 (S_ref 35) / =10, 20 steps dt 100,
#   snapshots at 10 and 20; diff_snap pairwise. A missed consumer shows as an O(1) error that
#   scales with S_ref; a correct port agrees to the rounding class. DP PASS = max|Δ| on T, S, eta
#   below TOL (default 1e-8) on every pair and off ⇒ identical to a knob-free run of the same
#   binary. SP (banner SINGLE): reported only (the same tolerance is meaningless in float).
#   Restart (zstar): (a) off-run restart @10 read WITH the knob → "converted" line + rounding-class
#   vs the straight on-run; (b) on-run restart @10 read with the knob → "no conversion" + BIT-IDENTICAL.
#   scripts/m16_salt_anomaly_gate.sh <build-dir|/abs/bin> [TOL=1e-8]
set -u
SRC=/home/a/a270088/port_kokkos_sp; PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
MESH=/home/a/a270088/port2/fesom2/test/meshes/pi
build=${1:?}; TOL=${2:-1e-8}
if [ -f "$build" ]; then BIN=$build; tag=$(basename "$(dirname "$build")"); else BIN=$SRC/$build/fesom_port; tag=$build; fi
ROOT=/work/ab0995/a270088/port2/m16/gate0/saltanom_$tag; rm -rf "$ROOT"; mkdir -p "$ROOT"
source /sw/etc/profile.levante >/dev/null 2>&1 || true; source "$SRC/env.sh" >/dev/null 2>&1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_[A-Za-z0-9_]*\)=.*/\1/p')
export FESOM_SSH_PRECOND=0
run(){ local o=$1 ns=$2 snap=$3; shift 3; mkdir -p "$o"; ( for kv in "$@"; do export "$kv"; done; env | grep -E '^FESOM_' | sort > "$o/ENV.txt"
       mpirun -np 2 "$BIN" "$MESH" "$o" 100 $ns $snap > "$o/run.log" 2> "$o/run.err"; echo $? > "$o/rc" ); cat "$o/rc"; }
maxd(){ # $1 dirA $2 dirB -> "T=.. S=.. eta=.. worst=.." from diff_snap (all snapshots); prints 0 for identical
  local out; out=$("$PY" "$SRC/scripts/diff_snap.py" "$1" "$2" 2>&1 | grep -v getfattr)
  if echo "$out" | grep -q 'BIT-IDENTICAL'; then echo "T=0 S=0 eta=0 (bit-identical)"; return; fi
  echo "$out" | awk '/max-abs-diff/ { split($1,a,":"); v=$2; sub("max-abs-diff=","",v); m[a[1]]=(v+0>m[a[1]]+0)?v:m[a[1]] }
      END { printf "T=%s S=%s eta=%s", (m["T"]==""?0:m["T"]), (m["S"]==""?0:m["S"]), (m["eta_n"]==""?0:m["eta_n"]) }'
}
within(){ awk -v s="$1" -v t="$2" 'BEGIN { n=split(s,a," "); ok=1; for(i=1;i<=n;i++){ split(a[i],kv,"="); if (kv[2]+0 > t+0) ok=0 } print ok }'; }
want=DOUBLE; fail=0
for cfg in default zstar; do
  extra=(); [ $cfg = zstar ] && extra=(FESOM_ALE=zstar)
  for leg in off:0 s35:1 s10:10; do IFS=: read -r name val <<< "$leg"
    o=$ROOT/${cfg}_$name; k=(); [ "$val" != 0 ] && k=(FESOM_SALT_ANOMALY=$val)
    rc=$(run "$o" 20 10 "${extra[@]}" "${k[@]}"); ns=$(ls "$o"/snap_*.nc 2>/dev/null | wc -l)
    line=$(grep -o 'use_salt_anomaly: salinity state = S - [0-9.]*' "$o/run.log" | head -1)
    echo "[$tag/$cfg/$name] rc=$rc snaps=$ns ${line:-"(no anomaly line)"}"
    [ "$rc" = 0 ] && [ "$ns" -ge 2 ] || fail=1
    if [ "$val" = 1 ] && [ "$line" != "use_salt_anomaly: salinity state = S - 35" ]; then echo "    knob did not FIRE (L80)"; fail=1; fi
  done
  grep -q 'PRECISION: SINGLE' "$ROOT/${cfg}_off/run.log" && want=SINGLE
  for pair in off:s35 off:s10 s35:s10; do IFS=: read -r a b <<< "$pair"
    d=$(maxd "$ROOT/${cfg}_$a" "$ROOT/${cfg}_$b"); ok=$(within "${d%% (*}" "$TOL")
    if [ $want = DOUBLE ]; then echo "    $cfg $a vs $b: $d  $([ "$ok" = 1 ] && echo OK || { echo "EXCEEDS $TOL"; fail=1; })"
    else echo "    $cfg $a vs $b: $d  (SP: reported)"; fi
  done
done
# off ⇒ byte-identical to the same binary with the variable UNSET vs "0"
o=$ROOT/default_off0; rc=$(run "$o" 20 10 FESOM_SALT_ANOMALY=0); d=$(maxd "$ROOT/default_off" "$o"); echo "    default unset vs =0: $d"; [[ "$d" == *bit-identical* ]] || fail=1
# restart legs (zstar)
A=$ROOT/rst_off_w;  rc=$(run "$A" 10 10 FESOM_ALE=zstar FESOM_RESTART_OUT=$A FESOM_RESTART_EVERY=0)
RA=$(ls "$A"/*.restart.nc | head -1)
Da=$ROOT/rst_off_to_on; rc=$(run "$Da" 10 10 FESOM_ALE=zstar FESOM_SALT_ANOMALY=1 FESOM_RESTART_IN=$RA FESOM_RESTART_EVERY=0)
conv=$(grep -c 'absolute-salinity restart detected -> converted' "$Da/run.log")
B=$ROOT/rst_on_w;   rc=$(run "$B" 10 10 FESOM_ALE=zstar FESOM_SALT_ANOMALY=1 FESOM_RESTART_OUT=$B FESOM_RESTART_EVERY=0)
RB=$(ls "$B"/*.restart.nc | head -1)
Db=$ROOT/rst_on_to_on; rc=$(run "$Db" 10 10 FESOM_ALE=zstar FESOM_SALT_ANOMALY=1 FESOM_RESTART_IN=$RB FESOM_RESTART_EVERY=0)
noconv=$(grep -c 'anomaly-salinity restart -> no conversion' "$Db/run.log")
S=$ROOT/zstar_s35   # the straight 20-step on-run; compare its LAST snapshot with the resumed runs' last
cmp(){ local x=$1 y=$2; local cx=$ROOT/cmp_$(basename "$x") cy=$ROOT/cmp_$(basename "$y"); rm -rf "$cx" "$cy"; mkdir -p "$cx" "$cy"
       cp "$(ls $x/snap_*.nc | tail -n 1)" $cx/snap_final.nc; cp "$(ls $y/snap_*.nc | tail -n 1)" $cy/snap_final.nc; maxd $cx $cy; }
da=$(cmp "$S" "$Da"); db=$(cmp "$S" "$Db")
echo "[restart] off-written → read with knob: converted-line=$conv  vs straight on-run: $da"
echo "[restart] on-written  → read with knob: no-conversion-line=$noconv  vs straight on-run: $db"
[ "$conv" = 1 ] && [ "$noconv" = 1 ] || fail=1
if [ $want = DOUBLE ]; then [ "$(within "${da%% (*}" "$TOL")" = 1 ] || fail=1; [[ "$db" == *bit-identical* ]] || fail=1; fi
[ $fail = 0 ] && echo "=== SALT-ANOMALY GATE PASS ($tag, $want) ===" || { echo "=== SALT-ANOMALY GATE FAIL ($tag, $want) ==="; exit 1; }
