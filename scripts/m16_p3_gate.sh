#!/usr/bin/env bash
# M16 Task P3 gate battery — the preconditioner default flip (plan: "gate (Serial, pi)").
#   scripts/m16_p3_gate.sh [build]      default build-m16serial
# Stages (each prints PASS/FAIL, the script exits 1 if any failed):
#   1. m16_gate0.sh all @np1 and @np2 — with PRECOND=0 exported the build is BYTE-IDENTICAL to ref0
#   2. default run (PRECOND unset) on pi DIFFERS from ref0 (liveness of the new default)
#   3. variants 0..4 on pi: symmetry-defect ratio ~0 for 1/2/3, >0 for 0/4; announce line present
#   4. FESOM_SSH_SYMPRE=1 under the default prints "skipped"
#   5. FESOM_SSH_SOLVER=pcsi + PRECOND=4 -> rc != 0
#   6. knob summary names the variant (both branches)
#   7. CORE2 np1 (login, PHC + JRA 1958, dt 1800, 20 steps): mean CG iterations variant 0 vs 1
#      (expect the #984 class, ~ -34 %), and pcsi-vs-cg solution distance under variant 1 is the
#      same class as under variant 0 (max-abs-diff of the step-20 snapshot)
set -u
SRC=/home/a/a270088/port_kokkos_sp
ROOT=${M16_ROOT:-/work/ab0995/a270088/port2/m16/gate0}
PI=/home/a/a270088/port2/fesom2/test/meshes/pi
CORE2=/work/ab0995/a270088/port2/mesh/core2
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
build=${1:-build-m16serial}; BIN=$SRC/$build/fesom_port
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 2; }
P3=$ROOT/p3_$build; rm -rf "$P3"; mkdir -p "$P3"

source /sw/etc/profile.levante >/dev/null 2>&1 || true
source "$SRC/env.sh" >/dev/null 2>&1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_[A-Za-z0-9_]*\)=.*/\1/p')

fail=0; ok(){ echo "PASS: $*"; }; bad(){ echo "FAIL: $*"; fail=1; }
# run <outdir> <np> <mesh> <dt> <nsteps> <snap> [phc year] with the KNOBS env string "A=1;B=2"
run(){ local out=$1 np=$2 mesh=$3 dt=$4 ns=$5 sn=$6 phc=${7:-} yr=${8:-}
  mkdir -p "$out"
  ( if [ -n "${KNOBS:-}" ]; then IFS=';' read -ra KV <<< "$KNOBS"; for kv in "${KV[@]}"; do export "$kv"; done; fi
    env | grep -E '^FESOM_' | sort > "$out/ENV.txt"
    if [ -n "$phc" ]; then mpirun -np $np "$BIN" "$mesh" "$out" $dt $ns $sn "$phc" $yr > "$out/run.log" 2> "$out/run.err"
    else mpirun -np $np "$BIN" "$mesh" "$out" $dt $ns $sn > "$out/run.log" 2> "$out/run.err"; fi
    echo $? > "$out/rc" ); cat "$out/rc"; }
# per-step counts live on stderr ("[step N] done — K CG iters"); the stdout step line prints only
# every `print every` steps and glues the count to "it=" on CORE2.
mean_it(){ awk '/CG iters/{s+=$(NF-2); n++} END{if(n) printf "%.2f", s/n; else print "nan"}' "$1/run.err"; }

echo "=== 1. byte gate with PRECOND=0 (np1, np2) ==="
"$SRC/scripts/m16_gate0.sh" "$build" all 1 | tail -1 | grep -q PASS && ok "gate0 np1 bitwise" || bad "gate0 np1"
"$SRC/scripts/m16_gate0.sh" "$build" all 2 | tail -1 | grep -q PASS && ok "gate0 np2 bitwise" || bad "gate0 np2"

echo "=== 2. default (PRECOND unset) differs from ref0 ==="
KNOBS="" rc=$(run "$P3/default_np1" 1 "$PI" 100 20 1)
[ "$rc" = 0 ] || bad "default run rc=$rc"
"$PY" "$SRC/scripts/diff_snap.py" "$ROOT/ref0/default_np1" "$P3/default_np1" > "$P3/default_np1/diff.txt" 2>&1
[ $? -ne 0 ] && ok "default differs from ref0 ($(grep -c max-abs "$P3/default_np1/diff.txt") differing fields)" || bad "default is byte-identical to ref0 — the default flip is DEAD"
grep -q "precond variant 1" "$P3/default_np1/run.err" && ok "knob summary names variant 1 with no knobs set" || bad "knob summary (no-knob branch) does not name the variant"

echo "=== 3. symmetry-defect ratio per variant (pi np2, 2 steps) ==="
for v in 0 1 2 3 4; do
  KNOBS="FESOM_SSH_PRECOND=$v" rc=$(run "$P3/var${v}_np2" 2 "$PI" 100 2 1)
  line=$(grep -m1 "\[ssh-precond\]" "$P3/var${v}_np2/run.err"); ratio=$(echo "$line" | sed -n 's/.*defect ratio \([0-9.e+-]*\).*/\1/p')
  echo "  variant $v: rc=$rc  $line"
  [ -n "$ratio" ] || { bad "variant $v: no announce line"; continue; }
  sym=$(awk -v r="$ratio" 'BEGIN{print (r < 1e-12) ? 1 : 0}')
  case $v in 1|2|3) [ "$sym" = 1 ] && ok "variant $v symmetric (ratio $ratio)" || bad "variant $v NOT symmetric ($ratio)";;
             0|4)   [ "$sym" = 0 ] && ok "variant $v asymmetric as expected (ratio $ratio)" || bad "variant $v unexpectedly symmetric";; esac
done
grep -q "knob(s) active: FESOM_SSH_PRECOND=3 | precond variant 3" "$P3/var3_np2/run.err" && ok "knob summary (active branch) names the variant" || bad "knob summary active branch"

echo "=== 4. SYMPRE=1 under the default -> skipped ==="
KNOBS="FESOM_SSH_SYMPRE=1;FESOM_SSH_SOLVER=cg2" rc=$(run "$P3/sympre_default_np2" 2 "$PI" 100 2 1)
grep -q "\[ssh-sympre\] skipped: FESOM_SSH_PRECOND=1" "$P3/sympre_default_np2/run.err" && ok "SYMPRE skipped under variant 1 (rc=$rc)" || bad "no 'skipped' line (rc=$rc)"
grep -q "\[ssh-sympre\] BUILT" "$P3/sympre_default_np2/run.err" && bad "SYMPRE was BUILT on variant 1" || ok "SYMPRE not built on variant 1"
KNOBS="FESOM_SSH_PRECOND=0;FESOM_SSH_SYMPRE=1;FESOM_SSH_SOLVER=cg2" rc=$(run "$P3/sympre_var0_np2" 2 "$PI" 100 2 1)
grep -q "\[ssh-sympre\] BUILT" "$P3/sympre_var0_np2/run.err" && ok "SYMPRE still builds on variant 0 (rc=$rc)" || bad "SYMPRE no longer builds on variant 0"

echo "=== 5. pcsi + variant 4 refuses; pcsi + variant 1 runs ==="
KNOBS="FESOM_SSH_PRECOND=4;FESOM_SSH_SOLVER=pcsi" rc=$(run "$P3/pcsi_var4_np2" 2 "$PI" 100 2 1)
[ "$rc" != 0 ] && ok "pcsi+4 refused (rc=$rc): $(grep -m1 -o 'refuses FESOM_SSH_PRECOND=4' "$P3/pcsi_var4_np2/run.err")" || bad "pcsi+4 ran"
KNOBS="FESOM_SSH_SOLVER=pcsi" rc=$(run "$P3/pcsi_var1_np2" 2 "$PI" 100 2 1)
[ "$rc" = 0 ] && ok "pcsi under variant 1 runs (rc=0)" || bad "pcsi under variant 1 rc=$rc: $(tail -2 "$P3/pcsi_var1_np2/run.err")"

echo "=== 7. CORE2 np1 login: CG iterations and pcsi solution class, variant 0 vs 1 ==="
for v in 0 1; do
  KNOBS="FESOM_SSH_PRECOND=$v" rc=$(run "$P3/core2_cg_v$v" 1 "$CORE2" 1800 20 20 "$PHC" 1958)
  echo "  cg   v$v rc=$rc mean it=$(mean_it "$P3/core2_cg_v$v")"
  KNOBS="FESOM_SSH_PRECOND=$v;FESOM_SSH_SOLVER=pcsi" rc=$(run "$P3/core2_pcsi_v$v" 1 "$CORE2" 1800 20 20 "$PHC" 1958)
  echo "  pcsi v$v rc=$rc mean it=$(mean_it "$P3/core2_pcsi_v$v")"
done
i0=$(mean_it "$P3/core2_cg_v0"); i1=$(mean_it "$P3/core2_cg_v1")
awk -v a="$i0" -v b="$i1" 'BEGIN{ d=(a-b)/a*100; printf "  CG mean iterations: v0 %.2f -> v1 %.2f  (%.1f %% fewer)\n", a, b, d; exit (d >= 20) ? 0 : 1 }' && ok "iteration drop in the #984 class" || bad "iteration drop below 20 %"
for v in 0 1; do
  "$PY" "$SRC/scripts/diff_snap.py" "$P3/core2_cg_v$v" "$P3/core2_pcsi_v$v" > "$P3/core2_pcsi_v$v/diff_vs_cg.txt" 2>&1
  echo "  pcsi vs cg under v$v (step-20 snapshot):"; grep -E "ssh|eta|temp|salt|u:|v:" "$P3/core2_pcsi_v$v/diff_vs_cg.txt" | head -6 | sed 's/^/     /'
done
"$PY" - "$P3/core2_pcsi_v0/diff_vs_cg.txt" "$P3/core2_pcsi_v1/diff_vs_cg.txt" <<'PYE' && ok "pcsi-vs-cg distance under v1 is the same class as under v0" || bad "pcsi-vs-cg distance under v1 is NOT the v0 class"
import sys,re
def mx(p):
    d={}
    for l in open(p):
        m=re.match(r'\s+(\S+): max-abs-diff=([0-9.e+-]+)',l)
        if m: d[m.group(1)]=float(m.group(2))
    return d
a,b=mx(sys.argv[1]),mx(sys.argv[2]); bad=0
for k in sorted(set(a)|set(b)):
    x,y=a.get(k,0.0),b.get(k,0.0); ref=max(x,1e-300)
    if k in ('Av','Kv'): continue      # KPP threshold fields: single-node bl-depth flips, not a solver class
    flag = y > 100*ref and y > 1e-12
    print(f"     {k:12s} v0 {x:.3e}  v1 {y:.3e}  {'!!' if flag else ''}")
    bad|=flag
sys.exit(1 if bad else 0)
PYE

echo; [ $fail = 0 ] && echo "=== P3 GATE PASS ===" || echo "=== P3 GATE FAIL ==="
exit $fail
