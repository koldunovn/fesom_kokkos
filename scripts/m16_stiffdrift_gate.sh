#!/usr/bin/env bash
# M16 Task B3b gate — the stiffness-drift instrument on pi zstar np2 (seconds; login node is fine).
#   SP binary, FESOM_DIAG_STIFF_DRIFT=<every>, N steps: every [STIFFDRIFT] line must be non-zero
#   (the real_t twin drifts from the dbl_t shadow) and the last must exceed the first (growing);
#   rc 0, banner SINGLE, no non-finite. The DP binary with the same knobs must print NO
#   [STIFFDRIFT] line (the instrument is compiled out) and run rc 0.
#   scripts/m16_stiffdrift_gate.sh <sp build-dir|/abs/bin> <dp build-dir|/abs/bin> [N=200] [every=20]
set -u
SRC=/home/a/a270088/port_kokkos_sp; MESH=/home/a/a270088/port2/fesom2/test/meshes/pi
bin_of(){ if [ -f "$1" ]; then echo "$1"; else echo "$SRC/$1/fesom_port"; fi; }
SP=$(bin_of "${1:?sp}"); DP=$(bin_of "${2:?dp}"); N=${3:-200}; EV=${4:-20}
ROOT=/work/ab0995/a270088/port2/m16/gate0/stiffdrift; rm -rf "$ROOT"; mkdir -p "$ROOT"/{sp,dp}
source /sw/etc/profile.levante >/dev/null 2>&1 || true; source "$SRC/env.sh" >/dev/null 2>&1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_[A-Za-z0-9_]*\)=.*/\1/p')
export FESOM_SSH_PRECOND=0 FESOM_ALE=zstar FESOM_DIAG_STIFF_DRIFT=$EV FESOM_PRINT_EVERY=$((N/4))
fail=0
for t in sp:$SP:SINGLE dp:$DP:DOUBLE; do IFS=: read -r tag bin want <<< "$t"
  o=$ROOT/$tag; mpirun -np 2 "$bin" "$MESH" "$o" 100 $N $N > "$o/run.log" 2> "$o/run.err"; rc=$?
  ban=$("$SRC/scripts/mp_assert_banner.sh" "$o/run.log" "$want" 2>&1 | head -1)
  nl=$(grep -c '^\[STIFFDRIFT\]' "$o/run.log"); nf=$(grep -cwiE 'nan|inf' "$o/run.err" || true)
  echo "[$tag $(md5sum "$bin" | cut -c1-8)] rc=$rc  ${ban%% (*}  STIFFDRIFT lines=$nl  nonfinite=$nf"
  [ $rc = 0 ] && [[ "$ban" == OK* ]] && [ "$nf" = 0 ] || fail=1
  if [ $tag = sp ]; then
    grep '^\[STIFFDRIFT\]' "$o/run.log" | sed 's/^/    /'
    [ "$nl" = $((N/EV)) ] || { echo "    expected $((N/EV)) drift lines"; fail=1; }
    zero=$(grep '^\[STIFFDRIFT\]' "$o/run.log" | awk '{ if ($5+0 == 0 || $7+0 == 0) z++ } END { print z+0 }')
    first=$(grep '^\[STIFFDRIFT\]' "$o/run.log" | head -1 | awk '{print $7}'); last=$(grep '^\[STIFFDRIFT\]' "$o/run.log" | tail -1 | awk '{print $7}')
    grow=$(awk -v a="$first" -v b="$last" 'BEGIN { print (b+0 > a+0) ? 1 : 0 }')
    echo "    zero-valued lines=$zero  offdiag first=$first last=$last growing=$grow"
    [ "$zero" = 0 ] && [ "$grow" = 1 ] || fail=1
  else
    [ "$nl" = 0 ] || { echo "    DP build printed drift lines — the instrument leaked into FP64"; fail=1; }
  fi
done
[ $fail = 0 ] && echo "=== STIFFDRIFT GATE PASS ===" || { echo "=== STIFFDRIFT GATE FAIL ==="; exit 1; }
