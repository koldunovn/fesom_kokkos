#!/usr/bin/env bash
# M14/M16 leg admission — ONE implementation for both ladders (extracted from jobs/job_m14_ladder_*:143,
# Task E1). rc=0 is not aliveness (M10 L106: a NaN-blind guard turns a broken run into a FAST clean exit;
# a zombie leg measured 10.8 % faster than a healthy one).
#   m14_zombie_check.sh <run.log> <rc> [WANT_PREC=DOUBLE|SINGLE] [SSH_VERIFY_BAR]
# Prints one line `t=<s/step|NA> live=yes|no [reasons]`; exit 0 = admitted.
# Rejects: rc≠0 · no final step line · non-finite in it · it=0 · CGPIPE-INACTIVE · no s/step timer ·
# banner precision ≠ WANT_PREC (L80/SP10: the arm must be what was requested) · [ssh-verify] gap above the bar.
log=${1:?run.log}; rc=${2:?rc}; want=${3:-}; bar=${4:-}
t=$(grep -oE '\->[[:space:]]*[0-9.]+ s/step' "$log" | tail -1 | grep -oE '[0-9.]+')
last=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+it=' "$log" | tail -1)
live=yes; why=""
[ "$rc" -eq 0 ] || { live=no; why="$why rc=$rc"; }
[ -n "$last" ] || { live=no; why="$why no-step-line"; }
echo "$last" | grep -qiE 'nan|inf' && { live=no; why="$why non-finite"; }
echo "$last" | grep -qE 'it=0[^0-9]' && { live=no; why="$why it=0"; }
grep -q 'CGPIPE-INACTIVE' "$log" && { live=no; why="$why CGPIPE-INACTIVE"; }
err=${log%run.log}run.err; [ -f "$err" ] && grep -q 'ssh-solver\] !! FALLBACK' "$err" && { live=no; why="$why solver-FALLBACK"; }   # M16 G3: a leg that fell back to CG did not time the solver it claims
[ -n "$t" ] || { live=no; why="$why no-timer"; }
if [ -n "$want" ]; then
    case "$want" in DOUBLE) bits=64;; SINGLE) bits=32;; *) echo "t=${t:-NA} live=no bad-WANT_PREC=$want"; exit 1;; esac
    grep -q "\[fesom_port\] PRECISION: ${want}  real_t=.*storage=${bits} bits" "$log" || { live=no; why="$why banner!=$want"; }
fi
if [ -n "$bar" ]; then
    gap=$(grep -oE '\[ssh-verify\].*gap[^0-9]*[0-9.eE+-]+' "$log" | tail -1 | grep -oE '[0-9.]+e[+-][0-9]+|[0-9.]+$' | tail -1)
    if [ -n "$gap" ]; then awk -v g="$gap" -v b="$bar" 'BEGIN { exit !(g+0 > b+0) }' && { live=no; why="$why ssh-verify-gap=$gap>$bar"; }; fi
fi
echo "t=${t:-NA} live=$live${why:+ [$why]}"
[ "$live" = yes ]
