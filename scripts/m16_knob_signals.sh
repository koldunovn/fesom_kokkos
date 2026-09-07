#!/usr/bin/env bash
# M16 Task E3 (Gate 3) — the knob → liveness-signal table, grep'd from ONE run log (never hand-written).
#   scripts/m16_knob_signals.sh <run.log> [<run.err>]
# Prints one line per knob family: LIVE (the announce/selfcheck that proves it ran), DEAD (an
# "requested but INACTIVE / RESOLVES TO OFF" line — L80: a passing gate on a dead knob is a null),
# or ABSENT (nothing mentioned it). Signals come from the strings in src/ (grep '"\[' src/*.cpp).
log=${1:?run.log}; err=${2:-${1%run.log}run.err}; [ -f "$err" ] || err=/dev/null
both(){ cat "$log" "$err" 2>/dev/null; }
row(){ printf "  %-22s %-7s %s\n" "$1" "$2" "$3"; }
st(){ [ -n "$1" ] && echo LIVE || echo "${2:-ABSENT}"; }
echo "knob liveness for $log"
b=$(both | grep -m1 'PRECISION:' | grep -oE 'PRECISION: [A-Z]+'); row precision "$(st "$b" 'ABSENT')" "${b:-no banner}"
k=$(both | grep -m1 '^\[m14\]'); row knob-summary "$(st "$k" 'ABSENT')" "${k:0:110}"
p=$(both | grep -m1 'ssh-precond\] FESOM_SSH_PRECOND'); row ssh-precond "$(st "$p" 'ABSENT')" "$(echo "$p" | grep -oE 'PRECOND = [^;]*;.*ratio [0-9.e+-]+')"
w=$(both | grep -m1 'wsplit\] FESOM_WSPLIT = ON'); row wsplit "$(st "$w" 'OFF/ABSENT')" "${w:0:80}"
d=$(both | grep -m1 'det extrap: .* fill sweeps'); row ic-extrap-det "$(st "$d" 'ABSENT')" "$(echo "$d" | grep -oE '[0-9]+ fill sweeps.*')$(both | grep -q 'relax cap' && echo ' !! relax cap')"
on=$(both | grep -oE '\[fesom_speed\] FESOM_SPEED_[A-Z0-9_]+ = ON' | sed 's/.*FESOM_SPEED_//; s/ = ON//' | sort -u | tr '\n' ' ')
off=$(both | grep -oE '\[fesom_speed\] !! FESOM_SPEED_[A-Z0-9_]+[^\n]{0,40}' | sed 's/.*!! //' | sort -u | tr '\n' ';')
row speed-levers "$(st "$on" 'ABSENT')" "on: ${on:-none}${off:+ | DEAD: $off}"
c=$(both | grep -m1 'cgpipe\] built'); cd_=$(both | grep -m1 'CGPIPE requested but INACTIVE'); cs=$(both | grep -oE 'cgpipe-selfcheck\] iter +[0-9]+: max[^=]*= [0-9.e+-]+' | tail -n 1)
row cgpipe "$([ -n "$cd_" ] && echo DEAD || st "$c")" "${cs:-${c:0:70}}"
g=$(both | grep -m1 'cgpoly\] ACTIVE'); gd=$(both | grep -m1 'CGPOLY requested but INACTIVE'); gs=$(both | grep -oE 'cgpoly-selfcheck\] iter +[0-9]+: max[^=]*= [0-9.e+-]+' | tail -n 1)
row cgpoly "$([ -n "$gd" ] && echo DEAD || st "$g")" "${gs:-${g:0:70}}"
e=$(both | grep -m1 'evpwide\] built'); ed=$(both | grep -m1 'EVPWIDE requested but'); es=$(both | grep -oE 'evpwide-self\] exch [0-9]+ +drift u=[0-9.e+-]+ v=[0-9.e+-]+' | tail -n 1); el=$(both | grep -m1 'EVPWIDE_LEAN=.*requested but')
row evpwide "$([ -n "$ed" ] && echo DEAD || st "$e")" "${es:-${e:0:70}}${el:+ | LEAN DEAD}"
s=$(both | grep -m1 'ssh_se\] FESOM_SSH_MODE = se'); sm=$(both | grep -m1 'ssh_se\]   M = '); sg=$(both | grep -oE 'ssh_se-gk\] step [0-9]+ sub [0-9]+ +eta +halo-vs-owner:.{0,40}' | tail -n 1)
row ssh-se "$(st "$s" 'ABSENT')" "$(echo "$sm" | grep -oE 'M = [0-9]+ -> dtbt = [0-9.]+ s')${sg:+ | $sg}"
o=$(both | grep -m1 'ssh-solver\] FESOM_SSH_SOLVER = '); row ssh-solver "$(st "$o" 'ABSENT(cg default)')" "$(echo "$o" | grep -oE 'SOLVER = [a-z0-9]+')"
fb=$(both | grep -c 'ssh-solver\] !! FALLBACK'); fb1=$(both | grep -m1 'ssh-solver\] !! FALLBACK' | cut -c1-110)
row solver-fallback "$([ "$fb" = 0 ] && echo none || echo DEAD)" "$([ "$fb" = 0 ] || echo "$fb fallback(s): $fb1")"   # a solver that fell back is NOT the solver under test (G3 2026-09-07: SP CUDA pcsi)
v=$(both | grep -m1 'ssh-verify\] FESOM_SSH_VERIFY = ON'); va=$(both | grep -oE 'ssh-verify\] AGGREGATE: max true-res=[0-9.e+-]+ +max \|true-rec\| gap=[0-9.e+-]+' | tail -n 1)
row ssh-verify "$(st "$v" 'ABSENT')" "${va:-${v:0:60}}"
n=$(both | grep -m1 'MP-NANSCAN ARMED'); row mp-nanscan "$(st "$n" 'ABSENT')" "$(both | grep -m1 'mp-nanscan\].*FIRST non-finite' | cut -c1-90)"
t=$(both | grep -m1 'mp-trace\] ARMED'); row mp-trace "$(st "$t" 'ABSENT')" "${t:0:70}"
cv=$(both | grep -m1 '^CONSERV step='); row mp-conserv "$(st "$cv" 'ABSENT')" "$(both | grep '^CONSERV step=' | tail -n 1 | cut -c1-90)"
sd=$(both | grep -m1 'STIFFDRIFT'); row stiff-drift "$(st "$sd" 'ABSENT')" "$(both | grep 'STIFFDRIFT' | tail -n 1 | cut -c1-80)"
sa=$(both | grep -m1 'use_salt_anomaly'); row salt-anomaly "$(st "$sa" 'ABSENT')" "${sa:0:80}"
last=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+it=' "$log" | tail -n 1); row last-step "$(st "$last" 'ABSENT')" "${last:0:90}"
