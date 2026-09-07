#!/usr/bin/env bash
# M8/M16 — L80 in-band: assert the binary ANNOUNCED the expected working precision.
# Every precision gate calls this on the run log, so a wrong-precision build fails every test
# loudly instead of silently running the wrong experiment (the FESOM2 #940 CI pattern).
#
# Usage: mp_assert_banner.sh <run_log> <SINGLE|DOUBLE>
# Banner shape (src/fesom_main.cpp):
#   [fesom_port] PRECISION: SINGLE  real_t=float  storage=32 bits  digits=6  epsilon=1.19e-07
set -euo pipefail
log=${1:?usage: mp_assert_banner.sh <run_log> <SINGLE|DOUBLE>}
want=${2:?usage: mp_assert_banner.sh <run_log> <SINGLE|DOUBLE>}
case "$want" in SINGLE) bits=32;; DOUBLE) bits=64;; *) echo "want must be SINGLE|DOUBLE" >&2; exit 2;; esac
if grep -q "\[fesom_port\] PRECISION: ${want}  real_t=.*storage=${bits} bits" "${log}"; then
    echo "OK: precision banner ${want} asserted (${log})"
else
    echo "FAIL: expected precision banner '${want}' missing in ${log} — wrong-precision build or banner regression" >&2
    exit 1
fi
