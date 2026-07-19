#!/usr/bin/env bash
# M8 / L80 (dead-knob rule, in-band): assert the binary ANNOUNCED the expected working precision.
# Every M8 gate script must call this on the run log — a wrong-precision build then fails every
# test loudly instead of silently running the wrong experiment (pattern from FESOM2 PR-940 CI).
#
# Usage: mp_assert_banner.sh <run_log> <SINGLE|DOUBLE>
set -euo pipefail
log=${1:?usage: mp_assert_banner.sh <run_log> <SINGLE|DOUBLE>}
want=${2:?usage: mp_assert_banner.sh <run_log> <SINGLE|DOUBLE>}
if grep -q "\[fesom_port\] PRECISION: ${want} (" "${log}"; then
    echo "OK: precision banner ${want} asserted (${log})"
else
    echo "FAIL: expected precision banner '${want}' missing in ${log} — wrong-precision build or banner regression" >&2
    exit 1
fi
