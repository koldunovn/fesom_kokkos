#!/bin/bash
# M1 ACCEPTANCE — step 4/4: compare each Kokkos backend's CORE2 snapshots vs the C-twin reference.
# Run AFTER job_m1accept_cref + job_m1accept_{serial,omp} have completed. diff_snap.py takes two
# DIRECTORIES (L19) and reports per-field bit-identity over every matching snap_*.nc.
#
# Usage: scripts/m1_accept_compare.sh [BASE]      (BASE default /scratch/a/a270088/m1_accept)
set -u
BASE=${1:-/scratch/a/a270088/m1_accept}
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
CREF="$BASE/cref"

[ -d "$CREF" ] || { echo "MISSING reference dir $CREF — run job_m1accept_cref first"; exit 2; }
ls "$CREF"/snap_*.nc >/dev/null 2>&1 || { echo "no snap_*.nc in $CREF — did the C-ref run finish?"; exit 2; }

rc=0
for be in serial omp cuda; do
    d="$BASE/$be"
    [ -d "$d" ] || { echo "== $be: SKIP (no $d) =="; continue; }
    ls "$d"/snap_*.nc >/dev/null 2>&1 || { echo "== $be: SKIP (no snap_*.nc — run not finished) =="; continue; }
    echo "============================================================"
    echo "== $be  vs  C-ref   ($d  vs  $CREF)"
    echo "============================================================"
    "$PY" scripts/diff_snap.py "$CREF" "$d"
    if [ $? -ne 0 ]; then echo "** $be: diff_snap reported a MISMATCH **"; rc=1; fi
done
echo "============================================================"
[ $rc -eq 0 ] && echo "M1 ACCEPTANCE: all present backends BIT-IDENTICAL to the C twin." \
              || echo "M1 ACCEPTANCE: at least one backend DIVERGED — see above (rule out the L18/§C ladder first)."
exit $rc
