#!/usr/bin/env bash
# M16 Task B8 — the MPI/disk pairing invariant (rule SP1): MPI_DOUBLE only over dbl_t/double
# storage, FESOM_MPI_REAL only over real_t storage; nc_*_var_double only over double staging.
# Heuristic, reviewable: for every MPI_DOUBLE / nc_*_double call, take the first buffer
# identifier and look up its declaration in the same file; flag it when that declaration says
# real_t. Prints every call with its verdict; exit 1 if any FLAG.
#   scripts/m16_mpi_invariant.sh [src/...]
set -u
cd "$(dirname "$0")/.."
files=${@:-$(ls src/*.cpp src/*.hpp src/*.h)}
flag=0
for f in $files; do
  case "$f" in *fesom_nc_real.h) continue;; esac   # the boundary helpers: double staging by construction
  grep -n "MPI_DOUBLE\|nc_put_var[a]*_double\|nc_get_var[a]*_double" "$f" | grep -v "^\S*:\s*\(\*\|//\|/\*\)" | while IFS=: read -r ln rest; do
    rest=$(echo "$rest" | sed 's/\.data()//g; s/ + [0-9]*\([,)]\)/\1/g')   # vec.data() -> vec ; ptr + k -> ptr
    # first argument-ish identifier after the call name
    if echo "$rest" | grep -q "nc_[a-z_]*_double"; then
      # netCDF: the data buffer is the LAST argument
      buf=$(echo "$rest" | sed -n 's/.*nc_[a-z_]*_double(\(.*\)).*/\1/p' | awk -F, '{print $NF}' | sed 's/^[ &(]*//; s/[ )]*$//; s/->/ /g; s/\./ /g; s/\[.*//' | awk '{print $NF}')
    else
      buf=$(echo "$rest" | sed -n 's/.*\(MPI_[A-Za-z]*\)(\s*\(MPI_IN_PLACE,\s*\)\?&\?(\?\([A-Za-z_][A-Za-z0-9_>.-]*\).*/\3/p' | sed 's/^(//; s/->/ /g; s/\./ /g' | awk '{print $NF}')
    fi
    [ -z "$buf" ] && { echo "?     $f:$ln  (unparsed)  $(echo "$rest" | cut -c1-90)"; continue; }
    decl=$(grep -n -E "\b(real_t|dbl_t|double|float)\b[^;]*\b$buf\b" "$f" | awk -F: -v L="$ln" '$1<=L{d=$0} END{print d}')
    kind=$(echo "$decl" | grep -o -E "\b(real_t|dbl_t|double|float)\b" | head -1)
    case "$kind" in
      real_t) echo "FLAG  $f:$ln  buffer '$buf' is real_t  <- $(echo "$rest" | cut -c1-80)"; touch /tmp/m16_inv_flag.$$;;
      dbl_t|double) echo "ok    $f:$ln  '$buf' $kind";;
      *) echo "?     $f:$ln  '$buf' (no local declaration found — check by hand)";;
    esac
  done
done
if [ -f /tmp/m16_inv_flag.$$ ]; then rm -f /tmp/m16_inv_flag.$$; echo "=== INVARIANT FAIL ==="; exit 1; fi
echo "=== INVARIANT PASS (no MPI_DOUBLE/nc_double over real_t found) ==="
