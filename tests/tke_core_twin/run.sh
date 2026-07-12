#!/usr/bin/env bash
#
# TKE column-core TWIN GATE — runs the C oracle's integrate_tke and the Kokkos port's
# transcription on 4000 identical synthetic columns and requires BIT-IDENTICAL outputs
# (tke_new, KappaM, KappaH + all 13 diag slabs).
#
# Why this exists: it isolates the ~250-line column MATH from the driver. If it passes and a
# full-model TKE bit-id gate still fails, the bug is provably in the DRIVER (column assembly,
# halos, Av/Kv wiring) — which is most of the search space, eliminated in 30 seconds.
#
# Not part of ctest: it needs the C oracle tree, which not every checkout has.
#
#   bash tests/tke_core_twin/run.sh
#
# Result 2026-07-12 (M6.1 Task 1.2): 2,358,224 values compared, 0 mismatches. PASS.

set -euo pipefail

ROOT=/home/a/a270088/port_kokkos
CSRC=/home/a/a270088/port2/fesom2_port_zstar/src     # the certified C oracle @ df8b9a8
KBUILD=$ROOT/build-serial                            # for the Kokkos headers + libkokkoscore
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

source "$ROOT/env.sh" >/dev/null 2>&1

[ -f "$CSRC/fesom_cvmix_tke.c" ] || { echo "C oracle not found at $CSRC"; exit 2; }
[ -d "$KBUILD" ] || { echo "build-serial not configured — cmake -S . -B build-serial first"; exit 2; }

# The Kokkos include set, taken from the real build so it cannot drift.
KINC=$(grep -m1 "CXX_INCLUDES" "$KBUILD/CMakeFiles/fesom_port.dir/flags.make" | sed 's/^CXX_INCLUDES = //')

# Both sides built with the flags their own trees use (-O3 -DNDEBUG).
gcc -O3 -DNDEBUG -c "$CSRC/fesom_cvmix_tke.c"  -I"$CSRC" -o "$WORK/c_core.o"
gcc -O3 -DNDEBUG -c "$HERE/tke_core_shim.c"    -I"$CSRC" -o "$WORK/c_shim.o"
g++ -O3 -DNDEBUG -std=c++17 -c "$HERE/tke_core_twin.cpp" $KINC -o "$WORK/twin.o"
g++ "$WORK/twin.o" "$WORK/c_core.o" "$WORK/c_shim.o" -o "$WORK/tke_core_twin" \
    -L"$KBUILD/kokkos/core/src" -lkokkoscore -ldl -lm

"$WORK/tke_core_twin"
