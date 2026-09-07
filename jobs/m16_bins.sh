#!/usr/bin/env bash
# M16 Task E1 — freeze the sha-named precision pair for the ladders:
#   port2/m16/bin/<tag>/{dp,sp}/{fesom_port_serial,fesom_port_cuda} + PROVENANCE.txt
#   bash jobs/m16_bins.sh <tag>      (from the four build dirs of ~/port_kokkos_sp; refuses a missing binary)
set -eu
SRC=/home/a/a270088/port_kokkos_sp; B=/work/ab0995/a270088/port2/m16/bin/${1:?tag}
for p in dp sp; do
  sd=$SRC/build-m16serial; cd_=$SRC/build-m16cuda; [ $p = sp ] && { sd=${sd}-sp; cd_=${cd_}-sp; }
  [ -x $sd/fesom_port ] && [ -x $cd_/fesom_port ] || { echo "REFUSE: missing $sd/fesom_port or $cd_/fesom_port"; exit 2; }
  mkdir -p $B/$p; cp $sd/fesom_port $B/$p/fesom_port_serial; cp $cd_/fesom_port $B/$p/fesom_port_cuda
  { echo "commit $(git -C $SRC rev-parse HEAD) ($(git -C $SRC log -1 --format=%s | cut -c1-80))"; echo "frozen $(date -Is) by jobs/m16_bins.sh";
    echo "USE_SINGLE_PRECISION=$([ $p = sp ] && echo ON || echo OFF)"; echo "kokkos $(grep -m1 'Kokkos_VERSION\b' $sd/CMakeCache.txt 2>/dev/null || echo 'vendored 4.4.01')";
    echo "flags $(grep -m1 CMAKE_CXX_FLAGS_RELEASE $sd/CMakeCache.txt)"; grep -m1 'libmpi.so' <(ldd $B/$p/fesom_port_cuda) ; md5sum $B/$p/fesom_port_serial $B/$p/fesom_port_cuda; } > $B/$p/PROVENANCE.txt
  echo "$1/$p: serial $(md5sum $B/$p/fesom_port_serial | cut -c1-8) cuda $(md5sum $B/$p/fesom_port_cuda | cut -c1-8)"
done
