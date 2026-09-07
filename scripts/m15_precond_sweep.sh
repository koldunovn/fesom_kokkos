#!/bin/bash
# M15 — SSH preconditioner variant sweep on a dumped system.
#
# Replays ONE dumped SSH system with the production baseline `cg` under each
# FESOM_SSH_PRECOND variant. The matrix A is the dump's, bit-for-bit; only M^-1
# changes, so the iteration count is the clean measurement of the preconditioner.
#
# Iteration counts here are EXACT, not statistical — the solve is deterministic —
# so there is no min-of-N to run. (Rank count shifts them by ~1-4 % via reduction
# order; that is why every mesh is swept at its dump's own np.)
#
# Usage: bash scripts/m15_precond_sweep.sh <mesh> <dumpdir> <tag> [outdir]
set -u
MESH=${1:?mesh}; DUMP=${2:?dump}; TAG=${3:?tag}; OUT=${4:-/tmp/m15_$TAG}
LAB=${BIN:?set BIN to the fesom_ssh_lab to pin (never let it float)}
mkdir -p "$OUT"
RUN=${RUN-srun}   # RUN="" runs in-process (login node); default srun

echo "=== M15 precond sweep: $TAG  mesh=$(basename "$MESH")  dump=$DUMP ==="
echo "    lab=$LAB  sha=$(sha256sum "$LAB" | cut -c1-16)"
printf "%-3s %-44s %8s %9s %13s\n" V FORMULA ITERS DELTA SYMDEFECT
BASE=""
for V in 0 1 2 3 4; do
  FESOM_SSH_PRECOND=$V $RUN "$LAB" "$MESH" "$DUMP" --maxiter 2000 \
      > "$OUT/solve_v$V.out" 2> "$OUT/solve_v$V.err"
  FESOM_SSH_PRECOND=$V $RUN "$LAB" "$MESH" "$DUMP" --sym-check \
      > "$OUT/sym_v$V.out" 2> "$OUT/sym_v$V.err"
  # the production wire observable is authoritative for iters (L80: it proves the solve ran)
  IT=$(grep -a "ssh-wire\] solve 1:" "$OUT/solve_v$V.err" | sed -n 's/.*iters=\([0-9]*\) .*/\1/p' | head -1)
  RES=$(grep -a "ssh-wire\] solve 1:" "$OUT/solve_v$V.err" | sed -n 's/.*res=\([0-9.e+-]*\) .*/\1/p' | head -1)
  RTOL=$(grep -a "ssh-wire\] solve 1:" "$OUT/solve_v$V.err" | sed -n 's/.*rtol=\([0-9.e+-]*\).*/\1/p' | head -1)
  SD=$(grep -a "lab-sym\] pr_values" "$OUT/sym_v$V.out" | grep -ao "ratio = [0-9.e+-]*" | awk '{print $3}')
  case $V in
    0) F="as-built  -0.5a/(di(di+dj))     asym 1/4x";;
    1) F="#984      -a/(di*dj)            SYM  1x";;
    2) F="M10 note  -0.5a/(sqrt*(di+dj))  SYM  1/4x";;
    3) F="sym-hdr   -2a/(sqrt*(di+dj))    SYM  1x";;
    4) F="hdr-lit   -2a/(di(di+dj))       asym 1x";;
  esac
  [ "$V" = 0 ] && BASE=$IT
  DL="-"
  if [ -n "${IT:-}" ] && [ -n "${BASE:-}" ] && [ "$V" != 0 ]; then
    DL=$(python3 -c "print(f'{100*($IT-$BASE)/$BASE:+.1f}%')")
  fi
  # a variant that never met rtol is a NON-CONVERGENCE, not a fast solve: flag it
  CONV=""
  if [ -n "${IT:-}" ] && [ "$IT" = "2000" ]; then CONV=" !MAXITER"; fi
  printf "%-3s %-44s %8s %9s %13s%s\n" "$V" "$F" "${IT:-FAIL}" "$DL" "${SD:-?}" "$CONV"
  echo "      res=$RES rtol=$RTOL" >> "$OUT/summary.txt"
done
echo "(logs in $OUT)"
