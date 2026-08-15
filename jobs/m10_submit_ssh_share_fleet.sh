#!/bin/bash
# M10 — the SSH-share fleet.
#
# WHY THIS EXISTS (user critique, 2026-08-06): the first A/B campaign measured NG5 only, and
# only whole-step time. Both are too narrow:
#   * NG5 is the LARGEST mesh — the SSH solve is ~10 % of its step, so even a large solver win
#     is diluted to ~2 % of the step. The solvers matter where the SSH solve is a LIMITING
#     factor, which is small meshes pushed to high node counts (few unknowns per rank ⇒ the
#     step is communication-bound ⇒ the allreduces dominate).
#   * A whole-step delta cannot say whether the METHOD works. The SSH-phase delta can.
#
# So: CORE2 (126 858 nodes — 58× smaller than NG5) pushed from 16 to 128 ranks, dars in the
# middle, and every rung reports BOTH deltas plus the SSH share of the step.
#
# The hope being tested: these solvers extend the scaling range of meshes that currently stop
# scaling, because what stops them is the very allreduce latency these methods remove.
#
# Usage: bash jobs/m10_submit_ssh_share_fleet.sh
set -u
ROOT=${M10_ROOT:-$HOME/port_kokkos_ssh}
cd "$ROOT"

CORE2=/work/ab0995/a270088/port2/mesh/core2       # the PRIVATE copy (standing rule, L73)
FARC=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5

# 🔴 R8 / E.T1: farc HANGS REPRODUCIBLY AT >= 128 RANKS. Every farc rung here stays at or
# below 64 ranks (16 GPU nodes) — deliberately, not by accident. Do not raise it without
# re-testing the hang first.
FARC_MAX_RANKS=64

# baseline = production; then the three distinct methods (pipecg is ≡ cg2 on this stack, R2)
LEGS="FESOM_SPEED=1;FESOM_SPEED=1+FESOM_SSH_SOLVER=cg2;FESOM_SPEED=1+FESOM_SSH_SOLVER=oati;FESOM_SPEED=1+FESOM_SSH_SOLVER=pcsi"

sub () {   # sub <nodes> <ntasks> <mesh> <dt> <tag> [nsteps]
    local N=$1 NT=$2 MESH=$3 DT=$4 TAG=$5 NS=${6:-300}
    if [ ! -d "$MESH/dist_$NT" ]; then
        echo "SKIP $TAG — $MESH/dist_$NT missing"; return
    fi
    case "$MESH" in *farc*)
        if [ "$NT" -ge 128 ]; then
            echo "REFUSE $TAG — farc at $NT ranks is the R8/E.T1 reproducible hang (>=128)"; return
        fi ;;
    esac
    sbatch -N "$N" --ntasks="$NT" \
        --export=ALL,MESH="$MESH",DT="$DT",TAG="$TAG",NSTEPS="$NS",LEGS="$LEGS" \
        jobs/job_m10_ab | sed "s/^/  $TAG: /"
}

echo "=== CORE2 — the small mesh, driven into its non-scaling regime ==="
echo "    126858 nodes: 7929/rank at 16, 1982/rank at 64, 991/rank at 128"
sub  4  16 "$CORE2" 1800 share_core2_g4n
sub 16  64 "$CORE2" 1800 share_core2_g16n
sub 32 128 "$CORE2" 1800 share_core2_g32n

echo "=== farc — 638387 nodes, ~5x CORE2 and ~5x smaller than dars ==="
echo "    39899/rank at 16, 19950/rank at 32, 9975/rank at 64 (capped: R8 hang at >=128)"
sub  4  16 "$FARC"  900  share_farc_g4n
sub  8  32 "$FARC"  900  share_farc_g8n
sub 16  64 "$FARC"  900  share_farc_g16n

echo "=== dars — the middle mesh ==="
echo "    3160340 nodes: 98761/rank at 32, 24690/rank at 128"
sub  8  32 "$DARS"  120  share_dars_g8n
sub 32 128 "$DARS"  120  share_dars_g32n

echo "=== NG5 — the large mesh, re-run WITH the SSH-phase split for comparability ==="
sub 16  64 "$NG5"   180  share_ng5_g16n

echo
squeue -u a270088 -o "%.10i %.14j %.2t %R" | head -12
