#!/bin/bash
# M10 — the GPU half of the M13 det re-run fleet.
#
# HELD until the CUDA fidelity gate is green (job 26961454, CUDA-vs-Serial with
# FESOM_IC_EXTRAP=det): the det fill has never run on the device path — M13's own CUDA leg
# (bg64det 26960226) was cancelled in the queue, so this campaign validates it.
#
# Rows, in priority order:
#   ng5  g16n  the ONLY NG5 row on the board and the one certainly measured on an artifact
#              trajectory (M13 §5b: no NG5 partition was clean under legacy fill).
#              +FESOM_WSPLIT=1, so it REPLACES rather than extends the old row.
#   farc g16n  the best GPU result in the campaign (oati -10.36 % at a 42.2 % SSH share).
#   core2 g1n  cheap control on the mesh with the least dummy coverage — if det moves this,
#              the effect is not about marginal seas at all.
#
# House rules: pinned BIN, 4 legs x 2 reps, 300 steps, ladder dt, <=16 GPU nodes, cheap
# walltimes, STALL_WINDOW=200 on the variants (the farc plateau is intrinsic — measured under
# det in 26961498/26961508, so the default 20 still produces false positives).
set -u
ROOT=${M10_ROOT:-$HOME/port_kokkos_ssh}
cd "$ROOT"
BIN=/work/ab0995/a270088/port2/m10/bin/det1_cuda
W=FESOM_SSH_STALL_WINDOW=200
L="FESOM_SPEED=1;FESOM_SPEED=1+FESOM_SSH_SOLVER=cg2+$W;FESOM_SPEED=1+FESOM_SSH_SOLVER=oati+$W;FESOM_SPEED=1+FESOM_SSH_SOLVER=pcsi+$W"
CORE2=/work/ab0995/a270088/port2/mesh/core2
FARC=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5

sbatch -N 16 --ntasks=64 -t 00:35:00 \
  --export=ALL,MESH="$NG5",DT=180,TAG=det_ng5_g16n,NSTEPS=300,BIN="$BIN",FESOM_IC_EXTRAP=det,FESOM_WSPLIT=1,LEGS="$L" \
  jobs/job_m10_ab | sed 's/^/  ng5  g16n: /'

sbatch -N 16 --ntasks=64 -t 00:35:00 \
  --export=ALL,MESH="$FARC",DT=900,TAG=det_farc_g16n,NSTEPS=300,BIN="$BIN",FESOM_IC_EXTRAP=det,LEGS="$L" \
  jobs/job_m10_ab | sed 's/^/  farc g16n: /'

sbatch -N 1 --ntasks=4 -t 00:35:00 \
  --export=ALL,MESH="$CORE2",DT=1800,TAG=det_core2_g1n,NSTEPS=300,BIN="$BIN",FESOM_IC_EXTRAP=det,LEGS="$L" \
  jobs/job_m10_ab | sed 's/^/  core2 g1n: /'
