#!/usr/bin/env bash
#
# Jupiter (JSC) DATA paths — meshes + forcing + IC for FESOM2-Kokkos. Source this
# AFTER env_jupiter.sh / env_jupiter_cuda.sh. Keeps the data location separate from
# the module env so job scripts read one thing. All four forcing paths are honoured
# by env-override hooks in src (FESOM_JRA55_DIR / FESOM_SSS_PATH / FESOM_RUNOFF_PATH /
# FESOM_CHL_FILE); the mesh and PHC IC are CLI args (exported here for the job scripts).

# Root of the staged data on JUPITER scratch.
# NB: /p/scratch/hclimrep is LOGIN-ONLY (not mounted on booster compute nodes). The
# compute-visible scratch for this project is /e/scratch/hclimrep (the /e Exascale tier,
# $SCRATCH_hclimrep). Data is staged there; the original lives at /p/scratch/.../meshes.
export FESOM_DATA_ROOT=/e/scratch/hclimrep/koldunov1/meshes

# Meshes (pick one in the job via $MESH): pi (3,140), core2 (126,858), ng5 (7,402,886).
export FESOM_MESH_PI=$FESOM_DATA_ROOT/pi
export FESOM_MESH_CORE2=$FESOM_DATA_ROOT/core2
export FESOM_MESH_NG5=$FESOM_DATA_ROOT/ng5

# PHC3.0 winter initial condition (CLI arg 6).
export FESOM_PHC=$FESOM_DATA_ROOT/phc3.0/phc3.0_winter.nc

# JRA55-do v1.4.0 atmospheric forcing (year via CLI arg 7) + SSS restoring + runoff + chl.
export FESOM_JRA55_DIR=$FESOM_DATA_ROOT/JRA55-do-v1.4.0
export FESOM_SSS_PATH=$FESOM_JRA55_DIR/PHC2_salx.nc
export FESOM_RUNOFF_PATH=$FESOM_JRA55_DIR/runoff.nc
export FESOM_CHL_FILE=$FESOM_DATA_ROOT/Sweeney/Sweeney_2005.nc
