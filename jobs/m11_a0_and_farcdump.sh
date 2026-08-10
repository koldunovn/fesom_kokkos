#!/bin/bash
#SBATCH --job-name=m11a0
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/port2/m11/a0.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/a0.%j.err
set -u
ROOT=/home/a/a270088/port_kokkos_part
SB=/work/ab0995/a270088/port2/mesh_m11
OUT=/work/ab0995/a270088/port2/m11/a0.${SLURM_JOB_ID}
echo "############ arm A0: METIS 5.2.1, otherwise the historical defaults"
env MESH=$SB/a0_m11 RANKS="8 16 32" TAG=a0 ROOT=$ROOT OUT=$OUT \
    PARTROOT=/work/ab0995/a270088/port2/partm11/fesom2_b \
    BIN=/work/ab0995/a270088/port2/partm11/fesom2_b/mesh_part/build/bin/fesom_meshpart \
    bash $ROOT/scripts/m11_partgen.sh
echo; echo "############ fArc CSR dump (exporter cross-check at the bigger mesh)"
env MESH=$SB/farc_dump RANKS="16" TAG=farcdump ROOT=$ROOT OUT=$OUT \
    PARTROOT=/work/ab0995/a270088/port2/partm11/fesom2 \
    BIN=/work/ab0995/a270088/port2/partm11/fesom2/mesh_part/build/bin/fesom_meshpart \
    FESOM_PART_GRAPH_DUMP=$OUT/farc_csr.txt bash $ROOT/scripts/m11_partgen.sh
