#!/bin/bash
#SBATCH --job-name=m11lvl
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /work/ab0995/a270088/port2/m11/lvl.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/lvl.%j.err
# M11 Task 4 follow-up: does the CURRENT tool reproduce the ARCHIVED smoothed levels?
# elvls_raw (pre-smoothing) already reproduces byte-for-byte on both meshes, so any
# difference is in the iterative rough-topography pass. Run on mesh dirs that carry only
# nod2d/elem2d/aux3d, so the tool must WRITE nlvls.out and elvls.out itself.
set -u
ROOT=/home/a/a270088/port_kokkos_part
SB=/work/ab0995/a270088/port2/mesh_m11
P=/work/ab0995/a270088/port2/partm11/fesom2
OUT=/work/ab0995/a270088/port2/m11/lvl.${SLURM_JOB_ID}
for m in core2 farc; do
  env MESH=$SB/lvl_$m RANKS="8" TAG=lvl_$m ROOT=$ROOT OUT=$OUT PARTROOT=$P \
      BIN=$P/mesh_part/build/bin/fesom_meshpart bash $ROOT/scripts/m11_partgen.sh
done
echo; echo "############ fresh vs archived levels"
cmp $SB/lvl_core2/nlvls.out $SB/core2_m11/nlvls.out && echo "  CORE2 nlvls IDENTICAL" || echo "  CORE2 nlvls DIFFERS"
cmp $SB/lvl_core2/elvls.out $SB/core2_m11/elvls.out && echo "  CORE2 elvls IDENTICAL" || echo "  CORE2 elvls DIFFERS"
cmp $SB/lvl_farc/nlvls.out  $SB/farc_m11/nlvls.out  && echo "  fArc  nlvls IDENTICAL" || echo "  fArc  nlvls DIFFERS"
cmp $SB/lvl_farc/elvls.out  $SB/farc_m11/elvls.out  && echo "  fArc  elvls IDENTICAL" || echo "  fArc  elvls DIFFERS"
