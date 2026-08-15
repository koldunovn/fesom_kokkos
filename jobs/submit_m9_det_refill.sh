#!/bin/bash
# M9 det fleet — resubmit the points the first pass lost, with walltimes sized from the MEASURED
# cost of the deterministic fill.
#
# 🔴 WHY THE FIRST PASS LOST THEM. `submit_m9_scaling.sh` was given walltimes padded by +5 min
#    over the legacy fleet, on the strength of a fArc probe at 288 and 1152 ranks where the whole
#    two-leg job finished in 95 s. That generalisation was wrong: **the det fill cost scales
#    inversely with rank count**, because it is a Jacobi iteration to a tolerance and each rank
#    owns more of the mesh at low rank counts. Measured, per model start (det elapsed minus its
#    legacy twin's, over the 6 starts a 3-leg x 2-rep job makes):
#
#      ranks:            4      8     16     32     64
#      CORE2           23 s   14 s   10 s    5 s    2 s
#      fArc            74 s   47 s   30 s   23 s   10 s
#      DARS              -      -   202 s  139 s   72 s
#      NG5               -      -      -      -   148 s
#
#    A job pays that SIX times (ten for an operating point, which has five legs). On NG5 at 8
#    ranks the extrapolation is ~10 min per start, i.e. an hour of fill before any timing.
#    ⇒ For big meshes at low rank counts, walltime is dominated by the fill, not by the model.
#
# Two of the losses are NOT fill cost but sporadic HANGS -- scd_gpu_farc_8n stalled inside the
# timestep loop and op6_farc before it, while both legacy twins ran clean. They are resubmitted
# unchanged; if either hangs again it is a separate investigation.
#
# NOT resubmitted: the six OOM points (cpu dars 1n, cpu ng5 1n/2n and their _phst twins). They
# fail identically in the legacy fleet -- 3.2 M and 7.4 M surface nodes do not fit in one or two
# 128-core nodes -- so they are absent from both fleets by the same physics.
set -u
ROOT=/home/a/a270088/port_kokkos_ice
BINDIR=${BINDIR:-/work/ab0995/a270088/port2/m9/bin/b_det1}
CUDA=$BINDIR/fesom_port_cuda
SER=$BINDIR/fesom_port_serial

FARC=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
DARS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
NG5=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
NG5W=/work/ab0995/a270088/port2/mesh/ng5_w3d

L1="standard::"
L2="lean2::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_EVPWIDE_FUSE=1;FESOM_SPEED_EVPWIDE_LEAN=1"
L3="lean4::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_MEVPDIV=1;FESOM_SPEED_EVPWIDE_LEAN=1"
O1="standard::"
O2="wide_k8::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_EVPWIDE_FUSE=1"
O3="wide_k8_lean::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_EVPWIDE_FUSE=1;FESOM_SPEED_EVPWIDE_LEAN=1"
O4="widediv_k8::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_MEVPDIV=1"
O5="widediv_k8_lean::FESOM_SPEED_EVPWIDE=8;FESOM_SPEED_MEVPDIV=1;FESOM_SPEED_EVPWIDE_LEAN=1"

n=0
sc() {  # sc <gpu|cpu> <tag> <nodes> <ranks> <mesh> <dt> <steps> <walltime> <phst?>
    local be=$1 tag=$2 N=$3 R=$4 mesh=$5 dt=$6 st=$7 wt=$8 ph=${9:-}
    local job=$ROOT/jobs/job_m9_ab_$be bin=$CUDA extra=""
    [ "$be" = cpu ] && bin=$SER
    [ -n "$ph" ] && extra=",PHST=1"
    sbatch -N"$N" --ntasks="$R" -t "$wt" \
      --export=ALL,BIN=$bin,LEG1=$L1,LEG2=$L2,LEG3=$L3,MESH=$mesh,DT=$dt,NSTEPS=$st,TAG=$tag,FESOM_IC_EXTRAP=det$extra \
      "$job" >/dev/null && { echo "  resubmitted $tag ($wt)"; n=$((n+1)); }
}
op() {  # op <tag> <nodes> <ranks> <mesh> <dt> <steps> <walltime> <phst?>
    local tag=$1 N=$2 R=$3 mesh=$4 dt=$5 st=$6 wt=$7 ph=${8:-} extra=""
    [ -n "$ph" ] && extra=",PHST=1"
    sbatch -N"$N" --ntasks="$R" -t "$wt" \
      --export=ALL,BIN=$CUDA,LEG1=$O1,LEG2=$O2,LEG3=$O3,LEG4=$O4,LEG5=$O5,MESH=$mesh,DT=$dt,NSTEPS=$st,TAG=$tag,FESOM_IC_EXTRAP=det$extra \
      "$ROOT/jobs/job_m9_ab_gpu" >/dev/null && { echo "  resubmitted $tag ($wt)"; n=$((n+1)); }
}

echo "--- GPU scaling, fill-dominated (walltime from the table above) ---"
sc gpu scd_gpu_dars_2n       2   8 "$DARS" 120  60 01:10:00
sc gpu scd_gpu_dars_2n_phst  2   8 "$DARS" 120  60 01:10:00 p
sc gpu scd_gpu_ng5_2n        2   8 "$NG5"  180  40 02:00:00
sc gpu scd_gpu_ng5_2n_phst   2   8 "$NG5"  180  40 02:00:00 p
sc gpu scd_gpu_ng5_4n        4  16 "$NG5"  180  40 01:30:00
sc gpu scd_gpu_ng5_4n_phst   4  16 "$NG5"  180  40 01:30:00 p
sc gpu scd_gpu_ng5_8n        8  32 "$NG5"  180  40 01:15:00
sc gpu scd_gpu_ng5_8n_phst   8  32 "$NG5"  180  40 01:15:00 p

echo "--- GPU scaling, sporadic hang (walltime unchanged) ---"
sc gpu scd_gpu_farc_8n       8  32 "$FARC" 900 100 00:30:00

echo "--- CPU scaling, fill-dominated ---"
sc cpu scd_cpu_ng5_4n_phst   4 512 "$NG5"  180  40 01:15:00 p

echo "--- operating points (FIVE legs = ten starts, so ten fills) ---"
op op6_dars       8  32 "$DARS" 120 100 01:10:00
op op6_dars_phst  8  32 "$DARS" 120 100 01:10:00 p
op op6_ng5       16  64 "$NG5W" 180  40 01:30:00
op op6_ng5_phst  16  64 "$NG5W" 180  40 01:30:00 p
op op6_farc       4  16 "$FARC" 900 100 00:40:00

echo "=== $n jobs resubmitted ==="
