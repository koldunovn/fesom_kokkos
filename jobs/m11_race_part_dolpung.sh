#!/bin/bash
#SBATCH --job-name=m11dolp
#SBATCH -p dolpung
#SBATCH -A mh1571
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/racedolp.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/racedolp.%j.err
#
# M11 — race PARTITION arms on dolpung (GH200), to ask whether the GPU lever transfers across
# GPU architectures.
#
# Why this is a real question and not a victory lap. The whole GPU mechanism in this campaign was
# established on A100: the currency is the NUMBER of communication partners, not the volume
# (Findings 27, 37), and `MINCONN` wins because it is the only METIS objective that targets it.
# dolpung's fabric CANNOT register CUDA memory — there is no GPUDirect, and halos travel device
# -> pinned host -> MPI -> pinned host -> device under `FESOM_HALO_STAGE=1`. That inserts a
# per-message host staging cost the A100 path does not pay, so a per-message lever could plausibly
# get LARGER here. It could also collapse if the staging cost is dominated by bytes rather than
# messages. Either answer is worth having, and neither is predictable from the A100 numbers.
#
#   ARMS="name=meshdir,..."  sbatch -N 1 --ntasks=4 --export=ALL jobs/m11_race_part_dolpung.sh
#
# 🔴 The binary is NOT the frozen h17 used everywhere else in M11 — h17 is an x86/A100 build and
# cannot run on aarch64. Its md5 is pinned and printed all the same, and the consequence is
# stated once here: dolpung s/step values are NOT comparable with any A100 number in this
# campaign. Only the arm-vs-arm ratios WITHIN one dolpung job are, which is exactly what the
# lever question needs.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
MAIN=${MAIN:-/home/a/a270088/port_kokkos}
source "$ROOT/jobs/m7_provenance.sh"
source "$ROOT/env_dolpung.sh"

# Intra-node winner + dc_mlx5 once the job spans nodes. NEVER a blanket "^gdr_copy": UCX then
# picks knem/xpmem/IB-loopback for intra-node GPU traffic and runs 3.6x slower.
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  export UCX_TLS=self,sm,cuda_copy,cuda_ipc,dc_mlx5
else
  export UCX_TLS=self,sm,cuda_copy,cuda_ipc
fi
export UCX_MEMTYPE_CACHE=n OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_io=romio321 HDF5_USE_FILE_LOCKING=FALSE
export FESOM_PRINT_EVERY=${FESOM_PRINT_EVERY:-999}

BIN=${BIN:-$MAIN/build-dolpung-cuda/fesom_port}
BIN_MD5_EXPECT=${BIN_MD5_EXPECT:-e3c0bb64813783926fd1216eac10553b}
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
DT=${DT:?DT required — use the cold-start ladder dt of the mesh: core2 1800, farc 900, dars 120, ng5 180}
NSTEPS=${NSTEPS:-300}; REPS=${REPS:-2}; YEAR=${YEAR:-1958}
NPES=$SLURM_NTASKS
ARMS=${ARMS:?ARMS="name=meshdir,..." required}
OUT=/work/ab0995/a270088/port2/m11/racedolp.${SLURM_JOB_ID}
mkdir -p "$OUT"

md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = "$BIN_MD5_EXPECT" ] || { echo "BIN md5 $md5 != expected $BIN_MD5_EXPECT"; exit 2; }
echo "=== M11 dolpung partition race  ranks=$NPES nodes=$SLURM_NNODES dt=$DT steps=$NSTEPS reps=$REPS"
echo "    BIN=$BIN md5=$md5   (GH200 build, NOT h17 — see header)"
echo "    UCX_TLS=$UCX_TLS  HALO_STAGE=1  SPEED=1"
echo "    nodes: $SLURM_JOB_NODELIST   $(date '+%F %T')"
m7_provenance "$OUT" "$BIN"

NAMES=""; declare -A MESH
IFS=',' read -ra SPECS <<< "$ARMS"
for spec in "${SPECS[@]}"; do
    n=${spec%%=*}; d=${spec#*=}
    [ -d "$d/dist_$NPES" ] || { echo "REFUSE: $n has no $d/dist_$NPES"; exit 2; }
    MESH[$n]=$d; NAMES="$NAMES $n"
done
echo "    arms:$NAMES"

# Same guard as every other M11 race: the arms must differ ONLY in the partition.
echo "--- mesh-file identity"
declare -A GRP
for n in $NAMES; do
    GRP[$n]=$(for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out; do
                  md5sum "${MESH[$n]}/$f" | cut -d' ' -f1; done | md5sum | cut -c1-8)
done
ngrp=$(for n in $NAMES; do echo "${GRP[$n]}"; done | sort -u | wc -l)
if [ "$ngrp" != 1 ] && [ "${MIXED_NUMBERING:-0}" != 1 ]; then
    echo "REFUSE: the arms carry $ngrp different meshes and MIXED_NUMBERING is not set"
    for n in $NAMES; do printf "    %-16s mesh-group %s\n" "$n" "${GRP[$n]}"; done
    exit 2
fi
for n in $NAMES; do printf "    %-16s mesh-group %s   %s\n" "$n" "${GRP[$n]}" "${MESH[$n]}"; done

run() {
    local a=$1 r=$2
    local o="$OUT/${a}_$r"
    mkdir -p "$o"
    srun -c72 --distribution=block:block bash -c "
        source $ROOT/env_dolpung.sh
        export FESOM_SPEED=1 FESOM_HALO_STAGE=1
        exec $BIN ${MESH[$a]} $o $DT $NSTEPS -1 $PHC $YEAR" > "$OUT/log_${a}_$r.txt" 2>&1
    local rc=$?
    local t
    t=$(grep -a "loop timing" "$OUT/log_${a}_$r.txt" | tail -1 | sed -E 's/.*-> +([0-9.]+) s\/step.*/\1/')
    printf "  %-14s rep%-2s rc=%-3s s/step=%s\n" "$a" "$r" "$rc" "${t:-FAILED}"
    grep -aiE "blow ?up|NaN|FATAL" "$OUT/log_${a}_$r.txt" | head -2 | sed 's/^/        !! /'
    echo "${t:-}" >> "$OUT/times_$a.txt"
}

for r in $(seq 1 "$REPS"); do for a in $NAMES; do run "$a" "$r"; done; done

echo
echo "=== min-of-$REPS per arm (s/step), $NPES ranks ==="
# awk, not python: dolpung nodes are aarch64 and this account's conda python3 is an x86 binary
# that sits early in PATH, so `python3` dies with "cannot execute binary file". env_dolpung.sh
# strips /sw/spack-levante/ but cannot know about every x86 tool a user has on PATH. awk is in
# /usr/bin on the node and cannot have this problem.
awk -v names="$NAMES" -v out="$OUT" '
BEGIN {
  n = split(names, a, " ")
  printf "  %-16s%12s%10s%14s\n", "arm", "min s/step", "spread", "vs " a[1]
  for (i = 1; i <= n; i++) {
    f = out "/times_" a[i] ".txt"; mn = ""; mx = ""
    while ((getline line < f) > 0) {
      if (line == "") continue
      v = line + 0
      if (mn == "" || v < mn) mn = v
      if (mx == "" || v > mx) mx = v
    }
    close(f)
    if (mn == "") { printf "  %-16s   FAILED\n", a[i]; continue }
    if (i == 1) base = mn
    d = (i == 1 || base == 0) ? "" : sprintf("%+.2f %%", 100 * (mn / base - 1))
    printf "  %-16s%12.4f%9.1f%%%14s\n", a[i], mn, 100 * (mx / mn - 1), d
  }
}'
echo
echo "  spread = max/min over the reps of that arm; a delta smaller than the spread is noise."
echo "  These s/step are GH200 numbers from a non-h17 build: compare arms to each other, never"
echo "  to an A100 figure elsewhere in this campaign."
echo "=== done $(date '+%F %T') ==="
