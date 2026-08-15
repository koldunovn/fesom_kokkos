#!/bin/bash
#SBATCH --job-name=m11rgpu
#SBATCH -p gpu
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH -C a100_80
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/port2/m11/racegpu.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/racegpu.%j.err
#
# M11 ordering race, GPU (CUDA h17), CORE2. 4 A100/node.
#
#   sbatch -N 1 --ntasks=4 jobs/m11_race_gpu.sh
#   sbatch -N 2 --ntasks=8 jobs/m11_race_gpu.sh
#
# `-C a100_80` is mandatory on every GPU absolute (L94: the gpu partition is heterogeneous).
# Same pre-registered protocol as the CPU race; the arms differ only in mesh numbering.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/jobs/m7_provenance.sh"
source "$ROOT/env_cuda.sh"
export OMPI_MCA_pml=ucx OMPI_MCA_btl=self UCX_NET_DEVICES=mlx5_0:1
export UCX_MEMTYPE_CACHE=n OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_io=romio321 HDF5_USE_FILE_LOCKING=FALSE
export FESOM_PRINT_EVERY=999
# production GPU environment, frozen in the session log
export FESOM_SPEED=${FESOM_SPEED:-1}

SB=/work/ab0995/a270088/port2/mesh_m11
BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_cuda}
BIN_MD5_EXPECT=f8384e86830620510568b95440e61eab
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
DT=${DT:-1800}; NSTEPS=${NSTEPS:-300}
NPES=$SLURM_NTASKS
OUT=/work/ab0995/a270088/port2/m11/racegpu.${SLURM_JOB_ID}
mkdir -p "$OUT"

md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = "$BIN_MD5_EXPECT" ] || { echo "BIN md5 $md5 is not certified h17 CUDA"; exit 2; }
echo "=== M11 ordering race GPU  CORE2  ranks=$NPES nodes=$SLURM_NNODES  dt=$DT steps=$NSTEPS"
echo "    BIN=$BIN md5=$md5  FESOM_SPEED=$FESOM_SPEED  $(date '+%F %T')"
echo "    nodes: $SLURM_JOB_NODELIST"; nvidia-smi -L | head -1
m7_provenance "$OUT" "$BIN"

# 🔴 base = core2_base, the SETTLED baseline, NOT the shipped core2_m11 (Finding 10). The GPU
# points (4 and 8 ranks) have byte-identical shipped and settled dists, so this changes nothing
# here — but the two race jobs must not disagree about what "base" means.
declare -A MESH=( [base]=$SB/core2_base [hil]=$SB/core2_hil [rcm]=$SB/core2_rcm )
for a in base hil rcm; do
    [ -d "${MESH[$a]}/dist_$NPES" ] || { echo "REFUSE: ${MESH[$a]}/dist_$NPES missing"; exit 2; }
done
PY=${PY:-/work/ab0995/a270088/mambaforge/envs/nereus/bin/python}
echo "--- pure-ordering precheck"
$PY "$ROOT/scripts/m11_pure_ordering_check.py" --npes "$NPES" --base "${MESH[base]}" \
    --arm "hil=${MESH[hil]}" --arm "rcm=${MESH[rcm]}" || exit 2

run() {   # run <arm> <rep>
    local a=$1 r=$2
    local o="$OUT/${a}_$r"
    mkdir -p "$o"
    srun "$BIN" "${MESH[$a]}" "$o" "$DT" "$NSTEPS" -1 "$PHC" 1958 > "$OUT/log_${a}_$r.txt" 2>&1
    local rc=$?
    local t=$(grep -a "loop timing" "$OUT/log_${a}_$r.txt" | tail -1 | sed -E 's/.*-> +([0-9.]+) s\/step.*/\1/')
    printf "  %-5s rep%-2s rc=%-3s s/step=%s\n" "$a" "$r" "$rc" "${t:-FAILED}"
    grep -aqE "identity test \(positive\)" "$OUT/log_${a}_$r.txt" || echo "      !! halo gate did not announce"
    grep -aiE "blow ?up|NaN|FATAL" "$OUT/log_${a}_$r.txt" | head -2 | sed 's/^/      !! /'
    echo "${t:-}" >> "$OUT/times_$a.txt"
}

for r in 1 2; do for a in base hil rcm; do run "$a" "$r"; done; done

echo
echo "=== min-of-2 per arm (s/step), CORE2 $NPES GPU ranks ==="
python3 - "$OUT" <<'EOF'
import sys, os
out = sys.argv[1]
res = {}
for a in ("base", "hil", "rcm"):
    p = f"{out}/times_{a}.txt"
    v = [float(x) for x in open(p).read().split() if x] if os.path.exists(p) else []
    res[a] = min(v) if v else None
b = res.get("base")
for a in ("base", "hil", "rcm"):
    t = res[a]
    if t is None:
        print(f"  {a:<5} FAILED"); continue
    d = "" if (a == "base" or not b) else f"   {100*(t/b-1):+.2f} % vs base"
    print(f"  {a:<5} {t:.4f} s/step{d}")
EOF
echo "=== done $(date '+%F %T') ==="
