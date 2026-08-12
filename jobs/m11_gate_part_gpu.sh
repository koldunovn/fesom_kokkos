#!/bin/bash
#SBATCH --job-name=m11gpgpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH -C a100_80
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/gpartgpu.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/gpartgpu.%j.err
#
# M11 — the ACCURACY + CORRECTNESS gate for PARTITION arms on the CUDA backend.
#
# The CPU gate (jobs/m11_gate_part.sh) is not transferable: the GPU winner is a DIFFERENT
# partition (MINCONN, chosen because it cuts the partner count) and the CUDA backend has its own
# C-vs-Kokkos floor (L79). What carries over is the yardstick: an arm is in class if it sits
# inside the spread of ordinary repartitioning CONTROLS of the same mesh, and out of class if it
# does not. So list controls as arms too — the gate is a comparison of spreads, not of an arm
# against a guessed bound.
#
#   ARMS="name=meshdir,..."   sbatch -N 1 --ntasks=4 --export=ALL jobs/m11_gate_part_gpu.sh
#
# The reference is the FIRST arm listed. Printing is on for every step so the same job yields
# the SSH-iteration trace and the field comparison.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/jobs/m7_provenance.sh"
source "$ROOT/env_cuda.sh"
export OMPI_MCA_pml=ucx OMPI_MCA_btl=self UCX_NET_DEVICES=mlx5_0:1
export UCX_MEMTYPE_CACHE=n OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_io=romio321 HDF5_USE_FILE_LOCKING=FALSE
export FESOM_SPEED=${FESOM_SPEED:-1}

BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_cuda}
BIN_MD5_EXPECT=f8384e86830620510568b95440e61eab
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
DT=${DT:-1800}; NSTEPS=${NSTEPS:-20}; YEAR=${YEAR:-1958}
NPES=$SLURM_NTASKS
ARMS=${ARMS:?ARMS="name=dir,..." required}
OUT=/work/ab0995/a270088/port2/m11/gpartgpu.${SLURM_JOB_ID}
mkdir -p "$OUT"
md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = "$BIN_MD5_EXPECT" ] || { echo "BIN md5 $md5 is not certified h17 CUDA"; exit 2; }
echo "=== M11 partition gate (CUDA)  ranks=$NPES  dt=$DT steps=$NSTEPS  BIN md5 $md5  $(date '+%F %T')"
nvidia-smi -L | head -1
m7_provenance "$OUT" "$BIN"

NAMES=""; declare -A MESH
IFS=',' read -ra SPECS <<< "$ARMS"
for spec in "${SPECS[@]}"; do
    n=${spec%%=*}; d=${spec#*=}
    [ -d "$d/dist_$NPES" ] || { echo "REFUSE: $n lacks dist_$NPES"; exit 2; }
    MESH[$n]=$d; NAMES="$NAMES $n"
done
for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out; do
    u=$(for n in $NAMES; do md5sum "${MESH[$n]}/$f" | cut -d' ' -f1; done | sort -u | wc -l)
    [ "$u" = 1 ] || { echo "REFUSE: $f differs across the legs"; exit 2; }
done
echo "    legs:$NAMES   (five mesh files md5-identical)"

for n in $NAMES; do
    o="$OUT/$n"; mkdir -p "$o"
    FESOM_PRINT_EVERY=1 srun "$BIN" "${MESH[$n]}" "$o" "$DT" "$NSTEPS" -1 "$PHC" "$YEAR" \
        > "$OUT/log_$n.txt" 2>&1
    rc=$?
    printf "  %-14s rc=%-3s %s\n" "$n" "$rc" \
        "$(grep -acE 'identity test \(positive\)' "$OUT/log_$n.txt" | sed 's/^0$/!! HALO GATE SILENT/;s/^[1-9].*/halo gate ok/')"
    grep -aiE "blow ?up|NaN|FATAL|diverged" "$OUT/log_$n.txt" | head -2 | sed 's/^/      !! /'
done

REF=$(echo $NAMES | awk '{print $1}')
echo
echo "=== SSH iterations vs $REF"
$PY - "$OUT" $NAMES <<'EOF'
import re, sys
out, names = sys.argv[1], sys.argv[2:]
pat = re.compile(r"^\s*(\d+)\s+it=\s*(\d+)\s")
def load(a):
    d = {}
    for line in open(f"{out}/log_{a}.txt", errors="ignore"):
        m = pat.match(line)
        if m: d[int(m.group(1))] = int(m.group(2))
    return d
ref = load(names[0])
if not ref: print("  no it= lines in the reference log"); raise SystemExit(1)
for a in names[1:]:
    d = load(a); s = sorted(set(ref) & set(d))
    if not s: print(f"  {a:<14} no overlap"); continue
    dd = [d[k] - ref[k] for k in s]
    n = len(dd)
    print(f"  {a:<14} n={n:<3} mean signed {sum(dd)/n:+6.3f}  mean|d| {sum(map(abs,dd))/n:6.3f}  "
          f"max|d| {max(map(abs,dd))}")
    print(f"      per-step: {dd}")
EOF

echo
echo "=== field comparison vs $REF (the yardstick is the spread of the CONTROL legs)"
for n in $NAMES; do
    [ "$n" = "$REF" ] && continue
    $PY "$ROOT/scripts/m11_accuracy_compare.py" "$OUT/$REF" "$OUT/$n" --vars temp,salt,ssh \
        --label "$n vs $REF" 2>/dev/null | sed 's/^/  /'
done
echo "=== done $(date '+%F %T')"
