#!/bin/bash
#SBATCH --job-name=m11ship
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH -N 7
#SBATCH --ntasks=864
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/port2/m11/ship864.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/ship864.%j.err
#
# M11 — RE-MEASURE the shipped-864 claim before explaining it any further.
#
# M10 reported the shipped CORE2 `dist_864` 7.4 % faster than a flat regeneration. Three
# independent offline explanations have now been tested against it and none accounts for the
# size of the effect:
#   * partition QUALITY (Task 2): cut 34,159 vs 34,157, halo 42.1 vs 42.1, element replication
#     1.473 vs 1.474, comm volume max/rank 2,735 vs 2,707 — indistinguishable.
#   * PLACEMENT (Finding 16): off-node 3-D halo 109,037 vs 112,246 — right direction, 2.9 %.
#   * per-NODE aggregate work (memory-bandwidth pressure, not per-rank load): 1.2380 vs 1.2315
#     max/mean — identical.
# The remaining possibility the data cannot exclude is that the 7.4 % is not a partition effect
# at all. This job settles that: three partitions of the SAME mesh (five mesh files verified
# md5-identical across all three directories), all arms in ONE allocation, interleaved reps,
# min-of-2.
#
#   ship  = the shipped dist_864 (an older tool; 71 isolated nodes)
#   base  = the same partition settled to a fixed point of check_partitioning (FIXISO)
#   wgt0  = M10's flat 2-D-weighted regeneration — the arm they measured as 7.4 % slower
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/jobs/m7_provenance.sh"
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')
export FESOM_PRINT_EVERY=999

SB=/work/ab0995/a270088/port2/mesh_m11
BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial}
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
DT=${DT:-1800}; NSTEPS=${NSTEPS:-300}
NPES=$SLURM_NTASKS
OUT=/work/ab0995/a270088/port2/m11/ship864.${SLURM_JOB_ID}
mkdir -p "$OUT"
md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = 5c3c90fc0ea3939df86cfbe275287c36 ] || { echo "BIN md5 $md5 not certified"; exit 2; }
echo "=== M11 shipped-864 re-measurement  CORE2 $NPES ranks / $SLURM_NNODES nodes  $(date '+%F %T')"
m7_provenance "$OUT" "$BIN"

declare -A MESH=( [ship]=$SB/core2_m11 [base]=$SB/core2_base
                  [wgt0]=/work/ab0995/a270088/port2/mesh/core2_wgt0 )
for a in ship base wgt0; do
    [ -d "${MESH[$a]}/dist_$NPES" ] || { echo "REFUSE: ${MESH[$a]}/dist_$NPES missing"; exit 2; }
done
echo "--- the three arms must differ ONLY in the partition"
for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out; do
    n=$(for a in ship base wgt0; do md5sum "${MESH[$a]}/$f" | cut -d' ' -f1; done | sort -u | wc -l)
    [ "$n" = 1 ] || { echo "REFUSE: $f differs between the arms"; exit 2; }
done
echo "    five mesh files md5-identical across all three"

run() {
    local a=$1
    local r=$2
    local o="$OUT/${a}_$r"
    mkdir -p "$o"
    srun "$BIN" "${MESH[$a]}" "$o" "$DT" "$NSTEPS" -1 "$PHC" 1958 > "$OUT/log_${a}_$r.txt" 2>&1
    local rc=$?
    local t
    t=$(grep -a "loop timing" "$OUT/log_${a}_$r.txt" | tail -1 | sed -E 's/.*-> +([0-9.]+) s\/step.*/\1/')
    printf "  %-5s rep%-2s rc=%-3s s/step=%s\n" "$a" "$r" "$rc" "${t:-FAILED}"
    grep -aiE "blow ?up|NaN|FATAL" "$OUT/log_${a}_$r.txt" | head -2 | sed 's/^/      !! /'
    echo "${t:-}" >> "$OUT/times_$a.txt"
}
for r in 1 2; do for a in ship base wgt0; do run "$a" "$r"; done; done

echo
echo "=== min-of-2 per arm (s/step), CORE2 $NPES ranks ==="
python3 - "$OUT" <<'EOF'
import sys, os
out = sys.argv[1]
res = {}
for a in ("ship", "base", "wgt0"):
    p = f"{out}/times_{a}.txt"
    v = [float(x) for x in open(p).read().split() if x] if os.path.exists(p) else []
    res[a] = min(v) if v else None
b = res.get("ship")
for a in ("ship", "base", "wgt0"):
    t = res[a]
    if t is None:
        print(f"  {a:<5} FAILED"); continue
    d = "" if (a == "ship" or not b) else f"   {100*(t/b-1):+.2f} % vs shipped"
    print(f"  {a:<5} {t:.4f} s/step{d}")
print("\n  M10 reported the flat regeneration 7.4 % SLOWER than shipped; if wgt0 does not land")
print("  near +7.4 % here, the effect is not a property of these partitions.")
EOF
echo "=== done $(date '+%F %T') ==="
