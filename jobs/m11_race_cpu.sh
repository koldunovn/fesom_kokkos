#!/bin/bash
#SBATCH --job-name=m11rcpu
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/port2/m11/racecpu.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/racecpu.%j.err
#
# M11 ordering race, CPU (Kokkos Serial = pure MPI), CORE2.
#
#   sbatch -N 4 --ntasks=512 jobs/m11_race_cpu.sh
#
# Pre-registered in docs/PARTITIONING_M11.md: 300 steps, dt 1800, PHC + JRA55 1958,
# snapshots off, printing off, all three arms inside ONE allocation, 2 reps each, min-of-2.
# The arms differ ONLY in mesh numbering — the partition content is identical by construction
# (label-permuted dists, invariant block verified equal before the race).
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source "$ROOT/jobs/m7_provenance.sh"
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
# no FESOM_SPEED_* leaks into a CPU baseline
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')
export FESOM_PRINT_EVERY=999

SB=/work/ab0995/a270088/port2/mesh_m11
BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial}
BIN_MD5_EXPECT=5c3c90fc0ea3939df86cfbe275287c36
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
DT=${DT:-1800}; NSTEPS=${NSTEPS:-300}
NPES=$SLURM_NTASKS
OUT=/work/ab0995/a270088/port2/m11/racecpu.${SLURM_JOB_ID}
mkdir -p "$OUT"

md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = "$BIN_MD5_EXPECT" ] || { echo "BIN md5 $md5 is not certified h17 Serial"; exit 2; }
echo "=== M11 ordering race CPU  CORE2  ranks=$NPES nodes=$SLURM_NNODES  dt=$DT steps=$NSTEPS"
echo "    BIN=$BIN md5=$md5   $(date '+%F %T')"
m7_provenance "$OUT" "$BIN"

declare -A MESH=( [base]=$SB/core2_m11 [hil]=$SB/core2_hil [rcm]=$SB/core2_rcm )
for a in base hil rcm; do
    [ -d "${MESH[$a]}/dist_$NPES" ] || { echo "REFUSE: ${MESH[$a]}/dist_$NPES missing"; exit 2; }
done

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

# interleave the reps (base,hil,rcm,base,hil,rcm) so a drift in node state hits every arm alike
for r in 1 2; do for a in base hil rcm; do run "$a" "$r"; done; done

echo
echo "=== min-of-2 per arm (s/step), CORE2 $NPES ranks ==="
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
