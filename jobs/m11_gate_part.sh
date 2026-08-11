#!/bin/bash
#SBATCH --job-name=m11gpart
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:40:00
#SBATCH -o /work/ab0995/a270088/port2/m11/gpart.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/gpart.%j.err
#
# M11 — the ACCURACY + CORRECTNESS gate for PARTITION arms. Everything raced so far measured
# speed only; nothing in the partition family has been through a gate.
#
# A partition arm is not a pure-ordering arm, so the ordering gates do not transfer: it changes
# which rank owns which node, so the solution moves within the SSH solver's tolerance ball
# (Finding 11) exactly as a repartitioning control does. That is the right yardstick and the
# project already accepts it (L79): an arm is in class if it sits inside the spread of ordinary
# repartitioning controls, and out of class if it does not.
#
#   ARMS="name=meshdir,..."  CONTROLS="name=meshdir,..."   sbatch -N 2 --ntasks=256 ...
#
# Every leg runs 20 steps with printing on, so the same job yields the SSH-iteration behaviour
# and the field comparison. The reference is the FIRST arm listed.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')

BIN=${BIN:-/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial}
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
DT=${DT:-1800}; NSTEPS=${NSTEPS:-20}
ARMS=${ARMS:?ARMS="name=dir,..." required}
OUT=/work/ab0995/a270088/port2/m11/gpart.${SLURM_JOB_ID}
mkdir -p "$OUT"
md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = 5c3c90fc0ea3939df86cfbe275287c36 ] || { echo "BIN md5 $md5 not certified"; exit 2; }
echo "=== M11 partition gate  ranks=$SLURM_NTASKS  dt=$DT steps=$NSTEPS  BIN md5 $md5  $(date '+%F %T')"

NAMES=""; declare -A MESH
IFS=',' read -ra SPECS <<< "$ARMS"
for spec in "${SPECS[@]}"; do
    n=${spec%%=*}; d=${spec#*=}
    [ -d "$d/dist_$SLURM_NTASKS" ] || { echo "REFUSE: $n lacks dist_$SLURM_NTASKS"; exit 2; }
    MESH[$n]=$d; NAMES="$NAMES $n"
done
for f in nod2d.out elem2d.out aux3d.out nlvls.out elvls.out; do
    u=$(for n in $NAMES; do md5sum "${MESH[$n]}/$f" | cut -d' ' -f1; done | sort -u | wc -l)
    [ "$u" = 1 ] || { echo "REFUSE: $f differs across the legs"; exit 2; }
done
echo "    legs:$NAMES   (five mesh files md5-identical)"

for n in $NAMES; do
    o="$OUT/$n"; mkdir -p "$o"
    FESOM_PRINT_EVERY=1 srun "$BIN" "${MESH[$n]}" "$o" "$DT" "$NSTEPS" -1 "$PHC" 1958 \
        > "$OUT/log_$n.txt" 2>&1
    rc=$?
    printf "  %-14s rc=%-3s %s\n" "$n" "$rc" \
        "$(grep -acE 'identity test \(positive\)' "$OUT/log_$n.txt" | sed 's/^0$/!! HALO GATE SILENT/;s/^[1-9].*/halo gate ok/')"
    grep -aiE "blow ?up|NaN|FATAL|diverged" "$OUT/log_$n.txt" | head -2 | sed 's/^/      !! /'
done

REF=$(echo $NAMES | awk '{print $1}')
echo
echo "=== SSH iterations vs $REF (a partition change reorders the CG reductions; a SYSTEMATIC"
echo "    shift would mean the operator changed, a symmetric excursion is round-off)"
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
echo "=== field comparison vs $REF (rms over the whole field; the yardstick is the spread of the"
echo "    CONTROL legs, which are ordinary repartitionings of the same mesh)"
for n in $NAMES; do
    [ "$n" = "$REF" ] && continue
    $PY "$ROOT/scripts/m11_accuracy_compare.py" "$OUT/$REF" "$OUT/$n" --vars temp,salt,ssh \
        --label "$n vs $REF" 2>/dev/null | sed 's/^/  /'
done
echo "=== done $(date '+%F %T')"
