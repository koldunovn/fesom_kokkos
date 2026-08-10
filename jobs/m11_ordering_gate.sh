#!/bin/bash
#SBATCH --job-name=m11ogate
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH -N 2
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:20:00
#SBATCH -o /work/ab0995/a270088/port2/m11/ogate.%j.out
#SBATCH -e /work/ab0995/a270088/port2/m11/ogate.%j.err
#
# M11 ordering race: the CORRECTNESS gate leg, run SEPARATELY from any timed leg because it
# turns per-step diagnostics on (M9 measured a per-step print at 41 % of a fArc GPU 2N step).
#
# Two pre-registered gates, both exploiting the fact that a pure relabelling changes only the
# ORDER of global reductions:
#   * SSH iterations per solve: |delta| <= 1 per step, mean |delta| < 0.5. A systematic shift
#     would mean the operator changed — a bug, not an ordering effect.
#   * the printed per-step maxima (uv, eta, w) and T/S ranges are global reductions and hence
#     permutation-INVARIANT: they can be compared directly, with no field permutation at all.
#     Bound: relative |delta| <= 1e-9 at step 1.
set -u
ROOT=${ROOT:-/home/a/a270088/port_kokkos_part}
source /sw/etc/profile.levante
source "$ROOT/env.sh"
ulimit -s 204800
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(FESOM_SPEED[A-Za-z0-9_]*\)=.*/\1/p')
export FESOM_PRINT_EVERY=1

SB=/work/ab0995/a270088/port2/mesh_m11
BIN=/work/ab0995/a270088/port2/m7/bin/h17/fesom_port_serial
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
NPES=$SLURM_NTASKS
OUT=/work/ab0995/a270088/port2/m11/ogate.${SLURM_JOB_ID}
mkdir -p "$OUT"
md5=$(md5sum "$BIN" | cut -d' ' -f1)
[ "$md5" = 5c3c90fc0ea3939df86cfbe275287c36 ] || { echo "BIN md5 $md5 not certified"; exit 2; }
echo "=== M11 ordering GATE leg  CORE2 $NPES ranks  20 steps dt 1800  BIN md5 $md5"

for a in base hil rcm; do
    case $a in base) M=$SB/core2_m11;; hil) M=$SB/core2_hil;; rcm) M=$SB/core2_rcm;; esac
    o="$OUT/$a"; mkdir -p "$o"
    srun "$BIN" "$M" "$o" 1800 20 -1 "$PHC" 1958 > "$OUT/log_$a.txt" 2>&1
    echo "  $a rc=$? $(grep -ac 'it=' "$OUT/log_$a.txt") printed steps"
done

python3 - "$OUT" <<'PYEOF'
import re, sys
out = sys.argv[1]
pat = re.compile(r"^\s*(\d+)\s+it=\s*(\d+)\s+uv=([\d.eE+-]+)\s+eta=([\d.eE+-]+)\s+w=([\d.eE+-]+)"
                 r"\s+\|\s+T\[([-\d.eE+]+),([-\d.eE+]+)\]\s+S\[([-\d.eE+]+),([-\d.eE+]+)\]")
def load(a):
    d = {}
    for line in open(f"{out}/log_{a}.txt", errors="ignore"):
        m = pat.match(line)
        if m:
            g = m.groups()
            d[int(g[0])] = dict(it=int(g[1]), uv=float(g[2]), eta=float(g[3]), w=float(g[4]),
                                Tmin=float(g[5]), Tmax=float(g[6]),
                                Smin=float(g[7]), Smax=float(g[8]))
    return d
base = load("base")
if not base:
    print("  PARSE FAILED — no 'it=' lines matched; check the print format"); raise SystemExit(1)
fail = 0
FIELDS = ["uv", "eta", "w", "Tmin", "Tmax", "Smin", "Smax"]
for a in ("hil", "rcm"):
    d = load(a)
    steps = sorted(set(base) & set(d))
    print(f"\n--- {a} vs base over {len(steps)} printed steps")
    di = [abs(d[s]["it"] - base[s]["it"]) for s in steps]
    mx, mean = max(di), sum(di) / len(di)
    ok = mx <= 1 and mean < 0.5
    print(f"  SSH iterations   max|d|={mx}  mean|d|={mean:.3f}   "
          f"[bound: max<=1, mean<0.5]  {'PASS' if ok else 'FAIL'}")
    print(f"    base its: {[base[s]['it'] for s in steps[:10]]}")
    print(f"    {a:<4} its: {[d[s]['it'] for s in steps[:10]]}")
    fail += 0 if ok else 1
    s1 = steps[0]
    worst1 = 0.0
    for f in FIELDS:
        b, v = base[s1][f], d[s1][f]
        r = abs(v - b) / max(abs(b), 1e-30)
        worst1 = max(worst1, r)
    ok1 = worst1 <= 1e-9
    print(f"  step {s1} maxima  worst relative |d| = {worst1:.3e}   [bound 1e-9]  "
          f"{'PASS' if ok1 else 'FAIL'}")
    fail += 0 if ok1 else 1
    sl = steps[-1]
    worstN = max(abs(d[sl][f] - base[sl][f]) / max(abs(base[sl][f]), 1e-30) for f in FIELDS)
    print(f"  step {sl} maxima  worst relative |d| = {worstN:.3e}   (growth, reported not gated)")
    for f in FIELDS:
        b, v = base[sl][f], d[sl][f]
        print(f"      {f:<5} base {b: .6e}   {a} {v: .6e}   rel {abs(v-b)/max(abs(b),1e-30):.2e}")
print(f"\n=== ordering gate leg: {'PASS' if fail == 0 else str(fail) + ' FAILURE(S)'}")
raise SystemExit(1 if fail else 0)
PYEOF
