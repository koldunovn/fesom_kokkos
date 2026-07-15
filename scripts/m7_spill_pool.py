#!/usr/bin/env python3
"""Package-C precondition: the SPILL POOL = per-kernel STACK bytes/thread (cuobjdump,
exact, no GPU) cross-ranked by measured GPU-busy ms/step (nsys sqlite, census window).

⚠️ In cuobjdump --dump-resource-usage the spills live in STACK (the ptxas stack frame:
spill slots + ABI stack), NOT in LOCAL (static .local arrays — 0 for every kernel in this
binary). And nsys's localMemoryPerThread is a NON-COLLECTION ARTIFACT (always 0) — never
use it. (Both wrong columns were tried; STACK matches the session-7 audit's 14 spillers.)

usage: m7_spill_pool.py cuobjdump.txt trace.sqlite [--lo 99] [--hi 296]
"""
import argparse, re, sqlite3, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("cuobjdump")
ap.add_argument("db")
ap.add_argument("--lo", type=int, default=99)
ap.add_argument("--hi", type=int, default=296)
args = ap.parse_args()

# ---- parse cuobjdump: Function <mangled>: / REG:n ... LOCAL:n ----------------
fun_re = re.compile(r"^ Function (\S+):")
res_re = re.compile(r"REG:(\d+).*?STACK:(\d+).*?LOCAL:(\d+)")
entries = []  # (mangled, reg, stack, local)
cur = None
for line in open(args.cuobjdump):
    m = fun_re.match(line)
    if m:
        cur = m.group(1)
        continue
    if cur:
        r = res_re.search(line)
        if r:
            entries.append((cur, int(r.group(1)), int(r.group(2)), int(r.group(3))))
            cur = None

# demangle in one c++filt call
names = subprocess.run(["c++filt"], input="\n".join(e[0] for e in entries),
                       capture_output=True, text=True).stdout.splitlines()
# short name: the ENCLOSING FUNCTION of the Kokkos functor/lambda (after ParallelFor<...).
# ⚠️ Never grab "the first fesom_* token": for static kernels that token is a PARAMETER
# TYPE (fesom_mesh...) or the internal-linkage module hash (fesom_tke_cpp_a1eea344).
pf_re = re.compile(r"Parallel(?:For|Reduce|Scan)<(?:[A-Za-z0-9_]+::)*([A-Za-z0-9_]+)[(<,]")
def short(n):
    m = pf_re.search(n)
    if m and m.group(1) not in ("ViewCopy", "ViewFill"):
        return m.group(1)
    return None

# per short name keep the MAX across instantiations (T and S instantiate separately)
spill = {}  # name -> (reg, stack, local)
for (ent, dem) in zip(entries, names):
    k = short(dem)
    if k is None:
        continue
    reg, stack, local = ent[1], ent[2], ent[3]
    old = spill.get(k)
    if old is None or stack > old[1] or (stack == old[1] and reg > old[0]):
        spill[k] = (reg, stack, local)

# ---- kernel-busy ms/step over the census window ------------------------------
con = sqlite3.connect(args.db)
rows = con.execute("""
    SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    ORDER BY k.start
""").fetchall()
anchors = [r[0] for r in rows if "fesom_ale_compute_cflz_kk" in r[2]]
lo, hi = args.lo, min(args.hi, len(anchors) - 2)
a, b = anchors[lo], anchors[hi + 1]
nsteps = hi + 1 - lo
busy = {}
for (s, e, n) in rows:
    if s < a or s >= b:
        continue
    k = short(n)
    if k:
        busy[k] = busy.get(k, 0.0) + (e - s)

NS = 1e-6
tot = sum(busy.values())
pool = 0.0
print(f"window steps {lo}..{hi} ({nsteps} steps), kernel-busy total {tot/nsteps*NS:.1f} ms/step")
print(f"{'ms/step':>8}  {'reg':>4}  {'STACK B/thr':>11}  kernel   [spillers only, by time]")
for k in sorted(busy, key=busy.get, reverse=True):
    if k in spill and spill[k][1] > 0:
        reg, stack, local = spill[k]
        ms = busy[k] / nsteps * NS
        pool += ms
        print(f"{ms:8.2f}  {reg:4d}  {stack:11d}  {k}")
print(f"\nSPILL POOL (busy ms/step of kernels with STACK>0): {pool:.1f} ms/step "
      f"= {pool/(tot/nsteps*NS)*100:.1f}% of kernel-busy")
print("\n[kernels in the binary with STACK>0 but NO time in the window:]")
timed = set(busy)
for k, (reg, stack, local) in sorted(spill.items(), key=lambda x: -x[1][1]):
    if stack > 0 and k not in timed:
        print(f"{'-':>8}  {reg:4d}  {stack:11d}  {k}")
