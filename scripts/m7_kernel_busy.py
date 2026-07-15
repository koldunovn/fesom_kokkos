#!/usr/bin/env python3
"""Per-kernel GPU-busy ms/step from an nsys sqlite (rank-0 device).

The complement of m7_gap_census.py: the census ranks the GAPS, this ranks the KERNELS.
Used to cross-rank cuobjdump's spill list by time (package C sizing).

Census conventions (m7_cg_share.py): step boundaries anchored on
fesom_ale_compute_cflz_kk; steady window = steps 99..296 (or what exists).

usage: m7_kernel_busy.py trace.sqlite [--lo 99] [--hi 296] [--top 40]
"""
import argparse, re, sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--lo", type=int, default=99)
ap.add_argument("--hi", type=int, default=296)
ap.add_argument("--top", type=int, default=40)
args = ap.parse_args()

con = sqlite3.connect(args.db)
rows = con.execute("""
    SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    ORDER BY k.start
""").fetchall()

# short name: the ENCLOSING FUNCTION of the Kokkos functor/lambda, i.e. the text right
# after ParallelFor</ParallelReduce<. ⚠️ Never grab "the first fesom_* token": for static
# kernels (diff_ver_part_impl_ale_kk, kpp_*_kk...) that token is a PARAMETER TYPE
# (fesom_mesh, fesom_kpp...) or the internal-linkage module hash — wrong kernel, silently.
pf_re  = re.compile(r"Parallel(?:For|Reduce|Scan)<(?:[A-Za-z0-9_]+::)*([A-Za-z0-9_]+)[(<,]")
def short(n):
    m = pf_re.search(n)
    if m and m.group(1) not in ("ViewCopy", "ViewFill"):
        return m.group(1)
    if "ViewCopy" in n or "ViewFill" in n:
        return "kokkos_view_copy_fill"
    return n[:80]

anchors = [r[0] for r in rows if "fesom_ale_compute_cflz_kk" in r[2]]
print(f"steps found: {len(anchors)}")
lo, hi = args.lo, min(args.hi, len(anchors) - 2)
a, b = anchors[lo], anchors[hi + 1]
nsteps = hi + 1 - lo

busy, count = {}, {}
for (s, e, n) in rows:
    if s < a or s >= b:
        continue
    k = short(n)
    busy[k] = busy.get(k, 0.0) + (e - s)
    count[k] = count.get(k, 0) + 1

NS = 1e-6  # ns -> ms
tot = sum(busy.values())
print(f"window steps {lo}..{hi} ({nsteps} steps)  kernel-busy total {tot/nsteps*NS:.1f} ms/step")
print(f"{'ms/step':>8}  {'%busy':>6}  {'n/step':>7}  kernel")
for k in sorted(busy, key=busy.get, reverse=True)[: args.top]:
    print(f"{busy[k]/nsteps*NS:8.2f}  {busy[k]/tot*100:6.2f}  {count[k]/nsteps:7.1f}  {k}")
