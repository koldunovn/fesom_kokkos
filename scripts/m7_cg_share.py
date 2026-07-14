#!/usr/bin/env python3
"""CG share of the step, from the h9 300-step nsys trace.

CG region per step = wall from the start of the FIRST CG kernel (cg_dot/cg_spmv/
fesom_ssh_solve_cg_kk lambda) to the end of the LAST one, within each step window.
Step boundaries anchored on fesom_ale_compute_cflz_kk (the census convention).
Steady window = steps 99..296 (the census default: skip the first third).
"""
import re, sqlite3, sys

db = sys.argv[1]
con = sqlite3.connect(db)

# rank-0 device only (the census convention: one device, the first seen)
rows = con.execute("""
    SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    ORDER BY k.start
""").fetchall()

anchor_re = re.compile(r"fesom_ale_compute_cflz_kk")
cg_re     = re.compile(r"\bcg_dot\b|\bcg_spmv\b|\bcg_axpy|fesom_ssh_solve_cg_kk")

anchors = [r[0] for r in rows if anchor_re.search(r[2])]
print(f"steps found: {len(anchors)}")
lo, hi = 99, min(296, len(anchors) - 2)

cg_wall = cg_busy = step_wall = 0.0
nsteps = 0
for i in range(lo, hi + 1):
    a, b = anchors[i], anchors[i + 1]
    kt = [(s, e, n) for (s, e, n) in rows if s >= a and s < b]
    cg = [(s, e) for (s, e, n) in kt if cg_re.search(n)]
    if not cg:
        continue
    cg_wall  += cg[-1][1] - cg[0][0]
    cg_busy  += sum(e - s for (s, e) in cg)
    step_wall += b - a
    nsteps += 1

NS = 1e-6
print(f"window steps {lo}..{hi}  ({nsteps} steps with CG)")
print(f"step wall : {step_wall/nsteps*NS:8.1f} ms/step")
print(f"CG region : {cg_wall/nsteps*NS:8.1f} ms/step  ({cg_wall/step_wall*100:.1f}% of step)")
print(f"CG busy   : {cg_busy/nsteps*NS:8.1f} ms/step  ({cg_busy/step_wall*100:.1f}% of step)")
