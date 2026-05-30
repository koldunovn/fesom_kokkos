#!/usr/bin/env python3
"""Summarize the M5.19 ncu A/B coalescing metrics: per-kernel mean duration / SM% /
occupancy / LD+ST sectors-per-request, before vs after. Usage:
    ncu_coalesce_summary.py <before.csv> <after.csv>"""
import csv, sys, re

DUR = 'gpu__time_duration.sum'
SM  = 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
OCC = 'sm__warps_active.avg.pct_of_peak_sustained_active'
LD  = 'l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio'
ST  = 'l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio'

def label(name):
    m = re.search(r'fesom_(\w+?)_kk\(.*?instance (\d+)', name)
    if m:
        fn, inst = m.group(1), m.group(2)
        # vel_rhs has 2 lambdas: instance1=vel_rhs_elem, instance2=vel_rhs_assembly
        if fn == 'compute_vel_rhs':
            return 'vel_rhs_elem' if inst == '1' else 'vel_rhs_assembly'
        return fn
    m = re.search(r'fesom_(\w+?)_kk', name)
    return m.group(1) if m else name[:40]

def load(path):
    out = {}
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception as e:
        return None
    rows = [r for r in rows if r.get('Kernel Name')]
    for r in rows:
        k = label(r['Kernel Name'])
        def f(col):
            try: return float(r.get(col, '') or 'nan')
            except: return float('nan')
        out.setdefault(k, []).append((f(DUR), f(SM), f(OCC), f(LD), f(ST)))
    agg = {}
    for k, v in out.items():
        n = len(v)
        agg[k] = tuple(sum(x[i] for x in v)/n for i in range(5)) + (n,)
    return agg

b = load(sys.argv[1]); a = load(sys.argv[2])
if b is None or a is None:
    print("missing csv"); sys.exit(1)
print(f"{'kernel':22s} {'dur ms (B→A)':>22s} {'SM% (B→A)':>16s} {'occ% (B→A)':>16s} {'LD s/req':>14s} {'ST s/req':>14s}")
for k in sorted(set(b)|set(a)):
    if k not in b or k not in a: continue
    bd,bs,bo,bl,bt,bn = b[k]; ad,as_,ao,al,at,an = a[k]
    spd = bd/ad if ad else float('nan')
    print(f"{k:22s} {bd:8.2f}→{ad:7.2f} ({spd:4.1f}x) {bs:6.1f}→{as_:5.1f} {bo:6.1f}→{ao:5.1f} {bl:5.1f}→{al:4.1f} {bt:5.1f}→{at:4.1f}")
