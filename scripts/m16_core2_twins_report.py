#!/usr/bin/env python
"""M16 — report over the arms of one jobs/job_m16_core2_twins run (login node: reads logs + snapshots only).

  m16_core2_twins_report.py <root> [--pairs a:b,c:d] [--ref dp_off]

Per arm: rc, precision banner, non-finite lines, CG iteration series (from the per-step diag line
`  <n>  it=<k> ...`), last-step T/S ranges. Per pair: mean/max |Δit|, and per-field max|Δ| and
relative L2 on the LAST snapshot (T, S, eta_n, u, v). `--ref` adds every arm vs the reference."""
import sys, os, re, glob, itertools
import numpy as np
try:
    import netCDF4 as nc
except ImportError:
    nc = None

root = sys.argv[1]
pairs = []; ref = None
a = sys.argv[2:]
while a:
    if a[0] == '--pairs': pairs += [tuple(p.split(':')) for p in a[1].split(',')]; a = a[2:]
    elif a[0] == '--ref': ref = a[1]; a = a[2:]
    else: a = a[1:]
arms = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, 'rc')))
it_re = re.compile(r'^\s+(\d+)\s+it=\s*(\d+)\s')
info = {}
for arm in arms:
    d = os.path.join(root, arm)
    log = open(os.path.join(d, 'run.log'), errors='replace').read().splitlines()
    err = open(os.path.join(d, 'run.err'), errors='replace').read()
    rc = open(os.path.join(d, 'rc')).read().strip()
    prec = next((l.split('PRECISION:')[1].split()[0] for l in log if 'PRECISION:' in l), '?')
    its = {int(m.group(1)): int(m.group(2)) for m in (it_re.match(l) for l in log) if m}
    nonfin = len(re.findall(r'\b(nan|inf)\b', err, flags=re.I))
    last = next((l for l in reversed(log) if it_re.match(l)), '')
    snaps = sorted(glob.glob(os.path.join(d, 'snap_*.nc')))
    info[arm] = dict(rc=rc, prec=prec, its=its, nonfin=nonfin, last=last.strip(), snaps=snaps)
    print(f"[{arm}] rc={rc} {prec} steps_with_it={len(its)} mean_it={np.mean(list(its.values())) if its else float('nan'):.2f} "
          f"nonfinite_lines={nonfin} snaps={len(snaps)}")
    if last: print("    last: " + last[:140])
if ref and ref in info:
    pairs += [(ref, x) for x in arms if x != ref and (ref, x) not in pairs]
def field(path, name):
    with nc.Dataset(path) as f:
        if name not in f.variables: return None
        return np.array(f.variables[name][:], dtype=np.float64)
for x, y in pairs:
    if x not in info or y not in info: print(f"pair {x}:{y}: missing arm"); continue
    ix, iy = info[x]['its'], info[y]['its']
    common = sorted(set(ix) & set(iy))
    if common:
        d = np.array([abs(ix[k] - iy[k]) for k in common])
        print(f"pair {x} vs {y}: CG |Δit| mean={d.mean():.2f} max={d.max()} over {len(common)} steps "
              f"(it_x mean {np.mean([ix[k] for k in common]):.1f}, it_y mean {np.mean([iy[k] for k in common]):.1f})")
    if nc is None or not info[x]['snaps'] or not info[y]['snaps']:
        print("    (no snapshot comparison: netCDF4 missing or no snapshots)"); continue
    sx, sy = info[x]['snaps'][-1], info[y]['snaps'][-1]
    for v in ('T', 'S', 'eta_n', 'u', 'v', 'w'):
        fx, fy = field(sx, v), field(sy, v)
        if fx is None or fy is None or fx.shape != fy.shape: continue
        m = np.isfinite(fx) & np.isfinite(fy)
        diff = fx[m] - fy[m]; den = np.sqrt(np.sum(fx[m] ** 2)) or 1.0
        print(f"    {v:6s} max|Δ|={np.max(np.abs(diff)):.3e}  relL2={np.sqrt(np.sum(diff**2))/den:.3e}  "
              f"mean(Δ)={diff.mean():+.3e}  (last snapshots {os.path.basename(sx)} / {os.path.basename(sy)})")
