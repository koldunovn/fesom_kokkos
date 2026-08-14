#!/usr/bin/env python3
"""M12b — what drives the barotropic block's load imbalance?

s4 measured that 24-48 % of the bt "wait" is the block's own imbalance being absorbed
at the exchange (scripts/m12b_wait_anatomy.py), which no message-count lever can
touch. The subcycle is 2-D work — eta over owned nodes, Ubt over owned elements, M
times per step — so if the partition balanced 2-D entities the block would be
balanced. This checks whether it does, by correlating each rank's measured bt busy
with what its my_list says it owns.

Usage: m12b_bt_imbalance.py <dist_dir> <run_dir_with_phasestats> [--phase bt]
"""
import sys, os, re
import numpy as np

argv = list(sys.argv[1:])
# 🔴 The per-rank columns follow the SUMMARY table's phase order, and that order has
# changed between binaries (an older CUDA build printed a header with 8 names against
# 9 columns). So the phase list is read from the summary rows of the run itself, with
# this list only as a fallback, and the extracted mean is checked against the summary.
PHASES = ['force', 'ice', 'icedyn', 'iceadv', 'coupl', 'ocean', 'cg', 'bt', 'other']


def phase_order(path):
    """Phase names in per-rank column order, from the run's own summary table."""
    names, means = [], {}
    row = re.compile(r'\[phasestats\]\s+(\w+)\s*\|([^|]*)\|([^|]*)\|\s*([\d.]+)')
    for line in open(path, errors='ignore'):
        m = row.search(line)
        if not m or m.group(1) in ('phase', 'TOTAL'):
            continue
        if m.group(1) in names:
            break                                   # second report; one is enough
        names.append(m.group(1))
        b = [float(x) for x in m.group(2).replace('/', ' ').split() if not x.startswith('@')]
        w = [float(x) for x in m.group(3).replace('/', ' ').split() if not x.startswith('@')]
        means[m.group(1)] = (b[1] if len(b) > 1 else None,
                             w[1] if len(w) > 1 else None,
                             float(m.group(4)))
    return (names or PHASES), means
phase = 'bt'
while '--phase' in argv:
    i = argv.index('--phase'); phase = argv[i + 1]; del argv[i:i + 2]
dist, run = argv[0], argv[1]
names, summary = phase_order(os.path.join(run, 'run.log'))
if phase not in names:
    sys.exit(f"no phase '{phase}' in {run}/run.log (found {names})")
col = names.index(phase)

rank_re = re.compile(r'\[phasestats-rank\]\s+(\d+)\s*\|(.*)\|(.*)$')
busy, wait = {}, {}
for line in open(os.path.join(run, 'run.log'), errors='ignore'):
    m = rank_re.search(line)
    if m:
        b = [float(x) for x in m.group(2).split()]
        w = [float(x) for x in m.group(3).split()]
        if len(b) > col:
            busy[int(m.group(1))] = b[col];  wait[int(m.group(1))] = w[col]

own_n, own_e, halo_n, halo_e = {}, {}, {}, {}
for fn in os.listdir(dist):
    if not fn.startswith('my_list'):
        continue
    tok = np.fromfile(os.path.join(dist, fn), sep=' ', dtype=np.int64, count=4)
    rank, myn, edn = int(tok[0]), int(tok[1]), int(tok[2])
    t = np.fromfile(os.path.join(dist, fn), sep=' ', dtype=np.int64)
    i = 3 + myn + edn
    mye, ede, exe = int(t[i]), int(t[i + 1]), int(t[i + 2])
    own_n[rank] = myn;  halo_n[rank] = edn
    own_e[rank] = mye;  halo_e[rank] = ede + exe

rs = sorted(set(busy) & set(own_n))
b  = np.array([busy[r] for r in rs]);      w  = np.array([wait[r] for r in rs])
on = np.array([own_n[r] for r in rs], float);  oe = np.array([own_e[r] for r in rs], float)
hn = np.array([halo_n[r] for r in rs], float); he = np.array([halo_e[r] for r in rs], float)

sb, sw, _ = summary.get(phase, (None, None, None))
if sb and abs(b.mean() - sb) > 0.05 * sb:
    sys.exit(f"per-rank busy mean {b.mean():.2f} != summary {sb:.2f} — column mapping is wrong")

print(f"=== {os.path.basename(dist)} / {os.path.basename(run)}  ({len(rs)} ranks, phase '{phase}') ===")
def spread(x, name, unit=''):
    print(f"  {name:<22} min {x.min():>9.1f} mean {x.mean():>9.1f} max {x.max():>9.1f}{unit}"
          f"   max/min {x.max()/max(x.min(),1e-9):.2f}")
spread(b,  f'{phase} busy (ms)')
spread(on, 'owned nodes')
spread(oe, 'owned elements')
spread(hn, 'halo nodes')
spread(he, 'halo elements')
print()
for name, x in (('owned nodes', on), ('owned elements', oe), ('owned nodes+elements', on + oe),
                ('halo nodes', hn), ('halo elements', he), ('owned+halo elements', oe + he)):
    print(f"  corr({phase} busy, {name:<21}) = {np.corrcoef(b, x)[0,1]:+.3f}")
print()
# how much of the busy spread would a perfectly 2-D-balanced partition remove?
pred = np.polyval(np.polyfit(on + oe, b, 1), on + oe)
r2 = 1 - ((b - pred) ** 2).sum() / ((b - b.mean()) ** 2).sum()
print(f"  linear fit  busy ~ a*(owned nodes+elements) + c   ->  R^2 = {r2:.3f}")
print(f"  busy spread max-min: {b.max()-b.min():.2f} ms; the part the fit explains: "
      f"{pred.max()-pred.min():.2f} ms")
print(f"  2-D entity imbalance max/mean = {(on+oe).max()/(on+oe).mean():.3f}; "
      f"{phase} busy max/mean = {b.max()/b.mean():.3f}")


def partition_only(dist):
    """Entity imbalance from the partition files alone — no run needed.
    Predicts the bt block's imbalance share at points we have not yet run."""
    on, oe = [], []
    for fn in os.listdir(dist):
        if not fn.startswith('my_list'):
            continue
        t = np.fromfile(os.path.join(dist, fn), sep=' ', dtype=np.int64)
        myn, edn = int(t[1]), int(t[2])
        i = 3 + myn + edn
        on.append(myn);  oe.append(int(t[i]))
    on = np.array(on, float);  oe = np.array(oe, float)
    return dict(ranks=len(on),
                n_max_mean=on.max() / on.mean(), n_maxmin=on.max() / on.min(),
                e_max_mean=oe.max() / oe.mean(), e_maxmin=oe.max() / oe.min(),
                e_per_n=oe.mean() / on.mean())
