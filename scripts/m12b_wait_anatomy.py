#!/usr/bin/env python3
"""M12b — what is the barotropic block's MPI wait actually made of?

The deep-K arithmetic (WIDEHALO_M12B §7) assumed `bt wait ∝ messages/step`. The W6
CPU pairs halved the message count at essentially unchanged compute, so they measure
that elasticity directly — and per-rank phasestats say whether what remains is
latency or imbalance absorption.

For each run directory it reports, over all ranks:
  * the elasticity of the mean bt wait to the bt message count (needs the pair);
  * the correlation of a rank's bt wait with its own bt busy — a strong negative
    correlation means the wait is the block's load imbalance being absorbed at the
    exchange, not the cost of the messages;
  * the imbalance decomposition  wait_i ≈ (busy_max − busy_i) + floor, and the
    floor left at the busiest rank — the part a message-count reduction can touch.

Usage: m12b_wait_anatomy.py <label>=<run_dir> [<label>=<run_dir> ...] [--phase bt]
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

rank_re = re.compile(r'\[phasestats-rank\]\s+(\d+)\s*\|(.*)\|(.*)$')
summ_re = re.compile(r'\[phasestats\]\s+(\w+)\s*\|([^|]*)\|([^|]*)\|\s*([\d.]+)')

res = {}
for arg in argv:
    label, d = arg.split('=', 1)
    log = os.path.join(d, 'run.log')
    names, summary = phase_order(log)
    if phase not in names:
        print(f"{label}: no phase '{phase}' in {log} (found {names})");  continue
    col = names.index(phase)
    busy, wait = {}, {}
    mpi = None
    for line in open(log, errors='ignore'):
        m = rank_re.search(line)
        if m:
            r = int(m.group(1))
            b = [float(x) for x in m.group(2).split()]
            w = [float(x) for x in m.group(3).split()]
            if len(b) > col and len(w) > col:
                busy[r] = b[col]; wait[r] = w[col]
            continue
        s = summ_re.search(line)
        if s and s.group(1) == phase:
            mpi = float(s.group(4))
    if not busy:
        print(f"{label}: no per-rank phasestats in {log}");  continue
    rs = np.array(sorted(busy))
    b = np.array([busy[r] for r in rs]);  w = np.array([wait[r] for r in rs])
    res[label] = dict(b=b, w=w, mpi=mpi)

    # self-check: the per-rank column must reproduce the summary's mean for this phase
    sb, sw, smpi = summary.get(phase, (None, None, None))
    for got, want, what in ((b.mean(), sb, 'busy'), (w.mean(), sw, 'wait')):
        if want and abs(got - want) > 0.05 * max(want, 1e-9):
            print(f"  🔴 {label}: per-rank {what} mean {got:.2f} != summary {want:.2f} — "
                  f"column mapping is wrong, do not use these numbers")
    if smpi:
        mpi = smpi

    bmax = b.max()
    corr = float(np.corrcoef(b, w)[0, 1])
    # imbalance model: wait_i = (bmax - b_i) + floor_i  ->  floor = wait - deficit
    floor = w - (bmax - b)
    # least-squares slope of wait on the deficit
    A = np.vstack([bmax - b, np.ones_like(b)]).T
    slope, icept = np.linalg.lstsq(A, w, rcond=None)[0]
    print(f"\n=== {label}  ({len(rs)} ranks, phase '{phase}', mpi/step {mpi}) ===")
    print(f"  busy  min {b.min():.2f} mean {b.mean():.2f} max {b.max():.2f} ms   "
          f"(spread {b.max()-b.min():.2f})")
    print(f"  wait  min {w.min():.2f} mean {w.mean():.2f} max {w.max():.2f} ms")
    print(f"  corr(busy, wait) = {corr:+.3f}   "
          f"[strongly negative => the wait is imbalance absorption]")
    print(f"  wait ~ {slope:.3f}*(busy_max - busy) + {icept:.2f} ms      "
          f"(intercept = the wait left at the busiest rank)")
    print(f"  imbalance-explained share of the mean wait: "
          f"{100*slope*(bmax-b).mean()/w.mean():.1f} %  |  residual floor "
          f"{icept:.2f} ms = {100*icept/w.mean():.1f} %")

if len(res) == 2:
    (la, ra), (lb, rb) = list(res.items())
    if ra['mpi'] and rb['mpi']:
        e = np.log(rb['w'].mean() / ra['w'].mean()) / np.log(rb['mpi'] / ra['mpi'])
        print(f"\n=== elasticity of the mean '{phase}' wait to the message count ===")
        print(f"  {la}: {ra['mpi']:.0f} msg/step, wait {ra['w'].mean():.2f} ms")
        print(f"  {lb}: {rb['mpi']:.0f} msg/step, wait {rb['w'].mean():.2f} ms")
        print(f"  dln(wait)/dln(msg) = {e:.3f}      "
              f"[1.0 = purely latency-bound, 0 = the messages are not what is waited on]")
        print(f"  busy: {ra['b'].mean():.2f} -> {rb['b'].mean():.2f} ms "
              f"({100*(rb['b'].mean()/ra['b'].mean()-1):+.1f} %)")
