#!/usr/bin/env python3
"""E.IMB analysis of record: join a [phasestats-rank] table (any PHASESTATS run log)
with per-rank static features (m7_rank_features.py output) and, optionally, the
per-rank ice mask from an a_ice NetCDF — print spreads + correlations.

usage: m7_phasestats_join.py <run_log> <mesh_dir> <npes> [--aice <a_ice.nc>]

Parses the LAST [phasestats-rank] block in the log (a 2-rep leg prints one per rep;
the reps correlate at 0.999 so last ≈ mean). Phase columns are read from the table
header, so it works with any phase set (phst0's 6, phst1's 8, future splits).
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_part_spread import load_ranks  # noqa: E402  (rule-0.36 verified parser)


def parse_log(path):
    """-> (phase_names, busy[nrank,nph], wait[nrank,nph]) from the LAST table."""
    hdr, rows = None, {}
    for line in open(path):
        m = re.match(r"(?:\d+: )?\[phasestats-rank\]\s+rk \|\s+busy: ([a-z+ ]+)\|", line)
        if m:
            hdr = m.group(1).split()
            rows = {}          # a new table header restarts the collection
            continue
        m = re.match(r"(?:\d+: )?\[phasestats-rank\]\s+(\d+) \|\s+([-\d. ]+)\|\s+([-\d. ]+)$",
                     line.rstrip())
        if m:
            rows[int(m.group(1))] = ([float(x) for x in m.group(2).split()],
                                     [float(x) for x in m.group(3).split()])
    if not hdr or not rows:
        sys.exit(f"no [phasestats-rank] table in {path}")
    n = max(rows) + 1
    busy = np.array([rows[r][0] for r in range(n)])
    wait = np.array([rows[r][1] for r in range(n)])
    return hdr, busy, wait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_log")
    ap.add_argument("mesh_dir")
    ap.add_argument("npes", type=int)
    ap.add_argument("--aice", help="a_ice NetCDF (nod2D field) for the real ice mask")
    a = ap.parse_args()

    phases, busy, wait = parse_log(a.run_log)
    n = busy.shape[0]
    assert n == a.npes, (n, a.npes)

    rank_of, cnt = load_ranks(a.mesh_dir, a.npes)
    nlvls = np.loadtxt(f"{a.mesh_dir}/nlvls.out", dtype=np.int64)
    feats = {"n2d": cnt.astype(float)}
    s3 = np.zeros(n)
    np.add.at(s3, rank_of, nlvls.astype(float))
    feats["n3d"] = s3
    # halo/partner features from my_list + com_info (nod2D recv side)
    myE = np.zeros(n); eE = np.zeros(n); eN = np.zeros(n); npart = np.zeros(n)
    for r in range(n):
        raw = np.fromstring(open(f"{a.mesh_dir}/dist_{a.npes}/my_list{r:05d}.out").read(),
                            dtype=np.int64, sep=" ")
        i = 1
        myN, eNr = raw[i], raw[i + 1]; i += 2 + myN + eNr
        myE[r], eE[r] = raw[i], raw[i + 1]
        eN[r] = eNr
        raw = np.fromstring(open(f"{a.mesh_dir}/dist_{a.npes}/com_info{r:05d}.out").read(),
                            dtype=np.int64, sep=" ")
        npart[r] = raw[1]
    feats.update(myElem=myE, eElem=eE, eNod=eN, nPart=npart)
    if a.aice:
        from netCDF4 import Dataset
        ai = np.array(Dataset(a.aice).variables["a_ice"][-1]).ravel()
        f15 = np.zeros(n); am = np.zeros(n)
        for r in range(n):
            m = rank_of == r
            am[r] = ai[m].mean(); f15[r] = (ai[m] > 0.15).mean()
        feats.update(aice_mean=am, aice_frac15=f15)

    print(f"# {a.run_log}  ranks={n}  phases={phases}")
    print(f"# {'phase':8s} {'min':>8s} {'mean':>8s} {'max':>8s} {'spread':>8s} {'argmax':>6s}")
    for i, ph in enumerate(phases):
        b = busy[:, i]
        if b.max() - b.min() < 0.05 and b.mean() < 0.1:
            continue
        print(f"  {ph:8s} {b.min():8.1f} {b.mean():8.1f} {b.max():8.1f} "
              f"{b.max()-b.min():8.1f} {int(b.argmax()):6d}")
    tot = busy.sum(1)
    print(f"  {'TOTAL':8s} {tot.min():8.1f} {tot.mean():8.1f} {tot.max():8.1f} "
          f"{tot.max()-tot.min():8.1f} {int(tot.argmax()):6d}")
    print("\n# correlations busy[phase] vs feature (|r| >= 0.35 only):")
    for i, ph in enumerate(phases):
        b = busy[:, i]
        if b.std() < 0.05:
            continue
        for fn, x in feats.items():
            if np.std(x) == 0:
                continue
            r = np.corrcoef(b, x)[0, 1]
            if abs(r) >= 0.35:
                print(f"  busy_{ph:8s} vs {fn:12s} r={r:+.2f}")


if __name__ == "__main__":
    main()
