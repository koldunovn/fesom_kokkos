#!/usr/bin/env python3
"""M11: target-share feedback — give the ranks with expensive HALOS less owned work.

The problem this addresses. On GPU at moderate rank counts the owned work is already balanced
(fArc, 16 GPUs: owned 2-D max/min 1.00, owned 3-D 1.02) and the per-rank busy time still spreads
1.30x. The one per-rank quantity that tracks it is the halo, which spreads 2.89x. No vertex
weight can fix that: the halo is a property of the CUT, not of a vertex, and it is not known
until after the partition exists.

What can be done instead is to compensate for it. METIS accepts per-part TARGET SHARES
(`tpwgts`, exposed as `FESOM_PART_TPWGTS_FILE`), so a rank that will carry an expensive halo can
be given proportionally less owned work. The halo changes when the partition changes, so this is
a fixed-point iteration, not a one-shot correction.

    cost_p  =  owned_p + kappa * halo_p            (kappa = cost of a halo node in units of an
                                                    owned 3-D node, measured, not assumed)
    share_p =  (1/cost-ratio) normalised to sum 1, damped by `--damping`

This file computes the shares and reports the predicted max-cost change. It does NOT decide
whether the lever works — that needs a race, and the predicted gain has to be checked against
the fit quality (at fArc/16 the model explained only 28 % of the busy variance, which is not
enough to justify shipping anything).

usage:
  m11_tpwgts.py <mesh_dir> --npes 16 --kappa 40 -o tpwgts.txt [--damping 0.5]
  m11_tpwgts.py <mesh_dir> --npes 16 --fit-from <phasestats.log>      # measure kappa instead
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_scorecard import Mesh, load_dist_files
from m7_part_spread import load_ranks


def per_rank(mesh_dir, npes):
    mesh = Mesh(mesh_dir, need_edges=False)
    d = load_dist_files(mesh_dir, npes)
    halo = np.array([x["eDim_nod2D"] for x in d["my_list"]], float)
    part, _ = load_ranks(mesh_dir, npes)
    own3d = np.bincount(np.asarray(part), weights=mesh.nlev_nod, minlength=npes)
    return own3d, halo


def busy_from_log(path):
    rows = {}
    for line in open(path, errors="ignore"):
        m = re.match(r"\[phasestats-rank\]\s+(\d+) \|(.*)\|(.*)", line)
        if m:
            rows[int(m.group(1))] = sum(float(x) for x in m.group(2).split()[1:])
    return np.array([rows[r] for r in range(len(rows))]) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir")
    ap.add_argument("--npes", type=int, required=True)
    ap.add_argument("--kappa", type=float, help="cost of one halo node in owned-3-D-node units")
    ap.add_argument("--fit-from", help="a PHASESTATS log to measure kappa from")
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    own, halo = per_rank(a.mesh_dir, a.npes)
    kappa = a.kappa
    if a.fit_from:
        busy = busy_from_log(a.fit_from)
        if busy is None or busy.size != a.npes:
            sys.exit("could not read a per-rank busy table of the right length")
        X = np.column_stack([np.ones(a.npes), own, halo])
        beta, *_ = np.linalg.lstsq(X, busy, rcond=None)
        pred = X @ beta
        r2 = 1 - ((busy - pred) ** 2).sum() / ((busy - busy.mean()) ** 2).sum()
        kappa = beta[2] / beta[1] if beta[1] else float("nan")
        print(f"  fit on {a.npes} ranks: busy = {beta[0]:.2f} + {beta[1]:.3e}*owned3d "
              f"+ {beta[2]:.3e}*halo   R^2={r2:.2f}")
        print(f"  => kappa = {kappa:.1f} owned-3-D-nodes per halo node")
        if r2 < 0.5:
            print("  !! R^2 below 0.5 — this cost model does not describe the machine well "
                  "enough to act on; treat the shares below as an experiment, not a fix")
    if kappa is None or not np.isfinite(kappa):
        sys.exit("give --kappa or --fit-from")

    cost = own + kappa * halo
    share = cost.mean() / cost                      # less work where the halo is expensive
    share = 1.0 + a.damping * (share - 1.0)
    share = share / share.sum()
    newcost = share * a.npes * own.mean() + kappa * halo    # halo held fixed: first-order only
    print(f"  owned 3-D  max/min {own.max()/own.min():.2f}   halo max/min {halo.max()/halo.min():.2f}")
    print(f"  cost       max/mean {cost.max()/cost.mean():.3f}  ->  predicted "
          f"{newcost.max()/newcost.mean():.3f}   ({100*(newcost.max()/cost.max()-1):+.1f} % on the max)")
    print(f"  shares     min {share.min():.5f}  max {share.max():.5f}  (uniform = {1/a.npes:.5f})")
    print("  NOTE first-order only: the halo moves when the partition moves, so iterate "
          "(regenerate, re-measure, recompute) until the max cost stops falling.")
    if a.out:
        np.savetxt(a.out, share, fmt="%.8f")
        print(f"  wrote {a.out} ({a.npes} shares, sum {share.sum():.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
