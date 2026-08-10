#!/usr/bin/env python3
"""M11 Task 10 — prune the zoo to a shortlist: the Pareto front over the objectives the
campaign has evidence for, plus the anchors that are exempt from pruning by design.

Objective set, and why each one is in it (nothing here is a guess; each traces to a measurement):

  commvol_max_rank  the volume the busiest rank waits on. M10: half the step is MPI wait.
  offnode3          3-D halo crossing a node boundary — the bytes that cost most. M11
                    Finding 16; supplied by `m11_placement.py --csv`, optional.
  n3d_maxmin        M10 measured ocean busy against 3-D nodes at r=0.967 (2-D: r=0.003).
  n2d_imb           the ice model and every per-rank 2-D allocation are billed in 2-D nodes;
                    M11 Finding 14 shows single-constraint engines trade one against the other.
  parts_disconnected  fragmentation. The authoritative fragmentation currency is element
                    replication (M11 Finding 3: it predicted M10's +20 % GPU ocean busy where
                    the halo-node count predicted +0.7 %), but that is read from the dist FILES
                    and an arm scored from a bare part vector has none. Rather than keep a
                    second, approximate implementation of it, the offline prune uses the
                    graph-side fragmentation count and `elem_repl` joins the objective set once
                    the shortlist has its dists generated (`--objectives ...,elem_repl`).

Anchors (`--anchor`) are kept whatever they score: the regression of predicted against measured
step time needs points that span the range, including deliberately bad ones.

usage:
  m11_pareto.py zoo_a.csv zoo_b.csv [--placement place.csv] [--npes 512]
                [--objectives commvol_max_rank,offnode3,n3d_maxmin,n2d_imb,elem_repl]
                [--anchor wgt2_512 --anchor a3_a0_512]
"""
import argparse
import csv
import os
import sys

DEFAULT_OBJ = "commvol_max_rank,offnode3,n3d_maxmin,n2d_imb,parts_disconnected"


def load(paths, placement=None):
    rows = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skipping missing {p})")
            continue
        for r in csv.DictReader(open(p)):
            rows.setdefault(r["arm"], {}).update(r)
    if placement and os.path.exists(placement):
        for r in csv.DictReader(open(placement)):
            if r["arm"] in rows:
                rows[r["arm"]]["offnode3"] = r["off3"]
                rows[r["arm"]]["offnode_frac3"] = r["frac3"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--placement")
    ap.add_argument("--npes", type=int)
    ap.add_argument("--objectives", default=DEFAULT_OBJ)
    ap.add_argument("--anchor", action="append", default=[])
    ap.add_argument("--constrain", action="append", default=[], metavar="KEY<=V",
                    help="feasibility filter applied BEFORE the front, e.g. n2d_imb<=1.3")
    a = ap.parse_args()

    rows = load(a.csvs, a.placement)
    if a.npes:
        rows = {k: v for k, v in rows.items() if int(float(v.get("npes", 0))) == a.npes}
    # Feasibility first, Pareto second. With five objectives almost nothing is dominated, so a
    # bare front is not a prune; the filter is where the modelling assumption lives, and it is
    # stated explicitly rather than hidden in a weighted score.
    for c in a.constrain:
        op = "<=" if "<=" in c else ">="
        key, val = c.split(op)
        val = float(val)
        keep = {k: v for k, v in rows.items()
                if key in v and v[key] not in ("", "nan")
                and (float(v[key]) <= val if op == "<=" else float(v[key]) >= val)}
        print(f"  constraint {key} {op} {val:g}: {len(keep)} of {len(rows)} arms feasible"
              f" (dropped: {', '.join(sorted(set(rows) - set(keep))) or 'none'})")
        rows = keep
    obj = [o for o in a.objectives.split(",") if o]
    usable = {k: v for k, v in rows.items() if all(o in v and v[o] not in ("", "nan") for o in obj)}
    dropped = sorted(set(rows) - set(usable))
    if dropped:
        print(f"  {len(dropped)} arm(s) lack an objective and are not ranked: {', '.join(dropped[:6])}"
              f"{' …' if len(dropped) > 6 else ''}")
    if not usable:
        return 1

    def vec(k):
        return [float(usable[k][o]) for o in obj]

    front, dominated = [], {}
    for k in usable:
        vk = vec(k)
        by = [j for j in usable if j != k and all(x <= y for x, y in zip(vec(j), vk))
              and any(x < y for x, y in zip(vec(j), vk))]
        if by:
            dominated[k] = by
        else:
            front.append(k)

    print(f"\n=== Pareto front over {', '.join(obj)}   ({len(front)} of {len(usable)} arms)")
    hdr = f"  {'arm':<24}" + "".join(f"{o[:12]:>14}" for o in obj)
    print(hdr)
    for k in sorted(front, key=lambda x: vec(x)[0]):
        print(f"  {k:<24}" + "".join(f"{float(usable[k][o]):>14,.4g}" for o in obj))
    anchors = [k for k in a.anchor if k in usable and k not in front]
    if anchors:
        print(f"\n  anchors kept despite being dominated (exempt by design):")
        for k in anchors:
            print(f"  {k:<24}" + "".join(f"{float(usable[k][o]):>14,.4g}" for o in obj))
    print(f"\n=== dominated ({len(dominated)})")
    for k in sorted(dominated, key=lambda x: vec(x)[0]):
        print(f"  {k:<24} dominated by {', '.join(sorted(dominated[k])[:3])}"
              f"{' …' if len(dominated[k]) > 3 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
