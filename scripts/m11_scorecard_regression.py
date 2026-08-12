#!/usr/bin/env python3
"""M11: does any scorecard column predict the measured step time?

This is the campaign's central question stated as a regression instead of as a handful of hand-read
points. The mechanism claims — "GPU pays per message, CPU pays for owned 3-D work" — were each
established at one or two points; this asks whether they hold across every arm we raced.

Method, and why it is not a single global fit:

  A cross-point fit would be dominated by between-point variation (a 2048-rank fArc arm and a
  4-rank CORE2 arm share nothing), and the project's standing rule is that a ratio is only earned
  on a matched pair (L88). So the correlation is computed WITHIN each (mesh, backend, ranks) group
  that raced at least three arms at its ladder dt, and the groups are then summarised. A metric
  that predicts within groups is a mechanism; one that only predicts across them is a mesh-size
  proxy wearing a mechanism's clothes.

  Spearman, not Pearson: we care about the ORDERING of arms, and one 105 %-slower outlier would
  otherwise set the slope.

    m11_scorecard_regression.py [--races m11_races.csv] [--csv out.csv]
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

# Scorecard columns worth testing, with the mechanism each one would support if it won.
METRICS = {
    "nbr_max": "max communication partners (the GPU claim)",
    "nbr_mean": "mean communication partners",
    "n3d_maxmin": "owned 3-D work imbalance (the CPU claim)",
    "n2d_imb": "owned 2-D imbalance",
    "commvol_total": "total communication volume",
    "commvol_max_rank": "max per-rank volume (known seed-noisy, Finding 18)",
    "edgecut_unweighted": "edge cut",
    "halo_nod_mean": "mean halo size",
    "halo_nod_max": "max halo size",
    "elem_repl": "element replication",
    "parts_disconnected": "disconnected parts",
}

ZOOARM = re.compile(r"/mesh_m11/zoo/([a-z0-9]+)/([^/]+)/?$")


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def load_scorecards():
    """(mesh, arm, ranks) -> metric dict, from the METIS zoo AND the engine scorecards.

    The two families name their arms differently: the zoo directory is `b_kahip_a100` while the
    engine scorecard row is `kahip_a100`. Register each row under both spellings — leaving the
    engines out would drop the winners at fArc 2048 CPU and CORE2 864 from the regression, which
    is precisely where a scorecard claim would matter most.
    """
    out = {}

    def put(mesh, arm, n, row):
        out[(mesh, arm, n)] = row
        if not arm.startswith("b_"):
            out.setdefault((mesh, "b_" + arm, n), row)

    for p in glob.glob("/work/ab0995/a270088/port2/m11/zooa.*/zoo_a_*.csv"):
        mesh = os.path.basename(p)[len("zoo_a_"):-len(".csv")]
        for r in csv.DictReader(open(p)):
            arm, _, n = r["arm"].rpartition("_")
            if n.isdigit():
                put(mesh, arm, int(n), r)

    for p in glob.glob("/work/ab0995/a270088/port2/m11/bdist.*/zoo_b_dists_*.csv"):
        m = re.match(r"zoo_b_dists_(.+)_k(\d+)\.csv$", os.path.basename(p))
        if not m:
            continue
        mesh, n = m.group(1), int(m.group(2))
        for r in csv.DictReader(open(p)):
            arm, _, tail = r["arm"].rpartition("_")
            if tail.isdigit():
                put(mesh, arm, int(tail), r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", default="/work/ab0995/a270088/port2/m11/m11_races.csv")
    ap.add_argument("--csv")
    a = ap.parse_args()

    sc = load_scorecards()
    if not sc:
        sys.exit("no scorecard CSVs found")

    groups = defaultdict(list)
    unmatched = 0
    for r in csv.DictReader(open(a.races)):
        if r["is_base"] == "True" or r["status"] != "ok":
            continue
        if r.get("protocol") == "False":
            continue
        m = ZOOARM.search(r["meshdir"])
        if not m:
            continue
        key = (m.group(1), m.group(2), int(r["ranks"]))
        if key not in sc:
            unmatched += 1
            continue
        groups[(m.group(1), r["backend"], int(r["ranks"]))].append((float(r["delta_pct"]), sc[key],
                                                                    m.group(2), r["job"]))

    usable = {k: v for k, v in groups.items() if len(v) >= 3}
    print(f"  {len(groups)} raced group(s); {len(usable)} with >=3 protocol arms; "
          f"{unmatched} arm(s) had no scorecard row\n")
    for k in sorted(usable):
        print(f"    {k[0]:<7} {k[1].upper():<4} {k[2]:>5} ranks   arms: "
              f"{', '.join(sorted(x[2] for x in usable[k]))}")
    if not usable:
        sys.exit("\nno group has three protocol arms with scorecards — nothing to regress")

    rows = []
    print(f"\n  {'metric':<22}{'backend':<8}{'groups':>7}{'median rho':>12}{'range':>18}   meaning")
    for backend in ("cpu", "gpu"):
        for met, meaning in METRICS.items():
            rhos = []
            for k, v in usable.items():
                if k[1] != backend:
                    continue
                xs, ys = [], []
                for d, row, _, _ in v:
                    if met not in row or row[met] in ("", None):
                        continue
                    xs.append(float(row[met]))
                    ys.append(d)
                rho = spearman(xs, ys) if len(xs) >= 3 else None
                if rho is not None:
                    rhos.append(rho)
                    rows.append(dict(backend=backend, metric=met, mesh=k[0], ranks=k[2],
                                     n_arms=len(xs), rho=rho))
            if not rhos:
                continue
            rhos.sort()
            med = rhos[len(rhos) // 2]
            print(f"  {met:<22}{backend:<8}{len(rhos):>7}{med:>12.2f}"
                  f"{f'[{min(rhos):+.2f}, {max(rhos):+.2f}]':>18}   {meaning}")

    print("\n  rho > 0 means a LARGER metric goes with a SLOWER step, i.e. the metric is a cost.")
    print("  Read the range, not the median: with a handful of groups a median is one number away")
    print("  from flipping, and Finding 18 already retired one metric for seed noise.")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {a.csv} ({len(rows)} per-group rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
