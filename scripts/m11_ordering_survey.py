#!/usr/bin/env python3
"""M11: does THIS mesh need renumbering? — the offline diagnosis, before any node-hour.

The ordering lever is NOT universal, and this is the tool that says so per mesh. CORE2's shipped
numbering is spatially arbitrary (mean |Δindex| over graph edges 32,043 of 126,858 nodes), so
every ordering helps it. fArc's is already local (956 over 638,387), so the two space-filling
curves make it WORSE (+10.6 %, +14.3 %) and only RCM helps. The decision therefore has to be
taken per mesh, from the numbers below, and it costs no cluster time to take.

Read-only: it loads `nod2d/elem2d/nlvls/elvls` and writes nothing, so it can point straight at
a production mesh.

usage:
  m11_ordering_survey.py <mesh_dir> [<mesh_dir> ...] [--orders hilbert-xyz,s2,rcm] [--csv f.csv]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_scorecard import Mesh
from m11_renumber import elem_order, node_order


def locality(gi, gj, new_of_old=None):
    if new_of_old is None:
        a, b = gi, gj
    else:
        a, b = new_of_old[gi], new_of_old[gj]
    d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    return float(d.mean()), float(np.percentile(d, 95))


def stride_l64(stream):
    s = np.abs(np.diff(stream.astype(np.int64)))
    return float(np.count_nonzero(s <= 64) / max(s.size, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--orders", default="hilbert-xyz,s2,rcm")
    ap.add_argument("--csv")
    a = ap.parse_args()
    orders = [o for o in a.orders.split(",") if o]
    rows = []

    for md in a.meshes:
        name = os.path.basename(md.rstrip("/"))
        t0 = time.time()
        mesh = Mesh(md, need_edges=False)
        gi, gj = mesh.graph()
        base_mean, base_p95 = locality(gi, gj)
        base_l64 = stride_l64(mesh.elem.reshape(-1))
        print(f"\n=== {name}   {mesh.nod2D:,} nodes, {mesh.elem2D:,} elements, "
              f"{gi.size:,} graph edges   ({time.time() - t0:.0f} s to load)")
        print(f"  {'ordering':<14}{'mean |di|':>14}{'p95 |di|':>14}"
              f"{'elem stride <=64':>19}{'verdict':>12}")
        print(f"  {'shipped':<14}{base_mean:>14,.0f}{base_p95:>14,.0f}"
              f"{base_l64:>18.1%}{'—':>12}")
        row = dict(mesh=name, nod2D=mesh.nod2D, elem2D=mesh.elem2D,
                   shipped_didx=base_mean, shipped_l64=base_l64)
        for o in orders:
            t1 = time.time()
            try:
                old_in_new = node_order(mesh, o)
            except Exception as exc:                      # noqa: BLE001 - report, do not abort
                print(f"  {o:<14}FAILED: {exc}")
                continue
            new_of_old = np.empty_like(old_in_new)
            new_of_old[old_in_new] = np.arange(old_in_new.size)
            m, p95 = locality(gi, gj, new_of_old)
            # The element-gather stride depends on the ELEMENT order too, and a real
            # renumbering reorders elements as well (by min new vertex, m11_renumber's
            # default). Measuring it with the old element order understates the improvement —
            # CORE2+hilbert reads 79.8 % that way against the 86.8 % the actual conversion
            # produced. So reorder the elements here as well, and quote the real number.
            eo = elem_order(mesh, new_of_old, "minvertex")
            l64 = stride_l64(new_of_old[mesh.elem[eo]].reshape(-1))
            gain = 100 * (m / base_mean - 1)
            verdict = "helps" if gain < -20 else ("HURTS" if gain > 0 else "marginal")
            print(f"  {o:<14}{m:>14,.0f}{p95:>14,.0f}{l64:>18.1%}{verdict:>12}"
                  f"   ({gain:+.1f} %, {time.time() - t1:.0f} s)")
            row[f"{o}_didx"] = m
            row[f"{o}_l64"] = l64
            row[f"{o}_gain_pct"] = gain
        rows.append(row)
        del mesh, gi, gj

    if a.csv and rows:
        import csv as _csv
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["mesh"] + [k for k in keys if k != "mesh"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n  csv -> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
