#!/usr/bin/env python3
"""M11: score the PLACEMENT of a partition — which subdomain becomes which MPI rank.

Why this tool exists. Task-2 Finding 4: the shipped CORE2 `dist_864` and our flat regeneration
are indistinguishable on every invariant metric (cut 34,159 vs 34,157, halo 42.1 vs 42.1, elem
replication 1.473 vs 1.474) yet M10 measured the shipped one 7.4 % faster. Partition QUALITY
cannot explain that, so the difference has to be in something the scorecard is blind to by
construction: the rank LABELS, and therefore which subdomains share a node, a socket, a GPU.

This tool measures exactly that. With R ranks per node, rank r lives on node r//R, and the
question is how much of the halo traffic has to leave the node:

    off-node volume  = sum over rank pairs on DIFFERENT nodes of the halo they exchange
    on-node fraction = 1 - off-node/total

Both in two currencies: 2-D halo nodes (message setup, ice) and 3-D halo nodes (the bytes the
ocean actually ships). A relabelling changes neither the partition nor any scorecard row — only
these numbers.

usage:
  m11_placement.py <mesh_dir> --dist 864 --ranks-per-node 128
  m11_placement.py <mesh_dir> --part-file v.txt --npes 512 --ranks-per-node 4 --compare-relabel
  m11_placement.py <mesh_dir> --dist 864 --ranks-per-node 128 --emit-relabelled out.txt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_scorecard import Mesh
from m7_part_spread import load_ranks


def halo_pairs(mesh, part, npes):
    """-> dict (p,q) -> (n2d, n3d): nodes owned by q that p must import (p != q)."""
    gi, gj = mesh.graph()
    pi, pj = part[gi], part[gj]
    cut = pi != pj
    gi, gj, pi, pj = gi[cut], gj[cut], pi[cut], pj[cut]
    # (importer, owner, node) both ways round the cut edge
    imp = np.concatenate([pi, pj])
    own = np.concatenate([pj, pi])
    nod = np.concatenate([gj, gi])
    key = (imp * npes + own).astype(np.int64) * mesh.nod2D + nod
    key = np.unique(key)                       # a node is imported once per (importer, owner)
    nod = (key % mesh.nod2D).astype(np.int64)
    po = (key // mesh.nod2D).astype(np.int64)
    imp, own = po // npes, po % npes
    lev = mesh.nlev_nod[nod]
    return imp, own, lev


def volumes(imp, own, lev, npes, rpn):
    node_of = np.arange(npes) // rpn
    off = node_of[imp] != node_of[own]
    tot2, tot3 = imp.size, int(lev.sum())
    off2, off3 = int(off.sum()), int(lev[off].sum())
    # per-node outgoing off-node volume, for the max
    per = np.zeros(node_of.max() + 1, dtype=np.int64)
    np.add.at(per, node_of[own[off]], lev[off])
    return dict(tot2=tot2, tot3=tot3, off2=off2, off3=off3,
                frac2=off2 / max(tot2, 1), frac3=off3 / max(tot3, 1),
                pernode_max=int(per.max()) if per.size else 0,
                pernode_mean=float(per.mean()) if per.size else 0.0)


def relabel_greedy(imp, own, lev, npes, rpn):
    """Group ranks into nodes by agglomerating the part-communication graph.

    Deliberately simple and deterministic: repeatedly take the unplaced rank with the largest
    remaining traffic, then fill its node with the unplaced ranks it talks to most. This is not
    a partitioner; it is a lower-effort bound on what placement can buy, and it needs no
    external dependency to be reproducible.
    """
    w = {}
    for a, b, l in zip(imp, own, lev):
        k = (a, b) if a < b else (b, a)
        w[k] = w.get(k, 0) + int(l)
    nbr = [dict() for _ in range(npes)]
    for (a, b), v in w.items():
        nbr[a][b] = nbr[a].get(b, 0) + v
        nbr[b][a] = nbr[b].get(a, 0) + v
    load = np.array([sum(d.values()) for d in nbr], dtype=np.int64)
    unplaced = set(range(npes))
    order = []
    while unplaced:
        seed = max(unplaced, key=lambda r: (load[r], -r))
        group = [seed]
        unplaced.discard(seed)
        while len(group) < rpn and unplaced:
            score = {}
            for g in group:
                for r, v in nbr[g].items():
                    if r in unplaced:
                        score[r] = score.get(r, 0) + v
            pick = max(score, key=lambda r: (score[r], -r)) if score else \
                max(unplaced, key=lambda r: (load[r], -r))
            group.append(pick)
            unplaced.discard(pick)
        order.extend(group)
    newlabel = np.empty(npes, dtype=np.int64)
    newlabel[np.array(order, dtype=np.int64)] = np.arange(npes)   # old rank -> new rank
    return newlabel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir")
    ap.add_argument("--dist", type=int)
    ap.add_argument("--part-file")
    ap.add_argument("--npes", type=int)
    ap.add_argument("--ranks-per-node", type=int, default=128)
    ap.add_argument("--compare-relabel", action="store_true")
    ap.add_argument("--emit-relabelled", metavar="FILE")
    ap.add_argument("--label", default="")
    ap.add_argument("--arm", help="arm name for the csv row (joins with the scorecard's `arm`)")
    ap.add_argument("--csv", help="append one row per run")
    a = ap.parse_args()

    npes = a.dist or a.npes
    if not npes:
        ap.error("--dist N or --part-file with --npes")
    mesh = Mesh(a.mesh_dir, need_edges=False)
    if a.part_file:
        part = np.loadtxt(a.part_file, dtype=np.int64).reshape(-1)
    else:
        part, _ = load_ranks(a.mesh_dir, npes)
        part = np.asarray(part, dtype=np.int64)

    imp, own, lev = halo_pairs(mesh, part, npes)
    rpn = a.ranks_per_node
    v = volumes(imp, own, lev, npes, rpn)
    tag = a.label or os.path.basename(a.mesh_dir)
    print(f"=== placement  {tag}  npes={npes}  ranks/node={rpn}  "
          f"({npes // rpn + (npes % rpn > 0)} nodes)")
    print(f"  halo pairs (rank<-rank) {len(set(zip(imp.tolist(), own.tolist()))):,}")
    print(f"  total halo   2-D {v['tot2']:>10,}   3-D {v['tot3']:>12,}")
    print(f"  OFF-NODE     2-D {v['off2']:>10,} ({v['frac2']*100:5.1f} %)"
          f"   3-D {v['off3']:>12,} ({v['frac3']*100:5.1f} %)")
    print(f"  off-node 3-D per node: max {v['pernode_max']:,}  mean {v['pernode_mean']:,.0f}")

    if a.csv:
        import csv as _csv
        row = dict(arm=a.arm or tag, npes=npes, ranks_per_node=rpn, **v)
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"  csv row -> {a.csv}")

    if a.compare_relabel or a.emit_relabelled:
        new = relabel_greedy(imp, own, lev, npes, rpn)
        v2 = volumes(new[imp], new[own], lev, npes, rpn)
        print(f"  -- greedy locality relabelling (identical partition, permuted rank ids) --")
        print(f"  OFF-NODE     2-D {v2['off2']:>10,} ({v2['frac2']*100:5.1f} %)"
              f"   3-D {v2['off3']:>12,} ({v2['frac3']*100:5.1f} %)")
        print(f"  off-node 3-D per node: max {v2['pernode_max']:,}  mean {v2['pernode_mean']:,.0f}")
        d3 = 100 * (v2['off3'] / max(v['off3'], 1) - 1)
        print(f"  => off-node 3-D volume {d3:+.1f} %   (negative = relabelling wins)")
        if a.emit_relabelled:
            np.savetxt(a.emit_relabelled, new[part], fmt="%d")
            print(f"  wrote relabelled part vector -> {a.emit_relabelled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
