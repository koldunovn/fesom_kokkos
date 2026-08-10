#!/usr/bin/env python3
"""M11: export the FESOM partitioning graph in the formats external engines read.

The graph is the one FESOM itself hands to METIS: `stiff_mat_ini` (fvom_init.F90:1608-1675)
declares all nodes of an element mutually adjacent, which for a triangular mesh is exactly the
mesh's edges. Reused from `m11_scorecard.Mesh.graph()` — a second implementation would be a
second thing to keep true (Task 4 diffs it against the partitioner's own CSR dump).

Formats
-------
METIS graph (KaMinPar, KaHIP, Mt-KaHyPar's graph mode, METIS itself):
    header:  nvtxs nedges [fmt [ncon]]      fmt = <has vsize><has vwgt><has adjwgt>
    line i:  [vsize] [vwgt_1..vwgt_ncon] (nbr weight)*      1-based neighbours
  Weight variants (--weights):
    none  fmt 000  — pure topology
    vwgt  fmt 010  — vwgt = a + nlev             (the single scalar weight the campaign wants)
    vsize fmt 100  — vsize = nlev                (METIS's comm-volume objective)
    both  fmt 110  — vwgt = a + nlev AND vsize = nlev, which METIS treats independently
    dual  fmt 010 ncon=2 — vwgt = (1, nlev+100), the legacy multi-constraint arm, for
                           reference only: it is the formulation ESA'26 shows crippling
                           refinement, and the one M10 measured as a net loss.
  --edge-weights adds adjwgt = nlev_i + nlev_j (fmt's last digit), reproducing what
  fort_part.c:191-205 hands METIS for weighted arms.

hMETIS hypergraph (Mt-KaHyPar `-o km1`):
    Star expansion — one net per vertex, net_v = {v} u N(v), net weight nlev(v). With that
    weighting km1 = sum over nets w*(lambda-1) equals METIS's total communication volume with
    vsize = nlev exactly, so the engine optimises the quantity we actually pay for.

usage:
  m11_graph_export.py <mesh_dir> -o core2.graph [--weights both] [--wgt-a 100] [--edge-weights]
  m11_graph_export.py <mesh_dir> -o core2.hgr --format hmetis [--wgt-a 100]
  m11_graph_export.py <mesh_dir> -o core2.graph --check [--check-part dist_8|file]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_scorecard import Mesh
from m7_part_spread import load_ranks


def csr(mesh):
    """Sorted adjacency CSR of the undirected graph (both directions present)."""
    gi, gj = mesh.graph()
    src = np.concatenate([gi, gj])
    dst = np.concatenate([gj, gi])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    indptr = np.zeros(mesh.nod2D + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dst


def write_metis(mesh, path, weights="none", wgt_a=0, edge_weights=False):
    indptr, adj = csr(mesh)
    nl = mesh.nlev_nod
    ncon = 1
    has_vsize = weights in ("vsize", "both")
    has_vwgt = weights in ("vwgt", "both", "dual")
    if weights == "dual":
        ncon = 2
    fmt = f"{int(has_vsize)}{int(has_vwgt)}{int(edge_weights)}"

    nbr = (adj + 1).astype(np.int64)
    ew = (nl[adj] + np.repeat(nl, np.diff(indptr))).astype(np.int64) if edge_weights else None

    lines = [f"{mesh.nod2D} {mesh.graph()[0].size} {fmt}" + (f" {ncon}" if ncon > 1 else "")]
    nbr_s = nbr.astype("U9")
    ew_s = ew.astype("U9") if ew is not None else None
    for v in range(mesh.nod2D):
        s, e = indptr[v], indptr[v + 1]
        head = []
        if has_vsize:
            head.append(str(int(nl[v])))
        if weights == "dual":
            head += ["1", str(int(nl[v]) + 100)]
        elif has_vwgt:
            head.append(str(int(wgt_a + nl[v])))
        if ew_s is None:
            body = " ".join(nbr_s[s:e])
        else:
            inter = np.empty((e - s) * 2, dtype="U9")
            inter[0::2] = nbr_s[s:e]
            inter[1::2] = ew_s[s:e]
            body = " ".join(inter)
        lines.append(" ".join(head + [body]) if head else body)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}: {mesh.nod2D:,} vertices, {mesh.graph()[0].size:,} edges, "
          f"fmt {fmt}, ncon {ncon}, weights={weights}"
          + (f", vwgt = {wgt_a} + nlev" if has_vwgt and weights != 'dual' else "")
          + (", adjwgt = nlev_i+nlev_j" if edge_weights else ""))
    return path


def write_hmetis(mesh, path, wgt_a=0):
    """Star expansion: net_v = {v} u N(v), net weight nlev(v); vertex weight a + nlev."""
    indptr, adj = csr(mesh)
    nl = mesh.nlev_nod
    lines = [f"{mesh.nod2D} {mesh.nod2D} 11"]          # weighted nets AND weighted vertices
    adj_s = (adj + 1).astype("U9")
    for v in range(mesh.nod2D):
        s, e = indptr[v], indptr[v + 1]
        lines.append(f"{int(nl[v])} {v + 1} " + " ".join(adj_s[s:e]))
    lines += [str(int(wgt_a + w)) for w in nl]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    pins = int(mesh.nod2D + 2 * mesh.graph()[0].size)
    print(f"wrote {path}: {mesh.nod2D:,} nets, {mesh.nod2D:,} vertices, {pins:,} pins, "
          f"net weight = nlev, vertex weight = {wgt_a} + nlev")
    return path


# ------------------------------------------------------------------ verification

def read_metis(path):
    """Re-read what we wrote (or any METIS graph) -> (n, m, fmt, ncon, vsize, vwgt, adj, ew)."""
    with open(path) as f:
        head = f.readline().split()
        n, m = int(head[0]), int(head[1])
        fmt = head[2] if len(head) > 2 else "000"
        ncon = int(head[3]) if len(head) > 3 else 1
        has_vsize, has_vwgt, has_ew = (c == "1" for c in f"{int(fmt):03d}")
        vsize = np.zeros(n, np.int64)
        vwgt = np.zeros((n, ncon), np.int64)
        adj, ew, ptr = [], [], [0]
        for i in range(n):
            t = np.fromstring(f.readline(), dtype=np.int64, sep=" ")
            k = 0
            if has_vsize:
                vsize[i] = t[0]; k = 1
            if has_vwgt:
                vwgt[i] = t[k:k + ncon]; k += ncon
            rest = t[k:]
            if has_ew:
                adj.append(rest[0::2]); ew.append(rest[1::2])
            else:
                adj.append(rest)
            ptr.append(ptr[-1] + adj[-1].size)
    return dict(n=n, m=m, fmt=fmt, ncon=ncon, vsize=vsize, vwgt=vwgt,
                indptr=np.array(ptr), adj=np.concatenate(adj) - 1,
                ew=np.concatenate(ew) if ew else None)


def check(path, mesh, part=None):
    g = read_metis(path)
    fails = []
    print(f"\n--- checking {path}")
    print(f"  header says n={g['n']:,} m={g['m']:,} fmt={g['fmt']} ncon={g['ncon']}")

    if g["n"] != mesh.nod2D:
        fails.append(f"n {g['n']} != nod2D {mesh.nod2D}")
    deg = np.diff(g["indptr"])
    if deg.sum() != 2 * g["m"]:
        fails.append(f"sum(degree)={deg.sum():,} != 2*m={2*g['m']:,}")
    else:
        print(f"  degree sum          {deg.sum():,} = 2 x {g['m']:,} edges  OK")

    # symmetry: the multiset of (i,j) must equal the multiset of (j,i)
    src = np.repeat(np.arange(g["n"]), deg)
    dst = g["adj"]
    a = np.sort(src.astype(np.int64) * g["n"] + dst)
    b = np.sort(dst.astype(np.int64) * g["n"] + src)
    if not np.array_equal(a, b):
        fails.append("adjacency is NOT symmetric")
    else:
        print(f"  symmetry            {src.size:,} directed entries, exact  OK")
    if np.any(src == dst):
        fails.append("self-loops present (METIS rejects them)")

    # identical to the in-memory graph the scorecard scores
    gi, gj = mesh.graph()
    ref = np.sort(np.concatenate([gi * g["n"] + gj, gj * g["n"] + gi]))
    if not np.array_equal(a, ref):
        fails.append("exported adjacency differs from Mesh.graph()")
    else:
        print("  vs Mesh.graph()     identical edge multiset  OK")

    if g["ew"] is not None:
        want = (mesh.nlev_nod[src] + mesh.nlev_nod[dst]).astype(np.int64)
        if not np.array_equal(g["ew"], want):
            fails.append("adjwgt != nlev_i + nlev_j")
        else:
            print(f"  adjwgt              = nlev_i+nlev_j on all {g['ew'].size:,} entries  OK")
    if g["fmt"][0] == "1" and not np.array_equal(g["vsize"], mesh.nlev_nod):
        fails.append("vsize != nlev")
    elif g["fmt"][0] == "1":
        print("  vsize               = nlev  OK")
    if g["fmt"][1] == "1":
        if g["ncon"] == 2:
            ok = (np.all(g["vwgt"][:, 0] == 1)
                  and np.array_equal(g["vwgt"][:, 1], mesh.nlev_nod + 100))
            print("  vwgt (dual)         = (1, nlev+100)  " + ("OK" if ok else "MISMATCH"))
            if not ok:
                fails.append("dual vwgt wrong")
        else:
            a0 = int(g["vwgt"][0, 0] - mesh.nlev_nod[0])
            if not np.array_equal(g["vwgt"][:, 0], mesh.nlev_nod + a0):
                fails.append("vwgt is not a + nlev for any constant a")
            else:
                print(f"  vwgt                = {a0} + nlev  OK  (sum {int(g['vwgt'].sum()):,}, "
                      f"int32 ledger {'OK' if g['vwgt'].sum() < 2**30 else 'EXCEEDED'})")

    if part is not None:
        cut = int(np.count_nonzero(part[gi] != part[gj]))
        cutw = int((mesh.nlev_nod[gi] + mesh.nlev_nod[gj])[part[gi] != part[gj]].sum())
        # the same quantity computed from the FILE, one direction only
        cut_f = int(np.count_nonzero(part[src] != part[dst]) // 2)
        print(f"  cut of given part   from file {cut_f:,} | from Mesh.graph() {cut:,} "
              f"| weighted {cutw:,}  " + ("OK" if cut_f == cut else "MISMATCH"))
        if cut_f != cut:
            fails.append("cut from file != cut from Mesh.graph()")

    if fails:
        for f_ in fails:
            print(f"  FAIL: {f_}")
        return 1
    print("  RESULT: PASS")
    return 0


def check_hmetis(path, mesh, part=None):
    """Star-expansion spot check: pin counts and net weights against the node patches."""
    print(f"\n--- checking {path}")
    with open(path) as f:
        nnets, nverts, fmt = f.readline().split()
        nnets, nverts = int(nnets), int(nverts)
        nets = [np.fromstring(f.readline(), dtype=np.int64, sep=" ") for _ in range(nnets)]
        vw = np.array([int(f.readline()) for _ in range(nverts)], dtype=np.int64)
    deg = mesh.degree()
    fails = []
    if nnets != mesh.nod2D or nverts != mesh.nod2D:
        fails.append(f"expected {mesh.nod2D} nets and vertices, got {nnets}/{nverts}")
    net_w = np.array([n[0] for n in nets], dtype=np.int64)
    size = np.array([n.size - 1 for n in nets], dtype=np.int64)
    if not np.array_equal(net_w, mesh.nlev_nod):
        fails.append("net weight != nlev")
    else:
        print("  net weights         = nlev on every net  OK")
    if not np.array_equal(size, deg + 1):
        bad = np.nonzero(size != deg + 1)[0][:5]
        fails.append(f"net size != degree+1 (first offenders {bad.tolist()})")
    else:
        print(f"  net sizes           = degree+1 on every net  OK (total pins {int(size.sum()):,} "
              f"= n + 2m = {mesh.nod2D + 2*mesh.graph()[0].size:,})")
    # spot-check three patches against the graph itself
    indptr, adj = csr(mesh)
    rng = np.random.default_rng(7)
    for v in rng.choice(mesh.nod2D, 3, replace=False):
        want = set(adj[indptr[v]:indptr[v + 1]].tolist()) | {int(v)}
        got = set((nets[v][1:] - 1).tolist())
        mark = "OK" if want == got else "MISMATCH"
        if want != got:
            fails.append(f"net {v} pins wrong")
        print(f"  patch spot-check    node {int(v)+1}: {len(got)} pins, "
              f"{{v}} u N(v) {mark}")
    if not np.array_equal(vw[:3], (vw[0] - mesh.nlev_nod[0]) + mesh.nlev_nod[:3]):
        fails.append("vertex weights are not a + nlev")
    else:
        print(f"  vertex weights      = {int(vw[0]-mesh.nlev_nod[0])} + nlev  OK")

    if part is not None:
        # the whole point of the star expansion: km1 of THIS file must equal METIS's
        # total communication volume with vsize = nlev, which is what we pay for.
        km1 = 0
        for v in range(nnets):
            km1 += int(net_w[v]) * (np.unique(part[nets[v][1:] - 1]).size - 1)
        from m11_scorecard import invariant_block
        vol = invariant_block(mesh, part, int(part.max()) + 1, None)["commvol_total"]
        mark = "OK" if km1 == vol else "MISMATCH"
        print(f"  km1 of this file    {km1:,} vs METIS totalv(vsize=nlev) {vol:,}  {mark}")
        if km1 != vol:
            fails.append("km1 != METIS total comm volume — the star expansion is wrong")
    if fails:
        for f_ in fails:
            print(f"  FAIL: {f_}")
        return 1
    print("  RESULT: PASS")
    return 0


def check_vs_dump(mesh, dump):
    """Diff our graph against the CSR the PARTITIONER handed METIS (FESOM_PART_GRAPH_DUMP).

    This is the authoritative check: everything else in the campaign assumes the Python
    graph is the Fortran graph. The dump also carries `nlevels_nod2D` as the partitioner
    RECOMPUTED it in memory, which settles review M5 — the partitioner keeps a pre-existing
    nlvls.out on disk while partitioning with freshly computed values, so the two can
    silently disagree.
    """
    print(f"\n--- {dump} vs Mesh.graph()")
    with open(dump) as f:
        head = f.readline().split()
        n, nnz = int(head[0]), int(head[1])
        has_w = len(head) > 2 and head[2] == "1"
        v = np.fromstring(f.read(), dtype=np.int64, sep="\n")
    indptr = v[:n + 1] - 1                      # dump is 1-based Fortran CSR
    adj = v[n + 1:n + 1 + nnz] - 1
    wgt = v[n + 1 + nnz:] if has_w else None
    fails = []
    print(f"  dump: n={n:,} nnz={nnz:,} weights={'yes' if has_w else 'no'}")
    if n != mesh.nod2D:
        fails.append(f"n {n} != nod2D {mesh.nod2D}")

    gi, gj = mesh.graph()
    ours_ptr, ours_adj = csr(mesh)
    if not np.array_equal(np.diff(indptr), np.diff(ours_ptr)):
        d = np.nonzero(np.diff(indptr) != np.diff(ours_ptr))[0]
        fails.append(f"degree differs at {d.size} vertices, first {d[:5].tolist()}")
    else:
        print(f"  rowptr              identical at all {n + 1:,} entries  OK")
    # colind order is insertion order in stiff_mat_ini, ours is sorted: compare per row
    same = True
    for k in range(n):
        a = np.sort(adj[indptr[k]:indptr[k + 1]])
        b = ours_adj[ours_ptr[k]:ours_ptr[k + 1]]
        if a.size != b.size or not np.array_equal(a, b):
            same = False
            fails.append(f"adjacency of vertex {k + 1} differs: dump {a.tolist()[:8]} "
                         f"vs ours {b.tolist()[:8]}")
            break
    if same:
        print(f"  colind (per row, sorted)  identical at all {nnz:,} entries  OK")

    if wgt is not None:
        if wgt.size != n:
            fails.append(f"weight array has {wgt.size} entries, expected {n}")
        elif np.array_equal(wgt, mesh.nlev_nod):
            print(f"  in-memory nlevels_nod2D == on-disk nlvls.out at all {n:,} nodes  OK "
                  "(review M5: no divergence on this mesh)")
        else:
            d = np.nonzero(wgt != mesh.nlev_nod)[0]
            fails.append(f"REVIEW M5 DIVERGENCE: in-memory nlevels differs from nlvls.out at "
                         f"{d.size} nodes, e.g. node {d[0]+1}: memory {wgt[d[0]]} vs file "
                         f"{mesh.nlev_nod[d[0]]}")
    if fails:
        for f_ in fails:
            print(f"  FAIL: {f_}")
        return 1
    print("  RESULT: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--vs-dump", metavar="CSR",
                    help="diff Mesh.graph() against a FESOM_PART_GRAPH_DUMP file and exit")
    ap.add_argument("--format", choices=["metis", "hmetis"], default="metis")
    ap.add_argument("--weights", choices=["none", "vwgt", "vsize", "both", "dual"],
                    default="none")
    ap.add_argument("--wgt-a", type=int, default=0, help="a in w = a + nlev")
    ap.add_argument("--edge-weights", action="store_true",
                    help="adjwgt = nlev_i + nlev_j (what fort_part.c does for weighted arms)")
    ap.add_argument("--check", action="store_true", help="re-read and verify the written file")
    ap.add_argument("--check-part", help="dist_N name or part-vector file for the cut check")
    a = ap.parse_args()

    mesh = Mesh(a.mesh_dir)
    if a.vs_dump:
        return check_vs_dump(mesh, a.vs_dump)
    if not a.out:
        ap.error("-o/--out is required unless --vs-dump is given")
    if a.format == "metis":
        write_metis(mesh, a.out, a.weights, a.wgt_a, a.edge_weights)
    else:
        write_hmetis(mesh, a.out, a.wgt_a)
    if not a.check:
        return 0
    part = None
    if a.check_part:
        if a.check_part.startswith("dist_"):
            part, _ = load_ranks(a.mesh_dir, int(a.check_part.split("_")[1]))
        else:
            part = np.loadtxt(a.check_part, dtype=np.int64).reshape(-1)
        part = np.asarray(part, dtype=np.int64)
    return check_hmetis(a.out, mesh, part) if a.format == "hmetis" else check(a.out, mesh, part)


if __name__ == "__main__":
    sys.exit(main())
