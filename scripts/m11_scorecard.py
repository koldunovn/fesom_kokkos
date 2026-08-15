#!/usr/bin/env python3
"""M11 partition scorecard — score a decomposition offline, before it costs cluster time.

The campaign's architecture is "score offline, then race": a large zoo of candidate
partitions is generated cheaply, this scorecard prunes it, and only the shortlist buys
node-hours. Everything here is computed from mesh files + a partition vector, with no
model run involved.

Two blocks of metrics, deliberately separated:

  * PERMUTATION-INVARIANT block — properties of the DECOMPOSITION. Renumbering the mesh
    and carrying the same partition through the permutation must leave every one of these
    bit-identical (`--permute-test` proves it). This is the block that ranks candidates.
  * ORDERING-SENSITIVE block — properties of the NUMBERING (index locality). Exempt from
    the invariance gate by construction: it is what Task 9's renumbering arms move.

Correctness gates (the Python half of the halo/dist gate; the model half is
`fesom_halo_identity_test`):
  * owned-set exact disjoint cover: the per-rank owned lists tile the mesh exactly once
    and agree with rpart.out;
  * com_info reciprocity: the halo gids rank r expects from rank q are, in order, exactly
    the gids q sends to r.

Units warning that this tool exists to settle (plan review B1, research digest §0):
`fort_part.c:11` defines USE_EDGE_WEIGHTS and lines 191-205 fill adjwgt ONLY when
wgt_type != 0, so METIS's "edgecut" print is an UNWEIGHTED CUT COUNT for 2D-only arms but
an nlev-WEIGHTED CUT SUM for weighted arms. M10's "edgecut ×91" compares the two. This
scorecard reports both quantities for every arm, always named in full.

usage:
  m11_scorecard.py <mesh_dir> --dist 8 [--dist 16 ...] [--arm NAME] [--csv out.csv]
  m11_scorecard.py <mesh_dir> --part-file p.txt --arm kaminpar_512 [--csv out.csv]
  m11_scorecard.py <mesh_dir> --dist 8 --permute-test
  m11_scorecard.py --regression            # reproduce the published M10 targets
  m11_scorecard.py --compare out.csv [--cols a,b,c]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_part_spread import load_ranks          # verified rpart parser — never re-implement

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    HAVE_SCIPY = True
except ImportError:                            # components metrics degrade, rest survives
    HAVE_SCIPY = False


# --------------------------------------------------------------------------- mesh

class Mesh:
    """The five gate files, plus edges when present."""

    def __init__(self, path, need_edges=True):
        self.path = path
        with open(f"{path}/nod2d.out") as f:
            self.nod2D = int(f.readline())
            nod = np.loadtxt(f, dtype=np.float64)
        assert nod.shape[0] == self.nod2D, (nod.shape, self.nod2D)
        self.lon, self.lat = nod[:, 1], nod[:, 2]
        self.coast = nod[:, 3].astype(np.int32)

        with open(f"{path}/elem2d.out") as f:
            self.elem2D = int(f.readline())
            elem = np.loadtxt(f, dtype=np.int64)
        assert elem.shape[0] == self.elem2D, (elem.shape, self.elem2D)
        self.elem = elem - 1                                    # 0-based (E, 3 or 4)

        self.nlev_nod = np.loadtxt(f"{path}/nlvls.out", dtype=np.int64)
        self.nlev_elem = np.loadtxt(f"{path}/elvls.out", dtype=np.int64)
        assert self.nlev_nod.size == self.nod2D
        assert self.nlev_elem.size == self.elem2D

        self.edges = None
        if need_edges and os.path.exists(f"{path}/edges.out"):
            self.edges = np.loadtxt(f"{path}/edges.out", dtype=np.int64) - 1
        self.edge2D = 0 if self.edges is None else self.edges.shape[0]

        self._graph = None

    # -- the METIS graph: exactly stiff_mat_ini (fvom_init.F90:1608-1675) --------
    # "all nodes in an element are adjacent" — for a triangle that is its 3 edges,
    # for a quad it additionally includes the diagonal (no edge, but METIS should
    # see the indirect coupling). Simple undirected graph, no self-loops.
    def graph(self):
        if self._graph is not None:
            return self._graph
        e = self.elem
        k = e.shape[1]
        cols = list(range(k))
        if k == 4:
            # stiff_mat_ini drops the duplicated vertex of a degenerate quad (triangle
            # stored in a hybrid mesh: nod(1) == nod(4)).
            tri = e[:, 0] == e[:, 3]
            pairs = []
            for a in range(k):
                for b in range(a + 1, k):
                    p = e[:, [a, b]]
                    if b == k - 1:
                        p = p[~tri]
                    pairs.append(p)
        else:
            pairs = [e[:, [a, b]] for a in cols for b in cols if a < b]
        p = np.vstack(pairs)
        i = np.minimum(p[:, 0], p[:, 1])
        j = np.maximum(p[:, 0], p[:, 1])
        keep = i != j
        i, j = i[keep], j[keep]
        key = np.unique(i.astype(np.int64) * self.nod2D + j)
        gi = (key // self.nod2D).astype(np.int64)
        gj = (key % self.nod2D).astype(np.int64)
        self._graph = (gi, gj)                                   # each edge ONCE, i < j
        return self._graph

    def degree(self):
        gi, gj = self.graph()
        d = np.zeros(self.nod2D, dtype=np.int64)
        np.add.at(d, gi, 1)
        np.add.at(d, gj, 1)
        return d


# --------------------------------------------------------------------------- dist

def read_my_list(dist_dir, rank):
    """-> dict of the per-rank dimensions and lists (mirrors fesom_partit.cpp:111-143)."""
    with open(f"{dist_dir}/my_list{rank:05d}.out") as f:
        v = np.fromstring(f.read(), dtype=np.int64, sep=" ")
    assert v[0] == rank, (v[0], rank)
    o = 1
    myd_n, ed_n = int(v[o]), int(v[o + 1]); o += 2
    nodes = v[o:o + myd_n + ed_n]; o += myd_n + ed_n
    myd_e, ed_e, exd_e = int(v[o]), int(v[o + 1]), int(v[o + 2]); o += 3
    o += myd_e + ed_e + exd_e
    myd_g, ed_g = int(v[o]), int(v[o + 1]); o += 2
    o += myd_g + ed_g
    assert o == v.size, (o, v.size)
    return dict(myDim_nod2D=myd_n, eDim_nod2D=ed_n, nodes=nodes,
                myDim_elem2D=myd_e, eDim_elem2D=ed_e, eXDim_elem2D=exd_e,
                myDim_edge2D=myd_g, eDim_edge2D=ed_g)


def read_com_info(dist_dir, rank, ed_n):
    """-> nod2D com block (rPE, rptr, rlist, sPE, sptr, slist); fesom_partit.cpp:154-207."""
    with open(f"{dist_dir}/com_info{rank:05d}.out") as f:
        v = np.fromstring(f.read(), dtype=np.int64, sep=" ")
    assert v[0] == rank, (v[0], rank)
    o = 1
    rn = int(v[o]); o += 1
    rPE = v[o:o + rn]; o += rn
    rptr = v[o:o + rn + 1]; o += rn + 1
    n_r = int(rptr[-1] - rptr[0])
    assert n_r == ed_n, (n_r, ed_n)
    rlist = v[o:o + n_r]; o += n_r
    sn = int(v[o]); o += 1
    sPE = v[o:o + sn]; o += sn
    sptr = v[o:o + sn + 1]; o += sn + 1
    n_s = int(sptr[-1] - sptr[0])
    slist = v[o:o + n_s]; o += n_s
    return dict(rPE=rPE, rptr=rptr, rlist=rlist, sPE=sPE, sptr=sptr, slist=slist)


def load_dist_files(mesh_dir, npes):
    """Per-rank dims + com blocks for every rank. Returns None if the files are absent."""
    d = f"{mesh_dir}/dist_{npes}"
    if not os.path.exists(f"{d}/my_list00000.out"):
        return None
    ml, ci = [], []
    for r in range(npes):
        m = read_my_list(d, r)
        ml.append(m)
        ci.append(read_com_info(d, r, m["eDim_nod2D"]))
    return dict(dir=d, my_list=ml, com=ci)


# --------------------------------------------------------------------------- gates

def gate_owned_cover(mesh, part, dist):
    """Owned lists tile the mesh exactly once AND agree with rpart.out."""
    if dist is None:
        return "skipped(no dist files)"
    npes = len(dist["my_list"])
    seen = np.zeros(mesh.nod2D, dtype=np.int32)
    bad_rank = 0
    for r in range(npes):
        m = dist["my_list"][r]
        owned = m["nodes"][:m["myDim_nod2D"]] - 1
        seen[owned] += 1
        bad_rank += int(np.count_nonzero(part[owned] != r))
    missing = int(np.count_nonzero(seen == 0))
    dup = int(np.count_nonzero(seen > 1))
    if missing or dup or bad_rank:
        return f"FAIL(missing={missing},dup={dup},rpart_mismatch={bad_rank})"
    return "ok"


def gate_reciprocity(dist):
    """The gids r expects from q are, in order, exactly the gids q sends to r.

    This is the Python twin of fesom_halo_identity_test: it compares the CONTENT of the
    halo contract, not just its sizes, so a count-preserving corruption (two swapped
    rlist entries) is caught here as well as in the model.
    """
    if dist is None:
        return "skipped(no dist files)"
    npes = len(dist["my_list"])
    gid = [dist["my_list"][r]["nodes"] for r in range(npes)]
    bad = 0
    detail = ""
    for r in range(npes):
        c = dist["com"][r]
        for k, q in enumerate(c["rPE"]):
            lo, hi = int(c["rptr"][k]) - int(c["rptr"][0]), int(c["rptr"][k + 1]) - int(c["rptr"][0])
            recv_gid = gid[r][c["rlist"][lo:hi] - 1]
            cq = dist["com"][int(q)]
            w = np.nonzero(cq["sPE"] == r)[0]
            if w.size != 1:
                bad += 1
                detail = detail or f"rank {q} has no send block for {r}"
                continue
            k2 = int(w[0])
            lo2 = int(cq["sptr"][k2]) - int(cq["sptr"][0])
            hi2 = int(cq["sptr"][k2 + 1]) - int(cq["sptr"][0])
            send_gid = gid[int(q)][cq["slist"][lo2:hi2] - 1]
            if recv_gid.size != send_gid.size or not np.array_equal(recv_gid, send_gid):
                bad += 1
                if not detail:
                    n = min(recv_gid.size, send_gid.size)
                    w2 = np.nonzero(recv_gid[:n] != send_gid[:n])[0]
                    detail = (f"rank {r}<-{q}: {w2.size} of {n} gids differ, "
                              f"first at slot {int(w2[0]) if w2.size else 'n/a'}")
    if bad:
        return f"FAIL({bad} blocks; {detail})"
    return "ok"


# --------------------------------------------------------------------- the metrics

def invariant_block(mesh, part, npes, dist, wgt_a=(0, 100)):
    """Everything that depends only on WHICH nodes go WHERE, never on their numbering."""
    out = {}
    gi, gj = mesh.graph()
    pi, pj = part[gi], part[gj]
    cut = pi != pj

    # -- balance -----------------------------------------------------------------
    cnt2 = np.bincount(part, minlength=npes).astype(np.float64)
    s3 = np.bincount(part, weights=mesh.nlev_nod.astype(np.float64), minlength=npes)
    out["n2d_max"] = int(cnt2.max())
    out["n2d_imb"] = cnt2.max() / cnt2.mean()
    out["n3d_max"] = int(s3.max())
    out["n3d_imb"] = s3.max() / s3.mean()
    out["n3d_maxmin"] = s3.max() / max(s3.min(), 1.0)
    for a in wgt_a:
        w = (a + mesh.nlev_nod).astype(np.float64)
        sw = np.bincount(part, weights=w, minlength=npes)
        out[f"w{a}_imb"] = sw.max() / sw.mean()
        out[f"w{a}_sum"] = float(w.sum())          # int32 ledger: keep Sum(w) <= 2^30

    # -- cut, in BOTH units ------------------------------------------------------
    out["edgecut_unweighted"] = int(cut.sum())
    ew = (mesh.nlev_nod[gi] + mesh.nlev_nod[gj]).astype(np.int64)   # fort_part.c:199-201
    out["cutweight_nlev"] = int(ew[cut].sum())
    out["edges_total"] = int(gi.size)
    out["mean_edge_weight"] = float(ew.mean())

    # -- METIS communication volume with vsize = nlev ----------------------------
    # totalv = sum_v vsize(v) * (lambda(v) - 1), lambda = #distinct parts in {v} u N(v)
    lam = _lambda_per_vertex(mesh, part)
    vol = mesh.nlev_nod * (lam - 1)
    out["commvol_total"] = int(vol.sum())
    per_part = np.bincount(part, weights=vol.astype(np.float64), minlength=npes)
    out["commvol_max_rank"] = int(per_part.max())
    out["commvol_imb"] = per_part.max() / max(per_part.mean(), 1.0)
    out["boundary_nodes"] = int(np.count_nonzero(lam > 1))

    # -- neighbour count per part ------------------------------------------------
    if cut.any():
        a = np.minimum(pi[cut], pj[cut]).astype(np.int64)
        b = np.maximum(pi[cut], pj[cut]).astype(np.int64)
        pk = np.unique(a * npes + b)
        na, nb = pk // npes, pk % npes
        deg = np.bincount(na, minlength=npes) + np.bincount(nb, minlength=npes)
    else:
        deg = np.zeros(npes, dtype=np.int64)
    out["nbr_max"] = int(deg.max())
    out["nbr_mean"] = float(deg.mean())

    # -- contiguity --------------------------------------------------------------
    out.update(_components(mesh, part, npes))
    out["isolated_nodes"] = int(np.count_nonzero(same_part_degree(mesh, part) <= 1))

    # -- halo / replication, from the dist files (authoritative) -----------------
    if dist is not None:
        ml = dist["my_list"]
        halo = np.array([m["eDim_nod2D"] for m in ml], dtype=np.float64)
        own = np.array([m["myDim_nod2D"] for m in ml], dtype=np.float64)
        out["halo_nod_mean"] = float(halo.mean())
        out["halo_nod_max"] = int(halo.max())
        out["halo_frac_mean"] = float((halo / own).mean())
        el = np.array([m["myDim_elem2D"] + m["eDim_elem2D"] + m["eXDim_elem2D"] for m in ml])
        ed = np.array([m["myDim_edge2D"] + m["eDim_edge2D"] for m in ml])
        out["elem_repl"] = float(el.sum() / mesh.elem2D)
        out["edge_repl"] = float(ed.sum() / mesh.edge2D) if mesh.edge2D else float("nan")
    else:
        for k in ("halo_nod_mean", "halo_nod_max", "halo_frac_mean", "elem_repl", "edge_repl"):
            out[k] = float("nan")
    return out


def same_part_degree(mesh, part):
    """#neighbours of v that share v's partition.

    `check_partitioning` (fvom_init.F90:1868) moves every node with <= 1 of these into
    an adjacent partition, AFTER METIS has printed its edgecut. A nonzero count on a
    finished dist therefore means the post-pass did not converge — and any injected
    partition (Task 4/6/8) should be checked for it.
    """
    gi, gj = mesh.graph()
    keep = part[gi] == part[gj]
    d = np.zeros(mesh.nod2D, dtype=np.int64)
    np.add.at(d, gi[keep], 1)
    np.add.at(d, gj[keep], 1)
    return d


def find_checkpart_moves(mesh, part, delta, weighted=True):
    """Which single `check_partitioning`-style move explains a cut discrepancy of `delta`?

    Reverse-engineers the post-pass: looks for a node v whose move BACK to an adjacent
    part p changes the cut by -delta, and which had <= 1 neighbour in p beforehand (the
    isolation criterion that made the post-pass pick it up). Used to explain a mismatch
    between the partitioner's printed edgecut and the scorecard's reading of the file.
    """
    gi, gj = mesh.graph()
    n = mesh.nod2D
    A = coo_matrix((np.ones(gi.size, np.int8), (gi, gj)), shape=(n, n))
    A = (A + A.T).tocsr()
    nl = mesh.nlev_nod
    hits = []
    for v in range(n):
        s, e = A.indptr[v], A.indptr[v + 1]
        nb = A.indices[s:e]
        if nb.size == 0:
            continue
        pb = part[nb]
        w = (nl[v] + nl[nb]).astype(np.int64) if weighted else np.ones(nb.size, np.int64)
        cur = part[v]
        s_cur = w[pb == cur].sum()
        for p in np.unique(pb[pb != cur]):
            inp = pb == p
            if inp.sum() > 1:                 # pre-move isolation criterion
                continue
            if int(w[inp].sum() - s_cur) == -delta:
                hits.append(dict(node0=int(v), gid=int(v) + 1, now_in=int(cur), was_in=int(p),
                                 nb=nb.tolist(), nb_parts=pb.tolist(),
                                 nlev=int(nl[v]), nlev_nb=nl[nb].tolist()))
    return hits


def _lambda_per_vertex(mesh, part):
    """#distinct partitions among {v} u N(v), vectorised."""
    gi, gj = mesh.graph()
    npes = int(part.max()) + 1
    # (vertex, part) pairs from both endpoint directions plus the vertex's own part
    v = np.concatenate([gi, gj, np.arange(mesh.nod2D)])
    p = np.concatenate([part[gj], part[gi], part])
    key = np.unique(v.astype(np.int64) * npes + p.astype(np.int64))
    vv = key // npes
    return np.bincount(vv, minlength=mesh.nod2D)


def _components(mesh, part, npes):
    """Connected components of the whole wet graph and of each part's induced subgraph."""
    out = {}
    gi, gj = mesh.graph()
    if not HAVE_SCIPY:
        for k in ("wet_components", "parts_disconnected", "components_total",
                  "noncore_vertices", "singleton_vertices", "components_max"):
            out[k] = -1
        return out
    n = mesh.nod2D
    full = coo_matrix((np.ones(gi.size, np.int8), (gi, gj)), shape=(n, n))
    out["wet_components"] = int(connected_components(full, directed=False)[0])

    keep = part[gi] == part[gj]
    sub = coo_matrix((np.ones(int(keep.sum()), np.int8), (gi[keep], gj[keep])), shape=(n, n))
    ncomp, lab = connected_components(sub, directed=False)
    # a vertex isolated inside its own part still counts as one component, which is
    # exactly the "stray vertex" defect we want to see
    key = part.astype(np.int64) * ncomp + lab
    uk, sizes = np.unique(key, return_counts=True)
    up = uk // ncomp
    per_part = np.bincount(up, minlength=npes)
    out["components_total"] = int(uk.size)
    out["parts_disconnected"] = int(np.count_nonzero(per_part > 1))
    out["components_max"] = int(per_part.max())
    # two different readings of "stray", kept apart on purpose: a part split into two
    # large lobes and a part trailing single loose vertices are different defects.
    order = np.lexsort((-sizes, up))
    up_s, sizes_s = up[order], sizes[order]
    first = np.ones(up_s.size, dtype=bool)
    first[1:] = up_s[1:] != up_s[:-1]
    out["noncore_vertices"] = int(sizes_s[~first].sum())     # outside their part's largest lobe
    out["singleton_vertices"] = int(sizes[sizes == 1].sum())  # alone in their part
    return out


def ordering_block(mesh):
    """Index-locality proxies. EXEMPT from the permutation-invariance gate by design."""
    out = {}
    gi, gj = mesh.graph()
    d = np.abs(gi - gj)
    out["ord_edge_didx_mean"] = float(d.mean())
    out["ord_edge_didx_p95"] = float(np.percentile(d, 95))
    e = mesh.elem
    out["ord_elem_span_mean"] = float((e.max(axis=1) - e.min(axis=1)).mean())
    out.update(_stride_hist(e.reshape(-1), "elem"))
    if mesh.edges is not None:
        out.update(_stride_hist(mesh.edges.reshape(-1), "edge"))
    else:
        for k in ("unit", "l64", "l4k", "g4k"):
            out[f"ord_stride_edge_{k}"] = float("nan")
    return out


def _stride_hist(stream, tag):
    """Fraction of consecutive gather addresses within a cache-line / page-ish distance."""
    s = np.abs(np.diff(stream.astype(np.int64)))
    tot = max(s.size, 1)
    return {f"ord_stride_{tag}_unit": float(np.count_nonzero(s <= 1) / tot),
            f"ord_stride_{tag}_l64": float(np.count_nonzero(s <= 64) / tot),
            f"ord_stride_{tag}_l4k": float(np.count_nonzero(s <= 4096) / tot),
            f"ord_stride_{tag}_g4k": float(np.count_nonzero(s > 4096) / tot)}


# --------------------------------------------------------------------------- driver

def score(mesh_dir, npes, arm, part=None, wgt_a=(0, 100), want_dist=True, mesh=None):
    mesh = mesh or Mesh(mesh_dir)
    dist = load_dist_files(mesh_dir, npes) if want_dist else None
    if part is None:
        part, _ = load_ranks(mesh_dir, npes)
    part = np.asarray(part, dtype=np.int64)
    assert part.size == mesh.nod2D, (part.size, mesh.nod2D)
    assert part.min() >= 0 and part.max() < npes, (part.min(), part.max(), npes)
    if np.bincount(part, minlength=npes).min() == 0:
        print(f"  WARNING: {arm}: at least one part is EMPTY", file=sys.stderr)

    row = dict(mesh=os.path.basename(os.path.normpath(mesh_dir)), arm=arm, npes=npes,
               nod2D=mesh.nod2D, elem2D=mesh.elem2D, edge2D=mesh.edge2D)
    row.update(invariant_block(mesh, part, npes, dist, wgt_a))
    row.update(ordering_block(mesh))
    row["gate_cover"] = gate_owned_cover(mesh, part, dist)
    row["gate_recip"] = gate_reciprocity(dist)
    return row


def print_row(row):
    print(f"\n=== {row['mesh']} / {row['arm']} / npes={row['npes']} "
          f"(nod2D={row['nod2D']} elem2D={row['elem2D']} edge2D={row['edge2D']}) ===")
    print("  -- invariant (ranks the candidate) --")
    print(f"  balance      2D max/mean {row['n2d_imb']:.4f} | "
          f"3D max/mean {row['n3d_imb']:.4f} (max/min {row['n3d_maxmin']:.2f}) | "
          f"w0 {row['w0_imb']:.4f} w100 {row['w100_imb']:.4f}")
    print(f"  cut          edgecut_unweighted {row['edgecut_unweighted']:,} "
          f"of {row['edges_total']:,} edges | "
          f"cutweight_nlev {row['cutweight_nlev']:,} "
          f"(mean edge weight {row['mean_edge_weight']:.1f})")
    print(f"  comm volume  total {row['commvol_total']:,} | max/rank {row['commvol_max_rank']:,} "
          f"(imb {row['commvol_imb']:.2f}) | boundary nodes {row['boundary_nodes']:,}")
    print(f"  neighbours   max {row['nbr_max']} mean {row['nbr_mean']:.2f}")
    print(f"  contiguity   wet graph components {row['wet_components']} | "
          f"parts disconnected {row['parts_disconnected']} | "
          f"components {row['components_total']} (max/part {row['components_max']}) | "
          f"outside main lobe {row['noncore_vertices']} | singletons {row['singleton_vertices']} | "
          f"isolated (<=1 same-part nb) {row['isolated_nodes']}")
    print(f"  halo/repl    halo nodes mean {row['halo_nod_mean']:.1f} max {row['halo_nod_max']} "
          f"| halo/owned {row['halo_frac_mean']:.4f} | elem repl {row['elem_repl']:.4f} "
          f"| edge repl {row['edge_repl']:.4f}")
    print(f"  gates        owned cover: {row['gate_cover']} | com_info reciprocity: {row['gate_recip']}")
    print("  -- ordering (mesh numbering only; exempt from the invariance gate) --")
    print(f"  locality     |di| edge mean {row['ord_edge_didx_mean']:.0f} p95 {row['ord_edge_didx_p95']:.0f} "
          f"| elem span mean {row['ord_elem_span_mean']:.0f}")
    print(f"  strides      elem <=64 {row['ord_stride_elem_l64']:.3f} >4k {row['ord_stride_elem_g4k']:.3f} "
          f"| edge <=64 {row['ord_stride_edge_l64']:.3f} >4k {row['ord_stride_edge_g4k']:.3f}")


INVARIANT_KEYS = ["n2d_max", "n2d_imb", "n3d_max", "n3d_imb", "n3d_maxmin",
                  "w0_imb", "w0_sum", "w100_imb", "w100_sum",
                  "edgecut_unweighted", "cutweight_nlev", "edges_total", "mean_edge_weight",
                  "commvol_total", "commvol_max_rank", "commvol_imb", "boundary_nodes",
                  "nbr_max", "nbr_mean", "wet_components", "parts_disconnected",
                  "components_total", "components_max", "noncore_vertices",
                  "singleton_vertices", "isolated_nodes"]


def write_csv(path, rows):
    keys = list(rows[0].keys())
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\ncsv: {len(rows)} row(s) -> {path}")


# ------------------------------------------------------------------ permute test

def permute_test(mesh_dir, npes, seed=12345):
    """Relabel the mesh at random, carry the SAME partition through, demand equality.

    This is the property Task 9's renumbering arms depend on: if the invariant block
    moved under a pure relabelling, every ordering A/B would be confounded.
    """
    mesh = Mesh(mesh_dir)
    part, _ = load_ranks(mesh_dir, npes)
    base = invariant_block(mesh, np.asarray(part, np.int64), npes, None)
    base_ord = ordering_block(mesh)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(mesh.nod2D)               # old index -> new index
    inv = np.empty_like(perm)
    inv[perm] = np.arange(mesh.nod2D)

    pm = Mesh.__new__(Mesh)
    pm.path = mesh_dir + " [permuted]"
    pm.nod2D, pm.elem2D = mesh.nod2D, mesh.elem2D
    pm.lon, pm.lat, pm.coast = mesh.lon[inv], mesh.lat[inv], mesh.coast[inv]
    pm.elem = perm[mesh.elem]
    pm.nlev_nod = mesh.nlev_nod[inv]
    pm.nlev_elem = mesh.nlev_elem
    pm.edges = perm[mesh.edges] if mesh.edges is not None else None
    pm.edge2D = mesh.edge2D
    pm._graph = None
    new = invariant_block(pm, np.asarray(part, np.int64)[inv], npes, None)
    new_ord = ordering_block(pm)

    bad = [k for k in INVARIANT_KEYS if not _close(base.get(k), new.get(k))]
    print(f"permutation test on {mesh_dir} dist_{npes} (seed {seed}):")
    print(f"  invariant keys checked: {len(INVARIANT_KEYS)}")
    if bad:
        for k in bad:
            print(f"  MISMATCH {k}: {base.get(k)} -> {new.get(k)}")
        print("  RESULT: FAIL")
        return 1
    print("  RESULT: PASS (all invariant metrics identical under relabelling)")
    moved = [k for k in base_ord if not _close(base_ord[k], new_ord[k])]
    print(f"  ordering block moved on {len(moved)}/{len(base_ord)} keys, as it must: "
          f"|di| edge mean {base_ord['ord_edge_didx_mean']:.0f} -> {new_ord['ord_edge_didx_mean']:.0f}")
    if not moved:
        print("  RESULT: FAIL — the ordering block did NOT move; it is not measuring order")
        return 1
    return 0


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if a != a and b != b:       # both nan
            return True
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    return a == b


# -------------------------------------------------------------------- regression

CORE2_WGT0 = "/work/ab0995/a270088/port2/mesh/core2_wgt0"
CORE2_WGT2 = "/work/ab0995/a270088/port2/mesh/core2_wgt2"
CORE2_SHIP = "/work/ab0995/a270088/port2/mesh/core2"
FARC_POOL = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc"

# (label, mesh, npes, key, expected, tol, source of the published number)
REGRESSION = [
    ("core2_wgt0 8r  METIS print", CORE2_WGT0, 8, "edgecut_unweighted", 1335, 0,
     "M10 L103 table (2D-only arm: adjwgt NULL => unweighted count)"),
    ("core2_wgt0 16r METIS print", CORE2_WGT0, 16, "edgecut_unweighted", 2549, 0, "M10 L103"),
    ("core2_wgt0 32r METIS print", CORE2_WGT0, 32, "edgecut_unweighted", 4307, 0, "M10 L103"),
    ("core2_wgt2 8r  METIS print", CORE2_WGT2, 8, "cutweight_nlev", 120883, 0,
     "M10 L103 table (dual arm: adjwgt = nlev_i+nlev_j => weighted sum)"),
    # The one target where the METIS print and the shipped file legitimately disagree:
    # check_partitioning runs AFTER the print and moved gid 125423 from part 9 to part 8
    # (degree 3, nlev 5; neighbour nlev 20/5/5), which lowers the UNWEIGHTED cut by 1 and
    # raises the nlev-weighted cut by exactly 5. The file is the truth; 217,791 is the
    # pre-post-pass print M10 quoted. Verified unique by find_checkpart_moves().
    ("core2_wgt2 16r on-disk", CORE2_WGT2, 16, "cutweight_nlev", 217796, 0,
     "M10 L103 prints 217,791 PRE-check_partitioning; +5 explained, see session log"),
    ("core2_wgt2 32r METIS print", CORE2_WGT2, 32, "cutweight_nlev", 375211, 0, "M10 L103"),
    ("core2 shipped 864r 3D imb", CORE2_SHIP, 864, "n3d_maxmin", 9.60, 0.05,
     "M10 phase-budget table (3D max/min)"),
    ("core2_wgt2 864r 3D imb", CORE2_WGT2, 864, "n3d_maxmin", 1.05, 0.02, "M10 phase budget"),
    ("core2_wgt0 864r halo/rank", CORE2_WGT0, 864, "halo_nod_mean", 42, 1.0,
     "M10 SSH_SOLVERS_M10.md:1820 (42 -> 59 nodes/rank)"),
    ("core2_wgt2 864r halo/rank", CORE2_WGT2, 864, "halo_nod_mean", 59, 1.0, "M10:1820"),
    ("farc /pool 2048r 2D imb", FARC_POOL, 2048, "n2d_imb", 1.01, 0.02, "M10 load-balance report"),
    ("farc /pool 2048r 3D imb", FARC_POOL, 2048, "n3d_maxmin", 9.40, 0.30, "M10 load-balance report"),
]


def regression():
    cache, fails = {}, 0
    print("M11 scorecard regression against the published M10 numbers")
    print("(the two cut rows are quoted in DIFFERENT units by METIS — see the header note)\n")
    rows = {}
    for label, mdir, npes, key, exp, tol, src in REGRESSION:
        ck = (mdir, npes)
        if ck not in rows:
            if mdir not in cache:
                cache[mdir] = Mesh(mdir)
            want_dist = any(k == "halo_nod_mean" or k.startswith(("halo", "elem_repl", "edge_repl"))
                            for l, m, n, k, e, t, s in REGRESSION if (m, n) == ck)
            rows[ck] = score(mdir, npes, arm=os.path.basename(mdir),
                             want_dist=want_dist, mesh=cache[mdir])
        got = rows[ck][key]
        ok = abs(float(got) - float(exp)) <= tol
        fails += 0 if ok else 1
        gs = f"{got:,.4f}" if isinstance(got, float) else f"{got:,}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:32s} {key:20s} "
              f"expected {exp:>10} got {gs:>12}   ({src})")

    print("\n--- the B1 units correction, measured ---")
    r0 = rows.get((CORE2_WGT0, 8)); r2 = rows.get((CORE2_WGT2, 8))
    for n in (8, 16, 32):
        a, b = rows.get((CORE2_WGT0, n)), rows.get((CORE2_WGT2, n))
        if a and b:
            print(f"  {n:3d}r  unweighted cut  wgt0 {a['edgecut_unweighted']:>7,} -> "
                  f"wgt2 {b['edgecut_unweighted']:>7,}  = x{b['edgecut_unweighted']/a['edgecut_unweighted']:.2f}"
                  f"   | nlev-weighted cut  wgt0 {a['cutweight_nlev']:>8,} -> wgt2 "
                  f"{b['cutweight_nlev']:>8,}  = x{b['cutweight_nlev']/a['cutweight_nlev']:.2f}"
                  f"   | M10 mixed-units ratio x{b['cutweight_nlev']/a['edgecut_unweighted']:.1f}")
    if r0 and r2:
        print(f"  mean edge weight (nlev_i+nlev_j) on CORE2: {r0['mean_edge_weight']:.1f}")
    print(f"\nregression: {len(REGRESSION) - fails}/{len(REGRESSION)} PASS")
    return 1 if fails else 0


# ------------------------------------------------------------------------- main

def compare(path, cols=None):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: no rows")
    keys = cols.split(",") if cols else [
        "n3d_imb", "n3d_maxmin", "edgecut_unweighted", "cutweight_nlev",
        "commvol_total", "commvol_max_rank", "nbr_max", "parts_disconnected",
        "noncore_vertices", "isolated_nodes", "halo_nod_mean", "elem_repl"]
    base = rows[0]
    w = max(len(f"{r['arm']}@{r['npes']}") for r in rows)
    print(f"{'arm':<{w}} " + " ".join(f"{k:>18}" for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            try:
                v, b = float(r[k]), float(base[k])
                d = "" if r is base or b == 0 else f" ({(v/b-1)*100:+.1f}%)"
                cells.append(f"{v:>10,.4g}{d:>8}")
            except (ValueError, KeyError):
                cells.append(f"{r.get(k, '?'):>18}")
        print(f"{r['arm']+'@'+r['npes']:<{w}} " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_dir", nargs="?")
    ap.add_argument("--dist", type=int, action="append", default=[],
                    help="score mesh_dir/dist_N (repeatable)")
    ap.add_argument("--part-file", help="raw part vector, one 0-based rank per node")
    ap.add_argument("--npes", type=int, help="part count for --part-file (default max+1)")
    ap.add_argument("--arm", default=None, help="label for the CSV row")
    ap.add_argument("--wgt-a", default="0,100", help="a values for w = a + nlev")
    ap.add_argument("--csv", help="append rows to this CSV")
    ap.add_argument("--no-dist-files", action="store_true",
                    help="skip my_list/com_info (no halo, replication or gates)")
    ap.add_argument("--permute-test", action="store_true")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--compare", help="print a CSV as a delta table")
    ap.add_argument("--cols", help="columns for --compare")
    a = ap.parse_args()

    if a.compare:
        return compare(a.compare, a.cols)
    if a.regression:
        return regression()
    if not a.mesh_dir:
        ap.error("mesh_dir is required (or use --regression / --compare)")
    wgt_a = tuple(int(x) for x in a.wgt_a.split(","))

    if a.permute_test:
        if not a.dist:
            ap.error("--permute-test needs --dist N")
        return permute_test(a.mesh_dir, a.dist[0])

    rows = []
    mesh = Mesh(a.mesh_dir)
    if a.part_file:
        part = np.loadtxt(a.part_file, dtype=np.int64).reshape(-1)
        npes = a.npes or int(part.max()) + 1
        rows.append(score(a.mesh_dir, npes, a.arm or os.path.basename(a.part_file),
                          part=part, wgt_a=wgt_a,
                          want_dist=not a.no_dist_files, mesh=mesh))
    for n in a.dist:
        rows.append(score(a.mesh_dir, n, a.arm or f"dist_{n}", wgt_a=wgt_a,
                          want_dist=not a.no_dist_files, mesh=mesh))
    if not rows:
        ap.error("nothing to score: give --dist and/or --part-file")
    for r in rows:
        print_row(r)
    if a.csv:
        write_csv(a.csv, rows)
    bad = [r for r in rows if r["gate_cover"].startswith("FAIL") or r["gate_recip"].startswith("FAIL")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
