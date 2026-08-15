#!/usr/bin/env python3
"""M12b wide-halo ring census — measure, per mesh and per partition, what a
K-ring wide halo would actually cost and whether the K=1 rung's assumptions hold.

For each partition (sampled ranks) it reports:
  * per-rank owned/halo sizes (nodes, elements) from the my_list files;
  * CONTAINMENT: are all elements incident to a ring-1 halo node present in the
    local element list?  (the K=1 rung needs Ubt valid there, and would use the
    existing com_elem2D_full list to fill them);
  * ENB TAIL: does an OWNED element ever have an edge-neighbour whose local
    index lands in the eXDim tail (>= myDim+eDim)?  fesom_ssh_se.cpp's viscosity
    reads Ubt[2*nb] with Ubt allocated over myDim+eDim only, so a nonzero count
    is an out-of-bounds read in the certified SE module;
  * RING GROWTH: node and element counts of rings 1..R and the cumulative ring
    zone as a fraction of the owned set — the redundant-compute cost of a K-ring
    exact wide halo (SE loses exactly one ring per substep, so K substeps need K
    rings).  Ring indices follow the BFS below: element ring r = elements first
    reached from node ring r-1, so printed ring 1 is "elements incident to owned
    nodes" (= myDim_elem2D) and printed ring 1 nodes are the eDim halo;
  * PARTNERS: rPEnum/sPEnum for com_nod2D, com_elem2D and com_elem2D_full.  The
    K=1 rung trades {eta over com_nod2D + Ubt over com_elem2D} for {Ubt over
    com_elem2D_full}, so messages/substep go (rPE_nod + rPE_elem) -> rPE_full.
    That ratio is the real message reduction — "halved" is a claim about
    PARTNERS, not about entities;
  * LOCAL-EDGE COVERAGE: what fraction of the edges incident to a ring-1 node is
    absent from the local edge list (myDim_edge2D + eDim_edge2D).  A nonzero
    fraction means a ring-1 div-CSR row cannot be assembled locally at any
    precision, so the owner's row must be shipped.

Usage:
  m12b_ring_census.py <mesh_dir> <dist_dir> [<dist_dir> ...]
                      [--rings R] [--sample N] [--no-enb]
"""
import sys, os, time
import numpy as np

argv = [a for a in sys.argv[1:]]
R      = 8
sample = 16
do_enb = True
out    = []
while '--rings' in argv:
    i = argv.index('--rings');  R = int(argv[i+1]);      del argv[i:i+2]
while '--sample' in argv:
    i = argv.index('--sample'); sample = int(argv[i+1]); del argv[i:i+2]
while '--no-enb' in argv:
    argv.remove('--no-enb'); do_enb = False
mesh_dir, dists = argv[0], argv[1:]


def log(*a):
    print(*a, flush=True)
    out.append(' '.join(str(x) for x in a))


def ragged(off, sel):
    """Indices into a CSR data array for all rows in `sel`."""
    starts = off[sel]
    counts = off[sel + 1] - starts
    total  = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    cum = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return np.repeat(starts - cum, counts) + np.arange(total, dtype=np.int64)


t0 = time.time()
elem = np.fromfile(os.path.join(mesh_dir, 'elem2d.out'), sep=' ', dtype=np.int64)
E = int(elem[0])
elem = (elem[1:1 + 3 * E] - 1).reshape(E, 3)
N = int(elem.max()) + 1
log(f"# mesh {mesh_dir}: {N} nodes, {E} elements  ({time.time()-t0:.0f}s)")

# node -> incident elements (CSR)
flat_n = elem.ravel()
flat_e = np.repeat(np.arange(E, dtype=np.int64), 3)
order  = np.argsort(flat_n, kind='stable')
n2e    = flat_e[order]
n_off  = np.zeros(N + 1, dtype=np.int64)
np.cumsum(np.bincount(flat_n, minlength=N), out=n_off[1:])

# element -> edge-neighbour elements (CSR), via sorted undirected edge keys
if do_enb:
    a, b, c = elem[:, 0], elem[:, 1], elem[:, 2]
    lo = np.concatenate([np.minimum(a, b), np.minimum(b, c), np.minimum(c, a)])
    hi = np.concatenate([np.maximum(a, b), np.maximum(b, c), np.maximum(c, a)])
    key = lo * np.int64(N) + hi
    own = np.tile(np.arange(E, dtype=np.int64), 3)
    o   = np.argsort(key, kind='stable')
    key, own = key[o], own[o]
    same = np.flatnonzero(key[1:] == key[:-1])
    p1, p2 = own[same], own[same + 1]
    src = np.concatenate([p1, p2])
    dst = np.concatenate([p2, p1])
    o2  = np.argsort(src, kind='stable')
    e2e = dst[o2]
    e_off = np.zeros(E + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=E), out=e_off[1:])
    del lo, hi, key, own, src, dst, p1, p2, same, o, o2
log(f"# adjacency built ({time.time()-t0:.0f}s)")


def read_my_list(path):
    tok = np.fromfile(path, sep=' ', dtype=np.int64)
    i = 0
    rank = int(tok[i]); i += 1
    myn  = int(tok[i]); i += 1
    edn  = int(tok[i]); i += 1
    nodes = tok[i:i + myn + edn] - 1;  i += myn + edn
    mye = int(tok[i]); i += 1
    ede = int(tok[i]); i += 1
    exe = int(tok[i]); i += 1
    elems = tok[i:i + mye + ede + exe] - 1;  i += mye + ede + exe
    myed = int(tok[i]); i += 1
    eded = int(tok[i]); i += 1
    edges_l = tok[i:i + myed + eded] - 1
    return rank, myn, edn, nodes, mye, ede, exe, elems, edges_l


def read_com_info(path, edn, ede, exe):
    """rPEnum/sPEnum per block; block order nod2D, elem2D, elem2D_full
    (fesom_partit.cpp:203-206). rlist_size is not in the file — it follows the
    partition convention, hence the sizes passed in."""
    tok = np.fromfile(path, sep=' ', dtype=np.int64)
    p, out = 1, []
    for rlist_size in (edn, ede, ede + exe):
        rPEnum = int(tok[p]); p += 1
        p += rPEnum + (rPEnum + 1) + rlist_size
        sPEnum = int(tok[p]); p += 1
        sptr = tok[p + sPEnum : p + 2 * sPEnum + 1]
        p += 2 * sPEnum + 1
        p += int(sptr[sPEnum] - sptr[0]) if sPEnum else 0
        out.append((rPEnum, sPEnum))
    return out


# node -> incident edges (CSR), for the local-edge-coverage check
edge_file = os.path.join(mesh_dir, 'edges.out')
n2ed = ed_off = None
if os.path.exists(edge_file):
    ed = np.fromfile(edge_file, sep=' ', dtype=np.int64)
    if ed.size % 2:
        ed = ed[1:]                      # tolerate a leading count line
    ep = ed.reshape(-1, 2) - 1
    fe = ep.ravel()
    ei = np.repeat(np.arange(ep.shape[0], dtype=np.int64), 2)
    oe = np.argsort(fe, kind='stable')
    n2ed = ei[oe]
    ed_off = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(fe, minlength=N), out=ed_off[1:])
    log(f"# edge adjacency built: {ep.shape[0]} edges ({time.time()-t0:.0f}s)")


g2l = np.full(E, -1, dtype=np.int64)

for dist in dists:
    files = sorted(f for f in os.listdir(dist) if f.startswith('my_list'))
    npes  = len(files)
    if sample and sample < npes:
        files = files[::max(1, npes // sample)][:sample]
    log(f"\n=== {os.path.basename(mesh_dir)} / {os.path.basename(dist)} "
        f"({npes} ranks, {len(files)} sampled) ===")

    stats, miss_tot, enb_tot, enb_miss = [], 0, 0, 0
    edge_need = edge_miss = 0
    partners = []
    ring_n = np.zeros((len(files), R));  ring_e = np.zeros((len(files), R))
    for ri, fn in enumerate(files):
        (rank, myn, edn, nodes, mye, ede, exe,
         elems, edges_l) = read_my_list(os.path.join(dist, fn))
        owned = nodes[:myn]
        ring1 = nodes[myn:]
        g2l[:] = -1
        g2l[elems] = np.arange(len(elems), dtype=np.int64)

        # containment of ring-1 nodes' incident elements
        inc = np.unique(n2e[ragged(n_off, ring1)]) if edn else np.empty(0, np.int64)
        miss_tot += int((g2l[inc] < 0).sum())

        # owned elements' edge-neighbours in the eXDim tail
        if do_enb:
            nb = e2e[ragged(e_off, elems[:mye])]
            l  = g2l[nb]
            enb_tot  += int((l >= mye + ede).sum())
            enb_miss += int((l < 0).sum())

        # can a ring-1 div-CSR row be assembled from the LOCAL edge list?
        if n2ed is not None and edn:
            need = np.unique(n2ed[ragged(ed_off, ring1)])
            edge_need += need.size
            edge_miss += int(np.isin(need, edges_l, assume_unique=True,
                                     invert=True).sum())

        # partner counts (messages/substep, today vs the K=1 rung)
        ci = os.path.join(dist, 'com_info' + fn[len('my_list'):])
        if os.path.exists(ci):
            partners.append(read_com_info(ci, edn, ede, exe))

        # ring growth
        seen_n = np.zeros(N, dtype=bool);  seen_n[owned] = True
        seen_e = np.zeros(E, dtype=bool)
        front = owned
        for r in range(R):
            en = np.unique(n2e[ragged(n_off, front)])
            en = en[~seen_e[en]];  seen_e[en] = True
            nn = np.unique(elem[en].ravel()) if en.size else np.empty(0, np.int64)
            nn = nn[~seen_n[nn]];  seen_n[nn] = True
            ring_e[ri, r] = en.size
            ring_n[ri, r] = nn.size
            front = nn
            if nn.size == 0:
                break
        stats.append((myn, edn, mye, ede, exe))

    s = np.array(stats, float)
    myn, edn, mye, ede, exe = s.mean(axis=0)
    log(f"per rank: nodes owned {myn:.0f} + halo {edn:.0f} ({100*edn/myn:.1f}%) | "
        f"elems owned {mye:.0f} + eDim {ede:.0f} + eXDim {exe:.0f} "
        f"(full/owned {(mye+ede+exe)/mye:.3f})")
    r1 = s[:, 1]
    log(f"ring-1 spread over ranks: min {r1.min():.0f} med {np.median(r1):.0f} "
        f"max {r1.max():.0f}  (max/med {r1.max()/max(np.median(r1),1):.2f} — the load "
        f"imbalance the ring itself adds)")
    log(f"containment: {miss_tot} elements incident to a ring-1 node missing from the local list")
    if do_enb:
        log(f"enb tail:    {enb_tot} owned-element edge-neighbours with local idx >= myDim+eDim"
            f"   (not in local list at all: {enb_miss})")
    if edge_need:
        log(f"local edges: {edge_miss}/{edge_need} ({100*edge_miss/edge_need:.1f}%) of the edges "
            f"incident to a ring-1 node are NOT in the local edge list "
            f"=> a ring-1 CSR row cannot be assembled locally; the owner's row must be shipped")
    if partners:
        pa = np.array(partners, float)          # [rank, block, (rPE,sPE)]
        base = pa[:, 0, 0] + pa[:, 1, 0]
        wide = pa[:, 2, 0]
        log(f"partners:    recv nod2D {pa[:,0,0].mean():.2f} | elem2D {pa[:,1,0].mean():.2f} | "
            f"elem2D_full {pa[:,2,0].mean():.2f}")
        log(f"messages/substep: {base.mean():.2f} -> {wide.mean():.2f} "
            f"= x{wide.mean()/base.mean():.3f}  (worst rank {base.max():.0f} -> {wide.max():.0f})")
        d_now = s[:, 1].mean() + 2 * s[:, 3].mean()          # eta(eDim) + Ubt(eDim, 2 comp)
        d_rng = 2 * (s[:, 3].mean() + s[:, 4].mean())        # Ubt over eDim+eXDim
        log(f"doubles/substep:  {d_now:.0f} -> {d_rng:.0f} = x{d_rng/d_now:.2f}")
    # rings 1..R:  ring r of this BFS = elements first reached from node ring r-1,
    # so ring index 0 below is "elements incident to owned nodes" (= myDim_elem2D).
    cn = np.cumsum(ring_n.mean(axis=0));  ce = np.cumsum(ring_e.mean(axis=0))
    log("ring |  nodes  cum  cum/owned |  elems  cum  cum/owned")
    for r in range(R):
        log(f"{r+1:>4} | {ring_n.mean(axis=0)[r]:>6.0f} {cn[r]:>5.0f} {cn[r]/myn:>10.2f} |"
            f" {ring_e.mean(axis=0)[r]:>6.0f} {ce[r]:>5.0f} {ce[r]/mye:>10.2f}")

if os.environ.get('CENSUS_OUT'):
    with open(os.environ['CENSUS_OUT'], 'a') as f:
        f.write('\n'.join(out) + '\n')
