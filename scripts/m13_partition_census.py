#!/usr/bin/env python3
"""M13 CG-blowups: structural census of a dist_N partition against its mesh files.

Verifies, per rank, the invariants every edge/element-loop assembly in FESOM
(and our port) silently assumes:
  V1  every node referenced by an owned/halo EDGE is in the rank's node set
      (owned+halo)
  V2  every vertex of every local ELEMENT (my+eDim+eXDim) is in the node set
  V3  for every OWNED edge, both edge_tri adjacent elements are present among
      the rank's FIRST myDim_elem2D elements (<=0 = ocean boundary is fine)
      -> the port's defensive skip (fesom_ssh.cpp:158) fires on violations
  V4  ROW COMPLETENESS: for every OWNED node, every globally-incident edge is
      present in the rank's OWNED edge list -> else that node's matrix row /
      flux sum misses a contribution (partition-locked local defect)
  V5  duplicates / out-of-range ids in the my_list arrays themselves

Usage: m13_partition_census.py MESH_DIR N [N2 ...] [--ranks=R0:R1]
"""
import sys, os, glob
import numpy as np

def parse_ints_file(path):
    s = open(path).read()
    try:
        a = np.fromstring(s, dtype=np.int64, sep=' ')
        if a.size:
            return a
    except Exception:
        pass
    return np.array(s.split(), dtype=np.int64)

def load_mesh(mesh):
    print(f"[mesh] loading {mesh} ...", flush=True)
    nod2D = int(open(os.path.join(mesh, 'nod2d.out')).readline())
    e = parse_ints_file(os.path.join(mesh, 'elem2d.out'))
    elem2D = int(e[0]); elem = e[1:].reshape(elem2D, 3)   # 1-based node ids
    en = parse_ints_file(os.path.join(mesh, 'edgenum.out'))
    edge2D = int(en[0])
    ed = parse_ints_file(os.path.join(mesh, 'edges.out')).reshape(edge2D, 2)
    et = parse_ints_file(os.path.join(mesh, 'edge_tri.out')).reshape(edge2D, 2)
    print(f"[mesh] nod2D={nod2D} elem2D={elem2D} edge2D={edge2D} "
          f"edge_tri range [{et.min()},{et.max()}]", flush=True)
    # node -> incident-edge CSR (for V4): sort edge endpoints by node
    both = np.concatenate([ed[:, 0], ed[:, 1]])
    eids = np.concatenate([np.arange(1, edge2D + 1, dtype=np.int64)] * 2)
    order = np.argsort(both, kind='stable')
    inc = eids[order]
    inc_off = np.zeros(nod2D + 2, dtype=np.int64)
    np.add.at(inc_off, both + 1, 1)
    inc_off = np.cumsum(inc_off)
    return dict(nod2D=nod2D, elem2D=elem2D, edge2D=edge2D,
                elem=elem, ed=ed, et=et, inc=inc, inc_off=inc_off)

def census(mesh, M, N, r0=0, r1=None):
    dist = os.path.join(mesh, f'dist_{N}')
    files = sorted(glob.glob(os.path.join(dist, 'my_list*.out')))
    assert len(files) == N, f"{dist}: {len(files)} my_list files != {N}"
    if r1 is None: r1 = N
    node_stamp = np.zeros(M['nod2D'] + 1, dtype=np.int32)
    edge_stamp = np.zeros(M['edge2D'] + 1, dtype=np.int32)
    elem_stamp = np.zeros(M['elem2D'] + 1, dtype=np.int32)
    stats = dict(v1=0, v2=0, v3_missing=0, v3_halo=0, v4=0, v5=0,
                 v1_ranks=set(), v2_ranks=set(), v3_ranks=set(), v4_ranks=set(),
                 first=[])
    hdr = np.zeros((N, 7), dtype=np.int64)   # mn,en, me,ee,xe, med,eed
    for r in range(r0, r1):
        a = parse_ints_file(files[r])
        p = 0
        rank = a[p]; p += 1
        assert rank == r, (files[r], rank)
        mn, en = int(a[p]), int(a[p+1]); p += 2
        nodes = a[p:p+mn+en]; p += mn + en
        me, ee, xe = int(a[p]), int(a[p+1]), int(a[p+2]); p += 3
        elems = a[p:p+me+ee+xe]; p += me + ee + xe
        med, eed = int(a[p]), int(a[p+1]); p += 2
        edges = a[p:p+med+eed]; p += med + eed
        assert p == len(a), (files[r], p, len(a))
        hdr[r] = (mn, en, me, ee, xe, med, eed)
        tag = r + 1
        # V5 range/dup checks
        bad = int(((nodes < 1) | (nodes > M['nod2D'])).sum()
                  + ((elems < 1) | (elems > M['elem2D'])).sum()
                  + ((edges < 1) | (edges > M['edge2D'])).sum())
        if len(np.unique(nodes)) != len(nodes): bad += 1
        if len(np.unique(edges)) != len(edges): bad += 1
        if bad:
            stats['v5'] += bad; stats['first'].append(('V5', r, bad))
            continue   # stamping with bad ids would crash; report and move on
        # stamps
        node_stamp[nodes] = tag
        elem_stamp[elems[:me]] = tag          # owned elems
        elem_stamp[elems[me:]] = -tag         # halo elems
        # V1: edge endpoints inside node set
        end = M['ed'][edges - 1]              # (med+eed, 2)
        v1 = int((node_stamp[end] != tag).sum())
        if v1:
            stats['v1'] += v1; stats['v1_ranks'].add(r)
            if len(stats['first']) < 24:
                w = np.argwhere(node_stamp[end] != tag)[0]
                stats['first'].append(('V1', r, int(edges[w[0]]), int(end[w[0], w[1]])))
        # V2: element vertices inside node set
        vv = M['elem'][elems - 1]
        v2 = int((node_stamp[vv] != tag).sum())
        if v2:
            stats['v2'] += v2; stats['v2_ranks'].add(r)
            if len(stats['first']) < 24:
                w = np.argwhere(node_stamp[vv] != tag)[0]
                stats['first'].append(('V2', r, int(elems[w[0]]), int(vv[w[0], w[1]])))
        # V3: owned edges' adjacent elements present locally as OWNED
        own_ed = edges[:med]
        adj = M['et'][own_ed - 1]             # (med,2), <=0 = boundary
        real = adj > 0
        st = elem_stamp[np.where(real, adj, 1)]
        miss = int((real & (st != tag) & (st != -tag)).sum())
        halo = int((real & (st == -tag)).sum())
        if miss:
            stats['v3_missing'] += miss; stats['v3_ranks'].add(r)
            if len(stats['first']) < 24:
                w = np.argwhere(real & (st != tag) & (st != -tag))[0]
                stats['first'].append(('V3', r, int(own_ed[w[0]]), int(adj[w[0], w[1]])))
        if halo:
            stats['v3_halo'] += halo; stats['v3_ranks'].add(r)
        # V4: row completeness for owned nodes
        edge_stamp[own_ed] = tag
        own_nodes = nodes[:mn]
        starts = M['inc_off'][own_nodes]; ends = M['inc_off'][own_nodes + 1]
        lens = ends - starts
        tot = int(lens.sum())
        idx = np.repeat(starts, lens) + (np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens))
        inc_edges = M['inc'][idx]
        v4 = int((edge_stamp[inc_edges] != tag).sum())
        if v4:
            stats['v4'] += v4; stats['v4_ranks'].add(r)
            if len(stats['first']) < 24:
                w = np.where(edge_stamp[inc_edges] != tag)[0][0]
                bad_node = own_nodes[np.searchsorted(np.cumsum(lens), w, side='right')]
                stats['first'].append(('V4', r, int(inc_edges[w]), int(bad_node)))
        if (r - r0) % 4096 == 0:
            print(f"  rank {r}: v1={stats['v1']} v2={stats['v2']} "
                  f"v3m={stats['v3_missing']} v3h={stats['v3_halo']} v4={stats['v4']}", flush=True)
    print(f"== {dist} ranks[{r0}:{r1}] SUMMARY")
    print(f"   V1 edge-endpoint-outside-nodeset    : {stats['v1']}  ranks={len(stats['v1_ranks'])}")
    print(f"   V2 elem-vertex-outside-nodeset      : {stats['v2']}  ranks={len(stats['v2_ranks'])}")
    print(f"   V3 owned-edge adj-elem MISSING      : {stats['v3_missing']}  (halo-only={stats['v3_halo']})  ranks={len(stats['v3_ranks'])}")
    print(f"   V4 owned-node incident-edge unowned : {stats['v4']}  ranks={len(stats['v4_ranks'])}")
    print(f"   V5 range/dup defects                : {stats['v5']}")
    print(f"   sums: nodes my/e {hdr[:,0].sum()}/{hdr[:,1].sum()}  elems my/e/x "
          f"{hdr[:,2].sum()}/{hdr[:,3].sum()}/{hdr[:,4].sum()}  edges my/e {hdr[:,5].sum()}/{hdr[:,6].sum()}")
    print(f"   header mins: mn={hdr[:,0].min()} me={hdr[:,2].min()} med={hdr[:,5].min()}")
    for f in stats['first'][:24]: print("   first:", f)
    sys.stdout.flush()
    return stats, hdr

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    mesh = args[0]; Ns = [int(x) for x in args[1:]]
    r0, r1 = 0, None
    for a in sys.argv[1:]:
        if a.startswith('--ranks='):
            r0, r1 = [int(x) for x in a.split('=')[1].split(':')]
    M = load_mesh(mesh)
    for N in Ns:
        census(mesh, M, N, r0, r1)
