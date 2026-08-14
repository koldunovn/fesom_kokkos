#!/usr/bin/env python3
"""M13: com_info census — validate halo-exchange routing tables of a dist_N.

Per rank r, per com block (nod2D, elem2D, elem2D_full):
  C1  rlist local indices land in the HALO range of that entity
      (nod2D: (mn, mn+en]; elem2D: (me, me+ee]; full: (me, me+ee+xe])
      -> an entry in the OWNED range means the exchange OVERWRITES owned data
  C2  slist local indices land in the OWNED range [1, myDim]
  C3  rptr/sptr monotone; rlist total == eDim (nod2D/elem2D) or eDim+eXDim
  C5  every halo slot appears exactly once in rlist (nod2D block)
  C4  cross-rank reciprocity in GID space: gids(A.slist->B) == gids(B.rlist<-A)
      elementwise IN ORDER (value routing correctness)

Usage: m13_cominfo_census.py MESH_DIR N [N...]
"""
import sys, os
import numpy as np

def ints(p):
    s = open(p).read()
    try:
        a = np.fromstring(s, dtype=np.int64, sep=' ')
        if a.size: return a
    except Exception: pass
    return np.array(s.split(), dtype=np.int64)

def census(mesh, N):
    dist = os.path.join(mesh, f'dist_{N}')
    print(f"== {dist}", flush=True)
    bad = dict(c1=0, c2=0, c3=0, c5=0, c4=0, ranks=set())
    # store per (src,dst) the gid sequences for C4
    send_gid = {}   # (a,b) -> gids a sends to b   (nod2D block)
    recv_gid = {}   # (b,a) -> gids b expects from a
    send_gid_e = {}; recv_gid_e = {}   # elem2D block
    hdrs = np.zeros((N, 4), dtype=np.int64)
    for r in range(N):
        a = ints(os.path.join(dist, f'my_list{r:05d}.out'))
        p = 0; p += 1
        mn, en = int(a[p]), int(a[p+1]); p += 2
        nodes = a[p:p+mn+en]; p += mn+en
        me, ee, xe = int(a[p]), int(a[p+1]), int(a[p+2]); p += 3
        elems = a[p:p+me+ee+xe]; p += me+ee+xe
        c = ints(os.path.join(dist, f'com_info{r:05d}.out'))
        q = 0
        assert c[q] == r, (r, c[q]); q += 1
        specs = [('nod2D', mn, en, en, nodes),
                 ('elem2D', me, ee, ee, elems),
                 ('full',  me, ee, ee+xe, elems)]
        for kind, my, e1, rl_sz, glist in specs:
            rPEnum = int(c[q]); q += 1
            rPE   = c[q:q+rPEnum]; q += rPEnum
            rptr  = c[q:q+rPEnum+1]; q += rPEnum+1
            rlist = c[q:q+rl_sz]; q += rl_sz
            sPEnum = int(c[q]); q += 1
            sPE   = c[q:q+sPEnum]; q += sPEnum
            sptr  = c[q:q+sPEnum+1]; q += sPEnum+1
            sl_sz = int(sptr[-1] - sptr[0])
            slist = c[q:q+sl_sz]; q += sl_sz
            # C3
            if (np.diff(rptr) < 0).any() or (np.diff(sptr) < 0).any() \
               or int(rptr[-1] - rptr[0]) != rl_sz:
                bad['c3'] += 1; bad['ranks'].add(r)
            # C1: rlist in halo range
            n_owned_hit = int(((rlist >= 1) & (rlist <= my)).sum())
            n_oor = int(((rlist < 1) | (rlist > my + rl_sz)).sum()) if kind != 'full' \
                    else int(((rlist < 1) | (rlist > my + e1 + (rl_sz - e1))).sum())
            if n_owned_hit or n_oor:
                bad['c1'] += n_owned_hit + n_oor; bad['ranks'].add(r)
                if bad['c1'] < 20:
                    print(f"   C1 rank {r} {kind}: {n_owned_hit} rlist entries in OWNED range, {n_oor} out of range (my={my})")
            # C2
            s_bad = int(((slist < 1) | (slist > my)).sum())
            if s_bad:
                bad['c2'] += s_bad; bad['ranks'].add(r)
                if bad['c2'] < 20:
                    print(f"   C2 rank {r} {kind}: {s_bad} slist entries outside owned range")
            # C5 (nod2D only): halo coverage exactly once
            if kind == 'nod2D':
                u = np.unique(rlist)
                if len(u) != rl_sz or (rl_sz and (u[0] != my+1 or u[-1] != my+en)):
                    bad['c5'] += 1; bad['ranks'].add(r)
                    if bad['c5'] < 10:
                        print(f"   C5 rank {r}: rlist covers {len(u)} unique of {rl_sz}, range [{u[0] if len(u) else 0},{u[-1] if len(u) else 0}] vs ({my},{my+en}]")
                # C4 harvest (guard indices to valid range to avoid crash on bad data)
                for k in range(rPEnum):
                    seg = rlist[int(rptr[k])-1:int(rptr[k+1])-1]
                    ok = (seg >= 1) & (seg <= mn+en)
                    recv_gid[(r, int(rPE[k]))] = glist[seg[ok]-1]
                for k in range(sPEnum):
                    seg = slist[int(sptr[k])-1:int(sptr[k+1])-1]
                    ok = (seg >= 1) & (seg <= mn+en)
                    send_gid[(r, int(sPE[k]))] = glist[seg[ok]-1]
            if kind == 'elem2D':
                for k in range(rPEnum):
                    seg = rlist[int(rptr[k])-1:int(rptr[k+1])-1]
                    ok = (seg >= 1) & (seg <= me+ee+xe)
                    recv_gid_e[(r, int(rPE[k]))] = glist[seg[ok]-1]
                for k in range(sPEnum):
                    seg = slist[int(sptr[k])-1:int(sptr[k+1])-1]
                    ok = (seg >= 1) & (seg <= me+ee+xe)
                    send_gid_e[(r, int(sPE[k]))] = glist[seg[ok]-1]
        assert q == len(c), (r, q, len(c))
        hdrs[r] = (mn, en, me, ee)
        if r % 4096 == 0:
            print(f"   rank {r}: c1={bad['c1']} c2={bad['c2']} c3={bad['c3']} c5={bad['c5']}", flush=True)
    # C4 join
    mism = 0; pairs = 0; examples = []
    for (dst, src), g_recv in recv_gid.items():
        g_send = send_gid.get((src, dst))
        pairs += 1
        if g_send is None or len(g_send) != len(g_recv) or (g_send != g_recv).any():
            mism += 1
            if len(examples) < 10:
                if g_send is None:
                    examples.append((src, dst, 'missing send block'))
                elif len(g_send) != len(g_recv):
                    examples.append((src, dst, f'len {len(g_send)} vs {len(g_recv)}'))
                else:
                    w = int(np.argmax(g_send != g_recv))
                    examples.append((src, dst, f'first mismatch at pos {w}: send gid {g_send[w]} vs recv gid {g_recv[w]}'))
    mism_e = 0; pairs_e = 0; examples_e = []
    for (dst, src), g_recv in recv_gid_e.items():
        g_send = send_gid_e.get((src, dst))
        pairs_e += 1
        if g_send is None or len(g_send) != len(g_recv) or (g_send != g_recv).any():
            mism_e += 1
            if len(examples_e) < 10:
                examples_e.append((src, dst, 'mismatch'))
    print(f"   SUMMARY {dist}: C1={bad['c1']} C2={bad['c2']} C3={bad['c3']} C5={bad['c5']} "
          f"| C4 nod2D mismatched pairs {mism}/{pairs} elem2D {mism_e}/{pairs_e} | bad-ranks {len(bad['ranks'])}")
    for e in examples: print("   C4:", e)
    for e in examples_e: print("   C4e:", e)
    sys.stdout.flush()

if __name__ == '__main__':
    mesh = sys.argv[1]
    for N in sys.argv[2:]:
        census(mesh, int(N))
