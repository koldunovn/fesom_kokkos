#!/usr/bin/env python3
"""M12b — how many PARTNER RANKS a K-ring halo touches (the deep-K decision input).

The K=1 rung halves the messages per substep for a reason that is specific to K=1:
`com_elem2D_full` reaches the SAME partner ranks as `com_elem2D` (measured ×0.500 at
11 points — more entities per partner, not more partners). Nothing guarantees that
at K≥2: a wider zone can reach round the corner to ranks that are not neighbours at
all, and then a deep rung trades FEWER exchanges for MORE messages each.

So count it. For sampled ranks this walks the BFS rings the census uses (ring r
elements = elements first reached from node ring r-1; ring r nodes = their new
vertices) and reports, per cumulative ring, how many distinct OTHER ranks own the
zone's entities — that is the partner count an exchange over that zone would need.

Conventions, stated because they matter:
  * node owner  = the rank whose my_list carries the node in its OWNED block.
  * element owner = the LOWEST-ranked claimant. `myDim_elem2D` is not a partition
    (M12b s3: ~0.55 % of elements are claimed by several ranks), so an owner rule
    has to be chosen; the rung's F-reconcile picks one the same way.
  * the K=1 rung's zones are node ring 1 (eDim) and element rings 1-2
    (eDim+eXDim = what com_elem2D_full delivers). A K-substep rung therefore needs
    node rings 1..K and element rings 1..K+1 — the rim algebra of WIDEHALO_M12B §2.

Usage:
  m12b_partner_growth.py <mesh_dir> <dist_dir> [<dist_dir> ...]
                         [--rings R] [--sample N] [--M m]
`--M` (per-mesh substeps) turns the partner counts into messages/step for K=1,2,4,8
against the certified 2M baseline.
"""
import sys, os, time
import numpy as np

argv = list(sys.argv[1:])
R, sample, M = 9, 8, None
while '--rings' in argv:
    i = argv.index('--rings');  R = int(argv[i + 1]);      del argv[i:i + 2]
while '--sample' in argv:
    i = argv.index('--sample'); sample = int(argv[i + 1]); del argv[i:i + 2]
while '--M' in argv:
    i = argv.index('--M');      M = int(argv[i + 1]);      del argv[i:i + 2]
mesh_dir, dists = argv[0], argv[1:]

out = []


def log(*a):
    print(*a, flush=True)
    out.append(' '.join(str(x) for x in a))


def ragged(off, sel):
    starts = off[sel]
    counts = off[sel + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    cum = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return np.repeat(starts - cum, counts) + np.arange(total, dtype=np.int64)


def read_my_list(path):
    tok = np.fromfile(path, sep=' ', dtype=np.int64)
    i = 0
    rank = int(tok[i]); i += 1
    myn = int(tok[i]); i += 1
    edn = int(tok[i]); i += 1
    nodes = tok[i:i + myn + edn] - 1;  i += myn + edn
    mye = int(tok[i]); i += 1
    ede = int(tok[i]); i += 1
    exe = int(tok[i]); i += 1
    elems = tok[i:i + mye + ede + exe] - 1
    return rank, myn, edn, nodes, mye, ede, exe, elems


t0 = time.time()
elem = np.fromfile(os.path.join(mesh_dir, 'elem2d.out'), sep=' ', dtype=np.int64)
E = int(elem[0])
elem = (elem[1:1 + 3 * E] - 1).reshape(E, 3)
N = int(elem.max()) + 1
log(f"# mesh {mesh_dir}: {N} nodes, {E} elements ({time.time()-t0:.0f}s)")

flat_n = elem.ravel()
flat_e = np.repeat(np.arange(E, dtype=np.int64), 3)
order = np.argsort(flat_n, kind='stable')
n2e = flat_e[order]
n_off = np.zeros(N + 1, dtype=np.int64)
np.cumsum(np.bincount(flat_n, minlength=N), out=n_off[1:])
log(f"# adjacency built ({time.time()-t0:.0f}s)")

for dist in dists:
    files = sorted(f for f in os.listdir(dist) if f.startswith('my_list'))
    npes = len(files)
    log(f"\n=== {os.path.basename(mesh_dir)} / {os.path.basename(dist)} ({npes} ranks) ===")

    # global owner maps — every rank's file is read once (the owned blocks only)
    node_owner = np.full(N, -1, dtype=np.int32)
    elem_owner = np.full(E, np.iinfo(np.int32).max, dtype=np.int32)
    keep = {}
    pick = set(range(0, npes, max(1, npes // sample)))
    for idx, fn in enumerate(files):
        rank, myn, edn, nodes, mye, ede, exe, elems = read_my_list(os.path.join(dist, fn))
        node_owner[nodes[:myn]] = rank
        np.minimum.at(elem_owner, elems[:mye], np.int32(rank))
        if idx in pick:
            keep[rank] = (nodes[:myn], myn, edn, mye, ede, exe)
    log(f"# owner maps built from {npes} my_list files, {len(keep)} ranks sampled "
        f"({time.time()-t0:.0f}s); unowned nodes {int((node_owner < 0).sum())}")

    pn = np.zeros((len(keep), R))     # partners of the cumulative NODE zone
    pe = np.zeros((len(keep), R))     # partners of the cumulative ELEM zone
    zn = np.zeros((len(keep), R))     # cumulative node zone / owned
    ze = np.zeros((len(keep), R))     # cumulative elem zone / owned
    for ri, (rank, (owned, myn, edn, mye, ede, exe)) in enumerate(sorted(keep.items())):
        seen_n = np.zeros(N, dtype=bool);  seen_n[owned] = True
        seen_e = np.zeros(E, dtype=bool)
        front = owned
        cum_n, cum_e = [], []
        for r in range(R):
            en = np.unique(n2e[ragged(n_off, front)])
            en = en[~seen_e[en]];  seen_e[en] = True
            nn = np.unique(elem[en].ravel()) if en.size else np.empty(0, np.int64)
            nn = nn[~seen_n[nn]];  seen_n[nn] = True
            cum_n.append(nn);  cum_e.append(en)
            # partners = distinct other-rank owners over the cumulative zone
            hz = np.concatenate(cum_n) if cum_n else np.empty(0, np.int64)
            ez = np.concatenate(cum_e) if cum_e else np.empty(0, np.int64)
            on = np.unique(node_owner[hz]);  on = on[on != rank]
            oe = np.unique(elem_owner[ez]);  oe = oe[oe != rank]
            pn[ri, r] = on.size
            pe[ri, r] = oe.size
            zn[ri, r] = hz.size / myn
            ze[ri, r] = ez.size / mye
            front = nn
            if nn.size == 0:
                break

    log("ring |  node zone/owned  partners | elem zone/owned  partners")
    for r in range(R):
        log(f"{r+1:>4} | {zn[:,r].mean():>15.2f} {pn[:,r].mean():>9.1f} |"
            f" {ze[:,r].mean():>15.2f} {pe[:,r].mean():>9.1f}")

    # the K=1 rung's own zones, for calibration against com_info's rPEnum
    log(f"K=1 rung zones: node ring 1 partners {pn[:,0].mean():.1f} | "
        f"elem rings 1-2 partners {pe[:,1].mean():.1f}  "
        f"(compare com_elem2D_full rPEnum from the ring census)")

    if M:
        log(f"\nmessages/step at M={M} (exchange every K substeps; +3 per-step coherence waves):")
        log("   K | exchanges/step | partners | messages/step | vs certified 2M")
        base = M * (pn[:, 0].mean() + pe[:, 1].mean())   # eta over ring1 + Ubt over rings1-2
        for K in (1, 2, 4, 8):
            if K > R - 1:
                break
            ex = int(np.ceil(M / K))
            part = pe[:, min(K, R - 1)].mean()           # elem zone for K substeps = rings 1..K+1
            msg = ex * part + 3 * part
            log(f"{K:>4} | {ex:>14} | {part:>8.1f} | {msg:>13.0f} | "
                f"x{msg/base:>5.3f}")
        log(f"   certified baseline: {M} substeps x (eta+Ubt) partners = {base:.0f} messages/step")

if os.environ.get('PARTNER_OUT'):
    with open(os.environ['PARTNER_OUT'], 'a') as f:
        f.write('\n'.join(out) + '\n')
