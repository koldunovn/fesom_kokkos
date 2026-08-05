#!/usr/bin/env python3
"""Does the delayed halo exchange leave an imprint of the DOMAIN DECOMPOSITION in the ice?

The concern is well posed and is not answered by a global correlation. Under the delayed
exchange the halo values a rank reads are up to K-1 sub-cycles old, and those halo nodes are
exactly the one-cell ring around each sub-domain. The partition is fixed for the whole run, so
whatever error is injected there is injected at the SAME nodes every step of every day. A
one-cell-wide feature along sub-domain edges is a tiny fraction of the mesh, so it can be
visually obvious and still leave a domain-wide correlation at 0.99999.

So the test is not "is the difference small" but "is the difference ORGANISED BY THE PARTITION".

Method
------
1. Read the partition actually used by the runs (`dist_<npes>/my_list*.out`, FESOM format:
   mype / myDim_nod2D / eDim_nod2D / the global node list, owned first then halo). The union of
   every rank's halo portion is EXACTLY the set of nodes that carry stale values -- not a proxy
   for it.
2. Grow that set outward through the element connectivity to get a ring distance per node:
   0 = a halo node itself, 1 = shares an element with one, and so on.
3. Bin |test - reference| by ring distance. **If the delayed exchange imprints the partition,
   the mean must peak at distance 0 and decay outward. If the curve is flat, there is no
   imprint** -- the difference is spread over the domain like any other perturbation.
4. Report the ratio (mean over nodes at distance 0-1) / (mean over the interior, distance >= 3)
   with a bootstrap confidence interval, so "flat" is a number and not an impression.

The reference leg must be the SAME binary and rank count with the option off; on CPU that leg is
exactly reproducible, so every difference seen is the approximation and nothing else.

usage:
  m9_partition_artefact.py --ref .../p1c_classic --test .../p1c_lag4 \
      --mesh /work/ab0995/a270088/port2/mesh/core2 --npes 256 --var a_ice --outdir ...
"""
import argparse, os, sys
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--npes", type=int, required=True,
                help="rank count the runs actually used -- its partition defines the rings")
ap.add_argument("--distnpes", type=int, default=None,
                help="PLACEBO: build the rings from a DIFFERENT partition, one the runs never "
                     "used. The imprint must follow the run's OWN decomposition; if the same "
                     "edge/interior ratio appears against a partition the model never saw, the "
                     "ratio is geography (rim nodes sitting in more dynamic ice), not an "
                     "artefact. Running both is what separates the two.")
ap.add_argument("--var", default="a_ice")
ap.add_argument("--pattern", default="{var}.fesom.{year}.monthly.nc")
ap.add_argument("--year", default="1958")
ap.add_argument("--outdir", default=".")
ap.add_argument("--label", default=None)
ap.add_argument("--maxring", type=int, default=6)
ap.add_argument("--icemin", type=float, default=0.01,
                help="keep nodes whose max|field| exceeds this fraction of the domain max "
                     "(0 = keep everything). Open ocean would otherwise dilute the statistic.")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
label = args.label or (f"{os.path.basename(args.test)}_vs_{os.path.basename(args.ref)}_{args.var}"
                       + (f"_placebo{args.distnpes}" if args.distnpes else ""))


def read_field(d):
    p = os.path.join(d, args.pattern.format(var=args.var, year=args.year))
    with nc.Dataset(p) as f:
        v = np.asarray(f.variables[args.var][:])
        lon = np.asarray(f.variables["lon"][:])
        lat = np.asarray(f.variables["lat"][:])
    return v, lon, lat


def halo_nodes(mesh, npes, nnodes):
    """Union of every rank's halo list = exactly the nodes that carry stale values."""
    mask = np.zeros(nnodes, dtype=bool)
    for r in range(npes):
        p = os.path.join(mesh, f"dist_{npes}", f"my_list{r:05d}.out")
        vals = np.fromstring(open(p).read().replace("\n", " "), sep=" ", dtype=np.int64)
        mype, mydim, edim = int(vals[0]), int(vals[1]), int(vals[2])
        lst = vals[3:3 + mydim + edim]
        if lst.size != mydim + edim:
            sys.exit(f"{p}: expected {mydim+edim} indices, got {lst.size}")
        halo = lst[mydim:] - 1                      # 1-based -> 0-based
        mask[halo] = True
    return mask


def ring_distance(mesh, nnodes, seed, maxring):
    """Hops through the element connectivity from the halo set. -1 = further than maxring."""
    el = np.loadtxt(os.path.join(mesh, "elem2d.out"), skiprows=1, dtype=np.int64) - 1
    # node -> elements, as a CSR built once
    order = np.argsort(el.ravel(), kind="stable")
    node_of_entry = el.ravel()[order]
    elem_of_entry = order // 3
    ptr = np.searchsorted(node_of_entry, np.arange(nnodes + 1))
    dist = np.full(nnodes, -1, dtype=np.int16)
    dist[seed] = 0
    front = np.flatnonzero(seed)
    for d in range(1, maxring + 1):
        if front.size == 0:
            break
        elems = np.unique(np.concatenate([elem_of_entry[ptr[n]:ptr[n + 1]] for n in front])) \
            if front.size else np.array([], dtype=np.int64)
        cand = np.unique(el[elems].ravel())
        new = cand[dist[cand] < 0]
        dist[new] = d
        front = new
    return dist


ref, lon, lat = read_field(args.ref)
tst, _, _ = read_field(args.test)
if ref.shape != tst.shape:
    sys.exit(f"shape mismatch {ref.shape} vs {tst.shape}")
nnodes = ref.shape[-1]

ring_npes = args.distnpes or args.npes
seed = halo_nodes(args.mesh, ring_npes, nnodes)
dist = ring_distance(args.mesh, nnodes, seed, args.maxring)
tagline = ("the runs' OWN partition" if ring_npes == args.npes
           else f"PLACEBO partition dist_{ring_npes} -- the runs used dist_{args.npes}")
print(f"rings from {tagline}: {seed.sum()} halo nodes of {nnodes} "
      f"({100*seed.sum()/nnodes:.1f}% of the mesh)")
if args.distnpes:
    own = halo_nodes(args.mesh, args.npes, nnodes)
    ov = (own & seed).sum() / seed.sum()
    print(f"   overlap with the real rim: {100*ov:.1f}%  "
          f"(chance level {100*own.sum()/nnodes:.1f}% -- a good placebo is near chance)")

D = np.abs(tst - ref)                       # (months, nodes)
Dann = D.mean(axis=0)                       # mean of |difference| over the record

# ⚠️ RESTRICT TO NODES THAT CARRY ICE. Without this the statistic is diluted by open ocean,
# where the difference is identically zero and which is most of the mesh on a global grid --
# and the dilution is not uniform, because the partition rim is distributed over the whole
# domain while the ice is not. On an Arctic-refined mesh (fArc) the sub-domains sit INSIDE the
# ice pack and the dilution largely disappears, which is exactly why that mesh is the harder
# test. `--icemin 0` disables the mask.
icevar = np.abs(ref).max(axis=0)
ice = icevar > args.icemin * (icevar.max() if icevar.max() > 0 else 1.0)
if not ice.any():
    sys.exit("no node passes the ice mask; check --var/--icemin")
print(f"ice mask: {ice.sum()} of {nnodes} nodes carry the field "
      f"(max |{args.var}| > {args.icemin:g} of its domain max)")

# ---- the decisive statistic: |difference| binned by distance to a partition boundary --------
rings = list(range(0, args.maxring + 1))
mean_by_ring, n_by_ring = [], []
for d in rings:
    m = (dist == d) & ice
    n_by_ring.append(int(m.sum()))
    mean_by_ring.append(float(Dann[m].mean()) if m.any() else np.nan)
far = (dist < 0) & ice
interior_mean = float(Dann[far].mean()) if far.any() else float(Dann[(dist >= 3) & ice].mean())

edge = np.isin(dist, (0, 1)) & ice
edge_mean = float(Dann[edge].mean())
ratio = edge_mean / interior_mean if interior_mean > 0 else np.nan

rng = np.random.default_rng(12345)          # fixed seed: the number must reproduce
boot = []
ie, ii = np.flatnonzero(edge), np.flatnonzero(far if far.any() else (dist >= 3) & ice)
for _ in range(400):
    boot.append(Dann[rng.choice(ie, ie.size)].mean() / Dann[rng.choice(ii, ii.size)].mean())
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f"\n{args.var}: mean |difference| by distance from a partition boundary")
for d, m, n in zip(rings, mean_by_ring, n_by_ring):
    print(f"   ring {d}: {m:.6g}   ({n} nodes)")
print(f"   interior (beyond ring {args.maxring}): {interior_mean:.6g} "
      f"({int((far).sum())} nodes)")
print(f"\n   EDGE/INTERIOR RATIO = {ratio:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
print("   1.0 = no partition imprint; >1 = the difference concentrates on the sub-domain rim")

# ---- figure: the ring curve + the difference map ---------------------------------------------
fig = plt.figure(figsize=(11.0, 4.3), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25])

ax = fig.add_subplot(gs[0, 0])
ax.plot(rings, mean_by_ring, marker="o", color="#c0392b")
ax.axhline(interior_mean, color="#333333", ls="--", lw=1.0,
           label=f"interior mean ({interior_mean:.3g})")
ax.set_xlabel("elements away from a sub-domain boundary")
ax.set_ylabel(f"mean |$\\Delta$ {args.var}|, annual")
ax.set_title(f"edge/interior = {ratio:.2f}  [{lo:.2f}, {hi:.2f}]", fontsize=9)
ax.grid(alpha=.3)
ax.legend(fontsize=7)

# northern hemisphere ice, where the signal would be
sel = (lat > 45) & ice
vmax = np.percentile(Dann[sel][Dann[sel] > 0], 99) if (Dann[sel] > 0).any() else 1e-12
ax2 = fig.add_subplot(gs[0, 1])
sc = ax2.scatter(lon[sel], lat[sel], c=Dann[sel], s=1.4, cmap="magma_r", vmin=0, vmax=vmax)
ax2.set_title(f"|$\\Delta$ {args.var}| annual mean, NH", fontsize=9)
ax2.set_xlabel("longitude"); ax2.set_ylabel("latitude")
fig.colorbar(sc, ax=ax2, shrink=.85)

ax3 = fig.add_subplot(gs[0, 2])
h = dist[sel] == 0
ax3.scatter(lon[sel][~h], lat[sel][~h], c="#dddddd", s=1.0)
ax3.scatter(lon[sel][h], lat[sel][h], c="#1f6fb4", s=1.0)
ax3.set_title(f"where the stale values live\n({args.npes} sub-domains, halo nodes in blue)",
              fontsize=9)
ax3.set_xlabel("longitude")
fig.suptitle(f"Partition imprint test: {os.path.basename(args.test)} vs "
             f"{os.path.basename(args.ref)}", fontsize=10)
fig.savefig(os.path.join(args.outdir, f"artefact_{label}.png"), dpi=140)
print(f"\nwrote {os.path.join(args.outdir, f'artefact_{label}.png')}")
