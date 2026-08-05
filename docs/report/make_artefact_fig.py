#!/usr/bin/env python3
"""Report figure: does the option imprint the domain decomposition?

Left  -- mean |difference| against distance from a sub-domain boundary, for four configurations.
         The two exact ones must be flat (they are bit-identical, so their only difference from
         the reference is summation order); the approximate ones must peak at the boundary.
Right -- the same thing as a map, so the reader can see it rather than take the ratio on trust.

Reads the fArc 60-day daily output directly; nothing is entered by hand.
"""
import os, sys
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

C = "/work/ab0995/a270088/port2/m9/clim"
MESH = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc"
OUT = os.path.dirname(os.path.abspath(__file__))
NPES, VAR, MAXRING = 16, "m_ice", 6

LEGS = [("standard_rep", "two identical runs (control)", "#7f7f7f", "o"),
        ("wide8",        "wide halo, exact ($K$=8)",     "#1f6fb4", "s"),
        ("lag2",         "delayed exchange, $K$=2",      "#e8896b", "^"),
        ("lag4",         "delayed exchange, $K$=4",      "#c0392b", "D")]


def rd(tag, v=VAR):
    with nc.Dataset(f"{C}/art_farc_g16_{tag}/{v}.fesom.1958.daily.nc") as f:
        return (np.asarray(f.variables[v][:]), np.asarray(f.variables["lon"][:]),
                np.asarray(f.variables["lat"][:]))


R, lon, lat = rd("standard")
nn = R.shape[1]
el = np.loadtxt(f"{MESH}/elem2d.out", skiprows=1, dtype=np.int64) - 1

# nodes carrying stale values = the union of every rank's halo list
owner = np.full(nn, -1, np.int32)
seed = np.zeros(nn, bool)
for r in range(NPES):
    v = np.fromstring(open(f"{MESH}/dist_{NPES}/my_list{r:05d}.out").read().replace("\n", " "),
                      sep=" ", dtype=np.int64)
    md, ed = int(v[1]), int(v[2])
    owner[v[3:3 + md] - 1] = r
    seed[v[3 + md:3 + md + ed] - 1] = True

order = np.argsort(el.ravel(), kind="stable")
noe, eoe = el.ravel()[order], order // 3
ptr = np.searchsorted(noe, np.arange(nn + 1))
dist = np.full(nn, -1, np.int16)
dist[seed] = 0
front = np.flatnonzero(seed)
for d in range(1, MAXRING + 1):
    if front.size == 0:
        break
    elems = np.unique(np.concatenate([eoe[ptr[n]:ptr[n + 1]] for n in front]))
    cand = np.unique(el[elems].ravel())
    new = cand[dist[cand] < 0]
    dist[new] = d
    front = new

ice = np.abs(R).max(axis=0) > 0.01 * np.abs(R).max()
rings = list(range(MAXRING + 1))

fig = plt.figure(figsize=(11.6, 4.4), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.32])
ax = fig.add_subplot(gs[0, 0])

for tag, lab, col, mk in LEGS:
    T, _, _ = rd(tag)
    D = np.abs(T - R).mean(axis=0)
    y = [D[(dist == d) & ice].mean() for d in rings]
    far = (dist < 0) & ice
    ratio = D[np.isin(dist, (0, 1)) & ice].mean() / D[far].mean()
    ax.plot(rings, y, marker=mk, ms=4, color=col, label=f"{lab}   [{ratio:.2f}]")
    ax.plot([MAXRING + 1.2], [D[far].mean()], marker=mk, ms=5, color=col, mfc="white")
ax.set_yscale("log")
ax.set_xticks(rings + [MAXRING + 1.2])
ax.set_xticklabels([str(d) for d in rings] + ["interior"])
ax.set_xlabel("elements away from a sub-domain boundary")
ax.set_ylabel(f"mean |$\\Delta$ {VAR}| over 60 days   [m]")
ax.set_title("Is the difference organised by the decomposition?\n"
             "[  ] = edge/interior ratio; 1 means no imprint", fontsize=9)
ax.grid(alpha=.3, which="both")
ax.legend(fontsize=7.4, loc="lower left")

# ---- the map, so the ratio can be seen and not just believed
T, _, _ = rd("lag4")
r_ = 90.0 - lat
x, y_ = r_ * np.sin(np.deg2rad(lon)), -r_ * np.cos(np.deg2rad(lon))
keep = (lat[el] > 55).all(axis=1)
tri = mtri.Triangulation(x, y_, el[keep])
D = (T - R)[-1]
lim = float(np.percentile(np.abs(D), 99.5))
ax2 = fig.add_subplot(gs[0, 1])
h = ax2.tripcolor(tri, D, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud",
                  rasterized=True)
oe = owner[el[keep]]
for t in el[keep][~((oe[:, 0] == oe[:, 1]) & (oe[:, 1] == oe[:, 2]))]:
    ax2.plot(x[list(t) + [t[0]]], y_[list(t) + [t[0]]], color="k", lw=0.3, alpha=.7,
             rasterized=True)
ax2.set_xlim(-30, 30); ax2.set_ylim(-30, 30); ax2.set_aspect("equal")
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title("delayed exchange $K$=4, day 60: difference in ice thickness,\n"
              "with the 16 sub-domain boundaries drawn on top", fontsize=9)
cb = fig.colorbar(h, ax=ax2, shrink=.85); cb.set_label("$\\Delta$ ice thickness  [m]", fontsize=8)
fig.savefig(os.path.join(OUT, "fig4_artefact.pdf"), dpi=200)
print("fig4 written")
