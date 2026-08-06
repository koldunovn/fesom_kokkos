#!/usr/bin/env python3
"""Report figure: the delayed exchange puts the domain decomposition into the ice; the exact wide
halo does not.

Two maps of the same quantity, on the same colour scale, with the sub-domain boundaries drawn on
top. This is the whole of the argument -- one panel traces the boundaries and the other does not --
and it replaces the ring-profile panel and the tables the earlier draft carried. A scheme that is
not usable does not need its statistics tabulated; it needs one picture showing why.

Ratios quoted in the caption come from scripts/m9_partition_artefact.py on the same runs.
"""
import os
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

C = "/work/ab0995/a270088/port2/m9/clim"
MESH = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc"
OUT = os.path.dirname(os.path.abspath(__file__))
NPES, VAR = 16, "m_ice"

PANELS = [("lag4",  "delayed exchange, $K$=4"),
          ("wide8", "wide halo, exact, $K$=8")]


def rd(tag):
    with nc.Dataset(f"{C}/art_farc_g16_{tag}/{VAR}.fesom.1958.daily.nc") as f:
        return (np.asarray(f.variables[VAR][:]), np.asarray(f.variables["lon"][:]),
                np.asarray(f.variables["lat"][:]))


R, lon, lat = rd("standard")
nn = R.shape[1]
el = np.loadtxt(f"{MESH}/elem2d.out", skiprows=1, dtype=np.int64) - 1

owner = np.full(nn, -1, np.int32)
for r in range(NPES):
    v = np.fromstring(open(f"{MESH}/dist_{NPES}/my_list{r:05d}.out").read().replace("\n", " "),
                      sep=" ", dtype=np.int64)
    owner[v[3:3 + int(v[1])] - 1] = r

rr = 90.0 - lat
x, y = rr * np.sin(np.deg2rad(lon)), -rr * np.cos(np.deg2rad(lon))
keep = (lat[el] > 55).all(axis=1)
tri = mtri.Triangulation(x, y, el[keep])
oe = owner[el[keep]]
cut = el[keep][~((oe[:, 0] == oe[:, 1]) & (oe[:, 1] == oe[:, 2]))]

D = {tag: (rd(tag)[0] - R)[-1] for tag, _ in PANELS}
lim = float(np.percentile(np.abs(D["lag4"]), 99.5))   # common scale, set by the larger field

fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.9), constrained_layout=True)
for ax, (tag, lab) in zip(axs, PANELS):
    h = ax.tripcolor(tri, D[tag], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud",
                     rasterized=True)
    for t in cut:
        ax.plot(x[list(t) + [t[0]]], y[list(t) + [t[0]]], color="k", lw=0.3, alpha=.7,
                rasterized=True)
    ax.set_xlim(-32, 32); ax.set_ylim(-32, 32); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lab, fontsize=10)
cb = fig.colorbar(h, ax=axs, shrink=.8)
cb.set_label("difference in ice thickness after 60 days  [m]", fontsize=9)
fig.savefig(os.path.join(OUT, "fig4_artefact.pdf"), dpi=200)
print("fig4 written")
