#!/usr/bin/env python3
"""Arctic maps and an animation of what the DELAYED HALO EXCHANGE does to the ice field.

Companion to m9_partition_artefact.py, which answers "is the difference organised by the domain
decomposition" with a number. This one answers "what does it look like", which is the question a
sea-ice modeller actually asks, and it is not the same question: a difference can be
statistically organised and still be invisible in the field, or the reverse.

Three panels per frame:
  1. the ice field itself from the delayed-exchange run  -- does it look physical?
  2. the signed difference against the reference run     -- where does it live?
  3. the same difference with the SUB-DOMAIN BOUNDARIES drawn on top -- does it trace them?

Panel 3 is the point. If the delayed exchange imprints the partition, panel 2 shows a polygonal
web and panel 3 shows it coinciding with the drawn boundaries.

Both runs must be the same binary and rank count, differing only in the knob.

usage:
  m9_artefact_maps.py --ref .../art_farc_g16_standard --test .../art_farc_g16_lag4 \
     --mesh /pool/.../farc --npes 16 --var m_ice --outdir ... [--anim]
"""
import argparse, os, sys
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import animation

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--npes", type=int, required=True)
ap.add_argument("--var", default="m_ice")
ap.add_argument("--pattern", default="{var}.fesom.{year}.daily.nc")
ap.add_argument("--year", default="1958")
ap.add_argument("--outdir", default=".")
ap.add_argument("--latmin", type=float, default=60.0)
ap.add_argument("--frame", type=int, default=-1, help="record for the still figure (-1 = last)")
ap.add_argument("--anim", action="store_true", help="also write an mp4 over all records")
ap.add_argument("--fps", type=int, default=6)
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
UNITS = {"m_ice": "m", "a_ice": "fraction", "uice": "m/s", "vice": "m/s"}


def read(d):
    with nc.Dataset(os.path.join(d, args.pattern.format(var=args.var, year=args.year))) as f:
        return (np.asarray(f.variables[args.var][:]),
                np.asarray(f.variables["lon"][:]), np.asarray(f.variables["lat"][:]))


R, lon, lat = read(args.ref)
T, _, _ = read(args.test)
if R.shape != T.shape:
    sys.exit(f"shape mismatch {R.shape} vs {T.shape}")
nt, nn = R.shape

# ---- triangulation, clipped to the Arctic and to triangles that do not straddle the date line
el = np.loadtxt(os.path.join(args.mesh, "elem2d.out"), skiprows=1, dtype=np.int64) - 1
# polar stereographic about the north pole -- no projection library needed and no distortion
# of the thing we are looking at (straight sub-domain boundaries stay straight enough)
r = 90.0 - lat
x = r * np.sin(np.deg2rad(lon))
y = -r * np.cos(np.deg2rad(lon))
keep = (lat[el] > args.latmin - 5).all(axis=1)
tri = mtri.Triangulation(x, y, el[keep])

# ---- sub-domain boundaries: an element whose three vertices are not all owned by one rank
owner = np.full(nn, -1, dtype=np.int32)
for rk in range(args.npes):
    p = os.path.join(args.mesh, f"dist_{args.npes}", f"my_list{rk:05d}.out")
    v = np.fromstring(open(p).read().replace("\n", " "), sep=" ", dtype=np.int64)
    mydim = int(v[1])
    owner[v[3:3 + mydim] - 1] = rk
oe = owner[el[keep]]
cut = ~((oe[:, 0] == oe[:, 1]) & (oe[:, 1] == oe[:, 2]))
cut_tris = el[keep][cut]
print(f"{cut.sum()} elements straddle a sub-domain boundary of dist_{args.npes}")

TESTNAME = os.path.basename(args.test.rstrip("/")).replace("art_farc_g16_", "")

D = T - R
lim = float(np.percentile(np.abs(D[np.isfinite(D)]), 99.5)) or 1e-12
fmax = float(np.nanpercentile(T, 99.5))
ext = 90.0 - args.latmin


def draw(fig, axs, k):
    for a in axs:
        a.clear(); a.set_xlim(-ext, ext); a.set_ylim(-ext, ext)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    h0 = axs[0].tripcolor(tri, T[k], cmap="Blues_r", vmin=0, vmax=fmax, shading="gouraud")
    axs[0].set_title(f"{args.var} — {TESTNAME}", fontsize=9)
    h1 = axs[1].tripcolor(tri, D[k], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
    axs[1].set_title(f"difference vs exchanging every sub-cycle", fontsize=9)
    axs[2].tripcolor(tri, D[k], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
    for t in cut_tris:                       # the decomposition, drawn on top
        axs[2].plot(x[list(t) + [t[0]]], y[list(t) + [t[0]]], color="k", lw=0.35, alpha=.65)
    axs[2].set_title(f"same, with the {args.npes} sub-domain boundaries drawn", fontsize=9)
    fig.suptitle(f"fArc, day {k + 1} of {nt}   |   "
                 f"if the delayed exchange imprints the partition, panel 3's colour follows the "
                 f"black lines", fontsize=9.5)
    return h0, h1


fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.9), constrained_layout=True)
k0 = args.frame if args.frame >= 0 else nt - 1
h0, h1 = draw(fig, axs, k0)
cb0 = fig.colorbar(h0, ax=axs[0], shrink=.8); cb0.set_label(UNITS.get(args.var, ""), fontsize=8)
cb1 = fig.colorbar(h1, ax=axs[2], shrink=.8); cb1.set_label(f"$\\Delta$ {UNITS.get(args.var,'')}", fontsize=8)
tag = os.path.basename(args.test.rstrip("/"))
still = os.path.join(args.outdir, f"artefact_map_{tag}_{args.var}_day{k0+1}.png")
fig.savefig(still, dpi=150)
print("wrote", still)

if args.anim:
    figA, axsA = plt.subplots(1, 3, figsize=(13.2, 4.9), constrained_layout=True)
    draw(figA, axsA, 0)
    anim = animation.FuncAnimation(figA, lambda k: draw(figA, axsA, k), frames=nt, blit=False)
    mp4 = os.path.join(args.outdir, f"artefact_anim_{tag}_{args.var}.mp4")
    anim.save(mp4, writer=animation.FFMpegWriter(fps=args.fps, bitrate=3200))
    print("wrote", mp4)
