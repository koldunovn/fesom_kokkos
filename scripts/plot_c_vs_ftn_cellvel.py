#!/usr/bin/env python3
"""C vs Fortran+PP central-Arctic surface CELL velocity (March monthly mean),
per-triangle, same scheme (PP) / dt (1800) / mesh. Tests whether the C carries
more grid-scale (cell-to-cell) velocity structure than Fortran at the same
config — the suspected reason its marginal biharmonic tips over at dt=1800.

Run:
  PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python plot_c_vs_ftn_cellvel.py
"""
import pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import netCDF4 as nc
import cartopy.crs as ccrs

SNAP = "/work/ab0995/a270088/port/dt1800_snap/snap_001340.nc"   # mesh source
CU   = "/work/ab0995/a270088/port/core2_864_2yr_dt1800_monthly/u.fesom.1958.monthly.nc"
FU   = "/scratch/a/a270088/fortran_pp_dt1800/u.fesom.1958.nc"
OUT  = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/compare_plots_dt1800")
MON  = 2   # March
proj = ccrs.NorthPolarStereo(); geo = ccrs.PlateCarree()

ds = nc.Dataset(SNAP)
lon = np.asarray(ds["lon"][:]); lat = np.asarray(ds["lat"][:])
en  = np.asarray(ds["elem_nodes"][:]);  en = en-1 if en.min()==1 else en
ds.close()
cu = np.asarray(nc.Dataset(CU)["u"][MON, 0, :])
fu = np.asarray(nc.Dataset(FU)["u"][MON, 0, :])
print(f"C  u: [{cu.min():.3f},{cu.max():.3f}]  Fortran u: [{fu.min():.3f},{fu.max():.3f}]")

xy = proj.transform_points(geo, lon, lat); x, y = xy[:,0], xy[:,1]
cx = x[en].mean(1); cy = y[en].mean(1); clat = lat[en].mean(1)
# frame on the blow-up region (high lat, lon 0-120E)
foc = np.where((clat>82) & (lon[en].mean(1)>20) & (lon[en].mean(1)<100))[0]
x0, y0 = cx[foc].mean(), cy[foc].mean()
HALF = 350e3
tri = mtri.Triangulation(x, y, triangles=en)
inwin = (np.abs(cx-x0)<HALF) & (np.abs(cy-y0)<HALF)
tri.set_mask(~inwin)

fig, ax = plt.subplots(1, 3, figsize=(20, 7))
for a,(dat,ttl,cm,lo,hi) in zip(ax, [
        (cu, "C port (PP)",     "RdBu_r", -0.3, 0.3),
        (fu, "Fortran (PP)",    "RdBu_r", -0.3, 0.3),
        (cu-fu, "C - Fortran",  "RdBu_r", -0.2, 0.2)]):
    tp = a.tripcolor(tri, facecolors=dat, cmap=cm, vmin=lo, vmax=hi,
                     shading="flat", edgecolors="0.3", linewidth=0.15)
    a.set_xlim(x0-HALF,x0+HALF); a.set_ylim(y0-HALF,y0+HALF)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(ttl, fontsize=12)
    plt.colorbar(tp, ax=a, shrink=0.7, label="surface u [m/s] (per cell)")
fig.suptitle("Central-Arctic surface CELL velocity, March mean — C(PP) vs Fortran(PP), dt=1800",
             fontsize=13)
fig.tight_layout()
fp = OUT/"cvf_cellvel_mar.png"; fig.savefig(fp, dpi=130, bbox_inches="tight")
print("wrote", fp)
