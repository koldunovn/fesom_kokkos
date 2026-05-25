#!/usr/bin/env python3
"""Where does the cell-velocity 2Δx checkerboard live? (Sergey's Q1: Arctic only
or everywhere?) Per-element local roughness = |u - mean(u of K nearest cells)|,
from the C dt=1800 step-5200 snapshot (strong checkerboard, pre-eruption).
Global Robinson + N-polar + S-polar.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python plot_checkerboard_where.py
"""
import pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import netCDF4 as nc
from scipy.spatial import cKDTree

SNAP = "/work/ab0995/a270088/port/dt1800_snap/snap_005200.nc"
OUT  = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/compare_plots_dt1800")

ds = nc.Dataset(SNAP)
lon = np.asarray(ds["lon"][:]); lat = np.asarray(ds["lat"][:])
en  = np.asarray(ds["elem_nodes"][:]); en = en-1 if en.min()==1 else en
ue  = np.asarray(ds["u"][0,0,:]); ve = np.asarray(ds["v"][0,0,:])
ds.close()
# element centroids
clon = lon[en].mean(1); clat = lat[en].mean(1)
# xyz for KDTree (handles poles/dateline)
r = np.pi/180.0
x = np.cos(clat*r)*np.cos(clon*r); y = np.cos(clat*r)*np.sin(clon*r); z = np.sin(clat*r)
tree = cKDTree(np.c_[x,y,z])
K = 8
d, idx = tree.query(np.c_[x,y,z], k=K)
# local roughness: deviation from mean of K nearest (the 2dx amplitude)
um = ue[idx[:,1:]].mean(1); vm = ve[idx[:,1:]].mean(1)
rough = np.hypot(ue-um, ve-vm)
print(f"roughness: median={np.median(rough):.4f} max={rough.max():.3f}  "
      f"frac>0.05: {(rough>0.05).mean():.3f}")
for nm,thr in [("Arctic >70N",clat>70),("Antarctic <-60S",clat<-60),
               ("Tropics |lat|<30",np.abs(clat)<30)]:
    print(f"  mean roughness {nm:18s}: {rough[thr].mean():.4f}")

def scat(ax, proj):
    sc = ax.scatter(clon, clat, c=rough, s=2, cmap="hot_r", vmin=0, vmax=0.15,
                    transform=ccrs.PlateCarree(), rasterized=True)
    ax.coastlines(linewidth=0.4); return sc

fig = plt.figure(figsize=(20,6))
a1 = fig.add_subplot(1,3,1, projection=ccrs.Robinson())
a1.set_global(); sc=scat(a1,None); a1.set_title("cell-velocity 2Δx roughness — global")
a2 = fig.add_subplot(1,3,2, projection=ccrs.NorthPolarStereo()); a2.set_extent([-180,180,60,90],ccrs.PlateCarree())
scat(a2,None); a2.set_title("Arctic")
a3 = fig.add_subplot(1,3,3, projection=ccrs.SouthPolarStereo()); a3.set_extent([-180,180,-90,-50],ccrs.PlateCarree())
scat(a3,None); a3.set_title("Antarctic")
fig.colorbar(sc, ax=[a1,a2,a3], shrink=0.6, label="|u - mean(8 nearest)| [m/s]")
fig.suptitle("C dt=1800 step 5200: where is the cell-velocity 2Δx checkerboard?", fontsize=13)
fp = OUT/"checkerboard_where.png"; fig.savefig(fp, dpi=120, bbox_inches="tight")
print("wrote", fp)
