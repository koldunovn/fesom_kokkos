#!/usr/bin/env python3
"""Compare the interpolated PHC initial condition (snap_000000.nc, step 0, before any
timestep) between two rank counts (864 vs 8). If T/S differ at the Kattegat (where PHC
is NaN -> extrapolated) but match in the Baltic interior (real PHC data), the
extrap_nod3D fill is rank-dependent — the seed of the Kattegat residual."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
A="/work/ab0995/a270088/port/ic_864/snap_000000.nc"
B="/work/ab0995/a270088/port/ic_8/snap_000000.nc"
FILL=1e29
da=nc.Dataset(A); db=nc.Dataset(B)
lat=np.asarray(da["lat"][:]); lon=np.asarray(da["lon"][:]); lon180=np.where(lon>180,lon-360,lon)
def f3(ds,v):
    a=np.asarray(ds[v][:])  # (time, nz_1, nod2) or (nz_1, nod2)
    a=a[0] if a.ndim==3 else a
    a=np.where(np.abs(a)<FILL,a,np.nan)
    return a  # (nz_1, nod2)
SA=f3(da,"S"); SB=f3(db,"S"); TA=f3(da,"T"); TB=f3(db,"T")
print(f"salt shape 864={SA.shape} 8={SB.shape}")
print(f"GLOBAL surface |ΔS|(864-8): max={np.nanmax(np.abs(SA[0]-SB[0])):.3f}  "
      f"mean={np.nanmean(np.abs(SA[0]-SB[0])):.4f}  nodes>0.1={np.sum(np.abs(SA[0]-SB[0])>0.1)}")
print(f"GLOBAL surface |ΔT|(864-8): max={np.nanmax(np.abs(TA[0]-TB[0])):.3f}  nodes>0.1={np.sum(np.abs(TA[0]-TB[0])>0.1)}\n")
def nr_(la,lo): return int(np.argmin((lat-la)**2+(lon180-lo)**2))
for nm,la,lo,note in [("Kattegat",57.44,12.11,"PHC NaN -> extrapolated"),
                      ("Skagerrak",58.20,11.68,"extrapolated"),
                      ("Oresund",55.96,12.56,"extrapolated"),
                      ("Baltic interior",58.50,20.08,"real PHC data (control)"),
                      ("Open Atlantic",50.0,-30.0,"real PHC data (control)")]:
    i=nr_(la,lo)
    print(f"{nm} ({lat[i]:.2f},{lon180[i]:.2f}) [{note}] surface IC:")
    print(f"   S: 864={SA[0,i]:.3f}  8={SB[0,i]:.3f}  Δ={SA[0,i]-SB[0,i]:+.3f}   "
          f"T: 864={TA[0,i]:.3f}  8={TB[0,i]:.3f}  Δ={TA[0,i]-TB[0,i]:+.3f}")
# top IC-difference nodes
d=np.abs(np.nan_to_num(SA[0]-SB[0]))
print("\ntop 10 surface-S IC differences (864 vs 8):")
for i in np.argsort(-d)[:10]:
    print(f"   ({lat[i]:6.2f},{lon180[i]:7.2f})  S_864={SA[0,i]:6.2f}  S_8={SB[0,i]:6.2f}  Δ={SA[0,i]-SB[0,i]:+.2f}")
