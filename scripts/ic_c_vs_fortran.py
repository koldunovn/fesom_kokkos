#!/usr/bin/env python3
"""Compare the interpolated PHC IC (surface salinity) between the C and the Fortran,
BOTH at 864 ranks with the same dist_864 partition. If they match at the Kattegat (and
everywhere), the C's extrap_nod3D is faithful and the residual is dynamical. If they
differ at the extrapolated Kattegat, the C extrap deviates from the Fortran at the same
partition (a port subtlety in extrap_nod3D)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
CIC="/work/ab0995/a270088/port/ic_864/snap_000000.nc"          # C IC @ 864 (step 0)
FIC="/work/ab0995/a270088/port/fort_ic864/salt.fesom.1958.nc"  # Fortran IC @ 864 (step 1)
DIAG="/scratch/a/a270088/fortran_pp_2yr/fesom.mesh.diag.nc"; FILL=1e29
d=nc.Dataset(DIAG); lat=np.asarray(d["lat"][:]); lon=np.asarray(d["lon"][:]); lon180=np.where(lon>180,lon-360,lon); nn=len(lat)
def surf(path,var):
    ds=nc.Dataset(path); a=np.asarray(ds[var][:])
    while a.ndim>1: a=a[0]            # peel time then level -> surface, node-vector
    return np.where(np.abs(a)<FILL,a,np.nan)
SC=surf(CIC,"S"); SF=surf(FIC,"salt")
print(f"C IC nodes={SC.size}  Fortran IC nodes={SF.size}")
if SC.size!=SF.size: raise SystemExit("node count mismatch — check global ordering")
d_=SC-SF
print(f"GLOBAL surface ΔS (C-F, both @864): max|Δ|={np.nanmax(np.abs(d_)):.4f}  "
      f"mean|Δ|={np.nanmean(np.abs(d_)):.5f}  nodes>0.01={np.sum(np.abs(d_)>0.01)}  nodes>0.1={np.sum(np.abs(d_)>0.1)}")
def nr_(la,lo): return int(np.argmin((lat-la)**2+(lon180-lo)**2))
for nm,la,lo,note in [("Kattegat",57.44,12.11,"PHC NaN->extrap"),("Skagerrak",58.20,11.68,"extrap"),
                      ("Oresund",55.96,12.56,"extrap"),("W.Baltic",54.1,11.6,"extrap"),
                      ("Baltic interior",58.50,20.08,"real PHC (control)"),("Open Atl",50.0,-30.0,"real PHC (control)")]:
    i=nr_(la,lo)
    print(f"  {nm:16s} ({lat[i]:.2f},{lon180[i]:.2f}) [{note}]: C={SC[i]:.3f}  F={SF[i]:.3f}  Δ={SC[i]-SF[i]:+.3f}")
print("\ntop 10 C-vs-Fortran IC differences (both @864):")
for i in np.argsort(-np.abs(np.nan_to_num(d_)))[:10]:
    print(f"   ({lat[i]:6.2f},{lon180[i]:7.2f})  C={SC[i]:6.2f}  F={SF[i]:6.2f}  Δ={SC[i]-SF[i]:+.2f}")
