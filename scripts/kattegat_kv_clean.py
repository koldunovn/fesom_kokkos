#!/usr/bin/env python3
"""Clean Kattegat Kv comparison after the output-grid fix: both C and Fortran now
write Kv/bvfreq on nz (48 interfaces) with the zbar coordinate, so index k aligns
directly (no offset). Confirms the earlier finding without any alignment ambiguity."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
F="/scratch/a/a270088/fortran_pp_2yr"; A="/work/ab0995/a270088/port/kvfix_1yr"
DIAG=f"{F}/fesom.mesh.diag.nc"; FILL=1e29
d=nc.Dataset(DIAG); lat=np.asarray(d["lat"][:]); lon=np.asarray(d["lon"][:]); lon180=np.where(lon>180,lon-360,lon)
nn=len(lat)
def L(path,var,suf):
    ds=nc.Dataset(f"{path}/{var}.fesom.1958{suf}.nc"); a=np.asarray(ds[var][:]); a=np.where(np.abs(a)<FILL,a,np.nan)
    a=np.nanmean(a,0)
    if a.shape[0]!=nn: a=a.T
    # vertical coordinate
    zc=None
    for zn in ("nz","nz_1"):
        if zn in ds.variables: zc=np.asarray(ds[zn][:]); break
    return a, (zc, ds.dimensions[ [k for k in ds.dimensions if k.startswith('nz')][0] ].size if any(k.startswith('nz') for k in ds.dimensions) else -1)
KvF,(zF,nF)=L(F,"Kv",""); KvC,(zC,nC)=L(A,"Kv",".monthly")
N2F,_=L(F,"N2",""); N2C,_=L(A,"bvfreq",".monthly")
print(f"Fortran Kv levels={nF} (zname has {None if zF is None else len(zF)})   C Kv levels={nC}")
print(f"  => grids identical? {nF==nC}\n")
def nr_(la,lo): return int(np.argmin((lat-la)**2+(lon180-lo)**2))
zint = zF if zF is not None else np.arange(KvF.shape[1])
for nm,la,lo in [("Kattegat",57.44,12.11),("Skagerrak",58.20,11.68),("Oresund",55.96,12.56)]:
    i=nr_(la,lo)
    print(f"=== {nm} ({lat[i]:.2f},{lon180[i]:.2f}) — Kv [m2/s] and N2 [1/s2], direct index ===")
    print(f"  {'z(m)':>7} {'Kv_F':>10} {'Kv_C':>10} {'ratio':>6}   {'N2_F':>9} {'N2_C':>9}")
    nval=min(KvF.shape[1],KvC.shape[1],6)
    for k in range(nval):
        zz=zint[k] if k<len(zint) else k
        kf,kc=KvF[i,k],KvC[i,k]
        if np.isfinite(kf) or np.isfinite(kc):
            r=kc/kf if (np.isfinite(kf) and kf>0) else np.nan
            print(f"  {zz:7.1f} {kf:10.2e} {kc:10.2e} {r:6.1f}   {N2F[i,k]:9.2e} {N2C[i,k]:9.2e}")
    print()
