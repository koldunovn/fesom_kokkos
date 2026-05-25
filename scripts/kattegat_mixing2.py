#!/usr/bin/env python3
"""Aligned Kattegat mixing comparison. C outputs Kv/bvfreq on nz_1=47, Fortran on nz=48.
Align by matching N^2 profiles, then compare Kv at true depth and compute Richardson
number Ri=N^2/shear^2 from u,v to test PP formula vs shear (currents)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
F="/scratch/a/a270088/fortran_pp_2yr"; A="/work/ab0995/a270088/port/monthfix_1yr"
DIAG=f"{F}/fesom.mesh.diag.nc"; FILL=1e29
d=nc.Dataset(DIAG); lat=np.asarray(d["lat"][:]); lon=np.asarray(d["lon"][:]); lon180=np.where(lon>180,lon-360,lon)
zint=np.asarray(d["nz"][:]); zlay=np.asarray(d["nz1"][:]); nn=len(lat)
def L(path,var,suf):
    a=np.asarray(nc.Dataset(f"{path}/{var}.fesom.1958{suf}.nc")[var][:]); a=np.where(np.abs(a)<FILL,a,np.nan)
    a=np.nanmean(a,0)
    if a.shape[0]!=nn: a=a.T
    return a
KvF=L(F,"Kv",""); N2F=L(F,"N2","")            # (nn,48) interfaces
KvC=L(A,"Kv",".monthly"); N2C=L(A,"bvfreq",".monthly")  # (nn,47)
uF=L(F,"u",""); vF=L(F,"v","")                # (nn,47) layers
uC=L(A,"u",".monthly"); vC=L(A,"v",".monthly")
def nr_(la,lo): return int(np.argmin((lat-la)**2+(lon180-lo)**2))
print(f"Fortran Kv/N2 levels={KvF.shape[1]} (interfaces nz)  C levels={KvC.shape[1]} (nz_1)")
print(f"zint(nz)[:6]={np.round(zint[:6],1)}  zlay(nz1)[:6]={np.round(zlay[:6],1)}\n")
for nm,la,lo in [("Kattegat",57.44,12.11),("Skagerrak",58.20,11.68)]:
    i=nr_(la,lo)
    print(f"=== {nm} ({lat[i]:.2f},{lon180[i]:.2f}) ===")
    print(" N2 profiles (find the alignment offset):")
    print(f"   Fortran nz : "+" ".join(f"{N2F[i,k]:8.1e}" for k in range(6)))
    print(f"   C    nz_1  : "+" ".join(f"{N2C[i,k]:8.1e}" for k in range(6)))
    # Assume C[k] = Fortran interface[k+1] (C drops the surface interface)
    print(" Kv at matched depth (Fortran interface k+1  vs  C nz_1 k):")
    print(f"   {'depth':>7} {'Kv_F':>9} {'Kv_C':>9} {'ratio':>6}")
    for k in range(5):
        zf=zint[k+1] if k+1<len(zint) else -1
        kf=KvF[i,k+1]; kc=KvC[i,k]
        if np.isfinite(kf) or np.isfinite(kc):
            r=kc/kf if (np.isfinite(kf) and kf>0) else np.nan
            print(f"   {zf:7.1f} {kf:9.2e} {kc:9.2e} {r:6.1f}")
    # shear & Richardson number between adjacent layers (u,v on nz1 layers)
    print(" shear^2=(du/dz)^2+(dv/dz)^2, Ri=N2/shear^2 at interfaces between layers:")
    print(f"   {'iface_z':>8} {'sh2_F':>9} {'sh2_C':>9} {'Ri_F':>8} {'Ri_C':>8}")
    for k in range(4):  # interface between layer k and k+1
        dz=zlay[k+1]-zlay[k]
        sh2F=((uF[i,k+1]-uF[i,k])**2+(vF[i,k+1]-vF[i,k])**2)/dz**2
        sh2C=((uC[i,k+1]-uC[i,k])**2+(vC[i,k+1]-vC[i,k])**2)/dz**2
        # N2 at this interface: Fortran interface k+1, C nz_1 k
        n2F=N2F[i,k+1]; n2C=N2C[i,k]
        RiF=n2F/sh2F if sh2F>0 else np.inf; RiC=n2C/sh2C if sh2C>0 else np.inf
        print(f"   {zint[k+1]:8.1f} {sh2F:9.2e} {sh2C:9.2e} {RiF:8.1f} {RiC:8.1f}")
    # surface speed
    print(f"   surface |vel|: F={np.hypot(uF[i,0],vF[i,0]):.3f}  C={np.hypot(uC[i,0],vC[i,0]):.3f} m/s")
    print()
