#!/usr/bin/env python3
"""monthly-read fix validation: did dropping the spurious +1 (which skipped February)
restore the seasonal restoring cycle in the marginal seas absent from PHC (Red Sea,
Baltic)? Compares Fortran vs C-reflocal (before, month-skip) vs C-monthfix (after).

Monthly evolution at the worst Red Sea / Baltic nodes + global/Baltic/Red Sea RMS.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
F="/scratch/a/a270088/fortran_pp_2yr"; B="/work/ab0995/a270088/port/reflocal_1yr"; A="/work/ab0995/a270088/port/monthfix_1yr"
DIAG=f"{F}/fesom.mesh.diag.nc"; FILL=1e29
lat=np.asarray(nc.Dataset(DIAG)["lat"][:]); lon=np.asarray(nc.Dataset(DIAG)["lon"][:])
ea=np.asarray(nc.Dataset(DIAG)["elem_area"][:])
fnv=np.asarray(nc.Dataset(DIAG)["face_nodes"][:]); fnv=fnv.T if fnv.shape[0]==3 else fnv; fnv=fnv-1 if fnv.min()==1 else fnv
na=np.zeros(len(lat));
for k in range(3): np.add.at(na,fnv[:,k],ea/3.0)
def mon(p,v,s): a=np.asarray(nc.Dataset(f"{p}/{v}.fesom.1958{s}.nc")[v][:]); return np.where(np.abs(a)<FILL,a,np.nan)
sF=mon(F,"sss",""); sB=mon(B,"sss",".monthly"); sA=mon(A,"sss",".monthly")
def nearest(la,lo): return int(np.argmin((lat-la)**2+(lon-lo)**2))
print("=== monthly SSS at marginal-sea nodes: Fortran / C-before / C-after ===")
for name,la,lo in [("Kattegat",57.44,12.11),("RedSea",16.89,42.69),("Skagerrak",58.20,11.68)]:
    i=nearest(la,lo)
    print(f"\n{name} ({lat[i]:.2f},{lon[i]:.2f})")
    print("  mon:   "+" ".join(f"{m+1:5d}" for m in range(12)))
    print("  F:     "+" ".join(f"{sF[m,i]:5.1f}" for m in range(12)))
    print("  before:"+" ".join(f"{sB[m,i]:5.1f}" for m in range(12)))
    print("  after: "+" ".join(f"{sA[m,i]:5.1f}" for m in range(12)))
mF=np.nanmean(sF,0); mB=np.nanmean(sB,0); mA=np.nanmean(sA,0)
wet=np.isfinite(mF)&np.isfinite(mB)&np.isfinite(mA)
def rms(d,m): return np.sqrt(np.sum(d[m]**2*na[m])/np.sum(na[m]))
print("\n=== annual-mean SSS RMS vs Fortran (before -> after) by region ===")
dB=np.where(wet,mB-mF,np.nan); dA=np.where(wet,mA-mF,np.nan)
for nm,m in [("GLOBAL",wet),("Arctic >60N",wet&(lat>60)),("Baltic/Kattegat 53-60N,5-15E",wet&(lat>53)&(lat<60)&(lon>5)&(lon<15)),
             ("Red Sea 12-30N,32-44E",wet&(lat>12)&(lat<30)&(lon>32)&(lon<44)),("Equator |lat|<10",wet&(np.abs(lat)<10))]:
    if m.sum()>5: print(f"  {nm:32s}: {rms(dB,m):.3f} -> {rms(dA,m):.3f}   (|max| {np.nanmax(np.abs(dB[m])):.2f} -> {np.nanmax(np.abs(dA[m])):.2f})")
print(f"\n  nodes |dSSS|>0.5: before {(wet&(np.abs(dB)>0.5)).sum()} -> after {(wet&(np.abs(dA)>0.5)).sum()}")
