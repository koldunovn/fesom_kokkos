#!/usr/bin/env python3
"""Instrument the Kattegat halocline mixing: compare C vs Fortran vertical diffusivity
(Kv) and stratification (N^2) at the Danish-straits nodes. Distinguish PP-shear mixing
(Kv modest, Ri-dependent) from convective adjustment (Kv -> instabmix_kv = 0.1)."""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
F="/scratch/a/a270088/fortran_pp_2yr"; A="/work/ab0995/a270088/port/monthfix_1yr"
DIAG=f"{F}/fesom.mesh.diag.nc"; FILL=1e29; OUT=pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
d=nc.Dataset(DIAG); lat=np.asarray(d["lat"][:]); lon=np.asarray(d["lon"][:]); lon180=np.where(lon>180,lon-360,lon)
zlev=np.asarray(d["nz"][:]); zlay=np.asarray(d["nz1"][:]); nn=len(lat)
def load(path,var,suf,reduce_mean=True):
    a=np.asarray(nc.Dataset(f"{path}/{var}.fesom.1958{suf}.nc")[var][:]); a=np.where(np.abs(a)<FILL,a,np.nan)
    return a  # (12, ?, ?)
def orient(a):  # -> (12, node, level)
    if a.shape[1]==nn: return a
    if a.shape[2]==nn: return np.transpose(a,(0,2,1))
    raise SystemExit(f"bad shape {a.shape}")
KvF=orient(load(F,"Kv","")); KvC=orient(load(A,"Kv",".monthly"))
N2F=orient(load(F,"N2","")); N2C=orient(load(A,"bvfreq",".monthly"))
def nr_(la,lo): return int(np.argmin((lat-la)**2+(lon180-lo)**2))
nodes=[("Kattegat",57.44,12.11),("Skagerrak",58.20,11.68),("Oresund",55.96,12.56)]
print(f"Kv shape {KvC.shape}  N2 shape {N2C.shape}  nz={len(zlev)} nz1={len(zlay)}")
print("instabmix_kv (convective) = 0.1 ; PP background K_ver = 1e-5\n")
for nm,la,lo in nodes:
    i=nr_(la,lo)
    kf=np.nanmean(KvF[:,i,:],0); kc=np.nanmean(KvC[:,i,:],0)
    nf=np.nanmean(N2F[:,i,:],0); ncv=np.nanmean(N2C[:,i,:],0)
    nl=int(np.sum(np.isfinite(kf)))
    print(f"=== {nm} ({lat[i]:.2f},{lon180[i]:.2f}) — annual-mean Kv [m2/s] and N2 [1/s2] by level ===")
    print(f"  {'z(m)':>6} {'Kv_F':>9} {'Kv_C':>9} {'Kv_C/F':>7}   {'N2_F':>9} {'N2_C':>9}")
    for k in range(min(nl+1,KvC.shape[2])):
        if np.isfinite(kf[k]) or np.isfinite(kc[k]):
            zz=zlev[k] if k<len(zlev) else -1
            r = kc[k]/kf[k] if (np.isfinite(kf[k]) and kf[k]>0) else np.nan
            print(f"  {zz:6.1f} {kf[k]:9.2e} {kc[k]:9.2e} {r:7.1f}   {nf[k]:9.2e} {ncv[k]:9.2e}")
    print()

# seasonal cycle of Kv at the halocline interface (~level 3-4, z~15-25m) at Kattegat
i=nr_(57.44,12.11)
# pick the interface level nearest 20 m
kint=int(np.argmin(np.abs(zlev-20.0)))
print(f"=== Kattegat Kv seasonal cycle at z={zlev[kint]:.0f} m (halocline interface) ===")
print("  mon:  "+" ".join(f"{m+1:8d}" for m in range(12)))
print("  Kv_F: "+" ".join(f"{KvF[m,i,kint]:8.1e}" for m in range(12)))
print("  Kv_C: "+" ".join(f"{KvC[m,i,kint]:8.1e}" for m in range(12)))

# figure: annual Kv + N2 profiles
fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True)
for c,(nm,la,lo) in enumerate(nodes):
    i=nr_(la,lo)
    kf=np.nanmean(KvF[:,i,:],0); kc=np.nanmean(KvC[:,i,:],0); nf=np.nanmean(N2F[:,i,:],0); ncv=np.nanmean(N2C[:,i,:],0)
    m=np.isfinite(kf)|np.isfinite(kc); zz=zlev[:KvC.shape[2]]
    ax=axes[0,c]; ax.semilogx(kf[m],zz[m],'k-o',label="Fortran"); ax.semilogx(kc[m],zz[m],'b-^',label="C")
    ax.axvline(0.1,color='r',ls=':',lw=1,label="instabmix 0.1"); ax.invert_yaxis(); ax.set_title(f"{nm}: Kv",fontsize=9); ax.grid(alpha=0.3,which='both')
    if c==0: ax.set_ylabel("depth [m]"); ax.legend(fontsize=7)
    ax2=axes[1,c]; ax2.plot(nf[m],zz[m],'k-o'); ax2.plot(ncv[m],zz[m],'b-^'); ax2.invert_yaxis(); ax2.set_title(f"{nm}: N2",fontsize=9); ax2.grid(alpha=0.3)
    if c==0: ax2.set_ylabel("depth [m]")
fig.suptitle("Kattegat halocline mixing: Kv (top) and N2 (bottom), C vs Fortran annual mean",fontsize=12)
fig.tight_layout(); fp=OUT/"kattegat_mixing.png"; fig.savefig(fp,dpi=120,bbox_inches="tight"); print("\nwrote",fp)
