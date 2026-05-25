#!/usr/bin/env python3
"""Residual ice diagnosis (after the 0.5 fix): why still ~2x slow + ~40% over-ridge?
Discriminator: ice speed binned by m_ice (C=ice_fix_2yr vs Fortran PP).
 - thin ice (strength->0, free drift): tests wind/ocean FORCING fidelity
 - thick ice: tests the internal-stress / over-ridge throttle
Plus: thickness histogram (>70N) and 24-month Arctic mean/max m_ice evolution.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python ice_residual_diag.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

C = pathlib.Path("/work/ab0995/a270088/port/ice_fix_2yr")
F = pathlib.Path("/scratch/a/a270088/fortran_pp_2yr")
DIAG = F/"fesom.mesh.diag.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_dt1200_vsPP")
FILL=1e20
lat=np.asarray(nc.Dataset(DIAG)["lat"][:]); area=np.asarray(nc.Dataset(DIAG)["nod_area"][0,:])
def clean(a): return np.where(np.abs(a)>FILL,np.nan,a)
def fld(d,var,suf,mon):
    return clean(np.asarray(nc.Dataset(d/f"{var}.fesom.{{}}{suf}.nc".format(YR))[var][mon]))

# ---- (1) ice speed binned by m_ice, March 1959 ----
YR=1959; MON=2; arc=lat>70
def load(d,suf):
    u=clean(np.asarray(nc.Dataset(d/f"uice.fesom.{YR}{suf}.nc")["uice"][MON]))
    v=clean(np.asarray(nc.Dataset(d/f"vice.fesom.{YR}{suf}.nc")["vice"][MON]))
    m=clean(np.asarray(nc.Dataset(d/f"m_ice.fesom.{YR}{suf}.nc")["m_ice"][MON]))
    a=clean(np.asarray(nc.Dataset(d/f"a_ice.fesom.{YR}{suf}.nc")["a_ice"][MON]))
    return np.hypot(u,v),m,a
spC,mC,aC=load(C,".monthly"); spF,mF,aF=load(F,"")
bins=[0.1,0.5,1.0,1.5,2.0,3.0,4.0,6.0,12.0]
print(f"=== March {YR} >70N: mean ice speed (m/s) binned by m_ice ===")
print(f"  {'m_ice bin':14s} {'C speed':>9s} {'F speed':>9s} {'C/F':>6s}  {'n_C':>7s} {'n_F':>7s}")
for lo,hi in zip(bins[:-1],bins[1:]):
    selC=arc&(mC>=lo)&(mC<hi)&np.isfinite(spC); selF=arc&(mF>=lo)&(mF<hi)&np.isfinite(spF)
    sc=np.nanmean(spC[selC]) if selC.sum() else np.nan
    sf=np.nanmean(spF[selF]) if selF.sum() else np.nan
    print(f"  {lo:4.1f}-{hi:<8.1f} {sc:9.4f} {sf:9.4f} {sc/sf:6.2f}  {selC.sum():7d} {selF.sum():7d}")

# ---- (2) thickness histogram >70N (area-weighted) ----
fig,ax=plt.subplots(1,3,figsize=(17,5))
hb=np.linspace(0,10,41)
for sp,m,a,nm,col in [(spC,mC,aC,"C (fixed)","tab:red"),(spF,mF,aF,"Fortran PP","k")]:
    sel=arc&np.isfinite(m)&(m>0.05)
    w=area[sel]; h,_=np.histogram(m[sel],bins=hb,weights=w); h=h/h.sum()
    ax[0].plot(0.5*(hb[:-1]+hb[1:]),h,color=col,label=nm,lw=1.8)
ax[0].set_title(f"m_ice distribution >70N, Mar {YR} (area-wtd)"); ax[0].set_xlabel("m_ice [m]"); ax[0].set_ylabel("frac area"); ax[0].legend(); ax[0].grid(alpha=.3)
# speed vs thickness curve
bc=[0.5*(lo+hi) for lo,hi in zip(bins[:-1],bins[1:])]
scv=[np.nanmean(spC[arc&(mC>=lo)&(mC<hi)]) for lo,hi in zip(bins[:-1],bins[1:])]
sfv=[np.nanmean(spF[arc&(mF>=lo)&(mF<hi)]) for lo,hi in zip(bins[:-1],bins[1:])]
ax[1].plot(bc,scv,'-s',color="tab:red",label="C (fixed)"); ax[1].plot(bc,sfv,'-o',color="k",label="Fortran PP")
ax[1].set_title("ice speed vs thickness >70N"); ax[1].set_xlabel("m_ice [m]"); ax[1].set_ylabel("speed [m/s]"); ax[1].legend(); ax[1].grid(alpha=.3)

# ---- (3) 24-month Arctic mean & max m_ice evolution ----
def series(d,suf):
    mean=[]; mx=[]
    for yr in (1958,1959):
        ds=nc.Dataset(d/f"m_ice.fesom.{yr}{suf}.nc")
        for mo in range(12):
            m=clean(np.asarray(ds["m_ice"][mo])); s=arc&np.isfinite(m)
            mean.append(np.nansum(m[s]*area[s])/np.nansum(area[s])); mx.append(np.nanmax(m[s]))
        ds.close()
    return np.array(mean),np.array(mx)
mnC,mxC=series(C,".monthly"); mnF,mxF=series(F,"")
t=np.arange(24)
ax[2].plot(t,mxC,'-s',color="tab:red",label="C max"); ax[2].plot(t,mxF,'-o',color="k",label="F max")
ax[2].plot(t,mnC,'--s',color="tab:orange",label="C mean"); ax[2].plot(t,mnF,'--o',color="0.5",label="F mean")
ax[2].set_title("Arctic >70N m_ice: 24-month evolution"); ax[2].set_xlabel("month (1958-59)"); ax[2].set_ylabel("m_ice [m]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT/"ice_residual_diag.png",dpi=120,bbox_inches="tight")
print("\nwrote",OUT/"ice_residual_diag.png")
print(f"\n24-mo Arctic max m_ice: C {mxC[0]:.1f}->{mxC[-1]:.1f}  F {mxF[0]:.1f}->{mxF[-1]:.1f}")
print(f"24-mo Arctic mean m_ice: C {mnC[0]:.2f}->{mnC[-1]:.2f}  F {mnF[0]:.2f}->{mnF[-1]:.2f}")
