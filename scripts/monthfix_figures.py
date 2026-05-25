#!/usr/bin/env python3
"""Figures for the month-skip fix: global ΔSSS before/after/change maps + the
Red Sea / Baltic seasonal-cycle line plots (Fortran vs C-before vs C-after).
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import nereus as nr

F="/scratch/a/a270088/fortran_pp_2yr"; B="/work/ab0995/a270088/port/reflocal_1yr"; A="/work/ab0995/a270088/port/monthfix_1yr"
DIAG=f"{F}/fesom.mesh.diag.nc"; FILL=1e29
OUT=pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
mesh=nr.fesom.load_mesh(DIAG); lon=np.asarray(mesh["lon"].values); lat=np.asarray(mesh["lat"].values)
def mon(p,v,s): a=np.asarray(nc.Dataset(f"{p}/{v}.fesom.1958{s}.nc")[v][:]); return np.where(np.abs(a)<FILL,a,np.nan)
sF=mon(F,"sss",""); sB=mon(B,"sss",".monthly"); sA=mon(A,"sss",".monthly")
mF=np.nanmean(sF,0); mB=np.nanmean(sB,0); mA=np.nanmean(sA,0)
wet=np.isfinite(mF)&np.isfinite(mB)&np.isfinite(mA)
dB=np.where(wet,mB-mF,np.nan); dA=np.where(wet,mA-mF,np.nan); dC=np.where(wet,mA-mB,np.nan)

fig=plt.figure(figsize=(18,9))
for j,(d,ttl,vl) in enumerate([(dB,"ΔSSS before (reflocal)−F",0.5),(dA,"ΔSSS after (monthfix)−F",0.5),(dC,"ΔSSS monthfix−reflocal",0.5)]):
    ax=fig.add_subplot(2,3,j+1,projection=ccrs.Robinson()); ax.set_global()
    nr.plot(d,lon,lat,projection="rob",cmap="RdBu_r",vmin=-vl,vmax=vl,ax=ax,colorbar=True,colorbar_label="ΔSSS [PSU]",title=ttl)
# seasonal cycle line plots
def nearest(la,lo): return int(np.argmin((lat-la)**2+(lon-lo)**2))
m=np.arange(1,13)
for k,(name,la,lo) in enumerate([("Red Sea (16.9N,42.7E)",16.89,42.69),("Kattegat (57.4N,12.1E)",57.44,12.11),("Skagerrak (58.2N,11.7E)",58.20,11.68)]):
    i=nearest(la,lo); ax=fig.add_subplot(2,3,4+k)
    ax.plot(m,sF[:,i],'k-o',lw=2,ms=4,label="Fortran")
    ax.plot(m,sB[:,i],'r--s',lw=1.5,ms=3,label="C before (month skip)")
    ax.plot(m,sA[:,i],'b-^',lw=1.5,ms=3,label="C after (fixed)")
    ax.set_title(name,fontsize=10); ax.set_xlabel("month"); ax.set_ylabel("SSS [PSU]"); ax.grid(alpha=0.3)
    if k==0: ax.legend(fontsize=8)
fig.suptitle("SSS month-skip fix: global ΔSSS maps (top) + marginal-sea seasonal cycles (bottom), 1958",fontsize=13)
fig.tight_layout(); fp=OUT/"monthfix_sss.png"; fig.savefig(fp,dpi=115,bbox_inches="tight")
print("wrote",fp)
