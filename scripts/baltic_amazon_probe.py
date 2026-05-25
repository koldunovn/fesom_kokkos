#!/usr/bin/env python3
"""Localize the remaining SSS residual (monthfix run) at the Baltic straits and the
Amazon, with regional maps (C, Fortran, diff) and monthly evolution. Looking for
whether the difference sits at the strait (exchange/geometry) or in the basin/plume."""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
F="/scratch/a/a270088/fortran_pp_2yr"; A="/work/ab0995/a270088/port/monthfix_1yr"
DIAG=f"{F}/fesom.mesh.diag.nc"; FILL=1e29
OUT=pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
ds=nc.Dataset(DIAG); lat=np.asarray(ds["lat"][:]); lon=np.asarray(ds["lon"][:])
def mon(p,v,s): a=np.asarray(nc.Dataset(f"{p}/{v}.fesom.1958{s}.nc")[v][:]); return np.where(np.abs(a)<FILL,a,np.nan)
sF=mon(F,"sss",""); sA=mon(A,"sss",".monthly"); mF=np.nanmean(sF,0); mA=np.nanmean(sA,0)
wet=np.isfinite(mF)&np.isfinite(mA); d=np.where(wet,mA-mF,np.nan)
lon180=np.where(lon>180,lon-360,lon)

print("=== top 20 residual nodes (monthfix − Fortran, annual SSS) ===")
print(f"{'lat':>7}{'lon180':>8}{'dSSS':>7}{'C':>7}{'F':>7}")
for i in np.argsort(-np.abs(np.nan_to_num(d)))[:20]:
    print(f"{lat[i]:7.2f}{lon180[i]:8.2f}{d[i]:7.2f}{mA[i]:7.1f}{mF[i]:7.1f}")

# regional scatter maps (node-based, since unstructured)
def regional(ax, m, field, vmin, vmax, cmap, title, box):
    lo0,lo1,la0,la1=box
    sel=m&(lon180>=lo0)&(lon180<=lo1)&(lat>=la0)&(lat<=la1)
    sc=ax.scatter(lon180[sel],lat[sel],c=field[sel],s=14,vmin=vmin,vmax=vmax,cmap=cmap,
                  transform=ccrs.PlateCarree(),edgecolors='none')
    ax.add_feature(cfeature.LAND,facecolor='0.85',zorder=0); ax.coastlines(lw=0.4)
    ax.set_extent([lo0,lo1,la0,la1],ccrs.PlateCarree()); ax.set_title(title,fontsize=9)
    plt.colorbar(sc,ax=ax,shrink=0.8)

for region,box,svmin,svmax in [("Baltic",(4,30,52,66),5,35),("Amazon",(-65,-35,-5,18),28,37)]:
    fig=plt.figure(figsize=(16,4.5))
    for k,(field,vmn,vmx,cmap,ttl) in enumerate([
        (mA,svmin,svmax,"viridis",f"{region} C-monthfix SSS"),
        (mF,svmin,svmax,"viridis",f"{region} Fortran SSS"),
        (d,-1.0,1.0,"RdBu_r",f"{region} ΔSSS (C−F)")]):
        ax=fig.add_subplot(1,3,k+1,projection=ccrs.PlateCarree()); regional(ax,wet,field,vmn,vmx,cmap,ttl,box)
    fig.tight_layout(); fp=OUT/f"probe_{region.lower()}.png"; fig.savefig(fp,dpi=120,bbox_inches="tight"); plt.close(fig)
    print("wrote",fp)

# Amazon mouth monthly evolution
def nearest(la,lo): return int(np.argmin((lat-la)**2+(lon180-lo)**2))
for nm,la,lo in [("Amazon mouth",0.5,-49.5),("Amazon plume NW",4.0,-51.0),("Baltic proper",58.5,20.0)]:
    i=nearest(la,lo)
    print(f"\n{nm} ({lat[i]:.2f},{lon180[i]:.2f}) monthly SSS:")
    print("  F:    "+" ".join(f"{sF[m,i]:5.1f}" for m in range(12)))
    print("  C:    "+" ".join(f"{sA[m,i]:5.1f}" for m in range(12)))
