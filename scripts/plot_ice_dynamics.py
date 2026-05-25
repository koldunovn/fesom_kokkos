#!/usr/bin/env python3
"""Ice dynamics diagnosis: is the C ice actually drifting, or only thermo?
March 1959, Arctic (North Polar Stereo). Rows:
 (1) m_ice  : C, Fortran, diff
 (2) ice drift speed + ice velocity vectors : C, Fortran
 (3) surface-ocean speed + OCEAN velocity vectors : C, Fortran  (does ice follow ocean?)

Headline metric: ice/ocean speed ratio (C~0.37 vs Fortran~1.1 in central Arctic).

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python plot_ice_dynamics.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import netCDF4 as nc

C = pathlib.Path("/work/ab0995/a270088/port/core2_864_2yr_dt1200")
F = pathlib.Path("/scratch/a/a270088/fortran_pp_2yr")
DIAG = F/"fesom.mesh.diag.nc"; SNAP="/work/ab0995/a270088/port/dt1800_snap/snap_001340.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_dt1200_vsPP"); OUT.mkdir(parents=True,exist_ok=True)
MON=2; YR=1959; FILL=1e20  # MON=2 -> March

lon=np.asarray(nc.Dataset(DIAG)["lon"][:]); lat=np.asarray(nc.Dataset(DIAG)["lat"][:])
en=np.asarray(nc.Dataset(SNAP)["elem_nodes"][:]); en=en-1 if en.min()==1 else en
lonE=lon[en].mean(1); latE=lat[en].mean(1)

def clean(a): return np.where(np.abs(a)>FILL,np.nan,a)
def nodevar(d,var,suf):  # 2D node field, month MON
    return clean(np.asarray(nc.Dataset(d/f"{var}.fesom.{YR}{suf}.nc")[var][MON]))
def icevec(d,suf):
    return clean(np.asarray(nc.Dataset(d/f"uice.fesom.{YR}{suf}.nc")["uice"][MON])), \
           clean(np.asarray(nc.Dataset(d/f"vice.fesom.{YR}{suf}.nc")["vice"][MON]))
def ocevec(d,suf):  # surface ocean (elements)
    return clean(np.asarray(nc.Dataset(d/f"u.fesom.{YR}{suf}.nc")["u"][MON,0,:])), \
           clean(np.asarray(nc.Dataset(d/f"v.fesom.{YR}{suf}.nc")["v"][MON,0,:]))

sC="​.monthly".replace("​",""); sF=""
mC=nodevar(C,"m_ice",sC); mF=nodevar(F,"m_ice",sF)
uiC,viC=icevec(C,sC); uiF,viF=icevec(F,sF)
uoC,voC=ocevec(C,sC); uoF,voF=ocevec(F,sF)
spiC=np.hypot(uiC,viC); spiF=np.hypot(uiF,viF)
spoC=np.hypot(uoC,voC); spoF=np.hypot(uoF,voF)

PC=ccrs.PlateCarree(); NPS=ccrs.NorthPolarStereo(central_longitude=0)
def setup(ax):
    ax.set_extent([-180,180,66,90],PC); ax.add_feature(cfeature.LAND,fc="0.8",zorder=3)
    ax.coastlines(lw=.3,zorder=4); ax.gridlines(lw=.2,color="0.5")
def scat(ax,lo,la,val,vmin,vmax,cmap):
    m=np.isfinite(val)&(la>64)
    return ax.scatter(lo[m],la[m],c=val[m],s=2,vmin=vmin,vmax=vmax,cmap=cmap,transform=PC,zorder=2,rasterized=True)
def quiv(ax,lo,la,u,v,step,color,scale):
    m=np.isfinite(u)&np.isfinite(v)&(la>66); idx=np.where(m)[0][::step]
    ax.quiver(lo[idx],la[idx],u[idx],v[idx],transform=PC,color=color,scale=scale,
              width=.003,zorder=5,regrid_shape=20)

fig=plt.figure(figsize=(13,17))
def panel(i,proj=NPS): ax=fig.add_subplot(3,3,i,projection=proj); setup(ax); return ax

# Row 1: thickness
for i,(val,ttl,cm,vM) in enumerate([(mC,"m_ice C",  "Blues",4),(mF,"m_ice Fortran","Blues",4)]):
    ax=panel(i+1); im=scat(ax,lon,lat,val,0,vM,cm); ax.set_title(ttl); plt.colorbar(im,ax=ax,shrink=.6,label="m")
ax=panel(3); im=scat(ax,lon,lat,mC-mF,-1.5,1.5,"RdBu_r"); ax.set_title("Δ m_ice (C−F)"); plt.colorbar(im,ax=ax,shrink=.6,label="m")

# Row 2: ice speed + ice vectors
for i,(sp,u,v,ttl) in enumerate([(spiC,uiC,viC,"C"),(spiF,uiF,viF,"Fortran")]):
    ax=panel(i+4); im=scat(ax,lon,lat,sp,0,0.12,"viridis"); quiv(ax,lon,lat,u,v,40,"white",2.0)
    ax.set_title(f"ICE drift {ttl} (mean>70N: C0.011 F0.037)" if i==0 else f"ICE drift {ttl}")
    plt.colorbar(im,ax=ax,shrink=.6,label="m/s")
ax=panel(6); im=scat(ax,lon,lat,spiC-spiF,-0.06,0.06,"RdBu_r"); ax.set_title("Δ ice speed (C−F)"); plt.colorbar(im,ax=ax,shrink=.6,label="m/s")

# Row 3: surface ocean speed + OCEAN vectors (elements)
for i,(sp,u,v,ttl) in enumerate([(spoC,uoC,voC,"C"),(spoF,uoF,voF,"Fortran")]):
    ax=panel(i+7); im=scat(ax,lonE,latE,sp,0,0.12,"viridis"); quiv(ax,lonE,latE,u,v,40,"white",2.0)
    ax.set_title(f"surface OCEAN {ttl}"); plt.colorbar(im,ax=ax,shrink=.6,label="m/s")
ax=panel(9)
# overlay: C ice vectors (red) vs C ocean vectors (blue) -> does ice follow ocean?
setup(ax); quiv(ax,lonE,latE,uoC,voC,40,"tab:blue",2.0); quiv(ax,lon,lat,uiC,viC,40,"tab:red",2.0)
ax.set_title("C: ice(red) vs ocean(blue) vectors")

fig.suptitle(f"Arctic ice dynamics — March {YR}: C(dt1200) vs Fortran(dt1800), both PP\n"
             "ice/ocean speed ratio: C=0.37  Fortran=1.11  → C ice not picking up momentum",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(OUT/"ice_dynamics.png",dpi=120,bbox_inches="tight"); print("wrote",OUT/"ice_dynamics.png")
