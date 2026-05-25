#!/usr/bin/env python3
"""A/B: ice_strength 0.5 restoration. March 1958, Arctic ice/ocean speed ratio.
BUGGY = core2_864_2yr_dt1200 (ice_strength 2x too stiff)
FIXED = ice_fix_test         (0.5 restored)
Both: same PHC IC, 864 ranks, dt=1200, full ice. Fortran(PP) shown for the target.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python ab_ice_fix.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature

BUG = pathlib.Path("/work/ab0995/a270088/port/core2_864_2yr_dt1200")
FIX = pathlib.Path("/work/ab0995/a270088/port/ice_fix_2yr")
FOR = pathlib.Path("/scratch/a/a270088/fortran_pp_2yr")  # 1959 ref (March)
DIAG= FOR/"fesom.mesh.diag.nc"; SNAP="/work/ab0995/a270088/port/dt1800_snap/snap_001340.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_dt1200_vsPP")
MON=2; YR=1959; FILL=1e20

lon=np.asarray(nc.Dataset(DIAG)["lon"][:]); lat=np.asarray(nc.Dataset(DIAG)["lat"][:])
en=np.asarray(nc.Dataset(SNAP)["elem_nodes"][:]); en=en-1 if en.min()==1 else en
latE=lat[en].mean(1); lonE=lon[en].mean(1)
def clean(a): return np.where(np.abs(a)>FILL,np.nan,a)
def ice(d,suf):
    u=clean(np.asarray(nc.Dataset(d/f"uice.fesom.{YR}{suf}.nc")["uice"][MON]))
    v=clean(np.asarray(nc.Dataset(d/f"vice.fesom.{YR}{suf}.nc")["vice"][MON])); return u,v,np.hypot(u,v)
def oce(d,suf):
    u=clean(np.asarray(nc.Dataset(d/f"u.fesom.{YR}{suf}.nc")["u"][MON,0,:]))
    v=clean(np.asarray(nc.Dataset(d/f"v.fesom.{YR}{suf}.nc")["v"][MON,0,:])); return np.hypot(u,v)
def mice(d,suf): return clean(np.asarray(nc.Dataset(d/f"m_ice.fesom.{YR}{suf}.nc")["m_ice"][MON]))

arcN=lat>70; arcE=latE>70
print(f"=== March {YR}, Arctic >70N: ice / ocean speed ratio ===")
rows=[]
for nm,d,suf in [("BUGGY (no 0.5)",BUG,".monthly"),("FIXED (0.5)",FIX,".monthly"),("Fortran PP",FOR,"")]:
    try:
        _,_,si=ice(d,suf); so=oce(d,suf); mi=mice(d,suf)
        sib=np.nanmean(si[arcN]); sob=np.nanmean(so[arcE]); mib=np.nanmean(mi[arcN])
        print(f"  {nm:16s}: ice={sib:.4f}  ocean={sob:.4f}  ratio={sib/sob:.2f}  m_ice(mean>70N)={mib:.2f}  m_ice(max)={np.nanmax(mi):.2f}")
        rows.append((nm,d,suf,si))
    except Exception as e:
        print(f"  {nm:16s}: NOT READY ({e})")

# before/after/target thickness + ice drift figure (NPS)
PC=ccrs.PlateCarree(); NPS=ccrs.NorthPolarStereo()
def setup(ax): ax.set_extent([-180,180,66,90],PC); ax.add_feature(cfeature.LAND,fc="0.8",zorder=3); ax.coastlines(lw=.3,zorder=4)
fig=plt.figure(figsize=(15,8))
panels=[("BUGGY (no 0.5)",BUG,".monthly"),("FIXED (0.5)",FIX,".monthly"),("Fortran PP",FOR,"")]
for j,(nm,d,suf) in enumerate(panels):
    try:
        u,v,sp=ice(d,suf); mi=mice(d,suf)
        ax=fig.add_subplot(2,3,j+1,projection=NPS); setup(ax)
        m=np.isfinite(mi)&(lat>64); im=ax.scatter(lon[m],lat[m],c=mi[m],s=2,vmin=0,vmax=4,cmap="Blues",transform=PC,zorder=2)
        ax.set_title(f"m_ice  {nm}"); plt.colorbar(im,ax=ax,shrink=.6,label="m")
        ax=fig.add_subplot(2,3,j+4,projection=NPS); setup(ax)
        m=np.isfinite(sp)&(lat>64); im=ax.scatter(lon[m],lat[m],c=sp[m],s=2,vmin=0,vmax=0.12,cmap="viridis",transform=PC,zorder=2)
        mm=np.isfinite(u)&np.isfinite(v)&(lat>66); idx=np.where(mm)[0][::40]
        ax.quiver(lon[idx],lat[idx],u[idx],v[idx],transform=PC,color="white",scale=2.0,width=.003,zorder=5,regrid_shape=20)
        ax.set_title(f"ice drift  {nm}"); plt.colorbar(im,ax=ax,shrink=.6,label="m/s")
    except Exception as e:
        print("skip panel",nm,e)
fig.suptitle(f"ice_strength 0.5 restoration — March {YR} (MATURITY) — buggy vs fixed vs Fortran)",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(OUT/"ice_fix_ab_1959.png",dpi=120,bbox_inches="tight")
print("wrote",OUT/"ice_fix_ab_1959.png")
