#!/usr/bin/env python3
"""Depth-resolved validation: C port (dt=1200, PP) vs Fortran (dt=1800, PP).
Annual-mean 1959. (1) global temp/salt profiles vs depth + bias, (2) temp/salt
difference maps at key depths, (3) zonal-mean lat-depth sections.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python plot_depth_diag.py
"""
import pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import netCDF4 as nc
import nereus as nr

C = pathlib.Path("/work/ab0995/a270088/port/core2_864_2yr_dt1200")
F = pathlib.Path("/scratch/a/a270088/fortran_pp_2yr")
DIAG = F/"fesom.mesh.diag.nc"; SNAP="/work/ab0995/a270088/port/dt1800_snap/snap_001340.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_dt1200_vsPP"); OUT.mkdir(parents=True,exist_ok=True)
YR=1959; FILL=1e29

mesh=nr.fesom.load_mesh(DIAG); lon=np.asarray(mesh["lon"].values); lat=np.asarray(mesh["lat"].values)
Z=-np.asarray(nc.Dataset(C/f"temp.fesom.{YR}.monthly.nc")["Z"][:])   # +down [m]
elem_area=np.asarray(nc.Dataset(DIAG)["elem_area"][:])
en=np.asarray(nc.Dataset(SNAP)["elem_nodes"][:]); en=en-1 if en.min()==1 else en
narea=np.zeros(len(lon))
for k in range(3): np.add.at(narea, en[:,k], elem_area/3.0)
NL=len(Z); print(f"{len(lon)} nodes, {NL} layers, depths {Z[0]:.0f}..{Z[-1]:.0f} m")

def annual3d(d, v, var):  # annual-mean 3D [nz,node]; NaN where below bathymetry
    ds=nc.Dataset(d/(f"{var}.fesom.{YR}.monthly.nc" if d==C else f"{var}.fesom.{YR}.nc"))
    acc=np.zeros((NL,len(lon))); cnt=np.zeros((NL,len(lon)))
    for m in range(12):
        a=np.asarray(ds[var][m,:,:]); a=np.where(np.abs(a)>FILL,np.nan,a); a=np.where(a==0,np.nan,a)
        ok=np.isfinite(a); acc[ok]+=a[ok]; cnt[ok]+=1.0
    ds.close()
    return np.where(cnt>0, acc/np.where(cnt>0,cnt,1.0), np.nan)   # below-bathy → NaN, not 0
print("reading annual-mean T,S (C & Fortran)...")
tC=annual3d(C,"temp","temp"); tF=annual3d(F,"temp","temp")
sC=annual3d(C,"salt","salt"); sF=annual3d(F,"salt","salt")

# ---- (1) global profiles + bias ----
def gprof(a):
    out=np.full(NL,np.nan)
    for k in range(NL):
        x=a[k]; w=narea; ok=np.isfinite(x)
        if ok.sum(): out[k]=np.sum(x[ok]*w[ok])/np.sum(w[ok])
    return out
tCp,tFp,sCp,sFp=gprof(tC),gprof(tF),gprof(sC),gprof(sF)
fig,ax=plt.subplots(1,3,figsize=(15,7))
ax[0].plot(tFp,Z,'-o',ms=3,c='k',label='Fortran PP'); ax[0].plot(tCp,Z,'-s',ms=3,c='tab:red',label='C port')
ax[0].set_title('global-mean temperature'); ax[0].set_xlabel('°C')
ax[1].plot(sFp,Z,'-o',ms=3,c='k',label='Fortran PP'); ax[1].plot(sCp,Z,'-s',ms=3,c='tab:red',label='C port')
ax[1].set_title('global-mean salinity'); ax[1].set_xlabel('PSU')
ax[2].plot(tCp-tFp,Z,'-s',ms=3,c='tab:red',label='ΔT (C−F)'); ax[2].plot((sCp-sFp)*10,Z,'-^',ms=3,c='tab:blue',label='ΔS×10')
ax[2].set_title('bias C−F vs depth'); ax[2].set_xlabel('ΔT [°C], ΔS×10 [PSU]'); ax[2].axvline(0,c='gray',lw=.7)
for a in ax: a.invert_yaxis(); a.set_ylabel('depth [m]'); a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle(f'Global T/S profiles, annual {YR} — C(dt1200,PP) vs Fortran(dt1800,PP)')
fig.tight_layout(); fig.savefig(OUT/"depth_profiles.png",dpi=120,bbox_inches="tight"); print("  wrote depth_profiles.png")

# ---- (2) diff maps at key depths ----
levs=[(11,"100m"),(20,"535m"),(24,"975m")]
fig=plt.figure(figsize=(19,10))
for i,(L,lab) in enumerate(levs):
    for j,(dd,vn,dv,u) in enumerate([(tC[L]-tF[L],"T",1.5,"°C"),(sC[L]-sF[L],"S",0.4,"PSU")]):
        ax=fig.add_subplot(2,3,j*3+i+1,projection=ccrs.Robinson()); ax.set_global()
        nr.plot(dd,lon,lat,projection="rob",cmap="RdBu_r",vmin=-dv,vmax=dv,ax=ax,colorbar=True,
                colorbar_label=f"Δ{vn} [{u}]",title=f"Δ{vn} (C−F) @ {lab} ann {YR}")
fig.savefig(OUT/"depth_diffmaps.png",dpi=110,bbox_inches="tight"); print("  wrote depth_diffmaps.png")

# ---- (3) zonal-mean lat-depth sections ----
latb=np.arange(-80,90,2.0); latc=0.5*(latb[:-1]+latb[1:])
def zonal(a):
    out=np.full((NL,len(latc)),np.nan)
    for j in range(len(latc)):
        sel=(lat>=latb[j])&(lat<latb[j+1])
        if sel.sum():
            for k in range(NL):
                x=a[k,sel]; ok=np.isfinite(x)
                if ok.sum(): out[k,j]=np.nanmean(x[ok])
    return out
tCz,tFz,sCz,sFz=zonal(tC),zonal(tF),zonal(sC),zonal(sF)
# data-driven bounds for the upper-2000m section so structure isn't saturated
m2k = Z <= 2000
slo,shi = np.nanpercentile(sFz[m2k], [2,98])
tlo,thi = np.nanpercentile(tFz[m2k], [2,98])
print(f"  S section bounds {slo:.2f}..{shi:.2f} PSU; T {tlo:.1f}..{thi:.1f} C")
fig,ax=plt.subplots(2,2,figsize=(15,9))
for a,(d,ttl,cm,lo,hi,u) in zip(ax.flat,[
        (tFz,"Fortran T","RdYlBu_r",tlo,thi,"°C"),(tCz-tFz,"ΔT (C−F)","RdBu_r",-1.5,1.5,"°C"),
        (sFz,"Fortran S","viridis",slo,shi,"PSU"),(sCz-sFz,"ΔS (C−F)","RdBu_r",-0.3,0.3,"PSU")]):
    pc=a.pcolormesh(latc,Z,d,cmap=cm,vmin=lo,vmax=hi,shading="auto")
    a.invert_yaxis(); a.set_title(ttl); a.set_xlabel("latitude"); a.set_ylabel("depth [m]")
    a.set_ylim(2000,0); plt.colorbar(pc,ax=a,label=u)
fig.suptitle(f"Zonal-mean sections, annual {YR} — C(dt1200,PP) vs Fortran(dt1800,PP)")
fig.tight_layout(); fig.savefig(OUT/"zonal_sections.png",dpi=120,bbox_inches="tight"); print("  wrote zonal_sections.png")
print("done →",OUT)
