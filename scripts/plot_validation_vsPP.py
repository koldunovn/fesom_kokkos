#!/usr/bin/env python3
"""2-year validation: C port (dt=1200, PP) vs Fortran reference (dt=1800, PP).
Spatial maps (nereus) + global time series. Both at CORE2, JRA55 1958-1959.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python plot_validation_dt1200.py
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
DIAG = F / "fesom.mesh.diag.nc"
SNAP = "/work/ab0995/a270088/port/dt1800_snap/snap_001340.nc"   # for elem_nodes
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_dt1200_vsPP")
OUT.mkdir(parents=True, exist_ok=True)
FILL = 1e29

mesh = nr.fesom.load_mesh(DIAG)
lon = np.asarray(mesh["lon"].values); lat = np.asarray(mesh["lat"].values)
# node areas from elem_area (mesh.diag) + elem_nodes (snapshot): node_area=Σ elem_area/3
elem_area = np.asarray(nc.Dataset(DIAG)["elem_area"][:])
en = np.asarray(nc.Dataset(SNAP)["elem_nodes"][:]); en = en-1 if en.min()==1 else en
node_area = np.zeros(len(lon))
for k in range(3):
    np.add.at(node_area, en[:, k], elem_area/3.0)
print(f"mesh {len(lon)} nodes; total ocean area {node_area.sum()/1e12:.1f} 1e6 km^2")

def cvar(v, m, surf=True):
    a = np.asarray(nc.Dataset(C/f"{v}.fesom.{YR}.monthly.nc")[v][m, ...])
    a = a[0] if (surf and a.ndim == 2) else a
    return np.where(np.abs(a) > FILL, np.nan, a)
def fvar(v, m, surf=True):
    a = np.asarray(nc.Dataset(F/f"{v}.fesom.{YR}.nc")[v][m, ...])
    a = a[0] if (surf and a.ndim == 2) else a
    return np.where(np.abs(a) > FILL, np.nan, a)
def cmask(a, mz=True):
    a = np.array(a, float)
    if mz: a[a == 0] = np.nan
    return a

def triptych(cdat, fdat, ttl, cmap, vmin, vmax, dvlim, unit, proj, fname, mz=True):
    cdat = cmask(cdat, mz); fdat = cmask(fdat, mz)
    fig = plt.figure(figsize=(20, 6))
    pj = {"rob": ccrs.Robinson(), "np": ccrs.NorthPolarStereo(), "sp": ccrs.SouthPolarStereo()}[proj]
    ext = {"np":[-180,180,55,90], "sp":[-180,180,-90,-50]}.get(proj)
    for i,(t,d,cm,lo,hi) in enumerate([("C port",cdat,cmap,vmin,vmax),
            ("Fortran",fdat,cmap,vmin,vmax),("C − Fortran",cdat-fdat,"RdBu_r",-dvlim,dvlim)]):
        ax = fig.add_subplot(1,3,i+1,projection=pj)
        if ext: ax.set_extent(ext, ccrs.PlateCarree())
        else: ax.set_global()
        nr.plot(d, lon, lat, projection=proj, cmap=cm, vmin=lo, vmax=hi, ax=ax,
                colorbar=True, colorbar_label=unit, title=f"{ttl} — {t}")
    fig.savefig(OUT/fname, dpi=110, bbox_inches="tight"); plt.close(fig); print("  wrote", fname)

# ---- spatial maps (annual mean year 2 = 1959) ----
YR = 1959
def annual(fn, v, surf=True):
    return np.nanmean([fn(v, m, surf) for m in range(12)], axis=0)
triptych(annual(cvar,"sst"), annual(fvar,"sst"), "SST annual 1959", "RdYlBu_r", -2,30,1.0,"degC","rob","map_sst_ann.png")
triptych(annual(cvar,"sss"), annual(fvar,"sss"), "SSS annual 1959", "viridis", 28,38,0.5,"PSU","rob","map_sss_ann.png")
triptych(annual(cvar,"ssh"), annual(fvar,"ssh"), "SSH annual 1959", "RdBu_r", -2,1.5,0.15,"m","rob","map_ssh_ann.png")
triptych(cvar("a_ice",2), fvar("a_ice",2), "a_ice Mar 1959", "Blues",0,1,0.3,"frac","np","map_aice_mar.png",mz=False)
triptych(cvar("m_ice",2), fvar("m_ice",2), "m_ice Mar 1959", "Blues",0,4,1.0,"m","np","map_mice_mar.png",mz=False)
triptych(cvar("a_ice",8), fvar("a_ice",8), "a_ice Sep 1959 (SH max)","Blues",0,1,0.3,"frac","sp","map_aice_sep_sh.png",mz=False)

# ---- global time series over 24 months ----
def gmean(fn, v):
    out=[]
    for m in range(12):
        a = fn(v, m); w = node_area.copy(); ok = np.isfinite(a)
        out.append(np.sum(a[ok]*w[ok])/np.sum(w[ok]))
    return out
def icearea(fn, hemi):
    out=[]
    for m in range(12):
        a = fn("a_ice", m); sel = (lat>0) if hemi=="N" else (lat<0)
        a = np.where(np.isfinite(a), a, 0.0)
        out.append(np.sum(a[sel]*node_area[sel])/1e12)   # 1e6 km^2
    return out
months=[]; cSST=[];fSST=[];cSSS=[];fSSS=[];cNH=[];fNH=[];cSH=[];fSH=[]
for YR in (1958,1959):
    cSST+=gmean(cvar,"sst"); fSST+=gmean(fvar,"sst")
    cSSS+=gmean(cvar,"sss"); fSSS+=gmean(fvar,"sss")
    cNH+=icearea(cvar,"N"); fNH+=icearea(fvar,"N")
    cSH+=icearea(cvar,"S"); fSH+=icearea(fvar,"S")
t=np.arange(1,25)
fig,ax=plt.subplots(2,2,figsize=(15,9))
for a,(c,f,ttl,u) in zip(ax.flat,[(cSST,fSST,"global-mean SST","degC"),
        (cSSS,fSSS,"global-mean SSS","PSU"),(cNH,fNH,"NH sea-ice area","1e6 km2"),
        (cSH,fSH,"SH sea-ice area","1e6 km2")]):
    a.plot(t,f,'-o',ms=3,label="Fortran (dt1800,PP)",color="k")
    a.plot(t,c,'-s',ms=3,label="C port (dt1200,PP)",color="tab:red")
    a.set_title(ttl); a.set_ylabel(u); a.set_xlabel("month (1958-1959)"); a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle("2-year validation: C port (dt=1200, PP) vs Fortran (dt=1800, PP)",fontsize=13)
fig.tight_layout(); fig.savefig(OUT/"timeseries.png",dpi=120,bbox_inches="tight")
print("  wrote timeseries.png")
print("done →", OUT)
