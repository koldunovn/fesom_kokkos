#!/usr/bin/env python3
"""Full validation of the C port (both fixes: ice_strength 0.5 + wind vector_g2r)
vs Fortran PP, using nereus built-in diagnostics. 2yr dt=1200, CORE2.
 C = ice_fix_windrot_2yr (monthly), F = fortran_pp_2yr.

Outputs to docs/validation_windrot/:
 - stats table (corr/bias/rmse) for key 2D fields, annual 1959
 - timeseries.png : global SST/SSS (nr.surface_mean), NH/SH ice area & volume
                    (nr.ice_area_nh/sh, nr.ice_volume_nh/sh), 24 months
 - maps_surface.png : SST/SSS/SSH annual 1959 (C, F, diff)
 - maps_ice.png : a_ice/m_ice Mar & Sep (C, F, diff)
 - profiles.png : global T/S vs depth (per-level nr.surface_mean) + bias;
                  depth-band volume means (nr.volume_mean)

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python validation_windrot.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import nereus as nr

C=pathlib.Path("/work/ab0995/a270088/port/ice_fix_windrot_2yr")
F=pathlib.Path("/scratch/a/a270088/fortran_pp_2yr")
DIAG=F/"fesom.mesh.diag.nc"
OUT=pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_windrot"); OUT.mkdir(parents=True,exist_ok=True)
FILL=1e20
mesh=nr.fesom.load_mesh(DIAG)
area=np.asarray(mesh["area"].values); lat=np.asarray(mesh["lat"].values); lon=np.asarray(mesh["lon"].values)
nan3=np.asarray(mesh["nod_area_nans"].values)          # (nz, npoints) NaN below bathy
depth=np.asarray(mesh["depth"].values); thick=np.asarray(mesh["layer_thickness"].values)
NL=len(depth)
def clean(a): return np.where(np.abs(a)>FILL,np.nan,a)
def cpath(d,var,yr): return d/(f"{var}.fesom.{yr}.monthly.nc" if d==C else f"{var}.fesom.{yr}.nc")
def m2d(d,var,yr,mon): return clean(np.asarray(nc.Dataset(cpath(d,var,yr))[var][mon]))
def ann2d(d,var,yr):   # annual-mean surface/2D field
    a=clean(np.asarray(nc.Dataset(cpath(d,var,yr))[var][:])); return np.nanmean(a,axis=0)

# ============ stats table (annual 1959) ============
print("=== annual-1959 2D field stats: C(windrot) vs Fortran PP ===")
print(f"  {'field':6s} {'C mean':>9s} {'F mean':>9s} {'bias':>8s} {'rmse':>7s} {'corr':>6s}")
def stat(var):
    c=ann2d(C,var,1959); f=ann2d(F,var,1959); ok=np.isfinite(c)&np.isfinite(f)
    cb=np.nansum(c[ok]*area[ok])/np.nansum(area[ok]); fb=np.nansum(f[ok]*area[ok])/np.nansum(area[ok])
    d=c[ok]-f[ok]; rmse=np.sqrt(np.nansum(d*d*area[ok])/np.nansum(area[ok]))
    corr=np.corrcoef(c[ok],f[ok])[0,1]
    print(f"  {var:6s} {cb:9.4f} {fb:9.4f} {cb-fb:+8.4f} {rmse:7.4f} {corr:6.3f}")
for v in ["sst","sss","ssh","a_ice","m_ice"]: stat(v)

# ============ timeseries (24 months) via nereus ============
mC={}; mF={}
for v in ["sst","sss","a_ice","m_ice"]:
    mC[v]=[m2d(C,v,yr,mo) for yr in (1958,1959) for mo in range(12)]
    mF[v]=[m2d(F,v,yr,mo) for yr in (1958,1959) for mo in range(12)]
t=np.arange(24)
g_sstC=[nr.surface_mean(x,area,mask=np.isfinite(x)) for x in mC["sst"]]
g_sstF=[nr.surface_mean(x,area,mask=np.isfinite(x)) for x in mF["sst"]]
g_sssC=[nr.surface_mean(x,area,mask=np.isfinite(x)) for x in mC["sss"]]
g_sssF=[nr.surface_mean(x,area,mask=np.isfinite(x)) for x in mF["sss"]]
aN_C=[nr.ice_area_nh(x,area,lat)/1e12 for x in mC["a_ice"]]; aN_F=[nr.ice_area_nh(x,area,lat)/1e12 for x in mF["a_ice"]]
aS_C=[nr.ice_area_sh(x,area,lat)/1e12 for x in mC["a_ice"]]; aS_F=[nr.ice_area_sh(x,area,lat)/1e12 for x in mF["a_ice"]]
vN_C=[nr.ice_volume_nh(x,area,lat)/1e12 for x in mC["m_ice"]]; vN_F=[nr.ice_volume_nh(x,area,lat)/1e12 for x in mF["m_ice"]]
vS_C=[nr.ice_volume_sh(x,area,lat)/1e12 for x in mC["m_ice"]]; vS_F=[nr.ice_volume_sh(x,area,lat)/1e12 for x in mF["m_ice"]]
fig,ax=plt.subplots(2,3,figsize=(17,8))
def ts(a,c,f,ttl,ylab):
    a.plot(t,c,'-s',ms=3,c='tab:red',label='C (port)'); a.plot(t,f,'-o',ms=3,c='k',label='Fortran PP')
    a.set_title(ttl); a.set_ylabel(ylab); a.set_xlabel('month (1958-59)'); a.grid(alpha=.3); a.legend(fontsize=8)
ts(ax[0,0],g_sstC,g_sstF,"global SST","°C"); ts(ax[0,1],g_sssC,g_sssF,"global SSS","PSU")
ts(ax[0,2],aN_C,aN_F,"NH sea-ice area","10⁶ km²")
ts(ax[1,0],aS_C,aS_F,"SH sea-ice area","10⁶ km²")
ts(ax[1,1],vN_C,vN_F,"NH sea-ice volume","10³ km³"); ts(ax[1,2],vS_C,vS_F,"SH sea-ice volume","10³ km³")
fig.suptitle("C port (ice_strength 0.5 + wind g2r) vs Fortran PP — nereus diagnostics, 24 months",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT/"timeseries.png",dpi=120,bbox_inches="tight"); print("wrote timeseries.png")

# ============ surface maps (annual 1959) ============
PC=ccrs.PlateCarree()
def robmap(ax,d,vmin,vmax,cmap,ttl,lab):
    nr.plot(d,lon,lat,projection="rob",cmap=cmap,vmin=vmin,vmax=vmax,ax=ax,colorbar=True,colorbar_label=lab,title=ttl)
fig=plt.figure(figsize=(18,12))
rows=[("sst","SST","°C",-2,30,2.0,"RdYlBu_r"),("sss","SSS","PSU",30,38,0.5,"viridis"),("ssh","SSH","m",-2,1,0.2,"RdBu_r")]
for i,(v,nm,u,lo,hi,dv,cm) in enumerate(rows):
    c=ann2d(C,v,1959); f=ann2d(F,v,1959)
    for j,(dat,ttl,cmap,a,b) in enumerate([(c,f"{nm} C",cm,lo,hi),(f,f"{nm} Fortran",cm,lo,hi),(c-f,f"Δ{nm} (C−F)","RdBu_r",-dv,dv)]):
        ax=fig.add_subplot(3,3,i*3+j+1,projection=ccrs.Robinson()); robmap(ax,dat,a,b,cmap,ttl,u)
fig.suptitle("Surface climate, annual 1959 — C port vs Fortran PP",fontsize=13)
fig.savefig(OUT/"maps_surface.png",dpi=110,bbox_inches="tight"); print("wrote maps_surface.png")

# ============ ice maps (Mar=2, Sep=8) NPS/SPS ============
NPS=ccrs.NorthPolarStereo(); SPS=ccrs.SouthPolarStereo()
def polmap(ax,d,vmin,vmax,cmap,ttl,lab,south=False):
    ax.set_extent([-180,180,-90,-50] if south else [-180,180,50,90],PC)
    ax.add_feature(cfeature.LAND,fc="0.8",zorder=3); ax.coastlines(lw=.3,zorder=4)
    m=np.isfinite(d)&((lat<-45) if south else (lat>45))
    im=ax.scatter(lon[m],lat[m],c=d[m],s=2,vmin=vmin,vmax=vmax,cmap=cmap,transform=PC,zorder=2); plt.colorbar(im,ax=ax,shrink=.6,label=lab); ax.set_title(ttl)
fig=plt.figure(figsize=(15,10))
specs=[("m_ice",2,"m_ice Mar",0,4,NPS,False),("a_ice",2,"a_ice Mar",0,1,NPS,False),("a_ice",8,"a_ice Sep",0,1,NPS,False),
       ("m_ice",8,"m_ice Sep SH",0,2,SPS,True)]
for k,(v,mon,ttl,lo,hi,proj,south) in enumerate(specs):
    c=m2d(C,v,1959,mon); f=m2d(F,v,1959,mon)
    ax=fig.add_subplot(3,4,k+1,projection=proj); polmap(ax,c,lo,hi,"Blues",f"{ttl} C","",south)
    ax=fig.add_subplot(3,4,k+5,projection=proj); polmap(ax,f,lo,hi,"Blues",f"{ttl} Fortran","",south)
    ax=fig.add_subplot(3,4,k+9,projection=proj); polmap(ax,c-f,-(hi-lo)/4,(hi-lo)/4,"RdBu_r",f"Δ{ttl}","",south)
fig.suptitle("Sea ice, 1959 — C port vs Fortran PP",fontsize=13)
fig.savefig(OUT/"maps_ice.png",dpi=110,bbox_inches="tight"); print("wrote maps_ice.png")

# ============ depth profiles + band volume-means (nereus) ============
def ann3d(d,var,yr):
    ds=nc.Dataset(cpath(d,var,yr)); acc=np.zeros((NL,len(lon))); cnt=np.zeros((NL,len(lon)))
    for m in range(12):
        a=clean(np.asarray(ds[var][m])); a=np.where(a==0,np.nan,a); ok=np.isfinite(a); acc[ok]+=a[ok]; cnt[ok]+=1
    ds.close(); return np.where(cnt>0,acc/np.where(cnt>0,cnt,1),np.nan)
tC=ann3d(C,"temp",1959); tF=ann3d(F,"temp",1959); sC=ann3d(C,"salt",1959); sF=ann3d(F,"salt",1959)
def prof(a):  # per-level area-wtd mean, weighted by wet area nan3[L]
    return np.array([nr.surface_mean(a[L],nan3[L]) for L in range(NL)])
tCp,tFp,sCp,sFp=prof(tC),prof(tF),prof(sC),prof(sF)
fig,ax=plt.subplots(1,3,figsize=(15,7))
ax[0].plot(tFp,depth,'-o',ms=3,c='k',label='Fortran'); ax[0].plot(tCp,depth,'-s',ms=3,c='tab:red',label='C'); ax[0].set_title('global T'); ax[0].set_xlabel('°C')
ax[1].plot(sFp,depth,'-o',ms=3,c='k',label='Fortran'); ax[1].plot(sCp,depth,'-s',ms=3,c='tab:red',label='C'); ax[1].set_title('global S'); ax[1].set_xlabel('PSU')
ax[2].plot(tCp-tFp,depth,'-s',ms=3,c='tab:red',label='ΔT'); ax[2].plot((sCp-sFp)*10,depth,'-^',ms=3,c='tab:blue',label='ΔS×10'); ax[2].axvline(0,c='gray',lw=.7); ax[2].set_title('bias'); ax[2].set_xlabel('ΔT[°C], ΔS×10')
for a in ax: a.invert_yaxis(); a.set_ylabel('depth [m]'); a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle("Global T/S profiles, annual 1959 — C port vs Fortran PP (nereus surface_mean per level)")
fig.tight_layout(); fig.savefig(OUT/"profiles.png",dpi=120,bbox_inches="tight"); print("wrote profiles.png")
print("\n=== depth-band volume-mean T/S (nr.volume_mean) C vs Fortran ===")
print(f"  {'band [m]':12s} {'T_C':>7s} {'T_F':>7s} {'S_C':>8s} {'S_F':>8s}")
for lo,hi in [(0,100),(100,700),(700,2000),(2000,6500)]:
    tc=nr.volume_mean(tC,area,thick,depth,depth_min=lo,depth_max=hi)
    tf=nr.volume_mean(tF,area,thick,depth,depth_min=lo,depth_max=hi)
    sc=nr.volume_mean(sC,area,thick,depth,depth_min=lo,depth_max=hi)
    sf=nr.volume_mean(sF,area,thick,depth,depth_min=lo,depth_max=hi)
    print(f"  {lo:4d}-{hi:<7d} {tc:7.3f} {tf:7.3f} {sc:8.4f} {sf:8.4f}")
print("done →",OUT)
