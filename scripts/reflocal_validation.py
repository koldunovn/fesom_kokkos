#!/usr/bin/env python3
"""ref_sss_local validation: does using LOCAL surface SSS as the virtual-salt-flux
reference salinity (Fortran namelist.tra ref_sss_local=.true.) shrink the Arctic
freshwater bias that MFCT alone could not?

Compares Fortran (ref) vs C-MFCT (ref_sss_local=false, before) vs C-MFCT+reflocal
(after). Global ΔSST/ΔSSS maps + a north-polar ΔSSS panel + regional stats with an
Arctic focus.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python reflocal_validation.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import nereus as nr

F  = "/scratch/a/a270088/fortran_pp_2yr"               # Fortran+PP reference
C1 = "/work/ab0995/a270088/port/mfct_1yr"              # MFCT, ref_sss_local=false (before)
C2 = "/work/ab0995/a270088/port/reflocal_1yr"          # MFCT + ref_sss_local=true (after)
DIAG = f"{F}/fesom.mesh.diag.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
FILL = 1e29

mesh = nr.fesom.load_mesh(DIAG)
lon = np.asarray(mesh["lon"].values); lat = np.asarray(mesh["lat"].values)
ds = nc.Dataset(DIAG)
ea = np.asarray(ds.variables["elem_area"][:])
fn = np.asarray(ds.variables["face_nodes"][:]); fn = fn.T if fn.shape[0]==3 else fn
fn = fn-1 if fn.min()==1 else fn
na = np.zeros(len(lat))
for k in range(3): np.add.at(na, fn[:,k], ea/3.0)

def ann(path, var, suffix):
    a = np.asarray(nc.Dataset(f"{path}/{var}.fesom.1958{suffix}.nc").variables[var][:])
    a = np.where(np.abs(a) < FILL, a, np.nan); return np.nanmean(a, 0)

def stats(tag, d, wet):
    print(f"\n=== {tag}: area-weighted mean (rms) by region ===")
    for nm, m in [("GLOBAL", wet), ("Arctic >60N", wet&(lat>60)), ("Arctic >70N", wet&(lat>70)),
                  ("SubpolarN 45-60", wet&(lat>=45)&(lat<60)), ("Equator |lat|<10", wet&(np.abs(lat)<10)),
                  ("S high <-60", wet&(lat<-60))]:
        if m.sum()>30:
            print(f"  {nm:18s}: mean {np.sum(d[m]*na[m])/np.sum(na[m]):+.4f}  "
                  f"rms {np.sqrt(np.sum(d[m]**2*na[m])/np.sum(na[m])):.3f}  (|max| {np.nanmax(np.abs(d[m])):.2f})")

fig = plt.figure(figsize=(18, 11))
for i, (var, vlim, unit) in enumerate([("sst", 1.0, "degC"), ("sss", 0.5, "PSU")]):
    cF  = ann(F,  var, ""); cC1 = ann(C1, var, ".monthly"); cC2 = ann(C2, var, ".monthly")
    wet = np.isfinite(cF) & np.isfinite(cC1) & np.isfinite(cC2)
    d1 = np.where(wet, cC1 - cF, np.nan)   # before fix − F
    d2 = np.where(wet, cC2 - cF, np.nan)   # after fix  − F
    for j, (d, ttl) in enumerate([(d1, f"Δ{var.upper()} MFCT−F (before)"),
                                  (d2, f"Δ{var.upper()} +reflocal−F (after)"),
                                  (np.where(wet, cC2-cC1, np.nan), f"Δ{var.upper()} reflocal−MFCT")]):
        ax = fig.add_subplot(2, 3, i*3+j+1, projection=ccrs.Robinson()); ax.set_global()
        vl = vlim if j < 2 else vlim*0.6
        nr.plot(d, lon, lat, projection="rob", cmap="RdBu_r", vmin=-vl, vmax=vl, ax=ax,
                colorbar=True, colorbar_label=f"Δ{var} [{unit}]", title=ttl)
    stats(f"Δ{var.upper()} MFCT−F (before)", d1, wet)
    stats(f"Δ{var.upper()} +reflocal−F (after)", d2, wet)
fig.suptitle("ref_sss_local validation: Fortran vs C-MFCT (before) vs C-MFCT+reflocal (after), annual 1958", fontsize=13)
fig.tight_layout(); fp = OUT/"reflocal_sst_sss.png"; fig.savefig(fp, dpi=110, bbox_inches="tight"); plt.close(fig)
print("\nwrote", fp)

# north-polar SSS before/after
figp = plt.figure(figsize=(13, 6))
cF = ann(F,"sss",""); cC1 = ann(C1,"sss",".monthly"); cC2 = ann(C2,"sss",".monthly")
wet = np.isfinite(cF)&np.isfinite(cC1)&np.isfinite(cC2)
for j,(d,ttl) in enumerate([(np.where(wet,cC1-cF,np.nan),"ΔSSS MFCT−F (before)"),
                            (np.where(wet,cC2-cF,np.nan),"ΔSSS +reflocal−F (after)")]):
    ax = figp.add_subplot(1,2,j+1, projection=ccrs.NorthPolarStereo()); ax.set_extent([-180,180,60,90],ccrs.PlateCarree())
    nr.plot(d, lon, lat, projection="np", cmap="RdBu_r", vmin=-1.0, vmax=1.0, ax=ax,
            colorbar=True, colorbar_label="ΔSSS [PSU]", title=ttl)
figp.suptitle("Arctic SSS bias: ref_sss_local fix (local vs global reference salinity)", fontsize=12)
figp.tight_layout(); fpp = OUT/"reflocal_arctic_sss.png"; figp.savefig(fpp, dpi=120, bbox_inches="tight")
print("wrote", fpp)
