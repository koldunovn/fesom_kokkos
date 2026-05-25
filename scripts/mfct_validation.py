#!/usr/bin/env python3
"""3-way MFCT validation: Fortran (ref) vs C-central (before) vs C-MFCT (after).

Does porting MFCT (3rd-order horizontal tracer advection) shrink the systematic
biases the over-diffusive 2nd-order central scheme produced — specifically the
river-mouth SSS hotspots and the equatorial-cold-tongue SST bias?

Panels: ΔSST and ΔSSS for (C-central − F) and (C-MFCT − F), plus (MFCT − central)
to isolate the scheme change. Regional + river-mouth + equatorial stats printed.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python mfct_validation.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import nereus as nr

F  = "/scratch/a/a270088/fortran_pp_2yr"              # Fortran+PP reference
C0 = "/work/ab0995/a270088/port/eps_swpene_albw_1yr"  # C, central HO (before)
C1 = "/work/ab0995/a270088/port/mfct_1yr"             # C, MFCT HO (after)
DIAG = f"{F}/fesom.mesh.diag.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
OUT.mkdir(parents=True, exist_ok=True)
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
    print(f"\n=== {tag}: area-weighted mean by region (|max|) ===")
    for nm, m in [("GLOBAL", wet), ("Equator |lat|<10", wet&(np.abs(lat)<10)),
                  ("Tropics |lat|<23", wet&(np.abs(lat)<23)),
                  ("Midlat 40-60", wet&(np.abs(lat)>=40)&(np.abs(lat)<60)),
                  ("N high >60", wet&(lat>60)), ("S high <-60", wet&(lat<-60))]:
        if m.sum()>50:
            print(f"  {nm:20s}: {np.sum(d[m]*na[m])/np.sum(na[m]):+.4f}  (|max| {np.nanmax(np.abs(d[m])):.2f}  rms {np.sqrt(np.sum(d[m]**2*na[m])/np.sum(na[m])):.3f})")

fig = plt.figure(figsize=(18, 11))
for i, (var, vlim, unit) in enumerate([("sst", 1.0, "degC"), ("sss", 0.5, "PSU")]):
    cF  = ann(F,  var, "")
    cC0 = ann(C0, var, ".monthly")
    cC1 = ann(C1, var, ".monthly")
    wet = np.isfinite(cF) & np.isfinite(cC0) & np.isfinite(cC1)
    d0 = np.where(wet, cC0 - cF, np.nan)   # before − F
    d1 = np.where(wet, cC1 - cF, np.nan)   # after  − F
    dd = np.where(wet, cC1 - cC0, np.nan)  # MFCT − central
    for j, (d, ttl, vl) in enumerate([(d0, f"Δ{var.upper()}  central−F", vlim),
                                       (d1, f"Δ{var.upper()}  MFCT−F",    vlim),
                                       (dd, f"Δ{var.upper()}  MFCT−central", vlim*0.6)]):
        ax = fig.add_subplot(2, 3, i*3+j+1, projection=ccrs.Robinson()); ax.set_global()
        nr.plot(d, lon, lat, projection="rob", cmap="RdBu_r", vmin=-vl, vmax=vl, ax=ax,
                colorbar=True, colorbar_label=f"Δ{var} [{unit}]", title=ttl)
    stats(f"Δ{var.upper()} central−F", d0, wet)
    stats(f"Δ{var.upper()} MFCT−F",    d1, wet)
    # river-mouth proxy: nodes where C-central had the largest |SSS bias| (low-salinity plumes)
    if var == "sss":
        rm = wet & (np.abs(d0) > 1.0)
        if rm.sum() > 0:
            print(f"\n  river-mouth proxy (|ΔSSS central−F|>1, n={rm.sum()}):")
            print(f"    central−F rms = {np.sqrt(np.nanmean(d0[rm]**2)):.3f}   "
                  f"MFCT−F rms = {np.sqrt(np.nanmean(d1[rm]**2)):.3f}  "
                  f"(improvement {100*(1-np.sqrt(np.nanmean(d1[rm]**2))/np.sqrt(np.nanmean(d0[rm]**2))):+.0f}%)")
fig.suptitle("MFCT validation: Fortran ref vs C-central (before) vs C-MFCT (after), annual 1958", fontsize=13)
fig.tight_layout(); fp = OUT/"mfct_3way_sst_sss.png"; fig.savefig(fp, dpi=110, bbox_inches="tight")
print("\nwrote", fp)
