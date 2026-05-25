#!/usr/bin/env python3
"""Surface SST + SSS difference maps, current C (with sw_pene) − Fortran+PP, annual
1958. Plus regional/regime stats to characterize the REMAINING systematic biases.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python surf_diff_maps.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import nereus as nr

C = "/work/ab0995/a270088/port/eps_swpene_albw_1yr"     # current C (sw_pene on)
F = "/scratch/a/a270088/fortran_pp_2yr"
DIAG = f"{F}/fesom.mesh.diag.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
FILL = 1e29
mesh = nr.fesom.load_mesh(DIAG)
lon = np.asarray(mesh["lon"].values); lat = np.asarray(mesh["lat"].values)
ea = np.asarray(nc.Dataset(DIAG).variables["elem_area"][:])
fn = np.asarray(nc.Dataset(DIAG).variables["face_nodes"][:]); fn = fn.T if fn.shape[0]==3 else fn
fn = fn-1 if fn.min()==1 else fn
na = np.zeros(len(lat));
for k in range(3): np.add.at(na, fn[:,k], ea/3.0)

def ann(path, var, suffix):
    a = np.asarray(nc.Dataset(f"{path}/{var}.fesom.1958{suffix}.nc").variables[var][:])  # (12,n)
    a = np.where(np.abs(a) < FILL, a, np.nan); return np.nanmean(a, 0)

fig = plt.figure(figsize=(16, 6))
for i, (var, vlim, unit) in enumerate([("sst", 1.0, "degC"), ("sss", 0.5, "PSU")]):
    cF = ann(F, var, ""); cC = ann(C, var, ".monthly")
    wet = np.isfinite(cF) & np.isfinite(cC)
    d = np.where(wet, cC - cF, np.nan)
    ax = fig.add_subplot(1, 2, i+1, projection=ccrs.Robinson()); ax.set_global()
    nr.plot(d, lon, lat, projection="rob", cmap="RdBu_r", vmin=-vlim, vmax=vlim, ax=ax,
            colorbar=True, colorbar_label=f"Δ{var} [{unit}]", title=f"Δ{var.upper()} = C − Fortran")
    # regional stats
    print(f"\n=== Δ{var.upper()} (C − Fortran), area-weighted mean by region ===")
    for nm, m in [("GLOBAL", wet), ("Tropics |lat|<23", wet&(np.abs(lat)<23)),
                  ("Subtropics 23-40", wet&(np.abs(lat)>=23)&(np.abs(lat)<40)),
                  ("Midlat 40-60", wet&(np.abs(lat)>=40)&(np.abs(lat)<60)),
                  ("N high >60", wet&(lat>60)), ("S high <-60", wet&(lat<-60))]:
        if m.sum()>50:
            print(f"  {nm:20s}: {np.sum(d[m]*na[m])/np.sum(na[m]):+.4f}  (|max| {np.nanmax(np.abs(d[m])):.2f})")
fig.suptitle("Remaining surface bias: C (sw_pene + albw=0.1) − Fortran+PP, annual 1958", fontsize=12)
fig.tight_layout(); fp = OUT/"surf_diff_sst_sss_albw.png"; fig.savefig(fp, dpi=120, bbox_inches="tight")
print("\nwrote", fp)
