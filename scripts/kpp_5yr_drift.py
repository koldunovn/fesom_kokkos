#!/usr/bin/env python3
"""K10 KPP 5-yr stability: area-weighted global-mean SST/SSS per year (1958-1962)
for the C+KPP run, to confirm bounded drift (no runaway). Mirrors the area
weighting in clim_validation_2yr.py."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import nereus as nr

C    = "/work/ab0995/a270088/port/kpp_5yr_dt1800"
DIAG = "/scratch/a/a270088/fortran_2yr_dt1800/fesom.mesh.diag.nc"
FILL = 1e29
mesh = nr.fesom.load_mesh(DIAG)
lat = np.asarray(mesh["lat"].values)
ea = np.asarray(nc.Dataset(DIAG).variables["elem_area"][:])
fn = np.asarray(nc.Dataset(DIAG).variables["face_nodes"][:]); fn = fn.T if fn.shape[0]==3 else fn
fn = fn-1 if fn.min()==1 else fn
na = np.zeros(len(lat))
for k in range(3): np.add.at(na, fn[:,k], ea/3.0)

def ann(var, year):
    a = np.asarray(nc.Dataset(f"{C}/{var}.fesom.{year}.monthly.nc").variables[var][:])
    a = np.where(np.abs(a) < FILL, a, np.nan); return np.nanmean(a, 0)
def wmean(d):
    m = np.isfinite(d); return np.sum(d[m]*na[m]) / np.sum(na[m])

print("K10 KPP 5-yr global-mean drift (area-weighted):")
for var, unit in [("sst","degC"), ("sss","PSU")]:
    gm = []
    for y in range(1958, 1963):
        try: gm.append(wmean(ann(var, y)))
        except Exception as e: gm.append(np.nan)
    s = "  ".join(f"{y}={v:.4f}" for y, v in zip(range(1958,1963), gm))
    drift = gm[-1]-gm[1] if np.isfinite(gm[-1]) and np.isfinite(gm[1]) else np.nan
    print(f"  {var.upper():4s}[{unit}]: {s}   (yr5-yr2 drift {drift:+.4f})")
