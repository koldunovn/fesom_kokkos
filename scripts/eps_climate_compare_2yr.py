#!/usr/bin/env python3
"""2-year climate validation of the AB2 epsilon fix.
C dt=1800 (both eps=0.1 fixes, eps_2yr_dt1800) vs Fortran+PP dt=1800 (fortran_pp_2yr).
Same dt, same scheme (PP), both eps=0.1 — apples-to-apples. Annual-mean spatial
stats per year + a DRIFT check (year-1 vs year-2 bias: stable = no runaway drift).

Run: /work/ab0995/a270088/mambaforge/envs/nereus/bin/python eps_climate_compare_2yr.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc

C = "/work/ab0995/a270088/port/eps_2yr_dt1800"
F = "/scratch/a/a270088/fortran_pp_2yr"

def surf_annual(path, var):
    d = nc.Dataset(path); a = d.variables[var]
    x = np.asarray(a[:, 0, :]) if a.ndim == 3 else np.asarray(a[:])  # (t,nz,n)->surf or (t,n)
    d.close()
    x = np.where(np.abs(x) < 1e30, x, np.nan)
    return np.nanmean(x, axis=0)   # annual mean

print(f"{'field':6s} {'year':4s} | {'corr':>7s} {'bias(C-F)':>10s} {'RMS':>8s} {'|Δ|max':>7s}   (annual-mean surface)")
biases = {}
for var in ("sst", "sss", "ssh", "a_ice", "m_ice"):
    biases[var] = {}
    for yr in (1958, 1959):
        try:
            c = surf_annual(f"{C}/{var}.fesom.{yr}.monthly.nc", var)
            f = surf_annual(f"{F}/{var}.fesom.{yr}.nc", var)
        except Exception as e:
            print(f"  {var} {yr}: skip ({e})"); continue
        g = np.isfinite(c) & np.isfinite(f)
        d = c[g] - f[g]
        corr = np.corrcoef(c[g], f[g])[0, 1]
        rms = np.sqrt(np.mean(d**2))
        biases[var][yr] = d.mean()
        print(f"{var:6s} {yr}   | {corr:7.4f} {d.mean():+10.4f} {rms:8.4f} {np.abs(d).max():7.3f}")
    print()

print("=== DRIFT check (year-2 bias − year-1 bias; ~0 = stable, no runaway) ===")
for var, by in biases.items():
    if 1958 in by and 1959 in by:
        print(f"  {var:6s}: yr1 {by[1958]:+.4f}  yr2 {by[1959]:+.4f}  drift {by[1959]-by[1958]:+.4f}")
