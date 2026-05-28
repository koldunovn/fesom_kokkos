#!/usr/bin/env python3
"""Spatial bias maps for M3.2 climate validation — CUDA vs Fortran + CUDA vs C-port.

Plots annual-mean Δfield for each surface field, Robinson projection (global) + a PlateCarree
zoom for ice fields where divergence concentrates polar. Style follows the C-port's
clim_validation_2yr.py (nereus.plot).

Defaults to the canonical KPP references (see docs/REFERENCE_RUNS.md). Override via
--fref / --cref / --tag if comparing a PP run or a different backend dir.

Run:
  PYTHONPATH=/home/a/a270088/PYTHON /work/ab0995/a270088/mambaforge/envs/nereus/bin/python \
      scripts/m32_bias_maps.py /work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_1yr_pin
"""
import argparse, pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import nereus as nr

# Defaults: canonical KPP refs (docs/REFERENCE_RUNS.md). kpp_5yr_fix is the same C-port SHA
# as kpp_2yr_rebase but contains 5 yr (1958-1962) and matches the C-port's published
# kpp_5yr_fix_figures script — picking it as the canonical so the same comparison code applies
# whether the Kokkos backend runs 1 yr or 5.
FORT_DEFAULT = "/scratch/a/a270088/fortran_kpp_5yr_fix"
CREF_DEFAULT = "/work/ab0995/a270088/port/kpp_5yr_fix"

ap = argparse.ArgumentParser()
ap.add_argument("backend_dir", nargs="?",
                default="/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_1yr_pin")
ap.add_argument("--fref", default=FORT_DEFAULT, help="Fortran reference dir")
ap.add_argument("--cref", default=CREF_DEFAULT, help="C-port reference dir")
ap.add_argument("--year", type=int, default=1958)
ap.add_argument("--tag",  default="kpp",
                help="tag appended to PNG filenames (eg 'kpp', 'pp'); '' for no tag")
ap.add_argument("--out",  default="/home/a/a270088/port_kokkos/docs/m32_bias_maps")
args = ap.parse_args()

BACKEND = pathlib.Path(args.backend_dir)
FORT    = pathlib.Path(args.fref)
CREF    = pathlib.Path(args.cref)
YEAR    = args.year
TAG     = ("_" + args.tag) if args.tag else ""
DIAG    = FORT/"fesom.mesh.diag.nc"
OUT     = pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

mesh = nr.fesom.load_mesh(str(DIAG))
lon, lat = np.asarray(mesh.lon), np.asarray(mesh.lat)

ICE_FIELDS = {"a_ice", "m_ice", "m_snow"}

def surf_monthly(path, var):
    """All 12 monthly surface slices (t, nodes) — keep NaN to identify open water for ice fields."""
    if not path.exists():
        return None
    d = nc.Dataset(path)
    a = d.variables[var]
    x = np.asarray(a[:, 0, :]) if a.ndim == 3 else np.asarray(a[:])
    d.close()
    return np.where(np.abs(x) < 1e30, x, np.nan)

def annual_mean(path, var, is_ice, wet_mask=None):
    """Annual mean. For ice fields: nan_to_num per month BEFORE temporal mean (Fortran masks
    open water as NaN, C writes 0 — without the per-month nan->0 the temporal nanmean drops
    open-water months at marginal-ice nodes and inflates the Fortran mean, spuriously
    diverging from C). After averaging, restrict ice fields to the wet (ocean) mask."""
    x = surf_monthly(path, var)
    if x is None: return None
    if is_ice:
        x = np.nan_to_num(x, nan=0.0)
    m = np.nanmean(x, axis=0)
    if is_ice and wet_mask is not None:
        m = np.where(wet_mask, m, np.nan)
    return m

# Wet mask = SST finite in BOTH refs (ocean domain), built once.
def build_wet():
    a = surf_monthly(BACKEND/f"sst.fesom.{YEAR}.monthly.nc", "sst")
    f = surf_monthly(FORT/f"sst.fesom.{YEAR}.nc", "sst")
    c = surf_monthly(CREF/f"sst.fesom.{YEAR}.monthly.nc", "sst")
    a, f, c = (np.nanmean(z, axis=0) if z is not None else None for z in (a, f, c))
    return np.isfinite(a) & np.isfinite(f) & np.isfinite(c)
WET = build_wet()

# 6 surface fields, sensible bias clip per field
FIELDS = [
    ("sst",   "ΔSST [°C]",    0.5),
    ("sss",   "ΔSSS [PSU]",   0.3),
    ("ssh",   "ΔSSH [m]",     0.02),
    ("a_ice", "Δa_ice [-]",   0.1),
    ("m_ice", "Δm_ice [m]",   0.2),
    ("uice",  "Δuice [m/s]",  0.05),
]

def plot_one(field, ref_label, ref_path, is_ice):
    f_back = BACKEND/f"{field}.fesom.{YEAR}.monthly.nc"
    f_ref  = ref_path/(f"{field}.fesom.{YEAR}.nc" if ref_label == "Fortran"
                       else f"{field}.fesom.{YEAR}.monthly.nc")
    a = annual_mean(f_back, field, is_ice, WET)
    b = annual_mean(f_ref,  field, is_ice, WET)
    if a is None or b is None:
        return None
    return a - b

for field, label, vlim in FIELDS:
    is_ice = field in ICE_FIELDS
    fig = plt.figure(figsize=(13, 5))
    for j, (ref_lbl, ref_path) in enumerate([("Fortran", FORT), ("C-port", CREF)]):
        d = plot_one(field, ref_lbl, ref_path, is_ice)
        ax = fig.add_subplot(1, 2, j+1, projection=ccrs.Robinson())
        ax.set_global()
        if d is None:
            ax.set_title(f"CUDA − {ref_lbl}: missing reference"); continue
        nr.plot(d, lon, lat, projection="rob", cmap="RdBu_r",
                vmin=-vlim, vmax=vlim, ax=ax,
                title=f"CUDA − {ref_lbl}  {label}  (annual {YEAR}; bias={np.nanmean(d):+.3e} RMS={np.sqrt(np.nanmean(d**2)):.3e})")
    fig.tight_layout()
    fp = OUT/f"bias_{field}_{YEAR}{TAG}.png"
    fig.savefig(fp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fp}")

print(f"\nAll bias maps -> {OUT}")
