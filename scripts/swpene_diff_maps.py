#!/usr/bin/env python3
"""2D difference maps for the shortwave-penetration port: C − Fortran(with pene),
annual-mean 1958, at the surface and at ~35 m (where the cold-subsurface dipole
peaked). Before (C no pene) vs after (C +sw_pene port). Masked to Fortran-wet nodes.

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python swpene_diff_maps.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import nereus as nr

COLD = "/work/ab0995/a270088/port/eps_2yr_dt1800/temp.fesom.1958.monthly.nc"   # C no pene
CNEW = "/work/ab0995/a270088/port/eps_swpene_1yr/temp.fesom.1958.monthly.nc"   # C +pene
FP   = "/scratch/a/a270088/fortran_pp_2yr/temp.fesom.1958.nc"                  # Fortran +pene
DIAG = "/scratch/a/a270088/fortran_pp_2yr/fesom.mesh.diag.nc"
OUT  = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
FILL = 1e29
mesh = nr.fesom.load_mesh(DIAG)
lon = np.asarray(mesh["lon"].values); lat = np.asarray(mesh["lat"].values)
Z = np.asarray(nc.Dataset(COLD).variables["Z"][:])

def annlev(path, nz):
    a = np.asarray(nc.Dataset(path).variables["temp"][:, nz, :])   # (12, n)
    a = np.where(np.abs(a) < FILL, a, np.nan)
    return np.nanmean(a, axis=0)

# surface (nz=0, -2.5m) and dipole-peak (nz=4, ~-35m)
levels = [(0, "surface (2.5 m)"), (4, f"{abs(Z[4]):.0f} m")]
fig = plt.figure(figsize=(15, 9))
for r, (nz, zlab) in enumerate(levels):
    fO = annlev(FP, nz)
    wet = np.isfinite(fO)
    dO = np.where(wet, annlev(COLD, nz) - fO, np.nan)   # before
    dN = np.where(wet, annlev(CNEW, nz) - fO, np.nan)   # after
    vlim = 1.0
    for ccol, (d, tag) in enumerate([(dO, "BEFORE (C no sw_pene)"), (dN, "AFTER (C +sw_pene port)")]):
        ax = fig.add_subplot(2, 2, r*2 + ccol + 1, projection=ccrs.Robinson())
        ax.set_global()
        nr.plot(d, lon, lat, projection="rob", cmap="RdBu_r", vmin=-vlim, vmax=vlim, ax=ax,
                colorbar=True, colorbar_label="ΔT [degC]",
                title=f"ΔT @ {zlab} — {tag}")
fig.suptitle("Shortwave penetration port: C − Fortran(with sw_pene), annual 1958\n"
             "subsurface cold bias (BEFORE, left) is removed by the port (AFTER, right)",
             fontsize=12)
fig.tight_layout()
fp = OUT / "swpene_diff_maps.png"; fig.savefig(fp, dpi=120, bbox_inches="tight")
print("wrote", fp)
