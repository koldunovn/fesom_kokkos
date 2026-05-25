#!/usr/bin/env python3
"""Exp #2 (DT1800_HANDOFF §4.2): re-measure C-vs-Fortran grid-scale (2Δx) element-
velocity energy at dt=1800 WITH both physics fixes in (ice 0.5 + wind g2r), now
that the wind-rotation direction confound is removed.

roughness = |vel_e - mean(vel over 8 nearest cells)|  (the 2Δx amplitude per cell)
C   = windrot_dt1800_test (HEAD 8dac997, dt=1800, both fixes)
Fortran+PP = fortran_pp_dt1800 (dt=1800, stable reference)
Both March monthly-mean surface element velocity (same mesh, 244659 elems).

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python exp2_gridscale_energy.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
from scipy.spatial import cKDTree

CU = "/work/ab0995/a270088/port/windrot_dt1800_test/u.fesom.1958.monthly.nc"
CV = "/work/ab0995/a270088/port/windrot_dt1800_test/v.fesom.1958.monthly.nc"
FU = "/scratch/a/a270088/fortran_pp_dt1800/u.fesom.1958.nc"
FV = "/scratch/a/a270088/fortran_pp_dt1800/v.fesom.1958.nc"
CA = "/work/ab0995/a270088/port/windrot_dt1800_test/a_ice.fesom.1958.monthly.nc"
FA = "/scratch/a/a270088/fortran_pp_dt1800/a_ice.fesom.1958.nc"
DIAG = "/scratch/a/a270088/fortran_pp_2yr/fesom.mesh.diag.nc"
MAR = 2  # March = 3rd monthly record (both start January)

def load(path, var, t, nz=0):
    d = nc.Dataset(path); a = np.asarray(d.variables[var][t, nz, :]); d.close(); return a

# element centroids from the C file (same mesh & global element order for both)
d = nc.Dataset(CU); clon = np.asarray(d.variables["lon_elem"][:]); clat = np.asarray(d.variables["lat_elem"][:]); d.close()

cu = load(CU, "u", MAR); cv = load(CV, "v", MAR)
fu = load(FU, "u", MAR); fv = load(FV, "v", MAR)

# ordering sanity: the large-scale speed fields must correlate if elem order aligns
cs = np.hypot(cu, cv); fs = np.hypot(fu, fv)
good = np.isfinite(cs) & np.isfinite(fs)
print(f"alignment check: corr(|uC|,|uF|) = {np.corrcoef(cs[good], fs[good])[0,1]:.4f}  "
      f"(high => element ordering aligned)")
print(f"max |uv|: C={cs[good].max():.3f}  F={fs[good].max():.3f}")

# KDTree on centroid unit vectors (pole/dateline-safe), 8 nearest
r = np.pi/180.0
x = np.cos(clat*r)*np.cos(clon*r); y = np.cos(clat*r)*np.sin(clon*r); z = np.sin(clat*r)
tree = cKDTree(np.c_[x, y, z])
_, idx = tree.query(np.c_[x, y, z], k=9)   # [:,0] is self
nbr = idx[:, 1:]

def rough(u, v):
    um = u[nbr].mean(1); vm = v[nbr].mean(1)
    return np.hypot(u - um, v - vm)

rC = rough(cu, cv); rF = rough(fu, fv)

# element a_ice (mean of 3 vertex node a_ice) for MIZ conditioning
dd = nc.Dataset(DIAG)
fn = np.asarray(dd.variables["face_nodes"][:]); dd.close()
fn = fn.T if fn.shape[0] == 3 else fn
fn = fn - 1 if fn.min() == 1 else fn
da = nc.Dataset(CA); aC_n = np.asarray(da.variables["a_ice"][MAR, :]); da.close()
da = nc.Dataset(FA); aF_n = np.asarray(da.variables["a_ice"][MAR, :]); da.close()
aC_e = aC_n[fn].mean(1); aF_e = aF_n[fn].mean(1)

def band(name, mask):
    m = mask & np.isfinite(rC) & np.isfinite(rF)
    if m.sum() == 0: print(f"  {name:28s}: (empty)"); return
    a, b = rC[m].mean(), rF[m].mean()
    print(f"  {name:28s}: C={a:.4f}  F={b:.4f}  C/F={a/b:.2f}  (n={m.sum()})")

print("\nMean cell-velocity 2Δx roughness  (C = windrot dt1800, F = Fortran+PP dt1800), March surface:")
band("GLOBAL", np.ones_like(rC, bool))
band("Arctic >70N", clat > 70)
band("central Arctic >80N", clat > 80)
band("Antarctic <-60S", clat < -60)
band("Tropics |lat|<30", np.abs(clat) < 30)
print("  -- conditioned on ice (element a_ice) --")
band("ice interior a_ice>0.8", (aC_e > 0.8))
band("MIZ 0.1<a_ice<0.5", (aC_e > 0.1) & (aC_e < 0.5))
band("ice-free a_ice<0.05", (aC_e < 0.05))
band("MIZ & Arctic", (aC_e > 0.1) & (aC_e < 0.5) & (clat > 60))

print("\nSUMMARY: C/F ratio > 1 means the C carries MORE grid-scale velocity energy "
      "upstream of the biharmonic (supports the upstream-source hypothesis).")
