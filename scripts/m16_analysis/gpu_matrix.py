import sys, os, numpy as np
sys.path.insert(0, "/home/a/a270088/port_kokkos_sp/scripts")
import m16_faith_compare as fc
from netCDF4 import Dataset
R = "/work/ab0995/a270088/port2/m16/faith/final_year_gpu"
F = "/work/ab0995/a270088/port2/m16/faith/final_year"
md = Dataset(os.path.join(F, "fdp", "output", "fesom.mesh.diag.nc")); lat = np.array(md.variables["lat"][:]); md.close()
if np.abs(lat).max() < 3.2: lat = np.degrees(lat)
ds = Dataset(os.path.join(R, "gsp", "output", "sst.fesom.1958.monthly.nc")); print("gsp sst records:", ds.variables["sst"].shape); ds.close()
def rel(a, b, m):
    n = np.linalg.norm(b[m]); return float(np.linalg.norm(a[m] - b[m]) / n)
for var in ["sst", "temp", "salt", "a_ice", "ssh"]:
    got = {}
    for a in ["gdp", "gdp_2", "gsp", "pdp", "psp"]:
        x, _ = fc.load(os.path.join(R, a), var, -1); got[a] = x
    for a in ["fdp", "fsp"]:
        x, _ = fc.load(os.path.join(F, a), var, -1); got[a] = x
    mask = np.ones(got["gdp"].shape, bool)
    for x in got.values(): mask &= np.isfinite(x) & (np.abs(x) < 1e30)
    allz = np.ones_like(mask)
    for x in got.values(): allz &= (x == 0.0)
    mask &= ~allz
    band = (lat >= -70) & (lat < -60); bm = mask & (band[None, :] if mask.ndim == 2 else band)
    def row(m):
        return (rel(got["gsp"], got["gdp"], m), rel(got["gdp_2"], got["gdp"], m), rel(got["gdp"], got["pdp"], m),
                rel(got["psp"], got["pdp"], m), rel(got["fsp"], got["fdp"], m), rel(got["gsp"], got["gdp_2"], m))
    g = row(mask); b = row(bm)
    print(f"{var:6s} GLOBAL  gsp-gdp {g[0]:.3e}  gsp-gdp_2 {g[5]:.3e} | self-noise gdp_2-gdp {g[1]:.3e} | cuda-vs-serial DP {g[2]:.3e} | serial(opt7) psp-pdp {g[3]:.3e} | fortran fsp-fdp {g[4]:.3e} | ratio gpu/F {g[0]/g[4]:.2f}  serial7/F {g[3]/g[4]:.2f}")
    print(f"{'':6s} 60-70S  gsp-gdp {b[0]:.3e}  gsp-gdp_2 {b[5]:.3e} | self-noise gdp_2-gdp {b[1]:.3e} | cuda-vs-serial DP {b[2]:.3e} | serial(opt7) psp-pdp {b[3]:.3e} | fortran fsp-fdp {b[4]:.3e} | ratio gpu/F {b[0]/b[4]:.2f}  serial7/F {b[3]/b[4]:.2f}")
