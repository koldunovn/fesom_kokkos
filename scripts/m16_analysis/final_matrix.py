#!/usr/bin/env python3
"""§5 final matrix reader: monthly SP-DP ratio with SP-ensemble error bars, global and by latitude band.

For each code the SP-DP departure is measured with EVERY SP member against the unperturbed DP arm
(fsp, fsp_r* vs fdp; psp, psp_r* vs pdp): the spread across members is the error bar the 60-70S
residual was missing. The FP64 envelope is the mean over the r-family (sigma 1e-6 K) twins.
"""
import sys, os, numpy as np
sys.path.insert(0, "/home/a/a270088/port_kokkos_sp/scripts")
import m16_faith_compare as fc
from netCDF4 import Dataset

R = sys.argv[1]
VARS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sst", "temp", "salt", "a_ice", "ssh"]
MONTHS = [int(m) for m in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(12))
YEAR = sys.argv[4] if len(sys.argv) > 4 else None
if YEAR: fc.YEAR = YEAR
names = sorted(d for d in os.listdir(R) if os.path.isdir(os.path.join(R, d)))
arm = lambda n: os.path.join(R, n)
md = Dataset(os.path.join(R, "fdp", "output", "fesom.mesh.diag.nc"))
lat = np.array(md.variables["lat"][:]); md.close()
if np.abs(lat).max() < 3.2: lat = np.degrees(lat)
BANDS = [("60-90S", -90, -60), ("60-70S", -70, -60), ("40-60S", -60, -40), ("20S-20N", -20, 20), ("40-60N", 40, 60), ("60-90N", 60, 90)]

def rel(a, b, m):
    n = np.linalg.norm(b[m]); return float(np.linalg.norm(a[m] - b[m]) / n) if n > 0 else np.nan

def bandmask(mask, lo, hi):
    bm = (lat >= lo) & (lat < hi)
    return mask & (bm[None, :] if mask.ndim == 2 else bm)

for var in VARS:
    print(f"\n#### {var}")
    hdr = f"{'mon':>3} | {'F sp-dp':>9} ±{'ens':>7} | {'P sp-dp':>9} ±{'ens':>7} | {'ratio':>6} ±{'':>5} | {'F env':>8} {'P env':>8} | {'F/env':>5} {'P/env':>5}"
    print(hdr)
    rows = {}
    for mon in MONTHS:
        got = {}
        for n in names:
            x, _ = fc.load(arm(n), var, mon)
            if x is not None: got[n] = x
        if not all(k in got for k in ("fdp", "fsp", "pdp", "psp")): print(f"{mon:3d} missing arms"); continue
        mask = np.ones(got["fdp"].shape, bool)
        for x in got.values(): mask &= np.isfinite(x) & (np.abs(x) < 1e30)
        allz = np.ones_like(mask)
        for x in got.values(): allz &= (x == 0.0)
        mask &= ~allz
        fsp = [n for n in got if n == "fsp" or n.startswith("fsp_r")]
        psp = [n for n in got if n == "psp" or n.startswith("psp_r")]
        fenv = [n for n in got if n.startswith("fdp_r")]; penv = [n for n in got if n.startswith("pdp_r")]
        def line(m, tag):
            F = np.array([rel(got[n], got["fdp"], m) for n in fsp]); P = np.array([rel(got[n], got["pdp"], m) for n in psp])
            Fe = np.mean([rel(got[n], got["fdp"], m) for n in fenv]) if fenv else np.nan
            Pe = np.mean([rel(got[n], got["pdp"], m) for n in penv]) if penv else np.nan
            ratio = P.mean() / F.mean(); dr = ratio * np.sqrt((P.std() / P.mean()) ** 2 + (F.std() / F.mean()) ** 2) if len(F) > 1 else np.nan
            print(f"{tag:>3} | {F.mean():9.3e} ±{F.std():7.1e} | {P.mean():9.3e} ±{P.std():7.1e} | {ratio:6.2f} ±{dr:5.2f} | {Fe:8.2e} {Pe:8.2e} | {F.mean()/Fe:5.2f} {P.mean()/Pe:5.2f}")
            return ratio, dr
        rows[mon] = line(mask, str(mon + 1))
    # latitude bands at the LAST requested month
    mon = MONTHS[-1]
    print(f"   -- month {mon+1} by latitude band ({len(fsp)} F members, {len(psp)} P members) --")
    for bname, lo, hi in BANDS:
        bm = bandmask(mask, lo, hi)
        if bm.sum() < 100: continue
        line(bm, bname)
