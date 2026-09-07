#!/usr/bin/env python3
"""M8 Gate-4 verdict: FP32 1-yr climate leg vs a reference run, M5.23-bar convention.

Follows the house eps_climate_compare.py convention exactly (surface level, finite &
|x|<1e30 mask, np.corrcoef) so numbers are comparable to the M5.23 / CGPOLY-cert class
(sst 1.00000 / sss 0.99996 / ssh 1.00000 / a_ice 0.99997 vs the FP64 twin).

Per variable: monthly pattern correlation + mean/max bias, and the ANNUAL-mean-map
correlation (the bar number). Handles port naming (<var>.fesom.YYYY.monthly.nc) and
Fortran naming (<var>.fesom.YYYY.nc) automatically.

Usage: mp_gate4_verdict.py TEST_DIR REF_DIR [--year 1958] [--vars sst sss ssh a_ice m_ice]
"""
import argparse
import pathlib
import warnings

warnings.filterwarnings("ignore")
import netCDF4 as nc
import numpy as np


def surf(d: pathlib.Path, var: str, year: int):
    for name in (f"{var}.fesom.{year}.monthly.nc", f"{var}.fesom.{year}.nc"):
        p = d / name
        if p.exists():
            ds = nc.Dataset(p)
            a = ds.variables[var]
            x = np.asarray(a[:, 0, :]) if a.ndim == 3 else np.asarray(a[:])
            ds.close()
            return x
    raise FileNotFoundError(f"{var} ({year}) in {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_dir", type=pathlib.Path)
    ap.add_argument("ref_dir", type=pathlib.Path)
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--vars", nargs="+", default=["sst", "sss", "ssh", "a_ice", "m_ice"])
    ap.add_argument("--monthly", action="store_true", help="also print per-month rows")
    args = ap.parse_args()

    print(f"# test={args.test_dir}  ref={args.ref_dir}  year={args.year}")
    print(f"{'var':6s} {'annual_corr':>12s} {'meanΔ':>10s} {'|Δ|max':>10s}   (annual-mean maps)")
    for var in args.vars:
        try:
            t = surf(args.test_dir, var, args.year)
            r = surf(args.ref_dir, var, args.year)
        except Exception as e:
            print(f"{var:6s} skip ({e})")
            continue
        nm = min(t.shape[0], r.shape[0])
        if args.monthly:
            for m in range(nm):
                tt, rr = t[m], r[m]
                g = np.isfinite(tt) & np.isfinite(rr) & (np.abs(tt) < 1e30) & (np.abs(rr) < 1e30)
                if g.sum() < 100:
                    continue
                corr = np.corrcoef(tt[g], rr[g])[0, 1]
                d = tt[g] - rr[g]
                print(f"  {var:5s} m{m+1:02d} corr={corr:8.5f} meanΔ={d.mean():+.5f} "
                      f"|Δ|max={np.abs(d).max():.4f}")
        ta, ra = t[:nm].mean(axis=0), r[:nm].mean(axis=0)
        g = np.isfinite(ta) & np.isfinite(ra) & (np.abs(ta) < 1e30) & (np.abs(ra) < 1e30)
        corr = np.corrcoef(ta[g], ra[g])[0, 1]
        d = ta[g] - ra[g]
        print(f"{var:6s} {corr:12.5f} {d.mean():+10.5f} {np.abs(d).max():10.4f}")


if __name__ == "__main__":
    main()
