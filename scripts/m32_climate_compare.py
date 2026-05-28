#!/usr/bin/env python3
"""M3.2 — climate validation of a Kokkos backend (CUDA or OpenMP) over a multi-year CORE2 run.

Serial is bit-identical to the C twin (M4 acceptance); CUDA/OpenMP are CLIMATE-CLOSE (D22): the ice
EVP/FCT `atomic_add` scatters + the oce_fluxes / CG `parallel_reduce` reassociate across threads/lanes
and compound over the run. This script quantifies that compounding via annual-mean surface stats
(corr / bias / RMS / |Δ|max) per year + a year-to-year DRIFT check, against TWO references:

  * Fortran KPP (`fortran_kpp_5yr_fix`) — "does the GPU reproduce the science?" budget
    (already includes the Fortran<->C-at-KPP budget; see commit 375f3eb "C+KPP reproduces F+KPP").
  * the C-port KPP rebase (`kpp_2yr_rebase`) — backend-vs-C ISOLATES the GPU/OpenMP scatter/reduce
    drift (the C twin == Serial == bit-identical). PASS = backend-vs-C is small & non-growing,
    and clearly BELOW the (larger) C-vs-Fortran budget. corr~1.0, bias/RMS bounded, drift ~0.

The default references match the Kokkos port HEAD physics config (KPP + ice_gamma_fct=0.5; the C-port
default flipped to KPP @ 8d0cdbc and γ was fixed @ 7c6663b — see docs/REFERENCE_RUNS.md for the
full ref-provenance table). If you ran the backend with a different physics config (e.g.
FESOM_MIX_SCHEME=PP), override --cref/--fref to point at the matching reference run.

Run (after the M3.2 run produces <var>.fesom.<yr>.monthly.nc in the backend dir):
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/m32_climate_compare.py \
      /work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_1yr_pin --label CUDA --years 1958
Record the table in docs/GPU_FIDELITY.md.
"""
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc

# Defaults: matched to the Kokkos port HEAD (KPP, gamma=0.5). See docs/REFERENCE_RUNS.md.
FORT_DEFAULT = "/scratch/a/a270088/fortran_kpp_5yr_fix"   # <var>.fesom.<yr>.nc        (Fortran-KPP, γ=0.5)
CREF_DEFAULT = "/work/ab0995/a270088/port/kpp_5yr_fix"    # <var>.fesom.<yr>.monthly.nc (C-port-KPP @ 6ecabe8, γ=0.5, 5 yr)
ICE_FIELDS = {"a_ice", "m_ice", "m_snow"}                  # need NaN→0 BEFORE temporal mean ([[feedback-ice-mask-averaging]])
FIELDS = ("sst", "sss", "ssh", "a_ice", "m_ice", "uice")   # surface climate + ice drift; missing ones skip

def surf_annual(path, var):
    """Annual mean of surface field. For ice fields, nan_to_num per month BEFORE the temporal
    mean (Fortran masks open water as NaN, C writes 0 — without the per-month nan->0 the
    temporal nanmean drops open-water months at marginal-ice nodes and spuriously inflates
    the Fortran mean → fake CUDA-vs-Fortran bias). See feedback-ice-mask-averaging memory."""
    d = nc.Dataset(path); a = d.variables[var]
    x = np.asarray(a[:, 0, :]) if a.ndim == 3 else np.asarray(a[:])   # (t,nz,n)->surf or (t,n)
    d.close()
    x = np.where(np.abs(x) < 1e30, x, np.nan)
    if var in ICE_FIELDS:
        x = np.nan_to_num(x, nan=0.0)
    return np.nanmean(x, axis=0)                       # annual mean per node

def stats(a, b):
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() == 0:
        return None
    d = a[g] - b[g]
    corr = np.corrcoef(a[g], b[g])[0, 1] if (a[g].std() > 0 and b[g].std() > 0) else float("nan")
    return corr, float(d.mean()), float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max())

ap = argparse.ArgumentParser()
ap.add_argument("backend_dir", help="dir with <var>.fesom.<yr>.monthly.nc from the M3.2 run")
ap.add_argument("--label", default="KK")
ap.add_argument("--years", nargs="+", type=int, default=[1958, 1959])
ap.add_argument("--cref", default=CREF_DEFAULT,
                help=f"C-port reference dir (default {CREF_DEFAULT} = C-port KPP, γ=0.5)")
ap.add_argument("--fref", default=FORT_DEFAULT,
                help=f"Fortran reference dir (default {FORT_DEFAULT} = Fortran KPP)")
args = ap.parse_args()

print(f"M3.2 climate validation — backend={args.label}  dir={args.backend_dir}")
print(f"  Fortran ref: {args.fref}\n  C-port ref:  {args.cref}")
print("  PASS = corr~1, bias/RMS bounded & non-growing; backend-vs-C (the scatter drift) <= C-vs-Fortran.\n")

for ref_name, ref_dir, ref_suffix in (("Fortran", args.fref, "nc"), ("C-port", args.cref, "monthly.nc")):
    print(f"================= {args.label} vs {ref_name} =================")
    print(f"{'field':6s} {'year':4s} | {'corr':>8s} {'bias':>12s} {'RMS':>11s} {'|d|max':>10s}")
    first_bias, last_bias = {}, {}
    for var in FIELDS:
        for yr in args.years:
            try:
                kk = surf_annual(f"{args.backend_dir}/{var}.fesom.{yr}.monthly.nc", var)
                rf = surf_annual(f"{ref_dir}/{var}.fesom.{yr}.{ref_suffix}", var)
            except Exception as e:
                print(f"  {var} {yr}: skip ({type(e).__name__}: {e})"); continue
            s = stats(kk, rf)
            if s is None:
                print(f"  {var} {yr}: skip (no overlap)"); continue
            print(f"{var:6s} {yr}   | {s[0]:8.5f} {s[1]:+12.4e} {s[2]:11.4e} {s[3]:10.3e}")
            if var not in first_bias: first_bias[var] = s[1]
            last_bias[var] = s[1]
        print()
    print("  DRIFT (last-year bias − first-year bias; ~0 = stable, no runaway):")
    for var in FIELDS:
        if var in first_bias and var in last_bias and len(args.years) >= 2:
            print(f"    {var:6s} {last_bias[var] - first_bias[var]:+.4e}")
    print()
