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
ICE_FIELDS = {"a_ice", "m_ice", "m_snow", "uice", "vice"}  # need NaN→0 BEFORE temporal mean ([[feedback-ice-mask-averaging]])
# ⚠️ uice/vice ADDED 2026-05-30: Fortran masks ice-free water as NaN, the C/Kokkos port writes 0 there
# (73.5% of nodes on CORE2). WITHOUT the per-month NaN→0, the Fortran annual nanmean drops ice-free months
# while the port's mean includes the zeros → a SPURIOUS uice-vs-Fortran decorrelation (the marginal-ice
# artifact) that hit ONLY the Fortran comparison (C/Kokkos share the 0-convention). It deflated uice-vs-
# Fortran to 0.850; with the fix it read 0.919.
# ⚠️ CORRECTION 2026-07-12 (M6 Task 0.2): that residual 0.919 was previously written up here as "a REAL
# (modest) C-port-vs-Fortran ice-velocity/EVP difference". IT IS NOT. It was the VECTOR-FRAME mismatch
# (see below): with the r2g rotation applied, uice-vs-Fortran is 0.9997 and vice-vs-Fortran is 0.9998.
# The port reproduces Fortran's ice velocity essentially perfectly; there is no ice-edge budget. The
# "~0.92" quoted in docs/GPU_FIDELITY §M5.13–§M5.15 is superseded — read it as ~1.0.
FIELDS = ("sst", "sss", "ssh", "a_ice", "m_ice", "uice", "vice")   # surface climate + ice drift
# ⚠️ vice ADDED 2026-07-12 (M6 Task 0.2). It was never compared — which is exactly why the
# VECTOR-FRAME mismatch below went unnoticed for the whole M5 campaign: uice degraded to a
# plausible-looking 0.92, but vice was at 0.43 and nobody was looking at it.

# ---- VECTOR FRAME (M6 Task 0.2, 2026-07-12) --------------------------------------------
# The port writes (u,v)/(uice,vice) in the model's native ROTATED frame. Fortran ALWAYS
# writes geographic (io_meandata rotates); the C port writes geographic from commit 75406d3
# on, rotated before it. Comparing across frames is an isometry, so |speed|/extent/volume
# look perfect while the COMPONENTS decorrelate — a silent, plausible-looking wrong answer.
# Everything rotated is now brought to GEOGRAPHIC before comparison (scalars are frame-free).
# Measured impact on the M5.23 CUDA 1-yr run vs the Fortran linfs+KPP reference:
#     uice  0.9187 -> 0.9997      vice  0.4266 -> 0.9998
# The 0.919 was on record here as the "known F<->C ice-edge budget". It was not physics.
from fesom_frame import Rotator, VECTOR_PAIRS, DEFAULT_MESH   # noqa: E402

def surf_annual(path, var, months=False):
    """Annual mean of surface field. For ice fields, nan_to_num per month BEFORE the temporal
    mean (Fortran masks open water as NaN, C writes 0 — without the per-month nan->0 the
    temporal nanmean drops open-water months at marginal-ice nodes and spuriously inflates
    the Fortran mean → fake CUDA-vs-Fortran bias). See feedback-ice-mask-averaging memory.
    months=True returns the (12, nod2D) monthly stack instead (needed to rotate vector pairs
    before averaging — rotation is linear, so either order works, but the nan->0 must come
    first either way)."""
    d = nc.Dataset(path); a = d.variables[var]
    x = np.asarray(a[:, 0, :]) if a.ndim == 3 else np.asarray(a[:])   # (t,nz,n)->surf or (t,n)
    d.close()
    x = np.where(np.abs(x) < 1e30, x, np.nan)
    if var in ICE_FIELDS:
        x = np.nan_to_num(x, nan=0.0)
    return x if months else np.nanmean(x, axis=0)      # annual mean per node

def load_field(dir_, var, yr, suffix, frame, rot):
    """Annual-mean surface field, rotated to GEOGRAPHIC if it is half of a vector pair and
    the source writes the rotated frame. Scalars are returned untouched."""
    partner = VECTOR_PAIRS.get(var) or next((k for k, v in VECTOR_PAIRS.items() if v == var), None)
    if partner is None or frame == "geo":
        return surf_annual(f"{dir_}/{var}.fesom.{yr}.{suffix}", var)
    a = surf_annual(f"{dir_}/{var}.fesom.{yr}.{suffix}", var, months=True)
    b = surf_annual(f"{dir_}/{partner}.fesom.{yr}.{suffix}", partner, months=True)
    u, v = (a, b) if var in VECTOR_PAIRS else (b, a)    # (u,v) order
    ug, vg = rot.r2g(u, v)
    out = ug if var in VECTOR_PAIRS else vg
    return np.nanmean(out, axis=0)

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
ap.add_argument("--cref-frame", choices=("geo", "rotated"), default="rotated",
                help="vector frame of the C reference's (u,v)/(uice,vice). C outputs from "
                     "commit 75406d3 (2026-06-11 20:37) on are 'geo'; older ones 'rotated'. "
                     "Default 'rotated' matches the KPP ref. See scripts/fesom_frame.py.")
ap.add_argument("--mesh", default=DEFAULT_MESH,
                help=f"mesh dir for the r2g rotation (default {DEFAULT_MESH})")
ap.add_argument("--backend-frame", choices=("geo", "rotated"), default="rotated",
                help="vector frame of the BACKEND dir. The Kokkos port is always 'rotated' (no "
                     "frame knob), which is the default. Set 'geo' to feed a C-port run in as the "
                     "backend -- that is how you measure the C-vs-Fortran BASELINE for a scheme, "
                     "which is what a backend-vs-C number has to be judged against.")
args = ap.parse_args()

rot = Rotator(args.mesh)
# Fortran is ALWAYS geographic; the Kokkos port is ALWAYS rotated (it has no frame knob).
FRAME = {"Fortran": "geo", "C-port": args.cref_frame}
KK_FRAME = args.backend_frame

print(f"M3.2 climate validation — backend={args.label}  dir={args.backend_dir}")
print(f"  Fortran ref: {args.fref}  (vectors: geo)")
print(f"  C-port ref:  {args.cref}  (vectors: {args.cref_frame})")
print(f"  backend vectors: {KK_FRAME}; mesh {args.mesh}")
print("  Vector pairs are rotated to GEOGRAPHIC before comparison (scripts/fesom_frame.py).")
print("  PASS = corr~1, bias/RMS bounded & non-growing; backend-vs-C (the scatter drift) <= C-vs-Fortran.\n")

for ref_name, ref_dir, ref_suffix in (("Fortran", args.fref, "nc"), ("C-port", args.cref, "monthly.nc")):
    print(f"================= {args.label} vs {ref_name} =================")
    print(f"{'field':6s} {'year':4s} | {'corr':>8s} {'bias':>12s} {'RMS':>11s} {'|d|max':>10s}")
    first_bias, last_bias = {}, {}
    for var in FIELDS:
        for yr in args.years:
            try:
                kk = load_field(args.backend_dir, var, yr, "monthly.nc", KK_FRAME, rot)
                rf = load_field(ref_dir, var, yr, ref_suffix, FRAME[ref_name], rot)
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
