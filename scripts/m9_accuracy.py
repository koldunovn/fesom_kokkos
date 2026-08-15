#!/usr/bin/env python3
"""M9 P1 — what does the lagged mEVP halo (cell ⑤) cost in simulated climate?

Pre-registration: docs/plans/20260805-m9-PREREG-P1.md, committed before the first leg ran.
It fixes the metrics, the yardstick and the decision rule; this script only applies them.

WHY THIS EXISTS AND m32_climate_compare.py DOES NOT SUFFICE
    Session 1 measured max|Δ| at 300 steps and got a table with no monotone trend in K, every
    large value resting on 1-5 nodes of 126858 — chaotic ice-edge divergence, not a K-dependent
    bias. max|Δ| cannot measure this approximation. The honest metrics are pattern statistics
    (corr / RMS / bias, which 1-5 points cannot carry) plus the integrated ice quantities a sea
    ice modeller actually reads: hemispheric area and extent. max|Δ| is still printed, and is
    still not a criterion.

THE YARDSTICK (docs/REFERENCE_RUNS.md:224-233, measured 2026-07-12, not invented here)
    mEVP's own C-port-vs-Fortran floor: two INDEPENDENT CORRECT implementations of the same
    scheme, same mesh, same config, same year. mEVP is a fixed-point iteration (alpha=beta=250)
    that is only approximately converged, so its ice velocity does not reproduce better than
    ~0.95 across implementations while its SST reproduces at 1.00000. Cell ⑤ changes how
    converged that same iteration is, so this is the right bar — and a demanding one.

VECTOR FRAME
    Every leg here is the same binary, same backend, same rotated native frame, so uice/vice
    are compared AS WRITTEN. No rotation is applied and none is needed. (Rotation matters only
    against Fortran/modern-C references — use m32_climate_compare.py for that; L74 is what
    happens when the distinction is missed.)

usage:
  m9_accuracy.py --ref <classic_dir> --legs lag2=<dir> lag4=<dir> ... [--year 1958]
                 [--mesh /work/ab0995/a270088/port2/mesh/core2] [--label CPU]
"""
import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset

# The mEVP scheme's own reproducibility floor. NOT a tolerance we chose — a measurement.
#
# ⚠️ IT IS QUOTED TO 5 DECIMALS, so it must be COMPARED at 5 decimals. Testing a computed
# correlation against a literal 1.00000 with `>=` marks everything that is not bitwise equal as
# a failure, including a leg that differs from the reference only by CUDA's own run-to-run
# atomic ordering. The GPU control leg caught this on the first run: `classic_rep` — the same
# binary, same knobs, same everything, differing only in atomic scheduling — "failed" sst and
# ssh at corr 1.000000. A bar that a run fails against ITSELF is a broken bar, not a result.
FLOOR = {"sst": 1.00000, "ssh": 1.00000, "a_ice": 0.99999,
         "m_ice": 0.99998, "uice": 0.95438, "vice": 0.93908}
FLOOR_DP = 5          # the precision the floor table is published at
FIELDS = ("sst", "sss", "ssh", "a_ice", "m_ice", "uice", "vice")
ICE_FIELDS = {"a_ice", "m_ice", "m_snow", "uice", "vice"}
EXTENT_THRESH = 0.15          # the standard sea-ice extent definition


def monthly(run_dir, var, year):
    """(12, nod) monthly means. Ice fields: NaN -> 0 per month BEFORE any temporal mean —
    open water is ZERO ice, not missing data ([[feedback-ice-mask-averaging]])."""
    path = os.path.join(run_dir, f"{var}.fesom.{year}.monthly.nc")
    with Dataset(path) as nc:
        v = np.asarray(nc.variables[var][:], dtype=np.float64)
    fill = 0.0 if var in ICE_FIELDS else np.nan
    return np.ma.filled(np.ma.masked_invalid(v), fill)


def stats(a, b):
    """Pattern stats of annual means, a vs b. Returns corr, rms, bias, max|d|."""
    a = np.nanmean(a, axis=0)
    b = np.nanmean(b, axis=0)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    # A constant field (or an exactly identical pair) has zero variance and np.corrcoef
    # returns nan. Bit-identical is the STRONGEST possible agreement, so report it as 1.0
    # rather than letting a nan read as a failure.
    if np.all(d == 0.0):
        return 1.0, 0.0, 0.0, 0.0
    sa, sb = a.std(), b.std()
    corr = float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else float("nan")
    return corr, float(np.sqrt((d * d).mean())), float(d.mean()), float(np.abs(d).max())


def ice_series(run_dir, year, area, lat):
    """NH/SH monthly ice AREA (sum a*A) and EXTENT (sum A where a>0.15), in 10^6 km^2."""
    a_ice = monthly(run_dir, "a_ice", year)
    out = {}
    for hemi, sel in (("NH", lat > 0), ("SH", lat < 0)):
        A = area[sel]
        a = a_ice[:, sel]
        out[f"{hemi}_area"] = (a * A).sum(axis=1) / 1e12
        out[f"{hemi}_extent"] = ((a > EXTENT_THRESH) * A).sum(axis=1) / 1e12
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the classic (reference) leg directory")
    ap.add_argument("--legs", nargs="+", required=True, help="name=dir ...")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--mesh", default="/work/ab0995/a270088/port2/mesh/core2")
    ap.add_argument("--label", default="")
    ap.add_argument("--control", default="",
                    help="name of the replicate leg (same knobs as --ref). On a nondeterministic "
                         "backend this is the noise floor every other leg must be read against.")
    args = ap.parse_args()

    with Dataset(os.path.join(args.mesh, "core2_griddes_nodes.nc")) as nc:
        area = np.asarray(nc.variables["cell_area"][:], dtype=np.float64)
        lat = np.asarray(nc.variables["lat"][:], dtype=np.float64)

    legs = []
    for spec in args.legs:
        name, _, d = spec.partition("=")
        if not d or not os.path.isdir(d):
            sys.exit(f"bad --legs entry (expected name=dir): {spec}")
        legs.append((name, d))

    print(f"M9 P1 accuracy — {args.label or 'legs'} vs classic, year {args.year}")
    print(f"  reference : {args.ref}")
    print(f"  yardstick : mEVP's own C-vs-Fortran floor (docs/REFERENCE_RUNS.md:227)")
    print(f"  ⚠️ max|Δ| is REPORTED, NOT a criterion (pre-registered — session 1 showed it")
    print(f"     cannot measure this approximation: 1-5 chaotic ice-edge nodes carry it).")
    print()

    hdr = f"{'leg':<12} {'field':<7} {'corr':>10} {'floor':>9} {'verdict':>9} " \
          f"{'RMS':>11} {'bias':>11} {'max|Δ|':>10}"
    print(hdr)
    print("-" * len(hdr))

    worst = {}
    corr = {}
    for name, d in legs:
        for f in FIELDS:
            try:
                c, r, b, m = stats(monthly(d, f, args.year), monthly(args.ref, f, args.year))
            except (OSError, KeyError) as e:
                print(f"{name:<12} {f:<7}  MISSING ({e.__class__.__name__})")
                continue
            corr[(name, f)] = c
            fl = FLOOR.get(f)
            if fl is None:
                verdict = "  --"                      # sss has no published floor
            elif not np.isfinite(c):
                verdict = "  n/a"
            else:
                # Compare at the precision the floor is published at (see FLOOR_DP).
                verdict = "PASS" if round(c, FLOOR_DP) >= fl else "**FAIL**"
                if name != args.control and c < worst.get(f, (2.0,))[0]:
                    worst[f] = (c, name)
            fls = f"{fl:.5f}" if fl is not None else "   --   "
            print(f"{name:<12} {f:<7} {c:10.6f} {fls:>9} {verdict:>9} "
                  f"{r:11.3e} {b:11.3e} {m:10.3e}")
        print()

    print("worst correlation per LEVER leg (the number that decides the verdict):")
    for f in FIELDS:
        if f not in worst:
            continue
        c, n = worst[f]
        fl = FLOOR[f]
        # Report DEPARTURES (1-corr), not ratios of them: with a floor published as 1.00000 the
        # ratio is either 0 or infinite and says nothing. The departure is what is comparable.
        line = f"  {f:<7} {c:.6f} (leg {n}); departure {1.0-c:.2e}, floor departure {1.0-fl:.2e}"
        if args.control and (args.control, f) in corr:
            cc = corr[(args.control, f)]
            line += f", CONTROL departure {1.0-cc:.2e}"
            # The only honest GPU statement: is the lever's departure distinguishable from the
            # departure two identical runs already show?
            if 1.0 - cc > 0:
                line += f"  -> lever/control = {(1.0-c)/(1.0-cc):.2f}x"
        print(line)
    if args.control:
        print(f"\n  ⚠️ '{args.control}' is the SAME configuration as the reference — its departure is")
        print(f"     pure backend nondeterminism (D22). Where a lever's departure is the same order,")
        print(f"     THIS ARM CANNOT RESOLVE THE APPROXIMATION and the deterministic arm decides.")
    print()

    print("hemispheric sea ice, monthly (10^6 km^2) — max |leg − classic| over the 12 months,")
    print("with the classic leg's own annual range for scale:")
    ref_s = ice_series(args.ref, args.year, area, lat)
    print(f"{'leg':<12} {'NH area':>12} {'NH extent':>12} {'SH area':>12} {'SH extent':>12}")
    for name, d in legs:
        s = ice_series(d, args.year, area, lat)
        row = [np.abs(s[k] - ref_s[k]).max() for k in
               ("NH_area", "NH_extent", "SH_area", "SH_extent")]
        print(f"{name:<12} " + " ".join(f"{v:12.5f}" for v in row))
    print(f"{'(classic)':<12} " + " ".join(
        f"{ref_s[k].min():5.2f}-{ref_s[k].max():<6.2f}" for k in
        ("NH_area", "NH_extent", "SH_area", "SH_extent")))


if __name__ == "__main__":
    main()
