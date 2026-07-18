#!/usr/bin/env python3
"""Verification figures: a Kokkos-port climate run vs the Fortran R2 hindcast.

usage: m7_climate_check_plots.py <run_dir> <outdir> [--years 1958 1959]
       [--fref /work/ab0995/a270088/fesom2_core2]
       [--mesh /work/ab0995/a270088/port2/mesh/core2]

Reads <run_dir>/<var>.fesom.<yr>.monthly.nc (ours) and <fref>/<var>.fesom.<yr>.nc
(Fortran). Scalars only (no vector-frame issues). Produces per year:
  sst_<yr>.png, sss_<yr>.png   3-panel scatter maps: ours | Fortran | ours-Fortran
  aice_<yr>.png                March + September, ours vs Fortran vs diff
and across all years:
  series_global.png            monthly node-mean sst/sss + polar a_ice means, both models
Node-mean (unweighted) is a QC proxy, not the paper reduction — good enough to catch
any parameter/config mismatch (wrong GM/ice_diff/mixing shows up at a glance).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

import nereus as nr   # the JAX-paper plotting pipeline — same look, comparable figures

# fixed diff scales per variable (paper convention: tight, FIXED — never autoscaled)
DIFF_LIM = {"sst": 0.25, "sss": 0.10, "a_ice": 0.20}
_INTERP = [None]   # nereus interpolator, built once and reused across every panel


def month_data(path, var):
    with Dataset(path) as nc:
        v = nc.variables[var][:]                # (12, nod); Fortran files mask no-ice/land
    fill = 0.0 if var == "a_ice" else np.nan    # missing ice IS zero ice; else NaN-out
    return np.ma.filled(np.ma.masked_invalid(v), fill)


def tri_panel(lon, lat, a, b, title_a, title_b, fname, units="", var="sst"):
    d = a - b
    rms = float(np.sqrt(np.nanmean(d * d)))
    dmax = float(np.nanmax(np.abs(d)))
    dlim = DIFF_LIM.get(var, max(rms * 5, 1e-6))
    vmin, vmax = np.nanpercentile(np.concatenate([a, b]), [1, 99])
    proj = nr.plotting.get_projection("rob")
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 12),
                            subplot_kw={"projection": proj}, constrained_layout=True)
    for ax, fld, ttl, cmap, vmi, vma in (
        (axs[0], a, title_a, "viridis", vmin, vmax),
        (axs[1], b, title_b, "viridis", vmin, vmax),
        (axs[2], d, f"{title_a} − {title_b}", "RdBu_r", -dlim, dlim),
    ):
        _, _, _INTERP[0] = nr.plot(fld, lon, lat, projection="rob", method="linear",
                                   ax=ax, interpolator=_INTERP[0], colorbar=False,
                                   cmap=cmap, vmin=vmi, vmax=vma,
                                   land=True, coastlines=True)
        ax.set_title(ttl, fontsize=10)
    axs[2].text(0.5, -0.06, f"RMS = {rms:.4f} {units}   max|d| = {dmax:.3f}",
                transform=axs[2].transAxes, ha="center", va="top", fontsize=9)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap="viridis")
    fig.colorbar(sm, ax=[axs[0], axs[1]], shrink=0.8, pad=0.02, label=units)
    smd = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-dlim, dlim), cmap="RdBu_r")
    fig.colorbar(smd, ax=axs[2], shrink=0.8, pad=0.02, label=units)
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print("wrote", fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("outdir")
    ap.add_argument("--years", nargs="+", type=int, default=[1958, 1959])
    ap.add_argument("--fref", default="/work/ab0995/a270088/fesom2_core2")
    ap.add_argument("--mesh", default="/work/ab0995/a270088/port2/mesh/core2")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    xy = np.loadtxt(f"{a.mesh}/nod2d.out", skiprows=1, usecols=(1, 2))
    lon, lat = xy[:, 0], xy[:, 1]
    nh, sh = lat > 50.0, lat < -50.0

    series = {k: {"ours": [], "fort": []} for k in ("sst", "sss", "aice_nh", "aice_sh")}
    for yr in a.years:
        ours = {v: month_data(f"{a.run_dir}/{v}.fesom.{yr}.monthly.nc", v)
                for v in ("sst", "sss", "a_ice")}
        fort = {v: month_data(f"{a.fref}/{v}.fesom.{yr}.nc", v)
                for v in ("sst", "sss", "a_ice")}
        for v, units in (("sst", "degC"), ("sss", "psu")):
            tri_panel(lon, lat, ours[v].mean(0), fort[v].mean(0),
                      f"Kokkos {v} {yr}", f"Fortran {v} {yr}",
                      f"{a.outdir}/{v}_{yr}.png", units, var=v)
        for mi, mon in ((2, "March"), (8, "September")):
            tri_panel(lon, lat, ours["a_ice"][mi], fort["a_ice"][mi],
                      f"Kokkos a_ice {mon} {yr}", f"Fortran a_ice {mon} {yr}",
                      f"{a.outdir}/aice_{mon.lower()[:3]}_{yr}.png", "frac", var="a_ice")
        for m in range(12):
            series["sst"]["ours"].append(ours["sst"][m].mean());  series["sst"]["fort"].append(fort["sst"][m].mean())
            series["sss"]["ours"].append(ours["sss"][m].mean());  series["sss"]["fort"].append(fort["sss"][m].mean())
            series["aice_nh"]["ours"].append(ours["a_ice"][m][nh].mean()); series["aice_nh"]["fort"].append(fort["a_ice"][m][nh].mean())
            series["aice_sh"]["ours"].append(ours["a_ice"][m][sh].mean()); series["aice_sh"]["fort"].append(fort["a_ice"][m][sh].mean())

    t = np.arange(len(series["sst"]["ours"])) / 12.0 + a.years[0]
    fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
    for ax, k, ttl in ((axs[0], "sst", "node-mean SST [degC]"),
                       (axs[1], "sss", "node-mean SSS [psu]"),
                       (axs[2], "aice_nh", "mean a_ice, lat>50N"),
                       (axs[3], "aice_sh", "mean a_ice, lat<50S")):
        ax.plot(t, series[k]["ours"], label="Kokkos", lw=1.6)
        ax.plot(t, series[k]["fort"], label="Fortran R2", lw=1.2, ls="--")
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
    axs[0].legend(); axs[-1].set_xlabel("year")
    fig.savefig(f"{a.outdir}/series_global.png", dpi=110)
    print("wrote", f"{a.outdir}/series_global.png")


if __name__ == "__main__":
    main()
