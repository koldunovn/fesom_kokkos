#!/usr/bin/env python
"""m15_precond_drift.py — the JAX-paper F4 (drift/heat-content) figure for the
M15 60-yr Fortran CORE2 preconditioner twins: precond_variant=0 (as coded) vs
precond_variant=1 (FESOM/fesom2#984 symmetric Jacobi).

Both arms are the SAME binary (fesom27 4ccdd35a + the #984 backport) on the same
mesh/forcing/IC; the only difference in the two run dirs is that one namelist
integer. So every curve separation here is the preconditioner and nothing else.

Reuses the paper pipeline's mesh/reduction machinery (paper_jax/scripts/common.py)
verbatim; only the file iteration differs (our runs write {var}.fesom.YYYY.monthly.nc,
Fortran {var}.fesom.YYYY.nc — both 12 monthly records/file, same node order, same
CORE2 mesh diag). Only COMPLETE years (12 records) enter — this implements the
rule-0.40 partial-final-year trim automatically, so the same command works for the
preliminary (runs in flight) and final harvests.

  $NEREUS_PYTHON scripts/m7_hindcast_drift.py [--reduce] [--outdir ...]

Reduction (~174 year-files) takes a few minutes; the result is cached in
<outdir>/drift_precond.nc and --reduce forces a refresh.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import paths    # noqa: E402
import common   # noqa: E402

CV0, CV1 = "#1f77b4", "#d62728"
DEPTH_TICKS = [0, 50, 100, 250, 500, 1000, 2000, 4000, 6000]
M15 = "/work/ab0995/a270088"
RUNS = {   # label -> (root, filename suffix after {var}.fesom.YYYY)
    "v0": (f"{M15}/fesom2_precond_v0", ".nc"),
    "v1": (f"{M15}/fesom2_precond_v1", ".nc"),
}


def complete_years(root, var, suffix):
    """Years whose {var}.fesom.YYYY{suffix} exists AND holds 12 monthly records
    (drops the in-flight/guillotined partial final year = rule 0.40)."""
    import xarray as xr
    rx = re.compile(rf"^{re.escape(var)}\.fesom\.(\d{{4}}){re.escape(suffix)}$")
    yrs = []
    for p in sorted(glob.glob(os.path.join(root, f"{var}.fesom.*{suffix}"))):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        with xr.open_dataset(p) as ds:
            if ds[var].sizes["time"] == 12:
                yrs.append(int(m.group(1)))
    return sorted(yrs)


def iter_months(root, var, suffix, years, pad_to_nz):
    """(decimal_year, field) per monthly record, chronological; 3-D fields padded to
    pad_to_nz interfaces with NaN layers (mirrors common.fortran_iter_months; the
    3-D depth dim is matched by name prefix so both 'nz1' and our 'nz_1' work)."""
    import xarray as xr
    for yr in years:
        with xr.open_dataset(f"{root}/{var}.fesom.{yr}{suffix}") as ds:
            v = ds[var]
            is3d = any(d.startswith("nz") for d in v.dims)
            for it in range(v.sizes["time"]):
                f = np.asarray(v.isel(time=it).values, float)
                if is3d and pad_to_nz is not None and f.shape[0] < pad_to_nz:
                    f = np.concatenate(
                        [f, np.full((pad_to_nz - f.shape[0], f.shape[1]), np.nan)], axis=0)
                yield yr + (it + 0.5) / 12.0, f


def reduce_run(label, mesh, zbar, nz, nlay, root, suffix, ylast):
    yrs = [y for y in complete_years(root, "temp", suffix) if y <= ylast]
    print(f"[reduce] {label}: {len(yrs)} complete years ({yrs[0]}-{yrs[-1]})")
    t, cols = [], {k: [] for k in ("sst", "sss", "tbar", "tbar700", "sbar", "ohc")}
    tz, sz = [], []
    tgen = iter_months(root, "temp", suffix, yrs, nz)
    sgen = iter_months(root, "salt", suffix, yrs, nz)
    for (dy, T), (dy2, S) in zip(tgen, sgen):
        assert abs(dy - dy2) < 1e-9, f"{label}: temp/salt month misalignment"
        t.append(dy)
        cols["sst"].append(common.area_weighted_mean(T[0], mesh.area2d))
        cols["sss"].append(common.area_weighted_mean(S[0], mesh.area2d))
        cols["tbar"].append(common.vol_weighted_mean(T, mesh.area3d, zbar))
        cols["tbar700"].append(common.vol_weighted_mean(T, mesh.area3d, zbar, depth_max=700.0))
        cols["sbar"].append(common.vol_weighted_mean(S, mesh.area3d, zbar))
        cols["ohc"].append(common.ocean_heat_content(T, mesh.area3d, zbar) / 1e21)  # ZJ
        tz.append(common.level_area_mean(T, mesh.area3d)[:nlay])
        sz.append(common.level_area_mean(S, mesh.area3d)[:nlay])
    out = {k: np.asarray(v, np.float64) for k, v in cols.items()}
    out["t"] = np.asarray(t)
    out["tz"] = np.asarray(tz, np.float64)
    out["sz"] = np.asarray(sz, np.float64)
    print(f"[reduce] {label}: Tbar {out['tbar'][0]:.4f}->{out['tbar'][-1]:.4f}  "
          f"OHC {out['ohc'][0]:.1f}->{out['ohc'][-1]:.1f} ZJ  ({len(t)} months)")
    return out


def reduce_all(ncpath):
    import xarray as xr
    mesh = common.load_mesh(paths.mesh_diag("core2"))
    zbar = mesh.zbar
    nz = zbar.size
    nlay = common.layer_thickness(zbar).size
    zmid = 0.5 * (zbar[:-1] + zbar[1:])[:nlay]
    data = {}
    for label, (root, suffix) in RUNS.items():
        data[label] = reduce_run(label, mesh, zbar, nz, nlay, root, suffix, 9999)
    dv, coords = {}, {"z": zmid.astype(np.float64)}
    for label, d in data.items():
        td = f"time_{label}"
        coords[td] = d["t"]
        for k in ("sst", "sss", "tbar", "tbar700", "sbar", "ohc"):
            dv[f"{k}_{label}"] = (td, d[k])
        dv[f"tz_{label}"] = ((td, "z"), d["tz"])
        dv[f"sz_{label}"] = ((td, "z"), d["sz"])
    ds = xr.Dataset(dv, coords=coords,
                    attrs=dict(mesh="core2", note="complete years only (rule 0.40)",
                               ohc_units="ZJ (1e21 J, ref 0 degC)"))
    ds.to_netcdf(ncpath, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"[reduce] wrote {ncpath}")


def running_mean(x, w=12):
    import pandas as pd
    return pd.Series(np.asarray(x, float)).rolling(
        w, center=True, min_periods=max(1, w // 2)).mean().to_numpy()


def _sqrt_depth(ax):
    from matplotlib.scale import FuncScale
    ax.set_yscale(FuncScale(ax, (lambda x: np.sqrt(np.maximum(x, 0.0)),
                                 lambda x: np.asarray(x) ** 2)))
    ax.set_yticks(DEPTH_TICKS)
    ax.set_ylabel("depth [m]")


def _gap_at_common_end(ds, base, run):
    """|run − v0| at the last month both share (mid-month key match)."""
    tf = np.asarray(ds["time_v0"].values)
    tr = np.asarray(ds[f"time_{run}"].values)
    idx = common.align_index(tf, tr)
    keep = np.where(idx >= 0)[0]
    if not keep.size:
        return np.nan
    i = keep[-1]
    return abs(float(ds[f"{base}_{run}"].values[i])
               - float(ds[f"{base}_v0"].values[idx[i]]))


def figure(ncpath, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import nereus as nr
    import xarray as xr
    common.set_style()
    ds = xr.open_dataset(ncpath)
    z = ds["z"].values
    # COMMON WINDOW: every run truncated to the smallest shared last year, so all
    # panels (and especially the end−start profiles) compare identical months —
    # different end years would put different interannual states into the upper
    # ocean and masquerade as model differences.
    ycom = min(int(ds[f"time_{r}"].values[-1]) for r in ("v0", "v1"))
    T, D = {}, {}
    for r in ("v0", "v1"):
        t = ds[f"time_{r}"].values
        m = t < ycom + 1
        T[r] = t[m]
        D[r] = {k: ds[f"{k}_{r}"].values[m]
                for k in ("sst", "sss", "tbar", "tbar700", "sbar", "ohc", "tz", "sz")}
    MODELS = [("v0", CV0, "precond 0 (as coded)"), ("v1", CV1, "precond 1 (#984)")]

    fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axT, axS, axO = (fig.add_subplot(gs[0, i]) for i in range(3))
    axHT, axHS, axP = (fig.add_subplot(gs[1, i]) for i in range(3))

    # (a) volume-mean T: colour = model, solid = full depth, dashed = 0-700 m.
    # Draw order thick-to-thin (v0 under, v1 over) so identical curves
    # read as a layered line instead of one model hiding the others.
    LW = {"v0": 2.4, "v1": 1.1}
    for r, c, lab in reversed(MODELS):
        axT.plot(T[r], D[r]["tbar"], color=c, lw=LW[r])
        axT.plot(T[r], D[r]["tbar700"], color=c, ls="--", lw=LW[r] * 0.8)
    axT.set_ylabel("volume-mean T [°C]"); axT.set_xlabel("year")
    axT.set_title("(a) global mean ocean temperature")
    axT.legend(handles=[Line2D([], [], color=c, label=lab) for _, c, lab in MODELS] +
               [Line2D([], [], color="0.35", ls="-", label="full depth"),
                Line2D([], [], color="0.35", ls="--", label="0–700 m")],
               fontsize=7, ncol=2, loc="center left")
    axT.text(0.98, 0.04, "|v1−v0| at common end: "
             f"{_gap_at_common_end(ds,'tbar','v1'):.4f} °C",
             transform=axT.transAxes, ha="right", va="bottom", fontsize=7, color="0.3")

    # (b) volume-mean S (fixed ±0.01 psu window — flat at the honest scale)
    for r, c, lab in reversed(MODELS):
        axS.plot(T[r], D[r]["sbar"], color=c, lw=0.6, alpha=0.25)
        axS.plot(T[r], running_mean(D[r]["sbar"]), color=c, lw=LW[r], label=lab)
    smid = float(np.nanmean(D["v0"]["sbar"]))
    axS.set_ylim(smid - 0.01, smid + 0.01)
    axS.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    axS.set_ylabel("volume-mean S [psu]"); axS.set_xlabel("year")
    axS.set_title("(b) global mean ocean salinity")
    axS.legend(fontsize=7, loc="lower right")

    # (c) OHC
    for r, c, lab in reversed(MODELS):
        axO.plot(T[r], D[r]["ohc"], color=c, lw=0.6, alpha=0.25)
        axO.plot(T[r], running_mean(D[r]["ohc"]), color=c, lw=LW[r], label=lab)
    axO.set_ylabel("OHC [ZJ, ref 0 °C]"); axO.set_xlabel("year")
    axO.set_title("(c) ocean heat content")
    axO.legend(fontsize=7, loc="best")
    axO.text(0.04, 0.04, "OHC = ρ₀c$_p$V·T̄ ⇒ tracks (a)\n|v1−v0| at common end: "
             f"{_gap_at_common_end(ds,'ohc','v1'):.1f} ZJ",
             transform=axO.transAxes, ha="left", va="bottom", fontsize=7, color="0.3")

    # (d),(e) Hovmöller drift of 63B (the all-optimizations run), anomaly vs start
    for ax, base, cmap, lab, ttl in [
            (axHT, "tz", "RdBu_r", "ΔT [°C]", "(d) T(z) drift (precond 1, vs start)"),
            (axHS, "sz", "BrBG_r", "ΔS [psu]", "(e) S(z) drift (precond 1, vs start)")]:
        field = D["v1"][base]
        anom = field - field[0:1, :]
        vmax = float(max(1e-4, np.nanpercentile(np.abs(anom), 98)))
        nr.plot_hovmoller(T["v1"], z, field, ax=ax, anomaly=True, mode="depth",
                          y_scale="sqrt", cmap=cmap, vmin=-vmax, vmax=vmax, colorbar=False)
        fig.colorbar(ax.collections[0], ax=ax, label=lab, pad=0.02)
        _sqrt_depth(ax)
        ax.set_xlabel("year"); ax.set_title(ttl)

    # (f) end-minus-start profiles over the COMMON window (identical months for
    # every model — a per-run span would put different interannual end states
    # into the upper ocean and fake a model difference)
    LS = {"v0": "--", "v1": "-"}
    axP.axvline(0.0, color="0.7", lw=0.8)
    axP2 = axP.twiny()
    for r, c, lab in reversed(MODELS):
        dT = D[r]["tz"][-1] - D[r]["tz"][0]
        dS = D[r]["sz"][-1] - D[r]["sz"][0]
        axP.plot(dT, z, color="#ff7f0e", ls=LS[r], lw=LW[r])
        axP2.plot(dS, z, color="#1f77b4", ls=LS[r], lw=LW[r])
    _sqrt_depth(axP)
    axP.set_ylim(6200, 0)
    axP.set_xlabel("ΔT [°C]", color="#ff7f0e"); axP2.set_xlabel("ΔS [psu]", color="#1f77b4")
    axP.tick_params(axis="x", colors="#ff7f0e"); axP2.tick_params(axis="x", colors="#1f77b4")
    axP.set_title(f"(f) drift profile (Dec {ycom} − Jan 1958)", pad=22)
    axP.legend(handles=[Line2D([], [], color="0.35", ls=LS[r], label=lab)
                        for r, _, lab in MODELS], fontsize=7, loc="lower center")

    fig.savefig(figpath, dpi=140)
    fig.savefig(figpath.replace(".png", ".pdf"))
    print(f"wrote {figpath} (+.pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=f"{M15}/fesom2_precond_figs")
    ap.add_argument("--reduce", action="store_true", help="force re-reduction")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    nc = os.path.join(a.outdir, "drift_precond.nc")
    if a.reduce or not os.path.exists(nc):
        reduce_all(nc)
    figure(nc, os.path.join(a.outdir, "fig_drift_precond.png"))


if __name__ == "__main__":
    main()
