"""m15_precond_diff.py — the JAX-paper F4 (drift residual) figure for the M15 60-yr
Fortran CORE2 preconditioner twins, expressed throughout as VARIANT 1 MINUS VARIANT 0.

Mirrors the paper's rewritten F4: plotting the residual directly puts the actual claim
on the axis, instead of overlaying two trajectories that are visually identical and
therefore tell the reader nothing. Four panels, 2x2:

  (a) volume-mean T difference (full depth + 0-700 m)   (c) OHC difference
  (b) volume-mean S difference                          (d) mean T(z), S(z) difference

All three time-series panels are ANNUAL means, as in the paper. Panel (d) is the
difference of the MEAN state over the F2/F3 analysis window (1980-2009), not an
end-minus-start drift profile: the claim is about the mean climate.

NB OHC uses a constant rho0*cp and the fixed mesh volume, so OHC == rho0*cp*V*Tbar
EXACTLY; panel (c) is panel (a)'s full-depth curve rescaled, and says so in the title.

Reads the reduction written by m15_precond_drift.py --reduce.

  $NEREUS_PYTHON scripts/m15_precond_diff.py [--y0 1980 --y1 2009]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.scale import FuncScale   # noqa: E402

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import paths    # noqa: E402
import common   # noqa: E402

OUTDIR = "/work/ab0995/a270088/fesom2_precond_figs"
NC = f"{OUTDIR}/drift_precond.nc"
TCOL, SCOL, UPC = "#1f77b4", "#2ca02c", "#ff7f0e"
DEPTH_TICKS = [0, 100, 250, 500, 1000, 2000, 4000, 6000]


def annual(t, y):
    """Collapse a monthly series to (year_centre, annual_mean), NaN-aware."""
    yr = np.floor(np.asarray(t, float)).astype(int)
    oy, ov = [], []
    for u in np.unique(yr):
        v = np.asarray(y, float)[yr == u]
        v = v[np.isfinite(v)]
        if v.size:
            oy.append(u + 0.5); ov.append(v.mean())
    return np.asarray(oy), np.asarray(ov)


def _scale(x):
    """Power-of-ten scale factor and its exponent for a residual series, so the axis
    reads in small integers rather than 1e-5 offsets."""
    m = np.nanmax(np.abs(x))
    if not np.isfinite(m) or m == 0:
        return 1.0, 0
    e = int(np.floor(np.log10(m)))
    return 10.0 ** (-e), e


def _sym_ylim(ax, *series, pad=1.15):
    m = max([np.nanmax(np.abs(s)) for s in series if np.isfinite(s).any()] or [1.0])
    ax.set_ylim(-pad * m, pad * m)


def _zeroline(ax):
    ax.axhline(0.0, color="0.4", lw=0.9, zorder=1)


def _sqrt_depth(ax):
    ax.set_yscale(FuncScale(ax, (lambda x: np.sqrt(np.maximum(x, 0.0)),
                                 lambda x: np.asarray(x) ** 2)))
    ax.set_yticks(DEPTH_TICKS)
    ax.set_ylabel("depth [m]")


def _exp_label(e):
    return "" if e == 0 else f"10$^{{{e}}}$ "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1980)
    ap.add_argument("--y1", type=int, default=2009)
    a = ap.parse_args()

    common.set_style()
    import xarray as xr
    ds = xr.open_dataset(NC)
    t0 = np.asarray(ds["time_v0"].values); t1 = np.asarray(ds["time_v1"].values)
    idx = common.align_index(t0, t1); keep = np.where(idx >= 0)[0]
    i1, i0 = keep, idx[keep]
    t = t0[i0]

    def d(base):
        return (np.asarray(ds[f"{base}_v1"].values)[i1]
                - np.asarray(ds[f"{base}_v0"].values)[i0])

    fig, axes = plt.subplots(2, 2, figsize=(6.97, 5.6), constrained_layout=True)
    axT, axS, axO, axP = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- (a) volume-mean temperature difference, full depth + 0-700 m ----
    dT, dT7 = d("tbar"), d("tbar700")
    ta, aT = annual(t, dT); _, aT7 = annual(t, dT7)
    sc, e = _scale(np.concatenate([aT, aT7]))
    _zeroline(axT)
    for arr, col, lab in ((aT, TCOL, "full depth"), (aT7, UPC, "0–700 m")):
        axT.plot(ta, arr * sc, color=col, lw=1.8, marker="o", ms=2.5, label=lab)
    _sym_ylim(axT, aT * sc, aT7 * sc)
    axT.set_ylabel(f"v1 − v0  [{_exp_label(e)}°C]")
    axT.set_xlabel("year"); axT.set_title("(a) global mean ocean temperature")
    axT.legend(fontsize=7, ncol=2, loc="upper left")

    # ---- (b) volume-mean salinity difference ----
    tb, aS = annual(t, d("sbar"))
    scS, eS = _scale(aS)
    _zeroline(axS)
    axS.plot(tb, aS * scS, color=SCOL, lw=1.8, marker="o", ms=2.5)
    _sym_ylim(axS, aS * scS)
    axS.set_ylabel(f"v1 − v0  [{_exp_label(eS)}psu]")
    axS.set_xlabel("year"); axS.set_title("(b) global mean ocean salinity")

    # ---- (c) ocean heat content difference ----
    tc, aO = annual(t, d("ohc"))
    _zeroline(axO)
    axO.plot(tc, aO, color=TCOL, lw=1.8, marker="o", ms=2.5)
    _sym_ylim(axO, aO)
    axO.set_ylabel("v1 − v0  [ZJ]")
    axO.set_xlabel("year"); axO.set_title("(c) ocean heat content")

    # ---- (d) mean-state vertical profile difference over the analysis window ----
    inw = (t >= a.y0) & (t < a.y1 + 1)
    z = np.asarray(ds["z"].values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dTz = np.nanmean(np.asarray(ds["tz_v1"].values)[i1][inw]
                         - np.asarray(ds["tz_v0"].values)[i0][inw], axis=0)
        dSz = np.nanmean(np.asarray(ds["sz_v1"].values)[i1][inw]
                         - np.asarray(ds["sz_v0"].values)[i0][inw], axis=0)
    scZ, eZ = _scale(dTz); scZs, eZs = _scale(dSz)
    axP.plot(dTz * scZ, z, color=TCOL, lw=1.8)
    axP2 = axP.twiny()
    axP2.xaxis.set_ticks_position("bottom"); axP2.xaxis.set_label_position("bottom")
    axP2.spines["bottom"].set_position(("outward", 38))
    axP2.spines["top"].set_visible(False); axP2.grid(False)
    axP2.plot(dSz * scZs, z, color=SCOL, lw=1.8)
    for ax_, arr, col in ((axP, dTz * scZ, TCOL), (axP2, dSz * scZs, SCOL)):
        m = np.nanmax(np.abs(arr))
        ax_.set_xlim(-1.15 * m, 1.15 * m)
        ax_.tick_params(axis="x", colors=col); ax_.spines["bottom"].set_color(col)
    axP.axvline(0.0, color="0.4", lw=0.9)
    _sqrt_depth(axP); axP.set_ylim(6200, 0); axP2.set_ylim(6200, 0)
    axP.set_xlabel(f"ΔT  [{_exp_label(eZ)}°C]", color=TCOL)
    axP2.set_xlabel(f"ΔS  [{_exp_label(eZs)}psu]", color=SCOL)
    axP.set_title(f"(d) mean-state profile difference, {a.y0}–{a.y1}")
    axP.legend(handles=[Line2D([], [], color=TCOL, label="ΔT"),
                        Line2D([], [], color=SCOL, label="ΔS")],
               fontsize=7, loc="lower right")

    fig.suptitle("CORE2 global drift — ssh CG preconditioner variant 1 minus variant 0 "
                 f"(1958–2017, {t.size} months)", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/fig_precond_diff.{ext}", dpi=200)
    print(f"[diff] wrote {OUTDIR}/fig_precond_diff.png")
    print(f"[diff] annual-mean residual: max|dTbar| {np.nanmax(np.abs(aT)):.3e} degC, "
          f"max|dTbar700| {np.nanmax(np.abs(aT7)):.3e} degC, "
          f"max|dSbar| {np.nanmax(np.abs(aS)):.3e} psu, max|dOHC| {np.nanmax(np.abs(aO)):.3e} ZJ")
    print(f"[diff] profile {a.y0}-{a.y1}: max|dT(z)| {np.nanmax(np.abs(dTz)):.3e} degC, "
          f"max|dS(z)| {np.nanmax(np.abs(dSz)):.3e} psu")
    ds.close()


if __name__ == "__main__":
    main()
