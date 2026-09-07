"""m15_precond_meanstate.py — the JAX-paper F2 (surface mean state) figure for the
M15 60-yr Fortran CORE2 preconditioner twins.

Two rows (SST, SSS) x three columns: variant0 - PHC3.0, variant1 - PHC3.0,
variant1 - variant0. The first two columns share the wide obs-bias scale and show
that the two preconditioners have the SAME skill against observations; the third
gets its own tight scale so the residual between them is resolved rather than
washed out to white (the F2 convention, USER 2026-07-30).

Climatology window 1980-2009, matching the paper's F2/F3. Surface = level 0 of the
3-D field; FESOM writes that as `sst`/`sss` (verified bit-identical to temp[:,0,:]),
so the cheap 2-D streams are read instead of the 3-D ones.

  $NEREUS_PYTHON scripts/m15_precond_meanstate.py [--y0 1980 --y1 2009]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl              # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import paths    # noqa: E402
import common   # noqa: E402
import nereus as nr   # noqa: E402

M15 = "/work/ab0995/a270088"
RUNS = {"v0": f"{M15}/fesom2_precond_v0", "v1": f"{M15}/fesom2_precond_v1"}
OUTDIR = f"{M15}/fesom2_precond_figs"
ROWS = ["sst", "sss"]
# obs-bias scale as in the paper's F2; dmax is set from the field's own 99th
# percentile at runtime so the difference column resolves its structure.
BIAS = {"sst": dict(vmax=4.0, label="SST bias [°C]", dlabel="v1 − v0 [°C]"),
        "sss": dict(vmax=2.0, label="SSS bias [psu]", dlabel="v1 − v0 [psu]")}


def surface_climatology(root, nod2, y0, y1):
    """Node-wise annual mean of sst/sss over [y0, y1]."""
    acc = {k: np.zeros(nod2) for k in ROWS}
    cnt = {k: np.zeros(nod2) for k in ROWS}
    nmon = 0
    import xarray as xr
    for key in ROWS:
        n = 0
        for y in range(y0, y1 + 1):
            p = os.path.join(root, f"{key}.fesom.{y}.nc")
            if not os.path.exists(p):
                continue
            with xr.open_dataset(p) as ds:
                f = ds[key].values                      # (12, nod2)
            for m in range(f.shape[0]):
                ok = np.isfinite(f[m])
                acc[key][ok] += f[m][ok]
                cnt[key][ok] += 1.0
                n += 1
        nmon = n
    return {k: np.where(cnt[k] > 0, acc[k] / np.maximum(cnt[k], 1), np.nan)
            for k in ROWS}, nmon


def phc_surface_nodes(mesh):
    """PHC3.0 annual surface T/S interpolated onto the model nodes."""
    import xarray as xr
    ds = xr.open_dataset(paths.OBS["phc"], decode_times=False)
    out = {}
    for key, var in (("sst", "temp"), ("sss", "salt")):
        f = ds[var].isel(depth=0).values
        out[key] = common.regular_to_nodes(ds["lon"].values, ds["lat"].values, f,
                                           mesh.lon, mesh.lat)
    ds.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1980)
    ap.add_argument("--y1", type=int, default=2009)
    args = ap.parse_args()

    common.set_style()
    mesh = common.load_mesh(paths.mesh_diag("core2"))
    nod2 = mesh.lon.size
    obs = phc_surface_nodes(mesh)
    clim = {}
    for lab, root in RUNS.items():
        clim[lab], nmon = surface_climatology(root, nod2, args.y0, args.y1)
        print(f"[meanstate] {lab}: {nmon} months {args.y0}-{args.y1}")

    fields, rms = {}, {}
    for var in ROWS:
        for lab in RUNS:
            fields[f"{var}_bias_{lab}"] = clim[lab][var] - obs[var]
            rms[f"{var}_bias_{lab}"] = common.node_weighted_rms(
                fields[f"{var}_bias_{lab}"], mesh.area2d)
        fields[f"{var}_diff"] = clim["v1"][var] - clim["v0"][var]
        rms[f"{var}_diff"] = common.node_weighted_rms(fields[f"{var}_diff"], mesh.area2d)
        d = np.abs(fields[f"{var}_diff"])
        BIAS[var]["dmax"] = float(np.nanpercentile(d, 99.0))
        print(f"[meanstate] {var}: RMSE v0 {rms[f'{var}_bias_v0']:.4f}  "
              f"v1 {rms[f'{var}_bias_v1']:.4f}  |  v1−v0 RMS {rms[f'{var}_diff']:.3e}  "
              f"p99 {BIAS[var]['dmax']:.3e}  max {np.nanmax(d):.3e}")

    cols = [("bias_v0", "variant 0 (as coded) − PHC", "RMSE"),
            ("bias_v1", "variant 1 (#984) − PHC", "RMSE"),
            ("diff", "variant 1 − variant 0", "RMS")]
    proj = nr.plotting.get_projection("rob")
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(4.2 * 3, 5.4),
                             subplot_kw={"projection": proj}, constrained_layout=True)
    axes = np.atleast_2d(axes)
    interp = None
    for r, var in enumerate(ROWS):
        vmax, dmax = BIAS[var]["vmax"], BIAS[var]["dmax"]
        for c, (suf, title, tag) in enumerate(cols):
            ax = axes[r, c]
            vm = dmax if suf == "diff" else vmax
            _, _, interp = nr.plot(fields[f"{var}_{suf}"], mesh.lon, mesh.lat,
                                   projection="rob", method="linear", ax=ax,
                                   interpolator=interp, colorbar=False,
                                   cmap="RdBu_r", vmin=-vm, vmax=vm,
                                   land=True, coastlines=True)
            if r == 0:
                ax.set_title(title, fontsize=10)
            ax.text(0.5, -0.06, f"{tag} = {rms[f'{var}_{suf}']:.4f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-vmax, vmax), cmap="RdBu_r")
        fig.colorbar(sm, ax=axes[r, :2].tolist(), shrink=0.85, pad=0.01,
                     label=BIAS[var]["label"])
        smd = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-dmax, dmax), cmap="RdBu_r")
        fig.colorbar(smd, ax=axes[r, 2], shrink=0.85, pad=0.01, label=BIAS[var]["dlabel"])

    fig.suptitle(f"CORE2 annual-mean surface bias vs PHC3.0, {args.y0}–{args.y1} — "
                 f"ssh CG preconditioner variant 0 vs variant 1", fontsize=11)
    for ax in axes.flat:
        for coll in ax.collections:
            coll.set_rasterized(True)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig_precond_meanstate.{ext}"), dpi=200)
    print(f"[meanstate] wrote {OUTDIR}/fig_precond_meanstate.png")


if __name__ == "__main__":
    main()
