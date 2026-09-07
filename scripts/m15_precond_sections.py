"""m15_precond_sections.py — the JAX-paper F3 (zonal-mean T/S sections) figure for the
M15 60-yr Fortran CORE2 preconditioner twins; the vertical complement of
m15_precond_meanstate.py.

Two rows (temperature, salinity) x three columns: variant0 - PHC3.0, variant1 - PHC3.0,
variant1 - variant0, on a (latitude, model mid-depth) grid with the sqrt-warped depth
axis. Columns 1-2 share the wide obs-bias scale (same skill against observations);
column 3 gets its own tight scale so the residual is resolved, not hidden.

Zonal means are area-weighted by the model's own per-level control volumes; the summary
RMS is VOLUME-weighted, both as in the paper's F3. PHC3.0 is zonally averaged on its
regular grid and interpolated onto the section grid.

  $NEREUS_PYTHON scripts/m15_precond_sections.py [--y0 1980 --y1 2009 --dlat 1.0] [--reduce]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl              # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.scale import FuncScale  # noqa: E402

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import paths    # noqa: E402
import common   # noqa: E402

M15 = "/work/ab0995/a270088"
RUNS = {"v0": f"{M15}/fesom2_precond_v0", "v1": f"{M15}/fesom2_precond_v1"}
OUTDIR = f"{M15}/fesom2_precond_figs"
NC = f"{OUTDIR}/sections_precond.nc"
VARS = ("temp", "salt")
DEPTH_TICKS = [0, 50, 100, 250, 500, 1000, 2000, 4000, 6000]
BIAS = {"temp": dict(vmax=3.0, label="T bias [°C]", dlabel="v1 − v0 [°C]"),
        "salt": dict(vmax=1.0, label="S bias [psu]", dlabel="v1 − v0 [psu]")}


def lat_bins(mesh, dlat):
    lo, hi = np.floor(float(mesh.lat.min())), np.ceil(float(mesh.lat.max()))
    edges = np.arange(lo, hi + dlat, dlat)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(mesh.lat, edges) - 1
    valid = (idx >= 0) & (idx < centers.size)
    return centers, idx, valid


def section_volume(mesh, nlay, idx, valid, nbin, dz):
    area = np.asarray(mesh.area3d, float)
    vol = np.zeros((nlay, nbin))
    for k in range(nlay):
        good = valid & (area[k] > 0)
        vol[k] = np.bincount(idx[good], weights=area[k][good], minlength=nbin) * dz[k]
    return vol


def zonal_section(root, mesh, nlay, idx, valid, nbin, y0, y1, nz):
    area = np.asarray(mesh.area3d, float)
    out, nmon = {}, 0
    for var in VARS:
        num = np.zeros((nlay, nbin)); den = np.zeros((nlay, nbin)); nm = 0
        for _dy, _mo, f in common.fortran_iter_months(root, var, y0, y1, pad_to_nz=nz):
            f = np.asarray(f, float)
            for k in range(nlay):
                good = valid & np.isfinite(f[k]) & (area[k] > 0)
                w = np.where(good, area[k], 0.0)
                num[k] += np.bincount(idx[good], weights=(w * f[k])[good], minlength=nbin)
                den[k] += np.bincount(idx[good], weights=w[good], minlength=nbin)
            nm += 1
        out[var] = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
        nmon = nm
    return out, nmon


def phc_sections(centers, zmid):
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    ds = xr.open_dataset(paths.OBS["phc"], decode_times=False)
    plat = np.asarray(ds["lat"].values, float)
    pdep = np.asarray(ds["depth"].values, float)
    out = {}
    for var in VARS:
        zm = np.nanmean(np.asarray(ds[var].values, float), axis=2)
        it = RegularGridInterpolator((pdep, plat), zm, method="linear",
                                     bounds_error=False, fill_value=np.nan)
        Z, L = np.meshgrid(zmid, centers, indexing="ij")
        out[var] = it(np.stack([Z.ravel(), L.ravel()], axis=1)).reshape(Z.shape)
    ds.close()
    return out


def reduce_all(y0, y1, dlat):
    import xarray as xr
    mesh = common.load_mesh(paths.mesh_diag("core2"))
    zbar = mesh.zbar
    dz = common.layer_thickness(zbar); nlay = dz.size
    zmid = common._layer_mid_depth(zbar)[:nlay]
    centers, idx, valid = lat_bins(mesh, dlat); nbin = centers.size
    print(f"[sections] {nlay} layers, {nbin} lat bins, window {y0}-{y1}")
    sec = {}
    for lab, root in RUNS.items():
        sec[lab], nm = zonal_section(root, mesh, nlay, idx, valid, nbin, y0, y1, zbar.size)
        print(f"[sections] {lab}: {nm} months")
    obs = phc_sections(centers, zmid)
    w2d = section_volume(mesh, nlay, idx, valid, nbin, dz)
    data = {"volume_weight": (("z", "lat"), w2d.astype(np.float32))}
    attrs = dict(mesh="core2", obs="PHC3.0_annual", window=f"{y0}-{y1}", dlat=dlat,
                 rms_weight="control-volume volume (wet nod_area by latitude * dz)")

    def wrms(d):
        m = np.isfinite(d) & (w2d > 0)
        return float(np.sqrt(np.sum((d[m] ** 2) * w2d[m]) / np.sum(w2d[m]))) if m.any() else np.nan

    for v in VARS:
        data[f"{v}_obs"] = (("z", "lat"), obs[v].astype(np.float32))
        for lab in RUNS:
            b = sec[lab][v] - obs[v]
            data[f"{v}_bias_{lab}"] = (("z", "lat"), b.astype(np.float32))
            attrs[f"rms_{v}_bias_{lab}"] = wrms(b)
        d = sec["v1"][v] - sec["v0"][v]
        data[f"{v}_diff"] = (("z", "lat"), d.astype(np.float32))
        attrs[f"rms_{v}_diff"] = wrms(d)
        print(f"[sections] {v}: RMS v0 {attrs[f'rms_{v}_bias_v0']:.4f}  "
              f"v1 {attrs[f'rms_{v}_bias_v1']:.4f}  |  v1−v0 {attrs[f'rms_{v}_diff']:.3e}")
    ds = xr.Dataset(data, coords={"z": zmid.astype(np.float64),
                                  "lat": centers.astype(np.float64)}, attrs=attrs)
    ds.to_netcdf(NC)
    print(f"[sections] wrote {NC}")


def _sqrt_depth(ax, zmax):
    ax.set_yscale(FuncScale(ax, (lambda x: np.sqrt(np.maximum(x, 0.0)),
                                 lambda x: np.asarray(x) ** 2)))
    ax.set_yticks([t for t in DEPTH_TICKS if t <= zmax])
    ax.set_ylim(zmax, 0)


def figure():
    import xarray as xr
    common.set_style()
    ds = xr.open_dataset(NC)
    lat = ds["lat"].values; z = ds["z"].values; zmax = float(z.max())
    cols = [("bias_v0", "variant 0 (as coded) − PHC"),
            ("bias_v1", "variant 1 (#984) − PHC"),
            ("diff", "variant 1 − variant 0")]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.4), constrained_layout=True)
    for r, var in enumerate(VARS):
        vmax = BIAS[var]["vmax"]
        dmax = float(np.nanpercentile(np.abs(ds[f"{var}_diff"].values), 99.0))
        BIAS[var]["dmax"] = dmax
        for c, (suf, title) in enumerate(cols):
            ax = axes[r, c]
            vm = dmax if suf == "diff" else vmax
            f = ds[f"{var}_{suf}"].values
            pc = ax.pcolormesh(lat, z, f, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
            _sqrt_depth(ax, zmax)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if r == 1:
                ax.set_xlabel("latitude")
            if c == 0:
                ax.set_ylabel("depth [m]")
            tag = "RMS" if suf == "diff" else "RMSE"
            ax.text(0.02, 0.06, f"{tag} = {ds.attrs[f'rms_{var}_{suf}']:.4f}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
            if c == 2:
                fig.colorbar(pc, ax=ax, pad=0.02).set_label(BIAS[var]["dlabel"])
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-vmax, vmax), cmap="RdBu_r")
        fig.colorbar(sm, ax=axes[r, :2].tolist(), pad=0.02).set_label(BIAS[var]["label"])
    fig.suptitle(f"CORE2 zonal-mean T/S bias vs PHC3.0, {ds.attrs['window']} — "
                 f"ssh CG preconditioner variant 0 vs variant 1", fontsize=11)
    for ax in axes.flat:
        for coll in ax.collections:
            coll.set_rasterized(True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/fig_precond_sections.{ext}", dpi=200)
    print(f"[sections] wrote {OUTDIR}/fig_precond_sections.png")
    for v in VARS:
        print(f"[sections] {v}: diff p99 {BIAS[v]['dmax']:.3e}, "
              f"max {np.nanmax(np.abs(ds[f'{v}_diff'].values)):.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1980)
    ap.add_argument("--y1", type=int, default=2009)
    ap.add_argument("--dlat", type=float, default=1.0)
    ap.add_argument("--reduce", action="store_true")
    a = ap.parse_args()
    if a.reduce or not os.path.exists(NC):
        reduce_all(a.y0, a.y1, a.dlat)
    figure()


if __name__ == "__main__":
    main()
