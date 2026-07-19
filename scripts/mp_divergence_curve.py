#!/usr/bin/env python3
"""M8 Gate-3a instrument: divergence curves vs the chaos envelope (cross-dtype safe).

Compares a TEST run (FP32 build, or FP64+1ulp perturbation control) against a REF run (FP64)
snapshot-by-snapshot, computing per-field relative L2 and Linf differences. All arrays are
upcast to float64 before differencing, so FP32-vs-FP64 comparisons are exact in the diff
(this is the plan's "cross-dtype diff" instrument — diff_snap.py stays the zero-tolerance
same-dtype byte oracle and is deliberately untouched).

Judgment (Gate 3a): the FP32 curve must track the SHAPE of the perturbation-control curve
(growth rate), offset by its ~1e-7 start; a curve that departs the envelope shape (faster
growth, jumps, saturation at large values) indicates a bug, not rounding.

Usage:
  mp_divergence_curve.py REF_DIR TEST_DIR [--label NAME] [--csv OUT.csv]
  mp_divergence_curve.py REF_DIR TEST_DIR --envelope ENV_DIR [--plot OUT.png]
                         (ENV_DIR = the FP64+1ulp control run; adds envelope columns/curves)

Output: one row per (snapshot, field): rel_L2 = ||t-r||_2 / (||r||_2 + tiny),
rel_Linf = max|t-r| / (max|r| + tiny). Plot (if requested): per-field log-y curves,
TEST solid, ENVELOPE dashed.
"""
import argparse
import pathlib
import sys

import netCDF4 as nc
import numpy as np

TINY = 1e-300


def snap_index(p: pathlib.Path) -> int:
    return int(p.stem.split("_")[1])


def load_fields(path: pathlib.Path):
    out = {}
    with nc.Dataset(path) as ds:
        for v in ds.variables:
            a = ds.variables[v][...]
            if np.issubdtype(a.dtype, np.floating):
                out[v] = np.asarray(a, dtype=np.float64)
    return out


def run_diffs(ref_dir: pathlib.Path, test_dir: pathlib.Path):
    """{field: (steps[], rel_l2[], rel_linf[])} over common snapshots."""
    rows = {}
    for rp in sorted(ref_dir.glob("snap_*.nc"), key=snap_index):
        tp = test_dir / rp.name
        if not tp.exists():
            continue
        r, t = load_fields(rp), load_fields(tp)
        for v in r:
            if v not in t:
                continue
            d = t[v] - r[v]
            rel_l2 = float(np.linalg.norm(d) / (np.linalg.norm(r[v]) + TINY))
            rel_linf = float(np.max(np.abs(d)) / (np.max(np.abs(r[v])) + TINY))
            rows.setdefault(v, ([], [], []))
            rows[v][0].append(snap_index(rp))
            rows[v][1].append(rel_l2)
            rows[v][2].append(rel_linf)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", type=pathlib.Path)
    ap.add_argument("test_dir", type=pathlib.Path)
    ap.add_argument("--envelope", type=pathlib.Path, default=None,
                    help="FP64+1ulp control run dir (chaos envelope)")
    ap.add_argument("--label", default="test")
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    ap.add_argument("--plot", type=pathlib.Path, default=None)
    args = ap.parse_args()

    test = run_diffs(args.ref_dir, args.test_dir)
    env = run_diffs(args.ref_dir, args.envelope) if args.envelope else {}

    lines = ["field,step,rel_L2,rel_Linf,env_rel_L2,env_rel_Linf"]
    for v in sorted(test):
        steps, l2s, linfs = test[v]
        e = env.get(v, ([], [], []))
        for i, s in enumerate(steps):
            el2 = e[1][i] if i < len(e[1]) else float("nan")
            elinf = e[2][i] if i < len(e[2]) else float("nan")
            lines.append(f"{v},{s},{l2s[i]:.6e},{linfs[i]:.6e},{el2:.6e},{elinf:.6e}")
            print(f"  {v:10s} step {s:6d}  relL2={l2s[i]:.3e} relLinf={linfs[i]:.3e}"
                  + (f"  env relL2={el2:.3e}" if args.envelope else ""))
    if args.csv:
        args.csv.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fields = sorted(test)
        ncols = min(4, len(fields))
        nrows = (len(fields) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False, sharex=True)
        for i, v in enumerate(fields):
            ax = axes[i // ncols][i % ncols]
            steps, l2s, _ = test[v]
            ax.semilogy(steps, np.maximum(l2s, 1e-20), "-o", ms=3, label=args.label)
            if v in env:
                es, el2, _ = env[v]
                ax.semilogy(es, np.maximum(el2, 1e-20), "--s", ms=3, label="1ulp envelope")
            ax.set_title(v)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(len(fields), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"M8 divergence curves: {args.label} vs FP64 ref (rel L2)")
        fig.supxlabel("step")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    sys.exit(main())
