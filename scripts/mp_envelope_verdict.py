#!/usr/bin/env python3
"""M8 Gate-3a verdict: FP32 divergence vs the ENSEMBLE chaos envelope.

The Gate-1 lesson (SP_PORTING_LESSONS / memory): single dt-seed controls are NON-MONOTONE
in seed size, so the envelope must be an ensemble maximum. This tool computes, per field and
snapshot step,
    sp_relL2   = relL2(SP run   vs FP64 base)
    env_relL2  = max over seed runs of relL2(seed run vs FP64 base)   (same for Linf)
and reports the ratio sp/env. Judgment (pre-registered, plan Gate 3a): the FP32 curve must
track the envelope's growth SHAPE — for chaos-saturated fields (mixing coefficients, ice) the
ratio should be O(1); for slowly-diverging fields (T/S/eta) FP32 sits at its direct-rounding
offset ABOVE a still-tiny envelope early on (ratio >> 1 with both values small is NOT a
failure), and the two growth shapes must agree over the window. Shape departure (runaway
ratio growth at large absolute values, jumps, saturation at O(1) while the envelope stays
small) = bug, not rounding.

STORAGE-class fields (static coordinates/axes: lat, lon, Z, zbar, time) are excluded — their
SP offset is float storage of constants, not divergence.

Usage:
  mp_envelope_verdict.py BASE_DIR SP_DIR SEED_DIR [SEED_DIR ...] [--csv OUT] [--plot OUT.png]
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mp_divergence_curve import run_diffs  # noqa: E402

STATIC = {"lat", "lon", "Z", "zbar", "time"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir", type=pathlib.Path)
    ap.add_argument("sp_dir", type=pathlib.Path)
    ap.add_argument("seed_dirs", type=pathlib.Path, nargs="+")
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    ap.add_argument("--plot", type=pathlib.Path, default=None)
    args = ap.parse_args()

    sp = run_diffs(args.base_dir, args.sp_dir)
    seeds = [run_diffs(args.base_dir, d) for d in args.seed_dirs]

    lines = ["field,step,sp_relL2,env_relL2,ratio_L2,sp_relLinf,env_relLinf,ratio_Linf"]
    table = {}
    for v in sorted(sp):
        if v in STATIC:
            continue
        steps, l2s, linfs = sp[v]
        env_l2 = []
        env_linf = []
        for i, s in enumerate(steps):
            e2 = max((sd[v][1][i] for sd in seeds if v in sd and i < len(sd[v][1])),
                     default=float("nan"))
            ei = max((sd[v][2][i] for sd in seeds if v in sd and i < len(sd[v][2])),
                     default=float("nan"))
            env_l2.append(e2)
            env_linf.append(ei)
            r2 = l2s[i] / e2 if e2 > 0 else float("inf")
            ri = linfs[i] / ei if ei > 0 else float("inf")
            lines.append(f"{v},{s},{l2s[i]:.6e},{e2:.6e},{r2:.3f},"
                         f"{linfs[i]:.6e},{ei:.6e},{ri:.3f}")
        table[v] = (steps, l2s, env_l2, linfs, env_linf)
        # console: first + last snapshot rows
        for i in (0, len(steps) - 1):
            print(f"  {v:16s} step {steps[i]:4d}  sp_relL2={l2s[i]:.3e} env={env_l2[i]:.3e} "
                  f"ratio={l2s[i]/env_l2[i] if env_l2[i] > 0 else float('inf'):8.2f}   "
                  f"sp_Linf={linfs[i]:.3e} env={env_linf[i]:.3e}")

    if args.csv:
        args.csv.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fields = sorted(table)
        ncols = min(4, len(fields))
        nrows = (len(fields) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False, sharex=True)
        for i, v in enumerate(fields):
            ax = axes[i // ncols][i % ncols]
            steps, l2s, env_l2, _, _ = table[v]
            ax.semilogy(steps, np.maximum(l2s, 1e-20), "-o", ms=3, label="FP32")
            ax.semilogy(steps, np.maximum(env_l2, 1e-20), "--s", ms=3,
                        label="seed envelope (max)")
            ax.set_title(v)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(len(fields), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("M8 Gate-3a: FP32 divergence vs FP64 seed-ensemble envelope (rel L2)")
        fig.supxlabel("step")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    sys.exit(main())
