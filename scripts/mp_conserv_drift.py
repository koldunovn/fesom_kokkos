#!/usr/bin/env python3
"""M8 Gate-3b: conservation drift, FP32 vs FP64 twin (the measurement PR-940 never made).

Parses `[fesom_port] CONSERV step= heat= salt= vol=` lines (the FESOM_MP_CONSERV hook:
FP64-diagnosed global integrals; heat = ∫T dV [°C·m³], salt = ∫S dV [PSU·m³], vol [m³])
from two run logs and reports, per quantity:
    drift(t)  = (X(t) - X(0)) / X(0)           per run   (forced runs DO drift — physics)
    gap(t)    = drift_sp(t) - drift_dp(t)                (the precision signal)
Judgment: |gap| must stay ≪ |drift| (precision error a small fraction of the physical
signal) and must not grow systematically faster than the drift itself.

Usage: mp_conserv_drift.py DP_LOG SP_LOG [--plot OUT.png] [--csv OUT.csv]
"""
import argparse
import pathlib
import re
import sys

RX = re.compile(r"CONSERV step=\s*(\d+)\s+heat=([-\d.e+]+)\s+salt=([-\d.e+]+)\s+vol=([-\d.e+]+)")


def parse(path: pathlib.Path):
    steps, heat, salt, vol = [], [], [], []
    for line in path.read_text().splitlines():
        m = RX.search(line)
        if m:
            steps.append(int(m.group(1)))
            heat.append(float(m.group(2)))
            salt.append(float(m.group(3)))
            vol.append(float(m.group(4)))
    return steps, {"heat": heat, "salt": salt, "vol": vol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dp_log", type=pathlib.Path)
    ap.add_argument("sp_log", type=pathlib.Path)
    ap.add_argument("--plot", type=pathlib.Path, default=None)
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    args = ap.parse_args()

    dsteps, dp = parse(args.dp_log)
    ssteps, sp = parse(args.sp_log)
    n = min(len(dsteps), len(ssteps))
    if n == 0 or dsteps[:n] != ssteps[:n]:
        sys.exit("CONSERV series missing or step-misaligned")

    lines = ["step,quantity,dp_drift,sp_drift,gap"]
    print(f"{'qty':6s} {'step':>6s} {'dp_drift':>12s} {'sp_drift':>12s} {'gap':>12s}")
    series = {}
    for q in ("heat", "salt", "vol"):
        d0, s0 = dp[q][0], sp[q][0]
        dd = [(x - d0) / abs(d0) for x in dp[q][:n]]
        sd = [(x - s0) / abs(s0) for x in sp[q][:n]]
        gap = [s - d for s, d in zip(sd, dd)]
        series[q] = (dsteps[:n], dd, sd, gap)
        for i in range(n):
            lines.append(f"{dsteps[i]},{q},{dd[i]:.9e},{sd[i]:.9e},{gap[i]:.9e}")
        for i in (1, n // 2, n - 1):
            print(f"{q:6s} {dsteps[i]:6d} {dd[i]:12.3e} {sd[i]:12.3e} {gap[i]:12.3e}")
        print(f"{q:6s}  final |gap|/|dp_drift| = "
              f"{abs(gap[-1]) / max(abs(dd[-1]), 1e-300):.3f}")

    if args.csv:
        args.csv.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.csv}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
        for j, q in enumerate(("heat", "salt", "vol")):
            steps, dd, sd, gap = series[q]
            ax = axes[0][j]
            ax.plot(steps, dd, label="FP64")
            ax.plot(steps, sd, "--", label="FP32")
            ax.set_title(f"{q}: relative drift")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8)
            ax2 = axes[1][j]
            ax2.plot(steps, gap, color="crimson")
            ax2.set_title(f"{q}: FP32−FP64 gap")
            ax2.grid(True, alpha=0.3)
        fig.supxlabel("step (dt=1800 s; 1440 = 30 d)")
        fig.suptitle("M8 Gate-3b: global conservation drift, options config, CORE2")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    sys.exit(main())
