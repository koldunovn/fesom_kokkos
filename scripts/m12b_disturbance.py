#!/usr/bin/env python3
"""M12b disturbance report — is the K=1 rung's perturbation distinguishable from
the model's own rank-dependence?

Why this framing. The rung's residual is a rank-ordering effect: the certified SE
module computes ~0.55 % of elements on more than one rank without reconciling
them, so the baseline's η encodes which rank owns each node (see the plan's
FINDING). A perturbation of that class must not be judged against zero — it must
be judged against the spread the model ALREADY shows when only the rank count
changes, which is the same class of perturbation and is accepted practice today.

So: controls are rung-OFF runs at several rank counts; the arm is rung-ON. If the
arm's difference from the reference sits inside the control spread, the rung
perturbs the model no more than running it on a different number of ranks — the
standard the project already accepts (L79, and M11's graded framework).

usage:
  m12b_disturbance.py --ref <dir> --controls <d1> <d2> ... --arms <a1> [a2 ...]
                      [--snap snap_000020.nc] [--vars temp,salt,ssh]
"""
import argparse
import os
import sys

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("netCDF4 not available in this interpreter")

DEFAULT_VARS = ["ssh", "temp", "salt", "u", "v"]


def load(d, snap, var):
    p = os.path.join(d, snap)
    if not os.path.exists(p):
        return None
    ds = nc.Dataset(p)
    if var not in ds.variables:
        ds.close()
        return None
    a = np.array(ds[var][:], dtype=np.float64)
    ds.close()
    return a


def stats(ref, other):
    """rms and max of |other - ref|, plus rms relative to the field's own spread."""
    d = other - ref
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    r = ref[np.isfinite(ref)]
    return dict(rms=float(np.sqrt(np.mean(d * d))),
                mx=float(np.max(np.abs(d))),
                fstd=float(np.std(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--controls", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--snap", default=None,
                    help="snapshot filename; default = the last snap_*.nc in --ref")
    ap.add_argument("--vars", default=",".join(DEFAULT_VARS))
    a = ap.parse_args()

    snap = a.snap
    if snap is None:
        snaps = sorted(f for f in os.listdir(a.ref) if f.startswith("snap_") and f.endswith(".nc"))
        if not snaps:
            sys.exit(f"no snap_*.nc in {a.ref}")
        snap = snaps[-1]
    print(f"# snapshot {snap}\n# reference {a.ref}")

    rc = 0
    for var in a.vars.split(","):
        ref = load(a.ref, snap, var)
        if ref is None:
            continue
        crows, arows = [], []
        for c in a.controls:
            o = load(c, snap, var)
            if o is not None and o.shape == ref.shape:
                s = stats(ref, o)
                if s:
                    crows.append((os.path.basename(c), s))
        for m in a.arms:
            o = load(m, snap, var)
            if o is not None and o.shape == ref.shape:
                s = stats(ref, o)
                if s:
                    arows.append((os.path.basename(m), s))
        if not crows or not arows:
            continue

        cmax = max(s["rms"] for _, s in crows)
        cmin = min(s["rms"] for _, s in crows)
        print(f"\n== {var}  (field std {crows[0][1]['fstd']:.4g})")
        for n, s in crows:
            print(f"   control {n:<28s} rms {s['rms']:.4g}  max {s['mx']:.4g}")
        # 🔴 controls must be DEMONSTRATED DISTINCT — a control set that is
        # identical to the reference has no spread and would pass anything.
        if cmax == 0.0:
            print("   !! controls are identical to the reference: the spread is vacuous")
            rc = 1
        for n, s in arows:
            verdict = ("INSIDE the control spread" if s["rms"] <= cmax else
                       f"OUTSIDE by x{s['rms']/cmax:.2f}" if cmax > 0 else "no spread")
            print(f"   ARM     {n:<28s} rms {s['rms']:.4g}  max {s['mx']:.4g}  -> {verdict}")
            if cmax > 0 and s["rms"] > cmax:
                rc = 1
        print(f"   control spread rms [{cmin:.4g}, {cmax:.4g}]")

    print("\n# rc=0 means every arm sits inside the control spread on every field")
    return rc


if __name__ == "__main__":
    sys.exit(main())
