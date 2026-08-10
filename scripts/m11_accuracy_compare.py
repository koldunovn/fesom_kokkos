#!/usr/bin/env python3
"""M11: compare two FESOM output trees, optionally through a node permutation.

Why this exists: an ordering arm cannot be judged by a bound someone guessed. FESOM's mixing
scheme flips boundary-layer depths on round-off, so a handful of columns will always diverge
by O(1) no matter how correct the mesh is — which makes the MAXIMUM difference a useless
statistic and the median an informative one. The honest test is a comparison against the
PARTITION-CLASS FLOOR: change only the partition, keep the numbering, and measure the same
statistics. A renumbering that stays inside that floor perturbs the model no more than
repartitioning already does, which is the standard the project already accepts (L79).

usage:
  m11_accuracy_compare.py <ref_dir> <test_dir> [--perm <mesh_with_m11_perm_node.npy>]
                          [--vars temp,salt,ssh,sst,m_ice] [--label NAME]
"""
import argparse
import os
import sys

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("netCDF4 not available in this interpreter")

DEFAULT_VARS = ["temp", "salt", "ssh", "sst", "sss", "m_ice", "a_ice"]


def load(d, name):
    p = f"{d}/{name}.fesom.1958.monthly.nc"
    if not os.path.exists(p):
        return None, None
    ds = nc.Dataset(p)
    if name not in ds.variables:
        ds.close()
        return None, None
    return np.array(ds[name][:]), ds


def stats(ref, test, perm=None):
    ax = None
    for i, s in enumerate(ref.shape):
        if s == (perm.size if perm is not None else s) and s > 1000:
            ax = i
    if ax is None:
        ax = int(np.argmax(ref.shape))
    if perm is not None:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        test = np.take(test, inv, axis=ax)
    d = np.abs(test - ref)
    d = d[np.isfinite(d)]
    return dict(rms=float(np.sqrt((d ** 2).mean())),
                p50=float(np.percentile(d, 50)),
                p99=float(np.percentile(d, 99)),
                p9999=float(np.percentile(d, 99.99)),
                mx=float(d.max()),
                n=d.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir")
    ap.add_argument("test_dir")
    ap.add_argument("--perm", help="mesh dir holding m11_perm_node.npy (renumbered test tree)")
    ap.add_argument("--vars", default=",".join(DEFAULT_VARS))
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    perm = np.load(f"{a.perm}/m11_perm_node.npy") if a.perm else None
    if perm is not None:
        # prove the permutation before trusting any physics comparison: lon/lat are written
        # from the mesh, so they must map back EXACTLY
        r, dsr = load(a.ref_dir, "ssh")
        t, dst = load(a.test_dir, "ssh")
        if dsr is not None and "lon" in dsr.variables:
            inv = np.empty_like(perm)
            inv[perm] = np.arange(perm.size)
            ok = np.array_equal(np.take(np.array(dst["lon"][:]), inv), np.array(dsr["lon"][:]))
            print(f"  permutation self-check on lon: {'EXACT' if ok else 'FAILED'}")
            if not ok:
                sys.exit("the permutation does not map the mesh back — comparison would be meaningless")
        for d in (dsr, dst):
            if d is not None:
                d.close()

    hdr = f"{'var':<7}{'n':>12}{'rms':>11}{'p50':>11}{'p99':>11}{'p99.99':>11}{'max':>11}"
    print(f"\n=== {a.label or (a.ref_dir + ' vs ' + a.test_dir)}")
    print(hdr)
    for v in a.vars.split(","):
        r, dsr = load(a.ref_dir, v)
        t, dst = load(a.test_dir, v)
        if r is None or t is None or r.shape != t.shape:
            print(f"{v:<7}  (missing or shape mismatch)")
            continue
        s = stats(r, t, perm)
        print(f"{v:<7}{s['n']:>12,}{s['rms']:11.3e}{s['p50']:11.3e}{s['p99']:11.3e}"
              f"{s['p9999']:11.3e}{s['mx']:11.3e}")
        for d in (dsr, dst):
            d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
