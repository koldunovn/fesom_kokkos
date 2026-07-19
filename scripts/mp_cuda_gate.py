#!/usr/bin/env python3
"""M8 per-slice CUDA gate: noise-envelope comparison (NOT bit-identity).

CUDA runs are run-to-run non-bit-reproducible (atomics scatter order; proven by the
same-binary control 2026-07-19: pi, 20 steps, per-field max|diff| 1e-20..1e-14 between two runs
of the IDENTICAL binary). Bit-identity therefore lives with the Serial oracle only; the CUDA
slice gate instead demands that the edit-vs-ref difference is INDISTINGUISHABLE from same-binary
noise: for every (snapshot, field),

    max|new - ref|  <=  max( K * noise(snap, field),  ABS_FLOOR )

where noise comes from a same-binary rerun pair, K=10 (pre-registered margin), and
ABS_FLOOR=1e-13 guards fields whose sampled noise happened to be ~0 while still being far below
any physical signal. Fields bit-identical in BOTH pairs stay bit-identical here (diff 0 <= floor).

Usage: mp_cuda_gate.py <ref_dir> <new_dir> <noise_a_dir> <noise_b_dir>
       (noise_a/noise_b = two runs of the SAME binary, typically new_dir and a rerun)
Exit 0 = PASS, 1 = FAIL.
"""
import sys
import pathlib
import netCDF4 as nc
import numpy as np

K = 10.0
ABS_FLOOR = 1e-13


def field_diffs(a_dir: pathlib.Path, b_dir: pathlib.Path):
    """{(snap_name, var): max|a-b|} over common snap_*.nc and common vars."""
    out = {}
    for a_path in sorted(a_dir.glob("snap_*.nc")):
        b_path = b_dir / a_path.name
        if not b_path.exists():
            sys.exit(f"FATAL: {b_path} missing — incomparable runs")
        with nc.Dataset(a_path) as da, nc.Dataset(b_path) as db:
            for v in da.variables:
                if v not in db.variables:
                    continue
                va, vb = da.variables[v][...], db.variables[v][...]
                if not np.issubdtype(va.dtype, np.floating):
                    continue
                out[(a_path.name, v)] = float(np.max(np.abs(np.asarray(va, dtype=np.float64)
                                                            - np.asarray(vb, dtype=np.float64))))
    return out


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    ref, new, na, nb = (pathlib.Path(p) for p in sys.argv[1:5])
    edit = field_diffs(ref, new)
    noise = field_diffs(na, nb)
    fails = []
    worst = (0.0, None)
    for key, d in sorted(edit.items()):
        allow = max(K * noise.get(key, 0.0), ABS_FLOOR)
        ratio = d / allow if allow > 0 else 0.0
        if ratio > worst[0]:
            worst = (ratio, key)
        status = "ok" if d <= allow else "FAIL"
        if d > 0 or status == "FAIL":
            print(f"  {key[0]} {key[1]:8s} edit={d:.3e} allow={allow:.3e} [{status}]")
        if d > allow:
            fails.append(key)
    if fails:
        print(f"CUDA GATE FAIL: {len(fails)} field(s) exceed the noise envelope: {fails}")
        sys.exit(1)
    wr, wk = worst
    print(f"CUDA GATE PASS (noise-envelope, K={K}, floor={ABS_FLOOR:g}); "
          f"worst margin used: {wr:.2f} of allowance at {wk}")


if __name__ == "__main__":
    main()
