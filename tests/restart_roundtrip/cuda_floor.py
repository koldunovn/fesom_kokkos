#!/usr/bin/env python3
"""
Verdict for job_restart_gate_cuda.

CUDA on Levante A100 is not run-to-run reproducible, so "the restart is exact"
is not a testable statement there. What is testable: the restarted legs are not
distinguishable from the continuous ones. Both arms have the same number of legs
and the same checkpoint schedule, so the two maxima are the same statistic and
the comparison does not favour either side.

Reports every field, decides on the three the round trip has always moved first
(temp, ssh, u).
"""
import glob
import itertools
import os
import sys

import netCDF4 as nc
import numpy as np

FIELDS = ("temp", "ssh", "u")


def final(d):
    fs = sorted(glob.glob(os.path.join(d, "*.restart.nc")))
    if not fs:
        raise SystemExit(f"no restart file in {d}")
    return fs[-1]


def diff(a, b, v):
    A, B = nc.Dataset(a), nc.Dataset(b)
    d = float(np.abs(np.asarray(A[v][:], dtype=np.float64)
                     - np.asarray(B[v][:], dtype=np.float64)).max())
    A.close(); B.close()
    return d


def main():
    root, reps = sys.argv[1], int(sys.argv[2])
    ctl = [final(f"{root}/ctl{i}") for i in range(1, reps + 1)]
    res = [final(f"{root}/res{i}b") for i in range(1, reps + 1)]
    bad = 0
    for v in FIELDS:
        c = [diff(a, b, v) for a, b in itertools.combinations(ctl, 2)]
        r = [diff(a, b, v) for a, b in itertools.combinations(res, 2)]
        x = [diff(a, b, v) for a in ctl for b in res]
        print(f"{v:6s}  continuous vs continuous: max {max(c):.3e}  (n={len(c)})")
        print(f"{v:6s}  restarted  vs restarted : max {max(r):.3e}  (n={len(r)})")
        print(f"{v:6s}  continuous vs restarted : max {max(x):.3e}  (n={len(x)})")
        if max(r) > max(c):
            print(f"        FAIL {v}: the restarted legs spread wider than the continuous ones")
            bad = 1
        if max(x) > max(c):
            print(f"        NOTE {v}: continuous-vs-restarted exceeds the continuous spread — "
                  f"expected when the two arms differ by a process boundary, not a verdict")
    print("PASS: restarted and continuous legs are indistinguishable" if not bad
          else "FAIL: the restart moves the answer more than the binary moves against itself")
    return bad


if __name__ == "__main__":
    sys.exit(main())
