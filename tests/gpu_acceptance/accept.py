#!/usr/bin/env python3
"""
Verdict for the CUDA acceptance gates (port-paper v2, plan Task 5).

CUDA on Levante A100 is not run-to-run reproducible, so "binary X and binary Y
give the same answer" is not a testable statement there.  Every gate on GPU is
therefore a statement about MAGNITUDE relative to the hardware's own spread:

    ratio = (largest difference BETWEEN the two arms)
          / (largest difference WITHIN one arm)

with n legs per arm, so both maxima are estimated from the same job, the same
allocation and the same number of legs per arm.

**The threshold is 10, and it is not a significance level.**  What these gates
exist to catch is a rails bug -- a missing modify_host()/sync_device() that lets
a kernel read a stale mirror.  That failure has a measured signature in this
project: ~1e-1 against a ~1e-3 floor (KOKKOS_PORTING_LESSONS L; the one real
instance, a pgf_x snapshot, read 1.4e-3 where 2.0e-6 was correct -- a factor of
700).  Ten is a decade below the smallest such signature and about five times
the largest ratio a legitimate device-side change has produced here.

**Distributional difference is NOT the failure criterion, and this is the point
the first version of this script got wrong.**  It used a permutation test on
"are these two samples of one population", which any change to device kernels
fails: reordering atomics changes the rounding DISTRIBUTION even when the code
is bit-identical on Serial.  Tier 1 as a whole separates from knobs-off that way
-- correctly, since it changes memory layout and fences -- and reading that as a
fidelity failure would be a category error.  The permutation p is still printed,
because "the arms are distinguishable at all" is worth knowing, but it is
DESCRIPTIVE.  A comparison passes on magnitude.

Fields identical in every leg (mesh statics, anything not yet touched at the
snapshot step) are reported as a count and excluded: a ratio of 0/0 is not a
number.

Usage:
    accept.py --root DIR --arm NAME=leg1,leg2,... --arm NAME=... \
              --compare A:B [--compare C:D] [--snap FILE] [--threshold 10]

Exit 0 all comparisons pass, 1 a comparison fails, 2 the data could not be read
-- which is a different thing, and conflating the two is how a gate reports a
crash as a scientific result.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import netCDF4 as nc
import numpy as np


def snapshot(leg_dir, snap):
    """The snapshot file to read from one leg directory."""
    if snap:
        p = os.path.join(leg_dir, snap)
        if not os.path.exists(p):
            raise LookupError(f"{p} does not exist")
        return p
    files = sorted(glob.glob(os.path.join(leg_dir, "snap_*.nc")))
    if not files:
        raise LookupError(f"no snap_*.nc in {leg_dir} — the leg did not run")
    return files[-1]


def field_names(paths):
    """Variables present in every leg, with their per-leg shape agreed."""
    common, shapes = None, {}
    for p in paths:
        with nc.Dataset(p) as ds:
            here = {n: ds.variables[n].shape for n in ds.variables}
        common = set(here) if common is None else (common & set(here))
        for n, s in here.items():
            shapes.setdefault(n, set()).add(s)
    return sorted(n for n in common if len(shapes[n]) == 1)


def pair_matrix(paths, name):
    """Symmetric matrix of max|a-b| over every pair of legs, for one field."""
    data = []
    for p in paths:
        with nc.Dataset(p) as ds:
            data.append(np.asarray(ds.variables[name][...], dtype=np.float64))
    n = len(data)
    m = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = float(np.abs(data[i] - data[j]).max())
        m[i, j] = m[j, i] = d
    return m


def splits(na, nb):
    """Every balanced relabelling of the na+nb legs, the observed one first.

    The complement of a split names the same two groups, so only half are
    enumerated when the arms are the same size.
    """
    legs = list(range(na + nb))
    out, seen = [], set()
    for a in itertools.combinations(legs, na):
        b = tuple(x for x in legs if x not in a)
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append((list(a), list(b)))
    return out


def cross_max(m, a, b):
    return max(m[i, j] for i in a for j in b)


def cross_mean(m, a, b):
    """Mean of the cross-arm pair differences.

    The statistic the permutation column uses, deliberately NOT the max: the max
    is what the verdict is built on because it is what a stale mirror moves, but
    it is a poor detector of a small systematic offset, and the whole point of
    the p column is to say whether such an offset exists at all.
    """
    return float(np.mean([m[i, j] for i in a for j in b]))


def within_max(m, a, b):
    inner = [m[i, j] for g in (a, b) for i, j in itertools.combinations(g, 2)]
    return max(inner) if inner else float("nan")


def compare(root, arms, a_name, b_name, snap, threshold):
    a_legs, b_legs = arms[a_name], arms[b_name]
    paths = [snapshot(os.path.join(root, d), snap) for d in a_legs + b_legs]
    na, nb = len(a_legs), len(b_legs)
    ia, ib = list(range(na)), list(range(na, na + nb))
    relab = splits(na, nb)

    print(f"\n=== {a_name} ({na} legs) vs {b_name} ({nb} legs) ===")
    print(f"    snapshot: {os.path.basename(paths[0])}   "
          f"threshold: cross/within < {threshold}")
    print(f"    {'field':16s} {'within-'+a_name:>13s} {'within-'+b_name:>13s} "
          f"{'cross':>13s} {'ratio':>7s} {'p':>6s}")

    names = field_names(paths)
    inert, over, tested, worst = [], [], 0, (0.0, None)
    joint_obs, joint_null = 0.0, np.zeros(len(relab))
    for name in names:
        m = pair_matrix(paths, name)
        if m.max() == 0.0:
            inert.append(name)
            continue
        tested += 1
        wa = max((m[i, j] for i, j in itertools.combinations(ia, 2)), default=float("nan"))
        wb = max((m[i, j] for i, j in itertools.combinations(ib, 2)), default=float("nan"))
        xs = cross_max(m, ia, ib)
        floor = within_max(m, ia, ib)
        ratio = xs / floor if floor > 0 else float("inf")

        # Per-field permutation p, on the same max statistic. Descriptive: with
        # ~19 fields at once these are not 19 independent verdicts, and the
        # joint p below is the one that answers "are the arms distinguishable".
        stats = [cross_mean(m, a, b) for a, b in relab]
        obs = cross_mean(m, ia, ib)
        p = sum(1 for s in stats if s >= obs - 1e-300) / len(stats)

        # Joint statistic: each field scaled by its own median pair difference,
        # so a field's units cannot decide the answer, then averaged.
        scale = float(np.median(m[np.triu_indices_from(m, 1)]))
        if scale > 0:
            joint_obs += obs / scale
            joint_null += np.array(stats) / scale

        flag = "  <-- OVER THRESHOLD" if ratio >= threshold else ""
        print(f"    {name:16s} {wa:13.3e} {wb:13.3e} {xs:13.3e} {ratio:7.2f} {p:6.3f}{flag}")
        if ratio >= threshold:
            over.append((name, ratio, floor, xs))
        if ratio > worst[0]:
            worst = (ratio, name)

    print(f"    ({len(inert)} field(s) identical in every leg: {', '.join(inert)})")
    if not tested:
        print("    ERROR: every field was identical in every leg — either the legs never "
              "advanced, or they are the same run. Not a result.")
        return 2, None

    joint_p = float(np.mean(joint_null >= joint_obs - 1e-300))
    print(f"    joint permutation p = {joint_p:.3f} over {len(relab)} relabellings "
          f"(descriptive — whether the arms are separable at all, not whether it matters;\n"
          f"     the smallest reachable value is {1/len(relab):.3f})")
    result = {"a": a_name, "b": b_name, "fields": tested, "worst_ratio": worst[0],
              "worst_field": worst[1], "joint_p": joint_p, "threshold": threshold}

    if over:
        print(f"    FAIL: {len(over)} of {tested} fields exceed the threshold — this is the "
              f"magnitude of a stale mirror, not of reordered arithmetic")
        for name, ratio, floor, xs in over:
            print(f"          {name}: {xs:.3e} across the arms against a {floor:.3e} floor "
                  f"({ratio:.1f}x)")
        return 1, result
    print(f"    PASS: the arms differ by at most {worst[0]:.2f}x the spread of the hardware "
          f"itself (worst field {worst[1]}, {tested} fields tested)")
    return 0, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=dir1,dir2,...")
    ap.add_argument("--compare", action="append", required=True, metavar="A:B")
    ap.add_argument("--snap", default=None,
                    help="snapshot file name; default = the last one in each leg")
    ap.add_argument("--threshold", type=float, default=10.0)
    ap.add_argument("--json", default=None, help="write the summary here")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, legs = spec.partition("=")
        arms[name] = [d for d in legs.split(",") if d]

    rc, summary = 0, []
    for spec in args.compare:
        a, _, b = spec.partition(":")
        for name in (a, b):
            if name not in arms:
                print(f"ERROR: --compare names arm {name!r}, which has no --arm", file=sys.stderr)
                return 2
        try:
            code, result = compare(args.root, arms, a, b, args.snap, args.threshold)
        except (LookupError, OSError) as exc:
            # "could not compare" is not "differs".
            print(f"\n=== {a} vs {b} ===\n    ERROR: {exc}")
            code, result = 2, None
        rc = max(rc, code)
        if result:
            result["verdict"] = {0: "pass", 1: "fail"}[code]
            summary.append(result)
    if args.json:
        json.dump(summary, open(args.json, "w"), indent=2)
        print(f"\nwritten: {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
