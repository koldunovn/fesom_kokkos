#!/usr/bin/env python3
"""M11 session 2: apply the re-registered ordering gates R1/R2/R3 to a gate-leg job tree.

The three gates are specified in docs/PARTITIONING_M11.md, "RE-REGISTRATION — ordering gates,
v2", written before the legs ran. Constants live here as named module-level values so the code
and the pre-registration can be diffed against each other:

  R1  step-1 field identity   median |dT|,|dS|,|dssh| <= 1e-12    (blind, first-principles)
  R2  partition-class floor   arm rms <= 3 x max(control rms)     (ensemble of 3 controls)
  R3  SSH iterations          no systematic shift, magnitude within 2x the control ensemble

Every arm AND every control passes through the same instrument, and the controls are printed
next to the arms so a reader can apply their own multiplier.

usage:
  m11_gate2_analyze.py <jobdir> --ref settled --controls ship,wgt0,seed --arms hil,rcm \
                       --perm hil=/path/to/core2_hil,rcm=/path/to/core2_rcm
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_accuracy_compare import load, stats          # one implementation, not two

R1_MEDIAN = 1e-12          # K / psu / m
R2_K = 3.0                 # arm rms <= R2_K * max(control rms)
R3_K = 2.0                 # magnitude multiplier on the control ensemble
R3_SHIFT_FLOOR = 0.5       # |mean signed d| is never failed below this
R3_MAX_FLOOR = 2.0         # max |d| is never failed below this
VARS = ["temp", "salt", "ssh"]

IT = re.compile(r"^\s*(\d+)\s+it=\s*(\d+)\s")


def iters(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path, errors="ignore"):
        m = IT.match(line)
        if m:
            d[int(m.group(1))] = int(m.group(2))
    return d


def field_stats(ref_dir, test_dir, perm):
    out = {}
    for v in VARS:
        r, dsr = load(ref_dir, v)
        t, dst = load(test_dir, v)
        if r is None or t is None or r.shape != t.shape:
            out[v] = None
            continue
        out[v] = stats(r, t, perm)
        for d in (dsr, dst):
            d.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdir")
    ap.add_argument("--ref", default="settled")
    ap.add_argument("--controls", default="ship,wgt0,seed")
    ap.add_argument("--arms", default="hil,rcm")
    ap.add_argument("--perm", default="", help="comma list arm=meshdir holding m11_perm_node.npy")
    a = ap.parse_args()

    J = a.jobdir
    controls = [c for c in a.controls.split(",") if c]
    arms = [x for x in a.arms.split(",") if x]
    permdir = dict(kv.split("=", 1) for kv in a.perm.split(",") if kv)
    perm = {k: np.load(f"{v}/m11_perm_node.npy") for k, v in permdir.items()}
    fail = 0

    # ---------------------------------------------------------------- permutation self-check
    print("=== permutation self-check on lon (must be EXACT before any physics is read)")
    for k, p in perm.items():
        r, dsr = load(f"{J}/s1/{a.ref}", "ssh")
        t, dst = load(f"{J}/s1/{k}", "ssh")
        inv = np.empty_like(p)
        inv[p] = np.arange(p.size)
        ok = np.array_equal(np.take(np.array(dst["lon"][:]), inv), np.array(dsr["lon"][:]))
        print(f"  {k:<8} {'EXACT' if ok else 'FAILED'}")
        fail += 0 if ok else 1
        dsr.close(); dst.close()

    # ---------------------------------------------------------------------------------- R1
    print(f"\n=== R1  step-1 field identity   [median |d| <= {R1_MEDIAN:g}]")
    print(f"  {'leg':<10}{'kind':<10}" + "".join(f"{v+' p50':>13}{v+' max':>13}" for v in VARS))
    r1 = {}
    for tag in controls + arms:
        s = field_stats(f"{J}/s1/{a.ref}", f"{J}/s1/{tag}", perm.get(tag))
        r1[tag] = s
        kind = "control" if tag in controls else "ARM"
        row = "".join((f"{s[v]['p50']:13.3e}{s[v]['mx']:13.3e}" if s[v] else f"{'--':>26}")
                      for v in VARS)
        print(f"  {tag:<10}{kind:<10}{row}")
    for tag in controls + arms:
        bad = [v for v in VARS if r1[tag][v] and r1[tag][v]["p50"] > R1_MEDIAN]
        verdict = "PASS" if not bad else "FAIL " + ",".join(bad)
        print(f"    {tag:<8} R1 {verdict}")
        if bad:
            fail += 1
            if tag in controls:
                print("      !! a CONTROL failed R1 — the instrument is not measuring round-off,"
                      " and the gate is void, not the arm")

    # ---------------------------------------------------------------------------------- R2
    print(f"\n=== R2  partition-class floor at 20 steps   [arm rms <= {R2_K:g} x max(control rms)]")
    print(f"  {'leg':<10}{'kind':<10}" + "".join(f"{v+' rms':>13}" for v in VARS))
    r2 = {}
    for tag in controls + arms:
        s = field_stats(f"{J}/s20/{a.ref}", f"{J}/s20/{tag}", perm.get(tag))
        r2[tag] = s
        kind = "control" if tag in controls else "ARM"
        row = "".join((f"{s[v]['rms']:13.3e}" if s[v] else f"{'--':>13}") for v in VARS)
        print(f"  {tag:<10}{kind:<10}{row}")
    floor = {v: max((r2[c][v]["rms"] for c in controls if r2[c][v]), default=None) for v in VARS}
    print("  " + "-" * 60)
    print(f"  {'floor':<20}" + "".join((f"{floor[v]:13.3e}" if floor[v] else f"{'--':>13}")
                                       for v in VARS))
    for tag in arms:
        bad = []
        for v in VARS:
            if r2[tag][v] and floor[v] and r2[tag][v]["rms"] > R2_K * floor[v]:
                bad.append(v)
        ratios = " ".join(f"{v}={r2[tag][v]['rms']/floor[v]:.2f}x" for v in VARS
                          if r2[tag][v] and floor[v])
        print(f"    {tag:<8} R2 {'PASS' if not bad else 'FAIL ' + ','.join(bad)}   ({ratios})")
        fail += 0 if not bad else 1

    # ---------------------------------------------------------------------------------- R3
    print(f"\n=== R3  SSH iterations over the 20-step leg   [no systematic shift; magnitude "
          f"within {R3_K:g}x the control ensemble]")
    ref = iters(f"{J}/log_s20_{a.ref}.txt")
    if not ref:
        print("  PARSE FAILED — no 'it=' lines in the reference log")
        return 1
    st = {}
    for tag in controls + arms:
        d = iters(f"{J}/log_s20_{tag}.txt")
        steps = sorted(set(ref) & set(d))
        dd = np.array([d[s] - ref[s] for s in steps], dtype=float)
        st[tag] = dict(n=len(steps), signed=float(dd.mean()), absmean=float(np.abs(dd).mean()),
                       absmax=float(np.abs(dd).max()) if len(dd) else 0.0,
                       seq=[int(x) for x in dd])
        kind = "control" if tag in controls else "ARM"
        print(f"  {tag:<10}{kind:<10}n={st[tag]['n']:<4}mean signed {st[tag]['signed']:+7.3f}   "
              f"mean|d| {st[tag]['absmean']:6.3f}   max|d| {st[tag]['absmax']:.0f}")
        print(f"      per-step d: {st[tag]['seq']}")
    c_shift = max(abs(st[c]["signed"]) for c in controls)
    c_absmean = max(st[c]["absmean"] for c in controls)
    c_absmax = max(st[c]["absmax"] for c in controls)
    lim_shift = max(R3_SHIFT_FLOOR, R3_K * c_shift)
    lim_absmean = R3_K * c_absmean
    lim_absmax = max(R3_MAX_FLOOR, R3_K * c_absmax)
    print(f"  control ensemble: |mean signed| <= {c_shift:.3f}, mean|d| <= {c_absmean:.3f}, "
          f"max|d| <= {c_absmax:.0f}")
    print(f"  bounds from it   : |mean signed| <= {lim_shift:.3f}, mean|d| <= {lim_absmean:.3f}, "
          f"max|d| <= {lim_absmax:.0f}")
    for tag in arms:
        s = st[tag]
        bad = []
        if abs(s["signed"]) > lim_shift:
            bad.append("systematic shift")
        if s["absmean"] > lim_absmean:
            bad.append("mean|d|")
        if s["absmax"] > lim_absmax:
            bad.append("max|d|")
        print(f"    {tag:<8} R3 {'PASS' if not bad else 'FAIL ' + '; '.join(bad)}")
        fail += 0 if not bad else 1

    print(f"\n=== gate2: {'ALL GREEN' if fail == 0 else str(fail) + ' FAILURE(S)'}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
