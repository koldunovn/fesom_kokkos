#!/usr/bin/env python3
"""Exp #1 (DT1800_HANDOFF §4.1): compare the standalone bidiff output (surface
uv_rhs per global element id) between the C and Fortran probes, run on the SAME
prescribed gid-hash velocity field. Identical => the biharmonic operator is
bit-faithful (the dt=1800 bug is upstream, not the operator); a real difference
localizes an operator bug.

Run: /work/ab0995/a270088/mambaforge/envs/nereus/bin/python exp1_compare_bidiff.py
"""
import glob, numpy as np

PROBE = "/work/ab0995/a270088/port/bidiff_probe"

def load(prefix):
    d = {}
    for f in sorted(glob.glob(f"{PROBE}/{prefix}_r*.txt")):
        a = np.loadtxt(f)
        if a.ndim == 1: a = a[None, :]
        for gid, u, v in a:
            d[int(gid)] = (u, v)
    return d

C = load("bidiff_c"); F = load("bidiff_f")
print(f"C elems: {len(C)}   F elems: {len(F)}")
common = sorted(set(C) & set(F))
print(f"common gids: {len(common)}   (only-C: {len(set(C)-set(F))}  only-F: {len(set(F)-set(C))})")

gid = np.array(common)
cu = np.array([C[g][0] for g in common]); cv = np.array([C[g][1] for g in common])
fu = np.array([F[g][0] for g in common]); fv = np.array([F[g][1] for g in common])

au = np.abs(cu-fu); av = np.abs(cv-fv)
# relative to the local magnitude scale of the field
scale = np.maximum(np.hypot(cu,cv), np.hypot(fu,fv))
sc = np.where(scale > 1e-30, scale, 1.0)
ru = au/sc; rv = av/sc

print(f"\n=== uv_rhs (surface) C vs Fortran on identical input ===")
print(f"field magnitude: C |uv_rhs| max={np.hypot(cu,cv).max():.3e}  mean={np.hypot(cu,cv).mean():.3e}")
print(f"max abs diff:  du={au.max():.3e}  dv={av.max():.3e}")
print(f"max rel diff:  du={ru.max():.3e}  dv={rv.max():.3e}")
print(f"mean abs diff: du={au.mean():.3e}  dv={av.mean():.3e}")
for thr in (1e-12, 1e-10, 1e-8, 1e-6, 1e-3):
    n = int(((ru>thr)|(rv>thr)).sum())
    print(f"  elems with rel diff > {thr:.0e}: {n}  ({100.0*n/len(common):.3f}%)")

# worst offenders
worst = np.argsort(-(ru+rv))[:10]
print("\nworst 10 (gid: C_u F_u | C_v F_v | reldiff):")
for i in worst:
    print(f"  gid {gid[i]:7d}: u {cu[i]:+.6e} {fu[i]:+.6e}  v {cv[i]:+.6e} {fv[i]:+.6e}  "
          f"rel {max(ru[i],rv[i]):.2e}")

verdict = "OPERATOR IDENTICAL (bug is upstream)" if max(ru.max(),rv.max()) < 1e-9 \
          else "OPERATOR DIFFERS -> candidate dt=1800 bug"
print(f"\nVERDICT: {verdict}")
