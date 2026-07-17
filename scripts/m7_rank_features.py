#!/usr/bin/env python3
"""E.IMB.0 harvest helper: per-rank static features of a FESOM dist, for joining
against the [phasestats-rank] table (busy_ice[rank] vs polar fraction is the
ice-concentration discriminator).

usage: m7_rank_features.py <mesh_dir> <npes> [<npes> ...]

Per rank: 2D verts, 3D nodes (Σ nlvls), polar fraction |lat|>50°, |lat|>60°,
mean |lat|. Rank assignment comes from m7_part_spread.load_ranks — the rule-0.36
VERIFIED rpart.out parser (gid → new-index permutation), never a re-implementation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_part_spread import load_ranks  # noqa: E402  (the verified parser, rule 0.36)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    base = sys.argv[1]
    nlvls = np.loadtxt(f"{base}/nlvls.out", dtype=np.int64)
    lat = np.loadtxt(f"{base}/nod2d.out", skiprows=1, usecols=(2,))
    assert lat.size == nlvls.size, (lat.size, nlvls.size)
    alat = np.abs(lat)
    for np_s in sys.argv[2:]:
        n = int(np_s)
        rank_of, cnt = load_ranks(base, n)
        print(f"# dist_{n}  (mesh {base}, nod2D={lat.size})")
        print("# rank\tn2d\tn3d\tpolar50%\tpolar60%\tNH50%\tSH50%\tmean|lat|")
        p50 = p60 = 0
        for r in range(n):
            m = rank_of == r
            f50 = 100.0 * np.mean(alat[m] > 50.0)
            f60 = 100.0 * np.mean(alat[m] > 60.0)
            nh = 100.0 * np.mean(lat[m] > 50.0)    # January ice lives here
            sh = 100.0 * np.mean(lat[m] < -50.0)
            p50 += f50 > 80.0
            p60 += f50 < 2.0
            print(f"{r}\t{int(cnt[r])}\t{int(nlvls[m].sum())}\t{f50:.1f}\t{f60:.1f}\t{nh:.1f}\t{sh:.1f}\t{np.mean(alat[m]):.1f}")
        print(f"# summary dist_{n}: {p50}/{n} ranks >80% polar50, {p60}/{n} ranks <2% polar50 (≈ice-free)")


if __name__ == "__main__":
    main()
