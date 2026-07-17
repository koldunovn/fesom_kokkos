#!/usr/bin/env python3
"""E.PART probe: per-rank 2D-vertex and 3D-node (Σ nlvls) balance of a FESOM dist.

usage: m7_part_spread.py <mesh_dir> <npes> [<npes> ...]

Reads <mesh_dir>/nlvls.out and <mesh_dir>/dist_<npes>/rpart.out
(format: line 1 = npes, then npes per-rank counts, then the concatenated
per-rank vertex id lists, 1-based). Prints min/mean/max and spread for both
the 2D and the Σnlvls (3D) axes — the numbers the session-13 recon (§6) and
the E.PART pre-registration are stated in.
"""
import sys
import numpy as np

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    base = sys.argv[1]
    nlvls = np.loadtxt(f"{base}/nlvls.out", dtype=np.int64)
    print(f"mesh {base}: nod2D={nlvls.size} nlvls[{nlvls.min()},{nlvls.max()}] "
          f"3D nodes total {nlvls.sum()/1e6:.1f}M")
    for np_s in sys.argv[2:]:
        n_req = int(np_s)
        raw = np.fromstring(open(f"{base}/dist_{n_req}/rpart.out").read(),
                            dtype=np.int64, sep=" ")
        n = int(raw[0])
        assert n == n_req, (n, n_req)
        cnt = raw[1:1 + n]
        ids = raw[1 + n:]
        assert ids.size == nlvls.size, (ids.size, nlvls.size)
        off = np.concatenate(([0], np.cumsum(cnt)))
        s3 = np.array([nlvls[ids[off[r]:off[r + 1]] - 1].sum() for r in range(n)],
                      dtype=np.float64)
        c2 = cnt.astype(np.float64)
        for name, v in (("2D verts", c2), ("3D nodes", s3)):
            print(f"  dist_{n} {name}: mean={v.mean():.0f} min={v.min():.0f} "
                  f"max={v.max():.0f} spread={(v.max()-v.min())/v.mean()*100:.2f}% "
                  f"worst=+{(v.max()/v.mean()-1)*100:.2f}%")

if __name__ == "__main__":
    main()
