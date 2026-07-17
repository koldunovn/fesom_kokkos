#!/usr/bin/env python3
"""E.PART probe: per-rank 2D-vertex and 3D-node (Σ nlvls) balance of a FESOM dist.

usage: m7_part_spread.py <mesh_dir> <npes> [<npes> ...]

Reads <mesh_dir>/nlvls.out and <mesh_dir>/dist_<npes>/rpart.out.

🔴 FORMAT (rule 0.36, session 13/14): rpart.out = [npes] [per-rank counts] then a
gid → NEW-INDEX PERMUTATION (1-based; entry g-1 holds the partition-sorted new
index of gid g). It is NOT rank-major gid lists — that misreading produced the
session-13 §6 "22 %/51 % 3D imbalance" artifact. rank(g) = the count-block the
new index falls in. Verified against check_partitioning's own numbers
(fvom_init.F90:1844-1856) to the digit.
"""
import sys
import numpy as np

def load_ranks(base, n_req):
    """-> (rank_of_gid [nod2D], counts [n]) under the verified format."""
    raw = np.fromstring(open(f"{base}/dist_{n_req}/rpart.out").read(),
                        dtype=np.int64, sep=" ")
    n = int(raw[0])
    assert n == n_req, (n, n_req)
    cnt = raw[1:1 + n]
    perm = raw[1 + n:]
    off = np.concatenate(([0], np.cumsum(cnt)))
    assert perm.size == off[-1], (perm.size, off[-1])
    rank_of = np.searchsorted(off[1:], perm, side="left")
    return rank_of, cnt

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    base = sys.argv[1]
    nlvls = np.loadtxt(f"{base}/nlvls.out", dtype=np.int64)
    print(f"mesh {base}: nod2D={nlvls.size} nlvls[{nlvls.min()},{nlvls.max()}] "
          f"3D nodes total {nlvls.sum()/1e6:.1f}M")
    for np_s in sys.argv[2:]:
        n = int(np_s)
        rank_of, cnt = load_ranks(base, n)
        assert nlvls.size == rank_of.size, (nlvls.size, rank_of.size)
        s3 = np.zeros(n)
        np.add.at(s3, rank_of, nlvls.astype(np.float64))
        c2 = cnt.astype(np.float64)
        for name, v in (("2D verts", c2), ("3D nodes", s3)):
            print(f"  dist_{n} {name}: mean={v.mean():.0f} min={v.min():.0f} "
                  f"max={v.max():.0f} spread={(v.max()-v.min())/v.mean()*100:.2f}% "
                  f"worst=+{(v.max()/v.mean()-1)*100:.2f}%")

if __name__ == "__main__":
    main()
