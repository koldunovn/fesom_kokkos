#!/usr/bin/env python3
"""M11: turn an external partitioner's output into a FESOM_PART_FILE part vector.

The injection seam is one line (`fvom_init.F90:1792`): everything downstream of `do_partit`
consumes only `part[]`. So any engine — KaMinPar, Mt-KaHyPar, KaHIP — can produce the
decomposition while the battle-tested Fortran tool still writes the dist files, which keeps
the on-disk format and Fortran FESOM compatibility for free.

FESOM_PART_FILE format: nod2D whitespace-separated integers, 0-based rank per node, in node
order. The injection path skips `fort_part.c`'s `part[i]--`, so what is written here is what
the partitioner uses.

Sanity checks refuse to emit a vector that would waste a partitioner run or, worse, produce a
dist that fails only at model start: exact length, 0-based contiguous range, and every part
non-empty (METIS/FESOM assume npes non-empty parts; an empty one makes a rank with no nodes).

usage:
  m11_part_import.py engine_out.txt -o part.txt [--npes 512] [--base 0|1|auto]
  m11_part_import.py --from-dist <mesh_dir> --npes 512 -o part.txt
  m11_part_import.py --verify part.txt --mesh <mesh_dir> [--npes 512]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_part_spread import load_ranks


def detect_base(v, npes_hint=None):
    """0- or 1-based block ids? Engines emit 0-based; the 1-based path is a safety net."""
    lo, hi = int(v.min()), int(v.max())
    if lo == 0:
        return 0, f"min=0 => 0-based (npes={hi + 1})"
    if lo == 1 and npes_hint is not None and hi == npes_hint:
        return 1, f"min=1, max=npes={npes_hint} => 1-based"
    if lo == 1 and npes_hint is None:
        return 1, f"min=1 => assuming 1-based (npes={hi}); pass --base 0 if the engine "
    raise SystemExit(f"cannot decide 0- vs 1-based (min={lo}, max={hi}); pass --base explicitly")


def sanity(part, npes, nod2D=None, label=""):
    bad = []
    if nod2D is not None and part.size != nod2D:
        bad.append(f"length {part.size:,} != nod2D {nod2D:,}")
    if int(part.min()) != 0:
        bad.append(f"min rank {int(part.min())} != 0")
    if int(part.max()) != npes - 1:
        bad.append(f"max rank {int(part.max())} != npes-1 = {npes - 1}")
    cnt = np.bincount(part, minlength=npes)
    empty = np.nonzero(cnt == 0)[0]
    if empty.size:
        bad.append(f"{empty.size} EMPTY part(s): {empty[:10].tolist()}")
    print(f"  {label}nodes {part.size:,} | npes {npes} | per-part min {int(cnt.min()):,} "
          f"max {int(cnt.max()):,} mean {cnt.mean():.1f} | imbalance {cnt.max()/cnt.mean():.4f}")
    if bad:
        for b in bad:
            print(f"  SANITY FAIL: {b}", file=sys.stderr)
        return False
    return True


def histogram(part, npes, nbins=10):
    cnt = np.bincount(part, minlength=npes)
    lo, hi = int(cnt.min()), int(cnt.max())
    if lo == hi:
        print(f"  size histogram: all {npes} parts hold {lo:,} nodes")
        return
    if hi - lo < nbins:                       # perfectly balanced: an exact tally reads better
        vals, cts = np.unique(cnt, return_counts=True)
        print(f"  part sizes ({npes} parts): "
              + ", ".join(f"{int(v):,}x{int(c)}" for v, c in zip(vals, cts)))
        return
    edges = np.linspace(lo, hi + 1, nbins + 1)
    h, _ = np.histogram(cnt, bins=edges)
    w = max(1, int(h.max()))
    print(f"  part-size histogram ({npes} parts, {lo:,}..{hi:,} nodes):")
    for k in range(nbins):
        bar = "#" * int(40 * h[k] / w)
        print(f"    {edges[k]:9.0f}-{edges[k+1]:9.0f} | {h[k]:5d} {bar}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine_out", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--npes", type=int)
    ap.add_argument("--base", choices=["0", "1", "auto"], default="auto")
    ap.add_argument("--from-dist", metavar="MESH_DIR",
                    help="extract the part vector from MESH_DIR/dist_<npes> instead")
    ap.add_argument("--verify", metavar="PART_FILE", help="check an existing part file")
    ap.add_argument("--mesh", help="mesh dir, for the nod2D length check")
    a = ap.parse_args()

    nod2D = None
    if a.mesh or a.from_dist:
        d = a.mesh or a.from_dist
        with open(f"{d}/nod2d.out") as f:
            nod2D = int(f.readline())

    if a.verify:
        part = np.loadtxt(a.verify, dtype=np.int64).reshape(-1)
        npes = a.npes or int(part.max()) + 1
        print(f"verify {a.verify}")
        ok = sanity(part, npes, nod2D, "")
        histogram(part, npes)
        return 0 if ok else 1

    if a.from_dist:
        if not a.npes:
            ap.error("--from-dist needs --npes")
        part, cnt = load_ranks(a.from_dist, a.npes)
        part = np.asarray(part, dtype=np.int64)
        print(f"extracted from {a.from_dist}/dist_{a.npes}/rpart.out")
        # rpart's own count block is an independent witness of the same partition
        got = np.bincount(part, minlength=a.npes)
        if not np.array_equal(got, np.asarray(cnt, dtype=np.int64)):
            sys.exit("FAIL: per-rank counts derived from the permutation disagree with "
                     "rpart's count block — the parser or the file is wrong")
        print(f"  rpart count block agrees with the derived assignment on all {a.npes} ranks")
        npes = a.npes
    else:
        if not a.engine_out:
            ap.error("give an engine output file, or --from-dist / --verify")
        v = np.loadtxt(a.engine_out, dtype=np.int64).reshape(-1)
        if a.base == "auto":
            base, why = detect_base(v, a.npes)
            print(f"{a.engine_out}: {why}")
        else:
            base = int(a.base)
            print(f"{a.engine_out}: --base {base} (explicit)")
        part = v - base
        npes = a.npes or int(part.max()) + 1

    ok = sanity(part, npes, nod2D, "")
    histogram(part, npes)
    if not ok:
        return 1
    if a.out:
        # one rank per line: the partitioner reads it with list-directed input, and a
        # column is diffable
        np.savetxt(a.out, part, fmt="%d")
        print(f"wrote {a.out} ({part.size:,} entries, 0-based, FESOM_PART_FILE format)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
