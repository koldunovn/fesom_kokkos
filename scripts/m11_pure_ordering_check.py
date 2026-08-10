#!/usr/bin/env python3
"""M11: refuse to race ordering arms that are not pure-ordering arms.

Finding 10: for one whole race day the base arm pointed at the SHIPPED mesh while the ordering
arms carried the SETTLED partition, and at N=256 those are different partitions. The offline
invariant-block gate (Task 5) would have caught it, but it is run at dist-BUILD time and cannot
see which mesh a race job later points at. This check runs INSIDE the race job, against the
files it is about to hand the model.

The test is exact and needs no tolerance: an arm is a pure-ordering arm iff its part vector
equals the reference's under the node permutation,

    part_arm[new] == part_ref[perm[new]]          with perm = m11_perm_node.npy (new -> old)

usage:
  m11_pure_ordering_check.py --npes 256 --base <mesh> --arm hil=<mesh> --arm rcm=<mesh>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_part_spread import load_ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=MESHDIR")
    a = ap.parse_args()

    ref, _ = load_ranks(a.base, a.npes)
    ref = np.asarray(ref, dtype=np.int64)
    print(f"  reference partition: {a.base}/dist_{a.npes}  ({ref.size} nodes, "
          f"{int(ref.max()) + 1} parts)")
    bad = 0
    for spec in a.arm:
        name, mesh = spec.split("=", 1)
        v, _ = load_ranks(mesh, a.npes)
        v = np.asarray(v, dtype=np.int64)
        permf = f"{mesh}/m11_perm_node.npy"
        if os.path.exists(permf):
            perm = np.load(permf)                       # new position -> old index
            ok = v.size == ref.size and np.array_equal(v, ref[perm])
            how = "under the node permutation"
        else:
            ok = v.size == ref.size and np.array_equal(v, ref)
            how = "directly (no permutation on this mesh)"
        n_diff = 0 if ok else int((v != (ref[perm] if os.path.exists(permf) else ref)).sum())
        print(f"  {name:<6} {'EXACT' if ok else f'DIFFERS at {n_diff} node(s)'}  {how}")
        bad += 0 if ok else 1
    if bad:
        print("  REFUSE: these are not pure-ordering arms — a race between them would be an "
              "ordering-plus-repartitioning race (Finding 10)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
