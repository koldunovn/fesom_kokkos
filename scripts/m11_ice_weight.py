#!/usr/bin/env python3
"""M11: build an ICE-AWARE vertex weight, w = a + nlev + b*ice, for the partitioners.

Why, and what is already known against it. M10 tested the ice hypothesis on CPU at 864 ranks by
correlating per-rank ocean busy time against node counts: 3-D nodes r=0.967, 2-D nodes r=0.003,
and CG wait against 3-D r=-0.771. Their conclusion — "we wait for DEEP ranks, not icy ones" —
is the reason FESOM's weighting is bathymetric and the reason this file needs a justification.

What is NOT covered by that test: the GPU at small rank counts. M9 measured the sea ice at
5-13 % of the GPU step and, separately, that the ice does not strong-scale on GPU at all. And
the ice work is spatially concentrated in a way the ocean work is not — measured on CORE2 with
the model's own January ice mask (a_ice > 0.01, 29.4 % of nodes):

    ranks   ice nodes/part max/mean     3-D nodes max/min
      4          1.86                        1.01
      8          3.40                        1.97
    512          3.42                        9.26

At 4 ranks the decomposition is balanced in every ocean currency and unbalanced by 1.86x in the
ice one. Whether that costs anything depends on whether the ice kernels are proportional to
ice-covered nodes or run unconditionally over all of them — which the PHASESTATS per-rank ice
sub-phase answers directly, and which this weight only becomes worth using if it does.

The mask comes from a model run rather than a latitude cut, so it is the ice the model actually
carries.

usage:
  m11_ice_weight.py <mesh_dir> --ice a_ice.npy -a 100 -b 45 -o w.txt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_scorecard import Mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir")
    ap.add_argument("--ice", required=True, help="per-node ice field (.npy) from a model run")
    ap.add_argument("--thresh", type=float, default=0.01)
    ap.add_argument("-a", type=int, default=100, help="constant term (2-D work)")
    ap.add_argument("-b", type=int, default=45, help="ice bonus per ice-covered node")
    ap.add_argument("--nlev-scale", type=float, default=1.0, help="multiplier on nlev (3-D work)")
    ap.add_argument("-o", "--out", required=True)
    x = ap.parse_args()

    mesh = Mesh(x.mesh_dir, need_edges=False)
    ice = np.load(x.ice).reshape(-1)
    if ice.size != mesh.nod2D:
        sys.exit(f"ice field has {ice.size} entries, mesh has {mesh.nod2D}")
    m = (ice > x.thresh).astype(np.int64)
    w = (x.a + np.round(x.nlev_scale * mesh.nlev_nod).astype(np.int64) + x.b * m)
    if w.min() < 1:
        sys.exit("weights must be >= 1")
    np.savetxt(x.out, w, fmt="%d")
    print(f"wrote {x.out}: w = {x.a} + {x.nlev_scale:g}*nlev + {x.b}*[a_ice>{x.thresh}]")
    print(f"  ice nodes {int(m.sum()):,} of {mesh.nod2D:,} ({100*m.mean():.1f} %)   "
          f"w range {int(w.min())}..{int(w.max())}   sum {int(w.sum()):,} "
          f"(int32 ledger {'OK' if w.sum() < 2**30 else 'EXCEEDED'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
