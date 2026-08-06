#!/usr/bin/env python3
"""Report Figure 1: cost of the surviving schemes, measured against the sea-ice cost.

Replaces the earlier model-step version. Two changes, both requested:

  - the denominator is the ICE COST (ice + ice dynamics + ice advection, computation together
    with time spent waiting for communication), not the whole model time step. The step divides
    an ice-only change by about ten and is the wrong measure for judging a sea-ice scheme.
  - the delayed exchange is not shown. It places the domain decomposition in the ice field and
    is not a usable scheme, so it does not belong in a figure comparing options.

Each mesh is at its own operating point -- the largest node count at which the model still gains
from more GPUs, read off the measured strong-scaling curve of the standard scheme.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("/work/ab0995/a270088/port2/m9/m9_results.json"))
OUT = os.path.dirname(os.path.abspath(__file__))

POINTS = [("ic_core2", "CORE2, 127 k nodes\n1 node / 4 GPUs"),
          ("ic_farc",  "fArc, 638 k\n4 nodes / 16 GPUs"),
          ("ic_dars",  "DARS, 3.2 M\n8 nodes / 32 GPUs"),
          ("ic_ng5",   "NG5, 7.4 M\n16 nodes / 64 GPUs")]
SER = [("divergence",   "divergence form alone",                    "#8c8c8c"),
       ("wide_std_k8",  "wide halo, standard form ($K$=8)",         "#1f6fb4"),
       ("wide_div_k8",  "wide halo, divergence form ($K$=8)",       "#7fb3dd")]


def ice_cost(ph):
    """The 2019 paper's definition; icedyn busy alone undercounts the ice by about 3x."""
    return sum(ph[k]["busy_mean"] + ph[k]["wait_mean"]
               for k in ("ice", "icedyn", "iceadv") if k in ph)


rows = []
for tag, lab in POINTS:
    r = next((x for x in D["runs"] if x.get("tag") == tag), None)
    if r is None:
        print(f"no data yet for {tag}", file=sys.stderr)
        continue
    L = r["legs"]
    if "standard" not in L or "phases" not in L["standard"]:
        print(f"{tag}: no instrumented reference", file=sys.stderr)
        continue
    t0 = ice_cost(L["standard"]["phases"])
    vals = {}
    for key, _, _ in SER:
        d = L.get(key)
        if d and "phases" in d and t0 > 0:
            vals[key] = 100.0 * (ice_cost(d["phases"]) - t0) / t0
    rows.append((lab, vals))

if not rows:
    sys.exit("no instrumented operating-point runs found")

fig, ax = plt.subplots(figsize=(7.2, 3.9), constrained_layout=True)
x = range(len(rows))
w = 0.26
for i, (key, lab, col) in enumerate(SER):
    xs = [xx + (i - 1) * w for xx in x]
    ys = [v.get(key, float("nan")) for _, v in rows]
    ax.bar(xs, ys, width=w, color=col, label=lab, zorder=3)
    for xi, yi in zip(xs, ys):
        if yi == yi:
            ax.annotate(f"{yi:+.0f}", (xi, yi), ha="center", fontsize=7.5,
                        va="top" if yi < 0 else "bottom",
                        xytext=(0, -3 if yi < 0 else 3), textcoords="offset points", zorder=4)
ax.axhline(0, color="k", lw=1.0, zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels([lab for lab, _ in rows], fontsize=8)
ax.set_ylabel("change of the sea-ice cost  [%]")
ax.set_title("Cost of the sea ice relative to the standard scheme,\n"
             "each mesh at a node count where the model still scales", fontsize=9.5)
ax.grid(axis="y", alpha=.3, zorder=0)
ax.legend(fontsize=8, loc="lower left")
fig.savefig(os.path.join(OUT, "fig1_icecost.pdf"))
print("fig1_icecost written")
