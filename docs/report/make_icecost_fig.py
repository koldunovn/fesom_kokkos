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

# Which fleet. op6_* ran under the deterministic cold-start hole fill (FESOM_IC_EXTRAP=det);
# p6_* is the original, where the fill was partition-dependent so the four points below --
# four different partitions -- did not start from the same ocean. The ratios plotted here are
# within-run and were protected either way; the cross-mesh comparison was not.
# See docs/plans/20260815-m9-det-rerun.md.
PFX = (sys.argv[1] if len(sys.argv) > 1 else "op6").rstrip("_")
POINTS = [(f"{PFX}_core2_phst", "CORE2, 127 k nodes\n1 node / 4 GPUs"),
          (f"{PFX}_farc_phst",  "fArc, 638 k\n4 nodes / 16 GPUs"),
          (f"{PFX}_dars_phst",  "DARS, 3.2 M\n8 nodes / 32 GPUs"),
          (f"{PFX}_ng5_phst",   "NG5, 7.4 M\n16 nodes / 64 GPUs")]
# Session 5: the same four schemes, but each wide-halo scheme now appears in BOTH writings --
# ring kernels of their own (as measured through session 4) and the ring folded into the loops
# that already exist (Danilov's structure). The pairs are the figure's whole point: the two bars
# of a pair compute the same ring, remove the same messages and put identical byte counts on the
# wire, so the distance between them is the port's kernel structure and nothing else.
SER = [("wide_k8",          "standard form, ring kernels",   "#1f6fb4"),
       ("wide_k8_lean",     "standard form, widened loops",  "#8fc4e8"),
       ("widediv_k8",       "divergence form, ring kernels", "#8a5a1f"),
       ("widediv_k8_lean",  "divergence form, widened loops", "#e0b072")]


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

fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
x = range(len(rows))
w = 0.20
for i, (key, lab, col) in enumerate(SER):
    xs = [xx + (i - 1.5) * w for xx in x]
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
ax.set_title("Cost of the sea ice relative to the standard scheme, each mesh at a node count\n"
             "where the model still scales. Within a pair only the kernel structure differs.",
             fontsize=9.5)
ax.grid(axis="y", alpha=.3, zorder=0)
# the deepest bar reaches about -58 %, and its value label sits below it, so leave room
# rather than letting the label collide with the legend (which is what the first draft did)
lo = min([v for _, vv in rows for v in vv.values()] + [0.0])
ax.set_ylim(lo * 1.18 - 4, max([v for _, vv in rows for v in vv.values()] + [0.0]) + 6)
fig.legend(fontsize=8, ncol=4, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, -0.045))
fig.savefig(os.path.join(OUT, "fig1_icecost.pdf"), bbox_inches="tight")
print("fig1_icecost written")
