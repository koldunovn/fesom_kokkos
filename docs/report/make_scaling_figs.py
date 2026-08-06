#!/usr/bin/env python3
"""Report Figures 2 and 3: strong scaling of the standard scheme and the two lean wide-halo
forms, on both backends, for the whole model step and for the sea ice separately.

Two figures because the two quantities answer different questions and have different shapes:

  fig2_scaling_step.pdf   seconds per model time step
  fig3_scaling_ice.pdf    seconds per step spent on the sea ice (ice + ice dynamics + ice
                          advection, computation together with time spent waiting for
                          communication) -- the measure appropriate to a sea-ice scheme, since
                          the ice is a small share of the step and the step divides an ice-only
                          change by roughly ten

The delayed exchange is not plotted. It is not a usable scheme (it leaves the domain
decomposition in the ice field) and it has been removed from this study.

Conventions follow scripts/m7_scaling_figs.py, which is the house style for this project:
log-base-2 node axis with PLAIN node-count tick labels, log y with plain decimal labels.

🔴 The two panels come from DIFFERENT runs on purpose. PHASESTATS perturbs the model, so the
step is read from the clean legs and the ice cost from the instrumented ones -- the same rule
the A/B job itself states. Mixing them would put an instrument's overhead into a SYPD number.

usage: make_scaling_figs.py [--root /work/ab0995/a270088/port2/m9]
"""
import argparse, glob, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/ab0995/a270088/port2/m9")
ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()

HDR = re.compile(r"^=== M9 (?:CPU|GPU) A/B\s+TAG=(\S+)\s+mesh=(\S+)\s+nodes=(\d+)\s+ntasks=(\d+)"
                 r"\s+dt=(\d+)\s+steps=(\d+)")
LEG = re.compile(r"^LEG\s+(\S+)\s+min\s+([0-9.]+)\s+s/step")
PHASE = re.compile(r"\[phasestats\]\s+(\w+)\s+\|\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|"
                   r"\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|\s+([0-9.]+)")
ICE = ("ice", "icedyn", "iceadv")
TAG = re.compile(r"^sc_(gpu|cpu)_(\w+?)_(\d+)n(_phst)?$")

MESHES = [("core2", "CORE2, 127 k nodes"), ("farc", "fArc, 638 k"),
          ("dars", "DARS, 3.2 M"), ("ng5", "NG5, 7.4 M")]
SER = [("standard", "standard (exchange every sub-cycle)", "#333333", "o", "-"),
       ("lean2",    "wide halo, standard form, widened loops",  "#1f6fb4", "s", "-"),
       ("lean4",    "wide halo, divergence form, widened loops", "#c46210", "^", "-")]


def ice_cost(legdir):
    """ms/step over the three ice phases, busy + wait, from rep 1 (the paired rule)."""
    tot, seen = 0.0, False
    try:
        with open(os.path.join(legdir, "run.1.log"), errors="replace") as f:
            for ln in f:
                m = PHASE.search(ln)
                if m and m.group(1) in ICE:
                    seen = True
                    tot += float(m.group(3)) + float(m.group(6))
    except OSError:
        return None
    return tot if seen else None


# (backend, mesh, nodes) -> {"step": {leg: s}, "ice": {leg: s}}
data = {}
for out in sorted(glob.glob(os.path.join(args.root, "ab*.out"))):
    tag = None
    legs = {}
    with open(out, errors="replace") as f:
        for ln in f:
            m = HDR.match(ln)
            if m:
                tag = m.group(1)
            m = LEG.match(ln)
            if m:
                legs[m.group(1)] = float(m.group(2))
    if not tag:
        continue
    t = TAG.match(tag)
    if not t:
        continue
    backend, mesh, nodes, phst = t.group(1), t.group(2), int(t.group(3)), bool(t.group(4))
    d = data.setdefault((backend, mesh, nodes), {"step": {}, "ice": {}})
    if phst:
        for lg in legs:
            c = ice_cost(os.path.join(args.root, tag, lg))
            if c is not None:
                d["ice"][lg] = c / 1000.0          # ms -> s, so both panels share units
    else:
        d["step"].update(legs)


def node_axis(ax, counts):
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def decimal_log_yaxis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def render(which, fname, title, ylabel):
    fig, axes = plt.subplots(2, len(MESHES), figsize=(12.6, 6.0), constrained_layout=True,
                             squeeze=False)
    any_point = False
    for col, (mesh, mlab) in enumerate(MESHES):
        for row, backend in enumerate(("gpu", "cpu")):
            ax = axes[row][col]
            counts = sorted(n for (b, m, n) in data if b == backend and m == mesh)
            drew = False
            for key, lab, col_, mk, ls in SER:
                xs, ys = [], []
                for n in counts:
                    v = data[(backend, mesh, n)][which].get(key)
                    if v:
                        xs.append(n); ys.append(v)
                if xs:
                    ax.plot(xs, ys, ls, color=col_, marker=mk, ms=4.2, lw=1.4,
                            label=lab if (row == 0 and col == 0) else None, zorder=3)
                    drew = True; any_point = True
            if drew:
                node_axis(ax, counts)
                decimal_log_yaxis(ax)
            else:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8,
                        color="#999999", transform=ax.transAxes)
            ax.grid(alpha=.3, which="both", zorder=0)
            ax.tick_params(labelsize=7.5)
            if row == 0:
                ax.set_title(mlab, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{'GPU' if backend=='gpu' else 'CPU'}\n{ylabel}", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("nodes  (4×A100 / 128 cores)", fontsize=8)
    if not any_point:
        sys.exit(f"no data for {which}")
    fig.suptitle(title, fontsize=10.5)
    fig.legend(loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    fig.savefig(os.path.join(args.outdir, fname), bbox_inches="tight")
    print(f"{fname} written")


render("step", "fig2_scaling_step.pdf",
       "Strong scaling of the whole model time step",
       "s / model step")
render("ice", "fig3_scaling_ice.pdf",
       "Strong scaling of the sea-ice cost (ice + ice dynamics + ice advection, busy and wait)",
       "s / step on the ice")
