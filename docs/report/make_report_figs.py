#!/usr/bin/env python3
"""Figures for the report to S. Danilov on the divergence-form / wide-halo mEVP work.

Deliberately NOT the campaign's internal figures (scripts/m9_figs.py): those carry the study's
own shorthand for the scheme variants, which means nothing outside the project. Here every
series is named the way the report names it.

All numbers come from /work/.../m9/m9_results.json, i.e. from the same measured runs; nothing
is entered by hand except the accuracy table, which comes from the 1-year runs analysed by
scripts/m9_accuracy.py and is quoted in the report text as well.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = "/work/ab0995/a270088/port2/m9/m9_results.json"
OUT = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(J))

C_LAG8, C_LAG4 = "#c0392b", "#e8896b"
C_WIDE, C_WDIV = "#1f6fb4", "#7fb3dd"
GREY = "#555555"


def run(tag):
    for r in D["runs"]:
        if r.get("tag") == tag:
            return r
    return None


def pct(r, leg):
    d = (r.get("legs") or {}).get(leg)
    return None if not d else d.get("pct_vs_ref")


# --------------------------------------------------------------------- Figure 1
# Each mesh at its OWN operating point -- the largest node count at which the model still gains
# from more GPUs, read off the measured strong-scaling curve of the standard scheme. Quoting a
# common node count would have flattered the result badly: CORE2 on 16 nodes gives -25.5% for
# the delayed exchange, and CORE2 on 16 nodes is not a configuration anyone runs (it is already
# 45% SLOWER there than on 2 nodes).
MESHES = [("op_core2_g4",  "CORE2, 127 k nodes\n1 node / 4 GPUs\n(knee at 2 nodes)"),
          ("op_farc_g16",  "fArc, 638 k\n4 nodes / 16 GPUs\n(flat past 4)"),
          ("op_dars_g32",  "DARS, 3.2 M\n8 nodes / 32 GPUs\n(still scaling)"),
          ("rep_ng5_g64",  "NG5, 7.4 M\n16 nodes / 64 GPUs\n(still scaling)")]
SER = [("lagged_k8", "delayed exchange, every 8th sub-cycle", C_LAG8),
       ("lagged_k4", "delayed exchange, every 4th sub-cycle", C_LAG4),
       ("wide_std_fused_k8", "wide halo, standard form (exact)", C_WIDE),
       ("wide_div_k8", "wide halo, divergence form (exact)", C_WDIV)]

rows = [(lab, run(tag)) for tag, lab in MESHES]
missing = [lab.split("\n")[0] for lab, r in rows if r is None]
if missing:
    print(f"figure 1: no data yet for {', '.join(missing)}", file=sys.stderr)

rows = [(lab, r) for lab, r in rows if r is not None]
if rows:
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    x = range(len(rows))
    w = 0.20
    for i, (key, lab, col) in enumerate(SER):
        xs = [xx + (i - 1.5) * w for xx in x]
        ys = [pct(r, key) for _, r in rows]
        ax.bar(xs, [y if y is not None else float("nan") for y in ys], width=w,
               color=col, label=lab, zorder=3)
        for xi, yi in zip(xs, ys):
            if yi is not None:
                ax.annotate(f"{yi:.1f}", (xi, yi), ha="center", va="top", fontsize=6.4,
                            xytext=(0, -3), textcoords="offset points", zorder=4)
    ax.axhline(0, color="k", lw=1.0, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for lab, _ in rows], fontsize=7.6)
    ax.set_ylabel("change of the model time step  [%]")
    ax.set_title("Cost of one model time step relative to the standard scheme,\n"
                 "each mesh at a node count where the model still scales", fontsize=9)
    ax.grid(axis="y", alpha=.3, zorder=0)
    # legend BELOW the axes: in-axes it sat on top of CORE2's two tallest bars and hid their
    # value labels, and CORE2 is the mesh whose numbers a reader checks first.
    fig.legend(fontsize=8, ncol=2, frameon=False, loc="outside lower center")
    fig.savefig(os.path.join(OUT, "fig1_meshes.pdf"))
    print("fig1 written")

# --------------------------------------------------------------------- Figure 2
# Why the gain varies: it tracks how much communication the run is already paying for.
def series(mesh, leg):
    out = {}
    for r in D["runs"]:
        if r.get("mesh") != mesh or r.get("backend") != "GPU" or r.get("mode") != "CLEAN":
            continue
        v = pct(r, leg)
        c = (r.get("legs") or {}).get("classic") or (r.get("legs") or {}).get("standard")
        if v is None or not c:
            continue
        n = r["nodes"]
        # keep the best-resolved value if a node count was measured twice
        if n not in out or abs(v) > abs(out[n][0]):
            out[n] = (v, c.get("s_per_step"))
    return out

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
for mesh, lab, mk in (("core2", "CORE2", "o"), ("farc", "fArc", "s")):
    s = series(mesh, "lag8")
    if not s:
        continue
    ns = sorted(s)
    axs[0].plot(ns, [s[n][0] for n in ns], marker=mk, color=C_LAG8 if mesh == "core2" else C_WIDE,
                label=lab)
    axs[1].plot(ns, [s[n][1] for n in ns], marker=mk, color=C_LAG8 if mesh == "core2" else C_WIDE,
                label=lab)
for a, ylab in ((axs[0], "change of the model time step  [%]"),
                (axs[1], "standard scheme: seconds per time step")):
    a.set_xscale("log", base=2)
    a.set_xticks([1, 2, 4, 8, 16])
    a.set_xticklabels(["1", "2", "4", "8", "16"])
    a.set_xlabel("GPU nodes (4 A100 each)")
    a.set_ylabel(ylab, fontsize=8.5)
    a.grid(alpha=.3)
    a.legend(fontsize=8)
axs[0].axhline(0, color="k", lw=.9)
axs[0].set_title("gain from delaying the exchange\n(every 8th sub-cycle)", fontsize=9)
axs[1].set_title("cost of the run it is applied to\n(the same runs, standard scheme)", fontsize=9)
fig.savefig(os.path.join(OUT, "fig2_scaling.pdf"))
print("fig2 written")

# --------------------------------------------------------------------- Figure 3
# Accuracy of the approximation, 1-year CORE2 runs (CPU, 256 ranks, deterministic).
# Correlation against the same run with exchange every sub-cycle; the horizontal lines are the
# difference between two independent, both-correct implementations of mEVP (C and Fortran).
K = [2, 4, 8]
ACC = {"u": [0.970111, 0.963328, 0.961359], "v": [0.952946, 0.950003, 0.944857]}
FLOOR = {"u": 0.95438, "v": 0.93908}
fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
for comp, lab, col, mk in (("u", "ice velocity, u", "#1f6fb4", "o"),
                           ("v", "ice velocity, v", "#c0392b", "s")):
    ax.plot(K, ACC[comp], marker=mk, color=col, label=lab)
    ax.axhline(FLOOR[comp], color=col, ls="--", lw=1.0, alpha=.8)
    ax.annotate(f"C vs Fortran, {lab.split(', ')[1]}: {FLOOR[comp]:.3f}",
                (8.15, FLOOR[comp]), fontsize=6.8, color=col, va="center")
ax.set_xscale("log", base=2)
ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K])
ax.set_xlim(1.8, 15)
ax.set_xlabel("exchange every $K$-th sub-cycle")
ax.set_ylabel("pattern correlation after one year")
ax.set_title("Accuracy cost of the delayed exchange\n"
             "1 year, CORE2, against the same run exchanging every sub-cycle", fontsize=9)
ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="lower left")
fig.savefig(os.path.join(OUT, "fig3_accuracy.pdf"))
print("fig3 written")
