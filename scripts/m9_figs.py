#!/usr/bin/env python3
"""M9 figures from scripts/m9_collect.py's JSON.

Conventions follow scripts/m7_scaling_figs.py (the standing rule: read it before writing any
new plot) — explicit integer ticks on the rank axis, decimal (not 1e-x) labels on a log y-axis,
SYPD = DT_PROD / (365 * s_per_step) at the production dt (core2 1800).

Panels:
  F1  icedyn phase time vs rank count, one line per cell, CPU and GPU side by side
  F2  model-step change vs rank count, per cell (the number a user actually feels)
  F3  K sweep: icedyn reduction and the ACCURACY COST on a twin axis — a speed curve for an
      approximation is not publishable without the error curve next to it
"""
import argparse, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--json", default="/work/ab0995/a270088/port2/m9/m9_results.json")
ap.add_argument("--l3", default="/work/ab0995/a270088/port2/m9",
                help="dir holding bitid_l3_* gate outputs, for the accuracy curve")
ap.add_argument("--outdir", default="/work/ab0995/a270088/port2/m9/figs")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
D = json.load(open(args.json))

DT_PROD = {"core2": 1800.0, "ng5_w3d": 240.0}
COL = {"classic": "#333333", "div": "#1f77b4", "div_masked": "#1f77b4", "div_unmasked": "#7fbfe0",
       "lag2": "#2ca02c", "lag4": "#ff7f0e", "lag8": "#d62728", "lag12": "#9467bd",
       "lag20": "#8c564b", "lag30": "#e377c2", "lag60": "#17becf", "lag120": "#bcbd22",
       "div_lag4": "#e6550d", "div_noeps": "#9ecae1", "classic_noeps": "#999999"}

def decimal_log_yaxis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

def rank_axis(ax, counts):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(counts)))
    ax.set_xticklabels([str(c) for c in sorted(set(counts))])
    ax.set_xlabel("MPI ranks")

def series(mode, backend):
    """(cell -> {ntasks: value}) for icedyn ms and s/step, restricted to one backend/mode."""
    ice, step, ranks = {}, {}, set()
    for r in D["runs"]:
        if r.get("mode", "").upper() != mode:            continue
        if r.get("mesh") != "core2":                     continue
        if r.get("backend") != backend:                  continue
        n = r["ntasks"]; ranks.add(n)
        for cell, d in r["legs"].items():
            if d.get("icedyn_busy_ms") is not None:
                ice.setdefault(cell, {})[n] = d["icedyn_busy_ms"]
            if d.get("s_per_step") is not None:
                step.setdefault(cell, {})[n] = d["s_per_step"]
    return ice, step, sorted(ranks)

# ---------------------------------------------------------------- F1: icedyn vs ranks
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
for ax, backend in zip(axes, ["CPU", "GPU"]):
    ice, _, ranks = series("INSTRUMENTED", backend)
    for cell, pts in sorted(ice.items()):
        xs = sorted(pts); ys = [pts[x] for x in xs]
        if len(xs) < 1: continue
        ax.plot(xs, ys, marker="o", ms=4, color=COL.get(cell, None), label=cell)
    if ranks:
        rank_axis(ax, ranks); decimal_log_yaxis(ax)
    ax.set_title(f"{backend} — CORE2")
    ax.set_ylabel("ice-dynamics phase  [ms/step]")
    ax.grid(alpha=.25, which="both")
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("M9 — mEVP ice-dynamics phase per cell (PHASESTATS, instrumented legs)", fontsize=10)
fig.savefig(os.path.join(args.outdir, "f1_icedyn_vs_ranks.png"), dpi=140)
print("F1 written")

# ---------------------------------------------------------------- F2: model step delta
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
for ax, backend in zip(axes, ["CPU", "GPU"]):
    _, step, ranks = series("CLEAN", backend)
    if not step:                                  # fall back: instrumented legs still have s/step
        _, step, ranks = series("INSTRUMENTED", backend)
    base = step.get("classic", {})
    for cell, pts in sorted(step.items()):
        if cell == "classic": continue
        xs = [x for x in sorted(pts) if x in base]
        if not xs: continue
        ys = [100.0 * (pts[x] - base[x]) / base[x] for x in xs]
        ax.plot(xs, ys, marker="o", ms=4, color=COL.get(cell, None), label=cell)
    ax.axhline(0, color="k", lw=.8)
    if ranks: rank_axis(ax, ranks)
    ax.set_title(f"{backend} — CORE2")
    ax.set_ylabel("model step change vs classic  [%]")
    ax.grid(alpha=.25)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("M9 — whole-model effect (negative = faster). Reported beside icedyn, never instead of it.",
             fontsize=10)
fig.savefig(os.path.join(args.outdir, "f2_step_delta.png"), dpi=140)
print("F2 written")

# ---------------------------------------------------------------- F3: K sweep + accuracy
def accuracy_by_K():
    """max|Δ| on the ice fields from the L3 gate outputs (bitid_l3_lag<K>)."""
    out = {}
    for d in sorted(os.listdir(args.l3)):
        m = re.match(r"^bitid_l3_lag(\d+)$", d)
        if not m: continue
        K = int(m.group(1))
        # the gate prints the magnitudes into the slurm .out; find it via the tag
        for o in os.listdir(args.l3):
            if not o.startswith("bitid.") or not o.endswith(".out"): continue
            p = os.path.join(args.l3, o)
            try: txt = open(p, "rb").read().decode("utf-8", "replace")
            except OSError: continue
            if f"TAG=l3_lag{K}" not in txt: continue
            vals = {}
            for f in ("a_ice", "m_ice", "uice", "vice"):
                mm = re.search(rf"^\s+{f}\s+max\|Δ\|=([0-9.eE+-]+)", txt, re.M)
                if mm: vals[f] = float(mm.group(1))
            if vals: out[K] = vals
            break
    return out

ice_g, step_g, _ = series("INSTRUMENTED", "GPU")
acc = accuracy_by_K()
if acc:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    Ks, red = [], []
    base = None
    for cell, pts in ice_g.items():
        if cell == "classic": base = pts
    for cell, pts in sorted(ice_g.items()):
        m = re.match(r"^lag(\d+)$", cell)
        if not m or not base: continue
        n = max(pts)                      # largest rank count available for that cell
        if n not in base: continue
        Ks.append(int(m.group(1)))
        red.append(100.0 * (pts[n] - base[n]) / base[n])
    if Ks:
        o = sorted(range(len(Ks)), key=lambda i: Ks[i])
        ax.plot([Ks[i] for i in o], [red[i] for i in o], marker="o", color="#d62728",
                label="icedyn change [%]")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(acc)); ax.set_xticklabels([str(k) for k in sorted(acc)])
    ax.set_xlabel("K  (exchange every K-th subcycle)")
    ax.set_ylabel("ice-dynamics phase change  [%]", color="#d62728")
    ax2 = ax.twinx()
    for f, mk in (("a_ice", "s"), ("uice", "^")):
        ks = [k for k in sorted(acc) if f in acc[k]]
        if ks: ax2.plot(ks, [acc[k][f] for k in ks], marker=mk, ms=4, ls="--", label=f"max|Δ| {f}")
    ax2.set_yscale("log"); ax2.set_ylabel("ice-field difference vs classic  (300 steps)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax.grid(alpha=.25)
    ax.set_title("M9 cell ⑤ — the lagged halo buys speed and costs accuracy.\nBoth axes, always.",
                 fontsize=10)
    fig.savefig(os.path.join(args.outdir, "f3_ksweep_speed_vs_accuracy.png"), dpi=140)
    print("F3 written")
else:
    print("F3 skipped — no L3 accuracy data yet", file=sys.stderr)
print(f"figures -> {args.outdir}")
