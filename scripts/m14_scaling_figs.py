#!/usr/bin/env python3
"""
M14 strong-scaling figures: the state BEFORE these campaigns vs the best combination AFTER.

Two figures, each with one panel per platform (Levante CPU / Levante A100 / dolpung GH200):
  m14_scaling_sstep.png  — wall time per step (log y, decimal ticks)
  m14_scaling_sypd.png   — SYPD at PRODUCTION dt (linear)

Conventions are taken from scripts/m7_scaling_figs.py (standing rule: read it before plotting):
log2 node axis with explicit tick labels, plain-decimal log y, paper mesh colours via
paper_jax/scripts/common, before = open marker + dotted, after = filled + solid.

SYPD = DT_PROD / (365 · s_per_step).

⚠️ NO CG dt-CORRECTION APPLIED (user decision, 2026-08-16: "for now we can live without a
factor"). dars is measured at dt120 and NG5 at dt180 but both are reported at dt240, and s/step is
NOT dt-independent for the implicit solve — a larger dt needs more CG iterations. m7_scaling_figs.py
carries measured factors (dars ×1.0222, NG5 ×1.0110) for exactly this. Omitting them OVERSTATES
dars and NG5 SYPD by ~2 % and ~1 %. Stated on the figure so it cannot be read as exact.
⚠️ The SE arm's cost is genuinely dt-dependent (its barotropic subcycle count M is set by the
CFL at the baroclinic dt), so an SE point rescaled to production dt is an estimate, not a
measurement — flagged in the caption.
"""
import glob
import re
import sys
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
try:
    import common
    MESH_COLOR, MESH_LABEL = common.MESH_COLOR, common.MESH_LABEL
    MESH_ORDER = common.MESH_ORDER
    common.set_style()
except Exception:                                   # keep the figure buildable off-paper
    MESH_COLOR = {"core2": "#4C72B0", "farc": "#DD8452", "dars": "#55A868", "ng5": "#C44E52"}
    MESH_LABEL = {"core2": "CORE2", "farc": "fArc", "dars": "DARS", "ng5": "NG5"}
    MESH_ORDER = ["core2", "farc", "dars", "ng5"]

M14 = "/work/ab0995/a270088/port2/m14"
DT_PROD = {"core2": 1800.0, "farc": 900.0, "dars": 240.0, "ng5": 240.0}
SYPD = lambda mesh, s: DT_PROD[mesh] / (365.0 * s)


def harvest():
    """-> {platform: {mesh: {units: {'base': s, 'best': s, 'lever': str}}}}
    units = nodes for CPU, GPUs for the GPU platforms."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))

    def put(plat, mesh, units, arm, val, lever=""):
        d = out[plat][mesh].setdefault(units, {})
        if arm not in d or val < d[arm]:          # keep the FASTEST measurement of each arm
            d[arm] = val
            if arm == "best":
                d["lever"] = lever

    # --- Levante CPU -------------------------------------------------------
    for f in glob.glob(f"{M14}/ladder.*.out"):
        txt = open(f, errors="replace").read()
        m = re.search(r"M14 CPU ladder — (\w+) ranks=(\d+)", txt)
        if not m:
            continue
        mesh, ranks = m.group(1), int(m.group(2))
        nodes = max(1, round(ranks / 128))
        bk = re.search(r"best knobs: (.*)", txt)
        bm = re.search(r"best mesh : \S*/(\w+_v1)/", txt)
        lever = (("part+" if bm else "") + (bk.group(1) if bk else "")).replace("FESOM_", "") \
                .replace("SSH_SOLVER=", "").replace("SSH_MODE=", "").replace("SPEED_", "").strip()
        if "EVPWIDE" in lever:                    # measured LOSS on CPU — never in the best arm
            lever = None
        for arm in ("base", "best"):
            mm = re.search(rf"\n\s+{arm}\s+legs.*?min=([\d.]+)", txt)
            if not mm:
                continue
            if arm == "best" and lever is None:
                continue
            put("cpu", mesh, nodes, arm, float(mm.group(1)), lever or "")
        mm = re.search(r"^CSV (\w+),(\d+),cfg=(\S*?),.*?base=([\d.]+)", txt, re.M)
        if mm:
            put("cpu", mesh, nodes, "base", float(mm.group(4)))

    # --- Levante A100 ------------------------------------------------------
    for f in glob.glob(f"{M14}/gladder.*.out"):
        txt = open(f, errors="replace").read()
        m = re.search(r"M14 CPU ladder — (\w+) ranks=(\d+)", txt)   # derived job keeps the header
        if not m:
            continue
        mesh, ngpu = m.group(1), int(m.group(2))
        for arm in ("base", "best"):
            mm = re.search(rf"\n\s+{arm}\s+legs.*?min=([\d.]+)", txt)
            if mm:
                put("a100", mesh, ngpu, arm, float(mm.group(1)))

    # --- dolpung GH200 -----------------------------------------------------
    for f in glob.glob(f"{M14}/dlad.*.out"):
        txt = open(f, errors="replace").read()
        m = re.search(r"M14 dolpung ladder — (\w+) gpus=(\d+)", txt)
        if not m:
            continue
        mesh, ngpu = m.group(1), int(m.group(2))
        for arm in ("base", "best"):
            mm = re.search(rf"\n\s+{arm}\s+min=([\d.]+)", txt)
            if mm and float(mm.group(1)) > 0:
                put("gh200", mesh, ngpu, arm, float(mm.group(1)))
    return out


PLATS = [("cpu",   "Levante CPU", "nodes  (128-core)"),
         ("a100",  "Levante A100", "number of GPUs"),
         ("gh200", "dolpung GH200", "number of GPUs")]


def decimal_log_yaxis(ax, lo, hi):
    ticks = [t for t in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3)
             if lo * 0.95 <= t <= hi * 1.05]
    if ticks:
        ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(lo * 0.85, hi * 1.15)


def log2_axis(ax, counts, label):
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    # thin the labels when the rungs crowd (128/160/192/256 overlapped otherwise)
    show = counts if len(counts) <= 7 else [c for i, c in enumerate(counts)
                                            if i % 2 == 0 or c == counts[-1]]
    ax.set_xticks(show)
    ax.set_xticklabels([str(c) for c in show], fontsize=7.5)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(label)


SMALL = ("core2", "farc")          # SYPD O(10-160)
LARGE = ("dars", "ng5")            # SYPD O(1-10) — a shared linear axis hides these entirely


def render(data, metric, fname, title):
    if metric == "sypd":
        panels = [("cpu", "Levante CPU — CORE2 / fArc", "nodes  (128-core)", SMALL),
                  ("cpu", "Levante CPU — DARS / NG5", "nodes  (128-core)", LARGE),
                  ("a100", "Levante A100", "number of GPUs", None),
                  ("gh200", "dolpung GH200", "number of GPUs", None)]
        fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.3))
    else:
        panels = [(p, n, x, None) for p, n, x in PLATS]
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    lo, hi = 1e9, 0
    for ax, (plat, pname, xlab, only) in zip(axes, panels):
        allx = []
        for mesh in [m for m in MESH_ORDER if m in data.get(plat, {})
                     and (only is None or m in only)]:
            col = MESH_COLOR.get(mesh, "k")
            pts = data[plat][mesh]
            for arm, ls, mfc in (("base", ":", "none"), ("best", "-", col)):
                xs = sorted(u for u in pts if arm in pts[u])
                if not xs:
                    continue
                ys = [pts[u][arm] if metric == "sstep" else SYPD(mesh, pts[u][arm]) for u in xs]
                allx += xs
                if metric == "sstep":
                    lo, hi = min(lo, min(ys)), max(hi, max(ys))
                ax.plot(xs, ys, marker="o", ms=4.5, ls=ls, color=col, markerfacecolor=mfc,
                        lw=1.6 if arm == "best" else 1.2,
                        label=None)
        if allx:
            log2_axis(ax, allx, xlab)
        ax.set_title(pname, fontsize=10)
        ax.grid(alpha=0.25, which="major")
        if metric == "sypd":
            ax.set_ylabel("SYPD  (simulated years / wall day)" if ax is axes[0] else "")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("wall time per step  (s)" if plat == "cpu" else "")
    if metric == "sstep":
        for ax in axes:
            decimal_log_yaxis(ax, lo, hi)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MESH_COLOR.get(m, "k"), marker="o", ms=4.5, ls="-",
                      label=MESH_LABEL.get(m, m)) for m in MESH_ORDER if any(m in data.get(p, {}) for p, _, _ in PLATS)]
    handles += [Line2D([], [], color="0.35", ls=":", marker="o", ms=4.5, markerfacecolor="none",
                       label="before campaigns (baseline)"),
                Line2D([], [], color="0.35", ls="-", marker="o", ms=4.5, label="best combination")]
    axes[0].legend(handles=handles, fontsize=7.5, frameon=False, loc="best")
    fig.suptitle(title, fontsize=11, y=0.99)
    note = ("SYPD = dt_prod/(365·s_step), dt_prod = CORE2 1800 · fArc 900 · dars 240 · NG5 240.  "
            "dars measured at dt120 and NG5 at dt180 — NO CG dt-correction applied (would be "
            "×1.022 / ×1.011), so dars/NG5 SYPD is ~2 %/1 % optimistic.  SE points rescaled to "
            "production dt are estimates: the barotropic subcycle count depends on dt."
            if metric == "sypd" else
            "Best combination = SSH solver (oati) or split-explicit (SE) + the M11 certified "
            "partition where one exists; the sea-ice wide halo is EXCLUDED — measured a loss on "
            "CPU at every K. Each pair measured base and best in ONE allocation, ABBA, min over legs.")
    fig.text(0.5, 0.005, note, ha="center", fontsize=6.6, color="0.35", wrap=True)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])
    fig.savefig(fname, dpi=140)
    print("wrote", fname)


if __name__ == "__main__":
    d = harvest()
    for plat, pname, _ in PLATS:
        for mesh in d.get(plat, {}):
            n = len(d[plat][mesh])
            nb = sum(1 for u in d[plat][mesh] if "best" in d[plat][mesh][u])
            print(f"  {pname:14s} {mesh:6s}: {n} points, {nb} with a best arm")
    render(d, "sstep", f"{M14}/m14_scaling_sstep.png", "M14 — strong scaling: wall time per step")
    render(d, "sypd",  f"{M14}/m14_scaling_sypd.png",  "M14 — strong scaling: SYPD at production dt")
