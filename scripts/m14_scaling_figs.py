#!/usr/bin/env python3
"""M14 JUPITER campaign: harvest the jlad fleet into a tidy CSV + render figures.

Harvests every /e/scratch/e-sta-destine/koldunov1/port2/m14/jlad.*.out (the
job_m14_jupiter_ladder output): mesh, ranks, cfg, the LEVER (recovered from the
job header's `best knobs:` / `best mesh :` lines — JCSV alone does not carry it),
the admitted mins from the JCSV line, rejected-leg count, and the per-best-leg
fallback count (grep -a semantics: NUL bytes have eaten legs before, M10 L.72).
A best arm with fallbacks is a variant/baseline MIXTURE and is flagged void.

Figures follow the M7 conventions (scripts/m7_scaling_figs.py — standing rule):
shared horizontal frameless legend BELOW the panels, log2 x-axis with plain
decimal labels, per-mesh colors, dotted 1/N ideal off the base anchor, caveats
as a small footnote line. paper_jax/common.py is not on JUPITER, so the mesh
palette/labels are vendored here (keep in sync if the paper's change).

  fig_m14_levers.png   per-mesh gain%% vs ranks, one curve per lever
  fig_m14_ng5.png      NG5 absolute s/step: base vs levers vs composition (the knee story)

usage: m14_scaling_figs.py [--outdir .../port2/m14/figs] [--csv-only]
Missing points are simply absent (re-run as jobs land). SYPD at production dt
(core2 1800, farc 900, dars/NG5 240) from dt120/dt180 runs, with the M7-measured
CG dt-correction (dars x1.0222, NG5 x1.0110) — footnoted.
"""
import argparse
import glob
import os
import re

M14 = "/e/scratch/e-sta-destine/koldunov1/port2/m14"
DT_PROD = {"core2": 1800.0, "farc": 900.0, "dars": 240.0, "ng5": 240.0}
DT_CORR = {"core2": 1.0, "farc": 1.0, "dars": 1.0222, "ng5": 1.0110}

MESH_ORDER = ["core2", "farc", "dars", "ng5"]
MESH_COLOR = {"core2": "tab:blue", "farc": "tab:orange", "dars": "tab:green", "ng5": "tab:red"}
MESH_LABEL = {"core2": "CORE2", "farc": "fArc", "dars": "DARS", "ng5": "NG5"}
LEVER_STYLE = {  # (linestyle, marker)
    "base": ("-", "o"), "oati": ("--", "^"), "se": ("-.", "s"),
    "part": (":", "D"), "part+oati": ("--", "v"), "evpwide": (":", "x"),
}


def classify_lever(knobs, mesh_lever):
    if "FESOM_SSH_MODE=se" in knobs:
        return "se"
    if "EVPWIDE" in knobs:
        return "evpwide"
    oati = "FESOM_SSH_SOLVER=oati" in knobs
    if mesh_lever and oati:
        return "part+oati"
    if mesh_lever:
        return "part"
    if oati:
        return "oati"
    return "base" if not knobs else "other:" + knobs


def harvest():
    rows = []
    for f in sorted(glob.glob(f"{M14}/jlad.*.out")):
        txt = open(f, errors="replace").read()
        m = re.search(r"=== M14 JUPITER ladder — (\w+) ranks=(\d+) nodes=(\d+) dt=[\d.]+ steps=(\d+)", txt)
        if not m:
            continue
        mesh, ranks, nodes, steps = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        job = re.search(r"jlad\.(\d+)\.out$", f).group(1)
        md5 = (re.search(r"^([0-9a-f]{32}) ", txt, re.M) or [None, ""])[1]
        knobs = (re.search(r"^best knobs: (.+)$", txt, re.M) or [None, ""])[1]
        mesh_lever = bool(re.search(r"^best mesh :", txt, re.M))
        wsplit = (re.search(r"^wsplit\s+: (wsplit\d)", txt, re.M) or [None, "?"])[1]
        jc = re.search(r"^JCSV \w+,\d+,cfg=([^,]+),(.*)$", txt, re.M)
        if not jc:
            continue  # REFUSE / node-fail / no admitted legs at all
        cfg, tail = jc.group(1), jc.group(2)
        mins = dict(re.findall(r"(\w+)=([\d.]+)", tail))
        base = float(mins["base"]) if "base" in mins else None
        best = float(mins["best"]) if "best" in mins else None
        if best is not None and best < 1e-9:
            best = None  # pre-fix awk artifact on baseline-only jobs
        rejected = txt.count("REJECTED")
        # fallback count over the best legs' stderr (mixture rule, M10)
        fb = 0
        for err in glob.glob(f"{M14}/jlad_{mesh}_{ranks}_{job}/[0-9]_best/run.err"):
            fb += open(err, errors="replace").read().count("FALLBACK")
        lever = classify_lever(knobs, mesh_lever)
        gain = (best - base) / base * 100.0 if (base and best) else None
        rows.append(dict(
            job=int(job), mesh=mesh, ranks=ranks, nodes=nodes, steps=steps,
            lever=lever, cfg=cfg, wsplit=wsplit, base=base, best=best,
            gain_pct=None if gain is None else round(gain, 2),
            rejected=rejected, fallbacks_best=fb,
            void_mixture=(fb > 0 and best is not None), md5=md5[:8],
            mtime=int(os.path.getmtime(f)),
        ))
    return rows


def write_csv(rows, path):
    cols = ["job", "mesh", "ranks", "nodes", "steps", "lever", "cfg", "wsplit",
            "base", "best", "gain_pct", "rejected", "fallbacks_best",
            "void_mixture", "md5", "mtime"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["mesh"], r["lever"], r["ranks"], r["job"])):
            fh.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    print(f"harvested {len(rows)} rows -> {path}")


def ranks_axis(ax, counts):
    import matplotlib
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts], fontsize=7)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("GPUs (= MPI ranks, 4/node GH200)")


def admitted(rows, mesh, lever):
    """clean pair points: both mins, no mixture; keep the BEST (min base) point
    per rank when several allocations measured the same (mesh, lever, rank)."""
    pts = {}
    for r in rows:
        if r["mesh"] != mesh or r["lever"] != lever or r["void_mixture"]:
            continue
        if r["base"] is None or (lever != "base" and r["best"] is None):
            continue
        k = r["ranks"]
        if k not in pts or r["base"] < pts[k]["base"]:
            pts[k] = r
    return [pts[k] for k in sorted(pts)]


def fig_levers(rows, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.7), sharey=True)
    for ax, mesh in zip(axes, MESH_ORDER):
        counts = []
        for lever, (ls, mk) in LEVER_STYLE.items():
            if lever in ("base",):
                continue
            pts = [r for r in admitted(rows, mesh, lever) if r["gain_pct"] is not None]
            if not pts:
                continue
            ax.plot([r["ranks"] for r in pts], [r["gain_pct"] for r in pts],
                    ls=ls, marker=mk, ms=4, color=MESH_COLOR[mesh], alpha=0.9)
            counts += [r["ranks"] for r in pts]
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.5)
        if counts:
            ranks_axis(ax, counts)
        ax.set_title(MESH_LABEL[mesh], fontsize=9)
    axes[0].set_ylabel("best vs base  [%]  (negative = faster)")
    handles = [Line2D([], [], color="0.35", ls=ls, marker=mk, ms=4, label=lv)
               for lv, (ls, mk) in LEVER_STYLE.items() if lv != "base"]
    fig.legend(handles=handles, ncol=len(handles), fontsize=7.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.text(0.995, 0.005,
             "within-allocation ABBA min-over-admitted-legs; mixture legs (oati fallbacks) excluded; "
             "farc oati void at every rung (M10 stall)",
             ha="right", fontsize=5, alpha=0.6)
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print("wrote", fname)


def fig_ng5(rows, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    counts = []
    handles = []
    # (a) absolute s/step: base curve + each lever's BEST arm
    base_pts = admitted(rows, "ng5", "base")
    for lever, (ls, mk) in LEVER_STYLE.items():
        pts = admitted(rows, "ng5", lever)
        if lever == "base":
            ys = [r["base"] for r in pts]
        else:
            ys = [r["best"] for r in pts]
        if not pts:
            continue
        xs = [r["ranks"] for r in pts]
        axA.plot(xs, ys, ls=ls, marker=mk, ms=4,
                 color="k" if lever == "base" else MESH_COLOR["ng5"],
                 alpha=1.0 if lever in ("base", "se") else 0.55)
        handles.append(Line2D([], [], color="k" if lever == "base" else MESH_COLOR["ng5"],
                              ls=ls, marker=mk, ms=4, label=lever))
        counts += xs
        axB.plot(xs, [DT_PROD["ng5"] / (365.0 * y * DT_CORR["ng5"]) for y in ys],
                 ls=ls, marker=mk, ms=4,
                 color="k" if lever == "base" else MESH_COLOR["ng5"],
                 alpha=1.0 if lever in ("base", "se") else 0.55)
    if base_pts:
        g0 = base_pts[0]
        gg = np.array(sorted(counts))
        axA.plot(gg, g0["base"] * g0["ranks"] / gg, ls=":", lw=0.8, color="0.5", alpha=0.5)
    axA.set_yscale("log")
    ranks_axis(axA, counts)
    axA.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axA.set_ylabel("time per step  [s]")
    axA.set_title("(a) NG5 strong scaling by lever  (dotted = 1/N)")
    ranks_axis(axB, counts)
    axB.set_ylabel("SYPD  (simulated yr / wall day)")
    axB.set_ylim(0, None)
    axB.set_title("(b) NG5 throughput at production dt")
    fig.legend(handles=handles, ncol=len(handles), fontsize=7.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.text(0.995, 0.005,
             "NG5 measured at dt180; SYPD at dt240 with measured CG dt-correction x1.011 (M7); "
             "base curves cross allocations — shape from the calm serial pass where present",
             ha="right", fontsize=5, alpha=0.6)
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print("wrote", fname)


def read_csv(path):
    import csv as _csv
    rows = []
    for r in _csv.DictReader(open(path)):
        for k in ("job", "ranks", "nodes", "steps", "rejected", "fallbacks_best", "mtime"):
            r[k] = int(r[k]) if r[k] else None
        for k in ("base", "best", "gain_pct"):
            r[k] = float(r[k]) if r[k] else None
        r["void_mixture"] = r["void_mixture"] == "True"
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=f"{M14}/figs")
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--from-csv", metavar="PATH",
                    help="plot from a committed CSV instead of harvesting job logs "
                         "(for Levante, which has no access to the JUPITER logs)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = read_csv(a.from_csv) if a.from_csv else harvest()
    write_csv(rows, f"{a.outdir}/m14_scaling.csv")
    if a.csv_only:
        return
    fig_levers(rows, f"{a.outdir}/fig_m14_levers.png")
    fig_ng5(rows, f"{a.outdir}/fig_m14_ng5.png")


if __name__ == "__main__":
    main()
