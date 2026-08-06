#!/usr/bin/env python3
"""M10 SSH-solver campaign figures.

House conventions are inherited from m7_scaling_figs.py and paper_jax/scripts/common.py
(mesh colours/labels, set_style, log-base-2 node axes with PLAIN tick labels). Read that
script before changing anything here.

  fig_m10_payoff.png    (a) whole-step gain vs the SSH share of the step, CPU and GPU
                        (b) the solver's OWN gain, which is what differs between backends
  fig_m10_budget.png    where the step actually goes: per-phase busy vs MPI wait,
                        baseline against the best solver
  fig_m10_balance.png   (a) per-rank ocean compute vs owned 3-D nodes (the imbalance IS
                        bathymetry) (b) what balancing it costs, CPU and GPU

Every A/B row plotted is verified fallbacks=0: a leg that fired the fallback guard is a
variant/baseline MIXTURE, not an A/B point.

usage: m10_figs.py [--outdir /work/ab0995/a270088/port2/m10/figs]
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import common  # noqa: E402

AB = "/work/ab0995/a270088/port2/m10/ab"
SOLVER_LABEL = {"cg2": "cg2 (Chronopoulos–Gear)", "oati": "oati (PIPECG-OATI)",
                "pcsi": "pcsi (Chebyshev P-CSI)"}
SOLVER_COLOR = {"cg2": "#7f7f7f", "oati": "#d62728", "pcsi": "#1f77b4"}


# --------------------------------------------------------------------------- harvest
def _mesh_of(tag):
    for m in ("core2", "farc", "dars", "ng5"):
        if m in tag:
            return m
    return None


def harvest_ab():
    """Every A/B table, with the per-leg fallback count. Rows with fallbacks>0 are kept
    but flagged, so the figure code can drop them explicitly rather than silently."""
    rows = []
    for f in sorted(glob.glob(f"{AB}/ab.*.out")) + sorted(glob.glob(f"{AB}/abcpu.*.out")):
        txt = open(f, errors="replace").read()
        m = re.search(r"A/B RESULT — (\S+)", txt)
        h = re.search(r"nodes=(\d+) ntasks=(\d+)", txt)
        if not m or not h:
            continue
        tag, nodes, ranks = m.group(1), int(h.group(1)), int(h.group(2))
        mesh = _mesh_of(tag)
        if mesh is None:
            continue
        backend = "cpu" if os.path.basename(f).startswith("abcpu") else "gpu"
        job = re.search(r"\.(\d+)\.out$", f).group(1)
        # per-leg fallbacks live in the run dirs; index legs in table order
        fb = {}
        for d in glob.glob(f"{AB}/{tag}*"):
            if not os.path.isdir(d):
                continue
            for i in range(6):
                tot = 0
                seen = False
                for rep in ("a", "b"):
                    p = f"{d}/err_{i}_{rep}.txt"
                    if os.path.exists(p):
                        seen = True
                        tot += open(p, "rb").read().count(b"FALLBACK on solve")
                if seen:
                    fb[i] = tot
        body = txt[txt.index("A/B RESULT"):]
        li = -1
        for line in body.splitlines():
            mm = re.match(r"^  (\S+)\s+([\d.]+)\s+([+-][\d.]+)% \|\s+([\d.]+)\s+([+-][\d.]+)%\s+"
                          r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if not mm:
                continue
            li += 1
            legname = mm.group(1)
            solver = "cg"
            for s in ("cg2", "oati", "pcsi", "pipecg"):
                if f"SOLVER={s}" in legname:
                    solver = s
            rows.append(dict(tag=tag, job=job, mesh=mesh, backend=backend, nodes=nodes,
                             ranks=ranks, solver=solver, sstep=float(mm.group(2)),
                             d_total=float(mm.group(3)), ssh_ms=float(mm.group(4)),
                             d_ssh=float(mm.group(5)), ssh_pct=float(mm.group(6)),
                             iters=float(mm.group(7)), fallbacks=fb.get(li, None)))
    return rows


def configs(rows):
    """Group into COMPLETE A/B points: a configuration counts only if its baseline AND at
    least one variant are both verified fallbacks=0 in the SAME job. Taking the "best solver"
    across a configuration where some legs were dropped for firing the guard would silently
    report the only survivor as the winner.

    NG5 CPU is excluded outright: those runs hit the model's vertical-CFL blow-up guard at
    step 150-175 (a documented dt180 cold-start limit shared with Fortran), so they are not
    valid A/B points whatever the fallback count says.

    Three further exclusions, each learned the hard way:
      * `ps_*`   phase-profiling runs. PMPI interposition adds overhead in proportion to MPI
                 call count, so they OVERSTATE the gain (farc 2048: -15.2 % here against
                 -13.3 % from the timing protocol). Never a timing point.
      * `eig_*`, `*checksweep*`  parameter sweeps whose legs are variants of ONE solver, not
                 the standard four-leg set.
      * a configuration missing ANY of its four legs. If cg2 and oati were dropped for firing
                 the guard, "best solver" silently becomes the only survivor — which is how
                 farc GPU 64 r would otherwise enter the figure as pcsi's +5.76 % LOSS."""
    SKIP = ("ps_", "eig_", "partab")
    byjob = {}
    for r in rows:
        if any(k in r["tag"] for k in SKIP) or "checksweep" in r["tag"]:
            continue
        byjob.setdefault((r["job"], r["mesh"], r["backend"], r["ranks"]), []).append(r)
    out = []
    for (job, mesh, backend, ranks), legs in byjob.items():
        if mesh == "ng5" and backend == "cpu":
            continue
        base = [l for l in legs if l["solver"] == "cg" and l["fallbacks"] == 0]
        var = [l for l in legs if l["solver"] != "cg" and l["fallbacks"] == 0]
        if not base or len(var) < 3:          # require the COMPLETE four-leg set
            continue
        out.append(dict(job=job, mesh=mesh, backend=backend, ranks=ranks,
                        ssh_pct=base[0]["ssh_pct"],           # the BASELINE share of the step
                        legs={l["solver"]: l for l in var}))
    # newest job per configuration
    best = {}
    for c in out:
        k = (c["mesh"], c["backend"], c["ranks"])
        if k not in best or int(c["job"]) > int(best[k]["job"]):
            best[k] = c
    return list(best.values())


# --------------------------------------------------------------------------- fig 1
def fig_payoff(cfgs, fname):
    common.set_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    for backend, marker, face in (("cpu", "o", "full"), ("gpu", "^", "none")):
        for mesh in [m for m in common.MESH_ORDER if m in {c["mesh"] for c in cfgs}]:
            g = sorted([c for c in cfgs if c["backend"] == backend and c["mesh"] == mesh],
                       key=lambda c: c["ssh_pct"])
            if not g:
                continue
            best = [min(c["legs"].values(), key=lambda l: l["d_total"]) for c in g]
            axA.plot([c["ssh_pct"] for c in g], [-b["d_total"] for b in best],
                     marker=marker, ls="none", ms=6, color=common.MESH_COLOR[mesh],
                     mfc=common.MESH_COLOR[mesh] if face == "full" else "none",
                     label=None)
    axA.axhline(0, color="k", lw=0.8, alpha=0.5)
    axA.set_xlabel("SSH solve, % of the model step")
    axA.set_ylabel("whole-step speed-up  [%]")
    axA.set_title("(a) the payoff follows the SSH share")

    # (b) the solver's own gain — the quantity that differs between backends
    labels, vals, cols, hatches = [], [], [], []
    for backend in ("cpu", "gpu"):
        for mesh in ("core2", "farc", "dars"):
            g = [c for c in cfgs if c["backend"] == backend and c["mesh"] == mesh
                 and "oati" in c["legs"]]
            if not g:
                continue
            c = min(g, key=lambda c: c["legs"]["oati"]["d_ssh"])
            labels.append(f"{mesh}\n{backend.upper()} {c['ranks']}r")
            vals.append(-c["legs"]["oati"]["d_ssh"])
            cols.append(common.MESH_COLOR[mesh])
            hatches.append("" if backend == "cpu" else "//")
    x = np.arange(len(vals))
    bars = axB.bar(x, vals, color=cols, edgecolor="k", lw=0.6)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=7)
    axB.axhline(0, color="k", lw=0.8)
    axB.set_ylabel("solve-phase speed-up, best case  [%]")
    axB.set_title("(b) the solver's own gain (oati); hatched = GPU")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", color=common.MESH_COLOR[m],
                      label=common.MESH_LABEL[m]) for m in ("core2", "farc", "dars")]
    handles += [Line2D([], [], ls="none", marker="o", color="k", label="CPU (filled)"),
                Line2D([], [], ls="none", marker="^", color="k", mfc="none", label="GPU (open)")]
    axA.legend(handles=handles, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(fname)
    print("wrote", fname)


# --------------------------------------------------------------------------- fig 2
PHASE_ORDER = ["force", "ice", "icedyn", "iceadv", "ocean", "cg", "other"]
PHASE_LABEL = {"force": "forcing", "ice": "ice thermo", "icedyn": "ice dynamics",
               "iceadv": "ice advection", "ocean": "ocean", "cg": "SSH solve",
               "other": "other"}


def read_phases(path):
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\[phasestats\]\s+(\w+)\s+\|\s+[\d.]+ /\s+([\d.]+) /\s+[\d.]+ @\d+\s+\|"
                     r"\s+[\d.]+ /\s+([\d.]+) /", line)
        if m and m.group(1) in PHASE_ORDER:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def fig_budget(fname):
    common.set_style()
    cases = [("CORE2, 864 CPU ranks", f"{AB}/ps_core2_864_26742297/log_0_a.txt",
              f"{AB}/ps_core2_864_26742297/log_1_a.txt", "baseline cg", "pcsi"),
             ("fArc, 2048 CPU ranks", f"{AB}/ps_farc_2048_26742298/log_0_a.txt",
              f"{AB}/ps_farc_2048_26742298/log_1_a.txt", "baseline cg", "oati")]
    cases = [c for c in cases if os.path.exists(c[1]) and os.path.exists(c[2])]
    if not cases:
        print("fig_budget: no phasestats logs found — skipped")
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.0))
    if len(cases) == 1:
        axes = [axes]
    for ax, (title, f0, f1, l0, l1) in zip(axes, cases):
        A, B = read_phases(f0), read_phases(f1)
        ph = [p for p in PHASE_ORDER if p in A and p in B]
        x = np.arange(len(ph))
        w = 0.38
        ax.bar(x - w/2, [A[p][0] for p in ph], w, color="#4c72b0", label=f"{l0}: compute")
        ax.bar(x - w/2, [A[p][1] for p in ph], w, bottom=[A[p][0] for p in ph],
               color="#c44e52", label=f"{l0}: MPI wait")
        ax.bar(x + w/2, [B[p][0] for p in ph], w, color="#4c72b0", alpha=0.55,
               label=f"{l1}: compute")
        ax.bar(x + w/2, [B[p][1] for p in ph], w, bottom=[B[p][0] for p in ph],
               color="#c44e52", alpha=0.55, label=f"{l1}: MPI wait")
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABEL[p] for p in ph], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("ms per model step")
        ax.set_title(title)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fname)
    print("wrote", fname)


# --------------------------------------------------------------------------- fig 3
def fig_balance(fname):
    common.set_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    n3p = "/scratch/a/a270088/farc_own3d.npy"
    log = f"{AB}/ps_farc_2048_26742298/log_0_a.txt"
    if os.path.exists(n3p) and os.path.exists(log):
        n3 = np.load(n3p)
        busy = {}
        for line in open(log, errors="replace"):
            m = re.match(r"\[phasestats-rank\]\s+(\d+) \|\s+([\d.\s]+)\|", line)
            if m:
                v = [float(x) for x in m.group(2).split()]
                if len(v) == 8:
                    busy[int(m.group(1))] = v[5]      # force ice icedyn iceadv coupl OCEAN cg other
        if busy:
            r = np.array([busy[i] for i in range(len(busy))])
            n = n3[:len(r)]
            axA.plot(n / 1000.0, r, "o", ms=2.0, alpha=0.35,
                     color=common.MESH_COLOR["farc"], mec="none")
            a, b = np.polyfit(n, r, 1)
            xs = np.array([n.min(), n.max()])
            axA.plot(xs / 1000.0, a * xs + b, "-", color="k", lw=1.2,
                     label=f"{a*1000:.2f} ms per 1000 3-D nodes\nr = {np.corrcoef(n, r)[0,1]:.3f}")
            axA.legend(fontsize=7, loc="upper left")
    axA.set_xlabel("3-D nodes owned by the rank  [thousands]")
    axA.set_ylabel("ocean-phase compute  [ms/step]")
    axA.set_title("(a) the imbalance is bathymetry, not ice")

    # (b) what balancing it costs. Measured, one allocation per rung.
    cpu = [(495, -4.62), (248, 0.00), (146, +8.71)]          # CORE2 CPU 256/512/864 r
    gpu = [(15857, +29.74)]                                   # CORE2 GPU 2 nodes / 8 r
    axB.plot([c[0] for c in cpu], [c[1] for c in cpu], "o-", color=common.MESH_COLOR["core2"],
             label="CPU (256 / 512 / 864 ranks)")
    axB.plot([g[0] for g in gpu], [g[1] for g in gpu], "^", ms=8, mfc="none",
             color=common.MESH_COLOR["core2"], label="GPU (2 nodes, 8 ranks)")
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xscale("log", base=2)
    ticks = [146, 248, 495, 15857]
    axB.set_xticks(ticks)
    axB.set_xticklabels([str(t) for t in ticks])
    axB.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axB.set_xlabel("vertices per rank")
    axB.set_ylabel("step time, balanced vs unbalanced  [%]")
    axB.set_title("(b) balancing costs more than it saves")
    axB.legend(fontsize=7)
    axB.text(0.5, 0.06, "below zero = balancing wins", transform=axB.transAxes,
             fontsize=6.5, ha="center", style="italic", alpha=0.7)
    fig.tight_layout()
    fig.savefig(fname)
    print("wrote", fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/work/ab0995/a270088/port2/m10/figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cfgs = configs(harvest_ab())
    print(f"harvested {len(cfgs)} COMPLETE A/B configurations (baseline + >=1 variant, all fallbacks=0)")
    for c in sorted(cfgs, key=lambda c: (c["mesh"], c["backend"], c["ranks"])):
        bs = min(c["legs"].values(), key=lambda l: l["d_total"])
        print(f"   {c['mesh']:6s} {c['backend']}  {c['ranks']:5d}r  SSH {c['ssh_pct']:5.1f}%  "
              f"best {bs['solver']:5s} {bs['d_total']:+6.2f}%  (job {c['job']})")
    fig_payoff(cfgs, f"{a.outdir}/fig_m10_payoff.png")
    fig_budget(f"{a.outdir}/fig_m10_budget.png")
    fig_balance(f"{a.outdir}/fig_m10_balance.png")


if __name__ == "__main__":
    main()
