#!/usr/bin/env python3
"""First-version updated scaling figures from the s14 re-measurement fleet.

Harvests the fleet's job logs (GPU 4-leg ab_env points + CPU scale points) into a
tidy CSV, then renders, in the JAX-paper fig_scaling style (same mesh colors/labels
via paper_jax/scripts/common.py):

  fig_m7_scaling_A.png   (a) s/step vs NODES, CPU + GPU class-A(u)   (b) SYPD
  fig_m7_scaling_B.png   same with GPU class-B = best(B, Bp) per point
  fig_m7_speedup.png     node-for-node speedup CPU/GPU vs nodes, classes A + B

usage: m7_scaling_figs.py [--outdir /work/ab0995/a270088/port2/m7/scaling_figs]
Missing fleet points are simply absent from the curves (re-run as jobs land).
SYPD = dt_prod/(365*sstep) at production dt (core2 1800, farc 900, dars/NG5 240);
NG5 measured at dt180, dars at dt120 (rule 0.41) — the CG-iteration dt-correction
(~1-3 %) is NOT applied in this first version (footnoted on the figure).
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
import pandas as pd

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import common  # noqa: E402  (the paper's style: MESH_COLOR/MESH_LABEL/set_style)

M7 = "/work/ab0995/a270088/port2/m7"
DT_PROD = {"core2": 1800.0, "farc": 900.0, "dars": 240.0, "ng5": 240.0}
DT_RUN = {"core2": 1800.0, "farc": 900.0, "dars": 120.0, "ng5": 180.0}  # dars: rule 0.41 (dt180 cold-start-unstable past ~35 steps; 120 = the JAX dars dt)


def harvest():
    rows = []
    for f in glob.glob(f"{M7}/abenv.*.out"):
        txt = open(f, errors="replace").read()
        m = re.search(r"=== M7 ENV A/B\s+sc_(\w+)_g(\d+)n\s+mesh=\S+ nodes=(\d+) ntasks=(\d+)", txt)
        if not m:
            continue
        mesh, nn, ranks = m.group(1), int(m.group(3)), int(m.group(4))
        for leg, r1, r2 in re.findall(r"^  (A|Au|B|Bp): ([\d.]+)  ([\d.]+)$", txt, re.M):
            rows.append(dict(mesh=mesh, backend="gpu", leg=leg, nodes=nn, ranks=ranks,
                             sstep=min(float(r1), float(r2)), source=os.path.basename(f)))
    for f in glob.glob(f"{M7}/scale.*.out"):
        txt = open(f, errors="replace").read()
        m = re.search(r"=== M7 CPU sc_(\w+)_c(\d+)n mesh=\S+ nodes=(\d+) ntasks=(\d+)", txt)
        if not m:
            continue
        reps = [float(x) for x in re.findall(r"loop timing: \d+ steps.*->\s+([\d.]+) s/step", txt)]
        if reps:
            rows.append(dict(mesh=m.group(1), backend="cpu", leg="cpu", nodes=int(m.group(3)),
                             ranks=int(m.group(4)), sstep=min(reps), source=os.path.basename(f)))
    df = pd.DataFrame(rows).sort_values(["mesh", "backend", "leg", "nodes"])
    df["sypd"] = df.apply(lambda r: DT_PROD[r.mesh] / (365.0 * r.sstep), axis=1)
    return df


def node_axis(ax, counts):
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("nodes  (4×A100  /  128-core CPU)")


def gpu_class(df, cls):
    """class A -> the Au leg (bit-identical set incl. certified unbind);
       class B -> min(B, Bp) per (mesh, nodes) = the best climate-identical config."""
    g = df[df.backend == "gpu"]
    if cls == "A":
        return g[g.leg == "Au"]
    best = (g[g.leg.isin(["B", "Bp"])]
            .sort_values("sstep").groupby(["mesh", "nodes"], as_index=False).first())
    return best


def gpu_axis(ax, counts):
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("number of GPUs (A100)")


def decimal_log_yaxis(ax, lo, hi):
    """log y-scale but PLAIN decimal tick labels (no powers of 10)."""
    ticks = [t for t in (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5)
             if lo * 0.95 <= t <= hi * 1.05]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(lo * 0.9, hi * 1.1)


def fig_scaling(df, cls, fname, ylims):
    """Mirror of the paper's fig10 layout: (a) s/step vs #GPUs (log-y, dotted 1/N),
    (b) SYPD linear for CORE2 & farc, (c) SYPD linear for the multi-million-node
    meshes. GPU-only, exactly like the original; the CPU comparison lives in the
    speedup figure. `ylims` is shared between the class-A and class-B figures so
    the two are directly comparable."""
    common.set_style()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.5, 3.9))
    gc = gpu_class(df, cls)
    gc = gc.assign(ngpu=gc.ranks)
    meshes = [m for m in common.MESH_ORDER if m in set(gc.mesh)]
    small = [m for m in meshes if m in ("core2", "farc")]
    large = [m for m in meshes if m not in ("core2", "farc")]
    ca = cb = cc = []
    for mesh in meshes:
        col = common.MESH_COLOR.get(mesh, "k")
        g = gc[gc.mesh == mesh].sort_values("ngpu")
        if not len(g):
            continue
        axA.plot(g.ngpu, g.sstep, marker="o", ms=4, ls="-", color=col,
                 label=common.MESH_LABEL.get(mesh, mesh))
        ca = ca + g.ngpu.tolist()
        gg = np.array(sorted(g.ngpu))
        axA.plot(gg, g.sstep.iloc[0] * g.ngpu.iloc[0] / gg, ls=":", lw=0.8,
                 color=col, alpha=0.4)
        ax = axB if mesh in small else axC
        ax.plot(g.ngpu, g.sypd, marker="o", ms=4, ls="-", color=col)
        if mesh in small:
            cb = cb + g.ngpu.tolist()
        else:
            cc = cc + g.ngpu.tolist()
    axA.set_yscale("log")
    decimal_log_yaxis(axA, *ylims["sstep"])
    gpu_axis(axA, ca)
    axA.set_ylabel("time per step  [s]")
    axA.set_title(f"(a) strong scaling, class {cls}  (dotted = ideal 1/N)")
    axA.legend(fontsize=6, loc="lower left")
    gpu_axis(axB, cb)
    axB.set_ylabel("SYPD  (simulated yr / wall day)")
    axB.set_ylim(0, ylims["small"])
    axB.set_title("(b) throughput, CORE2 & farc")
    gpu_axis(axC, cc)
    axC.set_ylabel("SYPD  (simulated yr / wall day)")
    axC.set_ylim(0, ylims["large"])
    axC.set_title("(c) throughput, multi-million-node meshes")
    fig.tight_layout()
    fig.text(0.995, 0.005, "dars/NG5 SYPD at dt240 from dt120/dt180 runs (CG dt-correction not applied)",
             ha="right", fontsize=5, alpha=0.6)
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print("wrote", fname)


def fig_speedup(df, fname):
    common.set_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    cpu = df[df.backend == "cpu"][["mesh", "nodes", "sstep"]].rename(columns={"sstep": "cpu"})
    counts = []
    for cls, ls, alpha in (("A", "-", 1.0), ("B", "--", 0.8)):
        g = gpu_class(df, cls)[["mesh", "nodes", "sstep"]]
        j = g.merge(cpu, on=["mesh", "nodes"])
        j["speedup"] = j.cpu / j.sstep
        for mesh in [m for m in common.MESH_ORDER if m in set(j.mesh)]:
            s = j[j.mesh == mesh].sort_values("nodes")
            if not len(s):
                continue
            ax.plot(s.nodes, s.speedup, marker="o" if cls == "A" else "^", ms=4,
                    ls=ls, alpha=alpha, color=common.MESH_COLOR.get(mesh, "k"),
                    label=common.MESH_LABEL.get(mesh, mesh) if cls == "A" else None)
            counts += s.nodes.tolist()
    ax.axhline(1.0, color="k", lw=0.7, alpha=0.5)
    node_axis(ax, counts)
    ax.set_ylabel("node-for-node speedup  (CPU s/step ÷ GPU s/step)")
    ax.set_title("GPU vs CPU speedup — solid/circles: bit-identical (A),\n"
                 "dashed/triangles: climate-identical (B)")
    ax.legend(fontsize=6)
    fig.text(0.99, 0.01, "NG5 32N CPU at dt60 (dist_4096 cold-start limit, 5 model-h window;\n"
             "CG share dt-flattered) — the 32N NG5 speedup point is conservative",
             ha="right", fontsize=5, alpha=0.6)
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print("wrote", fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=f"{M7}/scaling_figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = harvest()
    df.to_csv(f"{a.outdir}/m7_scaling.csv", index=False)
    print(f"harvested {len(df)} rows -> {a.outdir}/m7_scaling.csv")
    print(df.groupby(["mesh", "backend"]).nodes.apply(lambda s: sorted(set(s))).to_string())
    both = pd.concat([gpu_class(df, "A"), gpu_class(df, "B")])
    ylims = {
        "sstep": (both.sstep.min(), both.sstep.max()),
        "small": 1.10 * both[both.mesh.isin(["core2", "farc"])].sypd.max(),
        "large": 1.10 * both[~both.mesh.isin(["core2", "farc"])].sypd.max(),
    }
    fig_scaling(df, "A", f"{a.outdir}/fig_m7_scaling_A.png", ylims)
    fig_scaling(df, "B", f"{a.outdir}/fig_m7_scaling_B.png", ylims)
    fig_speedup(df, f"{a.outdir}/fig_m7_speedup.png")


if __name__ == "__main__":
    main()
