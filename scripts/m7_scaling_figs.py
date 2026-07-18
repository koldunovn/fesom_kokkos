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
dars/NG5 measured at dt180 — the CG-iteration dt-correction (~1-3 %) is NOT applied
in this first version (footnoted on the figure).
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
DT_RUN = {"core2": 1800.0, "farc": 900.0, "dars": 180.0, "ng5": 180.0}


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


def fig_scaling(df, cls, fname):
    common.set_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
    gc = gpu_class(df, cls)
    cpu = df[df.backend == "cpu"]
    counts = []
    for mesh in [m for m in common.MESH_ORDER if m in set(df.mesh)]:
        col = common.MESH_COLOR.get(mesh, "k")
        g = gc[gc.mesh == mesh].sort_values("nodes")
        c = cpu[cpu.mesh == mesh].sort_values("nodes")
        if len(g):
            axA.plot(g.nodes, g.sstep, marker="o", ms=4, ls="-", color=col,
                     label=common.MESH_LABEL.get(mesh, mesh))
            axB.plot(g.nodes, g.sypd, marker="o", ms=4, ls="-", color=col)
            counts += g.nodes.tolist()
        if len(c):
            axA.plot(c.nodes, c.sstep, marker="s", ms=3, ls="--", color=col, alpha=0.45)
            axB.plot(c.nodes, c.sypd, marker="s", ms=3, ls="--", color=col, alpha=0.45)
            counts += c.nodes.tolist()
        if len(g) >= 2:   # ideal 1/N anchored at the first GPU point
            gg = np.array(sorted(g.nodes))
            axA.plot(gg, g.sstep.iloc[0] * g.nodes.iloc[0] / gg, ls=":", lw=0.8,
                     color=col, alpha=0.4)
    axA.set_yscale("log")
    node_axis(axA, counts); node_axis(axB, counts)
    axA.set_ylabel("time per step  [s]")
    axA.set_title(f"(a) strong scaling — class {cls}  (solid GPU, dashed CPU, dotted 1/N)")
    axB.set_ylabel("SYPD @ production dt")
    axB.set_title("(b) throughput")
    axB.set_yscale("log")
    axA.legend(fontsize=6, loc="lower left")
    fig.suptitle("", fontsize=1)
    fig.text(0.995, 0.005, "dars/NG5 SYPD at dt240 from dt180 runs (CG dt-correction not applied)",
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
    fig_scaling(df, "A", f"{a.outdir}/fig_m7_scaling_A.png")
    fig_scaling(df, "B", f"{a.outdir}/fig_m7_scaling_B.png")
    fig_speedup(df, f"{a.outdir}/fig_m7_speedup.png")


if __name__ == "__main__":
    main()
