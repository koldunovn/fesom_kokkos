#!/usr/bin/env python3
"""JUPITER (GH200 booster) vs dolpung (GH200) vs Levante (A100) SYPD scaling.

Adds the JUPITER twin fleet to the existing dolpung-vs-A100 comparison. Same
conventions as dolpung_sypd_figs.py (paper common.py mesh colors; SYPD at
production dt; measured CG dt-corr dars ×1.022 / NG5 ×1.011). Small meshes
(CORE2+farc) and multi-million-node meshes (dars+NG5) get separate figures in
three precision variants:

  fig_jupiter_sypd_sp_{small,large}.{png,pdf}     single precision
  fig_jupiter_sypd_dp_{small,large}.{png,pdf}     double precision
  fig_jupiter_sypd_both_{small,large}.{png,pdf}   SP + DP overlaid

Series (fastest certified config per platform, min-of-2 everywhere):
  JUPITER = twin s25+STAGE fleet 2026-07-22/23 (docs/JUPITER_FLEET_RESULTS.md;
            Stages/2025 GCC13.3+PSMPI5.11+CUDA12.6, Kokkos 4.4.01,
            FESOM_HALO_STAGE=1) — the fastest measured JUPITER config at every
            production point. dp/sp below = min(plain, cgp) of the twin columns.
  dolpung = GH200 fleet v2 (STAGE transport), imported from dolpung_sypd_figs.
  A100    = Levante m8 Bp fleet (ends at g16 = 64 GPUs), imported likewise.

Encoding: color = mesh; PLATFORM = line style + marker fill —
  JUPITER solid + filled · dolpung dashed + filled · A100 dotted + open;
in the "both" variant marker shape = precision (circle DP, triangle SP).
Both GH200 machines are filled (solid vs dashed); A100 is open + dotted.

x-axis in GPUs — all three platforms have 4 GPUs/node. JUPITER's twin gN
points are N nodes (= 4N GPUs), the same node key dolpung/A100 use.

usage: jupiter_sypd_figs.py [--outdir /work/ab0995/a270088/port2/jupiter/figs]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import common  # noqa: E402  (MESH_COLOR/MESH_LABEL/MESH_ORDER/set_style)

# dolpung + A100 numbers, DT tables and the SYPD helper are the single source of
# truth in dolpung_sypd_figs.py — import them so the two figures never drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dolpung_sypd_figs as dsf  # noqa: E402

DOLPUNG = dsf.GH200
A100 = dsf.A100
sypd = dsf.sypd          # sypd(mesh, sstep) -> DT_PROD/(365·sstep·DT_CORR)
gpu_axis = dsf.gpu_axis
PREC_MARKER = dsf.PREC_MARKER
PREC_NAME = dsf.PREC_NAME

# JUPITER twin s25+STAGE fleet, s/step, keyed by NODES (4 GH200 each).
# dp = min(dp, dp_cgp), sp = min(sp, sp_cgp) from docs/JUPITER_FLEET_RESULTS.md
# (the twin columns) — the fastest DP-family / SP-family leg, matching how the
# dolpung + A100 tables were reduced. ng5_g2 has no cgp leg (plain used).
JUPITER = {
    "sp": {
        "core2": {1: 0.0329, 2: 0.0292, 4: 0.0302, 8: 0.0299},
        "farc":  {1: 0.0644, 2: 0.0470, 4: 0.0408, 8: 0.0386, 16: 0.0375, 32: 0.0416},
        "dars":  {1: 0.2977, 2: 0.1630, 4: 0.0939, 8: 0.0545, 16: 0.0389,
                  32: 0.0363, 64: 0.0328, 128: 0.0309},
        "ng5":   {2: 0.5133, 4: 0.2661, 8: 0.1467, 16: 0.0815, 32: 0.0537,
                  64: 0.0432, 128: 0.0410, 256: 0.0432},
    },
    "dp": {
        "core2": {1: 0.0370, 2: 0.0318, 4: 0.0325, 8: 0.0317},
        "farc":  {1: 0.0755, 2: 0.0540, 4: 0.0456, 8: 0.0418, 16: 0.0399, 32: 0.0450},
        "dars":  {1: 0.3445, 2: 0.1958, 4: 0.1128, 8: 0.0642, 16: 0.0459,
                  32: 0.0482, 64: 0.0376, 128: 0.0344},
        "ng5":   {2: 0.5800, 4: 0.3043, 8: 0.1722, 16: 0.0961, 32: 0.0630,
                  64: 0.0506, 128: 0.0453, 256: 0.0489},
    },
}

# name, table, linestyle, filled marker, legend label — JUPITER is the hero.
PLATFORMS = [
    ("jupiter", JUPITER, "-",  True,  "JUPITER (GH200)"),
    ("dolpung", DOLPUNG, "--", True,  "dolpung (GH200)"),
    ("a100",    A100,    ":",  False, "A100 (Levante)"),
]


def foot(variant):
    p = {"sp": "single precision (FP32, FP64 islands)",
         "dp": "double precision",
         "both": "SP = FP32 with FP64 islands"}[variant]
    return (p + "; JUPITER = twin s25+STAGE fleet (fastest measured config), dolpung = GH200 fleet v2 (STAGE); A100 = m8 Bp fleet (ends at 64 GPUs);\n"
            "dolpung big-mesh fleet is partial (dars to 64 GPUs, no NG5 point yet); SYPD at production dt (CORE2 1800 s, farc 1200 s, dars/NG5 240 s); measured CG dt-corr (dars ×1.022, NG5 ×1.011)")


def one_fig(meshes, variant, title, fname):
    precs = ["dp", "sp"] if variant == "both" else [variant]
    common.set_style()
    fig, ax = plt.subplots(figsize=(5.9, 4.0))
    counts = []
    for mesh in [m for m in common.MESH_ORDER if m in meshes]:
        col = common.MESH_COLOR.get(mesh, "k")
        for prec in precs:
            mk = PREC_MARKER[prec] if variant == "both" else "o"
            for pname, tbl, ls, filled, _ in PLATFORMS:
                pts = tbl[prec].get(mesh, {})
                ns = sorted(pts)
                if not ns:
                    continue
                gp = [4 * n for n in ns]
                mfc = col if filled else "none"
                # mesh color swatch: label only the first line drawn per mesh
                # (JUPITER, first precision) so the legend carries each mesh once
                lbl = (common.MESH_LABEL.get(mesh, mesh)
                       if (pname == "jupiter" and prec == precs[0]) else None)
                ax.plot(gp, [sypd(mesh, pts[n]) for n in ns], marker=mk, ms=4,
                        ls=ls, color=col, markerfacecolor=mfc, label=lbl)
                counts += gp
    gpu_axis(ax, counts)
    ax.set_ylabel("SYPD  (simulated yr / wall day)")
    ax.set_ylim(0, None)
    ax.set_title(title)
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color="0.35", ls="-", marker="s", ms=4, label="JUPITER (GH200)"),
        Line2D([], [], color="0.35", ls="--", marker="s", ms=4, label="dolpung (GH200)"),
        Line2D([], [], color="0.35", ls=":", marker="s", ms=4, markerfacecolor="none",
               label="A100 (Levante)")]
    if variant == "both":
        handles += [Line2D([], [], color="0.35", ls="", marker="o", ms=4, label="FP64"),
                    Line2D([], [], color="0.35", ls="", marker="^", ms=4, label="FP32")]
    # shared horizontal legend BELOW the panel (user rule: never in-axes, f2227c4);
    # wrap to <=4 per row so the platform + mesh + precision keys don't clip
    ncol = min(len(handles), 4)
    fig.legend(handles=handles, ncol=ncol, fontsize=7.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.005))
    rows = -(-len(handles) // ncol)
    fig.tight_layout(rect=[0, 0.045 * rows + 0.02, 1, 1])
    fig.text(0.995, -0.05, foot(variant), ha="right", fontsize=4.6, alpha=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname + ".png/.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="/work/ab0995/a270088/port2/jupiter/figs")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for variant in ("sp", "dp", "both"):
        one_fig({"core2", "farc"}, variant,
                f"throughput, CORE2 & farc — {PREC_NAME[variant]}",
                os.path.join(a.outdir, f"fig_jupiter_sypd_{variant}_small"))
        one_fig({"dars", "ng5"}, variant,
                f"throughput, multi-million-node meshes — {PREC_NAME[variant]}",
                os.path.join(a.outdir, f"fig_jupiter_sypd_{variant}_large"))


if __name__ == "__main__":
    main()
