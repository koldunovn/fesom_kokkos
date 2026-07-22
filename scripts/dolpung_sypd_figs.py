#!/usr/bin/env python3
"""dolpung (GH200) vs Levante (A100) SYPD scaling figures, per user request
2026-07-22: SEPARATE figures for small meshes (CORE2+farc) and multi-million-node
meshes (dars+NG5), in three precision variants:

  fig_dolpung_sypd_sp_{small,large}.{png,pdf}     single precision
  fig_dolpung_sypd_dp_{small,large}.{png,pdf}     double precision
  fig_dolpung_sypd_both_{small,large}.{png,pdf}   SP + DP overlaid

Series (fastest certified config per platform, min-of-2 everywhere):
  GH200 = dolpung fleet 2026-07-22 (docs/plans/20260722-dolpung-GH200-SCALING.md),
          FESOM_SPEED=1 + FESOM_HOST_HALO=1 (+CGPOLY3 where it helps), 300 steps;
          DP = m7-speed build, SP = m8 -DFESOM_PRECISION=single build;
          logs /work/ab0995/a270088/port2/dolpung/scale/.
  A100  = m8 Bp fleet (SPEED=1+EVPWIDE=8+CGPOLY=3+proto env), dp+sp legs,
          /work/ab0995/a270088/port2/mp/gate2/scal_bp_*. The Bp series ends at g16
          (64 GPUs) — no g32 Bp points exist; curves end there honestly.

Encoding: color = mesh (paper common.py); GH200 = solid+filled, A100 = dashed+open;
in the "both" variant marker shape = precision (circle DP, triangle SP).

SYPD = DT_PROD/(365*sstep*CORR), production dt: core2 1800, farc 1200 (user
2026-07-21: 20-min step, m8 convention), dars/NG5 240; measured CG dt-corrections
dars x1.0222 / NG5 x1.0110 (jobs/job_m7_dtpair) applied to BOTH platforms (both
measured at dt120/dt180). x-axis in GPUs — both platforms have 4 GPUs/node.

usage: dolpung_sypd_figs.py [--outdir /work/ab0995/a270088/port2/dolpung/figs]
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

DT_PROD = {"core2": 1800.0, "farc": 1200.0, "dars": 240.0, "ng5": 240.0}
DT_CORR = {"core2": 1.0, "farc": 1.0, "dars": 1.0222, "ng5": 1.0110}

# s/step, min-of-2, min over plain/cgp legs, keyed by NODES (4 GPU each).
# FLEET v2 (2026-07-22 evening, jobs 26405881+): FESOM_SPEED=1 + FESOM_HALO_STAGE=1
# (+CGPOLY3 where it wins) — the STAGE transport (device-packed halos, MPI on
# pinned-host mirrors; CGPIPE/CGPOLY live). v1 HOST_HALO numbers are SUPERSEDED
# and deliberately absent. Points still in flight (ng5, farc/dars g32) are
# simply missing and appear as the campaign closes.
GH200 = {
    "sp": {
        "core2": {1: 0.0397, 2: 0.0295, 4: 0.0303, 8: 0.0306},
        "farc":  {1: 0.0770, 2: 0.0524, 4: 0.0446, 8: 0.0427, 16: 0.0412},
        "dars":  {1: 0.3175, 2: 0.1682, 4: 0.0947, 8: 0.0590, 16: 0.0448},
        "ng5":   {},
    },
    "dp": {
        "core2": {1: 0.0463, 2: 0.0328, 4: 0.0328, 8: 0.0325},
        "farc":  {1: 0.0930, 2: 0.0606, 4: 0.0507, 8: 0.0472, 16: 0.0450},
        "dars":  {1: 0.3697, 2: 0.1964, 4: 0.1150, 8: 0.0704, 16: 0.0517},
        "ng5":   {},
    },
}
A100 = {
    "sp": {
        "core2": {1: 0.0486, 2: 0.0429, 4: 0.0462, 8: 0.0463},
        "farc":  {1: 0.1077, 2: 0.0790, 4: 0.0690, 8: 0.0648, 16: 0.0614},
        "dars":  {2: 0.3093, 4: 0.1759, 8: 0.1117, 16: 0.0860},
        "ng5":   {2: 0.9843, 4: 0.5195, 8: 0.2894, 16: 0.1629},
    },
    "dp": {
        "core2": {1: 0.0558, 2: 0.0443, 4: 0.0517, 8: 0.0499},
        "farc":  {1: 0.1308, 2: 0.1021, 4: 0.0705, 8: 0.0639, 16: 0.0639},
        "dars":  {2: 0.3756, 4: 0.2151, 8: 0.1314, 16: 0.0955},
        "ng5":   {2: 1.1573, 4: 0.6197, 8: 0.3466, 16: 0.2003},
    },
}

PREC_NAME = {"sp": "single precision", "dp": "double precision",
             "both": "single & double precision"}
PREC_MARKER = {"dp": "o", "sp": "^"}


def foot(variant):
    p = {"sp": "single precision (FP32, FP64 islands)",
         "dp": "double precision",
         "both": "SP = FP32 with FP64 islands"}[variant]
    return (p + "; GH200 = dolpung fleet v2 (STAGE transport) 2026-07-22, A100 = m8 Bp fleet (ends at 64 GPUs); GH200 NG5 + 128-GPU points in flight;\n"
            "SYPD at production dt (CORE2 1800 s, farc 1200 s, dars/NG5 240 s); measured CG dt-corr applied (dars ×1.022, NG5 ×1.011)")


def gpu_axis(ax, counts):
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("number of GPUs  (4 per node)")


def sypd(mesh, sstep):
    return DT_PROD[mesh] / (365.0 * sstep * DT_CORR[mesh])


def one_fig(meshes, variant, title, fname):
    precs = ["dp", "sp"] if variant == "both" else [variant]
    common.set_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    counts = []
    for mesh in [m for m in common.MESH_ORDER if m in meshes]:
        col = common.MESH_COLOR.get(mesh, "k")
        for prec in precs:
            mk = PREC_MARKER[prec] if variant == "both" else "o"
            for tbl, ls, mfc in ((GH200[prec], "-", col), (A100[prec], "--", "none")):
                pts = tbl.get(mesh, {})
                ns = sorted(pts)
                gp = [4 * n for n in ns]
                lbl = (common.MESH_LABEL.get(mesh, mesh)
                       if (ls == "-" and prec == precs[0]) else None)
                ax.plot(gp, [sypd(mesh, pts[n]) for n in ns], marker=mk, ms=4,
                        ls=ls, color=col, markerfacecolor=mfc, label=lbl)
                counts += gp
    gpu_axis(ax, counts)
    ax.set_ylabel("SYPD  (simulated yr / wall day)")
    ax.set_ylim(0, None)
    ax.set_title(title)
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color="0.35", ls="-", marker="s", ms=4, label="GH200 (dolpung)"),
        Line2D([], [], color="0.35", ls="--", marker="s", ms=4, markerfacecolor="none",
               label="A100 (Levante)")]
    if variant == "both":
        handles += [Line2D([], [], color="0.35", ls="", marker="o", ms=4, label="FP64"),
                    Line2D([], [], color="0.35", ls="", marker="^", ms=4, label="FP32")]
    # shared horizontal legend BELOW the panel (user rule: never in-axes, f2227c4)
    fig.legend(handles=handles, ncol=len(handles), fontsize=7.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.text(0.995, -0.045, foot(variant), ha="right", fontsize=4.6, alpha=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(f"{fname}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", fname + ".png/.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="/work/ab0995/a270088/port2/dolpung/figs")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for variant in ("sp", "dp", "both"):
        one_fig({"core2", "farc"}, variant,
                f"throughput, CORE2 & farc — {PREC_NAME[variant]}",
                os.path.join(a.outdir, f"fig_dolpung_sypd_{variant}_small"))
        one_fig({"dars", "ng5"}, variant,
                f"throughput, multi-million-node meshes — {PREC_NAME[variant]}",
                os.path.join(a.outdir, f"fig_dolpung_sypd_{variant}_large"))


if __name__ == "__main__":
    main()
