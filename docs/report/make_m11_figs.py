#!/usr/bin/env python3
"""Figures for the M11 partitioning report. Style follows paper_jax/scripts/common.py
(mesh colors and equivalent-resolution labels) per the standing figure conventions.

Every number is a vetted campaign value from docs/PARTITIONING_M11.md; the job id that
produced it is cited beside it. Gains are quoted as "% faster than the shipped partition"
(positive = faster), the sign convention stated in each caption.

  fig_m11_board.pdf      best candidate per point, GPU and CPU panels, gate status coded
  fig_m11_mechanism.pdf  (a) GPU gain vs fractional nbr_max reduction  (b) 300-step race
                         vs 3,000-step re-measurement, 8 matched pairs
  fig_m11_gh200.pdf      A100 vs GH200 at the three cross-checked points
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
import common  # noqa: E402

common.set_style()
OUT = os.path.dirname(os.path.abspath(__file__))
C = common.MESH_COLOR
L = common.MESH_LABEL

# ---------------------------------------------------------------- fig 1: the board
# (mesh, ranks, gain %, setting, status) — status: cert | pending | failed
GPU = [
    # dars: the -19.7 % three-option arm FAILED accuracy outright (26908840, 5 controls);
    # MINCONN alone is the in-contention candidate, out on ssh rms only (+11 %)
    ("dars",  64,  14.3, "MINCONN",                   "ssh"),       # 26895260 screen, 26908840 gate
    ("ng5",   64,   9.8, "MINCONN",                   "ssh"),       # 26908635 screen, 26911630 gate (+8 %)
    ("core2",  4,   8.1, "MINCONN",                   "cert"),      # 26892875 + 3,000-step re-proof
    ("farc",  16,   3.6, "MINCONN+CONTIG",            "failed"),    # accuracy gate, 4 controls
]
CPU = [
    ("farc",  2048, 7.5, "Mt-KaHyPar w=100+nlev",     "cert"),
    ("core2",  512, 5.8, "Hilbert + engine",          "cert"),
    ("core2",  864, 4.9, "KaMinPar",                  "failed"),    # 26904986, salt, 5 controls
    ("dars",  2048, 4.2, "KaMinPar w=100+nlev",       "cert"),      # 26895271 race, 26904739 gate
    ("core2",  512, 3.8, "UFACTOR=30 alone",          "cert"),
    ("ng5",   2048, 0.0, "no candidate beat shipped", "null"),
]

HATCH = {"cert": None, "ssh": "//", "failed": "xx", "null": None}
ALPHA = {"cert": 1.0, "ssh": 0.55, "failed": 0.35, "null": 0.2}

fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharex=True)
for ax, rows, title in ((axes[0], GPU, "GPU"), (axes[1], CPU, "CPU")):
    y = np.arange(len(rows))[::-1]
    for yi, (mesh, ranks, gain, setting, status) in zip(y, rows):
        ax.barh(yi, gain, color=C[mesh], alpha=ALPHA[status],
                hatch=HATCH[status], edgecolor=C[mesh], linewidth=0.8)
        label = f"{L[mesh].split(' (')[0]} · {ranks} ranks"
        ax.text(-0.35, yi + 0.13, label, ha="right", va="center", fontsize=8)
        ax.text(-0.35, yi - 0.22, setting, ha="right", va="center", fontsize=6.5, color="0.45")
        val = "null" if status == "null" else f"{gain:.1f} %"
        note = {"cert": "", "ssh": "out on SSH rms only", "failed": "accuracy failed",
                "null": ""}[status]
        if note:
            ax.text(max(gain, 0) + 0.35, yi + 0.13, val, ha="left", va="center", fontsize=8)
            ax.text(max(gain, 0) + 0.35, yi - 0.22, note, ha="left", va="center",
                    fontsize=6.5, color="0.45")
        else:
            ax.text(max(gain, 0) + 0.35, yi, val, ha="left", va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlim(0, 17)
    ax.set_xlabel("% faster than the shipped partition")
    ax.grid(axis="y", visible=False)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
fig.subplots_adjust(wspace=0.75, left=0.13, right=0.98)
fig.savefig(f"{OUT}/fig_m11_board.pdf"); fig.savefig(f"{OUT}/fig_m11_board.png")
plt.close(fig)

# --------------------------------------------- fig 2: mechanism + protocol
fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.1))

# (a) GPU gain vs fractional reduction of the MAX partner count (Finding 43)
NBR = [  # mesh, ranks, base nbr_max, cand nbr_max, gain %
    ("farc",  16,  7,  6,  3.6),
    ("core2",  4,  3,  2,  8.1),
    ("dars",  64, 12,  7, 19.7),
]
for mesh, ranks, nb, na, gain in NBR:
    red = 100.0 * (nb - na) / nb
    a.plot(red, gain, "o", color=C[mesh], markersize=7)
    a.annotate(f"{L[mesh].split(' (')[0]}, {ranks} ranks\n{nb} → {na} partners",
               (red, gain), textcoords="offset points", xytext=(8, -3), fontsize=8)
a.set_xlim(0, 55)
a.set_ylim(0, 23)
a.set_xlabel("reduction of the maximum per-rank partner count (%)")
a.set_ylabel("GPU gain (% faster)")
a.set_title("(a) the GPU pays per message: gain follows nbr$_{max}$")

# (b) 300-step race vs 3,000-step re-measurement (Finding 32 + later screens)
PAIRS = [  # mesh, backend, race %, 3000-step %
    ("core2", "cpu", 3.70, 3.83),   # UFACTOR=30, 512 r
    ("core2", "cpu", 5.54, 5.83),   # Hilbert+engine, 512 r
    ("core2", "gpu", 7.50, 8.06),   # MINCONN, 4 r
    ("farc",  "cpu", 4.80, 5.49),   # MINCONN, 2048 r
    ("farc",  "cpu", 7.30, 7.52),   # Mt-KaHyPar, 2048 r
    ("dars",  "gpu", 18.64, 19.70), # MINCONN+CONTIG+u30, 64 r  (26893037 / 26895260)
    ("ng5",   "gpu", 9.71, 9.76),   # MINCONN, 64 r             (26904577 / 26908635)
    ("core2", "cpu", 4.07, 4.92),   # KaMinPar, 864 r           (26855293 / 26904576)
]
lim = 21
b.plot([0, lim], [0, lim], color="0.6", linewidth=0.9, zorder=1)
for mesh, backend, r300, r3000 in PAIRS:
    marker = "o" if backend == "gpu" else "s"
    b.plot(r300, r3000, marker, color=C[mesh], markersize=6, zorder=2)
b.set_xlim(0, lim)
b.set_ylim(0, lim)
b.set_xticks(range(0, lim + 1, 5))
b.set_yticks(range(0, lim + 1, 5))
b.set_xlabel("gain in the 300-step race (%)")
b.set_ylabel("gain re-measured over 3,000 steps (%)")
b.set_title("(b) every gain re-measured at length is equal or larger")
b.plot([], [], "o", color="0.3", label="GPU")
b.plot([], [], "s", color="0.3", label="CPU")
b.legend(loc="lower right", frameon=False)
fig.subplots_adjust(wspace=0.28, left=0.07, right=0.99)
fig.savefig(f"{OUT}/fig_m11_mechanism.pdf"); fig.savefig(f"{OUT}/fig_m11_mechanism.png")
plt.close(fig)

# --------------------------------------------- fig 3: A100 vs GH200 (Findings 42/44)
ROWS = [  # label, mesh, GH200 nodes, A100 gain %, GH200 gain %
    ("CORE2 · 4 ranks\n1 GH200 node",  "core2", 1,  8.06, 8.88),   # 26901093 min-of-3
    ("fArc · 16 ranks\n4 GH200 nodes", "farc",  4,  3.59, 0.78),   # 26903022 median, min-of-5
    ("DARS · 64 ranks\n16 GH200 nodes","dars", 16, 18.64, 2.68),   # 26901899 median, min-of-2
]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
x = np.arange(len(ROWS))
w = 0.34
for i, (label, mesh, nn, a100, gh) in enumerate(ROWS):
    ax.bar(i - w / 2, a100, w, color=C[mesh], label="A100" if i == 0 else None)
    ax.bar(i + w / 2, gh, w, color=C[mesh], alpha=0.45,
           hatch="//", edgecolor=C[mesh], label="GH200" if i == 0 else None)
    ax.text(i - w / 2, a100 + 0.3, f"{a100:.1f}", ha="center", fontsize=8)
    ax.text(i + w / 2, gh + 0.3, f"{gh:.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in ROWS], fontsize=8)
ax.set_ylabel("% faster than the shipped partition")
ax.set_ylim(0, 21.5)
ax.set_yticks(range(0, 21, 5))
ax.grid(axis="x", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color="0.35"),
           plt.Rectangle((0, 0), 1, 1, color="0.35", alpha=0.45, hatch="//")]
ax.legend(handles, ["A100 (Levante)", "GH200 (dolpung, staged halos)"], frameon=False)
fig.savefig(f"{OUT}/fig_m11_gh200.pdf"); fig.savefig(f"{OUT}/fig_m11_gh200.png")
plt.close(fig)

print("wrote", ", ".join(f"fig_m11_{n}.pdf" for n in ("board", "mechanism", "gh200")))
