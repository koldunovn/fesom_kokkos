#!/usr/bin/env python3
"""Figures for the M11 partitioning report. Style follows paper_jax/scripts/common.py
(mesh colors and equivalent-resolution labels) per the standing figure conventions.

Every number is a vetted campaign value from docs/PARTITIONING_M11.md; the job id that
produced it is cited beside it. Gains are quoted as "% faster than the shipped partition"
(positive = faster), the sign convention stated in each caption.

  fig_m11_board.pdf       best candidate per point, GPU and CPU panels, disturbance coded
  fig_m11_mechanism.pdf   (a) GPU gain vs fractional nbr_max reduction  (b) 300-step race
                          vs 3,000-step re-measurement, 8 matched pairs
  fig_m11_gh200.pdf       A100 vs GH200 at the three cross-checked points
  fig_m11_disturbance.pdf per candidate and field: 20-step rms disturbance relative to the
                          top of the seed-control spread (harvested from the gate job logs)
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
# (mesh, ranks, gain %, setting, status). All arms shown passed the 3,000-step stability
# screen; status codes the 20-step DISTURBANCE vs the seed-control spread (user protocol
# decision 2026-08-13 — graded reporting, not a pass/fail accuracy verdict):
#   t1  = inside/below the seed spread     ssh = SSH-only excursion
#   ts  = temperature/salinity excursion   null = no candidate beat shipped
GPU = [
    ("dars",  64,  19.7, "MINCONN+CONTIG+UFACTOR=30",        "ts"),        # 26895260 screen; 26908840: T rms +13 %, T max x1.8-2.5
    ("dars",  64,  14.3, "MINCONN",                   "ssh"),       # 26895260 screen, 26908840 gate (+11 %, stopping mech.)
    ("ng5",   64,   9.8, "MINCONN",                   "ssh"),       # 26908635 screen, 26911630 gate (+8 %, unexplained)
    ("core2",  4,   8.1, "MINCONN",                   "t1"),        # 26892875 + 3,000-step re-proof
    ("farc",  16,   3.6, "MINCONN+CONTIG",            "ts"),        # 26893934: T rms +33 %, 4 controls
]
CPU = [
    ("farc",  2048, 7.5, "Mt-KaHyPar w=100+nlev",     "t1"),
    ("core2",  512, 5.8, "Hilbert renumbering + KaHIP",          "t1"),
    ("core2",  864, 4.9, "KaMinPar",                  "ts"),        # 26904986: salt rms +24 %, 5 controls
    ("dars",  2048, 4.2, "KaMinPar w=100+nlev",       "t1"),        # 26895271 race, 26904739 gate
    ("core2",  512, 3.8, "UFACTOR=30",          "t1"),
    ("ng5",   2048, 0.0, "no candidate was faster", "null"),
]

HATCH = {"t1": None, "ssh": "//", "ts": "xx", "null": None}
ALPHA = {"t1": 1.0, "ssh": 0.55, "ts": 0.35, "null": 0.2}

fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharex=True)
for ax, rows, title, unit in ((axes[0], GPU, "GPU", "GPUs"), (axes[1], CPU, "CPU", "cores")):
    y = np.arange(len(rows))[::-1]
    for yi, (mesh, ranks, gain, setting, status) in zip(y, rows):
        ax.barh(yi, gain, color=C[mesh], alpha=ALPHA[status],
                hatch=HATCH[status], edgecolor=C[mesh], linewidth=0.8)
        label = f"{L[mesh].split(' (')[0]} · {ranks} {unit}"
        ax.text(-0.35, yi + 0.13, label, ha="right", va="center", fontsize=8)
        ax.text(-0.35, yi - 0.22, setting, ha="right", va="center", fontsize=6.5, color="0.45")
        val = "none" if status == "null" else f"{gain:.1f} %"
        note = {"t1": "", "ssh": "ssh only, beyond seed scatter", "ts": "T or S beyond seed scatter",
                "null": ""}[status]
        if note:
            ax.text(max(gain, 0) + 0.35, yi + 0.13, val, ha="left", va="center", fontsize=8)
            ax.text(max(gain, 0) + 0.35, yi - 0.22, note, ha="left", va="center",
                    fontsize=6.5, color="0.45")
        else:
            ax.text(max(gain, 0) + 0.35, yi, val, ha="left", va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlim(0, 22)
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
    a.annotate(f"{L[mesh].split(' (')[0]}, {ranks} GPUs\n{nb} → {na} neighbours",
               (red, gain), textcoords="offset points", xytext=(8, -3), fontsize=8)
a.set_xlim(0, 55)
a.set_ylim(0, 23)
a.set_xlabel("reduction of the largest neighbour count (%)")
a.set_ylabel("GPU gain (% faster)")
a.set_title("(a) GPU gain follows the largest neighbour count")

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
b.set_xlabel("gain measured over 300 steps (%)")
b.set_ylabel("gain re-measured over 3,000 steps (%)")
b.set_title("(b) gains do not shrink over 3,000 steps")
b.plot([], [], "o", color="0.3", label="GPU")
b.plot([], [], "s", color="0.3", label="CPU")
b.legend(loc="lower right", frameon=False)
fig.subplots_adjust(wspace=0.28, left=0.07, right=0.99)
fig.savefig(f"{OUT}/fig_m11_mechanism.pdf"); fig.savefig(f"{OUT}/fig_m11_mechanism.png")
plt.close(fig)

# --------------------------------------------- fig 3: A100 vs GH200 (Findings 42/44)
ROWS = [  # label, mesh, GH200 nodes, A100 gain %, GH200 gain %
    ("CORE2 · 4 GPUs\n1 GH200 node",  "core2", 1,  8.06, 8.88),   # 26901093 min-of-3
    ("fArc · 16 GPUs\n4 GH200 nodes", "farc",  4,  3.59, 0.78),   # 26903022 median, min-of-5
    ("DARS · 64 GPUs\n16 GH200 nodes","dars", 16, 18.64, 2.68),   # 26901899 median, min-of-2
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

# ------------------------------------- fig 4: the disturbance, per candidate and field
# Every rms below is copied from the gate job's own output (job id on the row); the
# normaliser is the LARGEST rms among that gate's surviving controls (the top of the seed
# spread), so 1.0 = the largest deviation an ordinary seed re-roll produced. dars 2048 uses
# the 4 surviving controls of 26904739+26905507 (the fifth, seed 660013, is the Finding-45
# step-14 blowup); the Hilbert-renumbered CORE2 arm is compared under its own numbering and
# is not shown.
DIST = [  # label, mesh, {var: (arm_rms, ctl_top_rms)}, note
    ("DARS · 64 GPUs\nMINCONN+CONTIG+UFACTOR=30", "dars",   # 26908840, 5 controls
     {"temp": (7.343e-2, 6.489e-2), "salt": (1.788e-1, 1.940e-1), "ssh": (6.387e-3, 5.568e-3)},
     "T max 16.2 vs seeds 6.4–8.8"),
    ("DARS · 64 GPUs\nMINCONN", "dars",              # 26908840
     {"temp": (6.395e-2, 6.489e-2), "salt": (1.295e-1, 1.940e-1), "ssh": (6.199e-3, 5.568e-3)},
     "solver stops earlier"),
    ("NG5 · 64 GPUs\nMINCONN", "ng5",                # 26911630, 5 controls
     {"temp": (5.290e-2, 5.620e-2), "salt": (8.156e-2, 1.270e-1), "ssh": (3.076e-3, 2.839e-3)},
     "cause unknown"),
    ("CORE2 · 4 GPUs\nMINCONN", "core2",             # 26892982, 2 controls
     {"temp": (2.923e-2, 3.988e-2), "salt": (3.323e-2, 3.573e-2), "ssh": (4.015e-3, 7.244e-3)},
     ""),
    ("fArc · 16 GPUs\nMINCONN+CONTIG", "farc",       # 26893934, 4 controls
     {"temp": (4.499e-2, 3.374e-2), "salt": (7.255e-2, 1.158e-1), "ssh": (6.318e-3, 3.757e-3)},
     ""),
    (None, None, None, None),                       # separator GPU | CPU
    ("fArc · 2048 cores\nMt-KaHyPar", "farc",         # 26893320, 4 controls
     {"temp": (4.299e-2, 4.576e-2), "salt": (8.907e-2, 1.215e-1), "ssh": (4.354e-3, 5.445e-3)},
     ""),
    ("DARS · 2048 cores\nKaMinPar", "dars",           # 26904739 + 26905507, 4 surviving controls
     {"temp": (6.699e-2, 7.003e-2), "salt": (1.366e-1, 1.610e-1), "ssh": (7.079e-3, 7.086e-3)},
     ""),
    ("CORE2 · 512 cores\nUFACTOR=30", "core2",        # 26885353, 2 controls
     {"temp": (5.408e-2, 5.557e-2), "salt": (1.538e-1, 1.700e-1), "ssh": (6.404e-3, 6.421e-3)},
     ""),
    ("CORE2 · 864 cores\nKaMinPar", "core2",          # 26904986, 5 controls
     {"temp": (5.994e-2, 6.938e-2), "salt": (2.239e-1, 1.805e-1), "ssh": (7.047e-3, 6.514e-3)},
     ""),
]
MARK = {"temp": ("o", "T"), "salt": ("s", "S"), "ssh": ("^", "SSH")}

fig, ax = plt.subplots(figsize=(6.9, 3.9))
rows = [r for r in DIST]
y = np.arange(len(rows))[::-1]
for yi, (label, mesh, vals, note) in zip(y, rows):
    if label is None:
        ax.axhline(yi, color="0.85", linewidth=0.8)
        continue
    xmax = 0.0
    for var, (arm, top) in vals.items():
        r = arm / top
        xmax = max(xmax, r)
        filled = r > 1.0
        dy = {"temp": 0.17, "salt": 0.0, "ssh": -0.17}[var]
        ax.plot(r, yi + dy, MARK[var][0], color=C[mesh], markersize=6.0,
                markerfacecolor=C[mesh] if filled else "white",
                markeredgecolor=C[mesh], markeredgewidth=1.2, zorder=3)
    ax.text(0.02, yi, label.replace("\n", " — "), ha="left", va="center", fontsize=7.5,
            zorder=4)
    if note:
        ax.text(xmax + 0.045, yi, note, ha="left", va="center", fontsize=6.5, color="0.45")
ax.axvline(1.0, color="0.25", linewidth=1.0, linestyle="--", zorder=2)
ax.axvspan(0, 1.0, color="0.93", zorder=0)
ax.text(0.98, -0.62, "within the seed scatter", ha="right", va="bottom",
        fontsize=7, color="0.35", style="italic")
ax.text(1.02, -0.62, "largest difference between seed partitions", ha="left", va="bottom",
        fontsize=7, color="0.25", style="italic")
ax.set_yticks([])
ax.set_xlim(0, 1.85)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("20-step rms difference from the reference, divided by the largest difference between seed partitions")
handles = [plt.Line2D([], [], marker=MARK[v][0], linestyle="", color="0.3",
                      markerfacecolor="white", markeredgewidth=1.2, label=MARK[v][1])
           for v in ("temp", "salt", "ssh")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5, ncol=3)
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.02, right=0.99, bottom=0.13, top=0.99)
fig.savefig(f"{OUT}/fig_m11_disturbance.pdf"); fig.savefig(f"{OUT}/fig_m11_disturbance.png")
plt.close(fig)

print("wrote", ", ".join(f"fig_m11_{n}.pdf" for n in ("board", "mechanism", "gh200", "disturbance")))
