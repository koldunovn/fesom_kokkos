#!/usr/bin/env python3
"""Post-M5.13 re-profile (2026-05-29) — where the NG5 GPU step goes, and which lever to pull.

Fig  docs/figures/m513_reprofile_levers.png
  (a) nsys CUDA-trace three-way split of the NG5 dist_16 step (job 25237441, 8 steps, snapshots off):
      PCIe cudaMemcpy vs MPI/sync vs GPU kernels — confirms 44/38/19 on the campaign-final binary.
  (b) FESOM_STEP_PROFILE per-phase wall (job 25237442, 15 steps): the Lever-A flip targets
      (S in 13_fct, density in 1_eos, fer_w/w_i in 12_ale) highlighted; the ruled-out solver
      levers (CG in 7_ssh = 0.9 %, EVP in ice_dyn = 2.65 %) marked dead.

Data: nsys stats.txt + STEP_PROFILE run.log (both on build-cuda = M5.13 a-f+g1-uv+g1-T).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FIGS = "docs/figures"

# palette (matches plot_m513_progress.py)
GRAY, ORANGE, TEAL, GREEN, RED = "#9aa0a6", "#f4a259", "#5b8e7d", "#1f6f3f", "#b3261e"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))

# ---------------------------------------------------------------- (a) three-way split (nsys)
STEP_TRACED = 6.4188                     # nsys run.log: 8 steps -> 6.4188 s/step
seg_lbl = ["PCIe\ncudaMemcpy", "MPI / sync", "GPU kernels"]
seg_s   = [2.81, STEP_TRACED - 2.81 - 1.188, 1.188]   # 2.81 / 2.42 / 1.19
seg_col = [ORANGE, GRAY, TEAL]
left = 0.0
for s, c, lbl in zip(seg_s, seg_col, seg_lbl):
    axL.barh(0, s, left=left, color=c, edgecolor="white", height=0.55, zorder=3)
    axL.text(left + s/2, 0, f"{lbl}\n{s:.2f} s\n({100*s/STEP_TRACED:.0f} %)",
             ha="center", va="center", fontsize=10,
             color="white" if c != GRAY else "#222", fontweight="bold")
    left += s
# Lever-A pointer on the PCIe segment
axL.annotate("Lever A target\n(device residency)", xy=(2.81/2, 0.30), xytext=(2.81/2, 0.78),
             ha="center", va="bottom", fontsize=9.5, color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
axL.set_xlim(0, STEP_TRACED + 0.05); axL.set_ylim(-0.7, 1.15)
axL.set_yticks([]); axL.set_xlabel("NG5 dist_16 step  (s/step, nsys-traced = 6.42)")
axL.set_title("(a) Where the GPU step goes  —  nsys re-profile of the campaign binary")
axL.grid(axis="x", alpha=0.3, zorder=0)
# context box
axL.text(0.02, -0.55,
         "deep_copy = 11 GB/step over PCIe  →  the 2.81 s\n"
         "Ruled out (measured tiny):  CG 0.9 %  ·  EVP+ice 2.65 %  ·  kernel launches 0.2 %",
         transform=axL.get_yaxis_transform(), fontsize=8.7, color="#333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#ccc"))

# ---------------------------------------------------------------- (b) per-phase wall (STEP_PROFILE)
# (name, %loop, tag)  tag: "A"=Lever-A flip target, "X"=ruled-out lever, ""=other, "C"=future Lever C
phases = [
    ("13_fct",            18.06, "A"),   # S (x2 exch + floor + T sync) lives here
    ("3_mixing (KPP)",    13.50, ""),
    ("1b_gm",             11.17, ""),
    ("13b_trdiff",         6.16, ""),
    ("1_eos",              5.67, "A"),   # density_m_rho0
    ("smoother\n(bvfreq+blmc)", 5.07, "C"),
    ("12_ale",             5.05, "A"),   # fer_w, w_i
    ("ice_dyn (EVP)",      2.65, "X"),
    ("7_ssh (CG)",         1.60, "X"),
    ("other phases",       8.6, ""),     # velrhs+ivisc+viscfilt+ice_thermo+fct+coupling+forcing remainder
    ("rest: halos/host/MPI", 15.7, ""),  # profiler: not in any phase
]
names = [p[0] for p in phases]
vals  = [p[1] for p in phases]
tags  = [p[2] for p in phases]
tagcol = {"A": GREEN, "X": RED, "C": ORANGE, "": GRAY}
cols  = [tagcol[t] for t in tags]
y = np.arange(len(phases))[::-1]                     # top phase at top
axR.barh(y, vals, color=cols, edgecolor="white", height=0.72, zorder=3)
for yi, v, t in zip(y, vals, tags):
    axR.text(v + 0.25, yi, f"{v:.1f}%", va="center", ha="left", fontsize=8.8,
             fontweight="bold" if t in ("A", "X") else "normal")
    if t == "A":
        axR.text(0.4, yi, "flip", va="center", ha="left", fontsize=8, color="white", fontweight="bold")
    if t == "X":
        axR.text(v + 1.9, yi, "✗ dead lever", va="center", ha="left", fontsize=8, color=RED, style="italic")
axR.set_yticks(y); axR.set_yticklabels(names, fontsize=9)
axR.set_xlim(0, 21); axR.set_xlabel("% of loop  (STEP_PROFILE, 6.17 s/step)")
axR.set_title("(b) Per-phase wall  —  Lever-A targets (S → 13_fct biggest), CG/EVP ruled out")
axR.grid(axis="x", alpha=0.3, zorder=0)
leg = [Patch(fc=GREEN, label="Lever A flip target (S, density, fer_w/w_i)"),
       Patch(fc=ORANGE, label="future Lever C (smoother)"),
       Patch(fc=RED, label="ruled out (too small to matter)"),
       Patch(fc=GRAY, label="other / compute")]
axR.legend(handles=leg, loc="lower right", fontsize=8.0, framealpha=0.95)

fig.suptitle("FESOM-Kokkos post-M5.13 re-profile (NG5 dist_16, A100) — gap 1.41×, PCIe 44 % is the next lever",
             fontsize=12.5, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{FIGS}/m513_reprofile_levers.png"
fig.savefig(out, dpi=140)
print("wrote", out)
