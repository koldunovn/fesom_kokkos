#!/usr/bin/env python3
"""GPU step composition: CORE2 dist_8 (16k nod2D/rank, nod2D-halo-latency-bound) vs NG5 dist_16
(462k nod2D/rank, nod3D-PCIe-bandwidth-bound). Per-phase % of the timed step (FESOM_STEP_PROFILE=1).
CORE2 from docs/PROFILE_M59.md R1c; NG5 from the 2026-05-29 dist_16 profile.
NB: these are phase WALL %shares (kernel+PCIe+MPI). nsys shows the GPU computes only ~7% of the
NG5 step; the phases grew because their full-field PCIe halo syncs grew, NOT their compute.
-> docs/figures/profile_regime_shift.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# (label, CORE2 %loop, NG5 %loop, regime)
PH = [
    ("FCT advection (13)",    17.43, 22.76, "compute"),
    ("GM/Redi (1b)",           9.79, 11.16, "compute"),
    ("ALE (12)",               6.61,  9.10, "compute"),
    ("KPP mixing (3)",         6.76,  8.27, "compute"),
    ("tracer diffusion (13b)", 5.68,  7.20, "compute"),
    ("momentum RHS (4)",       5.24,  7.15, "compute"),
    ("impl. vert. visc (6)",   4.38,  5.94, "compute"),
    ("EOS / density (1)",      3.24,  4.50, "compute"),
    ("SSH solve: CG+halo (7)",10.92,  4.48, "halo"),
    ("sea-ice dyn: EVP halo", 11.26,  3.20, "halo"),
    ("visc filter (5)",        1.53,  1.94, "compute"),
]
PH.sort(key=lambda r: r[2], reverse=True)          # order by NG5 %loop
labels = [r[0] for r in PH]; c2=[r[1] for r in PH]; ng=[r[2] for r in PH]; typ=[r[3] for r in PH]
COMP="#2a9d8f"; HALO="#e63946"                     # compute teal, halo red
col=[COMP if t=="compute" else HALO for t in typ]

plt.rcParams.update({"font.size":11,"savefig.dpi":160,"figure.dpi":130})
fig, ax = plt.subplots(figsize=(11,7))
y=np.arange(len(labels)); h=0.38
# CORE2 = lighter/hatched (upper bar), NG5 = solid (lower bar), colored by regime
b1=ax.barh(y+h/2, c2, height=h, color=col, alpha=0.45, edgecolor="grey", hatch="///", label="_c2")
b2=ax.barh(y-h/2, ng, height=h, color=col, alpha=1.0, edgecolor="black", lw=0.6, label="_ng")
for yi,v in zip(y+h/2,c2): ax.text(v+0.2,yi,f"{v:.1f}",va="center",fontsize=8,color="grey")
for yi,v in zip(y-h/2,ng): ax.text(v+0.2,yi,f"{v:.1f}",va="center",fontsize=8.5,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
# color the y tick labels by regime
for tl,t in zip(ax.get_yticklabels(),typ): tl.set_color(COMP if t=="compute" else HALO); tl.set_fontweight("bold" if t=="halo" else "normal")
ax.set_xlabel("% of the timed GPU step"); ax.set_xlim(0,25)
ax.set_title("Dominant phases shift nod2D-halo (CORE2) → nod3D (NG5) — both data-movement-bound\n"
             "(phase WALL %; nsys: NG5 GPU computes only ~7% of the step, ~75% is PCIe memcpy)",
             fontsize=11.5, pad=12)
ax.grid(axis="x", alpha=0.3)

# arrows on the two halo phases (the collapse) + a couple compute (the growth)
for lab,dirn in [("SSH solve: CG+halo (7)","dn"),("sea-ice dyn: EVP halo","dn"),
                 ("FCT advection (13)","up"),("ALE (12)","up")]:
    i=labels.index(lab)
    ax.annotate("", xy=(ng[i]+ (0.0 if dirn=="up" else 0.0), y[i]-h/2),
                xytext=(c2[i], y[i]+h/2),
                arrowprops=dict(arrowstyle="-|>", color=("#b00" if dirn=="dn" else "#066"),
                                lw=1.4, alpha=0.7, connectionstyle="arc3,rad=0.0"))

# regime-sum callout box
comp_c2=sum(r[1] for r in PH if r[3]=="compute"); comp_ng=sum(r[2] for r in PH if r[3]=="compute")
halo_c2=sum(r[1] for r in PH if r[3]=="halo");    halo_ng=sum(r[2] for r in PH if r[3]=="halo")
txt=(f"nod3D phases (FCT/GM/ALE…):  {comp_c2:.0f}%  →  {comp_ng:.0f}%   (grow)\n"
     f"nod2D-halo (SSH+EVP):  {halo_c2:.0f}%  →  {halo_ng:.0f}%   (collapse)\n"
     f"CG share:  5.7%  →  0.3%\n"
     f"step wall:  0.52 s  →  16.3 s  (31×)\n"
     f"nsys: GPU compute only 7% — the rest is PCIe+MPI")
ax.text(0.985,0.045,txt,transform=ax.transAxes,ha="right",va="bottom",fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5",fc="#fff8e1",ec="#d4a72c"))

leg=[Patch(fc="grey",alpha=0.45,hatch="///",ec="grey",label="CORE2 dist_8  (16k nod2D/rank)"),
     Patch(fc="grey",alpha=1.0,ec="black",label="NG5 dist_16  (462k nod2D/rank)"),
     Patch(fc=COMP,label="nod3D phase (PCIe-bandwidth-bound at NG5)"),
     Patch(fc=HALO,label="nod2D-halo phase (latency-bound at CORE2)")]
ax.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0,0.30), fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig("docs/figures/profile_regime_shift.png", bbox_inches="tight")
print("wrote docs/figures/profile_regime_shift.png")
print(f"compute {comp_c2:.1f}->{comp_ng:.1f} ; halo {halo_c2:.1f}->{halo_ng:.1f}")
