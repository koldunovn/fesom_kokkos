#!/usr/bin/env python3
"""FESOM-Kokkos GPU-vs-CPU strong-scaling across CORE2 / farc / dars / NG5.
All ratios are Kokkos-CUDA (A100x4/node) vs Kokkos-Serial (128c/node) — apples-to-apples
(same codebase). The published CORE2/farc 8.9x/5.5x used the faster C-port CPU (overlaid).
Generates docs/figures/scaling_{overview,meshsize_trend}.png."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

# ---- data (s/step, mean of reps) -------------------------------------------------
# mesh: nodes(M), gpu {nodes:s/step}, cpu {nodes:s/step}, cport-4node-s/step (or None)
M = {
 "CORE2": dict(nodes=0.127, col="#1f77b4",
               gpu={1:0.8617, 2:0.4741, 4:0.2937},
               cpu={1:0.1856, 2:0.0946, 4:0.0555}, cport4=0.0329),
 "farc":  dict(nodes=0.638, col="#2ca02c",
               gpu={1:4.0193, 2:2.1225, 4:1.1580},
               cpu={1:0.8713, 2:0.4434, 4:0.2281}, cport4=0.2067),
 "dars":  dict(nodes=3.16,  col="#ff7f0e",
               gpu={4:6.003, 8:3.095},
               cpu={2:2.844, 4:1.465}, cport4=None),
 "NG5":   dict(nodes=7.40,  col="#d62728",
               gpu={2:30.39, 4:16.27, 8:8.698, 16:4.514},
               cpu={4:4.330, 8:2.239, 16:1.171}, cport4=None),
}
ORDER = ["CORE2","farc","dars","NG5"]
POWER = 2160/560.0   # GPU node W / CPU node W = 3.86

plt.rcParams.update({"font.size":11, "axes.grid":True, "grid.alpha":0.3,
                     "figure.dpi":130, "savefig.dpi":150})
os.makedirs("docs/figures", exist_ok=True)

# ============================ Figure 1: 2x2 overview ============================
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (a) strong scaling: s/step vs nodes (log-log) ----------------------------------
a = ax[0,0]
for m in ORDER:
    d = M[m]
    gn = sorted(d["gpu"]); a.plot(gn, [d["gpu"][n] for n in gn], "-o", color=d["col"], label=f"{m} GPU")
    cn = sorted(d["cpu"]); a.plot(cn, [d["cpu"][n] for n in cn], "--s", color=d["col"], mfc="white", label=f"{m} CPU")
# ideal 1/N guides anchored at each mesh's first GPU point
for m in ORDER:
    d=M[m]; gn=sorted(d["gpu"]); n0=gn[0]; y0=d["gpu"][n0]
    xs=np.array([n0, max(gn)]); a.plot(xs, y0*n0/xs, ":", color="grey", lw=0.8, alpha=0.5)
a.set_xscale("log",base=2); a.set_yscale("log"); a.set_xticks([1,2,4,8,16]); a.set_xticklabels([1,2,4,8,16])
a.set_xlabel("nodes"); a.set_ylabel("s/step"); a.set_title("(a) Strong scaling (log-log; dotted = ideal 1/N)")
a.legend(fontsize=7.5, ncol=2)

# (b) node-for-node GPU/CPU ratio vs nodes ---------------------------------------
b = ax[0,1]
for m in ORDER:
    d=M[m]; common=sorted(set(d["gpu"])&set(d["cpu"]))
    if common:
        b.plot(common, [d["gpu"][n]/d["cpu"][n] for n in common], "-o", color=d["col"], label=m)
b.axhline(1.0, color="k", lw=0.8); b.text(1.05,1.05,"parity",fontsize=8)
b.set_xscale("log",base=2); b.set_xticks([1,2,4,8,16]); b.set_xticklabels([1,2,4,8,16])
b.set_xlabel("nodes"); b.set_ylabel("GPU / CPU  (× slower per node)")
b.set_title("(b) Node-for-node ratio (Kokkos CUDA / Serial)"); b.set_ylim(0,6); b.legend(fontsize=9)

# (c) THE TREND: 4-node ratio vs mesh size ---------------------------------------
c = ax[1,0]
xs=[M[m]["nodes"] for m in ORDER]
rk=[M[m]["gpu"][4]/M[m]["cpu"][4] for m in ORDER]   # all have a 4-node point
c.plot(xs, rk, "-o", color="black", lw=2, ms=8, label="vs Kokkos-Serial CPU (consistent)")
for m,x,r in zip(ORDER,xs,rk): c.annotate(f"{m}\n{r:.1f}×", (x,r), textcoords="offset points", xytext=(6,8), fontsize=9)
cp=[(M[m]["nodes"], M[m]["gpu"][4]/M[m]["cport4"]) for m in ORDER if M[m]["cport4"]]
c.plot([p[0] for p in cp],[p[1] for p in cp], "^--", color="grey", label="vs C-port CPU (CORE2/farc only)")
c.axhline(3.6, color="red", ls=":", lw=1); c.text(0.13, 2.9, "~3.6–3.8× : Levante-A100 floor", color="red", fontsize=9)
c.set_xscale("log"); c.set_xlabel("mesh size (million nodes)"); c.set_ylabel("GPU/CPU per node (4 nodes)")
c.set_title("(c) The mesh-size lever ASYMPTOTES"); c.set_ylim(0,10); c.legend(fontsize=8.5)
c.set_xticks([0.127,0.638,3.16,7.40]); c.set_xticklabels(["0.13","0.64","3.16","7.40"])

# (d) per-watt vs mesh size ------------------------------------------------------
e = ax[1,1]
pw=[r*POWER for r in rk]
e.plot(xs, pw, "-o", color="purple", lw=2, ms=8)
for m,x,p in zip(ORDER,xs,pw): e.annotate(f"1/{p:.0f}", (x,p), textcoords="offset points", xytext=(6,6), fontsize=9)
e.set_xscale("log"); e.set_xlabel("mesh size (million nodes)"); e.set_ylabel("CPU-node energy advantage (× per Wh)")
e.set_title(f"(d) Per-watt (GPU node {2160} W / CPU {560} W = {POWER:.2f}×)")
e.set_xticks([0.127,0.638,3.16,7.40]); e.set_xticklabels(["0.13","0.64","3.16","7.40"]); e.set_ylim(0,24)

fig.suptitle("FESOM2 C++/Kokkos — GPU (A100) vs CPU (EPYC) strong-scaling, CORE2 → farc → dars → NG5", fontsize=13, y=0.995)
fig.tight_layout(rect=[0,0,1,0.98]); fig.savefig("docs/figures/scaling_overview.png", bbox_inches="tight"); plt.close(fig)

# ============================ Figure 2: headline trend ============================
fig2, g = plt.subplots(figsize=(8,5.5))
g.plot(xs, rk, "-o", color="black", lw=2.5, ms=10, label="GPU / Kokkos-Serial CPU")
for m,x,r in zip(ORDER,xs,rk): g.annotate(f"{m} ({M[m]['nodes']:.2g}M)\n{r:.1f}×", (x,r), textcoords="offset points", xytext=(8,10), fontsize=10, fontweight="bold")
g.plot([p[0] for p in cp],[p[1] for p in cp], "^--", color="grey", ms=8, label="GPU / C-port CPU (fastest CPU)")
g.axhspan(3.4,3.8, color="red", alpha=0.08); g.axhline(3.6, color="red", ls=":", lw=1.2)
g.text(0.115, 2.85, "≈3.6–3.8× : Levante-A100 floor for full-physics ocean+ice", color="red", fontsize=9)
g.annotate("", xy=(7.4,3.76), xytext=(0.127,5.29), arrowprops=dict(arrowstyle="->", color="black", alpha=0.3))
g.set_xscale("log"); g.set_xlabel("mesh size (million nodes)", fontsize=11)
g.set_ylabel("GPU node ÷ CPU node  (× slower, 4 nodes)", fontsize=11)
g.set_title("The mesh-size lever asymptotes: bigger meshes feed the GPU but the gap bottoms out", fontsize=11)
g.set_ylim(0,10); g.set_xticks([0.127,0.638,3.16,7.40]); g.set_xticklabels(["CORE2\n0.13","farc\n0.64","dars\n3.16","NG5\n7.4"])
g.legend(fontsize=10, loc="upper right")
fig2.tight_layout(); fig2.savefig("docs/figures/scaling_meshsize_trend.png", bbox_inches="tight"); plt.close(fig2)

print("wrote docs/figures/scaling_overview.png  +  docs/figures/scaling_meshsize_trend.png")
print("4-node Kokkos-CUDA/Serial ratios:", {m:round(rk[i],2) for i,m in enumerate(ORDER)})
print("per-watt (CPU advantage):", {m:round(pw[i],1) for i,m in enumerate(ORDER)})
