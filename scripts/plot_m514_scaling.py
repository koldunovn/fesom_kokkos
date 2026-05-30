#!/usr/bin/env python3
"""M5.14 consolidated strong-scaling + node-for-node parity figure.

The landed scaling sweep (NG5 7.4M/70lvl + dars 3.16M/47lvl, GPU A100x4/node vs
CPU Serial 128c/node, 2/4/8/16(/32) nodes, 30 timed steps, 2-rep means) on the
M5.13 campaign binary, overlaid with the M5.14 parity point (4N NG5 GPU = 3.805,
the Lever-A binary that crossed node-for-node parity).

All s/step values verified from the run-dir loop-timing logs (2026-05-29/30):
  NG5  GPU m513 : ng5_gpu_n{4,8,16}_m513 / ng5_prof_m513   (4N=6.118)
  NG5  GPU m514 : ng5prof.25238841.out                     (4N=3.805)
  NG5  CPU      : ng5_cpu_n{4(m514),8,16}_m513             (4N=4.327)
  dars GPU m513 : dars_gpu_n{2,4,8,16}_m513
  dars CPU      : dars_cpu_n{2,4}, dars_cpu_n32_m513
Panel (a): strong scaling (s/step vs nodes).  Panel (b): node-for-node GPU/CPU
ratio vs nodes — the M5.13 sweep sits at ~1.4-1.6x; M5.14 Lever-A drops NG5 4N to
0.879x, *below* parity (the GPU is faster than the CPU node-for-node)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- verified s/step (mean of 2 reps); None = partition n/a or job cancelled ---
NG5 = {   # 7.4M nodes, 70 levels
  4:  dict(gpu=6.118, gpu514=3.805, cpu=4.327, kpr=462),
  8:  dict(gpu=3.307, gpu514=None,  cpu=2.230, kpr=231),
  16: dict(gpu=1.813, gpu514=None,  cpu=1.171, kpr=116),
}
DARS = {  # 3.16M nodes, 47 levels
  2:  dict(gpu=4.310, cpu=2.845, kpr=395),
  4:  dict(gpu=2.342, cpu=1.465, kpr=197),
  8:  dict(gpu=1.264, cpu=None,  kpr=99),
  16: dict(gpu=0.750, cpu=None,  kpr=49),
  32: dict(gpu=None,  cpu=0.205, kpr=25),
}
NG5_RED, DARS_OR, GOLD = "#d62728", "#ff7f0e", "#e6a000"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))

def ser(M, key):
    ns = sorted(n for n in M if M[n].get(key) is not None)
    return ns, [M[n][key] for n in ns]

# ---------- (a) strong scaling ----------
for M, name, col in ((NG5, "NG5 (7.4M, 70lvl)", NG5_RED), (DARS, "dars (3.16M, 47lvl)", DARS_OR)):
    gn, gy = ser(M, "gpu"); cn, cy = ser(M, "cpu")
    axA.plot(gn, gy, "-o", color=col, lw=2, ms=8, label=f"{name}  GPU")
    axA.plot(cn, cy, "--s", color=col, lw=1.6, ms=7, mfc="white", label=f"{name}  CPU")
    n0, y0 = gn[0], gy[0]; xs = np.array([n0, max(gn)])
    axA.plot(xs, y0 * n0 / xs, ":", color=col, lw=0.9, alpha=0.45)  # ideal 1/N
# M5.14 parity point (NG5 4N GPU)
axA.plot(4, NG5[4]["gpu514"], "*", color=GOLD, ms=22, mec="k", mew=0.8, zorder=6,
         label="NG5 GPU  M5.14 (Lever A)")
axA.annotate("3.81 s/step\n(< CPU 4.33)", (4, NG5[4]["gpu514"]), textcoords="offset points",
             xytext=(12, -2), fontsize=8.5, color=GOLD, fontweight="bold")
axA.set_xscale("log", base=2); axA.set_yscale("log")
axA.set_xticks([2, 4, 8, 16, 32]); axA.set_xticklabels([2, 4, 8, 16, 32])
axA.set_xlabel("nodes (4 A100 GPU / 128 EPYC cores per node)")
axA.set_ylabel("s / step")
axA.set_title("(a) Strong scaling — GPU ~1.85×/doubling (92% eff.)\n"
              "dotted = ideal 1/N · 32N GPU cancelled (node contention)")
axA.legend(fontsize=8.5, ncol=1, loc="upper right")

# ---------- (b) node-for-node GPU/CPU ratio ----------
axB.axhspan(0.0, 1.0, color="#2ca02c", alpha=0.07)          # GPU-faster band
axB.axhline(1.0, color="k", lw=1.0, ls=":")
axB.text(15.3, 1.012, "parity (GPU = CPU)", fontsize=8.5, ha="right")
axB.text(2.05, 0.915, "GPU faster than CPU  ▼", fontsize=8.5, color="#2ca02c", fontweight="bold")
for M, name, col in ((NG5, "NG5", NG5_RED), (DARS, "dars", DARS_OR)):
    ns = sorted(n for n in M if M[n]["gpu"] is not None and M[n]["cpu"] is not None)
    rs = [M[n]["gpu"] / M[n]["cpu"] for n in ns]
    axB.plot(ns, rs, "-D", color=col, lw=2.2, ms=9, label=f"{name}  (M5.13 sweep)")
    for n, r in zip(ns, rs):
        axB.annotate(f"{r:.2f}×\n{M[n]['kpr']}k/rank", (n, r), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8, color=col, fontweight="bold")
# M5.14 parity point + the Lever-A drop arrow
r514 = NG5[4]["gpu514"] / NG5[4]["cpu"]
axB.annotate("", xy=(4, r514), xytext=(4, NG5[4]["gpu"] / NG5[4]["cpu"]),
             arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.4))
axB.plot(4, r514, "*", color=GOLD, ms=24, mec="k", mew=0.8, zorder=6,
         label="NG5 4N  M5.14 (Lever A)")
axB.annotate(f"{r514:.2f}×\nPARITY\nCROSSED", (4, r514), textcoords="offset points",
             xytext=(13, -6), fontsize=9, color=GOLD, fontweight="bold")
axB.set_xscale("log", base=2); axB.set_xticks([2, 4, 8, 16]); axB.set_xticklabels([2, 4, 8, 16])
axB.set_xlabel("nodes  (annotated: nod2D per rank — GPU feed rate)")
axB.set_ylabel("node-for-node  GPU ÷ CPU   (>1 = GPU slower)")
axB.set_title("(b) Node-for-node ratio — Lever A (S/ρ/fer_w/w_i residency)\n"
              "drops NG5 4N from 1.41× to 0.88× (GPU ~14% faster)")
axB.set_ylim(0.8, 1.72); axB.set_xlim(1.8, 18); axB.legend(fontsize=8.5, loc="upper left")

fig.suptitle("FESOM-Kokkos GPU scaling — the device-residency campaign reaches node-for-node parity "
             "(NG5, 4 nodes)", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "docs/figures/scaling_m514_parity.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

# ---------- console analysis ----------
def eff(a, b): return a / b, (a / b) / 2 * 100   # ratio, %/doubling
print("\n=== GPU strong-scaling (s/step → ratio, %eff per doubling) ===")
for M, nm in ((NG5, "NG5"), (DARS, "dars")):
    gn, gy = ser(M, "gpu")
    print(f"  {nm}:", {n: y for n, y in zip(gn, gy)})
    for i in range(len(gn) - 1):
        if gn[i+1] == 2 * gn[i]:
            r, e = eff(gy[i], gy[i+1]); print(f"     {gn[i]}→{gn[i+1]}N: {r:.3f}× ({e:.0f}%)")
print("\n=== node-for-node GPU/CPU ratio ===")
for M, nm in ((NG5, "NG5"), (DARS, "dars")):
    for n in sorted(M):
        if M[n]["gpu"] and M[n]["cpu"]:
            print(f"  {nm} {n}N: {M[n]['gpu']/M[n]['cpu']:.3f}×  ({M[n]['kpr']}k/rank)")
print(f"\n  NG5 4N M5.14 (Lever A): {r514:.3f}×  <-- PARITY CROSSED (GPU {(1-r514)*100:.0f}% faster)")
