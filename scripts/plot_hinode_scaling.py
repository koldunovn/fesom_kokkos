#!/usr/bin/env python3
"""M5.13 post-campaign HIGH-NODE strong-scaling — NG5 + dars on the campaign binary.
Fig: (a) GPU+CPU s/step vs nodes (strong scaling); (b) node-for-node GPU/CPU ratio vs nodes
(the per-rank-work drift: adding nodes thins the work/GPU and the ratio creeps toward the
launch/MPI floor — the data/compute-balance lesson L58 made visible).
Data: jobs/submit_m513_hinode_scaling.sh runs (2-rep means), campaign binary build-cuda
(a-f+g1-uv+g1-T). CPU is campaign-invariant (#ifdef CUDA). 32N GPU pending -> set when it lands."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- campaign-binary timings (s/step, mean of 2 reps); None = partition n/a or pending ---
# nodes : {gpu, cpu, nod2D-per-rank(k)}
NG5 = {  # 7.4M nodes, 70 levels
  4:  dict(gpu=6.12,   cpu=4.330, kpr=462),
  8:  dict(gpu=3.3066, cpu=2.2301, kpr=231),
  16: dict(gpu=1.8134, cpu=1.1710, kpr=116),
  32: dict(gpu=None,   cpu=None,  kpr=58),   # GPU dist_128 pending; NG5 has no dist_4096 CPU
}
DARS = {  # 3.16M nodes, 47 levels
  2:  dict(gpu=4.3108, cpu=2.844, kpr=395),
  4:  dict(gpu=2.3417, cpu=1.465, kpr=197),
  8:  dict(gpu=1.2643, cpu=None,  kpr=99),    # no dars dist_1024 CPU
  16: dict(gpu=0.7497, cpu=None,  kpr=49),    # no dars dist_2048 CPU
  32: dict(gpu=None,   cpu=0.2051, kpr=25),   # GPU dist_128 pending; CPU dist_4096 done
}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

def series(M, key):
    ns = sorted(n for n in M if M[n][key] is not None)
    return ns, [M[n][key] for n in ns]

# (a) strong scaling
for M, name, col in ((NG5, "NG5 (7.4M)", "#d62728"), (DARS, "dars (3.16M)", "#ff7f0e")):
    gn, gy = series(M, "gpu"); cn, cy = series(M, "cpu")
    axA.plot(gn, gy, "-o", color=col, lw=2, ms=8, label=f"{name} GPU")
    axA.plot(cn, cy, "--s", color=col, lw=1.6, ms=7, mfc="white", label=f"{name} CPU")
    # ideal 1/N guide from each mesh's first GPU point
    n0, y0 = gn[0], gy[0]; xs = np.array([n0, max(gn)])
    axA.plot(xs, y0 * n0 / xs, ":", color=col, lw=0.8, alpha=0.5)
axA.set_xscale("log", base=2); axA.set_yscale("log")
axA.set_xticks([2, 4, 8, 16, 32]); axA.set_xticklabels([2, 4, 8, 16, 32])
axA.set_xlabel("nodes"); axA.set_ylabel("s/step (campaign binary)")
axA.set_title("(a) Strong scaling — GPU strong-scales ~1.85×/doubling\n(dotted = ideal 1/N; 32N GPU not obtained — node contention)")
axA.legend(fontsize=9, ncol=2)

# (b) node-for-node ratio vs nodes (drift)
for M, name, col in ((NG5, "NG5", "#d62728"), (DARS, "dars", "#ff7f0e")):
    ns = sorted(n for n in M if M[n]["gpu"] is not None and M[n]["cpu"] is not None)
    rs = [M[n]["gpu"] / M[n]["cpu"] for n in ns]
    axB.plot(ns, rs, "-D", color=col, lw=2.2, ms=9, label=name)
    for n, r in zip(ns, rs):
        axB.annotate(f"{r:.2f}×\n{M[n]['kpr']}k/rank", (n, r), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8.5, color=col, fontweight="bold")
axB.axhline(1.0, color="k", lw=0.9, ls=":"); axB.text(2.1, 1.03, "parity", fontsize=8.5)
axB.annotate("fewer nodes = more work/GPU = nearer parity\n(more nodes thin the work → drift to the launch/MPI floor)",
             xy=(0.5, 0.06), xycoords="axes fraction", fontsize=8.5, color="#333", ha="left")
axB.set_xscale("log", base=2); axB.set_xticks([2, 4, 8, 16]); axB.set_xticklabels([2, 4, 8, 16])
axB.set_xlabel("nodes  (annotated: nod2D per rank)")
axB.set_ylabel("node-for-node GPU ÷ CPU  (× slower)")
axB.set_title("(b) The ratio drifts UP with node count\n(per-rank-work roll-off — the L58 balance lesson)")
axB.set_ylim(0.9, 1.8); axB.legend(fontsize=10)

fig.suptitle("M5.13 post-campaign high-node strong-scaling (campaign binary): the GPU is most "
             "competitive at FEW nodes / high per-rank work", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("docs/figures/scaling_hinode_m513.png", dpi=130, bbox_inches="tight")
print("wrote docs/figures/scaling_hinode_m513.png")
print("NG5 ratios:", {n: round(NG5[n]["gpu"]/NG5[n]["cpu"], 3) for n in NG5 if NG5[n]["gpu"] and NG5[n]["cpu"]})
print("dars ratios:", {n: round(DARS[n]["gpu"]/DARS[n]["cpu"], 3) for n in DARS if DARS[n]["gpu"] and DARS[n]["cpu"]})
print("GPU strong-scaling (s/step):")
print("  NG5 :", {n: NG5[n]["gpu"] for n in NG5 if NG5[n]["gpu"]})
print("  dars:", {n: DARS[n]["gpu"] for n in DARS if DARS[n]["gpu"]})
