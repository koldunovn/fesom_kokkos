#!/usr/bin/env python3
"""M5.13 NG5 device-residency campaign — progress figures (before -> after).

Fig 1  docs/figures/m513_ng5_progression.png   : NG5 dist_16 step time + node-for-node GPU/CPU ratio,
                                                  baseline -> a-f -> +g1-uv -> +g1-T (clean timings).
Fig 2  docs/figures/m513_deepcopy_proxy.png     : per-milestone full-field PCIe proxy (CORE2 dist_8
                                                  fidelity-gate deep_copy MB/step + calls/step).
(Fig 3 PCIe decomposition is plot_m513_decomp.py, once the final NG5 nsys trace lands.)

Data: docs/SCALING_NG5.md §M5.13 (clean NG5 dist_16, CPU dist_512=4.330 s/step unchanged) +
      docs/GPU_FIDELITY.md §M5.13 (per-milestone gate deep_copy).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = "docs/figures"

# ---------------------------------------------------------------- Fig 1: NG5 progression
stages   = ["baseline\n(pre-campaign)", "a–f\n(tier-1 + ALE)", "+ g1-uv\n(uv resid.)", "+ g1-T\n(final)"]
step_s   = [16.27, 10.88, 6.97, 6.12]            # clean NG5 dist_16 s/step
cpu_ref  = 4.330                                  # CPU dist_512 (4 nodes), UNCHANGED (#ifdef CUDA)
ratio    = [s / cpu_ref for s in step_s]          # node-for-node GPU/CPU  -> 3.76, 2.51, 1.61, 1.41
x = np.arange(len(stages))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))

cols = ["#9aa0a6", "#f4a259", "#5b8e7d", "#1f6f3f"]
b = axL.bar(x, step_s, color=cols, width=0.62, zorder=3)
axL.axhline(cpu_ref, ls="--", lw=1.4, color="#b3261e", zorder=2)
axL.text(len(stages)-0.5, cpu_ref+0.25, f"CPU node-for-node (dist_512) = {cpu_ref:.2f}",
         color="#b3261e", ha="right", va="bottom", fontsize=9)
for xi, s in zip(x, step_s):
    axL.text(xi, s+0.18, f"{s:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axL.set_xticks(x); axL.set_xticklabels(stages, fontsize=9)
axL.set_ylabel("NG5 dist_16 step time  (s/step, clean)")
axL.set_title("(a) NG5 (7.4 M, dist_16) GPU step:  16.27 → 6.12 s/step  (−62 %)")
axL.set_ylim(0, 17.5); axL.grid(axis="y", alpha=0.3, zorder=0)

b2 = axR.bar(x, ratio, color=cols, width=0.62, zorder=3)
axR.axhline(2.0, ls="--", lw=1.4, color="#1a73e8", zorder=2)
axR.text(0.0, 2.05, "~2× stretch target (charter)", color="#1a73e8", ha="left", va="bottom", fontsize=9)
for xi, r in zip(x, ratio):
    axR.text(xi, r+0.06, f"{r:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")
axR.set_xticks(x); axR.set_xticklabels(stages, fontsize=9)
axR.set_ylabel("node-for-node GPU/CPU  (×, lower = better)")
axR.set_title("(b) GPU/CPU gap:  3.76× → 1.41×  (past the ~2× target)")
axR.set_ylim(0, 4.2); axR.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle("M5.13 device-residency campaign — NG5 production-mesh progress (CPU unchanged: flips are #ifdef CUDA)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIGS}/m513_ng5_progression.png", dpi=130)
print("wrote", f"{FIGS}/m513_ng5_progression.png")

# ---------------------------------------------------------------- Fig 2: per-milestone PCIe proxy
ms    = ["a\ncfl_z", "b\nEOS", "c\nGM-quartet", "d\nuv_rhsAB", "e\nALE w/w_e", "f\ncommit", "g1-uv\nuv", "g1-T\nT"]
mb    = [1067.6, 1020.4, 857.2, 810.7, 746.5, 641.4, 342.3, 277.3]   # CORE2 dist_8 deep_copy MB/step
calls = [207.7, 199.7, 188.7, 186.7, 175.8, 163.8, 150.9, 139.9]     #               deep_copy calls/step
xm = np.arange(len(ms))

fig2, axM = plt.subplots(figsize=(12.5, 5.2))
bars = axM.bar(xm, mb, color="#c0392b", width=0.6, zorder=3, label="full-field PCIe deep_copy  (MB/step)")
for xi, m in zip(xm, mb):
    axM.text(xi, m+12, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
axM.set_xticks(xm); axM.set_xticklabels(ms, fontsize=9)
axM.set_ylabel("CORE2 dist_8 PCIe deep_copy  (MB/step)", color="#c0392b")
axM.tick_params(axis="y", labelcolor="#c0392b")
axM.set_ylim(0, 1180); axM.grid(axis="y", alpha=0.3, zorder=0)

axC = axM.twinx()
axC.plot(xm, calls, "o-", color="#1f3a93", lw=2, ms=6, zorder=4, label="deep_copy calls/step")
for xi, c in zip(xm, calls):
    axC.text(xi, c+3, f"{c:.0f}", ha="center", va="bottom", fontsize=8, color="#1f3a93")
axC.set_ylabel("deep_copy calls/step", color="#1f3a93")
axC.tick_params(axis="y", labelcolor="#1f3a93"); axC.set_ylim(0, 230)

# annotate the two biggest single wins
axM.annotate("GM quartet\n(−163 MB)", xy=(2, 857), xytext=(2, 980),
             ha="center", fontsize=8.5, arrowprops=dict(arrowstyle="->", color="k"))
axM.annotate("uv full residency\n(−299 MB, largest)", xy=(6, 342), xytext=(5.0, 520),
             ha="center", fontsize=8.5, arrowprops=dict(arrowstyle="->", color="k"))

axM.set_title("M5.13 per-milestone PCIe proxy (CORE2 dist_8, fidelity-gate, FESOM_STEP_PROFILE=1):\n"
              "full-field host↔device deep_copy  1068 → 277 MB/step (−74 %),  207.7 → 139.9 calls/step",
              fontsize=11, fontweight="bold")
fig2.tight_layout()
fig2.savefig(f"{FIGS}/m513_deepcopy_proxy.png", dpi=130)
print("wrote", f"{FIGS}/m513_deepcopy_proxy.png")

# ---------------------------------------------------------------- Fig 3: NG5 PCIe decomposition (nsys)
# nsys CUDA traces, NG5 dist_16 rank0, 8 steps. baseline = pre-campaign (job 25227869, charter);
# a-f = ckpt-2 (m513_ckpt/fesom_port_af); final = g1-T (m513_ckpt/fesom_port_final). GPU-compute is
# campaign-invariant (Approach B = same kernels) so ~1.19 s/step throughout; MPI/sync = step - PCIe - GPU.
dstage = ["baseline\n(pre-campaign)", "a–f", "final\n(+g1-uv +g1-T)"]
nstep  = [16.94, 11.17, 6.43]     # nsys-instrumented step (s/step)
pcie   = [12.74, 7.48, 2.83]      # nsys H2D+D2H cudaMemcpy
gpu    = [1.19, 1.19, 1.19]       # nsys kernel sum (compute unchanged by data-movement flips)
mpi    = [s - p - g for s, p, g in zip(nstep, pcie, gpu)]   # MPI / cudaDeviceSynchronize / host
xd = np.arange(len(dstage))

fig3, ax3 = plt.subplots(figsize=(8.6, 5.6))
ax3.bar(xd, pcie, width=0.55, color="#c0392b", zorder=3, label="PCIe cudaMemcpy (host↔device)")
ax3.bar(xd, mpi,  width=0.55, bottom=pcie, color="#f4a259", zorder=3, label="MPI / sync / host")
ax3.bar(xd, gpu,  width=0.55, bottom=[p+m for p, m in zip(pcie, mpi)], color="#1f6f3f", zorder=3,
        label="GPU compute (kernels)")
for xi, p, s in zip(xd, pcie, nstep):
    ax3.text(xi, p/2, f"PCIe\n{p:.2f} s\n({100*p/s:.0f}%)", ha="center", va="center",
             color="white", fontsize=9, fontweight="bold")
for xi, s, g in zip(xd, nstep, gpu):
    ax3.text(xi, s-g/2, f"GPU {100*g/s:.0f}%", ha="center", va="center", color="white", fontsize=8)
    ax3.text(xi, s+0.2, f"{s:.2f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax3.set_xticks(xd); ax3.set_xticklabels(dstage, fontsize=9.5)
ax3.set_ylabel("NG5 dist_16 step  (s/step, nsys-traced)")
ax3.set_ylim(0, 18.5); ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.legend(loc="upper right", fontsize=9)
ax3.set_title("M5.13: the NG5 GPU is no longer PCIe-starved\n"
              "PCIe cudaMemcpy 12.74 → 2.83 s/step (−78%); PCIe share 75% → 44%; compute now 7% → 19%",
              fontsize=11, fontweight="bold")
fig3.tight_layout()
fig3.savefig(f"{FIGS}/m513_ng5_pcie_decomp.png", dpi=130)
print("wrote", f"{FIGS}/m513_ng5_pcie_decomp.png")
