#!/usr/bin/env python3
"""CORE2 strong-scaling on LUMI-G (MI250X / gfx90a, Kokkos HIP backend).

127k ocean-node mesh, dt=1800, 200 steps (5 warmup excluded by the internal
loop timer), I/O off, JRA55 1958, full ocean + sea-ice + GM/Redi pipeline.
One MPI rank per GCD (8 per LUMI-G node). FESOM_GPU_RESIDENT device-halo
path active (see docs/PORT_HIP_LUMI.md).

CORE2 is small enough that the strong-scaling minimum sits around n8;
beyond that, communication overhead grows faster than per-rank work shrinks.
n32 was skipped — yesterday's run already showed ~0.110 s/step there
(saturated).

Captured 2026-06-04. Source job: jobs/job_gpu_scaling_lumi (dt=1800 default).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# (nodes, GCDs, s/step)
DATA = [
    (2,  16,  0.1129),
    (4,  32,  0.1062),
    (8,  64,  0.1037),
    (16, 128, 0.1203),
]
DT = 1800.0

n = np.array([d[0] for d in DATA], float)
g = np.array([d[1] for d in DATA], int)
s = np.array([d[2] for d in DATA], float)
speedup = s[0] / s
ideal   = n / n[0]
eff     = speedup / ideal * 100.0
sypd    = DT / (365.25 * s)

os.makedirs("docs/figures", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- LEFT: log-log s/step vs nodes ----
ax1.loglog(n, s, "o-", lw=2.0, ms=8, color="#1f77b4",
           label="CORE2 (this work)")
ideal_curve = s[0] * n[0] / n
ax1.loglog(n, ideal_curve, ":", lw=1.0, color="grey", label="ideal scaling (1/N)")
for nn, ss in zip(n, s):
    ax1.annotate(f"{ss:.4f}", xy=(nn, ss),
                 xytext=(6, -10), textcoords="offset points", fontsize=9)
ax1.set_xlabel("LUMI-G nodes  (8 GCDs / node)")
ax1.set_ylabel("s / step")
ax1.set_title("CORE2 strong scaling — wall time")
ax1.grid(True, which="both", alpha=0.3)
ax1.set_xticks(n)
ax1.set_xticklabels([str(int(x)) for x in n])
ax1.legend(loc="upper right", framealpha=0.9)

# ---- RIGHT: SYPD vs nodes ----
ax2.plot(n, sypd, "o-", lw=2.0, ms=8, color="#2ca02c", label=f"dt = {int(DT)} s")
for nn, y in zip(n, sypd):
    ax2.annotate(f"{y:.1f}", xy=(nn, y),
                 xytext=(6, 5), textcoords="offset points", fontsize=9)
ax2.set_xscale("log")
ax2.set_xlabel("LUMI-G nodes  (8 GCDs / node)")
ax2.set_ylabel("SYPD  (simulated years per wall-day)")
ax2.set_title("CORE2 throughput — SYPD")
ax2.set_xticks(n)
ax2.set_xticklabels([str(int(x)) for x in n])
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="lower right", framealpha=0.9)

fig.suptitle("FESOM2 Kokkos-HIP on LUMI-G  —  CORE2 (127k ocean nodes), JRA55-1958",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = "docs/figures/core2_scaling_lumi.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

print()
print(f"{'nodes':>5} {'ranks':>6} {'s/step':>10} {'speedup':>9} {'eff':>8} {'SYPD':>7}")
for i, (nn, gg, ss) in enumerate(DATA):
    print(f"{nn:>5} {gg:>6} {ss:>10.4f} {speedup[i]:>8.2f}x {eff[i]:>7.1f}% {sypd[i]:>7.2f}")
