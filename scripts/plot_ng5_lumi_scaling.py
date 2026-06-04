#!/usr/bin/env python3
"""NG5 strong-scaling on LUMI-G (MI250X / gfx90a, Kokkos HIP backend).

7.4M ocean-node mesh, 35 steps (5 warmup excluded by the internal loop timer),
I/O off, JRA55 1958, full ocean + sea-ice + GM/Redi pipeline. One MPI rank per
GCD (8 per LUMI-G node) with cray-mpich GPU-aware MPI
(MPICH_GPU_SUPPORT_ENABLED=1, libmpi_gtl_hsa); FESOM_GPU_RESIDENT device-halo
path active (see docs/PORT_HIP_LUMI.md).

Two timestep configurations:
  dt=180 s (3-min)  — captured 2026-06-04 with NSTEPS=35
  dt=240 s (4-min)  — captured 2026-06-04 with NSTEPS=35  (CG iters 84 -> 115)

The right panel reports SYPD (simulated years per wall-day) = dt / (365.25*s_per_step).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# (nodes, GCDs, s/step)
DATA_180 = [
    (2,  16,  3.3548),
    (4,  32,  1.6149),
    (8,  64,  0.8870),
    (16, 128, 0.4894),
    (32, 256, 0.2964),
]
DATA_240 = [
    (2,  16,  3.3690),
    (4,  32,  1.6372),
    (8,  64,  0.9339),
    (16, 128, 0.5012),
    (32, 256, 0.3263),
]

def unpack(D):
    n = np.array([d[0] for d in D], float)
    s = np.array([d[2] for d in D], float)
    return n, s

n180, s180 = unpack(DATA_180)
n240, s240 = unpack(DATA_240)
SYPD_180 = 180.0 / (365.25 * s180)
SYPD_240 = 240.0 / (365.25 * s240)

os.makedirs("docs/figures", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- LEFT: log-log s/step vs nodes ----
ax1.loglog(n240, s240, "o-", lw=2.0, ms=8, color="#d62728",
           label="dt = 240 s (4 min) — this work")
ax1.loglog(n180, s180, "s--", lw=1.2, ms=6, color="#d62728", alpha=0.45,
           label="dt = 180 s (3 min) — reference")
ideal = s240[0] * n240[0] / n240
ax1.loglog(n240, ideal, ":", lw=1.0, color="grey", label="ideal scaling (1/N)")
for n, s in zip(n240, s240):
    ax1.annotate(f"{s:.3f}", xy=(n, s),
                 xytext=(6, -10), textcoords="offset points", fontsize=9)
ax1.set_xlabel("LUMI-G nodes  (8 GCDs / node)")
ax1.set_ylabel("s / step")
ax1.set_title("NG5 strong scaling — wall time")
ax1.grid(True, which="both", alpha=0.3)
ax1.set_xticks(n240)
ax1.set_xticklabels([str(int(n)) for n in n240])
ax1.legend(loc="upper right", framealpha=0.9)

# ---- RIGHT: SYPD vs nodes ----
ax2.plot(n240, SYPD_240, "o-", lw=2.0, ms=8, color="#2ca02c",
         label="dt = 240 s (4 min)")
ax2.plot(n180, SYPD_180, "s--", lw=1.2, ms=6, color="#2ca02c", alpha=0.45,
         label="dt = 180 s (3 min)")
for n, y in zip(n240, SYPD_240):
    ax2.annotate(f"{y:.2f}", xy=(n, y),
                 xytext=(6, 5), textcoords="offset points", fontsize=9)
ax2.set_xscale("log")
ax2.set_xlabel("LUMI-G nodes  (8 GCDs / node)")
ax2.set_ylabel("SYPD  (simulated years per wall-day)")
ax2.set_title("NG5 throughput — SYPD")
ax2.set_xticks(n240)
ax2.set_xticklabels([str(int(n)) for n in n240])
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper left", framealpha=0.9)

fig.suptitle("FESOM2 Kokkos-HIP on LUMI-G  —  NG5 (7.4M ocean nodes), JRA55-1958",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = "docs/figures/ng5_scaling_lumi.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# ---- printed tables ----
def print_table(D, dt):
    n, s = unpack(D)
    speedup = s[0] / s
    eff = speedup / (n / n[0]) * 100.0
    sypd = dt / (365.25 * s)
    print(f"\n=== dt={dt} s ===")
    print(f"{'nodes':>5} {'ranks':>6} {'s/step':>10} {'speedup':>9} {'eff':>8} {'SYPD':>7}")
    for i, (nn, gg, ss) in enumerate(D):
        print(f"{nn:>5} {gg:>6} {ss:>10.4f} {speedup[i]:>8.2f}x {eff[i]:>7.1f}% {sypd[i]:>7.3f}")

print_table(DATA_180, 180)
print_table(DATA_240, 240)
