#!/usr/bin/env python3
"""NG5 strong-scaling on MN5 ACC (H100 / sm_90, Kokkos CUDA backend).

7.4M ocean-node mesh, 35 steps (5 warmup excluded by the internal loop timer),
I/O off, JRA55 1958, full ocean + sea-ice + GM/Redi pipeline. One MPI rank per
H100 (4 per MN5 node) with HPC-X / UCX CUDA-aware MPI
(OMPI_MCA_pml=ucx, UCX_TLS includes cuda_copy/cuda_ipc); FESOM_GPU_RESIDENT
device-halo path active (see memory project_mn5_cuda_build).

Captured 2026-06-05, sweep tag n2..n32 → dist_{8,16,32,64,128}. The n2
(dist_8, 8-rank NG5) job ABORTED mid-loop with std::bad_alloc-like throw on
rank 7 — too much per-rank load at this mesh × few-ranks combination. Plotting
the four good points (n4..n32). The LUMI dt=180 curve is overlaid as the
direct H100-vs-MI250X comparison (same dist_N partitions ⇒ same per-rank load).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# MN5 ACC, H100, dt=240 s — (mn5_nodes, h100_count, s/step). Captured 2026-06-05.
# Sweep tag n2_dt240..n32_dt240 → dist_{8,16,32,64,128}. n2 (dist_8, 8-rank NG5)
# ABORTED mid-loop with std::bad_alloc-class throw on rank 7 — too much per-rank
# load at this mesh × few-ranks combination; same failure as the dt=180 sweep.
DATA_MN5 = [
    # (2,   8,   FAILED),   # dist_8 aborted mid-loop
    (4,   16,  1.1493),
    (8,   32,  0.6492),
    (16,  64,  0.3533),
    (32,  128, 0.2564),
]

# LUMI-G, MI250X (gfx90a), dt=240 s, from scripts/plot_ng5_lumi_scaling.py
# (DATA_240). Same NG5 mesh + same dist_N partitions ⇒ per-rank load matches at
# equal GPU count (= apples-to-apples H100 vs MI250X-GCD on left panel x-axis).
DATA_LUMI = [
    (2,  16,  3.3690),
    (4,  32,  1.6372),
    (8,  64,  0.9339),
    (16, 128, 0.5012),
    (32, 256, 0.3263),
]
DT_S = 240.0

def unpack(D):
    n  = np.array([d[0] for d in D], float)   # node count
    g  = np.array([d[1] for d in D], float)   # GPU/GCD count
    s  = np.array([d[2] for d in D], float)   # s/step
    return n, g, s

n_mn5,  g_mn5,  s_mn5  = unpack(DATA_MN5)
n_lumi, g_lumi, s_lumi = unpack(DATA_LUMI)
SYPD_MN5  = DT_S / (365.25 * s_mn5)
SYPD_LUMI = DT_S / (365.25 * s_lumi)

os.makedirs("docs/figures", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- LEFT: log-log s/step vs GPU count (apples-to-apples per-rank load) ----
ax1.loglog(g_mn5,  s_mn5,  "o-",  lw=2.0, ms=8, color="#1f77b4",
           label="MN5 — H100 (4/node, this work)")
ax1.loglog(g_lumi, s_lumi, "s--", lw=1.2, ms=6, color="#d62728", alpha=0.6,
           label="LUMI-G — MI250X GCD (8/node)")
ideal = s_mn5[0] * g_mn5[0] / g_mn5
ax1.loglog(g_mn5, ideal, ":", lw=1.0, color="grey", label="ideal scaling (1/N)")
for g, s in zip(g_mn5, s_mn5):
    ax1.annotate(f"{s:.3f}", xy=(g, s),
                 xytext=(6, -10), textcoords="offset points", fontsize=9)
ax1.set_xlabel("GPU count  (1 rank per device)")
ax1.set_ylabel("s / step")
ax1.set_title("NG5 strong scaling — wall time")
ax1.grid(True, which="both", alpha=0.3)
# x-ticks: union of both sweeps' GPU counts, sorted
ticks = sorted(set(int(x) for x in np.concatenate([g_mn5, g_lumi])))
ax1.set_xticks(ticks)
ax1.set_xticklabels([str(t) for t in ticks])
ax1.legend(loc="upper right", framealpha=0.9)

# ---- RIGHT: SYPD vs GPU count ----
ax2.plot(g_mn5,  SYPD_MN5,  "o-",  lw=2.0, ms=8, color="#2ca02c",
         label="MN5 — H100")
ax2.plot(g_lumi, SYPD_LUMI, "s--", lw=1.2, ms=6, color="#9467bd", alpha=0.7,
         label="LUMI-G — MI250X GCD")
for g, y in zip(g_mn5, SYPD_MN5):
    ax2.annotate(f"{y:.2f}", xy=(g, y),
                 xytext=(6, 5), textcoords="offset points", fontsize=9)
ax2.set_xscale("log")
ax2.set_xlabel("GPU count  (1 rank per device)")
ax2.set_ylabel("SYPD  (simulated years per wall-day)")
ax2.set_title("NG5 throughput — SYPD")
ax2.set_xticks(ticks)
ax2.set_xticklabels([str(t) for t in ticks])
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper left", framealpha=0.9)

fig.suptitle(f"FESOM2 Kokkos-CUDA on MN5 ACC  —  NG5 (7.4M nodes), JRA55-1958, dt={int(DT_S)} s",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = "docs/figures/ng5_scaling_mn5.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

def print_table(D, label):
    n, g, s = unpack(D)
    speedup = s[0] / s
    eff = speedup / (g / g[0]) * 100.0
    sypd = DT_S / (365.25 * s)
    print(f"\n=== {label}  (dt={int(DT_S)} s) ===")
    print(f"{'nodes':>5} {'GPUs':>5} {'s/step':>10} {'speedup':>9} {'eff':>8} {'SYPD':>7}")
    for i, (nn, gg, ss) in enumerate(D):
        print(f"{nn:>5} {gg:>5} {ss:>10.4f} {speedup[i]:>8.2f}x {eff[i]:>7.1f}% {sypd[i]:>7.3f}")

print_table(DATA_MN5,  "MN5 ACC — H100")
print_table(DATA_LUMI, "LUMI-G — MI250X")
