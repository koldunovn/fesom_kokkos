#!/usr/bin/env python3
"""NG5 CPU vs GPU strong-scaling on LUMI.

Apples-to-apples on the LUMI-node axis:
  CPU: LUMI-C, AMD EPYC 7763, 128 MPI ranks/node (1 rank/core, no OpenMP).
       Kokkos Serial backend, build-cpu (PrgEnv-gnu, gcc-native).
  GPU: LUMI-G, AMD MI250X, 8 MPI ranks/node (1 rank/GCD).
       Kokkos HIP backend, build-hip (PrgEnv-amd, ROCm 6.3.4), GPU-aware MPI.

NG5 mesh (7.4M ocean nodes), dt=240 s, 35 steps (5 warmup excluded), I/O off,
JRA55 1958, full ocean + sea-ice + GM/Redi pipeline. Same binary code on both
sides (FESOM_GPU_RESIDENT==0 on CPU → host-staged halo; ==1 on GPU →
device-pointer halo).

NG5 CPU n1 (1 LUMI-C node, 128 ranks) OOM-killed: 256 GB / 128 ranks = 2 GB/rank
isn't enough for 7.4M-node mesh (~58k own + halo per rank → ~3 GB needed).
n2 is the practical CPU floor.

Captured 2026-06-04.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# (LUMI nodes, MPI ranks, s/step)
CPU = [
    (2, 256,   8.4047),
    (4, 512,   4.2788),
    (8, 1024,  2.2498),
    (16, 2048, 1.1321),
    # (32, 4096, BLOCKED): 4096 ranks saturates the Cassini NIC resource quota
    # (cxil_map: write error). FI_CXI_RX_MATCH_MODE=software didn't help.
    # Documented in LUMI user support; out of scope for this port.
]
GPU = [
    (2,  16,  3.3690),
    (4,  32,  1.6372),
    (8,  64,  0.9339),
    (16, 128, 0.5012),
    (32, 256, 0.3263),
]
DT = 240.0
SYPD = lambda s: DT / (365.25 * s)

def unpack(D):
    n = np.array([d[0] for d in D], float)
    s = np.array([d[2] for d in D], float)
    return n, s

ncpu, scpu = unpack(CPU)
ngpu, sgpu = unpack(GPU)

os.makedirs("docs/figures", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

# --- LEFT: wall vs LUMI nodes ---
ax1.loglog(ngpu, sgpu, "o-", lw=2.0, ms=8, color="#d62728",
           label="GPU  (LUMI-G, 8 GCDs/node)")
ax1.loglog(ncpu, scpu, "s-", lw=2.0, ms=8, color="#1f77b4",
           label="CPU  (LUMI-C, 128 ranks/node)")
for nn, ss in zip(ngpu, sgpu):
    ax1.annotate(f"{ss:.3f}", xy=(nn, ss), xytext=(6, 6),
                 textcoords="offset points", fontsize=9, color="#d62728")
for nn, ss in zip(ncpu, scpu):
    ax1.annotate(f"{ss:.3f}", xy=(nn, ss), xytext=(6, -14),
                 textcoords="offset points", fontsize=9, color="#1f77b4")
ax1.set_xlabel("LUMI nodes")
ax1.set_ylabel("s / step")
ax1.set_title("NG5 strong scaling — wall time")
all_nodes = sorted(set(list(ngpu) + list(ncpu)))
ax1.set_xticks(all_nodes)
ax1.set_xticklabels([str(int(x)) for x in all_nodes])
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="upper right", framealpha=0.9)

# --- RIGHT: SYPD ---
ax2.plot(ngpu, SYPD(sgpu), "o-", lw=2.0, ms=8, color="#d62728",
         label="GPU")
ax2.plot(ncpu, SYPD(scpu), "s-", lw=2.0, ms=8, color="#1f77b4",
         label="CPU")
for nn, y in zip(ngpu, SYPD(sgpu)):
    ax2.annotate(f"{y:.2f}", xy=(nn, y), xytext=(6, 6),
                 textcoords="offset points", fontsize=9, color="#d62728")
for nn, y in zip(ncpu, SYPD(scpu)):
    ax2.annotate(f"{y:.2f}", xy=(nn, y), xytext=(6, -14),
                 textcoords="offset points", fontsize=9, color="#1f77b4")
ax2.set_xscale("log")
ax2.set_xlabel("LUMI nodes")
ax2.set_ylabel(f"SYPD  (dt = {int(DT)} s)")
ax2.set_title("NG5 throughput — SYPD")
ax2.set_xticks(all_nodes)
ax2.set_xticklabels([str(int(x)) for x in all_nodes])
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper left", framealpha=0.9)

fig.suptitle("FESOM2 Kokkos on LUMI  —  NG5 (7.4M ocean nodes)  —  CPU vs GPU",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = "docs/figures/ng5_cpu_vs_gpu_lumi.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

print()
print("== NG5 CPU (LUMI-C, 128 ranks/node) ==")
print(f"{'nodes':>5} {'ranks':>6} {'s/step':>10} {'SYPD':>7}")
for n, r, s in CPU:
    print(f"{n:>5} {r:>6} {s:>10.4f} {SYPD(s):>7.3f}")
print()
print("== NG5 GPU (LUMI-G, 8 GCDs/node) ==")
print(f"{'nodes':>5} {'GCDs':>6} {'s/step':>10} {'SYPD':>7}")
for n, g, s in GPU:
    print(f"{n:>5} {g:>6} {s:>10.4f} {SYPD(s):>7.3f}")
print()
print("== GPU/CPU speedup at same node count ==")
print(f"{'nodes':>5} {'GPU':>9} {'CPU':>9} {'ratio':>7}")
for nn in sorted(set(list(ncpu)) & set(list(ngpu))):
    gs = float(sgpu[list(ngpu).index(nn)])
    cs = float(scpu[list(ncpu).index(nn)])
    print(f"{int(nn):>5} {gs:>9.4f} {cs:>9.4f} {cs/gs:>6.2f}x")
