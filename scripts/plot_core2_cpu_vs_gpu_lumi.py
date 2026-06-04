#!/usr/bin/env python3
"""CORE2 CPU vs GPU strong-scaling on LUMI.

Apples-to-apples on the LUMI-node axis:
  CPU: LUMI-C, AMD EPYC 7763, 128 MPI ranks/node (1 rank/core, no OpenMP).
       Kokkos Serial backend, build-cpu (PrgEnv-gnu, gcc-native).
  GPU: LUMI-G, AMD MI250X, 8 MPI ranks/node (1 rank/GCD).
       Kokkos HIP backend, build-hip (PrgEnv-amd, ROCm 6.3.4), GPU-aware MPI on.

CORE2 mesh (127k ocean nodes), dt=1800 s, 200 steps (5 warmup excluded), I/O
off, JRA55 1958, full ocean + sea-ice + GM/Redi pipeline. Same binary code
on both sides (FESOM_GPU_RESIDENT==0 on CPU → host-staged halo; ==1 on GPU →
device-pointer halo).

Captured 2026-06-04.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# (LUMI nodes, MPI ranks, s/step). None = run pending/missing.
CPU = [
    (1, 128,  0.1823),
    (2, 256,  0.0948),
    (4, 512,  0.0590),
    (8, 1024, 0.0543),
]
GPU = [
    (2, 16,  0.1129),
    (4, 32,  0.1062),
    (8, 64,  0.1037),
    (16, 128, 0.1203),
]
DT = 1800.0
SYPD = lambda s: DT / (365.25 * s)

def unpack(D):
    keep = [(n, r, s) for (n, r, s) in D if s is not None]
    n = np.array([k[0] for k in keep], float)
    s = np.array([k[2] for k in keep], float)
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
    ax1.annotate(f"{ss:.4f}", xy=(nn, ss), xytext=(6, 6),
                 textcoords="offset points", fontsize=9, color="#d62728")
for nn, ss in zip(ncpu, scpu):
    ax1.annotate(f"{ss:.4f}", xy=(nn, ss), xytext=(6, -14),
                 textcoords="offset points", fontsize=9, color="#1f77b4")
ax1.set_xlabel("LUMI nodes")
ax1.set_ylabel("s / step")
ax1.set_title("CORE2 strong scaling — wall time")
all_nodes = sorted(set(list(ngpu) + list(ncpu)))
ax1.set_xticks(all_nodes)
ax1.set_xticklabels([str(int(x)) for x in all_nodes])
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="lower left", framealpha=0.9)

# --- RIGHT: SYPD ---
ax2.plot(ngpu, SYPD(sgpu), "o-", lw=2.0, ms=8, color="#d62728",
         label="GPU")
ax2.plot(ncpu, SYPD(scpu), "s-", lw=2.0, ms=8, color="#1f77b4",
         label="CPU")
for nn, y in zip(ngpu, SYPD(sgpu)):
    ax2.annotate(f"{y:.0f}", xy=(nn, y), xytext=(6, 6),
                 textcoords="offset points", fontsize=9, color="#d62728")
for nn, y in zip(ncpu, SYPD(scpu)):
    ax2.annotate(f"{y:.0f}", xy=(nn, y), xytext=(6, -14),
                 textcoords="offset points", fontsize=9, color="#1f77b4")
ax2.set_xscale("log")
ax2.set_xlabel("LUMI nodes")
ax2.set_ylabel(f"SYPD  (dt = {int(DT)} s)")
ax2.set_title("CORE2 throughput — SYPD")
ax2.set_xticks(all_nodes)
ax2.set_xticklabels([str(int(x)) for x in all_nodes])
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="lower right", framealpha=0.9)

fig.suptitle("FESOM2 Kokkos on LUMI  —  CORE2 (127k ocean nodes)  —  CPU vs GPU",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = "docs/figures/core2_cpu_vs_gpu_lumi.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# table
print()
print("== CORE2 CPU (LUMI-C, 128 ranks/node) ==")
print(f"{'nodes':>5} {'ranks':>6} {'s/step':>10} {'SYPD':>7}")
for n, r, s in CPU:
    if s is None: print(f"{n:>5} {r:>6} {'pending':>10}"); continue
    print(f"{n:>5} {r:>6} {s:>10.4f} {SYPD(s):>7.2f}")
print()
print("== CORE2 GPU (LUMI-G, 8 GCDs/node) ==")
print(f"{'nodes':>5} {'GCDs':>6} {'s/step':>10} {'SYPD':>7}")
for n, g, s in GPU:
    if s is None: continue
    print(f"{n:>5} {g:>6} {s:>10.4f} {SYPD(s):>7.2f}")
