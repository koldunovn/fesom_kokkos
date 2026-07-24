# GPU↔MPI on the dolpung GH200 partition: measured findings (2026-07-22/23)

Context: FESOM2 C++/Kokkos port, 4 MPI ranks/node (1 per GH200 superchip), stack
`openmpi/4.1.8_cuda-12.9_nvhpc-25.7_gcc-14` + `ucx/1.19.0_cuda-12.9_gcc-14`
(`/sw/mpi/linux-rhel9-neoverse_v2/`), CUDA 12.9, driver 580.159.04. All numbers are
seconds/timestep, min-of-2, 300-step runs.

## 1. GPUDirect is not functional — device-pointer MPI crashes

Passing a `cudaMalloc` pointer to MPI fails with two distinct signatures:

- intra-node (first exchange):
  `gdr_copy_md.c:150 UCX ERROR gdr_pin_buffer failed. length :65536 ret:22`
  → `dt.c:86 Fatal: failed to register buffer with mem type domain cuda` → SIGABRT.
- inter-node:
  `ib_md.c:287 UCX ERROR ibv_reg_mr(address=…, length=…, access=0xf) failed: Bad address`
  → `failed to register address … (cuda) … on md[…]=mlx5_X: Input/output error`.

i.e. gdrcopy cannot pin CUDA memory and the HCAs cannot register it (no
nv_peer_mem/dmabuf path). Any "CUDA-aware MPI" code path is unusable on this
partition today. Worth re-testing after a driver/GPUDirect/dmabuf update.

## 2. If you must work around it, the UCX transport list matters — a lot

- `UCX_TLS=^gdr_copy` avoids the crash but is **3.6× slower** than the right list
  (0.779 vs 0.216 s/step, CORE2, 4 ranks/1 node): UCX then routes intra-node GPU
  traffic over knem/xpmem/IB-loopback.
- The measured-good list: `UCX_TLS=self,sm,cuda_copy,cuda_ipc` intra-node,
  `+,dc_mlx5` for multi-node.
- Neutral within noise: `UCX_MEMTYPE_CACHE=y|n`, `UCX_RNDV_SCHEME=get/put_zcopy`,
  `MALLOC_CONF=thp:never`, `OMPI_MCA_coll=^hcoll,ml` (vs plain hcoll off).
- Per-rank NIC pinning (`UCX_NET_DEVICES=mlx5_$LOCALID:1`, the ICON idiom) was
  **49% slower** for our 4-rank/node host-staged layout (0.293 vs 0.196, dars,
  2 nodes) — UCX's automatic distance-aware selection wins here. Layout-dependent:
  test before adopting either way.

## 3. The pattern that recovers full speed: stage the PACKED halo, not the field

Three ways to move a halo when the fabric can't take device pointers, measured on
the same code (CORE2, 1 node, 4 GPUs):

| pattern | s/step |
|---|---|
| full-field GPU→host sync, host MPI, host→GPU (naive fallback) | 0.079 |
| device pointers to MPI | crash |
| **pack on GPU → copy only packed bytes to pinned host → host MPI → unpack on GPU** | **0.046** |

The packed halo is 10–100× smaller than the field; the copies ride the 450 GB/s
C2C link. With this pattern (and nothing else exotic) the FESOM port runs
**1.8–2.0× faster node-for-node than its best tuned A100 configuration** on
multi-million-node meshes — i.e. the full GH200/A100 silicon ratio (measured 2.0×
chip-for-chip at np1) survives the missing GPUDirect. The naive full-field
fallback loses roughly that entire factor.

Implementation in FESOM-Kokkos: env knob `FESOM_HALO_STAGE=1`, branch `m7-speed`
(FP64) / `m8-precision` (FP32) at github.com/koldunovn/fesom_kokkos; details in
`docs/plans/20260722-dolpung-GH200-SCALING.md`.
