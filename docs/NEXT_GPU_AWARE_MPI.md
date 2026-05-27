# Next session — GPU-aware MPI (on-device halo exchange) · the M5 perf unlock

> **STATUS: M5.1a DONE (2026-05-27, commit `d6b0a1f`, session 18).** Plan executed. Outcome below;
> full write-up in lesson **L47** + `docs/GPU_FIDELITY.md` §M5.1 + the session-18 handoff entry.
> - **The gate (§4.1) was the crux:** Levante `openmpi/4.1.2` is `--without-cuda` → device-ptr MPI
>   **SEGFAULTs**. Switched build-cuda to **`openmpi/4.1.5-nvhpc-24.7`** via **`env_cuda.sh`** (CUDA-aware).
>   `jobs/job_mpi_cuda_smoke` proves it. [[reference-cuda-aware-mpi]]
> - **Built Approach B** (`src/fesom_halo_device.{hpp,cpp}`, `fesom_halo_field()` dispatch, `FESOM_HOST_HALO=1`
>   toggle). Flipped CG + momentum + gm + ice-FCT. Validated at the CUDA run-to-run floor (NOT byte-identity —
>   CUDA is non-deterministic run-to-run; §6's "byte-match" premise was wrong, see L47).
> - **Payoff = ~8% (0.780→0.716 s/step, CORE2 dist_8) — modest, and that's the finding.** The win is the
>   nod3D big fields; nod2D hot loops (CG/EVP) are PCIe-cheap; kpp/eos big halos are host-bound by
>   `fesom_smooth_nod3D`. So the "halo-bound" premise (§0) was only partly right.
> - **Remaining / real next levers** (bigger than the leftover halo flips): port the kpp/eos
>   `fesom_smooth_nod3D` to device; batch/fuse the CG's ~1000 kernel launches/step; a bigger mesh to fill the
>   A100s. Leftover flips (EVP nod2D + device coastal-BC; kpp/eos once smoothing→device) are low/blocked.
>
> The original plan follows for reference.

Paste this whole file as the next-session prompt. It is the "careful planning + implementation" handover
for replacing the host-staged halo exchange with a device-pointer / GPU-aware-MPI path.

## 0. Where we are

M0–M4 COMPLETE — the whole FESOM2 model (ocean + sea ice) is device-resident on the Serial bit-identity
oracle (tag `m4-full-device`; 1-yr CORE2 Serial all-fields bit-identical to `cref`). Repo
`/home/a/a270088/port_kokkos`, branch `master`, work tree clean. `git log --oneline -15` to orient.

**The measured problem (job `25163175`, post-M4): the GPU step is 0.731 s/step on dist_8 (8×A100/2 nodes)
— only ~15% faster than M3.1's 0.86 s/step, despite the CG (M4.2) and the whole sea-ice step (M4.3) now
being on the device.** It is **halo-bound, not compute-bound.** Moving the CG/EVP/FCT onto the device
*added* device↔host PCIe traffic because their per-iteration halos are still **host-staged**. GPU-aware
MPI is the unlock — not more kernel porting. (CORE2 is also small / bandwidth-bound; M3.1 note in
`docs/RUN_GPU.md`.)

(Orthogonal, already in flight: the **M3.2** CUDA 1-yr climate-fidelity run — job `25165660`, ~3.5 h,
→ `/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_1yr`. When it finishes, compare with
`scripts/m32_climate_compare.py` per `docs/GPU_FIDELITY.md`. That is a *fidelity* gate, independent of
this *perf* work — do it whenever; it does not block GPU-aware MPI.)

## 1. The current halo architecture (what to change)

**The D21 bracket** (every device kernel that needs a halo, at np>1) is, e.g. in `src/fesom_ice_fct.cpp`
`fct_halo_nod2D`, `src/fesom_ssh.cpp` CG `exch` lambda, `src/fesom_ice_evp.cpp` per-subcycle uice/vice:
```cpp
f.modify_device();
if (parallel) {
    f.sync_host();                                   // FULL-FIELD device->host PCIe copy  <-- cost
    fesom_halo_exchange(f.h_checked(), kind, nl, nc, partit);   // host pack -> MPI -> host unpack
    f.modify_host();
    f.sync_device();                                 // FULL-FIELD host->device PCIe copy  <-- cost
}
```
`f.sync_host()/sync_device()` copy the **ENTIRE field** (nod2D ~127k or nod3D ~6M doubles) over PCIe,
TWICE, per exchange — and there are MANY exchanges/step: CG ~90–127 iters × `pp`/`rr`; EVP 120 subcycles
× `uice`/`vice`; FCT ~21 brackets; + the ocean substep halos. That full-field PCIe traffic dominates.

**`src/fesom_halo.cpp` (285 LoC)** — one parameterised `fesom_halo_exchange(real_t* field, kind, n_levels,
n_components, partit)`:
- **pack**: per send-PE, gather `slist` segments → host `send_buf` (memcpy per contiguous segment);
- **MPI**: `MPI_Irecv`(recv_buf) / `MPI_Isend`(send_buf) / `MPI_Waitall`;
- **unpack**: scatter `recv_buf` → field halo slots (`rlist`, memcpy).
- index lists live in **`partit` (HOST `int*`)**: `fesom_com_struct { rPEnum, rPE[], rptr[N+1],
  rlist(size eDim), sPEnum, sPE[], sptr[N+1], slist }` × `com_nod2D` / `com_elem2D` / `com_elem2D_full`
  (`src/fesom_partit.h`). `rlist`/`slist` are 1-based LOCAL indices.

## 2. The goal

A device-pointer halo path: **(a)** pack the `slist` into a small **device** `send_buf` with a device
gather kernel (using device-resident `slist`); **(b)** hand the **device** `send_buf`/`recv_buf` pointers
to `MPI_Isend`/`Irecv` (GPU-aware MPI → UCX does the GPU↔GPU / GPUDirect transfer); **(c)** unpack
`recv_buf` → field halo with a device scatter kernel. This removes the two full-field PCIe syncs — only
the (small) packed halo moves, GPU-direct.

## 3. Two approaches — DECIDE (this is the "careful planning")

- **Approach A — unified device-View exchange.** Rewrite `fesom_halo_exchange` to take a Kokkos View (or
  device ptr) and do pack/unpack as device kernels + MPI on `buf.data()`. On Serial/OpenMP `buf.data()` is
  host memory → ordinary MPI (pack/unpack are pure data moves → **bit-identical**); on CUDA it is device
  memory → GPU-aware MPI. ONE path, all backends — elegant, but it changes the Serial-oracle path
  (a pack/unpack index bug would break bit-identity → must re-run the FULL Serial gate).
- **Approach B — separate device path, dispatch by backend (RECOMMENDED first).** Keep the proven host
  path for Serial/OpenMP untouched; add a parallel device-pointer exchange used only when
  `Kokkos::DefaultExecutionSpace` is `Cuda`. The D21 bracket dispatches: CUDA → device exchange (no
  `sync_host`); else → the current host bracket. More code, but ZERO risk to the Serial bit-identity
  oracle (it is literally unchanged). Generalise to Approach A later if the duplication hurts.

## 4. Prerequisites — do these FIRST (gating)

1. **Prove Levante's MPI is CUDA-aware.** `ucx_info -d | grep -i cuda` (want `cuda_copy`, `gdr_copy`,
   `cuda_ipc` TLS); `ompi_info --parsable | grep -i cuda` (want `mca:mpi:base:param:mpi_built_with_cuda_support:value:true`).
   The M3.1 GPU jobs use `openmpi/4.1.2-gcc-11.2.0` + UCX — confirm it was built `--with-cuda`. If NOT, find
   a CUDA-aware MPI module (or escalate — this is the gate). **Write a ~15-line `MPI_Send`/`Recv`
   device-pointer smoke** (2 ranks / 2 GPUs on `gpu-devel`: alloc a `Kokkos::View` on the device, fill it,
   `MPI_Sendrecv` device pointers, verify on the other rank) — PROVE GPU-aware MPI works before touching
   the halo layer. UCX env will likely need `UCX_TLS=...,cuda_copy,gdr_copy,cuda_ipc` + `UCX_MEMTYPE_CACHE=n`
   on top of the existing `job_gpu_core2` UCX block.
2. **Device-resident comm index lists.** `Field`/`IntField`-wrap `com_{nod2D,elem2D,elem2D_full}.rlist`/
   `slist` (+ push `rptr`/`sptr`/`rPE`/`sPE` or capture by value) — set-once at partition read → one-shot
   `modify_host()+sync_device()` (the M1.4 / M4.2-a CSR pattern). These are what the device pack/unpack
   kernels index.
3. **Persistent device `send_buf`/`recv_buf`** Views, sized to the max over all kinds × `nl` × `nc`,
   reused across exchanges (don't alloc per call).

## 5. Implementation plan (incremental — validate at each step)

1. Prereq smoke (§4.1) + the device comm lists (§4.2) + device buffers (§4.3).
2. Implement the device exchange for **ONE kind first: `FESOM_HALO_NOD2D`** (the simplest, and the CG/EVP/
   FCT/ice hot path). Device gather-pack → MPI on device ptrs → device scatter-unpack.
3. Wire ONE call site (e.g. the CG `exch` lambda in `fesom_ssh.cpp`) to dispatch to it on CUDA; validate
   (§6); re-time (§6, the payoff).
4. Generalise to `NOD3D` / `ELEM2D` / `ELEM3D` / `ELEM2D_FULL` (the `n_levels`/`n_components` strides) and
   flip the remaining brackets. Re-validate + re-time after each.

## 6. Validation — the key insight

**GPU-aware MPI changes the DATA PATH, not the arithmetic.** The packed bytes, the MPI copy, and the
unpack are identical to the host-staged path → the result must be **byte-identical to the current
host-staged CUDA run** (this is a *regression* check, NOT a new divergence class). So:
- **CUDA-GPU-aware vs CUDA-host-staged**: diff the snapshots → identical (or the exact same climate-close
  floor as today; the M3.2 budget is UNCHANGED). This is the primary gate.
- **Approach B**: Serial/OpenMP host path untouched → the per-kernel Serial `max|Δ|==0` gates, pi==golden
  (np=1 AND **np=2**, the halo is the whole point of np>1), and the 1-yr CORE2 Serial acceptance are all
  unchanged by construction — but RUN pi np=2 + the CORE2 ice verify to confirm the dispatch didn't leak.
  **Approach A**: must RE-RUN the full Serial gate (per-kernel verifies + pi np1/np2 + the 1-yr acceptance)
  since the pack/unpack was rewritten.
- **Re-measure** `jobs/job_gpu_time_core2` — the payoff. Expect a big drop from 0.731 s/step. Record in
  `docs/GPU_FIDELITY.md` (perf note) + `docs/RUN_GPU.md`.
- `fesom_halo_identity_test` (in `fesom_halo.cpp`) must still pass on the device path.

## 7. Risks / gotchas

- Levante GPUDirect/UCX-CUDA config is finicky — the smoke (§4.1) de-risks it before any halo work.
- The device `MPI_Isend`/`Irecv` **tag + PE order must match the host path exactly** (same `reqs` order,
  same `MPI_DOUBLE` count) — keep the loop structure of the C `fesom_halo_exchange`.
- Start with nod2D end-to-end + validate before generalising — a stride bug in nod3D/elem is easy.
- Watch `ulevels`/partial-cell strides in the `nl`/`nc` packing (mirror the host segment math exactly).

## 8. READ FIRST (absolute paths)

- `/home/a/a270088/port_kokkos/docs/NEXT_GPU_AWARE_MPI.md` (this) + `docs/KOKKOS_HANDOFF.md` (§2 build/CUDA
  recipe, §4 key paths, the M4 acceptance + M3.2 status) + `docs/RUN_GPU.md` (the GPU/UCX env, multi-GPU
  mapping, the M3.1 perf finding) + `docs/GPU_FIDELITY.md` (the 0.731 s/step perf note + the M3.2 budget)
- `/home/a/a270088/port_kokkos/src/fesom_halo.{h,cpp}` (the exchange to port) + `src/fesom_partit.h` (the
  `fesom_com_struct` index lists to device-wrap) + `src/fesom_field.hpp` (Field/IntField + the `.d()/.h()`/
  modify/sync API)
- the D21 bracket call sites that dispatch: `src/fesom_ssh.cpp` (CG `exch`), `src/fesom_ice_evp.cpp`
  (per-subcycle uice/vice), `src/fesom_ice_fct.cpp` (`fct_halo_nod2D`), `src/fesom_tracer_adv.cpp` (the FCT
  internal halos), + the driver halos in `src/fesom_step.cpp`
- `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md` — **D21** (the internal-exchange bracket),
  **D17/L14** (the DualView host-authoritative rails + write-via-alias), **L40** (multi-GPU rank→device +
  the same-rank rule), **L18** (struct-layout change → `touch src/*`), **D22** (scatter/reduce climate-close)
- project memory `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/`
  (`feedback-hpc-run-hygiene.md` = OUTPUT → `/work` or `/scratch`, NEVER `$HOME`; report wall-time/perf)

## 9. Build / run / invariants

- Serial/OpenMP: `source ./env.sh && cmake --build build-{serial,omp} -j 16`. CUDA: `source
  /sw/etc/profile.levante; module --force purge; module unload netcdf-c cdo ncview git; module load
  gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0 netcdf-c/4.8.1-gcc-11.2.0; export
  NVCC_WRAPPER_DEFAULT_COMPILER=g++; cmake --build build-cuda --target fesom_port -j 16`. ⚠️ struct-layout
  change (the comm-list wrap) → `touch src/*` first (L18). ⚠️ `source ./env.sh` doesn't persist across
  shells — source it in the SAME command as `mpirun`.
- GPU runs on the `gpu`/`gpu-devel` partition (12 h wall); the device-ptr MPI needs the CUDA-aware UCX env
  (extend `jobs/job_gpu_core2`'s block with the cuda TLS).
- INVARIANTS: never simplify physics; **Serial stays the bit-identity oracle** (`max|Δ|==0` vs the C twin);
  OpenMP race-free-bit-identical / scatter-reduce climate-close (D22); CUDA climate-close. GPU-aware MPI
  must not change ANY result (data path only). C twin oracle `/home/a/a270088/port2/fesom2_port/src` (SHA
  75de623). OUTPUT → `/work` or `/scratch`, never `$HOME`. Commit per step; update this doc + the handoff
  + a lesson (L47) as you go.
