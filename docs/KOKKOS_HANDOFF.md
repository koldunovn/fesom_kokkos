# FESOM2 C → C++/Kokkos port — session handoff

**Session 2 (2026-05-25).** Repo: `/home/a/a270088/port_kokkos` (git). Read this first, then
`docs/plans/20260525-kokkos-port.md`, `docs/KOKKOS_PORTING_LESSONS.md`, and the project memory
in `~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M0 COMPLETE** (tag `m0-baseline`): the C++/Kokkos build is **bit-for-bit identical to the C
  golden on Serial, OpenMP, AND CUDA** (pi smoke → `diff_snap.py` = ALL FIELDS BIT-IDENTICAL on
  all three). "Binary identity when possible," proven on every backend. At M0 there are **no
  device compute kernels** — the CUDA backend is initialised but all compute is still host C-code,
  which is why even CUDA is bit-identical.
- **The nvcc `void*` blocker is RESOLVED.** `scripts/cast_alloc_voidstar.py` cast all 305
  `malloc/calloc/realloc` sites; `-fpermissive` is gone. Removing it also surfaced 12 implicit
  `int→fesom_halo_kind` conversions in `fesom_mesh.cpp` (fixed → named enum constants). CUDA full
  model compiles under nvcc with zero errors. (Lessons L1, L8–L11.)
- **M1.1 DONE**: `src/fesom_field.hpp` — `FieldT<T>` (`Field`=double, `IntField`=int) over
  `Kokkos::DualView<T*, LayoutRight>` with `h()`/`d()`/modify/sync/`h_checked()`. `tests/test_field.cpp`
  (`ctest field`) passes on Serial+OpenMP+CUDA. (Lesson L12.)
- **NEXT: M1.2** — migrate `fesom_mesh` geometry fields to `Field` (Serial must stay bit-identical).

## 1. Git state

```
m0-baseline ─┐
8f95c1c  M1.1: fesom::Field DualView wrapper + round-trip test (Serial/OpenMP/CUDA green)
2c960bc  M0 DONE: cast void* allocs, drop -fpermissive — CUDA full model bit-identical on A100  (tag m0-baseline)
0bce139  docs: session-1 handoff + Decisions/Lessons log
5857165  M0: tag CPU baseline; document nvcc void*-cast finding  (tag m0-cpu-baseline)
6283369  M0.3-M0.5: C++/Kokkos flip — Serial & OpenMP bit-identical to C golden
535cc8c  M0.2: Kokkos 4.4.01 backends validated (Serial/OpenMP/CUDA-A100)
```
Working tree clean. First checkout elsewhere needs `git submodule update --init --recursive`.

## 2. Build & run (full recipe in `docs/BUILD.md`; MPI caveat in `docs/reference/PROVENANCE.md`)

```bash
cd /home/a/a270088/port_kokkos
source ./env.sh                                   # gcc11 + openmpi + netcdf (Serial/OpenMP)
# --- Serial ---
cmake -S . -B build-serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-serial -j 16
# pi smoke (login-node MPI override — UCX/IB unavailable off compute nodes):
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL \
      HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
./build-serial/fesom_port /home/a/a270088/port2/fesom2/test/meshes/pi /tmp/pi_check 100 20 10
/work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/diff_snap.py \
    docs/reference/c_baseline_snapshots/pi /tmp/pi_check        # expect ALL FIELDS BIT-IDENTICAL
( cd build-serial && ctest )                      # 4/4: calendar, io_stream_unit, io_config, field

# --- CUDA (now builds the full model; nvcc ~fast at M0, all host code) ---
module load nvhpc/24.7-gcc-11.2.0 ; export NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER=$PWD/externals/kokkos/bin/nvcc_wrapper
cmake --build build-cuda -j 16
sbatch jobs/job_pi_smoke_gpu                       # A100 pi smoke → runs/pi_check_cuda
#   then diff_snap.py docs/reference/c_baseline_snapshots/pi runs/pi_check_cuda
```
GPU unit test: `sbatch … --wrap "… ./build-cuda/test_field"` (see git history for the exact wrap),
or just run `test_field` inside any `gpu-devel` allocation.

## 3. THE NEXT TASK — M1.2: migrate `fesom_mesh` fields to `Field`

Per plan §M1.2. The data layer primitive (`fesom::Field`) is ready; now move the **persistent mesh
state** into it while keeping every legacy access compiling and the Serial run bit-identical.

1. In `src/fesom_mesh.h`, back each persistent geometry array with a `fesom::Field`
   (`real_t*` members) / `fesom::IntField` (`int*` members). **Keep every legacy `mesh->area[...]`
   access working** — the hot arrays have 28–84 call sites each, so do NOT rewrite call sites:
   keep the raw pointer member and re-point it at `field.h()` right after `alloc` (a non-owning
   alias; stable for set-once mesh data). The `Field` is the owner; `free()` it (not the raw ptr).

   Migration inventory (members of `struct fesom_mesh`, `src/fesom_mesh.h`):
   - `real_t*` → `Field`: `coord_nod2D geo_coord_nod2D zbar Z zbar_3d_n depth mesh_resolution
     elem_area area areasvol elem_cos metric_factor coriolis coriolis_node elem_center_x
     elem_center_y edge_dxdy edge_cross_dxdy gradient_sca hnode hnode_new helem hbar hbar_old
     bc_index_nod2D`
   - `int*` → `IntField`: `coast_flag elem_nodes edges edge_tri nlevels_nod2D nlevels_nod2D_min
     ulevels_nod2D ulevels_nod2D_max nlevels ulevels nod_in_elem2D_offsets nod_in_elem2D
     edge_up_dn_tri`
   - Migrate in waves (e.g. scalar CV areas first: `area areasvol elem_area`), gating Serial
     bit-identity after **each** wave so a regression is localised.
2. In `src/fesom_mesh.cpp`, replace the `calloc/malloc` of those arrays with `field.alloc(...)`;
   replace `free(...)` with `field.free()`; `sync_device()` the mesh **once** after the metrics are
   computed (set-once data — `compute_metrics`/`compute_node_areas`/`compute_metric_and_coriolis`).
3. **Gate: Serial pi smoke == golden (bit-identical) + `ctest` 4/4** before moving on. Then M1.3
   (dyn/aux/tracers), M1.4 (forcing/ice), M1.5 (sync map + 1-yr CORE2 acceptance, tag `m1-datalayer`).

**Invariant for all of M1:** the only device op is `deep_copy` of `double`/`int` (bitwise-exact) and
NO compute kernel runs on device → Serial+OpenMP+CUDA all stay bit-identical. Anything that breaks
M1 bit-identity (a stray device `parallel_for`/fill) is a bug. **Per-kernel scratch arrays
(GM/KPP/FCT/tracer-diff/CG) are NOT migrated in M1** — each is wrapped inside its own M2/M4 task.

**Always:** gate every step on Serial bit-identity vs the C twin; never simplify physics; and
**append every decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` in the same commit**.

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` · Lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D11,
  L1–L12) · C-port lessons: `docs/PORTING_LESSONS.md` · Build: `docs/BUILD.md` · Provenance/golden +
  MPI override: `docs/reference/PROVENANCE.md`
- Field primitive: `src/fesom_field.hpp` · its test: `tests/test_field.cpp` · cast codemod (reusable):
  `scripts/cast_alloc_voidstar.py` · GPU pi smoke: `jobs/job_pi_smoke_gpu`
- C source (twin) & Fortran ground truth: `/home/a/a270088/port2/fesom2_port/src` (SHA `75de623`),
  `/home/a/a270088/port2/fesom2/src`, ref run `/scratch/a/a270088/fortran_pp_2yr`
- Mesh (pi): `/home/a/a270088/port2/fesom2/test/meshes/pi` · CORE2: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`
- Kokkos submodule: `externals/kokkos` (4.4.01) · SLURM account `ab0995`; GPU partitions `gpu` / `gpu-devel`.

## 5. NEXT-SESSION PROMPT (paste this verbatim)

> Continue the FESOM2 C→C++/Kokkos port in `/home/a/a270088/port_kokkos` (git; branch `master`).
> HEAD is two commits past tag `m0-baseline` (M1.1 `fesom::Field`, then the session-2 handoff);
> `git log --oneline -5` to orient. **M0 is complete** — Serial+OpenMP+CUDA builds are all
> bit-for-bit identical to the C golden — and **M1.1 is done**: the `fesom::Field` DualView wrapper
> + `test_field` pass on all three backends. Work tree is clean.
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status, build/run recipes §2, the M1.2 field inventory §3)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M1.2)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D11, L1–L12) — APPEND to this every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps (dt=1800 AB2 eps, tracer stride nl, halo bounds)
> - `/home/a/a270088/port_kokkos/docs/BUILD.md` and `/home/a/a270088/port_kokkos/docs/reference/PROVENANCE.md`  ← backend modules + golden capture cmd / login-node MPI override
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/` (`project-kokkos-port.md`, `reference-c-port.md`, `reference-build-run.md`, `feedback-document-decisions.md`, `user-role.md`)
> - already built: `/home/a/a270088/port_kokkos/src/fesom_field.hpp` (the Field type) + `/home/a/a270088/port_kokkos/tests/test_field.cpp`; reusable cast codemod `/home/a/a270088/port_kokkos/scripts/cast_alloc_voidstar.py`; GPU smoke `/home/a/a270088/port_kokkos/jobs/job_pi_smoke_gpu`
>
> GOAL: **M1.2 — back the persistent `fesom_mesh` geometry arrays with `fesom::Field`/`IntField`**
> (per plan §M1.2 and the inventory in handoff §3). Keep every legacy `mesh->…[...]` access
> compiling by keeping the raw-ptr member and re-pointing it at `field.h()` after alloc (the hot
> arrays have 28–84 call sites — do NOT rewrite them). `Field` owns the storage; `free()` it.
> `sync_device()` the mesh once after the metrics are computed (set-once data). Migrate in waves,
> gating after each wave. Then M1.3 (`fesom_dyn`/`aux`/`tracers`), M1.4 (`forcing`/ice); M1.5 adds
> the sync map + a 1-yr CORE2 acceptance and tags `m1-datalayer`.
>
> GATE every step (recipe in §2): Serial pi smoke `./build-serial/fesom_port
> /home/a/a270088/port2/fesom2/test/meshes/pi /tmp/pi_check 100 20 10` then
> `/work/ab0995/a270088/mambaforge/envs/nereus/bin/python
> /home/a/a270088/port_kokkos/scripts/diff_snap.py
> /home/a/a270088/port_kokkos/docs/reference/c_baseline_snapshots/pi /tmp/pi_check` must print
> ALL FIELDS BIT-IDENTICAL, and `ctest` 4/4 (calendar, io_stream_unit, io_config, field). CUDA
> bit-identity via `sbatch /home/a/a270088/port_kokkos/jobs/job_pi_smoke_gpu`.
>
> INVARIANTS: M1 moves NO compute to the device (deep_copy of double/int only → Serial+OpenMP+CUDA
> all stay bit-identical; a stray device `parallel_for`/fill is a bug). Per-kernel scratch arrays
> (GM/KPP/FCT/tracer-diff/CG) are NOT migrated in M1 — each is wrapped inside its own M2/M4 task.
> Never simplify physics; preserve every constant/loop-bound verbatim (re-read PORTING_LESSONS
> before touching a kernel). Append every decision/lesson to KOKKOS_PORTING_LESSONS.md in the SAME
> commit; commit per milestone-step. C twin (the oracle): `/home/a/a270088/port2/fesom2_port/src`
> (SHA 75de623); Fortran ground truth `/home/a/a270088/port2/fesom2/src`, run `/scratch/a/a270088/fortran_pp_2yr`.
> SLURM account `ab0995`; GPU partitions `gpu`/`gpu-devel`. First fresh checkout: `git submodule update --init --recursive`.
