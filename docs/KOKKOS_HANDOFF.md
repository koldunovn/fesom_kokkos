# FESOM2 C → C++/Kokkos port — session handoff

**Session 1 (2026-05-25).** Repo: `/home/a/a270088/port_kokkos` (git). Read this first, then
`docs/plans/20260525-kokkos-port.md`, `docs/KOKKOS_PORTING_LESSONS.md`, and the project memory
in `~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M0 CPU baseline DONE** (tag `m0-cpu-baseline`): the C++/Kokkos build is **bit-for-bit
  identical to the C golden** on **Serial AND OpenMP** (pi smoke → `diff_snap.py` = ALL FIELDS
  BIT-IDENTICAL). This is the goal "binary identity when possible," proven.
- **Kokkos 4.4.01** vendored as a submodule; **all three backend smokes pass** — Serial, OpenMP,
  and **CUDA on a real A100** (`gpu-devel`). `kokkos_smoke/` = parallel_for + parallel_reduce + DualView.
- **OPEN (the one thing left in M0):** the **CUDA full-model build is blocked on `void*` casts.**
  nvcc's front-end rejects `void*→T*` and does **not** honor `-fpermissive` (g++-only). The Serial/
  OpenMP builds use `-fpermissive`; nvcc needs real casts. This is the **next task** (below).
- After that: **M1 — DualView data layer**.

## 1. Git state

```
m0-cpu-baseline → 5857165 M0: tag CPU baseline; document nvcc void*-cast finding
                  6283369 M0.3-M0.5: C++/Kokkos flip — Serial & OpenMP bit-identical to C golden
                  535cc8c M0.2: Kokkos 4.4.01 backends validated (Serial/OpenMP/CUDA-A100)
                  b6ab330 M0.2: vendor Kokkos 4.4.01 submodule + backend smoke
                  ffd9c72 M0.1: build C reference, capture pi golden, validate diff_snap gate
                  8ecdfc7 docs: add C++/Kokkos port implementation plan (M0-M6)
                  4778655 Import validated C port baseline (fesom2_port @ 75de623)
```
Working tree clean. First checkout elsewhere needs `git submodule update --init --recursive`.

## 2. Build & run (full recipe in `docs/BUILD.md`; MPI caveat in `docs/reference/PROVENANCE.md`)

```bash
cd /home/a/a270088/port_kokkos
# Serial (login)
source ./env.sh                                   # gcc11 + openmpi + netcdf
cmake -S . -B build-serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-serial -j 16
# pi smoke (login-node MPI override — UCX/IB not available off compute nodes):
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL \
      HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
./build-serial/fesom_port /home/a/a270088/port2/fesom2/test/meshes/pi  /tmp/pi_check  100 20 10
# bit-identity gate vs golden:
/work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/diff_snap.py \
    docs/reference/c_baseline_snapshots/pi  /tmp/pi_check        # expect ALL FIELDS BIT-IDENTICAL

# CUDA (config + Kokkos-CUDA link already work; model TUs need casts first — see §3)
module load nvhpc/24.7-gcc-11.2.0 ; export NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER=$PWD/externals/kokkos/bin/nvcc_wrapper
# GPU smoke job: jobs/job_kokkos_smoke_gpu  (or srun -p gpu-devel -A ab0995 --gres=gpu:1 ...)
```

## 3. THE NEXT TASK — finish the CUDA build (closes M0), then M1

1. **Cast codemod** (resolves the nvcc block — see lesson L1 in `docs/KOKKOS_PORTING_LESSONS.md`):
   write a script that casts all 303 `malloc/calloc/realloc` sites in `src/*.cpp`:
   - assignment `lhs = alloc(...)`  →  `lhs = (decltype(lhs))alloc(...)`
   - declaration `T *x = alloc(...)` →  `T *x = (T*)alloc(...)`
   Watch for multi-line alloc calls; fix stragglers by compiler error.
2. **Drop `-fpermissive`** from `CMakeLists.txt`, rebuild **Serial**, and confirm
   `diff_snap.py` vs `docs/reference/c_baseline_snapshots/pi` is **STILL ALL FIELDS
   BIT-IDENTICAL** (casts must not change codegen — this is the safety check).
3. **Rebuild CUDA** (background; ~10–20 min nvcc), run the pi smoke on a `gpu-devel` A100,
   `diff_snap.py` vs golden — **expect bit-identical** (all compute is host code at M0).
   Then **`git tag m0-baseline`** and tick the remaining M0.5 box.
4. **Begin M1 (DualView data layer)** per plan §M1: `fesom_field` wrapper (LayoutRight),
   migrate mesh/dyn/aux/tracers/forcing/ice; Serial stays bit-identical (no compute on device yet).

**Always:** gate every step on Serial bit-identity vs the C twin; never simplify physics; and
**append every decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` in the same commit** (this
pipeline gets reused for more components).

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` · Lessons: `docs/KOKKOS_PORTING_LESSONS.md` ·
  C-port lessons: `docs/PORTING_LESSONS.md` · Build: `docs/BUILD.md` · Provenance/golden + MPI override: `docs/reference/PROVENANCE.md`
- C source (twin) & Fortran ground truth: `/home/a/a270088/port2/fesom2_port/src` (SHA `75de623`),
  `/home/a/a270088/port2/fesom2/src`, ref run `/scratch/a/a270088/fortran_pp_2yr`
- Mesh (pi): `/home/a/a270088/port2/fesom2/test/meshes/pi` · CORE2: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`
- Kokkos submodule: `externals/kokkos` (4.4.01) · ICON's copy (flags ref): `/work/aa0049/a271109/icon-2026.04/externals/kokkos`
- SLURM: account `ab0995`; GPU partitions `gpu` / `gpu-devel`.

## 5. NEXT-SESSION PROMPT (paste this)

> Continue the FESOM2 C→C++/Kokkos port in `/home/a/a270088/port_kokkos`. Read
> `docs/KOKKOS_HANDOFF.md`, `docs/plans/20260525-kokkos-port.md`, and
> `docs/KOKKOS_PORTING_LESSONS.md` first, plus the project memory in
> `~/.claude/projects/-home-a-a270088-port-kokkos/memory/`. HEAD is tagged `m0-cpu-baseline`
> (Serial+OpenMP builds bit-identical to the C golden).
>
> Goal this session: **finish M0 (get the CUDA full-model build working), then start M1.**
> 1. The CUDA build is blocked because nvcc rejects `void*→T*` and ignores `-fpermissive`. Write a
>    codemod casting all 303 `malloc/calloc/realloc` sites in `src/*.cpp` (assignment →
>    `(decltype(lhs))alloc(...)`; declaration → `(T*)alloc(...)`), remove `-fpermissive` from
>    `CMakeLists.txt`, rebuild the **Serial** build, and confirm `diff_snap.py` vs
>    `docs/reference/c_baseline_snapshots/pi` is **still ALL FIELDS BIT-IDENTICAL** (casts must
>    not change codegen).
> 2. Rebuild the **CUDA** build, run the pi smoke on a `gpu-devel` A100, diff vs the golden
>    (expect bit-identical — all host code at M0), then `git tag m0-baseline`.
> 3. Begin **M1 (DualView data layer)** per the plan.
>
> Gate every step on Serial bit-identity against the C twin; never simplify physics; and append
> every decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` in the same commit (this Fortran→C→
> Kokkos pipeline will be reused to port more FESOM components).
