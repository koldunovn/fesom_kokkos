# FESOM2 C → C++/Kokkos port — session handoff

**Session 4 (2026-05-26) — M1.3 complete (commit `d42c7cc`).** Repo: `/home/a/a270088/port_kokkos` (git). Read this first, then
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
- **M1.2 DONE** (commits `5f5cb04` W1, `0229fff` W2, + the W3 commit at HEAD; build-green `01edc20`): **all 28
  persistent `fesom_mesh` arrays are now `Field`/`IntField`-backed**, raw pointer kept as a
  non-owning alias = `field.h()` (0 of the 28–124 call sites/array changed). Migrated in 3 gated
  waves — set-once geometry + state (W1), scatter-touched connectivity/coords + zbar/Z (W2),
  bc_index_nod2D (W3). `fesom_mesh_free` is now just `*m = fesom_mesh{}`; the set-once geometry is
  pushed to device once after `compute_metrics`. **Serial np=1 + np=2 (dist_2) ALL FIELDS
  BIT-IDENTICAL; OpenMP np=1 bit-identical; CUDA np=1 (A100) bit-identical; ctest 4/4.** New
  decisions D12–D14, lessons L13–L17. (M1 invariant intact: device does only deep_copy of double/int.)
- **M1.3 DONE** (commit at HEAD): **all `fesom_dyn` (19), `fesom_aux` (11), `fesom_tracers` (7)
  persistent arrays are now `Field`/`IntField`-backed** (same M1.2 alias pattern, D15). Serial
  np=1 == golden, **Serial np=2 == M1.2 oracle (CMA-off)**, ctest 4/4, **CUDA np=1 (A100)
  bit-identical** — all verified. ⚠️ **A multi-hour red herring resolved (L18): the np=2 gate's
  apparent divergence was a login-node `vader` CMA `MPI_Gatherv` artifact (identical sends →
  address-dependent gather), NOT the port** — the per-step OWNED state is byte-identical. The np=2
  gate now requires `OMPI_MCA_btl_vader_single_copy_mechanism=none` (recipe §2) and a regenerated
  oracle (`pi_np2_ref_m13_nocma`). New decision D15; lessons L18 (vader-CMA), L19 (diff_snap dirs-only).
- **NEXT: M1.4** — migrate `fesom_forcing` + sea-ice structs to `Field` (same alias pattern; honour
  halo-sized allocation `feedback_array_size_vs_reader_loop`). Serial must stay bit-identical.

## 1. Git state

```
HEAD     d42c7cc M1.3: migrate fesom_dyn/aux/tracers to fesom::Field (37 arrays); resolve np=2 vader-CMA gate artifact
e2dc45e  M1.2 Wave 3: migrate bc_index_nod2D — fesom_mesh fully Field-backed; M1.2 complete
01edc20  build: keep full CUDA build green — exclude host-only C unit tests (nvcc .c)
0229fff  M1.2 Wave 2: migrate scatter-touched fesom_mesh arrays to fesom::Field
5f5cb04  M1.2 Wave 1: back set-once fesom_mesh geometry+state with fesom::Field
2c960bc  M0 DONE: cast void* allocs, drop -fpermissive — CUDA full model bit-identical on A100  (tag m0-baseline)
```
(np=2 scatter oracle: **`/scratch/a/a270088/pi_np2_ref_m13_nocma`** — captured CMA-off, L18; the
old `…_m12` is CMA-tainted, do not use.) Working tree clean. First checkout elsewhere needs
`git submodule update --init --recursive`.

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
mkdir -p /tmp/pi_check                            # REQUIRED: a missing out-dir → NetCDF "Permission denied" (L15)
./build-serial/fesom_port /home/a/a270088/port2/fesom2/test/meshes/pi /tmp/pi_check 100 20 10
/work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/diff_snap.py \
    docs/reference/c_baseline_snapshots/pi /tmp/pi_check        # expect ALL FIELDS BIT-IDENTICAL
( cd build-serial && ctest )                      # 4/4: calendar, io_stream_unit, io_config, field
# np=2 scatter gate (login-node vader; exercises scatter_mesh, which np=1 skips — D14):
# ⚠️ MUST disable vader CMA or the snapshot MPI_Gatherv is buffer-address-dependent (L18) —
#    a false "divergence" that depends on struct sizes, NOT on the port.
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mkdir -p /tmp/pi_np2 && mpirun -np 2 ./build-serial/fesom_port \
    /home/a/a270088/port2/fesom2/test/meshes/pi /tmp/pi_np2 100 20 10
/work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/diff_snap.py \
    /scratch/a/a270088/pi_np2_ref_m13_nocma /tmp/pi_np2   # expect ALL FIELDS BIT-IDENTICAL
#   (robust CMA-off oracle; the old …_m12 was CMA-tainted — do NOT use it)

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

## 3. THE NEXT TASK — M1.4: migrate `fesom_forcing` + sea-ice structs to `Field`

Per plan §M1.4. **Reuse the exact M1.2/M1.3 pattern** (now proven on 28 mesh + 37 dyn/aux/tracers
arrays): `Field`/`IntField` member per persistent array; raw ptr = non-owning `field.h()` alias
re-pointed after `.alloc`; `memset(s,0,sizeof)` → `*s = T{}` (D13); allocate-once/free-once. Honour
**halo-sized allocation** (`feedback_array_size_vs_reader_loop`). Check the structs are
stack/`new` (Field ctors run), not `malloc`'d (L13). Convert forcing (`stress_*`, `heat_flux`,
`water_flux`, `virtual_salt`, …) and ice state (`a_ice/m_ice/m_snow/u_ice/v_ice` + EVP/FCT work).
**Gate** exactly as M1.3 below — and note the **np=2 gate needs
`OMPI_MCA_btl_vader_single_copy_mechanism=none`** vs `pi_np2_ref_m13_nocma` (L18).

⚠️ **M1.3 is DONE** — the rest of this section documents the M1.3 method (reuse it for M1.4):

Per plan §M1.3. **Reuse the exact M1.2 pattern** (now proven on 28 mesh arrays):
- Add a `fesom::Field`/`IntField` member per persistent array in `src/fesom_{dyn,aux,tracers}.{h,cpp}`;
  keep the legacy `real_t*`/`int*` member as a **non-owning alias re-pointed at `field.h()`** right
  after `field.alloc(label, n)` (count in **elements**) — do NOT touch call sites (D12).
- Replace `calloc/malloc` with `.alloc`; in the struct's `free`/`init`, replace
  `memset(s,0,sizeof)` with `*s = T{}` (D13/L13) and drop the per-array `free()` (the assignment
  releases every Field). If a struct is `malloc`'d (not a stack/`new` object), it must instead be
  constructed — check (`grep -nE 'malloc.*sizeof.*(dyn|aux|tracers)'`), since `malloc` skips Field
  ctors (L13).
- Arrays to convert: **dyn** `uv/uv_rhs/uv_rhsAB/w/w_e/w_i/eta_n/d_eta/ssh_rhs*/uvnode*/cfl_z`;
  **aux** `density_m_rho0/hpressure/bvfreq/sw_alpha/sw_beta/Kv/Av/pgf_x/pgf_y/…`; **tracers**
  `tracers->data[*].values` — ⚠️ honour the **stride `nl`** (`feedback_tracer_stride_nl`,
  `PORTING_LESSONS §5`): index with `FESOM_NODE3D`, alloc `N*nl` elements.
- ⚠️ **Per-kernel scratch arrays (GM/KPP/FCT/tracer-diff/CG `fer_*`, `blmc`, `r/p/Ap`, …) are NOT
  migrated in M1** — each is wrapped inside its own M2/M4 task (plan scope note).
- After the producing routine fills a Field on host via the alias, the **device copy needs
  `modify_host()` then `sync_device()`** or it stays stale (L14) — but at M1 nothing reads the
  device, so for now follow M1.2: a single `sync_device` pass once the data is set, where one exists.
- If you add a new model `#include` of `fesom_field.hpp`-bearing headers to a file built by a
  CUDA-excluded unit test, mind L17 (nvcc rejects `.c` sources).

**Gate (recipe §2): Serial np=1 == golden + np=2 (dist_2) == oracle + `ctest` 4/4; then a CUDA
np=1 smoke (`sbatch jobs/job_pi_smoke_gpu`) bit-identical.** Then M1.4 (forcing/ice), M1.5 (sync map
+ 1-yr CORE2 acceptance, tag `m1-datalayer`).

**Invariant for all of M1:** the only device op is `deep_copy` of `double`/`int` (bitwise-exact) and
NO compute kernel runs on device → Serial+OpenMP+CUDA all stay bit-identical. A stray device
`parallel_for`/fill is a bug.

**Always:** gate every step on Serial bit-identity vs the C twin; never simplify physics; and
**append every decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` in the same commit**.

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` · Lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D14,
  L1–L17) · C-port lessons: `docs/PORTING_LESSONS.md` · Build: `docs/BUILD.md` · Provenance/golden +
  MPI override: `docs/reference/PROVENANCE.md`
- Field primitive: `src/fesom_field.hpp` · its test: `tests/test_field.cpp` · **M1.2 mesh migration
  as the worked example of the pattern**: `src/fesom_mesh.{h,cpp}` (raw alias = `field.h()`; alloc
  waves; scatter realloc-cycle in `scatter_mesh`; `mesh_sync_geometry_device`) · cast codemod:
  `scripts/cast_alloc_voidstar.py` · GPU pi smoke: `jobs/job_pi_smoke_gpu` (build `--target fesom_port`)
- C source (twin) & Fortran ground truth: `/home/a/a270088/port2/fesom2_port/src` (SHA `75de623`),
  `/home/a/a270088/port2/fesom2/src`, ref run `/scratch/a/a270088/fortran_pp_2yr`
- Mesh (pi): `/home/a/a270088/port2/fesom2/test/meshes/pi` · CORE2: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`
- Kokkos submodule: `externals/kokkos` (4.4.01) · SLURM account `ab0995`; GPU partitions `gpu` / `gpu-devel`.

## 5. NEXT-SESSION PROMPT (paste this verbatim)

> Continue the FESOM2 C→C++/Kokkos port in `/home/a/a270088/port_kokkos` (git; branch `master`).
> `git log --oneline -8` to orient. **M0 + M1.1 + M1.2 + M1.3 are complete**: Serial+CUDA builds are
> bit-for-bit identical to the C golden; `fesom::Field` (DualView wrapper) exists; and **all 28
> `fesom_mesh` + 37 `fesom_dyn`/`fesom_aux`/`fesom_tracers` persistent arrays are now
> `Field`/`IntField`-backed** with the raw pointer kept as a non-owning `field.h()` alias (0 call
> sites changed). Serial np=1 + np=2 (CMA-off) + CUDA np=1 all ALL FIELDS BIT-IDENTICAL; ctest 4/4.
> Work tree clean (HEAD `d42c7cc`).
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status, build/run recipes §2 incl. the **np=2 vader-CMA gate fix**, the M1.4 task §3)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M1.4; §M1.3 is ticked with done-notes)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D15, L1–L19) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps (dt=1800 AB2 eps, **tracer stride nl**, halo bounds)
> - `/home/a/a270088/port_kokkos/src/fesom_{mesh,dyn,aux,tracers}.{h,cpp}`  ← **worked examples of the migration pattern** (Field member + raw alias = `field.h()`; `*x = T{}`)
> - `/home/a/a270088/port_kokkos/src/fesom_field.hpp` + `tests/test_field.cpp`; GPU smoke `jobs/job_pi_smoke_gpu`
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/`
>
> GOAL: **M1.4 — back the persistent `fesom_forcing` + sea-ice (`fesom_ice*`) arrays with
> `fesom::Field`/`IntField`** (per plan §M1.4 + handoff §3), reusing the M1.2/M1.3 pattern verbatim:
> add a Field member per array, keep the raw pointer as a `field.h()` alias re-pointed after `.alloc`,
> do NOT rewrite call sites; `memset(s,0,sizeof)` → `*s = T{}` (D13; first check the structs are
> stack/`new`, not `malloc`'d — Field has a non-trivial ctor, L13; and audit for pointer swaps as in
> M1.3/D15). Convert forcing (`stress_*`, `heat_flux`, `water_flux`, `virtual_salt`, …) honouring
> **halo-sized allocation** (`feedback_array_size_vs_reader_loop`) and ice state
> (`a_ice/m_ice/m_snow/u_ice/v_ice` + EVP/FCT work). Per-kernel scratch (GM/KPP/FCT/tracer-diff/CG) is
> NOT migrated in M1. Then M1.5 (sync map + 1-yr CORE2 acceptance, tag `m1-datalayer`).
>
> GATE every step (recipe §2): `mkdir -p /tmp/pi_check`, Serial pi smoke
> `./build-serial/fesom_port .../meshes/pi /tmp/pi_check 100 20 10` then `…/nereus/bin/python
> scripts/diff_snap.py docs/reference/c_baseline_snapshots/pi /tmp/pi_check` must print ALL FIELDS
> BIT-IDENTICAL; `ctest` 4/4; **the np=2 gate — `export OMPI_MCA_btl_vader_single_copy_mechanism=none`
> first (L18!) — vs `/scratch/a/a270088/pi_np2_ref_m13_nocma`** (`diff_snap.py` takes DIRECTORIES, L19);
> CUDA bit-identity via `sbatch jobs/job_pi_smoke_gpu` (build `--target fesom_port`, L17).
>
> ⚠️ If a gate "diverges": first rule out the **vader-CMA artifact (L18)** — dump the OWNED state
> right after the producing kernel; if byte-identical across builds, the divergence is in the
> snapshot `MPI_Gatherv`/transport, not the port (cost the M1.3 session hours).
>
> INVARIANTS: M1 moves NO compute to the device (deep_copy of double/int only → all backends stay
> bit-identical; a stray device `parallel_for`/fill is a bug). Never simplify physics; preserve every
> constant/loop-bound verbatim (re-read PORTING_LESSONS before touching a kernel). Append every
> decision/lesson to KOKKOS_PORTING_LESSONS.md in the SAME commit; commit per milestone-step. C twin
> (the oracle): `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623); Fortran ground truth
> `/home/a/a270088/port2/fesom2/src`, run `/scratch/a/a270088/fortran_pp_2yr`. SLURM account `ab0995`;
> GPU partitions `gpu`/`gpu-devel`. First fresh checkout: `git submodule update --init --recursive`.
