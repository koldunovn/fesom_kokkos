# FESOM2 C → C++/Kokkos Port (CPU + GPU)

## Overview

Port the existing, climate-validated **C** port of FESOM2 to **C++/Kokkos** so the
same source runs on CPU (Serial/OpenMP) and GPU (CUDA now; AMD/HIP later), with
**no NVIDIA vendor lock**.

- **Problem it solves**: the C port is CPU/MPI-only. We want a single performance-portable
  source that runs on Levante A100 *and* LUMI MI250x without rewriting per vendor.
- **Priority**: correctness/fidelity over initial performance. Per-step host↔device copies
  are explicitly acceptable at first.
- **Fidelity target** (the user's words, mapped to a validation ladder):
  - **Kokkos Serial == the C port, bit-for-bit** ("binary identity when possible") — the
    per-kernel gate.
  - **Kokkos OpenMP** ≈ Serial (reduction-order only) → climate-identical.
  - **Kokkos CUDA/HIP** = "very close on the climate time scale" (≈ the Fortran↔C difference
    we already accept) — fma contraction, libdevice/ROCm transcendentals, and reduction/atomic
    order make exact identity impossible on GPU, by design.
- **Integration**: the C port's entire validation apparatus (operator-diff harness,
  per-substep dumps, `scripts/*.py` comparators, `FESOM_NO_*` bisect knobs, the
  `fortran_pp_2yr` reference run) is reused unchanged.

## Context (from discovery)

- **Source C port** (`/home/a/a270088/port2/fesom2_port/`): ~17 kLoC across 35 `src/*.{c,h}`
  modules; CORE2 @ dt=1800, linfs ALE, KPP(default)/PP, FCT advection, GM/Redi, EVP sea ice,
  JRA55 forcing, PHC IC, CG SSH solver; MPI to 864 ranks; reproduces the Fortran climate to
  SST/SSS RMS ~0.005–0.04. **This validated behaviour is the thing we must not break.**
- **Data model**: flat `double*`, row-major `[entity*nl + level]` (level contiguous) via
  `FESOM_NODE3D/ELEM3D/ELEMVEC` macros; `real_t` is one typedef (`double`). Node↔element
  coupling is a **gather** over the `nod_in_elem2D` CSR (race-free). Host MPI halo pack/unpack
  follows each shared-field write (see `fesom_step.c`). SSH is a **parallel CG** (MPI-allreduce
  dot products).
- **Kernel shape**: `for entity { for level {…} }`, entity outer — a natural `parallel_for`
  over entity with the level loop inside the lambda.
- **C→C++ cost is low and mechanical**: ~303 `malloc`/`calloc` sites need casts; 1 designated
  initializer (`fesom_main.c:829`); 2 `restrict`; no VLAs; keyword "clashes" are all in comments.
- **Levante**: A100 (Ampere80), `nvhpc`+CUDA, gcc 11/13, cmake 3.31. GPU partitions `gpu`
  (a100_80, 4 GPUs/node) and `gpu-devel`. Account `ab0995`. I/O is gather→host serial-netcdf,
  so **no GPU-netcdf is needed**.
- **Kokkos availability** (checked 2026-05-25): no loadable `module`, but (1) it's an
  installable **spack** package — `spack install kokkos +cuda cuda_arch=80 +openmp`; the recipe
  also supports `+rocm amdgpu_target=gfx90a` (== LUMI MI250x), confirming the no-vendor-lock
  path; spack's *preferred* version is an older 4.0.01. (2) **ICON vendors Kokkos 4.4.1 as a
  git submodule** at `/work/aa0049/a271109/icon-2026.04/externals/kokkos/` (readable by us, full
  source, built in-tree via `add_subdirectory`) — this is the "ICON kokkos build" and it's the
  approach we adopt (self-contained, version-pinned, portable to LUMI without depending on each
  cluster's spack having the right variant).
- **Reusable scripts** (`scripts/`): `exp1_compare_bidiff.py` (operator diff), `diff_snap.py`,
  `compare_c_vs_fortran.py`, `clim_validation_2yr.py`, `drift_5yr_d1800.py`, KPP cross-checks.
- **Plan/format precedents**: `docs/plans/completed/{20260424-mpi-port,20260425-gm-redi-port,
  20260524-kpp-vertical-mixing}.md`.
- **Project memory**: `~/.claude/projects/-home-a-a270088-port-kokkos/memory/`
  (`project-kokkos-port.md`, `reference-c-port.md`, `user-role.md`).

## Development Approach

This is a **fidelity port of a scientific model**, not greenfield feature work. The
generic "unit-test-per-function/TDD" model does **not** fit physics kernels — the real
test is **bit-identity vs the C twin** and **climate-match vs Fortran**. Concretely:

- **Per-kernel gate (the "test")**: keep the legacy C function *and* the new Kokkos kernel
  in the binary; an env-gated verify mode (`FESOM_KK_VERIFY=<kernel>`) runs **both** on the
  same live state and asserts `max|Δ| == 0` on the **Serial** backend before the Kokkos
  version becomes default. This is the in-binary analogue of `exp1_compare_bidiff.py`.
- **Build gate**: every task keeps **all configured backends compiling** (Serial+OpenMP+CUDA),
  and keeps the **Serial run bit-identical** to the saved C reference snapshots.
- **Stay Kokkos-pure**: `KOKKOS_LAMBDA`/`KOKKOS_FUNCTION`, `Kokkos::deep_copy`, `Kokkos::`
  math only. **No** `cuda*`, `__device__`, `__CUDA_ARCH__`, `<cuda_runtime.h>`. A CI grep
  enforces this (Task M0.3).
- **Maintain a running Decisions & Lessons log — `docs/KOKKOS_PORTING_LESSONS.md` — as a
  first-class deliverable, updated EVERY session.** This Fortran→C→Kokkos pipeline will be
  reused to port *additional* FESOM components later, so every non-obvious **decision** (with
  its rationale) and every **lesson** (what bit us, what worked) must be captured while fresh —
  not reconstructed months later. It complements `docs/PORTING_LESSONS.md` (the Fortran→C
  lessons inherited from the C port). When a task surfaces a decision or a gotcha, append it to
  the log in the SAME commit. Treat an un-logged hard-won lesson as incomplete work.
- Complete each task fully (kernel + its verify gate + backends green) before the next.
- Preserve **every** physics constant and loop bound verbatim — re-read `PORTING_LESSONS.md`
  before touching any kernel (the "no-op at dt=500, decisive at dt=1800" traps).
- The classic unit tests that *do* exist (`tests/test_calendar.c`, `test_io_config.c`,
  `test_io_stream_unit.c`) are carried over and must keep passing.

## Testing Strategy

- **Bit-identity gate (primary)**: `FESOM_KK_VERIFY` in-binary C-vs-Kokkos diff (Serial,
  `max|Δ|==0`) per kernel; whole-run `diff_snap.py` Serial-vs-C-reference `==0`.
- **Climate gate (periodic)**: 2-yr CORE2 vs `fortran_pp_2yr` (+ KPP ref) — SST/SSS area-wtd
  RMS within the established Fortran↔C budget (~0.005–0.04); 5-yr stability run (`max uv < 5`).
  Scripts: `clim_validation_2yr.py`, `drift_5yr_d1800.py`, `kpp_climate_compare.py`.
- **Per-backend acceptance**: Serial = bit-identical · OpenMP = climate-identical (per-step
  Δ ≲ 1e-12) · CUDA/HIP = climate-close (≈ Fortran↔C noise).
- **Plumbing tests**: a `fesom_field` unit test (alloc / host-write→sync→device-read→sync-back
  identity / `deep_copy`); an env-gated `FESOM_KK_SYNCCHECK` coherence assertion.
- **C unit tests**: keep `ctest` (calendar, io_config, io_stream) green throughout.

## Progress Tracking

- mark completed items `[x]` immediately when done
- add newly discovered tasks with ➕ prefix
- document blockers with ⚠️ prefix
- M5/M6 are intentionally coarse — **expand them into detailed tasks when M4 completes**
- keep this plan in sync with actual work

## What Goes Where

- **Implementation Steps** (`[ ]`): everything achievable in this repo — build, code, the
  in-binary verify gates, Serial bit-identity runs.
- **Post-Completion** (no checkboxes): GPU/LUMI run configs, multi-GPU MPI mapping, perf
  targets, upstreaming — things needing external systems or long HPC jobs.

---

## Implementation Steps

## ───────────── M0 · C++/Kokkos bit-identical baseline ─────────────
*Goal: same code, compiled as C++ with Kokkos initialised, builds on Serial+OpenMP+CUDA, and
the Serial (and, since no kernels are on-device yet, CUDA) run is bit-identical to the C binary.*

### Task M0.1: Create fresh repo + import the C sources + capture the golden reference

**Files:**
- Create: `/home/a/a270088/port_kokkos/.git` (`git init`)
- Create: `src/`, `tests/`, `scripts/`, `docs/`, `env.sh`, `configure.sh`, `CMakeLists.txt`,
  job templates, `README.md` (copied from `port2/fesom2_port/`)
- Create: `.gitignore` (build/, *.o, runs/)
- Create: `docs/reference/c_baseline_snapshots/` (saved golden outputs)

- [x] `git init` in `/home/a/a270088/port_kokkos`; copy the C port tree in (sources, tests,
      scripts, docs/plans/completed, env.sh, configure.sh, jobs/ templates, README) — jobs moved
      to `jobs/`; plot/run artifacts excluded
- [x] commit "import validated C port baseline" (`4778655`) + the plan (`8ecdfc7`)
- [x] build the **unmodified C** binary (C99) — clean, exit 0, `build/fesom_port`; ran the **pi
      smoke** golden ref (20 steps, analytical forcing) → `docs/reference/c_baseline_snapshots/pi/`
- [x] record the C reference git SHA (`75de623`) in `docs/reference/C_PORT_SOURCE_SHA.txt` + PROVENANCE.md
- [x] write `docs/reference/PROVENANCE.md` (reference cmds + the login-node MPI override)
- [x] `diff_snap.py` sanity: golden-vs-golden → "ALL FIELDS BIT-IDENTICAL", exit 0 (harness works)
- ➕ [x] CORE2 C-twin reference captured at M1.5 (superseding the deferred 16-rank stub): a full
      **1-yr CORE2, 256-rank, dt=1800** reference at `/scratch/a/a270088/m1_accept/cref` (the M1
      acceptance oracle; `jobs/job_m1accept_cref`). Serial+OpenMP reproduce it bit-identically

### Task M0.2: Vendor Kokkos 4.4.1 as a submodule, built in-tree (ICON's approach)

**Files:**
- Create: `externals/kokkos` (git submodule, **pinned to 4.4.1** to match ICON)
- Modify: `CMakeLists.txt` (`add_subdirectory(externals/kokkos)`; backend via `-DKokkos_ENABLE_*`)
- Modify: `env.sh` (module loads: gcc, cuda via nvhpc, cmake)
- Create: `docs/BUILD.md`

- [x] Kokkos **4.4.01** pinned submodule at `externals/kokkos` (matches ICON); integrated via
      `add_subdirectory` (`kokkos_smoke/CMakeLists.txt`); backend per build dir
- [x] CUDA build: `nvcc_wrapper` as `CMAKE_CXX_COMPILER`, `NVCC_WRAPPER_DEFAULT_COMPILER=g++`,
      `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON`; modules gcc/11.2.0 + nvhpc/24.7 (nvcc 12.5)
- [x] module env documented in `docs/BUILD.md` (Serial/OpenMP = gcc/11.2.0; CUDA += nvhpc/24.7)
- [x] in-tree Kokkos smoke (`parallel_for`+`parallel_reduce`+DualView round-trip): **Serial PASS**,
      **OpenMP PASS** (login), **CUDA PASS on A100-PCIE-40GB** (`srun -p gpu-devel`)
- [x] `docs/BUILD.md` written (incl. spack alternative + LUMI/HIP note); `jobs/job_kokkos_smoke_gpu`
- [x] all three backends configure+smoke green — M0.2 done

### Task M0.3: Convert the build system to C++/Kokkos + add Kokkos-purity guard

**Files:**
- Modify: `CMakeLists.txt` (project `CXX`+`CUDA`; `find_package(Kokkos)`; link
  `Kokkos::kokkos`, `MPI::MPI_CXX`, netcdf, `m`)
- Rename: `src/*.c` → `src/*.cpp` (keep `.h`; git-mv to preserve history)
- Create: `scripts/check_kokkos_purity.sh` (grep gate)
- Create: `cmake/CompilerFlags.cmake` (`-ffp-contract=off` knob for host bit-identity diffs)

- [x] CMake → `project(... CXX)`, `add_subdirectory(externals/kokkos)`, backend via
      `-DKokkos_ENABLE_{SERIAL,OPENMP,CUDA}`; link `Kokkos::kokkos` + `MPI::MPI_CXX` + netcdf
- [x] `git mv` all 36 `src/*.c` → `.cpp`; **tests compiled as C++** (`LANGUAGE CXX`) — deviation
      from "keep tests as C" so linkage matches the now-C++ model objects (single compiler)
- [x] `scripts/check_kokkos_purity.sh` (greps cuda*/hip*/`__device__` etc.) — **passes** (no kernels)
- ➕ [ ] `-ffp-contract=off` **deferred to M2**: the captured golden was built fma=fast, so the
      C++ build matches it at default `-O3` (fma=fast). Adopt `-ffp-contract=off` + re-baseline the
      golden when the first kernel lands (it's the kernel-gate determinism knob, not needed at M0)
- [x] `cmake` configures + builds on Serial **and** OpenMP; purity passes — M0.3 done (CUDA below)

### Task M0.4: Make the code compile as C++ (mechanical)

**Files:**
- Modify: CMake compile flags (`-fpermissive` on the C++ flip), `src/fesom_main.cpp`
  (designated init), `src/fesom_step.cpp` (+`<cmath>`), others as the compiler flags them

> **Decision (M0 cast strategy):** only 74 of 303 alloc sites are the easy `TYPE *x = …`
> declaration pattern; 229 are bare `name = calloc(...)` struct-member assignments whose type a
> script can't infer. Rather than hand-cast 300 sites up front, the M0 flip uses **`-fpermissive`**,
> which downgrades the C++ `void*→T*` conversion from error to warning **without changing
> codegen** (bit-identity unaffected — it's a front-end diagnostic level). Proper casts are added
> **per file as it is C++-ified during M2 kernel conversion**; `-fpermissive` is removed once all
> files are clean. Tracked as a temporary bridge, not the end state.

- [x] `git mv src/*.c → src/*.cpp` (36 files; done in M0.3); tests compiled as C++
- [x] `-fpermissive` added (temporary M0 bridge; handled all 303 `void*→T*` sites — no codegen change)
- [x] **`-fpermissive` REMOVED 2026-05-25** — `scripts/cast_alloc_voidstar.py` cast all 305 alloc
      sites (decl `(T*)`, member `(decltype(lhs))`, subscript/deref `(remove_reference_t<decltype>)`);
      removal also surfaced 12 `int→fesom_halo_kind` conversions in `fesom_mesh.cpp` (fixed → named
      enum constants). Serial pi smoke still ALL FIELDS BIT-IDENTICAL. (Lessons L1/L8–L11)
- [x] designated initializer at `fesom_main.cpp` replaced with `= {}` + explicit assignment
- [x] `<cmath>` added to `fesom_step.cpp`; the **one** non-`-fpermissive` error fixed:
      `goto skip_rest_state` crossed `int iters_rest` init → wrapped the skipped sanity body in a
      block so the var's scope is fully contained (behavior-preserving)
- [x] `fesom_io_stream_dispatch.cpp` in the source list
- [x] build green on Serial **and** OpenMP (CUDA below) — M0.4 done

### Task M0.5: Initialise Kokkos in main; prove the bit-identical baseline

**Files:**
- Modify: `src/fesom_main.cpp` (`Kokkos::ScopeGuard` after `MPI_Init`, before any compute)
- Create: `tests/run_bitident_baseline.sh`

- [x] `Kokkos::initialize(argc,argv)` after `fesom_mpi_init`, `Kokkos::finalize()` before MPI
      finalize; prints the active backend at startup (`[fesom_port] Kokkos backend: …`)
- [x] **Serial** build, pi smoke → `diff_snap.py` vs golden = **ALL FIELDS BIT-IDENTICAL** (gate 0 ✓)
- [x] **OpenMP** build, pi smoke → vs golden = **BIT-IDENTICAL** (bonus; expected — no device kernels)
- [x] tag `m0-cpu-baseline` (Serial+OpenMP bit-identical milestone committed)
- [x] **CUDA full-model build UNBLOCKED 2026-05-25** — cast codemod (above) + the int→enum fix made
      nvcc compile all 36 model TUs with zero errors. CUDA pi smoke on A100 (`gpu-devel`/vader1,
      `Kokkos backend: Cuda`) → `diff_snap.py` vs golden = **ALL FIELDS BIT-IDENTICAL** (expected: no
      device kernels at M0). `jobs/job_pi_smoke_gpu` is the reusable GPU smoke. **Tagged `m0-baseline`.**
- [x] `ctest` green (calendar / io_stream_unit / io_config)
- [x] **M0 COMPLETE**: Serial + OpenMP + CUDA all bit-identical to the C golden ("binary identity
      when possible" achieved on all three backends). → M1.

## ───────────── M1 · DualView data layer (Serial stays bit-identical) ─────────────
*Goal: every PERSISTENT field becomes a `Kokkos::DualView<double*, LayoutRight>` owned by its
struct; legacy kernels keep using a host raw-pointer alias; a sync discipline in the step driver
keeps host/device coherent. No compute moves to device yet → all backends stay bit-identical.*

> **Scope note (review fix):** M1 migrates the *persistent state* arrays
> (mesh/dyn/aux/tracers/forcing/ice). **Per-kernel scratch arrays** owned by GM (`fesom_gm.cpp`:
> `fer_K/fer_C/fer_gamma/neutral_slope/slope_tapered/sigma_xy`), KPP (`fesom_kpp.cpp`:
> `blmc/ghats/profiles`), FCT (`fesom_tracer_adv.cpp`: `edge_up_dn_grad/fct_LO/tr_xy`),
> tracer-diff, and the CG vectors (`fesom_ssh.cpp`: `r/p/Ap`, stiffness CSR) are migrated to
> `Field` **inside their own M2/M4 conversion task** (each such task carries an explicit "wrap
> this kernel's scratch arrays in `Field`" checkbox). This avoids a false "everything migrated"
> claim and the `feedback_unported_consumer_gaps` trap (a device kernel with no device storage).
>
> **Invariant — why CUDA stays bit-identical through all of M1:** the only device operation is
> `Kokkos::deep_copy` of `double` (bitwise-exact) and **no compute kernel runs on device yet**;
> all compute is still host C-code. The first expected CUDA divergence is M2.1. Anything that
> breaks M1 CUDA bit-identity (a stray device `parallel_for`/fill) is a bug.

### Task M1.1: `fesom_field` wrapper + sync discipline primitives

**Files:**
- Create: `src/fesom_field.hpp` (the `Field` type)
- Create: `tests/test_field.cpp` + CMake `add_test`

- [x] `FieldT<T>` template in `src/fesom_field.hpp` (`Field=FieldT<double>`, `IntField=FieldT<int>`):
      `Kokkos::DualView<T*, LayoutRight>` + `alloc(label,n)`, `h()` (= `view_host().data()`),
      `d()` (device `View`), `hv()`, `modify_host()/modify_device()`, `sync_host()/sync_device()`,
      `size()`, `allocated()`, `free()`
- [x] explicit `LayoutRight` so the host mirror byte-matches the C `[entity*nl+lev]` layout
      (legacy macros keep working). Header documents the M5 flip to space-default layout as a
      **rank change** (1-D flat → 2-D/3-D `View`) for interleaved/3-D hot fields (`uv` etc. via
      `FESOM_ELEMVEC`), not a mere `LayoutRight`→default swap, requiring layout-agnostic accessors
- [x] `tests/test_field.cpp` (+ CMake `add_test(field)`): alloc; host write → `sync_device` →
      device mutate in a `parallel_for` → `sync_host` → assert round-trip identity; `deep_copy` path;
      exact-integer values so `==` is deterministic on every backend; runs for `Field` and `IntField`
- [x] **`FESOM_KK_SYNCCHECK` mechanism provided:** each `Field` carries an `Auth` tag
      (`Synced/Host/Device`) set by `modify_host()/modify_device()`; `h_checked()` asserts
      host-authoritative under `-DFESOM_KK_SYNCCHECK` (zero-cost otherwise). **Routing** halo/I/O/
      legacy host reads through `h_checked()` happens as each field migrates (M1.2–M1.5) — no
      Field-backed entry points exist yet. GPU analogue of the stale-halo guard.
- [x] `ctest` for `test_field` passes on **Serial + OpenMP + CUDA** (A100/vader1, `backend: Cuda`,
      `ALL PASS`); full suite 4/4 green on Serial+OpenMP — M1.1 done, → M1.2

### Task M1.2: Migrate `fesom_mesh` geometry fields to `Field`

**Files:**
- Modify: `src/fesom_mesh.h` (members → `Field`; raw-ptr accessors alias `.h()`),
  `src/fesom_mesh.cpp` (alloc/read/free), all call sites that take the raw pointers

- [x] convert **all 28** persistent mesh arrays to `Field`/`IntField` (done in 3 waves):
      Wave 1 = set-once geometry (`compute_metrics`: area/areasvol/elem_area/gradient_sca/coriolis(_node)/
      metric_factor/elem_cos/elem_center_*/edge_dxdy/edge_cross_dxdy/mesh_resolution/zbar_3d_n/
      nod_in_elem2D(_offsets)/ulevels(_nod2D)(_max)/nlevels_nod2D_min/edge_up_dn_tri) + state
      (`alloc_state`: hnode/hnode_new/helem/hbar/hbar_old). Wave 2 = scatter-touched
      (coord_nod2D/geo_coord_nod2D/coast_flag/depth/nlevels_nod2D/elem_nodes/nlevels/edges/edge_tri +
      zbar/Z). Wave 3 = bc_index_nod2D (alloc'd in `fesom_ice.cpp`).
- [x] keep every legacy access compiling via the raw-ptr alias (`mesh->area` still valid) — 0 call
      sites changed; raw ptr re-pointed at `field.h()` after each (re)alloc (incl. the scatter
      free+realloc cycle, L16). `memset(m,0,sizeof)` → `*m = fesom_mesh{}` (D13/L13).
- [x] `sync_device()` the mesh once after `compute_metrics` (`mesh_sync_geometry_device`:
      `modify_host()` then `sync_device()` per field — L14; bc_index sync deferred to M4/ice)
- [x] verify Serial smoke == golden (bit-identical) and `ctest` green — **DONE**: Serial np=1 +
      np=2(dist_2 vs captured oracle, D14) ALL FIELDS BIT-IDENTICAL, ctest 4/4; OpenMP np=1
      bit-identical; **CUDA np=1 (A100) bit-identical** (M1 invariant: device does only deep_copy).
      Commits 5f5cb04 (W1), 0229fff (W2), the W3 commit at HEAD; build-green fix 01edc20 (L17). → M1.3

### Task M1.3: Migrate `fesom_dyn`, `fesom_aux`, `fesom_tracers` to `Field`

**Files:**
- Modify: `src/fesom_dyn.{h,cpp}`, `src/fesom_aux.{h,cpp}`, `src/fesom_tracers.{h,cpp}` + call sites

- [x] convert all 19 `fesom_dyn` arrays (`uv/uv_rhs/uv_rhsAB/w/w_e/w_i/cfl_z/uvnode/uvnode_rhs/u_b/v_b/u_c/v_c/eta_n/d_eta/ssh_rhs/ssh_rhs_old/fer_uv/fer_w`) → `Field`
- [x] convert all 11 `fesom_aux` arrays (`density_m_rho0/hpressure/bvfreq/sw_alpha/sw_beta/Kv/dbsfc/Av/pgf_x/pgf_y` → `Field`, `MLD1_ind` → `IntField`)
- [x] convert `fesom_tracers` (`data[k].values/valuesAB/valuesold` ×2 + `del_ttf`) → `Field`, **stride `nl`** preserved (alloc `N*nl`, `feedback_tracer_stride_nl`)
- [x] same M1.2 pattern (D15): stack structs, raw alias = `field.h()` set once (no pointer swaps — all time-history updates are value/memcpy), `memset`→`*x=T{}` (D13), allocate-once/free-once, **no Field sync** for time-evolving state (cf. mesh `hnode/hbar`; sync is M1.5/M2)
- [x] **verify DONE**: Serial np=1 == golden bit-identical; `ctest` 4/4; **np=2 == M1.2 oracle bit-identical** — ⚠️ but only with `OMPI_MCA_btl_vader_single_copy_mechanism=none`. The default vader CMA path makes the snapshot `MPI_Gatherv` buffer-address-dependent (identical sends → different gather) → a false "divergence" that is NOT in the port (per-step OWNED state proven byte-identical). New robust oracle `/scratch/a/a270088/pi_np2_ref_m13_nocma`; old `…_m12` CMA-tainted. New decision D15, lessons **L18 (vader-CMA), L19 (diff_snap dirs-only)**. CUDA np=1 bit-identity via `jobs/job_pi_smoke_gpu`. — before M1.4

### Task M1.4: Migrate `fesom_forcing` + sea-ice structs to `Field`

**Files:**
- Modify: `src/fesom_forcing.{h,cpp}`, `src/fesom_ice*.{h,cpp}` (ice state arrays) + call sites

- [x] convert all 12 `fesom_forcing` arrays (`heat_flux/water_flux/stress_node_surf/stress_surf/
      runoff/Ssurf/virtual_salt/relax_salt/Ch_atm_oce/Ce_atm_oce/chl/sw_3d`) → `Field` — halo-sized
      extents kept verbatim (`feedback_array_size_vs_reader_loop`)
- [x] convert all 49 `fesom_ice` arrays → `Field`: top-level 19 (`uice*/vice*/stress_*/srfoce_*/
      flx_*/h_ice/h_snow`), `data[3]`×6 (`values/values_old/values_rhs/values_div_rhs/dvalues/
      valuesl`), `work`×15 (incl. `fct_massmatrix`, lazily alloc'd in `fesom_ice_fct.cpp`),
      `thermo`×9 per-node (`ustar/t_skin/thdgr*/apnd/hpnd/ipnd`). Embedded-by-value sub-structs +
      `data[3]` reset/release recursively via one `*ice = fesom_ice{}` (D16/L20)
- [x] same M1.2/M1.3 pattern (D16): stack structs (`fesom_main.cpp:347/361`), raw alias = `field.h()`
      set once (no pointer swaps — audited), `memset`→`*x=T{}` (D13), allocate-once/free-once, no
      Field sync (EVP/thermo/FCT go to device in M4.3)
- [x] **verify DONE**: Serial np=1 == golden bit-identical; `ctest` 4/4; **np=2 == `…m13_nocma` oracle
      bit-identical** (CMA-off, L18 — exercises scatter + halo on the new Field-backed forcing/ice +
      EVP/FCT); **CUDA np=1 (A100) == golden bit-identical** (M1 invariant: device does only deep_copy).
      Bit-identical on the first gate run. — before M1.5

### Task M1.5: Sync discipline in the step driver + M1 acceptance

**Files:**
- Modify: `src/fesom_step.cpp`, `src/fesom_ice.cpp`, `src/fesom_main.cpp`
- Create: `docs/SYNC_MAP.md` (per-substep host/device currency map — mirrors the halo map)

- [x] add coarse-but-correct sync rails (D17): **cadence = host-authoritative + LAZY device sync,
      NO eager per-step copies** — the per-kernel `sync_device(in)→modify_device(out)→sync_host(before
      halo/I/O)` brackets are owned by each M2/M4 kernel task (set-once geometry stays the one eager
      exception). M1.5 adds only `h_checked()` at representative halo/I/O host entry points
      (`fesom_step.cpp` density/bvfreq/uv/T-values + the whole `fesom_io.cpp` snapshot gather) —
      pointer-identical to the raw alias today, so production is unchanged (L22)
- [x] exercise a no-op device round-trip each step under `-DFESOM_KK_SYNCCHECK` (per-step H→D→H bounce
      in `fesom_step`/`fesom_ice`/`fesom_main`; `build-synccheck` Serial Release): pi smoke np=1 **and**
      np=2 (CMA-off) run to completion, **no guard fired**, **ALL FIELDS BIT-IDENTICAL** to golden →
      proves M1 is uniformly host-authoritative. Guard is `fprintf`+`abort`, not `assert` (NDEBUG, L21)
- [x] document the map in `docs/SYNC_MAP.md` (per-substep currency map mirroring the halo cheat sheet:
      ocean 14 substeps + ice + main forcing/IO, incl. KPP's 6 internal exchanges, the FCT pipeline,
      the mid-step CG host round-trip, and the M2/M4 kernel-author checklist)
- [x] per-change gate GREEN (commit `<M1.5>`): Serial np=1 == golden; ctest 4/4; np=2 (CMA-off) ==
      `…m13_nocma` oracle; SYNCCHECK np=1+np=2 == golden (no abort); **CUDA np=1 (A100) == golden**
- [x] **M1 acceptance PASSED** (jobs `25126935/6/7`, `docs/M1_ACCEPTANCE.md`): **1-yr CORE2**
      (360-day, dt=1800, 17280 steps, 256 ranks, monthly snaps) generated a fresh **C-twin reference**
      (none existed — M0.1 deferred it) and **Kokkos Serial + OpenMP each reproduced it ALL FIELDS
      BIT-IDENTICAL across all 13 snapshots**. Wall time Serial 1566 s ≈ C twin 1574 s (no M1
      overhead — identical host code). **CUDA CORE2 deferred to M3.1** (multi-GPU rank→device
      mapping); M1 CUDA does zero device compute → its data-layer identity is mesh-independent and
      already proven on the pi smoke (A100). **Tagged `m1-datalayer`.** → M2.1

## ───────────── M2 · Ocean hot-path kernels on device ─────────────
*Goal: convert leaf compute kernels to `parallel_for`, one (group) per task, each gated
`FESOM_KK_VERIFY` Serial `max|Δ|==0` vs its C twin. LayoutRight (correct-but-uncoalesced on
GPU — fine, slow-first is accepted). Order chosen so the Serial build stays green throughout.*

> **Cross-cutting for every M2/M4 task:**
> - Each task also **wraps that kernel's own scratch arrays in `Field`** (per the M1 scope note).
> - A ported routine that contains **internal `fesom_exchange_*` halo calls** (KPP has 6; the
>   FCT pipeline interleaves them) must round-trip per exchange: device-compute → `sync_host` →
>   halo → `sync_device` → device-compute. The sync map is **per-substep *including* intra-kernel
>   exchanges**, not just per-top-level-kernel — list these sync points as explicit checkboxes.
> - **Mid-step host round-trip is expected, not a regression:** the SSH **CG solve sits in the
>   middle of the step** (`fesom_step.c:147-164`), and stays on host until M4.2. So while the
>   ocean kernels around it are on device, each step does `sync_host(uv_rhs/ssh_rhs)` before CG
>   and `sync_device` after `update_vel`. State this in the M2 acceptance.
> - **GM/Redi scope:** GM runs **by default** in the C step (off only via `FESOM_NO_GMREDI=1`).
>   It IS in the conversion sequence (Task M2.5b). The **first GPU climate target (M3) is run
>   GM-off** (`FESOM_NO_GMREDI=1`) against a GM-off reference — this reuses the C port's existing
>   byte-identity off-switch invariant and shrinks the M3 surface; GM-on climate follows.

### Task M2.1: EOS / pressure_bv / sw_alpha_beta (first GPU-math test)

**Files:**
- Modify: `src/fesom_eos.cpp` (`fesom_pressure_bv`, `fesom_compute_sw_alpha_beta`),
  `src/fesom_step.cpp` (dispatch + verify hook)

- [x] `fesom_pressure_bv_kk`: `parallel_for` over owned nodes, level loops inside the lambda;
      JM-EOS core as a `KOKKOS_INLINE_FUNCTION` (`Kokkos::sqrt`), per-column temporaries lambda-local
- [x] `fesom_compute_sw_alpha_beta_kk` (`Kokkos::fabs`; pure polynomial map)
- [x] `FESOM_KK_VERIFY=eos`: runs the host C twin alongside the device kernel on the same live
      state, reports per-field max|Δ|, asserts `max|Δ|==0` on Serial; non-intrusive (restores the
      KK result). **All 20 pi steps `max|Δ|==0`** (density/hpressure/bvfreq/dbsfc/sw_α/sw_β + MLD1)
- [x] C twins (`fesom_pressure_bv`/`fesom_compute_sw_alpha_beta`/`fesom_eos_jm_components`) kept
      in-tree, untouched, dead-but-diffable until M2 closes (still used at `fesom_main.cpp:468` init)
- [x] SYNC_MAP §1 rails wired (driver IN `modify_host+sync_device` on T/S/hnode; kernel `mod_dev`
      outputs; driver `sync_host` all 7 before halos; halos + smoother via `h_checked`); `docs/SYNC_MAP.md`
      substep-1 row updated; `-ffp-contract=off` adopted + golden re-verified (D18/L23, commit `1f0a5e4`)
- [x] **verified**: Serial pi == golden (bit-identical, full 20-step `diff_snap`); **OpenMP also
      bit-identical** (pure map, no reduction — beats the ≲1e-12 budget); `ctest` 4/4; np=2 (CMA-off)
      == `…m13_nocma` oracle; SYNCCHECK build clean (guard now does real work) + bit-identical;
      **CUDA (A100) builds + runs + climate-close** (density Δ≈3e-12 stable; Av/Kv ≈0.095 isolated
      threshold-flips, bounded; no blow-up — the expected first CUDA divergence, D5) — → M2.2

### Task M2.2: PP mixing (`compute_vel_nodes` gather, `pp_mixing`, `mo_convect`)

**Files:**
- Modify: `src/fesom_pp.cpp`, `src/fesom_step.cpp`

- [x] `fesom_compute_vel_nodes_kk`: `parallel_for` over owned nodes, the inner gather over
      `nod_in_elem2D` accumulates into lambda-local `tx/ty/tvol` then writes only this node's
      `uvnode` slots → race-free; the private per-node reduction keeps the C accumulation order so
      **Serial AND OpenMP are bit-identical** (a private reduce, not a cross-thread one)
- [x] `fesom_pp_mixing_kk`: the 3 loops as **3 separate `parallel_for` launches** — the launch
      barrier preserves the ⚠️ loop-2-before-loop-3 ordering (Av from `Kv`-as-factor² before Loop 3
      cubes `Kv` in place), **D20**; `fesom_mo_convect_kk`: convective-adjustment maxes (race-free)
- [x] `FESOM_KK_VERIFY=pp` Serial `max|Δ|==0` all 20 pi steps — KPP path = compute_vel_nodes +
      mo_convect (default), PP path (`FESOM_MIX_SCHEME=PP`) adds pp_mixing. compute_vel_nodes/
      pp_mixing are EOS-style; mo_convect read-modify-writes its inputs so the driver captures the
      pre-kernel `Kv/Av` for the C-twin oracle (L26). Key match guards `"pp"`⊂`"kpp"` (L25)
- [x] SYNC_MAP §2 row-3 rails wired (driver IN `modify_host()+sync_device()`, kernel `mod_dev`,
      driver `sync_host()` before halos via `h_checked`); ⚠️ `mo_convect` IN rail does
      `modify_host()+sync_device(bvfreq)` — first device read AFTER the host `smooth_nod3D` (L14/L27)
- [x] **verified** (commit `17ea075`): Serial pi (KPP) == golden bit-identical (np=1 + np=2 CMA-off);
      OpenMP == golden; `ctest` 4/4; **SYNCCHECK clean + bit-identical on BOTH KPP and PP branches**;
      **CUDA (A100) builds + runs + climate-close** (density 3.18e-12 stable, Av/Kv ≈0.095 isolated
      threshold-flips, u/v ≈1e-4 drift — the M2.1 budget D5; no new divergence class) — → M2.3

### Task M2.3: KPP vertical mixing (the large mixing kernel)

**Files:**
- Modify: `src/fesom_kpp.cpp` (1046 LoC), `src/fesom_step.cpp`

- [x] **wrapped all 15 KPP scratch arrays in `Field`** (`diffK/viscA/blmc/ghats/dVsq/dkm1/hbl/bfsfc/
      caseA/stable/ustar/Bo/kbl/wmt/wst`) — M2.3a (`7c55255`), the D12/D16 pattern; pure data-layer,
      bit-identical. `wmt/wst` set-once → one-shot device push in `fesom_kpp_init`
- [x] **ported `ri_iwmix`/`bldepth`/`blmix`/`enhance`/`combine`/`viscAE` + the `prestep` as
      node/elem-parallel `parallel_for`s** (M2.3b, `61a4816`); `kpp_wscale` → a templated
      `KOKKOS_INLINE_FUNCTION` over the `wmt/wst` device Views; `Kokkos::` math; bvfreq smoothing +
      bottom-pad preserved verbatim. Multi-launch where an inter-stage dependency needs the barrier (D20)
- [x] ⚠️ KPP's **internal exchanges = 2 bracket points** (7 `fesom_exchange_nod3D`): `smooth_blmc`
      (blmc×3 + 3-sweep smoother) + the pre-elem-average (diffK×2/ghats/viscA). Each bracketed
      `modify_device→sync_host→halo+smooth (h_checked)→modify_host→sync_device` **inside**
      `fesom_kpp_mixing_kk` (D21 — the kernel owns its exchanges' syncs; the driver owns IN/OUT)
- [x] `FESOM_KK_VERIFY=kpp` Serial `max|Δ|==0` all 20 pi steps (Kv, Av) — bit-identical first complete
      run (L29). IN rail syncs ALL 11 inputs explicitly (L28: the Serial gate can't catch a missing one)
- [x] **verified** (`61a4816`): Serial pi (KPP) == golden (np=1 + np=2 CMA-off); OpenMP == golden;
      `ctest` 4/4; **SYNCCHECK clean + bit-identical** (internal-bracket + Kv/Av-halo guards transition
      `Device→Synced`); **CUDA (A100) builds + runs + climate-close** at the unchanged M2.1 budget
      (Av/Kv ≈0.095 same threshold-flip nodes; no new divergence class, D5) — → M2.4

### Task M2.4: PGF + momentum RHS + viscosity + implicit vertical viscosity

**Files:**
- Modify: `src/fesom_aux.cpp` (PGF), `src/fesom_momentum.cpp` (`compute_vel_rhs`,
  `visc_filt_bidiff`, `impl_vert_visc`), `src/fesom_step.cpp`

- [x] port `pressure_force_linfs_fullcell` (`fesom_pressure_force_linfs_fullcell_kk`, M2.4a `07094b5`):
      clean per-element map, EOS-style. INPUT rail re-pushes `hpressure` (the L27/L30 device→host(halo)→
      device hand-off — reads it at the element's 3 HALO vertices). `FESOM_KK_VERIFY=pgf`
- [x] port `compute_vel_rhs` (M2.4d `4e5c8ba`) — **AB2 `eps=0.1` preserved** (the dt=1800 trap). The
      MOST composite kernel: it embeds `momentum_adv_scalar_kk` (an edge→node **scatter** via
      `atomic_add` D22 + an internal `uvnode_rhs` halo D21) between two race-free element maps. 7
      `parallel_for`s + 1 halo bracket. `FESOM_KK_VERIFY=vrhs` (capture-before on `uv_rhsAB`)
- [x] port `visc_filt_bidiff` (M2.4c `5fde72c`): biharmonic ∇⁴, the **first SCATTER kernel** — two
      edge→element stages via `Kokkos::atomic_add` (**D22**, `docs/SCATTER_STRATEGY.md`) around an
      internal `Uc/Vc` elem3D halo (D21). `FESOM_KK_VERIFY=vfilt` (capture-before, full extent)
- [x] port `impl_vert_visc` (M2.4b `8a419e0`, per-element TDMA: parallel over element, the Thomas
      sweep sequential in level inside the lambda, per-column `[64]` scratch). Race-free → Serial AND
      OpenMP bit-identical. `FESOM_KK_VERIFY=ivisc` (capture-before, read-modify-write `uv_rhs`)
- [x] **left `compute_ssh_rhs_linfs`, `ssh_solve_cg`, `update_vel`, `compute_hbar` on host**
      (steps 7–10) — the mid-step round-trip is bridged: substep 6's OUT `sync_host(uv_rhs)` feeds the
      host CG (steps 7–10); the next step's substep-3 `compute_vel_nodes` IN rail re-`sync_device`s `uv`
      after the host `update_vel`. No extra code needed (the existing per-kernel rails compose) — SYNC_MAP §5
- [x] `FESOM_KK_VERIFY` Serial `max|Δ|==0` for each (all 20 pi steps; **all 7 M2 keys simultaneously =
      160 lines, 0 non-zero**); backends verified — Serial pi == golden (np=1 + np=2 CMA-off vs
      `…m13_nocma`); **OpenMP climate-close** (max |Δ|=8.3e-25 in `v`, the first non-bit-identical OpenMP
      — the expected D22 scatter regime change, ≪ the ≲1e-12 budget); SYNCCHECK clean + bit-identical;
      CUDA builds + pi smoke climate-close. → M2.5

### Task M2.5: ALE thickness / vertical velocity / CFLz / wsplit

**Files:**
- Modify: `src/fesom_ale.cpp`, `src/fesom_step.cpp`

- [x] port `ale_thickness_linfs`, `ale_vert_vel_linfs` (+ `fer_w` accumulator), `compute_cflz`,
      `compute_wvel_split` (preserve `use_wsplit=.false.` behaviour), `commit_thickness` (commit
      `d6937f1`). Shapes (L34): thickness/commit/cflz/wvel_split = race-free maps (bit-identical
      Serial+OpenMP); `vert_vel` = edge→node SCATTER (`atomic_add`, D22) + per-node level cumsum.
      ⚠️ **GM is ON in pi** (not off as the handoff said) → the `fer_w` accumulator is LIVE; the `gm_on`
      branch ported verbatim + `if(gm)` `fer_uv`/`fer_w` rails keep pi==golden. No internal halo (no D21).
- [x] `FESOM_KK_VERIFY=ale` Serial `max|Δ|==0` (all 5 kernels, 20 steps, 0 non-zero); Serial pi ==
      golden (np=1 + np=2 CMA-off); `ctest` 4/4; SYNCCHECK clean + bit-identical; **OpenMP** = 4 maps
      bit-identical + `vert_vel` climate-close (~1e-21, D22); **CUDA (A100)** climate-close at the
      **unchanged M2.1/M2.4 budget** (density 3.18e-12, Av/Kv 0.095 flips, u/v 3.7e-4/5.9e-5, no new
      divergence class, D5) — → M2.5b

### Task M2.5b: GM/Redi (sigma_xy, neutral_slope, streamfunction, bolus, horizontal Redi)

**Files:**
- Modify: `src/fesom_gm.cpp` (1077 LoC), `src/fesom_step.cpp`

- [x] wrap GM scratch arrays in `Field` (all 11: `sigma_xy/neutral_slope/slope_tapered/fer_tapfac/
      fer_gamma/fer_K/Ki/fer_C/fer_scal/tr_xy/tr_z`) — deferred from M1 (commit `8645824`, M2.5b-a,
      bit-identical; `*g = fesom_gm{}` not memset, D13/L13)
- [x] port `compute_sigma_xy`, `compute_neutral_slope`, `init_redi_gm`, `fer_solve_gamma`
      (per-node TDMA, L31), `fer_gamma2vel`; ODM95 tapering + `scaling_GMzexp` verbatim (commit
      `ab57fd6`, M2.5b-b). 5 `_kk` twins, all race-free maps/gathers/TDMA → Serial AND OpenMP
      bit-identical; C twins' halos move to the driver (ALE pattern), only `fer_gamma` re-pushed (L30/L35)
- [x] port `diff_ver_part_redi_expl` + `diff_part_hor_redi` (substep-13 Redi) on device — commit
      `4bccd69`, M2.5b-c. `diff_hor` = edge→node scatter (atomic_add, D22) + 5 partial-cell branches;
      both own their `tr_xy`/`tr_z` internal halo (D21). ⚠️ `feedback_bolus_divergence_balance` honoured
      (C verbatim, no per-cell clamp). **The bolus add/sub STAY ON HOST** — no device consumer of the
      bolus-augmented `uv`/`w`/`w_e` until the FCT moves to device (M2.6); deferred there (L36)
- [x] **gate two ways:** `FESOM_NO_GMREDI=1` Serial runs clean + differs from the GM-on golden (GM
      live, L34) — off-path byte-identical by construction (all changes `if(gm)`-guarded);
      `FESOM_KK_VERIFY=gm` Serial `max|Δ|==0` (140 lines = 5 chain + 2 redi × 20 steps)
- [x] backends verified: Serial pi == golden (np=1 + np=2 CMA-off vs `…m13_nocma`); `ctest` 4/4;
      SYNCCHECK clean + bit-identical; OpenMP bit-identical (chain + redi; whole-run only the M2.5
      vert_vel `w`-scatter floor ≈3.4e-21, no new class); CUDA builds + climate-close at the unchanged
      M2.1/M2.4/M2.5 budget (density 3.18e-12, u/v 3.8e-4/7.4e-5, pgf 6-8e-18, no new divergence). → M2.6

### Task M2.6: FCT tracer advection (the big one — scatter decision)

**Files:**
- Modify: `src/fesom_tracer_adv.cpp` (1301 LoC), `src/fesom_step.cpp`
- Create: `docs/SCATTER_STRATEGY.md`

- [x] wrap FCT scratch arrays in `Field` (all 12: `adv_flux_hor/adv_flux_ver/del_ttf_adv{horiz,vert}/
      fct_LO/fct_ttf_{min,max}/fct_plus/fct_minus/fct_aux/tr_xy/edge_up_dn_grad`) — M2.6-a (`d210025`),
      `*sc = T{}` not memset (D13/L13), bit-identical
- [x] port the MFCT 3rd-order horizontal + vertical FCT pipeline — M2.6-b: `fesom_tracer_advect_one_fct_kk`
      (~24 launches + 3 D21 internal-exchange brackets in ONE fn). ⚠️ MFCT element gradient from `values`
      (`feedback_mfct_gradient_from_values`) while the MFCT flux uses `valuesAB`; ⚠️ tracer AB2 `eps=0.1`
      (`oce_tracer_mod.F90:53`, init_AB). a3+a4 fused to column-local scratch (no `[N*nl]` tvert). Verify
      `tradv` = L26 capture-before on BOTH `values` and `valuesold` (L37)
- [x] ⚠️ **flux assembly is edge→node scatter** — M2.6-b: 3 scatters (`compute_fct_LO` divergence,
      Zalesak `fct_plus/minus`, `flux2dtracer` horizontal) via `Kokkos::atomic_add` in natural edge
      order (Serial bit-identical, OpenMP/CUDA climate-close — added no new class on pi; edge-coloring
      GPU-only). Recorded in `docs/SCATTER_STRATEGY.md` (D22) — first written M2.4, extended here
- [x] `FESOM_KK_VERIFY=tradv` Serial `max|Δ|==0` (40 lines, 0 non-zero); backends — Serial pi == golden
      (np=1 + np=2 CMA-off); `ctest` 4/4; SYNCCHECK clean + bit-identical; OpenMP climate-close (unchanged
      M2.5 budget); CUDA builds + climate-close. **+ M2.6-c: move bolus add/sub to device.** → M2.7

### Task M2.7: Tracer diffusion (implicit vertical + Redi K33) + M2 acceptance

**Files:**
- Modify: `src/fesom_tracer_diff.cpp`, `src/fesom_step.cpp`

- [ ] port `impl_vert_diff_tracers` (per-node TDMA) + surface flux BC + the salinity floor
- [ ] `FESOM_KK_VERIFY=trdiff` Serial `max|Δ|==0`
- [ ] **M2 acceptance**: full ocean step (minus CG/reductions/ice) on device; Serial 1-yr CORE2
      bit-identical to C; CUDA 1-yr runs (slow) and is physical. Tag `m2-ocean-device` — before M3

## ───────────── M3 · Climate validation on GPU ─────────────
*Goal: prove the CUDA build reproduces the Fortran climate within the Fortran↔C budget.*

### Task M3.1: GPU run configuration (multi-GPU MPI mapping)

**Files:**
- Create: `job_gpu_core2_*` SLURM templates, `docs/RUN_GPU.md`

- [ ] map MPI ranks → GPUs (1 rank/GPU, 4/node on `gpu`); pick `dist_<#gpus>` partitions;
      set `Kokkos` device-per-rank (local-rank → `cudaSetDevice` via `Kokkos` `--kokkos-device-id`)
- [ ] short multi-GPU smoke on `gpu-devel`; document in `docs/RUN_GPU.md`

### Task M3.2: 2-yr + 5-yr GPU climate validation

- [ ] CUDA 2-yr CORE2 vs `fortran_pp_2yr` (+ KPP ref): `clim_validation_2yr.py` — SST/SSS RMS
      within the Fortran↔C budget (~0.005–0.04)
- [ ] CUDA 5-yr stability (`drift_5yr_d1800.py`): completes, `max uv < 5`, drift non-growing
- [ ] document the **GPU↔CPU difference budget** (expect ≈ Fortran↔C) in `docs/GPU_FIDELITY.md`
- [ ] tag `m3-gpu-climate`

## ───────────── M4 · Reductions + CG SSH solver + sea ice on device ─────────────

### Task M4.1: Global reductions / diagnostics → `parallel_reduce`
**Files:** Modify `src/fesom_main.cpp`, `src/fesom_aux.cpp`, diagnostics
- [ ] port `ocean_area` and per-step stat reductions; Serial bit-identical, OpenMP/GPU climate-close
- [ ] verify

### Task M4.2: SSH RHS + CG solver + velocity update + hbar on device (closes the mid-step gap)
**Files:** Modify `src/fesom_ssh.cpp`, `src/fesom_step.cpp`
- [ ] wrap CG vectors + stiffness CSR in `Field` (`r/p/Ap`, matrix) — deferred from M1
- [ ] port `compute_ssh_rhs_linfs`, the CG (`SpMV + axpy + dot` via `parallel_reduce` +
      `MPI_Allreduce`), `update_vel`, `compute_hbar` — this removes the M2 mid-step host
      round-trip; Serial `max|Δ|==0`; document the GPU non-determinism source (dot-product order)
- [ ] verify a year on Serial (bit-identical) + CUDA (climate-close)

### Task M4.3: Sea ice (EVP / thermo / FCT) on device
**Files:** Modify `src/fesom_ice_evp.cpp`, `src/fesom_ice_thermo.cpp`, `src/fesom_ice_fct.cpp`, `src/fesom_ice.cpp`
- [ ] port EVP subcycle kernels, thermodynamics, ice FCT (same scatter decision as M2.6)
- [ ] `FESOM_KK_VERIFY` per kernel; **M4 acceptance**: whole model on device; 2-yr+5-yr re-validated;
      tag `m4-full-device`

## ───────────── M5 · Performance (expand after M4) ─────────────
*Coarse now; detail when reached.*
- [ ] per-field layout flip for hot fields whose kernels are all on-device; re-validate (Serial
      still bit-identical — only memory order changes, not arithmetic order). ⚠️ For a **1-D flat
      `Field`, LayoutLeft==LayoutRight (no-op for coalescing)** — the interleaved 2-component
      (`uv/uv_rhs/tr_xy` via `FESOM_ELEMVEC`) and 3-D node fields need a **rank change** to
      `View<double**[2]>`/`View<double**>`, and the `FESOM_NODE3D/ELEM3D/ELEMVEC` macros must
      become **layout-agnostic accessor functions** over the `View`. Extend the same
      layout-agnostic-accessor work to halo pack/unpack + I/O gather
- [ ] data residency: keep state on device across steps; sync host only for halo (pre-GPU-MPI) + I/O cadence
- [ ] GPU-aware MPI: device-pointer halo exchange (CUDA-aware OpenMPI); on-device pack/unpack kernels
- [ ] profiling + tuning (`TeamPolicy`/hierarchical over levels where it pays); tag `m5-perf`

## ───────────── M6 · HIP / LUMI bring-up (expand after M5) ─────────────
*Coarse now; should be config-only if Kokkos-purity held.*
- [ ] build Kokkos + model with HIP (`Kokkos_ARCH_AMD_GFX90A`) on LUMI; fix any surfaced non-portability
- [ ] ROCm-aware MPI; LUMI run configs (`docs/RUN_LUMI.md`)
- [ ] AMD climate-close acceptance (2-yr vs Fortran); tag `m6-lumi`

### Task FINAL-1: Verify acceptance criteria
- [ ] Serial == C bit-for-bit (smoke + 1-yr); OpenMP climate-identical; CUDA & HIP climate-close
- [ ] all `FESOM_KK_VERIFY` kernels pass; `ctest` green; purity check green
- [ ] 2-yr + 5-yr climate within budget on CPU and both GPUs

### Task FINAL-2: Documentation
- [ ] update `README.md` (build matrix, backends, run recipes), `docs/BUILD.md`, `docs/RUN_*.md`
- [ ] write/refresh `CLAUDE.md` with the Kokkos patterns + the bit-identity gate workflow
- [ ] review/finalize `docs/KOKKOS_PORTING_LESSONS.md` — it should already be complete (kept
      current every session); distill it into a reusable "Fortran→C→Kokkos playbook" for the
      next component port
- [ ] move this plan to `docs/plans/completed/`

## Technical Details

- **`Field` (`fesom_field.hpp`)**: `DualView<double*, LayoutRight>` + a host raw-ptr alias so
  legacy `entity*nl+lev` macro indexing is unchanged. `int` fields get an int-typed twin.
  M5 swaps `LayoutRight` → space-default per hot field (no Serial arithmetic change; GPU coalescing).
- **Sync model**: orchestrated in `fesom_step`/`fesom_ice`/`main`, mirroring the existing
  halo-exchange map (`docs/SYNC_MAP.md`), **per-substep including intra-kernel exchanges**
  (KPP's 6, the FCT pipeline). Coarse `sync_device(inputs)→kernel→modify_device(outputs)`;
  `sync_host` before halo/I/O/legacy. The SSH CG sits mid-step → a mandatory `sync_host`/`sync_device`
  bracket around it until M4.2. `FESOM_KK_SYNCCHECK` = per-`Field` authoritative-space tag +
  a checked `h_checked()` accessor on host entry points (not just DualView flags).
- **Determinism rule**: any reordering of a physics sum (edge-coloring) is **GPU-only**; the
  Serial path keeps natural order so `max|Δ|==0` vs C holds (coloring on Serial would also
  violate the no-simplification rule).
- **Per-kernel gate** (`FESOM_KK_VERIFY=<name>`): both C twin and Kokkos kernel run on the same
  live state; `max|Δ|` over all entities/levels printed + asserted `==0` on Serial.
- **Why GPU can't be bit-identical**: fma contraction, libdevice/ROCm transcendental ULPs,
  `parallel_reduce` tree order, scatter atomics — all expected; acceptance is climate-close.
- **Build**: `-DFESOM_BACKEND={Serial,OpenMP,Cuda,Hip}`; host builds add `-ffp-contract=off`
  for the Serial-vs-C diff. Kokkos from source (no Levante module).

## Post-Completion
*External / long-running — no checkboxes.*

**Manual verification / HPC jobs:**
- 2-yr & 5-yr climate runs are multi-hour SLURM jobs (CPU 864-rank and multi-GPU) — run on
  `compute`/`gpu`, compare with `scripts/*.py` against `fortran_pp_2yr` and the KPP ref.
- Multi-GPU rank↔device mapping and CUDA/ROCm-aware MPI need testing on real `gpu` nodes.

**External systems:**
- LUMI access + module environment (ROCm, Cray MPICH GPU-aware) for M6.
- Pre-generated mesh partitions for the GPU rank counts actually used (`dist_<#gpus>`).
- Optional: upstream the performance-portable port back to `koldunovn/fesom_port` or a new repo.
