# FESOM2 C → C++/Kokkos port — session handoff

**Session 5 (2026-05-26) — M1.4 complete (commit `076a56d`).** Repo: `/home/a/a270088/port_kokkos` (git). Read this first, then
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
- **M1.4 DONE** (commit `076a56d`): **all 12 `fesom_forcing` + 49 `fesom_ice` persistent arrays are
  now `Field`-backed** (top-level 19 + `data[3]`×6 + `work`×15 incl. `fct_massmatrix` + `thermo`×9;
  same alias pattern, D16). Embedded-by-value sub-structs + the `data[3]` array reset/release
  recursively via one `*ice = fesom_ice{}`; `fct_massmatrix` migrated at its lazy foreign call site
  in `fesom_ice_fct.cpp`. **Bit-identical on the first gate run**: Serial np=1 == golden, np=2
  (CMA-off) == `…m13_nocma` oracle (exercises scatter+halo on Field-backed forcing/ice + EVP/FCT),
  ctest 4/4, **CUDA np=1 (A100) == golden**. New decision D16, lesson L20. **This completes the M1
  persistent-state migration** (mesh+dyn+aux+tracers+forcing+ice = 28+37+61 = 126 arrays); only the
  gm/kpp/ocean-tradv/ssh per-kernel scratch remains (deferred to its M2/M4 kernel task).
- **NEXT: M1.5** — sync discipline in the step driver + M1 acceptance (`docs/SYNC_MAP.md`; 1-yr
  CORE2 bit-identical on Serial+OpenMP+CUDA; tag `m1-datalayer`). At M1 all compute is still host
  → the map starts "host authoritative"; this task lays the rails for the M2 device kernels.

## 1. Git state

```
HEAD     076a56d M1.4: migrate fesom_forcing + sea-ice to fesom::Field (61 arrays)   (+ this handoff commit on top)
1c444d8  docs: phantom-multi-rank-divergence debugging ladder + np=1-not-sufficient
6d00b2b  docs: handoff → M1.3 done / next M1.4
d42c7cc  M1.3: migrate fesom_dyn/aux/tracers to fesom::Field (37 arrays); resolve np=2 vader-CMA gate artifact
e2dc45e  M1.2 Wave 3: migrate bc_index_nod2D — fesom_mesh fully Field-backed; M1.2 complete
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

## 3. THE NEXT TASK — M1.5: sync discipline in the step driver + M1 acceptance

Per plan §M1.5. **This is the last M1 task and it's a DIFFERENT KIND of task** — not array
migration (M1.2–M1.4 already backed all 126 persistent arrays with `Field`), but laying the
host↔device **sync rails** the M2 device kernels will use, plus the M1 acceptance run + tag.
Deliverable = `docs/SYNC_MAP.md` (the per-substep host/device currency map, mirroring the halo map).

- **Lay the coarse sync rails** in `src/fesom_step.cpp` / `fesom_ice.cpp` / `fesom_main.cpp`: the
  contract is *before a (future) device kernel `sync_device()` its inputs; after it `modify_device()`
  its outputs; before halo/I/O/legacy-host-kernel `sync_host()`*. **At M1 all compute is still on
  host**, so the map is "host authoritative" everywhere and these calls are no-ops on Serial/OpenMP
  (host==device) and bitwise-exact `deep_copy`s on CUDA — so **the run must STAY bit-identical**.
  Recall L14: a host write through the raw alias is invisible to the DualView flags, so a real
  `sync_device` needs a preceding `modify_host()`; the M1.2 mesh did this once in
  `mesh_sync_geometry_device`. The set-once geometry is already synced; this task decides the
  cadence for the *evolving* state (dyn/tracers/forcing/ice) — likely a documented "synced lazily by
  the first M2 kernel that needs it" rather than eager per-step copies.
- **Prove the plumbing**: exercise a no-op device round-trip each step under `-DFESOM_KK_SYNCCHECK`
  (the `Auth` tag + `h_checked()` already exist in `fesom_field.hpp`) to assert host/device coherence
  without yet moving compute. Route a few host entry points (halo pack/unpack, I/O gather) through
  `h_checked()` as the proof.
- **Write `docs/SYNC_MAP.md`** — the deliverable/test for this task.
- **M1 acceptance**: Serial + OpenMP + CUDA all run a **1-yr CORE2** bit-identical to the C reference
  (compute still on host) — this is a multi-hour SLURM job (CORE2 mesh `/pool/data/AWICM/FESOM2/
  MESHES_FESOM2.1/core2`, account `ab0995`); the pi smoke is NOT sufficient for acceptance. Then
  **tag `m1-datalayer`** → M2 (first device compute kernel, M2.1 EOS, where CUDA bit-identity is
  expected to FIRST break — fma/transcendentals).

**Gate (recipe §2) for every code change in M1.5: Serial np=1 == golden + np=2 (CMA-off!) ==
`pi_np2_ref_m13_nocma` oracle + `ctest` 4/4; then CUDA np=1 smoke (`sbatch jobs/job_pi_smoke_gpu`)
bit-identical.** (The 1-yr CORE2 acceptance is the milestone gate, separate from the per-change pi gate.)

**Invariant for all of M1:** the only device op is `deep_copy` of `double`/`int` (bitwise-exact) and
NO compute kernel runs on device → Serial+OpenMP+CUDA all stay bit-identical. A stray device
`parallel_for`/fill is a bug. (M1.5 ADDS `sync_*` calls but still NO `parallel_for`.)

**The M1.2–M1.4 migration pattern (for the M2/M4 per-kernel scratch arrays still to come):**
`Field`/`IntField` member per persistent array; legacy raw ptr = **non-owning `field.h()` alias**
re-pointed right after `field.alloc(label, n)` (count in **elements**) — do NOT touch call sites
(D12). `calloc/malloc`→`.alloc`; `memset(s,0,sizeof)`→`*s = T{}` (D13/L13); drop per-array `free()`
(the assignment releases every Field). First **audit** (D15/D16): the owning struct must be a
stack/`new` object (Field ctors run; `malloc` skips them, L13), and the array must be updated
in-place (no pointer swaps) so the alias stays valid. Embedded-by-value sub-structs + arrays-of-
structs reset/release recursively via one `*x = T{}` (D16/L20). Mind L17 (nvcc rejects `.c` sources
— don't pull `fesom_field.hpp` into a CUDA-excluded `.c` unit test).

**Always:** gate every step on Serial bit-identity vs the C twin; never simplify physics; and
**append every decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` in the same commit**.

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` · Lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D16,
  L1–L20) · C-port lessons: `docs/PORTING_LESSONS.md` · Build: `docs/BUILD.md` · Provenance/golden +
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
> `git log --oneline -8` to orient. **M0 + M1.1 + M1.2 + M1.3 + M1.4 are complete**: Serial+CUDA
> builds are bit-for-bit identical to the C golden; `fesom::Field` (DualView wrapper) exists; and
> **ALL 126 persistent arrays (28 `fesom_mesh` + 37 `fesom_dyn`/`aux`/`tracers` + 12 `fesom_forcing`
> + 49 `fesom_ice`) are now `Field`/`IntField`-backed** with the raw pointer kept as a non-owning
> `field.h()` alias (0 call sites changed). The M1 persistent-state data-layer migration is DONE;
> only per-kernel scratch (gm/kpp/ocean-tradv/ssh) remains, deferred to its M2/M4 kernel task. Serial
> np=1 + np=2 (CMA-off) + CUDA np=1 all ALL FIELDS BIT-IDENTICAL; ctest 4/4. Work tree clean
> (HEAD = M1.4 commit `076a56d` + handoff commit).
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status, build/run recipes §2 incl. the **np=2 vader-CMA gate fix**, the M1.5 task §3)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M1.5; §M1.2–§M1.4 are ticked with done-notes)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D16, L1–L20) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps (dt=1800 AB2 eps, **tracer stride nl**, halo bounds)
> - `/home/a/a270088/port_kokkos/src/fesom_field.hpp` + `tests/test_field.cpp`  ← the `Field` type (incl. the `Auth`/`h_checked()` SYNCCHECK mechanism M1.5 uses) + GPU smoke `jobs/job_pi_smoke_gpu`
> - `/home/a/a270088/port_kokkos/src/fesom_{mesh,dyn,aux,tracers,forcing,ice*}.{h,cpp}`  ← **worked examples of the migration pattern** (Field member + raw alias = `field.h()`; `*x = T{}`); `mesh_sync_geometry_device` is the existing `modify_host()`+`sync_device()` example (L14)
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/`
>
> GOAL: **M1.5 — sync discipline in the step driver + M1 acceptance** (per plan §M1.5 + handoff §3).
> NOT array migration (that's done). Lay the coarse host↔device **sync rails** in
> `fesom_step.cpp`/`fesom_ice.cpp`/`fesom_main.cpp` (before a future device kernel `sync_device()`
> inputs; after, `modify_device()` outputs; before halo/I/O/legacy-host `sync_host()`) — at M1 all
> compute is still host so the map is "host authoritative" everywhere and the run **must stay
> bit-identical** (no-op on Serial/OpenMP, bitwise `deep_copy` on CUDA; recall L14: a real
> `sync_device` needs a preceding `modify_host()`). Exercise a no-op device round-trip per step under
> `-DFESOM_KK_SYNCCHECK` to prove coherence (route halo/I/O host reads through `h_checked()`). Write
> the deliverable **`docs/SYNC_MAP.md`**. Then the **M1 acceptance**: Serial+OpenMP+CUDA each run a
> **1-yr CORE2 bit-identical** to the C reference (a multi-hour SLURM job — the pi smoke is NOT
> sufficient), and **tag `m1-datalayer`**. NO `parallel_for` yet (the first device compute kernel is
> M2.1, where CUDA bit-identity is expected to first break).
>
> GATE every code change (recipe §2): `mkdir -p /tmp/pi_check`, Serial pi smoke
> `./build-serial/fesom_port .../meshes/pi /tmp/pi_check 100 20 10` then `…/nereus/bin/python
> scripts/diff_snap.py docs/reference/c_baseline_snapshots/pi /tmp/pi_check` must print ALL FIELDS
> BIT-IDENTICAL; `ctest` 4/4; **the np=2 gate — `export OMPI_MCA_btl_vader_single_copy_mechanism=none`
> first (L18!) — vs `/scratch/a/a270088/pi_np2_ref_m13_nocma`** (`diff_snap.py` takes DIRECTORIES, L19);
> CUDA bit-identity via `sbatch jobs/job_pi_smoke_gpu` (build `--target fesom_port`, L17). The 1-yr
> CORE2 run is the separate M1-acceptance milestone gate.
>
> ⚠️ If a gate "diverges": first rule out the **vader-CMA artifact (L18 / the §C debugging ladder)** —
> rebuild after `touch src/*` (kill stale `.o` after any layout change), then dump the OWNED state
> right after the producing kernel and `cmp` across builds; if byte-identical, the divergence is in
> the snapshot `MPI_Gatherv`/transport, not the port (cost the M1.3 session hours).
>
> INVARIANTS: M1 moves NO compute to the device (deep_copy of double/int only → all backends stay
> bit-identical; a stray device `parallel_for`/fill is a bug). Never simplify physics; preserve every
> constant/loop-bound verbatim (re-read PORTING_LESSONS before touching a kernel). Append every
> decision/lesson to KOKKOS_PORTING_LESSONS.md in the SAME commit; commit per milestone-step. C twin
> (the oracle): `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623); Fortran ground truth
> `/home/a/a270088/port2/fesom2/src`, run `/scratch/a/a270088/fortran_pp_2yr`. SLURM account `ab0995`;
> GPU partitions `gpu`/`gpu-devel`. First fresh checkout: `git submodule update --init --recursive`.
