# FESOM2 C → C++/Kokkos port — session handoff

**Session 6 (2026-05-26) — M1.5 complete + M1 acceptance PASSED → tagged `m1-datalayer`.** The
whole **M1 DualView data-layer milestone is DONE.** Repo: `/home/a/a270088/port_kokkos` (git,
branch `master`). Read this first, then `docs/plans/20260525-kokkos-port.md`,
`docs/KOKKOS_PORTING_LESSONS.md`, `docs/SYNC_MAP.md`, and the project memory in
`~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M0 + M1.1–M1.5 ALL COMPLETE.** The C++/Kokkos build is bit-for-bit identical to the C golden on
  **Serial, OpenMP, and CUDA**, and now also on a **full 1-yr CORE2 run** (Serial + OpenMP). At M1
  there are **no device compute kernels** — Kokkos is initialised, the only device op anywhere is
  `Kokkos::deep_copy` of `double`/`int` (bitwise-exact), so even CUDA is bit-identical.
- **M1.1–M1.4 (data migration):** all **126 persistent state arrays** are `fesom::Field`/`IntField`
  (`Kokkos::DualView<T*, LayoutRight>`)-backed — 28 `fesom_mesh` + 37 `fesom_dyn/aux/tracers` + 12
  `fesom_forcing` + 49 `fesom_ice` — each with the legacy raw pointer kept as a non-owning
  `field.h()` alias re-pointed once after `alloc()` (0 call sites changed). Only per-kernel scratch
  (gm/kpp/ocean-tradv/ssh) remains, deferred to its M2/M4 kernel task.
- **M1.5 (sync rails) DONE** (commit `e393bc7`): the host↔device sync discipline for the step driver,
  WITHOUT moving any compute to the device. **Cadence = host-authoritative + LAZY device sync, no
  eager per-step copies (D17)** — the per-kernel `sync_device(in)/modify_device(out)/sync_host(before
  halo|I/O)` brackets are owned by each M2/M4 kernel task. M1.5 added only (a) `h_checked()` at
  representative halo + the whole I/O-gather host entry points (pointer-identical to the raw alias
  today) and (b) a **`-DFESOM_KK_SYNCCHECK`-only per-step host→device→host round-trip** in
  `fesom_step`/`fesom_ice`/`fesom_main` — both compiled out / no-op in production. Deliverable:
  **`docs/SYNC_MAP.md`** (the per-substep currency map). New: decision D17, lessons L21 (assert vs
  NDEBUG), L22 (the rails are bit-identical by construction).
- **M1 ACCEPTANCE PASSED** (`docs/M1_ACCEPTANCE.md`, commit `6f3f203` infra): a fresh **1-yr CORE2**
  C-twin reference (none existed — M0.1 deferred it) at `/scratch/a/a270088/m1_accept/cref` (360-day,
  dt=1800, 17280 steps, 256 ranks, monthly snaps), and **Kokkos Serial + OpenMP each reproduced it
  ALL FIELDS BIT-IDENTICAL across all 13 snapshots.** No M1 perf penalty (Serial 1566 s ≈ C twin
  1574 s — identical host code). **CUDA CORE2 deferred to M3.1** (needs the multi-GPU rank→device
  mapping); M1 CUDA does zero device compute, so its data-layer identity is mesh-size-independent and
  already proven on the pi smoke (A100, every milestone). **Tagged `m1-datalayer`.**
- **NEXT: M2.1** — the FIRST device compute kernel (EOS / pressure_bv / sw_alpha_beta). This is where
  CUDA bit-identity is **expected to first break** (fma contraction + libdevice transcendentals), and
  where the validation model shifts from whole-run bit-identity to the per-kernel `FESOM_KK_VERIFY`
  Serial `max|Δ|==0` gate + the GPU climate-close budget. See §3.

## 1. Git state

```
HEAD  <handoff> docs: handoff → M1 complete / next M2.1   (this commit)
      6f3f203   M1.5: add 1-yr CORE2 acceptance infrastructure (cref + Serial/OpenMP jobs, compare, README)
      e393bc7   M1.5: sync rails + SYNCCHECK plumbing proof + docs/SYNC_MAP.md (host-authoritative, lazy)
      efcfb4a   docs: handoff → M1.4 done / next M1.5
      076a56d   M1.4: migrate fesom_forcing + sea-ice to fesom::Field (61 arrays)
tag   m1-datalayer  → on this milestone (annotated; records the CORE2 acceptance result + CUDA disposition)
tag   m0-baseline   → M0 (Serial+OpenMP+CUDA pi bit-identical)
```
Oracles: pi golden `docs/reference/c_baseline_snapshots/pi`; np=2 scatter `/scratch/a/a270088/pi_np2_ref_m13_nocma`
(CMA-off, L18; old `…_m12` is CMA-tainted); **1-yr CORE2 `/scratch/a/a270088/m1_accept/cref`**. Work
tree clean. First checkout elsewhere needs `git submodule update --init --recursive`.

## 2. Build & run (full recipe in `docs/BUILD.md`; MPI caveat in `docs/reference/PROVENANCE.md`)

```bash
cd /home/a/a270088/port_kokkos
source ./env.sh                                   # gcc11 + openmpi + netcdf (Serial/OpenMP)
# --- Serial ---
cmake -S . -B build-serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON   # (already configured)
cmake --build build-serial -j 16
# pi smoke (login-node MPI override — UCX/IB unavailable off compute nodes, L4):
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL \
      HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
mkdir -p /tmp/pi_check                            # REQUIRED: missing out-dir → NetCDF "Permission denied" (L15)
./build-serial/fesom_port /home/a/a270088/port2/fesom2/test/meshes/pi /tmp/pi_check 100 20 10
/work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/diff_snap.py \
    docs/reference/c_baseline_snapshots/pi /tmp/pi_check        # ALL FIELDS BIT-IDENTICAL
( cd build-serial && ctest )                      # 4/4: calendar, io_stream_unit, io_config, field
# np=2 scatter gate — MUST disable vader CMA FIRST (L18!):
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mkdir -p /tmp/pi_np2 && mpirun -np 2 ./build-serial/fesom_port \
    /home/a/a270088/port2/fesom2/test/meshes/pi /tmp/pi_np2 100 20 10
/work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/diff_snap.py \
    /scratch/a/a270088/pi_np2_ref_m13_nocma /tmp/pi_np2         # ALL FIELDS BIT-IDENTICAL (DIRS, L19)

# --- SYNCCHECK diagnostic build (M1.5; separate Release dir so it still matches the golden) ---
cmake -S . -B build-synccheck -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON -DFESOM_KK_SYNCCHECK=ON
cmake --build build-synccheck -j 16
# run the pi smoke with it: must finish (no SYNCCHECK abort) AND stay bit-identical to the golden.

# --- CUDA (build ONLY the model target — nvcc slow; verify "Built target fesom_port" in the log, L17) ---
source /sw/etc/profile.levante; module --force purge
module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0 netcdf-c/4.8.1-gcc-11.2.0
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake --build build-cuda --target fesom_port -j 16          # (build-cuda already configured)
sbatch jobs/job_pi_smoke_gpu                                 # A100 pi smoke → runs/pi_check_cuda
#   then: diff_snap.py docs/reference/c_baseline_snapshots/pi runs/pi_check_cuda
```
⚠️ After any struct-LAYOUT change, `touch src/*` before building to kill stale `.o` (L18). The 4
build dirs (`build-serial`, `build-omp`, `build-cuda`, `build-synccheck`) are already configured.

**1-yr CORE2 acceptance** (the milestone gate; rerun if needed): `sbatch jobs/job_m1accept_{cref,serial,omp}`
then `scripts/m1_accept_compare.sh` — see `docs/M1_ACCEPTANCE.md` (incl. the §ranks same-rank-count
rule and the §CUDA / M3.1 deferral).

## 3. THE NEXT TASK — M2.1: EOS / pressure_bv / sw_alpha_beta (the FIRST device kernel)

Per plan §M2.1 + the M2 cross-cutting notes. **This is a different kind of task from all of M1** —
the first compute moved onto the device. Expect the validation ladder to change here:

- **Port `fesom_pressure_bv`** to a `parallel_for` over nodes with the level loop inside the lambda
  (`Kokkos::` math for the JM-EOS `pow`/`sqrt` — the first transcendental-portability check), then
  **`compute_sw_alpha_beta`**. Preserve every constant/loop bound verbatim (re-read
  `docs/PORTING_LESSONS.md` first). Keep the C twin in-tree (dead-but-diffable) until M2 closes.
- **The per-kernel gate replaces whole-run bit-identity:** add `FESOM_KK_VERIFY=eos` — run the C twin
  AND the Kokkos kernel on the same live state and assert `max|Δ|==0` on **Serial** (the in-binary
  analogue of `exp1_compare_bidiff.py`). Serial must stay `max|Δ|==0`; OpenMP `Δ≲1e-12`; **CUDA is
  expected to FIRST diverge here** (fma + libdevice ULPs) → acceptance becomes *climate-close*, not
  bit-identical.
- **Adopt `-ffp-contract=off` + RE-BASELINE the golden** (deferred from M0.3 explicitly to "when the
  first kernel lands"): the captured golden was built fma=fast; the kernel-gate determinism knob is
  `-ffp-contract=off`, so re-capture the pi golden (and note it in PROVENANCE) at that setting when
  M2.1 lands. This is the M2 determinism foundation — do it as part of M2.1.
- **Wire the SYNC_MAP §1/§2 rails for EOS** (now they go live): `sync_device(tracers T/S)` before the
  kernel; `modify_device(density_m_rho0, hpressure, bvfreq, sw_alpha, sw_beta)` after; `sync_host()`
  those 5 before their halo exchanges (the `h_checked()` guards at `fesom_step.cpp:78,80` will then
  bite a missing sync under `-DFESOM_KK_SYNCCHECK`). Wrap any EOS scratch in `Field`. Update the
  `docs/SYNC_MAP.md` row for substep 1 (drop `[H]`) in the same commit.
- **Gate**: per-change pi gate (§2) stays green on Serial (== re-baselined golden); `FESOM_KK_VERIFY=eos`
  Serial `max|Δ|==0`; CUDA builds + runs (now climate-close). Commit per step; append lessons.

(Optional, if asked: **M3.1 multi-GPU mapping** to unblock the CUDA CORE2 acceptance row — small
`--kokkos-num-devices`/local-rank→device work + re-run cref/serial at a GPU rank count, §CUDA.)

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` (you are entering §M2.1) · Kokkos lessons:
  `docs/KOKKOS_PORTING_LESSONS.md` (D1–D17, L1–L22) · Fortran→C traps: `docs/PORTING_LESSONS.md`
  (dt=1800 AB2 eps=0.1, tracer stride nl, halo bounds) · **Sync map: `docs/SYNC_MAP.md`** · Acceptance:
  `docs/M1_ACCEPTANCE.md` · Build: `docs/BUILD.md` · Provenance/MPI: `docs/reference/PROVENANCE.md`
- `Field`: `src/fesom_field.hpp` (incl. `h_checked()` + the `Auth` tag) · its test `tests/test_field.cpp`
  · migration worked examples `src/fesom_{mesh,dyn,aux,tracers,forcing,ice*}.{h,cpp}` ·
  `mesh_sync_geometry_device` (the one-shot `modify_host()+sync_device()` example, L14) · SYNCCHECK
  round-trips: end of `fesom_timestep` (`fesom_step.cpp`), `fesom_ice_step`, before `fesom_timestep` (`fesom_main.cpp`)
- C twin (the bit-identity oracle): `/home/a/a270088/port2/fesom2_port/src` (SHA `75de623`), built bin
  `…/build/fesom_port` · Fortran ground truth `/home/a/a270088/port2/fesom2/src`, ref `/scratch/a/a270088/fortran_pp_2yr`
- Meshes: pi `/home/a/a270088/port2/fesom2/test/meshes/pi` · CORE2 `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`
  (`dist_16/32/144/256/288/432/512`) · Kokkos submodule `externals/kokkos` (4.4.01) · SLURM `ab0995`;
  GPU `gpu`/`gpu-devel`.

## 5. NEXT-SESSION PROMPT (paste this verbatim)

> Continue the FESOM2 C→C++/Kokkos port in `/home/a/a270088/port_kokkos` (git; branch `master`).
> `git log --oneline -8` to orient. **M0 + ALL of M1 are COMPLETE and tagged `m1-datalayer`**: the
> C++/Kokkos build is bit-for-bit identical to the C twin on Serial/OpenMP/CUDA (pi) AND on a full
> **1-yr CORE2** run (Serial+OpenMP, ALL FIELDS BIT-IDENTICAL across 13 monthly snapshots). All 126
> persistent arrays are `fesom::Field`-backed; the M1.5 sync rails are laid (host-authoritative +
> lazy, D17; `docs/SYNC_MAP.md`) and proven via `-DFESOM_KK_SYNCCHECK`. CUDA CORE2 is deferred to
> M3.1 (multi-GPU mapping; M1 CUDA does no device compute → identity is pi-proven and mesh-independent).
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status §0, build/run §2 incl. SYNCCHECK + acceptance, the M2.1 task §3)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M2.1; all of M1 is ticked)
> - `/home/a/a270088/port_kokkos/docs/SYNC_MAP.md`  ← the per-substep host/device currency map (substep 1 = EOS = your first device kernel; §9 kernel-author checklist)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D17, L1–L22) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps (dt=1800 AB2 eps=0.1, tracer stride nl, halo bounds) — re-read before touching a kernel
> - `/home/a/a270088/port_kokkos/src/fesom_field.hpp`, `src/fesom_eos.cpp`, `src/fesom_step.cpp`
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/`
>
> GOAL — **M2.1: the first device compute kernel** (EOS / `fesom_pressure_bv` / `compute_sw_alpha_beta`).
> Port them to `parallel_for` over nodes (level loop in the lambda; `Kokkos::` math for the JM-EOS
> `pow`/`sqrt`). Add an in-binary **`FESOM_KK_VERIFY=eos`** mode that runs the C twin AND the Kokkos
> kernel on the same state and asserts `max|Δ|==0` on **Serial** (keep the C twin in-tree, dead-but-
> diffable). **Adopt `-ffp-contract=off` and RE-BASELINE the pi golden** (deferred from M0.3 to "when
> the first kernel lands" — the kernel-gate determinism knob). Wire the SYNC_MAP §1 rails for EOS
> (sync_device T/S in; modify_device the 5 aux outputs; sync_host before their halos — the
> `fesom_step.cpp:78/80` `h_checked()` guards will catch a missing sync under SYNCCHECK). Wrap any EOS
> scratch in `Field`. **CUDA bit-identity is EXPECTED to first break here** (fma/transcendentals) →
> acceptance becomes climate-close on CUDA, still `max|Δ|==0` on Serial.
>
> GATE: Serial pi smoke == the (re-baselined) golden + `ctest` 4/4 + np=2 (CMA-off!) == oracle;
> `FESOM_KK_VERIFY=eos` Serial `max|Δ|==0`; CUDA builds (`--target fesom_port`, L17) + pi smoke runs
> (climate-close, no longer bit-identical). Recipe in §2. Append every decision/lesson to
> `docs/KOKKOS_PORTING_LESSONS.md` in the SAME commit; commit per step.
>
> INVARIANTS: never simplify physics; preserve every constant/loop bound verbatim. The Serial backend
> stays the bit-identity oracle (`max|Δ|==0` vs the C twin) for every kernel. C twin oracle:
> `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623). First fresh checkout: `git submodule update --init --recursive`.
