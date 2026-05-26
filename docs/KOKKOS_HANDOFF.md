# FESOM2 C → C++/Kokkos port — session handoff

**Session 8 (2026-05-26) — M2.3 COMPLETE (KPP vertical mixing on device).** The largest mixing kernel
(1046 LoC) is on the device; substep 3 of the ocean step is now fully device-resident on the default
(KPP) path except its internal halo round-trips. Repo: `/home/a/a270088/port_kokkos` (git, branch
`master`). Read this first, then `docs/plans/20260525-kokkos-port.md`,
`docs/KOKKOS_PORTING_LESSONS.md`, `docs/SYNC_MAP.md`, and the project memory in
`~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M2.3 COMPLETE — KPP vertical mixing on device** (commits `7c55255` Field-wrap, `61a4816` kernels).
  `fesom_kpp_mixing_kk` (`src/fesom_kpp.cpp`) ports the 1046-LoC KPP as **6 sub-stages, each a
  `parallel_for`** (prestep / `ri_iwmix` ×2 / `bldepth` ×3 / `blmix` / `enhance` / `combine` /
  `viscAE` / `Kv`-copy), flowing device→device with NO host round-trip except its **2 internal halo
  points** (`smooth_blmc`; the pre-elem-average exchanges). `kpp_wscale` → a templated
  `KOKKOS_INLINE_FUNCTION` over the `wmt/wst` lookup Views (set-once, one-shot-pushed in
  `fesom_kpp_init`); per-column scratch (`dthick`/`dcol`) is lambda-local. The C statics stay
  untouched as the oracle. **The 15 KPP scratch arrays are now `Field`-backed** (M2.3a, the deferred-
  from-M1 wrap). KPP is the DEFAULT scheme → it's on the golden path.
- **M2.3 gate — ALL GREEN**: `FESOM_KK_VERIFY=kpp` all 20 pi steps Serial `max|Δ|==0` (Kv, Av) —
  bit-identical on the **first complete run** (L29); Serial pi (KPP) == golden (np=1 **and** np=2
  CMA-off, exercising the internal brackets under MPI); **OpenMP == golden**; `ctest` 4/4; **SYNCCHECK
  clean + bit-identical** (the internal-bracket `h_checked` guards + the Kv/Av halos transition
  `Device→Synced` each step); **CUDA (A100) builds + runs + climate-close**.
- **CUDA (A100) — climate-close, the UNCHANGED M2.1 budget** (`runs/pi_check_cuda`): density Δ≈3.18e-12
  (STABLE), bvfreq Δ≈3e-15, `Av/Kv` Δ≈0.095 ISOLATED threshold-flips (now computed by device KPP — but
  at the **same nodes/magnitude** as M2.1/M2.2; KPP's own device-fma ULPs are below the bvfreq-seeded
  flip resolution), `u/v`≈1e-4 slow drift. **No new divergence class.** `diff_snap.py` prints
  "DIVERGENCE" for any non-zero diff — from M2.1 on, CUDA divergence of THIS shape is a PASS (D5).
- **Two KPP rail subtleties (carry forward):** (1) **D21 rail split** — KPP does its OWN halo
  exchanges, so the driver owns the IN rail (sync inputs→device) + OUT rail (sync `Av/Kv`→host), while
  `fesom_kpp_mixing_kk` owns the 2 INTERNAL exchange brackets. (2) **L28** — the Serial gate CANNOT
  catch a missing input `sync_device` (host==device aliases), so the KPP IN rail syncs **all 11**
  inputs explicitly (don't assume device-currency from substep 1); `forcing` is `const` in the driver
  → localized `const_cast` for its coherence sync.
- **M2.1 (EOS) + M2.2 (PP mixing) remain COMPLETE.** M2.2 (`17ea075`): `compute_vel_nodes_kk` (gather),
  `pp_mixing_kk` (3-launch loop-2-before-3, D20), `mo_convect_kk` — `src/fesom_pp.cpp`,
  `FESOM_KK_VERIFY=pp`. M2.1 (`e060473`): EOS in `src/fesom_eos.cpp`, `FESOM_KK_VERIFY=eos`.
  **`-ffp-contract=off`** is the standing M2+ determinism knob (D18; a codegen no-op on baseline
  x86-64, L23). `mo_convect_kk` is the first device reader of `bvfreq` after the host `smooth_nod3D`
  (the L27 hand-off → `modify_host()+sync_device(bvfreq)`).
- **M0 + ALL of M1 remain COMPLETE** (tags `m0-baseline`, `m1-datalayer`): 126 persistent arrays
  `fesom::Field`-backed; M1.5 lazy host-authoritative sync rails (D17, `docs/SYNC_MAP.md`); **M1
  acceptance = 1-yr CORE2 Serial+OpenMP ALL FIELDS BIT-IDENTICAL** to `/scratch/a/a270088/m1_accept/cref`.
  CUDA CORE2 still deferred to M3.1 (multi-GPU mapping). Per-kernel scratch (gm/tradv/ssh) remains
  un-migrated, each deferred to its own M2/M4 task (EOS/PP had none; **KPP's is now done — M2.3a**).
- **NEXT: M2.4** — PGF + momentum RHS + viscosity + implicit vertical viscosity (substeps 2/4/5/6).
  Leaves the SSH CG (substeps 7–10) on host until M4.2 → the expected mid-step round-trip. See §3.

## 1. Git state

```
HEAD  <handoff> docs: handoff → M2.3 done / next M2.4   (this commit)
      61a4816   M2.3b: KPP vertical mixing on device (the large mixing kernel, 1046 LoC)
      7c55255   M2.3a: Field-wrap the KPP scratch arrays (data layer, bit-identical)
      e42d105   docs: handoff → M2.2 done (PP mixing on device) / next M2.3 (KPP)
      17ea075   M2.2: PP mixing on device (compute_vel_nodes + pp_mixing + mo_convect)
      e060473   M2.1: EOS on device — first Kokkos compute kernels (pressure_bv + sw_alpha_beta)
tag   m1-datalayer  → end of M1 (annotated; CORE2 acceptance + CUDA disposition)
tag   m0-baseline   → M0 (Serial+OpenMP+CUDA pi bit-identical)
```
No tag for M2.1/M2.2/M2.3 (the M2 milestone tag `m2-ocean-device` is at M2.7, after the whole ocean
step is on device). Oracles: pi golden `docs/reference/c_baseline_snapshots/pi` (byte-identical at
`-ffp-contract=off`, L23); np=2 scatter `/scratch/a/a270088/pi_np2_ref_m13_nocma` (CMA-off, L18; old
`…_m12` CMA-tainted); **1-yr CORE2 `/scratch/a/a270088/m1_accept/cref`** (M1; not re-run at M2.1–M2.3 —
the per-kernel `FESOM_KK_VERIFY` gate + pi gate cover them). CUDA smoke `runs/pi_check_cuda` is now
climate-close, not a bit-identity oracle. Work tree clean. First checkout: `git submodule update --init --recursive`.

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

# --- per-kernel gate (M2.1+): FESOM_KK_VERIFY=<kernel> runs the C twin beside the device kernel
#     and asserts max|Δ|==0 on Serial. Substring match (eos,pp,...). Run on build-serial: ---
FESOM_KK_VERIFY=eos ./build-serial/fesom_port <pi mesh> /tmp/pi_v 100 20 10        # expect all steps max|Δ|=0
FESOM_KK_VERIFY=kpp ./build-serial/fesom_port <pi mesh> /tmp/pi_v 100 20 10        # KPP (default scheme): Kv, Av
FESOM_KK_VERIFY=pp  ./build-serial/fesom_port <pi mesh> /tmp/pi_v 100 20 10        # KPP path: compute_vel_nodes+mo_convect
FESOM_MIX_SCHEME=PP FESOM_KK_VERIFY=pp ./build-serial/fesom_port <pi mesh> /tmp/pi_v 100 20 10   # +pp_mixing

# --- CUDA (build ONLY the model target — nvcc slow; verify "Built target fesom_port" in the log, L17) ---
source /sw/etc/profile.levante; module --force purge
module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0 netcdf-c/4.8.1-gcc-11.2.0
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake --build build-cuda --target fesom_port -j 16          # (build-cuda already configured)
sbatch jobs/job_pi_smoke_gpu                                 # A100 pi smoke → runs/pi_check_cuda
#   then: diff_snap.py docs/reference/c_baseline_snapshots/pi runs/pi_check_cuda
#   ⚠️ from M2.1 CUDA is CLIMATE-CLOSE, not bit-identical (EOS + now PP on device): expect small
#   bounded diffs (density~3e-12, Av/Kv~0.1 isolated threshold-flips, u/v~1e-4), NOT "ALL FIELDS
#   BIT-IDENTICAL". M2.2 did not change this budget. See project memory reference-cuda-eos-divergence.md.
```
⚠️ After any struct-LAYOUT change, `touch src/*` before building to kill stale `.o` (L18). The 4
build dirs (`build-serial`, `build-omp`, `build-cuda`, `build-synccheck`) are already configured.

**1-yr CORE2 acceptance** (the milestone gate; rerun if needed): `sbatch jobs/job_m1accept_{cref,serial,omp}`
then `scripts/m1_accept_compare.sh` — see `docs/M1_ACCEPTANCE.md` (incl. the §ranks same-rank-count
rule and the §CUDA / M3.1 deferral).

## 3. THE NEXT TASK — M2.4: PGF + momentum RHS + viscosity + implicit vertical viscosity

Per plan §M2.4 + SYNC_MAP §2 rows 2/4/5/6. Four kernels (substeps 2/4/5/6), all on the **default
golden path** → the Serial pi smoke must STAY `== golden`. **Templates:** `src/fesom_eos.cpp` +
`src/fesom_pp.cpp` (entity-outer `parallel_for`, lambda-local scratch, `_kk` twin + untouched C twin,
`*_verify` gates) and especially **`src/fesom_kpp.cpp`** for the two harder shapes here — a kernel
with an **internal halo exchange** (D21 bracket; `visc_filt_bidiff`) and a **per-element TDMA**
(per-column lambda-local scratch + a sequential level sweep; `impl_vert_visc`). Re-read D19–D21,
**`PORTING_LESSONS.md §1` (the AB2 `eps=0.1` dt=1800 trap)** before `compute_vel_rhs`, and L28.

**No new scratch to wrap** — momentum's `u_c/v_c` scratch is already `Field`-backed in `dyn` (M1.3).

- **`fesom_pressure_force_linfs_fullcell`** (`src/fesom_eos.cpp:481`; substep 2, call `fesom_step.cpp:204`).
  Element-parallel: reads aux `density_m_rho0`, `hpressure` (EOS device outputs — already device-current
  this step), mesh `gradient_sca`; writes aux `pgf_x`, `pgf_y` (elem). Halo elem3D ×2. The simplest one
  — do it first.
- **`fesom_compute_vel_rhs`** (`src/fesom_momentum.cpp:49`; substep 4, call `:293`). ⚠️ **preserve the
  AB2 `eps=0.1`** (the dt=1800 trap — a no-op at dt=500, decisive at 1800). Element-parallel; reads dyn
  `uv`/`uv_rhsAB`/`eta_n`, aux `pgf`, mesh + the `is_first_step` flag; writes dyn `uv_rhs`/`uv_rhsAB`.
- **`fesom_visc_filt_bidiff`** (`src/fesom_momentum.cpp:654`; substep 5, call `:301`). Biharmonic ∇⁴
  (opt_visc=7); takes `partit` → it has an **internal halo exchange** between its Laplacian stages →
  **D21 bracket** (the `visc_filt_bidiff` internal exchange = the analogue of KPP's `smooth_blmc`:
  device-compute stage 1 → `sync_host` → halo → `sync_device` → stage 2). Element-parallel; uses dyn
  `u_c/v_c` scratch (Field-backed); reads/writes dyn `uv_rhs`.
- **`fesom_impl_vert_visc`** (`src/fesom_momentum.cpp:291`; substep 6, call `:307`). **Per-element TDMA**:
  `parallel_for` over elements, the tridiagonal forward-elim + back-sub **sequential in level INSIDE the
  lambda** (per-column scratch lambda-local, the EOS `bulk_*[64]` pattern). Reads dyn `uv_rhs`, aux `Av`
  (the KPP/PP output), forcing `stress_surf`; writes dyn `uv_rhs`.
- **Leave substeps 7–10 on host** (`compute_ssh_rhs_linfs`/`ssh_solve_cg`/`update_vel`/`compute_hbar`)
  until M4.2 — they bracket the SSH CG. After substep 6, `sync_host(uv_rhs)` before `compute_ssh_rhs`;
  `sync_device` the dyn inputs after `update_vel` resumes device work (none until M2.5 ALE). This is the
  **expected mid-step host round-trip** (SYNC_MAP §5), not a regression — state it in the M2 acceptance.
- **Gate** (recipe §2): `FESOM_KK_VERIFY=<k>` Serial `max|Δ|==0` per kernel (add an `aux`/`momentum`
  verify + the key detection in `fesom_step.cpp`, mirroring eos/pp/kpp); Serial pi == golden + OpenMP ==
  golden; `ctest` 4/4; np=2 (CMA-off) == oracle; SYNCCHECK clean; CUDA builds + climate-close. Commit per
  step; append lessons (D/L) + update SYNC_MAP rows 2/4/5/6 in the SAME commit.

(Optional, if asked: **M3.1 multi-GPU mapping** to unblock the CUDA CORE2 acceptance row — small
`--kokkos-num-devices`/local-rank→device work + re-run cref/serial at a GPU rank count, §CUDA.)

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` (you are entering §M2.4; §M2.1–§M2.3 ticked) · Kokkos
  lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D21, L1–L29) · Fortran→C traps: `docs/PORTING_LESSONS.md`
  (dt=1800 AB2 eps=0.1, tracer stride nl, halo bounds) · **Sync map: `docs/SYNC_MAP.md`** · Acceptance:
  `docs/M1_ACCEPTANCE.md` · Build: `docs/BUILD.md` · Provenance/MPI: `docs/reference/PROVENANCE.md`
- **Device-kernel worked examples (the M2 template, D19–D21):** `src/fesom_eos.cpp`
  (`fesom_pressure_bv_kk` + per-column lambda-local scratch, `fesom_eos_verify`); **`src/fesom_pp.cpp`**
  (`fesom_compute_vel_nodes_kk` gather, `fesom_pp_mixing_kk` 3-launch loop-ordering D20, `mo_convect_kk`,
  + the in-place-modify verify L26); **`src/fesom_kpp.cpp`** (the big one — 6-stage pipeline, the
  `kpp_wscale_kk` templated `KOKKOS_INLINE_FUNCTION` over lookup Views, `blmix` per-column scratch, and
  the **2 internal-exchange brackets** = the model for M2.4's `visc_filt_bidiff` and the M2.3/M2.6/M4.3
  intra-kernel-exchange pattern). Driver rails: the substep-1/substep-3 blocks in `src/fesom_step.cpp`
  (driver IN `modify_host+sync_device`, kernel `mod_dev`, driver `sync_host`, halos via `h_checked`;
  KPP's D21 split = driver IN/OUT + kernel-owned internal brackets).
- `Field`: `src/fesom_field.hpp` (incl. `h_checked()` + the `Auth` tag) · its test `tests/test_field.cpp`
  · data-migration worked examples `src/fesom_{mesh,dyn,aux,tracers,forcing,ice*}.{h,cpp}` ·
  `mesh_sync_geometry_device` (the one-shot `modify_host()+sync_device()` example, L14) · SYNCCHECK
  round-trips: end of `fesom_timestep` (`fesom_step.cpp`), `fesom_ice_step`, before `fesom_timestep` (`fesom_main.cpp`)
- C twin (the bit-identity oracle): `/home/a/a270088/port2/fesom2_port/src` (SHA `75de623`), built bin
  `…/build/fesom_port` · Fortran ground truth `/home/a/a270088/port2/fesom2/src`, ref `/scratch/a/a270088/fortran_pp_2yr`
- Meshes: pi `/home/a/a270088/port2/fesom2/test/meshes/pi` · CORE2 `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`
  (`dist_16/32/144/256/288/432/512`) · Kokkos submodule `externals/kokkos` (4.4.01) · SLURM `ab0995`;
  GPU `gpu`/`gpu-devel`.

## 5. NEXT-SESSION PROMPT (paste this verbatim)

> Continue the FESOM2 C→C++/Kokkos port in `/home/a/a270088/port_kokkos` (git; branch `master`).
> `git log --oneline -8` to orient. **M0 + ALL of M1 (tag `m1-datalayer`) + M2.1 (EOS) + M2.2 (PP
> mixing) + M2.3 (KPP) are COMPLETE.** M2.3 put the 1046-LoC KPP on the device (commits `7c55255`
> Field-wrap, `61a4816` kernels): `fesom_kpp_mixing_kk` (`src/fesom_kpp.cpp`) = 6 sub-stages each a
> `parallel_for`, device→device except 2 internal halo brackets, `kpp_wscale_kk` a templated
> `KOKKOS_INLINE_FUNCTION` over the `wmt/wst` lookup Views. Validation model (unchanged from M2.1):
> Serial/OpenMP stay bit-identical to the golden, the per-kernel **`FESOM_KK_VERIFY=<k>` gate is Serial
> `max|Δ|==0`** (all 20 pi steps), CUDA is climate-close (same budget as M2.1 — density Δ≈3e-12 stable,
> Av/Kv≈0.1 isolated threshold-flips, D5; no new divergence class). SYNCCHECK runs clean. Substep 3 is
> now fully device-resident on the default (KPP) path except its internal halo round-trips.
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status §0, build/run+VERIFY+SYNCCHECK §2, the **M2.4 task §3**)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M2.4; §M2.1–§M2.3 ticked)
> - `/home/a/a270088/port_kokkos/docs/SYNC_MAP.md`  ← per-substep host/device currency map (substeps 1=EOS + 3=PP/KPP DONE are the worked rails; rows 2/4/5/6 are your M2.4 kernels; **§6 = the intra-kernel-exchange bracket** `visc_filt_bidiff` needs; **§5 = the mid-step CG host round-trip** you bridge; §9 kernel-author checklist)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D21, L1–L29; **D19/D20 = template; D21 = rail split for internal-exchange kernels; L28 = sync ALL inputs (Serial gate can't catch a missing one)**) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps — ⚠️ **re-read §1 (AB2 `eps=0.1` dt=1800) before `compute_vel_rhs`**; tracer stride nl, halo bounds
> - **TEMPLATE: `/home/a/a270088/port_kokkos/src/{fesom_eos,fesom_pp,fesom_kpp}.cpp`** (`_kk` twins + `*_verify` gates) + the substep-1/3 rail blocks in `src/fesom_step.cpp`. `fesom_kpp.cpp` is the model for M2.4's two hard shapes: a kernel with an **internal halo exchange** (D21 bracket → `visc_filt_bidiff`) and a **per-element TDMA** (per-column lambda-local scratch → `impl_vert_visc`)
> - M2.4 targets: `fesom_pressure_force_linfs_fullcell` (`src/fesom_eos.cpp:481`, substep 2); `fesom_compute_vel_rhs` (`src/fesom_momentum.cpp:49`, substep 4, ⚠️AB2 eps=0.1); `fesom_visc_filt_bidiff` (`:654`, substep 5, internal halo exchange → D21); `fesom_impl_vert_visc` (`:291`, substep 6, per-element TDMA). Call sites `src/fesom_step.cpp` :204/:293/:301/:307. **No new scratch to wrap** (momentum `u_c/v_c` already Field-backed in `dyn`)
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/` (incl. `reference-cuda-eos-divergence.md` = what climate-close looks like)
>
> GOAL — **M2.4: PGF + momentum RHS + viscosity + implicit vertical viscosity** (substeps 2/4/5/6, all
> on the default golden path → Serial pi must stay `== golden`). Port the 4 kernels as `_kk` twins
> beside their untouched C twins (D19): `pressure_force_linfs_fullcell` (element map — do first);
> `compute_vel_rhs` (⚠️ **preserve AB2 `eps=0.1`**, the dt=1800 trap); `visc_filt_bidiff` (biharmonic;
> has an **internal halo exchange between its Laplacian stages → a D21 bracket**, the `smooth_blmc`
> analogue); `impl_vert_visc` (**per-element TDMA** — `parallel_for` over elements, the tridiagonal
> sweep sequential in level INSIDE the lambda, per-column scratch lambda-local). **Leave substeps 7–10
> (SSH RHS/CG/update_vel/hbar) on host** until M4.2 → `sync_host(uv_rhs)` before `compute_ssh_rhs`, the
> expected mid-step round-trip (SYNC_MAP §5). Each kernel gets a `FESOM_KK_VERIFY` gate (`max|Δ|==0` on
> Serial; add the key detection in `fesom_step.cpp` mirroring eos/pp/kpp) and its driver rail (IN
> `modify_host()+sync_device()` the host-authoritative inputs — **sync ALL of them, L28**; OUT
> `sync_host` before the halos via `h_checked`).
>
> GATE: Serial pi smoke == golden + `ctest` 4/4 + np=2 (CMA-off!) == `…m13_nocma` oracle;
> `FESOM_KK_VERIFY=<k>` Serial `max|Δ|==0`; OpenMP == golden; SYNCCHECK clean; CUDA builds
> (`--target fesom_port`, L17) + pi smoke runs (climate-close). Recipe in §2. Append every
> decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` + update SYNC_MAP rows 2/4/5/6 in the SAME commit;
> commit per step.
>
> INVARIANTS: never simplify physics; preserve every constant/loop bound verbatim. The Serial backend
> stays the bit-identity oracle (`max|Δ|==0` vs the C twin) for every kernel. C twin oracle:
> `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623). First fresh checkout: `git submodule update --init --recursive`.
