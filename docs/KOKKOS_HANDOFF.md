# FESOM2 C → C++/Kokkos port — session handoff

**Session 7 (2026-05-26) — M2.2 COMPLETE (PP mixing on device).** The second group of device
compute kernels (after M2.1 EOS) is live: substep 3 of the ocean step now runs on the device. Repo:
`/home/a/a270088/port_kokkos` (git, branch `master`). Read this first, then
`docs/plans/20260525-kokkos-port.md`, `docs/KOKKOS_PORTING_LESSONS.md`, `docs/SYNC_MAP.md`, and the
project memory in `~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M2.2 COMPLETE — PP mixing on device** (commit `17ea075`). Three `_kk` twins in `src/fesom_pp.cpp`
  beside their untouched C twins (the M2.1/D19 template): **`fesom_compute_vel_nodes_kk`** (node
  gather over the `nod_in_elem2D` CSR — accumulate into lambda-local `tx/ty/tvol` then write only this
  node's `uvnode` → race-free; the private per-node reduction keeps the C order so OpenMP is
  bit-identical too), **`fesom_pp_mixing_kk`** (the 3 loops as **3 separate `parallel_for` launches** —
  the launch barrier preserves the ⚠️ loop-2-before-loop-3 ordering, **D20**), **`fesom_mo_convect_kk`**
  (convective-adjustment maxes). They are the production path from the step driver.
- **M2.2 gate — ALL GREEN**: `FESOM_KK_VERIFY=pp` all 20 pi steps Serial `max|Δ|==0` — default KPP
  path verifies compute_vel_nodes + mo_convect; `FESOM_MIX_SCHEME=PP` adds pp_mixing (uvnode/Kv/Av);
  Serial pi (KPP) == golden bit-identical (np=1 **and** np=2 CMA-off); **OpenMP == golden** (private
  reductions, no cross-thread reduce); `ctest` 4/4; **SYNCCHECK clean + bit-identical on BOTH the KPP
  and PP branches** (the new substep-3 rails make the `h_checked()` guards transition `Device→Synced`
  each step); **CUDA (A100) builds + runs + climate-close**.
- **CUDA (A100) — climate-close, same budget as M2.1** (`runs/pi_check_cuda`): density Δ≈3.18e-12
  (STABLE steps 10→20), bvfreq Δ≈3e-15, `Av/Kv` Δ≈0.095 ISOLATED threshold-flips (a 1e-15 bvfreq
  nudge flips a `mo_convect` convective branch — now computed on device), `u/v`≈1e-4 slow drift.
  **M2.2 added no new divergence class** — the shape/magnitude matches the M2.1 reference
  (`reference-cuda-eos-divergence.md`). `diff_snap.py` prints "DIVERGENCE" for any non-zero diff — from
  M2.1 on, CUDA divergence of THIS shape is a PASS (D5). True CUDA climate acceptance is the 2-yr/5-yr
  run (M3.2), not this pointwise smoke.
- **The two M2.2 sync subtleties (carry forward):** (1) `mo_convect_kk` is the **first device reader
  of `bvfreq` AFTER the host `smooth_nod3D`** (substep 1) — its rail does
  `modify_host()+sync_device(bvfreq)`, not a bare `sync_device()` (the host smooth is invisible to the
  DualView, L14/L27). (2) On the **default KPP path** compute_vel_nodes_kk + mo_convect_kk run on the
  device while **KPP itself is still a HOST kernel between them** → `uvnode` round-trips device→host
  and `Kv/Av` round-trip host→device→host (an expected within-step bounce, like the CG round-trip).
- **M2.1 (EOS) remains COMPLETE** (commits `1f0a5e4`, `e060473`): `fesom_pressure_bv_kk` +
  `fesom_compute_sw_alpha_beta_kk` (`src/fesom_eos.cpp`), `FESOM_KK_VERIFY=eos` Serial `max|Δ|==0`.
  **`-ffp-contract=off`** is the standing M2+ determinism knob (D18; a codegen no-op on Levante
  baseline x86-64, L23 — golden unchanged). The real fma divergence is device-only.
- **M0 + ALL of M1 remain COMPLETE** (tags `m0-baseline`, `m1-datalayer`): 126 persistent arrays
  `fesom::Field`-backed; M1.5 lazy host-authoritative sync rails (D17, `docs/SYNC_MAP.md`); **M1
  acceptance = 1-yr CORE2 Serial+OpenMP ALL FIELDS BIT-IDENTICAL** to `/scratch/a/a270088/m1_accept/cref`.
  CUDA CORE2 still deferred to M3.1 (multi-GPU mapping). Per-kernel scratch (kpp/gm/tradv/ssh) remains
  un-migrated, each deferred to its own M2/M4 task (EOS + PP had none — temps are lambda-local).
- **NEXT: M2.3** — KPP vertical mixing (the big one, 1046 LoC, 6 internal halo exchanges). See §3.

## 1. Git state

```
HEAD  <handoff> docs: handoff → M2.2 done / next M2.3   (this commit)
      17ea075   M2.2: PP mixing on device (compute_vel_nodes + pp_mixing + mo_convect)
      9315570   docs: handoff → M2.1 done (EOS on device) / next M2.2 (PP mixing)
      e060473   M2.1: EOS on device — first Kokkos compute kernels (pressure_bv + sw_alpha_beta)
      1f0a5e4   M2.1: adopt -ffp-contract=off (M2 determinism knob) + golden re-verify
      f45662a   docs: handoff → M1 COMPLETE / next M2.1
tag   m1-datalayer  → end of M1 (annotated; CORE2 acceptance + CUDA disposition)
tag   m0-baseline   → M0 (Serial+OpenMP+CUDA pi bit-identical)
```
No tag for M2.1/M2.2 (the M2 milestone tag `m2-ocean-device` is at M2.7, after the whole ocean step is
on device). Oracles: pi golden `docs/reference/c_baseline_snapshots/pi` (byte-identical at
`-ffp-contract=off`, L23); np=2 scatter `/scratch/a/a270088/pi_np2_ref_m13_nocma` (CMA-off, L18; old
`…_m12` CMA-tainted); **1-yr CORE2 `/scratch/a/a270088/m1_accept/cref`** (M1; not re-run at M2.1/M2.2 —
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

## 3. THE NEXT TASK — M2.3: KPP vertical mixing (the big mixing kernel, 1046 LoC)

Per plan §M2.3 + SYNC_MAP §2 row 3 (KPP) + §6 (intra-kernel exchanges). **KPP is the DEFAULT scheme**
(`s_use_kpp` is the golden path) — so unlike PP it sits on the pi/CORE2 gate: porting it to device
means the default Serial pi smoke must STAY `== golden` bit-identical (the device-KPP result == the
host-KPP result on Serial). **The M2.1/M2.2 kernels are the template** (D19/D20): re-read
`src/fesom_eos.cpp` + `src/fesom_pp.cpp` (`fesom_pp_mixing_kk`'s 3-launch ordering is the closest
analogue) and their `fesom_step.cpp` substep-1/substep-3 rail blocks.

- **Entry point:** `fesom_kpp_mixing` (`src/fesom_kpp.cpp:770`), called from the `s_use_kpp` branch in
  `fesom_step.cpp` substep 3 (now ~L221, right after `compute_vel_nodes_kk`+halo). Internal static
  sub-kernels to port as node-parallel `parallel_for`s: `kpp_ri_iwmix` (L219), `kpp_bldepth` (L317),
  `kpp_blmix` (L449), `kpp_enhance` (L588), and the `kpp_wscale` (L173) velocity-scale helper (called
  per node — make it a `KOKKOS_INLINE_FUNCTION`; note the `wmt/wst` lookup table built in
  `fesom_kpp_init`). Use `Kokkos::` math throughout; preserve the `bvfreq` smoothing + bottom-pad fixes.
- **⚠️ Wrap the KPP scratch in `Field` first** (deferred from M1): the `fesom_kpp` struct's ~15 arrays
  (`diffK/viscA/blmc/ghats/dVsq/dkm1/hbl/bfsfc/caseA/stable/ustar/Bo/kbl/wmt/wst`, `src/fesom_kpp.h`).
  Same D12/D16 pattern (Field owns storage; raw ptr = `.h()` alias set once in `fesom_kpp_alloc`;
  `memset`→`*k = fesom_kpp{}` in `_free`, D13/L13). The wscale table `wmt/wst` is set-once → one-shot
  `modify_host()+sync_device()` like the mesh geometry.
- **⚠️ KPP does ~6–7 internal `fesom_exchange_nod3D`** (`fesom_kpp.cpp:863` blmc×3, `889–892`
  diffK×2/ghats/viscA). Each is a HOST halo op → on device bracket EACH:
  device-compute → `sync_host` → halo → `sync_device` → device-compute (SYNC_MAP §6; explicit
  checkboxes in the plan). These are M5 on-device-pack candidates; host round-trips until then.
- **Inputs** (driver IN rail): `aux->bvfreq` (host smooth, substep 1 → `modify_host()+sync_device()`,
  the L27 hand-off — same as `mo_convect`), `aux->dbsfc`, `dyn->uvnode` (host-halo'd), tracers T/S,
  forcing fluxes/stress. **Outputs**: `aux->Av` (elem), `aux->Kv` (node, = `diffK` T-channel). Then
  `mo_convect_kk` runs after (already on device) — so once KPP is on device, the whole substep-3 is
  device-resident except the internal halo round-trips.
- **Gate**: `FESOM_KK_VERIFY=kpp` Serial `max|Δ|==0` (the `"pp"`⊂`"kpp"` collision is already handled,
  L25 — but ADD the `kpp` key detection in `fesom_step.cpp`; `strstr(e,"kpp")` is safe, kpp is not a
  substring of pp). Rich cross-check refs exist: `scripts/kpp_{bldepth,blmix,ri_bvfreq,byteident}_*.py`
  + the `kpp_dump_*` hooks already in `fesom_kpp.cpp`. **DEFAULT pi gate (KPP) must stay == golden** on
  Serial + OpenMP; CUDA climate-close; `ctest` 4/4; np=2 (CMA-off) == oracle; SYNCCHECK clean.
  Commit per step; append lessons (D/L) in the same commit; update SYNC_MAP §2 row-3 (KPP) + §6.

(Optional, if asked: **M3.1 multi-GPU mapping** to unblock the CUDA CORE2 acceptance row — small
`--kokkos-num-devices`/local-rank→device work + re-run cref/serial at a GPU rank count, §CUDA.)

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` (you are entering §M2.3; §M2.1+§M2.2 ticked) · Kokkos
  lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D20, L1–L27) · Fortran→C traps: `docs/PORTING_LESSONS.md`
  (dt=1800 AB2 eps=0.1, tracer stride nl, halo bounds) · **Sync map: `docs/SYNC_MAP.md`** · Acceptance:
  `docs/M1_ACCEPTANCE.md` · Build: `docs/BUILD.md` · Provenance/MPI: `docs/reference/PROVENANCE.md`
- **Device-kernel worked examples (the M2 template, D19/D20):** `src/fesom_eos.cpp`
  (`fesom_pressure_bv_kk`/`fesom_compute_sw_alpha_beta_kk` + `fesom_eos_verify`) and **`src/fesom_pp.cpp`**
  (`fesom_compute_vel_nodes_kk` node gather, `fesom_pp_mixing_kk` 3-launch loop-ordering, `fesom_mo_convect_kk`,
  + the three `*_verify` gates incl. the in-place-modify mo_convect oracle, L26) — `_kk` twins beside
  untouched C twins, lambda-local scratch. Driver rails: the substep-1 (EOS) **and** substep-3 (PP) rail
  blocks in `src/fesom_step.cpp` (driver IN `modify_host+sync_device`, kernel `mod_dev`, driver
  `sync_host`, halos via `h_checked`).
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
> mixing) are COMPLETE.** M2.2 landed the second device-kernel group (commit `17ea075`) — substep 3
> of the ocean step: `fesom_compute_vel_nodes_kk` (node gather over `nod_in_elem2D`),
> `fesom_pp_mixing_kk` (3 loops = 3 `parallel_for` launches; the launch barrier preserves the
> loop-2-before-loop-3 order, D20), `fesom_mo_convect_kk` (convective adjustment) — in `src/fesom_pp.cpp`.
> Validation model (unchanged from M2.1): Serial/OpenMP stay bit-identical to the golden, the per-kernel
> **`FESOM_KK_VERIFY=pp` gate is Serial `max|Δ|==0`** (all 20 pi steps), CUDA is climate-close (same
> budget as M2.1 — density Δ≈3e-12 stable, Av/Kv≈0.1 isolated threshold-flips, D5; no new divergence
> class). SYNCCHECK runs clean on BOTH the KPP and PP branches.
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status §0, build/run+VERIFY+SYNCCHECK §2, the **M2.3 task §3**)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M2.3; §M2.1+§M2.2 ticked)
> - `/home/a/a270088/port_kokkos/docs/SYNC_MAP.md`  ← per-substep host/device currency map (substeps 1=EOS + 3=PP DONE are the worked rails; KPP is the M2.3 substep-3 row; **§6 = the intra-kernel-exchange bracket pattern KPP needs**; §9 kernel-author checklist)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D20, L1–L27; **D19/D20 = the M2 kernel-port template; L26 in-place-modify gate; L27 device→host→device hand-off**) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps (dt=1800 AB2 eps=0.1, tracer stride nl, halo bounds; KPP-specific cross-check notes) — re-read before touching a kernel
> - **TEMPLATE: `/home/a/a270088/port_kokkos/src/fesom_eos.cpp` + `src/fesom_pp.cpp`** (`_kk` twins + the `*_verify` gates; `fesom_pp_mixing_kk`'s 3-launch ordering is the closest analogue to KPP's multi-kernel pipeline) + the substep-1/substep-3 rail blocks in `src/fesom_step.cpp` — copy this shape for M2.3
> - M2.3 target: `src/fesom_kpp.cpp` (`fesom_kpp_mixing` L770; sub-kernels `kpp_ri_iwmix` L219, `kpp_bldepth` L317, `kpp_blmix` L449, `kpp_enhance` L588, `kpp_wscale` L173; scratch arrays in `src/fesom_kpp.h`; internal `fesom_exchange_nod3D` at L863/889–892); call site = the `s_use_kpp` branch in `src/fesom_step.cpp` substep 3 (~L221)
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/` (incl. `reference-cuda-eos-divergence.md` = what climate-close looks like)
>
> GOAL — **M2.3: KPP vertical mixing on device** (the big one, 1046 LoC; the DEFAULT scheme → on the
> golden path, so the default Serial pi smoke must STAY `== golden`). (1) **Wrap the ~15 `fesom_kpp`
> scratch arrays in `Field`** (deferred from M1; D12/D16 pattern; the `wmt/wst` wscale table is set-once
> → one-shot push). (2) Port `kpp_ri_iwmix`/`kpp_bldepth`/`kpp_blmix`/`kpp_enhance` as node-parallel
> `parallel_for`s + `kpp_wscale` as a `KOKKOS_INLINE_FUNCTION`; `Kokkos::` math; preserve the bvfreq
> smoothing + bottom-pad fixes. (3) ⚠️ **KPP does ~6–7 internal `fesom_exchange_nod3D`** (blmc×3,
> diffK×2, ghats, viscA) — bracket EACH: device-compute → `sync_host` → halo → `sync_device` (SYNC_MAP
> §6), as explicit checkboxes. (4) Input rail: `bvfreq` needs `modify_host()+sync_device()` (the L27
> device→host(smooth)→device hand-off, same as `mo_convect`); outputs `aux->Av`/`aux->Kv` (=`diffK`
> T-channel). Add `kpp` to the `FESOM_KK_VERIFY` detection (`strstr(e,"kpp")` is safe — kpp is not a
> substring of pp; the pp-side guard is already L25).
>
> GATE: Serial pi smoke (KPP=default) == golden + `ctest` 4/4 + np=2 (CMA-off!) == `…m13_nocma` oracle;
> `FESOM_KK_VERIFY=kpp` Serial `max|Δ|==0` (cross-check vs `scripts/kpp_{bldepth,blmix,ri_bvfreq}_*.py`
> + the `kpp_dump_*` hooks); OpenMP == golden; SYNCCHECK clean; CUDA builds (`--target fesom_port`, L17)
> + pi smoke runs (climate-close). Recipe in §2. Append every decision/lesson to
> `docs/KOKKOS_PORTING_LESSONS.md` + update SYNC_MAP §2 row-3 (KPP) + §6 in the SAME commit; commit per step.
>
> INVARIANTS: never simplify physics; preserve every constant/loop bound verbatim. The Serial backend
> stays the bit-identity oracle (`max|Δ|==0` vs the C twin) for every kernel. C twin oracle:
> `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623). First fresh checkout: `git submodule update --init --recursive`.
