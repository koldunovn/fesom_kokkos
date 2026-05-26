# FESOM2 C → C++/Kokkos port — session handoff

**Session 9 (2026-05-26) — M2.4 COMPLETE (PGF + momentum RHS + viscosity + implicit vertical
viscosity on device).** Substeps 2/4/5/6 of the ocean step are now device-resident, so **substeps
1–6 are all on the device** (modulo internal/boundary halos + the mid-step CG host round-trip). M2.4
landed the port's **first edge→entity scatters** (`Kokkos::atomic_add`, decision **D22**,
`docs/SCATTER_STRATEGY.md`): Serial stays bit-identical, but **OpenMP/CUDA are now climate-close (not
bit-identical) for the two scatter kernels** — the first such backend split, by the D5 ladder. Repo:
`/home/a/a270088/port_kokkos` (git, branch `master`). Read this first, then
`docs/plans/20260525-kokkos-port.md`, `docs/KOKKOS_PORTING_LESSONS.md`, `docs/SYNC_MAP.md`,
**`docs/SCATTER_STRATEGY.md`**, and the project memory in
`~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M2.4 COMPLETE — PGF + momentum RHS + viscosity + implicit vertical viscosity on device**
  (commits `07094b5` PGF, `8a419e0` impl_vert_visc, `5fde72c` visc_filt_bidiff, `4e5c8ba`
  compute_vel_rhs). Four `_kk` twins, substeps 2/4/5/6:
  - **`pressure_force_linfs_fullcell_kk`** (`src/fesom_eos.cpp`, substep 2) — clean per-element map,
    EOS-style; `FESOM_KK_VERIFY=pgf`.
  - **`impl_vert_visc_kk`** (`src/fesom_momentum.cpp`, substep 6) — **per-element TDMA**, the Thomas
    sweep sequential in level inside the lambda, per-column `[64]` scratch; race-free → Serial AND
    OpenMP bit-identical; `FESOM_KK_VERIFY=ivisc`.
  - **`visc_filt_bidiff_kk`** (substep 5) — biharmonic ∇⁴, **the first SCATTER kernel**: two
    edge→element stages via `Kokkos::atomic_add` around an internal `Uc/Vc` elem3D halo (D21);
    `FESOM_KK_VERIFY=vfilt`.
  - **`compute_vel_rhs_kk` + `momentum_adv_scalar_kk`** (substep 4) — ⚠️ **AB2 `eps=0.1` preserved**;
    the most composite kernel (7 `parallel_for`s + 1 halo bracket): two race-free element maps around a
    5-stage momentum advection that has an **edge→node scatter** (`atomic_add`) + an **internal
    `uvnode_rhs` halo** (D21); `FESOM_KK_VERIFY=vrhs`.
- **D22 — the SCATTER decision** (`docs/SCATTER_STRATEGY.md`): edge→entity scatters use
  `Kokkos::atomic_add` in natural edge order. **Serial bit-identical** (single-thread atomic = ordered
  `+=`); **OpenMP/CUDA climate-close (≲1e-12)** for the scatter kernels (`visc_filt_bidiff`,
  `momentum_adv_scalar`) — the **first time OpenMP is NOT bit-identical** to the golden (by design, the
  D5 ladder). A gather reformulation can't rescue OpenMP without breaking Serial — the trade-off is
  fundamental. Edge-coloring (if ever, M5) is GPU-only.
- **M2.4 gate — ALL GREEN**: Serial pi (KPP) == golden (np=1 **and** np=2 CMA-off vs `…m13_nocma` —
  exercises the new D21 brackets under MPI); **all 7 `FESOM_KK_VERIFY` keys simultaneously =
  `max|Δ|==0`** (160 lines, 0 non-zero: eos/pgf/vrhs/vfilt/ivisc/pp/kpp); **OpenMP climate-close** (max
  |Δ| = **8.3e-25** in `v` — the scatter regime change, ≪ the ≲1e-12 budget, ≈ FP floor); `ctest` 4/4;
  **SYNCCHECK clean + bit-identical** (all new IN/OUT rails + the 2 internal brackets transition
  `Device→Synced`); **CUDA (A100) builds + runs + climate-close** at the **unchanged M2.1 budget**
  (`runs/pi_check_cuda`, job `25133108`): density Δ≈3.18e-12 STABLE, Av/Kv≈0.095 ISOLATED
  threshold-flips, u/v≈3.7e-4/5.9e-5 slow drift, new `pgf_x/pgf_y`≈8e-18 (negligible device-fma ULP).
  **No new divergence class** (D5; the "DIVERGENCE" print is a PASS from M2.1 on).
- **The mid-step CG host round-trip is now LIVE** (SYNC_MAP §5): substeps 1–6 device → substep 6 OUT
  `sync_host(uv_rhs)` → **host CG (substeps 7–10)** → next step's substep-3 IN re-`sync_device(uv)`
  after the host `update_vel`. No extra code — the per-kernel rails compose. Stays until M4.2.
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
  un-migrated, each deferred to its own M2/M4 task (EOS/PP/momentum had none; **KPP's is done — M2.3a**;
  GM/FCT/CG scratch still pending their M2.5b/M2.6/M4.2 tasks).
- **NEXT: M2.5** — ALE thickness / vertical velocity / CFLz / wsplit (substeps 12/14). See §3.

## 1. Git state

```
HEAD  <handoff> docs: handoff → M2.4 done / next M2.5   (this commit)
      4e5c8ba   M2.4d: momentum RHS on device (compute_vel_rhs + momentum_adv_scalar, substep 4)
      5fde72c   M2.4c: biharmonic viscosity on device (visc_filt_bidiff, substep 5)
      8a419e0   M2.4b: implicit vertical viscosity on device (impl_vert_visc, substep 6)
      07094b5   M2.4a: PGF on device (pressure_force_linfs_fullcell, substep 2)
      be8121c   docs: handoff → M2.3 done (KPP on device) / next M2.4 (PGF+momentum+visc)
      61a4816   M2.3b: KPP vertical mixing on device (the large mixing kernel, 1046 LoC)
      7c55255   M2.3a: Field-wrap the KPP scratch arrays (data layer, bit-identical)
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
# M2.4 keys (substeps 2/4/5/6): pgf, vrhs, vfilt, ivisc. Comma-list any subset; all 7 at once is fine:
FESOM_KK_VERIFY=eos,pgf,vrhs,vfilt,ivisc,pp,kpp ./build-serial/fesom_port <pi mesh> /tmp/pi_v 100 20 10  # all → 0

# --- CUDA (build ONLY the model target — nvcc slow; verify "Built target fesom_port" in the log, L17) ---
source /sw/etc/profile.levante; module --force purge
module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0 netcdf-c/4.8.1-gcc-11.2.0
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake --build build-cuda --target fesom_port -j 16          # (build-cuda already configured)
sbatch jobs/job_pi_smoke_gpu                                 # A100 pi smoke → runs/pi_check_cuda
#   then: diff_snap.py docs/reference/c_baseline_snapshots/pi runs/pi_check_cuda
#   ⚠️ from M2.1 CUDA is CLIMATE-CLOSE, not bit-identical: expect small bounded diffs (density~3e-12,
#   Av/Kv~0.1 isolated threshold-flips, u/v~1e-4, pgf~8e-18), NOT "ALL FIELDS BIT-IDENTICAL". M2.2/2.3/
#   2.4 did NOT change this budget (no new divergence class). See memory reference-cuda-eos-divergence.md.
#   ⚠️ OpenMP is ALSO no longer bit-identical from M2.4 (the visc_filt_bidiff/momentum_adv scatters,
#   D22): expect a tiny climate-close diff (~1e-25..1e-12), NOT "ALL FIELDS BIT-IDENTICAL". Serial is
#   the only strictly bit-identical backend now (still the gate).
```
⚠️ After any struct-LAYOUT change, `touch src/*` before building to kill stale `.o` (L18). The 4
build dirs (`build-serial`, `build-omp`, `build-cuda`, `build-synccheck`) are already configured.

**1-yr CORE2 acceptance** (the milestone gate; rerun if needed): `sbatch jobs/job_m1accept_{cref,serial,omp}`
then `scripts/m1_accept_compare.sh` — see `docs/M1_ACCEPTANCE.md` (incl. the §ranks same-rank-count
rule and the §CUDA / M3.1 deferral).

## 3. THE NEXT TASK — M2.5: ALE thickness / vertical velocity / CFLz / wsplit

Per plan §M2.5 + SYNC_MAP §2 rows 12/14. All in **`src/fesom_ale.cpp`** (substeps 12 + 14 of
`fesom_timestep`; call sites ~`fesom_step.cpp:430-445` for substep 12 and ~`:573` for the commit —
grep, the line numbers shifted with the M2.4 edits). All on the **default golden path** → Serial pi
must STAY `== golden`. **Templates:** `src/fesom_eos.cpp` + `src/fesom_pp.cpp` (entity-outer
`parallel_for`, lambda-local scratch, `_kk` twin + untouched C twin, `*_verify` gates); `src/fesom_kpp.cpp`
+ **`src/fesom_momentum.cpp`** (the M2.4 twins — incl. the per-element TDMA `impl_vert_visc_kk` and the
scatter `visc_filt_bidiff_kk`/`momentum_adv_scalar_kk` shapes) for the internal-halo (D21) and
scatter (D22) patterns; `src/fesom_step.cpp` for the driver rails. Re-read **D19–D22, L26 (capture-before
for read-modify-write), L28 (sync ALL inputs), L33 (derive the kernel's actual shape from the BODY — the
handoff under-described `compute_vel_rhs` as 'element-parallel' when it embeds a scatter + an internal
halo; do NOT trust a one-line description, read the C twin)**.

Kernels (all in `fesom_ale.cpp`; verify key `ale`):
- **`fesom_ale_thickness_linfs`** (substep 12) — linfs: `hnode_new = hnode` essentially; writes mesh
  `hnode_new`. Likely a trivial per-node/elem map.
- **`fesom_ale_vert_vel_linfs`** (substep 12; takes a `gm_on` flag) — vertical velocity from the
  continuity integral. ⚠️ likely a **per-node sequential-in-level scan** (vertical integral) — check the
  body; + the `fer_w` GM accumulator (GM off by default → `fer_w` path is dead on the golden path).
  Writes dyn `w` (and `fer_w` if GM). Halo nod3D `w`.
- **`fesom_ale_compute_cflz`** (substep 12) — vertical CFL at interfaces; per-node map → dyn `cfl_z`.
- **`fesom_ale_compute_wvel_split`** (substep 12) — ⚠️ **preserve `use_wsplit=.false.`**
  (`FESOM_PHASE1_USE_WSPLIT=0`): the splitter always sets `w_e=w`, `w_i=0` (a no-op split). Writes dyn
  `w_e`, `w_i`. (This was a dt=1800 trap — PORTING_LESSONS §0 #3 — keep the `.false.` branch verbatim.)
- **`fesom_ale_commit_thickness`** (substep 14) — `hnode := hnode_new`; `helem` from the vertex mean.
  Writes mesh `hnode`, `helem` (both EVOLVING — feed substep-1 EOS + substep-6 TDMA next step). Halo
  nod3D `hnode` + elem3D `helem`.

Notes: **no new scratch to wrap** (these use mesh/dyn fields already `Field`-backed in M1). The IN rails
sync each kernel's host-authoritative inputs (L28); `hnode`/`helem`/`w*` are EVOLVING mesh/dyn state
(per-step sync, not the one-shot geometry push). Watch for any internal halo (the substep-12 block does
several `fesom_exchange_nod3D` between these — those are *driver* halos between kernels, NOT intra-kernel,
unless `vert_vel` exchanges internally → then a D21 bracket; check the body).

- **Gate** (recipe §2): `FESOM_KK_VERIFY=ale` Serial `max|Δ|==0` (add the key in `fesom_step.cpp`
  mirroring the M2.4 keys — note no substring collisions); Serial pi == golden (np=1 + np=2 CMA-off);
  **OpenMP** (expect bit-identical if no scatter — these look like maps/scans, but VERIFY, and report
  climate-close if a scatter appears); `ctest` 4/4; SYNCCHECK clean; CUDA builds + climate-close. Commit
  per step; append lessons (D/L) + update SYNC_MAP rows 12/14 in the SAME commit.

After M2.5: **M2.5b** (GM/Redi — wrap GM scratch in `Field`, `compute_sigma_xy`/`neutral_slope`/
`init_redi_gm`/`fer_solve_gamma`/`fer_gamma2vel` + the bolus add/sub + horizontal Redi), **M2.6** (FCT
tracer advection — the big one, edge→node scatter per D22/SCATTER_STRATEGY.md + internal exchanges),
**M2.7** (tracer diffusion + M2 acceptance, tag `m2-ocean-device`). (Optional if asked: **M3.1**
multi-GPU mapping to unblock the CUDA CORE2 acceptance row.)

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` (you are entering §M2.5; §M2.1–§M2.4 ticked) · Kokkos
  lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–L33) · **Scatter strategy: `docs/SCATTER_STRATEGY.md`**
  (D22) · Fortran→C traps: `docs/PORTING_LESSONS.md` (dt=1800 AB2 eps=0.1, **wsplit=.false.**, tracer
  stride nl, halo bounds) · **Sync map: `docs/SYNC_MAP.md`** · Acceptance: `docs/M1_ACCEPTANCE.md` ·
  Build: `docs/BUILD.md` · Provenance/MPI: `docs/reference/PROVENANCE.md`
- **Device-kernel worked examples (the M2 template, D19–D21):** `src/fesom_eos.cpp`
  (`fesom_pressure_bv_kk` + per-column lambda-local scratch, `fesom_eos_verify`); **`src/fesom_pp.cpp`**
  (`fesom_compute_vel_nodes_kk` gather, `fesom_pp_mixing_kk` 3-launch loop-ordering D20, `mo_convect_kk`,
  + the in-place-modify verify L26); **`src/fesom_kpp.cpp`** (the big one — 6-stage pipeline, the
  `kpp_wscale_kk` templated `KOKKOS_INLINE_FUNCTION` over lookup Views, `blmix` per-column scratch, and
  the **2 internal-exchange brackets**); and **`src/fesom_momentum.cpp`** (the M2.4 twins — the canonical
  shapes for the next tasks: `impl_vert_visc_kk` = per-element TDMA with `[64]` per-column scratch;
  `visc_filt_bidiff_kk` = the first **scatter** (`atomic_add`, D22) + a D21 internal halo;
  `momentum_adv_scalar_kk` = a 5-stage gather/scatter pipeline + internal halo; `compute_vel_rhs_kk` =
  the composite). Driver rails: the substep-1/2/3/4/5/6 blocks in `src/fesom_step.cpp` (driver IN
  `modify_host+sync_device` ALL inputs L28, kernel `mod_dev`, driver `sync_host`, halos via `h_checked`;
  D21 split = driver IN/OUT + kernel-owned internal brackets; L26 capture-before for read-modify-write).
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
> `git log --oneline -10` to orient. **M0 + ALL of M1 (tag `m1-datalayer`) + M2.1 (EOS) + M2.2 (PP) +
> M2.3 (KPP) + M2.4 (PGF + momentum RHS + viscosity + implicit vertical viscosity) are COMPLETE.** M2.4
> put substeps 2/4/5/6 on the device (commits `07094b5` PGF, `8a419e0` impl_vert_visc, `5fde72c`
> visc_filt_bidiff, `4e5c8ba` compute_vel_rhs) — so **substeps 1–6 of the ocean step are now
> device-resident** (modulo internal/boundary halos + the live mid-step CG host round-trip, SYNC_MAP §5).
> M2.4 introduced the port's **first edge→entity scatters** (`Kokkos::atomic_add`, **decision D22**,
> `docs/SCATTER_STRATEGY.md`): **Serial stays bit-identical** (single-thread atomic = ordered `+=`, the
> gate), but **OpenMP/CUDA are now climate-close (NOT bit-identical) for the two scatter kernels**
> (`visc_filt_bidiff`, `momentum_adv_scalar`) — the first backend split, by the D5 ladder. Validation
> model: per-kernel **`FESOM_KK_VERIFY=<k>` gate is Serial `max|Δ|==0`** (all 20 pi steps; M2.4 keys
> pgf/vrhs/vfilt/ivisc); CUDA climate-close (unchanged M2.1 budget — density Δ≈3e-12 stable, Av/Kv≈0.095
> flips, u/v≈1e-4, pgf≈8e-18; no new divergence class, D5). SYNCCHECK clean.
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status §0, build/run+VERIFY+SYNCCHECK §2, the **M2.5 task §3**)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M2.5; §M2.1–§M2.4 ticked)
> - `/home/a/a270088/port_kokkos/docs/SYNC_MAP.md`  ← per-substep host/device map (substeps 1–6 DONE are the worked rails; rows 12/14 are your M2.5 kernels; **§6 = intra-kernel exchanges**; **§5 = the now-LIVE mid-step CG host round-trip**; §9 kernel-author checklist)
> - `/home/a/a270088/port_kokkos/docs/SCATTER_STRATEGY.md`  ← **D22**: edge→entity scatters = `atomic_add`, Serial bit-identical / OpenMP+CUDA climate-close (matters again at M2.6 FCT)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D22, L1–L33; **D19/D20 = template; D21 = internal-exchange rail split; D22 = scatter/atomic_add; L26 = capture-before for read-modify-write; L28 = sync ALL inputs; L33 = derive the kernel shape from the BODY, not a one-line description**) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps — ⚠️ **`use_wsplit=.false.`** (§0 #3) before `compute_wvel_split`; AB2 eps=0.1; tracer stride nl; halo bounds
> - **TEMPLATE: `/home/a/a270088/port_kokkos/src/{fesom_eos,fesom_pp,fesom_kpp,fesom_momentum}.cpp`** (`_kk` twins + `*_verify` gates) + the substep-1..6 rail blocks in `src/fesom_step.cpp`. `fesom_momentum.cpp` has the freshest shapes (per-element TDMA, scatter+atomic, internal-halo D21, the L26 capture-before verify)
> - M2.5 targets — ALL in `src/fesom_ale.cpp` (substeps 12 + 14): `fesom_ale_thickness_linfs`, `fesom_ale_vert_vel_linfs` (+`fer_w`, GM off), `fesom_ale_compute_cflz`, `fesom_ale_compute_wvel_split` (⚠️ `use_wsplit=.false.` → `w_e=w`,`w_i=0`), `fesom_ale_commit_thickness`. Call sites `src/fesom_step.cpp` substeps 12/14 (grep — lines shifted). **No new scratch to wrap.** ⚠️ **derive each kernel's actual shape from the C body** (L33): check `vert_vel_linfs` for a sequential-in-level scan and any internal halo
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/` (incl. `reference-cuda-eos-divergence.md` = what climate-close looks like)
>
> GOAL — **M2.5: ALE thickness / vertical velocity / CFLz / wsplit** (substeps 12/14, all on the default
> golden path → Serial pi must stay `== golden`). Port the 5 `fesom_ale.cpp` kernels as `_kk` twins
> beside their untouched C twins (D19): `ale_thickness_linfs` (writes `hnode_new`); `ale_vert_vel_linfs`
> (continuity integral → `w`; + the GM `fer_w` accumulator, GM off by default; ⚠️ check the body for a
> level-scan / internal halo); `compute_cflz` (→ `cfl_z`); `compute_wvel_split` (⚠️ **preserve
> `use_wsplit=.false.`** → `w_e=w`, `w_i=0`); `commit_thickness` (`hnode:=hnode_new`, `helem` from
> vertex mean — both EVOLVING, feed next step). Each kernel: a `FESOM_KK_VERIFY=ale` gate (`max|Δ|==0` on
> Serial; add the key in `fesom_step.cpp` mirroring the M2.4 keys, no substring collisions) + its driver
> rail (IN `modify_host()+sync_device()` ALL host-authoritative inputs, L28; OUT `sync_host` before the
> halos via `h_checked`). Wrap any internal `fesom_exchange_*` in a D21 bracket.
>
> GATE: Serial pi smoke == golden + `ctest` 4/4 + np=2 (CMA-off!) == `…m13_nocma` oracle;
> `FESOM_KK_VERIFY=ale` Serial `max|Δ|==0`; **OpenMP** (bit-identical IF no scatter — these look like
> maps/scans; if a scatter appears report climate-close per D22); SYNCCHECK clean; CUDA builds
> (`--target fesom_port`, L17) + pi smoke runs (climate-close). Recipe in §2. Append every
> decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` + update SYNC_MAP rows 12/14 in the SAME commit;
> commit per step.
>
> INVARIANTS: never simplify physics; preserve every constant/loop bound verbatim. The Serial backend
> stays the bit-identity oracle (`max|Δ|==0` vs the C twin) for every kernel. C twin oracle:
> `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623). First fresh checkout: `git submodule update --init --recursive`.
