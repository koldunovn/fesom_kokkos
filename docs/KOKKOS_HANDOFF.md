# FESOM2 C → C++/Kokkos port — session handoff

**Session 10 (2026-05-26) — M2.5 COMPLETE (ALE thickness / vertical velocity / CFLz / wsplit on
device, substeps 12 + 14).** Five `_kk` twins (commit `d6937f1`); the whole ALE block is now
device-resident, so the ocean step is on the device for **substeps 1–6 AND 12/14** (modulo the
driver halos + the mid-step CG host round-trip). `vert_vel` is the port's **third+ scatter** (edge→node
`atomic_add`, D22) + a per-node level cumsum; the other four are race-free maps (bit-identical on
Serial AND OpenMP). **Key finding (L34): GM is ON by default in the pi smoke** (the prior handoff said
off), so `vert_vel`'s `fer_w` accumulator is LIVE — porting the `gm_on` branch verbatim + its
`fer_uv`/`fer_w` rails is what keeps pi == golden. Repo: `/home/a/a270088/port_kokkos` (git, branch
`master`). Read this first, then `docs/plans/20260525-kokkos-port.md`, `docs/KOKKOS_PORTING_LESSONS.md`,
`docs/SYNC_MAP.md`, **`docs/SCATTER_STRATEGY.md`**, and the project memory in
`~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M2.5 COMPLETE — ALE thickness / vertical velocity / CFLz / wsplit on device** (commit `d6937f1`,
  substeps 12 + 14, all in `src/fesom_ale.cpp`). Five `_kk` twins beside the untouched C twins (D19),
  one shared verify key `ale`:
  - **`thickness_linfs_kk`** (12a) — `hnode_new = hnode` flat copy; **`commit_thickness_kk`** (14) —
    `hnode:=hnode_new` + `helem` vertex-mean (2 launches, barrier-ordered). Both race-free maps →
    **bit-identical on Serial AND OpenMP**.
  - **`vert_vel_linfs_kk`** (12b) — the continuity integral: an **edge→node SCATTER** (`atomic_add`,
    D22) + a **per-node sequential level cumsum** + area divide. Serial bit-identical, OpenMP/CUDA
    climate-close (the scatter). Includes the **GM `fer_w` accumulator** (LIVE — see below).
  - **`compute_cflz_kk`** (12c) — per-node accumulation into the node's **OWN column** (NOT a
    cross-thread scatter) → bit-identical on Serial AND OpenMP. **`compute_wvel_split_kk`** (12d) —
    pure `(n,nz)` map, **⚠️ `use_wsplit=.false.` preserved verbatim** (w_e=w, w_i=0).
- **L34 finding — GM is ON by default in the pi smoke** (`s_no_gmredi=0` → `gm=ctx->gm` non-NULL →
  substep-1b `fer_gamma2vel` writes `dyn->fer_uv`). The prior handoff's "GM off by default → `fer_w`
  dead on the golden path" was WRONG. So `vert_vel`'s `fer_w` accumulator is exercised every step; the
  verify proved the `gm_on` branch **bit-identical on Serial** (`fer_w` max|Δ|=0) and **climate-close on
  OpenMP** (~7e-22). Porting the branch verbatim + the `if(gm)` `fer_uv`-IN / `fer_w`-OUT rails is what
  kept pi == golden. **Don't trust "dead on the golden path" — derive liveness from the run** (L24/L33).
- **No ALE kernel has an internal halo** (every `fesom_exchange_*` is a DRIVER halo between kernels) →
  **no D21 bracket**; the data bounces device→host(halo)→device and each kernel's IN rail re-pushes the
  just-halo'd input (`w` before `cflz`, `cfl_z` before `wvel_split`). `hnode_new` stays device-resident
  across 12a→12c→14 (`thickness` `sync_host`s it because the substep-13 HOST tracer adv/diff read it,
  leaving it `Synced`; `cflz`/`commit` read the device copy with a no-op `sync_device`, never `modify_host`).
- **M2.5 gate — ALL GREEN**: `FESOM_KK_VERIFY=ale` **all 5 kernels Serial `max|Δ|==0`** (100 lines, 0
  non-zero); Serial pi == golden (np=1 **and** np=2 CMA-off); `ctest` 4/4; **SYNCCHECK clean +
  bit-identical** (new `w`/`cfl_z`/`w_e`/`w_i`/`hnode`/`helem` halo guards transition `Device→Synced`);
  **OpenMP** = 4 maps bit-identical + `vert_vel` climate-close (per-kernel `w`≈3.4e-21, `fer_w`≈7e-22;
  whole-run `u`≈2e-19 — the vert_vel scatter seed advecting, ≪ the ≲1e-12 budget, D22); **CUDA (A100)
  climate-close at the UNCHANGED M2.1/M2.4 budget** (`runs/pi_check_cuda`, job `25133976`): density
  Δ=3.18e-12 STABLE, Av/Kv≈0.095 ISOLATED flips, **u/v≈3.7e-4/5.9e-5 (identical to M2.4)**, pgf≈8e-18,
  w≈1.7e-8 (dominated by the upstream u/v drift, not the ~1e-21 atomic reorder). **No new divergence
  class** (D5).
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
- **NEXT: M2.5b** — GM/Redi (wrap GM scratch in `Field`; `compute_sigma_xy`/`neutral_slope`/
  `init_redi_gm`/`fer_solve_gamma`/`fer_gamma2vel` on device + the bolus add/sub + horizontal Redi).
  ⚠️ **GM is ON in pi (L34)** so all of M2.5b is already exercised on the golden path — `FESOM_KK_VERIFY=gm`
  Serial `max|Δ|==0` + the `FESOM_NO_GMREDI=1` byte-identical off-switch are the two gates. See §3.

## 1. Git state

```
HEAD  <handoff> docs: handoff → M2.5 done / next M2.5b   (this commit)
      d6937f1   M2.5: ALE thickness/vert_vel/CFLz/wsplit on device (substeps 12/14)
      d399df6   docs: handoff → M2.4 done (PGF+momentum+visc on device) / next M2.5 (ALE)
      4e5c8ba   M2.4d: momentum RHS on device (compute_vel_rhs + momentum_adv_scalar, substep 4)
      5fde72c   M2.4c: biharmonic viscosity on device (visc_filt_bidiff, substep 5)
      8a419e0   M2.4b: implicit vertical viscosity on device (impl_vert_visc, substep 6)
      07094b5   M2.4a: PGF on device (pressure_force_linfs_fullcell, substep 2)
      61a4816   M2.3b: KPP vertical mixing on device (the large mixing kernel, 1046 LoC)
      e060473   M2.1: EOS on device — first Kokkos compute kernels (pressure_bv + sw_alpha_beta)
tag   m1-datalayer  → end of M1 (annotated; CORE2 acceptance + CUDA disposition)
tag   m0-baseline   → M0 (Serial+OpenMP+CUDA pi bit-identical)
```
No tag for M2.1–M2.5 (the M2 milestone tag `m2-ocean-device` lands at M2.7, after the whole ocean step
is on device). CUDA smoke `runs/pi_check_cuda` (job `25133976`) is climate-close, not a bit oracle.
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

## 3. THE NEXT TASK — M2.5b: GM/Redi (sigma_xy, neutral_slope, streamfunction, bolus, horizontal Redi)

Per plan §M2.5b. Mostly in **`src/fesom_gm.cpp`** (1077 LoC) + `src/fesom_step.cpp` rails. ⚠️ **GM is
ON by default in the pi smoke** (L34 — `s_no_gmredi=0` → `gm=ctx->gm` non-NULL), so **all of M2.5b is
already on the golden path** — Serial pi must STAY `== golden`, and the GM routines run every step.
**Templates:** `src/fesom_kpp.cpp` + **`src/fesom_momentum.cpp`** are the closest shapes — GM's
`fer_solve_gamma` is a **vertical solve** (the per-element/per-node TDMA shape, L31/`impl_vert_visc_kk`);
the bolus add/sub are simple per-entity maps; `diff_part_hor_redi` likely has **edge scatters (D22)** +
**internal exchanges (D21)** — ⚠️ **derive each kernel's actual shape from the C BODY (L33), do not trust
this one-liner.** Re-read **D19–D22, L26/L28/L31/L33/L34**.

Steps (verify key `gm` — check no substring collision; "gm" is short, guard like the `pp`⊂`kpp` case L25):
- **M2.5b-a — wrap the GM scratch arrays in `Field`** (`fer_K`/`fer_C`/`Ki`/`fer_gamma`/`sigma_xy`/
  `neutral_slope`/`slope_tapered`/`fer_tapfac`), the M2.3a/D12/D16 pure data-layer pattern (bit-identical;
  these were deferred from M1). Commit on its own.
- **M2.5b-b — port the substep-1b chain** (`fesom_step.cpp:206-211`, inside `if (gm)`): `compute_sigma_xy`,
  `compute_neutral_slope`, `init_redi_gm`, `fer_solve_gamma` (the vertical solve), `fer_gamma2vel`
  (writes `dyn->fer_uv`, which `vert_vel_kk` already consumes — M2.5). **⚠️ preserve ODM95 tapering +
  `scaling_GMzexp` verbatim** (numerics knobs, PORTING_LESSONS §1). `_kk` twins beside the C twins (D19);
  driver IN/OUT rails (L28); wrap any internal `fesom_exchange_*` in a D21 bracket.
- **M2.5b-c — port the bolus add/sub + horizontal Redi**: the bolus add (substep 13a, `fesom_step.cpp`
  ~`:515-532`: `uv+=fer_uv`, `w+=fer_w`, `w_e+=fer_w`) and sub (13c, ~`:618-635`, the restore); and
  `fesom_diff_ver_part_redi_expl` + `fesom_diff_part_hor_redi` (substep 13, ~`:546-555`). ⚠️ **honour
  `feedback_bolus_divergence_balance`** (never clamp `fer_uv` per-cell; scale `fer_gamma` uniformly so the
  FCT continuity holds — plan note). The bolus add/sub read `fer_w` (M2.5's device output, OUT-rail
  `sync_host`'d under `if(gm)`) and `fer_uv` — wire their device rails.
- **Gate** (recipe §2): **two ways** — (1) `FESOM_NO_GMREDI=1` Serial run is **byte-identical to the
  GM-off path** (the C port's existing off-switch invariant, D10); (2) `FESOM_KK_VERIFY=gm` Serial
  `max|Δ|==0` with GM on. Plus Serial pi == golden (np=1 + np=2 CMA-off); `ctest` 4/4; SYNCCHECK clean;
  **OpenMP** bit-identical for maps/solves, climate-close for any scatter (D22); CUDA builds + climate-close.
  Commit per step; append lessons (D/L) + update SYNC_MAP row 1b/13a/13/13c in the SAME commit.

After M2.5b: **M2.6** (FCT tracer advection — the big one, edge→node scatter per D22/`SCATTER_STRATEGY.md`
+ internal exchanges + wrap FCT scratch in `Field`), **M2.7** (tracer diffusion `impl_vert_diff_tracers`
per-node TDMA + S-floor + M2 acceptance, tag `m2-ocean-device`). (Optional if asked: **M3.1** multi-GPU
mapping to unblock the CUDA CORE2 acceptance row.)

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
> M2.3 (KPP) + M2.4 (PGF/momentum/viscosity/impl-vert-visc) + M2.5 (ALE) are COMPLETE.** M2.5 (commit
> `d6937f1`) put the **ALE block (substeps 12 + 14)** on the device — `thickness`/`commit` (flat copy +
> vertex-mean), `vert_vel` (continuity integral: edge→node SCATTER `atomic_add` D22 + per-node level
> cumsum), `cflz`, `wvel_split` (⚠️`use_wsplit=.false.` preserved) — so the ocean step is device-resident
> for **substeps 1–6 AND 12/14** (modulo driver halos + the mid-step CG host round-trip, SYNC_MAP §5).
> ⚠️ **Key finding (L34): GM is ON by default in the pi smoke** (`s_no_gmredi=0`), so `vert_vel`'s `fer_w`
> accumulator is LIVE and **all of M2.5b (GM/Redi) is already on the golden path**. Validation model:
> per-kernel **`FESOM_KK_VERIFY=<k>` gate is Serial `max|Δ|==0`** (M2.5 key `ale`, all 5 kernels); the
> four ALE maps are bit-identical on OpenMP too, only `vert_vel` (the scatter) is OpenMP/CUDA
> climate-close (D22). CUDA stays at the **unchanged M2.1/M2.4 budget** (density Δ≈3e-12, Av/Kv≈0.095
> flips, u/v≈3.7e-4/5.9e-5, pgf≈8e-18; no new divergence class, D5). SYNCCHECK clean.
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status §0, build/run+VERIFY+SYNCCHECK §2, the **M2.5b task §3**)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M2.5b; §M2.1–§M2.5 ticked)
> - `/home/a/a270088/port_kokkos/docs/SYNC_MAP.md`  ← per-substep host/device map (substeps 1–6 + 12/14 DONE are the worked rails; rows **1b/13a/13/13c** are your M2.5b GM kernels; **§6 = intra-kernel exchanges**; **§5 = the mid-step CG host round-trip**; §9 kernel-author checklist)
> - `/home/a/a270088/port_kokkos/docs/SCATTER_STRATEGY.md`  ← **D22**: edge→entity scatters = `atomic_add`, Serial bit-identical / OpenMP+CUDA climate-close (the GM horizontal-Redi flux + M2.6 FCT are the next scatters)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D22, L1–L34; **D19/D20 = template; D21 = internal-exchange rail split; D22 = scatter/atomic_add; L26 = capture-before; L28 = sync ALL inputs; L31 = per-element TDMA; L33 = derive the shape from the BODY; L34 = M2.5 ALE + the "GM is ON in pi" / "described-dead ≠ actually-dead" finding**) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps — ⚠️ GM/Redi numerics knobs (ODM95 tapering, `scaling_GMzexp`, `feedback_bolus_divergence_balance`); AB2 eps=0.1; tracer stride nl; halo bounds
> - **THE FILE TO PORT: `/home/a/a270088/port_kokkos/src/fesom_gm.cpp`** (1077 LoC, the M2.5b C twins — read the BODIES, L33) + its header `src/fesom_gm.h` (lists the deferred branches) · call sites in `/home/a/a270088/port_kokkos/src/fesom_step.cpp` (substep-1b `if(gm)` block ~`:206-211`; bolus add/sub 13a/13c; Redi 13) · Fortran ground truth `/home/a/a270088/port2/fesom2/src/oce_fer_gm.F90` (GM streamfunction/bolus = `sigma_xy`/`neutral_slope`/`init_Redi_GM`/`fer_solve`/`fer_gamma2vel`) + `oce_ale.F90`/`oce_ale_tracer.F90` (the Redi diffusion) — grep the routine names
> - build/run/VERIFY/SYNCCHECK recipe = handoff **§2** (and `/home/a/a270088/port_kokkos/docs/BUILD.md`); meshes/oracles/C-twin-bin = handoff **§4 Key paths**
> - **TEMPLATE: `/home/a/a270088/port_kokkos/src/{fesom_eos,fesom_pp,fesom_kpp,fesom_momentum,fesom_ale}.cpp`** (`_kk` twins + `*_verify` gates) + the substep rail blocks in `src/fesom_step.cpp`. `fesom_momentum.cpp` (per-element TDMA = the `fer_solve_gamma` vertical-solve analogue; scatter+atomic; internal-halo D21; L26 capture-before) and `fesom_ale.cpp` (the freshest: race-free maps + the `vert_vel` scatter/scan, all gated under one key) are the closest shapes
> - M2.5b targets — mostly in `src/fesom_gm.cpp` (1077 LoC): wrap the 8 GM scratch arrays in `Field` (M2.3a pattern), then port `compute_sigma_xy`/`compute_neutral_slope`/`init_redi_gm`/`fer_solve_gamma`(vertical solve)/`fer_gamma2vel` (substep-1b `if(gm)` block, `fesom_step.cpp:206-211`) + the bolus add/sub (13a/13c) + `diff_ver_part_redi_expl`/`diff_part_hor_redi` (13). ⚠️ **GM is ON in pi (L34)** → already on the golden path. ⚠️ **derive each kernel's actual shape from the C body** (L33); honour `feedback_bolus_divergence_balance`
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/` (incl. `reference-cuda-eos-divergence.md` = what climate-close looks like)
>
> GOAL — **M2.5b: GM/Redi** (substep-1b streamfunction/bolus chain + the bolus add/sub + horizontal
> Redi, substeps 13a/13/13c; ⚠️ **GM is ON in pi (L34)** so all on the default golden path → Serial pi
> must stay `== golden`). Three steps: (a) wrap the 8 GM scratch arrays (`fer_K/fer_C/Ki/fer_gamma/
> sigma_xy/neutral_slope/slope_tapered/fer_tapfac`) in `Field` (the M2.3a/D12 data-layer pattern,
> bit-identical, own commit); (b) port `compute_sigma_xy`/`compute_neutral_slope`/`init_redi_gm`/
> `fer_solve_gamma`(vertical solve, the L31 per-node/elem TDMA shape)/`fer_gamma2vel` as `_kk` twins
> (D19), ⚠️ preserving ODM95 tapering + `scaling_GMzexp` verbatim; (c) port the bolus add/sub (`uv`/`w`/
> `w_e` ± `fer_uv`/`fer_w`) + `diff_ver_part_redi_expl`/`diff_part_hor_redi`, ⚠️ honouring
> `feedback_bolus_divergence_balance`. Each kernel: a `FESOM_KK_VERIFY=gm` gate (`max|Δ|==0` on Serial;
> guard the "gm" token if it could substring-collide, L25) + its driver IN/OUT rail (L28; `sync_host`
> before halos via `h_checked`). Wrap any internal `fesom_exchange_*` in a D21 bracket; edge→entity
> Redi-flux scatters use `atomic_add` (D22).
>
> GATE: **two ways** — (1) `FESOM_NO_GMREDI=1` Serial run **byte-identical to the GM-off path** (D10
> off-switch invariant); (2) `FESOM_KK_VERIFY=gm` Serial `max|Δ|==0` with GM on. Plus Serial pi smoke ==
> golden + `ctest` 4/4 + np=2 (CMA-off!) == `…m13_nocma` oracle; **OpenMP** (bit-identical for maps/
> solves; climate-close for any scatter, per D22); SYNCCHECK clean; CUDA builds (`--target fesom_port`,
> L17) + pi smoke runs (climate-close). Recipe in §2. Append every decision/lesson to
> `docs/KOKKOS_PORTING_LESSONS.md` + update the SYNC_MAP rows in the SAME commit; commit per step.
>
> INVARIANTS: never simplify physics; preserve every constant/loop bound verbatim. The Serial backend
> stays the bit-identity oracle (`max|Δ|==0` vs the C twin) for every kernel. C twin oracle:
> `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623). First fresh checkout: `git submodule update --init --recursive`.
