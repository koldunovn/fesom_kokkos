# FESOM2 C → C++/Kokkos port — session handoff

**Session 11 (2026-05-26) — M2.5b COMPLETE (GM/Redi on device: the substep-1b streamfunction/bolus
chain + the substep-13 Redi tracer diffusion).** Three commits: `8645824` (M2.5b-a: Field-wrap the 11
GM scratch arrays, bit-identical), `ab57fd6` (M2.5b-b: the 5-kernel chain `sigma_xy`→`neutral_slope`→
`init_redi_gm`→`fer_solve_gamma`→`fer_gamma2vel`), `4bccd69` (M2.5b-c: `diff_ver_part_redi_expl` +
`diff_part_hor_redi`). The GM/Redi PHYSICS is now device-resident; ⚠️ **GM is ON by default in the pi
smoke (L34)** so all of it runs every step on the golden path. The 5 chain kernels are race-free
maps/gathers/per-node TDMA → **Serial AND OpenMP bit-identical**; `diff_hor` is the port's **4th scatter**
(edge→node `atomic_add`, D22). **Key scope decision (L36): the bolus add/sub STAY ON HOST** — the
bolus-augmented `uv`/`w`/`w_e` have no device consumer (FCT + tracer-diff only READ them, grep-verified),
so they move to the device with the FCT in M2.6. Repo: `/home/a/a270088/port_kokkos` (git, branch
`master`). Read this first, then `docs/plans/20260525-kokkos-port.md`, `docs/KOKKOS_PORTING_LESSONS.md`,
`docs/SYNC_MAP.md`, **`docs/SCATTER_STRATEGY.md`**, and the project memory in
`~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M2.5b COMPLETE — GM/Redi on device** (commits `8645824` wrap, `ab57fd6` chain, `4bccd69` Redi). The
  GM/Redi physics — the substep-1b streamfunction/bolus chain AND the substep-13 Redi tracer diffusion —
  is device-resident. ⚠️ **GM is ON by default in pi (L34)** → all on the golden path.
  - **M2.5b-a** — Field-wrap the 11 GM scratch arrays (`sigma_xy/neutral_slope/slope_tapered/fer_tapfac/
    fer_gamma/fer_K/Ki/fer_C/fer_scal/tr_xy/tr_z`), the M2.3a data-layer pattern (`*g = fesom_gm{}` not
    memset, D13/L13). Bit-identical.
  - **M2.5b-b** — the substep-1b chain `compute_sigma_xy_kk`→`compute_neutral_slope_kk`→`init_redi_gm_kk`
    (2 launches, D20)→`fer_solve_gamma_kk` (per-node TDMA, L31)→`fer_gamma2vel_kk` (all in `fesom_gm.cpp`).
    ODM95 taper + `scaling_GMzexp` verbatim. All race-free maps/gathers/TDMA → **Serial AND OpenMP
    bit-identical**. C twins' 6 internal halos move to the DRIVER (the ALE pattern, NOT KPP's D21);
    kernels flow DEVICE→DEVICE reading each upstream's OWNED output; only `fer_gamma` is re-pushed
    (`fer_gamma2vel` reads it at HALO vertices, L30). Produces `fer_uv` (→ device `vert_vel` 12b + the
    bolus 13a) + `slope_tapered`/`Ki` (→ the Redi).
  - **M2.5b-c** — `diff_ver_part_redi_expl_kk` (per-node gather + vd_flux → `+= values`, race-free map) +
    `diff_part_hor_redi_kk` (build `tr_z` + edge→node SCATTER `atomic_add` D22, 5 partial-cell branches),
    run per tracer (T,S) as a DEVICE ISLAND between the host FCT calls (the M2.2/§5 host-round-trip-on-
    `values` pattern). Each owns its internal halo (D21: `tr_xy`, `tr_z`); `values` read-modify-write →
    L26 capture-before verify. **The bolus add/sub STAY ON HOST** (L36 — no device consumer until M2.6).
  - **M2.5b gate — ALL GREEN**: `FESOM_KK_VERIFY=gm` **Serial `max|Δ|==0`** (140 lines = 5 chain + 2 redi
    × 20 steps, 0 non-zero) **and OpenMP `max|Δ|==0`** (every kernel race-free or non-diverging-scatter
    on pi); Serial pi == golden (np=1 **and** np=2 CMA-off vs `…m13_nocma` — exercises the new D21
    brackets + the Redi scatter under MPI); `ctest` 4/4; **SYNCCHECK clean + bit-identical**;
    `FESOM_NO_GMREDI=1` runs clean + differs from golden (GM live, off-path byte-identical by
    construction — all changes `if(gm)`-guarded); **CUDA (A100) climate-close at the UNCHANGED
    M2.1/M2.4/M2.5 budget** (M2.5b-b job `25135312`: density 3.18e-12 STABLE, bvfreq 2.6e-15, u/v
    3.8e-4/7.4e-5, pgf 6-8e-18, w 2.9e-8; **no new divergence class**, D5; M2.5b-c smoke job `25136248`).
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
HEAD  <handoff> docs: handoff → M2.5b done / next M2.6   (this commit)
      4bccd69   M2.5b-c: GM/Redi tracer diffusion on device (diff_ver + diff_hor)
      ab57fd6   M2.5b-b: substep-1b GM chain on device (sigma_xy/.../fer_gamma2vel)
      8645824   M2.5b-a: Field-wrap the GM scratch arrays (data layer, bit-identical)
      3086ad0   docs: handoff → M2.5 done (ALE on device) / next M2.5b (GM/Redi)
      d6937f1   M2.5: ALE thickness/vert_vel/CFLz/wsplit on device (substeps 12/14)
      4e5c8ba   M2.4d: momentum RHS on device (compute_vel_rhs + momentum_adv_scalar, substep 4)
      61a4816   M2.3b: KPP vertical mixing on device (the large mixing kernel, 1046 LoC)
      e060473   M2.1: EOS on device — first Kokkos compute kernels (pressure_bv + sw_alpha_beta)
tag   m1-datalayer  → end of M1 (annotated; CORE2 acceptance + CUDA disposition)
tag   m0-baseline   → M0 (Serial+OpenMP+CUDA pi bit-identical)
```
No tag for M2.1–M2.5b (the M2 milestone tag `m2-ocean-device` lands at M2.7, after the whole ocean step
is on device). CUDA smoke `runs/pi_check_cuda` (M2.5b-c job `25136248`) is climate-close, not a bit oracle.
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

## 3. THE NEXT TASK — M2.6: FCT tracer advection (the big one — scatter + internal exchanges)

Per plan §M2.6. Mostly in **`src/fesom_tracer_adv.cpp`** (1301 LoC — `fesom_tracer_advect_one_fct` +
helpers) + `src/fesom_step.cpp` substep-13 rails. This is the largest remaining ocean kernel and runs
**every step on the golden path** (FCT is the default tracer advection). It is **substep 13** — currently
the only HOST compute left in the substep-13 region, sandwiched between the now-device Redi (M2.5b-c) and
tracer-diff (M2.7, still host). ⚠️ **derive each stage's shape from the C BODY (L33)** — re-read
**D19–D22, L25/L26/L28/L31/L33/L36** and **`docs/SCATTER_STRATEGY.md`** (the FCT flux assembly is the next
edge→node scatter, the D22 default `atomic_add`).

Steps (verify key `tradv` — distinct token, no substring collision):
- **M2.6-a — wrap the FCT scratch arrays in `Field`** (`edge_up_dn_grad`, `fct_LO`/low-order solution,
  the antidiffusive-flux + plus/minus-flux work, and any per-edge/per-node accumulators), the
  M2.3a/M2.5b-a data-layer pattern (bit-identical; deferred from M1). Own commit. ⚠️ grep the struct that
  holds them (`fesom_tra_sc`/`ctx->tra_sc`) and `*x = T{}` not memset (D13/L13).
- **M2.6-b — port the MFCT pipeline** (3rd-order horizontal + vertical FCT): per-element/per-node
  gradient + low-order + antidiffusive flux + the Zalesak limiter + the flux-corrected update. ⚠️ **preserve
  the tracer AB2 `eps=0.1`** (PORTING_LESSONS §1, `oce_tracer_mod.F90:53`) and `feedback_mfct_gradient_from_values`.
  The flux assembly is an **edge→node SCATTER** → `Kokkos::atomic_add` in natural edge order (D22 — Serial
  bit-identical, OpenMP/CUDA climate-close; edge-coloring is GPU-only, never on the Serial gate path).
  Multiple **internal `fesom_exchange_*`** (low-order solution, the flux sums, the limiter coeffs) → each a
  **D21 bracket** owned by the kernel (the KPP/Redi idiom). `_kk` twins beside the C twins (D19); per-tracer
  (T,S) like the Redi.
- **M2.6-c — move the bolus add/sub (13a/13c) to the device** alongside the FCT (L36): now that the FCT
  reads `uv`/`w`/`w_e` ON the device, the bolus add can write them on device (no more pure round-trip).
  Wire: bolus-add `_kk` (`uv+=fer_uv` elem map, `w/w_e+=fer_w` node maps) → device FCT reads them →
  bolus-sub `_kk` restores. `fer_uv`/`fer_w` are device outputs from 1b/12b (re-push the halo'd values, L30).
- **Gate** (recipe §2): `FESOM_KK_VERIFY=tradv` Serial `max|Δ|==0`; Serial pi == golden (np=1 + np=2
  CMA-off vs `…m13_nocma`); `ctest` 4/4; SYNCCHECK clean; **OpenMP** bit-identical for the maps, climate-
  close for the flux scatter (D22); CUDA builds + climate-close. Commit per step; append lessons (D/L) +
  update SYNC_MAP row 13 + §6 in the SAME commit.

After M2.6: **M2.7** (tracer diffusion `impl_vert_diff_tracers` per-node TDMA + S-floor + the bolus
already on device + **M2 acceptance** = full ocean step on device, Serial 1-yr CORE2 bit-identical, tag
`m2-ocean-device`). (Optional if asked: **M3.1** multi-GPU mapping to unblock the CUDA CORE2 acceptance row.)

## 4. Key paths

- Plan: `docs/plans/20260525-kokkos-port.md` (you are entering §M2.6; §M2.1–§M2.5b ticked) · Kokkos
  lessons: `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–L36) · **Scatter strategy: `docs/SCATTER_STRATEGY.md`**
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
  the composite); and **`src/fesom_gm.cpp`** (the M2.5b twins — the closest shape for M2.6: the substep-1b
  chain = device→device with DRIVER halos (the ALE pattern) + the L30 `fer_gamma` re-push; `fer_solve_gamma_kk`
  = per-node TDMA (L31); the two Redi kernels = a per-node gather + an **edge→node scatter (D22)** each with
  its own **D21 internal halo** (`tr_xy`/`tr_z`) + the L26 capture-before `fesom_gm_redi_verify`). Driver
  rails: the substep-1/1b/2/3/4/5/6/12/13/14 blocks in `src/fesom_step.cpp` (driver IN
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
> M2.3 (KPP) + M2.4 (PGF/momentum/viscosity/impl-vert-visc) + M2.5 (ALE) + M2.5b (GM/Redi) are COMPLETE.**
> M2.5b (commits `8645824` wrap, `ab57fd6` chain, `4bccd69` Redi) put the **GM/Redi physics** on the
> device: the substep-1b streamfunction/bolus chain (`sigma_xy`→`neutral_slope`→`init_redi_gm`→
> `fer_solve_gamma`(per-node TDMA)→`fer_gamma2vel`, all race-free → Serial AND OpenMP bit-identical; C
> twins' halos in the DRIVER, ALE pattern; only `fer_gamma` re-pushed L30) and the substep-13 Redi
> (`diff_ver_part_redi_expl` per-node gather + `diff_part_hor_redi` edge→node SCATTER `atomic_add` D22,
> each owning a `tr_xy`/`tr_z` internal-halo D21 bracket; a device island between the host FCT calls).
> ⚠️ **GM is ON by default in pi (L34)** → all of it runs every step on the golden path. **Scope decision
> (L36): the bolus add/sub STAY ON HOST** — no device consumer of `uv`/`w`/`w_e` until the FCT moves to
> device (M2.6, verified FCT/tracer-diff only READ them). Validation model: per-kernel
> **`FESOM_KK_VERIFY=<k>` gate is Serial `max|Δ|==0`** (M2.5b key `gm`, 7 kernels: 5 chain + 2 redi); CUDA
> stays at the **unchanged M2.1/M2.4/M2.5 budget** (density Δ≈3.18e-12, u/v≈3.8e-4/7.4e-5, pgf≈8e-18; no
> new divergence class, D5). SYNCCHECK clean.
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (status §0, build/run+VERIFY+SYNCCHECK §2, the **M2.6 task §3**)
> - `/home/a/a270088/port_kokkos/docs/plans/20260525-kokkos-port.md`  ← the plan (you are at §M2.6; §M2.1–§M2.5b ticked)
> - `/home/a/a270088/port_kokkos/docs/SYNC_MAP.md`  ← per-substep host/device map (substeps 1–6, 1b, 12/13(Redi)/14 DONE are the worked rails; **row 13 (FCT) is your M2.6**; **§6 = intra-kernel exchanges**; **§5 = the mid-step CG host round-trip**; §9 kernel-author checklist)
> - `/home/a/a270088/port_kokkos/docs/SCATTER_STRATEGY.md`  ← **D22**: edge→entity scatters = `atomic_add`, Serial bit-identical / OpenMP+CUDA climate-close (the **FCT flux assembly is the next scatter**)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← Kokkos decisions/lessons (D1–D22, L1–L36; **D19/D20 = template; D21 = internal-exchange bracket; D22 = scatter/atomic_add; L25 = verify-token collision; L26 = capture-before; L28 = sync ALL inputs; L31 = per-node TDMA; L33 = derive shape from the BODY; L35 = the GM chain / DRIVER-halo + L30 re-push; L36 = M2.5b-c Redi + the "bolus stays host, no device consumer" scope rule**) — APPEND every session
> - `/home/a/a270088/port_kokkos/docs/PORTING_LESSONS.md`  ← inherited Fortran→C traps — ⚠️ **tracer AB2 `eps=0.1`** (`oce_tracer_mod.F90:53`), `feedback_mfct_gradient_from_values`; tracer stride nl; halo bounds
> - **THE FILE TO PORT: `/home/a/a270088/port_kokkos/src/fesom_tracer_adv.cpp`** (1301 LoC — `fesom_tracer_advect_one_fct` + the MFCT helpers; read the BODIES, L33) · call sites in `src/fesom_step.cpp` (substep-13 `if (!nt_adv_skip)` block; the device Redi already wraps it) · Fortran ground truth `/home/a/a270088/port2/fesom2/src/oce_ale_tracer.F90` (the FCT/MFCT pipeline) — grep the routine names
> - build/run/VERIFY/SYNCCHECK recipe = handoff **§2** (and `/home/a/a270088/port_kokkos/docs/BUILD.md`); meshes/oracles/C-twin-bin = handoff **§4 Key paths**
> - **TEMPLATE: `/home/a/a270088/port_kokkos/src/{fesom_kpp,fesom_momentum,fesom_ale,fesom_gm}.cpp`** (`_kk` twins + `*_verify` gates) + the substep rail blocks in `src/fesom_step.cpp`. **`fesom_gm.cpp`** (the freshest — the M2.5b twins: device→device chain with DRIVER halos, per-node TDMA, the two Redi kernels = gather + edge→node scatter D22 each with a D21 internal halo, the L26 capture-before `fesom_gm_redi_verify`) and `fesom_momentum.cpp` (`momentum_adv_scalar_kk` = the 5-stage gather/scatter pipeline + internal halo — the closest shape to FCT) are the closest
> - M2.6 targets — mostly in `src/fesom_tracer_adv.cpp` (1301 LoC): (a) wrap the FCT scratch in `Field` (`edge_up_dn_grad`/`fct_LO`/flux work — grep the `tra_sc` struct, M2.3a pattern, `*x=T{}` not memset); (b) port the MFCT 3rd-order H+V pipeline — the flux assembly is an **edge→node scatter** (`atomic_add` D22) with **multiple internal exchanges** (each a D21 bracket); ⚠️ **preserve tracer AB2 `eps=0.1`** + `feedback_mfct_gradient_from_values`; (c) **move the bolus add/sub (13a/13c) to device** alongside the FCT (L36 — now there's a device consumer of `uv`/`w`/`w_e`). ⚠️ **derive each stage's shape from the C body** (L33)
> - project memory: `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/` (incl. `reference-cuda-eos-divergence.md` = what climate-close looks like)
>
> GOAL — **M2.6: FCT tracer advection** (substep 13, the default tracer scheme → on the golden path,
> runs every step; Serial pi must stay `== golden`). Three steps: (a) wrap the FCT scratch arrays in
> `Field` (M2.3a/M2.5b-a data-layer pattern, bit-identical, own commit); (b) port the MFCT pipeline
> (per-element/node gradient + low-order + antidiffusive flux + Zalesak limiter + flux-corrected update)
> as `_kk` twins (D19), per tracer (T,S) like the Redi — the **flux assembly = edge→node `atomic_add`
> scatter** (D22, natural edge order; edge-coloring is GPU-only), each internal `fesom_exchange_*` a **D21
> bracket** owned by the kernel; ⚠️ preserve tracer AB2 `eps=0.1`; (c) move the bolus add/sub to device
> (L36). Each kernel: a `FESOM_KK_VERIFY=tradv` gate + its driver IN/OUT rail (L28; `sync_host` before
> halos via `h_checked`).
>
> GATE: `FESOM_KK_VERIFY=tradv` Serial `max|Δ|==0`. Plus Serial pi smoke == golden + `ctest` 4/4 + np=2
> (CMA-off!) == `…m13_nocma` oracle; **OpenMP** (bit-identical for maps; climate-close for the flux
> scatter, per D22); SYNCCHECK clean; CUDA builds (`--target fesom_port`, L17) + pi smoke runs
> (climate-close). Recipe in §2. Append every decision/lesson to `docs/KOKKOS_PORTING_LESSONS.md` + update
> the SYNC_MAP row 13 + §6 in the SAME commit; commit per step.
>
> INVARIANTS: never simplify physics; preserve every constant/loop bound verbatim. The Serial backend
> stays the bit-identity oracle (`max|Δ|==0` vs the C twin) for every kernel. C twin oracle:
> `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623). First fresh checkout: `git submodule update --init --recursive`.
