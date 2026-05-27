# FESOM2 C → C++/Kokkos port — session handoff

**Session 17 (2026-05-27) — M4.3c COMPLETE: the sea-ice FCT advection on the device.** Commit
**`ff11119`** — `fesom_ice_tg_rhs_kk` + `fesom_ice_fct_solve_kk` (`src/fesom_ice_fct.cpp`), the M2.6
ocean-FCT analogue but 2-D (single surface layer), reusing the M4.2 SSH-CSR machinery. **Three toolkit
patterns composed:** (1) `tg_rhs` = the EVP `stress2rhs` shape (zero-map + element→node SCATTER into
`values_rhs`, no halo); (2) `ice_solve_low/high_order` = the **M4.2 CG per-row CSR-gather SpMV** over the
`ssh_stiff` sparsity + `fct_massmatrix` (race-free → bit-id Serial AND OpenMP) — `high_order` is the
CG/EVP **host-loop** (iterate device sweeps + per-iter `dvalues` halo); (3) 3× Zalesak `ice_fem_fct` =
antidiff-flux map + cluster-min/max gather + `icepplus/icepminus` SCATTER + correction/limit maps + apply
(overwrite `vals=valuesl` + `vals += fluxes` SCATTER). **3 element→node SCATTERS (D22, `atomic_add`):**
tg_rhs assemble, fem_fct +/- sum, fem_fct vals apply — Serial bit-identical, OpenMP/CUDA climate-close.
`fct_massmatrix` shares the ssh_stiff sparsity → **one-shot push in `fesom_ice_mass_matrix_fill`** (the
M4.2-a CSR pattern; rowptr/colind already device-current from the preconditioner). Driver: IN push
`uice/vice` + the 3 `data[*].values` (L28; FCT-internal scratch fully recomputed → not pushed), OUT
`sync_host(data[*].values)` for the host cut_off+thermo, **L26 capture-before the 3 values** (verify
`icefct`). **M4.3c gate — ALL GREEN:** `FESOM_KK_VERIFY=icefct` Serial `max|Δ|=0` for a_ice/m_ice/m_snow
across all 60 steps × 16 ranks on a CORE2 dist_16 run with ACTIVE ice (`job 25158693`: 960 verify lines,
0 nonzero — alongside evp 960 + icemap 2880, all 0; ~116 CG iters/step → real dynamics); pi==golden (np=1
AND np=2 CMA-off — FCT trivial on pi's zero ice but the 21 D21 halo brackets + 3 scatters are exercised
under MPI); `ctest` 4/4; SYNCCHECK np=1+np=2 clean + bit-identical; **CUDA (A100) builds** (`fesom_ice_fct.cpp.o`
under nvcc — `INFINITY`/`atomic_add`/the lambdas all GPU-valid). Lesson **L44**; SYNC_MAP §3 FCT row +
plan M4.3c box updated; key `icefct`. The M4.3a/b detail follows.

**Session 17b — M4.3d-b: `oce_fluxes` (the ice→ocean flux coupling) on the device — the LAST sea-ice
kernel. M4.3 is COMPLETE + the M4 ACCEPTANCE PASSED → tag `m4-full-device`: the WHOLE MODEL (ocean + sea
ice) is now device-resident on the bit-identity oracle.** Commit **`9cb5d5b`** — `fesom_ice_oce_fluxes_kk`
(`src/fesom_ice_coupling.cpp`): maps over [0,N)
(heat/water flux = -flx_h/-flx_fw; virtual_salt; relax_salt) + **2 `integrate_nod_2D` GLOBAL reductions**
(`Kokkos::parallel_reduce` + `MPI_Allreduce` — the M4.2 `cg_dot` shape; Serial seq → bit-identical,
OpenMP/CUDA climate-close, D22; the FIRST sea-ice reductions) + the owned net-subtract maps (⚠️ virtual_salt
subtract skips cavity, relax_salt does NOT — literal C). The kernel owns its 4 forcing halos: `→host(forcing)`
then host exchange; the **ocean step's tracer-diff IN rail re-pushes** the halo'd host forcing → device (the
device-island handoff). `flx_h/flx_fw` flow **device→device from the M4.3d-a thermo (no push)** — the first
ice device→device handoff. EOS-style verify (full overwrite → no capture-before), key `iceflux`. **Gate ALL
GREEN**: CORE2 dist_16 ACTIVE-ice run (`job 25159374`) — **all 5 ice keys 0 nonzero** (evp 960, icemap 2880,
icefct 960, icethermo 960, iceflux 960); pi==golden np1+np2; ctest 4/4; SYNCCHECK clean; CUDA builds
(`parallel_reduce`/`integrate_nod_2D_kk` under nvcc). Lesson **L46**; SYNC_MAP §3 oce_fluxes row + plan
M4.3d-b box; key `iceflux`.

**M4 ACCEPTANCE — PASSED → tag `m4-full-device`.** The 1-yr CORE2 **Serial** run (`job 25159501`, 17280
steps, 256 ranks / 2 `compute` nodes, ~28 min, real JRA55 1958 forcing) reproduced **all 13 monthly
snapshots ALL FIELDS BIT-IDENTICAL** to `/scratch/a/a270088/m1_accept/serial` vs
`/scratch/a/a270088/m1_accept/cref` (`scripts/m1_accept_compare.sh` / `diff_snap.py`). Since M4.3 put the
ice kernels — incl. `oce_fluxes`, which writes the `heat_flux`/`water_flux`/`virtual_salt`/`relax_salt`
the ocean tracer step consumes — on the device, this is a true end-to-end re-validation that the device
ports COMPOSE bit-identically over a year (the ice→ocean→ice loop), not just per-kernel. **The whole model
(ocean step M2/M4.2 + the sea-ice step M4.3a–d) is device-resident on Serial**, save the trivial host
`eta_n` map + the salinity floor (L39). Tag **`m4-full-device`**. **NEXT (ASK the user): M5 (performance —
per-field layout flips, edge-coloring/GPU-aware MPI to kill the per-step host round-trips, a real GPU
benchmark now that the whole model is on device) OR M3.2 (CUDA/OpenMP CORE2 climate validation — the
active-ice EVP/FCT scatters compound over a year; characterize the budget in `docs/GPU_FIDELITY.md`).**
The M4.3a–d detail follows.

**Session 17a — M4.3d-a: the sea-ice thermodynamics (per-node column physics) on the device.** Commit
**`596f497`** — `fesom_ice_thermodynamics_kk` (`src/fesom_ice_thermo.cpp`). A single race-free map
over [0,N) (the M2.7 TDMA shape → bit-identical Serial AND OpenMP, NO scatter): `therm_ice`/`obudget`/
`budget`/`flooding`/`tfrez` became `KOKKOS_INLINE_FUNCTION` device twins taking a POD `IceThermC` (the
fesom_ice_thermo scalars — its Field members can't cross to device); Loop 1 (`ustar`) owns its nod2D halo.
**JRA55 was the only non-Field thermo input → its 8 physics arrays were Field-wrapped** (the M1.4 pattern,
`src/fesom_jra55.{h,cpp}`; jra is CORE2-only — pi never allocates it). Gate ALL GREEN: `FESOM_KK_VERIFY=icethermo`
Serial `max|Δ|=0` for m/a/ms/t_skin/flx_h/flx_fw/thdgr across 60 steps × 16 ranks on a CORE2 dist_16
ACTIVE-ice run (`job 25159173`, 960 lines, 0 nonzero — the 960 lines prove the thermo block IS entered);
pi==golden np1+np2; ctest 4/4; SYNCCHECK clean; CUDA builds (Newton-Raphson `budget`, `exp`/`pow`, the
POD capture all GPU-valid). Lesson **L45**; SYNC_MAP §3 thermo row + plan M4.3d-a box; key `icethermo`.
**NEXT = M4.3d-b (`oce_fluxes`: flux overwrite + 2 `integrate_nod_2D` reductions) → M4 acceptance/tag
`m4-full-device`.**

---

**Session 16 (2026-05-26) — M4.3a + M4.3b COMPLETE: sea-ice coupling/maps AND the EVP dynamics on the
device.** Two commits: **M4.3a `473c15b`** (`ocean2ice` gather + `cut_off` clamp + h_ice/h_snow diag);
**M4.3b `<this>`** (the EVP 120-subcycle rheology island). **M4.3b — `fesom_ice_evp_dynamics_kk`
(`src/fesom_ice_evp.cpp`):** the CG/M4.2 pattern applied to sea ice — HOST loop control (fixed 120
subcycles, no convergence reduce) + DEVICE per-subcycle kernels (`stress_tensor_kk` per-elem / `stress2rhs_kk`
map+SCATTER+map / save-old / velocity-update) + setup Steps 1-4 (4 kernels) + the per-subcycle uice/vice
halo bracket. 2 element→node SCATTERS (D22): Step 3 elevation-grad rhs + stress2rhs Loop 2 — Serial
bit-identical, OpenMP/CUDA climate-close (compounds over 120 subcycles → confirm at the M4 CORE2 OpenMP/
CUDA acceptance). The **coastal BC stays the verbatim C edge loop on the HOST** (needs `partit->myList_edge2D`,
not Field-backed), folded into the halo bracket's host phase (free at np>1; device-mask BC = M5).
**M4.3b gate — ALL GREEN**: `FESOM_KK_VERIFY=evp` Serial `max|Δ|=0` on a 60-step CORE2 dist_16 run with
ACTIVE ice (job `25149090`: 3840 lines, 0 nonzero; uice max 0.95 m/s, 33416 ice nodes; 120 subcycles/step)
— the meaningful gate (pi = 0 trivially); pi==golden (np=1 AND np=2 CMA-off); `ctest` 4/4; SYNCCHECK
np=1+np=2 clean; OpenMP pi at the M4.2 floor (ice trivial on pi). Verify = L26 capture-before uice/vice/
sigma11/12/22 (the rheology RMW state carried across subcycles + ocean steps). Lesson **L43**; SYNC_MAP §3
EVP row updated; key `evp`. **NEXT = M4.3c (ice FCT — `tg_rhs`+`fct_solve`, the M2.6 ocean-FCT analogue;
`src/fesom_ice_fct.cpp`), then M4.3d (thermo + `oce_fluxes`) → M4 acceptance/tag `m4-full-device`.** The
M4.3a detail + the two M4.3 findings follow.

**Session 16a — M4.3a: the simple sea-ice coupling/maps on the device — the
FIRST sea-ice kernels.** `ocean2ice` (gather) + `cut_off` (clamp) + the h_ice/h_snow diagnostic, in
`src/fesom_ice_coupling.cpp` / `fesom_ice_thermo.cpp` / `fesom_ice.cpp`. **Two defining findings:**
(1) **No data-layer step** — M1.4 already `Field`-wrapped EVERY ice array (work/thermo/data/top-level)
"for M4.3", so M4.3 goes straight to kernels. (2) **⚠️ The ice physics is FORCED-ONLY** — the pi smoke
is a warm basin (no freezing → zero ice → EVP/FCT/cut_off/diag run trivial, thermo gated off), so the
`FESOM_KK_VERIFY=<icekey>` gate is only meaningful on a short **CORE2** run (SLURM) — except `ocean2ice`
(reads the non-trivial ocean surface → pi-verifiable). This is the validation cadence for ALL of M4.3
(slower than M2's pi-based verify). Kernels: `ocean2ice` = the `compute_vel_nodes` L27 private-gather
(srfoce_u/v) + a map (srfoce_temp/salt/ssh); `cut_off`/diag = race-free node maps → **all bit-identical
Serial AND OpenMP** (no scatter/reduce). Device islands within the still-host ice step (round-trips
compose away as EVP/FCT/thermo move in M4.3b-d, L42); `dyn`/`tracers` const→`const_cast` for the IN rail.
**Gate — ALL GREEN**: `FESOM_KK_VERIFY=icemap` Serial `max|Δ|=0` on pi (ocean2ice non-trivial; cut_off/
diag trivial-on-zero-ice) **AND on a 120-step CORE2 dist_16 run with ACTIVE ice** (job `25148594` —
the meaningful cut_off/diag gate); **OpenMP `max|Δ|=0`** (gathers/maps); pi==golden (np=1 AND np=2
CMA-off); `ctest` 4/4; SYNCCHECK np=1+np=2 clean + bit-identical. ⚠️ np>1 verify subtlety: `ocean2ice`
runs the verify AFTER the `srfoce_u/v` driver halo (the kernel leaves halo=0 pre-exchange → a pre-halo
[0,N) diff would false-positive at np>1). Lesson **L42**; SYNC_MAP §3 rows updated; key `icemap`.
(→ led into M4.3b, done above this session.) **⚠️ Run OUTPUT → `/work` or `/scratch`, never `$HOME`.**
M2/M4.2 detail follows.

---

**Session 15 (2026-05-26) — M4.2 COMPLETE: the §5 SSH block (SSH RHS + CG solver + update_vel +
hbar) is on the device — the SYNC_MAP §5 mid-step host CG round-trip is CLOSED.** This was the M3.1
perf bottleneck (the serial host CG ~127 iters/step + the device→host→device PCIe round-trip = ~0.7 of
the GPU step, L40). Commits: **M4.2-a `4de3230`** (Field-wrap the stiffness CSR `rowptr/colind/values/
pr_values` + the CG scratch `rr/zz/pp/App`, pushed once in the preconditioner — data layer, bit-identical),
**M4.2-b `<this>`** (the device kernels). Four `_kk` twins replace the host §5 block (substeps 7-10) in
`src/fesom_step.cpp`:
- **`fesom_compute_ssh_rhs_linfs_kk`** (`src/fesom_ssh.cpp`) — edge→node SCATTER (`atomic_add`, D22) +
  the (1-α)`ssh_rhs_old` linfs map.
- **`fesom_ssh_solve_cg_kk`** (`src/fesom_ssh.cpp`) — **HOST loop control + DEVICE vector kernels**: the
  SpMV `App=A·p` is a per-ROW CSR GATHER (race-free → Serial AND OpenMP bit-identical); the dot products
  are the **FIRST `Kokkos::parallel_reduce`** in the port (Serial seq reduce == the C loop → bit-identical;
  **OpenMP/CUDA climate-close — the FP reduction associativity, the CG's GPU non-determinism source**, D22)
  + the unchanged scalar `MPI_Allreduce`; the AXPYs are maps. **The CG owns its per-iteration `pp`/`rr`/`X`
  halo brackets** (D21, host-staged device→host→MPI→host→device, no-op at np==1; GPU-aware = M5).
- **`fesom_update_vel_kk`** + **`fesom_compute_hbar_kk`** (`src/fesom_momentum.cpp`) — a race-free
  per-element map + an edge→node SCATTER (`atomic_add`, D22) + maps.
**`eta_n` (substep 11) STAYS HOST** (a trivial nod2D map, SYNC_MAP row 11 — no new round-trip; the CG
round-trip is what M4.2 closes). Driver: one shared IN rail (L28) + per-substep OUT `sync_host` before
each halo + two **L30 re-pushes** (`d_eta` after its halo — update_vel reads it at HALO vertices; `uv`
after its halo — compute_hbar reads it). **Gate — ALL GREEN**: `FESOM_KK_VERIFY=ssh` **Serial `max|Δ|==0`**
(20 steps × 7 read-modify-write fields = 140 lines, 0 nonzero — capture-before, L26); pi==golden (np=1
AND np=2 CMA-off — exercises the CG halo brackets + the per-iter `Allreduce` + the 2 scatters under MPI);
all 12 keys together (eos…trdiff,ssh) = 0 on Serial; `ctest` 4/4; **SYNCCHECK np=1+np=2 clean +
bit-identical**; **OpenMP climate-close** (NEW floor `T`≈1.8e-15 / `Av/Kv`≈2e-17 / `u/v`/`eta`≈1e-18 at
step 20 — the 2 SSH scatters + the dot reduce; ≪ the ≲1e-12 budget, NO blow-up; SpMV/maps bit-identical);
**CUDA (A100) climate-close at the UNCHANGED M2 budget** (job `25146872`: density 3.18e-12 stable, Av/Kv
0.095 flips, u/v 1.8e-4/3.1e-5 — identical to M2.5/6/7) **+ a new `eta_n`≈9.4e-11** (the device CG reduce
+ scatters; bounded, no new class). **M4.2 acceptance PASSED** (1-yr CORE2 Serial, job
`25146822`, 17280 steps × ~90 CG iters/step, real JRA55 forcing): **all 13 monthly snapshots ALL FIELDS
BIT-IDENTICAL** to `/scratch/a/a270088/m1_accept/cref` (`scripts/m1_accept_compare.sh`) — the device CG
touches every step, confirming the §5 device block is bit-identical to the host C over a full year.
Lesson **L41**. **The whole ocean step (substeps 1-14) now flows on
the device** except the host `eta_n` map, the salinity floor (L39), and the ice step (M4.3). Tag
`m2-ocean-device` still latest (M4.2 = no new tag until the §4 milestone). **⚠️ Run OUTPUT →
`/work/ab0995/a270088/port2`, never `$HOME`.** **NEXT = M4.3 (sea-ice on device — the last per-step host
compute, §3) OR M3.2 (CUDA CORE2 climate validation, now that the CG is on device so a GPU benchmark is
fair) OR M4.1 (device global reductions) — ASK the user.** Read §0/§3/§5; M3.1/M2 detail follows.

---

**Session 14 (2026-05-26) — M3.1 COMPLETE: multi-GPU run configuration (rank→GPU device mapping).**
The CUDA build now runs across multiple A100s, **one MPI rank ⇒ one GPU**, each rank bound to a
DISTINCT device by its node-local rank (`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` →
`Kokkos::InitializationSettings::set_device_id` in `src/fesom_main.cpp`; the old `initialize(argc,argv)`
made every rank grab GPU 0). Verified: pi `dist_2` (2 ranks/2 GPUs, 1 `gpu-devel` node —
rank0→dev0/rank1→dev1) + CORE2 `dist_8` (8 ranks/8 GPUs, 2 `gpu` nodes — node-local rank resets 0–3 per
node); both complete, physical, climate-close (np=2-CUDA vs the np=2 Serial oracle = density 2.75e-11,
the unchanged CUDA budget; the larger diff vs the np=1 golden is the pre-existing partition effect, L40).
**NO kernel changes** — halos already host-staged via the M1.5 rails, so multi-GPU uses regular MPI
(GPU-aware = M5). **NO CPU regression** (re-ran ALL gates Serial+OpenMP: 460 verify lines `max|Δ|=0`,
pi==golden np1+np2, ctest 4/4, SYNCCHECK; OpenMP at the unchanged M2.5 floor). New: **`docs/RUN_GPU.md`**
(mapping, SLURM binding contract, the same-rank-oracle rule, the perf finding), job scripts
`jobs/job_gpu_multi_pi` + `jobs/job_gpu_core2`, lesson **L40**. **⚠️ PERF (measured, honest): at M3.1
the CPU is ~17× FASTER node-for-node** — CORE2 dt=1800: GPU 8×A100/2 nodes `dist_8` = **0.86 s/step** vs
CPU 256 cores/2 nodes `dist_256` = **0.051 s/step**. The GPU is NOT slow; the **CG solver + sea ice are
STILL ON HOST** and run 8-way (`dist_8`) not 256-way, dominating the step (the §5 round-trip + serial
host CG ~127 iters/step + ice). **→ M4.2 (CG→device) is now the clear next priority** (the perf unlock);
M3.2 (CUDA climate validation) is the parallel option. ⚠️ Run OUTPUT → `/work/ab0995/a270088/port2`,
NEVER `$HOME` (60 GB home quota; a CORE2 GPU run = 3.5 GB). Tag `m2-ocean-device` is still the latest
(M3.1 is run-config — no new tag). Read §0/§3/§5 below; M2 detail follows.

---

**Session 13 (2026-05-26) — M2.7 COMPLETE + M2 MILESTONE ACCEPTED (tag `m2-ocean-device`). The whole
ocean step is now device-resident on the bit-identity oracle.** Commit `6e98fb3` (M2.7:
`fesom_impl_vert_diff_tracers_kk` — the implicit vertical tracer-diffusion TDMA, the LAST host ocean
compute in substep 13b, on the device). Per-node TDMA (the L31 `impl_vert_visc`/`fer_solve_gamma` shape
— the whole Thomas solve sequential in level inside the per-node lambda over 8×`[NL_MAX]` scratch; each
node owns its column → **race-free, NO scatter → Serial AND OpenMP bit-identical**) + the surface
heat/water-flux BC (`bc_surface_kk`, a templated `KOKKOS_INLINE_FUNCTION` over the 4 forcing Views) +
shortwave penetration + the Redi K33 `Ty/Ty1` augmentation (gm-only, empty `dev_view_t` captured when
`gm==NULL`). **The salinity floor STAYS HOST** (L36/L39 — idempotent clamp; the only device consumer is
next-step EOS via its own IN-rail re-sync). Landed **Serial bit-identical on the first complete run**
(`FESOM_KK_VERIFY=trdiff` 40 lines `max|Δ|==0`; all 11 keys together = 460 lines, 0 nonzero).
**M2 ACCEPTANCE PASSED**: the 1-yr CORE2 Serial run (job 25138814, 27 min, 256 ranks) reproduced all
**13** monthly snapshots **ALL FIELDS BIT-IDENTICAL** to `/scratch/a/a270088/m1_accept/cref` (real
JRA55 forcing → exercises the `bc_surface_kk` flux path the zero-flux pi smoke can't) → **tag
`m2-ocean-device`**. OpenMP bit-identical (no `temp`/`salt` class); CUDA (A100) pi smoke climate-close
at the **unchanged M2.5/M2.6 budget** (density 3.18e-12 stable, no new divergence class). Repo:
`/home/a/a270088/port_kokkos` (git, branch `master`). Read this first, then
`docs/plans/20260525-kokkos-port.md`, `docs/KOKKOS_PORTING_LESSONS.md`, `docs/SYNC_MAP.md`,
**`docs/SCATTER_STRATEGY.md`**, and the project memory in
`~/.claude/projects/-home-a-a270088-port-kokkos/memory/`.

## 0. TL;DR status

- **M2.7 COMPLETE + M2 MILESTONE ACCEPTED — tracer diffusion on device; the WHOLE ocean step is now
  device-resident on Serial** (commit `6e98fb3`; tag `m2-ocean-device`).
  - **`fesom_impl_vert_diff_tracers_kk`** (`src/fesom_tracer_diff.cpp`, substep 13b) — the implicit
    vertical tracer-diffusion TDMA (T then S), the LAST host ocean compute. Per-node TDMA (the L31
    `impl_vert_visc`/`fer_solve_gamma` shape — Thomas sweep sequential in level inside the per-node
    lambda over 8×`[NL_MAX]` scratch; each node its OWN column → race-free, NO scatter → **Serial AND
    OpenMP bit-identical**) + the surface heat/water-flux BC (`bc_surface_kk`, a templated
    `KOKKOS_INLINE_FUNCTION` over the 4 forcing Views) + shortwave penetration + the Redi K33 `Ty/Ty1`
    augmentation (gm-only; `slope_tapered`/`Ki` captured as default-empty `dev_view_t` when `gm==NULL`,
    indexed only under the captured `gm_on` int — the first kernel whose gm-dependence lives INSIDE
    the lambda). Driver IN rail syncs every input the body reads (L28; `forcing` `const`→localized
    `const_cast`); `values` read-modify-write → `FESOM_KK_VERIFY=trdiff` = L26 capture-before (T AND S).
  - **The salinity floor STAYS HOST** (L36/L39): idempotent clamp over myDim+eDim; the only device
    consumer of clamped S is next-step substep-1 EOS via its own IN-rail re-sync — moving it would be a
    pure round-trip. The device trdiff OUT-rail `sync_host` + the host halo+clamp leaving `values`
    Synced-but-silently-host-modified is the SAME shape the host trdiff had; only trdiff itself moved.
  - **M2.7 gate — ALL GREEN**: `FESOM_KK_VERIFY=trdiff` **Serial `max|Δ|==0`** (40 lines = 20 steps ×
    T,S); all 11 keys together = **460 lines, 0 nonzero**; Serial pi == golden (np=1 **and** np=2
    CMA-off); `ctest` 4/4; **SYNCCHECK clean + bit-identical**; **OpenMP bit-identical** for the kernel
    (per-node TDMA is race-free → no `temp`/`salt` divergence; whole-run floor stays the M2.5 vert_vel
    `w`≈3.4e-21 / `u`≈2.2e-19 — M2.7 added no scatter); **CUDA (A100) pi smoke climate-close** at the
    UNCHANGED M2.5/M2.6 budget (job `25138812`: density 3.18e-12 STABLE, Av/Kv 0.095 flips, u/v
    1.8e-4/3.1e-5, pgf ~8e-18, new T/S ULPs 1.9e-11/3.6e-14 feeding the same stable density; **no new
    divergence class**, D5).
  - **M2 ACCEPTANCE — PASSED → tag `m2-ocean-device`**: the 1-yr CORE2 **Serial** run (`job 25138814`,
    256 ranks / 2 `compute` nodes, 27 min, real JRA55 forcing) reproduced **all 13 monthly snapshots
    ALL FIELDS BIT-IDENTICAL** to `/scratch/a/a270088/m1_accept/cref` (`scripts/m1_accept_compare.sh`).
    The full ocean step minus the §5 mid-step CG (M4.2) + the M4.1 reductions + the ice step is now
    device-resident. **CUDA CORE2 stays deferred to M3.1/M3.2** (multi-GPU rank→device mapping).
- **M2.6 COMPLETE — FCT tracer advection + GM bolus on device** (commits `d210025` wrap, `706cd5e` FCT,
  `3380108` bolus). Ocean **substep 13 is now a DEVICE ISLAND**: bolus-add → FCT(T) → Redi(T) → FCT(S)
  → Redi(S) → [host tracer-diff 13b + sfloor, M2.7] → bolus-sub.
  - **M2.6-a** — Field-wrap the 12 FCT scratch arrays (`adv_flux_{hor,ver}`/`del_ttf_adv{horiz,vert}`/
    `fct_LO`/`fct_ttf_{min,max}`/`fct_plus`/`fct_minus`/`fct_aux`/`tr_xy`/`edge_up_dn_grad`), `*sc = T{}`
    not memset (D13/L13). Bit-identical.
  - **M2.6-b** — `fesom_tracer_advect_one_fct_kk` (`src/fesom_tracer_adv.cpp`): the MFCT pipeline as ~24
    `parallel_for` launches in ONE function owning its **3** internal-exchange D21 brackets (`fct_LO`
    nod3D; `tr_xy` `ELEM2D_FULL` at stride **nl** — not the GM Redi's nl-1; `fct_plus`+`fct_minus`
    nod3D) + **3** edge→node `atomic_add` SCATTERS (D22: `compute_fct_LO` divergence, Zalesak
    `fct_plus/minus` b1, `flux2dtracer` horizontal). ⚠️ AB2 `eps=0.1`; ⚠️ MFCT element gradient from
    `values` while the flux uses `valuesAB`. The Zalesak `a3+a4` fused to column-local `tvert[NL_MAX]`
    scratch (no `[N*nl]` temp). Verify `tradv` = L26 capture-before on BOTH `values` AND `valuesold`.
    The upwind path (`fesom_tracer_advect_one`) stays untouched C (dead on the golden path). **Serial
    bit-identical on the first complete run** (L37).
  - **M2.6-c** — `fesom_gm_bolus_apply_kk` (`src/fesom_gm.cpp`): the bolus add/sub as one
    sign-parameterised elementwise map (`uv += sgn*fer_uv`, `w`/`w_e += sgn*fer_w`), bit-identical
    Serial+OpenMP. The 13a add `sync_host`s `uv`/`w`/`w_e` so the host mirrors the augmented velocity
    (the `tradv` C twin reads host `uv`; the next-step substep-3 host readers need it) — then they stay
    device-current through the FCT region and 13c restores them with no IN push (L38, L36).
  - **M2.6 gate — ALL GREEN**: `FESOM_KK_VERIFY=tradv` **Serial `max|Δ|==0`** (40 lines = 20 steps × T,S,
    0 non-zero); Serial pi == golden (np=1 **and** np=2 CMA-off vs `…m13_nocma` — exercises the 3 D21
    brackets + 3 scatters under MPI); `ctest` 4/4; **SYNCCHECK clean + bit-identical**; `FESOM_NO_GMREDI=1`
    clean (bolus gm-gated); **OpenMP climate-close** at the UNCHANGED M2.5 budget (whole-run `w`≈3.4e-21,
    `u`≈2.2e-19 — the existing vert_vel floor; the FCT scatters + bolus added no new class on pi);
    **CUDA (A100) climate-close** at the UNCHANGED M2.5 budget (jobs `25137279`/`25137750`: density
    3.18e-12, Av/Kv 0.095 isolated flips, u/v 3.72e-4/5.87e-5 — IDENTICAL to M2.5; no new divergence
    class, D5).
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
- **NEXT: M3.1 or M4.2 (ASK the user — they are independent and either can go first).**
  - **M4.2 — SSH RHS + CG solver on device** (closes the §5 mid-step host round-trip): the last
    per-step host compute on the golden path besides the ice step. `compute_ssh_rhs` + the CG iteration
    (the SpMV `App`, the dot-products → M4.1 reductions, the AXPYs) move to device so substeps 1–14
    flow without the mid-step device→host→device bounce. **This is the prerequisite for a FAIR
    multi-GPU CORE2 benchmark** — today the CG round-trips every step, which would dominate GPU
    runtime. Pairs with M4.1 (device global reductions).
  - **M3.1 — GPU run configuration (multi-GPU MPI mapping)**: map MPI ranks → GPUs (1 rank/GPU, 4/node
    on Levante `gpu`), pick `dist_<#gpus>` CORE2 partitions, `cudaSetDevice` via local-rank. Unblocks
    the CUDA CORE2 acceptance row (M3.2: 2-yr/5-yr GPU climate validation vs `fortran_pp_2yr`). NOTE
    the GPU-sizing reasoning recorded this session: CORE2 (~127k surface nodes, ~4M wet 3D pts) is
    bandwidth-bound → ~1 A100 ≈ the 2-node/256-rank CPU job; CORE2 is too small to scale past ~1–2
    GPUs (the real lever is higher-res meshes). A fair benchmark wants M4.2 done first.
  - The whole-ocean-step device residency is COMPLETE (M2 done); the remaining host compute is the §5
    CG + M4.1 reductions + the ice step (M5?). See §3 for the chosen task's detail.

## 1. Git state

```
HEAD  <handoff> docs: handoff → M2.7 done + M2 accepted / tag m2-ocean-device   (this commit)
tag   m2-ocean-device → M2 (whole ocean step device-resident; 1-yr CORE2 Serial bit-identical) ← on 6e98fb3
      6e98fb3   M2.7: implicit vertical tracer diffusion on device (fesom_impl_vert_diff_tracers_kk)
      217073e   docs: handoff → M2.6 done (FCT advection + GM bolus on device) / next M2.7
      3380108   M2.6-c: GM bolus velocity add/sub on device (fesom_gm_bolus_apply_kk)
      706cd5e   M2.6-b: FCT tracer advection on device (fesom_tracer_advect_one_fct_kk)
      d210025   M2.6-a: Field-wrap the FCT tracer-advection scratch arrays (data layer)
      4bccd69   M2.5b-c: GM/Redi tracer diffusion on device (diff_ver + diff_hor)
      ab57fd6   M2.5b-b: substep-1b GM chain on device (sigma_xy/.../fer_gamma2vel)
      8645824   M2.5b-a: Field-wrap the GM scratch arrays (data layer, bit-identical)
      d6937f1   M2.5: ALE thickness/vert_vel/CFLz/wsplit on device (substeps 12/14)
      4e5c8ba   M2.4d: momentum RHS on device (compute_vel_rhs + momentum_adv_scalar, substep 4)
      61a4816   M2.3b: KPP vertical mixing on device (the large mixing kernel, 1046 LoC)
      e060473   M2.1: EOS on device — first Kokkos compute kernels (pressure_bv + sw_alpha_beta)
tag   m1-datalayer  → end of M1 (annotated; CORE2 acceptance + CUDA disposition)
tag   m0-baseline   → M0 (Serial+OpenMP+CUDA pi bit-identical)
```
`m2-ocean-device` is the M2 milestone tag (annotated, on `6e98fb3`): the whole ocean step is
device-resident and the 1-yr CORE2 Serial run is bit-identical to the C twin. No intermediate tag for
M2.1–M2.6 (this single tag covers the whole ocean step). Oracles: pi golden `docs/reference/c_baseline_snapshots/pi` (byte-identical at
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
# Keys so far: eos pp kpp (M2.1-3); pgf vrhs vfilt ivisc (M2.4); ale (M2.5); gm (M2.5b, 5 chain+2 redi);
# tradv (M2.6, FCT — 40 lines = 20 steps × T,S). Comma-list any subset; all at once is fine:
FESOM_KK_VERIFY=eos,pgf,vrhs,vfilt,ivisc,pp,kpp,ale,gm,tradv ./build-serial/fesom_port <pi mesh> /tmp/pi_v 100 20 10  # all → 0

# --- CUDA (build ONLY the model target — nvcc slow; verify "Built target fesom_port" in the log, L17) ---
source /sw/etc/profile.levante; module --force purge
# ⚠️ --force purge leaves sticky netcdf-c/cdo/ncview/git loaded on the login shell → the build's
# netcdf-c/4.8.1 CONFLICTS and nvhpc never loads (nvcc not found → Error 127). Unload them first:
module unload netcdf-c cdo ncview git 2>/dev/null
module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0 netcdf-c/4.8.1-gcc-11.2.0
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
which nvcc   # sanity: must resolve (…/nvhpc-24.7-…/compilers/bin/nvcc), else the load failed
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

## 3. THE NEXT TASK — M3.1 or M4.2 (ASK the user; independent, either can go first)

M2 is DONE: the whole ocean step (substeps 1–6, 1b, 12, 13a/13/13(Redi)/13b/13c, 14) is device-resident
and tag `m2-ocean-device` is placed. The remaining per-step host compute on the golden path is the §5
mid-step CG round-trip + the M4.1 global reductions + the ice step. Two candidate next tasks:

**Option A — M4.2: SSH RHS + CG solver on device** (plan §M4.2; closes the §5 round-trip).
- `compute_ssh_rhs` (`src/fesom_ssh.cpp`?) + the CG iteration in `src/fesom_cg.cpp`(?): the SpMV `App`
  (the SSH stiffness matrix-vector, an edge/elem→node scatter, D22), the dot-products (→ M4.1 device
  reductions: `Kokkos::parallel_reduce` + an `MPI_Allreduce` on the scalar), the AXPYs (maps). The
  matrix is set-once (build once on host, push to device). Per-step: RHS → CG loop entirely on device,
  only the per-iteration scalar `Allreduce` touches the host.
- **Why this likely first:** it is the prerequisite for a FAIR multi-GPU CORE2 benchmark — today the CG
  round-trips device→host→device every step (SYNC_MAP §5), which on GPU is a PCIe stall that would
  dominate. Until M4.2, a CUDA CORE2 timing is measuring transfers, not kernels.
- Gate: a new `FESOM_KK_VERIFY=cg`(?) Serial `max|Δ|==0` (the CG is deterministic → bit-identical on
  Serial; the reduction order must match the C — derive from the body, L33); pi == golden (np=1 + np=2);
  the CG dot-product reduction is the first `parallel_reduce` → watch associativity on OpenMP/CUDA (D22
  ladder — likely climate-close, not bit-identical, on the threaded/GPU backends).

**Option B — M3.1: GPU run configuration (multi-GPU MPI mapping)** (plan §M3.1; unblocks M3.2).
- Map MPI ranks → GPUs: 1 rank/GPU, 4/node on Levante `gpu`; `cudaSetDevice(local_rank)` (Kokkos
  `--kokkos-device-id` or `Kokkos::initialize` with the device-id from `MPI_COMM_TYPE_SHARED` local
  rank); pick the `dist_<#gpus>` CORE2 partition. Short multi-GPU smoke on `gpu-devel`; doc `docs/RUN_GPU.md`.
- Then **M3.2**: 2-yr/5-yr CUDA CORE2 climate validation vs `/scratch/a/a270088/fortran_pp_2yr` (+ KPP
  ref) — SST/SSS RMS within the Fortran↔C budget; document the GPU↔CPU budget in `docs/GPU_FIDELITY.md`.
- **GPU-sizing note (recorded this session):** CORE2 is ~127k surface nodes × 48 levels (~4M wet 3D
  pts), bandwidth-bound → **~1 A100 ≈ the 2-node/256-rank CPU job**; CORE2 is too small to scale past
  ~1–2 GPUs (occupancy/halo-bound) — the real multi-GPU lever is higher-res meshes. M3.2 measures the
  real number. A fair benchmark wants M4.2 (the §5 CG) done first.

Standing invariants for either: Serial backend is the bit-identity oracle (`max|Δ|==0` vs the C twin);
OpenMP = climate-identical (race-free) or climate-close (scatter/reduce, D22); CUDA = climate-close
(≈ Fortran↔C). Per-kernel `FESOM_KK_VERIFY` gate + pi == golden (np=1 **and** np=2 CMA-off) + `ctest`
4/4 + SYNCCHECK clean every step; append the lesson to `docs/KOKKOS_PORTING_LESSONS.md` + update
`docs/SYNC_MAP.md` in the SAME commit; commit per step.

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

## 5. NEXT-SESSION PROMPT (M3.2 — CUDA/OpenMP CORE2 multi-year climate validation; paste this verbatim)

> Continue the FESOM2 C→C++/Kokkos port in `/home/a/a270088/port_kokkos` (git; branch `master`).
> `git log --oneline -15` to orient. **M0–M4 are ALL COMPLETE — the WHOLE MODEL (ocean + sea ice) is
> device-resident on the bit-identity oracle, tag `m4-full-device`** (M4 acceptance: 1-yr CORE2 Serial job
> `25159501`, all 13 monthly snaps ALL FIELDS BIT-IDENTICAL to `/scratch/a/a270088/m1_accept/cref`).
> **THIS SESSION = M3.2: confirm the CUDA and OpenMP backends are CLIMATE-FAITHFUL over a multi-year CORE2
> run** — the gate M4 (Serial bit-identity) does NOT cover. Serial is bit-identical; OpenMP/CUDA are
> climate-close (D22): the edge/element→node `atomic_add` SCATTERS (ocean visc/momentum/vert_vel/Redi/FCT/
> SSH + ice EVP×2/FCT×3/oce_fluxes) and the `parallel_reduce` REDUCTIONS (CG dots, `integrate_nod_2D`)
> reassociate across threads/lanes and **COMPOUND over a year on ACTIVE ice — UNMEASURED** (the per-kernel
> gate is Serial-only; pi has ZERO ice so the sea-ice scatters never fired in anger). M3.2 measures the
> compounding and writes the budget to **`docs/GPU_FIDELITY.md`** (skeleton already there).
>
> **EVERYTHING IS PREPARED — the workflow (all scaffolding committed this session):**
> 1. **Rebuild `build-omp`** — it is STALE (last built at M4.3c, before M4.3d-a/b). `source ./env.sh &&
>    cmake --build build-omp -j 16`. (`build-cuda` is current — rebuilt through M4.3d-b.) If you touched
>    any struct layout since, `touch src/*` first (L18).
> 2. **GPU step timing — DONE (job `25163175`): 0.731 s/step** post-M4 (only ~15% faster than M3.1's
>    0.86 despite CG+ice now on device — the per-iter host-staged halos dominate; M5 territory, see
>    `docs/GPU_FIDELITY.md` perf note). So **2-yr ≈ 7.0 h, fits the 12 h gpu wall** → launch
>    `jobs/job_m32_cuda_core2` AS-IS (NSTEPS=34560). (Re-time only if you change the halo path. To
>    re-measure: `sbatch jobs/job_gpu_time_core2` → `grep TIMING …/gtime.<jid>.out`.)
> 3. **Launch both 2-yr runs**: `sbatch jobs/job_m32_cuda_core2` (CUDA dist_8, ~7 h, measured) + `sbatch
>    jobs/job_m32_omp_core2` (OpenMP 32 ranks × 8 threads, ~1–2 h). Both write
>    `<var>.fesom.{1958,1959}.monthly.nc` → `/work/…/{kokkos_gpu_runs/m32_cuda, m32_omp}`. ⚠️ Confirm the
>    JRA55 **1958→1959 year rollover** fired (the job greps for it) — 2-yr is the first multi-year run.
> 4. **Compare**: `scripts/m32_climate_compare.py <dir> --label CUDA|OpenMP` — annual-mean surface
>    corr/bias/RMS/|Δ|max per field per year + a year-to-year DRIFT check, vs the **C-port** (`eps_2yr_dt1800`
>    == Serial == bit-identical — this diff IS the scatter/reduce drift, the headline metric) AND vs
>    **Fortran** (`fortran_pp_2yr` — the absolute budget). For the C↔Fortran reference budget run the
>    existing `scripts/eps_climate_compare_2yr.py`.
> 5. **Fill `docs/GPU_FIDELITY.md`** (Results + Verdict). **PASS** = corr≈1, backend-vs-C bias/RMS bounded
>    and ≤ the C-vs-Fortran budget, **DRIFT≈0** (no runaway across years), no NaN/blow-up. If a field
>    drifts, name the suspect scatter/reduction and whether a deterministic reduction / edge-coloring (M5)
>    is warranted. Commit GPU_FIDELITY.md + a lesson (L47) + the handoff/plan update in one commit.
>
> ⚠️ The climate metric (annual-mean RMS/corr) is ROBUST to rank count — so CUDA dist_8 vs the C-port
> (whatever its rank count) is valid for the climate comparison; the L40 same-rank-oracle rule is for
> BIT-identity, not this coarse statistical check. (If backend-vs-C looks concerning, a short same-rank
> CUDA-vs-Serial run isolates the GPU drift from the partition effect — optional deeper check.)
>
> READ FIRST (absolute paths):
> - `/home/a/a270088/port_kokkos/docs/GPU_FIDELITY.md`  ← **THE M3.2 doc** (why, the runs table, the compare, the PASS criteria, the Results/Verdict to fill) + `memory/reference-cuda-eos-divergence.md` (the known CUDA pi budget — zero ice, so it says nothing about the ice scatters)
> - `/home/a/a270088/port_kokkos/jobs/job_m32_cuda_core2` + `job_m32_omp_core2` + `job_gpu_time_core2` (the prepared runs) · `scripts/m32_climate_compare.py` + `scripts/eps_climate_compare_2yr.py` (the C↔Fortran baseline)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_HANDOFF.md`  ← this handoff (Session 17b = the M4 acceptance; §2 build/CUDA recipe; §4 key paths) · `docs/RUN_GPU.md` (multi-GPU mapping, the M3.1 0.86 s/step finding, the same-rank-oracle rule)
> - `/home/a/a270088/port_kokkos/docs/KOKKOS_PORTING_LESSONS.md`  ← D1–D22, L1–L46 (esp. **D22** scatter/reduce → OpenMP/CUDA climate-close; **L40** M3.1 perf + same-rank-oracle; **L41** the CG dot reduce; **L46** the ice oce_fluxes reductions)
> - project memory `/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/` (incl. `feedback-hpc-run-hygiene.md` = report wall-time/perf parity; OUTPUT → `/work` or `/scratch`, NEVER `$HOME`)
>
> **AFTER M3.2 → M5 (performance):** GPU-aware MPI to halo on-device (kill the per-iter CG/EVP/FCT
> host-staged round-trips) · per-field layout flips for hot device-only fields (re-validate Serial
> bit-identical) · fuse the ice/ocean device-islands · a fair multi-GPU CORE2 benchmark vs the CPU now the
> whole model is on device (M3.1's CPU-17×-faster was WITH CG+ice on host). Expand the plan's M5 box.
>
> The standing gates still apply to any change: `FESOM_KK_VERIFY` per-kernel Serial `max|Δ|==0` (ocean keys
> on pi: `eos,pp,kpp,pgf,vrhs,vfilt,ivisc,ale,gm,tradv,trdiff,ssh`; ice keys on CORE2
> `jobs/job_ice_verify_core2`: `evp,icemap,icefct,icethermo,iceflux`); pi == golden (np=1 AND np=2 CMA-off,
> `OMPI_MCA_btl_vader_single_copy_mechanism=none` L18); `ctest` 4/4; SYNCCHECK np=1+np=2 clean; the 1-yr
> CORE2 Serial acceptance still bit-identical to cref after any change. ⚠️ Run OUTPUT →
> `/work/ab0995/a270088/port2` or `/scratch`, never `$HOME`.
>
> BUILD (recipe §2): Serial/OpenMP via `source ./env.sh` then `cmake --build build-serial -j 16` (+
> build-omp, build-synccheck). ⚠️ `source ./env.sh` does NOT persist across shells — source it in the SAME
> command as any `mpirun` (or `mpirun` is not-found). pi smoke + verify on build-serial with the login-node
> MPI override (`export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader`; unset the UCX/HCOLL vars); the CORE2 ice
> verify via `sbatch jobs/job_ice_verify_core2`. CUDA: `source /sw/etc/profile.levante; module --force
> purge; module unload netcdf-c cdo ncview git; module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0
> openmpi/4.1.2-gcc-11.2.0 netcdf-c/4.8.1-gcc-11.2.0; export NVCC_WRAPPER_DEFAULT_COMPILER=g++;
> cmake --build build-cuda --target fesom_port -j 16` (verify "Built target fesom_port", L17).
>
> INVARIANTS: never simplify physics; Serial stays the bit-identity oracle (`max|Δ|==0` vs the C twin);
> OpenMP race-free-bit-identical / scatter-reduce-climate-close (D22); CUDA climate-close (≈ Fortran↔C). C
> twin oracle `/home/a/a270088/port2/fesom2_port/src` (SHA 75de623). First fresh checkout:
> `git submodule update --init --recursive`. ⚠️ struct-layout change → `touch src/*` before building (L18).
