# M6: Port TKE mixing, mEVP sea-ice, and zstar vertical coordinate to the Kokkos port

> From the 2026-07-12 brainstorm (all decisions user-approved). Campaign start state:
> tag `m5.24-tdma-scaling` @ `dca1e2b` (created+pushed 2026-07-12, the paper-reference
> state; working tree was clean).

## Overview

Port three FESOM2 physics options from the validated C port to the C++/Kokkos port:

1. **M6.1 — TKE vertical mixing** (`mix_scheme='cvmix_TKE'`, C: `mix_scheme_nmb==5`)
2. **M6.2 — mEVP sea-ice dynamics** (`whichEVP=1`)
3. **M6.3 — zstar vertical coordinate** (`which_ALE='zstar'`)
4. **M6.4 — combined twin**: zstar+TKE+mEVP CORE2 1-yr climate validation closes the campaign.

**Order rationale**: easiest→hardest (TKE column-local, mEVP ice-contained, zstar invasive),
so the new-oracle workflow is rehearsed twice before the invasive milestone.

**GOLDEN RULE (user-emphasized, non-negotiable)**: strictly faithful to the C port —
every formula, constant, loop bound, branch, halo exchange, and evaluation order is
transcribed exactly. The C oracle is `/home/a/a270088/port2/fesom2_port_zstar/`
(HEAD `df8b9a8`; has all 3 features with its own validated ladder vs Fortran, reference
namelists + PROVENANCE per feature, and completed plan docs to crib from).

**Explicitly OUT of scope** (user decisions): wsplit-ON path (C's impl is known-imperfect;
port zstar exactly as validated = wsplit OFF), NG5/dars perf re-measure, zlevel /
local-zstar fallback machinery, `ldiag_DVD`, cavity/icepack active paths,
`use_ssh_se_subcycl` (CG path only).

**Mechanics** (user decision): direct-to-Kokkos literal transcription — no host-twin-first
stage. Host twins only as throwaway debug aids for a kernel that refuses bit-id.

## Context (from discovery)

### The C oracle tree
- Source: `/home/a/a270088/port2/fesom2_port_zstar/src/` — CMake build, `build/fesom_port`.
- TKE: `fesom_tke.c` (534 L driver, port of `gen_modules_cvmix_tke.F90`) +
  `fesom_cvmix_tke.c` (404 L column core, port of `cvmix_tke.F90`). Knob:
  `FESOM_MIX_SCHEME` (getenv, `fesom_main.c:366` area).
- mEVP: `fesom_ice_maevp.c` (363 L, `EVPdynamics_m` = `ice_maEVP.F90:429-882`,
  NR-optimized, ssh2rhs/stress_tensor_m/stress2rhs_m inlined). Knob: `FESOM_WHICH_EVP`
  (`fesom_ice.c:76`; unset/`0`=std EVP, `1`=mEVP, else abort). Header carries the
  fidelity-traps checklist. Dump rail: `FESOM_EVP_DUMP_DIR`.
- zstar: spread across `fesom_ale.c` (270→518 L) + eos (222 diff-lines) + step (148) +
  ssh (111) + forcing/ice_coupling/tracer_adv/tracer_diff/momentum/mesh. Knob:
  `FESOM_ALE=linfs|zstar`. Completed plan to crib:
  `docs/plans/completed/20260610-zstar-vertical-coordinate.md` (routine map + verified
  reference-value table). Dump rail: `fesom_ale_dump.c` dual dump harness.
- C jobs to pattern-match: `jobs/job_tke_*`, `job_mevp_*`, `job_zstar_*`, `job_all3_1yr`.

### Archived references on /work (verified present 2026-07-12)
| Feature | Fortran ref | C-port output | Dumps / gates |
|---|---|---|---|
| TKE | `/work/ab0995/a270088/port/tke/fortran_linfs_tke/` (2yr), `fortran_zstar_tke` | `c_tke_2yr` (linfs+TKE), `c_zstar_tke_2yr`, `c_zstar_tke_5yr` | `fdump/`, `cdump/`, `cdump_v2/`, `replay/`, `t0_byteident`, `t4_xrank` |
| mEVP | `/work/ab0995/a270088/port/mevp/fortran_mevp_2yr/` | `c_mevp_2yr`, `c_mevp_5yr`, `c_evp_2yr` (std-EVP control) | `cdump_16r/`, `fdump_16r/`, `m1_byteident`, `m4_xrank` |
| zstar | `/work/ab0995/a270088/port/zstar/fortran_zstar_2yr/`, `fortran_linfs_2yr_b` (paired baseline) | `c_zstar_2yr` | `fdump/`, `fdump_k2/`, `z0_byteident`, `z2_cdump`, `z7_xrank` |
| **all-3** | `/work/ab0995/a270088/port/mevp/fortran_all3/` | **`c_all3_1yr`** (triple config already run by C campaign!) | `iovec_gate` |

Reference namelists + PROVENANCE.md: `fesom2_port_zstar/docs/{tke,mevp,zstar}_reference_namelists/`.
Every feature config is a proven **single-knob** clone of the linfs+KPP baseline.

### Kokkos-port integration points (file:line verified 2026-07-12)
- Mixing dispatch: `src/fesom_step.cpp:105-118` (`s_use_kpp` from `FESOM_MIX_SCHEME`,
  default KPP / `P*`→PP); KPP branch `:358`, PP `:398-410`; `mo_convect_kk` call `:428`;
  shared post-mo_convect halos `:436-437` (`Kv` NOD3D + `Av` ELEM3D via `fesom_halo_field`).
- Snapshot writer: `fesom_io_write_snapshot` `src/fesom_io.cpp:253` — **4 edit sites per
  new field** (malloc `:286-318`, gather `:367-387`, `nc_def_var` `:435-457`, `nc_put_var`
  below `:472`) + `sync_host()` before the call (`src/fesom_main.cpp:1337-1346`). Gated by
  CLI arg `snap_every` (argv[5]).
- Bit-id compare: `scripts/diff_snap.py` (zero-tolerance `np.array_equal` per field).
- CUDA gate: `scripts/gpu_fidelity_gate.sh` (needs build-serial + build-cuda; runs
  `jobs/job_core2_serial_ref` oracle + `jobs/job_gpu_fidelity_dev`, then
  `scripts/gpu_fidelity_check.py`). Serial bit-id job pattern:
  `jobs/job_core2_serial_m523fN` — **8 ranks, dist_8, dt=1800, 20 steps, snap_every=10**.
- Climate job: `jobs/job_m32_cuda_core2` (2 nodes / 8×A100, 1 yr via
  `M32_NSTEPS=17280`, monthly `SNAP_EVERY=1440`) + `scripts/m32_climate_compare.py`.
- Monthly output table: `fesom_default_monthly_table` `src/fesom_io.cpp:823-849`
  (17 vars incl. Kv/Av) + resolver fns `:615-810` + `FESOM_DEFAULT_MONTHLY_NVARS` `:821`.
- Mesh struct (`src/fesom_mesh.h:16-158`): **EXISTS** — `bc_index_nod2D` (:119, DualView
  :157), `zbar_3d_n` (:58, DualView :143, static values today), `hnode/hnode_new/helem`
  (:109-111, DualViews :144), `nlevels_nod2D_min` (:49). **ABSENT** — `Z_3d_n`, `dhe`,
  `bottom_node_thickness`, `bottom_elem_thickness`.
- SSH stiffness: `struct fesom_ssh_stiff` `src/fesom_ssh.h:29`, device-backed
  (`values_fld`, `pr_values_fld` :43-46); instance `src/fesom_main.cpp:556-558`.
- Std-EVP device template (`src/fesom_ice_evp.cpp`): subcycle loop `:627`
  (`ice->evp_rheol_steps`=120); elem→node scatter = **`Kokkos::atomic_add` element-order
  (D22: Serial bit-identical)** `:666-667`; per-substep fused halo
  `fesom_halo_field2(uice,vice,NOD2D)` `:716`; sigma11/12/22 persistent (never zeroed on
  entry), live in `fesom_ice_work` (`src/fesom_ice_types.h:74,85`); `eps11/12/22`,
  `inv_areamass`, `inv_mass` **already exist** (:75-78,86-87). mEVP-only arrays
  (`uice_aux/vice_aux/...`) deliberately not mirrored yet (:214-215); the dispatcher
  abort itself lives in `fesom_ice.cpp:466,494-495`.
- wsplit plumbing present (off by default): `src/fesom_ale.cpp:236-271` host,
  `:509-545` device (`FESOM_PHASE1_USE_WSPLIT`, `w_e/w_i` dyn fields).
- Template plans: `docs/plans/completed/20260524-kpp-vertical-mixing.md` (for TKE),
  `docs/plans/20260425-sea-ice-port.md` (for mEVP).

### ⚠️ Known gap discovered: output vector frame
The C tree writes output vector pairs in **geographic** frame by default
(`FESOM_IO_VECTOR_FRAME=geo|rotated`, `fesom_io_stream.c:456-492`, since commit
`75406d3`). The Kokkos port has **no such knob** — vectors (`u,v,uice,vice`) are written
as stored (rotated frame). **Scope confirmed by plan-review: the C rotation is
STREAM-only — the snapshot writer deliberately stays native-frame
(`fesom_io_stream.c:470-474`), so ALL bit-id gates are unaffected.** Only monthly-output
climate compares of vector fields are exposed. Handling (Phase 0 decides per leg):
regenerate needed C comparator legs with `FESOM_IO_VECTOR_FRAME=rotated` (preferred —
cheap CPU runs), or restrict vector-field comparison to script-side rotation. The Fortran
anchors are ALWAYS geo-frame (io_meandata rotates, cannot be re-run cheaply) — vector
fields vs Fortran need script-side r2g rotation or scalar-only restriction. Porting the
geo-frame writer to the Kokkos port is a Post-Completion item, not in scope.

## Development Approach

- **Faithfulness first**: transcribe the C oracle line-by-line; port its solved
  bit-fidelity landmines verbatim; do NOT "fix" or "improve" anything (the mEVP header
  explicitly lists asymmetries that must not be normalized against the std-EVP template).
- **Direct-to-Kokkos**: kernels authored as `parallel_for` directly; DualViews,
  device-resident from day one; halos via the existing device entry points
  (`fesom_halo_field`/`_field2`/`_fieldN`). Elem→node and edge→matrix scatters use the
  established D22 `Kokkos::atomic_add` pattern (Serial = loop-order deterministic =
  bit-id; CUDA covered by the fidelity gate).
- **Feature invisibility**: every knob defaults OFF; with knobs OFF the binary must stay
  bit-identical (Serial) and gate-clean (CUDA) at every commit.
- Complete each task fully before the next; small focused changes; update this plan file
  as work proceeds (`[x]` immediately, `➕` new tasks, `⚠️` blockers).

## Testing Strategy

This project's "tests" are the validation ladder (no unit-test suite; the model IS the
test). Every task ends with its gate rung; a task is not done until its gate passes.

1. **Knob-OFF byte gate** (every task that touches shared code): rebuild `build-serial`,
   run the 8r/20-step/snap10 CORE2 job, `diff_snap.py` vs the same-day pre-change
   baseline → ALL FIELDS BIT-IDENTICAL. (Same-day baseline discipline —
   `feedback-perf-same-day-baseline`.)
2. **Knob-ON Serial bit-id** (per milestone): C oracle binary run at the feature config
   with snapshots, vs Kokkos-Serial same CLI + knob → `diff_snap.py` zero-tolerance.
   On mismatch: bisect with the C-side dump rails (TKE `cdump/replay`, mEVP
   `FESOM_EVP_DUMP_DIR`, zstar `fesom_ale_dump`).
3. **Knob-ON CUDA gate** (per milestone): `gpu_fidelity_gate.sh`-class CORE2 active-ice
   CUDA-vs-Serial at the feature config (expected: M2.1-class divergence magnitudes, the
   documented "climate-close" floor).
4. **1-yr CORE2 climate close** (per milestone): CUDA 1-yr run vs C-port same-config
   (corr ~1.0 class, as M5.23); Fortran archived ref as secondary anchor. Sea-ice fields
   compared under the NaN-vs-0 mask-averaging rule (`feedback-ice-mask-averaging`).
   Vector fields per the frame decision from Phase 0; vs the Fortran anchor, vector
   fields require script-side r2g rotation (Fortran outputs are always geo-frame) or
   are skipped in favor of scalars.
5. **Perf sanity only** (per milestone): s/step vs same-day default-config run noted in
   GPU_FIDELITY §M6.x; `build-cuda-synclog` (`FESOM_SYNC_LOG`) shows no new per-step
   PCIe round-trippers. No perf campaign.
6. **Run hygiene**: SLURM compute nodes only; outputs under
   `/work/ab0995/a270088/port2/<unique-dir-per-job>`; never $HOME.

## Progress Tracking
- mark completed items `[x]` immediately; `➕` for discovered tasks; `⚠️` for blockers;
  keep this file in sync with actual work.

## What Goes Where
- **Implementation Steps**: code, jobs, validation runs, docs — all achievable here.
- **Post-Completion**: NG5 perf re-measure, wsplit fix, geo-frame output port — future.

## Implementation Steps

### Task 0.1: Phase 0 — oracle-switch certification ✅ DONE (2026-07-12)

**Files:**
- Create: `jobs/job_m6_oracle_cert` (C-oracle + Kokkos-Serial paired short runs) ✅
- Modify: `docs/REFERENCE_RUNS.md` (record the oracle switch) ✅

- [x] rebuild the C oracle fresh: `fesom2_port_zstar` @ `df8b9a8`, CMake Release into
      `build-m6oracle/` (non-destructive — the pre-existing `build/` predates `75406d3`
      and is left intact). Flags: `-O3 -DNDEBUG`, mpicc/openmpi-4.1.2, gcc-11.
      Both trees' `env.sh` are byte-identical.
- [x] run C oracle at DEFAULT config (linfs+KPP+EVP, CORE2 dist_8, 8r, dt=1800, 20 steps,
      snap_every=10) on SLURM; run Kokkos-Serial (freshly rebuilt at HEAD) same CLI.
      Both legs `OMPI_MCA_btl_vader_single_copy_mechanism=none` (L18: the vader CMA gather
      is deterministic WITHIN a build but not ACROSS builds — and this is exactly a
      cross-build comparison).
- [x] `diff_snap.py` C-oracle vs Kokkos-Serial → **ALL FIELDS BIT-IDENTICAL** (3 snapshots,
      27 vars; the printed step diagnostics match digit-for-digit too). SLURM 26210028.
      **The zstar-tree C binary is certified as the M6 oracle.**
- [x] M6 knob-OFF baseline saved → `/work/ab0995/a270088/port2/m6_baseline_serial/`
      (+ `PROVENANCE.md` recording binary/commit/config/MPI knob/**mesh level-sums**)
- [x] confirmed `bc_index_nod2D_fld` is HOST-ONLY (`fesom_ice.cpp:229-241`; no
      `sync_device()` and no `.d()` read anywhere in the tree).
      ⚠️ **Sharper than the plan assumed — see the Task 2.1 note below: a bare
      `sync_device()` there is a silent NO-OP.**

**➕ DISCOVERED — the `/pool` mesh changed under us (blocker, resolved):**
A regression gate added to the cert job (Kokkos-Serial vs the standing `serref_m522_saved`
oracle) failed at **step 0** on `nlevels`/`nlevels_nod2D` — static topology fields that
cannot change unless the mesh does. Root cause: `/pool/.../core2/nlvls.out`+`elvls.out`
were **replaced on 2026-07-03** (2 nodes / 4 elems, Ross Sea shelf ≈154°W/77°S, all
shallower; sum(nlevels_nod2D) 3 832 750 → 3 832 745). The port reads those files directly,
so it is a real bathymetry change, and **every archived C/Fortran/Kokkos reference predates
it**. Resolution (user decision): a **private mesh copy** at
**`/work/ab0995/a270088/port2/mesh/core2`** with the pre-2026-07-03 levels restored from the
in-place `*.20260528_regenerated` backups (July-3 version kept as `*.20260703_pool`; `/pool`
untouched). Proof it is the right bathymetry: with it, gate 2 passes — a fresh Kokkos-Serial
run is **bit-identical to `serref_m522_saved`** (2026-05-31). **ALL M6 jobs must use
`MESH=/work/ab0995/a270088/port2/mesh/core2`.** See `$MESH/MESH_PROVENANCE.md` and the
warning block at the top of `docs/REFERENCE_RUNS.md`.

### Task 0.2: Phase 0 — reference inventory + vector-frame decision ✅ DONE (2026-07-12)

**Files:**
- Modify: `docs/REFERENCE_RUNS.md` (M6 reference table) ✅
- Create: `scripts/fesom_frame.py` (r2g rotation) + `scripts/m32_climate_compare.py` rework ✅
- Create: `jobs/m6_namelists/{tke,mevp,zstar}/` ✅

- [x] read all four PROVENANCE.md files; all archived /work refs verified present and
      openable at the right shape (12 months × 126858 nodes) — `c_tke_2yr`,
      `c_mevp_2yr`, `c_zstar_2yr`, `c_all3_1yr`, `fortran_linfs_tke`, `fortran_mevp_2yr`,
      `fortran_all3`, `fortran_linfs_2yr_b`. Every feature config re-confirmed single-knob
      (`mix_scheme='cvmix_TKE'` / `which_ALE='zstar'` / `whichEVP=1`).
      ⚠️ `/scratch/a/a270088/fortran_kpp_5yr_fix` (the old default Fortran anchor in
      `REFERENCE_RUNS.md`) has been **PURGED down to restart files** — its output `.nc` are
      gone. Use the purge-safe `/work/.../zstar/fortran_linfs_2yr_b` instead (same
      linfs+KPP config, and its binary POST-dates the `2682a9fb` sbc cold-start rotation fix).
- [x] vector frame of every C-port output determined — and **verified empirically**, not
      just inferred from commit dates (each rotated against its Fortran twin; r2g must
      improve a rotated leg and degrade a geo leg):
      **ROTATED** (pre-`75406d3`): `c_tke_2yr`, `c_zstar_2yr`, `c_zstar_tke_*`, `kpp_5yr_fix`.
      **GEO** (≥`75406d3`): `c_mevp_2yr`, `c_mevp_5yr`, `c_evp_2yr`, `c_all3_1yr`.
      Table lives in `scripts/fesom_frame.py:frame_of_c_output()`.
- [x] **DECISION — script-side rotation, NO C-leg regeneration.** This overrides the plan's
      stated default ("regenerate"), because the two reasons to regenerate both evaporated:
      (a) the mesh problem is fixed at the source by the private mesh (Task 0.1), and
      (b) the **Fortran anchors are geographic and can never be re-run** — so a script-side
      r2g is REQUIRED for every leg regardless. Once it exists it covers all four C legs at
      zero marginal cost. Equivalence is not assumed: the C campaign gated its in-model
      rotation against this exact offline transform at 7e-15 (job 25524763), and the
      transform is verified here as an isometry to 1e-16.
      Implemented in `scripts/fesom_frame.py`; wired into `scripts/m32_climate_compare.py`
      via `--cref-frame {geo,rotated}`. Policy: **everything rotated → geographic, then
      compare** (scalars are frame-free). Saves ~4 C CPU runs.
- [x] copied the three reference namelist sets → `jobs/m6_namelists/{tke,mevp,zstar}/`
      (all 10 namelists each + the upstream `PROVENANCE.md` + a README)
- [x] per-feature VERIFIED constant tables consolidated below (each C source value
      re-confirmed against the archived namelist — all match)

**➕ DISCOVERED — the "known F↔C ice-edge budget" (uice ≈0.92) is a FRAME ARTIFACT, not physics.**
`m32_climate_compare.py` compared `uice` but never `vice`, and applied no rotation — so it
was comparing the port's ROTATED components against Fortran's GEOGRAPHIC ones. The rotation
is an isometry, so |speed|/extent/volume looked perfect while the components decorrelated:
a plausible-looking wrong answer. Measured on the M5.23 CUDA 1-yr run vs Fortran linfs+KPP,
everything else held fixed:

| | as-written | after r2g |
|---|---|---|
| `uice` corr vs Fortran | 0.9187 | **0.9997** |
| `vice` corr vs Fortran | 0.4266 | **0.9998** |

0.9187 reproduces the recorded 0.919 exactly. Cross-checked on the C side too:
`c_tke_2yr` vs `fortran_linfs_tke` goes 0.9187 → **1.0000**. So the port reproduces
Fortran's ice velocity essentially perfectly; there is no ice-edge budget to explain.
Fixed in `scripts/m32_climate_compare.py` (rotation + `vice` added to `FIELDS`).
**Consequence for M6: the Fortran anchors are first-class, not caveated secondaries** — the
plan's "vector fields vs Fortran need … or scalar-only restriction" caveat is retired.
Corrected verdict table in `docs/REFERENCE_RUNS.md`; memory + `GPU_FIDELITY.md` updated.

#### VERIFIED constant tables (C source ⇄ archived namelist — all confirmed)

**TKE** — `fesom_tke.c:227-241` ⇄ `jobs/m6_namelists/tke/namelist.cvmix` `&param_tke`:

| C arg | value | namelist key | note |
|---|---|---|---|
| `c_k` | 0.1 | `tke_c_k` | |
| `c_eps` | 0.7 | `tke_c_eps` | |
| `cd` | **3.75** | `tke_cd` | ⚠️ NAMELIST value — the module default 1.0 LOSES |
| `alpha_tke` | 30.0 | `tke_alpha` | |
| `mxl_min` | 1.0e-8 | `tke_mxl_min` | |
| `kappaM_min` | 0.0 | `tke_kappaM_min` | |
| `kappaM_max` | 100.0 | `tke_kappaM_max` | |
| `tke_mxl_choice` | 2 | `tke_mxl_choice` | Blanke & Delecluse (only option) |
| `use_ubound_dirichlet` | 0 | — | module default F |
| `use_lbound_dirichlet` | 0 | — | module default F |
| `only_tke` | 1 | `tke_only` | |
| `l_lc` | 0 | `tke_dolangmuir` | |
| `clc` | 0.3 | — | module default; gate-only |
| `tke_min` | 1.0e-6 | `tke_min` | |
| `tke_surf_min` | 1.0e-4 | `tke_surf_min` | |

Plus the bit-fidelity landmines to transcribe verbatim: `TKE_C66 = 6.6` as a **plain double**
(the `-r8` rule), `TKE_POW32(x) = pow(x, 1.5)`, `TKE_MIN2/MAX2` as compare-select ternaries.
Port the C's **gate-only abort** too (`fesom_tke.c:246-253`): fail loudly if
`only_tke`/`l_lc`/either dirichlet/`mxl_choice != 2` ever deviate from the ported path.

**mEVP** — `fesom_ice.c:73-95` ⇄ `jobs/m6_namelists/mevp/namelist.ice`:

| param | value | note |
|---|---|---|
| `whichEVP` | 1 | `FESOM_WHICH_EVP=1`; 0=std EVP default; anything else → abort |
| `alpha_evp` | 250.0 | set UNCONDITIONALLY in C (`fesom_ice.c:91`) |
| `beta_evp` | 250.0 | set UNCONDITIONALLY in C (`fesom_ice.c:92`) |
| `evp_rheol_steps` | 120 | same subcycle count as std EVP |
| `Pstar` | 30000.0 | |
| `ellipse` | 2.0 | |
| `c_pressure` | 20.0 | |
| `delta_min` | 1.0e-11 | |
| `Cd_oce_ice` | 0.0055 | |
| `ice_ave_steps` | 1 | ⇒ `ice_dt` = ocean dt = 1800 s ⇒ **`rdt` = FULL step** (trap 1) |
| `theta_io` | 0.0 | present in the namelist but **NOT read** by the mEVP path (trap 3) |

**zstar** — archived namelists + the C plan's verified table:

| param | value | note |
|---|---|---|
| `which_ALE` | `'zstar'` | the one knob |
| `which_pgf` | `'shchepetkin'` | NOT in namelist.oce → module default |
| `use_virt_salt` | `.false.` | **DERIVED** from which_ALE ⇒ `is_nonlinfs = 1.0` |
| `use_floatice` | `.false.` | ⇒ `use_pice = 0` |
| `use_partial_cell` | `.false.` | |
| `i_vert_diff` | `.true.` | implicit TDMA carries the surface BCs |
| `use_wsplit` | `.false.` | (wsplit-ON is out of scope) |
| `use_ssh_se_subcycl` | `.false.` | ⇒ CG path |
| `opt_visc` | 7 | as in the linfs pair |
| `alpha` / `theta` | module defaults | not in namelist.dyn |

### Task 1.1: M6.1 — three-way mixing dispatch + TKE state ✅ DONE (2026-07-12)

**Files:**
- Create: `src/fesom_tke.h`, `src/fesom_tke.cpp` (state struct, alloc, dispatch entry) ✅
- Create: `src/fesom_cvmix_tke.hpp` (params block — pulled forward from Task 1.2, see note) ✅
- Create: `jobs/job_m6_gate_serial` (the reusable knob-OFF byte gate) ✅
- Modify: `src/fesom_step.cpp` / `.h`, `src/fesom_main.cpp`, `CMakeLists.txt` ✅

- [x] `FESOM_MIX_SCHEME` parse extended to `{KPP(default), PP, TKE}` — transcribed
      arg-for-arg from the C (`fesom_step.c:76-88`): leading `P`/`p` → PP; **exact** strings
      `TKE` or `cvmix_TKE` → TKE; everything else (incl. unset) → KPP. Anything looser would
      diverge from the oracle on a typo'd knob.
- [x] TKE state transcribed from `fesom_tke.{h,c}`: `tke` (prognostic — the Fortran comment
      calling it "diagnostic" is wrong), `tke_Av`, `tke_Kv` `[N*nl]`; `forc_normstress` /
      `forc_botfrict` / `forc_rhosurf` `[N]`; the 13 diag slabs `[N*nl]` gated by `diag_on`.
      All `fesom::Field` + raw non-owning alias (D12/M2.3 pattern, as `fesom_kpp.h`);
      zero-init via `Field::alloc` (== the C's `calloc`). `TKE_NL_MAX=128` guard ported.
- [x] allocated ONLY when the knob selects TKE — matches the C (`fesom_main.c:357-377`) and
      the Fortran (`oce_setup_step.F90:185-187`, init runs only for `mix_scheme_nmb==5`).
      `diag_on = FESOM_TKE_DIAG || FESOM_TKE_DUMP_DIR` exactly as C.
- [x] **knob-OFF byte gate: ALL FIELDS BIT-IDENTICAL** vs `m6_baseline_serial` (SLURM 26210193).
- [x] `build-cuda` compiles clean (no device kernels added yet).
- [x] ➕ **dispatch reachability smoke** (SLURM 26210206, not in the original plan): with
      `FESOM_MIX_SCHEME=TKE` the state allocates AND the step reaches the TKE branch (the
      Task-1.3 stub aborts, exit 134). Without this, a wrong string match would silently run
      KPP and make Task 1.5's bit-id gate fail in a maximally confusing way.

**Note on file staging:** `src/fesom_cvmix_tke.hpp` was created here (not in Task 1.2) holding
just the parameter block, because `fesom_tke_alloc`'s gate-only guard needs it — exactly as in
the C, where the params live in `fesom_cvmix_tke.h` and `fesom_tke_alloc` calls
`fesom_cvmix_init_tke`. Task 1.2 appends the column core to the same file. The 15 params are
`constexpr` (device-usable with no `__constant__` copy or params-struct push), so the C's
runtime abort on an unported option additionally becomes a **compile-time `static_assert`** —
strictly stronger; the runtime check is kept too.

### Task 1.2: M6.1 — cvmix-TKE column core as device function ✅ DONE (2026-07-12)

**Files:**
- Create: `src/fesom_cvmix_tke.hpp` (header-only KOKKOS_INLINE_FUNCTION column core) ✅
- Create: `tests/tke_core_twin/` (➕ the column-core twin gate — see below) ✅

- [x] `integrate_tke` (:415-987) + `solve_tridiag` transcribed as `KOKKOS_INLINE_FUNCTION`s;
      per-thread scratch capped at `TKE_NL_MAX = 128` (== `FESOM_MAX_LEVELS`).
- [x] landmines ported VERBATIM: `TKE_C66 = 6.6` as a plain double (the `-r8` rule),
      `tke_pow32(x) = Kokkos::pow(x, 1.5)` (a pow call, NOT `x*sqrt(x)`), tridiag via
      **reciprocal** (`fxa = 1.0/m; cp = c*fxa`, never `c/m`), compare-select min/max,
      exact evaluation order.
- [x] the C's dead-argument drops (`old_KappaM`/`old_KappaH`/`handle_old_vals`/`max_nlev`/
      `i`/`j`/`tstep_count`) and gate-only branch eliminations (IDEMIX, Langmuir, both
      Dirichlet BCs) kept exactly; `bottom_fric` kept in the signature though unread;
      `iw_diss` read UNCONDITIONALLY (:898) so the driver must pass a REAL zero column.
- [x] both backends compile clean.

**➕ COLUMN-CORE TWIN GATE (not in the original plan — added as cheap insurance):**
`tests/tke_core_twin/` builds the **C oracle's own `integrate_tke`** and the Kokkos
transcription into one binary and runs both over 4000 identical synthetic columns —
randomised over plausible ranges plus the edge cases the ocean actually produces (negative
`Nsqr`/unstable columns, zero shear, cold-start zero TKE, zero surface forcing, 2-level
columns). **Result: 2,358,224 values compared (tke_new, KappaM, KappaH + all 13 diag slabs),
ZERO mismatches — BIT-IDENTICAL.** Worth the 20 minutes: it isolates the ~250 lines of column
math from the driver, so if Task 1.5's full-model bit-id fails, the bug is *provably* in the
driver (column assembly / halos / Av-Kv wiring) and most of the search space is already gone.
Re-runnable any time: `bash tests/tke_core_twin/run.sh`.

**Design note — `WITH_DIAG` template parameter.** The 13 budget slabs are pure OUTPUTS
(nothing in `tke_new`/`KappaM_out`/`KappaH_out` reads them back), so the core is templated on
`WITH_DIAG` and the copy-out sits behind `if constexpr`. With diag off (the default, and the
bit-id gate config) the compiler proves all 13 `[129]`-double locals dead and eliminates them,
cutting the per-thread frame from ~33 KB to ~20 KB on CUDA. The COMPUTATION is still written
unconditionally, exactly as the C writes it — this is dead-code elimination of unused outputs,
not a reordering, so numerics and evaluation order are untouched (and the twin gate above runs
with `WITH_DIAG=true`, so the diag path is verified too). ⚠️ Even at ~20 KB/thread this is
2× the M5.24 TDMA kernels' frame (L72), so expect local-memory pressure on CUDA — note it at
Task 1.6's perf sanity, do NOT pre-optimise.

### Task 1.3: M6.1 — TKE driver kernel, halos, step wiring

**Files:**
- Modify: `src/fesom_tke.cpp` (driver kernel `fesom_tke_mixing_kk`)
- Modify: `src/fesom_step.cpp` (TKE branch calls driver; shared post-mo_convect halos
  unchanged at `:436-437`)

- [ ] transcribe the `calc_cvmix_tke` driver: column `parallel_for` over OWNED nodes only
      (gen:296,311 — do NOT extend to halo); build per-column inputs; call the core;
      write `tke`, node-`Av`, `Kv`
- [ ] surface forcing (`forc_tke_surf` from stress) transcribed exactly, incl. its
      `pow(x,1.5)`
- [ ] fused `tke_Kv`+`tke_Av` node halo via `fesom_halo_field2` BEFORE the Kv copy /
      node→elem average (C invariant; `tke` itself NEVER exchanged)
- [ ] node→elem Av average `parallel_for` over OWNED elements only (gen:500); elem-Av halo
      comes from the existing shared `:437` exchange — verify ordering matches C's step
- [ ] knob-OFF byte gate (step.cpp touched) → bit-identical vs baseline

### Task 1.4: M6.1 — snapshot + verify-key + output extension

**Files:**
- Modify: `src/fesom_io.cpp` (snapshot 4 sites × {tke, tke_Kv, tke_Av}; monthly table if C
  outputs tke)
- Modify: `src/fesom_main.cpp` (`sync_host()` for new fields pre-snapshot, `:1342-1346`)
- Modify: `src/fesom_io.h` (`struct fesom_state` additions)

- [ ] check what the C oracle writes in snapshots + monthly at TKE config (its io.c diff);
      mirror EXACTLY — snapshot field set must match the C oracle's for `diff_snap.py`
- [ ] add tke/tke_Kv/tke_Av at the 4 snapshot sites + sync_host; fields written only when
      knob=TKE if that is what C does (match C)
- [ ] if C has monthly `tke` output: add resolver + table row + bump NVARS
- [ ] knob-OFF byte gate (io.cpp touched) → bit-identical vs baseline
- [ ] knob-ON smoke: 1-step Serial run writes snapshots with the new fields, openable

### Task 1.5: M6.1 — knob-ON Serial bit-id vs C oracle

**Files:**
- Create: `jobs/job_m6_tke_serial_bitid` (paired C-oracle + Kokkos-Serial legs, TKE config)

- [ ] run C oracle with `FESOM_MIX_SCHEME=TKE` (8r, dist_8, dt=1800, 20 steps, snap 10);
      run Kokkos-Serial identically
- [ ] `diff_snap.py` → ALL FIELDS BIT-IDENTICAL (incl. tke fields)
- [ ] on mismatch: bisect via the C dump/replay rails
      (`/work/.../tke/cdump_v2`, `replay/`, C `job_tke_t2_*` patterns); throwaway host
      twin only if a single kernel resists
- [ ] record PASS (fields, worst-case) in GPU_FIDELITY §M6.1 draft notes

### Task 1.6: M6.1 — knob-ON CUDA gate + sync-log

**Files:**
- Create: `jobs/job_m6_tke_gpu_gate` (gate legs with `FESOM_MIX_SCHEME=TKE`)

- [ ] run the CORE2 active-ice CUDA-vs-Serial gate pair at TKE config
      (`gpu_fidelity_check.py`; expected M2.1-class floor ~1e-3)
- [ ] `build-cuda-synclog` one short TKE run: no new per-step D2H/H2D round-trippers
      (forcing/nod2D-class traffic only)
- [ ] note s/step vs same-day KPP-config run (perf sanity, no optimization)
- [ ] knob-OFF gate re-run (`gpu_fidelity_gate.sh`) still PASSES (default path clean)

### Task 1.7: M6.1 — 1-yr climate close (linfs+TKE)

**Files:**
- Create: `jobs/job_m6_tke_cuda_1yr` (from `job_m32_cuda_core2`, `FESOM_MIX_SCHEME=TKE`)
- Create: `jobs/job_m6_tke_c_1yr` (C comparator leg, only if Phase-0 decided regenerate)

- [ ] run Kokkos-CUDA 1 yr (17280 steps, monthly) at linfs+TKE
- [ ] comparator: `c_tke_2yr` year-1 (or regenerated rotated-frame leg per Phase 0)
- [ ] `m32_climate_compare.py`: sst/sss/ssh/ice corr ~1.0-class vs C; Kv correlation
      included (TKE changes Kv directly); ice fields under the mask rule
- [ ] secondary anchor: vs `fortran_linfs_tke` yr-1 (expect the known F↔C marginal-ice
      class of diffs, uice caveat)
- [ ] verdict recorded; no runaway/bounded T-S sanity

### Task 1.8: M6.1 — docs, commit, tag

**Files:**
- Modify: `docs/KOKKOS_PORTING_LESSONS.md` (L72+: TKE lessons)
- Modify: `docs/GPU_FIDELITY.md` (§M6.1: gates, climate, perf note)
- Modify: `docs/REFERENCE_RUNS.md` (TKE rows final)

- [ ] write the docs entries (what ported, gates passed, landmines encountered)
- [ ] commit series (src / jobs / docs split as usual); tag `m6.1-tke`; push
- [ ] verify knob-OFF gate one final time at the tag

### Task 2.1: M6.2 — whichEVP dispatch + mEVP state

**Files:**
- Modify: `src/fesom_ice_types.h` (add `uice_aux/vice_aux` + mEVP scratch per C's
  `fesom_ice_types.h` diff; Fields for each)
- Modify: `src/fesom_ice.cpp` (dispatch: mirror C `fesom_ice.c:73-90` semantics — unset/0
  std EVP, 1 → mEVP, else abort)
- Create: `src/fesom_ice_maevp.h` (decl)

- [ ] transcribe the C ice_types additions exactly (which arrays, sizes, zero-init)
- [ ] dispatch reads `FESOM_WHICH_EVP` once, static-cached, same strings/errors as C
- [ ] push `bc_index_nod2D_fld` to device after host population and wire the device read
      for the mEVP node-solve det. ⚠️ **It must be `modify_host(); sync_device();` — a bare
      `sync_device()` is a SILENT NO-OP.** Verified in Task 0.1: `Field::alloc`
      (`fesom_field.hpp:61-65`) tags the field `Auth::Synced` with **both spaces zeroed**,
      and `fesom_ice.cpp:229-241` then fills the mask **through the raw host pointer**,
      which never sets the dirty tag (L14). So today the device mirror is *all zeros* AND
      tagged clean → `dv_.sync_device()` sees `need_sync_device()==false` and copies
      nothing. Under CUDA the mEVP node-solve det would be multiplied by an all-zero
      `bc_index` (catastrophic, and invisible on Serial where host==device).
- [ ] knob-OFF byte gate (ice.cpp touched) → bit-identical
- [ ] `gpu_fidelity_gate.sh` knob-OFF still PASSES

### Task 2.2: M6.2 — mEVP device kernels (EVPdynamics_m)

**Files:**
- Create: `src/fesom_ice_maevp.cpp`
- Modify: `CMakeLists.txt` (explicit `FESOM_SRC` list at `:59-74` — new `.cpp` must be added)

- [ ] transcribe `fesom_ice_evp_dynamics_m` per-substep structure into kernels mirroring
      the device std-EVP shape (`fesom_ice_evp.cpp:627-716` as template for STRUCTURE
      ONLY — values/branches from the C mEVP, never from std-EVP)
- [ ] port the fidelity-traps checklist AS-IS: rdt=FULL `ice_dt`; no 0.5 in
      `pressure_fac`; no `theta_io` rotation; elem mask `mean-msum>0.01` / node mask
      `a_ice>=0.01`; non-ice nodes SKIPPED (velocity retained) not zeroed;
      `uice_old/vice_old` untouched; sigma11/12/22 NOT zeroed on entry (persist);
      `bc_index_nod2D` multiplies the node-solve det (redundant with edge-BC loop — port
      BOTH)
- [ ] elem→node RHS assembly via `Kokkos::atomic_add` element-order scatter (D22),
      including the unguarded halo-entry writes (trap 6) exactly as C
- [ ] per-substep fused `fesom_halo_field2(uice_aux, vice_aux, NOD2D)` replacing C's two
      blocking `exchange_nod2D` (proven bit-id-neutral, M5.23-L1); owned-only rhs zeroing
      order preserved (result-identical note in C `:339-342`)
- [ ] final owned+eDim copy with NO extra exchange (trap 8)
- [ ] compile both backends; knob-OFF byte gate (no shared files should have changed —
      confirm with git diff)

### Task 2.3: M6.2 — snapshot/verify keys + dump rail

**Files:**
- Modify: `src/fesom_io.cpp` (+ `src/fesom_main.cpp` sync_host) — only if the C oracle's
  snapshot set differs at mEVP config
- Modify: `src/fesom_ice_maevp.cpp` (env-gated `FESOM_EVP_DUMP_DIR`-equivalent dumps)

- [ ] verify sigma11/12/22 + uice_aux/vice_aux presence in snapshots matches the C
      oracle's set exactly (C snapshot code is shared — likely already aligned; verify,
      then extend only if C does)
- [ ] port the C's mEVP per-substep dump machinery (same file format/naming) so
      Kokkos-side bisection dumps are diffable against `/work/.../mevp/cdump_16r`
      (proactive rail — the deliberate exception to the throwaway-aids rule: mEVP is
      NR-optimized with three routines inlined and carries the densest fidelity-trap
      list of the campaign, so the bisection rail is expected to be needed)
- [ ] knob-OFF byte gate if io touched; knob-ON 1-step smoke with dumps on

### Task 2.4: M6.2 — knob-ON Serial bit-id vs C oracle

**Files:**
- Create: `jobs/job_m6_mevp_serial_bitid`

- [ ] paired 8r/20-step/snap10 runs: C oracle `FESOM_WHICH_EVP=1` vs Kokkos-Serial same
- [ ] `diff_snap.py` → ALL FIELDS BIT-IDENTICAL
- [ ] on mismatch: per-substep dump diff (Task 2.3 rail) vs C dumps; the C `job_mevp_*`
      jobs show the exact dump-run configs
- [ ] record PASS in GPU_FIDELITY §M6.2 draft

### Task 2.5: M6.2 — knob-ON CUDA gate + sync-log

**Files:**
- Create: `jobs/job_m6_mevp_gpu_gate`

- [ ] CUDA-vs-Serial gate at mEVP config (active-ice CORE2 — mEVP is THE ice change;
      scrutinize ice-field ceilings)
- [ ] synclog run: no new PCIe round-trippers in the subcycle loop
- [ ] s/step noted vs same-day std-EVP run (mEVP substep count differs — note only)
- [ ] knob-OFF `gpu_fidelity_gate.sh` re-run PASSES

### Task 2.6: M6.2 — 1-yr climate close (linfs+KPP+mEVP)

**Files:**
- Create: `jobs/job_m6_mevp_cuda_1yr`; C comparator leg job if regenerating

- [ ] Kokkos-CUDA 1 yr at mEVP config; comparator `c_mevp_2yr` yr-1 (frame decision from
      Phase 0; ice vectors affected — likely regenerate rotated or compare via script
      rotation)
- [ ] `m32_climate_compare.py` + ice mask rule; expect a_ice/m_ice/uice corr ~1.0-class
      vs C; `fortran_mevp_2yr` secondary anchor (uice F↔C caveat applies)
- [ ] `c_evp_2yr` (std-EVP control) available to attribute any diff class
- [ ] verdict recorded

### Task 2.7: M6.2 — docs, commit, tag

**Files:**
- Modify: `docs/KOKKOS_PORTING_LESSONS.md`, `docs/GPU_FIDELITY.md` (§M6.2),
  `docs/REFERENCE_RUNS.md`

- [ ] docs entries; commit series; tag `m6.2-mevp`; push
- [ ] final knob-OFF gate at the tag

### Task 3.1: M6.3 — FESOM_ALE knob + zstar geometry state + thickness init/commit

**Files:**
- Modify: `src/fesom_mesh.h` (+alloc site): `Z_3d_n` [nod2D*nl] DualView, `dhe` [elem2D],
  `bottom_node_thickness`/`bottom_elem_thickness` (static); zbar_3d_n comment → LIVE
  under zstar
- Modify: `src/fesom_ale.cpp`/`.h` (knob; `init_thickness_ale` zstar case;
  `update_thickness_ale` zstar commit kernel)
- Modify: `src/fesom_ic.cpp` (zstar IC path)

- [ ] `FESOM_ALE=linfs|zstar` knob (default linfs), read-once static, mirroring C
- [ ] new arrays allocated + initialized per C (`bottom_*` from nominal zbar spacing,
      full-cell branch; `Z_3d_n` init = Z[nz] pattern; `dhe` zeroed)
- [ ] transcribe `init_thickness_ale` zstar case: hnode=(zbar spacing)·(1+hbar/dd) for
      nz=1..nlevels_nod2D_min-2, bottom keeps nominal, hnode_new=hnode, helem=mean(3),
      dhe=mean(hbar), `exchange_elem(helem)` device halo
- [ ] transcribe `update_thickness_ale` zstar commit: per-step over myDim+eDim,
      bottom→top hnode=hnode_new + zbar_3d_n/Z_3d_n rewrite + helem mean +
      `exchange_elem(helem)`; skip ldiag_DVD rescue
- [ ] wire the linfs/zstar branch so the linfs path is UNTOUCHED code (the step-1
      hnode_new seed stays linfs-only)
- [ ] knob-OFF byte gate (shared files touched) → bit-identical

### Task 3.2: M6.3 — forcing flip (real water/salt fluxes)

**Files:**
- Modify: `src/fesom_forcing.cpp`, `src/fesom_ice_coupling.cpp` (real_salt_flux wiring),
  tracer surface-BC site (per C's `fesom_tracer_diff.c` diff)

- [ ] transcribe the `use_virt_salt=.false.` derived path: `is_nonlinfs=1.0`,
      water_flux as REAL volume flux, `real_salt_flux` as a LIVE producer (C plan-review
      finding — the C wires `rsf` under zstar; mirror its final state)
- [ ] S surface-BC = `+dt·(virtual_salt + relax_salt + real_salt_flux·is_nonlinfs)`
      (no sval·wf term) — transcribed into the implicit-diffusion BC kernel
- [ ] ordering: the flip lands BEFORE the SSH/vert-vel phases in the step (C plan
      ordering; their dumps consume water_flux)
- [ ] under linfs the derived flags reproduce today's virtual-salt path exactly
- [ ] knob-OFF byte gate → bit-identical

### Task 3.3: M6.3 — SSH plumbing (stiffness update + rhs/hbar tails)

**Files:**
- Modify: `src/fesom_ssh.cpp`/`.h` (`update_stiff_mat_ale` device kernel; per C's
  fesom_ssh.c diff incl. any pr_values/preconditioner handling)
- Modify: `src/fesom_step.cpp` (call gate: non-linfs → update_stiff_mat BEFORE ssh_rhs)
- Modify: `src/fesom_momentum.cpp` if the dhe fill lives there (C: compute_hbar_ale)

- [ ] transcribe `update_stiff_mat_ale`: CUMULATIVE edge-loop
      `values[npos] += -dhe(elem)·(gradient_sca×edge_cross_dxdy)·g·dt·alpha·theta` with
      i/j sign flips + npos column matching — device kernel over edges with
      `Kokkos::atomic_add` on `values_fld` (D22 pattern; Serial loop-order deterministic)
- [ ] mirror EXACTLY what C does about the preconditioner (`pr_values`) after the matrix
      update — transcribe, don't infer
- [ ] transcribe `compute_ssh_rhs_ale` non-linfs tail
      (`-alpha·water_flux·areasvol + (1-alpha)·ssh_rhs_old`)
- [ ] transcribe `compute_hbar_ale` non-linfs bits: `ssh_rhs_old -= water_flux·areasvol`
      + `exchange_nod(ssh_rhs_old)` (non-linfs-ONLY exchange) + the unconditional dhe
      fill (runs under linfs too, unused — port as written)
- [ ] knob-OFF byte gate → bit-identical (ssh/step touched — the highest-risk gate; also
      re-run `gpu_fidelity_gate.sh` knob-OFF here)

### Task 3.4: M6.3 — vert_vel zstar branch + the hnode_new rail

**Files:**
- Modify: `src/fesom_ale.cpp` (vert_vel zstar branch kernel; halo calls)

- [ ] transcribe `vert_vel_ale` zstar branch over myDim: dd/dddt from
      (hbar-hbar_old)/(zbar_3d_n span); `Wvel(nz) -= (zbar_3d_n(nz)-dd1)·dddt`;
      `hnode_new(nz) = hnode(nz) + (zbar_3d_n span)·dd`; `Wvel(1) -= water_flux(n)`;
      NaN check as C does it
- [ ] shared tail: negative-hnode_new fatal check (myDim+eDim) +
      `exchange_nod(Wvel)` + `exchange_nod(hnode_new)` as device halos — **this restores
      the M5.20 hnode_new rail under zstar**; linfs keeps the seed/no-halo fast path
- [ ] wsplit interaction: `w_e=w, w_i=0` unchanged (wsplit OFF; zstar reference runs
      use_wsplit=F — no new coupling)
- [ ] knob-OFF byte gate → bit-identical
- [ ] knob-ON proactive GEOMETRY diff (plan-review de-risk): pull the thickness/geometry
      snapshot-key extension forward from Task 3.6 if the C oracle's zstar snapshot set
      carries them, then paired 3-step snap_every=1 runs + `diff_snap.py` on
      hnode/hnode_new/hbar/zbar_3d_n/Z_3d_n (else minimal ale-dump twin vs
      `/work/.../zstar/z2_cdump`) — geometry must match bit-for-bit BEFORE the PGF/SSH
      layers build on it; fatal checks stay silent

### Task 3.5: M6.3 — Shchepetkin PGF + hpressure gating

**Files:**
- Modify: `src/fesom_eos.cpp`/`.h` (new `pressure_force_4_zxxxx_shchepetkin` kernel;
  hpressure gated linfs-only; dispatcher per C)

- [ ] transcribe the ~235-line `pressure_force_4_zxxxx_shchepetkin` (density-Jacobian on
      moving levels; reads `density_m_rho0` + LIVE `Z_3d_n`/`zbar_3d_n`; self-contained —
      zero hpressure references)
- [ ] gate the existing hpressure computation to linfs (under zstar Fortran computes NO
      hpressure — C mirrors this; so do we)
- [ ] PGF dispatcher: linfs → existing `pressure_force_linfs_fullcell_kk`; zstar →
      shchepetkin (which_pgf='shchepetkin' is the module default the reference uses)
- [ ] knob-OFF byte gate → bit-identical (eos.cpp heavily shared)

### Task 3.6: M6.3 — geometry re-points audit + snapshot keys

**Files:**
- Modify: audit-driven (expected mostly verification): `src/fesom_eos.cpp`,
  `src/fesom_gm.cpp` (scaling_GMzexp), `src/fesom_pp.cpp` (dz_inv), `src/fesom_kpp.cpp`,
  `src/fesom_tke.cpp`, `src/fesom_tracer_adv.cpp` (dual-geometry vertical advection),
  `src/fesom_tracer_diff.cpp`, mo_convect
- Modify: `src/fesom_io.cpp` + `src/fesom_main.cpp` (snapshot keys)

- [ ] grep-audit every consumer of static `zbar`/`Z`/`zbar_3d_n` in device kernels against
      the C oracle's final state (C plan: audit greps the geometry ARRAYS, not comments);
      re-point to live `zbar_3d_n`/`Z_3d_n` wherever C does; comment-only sites
      (`fesom_eos.cpp:184,795`, `fesom_pp.cpp:151`, `fesom_tracer_adv.cpp:616`) become
      real reads where C made them real
- [ ] TKE + mEVP kernels (already ported reading C's arrays): verify they read the live
      arrays — expected no-change if Tasks 1.x/2.x transcribed faithfully
- [ ] snapshot/verify keys gain hnode/hnode_new/helem/zbar_3d_n/Z_3d_n/dhe to match the
      C oracle's zstar snapshot set (mirror exactly)
- [ ] knob-OFF byte gate → bit-identical (the widest-touch task — gate carefully)

### Task 3.7: M6.3 — knob-ON Serial bit-id vs C oracle (zstar)

**Files:**
- Create: `jobs/job_m6_zstar_serial_bitid`

- [ ] paired 8r/20-step/snap10: C oracle `FESOM_ALE=zstar` vs Kokkos-Serial same
- [ ] `diff_snap.py` → ALL FIELDS BIT-IDENTICAL (incl. thickness/geometry fields)
- [ ] on mismatch: the C `fesom_ale_dump` dual-dump harness + `/work/.../zstar/z2_cdump`,
      `fdump` (+ `fdump_k2`) rails; bisect phase-by-phase (Z-ladder order)
- [ ] record PASS in GPU_FIDELITY §M6.3 draft

### Task 3.8: M6.3 — knob-ON CUDA gate + sync-log

**Files:**
- Create: `jobs/job_m6_zstar_gpu_gate`

- [ ] CUDA-vs-Serial gate at zstar config
- [ ] synclog: the thickness/geometry chain stays device-resident — no new per-step
      round-trippers (hnode_new/Z_3d_n/zbar_3d_n/dhe must NOT ping-pong)
- [ ] s/step noted vs same-day linfs run (zstar adds stiffness update + thickness
      kernels + 2 halos — note the cost, no optimization)
- [ ] knob-OFF `gpu_fidelity_gate.sh` re-run PASSES

### Task 3.9: M6.3 — 1-yr climate close (zstar+KPP+EVP)

**Files:**
- Create: `jobs/job_m6_zstar_cuda_1yr`; C comparator leg if regenerating

- [ ] Kokkos-CUDA 1 yr at zstar config; comparator `c_zstar_2yr` yr-1 (frame decision)
- [ ] `m32_climate_compare.py` + mask rule; `fortran_zstar_2yr` secondary anchor;
      ssh corr scrutinized (zstar changes the SSH operator)
- [ ] bounded hnode sanity (no drift in total thickness vs C)
- [ ] verdict recorded

### Task 3.10: M6.3 — docs, commit, tag

**Files:**
- Modify: `docs/KOKKOS_PORTING_LESSONS.md`, `docs/GPU_FIDELITY.md` (§M6.3),
  `docs/REFERENCE_RUNS.md`

- [ ] docs entries (esp. the hnode_new rail restoration + stiffness-update pattern);
      commit series; tag `m6.3-zstar`; push
- [ ] final knob-OFF gate at the tag

### Task 4.1: M6.4 — triple-config Serial bit-id

**Files:**
- Create: `jobs/job_m6_all3_serial_bitid`

- [ ] paired 8r/20-step/snap10: C oracle with ALL THREE knobs
      (`FESOM_ALE=zstar FESOM_MIX_SCHEME=TKE FESOM_WHICH_EVP=1`) vs Kokkos-Serial same
- [ ] `diff_snap.py` → ALL FIELDS BIT-IDENTICAL — the composition proof; expected zero
      new code (faithful transcription pre-composes: TKE reads live geometry, mEVP's
      levitating ssh2rhs branch covers zstar)
- [ ] any composition mismatch → bisect by knob-pair (zstar+TKE, zstar+mEVP, TKE+mEVP)

### Task 4.2: M6.4 — triple-config CUDA gate

**Files:**
- Create: `jobs/job_m6_all3_gpu_gate`

- [ ] CUDA-vs-Serial gate at the triple config + synclog spot-check
- [ ] knob-OFF `gpu_fidelity_gate.sh` PASSES (campaign-long invariant holds)

### Task 4.3: M6.4 — the combined climate twin (1 yr)

**Files:**
- Create: `jobs/job_m6_all3_cuda_1yr`; C comparator leg if regenerating

- [ ] Kokkos-CUDA 1 yr at the triple config
- [ ] comparator: the EXISTING `/work/.../mevp/c_all3_1yr` (C campaign already ran the
      triple — verify its exact config vs `job_all3_1yr` in Task 0.2; frame decision
      applies) — regenerate rotated-frame leg only if needed
- [ ] `m32_climate_compare.py` full-field verdict vs C (+ `fortran_all3` anchor)
- [ ] verdict: corr ~1.0-class vs C twin = campaign PASS

### Task 4.4: M6.4 — acceptance sweep, campaign docs, close-out

**Files:**
- Modify: `docs/KOKKOS_PORTING_LESSONS.md`, `docs/GPU_FIDELITY.md` (§M6.4 + campaign
  summary), `docs/REFERENCE_RUNS.md`, `README.md` (supported-options table)
- Move: this plan → `docs/plans/completed/`

- [ ] acceptance sweep: all Overview claims verified (3 knobs functional, defaults
      untouched, 4 tags exist, every ladder rung green); re-run knob-OFF Serial bit-id +
      gpu gate one last time
- [ ] README options table updated (linfs|zstar, KPP|PP|TKE, EVP|mEVP now supported)
- [ ] campaign summary in GPU_FIDELITY §M6 (per-feature gates table, climate verdicts,
      s/step deltas)
- [ ] commit; tag `m6.4-options-twin`; push
- [ ] move this plan to `docs/plans/completed/20260712-m6-options-tke-mevp-zstar.md`

## Technical Details

- **MESH (campaign-wide, non-negotiable)**: `MESH=/work/ab0995/a270088/port2/mesh/core2`
  — the M6 private copy, NOT `/pool`. `/pool`'s `nlvls.out`/`elvls.out` were replaced on
  2026-07-03 and are NOT the bathymetry any archived reference was run on (Task 0.1).
  Expected level-sums: `sum(nlvls)=3832750`, `sum(elvls)=7366752`.
- **Bit-id standard run**: CORE2 dist_8, 8 ranks, dt=1800, 20 steps, snap_every=10,
  `OMPI_MCA_btl_vader_single_copy_mechanism=none` (L18 — required for any comparison
  across two different binaries), `diff_snap.py` zero tolerance (pattern:
  `jobs/job_m6_oracle_cert`). Knob-OFF baseline:
  `/work/ab0995/a270088/port2/m6_baseline_serial/`.
- **C oracle binary**: `/home/a/a270088/port2/fesom2_port_zstar/build-m6oracle/fesom_port`
  (fresh Release build @ `df8b9a8`; the tree's own `build/` predates `75406d3` — do not use it).
- **Climate run**: `job_m32_cuda_core2` derivative, `M32_NSTEPS=17280` (1 yr),
  `SNAP_EVERY=1440` (monthly), 2 nodes / 8×A100, `m32_climate_compare.py`.
- **Scatter pattern**: all new elem→node and edge→CSR accumulations use element/edge-order
  `Kokkos::atomic_add` (D22): Serial backend = sequential loop order = bit-identical;
  CUDA nondeterminism is covered by the fidelity gate (climate-close class).
- **Halo fusions allowed from day one** (proven bit-id-neutral M5.23): `fesom_halo_field2`
  for (tke_Kv,tke_Av) and (uice_aux,vice_aux); zstar's Wvel/hnode_new exchanges stay as C
  wrote them (two calls) unless trivially fusable — faithfulness beats micro-perf here.
- **Env knobs** (all read-once, static-cached, C-identical semantics):
  `FESOM_MIX_SCHEME=KPP|PP|TKE`, `FESOM_WHICH_EVP=0|1`, `FESOM_ALE=linfs|zstar`.
- **New DualViews**: TKE (tke, tke_Kv, tke_Av, forc, 13 diag slabs); mEVP (uice_aux,
  vice_aux + C's scratch additions); zstar (Z_3d_n, dhe, bottom_node/elem_thickness;
  zbar_3d_n/hnode/hnode_new/helem go live-updating).
- **Reference-value table for zstar** (from the C plan, verified): which_pgf=shchepetkin
  (module default), use_virt_salt=false→is_nonlinfs=1.0, use_floatice=false,
  use_partial_cell=false, i_vert_diff=true, alpha/theta from namelist.oce,
  use_wsplit=false, use_ssh_se_subcycl=false.
- **SLURM budget** (approx): per feature 2 short paired bit-id runs + 2 gate legs +
  1×1-yr CUDA (~2-3 h wall on 2 GPU nodes) + optional C comparator leg (CPU); Phase 0
  two short runs. Existing 2-yr C/Fortran outputs reused as comparators where
  frame-compatible.

## Post-Completion

**Deferred by user decision:**
- NG5/dars production-config GPU perf re-measure with the new physics (paper-grade
  SYPD with zstar+TKE+mEVP) — natural next campaign after M6.4.
- wsplit-ON path: fix the C port's known-imperfect implementation first (vs Fortran
  wsplit references), then mirror in Kokkos. Relevant for NG5 cold-start robustness.

**Discovered, optional:**
- Port the C tree's `FESOM_IO_VECTOR_FRAME` geographic-frame vector output (default geo)
  to the Kokkos port — output-parity nicety, not physics.
- Mixed precision (the standing ≈×2 SYPD lever) — unchanged by this campaign; the three
  new options add column kernels/halos that would benefit identically.

**Manual verification:**
- After M6.4: eyeball the combined-twin monthly maps (sst/ssh/ice) vs the C twin — the
  correlation gate is necessary but a human look at the fields is the tradition here.
