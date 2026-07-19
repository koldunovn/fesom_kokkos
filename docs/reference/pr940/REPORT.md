# PR 940 investigation report: FESOM2 single-precision branch

*Mined 2026-07-19 for the M8 mixed-precision campaign (agent-collected from
github.com/FESOM/fesom2 pull 940; raw diff + head-branch files in this directory).*

## 1. Metadata

- **Title:** "Support Single Precision mode across the model" — https://github.com/FESOM/fesom2/pull/940
- **Author:** suvarchal (Suvarchal Kumar Cheedela, AWI)
- **State:** OPEN, **DRAFT**. Created 2026-06-23, last updated 2026-07-06 (last commit `636a60a3`). Not merged, not closed.
- **Branches:** `support_sp` → `main`. Milestone: **FESOM 2.8** (due 2026-08-28) — i.e., slated for a real release.
- **Size:** 37 files, +1088/−529. Zero review comments, zero issue comments, zero reviews, no linked issues — all knowledge is in the diff, commit messages, and PR body.
- **Commits (the story arc):** `5a91ce60` add WP=4 build support → `48c37328` **fix SP NaN in KPP (epsln underflow)** → `e88de1ac` raise stack limit in tests → `01fa1cfd` **run CVMix in double precision under SP builds** → `76ac7a3b` report output-precision>WP fields → `5b3d7500` CVMix DP+SP integration tests → `d224cdf9` SP-only CI workflow → `636a60a3` YAML fix.

## 2. The mechanism

WP flip plus three synchronized companions:

**(a) WP parameter** — `src/oce_modules.F90`, MODULE o_PARAM:
```fortran
#if defined(USE_SINGLE_PRECISION)
integer, parameter :: WP=4        ! Single precision
#else
integer, parameter :: WP=8        ! Double precision (default)
#endif
```

**(b) CMake flag** — `USE_SINGLE_PRECISION` (OFF by default), defined in `src/CMakeLists.txt`, sets the preprocessor macro AND flips the compiler default-real kind per compiler: Intel `-r8`→`-r4`, GNU `-fdefault-real-8 -fdefault-double-8`→(nothing), Cray `-s real64`→`-s real32`, NVHPC `-Mr8`→`-Mr4`. This second half is essential: FESOM has many bare `real ::` declarations that ride on the default-real promotion, so the macro alone would produce a mixed-kind mess.

**(c) MPI datatype** — `src/MOD_PARTIT.F90`:
```fortran
#if defined(USE_SINGLE_PRECISION)
integer, parameter :: MPI_WP = MPI_REAL             ! single precision
#else
integer, parameter :: MPI_WP = MPI_DOUBLE_PRECISION ! double precision (default)
#endif
```
Every `MPI_DOUBLE`/`MPI_DOUBLE_PRECISION` in halo derived types (`gen_modules_partitioning.F90` `MPI_TYPE_INDEXED`), broadcasts, gathers, and Allreduce calls becomes `MPI_WP` (~150 sites across solver.F90, write_step_info.F90, gen_support.F90, oce_mesh.F90, gen_ic3d.F90, gen_surface_forcing.F90, icb_step.F90, cavity_param.F90, oce_dyn.F90...). All `real(real64)` dummy arguments in `gen_halo_exchange.F90` become `real(kind=WP)`.

**(d) Generic netCDF interface** — `src/io_netcdf_nf_interface.F90` gains `nf_get_vara_x`/`nf_put_vara_x` generics that dispatch on the actual argument's kind (real(4) vs real(8)), so WP buffers automatically read double files with netCDF doing the conversion. Caveat found: the generic only resolves rank-1 actuals, forcing a read-into-flat-buffer + `reshape` workaround in `gen_ic3d.F90`.

## 3. The exception list — what stays real64 despite the SP switch

Short, and that is itself the finding. "Pure SP" with four deliberate FP64 islands:

| What stays double | Where | Why |
|---|---|---|
| **The entire CVMix library** (KPP, TKE, IDEMIX, PP, tidal) | `src/cvmix_driver/gen_modules_cvmix_{kpp,tke,idemix,pp,tidal}.F90` | CVMix has a fixed kind `cvmix_r8 = real64`. FESOM shims copy-in/copy-out WP↔cvmix_r8 buffers at every call site: "convert WP<->cvmix_r8 at the CVMix call boundary so CVMix always runs in double precision. In a double-precision FESOM build cvmix_r8==WP, so the casts/temps are exact no-ops and the result is bit-identical." Scalars are cast inline (`real(dt, cvmix_r8)`), literals re-suffixed (`1.0_WP`→`1.0_cvmix_r8`), and aliased inout args (Mdiff_out and old_Mdiff bound to the same actual) are served by ONE seeded buffer to preserve aliasing semantics. |
| **On-disk file format + I/O gather path** | `src/io_fesom_file.F90` | `global_level_data(:)` and the `laux` gather/scatter buffers stay `real(kind=8)`. PR body: "Mesh/restart/forcing files remain double on disk; WP arrays convert on I/O." SP and DP builds interchange restart files. |
| **Output mean accumulation for accuracy-8 fields** | `src/io_meandata.F90` | Pre-existing `local_values_r8` `real(real64)` accumulation buffers are kept; a namelist.io field requesting precision 8 in a WP=4 build is "still honoured (written as NF_DOUBLE; means accumulated in real64), but its samples are single-precision-sourced" — with a once-per-run startup report listing such fields (`n_wp_promoted` tally). |
| **FillValue/missing_value attribute reads** | `src/gen_ic3d.F90` | Read into `real(8) :: FILL_VALUE_r8` because "file stores fill/missing as double", then `FILL_VALUE = real(FILL_VALUE_r8, WP)` — "comparison below uses the same WP rounding as the data read". Comparing WP-rounded data against an unrounded double fill value never matches. |

**Equally important — deliberate demotions to WP (NOT kept double):**
- **All global MPI reductions**: CG solver dot products (`solver.F90` ssh_solve_cg `MPI_Allreduce(...MPI_WP...)`), ocean area/volume sums (`oce_mesh.F90` mesh_areas, check_total_volume), tracer/field diagnostics (`write_step_info.F90`, ~40 reductions), forcing integrals (`gen_support.F90` integrate_nod/elem). Global conservation sums run in FP32 in an SP build. No compensated summation anywhere. **[M8 deliberately does NOT copy this — FP64 accumulators are free in Kokkos.]**
- **The EOS**: densityJM was already `real(kind=WP)`; the PR even demoted hardcoded `real(kind=8)` locals in `init_ref_density_advanced` to WP (one dead `real(kind=8) :: T, S` leftover survives in `init_ref_density`, harmless because unused).
- **Clock/calendar**: `gen_modules_clock.F90` `timeold, timenew` (seconds-in-day) are `real(kind=WP)` — FP32 time-of-day (ulp ~0.008 s at 86400; fine for dt≥minutes, a trap for sub-second dt). **[M8 calendar stays a double island.]**
- **Iceberg thermodynamics** (`icb_thermo.F90`), **transient tracer gas fluxes** (`mod_transit.F90`), **forcing interpolation** (`gen_modules_read_NetCDF.F90`): all hardcoded `real(kind=8)`→WP.
- **pi itself**: `real(kind=WP), parameter :: pi=3.14159265358979` becomes an FP32 constant.

**What needed no changes at all** (already rigorously WP-kind-disciplined, flips silently): sea-ice EVP dynamics (`ice_maEVP.F90` — verified on head: 164 WP-kind uses, zero real(8)), ice thermodynamics, FCT tracer advection, pressure/ALE core, mixing schemes other than the epsln line. This is why the PR is only 37 files.

## 4. Problems found and fixes

1. **THE headline bug — SP NaN in KPP at cold start from a denormal epsilon** (commit `48c37328`, `src/oce_ale_mixing_kpp.F90`):
```fortran
! KPP divide-by-zero guard added to denominators (/(wm+epsln), /(hbl+epsln),
! /(dVsq+Vtsq+epsln), (ws*hbl+epsln), ...). It MUST stay a NORMAL number in the
! active working precision: 1.0e-40 is a denormal in real32 and Intel -O3 flushes
! it to 0 (FTZ/DAZ), which defeats every guard -> 0/0 = NaN in single precision at
! the cold start. Single precision therefore uses a value that survives FTZ; double
! precision keeps 1.0e-40 so the default build is numerically unchanged.
#if defined(USE_SINGLE_PRECISION)
  real(KIND=WP), parameter :: epsln = 1.0e-20_WP ! a small value (SP: normal, FTZ-safe)
#else
  real(KIND=WP), parameter :: epsln = 1.0e-40_WP ! a small value
#endif
```
2. **CVMix cannot run natively in SP** — resolved by the DP-island shim, not by porting CVMix.
3. **Intel builds SIGSEGV on core2-size meshes under the default 8 MB stack** ("large per-column automatic arrays"); fixed with `ulimit -s unlimited` around every test launch. Not SP-specific, but surfaced by the SP campaign — precision changes shift resource footprints.
4. **Vector rotation of double output buffers** (`io_meandata.F90` `io_r2g`): r8-accuracy output buffers are real64 while `vector_r2g` works in WP — fixed by round-tripping through WP temps; also fixed a latent OpenMP bug (temps added to `PRIVATE`).
5. **Generic netCDF interface rank limitation** — rank-1 buffer + `reshape` workaround.
6. **FillValue comparison mismatch** — cast-then-compare fix.

No reported drift, conservation violation, or solver stagnation — but nothing longer than 1 year was run, and nobody has reviewed the PR yet.

## 5. Validation evidence

- **CORE2 mesh, 1 year, Levante 1 node / 64 Intel tasks, native KPP** (not CVMix): SP **22.0574 SYPD** vs DP **13.3427 SYPD** — PR claims "~1.69x" (the quotient is 1.65x). Spatial SP-vs-DP plots for a_ice (both poles), SST, SSH plus 1-year timeseries of a_ice/SST/SSS/SSH — qualitative "stay close", no norms given.
- **PI mesh (~3.1k nodes), 1 month, GNU/OpenMPI, mpi2 + mpi8**: ~1.21x wall time, SYPD +21%/+23%. **Per-component SP/DP gains: oce_mix_pres ~1.6x, oce_ssh_solve ~1.55x, oce_timestep_ale ~1.27x, ice_timestep ~1.01x (neutral).** Field deltas after 1 month at mpi2: SST max|Δ| ≈ 0.24 °C, SSS ≈ 0.05 psu, SSH ≈ 2.3 mm.
- **CI/test infra**: every fesom.x run prints a precision banner ("SINGLE/DOUBLE PRECISION MODE", `fesom_module.F90`) which CTest asserts as a mandatory success marker; whole existing local suite reruns under the SP build; new workflow `fesom2_sp_ctest.yml`, `continue-on-error: true` while stabilising.
- **No bit-repro or restart-equivalence tests, no conservation numbers, no multi-year run, no GPU numbers.** DP default build claimed numerically unchanged (flag off; CVMix casts no-ops).

## 6. Status

Open **draft**, actively developed until 2026-07-06, milestoned FESOM 2.8 (late Aug 2026), zero review activity. Head branch `support_sp` in FESOM/fesom2 (tip `636a60a3`). Alive and release-tracked, but pre-review; validation shown is engineering-grade (1 month + 1 year), not climate-grade.

## 7. Top transferable lessons for the Kokkos MP port

1. The mechanism is one typedef + one MPI trait; the cost is chasing strays (hardcoded kinds in comms, I/O, side modules). Kokkos advantage: template kind mismatches are compile errors, not silent Fortran promotions.
2. **Epsilon guards must be NORMAL numbers in FP32 under FTZ/DAZ** — the cold-start NaN class (fields still zero ⇒ degenerate denominators). Audit every additive epsilon; ≥ ~1e-20 in FP32; per-precision so FP64 unchanged. Mechanically rhymes with the M5.24/rule-0.41 cold-start CFLz class.
3. Trusted DP components become DP islands with copy-in/copy-out shims designed so the DP build's casts are exact no-ops (bit-identical default) — maps onto the house FORCE_SERIAL byte-proof discipline.
4. Keep files FP64 on disk; convert at the I/O boundary; cast fill values to WP before comparison; accumulate long-period means in FP64.
5. **They put ALL global reductions in working precision — do NOT copy.** FP32 sums over 1e7–1e9 nodes perturb CG α/β and destroy conservation measurement. `Kokkos::parallel_reduce` with a double reducer over float views is free. If PR 940 later shows decadal drift, look here first.
6. Expected CPU payoff component-resolved: memory-bound ~1.5–1.6×, whole model 1.2× (cache-resident small mesh) to 1.65× (CORE2 full node); ice neutral. GPU: the same logic predicts wins in the BW-bound CG solver + halved halo/allreduce bytes (16N comm-bound regime, per-partner toll).
7. Precision must be a build dimension that self-certifies: banner printed by the binary + asserted by every test (L80 dead-knob rule, in-band).
8. Watch aliasing at precision boundaries (same array as in + inout ⇒ one seeded buffer).
9. Most of a WP-disciplined codebase flips for free — budget effort by category, not lines: comms datatypes (mechanical), I/O boundary (careful), DP-library shims (bulk), one numerical fix (epsln), zero changes in ice/advection/EOS cores.
10. Their validation bar is the floor, not the ceiling: 1-month SST max|Δ| 0.24 °C accepted qualitatively. The Kokkos ladder (per-scheme floors, M5.23 bar, 63-yr drift/OHC co-tracking, conservation series) is precisely what PR-940 lacks. Their CG stopping criterion computes in FP32 ⇒ iteration counts are the sensitive canary — CGPOLY/CGPIPE instrumentation already tracks that per step.
11. Partial mixed-precision deployment is viable (their native-KPP SP benchmark vs CVMix DP island): run the tolerant 90 % in FP32 with DP islands for the sensitive piece.
