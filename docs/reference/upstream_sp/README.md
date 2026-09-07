# Upstream single-precision material (FESOM/fesom2), archived 2026-09-02

Raw diffs fetched from GitHub on 2026-09-02, merged state:

| file | PR | merged | what |
|---|---|---|---|
| `pr940.diff` | FESOM/fesom2#940 | 2026-08-28 (40 commits) | single precision: `WP=real32|real64`, `MPI_WP`, forcing time axis real64 + point-slope interpolation, stiffness shadow `values_full`, CVMix fixed real64 with shims, `WP_full` global integrals, KPP `epsln` 1e-20 in SP, precision banner, I/O precision report |
| `pr984.diff` | FESOM/fesom2#984 | 2026-08-19 | `precond_variant` (variant 1 = `-a_ri/(a_rr a_ii)`, symmetric); shipped `namelist.dyn` = 1 |
| `pr986.diff` | FESOM/fesom2#986 | (folded into #940) | `use_salt_anomaly`, `S_ref_anomaly` = 35 or 0; consumer offsets, restart detection (global max S > 20), output stream offsets, refuse-to-start guard |
| `pr995.diff` | FESOM/fesom2#995 | 2026-08 | late #940 fix |
| `pr997.diff` | FESOM/fesom2#997 | 2026-08 | late #940 fix |
| `pr940_mech.txt` | — | — | mechanism extract of #940 (1362 lines): every WP/MPI_WP/real64 site with its Fortran location, used to build the M16 conformance table |

## Element → Fortran location

| upstream element | Fortran location | note |
|---|---|---|
| `WP = real32|real64`, `MPI_WP` | `oce_modules.F90` o_PARAM, `MOD_PARTIT.F90` | CMake `USE_SINGLE_PRECISION` (default OFF); `-r4` flips every unkinded `real` |
| forcing time axis real64: `nc_time`, `rdate`, `delta_t`, `time_t0`, `binarysearch_r8` | `gen_surface_forcing.F90` | point-slope `atm = coef_b + (rdate − time_t0)·coef_a`, unconditional upstream (changes FP64 rounding too) → SP-only in the port (plan D5) |
| stiffness shadow `values_full` (real64), SP-only `#ifdef` | `MOD_MESH.F90`, `oce_ale.F90` update_stiff_mat_ale | seeded on first update, refreshed into `%values` per step; `DIAG_STIFF_DRIFT` |
| CVMix fixed real64, WP↔`cvmix_r8` shims | `gen_modules_cvmix_*.F90` | exact no-op in DP |
| `WP_full` (real64) global integrals | `oce_modules.F90`, integrate_nod | everything else is `MPI_WP` |
| KPP guard `epsln = 1e-20` in SP | `oce_ale_mixing_kpp.F90` | 1e-40 flushes to 0 under FTZ |
| precision banner + I/O precision report | `fesom_module.F90`, `io_meandata.F90` | |
| 8-byte streams accumulate real64, 4-byte real32 | `io_meandata.F90` | |
| `use_salt_anomaly`, `S_ref_anomaly` | `oce_modules.F90`, `oce_setup_step.F90`, EOS/coupling/tracer/KPP/pressure_bv/io/restart sites | see `pr986.diff` |
| `precond_variant` (code default −1 = auto: 1 in SP, 0 in DP; namelist ships 1) | `MOD_DYN.F90`, `oce_setup_step.F90`, `oce_ale_ssh` | port: `FESOM_SSH_PRECOND`, default 1, no auto mode (plan D2) |

July's (M8) snapshot of the then-unmerged #940 lives at `~/port_kokkos_mp/docs/reference/pr940/`.
The two SP defects M8 found in July (JRA time axis, stiffness drift) were fixed upstream independently.
