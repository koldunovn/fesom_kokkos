# Provenance of the imported C baseline

This repo (`port_kokkos`) is the **C++/Kokkos port** of the validated C FESOM2 port.
The C sources here were imported verbatim as the starting point and must stay
**bit-identical in behaviour** through milestone M0 (see `docs/plans/20260525-kokkos-port.md`).

## Source

- **From**: `/home/a/a270088/port2/fesom2_port/`
- **C port git HEAD at import**: `75de623` (recorded in `C_PORT_SOURCE_SHA.txt`)
- **Imported on**: 2026-05-25
- **Fortran ground truth**: `/home/a/a270088/port2/fesom2/src`
- **Reference Fortran run**: `/scratch/a/a270088/fortran_pp_2yr`

## What was imported vs skipped

- **Imported**: `src/`, `tests/`, `scripts/`, `CMakeLists.txt`, `env.sh`, `configure.sh`,
  `README.md`, `jobs/` (the `job_*` templates), `docs/*.md`, `docs/plans/`,
  `docs/kpp_reference_namelists/`, `docs/FRESH_START.md` (the domain bootstrap doc).
- **Skipped** (run artifacts, regenerable): `build/`, `runs/`, and the validation/plot output
  dirs under `docs/` (`validation_*`, `compare_plots*`, `drift_*`, `kpp_5yr_figures/`).

## Golden reference (M0.1 — to be captured)

The unmodified C binary's output is the **golden reference** for the M0 bit-identity gate
(`diff_snap.py` must read `0.0` for the C++/Kokkos Serial build). Capture two smoke runs:

1. **Pi mesh, short run** — fast, single-rank sanity.
2. **CORE2, 16-rank, ~50 steps** — exercises MPI + the real config.

Store the resulting `snap_*.nc` under `docs/reference/c_baseline_snapshots/` and record the
exact invocation (mesh, dt, nsteps, snap_every, PHC, jra55_year) here once run on a compute node.

> Note: building/running the C reference needs the Levante build env (`source env.sh`) and,
> for CORE2, a compute-node SLURM job — done as part of completing M0.1/M0.5, not on the login node.
