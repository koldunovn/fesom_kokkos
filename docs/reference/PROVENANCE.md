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

## Golden reference (M0.1)

The unmodified C binary's output is the **golden reference** for the M0 bit-identity gate
(`diff_snap.py` must read `0.0` for the C++/Kokkos Serial build).

### Captured 2026-05-25 — pi mesh, single-rank, analytical forcing

- **Built**: `bash -l configure.sh` (Release) → `build/fesom_port` — clean, exit 0.
- **Invocation** (login node, singleton MPI):
  ```bash
  source ./env.sh
  # login-node MPI: no UCX/IB → use ob1 + shared-mem/self (see below)
  export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
  unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL \
        HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
  ./build/fesom_port /home/a/a270088/port2/fesom2/test/meshes/pi \
      docs/reference/c_baseline_snapshots/pi  100 20 10
  #   mesh                                       dt nsteps snap_every  (no PHC, no JRA → analytical)
  ```
- **Result**: 20 steps, exit 0, physical (T[10,15] S[35], CG 2 iters); `snap_000000/010/020.nc`
  + monthly streams in `docs/reference/c_baseline_snapshots/pi/`.
- **Gate harness validated**: `python3 scripts/diff_snap.py <golden> <golden>` →
  "ALL FIELDS BIT-IDENTICAL", exit 0.

> The `*.nc` are gitignored (regenerable reference data, not source) — this recipe is the
> source of truth. The **login-node MPI override** above is required because UCX/IB transports
> aren't available off the compute nodes.

### Still to capture (when needed, on a compute node)

- **CORE2, 16-rank, ~50 steps** (real MPI + production config) — a SLURM job, not login-node.
  Not required for the M0 Serial bit-identity gate (pi suffices); capture before the M3 climate work.
