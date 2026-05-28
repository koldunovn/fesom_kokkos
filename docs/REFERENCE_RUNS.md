# Reference runs for FESOM2 Kokkos-port climate validation

Catalog of the canonical CORE2 reference runs used to validate Kokkos-backend
climate output (M3.2 and beyond). For each ref: path, physics knobs, build
provenance, and what it's good for.

**Default policy (2026-05-28)** — KPP is the default mix scheme for everything in
this repo going forward (CORE2 production, comparison scripts, ref selection).
Only use PP references when the user explicitly opts in. The Kokkos port HEAD
defaults to KPP (`src/fesom_step.cpp:113`: `s_use_kpp=1`).

---

## Canonical references (DEFAULT for KPP runs)

### Fortran-KPP — `/scratch/a/a270088/fortran_kpp_5yr_fix`
- **Length:** 5 yr (1958–1962), instantaneous (`<var>.fesom.<yr>.nc`)
- **Source workdir:** `/home/a/a270088/port2/fesom2/work_kpp_5yr_d1800/`
- **Physics:** `mix_scheme='KPP'`, `ice_gamma_fct=0.5`, `linfs`, dt=1800, CORE2 namelist
- **Use:** absolute "does the GPU reproduce the science?" budget. Combines
  Fortran↔C divergence + C↔CUDA divergence. CUDA-vs-F = the full skill gap.

### C-port-KPP — `/work/ab0995/a270088/port/kpp_5yr_fix`
- **Length:** 5 yr (1958–1962), monthly means (`<var>.fesom.<yr>.monthly.nc`)
- **Build:** C-port `6ecabe8` ("KPP K11: wrap-up — KPP complete & default"), May 25 2026
- **Job:** `jobs/job_kpp_5yr_fix` (sets `FESOM_MIX_SCHEME=KPP`, dt=1800, 864r); same SHA as `kpp_2yr_rebase` (its 2-yr cousin), but matches the C-port's published `kpp_5yr_fix_figures` validation set.
- **Physics:** KPP, `ice_gamma_fct=0.5` (post-fix `7c6663b`), linfs, dt=1800
- **Use:** **the canonical backend-vs-C isolation reference for KPP Kokkos runs.**
  C twin == Serial Kokkos == bit-identical, so CUDA-vs-this isolates the GPU
  scatter/reduce drift (D22). Validated against Fortran-KPP at C-port commit
  `375f3eb` ("C+KPP reproduces Fortran-KPP").

---

## Secondary references (PP only — use only when explicitly running PP)

### Fortran-PP — `/scratch/a/a270088/fortran_pp_2yr`
- 2 yr (1958–1959), instantaneous. Source: `work_linfs_pp_2yr/` (`mix_scheme='PP'`).
- Only use if the Kokkos backend was run with `FESOM_MIX_SCHEME=PP`.

### C-port-PP (post-fixes) — `/work/ab0995/a270088/port/pp_2yr_rebase`
- 2 yr (1958–1959), monthly. C-port post-bvfreq + post-gamma fixes.
- Job: `jobs/job_pp_2yr_dt1800` (sets `FESOM_MIX_SCHEME=PP`).
- Use only paired with a PP Kokkos run.

---

## Deprecated references (do NOT use for new Kokkos-port comparisons)

These were the original handout choices (`docs/NEXT_SESSION_BIAS_INVESTIGATION.md`).
They predate the KPP default flip and the `ice_gamma_fct=0.25→0.5` fix; using
them against a KPP Kokkos run conflates port drift with **2 physics-config deltas**
and produced the spurious "M3.2 paradox" (CUDA-vs-C ≈ −0.236 °C SST bias).

| Path                                            | Build       | Why deprecated                                                                   |
|-------------------------------------------------|-------------|-----------------------------------------------------------------------------------|
| `/work/ab0995/a270088/port/eps_2yr_dt1800`      | C-port `8dac9975` (May 23) | Default mix at that commit was PP (KPP flip came @ `8d0cdbc`); `ice_gamma_fct=0.25` (pre-fix `7c6663b`). |
| Fortran-PP refs used for KPP backends           | —           | Scheme mismatch on its own ~+0.2 °C SST.                                          |

---

## Comparison verdict (M3.2 CUDA 1-yr 1958)

After re-running `scripts/m32_climate_compare.py` against the canonical KPP refs:

| field | CUDA-vs-Fortran-KPP                          | CUDA-vs-C-port-KPP                          |
|-------|----------------------------------------------|---------------------------------------------|
| sst   | corr 1.0000, bias +3.1e-5 °C, RMS 1.45e-2    | corr 1.0000, bias +1.0e-4 °C, RMS 1.41e-2  |
| sss   | corr 0.99996, bias −5.3e-4, RMS 2.62e-2      | corr 0.99996, bias −1.8e-4, RMS 2.61e-2     |
| ssh   | corr 1.0000, bias +2.0e-5, RMS 1.01e-3       | corr 1.0000, bias −1.4e-5, RMS 9.05e-4      |
| a_ice | corr 0.9066, bias −0.103, RMS 0.170          | corr 0.99997, bias +1.6e-4, RMS 2.86e-3     |
| m_ice | corr 0.98244, bias −0.102, RMS 0.167         | corr 0.99998, bias −1.5e-4, RMS 3.57e-3     |
| uice  | corr 0.8502, bias −1.0e-3, RMS 2.79e-2       | corr 0.99978, bias −7.5e-5, RMS 5.34e-4     |

vs the handout's deprecated-ref numbers (CUDA−C sst bias `−0.236 °C`, RMS 0.370):
**the M3.2 "paradox" was an artifact of comparing CUDA-KPP against C-port-PP**.
CUDA-vs-C-port-KPP shows the expected pure scatter-drift signature: O(1e-4) bias
on ocean fields, O(1e-3) on ice (corr ~0.9999 across the board). The CUDA-vs-F
ice numbers (corr ~0.9, bias ~−0.1) ARE the genuine C-vs-Fortran-at-KPP physics
budget and match the C-port's own KPP-vs-F validation (commit `375f3eb`).

See `docs/m32_bias_investigation.md` for the full physics-config table and
evidence trail.

---

## Quick-pick by use case

- **Validating any Kokkos-port CORE2 KPP run (the default):**
  `--fref /scratch/a/a270088/fortran_kpp_5yr_fix --cref /work/ab0995/a270088/port/kpp_5yr_fix`
- **Validating a Kokkos-port PP run (only if explicit):**
  `--fref /scratch/a/a270088/fortran_pp_2yr --cref /work/ab0995/a270088/port/pp_2yr_rebase`

`scripts/m32_climate_compare.py` defaults to the KPP pair; override via the
`--fref` / `--cref` flags if needed.
