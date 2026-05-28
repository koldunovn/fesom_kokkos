# M3.2 climate-bias paradox — root cause: stale references

The handout (`docs/NEXT_SESSION_BIAS_INVESTIGATION.md`) flagged that the M3.2
CUDA 1-yr SST is closer to Fortran (−0.028 °C) than to the C-port reference
(−0.236 °C), contradicting the implicit assumption in `scripts/m32_climate_compare.py`
that the C-port reference is bit-identical to Serial Kokkos. **The cause is a
mismatched reference choice, not a port bias.** Both handout references were run
with a different mix scheme than the CUDA M3.2 run, and the C-port reference is
additionally stale on the `ice_gamma_fct` fix.

## Physics-config comparison

| key                | CUDA M3.2 `m32_cuda_1yr_pin` | C-port `eps_2yr_dt1800` (handout ref) | C-port `kpp_2yr_rebase` (correct ref) | Fortran `fortran_pp_2yr` (handout ref) | Fortran `fortran_kpp_5yr_fix` (correct ref) |
|--------------------|:---------------------------:|:-------------------------------------:|:-------------------------------------:|:--------------------------------------:|:-------------------------------------------:|
| build / SHA        | kokkos `05182aa` (May 28)   | C-port `8dac9975` (May 23)            | C-port `6ecabe8` (May 25, KPP K11)    | Fortran build                          | Fortran build                               |
| **mix_scheme**     | **KPP** (default)           | **PP** (default @ `8dac9975`, pre-`8d0cdbc`) | **KPP** (env opt-in)            | **PP** (namelist)                      | **KPP** (namelist)                          |
| **ice_gamma_fct**  | **0.5**                     | **0.25** (pre-`7c6663b`)              | **0.5** (post-fix)                    | 0.5 (namelist)                         | 0.5 (namelist)                              |
| dt / step_per_day  | 1800 / 48                   | 1800 / 48                             | 1800 / 48                             | 1800 / 48                              | 1800 / 48                                   |
| run length         | 17280 (1 yr)                | 35100 (2 yr + spill)                  | 35200 (2 yr + spill)                  | 2 yr                                   | 5 yr                                        |
| MPI ranks          | 8 (CUDA dist_8)             | 864                                   | 864                                   | 864                                    | 864                                         |
| C_d                | 0.0025                      | 0.0025                                | 0.0025                                | 0.0025                                 | 0.0025                                      |
| A_ver              | 1.0e-4                      | 1.0e-4                                | 1.0e-4                                | 1.0e-4                                 | 1.0e-4                                      |
| K_GM_max           | 1000                        | 1000                                  | 1000                                  | 1000                                   | 1000                                        |
| Redi_Kmax          | sync to K_GM_max            | sync                                  | sync                                  | sync (= 1000)                          | sync (= 1000)                               |
| opt_visc           | 7                           | 7                                     | 7                                     | 7                                      | 7                                           |
| momadv_opt         | 2                           | 2                                     | 2                                     | 2                                      | 2                                           |
| visc_sh_limit      | 5e-3                        | 5e-3                                  | 5e-3                                  | 5e-3                                   | 5e-3                                        |
| Ricr / concv       | 0.3 / 1.6                   | 0.3 / 1.6                             | 0.3 / 1.6                             | 0.3 / 1.6                              | 0.3 / 1.6                                   |
| use_partial_cell   | false                       | false                                 | false                                 | false                                  | false                                       |
| use_floatice       | false                       | false                                 | false                                 | false                                  | false                                       |
| use_global_tides   | false                       | false                                 | false                                 | false                                  | false                                       |
| linfs vs zstar     | linfs                       | linfs                                 | linfs                                 | linfs (`work_linfs_pp_2yr`)            | linfs                                       |
| Pstar              | 30000                       | 30000                                 | 30000                                 | 30000                                  | 30000                                       |
| c_pressure         | 20                          | 20                                    | 20                                    | 20                                     | 20                                          |
| i_vert_diff (impl) | true                        | true                                  | true                                  | true                                   | true                                        |

## Verdict (2 sentences)

The handout's "paradox" is entirely explained by **reference mismatch, not a GPU
port bias.** The C-port reference `eps_2yr_dt1800` differs from the CUDA M3.2 run
in TWO physics knobs (PP vs KPP **and** ice_gamma_fct=0.25 vs 0.5); the Fortran
reference `fortran_pp_2yr` differs in ONE (PP vs KPP). The correct apples-to-apples
references for CUDA-M3.2 (which runs KPP + gamma=0.5) are
**`/work/ab0995/a270088/port/kpp_2yr_rebase`** (C-port KPP @ `6ecabe8`, post-fix)
and **`/scratch/a/a270088/fortran_kpp_5yr_fix`** (Fortran KPP, namelist gamma=0.5).
Both already exist on disk — no rerun needed.

## Evidence trail

1. C-port reference `eps_2yr_dt1800` SHA: `8dac9975` (`fix(forcing): rotate JRA winds geo->rotated frame`, May 23).
2. C-port default `mix_scheme` was flipped from PP to KPP in commit `8d0cdbc` (May 25, after the reference run).
   - diff: `static int s_use_kpp = 0;` → `1`; `(e[0]=='K')` → `!(e[0]=='P')`.
3. C-port `ice_gamma_fct` was fixed from 0.25 → 0.5 in commit `7c6663b` (May 25, also after the reference run).
   - The handout already references this fact via `feedback_bvfreq_smoothing_gap` / `feedback_namelist_over_codedefault`.
4. `eps_2yr_dt1800/job_eps_2yr_dt1800` does NOT export `FESOM_MIX_SCHEME` → ran with default = PP at that commit.
5. Kokkos port `src/fesom_step.cpp:113` has `static int s_use_kpp = 1;` → default KPP.
6. Kokkos port `src/fesom_ice.cpp:86` has `ice->ice_gamma_fct = 0.5` (post-fix).
7. `jobs/job_m32_cuda_core2` does NOT export `FESOM_MIX_SCHEME` → CUDA M3.2 ran with default = KPP.
8. Fortran ref `/scratch/a/a270088/fortran_pp_2yr/` is the output of `/home/a/a270088/port2/fesom2/work_linfs_pp_2yr/`
   whose `namelist.oce` has `mix_scheme = 'PP'`.
9. Fortran KPP ref `/scratch/a/a270088/fortran_kpp_5yr_fix/` is the output of `work_kpp_5yr_d1800/` with `mix_scheme = 'KPP'`.
10. C-port KPP refs:
    - `kpp_2yr_dt1800` SHA `fcfca25` (KPP K8, gamma=0.25 pre-fix) — older
    - `kpp_5yr_dt1800` SHA `4038c2b` (K10 5-yr stability, gamma=0.25 pre-fix) — used for the C+KPP-vs-F+KPP figures (`4d2718a`)
    - **`kpp_2yr_rebase` SHA `6ecabe8` (post-K11, gamma=0.5 post-fix)** — the correct one

## Re-comparison result (DONE, 2026-05-28)

`scripts/m32_climate_compare.py` updated to default to the canonical KPP refs,
re-run against the M3.2 CUDA 1-yr (1958) output. Numbers below are the
annual-mean surface statistics.

### CUDA-KPP vs Fortran-KPP (`fortran_kpp_5yr_fix`)

| field | corr     | bias          | RMS        | |d|max |
|-------|----------|---------------|------------|--------|
| sst   | 1.00000  | +3.1472e-05   | 1.4541e-02 | 0.313  |
| sss   | 0.99996  | −5.3230e-04   | 2.6202e-02 | 1.976  |
| ssh   | 1.00000  | +2.0300e-05   | 1.0136e-03 | 0.024  |
| a_ice | 0.90662  | −1.0303e-01   | 1.7039e-01 | 0.656  |
| m_ice | 0.98244  | −1.0228e-01   | 1.6718e-01 | 0.834  |
| uice  | 0.85019  | −1.0131e-03   | 2.7949e-02 | 0.228  |

### CUDA-KPP vs C-port-KPP (`kpp_2yr_rebase`)

| field | corr     | bias          | RMS        | |d|max |
|-------|----------|---------------|------------|--------|
| sst   | 1.00000  | +1.0415e-04   | 1.4065e-02 | 0.314  |
| sss   | 0.99996  | −1.7994e-04   | 2.6125e-02 | 1.970  |
| ssh   | 1.00000  | −1.4326e-05   | 9.0499e-04 | 0.024  |
| a_ice | 0.99997  | +1.6348e-04   | 2.8583e-03 | 0.178  |
| m_ice | 0.99998  | −1.5053e-04   | 3.5701e-03 | 0.148  |
| uice  | 0.99978  | −7.4589e-05   | 5.3412e-04 | 0.011  |

### Comparison with the handout's (deprecated) numbers

| pair                                 | sst bias    | sst RMS    | source                          |
|--------------------------------------|-------------|------------|---------------------------------|
| CUDA  −  Fortran (handout, PP ref)   | −0.028 °C   | 0.109      | wrong scheme (CUDA=KPP, ref=PP) |
| **CUDA  −  Fortran-KPP (canonical)** | **+3e-5 °C** | **0.015**  | same scheme, just GPU drift     |
| CUDA  −  C-port (handout, PP+γ=0.25) | −0.236 °C   | 0.370      | 2 stale knobs                    |
| **CUDA  −  C-port-KPP (canonical)**  | **+1e-4 °C** | **0.014**  | pure scatter-drift              |

Bias dropped by **3 orders of magnitude** on SST. Ice fields show the expected
pattern: CUDA-vs-C ≈ scatter floor (~1e-4), CUDA-vs-F retains the genuine
Fortran↔C-port physics gap (~0.1 a_ice — already validated as the C-port's own
KPP-vs-F budget per commit `375f3eb` and `4d2718a`).

**Verdict: M3.2 PASSES.** No port bug; the original paradox was entirely a
reference-mismatch artifact.

## Going forward

- Reference catalog: `docs/REFERENCE_RUNS.md` (lives next to this doc).
- Default policy: KPP everywhere unless the user explicitly opts to PP (see
  `feedback-kpp-default` memory).
- `scripts/m32_climate_compare.py` defaults to the canonical KPP refs; override
  via `--cref`/`--fref` for PP runs.
