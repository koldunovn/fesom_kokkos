# Next-session prompt — investigate why CUDA is closer to Fortran than to the "C-port reference"

Paste the block below as the next session's opening prompt.

---

> Continue the FESOM2 Kokkos port. **The M3.2 1-yr CUDA climate validation revealed a paradox**
> we want to resolve this session — and the result hints that one of the two reference runs is
> a stale baseline, not that the GPU port has a bias.
>
> **The paradox.** On a 1-yr CORE2 (dt=1800) run, the CUDA Kokkos backend (`m32_cuda_1yr_pin`,
> binary `05182aa` = M5.9-pin) gives annual-mean global SST=+11.2228 °C, vs Fortran 11.2505,
> vs C-port reference `eps_2yr_dt1800` 11.4586. So:
>
> | Pair                       | bias (°C) | RMS    | |max|  |
> |:---------------------------|:---------:|:------:|:------:|
> | CUDA  −  Fortran           | −0.028    | 0.109  | 1.114  |
> | CUDA  −  C-port            | **−0.236**| **0.370** | 2.866 |
> | **C-port  −  Fortran (BUDGET)** | **+0.208** | **0.301** | 2.121 |
>
> The script `m32_climate_compare.py` was written assuming **C-port == Serial Kokkos == bit-identical**
> (so backend-vs-C measures only GPU scatter/reduce drift, and is expected ≪ C-vs-Fortran). But
> CUDA matches Fortran *better* than C-port matches Fortran — i.e. the C-port reference run isn't
> being reproduced by anyone. SSS shows the same shape (CUDA−F: +0.032, CUDA−C: +0.098,
> C−F: −0.065). SSH is fine (≪ noise floor on both sides) — so this is a thermohaline/mixing
> divergence, not a momentum or solver one. **Bias maps**:
> `/home/a/a270088/port_kokkos/docs/m32_bias_maps/bias_{sst,sss,ssh,a_ice,m_ice,uice}_1958.png`
> — open them to see *where* the C-port warm bias concentrates (look for tropics vs polar pattern).
> Full table: `docs/SCALING_CORE2.md` + the climate-compare numbers in this prompt.
>
> **The leading hypothesis: mix-scheme mismatch.** The CUDA build defaults to **KPP** (per the C
> port's recent default flip — `[[reference_c_port_kpp_default]]`-equivalent). The Fortran
> reference dir is literally named `fortran_pp_2yr` — explicitly **PP**. The C-port reference
> `eps_2yr_dt1800` was made 2026-05-23 and its mix scheme isn't recorded; it may be PP from
> before the KPP default flip. If CUDA=KPP, F=PP, C=PP, then CUDA−F ≈ (KPP_skill_difference)
> and CUDA−C ≈ (KPP_skill_difference) + (C-port-physics-drift). That matches the shape exactly,
> *but doesn't fully explain* why CUDA-KPP would match F-PP at ~0.03 — they should differ more.
> Alternate / additional hypotheses below.
>
> **Diagnostic plan (in this order).**
>
> 1. **Pin the actual physics in each reference.** Find each run's namelist (or its inferable
>    source). For C-port: `ls -la /work/ab0995/a270088/port/eps_2yr_dt1800/` — look for a `.log`
>    or any captured config; if missing, ask the user, or check
>    `/home/a/a270088/port2/fesom2_port/docs/` and the `feedback_namelist_over_codedefault`
>    auto-memory of the C-port project. For Fortran: check `fortran_pp_2yr/`. For CUDA:
>    `jobs/job_m32_cuda_core2` + the runtime print in `m32_cuda_1yr_pin/run.log` ("mix_scheme=...").
>    Look for: `mix_scheme`, `use_kpp`, `ice_gamma_fct`, `Pstar`, `c_pressure`, `opt_visc`,
>    `K_GM`, `K_hor`, `K_ver`, dt (= step_per_day), `use_FCT_3rd`, `linfs` vs `zstar`, etc.
>    **Write the answers into a table at the top of `docs/m32_bias_investigation.md`.**
>
> 2. **Test the mix-scheme hypothesis.** If the answers from (1) show CUDA=KPP and refs=PP:
>    submit a **1-yr CUDA run with `FESOM_MIX_SCHEME=PP`** (same job `jobs/job_m32_cuda_core2`
>    with `--export=ALL,M32_NSTEPS=17280,M32_TAG=_1yr_pp,FESOM_MIX_SCHEME=PP`). If it lands
>    on the Fortran or C-port baseline, the paradox is solved — mix scheme alone.
>
> 3. **Where is the bias spatially?** Open the 6 bias PNGs. If C-port's warm bias is in the
>    tropics → suspect insolation/penetration depth (`use_shortwave_pene`, chlorophyll source).
>    If polar → ice coupling differences. If basin-uniform → IC drift. Tag the maps in the doc.
>
> 4. **Regenerate the C-port reference at the same config as CUDA.** Best alignment: take the
>    current `/home/a/a270088/port2/fesom2_port` HEAD (it has the KPP default + the bvfreq fix
>    per memory's `feedback_bvfreq_smoothing_gap`), run 1-yr CORE2 dt=1800 with JRA55 1958 +
>    PHC IC — exact same setup as the CUDA M3.2 job — and store at
>    `/work/ab0995/a270088/port/eps_kpp_1yr_dt1800_kokkosalign/`. Then re-run
>    `m32_climate_compare.py` against THIS dir. Expect: CUDA−C ≈ 1e-3 (the scatter floor) instead
>    of 0.2. **If that happens, the paradox is just a stale reference, not a port bug.**
>
> 5. **Confirm Serial-Kokkos bit-identity to the new C-port reference.** Run
>    `jobs/job_core2_serial_ref` on the M5.9-pin binary at 1 yr; diff every monthly mean field
>    against the fresh C-port from (4) with `scripts/diff_snap.py` (zero-tolerance, all fields
>    not a subset — L48). They must be byte-identical. If not, the Serial port itself has drifted
>    — a much bigger problem to find.
>
> 6. **Run the C-vs-Fortran 2-yr baseline** with `scripts/eps_climate_compare_2yr.py` to see
>    what the genuine "C-port physics drift vs Fortran" budget looks like at HEAD. Compare to
>    the historical (sub-0.02 RMS) numbers in the C-port memory's `project_sea_ice_port_state`
>    / KPP-validation entries.
>
> 7. **If the mix-scheme hypothesis is REJECTED** (CUDA-PP run still bias~0.2 vs Fortran):
>    pivot to the *atomic-scatter-compounding-over-1-yr* hypothesis. Force-disable the device
>    scatters (`FESOM_HOST_HALO=1` makes halos identical, and there's already host-fallback for
>    several scatters — see `docs/SCATTER_STRATEGY.md`). A run with host-fallback scatters
>    should match Serial-Kokkos (= the new C-port reference) bit-identically by construction.
>    Difference between that and the device-scatter run is the pure GPU climate cost.
>
> **Success criteria.**
> - We can explain ≥90 % of the CUDA-vs-C-port SST bias from a single named cause (mix scheme,
>   stale reference, or compounded atomic scatters).
> - A re-baselined C-port reference brings CUDA−C ≪ C−F (the original `m32_climate_compare.py`
>   assumption).
> - The bias maps show no localised pathology (no single basin/grid-cell with a wild outlier).
>
> **What I want back from the next session, in `docs/m32_bias_investigation.md`:**
> 1. The actual physics config of each reference (the table from step 1).
> 2. Which hypothesis the data supports (verdict in 1-2 sentences).
> 3. The new CUDA−C bias from the re-baselined reference (step 4).
> 4. Updated `m32_climate_compare.py` (or replacement) pointing to the new reference, plus a
>    `docs/REFERENCE_RUNS.md` recording reference provenance going forward so this doesn't
>    bite again.
> 5. A one-paragraph entry in `docs/KOKKOS_PORTING_LESSONS.md` (probably L51).
>
> **Things you can skip / don't redo.** The M3.2 1-yr CUDA run itself is healthy — don't re-run.
> The bit-identity oracle for Serial Kokkos is still bit-identical to the C port at the M4
> acceptance commit; don't doubt the port. The farc + CORE2 scaling tables (`docs/SCALING_*.md`)
> are good. The 12 jobs/job_m32_* and jobs/job_scaling_* scripts are correct as committed.
>
> Build: `source ./env_cuda.sh && cmake --build build-cuda --target fesom_port -j 16`.
> Serial: `source ./env.sh && cmake --build build-serial --target fesom_port -j 16`.
> Python: `PYTHONPATH=/home/a/a270088/PYTHON /work/ab0995/a270088/mambaforge/envs/nereus/bin/python`.

---

## Quick reference — paths + commands

- CUDA M3.2 1-yr run: `/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_1yr_pin/`
- C-port reference (suspect):
  `/work/ab0995/a270088/port/eps_2yr_dt1800/` (created 2026-05-23)
- Fortran reference: `/scratch/a/a270088/fortran_pp_2yr/`
- Bias maps (CUDA−Fortran + CUDA−C side by side): `docs/m32_bias_maps/bias_*.png`
- Compare script: `scripts/m32_climate_compare.py`
- C-vs-F baseline script: `scripts/eps_climate_compare_2yr.py`
- Map-making (style guide): `/home/a/a270088/port2/fesom2_port/scripts/clim_validation_2yr.py`
- Bit-id gate script: `scripts/diff_snap.py` (takes two DIRECTORIES, not files — lesson L19)
- Run-output **MUST** go to `/work/ab0995/a270088/port2/...` not $HOME (60 GB quota — see
  `[[feedback-hpc-run-hygiene]]`).
- Job templates: `jobs/job_m32_cuda_core2`, `jobs/job_core2_serial_ref`.
