# Next session — M5.22 / Lever 2b: kill the redundant SSS-restoring host path (the last big useless PCIe round-trip)

*Paste this whole file to start. Self-contained. Written 2026-05-30 at the close of M5.21 (Lever 1 coalescing + Lever 2a ghats). The target — the `tracers.S` 249 MB/step D2H — was scoped this session after the user asked "didn't we already do the salinity floor on device?" and the answer turned out to unravel the whole prompt-§2 PCIe plan.*

---

## 0. TL;DR — where we are

The whole FESOM2 **C→C++/Kokkos** port (ocean + sea-ice) is device-resident + validated. **`master` is at tag `m5.20-pcie-residency` (`4b4da79`; NO git remote → local-only).** M5.21 landed two BIT-IDENTICAL wins **in the working tree, UNCOMMITTED** (ask the user before committing; they are file-disjoint → separate commits):
- **Lever 1** (`fesom_tracer_adv.cpp` + `fesom_momentum.cpp`): the coalescing flat lever finished on bucket-A — 8 FCT sub-kernels + `update_vel` → one-thread-per-(node/elem,LEVEL). **−8.07 %**, climate corr=1.00000 vs m520 → SEALED. (§M5.21, L66.)
- **Lever 2a** (`fesom_kpp.cpp`): removed the `kpp.ghats` placebo `sync_host`+halo (256 MB/step D2H, consumed by nothing since `use_kpp_nonlclflx=false`). **−3.08 %**, bit-identical. (§M5.21 Lever-2, L67.1.)

**Combined m520→m521b ≈ −11.1 %; NG5 dist_16 ≈ 1.45 s/step (~2.7× a CPU node).** Binaries: `build-cuda/fesom_port` == `fesom_port_m521b` (Lever 1+2a); `fesom_port_m521` = Lever-1-only; `fesom_port_m520` = pre-M5.21. The post-m521 per-field PCIe (synclog rebuilt this session → `build-cuda-synclog/`, also `fesom_port_synclog`): D2H led by **`kpp.ghats` 256 (now removed in 2a)** + **`tracers.S` 249** (THIS session's target). Full state: [[project-m521-coalescing-finish]], `docs/GPU_FIDELITY.md` §M5.21, `docs/KOKKOS_PORTING_LESSONS.md` L66–L67.

---

## 1. THE FINDING — the `S` 249 MB/step D2H is REDUNDANT host work; SSS restoring + runoff are ALREADY on device

This was traced from current code this session (the prompt's old framing — "port the salinity floor / the SSS-runoff flux to device" — is **wrong**: the floor went device at M5.14, and the restoring is device too). The facts:

1. **`use_sr` is ON in EVERY JRA55 run** (gate, climate, NG5): `fesom_main.cpp:836-846` sets `use_sr=1` whenever `jra55_year>0`, with the files **hardcoded** — `sss_path=/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/PHC2_salx.nc` (monthly SSS climatology) + `runoff_path=…/CORE2_runoff.nc` (constant, read ONCE at init → folded into the ice freshwater flux). Restoring is *meaningfully* active: `surf_relax_S=1.929e-6` (10-day), `use_virt_salt=1`, `ref_sss_local=1` (`fesom_sss_runoff.cpp:305-310`). The climate run.log prints `SSS restoring: …PHC2_salx.nc`. **(The stale comment at `fesom_main.cpp:1067-1071` claiming "OFF in gate+climate" is WRONG — ignore it; fix it.)**

2. **The SSS restoring + virtual-salt + the global conservation integrals are ALREADY ON DEVICE.** `fesom_ice_oce_fluxes_kk` (the ice step's oce_fluxes, called at `fesom_ice.cpp:616`, gated on `!s_no_ice_thermo && forcing && jra && sr`) reads **S on device** (`trS = …values_fld.d()`, surface level), computes `virtual_salt = rsss·water_flux` (rsss = device S surface, `uvs`-gated) + `relax_salt = surf_relax_S·(Ssurf − S)` on device, **each with a device-side global integral** `integrate_nod_2D_kk` (`fesom_ice_coupling.cpp:341-400`), and **OVERWRITES** heat/water/virtual/relax_salt (`fesom_ice.cpp` comment 1104/607). Runoff is consumed on device in the ice thermo (`fesom_ice_thermo.cpp:884`, `runoff_fld.d()`).

3. **The host `fesom_sss_runoff_step_cal` (`fesom_main.cpp:1073`, runs BEFORE the ice step) is REDUNDANT.** It recomputes the same `virtual_salt`/`relax_salt`/`water_flux` on the HOST (`fesom_sss_runoff.cpp:341-441`), which forces the **S sync_host at `fesom_main.cpp:1072` = the 249 MB/step D2H** — only to have the device ice `oce_fluxes` overwrite all of them one substep later. **Verified this session:** the ice thermo (which runs between `sss_runoff` and `oce_fluxes`) does NOT read `water_flux`/`virtual_salt`/`relax_salt` (grep empty) → the host fluxes are consumed by nothing. Its ONLY non-redundant job is the **monthly `Ssurf` climatology NetCDF read** (`fesom_sss_runoff.cpp:360-365`), which the device `relax_salt` needs (Ssurf is pushed to device at `fesom_ice.cpp:615`).

**So this is a redundancy TRIM, not a device port** — the kernels already exist. Removing the host per-step flux computation (keeping the monthly `Ssurf` read) eliminates the S D2H and is **bit-identical** (the removed values were overwritten + unread). The win is **in the benchmark** (NG5 −249 MB/step; CORE2 proportionally smaller ~8–9 MB/step). History: the host `sss_runoff_step` is the original C-port CPU path; M4.3 ported the ice step (incl. its device `oce_fluxes` with the SSS restoring via `sr`) → the host path became dead weight but was never removed.

---

## 2. THE PLAN

1. **Split `fesom_sss_runoff_step`** (`fesom_sss_runoff.cpp:341`) into:
   - (a) `fesom_sss_runoff_read_clim` — the monthly `Ssurf` read (`:360-365`), gated on `update_monthly_flag`. **KEEP.** (Optionally move it next to the `Ssurf` device push at `fesom_ice.cpp:615` to co-locate read+push and make the push monthly too — minor.)
   - (b) the per-step virtual_salt/relax_salt/water_flux computation (`:382-441`). **REMOVE** (the device ice `oce_fluxes` is authoritative).
2. **`fesom_main.cpp:1066-1075`:** replace the `if (use_sr){ S.sync_host(); fesom_sss_runoff_step_cal(...); }` block with just the monthly `Ssurf` read (no `S.sync_host()`). The S D2H disappears.
3. **Fix the stale comment** at `fesom_main.cpp:1067-1071`.
4. ⚠️ **Guard for the future:** the per-step host flux path is only ever needed if the device ice `oce_fluxes` does NOT run (e.g. an `FESOM_NO_ICE_THERMO`/`s_no_ice_thermo=1` debug run, or a future ice-off ocean-only mode). Either keep (b) behind `if (s_no_ice_thermo)` as a fallback, OR delete it and note that `FESOM_NO_ICE_THERMO` now needs the device oce_fluxes path. Decide based on whether `FESOM_NO_ICE_THERMO` is a supported mode (it is currently a bisect toggle, `fesom_ice.cpp:332`).

---

## 3. CORRECTNESS CHECKS — verify BEFORE trusting the trim (the redundancy is the whole argument)

- ✅ **Already verified this session:** the ice thermo does not read `water_flux`/`virtual_salt`/`relax_salt`; the device `oce_fluxes` overwrites all four; nothing between `sss_runoff` (1073) and the ice step (1108) reads them except the `FESOM_NO_HFLUX` bisect memset.
- ⚠️ **`water_flux` net-mass conservation:** the host does `water_flux += net` (`fesom_sss_runoff.cpp:419-440`); the device `oce_fluxes` sets `wf = -flx_fw` (`fesom_ice_coupling.cpp:364`) **without** re-applying that correction. Confirm the device path is the intended behavior (flx_fw is built from rain/runoff/snow/evap in the ice thermo, NOT from `water_flux` — so the host's `+= net` is overwritten regardless). The current ice-on runs ALREADY use the device's `wf=-flx_fw`, so removing the host conservation does NOT change today's behavior — but record whether the conservation is *intended* to be present (a possible separate physics question, NOT a regression of this change).
- ⚠️ **`Ssurf` lifetime:** make sure `Ssurf` is still read monthly + pushed to device after the trim (the device `relax_salt` reads it). The crash signature if you break it = `relax_salt` wrong → SSS drifts / a salinity blow-up over a month.

---

## 4. VALIDATION LADDER

⚠️ **pi is USELESS here** — pi uses analytical forcing (`jra55_year≤0` → `use_sr=0`), so it never runs `sss_runoff`. Validate on CORE2 (JRA55, `use_sr=1`).

1. **CORE2-Serial BIT-IDENTICAL (the decisive proof).** The host `sss_runoff` is redundant on Serial too (host==device; the device oce_fluxes recomputes the same values). So: `cp -r serref_core2 serref_saved`; rebuild `build-serial` with the trim; gate `--fresh-oracle` rebuilds the CORE2 Serial oracle; `diff_snap.py serref_saved serref_core2` = **ALL FIELDS BIT-IDENTICAL** → the trim changes nothing on Serial. (Same harness as M5.20/L65.6, M5.21/L66.2.)
2. **SYNCCHECK on a JRA55 run** (not pi — pi skips sss_runoff). A short CORE2 dist_1/dist_8 `build-synccheck` run with `…1958` forcing: removing the S sync_host leaves S device-authoritative across the sss_runoff window → SYNCCHECK aborts if any host reader of S remains there. 0 aborts = clean. (Reuses the build-synccheck binary; needs the JRA55 path. May need a small job — there isn't a standing CORE2-synccheck job; clone `job_core2_serial_ref` onto `build-synccheck` with `FESOM_PRINT_EVERY` low.)
3. **CUDA fidelity gate** `scripts/gpu_fidelity_gate.sh` (CORE2 dist_8 ice-active — this DOES run restoring): **PASS at the same floor** as m521b (worst ~1.1e-2), no new divergence (bit-identical ⇒ zero).
4. **The CORE2-MONTH restoring/runoff test (the user's ask — fast regression + physics sanity).** A dedicated short run that EXERCISES + checks the SSS path explicitly (the standard climate covers it, but a month is ~3 min and isolates it):
   - Clone a job → CORE2 dist_8, JRA55 1958, **1460 steps (1 month @ dt=1800)**, `snap_every` ~ every 5–10 days, monthly stream ON. Run BOTH `fesom_port_m521b` (before) and the new binary (after).
   - **Regression:** `diff_snap.py` before-vs-after = bit-identical (Serial) / climate-close (CUDA gate-style).
   - **Physics sanity (does restoring+runoff WORK?):** check `relax_salt`/`virtual_salt` are nonzero + physical (relax_salt sign restores SSS toward PHC2; runoff freshens river-mouth nodes — Amazon/Ob/Lena/Yenisei). Confirm surface S over the month moves toward the PHC2 climatology where the model was off (Baltic/Arctic). A small diagnostic: dump `relax_salt` min/max + the SSS field, compare to PHC2. (The Fortran/C reference is the ground truth — `docs/REFERENCE_RUNS.md`; or just confirm the field is nonzero and the run is stable, since the trim is bit-identical so the physics is unchanged from m521b.)
5. **Same-node A/B (NG5 dist_16)** clone `jobs/job_ng5_m521b_ab` (BEFORE=`fesom_port_m521b`, AFTER=new): the s/step delta + the **`deep_copy` line should drop ~249 MB/step** (the S D2H gone; calls/step −1). By L65 the blocking-fence value may exceed the 249 MB bandwidth (like ghats's −3.08 % from 256 MB).
6. **1-yr CORE2 CUDA climate to close** (`jobs/job_m32_cuda_core2`, `M32_NSTEPS=17280 M32_TAG=_<tag>`): corr=1.00000 vs `m32_cuda_m521b_1yr` (this session's m521b climate, job 25251745 — confirm it finished + use it as the cref). Bit-identical ⇒ corr=1.0 guaranteed.

---

## 5. HARD CONSTRAINTS (carry every session)
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; env.sh's 4.1.2 SEGFAULTs on device ptrs). ⚠️ `env_cuda.sh` PURGES `git` — git ops in a separate shell. CPU builds use `env.sh`. Build dirs `build-cuda`/`build-serial`/`build-synccheck`/`build-omp` carry m521b; `build-cuda-synclog` = the `FESOM_SYNC_LOG=ON` rebuild.
- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB quota; ~14 GB free). Big/NG5/CORE2 runs via SLURM, never login. ⚠️ NG5 perf jobs write ~50 GB `*.monthly.nc` even at `snap_every=-1` — `rm` them (the `job_ng5_*_ab` templates do; `job_ng5_prof` does NOT — clean it manually).
- **Same-day same-node perf baselines only** ([[feedback-perf-same-day-baseline]]); s/step varies ~7–10 %/day by node mix.
- **Device/forcing changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); **pi is insufficient here (and skips sss_runoff entirely).** **Commit/push only when the user asks.** KPP is the default mix_scheme.
- ⚠️ The coupled-vs-ocean-only distinction (user, 2026-05-30): SSS restoring is used in ocean-only (not coupled); runoff is used always. In THIS port, both are tied to `use_sr` (= `jra55_year>0`), and there is no coupled/atmosphere-model mode yet — so all current runs are JRA55-forced with restoring+runoff ON. A future coupled mode (use_jra=0) would want runoff WITHOUT restoring — note that `runoff` being gated under `use_sr` is a structural limitation to revisit then.

## 6. BINARIES, TAGS, STATE
- `build-cuda/fesom_port` == `fesom_port_m521b` (Lever 1+2a, validated). `fesom_port_m521` = Lever-1-only. `fesom_port_m520` = pre-M5.21. `serref_core2` = the m521(b) CORE2 Serial oracle (bit-identical m521≡m521b≡m520 on Serial). `m32_cuda_m521b_1yr` = the m521b 1-yr climate (the cref for closing M5.22).
- `master` @ `m5.20-pcie-residency` (`4b4da79`). M5.21 (Lever 1 + 2a) is UNCOMMITTED in the working tree (`fesom_tracer_adv.cpp`, `fesom_momentum.cpp`, `fesom_kpp.cpp` + docs). **Ask the user about committing M5.21 before/at the start of M5.22.**

## 7. POINTERS
- **Memory:** [[project-m521-coalescing-finish]] (Lever 1+2a + this finding), [[project-m520-pcie-residency]] (L65 deep_copy≠wall-clock), [[feedback-gpu-fidelity-gate]], [[feedback-perf-same-day-baseline]], [[reference-cuda-aware-mpi]], [[reference-build-run]].
- **Docs:** `docs/GPU_FIDELITY.md` §M5.21 "Lever-2 PCIe" (the full ghats + SSS finding); `docs/KOKKOS_PORTING_LESSONS.md` **L67** (the "a doc's host-stay is a hypothesis — re-derive from current code" lesson + the redundant-pre-compute trap).
- **Code map:** `fesom_main.cpp:836-846` (use_sr init + hardcoded files), `:1066-1075` (the host call to remove + the stale comment), `fesom_sss_runoff.cpp:341-441` (the host step — split it), `:289-334` (init defaults + runoff one-shot read), `fesom_ice.cpp:559-617` (the ice thermo + the device oce_fluxes caller + the Ssurf push at 615), `fesom_ice_coupling.cpp:341-400` (the DEVICE oce_fluxes = the authoritative virtual_salt/relax_salt + `integrate_nod_2D_kk`), `fesom_ice_thermo.cpp:884` (runoff on device).
- **Tools:** `jobs/job_ng5_m521b_ab` (clone for the A/B), `jobs/job_core2_serial_ref` + `job_gpu_fidelity_dev` (the gate legs), `scripts/gpu_fidelity_gate.sh --fresh-oracle`, `scripts/diff_snap.py`, `scripts/m32_climate_compare.py`, `jobs/job_m32_cuda_core2`. For the CORE2-month test: clone `job_m32_cuda_core2` with `M32_NSTEPS=1460 M32_TAG=_sssmonth`.

## 8. Bottom line
`tracers.S` 249 MB/step D2H is the #2 PCIe driver and it is **pure redundancy** — the SSS restoring/virtual-salt/runoff are already device-resident (`fesom_ice_oce_fluxes_kk`), and the host `fesom_sss_runoff_step_cal` recomputes them only to be overwritten. **Trim the host per-step flux (keep the monthly `Ssurf` read), drop the S sync_host → bit-identical, −249 MB/step, a real NG5/campaign win.** Restoring+runoff ARE active in the standard CORE2 gate+climate (so the normal ladder validates it); the CORE2-month run is a fast regression + a physics sanity check that restoring/runoff are functioning. Verify the redundancy (done this session: ice thermo doesn't read the fluxes), hold the CORE2-Serial-bit-identical gate + the gate + the A/B + the 1-yr climate. Re-confirm m521 (Lever 1+2a) is committed/handled first. This closes the last big per-step host↔device round-trip in the JRA55-forced step.
