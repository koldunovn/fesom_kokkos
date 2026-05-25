# dt=1800 blow-up — handoff for the next session

> ## ⚡ SESSION UPDATE 2026-05-23 — LEADING DIAGNOSIS FALSIFIED (read this first)
> The two decisive experiments in §4 are now DONE, and BOTH ruled out the leading
> diagnosis below ("biharmonic ~20-35% less stable" / "upstream grid-scale energy"):
>
> - **Exp #1 (standalone bidiff diff, §4.1) — DONE, DECISIVE.** Added `FESOM_BIDIFF_PROBE`
>   to BOTH codes (C `fesom_main.c` before the step loop; Fortran `oce_ale.F90` top of
>   `oce_timestep_ale`): prescribe element velocity from an IDENTICAL gid/level/component
>   integer hash, zero `uv_rhs`, run ONE `visc_filt_bidiff`, dump owned surface `uv_rhs`
>   by gid. Ran np=8 / dist_8 (job 25087601). **244659/244659 elements match; max abs
>   diff 1.2e-12 on a field of magnitude 0.75-14.3 (≈1e-12 relative to scale); 0 elems
>   differ >1e-8.** => the biharmonic operator is **BIT-IDENTICAL** to Fortran on identical
>   input. **The dt=1800 bug is NOT the biharmonic operator.** (Scripts:
>   `scripts/exp1_compare_bidiff.py`; job `job_bidiff_probe`; dumps `/work/.../bidiff_probe/`.)
> - **Exp #2 (grid-scale velocity energy, §4.2) — DONE.** Re-measured cell-velocity 2Δx
>   roughness C(`windrot_dt1800_test`) vs Fortran+PP(`fortran_pp_dt1800`), March surface,
>   WITH both fixes in. **In the central-Arctic blow region the C is NOT noisier:** >80N
>   C/F=0.94, MIZ 0.97, Arctic 0.98, ice-interior 0.96. (C slightly noisier only in
>   tropics 1.22 / ice-free 1.12 / global 1.08 — NOT where it blows.) The wind-rotation
>   fix removed the prior 1.55× MIZ excess. => **no steady upstream Arctic grid-scale
>   velocity excess.** (Script `scripts/exp2_gridscale_energy.py`.) Caveat: monthly mean
>   could hide a sign-flipping instantaneous 2Δx; April onset (~step 5000) not yet
>   measured instantaneously.
>
> ### 🎯 ROOT CAUSE FOUND (2026-05-23, test running) — AB2 epsilon mis-port
> After exonerating the biharmonic, the chain led here. The growing mode is a **free
> velocity 2Δx** (probe at the eruption node: `velsprd` grows steadily while `pgf_sprd`
> AND `rho_sprd` stay flat → NOT buoyancy-forced), localized to the **highest-latitude
> central-Arctic** nodes, dt-dependent, slow (~110-day) growth. Vertical advection is
> net-STABILIZING (`FESOM_NO_VADV` blew at step 395, far earlier), so the earlier
> "w-driven" idea is wrong (w just follows the velocity divergence). A free, high-latitude,
> dt-dependent velocity instability = the **Coriolis time integration**.
>
> **THE BUG:** the AB2 *stabilization offset* `epsilon` was mis-ported. Fortran `o_PARAM`
> uses **`epsilon = 0.1`** ("AB2 offset", `oce_modules.F90:92`; no namelist override; used
> at `oce_ale_vel_rhs.F90:98-99` as `ab1=-(0.5+ε)`, `ab2=(1.5+ε)`). The C used
> **`eps = 1.0e-9`** (`fesom_momentum.c:64`) with a comment that *wrongly* claimed Fortran
> uses 1e-9. So the C ran **pure AB2 (−0.5, 1.5)** while Fortran runs the **stabilized
> offset-AB2 (−0.6, 1.6)**. Pure AB2 is marginally unstable for the oscillatory Coriolis
> term; the offset moves the stability region to damp it. Without it the C grows a velocity
> 2Δx worst where `f·dt` is largest (central Arctic) at dt=1800 — invisible at dt=500
> (small `f·dt`). This fits EVERY observation (free velocity mode, highest latitude,
> dt-dependent, slow growth, biharmonic/advection faithful). Classic
> `feedback_port_used_terms` pattern (a USED parameter mis-ported, no-op at dt=500).
> **FIX:** `eps = 0.1` in `fesom_momentum.c` (and the same mis-port in
> `fesom_tracer_adv.c:175` for the tracer AB2 — fixed too for fidelity).
> **✅ CONFIRMED (job 25087877, dt=1800, momentum fix): reached step 11000 (~day 229),
> exit 0, healthy throughout.** At step 5000 global `uv = 1.51` (BOUNDED) vs the baseline's
> `3.70` (runaway); sailed **past the old blow step 5308 AND the dt=1740 blow 7666**; at
> step 11000 m_ice 4.64, uice 0.81, `velsprd` bounded 1.4e-3-1.5e-2 (inside Fortran's
> 0.006-0.039 band) the entire run, `pgf_sprd` flat, no eruption — matching the stable
> Fortran+PP reference (also day 229). `uv` drifts to ~3.0-3.4 in the fresh-surface
> summer (bounded variability, not a runaway; vs Fortran+PP's ~1.5-2.0 — quantify in
> climate validation). **The day-110 dt=1800 blow-up is FIXED.** Remaining (task #5):
> climate-validate dt=1800 (both eps fixes) vs `fortran_pp_dt1800` (job 25087954 in flight),
> verify dt=1200 didn't regress, and commit the two `eps` fixes. (Jobs: `job_eps_fix_dt1800`,
> `job_eps_validate_1yr`; `FESOM_DIAG_SPREAD` trajectory in `/work/.../eps_fix_dt1800/run.err`.)
>
> **NEW PICTURE (pre-epsilon):** operator faithful + March Arctic velocity equal => the day-111 eruption
> is a **fast APRIL event** with a DIFFERENT cause than the (now-exonerated) biharmonic.
> Next: find which term seeds the April element-velocity 2Δx. Untested candidates:
> **vertical momentum advection w·du/dz** (the use_wsplit history #5 already proved the
> central-Arctic *vertical* CFL exceeds 1 in early April), PGF/density from spring-melt
> freshwater stratification, the velocity update / SSH-grad correction, or AB2 accumulation.
> Plan: (a) C-only April-onset instrumentation — which field erupts first (w-spread vs
> velsprd vs pgf_sprd) — then (b) dual-instrument that term vs Fortran+PP at the eruption
> node to step ~5000. The §1-§3 "leading diagnosis / RULED OUT" text below PRE-DATES this
> and is partly obsolete (much of it was reasoned FROM the biharmonic narrative); trust the
> SOLID rule-outs (rank-independence, operator bit-identity, use_wsplit, PP-authorized) and
> RE-OPEN the momentum-RHS components (advection/PGF/Coriolis/SSH-grad) and the vertical path.


**Goal:** make the C port run stably at Fortran's CORE2 timestep **dt=1800** (30 min).
Today it blows up at **~model-day 111** (central Arctic) and the usable workaround is
**dt=1200** (20 min). This doc consolidates everything tried so you don't repeat it, how
to run C **and** Fortran experiments, and the decisive next steps.

Repo: `/home/a/a270088/port2/fesom2_port` (C port) and `/home/a/a270088/port2/fesom2` (Fortran).
Current HEAD: **`8dac997`** (both physics fixes in — see below).

---

## 1. TL;DR — current state & best diagnosis

- **dt=1800 blows up at step ~5300–5400 (~day 111), central Arctic**, via a growing
  **2Δx (checkerboard) mode in the ELEMENT (cell) ocean velocity**. Node moves with the
  partition (gid 23926 / 87875 / 47138 at ~82–84°N) → marginal instability.
- **Best diagnosis (still the leading one):** the C's **explicit horizontal biharmonic
  viscosity** (`visc_filt_bidiff`, opt_visc=7) is at/over its explicit-stability (CFL)
  limit at dt=1800, so it can't damp the central-Arctic velocity 2Δx; Fortran's (same
  formula/coeffs) stays under. The C's effective biharmonic stability limit is
  **~dt 1450–1500 vs Fortran's >1800 → the C operator is ~20–35% "less stable"** for
  reasons invisible to line-by-line formula comparison (formula/coeffs/elem_area/coverage
  ALL verified identical to Fortran).
- **The bidiff CODE is provably identical to Fortran**, so the residual is most likely in
  the **velocity field fed to it** carrying slightly more grid-scale (2Δx) energy than
  Fortran's — OR a subtle operator-application difference (sub-cycling / ordering / a
  limiter / smoothed coeff velocity) that Sergey would recognize.
- **dt=1200 is the validated workaround** (2-yr clean; 5-yr reached year 4.9 then tripped a
  conservative `uv>5` guard — see §6, likely a transient).

### IMPORTANT: two physics fixes landed AFTER most of the dt=1800 hunt
The dt=1800 notes in memory (`project_dt1800_state`) predate these. Both are committed and
**neither fixes dt=1800** (re-verified):
- **`ad1d3b7`** restore `0.5` in `ice_strength` (ice_EVP.F90:599) — reverts the WRONG
  `8d31e16`. Unlocked the Arctic ice pack (drift/ridging).
- **`8dac997`** rotate JRA winds geo→rotated frame (`fesom_vector_g2r`) — was missing;
  fixed mis-directed Arctic wind stress on ocean+ice.

**dt=1800 with BOTH fixes (tested 2026-05-23, `windrot_dt1800_test`): still blows at
step 5308 (~day 111)** — same timing as before (5386 buggy, 5339 with 0.5 only). One
change: the blow-up now manifests as **ice-velocity overflow** (`max_uice→1e19`, CG
diverged=0) instead of the earlier **ocean CG divergence**, because the now-mobile
(correct) ice participates in the same central-Arctic instability the locked ice couldn't.
**Net: dt=1800 root cause is unchanged and ocean-side.**

> **Re-test opportunity:** the wind-rotation fix makes the C ocean velocity *direction*
> correct (it was mis-rotated before, a confound). The "C carries ~20–35% more grid-scale
> velocity energy" hypothesis should be **re-measured cleanly now** with both fixes in.

---

## 2. The blow-up signature (so you recognize it)
At dt=1800, ~day 111 (step ~5300–5400): a single central-Arctic node (~82–84°N, gid moves
with partition) erupts in ~1–5 steps — ocean `uv`→runaway, then `m_ice`/`uice`→1e8…1e274.
Pre-cursor: global max ocean `uv` is bounded ~1.5–2.0 until ~step 3000, then the C's runs
away (4.1 @5000) while Fortran+PP stays 1.5–2.0 all year. The seed is an **element-velocity
checkerboard** in the central Arctic; SSH/`hbar` 2Δx FOLLOWS it (not the driver).

---

## 3. RULED OUT — do **not** re-investigate these (all verified == Fortran or tested)
- **PP-vs-KPP / vertical mixing** — authorized simplification, NOT the bug (user emphatic).
  Kv/Av match Fortran+PP (corr 0.89). Don't chase the ~0.24°C SST bias either (it's the dt cap).
- **Biharmonic FORMULA / coeffs / elem_area / cyclic / edges / U_c exchange** — byte-identical
  to Fortran `oce_dyn.F90` visc_filt_bidiff (two-stage, γ 0.003/0.1/0.285, /elem_area,
  VISCEDGE skip count=0, exchange count=1). `elem_area` byte-identical (cos(mean-lat)·cross·0.5·R²).
- **VISC_MULT scan** — more viscosity does NOT help (×0.25→blow5422, ×1→5386, ×2→5054,
  ×4→609); γ1 (flow-aware) is the active term but scaling it any way doesn't fix it →
  STRUCTURAL, not a coefficient. (Knobs `FESOM_VISC_MULT`, `_G0/G1/G2_MULT` still in tree.)
- **PGF / density / hpressure** — `pgf_sprd` flat ~3.7e-8 the whole run while velsprd grows 100×.
- **SSH solver (CG)** — converges <100 iters (cap 500), `it=66` even at the tipping; not the cause.
- **Halo / partition / exchange** — 256r vs 864r blow within ~0.8% (5342 vs 5386) → INTRINSIC,
  rank-independent. Not the U_c exchange.
- **Momentum advection** — net-STABILIZING (removing it blows earlier); node↔element averaging
  is a damper, not an injector. (Committed `2f29b24`.)
- **Ice as the DRIVER** — `FESOM_NO_ICE_DYN` (ice vel off) still blows ~day 111 (1 day earlier).
  Ice dynamics is mildly stabilizing via drag. The MIZ ice-vel 2Δx (C ~1.34–1.55× Fortran)
  is REAL but SECONDARY (Sergey's "ice velocity is the cause" → refuted by NO_ICE_DYN).
- **use_wsplit** — already fixed (`d9084c0`, =.false. to match CORE2). check_opt_visc,
  trim_cyclic/cyclic_length — all verified identical.
- **dt threshold is NOT a clean CFL cliff** — dt=1740 blows day155, 1500 day399, 1200 stable.
  Smaller dt only DELAYS the 2Δx (nonlinear, near-threshold), it doesn't cleanly switch off.

---

## 4. The decisive experiments NOT yet done (start here)
1. **Standalone bidiff diff at identical state (the cleanest test, never done).** Dump the
   full element velocity field from BOTH C and Fortran at a pre-blow step (~5000) at the
   SAME partition; run ONE `visc_filt_bidiff` step in each on the SAME input; diff the
   `uv_rhs` output. If identical → bidiff is fine, the 2Δx energy is upstream in the
   velocity (re-measure with both fixes in). If different → a bidiff implementation bug.
2. **Re-measure C-vs-Fortran grid-scale velocity energy at dt=1800 WITH both fixes** (wind
   rotation now removed a direction confound). Use the roughness/2Δx metric
   (`scripts/plot_checkerboard_where.py` style) on a ~step-5000 snapshot, C vs Fortran+PP.
3. **Ask Sergey the sharpened question:** the biharmonic formula/coeffs are identical, the
   deficit is ~20–35% in effective stability, intrinsic & rank-independent, velocity states
   match until the tipping. *What in FESOM2's `visc_filt_bidiff` controls its effective
   explicit-stability margin* — is the flow-aware |∇u| coefficient computed from a
   smoothed/previous-substep velocity, is there a limiter, sub-cycling, or a different
   apply-order vs the velocity halo exchange?

---

## 5. How to run experiments

### 5a. C port
- **Build** (login node): `cd /home/a/a270088/port2/fesom2_port && bash -l configure.sh`
  for a clean build, or incremental: `source /sw/etc/profile.levante && source env.sh &&
  cd build && make fesom_port`. Strip `OMPI_MCA_*` on the login node. (See memory
  `feedback_levante_build`.) Binary: `build/fesom_port`.
- **dt is a RUNTIME arg** — no recompile needed to change it. Args:
  `srun ./build/fesom_port <MESH> <OUT_DIR> <dt_sec> <nsteps> <snap_every|-1> <PHC_nc> <start_year>`
- **Job template:** `job_c_5yr_dt1200` (7 nodes / 864 ranks). For a dt=1800 probe copy it,
  set `OUT_DIR` to a NEW dir (memory: unique OUT_DIR per job!), and set the args, e.g.
  `... <OUT_DIR> 1800 6500 -1 <PHC> 1958` (6500 steps clears the ~5300 blow point).
- MESH=`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`;
  PHC=`/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc`.
- **Watch the blow-up:** `export FESOM_DIAG_MICE=1` then
  `grep '\[mice-mon\]' run.err` (per-step max_mice/max_uice; the uv spike + overflow).
  `run.log` prints global `uv`/`eta`/CG-iters every 1000 steps.
- **Job-script gotcha:** a trailing `grep -c "...diverged"` returns exit 1 on 0 matches →
  SLURM marks the job FAILED even though `srun` exited 0. Check `Exit code:` in slurm.out.
  Add a final `true` (as in `job_ice_fix_2yr`).
- **`uv>5` guard:** `fesom_main.c:1100` `MPI_Abort(99)` if `uv_max>5` or NaN. It's a
  tripwire, not a blow-up — raise the threshold if probing past a transient.

### 5b. Fortran reference (this is what you compare against)
- **Build:** `cd /home/a/a270088/port2/fesom2 && <env>; make` (see `bin/current_shell_path`).
  **GOTCHA:** `fesom.x` rpath loads `lib64/libfesom.so` (`$ORIGIN/../lib64`), NOT
  `build/lib/` — after editing+building you MUST `cp build/lib/libfesom.so lib64/libfesom.so`
  (and `touch` the edited .F90 to force recompile). This bit twice.
- **Set up a fresh run (copy a work dir, retarget paths, cold-start clock):** see the
  ready example **`work_linfs_pp_5yr_d1200/`** built this session. Steps:
  1. `cp work_linfs_pp_2yr/namelist.* NEWDIR/ ; cp -P work_linfs_pp_2yr/fesom.x NEWDIR/`
  2. Edit `NEWDIR/namelist.config`: `step_per_day` (dt=86400/spd → **dt=1800 ⇒ 48**,
     dt=1200 ⇒ 72), `run_length` (years), `ResultPath = '/scratch/.../NEWRES/'`.
     PP is in `namelist.oce` (`mix_scheme='PP'`); `which_ALE='linfs'`; opt_visc/use_wsplit
     already match CORE2 in these dirs.
  3. **Cold-start clock** (else MPI_ABORT in setup): write `<NEWRES>/fesom.clock` with TWO
     identical lines `  0.0   1   1958` (timeold/dayold/yearold == timenew/daynew/yearnew
     ⇒ cold start). Restart clock looks like `85200 365 1962 / 0.0 1 1963`.
  4. Job script: see `work_linfs_pp_5yr_d1200/job_ftn_5yr_d1200` (cd to dir, `srun -l ./fesom.x`).
- **Fortran dual-instrumentation already in tree** (`src/oce_ale.F90`): `[FSSH]` (hbar +
  1-ring spread at chosen gids), `FDBG_GLOBAL` (global max uv), `FHBAR_DBG`. The
  `FDBG_GLOBAL` print cap was widened to `n<=50 .or. mod(n,100)==0` (oce_ale.F90 ~3908).
  These produce the verbose per-step output in `fesom2.0.out` — fine for diagnosis, but
  **disable for any long/production Fortran run** (it ~doubles wallclock & bloats the log).
- **Confound-free baseline already exists:** Fortran+PP is STABLE at dt=1800
  (`work_linfs_pp`, ran to day 229) → proves dt=1800 is a C bug, not PP being too weak.

### 5c. Both at once / drift
`scripts/drift_5yr.py` (nereus) compares C vs Fortran over N years. Speed: measure step
rate of both live (head-to-head they're equal, 1.07×).

---

## 6. dt → blow-up day (864r, full ice) — the empirical curve
| dt (s) | min | outcome |
|---|---|---|
| 1800 | 30 | blow **day 112** (step 5386 buggy / 5308 both-fixes), central Arctic |
| 1740 | 29 | blow day 155 (step 7666) |
| 1500 | 25 | blow day 399 |
| 1200 | 20 | **STABLE 2 yr**; 5-yr reached year 4.9 (step 128000) then tripped `uv>5` guard (uv 2.5→6.84 in 1 step — sudden, likely transient; NOT a numeric blow-up, CG fine, fields sane). **OPEN: restart past it with a higher guard to confirm benign.** |

---

## 7. Diagnostic knobs (all live in the C source)
- `FESOM_PHASE1_DT` — set from argv[3] (runtime dt, no recompile).
- `FESOM_DIAG_MICE=1` — per-step `[mice-mon]` max m_ice/uice (+gid).
- `FESOM_DIAG_SPREAD=<gid>` — per-step 1-ring spread of hbar/d_eta/ssh_rhs + surrounding-
  element surface |uv| min/max + pgf_sprd + uvrhs_sprd (the checkerboard probe).
- `FESOM_DIAG_SSHSLV` — ssh_rhs/d_eta/hbar (SSH solve runaway).
- `FESOM_DIAG_VISCDUMP=<gid>` — surface uv_rhs before bidiff, then u / bidiff_du / u·du.
- `FESOM_VISC_MULT`, `FESOM_VISC_G0_MULT/_G1_MULT/_G2_MULT` — scale the whole / per-coeff
  biharmonic (the VISC_MULT scan used these).
- `FESOM_NO_ICE_DYN / _ADV / _THERMO` — turn off parts of the sea ice (fesom_ice.c:294).
- `FESOM_EVP_DUMP_DIR` — per-substep EVP dumps (memory `reference_evp_dump_diagnostic`).
- `FESOM_DIAG_GID`, `FESOM_DIAG_VISCEDGE` — node ice+ssh spread / interior-edge-skip count.

---

## 8. Key paths
- C source: `/home/a/a270088/port2/fesom2_port/src` ; binary `build/fesom_port` ; jobs `job_*`.
- Fortran: `/home/a/a270088/port2/fesom2` ; ref run dirs `work_linfs_pp*`, `work_linfs_d1800`.
- C outputs: `/work/ab0995/a270088/port/<run>` (e.g. `c_5yr_dt1200`, `core2_864_2yr_dt1200`,
  `windrot_dt1800_test`, and the many `dt1800_*` probe dirs = the experiments already run).
- Fortran outputs: `/scratch/a/a270088/<run>` (e.g. `fortran_pp_2yr`, `fortran_5yr_d1200`).
- mesh.diag (for analysis): `/scratch/a/a270088/fortran_pp_2yr/fesom.mesh.diag.nc`.
- Analysis env: `PYTHONPATH=/home/a/a270088/PYTHON /work/ab0995/a270088/mambaforge/envs/nereus/bin/python`.
- Memory (auto-loaded, READ FIRST): `project_dt1800_state` (full hunt log),
  `feedback_port_used_terms`, `feedback_write_loops_halo`, `project_5yr_milestone`,
  `feedback_ice_strength_halffactor`, `feedback_wind_rotation_g2r`.

---

## 9. One-line status
dt=1800 = a central-Arctic explicit-biharmonic 2Δx instability; C's effective biharmonic
stability is ~20–35% tighter than Fortran's despite an identical operator → the extra
grid-scale velocity energy (or a subtle apply-order difference) is the target. Both physics
fixes (ice 0.5, wind g2r) are in and don't change it. dt=1200 works. Next: standalone
bidiff diff at identical state + re-measure grid-scale energy with both fixes + Sergey's
input on the biharmonic's effective stability margin.
