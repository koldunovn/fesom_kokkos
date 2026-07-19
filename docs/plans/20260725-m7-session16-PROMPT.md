# SESSION 16 PROMPT — solve wsplit; harvest the scaling-fleet tail

*Handoff written 2026-07-19 end of session 15. Two work streams: (A) the wsplit
debugging loop until the NG5 integration test passes, (B) passive: harvest the last
scaling points as the queue drains. Read this top to bottom before touching code.*

---

## A. THE WSPLIT PROBLEM — state of the hunt

### What wsplit is and why we need it
`use_wsplit` splits the vertical advection velocity at CFLz>maxcfl cells into an
explicit part `w_e` (CFL-capped) and an implicit part `w_i`. Production FESOM2
always runs it on large meshes (user). Without it, cold starts on fine partitions
blow up at the Strait of Gibraltar (rule 0.41); with it, **Fortran** rides the
CFLz spike and completes. The port must match this to run cold starts at
production dt (dars/NG5 dt240) and for JUPITER-scale campaigns.

### Proven facts (do not re-derive)
| fact | evidence |
|---|---|
| Fortran WITHOUT wsplit dies too | F0 = job 26360443: T-NaN at step 230, blowup dump on disk |
| Fortran WITH wsplit completes | F1 = 26360444: rc=0, 300 steps, rides 334 CFLz>1.75 warnings |
| The instability site is THE SAME in both models | rank 2813, glon/glat **−5.50/35.98** (Gibraltar); F0's CFLz warnings 108× there; the port's uv-max settles exactly there from step ~110 |
| The port's FCT tracer path was missing wsplit machinery | root cause `e0c963e`: no `adv_tra_vert_impl`, HO fluxes used `w_e` not full `w` |
| That machinery is NOW IMPLEMENTED | `45dcde0` (+diag `3dc091e`): knob `FESOM_WSPLIT` (default OFF), adv_tra_vert_impl host+device, LO-flux recompute with full w, HO qr4c full w |
| Knob-off is UNTOUCHED | serial byte gate 26364691 **BIT-IDENTICAL**; CUDA fidelity 26364696 **PASS** (floor 3.913e-03). The branch is safe. |
| The port + implemented wsplit STILL dies | 26364692/26364945/26365082: uv-guard rc=99 at step ~139 (uv ramps 4.79→5.03), or CG "residual diverged" ~200 when unchecked (guard fires only at print steps). Deterministic (both reps identical). |
| T/S stay clean through the death | per-step diag: T[−2.07,30.15] S[5.64,41.16] to the end ⇒ tracer path not corrupting; the ramp is MOMENTUM |
| Structure audits are EXHAUSTED | tracer TDMA faithful (F90:90-240 incl. block order); momentum TDMA line-by-line vs oce_ale.F90:3160-3231 ✓; cfl_z formula equivalent (Fortran's overwrite is a bit-repro quirk, values equal); dt is runtime (`FESOM_PHASE1_DT`→global); Fortran vert_vel_ale's Wvel_i association is DEAD code; partial cells OFF both sides; explicit momentum adv uses w_e both sides (vel_rhs.F90:403 ↔ momentum.cpp:171/302) |

**Bottom line: same instability, same cells, both models; wsplit cures only
Fortran. The port's wsplit is functionally short of Fortran's in a way the
structural audit cannot see. Find the functional difference.**

### ⚠️ Lesson from tonight's misread (don't repeat)
The `[uvmax]` diag prints CFLz at the *uv-argmax cell only*. That showed
CFLz 0.01–0.4 while uv ramped, which I over-read as "not CFL-driven" — wrong:
the COLUMN/global CFLz max lives at the same strait (F0 proves it). Always
compare the **global cflz_max** and its location, not a single cell.

### Next probes, in priority order

**P1 — cflz_max(t): port vs Fortran (the designed discriminator, ~30 min).**
1. Extend the `FESOM_DIAG_UVMAX` block (fesom_main.cpp, search `[uvmax]`) to also
   compute + print the global max of `dyn.cfl_z` and its (node, nz, glon/glat)
   per print step (same pattern as uv; cfl_z is host-valid on Serial).
2. Rebuild serial, resubmit the probe:
   `sbatch -N32 --time=00:30:00 --export=ALL,MESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5,DT=180,NSTEPS=150,TAG=ws_ng5_dbg3,FESOM_WSPLIT=1,FESOM_PRINT_EVERY=5,FESOM_DIAG_UVMAX=1 jobs/job_m7_scale_cpu`
3. Extract Fortran F1's CFLz_max(t): the warning lines in
   `/work/ab0995/a270088/port2/fortran_timing/wsF1_ng5c32_dt180_26360444/fesom2.0.out`
   carry `CFLz_max=X.XX, mstep=N` (only when >1.75; absence ⇒ below 1.75).
4. Compare the two curves.
   - Port cflz_max ≪ Fortran's at the same steps ⇒ the divergence is **upstream
     of wsplit** — the port's `w`/continuity/eta at the strait differs. Then dig
     into `w` at the strait: compare the port's w column at rank-2813's cells vs
     F0's blowup dump (`fesom.1958.oce.blowup.nc` has w and `w_impl`), and audit
     vert_vel/ssh at steep topography (thin bottom cells at the sill).
   - Curves track ⇒ the divergence is **in the wsplit response** — go to P2.

**P2 — momentum-TDMA response check (if P1 says "response").**
The momentum implicit TDMA (impl_vert_visc, fesom_momentum.cpp ~640-780) is
structurally faithful but numerically unvalidated at w_i≠0. Options:
- Cross-check on a synthetic column: lift the port TDMA + the Fortran formula
  into a tiny offline test (same inputs → compare outputs to machine precision).
- Or instrument: print the TDMA's uv before/after at the rank-2813 hot column
  for steps 100-140 and compare growth against F1's behavior (Fortran has no
  such print — compare qualitatively: uv must stop ramping).
- Re-audit the **rhs** side of the momentum TDMA (I only verified coefficients;
  the rhs/friction/stress blocks below line 3231 of oce_ale.F90 were NOT
  compared — read oce_ale.F90:3232-3300 vs fesom_momentum.cpp continuation).

**P3 — candidate functional gaps not yet ruled out.**
- The M5.14 **w_i one-step lag**: port momentum (substep 6) reads LAST step's
  w_i; port tracers (substep ~13) read THIS step's. Verify Fortran's call order
  matches exactly (oce_timestep_ale in oce_ale.F90: where impl_vert_visc_ale and
  compute_Wvel_split sit relative to each other and to the tracer step). A
  one-step-early/late w_i in momentum could halve the effective damping.
- **GM/bolus add**: step.cpp:1090 adds fer bolus to uv/w/w_e (not w_i) before
  tracers. Check Fortran's fer_Wvel add points cover the same arrays at the same
  places (w = w_e + w_i stays consistent, but WHICH arrays carry bolus into the
  HO/LO/TDMA calls matters under wsplit).
- **wsplit_maxcfl**: port constant 1.0 (`FESOM_PHASE1_WSPLIT_MAXCFL`); check the
  F1 namelist value (`grep wsplit /work/.../wsF1_*/namelist.dyn`) — if Fortran
  ran 0.5, the port splits far less aggressively. Cheap: make it env-overridable
  and probe maxcfl=0.5.
- The **ice stress / EVP** at the strait is NOT a suspect (T/S clean, site is
  ice-free Gibraltar).

**P4 — the heavy hammer (only if P1-P3 fail).**
Field-level dump comparison: F0's blowup dump has the full state at death;
teach the port to dump the same fields at a chosen step (snapshot machinery
exists — SNAP_EVERY) and difference the Gibraltar columns port-vs-F0 at matched
model times to find the first diverging field.

### After the fix works (NG5 dist_4096 dt180 WSPLIT=1 completes 300 steps)
1. Re-run knob-off gates (serial byte + CUDA fidelity) — MANDATORY after every
   src change (`M7_TAG=... sbatch jobs/job_m7_gate_serial` / `job_m7_gpu_gate`).
2. CUDA wsplit-on smoke (the device TDMA kernel is untested at w_i≠0).
3. Fortran wsplit-ON CORE2 reference (`job_m524_scale_fortran` + `WSPLIT=true`,
   CORE2 dt1800 — the splitter fires in the deep-convecting central Arctic) +
   port CORE2 wsplit-on run → climate-level comparison at the scheme-floor class.
4. Options ×3 ladder (L91) for the knob; document in the options matrix as a
   MANUAL knob (physics option class, like FESOM_MIX_SCHEME — never in
   FESOM_SPEED).
5. Payoffs to then consider: dars/NG5 measurement legs at production dt240;
   JUPITER cold starts; note in SCALING-REMEASURE + JUPITER plan docs.

### Debug hygiene
- Iteration cost ~8 min/rep: 32-node compute, short walltime backfills in
  14 s–11 min (measured). CPU-side only — no GPU queue.
- Diag tools: `FESOM_WSPLIT=1`, `FESOM_DIAG_UVMAX=1`, `FESOM_PRINT_EVERY=N`
  (scale job passes all through). The uv>5 guard fires ONLY at print steps.
- Serial build: `source env.sh; cmake --build build-m7serial -j 16`.
  CUDA: `source env_cuda.sh; cmake --build build-m7cuda -j 16`.
- After ANY src edit: knob-off byte gate before trusting anything else.
- Commits LOCAL; **ask before push** (branch m7-speed, local through `3dc091e`+).

---

## B. THE SCALING FLEET — passive watch

**Pending (weekend queue, monitors were session-bound — re-arm or poll):**
| job | point | feeds |
|---|---|---|
| 26354135 | dars g16n (dt120) | scaling A/B + speedup 16N + before/after |
| 26354136 | dars g32n (dt120) | same, 32N |
| 26351275 | ~~NG5 g16n~~ LANDED (in figs) | — |
| 26351276 | NG5 g32n (dt180) | scaling + speedup 32N (CPU partner = the dt60 point, adopted) |
| 26351266 | farc g32n (dt900) | scaling panels only (no CPU partner) |

**On each landing:** `python3 scripts/m7_scaling_figs.py` regenerates everything
(harvest → CSV → `fig_m7_scaling.png` + `fig_m7_speedup.png` +
`fig_m7_before_after.png` in `/work/ab0995/a270088/port2/m7/scaling_figs/`).
L80 announce-check the new legs (knobs printed in the abenv.<job>.out).
When ALL landed: mark task #11 done; re-derive the CG dt-correction 120→240 for
the SYPD footnote (from measured iters in the dars logs); update
`docs/plans/20260724-m7-SCALING-REMEASURE.md` with the final table.

**Figure conventions locked by the user:** one combined scaling figure (A solid/
circles, B dashed/triangles); NG5 32N CPU = the dt60 point (caveat in the CAPTION
not on the figure — caption text ready in SCALING-REMEASURE doc); dars at dt120
(= JAX dt); SYPD at production dt; per-mesh dt footnoted.

## C. Everything else (done, don't reopen)
- Hindcasts harvested: 63A 41 yr / 63B 52 yr (1958–2009 = the full RMSE window),
  zero NaN; drift/OHC figure + QC maps in `/work/ab0995/a270088/port2/climate63/checks/`
  (Tbar gap 0.0000 °C, OHC 0.1/0.2 ZJ; COMMON-WINDOW rule for profiles).
- dars saga resolved (rule 0.41), full ladders at dt120 green.
- Local commits through `3dc091e` + this handoff — ask before push.
- Detailed session log: memory `project-m7-speed-campaign` + task #12 (wsplit
  debug state, kept current).
