# M16 — Mixed precision (exact port of upstream #940) + the #984 SSH preconditioner as default

**Date:** 2026-09-02 · **Branch:** `m16-precision` (new worktree `~/port_kokkos_sp`, off `m14-integrate`
`d4a9fe0`) · **Status:** PLANNED — brainstormed section by section and user-validated 2026-09-02; **Rev 2**
after plan-review (17 findings applied, verified against source; see Review history) · **Plan file
lives in the main checkout** (`~/port_kokkos/docs/plans/`) and is copied into the worktree by Task P2.

---

## Overview

Bring single precision to the integration branch as an **option**, as an **exact port of what the
Fortran model merged** (FESOM/fesom2 #940 with its companions #986 salt anomaly, #995 preconditioner
default, #997 stiffness shadow), and make the **#984 symmetric preconditioner the default** so that
every later experiment (SSH solvers, split-explicit, wide halo, mEVP lean, det IC, restart I/O) can be
run at either precision from one tree.

What exists already, and why this is not a from-scratch port:

- **July's M8 track** (`~/port_kokkos_mp`, branch `m8-precision`, 61 commits off `1df683b`, never
  merged) implemented the mechanism (`real_t`/`dbl_t`/`FESOM_MPI_REAL`), passed Gates 0–4, and found
  two single-precision-only defects that upstream later fixed independently (the JRA time axis and the
  SSH stiffness-matrix drift). Its **run artifacts and frozen binaries are gone** —
  `/work/ab0995/a270088/port2/mp` no longer exists — so every single-precision verdict must be
  re-earned. Its code, registry, lessons and scripts survive on the branch and are the reference.
- **The #984 knob** (`FESOM_SSH_PRECOND=0..4`) exists only as an uncommitted diff in
  `~/port_kokkos_pre` (`src/fesom_ssh.cpp` +91, `tools/fesom_ssh_lab.cpp` +33), default 0, silent.
- **`m14-integrate`** (`~/port_kokkos_int`, HEAD `d4a9fe0`, 489 commits ahead of `main`) contains
  M9+M10+M11+M12b+M13 plus restart I/O, and carries **none** of the precision machinery: its
  `fesom_types.h` is the pre-M8 `typedef double real_t;`. Its new code brings 127 `MPI_DOUBLE` sites and
  ~950 raw-`double` lines that need classifying no matter how M8 is brought over.

The user's steer that shapes everything below: **"make it the EXACT port of the original Fortran; even
where M8 already had an option, re-evaluate it."** Where July was more cautious than Fortran, the
caution is dropped and must be re-earned by a failing gate. Where Fortran is more cautious (CVMix in
double), Fortran wins and the cost is measured.

**Declared divergences from the exact port** (documented, outside the D5 invariant): (a) upstream's
merged code defaults `precond_variant = −1` (auto: 1 in SP, 0 in DP) with the shipped namelist at 1;
the port has no auto mode and defaults to 1 in both builds. (b) On a single-precision restart the
port writes the FP64 stiffness shadow into the (already double) `stiff_values` restart variable so the
SP round-trip stays bit-exact; upstream re-seeds from the WP matrix (Task B3b). (c) Upstream's
point-slope forcing interpolation is compiled only in the SP build (Task B6b).

**Non-goals.** Half precision; anomaly variables beyond salinity; coupled builds; the 63-year hindcast
(deferred to a user decision after Gate 4).

---

## Context (from discovery, verified 2026-09-02; review-verified against source)

### Environment right now

- `/work` and `/scratch` are **read-only**; `$HOME` is writable (224 TB free). Meshes and the m14
  reference binaries on `/work` are **readable**. The Claude Bash tool is dead this session (scratchpad
  on `/scratch`); the Monitor tool's shell works.
- Consequence: code, Serial builds and pi-mesh + login-node CORE2 gates proceed now with outputs in a
  time-boxed `~/m16_scratch`; everything that needs `/work` (SLURM, frozen binaries, campaign legs)
  waits. Tasks are tagged **[NOW]** or **[BLOCKED on /work]**.

### Upstream mechanism (merged state; diffs archived at `~/tmp_claude/pr{940,984,986,995,997}.diff`,
extract `~/tmp_claude/pr940_mech.txt`; copied into `docs/reference/upstream_sp/` in Task P2)

| upstream element | Fortran location | note |
|---|---|---|
| `WP = real32|real64`, `MPI_WP = MPI_REAL|MPI_DOUBLE_PRECISION` | `oce_modules.F90` o_PARAM, `MOD_PARTIT.F90` | CMake `USE_SINGLE_PRECISION` (default OFF); compiler default-real flag flips (`-r4`), so every unkinded `real` is WP |
| forcing time axis real64: `nc_time`, `rdate`, `delta_t`, `time_t0`, `binarysearch_r8` | `gen_surface_forcing.F90` | **point-slope form** `atm = coef_b + (rdate − time_t0)·coef_a`, `coef_a/b` in WP, `time_t0` real64 — replaces the affine line **unconditionally** (`pr940.diff:2888-2928`), i.e. changes FP64 rounding too |
| stiffness shadow `values_full` (real64), SP-only `#ifdef` | `MOD_MESH.F90`, `oce_ale.F90` update_stiff_mat_ale | seeded on **first update** (restart-correct), refreshed into `%values` once per step; optional `DIAG_STIFF_DRIFT` |
| CVMix fixed real64, WP↔`cvmix_r8` shims at the call boundary | `gen_modules_cvmix_{tke,kpp,idemix,tidal,pp}.F90` | exact no-op in DP |
| `WP_full` (real64) for global integrals, matching MPI type | `oce_modules.F90`, integrate_nod | everything else (min/max diagnostics, mesh volumes, Bcasts, halo, `MPI_TYPE_INDEXED`) is `MPI_WP` |
| KPP guard `epsln = 1e-20` in SP (1e-40 flushes to 0 under FTZ) | `oce_ale_mixing_kpp.F90` | |
| precision banner (kind, storage bits, digits, epsilon) | `fesom_module.F90` | + once-per-run I/O precision report (`note_output_precision`, `io_meandata.F90`) |
| 8-byte output streams accumulate in real64, 4-byte in real32 | `io_meandata.F90` | |
| `use_salt_anomaly` (namelist `&oce_dyn`, default `.false.`), `S_ref_anomaly = 35|0` | `oce_modules.F90`, `oce_setup_step.F90` | consumers add it back: EOS ×3 + `sw_alpha_beta` (`oce_ale_pressure_bv.F90`), ice–ocean gather `S_oc`, `rsss`, `relax_salt`, `dens_flux` (`ice_oce_coupling.F90`), surface BC `+ S_ref·water_flux·is_nonlinfs` (`oce_ale_tracer.F90`), KPP surface buoyancy flux, `pressure_bv` screening + blow-up bounds, clipping 3/45 psu shifted, `salt`/`sss` stream offsets, restart detection `global max > 20 psu` ⇒ convert `values/valuesAB/valuesold`, refuse-to-start guard for unsupported features (`fesom_module.F90`, `write_step_info.F90`) |
| `precond_variant`: code default −1 = auto (1 in SP, 0 in DP); **shipped `namelist.dyn` = 1** | `MOD_DYN.F90`, `oce_setup_step.F90`, `oce_ale_ssh` preconditioner | variant 1 = `−a_ri/(a_rr·a_ii)` |

### Our tree

**M8 source diff vs its base** (`git diff 1df683b m8-precision -- src/`): 25 files, +762/−218.

| class | files | treatment |
|---|---|---|
| M8 touched, m14 did not (14) | `fesom_bulk.cpp`, `fesom_cvmix_tke.hpp`, `fesom_eos.{cpp,h}`, `fesom_field.hpp`, `fesom_halo.cpp`, `fesom_ice_coupling.cpp`, `fesom_io_stream.cpp`, `fesom_jra55.{cpp,h}`, `fesom_kpp.cpp`, `fesom_nc_real.h`, `fesom_sss_runoff.cpp`, `fesom_types.h` | apply July's per-file diff (`git diff 1df683b m8-precision -- <file> | git apply`), then review each hunk against the conformance table |
| both touched (11) | `fesom_halo_device.{cpp,hpp}`, `fesom_ice_evpwide.cpp`, `fesom_io.cpp`, `fesom_main.cpp`, `fesom_mesh.cpp`, `fesom_phc.cpp`, `fesom_ssh.{cpp,h}`, `fesom_step.{cpp,h}` | re-sweep by hand on m14's version, July's diff as checklist |
| new on m14 (5) | `fesom_io_restart.{cpp,h}`, `fesom_ssh_dump.h`, `fesom_ssh_se.{cpp,h}` | fresh sweep |

Raw-`double` / `MPI_DOUBLE` counts on m14 (largest first): `fesom_ssh.cpp` 219/35 (4406 lines),
`fesom_ssh_se.cpp` 137/30, `fesom_main.cpp` 92/3, `fesom_ice_evpwide.cpp` 55/17, `fesom_phc.cpp` 50/2,
`fesom_halo_device.cpp` 45/13, `fesom_ice_maevp.cpp` 27/1, `fesom_eos.cpp` 27/0, `fesom_kpp.cpp` 23/0,
`fesom_calendar.cpp` 21/0, `fesom_ice.cpp` 20/2, `fesom_step.cpp` 19/0, `fesom_io.cpp` 9/8,
`fesom_mesh.cpp` 4/6, `fesom_io_restart.cpp` 4/3. Full table: `~/tmp_claude/mp_facts.txt`.

**Source facts that shape tasks (review-verified):**
- `cg_dot` already returns `real_t` with a `real_t` reduction accumulator (`fesom_ssh.cpp:601-607`) and
  `rtol = soltol·sqrt(s0/N_global)` is `real_t` (`:449`, `:2781`, `:3049`, `:3509`) — the plain-CG scalar
  chain flips to float **by itself** the moment `real_t` is float. The dangerous failure mode is
  **false convergence** (fewer iterations, wrong solve), not more iterations. An `[ssh-verify]`
  true-residual instrument (`true= rec= rtol= gap=`) already exists for the M10 solvers (`:2915`,
  `:3172`, `:3578`) but **not** for plain `cg` / CGPIPE.
- `ssh_sympre_build` is called at **four** sites: `:2724` (cg2), `:2988` (pipecg), `:3627` (oati) —
  gated on `ssh_sympre_on()` — and **`:3428` unconditionally for `pcsi`**, which also reads
  `g_sympre.pr_h.data()` at `:3293` and `g_sympre.pr_d` at `:3450`. `cgpipe_ship_pr` is a global
  defaulting to `NULL` (`:697`); the NULL fallback ships `S->pr_values` (`:755`).
- The `[m14]` knob summary (`fesom_main.cpp:283-324`) scans `environ` against `kPrefix[]` (`:290-302`),
  which lists **no** `FESOM_SSH_PRECOND`, `FESOM_SALT_ANOMALY` or `FESOM_MP_*`; with nothing set it prints
  "no M14 knobs active — default path (… certifies against main)".
- JRA55 forcing is opt-in: `argv[7] jra55_year`, default 0 = analytical (`fesom_main.cpp:351`). A pi
  run **never** executes `fesom_jra55.*`, `fesom_bulk.cpp` or the SSS-restoring path.
- det fill (`fesom_phc.cpp:495-575`): relaxes to `FESOM_IC_EXTRAP_TOL` (default 1e-3) with a 20 000-sweep
  cap that prints and proceeds; **two `MPI_Allreduce(…, MPI_DOUBLE, MPI_MAX)` over `real_t` scalars**
  (`:310`, `:555`) — SP1 stack-smash sites.
- `fesom_io_restart.cpp`: `MPI_Gatherv/Scatterv` of `real_t` matrix values under `MPI_DOUBLE` (`:334`,
  `:421`) — flip; `header_d[2]` Bcast (`:768`) is genuinely double — stays; `stiff_values` is stored
  `NC_DOUBLE` (`:576`, `:838`) because the ALE matrix is the integral of the run.
- `fesom_update_stiff_mat_ale_kk` is a **device** `parallel_for` (`fesom_ssh.cpp:2013-2031`) ⇒ the
  stiffness shadow is a whole-field device `dbl_t` array (M8's `FieldT<dbl_t> values_dbl_fld`).
- `scripts/m14_collect.py` and `scripts/m14_zombie_check.py` **do not exist**; the collector is an
  inline python block in `jobs/job_m14_ladder_cpu:171` and the zombie check is inline at `:143`.
  `jobs/job_m14_gate_serial`: `M14_BASELINE` is a **snapshot directory** (default
  `port2/m6_baseline_serial`), `M14_EXPECT` defaults to `differ`. `scripts/diff_snap.py` is
  `np.array_equal`, zero tolerance — no relative mode.
- **Reference binaries:** `port2/m14/bin/i3` = HEAD `fe01a2bf` (M14 Phase A, 2026-08-15);
  `port2/m14/bin/i4` = commit `4436583` (post-EVPWIDE-staging-fix `94877a9`, 2026-08-16). The M16 base
  `d4a9fe0` differs from `4436583` by **8 src files, +1477 lines** (restart I/O `85c7aae`/`a0b474b`,
  SE-wide contract + KPERIOD knob `bde2469`). **Neither is the base.** A `ref0` binary built from
  unmodified `d4a9fe0` is the oracle; `i4` links it to M14's certified chain (Task P2).
- The M8 instruments `FESOM_MP_NANSCAN` / `FESOM_MP_TRACE_NODE` / `FESOM_MP_CONSERV` live in
  `~/port_kokkos_mp/src/{fesom_bulk,fesom_step,fesom_main}.cpp`.

**Salinity consumer sites on m14** (for the anomaly knob; inventory `~/tmp_claude/salt_precond_sites.txt`):
`fesom_eos.cpp` (densityJM host `:52` + device `:114`, `sw_alpha_beta` `:880-1006`),
`fesom_ice_coupling.cpp` (42 hits: `S_oc`, `rsss`, `relax_salt`, `dens_flux`), `fesom_sss_runoff.cpp`
(`virtual_salt` `:398-432`, `relax_salt` from `Ssurf`), `fesom_tracer_diff.cpp` surface BC `:77-83` /
`:392-394` (needs `+ S_ref·water_flux·is_nonlinfs`), `fesom_kpp.cpp` buoyancy flux `:1385/:1577`,
`fesom_main.cpp` step-diag S bounds `:1485-1561` + `Ssurf` `:574`, `fesom_io.cpp` `salt`/`sss` streams
`:1126/:1129`, `fesom_io_restart.cpp` `salt/salt_AB/salt_M1` `:116-121`, `fesom_phc.cpp` `insitu2pot`
`:773` (convert **after** it), `fesom_ice_thermo.cpp` (via `S_oc`). `fesom_gm.cpp` uses `sw_beta·∇S`
— a gradient, unaffected. No salinity sites in `fesom_ssh_se.cpp`.

**Preconditioner on m14:** CSR fill in `fesom_ssh_preconditioner` (`fesom_ssh.cpp:245-291`,
`pr_values` set once, Fortran `lfirst` precedent). `FESOM_SSH_SYMPRE` (`:2436-2533`, default 1 for
non-cg solvers) rebuilds `D^{-1/2} C D^{-1/2}` from the **variant-0** `pr_values`; applied on top of
variant 1 it would destroy the symmetry. The symmetry-defect computation at `:2513-2528` is reusable
as a gate.

**M14 harness to reuse** (`~/port_kokkos_int`): `jobs/job_m14_ladder_{cpu,gpu}` (both arms in one
allocation, warm-up leg discarded, inline zombie check + collector, `BIN=` pinned, `WSPLIT=`
mandatory), `jobs/job_m14_gate_serial`, `jobs/job_m14_gate_cuda` + `jobs/job_m14_cudarepro2` (3
self-control legs), `scripts/m14_coverage.py`, `scripts/m14_scaling_figs.py`.

**M8 scripts to carry over** (`~/port_kokkos_mp/scripts/`): `mp_assert_banner.sh`, `mp_cuda_gate.py`
(CUDA noise-envelope gate: per-field diff ≤ max(10× same-binary rerun noise, 1e-13)),
`mp_divergence_curve.py` (cross-dtype relative comparison — the solution-class tool),
`mp_envelope_verdict.py`, `mp_conserv_drift.py`, `mp_gate4_verdict.py`.

---

## Decisions taken (brainstorm 2026-09-02 — do not relitigate)

| # | decision |
|---|---|
| D1 | Track = **M16**; worktree `~/port_kokkos_sp`, branch `m16-precision` off `m14-integrate` `d4a9fe0`. Merge back into m14 by a **real merge** when green; m14 → `main` is the user's call. **Ask before every push.** |
| D2 | `FESOM_SSH_PRECOND` **default = 1** (mirrors upstream `main`). The announce line prints on **every** run, including 0. Explicit `FESOM_SSH_PRECOND=0` reproduces `main`. **SYMPRE rule:** with variant ∈ {1,2,3} (symmetric by construction) the SYMPRE build is skipped at **all four** build sites and both `pcsi` reads use `S->pr_values`; the `pcsi` self-adjoint check accepts the variant; SYMPRE is meaningful only with variant 0. Variants 0–4 and the lab tool stay. All M10 CA-solver numbers re-base under variant 1. |
| D3 | Revert gate amended from M14's: **all new knobs unset plus `FESOM_SSH_PRECOND=0` ⇒ bitwise to the `ref0` oracle on Serial** (`ref0` = unmodified `d4a9fe0`, itself proven equal to `i4` knobs-off, which M14's G1 certified against `main`); **on CUDA within the binary's own run-to-run noise (3 control legs).** Every byte-gate script exports `FESOM_SSH_PRECOND=0` itself. |
| D4 | **Approach A — transplant onto m14**, not a merge or rebase of `m8-precision`. Every slice byte-gated at FP64 against `ref0` (Serial = byte oracle; CUDA = envelope). `m8-precision` is a reference only. |
| D5 | **Exact port; placement by a #940 conformance table** in `PRECISION_ISLANDS.md` (new column naming the Fortran line mirrored). Five classes — see Technical details. Two invariants regardless of class: `MPI_DOUBLE` only over `dbl_t` storage (grep-enforced); accumulation ledger for every `state += small` site. |
| D6 | CMake option name = upstream's **`USE_SINGLE_PRECISION=ON|OFF`** (no alias). Banner prints precision, kind, storage bits, digits, epsilon; plus the once-per-run I/O precision report. |
| D7 | **Salt anomaly = exact port of #986**, line by line against the Fortran, behind `FESOM_SALT_ANOMALY=1` (default off; off = `+0.0`, bit-identical). A numeric value other than 1 sets a test reference (`FESOM_SALT_ANOMALY=10`) — measurement-only, used by the invariance gate. |
| D8 | Scope order: **M14 best-arm recipe first** (`det` + `WSPLIT=1` + `FESOM_SPEED=1` + `FESOM_WHICH_EVP=1` + `SPEED_EVPWIDE=8` + `LEAN=1` + `SSH_MODE=se` (+`SE_WIDE` on GPU) + `oati` on dars/NG5 CPU + `PRECOND=1`); second wave `pipecg`/`pcsi`/`cg2`, restart round-trip, remaining lab knobs. |
| D9 | Gate ladder 0→4 (below); 63-yr hindcast later, user decision. Failure protocol: promote **one** island at a time, log the failing signature and the pinned-pair give-back. |
| D10 | Read-only `/work` staging: **now** = code + Serial builds (+ CUDA compile if the login node compiles it) + pi-mesh and login-node CORE2 gates with outputs in `~/m16_scratch` (explicit, time-boxed exception to "never `$HOME`"; deleted when `/work` returns). **Later** = SLURM, frozen bins under `/work/ab0995/a270088/port2/m16/bin` (sha-named, never committed), every campaign leg. |
| D11 | **One harness:** extend `jobs/job_m14_ladder_{cpu,gpu}` with a precision axis instead of reviving `jobs/job_mp_*`; extract the inline collector/zombie logic into scripts. Carry over the M8 verdict scripts. |
| D12 | Docs: registry moved + conformance column; lessons SP1–SP11 folded into `docs/KOKKOS_PORTING_LESSONS.md` (check L-number collisions, m14 reached ≥L122); campaign doc `docs/MIXED_PRECISION_M16.md`. Nothing to report upstream (both July findings fixed there independently). |

---

## Development approach

This is a port-plus-measurement campaign. **"Tests" means gates**: byte gates (`scripts/diff_snap.py`,
zero tolerance, same dtype, **snapshot files, never logs** — the P3 announce line changes every log),
envelope gates (CUDA), liveness gates (a knob must change something and announce itself), screens
(3000 steps, finite diagnostics, non-vacuous solver trace), and pinned timing pairs. **Every task ends
with one, and a gate must be able to fail** — a gate that passes on absent code is a defect of the
plan. No task starts until the previous task's gate is green. Each sweep slice keeps the FP64 build
**bit-identical** so any breakage bisects to one slice.

Standing rules, non-negotiable, on every job:

- **`BIN=` pinned** on every job (multi-srun re-execs at each `srun`)
- ladder `dt` per mesh: CORE2 1800 · fArc 900 · dars 120 · NG5 180
- **`WSPLIT=1` on both arms** for every fArc/dars/NG5 cold start (rule 0.41 — the "random CG NaN")
- `-C a100_80` on every GPU absolute; **16 GPU nodes is the cap** — ask before 32
- `snap_every=-1` at ≥4096 ranks; `FESOM_SE_M` per mesh (fArc 90 · dars 20 · CORE2 ≥50)
- Serial speed levers need `FESOM_SPEED_FORCE_SERIAL=1` on **both** arms; CUDA must **not** carry it
- `env_cuda.sh` is a **build** rule (RPATH) — `ldd` every new CUDA binary
- **fArc ≥128 ranks: never the E.T1 UCX proto package** (reproducible hang); the M14 fArc ladders ran
  without it — keep it that way
- a cheap gate must look cheap (`-t 00:06:00` class); budget the det fill (≈7 min at NG5/64 ranks) as
  setup outside the timing window
- **equal leg count per arm at a point**; the noisier arm sets it; estimator = min over all legs; if
  two pairs of one point disagree by more than the arm spread, **add legs, do not choose**
- every byte gate exports `FESOM_SSH_PRECOND=0` (D3) and compares snapshots, not logs
- figures, if any: read `scripts/m7_scaling_figs.py` first (standing user rule)
- output to `/work`, never `$HOME` — **except D10 while `/work` is read-only**
- **never commit binaries**; frozen bins → `/work/ab0995/a270088/port2/m16/bin/<tag>/`, shas in the docs
- **ask before every push**; commit locally freely
- 🔴 **no model runs on the login node** beyond the seconds-long pi smokes: every CORE2 leg, oracle, or multi-rank run is a SLURM job (`jobs/job_m16_gate_serial`, 8 ranks, `-p compute`); the "[NOW: login CORE2]" tags below were written for the read-only-`/work` day and are superseded (user, 2026-09-07)
- update this plan when scope changes (`[x]` immediately, ➕ discovered, ⚠️ blocker)

---

## Testing strategy — the gate ladder (D9)

| gate | what | points / configs | bar |
|---|---|---|---|
| **G0 — FP64 inert** | double build, all new knobs unset, `FESOM_SSH_PRECOND=0` exported by the script | Serial: pi np1/np2, CORE2 np8/np128 · **five configs**: default (EVP+KPP), mEVP, zstar, TKE, SE · 20 steps, snapshots at 1, 2, 20 (Z7) · **plus the arms later gates use**: SE-wide, EVPWIDE-lean, each M10 solver · CUDA: np8/np16, 3 self-control legs | Serial: **zero differing bytes** vs `ref0` · CUDA: within the binary's own rerun noise |
| **G1 — SP runs** | single build | pi np1/np2 both backends · CORE2 np1 **60 steps dt 1800** with JRA (past the first forcing-record boundary — July's 9.4-model-hour death, SP3) | rc 0, no non-finite, banner `SINGLE` asserted, NANSCAN's own ARMED banner present |
| **G2 — prize sizing** | 300-step DP/SP pairs, **one allocation**, equal leg counts, min over legs | CORE2 4N + 16N GPU · fArc + dars CPU at their knees · NG5 16N GPU · device memory per GPU · the CVMix-double give-back pair (Task B7 define) | July priors: 1.3–1.6× speed, 0.51× memory, CG +1 iteration; report, no pass/fail |
| **G3 — knobs at SP** | the D8 recipe, knob by knob | per-knob **liveness signal** at SP · SE wide-halo drift 0.0 every step · NaN-zombie detector on every leg · **true-residual gap** (`[ssh-verify]`) SP vs DP for `cg`/CGPIPE/`oati` across a **rank sweep** up to the largest CPU point · **30-day conservation** (`FESOM_MP_CONSERV`) · one DP control leg with the point-slope form compiled in (bounds the class-5 confound) | every signal reproduces; gap bar pre-registered in B3; July: heat gap 0.2 % of the 30-d signal |
| **G4 — stays physical** | 3000-step screens per **mesh × backend** of the recipe (8 cells) + **one 1-yr CORE2 twin** SP vs DP | M5.23 pattern-correlation bar (sst/sss/ssh/a_ice; July: 1.00000/1.00000/1.00000/0.99999) and **SP-vs-Fortran ≡ DP-vs-Fortran to the printed digit** | pass ⇒ option declared usable |
| *(later)* G5 | 63-yr CORE2 hindcast vs 63A/63B | user decision after G4 | Tbar ≤ 0.001 °C, OHC ≤ 2 ZJ, co-track/flatten |

**Salt-anomaly gates (D7):** off ⇒ bit-identical to a build without the knob (DP and SP); **S_ref
invariance** (DP, pi, NOW): solutions at `S_ref ∈ {35, 10}` differ from each other and from off only
by the rounding class — a missed consumer produces an error that scales with the reference value; on,
DP, CORE2 ⇒ solution-class vs off with the expected ~8e-6/step surface residual; on, SP, CORE2 ⇒ 1-day
salt error vs the DP reference, expecting the ~38 % reduction upstream measured.

**Preconditioner gates (D2/D3):** `PRECOND=0` bitwise to `ref0`; `PRECOND=1` differs **and** the CG
iteration count drops by the #984 class (pi 12.3→8.6 mean; CORE2 ~34 %); symmetry-defect ratio
(`:2513-2528` computation) ≈ 0 for variants 1/2/3 and > 0 for 0/4; `pcsi` under variant 1 converges to
the **same solution as `cg` under variant 1** (`mp_divergence_curve.py`, solution class), not merely
runs; variant 4 + `pcsi` ⇒ `rc ≠ 0`; `SYMPRE=1` + variant 1 announces "skipped".

---

## Progress tracking

- `[x]` immediately when done · ➕ discovered tasks · ⚠️ blockers · keep the plan in sync
- tags: **[NOW]** = executable under read-only `/work` · **[BLOCKED on /work]** = wait

---

# Phase P — the #984 preconditioner default **[NOW]**

### Task P1: Commit the M15 diff where it belongs

**Files:** `~/port_kokkos_pre`: `src/fesom_ssh.cpp`, `tools/fesom_ssh_lab.cpp`; untracked
`jobs/job_m15_*`, `scripts/m15_precond_*.py`

- [x] review the uncommitted diff (+120/−4) — it is the **only copy** of the knob
- [x] commit it on `m15-precond` together with the M15 jobs/scripts (one commit: "M15: FESOM_SSH_PRECOND
      variants 0–4 (#984 var 1, M10 sqrt patch, header forms) + A/B jobs + drift figures") → `39d7a30` (2026-09-07)
- [x] gate: `git status` clean in `~/port_kokkos_pre`; `git log -1` shows the commit

### Task P2: Worktree, branch, reference material, **and the `ref0` oracle**

**Files:** Create `~/port_kokkos_sp` (worktree); copy this plan; `docs/reference/upstream_sp/`;
Create `scripts/m16_gate0.sh`

- [x] `git worktree add ~/port_kokkos_sp -b m16-precision m14-integrate` (HEAD must be `d4a9fe0`) ✓ 2026-09-07; Kokkos submodule initialised
- [x] copy the gitignored third-party dirs (`ice_sergey/`, `ssh_sergey/`) if the build needs them — not needed (CMake does not reference them)
- [x] copy this plan into the worktree's `docs/plans/` and commit it there
- [x] `docs/reference/upstream_sp/`: `pr940.diff`, `pr984.diff`, `pr986.diff`, `pr995.diff`, `pr997.diff`,
      `pr940_mech.txt`, plus a `README.md` with merge dates and the file→mechanism table above; keep
      July's snapshot pointer (`~/port_kokkos_mp/docs/reference/pr940/`)
- [x] ~~`mkdir ~/m16_scratch`~~ — **`/work` was writable again on 2026-09-07**, so the D10 exception never fired: everything goes to `/work/ab0995/a270088/port2/m16/` (`gate0/`, `bin/`, `logs/`)
- [x] **`ref0`:** build unmodified `d4a9fe0` Serial (`build-m16-ref0`); run it on pi np1/np2 for the
      five G0 configs **and** the later-gate arms (SE-wide, EVPWIDE-lean with `WHICH_EVP=1`, each M10
      solver) into `…/m16/gate0/ref0/<config>_np<N>/`; these snapshot dirs are the byte oracle for every
      Phase-B slice — built by `build_m16.sh ref0` (refuses unless HEAD is a clean `d4a9fe0`); 11 configs
      (`default mevp zstar tke se sewide evpwlean cg2 pipecg oati pcsi`) × np1/np2, 21 snapshots each
      (dt 100, 20 steps, snap every step); `evpwlean` is np≥2 only (M9 FATAL at np1 by design)
- [x] `scripts/m16_gate0.sh <build> <config|five|all> [np]`: runs the config with `FESOM_SSH_PRECOND=0` exported, all
      other new knobs unset, into `…/m16/gate0/<build>/<config>_np<N>/`, then `diff_snap.py` vs `ref0`;
      exits non-zero on any difference or on a non-zero run rc (the M14 gate once passed two segfaults);
      accepts an absolute binary path too (used for the `i4` link check)
- [x] gate: `git status` clean; `src/fesom_ssh_se.cpp` and `src/fesom_io_restart.cpp` present;
      **`i4` (frozen `8804edd9`) is bit-identical to `ref0` on the five G0 configs at np1 AND np2** (2026-09-07);
      the gate can fail: `default` vs `zstar` oracles differ (rc 1); `se` vs `sewide` at np2 identical
      (the wide halo's EXACT claim holds on pi too)

### Task P3: Cherry-pick the knob; default 1; announce always; SYMPRE rule; knob summary

**Files:** Modify `src/fesom_ssh.cpp`, `src/fesom_main.cpp`, `tools/fesom_ssh_lab.cpp`

- [x] `git cherry-pick` P1's commit onto `m16-precision` (expect a clean pick — m15 == m10 HEAD, which
      m14 contains)
- [x] `ssh_precond_variant()` (now exported as `fesom_ssh_precond_variant()`): default `c = 1`; keep 0–4 and the abort on other values
- [x] the `[ssh-precond]` announce line prints on **every** run, including variant 0, naming the formula
      (e.g. `variant 1 = -a/(d_i d_j) (FESOM/fesom2#984, symmetric)`)
- [x] **SYMPRE rule, at every site:** `ssh_sympre_on()` returns false when variant ∈ {1,2,3}, with a
      one-time `[ssh-sympre] skipped: FESOM_SSH_PRECOND=<v> is symmetric by construction` if the user
      set `FESOM_SSH_SYMPRE=1`; the three gated build sites (`:2724`, `:2988`, `:3627`) follow; the
      **unconditional `pcsi` build at `:3428`** becomes conditional on variant 0, and the two `pcsi` reads
      (`:3293` host, `:3450` device) select `S->pr_values` / `S->pr_values_fld.d()` when SYMPRE is not
      built; the `pcsi` `FESOM_CHECK` at `:2575` accepts variant ∈ {1,2,3}; `cgpipe_ship_pr` stays `NULL`
      (`:697`) so the ring-1 rows ship `S->pr_values` (`:755`); the teardown at `:3900-3901` tolerates
      never-built
- [x] variant 4 (header form, asymmetric) + `pcsi` ⇒ refuse (`FESOM_CHECK`, same class as `:2575`)
- [x] `[m14]` knob summary (`fesom_main.cpp:290-302`): add `FESOM_SSH_PRECOND`, `FESOM_SALT_ANOMALY`,
      `FESOM_MP_`, `USE_SINGLE_PRECISION` (as a build fact); the "no knobs active" branch prints the
      resolved precond variant and the precision banner line so it can no longer claim "default path
      certified against main" while variant 1 is in effect
- [x] lab tool: same default and announce
- [x] gate (Serial, pi) — `scripts/m16_p3_gate.sh`, 2026-09-07: stages 1–6 PASS (gate0 np1/np2 bitwise with PRECOND=0; default differs in 235 fields; defect ratio 3.9e-14 for 1/2/3 vs 0.299 for 0/4 on pi, **0.638 on CORE2 = the M10 F1 number**; SYMPRE skipped/BUILT as specified; pcsi+4 rc=1; knob summary names the variant in both branches). CORE2 np1 login (PHC+JRA 1958, 20 steps): **CG mean iterations 133.2 → 81.3 (−39.0 %)**, pcsi 146.8 → 97.5 (−33.6 %); pcsi-vs-cg step-20 distance under v1 is the v0 class for eta/S/T/u/v (Av/Kv are KPP threshold fields, excluded; **controls settle it**: cg2-vs-cg and oati-vs-cg under v1 show the SAME Av/Kv jumps at the same nodes (9.8e-2/9.7e-2 @352202/160815) — mixed-layer threshold sensitivity of this 20-step CORE2 np1 config, not a pcsi property; pcsi's eta/T/S/u/v are within 3× of the cg2 control; `[ssh-verify]` true residual below rtol on every solve for pcsi/cg2/oati; a pcsi rerun is bit-identical (Serial repro)) : `m16_gate0.sh` (which exports `PRECOND=0`) **bit-identical**; default run
      differs **and** mean CG iterations drop by the #984 class; symmetry-defect assertion (reuse
      `:2513-2528`): ≈ 0 for 1/2/3, > 0 for 0/4; `FESOM_SSH_SOLVER=pcsi` under the default converges to
      the same solution class as `cg` under the default (`mp_divergence_curve.py`); variant 4 + `pcsi`
      ⇒ `rc ≠ 0`; `FESOM_SSH_SYMPRE=1` under the default prints "skipped"; knob summary names the variant
- [x] ➕ one paragraph in `docs/SSH_SOLVERS_M10.md`: every M10 whole-step number is a variant-0/SYMPRE
      number and re-bases under variant 1 (the M15 memory's prediction); no re-measurement here

---

# Phase A — precision scaffolding and the conformance table **[NOW]**

### Task A1: Type switch, CMake option, banner, assert script

**Files:** Modify `src/fesom_types.h`, `CMakeLists.txt`, `src/fesom_main.cpp`; Create
`scripts/mp_assert_banner.sh` (from M8)

- [x] `fesom_types.h`: apply July's diff — `#if defined(FESOM_SINGLE_PRECISION)` → `real_t = float`,
      `FESOM_MPI_REAL = MPI_FLOAT` else `double`/`MPI_DOUBLE`; `typedef double dbl_t;` with the comment
      "deliberate FP64 island — never flips; every use is a row in PRECISION_ISLANDS.md"
- [x] `CMakeLists.txt` (directory-scoped `add_compile_definitions` so every TU sees one `real_t`): `option(USE_SINGLE_PRECISION "Build the model state in single precision (real_t=float)" OFF)`
      → `target_compile_definitions(... FESOM_SINGLE_PRECISION)`; `message(STATUS ...)`; **no alias**
      (D6); reject any non-boolean value
- [x] banner (rank 0, after MPI init, before Kokkos init) in the upstream shape:
      `[fesom_port] PRECISION: SINGLE|DOUBLE  real_t=float|double  storage=32|64 bits  digits=6|15  epsilon=1.19e-07|2.22e-16`
- [ ] `scripts/mp_assert_banner.sh <log> <SINGLE|DOUBLE>` (July's), exercised on np1 and np2 logs
- [x] gate (2026-09-07): FP64 Serial build, `m16_gate0.sh` on pi np1 + np2 **bit-identical**; SP configure succeeds
      **and** `FESOM_SINGLE_PRECISION` appears in every TU of `compile_commands.json` (July: 43/43) — **54/54 model TUs** carry it, the 20 without are `externals/kokkos` + generated;
      `USE_SINGLE_PRECISION=maybe` is rejected (FATAL_ERROR); gate0 all 11 configs np1+np2 bitwise; banner DOUBLE asserted on np1 and np2 logs

### Task A2: The conformance table — classify before sweeping

**Files:** Create `docs/PRECISION_ISLANDS.md` (from M8, restructured); Modify `docs/reference/upstream_sp/README.md`

- [x] move July's registry over; add the column **"upstream (#940) placement / Fortran line"** and the
      class tag (1–5, Technical details)
- [x] classify **every** existing row and every `dbl_t` July introduced; the class-3 rows (CG scalar
      chain + dot accumulators + Allreduce scalars, mesh metrics precompute, PHC init path, min/max
      diagnostics, mesh volume sums, calendar seconds-in-day) are marked **"flip to real_t in Phase B"**
- [x] add rows for the m14-only code (class 4): SE barotropic state and its `H0e`/wide-halo buffers,
      EVPWIDE lean buffers, CGPIPE/CGPOLY payloads and eigenbounds, M10 CA-solver recurrences (`cg2`,
      `pipecg`, `oati`, `pcsi` Lanczos/Chebyshev), det IC fill + `FESOM_IC_EXTRAP_TOL`, restart I/O staging,
      dumps/verify twins
- [x] add the class-2 row (CVMix TKE → `dbl_t` inside the kernel), the class-5 row (point-slope forcing,
      SP-only), and the **stiffness shadow as a whole-field device `dbl_t` promotion** (memory give-back
      = one nnz-sized double array, measured at G2)
- [x] carry the **accumulation ledger** over and extend it with every `+=` site in `fesom_ssh_se.cpp`
      (the barotropic subcycle: 20–90 substeps per step is the stiffness-drift class) and
      `fesom_io_restart.cpp`; **the ledger is grep-generated** (`+=` over `real_t` state), not hand-written
- [x] gate (2026-09-07): `docs/PRECISION_ISLANDS.md` committed before Phase B; ledger = `scripts/m16_accum_ledger.py` (215 indexed `+=`/`-=` sites, embedded verbatim, count printed by `--count`); the inventory's MPI/disk pairing audit lists every `MPI_DOUBLE`-over-`real_t`, `nc_*_double`-over-`real_t`, hard-`double` device view and `sizeof(double)` accounting line by line; every `dbl_t` that will exist after Phase B has a row; every row has a class; every class-1,
      -2 and -5 row has a **non-empty Fortran-line cell**; the ledger's site count equals the grep count;
      the table is committed **before** the first Phase-B slice (placement is pre-registered)

---

➕ **Finding (2026-09-07, P2 oracle):** the UNMODIFIED `d4a9fe0` Serial binary segfaults with the EVPWIDE-lean arm on **CORE2 at np2** on the login node (`ref0/evpwlean_np2`, rc 139, after the extended zone is built: K=8 R=8 ext-nodes 3286); pi np2 and every other CORE2 config at np2 run. M14 only ever ran this arm at ≥8 ranks. Not an M16 defect; the CORE2 evpwlean oracle will be taken at np8 under SLURM. The backtrace puts the crash in `fesom_jra55_step()` (not the ice code): the JRA per-node forcing arrays are sized without the extended-zone nodes, or the zone's node count differs at 2 ranks. To be understood before the E-phase ladders.

# Phase B — the sweep, slice by slice, each slice FP64 bit-identical **[NOW]** (CUDA envelope legs **[BLOCKED on /work]**)

**Per-slice gate, every task below:** FP64 Serial rebuild → `scripts/m16_gate0.sh` on pi np1 + np2
for the configs the slice touches (it exports `FESOM_SSH_PRECOND=0` and compares snapshots vs `ref0`).
**Slices whose code the pi run never executes** (JRA forcing, bulk formulae, SSS restoring, PHC/det
init) are gated on a **CORE2 np1/np2 login run with `jra55_year=1958` and the PHC path** (20 steps
dt 1800, snapshots @1/2/20, ~0.5 GB per snapshot set in `~/m16_scratch`; `ref0` run the same way).
Slices touching device code additionally get the CUDA envelope leg (`scripts/mp_cuda_gate.py`) once
`/work` is back — list them here as they land (➕).

### Task B1: The 14 M8-only files, by per-file diff

**Files:** Modify `src/fesom_bulk.cpp`, `src/fesom_cvmix_tke.hpp`, `src/fesom_eos.{cpp,h}`,
`src/fesom_field.hpp`, `src/fesom_halo.cpp`, `src/fesom_ice_coupling.cpp`, `src/fesom_io_stream.cpp`,
`src/fesom_jra55.{cpp,h}`, `src/fesom_kpp.cpp`, `src/fesom_nc_real.h` (create), `src/fesom_sss_runoff.cpp`

- [x] for each file: `git -C ~/port_kokkos diff 1df683b m8-precision -- src/<file> | git apply`; on a
      rejected hunk, apply by hand — do not skip silently
- [x] review each applied hunk — no class-3 hunk in this set (the min/max reduces and mesh sums live in B2/B6c files); every hunk is class 1 or type-neutral; ➕ the `fesom_bulk.cpp` hunk is the M8 NaN-scan forensic hook and needs `fesom_mp_nanscan_enabled()` from `fesom_step.h` → applied in Task B6d, not here against A2: class-1 hunks stay (e.g. `integrate_nod_2D` accumulators
      `dbl_t` = upstream `WP_full`); class-3 hunks are reverted to `real_t` **in this task** (e.g. any
      min/max reduce or mesh-area sum July promoted); note each reversal in the registry row
- [x] `fesom_io_stream.cpp`: also the **once-per-run I/O precision report** (D6): per stream the on-disk
      type and the accumulator kind, one summary at the end of stream definition (upstream
      `note_output_precision`)
- [x] `fesom_cvmix_tke.hpp`: leave July's `real_t` form here; the class-2 `dbl_t` kernel is Task B7
- [x] `fesom_jra55.*`: keep July's `FieldT<dbl_t>` time chain (class 1); the point-slope form is Task B6b
- [x] gate (2026-09-07; B1 and B2 gated TOGETHER because their edits overlap in the two I/O files — attribution by bisection if the combined gate had failed): pi np1 all configs bitwise on the B1+B2 binary; np2 + CORE2 legs re-run on the frozen copy `…/m16/bin/b12/fesom_port` (md5 `7f1ce202`) — see the B2 gate line. ⚠️ Two process lessons paid here: (i) a FAILED build left the previous binary in place and the gate 'passed' on it — `build_m16.sh` now deletes the binary before building; (ii) editing `m16_gate0.sh` while a background gate was executing it corrupted that run (bash reads scripts incrementally) — never edit a script that is running. Original spec: pi per-slice gate after each of the three file groups (eos+kpp / halo+coupling+runoff /
      io_stream+nc_real+jra55+bulk+field); **group 2 and group 3 additionally on the CORE2+JRA login
      gate** (pi never runs them); the I/O report line present in the DP log

### Task B2: Halo, mesh, I/O

**Files:** Modify `src/fesom_halo_device.{cpp,hpp}`, `src/fesom_mesh.cpp`, `src/fesom_io.cpp`

- [x] `fesom_halo_device`: host+device pack buffers and pointers → `real_t`, `MPI_DOUBLE` → `FESOM_MPI_REAL`
      on every payload (13 sites); timing/selfcheck reductions stay `dbl_t` (diagnostic, documented);
      `sizeof(real_t)` in the profiler byte counts; `FESOM_HALO_STAGE=1` pinned-host path included
- [x] `fesom_mesh.cpp`: Bcasts → `FESOM_MPI_REAL` (6 sites); metrics **computed in `real_t`** (class 3 —
      upstream computes areas/gradients in WP from WP coordinates); `ocean_area` / volume sums →
      `real_t` + `FESOM_MPI_REAL` (upstream `vol_n/vol_e` are `MPI_WP`)
- [x] `fesom_io.cpp`: gathers templated on the element type (`gather_node_T<T>` with `fesom_mpi_of<T>()`, so the MPI type follows the storage type by construction); snapshot writer staged through double (July's hunks by hand) ➕ **discovered, exact port:** the time-mean accumulators are `dbl_t` (`fesom_io_acc_t`; upstream's 8-byte streams accumulate real64) — host + device accumulators, 34 resolvers, flush divides in double, `gather_*_d` (`MPI_DOUBLE` over `dbl_t`), written with `nc_put_vara_double` directly; the registry §1 row now points at this
- [x] gate (B1+B2 together, binary `b12` md5 `7f1ce202`): pi np1 all 12 configs bitwise ✓; pi np2 all bitwise ✓; **CORE2 np8 SLURM 2026-09-07: all 14 configs BYTE-IDENTICAL to ref0 (jobs 27286999 b12 / 27287001 b6 / 27287827 b3+verify / 27287828 b7+verify, GATE 0 PASS)**. Original spec: per-slice gate; np2 leg exercises the swept exchange; `m16_mpi_invariant.sh` (B8) run early
      on these files shows only `dbl_t` storage under `MPI_DOUBLE`

### Task B3: The SSH solver file (`fesom_ssh.{cpp,h}`, 4406 lines, 219 doubles, 35 MPI)

**Files:** Modify `src/fesom_ssh.cpp`, `src/fesom_ssh.h`, `src/fesom_ssh_dump.h`

- [x] vectors, SpMV, `pr_values`, halo `Isend/Irecv` and their `.dbls` buffers → `real_t` + `FESOM_MPI_REAL`
      (July's 18-site split as the checklist, extended to the M10 solvers)
- [x] **class 3 flip:** CG scalar chain (residual, rtol, α, β), dot-product accumulators and the
      `Allreduce` scalars → `real_t` + `FESOM_MPI_REAL` (upstream's CG is entirely WP; `cg_dot` and `rtol`
      already flip by themselves); the registry rows say "flipped 2026-09; re-earn on evidence"
- [x] (already present for plain `cg` in `fesom_ssh_solve_cg_kk` since M10; retyped to `dbl_t` at all four sites, SpMV in `dbl_t` over the `real_t` matrix) **the detector for the flip's real failure mode (false convergence):** extend the existing
      `[ssh-verify]` true-residual instrument (`:2915/:3172/:3578`, M10 solvers) to plain `cg` and
      CGPIPE — `‖b−Ax‖₂/‖b‖₂` accumulated in `dbl_t`, env-gated (`FESOM_SSH_VERIFY=1`), printed at solve
      exit with the recurrence residual and the gap; **pre-register the G3 bar here before any SP leg**:
      gap(SP) ≤ 10 × gap(DP) at every rank count of the sweep, and no solve whose true residual exceeds
      `rtol` by more than the DP run's own worst
- [x] class 4: CGPIPE/CGPOLY payloads, eigenbound power iteration, Chebyshev recurrence → `real_t`; M10
      `cg2`/`pipecg`/`oati`/`pcsi` recurrences, Lanczos, `PCSI_EIG` → `real_t`; the NaN-blind stall-guard
      fix must survive (`resid >= rtol` false for NaN)
- [x] `fesom_ssh_dump.h`: dump format stays double on disk; the writer stages `real_t` arrays through double under SP (the July `static_assert(sizeof(real_t)==8)` is gone); the lab reader is Task E5's
- [x] gate (2026-09-07, binary `b3` md5 `144410db`): pi np1 + np2 bitwise on all 14 configs incl. `cgpipe`/`cgpoly` (`FESOM_SPEED_FORCE_SERIAL=1`); `test_field` + `test_ssh_solvers` pass; CORE2 np8 SLURM gate + `verify` leg → see the job results line. Original: per-slice gate on **all** of `cg`, `cg2`, `pipecg`, `oati`, `pcsi` (`FESOM_SSH_SOLVER=`),
      CGPIPE + CGPOLY with `FORCE_SERIAL=1`, selfchecks 0.000e+00, iteration counts identical to `ref0`;
      `[ssh-verify]` on plain `cg` prints a DP gap in the M10 solvers' class; `test_ssh_solvers` passes

### Task B3b: The stiffness shadow (SP-only new code — its own task, its own gate)

**Files:** Modify `src/fesom_ssh.cpp` (`fesom_ssh_preconditioner`, `fesom_update_stiff_mat_ale_kk`
`:2013-2031`), `src/fesom_ssh.h`, `src/fesom_io_restart.cpp` (`:229-243`, `:576`, `:838`)

- [x] (2026-09-07, `fesom_ssh.{h,cpp}`) **upstream #997 semantics**: `values_full` = whole-field device `FieldT<dbl_t>` sized to the local
      nnz; compiled only under `FESOM_SINGLE_PRECISION`; **seeded from `values` on the first
      `update_stiff_mat_ale` call** (not in the preconditioner build); every increment `atomic_add` in
      `dbl_t`; `values = real_t(values_full)` once per update; the CG SpMV keeps float bandwidth
- [x] (runtime knob `FESOM_DIAG_STIFF_DRIFT=N` = print every N updates; the port has no `logfile_outfreq`) `FESOM_DIAG_STIFF_DRIFT=1` (upstream's instrument, **not optional** — it is the only gate this
      slice can have): a WP-accumulated twin and the once-per-`logfile_outfreq` print
      `[STIFFDRIFT] relL2(diag)= relL2(offdiag)=`
- [x] (`stiff_pack/gather/scatter<T>` + `rst_mpi_of<T>`; SP writer gathers the shadow, widened `values` under linfs; reader seeds the shadow and rounds into `values`) **restart (declared divergence, Overview b):** in the SP build the restart writer stores
      `values_full` into `stiff_values` (already `NC_DOUBLE`) and the reader seeds `values_full` from it
      and rounds into `values`; the DP build is unchanged; document in the registry row
- [x] **gate PASS 2026-09-07** (frozen `bin/b3b` DP `aa09820b`, `bin/b3bsp` SP `96e120c2`): DP `m16_gate0.sh all` np1 + np2 all 14 configs BYTE-IDENTICAL to ref0; DP restart round-trip bitwise; **SP restart round-trip (zstar np2, write@10 → run to 20) BIT-IDENTICAL** to the straight SP run; `scripts/m16_stiffdrift_gate.sh` SP pi zstar np2 200 steps `DRIFT=20`: relL2(offdiag) 1.89e-7 → 1.25e-6 monotone, relL2(diag) 5e-9 → 7e-8 (10 lines, none zero), DP build prints no `[STIFFDRIFT]` line. Spec: gate (DP): `m16_gate0.sh` on zstar — bit-identical by construction (the code is compiled out);
      **gate (SP, pi zstar, NOW):** with `FESOM_DIAG_STIFF_DRIFT=1` the WP twin's drift is **non-zero and
      growing** over 200 steps while the shadow's working copy tracks `values_full` to one rounding —
      i.e. the instrument can see the defect and the shadow removes it; SP restart at step 10 → run to
      20 **bit-identical** to the straight SP run

### Task B4: Split-explicit SSH (`fesom_ssh_se.{cpp,h}`, new, 137 doubles, 30 MPI)

**Files:** Modify `src/fesom_ssh_se.cpp`, `src/fesom_ssh_se.h`

- [x] barotropic state (reconstruction exchange + coefficient ship → `FESOM_MPI_REAL`; operator assembly, host viscosity path, CFL probe → `real_t`; SE_CHECK conservation sums and every cross-rank audit stay `double` as diagnostics) (η, Ū, Fbt, H0e), AB3-AM4 history, wide-halo reconstruction buffers, the device
      gather/scatter staging (the L121 lean fix) → `real_t`; exchanges → `FESOM_MPI_REAL` (30 sites)
- [x] ledger every `+=` (A2, 3 indexed sites in `fesom_ssh_se.cpp`) in the subcycle (A2) with its increment/state scale; the selfcheck/drift
      diagnostics (`FESOM_SE_CHECK`, `SE_WIDE_SELFCHECK`, drift print) stay `dbl_t` (diagnostic)
- [x] `FESOM_SE_M` CFL probe arithmetic: `real_t` (upstream SE is WP under `-r4`)
- [x] gate (B4+B5 together, `b6`/`b7`; **CORE2 np8 SLURM 2026-09-07: all 14 configs BYTE-IDENTICAL to ref0 (jobs 27286999 b12 / 27287001 b6 / 27287827 b3+verify / 27287828 b7+verify, GATE 0 PASS)**): per-slice gate with `FESOM_SSH_MODE=se` and with `se + FESOM_SE_WIDE=1 FESOM_SE_H0E_XCHG=1
      FESOM_SE_WIDE_RECON=1` (drift 0.0 every step at np2 pi) — both bit-identical to `ref0`'s SE arms

### Task B5: Sea ice (EVP, mEVP, EVPWIDE lean, coupling, thermo, FCT)

**Files:** Modify `src/fesom_ice_evpwide.cpp`, `src/fesom_ice_maevp.cpp`, `src/fesom_ice.cpp`,
`src/fesom_ice_evp.cpp`, `src/fesom_ice_fct.cpp`, `src/fesom_ice_thermo.cpp`

- [x] `fesom_ice_evpwide.cpp` (55/17): the `static_assert(sizeof(real_t)==sizeof(double))` removed; 10 ship `Irecv/Isend` → `FESOM_MPI_REAL`; the 6 `MPI_Reduce` audits stay `double`: wide-halo pack buffers, lean-path staging → `real_t` +
      `FESOM_MPI_REAL`; `EVPWIDE_SELFCHECK`/`SHIPCHK`/`MEVPDIV_SELFCHECK` reductions stay `dbl_t`
- [x] (audited 2026-09-07: every remaining raw `double` in `fesom_ice_maevp/ice/thermo/fct/evp.cpp` is a verify-twin max-diff, a `Kokkos::Max<double>` selfcheck, a profiler tic, or the `MPI_MAXLOC` (double,int) pair — diagnostics; the state code was already precision-generic, as July found) `fesom_ice_maevp.cpp`, `fesom_ice_evp.cpp`, `fesom_ice_fct.cpp`, `fesom_ice_thermo.cpp`: raw
      doubles → `real_t` (July found the pre-M9 ice tree precision-generic; M9 rewrote 2272 lines — audit
      them all)
- [x] gate (with B4; `evpwlean` config in the CORE2 np8 jobs above, BYTE-IDENTICAL): per-slice gate with `FESOM_WHICH_EVP=1 FESOM_SPEED_EVPWIDE=8 FESOM_SPEED_EVPWIDE_LEAN=1
      FESOM_SPEED_FORCE_SERIAL=1` vs `ref0`'s EVPWIDE-lean arm, **and** the announce line confirms the
      wide halo and the lean path are running (L80 — the naive `EVPWIDE_RINGS` form passes on a no-op)

### Task B6a: Initial conditions and the det fill (`fesom_phc.cpp`, 50/2)

**Files:** Modify `src/fesom_phc.cpp`

- [x] **class 3** — PHC load/interpolation/`insitu2pot` in `real_t` (upstream `gen_ic3d` is WP with
      `MPI_WP` Bcasts); fill value cast to `real_t` **before** comparison; any stray `FieldT<double>`
- [x] det fill (`:310`/`:555` → `FESOM_MPI_REAL`; UNESCO polynomials + interpolation weights `real_t`; netCDF axes/data staging stay `double`) (`:495-575`) in `real_t`; **the two `MPI_Allreduce(MPI_DOUBLE, MPI_MAX)` over `real_t`
      scalars (`:310`, `:555`) → `FESOM_MPI_REAL`** (SP1 sites); `FESOM_IC_EXTRAP_TOL` default per
      precision (document the SP value and why: 1e-3 is ~8 000 float ulps at S≈35, so the DP default
      is reachable in SP — keep 1e-3 unless the sweep count says otherwise)
- [x] gate DP part (`det` config BYTE-IDENTICAL in the np8 jobs above; **SP part 2026-09-07, GPU pair job 27289143: DP and SP legs both `928 fill sweeps` (relax sweeps 7984/7443 vs 7988/7442, no relax-cap line) — the fill count matches; the ≤1e-6 initial-field check is still untested**): **CORE2 gate with PHC + det** (pi has no PHC path) bit-identical to `ref0`;
      **plus, once the SP build exists (C1):** the `[fesom_phc] det extrap: N fill + M relax sweeps` line
      shows the **same fill count** in DP and SP, no "relax cap" line in either, and the SP initial T/S
      fields agree with DP to ≤ 1e-6 relative L2 — the twins in G2–G4 must start from the same state,
      or every later difference is misattributed to precision

### Task B6b: Forcing and calendar (`fesom_jra55.*`, `fesom_calendar.*`) — the class-5 slice

**Files:** Modify `src/fesom_jra55.{cpp,h}`, `src/fesom_calendar.{cpp,h}`

- [x] `fesom_jra55.*`: **class 5** — `FESOM_JRA_POINTSLOPE` selector, `coef_a/coef_b` back to WP (`real_t`, upstream), `time_t0` `dbl_t` per field, `FESOM_JRA_AT(rd, dt, a, b)` at all 16 sites (device + host); DP affine line untouched — under `FESOM_SINGLE_PRECISION` only, upstream's point-slope form
      (`coef_b = data1`, `time_t0[fld]` `dbl_t`, `dt_elapsed = real_t(rdate − time_t0)`,
      `atm = coef_b + dt_elapsed·coef_a`, `coef_a/b` `real_t`); FP64 build keeps the current affine form
      **unchanged**; a compile define `FESOM_FORCING_POINTSLOPE` forces the form on in DP for the
      **G3 control leg only** (never a shipped default)
- [ ] `fesom_calendar.*` (deferred to the SP flip, C1: verify seconds-in-day kind then): seconds-in-day follows upstream (WP) — verify against the merged diff first
      (`gen_modules_clock.F90` is **not** in the merged file list, so the clock's kind is whatever the
      default-real flag makes it); record the finding in the registry either way
- [x] gate DP (job 27287828 on b7: every config BYTE-IDENTICAL; **SP part still open**): CORE2+JRA gate bit-identical to `ref0` (pi never runs this code);
      **gate (SP): PASSED INDIRECTLY by G1 (job 27288954 runs 30 h = 10 record boundaries, forcing maxima track DP: hf 2.90e3 both at step 60); the failed first attempt was exactly the collision class (registry log)** — interpolated forcing values across one JRA record boundary (3-h,
      the SP2 collision point) agree with a `dbl_t` reference evaluation of the same records to float
      rounding at every node, and the `[forcing]` inferred-resolution line says 3 h, not 6

### Task B6c: Main, step, restart I/O, phase stats

**Files:** Modify `src/fesom_main.cpp`, `src/fesom_step.{cpp,h}`, `src/fesom_io_restart.{cpp,h}`,
`src/fesom_phasestats.cpp`

- [x] `fesom_main.cpp` (92/3) + `fesom_step.cpp`: step-diag min/max buffers → `real_t` +
      `FESOM_MPI_REAL` (upstream `write_step_info` is `MPI_WP`; **SP1**: the July stack smash was exactly
      a `real_t` buffer under `MPI_DOUBLE`); timing/profile doubles stay; the M13 elemprobe block
- [x] `fesom_io_restart.*` (`fesom_nc_get_var_real` added to `fesom_nc_real.h`; plane/stiff reads+writes staged; Gatherv/Scatterv → `FESOM_MPI_REAL`; `header_d` stays `MPI_DOUBLE`): on-disk stays `NC_DOUBLE`; staging casts both ways; **`:334` Gatherv and
      `:421` Scatterv of `real_t` matrix values → `FESOM_MPI_REAL`; `:768` `header_d[2]` stays
      `MPI_DOUBLE`** (genuinely double); `FESOM_RESTART_IC` path
- [x] `fesom_phasestats.cpp`: timing only, stays double (documented)
- [x] gate (b7 md5 `5440d4e9`, 2026-09-07): pi np1+np2 all 14 configs bitwise; `scripts/m16_restart_gate.sh` zstar np2 write@10/read/run-to-20 **BIT-IDENTICAL** to the straight run; CORE2 np8 SLURM PASS 2026-09-07 (jobs 27287827/27287828). Spec: per-slice gate incl. one **restart round-trip at FP64** (write at step 10, read, run to
      20 — bit-identical to the straight run, the m14 restart gate's own claim) on pi np2

### Task B6d: Port the M8 instruments (`FESOM_MP_NANSCAN`, `FESOM_MP_TRACE_NODE`, `FESOM_MP_CONSERV`)

**Files:** Modify `src/fesom_main.cpp`, `src/fesom_step.cpp`, `src/fesom_bulk.cpp` (from
`~/port_kokkos_mp/src/`)

- [x] `FESOM_MP_NANSCAN=1` (functions + `fesom_bulk.cpp` hook from July verbatim; probes at step entry, after the SSH solve, after FCT T/S, at step end — each behind `if (fesom_mp_nanscan_enabled())` with the needed `sync_host()`): per-phase **non-finite** scan (NaN|Inf), located reports (elem/node, nz,
      comp, geo, owned/halo), rank-0 **ARMED banner**, momentum/ice/ssh probes included (July's 58b/58c
      blind spots)
- [x] `FESOM_MP_TRACE_NODE=<1-based gid>` + `FESOM_MP_TRACE_FROM=<step>`: end-of-step single-node series
- [x] `FESOM_MP_CONSERV=N` (call in `fesom_main.cpp` after `fesom_timestep`, July's cadence): `dbl_t` Kokkos reduce over owned wet columns + `MPI_DOUBLE` Allreduce of
      volume/heat/salt every N steps, device-current views
- [x] gate (b7): knobs off ⇒ pi bitwise (the b7 gate); knobs armed on pi np1 ⇒ `MP-NANSCAN ARMED`, `[mp-trace] ARMED global 100 = local 99` + per-step line, `CONSERV step= 1 heat=1.241289840326604e+19 …` (rc 0); CORE2 np8 SLURM PASS 2026-09-07 (job 27287828, b7). Spec: knobs off ⇒ per-slice gate bit-identical; **knobs armed at DP ⇒ still bit-identical and each
      instrument prints its own banner** (an instrument that is silent when armed is a dead knob)

### Task B7: CVMix TKE in double (class 2) + guard constants

**Files:** Modify `src/fesom_cvmix_tke.hpp`, `src/fesom_kpp.cpp`, `src/fesom_pp.cpp`, `src/fesom_constants.h`

- [x] `fesom_cvmix_tke.hpp` (2026-09-07): `tke_t = dbl_t`; inputs copied into `tke_t` column arrays at entry (the WP→cvmix_r8 shim), every scratch array/scalar/constant `tke_t`, tridiag in `tke_t`, the three outputs narrowed once at exit; the per-column CVMix arithmetic runs in `dbl_t` — inputs cast at kernel
      entry, outputs cast at exit (upstream's `cvmix_r8` shim, moved inside the kernel so no extra
      device arrays are needed); `tke_min2/max2` common-type literals
- [x] a compile define `FESOM_MP_TKE_REAL` keeps the TKE arithmetic in `real_t` **for the G2 give-back
      pair only** (DP-inert by construction; never a shipped option) — TKE(dbl) vs TKE(real_t) at SP on
      the same mesh and rank count is the only experiment that measures the class-2 cost
- [x] `KPP_EPSLN` per precision (B1, July's hunk) (1e-20 SP, 1e-40 DP unchanged — July's audit found it the only sub-1e-38
      guard); re-audit for guards added since (m9/m10/m12b code)
- [x] gate (b7): pi `tke` np1/np2 bitwise (the DP kernel is the same arithmetic); CORE2 `tke` np8 SLURM PASS 2026-09-07 (jobs 27287827/27287828); give-back at Gate 2. Spec: per-slice gate with `FESOM_MIX_SCHEME=TKE` and with KPP (the DP build is unchanged by
      construction — casts are no-ops); the `FESOM_MP_TKE_REAL` build is also DP bit-identical

### Task B8: Invariant grep, ledger close, Gate 0 **[NOW: pi + login CORE2]** / **[BLOCKED on /work: np128 + CUDA]**

**Files:** Create `scripts/m16_mpi_invariant.sh`; Modify `docs/PRECISION_ISLANDS.md`

- [x] `scripts/m16_mpi_invariant.sh` written and run 2026-09-07 (heuristic: every `MPI_DOUBLE`/`nc_*_double` call's buffer against its nearest declaration; `FLAG` lines are review items). Result after B1–B7: no `MPI_DOUBLE` over `real_t`; the three FLAGs are parser false positives reviewed by hand (`fesom_sss_runoff.cpp:161` and `fesom_io_stream.cpp:296-300` stage through `double`; `fesom_ice.cpp:1031` is an `MPI_DOUBLE_INT` MAXLOC pair). The SP compile probe (`build-m16serial-sp`, `--target fesom_core -- -k`) is the second judge for pointer seams: **zero errors after B6c** (probe 3).
- [ ] `scripts/m16_mpi_invariant.sh`: every `MPI_DOUBLE` call site's buffer resolves to `dbl_t`/`double`
      storage, every `FESOM_MPI_REAL` to `real_t` — zero violations; run in the gate scripts
- [ ] `grep -nE '\bdouble\b' src/` audit: every remaining raw `double` is diagnostic/timing/I/O staging
      or a registry row — list the exceptions in the registry
- [ ] **G0 now**: the five configs + the later-gate arms on pi np1/np2 and CORE2 np2 (login, PHC+JRA),
      bit-identical vs `ref0`
- [ ] **G0 later**: `jobs/job_m14_gate_serial` with `M14_BASELINE=<ref0 snapshot dir> M14_EXPECT=same`
      at CORE2 np8/np128 on the five configs; `jobs/job_m14_gate_cuda` at np8/np16 with the 3-leg self
      control (`jobs/job_m14_cudarepro2`)
- [ ] freeze the FP64 pair `port2/m16/bin/dp0/` (later); record shas here
- [ ] gate: all of the above green ⇒ Phase B closed

---

# Phase C — the single-precision flip and Gate 1 **[NOW]** (CUDA run **[BLOCKED on /work]**)

### Task C1: SP Serial build compiles and runs

**Files:** whatever the compiler finds (`src/fesom_nc_real.h`, `src/fesom_io_stream.cpp`, evpwide, tke,
…); `configure.sh`

- [x] **2026-09-07, first SP build:** `build_m16.sh serial-sp` links after fixing two unit tests that
      assumed `double` (`test_field` instantiation, `test_ssh_solvers` exact-zero rounding checks →
      precision-scaled bounds); the model sources needed **no** further seam fix beyond B1–B7. pi np1+np2,
      five configs, 20 steps: rc 0, zero non-finite, banner `SINGLE … epsilon=1.19e-07` asserted, I/O
      report says `MEAN accumulated in double`. ➕ **Gate-1 observation (pi, default, from rest):** SP vs
      DP after ONE step differs by up to 21 % of max|u| at a single slot (DP −8.86e-4, SP −6.96e-4 m/s),
      192 of 274 k slots > 1 %, RMS 3.6e-3 of max; T/S differ at 1e-5/1e-7 relative (rounding class),
      eta at 1e-2 only where |eta| ≈ 1e-6. Absolute Δu ≈ 1e-4 m/s is the float noise floor of the
      hydrostatic-pressure running sum (`fesom_eos.cpp` hpressure, the registry's "promote FIRST"
      suspect) driving spurious PGF at tiny velocities; not a defect by the plan's definition (G4 judges
      pattern correlation and SP-vs-Fortran ≡ DP-vs-Fortran), but the first candidate if G3/G4 object —
      a `dbl_t` hpressure column sum is a one-array promotion.
- [ ] `USE_SINGLE_PRECISION=ON` Serial build: fix every compile error at the seams (July: 35, all
      boundary seams — nc-write staging, evpwide MPI types, cgpipe Views, tke common_type); expect new
      ones in SE, restart I/O, M10 solvers
- [ ] `-Wdouble-promotion` pass over hot kernels; literal hygiene (`real_t(0.5)` forms, `Kokkos::sqrt`
      over `sqrt`) — a silently promoting kernel gives the speed back without saying so
- [ ] banner prints `SINGLE`, epsilon `1.19e-07`; `mp_assert_banner.sh` passes; the I/O precision report
      says 4-byte working precision
- [ ] gate: pi np1 + np2, 20 steps, rc 0, zero non-finite; **`FESOM_MP_NANSCAN=1` prints its ARMED banner
      and no hit**; then the SP-side gates of B3b, B6a and B6b run (they were waiting for this binary)

### Task C2: SP CUDA build

**Files:** `build_m14.sh` → `build_m16.sh` (dp/sp × serial/cuda)

- [x] `build_m16.sh` producing four binaries; CUDA half sources `env_cuda.sh` (RPATH rule); `ldd` check (2026-09-07: `cuda` `ba6334ad`, `cuda-sp` `f3c8c4a4` after the one-line `fesom_halo_device.cpp` selfcheck fix — `std::vector<real_t>`; both link `openmpi-4.1.5/lib/libmpi.so.40`; frozen `port2/m16/bin/cuda0` + `cudasp0`)
- [x] compiled on the login node (nvcc is there; ~10 min at -j16)
- [x] CUDA FTZ/denormal posture: `flags.make` carries only `-O3` — no `--use_fast_math`, `-ftz`, `-prec-div`, `-prec-sqrt` ⇒ nvcc defaults `-ftz=false -prec-div=true -prec-sqrt=true -fmad=true`: denormals are NOT flushed on the GPU, i.e. the same posture as the host `-O3` build (no `-ffast-math`); documented in the registry §1 (epsilon row)
- [x] gate: both SP binaries exist, banner `SINGLE`; **CUDA pi smoke PASS** (`jobs/job_m16_smoke_cuda`, job 27288722, gpu `-C a100_80`, 84 s): DP+SP × {default np1, default np2, zstar np2} + SP `FESOM_MP_NANSCAN=1` all rc 0, 21 snapshots, banner asserted, ARMED banner present, no non-finite. Reported (not gated): DP-CUDA vs Serial ref0 max|Δv| 3.1e-5 (CUDA atomic-scatter class); SP-CUDA vs DP-CUDA at step 20: u 5.2e-3, v 5.3e-4, T 2.0e-4, S 7.6e-5, eta 8.1e-5, Av/Kv 0.1 (mixed-layer threshold flips), lon/lat 2e-5 (WP coordinates, class 3) — the same u-sized SP gap as the Serial pi finding (hpressure running sum, registry suspects, first candidate if G3/G4 object)

### Task C3: Gate 1 — it runs past the first forcing boundary

- [x] **G1 PASS 2026-09-07 — on SLURM np8, not the login node (standing rule)**: `jobs/job_m16_core2_twins` `core2_g1b` job 27288954 (`bin/d1sp` `722e3002`), 60 steps dt 1800 PHC+JRA 1958: rc 0, no non-finite, CG |Δit| vs DP mean 0.43 max 1 (July 2.2/4); SP vs DP at step 60: S relL2 1.15e-5 (mean +3.6e-6 psu), T relL2 1.4e-4, eta relL2 1.2e-4, u/v relL2 8e-3. **First attempt (`d0sp`, job 27288894) DIED at step 5** — `FESOM_JRA_POINTSLOPE` used above its definition (registry log); `-Wundef` now in the build. Spec: CORE2 np1 on the login node (private mesh `port2/mesh/core2`, readable), `jra55_year=1958`,
      **60 steps dt 1800**, `snap_every=-1`, log to `~/m16_scratch/g1/`: rc 0, no non-finite, CG
      iterations tracking the DP twin (July: mean |Δit| 2.2, max 4), `[ssh-verify]` gap in the DP class
- [x] (np8 exercises the swept exchange; np2 not run separately) same at np2 (login) to exercise the swept exchange
- [x] frozen `port2/m16/bin/d1` (DP `f2eb28b1`) / `d1sp` (SP `722e3002`) — the current pair (Serial); the CUDA pair `cuda0`/`cudasp0` is pre-Phase-D and must be rebuilt
- [x] gate: G1 green ⇒ Phase D gates may run (Phase D code may be written in parallel on separate
      commits)

---

# Phase D — the salt-anomaly option, exact port of #986 **[NOW]**

### Task D1: Knob, reference value, one-time conversion, guard

**Files:** Modify `src/fesom_tracers.h` (or `fesom_constants.h`), `src/fesom_main.cpp`, `src/fesom_phc.cpp`

- [x] (2026-09-07: `fesom_tracers.{h,cpp}` `fesom_S_ref_anomaly` + `fesom_salt_anomaly_setup`, abort on other values) `FESOM_SALT_ANOMALY` env knob: unset/0 = off; `1` = on with `S_ref = 35`; any other positive number
      = on with that reference (**measurement-only**, for the invariance gate); abort on other values
      (the wsplit precedent); a global `S_ref_anomaly` (`real_t`) set **once** in setup
- [x] (called in `fesom_main.cpp` after the PHC/blob IC, before the ice IC and the restart read; whole array as upstream) conversion `S -= S_ref` **after** the PHC load and `insitu2pot` (`fesom_phc.cpp:773` — it needs
      absolute S), before the AB copies are made; rank-0 line `use_salt_anomaly: salinity state = S - 35`
- [x] (enumerated 2026-09-07 by grepping every `FESOM_TRACER_S].values` reader: EOS ×2, sw_alpha_beta ×2, ocean2ice ×2, oce_fluxes rsss/relax ×2, sss_runoff rsss/relax, bc_surface ×2, KPP Bo ×2, the port-only salinity floor, io salt/sss ×2 + the snapshot writer, restart read, step/probe/conserv diagnostics — ALL carry the offset; GM's S gradient is offset-invariant to rounding as upstream; ddmix is `#error`-guarded, KPP nonlocal is never consumed, no cavities/icebergs/age/clim_relax exist ⇒ **no guard needed, none added**) refuse-to-start guard for every port feature whose consumer carries no offset — enumerate them by
      grepping the D2 list against the port's option set (cavities/icebergs/age tracer do not exist here;
      check GM, KPP nonlocal, 3-D relaxation, any salt diagnostic)
- [x] **PASS 2026-09-07** (`scripts/m16_salt_anomaly_gate.sh`, frozen `bin/d0` DP `7c1d83df` / `bin/d0sp` SP `8ad967fe`): knob unset ⇒ `m16_gate0.sh all` np1 14/14 BYTE-IDENTICAL to ref0; unset ≡ `=0` bitwise (DP and SP); DP restart round-trip (no knob) bitwise. Spec: off ⇒ pi np1/np2 bit-identical to the SP binary built at the same commit in `~/m16_scratch`
      (SP) and to `ref0` via `m16_gate0.sh` (DP)

### Task D2: Consumers add the offset back — line by line against the Fortran

**Files:** Modify `src/fesom_eos.cpp` (host + device densityJM, linear EOS variants, `sw_alpha_beta`),
`src/fesom_ice_coupling.cpp`, `src/fesom_sss_runoff.cpp`, `src/fesom_tracer_diff.cpp`, `src/fesom_kpp.cpp`,
`src/fesom_ice_thermo.cpp` (if it reads S directly)

- [x] (every hunk applied, each site comments its Fortran line; `density_linear`, the 3/45 clip and the `s<0` blow-up screen do not exist in the port — only the port's own 0.5 floor, shifted) for each hunk in `pr986.diff` + the `S_ref_anomaly` hunks of `pr940.diff` (`oce_ale_pressure_bv`,
      `ice_oce_coupling`, `oce_ale_tracer`, `oce_ale_mixing_kpp`, `oce_setup_step`, `write_step_info`):
      find the port twin, apply `+ S_ref_anomaly` in the **same expression**, cite the Fortran line in a
      comment; the checklist is the diff, not July's notes
- [x] (`s_ref` is a parameter of `fesom_eos_jm_components{,_kk}` — a device lambda cannot read the host global) EOS: `s_abs = s + S_ref` then the polynomial in `s_abs` (both host twin and device kernel); the
      McDougall `sw_alpha_beta` `s1 = SF1 + S_ref`; linear/`density_linear` forms `(s_abs − 35)`
- [x] ice–ocean: `S_oc = salt + S_ref` (both the direct and the running-mean branches), `rsss =
      salt(top) + S_ref` under `ref_sss_local`, `relax_salt = surf_relax_S·(Ssurf − S_ref − S_top)`,
      `dens_flux` with `(salt + S_ref)`
- [x] surface BC: `flux = virtual_salt + relax_salt + (real_salt_flux + S_ref·water_flux)·is_nonlinfs`
      (both the host path `:77-83` and the device functor `:392-394`)
- [x] KPP surface buoyancy flux: `sw_beta·water_flux·(S_top + S_ref)` (`:1385`, `:1577`)
- [x] (not in the port; the port-only `fesom_salinity_floor_kk` 0.5 psu floor shifted instead) clipping 3/45 psu and the `pressure_bv` "blows up" screen shifted by `−S_ref`
- [x] **PASS 2026-09-07** (DP pi np2, 20 steps, snapshots 10+20; linfs: off/35/10 pairwise T ≤ 3.2e-10, S ≤ 1.4e-13, eta ≤ 3.3e-13; zstar: T ≤ 1.3e-10, S ≤ 1.4e-13, eta ≤ 4.3e-10 — rounding class, all pairs; SP reported: T ~1e-4, S ~8e-5, eta ~3e-5 = float rounding class). Spec: gate (the one that can actually find a missed site — **S_ref invariance, DP, pi, NOW**): three
      runs, off / `FESOM_SALT_ANOMALY=1` (35) / `=10`, 20 steps; `mp_divergence_curve.py` pairwise:
      all three agree to the rounding class (relative L2 ≲ 1e-13 on T/S/η) — a missed consumer shows
      as an O(1) error that scales with the reference; plus off ⇒ still bit-identical

### Task D3: Restart detection, output offset, step diagnostics

**Files:** Modify `src/fesom_io_restart.cpp`, `src/fesom_io.cpp`, `src/fesom_io_stream.cpp`,
`src/fesom_main.cpp`

- [x] restart read: global `max(salt)` (`FESOM_MPI_REAL`, `MPI_MAX`); `> 20` ⇒ subtract `S_ref` from
      `values`, `valuesAB`, `valuesold` once, print "absolute-salinity restart detected → converted";
      else "anomaly restart → no conversion"; restart **write** stores the state as held (upstream)
- [x] (offset added inside the four salt/sss resolvers — the port's equivalent of upstream's per-stream `offset` at accumulation — plus the port-only snapshot writer) output: per-stream additive offset applied at accumulation; `salt` and `sss` register `S_ref`
      so files stay absolute psu (both the host and the device accumulation paths)
- [x] (the step line prints `S_min/max + S_ref`, i.e. absolute psu; the IC print at `:600` runs BEFORE the conversion and is left alone; probe/mp-trace/conserv diagnostics add `S_ref`) step-diag `S[min,max]` bounds shifted by `−S_ref` (upstream `write_step_info`); the `Ssurf` read at
      `fesom_main.cpp:574` gets `+ S_ref`
- [x] **PASS 2026-09-07** (zstar np2: anomaly restart @10 → "no conversion" → run to 20 BIT-IDENTICAL to the straight anomaly run; absolute restart @10 read with the knob → "converted" → T 1.2e-10, S 7e-14, eta 4.2e-10 vs the anomaly run; same shape in SP, T 1.1e-4 / bitwise; the `salt` stream check needs an io config and is deferred to CORE2). Spec: gate (DP, pi, on): restart at step 10 written in anomaly, read back, "no conversion" printed,
      run to 20 bit-identical to the straight anomaly run; a restart written **without** the knob read
      **with** it prints "converted" and the run then matches the anomaly run to the rounding class; the
      `salt` stream mean equals the off run's to rounding (absolute psu on disk)
- [x] **measured 2026-09-07** (`core2_all/REPORT.txt`, DP np8 60 steps): on vs off S relL2 1.9e-4, mean −1.9e-4 psu (≈3e-6 psu/step — the upstream freshwater-driven surface residual class), T relL2 1.2e-3, eta relL2 1.0e-3, CG |Δit| mean 0.10. Spec: gate (DP, CORE2, on vs off): `mp_divergence_curve.py` solution-class
      comparison, expecting the surface conservation residual upstream reports (~8e-6/step,
      freshwater-driven, 99.4 % cancelled by the `S_ref·water_flux` term)
- [x] **measured 2026-09-07 (30 h, np8): improves, does not regress** — SP-vs-DP twin error with the anomaly (dp_on vs sp_on) S mean +9.2e-7 psu, relL2 1.11e-5; without (dp_off vs sp_off) S mean +3.6e-6, relL2 1.15e-5 ⇒ mean salt error −74 % (upstream −38 %), rms −3.5 %; T mean 6.2e-7→7.9e-7 (noise), eta relL2 1.2e-4→1.0e-4, w relL2 4.5e-2→3.9e-2. Spec: gate (SP, CORE2, 1 day, on vs DP reference): mean/rms salt error vs the SP
      run without the knob; expect the ~38 % mean-error reduction (1.55e-5 → 0.96e-5 psu upstream);
      report the number, pass/fail is "improves, does not regress"

---

# Phase E — the measured ladder (/work writable again 2026-09-07)

### Task E1: One harness with a precision axis (D11)

**Files:** Modify `jobs/job_m14_ladder_cpu`, `jobs/job_m14_ladder_gpu`; **Create**
`scripts/m14_collect.py`, `scripts/m14_zombie_check.py` (extracted from the inline python at
`jobs/job_m14_ladder_cpu:171` and `:143`), `jobs/m16_bins.sh`; copy `scripts/mp_{cuda_gate,
divergence_curve,envelope_verdict,conserv_drift,gate4_verdict}.py`

- [x] (2026-09-07: `scripts/m14_collect.py` + `scripts/m14_zombie_check.sh`, both ladders call them; round-trip on `ladder_farc_1024_26985036/legs.txt` identical to the inline python; the zombie check rejects that M14 log when SINGLE is demanded) extract the collector and the zombie check into scripts the jobs call (one implementation, both
      ladders); byte-for-byte same CSV rows on an existing M14 log
- [x] (`PREC=dp|sp` for base/best; `ARMS="dp sp"` = the precision pair itself; `M16_BINS=<pair dir>`; `preflight_bin` refuses a mismatched or provenance-less sp binary; `DRYRUN=1`; `SSH_VERIFY_BAR`) `PREC=dp|sp` axis: the arm pairs become `(BIN_DP, BIN_SP)` at identical knobs, ABBA in one
      allocation, warm-up discarded, **equal leg counts**, min over legs; `cfg=` stamps `prec=`; the
      collector refuses a log without the banner **and** the per-step timer; the banner precision must
      match the requested arm (L80/SP10)
- [x] (`jobs/m16_bins.sh <tag>` → `bin/<tag>/{dp,sp}/{fesom_port_serial,fesom_port_cuda}` + PROVENANCE; **`e0` = commit `f95eaef`: dp serial f2eb28b1 / cuda 40374895, sp serial 722e3002 / cuda c21fd52b; CUDA smoke on this pair job 27289067 PASS**) `m16_bins.sh`: sha-named frozen pairs under `port2/m16/bin/<tag>/{fesom_port_serial,fesom_port_cuda}`,
      `PROVENANCE.txt` with commit, flags, Kokkos version
- [x] zombie check extended: non-finite step diag, `it=0`, CGPIPE-INACTIVE, **and** a `SINGLE` banner
      where `sp` was requested, **and** an `[ssh-verify]` gap above the pre-registered bar
- [x] gate PASS (login dry runs: `PREC=sp` with a DP-provenance pair → REFUSE; `PREC=sp` with no sp binary → REFUSE; legacy M14 `i1` accepted as dp; `ARMS="dp sp"` with a proper pair → OK): the extracted scripts round-trip one existing M14 log unchanged; a dry run with `PREC=sp`
      refuses a DP binary

### Task E2: Gate 2 — prize sizing (D9)

- [ ] **in progress 2026-09-07** — CORE2 1N GPU (4 A100): DP 0.0618 / SP 0.0531 s/step, **SP/DP 0.859**, CG 60/62, job 27289143 (`docs/MIXED_PRECISION_M16.md` §1); CORE2 16N GPU job 27289163 submitted; NG5 16N next; **fArc 4096 / dars 8192 CPU exceed the 16-node cap → user decision**. Spec: pairs, 300 steps, `WSPLIT=1` where the mesh needs it: CORE2 4N + 16N GPU · fArc CPU at 4096 ·
      dars CPU at 8192 · NG5 16N GPU · device memory per GPU (dars 2N sampled mid-run, July 23.2 → 11.9 GB)
- [ ] knobs-off pairs **and** recipe pairs (D8) — July's headline: SP and the speed stack overlap in the
      communication bytes (SP under the stack 1.11–1.24× vs 1.26–1.48× knobs-off); both rows go on the board
- [ ] CG iterations SP vs DP per leg (July +1) **and** the `[ssh-verify]` gap per leg
- [ ] **class-2 give-back:** one CORE2 SP pair TKE(`dbl_t`) vs TKE(`FESOM_MP_TKE_REAL` build), same mesh,
      same ranks; **stiffness-shadow give-back:** device memory with and without zstar at SP
- [ ] gate: board recorded in `docs/MIXED_PRECISION_M16.md` with job ids before any G3 leg

### Task E3: Gate 3 — every recipe knob re-certified at SP

- [ ] table: knob → liveness signal (grep'd from the merged tree, not hand-written: `FESOM_SPEED=1`
      selfchecks, CGPIPE/CGPOLY selfcheck 0.000e+00, EVPWIDE announce + `LEAN` running, SE announce +
      drift 0.0, `oati` iteration trace, `det` fill announce + sweep counts, `PRECOND` announce,
      `WSPLIT` announce, NANSCAN/CONSERV ARMED banners)
- [ ] **both halves run 2026-09-07 (CUDA job 27289199, Serial job 27289198): 12/15 live on each; `pipecg`/`oati`/`pcsi` fall back on every solve at SP (0 in FP64) — root cause = the SP true-residual floor (registry G3 entry); response = CA scalar chains → `dbl_t` + `FESOM_SSH_FLOOR` (announced, counted); **re-test on `e1` (jobs 27289394/27289407): `pcsi` live, `pipecg`/`oati` still diverge (13/20) → declared unusable at SP as built, E5 residual replacement; the G3 knob table is otherwise complete: 13/15 live.** Spec: each signal reproduced at SP on CORE2 np8 (Serial, `FORCE_SERIAL=1`) and on 2 GPU nodes; the
      NaN-zombie check on every leg
- [ ] (first data point 2026-09-07, CORE2 np8 Serial+CUDA, `cg`: true/rtol 1.11 at solve 1 → 2.06 at solve 20, gap ≡ true−rec 0.82–0.90; FP64 gap 1e-11 — the bar is now measured, the sweep over ranks/meshes remains) **true-residual rank sweep**: `cg` (CORE2 128→864), CGPIPE (GPU 1→16 N) and `oati` (fArc
      512→4096, dars 1024→8192) at SP with `FESOM_SSH_VERIFY=1`; gap vs the B3 pre-registered bar; a
      failure here is the first class-3 promotion (scalars back to `dbl_t`), logged with the signature
- [ ] 30-day CORE2 conservation, `FESOM_MP_CONSERV=10`, SP vs DP (`mp_conserv_drift.py`): heat gap vs the
      physical 30-d signal, salt/volume random-walk vs drift — July: 0.2 %, sign-changing
- [ ] divergence curve vs the FP64 dt-seed **ensemble** envelope (`mp_envelope_verdict.py`; single seeds
      are non-monotone — July's finding), **plus one DP leg built with `FESOM_FORCING_POINTSLOPE`** so
      the class-5 formulation delta is separable from the precision delta; record its size
- [ ] gate: no dead knob, no stagnating or falsely-converged solver, no leak signature; failures ⇒ D9
      protocol, one island at a time, registry row with signature + give-back

### Task E4: Gate 4 — screens and the 1-year twin

- [ ] 3000-step screens, SP, for each of the 8 mesh × backend cells of the D8 recipe (CORE2/fArc/dars/NG5
      × CPU/GPU) at the M14 knee points; `WSPLIT=1`; SE drift 0.0 where the wide halo is on
- [ ] 1-year CORE2 SP vs DP twin at the 63A posture (2N × 4 A100-80, 25-min-class walltime resubmits,
      output-protected), `mp_gate4_verdict.py`: annual-map correlations sst/sss/ssh/a_ice + the Fortran
      control row (SP-vs-Fortran ≡ DP-vs-Fortran to the printed digit)
- [ ] gate: G4 green ⇒ **the option is declared usable**; the 63-yr decision goes to the user

### Task E5: Second wave — CA solvers, restart interchange, remaining knobs (D8)

- [ ] `pipecg`, `pcsi`, `cg2` at SP: liveness + `[ssh-verify]` gap at CORE2 np8 and fArc 2048; the
      recurrence promotions expected here are logged as class-4 promotions
- [ ] restart round-trip at SP: SP → disk (double, shadow included per B3b) → SP **bit-exact**; SP
      restart into a DP binary runs (the shadow seeds the DP matrix exactly); DP → SP truncates by
      design (documented)
- [ ] remaining knobs from the grep list: each gets an SP status line **with evidence** (job id + the
      signal) for "live", or "inert-by-design" with the source line, or "untested" — and the untested
      list is printed in the campaign doc as such
- [ ] gate: every D8-recipe and second-wave knob has an evidenced "live" line; the untested list is
      explicit

---

# Phase F — documents and merge readiness

### Task F1: Registry, lessons, campaign document

**Files:** Modify `docs/PRECISION_ISLANDS.md`, `docs/KOKKOS_PORTING_LESSONS.md`; Create
`docs/MIXED_PRECISION_M16.md`; Modify `docs/SSH_SOLVERS_M10.md` (re-base note)

- [ ] registry final: every `dbl_t` a row, every row a class and a Fortran line, every promotion a
      signature and a measured give-back
- [ ] fold SP1–SP11 into the lessons file as new L-numbers (check the highest L on m14 first — ≥L122);
      add the lessons this campaign earns
- [ ] `docs/MIXED_PRECISION_M16.md`: the G2 board (knobs-off and recipe rows), G3 knob table, the
      true-residual sweep, G4 verdicts with job ids, the conformance table summary, the three declared
      divergences, salt-anomaly numbers, per mesh/backend recommendation, and the honest list of what is
      untested; **no new figures** unless asked (then `m7_scaling_figs.py` first)
- [ ] gate: no number in the document lacks a job id; `scripts/m14_coverage.py`-style check that every
      D8 cell has a screen

### Task F2: Merge readiness

- [ ] re-run the amended revert gate (D3) on the final tree, both backends: knobs unset +
      `FESOM_SSH_PRECOND=0` vs `ref0` (Serial, bitwise) — and `ref0` vs `i4` vs `main`'s `h17` snapshots
      stated once as the chain to `main`
- [ ] `git merge` `m16-precision` into `m14-integrate` (real merge, in `~/port_kokkos_int`); re-run
      `jobs/job_m14_gate_serial` on the merged tree knobs-off + `PRECOND=0`
- [ ] `git diff main...m14-integrate -- src/` reviewed end to end for the precision changes
- [ ] binaries in `port2/m16/bin`, shas in the docs, none committed; `~/m16_scratch` deleted
- [ ] open the merge to `main` for the user's decision — **do not merge or push unasked**
- [ ] move this plan to `docs/plans/completed/`

---

## Technical details

**Type mapping (D5/D6).**

| Fortran (#940) | port | meaning |
|---|---|---|
| `real(kind=WP)` / unkinded `real` | `real_t` | working precision, float under `USE_SINGLE_PRECISION=ON` |
| `real(kind=real64)`, `WP_full`, `cvmix_r8` | `dbl_t` | deliberate FP64 island; every use is a registry row |
| `MPI_WP` | `FESOM_MPI_REAL` | payload type for `real_t` storage |
| `MPI_DOUBLE_PRECISION` (kept) | `MPI_DOUBLE` | only over `dbl_t`/`double` storage — grep-enforced |
| `-DUSE_SINGLE_PRECISION` | `-DUSE_SINGLE_PRECISION=ON` → `FESOM_SINGLE_PRECISION` define | |
| `use_salt_anomaly` (namelist) | `FESOM_SALT_ANOMALY=1` (env; numeric = test reference) | port convention: knobs are environment variables |
| `precond_variant` (namelist, ships 1; code auto −1) | `FESOM_SSH_PRECOND` (default 1, no auto) | declared divergence (a) |
| `DIAG_STIFF_DRIFT` (compile) | `FESOM_DIAG_STIFF_DRIFT=1` (env, SP build) | the shadow slice's only gate |
| — | `FESOM_SSH_VERIFY=1`, `FESOM_MP_TKE_REAL`, `FESOM_FORCING_POINTSLOPE` | measurement-only instruments/defines, never defaults |

**The five conformance classes (D5).**

1. **Converged** — keep: forcing time chain `dbl_t` + `dbl_t` binary search; stiffness shadow (upstream
   semantics); FTZ-safe guards (1e-20 SP); files double on disk, cast at the boundary; fill value cast
   to `real_t` before comparison; 8-byte output means accumulated in `dbl_t`; global integrals
   (`integrate_nod_2D/3D` class) in `dbl_t` with `MPI_DOUBLE`.
2. **Upstream stricter** — follow upstream, measure the cost: CVMix-derived TKE in `dbl_t` inside the
   kernel with `real_t` in/out.
3. **M8 stricter than upstream** — flip to `real_t`, re-earn on a failing gate: CG scalar chain, dot
   accumulators, `Allreduce` scalars (detector = `[ssh-verify]` true-residual gap, rank sweep); mesh
   metrics precompute; PHC init path; min/max diagnostics; mesh volume sums; calendar seconds-in-day
   (verify).
4. **Port-only code** — `real_t` except global integrals: SE subcycle and wide-halo buffers, EVPWIDE
   lean, CGPIPE/CGPOLY, M10 CA solvers, det IC fill (`FESOM_IC_EXTRAP_TOL` per precision), restart I/O,
   dumps, verify twins. Rounding-fragile recurrences (M10's fArc stall was recurrence rounding) are the
   expected first promotions. Diagnostic-only reductions (selfchecks, drift prints, timers) may stay
   `dbl_t` and are marked "diagnostic" in the registry.
5. **Exact port conflicts with Gate 0** — SP-only: upstream's point-slope forcing interpolation; a DP
   control build with `FESOM_FORCING_POINTSLOPE` bounds the confound in G3.

**Kernel pattern.** For class-2 kernels and `dbl_t` scalars: cast at the kernel boundary, compute in
`dbl_t`, cast once on the way out; no extra device arrays. **Exception, registered as a whole-field
promotion:** the stiffness shadow is a device `FieldT<dbl_t>` because `fesom_update_stiff_mat_ale_kk`
is a device kernel accumulating into it with atomics (M8's `values_dbl_fld`). Mixed-type expressions
promote to double automatically — the `-Wdouble-promotion` pass catches **unintended** promotions in
hot `real_t` kernels.

**Stiffness shadow (upstream #997 semantics + divergence b).** `values_full` (`dbl_t`, sized to the
**local** nnz) allocated and seeded from `values` on the **first** `update_stiff_mat_ale` call (correct
for cold start and restart alike); every increment `values_full += fy·factor` in `dbl_t`;
`values = real_t(values_full)` once per update; compiled only under `FESOM_SINGLE_PRECISION`;
`FESOM_DIAG_STIFF_DRIFT=1` prints relL2(diag)/relL2(offdiag) of a WP-accumulated twin. The SP restart
writes `values_full` into `stiff_values` (already `NC_DOUBLE`) so SP → disk → SP is bit-exact.

**Point-slope forcing (upstream, SP-only).** `time_t0[fld] = nc_time[t_indx]` (`dbl_t`);
`coef_b = data1`, `coef_a = (data2 − data1)/delta_t` (`real_t`, slope formed in `dbl_t`);
`dt_elapsed = real_t(rdate − time_t0[fld])`; `atm = coef_b + dt_elapsed·coef_a`.

**True-residual instrument (`FESOM_SSH_VERIFY=1`).** After each solve: `r_true = b − A·x` in `dbl_t`
(SpMV in `dbl_t` over the `real_t` matrix), `‖r_true‖₂/‖b‖₂` and the solver's own recurrence residual,
printed as `[ssh-verify] solve N: true= rec= rtol= gap=` — the existing M10 line, extended to `cg`
and CGPIPE. Pre-registered bar (B3): gap(SP) ≤ 10 × gap(DP) at every rank count; no solve whose true
residual exceeds `rtol` by more than the DP run's own worst.

**Salt anomaly (upstream #986).** `S_ref_anomaly ∈ {0, 35}` set once after the initial state; every
absolute-salinity consumer adds it back unconditionally (bit-identical no-op when off); restart
detection by global maximum (> 20 psu ⇒ absolute ⇒ convert all three time levels); output streams
carry a per-stream offset applied at accumulation; blow-up/clipping bounds shift by `−S_ref`. Upstream's
own caveat, carried verbatim into the docs: enabling it is **not** exactly DP-neutral — the free-surface
advection is not perfectly constancy-preserving, leaving a ~8e-6/step surface residual. The invariance
gate (`S_ref ∈ {35, 10}` vs off) is the completeness check: a missed consumer scales with `S_ref`.

**Preconditioner variants (kept from M15).** 0 as-built `−0.5a/(d_i(d_i+d_j))` (asym, ¼×); 1 = #984
`−a/(d_i d_j)` (sym, 1×, **default**); 2 = M10 note `−0.5a/(√(d_i d_j)(d_i+d_j))` (sym, ¼×); 3
symmetrised header `−2a/(√(d_i d_j)(d_i+d_j))` (sym, 1×); 4 header as written `−2a/(d_i(d_i+d_j))`
(asym, 1× — diverges). SYMPRE only composes with 0; under 1/2/3 every solver reads `S->pr_values`.

**Test commands.** Byte gate: `scripts/m16_gate0.sh <build> <config>` (wraps
`python3 scripts/diff_snap.py <ref_dir> <new_dir>`, zero tolerance, exports `FESOM_SSH_PRECOND=0`).
Solution class: `python3 scripts/mp_divergence_curve.py <ref_dir> <new_dir>`. CUDA envelope:
`python3 scripts/mp_cuda_gate.py --noise <rerun_dir> <ref_dir> <new_dir>`. Banner:
`scripts/mp_assert_banner.sh <log> SINGLE|DOUBLE`. MPI invariant: `scripts/m16_mpi_invariant.sh src/`.
Unit tests: `ctest` in the build dir (`test_field`, `test_ssh_solvers`).

---

## Post-completion

*Requires external action or a user decision.*

**User decisions needed:**
- authorise each push of `m15-precond` / `m16-precision` (D1)
- the 63-yr SP hindcast after G4 (D9 / brainstorm: "we start with 1, but might eventually do 2")
- merge `m14-integrate` (with M16) to `main`
- if `ref0` and `i4` differ knobs-off on any G0 config (P2 gate): what changed in `d4a9fe0` and whether
  M14's certified chain needs re-anchoring

**External:**
- when `/work` returns: delete `~/m16_scratch`, freeze binaries, run every **[BLOCKED on /work]** gate
- JUPITER: the SP footprint and communication numbers feed the GH200 plan (`reference-jupiter-plan`)
- paper: the conformance table + promotion log + G2/G3 boards are the FESOM2 precision-sensitivity
  material; coordinate with the #940 author

**Deferred:** half precision (per-field `half_t` under the same registry discipline; whole-model out);
anomaly variables beyond salinity; coupled builds; `FESOM_SE_WIDE` deep-K at SP.

---

## Review history

**Rev 2 (2026-09-02)** — plan-review agent, 17 findings + 10 minor, every source claim re-verified in
`~/port_kokkos_int` before acceptance. Applied: (1) the P3 default flip collided with every Phase-B byte
gate — every gate script now exports `FESOM_SSH_PRECOND=0` (D3 rewritten); (2) `i3` (`fe01a2bf`) and
`i4` (`4436583`) are both older than the base `d4a9fe0` (8 src files, +1477 lines: restart I/O, SE-wide
contract) — a `ref0` oracle built from the unmodified base replaces them, with an `i4` link gate;
(3) the SYMPRE rule now names all four build sites incl. the unconditional `pcsi` one at `:3428`, both
`pcsi` reads and `cgpipe_ship_pr`, and `pcsi` is gated on solution equality, not on running; (4) the M8
instruments got an owning task (B6d) and gates that assert their banners; (5) the salt-anomaly
completeness gate is now the S_ref-invariance test (35 vs 10 vs off), executable now; (6) the SP restart
round-trip is made bit-exact by writing the shadow to disk (declared divergence b) instead of an
unachievable bar; (7) forcing/bulk/PHC slices are gated on a login-node CORE2+JRA run, since pi never
executes them; (8) the CG float flip has a named detector — the existing `[ssh-verify]` true residual
extended to `cg`/CGPIPE, with a pre-registered gap bar and a rank sweep in G3; (9) the two SP-only slices
(shadow, point-slope) got SP-side gates and their own tasks (B3b, B6b); (10) `m14_collect.py` /
`m14_zombie_check.py` do not exist — E1 extracts them from the ladder job; (11) the `[m14]` knob summary
is extended so it cannot claim "default path" under variant 1; (12) the CVMix give-back is measured by
a TKE(dbl) vs TKE(real_t) define pair, not TKE vs KPP; (13) the point-slope confound gets a DP control
leg; (14) the det fill's two `MPI_DOUBLE`-over-`real_t` Allreduces are named and a same-IC gate added;
(15) four M14 rules restored (equal leg counts, figure rule, fArc proto package, snapshots-not-logs);
(16) the restart `MPI_DOUBLE` split is specified (two flip, header stays); (17) B3 and B6 split for
bisection. Minor: `M14_BASELINE` is a snapshot dir + `M14_EXPECT=same`; `diff_snap` has no relative
mode (use `mp_divergence_curve.py`); A1 asserts the define in `compile_commands.json`; A2 requires the
Fortran-line cell; variants 2–4 and the always-announce rule gated; E5's status lines need evidence;
D1's gate references the same-commit SP binary; the I/O precision report got an owner (B1); the
no-auto-mode divergence is declared in the Overview; the `ref0 → i4 → main` chain is stated in F2.
Rejected: none.
