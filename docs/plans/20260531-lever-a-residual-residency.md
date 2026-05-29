# Lever A — Residual Device-Residency (post-M5.13) — REVISED after plan-review

> Revised after the `planning:plan-review` agent found 6 blockers (all addressed below). Key change:
> the monthly-mean output accumulates `salt`/`density`/`temp` on the **host every step**, so S/density/T
> residency is capped by a per-step D2H **unless** mean accumulation moves to the device. The plan is now
> two-phase: clean wins first (fer_w/w_i, no mean entanglement), then **device-side mean accumulation**
> as the enabler for full S/density/T residency.

## Overview
Close the residual NG5 GPU↔CPU gap (today **1.41×**) by finishing device residency on the
**host-staged nod3D fields** — the root of the **44 % PCIe** chunk (`cudaMemcpy` 2.81 s/step;
`deep_copy` **11 GB/step**) the post-M5.13 re-profile identified as the biggest **climate-safe** lever.

Continuation of the M5.13 split-rail campaign into the deferred (`S` = g2) / missed (`density`,
`fer_w`, `w_i`) fields. Climate-safe throughout (Serial bit-identical flips, a bit-identical device clamp,
bit-identical device mean accumulation, or a surface-only trim).

**Why not the other levers** (measured tiny in the re-profile, explicitly OUT): CG 0.9 %, EVP+ice 2.65 %,
launches 0.2 % → no pipelined/CA-CG, no EVP-subcycle, no Kokkos-Graphs. Smoother (0.31 s/step) = future
Lever C. Figure: `docs/figures/m513_reprofile_levers.png`.

**⚠️ The mean-output entanglement (review BLOCKER 5, confirmed in code).** `fesom_io_init` sets up the
monthly MEAN streams whenever an output dir is passed (`fesom_main.cpp:982`), and the MEAN hot path
accumulates each var **every step** from the **raw host alias** (`fesom_io_stream.cpp:570-575` →
`resolve_salt`/`resolve_density`/`resolve_temp` at `fesom_io.cpp:622/689/...`). So a device-resident
S/density/T must be host-fresh every step for the mean to be correct — which is *literally why g1-T kept
T's full-field `sync_host`* (T's mean is valid only because of that D2H). Consequences:
- **fer_w / w_i are NOT output fields → clean full wins** (Phase 1).
- **S / density / T residency is capped by the mean's per-step D2H** unless we accumulate the mean on
  device (Phase 2). Device-mean-accumulation is **bit-identical** (per-element time-sum, no reordering)
  and removes the D2H for *all* output fields.
- The headline NG5 perf runs also pass an output dir → they accumulate means too, so even there the
  S/density win is bounded until Phase 2. (Task 3 re-measure quantifies this.)

**Branch:** fresh `m514-residency` off `m513-pcie @ 7ad27fd` (M5.13 was closed out). Confirm before first commit.

**Honest payoff:** Phase 1 banks the clean fer_w/w_i wins + validates the pipeline. Phase 2 (device-mean-
accum + full S/density/T) is the big-but-bounded-by-scope unlock — committed only after Task 3 confirms the
remaining S/density D2H is worth it. Parity is not promised; after Lever A the MPI/sync 38 % dominates (a
later B-overlap lever).

## Context (from discovery)
**Files:**
- `src/fesom_step.cpp` — exchanges **201** (`density`), **700** (`fer_w`), **722** (`w_i`), **839**+**901**
  (`S`); floor **923-942**; T `sync_host` **900**; S re-pushes (from review, to be grep-confirmed in Task 0):
  **174-176, 248-250, 359-360, 812-813, 822, 829-830, 836, 886-887, 891**; **w_i re-push 494** (cross-step).
- `src/fesom_halo_device.{hpp,cpp}` — `fesom_halo_field()`.
- `src/fesom_io.cpp` / `src/fesom_io_stream.cpp` — MEAN accumulation (`570-575`), resolvers (`622/689/730-745`),
  accumulator buffers — **the Phase-2 target**.
- `src/fesom_main.cpp` — `fesom_io_init` @982 (means always on w/ out_dir); init pushes (mirror T @1024-1026);
  pre-I/O snapshot sync; S diagnostics @458/478/772 (cosmetic).
- `src/fesom_ice.cpp` — `447`,`618` S re-pushes; `ocean2ice_kk`/`oce_fluxes_kk` read S on device (production).
- `src/fesom_sss_runoff.cpp` — **`407` `relax_salt` reads `S[surface]` on host**, `389` `virtual_salt`; gated
  on `sss_path`/`ref_sss_local`. **OFF in the gate + climate runs** → not exercised by current validation.
- `src/fesom_momentum.cpp:812` (`impl_vert_visc` reads w_i on device, cross-step), `src/fesom_gm.cpp:1579`
  (bolus reads fer_w on device, same-step), `src/fesom_tracer_diff.cpp` (trdiff = last S write).

**Patterns / lessons:** L48 split-rail + remove EVERY re-push + the I/O-staleness trap; L50 surface-reader
sync (gated); L57 init push for cross-step non-zero-init device fields; the NaN-poison discriminator;
L58 (the per-step gate is a tripwire, not a fidelity metric).

**Measurement (done):** nsys job `25237441` (6.42 s: PCIe 2.81 / MPI-sync 2.42 / kernels 1.19);
STEP_PROFILE job `25237442` (deep_copy 11 GB/step, 139 calls; 13_fct 18.1 %, 1_eos 5.7 %, 12_ale 5.1 %).

## Development Approach
Numerical-kernel port — "tests" = the validation ladder, per change. NO unit tests / TDD.

Per-change ladder:
1. **Serial `FESOM_KK_VERIFY=<key>` `max|Δ|==0`** (clamp + each flip Serial bit-identical).
2. **Serial pi np1 + np2 bit-identical** (`scripts/diff_snap.py`, DIRS; np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`).
3. **SYNCCHECK build clean** (`build-synccheck` — free now).
4. **CORE2-active-ice `scripts/gpu_fidelity_gate.sh --fresh-oracle`** (pi INSUFFICIENT — no ice).
5. **Perf:** same-day NG5 dist_16 re-time + `deep_copy` MB/step drop.

**⚠️ Two failure modes the gate CANNOT see (review BLOCKERs 5,6) — extra validation required:**
- **MEAN-output staleness:** the gate diffs snapshots only. Any flip/refactor of S/density/T MUST also diff
  the `.monthly.nc` **means** of a CORE2 run pre/post (the device-mean-accum refactor must be bit-identical
  there; the field flips must leave the means corr~1).
- **SSS restoring is OFF in the gate/climate runs** → the `relax_salt` host S read never fires. Keep the
  gated surface sync **defensively**; note that validating it needs a future SSS-active run (out of scope,
  flagged).

**Final fidelity verdict:** 1-yr CORE2 CUDA climate compare (`scripts/m32_climate_compare.py`) vs the
pre-Lever-A binary — snapshot AND mean — statistically identical (corr~1, bias O(1e-4)). Never the 20-step gate.

## ⚠️ Build constraint
`build-cuda` + `build-serial` are in use by the scaling sweep — do not rebuild until `squeue -u a270088`
is clear (check first). `build-synccheck` is free. Code edits + SYNCCHECK proceed now; Serial-verify / pi /
gate / NG5 builds wait for the sweep (or a scratch build dir). Build GPU with `source ./env_cuda.sh`.

## Implementation Steps

### Phase 0 — Audit (read-only, no commit)

#### Task 0: Complete the S audit + confirm the mean-stream behavior — ✅ DONE (2026-05-29)
**Files:** (read-only) `fesom_step.cpp`, `fesom_ice.cpp`, `fesom_sss_runoff.cpp`, `fesom_io*.cpp`, `fesom_main.cpp`
- [ ] Create branch `m514-residency` off `m513-pcie @ 7ad27fd` (per user) — deferred to first commit.
- [x] **COMPLETE S re-push list (grep-derived, values AND valuesold):**
      `values` — `fesom_step.cpp` 174-176 (EOS IN), 248-250 (GM IN), 359-360, 812 (FCT IN), 822 (FCT OUT),
      829 (Redi IN), 836 (Redi OUT), 886-887 (forcing IN), 891 (forcing OUT), + the two host exchanges
      839, 901; `fesom_ice.cpp` 447 (ocean2ice), 618 (oce_fluxes).
      `valuesold` — `fesom_step.cpp` 813 (FCT IN), 822 (FCT OUT), 830 (Redi IN).
      → 13 site-groups (the original hard-coded 6 would have clobbered — review BLOCKER 1/2 confirmed).
- [x] **Host-reader classification:** ALL S host reads are non-production-per-step:
      verify-twins (eos:156/711, gm:124, kpp:1252 — gated `FESOM_KK_VERIFY`; ice_coupling 67/79/272/289 =
      `*_verify`, gated `s_verify_icemap`/`s_verify_iceflux` — production = `ocean2ice_kk` ice.cpp:451 +
      `oce_fluxes_kk` ice.cpp:620, both DEVICE) · setup (phc:466, ic:68) · gated I/O (io:622/689) ·
      cosmetic diagnostics (main:458/478/772). **The ONLY real per-step production host readers are
      (a) SSS restoring and (b) the MEAN accumulation.**
- [x] **SSS restoring:** `fesom_sss_runoff.cpp:404-409` `relax_salt = surf_relax_S*(Ssurf - S[surface])`,
      gated `sr->sss_path[0] && sr->sss_month_loaded>0` → fires ONLY with an SSS-restoring file, which the
      gate + climate runs do NOT pass (grep of the jobs = none). → keep a *gated* surface S sync defensively
      (L50); it is NOT exercised by current validation (flagged for a future SSS-active run).
- [x] **MEAN vars:** `temp, salt, ssh, sst, sss, u, v, w, Kv, Av, density, bvfreq, a_ice, ...`
      (`fesom_io.cpp:728-748`), accumulated EVERY step (`fesom_io_stream.cpp:570`), raw host alias. NG5 perf
      runs pass out_dir → means on there too (so even the headline S/density win is bounded pre-Phase-2).

> **⚠️ Task-0 DISCOVERY (latent, pre-existing — beyond S).** The per-step MEAN accumulation reads MANY
> already-device-resident fields (`u, v, w, Kv, Av, bvfreq`) on the raw host alias, but those fields are
> `sync_host`'d only in the **snapshot-gated** block (`fesom_main.cpp:1295-1306`, `if snap_every>0 &&
> n%snap_every==0`). The mean (`fesom_io_step`, `fesom_main.cpp:1166`) runs unconditionally. So on
> non-snapshot steps (and entirely when `snap_every≤0`) **the current binary accumulates STALE host values
> for the 3-D `u/v/w/Kv/Av/bvfreq` means** — a latent correctness issue introduced by the M5.x residency
> campaigns, invisible to the surface-only climate validation (`temp`/`sst` are fine via T's line-900 sync;
> `salt`/`sss` are fine only because S is still host-staged today). **This makes Phase-2 device-mean-
> accumulation a CORRECTNESS FIX for the whole device-resident 3-D output class, not merely the S enabler.**
> Strong code-path evidence; worth a quick empirical confirm (dump a `u`-mean vs a fresh `u`) once builds are free.

### Phase 1 — Clean wins (no mean entanglement) + re-measure

#### Task 1: `fer_w` device-halo (12_ale) — clean full win
**Files:** Modify `src/fesom_step.cpp` (@700)
- [ ] `fesom_step.cpp:700` `fesom_exchange_nod3D(fer_w)` → `fesom_halo_field(NOD3D)`; remove the re-push.
      (Device consumer = bolus-apply `fesom_gm.cpp:1579`, **same step** → no cross-step, no init push.
      Host reads = `FESOM_KK_VERIFY=ale` C-twin only. Not an output field → no mean entanglement.)
- [ ] Validate ladder 1-5 (`FESOM_KK_VERIFY=ale`).

#### Task 2: `w_i` device-halo (12_ale) — clean full win, +remove the cross-step re-push
**Files:** Modify `src/fesom_step.cpp` (@722, **@494**)
- [ ] `fesom_step.cpp:722` `fesom_exchange_nod3D(w_i)` → `fesom_halo_field(NOD3D)`.
- [ ] **Remove the cross-step re-push at `fesom_step.cpp:494`** (w_i is consumed at substep 6
      `impl_vert_visc` `fesom_momentum.cpp:812` BEFORE produced at 12d → crosses the boundary; leaving 494
      clobbers). At step 1, substep 6 reads device w_i before 12d ever runs — benign because `use_wsplit=false`
      → w_i≡0 and Kokkos zero-inits the device View; state this explicitly (no init push needed, but verify).
- [ ] Validate ladder 1-5 (`FESOM_KK_VERIFY=vrhs`+`ale`).

#### Task 3: Re-measure + decide Phase 2
**Files:** (profiling) `jobs/job_nsys_ng5`, `jobs/job_ng5_prof`
- [ ] Re-run nsys + STEP_PROFILE; record the fer_w/w_i delta (s/step + deep_copy MB).
- [ ] Quantify the residual S/density mean-D2H (how much PCIe remains attributable to the per-step mean
      accumulation of S/density vs their halos). **Decision gate:** if the S/density win is worth the
      device-mean-accum scope → proceed to Phase 2; else stop Lever A here and move to B-overlap.

### Phase 2 — Device-mean-accumulation + full S/density/T residency (leading approach; gated on Task 3)

#### Task 4: Device-side mean accumulation (the enabler — AND a correctness fix per the Task-0 discovery)
**Files:** Modify `src/fesom_io_stream.cpp`, `src/fesom_io.cpp`
> Dual purpose: (1) enables full S/density/T residency with no per-step D2H; (2) FIXES the latent stale
> 3-D means of the already-device-resident `u/v/w/Kv/Av/bvfreq` (Task-0 discovery). Validate (2) by checking
> a device-resident field's mean is no longer frozen between snapshots.
- [ ] Make the MEAN accumulators (`s->accum[v]`) **device Views** (per var; host-only output fields keep host accum).
- [ ] Convert the device-resident-field resolvers (`resolve_salt`/`resolve_temp`/`resolve_density`/sst/sss/...)
      to launch a device kernel `accum_dev(i) += field_dev(i)` (host function launches the `parallel_for` — no
      device function pointers). Sync the accumulator host **once** at the period boundary (`fesom_io_stream.cpp:346/476`).
- [ ] **Validate (bit-identity of the refactor):** a CORE2 run with mean output → `.monthly.nc` **bit-identical**
      pre/post (no field flipped yet → pure no-op proof; per-element time-sum, same order). + SYNCCHECK + pi.

#### Task 5: S full residency (the g2 flip — biggest, atomic per L48)
**Files:** Modify `src/fesom_step.cpp`, `src/fesom_main.cpp`, `src/fesom_ice.cpp`
- [ ] Make BOTH `S.values_fld` AND `S.valuesold_fld` device-resident (the MFCT needs the coherent pair — g1-T lesson).
- [ ] Flip both S halos (`839`, `901`) → `fesom_halo_field(NOD3D)`.
- [ ] **Device salinity floor:** `S(i)=Kokkos::max(S(i),0.5)` over `myDim+eDim × nlevels_nod2D[n]-1`,
      placed AFTER the post-trdiff device halo (matches the current exchange-then-floor order → bit-identical;
      review-confirmed bounds). Guard `FESOM_NO_SFLOOR`. Replace the host loop `923-942`.
- [ ] **Remove the COMPLETE re-push list** (Task-0-derived, values AND valuesold; review set: step
      174-176/248-250/359-360/812-813/822/829-830/836/886-887/891 + ice 447/618). Partial = clobber (L48).
- [ ] **L57 init push** of BOTH `S.values_fld` AND `S.valuesold_fld` before the loop (mirror T @1024-1026;
      S is non-zero PHC init, read at step-1 EOS).
- [ ] **Gated surface SSS sync** (defensive, L50): keep a surface-only `sync_host(S)` guarded on
      `use_sr`/SSS-active (off in our runs → zero cost; on → correct for `relax_salt`). Note it's not gate-validated.
- [ ] Add S to the snapshot-gated pre-I/O `sync_host` (S is a snapshot output → I/O-staleness trap).
- [ ] (Phase-2 effect) the mean now reads S on device (Task 4) → **no per-step D2H**.
- [ ] Validate ladder 1-5 (`FESOM_KK_VERIFY=trdiff`) + **diff the `.monthly.nc` salt/sss means** + diff ALL snapshot fields.

#### Task 6: density full residency (1_eos)
**Files:** Modify `src/fesom_step.cpp` (@201), `src/fesom_main.cpp`
- [ ] `fesom_step.cpp:201` `fesom_exchange_nod3D(density)` → `fesom_halo_field(NOD3D)`; remove re-push.
      (Rationale corrected: density has **no on-device consumer** — its halo exists only for host I/O; the flip
      defers the every-step host-MPI/D2H, and with Task 4 the mean reads it on device → fully free.)
- [ ] Add density to the snapshot-gated pre-I/O `sync_host`.
- [ ] Validate ladder 1-5 (`FESOM_KK_VERIFY=eos`) + diff the `.monthly.nc` density mean.

#### Task 7: T D2H removal (now safe via Task 4)
**Files:** Modify `src/fesom_step.cpp` (@900)
- [ ] With the mean accumulating T on device (Task 4), the line-900 full-field `sync_host` is needed ONLY for
      the bulk SST host reader (`fesom_bulk.cpp:259`, surface row) → trim it to **surface-only**. (Pre-Task-4 this
      would have staled the 3-D T mean — review BLOCKER 5; now safe.) Watch the per-step alloc overhead the
      M5.9-pin surface trim hit; decide by the NG5 re-time.
- [ ] Validate ladder 1-5 + diff the `.monthly.nc` temp/sst means.

### Phase 3 — Verify + close

#### Task 8: Re-measure full residency
- [ ] nsys + STEP_PROFILE on the final binary; record PCIe / deep_copy / new GPU/CPU ratio vs 2.81 / 11 GB / 1.41×.
- [ ] If a meaningful residual remains (139 calls implies more host-staging), identify the next fields (per-field
      deep_copy instrumentation is buildable once the sweep is done) and decide extend-vs-stop (→ B-overlap).

#### Task 9: Final climate validation + docs
- [ ] 1-yr CORE2 CUDA climate compare vs pre-Lever-A — **snapshot AND `.monthly.nc` means** — corr~1, bias O(1e-4).
- [ ] Cross-mesh dars re-time (the M5.13 generalization check).
- [ ] Update `docs/SCALING_NG5.md`, `docs/GPU_FIDELITY.md`, `docs/KOKKOS_PORTING_LESSONS.md` (new lesson:
      the mean-accumulation/residency entanglement + device-mean-accum), the handoff, and the campaign memory.
- [ ] Move this plan to `docs/plans/completed/`.

## Technical Details
- **Device clamp:** elementwise `Kokkos::max(S,0.5)`, race-free, no reduction/scatter → bit-identical Serial AND
  CUDA. ~0.13 ms/step on-device vs the ~20 ms host round-trip it replaces. Review-confirmed bit-identical to the
  host floor at `923-942` (same index set, same exchange-then-floor order).
- **Device mean accumulation:** `accum_dev(i) += field_dev(i)` per var per step (host launches the kernel);
  bit-identical because each element's accumulator is an independent time-sum (no cross-element reordering).
  Sync host once per output period. This is the piece that makes output-field residency free.
- **S blast radius:** S (values+valuesold) is read on device by EOS/GM/FCT/Redi/trdiff/bc_surface + ocean2ice +
  oce_fluxes → the flip spans into `fesom_ice.cpp`; every re-push must go (L48).

## Post-Completion
*Informational — external/manual:*
- **SSS-restoring validation gap:** the gated surface S sync for `relax_salt` is not exercised by the gate/climate
  configs (SSS off). A future SSS-active CORE2 run would validate it. Flagged, not in scope.
- **Cosmetic:** the per-print host diagnostics (`fesom_main.cpp:458/478/772/1233`) read S/density on the raw host
  alias → console min/max go stale on device-residency (like T post-g1-T). Accepted cosmetic class; note for bisection.
- **Next lever:** after Lever A, MPI/sync (38 %) dominates → B-overlap (climate-safe interior/boundary overlap of the
  device-halo MPI); then the smoother (Lever C).
