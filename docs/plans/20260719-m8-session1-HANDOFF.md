# M8 session-1 handoff (2026-07-19) — mixed precision: P1+Gates 0-2 DONE, D1 PASSED → resume at Gate 3

**Read first:** `docs/plans/20260719-m8-mixed-precision.md` (the plan, checkboxes current) ·
`docs/PRECISION_ISLANDS.md` (islands + accumulation ledger + promotion log) ·
`docs/SP_PORTING_LESSONS.md` (SP1–SP9 — the traps this session paid for; SP3 changes how you
design EVERY run below).

## Where everything lives

- **Worktree** `/home/a/a270088/port_kokkos_mp`, branch `m8-precision`, base `1df683b`
  (= 63A/63B physics). NEVER touch `/home/a/a270088/port_kokkos` (speed session's checkout).
  Base-code reference worktree for byte gates: `/home/a/a270088/port_kokkos_base` @1df683b (built).
- **Commits** `816eded..c1df19b` (~25 M8 commits, all local — ASK BEFORE PUSH).
- **Frozen bins** `mp/bin/`: **mp64-1-serial 547941c9** · **mp32-2-serial 2a4bc3e9** ·
  **mp32-2-cuda** (+ mp64-0/-0-cuda = the Gate-0/2 FP64 basis; mp32-0/-1 RETIRED, buggy).
  ALWAYS `BIN=`-pin. Job templates: `jobs/job_mp_core2_serial`, `jobs/job_mp_dars_cpu`,
  `jobs/job_mp_dars_gpu` (BIN/TAG/DT/NSTEPS via `--export`; core2 needs
  `--ntasks-per-node` override when changing rank counts).
- **Run artifacts** `/work/ab0995/a270088/port2/mp/{task1,gate_*,t5,gate1,gate2,dbg}`.
  CUDA noise basis for the envelope gate: `gate_2h/{cuda,cuda_rerun}`.
- **Builds**: `build-mp-serial{,-sp}`, `build-mp-cuda{,-sp}`. Serial login smokes need the
  ob1/vader env override block (in every gate command below). ⚠️ NEVER edit src while a build
  runs (we paid for this twice).

## State (proven, committed)

1. **Gate 0**: fully-swept FP64 ≡ base bit-identical — pi np1+np2 AND CORE2 dist_8
   (ice+PHC+JRA, 20 steps); CGPIPE ON≡OFF byte-claim intact; CGPIPE+CGPOLY selfchecks
   0.000e+00; CGPOLY iters 42-44. Re-proven after every later fix.
2. **Gate 1**: SP builds run NaN-free, CPU+CUDA, banner-certified
   (`scripts/mp_assert_banner.sh <log> SINGLE`).
3. **Gate 2 + D1 (PASSED)**: CORE2 c1 CPU **1.51×** · dars c2 CPU **1.57×** · dars g2 GPU
   **1.47×** · device mem **0.51×** (23.2→11.9 GB) · CG iters +1. 300-step min-of-2 pairs,
   knobs-off posture.
4. **Two SP bugs root-caused+fixed** (both FP64-inert, re-gated): diag-Allreduce stack smash
   (`7e90742`, → SP1) and the **JRA absolute-Julian-date time axis** (`7247412`, → SP2; first
   `FieldT<dbl_t>` per-field promotion; PR-940's SP branch carries it latent — upstream-report
   material, user's call; JAX `jra55.py` same design, x64-protected).
5. **Instruments in tree** (all env-gated/inert): `FESOM_MP_NANSCAN=1` per-phase NaN scan
   (step/main/bulk probes, Serial host-alias, dies at first poisoned phase) ·
   `scripts/mp_cuda_gate.py` (envelope, K=10 floor 1e-13) · `scripts/mp_divergence_curve.py`
   (cross-dtype relL2/Linf curves + plots) · `scripts/mp_assert_banner.sh`.

## The gate recipes (verbatim-usable)

Serial byte gate (FP64 refactor checks — refs `task1/new{,_np2}` are the standing base outputs):
```
cd /home/a/a270088/port_kokkos_mp; source ./env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
unset OMPI_MCA_osc OMPI_MCA_coll OMPI_MCA_coll_hcoll_enable HCOLL_ENABLE_MCAST_ALL HCOLL_MAIN_IB UCX_NET_DEVICES UCX_TLS UCX_IB_ADDR_TYPE UCX_UNIFIED_MODE
# np2 additionally: export OMPI_MCA_btl_vader_single_copy_mechanism=none
./build-mp-serial/fesom_port /home/a/a270088/port2/fesom2/test/meshes/pi OUT 100 20 10
/work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/diff_snap.py REF OUT
```
CUDA envelope gate: srun `-p gpu-devel -A ab0995 --gres=gpu:1 -n1 -t 00:06:00` the cuda binary
on pi, then `mp_cuda_gate.py gate_refs/cuda_ref OUT gate_2h/cuda gate_2h/cuda_rerun`.
SP CORE2 login repro shape (the debugging workhorse): `mp32-2-serial
/work/ab0995/a270088/port2/mesh/core2 OUT 1800 <steps> 999999 <PHC> 1958` with
`FESOM_PRINT_EVERY=1`; PHC = `/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc`.

## NEXT — Gate 3 on the OPTIONS config (the endgame physics)

Options env (the 63A/63B config): `FESOM_MIX_SCHEME=TKE FESOM_WHICH_EVP=1 FESOM_ALE=zstar`
(GM on by default). ⚠️ ALL SP runs so far were KPP/linfs defaults — the options config is
UNTESTED at SP and contains the top suspects (cvmix_TKE = highest-probability promotion; mEVP).
First SP options smoke may fail → that's the failure protocol working (promote via registry).

1. **SP options bring-up** (cheap, login): np1 CORE2 SP with options env, ≥48 steps
   (SP3: must clear the 9-h forcing boundary AND rule-0.41 cold-start window). Expect NaN
   possibly — nanscan is armed for exactly this.
2. **Gate 3a divergence-vs-ensemble** (CORE2 np128 compute jobs, options config, ~1-day runs,
   SNAP so curves have ≥5 points): FP32 vs FP64 + ENSEMBLE of FP64 dt-seed controls
   (dt 1800.0000001 / 1800.00001 / 1800.001 — envelope = max over seeds; single seeds are
   non-monotone, proven). Judge with `mp_divergence_curve.py --envelope`.
3. **Gate 3b conservation**: needs a small src hook — env-gated FP64 global heat/salt/volume
   printout every N steps (design: dbl_t reductions over T,S,vol via areasvol×hnode; print
   `[fesom_port] CONSERV step= heat= salt= vol=`); then 1-month FP32-vs-FP64 drift compare.
   THE measurement PR-940 never made — paper centerpiece.
4. **Gate 3c solver health**: CG iters FP32 vs FP64 across a month (grep logs); CGPOLY variant
   (`FESOM_SPEED=1 FESOM_SPEED_FORCE_SERIAL=1 FESOM_SPEED_CGPOLY=3 FESOM_CGPOLY_SELFCHECK=1`)
   — does the Chebyshev selfcheck still print 0.000e+00 at SP? (Its byte-identity claim is
   WITHIN-run, so it should — verify, don't assume.)
5. **Gate 3k knobs at SP**: SPEED/CGPIPE (`FESOM_CGPIPE_SELFCHECK=1`), CGPOLY d3, EVPWIDE —
   Gate-3-class runs each, both backends where relevant.
6. **Restart round-trip gate (STILL OPEN, Task 6 checkbox)**: write SP restart → re-read SP
   (bit-compare) + read into FP64 build (exact embed). Check the reader path exists first.
7. Then **Gate 4**: 1-yr FP32 CUDA, 63A posture, `m7_climate_check_plots` vs
   `/work/ab0995/a270088/fesom2_core2` + the FP64 twin at the M5.23 bar
   (sst 1.00000/sss 0.99996/ssh 1.00000/a_ice 0.99997 class). **63A/63B are DONE**
   (41/52 complete yr, `/work/ab0995/a270088/port2/climate63/`) — the twins are ready.
8. Then **Gate 5**: 63-yr "63C-MP" (63A posture, no CGPOLY), bars pre-registered in the plan
   (Tbar ≤0.001 °C, OHC ≤~10× port↔Fortran gap, co-track/flatten) — re-affirm numbers against
   the final 63A harvest WITH THE USER before submitting (12-h 2N job).

## Traps for the next session (earned today)

- **SP3 run-length illusion**: nothing counts as "passing" unless it cleared the first forcing
  record boundary and the cold-start window. 5-step smokes prove NOTHING at SP.
- **SP1 invariant** (grep-enforceable): `MPI_DOUBLE` only over `dbl_t` storage.
- Job state COMPLETED ≠ run success — grep `rep_ rc=` and FATAL in every fleet epilogue.
- Perf pairs: FP64 numbers banked (board above) remain the pair basis as long as the FP64
  build stays bit-identical (re-prove after any src change — the pi gate is 2 minutes).
- The it= step-diag line: `T[+1e30,-1e30]` sentinels = all-NaN field (SP7).
- dt-seed single controls are non-monotone — envelopes need ≥3 seeds (Gate-3a design).

## Standing user decisions

All four prizes in scope; islands extensible with registry evidence; anomaly/increment work =
parked M9 (NOT M8); half precision = parked pathway (per-field only); mixed precision remains
BANNED on the main/m7 line — M8 branch only; merge-back only after M7 settles; ask before push.
