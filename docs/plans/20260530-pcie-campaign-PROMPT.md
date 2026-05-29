# Next-session prompt — M5.13 NG5 device-residency / PCIe-reduction campaign

Copy everything below the line into the next session.

---

We're starting **M5.13 — the NG5 device-residency / PCIe-reduction campaign** on the FESOM2
C→C++/Kokkos port (`/home/a/a270088/port_kokkos`, branch `m512-fusion @ fc156c3`). Start by
making a new branch `m513-pcie` off it.

**Context (the mandate).** An `nsys` CUDA trace on the production mesh NG5 (7.4 M nodes, dist_16)
proved the GPU step is **PCIe-data-movement-bound**: GPU kernels are only **7 %** of the step,
**~75 %** is full-field host↔device `cudaMemcpy` from host-staged halos (`cudaMemcpy` = 90.6 % of
CUDA API time, ~4 575 transfers/step). The lever is **DEVICE RESIDENCY** — flip the remaining
host-staged nod3D/elem3D halos to the on-device GPU-aware-MPI path (`fesom_halo_field`) and
eliminate host-op syncs with a viable device twin. Lever C (memory coalescing) and launch-fusion
are NOT the lever (they touch the 7 %, not the 75 %).

**Read these first, in order (they are load-bearing):**
1. `docs/plans/20260530-pcie-residency-campaign.md` — **the full handout**: the infra, the four
   flip patterns, the ranked target worklist (Tier 1/1b/3), what must stay host, the validation
   gate, build/run/profile/measure, the traps, and the M5.13a-g milestone sequence. This is your
   playbook.
2. `docs/SCALING_NG5.md` § "nsys decomposition" + § "the production lever is DEVICE RESIDENCY,
   not Lever C" — the charter and the evidence; `docs/figures/nsys_ng5_breakdown.png`.
3. `docs/KOKKOS_PORTING_LESSONS.md` — L56 (the corrected NG5 finding), L47 (the GPU-aware-MPI
   halo machinery + why some halos didn't pay), L48 (the four flip patterns + the I/O-staleness
   trap), L50 (the NaN-poison discriminator), L36 (when a field stays host), L51 (re-bake the gate
   oracle). `docs/SYNC_MAP.md` for the rails.
4. `src/fesom_halo_device.hpp` (`fesom_halo_field` @ :94), `src/fesom_field.hpp` (the sync rails,
   `h_checked` @ :85), `src/fesom_step.cpp` (the substep driver + the targets + `FESOM_KK_VERIFY`
   dispatch @ :137), `src/fesom_main.cpp:1283` (the pre-I/O sync gate).

**The work.** Execute the milestones in §9 of the handout, starting with **M5.13a (`cfl_z`)** as
the lowest-risk proof of the per-flip loop, then M5.13b (EOS hpressure/sw_alpha/sw_beta), then the
GM quartet, etc. For each flip: apply the L48 split-rail recipe (replace OUT-rail+halo with
`fesom_halo_field`, REMOVE the downstream IN re-push, keep genuine host-reader syncs); if it's a
snapshot-output field add it to the `fesom_main.cpp:1283` pre-I/O sync block (the I/O-staleness
trap, L48). Use the NaN-poison discriminator (`jobs/job_poison_dev`, L50) before deleting any sync
you think is redundant.

**Validate every flip (handout §6):** Serial per-kernel `FESOM_KK_VERIFY=<key>` max|Δ|==0 +
Serial pi np1+np2 bit-identical (`diff_snap.py`, dirs not files; np2 needs the vader override) +
SYNCCHECK clean + CUDA A/B at the run-to-run noise floor + **the mandatory CORE2-active-ice
`scripts/gpu_fidelity_gate.sh` (pi is INSUFFICIENT — that's how the M5.9 bug hid)**. Diff ALL
output fields, never a subset. Expect to re-bake the oracle (`--fresh-oracle`, L51) since we edit
`fesom_step.cpp` heavily. Build `build-cuda` with `source ./env_cuda.sh` (CUDA-aware MPI; env.sh
SEGFAULTs on device ptrs).

**Measure (handout §7):** per-flip CORE2 dist_8 same-day before/after (internal loop timer; many
of these halos are cheap at CORE2, so CORE2 may be flat — that's expected). The REAL acceptance is
the NG5 re-trace (`jobs/job_nsys_ng5` — PCIe % down) + `jobs/submit_ng5_scaling.sh` (the 3.8×
GPU/CPU ratio down). Write all output to `/work/ab0995/a270088/port2/…`, never `$HOME`.

Keep `docs/KOKKOS_PORTING_LESSONS.md`, `docs/GPU_FIDELITY.md`, `docs/SCALING_NG5.md`, and the
handoff current as you go. Confirm the plan with me before the first commit, then proceed
milestone by milestone, reporting payoff after each.
