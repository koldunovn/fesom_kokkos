# Next-session prompt — closing the residual CPU↔GPU gap (post-M5.13)

> Copy-paste this whole file to start the session. It is self-contained; the pointers at the
> end have the detail. Suggested flow (the user's standard): read this → load memory →
> `brainstorm:do` the lever choice **after** the re-profile → `planning:make` the chosen lever.

---

## Where we are (read first — this is the premise)

The **M5.13 device-residency campaign is COMPLETE + climate-validated** (branch `m513-pcie`). It
flipped the remaining host-staged nod3D/elem3D halos to the on-device GPU-aware-MPI path
(`fesom_halo_field`) + made `uv` and tracer `T` fully device-resident. Result:

- **NG5 dist_16: 16.27 → 6.12 s/step (−62 %); node-for-node GPU/CPU 3.76× → 1.41×.**
- **dars: 4.10× → 1.60×** (denser dist_8 → 1.52× — the data/compute-balance probe).
- **PCIe `cudaMemcpy` 12.74 → 2.83 s/step (share 75 % → 44 %).**
- **Climate: zero regression** — 1-yr CORE2 CUDA on the campaign binary is statistically identical
  to the pre-campaign binary (sst corr 1.00000/bias 1.2e-4; see `docs/GPU_FIDELITY.md` § Climate validation).

**The gap is now ~1.4× and — crucially — it is NO LONGER ONE WALL.** The old story (PCIe = 75 %)
is gone. The nsys-final NG5 dist_16 step (6.43 s/step traced) now splits into three *comparable*
chunks:

| chunk | s/step | share | what it is |
|------|-------:|------:|------------|
| PCIe `cudaMemcpy` | 2.83 | **44 %** | the host↔device transfers the campaign did NOT remove (composition UNKNOWN — must profile) |
| MPI / sync / host | ~2.41 | **37 %** | inferred (step − PCIe − kernels); dominated by the iterative solvers' per-iter comms |
| GPU kernels | ~1.19 | **19 %** | campaign-invariant (Approach B = same kernels); memory-leaning per `ncu` |

**Data/compute-balance finding (L58):** once PCIe is gone the ratio is a function of *per-rank work*
— more work per GPU → nearer parity, MONOTONE (dars 197k nod2D/rank = 1.60×, dars 395k = 1.52×,
NG5 462k = 1.41×) — but it ASYMPTOTES ~1.4×, the residual being per-step launch/MPI overhead
(~720 kernel launches/step + MPI). The 8/16/32-node strong-scaling sweep (`docs/SCALING_NG5.md`
§ high-node scaling, `jobs/submit_m513_hinode_scaling.sh`) maps the drift: as nodes go up, work/GPU
falls and the ratio creeps from 1.41× toward the floor (early: NG5 8N ≈ 1.49×).

## The mandate

**Explore how to close the residual ~0.4× (1.4× → toward parity).** This is exploratory, not a
pre-decided implementation — the breakdown SHIFTED, so:

## Step 1 (MANDATORY, before choosing a lever): RE-PROFILE the post-campaign step at NG5

L56's hard-won rule: **the per-phase wall timer cannot separate compute from data-movement —
the decisive instrument is an `nsys` CUDA trace (kernels vs memcpy vs MPI).** Do this first; do
NOT assume the table above is still the breakdown after any change.

- `jobs/job_nsys_ng5` (NG5 dist_16, rank 0, ~8 steps, snapshots off). Compare to the campaign-final
  trace (job in `docs/SCALING_NG5.md` § M5.13). Re-confirm the 44/37/19 split on the CURRENT binary.
- **Drill into the residual 2.83 PCIe (44 %) — this is the biggest single unknown.** What is still
  copying? Candidates: the deferred **g2 salinity floor** (S is host-authoritative for the clamp →
  a full nod3D round-trip every step — potentially a large, easy win now that the big fields are
  flipped), the L50 surface syncs (`uvnode`/SST/SSS — small), the GM-chain leftovers / `ssh_rhs` /
  tracer-diff, and the pre-I/O snapshot syncs (gated off in the timed runs). `FESOM_STEP_PROFILE=1`
  prints `deep_copy calls/step + MB/step` — use it to attribute the 2.83.
- **Quantify the MPI/sync 37 %.** nsys shows `cudaDeviceSynchronize` + the MPI calls. The suspects
  are the iterative solvers: the **CG (~219 iters/step**, each a SpMV halo + 1–2 Allreduce) and the
  **EVP (120 subcycles**, each a uice/vice halo). 219 + 120 small latency-bound messages/step is a
  lot. Measure how much of the 2.41 is CG vs EVP vs substep halos.

## The candidate levers (rank them by the profile, then brainstorm → plan)

### Lever A — finish device residency (attack the residual PCIe, 44 %)
- **g2 salinity-floor → device clamp** (deferred in M5.13). If the profile shows S round-trips a full
  nod3D field every step, a device-side clamp removes a big chunk. Risk: the floor pins S's OUT-rail
  (L39) and S feeds the L50 SSS host reader (`sss_runoff`) — so it needs the same "device-resident +
  targeted surface sync" pattern as g1-T's bulk-SST fix. Bounded, well-understood.
- The other leftovers (GM-chain `fer_gamma` read-at-halo, `ssh_rhs` nod2D, tracer-diff) were judged
  low-payoff at a–f; re-check at 2.83.
- Risk profile: LOW (the L48 split-rail recipe, Serial byte-identical, the gate catches staleness).

### Lever B — cut the iterative-solver communication (attack MPI/sync, 37 %) ← likely highest upside
- **CG iteration count (~219) is high.** A cheap preconditioner (block-Jacobi / diagonal already?)
  or a better one could cut iters → less MPI AND less compute. ⚠️ Changes numerics → breaks Serial
  bit-identity → needs the FULL climate revalidation (not just the gate).
- **Pipelined / communication-avoiding CG**: overlap the Allreduce with the SpMV to hide the 219
  Allreduce latencies. Numerically a reassociation → climate-close (like the existing scatters), not
  bit-identical. Lower risk than changing the preconditioner.
- **EVP subcycles (120)**: fewer = less comms but changes ice stability — risky, probably leave alone.
- **Overlap compute + halo** (start halo, compute interior, finish boundary): a general win but a
  bigger refactor.
- Risk profile: MEDIUM–HIGH (numerics) but the BIGGEST potential reduction.

### Lever C — kernel coalescing / memory-layout refactor (attack compute, 19 %)
- The long-shelved `fesom_field.hpp` rank-1 → `View<double**>` refactor + kernel tuning. Shelved
  when compute was 7 %; now it's 19 %, so a 2× kernel speedup ≈ −10 % step. `ncu` showed the hot
  kernels (FCT, smoother, momentum) are memory-leaning (SOL Memory ~50–58 %) → coalescing should help.
- Same math, just faster memory access → **climate-close-safe** (the safest non-trivial lever).
- Risk profile: LOW numerically, but INVASIVE (touches the data layer) → its own branch + session.

### Lever D — the config/balance lever (free, no code)
- The scaling sweep proves more work/GPU → nearer parity. **Deployment guidance, not a code change**:
  for production, don't over-decompose (fewer nodes, more work each) — run near the A100-80GB ceiling
  (~900k nod2D/rank fits; the dist_8 NG5 point). State the recommended per-rank-work operating point.

## Validation ladder (unchanged — but mind which levers need MORE)
Per change: (1) Serial `FESOM_KK_VERIFY=<key>` max|Δ|==0; (2) Serial pi np1+np2 bit-identical
(`scripts/diff_snap.py`, DIRS; np2 needs `OMPI_MCA_btl_vader_single_copy_mechanism=none`);
(3) SYNCCHECK clean; (4) the MANDATORY CORE2-active-ice `scripts/gpu_fidelity_gate.sh --fresh-oracle`
(pi is INSUFFICIENT — no ice). **⚠️ Levers B/C that change numerics will FAIL Serial bit-identity by
design** — for those the bar is the **1-yr CORE2 CUDA climate compare** (`scripts/m32_climate_compare.py`,
apples-to-apples vs the pre-change binary with the SAME script, L58) showing corr~1 / no new drift, NOT
the per-step gate. Build with `source ./env_cuda.sh` ([[reference-cuda-aware-mpi]]).

## Hard constraints / gotchas (do not relearn these)
- Build GPU with **`source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware); env.sh's 4.1.2
  SEGFAULTs on device pointers.
- **MUST stay host:** salinity floor (unless Lever A ports it), `uvnode`→bulk wind, SST/SSS surface
  readers (L50), `eta_n` map, all `FESOM_KK_VERIFY` C-twin exchanges, setup-time exchanges.
- A field made device-resident across the step boundary with non-zero init needs a **one-time init
  push** before the loop (L57). A PARTIAL flip clobbers — remove EVERY re-push (L48).
- **A per-step gate's worst-cell |Δ| is a tripwire, not a climate metric (L58)** — settle fidelity
  with the multi-year climate compare, never the 20-step gate.
- **Output → `/work/ab0995/a270088/port2/…`, NEVER `$HOME`** (60 GB home quota). NG5 needs
  `snap_every=-1` (rank-0 gather OOMs ~66 GB).
- Same-day perf baseline only (rebuild + run the prior commit in the same session; don't compare to a
  memory-recorded number from another week).

## Pointers
- Status/result: `docs/SCALING_NG5.md` § M5.13 + § dars cross-mesh + § high-node scaling;
  `docs/GPU_FIDELITY.md` § M5.13 + § Climate validation; lessons **L48, L50, L56, L57, L58**.
- Campaign memory: [[project-m513-pcie-campaign]]; the flip plan `docs/plans/20260530-m513-pcie-residency-tasks.md`.
- Profilers: `jobs/job_nsys_ng5` (the decisive trace); `FESOM_STEP_PROFILE=1` (per-substep wall +
  deep_copy proxy); `src/fesom_profile.{hpp,cpp}`. Scaling: `jobs/submit_m513_hinode_scaling.sh`,
  `scripts/plot_scaling.py`.
- The on-device halo machinery: `src/fesom_halo_device.{hpp,cpp}`, dispatch `fesom_halo_field()`.
- The CG (Lever B target): `fesom_ssh.cpp` (CG_kk); the EVP: `fesom_ice_evp.cpp` (120-subcycle loop).
- Lever C target: `src/fesom_field.hpp` (rank-1 storage → `View<double**>`).

## Bottom line for the next session
PCIe is solved; the gap is ~1.4× and three-way. **Profile first** (the wall moved). Then: Lever A
(g2 S-floor) is the safe quick check on the residual PCIe; **Lever B (CG/EVP comms) is the highest
upside** but needs climate revalidation; Lever C (coalescing) is the safe, invasive, well-scoped
compute lever. Don't expect a single 3.8×→1.4×-style win again — this is the long tail, so measure
the upside of each lever before committing, and be honest if a lever isn't worth its risk.
