# Project state — 2026-05-28 (end-of-day snapshot)

A clean reference for "what's good", "what's stale", "what we compare to". Created
during the M5.11 profile pass after a stale-oracle false alarm on the GPU fidelity
gate — see § The 2026-05-28 cleanup.

## What we have, at climate-close vs Fortran (the bottom line)

| backend | last validated against | status |
|:--------|:-----------------------|:------:|
| Kokkos Serial (build-serial) | C twin (`/home/a/a270088/port2/fesom2_port`) | **bit-identical** (per-kernel + 1-yr CORE2) |
| Kokkos OpenMP (build-omp, OMP_NUM_THREADS=1) | Kokkos Serial | **bit-identical** at 1 thread |
| Kokkos CUDA (build-cuda, master) | Kokkos Serial | **climate-close** (worst 1.0e-2 on h_ice @ 20-step CORE2 dist_8, today_serial as oracle) |
| Kokkos CUDA M3.2 (1-yr CUDA, 2-yr OMP) | Fortran-KPP + C-port-KPP | **PASS** (M3.2 closed `466ea3e`; corr 1.0 on every ocean field, scatter-drift floor on ice) |

**Confirmed today**: ocean/ice on CUDA reproduces the C twin at the canonical
CUDA climate-close floor. The 20-step gate result matches the M5.9-pin claim
(worst-case T ≈ 6e-4 vs the 1e-3 budget memory recorded).

## Canonical references (the things we COMPARE TO)

### Long-timescale (climate validation, M3.2+)

Per [docs/REFERENCE_RUNS.md](REFERENCE_RUNS.md) — KPP default:

- **Fortran-KPP**: `/scratch/a/a270088/fortran_kpp_5yr_fix` (5 yr, 1958–1962)
- **C-port-KPP**: `/work/ab0995/a270088/port/kpp_5yr_fix` (5 yr, monthly means,
  C-port commit `6ecabe8`)
- Comparison driver: `scripts/m32_climate_compare.py`.

### Short-timescale (the M5.9 GPU fidelity gate — 20-step CORE2 dist_8)

- **Serial oracle** (the OUTPUT of build-serial @ master HEAD):
  `/work/ab0995/a270088/port2/kokkos_gpu_runs/serref_core2/snap_000020.nc`
  - **Rebuilt 2026-05-28** from `master @ 466ea3e` via `jobs/job_today_serial`
    (then promoted into the gate's path).
  - The previous build (from `master @ c2fa25e`, 2026-05-27 20:46) was renamed
    to `serref_core2.STALE_pre_m59fix/` — see § The 2026-05-28 cleanup.
- **CUDA gate-pass exemplar** (the OUTPUT of build-cuda @ M5.9-pin):
  `/work/ab0995/a270088/port2/bisect/m59pin_final/snap_000020.nc`
- Comparison driver: `scripts/gpu_fidelity_gate.sh` (uses the cached oracle;
  pass `--fresh-oracle` to rebuild via `jobs/job_core2_serial_ref`).

### Today's same-day baselines (the M5.11 cleanup)

Built 2026-05-28 18:30–19:15, all from `master @ 466ea3e`:

- `/work/ab0995/a270088/port2/today/serial/` — Serial 20-step CORE2 dist_8 (= the new oracle)
- `/work/ab0995/a270088/port2/today/omp/` — OMP 20-step CORE2 dist_8 (OMP_NUM_THREADS=1, byte-identical to Serial)
- `/work/ab0995/a270088/port2/today/cuda/` — CUDA 20-step CORE2 dist_8 (climate-close to today_serial, worst 1.0e-2 on h_ice)

Jobs: `jobs/job_today_{serial,omp,cuda}` — used these same-day to validate after a stale-oracle false alarm.

## What's STALE (do not use; keep for forensics only)

| path | why stale |
|:-----|:---------|
| `/work/.../kokkos_gpu_runs/serref_core2.STALE_pre_m59fix/` | Built 2026-05-27 20:46 from build-serial against master @ `c2fa25e` (pre-M5.9-FIX). After the M5.9 FIX (`6ba27e9`) and M5.9-pin (`05182aa`) added/dropped `sync_host()` calls (no-ops on Serial), recompilation produced subtly different fp ordering on 5 cells at node 125225, levels 17–21 (compiler reordering of the host-write/host-read paths). The cells used to be 0 in the cached oracle but have small real values today. Gate run today against the STALE oracle showed max\|Δ\|=1.28 (false fail); fresh oracle re-runs at 6e-4 (real climate-close floor). |
| Deprecated long-timescale refs | See [docs/REFERENCE_RUNS.md § Deprecated references](REFERENCE_RUNS.md) — the handout's pre-KPP-flip / pre-`ice_gamma_fct` refs. |

## The validation ladder (what gates what)

```
PHYSICS PARITY                                          (M3.2 long runs)
   ↑
   CUDA-vs-C-port-KPP    (corr 1.0, bias ~1e-4 ocean, ~1e-3 ice)
   ↑
   CUDA-vs-Serial-Kokkos (climate-close, ~1e-3 floor at 20 steps)   ← short-run GATE
   ↑
   Serial-vs-C-port      (BIT-IDENTICAL — per-kernel verify + 1-yr CORE2 acceptance)
   ↑
   Per-kernel verify on Serial pi (FESOM_KK_VERIFY=<k> max|Δ|==0 for 12 ocean keys + 5 ice keys)
   ↑
   Build correctness (build-serial, build-omp, build-cuda, build-synccheck all green)
```

A regression at ANY level is caught here. Today's bisect: levels 1–4 pass; level 5 (M3.2) is closed `466ea3e` — all green.

## The 2026-05-28 cleanup (this section's reason for existing)

During M5.11 profile pass execution today, the GPU fidelity gate FAILED with
worst-case T=1.28 (4 fields over ceiling). Investigation showed:

1. **Master had only docs changes since M5.9-pin** (`466ea3e` is `05182aa` + M3.2
   docs commit, no code change).
2. **M5.11-a's instrumentation is 100% inert when `FESOM_STEP_PROFILE` is unset**
   (`install_callbacks()` early-returns, no Kokkos profiling hooks registered).
3. **Master binary with my M5.11-a changes vs without — same numbers** to the
   last digit, confirming inertness.
4. **OMP and CUDA M3.2 long runs from earlier today both passed** (closed `466ea3e`).

So the failure had to be in the CACHED ORACLE — which it was. Diagnosis:

- Cached oracle built 2026-05-27 20:46 from build-serial @ master `c2fa25e`.
- Today's build-serial @ master `466ea3e` produces output that differs from the
  cached oracle at 5 cells (node 125225, levels 17–21) — all clustered at the
  same node, all at the bathymetry transition.
- Bit-identical OMP today (`OMP_NUM_THREADS=1`) confirms current build-serial
  is the ground truth. Today's CUDA at climate-close 6e-4 vs today_serial
  confirms the CUDA path is healthy.
- **Cause: compile-time fp reordering** from M5.9 FIX / M5.9-pin adding/dropping
  `sync_host()` calls in `fesom_step.cpp` — no-ops on Serial at runtime, but the
  surrounding source changes were enough for the optimizer to reorder some host
  loads/stores, producing ULP-level differences in masked-cell I/O writes.
- **Fix**: replace cached oracle with `today_serial` (fresh build from master HEAD).

**Lesson (add to KOKKOS_PORTING_LESSONS.md, L51):** the GPU fidelity gate's
*cached Serial oracle has a freshness budget* — any source change that touches
fesom_step.cpp (even no-ops on Serial like sync_host calls) can invalidate it
via compile-time fp reordering. Rebuild the oracle whenever:

- A src/ commit touches `fesom_step.cpp` or anything called from it, OR
- The build env modules change (compiler, openmpi, netcdf), OR
- The gate fails unexpectedly on a binary that should be inert.

The script `scripts/gpu_fidelity_gate.sh --fresh-oracle` rebuilds it automatically.

## Open items (heading into next session)

- M5.11 profile pass is mid-flight: C1 committed (`366887c` on `profile-m511`),
  R1abc + R2 done, R3 (ncu) waits for top-5 selection from R1c. PROFILE_M59.md
  has R1a/R2a data; needs R1b/R1c added + the top-5 ncu picked + the M5.12 lever
  synthesised.
- The M3.2 climate validation closed today (`466ea3e`) is the **canonical
  fidelity status** going forward; M5.11 does not affect numerics.
