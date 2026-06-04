# ng5 multi-month run on JUPITER GH200 — working notes (branch `ng5-long-run`)

Goal: run ng5 (7.4 M nodes) for a few simulated months on GH200, chained across the 12 h
walltime via restart, with periodic snapshots. Needs: a stable dt, restart, and ng5-capable
snapshot output.

## 1. Timestep stability (RESOLVED — 2026-06-04)

Probes: ng5, 8 nodes (32 GPU), JRA55 1958 + PHC IC, watched T/S every 25–50 steps.

| dt (s) | steps run | sim time | result |
|:------:|:---------:|:--------:|:-------|
|   60   |   1500    |  25 h    | ✓ stable (CG ~20 it) |
|   90   |   1000    |  25 h    | ✓ stable (CG ~32 it) |
|  180   |    500    |  25 h    | ✓ stable (CG ~70 it) |
|  240   |   ~200    |  13 h    | ✗ **FAILED — `CG_kk residual diverged`** |
|  300   |   ~150    |  12 h    | ✗ **FAILED — `CG_kk: pp·App is -nan`** |

**The instability is the SSH (barotropic) conjugate-gradient solver diverging at large dt** —
a CFL/stiffness limit, not ice or tracers. T/S stay bounded right up to the abort. This is
almost certainly the same wall hit on Levante ("couldn't pass ~100 steps"): at too-large a dt
the CG blows up after O(100–200) steps.

**Decision: run the multi-month experiment at dt = 180 s** (well inside the stable envelope;
500 steps = 25 h proven clean; use 120–150 for extra margin over months if drift appears).
dt=240 ("production") is NOT viable as-is — it would need barotropic-CG stabilisation
(better preconditioner / tolerance, or a split-explicit barotropic mode). That's a separate
task; flagged in `docs/JUPITER_PORT_AND_SCALING.md` so the paper's SYPD@240 column carries the
caveat. Stable production throughput is **SYPD@180 ≈ 3.5 at 32 nodes**.

A few-months run at dt=180: 1 month ≈ 14,400 steps; 3 months ≈ 43,200 steps ≈ 1.7 h wall at
32 nodes (0.14 s/step) — fits one 12 h job, but restart is still wanted for safety + chaining.

## 2. Snapshot output — ng5 OOM (TO DO, task #10)

`fesom_io_write_snapshot` allocates ALL ~10 global 3-D fields on rank 0 at once (g_T/S/w +
g_uv on elements ~16 GB + dens/bv/pgf/Kv/Av) ≈ 70 GB → OOM on the step-0 snapshot. The step-0
snapshot at line `fesom_main.cpp:1023` is why ng5 runs use `snap_every` huge today.
**Fix:** define the NetCDF file first, then **gather → write → free one field at a time**
(peak ≈ one field, ~24 GB). Format unchanged. (Optional `FESOM_SNAP_LEAN` to drop the 6
diagnostic 3-D fields → smaller files for long runs.)

## 3. Restart (TO DO, task #11)

No restart today. Plan: **per-rank binary checkpoints** (each rank writes/reads its local
arrays — no gather, no OOM; requires same node/rank count on resume — fine for chaining). Save
the prognostic state + AB2 carry-overs + step + calendar:
- tracers: T, S (values, valuesAB, valuesold)
- dyn: uv, uv_rhsAB, eta_n, d_eta, ssh_rhs, ssh_rhs_old
- ice: a_ice/m_ice/m_snow (+ _old), uice/vice (+ _old), EVP sigma11/12/22
- meta: step number, calendar date (so JRA55 resumes at the right time)
Knobs: `FESOM_RESTART_DIR`, `FESOM_RESTART_EVERY=N`; auto-resume if a checkpoint is present.
Verify: N steps straight vs N-with-restart-in-the-middle match to round-off.
