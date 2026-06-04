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

## 2. Output — lean daily means (DONE, task #10)

Per the run spec we do NOT need the full 70 GB snapshot — only 8 lean fields, **daily**:
`sst, sss, u_surf, v_surf, u_100m, v_100m, a_ice, m_ice`. Implemented on the existing
time-mean stream infrastructure (`fesom_io_stream` + `fesom_io_config`), which already does
daily means + per-var NetCDF with a time-unlimited dim:
- Added 4 resolvers (host + **device** — uv is device-resident) for surface (layer 0) and
  ~100 m (layer nearest 100 m via mesh Z) velocity, as `FESOM_VAR_2D_ELEM`; registered in the
  default table. `sst/sss/a_ice/m_ice` already existed.
- Added `FESOM_IO_EXCLUSIVE` so a config writes ONLY its listed vars (and, with no config,
  NOTHING — the clean timing-run mode). Job scripts now default `snap_every=-1` +
  `FESOM_IO_EXCLUSIVE=1` so timing runs write no output (the old default monthly means +
  step-0 snapshot were filling scratch — 9 TB cleaned).
- Config: `io_lean_daily.conf`. Run with `FESOM_IO_CONFIG=$PWD/io_lean_daily.conf`.
- Verified (core2, 100 steps → 3 daily records): sst [-1.90,30.04]°C, u_surf/v_surf
  [-1.05,1.19] m/s nonzero (device resolver correct). Output: `<out>/<var>.fesom.<yr>.daily.nc`.
- ⚠️ TODO polish: `u_100m/v_100m` need masking at elements shallower than 100 m (currently a
  sentinel ~100 there); guard the resolver with `mesh->nlevels_fld` (device IntField available).
- Full 70 GB snapshot OOM fix (gather→write→free per field) deferred — not needed for this run.

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
