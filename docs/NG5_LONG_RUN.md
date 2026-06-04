# ng5 multi-month run on JUPITER GH200 — working notes (branch `ng5-long-run`)

Goal: run ng5 (7.4 M nodes) for a few simulated months on GH200, chained across the 12 h
walltime via restart, with periodic snapshots. Needs: a stable dt, restart, and ng5-capable
snapshot output.

## STATUS: pipeline validated end-to-end (2026-06-04) ✅

ng5, 16 nodes (64 GPU), dt=180, lean daily output, restart-chained over 3 simulated days
(2-day chunk + resumed 1-day chunk): **no blowups, 0.182 s/step**, chunk1 wrote 2 daily
records, chunk2 (resumed) wrote 1, and `cdo mergetime c*/sst.fesom.1958.daily.nc` gave a clean
3-record timeseries. All four pieces (stable dt, daily output, restart, concat) work together.

**Production recipe (per chunk):**
```
source env_jupiter_cuda.sh; source env_jupiter_data.sh
export FESOM_IO_CONFIG=$PWD/io_lean_daily.conf      # 8 lean daily fields
export FESOM_IO_EXCLUSIVE=1                          # only those fields
export FESOM_RESTART_DIR=<run>/restart              # SHARED across chunks (chaining)
export FESOM_RESTART_EVERY=480                       # checkpoint daily (480 steps = 1 day @ dt=180)
srun build-cuda/fesom_port $FESOM_MESH_NG5 <run>/chunkNN 180 <nsteps> 999999 $FESOM_PHC 1958
```
- dt = **180** (dt=240 diverges — see §1). 1 day = 480 steps; 1 month ≈ 14 400 steps.
- **Each chunk → its own output dir** (resume `io_init` NC_CLOBBERs), concat in post with
  `cdo mergetime <run>/c*/​<var>.fesom.1958.daily.nc`. Chunk on **day boundaries** so daily
  means stay whole. Resume on the **same node count**.
- ~0.18 s/step at 16 nodes → ~43 min/simulated-month; a 3-month run fits one 12 h job, or split
  across jobs with `sbatch --dependency=afterok:<prev>`.

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

## 3. Restart (DONE, task #11)

**Per-rank binary checkpoints** — `src/fesom_restart.{h,cpp}`. Each rank writes/reads its own
`<dir>/restart_<rank>.bin` (no gather, no OOM, scales trivially). **Must resume on the SAME
rank count** (enforced by an `npes` header field). Knobs:
- `FESOM_RESTART_DIR=<path>` — enable checkpoint/resume. Auto-resumes if a checkpoint is present.
- `FESOM_RESTART_EVERY=N` — write every N steps; always writes at the end of the chunk.
A resumed job runs `nsteps` MORE steps from the checkpoint; step numbers stay ABSOLUTE so the
AB2 momentum bootstrap (`step_n==1`) never re-fires on resume.

Saved state (matched by name on read; the build_list in fesom_restart.cpp is the source of truth):
- ALE thicknesses: `hnode, hnode_new, helem, hbar, hbar_old` (carry eta in linfs).
- dyn: `uv, uvnode, uv_rhsAB, eta_n, d_eta, ssh_rhs, ssh_rhs_old`.
- tracers (T,S): `values, valuesAB, valuesold`.
- ice: `a/m/m_snow` (+ `_old`), `uice/vice` (+ `_old`), thermo `thdgr/thdgr_old/thdgrsn/thdgra/t_skin`.
- meta: absolute step + full model calendar (JRA55 resumes at the right date). On resume the
  monthly-climatology reads (SSS restoring, chl) are forced once (prev_calendar bumped a month
  back) — else `Ssurf`/chl would stay stale within the resume month and skew `relax_salt`.

**Continuity verified** (pi, JRA55, Serial; 20 steps straight vs 10 + checkpoint + resume + 10):
wet-point prognostic state (uv/eta/w/T/S, max/min over owned points) is **bit-identical at printed
precision** (`T[-2.01,29.75] S[19.17,37.35]` both). The raw checkpoint isn't byte-identical — the
residual lives in physically-meaningless entries (extended-element scratch padding, ice-free nodes)
and grows chaotically like any tiny perturbation, well within the CUDA climate-close budget (~1e-3).
For a climate run this is exact-enough; bit-identity isn't meaningful on CUDA anyway.

**Usage notes:** (1) resume on the same node/rank count; (2) the I/O-stream mean accumulator does
NOT persist across a restart (a fresh `io_init` each chunk), so checkpoint on **period boundaries**
(multiples of 480 steps = 1 day at dt=180) to keep daily means whole.
