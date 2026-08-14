# M13 — the per-partition blow-ups: root cause and fix (2026-08-14)

**Verdict: there is no solver bug. The "stock CG blows up on specific partitions" phenomenon is
partition-dependent INITIAL CONDITIONS produced by the climatology hole-filling
(`extrap_nod3D`, a literal port of upstream `gen_support.F90:400-507`). On unlucky partitions
the fill lays a large spurious density front across single elements in marginal straits; the
resulting constant pressure-gradient force drives a linear velocity ramp from step 1 into CFL
blow-up. Every scheme dies on such a partition — the stock CG is merely the only one that dies
*loudly*.** Fix: deterministic partition-independent fill behind `FESOM_IC_EXTRAP=det`.

## 1. How the handoff's premise fell

The 2026-08-14 handoff (docs/plans/20260814-cg-blowups-HANDOFF.md) framed the problem as
CG-specific because "SE and oati run clean" on the failing partitions. That was an exit-code
illusion:

- All six failing stock-CG runs share one signature: normal iteration counts, then 2–3 steps of
  collapse (74→57→22 class), then residual 1e+52…1e+69 or `pp·App = nan` at the next step's
  iteration 1. The dist_32768 "NaN@iter1" was step 15, not step 1.
- The **oati** legs on the same partitions hit the same event at the same step and then report
  `0 CG iters` for every remaining step — the stall/convergence logic is NaN-blind (NaN
  comparisons are false), so the run "completes" as a zombie. Their timings are meaningless.
- The **SE** legs have no guard at all; their step-300 diag rows show `uv=0 eta=0` with
  inverted ±1e30 T/S sentinel ranges = all-NaN state.
- The `BLOWUP` guard (|uv|>5) only executes at print cadence; at `print_every=1000` a 300-step
  run never checks it. With `FESOM_PRINT_EVERY=1` every leg on a failing partition aborts at
  the step |uv| crosses 5.

Truth table at 300 steps (G4 jobs 26944983–26952348 + diag re-runs 26959687–26959699):

| partition | stock CG | oati | split-explicit |
|---|---|---|---|
| ng5 /pool dist_4096   | dies s200 | zombie ~s205 | NaN zombie |
| ng5 /pool dist_8192   | dies s24  | zombie s25   | NaN zombie (guard: s10) |
| bigpart dist_16384    | clean     | clean        | clean |
| bigpart dist_20480    | dies s48  | zombie s54   | **clean & healthy** |
| bigpart dist_24576    | clean     | clean        | clean |
| bigpart dist_32768    | dies s15  | zombie s15   | NaN zombie |

Only dist_20480 shows a genuine scheme split (its event is the weakest — |uv| crosses 5 at
step 38; SE rides it out, SI does not).

## 2. Localisation

`FESOM_PRINT_EVERY=1` + `FESOM_DIAG_UVMAX=1` (both already in the frozen se0 bin) on the
failing partitions shows, in every case, a **linear |uv| ramp from step 1** — a constant force,
not an instability — at one mid-depth element in a marginal strait:

- dist_8192: rank 5892, elem gid 8241930, nz=36, 27.49E 40.82N — **Sea of Marmara**;
  ramp 0.5 m/s per step, identical in SI (zstar), SI (linfs) and SE to ~4 digits.
- dist_20480: 27.38E 40.68N (2-rank element) — Marmara; ~0.3 m/s per step decaying.
- dist_4096: −4.7E 36.2N — **Gibraltar**; slower ramp, |uv| crosses 5 at step 142.

The scheme-identity of the ramp kills every solver/dynamics hypothesis: the force is static
and pre-barotropic.

## 3. The partitions are structurally innocent

`scripts/m13_partition_census.py` (V1: edge endpoints in node set; V2: elem vertices in node
set; V3: owned edges' adjacent elems present locally owned; V4: every globally-incident edge of
every owned node present in the owned edge list; V5: ranges/dups) and
`scripts/m13_cominfo_census.py` (C1: rlist targets the halo range — an owned-range entry would
overwrite owned data; C2: slist inside owned; C3: ptr sanity; C5: rlist covers each halo slot
exactly once; C4: cross-rank gid reciprocity of send/recv sequences, order-sensitive, nod2D and
elem2D): **all pass with zero violations on failing and working partitions alike** (V2 hits are
the documented outer-eXDim ring class, present everywhere, tolerated by the reader). The
my_list/com_info files are valid; the halo routing is exact.

## 4. The smoking gun

New probe `FESOM_DIAG_ELEM=<gid>` / `FESOM_DIAG_ELEM_NZ=<nz>` (fesom_main.cpp print block,
env-gated, byte-inert) at elem 8241930, nz=36 (≈300 m), step 1, jobs 26959921/26959922:

| | dist_4096 (quiet) | dist_8192 (runaway) |
|---|---|---|
| S at nodes 4135022/24/23 | 18.611 / 18.611 / 18.611 | **21.90 / 25.52 / 27.94** |
| T | 9.182 / 9.182 / 9.182 | 10.21 / 11.09 / 11.54 |
| rho−rho0 | −14.0376 (all three) | −12.24 / −10.27 / −4.58 |
| pgf_x | 6.6e−19 | **−5.5e−3 m/s²** |
| uv after step 1 | 1.5e−6 | 0.678 m/s |

A ~10 kg/m³ density jump across one ~2-km element. 0.0055 m/s² × 180 s ≈ 1 m/s per two steps —
the observed ramp. (Neither fill is oceanographically right — real Marmara deep water is
Mediterranean, S≈38.5 — but only the internally *inconsistent* fill is dynamically fatal.)

## 5. Mechanism

`extrap_nod3D` fills every PHC-dummy node (any land corner among the four 1°-grid neighbours
marks a node dummy — the entire Marmara, and most straits/marginal seas, are dummy) by
averaging already-valid neighbours: in-place Gauss–Seidel in local-numbering order, each rank
run to local exhaustion (`while (success)`) before the next halo exchange, filled nodes never
revisited (first-fill-wins), outer continuation testing layer 0 only. The result depends on the
local numbering and on where rank boundaries fall. Water masses on opposite sides of a strait
differ by ~20 PSU (Black Sea ≈18 vs Aegean ≈38.5), so an unlucky decomposition writes a violent
front inside the basin. Deterministic per partition, 2/2 reproducible, scheme-independent —
exactly the observed phenomenology, including M11's F34/F45 "fragile partitions, per-partition
in both directions, nothing offline predicts which" and NG5 being the most fragile mesh (5 km
resolves many marginal seas).

The same algorithm runs upstream in Fortran (`gen_support.F90:400-507`): cold-started
production runs on unlucky partitions inherit the same fragility (restarts are immune, which is
why it is rarely seen). → Sergey packet material.

## 5b. No NG5 partition was ever clean under legacy fill

dist_16384 — a "working" partition — re-run with per-step diag and the guard raised
(job 26959926): its global |uv| max is ALSO a Marmara ramp from step 1, cresting at
**4.86 m/s at step 95** (just under the fatal range) before recovering. The working/failing
split is only the accidental amplitude of the same IC artifact. Corollary: cross-partition
comparisons of NG5 cold starts under legacy fill compare *different physical transients*.

## 6. The fix — `FESOM_IC_EXTRAP=det` (fesom_phc.cpp)

Two-phase, per layer, all constructed to be decomposition-independent (Jacobi with validity
and values frozen per sweep; neighbour (gid,value) pairs sorted by ascending global id before
summing so the FP sum is partition-independent; halo exchange every sweep; global Allreduce
termination):

1. **Ring fill** — every still-dummy node with ≥1 valid neighbour gets the
   multiplicity-weighted mean of those neighbours (legacy stencil and eligibility tests).
2. **Relaxation to quasi-harmonic** — the filled nodes (original valid data held fixed as
   boundary values) are Jacobi-relaxed until the global max per-sweep change <
   `FESOM_IC_EXTRAP_TOL` (default 1e-3, 20k sweeps/layer cap). ⚠️ Phase 1 alone is NOT enough
   — measured (det0, jobs 26960049/51): ring filling from two water masses meets in an
   equidistant front of maximal contrast across ~one ring; on NG5 that gave a
   partition-INDEPENDENT step-1 |uv| of 3.06 m/s and death at step 9 on every partition.
   The relaxation replaces meeting fronts with the smooth interior interpolation of the valid
   boundary data (no internal extrema).

Legacy vertical fill and cleanup unchanged. Default OFF (`legacy`); unknown values refuse
loudly. On np=1 det ≠ legacy (Jacobi ≠ Gauss–Seidel), so certified byte baselines are
untouched with the knob off.

**pi-mesh proof** (login node, np ∈ {1,2,8}, PHC IC, surface T/S dumps via
`FESOM_EVP_DUMP_DIR`): legacy differs across np at up to **1.33 °C/PSU** (147 of 3140 surface
nodes np1-vs-np8); det is **bitwise identical across np** (0 differing nodes, all pairs; same
169 fill + 476 relax sweeps at every np). NG5 fill: 260 ring sweeps (phase 1).

## 7. Acceptance ladder (in flight / to complete)

1. Knob-off byte gate vs certified baseline (`jobs/job_m7_gate_serial`, job 26960066).
2. Failing-partition flips under det: dist_8192 (26960049), dist_4096 (26960051),
   dist_20480 (26960052), SI 300 steps; SE arm on dist_8192 (26960059).
   Pre-registered: probe T/S at elem 8241930 bitwise-identical between dist_4096 and
   dist_8192 under det; pgf ~1e−18; all runs 300 steps clean.
3. Working-partition control under det: dist_16384 (26960058) stays clean.
4. dist_32768 (256N) re-run under det — pending the above (16-node-cap-class courtesy:
   big allocation, run once the small ones are green).
5. CUDA fidelity gate before any default-on adoption (host-side init code, but the ladder is
   the ladder).

## 8. Consequences for the M12 board (flag to the M12 session)

- NG5 dist_4096/8192/32768 rows: SE and oati "successes" are NaN-zombie runs; their timings
  (incl. oati's 0-iteration steps) are not comparable. "CG diverges where SE survives" is only
  true on dist_20480.
- NG5 128/160/192N rows (dist_16384/20480/24576): SE numbers stand (healthy states verified at
  step 300); CG numbers stand where CG survived.
- Any cross-partition comparison on NG5 compares runs with *different initial conditions* in
  extrapolated regions (legacy fill) — a caveat for scaling tables until det is adopted.

## 9. Instruments added

- `FESOM_DIAG_ELEM=<gid>` / `FESOM_DIAG_ELEM_NZ=<nz>` element probe (print block; also dumps
  one-time IC T/S columns of the element's nodes at step 1).
- `scripts/m13_partition_census.py`, `scripts/m13_cominfo_census.py` — offline dist_N
  validators (V1–V5, C1–C5 + reciprocity).
- Existing but newly load-bearing: `FESOM_PRINT_EVERY`, `FESOM_DIAG_UVMAX`, `FESOM_UV_GUARD`,
  `FESOM_EVP_DUMP_DIR` (per-rank per-gid pre-extrap and post-load surface T/S).

Bins: `/work/ab0995/a270088/port2/m13/bin/fesom_port_serial_probe0` (probe only, sha
68775f20…), `fesom_port_serial_det0` (probe + det fill, sha 8aa313a3…).
