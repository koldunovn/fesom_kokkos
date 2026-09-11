# M16 — mixed precision (FESOM #940 exact port) on m14-integrate: the measured campaign

Companion to `docs/plans/20260902-m16-mixed-precision.md` (plan + ticks) and `docs/PRECISION_ISLANDS.md`
(the conformance registry + promotion log). **Every number here carries a job id.** Binaries are the
sha-named pairs from `jobs/m16_bins.sh` (`/work/ab0995/a270088/port2/m16/bin/<tag>/{dp,sp}/`).

## 0. Gates 0–1 (closed 2026-09-07)
| gate | evidence |
|---|---|
| G0 FP64 byte identity (knobs unset + `FESOM_SSH_PRECOND=0` vs the unmodified `d4a9fe0` oracle) | pi np1+np2 14 configs bitwise after every slice; **CORE2 np8 SLURM 27286999 / 27287001 / 27287827 / 27287828 all BYTE-IDENTICAL**; re-earned on the final Serial binary (`d1` f2eb28b1) on pi |
| G1 SP runs past the forcing boundaries (CORE2 np8, JRA 1958, 60 steps = 30 h) | **job 27288954** (`d1sp` 722e3002): rc 0, no non-finite, CG \|Δit\| vs DP mean 0.43 max 1; SP-vs-DP S relL2 1.15e-5, T 1.4e-4, u/v 8e-3. First attempt (job 27288894) died at step 5: the JRA point-slope macro used above its definition — registry log |
| Salt anomaly (#986) CORE2 30 h | DP on-vs-off S mean −1.9e-4 psu (upstream residual class); **SP mean salt error vs DP 3.6e-6 → 0.92e-6 psu (−74 %)** — jobs 27288895 + 27288954, `core2_all/REPORT.txt` |
| CUDA smoke (pi, DP+SP, NANSCAN) | jobs 27288722 (`cuda0/cudasp0`) and **27289067 (`e0`)** PASS |

## 1. Gate 2 board — prize sizing (`ARMS="dp sp"`, identical knobs, ABBA, warm-up discarded, min over 2+2 legs)
Protocol: `jobs/job_m14_ladder_{cpu,gpu}` with `M16_BINS=…/e0` (commit `f95eaef`), 300 steps at the
protocol dt, `FESOM_SPEED=1 FESOM_IC_EXTRAP=det` on both arms (the M14 "base" configuration), `WSPLIT`
per the mesh rule. `cfg=` stamps `prec=`. CG iterations = the step-300 `it=` of each leg.

| mesh | backend | nodes × ranks | knobs | DP s/step | SP s/step | SP/DP | CG it DP / SP | legs spread | job |
|---|---|---|---|---|---|---|---|---|---|
| CORE2 | GPU A100-80 | 1 × 4 | knobs-off (`FESOM_SPEED=1`, det) | 0.0618 | 0.0531 | **0.859 (−14.1 %)** | 60 / 62 | 0.16 % / 0.38 % | 27289143 (node l50003) |
| CORE2 | GPU A100-80 | 16 × 64 | knobs-off | 0.0794 | 0.0743 | **0.936 (−6.4 %)** | 60 / 62 | 0.76 % / 0.27 % | 27289163 |
| NG5 | GPU A100-80 | 16 × 64 | knobs-off, `WSPLIT=1` | **FP64 legs DIED after step 1 (rc 1, both legs)** | 0.1904 | — | (65 @1) / 49 | — / 0.16 % | 27289174 — FP64 legs: `[fesom_port FATAL] CG_kk: pp·App is -nan` at step 2 on all 64 ranks = the **M14-documented NG5 A100 FP64 failure class** (M14 handoff: base legs fail at 32/64 GPUs, promotion screens fail with the same NaN, stock partitions too); SP survives it (roundoff-seeded onset, rule 0.41). The FP64 NG5 16N point is UNAVAILABLE on this hardware this session; the SP time stands alone (M14's FP64 i1 at the same point measured 0.381 s/step on another day — not a pair, not a ratio). |
| CORE2 | GPU A100-80 | 16 × 64 | **recipe**: `FESOM_WHICH_EVP=1 FESOM_SPEED_EVPWIDE=8 FESOM_SPEED_EVPWIDE_LEAN=1 FESOM_SALT_ANOMALY=1` | 0.0607 | 0.0561 | **0.924 (−7.6 %)** | 60 / 62 | 1.32 % / 0.18 % | 27294187 (`e2`); the recipe alone: FP64 0.0794 → 0.0607 (−23.6 %), SP 0.0743 → 0.0561 (−24.5 %); EVPWIDE lean announced live in both arms (`knobs.txt`) |
| dars | CPU (Serial) | 64 × 8192 | **recipe**: `FESOM_ALE=zstar FESOM_SSH_MODE=se FESOM_SE_M=20 FESOM_SALT_ANOMALY=1`, `WSPLIT=1` | 0.0971 | 0.0710 | **0.731 (−26.9 %)** | — (SE) | 0.10 % / 1.41 % | 27293954 (`e2`); the SE recipe alone is a wash on dars 8192 (FP64 0.0979 → 0.0971, −0.8 %) — the M14 dars lever was `oati`, unusable at SP; SP is the dars lever |
| fArc | CPU (Serial) | 32 × 4096 | **recipe**: `FESOM_ALE=zstar FESOM_SSH_MODE=se FESOM_SE_M=90 FESOM_SALT_ANOMALY=1`, `WSPLIT=1` | 0.0453 | 0.0376 | **0.830 (−17.0 %)** | — (SE, no CG) | 0.00 % / 0.53 % | 27293953 (`e2`); the recipe alone: FP64 0.0578 → 0.0453 (−21.6 %), SP 0.0494 → 0.0376 (−23.9 %) |
| fArc | CPU (Serial, 128/node) | 32 × 4096 | knobs-off (`FESOM_SPEED=1` inert on CPU, det), `WSPLIT=1` | 0.0578 | 0.0494 | **0.855 (−14.5 %)** | 149 / 149 | 1.38 % / 1.01 % | 27289249 |
| dars | CPU (Serial, 128/node) | 64 × 8192 | knobs-off, `WSPLIT=1` | 0.0979 | 0.0717 | **0.732 (−26.8 %)** | 22 / 22 | 0.10 % / 0.28 % | 27289250 |

Recipe rows (BASE_KNOBS = the per-point M14 recipe) follow once the knobs-off rows are in. ⚠️ The M14 CPU recipe lever `FESOM_SSH_SOLVER=oati` is unusable at SP as built (§2) — the SP recipe row on fArc/dars uses `pcsi` or plain `cg`.

Reading so far: SP and the recipe levers are close to multiplicative (CORE2 16N: recipe −23.6 %, SP −6.4 % knobs-off / −7.6 % under the recipe; fArc 4096: recipe −21.6 %, SP −14.5 % / −17.0 %) — July's "they overlap in the communication bytes" is at most a few percent here. The SP gain grows with the BYTE share of the step and shrinks where latency rules (CORE2 1N GPU −14 %, CORE2 16N GPU −6.4 % at 0.08 s/step, fArc 4096 CPU −14.5 %, dars 8192 CPU −27 %) — July's headline ("SP and the speed stack overlap in the communication bytes") reproduced on the m14 tree with knobs off.

### 1b. Re-measure on the fixed allocator (`e3` = `Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF`, 2026-09-08)
Every row above was measured on Kokkos' cudaMallocAsync pool, which corrupts CUDA-aware-MPI halos (registry 2026-09-08) and
costs ~17 % on the device path. Re-measured rows:

| mesh | backend | nodes × ranks | knobs | DP s/step | SP s/step | SP/DP | job |
|---|---|---|---|---|---|---|---|
| CORE2 | GPU | 1 × 4 | knobs-off | 0.0601 (pool 0.0618) | 0.0523 (pool 0.0531) | **0.870** | 27313842 |
| CORE2 | GPU | **2 × 8** | knobs-off | 0.0488 | 0.0436 | **0.893** | 27339007 |
| CORE2 | GPU | **2 × 8** | recipe (EVPWIDE lean + anomaly) | 0.0415 | 0.0369 | **0.889** | 27339008 |
| NG5 | GPU | **4 × 16** | knobs-off, `WSPLIT=1` | 0.6111 | 0.5276 | **0.863** | 27339009 |
| CORE2 | GPU | 16 × 64 | knobs-off | 0.0627 | 0.0590 | **0.941** | 27313843 |
| CORE2 | GPU | 16 × 64 | recipe | 0.0486 | 0.0474 | **0.975 — inside leg noise, see below** | 27313844 |
| NG5 | GPU | **16 × 64** | knobs-off, `WSPLIT=1` | 0.1924 | 0.1661 | **0.863** | 27313845 |

**Device memory** (new: the ladder now polls `nvidia-smi` around every leg and prints `gpumem_max=`;
until 2026-09-09 the "0.51× memory" claim rested on one July hand-sample on dars). CORE2 8 ranks on
2 nodes, high-water per GPU including the CUDA context: knobs-off **2635 → 1537 MiB = 0.583×**
(27339007); recipe **2639 → 1539 MiB = 0.583×** (27339008). Above the July 0.51× because the double
islands and the fixed context do not halve.

Reading at the production posture: SP/DP is **0.870 at 1 node, 0.893 at 2 nodes** (and 0.936 at 16 on
the old allocator) — the prize shrinks monotonically as the per-GPU work falls and latency takes over,
so the number quoted for the hindcast must be the 1–2 node one. Leg spreads 0.00–0.46 %. Both pairs
cost 3 minutes on 2 nodes; the 16-node versions they replaced had queued 20 h (plan D13).

**Superseded plan (2026-09-09):** the 16 × 64 rows were dropped from the critical path (D13). CORE2
does not scale past ~2 GPU nodes on Levante (board §1: 0.0618 s/step at 1 N vs 0.0794 at 16 N), so a
16-node CORE2 row sizes the prize in a configuration nobody runs. **They were dropped, not cancelled:
27313843/44/45 reached the front of the queue and ran on 2026-09-09 before the cancellation, so their
rows above are free data on the fixed allocator.** Only 27313846 (NG5 16 N recipe) is still PENDING
(start 2026-09-10T01:40) and should be cancelled — the recipe *is* the three M14 levers, which D13
puts outside the SP paper.

**🔴 The two mesh families say opposite things, and that is the result.** On CORE2 the SP prize decays
monotonically with node count — **0.870 (1 N) → 0.893 (2 N) → 0.941 (16 N)** — because CORE2 is past its
knee and the step is latency-bound, where halving the payload buys nothing. On **NG5 the ratio does not
move: 0.863 at 4 nodes and 0.863 at 16 nodes** (0.6111 → 0.1924 s/step is 3.18× over 4× the nodes, so
NG5 at 16 N is still on the scaling curve). The SP gain therefore tracks the **byte share of the step**,
not the node count: it survives wherever there is still work per GPU. This is the honest way to quote
the prize — a single headline number is a statement about the mesh and the node count, not about SP.

⚠️ **The CORE2 16 N recipe row (27313844) is not a measurement of anything.** Its SP legs are
[0.0474, 0.0488] = **2.95 % spread**, wider than the 1.2 % SP−DP gap it reports; taking the other SP leg
flips the ratio to 1.004. Read it as "at 16 nodes under the recipe, SP's advantage on CORE2 has fallen
into the leg noise", which is what the 0.870 → 0.941 trend predicts. Not a row to quote.

**Device memory, first large-mesh number** (NG5 4 N, 27339009): **36765 → 20129 MiB = 0.547×** per GPU,
below CORE2's 0.583× (27339007) exactly as expected — the FP64 islands and the fixed CUDA context are
a constant, so their share shrinks as the mesh grows. Host high-water falls with it (sacct MaxRSS
50.5 → 35.2 GB = 0.70×). **0.547× is the number to quote for a big mesh; 0.583× for CORE2.** Neither
is the naive 0.5×, and July's hand-sampled 0.51× was optimistic.

Device-pointer halo path vs host-staged (`FESOM_HALO_STAGE=1`), same allocation, CORE2 4 nodes, 300 steps, FP64
(leg 1 of the device arm only — the ladder's env reset did not clear `FESOM_HALO_STAGE` between arms until 2026-09-08,
so the 4th leg ran staged; fixed in both ladder jobs): pool build device 0.0629 → staged 0.0414 (−34 %, job 27310269);
fixed build device 0.0534 → staged 0.0408 (−24 %, job 27313649). **The staged path is the faster halo on Levante A100
by a wide margin** (dolpung already runs it).

**The 16 N pairs ran (27310270 pool / 27313650 fixed) and the staged path holds at scale — but their
printed `GAIN` lines are wrong and must not be quoted.** Both were submitted (2026-09-08 14:04 and
15:41) *before* the env-reset fix was committed (`f91aa86`, 17:17 the same day), and SLURM snapshots the
batch script at submit time — so both ran the leaking version, leg 4 ("base") inherited
`FESOM_HALO_STAGE=1` from the best arm, and min-over-legs then selected that leaked leg as the baseline.
The printed −0.24 % / −0.48 % are therefore staged-vs-staged. Discarding the leaked legs and reading the
clean device-path legs only: **pool build 0.0798 → 0.0420 (−47 %)**, **fixed build 0.0634 (leg 1),
0.0663 (warm-up) → 0.0412 (−35 %)**. The fixed-build number rests on two independent device-path
observations; the pool number on one (its warm-up was rejected, rc 1). So the staged path's advantage
*grows* from −24 % at 4 nodes to ~−35 % at 16 — consistent with it being a latency effect. ⚠️ This is
the third time the min-over-legs rule has quietly rewarded a contaminated leg: **a leg that inherits the
other arm's knob is always the fast one, so it always wins the min.** The zombie check rejects dead legs;
it does not reject *mislabelled* ones.

Incidents: job 27289077 (same pair, node **l50154**) hung after the speed-knob lines; the M14 `i1`
warm-up segfaulted there with UCX `VM_UNMAP` warnings. Excluding the node fixed it — the gpu partition
is heterogeneous (memory rule); submit GPU ladders with `--exclude=l50154` until DKRZ confirms the node.

## 2. Gate 3 — knob liveness at SP
Driver: `jobs/job_m16_gate_{serial,cuda}` → `scripts/m16_gate0.sh` `M16_MODE=live` → `scripts/m16_knob_signals.sh`
per config (CORE2 np8, 20 steps, `FESOM_SSH_PRECOND=0` + the config's knobs, the 15 gate-0 configs).

| backend | job | verdict | detail |
|---|---|---|---|
| CUDA SP, 2 nodes (`e0/sp/fesom_port_cuda`) | 27289199 | 12 / 15 live; **pipecg, oati, pcsi DEAD** | every config rc 0 and finite; CG iterations at step 20 within 2 of the FP64 oracle for cg/cg2/cgpipe/cgpoly/se; **`[ssh-solver] !! FALLBACK … residual stalled or grew` on 20/20 solves (pipecg), 20/20 (oati), 19/20 (pcsi)** — the FP64 Serial oracle has 0 fallbacks in all three. The first liveness pass called them LIVE because the announce line prints before the fallback; `m16_knob_signals.sh` now has a `solver-fallback` row and `m14_zombie_check.sh` rejects a leg with a fallback. |
| Serial SP, np8 (`e0/sp/fesom_port_serial`) | 27289198 | same 12 / 15; **pipecg 20/20, oati 20/20, pcsi 19/20 fallbacks** | SP-generic, not CUDA: the true-residual floor is the same on both backends (verify solve 1 true 4.823 Serial / 4.830 CUDA vs rtol 4.339; FP64 true = rec = 4.002). Mechanism + response in the registry log (2026-09-07 G3 entry): `soltol=1e-5` is below float `eta` resolution (upstream #940 says so and ships it); port: CA-solver scalar chains → `dbl_t` + `FESOM_SSH_FLOOR` acceptance (announced, counted). **Re-test on `e1` (= `084973c`, scalar chains `dbl_t` + `FESOM_SSH_FLOOR`): Serial job 27289394 / CUDA job 27289407 agree — `pcsi` LIVE, 0 fallbacks (iterations 155/240/290 at steps 1/10/20 vs FP64 155/145/140: it converges through the floor rule at ~2× the FP64 count); `pipecg`/`oati` still fall back on 13/20 solves, now by the DIVERGENCE branch (recurred residual grows to 1e1–1e4 at ~130 iters: the pipelined recurrences lose the residual in float — the textbook pipelined-CG instability that needs residual replacement, not a wider scalar). Verdict: `pipecg`/`oati` are NOT usable at SP as built (E5 item: residual replacement every k iterations); the SP recipe rows use `cg`/`cgpipe`/`cgpoly` (GPU) and `pcsi`/`cg` (CPU).** |

### 🔴 2c. WHICH half of the e1 fix was load-bearing (2026-09-09) — the board was ambiguous, now it isn't
`e1` changed two things at once (`dbl_t` scalar chains **and** `FESOM_SSH_FLOOR`) and the row above
credits the pair. **Job 27359018 separates them on the real matrix**, same SP binary
(`f1/sp`, `3459dd3a`), CORE2 np8, 20 steps, the only difference being the new `pcsi_nofloor` config:

| config | verdict |
|---|---|
| `pcsi` (floor on, default factor 8) | **LIVE**, 7 signals, `it=230` at step 20 |
| `pcsi_nofloor` (`FESOM_SSH_FLOOR=0`) | **NOT LIVE — `solver-fallback`**, i.e. exactly the pre-`e1` failure |

**The float-floor acceptance rule is what made `pcsi` usable at SP; the `dbl_t` scalar chain is not.**
This was predicted from first principles by the new unit test (`tests/test_ssh_solvers.cpp`, §M16):
the true-residual floor is set by float **vector storage** and is essentially unmoved by the scalar
accumulator width. The `dbl_t` chains remain the correct class-4 promotion and cheap insurance — and
they may still matter for `pipecg`/`oati`, which fail by a different mechanism (divergence, not a
floor) — but they should not be credited with this fix.

**Reading:** the port-only communication-avoiding solvers carry their recurrence scalars in `real_t` (class 4,
"real_t except global integrals") — the pipelined/Chebyshev recurrences lose the residual in float. Upstream
has none of these solvers, so this is the class-4 promotion the plan reserved for E5: the scalar chains of
`pipecg`/`oati`/`pcsi` (dots, α/β/γ recurrences, Lanczos estimates) → `dbl_t`, measured give-back per solver.
Plain `cg`, `cg2`, `cgpipe`, `cgpoly` are live at SP as built.
Floor announcement verified on CORE2 (job 27289526, `d3sp`): `[ssh-solver] SP float-floor acceptances: floor-hits=19 of
20 solves (stall with resid < 8 x rtol; FESOM_SSH_FLOOR)`; pcsi iterations 155/280/230 at steps 1/10/20 (FP64 155/145/140).

### 30-day conservation twin (`jobs/job_m16_conserv`, CORE2 np8, 1440 steps dt 1800, `FESOM_MP_CONSERV=10`, `d3`/`d3sp`)
Jobs 27289583 (FP64) / 27289584 (SP), both rc 0, no non-finite; `scripts/mp_conserv_drift.py` → `m16/conserv_drift.csv`.

| quantity | FP64 drift @1440 | SP drift @1440 | gap | \|gap\|/\|FP64 drift\| | reading |
|---|---|---|---|---|---|
| heat | −4.358e-4 | −4.367e-4 | −9.4e-7 | **0.002** | July's 0.2 % reproduced; the physical 30-d signal dominates both |
| salt | −1.748e-6 | −2.983e-6 | −1.24e-6 | **0.71** | SP salt drift 1.7× FP64, and it GROWS LINEARLY (−1e-8 @10, −5.9e-7 @720, −1.2e-6 @1440) — a drift, not a random walk; the salt-anomaly twin (jobs 27290582 FP64-on / 27290583 SP-on) is the direct test |
| volume | 0 | 0 | 0 | 0 | exact in both (linfs) |

**Salt-anomaly conservation twin** (same protocol, `FESOM_SALT_ANOMALY=1` in both precisions; jobs 27290582 FP64-on / 27290583 SP-on, `m16/conserv_drift_salt.csv`):

| pair | heat gap / FP64 drift | salt @1440 (FP64 → SP) | salt gap / FP64 drift | reading |
|---|---|---|---|---|
| FP64-on vs SP-on | 0.002 | −1.746e-6 → −1.739e-6 | **0.004** | the SP salt drift is gone (was 0.71 without the anomaly) |
| FP64-off vs FP64-on | 0.000 | −1.748e-6 → −1.746e-6 | 0.001 | the anomaly is FP64-neutral at the conservation level (the ~3e-6 psu/step surface residual is below this diagnostic) |
| FP64-off vs SP-on | 0.002 | −1.748e-6 → −1.739e-6 | **0.005** | SP with the anomaly sits on the FP64-without line |

**Verdict for the campaign: `FESOM_SALT_ANOMALY=1` is part of the SP recipe** — it turns the one SP conservation defect found in 30 days (salt) into rounding, at zero cost (the same knob is a no-op in FP64 by construction, gate 0).


## 3. Gate 4 — screens and the 1-year twin
3000-step SP screens at the recipe points (`e2`, `ARMS=sp NSTEPS=3000`, warm-up + 2 legs, `WSPLIT` per mesh rule):

| cell | recipe | legs | job | verdict |
|---|---|---|---|---|
| fArc CPU 4096 | zstar + SE M=90 + salt anomaly | 2/2 × 3000 steps finite, 0.0387 s/step, CG-free (SE) | 27294513 | **PASS** |
| dars CPU 8192 | zstar + SE M=20 + salt anomaly | 2/2 × 3000, 0.0726 s/step | 27294514 | **PASS** |
| CORE2 GPU 16N | EVPWIDE lean + salt anomaly | leg 1 3000 finite (0.0549 s/step, it 62); leg 2 died at step 2000, warm-up died | 27294510 | partial — see the CUDA flake entry (registry 2026-09-08); the leg that ran is a pass, the one that died is the infrastructure failure |
| NG5 GPU 16N | EVPWIDE lean + salt anomaly | both SP legs died at step 1–2 (knobs-off SP ran 300 steps at 0.1904 on 27289174) | 27294512 | FAIL — under investigation (recipe vs flake) |

### 3b. The faithfulness matrix — G4's actual bar, first numbers (2026-09-09)

Plan **D14**. Until today G4's bar ("SP-vs-Fortran ≡ DP-vs-Fortran") had nothing on the Fortran side
to stand on. It now runs as **four arms on the one setup both codes can share** — upstream's
`setups/test_core2` (CORE2 mesh + JRA55 + PHC) — plus **FP64 noise twins** from upstream's own
`&oce_perturb`. Driver `scripts/m16_faith_setup.sh` (which argues the four deliberate deviations from
upstream's `setup.yml`), `jobs/job_m16_faith_{fortran,port}`, analysis `scripts/m16_faith_compare.py`.

**Pilot: January 1958, 64 ranks / 1 node, dt 2700 (upstream's CI value), Serial port vs Intel oracle
`a62f180`.** All arms rc 0, no non-finite. Both codes wrote the *same window* — one record, shape
(1, 47, 126858), time stamp 1339200 s — so the comparison is like-for-like. The Fortran SP arm
printed `SINGLE PRECISION MODE`, which is the one hard precision fact the harness gives: **219 s vs
DP's 389 s = 1.78×**, against the 1.69× upstream reports for this posture (jobs 27355327, 27355390,
27355632; det pair 27355922/27355923).

| quantity (relL2, monthly mean, valid points) | sst | a_ice | temp | salt |
|---|---|---|---|---|
| **Fortran SP vs its own DP** | 9.09e-05 | 6.51e-04 | 1.20e-04 | 9.24e-06 |
| **port SP vs its own DP** | 1.25e-04 | 6.12e-04 | 1.55e-04 | 1.42e-05 |
| **RATIO port / Fortran** | **1.38** | **0.94** | **1.29** | **1.54** |
| FP64 envelope, σ = 1e-6 K | 3.23e-05 | 4.16e-04 | 4.79e-05 | 3.24e-06 |
| FP64 envelope, σ = 2e-4 K | 4.02e-05 | 5.82e-04 | 5.89e-05 | 4.75e-06 |
| **port vs Fortran at EQUAL precision (DP)** | 3.49e-03 | 3.18e-03 | 2.29e-03 | 1.96e-04 |

**Four readings, in order of how much they matter.**

**1. The G4 statement, first quantified: the ratio is 0.94–1.54.** The port loses to single precision
what upstream loses, to within a factor of 1.5 on the worst variable and 6 % on the best. This is the
number G4 asks for, and it does not depend on either departure being small.

**2. 🔴 The envelope has SATURATED by one month, and SP sits above it.** The two amplitudes differ by
**200×** and their envelopes differ by **under 20 %** — so by 31 days both perturbations have already
grown to the same level, and that level is the model's own noise floor rather than a property of the
nudge. That makes it a real bar, and every SP−DP departure is **1.1–2.8× above it**. So the
comfortable sentence "SP is buried in the noise" is **not true at one month** — it will only become
true at climate length, where the comparison has to be re-made. *(Predicted before the runs: the
opposite. The reason is that a perturbed IC is one kick that then grows at the flow's own rate, while
single precision injects rounding at every operation of every one of 992 steps; a continuous source
beats a single kick of 200× the amplitude.)*

**3. SP carries a small systematic component, not only decorrelation — and BOTH codes carry it.** The
mean shift of sst against each code's own DP arm is **−1.53e-05 (Fortran-SP)** and **−2.84e-05
(port-SP)**, while the FP64 noise twins shift only −3.3e-07 and −9.8e-07 — twenty times less. A pure
re-seeding of chaos would shift the mean like the noise twins do; this does not. The port reproduces
upstream's SP behaviour *including its bias*, which is a stronger faithfulness result than the ratio
alone.

**4. The two codes differ from EACH OTHER, at equal precision, 20–30× more than either differs from
its own DP arm.** That is the backdrop every ratio above rides on, and it needed an explanation.

- **It is not the ice edge.** For sst the gap is concentrated — **44.7 % of the sum of squares in the
  top 1 % of nodes** — in the energetic mid-latitude Southern Ocean (top-400 nodes: median latitude
  −36, only 5 % poleward of 60), and the ice edge is *under*-represented (0.04 of the top nodes
  against a 0.14 baseline). So this is **not** the F↔C ice-edge comparator class from M9.
- **It is not the partition-dependent IC hole fill**, which was the obvious suspect (M13). Repeating
  the 1-month pair with `ic_extrap_det` on in **both** codes (upstream `namelist.tra`, our own PR
  #979; the port's `FESOM_IC_EXTRAP=det`) moves the code-to-code gap by **0.4 %, 2.9 %, 0.8 %** for
  sst/a_ice/temp and 31 % for salt — i.e. essentially not at all. **The knob demonstrably fired:**
  det-vs-legacy *within* each code is 3.6e-03…3.6e-02, and the Fortran side prints no banner, so this
  empirical check is the only proof available (L80).
- **The scale anchor that makes it readable:** flipping `ic_extrap_det`, a supported upstream option,
  moves each code's own solution **more** (sst 4.86e-03 Fortran, 4.94e-03 port) than the two codes
  differ from each other (3.49e-03). The port-vs-Fortran gap is **smaller than the spread of
  legitimate configuration choices** — which is the honest way to report it.

**Consequence for the campaign:** run the matrix with det **off** (the shared default) — it changes
nothing and costs a knob. The 1-year matrix at the protocol dt 1800 is in flight: jobs 27355667–72
(Fortran fdp/fsp + four noise twins) and 27355673/27355675 (port pdp/psp), 8 × 1 node, ~2–2.5 h.

### 3b-year. The 1-year matrix (dt 1800, 64 ranks/1 node, 1958) — jobs 27355667–72, 27355673/75

Eight arms, all rc 0, 12 monthly records each; `fsp` printed `SINGLE PRECISION MODE` and all four
noise arms printed `PERTURBATION TRIGGER` (the automatic fired-check). Wall: Fortran SP/DP
3683/6560 s = **1.78×**, port SP/DP 5075/7704 s = **1.52×**.

relL2 against each code's **own** DP arm; envelope = max over the two σ=1e-6 K seeds.

| month | sst F-SP | sst P-SP | ratio | sst env | temp F-SP | temp P-SP | ratio | a_ice ratio | code-vs-code (sst) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 9.39e-05 | 9.45e-05 | **1.01** | 2.36e-05 | 1.14e-04 | 1.64e-04 | 1.43 | 0.98 | 3.51e-03 |
| 3 | 1.93e-04 | 4.11e-04 | 2.13 | 7.39e-05 | 2.15e-04 | 6.23e-04 | 2.90 | 1.09 | 4.60e-03 |
| 5 | 1.73e-04 | 7.68e-04 | **4.43** | 5.07e-05 | 3.57e-04 | 1.26e-03 | 3.54 | 2.72 | 9.68e-03 |
| 8 | 4.18e-04 | 1.26e-03 | 3.02 | 1.46e-04 | 1.13e-03 | 2.77e-03 | 2.46 | 2.46 | 1.45e-02 |
| 12 | 1.53e-03 | 2.57e-03 | **1.67** | 1.02e-03 | 2.74e-03 | 4.53e-03 | 1.65 | **2.61** | 9.78e-03 |

**Four readings.**

**1. The ratio is not stationary — it rises then falls, and a single number for it is wrong.** sst
goes 1.01 → 4.43 (May) → 1.67 (Dec); temp 1.43 → 3.73 (Apr) → 1.65; a_ice 0.98 → 2.95 (Nov) → 2.61.
The arc is what saturation looks like: the port's SP departure grows faster early, then both
approach the same chaotic ceiling. **Quote the ratio with its month, or quote the year-end value.**
The one-month pilot's 0.94–1.54 was not wrong, it was early.

**2. 🔴 The port loses MORE to single precision than upstream does, and by year end the gap is a
factor 1.65–1.7 on the ocean fields and 2.6 on sea ice.** That is a real result and should not be
softened: mid-year it reaches 4.4×. It is still the same *order*, which is the substance of G4, but
"the port loses what upstream loses" is only true to a factor of ~2 over a year, not to the printed
digit.

**3. 🔴 The envelope never overtakes SP−DP within the year — the answer to the headline question is
"not in 12 months".** Fortran-SP stays **1.2–4.3×** the σ=1e-6 K envelope and port-SP **1.6–19×**.
But the envelope is *closing*: it grows 43× over the year (sst 2.4e-05 → 1.0e-03) against Fortran-SP's
16×, so the ratio SP/env falls 3.98 → 1.50. They converge somewhere past one year. **This is a
multi-year question — which is exactly why Suvarchal ran 60.** The comfortable "SP is inside the
noise" claim cannot be made at one year and must be made, if at all, at climate length.

**4. The code-to-code gap grows to 1.0e-02 (sst) … 2.4e-02 (a_ice) and stays 4–6× the SP−DP
departures.** Precision remains the smaller effect all year.

### 🔴 Does the FORTRAN sit inside its own noise? No — it has the same problem
The obvious worry about §3b-year's third reading is that "SP is above the envelope" is a defect of
*our port*. It is not. Fortran-SP against its **own** DP arm and its **own** FP64 envelope, every
month of 1958 (envelope = max over the two σ=1e-6 K seeds):

| month | sst | temp | a_ice | salt | | port, for contrast (sst) |
|---|---|---|---|---|---|---|
| 1 | 3.98 | 4.01 | 1.66 | 4.00 | | 4.00 |
| 3 | 2.62 | 2.37 | 1.56 | 3.46 | | 5.57 |
| 6 | 4.29 | 3.48 | 2.92 | 4.56 | | 18.98 |
| 9 | 1.47 | 1.48 | 1.39 | 1.89 | | 3.74 |
| 12 | **1.50** | **1.31** | **2.05** | **1.36** | | 2.51 |

**Upstream's own single precision sits 1.3–4.6× above its own FP64 noise, in every variable, at every
month. It never goes inside.** The port is consistently further out (2.2–5.4× at year end) but the
two are on the **same side of the bar** — so "SP is not inside the noise at one year" is a property
of single-precision FESOM, not of this port, and the paper should state it that way.

The envelope is not the weak link in that statement: the two independent seeds agree to **1.6–3.5 %**
at month 12 (sst 1.022e-03 vs 9.870e-04; temp 2.062e-03 vs 2.096e-03), so a ratio of 1.3–1.5 is well
outside the seed spread. Both codes' ratios are falling toward 1 as the year runs (Fortran sst
3.98 → 1.50), consistent with a crossover past 12 months — which is the multi-year question.

### 🔴 3b-year-b. The port's OWN envelope (2026-09-10) — the hypothesis was WRONG
§3b-year proposed that part of the port's larger SP−DP ratio might be its DP trajectory sitting in a
more sensitive region rather than its SP arithmetic being worse. The port now has `do_perturb`
(§3e), so that was testable. **It is refuted.** Five arms (27369058–64), December 1958:

| | fortran SP−DP | port SP−DP | fortran own env | port own env |
|---|---|---|---|---|
| sst | 1.534e-03 | 2.569e-03 | 8.7e-04 … 1.4e-03 | **2.9e-04 … 5.6e-04** |
| temp | 2.738e-03 | 4.527e-03 | 1.7e-03 … 2.7e-03 | **4.0e-04 … 8.3e-04** |
| a_ice | 2.614e-03 | 6.835e-03 | 1.0e-03 … 1.3e-03 | 1.4e-03 … 1.7e-03 |
| salt | 1.732e-04 | 2.805e-04 | 1.1e-04 … 1.7e-04 | **3.0e-05 … 6.1e-05** |

**The port's DP trajectory is markedly LESS sensitive to a rounding-sized IC nudge than the
Fortran's** — for sst, temp and salt its envelope is 2–4× smaller, and the separation survives the
worst-case seed pairing. So the port is *less* sensitive to perturbation and *more* affected by
single precision. **Normalising each code by its own envelope therefore makes the port look worse,
not better:**

| | fortran, ×own env | port, ×own env |
|---|---|---|
| sst | 1.06 – 1.75 | **4.63 – 8.93** |
| temp | 1.02 – 1.60 | **5.43 – 11.27** |
| a_ice | 2.05 – 2.63 | **4.10 – 4.99** |
| salt | 1.04 – 1.55 | **4.61 – 9.29** |

Reading the tighter-seed family for each code: **upstream's SP sits at ~1–2× its own noise floor;
the port's at ~4–6×.** A factor of about three, robust to which seed is chosen. And it says the
port's SP penalty is *real arithmetic*, not a chaotic-sensitivity artefact — which also fits the
1-month mean-shift result (§3b reading 3): SP carries a systematic component, and a pure re-seeding
of chaos would scale with trajectory sensitivity, which here runs the other way.

⚠️ **TWO SEEDS IS TOO FEW, and the numbers above show it.** Seed-to-seed spread within one family
reaches **50 %** (port, σ=1e-6, sst/temp/salt), so "max over two seeds" is a noisy estimator and the
ranges above are roughly a factor-2 wide. The families also disagree about which is tight — the
Fortran's σ=1e-6 pair agrees to 0.1–10 % while its σ=2e-4 pair spreads 33–40 %; the port is the
opposite. With two members that could easily be chance. **Before any of these ratios goes in a
paper, run ~5 seeds per code.** The *direction* of every statement above is robust; the magnitudes
are not.

✅ **Byte-neutrality confirmed over a full year, not just a 20-step gate:** `pdp_f2` (the `f2` binary,
carrying the forcing descriptor and the perturbation hook) is **BITWISE EQUAL** to `pdp` (the `e3`
binary) across all 12 monthly records of every variable. Serial is bit-reproducible at fixed rank
count, so this is the strongest available confirmation that both changes are byte-neutral in FP64.

⚠️ **The limitation this exposes, and the work it implies.** The two codes' DP trajectories have
themselves diverged by ~1e-02, so "port SP−DP" and "Fortran SP−DP" are sensitivities measured about
*different* trajectories. The clean fix is to normalise each code's SP departure by **its own** noise
envelope — which needs an IC perturbation in the port, since only the Fortran has `&oce_perturb`
today. Porting `do_perturb` (`gen_ic3d.F90`) is small and would make the comparison
apples-to-apples; without it, part of the 1.65–2.6 ratio may be the port's DP trajectory sitting in a
more sensitive region rather than its SP arithmetic being worse.

1-year GPU twin: not started (needs the CUDA flake resolved or a Serial 2N×4-GPU-equivalent CPU
posture).

## 3c. Atmospheric forcing datasets — CORE2 and ERA5 (2026-09-09)

The port read **JRA55-do only**, which blocked the like-for-like comparison against Suvarchal's
60-year runs (CORE2 forcing). It now reads three sets, selected by **`FESOM_FORCING_SET=jra55 |
core2 | era5`** (default `jra55`). A dataset is one descriptor (`fesom_forcing_dataset`,
`src/fesom_jra55.h`); the three entries come from upstream's own namelists, checked against /pool.

**Most of what was needed already existed.** The only genuinely hardcoded thing was one table of 8
names used as *both* the file prefix and the variable name — true only of JRA55. The time-axis
transform was already parametrised (`nm_nc_iyear/imm/idd/freq/tmid`), `fesom_jra_julday` already had
both the gregorian and the 365-day branch, the dimension lookup already tried `LAT/LON/TIME`,
`flip_lat` and the cyclic halo were already derived from the file, `Ntime` was already per field, the
bulk formula already took `z_wind/z_tair/z_shum` as arguments, and `FESOM_CAL_NOLEAP_365` already
existed as a model calendar.

| | CORE2 | ERA5 |
|---|---|---|
| prefix ≠ variable | `ncar_rad.`→`SWDN_MOD`+`LWDN_MOD`, `ncar_precip.`→`RAIN`+`SNOW` | 1:1 (`t2m.`→`t2m`) |
| two fields, one file | yes — **no special case**: the reader keeps one ncid *per field*, so the file is opened twice | no |
| cadence | mixed *within* the set: `t_10` 6-hourly, `ncar_rad` daily, `ncar_precip` **monthly** — already fine, `Ntime` is per field | hourly, 8760/yr |
| time axis | origin 1948, **`tmid=1`** (stamps at interval mid-points) | origin 1900, `tmid=0` — **identical to JRA55** |
| calendar | files `NOLEAP` ⇒ model must be `FESOM_CAL_NOLEAP_365` | `proleptic_gregorian` — no calendar work |
| bulk heights | 10 / 10 / 10 | 🔴 **`z_tair`=`z_shum`=2.0 m** (ERA5 ships 2-metre T and q; wind stays 10 m) |
| runoff name | `runoff.nc` | `CORE2_runoff.nc` |
| size | ~1 GB/var-yr | ⚠️ **28.7 GB/var-yr** — ~260 GB of forcing per simulated year |

**The leap-day jump-over branch in `getcoeffld` stays unimplemented, and now the comment says why.**
Upstream guards it on the *forcing file's* calendar being gregorian-family
(`gen_surface_forcing.F90:878, 1719`); CORE2's files are NOLEAP with **365 records even in a leap
year** (verified on `t_10.1960.nc`), so there is no 29 February in them to step over. It is needed
only for the re-linked-forcing case — a no-leap model against gregorian files.

SSS restoring and runoff differ only in **name**: the files are byte-identical across all three /pool
copies (verified), so that forcing is provably the same in every set.

### Evidence
| test | result |
|---|---|
| 🔴 **Gate 0, set unselected** (27356584, CORE2 preset np8) | **BYTE-IDENTICAL to `ref0`, 3 snapshots.** Making the `jra55` entry reproduce the old literals exactly was the point: a forcing-path change re-opens G0, and this way it *re-passes* instead of being re-derived. |
| Smoke, 20 steps np8, 1958 **and** leap 1960 (27356604, 27356737) | all three sets rc 0, no non-finite, physical T/S. Each resolves its own dataset; in the leap year CORE2 correctly stays at 365 days while ERA5 (8784) and JRA55 (2928) pick up 366. |
| dead-knob check | the three sets give **different** answers (T max 30.23 / 29.94 / 29.87, hf 3.30e3 / 3.90e3 / 3.15e3) — a set that silently fell back to JRA55 would match it exactly. |

**Acceptance test — the one that matters.** The smoke only proves files open. The reader is *right*
only if switching forcing does not enlarge the port-vs-Fortran gap. Port-DP vs Fortran-DP, January
1958, identical mesh/init/dt (27356880/27356908 CORE2, 27357560/27357176 ERA5):

| var | JRA55 | CORE2 | ERA5 | CORE2/JRA | ERA5/JRA |
|---|---|---|---|---|---|
| sst | 3.4906e-03 | 3.7624e-03 | 3.6860e-03 | 1.078 | 1.056 |
| a_ice | 3.1796e-03 | 3.5289e-03 | 3.0411e-03 | 1.110 | 0.956 |
| temp | 2.2938e-03 | 2.3881e-03 | 2.4060e-03 | 1.041 | 1.049 |
| salt | 1.9560e-04 | 1.9615e-04 | 2.0576e-04 | 1.003 | 1.052 |

**All ratios 0.96–1.11: the new readers add essentially nothing to the code-to-code gap**, so that gap
is still the chaotic divergence of §3b, not a reader defect. What gives the test teeth is the
sanity line — the *forcings* separate by **1.1e-02…1.6e-02** (Fortran vs Fortran), 4–5× the
code-to-code gap. A wrong variable, a wrong time origin or a missed `tmid` would have moved the
port's column by about that much; it moved by ≤11 %.

🔴 **Upstream bug found on the way** (`a62f180`): `config/namelist.forcing.era5` is **missing the
`&age_tracer` group** that the JRA and CORE2 templates both carry. `namelist.forcing` is read
sequentially (`gen_model_setup.F90:181-190`), so the scan for `age_tracer` runs past `&nam_sbc`, hits
EOF, and the model dies before step 1 with `forrtl: severe (24)`. That template is stale in a second
way: its paths still point at mistral. `scripts/m16_faith_setup.sh` grafts the group in and repoints
the paths — **worth an upstream PR**. The *port* read ERA5 fine through the same failure, because it
takes its configuration from the descriptor rather than a namelist; only the Fortran arm fell over.

## 3d. Single-precision UNIT tests (2026-09-09) — `ctest` now asserts something about SP

**The finding that prompted this: the port's `ctest` passed 5/5 in the SP build while testing nothing
about single precision.** `calendar`, `io_config`, `io_stream_unit` carry no floating-point type at
all; `test_field`'s round-trip passes in any precision; and `test_ssh_solvers` promoted every
accumulation to `double`, so its assertions were precision-insensitive. Green in an SP build was
evidence of nothing. (Upstream's own SP ctest has the same shape — day-4 handoff: exit code +
success markers, "a robustness check, not a numeric one".)

`tests/test_ssh_solvers.cpp` now carries an **M16 section** that pins the SP floor mechanism, with
`float`/`double` as **explicit** template parameters rather than `real_t` — so it tests the mechanism
and runs identically in a DP and an SP build. Measured, not assumed (probes before assertions):

| fixture | double/double | float/double (**shipped**) | float/float |
|---|---|---|---|
| long solve, 93 iterations | true/rtol **0.97** | **16.60** | 15.95 |
| short solve, 11 iterations | 0.89 | **0.89** | 0.89 |

Three things it asserts, each of which a mutation test confirmed can fail:
1. **Double vectors always meet the tolerance; float vectors miss it by 16× on a long solve.** That
   floor is what `FESOM_SSH_FLOOR` accepts.
2. **The floor grows with ITERATION COUNT, not simply conditioning** — the same matrix solved in 11
   iterations shows no floor at all. That is why the rule is a *multiple of rtol* rather than a fixed
   number.
3. 🔴 **The scalar chain is NOT the cause** (float vs double accumulator moves the floor by ~4 %).
   Asserted deliberately, so that a future reader meeting a floor problem does not try to fix it by
   widening scalars. This is the prediction §2c then confirmed on the real matrix.

**Mutation-checked:** swapping the shipped `float/double` arm to `double/double` turns 3 of the
assertions red. A test that cannot fail is not a test.

⚠️ Still untested at SP: `pipecg`/`oati`'s divergence mechanism (different from the floor — the
recurred residual *grows*), and everything outside the solver. The fixture is a Laplacian, not the
SSH stiffness matrix, so its floor *magnitudes* are a stress case and do not predict the real one
(the real matrix showed ~1.1× rtol); what transfers is the mechanism and its scaling.

## 3e. The port has its own IC perturbation (2026-09-10) — `src/fesom_perturb.{h,cpp}`

§3b-year's limitation was that only the Fortran had `&oce_perturb`, so the port's SP departure had to
be judged against the *Fortran's* envelope — two sensitivities measured about trajectories that have
themselves diverged by ~1e-02. The port now has the facility, so each code can be normalised by its
own envelope.

Knobs (all unset = OFF): `FESOM_PERTURB=1`, `FESOM_PERTURB_MODE=initial_only|first_step`,
`FESOM_PERTURB_METHOD=gaussian|uniform`, `FESOM_PERTURB_SEED` (**required** when on),
`FESOM_PERTURB_TEMP="mean,sigma"`, `FESOM_PERTURB_SALT`. Hook is where upstream's is — after the
tracer IC, **before** the salt anomaly and the ice IC, so the ice cold start sees the perturbed SST
exactly as upstream does.

**🔴 One deliberate divergence: the port's perturbation is PARTITION-INDEPENDENT; upstream's is not.**
Upstream seeds the Fortran intrinsic RNG with `perturb_seed + mype + 37*i` and draws one number per
*local* node, so both seed and draw order depend on the decomposition — the same `perturb_seed` gives
a different field at a different rank count. The port hashes (seed, **global** node id, tracer)
through splitmix64, so the field is a property of the seed alone. This is the M13 lesson
(`ic_extrap_det`, our PR #979): an ensemble whose members cannot be reproduced at another rank count
is not much of an ensemble.

**A second divergence is unavoidable and worth stating: the port cannot be bit-identical to the
Fortran here, and neither can the Fortran to itself.** `random_number` is not specified by the
standard — ifort and gfortran give different sequences from the same seed — so there is no sequence
to match. What *is* matched is the transform and the statistics: one draw per node applied to every
wet level of that column (a 2-D field constant in the vertical, as upstream), the same
Box-Muller/uniform formulae, and upstream's `u1 < 1e-10` clamp.

| check | result |
|---|---|
| **G0 byte gate, knob unset** | **BYTE-IDENTICAL to `ref0`, np1 and np2, 21 snapshots each** |
| reproducible (same seed, rerun) | IC temperature field **identical** |
| **partition-independent (np1 vs np2, same seed)** | IC temperature field **identical** — the divergence works |
| it fires | perturbed vs unperturbed max \|Δ\| 4.02e-03 K |
| amplitude is what was asked | realised σ **1.0223e-03 K** against 1.0e-03 requested (3140 nodes) |
| ctest, both precisions | 5/5 |

**Refused rather than guessed** (dead-knob discipline): `FESOM_PERTURB=1` without a seed, an
unrecognised mode or method, and `first_step` **on a restart** — the port's hook sits before the
restart read, so that combination would perturb a state the restart then overwrites. Upstream
supports it; the port refuses it loudly instead of silently perturbing nothing. That is the one
capability gap against `&oce_perturb`.

## 3f. WHY the port is worse than the Fortran at SP — the trail so far (2026-09-10)

**Step 1: it is SALINITY, and it is there from month 1.** Month-by-month, SP−DP against each code's
own DP arm:

| | month 1 | port / fortran |
|---|---|---|
| **salt** | fortran 5.97e-06 · port **1.65e-05** | **2.77** |
| temp | fortran 1.14e-04 · port 1.64e-04 | 1.43 |
| sst | fortran 9.39e-05 · port 9.45e-05 | 1.01 |

One month is 1488 steps — far too short for chaotic amplification to produce a factor 2.8, so this is
a **static arithmetic difference**, not amplification. sst and temp start equal and diverge later,
which is the signature of salinity error feeding density and density feeding everything else. The
mean drifts confirm it is systematic and structural: the Fortran's salt mean drifts **positive** and
monotonically (+1.8e-06 → +8.2e-05), the port's **negative** (−5.2e-06 → −2.9e-05). **Opposite signs.**

**Step 2: the absolute-salinity path carries much of it.** Upstream's #986 salt anomaly stores S−35
instead of absolute S — float32 eps at S≈35 is ~2.4e-6 psu, larger than the per-step surface
freshwater increments. Both codes default it OFF, which is how the whole matrix ran. Turning it on in
**both** (1 month, jobs 27377022 / 27376304):

| var | fortran OFF | port OFF | ratio | fortran ON | port ON | ratio |
|---|---|---|---|---|---|---|
| **salt** | 5.97e-06 | 1.65e-05 | **2.77** | 4.90e-06 | 8.60e-06 | **1.76** |
| temp | 1.14e-04 | 1.64e-04 | 1.43 | 9.94e-05 | 1.18e-04 | **1.19** |
| sst | 9.39e-05 | 9.45e-05 | 1.01 | 7.05e-05 | 1.16e-04 | 1.64 |
| a_ice | 5.55e-04 | 5.44e-04 | 0.98 | 5.29e-04 | 5.68e-04 | 1.07 |

The anomaly **halves the port's salt SP−DP** (0.52×) while moving the Fortran's only 18 % (0.82×) —
i.e. the port was losing far more to absolute salinity than upstream was. The salt gap closes from
**2.77 → 1.76** and temp from 1.43 → 1.19. ⚠️ **sst moves the wrong way** (1.01 → 1.64); that is one
month and one realisation of a surface field, so it is not yet a fact — do not build on it.

**Step 3: 🔴 upstream PR #1054 (opened 2026-09-10, Jan Streffing) names the exact mechanism, and the
port has the identical defect.** *"In single-precision builds the FCT path does not conserve tracer
content. `oce_tra_adv_flux2dtracer` recovers the low-order tendency as `LO*hnode_new − ttf*hnode` …
Where the per-step change is below half an ulp of the tracer, `LO` rounds back to `ttf` and that
cell's low-order flux divergence is lost."* Measured upstream in coupled AWI-ESM3: SP −0.50 W/m²
against DP −0.01, and +0.01 with the fix; the deficit held at −0.52 W/m² over 40 years. It is an
explicit follow-up to #940, the PR M16 ports.

**The port's line is the same one** (`src/fesom_tracer_adv.cpp`, `flux2dtracer_fct`):
```c
dttf_v[k] += -ttf[k] * mesh->hnode[k] + lo[k] * mesh->hnode_new[k];
```
and `fesom_tracer_compute_fct_LO` builds `LO` by the same `(ttf*hnode + tend)/hnode_new` formula. The
mechanism explains the variable ordering exactly: the loss scales with the tracer's **ulp**, and
ulp(35 psu) ≈ 2.4e-6 against ulp(4 °C) ≈ 2.4e-7 — ten times larger, which is why salt is hit hardest
and why the S−35 anomaly halves it.

⚠️ **What is NOT yet explained.** #1054 is a *shared* defect — the Fortran has it too — so on its own
it does not explain why the port is worse. One structural difference is already visible and points
the *other* way: the Fortran evaluates `(dttf_v − A) + B` left-to-right while the port computes
`dttf_v + (−A + B)`, forming the small difference first, which is the better association. So the
port's residual 1.76× on salt is still unaccounted for. **Next probe: port the #1054 fix and
re-measure** — it is a direct transliteration, and if the residual is the low-order tendency loss it
should close further.

### 3g. #1054 PORTED — and it does NOT explain the port/Fortran gap (2026-09-10)

Ported into **both** the live Kokkos kernel (`fesom_tracer_advect_one_fct_kk`) and the host C twin,
in the three places upstream touches: keep `lo_tend` at the LO finalisation, clear the flag where the
implicit vertical split rewrites `fct_LO`, and add the kept tendency in `flux2dtracer` instead of
recovering it. SP-gated exactly as upstream gates it. Knob `FESOM_FCT_LO_FLUXFORM` (default 1 =
upstream behaviour) exists so the defect can be **measured**, not just asserted.

🔴 **Process note — the fix first went into DEAD CODE.** `fesom_step.cpp` calls the Kokkos kernel;
the host C functions with the same names are twins used by a probe and the host driver. The first
build, first byte gate and first A/B all passed while changing nothing that runs. **What caught it
was the announcement line failing to appear in the run log** — the instrument, not the gate. A gate
that passes on a no-op looks exactly like a gate that passes.

**Result (1 month, CORE2/JRA55, dt 1800; jobs 27377912/13/14 — the Fortran column is unchanged):**

| var | fortran | port PRE-#1054 | ratio | port WITH #1054 | ratio | change |
|---|---|---|---|---|---|---|
| **salt** | 5.969e-06 | 1.652e-05 | **2.77** | 1.655e-05 | **2.77** | **1.00×** |
| temp | 1.144e-04 | 1.637e-04 | 1.43 | 1.576e-04 | 1.38 | 0.96× |
| sst | 9.393e-05 | 9.449e-05 | 1.01 | 1.205e-04 | 1.28 | 1.28× |
| a_ice | 5.547e-04 | 5.443e-04 | 0.98 | 5.135e-04 | 0.93 | 0.94× |

The fix is live and does change the solution (`psp_on` vs `psp_off`: relL2 5.0e-06 salt … 4.8e-04
a_ice). **But it leaves the salt gap at exactly 2.77 — it explains none of it.** temp and a_ice move
4–6 % in the right direction, sst 28 % the wrong way; at one month sst has already been seen to move
the wrong way in the salt-anomaly test too, so it is not a reliable discriminator here.

**Two conclusions, and they are separate.**
1. **KEEP the fix.** #1054 addresses a real *conservation* defect that upstream measured at −0.50
   W/m² against DP's −0.01, holding at −0.52 W/m² over 40 years. **This test measured the wrong
   thing for that claim** — a 1-month relL2 state difference is not a 40-year conservation drift.
   The correct validation is `FESOM_MP_CONSERV` with the knob on and off, which is the next task and
   is also the check that says whether the port carried the defect at upstream's magnitude.
2. 🔴 **The port/Fortran gap remains UNEXPLAINED.** The salinity localisation (§3f) still stands —
   it is salt, and it is there at month 1 — and the #986 anomaly still halves it. But the FCT
   low-order recovery is not the cause. This is consistent with the structural observation in §3f:
   the port already forms the small difference first (`dttf_v + (−A + B)`) where the Fortran does
   not, so the port was *less* exposed to that particular line, not more.

**Next probe:** the defect is visible at month 1, so run both codes SP and DP for a handful of steps
and diff the salinity field directly — at ~10 steps the difference is nearly pure arithmetic and can
be bisected within the step. Candidates from the registry's own suspect list, in order: the
EOS/pressure-gradient chain ("promote FIRST"), then the class-3 flips (things July kept in `dbl_t`
that M16 flipped to `real_t` to match upstream — if any flip was wrong, the port is worse than
upstream by construction).

### 🔴 3h. FOUND IT — the port's SP salt error is in the INITIAL CONDITION, not the timestepping

**Short-horizon bisection** (20 steps, CORE2/JRA55, dt 1800; jobs 27378655 / 27378689), per-step
output on both sides, SP−DP within each code:

| step | fortran salt | port salt | ratio |
|---|---|---|---|
| 1 | 2.221e-06 | **1.077e-05** | **4.85** |
| 5 | 2.322e-06 | 1.074e-05 | 4.62 |
| 10 | 2.827e-06 | 1.087e-05 | 3.84 |
| 20 | 2.801e-06 | 1.093e-05 | 3.90 |

🔴 **The port's salt SP−DP is FLAT** — 1.077e-05 at step 1, 1.093e-05 at step 20, a 1.5 % change over
20 steps. It does not accumulate. Whatever creates it happens **once, before the dynamics run**. The
Fortran's starts 4.85× lower and *grows* (2.22e-06 → 2.80e-06, +26 %), which is what ordinary
per-step rounding looks like.

**Confirmed directly** by dumping the initial condition (`FESOM_RESTART_IC`, partition-independent by
construction, job 27378753) before any timestep:

| | port IC, SP vs DP |
|---|---|
| **salt** | relL2 **1.023e-05** — i.e. **95 % of the step-1 value of 1.077e-05** |
| temp | relL2 4.858e-06 |

**And it is not rounding — it is a hundred broken points.** Over 3 705 892 wet points: median
|Δ| = 1.34e-06 psu = **0.4× float eps at S=35**, i.e. the bulk is clean. But **99.6 % of the sum of
squares lives in the top 100 points**, 166 points exceed 1e-3 psu, 102 exceed 1e-2, and the worst is
**0.118 psu**. A diffuse narrowing loss cannot do that; an *iterative* process converging to a
different answer at isolated points can, and that is what the IC hole fill (`extrap_nod3D`) is.

**The deterministic fill halves it but does not fix it** (job 27378806, `FESOM_IC_EXTRAP=det` in both
precisions):

| fill | relL2 | max Δ | points > 1e-2 psu |
|---|---|---|---|
| legacy | 1.023e-05 | 0.118 psu | 102 |
| **det** (`ic_extrap_det`, our PR #979) | **4.193e-06** | 0.074 psu | 177 |

So the locus is the fill itself, in either variant, run at `real_t`. This is precisely the registry's
class-3 row *"PHC climatology / init path … detector: same-IC gate (det fill at SP equals DP fill to
the rounding class)"*, status **flip-B** — **the flip was made and its own designated detector was
never run.** It has now been run, and it fails.

### 🔴 3h-b. Upstream is NOT clean either — the defect is SHARED, the port just has it worse
The inference in the paragraph below was **wrong** and is corrected here. Comparing the two codes
like for like at **step 1** (both fields are IC + exactly one step; salt, wet points only — note the
Fortran writes `9.969e+36` at dry points where the port writes 0, so the fill mask is load-bearing):

| | relL2 | median \|Δ\| | max \|Δ\| | pts > 1e-3 psu | pts > 1e-2 | top-100 share |
|---|---|---|---|---|---|---|
| **Fortran** | 2.221e-06 | 2.05e-06 (0.54× eps) | **3.62e-02 psu** | 1303 | **29** | 62.8 % |
| **port** | 1.077e-05 | 3.17e-06 (0.83× eps) | **1.18e-01 psu** | 4345 | **180** | 89.6 % |

**Upstream's float hole fill produces the same isolated broken points** — up to 0.036 psu, 29 of them
past 1e-2. It is not a port-only defect. But **the port's instance is ~3× worse in magnitude and ~6×
in count**, which is where its 4.85× step-1 excess comes from.

So both conclusions hold at once:
1. **This is an upstream SP bug worth reporting**, same family as our #979 (`ic_extrap_det`): at
   single precision the IC extrapolation lands on visibly different values at isolated points, and
   nothing in either code checks it. Promoting the fill to `dbl_t` costs nothing (once, at startup).
2. **The port has an additional factor of ~3 to find** on top of the shared defect.

Also worth noting: one timestep **spreads** the damage — the port's IC has 166 points past 1e-3, its
step-1 field has 4345. The bad points are advected and diffused into their neighbourhoods immediately.

⚠️ *(superseded by 3h-b above — kept for the record)* **What was NOT established: that upstream's IC is clean.** The Fortran's step-1 error of
2.22e-06 sits near the bulk-rounding level, which *suggests* its fill does not produce the
pathological points — but that is an inference, not a measurement. **Next check: dump the Fortran's
IC the same way.** It decides the response:
- if upstream is clean → the port's fill has a specific defect to fix;
- if upstream is equally corrupted → this is an **upstream SP bug worth reporting**, in the same
  family as #979 (`ic_extrap_det`), and both codes need the IC path promoted to `dbl_t`.

Either way the fix direction is the registry's own promotion order: **run the IC extrapolation in
`dbl_t`**. It is computed once at startup, so there is no runtime cost — and note it would make the
port's SP IC *better* than upstream's, which is a deliberate divergence to raise before taking.

### 🔴 3i. The IC defect: a mixed-precision comparison in the PHC bracket search — found and fixed
*(⚠️ read §3i-b: this is a real defect worth 216× at t=0, but it is NOT the factor of 3.)*

**The port's excess SP initial-condition error was caused by keeping the PHC grid axes in `double`
while the node coordinate is `real_t`.** That is, by being *more* accurate than upstream in one
place and not the other.

`binarysearch_d` takes an exact-match shortcut, `if (fabs(arr[middle] - value) <= 1e-9) return
middle;`. Upstream compares WP against WP (`gen_ic3d.F90:955`, `d = 1e-9_WP`), so at SP **both sides
are float** and a node lying exactly on a grid line still matches. The port kept `nc_lat`/`nc_lon` in
double and therefore compared a **double** gridline against a **float-rounded** coordinate:
|70.5 − 70.50000381| = 3.8e-06, far past the 1e-9 tolerance. The exact match is missed, the bracket
lands **one cell away**, and where that neighbouring cell is land the node flips from "interpolated
from PHC" to "left as a hole for `extrap_nod3D`" — a completely different value.

Caught by tracing the three flipping nodes (`FESOM_IC_TRACE_GID`), which showed it outright:

| | i | j | corners | outcome |
|---|---|---|---|---|
| **DP** | 169 | **160** | 28.41, 28.76, 29.13, 29.42 | interpolated |
| **SP** | 169 | **159** | **1e10, 1e10**, 28.41, 28.76 | `surface_skip=1` → hole |

And the confirmation is in the mesh: **all three flipping nodes sit exactly on PHC grid lines** —
gid 49751 and 69121 at latitude **70.5000**, gid 33626 at **51.5000**.

**Fix** (`FESOM_IC_BRACKET_MIXED=1` restores the defect for A/B): narrow the axis to `real_t` for the
bracket comparison, so both sides are consistent — which is what upstream does. FP64 unaffected
(`real_t == double` there); **G0 byte gate re-passed BYTE-IDENTICAL at np2**.

| PHC initial condition, SP vs DP | pre-fill holes DP / SP | differing | relL2 salt | max \|Δ\| | pts > 1e-3 |
|---|---|---|---|---|---|
| before | 587099 / 587096 | **13** | 1.023e-05 | 1.185e-01 psu | 166 |
| **after** | **587099 / 587099** | **0** | **4.736e-08** | **2.43e-05 psu** | **0** |

**216× on relL2, 4880× on the worst point, and the hole sets are now identical.** The residual
4.7e-08 is pure rounding. The port's SP initial condition is now *better* than upstream's, whose
step-1 field still carries 29 points past 1e-2 psu from a residual whose cause is not established
here.

**Dead ends, recorded because they cost jobs and would otherwise be retried:** the bilinear weights
(`FESOM_IC_INTERP_DBL`) and the depth-bracket search key (`FESOM_IC_DEPTH_DBL`) were both promoted to
`dbl_t` and both changed the result *not at all* — bit-identical hole counts and relL2. Neither was
the cause. Both knobs are kept as instruments.

**Worth reporting upstream**, with care about what is claimed: upstream's float-vs-float comparison
protects it from *this* flip, so this exact defect is the port's. But the underlying fragility — an
IC bracket search whose outcome depends on the working precision, at nodes that sit on grid lines —
is structural, and upstream's own 29 bad points say it has a residual of its own.

### ⚠️ 3i-b. …but it is NOT the factor of 3. The end-to-end gain is 9 %.
The heading of §3i was written before the end-to-end measurement and **overclaimed**. Repeating the
1-month comparison with the fixed binary (job 27383020):

| var | fortran | port BEFORE | ratio | port AFTER fix | ratio |
|---|---|---|---|---|---|
| **salt** | 5.969e-06 | 1.652e-05 | **2.77** | 1.510e-05 | **2.53** |
| temp | 1.144e-04 | 1.637e-04 | 1.43 | 1.593e-04 | 1.39 |
| sst | 9.393e-05 | 9.449e-05 | 1.01 | 1.052e-04 | 1.12 |
| a_ice | 5.547e-04 | 5.443e-04 | 0.98 | 6.225e-04 | 1.12 |

**A 216× improvement at t=0 buys 9 % after one month.** So the initial-condition corruption, though
real and now fixed, is **not** what makes the port worse at any horizon anyone cares about. It
dominated the 1–20 step window — which is exactly why the bisection found it there, and why "the
port's salt error is flat over 20 steps and equals the IC" was true and still misleading. By 1488
steps the IC error has been mixed away and something else dominates.

**The corrected picture, and the sharper question.** With the IC now clean (SP−DP 4.7e-08 at t=0),
the two codes start together and the port's error grows **≈3× faster during integration**: the
Fortran goes 2.22e-06 (step 1) → 5.97e-06 (month 1), a factor 2.7; the port now starts near rounding
and reaches 1.51e-05, a far steeper climb. **So the remaining factor of 3 is in the TIMESTEPPING after
all — it was simply invisible at 20 steps because the IC error masked it.** A post-fix 20-step
bisection (job 27383261) is running to pin the growth curve now that the offset is gone.

⚠️ sst and a_ice moved the *wrong* way (1.01 → 1.12, 0.98 → 1.12). One month, one realisation, and
both variables have moved the wrong way before in this campaign — not a fact, but not dismissable
either.

**Keep the fix regardless**: it is byte-neutral in FP64, it costs nothing, and an initial condition
that depends on working precision at 100 points is wrong on its own terms.

### 3i-c. The post-fix growth curve — the remaining factor splits cleanly in two
Post-fix 20-step bisection (job 27383261), salt SP−DP against each code's own DP arm:

| | step 1 | step 20 | month 1 | growth step1→month |
|---|---|---|---|---|
| fortran | 2.221e-06 | 2.801e-06 | 5.969e-06 | **2.7×** |
| port BEFORE the IC fix | 1.077e-05 | 1.093e-05 | 1.652e-05 | 1.5× |
| **port AFTER the IC fix** | **3.541e-06** | 4.150e-06 | 1.510e-05 | **4.3×** |
| ratio port/fortran | **1.59** | 1.48 | **2.53** | |

**The IC fix cut the step-1 ratio from 4.85 to 1.59** — that part was real and is now banked. What
remains factorises almost exactly:

* **≈1.59× is generated in the FIRST TIMESTEP.** With the IC at 4.7e-08 (rounding), the port's
  3.54e-06 after one step is made *by the step*, against the Fortran's 2.22e-06. Single-step, single
  precision, same IC — a pure arithmetic difference in the step itself.
* **≈1.6× more is accumulated over the month** — the port grows 4.3× from step 1 to month 1 where
  the Fortran grows 2.7×.
* 1.59 × 1.6 ≈ 2.5, which is the observed month-1 ratio of 2.53.

**This is the sharpest form of the question yet, and it is now cheap to attack:** one timestep, one
tracer, identical initial conditions, 1.59× more single-precision error. That is bisectable *within*
the step — the registry's suspect order applies (EOS/pressure-gradient chain first, then the class-3
flips), and a per-routine dump over a single step separates them without any long integration.

### 🔴 3j. WITHIN-STEP BISECTION — it is the FCT tracer advection
With the IC clean to rounding (§3i), salt was dumped at four points of step 1 in both precisions
(`FESOM_SALT_TRACE`, job 27383405). The port's SP−DP against its own DP arm:

| stage | relL2 | change |
|---|---|---|
| A — step entry | 4.736e-08 | — (the clean IC) |
| B — after forcing / ice-ocean coupling, before advection | 4.736e-08 | **1.00×** |
| **C — after `fesom_tracer_advect_one_fct_kk(S)`** | **2.865e-06** | **60.5×** |
| D — after implicit vertical diffusion | 3.541e-06 | 1.24× |

**The FCT tracer advection generates essentially all of it — a 60× jump in one call.** The surface
salinity flux and the whole ice-ocean coupling contribute *nothing* (1.00×, bit-for-bit at this
resolution), which rules out the entire forcing path in one measurement. Vertical diffusion adds a
further 24 %, and D matches the independently measured step-1 value of 3.541e-06 exactly, so the
four stages account for the whole step.

For scale: the Fortran's step-1 total is 2.221e-06, so its *entire* step generates less than the
port's FCT call alone.

**Note what this does NOT say.** #1054 lives in this same routine and is already ported (§3g), and it
moved the ratio not at all — so the remaining excess is a *different* defect inside FCT. The
registry lists "FCT tracer advection, ice FCT" as `suspect-fp32`. The live candidates now are the
Zalesak limiter bounds, the MFCT high-order flux, and the edge→node scatter accumulations — the port
fuses several of these loops (the M5.21 "flat lever"), which reorders float accumulation relative to
the Fortran. **Next: bisect inside FCT the same way — dump at the limiter boundary and around the
scatter.**

### ✅ 3k. FIXED — IC bracket fix + the salt anomaly bring the port to parity (2.53 → 0.97)
1 month, CORE2/JRA55, dt 1800, salt SP−DP against each code's own DP arm:

| configuration | fortran | port | ratio |
|---|---|---|---|
| baseline (port already has the §3i IC fix) | 4.897e-06 … 5.969e-06 | 1.510e-05 | **2.53** |
| **+ salt anomaly (#986) in BOTH codes** | 4.896e-06 | **4.739e-06** | **0.97** |

**The port is now marginally BETTER than upstream on salt.** temp improves too (1.39 → **1.10**).

**The tell is in how much each code gains from the anomaly: fortran 1.22×, port 3.19×.** Both codes
store absolute salinity ≈35 in float, but the port had ~1e-05 of magnitude-driven error to give back
where the Fortran had ~1e-06. That is the port-specific factor, and it is consistent with everything
measured: it is salt-only (§3f), it is generated in FCT (§3j), and #986 — which exists precisely to
stop float from spending its digits on the constant part of S — removes it.

**The practical fix, then, is two things:**
1. the §3i IC bracket fix (port-side, byte-neutral in FP64, and correct on its own terms);
2. **run SP with `FESOM_SALT_ANOMALY=1`** — upstream's own measure for exactly this, and already the
   M16 recipe (board §2's 30-day conservation twin reached the same conclusion from a different
   direction: "`FESOM_SALT_ANOMALY=1` is part of the SP recipe").

⚠️ **What is fixed and what is only masked.** The anomaly *removes the symptom* by making the working
values O(1); it does not explain why the port is more sensitive to the absolute magnitude than
upstream inside FCT. That mechanism is still unidentified, and it would resurface in any
configuration that runs absolute salinity. The honest statement for the paper is: *at SP the port
matches upstream once both use the salt anomaly, which upstream recommends for SP anyway; without it
the port is ~2.5× worse on salt and the reason sits inside the FCT advection.*

⚠️ Also unresolved: **sst gets worse with the anomaly** (1.12 → 1.47) while salt and temp improve.
One month, one realisation, and sst has moved unpredictably at this horizon throughout the campaign —
flagged, not explained.

**Also settled here: PR #1054 is neutral on this metric even with a clean IC** — 1.5103e-05 (on) vs
1.5076e-05 (off), a factor 0.998. It is a *conservation* fix and conservation is a different
measurement; keep it, and validate it with `FESOM_MP_CONSERV` rather than with a state difference.

### 3l. INSIDE FCT: the loss is at flux FORMATION, not in any accumulation — and that is why #986 is a real fix
The anomaly gain (port 3.19× vs upstream 1.22×) said the port has an operation whose rounding scales
with |S|≈35. Dumping the FCT internals with the anomaly on and off localised it precisely — mean
absolute SP−DP error, step 1, salt:

| intermediate | anomaly OFF | anomaly ON | gain |
|---|---|---|---|
| `LO` (low-order solution, psu) | 2.088e-06 | 1.450e-06 | 1.4× |
| **`del_ttf` BEFORE the ALE term (psu·m)** | **2.356e-03** | **2.691e-05** | **87.5×** |
| `del_ttf` after the ALE term | 2.368e-03 | 2.701e-05 | 87.7× |
| final salinity (psu) | 1.916e-05 | 1.604e-06 | 11.9× |

The whole magnitude sensitivity is in the **increment**, and it is **already complete before the ALE
reconstruction** — so it is made in the flux sums, not in the reconstruction.

**Two fixes were then implemented and measured, and BOTH are no-ops.** Recorded because they are the
obvious things to try and each cost a build and a job:

| attempted fix | knob | 1-month salt SP−DP | effect |
|---|---|---|---|
| ALE reconstruction term in `dbl_t` | `FESOM_FCT_ALE_DBL` | 1.5103e-05 → 1.5093e-05 | **0.999×** |
| whole increment (`dth`/`dtv`) accumulated in `dbl_t` | `FESOM_FCT_INC_DBL` | 1.5103e-05 → 1.5067e-05 | **0.998×** |

Both default to **0** now: they allocate two extra `dbl_t` fields and buy nothing.

🔴 **What that null result actually means, and it corrects the framing of §3k.** The error is not
*created* by any summation — it is created when the **flux is formed** from a tracer already stored
in float at ≈35 psu, whose absolute resolution is only ~2e-06 psu. Every flux inherits that, and the
divergence then exposes it because the net convergence is far smaller than the fluxes it is made
from. **No accumulator can recover information destroyed at storage.**

So **#986 is not a mask — it is the correct fix for a representational problem.** Subtracting the
reference salinity is the only way to give the stored value more usable digits. §3k's caveat ("the
anomaly removes the symptom, not the cause") was wrong: the cause *is* the stored magnitude.

⚠️ **Still open: why the port loses ~4× more than upstream to the same representational limit.** Both
store S in float at ≈35 and both use flux-form advection. Ruled out so far: the IC (§3i, fixed
separately), the #1054 low-order cancellation, the ALE reconstruction, and every increment
accumulation. What remains is **flux formation itself** — the MFCT high-order horizontal flux, the
QR4C vertical flux, and the upwind fluxes — where the port may simply perform more float operations
on ≈35-sized values than upstream does. That is the next place to look, and it is now the only place
left.

### 🔴🔴 3m. THE MECHANISM: the antidiffusive flux `HO − LO` cancels, and the loss scales with |S|
Dumping every FCT intermediate per level, SP vs DP, **relative** error (port, step 1, salt, no anomaly):

| level | bounds `fct_ttf_max` | limiter `fct_plus` | **raw antidiff flux** | limited flux |
|---|---|---|---|---|
| 6 | 3.1e-05 | 5.8e-04 | **1.04e-02** | 1.17e-02 |
| 18 | 3.5e-05 | 6.0e-04 | **1.96e-02** | 2.06e-02 |
| 30 | 1.6e-04 | 3.7e-03 | **3.81e-02** | 4.33e-02 |
| 42 | 7.3e-04 | 5.5e-03 | **2.57e-02** | 3.25e-02 |

**The percent-level error is already in the RAW antidiffusive flux, before the limiter touches it.**
The bounds (3e-05) and the limiter factors (5e-04) are 10–100× cleaner, so neither the Zalesak
bounds' `tvert_max − LO` cancellation nor the limiter is the cause.

**The mechanism.** The antidiffusive flux is formed as `HO − LO`
(`o_init_zero=.false.` → `flux = HO_expression − flux_LO`, `oce_adv_tra_ver.F90:410`, and the port's
identical `init_zero=0` path). Both fluxes are **proportional to the ABSOLUTE tracer**, ≈35 psu for
salinity, and in the smooth deep interior they agree to roughly **1 part in 10⁵–10⁶**. In float that
subtraction leaves one or two significant digits, so the antidiffusive flux carries **1–4 % relative
error**. It then enters `del_ttf`, whose absolute error grows **130× with depth** (6.6e-05 →
8.6e-03 psu·m, §3l) because the fluxes scale with layer thickness — and the final salinity error sits
**16–18× above the float floor at levels 20–40**, while upstream sits *at* the floor (~2e-06 psu).

**This explains every earlier observation at once:** why it is salt and not temperature (ulp(35) is
10× ulp(4)); why it appears in FCT and nowhere else (§3j); why `LO` itself is clean at 2e-06 (§3l) —
LO is a *value*, not a difference of large fluxes; why widening any accumulator was a no-op (the
information is destroyed when the difference is formed, not when it is summed); and why #986 buys
**87×** — shifting S by −35 shrinks both fluxes ~35× while leaving their difference unchanged,
because **both schemes are consistent for a constant field**, so `HO(T+c) − LO(T+c) = HO(T) − LO(T)`.

### The fix that is not a mask
That last identity is the fix. The antidiffusive flux is **invariant under a constant shift of the
tracer**, so it can be computed from `T − T_ref` *inside the flux routines* — mathematically
identical, numerically free of the cancellation — **without** changing the model state, for **every**
tracer, and **regardless of whether the user enables #986**. #986 achieves the same thing globally
and is the reason it works; doing it locally in FCT makes it unconditional.

Proof of concept already measured: with #986 on, the port's salt SP−DP is 4.739e-06 against
upstream's 4.896e-06 — **parity** (§3k).

⚠️ Still not explained: upstream forms `HO − LO` the same way yet sits at the float floor at depth
where the port is 16–18× above it. The cancellation is shared; its *severity* is not. Candidates now
narrow to the flux expressions themselves — the port's `adv_tra_ver_qr4c`/`adv_tra_hor_mfct` versus
upstream's — and specifically to how many float operations each performs on ≈35-sized values before
the subtraction. That is the next and, on this evidence, last place to look.

### 3n. The subtraction is NOT the loss — the error is inherited from the flux computations
§3m's candidate was that the port rounds `HO` to float into an explicit temporary immediately before
subtracting `LO`, where upstream writes one Fortran expression that ifort can contract into an FMA.
Implemented (`FESOM_FCT_HO_DBL`: form the whole `HO − LO` in `dbl_t`, all four branches of
`fct_qr4c_v`) and measured — **bit-for-bit identical, at every level**:

| level | legacy float | `HO−LO` in `dbl_t` | gain |
|---|---|---|---|
| 12 | 1.345e-02 | 1.345e-02 | **1.0×** |
| 30 | 3.812e-02 | 3.812e-02 | **1.0×** |
| final salinity, all levels | — | — | **1.0×** |

Default flipped to **0**. That null is informative rather than disappointing: **the 1–4 % is already
present in `HO` and in `LO` before they meet.** Each flux is computed from a **float tracer of
magnitude ≈35**, so each carries an absolute error ≈ |flux|·6e-08; their difference is ~1e-06 of
|flux|, so the inherited error is ~6e-08·1e+06 ≈ **6 % relative** — which is what is measured. No
arithmetic performed *at* the subtraction can recover it, exactly as widening the accumulators could
not (§3l).

**So the chain is closed on the mechanism side:** the only lever is the tracer's magnitude when the
flux is formed. `HO(T+c) − LO(T+c) = HO(T) − LO(T)` because both schemes are exact for constants, so
computing the FCT fluxes from `T − T_ref` is mathematically identical and ~35× more accurate for
salinity. **#986 is that shift, applied globally — which is why it delivers parity (§3k) and why it
is the right setting for SP rather than a workaround.** Generalising it *inside* FCT would make it
unconditional and extend it to temperature.

⚠️ **What remains genuinely unexplained.** Upstream computes its fluxes from a float tracer of the
same magnitude, so it should inherit the same ~6 % — yet its salinity error sits **at** the float
floor (~2e-06 psu) at depth where the port is 16–18× above it. Every shared-mechanism explanation
now fails to account for that asymmetry, and the remaining way to settle it is to **instrument the
Fortran** (`oce_adv_tra_ver.F90`) and dump its antidiffusive flux for the same step, rather than
inferring upstream's internals from its output. That is a rebuild of the oracle with a dump, which is
straightforward — the oracle build recipe is in §0 — and it is the honest next step rather than
another hypothesis.

## 4. Untested list (kept honest)
- every M14 recipe knob at SP (G3); CA solvers `pipecg`/`pcsi`/`cg2` at SP; `FESOM_FORCING_POINTSLOPE`
  DP control leg; TKE `dbl_t` give-back; stiffness-shadow device-memory give-back.
