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

### 🔴🔴 3o. THE FORTRAN INSTRUMENTED: same relative error, but the port's antidiffusive flux is 500–12000× LARGER
`src/oce_adv_tra_ver.F90` in `~/fesom2_sp` now dumps the raw antidiffusive vertical flux in the
port's binary layout (`FESOM_FCT_TRACE`, first step, tagged `tr0`=T / `tr1`=S). Staged as
`oracle/{dp,sp}_instr` so the **validated oracle is untouched**; `job_m16_faith_fortran` selects it
with `ORACLE_VARIANT=_instr`. Salinity, step 1, both codes:

| level | \|flux\| fortran | rel err F | \|flux\| port | rel err P | **mag P/F** | err P/F |
|---|---|---|---|---|---|---|
| 6 | 1.274e+02 | 7.89e-03 | 1.028e+05 | 1.04e-02 | **807** | 1.32 |
| 18 | 7.961e+02 | 1.45e-02 | 4.355e+05 | 1.96e-02 | **547** | 1.35 |
| 30 | 5.501e+02 | 2.79e-02 | 9.807e+05 | 3.81e-02 | **1783** | 1.37 |
| 42 | 6.005e+01 | 1.90e-02 | 7.479e+05 | 2.57e-02 | **12455** | 1.35 |

**Two facts, and the second is the lead.**

1. **The relative error is the same in both codes** — 0.94–2.07×, typically ~1.35. So the precision
   mechanism of §3m/§3n is genuinely *shared*: upstream's antidiffusive flux is just as damaged, in
   relative terms, as the port's. That closes the question "does upstream avoid the cancellation" —
   it does not.
2. 🔴 **The port's antidiffusive flux is 500–12000× LARGER in magnitude, and the ratio grows with
   depth.** That is far too large to be a precision effect. Since both codes' FP64 solutions agree to
   ~2e-03, the physics cannot actually differ by that much — so either `area`/`areasvol` are scaled
   consistently between the two (cancelling in the divergence), or **the port's `adv_flux_ver` does
   not hold the low-order flux when `qr4c` subtracts it**, in which case the port's "antidiffusive"
   flux is close to the *full* high-order flux rather than the small `HO − LO` correction.

**Why that would explain the 16–18× final gap.** A hugely oversized raw antidiffusive flux is then
*clipped by the Zalesak limiter* in most cells, so the port's result becomes governed by the **bounds**
`tvert_max − LO` — which are exactly the ≈35-psu cancellation — while upstream's raw flux is small
enough that the limiter rarely binds and its result stays governed by the (relatively cleaner) flux.
That is consistent with §3l's observation that the port's limiter factors carry 3.4e-04…5.5e-03
relative error while its bounds carry 3e-05…7e-04: a limiter that binds everywhere transmits them.

**Next: settle the magnitude.** Compare `area`/`areasvol` between the codes at the same node/level,
and dump the port's `adv_flux_ver` immediately *before* `qr4c` runs to confirm it holds the low-order
flux. Both are single short jobs and one of them is the answer.

⚠️ **Method correction worth recording:** the first version of this comparison dumped the Fortran's
**first** `qr4c` call, which is *temperature*, and compared it against the port's *salinity* — the
routine takes no tracer index, so "first call" is not "tracer 1". The instrument now counts calls and
tags `tr0`/`tr1`. The numbers above are the corrected, same-tracer comparison.

### 🔴🔴🔴 3p. A REAL BUG: the port never seeds the Adams-Bashforth history
Both §3o follow-ups ran and the second one answered it.

**Test 1 — the areas are identical.** `area` and `areasvol` agree between the codes to **1.0000** at
every level, so the 500–12000× flux-magnitude gap was never a units or scaling artefact.

**Test 2 — the port's `adv_flux_ver` before `qr4c`, versus after.** `|HO − LO| / |LO| = 0.600` at
**every level**, to three digits. That number is not physical: it is the Adams-Bashforth coefficient.

**The bug.** Upstream seeds the AB history before the first step
(`oce_setup_step.F90:310`: `tracers%data(n)%valuesold(i,:,:) = tracers%data(n)%values`), so at step 1
`valuesAB = −(0.5+ε)·valuesold + (1.5+ε)·values = values`. **The port allocates `valuesold`
zero-initialised and never assigns it**, so at step 1

    valuesAB = 1.6 · values

and every high-order flux built from `valuesAB` is **60 % too large**, making the antidiffusive flux
`HO − LO = 0.6·LO` instead of the ~1e-04·LO correction it should be.

**Fixed and verified** (`FESOM_AB_INIT_OLD=1`, job 27391185) — the port's antidiffusive flux now
matches upstream's **exactly**:

| level | port BEFORE | port AFTER | fortran | after/fort | \|HO−LO\|/\|LO\| after |
|---|---|---|---|---|---|
| 12 | 1.963e+05 | **2.614e+02** | 2.614e+02 | **1.00** | 8.0e-04 |
| 24 | 7.464e+05 | **6.874e+02** | 6.874e+02 | **1.00** | 5.5e-04 |
| 36 | 9.731e+05 | **1.765e+02** | 1.765e+02 | **1.00** | 1.1e-04 |
| 42 | 7.479e+05 | **6.005e+01** | 6.005e+01 | **1.00** | 4.8e-05 |

**🔴 FIXED UNCONDITIONALLY and G0 RE-BASED (user decision, 2026-09-11: "our aim is to be faithful to
Fortran, so this bug should be fixed and it should be in the port. if it means we rerun the gates so
be it.").** No knob — a switch here would let someone run advection knowingly different from
upstream's, which is what this track exists to prevent.

The baseline move is **provably caused by this change alone**: the same source with the seeding
removed reproduced the OLD `ref0` byte-identically (np2, all 14 configs). Old oracles archived with
their provenance at `port2/m16/oracle_archive/pre-abfix/` (`gate0_ref0` 9.5 G, `gate0_core2_ref0`
92 G, `WHY_ARCHIVED.txt`). New baseline regenerated and verified:

| gate | status |
|---|---|
| pi, np1, all 14 configs | oracle rewritten, **PASS** |
| pi, np2, all 14 configs | oracle rewritten, **PASS** |
| production binary vs the NEW baseline, np2, all 14 | **BYTE-IDENTICAL, PASS** |
| CORE2 np8, all 14 configs | oracle rewritten, **PASS** (27392164, 14 min) |
| CORE2 production binary vs the NEW baseline, np8, all 14 | **BYTE-IDENTICAL, PASS** (27392407) |

**The re-base is complete and closed.** Both presets, both rank counts, all fourteen configs, oracle
and verification. `docs/plans/20260902-m16-mixed-precision.md` decision **D9/G0** now reads against
the post-AB-fix baseline; the pre-fix oracles remain at `oracle_archive/pre-abfix/` with their
`WHY_ARCHIVED.txt`, so the move is auditable in both directions.

**Its effect on the SP faithfulness metric is small** (1 month, no anomaly): salt 2.53 → **2.43**×
upstream, temp 1.39 → 1.37, a_ice 1.12 → **0.93**. So the oversized antidiffusive flux was **not**
the driver of the SP gap either — the limiter was evidently absorbing most of it, which is why the
solutions agreed despite a 1000× wrong intermediate. **That is exactly why it went unnoticed: FCT is
self-limiting, so a grossly wrong antidiffusive flux is clipped back to a plausible answer.**

**But it is a genuine correctness defect and should be fixed on its own merits**, independently of
precision: the port's first timestep applies a different high-order advection than upstream's, in
both precisions.

### 3b-year-g1. The 1-year matrix RE-RUN with both fixes (jobs 27393064–69) — the year is unchanged
Six port arms re-run with `bin/g1` (IC-bracket fix + AB seeding fix) into `faith/year_g1`; the six
Fortran arms are reused unchanged from `year_1958` (upstream always had both behaviours right).
1958, dt 1800, 17520 steps, 64 ranks, JRA55, no salt anomaly. ⚠️ The AB fix moved the port's **DP**
arm too, so the new port numbers are comparable to the **Fortran**, not to the old port numbers.

RATIO = port SP−DP / fortran SP−DP, each against its own DP arm:

| month | salt old → **new** | temp old → **new** | sst old → **new** | a_ice old → **new** |
|---|---|---|---|---|
| 1 | 2.77 → **2.43** | 1.43 → **1.37** | 1.01 → **1.11** | 0.98 → **0.93** |
| 3 | 3.04 → **3.07** | 2.90 → **2.89** | 2.13 → **2.08** | 1.09 → **1.10** |
| 6 | 3.03 → **3.01** | 2.94 → **2.87** | 4.42 → **4.32** | 2.85 → **2.99** |
| 9 | 2.44 → **2.44** | 2.19 → **2.18** | 2.55 → **2.51** | 2.69 → **2.84** |
| 12 | 1.62 → **1.56** | 1.65 → **1.62** | 1.68 → **1.66** | 2.61 → **2.80** |

🔴 **Beyond month 1 the two fixes change nothing** — every ratio moves by less than 5 %. Month 1
improves (salt 2.77 → 2.43) and that is the whole of the gain. The ratio's shape over the year is
also unchanged: it rises to ~3 by month 3, holds through month 9, and falls to ~1.6 by month 12 as
both codes' SP−DP saturate toward the same chaotic ceiling. **The ratio is not stationary and never
was** (§3b-year); a single-month number is not a summary of it.

🔴 **A new and sharper reading — normalise each code by ITS OWN noise envelope.** Salt, SP−DP
divided by that code's own FP64 σ=2e-4 K twin spread:

| month | fortran SP−DP / own envelope | **port SP−DP / own envelope** |
|---|---|---|
| 1 | 5.97e-06 / ≈2.0e-06 ≈ **3.0** | 1.45e-05 / ≈2.2e-06 ≈ **6.6** |
| 6 | 3.83e-05 / ≈7.6e-06 ≈ **5.0** | 1.15e-04 / ≈5.6e-06 ≈ **20.6** |
| 12 | 1.73e-04 / ≈1.4e-04 ≈ **1.2** | 2.71e-04 / ≈7.6e-05 ≈ **3.6** |

**The port's FP64 noise envelope is consistently SMALLER than upstream's** (month 12: 6.9–8.2e-05
against 1.12–1.66e-04) — the port's double-precision trajectory is *less* sensitive to a
rounding-sized nudge — **while its SP−DP is larger.** Those two facts together are hard to reconcile
with "the port amplifies perturbations more". They fit a **systematic drift** much better than
chaotic amplification: a drift adds to SP−DP without touching the FP64 twin spread. §3f already saw
the signature (the two codes' salt mean drifts have **opposite signs**, fortran +, port −); this is
the same statement made quantitative.

### 🔴🔴 3q. AFTER the AB fix: the per-step gap is GONE — what is left is GROWTH
The §3o flux comparison was re-run against the post-AB-fix binary (`bin/g1`), and the whole FCT
chain was instrumented on the Fortran side this time, not just the raw antidiffusive flux
(`oce_adv_tra_fct.F90` now also dumps `LO`, `FTTFMAX`, `FPLUS`, `AFLUXV`; `oce_adv_tra_ver.F90`
dumps `AFLUXVLO` = what `flux` holds on ENTRY to `qr4c`, i.e. the low-order vertical flux).
Jobs 27395691 (port, `fluxg1`) and 27395935 (Fortran, `fort_flux3`). Salinity, step 1, SP-vs-DP
relative error pooled over all nodes and levels:

| stage | what it is | relerr port | relerr fortran | **P/F** | mag P/F |
|---|---|---|---|---|---|
| `LO` | low-order solution | 8.08e-08 | 9.69e-07 | **0.083** | 1.000 |
| `AFLUXVLO` | LO vertical flux (`−w·T·area`) | 1.964e-02 | 1.716e-02 | **1.145** | 1.000 |
| `AFLUXVRAW` | antidiffusive flux `HO − LO` | 1.158e-02 | 9.992e-03 | **1.159** | 1.000 |
| `FTTFMAX` | Zalesak bounds | 2.38e-05 | 1.400e-04 | 0.170 | 1.000 |
| `FPLUS` | limiter factors | 5.161e-02 | 5.131e-02 | **1.006** | 1.000 |
| `AFLUXV` | flux AFTER limiting | 1.233e-02 | 1.088e-02 | **1.133** | 1.000 |

**Four readings, in order of importance.**

1. 🔴 **Every FP64 magnitude now matches 1.000 at every stage.** The 500–12000× of §3o is gone; the
   two codes compute the same intermediates in double precision, stage by stage. This is the
   strongest structural check the two codes have ever passed.
2. 🔴 **FCT is NOT the amplifier.** The port/Fortran ratio is already **1.145 at `AFLUXVLO`** — the
   low-order vertical flux, built *before* any FCT arithmetic — and it does not grow through
   `HO − LO` (1.159), the limiter factors (1.006) or the limited flux (1.133). Whatever excess the
   port carries into FCT, FCT passes through unchanged. **The §3o hypothesis that a binding limiter
   transmits the ≈35-psu bounds is refuted:** `FPLUS` is 1.006, dead level.
   `AFLUXVLO = −w·T·area` with `area` identical (§3p) and `T` at the float floor, so its relative
   error is essentially `w`'s — the lead, if this mattered, would be the vertical velocity. It does
   not appear to matter (see 4).
3. **The port's `LO` and bounds are 6–12× BETTER than upstream's** — because the port carries #1054
   and the oracle (`a62f180`) predates it. That is the first direct measurement of what #1054 buys:
   **12× less single-precision error in the low-order solution.** It is also why `LO` must not be
   read as a port defect here: the two codes are deliberately different at that line.
4. 🔴🔴 **And none of it survives to the tracer.** The 20-step bisection re-run with `bin/g1`
   (job 27395982, `bisect_20s_g1`, Fortran arms reused unchanged from `bisect_20s`):

| salt SP−DP | step 1 | step 5 | step 10 | step 20 | month 1 |
|---|---|---|---|---|---|
| fortran | 2.221e-06 | 2.322e-06 | 2.827e-06 | 2.801e-06 | 5.969e-06 |
| port, before the AB fix (§3i-c) | 3.541e-06 | — | — | 4.150e-06 | 1.510e-05 |
| **port, after the AB fix** | **2.055e-06** | 2.158e-06 | 3.095e-06 | **3.167e-06** | (rerunning) |
| **ratio port/fortran** | **0.925** | 0.929 | 1.095 | **1.131** | 2.43 |

**The step-1 ratio went 1.59 → 0.925: the port is now marginally BETTER than upstream after one
timestep.** §3i-c had factorised the month-1 2.53× as ≈1.59 (made in the first step) × ≈1.6
(accumulated over the month). **The first factor is gone.** A 1.13–1.16× flux-level difference
exists but does not reach the tracer, because the limiter absorbs it.

🔴 **So the diagnosis has changed shape.** The port's single-precision penalty is no longer a
per-step arithmetic difference — at 20 steps the two codes are within 13 % of each other. It is
**growth**: something accumulates over ~1500 steps that does not accumulate upstream (or accumulates
with the opposite sign — §3f's mean drifts are opposite in sign, fortran **+**, port **−**).
That is a conservation/drift question, not a rounding question, and it needs a different instrument:
the shape of the curve between step 20 and month 1. `growth_1m` (jobs 27396106 Fortran / 27396107
port, daily salt/temp/sst, 1488 steps dt 1800, both precisions) measures exactly that.

⚠️ **What this does NOT yet say.** The month-1 number in the table above is the pre-rerun 2.43; the
post-fix 1-month and 1-year arms are in flight. And a ratio of 0.93 at step 1 is one realisation of
one metric — it is a strong indication the arithmetic gap closed, not proof that nothing else is
wrong per step.

### 3r. A REAL conformance defect in the SSH solver — and it is NOT the gap
🔴 **Upstream keeps the entire CG *arithmetic* in FP64 even in a single-precision build.**
`solver.F90` @ a62f180 (the oracle) :150 declares `real(kind=WP_full) :: sprod(2), s_old, s_aux,
al, be, rtol`, and :194-196 / :207-208 / :240-241 / :304-313 cast **every sparse-row term** of the
residual, the preconditioner apply and the SpMV to `WP_full` before multiplying, rounding once into
the WP vector. Its own comment says why:

> *"they set the search direction and the stopping test — so rounding here steers the iteration
> itself rather than just reporting on it. The vectors (rr/zz/pp/App) and the matrix stay WP: the
> bandwidth is theirs, the accuracy is these."*

**The port ran that whole chain in `real_t`.** `docs/PRECISION_ISLANDS.md` carried it as **class 3**
("July stricter — upstream runs it in WP") and Phase B3 duly flipped it to float. That reading is
**false for the oracle**: it is class **2**, upstream stricter. Fixed unconditionally (no knob —
same principle as §3p): `cg_spmv`, `cg_dot`, the two fused reduce kernels, the axpy/`pp` recurrences
and every CG scalar now accumulate in `dbl_t` and store `real_t`; the reduces pair with `MPI_DOUBLE`
(registry rule 3, lesson SP1). Liveness is announced on rank 0 — `[fesom_ssh] CG arithmetic in
dbl_t (8-byte accumulators, 4-byte vectors)` — because a silent no-op here would look exactly like
a null result (the #1054 dead-code lesson).

**Gate 0: PASS np1 AND np2, all 14 configs, BYTE-IDENTICAL.** No re-base. In FP64 `dbl_t == real_t`,
so every cast is the identity — and the FMA-contraction worry was checked, not assumed.

⚠️ **And it does NOT move the SP gap.** Binary `g2` (= `g1` + this fix), 1 month, daily salt:

| day | 1 | 5 | 9 | 15 | 21 | 31 |
|---|---|---|---|---|---|---|
| ratio, g1 (no CG fix) | 1.053 | 1.349 | 1.646 | 1.788 | 1.699 | **1.948** |
| ratio, g2 (CG in dbl_t) | 1.054 | 1.393 | 1.698 | 1.792 | 1.734 | **1.937** |

**A null.** It is kept because it is what upstream does and the port exists to be faithful — not
because it bought anything.

### 🔴 3s. The faithfulness matrix was comparing two DIFFERENT viscosity schemes
Upstream's `config/namelist.dyn` ships **`opt_visc = 5`** (easy backscatter) and `setups/test_core2`
does not override it, so every **Fortran** arm of this matrix has run scheme 5 — its
`--check opt_visc-->` line in `run.log` is the proof. **The port's default is 7** (biharmonic), and
`jobs/job_m16_faith_port` never set `FESOM_VISC_OPT`. So from the first pilot to the 1-year matrix,
**the two arms ran different viscous operators** — a physics mismatch, not a precision one.

Fixed: the job now pins `FESOM_VISC_OPT=${VISCOPT:-5}`.

⚠️ **This too is a null for the gap.** `g1` re-run at 1 month with the viscosity matched:

| day | 1 | 5 | 9 | 15 | 21 | 31 |
|---|---|---|---|---|---|---|
| ratio, opt_visc 7 (mismatched) | 1.053 | 1.349 | 1.646 | 1.788 | 1.699 | **1.948** |
| ratio, opt_visc 5 (matched) | 1.094 | 1.384 | 1.658 | 1.821 | 1.804 | **1.940** |

The mismatch was real and had to be fixed — every number in §3b/§3b-year was measured under it — but
it is not what makes the port worse at SP.

### 🔴🔴 3t. The gap, finally characterised: it is ENTIRELY magnitude-dependent, and it is NOT chaos
Two 1-month daily curves, both codes, salt SP−DP against each code's own DP arm (`growth_1m`,
`growth_1m_anom`; jobs 27396106/07 and 27396546/47):

| day | 1 | 3 | 5 | 9 | 15 | 21 | 27 | 31 |
|---|---|---|---|---|---|---|---|---|
| **ratio, absolute S** | 1.05 | 1.23 | 1.35 | 1.65 | 1.79 | 1.70 | 1.70 | **1.95** |
| **ratio, #986 anomaly ON in BOTH** | 0.90 | 0.91 | 1.01 | 1.10 | 1.24 | 1.07 | 1.00 | **0.83** |

🔴 **With `S − 35` the gap never appears at all** — the ratio is flat about 1.0 for the whole month,
with no trend. Without it, it climbs from 1.05 to ~1.9 by day 9 and stays there. And the global mean
SP−DP shifts, day 31: absolute S gives fortran **+5.15e-06** against port **−3.50e-06** (opposite
signs, §3f); with the anomaly both become **+8.96e-07 / +8.41e-07** — same sign, same size.

**Read what that rules out.** Not chaotic amplification: the port's own FP64 noise envelope is
*smaller* than upstream's (§3b-year-g1), and a chaos story cannot be switched off by re-centring the
tracer. Not the flux chain: §3q measured it at 1.13–1.16× and showed nothing downstream amplifies it.
It is an operation whose rounding scales with **|S| ≈ 35** (float ulp 2.4e-06), present in the port
and not in upstream.

**Upstream gains nothing from the anomaly** (1.540e-05 → 1.648e-05, i.e. ×1.07 — very slightly
worse); **the port gains a factor 2.2** (2.983e-05 → 1.374e-05, ×0.46). So the port carries a
magnitude-sensitive term that upstream simply does not have.

**Stage sweep, anomaly ON/OFF, both codes** (step 1, absolute psu, `fluxg2`/`fluxg2_anom` vs
`fort_flux3`/`fort_flux4`) — the stage whose error collapses when S becomes O(1) is the culprit:

| stage | port OFF | port ON | ON/OFF | fort OFF | fort ON | ON/OFF |
|---|---|---|---|---|---|---|
| `LO` | 2.79e-06 | 2.08e-06 | 0.747 | 3.35e-05 | 3.34e-05 | 0.998 |
| `AFLUXVLO` | 4.41e+04 | 4.62e+02 | 0.010 | 3.85e+04 | 4.03e+02 | 0.010 |
| `AFLUXVRAW` | 2.13e+01 | 2.13e+01 | **1.000** | 1.84e+01 | 1.84e+01 | **1.000** |
| `FPLUS` | 5.15e-02 | 2.63e-02 | 0.511 | 5.12e-02 | 2.64e-02 | 0.515 |
| `DELTTF9` (pre-ALE) | 6.26e-05 | 4.92e-05 | 0.786 | — | — | — |
| **`DELTTF`** (post-ALE) | **9.74e-05** | **4.93e-05** | **0.506** | — | — | — |

`AFLUXVRAW` is magnitude-**in**sensitive in both (1.000) — so the `HO − LO` cancellation of §3m is
*not* the magnitude term. The port's jump is at **`DELTTF9 → DELTTF`**, the ALE reconstruction:
without the anomaly it inflates the increment error by **1.56×**; with it, by **1.00×**.

🔴 **The structural candidate (source-level, not yet measured).** Upstream accumulates advection
**and** the explicit Redi/GM diffusion into `del_ttf` and performs **one** ALE reconstruction
(`oce_ale_tracer.F90`: `diff_part_hor_redi` writes `temporary_ttf => tracers%work%del_ttf`, then
:775 `del_ttf += values*(hnode−hnode_new)`, `values += del_ttf/hnode_new`). **The port reconstructs
inside FCT on advection-only `del_ttf`, then Redi scatters straight onto `values`**
(`fesom_gm.cpp` `fesom_diff_part_hor_redi_kk`: *"edge→node SCATTER into `values` (atomic_add)"*).
Each extra write to `values` discards everything below ulp(35) = 2.4e-06 psu — invisible in FP64,
once per tracer per step at SP. The setup runs `Fer_GM = .true.` and `Redi = .true.`, so it is live.

### 🔴🔴🔴 3u. CONFIRMED — the port's Redi/GM diffusion writes onto ABSOLUTE salinity
**The test: the same 1-month pair with GM/Redi OFF in both codes** (`Fer_GM=.false.`,
`Redi=.false.` on the Fortran side; `FESOM_NO_GMREDI=1` on the port's — jobs 27397494/27397495).

| day | 1 | 5 | 9 | 13 | 19 | 25 | 31 |
|---|---|---|---|---|---|---|---|
| ratio, **GM/Redi ON** | 1.05 | 1.35 | 1.65 | 1.69 | 1.82 | 1.62 | **1.95** |
| ratio, **GM/Redi OFF** | 0.76 | 0.98 | 0.96 | 0.69 | 1.03 | 1.02 | **1.06** |

🔴 **The gap is gone.** Flat about 1.0 for the whole month, no trend. The two codes' mean SP−DP
drifts also stop disagreeing: with Redi on they are opposite in sign (fortran **+5.15e-06**, port
**−3.50e-06**); with Redi off both are negative and track each other (−3.66e-05 / −2.32e-05).

⚠️ **Knob liveness checked before believing it** (L80): the GM-off arms differ from the GM-on arms
at FP64 by relL2 **1.11e-03** (port) and **9.72e-04** (Fortran), so both switches genuinely fired,
and by comparable amounts. Note also that turning Redi off makes the SP−DP error *larger* in both
codes (3.9e-05 vs 1.5e-05) — the finding is that the **asymmetry** vanishes, not the error.

**The defect, at source level.** Upstream's `diff_part_hor_redi` (`oce_ale_tracer.F90`:1744-1748)
accumulates its increment into `del_ttf`:
```fortran
del_ttf(ul12:nl12,enodes(1)) = del_ttf(ul12:nl12,enodes(1)) + rhs1(ul12:nl12)*dt/areasvol(...)
```
and the single ALE reconstruction later divides by `hnode_new` (:775-778). **The port scatters the
same quantity straight onto `values`** (`fesom_gm.cpp`:2269/:2274):
```c
Kokkos::atomic_add(&vals((size_t)e1 * nl + nz), rhs1[nz] * dt / (av1 * hn1));
```
Algebraically identical — `del_ttf/hnode_new` is exactly `rhs*dt/(areasvol*hnode_new)`. **Numerically
it is not.** Upstream rounds an *increment* of order 1e-05 psu (ulp ≈ 1e-12); the port rounds the
*sum* `35 + 1e-05` (ulp = **2.4e-06**). Everything below 1.2e-06 psu of the Redi tendency is
discarded, **once per tracer per step**, and it is invisible at FP64 where ulp(35) = 7e-15.

**Why it survived every gate.** It is not a bug in FP64 — the port's byte gates compare the port
against *itself*, and the port's FP64 trajectory is unaffected at the 1e-15 level. It only shows
when `real_t` is float, and only as a slow divergence, and only against the Fortran.

**The fix (specified, NOT yet applied).** Restore upstream's order:
1. `fesom_diff_part_hor_redi_kk` and `fesom_diff_ver_part_redi_expl_kk` (`fesom_gm.cpp`) scatter into
   **`tracers->del_ttf`** and drop the `/hn` — 2 lines each, plus their host twins.
2. Split the ALE reconstruction (step 10, `fct_ale_recon`) out of
   `fesom_tracer_advect_one_fct_kk` into its own entry point.
3. Call it in `fesom_step.cpp` **after** the `gm` block for each tracer, so one reconstruction folds
   advection **and** Redi, exactly as `oce_ale_tracer.F90` does.
4. Update the `FESOM_KK_VERIFY=tradv` / `=gm` capture-before twins, which currently bracket a
   different set of writes.

🔴 **This changes FP64 results, so Gate 0 re-bases again** (both presets, both rank counts, the
production-binary verification) — the same procedure as §3p, and for the same stated reason: the port
exists to be faithful to upstream.

### ✅🔴 3v. FIXED — the Redi/GM restructuring closes the gap (1.95 → 1.02)
The §3u mechanism, repaired the way upstream does it. Binary tag **`g3`**.

**What changed** (4 files, no knob — same principle as §3p/§3r):
1. `fesom_gm.cpp` — `fesom_diff_part_hor_redi_kk`, `fesom_diff_ver_part_redi_expl_kk` (both its
   array and its `REDISWEEP` variant) and the two host twins now accumulate
   `rhs*dt/areasvol` into **`tracers->del_ttf`** instead of `rhs*dt/(areasvol*hnode_new)` onto
   `values`. The `modify_device()` markers follow.
2. `fesom_tracer_adv.cpp` — the ALE reconstruction (step 10, `fct_ale_recon`) is **split out** of
   `fesom_tracer_advect_one_fct_kk` into **`fesom_tracer_ale_recon_kk()`**, with a host twin
   `fesom_tracer_ale_recon()`. The FCT now ends at `del_ttf`.
3. `fesom_step.cpp` — the reconstruction is called per tracer **after** the `gm` block, so one
   reconstruction folds advection **and** the explicit Redi terms, exactly as
   `oce_ale_tracer.F90` does (`diff_tracers_ale` → :775-778).
4. The two capture-before verify twins follow the data: `FESOM_KK_VERIFY=tradv` and `=gm` now
   snapshot and diff **`del_ttf`**, not `values`.

**Internal consistency first.** Both verify gates are **bit-identical on Serial** after the
restructure — `max|Δ| = 0.000e+00` at every step for `fct(tr0)`, `fct(tr1)`, `redi(tr0)`, `redi(tr1)`
— so the Kokkos kernels and the C twins still agree exactly.

🔴 **The result.** 1 month, daily salt, SP−DP against each code's own DP arm, port/fortran ratio:

| day | 1 | 5 | 9 | 13 | 17 | 21 | 25 | 31 |
|---|---|---|---|---|---|---|---|---|
| `g1` (before) | 1.05 | 1.35 | 1.65 | 1.69 | 1.79 | 1.70 | 1.61 | **1.95** |
| **`g3` (after)** | 0.93 | 0.86 | 1.05 | 1.13 | 1.06 | 1.07 | 0.92 | **1.02** |

**Flat about 1.0 for the whole month, with no trend** — the same shape the GM-off twin (§3u) and the
#986 anomaly (§3t) produced, now at **absolute salinity with GM/Redi on**, i.e. in the configuration
the model actually runs. Temperature follows: day-31 ratio 1.465 → **1.120**.

**The mean drift changes sign with it.** Day 31, global mean SP−DP: fortran **+5.15e-06**, port
**+2.55e-06** — same sign, same order. Before the fix the port drifted **−3.50e-06**, opposite to
upstream. §3f's oldest unexplained observation is now explained and gone.

20-step bisection (`bisect_20s_g3`): salt ratio 0.925 (step 1) · 0.973 (10) · 1.099 (20).

#### Gate 0 re-based (second time; the first was §3p)
| step | result |
|---|---|
| old oracles archived | `oracle_archive/pre-redifix/` — `gate0_ref0` 9.5 G + `gate0_core2_ref0` 93 G + `WHY_ARCHIVED.txt` |
| 🔴 **provenance proof** | the same source with **§3v reverted** reproduces the **OLD** `ref0` **byte-identically**, all 14 configs, np1 — so the baseline move is caused by this change alone |
| pi, np1, 14 configs | oracle rewritten, **PASS** |
| pi, np2, 14 configs | oracle rewritten, **PASS** |
| production binary vs the NEW baseline, np1 + np2 | **BYTE-IDENTICAL, PASS** |
| CORE2, np8, 14 configs | oracle rewritten, **PASS** (27400103) |
| CORE2 production binary vs the NEW baseline, np8 | **BYTE-IDENTICAL, PASS** (27400495) |

**The re-base is complete and closed** — both presets, both rank counts, all fourteen configs,
oracle and production verification, with the reverted-source provenance proof on top.
`docs/plans/20260902-m16-mixed-precision.md` decision **D9/G0** now reads against the post-§3v
baseline.

**Everything else re-checked on the new binaries:** `ctest` **5/5 PASS in both precisions**;
Gate 3 (knob liveness at SP, `M16_MODE=live`) **PASS at np1 AND np2**, all 14 configs.
⚠️ One harness fix was needed to get there: `cgpipe` cannot arm at `npes==1` by construction
(`fesom_ssh.cpp`: *"requested but INACTIVE (npes==1 …)"*), so the live gate scored it DEAD and
failed the whole run at np1. **Verified pre-existing** — the pre-§3v `g2` binary fails it
identically — so it is a gate artefact, not a regression. `scripts/m16_gate0.sh` now skips it at
np1 exactly as it skips `evpwlean`; np≥2 still checks it (and it is LIVE there).

#### The 1-year matrix, re-run on `g3` (jobs 27400326–332, `faith/year_g3`)
Six port arms with `g3` and `FESOM_VISC_OPT=5`; the six Fortran arms reused unchanged from
`year_1958`. RATIO = port SP−DP / fortran SP−DP, each against its own DP arm:

| month | salt old → **new** | temp old → **new** | sst old → **new** | a_ice old → **new** |
|---|---|---|---|---|
| 1 | 2.43 → **0.98** | 1.37 → **1.03** | 1.11 → **0.88** | 0.93 → **1.03** |
| 3 | 3.07 → **1.02** | 2.89 → **1.01** | 2.08 → **1.08** | 1.10 → **0.83** |
| 6 | 3.01 → **1.08** | 2.87 → **0.81** | 4.32 → **1.28** | 2.99 → **1.76** |
| 9 | 2.44 → **0.98** | 2.18 → **0.70** | 2.51 → **0.80** | 2.84 → **1.93** |
| 12 | 1.56 → **0.85** | 1.62 → **0.70** | 1.66 → **0.74** | 2.80 → **1.92** |

🔴 **Gate G4's bar is met for the ocean.** salt, temp and sst sit between **0.70 and 1.28 for the
whole year** — the port loses no more to single precision than upstream does, and past mid-year it
loses rather less. The ratio is also no longer a rising curve: the shape that ran 2.4 → 3.0 → 1.6
across the year is gone. Every earlier statement in §3b/§3b-year/§3b-year-g1 was measured under the
§3u defect **and** the §3s viscosity mismatch and should be read as history, not as results.

⚠️🔴 **The remaining outlier is SEA ICE.** `a_ice` is at parity to month 3 and then settles at
**1.76–1.93 from month 6 on**. That is a different code path (mEVP + ice thermodynamics), it was
never touched by §3v, and it is now the only variable above the bar. It is the next question, and
it is a *new* one — the ocean-tracer story that occupied §3f–§3v is closed.

**Against each code's own FP64 noise envelope**, salt: month 6 fortran 3.83e-05 vs an envelope of
6.6–9.0e-06 (≈4.9×), port 4.13e-05 vs 4.2–5.8e-06 (≈8.2×); month 12 fortran 1.73e-04 vs
1.12–1.66e-04 (≈1.2×), port 1.47e-04 vs 3.8–7.8e-05 (≈2.5×). So SP−DP still exceeds the seed spread
at one year in **both** codes — "SP is buried in the noise" remains a multi-year claim (§3b-year),
unchanged by this fix. What changed is that the two codes now sit the same distance from double
precision, which is what G4 actually asks.

#### Bonus: FP64 faithfulness improved too — but from §3s, not §3v
`port-DP` vs `Fortran-DP`, salt relL2 over month 1 (all four arms against the same Fortran DP run):

| arm | day 1 | day 15 | day 31 |
|---|---|---|---|
| `g1`, opt_visc 7 (the historical matrix) | 1.208e-05 | 2.785e-04 | 3.759e-04 |
| `g1`, opt_visc **5** | 9.354e-06 | 1.657e-04 | **2.005e-04** |
| `g2` (+ CG `dbl_t`), opt_visc 7 | 1.208e-05 | 2.785e-04 | 3.759e-04 |
| **`g3`** (+ CG + §3v), opt_visc 5 | 9.353e-06 | 1.657e-04 | **2.006e-04** |

**The double-precision port and the Fortran now agree ~1.9× better — and every bit of that comes
from §3s, the viscosity-scheme pin.** §3v and the CG promotion move FP64 only at rounding level, as
designed. Worth stating plainly because it would be easy to credit the wrong change: **§3s bought the
FP64 agreement, §3v bought the single-precision parity.**

#### What is still divergent, and deliberately so
- **Order within the diffusion block.** Upstream runs `diff_part_hor_redi` *then*
  `diff_ver_part_redi_expl`; the port runs vertical first. Both now `+=` into `del_ttf`, so the
  difference is a rounding order on an *increment* (~1e-12 relative), not on absolute S. Left alone:
  changing it would be a second, unmeasured move of the baseline.
- **`FESOM_FCT_ALE_DBL`** stays as the instrument that named the magnitude dependence (§3l). It is a
  no-op now that the Redi terms no longer touch `values`.

## 3w. SEA ICE — the remaining outlier (2026-09-11)

### 4a. Characterisation: it is the SOUTHERN hemisphere, and it is not a comparator artefact
With the ocean at parity (§3v), `a_ice` is the only variable left above the bar. The monthly curve,
`year_g3`, RATIO = port SP−DP / fortran SP−DP:

| month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a_ice` | 1.03 | 1.21 | 0.83 | 1.39 | 1.46 | **1.76** | **1.88** | **1.98** | **1.93** | 1.32 | **1.80** | **1.92** |
| `m_ice` | 0.97 | 1.08 | 0.96 | 1.09 | 1.02 | 1.13 | 1.21 | 1.32 | 1.27 | 1.28 | 1.39 | 1.38 |

⚠️ **The comparator was checked first** — [[feedback-ice-mask-averaging]] records two separate
sea-ice comparator bugs that each masqueraded as physics. `m16_faith_compare.py` builds **one shared
validity mask across all four arms**, so the Fortran's NaN-over-open-water points are excluded from
the port's sum too, and points that are exactly zero in every arm are dropped. `a_ice`/`m_ice` are
scalars, so the rotated-vs-geographic vector-frame trap does not apply. The number stands.

🔴 **Where it lives** (`scratchpad/icewhere.py`, same shared mask):

| month 9 | npts | fortran | port | ratio |
|---|---|---|---|---|
| ALL | 29462 | 2.378e-03 | 4.595e-03 | **1.93** |
| **NH** | 15736 | 2.107e-03 | 2.656e-03 | **1.26** |
| **SH** | 13726 | 2.559e-03 | 5.602e-03 | **2.19** |
| NH pack a>0.8 | 8322 | 4.286e-04 | 5.602e-04 | 1.31 |
| SH pack a>0.8 | 12276 | 5.991e-04 | 1.278e-03 | 2.13 |

and at month 12 the split is sharper still: **NH 0.87** (the port is *better*), **SH 2.20**. It is not
an ice-edge effect — within the SH the pack interior, the 0.15–0.8 band and the marginal zone are all
around 2×. **It is Antarctic sea ice**, which is where the ice is free-drifting and dynamics-dominated
rather than landlocked.

### 🔴 4b. A configuration mismatch in the ice advection — `ice_diff`, the §3s mistake one milestone later
Auditing every `&ice_dyn`/`&ice_therm` parameter against the port's defaults found **one** difference,
and it is in the advection operator:

| | Fortran (upstream `config/namelist.ice`, not overridden by `setups/test_core2`) | port default |
|---|---|---|
| `ice_diff` | **0.0** | **10.0** |

Every other parameter matches exactly — `whichEVP` 0, `Pstar` 30000, `ellipse` 2.0, `c_pressure` 20,
`delta_min` 1e-11, `evp_rheol_steps` 120, `ice_gamma_fct` 0.5, `theta_io` 0, `Cd_oce_ice` 5.5e-3,
`ice_ave_steps` 1, `Sice` 4.0, `iclasses` 7, `hmin`/`armin` 0.01, `h0`=`h0_s`=0.5, all six albedos,
`con`/`consn`, `snowdist`, `c_melt`, `open_water_albedo` 0; `new_iclasses`/`use_meltponds` are
`.false.` in both and `h_snowscale`=0 is inert (its only use is the coupled thermodynamics).
The hemisphere branches were checked too: both codes select `h0` vs `h0_s` on
`geo_coord_nod2D(2,·)` — **geographic** latitude on both sides, so the rotated-frame trap does not fire
(and with `h0 == h0_s` it could not have mattered numerically anyway).

`ice_diff` is live: it enters the ice-FCT low-order rhs as
`diff = ice_diff*sqrt(elem_area/scale_area)` (`ice_fct.F90`:175, `fesom_ice_fct.cpp`:184/:610). So the
port has been running ~10–70 m²/s of artificial diffusion on `a_ice`/`m_ice`/`m_snow` where upstream
runs **none** — a different advection operator, in exactly the fields that show the gap, with exactly
the free-drift/sharp-gradient signature that would make it hemispherically asymmetric.

**Handled like §3s**, not like §3v: the port's **default stays 10.0**, because that is the value this
project's own CORE2 reference namelists carried and every existing baseline and `REFERENCE_RUNS`
floor was measured with it. A new knob **`FESOM_ICE_DIFF`** pins it (announced on rank 0), and
`jobs/job_m16_faith_port` now sets `FESOM_ICE_DIFF=${ICEDIFF:-0}` to match the Fortran.
**Gate 0: PASS byte-identical at np1** — knob-off is byte-neutral, so nothing re-bases.

⚠️ **MEASURED: another null** (jobs 27404226/27404228, `faith/year_g4`, 1-year `pdp`/`psp` pair with
`ice_diff = 0`; Fortran arms unchanged and reused). `a_ice` RATIO:

| month | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| `ice_diff` 10 (mismatched) | 1.03 | 1.21 | 0.83 | 1.39 | **1.76** |
| `ice_diff` 0 (**matched**) | 1.15 | 1.27 | 0.81 | 1.32 | **1.80** |

The mismatch was real and is now pinned — it has to be, the two codes must run the same advection
operator — **but it is not the ice gap.** That is three configuration/precision findings in a row
(§3r CG, §3s viscosity, §4b ice_diff) that were genuine defects and measured as nulls, against one
(§3v) that was the mechanism. The pattern is worth naming: *finding a difference is easy; finding
the one that carries the signal takes the measurement every time.*

### 4c. Splitting the ice gap: dynamics or the advected scalars?
The next discriminator, since the remaining candidates divide cleanly:

* if the **ice velocity** `uice`/`vice` carries the same ~2× SP−DP ratio, the gap is in the **EVP
  dynamics** (120 subcycles per step — a long float recurrence);
* if `uice`/`vice` is at parity while `a_ice` is at 2×, the gap is downstream, in the **FCT advection
  of the ice scalars or the thermodynamics**.

The port already writes `uice`/`vice`/`m_snow` monthly by default; the Fortran's `io_list` did not, so
`faith/year_ice/{fdp,fsp}` adds them (jobs 27404993/27404994, 1 year). The port side is `year_g4`,
already running with the matched `ice_diff`.

**RESULT — the ice VELOCITY carries the gap** (`year_ice` Fortran arms, jobs 27404993/27404994;
`year_g4` port arms; `scratchpad/icesplit.py`):

| var | M6 | M9 | M12 |
|---|---|---|---|
| `a_ice` | 1.80 | 1.90 | **1.96** |
| **`uice`** | **1.57** | **1.75** | **1.75** |
| **`vice`** | **1.63** | **1.70** | **1.97** |
| `m_ice` | 1.26 | 1.33 | 1.41 |
| `m_snow` | 1.20 | 1.35 | 1.20 |

So it is **not** downstream of the dynamics — `uice`/`vice` are already at 1.6–2.0. Concentration
tracks the velocity (a_ice has the sharpest gradients and a hard [0,1] bound, so it is the most
sensitive advected field); the masses lag at 1.2–1.4, partly reset each step by thermodynamics.

🔴 **And it is NOT amplification.** `a_ice` SP−DP against each code's **own** FP64 noise envelope:

| month | fortran envelope | port envelope | port/fortran envelope | SP−DP ratio |
|---|---|---|---|---|
| 6 | 4.48–5.18e-04 | 4.99–5.99e-04 | ~1.1 | **1.76** |
| 9 | 1.11–1.71e-03 | 1.14–1.33e-03 | ~0.9 | **1.93** |
| 12 | 1.00–1.28e-03 | 0.88–1.05e-03 | ~0.85 | **1.92** |

**The two codes' sea ice is equally sensitive to a rounding-sized nudge** — the port marginally
*less* so — while the port loses twice as much to single precision. That is the §3u signature
exactly: a **systematic per-step precision loss**, not a faster-growing trajectory.

**Source audit of the whole EVP: faithful.** `stress_tensor` (strain rates, `delta`, the Hunke
`det1`/`det2` stress update), `stress2rhs` (the `elem_area·(σ·∇)` scatter and the `inv_areamass`
normalisation), the per-subcycle implicit drag/Coriolis velocity solve
(`det = 1/(r_a²+r_b²)`, then multiply), `ice_strength = 0.5·pstar·m̄·exp(-c_pressure(1-ā))`,
`vale`/`dte`/`det1`, the subcycle order (stress → rhs → velocity → BC → halo), and the ice initial
condition are all one-to-one with `ice_EVP.F90`. Upstream's ice code has **no** explicit-double
sites, so the §3r class cannot apply. **One rounding-level divergence found:** the port computes
`zeta = p/max(δ,δ_min)` (one divide) where upstream computes `delta_inv = 1/max(δ,δ_min)` then
`zeta = p·delta_inv` (reciprocal then multiply). That makes the port *more* accurate, not less, and
one rounding cannot produce a factor 2 — recorded, not pursued.

### 4d. Does the gap scale with the EVP subcycle count?
The subcycle is a **120-deep float recurrence per ice step** (~525 000 per year), so a per-subcycle
loss is the natural shape for "systematic per-step, in the velocity". The test: run **both** codes
with `evp_rheol_steps = 240` and see whether the ratio moves. New knob `FESOM_EVP_STEPS` in the port
(default 120, announced on rank 0; **Gate 0 PASS byte-identical** with it unset), matched by
`&ice_dyn/evp_rheol_steps` in the Fortran namelist. 6 months, 4 arms
(`faith/evp240`, jobs 27406623–27406625) — month 6 is where the 120-subcycle ratio reads **1.80**.

⚠️ **Frame note.** `uice`/`vice` are VECTORS and the two codes write them in different frames — the
Fortran rotates to geographic at output, the port writes the rotated-grid components
([[feedback-ice-mask-averaging]], the trap that twice masqueraded as physics). **This measurement is
immune:** the r2g rotation is per-node and norm-preserving, so relL2(sp, dp) *within* one code is
exactly frame-invariant. Only the compare script's `port-vs-fortran at equal precision` line is
frame-contaminated for these two variables and must be ignored.

### 4d-result. The gap does NOT scale with the EVP subcycle count — another null
Both codes at `evp_rheol_steps = 240` against both at 120 (`faith/evp240`, jobs 27406623–25;
6 months, `ice_diff = 0` on both sides; both port knobs verified live in the run log):

| | M4 | M5 | M6 |
|---|---|---|---|
| `a_ice` ratio, 120 subcycles | 1.32 | 1.48 | **1.80** |
| `a_ice` ratio, 240 subcycles | 1.18 | 1.34 | **1.62** |

Doubling the recurrence depth made the ratio **fall** ~10 %, not rise. Look at the absolute numbers
and the reason is clear: at 240 the **Fortran's** SP−DP rose 12 % (1.327e-03 → 1.490e-03) while the
**port's barely moved** (2.392e-03 → 2.416e-03). **The port's ice SP error is insensitive to the EVP
configuration** — so whatever generates it is not the subcycle recurrence.

### 4e. Following the forcing out of the EVP
If the loss is not in the subcycle, it is in what the subcycle is *fed*. The SH signature is the
constraint: Antarctic ice is in **free drift** (thin, low concentration, weak internal stress), so its
velocity is set by **wind stress, ocean drag and the SSH gradient** — not by the rheology that
dominates the landlocked Arctic pack. That is exactly the hemispheric split observed (NH 0.87–1.26,
SH 2.19–2.20).

**`ssh` checked first — it is elevated but not enough.** The SSH gradient forces the ice momentum
(`rhs_a`/`rhs_m`, `ice_EVP.F90`:604-619, a faithful transcription in the port). `ssh` was never in the
faithfulness matrix's variable list, and both codes already write it:

| month | 1 | 6 | 9 | 12 |
|---|---|---|---|---|
| `ssh` ratio | 1.06 | **1.31** | **1.32** | 1.15 |

Real but ~1.3, against the ice velocity's 1.6–2.0. It cannot be the whole story.

⏳ **Next: ocean surface velocity** — the drag term `cd_oce_ice·|u_ice − u_w|·ρ₀·inv_mass` feeds on
`u_w`, and the Southern Ocean is where the surface currents are strongest (the ACC), which is the
one forcing whose geography matches the NH/SH split. **`u`/`v` have never been in the matrix either**
— the Fortran's `io_list` does not carry them. `faith/year_uv/{fdp,fsp}` adds `u`, `v` (plus
`uice`/`vice`) for a year (jobs 27408009/27408010); the port already writes them.

### 4e-result. The ocean velocity is the OPPOSITE of the prediction — the port is 2× better
`faith/year_uv/{fdp,fsp}` (jobs 27408009/27408010) added `u`,`v` to the Fortran's `io_list` for a
year; the port already writes them. Within-code SP−DP relL2, elements × levels
(`scratchpad/uvsplit.py`; frame-invariant within a code, see the note above):

| var | month | slice | fortran | port | **ratio** |
|---|---|---|---|---|---|
| u | M6 | surface | 6.175e-03 | 4.372e-03 | **0.71** |
| u | M9 | surface | 2.466e-02 | 9.306e-03 | **0.38** |
| u | M12 | surface | 7.673e-02 | 4.205e-02 | **0.55** |
| v | M6 | surface | 1.073e-02 | 8.183e-03 | **0.76** |
| v | M9 | surface | 4.390e-02 | 1.750e-02 | **0.40** |
| v | M12 | surface | 1.005e-01 | 6.420e-02 | **0.64** |

**The port's ocean velocity loses HALF what upstream's does.** So the ice velocity gap is not
inherited from the current the ice drags against — the drag input is *cleaner* in the port. Together
with `ssh` at 1.3, the whole forcing side is eliminated: the ice momentum error is generated inside
the ice. (Incidental but worth recording: ocean `u`/`v` were never in the faithfulness matrix, and
the port is comfortably inside the bar on them.)

### 4f. Where the ice dig stands — characterised, four hypotheses eliminated, NOT solved
**What is established.**
1. The gap is **Antarctic** (M12: NH 0.87, SH 2.20) and lives in the pack interior as much as the
   edge, so it is not a marginal-ice-zone artefact. The comparator was verified first.
2. It is a **systematic per-step precision loss, not amplification** — the two codes' `a_ice` FP64
   noise envelopes agree to 0.85–1.1× while the port's SP−DP is 1.9×.
3. It is carried by **`uice`/`vice` (1.6–2.0)** and **`a_ice` (1.8–2.0)**, much less by `m_ice`/
   `m_snow` (1.2–1.4).
4. Both codes' ice SP−DP exceeds their own noise envelope (2.6–4.8× at M6), so this is signal.

**What is eliminated, each by its own measurement — do not retry:**

| hypothesis | verdict |
|---|---|
| `ice_diff` mismatch (port 10.0 vs Fortran 0.0) — **a real defect, found and pinned** | **null** (M12 1.96 vs 1.92) |
| per-subcycle accumulation in the 120-deep EVP recurrence | **null** — 240 subcycles *lowered* the ratio (M6 1.80 → 1.62); the port's absolute error barely moved |
| SSH-gradient forcing of the ice momentum | ratio only **1.3** |
| ocean surface velocity (the ACC drag hypothesis) | **refuted, inverted** — the port is **0.38–0.76×**, i.e. 2× better |
| inherited from the ocean tracers | no — they are at parity (§3v) while `a_ice` is at 1.9 |

**Source audit: faithful throughout.** `stress_tensor`, `stress2rhs`, the implicit drag/Coriolis
velocity solve, `ice_strength`, `vale`/`dte`/`det1`/`Tevp_inv`, the subcycle order, the SSH-gradient
`rhs_a`/`rhs_m` assembly, the atmospheric ice stress (and `Cd_atm_ice` = 1.2e-3 on both sides, with
`AOMIP_drag_coeff=.false.`), the ice initial condition, the ice-FCT low-order solve, and the Hibler
concentration equation `A += c_melt·min(rh,0)·A/max(h,hmin) + max(rA,0)(1−A)/lid_clo` are all
one-to-one with the Fortran. Upstream's ice code has **no** explicit-double sites, so the §3r class
cannot apply. Every `&ice_dyn`/`&ice_therm` parameter matches except the `ice_diff` above.
**One rounding-level divergence recorded, not pursued:** the port's `zeta = p/max(δ,δ_min)` (one
divide) against upstream's `delta_inv = 1/max(δ,δ_min)` then `p·delta_inv` — which makes the port
*more* accurate, and one rounding cannot make a factor 2.

🔴 **The decisive measurement not yet done** is the one that cracked the ocean: instrument **both**
codes to dump the ice state at matched points of a single step — `a_ice`/`m_ice` at entry, after
thermodynamics, after FCT advection, and `u_ice`/`v_ice` plus `sigma`/`eps` at chosen subcycles —
and compare the SP−DP of each stage. §3q/§3u found the Redi defect that way after the same run of
plausible-but-null hypotheses. The port already has half of it: `FESOM_EVP_DUMP_DIR` writes
`a_ice`/`m_ice`/`m_snow`/`elevation` at entry, `u_ice`/`v_ice`, and `inv_mass`/`inv_areamass`/
`rhs_a`/`rhs_m`/`ice_strength`, keyed by global id. The Fortran side needs the same instrument in
`ice_EVP.F90` and `ice_thermo_oce.F90` — the pattern is the one already used in `oce_adv_tra_ver.F90`
and `oce_adv_tra_fct.F90`.

### 4g. THE INSTRUMENT: matched ice-stage and EVP-subcycle traces in both codes
Built 2026-09-12. `FESOM_ICE_TRACE=<dir>` writes, in the tracer-FCT binary layout (int32 gid,
int32 1, float64) so the existing readers apply:

| stage | where | fields |
|---|---|---|
| `A_entry` | after `ocean2ice`, before dynamics | a_ice m_ice m_snow uice vice |
| `B_postdyn` | after the EVP | uice vice |
| `C_postadv` | after ice-FCT + `cut_off` | a_ice m_ice m_snow |
| `D_postthermo` | after thermodynamics | a_ice m_ice m_snow |
| `evp.*.subNNN` | inside the first ice step, at `FESOM_ICE_TRACE_SUBS` (default 1,2,10,60,120) | uice vice σ11 σ12 σ22 ε11 ε12 ε22 |

Knobs `FESOM_ICE_TRACE_STEPS` (how many ice steps), `FESOM_ICE_TRACE_EVERY` (stride). Port:
`fesom_ice.cpp` (`ice_trace_stage`) + `fesom_ice_evp.cpp` (`evp_subcycle_trace`, inside the live
Kokkos subcycle loop, after the halo). Fortran twin (LOCAL, not for upstream): `ice_setup_step.F90`
(`m16_ice_trace`, `contains`-ed) + `ice_EVP.F90` (`m16_evp_trace`). Job knobs `ICETRACE=1`,
`ICETRACE_STEPS`, `ICETRACE_EVERY`, `ICETRACE_SUBS`; Fortran needs `ORACLE_VARIANT=_instr`.
Both codes wrote identical file inventories on the first run (3392 files each, 64 ranks).
Binary tag **`g6`**; byte-neutral when unset.

**Result 1 — ice step 1, every stage (`faith/icetrace`, jobs 27414200/27414201):**

| stage | relerr F | relerr P | **P/F** | \|DP\| P/F |
|---|---|---|---|---|
| `B_postdyn.uice` / `.vice` | 1.47e-04 / 1.96e-04 | 1.48e-04 / 1.96e-04 | **1.007 / 0.999** | 1.000 |
| `C_postadv.a_ice` / `m_ice` / `m_snow` | 9.75e-07 / 8.45e-07 / 7.08e-07 | 9.98e-07 / 8.67e-07 / 7.26e-07 | **1.024 / 1.025 / 1.026** | 1.000 |
| `D_postthermo.a_ice` / `m_ice` / `m_snow` | 9.72e-07 / 8.50e-07 / 7.13e-07 | 9.96e-07 / 8.72e-07 / 7.31e-07 | **1.024 / 1.025 / 1.026** | 1.000 |
| EVP sub 120: σ11 / σ12 / σ22 | 5.88e-03 / 8.79e-03 / 6.00e-03 | 5.57e-03 / 8.85e-03 / 5.63e-03 | 0.95 / 1.01 / 0.94 | 1.000 |
| EVP sub 120: ε11 / ε12 / ε22 | 1.36e-03 / 1.41e-03 / 1.55e-03 | 1.47e-03 / 1.42e-03 / 1.50e-03 | 1.08 / 1.00 / 0.97 | 1.000 |

Every FP64 magnitude matches to 1.000 (the DP-vs-DP columns are 1e-14 at sub 1, growing to
1e-6 by sub 120 — pure rounding-order drift). **Every stage of the ice step is at parity (0.92–1.08)
on step 1**, subcycle by subcycle. (Early subcycles show the port *better*, 0.27–0.57 at sub 1–10
on uice — the σ11/σ22 3.2× at sub 1 is the first-subcycle stress on a uniform IC, magnitudes 5e-07,
and is gone by sub 2.)

**Result 2 — the month-long daily trace (`faith/icetrace_1m`, jobs 27414257/27414258; 1488 ice
steps, one traced per day):** a_ice's within-step ratio at the three stages of the same step:

| step | 48 | 144 | 288 | 432 | 576 | 720 | 864 | 960 |
|---|---|---|---|---|---|---|---|---|
| `A_entry` | 1.33 | 1.00 | 1.14 | 0.96 | 0.87 | 1.40 | 1.27 | 0.89 |
| `C_postadv` | 1.11 | 0.69 | 0.61 | 1.47 | 0.90 | 1.39 | 1.26 | 0.95 |
| `D_postthermo` | 1.12 | 0.98 | 1.16 | 1.47 | 0.91 | 1.38 | 1.25 | 0.95 |

🔴 **The three stages of any one step move together.** Where `D ≠ A` on a given day, `C` explains
it (the FCT + `cut_off` stage, days 144/288/432), and the direction is *not* consistently upward —
it is 0.61 at step 288 and 1.47 at step 432. **Neither the ice advection nor the thermodynamics
systematically inflates the port's ratio within a step.** What the trace shows instead is that the
`A_entry` ratio itself drifts 0.9 → 1.3 over the month — the excess is accumulating **between** ice
steps, i.e. through the ocean the ice is coupled to.

**Which eliminates the last in-ice candidates.** With the EVP (§4d, and now traced), the advection
and the thermodynamics all at parity within the step, and the ocean→ice mapping (`ocean2ice`:
area-weighted surface `UV` → `u_w`/`v_w`, `hbar` → `elevation`, `T`/`S` → `srfoce_*`) verified
faithful line by line, the remaining directions are the **ice→ocean** half of the coupling
(`oce_fluxes`: the stress the ice puts on the ocean, the fresh-water and heat fluxes) and the
ocean's own SP behaviour *under* ice. Note that this is consistent with everything measured: the
Southern Ocean is where the ice-driven surface stress and buoyancy fluxes matter most for the
surface ocean, and the ocean `u`/`v` is at 0.4–0.8 *globally* — a hemispheric split of that field
has not been taken.

⚠️ **Instrument caveats recorded.** The Fortran `FESOM_ICE_TRACE_EVERY` stride did not fire on the
first month run (a `read` into the iostat variable; fixed) — it wrote every step, 48× more files, and
the analysis read the port's strided subset, so the result stands. Its step field was `I3.3` and
wrapped at step 1000 (now `I0.3`, matching the port's `%03d`); steps ≥ 1008 are absent from that run.

### 🔴🔴 4h. REFRAMED: it is not a sea-ice problem — it is a 60–70°S OCEAN problem the ice inherits
The §4g finding ("the excess accumulates *between* ice steps") sent the question back to the ocean,
and the ocean's *global* parity turned out to be hiding a **latitudinal structure**.
Month 12, port/fortran SP−DP ratio by latitude band, surface fields (`year_g4`/`year_ice`):

| band | sst | sss | ssh | | band | sst | sss | ssh |
|---|---|---|---|---|---|---|---|---|
| 90–70°S | 1.04 | 1.48 | 1.38 | | 20°S–0 | **0.43** | **0.45** | 0.59 |
| **70–60°S** | **2.63** | **2.62** | **2.15** | | 0–20°N | 0.54 | 0.65 | 0.51 |
| 60–50°S | 1.45 | 1.81 | 1.39 | | 40–50°N | 1.16 | 1.10 | 1.33 |
| 50–40°S | 1.11 | 1.16 | 1.02 | | 50–60°N | 1.10 | 1.04 | 1.23 |
| 40–20°S | 1.19 | 1.40 | 1.11 | | 70–90°N | 1.14 | 1.90 | 1.55 |

Month 6 has the same shape (70–60°S: 2.24 / 1.40 / 2.19). **The port is 2× better than upstream
in the tropics and 2.6× worse in one band, 60–70°S** — the global ratios of §3v (0.70–1.28) are the
*average* of those two. The Antarctic ice sits in exactly that band, and its SP−DP ratio there
(a_ice 2.15, month 12) matches the surface ocean's. **The ice is following the ocean, not driving
it.** The SH *open* ocean south of 40°S is as bad as the ice zone (sss 2.33 vs 2.25).

**And the noise-envelope test localises it the same way** (`year_g3`, month 12, `sst`):

| band | F spdp/env | P spdp/env | spdp P/F | **env P/F** |
|---|---|---|---|---|
| **70–60°S** | 4.9 | **13.2** | **2.61** | **0.98** |
| 60–50°S | 8.0 | 7.6 | 1.43 | 1.51 |
| 50–60°N | 17.8 | 18.0 | 1.10 | 1.09 |
| 20°S–20°N | 1.2 | 1.5 | 0.56 | 0.43 |

In 60–70°S the two codes' FP64 noise envelopes are **identical (0.98)** while the port's SP−DP is
2.6× — the port sits **13× above its own envelope** there, upstream 5×. Everywhere else the two
ratios track each other. **A per-step precision loss confined to one latitude band**, with the
northern analogue (50–60°N) clean.

**What is distinctive about 60–70°S in this setup.** It is the Antarctic coastal/shelf zone: the
coldest, freshest, densest surface water in the model (near-freezing SST, brine rejection, the
densest water masses forming), the strongest surface buoyancy fluxes from the ice, and — on the
CORE2 mesh — no cavities, so the shelf is the model's southern boundary. The physics that is
*specific* to that band and not to 50–60°N: (a) the freezing-point / near-freezing branch of the
EOS and of the ice–ocean heat flux, (b) brine-rejection salt fluxes into a very dense water column,
(c) the deepest convective mixing (KPP boundary-layer depth reaching the bottom on the shelf). Each
of those is a *different* code path from the one the northern band exercises. **This is the next
thing to instrument — per-band, per-stage, in the ocean, not the ice.**

Month 1 (daily, `growth_1m_g3`): the 60–70°S sst band is already the most consistently elevated
(1.0–1.4 while other bands swing 0.5–1.7), so the mechanism is present from the start and compounds.

### 4i. The ocean stage trace, per band — the excess is set OUTSIDE the tracer stages, at the surface
The tracer-step instrument (§3j's `FESOM_SALT_TRACE`) was generalised to both tracers, any step
and a stride, and given a Fortran twin (`oce_ale_tracer.F90`, `m16_salt_stage`, LOCAL): stages
`A_entry` (top of the per-tracer loop), `C2_post_recon` (after the §3v single ALE reconstruction —
advection + Redi folded), `D_post_vdiff` (after the implicit vertical diffusion). One month, one
traced step every two days (`faith/oceantrace_1m`, jobs 27414753/27414754). Salinity, **surface
level**, port/fortran SP−DP ratio:

| step | 70–60°S A / C2 / D | 50–60°N A / C2 / D | 20°S–20°N A / C2 / D |
|---|---|---|---|
| 96 | 1.94 / 1.95 / 2.01 | 0.75 / 0.72 / 0.66 | 0.96 / 0.95 / 1.14 |
| 192 | **3.26 / 3.28 / 3.19** | 1.12 / 1.15 / 1.06 | 1.69 / 1.67 / 1.06 |
| 288 | 2.18 / 2.20 / 2.17 | 0.85 / 0.91 / 0.87 | 1.05 / 1.05 / 1.07 |
| 864 | 1.14 / 1.13 / 1.12 | 2.20 / 2.22 / 2.20 | 1.27 / 1.27 / 1.39 |
| 1440 | 1.40 / 1.38 / 1.39 | 1.10 / 1.10 / 1.11 | 1.24 / 1.25 / 0.93 |

🔴 **In every band and every step, A → C2 → D move together.** The tracer advection, the Redi
terms, the reconstruction and the implicit vertical diffusion all leave the ratio where they found
it. (The one exception is the tropics' D column, which moves the ratio *down* — the vertical
diffusion is where the port is *better*.) Same verdict as the ice trace (§4g): the excess is
generated **between** the tracer stages — in the dynamics half of the step or in the surface
forcing — and the tracers merely carry it.

**And it is surface-intensified.** Month 12, port/fortran ratio by depth level in the 60–70°S band
(`year_g4`/`year_ice`):

| level | 0 | 2 | 4 | 6 | 9 | 12 | 15 | 18 | 22 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| temp | **2.63** | **2.70** | 2.47 | 2.42 | 2.11 | 1.76 | 1.54 | 1.34 | 1.33 | 1.67 | 1.62 |
| salt | **2.62** | **2.87** | 1.90 | 1.93 | 1.72 | 1.69 | 1.64 | 1.69 | 1.07 | 1.47 | 1.35 |

2.6–2.9 in the top ~50 m, decaying to ~1.5 below; 50–60°N is 0.9–1.3 at every level; the tropics
0.5–0.8 at every level. **A surface-flux signature, confined to the Antarctic band.**

**What that leaves.** Not the EOS/pressure-gradient chain (that would be interior-weighted, and the
dynamics feed the tracers only through advection, which is at parity). Not the ice's own internals
(§4g). What is specific to 60–70°S, at the surface, in the forcing: the **bulk formulae over ice
and near-freezing water** (`gen_forcing_couple.F90` / `fesom_bulk.cpp`: latent/sensible heat with
the ice-surface branch, the freezing-point limit), the **ice–ocean fluxes** into the ocean
(`oce_fluxes`: the heat and virtual-salt/fresh-water fluxes from ice growth and melt, brine
rejection), and the **surface stress under ice** (the ice–ocean drag applied to the ocean). Each
is a port-side code path with its own float arithmetic on quantities that are large and cancelling
(latent heat ≈ 2.5e6 J/kg × evaporation; freezing point vs SST differences of 1e-2 K on a 271 K
base). **The next instrument is on those flux fields, per band, matched between the codes** —
`heat_flux`, `water_flux`, `stress_surf`, the ice `flx_h`/`flx_fw` — and it is the first one that
needs the *forcing* dumped rather than the state.

### 4j. THE FLUX INSTRUMENT — the surface forcing is CLEARED
Built in both codes (2026-09-13). `FESOM_FLUX_TRACE=<dir>` dumps, at the point where the coupling
has finished and the ocean step is about to consume it (port `fesom_main.cpp` after
`fesom_ice_oce_fluxes_mom`; Fortran `fesom_module.F90` after `oce_fluxes_mom` + `oce_fluxes`,
twin `m16_flux_trace` in `ice_oce_coupling.F90`, LOCAL): `heat_flux`, `water_flux`,
`real_salt_flux`, `relax_salt`, `stress_x/y` (the blended `stress_node_surf`), `stress_iceoce_x/y`,
`stress_atmice_x/y`, `flx_h`, `flx_fw`, `thdgr`, `thdgrsn`, `a_ice`. One float64 per owned node,
ice-trace layout. Knobs `_STEP`, `_EVERY`; job knob `FLUXTRACE=1`. `faith/fluxtrace_1m`, jobs
27434784/27434785, one traced step per day for a month.

**The instantaneous view is chaos-dominated and cannot be read.** `heat_flux` differs between the
two codes' *double* runs by relL2 **0.63–0.93** at a daily instant — the forcing is a fast, noisy
field — so a per-day SP−DP ratio of it is meaningless. What the ocean integrates is the sum, and a
systematic per-step precision bias survives a sum where chaos averages down. **Time-accumulated
over the 31 traced steps**, port/fortran ratio of the accumulated SP−DP by band:

| flux | 70–60°S | 60–50°S | 20°S–20°N | 50–60°N | 70–90°N | **net bias 70–60°S, P/F** |
|---|---|---|---|---|---|---|
| `heat_flux` | **0.47** | 0.37 | 0.58 | 1.05 | 1.28 | **0.09** |
| `water_flux` | 1.21 | 0.94 | 1.02 | 1.00 | 1.26 | **0.00** |
| `real_salt_flux` | 1.20 | 0.93 | — | 1.00 | 1.28 | **0.05** |
| `relax_salt` | 1.19 | 1.09 | 0.98 | 1.09 | 1.01 | 0.56 |
| `stress_x` / `stress_y` | 0.78 / 0.81 | 1.08 / 0.97 | 0.89 / 0.96 | 1.06 / 0.92 | 1.05 / 1.13 | 0.89 / 0.41 |
| `flx_h` / `flx_fw` | 1.26 / 1.21 | 0.99 / 0.94 | 0.95 / 0.92 | 1.00 / 1.00 | 1.28 / 1.26 | 0.24 / 0.15 |

🔴 **In the 60–70°S band every flux the ocean receives is at 0.5–1.3, and the port's *net* SP−DP
bias is far smaller than upstream's** (heat 0.09×, fresh water ~0, salt 0.05×). The forcing is not
where the 2.6× is made. (`stress_iceoce` shows 24× / 81× at 60–50°S — the marginal ice zone, where
the DP magnitude is near zero and the ratio is of two rounding-sized numbers; it is not in the
60–70°S band and the blended `stress_x/y` that the ocean actually feels is 0.8–1.1 there.)

### 4k. Where the 60–70°S excess now stands — cornered, not caught
Everything measured, in order:

| suspect | instrument | verdict in 60–70°S |
|---|---|---|
| sea-ice internals (EVP, advection, thermo) | §4g ice-stage trace | at parity within the step |
| tracer advection + Redi + reconstruction | §4i ocean-stage trace | at parity within the step |
| implicit vertical diffusion | §4i | at parity (port *better*) |
| every surface flux the ocean receives | §4j flux trace, accumulated | at parity or better |
| noise-envelope (chaos) | §4h | **identical** envelopes, 2.6× SP−DP |

**What is left is the dynamics half of the ocean step**: the EOS / pressure gradient → momentum →
SSH solve → vertical velocity chain, evaluated in that band. Three things make 60–70°S different
for *that* chain and not for the tracers or the forcing: (a) the coldest, densest water in the model,
so the EOS is evaluated where `T − T_freeze` and the density anomaly are smallest relative to the
absolute values — the same magnitude-cancellation class as §3u, now in `ρ(T,S,p)`; (b) the steepest
bathymetry (the Antarctic shelf break) so the pressure-gradient error has the largest σ-coordinate
component; (c) the strongest barotropic signal (ACC) per unit surface area. The next stage
instrument is on **`density`, `hpressure`, `w`, `ssh` at matched points of the dynamics**, per band —
the port already writes `density`/`w`/`ssh` monthly and the Fortran can be asked to.

⚠️ The signature "surface-intensified" (§4i) does *not* point at surface forcing after all — the
forcing is cleared — so it must be read as **where the dynamics' error projects onto the tracers**
(the mixed layer, where `w` and the surface-intensified currents act). That is consistent with the
SSH ratio being 2.15 in the band while the fluxes are at 1.

**Confirmed from data already in hand** (`year_uv`/`year_g4`, month 12, ocean velocity on
elements, port/fortran SP−DP ratio by band and level):

| band | `u` lev0 / lev5 / lev10 / lev20 / column | `v` lev0 / lev5 / lev10 / lev20 / column |
|---|---|---|
| **70–60°S** | **1.82 / 1.81 / 1.81 / 1.79 / 1.84** | **1.98 / 2.03 / 2.03 / 1.96 / 2.02** |
| 60–50°S | 1.20 / 1.20 / 1.21 / 1.26 / 1.27 | 1.25 / 1.27 / 1.27 / 1.29 / 1.30 |
| 20°S–20°N | 0.54 / 0.49 / 0.41 / 0.54 / 0.52 | 0.63 / 0.59 / 0.52 / 0.47 / 0.59 |
| 50–60°N | 1.01 / 1.02 / 1.03 / 1.01 / 1.01 | 1.01 / 1.02 / 1.03 / 1.03 / 1.01 |

🔴 **The ocean velocity in 60–70°S is at 1.8–2.0 at EVERY depth, uniform through the column** —
a **barotropic** signature — while its northern analogue is 1.01 at every depth. The tracers'
surface intensification (§4i) is the mixed layer responding to a depth-uniform velocity error.
That narrows the dynamics chain to its barotropic part: the **SSH solve and the barotropic
pressure gradient** (`ssh` itself is 2.15 in the band), not the baroclinic EOS/hpressure chain
(which would be depth-structured). And the ice, which drags on that velocity and whose own ratio
is 2.15 there, is simply following it.

**Why 60–70°S and not 50–60°N for a barotropic error?** Two candidates, both testable:
(a) the **ACC** — the only band with a strong, deep-reaching, zonally unbounded barotropic flow,
so the SSH gradient balances a large transport and a small relative SSH error is a large velocity
error; (b) the **CG solver** in that band — the port's CG arithmetic was promoted to `dbl_t` in
§3r (measured as a null *globally*, but the global metric averages the tropics' gain against
this band), and the two codes' preconditioners differ in form. The cheap discriminator is the
§3r knob in reverse: run `g3` (CG in `dbl_t`) against `g1` (CG in `real_t`) **per band** — that
data exists (`growth_1m_g2` vs `growth_1m`) and costs nothing to read.

### 4l. The §3r-in-reverse test — and a clue about what the band's residual is
`ssh` in the 70–60°S band, month-1 mean, port/fortran SP−DP ratio, across the binary ladder:

| binary | CG arithmetic | Redi → del_ttf (§3v) | 70–60°S | 60–50°S | 20°S–20°N | 50–60°N |
|---|---|---|---|---|---|---|
| `g1` | real_t | no | **4.89** | 4.22 | 3.26 | 2.47 |
| `g2` | **dbl_t** | no | **4.90** | 4.23 | 3.26 | 2.47 |
| `g3` | dbl_t | **yes** | **1.54** | 1.12 | 0.93 | 0.84 |

**The CG promotion is a null in the band too** (4.89 → 4.90) — hypothesis (b) of §4k is dead.
**The §3v Redi fix cut the band's SSH excess from 4.9 to 1.5** — so most of the band's SSH error
was a *density* consequence of the Redi defect, not a solver one. What remains, 1.54 at month 1
growing to 2.15 by month 12, is the residual that the ice and the barotropic velocity follow.

**The residual has the shape of a second, smaller instance of the same class.** It is barotropic,
it is in the one band where density is set by the coldest, densest, most weakly stratified water
in the model, and it grows over the year rather than appearing at step 1. The §3u mechanism was
"an increment rounded on top of an absolute value"; the candidates for a second one, specific to
the density → SSH path, are the places where the port forms the **baroclinic pressure / density
anomaly** from absolute quantities in `real_t` where upstream forms it from an anomaly or in a
different order — `fesom_pressure_bv` (`density_m_rho0`, `hpressure`) and the ALE `hbar`/`ssh`
update. That is the next matched-stage instrument: **`density_m_rho0`, `hpressure` (per level),
`hbar`/`ssh`, `w`, after the dynamics of the step, per band** — the last untraced part of the
timestep.

### 4m. THE DYNAMICS INSTRUMENT — the SSH solve and w are CLEAN; the density anomaly is elevated at the surface
Built in both codes (2026-09-13): `FESOM_DYN_TRACE=<dir>` at four matched points of the dynamics —
`P1_post_pbv` (`density_m_rho0`, `hpressure`, per level), `P2_post_ssh` (`d_eta`, `ssh_rhs`),
`P3_post_hbar` (`hbar`), `P4_post_wvel` (`w`, interfaces). Port `fesom_step.cpp`; Fortran twin
`m16_dyn_trace2/3` in `oce_ale.F90` (LOCAL — explicit-shape dummies, after a first cut with
assumed-shape arrays segfaulted for want of an explicit interface). Knobs `_STEP`, `_EVERY`; job knob
`DYNTRACE=1`. `faith/dyntrace_1m`, jobs 27435321/27435221, every 96th step for a month. `hpressure`
is identically zero in both codes in this configuration (the pressure gradient is formed elsewhere) —
inert, dropped from the reading.

Port/fortran SP−DP ratio, **60–70°S band**, three traced steps:

| field | level | step 96 | step 768 | step 1440 | (50–60°N at 1440) |
|---|---|---|---|---|---|
| `density_m_rho0` | **surface** | **2.20** | 1.13 | **1.36** | 1.14 |
| `density_m_rho0` | level 10 | 0.98 | 0.98 | 1.01 | 0.92 |
| `density_m_rho0` | column | 1.42 | 1.13 | 1.18 | 1.21 |
| `d_eta` (SSH solve) | | **0.68** | 0.86 | 0.86 | 0.91 |
| `ssh_rhs` | | **0.66** | 0.83 | 0.85 | 0.91 |
| **`hbar`** | | 1.19 | **1.51** | **1.55** | 1.15 |
| `w` | interface 1 | 1.25 | 1.09 | 1.02 | 1.03 |
| `w` | level 10 / column | 1.06 / 0.91 | 0.85 / 0.84 | 0.84 / 0.84 | 0.91 / 0.90 |

**Three readings.**
1. 🔴 **The SSH solve is CLEAN in the band — the port is *better* (0.66–0.86).** With `d_eta` and
   `ssh_rhs` at parity-or-better, the barotropic-velocity error of §4k is **not generated by the
   solve**. And `w` is at parity (0.84–1.25). Hypothesis (b) of §4k is dead in its last form.
2. **`density_m_rho0` is elevated at the SURFACE only** (2.20 → 1.13 → 1.36; level 10 is 0.98–1.01
   throughout). That is the EOS evaluated on the surface T/S — which §4i showed are themselves at
   ~2.6 in the band by month 12. So the surface density anomaly is *inheriting* the tracer error,
   not creating it: it is not depth-structured the way an EOS defect would be.
3. **`hbar` is the one dynamics field that GROWS in the band** (1.19 → 1.51 → 1.55) while its own
   input `ssh_rhs` stays at 0.66–0.85. `hbar = hbar_old + ssh_rhs_old·dt/areasvol` in **both** codes
   (`oce_ale.F90` `compute_hbar_ale`; `fesom_momentum.cpp` `fesom_compute_hbar_kk`) — the
   increment-on-absolute accumulator of the §3u class, but written identically on both sides. Its
   growth is therefore a *consequence*: it integrates the divergence of the depth-integrated
   velocity (`c1 + c2` over `UV·helem`), and the band's velocity is at 1.8–2.0.

**So the loop closes on the velocity itself.** `u`/`v` at 1.8–2.0 at every depth (§4k), fed by an
SSH gradient whose solve is clean (`d_eta` 0.7–0.9) and by a density whose interior is clean
(level 10 at 1.0). What is left between "clean inputs" and "velocity at 2×" is the **momentum
step itself** in that band: `compute_vel_rhs` (Coriolis, the pressure-gradient force from
`density_m_rho0`, the SSH gradient), the horizontal viscosity (`opt_visc=5` easy backscatter —
§3s pinned the *scheme*, not audited its arithmetic at SP), `impl_vert_visc`, and `update_vel`.
That is the one part of the step no instrument has touched.

**Why 60–70°S would single out the momentum step:** it is the band of the **ACC** — the only
place where the barotropic flow is strong, deep and zonally unbounded, so the Coriolis and
pressure-gradient terms are each large and nearly balancing (geostrophy), and the momentum
tendency is a small residual of two large `real_t` terms — **the §3u cancellation class, in the
momentum equation**. The northern analogue (50–60°N) has no such flow. The next instrument is on
**`uv_rhs` after each term of `compute_vel_rhs`, and `uv` after `update_vel`**, per band.

### 4n. THE MOMENTUM INSTRUMENT — the momentum step is at parity too; the excess is a SLOW SECULAR divergence
Extended the dynamics trace to elements: `M1_post_pgf` (`pgf_x/y`), `M2_post_velrhs`
(`uv_rhs_x/y`, the whole explicit tendency — Coriolis + PGF + SSH gradient + viscosity),
`M3_post_updvel` (`u`/`v` after the barotropic correction), keyed by global element id. Port
`fesom_step.cpp` (`dyn_trace_write_elem`); Fortran `m16_dyn_trace3e` in `oce_ale.F90` (LOCAL).
`faith/dyntrace2_1m`, jobs 27435506/27435507, every 96th step. **Internal arrays, rotated frame in
both codes** — so both the within-code ratio and the cross-code magnitude are meaningful here.

**60–70°S, port/fortran SP−DP ratio (column), instantaneous, steps 96 / 768 / 1440:**

| field | 96 | 768 | 1440 |
|---|---|---|---|
| `pgf_x` / `pgf_y` | 1.05 / 1.04 | 1.16 / 1.11 | 1.25 / 1.20 |
| `uv_rhs_x` / `uv_rhs_y` | 1.13 / 1.05 | 0.85 / 1.04 | **0.91 / 0.73** |
| **`u` / `v` after `update_vel`** | 1.09 / 1.15 | 1.07 / 1.06 | **1.23 / 0.96** |

**Time-accumulated over the 15 traced steps** (the statistic that survives chaos), surface / column:
`pgf_x` 1.00 / 1.46 · `uv_rhs_x` 0.87 / 0.90 · `uv_rhs_y` 0.91 / 0.93 · `u` 1.11 / 1.12 · `v` 1.01 / 1.05.

🔴 **The momentum step is at parity in the band.** Neither the pressure-gradient force nor the
assembled tendency nor the corrected velocity shows the 2× at any instant of month 1, nor in the
accumulation. The §4m/§4k hypothesis — a §3u-class cancellation in the momentum equation — is
**dead**: the residual of Coriolis against the PGF is formed equally well in both codes.

**And that is consistent with the monthly output, once it is read month by month.** Monthly-mean
`u`, column relL2 ratio (`year_uv`/`year_g4`):

| month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **70–60°S** | **1.15** | 1.19 | 1.31 | 1.46 | 1.63 | 1.78 | 1.79 | 1.84 | 1.84 | 1.89 | 1.89 | **1.84** |
| 60–50°S | 1.13 | 1.17 | 1.25 | 1.27 | 1.27 | 1.32 | 1.32 | 1.34 | 1.36 | 1.31 | 1.26 | 1.27 |
| 50–60°N | 1.02 | 0.94 | 0.96 | 1.00 | 1.02 | 1.08 | 1.09 | 1.10 | 1.11 | 1.08 | 1.00 | 1.01 |

**Month 1 is 1.15 — matching the instrument — and it climbs monotonically to 1.9 by month 10,
while the northern band stays at 1.0 all year.** The excess is not made in any step's arithmetic.
It is a **slow, secular divergence of the SP trajectory from the DP trajectory that is ~2× faster in
the port than in upstream, in one band, over months.** That is a different class of thing from
everything found so far.

### 4o. What the whole ladder now says, and the one candidate it leaves
Every stage of every timestep has been traced in both codes — ice (§4g), tracers (§4i), surface
fluxes (§4j), density/SSH/w (§4m), momentum (§4n) — and **every one is at parity in 60–70°S at the
per-step level, in month 1**. Yet the band's SP−DP grows 1.15 → 1.9 over the year in the port and
does not in upstream, with **identical FP64 noise envelopes** (§4h: 0.98) — so it is not chaos.

A slow secular SP divergence with identical envelopes and clean per-step arithmetic has one
mechanism: a **conserved-quantity drift** — a small systematic bias per step that the dynamics
cannot damp because it is in a conserved or nearly-conserved quantity, so it integrates. In this
band the candidates are the **barotropic transport of the ACC** (set by the vertically integrated
momentum balance, and free to drift if the SSH/`hbar` accumulation biases) and the **layer
thicknesses** (`hnode`, under zstar the whole column stretches with `hbar`). Both would show as a
*mean* drift, not a variance — and `hbar` was the one dynamics field that grew (§4m, 1.19 → 1.55)
while its input `ssh_rhs` stayed at 0.7–0.85. `hbar = hbar_old + ssh_rhs_old·dt/areasvol` is
identical in both codes — but in the port, under zstar, `hbar` is **also** what the §3v ALE
reconstruction and `hnode_new` are built from each step, and the port's zstar thickness update
(`fesom_ale_update_thickness_zstar_kk`) is a port-only device kernel.

**The discriminating measurement is a mean-drift ledger, not another stage ratio**: the
band-mean of `hbar`, of the vertically-integrated transport, and of the column-integrated `T`
and `S`, SP−DP, as a time series in both codes. A ratio of variances (what every instrument so far
computed) cannot see a bias; a time series of the band mean can. It is cheap: all of it is
already in the monthly output.

### 4p. The mean-drift ledger — the bias hypothesis is REFUTED; the excess is in the VARIANCE
Band-mean of (SP − DP), monthly, both codes (`scratchpad/drift_ledger.py`, `year_uv`/`year_g4`).
70–60°S, month 12: `ssh` F +3.53e-04 / P +1.87e-04 · `temp` (column) F −2.16e-04 / P −1.12e-04 ·
`salt` (column) F +1.67e-04 / P +1.28e-04 · `sst` F −5.79e-04 / P −2.97e-04. Same sign in every
field, and the **port's mean drift is the smaller one**. There is no conserved-quantity bias
that the port carries and upstream does not. §4o's mechanism is dead too.

**So the excess is entirely in the variance** — the port's SP trajectory develops a *spatial
pattern* of departure from its DP twin in 60–70°S that is ~2× larger than upstream's by month 6,
with the same mean, the same per-step arithmetic at every traced stage, and an identical FP64
noise envelope. That combination has one remaining reading: the two codes' DP trajectories in that
band are **differently unstable to a rounding-sized perturbation in the specific direction that
single precision perturbs them** — the envelope (a random T perturbation) does not probe that
direction; SP (a structured, every-field, every-step perturbation) does. That is a property of
the *flow* in the band (the ACC's eddy field on the CORE2 mesh, ~1° there), not of any line of
code, and it is consistent with everything measured: no per-step defect, growth over months,
one band, port DP itself differing from Fortran DP by 1–2% in `u` by month 12.

### 4q. Where the SP faithfulness question now stands (2026-09-13)
| | verdict |
|---|---|
| ocean tracers, global | **at parity** (§3v, 0.70–1.28 all year) |
| ocean tracers, tropics | port **2× better** |
| ocean tracers, 60–70°S | port 2.6× worse at the surface (§4h) — **variance, not bias; not any traced stage** |
| sea ice | follows the 60–70°S ocean (§4g–§4h) |
| every stage of every timestep, both codes, per step | **parity** (§4g ice · §4i tracers · §4j fluxes · §4m density/SSH/w · §4n momentum) |
| conserved-quantity drift | **none** (§4p) |
| FP64 noise envelope | **identical** in the band (§4h) |

**What I would tell a reader.** Two real port defects were found by this ladder and fixed (§3p AB
seeding, §3v Redi→`del_ttf`), plus three conformance defects (§3r CG, §3s viscosity, §4b
`ice_diff`), and the port went from 2.5× worse than upstream at SP to global parity. The residual
is a single latitude band where the port's SP trajectory diverges from its DP twin ~2× faster than
upstream's over months, with no per-step arithmetic difference at any traced stage and no mean
bias. **That is not a bug that a stage instrument can find**, because no stage makes it; it is a
sensitivity of the band's flow. Whether it is *worth* chasing further depends on whether a
band-local 2× in a 1-year SP−DP ratio matters for the paper's claim — and the honest statement is
that after §3v the port and upstream are equally faithful to double precision everywhere except
one band where the port is 2× less so and one band where it is 2× more so.

**If it is chased further, the only remaining discriminator is a *structured* perturbation
envelope**: perturb the DP runs not with random T noise but with the actual SP−DP field at step
1 (or a scaled version), in both codes, and see whether the port's DP trajectory amplifies *that*
direction 2× faster. If it does, the band's difference is in the DP dynamics (the port's DP
already differs from upstream's DP by 1–2% in `u` there) and single precision is only the probe
that reveals it. That is a two-run experiment on existing infrastructure (`FESOM_PERTURB` reads a
field it can be given).

### 4r. The STRUCTURED-perturbation test — the sensitivity hypothesis is REFUTED, and the residual is re-diagnosed
Built in both codes: `FESOM_PERTURB_METHOD=file` (port `fesom_perturb.cpp`) / `FESOM_PERTURB_FILE`
(Fortran `m16_perturb_file` in `gen_ic3d.F90`, LOCAL, entered from `do_perturb`). The field is the
port's own step-1 (SP − DP) of T and S (`bisect_20s_g3`; rms 3.5e-04 K / 5.6e-05 psu, |max| 0.18 K
— the same order as the 2e-4 K random envelope), one file, applied identically to **both** codes'
DP initial state (both logs: rank 0, 1982 nodes, |max ΔT| 1.127e-02). One year, `faith/year_spert`,
jobs 27443872/27443911.

**60–70°S, port/fortran ratio of the departure (DP-perturbed vs DP) against the SP−DP ratio:**

| month | 1 | 3 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| sst: **structured P/F** | **1.02** | 0.98 | 0.90 | 0.79 | 0.93 | 0.89 | 1.30 |
| sst: SP−DP P/F | 1.33 | 2.00 | 2.49 | 2.24 | 2.20 | 2.31 | 2.63 |
| salt: **structured P/F** | **1.02** | 0.99 | 1.50 | 1.66 | 0.83 | 0.96 | 1.25 |
| salt: SP−DP P/F | 1.21 | 1.86 | 1.75 | 1.40 | 1.64 | 2.01 | 2.62 |
| ssh: **structured P/F** | **1.01** | 0.96 | 1.01 | 0.80 | 0.81 | 0.73 | 0.85 |
| ssh: SP−DP P/F | 1.52 | 1.98 | 2.16 | 2.19 | 2.24 | 2.33 | 2.15 |

🔴 **The two codes' DP dynamics amplify the SP direction IDENTICALLY** (0.7–1.1 all year, in the
very band where SP−DP is at 2–2.6). §4q's "band-local sensitivity of the flow" is **dead**.

**And the test says something sharper about what the SP−DP is.** In the band, the structured
perturbation's departure **decays** (fortran sst 3.1e-03 at month 1 → 3.0e-04 at month 6 → 1.0e-03
at month 12; the port the same) — the band's dynamics *damp* that direction — while the SP−DP
departure **grows** (2.5e-04 → 5.9e-03 upstream, 3.4e-04 → 1.5e-02 port). A one-shot perturbation
decays; SP−DP keeps growing. **So the SP−DP is continuous-injection dominated:** departure ≈
(per-step injection) × (damping time), the damping time is the same in both codes, and the 2× must
be in the injection.

**But the injection at step 1 is ~1× in the band** (`bisect_20s_g3`, step 1, 70–60°S: sst 1.09,
salt 1.33; step 20: 1.09 / 0.87), and the per-step tendency ledger over month 1 is noisy about 1
(0.6–2.2, no trend). And **it is not ice cover**: the band is already 71 % ice-covered in month 1
(cold start from winter climatology), its ice fraction *falls* to 57 % by month 3 while the ratio
*rises* 1.33 → 2.00, and the ice-free nodes of the band carry the same ratio as the whole band
(2.41 vs 2.36 at month 4).

**What that leaves is precise.** An injection that is ~1× at the start and ~2× later, with zero
band-mean bias (§4p), no amplification difference (§4r), no per-step arithmetic difference at any
traced stage (§4g–§4n) — a **patterned, systematic, threshold-driven difference that emerges as the
SP state evolves**: the signature of a **switching parameterisation** whose branch decisions flip
under single precision more often in the port than upstream, in the band where the decision is
marginal. In the Southern Ocean that is **KPP**: the boundary-layer depth is set by a bulk-Richardson
threshold, and 60–70°S winter is where stratification is weakest so the criterion is crossed over the
widest depth range by the smallest margins. A flipped column deepens or shoals its mixing for a step —
zero mean over the band, a persistent spatial pattern, invisible to a stage ratio of the *tracers*,
invisible to a random or structured perturbation of the DP run, and growing as the SP state drifts
into more marginal columns. The port's KPP (`fesom_kpp.cpp`) is its own implementation of the scheme.

**The discriminator is `Kv`** (the vertical diffusivity the scheme produces — the port writes it
monthly already) and `bvfreq`: their SP−DP ratio by band. If the mixing coefficient's SP−DP is at
2× in the band while the tracers feeding it are at ~1.2× in month 1, the scheme is the injector.
`faith/year_kv/{fdp,fsp}` adds both to the Fortran's `io_list` (jobs 27445755/27445756).

### 4s. `Kv` — KPP is at parity too; the switching-parameterisation hypothesis is dead
`faith/year_kv/{fdp,fsp}` added `Kv` to the Fortran's `io_list` (`bvfreq` is not a Fortran stream
name — "stream bvfreq is not defined" — dropped). Port/fortran ratio of the within-code SP−DP of
`Kv` by band, monthly:

| depth | month | 70–60°S | 60–50°S | 20°S–20°N | 50–60°N |
|---|---|---|---|---|---|
| top 5 levels | 1 / 3 / 6 / 9 / 12 | 1.03 / 1.07 / 1.03 / 1.01 / 1.10 | 1.06 / 1.08 / 1.16 / 1.08 / 1.05 | 1.01 / 1.05 / 0.93 / 0.51 / 0.58 | 1.07 / 1.06 / 1.01 / 1.02 / 1.19 |
| levels 5–20 | 1 / 3 / 6 / 9 / 12 | 1.01 / 0.82 / 0.95 / 1.03 / 1.03 | 0.99 / 0.83 / 1.09 / 1.06 / 1.07 | 1.08 / 1.08 / 1.02 / 0.67 / 0.68 | 1.05 / 1.08 / 0.95 / 0.95 / 1.03 |
| column | 1 / 3 / 6 / 9 / 12 | 1.00 / 0.96 / 0.99 / 1.11 / 1.22 | 0.97 / 1.04 / 1.08 / 1.13 / 1.20 | 1.04 / 1.05 / 0.94 / 0.79 / 0.71 | 1.05 / 1.04 / 0.94 / 1.00 / 0.89 |

🔴 **The mixing coefficient's single-precision departure is the same in both codes in the band —
0.93–1.22 all year, at every depth** — while its absolute SP−DP is enormous in both (upstream
0.04–0.73 relL2 in 60–70°S: KPP *is* a switching scheme, and both implementations switch equally
often under float). The "port's KPP flips more" hypothesis is **dead**. (The tropics show the port
*better* at 0.5–0.7 late in the year, matching the tracers' tropical advantage.)

### 4t. Where this leaves the 60–70°S residual — an honest close
Every hypothesis with a measurable consequence has now been tested with a matched instrument in
both codes, and every one is at parity in the band, per step:

| mechanism class | instrument | 60–70°S verdict |
|---|---|---|
| ice internals | §4g stage + subcycle trace | parity |
| tracer advection / Redi / reconstruction / vdiff | §4i stage trace | parity |
| surface fluxes | §4j, time-accumulated | parity or better |
| density / SSH solve / hbar / w | §4m | parity (SSH solve better) |
| momentum: PGF / tendency / velocity | §4n | parity |
| conserved-quantity bias | §4p mean-drift ledger | port smaller |
| chaotic amplification, random direction | §4h envelope | identical |
| amplification of the SP direction itself | §4r structured perturbation | **identical** |
| switching mixing scheme (KPP) | §4s `Kv` | parity |
| ice cover | §4r | refuted |
| configuration (opt_visc, ice_diff, CG precision) | §3s, §4b, §3r | fixed; nulls |

And yet the SP−DP variance in the band grows to 2–2.6× upstream's over months 2–6, with the
structured perturbation *decaying* over the same period in both codes. **A continuously injected,
zero-mean, band-local, time-growing SP−DP difference that no per-step instrument sees.** The one
consistent reading left is that the injection *is* at parity per step but the port's SP state in
the band drifts — within its own DP's basin, not into a different regime — to somewhere its float
rounding projects onto a slower-decaying mode; the tracer-stage ratios (§4i) do show the band's
*state* ratio rising 1.1 → 1.4 inside month 1 while the per-step tendency ratio stays ~1. That is a
property of the SP *trajectory*, not of any operation, and the tools built here — all of which
compare an operation's output given its input — cannot resolve it further. It would need an
ensemble of SP runs per code (the SP analogue of the noise twins) to even state it with error bars.

**What the paper can say, with the numbers behind it:** after the two real fixes (§3p, §3v) and
three conformance fixes (§3r, §3s, §4b) the port's single precision is as faithful to double as
upstream's globally (0.70–1.28, §3v), *more* faithful in the tropics (0.4–0.7), and *less* faithful
in one band, 60–70°S (2–2.6×), where every operation of the timestep, every surface flux, the mixing
scheme, the dynamics' amplification of random and of structured perturbations, and the mean drift
have been measured against upstream at parity. The band-local residual is reported as observed and
not explained. That is the honest state, and it took 26 instrumented comparisons to earn it.

## 5. The evidence-completion campaign (launched 2026-09-14, binary tag `final`)
After §4t the mechanism hunt is closed; what remained were runs the paper's *claims* depend on.
Binary **`bin/final`** = the current source (every fix + every env-gated instrument; Gate 0 PASS
np1 byte-identical, PROVENANCE lists md5s), Serial `{dp,sp}/fesom_port_serial` and CUDA
`{dp,sp}/fesom_port_cuda` (`Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF`). Faithfulness config on the
port job: `FESOM_VISC_OPT=5`, `FESOM_ICE_DIFF=0`, zstar, JRA55, no anomaly. 40 jobs:

| # | purpose | arms | root | jobs |
|---|---|---|---|---|
| 1 | **the final matrix, one binary, 5 seeds × 2 amplitudes** | port `pdp psp pdp_{r,s}{12345,22318,31415,27182,16180}` (12); Fortran `fdp_{r,s}{31415,27182,16180}` (6) + the 6 existing `year_1958` arms symlinked | `faith/final_year` | 27459756–767, 27459772–777 |
| 6 | **SP ensemble** (the one experiment that can put error bars on the 60–70°S residual) | `psp_r{12345,22318,31415}`, `fsp_r{12345,22318,31415}` — SP runs from perturbed ICs | `faith/final_year` | 27459768–770, 27459778–780 |
| 4 | **salt anomaly ON**, full year, both codes | `fdp fsp pdp psp` with #986 | `faith/final_year_anom` | 27459781–784 |
| 5 | **multi-year**: 3 years (the 8 h compute limit; 1 yr = 2h10 port / 1h50 Fortran) | `fdp fsp pdp psp` | `faith/final_3yr` | 27459785–788 |
| 2 | **speed on the final binary**: CORE2 GPU 1×4 and 2×8 knobs-off (the G2 production rows), CPU 1×64 | `dp sp` ABBA, `M16_BINS=bin/final` | `port2/m14/gladder.*` | 27459808, 27459809, 27459810 |
| 3 | **GPU faithfulness**: CUDA DP ×2 (self-noise floor), CUDA SP; + a Serial pair at the same physics | `gdp gdp_2 gsp` (new `jobs/job_m16_faith_gpu`, 2×4 GPUs) + Serial `pdp psp` with `VISCOPT=7` | `faith/final_year_gpu` | 27459817–819, 27459820–821 |

⚠️ **Two things to read the GPU rows with.** (a) `visc_filt_bcksct` (opt 5) is host-only — not
ported to CUDA — so the GPU arms run **opt 7**; the Serial pair in the same root runs opt 7 too so
the Serial-vs-GPU statement is same-physics. §3s measured the 5-vs-7 choice as a null for the SP
ratio, so the GPU ratio can still be set beside the Fortran's, with that caveat stated. (b) CUDA is
not run-to-run reproducible: `gsp − gdp` is read against `gdp_2 − gdp`, never against zero.

The multi-year port arm relies on the port's year-rollover output (`*.1959.*`, `*.1960.*`); the
compare script reads one file per arm, so a 3-year reader is needed when it lands.

## 6. GPU strong scaling, SP vs DP, four meshes (launched 2026-09-14, binary `final`)
The D13 campaign shape: **SP alone**, knobs-off (`FESOM_SPEED=1` — the certified M7 baseline — and
`FESOM_IC_EXTRAP=det` on both arms), the **CG solver**, none of the M9/M10/M11 levers, `WSPLIT=1` on
fArc/dars/NG5 (the cold-start rule), `-C a100_80`, 16 GPU nodes = 64 A100 cap. Protocol dts
CORE2 1800 · fArc 900 · dars 120 · NG5 180; 300 steps; ABBA `dp sp sp dp` after a discarded warm-up;
the number is the model's own per-step timer, min over legs; `gpumem_max=` polled per leg.
`jobs/job_m14_ladder_gpu`, `M16_BINS=bin/final`, one job per point:

| mesh | nodes (GPUs) | jobs |
|---|---|---|
| CORE2 | 1 (4) · 2 (8) · 4 (16) · 8 (32) · 16 (64) | 27459844, 848, 849, 850, 851 |
| fArc | 2 (8) · 4 (16) · 8 (32) · 16 (64) | 27459852, 854, 857, 859 |
| dars | 2 (8) · 4 (16) · 8 (32) · 16 (64) | 27459860, 861, 862, 863 |
| NG5 | 4 (16) · 8 (32) · 16 (64) | 27459864, 865, 866 |

Things to read the rows with: CORE2 is past its knee by 16 nodes (G2: 0.0618 s/step at 1N vs
0.0794 at 16N) — the curve is the point. The GPU runs `opt_visc=7` (bcksct is host-only). **NG5 FP64
at 16 nodes died at step 2 with the CG-NaN class on the `e0` binary (§1, 2026-09-08)** — that
allocator defect is fixed (`MALLOC_ASYNC=OFF`, pair `e3` onward), so whether the FP64 arm now
survives at 32/64 GPUs is itself a result; the SP arm survived then. dars/NG5 at 2/4 nodes are
memory attempts (`gpumem_max` will say). The det fill (~7 min NG5) sits outside the timing window
but inside the wall — NG5 jobs carry 1h40.

## 4. Untested list (kept honest)
- every M14 recipe knob at SP (G3); CA solvers `pipecg`/`pcsi`/`cg2` at SP; `FESOM_FORCING_POINTSLOPE`
  DP control leg; TKE `dbl_t` give-back; stiffness-shadow device-memory give-back.
