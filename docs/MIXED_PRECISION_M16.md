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

## 4. Untested list (kept honest)
- every M14 recipe knob at SP (G3); CA solvers `pipecg`/`pcsi`/`cg2` at SP; `FESOM_FORCING_POINTSLOPE`
  DP control leg; TKE `dbl_t` give-back; stiffness-shadow device-memory give-back.
