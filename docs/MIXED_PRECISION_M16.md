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
| fArc | CPU (Serial) | 32 × 4096 | **recipe**: `FESOM_ALE=zstar FESOM_SSH_MODE=se FESOM_SE_M=90 FESOM_SALT_ANOMALY=1`, `WSPLIT=1` | 0.0453 | 0.0376 | **0.830 (−17.0 %)** | — (SE, no CG) | 0.00 % / 0.53 % | 27293953 (`e2`); the recipe alone: FP64 0.0578 → 0.0453 (−21.6 %), SP 0.0494 → 0.0376 (−23.9 %) |
| fArc | CPU (Serial, 128/node) | 32 × 4096 | knobs-off (`FESOM_SPEED=1` inert on CPU, det), `WSPLIT=1` | 0.0578 | 0.0494 | **0.855 (−14.5 %)** | 149 / 149 | 1.38 % / 1.01 % | 27289249 |
| dars | CPU (Serial, 128/node) | 64 × 8192 | knobs-off, `WSPLIT=1` | 0.0979 | 0.0717 | **0.732 (−26.8 %)** | 22 / 22 | 0.10 % / 0.28 % | 27289250 |

Recipe rows (BASE_KNOBS = the per-point M14 recipe) follow once the knobs-off rows are in. ⚠️ The M14 CPU recipe lever `FESOM_SSH_SOLVER=oati` is unusable at SP as built (§2) — the SP recipe row on fArc/dars uses `pcsi` or plain `cg`.

Reading so far: SP and the recipe levers are close to multiplicative (CORE2 16N: recipe −23.6 %, SP −6.4 % knobs-off / −7.6 % under the recipe; fArc 4096: recipe −21.6 %, SP −14.5 % / −17.0 %) — July's "they overlap in the communication bytes" is at most a few percent here. The SP gain grows with the BYTE share of the step and shrinks where latency rules (CORE2 1N GPU −14 %, CORE2 16N GPU −6.4 % at 0.08 s/step, fArc 4096 CPU −14.5 %, dars 8192 CPU −27 %) — July's headline ("SP and the speed stack overlap in the communication bytes") reproduced on the m14 tree with knobs off.

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


## 3. Gate 4 — screens and the 1-year twin (pending)
## 4. Untested list (kept honest)
- every M14 recipe knob at SP (G3); CA solvers `pipecg`/`pcsi`/`cg2` at SP; `FESOM_FORCING_POINTSLOPE`
  DP control leg; TKE `dbl_t` give-back; stiffness-shadow device-memory give-back.
