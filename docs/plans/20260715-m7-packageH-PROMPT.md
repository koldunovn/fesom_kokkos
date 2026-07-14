# M7 next session — PROMPT

*Written 2026-07-14 ~21:00, close of session 5. Branch `m7-speed`, **HEAD `783e035`**, working tree
clean. **NOTHING PUSHED. NOT TAGGED.** (Both are standing user decisions — ask before doing either.)*

**Read in this order:** this file → `docs/plans/20260714-m7-HANDOFF-H.md` → `docs/plans/20260714-m7-PACKAGE-H-rails.md`.

---

## 0. 🔴 THE FOUR RULES. Read them before you touch anything. They are not boilerplate — every one of
## them was paid for with a wasted job or a wrong answer THIS session.

**0.1 — "ALWAYS MEASURE, DO NOT GUESS."** *(User directive, verbatim, 2026-07-14.)*
In one session I reasoned my way to **four** confident, wrong answers, each killed in minutes by a
cheap instrument:

| I reasoned | the instrument said |
|---|---|
| "94.5 % of PCIe copies are inside MPI ⇒ CUDA-aware MPI is host-staging the halo — the biggest lever in the campaign" | **Opposite culprit.** That was a **COUNT**; the MPI copies are the tiny ones. By **TIME** it is the model's OWN DualView rails (77.6 ms vs 18.2). → **L85** |
| "D.1 will save 57.3 ms (−6.6 %)" — off the FPROF host timer's 75.2 ms | **−2.16 %.** A host timer says how long the HOST SAT THERE, not what is **REMOVABLE**. → **L87** |
| "`getcoeffld` is a ~5.6 % production lever" | A **cold-start artifact**. (And L82 had reasoned it "runs ZERO times" — it runs **8.0 calls/step**.) |
| "corrected ratio ≈ 5.83× / 6.2×" | **Unearned. Retracted.** It mixed a long-run GPU number with a 35-step CPU anchor. |

Three separate airtight-looking hypotheses for the `getcoeffld` guard bug each DIED on inspection.
The fourth approach — **print the actual numbers** — took four minutes and was right.
**Both levers that BEAT their pre-registration were sized from MEASUREMENTS. The one that missed by 3×
was sized from an ARGUMENT.**
→ `[[feedback-always-measure]]`, `[[feedback-profiler-traps]]`

**0.2 — L86: NO SERIAL GATE CAN EVER VALIDATE A COHERENCE INVARIANT.**
On Serial `.d()` and `.h()` are the **same memory**, so any rail / `modify_*` / `sync_*` change is a
**no-op** there. **Proven the hard way:** ICERAILS' first cut passed the knob-OFF byte gate **and the
FORCE_SERIAL byte proof BIT-IDENTICALLY (`diff_snap rc=0`)** — while the CUDA fidelity gate showed
**`a_ice max|Δ| = 9.83e-01`: THERE WAS NO SEA ICE.** The project's *strongest* gate certified a lever
that deletes the sea ice. **The standalone CUDA fidelity gate costs 42 seconds. Run it.**
**Know which class your change is in:** a *re-execution elimination* (SWSKIP / the getcoeffld fix) IS
provable by the byte gates. A *coherence invariant* is NOT.

**0.3 — Before deleting a rail because "both sides are device kernels", ASK WHO PUT THE INITIAL VALUE
THERE.** `Field::alloc()` zero-inits BOTH spaces and marks them `Synced`; an IC written host-side
through a raw alias is invisible to the DualView. **The per-step IN rail is often ALSO, silently, the
IC push** — true on every step but the first, and step 1 being wrong is the whole run.

**0.4 — ALWAYS PIN A JOB WITH `BIN=`. NEVER rely on a build tree.**
The old rule ("never rebuild while a job is PENDING") is **too weak, and I broke it**: a **RUNNING
multi-`srun` job re-execs the binary at EVERY `srun`**. The A/B jobs run one `srun` per rep and report
the **min**, so a mid-job rebuild silently mixes two binaries and reports the faster one. *(Job 26255761
was killed for exactly this.)*
Frozen binaries: `m7/bin/{tier1,a1,packa,h1,h2,h3,h4,h5}/`.
🔴 **`h3` = the BROKEN ICERAILS (no sea ice). DO NOT USE IT.**
✅ **`h5` = the current best** (all levers + the getcoeffld fix): CUDA `0d39d8a2`, Serial `950ee0f9`.

*(Also standing: **L80** — every lever announces on rank 0; **check the line** before believing a null
A/B. And a knob-OFF byte gate must be green at every commit.)*

---

## 1. 🔴 HARVEST THESE FIRST. Every expectation below was PRE-REGISTERED before the job ran — score
## them honestly and **DO NOT RETROFIT.** (They are also in the commit messages, so you can't quietly
## rewrite them.)

| job | what | PRE-REGISTERED |
|---|---|---|
| **26248860** | **NG5@16N 4-leg scan** (`t1`/`flat`/`rot`/`both`), pinned `m7/bin/packa` | 🔴 **THE LADDER TEST — it decides packages B and C.** `FLAT` near **−2.0 %** ⇒ the L84(b) comm-decay story is WRONG and **B/C go back on top**. Near **−0.9 %** ⇒ it stands. `ROTCACHE` −1.4 % either way. ⚠️ The **NG5@8N** scan already weakened the decay story (at 8N the ordering does NOT invert: FLAT −2.55 % still beats ROT −2.23 %), so the dars@8N inversion is at least partly a **mesh** artifact. **Either way it does not touch package H, which is host-class.** |
| ~~26255934~~ | 25-step step-profile, `h5` | ✅ **HARVESTED — HIT.** loop **0.8778 → 0.8297** (pre-reg ~0.829). `force:getcoeffld` + `force:jra55_read` **vanished from the profile** (below the 0.45 % print cutoff; were 5.62 % / 5.76 %). **CONTROL: CG stayed at 90.3 iters/step and 1.134 s — identical. The fix was surgical.** |
| ~~26255935~~ | same-alloc A/B on `h5`: `NOCOEFCACHE=1` vs default | ✅ **HARVESTED.** `nocache` 0.7852 (reproduces ICERAILS' 0.7857 to 0.06 %) → `cache` **0.7435 = −5.31 % (41.7 ms)**. Pre-reg −5.7 % (45 ms) — **I over-predicted by 7 %; own it.** |
| **26255936** | **GPU anchor @300 steps**, `h5`, `FESOM_SPEED=1`, unprofiled, **pinned** | — (this IS the measurement) |
| **26255937** | **CPU anchor @300 steps**, serial `h5`, dist_512, **pinned** | — (this IS the measurement) |
| **26256274** | **nsys GAP CENSUS, 300 steps**, `h5`, `NSYS_TRACE=cuda,mpi NSYS_SAMPLE=none` | names the last **~22 ms** (see §3.2) |
| **26256275** | **nsys GAP CENSUS, 25 steps**, SAME binary, SAME flags — the apples-to-apples baseline | (without it the 300-step trace has nothing to diff against) |

---

## 2. WHERE THE CAMPAIGN IS

**Package H (THE RAILS) — the biggest lever in the campaign. Found by re-mining an nsys sqlite that was
already on disk, with ZERO new allocation.** The plan's budget row `memcpy 89.8 ms (9.8 %)`, labelled
*"MPI staging + forcing HtoD"*, was **mis-attributed**: split by TIME, MPI/UCX staging is 18.2 ms and
**the model's OWN full-field DualView rails are 77.6 ms/step = 9 % of the step** (114 deep-copies of a
FULL nod2D field — 3.71 MB = 463,386 doubles = the whole local domain — per step, all on the critical
path). Confirmed by **three independent instruments**: memcpy accounting (77.6 ms), a source audit
(67 rails in the ice step), and a **GPU-idle gap census** (58.7 ms of the GPU doing *nothing* before the
three sea-ice kernels). Host-class ⇒ by **L84(b)** it HOLDS at 16N.

| commit | lever | NG5@4N | vs pre-registered |
|---|---|--:|---|
| `c0c967a` | **eta_n / IOACC-ssh correctness fix** (no knob) | — | *proven* by a controlled differential: monthly `ssh` **7.06e-02 → 2.36e-06** while every other field stayed at the CUDA noise floor |
| `92a43d8` | **H.1 `FESOM_SPEED_FLUXDEV`** — 4 forcing fluxes device-authoritative, 11 rails killed | **−1.45 %** | −1.1 % — **BEAT it** (sized from *measured* per-copy PCIe costs) |
| `a96e299` | **H.2 `FESOM_SPEED_ICERAILS`** — **55 rails** killed from the ice step | **−6.03 %** | −4.1 % — **BEAT it** (the *gap census* predicted 58.7 ms; the memcpy arithmetic said 34; we got 50.4) |
| `7f64be1` | **the `getcoeffld` guard fix** (bit-identical, unconditional) | *benchmark only* | see §3.1 |

**Ratio ladder this session (same-day CPU anchor 4.5853, 35-step protocol):**
5.29× (packA) → 5.41× (D.1) → 5.49× (FLUXDEV) → **5.84× (ICERAILS)**.
🔴 **That 5.84× is measured on a CONTAMINATED protocol — see §3.1. Do not quote it as final.**

---

## 3. 🔴🔴 THE TWO OPEN QUESTIONS. Both are measurement questions. Do not answer them with arithmetic.

### 3.1 THE RATIO IS UNEARNED. Re-measure it; do not "adjust" it.

**`getcoeffld` rebuilt the JRA interpolation coefficients 8.0 times per step, EVERY step** (49.3 ms/step
= 5.62 %), because **the model starts BEFORE the first JRA record** (`uas/vas/huss/tas` at **+1.5 h**;
`rsds/rlds/prra/prsn` at **+3 h**). `binarysearch` returns 0 → the "no extrapolation back in time" clamp
sets **`t_indx_p1 = t_indx`** → `lo == hi` → `rdate < lo` is **true forever**. `rdate` advances 3 min/step,
so it takes **30–60 steps to escape — and every benchmark in this campaign is 35 steps.**
**L82's "getcoeffld runs ZERO times" is RETRACTED.** *(Its rule — ask how many times the thing is CALLED
— is right. It was answered with an argument instead of a counter.)*

**FIXED bit-identically in `7f64be1`** (cache the `(t_indx, t_indx_p1)` bracket; skip the rebuild;
invalidate on year change). **Deliberately UNCONDITIONAL, not a `FESOM_SPEED_*` knob** — those resolve
OFF on non-CUDA builds, which would fix the GPU benchmark and leave the CPU one broken, *manufacturing
the very asymmetry we are removing*. Only the DISABLE is a knob (`FESOM_SPEED_NOCOEFCACHE`, opt-in-only).
Gates: knob-OFF byte `rc=0` (**the bit-identity proof, and it is MEANINGFUL here**) · FORCE_SERIAL `rc=0`
· CUDA fidelity all-knobs PASS.

**⚠️ THIS IS NOT A PRODUCTION SPEEDUP.** In a 1-yr run the clamp lasts ~30–60 of ~175,000 steps. **It is
a MEASUREMENT bug — and an ASYMMETRIC one.** `getcoeffld` is HOST code over `myDim_nod2D`, and per
**L84** the GPU config carries **463 k nodes/rank** against the CPU's **14.5 k** (4 ranks/node vs 128):

| | nodes/rank | artifact | step | inflation |
|---|--:|--:|--:|--:|
| NG5@4N **GPU** | 463 k | 45 ms | 0.786 s | **5.7 %** |
| NG5@8N GPU | 231 k | 22 ms | ~0.50 s | 4.5 % |
| NG5@16N GPU | 116 k | 11 ms | ~0.32 s | 3.5 % |
| NG5 **CPU** (any) | 3.6–14.5 k | 0.4–1.4 ms | 1.2–4.6 s | **0.03 %** |

**✅ MEASURED (26255935): the 35-step ratio moved 5.83× → 6.17×, i.e. ×1.057. I pre-registered ×1.06
→ 6.19×: a HIT, which also validates the scaling model behind it (artifact ∝ nodes/rank; the GPU at
463 k/rank paid ~150× what the CPU paid at 14.5 k/rank — L84).**

🔴 **BUT 6.17× IS THE 35-STEP NUMBER, NOT THE FINAL ONE.** The protocol still carries **~31 ms** the
guard fix does not touch — the **CG spin-up (8.7 ms; the post-fix profile still shows 90.3 iters/step)**
plus **~22 ms still unattributed** (§3.2). That is ~4 % of the GPU step, and the CPU's share is unknown.
**Original arithmetic, now superseded, kept for the record: ~5.84× → ~6.2×.** *(I already
quoted "5.83× / 6.2×" once and had to retract it — it mixed a long-run GPU number with a 35-step CPU
anchor. **You cannot mix protocols and get an honest ratio.**)*
🔴 **Jobs 26255936 + 26255937 are the MATCHED, PINNED, BOTH-POST-FIX pair. Harvest both, then re-derive.**

**⇒ THE RATIO LEDGER (`docs/GPU_SPEED_M7.md`) MUST BE RE-MEASURED, NOT ADJUSTED.** Its rows are Tier-1
vintage and contain none of D.1 / FLUXDEV / ICERAILS anyway. The standard set (NG5@4N/8N/16N + dars@8N,
GPU **and** CPU) needs re-running on `h5` with a protocol that clears the cold start.
*(Per-lever MARGINAL A/Bs do NOT need re-running — the artifact sits in both legs and only inflates the
denominator; the percentages are understated by ~5 % of themselves. A footnote fixes them.)*

### 3.2 THE 35-STEP PROTOCOL IS CONTAMINATED BY MORE THAN `getcoeffld`. ~22 ms is still unattributed.

The 300-step run came in **72.6 ms/step faster** than the 25-step run (same binary, same config,
job 26255546 — a **pre-registered falsification test that PASSED**: `getcoeffld` 8.0 → 1.2 calls/step,
49.3 → 7.7 ms, exactly as predicted). Diffing the two step-**profiles** attributes 50 of it:

| | 25-step | 300-step | Δ |
|---|--:|--:|--:|
| `force:jra55_read` (contains getcoeffld) | 50.6 | 8.7 | **−41.9 ms** ← FIXED (`7f64be1`) |
| **`7_ssh`** (the SSH solve) | 84.9 | 76.2 | **−8.7 ms** |
| **CG iterations/step** | **90.3** | **76.6** | **−15 %** |
| loop timing | 877.8 | 805.2 | **−72.6 ms** |

**The CG solver needs ~15 % more iterations at cold start** — the barotropic mode is still adjusting from
rest. That is **physics, not a bug**, but it is still a benchmark artifact: a production run from a
spun-up restart converges in ~76 iters, not 90. *(Knock-on: the SYPD projection uses a **×1.03 CG
correction calibrated on 90-iteration behaviour**. If production is 76, that correction is wrong —
pessimistically. Re-derive it.)*

**~22 ms remains unattributed.** It sits in the **halos / host / MPI remainder that the PHASE profiler
structurally cannot see** (the phases only sum to ~96 % of the step). **Only a trace reaches it.**
🔴 **Jobs 26256274 (300 steps) + 26256275 (25 steps) are the matched nsys gap-census pair. Diff their
GPU-idle gap censuses and NAME IT.**

**THE GPU-IDLE GAP CENSUS is the cheapest and most honest diagnostic in this project** — free, from any
nsys sqlite already on disk, and it beat the memcpy arithmetic on ICERAILS (predicted 58.7 ms, actual
50.4; the arithmetic said 34). Recipe: bracket each step with a once-per-step kernel (`fesom_timestep`),
then for every gap `> 1 ms` between consecutive kernels, group by *the kernel the gap ENDS at*.
⚠️ **Two nsys traps**: kernel `shortName` collapses to `cuda_parallel_launch_local_memory` — **use
`demangledName`** and regex the functor out of `ParallelFor<...>`; and `MPI_START_WAIT_EVENTS` emits
**one row per request**, so a `Waitall` on 10 requests appears 10× — **dedupe by `(start,end)`** or you
will over-count its time 10× (raw 22,460 ms vs deduped 66 ms/step).

---

## 4. THE LADDER, re-ranked on MEASUREMENT (not on the plan's estimates, which have run 2–3× optimistic)

| pkg | class | 4N | holds at 16N? | status / what is actually known |
|---|---|--:|---|---|
| ~~H.1 FLUXDEV~~ | host | −1.45 % | ✅ | **LANDED** |
| ~~H.2 ICERAILS~~ | host | **−6.03 %** | ✅ | **LANDED** |
| **H.3 BULKTAIL** | host | ~1 % | ✅ | **DO THIS NEXT.** `fesom_bulk.cpp:629-650`: **7 dead `sync_host()`** (~30 MB/step — every one re-pushed or overwritten before any host read) **+ a dead 930 k-element host interpolation loop** overwritten µs later by `oce_fluxes_mom`. Pure SWSKIP-pattern deletion, **no new kernel**. |
| **H.4 SSHRAILS** | host | ~0.7 % | ✅ | `fesom_step.cpp:638-674`: 5 D2H + 4 H2D + 4 host-staged halos — the last host halos in an otherwise device-halo'd ocean step |
| **H.5 ETADEV** | host | ~0.2 % | ✅ | `eta_n` substep-11 host loop (466 k iters) → device. Also deletes the rail `c0c967a` added **and** the now-provably-redundant substep-4 push (`fesom_step.cpp:511`) |
| **halo depth-pack** | comm | ~1.5 % | ✅ grows | `fesom_halo_device.cpp:348` ships `n_levels = mesh->nl` (=70) for **every** halo node regardless of seafloor depth — **a 1-level shelf node transmits 70 doubles, 69 of them below the bottom**, while the compute kernels beside it ARE depth-masked. Plus **~7 of the ~44 3D exchanges/step are provably redundant** (`helem` twice on zstar; `hnode` dead on zstar; the KPP blmc smoother's last-sweep halos — and **blmc alone is 12 of the 44**) |
| **E — comm** | comm | ~2 % *(plan said 4.6–9 %)* | ✅ grows | **RE-SIZED.** `fesom_profile.cpp:61,:73` fences around every labelled kernel → with 5 labelled kernels × 120 EVP subcycles it **inflates the phase it measures**. nsys says the EVP loop is **38.3 ms, not 73.5**. Whole EVP comm pool ≈ 26 ms ⇒ EVPWIDE k=2 ≈ **1.5 %** at 4N. `MPI_Allreduce` = **3.9 ms/step TOTAL** ⇒ CG1R ≈ **2 ms**. Still the right **Stage-2** lever; **not** the big one at 4N. |
| **B — FCT2** | **kernel** | 182.5 ms pool | 🔴 **26248860 decides** | FCT is **bandwidth-bound, NOT occupancy- or launch-bound** (16 registers, 3.3 ms/launch) ⇒ B.1's T+S batching is a **data-reuse** play, and its payoff hinges on **how much of FCT's traffic is tracer-invariant. NOBODY HAS MEASURED THAT. Measure it before committing to B.** |
| **C — TDMA/spill** | **kernel** | ~166 ms pool | 🔴 **26248860 decides** | **The plan's pool is WRONG.** ⚠️ nsys `localMemoryPerThread = 0` is a **NON-COLLECTION ARTIFACT** — use **`cuobjdump --dump-resource-usage`** (free, exact, no GPU). It confirms **14 kernels DO spill**. Cross-referenced against time: the spilling pool is **~166 ms/step, not the ~107 budgeted** — and **the biggest spiller by time, `diff_ver_part_redi_expl` (42.3 ms, 5,120 B/thread, 58 reg → 55 % occupancy), IS NOT IN PACKAGE C'S LIST AT ALL.** Also `diff_part_hor_redi` (23.7 ms, **80 reg → 40 % occ**) and `impl_vert_visc` (13.4 ms, **82 reg → 39 % occ**, 7,168 B). |
| F — ICELAG | physics | −90..120 ms | ? | user-approved **experiment**; needs its own 1-yr climate + sign-off |

**Honest 8× outlook:** post-correction ≈ **6.2×** at 4N (hypothesis — see §3.1). The stretch needs
another **−22 %**, which only **B / C / F** are big enough to deliver. **Every package estimate in this
plan has come in 2–3× optimistic when measured.** ⇒ **~7× is the realistic landing zone without F.**
Stage 2 (2 SYPD at 16N) looks comfortably met *on projection* — but that needs the 16N standard set.

---

## 5. SUGGESTED ORDER OF WORK

1. **Harvest §1** (all 7). Score against the pre-registrations. **If a number contradicts a claim in
   this file, the claim is wrong — say so and retract it. Do not retrofit.**
2. **§3.2** — diff the two gap censuses (26256274 vs 26256275) and **NAME THE ~22 ms.**
3. **§3.1** — from the matched anchors (26255936 + 26255937), **RE-DERIVE THE RATIO.** Then re-measure
   the standard set on `h5` with a protocol that clears the cold start, and **rewrite the ledger in
   `docs/GPU_SPEED_M7.md`** (do not adjust it). Re-derive the ×1.03 SYPD CG correction from the settled
   76.6 iters/step, not the cold-start 90.3.
4. **§4 — implement H.3 BULKTAIL** (the next lever; pure deletion, no new kernel).
5. **26248860 decides B vs C vs E.** Before committing to **B**, measure FCT's tracer-invariant traffic
   fraction. Before committing to **C**, note `redi_expl` (42.3 ms) is the biggest spiller and is not in
   the package.

**Gate ladder for every lever (unchanged):** knob-OFF byte gate → FORCE_SERIAL byte proof (*only
meaningful for re-execution eliminations — see 0.2*) → **standalone CUDA fidelity gate** → same-alloc
A/B → ledger → commit. **Pre-register the number BEFORE the job lands.** Check the **announce** line.

## 6. USER PREFERENCES (standing)

- **"Always measure, do not guess."**
- **Nothing is pushed, and nothing is tagged, without asking.**
- Mixed precision is **BANNED** — everything stays FP64.
- Big runs go through SLURM, never the login node. Output under `/work/ab0995/a270088/port2/m7/` only
  (`$HOME` has a 60 GB quota and is at 54 GB).
- Use the **private mesh** `/work/ab0995/a270088/port2/mesh/core2` for CORE2 gates, **never `/pool`**
  (L73 — /pool's core2 `nlvls.out`/`elvls.out` were swapped 2026-07-03). *NG5/dars perf runs DO use
  /pool; the swap was core2-only.*
