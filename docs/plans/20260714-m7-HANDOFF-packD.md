# M7 HANDOFF — package A is CLOSED; the FORCING is the prize. Next: D.1.

*Written 2026-07-14 (Opus session 4). Supersedes `20260714-m7-HANDOFF-packA.md`, whose work queue is
DONE. Branch `m7-speed`, **NOTHING pushed** (standing user decision).*

---

## 0. What landed this session

| commit | what | gates |
|---|---|---|
| `96dcf8a` | **Task A.1 `FESOM_SPEED_FLAT`** — 4 column-loop kernels flattened to one-thread-per-(column,level); pool 52.6 ms | knob-OFF byte ✅ · FORCE_SERIAL byte proof ✅ (**both** streams) · CUDA fidelity ✅ · announce ×3 |
| `faae871` | **Task D.0 `FESOM_SPEED_ROTCACHE`** — cache the JRA wind-rotation `sincos` (a MESH CONSTANT recomputed 1.85 M×/rank/step) | knob-OFF byte ✅ · FORCE_SERIAL byte proof ✅ **BIT-IDENTICAL** · CUDA fidelity ✅ (×2) · announce ✅ |
| `db5a324` | plan: D.1 data-flow map; killed a lever the arithmetic refutes | — |

**Tasks A.2 (hostprof) and A.3 (UCX) are CLOSED.** A.3 adopted nothing (`get_zcopy`/`put_zcopy` are
**+35%**; the only positive leg is −1.15%, below the >2% bar). A.2 is where the session turned — see §2.

## 0b. ✅ ALL A/Bs LANDED — package A is CLOSED, tags cut

| leg | NG5@4N (26244262) | dars@8N (26245783) |
|---|--:|--:|
| `t1` (Tier-1) | 0.9154 s/step | 0.2625 s/step |
| `flat` | **−3.07%** | −1.03% |
| `rot` | **−2.15%** | **−1.83%** |
| **`both`** | **−5.37%** → 0.8662 | **−2.90%** → 0.2549 |

**Ratio 5.01× → 5.29× at NG5@4N**, CPU anchor **4.5853 s/step** measured the same day (job 26244994).
Announce armor clean in every leg (`t1` 0/0 · `flat` 3/0 · `rot` 0/1 · `both` 3/1) — no dead knobs.
Tags: **`m7.2-packA`** (`96dcf8a`), **`m7.2b-rotcache`** (`faae871`).

**Both levers under-shot my headline estimate, and both times the PREDICTION was wrong, not the lever:**
- `ROTCACHE` removes only the **`sincos`** — not the `g2r` matrix arithmetic I had also counted. Correct
  pool = 23.5 − ~4 (new table loads) ≈ **20 ms**; measured **19.7 ms**. *Closes exactly.*
  → **Size a lever by the work it REMOVES, not by the whole function it lives in.**
- `FLAT` hit the **~50%-sector-efficiency ceiling** that the stride-2 interleaved `uv` layout imposes
  (the packA handoff called this out as trap #3). 53% of the pool is the right answer, not a shortfall.

## 0c. 🔴 THE FINDING THAT RE-RANKS THE LADDER — read before choosing the next package

**(a) L84 confirmed by a PRE-REGISTERED falsifiable test** (committed `8467cc6` *before* dars ran):
serial host work scales with **nodes/rank**. Predicted ROTCACHE at dars@8N = 4.19 ms; measured
**4.80 ms**. The rival hypothesis ("host cost doesn't scale") predicted ~19.7 ms = 7.5% of the step;
we measured 1.83%. **Decisive.**

**(b) UNPREDICTED, and it outranks both levers.** I predicted the kernel lever would hold up *better*
than linear. **It held up 2.2× WORSE** — and **the ordering INVERTS between scale points:**

| | NG5@4N (compute-bound) | dars@8N (comm-bound) |
|---|--:|--:|
| `FLAT` (GPU **kernel** lever) | **−3.07%** | −1.03% |
| `ROTCACHE` (**host** lever) | −2.15% | **−1.83%** |

*Mechanism:* in a comm-bound step, freed GPU-kernel time is absorbed by MPI wait the GPU was doing
anyway; freed **host** time is on the critical path regardless.

**dars@8N = 98.5 k nodes/rank ≈ NG5@16N's 115 k → it IS the Stage-2 proxy.** Therefore:
- 🔴 **Packages B (FCT2, 181.8 ms pool) and C (TDMA) are pure GPU-KERNEL levers — their 4N payoff will
  NOT carry to 16N**, which is exactly where Stage 2 / 2 SYPD lives. Size them at 4N, but **measure at
  dars@8N before banking anything at 16N.**
- ✅ **D.1 (forcing → device) is a HOST lever** → holds its value into the Stage-2 regime, *and* L84 says
  host work is 32× amplified on the GPU config. **Best-evidenced lever in the ladder.**
- ✅ **Package E (CG1R/EVPWIDE/CGPOLY) attacks the comm that is absorbing the kernel gains.** Priority UP.

⚠️ **Caveat:** dars is a different *mesh*, so "comm-bound" is confounded with geometry. The per-rank-proxy
method is faithful for halo/comm (not for CG iters). **Confirm at NG5@16N before betting the ladder** —
but the direction is strong enough to re-rank **D and E above B and C** now.

---

## 1. 🔴 The rank-count asymmetry (L84) — the most useful thing learned this session

ROTCACHE is a HOST fix, so I flagged it as a threat to the GPU-vs-CPU ratio ("it speeds the CPU
reference up too"). Right in principle, **wrong by 160×** — and the reason matters far more than the
lever:

| | ranks/node | **nodes/rank** @ NG5 4N | rotation trig |
|---|--:|--:|--:|
| **GPU** run | **4** (one per GPU) | **463 k** | 37 ms of a 913 ms step = **4.05%** |
| **CPU** run | **128** (one per core) | **14.5 k** | 1.2 ms of a 4599 ms step = **0.025%** |

The forcing loop is **per-rank serial host work**. The CPU run spreads the mesh over 128 processes per
node; the GPU run concentrates it into 4. **The GPU config therefore carries ~32× more host work per
rank, for identical code.** (Measured, not just derived: `jobs/job_m7_ab_cpu` (new), job **26244994**.)

**Three things follow, and they should steer the rest of the campaign:**
1. A host cost that is *invisible* in a CPU profile can *dominate* the GPU step. 0.025% vs 4%.
2. This is **why host code keeps being the answer in M7** — SWSKIP (−26%), IOACC, ICEFLUXDEV, and now
   the forcing. It is structural, not coincidence. **Expect the next bottleneck to be host code too.**
3. **The fix for host code here is "move it to the device", never "make the host loop faster."**
   D.0 (−4%) buys the amplification factor once; **D.1 (−8%) removes it.** D.0 is a down payment.

A host lever therefore does *not* inflate the ratio — but only because of the rank asymmetry, not
because it is GPU-specific. **Still re-measure the CPU anchor same-day when quoting a ratio** (that
rule is about ±5% cluster noise, not about this lever).

---

## 2. Why the campaign re-pointed: the host segment is the FORCING

The hostprof (job 26243196) was supposed to name the 66.9 ms "unnamed host segment". It did — and it
**mis-named the function**, which is now lesson **L82**.

- The **host timer** (not the sampler) says: **`force:jra55_read` = 75.2 ms/step at 1.0 calls/step**,
  plus `force:bulk_compute` 21.9 → the measured 97.1 ms forcing phase.
- The **sampler** blamed `getcoeffld` (38.3 ms/step). **That function's body runs ZERO times in a
  25-step run** — it refreshes only when `rdate` leaves the 3-hourly JRA interval (1 step in 60 at
  dt180), and the year-start prefetch covers step 1. (Verified by transcribing the C control flow to
  Python and running it against the real `uas.1958.nc` axis: refreshes land at steps 61/121/181.)
  `getcoeffld` is `static`→inlined, and `--backtrace=dwarf` **expands DWARF inline frames**, so the
  *caller's* per-node loop is reported under the *inlined callee's* name.
- **The real cost** is the per-node time-interp loop (`fesom_jra55.cpp:665-733`) over **463 k
  nodes/rank**, every step:

| | ms/step | killed by |
|---|--:|---|
| the rotation — 4 `sincos`/node (`fesom_vector_g2r`) | ~37 | ✅ **D.0** (it is a mesh constant) |
| the loop's own traffic — **24 concurrent streams** (16 `coef_a/coef_b` in, 8 arrays out, 89 MB/step) | ~38 | **D.1 only** |
| the 8 host halo exchanges | ~2.5 | **nothing — do not build this lever** (0.32 ms/exchange × 8; the arithmetic refutes it) |

**Rule that generalises (L82):** *before believing a per-step cost, ask how many times the thing is
CALLED per step.* Control flow beats any profile. A **host timer** around the call is the only host
measurement that cannot misattribute *which call* it timed.

---

## 3. NEXT TASK: D.1 — forcing loop → device (expected **−8%**, the biggest lever after FCT)

The full data-flow map and port checklist are in the plan (`Task D.1`, "D.1 DATA-FLOW MAP"). **Read it;
do not re-derive it.** The headline:

🔴 **THE LANDMINE.** The 8 forcing arrays are DualView `Field`s whose *raw host pointers* are written
in place by the host producer, invisibly to the DualView. So two rails —
**`fesom_bulk.cpp:520-531`** and **`fesom_ice.cpp:647-656`** — call `modify_host(); sync_device()` on
all 8, **unconditionally, every step**. The instant the producer becomes a device kernel, those rails
**deep-copy the stale host mirror over your fresh device data**, every step, before anything reads it.

🔴 **And the FORCE_SERIAL byte proof CANNOT catch it** — on Serial the host and device views are the
same memory, so the rail is a no-op. **It is CUDA-only and silent.** For D.1 the byte proof is
necessary but *not sufficient*; **the CUDA fidelity gate is the real gate.** (This is L80's shape
again: a gate that structurally cannot see the bug still passes.)

Also from the map: the **one remaining production host reader** is `fesom_cal_shortwave_rad`
(`fesom_bulk.cpp:757`, reads `jra->shortwave` every step — and **`SWSKIP` does NOT skip it**: the
`heat_flux += swsurf` side-effect always runs). `h_checked()`/SYNCCHECK structurally cannot trap it
(raw alias). The 8 host halo exchanges must become one `fesom_halo_fieldN` call, because the producer
covers `myDim` only while every device consumer reads `[0, myDim+eDim)`.

After D.1: **package B (FCT2)** — 181.8 ms pool, still the single biggest.

---

## 4. Machinery notes added this session

- **`scripts/diff_snap.py --pattern GLOB`** (default `snap_*.nc`, so every existing gate is unchanged).
  Needed because the `io_acc_*` accumulators write ONLY to `*.monthly.nc` — so the FORCE_SERIAL proof
  would have run those kernels and **compared nothing they wrote**, and passed (**L83**). Any lever
  touching an io resolver must run the proof on `--pattern '*.monthly.nc'` too.
- **`fesom_speed_on_exp()`** (`src/fesom_speed.hpp`) — a lever that the `FESOM_SPEED` master switch
  does **not** turn on; it must be named explicitly. `SYNCSTATS` now uses it (a diagnostic must not
  fire because a perf config was requested). **ICELAG (F.1) must use it too.**
- Compile-checking without a rebuild race: extract the compile line from
  `build-m7cuda/compile_commands.json` and compile the changed TU to a scratch `.o`. Catches syntax/type
  errors while a job is queued against `build-m7cuda`'s *binary*.

## 5. Standing rules (unchanged, still binding)

- **Never rebuild a tree while a job is PENDING against it** — queued jobs run whatever binary is on
  disk at start time. Batch changes, rebuild once, then submit. (The Serial tree is independent of the
  CUDA tree; rebuilding `build-m7serial` while a GPU job is queued is safe.)
- Same-day, same-allocation A/Bs only; 35 steps, 2 reps, min; dt180. Output under
  `/work/ab0995/a270088/port2/m7/` only.
- Every lever announces itself on rank 0. **Check the line before believing a null result.**
- **Nothing is pushed.** Ask first.
