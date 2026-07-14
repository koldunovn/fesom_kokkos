> # ⛔ SUPERSEDED (2026-07-14, Opus session 4) — its work queue is DONE.
> **Read `docs/plans/20260714-m7-HANDOFF-packD.md` instead.** Task A.1 (`FESOM_SPEED_FLAT`) landed
> @ `96dcf8a` with all gates green; A.2 (hostprof) and A.3 (UCX) are closed. The §1 work queue, the
> §2 A.1 spec and the §3 gate ladder below are all executed. **Still true and still binding:** §4's
> traps (esp. #1, the rebuild race) and §5.
> **What changed:** A.2's answer re-pointed the campaign at the FORCING (75.2 ms/step of host time),
> and the sampler MIS-NAMED the function inside it (→ L82). New Task D.0 `ROTCACHE` landed @ `faae871`.

# M7 HANDOFF — Package A implementation (re-scope is DONE; this is a build-and-gate session)

*Written 2026-07-14 at the close of the re-scope session. The NEXT session implements — follow this
document and the plan literally; every design decision here is already made and argued, and the
measured numbers below are the acceptance criteria. Branch `m7-speed`, NOTHING pushed (user
decision 2026-07-14: stays local).*

**Read first, in this order:**
1. The **RE-SCOPE section** of `docs/plans/20260714-m7-speed-fp64.md` — the plan of record
   (packages A–F, measured budget, the 8× arithmetic, what got demoted and why).
2. §1 (L80/L81), §5 (machinery), §6 (standing rules) of `20260714-m7-HANDOFF-next.md` — still
   binding. Its §3/§4 are superseded.
3. This file — the concrete work queue.

---

## 0. State at handoff

- **Tier 1 complete + validated:** NG5@4N **5.03×** (0.9145 s/step), 8N 4.28×, 16N 3.55×
  (SYPD@dt240 1.86), dars@8N 3.27×. All bit-identical, climate gate identical to the un-levered
  bar to five decimals. Tags `m7.0-baseline`, `m7.1-bitid`, `m7.1-stage1`.
- **Measured post-Tier-1 budget** (jobs 26242512 nsys + 26242513 phase profile, frozen Tier-1
  binary `788844b3`, knobs verified live): step **913.5 ms** = kernels 591.8 (64.8%) · memcpy 89.8 ·
  MPI 138.1 · launch/API 46.5 · fence spin 18.1 · **unnamed host segment 66.9**. Phases: ocean
  71.9% · **sea-ice 129.7 ms** · **forcing 97.1 ms** · coupling 0.3% (dead). Full tables:
  `docs/GPU_SPEED_M7.md` ("Post-Tier-1 MEASURED budget") and the plan's RE-SCOPE section.
- **User decisions (2026-07-14):** packages A–F approved, start with A; **ICELAG approved as an
  EXPERIMENT** (implement without re-asking; adoption still needs its own 1-yr climate + user
  review); **no push**.
- **Nothing in `src/` changed this session.** Working tree at handoff: plan re-scope, this handoff,
  `docs/GPU_SPEED_M7.md` budget section, new `jobs/job_m7_ab_env`. Task A.1 is fully designed
  (§2) but **unimplemented**.

### In-flight jobs — HARVEST THESE FIRST

| job | what | where |
|---|---|---|
| **26243196** | Task A.2 hostprof (CPU call-stack sampling, Tier-1 binary, `FESOM_SPEED=1`, NG5@4N, 25 steps) | `/work/ab0995/a270088/port2/m7/hostprof_t1/` + `…/m7/hostprof.26243196.out` |
| **26243303** | Task A.3 UCX 4-leg env A/B (ref / `UCX_RNDV_SCHEME=get_zcopy` / `UCX_RNDV_FRAG_MEM_TYPE=cuda` / `put_zcopy`) + transport diag | `/work/ab0995/a270088/port2/m7/abenv_ucx_t1/` + `…/m7/abenv.26243303.out` |

- **hostprof:** the answer is in the sqlite (`hostprof.sqlite`, table `SAMPLING_CALLCHAINS`) — the
  `nsys stats --report cpu_profile` call in the job FAILS by design (report doesn't exist in this
  nsys; L81 note); ignore that error, query the sqlite. Precedent + query pattern: the SWSKIP
  sampling analysis, `docs/GPU_SPEED_M7.md` §"Why this took three attempts" (job 26237176). Subtract
  init: keep samples in the steady window only. Deliverable: name the 66.9 ms → ➕ tasks for
  anything ≥1% of step; anything forcing-phase feeds package D scoping.
- **UCX A/B:** result block at the end of `abenv.26243303.out` (min of 2 reps per leg, % vs ref).
  Rule: a winner needs **>2%** AND must then pass the CUDA fidelity gate under the same env before
  adoption (edit the exported env in ALL `jobs/job_m7_*` headers + README note). Nothing >2% →
  record + close Task A.3. The `UCX_LOG_LEVEL=info` diag at the end of the log says which
  transports/rndv scheme actually ran — record it in the ledger either way.

---

## 1. The work queue (plan tasks, in order)

1. Harvest 26243196 / 26243303 (→ tick items in Tasks A.2 / A.3).
2. **Implement Task A.1** exactly as §2 below; gates in §3.
3. Task A.4: package-A close (marginal A/B vs Tier-1, ledger row "Package A", tag `m7.2-packA`).
4. Package B (FCT2) — spec in the plan; biggest single lever (FCT = 181.8 ms/step).

---

## 2. Task A.1 spec — `FESOM_SPEED_FLAT` (fully designed; implement as written)

**One boolean knob `FESOM_SPEED_FLAT` covering four sites.** Lever class: re-parallelize
independent column-loop kernels to one-thread-per-(column,level). Every site is bit-identical by
argument below → the FORCE_SERIAL byte proof is claimed for the whole knob. Serial stays legacy
(the knob helper enforces it). Expected payoff (pool 52.6 ms at NG5@4N): **−4..5% of step**.

**Index identity used everywhere:** `FESOM_NODE3D(n,nz,nl) == (size_t)n*nl + nz == the flat index i`
(`fesom_types.h:33`). `FESOM_ELEMVEC(e,k,nl) = (size_t)e*nl*2 + k*2` (interleaved u/v, stride-2).

### 2.1 `src/fesom_speed.hpp` — add the no-master helper (needed for SYNCSTATS now, ICELAG later)

Add `fesom_speed_resolve_exp(lever)` + `fesom_speed_on_exp(lever, cache)`: identical to
`fesom_speed_resolve`/`fesom_speed_on` **except the `FESOM_SPEED` master fallthrough is skipped**
(per-lever env var only). Cleanest: refactor the body of `fesom_speed_resolve` into
`fesom_speed_resolve_impl(const char *lever, bool use_master)` and make both public functions
one-line wrappers; keep the Serial-stays-legacy guard and the rank-0 announce EXACTLY as they are
(the announce is L80 armor — do not weaken it).

### 2.2 `src/fesom_halo_device.cpp` — SYNCSTATS off the master switch

Line ~145: `return fesom_speed_on("SYNCSTATS", &c);` → `fesom_speed_on_exp`. Rationale (handoff
§4.4): a diagnostic must not ride `FESOM_SPEED=1`. Print-only change; byte gate unaffected.

### 2.3 `src/fesom_io.cpp` — `resolve_u_dev` (`:765`) / `resolve_v_dev` (`:776`)

Measured 10.9 ms each vs ~0.6 ms streaming floor (uncoalesced: adjacent threads nl·16 B apart).
Add one file-static helper near the resolvers so the TU announces once:
`static bool m7_io_flat_on() { static int c = -1; return fesom_speed_on("FLAT", &c); }`

Flat variant (u; v is `+ 1`):
```cpp
if (m7_io_flat_on()) {
    Kokkos::parallel_for("io_acc_u_flat", n, KOKKOS_LAMBDA(const size_t i){
        const size_t e = i / (size_t)nl, k = i % (size_t)nl;
        out(i) += uv(e * (size_t)nl * 2 + k * 2 + 0);
    });
    return;
}
/* legacy column loop unchanged below */
```
**Bit-id argument:** each slot `i` receives exactly one `+=` computed from the same operands —
execution order across slots cannot change any byte. (Serial FORCE_SERIAL iteration order is even
identical: i-ascending == e-major/k-minor.)

### 2.4 `src/fesom_kpp.cpp` — `kpp_ri_iwmix_kk` (`:343`; 17.0 ms, 23.6 sec/req, SM 2.8%)

`#include "fesom_speed.hpp"` (safe anywhere — it self-includes `<Kokkos_Macros.hpp>`, L80 fix).
Knob branch at the top of the function; flat path = **four launches replacing two**:

1. `kpp_ri_iwmix_Ri_flat` — `RangePolicy<size_t>(0, (size_t)Nmy*nl)`; derive `n = i/nl`,
   `nz = i%nl`; **guard `if (nz <= nzmin || nz >= nzmax) return;`** (legacy interior is
   `nz ∈ [nzmin+1, nzmax-1]`); body verbatim from the legacy loop-1 interior; write
   `diffK(0*slab + i)`.
2. `kpp_ri_iwmix_Ri_edge` — `RangePolicy<>(0, Nmy)`, per node the TWO edge-copy lines verbatim.
3. `kpp_ri_iwmix_shape_flat` — same flat pattern, legacy loop-2 interior verbatim
   (reads `diffK(0*slab+i)`, writes `viscA(i)`, `diffK(0*slab+i)`, `diffK(1*slab+i)`).
4. `kpp_ri_iwmix_shape_edge` — per node, the six edge-copy assignments verbatim.

**Keep the 1→2→3→4 launch order** (stream order provides the dependency; D20's "two launches are
load-bearing" becomes "four"). **Bit-id argument:** interior slots are written once each from
operands no other slot writes; edge copies run after the FULL interior kernel and therefore read
exactly the values the C per-column order would (including the degenerate ≤2-level-column
stale-carry case — the edge kernels reproduce it verbatim because the interior kernel never touched
those slots either).

### 2.5 `src/fesom_ale.cpp` — divide (`:456`) + wvel_split (`:518`); **do NOT touch scatter/cumsum**

`#include "fesom_speed.hpp"`; one file-static `m7_ale_flat_on()` helper (announce once).

- `fesom_ale_vvel_divide` (6.7 ms): flat over `RangePolicy<>(0, total)` (`total = N*nl` already
  exists, int is fine — 26 M at NG5@4N); guard `if (nz < nzmin || nz >= nzmax) return;`; body
  verbatim (`area(i)`, `w(i)`, `fer_w(i)`; note `FESOM_NODE3D(n,nz,nl)==i`).
- `fesom_ale_compute_wvel_split_kk` (7.2 ms): flat over `N*nl`; guard
  `if (nz < nzmin || nz > nzmax) return;` — **INCLUSIVE upper bound** (legacy loop is
  `nz <= nzmax`); body verbatim.
- **Do NOT touch** `fesom_ale_vvel_scatter` (29.5 ms — package B.3 store+gather) and
  `fesom_ale_vvel_cumsum` (7.9 ms — order-critical serial scan, package C.2). Their per-column
  order IS the bit-identity.

**Bit-id argument for both:** pure per-(n,nz) maps, every slot written once (divide: read-modify-
write of its own slot only).

### 2.6 Naming + docs

Flat kernels get `_flat`-suffixed Kokkos labels (profiler symbol comes from the enclosing function
either way). Add `FLAT` to the knob registry table in `docs/GPU_SPEED_M7.md` + README knob table
(the README update can ride the A.4 close commit).

---

## 3. A.1 gate ladder (exact commands; run from repo root)

Build first — **no queued job uses `build-m7cuda`/`build-m7serial` at handoff** (both in-flight
jobs pin the frozen Tier-1 binary), so rebuilding is race-free NOW; it stops being race-free the
moment the gates below are queued:
```
cmake --build build-m7cuda  -j 32   # CUDA+Serial
cmake --build build-m7serial -j 32  # Serial-only (byte gates)
```

| # | gate | command |
|---|---|---|
| 1 | knob-OFF byte gate | `sbatch jobs/job_m7_gate_serial` |
| 2 | FORCE_SERIAL byte proof | `sbatch --export=ALL,KNOBS="FESOM_SPEED_FORCE_SERIAL=1;FESOM_SPEED_FLAT=1" jobs/job_m7_gate_serial` |
| 3 | CUDA fidelity gate | `sbatch --export=ALL,KNOBS="FESOM_SPEED_FLAT=1" jobs/job_m7_gpu_gate` |
| 4 | A/B, NG5@4N | see below — **use `job_m7_ab_env`, not `job_m7_ab_gpu`** |
| 5 | A/B, dars@8N | same, `-N8 --ntasks=32`, `MESH=…/dars` |

**The marginal-A/B subtlety (this is why #4 uses the env job):** `FESOM_SPEED_FLAT` is a boolean
lever, so the master switch turns it ON. The number that matters is *FLAT's marginal gain on top of
Tier 1*, so leg A must be `FESOM_SPEED=1` **with FLAT explicitly overridden OFF** (per-lever knob
beats master — verified semantics of `fesom_speed_resolve`). `job_m7_ab_gpu`'s leg A hardcodes
all-knobs-cleared, which would measure FLAT+Tier1 vs nothing. So:
```
export MESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5 DT=180 TAG=abenv_flat_ng5_4n \
  LEG1='t1::FESOM_SPEED=1;FESOM_SPEED_FLAT=0' LEG2='flat::FESOM_SPEED=1'
sbatch -N4 --ntasks=16 jobs/job_m7_ab_env
```
(dars@8N: `MESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars TAG=abenv_flat_dars_8n`,
`sbatch -N8 --ntasks=32 …`.)

**Acceptance arithmetic (L80 armor — check BEFORE writing the ledger row):** the pool is 52.6 ms
of 913.5 → expect **−4..5%** at NG5@4N (resolvers ~−20 ms, iwmix ~−13, divide+split ~−11), less at
dars@8N (~−2..3%). The ON leg's log MUST contain `[fesom_speed] FESOM_SPEED_FLAT = ON` (announce =
knob fired). **A result near −0.0% is a DEAD KNOB, not a null lever** — a 52.6 ms pool cannot cost
nothing; check the announce line and the include-order story (L80) before any other conclusion. A
partial result (−1..2%) → re-run `jobs/job_m7_nsys` with `FESOM_SPEED=1` and compare the four
kernels' ms/step against the 26242512 trace to see which site under-delivered.

Commit rhythm (standing): implement → gates 1–3 green → A/B → ledger (`docs/GPU_SPEED_M7.md` lever
log + knob registry) → commit → then Task A.4 close (all-A marginal A/B, ledger row, tag
`m7.2-packA`).

---

## 4. Traps for this session (beyond the standing L80/L81)

1. **Rebuild race:** never rebuild while gates/A-Bs are PENDING — queued jobs run whatever binary
   is on disk at start time. Batch code changes, rebuild once, then submit.
2. **Guard direction typos are the whole risk here:** interior `[nzmin+1, nzmax)` for iwmix,
   `[nzmin, nzmax)` for divide, `[nzmin, nzmax]` (inclusive!) for wvel_split. The FORCE_SERIAL
   proof catches any slip — run it before believing anything else.
3. **`uv`/`uvnode` are interleaved** (`FESOM_ELEMVEC` stride-2): the flat resolvers/iwmix reads are
   ~50% sector-efficient by construction — that is fine and expected; do NOT "fix" the layout.
4. Announce lines: `m7_io_flat_on`-style one-per-TU helpers → expect exactly THREE
   `FESOM_SPEED_FLAT = ON` lines (io, kpp, ale) on rank 0.
5. The two in-flight diagnostics used `FESOM_SPEED=1` on the OLD (Tier-1) binary — FLAT does not
   exist there; no interaction with your new builds.
6. dt180, 35 steps, `snap_every=-1`, same-day same-alloc only; output under
   `/work/ab0995/a270088/port2/m7/` only.

## 5. After package A

Packages B–F are fully specified in the plan (RE-SCOPE section + Tasks B.1–F.1). B (FCT2, 181.8 ms
pool) is next and is the biggest lever; its FORCE_SERIAL condition is "T before S at every fused
site". ICELAG (F.1) is user-approved as an experiment but is LAST in the ladder — and it must use
`fesom_speed_on_exp` (never the master switch) and re-audit NOFENCE2 under multi-stream first.
The 2-SYPD Stage-2 goal falls out of packages B+C+E at 16N — measure it at E.5, not before.
