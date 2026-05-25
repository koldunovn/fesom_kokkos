# Porting FESOM2 → C: lessons & "don't overlook" checklist

Distilled from the dt=1800 blow-up hunt (root-caused 2026-05-23) and the string of
bugs before it. **Every item below actually bit this port.** The common thread: most
were *invisible at the dt=500 validation timestep* and only surfaced at the CORE2
production timestep (dt=1800). Read this before porting (or trusting) any term.

---

## 0. The meta-pattern — "used-but-mis-ported, no-op at small dt"

The single most expensive class of bug here. A term or parameter that **is used in the
CORE2 run** but was deferred, simplified, or given a wrong value in the C — and whose
effect is *negligible at small dt*, so it sails through dt=500 validation and only blows
up at dt=1800. It has struck **6+ times**:

| # | term | how it was wrong | symptom at dt=1800 |
|---|------|------------------|--------------------|
| 1 | momentum advection | entirely missing ("not in first slice") | earlier blow |
| 2 | `opt_visc` | ported 5 (harmonic), CORE2 uses 7 (biharmonic) | 2Δx grows |
| 3 | `use_wsplit` | hardcoded `.true.` (from work_pi), CORE2 = `.false.` | day-92 blow |
| 4 | Ch/Ce bulk coeffs | looped `myDim`, Fortran loops `myDim+eDim` | non-conservative SSH |
| 5 | wind-on-ice stress | allocated + read but **never computed** (≡0) | ice felt no wind |
| 6 | **AB2 `epsilon`** | **hardcoded `1e-9`, Fortran `o_PARAM` = `0.1`** | **day-110 central-Arctic blow** |

> **Rule:** validate at the **target configuration** (CORE2, dt=1800) to a meaningful
> length (≥1–2 model years), not just dt=500 smoke tests. "Bit-identical at dt=500" says
> *nothing* about stability at dt=1800. Anything that scales with `dt`, `CFL`, or `f·dt`
> is dormant at small dt.

---

## 1. Module-default scalar parameters carry physics — copy the *value*, verify the *role*

The dt=1800 root cause: FESOM's AB2 time-stepping uses a **stabilization offset**
`epsilon = 0.1` (`oce_modules.F90:92`, `o_PARAM`, commented "AB2 offset"), applied as
`ab1=-(0.5+ε)`, `ab2=(1.5+ε)` in **both** momentum (`oce_ale_vel_rhs.F90:98`) and tracer
(`oce_tracer_mod.F90:53`) AB2. The C hardcoded `eps = 1.0e-9` — assuming it was a tiny
"first-step guard" epsilon. It is not: with ε≈0 the scheme is *pure* AB2, which is
marginally **unstable for the oscillatory Coriolis term**; the offset is what stabilizes
it. Worst where `f·dt` is largest → the central Arctic at dt=1800.

> **Rule:** every scalar default in `oce_modules.F90` / `MOD_*` / `o_PARAM` is a real
> numerics/physics parameter until proven otherwise. **Read the actual value and its
> use-site** — never assume a small number is a guard epsilon, and never paraphrase a
> value "by convention." Other dangerous innocuous-looking ones already verified here:
> `visc_gamma0=0.003` (NOT 0.03), `instabmix_kv=0.1`, `wsplit_maxcfl=1.0`, `gamma_*`.

## 2. A wrong/assumed comment is worse than no comment

The C line read `eps = 1.0e-9; /* matches Fortran's epsilon (=1e-9 by convention) */` —
confidently wrong, and it **blocked discovery for multiple sessions** (readers trusted it
and moved on).

> **Rule:** when you port a constant, cite the **exact Fortran `file:line` you read it
> from and the literal value**, so the next reader can verify in one grep. If you didn't
> read it, don't claim what Fortran does.

## 3. Match the RUN-DIRECTORY namelists, not the `config/` template

`use_wsplit` and `opt_visc` bugs both came from porting the template/`work_pi` defaults
instead of the values the **CORE2 run dirs** (`work_core`, `work_linfs_d1800`) actually
set. The template often disagrees with the run config.

> **Rule:** the source of truth for any namelist value is the **run directory you're
> reproducing**, not `config/namelist.*`. Diff them.

## 4. Halo loop bounds & array sizing (cell-vertex grid)

Recurring: a Fortran loop runs `1, myDim+eDim` (owned **+ halo**) for any field **read
downstream**; the matching C loop written over `myDim` only leaves stale halo entries →
silent per-rank divergence (Ch/Ce bug → CG NaN after ~85 steps). Likewise an array sized
`myDim` only, then read at the halo by a downstream consumer, is a silent OOB read.

> **Rule:** for every array, ask "who reads this, and do they loop into the halo?" If yes,
> the producing loop must cover `myDim+eDim` **and** the array must be allocated
> `myDim+eDim(+eXDim)`. An "exchange-and-compare" probe (exchange a field, diff against
> pre-exchange) finds stale-halo bugs fast.

## 5. Storage strides

3D node/element fields and tracers are stored with stride **`nl`** (`[N*nl]`), even though
only `nl-1` layers are used. Using `nl-1` silently shifts every read by one level and can
inflate derived fields ~1000×.

> **Rule:** confirm the stride at the `allocate` site (`calloc(N*nl, …)`), and index with
> the shared `FESOM_NODE3D/ELEM3D(n, nz, nl)` macros — never `nz*(nl-1)`.

## 6. "Allocated + exchanged" ≠ "populated"

`stress_atmice_x/y` was allocated and read by the EVP but **never computed** (identically
zero) — the sea ice felt no wind. Such gaps hide as "bit-identical to the baseline"
because the baseline never populated the field either.

> **Rule:** a field existing and being halo-exchanged does **not** mean a producer fills
> it. When a new consumer is added, verify its inputs are actually non-zero/computed.

## 7. The symptom location is not the bug location

The dt=1800 blow-up is a 2Δx velocity checkerboard, which on the cell-vertex grid is
damped **only by the horizontal viscosity** — so everyone (including the FESOM developer)
suspected the biharmonic. Sessions were spent proving the operator byte-identical, hitting
the paradox "identical operator, yet less stable." The viscosity was the **damper**; the
**source** was the AB2-Coriolis instability. A correct damper that's slightly out-run
still blows up.

> **Rule:** when a mode is damped by operator X and grows anyway, X being correct does
> **not** clear it — find what *feeds* the mode. The decisive move that broke the
> fixation: **empirically prove the suspect operator is bit-identical**, which forces the
> search upstream (see technique A).

---

## Debugging techniques that worked (build these when stuck)

**A. Identical-input operator diff (highest leverage).** Prescribe the *same* input field
to the C and Fortran kernel via a **deterministic integer hash of the global id**
(bit-reproducible across languages), run **one** call, dump the output by gid, diff.
Cleanly separates "operator bug" from "upstream-state difference" — something neither
line-by-line reading nor full-run comparison can do (they diverge chaotically). This is
how `visc_filt_bidiff` was exonerated to ~1e-12 (244 659/244 659 elements). Run it at
np=1 or set halos consistently so the exchange is a no-op and only the per-element
arithmetic is tested. (`FESOM_BIDIFF_PROBE`, `scripts/exp1_compare_bidiff.py`.)

**B. Per-step, per-node "what leads" probe.** At the eruption node, dump the 1-ring spread
of every candidate field each step (velocity 2Δx, PGF, density, w, ssh_rhs). Seeing
`velsprd` grow while `pgf`/`density` stayed flat proved it was a **free velocity mode, not
buoyancy-forced** — which redirected the whole hunt. (`FESOM_DIAG_SPREAD` in `fesom_step.c`.)

**C. Isolate terms with env flags.** `FESOM_NO_VADV`, `FESOM_NO_ICE_DYN`, `FESOM_VISC_MULT`
let you turn one term off/scale it and watch the blow-step move. Caveat: removing a term
that is *net-stabilizing* (vertical advection, horizontal advection) makes it blow
*earlier* — don't misread "earlier blow" as "this term was the cause."

**D. dt sweep.** Blow-day vs dt is diagnostic: a clean CFL cliff vs a marginal mode that
only *delays* with smaller dt tells you whether you're at an explicit-stability limit.
dt is a runtime arg here (`FESOM_PHASE1_DT` ← argv[3]) — no recompile.

**E. Always run PAST the known blow step.** Never declare a fix from a run that stops at
the old blow point — run ≥1–2 model years. (A "clean 5000-step" run hid a day-110 blow.)

---

## One-line takeaway

In FESOM, the dangerous bugs are not the big missing routines (those announce themselves)
— they are **small numeric parameters and loop bounds whose effect is zero at dt=500 and
decisive at dt=1800**. Read the run-dir namelists and the module defaults literally,
validate at the production timestep for ≥1–2 years, and when an operator looks guilty,
prove it bit-identical before chasing it.
