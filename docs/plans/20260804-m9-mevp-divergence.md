# M9 — Divergence-subcycled mEVP (S. Danilov): build + controlled-study plan

*Worktree `/home/a/a270088/port_kokkos_ice`, branch `m9-mevp-double` (from `35867ac`).*
*🔴 Do NOT touch `/home/a/a270088/port_kokkos` — the wsplit session works there. Everything below
happens in this worktree.*

> **REV 2 (2026-08-04, after plan review).** Three critical and eight important findings were
> verified against source and folded in. The changes that alter the *study*, not just the code:
> 1. **The L2 selfcheck as first written could not pass** — its own non-equivalence (ii) guarantees
>    `R ≠ ∇·σ` at every ocean-step boundary. It is now **intra-step and relative**, with the
>    step-entry jump reported separately as the mask-flip instrument (§L2 below).
> 2. **② − ④ is NOT "compute identical"** — ②'s ghost element kernel does the σ read-modify-write
>    that ④'s does not, so the delta carries a ghost-zone Win-A term. Decomposition stated in §H3.
> 3. **②'s two-messages-per-partner is an implementation choice, not a property of the classic
>    form** — both segments already loop the same `S.partner` vector
>    (`fesom_ice_evpwide.cpp:828` vs `:842`), so they can be concatenated into one send. The
>    intrinsic advantage of the divergence form is the **2× byte ratio**, not the message count.
>    ⚠️ **OPEN DECISION D1** (§Open Decisions) — this is a paper-facing claim.
> 4. **Four arrays the ghost node solve reads have no ghost storage at any ring**, three of them
>    sized `myDim` (not even the ordinary `eDim` halo). Task 12 now opens with a storage audit.

---

## Overview

Sergey Danilov's note (`ice_sergey/evp_div.tex`, `ice_sergey/ice_mevp_double.F90`) applies the
divergence operator to the mEVP stress recursion. `∇·` is linear and the discrete divergence
operator (element geometry → node assembly) is time-invariant, so

```
σ^{p+1} = det1·σ^p + det2·σ̃(ε^p)        ⟹        d^{p+1} = det1·d^p + det2·∇·σ̃(ε^p)
```

exactly, with `d = ∇·σ`. Carried state moves from **3 element arrays** (`σ11/σ12/σ22`) to **2 node
arrays** (`R_u/R_v`). Two separable consequences:

- **Win A — no communication change at all.** The element kernel stops loading and storing three
  element arrays every subcycle (×120/step). Pure bandwidth and footprint. `R` is read only by the
  node solve over owned nodes, so at K=1 it is never exchanged.
- **Win B — fewer bytes on the wire.** Every subcycled quantity becomes a node quantity, so
  `u, v, R_u, R_v` ship as one node segment (4·Ng doubles) where the classic form must also ship
  σ on ghost elements (8·Ng doubles total).

Possible **only for mEVP**, where all σ components share one α. Standard EVP subcycles them
differently and is excluded by construction.

Set up as a **controlled study aimed at a small paper**: primary metric is the `icedyn` phase
time, SYPD always reported beside it, every cell on CPU and GPU across a range of per-rank
workloads.

### The 2 × 3 cell matrix

| | every subcycle (K=1) | **lagged** single halo (K) | **exact wide** halo (K) |
|---|---|---|---|
| **classic** σ (elements) | ① today's mEVP — baseline | ⑤a | ② u,v + σ on ghost elements |
| **divergence** d (nodes) | ③ Win A isolated | ⑤b | ④ Sergey's proposal |

- ③ vs ① isolates Win A with communication held *identical*.
- ② vs ④ isolates the shipped state. **Not** pure communication — see §H3.
- ⑤ vs ④ at equal K asks whether the exact machinery is needed, or whether a stale halo is
  absorbed by the relaxation.

---

## Context (from discovery)

**Verified facts** (checked against source; line numbers are current)

| fact | where |
|---|---|
| `pfac` carries `det2`; `pressure = pfac/(delta+delta_min)` | `fesom_ice_maevp.cpp:213`, `:251` |
| `alpha_evp = 250` ⇒ `det1 = 250/251`, `1/(1−det1) = 251`; `evp_rheol_steps = 120` | `fesom_ice.cpp:123`, `:93` |
| `eps11/12/22` written in mEVP, **read only by std EVP**, not snapshot variables | write `fesom_ice_maevp.cpp:243`; reads `fesom_ice_evp.cpp:117-129,698,737`; `fesom_io.cpp:547-560` |
| element mask `mean(m_ice) > 0.01` vs node mask `a_ice >= 0.01` — **different criteria**, so σ evolves on elements whose vertices are not ice nodes | `:210` vs `:185` |
| assembly guarded to owned nodes; solve runs `[0,myDim)` ⇒ R never needed on halo at K=1 | `:265`, `:280` |
| framework ring depth is already `R = K` | `fesom_ice_evpwide.cpp:129`, announce `:749` |
| `icedyn` phasestats bucket exists | `fesom_phasestats.cpp:79` |
| prestep echo "MUST be 0.000e+00" | `fesom_ice_evpwide.cpp:1032-1051` |

**⚠️ Four traps the review found, all confirmed**

1. **`fesom_speed_on_exp` DOES announce** (`fesom_speed.hpp:117-122`, plus a shout on
   requested-but-off). Only `fesom_speed_int` is silent. So `MEVPDIV`/`MEVPNOEPS` announce for
   free; **only `MEVPLAG` needs a hand-written announce**. `FESOM_SPEED_EVPWIDE` already announces
   (`fesom_ice_evpwide.cpp:747-755`) and already carries `FESOM_CHECK(evp_rheol_steps % K == 0)`
   (`:738`) and the `FESOM_KK_VERIFY=evp` abort (`:740-745`) — **do not duplicate any of them**.
2. **Both helpers force OFF on Serial without `FESOM_SPEED_FORCE_SERIAL=1`**
   (`fesom_speed.hpp:111-113`, `:168-170`). A Serial null rung that forgets it is byte-identical
   *because the knob is dead* — the L80 class. Every Serial gate line below requires
   `FORCE_SERIAL=1` **and** an announce grep.
3. **Ghost storage does not exist** for four arrays the node solve reads —
   `mevp_inv_thickness`, `mevp_mass`, `mevp_ice_nod` are **`myDim`-sized**
   (`fesom_ice.cpp:211-213`), `mevp_pressure_fac`/`mevp_ice_el` are `myDim_elem2D`-sized
   (`:214-215`), and `bc_index_nod2D` is `n`-sized with no tail (`:286`) while multiplying the
   determinant (trap 9, `fesom_ice_maevp.cpp:304`). `FesomEvpwideDev` supplies only
   `area0x/corx/coastx` (`fesom_ice_evpwide.h:52-54`) because std EVP never needed the rest.
4. **Two EVPWIDE refusal sites, and the resolve is unreachable from mEVP.**
   `fesom_ice.cpp:752-766` prints its own "lever is NOT running" inside the mEVP branch, and
   `fesom_evpwide_K()` — which runs the guards, the collective `evpw_build` and the announce — is
   called **only** from `fesom_ice_evp.cpp:611`.

**Files involved**

| file | role |
|---|---|
| `src/fesom_ice_maevp.cpp` | the mEVP kernel — cells ①③⑤ plus the ghost twin bodies for ②④ |
| `src/fesom_ice_maevp.h` | the trap list (numbered to 13, 12 entries — no #11) |
| `src/fesom_ice_evpwide.{h,cpp}` | wide-halo framework; header still documents the superseded `R=2K−1` |
| `src/fesom_ice.cpp` | mEVP allocs `:203-221`; **σ rails `:737-751` (H2D) and `:768-771` (D2H)**; IC push `:592-594`; refusal `:752-766`; phasestats mark `:676` |
| `src/fesom_ice_types.h` | `fesom_ice_work` — where `R_u/R_v` go |
| `src/fesom_speed.hpp` | knob helpers (announce semantics above) |

**Patterns reused**: `jobs/job_m6_mevp_serial_bitid` (the byte reference for exactly our scheme:
Kokkos-Serial `FESOM_WHICH_EVP=1` vs the C oracle `build-m6oracle`),
`jobs/job_m6_mevp_gpu_gate` (the natural base for `job_m9_gpu_gate`, Serial reference at
`/work/.../m6/mevp_bitid/kk_mevp`), `jobs/job_m6_all3_*` (options ×3),
`jobs/job_m7_ab_env` (LEG1..4 same-allocation A/B), `scripts/diff_snap.py`,
`scripts/m7_phasestats_join.py`, `scripts/m7_scaling_figs.py`.

**⚠️ Job-script hazard**: every `jobs/job_*` hardcodes `ROOT=/home/a/a270088/port_kokkos` (e.g.
`job_m6_mevp_serial_bitid:28`, `job_m6_mevp_gpu_gate:30`). Run unmodified from this worktree they
pull the *other* session's tree.

**Build deviation (done, 2026-08-04)**: `externals/kokkos` was uninitialised here and `.git/modules`
is shared with the other worktree (its config has `worktree = ../../../../externals/kokkos`
pointing at *their* tree), so `git submodule update --init` would likely have repointed shared git
state. Instead the pinned 4.4.01 source was **copied** (`rsync -a --exclude=.git`), verified
byte-identical, 1456 files. No git state touched; `git status` stays clean.

---

## Development Approach

- **Gate-first, not TDD.** No unit-test framework exists; the equivalent is the SLURM **gate
  ladder**. Each task ends with the gate that must pass before the next starts.
- **The knob-off byte gate (L0) runs on every build** — the merge-back guarantee.
- 🔴 **ALWAYS PIN `BIN=`**; frozen bins to `/work`, sha256 in docs, **never committed**.
- 🔴 SLURM output to `/work`; a cheap gate must *look* cheap (`-t 00:06:00` flips `(Priority)`).
- 🔴 Private CORE2 mesh `/work/ab0995/a270088/port2/mesh/core2` for gates (L73).
- 🔴 **L86: no Serial gate can validate a coherence invariant** — and the precedent is
  *specifically mEVP* (`fesom_ice.cpp:548-551`, `:572-580`, and the `ICERAILS`+`WHICH_EVP=1`
  clobber at `:722-736` that every Serial gate blessed). Any task touching a cross-step rail gets
  a CUDA gate in the same task.
- Rebase onto `main` (3 commits ahead: the wsplit close) before the first code commit.

## Testing Strategy — the gate ladder

| rung | what | when |
|---|---|---|
| **L0** knob-off byte | Serial, `FESOM_WHICH_EVP=1`, all M9 knobs unset, vs the C oracle — every field bit-identical | every build |
| **L1** null rungs | `MEVPLAG=1` and `EVPWIDE=1`-under-mEVP byte-for-byte vs today; `MEVPNOEPS=1` **exactly zero** diff. **All with `FESOM_SPEED_FORCE_SERIAL=1` + an announce grep**, else the rung is vacuous | each knob's task |
| **L2** selfcheck | **REDEFINED — see below** | every DIV/wide task |
| **L3** trajectory | 300-step CORE2, each cell vs baseline, max + RMS on `a_ice, m_ice, u_ice, v_ice` vs the mEVP scheme floor (`docs/REFERENCE_RUNS.md`, L79). **Most cells rest here** | every cell |
| **L4** CUDA-vs-Serial | every new path; **not deferred** — attaches to the task that creates the exposure | Tasks 5, 13 |
| **L5** options ×3 | each knob is a field-ownership change (L91) | Tasks 8, 13 |
| **L6** climate | 1 yr at the M5.23 bar + hemispheric ice area/extent, drift RMS. **A few cells, decided later, probably farc** | later |

### L2 redefined (review Critical #1)

The naive check `max|R − ∇·σ| ≈ 0 for the whole run` **cannot pass**, and its own Technical
Details say why: the masks are recomputed at every ocean step (`fesom_ice_maevp.cpp:203-215`,
outside the subcycle loop) and the classic assembly is restricted to `ice_el` (`:228`), so `∇·σ`
drops a departing element's contribution instantly while `R` retains it decaying at `det1`. Within
a step the masks are **fixed**, so the two forms are algebraically identical there. Therefore:

- **`FESOM_MEVPDIV_SELFCHECK=1` runs per ocean step**: (a) compute `∇·σ` from the carried σ at step
  entry; (b) **report `|R − ∇·σ| / max|∇·σ|` as the mask-flip instrument** — this is the
  non-equivalence, measured, not a failure; (c) re-baseline `R := ∇·σ`; (d) run the step carrying
  both, and gate the **intra-step** relative drift at roundoff.
- **Criterion is relative.** `|R − ∇·σ|` is dimensional (stress divergence × area); an absolute
  `1e-12` is not a criterion. Gate: intra-step `max|R−∇·σ| / max|∇·σ| ≤ 1e-13` on Serial.
- **Step 1 from cold start is the one place absolute zero is legitimate** (σ and R both start 0).
- SELFCHECK=1 perturbs the trajectory (the re-baseline removes non-equivalence (ii)) ⇒ it is a
  **diagnostic mode only**, never a timing or L3 leg.

---

## Open Decisions

**D1 — ✅ DECIDED (user, 2026-08-04): option (c), build BOTH.** Cell ② ships in a fused and an
unfused form behind one flag, giving three data points at each K. The paper can then separate the
**intrinsic** advantage of the divergence form (the 2× byte ratio) from the **convenience**
advantage (the message count that Fortran's per-field-type exchange routines would actually cost),
instead of conflating them. Task 13 and H3 are already written for this.

*Original framing, kept for the record — how is cell ② allowed to send? (paper-facing)*
Both segments already loop the same `S.partner` vector, so the classic form *can* concatenate node
and element data into one buffer and one `Isend` per partner. Options:
- **(a) build ② fused (fair control) and report bytes-only.** ② and ④ then both send 1
  message/partner and the result is the intrinsic 2× byte ratio. Most defensible; costs a fused
  pack path.
- **(b) build ② unfused (status-quo control).** Matches what FESOM-Fortran's per-field-type
  exchange routines would naturally do, and matches Sergey's "cannot be conveniently combined".
  Keeps the message-count term, but a referee will call it an implementation artifact.
- **(c) build both** — ②-fused and ②-unfused are the same code with one flag; three data points,
  and the paper can say exactly how much of the win is intrinsic and how much is convenience.
**Recommendation: (c).** It is nearly free once the fused path exists and it is the honest answer
to Sergey's sentence. **H3 below is written for (c).**

---

## Implementation Steps

### Stage 0 — bring-up

### Task 1: Build both backends and freeze baselines ✅ DONE (2026-08-04)

**Files:** `build-serial/`, `build-m9cuda/` (build trees; not committed)

- [x] copy pinned Kokkos 4.4.01 into `externals/kokkos` (see Build deviation above) — byte-identical, 1456 files
- [x] CUDA build: `build-m9cuda/fesom_port`, `Device Parallel: Kokkos::Cuda`, `AMPERE80`, CUDA 12.5, CUDA-aware `openmpi-4.1.5-nvhpc`
- [x] Serial build: `build-serial/fesom_port`, spack `openmpi-4.1.2`
- [x] freeze to `/work/ab0995/a270088/port2/m9/bin/base/` — **Serial `1e69d43f…`, CUDA `ac9a1428…`** (full sha256 in `SHA256` beside them)
- [x] L0 prerequisites verified present: C oracle `build-m6oracle/fesom_port`, private CORE2 + `dist_8`, PHC forcing
- [ ] ⚠️ **the L0 gate itself moves to Task 3** — it needs the Task-2 scaffolding (review Important #10: Tasks 1 and 2 were mutually blocking)

### Task 2: Worktree-rooted M9 job scaffolding

**Files:**
- Create: `jobs/job_m9_mevp_bitid`, `jobs/job_m9_gpu_gate`, `jobs/job_m9_options3`, `jobs/job_m9_ab_cpu`, `jobs/job_m9_ab_gpu`

- [ ] all default `ROOT=${ROOT:-/home/a/a270088/port_kokkos_ice}` and **require** `BIN=`; outputs under `/work/ab0995/a270088/port2/m9/`
- [ ] `job_m9_mevp_bitid` from `job_m6_mevp_serial_bitid`; `job_m9_gpu_gate` from `job_m6_mevp_gpu_gate`; `job_m9_options3` from `job_m6_all3_*` (L5 is required by Tasks 8 and 13 and had no job)
- [ ] `KNOBS=` passthrough so every cell is selectable without editing the script; A/B jobs keep `--constraint=a100_80` (L94), subshell per leg, 2 reps, min, leg 1 = reference
- [ ] **gate**: each job runs against the frozen base bins and completes rc=0
- [ ] gate must pass before Task 3

### Task 3: L0 baseline gate and the study's zero point

**Files:** Modify: this plan (record job IDs + numbers)

- [ ] **gate L0**: `job_m9_mevp_bitid` on `1e69d43f` — `diff_snap.py` all fields bit-identical vs the C oracle
- [ ] record baseline `icedyn` (instrumented leg) and SYPD (clean leg) for CORE2 np8
- [ ] ⚠️ these are timing runs but **not fleet jobs** — the pre-registration (Task 9) gates the *fleet*, not the baseline (review Important #10B)
- [ ] gate must pass before Task 4

### Stage 1 — cells ③ and ⑤ (no framework needed)

### Task 4: Knob plumbing — only what is genuinely missing

**Files:** Modify: `src/fesom_ice_maevp.cpp`, `src/fesom_ice_maevp.h`

- [ ] `FESOM_SPEED_MEVPDIV`, `FESOM_SPEED_MEVPNOEPS` via `fesom_speed_on_exp` — **no hand-written announce**, the helper prints `[fesom_speed] X = ON` and shouts on requested-but-off
- [ ] `FESOM_SPEED_MEVPLAG=K` via `fesom_speed_int` — **this one needs** an explicit rank-0 announce and its own `FESOM_CHECK(evp_rheol_steps % K == 0)`
- [ ] `FESOM_MEVPDIV_SELFCHECK` (plain getenv, diagnostic)
- [ ] **do not** re-resolve `FESOM_SPEED_EVPWIDE` here — it already announces and guards; a second source of truth would bypass the lazy build (review Important #4)
- [ ] **gate L0** knob-off byte; **gate**: each announce appears exactly once on rank 0 with the right value; each guard aborts on a deliberately bad input
- [ ] gates must pass before Task 5

### Task 5: `R_u`/`R_v` fields, the rails decision, and cell ③

**Files:**
- Modify: `src/fesom_ice_types.h` (`mevp_Ru_fld`, `mevp_Rv_fld`, + the selfcheck reference pair)
- Modify: `src/fesom_ice.cpp` (alloc inside the existing `if (whichEVP == 1)` block `:203-221`; **σ rails `:737-751`/`:768-771`**)
- Modify: `src/fesom_ice_maevp.cpp`

- [ ] allocate `R_u/R_v` node-sized, zero-init, **inside the existing `whichEVP == 1` block** (matching the file's convention rather than allocating unconditionally)
- [ ] 🔴 **RAILS (review Important #7).** `fesom_phasestats_mark(FESOM_PH_ICE_DYN)` is at `:676`, *before* the branch, so the σ H2D/D2H rails are billed to `icedyn` — the primary metric. Decision: **R gets no host rail at all** (no host code reads it; it is device-authoritative across steps by construction) **and under `MEVPDIV` the σ rails are skipped** (σ is dead in that form). Verified safe: σ is not a snapshot variable and the host copies exist only to feed the next step's IN rail. Additionally **every timing leg runs `FESOM_SPEED=1`** (⇒ `ICERAILS` on, `fesom_ice.cpp:559`, so no rails exist at all) — stated in the protocol so the bias cannot reappear
- [ ] element kernel: under `MEVPDIV` drop the `det1*s{11,12,22}(el)` terms and the three stores; everything through `delta`/`pressure` untouched; assembly and its owned-node guard (trap 6) untouched; launch count unchanged
- [ ] node kernel: `R_u(i) = det1*R_u(i) + u_rhs(i)`, then the solve reads `R_u(i)*mass(i) + rhs_a(i)`
- [ ] 🔴 **THE TRAP**: the R update runs **unconditionally over owned nodes, OUTSIDE the `ice_nod` guard at `:283`**. The element mask (`mean(m_ice) > 0.01`, `:210`) and the node mask (`a_ice >= 0.01`, `:185`) are *different criteria*, so σ genuinely evolves on elements whose vertices are not ice nodes. Only the velocity solve stays masked
- [ ] store R **unscaled** (mass applied at use) — deliberately unlike Sergey's F90; **record in PREREG and RESULTS**, since the paper claims to implement his proposal
- [ ] implement the redefined L2 selfcheck (§L2): needs its own reference node-array pair so it cannot corrupt the DIV path's `u_rhs`; if it adds a file-scope `Kokkos::View`, release it in `fesom_ice_maevp_free()` (the exit-134 trap, `fesom_ice_maevp.cpp:47-52`)
- [ ] mask-flip instrument: per step, count `ice_el` membership changes **in both directions** — leaving (contribution decays over ~251 subcycles instead of dropping) *and* re-entering (classic resumes from frozen stale σ per trap 12; R has decayed) — plus the induced `max|Δu_ice|`
- [ ] **gate L2**: intra-step relative drift ≤ 1e-13 on Serial; step 1 from cold exactly 0; mask-flip jump reported
- [ ] **gate L0** knob-off byte
- [ ] **gate L4 (not deferred)**: CUDA-vs-Serial smoke at `MEVPDIV=1` — this task changes a cross-step device rail and **L86 says no Serial gate can validate a coherence invariant**, with mEVP as the precedent
- [ ] **gate L3**: 300-step CORE2 vs baseline
- [ ] gates must pass before Task 6

### Task 6: Cell ⑤ — `FESOM_SPEED_MEVPLAG=K`

**Files:** Modify: `src/fesom_ice_maevp.cpp`

- [ ] condition the existing `fesom_halo_field2` (`:322`) on `sub % K == 0`
- [ ] the `steps % K == 0` guard makes the **last** subcycle always exchange — otherwise the final copy over `myDim+eDim` (`:330-334`) publishes a stale halo into the ocean coupling
- [ ] verify composition with `MEVPDIV` both ways (⑤a, ⑤b)
- [ ] **gate L1**: `MEVPLAG=1` byte-identical **with `FESOM_SPEED_FORCE_SERIAL=1` and an announce grep** (without both, the rung is vacuous — the knob is simply dead on Serial)
- [ ] **gate L3**: K ∈ {2,4,8} × both forms; a measurable difference here is a *result*
- [ ] gates must pass before Task 7

### Task 7: `FESOM_SPEED_MEVPNOEPS=1` — the confound control

**Files:** Modify: `src/fesom_ice_maevp.cpp`

- [ ] skip the three `eps` stores (`:243`) under the knob; keep them by default in **both** forms
- [ ] **gate L1 (exact)**: `MEVPNOEPS=1` gives an **exactly zero** diff — the arrays are written and never read in this path and are not snapshot variables, so this is genuinely runnable through `diff_snap.py`. A nonzero diff falsifies the premise and must be chased
- [ ] **gate L0** knob-off byte
- [ ] gates must pass before Task 8

### Task 8: Stage-1 certification and freeze

- [ ] **gate L5**: `job_m9_options3` (TKE / mEVP / zstar) with the knobs set
- [ ] **gate L2 on CUDA**: report the relative magnitudes
- [ ] rebuild BOTH backends after the LAST edit, THEN freeze (R5 stale-freeze trap) to `/work/.../m9/bin/s1/`; sha256 recorded here
- [ ] gates must pass before Task 9

### Stage 2 — measure cells ① ③ ⑤

### Task 9: Pre-registration, committed before the first **fleet** job

**Files:** Create: `docs/plans/20260804-m9-PREREG.md`

- [ ] H1–H4 with numeric ranges and decision rules (§Technical Details), **including the H3 decomposition and D1's resolution**
- [ ] protocol: `icedyn` from matched **instrumented** legs, SYPD from **clean** legs, never the same run; **every timing leg runs `FESOM_SPEED=1`** (rails off); min-of-2; matched pairs in one allocation; same day
- [ ] record the unscaled-R deviation from Sergey's F90
- [ ] floors: 0 for every win
- [ ] gate: committed, git timestamp preceding the first fleet job

### Task 10: Fleet 1A/1B — staged, not a full cross

**Files:** Create: `scripts/m9_collect.py`; Modify: this plan

- [ ] ⚠️ **staged** (review Important #11 — the full cross is several hundred runs against an H4 expectation of ~2 % at model level): **first** the complete 2×3 matrix at ONE reference point (the F1 table), CPU and GPU; **then** sweep per-rank workload only for the cells that moved. F2 needs only the cells with a slope, so this costs nothing scientifically
- [ ] 1A: ① vs ③ vs ③+NOEPS (communications untouched)
- [ ] 1B: ⑤a, ⑤b at K ∈ {2,4,8}
- [ ] sweep by node count at fixed mesh (2D-verts/rank), core2 → farc → dars → ng5
- [ ] mechanism panel from `fesom_halo_prof_*`: messages/step, bytes/partner
- [ ] gate: every leg rc=0, announce present in each, L3 diffs recorded per leg

### Stage 3 — the framework and cells ② ④

### Task 11: Generalize `evpw_exchange` to a caller-supplied field list

**Files:** Modify: `src/fesom_ice_evpwide.{h,cpp}`

- [ ] replace the `nf`-indexed table with a caller-supplied list + `ship_sigma`; std EVP passes its existing lists ⇒ unchanged by construction
- [ ] ⚠️ **buffers are sized `nsend*11` / `nrecv*11`** (`:672-673`) — a caller-supplied list is capped at 11 fields unless the sizing is made dynamic. Cell ④ needs only 4, so this is a guard, not a blocker
- [ ] fix the stale `R = 2K−1` contract text in `fesom_ice_evpwide.h:8-11` (code is `R = K`)
- [ ] **gate**: the std-EVP FORCE_SERIAL byte proof at K ∈ {1,2,4} plus knob-off
- [ ] 🔴 **gate (dead-diagnostic, review Important #9)**: the byte proof passes just as happily with every diagnostic dead, because they are all keyed on `nf` **values** — `sig = (nf<=6)` `:796`, selfcheck `:868`, `nf==6` `:884`/`:915`, `nf==2` `:901`, echo `nf==11` `:1036`. Require: the K=2 leg still prints the echo as `0.000e+00` **and** the drift diagnostic still appears
- [ ] gates must pass before Task 12

### Task 12: `evpw_build` mEVP mode — opening with the storage audit

**Files:** Modify: `src/fesom_ice_evpwide.{h,cpp}`, `src/fesom_ice.cpp`

- [ ] 🔴 **FIRST: audit every array the mEVP element kernel and node solve index**, and for each decide *extended tail* / `D.*_g` ghost array / *owner-shipped bytes*. Known gaps (review Critical #2), none of which std EVP ever needed:
  - `mevp_inv_thickness`, `mevp_mass`, `mevp_ice_nod` — **`myDim`-sized** (`fesom_ice.cpp:211-213`), no storage even for the ordinary `eDim` halo; read at `fesom_ice_maevp.cpp:285-304`
  - `mevp_pressure_fac`, `mevp_ice_el` — `myDim_elem2D`-sized (`:214-215`); the ghost element kernel needs them for `Eg` ghost elements
  - `bc_index_nod2D` — `n`-sized, no tail (`fesom_ice.cpp:286`), multiplies the determinant (`:304`, trap 9); **ship as owner bytes** per correction 4 of the design doc (recompute nothing the owner computes)
  - `uice_aux`, `vice_aux` — `n`-sized (`:209-210`); need the `fesom_evpwide_next()` tail
  - `R_u`, `R_v` — need the tail
  - **out-of-bounds ghost reads are silent garbage on CUDA, not a crash** ⇒ this audit is the gate, not the build-summary print
- [ ] mEVP mode: ghost `pfac`/`ice_el` in place of std EVP's ghost `istr`; keep ghost `s11/s12/s22` for ②; ghost `R_u/R_v` for ④
- [ ] 🔴 **lift BOTH refusals and wire the resolve** (review Critical #3): `fesom_ice_evpwide.cpp:719` **and** `fesom_ice.cpp:752-766`, **and add the `fesom_evpwide_K()` call to the mEVP path** — it is currently called only from `fesom_ice_evp.cpp:611`, so without it the wide zone never builds and the model prints a message contradicting the announce
- [ ] **gate**: build summary prints rings/partners/bytes; `npes==1` and knob-off announced no-ops; a deliberate 1-rank-past-the-end read is caught by a bounds-checked debug build
- [ ] gates must pass before Task 13

### Task 13: mEVP ghost twin bodies and the wide cadence — cells ② and ④

**Files:** Modify: `src/fesom_ice_maevp.cpp`, `src/fesom_ice.cpp` (call site)

- [ ] ⚠️ these are **twin bodies, not shared kernels** — the byte argument is *identical expression
      text in the same TU* reading `W.area0x/corx/coastx/gs_g/…`, exactly as std EVP's
      "EDIT WITH ITS TWIN / EDIT BOTH OR NEITHER" blocks (`fesom_ice_evp.cpp:616-643`, `:721-786`,
      `:832-916`). mEVP needs **3–4** of them; budget accordingly
- [ ] prestep: the 11 node fields, `rhs_a/rhs_m` shipped **after** their area scaling (trap 7)
- [ ] ghost per-step maps by running the owned maps over extended ranges — **including the two the
      first draft omitted**: `maevp_aux_init` (`:142-146`, spans `[0,N)` today) and the coastal BC
      (`:313-316`, std EVP replays it via `W.coastx`, `fesom_ice_evp.cpp:915`)
- [ ] per subcycle: element kernel over owned + ghost; node update over owned + ghost rings 1..K−1; ring K frozen; refresh every K-th
- [ ] window ship — **per D1**: ④ = `(u,v,R_u,R_v)` one segment; ② = node + σ segments, built **fused and unfused** behind a flag
- [ ] ghost `u_rhs/v_rhs` by owner-order gather (`gath_ptr/gath_elem/gath_k`), never a scatter
- [ ] **gate L1**: K=1 reduces to today byte-for-byte, both forms, with `FORCE_SERIAL` + announce grep
- [ ] **gate L2**: prestep echo exactly `0.000e+00`; drift diagnostic present; selfcheck within its relative bound
- [ ] **gate L3**: K ∈ {2,4,8} × {②,④}; **gate L4** CUDA fidelity; **gate L5** options ×3
- [ ] rebuild both, THEN freeze to `/work/.../m9/bin/s2/`; sha256 here
- [ ] gates must pass before Task 14

### Stage 4 — measure and synthesise

### Task 14: Fleet 2A/2B

- [ ] 2A: ② (both send forms, per D1) vs ④ at K ∈ {2,4,8}
- [ ] 2B: ④ vs ⑤b at equal K
- [ ] same staging as Task 10; record messages/partner and bytes/partner to test H3 directly
- [ ] gate: every leg rc=0, mechanism panel complete

### Task 15: Figures and results

**Files:** Create: `scripts/m9_figs.py`, `docs/plans/20260804-m9-RESULTS.md`

- [ ] 🔴 **read `scripts/m7_scaling_figs.py` FIRST** (node-axis ticks, decimal y, SYPD at production dt 1800/900/240/240)
- [ ] F1 cell table (icedyn + SYPD); F2 Win A / Win B vs verts-per-rank (opposite slopes); F3 K sweep; F4 ice-field difference vs K; F5 mechanism panel
- [ ] state measured vs pre-registered for H1–H4, including where the pre-reg was wrong
- [ ] gate: figures regenerate from JSON with one command; no hand-edited numbers

### Task 16: Verify acceptance criteria

- [ ] every cell measured on CPU and GPU, SYPD beside every `icedyn` number
- [ ] L0 green on the final build, both backends; all L1 rungs byte-identical **with FORCE_SERIAL + announce greps**; `MEVPNOEPS` exactly zero
- [ ] L2 within its **relative** bound; mask-flip jumps reported, not hidden
- [ ] L3 recorded for every cell; L4 and L5 green
- [ ] pre-registration committed before the first **fleet** job
- [ ] no binaries committed; frozen bins in `/work` with sha256

### Task 17: [Final] Documentation and lessons

- [ ] `docs/KOKKOS_PORTING_LESSONS.md`: the `ice_nod`-guard trap, the unscaled-R decision, the rails-inside-`icedyn` measurement trap, and whatever the measurement falsifies
- [ ] README knob table; the restart note (below)
- [ ] rebase onto `main` — **then re-run L0** (a rebase moves the tree under the frozen binaries and invalidates their provenance) *or* record provenance explicitly against the pre-rebase commit
- [ ] move this plan to `docs/plans/completed/`

---

## Technical Details

### The two honest non-equivalences

1. **Roundoff.** `det1·Σσ + Σdet2σ̃` vs `Σ(det1σ + det2σ̃)` — identical in exact arithmetic. The
   recursion contracts (`det1 = 250/251 < 1`), so this stays at roundoff rather than growing.
2. **`ice_el` mask membership across ocean steps**, in **both** directions. *Leaving*: today the
   contribution drops discontinuously and σ freezes; in divergence form it decays over ~251
   subcycles ≈ 2 steps. *Re-entry* (the larger asymmetry): today the element resumes from its
   **frozen stale σ** (trap 12, `fesom_ice_maevp.h:44`), whereas R has decayed. Confined to
   marginal ice (`msum ≤ 0.01` ⇒ `P` ≲ 1 % of typical). Measured by the mask-flip instrument.

### Knob table

| knob | helper | announces? | cells |
|---|---|---|---|
| `FESOM_SPEED_MEVPDIV` | `fesom_speed_on_exp` | **yes, automatically** | ③④⑤b |
| `FESOM_SPEED_MEVPNOEPS` | `fesom_speed_on_exp` | **yes, automatically** | control |
| `FESOM_SPEED_MEVPLAG=K` | `fesom_speed_int` | **no — hand-written** | ⑤a/⑤b |
| `FESOM_SPEED_EVPWIDE=K` | existing | already announces + guards | ②④ |
| `FESOM_MEVPDIV_SELFCHECK` | getenv | diagnostic | — |

All default-off; all forced OFF on Serial without `FESOM_SPEED_FORCE_SERIAL=1`.

### Wide-halo contract

R = K rings. Node rings 1..K and every local element with maxring ≤ K. Each subcycle: element
kernel over owned + ghost; node update over owned + ghost rings 1..K−1; ring K frozen; refresh
every K-th. Ring ρ is clean for K−ρ subcycles; dirt advances one ring per subcycle and reaches
ring 1 only *at* the refresh subcycle, after the owned reads.

Per-window ship (`Ng` = ghost nodes; ghost elements ≈ 2·`Ng`):

| cell | ships | doubles | msgs/partner |
|---|---|---|---|
| ② classic | u,v on rings + σ11,σ12,σ22 on ghost elements | **8·Ng** | 2 unfused / **1 fused** |
| ④ divergence | u,v,R_u,R_v on rings | **4·Ng** | 1 |

### Pre-registration

- **H1** Win A grows with per-rank workload (σ streams from DRAM on large subdomains, may sit in
  cache on small ones) ⇒ biggest on GPU and ng5/dars. Element-kernel streamed traffic −~25 %;
  `icedyn` −8…15 % on GPU.
- **H2** Win B grows with rank count ⇒ biggest at 16–32N on core2/farc. Exchange count
  120 → 120/K (+1).
- **H3 (revised).** `② − ④` is **not** pure communication. It decomposes as
  `(bytes: 8·Ng → 4·Ng) + (messages: 2 → 1, only in the unfused build) + (ghost-share × per-row
  Win A)` — the last because ②'s ghost element kernel does the σ read-modify-write that ④'s does
  not. The per-row Win A is already measured by `③ − ①`, and the ghost share by the build summary,
  so the decomposition is closable from measurements we already take. With ②-fused, the message
  term is zero by construction and the residual is bytes + ghost-Win-A.
- **H4** banked negative: std-EVP EVPWIDE gave **−2.2 % of total at 16N, K=8** against a −27 %
  ceiling. Expect that order at model level; `icedyn`-relative will be several times larger —
  which is why `icedyn` is the headline and SYPD sits beside it, not instead of it.
- The **H1 × H2 crossing** is the punchline: the two wins depend on per-rank workload in opposite
  directions and the divergence form is the only cell that collects both.
- Floors 0. History: five consecutive M7 pre-regs measured wrong-LOW (L93); do not adjust on
  harvest, run the L94 checklist.

### Restart note (documentation only)

No ice restart path exists (`fesom_ice.cpp:582`); σ is cold-start-zero. When restart arrives the
carried state must be restored and **follows the form knob**: the divergence set is 3× smaller and
on node fields; conversion is **one-way** (`R = ∇·σ` computable exactly from a restored σ — it is
just the assembly; σ not reconstructible from R); either state is ~2 ocean steps of memory, so
dropping it costs a transient. *This is not a justification for the R rail* — R needs no host rail
because no host code reads it (review Over-engineering note); the rail question is settled in
Task 5 on measurement grounds.

---

## Post-Completion

**L6 climate legs** — which cells earn a 1-year leg is decided after Stage 4, probably on farc.

**The paper** — cell table, the opposite-slope figure, the K sweeps and the accuracy figure.
Sergey's note is the theory; the measured ②-vs-④ delta is the result. Note the paper must state
that ②'s message-count penalty is implementation-dependent (D1) and that our cell ③ deviates from
his F90 in storing R unscaled.

**FESOM2 Fortran** — soft: a transliteration needs a K-wide halo there too (`eXDim_nod2D` /
`exchange_nod2DX` already exist, which is what his F90 uses).

**Adoption** — everything stays behind default-off knobs and the knob-off byte gate, so adopting a
cell later is flipping a default.
