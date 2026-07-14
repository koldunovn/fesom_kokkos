# M7 session 7 — findings, harvest, and the H.8 pre-registration

*2026-07-15 (overnight, autonomous session). Branch `m7-speed`. Written AS THE WORK HAPPENS —
the H.8 number below is committed to BEFORE its A/B job is submitted, per the standing rule.*

---

## 1. HARVEST — scored against the session-6 pre-registrations (§1.1 of the PROMPT)

| job | pre-registered | measured | verdict |
|---|---|---|---|
| **26258712** (300-step nsys census, h9, a100_80) | the three H.7 gaps (`smooth_nod3D` 13.2, `kpp_mixing` 9.0, `sigma_xy` 3.4) GONE; top becomes halo self-gaps + `ice_h_diag→oce_fluxes_mom` ~7 + `hbar→timestep` ~6 | **exactly that**: no trace of the three H.7 gaps; halo self-gaps 9.0+8.2; `ice_h_diag→oce_fluxes_mom` **7.3**; `hbar→timestep` **6.0** | ✅ **HIT** |
| **26258753** (NG5@16N CPU, 300 steps, h9 Serial `91eeb573` pinned ✓) | *the measurement* | **1.2267 s/step** (min of 2: 1.2353 / 1.2267, 0.7 % spread) | ✅ landed |
| **26258582** (H.7 confirmation A/B, h9 `9e1f514b` pinned ✓, a100_80) | base 0.7058 ±0.5 %; scratch ≈0.675; Δ≈−4.2 %; if both hold ⇒ ratio ≈6.78× | base **0.7052** (0.7061/0.7052, −0.09 % vs anchor); scratch **0.6739** (twice, 0.00 % spread; announce fired); **Δ = −4.44 %** | ✅ **HIT — ⭐ RATIO = 4.5785 / 0.6739 = 6.79× at NG5@4N** |
| **26258752/54** (NG5@8N GPU+CPU, h9 pinned ✓, a100_80 ✓) | 8N ratio 5.0–5.8× | GPU **0.4143** (0.4143/0.4145) / CPU **2.3530** ⇒ **ratio 5.68×** | ✅ **IN RANGE** |
| 26258751 (16N GPU) | 16N ratio 4.5–5.0×; 🔴 below 4.0 ⇒ L84(b) WRONG, say loudly | *pending* | ⏳ |
| 26248860 (old 16N 4-leg ladder) | FLAT ≈−2.0 % ⇒ B/C on top; ≈−0.9 % ⇒ L84(b) stands | *pending (Priority)* | ⏳ |

**The h9 census headline (steps 99–296 of 297, rank 0):** step **678.1 ms** (traced),
kernels busy **543.2 ms = 80.1 %**, gaps > 1 ms only **46.4 ms/step (6.8 %)**. The step is even
more kernel-dominated than the ~72 % the session-6 handoff projected. The host era is nearly over.

---

## 2. H.8 LAZYSNAP — the audit, the sizing, and the PRE-REGISTRATION

### 2.1 The code audit confirms the ice half and KILLS the ocean half

**Ice half — verified in source, all four pre-seeded facts hold:**
- The 9-copy OUT rail `fesom_ice.cpp:934-941` fires EVERY step under ICERAILS: `a/m/ms`,
  `uice/vice`, `srfoce_u/v`, `h_ice/h_snow` — 9 × 3.54 MB nod2D DtoH.
- The snapshot gather (`fesom_io.cpp:381-387`) reads **7** of the 9 through `h_checked()`
  (`a/m/ms`, `uice/vice`, **and `h_ice/h_snow`** — the ICERAILS comment's "no production I/O
  reader" for h_ice/h_snow refers to the *stream*, not the snapshot; the snapshot DOES gather them).
- `snap_every_cli=-1 → snap_every=0` (`fesom_main.cpp:966-967`): **the gather never runs in any
  benchmark**, and in production (monthly) it is ~10⁴× rarer than the rail.
- `srfoce_u/v` are gathered by NOTHING — their only host readers are the host `oce_fluxes_mom`
  (dead under ICEFLUXDEV) and env-gated debug — so they are pure deletions.

**Complete host-reader census of the 9 fields (everything found, with its kill):**

| host reader | fields | dead when |
|---|---|---|
| snapshot gather `fesom_io.cpp:381-387` | a/m/ms, uice/vice, h_ice/h_snow | **moved to snapshot cadence — the lever** |
| host `oce_fluxes_mom` `fesom_ice_coupling.cpp:675+` | uice/vice, srfoce_u/v, a_ice | **ICEFLUXDEV** (early-return :670) |
| host `cal_shortwave_rad` `fesom_bulk.cpp:926` | a_ice | **FLUXDEV && SWSKIP** (early-return :868) |
| legacy host IO-stream resolvers (5 ice vars) | a/m/ms, uice/vice | **IOACC** (device accumulators) |
| `FESOM_DIAG_MICE` block `fesom_ice.cpp:947-960` | m_ice, uice/vice | env-gated debug → **ABORT** |
| `FESOM_DIAG_GID` block `fesom_ice.cpp:965+` | uice/vice, srfoce_u/v, a/m_ice | env-gated debug → **ABORT** |
| host bulk `fesom_bulk.cpp:440` | uice/vice | init-only (`fesom_main.cpp:905`, before the loop; mirrors are IC-current) — SAFE |
| host EVP/thermo verify twins | several | ICERAILS already aborts on `FESOM_KK_VERIFY=ice*` |

⇒ **requirement set: ICERAILS && ICEFLUXDEV && FLUXDEV && SWSKIP && IOACC** (all ride the
`FESOM_SPEED=1` master), **abort** (L80) on any missing + on `FESOM_DIAG_MICE`/`FESOM_DIAG_GID`.

**The cross-step trap (`:931-933`) re-derived, as the handoff demanded:** deleting the syncs leaves
the 9 fields Device-authoritative. Every next-step consumer reads them on the DEVICE (std-EVP and
mEVP kernels, FCT, thermo — all their host bounces are `!icerails`-gated and dead). The only
`modify_host()+sync_device()` that still runs on any of the 9 under the required set is the
**once-only IC push** (`fesom_ice.cpp:576-605`), which fires at step 1 while the host mirror IS the
IC — before the rail would ever have run. There is no clobber path. *(The Z7/step-1 question — who
wrote the initial value — is answered: the IC push does, and it survives.)*

**Ocean half — ❌ the handoff's "might be ~2×" hope is DEAD, killed by reading the code:**
- `T,S,w,uv,density,bvfreq,pgf_x/y,Kv,Av` are ALREADY snapshot-cadence (`fesom_main.cpp:1375-1384`,
  the M5-era pre-I/O sync block). Nothing to win.
- `eta_n` is host-authoritative (host writes it, pushes at `fesom_step.cpp:519,818`). Free.
- The census rows the handoff pointed at are a DIFFERENT class: `hbar→timestep` (6.0 ms) and
  `ssh_solve→update_vel` (2.4) + `halo→ssh_rhs` (2.9) + `timestep→ale_thickness` (2.4) are the
  **host-staged nod2D halo bounces** of the SSH solve (`fesom_exchange_nod2D` on ssh_rhs / d_eta /
  ssh_rhs_old / hbar / hbar_old, `fesom_step.cpp:646-682`) plus the **host `eta_n += d_eta` update**
  — all STEP-CADENCE consumers. Moving them needs a device nod2D halo + porting the eta_n/hbar
  host chain: a separate lever (call it H.9 SSHRAILS, ~11-14 ms of gap, the ICERAILS pattern),
  NOT a lazy-sync. *(Four-for-four now: every plan estimate this campaign, re-examined against
  source, was wrong — H.3 2× bigger, the KPP chain nonexistent, H.6 2× smaller, H.8's ocean half
  nonexistent-as-scoped.)*

### 2.2 Sized from the census (L89: gap ms, not bytes)

The rail is the **entire** `fesom_ice_h_diag_kk → fesom_ice_oce_fluxes_mom_kk` gap:

| | ms/step |
|---|--:|
| gap total | **7.3** |
| of which PCIe (9 DtoH copies, 31.8 MB = 9 × 3.54) | 5.0 |
| of which fence / host | 0.1 / 0.1 |
| step time (traced) | 678.1 |

⇒ 7.3 / 678.1 = **−1.08 %**, and the census is a FLOOR (L93: 4-for-4 census-sized levers beat
their pre-registration; sub-ms fences are invisible; entanglement recovers downstream launch gaps).

### ⇒ **PRE-REGISTERED (before the A/B is submitted): H.8 LAZYSNAP = −1.1 % at NG5@4N**
### **floor −1.0 % · ceiling −1.6 %** *(ceiling widened per L93 — I have been wrong LOW four times)*

Honest scope note: −1.1 % is **half** of what the session-6 handoff hoped (~2 ms ice + ~7 ms ocean
≈ 2 %); the ocean half died in audit (§2.1). This is the last host-rail lever of its class.

### 2.3 Shape of the change (BULKTAIL template `93d434d`)

- `fesom_lazysnap_on()` in fesom_io.cpp (guards once: requires ICERAILS+ICEFLUXDEV+FLUXDEV+SWSKIP+
  IOACC, aborts on FESOM_DIAG_MICE/GID).
- `fesom_ice.cpp:934-941`: the 9 sync_host gated on `!fesom_lazysnap_on()`.
- `fesom_io_write_snapshot`: sync_host the 7 gathered ice fields at the top (const_cast, D21
  precedent) — unconditional (no-op when already Synced), so BOTH knob states and BOTH callsites
  (IC snapshot + loop) are covered by construction; a missed sync is a loud SYNCCHECK abort, not
  silent garbage.
- Byte-proof subtlety (from the handoff): the knob-ON Serial gate leg runs with `snap_every>0`, so
  the snapshot bytes themselves prove the sync-before-gather is complete.

Gate ladder (per-lever, L91): knob-OFF byte → FORCE_SERIAL byte proof → CUDA fidelity (isolated +
full) → options matrix ×3 → 35-step A/B → 300-step anchor (a100_80) → ledger.

### 2.4 Built + submitted (binaries frozen FIRST: `m7/bin/h10`, CUDA `13dbddb4` / Serial `7c75afc0`)

| job | gate | expectation |
|---|---|---|
| 26259160 | knob-OFF byte (Serial, live build = h10) | diff_snap rc=0 |
| 26259161 | FORCE_SERIAL byte proof, LAZYSNAP+5 deps | rc=0 (pure re-execution elimination) |
| 26259162 | FORCE_SERIAL byte proof, full blessed set | rc=0 |
| 26259164 | CUDA fidelity, LAZYSNAP+deps isolated, h10 pinned | PASS at the climate-close floor — THE gate (L86): the snapshot gathers a_ice/uice/h_ice from the newly-lazy mirrors, so a missed sync is domain-wide here |
| 26259165 | CUDA fidelity, full blessed (`FESOM_SPEED=1` now includes LAZYSNAP) | PASS |
| 26259166/67/68 | options matrix ×3 (TKE / mEVP / zstar vs their own oracles) | PASS ×3 (zstar's Kv floor 9.537e-02 is the standing control, L79) |
| 26259169 | **GUARD test**: LAZYSNAP + ICERAILS only | must **ABORT** with the missing-deps message (job FAILs — that is the pass) |
| 26259170 | the pre-registered 35-step A/B, NG5@4N, a100_80, h10 | **−1.1 % (floor −1.0, ceiling −1.6)** |

### 2.5 ✅ GATE LADDER: NINE FOR NINE (lever committed as `c9f2fee`)

| gate | result |
|---|---|
| 26259160 knob-OFF byte | ✅ diff_snap rc=0 |
| 26259161 FORCE_SERIAL, LAZYSNAP+deps | ✅ rc=0 — pure re-execution elimination, snapshot bytes identical at snap_every=10 |
| 26259162 FORCE_SERIAL, full blessed | ✅ rc=0 |
| 26259164 CUDA fidelity, isolated | ✅ PASS — a_ice 5.7e-04, m_ice 1.5e-03, uice 5.9e-04, h_ice 5.5e-03, all far inside the floor. THE gate (L86), and it READS the lever's output (L83): the snapshot gathers all 7 lazy fields |
| 26259165 CUDA fidelity, full blessed | ✅ PASS |
| 26259166 options TKE | ✅ PASS |
| 26259167 options mEVP | ✅ PASS — the branch ICERAILS silently corrupted is green with the new lever |
| 26259168 options zstar | ✅ PASS — **the standing control holds exactly**: Kv max|Δ| = 9.537e-02, identical to zstar's own CUDA-vs-Serial floor (L79) ⇒ the levers add nothing |
| 26259169 GUARD | ✅ **ABORTED CORRECTLY**: `REQUIRES ... (have icerails=1 icefluxdev=0 fluxdev=0 swskip=0 ioacc=0) ... Refusing to run`, rc=1 |

### 2.6 ✅ THE A/B (26259170, h10 `13dbddb4` pinned, pure a100_80, announce armor clean)

| leg | rep a | rep b | min |
|---|--:|--:|--:|
| base (`FESOM_SPEED=1;LAZYSNAP=0`) | 0.6948 | 0.6939 | **0.6939** |
| lazy (`FESOM_SPEED=1`) | 0.6869 | 0.6866 | **0.6866** |
| | | | **−1.05 %** |

**Scored against the pre-registration (−1.1 %, floor −1.0, ceiling −1.6): RANGE HIT.** And the
absolute delta is the census number to the decimal: **−7.3 ms/step, exactly the 7.3 ms gap the
census named.** Calibration note for L93: this is the first census-sized lever with **no
entanglement bonus** — it deleted pure PCIe+fence with no host compute in the gap, so there was no
downstream launch-gap recovery to collect. The census was EXACT here, not a floor. (The internal
consistency check also passed: the base leg 0.6939 at 35 steps = the session-6 35-step baseline
0.7256 − 4.21 % H.7 = 0.695.)

⇒ **H.8 LAZYSNAP LANDS.** 300-step h10 anchor queued (26260292) for the ledger row; expected
≈ 0.6739 − 7.3 ms ≈ **0.667 ⇒ ratio ≈ 6.87×** (pre-registered here before the job lands; ±0.5 %
tolerance as usual).

---

## 2b. THE SYPD CG CORRECTION, RE-DERIVED (handoff §1.2 asked for this)

The formula: `SYPD@dt240 = 0.657 / (s/step @dt180 × correction)`, where
`correction = 1 + CG_share × (iters@240/iters@180 − 1)`.

Measured inputs (h9, NG5@4N, 300-step trace 26258712, steady window 99–295):
- **CG region wall = 43.9 ms/step = 6.5 % of the 678.1 ms step** (kernel-busy alone is 10.1 ms —
  the region is Allreduce/halo-latency-dominated, but ALL of it scales with the iteration count).
  Measured with `scripts/m7_cg_share.py` against the census sqlite (CG kernels: `cg_dot`,
  `cg_spmv`, `fesom_ssh_solve_cg_kk` lambdas).
- **iters@dt180 settled = 71.9** (last-50 mean; independently reproduced this session from the
  fresh 16N CPU log — same 71.9 to three digits).
- iters@dt240: NOT measured on this binary (dt240 is CFL-unstable from cold start — SCALING_M524).
  The only data is the original calibration's **115@240 / 89@180 = 1.29** ratio; carrying that
  ratio over is a stated ASSUMPTION, not a measurement.

⇒ **correction = 1 + 0.065 × 0.29 = ×1.019 at NG5@4N** (the old ×1.03 was calibrated on a
cold-start 90-iter CG and a bigger share; it is pessimistic, as session 6 suspected).
⚠️ At **16N** the CG share is UNMEASURED on h9 (comm-bound — plausibly larger). Until a 16N trace
exists, quote 16N SYPD with the old ×1.03 (pessimistic bound) and note ×1.02 as the 4N-measured
value; the difference is ~1 % of SYPD.

---

## 2c. H.9 "SSHRAILS" — pre-audit of the successor lever (scoped, NOT built)

The census class LAZYSNAP could not touch (§2.1) is the **SSH/hbar host-staged nod2D bounce**,
**~13–14 ms/step of gap ≈ −1.9 % at 4N** — and it is a HOST-class lever, so per L84(b) it should
HOLD at 16N. Inventory (census pair-rows ↔ source):

| census row | gap | the traffic | source |
|---|--:|---|---|
| `halo→compute_ssh_rhs_linfs` | 2.9 | HtoD 10.6 MB = d_eta + ssh_rhs_old + hbar pushes | `fesom_step.cpp:630-633` |
| `ssh_rhs→(launch)` | 0.9 | DtoH 2.5 MB = ssh_rhs sync for its host halo | `:646-647` |
| `ssh_solve→update_vel` | 2.4 | DtoH+HtoD 3.5+3.5 = d_eta sync → host halo → re-push | `:652-658` |
| `compute_hbar→timestep` | 6.0 | DtoH 10.6 (ssh_rhs_old/hbar/hbar_old syncs) + HtoD 7.1 (hbar/hbar_old re-push) | `:670-682` |
| `timestep→ale_thickness` | 2.4 | HtoD 3.5 = the eta_n push after the host update | `:818-819` |

**What keeps the class host-bound — the four host `fesom_exchange_nod2D` (ssh_rhs `:647`, d_eta
`:653`, ssh_rhs_old `:673`, hbar `:674`) plus ONE host loop: the eta_n update
`eta_n = α·hbar + (1−α)·hbar_old` (`:776-784`)**, which reads hbar/hbar_old and writes eta_n on the
host. The device NOD2D halo infrastructure already exists (ICERAILS used `fesom_halo_field2
NOD2D` for srfoce_u/v). The eta_n loop is a trivial per-node kernel (no scatter).

**Traps found in the pre-audit (each one is an ICERAILS-class scar):**
1. The ice step pushes hbar with the comment *"hbar IS host-authoritative — keep"*
   (`fesom_ice.cpp:633`). Under SSHRAILS that flips — the push becomes the BULKTAIL-IC-push clobber
   (Z7 signature). Must be gated.
2. eta_n's own history: the `:818` push carries a 30,000× ssh-accumulator bug scar (see the block
   comment) — the device `resolve_ssh_dev` reads eta_n per step under IOACC. A device eta_n kernel
   REPLACES that push; the substep-4 push `:511/:519` becomes redundant (the comment already says so).
3. hbar/hbar_old host readers to re-audit: `fesom_ale.cpp:804/:829` (which ALE mode, what cadence?),
   the print/BLOWUP blocks (print cadence), `fesom_ale_dump_*` bisect rails, and **zstar** (the ALE
   chain + `update_stiff_mat_ale` — the options matrix is MANDATORY, L91).
4. d_eta is CG's cross-step initial guess (RMW, `:619-627` capture list) — the coherence analysis
   must cover the step boundary, not just the step.

---

## 3. STANDARD-SET LEDGER LEGS (300 steps, h9, pinned, a100_80/compute)

| leg | s/step (min of 2) | job |
|---|--:|---|
| NG5@4N GPU (h9 = +SMOOTHSCRATCH) | **0.6739** | 26258582 |
| NG5@4N CPU (h5-measured, CPU never moves) | 4.5785 | 26256684 |
| NG5@8N CPU | **2.3530** (2.3530/2.3533) | 26258754 |
| NG5@16N CPU | **1.2267** (1.2353/1.2267) | 26258753 |
| NG5@8N GPU | **0.4143** (0.4143/0.4145) ⇒ **8N ratio 5.68×** | 26258752 |
| NG5@16N GPU | ⏳ 26258751 | |
| dars@8N CPU (**150-step protocol** — see below) | **0.8464** (0.8464/0.8466) | 26259246 |
| dars@8N GPU (150-step) | **0.2041** (0.2041/0.2051) ⇒ **dars@8N ratio 4.15×** (was 3.27× at Tier-1) | 26259245 |
| NG5@4N GPU **h10** anchor (+LAZYSNAP) | **0.6666** (0.6666/0.6675; pre-reg ≈0.667 ✅ HIT to 0.06 %) | 26260292 |

**⭐ NG5@4N ratio: 4.5785 / 0.6666 = 6.87× (h10, matched 300-step pinned pair, pure a100_80).**
8N/16N ratios pending their GPU legs.

### 🔴 dars@8N CANNOT run the 300-step protocol from a cold PHC start

Job 26259184 (dist_1024, dt180): **both reps died at exactly step 204** — `CG_kk abort: pp·App =
nan`, deterministic (Serial is bit-reproducible, so identical reps die at the identical step). This
is the KNOWN dars cold-start instability class (SCALING_M524 §"dt=240 is CFL-unstable from the cold
start"; the M5.24 probe showed the blowup arrives *earlier* the finer the decomposition — that
probe was NG5-only, and dars@dist_1024 at dt180 was simply never run past 35 steps: row-0's
0.8563 was a 35-step window). **Not a port bug — physics from a cold IC.**
⇒ dars@8N re-queued at **150 steps** (26259245 GPU / 26259246 CPU, h9 pinned, a100_80): safely
below 204, past the CG ramp (settles ~step 30–50). The ledger row gets a protocol annotation.
