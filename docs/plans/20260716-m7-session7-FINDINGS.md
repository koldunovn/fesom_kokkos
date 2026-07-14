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
| 26258751/52/54 (16N GPU, 8N GPU, 8N CPU) | 16N ratio 4.5–5.0×; 8N 5.0–5.8× | *in flight / pending* | ⏳ |
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

---

## 3. STANDARD-SET LEDGER LEGS (300 steps, h9, pinned, a100_80/compute)

| leg | s/step (min of 2) | job |
|---|--:|---|
| NG5@4N GPU (h9 = +SMOOTHSCRATCH) | **0.6739** | 26258582 |
| NG5@4N CPU (h5-measured, CPU never moves) | 4.5785 | 26256684 |
| NG5@8N CPU | **2.3530** (2.3530/2.3533) | 26258754 |
| NG5@16N CPU | **1.2267** (1.2353/1.2267) | 26258753 |
| NG5@8N GPU | ⏳ 26258752 | |
| NG5@16N GPU | ⏳ 26258751 | |

**NG5@4N ratio: 6.79×.** 8N/16N ratios pending their GPU legs.
