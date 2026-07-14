# M7 session 6 — findings, retractions, and the H.3 pre-registration

*2026-07-15. Branch `m7-speed`. Written AS THE WORK HAPPENED — the H.3 number below was committed to
BEFORE the A/B job was submitted, and the retractions were written BEFORE the ledger was touched.*

---

## 0. HEADLINE

| | |
|---|---|
| **THE RATIO, RE-MEASURED** | **NG5@4N: GPU 0.7239 s/step vs CPU 4.5785 s/step ⇒ 6.33×** *(provisional — CPU min-of-2 pending)* |
| **§3.2's "~22 ms unattributed"** | ❌ **RETRACTED. It never existed.** |
| **H.3 BULKTAIL** | pool re-sized **~1 % → 2.2 %** by measurement; it is **the single largest gap in the step** |
| **New bug found** | 🔴 `ICERAILS` + `mEVP` **clobbers `srfoce_u/v/ssh`** — pre-existing, in the landed `a96e299` |

---

## 1. ✅ HARVEST — scored against the pre-registrations, honestly

| job | pre-registered | measured | verdict |
|---|---|---|---|
| 26255934 (25-step profile, h5) | loop ~0.829 | 0.8297 | ✅ HIT (scored last session) |
| 26255935 (A/B `NOCOEFCACHE`) | −5.7 % | −5.31 % | ⚠️ over-predicted by 7 % — owned last session |
| **26255936** (GPU@300, h5) | *the measurement* | **0.7247 / 0.7239 s/step** (min of 2; 0.11 % spread) | ✅ |
| **26255937** (CPU@300, h5) | *the measurement* | **TIMED OUT** at 45 min after rep A (4.6519). Requeued as **26256684** → rep A **4.5785** | ⚠️ requeued |
| **26256274 + 26256275** (gap censuses) | "names the last ~22 ms" | ❌ **there is no ~22 ms** — see §2 | ❌ **PREMISE FALSIFIED** |
| 26248860 (NG5@16N ladder) | FLAT ≈ −2.0 % ⇒ B/C back on top; ≈ −0.9 % ⇒ L84(b) stands | **STILL PENDING** (16 GPU nodes) | ⏳ |

**The binaries were pinned and it worked.** Both anchors' `SHA.txt` show the frozen `h5` md5s exactly
(`0d39d8a2` CUDA, `950ee0f9` Serial) even though the two jobs recorded *different git HEADs* at submit
time. Rule 0.4 did its job: **`BIN=` is what ran, the build tree is irrelevant.**

---

## 2. 🔴 RETRACTION: THE "~22 ms UNATTRIBUTED" WAS NEVER REAL

The handoff (§3.2) said the 35-step protocol still carried **~31 ms** the `getcoeffld` fix does not
touch — 8.7 ms of CG spin-up plus **~22 ms unattributed**, which "sits in the halos / host / MPI
remainder that the PHASE profiler structurally cannot see. **Only a trace reaches it.**"

**A trace reached it. There is nothing there.** Diffing the two matched gap censuses (26256274 @300
vs 26256275 @25, same binary, same flags):

| | 300-step (steps 100-299) | 25-step (steps 8-24) | A − B |
|---|--:|--:|--:|
| step time (from the trace) | 730.8 | 738.8 | **−8.1 ms** |
| kernels busy | 545.3 | 545.9 | −0.6 ms |
| **GPU-idle gaps > 1 ms** | **94.1** | **93.6** | **+0.5 ms** |

The gap budgets are **identical**. No kernel's stall moves by more than 2.4 ms. And the per-step wall
series settles it outright:

| steps | mean ms/step |
|---|--:|
| **1** | **1235** ← a one-off (CUDA context / JIT / first-touch). *Already excluded: the loop timer drops 5 warmup steps.* |
| 6–10 | 745.4 |
| 21–35 | 738.6 |
| 101–200 | 732.3 |
| 201–299 | 729.3 |

**Post-fix, the 25-vs-300 loop-timing delta is 8.2 ms, not ~31** — a smooth ~16 ms decay over ~200
steps that tracks the CG iteration ramp (86 → 72 iters) exactly. **That is the CG spin-up, it is
physics, and it is the WHOLE residual.**

**Where the phantom came from:** §3.2 decomposed a **pre-fix** 72.6 ms delta by *subtracting an
approximate model* of the `getcoeffld` escape (49.3 ms/step for K steps, then 7.7). The leftover
after that subtraction — ~22 ms — was **model error, not a physical cost**, and it was then given a
physical story ("the profiler is structurally blind to it"). **Do not decompose a measurement by
subtracting a model and then name the remainder.** The phase profiler was never blind to 22 ms.

**Consequence: the 300-step protocol is clean.** Its reported loop timing sits within ~3 ms of its own
steady state, so the anchors below need no correction.

---

## 3. THE RATIO — RE-MEASURED, NOT ADJUSTED

Matched pair, both on the frozen `h5` binaries, both 300 steps (5 warmup excluded), both NG5@4N
node-for-node (4 GPU nodes × 4 ranks vs 4 CPU nodes × 128 ranks), same day:

| | s/step | job |
|---|--:|---|
| **GPU** (CUDA `0d39d8a2`, `FESOM_SPEED=1`) | **0.7239** (min of 2: 0.7247 / 0.7239) | 26255936 |
| **CPU** (Serial `950ee0f9`, dist_512) | **4.5785** (rep A; min-of-2 in flight) | 26256684 |
| **RATIO** | **6.33×** *(provisional)* | |

*(The first CPU job timed out after one rep at 4.6519 — 1.6 % slower than the requeued rep. Run-to-run
CPU spread is real; the min-of-2 is why this is still marked provisional.)*

**Superseded numbers, kept so they cannot quietly return:** 5.84× (35-step, contaminated), 6.17×
(35-step, post-fix), and the retracted "5.83× / 6.2×" which mixed protocols. **6.33× is the first
number in this campaign measured on a matched, pinned, both-post-fix, cold-start-free pair.**

**CG correction, re-derived:** the SYPD projection used **×1.03 calibrated on 90 iters/step**. The
settled value is **~72** (last-50 mean 71.9; the 300-step *mean* of 76.9 still carries the ramp).
The old correction is calibrated on a cold-start transient and is **pessimistic**. Re-derive it from
72, not 90, and not 76.6.

---

## 4. 🔴 H.3 BULKTAIL — PRE-REGISTERED **−2.2 %** *(written before the A/B was submitted)*

**The plan said ~1 %. The measurement says 2.2 %, and it is the single largest gap in the step.**

The nsys timeline, one steady-state step, NG5@4N, h5, `FESOM_SPEED=1` — the seven `sync_host()` calls
of `fesom_bulk.cpp:629-635` appear **in source order, with exactly the right sizes**:

```
318.4  KERN fesom_bulk_compute_kk
319.3  copy DtoH 7.07 MB   <- stress_node_surf (2N doubles)   :629
321.8  copy DtoH 3.54 MB   <- heat_flux        (N)            :630
323.0  copy DtoH 3.54 MB   <- water_flux                      :631
324.1  copy DtoH 3.54 MB   <- Ch_atm_oce                      :632
325.3  copy DtoH 3.54 MB   <- Ce_atm_oce                      :633
326.4  copy DtoH 3.54 MB   <- stress_atmice_x                 :634
327.3  copy DtoH 3.54 MB   <- stress_atmice_y                 :635
327.9 ─────── 7.8 ms of UNTRACED HOST ───────  <- the memset + the 930k-elem interp loop :640-650
335.7  copy HtoD 3.54 MB   <- hbar (fesom_ice.cpp:609 — NOT ours, host-authoritative, stays)
336.1  KERN fesom_ocean2ice_kk
       ══ 16.8 ms of GPU IDLE ══
```

| | ms/step |
|---|--:|
| the 7 DtoH copies (28.3 MB) | 6.3 |
| inter-copy host gaps | 2.3 |
| the memset + the interp loop | 7.8 |
| **⇒ removable (319.3 → 335.7)** | **16.4** |
| the hbar HtoD (not ours) | 0.45 |
| **total gap** | **16.8** |

**16.4 ms / 743.5 ms (the 35-step A/B baseline) = −2.21 %.**

### ⇒ **PRE-REGISTERED: −2.2 %** · floor **−1.9 %** · ceiling **−2.4 %**

**Why the plan under-sized it by 2×:** it costed the *rails* (30 MB ⇒ ~1 %) and never noticed that a
**dead 930k-element host interpolation loop sits in the same gap and costs another ~8 ms**. Both
halves are dead, and they are dead for different reasons — which is why only the gap census, which
counts *time the GPU is idle* rather than *bytes moved*, found both.

**Both the 25-step and the 300-step census report this gap (16.8 / 17.2 ms), so it is steady state,
not a cold-start artifact.**

### The audit said 3 of the 7 copies were already no-ops. The trace said all 7 fire. The trace won.

A source audit concluded that `stress_node_surf`, `heat_flux` and `water_flux` were left `Synced` by
their halo exchange, making `:629-631` no-ops at `npes>1`. **Wrong.** On CUDA, `fesom_halo_field`
**returns early** at `fesom_halo_device.hpp:129` after the device pack/unpack and leaves the field
`Auth::Device`; the `sync_host()/modify_host()/sync_device()` the audit quoted is the **host-staged
fallback** for Serial/OpenMP. All seven copies are real, and the timeline shows all seven.

*(This is L85 again, from the other direction: an argument from source and a measurement disagreed,
and the measurement was right. The argument failed on a single `#ifdef` early-return.)*

---

## 5. WHY BULKTAIL IS NOT INDEPENDENT — and the step-1 trap it walked into

The tail is dead **only under the blessed set**. Each of the eight arrays is kept alive by a
*different* lever:

| array | what keeps the host mirror needed | the lever that kills it |
|---|---|---|
| `stress_node_surf`, `stress_surf` | `oce_fluxes_mom`'s HOST loop reads `sns` from the raw alias | **ICEFLUXDEV** (makes it a device kernel) |
| `heat_flux`, `water_flux` | two host re-pushes (`fesom_step.cpp:418,:1081`) + `cal_shortwave_rad`'s host `+=` | **FLUXDEV** |
| `Ch_atm_oce`, `Ce_atm_oce` | the thermo IN rail (`fesom_ice.cpp:809`) | **ICERAILS** |
| `stress_atmice_x/y` | the EVP IN rail (`fesom_ice.cpp:660`) | **ICERAILS** |

With any one of them OFF, a downstream `modify_host(); sync_device()` finds `host_count > dev_count`
and **deep-copies the stale host mirror back over the fresh device data** — silently. So BULKTAIL
**aborts loudly** if `ICERAILS && ICEFLUXDEV && FLUXDEV` are not all on (never silently downgrades —
L80), and also on `FESOM_KK_VERIFY=bulk`, on `FESOM_NO_ICE_THERMO / NO_WIND / NO_HFLUX`, and on
`whichEVP != 0`. **Gate 26256926 tests that the abort actually fires.**

### 🔴 The trap: the belt-and-braces IC push becomes the bug it was insuring against

`fesom_ice.cpp:590-591` pushes `stress_atmice_x/y` host→device once, at step 1, under ICERAILS. Its
own comment says these fields "**need no IC push**… Pushed anyway: it is once, and an IC bug in this
class costs a whole gate cycle."

Under BULKTAIL that push **becomes the bug**. Bulk's tail is what kept the host mirror current; delete
it and `stress_atmice` is device-authoritative with a host mirror still holding `Field::alloc()`'s
**zeros**. `modify_host()` then bumps `host_count` above `dev_count` and `sync_device()` **copies the
zeros over the fresh device stress — killing the wind forcing on the ice, on step 1 only.**

**That is the Z7 signature exactly** (bitwise-equal at cold start is what makes step 1 the *only*
wrong step) — and step 1 being wrong is the whole run. Fixed: the push is skipped under BULKTAIL.
**Found by asking rule 0.3's question — "who put the initial value there?" — before running a gate,
not after.**

---

## 6. 🔴 NEW BUG, PRE-EXISTING AND LANDED: `ICERAILS` + `mEVP` clobbers `srfoce_u/v/ssh`

**Not caused by this session's work. It is live in the committed `a96e299`.**

Under `FESOM_SPEED_ICERAILS=1`, `fesom_ice_step` leaves `srfoce_u/v/ssh` **device-authoritative**: the
knob replaces ocean2ice's five `sync_host()` calls with a device halo (`fesom_ice.cpp:611-623`), and
on CUDA `fesom_halo_field2` returns early leaving `Auth::Device`.

But the **mEVP branch** (`whichEVP==1`, `fesom_ice.cpp:681-693`) is **NOT gated on `icerails`** — it
still does `modify_host(); sync_device()` on `srfoce_u`, `srfoce_v`, `srfoce_ssh` (and on
`stress_atmice_x/y`) **every step**. That deep-copies the **stale host mirror over the fresh device
data**: mEVP runs on last step's surface currents, and on **zeros at step 1**.

- The standard-EVP branch right above it (`:653-667`) **is** correctly gated. The mEVP branch was missed.
- **No gate could have caught it.** The default is `whichEVP=0`, and M6's mEVP validation was a
  **Serial** bit-identity test — where `.d()` *is* `.h()` and every one of these rails is a no-op
  (**L86**).
- **Proposed fix:** gate the mEVP IN rail (`:681-693`) and OUT rail (`:695-696`) on `!icerails`,
  exactly as the EVP branch is. It needs its **own CUDA fidelity gate with `FESOM_WHICH_EVP=1`** —
  which does not currently exist and should be added to the gate ladder.
- **Not fixed in the H.3 commit** (different lever, different gate). BULKTAIL refuses to run with
  `whichEVP != 0` so it cannot compound it.

**The lesson is L86's, restated: every knob that makes a field device-authoritative must be checked
against EVERY consumer of that field — including the ones behind a non-default option knob.**

---

## 7. THE REMAINING GAP BUDGET — the ladder, re-ranked on the census

Package H took the GPU-idle gap budget from **222.6 → 93.6 ms/step**. What is left, measured:

| gap (predecessor → kernel kept waiting) | ms/step | PCIe | host | what it is |
|---|--:|--:|--:|---|
| `halo_device2` → `ocean2ice` | **16.8** | 6.7 | 10.1 | **H.3 BULKTAIL — in flight** |
| `halo_device` → `halo_device` | 12.8 | 2.6 | — | comm (package E) |
| `halo_device` → `smooth_nod3D` | 8.7 | 0.8 | 7.9 | **NEW — the KPP blmc smoother's host code** |
| `halo_device` → `kpp_mixing` | 8.4 | 0.0 | **8.4** | **NEW — pure host, zero PCIe** |
| `halo_device2` → `halo_device2` | 9.0 | 1.5 | — | comm (package E) |
| `ice_h_diag` → `oce_fluxes_mom` | **7.2** | **4.8** | 2.1 | **NEW — H.6 "ICETAIL": 9 DtoH, 31.8 MB, a second ~30 MB tail** |
| `compute_hbar` → `fesom_timestep` | 6.0 | 3.9 | 1.9 | step-top rails (H.4/H.5) |
| `halo_device` → `compute_sigma_xy` | 3.8 | 0.0 | 3.8 | host |
| `halo_device` → `compute_ssh_rhs_linfs` | 2.9 | 2.1 | — | **H.4 SSHRAILS** (10.6 MB HtoD) |
| `timestep` → `ale_thickness_linfs` | 2.6 | 0.6 | 2.0 | |
| `ssh_solve_cg` → `update_vel` | 2.4 | 1.6 | — | |
| `halo_device` → `compute_vel_rhs` | 2.2 | 1.2 | — | |

**Two levers the plan never listed, both bigger than H.4 and H.5:**
1. **H.6 ICETAIL** — a *second* ~30 MB DtoH block (9 copies) at the end of the ice step, 7.2 ms.
2. **The KPP host chain** — `kpp_mixing` + `smooth_nod3D` = **21.8 ms of gap, 16.3 ms of it pure
   host with ZERO PCIe.** This is not a rail at all; it is host *compute* the GPU waits on. It is
   now the largest host-class pool after H.3.

---

## 8. TOOLING ADDED

`scripts/m7_gap_census.py` — the gap census, saved as a reusable script with a **`--diff` mode** so
two traces get a byte-identical instrument. It reports each gap **by the pair it sits between**
(`predecessor → victim`) and itemises the PCIe traffic inside it by **count, MB and direction** —
which is what turned "ocean2ice waits 16.8 ms" into "seven named `sync_host()` calls at
`fesom_bulk.cpp:629-635`". Both nsys traps (`demangledName`, and deduping `MPI_START_WAIT_EVENTS`)
are handled and documented in the file.
