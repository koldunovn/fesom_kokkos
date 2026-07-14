# M7 Package H — THE RAILS: the model's own full-field PCIe traffic

*Opened 2026-07-14 (session 5), from a re-analysis of the EXISTING nsys trace
`/work/ab0995/a270088/port2/m7/hostprof_t1/hostprof.sqlite` (job 26243196, NG5@4N, Tier-1
`FESOM_SPEED=1`, binary `788844b3`). **No new HPC allocation was used to find this.***

---

## 1. The finding

The M7 budget table (`20260714-m7-speed-fp64.md:44`) carries this line:

| pool | ms/step | % | notes |
|---|--:|--:|---|
| memcpy | 89.8 | 9.8 | **MPI staging + forcing HtoD** |

**The label is wrong, and the mislabel is why this never became a package.** Decomposed from the
trace (my total: 95.8 ms/step — it closes against the plan's 89.8):

| | calls/step | ms/step | MB/step | share of the pool |
|---|--:|--:|--:|--:|
| inside an MPI call (UCX staging) | 4 157 | **18.2** | 184 | 19 % |
| **NOT in MPI — the model's own DualView rails** | **244** | **77.6** | **455** | **81 %** |

**114 times per step the model deep-copies a FULL nod2D field across PCIe.** 3 707 088 B =
463 386 doubles = exactly the local domain at NG5@4N (memory: "463 k nodes/rank"). Split by
direction: **67 full-field H2D** (`sync_device()` rails) + **41 full-field D2H** (`sync_host()`),
plus 6 elem2D (7.39 MB). Kokkos `deep_copy` fences and the port is single-stream, so **all of it is
on the critical path.**

**77.6 ms/step = 9.0 % of the current 866 ms step.** That is larger than the entire measured payoff
of package A (−5.37 %), and larger than the whole EVP comm pool.

### Corroboration — two independent instruments agree

An independent **source audit** (never shown the trace) counted the sea-ice step alone at
**40 H2D + 27 D2H = 67 rails** (`fesom_ice.cpp:503-716`), plus the bulk-compute tail (7 D2H), the
forcing round trip (4 D2H + 6 H2D), the SSH block (5 D2H + 4 H2D) and `eta_n`. The trace says
67 H2D + 41 D2H + 6 elem2D. **The counts close.**

**The comments justifying the ice rails are STALE.** They say *"sync_host … for the host EVP / the
host cut_off / the host thermo"* (`fesom_ice.cpp:496,:521,:586,:609,:635`) — every one of those is a
`_kk` device kernel now. The whole chain ocean2ice → EVP → FCT → cut_off → thermo → oce_fluxes →
h_diag is device-side. Concretely dead:
- `:505` `srfoce_temp/srfoce_salt` D2H → re-pushed H2D at `:670-671`. **No host reader between them.**
- `:715-716` `h_ice`/`h_snow` D2H → **no production host reader at all.**
- `a_ice/m_ice/m_snow` bounce D2H→H2D **four times in one step**.

### Why this is the right class of lever (L84)

Rails are **host-class**, not kernel-class. Per L84(b) — measured last session — a GPU-kernel lever
DECAYS in the comm-bound regime (freed kernel time is absorbed by MPI wait the GPU was doing anyway)
while a host lever does NOT. **The rails hold their value into Stage 2 (16N).** This is the same
structural class as SWSKIP (−26 %), IOACC, ICEFLUXDEV and FORCEDEV — and it is why host code keeps
being the answer in M7.

---

## 2. 🔴 PRE-REGISTERED — written BEFORE any measurement. Do not retrofit.

### H.1 `FESOM_SPEED_FLUXDEV` (implemented this session)

Keeps the 4 forcing fluxes (`heat_flux`, `water_flux`, `relax_salt`, `virtual_salt`)
device-authoritative end to end. Sites: `fesom_ice_coupling.cpp` (drop the 4 D2H, 4 host halos → 1
device `fesom_halo_fieldN`), `fesom_bulk.cpp` (the `heat_flux += swsurf` accumulation moves into
`fesom_cal_shortwave_rad_kk`; the host `fesom_cal_shortwave_rad` early-returns), `fesom_step.cpp`
(skip the substep-3 and substep-13b IN rails, which would clobber the device result — the D.1 trap).

**What it REMOVES** (sized by the work removed, not the function it lives in — the session-4 lesson):

| item | count | ms/step |
|---|--:|--:|
| `sync_host()` D2H in `oce_fluxes` (4 fluxes) | 4 × 0.613 | 2.45 |
| `sync_device()` H2D at substep 3 (`:411`, 2 fields) | 2 × 0.620 | 1.24 |
| `sync_device()` H2D at substep 13b (`:1032`, 4 fields) | 4 × 0.620 | 2.48 |
| `shortwave_fld.sync_host()` D2H (added by D.1; the early return kills it) | 1 × 0.613 | 0.61 |
| host `fesom_cal_shortwave_rad` nod2D loop (hostprof, SWSKIP on) | 466 k iters | 1.50 |
| 4 host-staged MPI halos → 1 device halo | 3 fewer exchanges | ~0.5 |
| **ADDS:** `hf(n2) += swsurf` in the existing kernel (7.4 MB @ ~1.5 TB/s) | | ~0.0 |
| **TOTAL** | **11 full-field copies** | **≈ 8.8** |

Per-copy costs are MEASURED from the trace, not assumed: H2D 41.53 ms / 67 = **0.620 ms**;
D2H 25.14 ms / 41 = **0.613 ms** (≈ 6 GB/s effective — PCIe gen4 peak is ~25).

> **PREDICTION: FLUXDEV marginal at NG5@4N = −1.1 % (range −0.9 % … −1.3 %), ≈ 8.8 ms/step.**
> Against a post-D.1 baseline of ~0.809 s/step that is ~0.800 s/step.

**This is BELOW the 2 % adoption bar, and that is expected. FLUXDEV is a KEYSTONE, not a payoff.**
Its job is to remove the last production host reader of `jra->shortwave`, `a_ice` and `heat_flux`,
which is what *pins* the 67-rail ice-step stack to the host. The payoff is H.2.

🔴 **A result near −0.0 % is a DEAD KNOB (L80), not a null lever.** FLUXDEV announces on rank 0 from
**5 call sites**. Before believing any A/B: `grep -c 'FESOM_SPEED_FLUXDEV = ON'` — **expect 5.**

🔴 **THE D.1 TRAP APPLIES VERBATIM.** On Serial `.d()` and `.h()` are the SAME MEMORY, so a rail that
is wrongly left in (or wrongly taken out) is a no-op there: **the FORCE_SERIAL byte proof passes
either way.** It proves only the ARITHMETIC (the accumulation moving into the kernel). **The CUDA
fidelity gate is the ONLY gate that can validate the rail removal.** Do not read a green
FORCE_SERIAL as clearance for the rails.

### H.2 `ICERAILS` — the payoff (NOT yet implemented)

The 67-rail ice-step stack (`fesom_ice.cpp:503-716`), ≈ 250–300 MB/step. Only **three** things pin it
to the host, and **FLUXDEV removes two of them**:
1. the 2 `srfoce_u/v` host halos (`fesom_ice.cpp:507-508`) ← **the one remaining blocker**
2. ~~the 4 forcing host halos~~ → killed by H.1
3. ~~the host `fesom_cal_shortwave_rad` reading `a_ice`~~ → killed by H.1

> **PREDICTION: H.2 ≈ −40 ms/step ≈ −4.6 % at NG5@4N.** The single biggest lever left in the campaign.

### H.3 `BULKTAIL` · H.4 `SSHRAILS` · H.5 `ETADEV`

| | what | est. |
|---|---|--:|
| **H.3** | `fesom_bulk_compute_kk`'s host tail (`fesom_bulk.cpp:629-650`): **7 dead `sync_host()`** (~30 MB/step — every one re-pushed or overwritten before any host read) + a dead **930 k-element** host interpolation loop overwritten microseconds later by `oce_fluxes_mom`. Pure SWSKIP-pattern deletion, no new kernel. | ~1 % |
| **H.4** | the SSH-block host halos (`fesom_step.cpp:638-674`): 5 D2H + 4 H2D + 4 host-staged exchanges — the last host-staged halos in an otherwise device-halo'd ocean step. | ~0.7 % |
| **H.5** | `eta_n` substep-11 host loop (466 k iters) → device. Also deletes the rail the correctness fix just added AND the now-redundant substep-4 push (`:511`). | ~0.2 % |

**Package H total ≈ 7.5–8 % at NG5@4N, and it is host-class → it should HOLD at 16N.**

---

## 3. Ladder corrections this analysis forces

### (a) Package E is mis-sized — the profiler was inflating its own target

`fesom_profile.cpp:61,:73` inserts a **`Kokkos::fence()` before AND after every labeled kernel**.
The EVP subcycle has 5 labeled kernels × 120 subcycles → **~1 200 extra fences/step inside the phase
the profiler is measuring.** The reported `ice_dyn(o2i+EVP)` = 73.5 ms is therefore inflated.

**From nsys (no step-profiler fences), the EVP subcycle loop is 38.3 ms/step:**

| | ms/step |
|---|--:|
| EVP compute kernels (3/subcycle) | 7.3 |
| halo pack/unpack kernels | 0.7 |
| MPI_Waitall | 12.4 |
| MPI_Isend | 8.4 |
| MPI_Irecv | 2.8 |
| fence + launch gap | 6.7 |
| **wall** | **38.3** |

120 exchanges/step, 600 Isends — exactly as specified. So the whole EVP comm pool is **~26 ms**, and
**EVPWIDE k=2 is worth ~13 ms (1.5 %) at 4N, not the 40–80 ms budgeted for package E.**
CG1R is worth ~2 ms at 4N (`MPI_Allreduce` = **3.9 ms/step total**, 184 calls).

**E remains the right STAGE-2 lever** (comm is ~110 ms/step in absolute terms and the 16N step is
~4× shorter, so comm goes to ~30 % there) **— but it is not the big lever at 4N. Package H is.**

### (b) A.3's conclusion was half-wrong

A.3 closed with *"transport is `tag(rc_mlx5/mlx5_0:1)` as intended, **no host-staging pathology to
chase**."* There IS host staging — 4 157 UCX staging copies/step — but it costs only **18.2 ms**, and
the transport being rc_mlx5 says nothing about whether the NIC reads GPU memory directly. The
`get_zcopy`/`put_zcopy` legs at **+35 %** are the signature of GPUDirect RDMA being unavailable on
this partition and falling back to something pathological. **Do not re-chase this; the staging is
forced and it is not where the time is.** (It does mean halo VOLUME crosses PCIe twice — see below.)

### (c) A second, independent finding: the halo pack is NOT depth-aware

`fesom_halo_device.cpp:348` — `stride = n_levels * n_components`, and **every 3D caller passes
`n_levels = mesh->nl`** (the GLOBAL max, 70 on NG5). There is **no `nlvls`/`elvls` lookup anywhere**
in the pack, the buffer sizing, or the MPI count. **A 1-level shelf node in the halo ships 70
doubles, 69 of them below the seafloor** — while the compute kernels right next to it *are*
depth-masked (`fesom_tracer_adv.cpp:1573`, `fesom_eos.cpp:585`).

Waste factor = `nl / mean(nlvls over the halo list)`. Cutting to actual depth scales **all** halo
traffic by that factor — and because staging is forced, every byte saved crosses PCIe **twice**.
Needs a per-halo-entry prefix-sum offset table (breaks the buffer-offset-collapse invariant
documented at `fesom_halo_device.cpp:10-14`).

**And ~7 of the ~44 3D exchanges/step are provably redundant:**
- `helem` exchanged **twice** on zstar with identical values (`fesom_ale.cpp:928` + `fesom_step.cpp:1123`)
- `hnode` (`fesom_step.cpp:1122`) dead on zstar (`ale_zstar_commit_node` already writes the halo)
- the KPP blmc smoother's **last sweep's 3 halos** are dead (sole consumer `kpp_combine` is owned-only)
- the pre-trdiff T/S halos (`:962`, `:991`) look unconsumed — needs a poison test before removal

*(The KPP blmc smoother alone is **12 of the ~44** 3D exchanges/step — 27 % — for one scratch field.)*

---

## 4. A correctness defect found and fixed (separate commit, no knob)

`fesom_step.cpp:773` (substep 11) writes `dyn->eta_n[]` through the **raw host alias** with no
`modify_host()`. Its only rail is substep 4 (`:511`), which fires **before** that loop. So on CUDA the
DEVICE `eta_n` spends the back half of every step holding the **previous step's** value, while the
Field still reports `Synced`. `Field::d()` hands out the device view with **no sync**
(`fesom_field.hpp:74`) — and `resolve_ssh_dev` (`fesom_io.cpp:868`), the `FESOM_SPEED_IOACC` mean
accumulator **which is in the BLESSED set**, reads exactly that view.

**Impact:** the IOACC monthly-mean `ssh` accumulated one step stale on CUDA. The physics is
unaffected (`compute_vel_rhs` reads the fresh substep-4 push, which is correct by design).

**Why every gate passed it — this is the lesson:**
- FORCE_SERIAL **structurally cannot** see it (Serial: `.d()` and `.h()` are the same memory).
- the knob-OFF byte gate reads `snap_*.nc`, written from the HOST path (`fesom_io.cpp:370`,
  `.h_checked()`) — always correct.
- one stale step in ~14 000 is far below the climate gate's floor.

**Only the arithmetic finds this class.** (L80/L83, again.) Fixed by an additive
`modify_host(); sync_device()` after the loop — provably safe: no device kernel reads `eta_n`
between there and the next substep 4, so nothing the momentum solver sees changes.

> **NOTE:** `modify_host()` **alone is NOT the fix** — `d()` does not sync, so the device view would
> stay stale and only a *later* `sync_device()` would copy. The push is required.

---

## 5. Re-ranked ladder

| pkg | class | 4N | holds at 16N? | status |
|---|---|--:|---|---|
| **H — rails** | **host** | **~7.5–8 %** | **✅ yes (L84)** | H.1 implemented; H.2 is the payoff |
| D.1 FORCEDEV | host | (A/B in flight) | ✅ yes | gated green |
| E — comm | comm | ~2 % *(was 4.6–9 %)* | ✅ grows | the Stage-2 lever |
| halo depth-pack + dead exchanges | comm | ~1.5 % | ✅ grows | new, from §3(c) |
| B — FCT2 | **kernel** | large | 🔴 **decays** (L84b) | pending job 26248860 |
| C — TDMA | **kernel** | medium | 🔴 **decays** | pending job 26248860 |
| F — ICELAG | physics | −90..120 ms | ? | experiment, user-approved |

**Job 26248860 (NG5@16N) still decides B/C** — it does not touch H, which is host-class either way.

---

## 5b. 🔴 RESULTS THAT LANDED WHILE THIS WAS BEING WRITTEN (2026-07-14, session 5)

### D.1 (`FORCEDEV`) — job 26249153: **−2.16 %**, vs **−6.6 %** pre-registered

| leg | s/step |
|---|--:|
| `packa` (ref; `FESOM_SPEED=1`, `FORCEDEV=0`) | 0.8659 *(matches the 0.8662 baseline ✓ same-day, same-alloc)* |
| `fdev` (`FESOM_SPEED=1`) | **0.8472** → **−2.16 %** |

**L80 clear — the knob FIRED** (`FESOM_SPEED_FORCEDEV = ON` in both rep logs, no "RESOLVES TO OFF").
This is a **real lever that under-delivered**, not a dead knob. → **L87.** The pre-registration was
built on the FPROF host timer (75.2 ms); the honest exposed cost is the **GPU-idle gap = 60.0 ms**,
and ROTCACHE had already taken 19.7 of it, so the real pool was **~35–40 ms**. D.1 got 18.7 —
**roughly half the pool survived the port.** Job **26253000** (nsys of the `fdev` config) will name
the residue.

⚠️ **This also re-prices H.1.** My FLUXDEV pre-registration (−1.1 %, §2) was sized from **measured
per-copy costs** (0.620 ms/H2D, 0.613 ms/D2H — from the trace, not assumed) plus the hostprof loop
time, so it does *not* inherit D.1's FPROF error. **It stands unchanged.** But the general warning
holds: **size the residue, not just the removal.**

### The GPU-IDLE GAP census — a THIRD independent confirmation of package H

Same trace, different question: where is the GPU doing **nothing**?

```
GPU-idle gaps >1 ms = 222.6 ms/step  (24 % of the step)
   before fesom_bulk_compute_kk         60.0   ← the forcing (D.1's target)
   before fesom_ice_thermodynamics_kk   22.7   ← the ice THERMO in-rail (14 H2D)
   before fesom_ice_evp_dynamics_kk     19.5   ← the EVP in-rail (13 H2D)
   before fesom_ocean2ice_kk            16.5   ← ocean2ice rails + 2 host halos
   before fesom_smooth_nod3D_kk         15.3
   before fesom_halo_exchange_device    14.3   (×10.7/step)
```

**The three sea-ice gaps sum to 58.7 ms of pure GPU idle** — the GPU sitting still while the ice-step
rail stack shuffles full fields across PCIe. **That is H.2, measured from a completely different angle
than the memcpy accounting (77.6 ms) or the source audit (67 rails). Three instruments, one answer.**

### NG5@8N scan — job 26248859 (pinned packA binary, md5 `8b2cdd5c` ✓)

| leg | s/step | Δ vs t1 |
|---|--:|--:|
| t1 (ref) | 0.5523 | — |
| flat | 0.5382 | **−2.55 %** |
| rot | 0.5400 | **−2.23 %** |
| both | 0.5261 | −4.74 % |

🔴 **At NG5@8N the ordering does NOT invert** (FLAT −2.55 % still > ROTCACHE −2.23 %). The inversion
seen at dars@8N (FLAT −1.03 % < ROT −1.83 %) is therefore **at least partly a MESH artifact**, not
purely the comm-bound decay of L84(b). Per-rank size: NG5@4N 463 k → NG5@8N 231 k → NG5@16N 116 k ≈
dars@8N 98.5 k. **Job 26248860 (NG5@16N) is now genuinely decisive:** near **−2 %** for FLAT ⇒ the
decay story is wrong and **B/C go back on top of the ladder**; near **−1 %** ⇒ the re-ranking stands.
**Either way it does not touch package H, which is host-class.**

---

## 6. Method note — the trap I nearly shipped

My first attribution test asked *"what fraction of PCIe copies happen inside an MPI call?"* Answer:
**94.5 %** — confident, correct, and useless. It was a count, and the MPI copies are the **tiny**
ones. Splitting the same data **by time** inverted the conclusion completely (81 % of the *time* is
the model's own rails, not MPI). I was one step from reporting "CUDA-aware MPI is host-staging the
halo — biggest lever in the campaign", which is **false**.

**L85: an attribution by COUNT and an attribution by TIME can point at opposite culprits. Always
weight by the quantity you actually intend to spend.** This is L81/L82 (profilers localise, never
attribute) wearing a new hat.
