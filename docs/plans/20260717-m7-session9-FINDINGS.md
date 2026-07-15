# M7 session 9 — findings (written as the work happens)

*Branch `m7-speed` from `11ebe37` (the session-9 PROMPT). Plan: PROMPT §2 (four background
measurements) + §3 (C.1 REDISWEEP foreground). All rules per PROMPT §0.*

## 1. THE FOUR BACKGROUND MEASUREMENTS — submitted, pre-registrations restated

All six jobs pinned `BIN=m7/bin/h11/fesom_port_cuda` (`d74d31b4`), submitted ~10:55 CEST,
walltimes sized to the measured session-8 twins (rule 0.7: ncu twin ran 3:05, census 7:54,
1-yr gate 20:35, anchor 14:22).

| job | what | pre-registration (from the committed PROMPT §2.1) |
|---|---|---|
| 26267146 | ncu local-memory counters, 7-family regex | none — this is the INPUT to C.1's pre-reg. skip/count re-derived from the h10 census: the 7-family regex matches **71 launches/step** (measured, not guessed — the default skip 650 was tuned for the ~62/step default regex and would have consumed 10 of 12 steps), so `NCU_SKIP=710 NCU_COUNT=142` = exactly steps 11–12 |
| 26267147 | gap300_h11 census | the five SSH/hbar rows (14.8 ms class) GONE; halo self-gap rows grow by the four new NOD2D exchanges' wait; ice-thermo ≈4.3 ms class persists |
| 26267148 | NG5@8N GPU 300-step | **0.407 ±0.5 %** |
| 26267149 | NG5@16N GPU 300-step | **0.2650 (−1.4 ±0.5 %) ⇒ SYPD@dt240 ≈ 2.40**; below −0.5 % ⇒ H.9 decayed like a kernel lever |
| 26267150 | dars@8N GPU 150-step (L95) | **0.201 ±0.7 %** |
| 26267151 | h11 1-yr climate gate | sst 1.00000 · sss 0.99996 · ssh 1.00000 · a_ice 0.99997; job state FAILED expected (teardown wart 0.10a) — judge by the compare table |

## 2. HARVEST (filled as jobs land)

### 2.1 ✅ Job 26267146 — the spill-traffic measurement (h11 `d74d31b4` ✓, `FESOM_SPEED=1`
announce ✓, 142/142 launches captured = steps 11–12, ranks on l50xxx = a100_80)

Per-step, rank-0 GPU (`ncu_spill_h11/sol.csv`; local = `l1tex..mem_local_op_{ld,st}`, the spill
traffic itself; busy = the census number, un-perturbed):

| kernel (grid) | busy ms/step | local GB/step | L2 GB/step | DRAM GB/step | **local/L2** | local/DRAM |
|---|--:|--:|--:|--:|--:|--:|
| redi_expl NODE (3607) — **C.1** | 36.8 | 10.25 | 85.7 | 35.5 | **12.0 %** | 28.8 % |
| impl_ale TDMA (3607) — C.2 | 32.6 | 20.37 | 99.2 | 45.2 | **20.5 %** | 45.1 % |
| impl_vert_visc (7192) | 13.4 | 18.75 | 41.6 | 17.2 | **45.1 %** | >100 % (L2-fed) |
| momentum_adv node (3607) | ~12 | 11.81 | 35.7 | 12.4 | 33.1 % | 95 % |
| momentum_adv elem (10799) | ~12 | 7.68 | 37.5 | 11.1 | 20.5 % | 69 % |
| fer_solve_gamma (3607) | 9.4 | 8.98 | 25.6 | 9.9 | 35.0 % | 91 % |
| FCT `zal_a34` (3607) — B∩C | 22.5 | 2.71 | 44.9 | 19.3 | **6.0 %** | 14 % |
| pressure_bv (3607) | 11.4 | 3.14/launch | 21.1/l | 9.1/l | 14.9 % | 34 % |
| FCT all other lambdas | ~160 | **0.000** | — | — | 0 % | — |

**Two findings that re-shape package C, recorded before the C.1 A/B:**
1. **C.1's spill share is the LOWEST of the big spillers (12 % of L2).** Its #1 pool rank was
   busy-time, not spill-intensity — the "L2 ≈ 2.4× DRAM amplification" is mostly the gather's
   irregular re-reads, not spills. The audit prior ("spills 30–50 % of DRAM ⇒ 1.7–2.5 %") was
   HIGH; the measurement replaces it (§2.2).
2. **The spill-intensity ranking inverts the busy ranking**: `impl_vert_visc` is 45 % local
   (18.8 of its 41.6 L2 GB/step!), `fer_solve_gamma` 35 %, `momentum_adv` 33/21 % — while the
   two biggest-busy rows (redi_expl 12 %, zal_a34 6 %) are the LEAST spill-bound. The B∩C
   overlap is small in spill terms (2.7 GB/step). Re-rank for C.3+ is in §4.

### 2.1b 🔴 Census attempt 1 (26267147) LOST AT TEARDOWN — the 0.10a wart now has a NAME

The run itself completed (loop timing **0.6633 s/step traced** — vs h10's 0.6771 traced, −2.0 %,
consistent with SSHRAILS) but **nsys wrote NO report**: every rank SIGABRTed at teardown and
slurmstepd CANCELLED the step, SIGKILLing rank 0's nsys mid-finalization. The abort message
identifies the wart precisely:

> `what():  Kokkos allocation "smooth.work" is being deallocated after Kokkos::finalize was called`

**The 0.10a teardown SIGABRT = H.7 SMOOTHSCRATCH's persistent scratch View is a static that
outlives Kokkos::finalize.** Every full-blessed FAILED job state since h7 is this. The h10
census survived by RACE (its report generated before/despite rank 0's own abort; only task 0
exited 134 there — this time all 16 aborted and one landed first).

- Measurement-level fix (keeps the frozen h11 bytes): `srun --kill-on-bad-exit=0` in
  `job_m7_hostprof` — aborting ranks can no longer cancel rank 0's nsys finalization; the abort
  is post-MPI_Finalize/post-output so there is no hang risk. Census resubmitted: **26267784**.
- Root-cause fix (a `Kokkos::push_finalize_hook` releasing the scratch, unconditional like the
  getcoeffld precedent): **NOT applied mid-ladder** — h12 is frozen and 6 gates already ran on
  its bytes. Queued for the next rung; it will make full-blessed job states readable again.

### 2.2 🔴 C.1 PRE-REGISTRATION (recorded BEFORE the A/B was submitted)

Model: saving = node-kernel busy (36.8 ms/step census) × spill share of the binding traffic
level. Floor = local/L2 share (12.0 %) = **4.4 ms/step = −0.65 %**. Ceiling = local/DRAM share
(28.8 %) = **10.6 ms/step = −1.6 %** — physical because the spill working set is 2.36 GB/launch
(461k threads × 5,120 B) ≫ 40 MB L2, so spill lines cannot cache; but the kernel is not
BW-saturated (1129 GB/s DRAM ≈ 58 % of a100_80 roof — latency/mixed-bound), so full DRAM-share
conversion is optimistic.

**⇒ C.1 REDISWEEP pre-registered: −1.2 % at NG5@4N (35-step A/B, h12, base leg
`FESOM_SPEED=1;FESOM_SPEED_REDISWEEP=0`), floor −0.65 %, ceiling −1.6 %.**
Anchor formula (H.8's): h12 anchor = 0.6503 × (1 + Δ_A/B), tolerance ±0.5 %.
Kernel-class lever ⇒ expect ~56 % retention at 16N.

## 3. C.1 REDISWEEP — built, STACK==0 on first compile, ladder submitted

- **The lever** (`src/fesom_gm.cpp`): new file-scope `fesom_gm_redi_ver_node_sweep` — single
  bottom-up column sweep, O(1) carried scalars (`zbar_below/z_cur/tx_cur/ty_cur/flux_below`),
  no local arrays. Knob branch INSIDE `fesom_diff_ver_part_redi_expl_kk` (both tracers flip
  together, trap 7). The sweep is its OWN enclosing function so `m7_spill_pool.py` gives it its
  own row forever — inside the old function, the MAX-across-instantiations rule would have
  pinned the row at the old kernel's 5,120 B (an L96-class trap for future censuses).
- **Bit-identity argument** (the banner comment in source, trap list §3.3 worked through):
  depth recurrence already bottom-up (same adds, same order); per-level gather is
  level-independent, evaluated once per level, one level ahead of consumption; vd_flux rolling
  pair with EXPLICIT boundary zeros (vd_flux[nle], vd_flux[ule] were init-zeros never written —
  the literal 0.0 enters the apply unchanged); the apply loop is order-free across nz
  (iteration nz writes only vals[nz] +=, reads only its own level) ⇒ bottom-up apply
  bit-identical per element; zbar_n[ule] was computed-but-never-read (flux loop reads zbar_n on
  [ule+1, nle−1] only) — the sweep omits the dead value; both guards (`nle<=ule` return,
  `av>0&&hn>0`) kept verbatim; host C twin untouched.
- **✅ Acceptance (PROMPT §3.3-4, login node, before any GPU hours):
  `fesom_gm_redi_ver_node_sweep` REG **54** / STACK **0**** (old kernel 58/5,120 stays as the
  knob-OFF path; trxy 48/0 and zero 10/0 clean, as expected).
- Frozen FIRST (session-8 practice): `m7/bin/h12` CANDIDATE, CUDA **`501cc4f6`** / Serial
  **`86079b86`** (PROVENANCE.txt in the dir; gates run these bytes).

### 3.1 Gate ladder (8 gates — no guard-abort: REDISWEEP has no dependency set)

| job | gate | expectation |
|---|---|---|
| 26267366 | knob-OFF byte (Serial, live = h12) | diff_snap rc=0 |
| 26267368 | FORCE_SERIAL byte proof, REDISWEEP isolated | rc=0 — **THE gate** (rule 0.3 inverted: the lever claims bit-identity) |
| 26267369 | FORCE_SERIAL byte proof, full blessed | rc=0 |
| 26267371 | CUDA fidelity, isolated | PASS at the climate-close floor |
| 26267372 | CUDA fidelity, full blessed | PASS |
| 26267373/74/75 | options ×3 (TKE / mEVP / zstar vs own oracles) | PASS ×3; zstar control must reproduce Kv max\|Δ\| = 9.869e-02 EXACTLY (L79/h11 value) |

Then: the §2.2 pre-registered A/B → 300-step h12 anchor (a100_80) → ledger + freeze.

### 3.2 Gate results (filled as they land)

| gate | result |
|---|---|
| 26267366 knob-OFF byte | ✅ rc=0, ALL FIELDS BIT-IDENTICAL |
| 26267368 FORCE_SERIAL byte proof, isolated | ✅ **diff_snap rc=0 — THE gate. The sweep is PROVEN bit-identical** (announce `FESOM_SPEED_REDISWEEP = ON` fired ✓ — L80 checked; a dead knob would pass this gate vacuously) |
| 26267369 FORCE_SERIAL byte proof, full blessed | ✅ diff_snap rc=0 (run rc=134 = the known full-blessed teardown wart, content complete) |
| 26267371 CUDA fidelity, isolated | ✅ PASS (T 1.494e-03, S 5.160e-04 — the climate-close floor) |
| 26267372 CUDA fidelity, full blessed | ✅ PASS (T 1.136e-03) |
| 26267373 options TKE | ✅ PASS |
| 26267374 options mEVP | ✅ PASS |
| 26267375 options zstar | ✅ PASS — **both standing controls reproduce TO THE DIGIT: Kv 9.537e-02, Av 9.869e-02** (all other fields at atomic-scatter noise, far under ceiling) ⇒ the lever adds nothing to the zstar leg |

**⇒ LADDER 8/8 GREEN.** Number-conflict resolved in the open: session 8's "zstar control
9.869e-02" note mislabeled the WORST-of-table (which is **Av**) as Kv; the L79 Kv control was
and is **9.537e-02**. Both values are standing controls now; both held exactly on h11 AND h12.

## 4. C.2/C.3 AUDIT NOTES (source read while the ladder ran)

- **C.2 `diff_ver_part_impl_ale_kk`** (fesom_tracer_diff.cpp:403; 6 arrays after M5.24's
  cp→c/tp→tr aliasing): the PROMPT's "O(1) impossible for Thomas" holds (cp/tp must persist for
  back-substitution), **but the REDISWEEP structure transfers**: the coefficient construction
  (a/b/c + RHS tr, incl. the zinv1→zinv2 carry, the BC and sw adds) can fuse into a BOTTOM-UP
  pass with O(1) depth carry — at iteration nz the depth recurrence produces exactly the
  Z_n[nz−1] the coefficients need (same discovery as the redi sweep). That eliminates
  zbar_n/Z_n as arrays ⇒ **4 arrays = 4,096 B/thread (−33 %)**, valid for all nl ≤ NL_MAX, and
  every expression keeps its form ⇒ byte-provable again. **Bonus found: the :462-466 zero-init
  stores 6,144 B/thread over ALL 128 slots — ×461k threads ×2 launches ≈ 5.7 GB/step, roughly
  HALF the kernel's measured local-store traffic (11.1 GB/step) is pure init stores.** Init only
  the [nzmin..nzmax] slots actually read (verify no out-of-range reads first) — a cheap
  sub-lever even before the fusion.
- **`fesom_impl_vert_visc_kk`** (fesom_momentum.cpp:826; 7 arrays = 7,168 B after M5.24): same
  family (bottom-up depth recurrence :868-878 + Thomas on u,v) ⇒ same fusion applies
  (zbar_n/Z_n eliminable ⇒ 5 arrays, −29 %) + the same zero-init sub-lever (:869-870, partial).
  With its 45 % local/L2 share it is now a live C.3 co-candidate with C.2 — decide on the C.1
  realized factor.

## 4. C.2+ RE-RANK (preview, from §2.1 — final after C.1 lands)

By recoverable ms/step (busy × local/L2 floor share … busy × local/DRAM ceiling share):
`impl_ale` 6.7…14.7 · `redi_expl` 4.4…10.6 (=C.1) · `impl_vert_visc` 6.0…13.4(!) ·
`momentum_adv` 3.2…9.7 · `fer_solve_gamma` 3.3…8.6 · `zal_a34` 1.3…3.2 · `pressure_bv` 1.7…3.9.
**`impl_vert_visc` (13.4 ms busy, 45 % local) may outrank C.2's TDMA** on floor terms and is a
per-element column kernel (7 NL_MAX arrays, 7,168 B) — audit whether the O(1)-sweep pattern
applies before committing to the TDMA audit order. The B∩C overlap (zal_a34) is small in spill
terms — fixing it under C would remove little of B's remaining case (B stays parked on its own
merits, f=0.242).
