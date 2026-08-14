# M12b wide halo — HANDOFF after session 4 (2026-08-14 late)

**You are a fresh session in the worktree `~/port_kokkos_wh`, branch `m12b-widehalo`.**
🔴 Your auto-memory index is worktree-scoped — read the MAIN index first:
`/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/MEMORY.md`, then
[[project-m12b-widehalo]].

**Read in this order:** this file → `docs/WIDEHALO_M12B.md` (§3/§3e/§3f root cause + fix + s3
board, §6 the s4 hygiene closure, §7 the deep-K arithmetic) → the SESSION 3 and SESSION 4
addenda at the tail of `docs/plans/20260814-m12b-widehalo.md`. The s2 **and** s3 handoffs are
history; s3's first action (the CUDA gate) is still the first action here, unchanged, because
the GPU queue never opened.

## 1. State in one paragraph

The rung is **CPU-complete and bitwise-exact free-running** (drift ≡ 0.0; both 3000-step screens
green; W6 CPU farc −1.8 %, dars +0.4 % wash) — that is s3 and it has not changed. Session 4 did
the two things that did not need a GPU: it **closed the `elem_nodes(-1)` hygiene item** (audit of
all 45 read sites, one real defect fixed, the containment invariant now measured at startup and
green at seven points from pi np2 to dars 8192) and it **measured the deep-K decision input that
the GPU queue does not gate** (how many partner ranks a K-ring zone touches). The GPU board did
not move: all five of the account's GPU job slots were held by another track's `nn_evalrun` jobs
for the whole session, with thirteen jobs queued ahead of ours. Bins: serial
`fesom_port_serial_4cc9eda4`, cuda `fesom_port_cuda_3a584dc9` (both in
`/work/ab0995/a270088/port2/m12b/bin/`, shas in `SHA256.m12b`). Tree clean.

## 2. 🔴 FIRST ACTIONS (in order)

0. ✅ **DONE in s4 — the W2 CUDA gate PASSED** (26960090): drift nonzero-steps=0, free and off
   arms both η=2.02 with identical prints, matching the M12 CUDA SE reference. The **W6 GPU pairs
   are already submitted** on the gate's binary — `26962396` (core2), `26962397` (ng5). So the
   first action is to **harvest those**: `loop timing` min-of-2 per arm, check the `it=` lines
   (trap 6), then read the wait-anatomy and bt-imbalance blocks the job prints at the end, and
   compare with §7's pre-registered band (**−2.1 % … −6.6 %** at CORE2 16N).

1. *(superseded)* If for any reason `26960090` needs re-running, harvest
   `/work/ab0995/a270088/port2/m12b/w2cu_26960090.out`: gate = `drift nonzero-steps=0` and
   healthy `it=` lines on all three arms. It is the FIRST-EVER CUDA run of the `ELEM2D_FULL`
   device-halo path and of the F-reconcile's per-step host round-trip — a crash there is
   diagnostic, not noise. If it has been cancelled or never ran, resubmit:
   `sbatch --export=ALL,BIN=/work/ab0995/a270088/port2/m12b/bin/fesom_port_cuda_3a584dc9 jobs/job_m12b_w2cuda`
2. **GREEN → the W6 GPU pairs**, the points the whole track was built for (CORE2 16N bt = 7.8 ms
   busy + 11.8 ms MPI wait of an 84 ms step):
   `sbatch --export=ALL,POINT=core2,BIN=/work/ab0995/a270088/port2/m12b/bin/fesom_port_cuda_3a584dc9 jobs/job_m12b_w6_gpu`
   and the same with `POINT=ng5`. 16 nodes each = AT the gpu cap; do not exceed it.
   🔴 Pin ONE binary for both arms of a pair. `3a584dc9` and s3's `674a53bc` differ only by the
   s4 startup census (byte-inert, proven on Serial), but never mix them inside a pair.
   - Drift small+flat but nonzero → judge against the SE CUDA floor (η ~1e-3,
     `docs/REFERENCE_RUNS.md`) before touching anything; `GEOCHK=2` probes work on CUDA too.
   - Drift GROWING → stop, do not run perf; instrument as in s3 (probe the READS, both
     components, halo-vs-owner).
3. **Run `scripts/m12b_wait_anatomy.py off=<off_ph1_dir> on=<on_ph1_dir>` on the W6 GPU pair's
   phasestats legs.** This is the s4 finding that reframes the whole deep-K question: on CPU the
   bt wait's elasticity to the exchange count is only **0.17–0.21**, because a quarter to a half
   of that "wait" is the block's own load imbalance (a rank's wait correlates with its own busy
   at −0.6) and even the busiest rank's floor has elasticity ~0.45. If the GPU behaves the same
   way, the rung is a ~2 % lever there and deep K is not worth building; if the GPU is
   latency-bound, K=2–4 buys another ~40 % over K=1. §7 has the pre-registered decision rule
   (build the extended-mesh layer only if the GPU floor elasticity ≥0.7 and imbalance share
   ≤0.3) and a pre-registered prediction band for the CORE2 16N pair itself: **−2.1 % to −6.6 %**.
4. **Harvest the M-sweep 26952126/27** — the independent check of the same elasticity on the
   baseline `se0`.
5. Then the deep-K decision (§7 has the arithmetic; it needs exactly the W6 GPU numbers). The
   Sergey packet is already updated (addendum 2 in `docs/report/M12_SERGEY_PACKET.md`).

## 3. What session 4 added

- **§6 of the reference — the hygiene closure.** All 203 `elem_nodes` references / 45 read sites
  audited. The model's discipline is: read `elem_nodes` only at owned elements or through
  `nod_in_elem2D` rows of owned nodes; both are `-1`-free *if* an element incident to an owned
  node has all three vertices local. That proviso is now MEASURED at startup
  (`audit_elem_nodes_unmappable` in `fesom_mesh.cpp`) with a hard abort if an owned row ever
  reaches a `-1` element — green at pi np2, CORE2 np8/np128, farc np128/2048, NG5 np128, dars
  8192. One real defect found and fixed: `fesom_forcing_analytical.cpp` read `geo_coord_nod2D`
  at a `-1` vertex over the full element extent (same defect class as the H0e one).
- **§7 — the deep-K arithmetic**, with `scripts/m12b_partner_growth.py` (job 26961509) as its
  measured input: the partner count per cumulative ring, i.e. whether the K=1 message halving
  survives K≥2. It does on GPU-sized subdomains (partner count flat out to ring 9) and it does
  NOT at the big-CPU points (dars 8192: 7.6 → 13.2 partners, messages/step stop falling past
  K=4). Predicted optimum at CORE2 16N GPU: K=2–4, ~1.5× the K=1 gain, saturating; at NG5/dars
  16N (zone ≤0.14 even at K=8) the optimum is deep, ~2×. The model's one unmeasured assumption
  is that bt wait tracks the message count — that is exactly what the queued M-sweep tests.
- **§8 — where the barotropic imbalance comes from, and the whole-step wait budget.** The bt
  block's imbalance is an owned-**element** effect (corr +0.96 farc 2048 / +0.85 dars 8192) that
  the node-balanced partition does not control: owned nodes are equal to 1 %, owned elements vary
  by 47 % / 22 %. A pre-registered test (job 26961927) confirmed the mechanism on compute and
  **refuted the first size estimate**: on M11's `nlev`-weighted partition the bt busy imbalance
  grew as predicted but the bt *wait* fell, because the wait is set by arrival skew from the 3-D
  phase. The currency that works is the critical path (TOTAL busy max 62.9 → 56.0 ms); revised
  prize for an element constraint ≈1.0–1.5 ms (1.5–2.2 %). **Side result worth having: the
  M11-certified farc partition is −6.7 % under SE** (0.0728 → 0.0679, min-of-2, same binary).
  Running the same decomposition on every phase gives the budget: at farc 2048, **70 % of the
  28.9 ms MPI wait (27 % of the step) is imbalance absorption, only ~7.5 ms is a message floor**;
  at dars 8192 it is ~45/55. Handed to M11 (multi-constraint partition: nodes AND elements) with
  a mechanism test pre-registered — job 26961927, predictions P1–P3 in §8.
- **Lessons L114–L119** in `docs/KOKKOS_PORTING_LESSONS.md` (the M12b track had none before).
- **Sergey packet addendum 2** — the two s3 findings, the amplification measurement, the rim
  algebra, the CPU board, questions 6–8.

## 4. Traps (s2's seven and s3's five still stand; s4 adds two)

1. **A cheap gate does not help when the block is `AssocMaxJobsLimit`.** The GPU association
   allows 5 concurrent jobs across ALL tracks on this account; shrinking walltime moves nothing.
   The levers are: wait, cancel one of your own pending jobs, or submit under another of the
   user's GPU associations (`ab1011_gpu`, `bk1341_gpu`, …) — the last one spends another
   project's budget, so it is the user's call (asked in s4; the answer was **wait**).
2. **A job that carries CORE2's `FESOM_SE_M=50` to another mesh dies at the SE startup CFL
   guard** (farc dt900 needs M≥82). Two s4 census legs died that way after printing what they
   were for. Per-mesh M: CORE2 50 · farc 90 · dars 20 · NG5 20. And NG5 at 128 ranks does not
   fit on ONE node — it was OOM-killed.

## 5. Where everything lives

Code `src/fesom_ssh_se.cpp` (rung) · `src/fesom_mesh.cpp` (scatter note + s4 census) ·
`src/fesom_forcing_analytical.cpp` (s4 fix) · jobs
`jobs/job_m12b_{fixgate,fix128,fixfarc,seed4,w2cuda,w6_cpu,w6_gpu,screen,w5b_disturb,probe2,s4_audit,s4_census,partners}`
· run dirs + evidence `/work/ab0995/a270088/port2/m12b/` (s4 jobs: audit 26961249, census
26961257/58, partners 26961509, partition split 26961927) · disturbance analysis is PAIRED at one rank count
(`scripts/m12b_disturbance.py --pair OFF ON`).
