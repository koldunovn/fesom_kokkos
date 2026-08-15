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

## 2. 🔴 STATE OF THE BOARD — the GPU rows are IN

| point | off | on | Δ |
|---|---|---|---|
| CORE2 16N GPU dt1800 M=50 | 0.0841 | 0.0744 | **−11.5 %** |
| NG5 16N GPU dt180 M=20 | 0.2461 | 0.2469 | +0.3 % (wash) |
| farc 2048 CPU | 0.0731 | 0.0718 | −1.8 % |
| dars 8192 CPU | 0.0976 | 0.0980 | +0.4 % (wash) |

The W2 CUDA gate passed first (drift 0.000000e+00 every step). All eight GPU timing legs are alive
and end at the same physical state per point. **The rung ships as a GPU recommendation at
small-per-rank-size points** — the payoff is a per-rank-size law, not a mesh law, and it sits at
the strong-scaling limit.

### FIRST ACTIONS

1. **Read §7b before quoting anything from §7.** The pre-registered band was −1.5…−2.9 % at
   CORE2 16N and the measurement was −11.5 %. The failed term is named (bt_busy, calibrated on
   CPU where the pack is a memcpy; on GPU the pack is kernels+copies and lands in busy at ~51 µs
   per exchange). §7's own prediction table is marked SUPERSEDED — do not quote it.
2. **The deep-K "no" is RETRACTED (§7c).** Rebuilt on the measured staging law, K=2–4 plausibly
   adds ~3 more points at CORE2 16N. It is a two-point extrapolation. Before any §4
   extended-mesh build, run the cheap test: a **k-periodic η exchange arm** (exchange every k
   substeps using the existing ring-1 data) traces the staging curve at K=2 and 4 with no new
   mesh layer. That is the highest-value next job in this track.
3. **CORE2 4N GPU is the obvious missing row** (census says +2.0 % ring cost, bt is 21 % of the
   step, 7 928 nodes/rank — between the two measured points). `sbatch --nodes=4 --ntasks=16
   --export=ALL,POINT=core2,BIN=/work/ab0995/a270088/port2/m12b/bin/fesom_port_cuda_674a53bc
   jobs/job_m12b_w6_gpu` would place the per-rank-size law on a third GPU point.
4. Then the write-up: the M12b recommendation, and the Sergey packet's addendum 2 needs its
   GPU paragraph replaced with the measured board.

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
