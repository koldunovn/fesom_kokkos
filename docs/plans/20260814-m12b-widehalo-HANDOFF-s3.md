# M12b wide halo — HANDOFF after session 3 (2026-08-14 night)

**You are a fresh session in the worktree `~/port_kokkos_wh`, branch `m12b-widehalo`.**
🔴 Your auto-memory index is worktree-scoped — read the MAIN index first:
`/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/MEMORY.md`, then
[[project-m12b-widehalo]].

**Read in this order:** this file → `docs/WIDEHALO_M12B.md` §3/§3e/§3f (root cause, fix,
verification board — ALL numbers live there) → the SESSION 3 ADDENDUM at the tail of
`docs/plans/20260814-m12b-widehalo.md`. The s2 handoff is bannered SUPERSEDED — do not act on it.

## 1. State in one paragraph

The s2 instability is ROOT-CAUSED and FIXED. Two coherence repairs, both default-on: a per-step
`H0e` exchange (`FESOM_SE_H0E_XCHG=1` — halo H0e was η-class wrong via an `eta0[-1]` OOB read at
−1 halo vertices) and owner-wins `Fbt` reconciliation over the ~0.55 % multi-claimed elements
(`FESOM_SE_WIDE_RECON=1`, rung-only, tag 2303 — the free interface iteration amplifies ANY
inconsistency ×5.35/step at farc 2048, so the seed had to be exactly zero). With every k3 input
single-valued the rung is **BITWISE-exact free-running** (drift ≡ 0.0: np8 ×25, np128 ×300,
farc-2048 ×50), both 3000-step screens are green (farc η=2.05, np128 η=1.89 = the SE references),
W0 rc=0, W1 resurrected, W5b rc=0. W6 CPU: **farc 2048 −1.8 %** (honest, replaces the retracted
−8.9 %), **dars 8192 +0.4 % wash** — the CPU latency share bounds the payoff, as Sergey said.
Bins: serial `fesom_port_serial_90c5c12d`, cuda `fesom_port_cuda_674a53bc` (commit f95aa73;
`/work/ab0995/a270088/port2/m12b/bin/`, shas in `SHA256.m12b`). 13 commits, tree clean.

## 2. 🔴 FIRST ACTIONS (in order)

1. **Harvest the W2 CUDA drift gate, job 26960090** (`/work/ab0995/a270088/port2/m12b/w2cu_26960090.out`;
   it was PENDING behind the account's 5-GPU-job limit + other tracks' backlog when s3 ended).
   Gate: `drift nonzero-steps=0` and healthy `it=` lines on all three arms. ⚠️ This is the
   FIRST-EVER CUDA run of the `ELEM2D_FULL` device-halo path AND of the F-reconcile's per-step
   host round-trip — a crash/abort here is diagnostic, not noise.
   - GREEN → submit the W6 GPU pairs (the track's motivating points — CORE2 16N bt = 7.8 ms busy
     + 11.8 ms MPI wait of an 84 ms step):
     `sbatch --export=ALL,POINT=core2,BIN=/work/ab0995/a270088/port2/m12b/bin/fesom_port_cuda_674a53bc jobs/job_m12b_w6_gpu`
     and the same with `POINT=ng5`. 16 nodes each = AT the gpu cap, allowed, do not exceed it.
   - Drift small+flat but nonzero → judge against the SE CUDA floor (η ~1e-3,
     `docs/REFERENCE_RUNS.md`) before touching anything; a CUDA-only channel (helem sync timing,
     device-halo pack) is possible — the GEOCHK=2 probes work on CUDA too.
   - Drift GROWING → stop, do not run perf; instrument as in s3 (probe the READS, both
     components, halo-vs-owner).
2. **Harvest the s2 M-sweep jobs 26952126/27** (same queue) — they measure the BASELINE se0
   latency structure (does bt wait track exchange count); input for the deep-K decision.
3. Then the remaining board rows + verdicts (below).

## 3. What remains for the track after the GPU board

- **W6 GPU verdict** — the rung ships as a GPU recommendation if CORE2/NG5 16N show real gains
  (census says the redundant compute is +1.2…+6.1 % there, and the bt wait share is 3-8× the CPU
  points'). Quote loop-only s/step, min-of-2, same-day, check `it=` lines (trap 6).
- **Deep-K decision** (plan Task 8 / §4): now UNBLOCKED — the s3 induction transfers to deep K
  only if ring-K inputs are owner-coherent the same way (H0e over the K-ring extent + F still
  reconciled; the §4 extended-mesh layer ships owner bytes by construction, compatible). Decide on
  the W6 GPU numbers + the M-sweep latency answer; K=8 node-zone costs are in §2 of the reference.
- **Sergey packet** (`docs/report/M12_SERGEY_PACKET.md` exists on the m12 side): add the two s3
  findings — (a) the halo-H0e/`eta0[-1]` defect + fix, (b) the measured exponential interface
  amplification (×5.35/step) that makes ANY nonzero seed fatal — plus the rim algebra and census
  numbers per the plan's Post-Completion note.
- **Merge path unchanged**: M12 merges to main first, this branch rebases. 🔴 The fixed SE
  trajectory differs from the `se0` bins at ROUNDING CLASS (H0e exchange on the certified path) —
  same-binary pairing for any future SE comparison; the M12 memory file carries this warning.

## 4. Traps for this session (the s3 additions; s2's seven still stand)

1. A probe that compares only myDim copies, or only the x-component, certifies nothing about the
   bytes actually READ — probe halo-vs-owner, both components (`FESOM_SE_WIDE_GEOCHK=2`).
2. `elem_nodes` at HALO elements contains −1 for eDim edge-neighbours too (NOT only eXDim);
   every consumer must guard. Other 3-D consumers were NOT audited (open hygiene item).
3. `FESOM_SE_WIDE_SELFCHECK=1` (0.0-abort) is the proof instrument; with `FESOM_SE_WIDE_RECON=0`
   it trips on the 3-D ulp from step ~4 — any long run uses `=2`.
4. The wire is **M+3 per step** (subcycle + η coherence + H0e + F-reconcile) vs certified 2M —
   quote M+3, not M+1 (bins built before commit a48b0e5 still PRINT the old M+1 line).
5. `FESOM_SE_H0E_XCHG=0` / `FESOM_SE_WIDE_RECON=0` re-arm the instability — diagnostic arms only.

## 5. Where everything lives

Code `src/fesom_ssh_se.cpp` (+ scatter comment fix in `src/fesom_mesh.cpp`) · jobs
`jobs/job_m12b_{fixgate,fix128,fixfarc,seed4,w2cuda,w6_cpu,w6_gpu,screen,w5b_disturb,probe2}` ·
run dirs + evidence `/work/ab0995/a270088/port2/m12b/` (key jobs: probes 26959682/26959760/26959826,
exponential 26959818/19, green wave 26959980/81/82, screens 26960088, W5b 26960089, W6 CPU
26960156/57) · disturbance analysis is PAIRED at one rank count
(`scripts/m12b_disturbance.py --pair OFF ON`; the job script is fixed).
