# M9 session 5 — build and measure the lean wide halo

Work in the worktree **`/home/a/a270088/port_kokkos_ice`** (branch `m9-mevp-double`, ~40 commits,
LOCAL, not pushed). A session started here gets a different memory index — point it at
`~/.claude/projects/-home-a-a270088-port-kokkos/memory/MEMORY.md`.

**Read first:** `docs/plans/20260806-m9-P5-lean-wide-halo.md` (the design, with the exact kernel
sites), then `docs/plans/20260805-m9-HANDOFF.md` §3c.

## The task

Build the wide-halo variant S. Danilov describes and measure it. His `ice_sergey/ice_mevp_double.F90`
extends the **bounds** of the loops that already exist; ours launches separate ghost kernels —
2 extra per sub-cycle in the classic form, 3 in the divergence form, so **240–360 extra launches per
model step** on a phase where only 1.25 ms of 8.2 ms is arithmetic over ~480 launches. That is the
last unmeasured term, and it is plausibly worth a large share of the gap between the wide halo
(−11…−29 % of the ice cost) and the delayed exchange (−37…−58 %) — obtainable **without** the
partition imprint, because the ring is still recomputed.

Three steps, in order:

1. **Finish the launch measurement before writing code.** `nsys` traces classic vs wide on fArc
   np16 are jobs **26738629 / 26738630**; classic returned 273 656 launches over 20 steps
   (13 683/step), the wide trace was still queued. Compare them. This bounds the prize.
2. **Build it** behind `FESOM_SPEED_EVPWIDE_LEAN=1`, default off: fuse each `*_w` ghost kernel into
   its owned twin behind an index branch, drop the owner-order gather, drop the 11-field prestep
   ship at K=2 (at K>2 it must stay — rings 2…K are absent from FESOM's data structures, not
   last-bit different). Sites are listed in the plan.
3. **Measure** at the four operating points — CORE2 1 node/np4, fArc 4/np16, DARS 8/np32, NG5
   16/np64 — clean and with `PHST=1`, against the exact wide halo at the same points.

## The gate that matters

The lean path is **not bit-identical by design**, so the byte rungs do not apply to it (knob-off
byte still does). That removes the safety net, and the replacement is:

🔴 **`scripts/m9_partition_artefact.py` becomes a GATE.** On a 60-day fArc pair it must return an
edge/interior ratio of ~0.85–0.90, the control value. Above ~1.2 the variant has reintroduced
staleness and is worthless whatever its speed. The reference runs already exist:
`/work/ab0995/a270088/port2/m9/clim/art_farc_g16_{standard,standard_rep,wide8}`, produced by
`jobs/job_m9_climate1yr` with `IOCFG=jobs/io.config.m9_daily_ice`, `NSTEPS=5760`.
🔴 Always run the **placebo partition** (`--distnpes`): a CORE2 np256 test gave 1.66 and a partition
the runs never used gave 1.63, i.e. mostly geography. The GPU 16-rank case is the clean one.

## State

- Frozen bins: serial `b_a300277a`, CUDA **`b_509a5771`** (has the `[evpwide-wire]` per-ship split).
- Deliverables to update when the number lands: `docs/report/danilov_mevp_report.pdf` (§6.3 and
  §8), `docs/report/reply_danilov.pdf` — which already commits us to doing this and reporting
  either way — and `onepager.pdf`. Figures are built by `make_icecost_fig.py` and
  `make_artefact_fig.py`; `make_report_figs.py` no longer makes report figures.
- **`ic_ng5` (job 26734334) may still be queued.** It is the fourth mesh of Figure 1; when it
  lands, rerun `scripts/m9_collect.py` then `docs/report/make_icecost_fig.py`.
- Quote every timing **twice**: model step and ice cost (ice + icedyn + iceadv, busy *and* wait).
  Reporting only the step is what made us describe the divergence form as a null result when on
  CPU at ≥512 cores it removes 2–8.5 % of the ice.
- Every mesh at its **operating point**, not the largest node count available. CORE2 on 16 nodes is
  45 % slower than on 2 and flatters every option.

## What to report

The answer to Danilov's question is one number: the lean variant's ice-cost saving against our
exact wide halo's, at the same operating points. If it recovers most of the gap to the delayed
exchange, his design is right and ours carries avoidable machinery. If it does not, the ghost
*arithmetic* rather than the launches is the cost — which means the exact halo's price is
intrinsic, and that is worth reporting too.
