# M9 P5 — the lean wide halo (Danilov's variant): build it and measure it

**Status: designed, not built. This is the next task.** Written 2026-08-06 at the end of session 4.

## Why

S. Danilov's comment (2026-08-06) makes two points, and answering the second one properly changed
what the experiment is:

> *"In the delayed exchange, calculations have to be extended over the K−1 ring nodes instead of
> calculations on owned nodes. This will remove most of errors, and will leave only the last bit
> errors."*
> *"If we sacrifice this exactness, and recompute what is possible to recompute, exchanging the
> minimum of what needs to be exchanged, what is the possible gain in performance?"*

He is right that extending the computation over the ring removes the imprint — we measured that
configuration at 0.87 against 0.85 for two identical runs (§6.4 of the report). What he describes
is our wide halo. **But his implementation and ours differ in a way that is not about exactness at
all, and that difference is probably worth more than everything else in this campaign.**

## The actual difference, and why it is promising

`ice_mevp_double.F90` extends the **loop bounds** of the existing kernels
(`clength = myDim_elem2D+eDim_elem2D+eXDim_elem2D`, `clen = myDim_nod2D+eDim_nod2D`). It adds
**no new kernels**.

Ours launches separate ghost kernels every sub-cycle — `maevp_stress_w` (`fesom_ice_maevp.cpp:662`),
`maevp_node_solve_w` (`:944`) or `maevp_node_solve_div_w` (`:1070`), and in the divergence form
`maevp_gather_div_w` (`:776`) — i.e. **2 extra launches per sub-cycle in the classic form and 3 in
the divergence form, so 240 or 360 extra launches per model step** at 120 sub-cycles.

That lands on the one thing this sub-cycle cannot afford. §4 of the report: of the 8.2 ms
ice-dynamics phase at CORE2 np8, only ~1.25 ms is arithmetic, spread over ~480 launches per step.
**L107 already said the exact halo "relocates work rather than removing it" and that ②/④ leave
`icedyn` busy flat while cell ⑤ takes it down 71 % — the extra launches are that relocation, named.**

So the prize is the gap between the wide halo (−11…−29 % of the ice cost) and the delayed exchange
(−37…−58 %), and the lean variant should recover part of it *without* the imprint, because the ring
is still recomputed.

## Measurement in flight

`jobs/job_m9_nsys`, farc np16, 20 steps, classic vs wide (jobs 26738629 / 26738630). Classic
returned **273 656 kernel launches over 20 steps = 13 683 per step** (whole model). The wide trace
was still queued when the session ended. The comparison bounds the prize before any code is
written — do not skip it.

## Build

Not a new scheme; a restructuring of the wide path that also drops the exactness machinery.

1. **Fuse each `*_w` ghost kernel into its owned twin.** The ghost bodies are documented as
   character-for-character copies of the owned ones with the views swapped, so each pair becomes
   one `RangePolicy<>(0, owned + ghost)` with an index branch selecting the `_g` arrays.
   Sites: `:446` `aux_init`, `:457` `node_pre`, `:476` `elem_pre`, `:507`, `:518`, `:662` `stress`,
   `:776` `gather_div`, `:944`/`:1070` `node_solve`, `:1157` `edge_bc`.
2. **Drop the owner-order gather** (`gath_ptr/gath_elem/gath_k`) and assemble the ghost node
   right-hand side the way the owned path does. This is what forfeits bit-identity, deliberately.
3. **Drop the 11-field prestep ship at $K$=2**, where the ring lies inside the halo FESOM already
   maintains. For $K>2$ it must stay: rings 2…K do not exist in the data structures, so those
   fields are absent rather than last-bit different.

Knob: `FESOM_SPEED_EVPWIDE_LEAN=1`, default off, so the certified exact path is untouched.

## Gates — the class changes, so the ladder does

- **NOT bit-identical by design.** The L0/L1 byte rungs do not apply to the lean path; the
  knob-off byte gate still does, and still must pass.
- 🔴 **The imprint test is now a gate, not a diagnostic.** `scripts/m9_partition_artefact.py` on a
  60-day fArc pair must return ~0.85–0.90, i.e. the control value. If it comes back above ~1.2 the
  variant has reintroduced staleness somewhere and is worthless regardless of its speed.
- Accuracy against the **exact wide halo** (not against classic): the difference should be
  last-bit in character — no structure, no growth with rank count.
- CUDA-vs-Serial fidelity, options ×3 as usual.

## What to report either way

The answer to Danilov's question is a number: the lean variant's ice-cost saving against our exact
wide halo's, at the same operating points. If it recovers most of the gap to the delayed exchange,
his design is the right one and ours was carrying avoidable machinery. If it does not, the ghost
*arithmetic* rather than the launches is the cost, and that is worth knowing too — it would mean
the exact halo's price is intrinsic.
