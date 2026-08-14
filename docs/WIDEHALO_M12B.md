# M12b — wide halo for the SE barotropic subcycle (`FESOM_SE_WIDE`)

**Status 2026-08-14: K=1 rung implemented and gated on Serial; GPU board pending the maintenance
window.** Plan + full gate record: `docs/plans/20260814-m12b-widehalo.md`. SE reference:
`docs/SSH_SE_M12.md`.

## What it is

The SE block does **two halo exchanges per substep** — η over `NOD2D`, Ū over `ELEM2D` — at
M = 20…90 substeps per step, and at CORE2 16N GPU that block is 7.8 ms busy + **11.8 ms MPI wait**
of an 84 ms step. The K=1 rung removes half of them: **η is computed on the ring-1 (eDim) nodes
instead of being received**, and only Ū is exchanged, over the *existing* `com_elem2D_full` list.

Nothing new is allocated in the node index space — ring-1 nodes *are* the eDim halo slots. The
element state grows to the full local extent (`myDim+eDim+eXDim`) because `com_elem2D_full`
delivers element rings 1 **and** 2 (it replaces `com_elem2D`'s coverage rather than extending it,
which is exactly why the rung is a message *replacement*, not an extra message).

Each ring-1 node needs its div-CSR row, and that row **cannot be assembled locally at any
precision**: 40–44 % of the edges incident to a ring-1 node are absent from the local edge list
(an edge joining two ring-1 nodes touches no owned node). So the owner ships its row — the final
post-`inv_a` coefficients, in its own accumulation order — once at startup, over the `com_nod2D`
partner structure, tags 2300–2302.

## Knobs

| knob | default | notes |
|---|---|---|
| `FESOM_SE_WIDE` | 0 | `1` = the rung; **`≥2` aborts** — deep K needs the extended-mesh layer (§4). Requires `FESOM_SSH_MODE=se`; set while SE is off it prints the L80 dead-knob note |
| `FESOM_SE_WIDE_SELFCHECK` | 0 | `1` = compute ring-1 η locally **and** exchange, diff, abort on nonzero; `2` = report without aborting (growth curves) |
| `FESOM_SE_WIDE_GEOCHK` | off | debug: min/max over holders of `elem_area`, the viscosity stencil size, `Fbt`/`H0e`, `Ubt` — the probe that produced §3 |

Startup aborts, both from measured assumptions rather than belief: an owned element whose
viscosity neighbour lands in the eXDim tail, and any `com_elem2D_full` rlist that fails to cover
`[myDim, myDim+eDim+eXDim)` exactly once.

## 1. What it costs and what it buys (census, 11 operating points)

`scripts/m12b_ring_census.py`, jobs 26949302 / 26951957; CORE2, farc, dars, NG5; 16–8192 ranks.
Containment holds at **all 11 points** (0 elements incident to a ring-1 node missing locally), and
the eXDim viscosity-neighbour hazard fires at **none** of them.

| point | ranks | ring-1 redundant compute | messages/substep | doubles/substep |
|---|---|---|---|---|
| NG5 16N GPU | 64 | **+1.2 %** | ×0.500 | ×1.34 |
| dars 16N GPU | 64 | +1.7 % | ×0.500 | — |
| farc 16N GPU | 64 | +2.9 % | ×0.500 | ×1.44 |
| CORE2 4N GPU | 16 | +2.0 % | ×0.500 | ×1.45 |
| CORE2 16N GPU | 64 | +6.1 % | ×0.500 | ×1.46 |
| NG5 CPU | 8192 | +13.8 % | ×0.500 | — |
| farc CPU | 2048 | **+19.7 %** | ×0.503 | ×1.49 |
| dars CPU | 8192 | **+28.2 %** | ×0.505 | ×1.41 |

**Messages halve because `com_elem2D_full` reaches the same partner ranks as `com_elem2D`** — more
entities per partner, not more partners. That is a claim about `rPEnum`, so it was counted, not
assumed. The GPU/CPU split is Sergey's "GPU real advantage, CPU just better scaling" as a number.

## 2. Rim algebra (what deep K would cost)

With Ū valid to element ring E and η to node ring D: η^{m+1} is valid wherever its incident
elements are (D′ = E), and Ū^{m+1} loses one ring to the viscosity's edge-neighbour term
(E′ = E−1). **K substeps therefore need K rings, not the 2K−1 of mEVP** — mEVP carries element σ
across substeps, SE's η/Ū alternation does not. Cumulative node zone / owned:

| point | K=2 | K=4 | K=8 |
|---|---|---|---|
| NG5 64 GPU | 0.02 | 0.05 | **0.10** |
| CORE2 64 GPU | 0.13 | 0.30 | 0.66 |
| farc 2048 CPU | 0.44 | 1.01 | 2.40 |
| dars 8192 CPU | 0.59 | 1.30 | 3.02 |

Deep K is cheapest where the mesh is **biggest per rank** — nearly free at NG5/dars GPU, hopeless
at the big-CPU points.

## 3. 🔴 The rung is exact, but not bitwise — and the reason is in the SE module

`myDim_elem2D` is **not a partition**: 1341 of 244659 CORE2 `dist_8` elements (0.55 %) sit in the
`myDim` range of more than one rank, are computed redundantly, and are reconciled by no exchange
(rlist covers only `[myDim, …)`). With the two-term viscosity V[Ūᵐ]−V[Ūⁿ] those copies diverge
from the first step whose anchor Ūⁿ is nonzero — step 2, because the cold start makes it exactly
zero at step 1. The certified path hides this completely (η is exchanged, so the node owner's copy
wins); the rung recomputes η from its own copies and so reproduces a different, equally valid
member of the family.

Each hypothesis was killed by measurement, not argument:

| question | answer |
|---|---|
| shipped CSR row / its order right? | **exactly 0 through all 50 substeps of step 1** |
| the viscosity? | `FESOM_SE_VISC=0` → **0 at every substep of the run** |
| the neighbour summation order? | canonicalised by global id → no change, to the last digit |
| the geometry? | `elem_area` differs for **0 of 244659** elements across holders |
| a dropped neighbour? | stencil size differs for **0 of 244659** |
| the forcing? | `Fbt`/`H0e` identical at steps 1 **and** 2 |
| Ū itself? | step 1 END **0**; step 2 END **529 elements, max 5.2e-07** |

**Residual** (CORE2 np8, 200 steps, job 26952789): 0 at step 1 → 6.1e-10 (step 2) → 1.5e-6
(step 50) → **2.4e-6 m at step 200**, saturating, while η itself reaches 3.01 m. For scale, the SE
CUDA coupled floor is η ~1e-3 and the M12 1-year SE-vs-SI twin differs by 2e-4 m rms — the
residual sits ~400× below the floor the SE track already judges GPU runs against.

Consequence for the ladder: the byte gate is retired and replaced by **exactness where the model
is rank-consistent** (≡ 0.0 at step 1 and for every `VISC=0` substep) plus the stability screen and
the graded disturbance report, whose controls are rung-off runs at other rank counts — the same
perturbation class the residual belongs to.

*Independently of M12b, this says the SE viscosity yields slightly different Ū on ~0.5 % of
elements depending on which rank you ask. Worth putting to Sergey on its own.*

## 4. The extended-mesh contract (design of record; not built)

The rule that makes deep K and higher-order advection share one layer: **extended entities are
appended in BFS ring order**, so "everything within ring r" is the contiguous range `[0, N_r)` and
a consumer expresses rim shrink as a `RangePolicy` bound — no indirection, no ghost kernel, no
extra launch (L109; M9's LEAN writing beat its separate-kernel twin by ×1.8–5 for this reason).
Shipped as owner bytes, never recomputed: `elem_nodes` as local ids, `gradient_sca`, `elem_area`,
metric factor, coriolis, H0e/depth, `ulevels`/`nlevels`, and per-node operator rows in owner order.
Build by BFS at scatter time (`fesom_evpwide_mesh_hook` precedent). SE would take ring values +
operator rows + a window scheduler; higher-order advection takes ring values + geometry + vertex
maps once per step, no scheduler. The certified ice EVPWIDE does **not** migrate onto this.

## Open items

GPU board (CORE2 16N + NG5 16N, jobs 26953010/26953012) and the CUDA exactness gate (26953032) —
queued behind the 2026-08-14 GPU maintenance window · CPU board (farc 2048, dars 8192;
26953153/26953154) · 3000-step screen (26952979) · disturbance report (26953179) · deep-K decision,
on those numbers.
