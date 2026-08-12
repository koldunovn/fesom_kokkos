# M11 — what to change, and what it is worth

One page. The evidence is in `PARTITIONING_M11.md` (Findings 1–36); the per-race table is
`scripts/m11_harvest_races.py --best`.

## The short version

FESOM's mesh partitioner calls METIS through `PartGraphRecursive`, which **silently ignores three
of the options the code sets**. One of them, `MINCONN` — minimise the maximum number of
neighbouring sub-domains — turns out to be the single most valuable partitioning knob we have on
GPU, worth up to **−18.6 % of the model step**. It has never been active in any FESOM partition
ever generated.

Switching the call to `PartGraphKway` is a few lines. Everything below follows from that.

## Measured gains (min-of-N, matched same-day pairs, each mesh at its cold-start ladder dt)

| mesh | backend | ranks | setting | gain | gated |
|---|---|--:|---|--:|---|
| dars | GPU | 64 | `MINCONN`+`CONTIG`+`UFACTOR=30` | **−18.6 %** | pending |
| NG5 | GPU | 64 | `MINCONN` | −10.0 %* | pending |
| CORE2 | GPU | 4 | `MINCONN` | **−8.1 %** | ✅ accuracy + 3,000-step stability |
| fArc | GPU | 16 | `MINCONN`+`CONTIG` | −2.9 % | pending |
| fArc | CPU | 2048 | Mt-KaHyPar `w=100+nlev` | **−7.5 %** | ✅ accuracy (4 controls) + stability |
| CORE2 | CPU | 512 | Hilbert renumbering + engine | **−5.8 %** | ✅ accuracy + stability |
| dars | CPU | 2048 | `MINCONN` | −4.5 % | stability running |
| CORE2 | CPU | 864 | KaMinPar `w=100+nlev` | −4.1 % | not gated |
| CORE2 | CPU | 512 | `UFACTOR=30` alone | −3.8 % | ✅ accuracy + stability |
| NG5 | CPU | 2048 | — | **null** | — |

\* measured at production dt 240; re-race at the ladder dt 180 in flight.

## The two rules that fall out of it

**1. The backend wants opposite things, and the split holds at every scale.** On GPU the currency
is the NUMBER of communication partners — `MINCONN` targets it directly, and partner count orders
the GPU table perfectly (3.00 → 2.00 neighbours/rank = −7.5 %). On CPU the currency is owned 3-D
work (busy-time vs owned 3-D nodes, r = 0.91), which `UFACTOR` slack buys. The same two extra
knobs that **cost** a point on dars CPU **gain five** on dars GPU.

**2. There is no single recipe.** The best setting is mesh- and rank-dependent. What generalises
is `MINCONN` on GPU (best or within a point of best at every GPU point measured) and *some* slack
on CPU.

## Adoption procedure

```
FESOM_PART_KWAY=1 FESOM_PART_OBJ=vol FESOM_PART_VSIZE=1 FESOM_PART_WGT_A=100 \
FESOM_PART_MINCONN=1 [FESOM_PART_CONTIG=1] [FESOM_PART_UFACTOR=30]
```

then, **without exception**:

1. **Screen at protocol length at the target rank count** — 3,000 steps at the mesh's cold-start
   ladder dt (CORE2 1800 · fArc 900 · dars 120 · NG5 180).
2. **If it fails, re-roll `FESOM_PART_SEED` and screen again.** A failing partition is a lottery
   ticket, not a verdict on the knobs (Finding 34).
3. `m11_promote` enforces both: it refuses any `dist_N` whose evidence lacks
   `run=<jobid> steps=>=3000 rc=0` at N ranks.

### Two things that do NOT work as gates

- **The scorecard.** It is a design tool for ranking candidates. On NG5 it failed on four arms to
  identify a partition that destroys the run — every column put a dying arm on the *better* side
  of a surviving one.
- **A short smoke run.** The NG5 partition that blows up at step 71 passes a 5-step smoke and
  would pass a 20-step gate.

## Renumbering: CORE2 only

Hilbert (`hilbert-xyz`) renumbering is worth −1.2…−2.4 % on CPU and −5.0 % on GPU at 1 node, and
it is **~94 % additive** with repartitioning (−5.8 % combined at CORE2 512). It is worth nothing
anywhere else: fArc, FORCA20, dars and NG5 already ship at 88 % element-gather locality against
CORE2's 27.6 % (Finding 17).

## Upstream (FESOM/fesom2)

1. `PartGraphRecursive` → `PartGraphKway` so `MINCONN`/`CONTIG`/`OBJ=vol` actually reach METIS.
2. The `FESOM_PART_*` runtime knob family, so partition choices stop being a recompile.
3. The `check_partitioning` isolated-node fix (`FIXISO`), which currently seeds its candidate list
   with one neighbour instead of building the whole list.
4. METIS 5.2.1.
