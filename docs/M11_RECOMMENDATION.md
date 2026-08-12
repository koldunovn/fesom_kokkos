# M11 — what to change, and what it is worth

One page. The evidence is in `PARTITIONING_M11.md` (Findings 1–45); the per-race table is
`scripts/m11_harvest_races.py --best`.

## 🔴 State of the evidence (read before quoting any number below)

Finding 39 changed what this page can claim. A partition that races well over 150–300 steps can
still diverge before step 3,000: the dars CPU winner did exactly that, and the short-race protocol
that produced every headline here would have shipped it. So the table below is split by what has
actually been proven, and **an un-screened gain is not a recommendation**:

| status | points |
|---|---|
| **screened + accuracy-gated → recommend** | CORE2 4 GPU · CORE2 512 CPU · fArc 2048 CPU · **dars 2048 CPU (KaMinPar)** |
| stability-clean, accuracy FAILED → do not ship | fArc 16 GPU · CORE2 864 CPU (Finding 44 follow-up, job 26904986) |
| **stability FAILED → withdrawn** | dars 2048 CPU `MINCONN` (the KaMinPar survivor is certified instead) |
| **screened ✅, accuracy gate running** | dars 64 GPU (**−19.7 % at 3,000 steps**, job 26895260) |
| **screened ✅ (−9.8 % at 3,000 steps), accuracy gate running** | NG5 64 GPU (3 of 4 alternates diverge at the ladder dt — see adoption procedure) |
| measured null | NG5 2048 CPU |

## The short version

FESOM's mesh partitioner calls METIS through `METIS_PartGraphRecursive`, and that choice makes
three METIS options unavailable: `OBJTYPE=VOL`, `CONTIG` and `MINCONN`. `MINCONN` — minimise the
maximum number of neighbouring sub-domains — is **never set anywhere in `fort_part.c`**, so no
partition produced by stock FESOM has ever had it active.

We measure `MINCONN` as the single most valuable partitioning knob available on GPU: it is the
best or near-best arm at every GPU point measured, and the largest gain is **−19.7 % of the
model step** (dars, 64 GPU — clean at 3,000 steps; accuracy gate in progress; see the evidence
table above). Switching the call to `PartGraphKway` is a few lines.

The campaign has also seen four partitions fail at length (Findings 34, 39, 45) — three from
`MINCONN`-family arms and one from **the stock shipped recipe itself** (a seed-only re-roll,
Finding 45). The fragility is a property of repartitioning these meshes at these rank counts,
not of any particular knob — which is why the adoption procedure below is not optional, for any
recipe, including re-generating today's defaults with a new seed.

To be fair to the original authors, this is a documented choice rather than an oversight:
`fort_part.c:328-355` records that `PartGraphRecursive` "resulted in a far better partition than
Kway" on one test mesh (`mesh_aguv`, `wgt_type=2`), notes that "there is no rule which one works
best", and carries explicit comments that `CONTIG` is "ignored by METIS_PartGraphRecursive" and
that "`_VOL` only works with `METIS_PartGraphKway`". What has changed since is the hardware: the
comparison predates GPU backends, and on GPU the objective those options serve is the one that
matters.

## Measured gains (min-of-N, matched same-day pairs, each mesh at its cold-start ladder dt)

| mesh | backend | ranks | setting | gain | gated |
|---|---|--:|---|--:|---|
| dars | GPU | 64 | `MINCONN`+`CONTIG`+`UFACTOR=30` | **−19.7 %** | stability ✅ (3,000 steps, grew from −18.6 %) · accuracy gate running |
| NG5 | GPU | 64 | `MINCONN` | **−9.7 %** | stability ✅ (−9.8 % at 3,000 steps) · accuracy gate running |
| CORE2 | GPU | 4 | `MINCONN` | **−8.1 %** | ✅ accuracy + 3,000-step stability |
| fArc | GPU | 16 | `MINCONN`+`CONTIG` | −3.6 % | stability ✅ · 🔴 **accuracy FLAGGED — not recommended** |
| fArc | CPU | 2048 | Mt-KaHyPar `w=100+nlev` | **−7.5 %** | ✅ accuracy (4 controls) + stability |
| CORE2 | CPU | 512 | Hilbert renumbering + engine | **−5.8 %** | ✅ accuracy + stability |
| dars | CPU | 2048 | **KaMinPar `w=100+nlev`** | **−4.2 %** | ✅ accuracy (below every control on temp/salt) + 3,000-step stability |
| ~~dars~~ | ~~CPU~~ | ~~2048~~ | ~~`MINCONN`~~ | ~~−4.5 %~~ | 🔴 **FAILS the 3,000-step screen — withdrawn** |
| CORE2 | CPU | 864 | KaMinPar `w=100+nlev` | −4.1 % | stability ✅ (−4.9 % at length) · 🔴 **accuracy FAILED (salt, 5-control envelope) — not recommended** |
| CORE2 | CPU | 512 | `UFACTOR=30` alone | −3.8 % | ✅ accuracy + stability |
| NG5 | CPU | 2048 | — | **null** | — |

At the NG5 re-race (ladder dt 180) the three alternate arms — `MINCONN`+`CONTIG`, +`UFACTOR=30`,
and slack — all diverged within 300 steps in both reps while `MINCONN` ran clean at −9.7 %:
NG5 is the most partition-fragile mesh in the campaign, and the adoption screen is doing exactly
the work Finding 45 says it must.

## The two rules that fall out of it

**1. The backend wants opposite things, and the split holds at every scale.** On GPU the currency
is the NUMBER of communication partners — `MINCONN` targets it directly. Across all nine raced
groups it is the only metric family with a consistent positive correlation to step time, while
every volume-like metric (edge cut, halo size, element replication) correlates **negatively**:
on GPU the arms that ship more data are faster. On CPU the picture is the mirror image, and the
same two extra knobs that **cost** a point on dars CPU **gain five** on dars GPU.

Two caveats that Finding 37 established and that must travel with this claim:

* Partner count is a **threshold, not a ranking** — `nbr_max` is a small integer taking two or
  three values per point, so it separates fast arms from slow ones but says nothing about
  ordering within a level.
* **No scorecard column orders CPU arms.** The r = 0.91 result often quoted for the CPU side is
  per-rank busy time vs per-rank owned 3-D work *inside one partition* — a statement about where
  time goes, not about which partition to pick. Arm-level 3-D imbalance does not predict arm
  speed (median ρ −0.19, range straddling zero). On CPU you have to race.

**2. There is no single recipe.** The best setting is mesh- and rank-dependent. What generalises
is `MINCONN` on GPU (best or within a point of best at every GPU point measured) and *some* slack
on CPU.

**Cross-architecture status (GH200, Findings 42/44):** the GPU lever reproduces on GH200 at
single-node scale (−8.9 % vs −8.1 % on A100, CORE2 4 GPU) — it is not an A100 artefact — but at
4 and 16 GH200 nodes on a fabric without GPUDirect it is null-to-small, and a pre-registered
prediction that dars/64 would exceed its A100 −18.6 % failed. Multi-node gains on other machines
must be measured there, not extrapolated.

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

- **The scorecard.** It is a design aid, never a gate (Findings 18, 34, 37). On NG5 it failed on four arms to
  identify a partition that destroys the run — every column put a dying arm on the *better* side
  of a surviving one.
- **A short smoke run.** The NG5 partition that blows up at step 71 passes a 5-step smoke and
  would pass a 20-step gate.

## Renumbering: CORE2 only

Hilbert (`hilbert-xyz`) renumbering is worth −1.2…−2.4 % on CPU and −5.0 % on GPU at 1 node, and
it is **~94 % additive** with repartitioning (−5.8 % combined at CORE2 512). It is worth nothing
anywhere else: fArc, FORCA20, dars and NG5 already ship at 88 % element-gather locality against
CORE2's 27.6 % (Finding 17).

**Weigh the disruption before taking it.** Renumbering changes the mesh files themselves, not just
a `dist_N`, so every C↔Kokkos floor in `docs/REFERENCE_RUNS.md` has to be re-baselined on the new
numbering, per scheme, and every other track pinning a CORE2 reference has to move with it. The
lever is worth **+2.0 pp on top of repartitioning alone** at CORE2 512 (−5.8 % vs −3.8 %). Take
the repartitioning first — it is a drop-in `dist_N` swap with no re-baselining — and treat the
renumbering as a separate decision with its own cost.

## Upstream (FESOM/fesom2)

1. `PartGraphRecursive` → `PartGraphKway` so `MINCONN`/`CONTIG`/`OBJ=vol` actually reach METIS.
2. The `FESOM_PART_*` runtime knob family, so partition choices stop being a recompile.
3. The `check_partitioning` isolated-node fix (`FIXISO`), which currently seeds its candidate list
   with one neighbour instead of building the whole list.
4. METIS 5.2.1.
