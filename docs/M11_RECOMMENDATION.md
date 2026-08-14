# M11 — what to change, and what it is worth

One page. The evidence is in `PARTITIONING_M11.md` (Findings 1–45); the per-race table is
`scripts/m11_harvest_races.py --best`.

## 🔴🔴 2026-08-14/15 — THE DISTURBANCE COLUMN BELOW IS SUPERSEDED; RE-RUN IN PROGRESS

M13 found that the climatology hole-filler gives each decomposition a **different initial
condition**, and the campaign is being re-measured with the deterministic fill
(`FESOM_IC_EXTRAP=det`). Plan and status: `docs/plans/20260815-m11-det-rerun.md`; evidence:
`PARTITIONING_M11.md` session 7.

What that means for this page, stated in the order it matters:

1. **The disturbance/tier column was largely measuring the hole-filler, not the partition.**
   Measured directly: two partitions of CORE2 differ in their *initial* salinity by 27.4 PSU and
   fArc by 26.0 PSU, against zero under det. Re-derived with det, the CORE2 864 field differences
   shrink by five to six orders of magnitude and the `+24 %` salt excursion that removed that
   point from the recommendation **disappears**. Do not quote a tier from the table below until
   its row says det.
2. **The stability column's failures are void, not confirmed.** Every "diverges / blows up"
   verdict (Findings 34, 39, 45) was earned on a run whose initial state was an artefact of the
   very partition being judged. Re-screens are running.
3. **The gains appear to survive.** Re-raced under det so far: CORE2 512 slack −3.87 %
   (was −3.83), CORE2 512 KaHIP −4.03 % (−4.03), CORE2 512 Hilbert+engine −5.08 % (−5.83),
   CORE2 864 KaMinPar −4.25 % (−4.07/−4.92), fArc 2048 Mt-KaHyPar −7.37 % (−7.52), fArc 2048
   `MINCONN` −5.07 % (−5.49). Ordering unchanged, magnitudes inside the usual spread.
4. **The tier-2 "stopping mechanism" needs re-deriving rather than re-quoting.** Under det every
   leg of the CORE2 864 and CORE2 512 gates — arms and controls — takes an *identical* CG
   iteration path for all twenty steps. The differing iteration counts that the explanation rests
   on followed from the legs starting in different states.
5. The four promoted meshes under `mesh_m11_certified/` are unaffected **as artefacts** — a
   `dist_N` file does not change. Their evidence lines were recorded under the legacy fill and
   are being re-earned.

## 🔴 State of the evidence (read before quoting any number below)

**Protocol decision (user, 2026-08-13, supersedes the binary yardstick of 2026-08-12):** there
is no certified/not-certified accuracy verdict any more. Two tiers remain:

1. **Stability at length — hard pass/fail, unchanged.** A partition that makes the model
   diverge or blow up (Findings 34/39/45; eight caught, one from the stock recipe) is not a
   "disturbance", it is unusable. Nothing un-screened is a recommendation, and `m11_promote`
   keeps enforcing ≥3,000 steps at the target rank count.
2. **Accuracy — a graded disturbance report, not a gate.** Every surviving partition changes
   the solution a bit; the question is how big the change is against the natural scale bar —
   the spread of ordinary seed re-rolls of the stock recipe — and what it means. The
   per-point numbers and their interpretation are in the disturbance column of the table
   below and in the "how to read it" block after it.

**Scope of the disturbance numbers, stated honestly:** they are 20-step cold-start divergences
— a *detectability* statement ("can this partition be told apart from re-rolling the seed?"),
not a climate measurement. Long-term climate impact was not measured in M11. The cheap next
tier is comparing end states after 3,000 steps against 3 controls; the real answer is a long
twin run (the M7 precedent: a 63-year hindcast separated port from reference at climate level).

## The short version

FESOM's mesh partitioner calls METIS through `METIS_PartGraphRecursive`, and that choice makes
three METIS options unavailable: `OBJTYPE=VOL`, `CONTIG` and `MINCONN`. `MINCONN` — minimise the
maximum number of neighbouring sub-domains — is **never set anywhere in `fort_part.c`**, so no
partition produced by stock FESOM has ever had it active.

We measure `MINCONN` as the single most valuable partitioning knob available on GPU: it is the
best or near-best arm at every GPU point measured. The largest gain is **−19.7 %** (dars 64
GPU, `MINCONN`+`CONTIG`+u30 — the arm with the campaign's largest disturbance, see below);
`MINCONN` alone is **−14.3 %** there with a disturbance confined to SSH. Switching the call to
`PartGraphKway` is a few lines.

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

Stability = the 3,000-step screen at the target rank count (hard requirement). Disturbance =
20-step rms deviation from base, quoted **relative to the spread of ≥3 seed-control re-rolls**
of the stock recipe on the same mesh.

| mesh | backend | ranks | setting | gain | stability | disturbance vs seed spread |
|---|---|--:|---|--:|---|---|
| dars | GPU | 64 | `MINCONN`+`CONTIG`+u30 | **−19.7 %** | ✅ | 🔴 **largest in campaign**: temp rms +13 %, temp max ×1.8–2.5 the class (tier 4 — long twin run before production) |
| dars | GPU | 64 | `MINCONN` alone | **−14.3 %** | ✅ | ssh rms +11 % only; stopping mechanism (tier 2 — bounded by solver tolerance) |
| NG5 | GPU | 64 | `MINCONN` | **−9.8 %** | ✅ | ssh rms +8 % only; mechanism ruled out (tier 3 — small, unexplained) |
| CORE2 | GPU | 4 | `MINCONN` | **−8.1 %** | ✅ | below every control on all three fields (tier 1 — indistinguishable) |
| fArc | GPU | 16 | `MINCONN`+`CONTIG` | −3.6 % | ✅ | temp rms +33 %, ssh +68 % above a 4-control envelope (tier 4; also the smallest gain measured) |
| fArc | CPU | 2048 | Mt-KaHyPar `w=100+nlev` | **−7.5 %** | ✅ | in class on all three (4 controls) — tier 1 |
| CORE2 | CPU | 512 | Hilbert renumbering + engine | **−5.8 %** | ✅ | in class — tier 1 |
| dars | CPU | 2048 | **KaMinPar `w=100+nlev`** | **−4.2 %** | ✅ | below every control on temp/salt — tier 1 |
| ~~dars~~ | ~~CPU~~ | ~~2048~~ | ~~`MINCONN`~~ | ~~−4.5 %~~ | 🔴 **diverges before step 3,000 — unusable** | — |
| CORE2 | CPU | 864 | KaMinPar `w=100+nlev` | −4.1 % | ✅ (−4.9 % at length) | salt rms +24 % above a 5-control envelope, temp below every control (tier 4, mixed) |
| CORE2 | CPU | 512 | `UFACTOR=30` alone | −3.8 % | ✅ | in class, ssh below both controls — tier 1 |
| NG5 | CPU | 2048 | — | **null** | — | — |

### How to read the disturbance column

* **Tier 1 — inside (or below) the seed spread.** Adopting this partition disturbs the solution
  no more than re-rolling the METIS seed of the stock recipe, which nobody audits. Nothing to
  decide.
* **Tier 2 — SSH-only excursion with the stopping mechanism.** The barotropic solver stops on a
  *relative* residual; a partition on which it converges faster stops earlier and lands
  measurably elsewhere *inside the same requested tolerance*. Both runs satisfy the tolerance;
  the difference lives in the solver's own tolerance ball, and tightening the tolerance would
  shrink it. This is a numerics artefact, not a physics bias.
* **Tier 3 — SSH-only, unexplained.** Same signature as tier 2 but the iteration-count
  mechanism is ruled out (NG5: the winner's CG trace matches the controls exactly). Small and
  confined to one diagnostic field, but without a mechanism it stays labelled unexplained.
* **Tier 4 — temperature/salinity excursions, especially in the max.** T and S carry the
  model's long-term state, and a local *max* excursion (dars u30: temp max ×1.8–2.5 the class
  at 20 steps) is the same signature family as the local runaways that killed partitions
  outright (Findings 34/45 traces). These are the disturbances that earn a long twin run
  before production use — the speed gain may well be worth it, but on evidence, not hope.

### What the external engines buy over METIS knobs (the workflow-cost question)

Adding KaMinPar/Mt-KaHyPar to the workflow is a real cost (a build, a graph export, an
injection step — though all offline and once per mesh×N). What it buys, against the best
METIS-knobs-only arm **at the same point and gate standard**:

| point | best METIS-only arm | its gates | engine premium |
|---|---|---|---|
| CORE2 512 | `UFACTOR=30` −3.8 % | ✅ tier 1 | ≈ 0 pp (KaHIP raced −4.0 %, inside noise) — **skip the engine** |
| fArc 2048 | `MINCONN` −5.5 % at length | ✅ screen + accuracy (26892880/26892879) | **+2.0 pp** (−7.5 %) — adopter's choice |
| dars 2048 | slack −3.1 % / `a4` −2.8 % | 🔴 both tier 4 (temp/salt above the 4-control spread); `MINCONN` −4.5 % diverges at length | **engine is the only stable + tier-1 arm** |

On GPU the engines never won a point — `MINCONN` (a plain METIS knob) is the whole GPU story —
so the engine question is CPU-only. Pattern (Finding 45 block): at dars and fArc the engines are
simultaneously the fastest arms and the best-behaved in the accuracy gate.

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

## Certified deliverables (promoted 2026-08-13)

The four certified points are packaged under `/work/ab0995/a270088/port2/mesh_m11_certified/`,
each with a `MESH_PROVENANCE.md` (screen/gate job ids, source manifest) and its own md5 manifest;
evidence files are committed under `docs/promotions/`:

| certified mesh | dists | what | use |
|---|---|---|---|
| `core2_v1` | `dist_4` · `dist_512` | `MINCONN` (GPU −8.1 %) · `UFACTOR=30` (CPU −3.8 %) | **drop-in `MESH=` swap** |
| `core2hil_v1` | `dist_512` | Hilbert renumbering + KaHIP `w=100+nlev` (−5.8 %) | ⚠️ renumbered mesh — separate adoption decision (re-baselines every C↔K floor) |
| `farc_v1` | `dist_2048` | Mt-KaHyPar `w=100+nlev` (−7.5 %) | drop-in |
| `dars_v1` | `dist_2048` | KaMinPar `w=100+nlev` (−4.2 %) | drop-in |

The dars 64 and NG5 64 `MINCONN` winners are not yet packaged. Under the 2026-08-13 reporting
framework they are ordinary recommendations (stability ✅, disturbance tier 2/3, SSH-only) —
say the word and they promote as `dars_gpu_v1` / `ng5_gpu_v1`; both already satisfy
`m11_promote`'s stability requirement (screens 26895260 / 26908635).

## Upstream (FESOM/fesom2)

1. `PartGraphRecursive` → `PartGraphKway` so `MINCONN`/`CONTIG`/`OBJ=vol` actually reach METIS.
2. The `FESOM_PART_*` runtime knob family, so partition choices stop being a recompile.
3. The `check_partitioning` isolated-node fix (`FIXISO`), which currently seeds its candidate list
   with one neighbour instead of building the whole list.
4. METIS 5.2.1.
