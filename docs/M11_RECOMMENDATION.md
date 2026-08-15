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
best or near-best arm at every GPU point measured. The largest gain is **−18.5 %** (dars 64 GPU,
`MINCONN`+`CONTIG`+`UFACTOR=30`), and under the deterministic initial condition that arm is also
**tier 1** — its temperature deviation is the smallest of any leg in its gate, controls included.
`MINCONN` alone is −12.9 % there. Switching the call to `PartGraphKway` is a few lines.

**The GPU story is the part of this campaign that the M13 fix leaves standing, and strengthens.**
The gains reproduce within a point, and the two accuracy rejections that removed dars/u30 and
fArc 16 from the recommendation were both artefacts of the initial condition, not of the
partitions.

🔴 **The "fragile partitions" of Findings 34, 39 and 45 do not exist.** Every one of those
failures was a cold start from a partition-dependent initial condition, and under the
deterministic fill the same partitions run the full 3,000-step screen: dars `MINCONN` −4.24 %,
the Finding-45 "killer" `dars_seed660013` −4.53 %, and both NG5 arms that "died" at steps 71 and
63. Measured cause: the hole-filler starts two partitions of the same mesh **27 PSU apart** in
salinity (all four meshes; zero under det). The adoption screen below is still worth running —
partitions can in principle be bad — but it is guarding a far rarer failure than this campaign
believed, and none of the eight failures it caught was real.

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

**The CPU rows below are re-measured under det** (2026-08-15). GPU rows are being re-measured;
those still carrying legacy numbers are marked.

| mesh | backend | ranks | setting | gain | stability | disturbance vs seed spread |
|---|---|--:|---|--:|---|---|
| **dars** | GPU | 64 | **`MINCONN`+`CONTIG`+u30** | **−18.5 %** (legacy −18.6) | ✅ | **tier 1 — the LOWEST temp deviation of any leg incl. 5 controls; temp max below the class; identical CG paths.** The legacy "largest disturbance in the campaign" was the hole-filler |
| dars | GPU | 64 | `MINCONN` alone | **−12.9 %** (legacy −13.6) | ✅ | temp/ssh in class, salt +12 % over a tight 5-control top — tier 1/2 boundary |
| CORE2 | GPU | 4 | `MINCONN` | **−7.1 %** (legacy −8.1) | ✅ | at or below the 4-control top on temp/salt/ssh, identical CG paths — **tier 1** |
| fArc | GPU | 16 | `MINCONN`+`CONTIG` | −4.3 % *(rep 1; race completing)* | ✅ | **tier 1 — lowest temp deviation of any leg incl. 3 controls.** The legacy "FAILED on accuracy" (temp +33 %, ssh +68 %) was the hole-filler |
| NG5 | GPU | 64 | `MINCONN` | *(legacy −9.8 %; det race+gate running)* | ✅ | partial gate: temp rms 1.3e−7 (legacy 1e−2 class) |
| fArc | CPU | 2048 | Mt-KaHyPar `w=100+nlev` | **−7.8 %** at length | ✅ 3,000 steps | in class on all three (3 controls) — tier 1 |
| CORE2 | CPU | 512 | Hilbert renumbering + engine | **−5.4 %** at length | ✅ 3,000 steps | in class — tier 1 |
| CORE2 | CPU | 864 | KaMinPar `w=100+nlev` | **−5.1 %** at length | ✅ 3,000 steps | **at or below every one of 5 controls** — tier 1 (the legacy salt +24 % was the hole-filler) |
| **dars** | CPU | 2048 | **a fresh seed of the STOCK recipe** | **−4.6 %** | ✅ 3,000 steps | tier 1 — see the dars note below |
| dars | CPU | 2048 | `MINCONN` | **−4.3 %** at length | ✅ 3,000 steps (the F39 failure was the hole-filler) | tier 1 — no arm distinguishable from a re-roll on any field |
| CORE2 | CPU | 512 | KaHIP `w=100+nlev` | −3.9 % | ✅ 3,000 steps | tier 1 |
| CORE2 | CPU | 512 | `UFACTOR=30` alone | **−3.7 %** at length | ✅ 3,000 steps | tier 1 |
| dars | CPU | 2048 | KaMinPar `w=100+nlev` | −3.6 % | ✅ | tier 1 |
| NG5 | CPU | 2048 | — | **null** (best arm +0.4 %) | ✅ all four arms run | — |

### 🔴 dars 2048 CPU is a different recommendation from the rest

At every other point the shipped partition is a *good* draw and the gain comes from the knob or
the engine: a stock-recipe seed re-roll costs **+8…+15 %** at CORE2 864 and **+3…+8 %** at fArc.
At dars it is the opposite — three independent re-rolls of the stock recipe all beat the shipped
`dist_2048` by 4.55–4.86 %, and at min-of-3 a plain re-roll (−4.61 %) ties `MINCONN` (−4.27 %)
inside the repetition spread. The dars baseline is byte-identical to `/pool`'s shipped partition.

⇒ At dars, **re-run the stock partitioner with a different seed**; the knobs add nothing beyond
that. Elsewhere, use the knob/engine rows and keep the shipped partition as the thing to beat.

### How to read the disturbance column

🔴 **Every CPU point re-gated under det is tier 1**, and the tier-4 verdicts that removed CORE2
864 and the dars alternates from the recommendation are gone. The absolute numbers moved by five
to six orders of magnitude (CORE2 864 salt: 2.24e−1 → 1.43e−7), because the legacy envelopes were
measuring the initial condition rather than the model's response to the decomposition. Tiers 2–4
are kept below because the GPU points have not all been re-gated yet — but see the caveat under
tier 2.

* **Tier 1 — inside (or below) the seed spread.** Adopting this partition disturbs the solution
  no more than re-rolling the METIS seed of the stock recipe, which nobody audits. Nothing to
  decide.
* **Tier 2 — SSH-only excursion with the stopping mechanism.** The barotropic solver stops on a
  *relative* residual; a partition on which it converges faster stops earlier and lands
  measurably elsewhere *inside the same requested tolerance*. Both runs satisfy the tolerance;
  the difference lives in the solver's own tolerance ball, and tightening the tolerance would
  shrink it. This is a numerics artefact, not a physics bias.
  🔴 **Caveat (2026-08-15): this mechanism may not survive.** Under det, every leg of every CPU
  gate re-run so far — arms *and* controls — takes an **identical** CG iteration path for all
  twenty steps, at CORE2 512, CORE2 864, dars 2048 and CORE2 4 GPU. The differing iteration
  counts the explanation rests on were a consequence of the legs starting from different states.
  Whether any iteration-count difference remains at dars 64 and NG5 64 GPU is being measured.
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

Re-measured under det, every arm below is tier 1, so the question is purely speed:

| point | best METIS-only arm | engine | verdict |
|---|---|---|---|
| CORE2 512 | `UFACTOR=30` −3.7 % | KaHIP −3.9 % | ≈ 0.2 pp — **skip the engine** |
| CORE2 864 | (none raced under det) | KaMinPar **−5.1 %** | the only arm measured; a seed re-roll is +8…+15 % |
| fArc 2048 | `MINCONN` −5.3 % at length | Mt-KaHyPar **−7.8 %** at length | **+2.5 pp** — worth the workflow cost |
| dars 2048 | `MINCONN` −4.3 % | KaMinPar −3.6 % | **−0.7 pp — the engine LOSES**; and a plain seed re-roll (−4.6 %) beats both |

🔴 **This reverses the campaign's dars conclusion.** The engine was recommended there because
`MINCONN` "diverged at length" (Finding 39) and the METIS alternates were tier 4 — both artefacts
of the hole-filler. On the re-measurement the engine is the *slowest* of the three options at
dars. On GPU the engines never won a point, so the engine question stays CPU-only, and the
honest summary is now: **the engine is worth a build only at fArc.**

NG5 needs no engine and no knob — all four arms now run and every one is slower than the shipped
partition (+0.4 to +2.1 %). It is a null point, not a fragile mesh.

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

**Run cold starts with `FESOM_IC_EXTRAP=det`** — otherwise the initial condition depends on the
decomposition and neither a timing comparison nor a stability screen means what it appears to
(measured: two partitions of the same mesh start 27 PSU apart on all four meshes). This is the
single most important line on the page, and it applies to anyone comparing partitions, not just
to this campaign.

Then:

1. **Screen at protocol length at the target rank count** — 3,000 steps at the mesh's cold-start
   ladder dt (CORE2 1800 · fArc 900 · dars 120 · NG5 180). Cheap, and still the only honest proof
   that a partition runs.
2. **If it fails, re-roll `FESOM_PART_SEED` and screen again.**
3. `m11_promote` enforces the screen: it refuses any `dist_N` whose evidence lacks
   `run=<jobid> steps=>=3000 rc=0` at N ranks.

**Race the seed re-roll as well as the knobs.** At dars a plain re-roll of the stock recipe beats
every knob (−4.6 %); at CORE2 864 and fArc it is 3–15 % *worse* than the shipped partition. It
costs one partitioner run to find out which kind of point you have, and nothing else in this
campaign predicts it.

### Two things that do NOT work as gates

- **The scorecard.** It is a design aid, never a gate (Findings 18, 37). It cannot rank arms on
  CPU at all, and no column of it predicts the dars seed effect.
- **A short smoke run.** Unchanged in principle, though the concrete example this rule was
  written from (an NG5 partition "blowing up at step 71") was the initial condition, not the
  partition.

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
