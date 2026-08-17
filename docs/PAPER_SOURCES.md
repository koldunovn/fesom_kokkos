# Paper source map — sea ice, partitioning, solvers/SE, and the combination

**Written 2026-08-17.** This file **points**; it copies nothing. Every number a paper might quote
lives somewhere below, and the location is given so it can be re-derived rather than trusted.

Paper shape as described: three lever chapters (sea ice · partitioning · solvers/split-explicit),
closing with all of them applied together.

🔴 **Read §0 before quoting any number.** Several headline results in the older documents are
superseded or retracted, and the documents that contain them were not always updated.

---

## 0. Retractions and supersessions — the traps a paper would fall into

| stale claim | where it still appears | what replaced it | evidence |
|---|---|---|---|
| **"The ice wide halo loses on GH200"** | `docs/plans/20260722-dolpung-GH200-SCALING.md`, M14 handoff §2, older figure captions | **Retracted.** Those runs posted MPI on device pointers; on a fabric without GPUDirect that crashes. Post-fix it **wins** on JUPITER at every mesh and rung (−2…−12 %). | commit `94877a9`; validation §3 below |
| **Every pre-`det` accuracy/stability verdict in M11** | `docs/PARTITIONING_M11.md` (large parts) | Superseded by the 2026-08-15 `FESOM_IC_EXTRAP=det` re-run: differences shrank 5–6 orders, all points became tier 1, and 8 "fragile partition" failures were refuted. **Speed survived; accuracy did not.** | `docs/plans/20260815-m11-det-rerun.md` |
| **All 8 M9 legacy scaling curves** | `docs/plans/20260805-m9-RESULTS.md` | Partition-dependent initial conditions. CPU verdict confirmed unchanged, but use `pgf` (step-1 pressure-gradient force), **never** the global T/S extremes, as the discriminant. | `docs/plans/20260815-m9-det-rerun.md` |
| **fArc `oati`/`cg2` "divergences"** | M10 material | False positives of our own stall guard. Baseline `cg` is **unwatched**, so its `fallbacks=0` is vacuous, not evidence of robustness. Root cause still open — see §6. | `docs/SSH_SOLVERS_M10.md` §"What the guard was actually reporting" |
| **"NG5 is the most fragile mesh"** | M11 pre-det text | Retired. Plain null under `det`; all four arms run. | `docs/plans/20260815-m11-det-rerun.md` |
| **A100 dars partition "−10.1 %"** | earlier M14 figures | Estimator error: one job's best divided by another job's base, where the first job's baseline legs were rejected. Defensible value **1.046× at 16 GPUs**. | `scripts/m14_scaling_figs.py` header |
| **"CG NaN blocks the merge"** | M14 handoff §0.1 (original text) | Not a bug. The M5.24 cold-start vertical CFL blow-up (rule 0.41) with `wsplit` off. `h17` fails the same condition; the merged binary does not. | M14 handoff §0.1 amendment + §6 |
| **"the wide SE rung is bitwise-exact"** | my own memory index; easy to infer from the M12b handoffs | **Misreading.** That claim is the `FESOM_SE_WIDE_SELFCHECK` metric — the rung's locally computed ring-1 η against the owner's exchanged bytes. M12b **explicitly retired** byte identity against plain SE as "unattainable by construction" (`docs/plans/20260814-m12b-widehalo.md:157,:462`). The shipped contract is a **rounding-class pair** (`docs/WIDEHALO_M12B.md:211`), because the rung applies an owner-wins F-reconcile and a viscosity neighbour-order canonicalisation over multi-claimed elements that plain SE does not. | JUPITER session, 2026-08-17, incl. a pre-merge-binary reproduction |
| **dolpung GH200 as "the GH200 result"** | pre-2026-08-16 figures | Superseded by JUPITER, which reaches 2048 GPUs. dolpung was a 42-node proxy. | `docs/plans/20260816-m14-JUPITER-scaling-FINDINGS.md` |

---

## 1. Where the code is

| branch | worktree | holds |
|---|---|---|
| **`m14-integrate`** | `~/port_kokkos_int` | **the integration of all four levers — the paper's "combined" configuration.** Pushed to `github.com/koldunovn/fesom_kokkos`. |
| `m9-mevp-double` | `~/port_kokkos_ice` | sea-ice dynamics work (mEVP, wide ice halo) |
| `m10-ssh-solvers` | `~/port_kokkos_ssh` | the SSH Krylov solver family, and `fesom_ssh_lab` (the offline solver laboratory) |
| `m11-partition` | `~/port_kokkos_part` | partitioning: engines, the zoo, the scorecard |
| `m12b-widehalo` | `~/port_kokkos_wh` | split-explicit wide rung |
| `m13-cg-robustness` | `~/port_kokkos` | the deterministic initial-condition fill (`FESOM_IC_EXTRAP=det`) |

All per-track documents are also present on `m14-integrate`, so the paper can be written against
that one checkout.

---

## 2. Chapter: sea ice

- **Primary:** `docs/plans/20260804-m9-mevp-divergence.md` · results
  `docs/plans/20260805-m9-RESULTS.md` · handoff `docs/plans/20260805-m9-HANDOFF.md`
- **The lever itself (wide ice halo, lean writing):** `docs/plans/20260806-m9-P5-lean-wide-halo.md`
- **Pre-registrations** (useful for a methods section — the predictions were recorded before the
  runs): `20260804-m9-PREREG.md`, `20260805-m9-PREREG-P0b.md`, `20260805-m9-PREREG-P1.md`
- **Deterministic-IC re-run, which every M9 number now depends on:** `docs/plans/20260815-m9-det-rerun.md`
- **Source:** `src/fesom_ice_evpwide.cpp` · knobs `FESOM_SPEED_EVPWIDE=K`, `FESOM_SPEED_EVPWIDE_LEAN=1`
- **Data:** `/work/ab0995/a270088/port2/m9/` · figures `/work/ab0995/a270088/port2/m9/figs/`
- **A written report already exists** (for Danilov): `docs/report/danilov_mevp_report.tex|pdf`,
  with `fig1_icecost.pdf`, `fig2_scaling_step.pdf`
- **Combined-campaign numbers:** `docs/figures/m14_lever_ice.png` (absolute + speedup, three
  platforms, K=2/4/8)
- 🔴 The transport bug and its fix — essential, because it invalidates every pre-fix GH200 ice
  measurement: commit `94877a9`, and the Levante validation in §3.

**Claims the chapter rests on:** the halo trades messages for replicated ghost work, so it pays
only where a flat per-message cost dominates — a win on GPU, a loss on CPU at every K and every
rank count. Sea ice does not strong-scale on GPU; the lean wide form is the only configuration
whose ice cost *falls* with node count.

---

## 3. Chapter: partitioning

- **Primary:** `docs/PARTITIONING_M11.md` (large; treat pre-`det` accuracy sections as superseded)
  · research survey `docs/PARTITIONING_M11_RESEARCH.md` · recommendation `docs/M11_RECOMMENDATION.md`
- **Plans/handoffs:** `docs/plans/20260810-m11-partitioning.md`, `20260810-m11-HANDOFF.md`,
  `20260812-m11-HANDOFF.md`
- 🔴 **The re-measurement that supersedes the accuracy half:** `docs/plans/20260815-m11-det-rerun.md`
- **Generation harness:** `~/port_kokkos_part/jobs/m11_zoo_a.sh` (built-in METIS arms),
  `scripts/m11_engines.sh` + `jobs/m11_zoo_b_dists.sh` (KaHyPar/KaMinPar/mtKaHyPar)
- **Partitions:** zoo `/work/ab0995/a270088/port2/mesh_m11/zoo/<mesh>/<arm>/dist_N` ·
  promoted `/work/ab0995/a270088/port2/mesh_m11_certified/{core2_v1,core2hil_v1,dars_v1,farc_v1}`
  with their evidence files in `docs/promotions/*.evidence`
- **Upstream patches** (partitioner changes, if the paper describes them):
  `docs/partm11/*.patch` + `docs/partm11/README.md`
- **Data:** `/work/ab0995/a270088/port2/m11/`
- **Combined-campaign numbers:** `docs/figures/m14_lever_part.png`

**Claims the chapter rests on:** load imbalance is set by bathymetry (r = 0.967 against 3-D node
count), not by 2-D node count; the winning partition is point-specific; GPU pays per *message*
(`nbr_max`) while CPU has no offline predictor. The M11 gains were all earned on Levante at ≤64
GPUs and **do not transfer to GH200 at small scale** (JUPITER findings §Q2).

**Open, and relevant if the paper wants a mechanism:** the JUPITER phase table indicates the
2-D-heavy weight trades compute balance for message shape, so the winner may differ at 512–2048
GPUs. Sixteen new NG5 partitions were generated 2026-08-17 to test exactly this — arms `a3_a0`,
`a3_a15`, `a3_a40`, `a3_a100` at 256/512/1024/2048 ranks, under
`/work/ab0995/a270088/port2/mesh_m11/zoo/ng5/`. Comparing `a3_a100` against `a5_u30` separates the
weight from the imbalance slack. **Not yet raced.**

---

## 4. Chapter: solvers and split-explicit

### 4a. The Krylov solver family
- **Primary:** `docs/SSH_SOLVERS_M10.md` (the campaign document; long, and the most careful of the
  set) · derivations `docs/plans/20260805-m10-ssh-derivations.md` · plan
  `docs/plans/20260805-m10-ssh-solvers.md` · handoff `docs/plans/20260806-m10-HANDOFF.md`
- **Source:** `src/fesom_ssh.cpp` (`CG_kk` and the variants; M10 added ~1991 lines)
- **`oati`** = the one-allreduce-per-two-iterations CG of **JPDC 163 (2022) 147–155, Alg. 4+5** —
  the citation the paper needs. Implemented as the *shallow* variant; §T7 of the M10 doc explains
  why the deep chain was rejected (no async progression on this stack).
- **Offline laboratory:** `fesom_ssh_lab` + `jobs/job_m10_lab` — symmetry check, σ-drift
  falsification, tolerance ladder. Dumped systems in
  `/work/ab0995/a270088/port2/m10/labdumps/` (note `farc_np32_fb/step0037` is a step where the
  fallback fired, with `farc_np32_ctrl` as its control).
- **Data:** `/work/ab0995/a270088/port2/m10/` · figures `.../m10/figs/` · five reports in `docs/report/`

### 4b. Split-explicit barotropic subcycling
- **Primary:** `docs/SSH_SE_M12.md` · plan `docs/plans/20260813-m12-split-explicit.md`
- **Scheme provenance:** AB3–AM4 subcycling per **Sergey Danilov's `subcycling.tex`**; the Zenodo
  FB-θ reference used for comparison is at `ssh_sergey/zenodo_se/` (gitignored — third-party)
- **Source:** `src/fesom_ssh_se.cpp` · knobs `FESOM_SSH_MODE=se` (requires `FESOM_ALE=zstar`),
  `FESOM_SE_M=<subcycles>`
- 🔴 **`FESOM_SE_M` is per-mesh. Use the CERTIFIED value, never the CFL guard minimum.**
  Certified ladder values are in `docs/SSH_SE_M12.md:25`: **CORE2 50**, fArc **90**, dars **20**.
  The startup guard prints the mesh *minimum* and aborts below it — that minimum is not a usable
  setting. CORE2's minimum is 35 and running there blows up at step 2–3 in **both** arms; my
  attempt to "add margin" to it (45) also failed, at 64 ranks. The default 50 is right for CORE2,
  *aborts* on fArc and *inverts* the verdict on dars. NG5 was never certified — its guard prints
  M_min 17 and the JUPITER campaign ran 20.
- 🔴 **The wide rung's equivalence contract:** knob-on vs knob-off is a **rounding-class pair**,
  not a byte pair. Gate it with `FESOM_SE_WIDE_SELFCHECK=1` (drift ≡ 0.0) and judge A/B state
  differences against the rounding floor — never by byte identity of coupled state. The seed is a
  3-D-born last-bit difference in `Fbt` at the ~1341 multi-claimed elements of CORE2 dist_8,
  present under both IC modes from step 3–4, which the coupled model then amplifies.
- **Wide rung (K-ring):** `docs/WIDEHALO_M12B.md` · `docs/plans/20260814-m12b-widehalo.md` +
  four session handoffs. Only **K=1** is implemented; `src/fesom_ssh_se.cpp:141` aborts on K≥2 and
  names what is missing. K≥2 is under development on JUPITER.
- **Data:** `/work/ab0995/a270088/port2/m12b/`
- **Combined-campaign numbers:** `docs/figures/m14_lever_ssh.png` (`oati` and SE as separate curves)

**Claims the chapter rests on:** split-explicit *eliminates* the barotropic solve's global
`Allreduce` rather than reducing its count, which is why its payoff keeps growing where `oati`'s
fades. That is the paper's central mechanism and the JUPITER data is its strongest evidence.

⚠️ **Fidelity caveat the paper must carry:** split-explicit is a *different discretisation*. Every
speed number for it is protocol timing only. Adoption needs the M12b fidelity gates re-run at NG5
— open item 5 of the JUPITER findings, **not yet done**.

---

## 5. Chapter: everything together

- **Plan:** `docs/plans/20260815-m14-integration-campaign.md` · **handoff (current state, and the
  place to start):** `docs/plans/20260816-m14-HANDOFF.md`
- **JUPITER campaign findings — the paper's headline scaling result:**
  `docs/plans/20260816-m14-JUPITER-scaling-FINDINGS.md` (227 jobs, 2420 node-h)
- **Tidy data, 219 rows, one row per job:** `docs/plans/20260816-m14-scaling.csv`
  (columns include `lever`, `cfg`, `wsplit`, `base`, `best`, `gain_pct`, `rejected`,
  `void_mixture`, `md5` — the last of these is load-bearing, see §0)
- **Figures, all regenerated from logs with no manual editing:**
  `scripts/m14_scaling_figs.py` → `docs/figures/m14_scaling_sstep.png`,
  `m14_scaling_sypd.png` (throughput; the figure with the story),
  `m14_speedup.png`, and the three per-lever figures.
  JUPITER-only figures: `scripts/m14_jupiter_figs.py`
- **Levante harnesses:** `jobs/job_m14_ladder_cpu`, `job_m14_ladder_gpu`, `job_m14_dolpung_ladder`
  · **JUPITER:** `jobs/job_m14_jupiter_ladder` · transfer `scripts/jupiter_fetch.sh`
  · coverage audit `scripts/m14_coverage.py`
- **JUPITER machine/environment record:** `env_jupiter.sh`, `docs/JUPITER_FLEET_RESULTS.md`,
  `docs/plans/20260723-m7-JUPITER-scaling-FINDINGS.md` (the July precursor, which supplies the
  "before" knee that the M14 result erases)

**Composition:** measured **multiplicative** to 0.1 percentage points on Levante (CORE2 at 2048
ranks: partition −21.98 % × `oati` −17.74 % predicted −35.82 %, measured −35.92 %; independently
confirmed at 1536 and 512 ranks). Details in the M14 handoff §2.

---

## 6. Still open — anything the paper claims here needs a caveat

1. **fArc `oati` stall root cause.** The σ-recurrence hypothesis has a purpose-built test
   (`fesom_ssh_lab --sigma-drift`) that died on walltime on 2026-08-06 and was re-run 2026-08-17
   (jobs 27021005 / 27021007, on the fired-fallback dump and its control). Until it returns, the
   paper should describe the fallback as a robustness limit *of the guard or of the method* —
   the two have not been separated.
2. **NG5 split-explicit fidelity** (see §4b).
3. ~~wide rung not byte-exact~~ — **CLOSED, was not a defect.** My gate tested a premise M12b
   had already retired; see §0. `jobs/job_m14_sewide_bytegate` encodes that invalid premise and
   should not be reused as written.
4. ~~plain SE NaN at 64 ranks~~ — **almost certainly the same error**: I ran M=45, below CORE2's
   certified 50. Not re-tested at 50.
5. **Partition promotions** `dars_gpu_v1` / `ng5_gpu_v1` lack their 3000-step screen.
6. **JUPITER numbers are single-allocation on a production-loaded fabric.** The campaign's ruling
   is that a loaded fabric is the deployment-relevant condition, but the largest gains carry an
   allocation-to-allocation spread the CSV does not record. Numbers marked 🔶 in the findings doc
   are single-allocation.

---

## 7. Cross-cutting material a methods section will want

**Measurement protocol** (in every M14 harness header, and worth stating once in the paper):
300-step legs · a warmup leg discarded because the first leg of an allocation is systematically
slow · both arms in ONE allocation in ABBA order · min over admitted legs · a zombie check on
stdout *and* stderr, because `rc=0` is not aliveness · `snap_every=-1` on timing runs.

**Two estimator rules, both learned by getting them wrong:**
- a speedup is only ever a **within-job** ratio — never one job's minimum divided by another's;
- absolutes take the **min over every admitted leg**, not min-of-2, because a bimodal arm is not
  reproducible at n=2.

**Meshes** (all four used throughout): CORE2 1°, fArc 4.5 km Arctic, DARS 10 km, NG5 5 km global.
🔴 CORE2 must come from the private copy `/work/ab0995/a270088/port2/mesh/core2` — `/pool`'s
`nlvls`/`elvls` are swapped. Production timesteps: CORE2 1800 s, fArc 900 s, dars/NG5 240 s;
measured at dars 120 s and NG5 180 s, with CG dt-corrections ×1.0222 / ×1.0110 available but
**not applied** in the M14 figures (stated on them).

**Machines:** Levante CPU (128-core nodes) · Levante A100 (4 GPU/node, `-C a100_80` mandatory —
the partition is heterogeneous) · JUPITER booster (quad-GH200, `FESOM_HALO_STAGE=1` is the
transport) · dolpung GH200 (42 nodes; superseded by JUPITER).

**Correctness chain:** `docs/REFERENCE_RUNS.md` (per-scheme C↔Fortran floors — every scheme has
its own), `docs/GPU_FIDELITY.md`, `docs/M1_ACCEPTANCE.md`.
🔴 **CUDA is not run-to-run reproducible on Levante A100** (threshold flips in the instability
mixing). Any bitwise claim on GPU is only meaningful as a `FORCE_SERIAL` proof or an in-run
selfcheck, never a trajectory comparison. The amended acceptance form is in the M14 handoff §3.

**Lessons file** — 300 kB, numbered `L1`–`L121` and `D1`–`D22`, and the place where most
mechanisms were first written down: `docs/KOKKOS_PORTING_LESSONS.md`.

**Port provenance** (if the paper describes the port itself, not just the optimisations):
`docs/PORT_EXPERIENCE_REPORT.md`, `docs/MPI_PORT_REPORT.md`, `docs/KOKKOS_HANDOFF.md`,
`docs/BUILD.md`.

**Earlier scaling campaigns** that supply "before" baselines: `docs/GPU_SPEED_M7.md`,
`docs/SCALING_M524.md`, `docs/SCALING_{CORE2,FARC,DARS,NG5}.md`, `docs/PROFILE_M522.md`.
