# Session prompt — run the M14 campaign on JUPITER

*Paste this as the opening prompt of a fresh session on JUPITER (or read it yourself). It is
self-contained: everything you need to know is here or named by path. Written on Levante,
2026-08-16, by the session that prepared the package.*

---

## Who you are and what this is

You are continuing the FESOM2 C++/Kokkos port campaign. FESOM2 is an unstructured-mesh ocean
model; this repo is a Kokkos port of it that runs device-resident on GPU. Five optimisation
campaigns (M9 sea ice, M10 SSH solvers, M11 partitioning, M12b wide halo, M13 deterministic
initial conditions) were merged into one branch, `m14-integrate`, and measured on Levante.

**Your job on JUPITER is to run the combined strong-scaling campaign at a scale Levante cannot
reach: up to 256 quad-GH200 nodes = 1024 GPUs = 1024 ranks.** Levante's GPU partition caps at
16 nodes, and there is a standing rule not to exceed it.

**Read `docs/plans/20260816-m14-JUPITER-PACKAGE.md` first.** It is the execution document —
build, fetch, the job list with concrete `sbatch` lines, how to read results, the traps, and a
node-hour estimate. This prompt is the framing and the handoff state around it.

---

## The two questions worth 1200 node-hours

**1. Does M14 move NG5's knee outward?** In July the M7 campaign measured on this machine that
NG5 peaks at **g128 (512 ranks) and regresses ~14 % at g256**. M14's central finding is the
opposite-facing one: every lever it added attacks *communication*, so each lever's payoff grows
with how far past the knee you are — `oati` on NG5 goes from −0.2 % at 2048 CPU ranks to −15.8 %
at 40960; fArc split-explicit from −2.5 % to −47.5 %. Those two facts meet exactly at 256 nodes.
Either the levers convert that regression into a gain, or they don't, and both answers are worth
having.

**2. What does the partition lever do on GPU at scale?** M11 measured real GPU gains (dars 64 GPU
−18.5 %, NG5 64 GPU −9.6 %, CORE2 4 GPU −7.1 %, fArc 16 GPU −3.9 %) — but all on Levante, all at
≤64 GPUs, and all before the merge. The Levante GPU ladder could not even *express* the lever
until 2026-08-16: its harness had no `BEST_MESH` arm, so the best arm silently ran the same mesh
as the base. **There is no GPU partition measurement anywhere in this project above 64 GPUs.**

Everything else on the list is supporting evidence for those two.

---

## State at handoff

**Pushed:** branch `m14-integrate` at `github.com/koldunovn/fesom_kokkos`, HEAD `aa1a442`.
Knobs-off is byte-identical to `main` on Serial.

**Already on JUPITER from the July M7 campaign** (do not re-fetch): meshes, stock partitions
(ng5 to 8192, dars to 4096), the 1958 JRA55 forcing, the PHC initial condition, and the settled
toolchain — `env_jupiter.sh` = Stages/2025, GCC 13.3, ParaStationMPI 5.11, CUDA 12, vendored
Kokkos 4.4.01, with `FESOM_HALO_STAGE=1` as the transport.

**Generated on Levante for this trip** — the M11 optimised partitions, engine = M11's per-mesh
GPU winner. `scripts/jupiter_fetch.sh` pulls exactly the reachable rungs (~6.3 GB):

| mesh | engine | rungs |
|---|---|---|
| core2 | `a5_u30` | 4, 8, 16, 32, 64 ✅ |
| farc | `a5_u30` | 4, 8, 16, 32, 64, 128 ✅ |
| dars | `a4u30` | 16, 32, 64, 128, 256, 512 ✅ |
| ng5 | `a5_u30` | 16, 32, 64, 128, 256 ✅ — **512 and 1024 were still generating at handoff** |

🔴 **Check NG5 512 and 1024 exist on Levante before you plan the 128- and 256-node partition
pairs.** `ssh a270088@levante.dkrz.de 'ls -d /work/ab0995/a270088/port2/mesh_m11/zoo/ng5/a5_u30/dist_*'`.
If they are missing, the generating job was 27001775 (`jobs/m11_zoo_a.sh`); re-run it with
`MESHTAG=ng5 SRC=/work/ab0995/a270088/port2/mesh_m11/ng5_m11 ARMS=a5_u30 A5_A=100 RANKS="512 1024"`.

🔴 **These partitions are CANDIDATES, not winners.** M11's own conclusion is that the best
partition is point-specific; GPU pays per *message* (`nbr_max`) and JUPITER's fabric is not
Levante's. Race them in one allocation, never assume them.

---

## The one genuinely open problem — read before you interpret any failure

On Levante's A100 partition, NG5 lost its entire GPU arm to
`[fesom_port FATAL] CG_kk: pp·App is -nan` — **7 of 37 timing legs survived**, hitting both arms
alike. It is **not a solver bug**. It is the M5.24 cold-start vertical CFL blow-up (rule 0.41),
and three things make it maximally deceptive:

- **The onset step is roundoff-seeded.** Identical binary, config and initial condition die
  anywhere from step 4 to step 291, sometimes within one allocation. A real physics instability
  dies at a fixed step; this does not. If you see random death steps, this is what you have.
- **A 35-step leg sits under the wall.** `job_m7_ab_env` defaults `NSTEPS=35`, which is why the
  July record looks clean and everyone remembers NG5 GPU "working fine". M7's own words:
  *"'dt180 stable' was only ever a 35-step statement."*
- **The FATAL goes to stderr, not stdout.** Grepping `run.log` alone reads as "no NaN anywhere".
  That cost this project a wrong diagnosis. The JUPITER ladder's zombie check greps both.

**The believed cure is `wsplit`, and it is NOT yet confirmed.** The July Fortran discriminator is
solid — Fortran *without* wsplit dies in the same window (rc=1, NaN after 261 CFLz warnings),
Fortran *with* wsplit completes. But every M14 leg ran with it off and nothing in the log said so,
and when it was finally switched on (Levante jobs 27001723/24/25), **legs still died**: at 32 GPUs
the warmup and one `best` leg failed with wsplit demonstrably ON
(`[wsplit] FESOM_WSPLIT = ON (maxcfl=1.00)`, matching the Fortran namelist default), while a
`base` leg completed. Those jobs were still running at handoff — **read them before trusting or
dismissing wsplit**:
`ssh a270088@levante.dkrz.de 'grep -E "wsplit|^leg |min=|GAIN" /work/ab0995/a270088/port2/m14/gladder.270017{23,24,25}.out'`

🔴🔴 **UPDATE, same day, jobs 27001723/24 finished — wsplit WORKS for the baseline and the
failure is now `oati`-SPECIFIC.** NG5 A100 at 16 and 32 GPUs with `FESOM_WSPLIT=1` demonstrably on:

| arm | legs completing 300 steps |
|---|---|
| baseline config (incl. warmup) | **4 of 5** — was 5 of 21 with wsplit off |
| `FESOM_SSH_SOLVER=oati` | **0 of 4** — died at steps 96, 115, 117, 190 |

Before wsplit, both arms died at similar rates (24 % vs 13 %) and the failure was clearly
configuration-wide. With wsplit on, the baseline runs and **only the `oati` arm dies.** That is a
different problem wearing the same costume: it points at the M10 SSH solver on NG5 GPU, not at the
rule-0.41 cold-start class. Note also that the two 16-GPU `oati` deaths carry **no CG NaN message
at all** (the 32-GPU ones do), so there may be two distinct signatures.

**Consequence for this trip, act on it before spending an allocation:** `oati` is the headline
lever in §5.3 and the whole NG5 g256 question rests on it. **Run the NG5 `oati` pair at 4 nodes
FIRST and confirm both arms complete 300 steps.** If the `oati` legs die there too, do not submit
the 64/128/256-node NG5 `oati` jobs — switch the NG5 question to the partition lever (which has
no such problem) and report the `oati` failure as a finding. n is 4 legs, so confirm rather than
assume, but the contrast with 4-of-5 is not subtle.

If wsplit had not helped, the live hypothesis would have been that the *port's* wsplit
implementation still does not match Fortran's — M5.24 recorded "CG NaN ~step 85 when enabled" and M7 left "debug
port-wsplit to bit-match Fortran-wsplit" as an open work item that may never have been closed at
NG5 scale. That would be a real finding, and it would mean NG5 GPU numbers need a shorter
measurement window or a different dt until it is fixed. **Do not report it as a merge regression:
`h17`, the pre-campaign certified binary, fails the same condition today (bisect 26997395, 0/2
clean at NG5 64 GPUs) while the merged `i3` swept 2/2.**

On JUPITER the ladder defaults wsplit ON for fArc/dars/NG5, off for CORE2, prints
`wsplit    : wsplitN`, and stamps it into `cfg=`. **Gate G3 in the package doc exists to prove it
fired. Do not skip it.**

---

## How to work

Follow §3–§5 of the package doc in order: build → smoke → fetch → three gates → baseline ladders
→ lever pairs. Do not submit a 256-node job before a 4-node job of the same shape has passed.

**Two estimator rules this project has already violated and paid for:**

1. **A speedup is a WITHIN-job ratio.** Both arms shared an allocation. Never divide one job's
   min by another job's min — on Levante that manufactured a "−10.1 %" at a point whose baseline
   legs had all been rejected, so the pair never existed.
2. **Absolutes take the min over every admitted leg**, not min-of-2. A 4–5 % bimodal arm is not
   reproducible at min-of-2, and a previous session quoted a number that did not survive a repeat.

**And three habits:**

- `rc=0` is not aliveness. A NaN-blind guard once turned a broken run into a leg that measured
  10.8 % *faster* than a healthy one. The ladder's zombie check is not decoration.
- Allocation variation is real and rung-dependent (1 % at 512 ranks, 24 % at 1536 on Levante),
  worst where the mesh is latency-bound. It does not touch A/B gains — both arms always share one
  allocation — but never redraw a ladder's *shape* from one allocation. N=5 on Levante showed a
  single "control" was itself the outlier.
- A wsplit-on row and a wsplit-off row are different configurations. `cfg=` keeps them apart; do
  not merge them into one curve or one min.

**Measure, don't guess.** If you find an anomaly, the answer is a control job or an explicit
"untested — candidate explanation only" label, never a mechanism stated as established. Every
coverage gap in the Levante campaign was found by the user asking, not by the session checking.

---

## What to produce

1. A `JCSV` harvest of every point:
   `grep -h JCSV /e/scratch/e-sta-destine/koldunov1/port2/m14/jlad.*.out`
2. Answers to the two questions above, stated as measurements with their conditions.
3. Anything that contradicts a Levante finding — those are the valuable results, not the
   confirmations. In particular: the sea-ice wide halo is a win on A100 and a measured loss on
   GH200, so **do not run it here**; and if `oati`'s payoff does *not* keep growing past g128,
   that falsifies the campaign's unifying mechanism and needs saying plainly.
4. A findings doc alongside `docs/plans/20260723-m7-JUPITER-scaling-FINDINGS.md`, and figures via
   the harvester pattern in `scripts/m14_scaling_figs.py` (read `scripts/m7_scaling_figs.py`
   first — standing rule on figure conventions).

Ask before anything expensive or irreversible: large allocations, pushes, deletions.
