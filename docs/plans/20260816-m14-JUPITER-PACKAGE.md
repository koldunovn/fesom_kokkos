# M14 on JUPITER — transfer, checkout, build, run

**Written 2026-08-16.** Branch `m14-integrate` (worktree `~/port_kokkos_int` on Levante).
Target: JUPITER booster, quad-GH200 nodes, **up to 256 nodes = 1024 GPUs = 1024 ranks**.

This is written so you can execute it top-to-bottom from a JUPITER login node. Everything it
depends on that already exists there is named explicitly, so nothing is re-copied for no reason.

---

## 0. The one-paragraph version

The July M7 campaign already put the meshes, the stock partitions (ng5 to 8192, dars to 4096),
the 1958 forcing and the PHC initial condition on JUPITER, and it already settled the toolchain
(`env_jupiter.sh` = Stages/2025, GCC 13.3, ParaStationMPI 5.11, CUDA 12, vendored Kokkos 4.4.01)
and the transport (`FESOM_HALO_STAGE=1`). **The only new data M14 needs is the M11 optimised
partitions, about 6.3 GB** — NG5's set alone is ~4.3 GB, since an optimised partition is the
same size as the stock one it replaces. So: push the branch, `git clone` it into a new directory on JUPITER,
build once, run `scripts/jupiter_fetch.sh`, then submit the ladders in §5.

---

## 1. What M14 asks JUPITER that Levante cannot answer

M7 measured on JUPITER that **NG5 peaks at g128 (512 ranks) and *regresses ~14 % at g256***, and
dars knees at g64. M14's central finding is that **every lever attacks communication, so each
one's payoff grows with how far past the knee you are** — `oati` on NG5 goes from −0.2 % at
2048 CPU ranks to −15.8 % at 40960; fArc SE from −2.5 % to −47.5 %.

Those two facts collide exactly at 256 nodes. The question is therefore sharp and worth the
allocation: **do the M14 levers move NG5's knee outward and convert the g256 regression into a
gain?** Levante cannot ask it — its GPU partition caps out at 16 nodes.

Secondary: a GH200 partition-lever number at scale. M11's partition gains were all measured on
Levante, and the Levante GPU ladder could not even express the lever (its harness had no
`BEST_MESH` arm until 2026-08-16), so **there is no GPU partition measurement anywhere in this
project above 64 GPUs.**

---

## 2. Push the branch (on Levante, needs your OK)

```bash
cd ~/port_kokkos_int
git log --oneline -1                      # expect the latest M14 commit
git push -u origin m14-integrate          # 🔴 standing rule: never pushed without asking
```

The branch is `m14-integrate` = `m13-cg-robustness` + `m12b-widehalo` + `m11-partition` +
`m9-mevp-double` + `m10-ssh-solvers` + knob hygiene. Knobs-off is byte-identical to `main` on
Serial (gate jobs 26983815 / 26983871 / 26983939 / 26984090).

---

## 3. Check out and build on JUPITER

```bash
ssh <you>@jupiter.fz-juelich.de
cd /e/home/jusers/koldunov1/jupiter
git clone -b m14-integrate https://github.com/koldunovn/fesom_kokkos.git port_kokkos_m14
cd port_kokkos_m14

# modules do NOT persist — source in the same command as the build, and again inside every srun
source env_jupiter.sh
mkdir -p build-jupiter-cuda && cd build-jupiter-cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DFESOM_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON
make -j32
```

Notes carried from the July campaign, all of them earned:

- **Stages/2025, not 2026.** Stage-2026 (Kokkos 4.7.03 + CUDA 13 + PSMPI 5.13) is ~2 % faster at
  core2 g1/g2 and **11–20 % slower on the big meshes at scale** (ng5_g32 0.0869 vs 0.0723;
  ng5_g128 0.0657 vs 0.0570). `env_jupiter_s26.sh` exists as the documented fallback.
- **Host compiler is irrelevant** — clang 20.1.8 measured equal to gcc (0.0426 vs 0.0423 at
  core2 g1); nvcc/ptxas make the device SASS. Don't spend time here.
- **Prove CUDA-awareness before believing any halo crash** (L77 as a class): an MPI built without
  CUDA support segfaults on device pointers, and on Levante that masqueraded perfectly as a merge
  regression. Check `ldd` on the binary and confirm `MPI-settings/CUDA` is loaded.
- If the build errors on an aarch64/x86 mismatch, it is a module-purge problem, not a code
  problem — `env_jupiter.sh` starts with `module --force purge` for exactly that reason.

**Smoke test before anything else** (needs `dist_1`, which is in the private CORE2 copy):

```bash
source env_jupiter.sh
srun -A e-sta-destine -p booster -N1 -n1 -c72 --gres=gpu:4 --gpu-bind=none \
  ./build-jupiter-cuda/fesom_port \
  /e/scratch/e-sta-destine/koldunov1/meshes/core2_private $PWD/smoke 1800 10 -1 \
  /e/scratch/hclimrep/koldunov1/meshes/phc3.0/phc3.0_winter.nc 1958
```

---

## 4. Fetch the data (run ON JUPITER)

```bash
cd /e/home/jusers/koldunov1/jupiter/port_kokkos_m14
DRYRUN=1 bash scripts/jupiter_fetch.sh     # look first
bash scripts/jupiter_fetch.sh
```

It pulls from Levante over ssh, so `ssh a270088@levante.dkrz.de` has to work from the JUPITER
login node (`ssh-copy-id`, or forward your agent). Nothing writes back to Levante.

| what | where it lands | size | status |
|---|---|---|---|
| **M11 optimised partitions** | `$SCRATCH/meshes_m11/<mesh>/<engine>/dist_N` | **6.3 GB** (ng5 4.3, dars 1.6, farc 0.3, core2 0.05) | **NEW — the real payload** |
| M11 certified promotions | `$SCRATCH/meshes_m11/certified/` | small | new |
| mesh statics (7 files/mesh) | `$SCRATCH/meshes/<mesh>/` | 3.1 GB | already there; guarded |
| stock `dist_N` | `$SCRATCH/meshes/<mesh>/dist_N` | 6.3 GB | already there to ng5 8192 |
| JRA55 1958 forcing | `$SCRATCH/forcing_1958/` | 11 GB | already there |
| PHC IC + Sweeney chl | `/e/scratch/hclimrep/koldunov1/meshes/` | 33 MB | already there |

Two things worth knowing about the mesh copy:

- **The port opens exactly seven static files** — `nod2d.out elem2d.out aux3d.out nlvls.out
  elvls.out edges.out edge_tri.out`. NG5's source directory is 27 GB; those seven are **2.0 GB**.
  The rest is `fesom.mesh.diag.nc`, IFS griddes files and pyfesom caches, none of it read.
- **CORE2 must come from the private copy** `/work/ab0995/a270088/port2/mesh/core2`, never
  `/pool` — `/pool`'s `nlvls`/`elvls` are swapped (L73). It is the gate mesh: copy it, never
  regenerate it.

A complete `dist_N` holds `rpart.out` plus one `com_info<rank>.out` and one `my_list<rank>.out`
per rank — exactly **2N+1 files**. The script prints an inventory at the end; check it.

### 4b. Partition availability

Stock partitions cover every rung to 1024 already. The optimised ones are being generated on
Levante now (jobs 27001771/72/74/75, `jobs/m11_zoo_a.sh`), engine = M11's per-mesh GPU winner:

| mesh | engine | had | generating |
|---|---|---|---|
| core2 | `a5_u30` | 4, 128, 256, 512, 864, 1024 | 8, 16, 32, 64 |
| farc | `a5_u30` | 64 | 4, 8, 16, 32, 128 |
| dars | `a4u30` | 64 | 16, 32, 128, 256, 512 |
| ng5 | `a5_u30` | 64, 2048, 4096, 8192, 16384, 40960 | 16, 32, 128, 256, 512, 1024 |

🔴 **The fetch script copies only the rungs a 256-node run can reach.** The engine directories on
Levante also carry CPU-scale partitions — `ng5/a5_u30` goes up to `dist_40960`, dars/farc/core2 to
2048 — and rsyncing a whole engine directory would pull all of them for nothing.

🔴 **Treat these as candidates, not winners.** M11's own conclusion is that the best partition is
point-specific, and every M11 number was earned on Levante hardware — GPU pays per *message*
(`nbr_max`) and JUPITER's fabric is not Levante's. Race them (§5), don't assume them.

---

## 5. What to run

All rungs are **nodes × 4 = ranks**. Ladders stop where the mesh stops paying, so the 256-node
runs go to NG5 where the open question is.

### 5.1 Gates first — three jobs, cheap

| gate | command | pass bar |
|---|---|---|
| G1 smoke | the §3 smoke line | rc=0, 10 steps, finite |
| G2 knob liveness | `POINT=core2 ARMS="base best" BEST_KNOBS=FESOM_SSH_SOLVER=oati`, 1 node | the arms differ |
| G3 wsplit fires | `POINT=ng5 ARMS=base`, 4 nodes | log says `wsplit    : wsplit1`, `[wsplit] FESOM_WSPLIT = ON` |

Do not skip G3. See §7.

### 5.2 The baseline ladders (arm: `base` only)

```bash
cd /e/home/jusers/koldunov1/jupiter/port_kokkos_m14
for n in 1 2 4 8 16;             do sbatch --nodes=$n --time=00:40:00 --export=ALL,POINT=core2,ARMS=base jobs/job_m14_jupiter_ladder; done
for n in 1 2 4 8 16 32;          do sbatch --nodes=$n --time=00:40:00 --export=ALL,POINT=farc,ARMS=base  jobs/job_m14_jupiter_ladder; done
for n in 4 8 16 32 64 128;       do sbatch --nodes=$n --time=01:00:00 --export=ALL,POINT=dars,ARMS=base  jobs/job_m14_jupiter_ladder; done
for n in 4 8 16 32 64 128 256;   do sbatch --nodes=$n --time=01:30:00 --export=ALL,POINT=ng5,ARMS=base   jobs/job_m14_jupiter_ladder; done
```

### 5.3 The lever pairs (arm: `base best`, both in one allocation, ABBA)

`oati` — the SSH solver, the lever that grows fastest with scale, and the one that should decide
the NG5 g256 question:

```bash
for n in 4 8 16 32 64 128 256; do sbatch --nodes=$n --time=02:00:00 \
  --export=ALL,POINT=ng5,ARMS="base best",BEST_KNOBS="FESOM_SSH_SOLVER=oati" \
  jobs/job_m14_jupiter_ladder; done
for n in 4 8 16 32 64 128; do sbatch --nodes=$n --time=01:30:00 \
  --export=ALL,POINT=dars,ARMS="base best",BEST_KNOBS="FESOM_SSH_SOLVER=oati" \
  jobs/job_m14_jupiter_ladder; done
for n in 2 4 8 16 32; do sbatch --nodes=$n --time=01:00:00 \
  --export=ALL,POINT=farc,ARMS="base best",BEST_KNOBS="FESOM_SSH_SOLVER=oati" \
  jobs/job_m14_jupiter_ladder; done
```

Split-explicit SSH — 🔴 **`FESOM_SE_M` is per-mesh and the default 50 is wrong on both**: it
ABORTS on fArc (`dtbt 18.00 s > limit 11.09 s → M_min=82`) and *inverts the verdict* on dars
(+2.32 % at M=50, −3.72 % at the tuned M=20). SE also requires zstar on both arms:

```bash
for n in 4 8 16 32; do sbatch --nodes=$n --time=01:00:00 \
  --export=ALL,POINT=farc,ARMS="base best",BASE_KNOBS="FESOM_ALE=zstar",BEST_KNOBS="FESOM_SSH_MODE=se FESOM_SE_M=90" \
  jobs/job_m14_jupiter_ladder; done
for n in 16 32 64 128; do sbatch --nodes=$n --time=01:30:00 \
  --export=ALL,POINT=dars,ARMS="base best",BASE_KNOBS="FESOM_ALE=zstar",BEST_KNOBS="FESOM_SSH_MODE=se FESOM_SE_M=20" \
  jobs/job_m14_jupiter_ladder; done
```

Partitioning — the measurement that exists nowhere above 64 GPUs:

```bash
M=/e/scratch/e-sta-destine/koldunov1/meshes_m11
for n in 4 16 64 128 256; do sbatch --nodes=$n --time=02:00:00 \
  --export=ALL,POINT=ng5,ARMS="base best",BEST_MESH=$M/ng5/a5_u30 \
  jobs/job_m14_jupiter_ladder; done
for n in 4 16 64 128; do sbatch --nodes=$n --time=01:30:00 \
  --export=ALL,POINT=dars,ARMS="base best",BEST_MESH=$M/dars/a4u30 \
  jobs/job_m14_jupiter_ladder; done
```

**Sea ice wide halo: do NOT run it here.** It is a win on A100 (up to −19.4 %) and a measured
**loss** on GH200 and on CPU. It trades messages for replicated ghost work, so it pays only where
a flat per-message staging cost dominates — which is an A100 property, not a GH200 one.

### 5.4 Composition, at the two points that matter

Composition measured multiplicative on Levante to 0.1 pp (CORE2 2048: partition −21.98 % ×
`oati` −17.74 % ⇒ predicted −35.82 %, measured −35.92 %). Worth one confirmation at scale:

```bash
sbatch --nodes=128 --time=02:00:00 --export=ALL,POINT=ng5,ARMS="base best",\
BEST_MESH=$M/ng5/a5_u30,BEST_KNOBS="FESOM_SSH_SOLVER=oati" jobs/job_m14_jupiter_ladder
sbatch --nodes=256 --time=02:30:00 --export=ALL,POINT=ng5,ARMS="base best",\
BEST_MESH=$M/ng5/a5_u30,BEST_KNOBS="FESOM_SSH_SOLVER=oati" jobs/job_m14_jupiter_ladder
```

---

## 6. Reading the results

Each job prints a `JCSV` line: `JCSV <mesh>,<ranks>,cfg=<shared config>+<wsplit state>,base=…,best=…`.
Harvest with `grep -h JCSV /e/scratch/e-sta-destine/koldunov1/port2/m14/jlad.*.out`.

Two estimator rules, both of which this project has already violated once and paid for:

- **A speedup is a WITHIN-job ratio.** Both arms shared an allocation there. Never divide one
  job's min by another job's min — on Levante that manufactured a "−10.1 %" at a point whose
  baseline legs had all been rejected.
- **Absolute numbers take the min over every admitted leg**, not min-of-2. A 4–5 % bimodal arm
  is not reproducible at min-of-2.

And: a wsplit-on row and a wsplit-off row are different configurations. `cfg=` keeps them apart —
do not merge them into one curve.

---

## 7. Traps, in the order they will bite

1. 🔴 **wsplit.** ON for fArc/dars/NG5 — the job defaults it that way from `POINT`, and prints
   `wsplit    : wsplit1`. Without it a cold PHC start at the protocol dt rides the M5.24 vertical
   CFL blow-up (rule 0.41). It surfaces as `[fesom_port FATAL] CG_kk: pp·App is -nan` with a
   **roundoff-seeded onset step**, so identical runs die anywhere from step 4 to 291 and it reads
   as a random solver bug. It cost the Levante A100 campaign its entire NG5 arm (7 of 37 legs
   survived) and was written up as an unresolved regression blocking the merge. The FATAL goes to
   **stderr**, not stdout.
2. 🔴 **`FESOM_HALO_STAGE=1` is the transport**, set by the job for every leg. Device-pointer MPI
   is broken on Stages/2025 PSMPI 5.11 and loses badly at scale on 2026. Never `HOST_HALO` — it
   kills CGPIPE/CGPOLY.
3. 🔴 **Modules do not persist.** `env_jupiter.sh` is sourced by the batch script *and* again
   inside each `srun bash -c`. Keep it that way.
4. **Budget the `det` fill.** `FESOM_IC_EXTRAP=det` is on in both arms (a correctness fix, not a
   speed knob). It is startup cost, not per-step, but it grows with mesh and falls with rank
   count — NG5 at 64 ranks measured ≈7 min against 71 s of actual compute. Four Levante jobs died
   on this. The reported number is the model's own per-step timer, so the timing is unaffected;
   the *walltime* is not.
5. **Discard the warmup leg.** The first leg of an allocation is systematically slow. The job
   already runs and drops one.
6. **`snap_every=-1`** on every timing run; mandatory at ≥4096 ranks anyway.
7. **The zombie check is not optional.** `rc=0` is not aliveness — a NaN-blind guard once turned
   a broken run into a leg that measured 10.8 % *faster* than a healthy one. The job checks the
   final step line for presence, finiteness and a non-zero iteration count, and greps stderr for
   the CG NaN.
8. **Allocation variation is real and rung-dependent** (1 % at 512 ranks, 24 % at 1536 on
   Levante), worst where the mesh is latency-bound. It does not affect A/B gains — both arms
   always share one allocation — but it does affect ladder *shape*. Do not redraw a knee from one
   allocation; N=5 on Levante showed a single "control" was itself the outlier.

---

## 8. Cost estimate

At 300 steps per leg, 5 legs per pair job (warmup + ABBA):

| ladder | node-hours (rough) |
|---|---|
| baselines, all four meshes | ~120 |
| `oati` pairs (ng5+dars+farc) | ~400 |
| SE pairs (farc+dars) | ~150 |
| partition pairs (ng5+dars) | ~350 |
| composition at 128+256 | ~180 |
| **total** | **~1200 node-hours** |

The 256-node NG5 points dominate. If the allocation is tight, drop in this order: composition →
farc `oati` → dars partition. Keep NG5 `oati` and NG5 partition at 128 and 256 — that pair of
points is the whole reason to be on JUPITER.
