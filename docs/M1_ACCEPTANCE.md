# M1 acceptance — 1-yr CORE2 bit-identity (the `m1-datalayer` milestone gate)

The per-change pi-smoke gate (handoff §2) proves the M1 data layer is bit-identical on a tiny mesh.
The **milestone** gate is heavier: each Kokkos backend reproduces the **C twin** byte-for-byte over
**one model year of CORE2** (a real partitioned mesh, a full year of JRA55 forcing, sea ice, GM,
KPP). Passing it → `git tag m1-datalayer` → M2.

## Why this is a fresh run

No CORE2 C reference was ever captured (plan M0.1 deferred it: "CORE2 golden ref … capture before
M3"). So the oracle must be **generated now** by running the validated C twin
(`/home/a/a270088/port2/fesom2_port/build/fesom_port`, SHA 75de623) at the acceptance config. All
four runs share one config block (`MESH`, `DT=1800`, `NSTEPS=17280` = 360-day year, `SNAP_EVERY=1440`
= monthly, PHC, `JRA55_YEAR=1958`) — see the `jobs/job_m1accept_*` headers.

## Procedure

```bash
cd /home/a/a270088/port_kokkos
# 1. C-twin reference (the oracle) — ~1-2 h on 256 ranks/2 nodes:
sbatch jobs/job_m1accept_cref
# 2. Kokkos backends (can run concurrently with the C-ref; only the COMPARE waits on the C-ref):
sbatch jobs/job_m1accept_serial
sbatch jobs/job_m1accept_omp
#    (CUDA: see §CUDA below — deferred to M3.1)
# 3. After all finish, compare each backend vs the C-ref:
scripts/m1_accept_compare.sh        # → "all present backends BIT-IDENTICAL to the C twin."
# 4. On success:
git tag m1-datalayer
```

Outputs land in `/scratch/a/a270088/m1_accept/{cref,serial,omp[,cuda]}/`.

## §ranks — why all runs use the SAME 256-rank layout

Bit-identity is guaranteed only at a **fixed MPI rank count**. Across different rank counts the halo
exchange order and the CG solver's `MPI_Allreduce` dot-product order change, giving climate-identical
but not byte-identical fields (this is exactly why the D14 / np=2 gate compares same-rank-count, and
why L18's gather is rank-count-deterministic only *within* a layout). So cref / serial / omp all run
**256 ranks** (`dist_256` exists). At M1 the **OpenMP backend has no `parallel_for`**, so it executes
the identical single-threaded host code as Serial — `OMP_NUM_THREADS=1`, threads unused; a genuine
multi-thread climate-identity check belongs after the first device/threaded kernel (M2+), not here.

## §CUDA — deferred to M3.1, and why that is sound

Running CUDA on CORE2 needs the **multi-GPU rank→device mapping** that is plan task **M3.1**
(`--kokkos-num-devices` / local-rank→`cudaSetDevice`, `dist_<#gpus>` partitions) — not yet built. At
M1 the CUDA backend runs **zero device compute** (the only device op anywhere is `deep_copy` of
`double`/`int`); its data-layer bit-identity is therefore **mesh-size-independent** and is already
proven on the pi smoke (np=1, A100, ALL FIELDS BIT-IDENTICAL, every milestone M0→M1.5). A 126k-node
CORE2 deep_copy is the same operation as a 3k-node pi deep_copy.

Two honest options (a decision for the run owner):
1. **Defer** the CUDA CORE2 row to M3.1 (tag `m1-datalayer` on Serial+OpenMP CORE2 + the CUDA pi
   proof). Lowest cost; the CUDA data layer is already demonstrated.
2. **Pull M3.1 forward**: add the rank→GPU mapping, then run CUDA CORE2 at a GPU-feasible rank count
   (e.g. `dist_16` across 4 A100 nodes) — but then cref/serial must ALSO be re-run at 16 ranks to
   compare same-rank-count (§ranks).

This file records the choice made; update the tag note in `docs/KOKKOS_HANDOFF.md` accordingly.

## If a backend "diverges"

Follow the §C debugging ladder in `docs/KOKKOS_PORTING_LESSONS.md` BEFORE suspecting the port: on
compute nodes the login-node vader-CMA artifact (L18) does **not** apply (real UCX/IB), but still
(1) rebuild after `touch src/*` if a layout changed, (2) dump the OWNED state right after the
producing kernel and `cmp` — byte-identical owned ⇒ the divergence is in the gather/transport, not
the physics. `diff_snap.py` takes DIRECTORIES (L19).
