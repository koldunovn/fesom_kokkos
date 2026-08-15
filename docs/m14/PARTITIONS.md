# M14 Task P1 — partition inventory

**Date:** 2026-08-15 · enumerated from `/work/ab0995/a270088/port2/` and `/pool/data/AWICM/FESOM2/`

Purpose: mark every (mesh, backend, rank-count) cell of the C3 ladders as **have** /
**must-generate** / **impossible**, for the **base** arm and for the **best** (partition-lever) arm
separately. The base-arm answer decides whether a ladder point can run at all; the best-arm answer
decides whether the partition lever is even present at that point.

---

## Two corrections to the plan (Rev 2 → Rev 3)

1. **CORE2 CPU above 864 already exists.** Rev 2 said it "does not exist today and must come from
   P2". `/work/.../port2/mesh/core2_bigpart` carries `dist_{864,1024,1536,2048}`. The M10 doc's
   *"CORE2 has no 1024-rank partition; 864 is the largest that exists"* was true when M10 was
   written and has since been superseded by the bigpart trees. **No CORE2 generation needed** — but
   the job must switch `MESH=` from `mesh/core2` to `mesh/core2_bigpart` at 1024.
2. **fArc is the only generation task, and its ladder is shorter than planned.** The largest fArc
   partition anywhere is **2304** (`/pool/.../MESHES_FESOM2.1/farc`, and `fArc_sorted` tops out at
   6912 but is a different, sorted mesh). fArc CPU 4096 and 8192 do not exist.

---

## Base arm — can the point run at all?

### CPU

| mesh | rank | source tree | status |
|---|---|---|---|
| CORE2 | 128, 256, 432, 512, 864 | `port2/mesh/core2` (private — standing rule, never `/pool` CORE2) | **have** |
| CORE2 | 1024, 1536, 2048 | `port2/mesh/core2_bigpart` | **have** (`MESH=` switches here) |
| fArc | 512, 1024, 2048 | `/pool/…/MESHES_FESOM2.1/farc` | **have** |
| fArc | 4096, 8192 | — | **must-generate** (max existing is 2304) |
| dars | 1024, 2048, 4096 | `/pool/…/MESHES_FESOM2.1/dars` | **have** |
| dars | 6144, 8192 | `port2/mesh/dars_bigpart` (also 10240) | **have** |
| NG5 | 2048, 4096, 8192 | `/pool/…/MESHES_FESOM2.1/ng5` | **have** |
| NG5 | 16384 | `port2/mesh/ng5_bigpart` (also 20480, 24576, 32768) | **have** |

### GPU (ranks = nodes × 4 A100)

| mesh | nodes → ranks | status |
|---|---|---|
| CORE2 | 1,2,4,8,16 → 4,8,16,32,64 | **have** (all in `port2/mesh/core2`) |
| fArc | 1,2,4,8,16 → 4,8,16,32,64 | **have** |
| dars | 2,4,8,16 → 8,16,32,64 | **have** |
| NG5 | 4,8,16 → 16,32,64 | **have** |

**Base arm is fully covered except fArc CPU 4096 and 8192.** Every GPU point exists.

---

## Best arm — where does the partition lever actually exist?

`M11_RECOMMENDATION.md:265-281` — the certified set is **four usable points**, each produced by a
**different partitioner**, each selected at **one** rank count:

| certified mesh | dist | partitioner | gain | M14 use |
|---|---|---|---|---|
| `core2_v1` | `dist_4` | METIS `MINCONN` | GPU −8.1 % | CORE2 GPU **1 node** |
| `core2_v1` | `dist_512` | KaHIP `UFACTOR=30` | CPU −3.8 % | CORE2 CPU **512** |
| `farc_v1` | `dist_2048` | Mt-KaHyPar `w=100+nlev` | −7.5 % | fArc CPU **2048** |
| `dars_v1` | `dist_2048` | KaMinPar `w=100+nlev` | −4.2 % | dars CPU **2048** |
| `core2hil_v1` | `dist_512` | Hilbert renumber + KaHIP | −5.8 % | ⛔ **EXCLUDED from M14** |

Pending the user's word (`:278-281`) — *"the dars 64 and NG5 64 MINCONN winners are not yet
packaged … say the word and they promote as `dars_gpu_v1` / `ng5_gpu_v1`; both already satisfy
`m11_promote`'s stability requirement (screens 26895260 / 26908635)"*:

| candidate | dist | M14 use |
|---|---|---|
| `dars_gpu_v1` | `dist_64` | dars GPU **16 nodes** |
| `ng5_gpu_v1` | `dist_64` | NG5 GPU **16 nodes** |

`core2hil_v1` is excluded because it **renumbers the mesh**, which re-baselines every C↔K floor in
`docs/REFERENCE_RUNS.md` — unacceptable inside a campaign whose acceptance bar is bitwise identity
to `main`.

The `port2/mesh_m11/*_m11/` trees (`core2_m11`, `farc_m11`, `dars_m11`, `ng5_m11`, and the many
`*_seed*` trees) are **M11 campaign working arms, not products**. They are not certified, and M11's
own warning is that a partitioner winner is point-specific — a re-roll is +8…+15 % slower at
CORE2 864 and +3…+8 % at fArc. They are not promoted into M14 on their own.

---

## Consequence for the campaign — state this on the figures

**The partition lever exists at 4 of ~40 ladder points** (6 if `dars_gpu_v1` / `ng5_gpu_v1`
promote). At every other point the best arm runs the **same partition as the base arm**, and the
"all knobs and partitions applied" curve is carrying the SSH and ice levers only.

This is not a defect to engineer around — it is what M11 actually certified, and manufacturing a
partition per ladder point would mean shipping ~34 untested draws into a speed board, against
M11's explicit finding that an untested re-roll is frequently **slower**. The honest presentation
is a per-point marker showing where the partition lever is live.

It also means the D2 interaction hunt can only test `partition × comm lever` at the four certified
points — which is exactly why D2 was cut from 8 probes to 4.

---

## Actions

- [ ] **P2-a**: generate fArc CPU 4096 and 8192 (base arm), or shorten the fArc CPU ladder to
      512/1024/2048 and note that fArc's turnover cannot be shown without new partitions
- [ ] **P2-b**: ask the user about promoting `dars_gpu_v1` / `ng5_gpu_v1` (Task C1)
- [ ] **P2-c**: `jobs/m14_config.sh` must carry `MESH=` per point — CORE2 switches tree at 1024,
      dars at 6144, NG5 at 16384
- [x] **P1**: inventory complete; no cell is "impossible" except fArc CPU > 2304 pending generation
