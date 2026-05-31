# Next session — M5.23: harvest the COMM-regime levers (toward 1–2 SYPD on NG5)

*Paste this whole file to start. Self-contained. Written 2026-05-31 at the close of the M5.22 deep-profile + the first M5.23 comm lever (L1). A continuation of an IMPLEMENT+VALIDATE campaign: the budget is established, the lever menu is mapped + code-read, and L1 is landed & validated. Pick the next lever(s) off the menu in §4, build, validate the full ladder, measure in the COMM regime.*

---

## 0. State — where we are (read first)

The FESOM2 **C→C++/Kokkos** port (ocean + sea-ice) is device-resident + validated. **`master` @ tag `m5.21-coalescing-ghats-sss` (`4f9ea70`; NO git remote → local-only).** Everything below is **UNCOMMITTED working-tree** (the user commits only when asked — do NOT commit unless asked).

**Binary chain (all in `build-cuda/`, same node-mix, A/B-comparable):**
- `fesom_port_m522`  = pre-M5.22 (= the old m521 tip rebuilt) — the b-cluster A/B BEFORE.
- `fesom_port_m522b` = + M5.22 b-cluster flat lever (`fct_zal_b1v/b2/b3v`), **−3.10 % NG5 dist_16, bit-identical**.
- `fesom_port_m522c` = + M5.23 **L1 EVP two-field fused halo**, **−9.1 % at the NG5@16N per-rank size, bit-identical**. **← `build-cuda/fesom_port` == this (the live binary).**
- `build-serial/fesom_port` also carries L1 (used for the bit-id proof). ⚠️ `build-synccheck` + `build-omp` are STALE (m521-era) — rebuild only if you touch device-sync/host code. `build-cuda-synclog` = m522 (fine for PCIe attribution; L1 adds no syncs).

**What M5.22 + M5.23-L1 established (full detail: `docs/PROFILE_M522.md`, `docs/GPU_FIDELITY.md` §M5.22/§M5.22-comm, lessons L68/L69, memory [[project-m522-deep-profile]]):**

1. **The step budget was re-derived from scratch (the L56 "75 % PCIe" is DEAD).** It is **regime-dependent**:
   - **NG5 4 nodes (dist_16, ~1.34 s/step) = COMPUTE-bound** — Σ GPU-kernel ≈ 46 % of wall; PCIe RETIRED (~2–4 %; the big nod3D round-trippers are gone); halo comm 4.6 % / load-imbalance 9.7 %.
   - **NG5 16 nodes (dist_64, 0.56 s/step) = COMM-bound** — GPU-active drops to ~28 %; `MPI_Waitall` = 82.6 % of MPI time. **The optimization frontier SPLITS:** compute levers (Lever C / edge-coloring) help at low node-count, comm/scaling levers help at high. This IS the SCALING_M522 shrink (NG5 2.95×@4N → 2.38×@16N).
2. **The comm wall (at the NG5@16N per-rank size, from the dars-8N proxy):** halo `MPI_Waitall` ≈ **37 % of step** (≈63 % of it = load imbalance ≈23 % of step; ≈37 % = genuine comm ≈14 %) + **CG ≈ 12.5 %** (~67 iters/step, 2 Allreduce + 1 halo each). Both comm fractions GROW as you subdivide (adding nodes worsens the comm share — the throughput trap).
3. **The SYPD reality (the user's framing — this is the real goal):** SYPD = 0.493 / (s/step) at dt=180. NG5 GPU = 0.34 (4N) / 0.62 (8N) / **1.0 (16N)** / ~1.4 (32N) / ~1.8 (64N extrap.). Production needs **1–2 SYPD** (CPU does 1–2 on large allocations — "if we can't, we're screwed"). NOT a dead end: (a) **64 nodes is acceptable** (GPU-nodes-available vs CPU-nodes-available, not node-for-node); (b) **mixed precision ≈ ×2** is the single largest un-pulled lever; (c) the climate-safe comm levers stack toward the margin.
4. **TWO levers landed, both bit-identical:** M5.22 b-cluster (−3.10 %, compute) and M5.23-L1 EVP fused halo (−9.1 %, comm — the campaign's FIRST comm-regime win).

⚠️ **Repartitioning (Lever D) is the biggest single climate-safe lever (~23 % of step = the load imbalance), but the USER has SHELVED it for now** — keep it in memory, do NOT propose/pursue it this session unless the user re-opens it.

---

## 1. The mission

**Harvest more of the comm-regime menu (§4), measuring each in the COMM regime, toward closing the gap to 1–2 SYPD.** The discipline that has held all campaign: measure-first, one lever at a time, full validation ladder (§5), report the same-node A/B delta + bit-identity/gate. L1 built the shared machinery (the two-field fused-halo entry point) — several follow-ons are now cheap.

The big strategic question to keep in view: **the climate-safe comm levers alone probably reach only ~1.1–1.2 SYPD at 16N** (they target the ~14 % comm-proper + ~12 % CG, and repartitioning's ~23 % is shelved). **Mixed precision (~×2) is the lever that actually gets to 2 SYPD** — at some point this session or the next, the honest recommendation may be to scope a mixed-precision validation campaign rather than chase the last few % of comm. Flag this to the user when the cheap comm wins plateau.

---

## 2. The proven L1 machinery (build on it)

**The two-field fused-halo entry point is DONE and validated** — `src/fesom_halo_device.{cpp,hpp}`:
- `fesom_halo_exchange_device2(f0, f1, kind, nl, nc, p, base_off)` — co-packs two same-kind/same-stride fields into ONE message per neighbour (`[f0(stride) f1(stride)]` per halo node, stride 2×). The flat buffer-offset collapse + race-free unpack are unchanged → **byte-identical to two separate exchanges** (pure message-count cut, no new arithmetic, no atomics).
- `fesom_halo_field2(f0, f1, kind, nl, nc, p, base_off)` — the dispatch wrapper. CUDA → fused; Serial/OpenMP/`FESOM_HOST_HALO=1` → the EXACT two legacy host brackets (M5.1 Approach-B → the bit-identical oracle is untouched **by construction**).
- Applied at `src/fesom_ice_evp.cpp:716` (the EVP subcycle uice/vice, 240→120 msgs/step). **−9.1 % at dars 8N (99k 2D-pts/rank = the NG5@16N proxy), −7.4 % at farc 2N; bit-identical** (Serial all-fields + independent np.array_equal 81/81 + CUDA gate worst 4.5e-3).

**So L3 (adjacent same-kind PAIRS) is now a drop-in: just call `fesom_halo_field2` instead of two `fesom_halo_field`s** where two adjacent same-kind/same-stride exchanges have no compute between them.

---

## 3. The comm code-read — what's where (verified file:line, `docs/PROFILE_M522.md` §8.5)

- Halo routine `src/fesom_halo_device.cpp:204` `fesom_halo_exchange_device`: pack (`:244`) → `Kokkos::fence()` (`:251`) → Irecv/Isend loops (`:262`/`:268`) → `MPI_Waitall` (`:274`) → unpack (`:279`) → fence (`:286`). Requests **NOT persistent** (fresh Isend/Irecv every call; comm-lists `g_dev[5]` built-once `:159`). Two full-device fences/exchange.
- CG: `src/fesom_ssh.cpp` `fesom_ssh_solve_cg_kk` (`:680`) — **2 MPI_Allreduce/iter** (`:772` pAp + `:805` the already-fused rz/rr 2-elem) + 2 halos/iter (`:758` pp, `:787` rr), ~67–89 iters/step.
- EVP: `src/fesom_ice_evp.cpp` subcycle (`:627`), 120 subcycles; the uice/vice halo was `:713-714` → now the fused `fesom_halo_field2` at `:716` (L1, done).
- ⚠️ **`Kv`+`Av` and `hnode`+`helem` are DIFFERENT halo kinds** (NOD3D vs ELEM3D) → NOT batchable by `fesom_halo_field2` (a first code-read over-listed them). The real same-kind adjacent PAIRS for L3: FCT `fct_plus`+`fct_minus` (`src/fesom_tracer_adv.cpp:1798-1799`, ×2 tracers, NOD3D), PGF `pgf_x`+`pgf_y` (`src/fesom_step.cpp` ~319-320, ELEM3D), visc `u_b`+`v_b` (`src/fesom_momentum.cpp:1493-1494`, ELEM3D), GM `neutral_slope`+`slope_tapered` and `fer_K`+`Ki` (`src/fesom_step.cpp` ~278-279/287-288, NOD2D) — **verify each pair's kind/stride/adjacency before fusing** (don't trust this list blind; the FCT pair is the surest + highest-freq).

---

## 4. The ranked lever menu (pick 1–2; all climate-safe unless flagged)

| # | lever | where | attacks | est. effect | effort | climate |
|--:|---|---|---|---|---|---|
| **L3** | **adjacent same-kind PAIR fusion** — call `fesom_halo_field2` on the verified pairs (FCT plus/minus first, ×2 tracers) | `fesom_tracer_adv.cpp:1798-9`, then PGF/visc/GM pairs | comm msg-count | small per pair, stacks; FCT runs 2×/step | **LOW** (entry point exists) | **bit-identical** |
| **L2** | **persistent MPI requests** — `MPI_Send_init`/`Recv_init` once (keyed on kind/stride/buf) + `Startall`/`Waitall` per call | `fesom_halo_device.cpp:262-274` (+ host twin `fesom_halo.cpp`) | comm setup latency on ALL ~970 calls/step (broadest reach) | modest, broad | MED (re-init on buffer grow `:231`) | **bit-identical** |
| **L5** | **dead-exchange poison-test** — NaN-poison a halo's result, confirm zero model effect, drop it (L67 method) | candidates: `uv_rhsAB` `fesom_step.cpp:462`, pre-smoother `bvfreq` `:214` | whole exchanges + fences | unknown (may be 0) | LOW per candidate | **bit-identical IF dead** |
| **L4** | **CG 2→1 Allreduce** (Chronopoulos/Eijkhout reorder: form pAp from prev-iter vectors) | `fesom_ssh.cpp:772,805` | CG collective latency (134→67 Allreduce/step) | modest | MED–HIGH | **CUDA-gate-class, NOT Serial-bit-id** |
| **MP** | **mixed precision (FP32/mixed for selected fields+halos)** — the LARGEST lever (~×2 compute + comm bytes) | new track | the whole step | **~×2** → ~2 SYPD @32N | HIGH | **changes climate → own validation campaign** |
| **LC** | **Lever C prototype on ONE TDMA** (`impl_vert_diff_tracers`) — dedicated `View<double**,LayoutLeft>` scratch + transpose-in/out, ncu the coalescing win | `fesom_tracer_diff.cpp:438` | compute (the 4N regime) | de-risk only | MED | bit-identical |
| — | comm/compute overlap | — | comm | **SKIP** (4.6 % ceiling) | HIGH | — |
| — | on-node NVLink/NCCL | `fesom_halo_device.cpp:262` | on-node bw | med | HIGH | **vendor-lock → defer** |
| — | **repartition (Lever D)** | mesh partition | the ~23 % imbalance | up to 23 % | LOW | climate-id — **SHELVED by user; do not pursue** |

**Recommended order (the cheap, certain, stacking wins first):**
1. **L3 FCT pair** (then PGF/visc/GM pairs) — drop-in on the L1 entry point, bit-identical, measure stacked at dars-8N.
2. **L5 poison-test** the 2 candidates — may bank a free removal (it's pure measurement first).
3. **L2 persistent requests** — broadest reach, bit-identical, one focused change.
4. Then step back: if the comm levers have plateaued well short of 2 SYPD, **put the mixed-precision question to the user** (it's the only path to ×2; needs its own validation plan). Lever C is the parallel compute-frontier option if 4-node throughput is the target.

---

## 5. Validation ladder (every code change — the discipline that has held all campaign)
1. **Per-kernel `FESOM_KK_VERIFY=<key>` Serial `max|Δ|==0`** where applicable (FCT=`tradv`, EVP=`evp`, etc.). For a pure halo-batching change there's no kernel arithmetic change → the Serial whole-field diff (step 2) is the real proof.
2. **CORE2-Serial fresh-vs-saved ALL-FIELDS-BIT-IDENTICAL** — `jobs/job_core2_serial_m522c` is the template (clones trivially; runs build-serial CORE2 dist_8, `diff_snap.py` vs the preserved oracle **`serref_m522_saved`**). ⚠️ **a halo batching is bit-identical by construction (M5.1 Approach-B), but PROVE it** — and cross-check with an independent `np.array_equal` if diff_snap ever surprises you (see L69).
3. **CUDA fidelity gate** `scripts/gpu_fidelity_gate.sh` (CORE2 dist_8 ice-active) — PASS = the climate-close floor (~1e-3..1e-2; L1 was 4.5e-3). A halo batch adds zero atomics → the floor must be UNCHANGED.
4. **A/B in the COMM regime** (NOT the default 4-node compute regime — L69): clone `jobs/job_farc_l1_ab` / `jobs/job_dars_l1_ab` (BEFORE=`fesom_port_m522c`, AFTER=your new build). **dars 8N (99k 2D-pts/rank) is the faithful NG5@16N proxy; farc 2N (80k) brackets it.** Read `loop timing` + the `[halo-mpi-prof]` calls/step drop. (NG5 dist_16 = compute regime = will look ~flat — don't measure a comm lever there.)
5. **1-yr CORE2 CUDA climate** to close a milestone (corr=1.00000 vs m522 expected for bit-identical changes; the gate + Serial bit-id make this a formality for halo batching).

---

## 6. The per-rank-size proxy method (how to measure comm levers cheaply — [[feedback-per-rank-proxy]])
The comm/strong-scaling regime is set by **2D-vertices per rank**, NOT mesh size or node count. Reproduce NG5 dist_64 (16 nodes = 115.7k/rank) cheaply: **dars dist_32 (8 nodes) = 98.8k/rank** (closest, same CG-conditioning class — the best proxy); farc dist_4/8/16 (1/2/4 nodes) = 160/80/40k/rank (brackets it, ~free to schedule mid-week). ⚠️ faithful for HALO/geometry, NOT for CG iters (conditioning: farc 229 vs dars 67 vs NG5 89 — use dars for anything CG). Jobs: `job_farc_l1_ab`/`job_dars_l1_ab` (A/B two binaries), `job_farc_halo_split`/`job_dars_halo_split` (barrier-isolation imbalance-vs-comm), `job_farc_nsys` (MPI message profile). nod2D: farc 638387, dars 3160340, ng5 7402886.

---

## 7. Hard constraints (carry every session)
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; env.sh's 4.1.2 SEGFAULTs on device ptrs). ⚠️ `env_cuda.sh` PURGES `git` — git ops in a separate shell. CPU/Serial builds use `env.sh`.
- **Output → `/work/ab0995/a270088/port2/kokkos_gpu_runs/…`, NEVER `$HOME`** (60 GB quota; ~47/61 used). Big/NG5/CORE2 runs via SLURM, never login. NG5/dars jobs `rm` the ~50 GB `*.monthly.nc`.
- **⚠️⚠️ NEVER double-submit two jobs that `rm -rf`+write the SAME output dir** (L69): it gives a FALSE diff_snap DIVERGENCE (a deleted snapshot mid-compare). Before re-issuing a submit after any channel hiccup, `squeue` to check what already went. A collision only ever BREAKS a result, never fakes a PASS → on a surprise DIVERGENCE, first check all expected snapshots exist.
- **Same-day same-node perf baselines only** ([[feedback-perf-same-day-baseline]]); A/B both binaries back-to-back in ONE allocation. **Device/halo/sync changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); pi is insufficient (no ice). **Commit/push only when the user asks.** KPP is the default mix_scheme.
- ⚠️ `hnode_new` device-residency is LINFS-specific (zstar would need its host rail restored) — unchanged this session, but carry the caution for any new residency work.

## 8. Pointers
- **Docs:** `docs/PROFILE_M522.md` (THE deliverable — §1 budget 4N/16N, §2 kernel ranking, §3 PCIe, §4 MPI split, §5 candidate table, §6 b-cluster, §8 comm deep-dive + SYPD verdict + §8.5 code-read + §8.6 L1); `docs/GPU_FIDELITY.md` §M5.22 + §M5.22-comm; `docs/KOKKOS_PORTING_LESSONS.md` **L68** (budget/frontier-split), **L69** (L1 comm win + comm-regime-measurement + double-submit trap), L63–L67 (the flat-lever + PCIe history), L56/L62 (nsys-decisive / barrier-isolation).
- **Memory:** [[project-m522-deep-profile]] (the full M5.22+L1 record), [[feedback-per-rank-proxy]] (the proxy method), [[feedback-perf-same-day-baseline]], [[feedback-gpu-fidelity-gate]], [[reference-build-run]], [[reference-cuda-aware-mpi]].
- **Code (the L1 machinery to build on):** `src/fesom_halo_device.{cpp,hpp}` (`fesom_halo_exchange_device2` + `fesom_halo_field2`), `src/fesom_ice_evp.cpp:716` (the EVP call site), `src/fesom_ssh.cpp` (CG), `src/fesom_tracer_adv.cpp:1798` (FCT pair = the L3 first target).
- **Jobs:** `job_farc_l1_ab` / `job_dars_l1_ab` (comm-regime A/B, clone for the next lever), `job_core2_serial_m522c` (Serial bit-id, clone), `job_ng5_m522c_ab` (compute-regime A/B), `job_{farc,dars}_halo_split` (barrier-isolation), `job_farc_nsys` (message profile), `scripts/gpu_fidelity_gate.sh`, `scripts/diff_snap.py`, `scripts/m32_climate_compare.py`. Oracle: `serref_m522_saved`.
- **Binaries:** `build-cuda/fesom_port` == `fesom_port_m522c` (L1, live); `_m522b` (b-cluster) + `_m522` (pre-M5.22) kept as A/B baselines.

## 9. Bottom line
The budget is re-established and the frontier is split: **4-node = compute-bound, 16-node = comm-bound, and production throughput DEMANDS the many-node (comm-bound) regime** (1 SYPD @16N, the climate-safe comm levers stack toward ~1.1–1.2, mixed precision ≈ ×2 is the path to 2). Two bit-identical levers are landed (b-cluster −3.10 %, L1 EVP fused halo −9.1 % at scale). **This session: harvest the cheap stacking comm wins on the L1 entry point (L3 FCT/PGF/visc pairs → L5 poison-test → L2 persistent requests), each measured in the COMM regime (dars-8N proxy) with the full ladder; then put the mixed-precision-vs-comm-plateau question to the user.** Measure before building, A/B in the right regime, hold the Serial-bit-id + gate on every change, never double-submit to a shared dir, and don't claim a win until it's same-node A/B'd + gate PASS. Repartitioning stays shelved (user's call).
