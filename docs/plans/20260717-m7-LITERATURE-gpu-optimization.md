# M7 literature survey — what the field knows about making an ocean model like this faster on GPUs

**Date:** 2026-07-17 · **Author:** parallel research session (no code touched, no jobs submitted)
**Method:** deep-research workflow — 5 search agents → 25 primary sources fetched → 123 claims
extracted → top-25 adversarially verified (3 independent refutation votes each; 18 confirmed 3-0,
1 refuted, 6 unverified when the vote budget ran out) → inline synthesis against the campaign
ledger + Levante stack checks. Verification tags below: **[3-0]** = adversarially confirmed;
**[read]** = extracted from the paper by a fetch agent but the verification votes errored out;
**[my sizing]** = this report's arithmetic against the campaign's measured pools — pre-register
before believing any of it (campaign rule 8).

**Baseline this report targets (updated with the session-13/14 CGPOLY harvest):** h17+CGPOLY d3 =
**7.37× @4N (0.6213 s/step), 5.31× @16N (0.2314), SYPD@dt240 ≈ 2.76; settled CG iters ≈ 23**.
8× @4N needs another **−7.9 %**. Pools of record: halo-wait 97.3 ms/4N (14.8 %) / 124.8 ms/16N
(45 %), split comm/imbalance 58/42 @4N and 49/51 @16N; **rank-imbalance pool 36.1/53.0 ms**
(Σnlvls spread 22 %/51 %, ice polar fraction 0–90 %/rank); staging PCIe 16.5 ms both scales;
launch gap 16.8 + fence spin 16.1 ms @4N (h8-era stall budget — needs re-measure on h17);
kernels 74.6 % @4N. EVP = 120 exchanges/step (sEVP default; EVPWIDE K=8 certified opt-in,
−2.2 % @16N). Lesson 0.31 (session 14) applies to everything below: **a site's marginal cost
per deleted event decays as the site shrinks** — the literature's favourite solver toys are
worth much less against a 23-iteration CG than against the 128-iteration one they were
published against.

---

## 0. Executive ranking (details + citations in the numbered sections)

| # | lever | attacks | literature anchor | my sizing (honest range) | effort | risk class |
|---|---|---|---|---|---|---|
| 1 | **E.PART multi-constraint partition** (Σnlvls + optional ice weight) | imbalance 36/53 ms | FESOM2's own dual-weight METIS capability (Koldunov et al. 2019); Oceananigans' remaining scaling loss = the same imbalance | −10..25 ms @4N, −20..35 ms @16N | S-M (offline tooling, zero runtime code) | partition swap; byte-class per mesh |
| 2 | **Launch/fence hygiene + CUDA-graph capture** of the EVPWIDE window & CG iteration | 33 ms launch+fence @4N | −37 % total from single-queue async in a launch-bound atmos model (IEEE 2024); ICON-A nproma=1-block; 7.8 µs/sync floor (ExaMPI'24); Kokkos ≈ CUDA except small kernels −50 % (neXtSIM-DG) | −8..20 ms @4N, less @16N | M | no-arithmetic-change ⇒ byte-class if capture is faithful |
| 3 | **GPUDirect probe chain** (peermem? → gdr_copy env leg → HPC-X leg) = E.4 finish | 16.5 ms staging + per-event latency | Omega: GPU-aware halo = **4–6× less halo time** at scale; ICON-A: +10 % whole model on 4×A100 nodes; EAMxx: host-staging ≈5–7 % of total | 0 (if peermem absent) .. −25 ms @16N | S (env/module legs) | A.3 precedent: forcing transports can be catastrophic — A/B only |
| 4 | **SSH initial-guess extrapolation** (stabilized polynomial, communication-free) | CG pool post-CGPOLY (~23 iters) | Austin & Chow 2020: competitive with Fischer projection at half the storage, zero Allreduces; libParanumal ships it | −1..3 ms @16N (0.31-discounted) | S | solver-class (same tolerance) |
| 5 | **P-CSI / reduction-free Chebyshev outer iteration** (CGPOLY machinery reused) | the 2 Allreduces/iter that CGPIPE+CGPOLY still pay | P-CSI: **zero inner products/iter**, POP/CESM default, whole-model 1.7× at 16 875 cores | −2..4 ms @16N now; the real payoff is at ≥64N | M | solver-class; production-proven in CESM |
| 6 | **TDMA layout + register Thomas** (SLIM cell-AoSoA / Giles hybrid) | the 74.6 % kernel share (vertical implicit family) | layout = up to 16.75× on the solve; register Thomas-PCR 1.8× FP64; FP64 spill cliff at n≈384 ≫ our nl≤70; cuSPARSE loses 2.5–4.3× | −1..4 % @4N (needs ncu sizing first) | M-L | transpose-only variant stays byte-class |
| 7 | **aEVP / reduced-N EVP** (config, not code) | EVP 25.3/31.6 ms + its 120 exchanges | the user's own JAMES 2019 (mEVP N=100 ≈ sEVP 550; N=50 unrealistic) + Kimmritz 2016 aEVP (~3× faster convergence, N cut ~2.5× in box tests) | −4..8 ms both scales | S (mEVP knob exists; aEVP = small port) | PHYSICS-class ⇒ user's call + climate leg |
| 8 | **Device-initiated halo (NVSHMEM / NCCL-GIN) for the 2D-latency class** | ~292 latency-bound exch/step | GROMACS SC'25: 1.3× multi-node over IB in a sub-ms-iteration regime; NCCL 2.28 GIN ≈ NVSHMEM parity; **both ship in NVHPC 24.7 on Levante** | big-if-true (−10..20 ms @16N); gated on probe #3 | L | new comm path; certify like CGPIPE |

Already on the campaign's books and **validated by this sweep** (no re-rank needed): E.1 fuses
(LICOM's edge-strip packing = −10 % total precedent), E.2 interior/boundary overlap (SLIM: fully
hides the 3D-bandwidth class on unstructured-ocean A100s — and **cannot** hide the 2D-latency
class, which is exactly the split the E-ledger already made).

---

## 1. Solver / barotropic mode (the #1 site — now largely spent, and the literature agrees with 0.31)

**What the field measured about CG-class barotropic solvers at scale.** In POP, global reductions
alone were **74 % (PCG) / 68 % (ChronGear) of the entire barotropic-mode time at ~4 000 cores**,
and the solver share grew 5 %→50 % of the model from 470→16 875 cores [3-0]
(P-CSI paper, https://gmd.copernicus.org/articles/9/4209/2016/). LICOM3-HIP: barotr = 50 %→62 %
of runtime from 384→9 216 GPUs, all halo/copy latency [3-0]
(https://gmd.copernicus.org/articles/14/2781/2021/). MPAS-O names FESOM2's solver saturation
(500–250 verts/proc) explicitly and prescribes iteration-count reduction — the CGPOLY class [3-0]
(https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002238). **The campaign got here
first: CGPIPE + CGPOLY d3 already cut the site to ~23 iterations; every remaining solver lever
below is priced against that remnant, not the published baselines.**

- **P-CSI (preconditioned Classical Stiefel Iteration)** — Chebyshev-recurrence *outer* solver:
  **zero inner products per iteration** (convergence check every ~10 iters), parameters from
  Lanczos/power eigen-bounds. Production default in POP/CESM2; whole-ocean-model **1.7×
  (6.2→10.5 SYPD) at 16 875 cores** [3-0]. Fit here: the distributed power-iteration bounds and
  the ring-extent Chebyshev apply built for CGPOLY are exactly P-CSI's prerequisites — the lever
  is "drop the PCG wrapper, keep the machinery". At 23 iters × 2 Allreduces ≈ 46 reduces/step
  (~27 µs each at 4N, more at 16N) it buys only ~−2..4 ms @16N today [my sizing] — **pocket it
  for the ≥64N future**, where the POP curves say it becomes first-order.
- **Initial-guess extrapolation** (stabilized least-squares polynomial over previous solutions,
  M = O(m²) history for degree m): **entirely communication-free** (no inner products, unlike
  Fischer's projection) and competitive with projection at half the storage [3-0 ×2]
  (https://arxiv.org/abs/2009.10863). Composes with CGPOLY (it shortens the Krylov path, not the
  preconditioner). Cheapest lever in this report; also 0.31-discounted: 23→~15 iters ≈ −1..3 ms
  @16N [my sizing]. Guardrail: naive equispaced Lagrange extrapolation is exponentially
  ill-conditioned (Runge) — use the stabilized LS variant [read].
- **Deep pipelines p(l)-CG** (Ghysels/Cools/Cornelis/Vanroose, PETSc KSP PIPELCG): O(l) hiding of
  the reduction **only in strongly comm-bound regimes**; l ≥ 2 pays only when glred ≫ spmv; needs
  MPI async-progress threads whose overhead the authors themselves flag; square-root breakdown
  restarts [3-0 ×2] (https://arxiv.org/abs/1905.06850). With 23 iters and CSI available as the
  simpler zero-reduction endpoint: **do-not-chase**.
- **RAS (restricted additive Schwarz) preconditioning** held MPAS-O at **7 iterations flat to
  16 320 procs** (block-Jacobi drifted 7→10) [3-0]. That is the one preconditioner class in the
  sweep that beats CGPOLY-d3's ~23. But it needs overlapped local triangular/ILU solves —
  GPU-hostile vs our 3-SpMV Chebyshev — and at 23 iters the ceiling is small. Watch-list only.
- **Split-explicit barotropic instead of the elliptic solve** — the literature is genuinely
  split, and the resolution is instructive:
  - Oceananigans (JAMES 2025, FP64, 64×A100 = **exactly our scale**): split-explicit with
    **halos widened to the substep count (30–50)** → **one exchange per baroclinic step**,
    overlapped with vertical diffusion; barotropic <10 % of step vs "40–60 % typical"; **no
    global reduction anywhere in the model** [3-0-class, read]
    (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004465).
  - MPAS-O (CPU, 16 320 procs): semi-implicit **beat** split-explicit — barotropic 2.9× faster,
    whole model 1.89× [3-0].
  - Verdict for us: the Oceananigans trick is the same wide-halo family as EVPWIDE/CGPOLY rings —
    and CGPOLY+CSI reaches the same "no-reduction, few-exchange" endpoint **without changing the
    model's numerics**. A discretization rewrite is not warranted; steal the *shape* (wide halos
    trade bytes for latency — already our house style), not the scheme.
- **AMG (AmgX/hypre/Ginkgo)**: coarse-grid collectives are exactly the latency class that owns us;
  nothing in the sweep shows AMG winning a latency-bound 10–20 KB-message barotropic solve at
  ~23 iterations. Hypre BoomerAMG *does* matter in the implicit-VP ice context (§3). Do-not-chase
  for SSH.
- **ML-learned preconditioners/guesses**: nothing production-credible surfaced for ocean
  barotropic solves. The extrapolation lever (#4) is the defensible version of the same idea.

## 2. Sea ice (the #2 site)

- **The port already leads the published practice on communication**: CICE's 2024 EVP refactor
  still exchanges **every subcycle** (wide halos only mentioned as future work) — the workflow's
  verification panel *refuted* the claim that it offers a comm lever beyond EVPWIDE (1-2 vote)
  (https://gmd.copernicus.org/articles/17/6529/2024/). What it does offer:
  **active-point compaction** (2D→1D gather over ice-covered points): 5×/13×/35× on
  CPU generations, memory-bandwidth-bound diagnosis, **bit-for-bit verified refactor** [3-0 ×2] —
  the GPU translation is layout/bandwidth work on the ice kernels plus a reminder that ice-aware
  partition weights (§4) also fix the imbalance axis it exposes.
- **neXtSIM-DG GPU (GMD 2025)** — the only published GPU sea-ice dynamical core, and it is
  Kokkos: **Kokkos ≈ hand-CUDA asymptotically but ~50 % slower on small problems** (framework/
  launch overhead → feeds lever #2) [3-0]; mEVP stress kernels are memory-bound (70–80 % mem
  throughput) [read]; two measured warnings: **column-major flip SLOWED their unstructured FE
  kernels** and shared-memory staging gave nothing over constant cache [read]; on-the-fly
  recompute of geometry beat loading (+14 % CUDA / +35 % Kokkos) **until register pressure
  flipped it 8× the other way** [3-0] (https://gmd.copernicus.org/articles/18/3017/2025/) —
  the same cliff as our L97 REDISWEEP counter-example. Single-device only; no comm evidence.
- **Cutting N itself (config lever, the user's own literature):** sEVP needs ≥550 subcycles for
  LKF-quality ice at 4.5 km (ice = 80 % of model runtime there); **mEVP α=β=500, N=100
  reproduces it — 6× cheaper ice, whole model 25→40 SYPD on 1 728 cores**; N=50 was unrealistic
  ⇒ floor ≈100 at that resolution [3-0 ×3]
  (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018MS001485). aEVP (Kimmritz et al.
  2016) adapts α,β locally: ~3× faster convergence in the box benchmark, N cut ~2.5× while
  beating fixed-mEVP accuracy, lower bound (α,β) ≥ 50 [read — votes errored; but this is the
  user's home literature] (https://epic.awi.de/40430/). Our benchmark runs sEVP N=120: switching
  the *benchmark config* to mEVP/aEVP N≈100 cuts the EVP exchange+compute pool ~17 % before any
  code; composes multiplicatively with EVPWIDE (fewer subcycles × fewer exchanges per subcycle).
  **Physics-class — the user's call**, with the mEVP near-freezing flip floor (session-13) as
  known context.
- **Implicit VP (JFNK / stress-velocity Newton)**: svN converges where JFNK fails (≈7× fewer
  iterations; JFNK dies at 2 km) [3-0], needs hypre BoomerAMG per Newton step [read], **CPU-only,
  no GPU implementation exists**, and each Newton-Krylov step lands on exactly our latency pool.
  Do-not-build; re-read when someone publishes a GPU version.

## 3. Communication stack

- **GPUDirect / host-staging, quantified by peers:**
  - Omega v0.1.0 (GMD 2026): GPU-aware MPI (device buffers straight to MPI) = **"approximately a
    4–6× reduction in halo exchange time per time step compared to host-staged MPI at large node
    counts, where communication is latency dominated"** (Frontier/Cray MPICH)
    (https://gmd.copernicus.org/articles/19/3569/2026/).
  - ICON-A (GMD 2022): direct GPU-GPU vs host-staged = **+10 % whole model** on Piz Daint (then
    disabled there for MPICH crashes); **worked reliably on Juwels-Booster = 4×A100/node with
    OpenMPI 4.1.1** — our node architecture [3-0-class, read]
    (https://egusphere.copernicus.org/preprints/2022/egusphere-2022-152/egusphere-2022-152.pdf).
  - EAMxx/SCREAM (JAMES 2024): host-staging cost ≈ **10 % of dycore / 5–7 % of total** on Summit;
    GPU buffers flaky on Summit but **reliable everywhere on Perlmutter (4×A100+Slingshot) and
    Frontier** [read] (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004314).
  - LICOM3-HIP staged through host because GPU-aware MPI was unavailable and named it the
    unremovable bottleneck [3-0].
  - **Levante facts (checked this session):** system UCX 1.18.0 built `--with-gdrcopy
    --with-cuda=12.6`, libgdrapi present; whether `nvidia-peermem` is loaded on gpu compute
    nodes is THE open question (login has no GPU). The campaign's A.3 verdict "GPUDirect RDMA
    unavailable" dates from the get_zcopy/put_zcopy +35 % catastrophe — which was a *rendezvous
    forcing* experiment, not an eager/gdr_copy probe. E.4's remaining legs are cheap and now
    literature-backed. Probes in §7.
- **Device-initiated (the frontier lever):**
  - GROMACS NVSHMEM halo redesign (SC'25 wksp): kernel-initiated exchange, pack fused with
    communication — **up to 2× multi-node NVLink, 1.3× multi-node NVLink+IB** vs CPU-MPI, in a
    **sub-millisecond-iteration latency-bound regime** (our regime) [read]
    (https://arxiv.org/abs/2509.21527).
  - NCCL 2.28 "GIN" device API: GPU-initiated networking at **production parity with NVSHMEM**
    (16.7 vs 16.0 µs small-message RT; MoE benchmarks within 1–2 % at 16–64 GPUs); the fast GDAKI
    backend needs ConnectX-6 **Dx**/BlueField + DOCA GPUNetIO (Levante's HDR ConnectX-6 likely
    not) but the **Proxy backend runs on any RDMA at only +1.3 µs** [read]
    (https://arxiv.org/abs/2511.15076). **Both NVSHMEM and NCCL ship in NVHPC 24.7's comm_libs on
    Levante today** — no new system software to try them.
  - Sync-cost floor that device-initiation removes: **cudaStreamSynchronize = 7.8 µs constant;
    71.6–78.9 % of total time for small kernels**; block-aggregated readiness signalling is 271×
    cheaper than per-thread [read] (ExaMPI'24,
    https://www.queensu.ca/academia/afsahi/pprl/papers/ExaMPI-2024.pdf). SLIM independently
    measured **~7.5 µs per launch/sync/MPI op** as its constant multi-GPU latency floor.
- **MPI-4 partitioned / GPU-triggered MPI: not yet.** Stock UCX has **no GPU-initiated support at
  all** (the ExaMPI prototype had to modify UCX); best case measured is 1.30× on a 2-node Jacobi
  halo; OpenMPI on Levante is 4.1.x = MPI-3.1, no 5.x module exists [read ×2]. The triggering-API
  survey (https://arxiv.org/abs/2406.05594) reports **zero quantitative results** and flags
  semantic lock-in per vendor [read]. Do-not-chase until the stack moves.
- **Message packing/coalescing precedent** (E.1 support): LICOM3-HIP's edge-strip halo packing =
  **−10 % total model / −30 % barotr** [3-0-class, read]. Our E.1 ledger ceiling (−7-8/−12-13 ms)
  is the honest unstructured version of the same lever.
- **Overlap precedent** (E.2 support): SLIM's two-stream interior/boundary split **fully hides
  the 3D baroclinic exchange cost and cannot hide the 2D barotropic latency** — strong-scaling
  then follows Amdahl on the 2D mode [read]
  (https://arxiv.org/pdf/2605.16082). Bonus: splitting let interior kernels drop boundary
  conditionals (less divergence, fewer registers). This is precisely the E-ledger's 2D-latency /
  3D-bandwidth split; it says E.2 pays on the KPP/tracer/GM class and NOT on CG/EVP.
- **Lossless FP64 halo compression:** no ocean/weather result surfaced; at 10–20 KB per message
  the pool is latency- not bandwidth-bound, so compression adds kernel time + latency to the
  wrong term. Do-not-chase.

## 4. Partitioning / load balance (the biggest single literature-backed pool)

- **FESOM2's own partitioner already documents the fix**: Koldunov et al. 2019 describes METIS
  partitioning with a **dual-weighted criterion balancing 2D and 3D vertex counts** — i.e. the
  Σnlvls weight E.PART needs is native FESOM2 capability, not new tooling [3-0-class, read]
  (https://gmd.copernicus.org/articles/12/3991/2019/). The same paper: **hierarchical
  (topology-aware) METIS wrapper exists but bought little on good interconnects** — de-prioritize
  that half; and it documents **sea-ice-induced imbalance** (ranks idle in ice halo waits) —
  the second constraint axis. First action: check which weights produced the current `dist_*`
  directories; the measured 22 %/51 % Σnlvls spread says 2D-only.
- **Oceananigans hit the identical wall**: with all exchanges overlapped and zero reductions,
  their remaining strong-scaling loss (~70 % efficiency at 16×) is **"poor load balancing …
  some GPUs having more active cells"** [read]. The imbalance pool is what's left when comm is
  beaten — at 16N ours is already the larger half (53 ms).
- Multi-constraint METIS (2D verts + Σnlvls + ice mask) is standard practice; nothing exotic
  needed. Omega does METIS at startup (no 3D weights mentioned — V0 has no vertical dynamics yet).
- [my sizing] Balancing Σnlvls to the few-% level attacks most of 36/53 ms: **−10..25 ms @4N,
  −20..35 ms @16N**; the ice-fraction weight is the second-order term to A/B separately.

## 5. Kernel / memory level (the 74.6 % @4N)

- **Vertical implicit (TDMA family — the known frontier):**
  - Giles et al. (TOMS): batched tridiagonal is bandwidth-bound (0.225 FLOP/byte FP64); **layout
    dominates — solving along the contiguous dimension is 16.75× slower** than batch-coalesced;
    register Thomas-PCR hybrid = **1.8× (FP64) over naive coalesced Thomas**, spill cliff at
    n≈384 FP64 — our nl ≤ 70 is deep inside the safe zone; **cuSPARSE gtsvStridedBatch loses to
    hand-tuned by 2.5–4.3×** (don't reach for the vendor lib) [3-0 + read ×3]
    (https://people.maths.ox.ac.uk/gilesm/files/toms_16b.pdf).
  - SLIM's working pattern on A100: **AoSoA "cell" batches of 128 columns padded to the deepest**,
    one thread per column, banded solve from a 36-entry local buffer with RHS in registers —
    "eliminates register spilling", 80 % of peak BW on memory-bound kernels, ~30 % of peak
    averaged over the whole step [read] (https://arxiv.org/pdf/2605.16082).
  - SCREAM/EKAT nuance [read]: their famous "packs" are a **CPU vectorization lever, not a GPU
    one** — on GPU their wins are TeamPolicy hierarchy + shared workspaces (fewer allocations,
    fewer launches). Matches the ICON/SCREAM eval memory (Pack parked). The GPU translation of
    "vector over columns" for us is the SLIM/Giles layout, not packs.
  - Composition with the campaign: the bit-id levers already identified (scratch shrink,
    level-major transpose) are the first rung; the Giles register-hybrid is the non-bit-id second
    rung. ncu the four TDMA-family kernels first (campaign rule: measure, then code).
- **Mesh reordering / indirect loads:** OP2 study — METIS-block reordering raised in-block reuse
  2.0→3.6 and cut global reads 35 % for a **+19 % kernel** gain, but **+3.3 % whole-app** (P100);
  and **atomics beat coloring/shared-mem staging on V100 (2×, 76 % of peak BW)** — the trend that
  buries the coloring literature on Ampere [3-0-class, read ×3]
  (https://warwick.ac.uk/fac/sci/dcs/people/gihan_mudalige/jpdc2019_acceptedpreprint.pdf).
  Our scatter-adds already use atomics — correct choice, keep. A cheap ncu look at L2 hit rates
  on the top gather kernels decides whether NG5's native ordering leaves anything; expect
  single-digit whole-model at best. fesom2-accelerate already measured FESOM2's FCT kernels
  "approach peak memory bandwidth" on Ampere [read] — kernel headroom is bounded.
- **Fusion has a measured ceiling**: Oceananigans' fused tendency megakernels hit 255 registers →
  **11 % occupancy → spills → 2.6 of 6.18 TFLOP/s** [read]. Same physics as L97. Fuse for launch
  count (lever #2), not for its own sake.
- SELL-C-σ / SpMV formats: bounded by the SSH SpMV's small share post-CGPOLY. Skip.

## 6. Runtime / scheduling

- **Async execution & graphs:** the overhead-sensitive atmos-model study (A100): default
  synchronous launches made the GPU **slower than an A64FX CPU** in strong scaling;
  **single-queue async −37 % total runtime; kernel fusion another ~10 %; CUDA Graphs recommended
  once a small kernel set repeats frequently** [read ×3]
  (https://ieeexplore.ieee.org/iel8/10820557/10820558/10820817.pdf). ICON-A: land physics was
  launch-bound ("launch time comparable to compute"), fixed by nproma = whole-subdomain single
  block + loop fusion + ASYNC everywhere [read]. neXtSIM-DG's Kokkos-vs-CUDA −50 % at small sizes
  is the same overhead class [3-0].
- **Fit to us:** the h8-era stall budget had launch gap 16.8 + fence spin 16.1 ms @4N (~4.5 %,
  and a larger *fraction* now that the step is 0.62 s — re-measure first). The natural graph
  bodies: the **EVPWIDE K=8 window** (8 subcycles of fixed kernels, no MPI inside — a gift of a
  capture region), the CGPOLY iteration body, and the KPP smoother sweeps. Kokkos 4.4.01 has
  `Kokkos::Experimental::Graph` on CUDA; raw stream-capture around the fixed sequences is the
  fallback. Fence audit rides along (the census FENCE column exists since L92).
- **MPS / multi-rank-per-GPU:** no ocean/weather evidence surfaced; more ranks = more exchanges
  on our latency-bound profile. Skip.

## 7. Cheap machine probes this report pre-registers (all S-effort, no code)

1. **peermem probe (gates lever #3 and #8):** on a gpu-partition node:
   `lsmod | grep -iE 'nvidia_peermem|nv_peer_mem'` + `ucx_info -d | grep -iE 'gdr|cuda'`.
2. **gdr_copy / eager-path env leg** (ab_env, both scales, h17+CGPOLY BIN): keep the rndv
   pipeline (A.3 says it's load-bearing) but let small eager messages use gdr_copy:
   `UCX_TLS=rc_x,cuda_copy,gdr_copy` vs baseline; watch the 10–20 KB class per-event latency.
3. **HPC-X leg**: NVHPC 24.7 `comm_libs/hpcx` as the MPI (same binary, module swap) — different
   UCX build, different defaults; ICON-A's +10 % came from exactly this class of swap.
4. **NCCL/NVSHMEM microbench** from `comm_libs/12.5/{nccl,nvshmem}`: ring latency at 16 KB × 64
   ranks vs MPI_Isend baseline — prices lever #8 before any port work.
5. **Partition-weight audit:** regenerate one CORE2 + one NG5 partition with the dual 2D/3D
   weights (native FESOM2 tooling per Koldunov 2019) and diff Σnlvls spread — prices E.PART
   before touching production meshes.
6. **Launch/fence census refresh** on h17+CGPOLY at both scales (`--min-gap-ms 0.1` setting of
   record): sizes lever #2 honestly before any graph work.

## 8. Whole-model calibration (where 7.37×/5.31× sits)

| model | hardware | FP | reported | source |
|---|---|---|---|---|
| **this port** | 4N/16N × 4×A100, 128-core CPU nodes | 64 | **7.37× @4N, 5.31× @16N node-for-node; SYPD 2.76 (NG5, dt240)** | campaign ledger |
| Oceananigans | 64×A100 (16 nodes) | 64 | 10 SYPD @1/12°·100L (ocean-only, **no sea ice**); 75 SYPD @1/4° on 16 GPUs | JAMES 2025 |
| SLIM (DG) | A100; LUMI MI250X | 64 | 1 A100 = 515× 1 CPU core; 4×A100 node ≈ 50× a 128-core node; latency floor 6 ms/iter multi-GPU | arXiv 2605.16082 |
| ICON-A | Juwels-Booster 4×A100 | 64 | node speedup ~6.3× (P100-era); GPU-GPU MPI +10 %; strong-scaling wall at 2–4× nodes | GMD 2022 |
| SCREAM/EAMxx | Perlmutter 4×A100 | 64* | 6× node-for-node (physics 12×); "comm volume/frequency ≫ per-GPU work volume" | JAMES 2024 |
| LICOM3-HIP | 384–26 200 AMD GPUs | 64 | 42× vs same-# CPU cores @384; barotr 50→62 % of time; 2.72 SYPD @1/20° | GMD 2021 |
| LICOM3-Kokkos | V100 | 64 | tuned **Kokkos 1.9× their own raw CUDA** | FGCS 2024 |
| Omega-V0 | Perlmutter A100 | 64 | **6.1× node-for-node**; GPU-aware halo = 4–6× less halo time | GMD 2026 |
| NEMO (PSyclone) | V100/A100 | 64 | 1 V100 ≈ 3.6× an Intel socket @1°; ~90 A100 ≈ 270 sockets @1/12° | Copernicus SoP 2025 |

The port's node-for-node ratio is at or above every published FP64 peer on comparable hardware,
including the two other Kokkos ocean efforts (LICOM3-Kokkos, Omega-V0). The field's shared
residual frontier is exactly ours: **2D-mode latency + load imbalance** — and the two shops that
attacked imbalance last (Oceananigans, SCREAM) both concluded it dominates once comm is hidden.

## 9. Do-not-chase list (each with the literature reason)

- Split-explicit barotropic rewrite (MPAS-O: semi-implicit 1.89× better at scale; CSI+guess
  reaches reduction-free within current numerics).
- AMG for SSH (coarse-grid collectives on a 10–20 KB latency floor; 23-iter remnant too small).
- Deep pipelines l≥2 / MPI async-progress (gains only when glred ≫ spmv; thread-safety overhead;
  CSI is the simpler endpoint).
- MPI-4 partitioned / GPU-triggered MPI on today's stack (stock UCX: zero GPU-initiated support;
  max 1.30× in a prototype; OMPI 4.1 = MPI-3.1; MPI-5 not settled).
- FP64 halo compression (latency-bound 10–20 KB messages; no ocean precedent).
- Implicit VP sea ice on GPU (doesn't exist yet; per-Newton Krylov lands on the same latency pool).
- Coloring/shared-memory scatter instead of atomics (V100+ atomics won 2×; A100 more so).
- Column-major flip "for coalescing" without profiling (neXtSIM-DG measured a slowdown).
- cuSPARSE batched tridiagonal (loses 2.5–4.3× to hand-tuned).
- Kokkos 5.0 chase (prior in-house eval stands); SCREAM packs on GPU (CPU lever); mixed precision
  (BANNED, and LICOM proves FP64-only viability at 26 200 GPUs).

## 10. Verification caveats

18 claims survived 3-vote adversarial verification (3-0); 1 was refuted and is reported as such;
6 (mostly aEVP quantitative + one GPU-MPI mechanism claim) lost their votes to the session budget
and are tagged [read] — they come from fetch-agent extraction of the primary PDFs, several from
the user's own group's papers, but were not independently re-derived. The synthesis step of the
workflow also hit the budget; this document is the manual synthesis. Sizing arithmetic is mine
and labelled; per campaign rule 8, nothing here is a pre-registration — each lever needs its own
measured pre-reg before code.

## Sources (fetched primaries)

P-CSI/POP: https://gmd.copernicus.org/articles/9/4209/2016/ · MPAS-O semi-implicit:
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002238 · p(l)-CG:
https://arxiv.org/abs/1905.06850 · initial guesses: https://arxiv.org/abs/2009.10863 ·
neXtSIM-DG GPU: https://gmd.copernicus.org/articles/18/3017/2025/ · CICE EVP refactor:
https://gmd.copernicus.org/articles/17/6529/2024/ · Fast EVP (mEVP/aEVP in FESOM2):
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018MS001485 · mEVP convergence:
https://www.sciencedirect.com/science/article/abs/pii/S0021999115003083 · aEVP:
https://epic.awi.de/40430/ · implicit-VP svN: https://arxiv.org/abs/2204.10822 ·
GPU triggering survey: https://arxiv.org/abs/2406.05594 · GROMACS NVSHMEM halo:
https://arxiv.org/abs/2509.21527 · NCCL GIN: https://arxiv.org/abs/2511.15076 ·
GPU-initiated MPI partitioned: https://www.queensu.ca/academia/afsahi/pprl/papers/ExaMPI-2024.pdf ·
ICON-A GPU: https://egusphere.copernicus.org/preprints/2022/egusphere-2022-152/egusphere-2022-152.pdf ·
async/fusion atmos model: https://ieeexplore.ieee.org/iel8/10820557/10820558/10820817.pdf ·
FESOM2 scalability: https://gmd.copernicus.org/articles/12/3991/2019/ · SLIM multi-GPU:
https://arxiv.org/pdf/2605.16082 · SCREAM/EAMxx: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004314 ·
OP2 locality: https://warwick.ac.uk/fac/sci/dcs/people/gihan_mudalige/jpdc2019_acceptedpreprint.pdf ·
Giles batched tridiagonal: https://people.maths.ox.ac.uk/gilesm/files/toms_16b.pdf ·
Oceananigans dycore: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004465 ·
LICOM3-HIP: https://gmd.copernicus.org/articles/14/2781/2021/ · LICOM3-Kokkos:
https://www.sciencedirect.com/science/article/abs/pii/S0167739X24003285 · fesom2-accelerate:
https://ui.adsabs.harvard.edu/abs/2021EGUGA..2311551V/abstract · Omega v0.1.0:
https://gmd.copernicus.org/articles/19/3569/2026/ · GPU ocean forecasting review:
https://sp.copernicus.org/articles/5-opsr/23/2025/
