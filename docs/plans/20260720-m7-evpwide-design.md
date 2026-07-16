# E.EVP0/E.EVP1 — wide-halo EVP (`FESOM_SPEED_EVPWIDE=K`) audit + design + pre-registration

*Session 12 (Fable, 2026-07-16). The E.EVP0 deliverable per the session-12 PROMPT §2.2: the EVP
data-flow table, the K-ring state contract, machinery inventory, implementation plan, gate
ladder, and the pre-registration for E.EVP1. Sources read: `fesom_ice_evp.cpp` (whole file),
`fesom_partit.h/.cpp`, `fesom_halo.h`, `fesom_ssh.cpp` (cgpipe_build + cgpipe_exchange_rr),
`fesom_mesh.cpp` (scatter_mesh, area/coriolis build), `fesom_ice.cpp` (call site, allocs),
`fesom_ice_maevp.cpp:280-350`, `fesom_speed.hpp`.*

---

## 0. VERDICT (the headline upgrade over the session-11 assumption)

**The wide-halo EVP can be made FORCE_SERIAL BYTE-PROVABLE — the same certification class as
CGPIPE — not merely climate-close.** The session-11 PROMPT §2.3 assumed "NO byte proof
available: the element→node scatter is atomic, redundant ghost compute cannot be bitwise".
That is true only for the naive form. Three ingredients repair it exactly:

1. **Rings R = 2K−1** (not K) — the classic communication-avoiding redundancy for carried
   two-level state (sigma is element state carried across subcycles AND steps);
2. **Owner-order gather** for the ghost `u_rhs/v_rhs` sums (the owner's Serial scatter order =
   its local element index order; ship that order at setup, replay it as a per-ghost-node
   gather) + **ship owner bytes** for every order-sensitive or non-local static
   (`area(:,0)`, per-step `rhs_a/rhs_m`);
3. **Freeze ring R between refreshes** (never velocity-update it locally; it lacks a complete
   incident-element set).

Then every ghost update replays the owner's arithmetic bytewise (same TU ⇒ same FMA
contraction — the cgpipe argument), owned loops are UNTOUCHED (same code, same order), and by
induction the whole trajectory is bit-identical to per-subcycle exchange **on Serial**. On CUDA
the owner's own atomics make byte-identity impossible for ANY path (today's baseline included)
— CUDA stays climate-close class, certified by the CUDA-vs-Serial fidelity gate as usual.

Cost of exactness vs the approximate K-ring variant: rings 2K−1 instead of K (ghost zone
~2×), which is still only ~4 % of rank rows at K=4 @4N (~8 % @16N), executed INSIDE the
existing 3 per-subcycle kernel launches (extended ranges — zero extra launches). Message
COUNT (the thing that pays, 2D-latency class) is identical in both variants: 120 → 120/K + 1.
**Recommendation: build the exact variant.** It also buys an exact in-vivo selfcheck
(refresh-vs-local must be 0.000e+00 on Serial), which converts every logic bug into a loud
failure instead of a silent climate drift.

---

## 1. THE EVP DATA-FLOW TABLE (per ocean step; std EVP, `whichEVP==0`)

Setup, once per step (before the subcycle loop; all Fortran-mirrored in
`fesom_ice_evp_dynamics_kk`):

| phase | domain | reads | writes | class |
|---|---|---|---|---|
| Step 1 zero | nodes [0, N=myDim+eDim) | — | inv_am, inv_m, rhs_a, rhs_m | map |
| Step 2 mass | OWNED nodes | a_ice, m_ice, m_snow, area(n,0), ulev_n | inv_am, inv_m | pure map |
| Step 3 strength+grad | OWNED elems (maxring ≤ 1) | m_ice, a_ice, elevation @3 verts; elem geometry | istr(el); **scatter** rhs_a/rhs_m @3 verts | istr = pure map; rhs = order-sensitive scatter |
| Step 4 divide | OWNED nodes | area(n,0), ulev_n | rhs_a, rhs_m | pure map |

Subcycle ×120 (`evp_rheol_steps=120`, `fesom_ice.cpp:91`; kernels M5.12b-fused 7→3):

| kernel | domain | halo READS | writes | carried state |
|---|---|---|---|---|
| K1 zero | OWNED nodes | — | u_rhs, v_rhs | — |
| K2 stress+scatter | OWNED elems | **u_ice, v_ice @3 verts (ring ≤ 1)**; istr, σ11/12/22 (RMW), gs, mf, ea, ulev | eps* (diag), **σ* (RMW)**, scatter u_rhs/v_rhs @3 verts | **σ11/12/22 per element, carried across subcycles AND steps** |
| K3 node update | ALL nodes [0,N) | u_rhs (own sum), inv_am, rhs_a/m, inv_m, a_ice, u_w, v_w, sax, say, cor, area, ulev_n, coastal mask | u_old/v_old (unused downstream), **u_ice/v_ice OWNED** + coastal zeros | u_ice/v_ice |
| halo | — | — | **u_ice/v_ice ring 1 ← owners** (`fesom_halo_field2`, 1 fused msg/partner) | — |

Per-step inputs, all FINAL at `fesom_ice_evp_dynamics_kk` entry (`fesom_ice.cpp:663-700`:
ocean2ice + its srfoce_u/v halo, bulk stresses, prev-step a/m/ms all precede it):
`a_ice, m_ice, m_snow, srfoce_u(u_w), srfoce_v(v_w), srfoce_ssh(elevation),
stress_atmice_x/y(sax/say)` — 8 node fields + the RMW state `(uice, vice, σ11/12/22)`.
`elevation` feeds ONLY the Step-3 rhs scatter ⇒ if owner-final `rhs_a/rhs_m` are shipped,
elevation is NOT needed on rings at all.

Statics: `gradient_sca[6], elem_area, metric_factor, ulevels` (per element — pure functions of
vertex coords ⇒ locally recomputable byte-equal from global data at scatter time);
`coriolis_node` (pure per-node, `fesom_mesh.cpp:884`), `ulevels_nod2D` (global data);
`area(n,0)` (**cluster sum over incident elements, `fesom_mesh.cpp:822` — ORDER-SENSITIVE ⇒
must be SHIPPED from owner**); coastal mask (node property from GLOBAL edge criterion
`myList_edge2D(ed) > edge2D_in`, computable from global edges at scatter; ASSERT owner-parity
on owned nodes at build).

## 2. THE K-RING STATE CONTRACT (the theorem E.EVP1 is built on)

Definitions: ring 0 = owned nodes; ring r = nodes at element-adjacency distance r from the
owned set (an element's 3 nodes span ≤ 2 consecutive rings); element maxring m = max ring of
its 3 vertices. Today's `eDim` halo = ring 1; `myDim_elem2D` = elements maxring ≤ 1 (every
element incident to an owned node is local — the FESOM assembly invariant).

**Contract.** To advance owned `u_ice/v_ice` K subcycles between exchanges with owned results
byte-identical (Serial) to per-subcycle exchange, hold and operate:

- node rings 1..R with **R = 2K−1**: refresh `(uice, vice)` on ALL R rings every K-th
  subcycle; velocity-update rings 1..R−1 locally every subcycle; **freeze ring R** between
  refreshes;
- ALL local elements (maxring ≤ R): sigma-update every subcycle (K2 extended range);
- ghost `u_rhs/v_rhs`: per-ghost-node **gather in the owner's element order** (K3 extended
  range), never a scatter;
- per step, after owned Step 4, ONE extended exchange shipping owner bytes of
  `uice, vice, a_ice, m_ice, m_snow, u_w, v_w, sax, say, rhs_a, rhs_m` (11 fields) on rings
  1..R; then recompute ghost `inv_am/inv_m` (Step-2 map) and ghost `istr` (Step-3 map) locally;
- statics at setup: local recompute of pure per-element geometry + coriolis + ulev + coastal
  (from global mesh data at scatter time), owner-shipped `area(:,0)`.

**Why R = 2K−1 (proof sketch + tightness).** Validity frontier: after j subcycles since a
refresh, ring ρ is byte-clean iff ρ ≤ R−j (each subcycle consumes one ring: a ring-ρ update
needs elements maxring ≤ ρ+1, which need velocities on rings ≤ ρ+1 from the previous
subcycle). Cross-window induction: element maxring m stays PERMANENTLY sigma-clean iff it is
cleanly updated at every subcycle k=1..K of a window, i.e. m ≤ R−K+1. The owned chain at
subcycle K unrolls owned ← ring-1@K−1 ← maxring-2 elems ← ring-2@K−2 ← … ← ring-(K−1)@1 ←
**maxring-K elements ← ring-K@0 (post-refresh only)**; the binding constraint is that
maxring-K elements be permanently clean: K ≤ R−K+1 ⇔ **R ≥ 2K−1**. Tight: with R = 2K−2 the
maxring-K elements go stale on each window's last subcycle and pollute ring-(K−1) reads.
Pollution from beyond (elements maxring > K are never permanently clean) advances exactly one
ring per subcycle and reaches ring 1 only AT subcycle K — after the owned K2 reads of that
subcycle (which consume post-(K−1) values) ⇒ owned reads are never touched. Ring R frozen is
safe for the same reason (its staleness enters the same frontier).

Sanity anchors: K=1 ⇒ R=1 = today's scheme exactly (null rung, byte-provable through the new
code path). The per-window refresh must land on the LAST subcycle ⇒ **require 120 % K == 0**
(abort otherwise); sweep set {2,4,8} all divide 120.

**Why not the cheaper approximate variant (R=K, stale outer sigma):** outer-ghost elements
(m ≥ 2) receive only K+1−m updates per K subcycles ⇒ their sigma falls behind WITHOUT BOUND
(det1 ≈ 0.986/update damping makes it bounded in magnitude but it is a genuinely different
trajectory, ~30·(K−1) subcycles stale after one step at K=4); the error enters owned every
window through rings 1..K−1. It saves ~half the ghost zone (compute we do not care about, it
is ~2-8 % inside existing launches) and loses the byte proof, the exact selfcheck, and the
clean certification story. Rejected. Likewise rejected: exchanging sigma every window
("variant X") — same message count, MORE bytes (3 element fields ≈ 6·ring vs 2·(K−1)·ring for
the extra node rings), plus a whole second (element-owner) exchange machinery.

## 3. MACHINERY INVENTORY (what exists, what is missing)

EXISTS and is reused:
- **Global mesh on every rank during `scatter_mesh`** (`fesom_mesh.cpp:1088-1281`): global
  `elem_nodes`, coords, edges, nlevels are broadcast, sliced, then FREED (:1259-1281). Ring
  BFS + extended-zone geometry can therefore be built LOCALLY at scatter time — no iterative
  MPI discovery needed. The "4740 halo-element vertex refs unmappable" (-1 refs, :1194-1219)
  is exactly the ring-2 vertex problem the extended node set removes (for the EVPWIDE side
  structure; the mesh's own arrays stay untouched).
- **gid→owner**: NEVER `partit->part` ranges (gids are NOT contiguous per rank — the
  session-11 cgpipe bug, `fesom_ssh.cpp:703-707`). For EVPWIDE: build the TRUE global owner
  vector once at scatter (int[nod2D], set own gids to mype, `MPI_Allreduce(MAX)`) — trivially
  correct, sized like the already-broadcast global arrays.
- **The cgpipe pattern** (`cgpipe_build` + `cgpipe_exchange_rr`, `fesom_ssh.cpp:689-1000`):
  want-lists per owner, `MPI_Alltoall` count handshake (new diagonal partners possible),
  gid→local translation with appended extended slots, flat per-partner sidx/ridx device lists,
  pack → fence → Irecv/Isend (one tag) → Waitall → unpack, M5.17 prof hooks
  (`fesom_halo_prof_barrier/waitall/bytes`), build-summary print. EVPWIDE clones this with
  nf-field packing (nf=2 per window, nf=11 per step; same lists, stride nf).
- **`fesom_speed_int("EVPWIDE", 0, &cache)`** (`fesom_speed.hpp:161`) — the value-knob helper;
  opt-in by construction (master never implies a value knob). ⚠️ It does NOT announce ⇒ add an
  explicit rank-0 `[fesom_speed] FESOM_SPEED_EVPWIDE = K (R=2K-1 rings, msgs/step 120->120/K+1)`
  announce at first resolve — the L80 dead-knob rule, and the gpu-gate grep needs the line.
- **Serial-stays-legacy guard** is inside `fesom_speed_int` (needs `FESOM_SPEED_FORCE_SERIAL=1`
  on Serial) — exactly what the FORCE_SERIAL byte proof rides (cgpipe precedent).
- `job_m7_gate_serial` (KNOBS ⇒ doubles as FORCE_SERIAL byte proof), `job_m7_gpu_gate`,
  `job_m7_ab_env` (LEG1..4 KNOB legs ⇒ the 16N K-sweep is ONE same-alloc job: OFF/2/4/8).

MISSING (to build):
- Scatter-time hook (knob-gated): ring BFS 1..R, extended node gid/owner/ring lists, extended
  element list (gid, 3 vertex gids→extended-local, gradient_sca/elem_area/metric_factor/ulevels
  recomputed from global coords), global-edge coastal mask, ulev_n/coriolis for ghost slots.
  Stored in a small side struct owned by the EVPWIDE module.
- Lazy first-EVP-call build (collective): the cgpipe-style handshake shipping (a) owner-order
  incident-element gid lists per wanted ghost node (the gather adjacency + its order), (b)
  `area(:,0)` owner bytes for ghost slots; then runtime send/recv lists + device pushes.
- The nf-field extended exchange (module-private buffers, new tag, e.g. 2200).
- Ghost kernel ranges fused into K1/K2/K3 + ghost Step-2/Step-3 maps (same TU =
  `fesom_ice_evp.cpp` — the FMA-contraction guarantee).
- Extended tails on 11 ice fields (`alloc` size + nring_ext at ice init when knob ON — all
  other code reads [0,N) and is unaffected; `fesom_ice.cpp:175-194`).
- `FESOM_EVPWIDE_SELFCHECK=1`: at each refresh, before unpack, max|recv − local| over rings
  ≤ K−1; MUST print 0.000e+00 on Serial; report magnitude on CUDA (expect ≲1e-12).
- Guards: abort if `whichEVP != 0` requests EVPWIDE (announce no-op + resolve OFF under mEVP —
  mEVP has its OWN subcycle exchange, `fesom_ice_maevp.cpp:322`, aux-velocity scheme, out of
  scope); abort `FESOM_KK_VERIFY=evp` + EVPWIDE (the host twin is not wide-aware; the ICERAILS
  precedent, `fesom_ice.cpp:620`); `FESOM_CHECK(120 % K == 0)`.

## 4. IMPLEMENTATION PLAN (E.EVP1)

New `src/fesom_ice_evpwide.{h,cpp}` (~450 lines: scatter hook, owner vector, BFS, geometry
extract, handshake, lists, exchange, selfcheck) + edits: `fesom_mesh.cpp` (one hook call
before the global frees), `fesom_ice.cpp` (alloc tails, guards), `fesom_ice_evp.cpp`
(ghost lambdas in K1/K2/K3 extended ranges — knob-gated so OFF compiles to today's exact
ranges; ghost Step-2/3 maps; cadence `if ((sub+1)%K==0) evpwide_exchange(u,v) else nothing`;
per-step 11-field exchange after Step 4). ~800 lines total, CGPIPE-scale.

Execution order inside `fesom_ice_evp_dynamics_kk` (knob ON):
owned Steps 1-4 (byte-untouched) → extended 11-field exchange → ghost Step-2/3 maps →
subcycle loop [K1/K2/K3 with extended ranges; every K-th subcycle the 2-field wide refresh].
Ghost K3 branch per node: gather u_rhs in owner order → finalize → velupd → coastal (replay
of the owner's per-node sequence; skip save-old — u_old is unused downstream, write-only).
Ring-R slots: refreshed, never updated. Owned branches: EXACTLY today's code paths.

## 5. GATE LADDER (per-lever, L91 full matrix)

1. **Bring-up (login, pi mesh, np2+np8)**: K=1 (null rung — new path, old semantics) then
   K=2 (all machinery live: R=3, ring-2/3 discovery, gather, frozen ring). Selfcheck prints
   0.000e+00 (Serial). diff vs knob-OFF run.
2. **FORCE_SERIAL byte proof (CORE2 np8, private mesh L73)**: K ∈ {1,2,4,8} — diff_snap rc=0
   vs the certified baseline. **The headline gate: if §2 is right, ALL K pass bytewise.**
3. **Knob-OFF byte gate** on the new build: rc=0 (default path untouched).
4. **CUDA fidelity vs Serial baseline**, K=4 ON, bare FESOM_SPEED=1 + knob: PASS at the
   climate-close floors, ice-focused attention (uice/vice floors, L79/L75); the announce line
   must show `FESOM_SPEED_EVPWIDE = 4` (L80).
5. **In-vivo CUDA selfcheck magnitude**: report; hard-require Serial == 0.
6. **Options ×3** (TKE / mEVP / zstar) WITH EVPWIDE=4 set: mEVP leg must announce the no-op
   and stay byte-identical to its own baseline (the knob must not fire under whichEVP=1);
   TKE/zstar at their L79 floors. (EVPWIDE changes who owns ghost uice/vice ⇒ full matrix.)
7. **A/B same-alloc 300 steps, 16N FIRST** (bigger pool): ONE `job_m7_ab_env` with
   LEG1..4 = OFF / K=2 / K=4 / K=8. Then 4N: OFF / K* / one flank. Node-mix immune; keep
   `-C a100_80` habit for the paired anchors.
8. **1-yr climate leg at K*** before any promotion talk (user rule 0.24: stays OPT-IN).

## 6. PRE-REGISTRATION (L93 discipline: census floors, honest ranges, decided BEFORE jobs)

Site of record (L100 walker, findings-11 §2): EVP = 120 fused (uice,vice) NOD2D events/step,
**25.3 ms @4N / 31.6 ms @16N**; ~0.21/0.26 ms/event, 2D-latency class.

Event count: 120 → 120/K + 1 (the +1 = the per-step 11-field exchange).

| K | R | events | ceiling @4N (ms) | ceiling @16N | central (0.65×ceil) @4N/@16N | ghost rows @4N/@16N |
|---|---|---|---|---|---|---|
| 2 | 3 | 61 | −12.4 | −15.5 | −8 / −10 | ~1.8 % / ~3.5 % |
| 4 | 7 | 31 | −18.8 | −23.4 | −12 / −15 | ~4 % / ~8 % |
| 8 | 15 | 16 | −21.9 | −27.4 | −14 / −18 | ~9 % / ~18 % |

Floors: 0 (the handoff's rule; redundant compute + bigger messages + new partners eat some).
History note: 5 consecutive pre-regs measured wrong-LOW (L93 entanglement) — the ceiling may
be beaten at 16N; do NOT adjust on harvest, run the L94 checklist instead.

Central pre-reg (K=4): **4N 0.6381 → 0.6261 (−1.9 %) ⇒ 7.31×; 16N 0.2414 → 0.2264 (−6.2 %)
⇒ ~5.43×, Stage-2 SYPD ≈ 2.83.** At ceiling: 7.39× / 5.64×, SYPD ≈ 2.94. 8× @4N needs
−10.3 % total ⇒ EVPWIDE alone does not close it (imbalance 36 ms + CG transport ~18 ms + E.1
fuses remain on the list).

Decision rule (pre-registered): K* = argmax measured Δ at 16N with 4N non-regression;
adopt as OPT-IN regardless of size; promotion = user's call after the climate leg.

Watch-items measured in the same jobs: per-partner message bytes (wide refresh @4N K=4 ≈
50 KB/partner, @16N ≈ 25 KB — both may cross UCX_RNDV_THRESH ⇒ rndv/pinned-bounce regime; if
the A/B under-delivers, sweep `UCX_RNDV_THRESH` in the E.4 env leg before judging the lever);
partner-count growth (16N runs dc_mlx5 — new diagonal partners = new connects, one-time);
build-summary prints rings/partners/bytes like cgpipe.

## 7. RISKS

- R1 `eDim ≠ BFS-ring-1` (partition-tool quirk): ASSERT at build; if eDim ⊋ ring-1, define
  ring-1 := eDim ∪ BFS-ring-1 and BFS outward from it (superset refresh is harmless). Log.
- R2 coastal-mask semantics on ghost rings: assert global-edge mask == owner mask on MY owned
  nodes at build; abort on mismatch (falsifies the pure-global-mask assumption).
- R3 rndv threshold crossing (above) — env-first remedy, E.4.
- R4 thin ranks: rings defined by BFS distance; owners arbitrary (machinery handles
  non-neighbor owners via the handshake). No correctness constraint; memory trivial.
- R5 stale-freeze trap (0.23): rebuild BOTH backends after the LAST edit, THEN freeze.
- R6 K3 ghost gather reads K2 ghost sigma — cross-launch dependency (fine); NEVER fuse the
  gather into K2 (same-launch ordering is undefined on CUDA).
- R7 IO/restart: extended tails never leave the module ([0,myDim) IO unaffected). Snapshots
  byte-identical is exactly what gate 2 proves end-to-end.
- R8 `fesom_speed_int` is silent ⇒ explicit announce (L80) — the fidelity-gate grep depends
  on it.

## 8. OPEN ITEMS FOR THE USER

- Sessions 9-11 are LOCAL on `m7-speed` (through the session-11 close + this session's
  commits) — push approval? (Standing ask, PROMPT §2.1.)
- h17 anchors 26299413/14: harvest in flight this session (gates 26299411/12 already PASS).
- Pre-reg above stands unless the user objects; jobs follow the ladder order.
