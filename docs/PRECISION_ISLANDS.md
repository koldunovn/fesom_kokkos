# FESOM2-Kokkos precision registry (M16 — the conformance table)

**What this file is.** The authoritative list of everything deliberately kept FP64 (`dbl_t`) while
the working precision (`real_t`) is FP32, *and* the placement decision for every kind of code the
sweep touches, pre-registered before the first Phase-B slice (plan
`docs/plans/20260902-m16-mixed-precision.md`, Task A2). It restructures July's M8 registry
(`~/port_kokkos_mp/docs/PRECISION_ISLANDS.md`) around one question: **what does upstream FESOM2 do
(merged FESOM/fesom2#940), and where do we deliberately differ?**

**Rules.** (1) After the sweep every `dbl_t` in `src/` corresponds to a row here; a raw `double` in
state or kernel code is a sweep bug (grep-enforced, `scripts/m16_mpi_invariant.sh`). (2) A row is
added or changed only with evidence: *pre-registered* (design, cited) or *promoted* (the failing gate
signature that demanded it). (3) `MPI_DOUBLE` only ever pairs with `dbl_t`/`double` storage;
`FESOM_MPI_REAL` only with `real_t` (lesson SP1: the July stack smash). (4) Every promotion re-measures
the pinned pair and records the give-back. The history of this file **is** the precision-sensitivity map.

**Classes** (plan D5 — the conformance table has five):

| class | meaning | rule |
|---|---|---|
| **1 converged** | upstream and July agree | keep |
| **2 upstream stricter** | upstream keeps FP64 where July ran FP32 | follow upstream; measure the give-back with a define |
| **3 July stricter** | July islanded what upstream runs in WP | **flip to `real_t` in Phase B**; re-earn `dbl_t` only on a failing gate |
| **4 port-only** | code upstream does not have | `real_t`, except global integrals and diagnostic-only reductions |
| **5 exact port conflicts with Gate 0** | upstream's change alters FP64 rounding | compile SP-only; DP control define bounds the confound |

Status legend: `pre-registered` · `suspect-fp32` (runs FP32 until a gate objects) · `promoted`
(signature required) · `cleared` · **`flip-B`** (a July island going to `real_t` in Phase B).

---

## 1. Islands (`dbl_t`, never FP32) — with the upstream placement

| island | scope | class | upstream (#940) placement / Fortran line | reason | status |
|---|---|---|---|---|---|
| Forcing time chain: `rdate`, `nc_time`, `delta_t`, `time_t0`, `coef_a/b` formation, `dbl_t` binary search (`fesom_jra55.*`) | module | 1 | `gen_surface_forcing.F90`: `nc_time`/`rdate`/`delta_t`/`time_t0` real64, `binarysearch_r8`, `coef_a/coef_b` in WP (the VALUE narrowed once) | float ulp at JD≈2.44e6 is 6 h > 3-h JRA records ⇒ record-time collision ⇒ Inf coefficients (July promotion `7247412`, SP2); upstream fixed the same defect independently | pre-registered |
| Stiffness accumulation shadow `values_full` (`FieldT<dbl_t>`, local nnz, device) | zstar ALE | 1 (+ whole-field device promotion, declared divergence b) | `MOD_MESH.F90` `values_full` real64 under `#ifdef` SP; `oce_ale.F90 update_stiff_mat_ale` seeds on first call, accumulates real64, refreshes `%values` per step; `DIAG_STIFF_DRIFT` | 118k-step open-loop CSR increments in float de-tune the operator (July's 1964-10-04 death, `ssh-stiff-ale-acc`); upstream #997 has the same island | **ported 2026-09-07 (B3b):** `values_full_fld` seeded on the first `fesom_update_stiff_mat_ale_kk` call, `atomic_add` in `dbl_t`, `values = real_t(full)` per update; `FESOM_DIAG_STIFF_DRIFT=N` runtime twin + `[STIFFDRIFT]` every N updates (upstream: compile-time, every `logfile_outfreq`); SP restart writes/reads the shadow (divergence b) |
| Guard epsilons: `KPP_EPSLN` 1e-20 in SP (1e-40 flushes under FTZ), every additive `+eps` audited against denormal flush **and** overflow of the expression it feeds | constants | 1 | `oce_ale_mixing_kpp.F90` `epsln = 1e-20` in SP (upstream's headline SP bug, commit 48c37328) | SP4 | pre-registered. **Denormal posture (C2, 2026-09-07):** the CUDA build passes nvcc only `-O3` (no `--use_fast_math`/`-ftz`), so nvcc's defaults hold: `-ftz=false -prec-div=true -prec-sqrt=true -fmad=true` — denormals are NOT flushed on the GPU; the host builds are `-O3` without `-ffast-math`. Both backends therefore keep float denormals; a 1e-40 guard would survive here but stays 1e-20 to match upstream (and any future `-ffast-math`/`--use_fast_math` build). |
| On-disk schema: restarts and means stay `NC_DOUBLE`; values cast at the I/O boundary; fill value cast to `real_t` **before** comparison | I/O | 1 | `io_meandata.F90` / restart: 8-byte streams accumulate real64, 4-byte real32; `note_output_precision` | double holds every float exactly ⇒ SP→disk→SP bit-exact; DP→SP truncates by design | pre-registered |
| Output / time-mean accumulators (`fesom_io_acc_t = dbl_t`: host `accum[v]`, device `accum_dev[v]`, 34 resolvers, `gather_*_d`, `nc_put_vara_double`) | buffers | 1 | `io_meandata.F90` `local_values_r8` (8-byte streams accumulate real64; every port stream is NC_DOUBLE) | FP64 sums of FP32 samples over ~1e4 steps | pre-registered; **implemented B2 2026-09-07** |
| Global integrals (`integrate_nod` class: area/volume-weighted global sums, conservation diagnostics) + their `MPI_DOUBLE` reductions | scalars | 1 | `oce_modules.F90` `WP_full`; `integrate_nod_2D/3D` real64 with matching MPI type | FP32 sum over 1e6–1e9 elements loses ~n·2⁻²⁴ | pre-registered |
| CVMix-derived TKE arithmetic inside the kernel (`fesom_cvmix_tke.hpp`): `dbl_t` locals, `real_t` in/out | kernel | **2** | `gen_modules_cvmix_tke.F90`: CVMix is fixed real64, WP↔`cvmix_r8` shims at the call boundary | upstream stricter; July ran it FP32 and listed it "highest-probability promotion" without a failing gate; give-back measured with `FESOM_MP_TKE_REAL` (Task B7, Gate 2) | **implemented B7 2026-09-07** (`tke_t`) |
| Calendar day/month/year bookkeeping, integer seconds (`fesom_calendar.*`) | module | 1 for the integer/Julian part | `gen_modules_clock.F90` (integers + real64 day) | integers are precision-free; the real64 day joins the forcing chain above | pre-registered |

## 2. July islands that go to `real_t` in Phase B (class 3 — flip, then re-earn)

Every row here is something July kept double that upstream runs in WP. The flip is the exact port;
the **detector** column names the gate that would send it back.

| July island | files | upstream placement | detector that re-earns `dbl_t` | status |
|---|---|---|---|---|
| CG scalar chain: residual, `rtol`, α, β; `cg_dot` accumulators; `Allreduce` scalars of dots/norms | `fesom_ssh.cpp` (`cg_dot` :601-607, `rtol` :449/:2781/:3049/:3509 on m14) | `solver.F90` / `oce_ale_ssh`: all WP, `MPI_WP` | `[ssh-verify]` true-residual gap SP vs DP across a rank sweep (bar: gap(SP) ≤ 10×gap(DP); no solve past `rtol` by more than DP's worst); false convergence, not extra iterations, is the risk | **flipped B3 2026-09-07** (`cg_dot`, recurrence scalars, `ALLREDUCE_SUM` → `FESOM_MPI_REAL`; cg2/pipecg/oati fused reduces; Lanczos; pcsi; `[ssh-verify]` retyped `dbl_t`) |
| CGPIPE / CGPOLY eigen-bounds (λmax power iteration, Chebyshev coefficients) | `fesom_ssh.cpp` | port-only (class 4 by content) — upstream has no pipelined CG | same detector + `pcsi`/`oati` liveness on farc (the M10 stall was recurrence rounding) | **flipped B3 2026-09-07** |
| Mesh metrics precompute (areas, gradient coefficients, `ocean_area` local sum) | `fesom_mesh.cpp` | `oce_mesh_setup.F90`: WP after `-r4`; coordinates read as real64 then WP | mesh-metric byte gate is impossible at SP; detector = SE wide-halo drift 0.0 and the G4 pattern-correlation bar | **flipped B2 2026-09-07** (`ocean_area` real_t + `FESOM_MPI_REAL`; Bcasts) |
| PHC climatology / init path | `fesom_phc.cpp` (`:310`, `:555` Allreduce → `FESOM_MPI_REAL`) | `oce_setup_step.F90` / `gen_ic` WP | same-IC gate (det fill at SP equals DP fill to the rounding class) | flip-B |
| Min/max step diagnostics (the 16-scalar Allreduce) | `fesom_main.cpp` (`:334`, `:421`) | `write_step_info.F90` WP, `MPI_WP` | none needed (diagnostic); the SP1 stack-smash class is prevented by pairing the buffer type with the MPI type | flip-B |
| Mesh volume sums used by diagnostics (not the global-integral class) | `fesom_mesh.cpp` | WP | G4 conservation screen (`FESOM_MP_CONSERV`) | flip-B |
| Calendar seconds-in-day as a real | `fesom_calendar.*` | `gen_modules_clock.F90` `timenew` real64 → **stays** where it feeds the forcing chain; the seconds counter is an integer | verify at B6b; keep integer | flip-B (verify) |
| `parallel_reduce` accumulators of non-global-integral kind (23 sites in July) | kernels | WP in the Fortran (plain sums) | Kokkos reduces into `double` for free on the host; on device the reduce type follows the storage → `real_t`; global integrals stay class 1 | flip-B |

## 3. Port-only code (class 4) — `real_t` except global integrals / diagnostics

Inventory of 2026-09-07 over the m14-only files (`fesom_ssh_se.cpp` 141 `double` lines / 30
`MPI_DOUBLE`; `fesom_io_restart.cpp` 13/3; `fesom_ice_evpwide.cpp` 63/17; `fesom_halo_device.cpp` 45/13;
`fesom_phc.cpp` 56/2; `fesom_ssh.cpp` M10 scope 204/34; `fesom_main.cpp` 99/3). "diag" = diagnostic-only
reduction (selfcheck, audit, drift print, timer) that may stay `double` and is not a registry island.

| code | sites (m14 lines) | decision | note |
|---|---|---|---|
| **SE barotropic state**: H0e/Hep/metap, subcycle η/U, viscosity per element, AB3-AM4 weights | `fesom_ssh_se.cpp` :1815-1827, :2141-2160, :1701-1745 | `real_t` (weights formed in `double`, cast once — setup scalars) | the 20–90-substep `+=` chain is in the ledger; detector = H0e wide-halo drift 0.0 every step |
| SE reconstruction halo exchange | :1419/:1425 `MPI_DOUBLE` over `real_t*` | **`FESOM_MPI_REAL`** | silent corruption class (SP1) |
| SE divergence-coefficient ship (2 per nz, `std::vector<double>` + `MPI_DOUBLE` :1024/:1034), operator assembly from `edge_cross_dxdy` (:1106-1148) | geometry ship | `real_t` + `FESOM_MPI_REAL` | mesh metrics are class 3 (WP upstream) |
| SE barotropic CFL limit `dtbt_lim` (:542-557, 1-scalar `MPI_DOUBLE` MIN) | setup scalar | `double` (storage is `double`, MPI type matches) | setup-once; diag |
| SE conservation accumulators `sumh0`, `eta_area0`, `wf_area_acc`, ∫η dA / ∫W dA sums (:466-475, :2487-2527) | global integrals | **`dbl_t` + `MPI_DOUBLE`** (class 1, `WP_full`) | |
| SE audits: cross-rank determinism min/max (`std::vector<double>` + Allreduce :265-292, :696-762, :1835-1932), device `Kokkos::Max<double>` wide-halo checks (:2022-2056, :2223-2270, :2386-2451), `se_hash11` test vector | diag | `double` (diag) | unchanged; excluded from the invariant grep by the `diag` marker |
| **Restart I/O** staging: `nc_put_var_double`/`nc_get_var_double` over `real_t *plane`/`sval` (:658/:667/:799/:839) | disk boundary | `nc_*_var_float` under SP via `fesom_nc_real.h` helpers (July), file stays `NC_DOUBLE` | class 1: double on disk, cast at the boundary; SP→disk→SP exact |
| Restart stiffness gather/scatter `MPI_DOUBLE` over `real_t *val` (:334/:421) | MPI | `FESOM_MPI_REAL`; **under SP the shadow (`dbl_t`) is what is written** ⇒ `MPI_DOUBLE` over `dbl_t` (Task B3b, divergence b) | done 2026-09-07: `stiff_pack/gather/scatter<T>` with `rst_mpi_of<T>()`; SP writer gathers `values_full` (widened `values` under linfs), reader seeds `values_full` from the NC_DOUBLE file and rounds into `values` |
| Restart header `dt`, `second` (`double`, `MPI_Bcast` 2 doubles :701-768) | header | `double` | genuine double storage |
| **EVPWIDE lean**: `static_assert(sizeof(real_t)==sizeof(double))` (:41) | compile gate | **remove**; replace by the MPI-type pairing | the file currently refuses to build at SP |
| EVPWIDE ghost ships: build-time (area0/coriolis/bc :507/:516; 8 element metrics :585/:589) and per-substep (six `MPI_DOUBLE` over `real_t` :1086-1126) | MPI | `FESOM_MPI_REAL` | wire counters `dbl_count` etc. stay `double` (timing) |
| EVPWIDE selfchecks/audits (`Kokkos::Max<double>` + `MPI_Reduce` 8/11/1/6 scalars :632-689, :1156-1184, :1341-1394, :1545-1557; host re-derivation :1211-1309) | diag | `double` (diag) | |
| **Device halo staging** `Kokkos::View<double*>` send/recv + pinned mirrors (:210-213, :320-378), `MPI_DOUBLE` at :454/:460/:556/:562/:662/:668, byte accounting `sizeof(double)` (:410/:511/:616) | halo traffic | **`real_t` views + `FESOM_MPI_REAL` + `sizeof(real_t)`** | upstream halo is `MPI_WP`; the M8 diff for this file (+119) is the template |
| Halo host-vs-device verification max (:705-727) | diag | `double` | |
| **PHC**: netCDF axes/data staging `double` + `nc_get_var(a)_double` (:99-128, :654-665, :696-703) | disk boundary | `double` staging (file type), interpolation result cast to `real_t` (:244) | class 1 boundary |
| PHC `atg`/`ptheta` UNESCO polynomials, bilinear/vertical interpolation weights (:56-95, :177-244) | init path | **`real_t`** (class 3 flip) | same-IC gate |
| PHC extrapolation Allreduces `MPI_DOUBLE` over `real_t` (:310, :555) | MPI | `FESOM_MPI_REAL` | the det fill's `FESOM_IC_EXTRAP_TOL` per precision |
| **M10 solvers**: preconditioner views cast `Kokkos::View<const double*>` from a `real_t` Field (:2890, :3153, :3607, :3796); `g_sympre.pr_d` hard `double` (:2602/:2650) | STATE | `real_t` | |
| cgpipe/cgpoly ring buffers (`View<double*>` :809-816, :1038-1055, :1220 `RDV`, :1894), preconditioner-row/Ã-row/diagonal ships (`std::vector<double>` + `MPI_DOUBLE` :902/:908, :1564-1570, :1663-1669, :1726/:1739), `pv2` (:916-1055), D̃⁻¹/D̃^{-1/2} arrays (:1762-1832), `8.0*` byte accounting (:1112, :1291) | STATE / MPI | `real_t` + `FESOM_MPI_REAL` + `sizeof(real_t)` | |
| CG recurrence scalars and dots: baseline `ALLREDUCE_SUM` macro `MPI_DOUBLE` over `real_t` (:4206), cg2/pipecg/oati fused reduces of 3/3/10 `double` scalars (:2973, :3022, :3257 `MPI_Iallreduce`, :3858), pcsi `resid_norm` (:3660), power-iteration norm (:1946), Lanczos dot (:3459) | SCALAR-ACC | **`real_t` + `FESOM_MPI_REAL`** (class 3 flip; upstream WP/`MPI_WP`) | detector: `[ssh-verify]` gap + rank sweep; the fArc stall was recurrence rounding — expected first promotion |
| Eigen-bounds: cgpoly `lam_min/lam_max/kappa` + Chebyshev θ/δ/σ/ρ (:1244, :1361-1432), Lanczos Sturm/bisection/ν,μ + 2-scalar MIN/MAX agreement (:3356-3533), pcsi γ/α/ω (:3678-3681), cgpoly guard 2-scalar reduce (:4357) | SCALAR-ACC | `real_t` (flip), bounds agreement reduce over `real_t` with `FESOM_MPI_REAL` | wrong bounds ⇒ divergent polynomial: liveness of `pcsi`/`cgpoly` at SP is the detector |
| `[ssh-verify]` true residual (`tn` + 1-scalar reduce :3061, :3318, :4002, :4431; `v_maxtrue/v_maxgap` :2275) | instrument | **`dbl_t` + `MPI_DOUBLE`** (measurement of record, plan B3) | SpMV in `dbl_t` over the `real_t` matrix |
| Solver dump header `f64` (:2358-2386), wire report/timers (:2427-2454, :4096, :4285) | IO / timing | `double` | |
| `fesom_ssh_block_verify` host-vs-device maxima (:4527-4552) | diag | `double` | |
| **Main**: step-diag 16-max/3-min `MPI_DOUBLE` over `real_t` (:1570/:1571), `[cflzmax]` (:1633) | MPI | `FESOM_MPI_REAL` (class 3 row) | SP1 |
| Main timing/profile/phase stats (:1249-1259, :1780-1795), restart `dt` casts, IC/probe prints | timing / prints | `double` | |

**MPI/disk pairing audit (the B8 invariant's seed, all m14 lines):** `MPI_DOUBLE` bound to `real_t`
storage — `fesom_ssh_se.cpp:1419,1425`; `fesom_io_restart.cpp:334,421`;
`fesom_ice_evpwide.cpp:507,516,585,589,1086,1094,1101,1109,1118,1126`; `fesom_phc.cpp:310,555`;
`fesom_ssh.cpp:4206` (macro, ~10 uses); `fesom_main.cpp:1570,1571,1633`. `nc_*_var_double` bound to
`real_t` — `fesom_io_restart.cpp:658,667,799,839`. Hard-`double` device views —
`fesom_halo_device.cpp:210-213,320,337-344,350,364,375-378`;
`fesom_ssh.cpp:809-816,1038-1042,1055,1220,1894-1895,2602,2650,2890-2891,3153-3154,3607-3608,3796-3797`.
8-bytes-per-element accounting — `fesom_halo_device.cpp:410,511,616`; `fesom_ssh.cpp:1112,1291`.
Compile-time FP64 gate — `fesom_ice_evpwide.cpp:41`. Every one of these is a Phase-B slice item.

## 4. Exact-port conflicts with Gate 0 (class 5 — SP-only)

| element | upstream | port | control |
|---|---|---|---|
| Point-slope forcing interpolation `atm = coef_b + (rdate − time_t0)·coef_a` | `gen_surface_forcing.F90` (unconditional, changes FP64 rounding) | compiled under `FESOM_SINGLE_PRECISION` only; FP64 keeps the affine line bit-for-bit (**implemented B6b 2026-09-07**: `FESOM_JRA_POINTSLOPE`, `coef_a/b` WP, `time_t0` dbl_t per field) | `FESOM_FORCING_POINTSLOPE` DP define; one DP control leg in Gate 3 bounds the confound |

## 5. Suspects (FP32 until a gate objects — promotion order on failure)

| suspect | scope | why suspect | status |
|---|---|---|---|
| EOS + pressure-gradient chain (`fesom_eos.cpp`) | module | PGF cancellation; anomaly form helps; upstream ran EOS in WP for its 1-yr twin without incident | suspect-fp32 (promote FIRST) |
| Salt anomaly consumers (`FESOM_SALT_ANOMALY`, Phase D) | module | upstream's own SP measure: S−35 halves the absorbed ulp; the invariance gate (S_ref 35 vs 10 vs off) is the completeness check | suspect-fp32 (it is the mitigation, not a hazard) |
| FCT tracer advection, ice FCT | module | conservation under limiter arithmetic over decades | suspect-fp32 |
| mEVP stress iteration; EVPWIDE lean ghosts | module | stiff subcycled relaxation; the lean ghost kernels recompute halo rows locally (drift 0.0 assertion exists) | suspect-fp32 |
| KPP stability functions | kernels | Richardson-number cancellations; guard class handled above | suspect-fp32 |
| SE barotropic subcycle (`fesom_ssh_se.cpp`, 20–90 substeps per step, `+=` sites in the ledger) | module | the stiffness-drift class in miniature: many small increments into a running state; the H0e wide-halo drift check is the detector | suspect-fp32 |
| M10 CA-solver recurrences (`cg2`, `pipecg`, `oati`, `pcsi` Lanczos/Chebyshev) | solvers | rounding-fragile by construction (the fArc stall); `[ssh-verify]` + rank sweep is the detector | suspect-fp32 |

## 6. Accumulation ledger (grep-generated — `scripts/m16_accum_ledger.py`)

Every `x[...] += …` / `-=` site whose left side is an indexed array element or Kokkos view (model state
or a running sum). Definition of record: indexed lvalues only; scalar per-column temporaries (`t +=
res/deriv` in the thermodynamics Newton loop) are not time-accumulating state and are excluded; timing,
wire counters and integer offsets are excluded. The site count printed at the end is the A2 gate value.

<!-- LEDGER START -->
| site | statement |
|---|---|
| `src/fesom_ale.cpp:141` | `dyn->w[FESOM_NODE3D(n1, nz, nl)] += c1;` |
| `src/fesom_ale.cpp:142` | `dyn->w[FESOM_NODE3D(n2, nz, nl)] -= c1;` |
| `src/fesom_ale.cpp:147` | `dyn->fer_w[FESOM_NODE3D(n1, nz, nl)] += c2;` |
| `src/fesom_ale.cpp:148` | `dyn->fer_w[FESOM_NODE3D(n2, nz, nl)] -= c2;` |
| `src/fesom_ale.cpp:163` | `dyn->w[FESOM_NODE3D(n1, nz, nl)] += c1;` |
| `src/fesom_ale.cpp:164` | `dyn->w[FESOM_NODE3D(n2, nz, nl)] -= c1;` |
| `src/fesom_ale.cpp:169` | `dyn->fer_w[FESOM_NODE3D(n1, nz, nl)] += c2;` |
| `src/fesom_ale.cpp:170` | `dyn->fer_w[FESOM_NODE3D(n2, nz, nl)] -= c2;` |
| `src/fesom_ale.cpp:183` | `dyn->w[FESOM_NODE3D(n, nz, nl)] +=` |
| `src/fesom_ale.cpp:186` | `dyn->fer_w[FESOM_NODE3D(n, nz, nl)] +=` |
| `src/fesom_ale.cpp:241` | `dyn->cfl_z[FESOM_NODE3D(n, nz,     nl)] += c1;` |
| `src/fesom_ale.cpp:242` | `dyn->cfl_z[FESOM_NODE3D(n, nz + 1, nl)] += c2;` |
| `src/fesom_ale.cpp:549` | `w(FESOM_NODE3D(n, nz, nl)) += w(FESOM_NODE3D(n, nz + 1, nl));` |
| `src/fesom_ale.cpp:551` | `fer_w(FESOM_NODE3D(n, nz, nl)) += fer_w(FESOM_NODE3D(n, nz + 1, nl));` |
| `src/fesom_ale.cpp:626` | `cfl_z(FESOM_NODE3D(n, nz,     nl)) += c1;` |
| `src/fesom_ale.cpp:627` | `cfl_z(FESOM_NODE3D(n, nz + 1, nl)) += c2;` |
| `src/fesom_ale.cpp:1081` | `w(k)         -= (zbar3d(k) - dd1) * dddt;` |
| `src/fesom_ale.cpp:1086` | `w(FESOM_NODE3D(n, nzmin_f - 1, nl)) -= wf(n);` |
| `src/fesom_bulk.cpp:932` | `forcing->heat_flux[n2] += swsurf;   /* M7 FLUXDEV: moved to the _kk twin (device) */` |
| `src/fesom_bulk.cpp:1016` | `if (fluxdev) hf(n2) += swsurf;` |
| `src/fesom_eos.cpp:1230` | `int_dp_dx[0] += aux_sum;` |
| `src/fesom_eos.cpp:1237` | `int_dp_dx[1] += aux_sum;` |
| `src/fesom_gm.cpp:166` | `vol[nz] += a;` |
| `src/fesom_gm.cpp:167` | `tx[nz] += (g[0]*T[i0] + g[1]*T[i1] + g[2]*T[i2]) * a;` |
| `src/fesom_gm.cpp:168` | `ty[nz] += (g[3]*T[i0] + g[4]*T[i1] + g[5]*T[i2]) * a;` |
| `src/fesom_gm.cpp:169` | `sx[nz] += (g[0]*S[i0] + g[1]*S[i1] + g[2]*S[i2]) * a;` |
| `src/fesom_gm.cpp:170` | `sy[nz] += (g[3]*S[i0] + g[4]*S[i1] + g[5]*S[i2]) * a;` |
| `src/fesom_gm.cpp:915` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:916` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:929` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:930` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:954` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:955` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:967` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:968` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:981` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:982` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:2185` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:2186` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:2199` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:2200` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:2223` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:2224` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:2236` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:2237` | `rhs2[nz] -= c;` |
| `src/fesom_gm.cpp:2250` | `rhs1[nz] += c;` |
| `src/fesom_gm.cpp:2251` | `rhs2[nz] -= c;` |
| `src/fesom_ic.cpp:126` | `T[FESOM_NODE3D(n, nz, nl)] += amp_C * prof_h * prof_z;` |
| `src/fesom_ice_coupling.cpp:283` | `forcing->virtual_salt[n] -= net;` |
| `src/fesom_ice_coupling.cpp:297` | `forcing->relax_salt[n] -= net_relax;` |
| `src/fesom_ice_coupling.cpp:382` | `KOKKOS_LAMBDA(const int n) { if (ulev_n(n) > 1) return; vs(n) -= net; });` |
| `src/fesom_ice_evp.cpp:181` | `u_rhs[n] -= a * (s11*gs[k]   + s12*gs[k+3]` |
| `src/fesom_ice_evp.cpp:183` | `v_rhs[n] -= a * (s12*gs[k]   + s22*gs[k+3]` |
| `src/fesom_ice_evp.cpp:346` | `rhs_a[n] -= aa * edx;` |
| `src/fesom_ice_evp.cpp:347` | `rhs_m[n] -= aa * edy;` |
| `src/fesom_ice_evpwide.cpp:1359` | `for (int i = 0; i < S.nrecv; ++i) (ridx_h(i) < S.N ? nloc[0] : nloc[1]) += 1;` |
| `src/fesom_ice_fct.cpp:120` | `mm[ipos] += mesh->elem_area[elem] / 12.0;` |
| `src/fesom_ice_fct.cpp:122` | `mm[ipos] += mesh->elem_area[elem] / 12.0;` |
| `src/fesom_ice_fct.cpp:227` | `rhs_m [row] += sm;` |
| `src/fesom_ice_fct.cpp:228` | `rhs_a [row] += sa;` |
| `src/fesom_ice_fct.cpp:229` | `rhs_ms[row] += sms;` |
| `src/fesom_ice_fct.cpp:460` | `if (flux > 0.0) icepplus [n] += flux;` |
| `src/fesom_ice_fct.cpp:461` | `else            icepminus[n] += flux;` |
| `src/fesom_ice_fct.cpp:521` | `vals[en[q]] += icefluxes[elem * 3 + q];` |
| `src/fesom_io.cpp:818` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:823` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:830` | `for (size_t i = 0; i < n; ++i) out[i] += T[i * (size_t)nl + 0];` |
| `src/fesom_io.cpp:836` | `for (size_t i = 0; i < n; ++i) out[i] += S[i * (size_t)nl + 0];` |
| `src/fesom_io.cpp:841` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:873` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:878` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:885` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:890` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:895` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:900` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:905` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:910` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:915` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:920` | `for (size_t i = 0; i < n; ++i) out[i] += src[i];` |
| `src/fesom_io.cpp:992` | `out(i) += uv(e * (size_t)nl * 2 + k * 2 + 0);` |
| `src/fesom_io.cpp:1010` | `out(i) += uv(e * (size_t)nl * 2 + k * 2 + 1);` |
| `src/fesom_io.cpp:1207` | `for (int c = 0; c < e->n_cadences; ++c) per_cad_count[e->cadences[c]] += 1;` |
| `src/fesom_io.cpp:1209` | `per_cad_count[FESOM_PERIOD_MONTHLY] += 1;` |
| `src/fesom_io_stream.cpp:473` | `s->time_index[v] += 1;` |
| `src/fesom_jra55.cpp:214` | `if (buf[i] >= 'A' && buf[i] <= 'Z') buf[i] += 32;` |
| `src/fesom_jra55.cpp:270` | `flf->nc_time[flf->Ntime - 1] += (flf->nc_time[flf->Ntime - 1]` |
| `src/fesom_mesh.cpp:528` | `m->nod_in_elem2D_offsets[i] += m->nod_in_elem2D_offsets[i - 1];` |
| `src/fesom_mesh.cpp:577` | `if      (e < Eown)       c[0] += bad;` |
| `src/fesom_mesh.cpp:578` | `else if (e < Eown + EeD) c[1] += bad;` |
| `src/fesom_mesh.cpp:579` | `else                     c[2] += bad;` |
| `src/fesom_mesh.cpp:887` | `m->area[FESOM_NODE3D(n, nz, m->nl)] += third;` |
| `src/fesom_mesh.cpp:923` | `if (ax[k] - amin >=  half_cyc) ax[k] -= cyc;` |
| `src/fesom_mesh.cpp:924` | `if (ax[k] - amin <  -half_cyc) ax[k] += cyc;` |
| `src/fesom_momentum.cpp:114` | `dyn->uv_rhs[k + 0] += (Fx - pgfx) * area;` |
| `src/fesom_momentum.cpp:115` | `dyn->uv_rhs[k + 1] += (Fy - pgfy) * area;` |
| `src/fesom_momentum.cpp:187` | `wu[0] += uv[FESOM_ELEMVEC(el, 0, nl) + 0] * a;` |
| `src/fesom_momentum.cpp:188` | `wv[0] += uv[FESOM_ELEMVEC(el, 0, nl) + 1] * a;` |
| `src/fesom_momentum.cpp:191` | `wu[j] += 0.5 * (uv[FESOM_ELEMVEC(el, j, nl) + 0] + uv[FESOM_ELEMVEC(el, j - 1, nl) + 0]) * a;` |
| `src/fesom_momentum.cpp:192` | `wv[j] += 0.5 * (uv[FESOM_ELEMVEC(el, j, nl) + 1] + uv[FESOM_ELEMVEC(el, j - 1, nl) + 1]) * a;` |
| `src/fesom_momentum.cpp:230` | `un[FESOM_ELEMVEC(n1,nz,nl)+0] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+0];` |
| `src/fesom_momentum.cpp:231` | `un[FESOM_ELEMVEC(n1,nz,nl)+1] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+1];` |
| `src/fesom_momentum.cpp:235` | `un[FESOM_ELEMVEC(n2,nz,nl)+0] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+0];` |
| `src/fesom_momentum.cpp:236` | `un[FESOM_ELEMVEC(n2,nz,nl)+1] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1] + un2[nz]*uv[FESOM_ELEMVEC(el2,nz,nl)+1];` |
| `src/fesom_momentum.cpp:243` | `un[FESOM_ELEMVEC(n1,nz,nl)+0] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0];` |
| `src/fesom_momentum.cpp:244` | `un[FESOM_ELEMVEC(n1,nz,nl)+1] += un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1];` |
| `src/fesom_momentum.cpp:248` | `un[FESOM_ELEMVEC(n2,nz,nl)+0] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+0];` |
| `src/fesom_momentum.cpp:249` | `un[FESOM_ELEMVEC(n2,nz,nl)+1] -= un1[nz]*uv[FESOM_ELEMVEC(el1,nz,nl)+1];` |
| `src/fesom_momentum.cpp:273` | `dyn->uv_rhsAB[FESOM_ELEMVEC(el,nz,nl)+0] += a * (un[FESOM_ELEMVEC(v0,nz,nl)+0]` |
| `src/fesom_momentum.cpp:275` | `dyn->uv_rhsAB[FESOM_ELEMVEC(el,nz,nl)+1] += a * (un[FESOM_ELEMVEC(v0,nz,nl)+1]` |
| `src/fesom_momentum.cpp:332` | `wu[0] += uv(FESOM_ELEMVEC(el, 0, nl) + 0) * a;` |
| `src/fesom_momentum.cpp:333` | `wv[0] += uv(FESOM_ELEMVEC(el, 0, nl) + 1) * a;` |
| `src/fesom_momentum.cpp:336` | `wu[j] += 0.5 * (uv(FESOM_ELEMVEC(el, j, nl) + 0) + uv(FESOM_ELEMVEC(el, j - 1, nl) + 0)) * a;` |
| `src/fesom_momentum.cpp:337` | `wv[j] += 0.5 * (uv(FESOM_ELEMVEC(el, j, nl) + 1) + uv(FESOM_ELEMVEC(el, j - 1, nl) + 1)) * a;` |
| `src/fesom_momentum.cpp:441` | `uv_rhsAB(FESOM_ELEMVEC(el,nz,nl)+0) += a * (un(FESOM_ELEMVEC(v0,nz,nl)+0)` |
| `src/fesom_momentum.cpp:443` | `uv_rhsAB(FESOM_ELEMVEC(el,nz,nl)+1) += a * (un(FESOM_ELEMVEC(v0,nz,nl)+1)` |
| `src/fesom_momentum.cpp:533` | `uv_rhs(k + 0) += (Fx - pgfx) * areav;` |
| `src/fesom_momentum.cpp:534` | `uv_rhs(k + 1) += (Fy - pgfy) * areav;` |
| `src/fesom_momentum.cpp:734` | `ur[nzmin] += zinv_top * forcing->stress_surf[2*e + 0] * inv_density_0;` |
| `src/fesom_momentum.cpp:735` | `vr[nzmin] += zinv_top * forcing->stress_surf[2*e + 1] * inv_density_0;` |
| `src/fesom_momentum.cpp:746` | `ur[nz] += zinv_bot * friction * u_bot;` |
| `src/fesom_momentum.cpp:747` | `vr[nz] += zinv_bot * friction * v_bot;` |
| `src/fesom_momentum.cpp:759` | `ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th + c[nz]*u_dn;` |
| `src/fesom_momentum.cpp:760` | `vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th + c[nz]*v_dn;` |
| `src/fesom_momentum.cpp:768` | `ur[nz] -= (b[nz] - 1.0)*u_th + c[nz]*u_dn;` |
| `src/fesom_momentum.cpp:769` | `vr[nz] -= (b[nz] - 1.0)*v_th + c[nz]*v_dn;` |
| `src/fesom_momentum.cpp:777` | `ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th;` |
| `src/fesom_momentum.cpp:778` | `vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th;` |
| `src/fesom_momentum.cpp:959` | `ur[nzmin] += zinv_top * stress(2*e + 0) * inv_density_0;` |
| `src/fesom_momentum.cpp:960` | `vr[nzmin] += zinv_top * stress(2*e + 1) * inv_density_0;` |
| `src/fesom_momentum.cpp:970` | `ur[nz] += zinv_bot * friction * u_bot;` |
| `src/fesom_momentum.cpp:971` | `vr[nz] += zinv_bot * friction * v_bot;` |
| `src/fesom_momentum.cpp:983` | `ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th + c[nz]*u_dn;` |
| `src/fesom_momentum.cpp:984` | `vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th + c[nz]*v_dn;` |
| `src/fesom_momentum.cpp:992` | `ur[nz] -= (b[nz] - 1.0)*u_th + c[nz]*u_dn;` |
| `src/fesom_momentum.cpp:993` | `vr[nz] -= (b[nz] - 1.0)*v_th + c[nz]*v_dn;` |
| `src/fesom_momentum.cpp:1001` | `ur[nz] -= a[nz]*u_up + (b[nz] - 1.0)*u_th;` |
| `src/fesom_momentum.cpp:1002` | `vr[nz] -= a[nz]*v_up + (b[nz] - 1.0)*v_th;` |
| `src/fesom_momentum.cpp:1106` | `dyn->uv[k + 0] += dyn->uv_rhs[k + 0] + Fx;` |
| `src/fesom_momentum.cpp:1107` | `dyn->uv[k + 1] += dyn->uv_rhs[k + 1] + Fy;` |
| `src/fesom_momentum.cpp:1154` | `uv(k + 0) += uv_rhs(k + 0) + Fx;` |
| `src/fesom_momentum.cpp:1155` | `uv(k + 1) += uv_rhs(k + 1) + Fy;` |
| `src/fesom_momentum.cpp:1221` | `dyn->u_b[FESOM_ELEM3D(el1, nz, nl)] -= du / a1;` |
| `src/fesom_momentum.cpp:1222` | `dyn->v_b[FESOM_ELEM3D(el1, nz, nl)] -= dv / a1;` |
| `src/fesom_momentum.cpp:1223` | `dyn->u_b[FESOM_ELEM3D(el2, nz, nl)] += du / a2;` |
| `src/fesom_momentum.cpp:1224` | `dyn->v_b[FESOM_ELEM3D(el2, nz, nl)] += dv / a2;` |
| `src/fesom_momentum.cpp:1380` | `Uc[FESOM_ELEM3D(el1, nz, nl)] -= du;` |
| `src/fesom_momentum.cpp:1381` | `Vc[FESOM_ELEM3D(el1, nz, nl)] -= dv;` |
| `src/fesom_momentum.cpp:1382` | `Uc[FESOM_ELEM3D(el2, nz, nl)] += du;` |
| `src/fesom_momentum.cpp:1383` | `Vc[FESOM_ELEM3D(el2, nz, nl)] += dv;` |
| `src/fesom_momentum.cpp:1416` | `dyn->uv_rhs[FESOM_ELEMVEC(el1, nz, nl) + 0] -= du / a1;` |
| `src/fesom_momentum.cpp:1417` | `dyn->uv_rhs[FESOM_ELEMVEC(el1, nz, nl) + 1] -= dv / a1;` |
| `src/fesom_momentum.cpp:1418` | `dyn->uv_rhs[FESOM_ELEMVEC(el2, nz, nl) + 0] += du / a2;` |
| `src/fesom_momentum.cpp:1419` | `dyn->uv_rhs[FESOM_ELEMVEC(el2, nz, nl) + 1] += dv / a2;` |
| `src/fesom_momentum.cpp:1665` | `dyn->ssh_rhs_old[n1] += (c1 + c2);` |
| `src/fesom_momentum.cpp:1666` | `dyn->ssh_rhs_old[n2] -= (c1 + c2);` |
| `src/fesom_momentum.cpp:1772` | `ssh_rhs_old(n) -= wf(n) * areasvol(FESOM_NODE3D(n, nzmin_f - 1, nl));` |
| `src/fesom_phc.cpp:769` | `if (T[k] > 100.0) T[k] -= 273.15;` |
| `src/fesom_ssh.cpp:199` | `S->values[npos[k]] += fy[k] * factor;` |
| `src/fesom_ssh.cpp:211` | `S->values[npos[k]] -= fy[k] * factor;` |
| `src/fesom_ssh.cpp:226` | `S->values[diag] += mesh->areasvol[FESOM_NODE3D(row, top_layer, mesh->nl)] * inv_dt;` |
| `src/fesom_ssh.cpp:482` | `dyn->ssh_rhs[n1] += (c1 + c2);` |
| `src/fesom_ssh.cpp:483` | `dyn->ssh_rhs[n2] -= (c1 + c2);` |
| `src/fesom_ssh.cpp:490` | `dyn->ssh_rhs[n] += one_minus_alpha * dyn->ssh_rhs_old[n];` |
| `src/fesom_ssh.cpp:623` | `X [row] += al * pp [row];` |
| `src/fesom_ssh.cpp:624` | `rr[row] -= al * App[row];` |
| `src/fesom_ssh.cpp:2100` | `ssh_rhs(n) += -alpha * wf(n)` |
| `src/fesom_ssh.cpp:3005` | `X (row) += al * pp(row);` |
| `src/fesom_ssh.cpp:3006` | `rr(row) -= al * ss(row);` |
| `src/fesom_ssh.cpp:3295` | `X (i) += al * p;` |
| `src/fesom_ssh.cpp:3296` | `rr(i) -= al * s;` |
| `src/fesom_ssh.cpp:3297` | `uu(i) -= al * q;` |
| `src/fesom_ssh.cpp:3298` | `ww(i) -= al * z;` |
| `src/fesom_ssh.cpp:3692` | `KOKKOS_LAMBDA(const int i) { const real_t d = ig * rp(i); dx(i) = d; X(i) += d; });` |
| `src/fesom_ssh.cpp:3718` | `dx(i) = d; X(i) += d; });` |
| `src/fesom_ssh.cpp:3871` | `X (i) += al * p;` |
| `src/fesom_ssh.cpp:3872` | `rr(i) -= al * s;` |
| `src/fesom_ssh.cpp:3873` | `uu(i) -= al * q;` |
| `src/fesom_ssh.cpp:3874` | `ww(i) -= al * z;` |
| `src/fesom_ssh.cpp:4318` | `X (row) += al * pp (row);` |
| `src/fesom_ssh.cpp:4319` | `rr(row) -= al * App(row);` |
| `src/fesom_ssh_se.cpp:1947` | `Umean[i] += uab * invM;` |
| `src/fesom_ssh_se.cpp:2522` | `loc[0] += (double)hb[n] * A;` |
| `src/fesom_ssh_se.cpp:2523` | `loc[1] += (double)wfh[n] * A;` |
| `src/fesom_sss_runoff.cpp:168` | `if (lon[i] < 0.0) lon[i] += 360.0;` |
| `src/fesom_sss_runoff.cpp:412` | `forcing->virtual_salt[n] -= net;` |
| `src/fesom_sss_runoff.cpp:429` | `forcing->relax_salt[n] -= net;` |
| `src/fesom_sss_runoff.cpp:453` | `forcing->water_flux[n] += net;` |
| `src/fesom_tracer_adv.cpp:149` | `sc->fct_LO[FESOM_NODE3D(n1, nz, nl)] += f;` |
| `src/fesom_tracer_adv.cpp:150` | `sc->fct_LO[FESOM_NODE3D(n2, nz, nl)] -= f;` |
| `src/fesom_tracer_adv.cpp:767` | `dttf_v[FESOM_NODE3D(n, nz, nl)] += (f_top - f_bot) * dt / a;` |
| `src/fesom_tracer_adv.cpp:790` | `if (a1 > 0.0) dttf_h[FESOM_NODE3D(n1, nz, nl)] += f * dt / a1;` |
| `src/fesom_tracer_adv.cpp:791` | `if (a2 > 0.0) dttf_h[FESOM_NODE3D(n2, nz, nl)] -= f * dt / a2;` |
| `src/fesom_tracer_adv.cpp:814` | `del_ttf[k] += T[k] * (hnode_old - hnode_new);` |
| `src/fesom_tracer_adv.cpp:816` | `T[k] += del_ttf[k] / hnode_new;` |
| `src/fesom_tracer_adv.cpp:1011` | `fct_plus [k] += pos;` |
| `src/fesom_tracer_adv.cpp:1012` | `fct_minus[k] += neg;` |
| `src/fesom_tracer_adv.cpp:1035` | `fct_plus [k1] += (f > 0.0 ? f : 0.0);` |
| `src/fesom_tracer_adv.cpp:1036` | `fct_minus[k1] += (f < 0.0 ? f : 0.0);` |
| `src/fesom_tracer_adv.cpp:1037` | `fct_plus [k2] += (-f > 0.0 ? -f : 0.0);` |
| `src/fesom_tracer_adv.cpp:1038` | `fct_minus[k2] += (-f < 0.0 ? -f : 0.0);` |
| `src/fesom_tracer_adv.cpp:1168` | `dttf_v[k] += -ttf[k] * mesh->hnode    [k]` |
| `src/fesom_tracer_adv.cpp:1176` | `dttf_v[FESOM_NODE3D(n, nz, nl)] += (f_top - f_bot) * dt / a;` |
| `src/fesom_tracer_adv.cpp:1200` | `if (a1 > 0.0) dttf_h[FESOM_NODE3D(n1, nz, nl)] += f * dt / a1;` |
| `src/fesom_tracer_adv.cpp:1201` | `if (a2 > 0.0) dttf_h[FESOM_NODE3D(n2, nz, nl)] -= f * dt / a2;` |
| `src/fesom_tracer_adv.cpp:1294` | `ttf[FESOM_NODE3D(n, nz, nl)] += tr[nz];` |
| `src/fesom_tracer_adv.cpp:2061` | `dtv(k) += -vals(k)*hnode(k) + fctLO(k)*hnode_new(k);      /* LO transition */` |
| `src/fesom_tracer_adv.cpp:2065` | `dtv(k) += (f_top - f_bot) * dt / a;` |
| `src/fesom_tracer_adv.cpp:2096` | `delttf(k) += vals(k) * (hnode_old - hn_new);` |
| `src/fesom_tracer_adv.cpp:2097` | `if (hn_new > 0.0) vals(k) += delttf(k) / hn_new;` |
| `src/fesom_tracer_diff.cpp:304` | `tr[nz] += bc_surface(n, id, sval, forcing);` |
| `src/fesom_tracer_diff.cpp:320` | `tr[nz] += (top - bot * ar) * dtl;` |
| `src/fesom_tracer_diff.cpp:344` | `trarr[FESOM_NODE3D(n, nz, nl)] += tr[nz];` |
| `src/fesom_tracer_diff.cpp:614` | `tr[nz] += bc_surface_kk(n, id, sval, dt, vcpw,` |
| `src/fesom_tracer_diff.cpp:627` | `tr[nz] += (top - bot * ar) * dtl;` |
| `src/fesom_tracer_diff.cpp:648` | `trv(FESOM_NODE3D(n, nz, nl)) += tr[nz];` |

sites: 215  (generated 2026-09-07 by scripts/m16_accum_ledger.py)
<!-- LEDGER END -->

## 7. Promotion log (carried over from July; the evidence base for the run data is gone — see the M8 memory)

- **2026-07-19 — RULE (SP1): every `MPI_DOUBLE` reduce buffer MUST be `dbl_t`, never `real_t`.** Gate-2
  fleet: the step-diag Allreduce staged `real_t buf_max[16]` under `16, MPI_DOUBLE` ⇒ 2× over-read/write
  = stack smash at npes>1, step-1-clean/step-2-NaN (Z7), partition-dependent. Fix `7e90742` (M8).
  Grep-enforceable; in M16 the same buffer flips to `real_t` **with** `FESOM_MPI_REAL` (class 3).
- **2026-07-19 — PROMOTED: JRA forcing-time machinery → `dbl_t` (first `FieldT<dbl_t>`).** Deterministic
  global T/S NaN at ~9.4 model-hours (dt1800 step 19 / dt900 step 37), scheme- and rank-independent:
  absolute-Julian-day float ulp (6 h) > 3-h records ⇒ `delta_t = 0` ⇒ Inf coefficients. Fix `7247412`.
  Upstream fixed the same (real64 axis + `binarysearch_r8`) independently in #940 — class 1 now.
- **2026-07-22 — PROMOTED: zstar SSH-stiffness cumulative ALE increments → `dbl_t` shadow.** Both 63-yr SP
  arms and two cold-1958 replays died at the same model date 1964-10-04/05 (step 118503±1) across
  backends and rank counts; exponential eta mode at Bering-basin node 118958; root cause = float
  absorption of sub-ulp CSR increments over 118k steps until an eigenmode crossed |λ|=1. Upstream #997
  has the same island (`values_full`) — class 1 now, with divergence (b): our SP restart writes the
  shadow into `stiff_values`.
- **2026-09-08 — the multi-node CUDA intermittent NaN is NOT first-leg: it is the FIRST IN-LOOP FORCING REFRESH.**
  Test job 27294339 (CORE2, 4 GPU nodes, FP64 `e2`, same leg ×3 with `FESOM_MP_NANSCAN=1`): warm-up and leg 1 clean
  (0.0769 / 0.0751 s/step), leg 2 dies at **step 6 = 3 h = the first JRA record boundary** with `[bulk-nan] …
  T_oc=-277.8 … cd=-nan` — the bulk formulae read an unphysical SST at the forcing refresh. Same phase as the SP JRA
  macro bug of 2026-09-07 (deterministic there); here FP64, intermittent (~1 leg in 3 today on 4–16 nodes; the 1-node
  pair never failed; NG5 64 GPUs failed 2/2 FP64 legs), i.e. a device-side race or stale/unsynced device data in the
  `getcoeffld`→bulk path under the M7 speed levers (`FORCEDEV`, `NOFENCE2` are ON in the base arm). The host-alias
  NANSCAN cannot see device state. **Not a precision island — a campaign-wide CUDA flake, M14 saw it on NG5/dars.**
  Bisect round 1 (4 nodes, 5 FP64 legs each): control 1/5 failed (job 27294392; +0 % cost), `FESOM_SPEED_NOFENCE2=0` 0/5
  (27294393; +6 % step time), `FESOM_SPEED_FORCEDEV=0` 0/5 (27294394; +2 %); 16-node ×3 control clean (27294340).
  Round 2 (jobs 27294445/27294446 control, 27294447/27294448 NOFENCE2=0): control 0/10, NOFENCE2=0 1/10 (a step-2
  `pp·App is -nan`). **Totals at 4 nodes: control 1/15, NOFENCE2=0 1/15, FORCEDEV=0 0/5 — lever-independent.** Two
  signatures (step-2 CG garbage/NaN; step-6 `[bulk-nan]` at the forcing refresh), both "device data wrong at a phase
  boundary", rate ~1/15 legs at 4 nodes, higher at 16 nodes (2/3 warm-ups) and NG5 64 GPUs (2/2). Handed to the
  campaign infrastructure track: next tool is `FESOM_HALO_SELFCHECK` (device-vs-host halo verification) on a
  multi-node leg, and a CUDA-aware-MPI/UCX transport A/B (`UCX_TLS` without cuda_ipc). Not a precision island.
  **Lead (2026-09-08, G4 screens):** two more deaths — CORE2 16N SP leg at step 2001 (job 27294510) and NG5 16N SP
  legs at step 2 (27294512), both `CG_kk: pp·App = nan` at iteration 1 — i.e. right after the step-diagnostic print at
  steps 2000 and 1 (`FESOM_PRINT_EVERY=1000` in the ladders; step 1 always prints). The print pulls device fields to the
  host (`sync_host` on device-authoritative Fields). Discriminator submitted: 4-node ×5 legs with `FESOM_PRINT_EVERY=1`
  (jobs 27294800, 27294801) vs `=100000` (27294802). On NG5 at 64 GPUs the "flake" is 4/4 legs (2 FP64 + 2 SP) — deterministic there.
- **2026-09-08 — OBSERVATION (superseded by the entry above): the FIRST leg of a fresh GPU allocation can NaN where the identical leg run
  next succeeds.** Job 27294187 (CORE2 16N, recipe): warm-up FP64 leg `CG_kk: pp·App is -nan` at iteration 1; leg 1
  (same binary, same knobs, same nodes) 300 clean steps. Job 27289163 warm-up also failed (rc 1). The NG5 FP64 deaths
  (27289174) hit legs 1 and 4 too, so it is not only the first leg — but a first-leg-only NaN would point at
  uninitialised DEVICE state on a cold node (a Field read before its first sync) rather than at physics. Test: one
  allocation, the same leg twice, `FESOM_MP_NANSCAN=1` on the first. Until then the discarded warm-up is doing its job.
- **2026-09-08 — NG5 16N GPU pair (job 27289174): FP64 dies, SP lives.** Both FP64 CUDA legs hit the M14-known
  `CG_kk: pp·App is -nan` at step 2 (NG5 A100 at 64 GPUs, wsplit ON, det ON); both SP legs run 300 steps at 0.1904
  s/step, 49 CG iterations. Not a precision finding — the onset is roundoff-seeded (rule 0.41) and SP perturbs the
  seed. Consequence: no FP64 twin ⇒ no SP/DP ratio at NG5 16N; the point is reported SP-only.
- **2026-09-07 — 30-day CORE2 conservation twin (np8, jobs 27289583 FP64 / 27289584 SP):** heat drift gap 0.2 % of the
  FP64 drift (July's number); **salt: SP drift −2.98e-6 vs FP64 −1.75e-6 at day 30, growing linearly** — the salt
  integral at SP carries a genuine drift (float `S` ≈ 35 ± 1.9e-6 psu ulp, the #986 motivation); the salt-anomaly twin
  (jobs 27290582/27290583) **measured: with `FESOM_SALT_ANOMALY=1` the SP salt drift is −1.739e-6 vs FP64 −1.746e-6 (gap 0.4 % of the drift; 71 % without), FP64 on-vs-off 0.1 %, heat unchanged — the #986 island is confirmed on the port at 30 days and the knob joins the SP recipe.** Volume exact in both.
- **2026-09-07 — SP-only memory bug class found by `-Wformat`: `sscanf("%lf", &real_t)`** at two `FESOM_PCSI_EIG` /
  `FESOM_PCSI_MARGIN` override sites (`fesom_ssh.cpp` ~3540/3565) — an 8-byte write into a 4-byte float under SP
  (stack corruption whenever the override is set). Fixed: parse into `double`, assign. Rule for the registry: every
  `scanf`-family `%lf`/`%le` target must be `double`, never `real_t` (the mesh readers already are).
- **2026-09-07 — G3 FINDING: the SP true-residual floor of the SSH solve, and the class-4 promotion it forced.**
  Gate-3 liveness (CORE2 np8, 20 steps, `PRECOND=0`): CUDA job 27289199 and Serial job 27289198 agree — the
  port-only communication-avoiding solvers `pipecg`/`oati` fall back to cg on 20/20 solves, `pcsi` on 19/20
  (`[ssh-solver] !! FALLBACK … residual stalled or grew`), 0 fallbacks in the FP64 oracle; `cg`, `cg2`,
  `cgpipe`, `cgpoly`, SE and every non-solver knob are live. `FESOM_SSH_VERIFY=1` gives the mechanism: at SP
  plain cg's TRUE residual sits at 1.1–2.1× rtol on every solve (solve 1: true 4.82, rec 4.00, rtol 4.34; solve
  20: true 1.67, rec 0.77, rtol 0.81; identical on Serial and CUDA), while in FP64 true ≡ rec (gap 1e-11). The
  requested `soltol = 1e-5` is below what a float `eta` can resolve on CORE2; upstream #940 says exactly this in
  `namelist.dyn` ("soltol … cannot see the float32 residual floor, so the solver will report success it has not
  achieved") and ships it. cg "converges" on its recurrence; the honest solvers (pcsi recurs the TRUE residual)
  stall at the floor and the M10 fallback fires. **Port response (class 4, SP-only, announced + counted):**
  (a) the scalar chains of `pipecg`/`oati`/`pcsi` (dots, α/β/γ/δ recurrences, Chebyshev ω, residual bookkeeping)
  → `dbl_t`/`MPI_DOUBLE` (the plan's prescribed promotion; vectors stay `real_t`; FP64 bitwise by construction);
  (b) `FESOM_SSH_FLOOR` (default 8, 0 = off, compiled out in FP64): a stall with resid < 8×rtol is accepted as
  converged-at-the-float-floor, counted in the `[ssh-wire] AGGREGATE … floor-hits=N` line and shown by
  `m16_knob_signals.sh` (`solver-floor`). This puts the CA solvers on the same footing as cg, no better: the SP
  SSH solution is float-resolution-limited by construction, whichever solver produced it. The G4 twin decides
  whether that resolution is climate-acceptable (it was in July's 63-yr arms, which ran plain cg).
  **Re-test (`e1`, jobs 27289394/27289407):** `pcsi` live (0 fallbacks, ~2× FP64 iterations via the floor rule);
  `pipecg`/`oati` still fall back on 13/20 solves — now the DIVERGENCE exit (recurred residual grows to 1e1–1e4
  at ~130 iterations): the float VECTOR recurrences of the pipelined methods drift from the true residual; a
  wider scalar chain cannot fix that, residual replacement can (E5). At SP the two are declared unusable as built.
- **2026-09-07 — G1 FIRST ATTEMPT FAILED, root-caused, fixed (pending re-run): CORE2 SP died at step 5.**
  Jobs 27288894/27288895 (`d0sp` `8ad967fe`, np8, JRA 1958): both SP arms (with and without the salt
  anomaly) hit `CG_kk: pp·App is -nan` at step 5; DP arms clean for 60 steps. `FESOM_MP_NANSCAN=1`
  (job 27288932): first non-finite at `step-entry(hf)`, and `[bulk-nan]` shows the forcing itself was
  garbage (`ua ~ -3e8 m/s`, `ta ~ -6e7`). Cause: in `fesom_jra55.cpp` the class-5 selector block
  (`FESOM_JRA_POINTSLOPE` / `FESOM_JRA_AT`) was defined BELOW `getcoeffld`, whose `#if
  FESOM_JRA_POINTSLOPE` therefore read an undefined macro (= 0) and built the affine intercept
  `coef_b = d1 − coef_a·t1` (t1 ≈ 2.4e6 days) while the interpolation kernels below the definition
  evaluated `coef_b + dt·coef_a` — consistent in DP (both affine, hence the byte gate never saw it), a
  `coef_a·2.4e6` mismatch in SP from the first in-loop coefficient refresh (step 5). Fix: the block now
  precedes every use (+ a lesson comment at the top of the file); build with `-Wundef` to catch a repeat.
  **Second SP-only finding, same job:** `fesom_ice_fct.cpp` mass-matrix row-sum check uses the Fortran's
  ABSOLUTE 0.1 m² tolerance — 3 orders below float ulp at 1e9 m² — so every open-ocean row printed
  `#### MASS MATRIX PROBLEM` (66k lines/rank). SP-only slack of 32 float ulps × area added (guard-epsilon
  class, §1); the DP check is untouched. Upstream #940 changed neither site.
  **Re-run job 27288954 (`d1sp` 722e3002): G1 PASS** — SP 60 steps rc 0, CG |Δit| vs DP mean 0.43 max 1;
  SP-vs-DP S relL2 1.15e-5 / T 1.4e-4 / u,v 8e-3 at 30 h. **Salt anomaly in SP (CORE2, 30 h):** mean SP-vs-DP
  salt error 3.6e-6 → 0.92e-6 psu (−74 %; upstream −38 %), rms −3.5 %, eta/w slightly better, T unchanged —
  improves, does not regress. DP on-vs-off residual S mean −1.9e-4 psu over 60 steps (~3e-6/step, the upstream
  surface-freshwater class).
- **2026-09-07 — Phase D PORTED: `FESOM_SALT_ANOMALY` (upstream #986 `use_salt_anomaly`).** Every
  `FESOM_TRACER_S].values` reader carries `+ fesom_S_ref_anomaly` (0 unless on): EOS/sw_alpha_beta host+device,
  ocean2ice, rsss/relax_salt (coupling + sss_runoff twins), the surface-BC dilution term `S_ref·water_flux`, KPP Bo,
  the port-only 0.5-psu floor, salt/sss resolvers + the snapshot writer, restart detection (`max S > 20`), the
  step-diagnostic bounds. **S_ref invariance (pi np2, 20 steps, DP):** off / 35 / 10 agree to T ≤ 3.2e-10,
  S ≤ 1.4e-13, eta ≤ 4.2e-10 under linfs and zstar (rounding class — no missed consumer); knob unset ≡ `=0`
  bitwise; knob unset ⇒ all 14 gate-0 configs bitwise vs ref0 (the `+0.0` no-ops hold). SP: off vs on
  differ by T ~1e-4, S ~8e-5 (20 float ulps at 35), eta ~3e-5 — the float rounding class, as expected;
  whether the anomaly REDUCES the SP salt error is the CORE2 question (jobs `core2_g1` / `core2_danom`).
  Port-only divergences: the snapshot writer adds the offset (upstream has no snapshot); the knob carries
  a measurement-only non-35 reference; `density_linear`/3-45 clip/`s<0` screen do not exist here.
- **2026-09-07 — B3b PORTED and measured: the stiffness shadow.** SP pi zstar np2, 200 steps dt 100:
  the real_t-accumulated twin drifts from the dbl_t shadow monotonically, relL2(offdiag) 1.9e-7 → 1.25e-6
  (diag 5e-9 → 7e-8) — the instrument sees the defect at 200 steps that killed July's run at 118k; the
  working copy is `real_t(values_full)` by construction. SP restart round-trip through the shadow is
  bit-identical; DP bit-identical to ref0 (compiled out). CUDA smoke (C2): SP vs DP on pi after 20 steps
  u 5.2e-3, T 2e-4, S 8e-5 — the `hpressure` running-sum suspect (§5) is the first candidate if G3/G4 object.
- **2026-09-07 — M16 baseline.** Registry restructured as the conformance table; class-3 rows flip in
  Phase B; class-2 (CVMix TKE) enters as `dbl_t`; the m14-only code enters as class 4.

- **2026-09-07 — Phase B flips (no promotions).** Class-3 rows flipped to `real_t` in B2 (`ocean_area`,
  mesh Bcasts), B3 (CG scalar chain, `cg_dot`, `ALLREDUCE_SUM`, cg2/pipecg/oati/pcsi/Lanczos/cgpoly
  scalars), B6a (PHC polynomials + weights, det-fill reduces), B6c (step-diag 16/3 and `[cflzmax]`
  reduces). Class 1 kept: forcing time chain (with `coef_a/b` back to WP + `time_t0` dbl_t for the
  SP-only point-slope form, B6b), `[ssh-verify]` retyped `dbl_t` (B3), mean accumulators `dbl_t`
  (B2, new), CVMix TKE `tke_t = dbl_t` (B7, class 2). FP64 byte gates green on pi np1/np2 for every
  slice; CORE2 np8 SLURM gates pending at the time of writing. Give-backs are measured in Gate 2.

*(further entries: date, gate + signature, island added, pinned-pair give-back)*
