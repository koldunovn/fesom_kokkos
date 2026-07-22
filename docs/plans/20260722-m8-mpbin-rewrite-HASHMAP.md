# m8-precision history rewrite 2026-07-22: mp/bin purge — commit hash map

The frozen campaign binaries (mp/bin, ~0.9 GB) were removed from ALL m8-side
history (60 commits since merge-base 1df683b; m7-speed hashes untouched;
content of every rewritten tree verified byte-identical minus mp/bin, tip
diff = 0 lines). Binaries live at /work/ab0995/a270088/port2/mp/bin (38
files + sha256). Rollback anchors: local tag backup/m8-prebin-rewrite +
bundle /work/.../mp/m8-precision-prebin-20260722.bundle.
Hashes cited in older M8 docs/handoffs are the OLD column below.

```
078b8573d -> 3b0cc6bf4   mp/bin: remove frozen campaign binaries from the repo (use
40e233726 -> f37490e40   M8 x dolpung: FESOM_HALO_STAGE transport (real_t port of m
0ff8e3b58 -> 5a0394c7a   M8 session-4 HANDOFF: storm hunt solved end-to-end (probe 
3de1a9d76 -> a320f40c6   M8 scaling figs: apply the m7-s16b MEASURED CG dt-correcti
2f0cb56e5 -> 7fa6e3642   M8 Bp fleet: NOPROTO=1 class-B fallback (m7 precedent: far
7723e7ab1 -> 2ae2eee19   M8 THE FIX (ssh-stiff-ale-acc): zstar stiffness ALE increm
44ab2a73e -> c8d8213ea   M8 scaling: Bp harvest + companion figure (mp_scaling_bp) 
4d9d4dea4 -> aed063e14   M8 storm-hunt s4: REAL first corruption = pressure_bv(rho)
bca4259a9 -> 3932e757b   M8 Bp fleet: one-shot per-rep retry in job_mp_scale_gpu_bp
d92964258 -> a0e4b37e3   M8 storm-hunt s4: VERDICT vel-rhs(uv_rhs) + instrument upg
8244275b7 -> d24ae7e10   M8 s4: class-Bp GPU scaling fleet (user: 'SP with all spee
b3c6413e0 -> d6d2b2038   M8 scaling: dars GPU g16/g32 pairs submitted (26386400-03,
396c811ee -> db8040c04   M8 scaling fig: SYPD as two LINEAR throughput panels (user
66d0cb069 -> c85646158   M8 scaling fig: farc production dt 1200 s (user: 20 min, n
7308b86a6 -> 2bf11e439   M8 scaling fig: shared legend row below panels (user: lege
ef4ec4428 -> 52723d4f6   M8 s4: NG5 c32 scaling-pair failure ROOT-CAUSED + SNAP par
d8585c50f -> 9510d44db   M8 storm-hunt session-4: nanscan momentum/ice-chain probes
57966cbb8 -> 17f6385cb   M8 session-3 HANDOFF: Gates 0-4 PASSED / Gate-5 storm hunt
f15cf924c -> 3e16a979b   M8 scaling fig: m7 house tick style (user catch #2 on figu
bc4a95649 -> 636879906   M8 scaling fig: SYPD convention corrected to the m7 standa
98e27507d -> 818030eab   M8 scaling fig: SYPD panel at working dt (user ask) — per-
dbdc5797a -> f654f9051   M8 scaling: SMALL/MID-MESH pairs added (user: 'we don't ha
bf6b07417 -> e778d916e   M8 scaling fig: NG5 GPU family (scal_ng5_g4..32)
04f82e585 -> a27430abe   M8 NG5 GPU scaling pairs (user: 'they can just stay hangin
fb15f20dd -> d3712301f   M8 scaling fig: NG5 family added (scal_ng5_c4..32, m7 anch
a67b5be55 -> f168166e7   M8 NG5 SP-scaling fleet (user ask): 8 same-day pinned pair
ae60cdb63 -> 16c6b9d77   M8 SP scaling fleet SUBMITTED (user ask: SP scaling figure
e8ad69170 -> 1e2a1fbd9   M8 GATE-5 FAILURE TRIAGE: both arms died DATE-LOCKED at st
2f3fcbee2 -> ac77d25c4   M8 Gate-5 SECOND ARM (user call): 63D-MP = 63-yr FP32 twin
0ef909ac9 -> 032fbf2c0   M8 GATE 5 SUBMITTED (user 'go' 2026-07-20): 63C-MP = 63-yr
df124f491 -> 9a6ab746e   M8 GATE 4 PASSED AT THE M5.23 BAR (job 26365487): 1-yr FP3
479ac66c4 -> 7601e2559   M8 GATE 3k CLOSED ON BOTH BACKENDS: g4_sp (4 ranks x 4 GPU
e68b317c6 -> 0f3238336   M8 Gate-4 job: walltime 45->25 min (user call — fairshare 
ae2888a7e -> 72ac818c5   M8 GATE 3 PASSED IN FULL — zero island promotions (options
b85852d74 -> 84e300a09   M8 Gate-3 harvest wave 1: 3b conservation VERDICT (30-d op
e17275838 -> 5147880be   M8 Gate-3b instrument: FESOM_MP_CONSERV=N env-gated FP64 c
d4852c046 -> 2e82db0ef   M8 session-1 HANDOFF: state through Gate-2/D1-PASSED, froz
c1df19b17 -> ff3b8135d   M8 GATE 2 COMPLETE — D1 PASSED ALL AXES: CORE2 c1 CPU 1.51
115dba3e4 -> 74ae455d8   M8 lessons SP2 refinement: WHY PR-940 could not catch the 
705a9d563 -> 4a140e848   M8 docs: SP_PORTING_LESSONS.md (SP1-SP9, house-numbered tr
7247412e4 -> b971a802b   M8 SP BUG #2 ROOT-CAUSED AND FIXED: JRA forcing time axis 
a1942e124 -> 4d5879d8b   M8 registry: promotion-log rule from the stack-smash — MPI
7e9074255 -> 0dd589e82   M8 CRITICAL FIX: step-diag Allreduce stack smash under SP 
56d0d1919 -> 1844f5733   M8 Gate 1 COMPLETE both backends (SP CUDA pi NaN-free rc=0
8db87d782 -> 9035a6c7d   M8 Task 6: FIRST SINGLE-PRECISION BUILD COMPILES AND RUNS 
640f83eda -> 640f83eda   M8 Task 4 complete + docs sync: last 3 reduce-into-real_t 
228d4e543 -> 228d4e543   M8 slice 2i: io/io_stream/io_config/forcing/jra55/sss_runo
3cdb388d9 -> 3cdb388d9   M8 slice 2h: fesom_halo/halo_device/mpi/partit real_t swee
e3eff9bcc -> e3eff9bcc   M8 Tasks 3+4 (partial): KPP_EPSLN per-precision (FP32 1e-2
2ad5379b5 -> 2ad5379b5   M8 slice 2g: fesom_ssh real_t sweep + CG islands — scalar 
744dbb55c -> 744dbb55c   M8 slice 2f: fesom_eos real_t sweep — 9 sites promoted (sm
b486bff8e -> b486bff8e   M8 slice 2e: fesom_kpp/tke/pp/gm/cvmix_tke real_t sweep — 
c7251f446 -> c7251f446   M8 slice 2d: fesom_ice{,_evp,_maevp,_evpwide,_fct,_thermo,
557ed3d73 -> 557ed3d73   M8 slice 2c: fesom_tracers/tracer_adv/tracer_diff real_t s
053e4a272 -> 053e4a272   M8 slice 2b: fesom_dyn/momentum/ale/ale_dump real_t sweep 
8cc51d3b9 -> 8cc51d3b9   M8 slice 2a: Field=FieldT<real_t> (+types include; test_fi
ee06611de -> ee06611de   M8 Task 1: precision switch scaffolding — fesom_types.h if
caa3076ba -> caa3076ba   M8 plan: user's anomaly/increment proposal assessed + park
21abad062 -> 21abad062   M8 plan: auto-review applied (14 findings) — BLOCKER fixed
816eded03 -> 816eded03   M8 kickoff: mixed-precision campaign plan (FP32 working pr
```
