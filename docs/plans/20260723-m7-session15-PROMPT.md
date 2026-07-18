# M7 next session (session 15) — PROMPT

*Written 2026-07-18 at the end of session 14 (Fable). Branch `m7-speed`. Read in this
order: this file → session-14 findings (`docs/plans/20260722-m7-session14-FINDINGS.md`,
§15 = the exit table, §12 = E.T1, §13 = the E.1 pre-reg) → the JUPITER plan
(`docs/plans/20260722-m7-JUPITER-scaling-PLAN.md`) if that thread activates.*

## 0. RULE DELTAS (on top of every prior rule)

- **0.37 🔴 An env-swapped-library test is only as real as the binary's RPATH allows.**
  The binary carries DT_RPATH with the openmpi-4.1.5 lib dir → LD_LIBRARY_PATH can
  NEVER swap libmpi on it. `readelf -d` the binary FIRST; `ldd` under the leg env
  ALWAYS (the s14 §12 armor caught a would-be fictional HPC-X verdict; what it measured
  instead — env/launcher deltas — became the session headline).
- **0.38 srun's default CPU binding is a measured −5 % @16N / −1.35 % @4N tax** on this
  app (it strangles the UCX/driver progress engine). `SLURM_CPU_BIND=none` releases
  it. Any future absolute anchor must STATE its binding; A/Bs immune as always.
- **0.39 UCX proto-v2 (`UCX_PROTO_ENABLE=y` + GDR RDMA + rail-all) is SCALE-SIGNED:**
  +3.8 % @4N, −5.3 % @16N. `UCX_NET_DEVICES=all` WITHOUT proto-v2 OOM-kills (SIGKILL
  at init) — the package is indivisible on that axis. And it is NOT bit-transparent:
  collective algorithms change ⇒ Allreduce summation order shifts ⇒ solver-class
  divergence (zstar Kv floor 9.537e-02 → 9.869e-02) — fidelity-clean, 4/4 gates PASS.
- 0.31 addendum: the knob pair composed EXACTLY additively at both scales — the
  marginal-decay haircut is a risk term, not an automatic subtraction.

**Binaries** `m7/bin/…`: `h17` = master (unchanged). `cgpoly0` = the knob-pair binary.
**`phst0`/`phst1` = the PHASESTATS diagnostic binaries (phst1 adds ICE_DYN/ICE_ADV
sub-phases), both triple-gated (knob-off byte ✓, knob-ON byte-identical ✓, CUDA
fidelity ✓)** — the per-rank attribution instrument is now permanent machinery.
`FESOM_FORCING_DIR` exists (portability, byte-gated). 🔴 `h3` broken, never use.

## 1. WHERE THE CAMPAIGN IS (post-s14; all pre-registrations scored in the findings)

- **The s14 exit table (findings §15) NAMES every pool.** One mechanism spans the
  imbalance AND much of flat comm: the per-exchange per-partner toll (~30-50 µs/
  partner/exchange; busy-vs-nPart r=+0.96 ice, +0.74 ocean @64 ranks; knob-collapse
  causal for ice/cg). Ice COVER is irrelevant (real-a_ice r=−0.21). Hardware
  exonerated (cross-allocation r=+0.97). CUDA graphs and UCX_TLS/RNDV chains are
  measured-dead. GPUDirect kernel path OPEN (peermem+gdrdrv loaded).
- **E.T1 RESOLVED: the srun env package** `UCX_PROTO_ENABLE=y UCX_IB_GPU_DIRECT_RDMA=yes
  UCX_NET_DEVICES=all SLURM_CPU_BIND=none` = **−10.3 % @16N (0.2169 master)**;
  unbind-only = −1.35 % @4N (0.6297). Gates 4/4 PASS. **Adoption = the USER's call**
  (split shape proposed: unbind everywhere; proto ≥16N; 0.33-spirit caveat on proto's
  non-bit-transparency).
- **Knob pair certified stacking both scales: 4N 0.6164 (7.42×) · 16N 0.2257 (5.44×).**
- Numbers pending harvest (§2): env×knob composition finals.
- 8×@4N arithmetic: from knobbed 0.6164, unbind (−1.35 %) ⇒ ~0.608; E.1 (−3..5) ⇒
  ~0.603-0.605 ⇒ ~7.6×; the last ~20 ms = partner-balance partition and/or TDMA
  kernel work. At 16N the compounded stack likely lands ~0.203-0.208 (≈6×, SYPD ≈3.15).

## 2. IN FLIGHT AT HANDOFF — HARVEST FIRST

1. **26350605 s14_final_4n** (BIN=cgpoly0: ref / unbind / unbind+knobs) — pre-reg:
   unbind+knobs central 0.608 ±0.5 %.
2. **26350606 s14_final_16n** (ref / pkg / pkg+knobs) — pre-reg: pkg+knobs 0.203-0.208
   (sub-additive risk flagged: the package eats latency the knobs also target).
   Harvest checks: announces (L80), md5 provenance, ref vs anchors.

## 3. SESSION-15 SHAPE

1. **Harvest §2 + finalize the s14 findings** (composition rows).
2. **USER DECISIONS to collect:** (a) env-package adoption (split shape? job headers +
   README, or documented-recommended only; optional extra: byte-gate unbind-only to
   check if IT alone is bit-transparent); (b) push approval (commits after `4b68892`
   are local: `7b553ff…` through the s14 tail); (c) partner-balance partition (E.PART2)
   go/no-go for its AUDIT (offline tooling, mesh copies under /work, rule 0.32);
   (d) whether the 1-yr climate leg should run under the env package (the promotion
   arbiter per standing policy).
3. **THE BUILD LEVER (pre-registered, findings §13): E.1a FCT T+S co-pack + E.1c
   Redi/GM co-packs** — central −3..5 ms @4N, −5..8 @16N, byte-class, FORCE_SERIAL
   proof per pair, full ladder. **Rider: the ocean busy spread must shrink ∝ deleted
   ocean events** (the causal test; run the phst1 barrier legs before/after).
4. Then: E.1b smoother ring-ization (second wave) · E.PART2 audit if approved ·
   the JUPITER thread (plan ready; data transfer can start any time).
5. Do-not-chase (measured this session): CUDA graphs, UCX_TLS forcing, RNDV≥256k,
   ice-weighted partitions (ice cover is NOT the driver), LD_PRELOAD HPC-X swap
   (moot — the env package reproduces the win on the stock stack).

## 4. MACHINERY (new this session)

`FESOM_SPEED_PHASESTATS=1` (+`FESOM_HALO_MPI_PROF=1`/`FESOM_HALO_BARRIER=1` legs) ·
`scripts/m7_phasestats_join.py` (attribution analysis of record: log × partition
features × real a_ice) · `scripts/m7_rank_features.py` · `jobs/job_m7_hpcx` (stack
probe, RPATH-aware) · `jobs/job_m7_gdrprobe` · the env package string above ·
`FESOM_FORCING_DIR` · JUPITER plan + `reference-jupiter-plan` memory. Protocol
unchanged: std300, min-of-2, same-alloc, `BIN=` pinning, `-C a100_80` anchors,
pre-register before submission, walltime honesty.
