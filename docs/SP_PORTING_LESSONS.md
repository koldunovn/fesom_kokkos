# SP porting lessons (M8 mixed-precision campaign)

Transferable lessons from moving the FESOM2 C++/Kokkos port to FP32 working precision.
House style: numbered, one falsifiable statement each, with the incident that earned it.
Companion to `docs/PRECISION_ISLANDS.md` (the what) — this file is the why/how-to-avoid.
Candidates for `docs/KOKKOS_PORTING_LESSONS.md` at merge-back.

**SP1 — Every `MPI_DOUBLE` reduce buffer must be `dbl_t`, never `real_t`.** Under SP the reduce
over-reads AND over-writes 2× the buffer (stack smash), with layout-lucky, rank-count-dependent
symptoms one step later (Z7 shape). Earned: fesom_main step-diag Allreduce (`7e90742`).
Grep-enforceable invariant: `MPI_DOUBLE` ⇔ `dbl_t`/`double` storage; `FESOM_MPI_REAL` ⇔ `real_t`.

**SP2 — Absolute-calendar quantities are FP32-IMPOSSIBLE whenever record spacing < ulp(magnitude).**
The JRA time axis is absolute Julian days (~2.44e6): float ulp there is 0.25 d = 6 h, coarser than
the 3-h records ⇒ adjacent record times collide ⇒ `delta_t = 0` ⇒ interp coefficients Inf ⇒
`field = rdate·a + b` = NaN globally at the first colliding boundary (~9 model-hours; dt-independent
in model time). Fix class: the ENTIRE time-affine machinery (rdate, nc_time, delta_t, coef_a/b)
joins the calendar `dbl_t` island; interpolate in double, narrow the VALUE once (`7247412`, first
use of per-field `FieldT<dbl_t>`).
⚠️ **The same scheme exists in Fortran FESOM2 (`gen_surface_forcing.F90` sbc_do/getcoeffld) and in
the JAX port (`fesom_jax/jra55.py`, faithful port of the same design).** The Fortran SP branch
(PR-940) actively demoted `nc_time` to WP (`MPI_DOUBLE_PRECISION → MPI_WP` Bcast) — it carries this
landmine latent. The JAX port is safe ONLY because it forces x64. Report upstream before anyone
runs those configs in SP.
**Why PR-940's validation could not catch it (the alignment accident):** their benchmark used
"core2 mesh, core2 forcing" (PR body). CORE-II's fastest records are 6-hourly = 0.25 d — EXACTLY
the float ulp at JD ≈ 2.44e6 (between 2²¹ and 2²²). Integer JD + k·0.25 is exactly representable
there, so CORE-II record times survive float with delta_t = one ulp = 0.25 exact. JRA55-do
(3-hourly = 0.125 d, +1.5 h offsets) is sub-ulp ⇒ collides. The general criterion stands:
FP32-safe iff record spacing ≥ ulp(time magnitude) AND grid-aligned — which is luck, not design;
keep the time axis double.

**SP3 — A correctness verdict is only as long as its RUN LENGTH (the run-length illusion; SP twin
of rule 0.41).** Every "passing" SP probe (np1/np2/np4..np128, both transports) was ≤5 steps; the
failure lived at step 19. Two hours of transport/rank-count forensics chased a phantom. Minimum
honest SP smoke: past the FIRST forcing-record boundary AND ≥2× the model-time of any known
cold-start trigger. Perf legs double as correctness probes: capture rc + FATAL greps in the job
epilogue — SLURM job state COMPLETED says nothing about the run inside.

**SP4 — Guard constants must be audited against BOTH float cliffs.** (a) Additive epsilons must be
NORMAL in FP32 under FTZ (KPP_EPSLN 1e-40 → per-precision 1e-20; PR-940's headline bug class).
(b) Divisor floors must not OVERFLOW: `2.7/u` with a too-tiny floor exceeds FLT_MAX. Check value
against denormal flush AND overflow of every expression it feeds.

**SP5 — NaN forensics: distinguish LOADED from COMPUTED NaN.** `feenableexcept` traps do not fire
for quiet-NaN loads, and init layers (MPI/Kokkos) reset the FP environment — arm traps late, and
when they stay silent, suspect data (or index-from-float OOB) rather than arithmetic. The decisive
tool was the env-gated per-phase NaN scan (`FESOM_MP_NANSCAN`, Serial host-alias reads, die at
first poisoned phase): three runs walked the chain diag→impl-vdiff→bulk→obudget→JRA where gdb saw
nothing.

**SP6 — Global balance corrections are 1-node→global NaN amplifiers.** Any integrate-then-
subtract-everywhere step (salt/heat conservation corrections) converts ONE bad node into a
whole-field NaN in a single step. Signature: entire tracer family NaNs at once while dynamics
stays sane ⇒ hunt one node upstream of a global reduce, not a global bug.

**SP7 — Read sentinel diagnostics literally.** A min/max scan over an all-NaN field never updates
its initializers (NaN comparisons are false): `T[+1e30, -1e30]` (min>max) means ALL-NaN, and the
printed `1000000015047466…e30` is exactly float(1e30) — the decimal expansion itself tells you a
float sentinel passed through.

**SP8 — CUDA has no byte oracle (D22 rediscovered).** CUDA runs are not run-to-run
bit-reproducible (atomics order, ε-class). Per-change CUDA gating = noise-envelope
(`scripts/mp_cuda_gate.py`, pass ≤ 10× same-binary rerun noise, floor 1e-13); Serial is the sole
byte oracle; refresh the noise basis after kernel-set-changing edits.

**SP9 — At FP64 the sweep is invisible; the compiler and the flip are the reviewers.** Gate 0
(bit-identity at FP64) proves refactor inertness but CANNOT see over/under-sweeping — those
surface only at the SP flip (pointer mismatches: 35 compile errors, all boundary seams) and in SP
runs. Budget the flip as its own debugging phase, not a formality.
