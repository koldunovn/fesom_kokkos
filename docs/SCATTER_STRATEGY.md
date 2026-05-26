# FESOM2 Kokkos port — edge→entity SCATTER strategy

**What this is.** The port's decision (D22) for how to handle **scatter** kernels — where
many producers (edges) accumulate into shared receivers (nodes/elements) — on the three
backends, and *why* the choice is forced by the validation ladder. First written at **M2.4**
(the first scatters: `momentum_adv_scalar`'s edge→node horizontal advection and
`visc_filt_bidiff`'s two edge→element stages); extended at **M2.6** (FCT flux assembly) and
**M4.3** (ice FCT, ice EVP). Read with `docs/KOKKOS_PORTING_LESSONS.md` (D22, D20, L25/L28) and
the plan's M2.6 note + D5 ladder.

---

## 0. The decision (D22)

> **Edge→entity scatters are ported with `Kokkos::atomic_add`, accumulating in the kernel's
> natural (global) edge order. The Serial backend keeps that order on a single thread → it is
> bit-identical to the C twin's `+=` (the per-kernel `FESOM_KK_VERIFY` gate `max|Δ|==0` holds).
> OpenMP and CUDA reorder the atomics non-deterministically → they are climate-close (≲1e-12 per
> step), which is the D5 ladder's stated acceptance for those backends — NOT bit-identical.**

This is the user-confirmed choice (M2.4): "atomic_add (plan M2.6 default)".

## 1. Why a scatter cannot be bit-identical on BOTH Serial and OpenMP

A scatter is a floating-point **sum reduction** into each receiver, over a producer order the C
code fixes (the global edge loop). Bit-identity to that sum means reproducing its **association
order** exactly.

- **Serial:** a `parallel_for` over edges runs sequentially in increasing index order on one
  thread; `Kokkos::atomic_add` with no contention is a plain load-add-store, so the receiver
  accumulates in *exactly* the C edge order → byte-for-byte the C twin's `+=`. ✔ The
  `-ffp-contract=off` build (D18) guarantees the single add isn't fused with a neighbouring
  multiply, so the arithmetic matches the host C twin's. Proven at M2.4: `visc_filt_bidiff_kk`
  Serial `FESOM_KK_VERIFY=vfilt max|Δ|==0` for all 20 pi steps, and the full-run pi smoke stayed
  `ALL FIELDS BIT-IDENTICAL` to the golden.
- **OpenMP / CUDA:** multiple threads race on the atomic, so the add order is run-dependent →
  ULP-level differences. There is **no** way to enforce the sequential order without serialising
  the loop (which defeats the parallelism). So a scatter is necessarily *not* bit-identical on a
  multi-threaded backend.
- **A gather reformulation does NOT rescue OpenMP bit-identity either:** rewriting the scatter as
  a per-receiver gather (each element sums its 3 edges) changes the **association order** (the
  receiver sums its edges in *its* adjacency order, not the global edge order), so it would break
  the **Serial** `max|Δ|==0` gate. Bit-identity to the C edge order requires the edge order;
  the edge order requires atomics; atomics break multi-thread reproducibility. The trade-off is
  fundamental — you cannot have Serial-bit-identical AND OpenMP-bit-identical for a physics sum
  whose order the C twin fixed.

## 2. Consequence for the gate (a regime change at M2.4)

Through M2.1–M2.3 every ported kernel was a pure map or a *private* gather (each entity owns its
output; `compute_vel_nodes` accumulates a node's surrounding elements into lambda-local scalars),
so **OpenMP happened to be bit-identical too**. From M2.4 on, the scatter kernels
(`visc_filt_bidiff`, `compute_vel_rhs`'s `momentum_adv_scalar`, `ale_vert_vel_linfs` (M2.5), and later
FCT/EVP) are the first where **OpenMP is climate-close, not bit-identical**. This is within spec:

- **Serial** = bit-identical (the bit-identity oracle, the `FESOM_KK_VERIFY` and `diff_snap` gate). Unchanged.
- **OpenMP** = climate-identical, per-step Δ ≲ 1e-12 (the plan's per-backend acceptance / D5).
  `diff_snap.py` (zero-tolerance) on an OpenMP run of a scatter kernel will report a small non-zero
  diff — that is a PASS for OpenMP from M2.4 on, not a regression. (For a *whole-run* OpenMP smoke,
  expect the scatter-seeded ≲1e-12 to advect into a bounded climate-close diff, like CUDA.)
- **CUDA** = climate-close (atomic order + device fma/transcendentals), the M2.1 budget.

## 3. The determinism rule (unchanged, restated)

Any reordering of a physics sum is **GPU/OpenMP-only**, never on Serial. In particular,
**edge-coloring** (a deterministic-per-color scatter that avoids atomics) — if ever adopted for
GPU performance — is **GPU-only**: it reorders the sum, so it would break the Serial `max|Δ|==0`
gate and violate the no-simplification rule. The Serial path always keeps the natural edge order.
We do **not** color at M2.4 (atomics are simplest and correctness-first, D6); revisit only in M5
perf if the atomics are a measured bottleneck.

## 4. Where atomics are used (running list)

| Kernel | Scatter | Receiver | Status |
|---|---|---|---|
| `visc_filt_bidiff_kk` stage 1 (M2.4) | edge → element | `u_b`/`v_b` (Uc/Vc Laplacian) | ✅ atomic_add |
| `visc_filt_bidiff_kk` stage 2 (M2.4) | edge → element | `uv_rhs` | ✅ atomic_add |
| `momentum_adv_scalar_kk` horiz adv (M2.4) | edge → node | `uvnode_rhs` | ✅ atomic_add |
| `ale_vert_vel_linfs_kk` edge flux (M2.5) | edge → node | `w` (+`fer_w` when GM on) | ✅ atomic_add |
| FCT flux assembly (M2.6) | edge → node | tracer flux | ⏳ (same decision) |
| ice FCT / EVP (M4.3) | edge → node | ice flux / stress | ⏳ (same decision) |

(Note: a *gather* — each entity reads its neighbours into private accumulators and writes only its
own slot — is race-free and bit-identical on every backend; it is NOT a scatter and needs no
atomics. `compute_vel_nodes_kk`, `mo_convect_kk`, the `momentum_adv_scalar` vertical-advection
node loop, and the vertex→element average are gathers. Only the genuine edge→shared-entity
accumulations above use atomics.)
