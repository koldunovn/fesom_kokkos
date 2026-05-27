# Next session — pin the M5.9 host-reader, make the sync minimal (recover ~6%)

> **✅ RESOLVED (session 20, 2026-05-28).** The host reader is **`uvnode` → `fesom_bulk_compute`** (the
> JRA55 bulk wind-stress formula, surface row, every CORE2 step) — and it is the ONLY one. `bvfreq`,
> `pgf_x/y`, `uv_rhs` are read **only by device kernels**; their M5.9 syncs were placebos (the
> leave-one-out's 0.4 was sync-FENCE chaos sensitivity, not a stale read). Proven by a **NaN-poison
> discriminator** (keep the sync, NaN the host copy → only a host reader sees it): poisoning the three →
> model byte-identical; poisoning `uvnode` → CG NaN-abort. **Fix: dropped the 3 placebo syncs, kept
> `uvnode`'s** (`src/fesom_step.cpp`). Gate PASS, Serial pi np1+np2 bit-identical, **~3.1%** recovered
> (0.4931→0.4777 s/step). Full write-up: `docs/GPU_FIDELITY.md` §M5.9-pin + lessons L49/L50. The text
> below is the original prompt, kept for provenance.

## The one-line goal
M5.9 fixed a CORE2 GPU correctness bug by `sync_host()`-ing **4 device-resident fields after every device
halo**. That is correct but blunt (re-adds ~5 full-field PCIe copies/step = +6.2%). **Find the exact host
operation(s) that read those 4 fields stale, then replace the blanket per-halo sync with a single targeted
`sync_host()` right before that reader** — recovering the 6% while staying correct. The fidelity gate
(below) is the pass/fail oracle.

## Background — what M5.9 found (read `docs/GPU_FIDELITY.md` §M5.6–§M5.9 for the full story)
- The M5.1–M5.8 device-halo flips (GPU-aware MPI) made fields **device-authoritative** and removed their
  OUT-rail `sync_host`. On **CORE2 with active ice**, a remaining **host** op reads 4 of them from the
  **stale host copy**, seeding a chaotic divergence that saturates to **~0.4 vs the Serial oracle** (Serial
  is bit-identical to the C twin = ground truth). The device *exchange itself is byte-perfect*
  (`FESOM_HALO_SELFCHECK` = 0 mismatches) and the halo already fences — so it is stale **host data**, not a
  race or a bad exchange.
- **Invisible on pi** (no ice + idealised → stays ~1e-17) and on Serial (host==device). That is why it
  shipped for ~8 commits. **pi is NOT a sufficient gate for this class** — always use the CORE2 gate below.
- The fix (`6ba27e9`): `sync_host()` after the device halo of **`bvfreq`, `pgf_x`/`pgf_y`, `uvnode`,
  `uv_rhs`**. Leave-one-out proved **all 4 are required** (each, left stale, alone decorrelates to ~0.4).
  Result: CORE2 dev-vs-Serial **0.41 → 1.1e-3** (= the host-halo path / the CUDA climate-close floor).
- **The exact host reader was never pinned** — static analysis ruled out the obvious ones (the ice
  `ocean2ice` reads `dyn->uv`/T/S/`hbar` — none flipped; the `eta_n` map reads `hbar`/`hbar_old`; the
  salinity floor reads owned `S`; the diagnostic print is read-only) and there is NO surviving
  `modify_host()+sync_device()` re-push of a flipped field. So the reader is a non-obvious raw `h()` read.
  **That hunt is this session's job.**

## The 4 sync sites (the current blunt fix — `src/fesom_step.cpp`)
- `:207` `aux->bvfreq_fld.sync_host();`   (substep 1, after the bvfreq device-halo+smoother)
- `:311` `aux->pgf_x_fld.sync_host(); aux->pgf_y_fld.sync_host();`   (substep 2, PGF)
- `:331` `dyn->uvnode_fld.sync_host();`   (substep 3, node velocity)
- `:488` `dyn->uv_rhs_fld.sync_host();`   (substep 6, after `impl_vert_visc`; uv_rhs is also halo'd at ~:435,:459)
Each is grep-able by `M5.9 FIX`. On Serial they are no-ops (host==device).

## Candidate host readers to investigate (the per-step CPU compute that survives on the device path)
1. **Salinity floor** — `src/fesom_step.cpp:892` (host loop; reads/writes owned `S`; the L36/L39 "stays HOST").
2. **`eta_n` map** — `src/fesom_step.cpp:629` (`eta_n = α·hbar + (1-α)·hbar_old`; host).
3. **`compute_ssh_rhs`/CG/`update_vel` host driver** — `src/fesom_ssh.cpp` (M4.2: the CG host loop drives
   device kernels; the host `fesom_compute_ssh_rhs_linfs` at :291 is the verify twin, the *active* one is
   `_kk` at :605 → reads `uv_rhs` on device. Double-check nothing host-side reads `uv_rhs`/`pgf`).
4. **The diagnostic range print** — `src/fesom_main.cpp:1171`+ (reads `S`, `bvfreq`, `pgf`, `Kv`, `Av` on host
   for min/max). Read-only → shouldn't feed back, BUT confirm `FESOM_PRINT_EVERY` isn't forcing a path.
5. **The M4.1 reductions / `integrate_nod_2D`** and the **ice coupling IN rails** (`src/fesom_ice.cpp` ~:445,
   :477 — `ocean2ice`/EVP push T/S/hbar/uv/srfoce via `modify_host()+sync_device()`; verify none of the 4
   flipped fields is among what the ice reads, and that nothing does `modify_host()` on a flipped field).
6. **`fesom_step.cpp` substeps 7–14** between the syncs and the step end — any host touch of `uv_rhs`/`pgf`/
   `uvnode`/`bvfreq` (e.g. a verify-capture left un-gated, an ALE/commit host read).

## Tools already built (all committed) — use these
- **`FESOM_DBG_SYNC`** (`src/fesom_halo_device.hpp:86`, `fesom_dbg_sync`): env comma-list of field ids;
  syncs only those after their halo. The 4 sites are wired (`bvfreq,pgf,uvnode,uvrhs`). Use it to confirm a
  *single* targeted sync (once the reader is found) suffices.
- **`FESOM_HALO_SELFCHECK=1`** (`fesom_halo_device_selfcheck`): device-exchange + host-exchange on the same
  owned data, diff the halo → proves the exchange is byte-correct (it is). Also keeps host fresh (= the
  "all-synced" reference that gives 1e-3).
- **The fidelity GATE** — `./scripts/gpu_fidelity_gate.sh` (+ `scripts/gpu_fidelity_check.py`): build-serial
  CORE2 oracle vs build-cuda CORE2 dist_8 (ICE ACTIVE), asserts dev-vs-Serial ≤ the CUDA-floor ceiling.
  **PASS ~1e-3, FAIL ~1e-1. This is the pass/fail for any change here.** Jobs: `jobs/job_core2_serial_ref`
  (oracle, cached at `/work/.../kokkos_gpu_runs/serref_core2`), `jobs/job_gpu_fidelity_dev` (CUDA leg).
- **Bisect/run harness**: `/work/ab0995/a270088/port2/bisect/run_dev_core2.sh` (CORE2 dev, 20 steps, snap 10
  → diff vs `serref_core2` @ step 20). Step-1 onset jobs: `jobs/job_dbg_{serial,dev,host}_step1`.
- **Per-kernel Serial gate** still applies: `FESOM_KK_VERIFY=<key>` on `build-serial` (`max|Δ|==0`).

## Suggested methodology (pick one; the first is the most direct)
1. **Instrument the read.** Add a SYNCCHECK-style guard that *logs the call site* (not just aborts) when a
   field tagged "device-halo'd this step" is read via `h()`/`h_checked()` on CUDA before a `sync_host`. Tag
   the 4 fields after their halo; build-cuda + SYNCCHECK; run CORE2 1–2 steps; the first logged read names
   the op + file:line. (SYNCCHECK on CUDA was built this session by reconfiguring build-cuda with
   `-DFESOM_KK_SYNCCHECK=ON`; remember to revert.)
2. **Move-the-sync bisect.** For each of the 4, move its `sync_host` *earlier* (right after the producing
   kernel) vs *later* (end of step) and gate-test — narrows where the reader sits in the substep order.
3. **Per-substep `FESOM_DBG_SYNC`** + the leave-one-out style already used (`/work/.../bisect/dbg_*`), but
   now to find the *earliest* point each field can be synced and still PASS.

Once the reader is known: replace the per-halo `sync_host` with **one** `sync_host` immediately before it
(or, if the reader is itself portable to device, move it on-device and drop the sync entirely — the better
outcome). Re-measure perf (target: back toward the 0.464 s/step the broken path had, vs the 0.493 fixed).

## Validation (MANDATORY before commit — the M5.9 lesson)
1. `./scripts/gpu_fidelity_gate.sh` → **PASS** (the CORE2-active-ice CUDA-vs-Serial gate; pi is insufficient).
2. Serial `build-serial` pi np1 + np2 **bit-identical** to golden (`scripts/diff_snap.py` over dirs, np2 with
   `OMPI_MCA_btl_vader_single_copy_mechanism=none`); the touched kernels' `FESOM_KK_VERIFY` `max|Δ|==0`.
3. **Diff ALL output fields, never a subset** (L48 — the I/O-staleness trap).
4. Re-measure CORE2 dist_8 clean timing (host-staged vs device-halo) — report the recovered %.

## Build / run recipe
- CUDA: `source ./env_cuda.sh` (⚠️ `openmpi/4.1.5-nvhpc-24.7`, CUDA-aware — NOT env.sh's 4.1.2 which
  SEGFAULTs on device ptrs); `cmake --build build-cuda --target fesom_port -j 16`.
- Serial: `source ./env.sh`; `cmake --build build-serial -j 16`; login-node MPI override
  `export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` for pi smoke.
- ⚠️ The GPU partition has an **intermittent device-ptr UCX segfault** (`invalid permissions for mapped
  object`); CORE2 dist_8 CUDA runs crash ~randomly — just re-run (the gate FAILs cleanly on a missing
  snapshot). It is NOT a code bug and NOT one bad node (time-correlated flakiness).

## Key files
- `src/fesom_step.cpp` — the 4 sync sites + the ocean substep driver (the host ops 1,2,3,6 above).
- `src/fesom_halo_device.{hpp,cpp}` — the device halo + `fesom_halo_field` dispatch + `fesom_dbg_sync` +
  `fesom_halo_device_selfcheck` (the M5.1 + M5.9 tooling).
- `src/fesom_main.cpp` — the per-step host loop: diagnostic print (:1171), pre-I/O sync (L48), reductions.
- `src/fesom_ssh.cpp`, `src/fesom_ice*.cpp`, `src/fesom_eos.cpp` — the SSH/CG, ice coupling, smoother host ops.
- `docs/GPU_FIDELITY.md` §M5.9 — the bug, the bisect table, the fix, the gate, the measured perf.
- `docs/SYNC_MAP.md` — the M1.5 host/device authority map (what *should* read each field where).
- `scripts/gpu_fidelity_{gate.sh,check.py}`, `jobs/job_{gpu_fidelity_dev,core2_serial_ref}`.

## Commits / refs
- `6ba27e9` M5.9 FIX (the 4 syncs) · `1730be1` the gate · `7d59a32` measured perf (0.493 vs 0.812 = +39%).
- `5b67666` M5.7 KPP · `c2fa25e` M5.8 EVP · `d6b0a1f` M5.1 (clean baseline) · `0b329e3` M5.5a (first broken).
- Bisect data: `/work/ab0995/a270088/port2/bisect/` (`m59_fix_final`=PASS, `dbg_noBVF`/etc=FAIL refs,
  `serref_core2`=the oracle, `devtime_{fix,host}`=the perf runs).
- Memory: `feedback-gpu-fidelity-gate`, `reference-cuda-aware-mpi`. Orthogonal: M3.2 climate run (job
  launched on the fixed binary → `/work/.../kokkos_gpu_runs/m32_cuda_fixed`; compare with
  `scripts/m32_climate_compare.py` when done).

## Lower-priority follow-ups (mention, don't chase)
- M5.4a/b/c per-commit bisect verdicts (crash-blocked; the field-level answer already supersedes — the bug
  entered at M5.4a/uv_rhs; FCT exonerated).
- **Lever C** (the big next perf lever, SEPARATE session + branch): the heavy kernels (FCT ~30%, momentum,
  diffusion) via the memory-layout/coalescing refactor (`fesom_field.hpp` rank-1 → `View<double**>`).
