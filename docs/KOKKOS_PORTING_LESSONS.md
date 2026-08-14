# Kokkos Porting — Decisions & Lessons Log (Fortran → C → Kokkos)

**Why this file exists:** this pipeline (Fortran → C → C++/Kokkos) will be reused to port
**more** FESOM components later. Capture every non-obvious **decision** (with rationale) and
every **lesson** (what bit us / what worked) *while it's fresh*. Append in the same commit that
produces the lesson. Treat an un-logged hard-won lesson as incomplete work.

Complements **`docs/PORTING_LESSONS.md`** — the Fortran→C lessons inherited from the C port
(small-param/loop-bound traps, no-op-at-dt=500, halo loop bounds, tracer stride `nl`, …). Those
still apply; this file adds the **C→Kokkos / CPU↔GPU** layer.

---

## A. Decisions (with rationale)

| # | Decision | Why |
|---|----------|-----|
| D1 | **Strategy A — incremental + DualView**, never a clean rewrite, never simplify physics | The C port's fidelity was won by literal porting + per-step validation; a rewrite re-opens every subtle-param trap |
| D2 | **Fresh repo** importing the C sources; **Kokkos 4.4.01 as a submodule** (`externals/kokkos`, matches ICON) | Self-contained, version-pinned, portable to LUMI without depending on each cluster's spack variant |
| D3 | **No vendor lock** — strictly Kokkos-pure; backend = build config (Serial/OpenMP/CUDA now, HIP later) | Develop on Levante A100, must later run on LUMI MI250x; a `scripts/check_kokkos_purity.sh` grep enforces it |
| D4 | `real_t` stays **double** | Match Fortran WP=8; precision experiments are a single typedef later |
| D5 | **Validation ladder**: Serial = bit-identical to C; OpenMP = climate-identical; CUDA/HIP = climate-close (≈ Fortran↔C noise) | Maps directly to the goal "binary identity when possible, else very close on climate scale" |
| D6 | **LayoutRight everywhere first** (correctness phase), per-space default layout later (perf) | Host mirror == C layout → un-ported host kernels + ported Serial kernels coexist bit-identically; GPU correct-but-uncoalesced is fine first (slow-first accepted) |
| D7 | **M0 C→C++ bridge = `-fpermissive` + gnu++17** (for the 303 `void*→T*` sites) | No codegen change → bit-identity preserved; avoids hand-casting 300 sites up front. **BUT see L1 — only valid for g++, not nvcc** |
| D8 | Keep **`-O3` default (fma=fast)** to match the captured golden; **defer `-ffp-contract=off`** to M2 (re-baseline golden then) | The golden was built fma=fast; to compare bit-for-bit the C++ build must use the same contraction. `-ffp-contract=off` is the *kernel-gate* determinism knob, needed only once kernels exist |
| D9 | **Tests compiled as C++** (`LANGUAGE CXX`), not C | The model objects are now C++ (name mangling); a C test linking a C++ `fesom_calendar.o` would mismatch |
| D10 | First GPU climate target (M3) runs **GM-off** (`FESOM_NO_GMREDI=1`) | Reuses the C port's byte-identity off-switch invariant; shrinks the validation surface before GM-on |
| D11 | **Cast codemod ends the `-fpermissive` bridge** (`scripts/cast_alloc_voidstar.py`): declaration → `(T *)`; member/id assignment → `(decltype(lhs))`; subscript/deref LHS → `(std::remove_reference_t<decltype(lhs)>)`. Run keep-going to surface every straggler at once | Mechanical and **type-exact** — `decltype(lhs)` can never disagree with the LHS, so it can't introduce a wrong-type bug; a pointer cast changes no codegen → Serial stays bit-identical (proven, see L8). Supersedes D7 now that nvcc needs real casts (L1) |
| D12 | **M1.2 mesh migration = `Field`-owns-storage + raw-ptr-is-non-owning-alias.** Add a `fesom::Field`/`IntField` member per persistent array; keep the legacy `real_t*`/`int*` member and re-point it at `field.h()` right after `field.alloc(label,n)`. Replace `calloc/malloc` with `.alloc` (count in **elements**, not bytes); do NOT `free()` the alias | The hot mesh arrays have 28–124 call sites each (`elem_nodes` 124, `nlevels_nod2D` 78). Rewriting them to a `View`/accessor at M1 is huge churn and risk for zero correctness gain (no device compute yet). A stable alias for set-once data keeps every `mesh->X[i*nl+lev]` and the `FESOM_NODE3D/ELEM3D` macros byte-identical → Serial stays bit-identical. The `View`/accessor flip is M5 (only the on-device hot fields, and as a rank change — see L6) |
| D13 | **Replace `memset(m,0,sizeof(*m))` with `*m = fesom_mesh{};`** once the struct holds `Field` members | A `Field` wraps a `Kokkos::DualView` (refcounted, non-trivial). `memset` over it is UB (L13). Value-initialising a temporary and assigning it zeros every POD member **and** properly resets each DualView (assigning an empty DualView drops the old allocation's refcount → frees it). So in `fesom_mesh_free`, the `*m = fesom_mesh{}` *is* the release for every migrated Field — only the not-yet-migrated raw arrays still need explicit `free()` |
| D14 | **np=2 pi run on `dist_2` (login-node, `vader` shmem) is the cheap gate for `scatter_mesh`.** Captured a pre-change np=2 snapshot oracle (`/scratch/a/a270088/pi_np2_ref_m12`); a correct scatter refactor must reproduce it byte-for-byte. ⚠️ **MUST run with `OMPI_MCA_btl_vader_single_copy_mechanism=none`** and the oracle must be captured with the same — the default vader CMA path makes the gather buffer-address-dependent (L18). Robust oracle (CMA-off): `/scratch/a/a270088/pi_np2_ref_m13_nocma`; the old `…_m12` is CMA-tainted, do not use | The np=1 pi smoke does **not** call `scatter_mesh` (single rank → identity), so it can't catch a bug in the MPI mesh redistribution that Wave 2 edits. The global gathered snapshot is rank-count-deterministic, so np2-after == np2-before is a zero-tolerance gate for the data shuffle, runnable in seconds without a compute node |
| D15 | **M1.3 dyn/aux/tracers migration = the verbatim M1.2 pattern, allocate-once/free-once (no scatter realloc cycle, no Field sync helper).** All three structs are stack objects in `fesom_main.cpp` (default-constructed → Field ctors run); `*x = T{}` replaces `memset(x,0,sizeof)` in both `_alloc` (top) and `_free` (D13/L13). **No device sync** is added for these arrays — they are time-evolving state recomputed on host every step (cf. the M1.2 mesh STATE arrays `hnode/hbar`, likewise absent from `mesh_sync_geometry_device`); the step-driver sync discipline for evolving state is M1.5/M2. `tracers->data[*].values/valuesAB/valuesold` keep **stride `nl`** (alloc `N*nl`, `feedback_tracer_stride_nl`) | All time-history updates (`valuesold=values`, `hbar_old=hbar`, `ssh_rhs_old`) are **value copies/memcpy, never pointer swaps** (audited before migrating) — so the `raw = field.h()` alias set once at alloc stays valid for the whole run, exactly like set-once mesh data. A device sync would be dead weight (nothing reads these on device at M1) and would only mask the real M1.5 sync work |
| D16 | **M1.4 forcing + sea-ice migration = the verbatim M1.2/M1.3 pattern, extended to EMBEDDED-BY-VALUE sub-structs and an array-of-structs.** `fesom_forcing` (12 arrays) and the whole `fesom_ice` tree (49 arrays: 19 top-level + `data[3]`×6 + `work`×15 + `thermo`×9, all embedded **by value**) gain a `Field` member per persistent array; the raw ptr is a non-owning alias = `field.h()` set once after `.alloc`. `memset(s,0,sizeof)` → `*s = T{}` in both `_alloc`/`_init` and `_free` (D13). The lone array allocated **outside** its struct's own init — `work.fct_massmatrix`, filled lazily by `fesom_ice_mass_matrix_fill` in a **different TU** (`fesom_ice_fct.cpp`, sized `stiff->nnz`) — gets its `Field` declared in `fesom_ice_work` and `.alloc`'d at that existing call site. **No device sync** at M1 (the EVP/thermo/FCT kernels move to device in M4.3). | Both structs are stack objects (`fesom_main.cpp:347/361`) → default-ctor runs every nested Field ctor and `*ice = fesom_ice{}` recursively releases all 49 nested DualViews (incl. the `data[3]` array elements) in ONE assignment — no per-substruct / per-element free. Audited up front (D15 discipline): no pointer swaps (all time-history updates such as `values_old=values` are element-wise value copies), so the set-once alias stays valid the whole run. The `fct_massmatrix == NULL` lazy-init guard keeps working because the raw alias stays NULL until `.alloc` re-points it. **Completes the M1 persistent-state migration** (mesh+dyn+aux+tracers+forcing+ice = 28+37+61 = 126 arrays); only the gm/kpp/ocean-tradv/ssh per-kernel scratch remains, deferred to M2/M4. |
| D17 | **M1.5 sync cadence = host-authoritative + LAZY device sync (no eager per-step copies); the per-kernel `sync_device(inputs)→modify_device(outputs)→sync_host(before halo/I/O)` brackets are OWNED by each M2/M4 kernel task, not by a central per-step copy.** The set-once geometry stays the one eager exception (pushed once, D-via `mesh_sync_geometry_device`). At M1 the map is "host authoritative" everywhere; M1.5 adds only (a) `h_checked()` at representative halo/I/O host entry points (pointer-identical to the raw alias today) and (b) a `-DFESOM_KK_SYNCCHECK`-only per-step host→device→host round-trip — both compiled out / no-op in production. Deliverable = **`docs/SYNC_MAP.md`** (per-substep currency map mirroring the halo cheat sheet, incl. KPP's 6 internal exchanges, the FCT pipeline, and the mid-step CG host round-trip). | A blanket per-step `sync_device` of all 126 fields would be dead weight at M1 (nothing reads them on device), would have to be deleted + re-placed at the true kernel boundaries in M2 anyway, and would *mask* the real M2 sync work rather than expose it. Lazy sync keeps the data layer at **zero** host↔device traffic (so all backends stay bit-identical) while the map fixes the exact contract M2/M4 implement field-by-field. The SYNCCHECK round-trip makes that contract *executable* (it walks the rails every step and aborts on a violation) without yet moving any compute — see L22. |
| D19 | **M2.1 EOS port = the TEMPLATE for every M2/M4 leaf kernel.** (1) `parallel_for` over the OUTER entity (nodes), the **entire per-entity body incl. the level loops INSIDE the `KOKKOS_LAMBDA`** → each entity owns its column, no cross-entity write → the Serial range is sequential == the C-twin loop (bit-identical) and **even OpenMP is bit-identical** (a pure map, no reduction/atomic; the `≲1e-12` budget only applies to kernels that reduce). (2) Per-column temporaries stay **lambda-local** (per-thread local memory on device; uncoalesced/slow-first accepted). (3) The host C twin stays in-tree **untouched** as the oracle; the device path is a `_kk` twin calling a `KOKKOS_INLINE_FUNCTION` core (the JM-EOS polynomial) — a deliberate **DUPLICATE** of the 40 constants, gated by `FESOM_KK_VERIFY` (a copy typo → gate fails on Serial), collapsed to one definition when the C twin is deleted at M2-close. (4) **Sync-rail split:** the DRIVER (`fesom_step.cpp`) owns the input `modify_host()+sync_device()` and the output `sync_host()` — placed right next to the halos so the sync map literally mirrors the halo map; the KERNEL co-locates `modify_device()` with its writes. (5) The gate (`FESOM_KK_VERIFY=<k>`) runs the C twin into the real `aux`, snapshots the KK result first, diffs, and **RESTORES the KK result** → non-intrusive, so a whole run can carry the verify on and still produce the KK production output (== golden on Serial). | This is the reusable shape for ~15 remaining kernels. Entity-outer/level-inner is the one decomposition that is race-free *and* preserves the C accumulation order (→ Serial `max|Δ|==0`). Keeping the C twin pristine + a gated duplicate core means the gate independently cross-checks both the loop port AND the polynomial, at the cost of a temporary 40-constant copy the gate guards. The driver-owns-rails split keeps `fesom_step.cpp` the single readable place where halo ⇄ sync correspondence lives. |
| D21 | **For a self-contained kernel that does its OWN halo exchanges (KPP, M2.3), the rail ownership splits: the DRIVER owns the IN rail (sync inputs→device) and OUT rail (sync outputs→host); the KERNEL owns the INTERNAL exchange brackets** (`modify_device`→`sync_host`→halo+smooth via `h_checked`→`modify_host`→`sync_device`, co-located with each internal `fesom_exchange_*`). This refines D19's "driver owns the rails" for the internal-exchange case: the driver can't see the mid-kernel exchanges (they're inside `fesom_kpp_mixing_kk`), so the sync⇄halo correspondence for them must live where the exchanges are. The IN rail still goes in the driver because the input structs (`tracers`/`forcing`/`dyn`/`mesh`) are **`const`** in `fesom_timestep` — only the driver has the non-const handles the `modify_host()/sync_device()` need (the EOS/PP rails are there too); `forcing` is `const` even in the driver, so its coherence sync uses a localized `const_cast` (a pure host→device copy, no logical mutation). | KPP's 7 internal exchanges (2 bracket points) are *inside* the kernel; putting their syncs in the driver would split each bracket across two files and lose the halo⇄sync locality D19 prizes. Keeping IN/OUT in the driver preserves the "fesom_step.cpp is the single place the substep's halo map lives" property for the *boundary* of the kernel, while the kernel self-documents its internal coherence. |
| D20 | **A multi-loop kernel with an inter-loop data dependency → one `parallel_for` per loop; the launch boundary IS the barrier** (M2.2 `pp_mixing_kk`). The C `pp_mixing` is three sequential loops where Loop 2 reads `Kv` while it still holds the dimensionless "factor" from Loop 1, and Loop 3 then cubes `Kv` in place — a real read-before-overwrite dependency (the ⚠️ loop-2-before-loop-3 trap). Porting it as **three separate `parallel_for` launches** preserves that ordering for free: each `parallel_for` completes (a full device barrier) before the next starts, so Loop 2 is guaranteed to see factor-`Kv`, never cubed-`Kv`. **Do NOT fuse loops that share a read-then-overwrite on a field**, and do not rely on within-launch ordering across entities. (Contrast the M2.1 EOS template D19, which was independent single-loop maps with no inter-loop hazard.) No manual `Kokkos::fence()` is needed — the implicit per-launch sync on the default execution space suffices. | The whole M2 gate is Serial `max\|Δ\|==0` vs the C twin; an accidental loop fusion or a reordered launch would silently build `Av` from `factor³+K_ver` instead of `factor²` — a physics change that the gate would catch on Serial, but the *design rule* (one launch per dependent loop) prevents it by construction and documents the intent for the next multi-loop port (KPP, FCT). |
| D18 | **M2.1 adopts `-ffp-contract=off` (host compiler) as the standing M2+ determinism knob** (`CMakeLists.txt` `add_compile_options`), deferred here from M0.3. It disables fused-multiply-add *contraction* on the host so a host `a*b+c` and its Kokkos **Serial** port compile to the identical mul+add → the `FESOM_KK_VERIFY` per-kernel gate can demand `max|Δ|==0` on Serial. On the CUDA build it reaches the host portion only (nvcc_wrapper forwards to g++); **device fmad stays on by design** — that device contraction (+ libdevice transcendentals) is exactly the expected first CUDA divergence at M2.1 (climate-close, D5). **Keep the flag even though it is a codegen no-op on the current host (L23)** — it is the explicit, portable standard the gate depends on and future-proofs against an FMA-capable target build. | The per-kernel Serial bit-identity gate (the whole validation model from M2 on) is only trustworthy if the two implementations can't differ by an opportunistic fma in one but not the other. Setting it once, build-wide, removes that variable for every present and future kernel rather than per-file. The golden was re-verified at this setting and is unchanged (L23). |

| D22 | **Edge→entity SCATTERS are ported with `Kokkos::atomic_add` in the kernel's natural (global) edge order (M2.4, the first scatters: `visc_filt_bidiff`, `momentum_adv_scalar`). Serial stays bit-identical; OpenMP/CUDA become climate-close (≲1e-12) for these kernels — the first time OpenMP is NOT bit-identical.** Full rationale in `docs/SCATTER_STRATEGY.md`. The crux: bit-identity to the C twin's `+=` means reproducing its accumulation **association order** (the global edge loop). On Serial, a single-thread `atomic_add` accumulates in exactly that order → byte-identical (`-ffp-contract=off` keeps the lone add unfused, D18). On OpenMP/CUDA the atomics race → ULP reorder → climate-close (the D5 ladder's stated OpenMP/GPU acceptance). A **gather reformulation cannot save OpenMP bit-identity** — summing each receiver's edges in *receiver-adjacency* order changes the association vs the global edge order, which would break the **Serial** gate. So Serial-bit-identical ⟺ edge order ⟺ atomics ⟺ OpenMP-not-bit-identical: the trade-off is fundamental, not a tuning choice. Edge-**coloring** (if ever needed for GPU perf, M5) is **GPU-only** for the same reason (it reorders the sum). User-confirmed at M2.4 ("atomic_add, plan M2.6 default"); applies forward to FCT (M2.6) and ice FCT/EVP (M4.3). | The whole validation model is Serial `max\|Δ\|==0`; that gate is only meetable for a scatter by preserving the C edge order, which forces single-thread-ordered atomics. Accepting OpenMP climate-close (already the D5 acceptance, just not yet exercised) is the price; the alternative (keep scatters on host) would leave `visc_filt_bidiff` essentially un-ported and inject mid-kernel host round-trips. Correctness-first (D6): atomics now, color only if measured-slow in M5. |

## B. Lessons (what bit us / what worked)

- **L1 — `-fpermissive` is g++-only; nvcc does NOT honor it.** The host C→C++ bridge (D7) gets
  Serial/OpenMP compiling instantly, but nvcc's front-end (EDG/cudafe) rejects `void*→T*`
  regardless of `-fpermissive` (a host-g++ flag). **Any file nvcc compiles needs *real* casts.**
  Cast pattern that handles all 303 sites without inferring types:
  - assignment `lhs = alloc(...)` → `lhs = (decltype(lhs))alloc(...)`
  - declaration `T *x = alloc(...)` → `T *x = (T*)alloc(...)` (decltype can't take a declaration)
  Casts don't change codegen, so Serial stays bit-identical after applying them.
  **RESOLVED 2026-05-25:** codemod `scripts/cast_alloc_voidstar.py` cast all **305** sites
  (47→70 decl, 231 member/id, 4 subscript/deref); `-fpermissive` removed; Serial pi smoke still
  `ALL FIELDS BIT-IDENTICAL`; the **CUDA full model now compiles under nvcc with zero errors**
  (closes the M0 blocker). See L8–L11 for what the removal surfaced and the codemod gotchas.
- **L2 — `goto`-crosses-initialization is a hard C++ error `-fpermissive` won't fix.** Wrap the
  skipped region in a `{ }` block so the crossed variable's scope is fully contained between the
  `goto` and its label (behavior-preserving). (Hit once in `fesom_main.cpp`.)
- **L3 — The C→C++ flip alone is numerically inert.** At the same compiler/flags (gcc `-O3`),
  the C++ build is **bit-for-bit identical** to the C build (verified on the pi smoke, all
  fields). This is the foundation that makes the incremental approach safe — the Serial backend
  is a perfect oracle.
- **L4 — Login-node MPI**: UCX/IB transports are unavailable off the compute nodes; `env.sh`'s
  `OMPI_MCA_pml=ucx` fails with "No components in the pml framework". For a single-rank smoke:
  `export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` and `unset` the UCX/HCOLL knobs.
- **L5 — `.gitignore` `build*/`, not specific names.** A too-specific ignore (`build-cuda/` but
  not `build-smoke-cuda/`) let ~1000 CMake artifact files into a commit; `git rm --cached` + amend
  fixed it. Use the glob.
- **L6 — DualView mirror-layout (anticipated, M5):** pinning `LayoutRight` keeps the host mirror
  byte-matching the C layout. The later flip to space-default layout for the interleaved 2-comp
  (`uv` via `FESOM_ELEMVEC`) and 3-D node fields is a **rank change** (1-D→2-D/3-D View) plus
  making the index macros layout-agnostic accessors — *not* a one-line layout swap. (LayoutLeft ==
  LayoutRight for a rank-1 View, so a flat flip buys no coalescing.)
- **L7 — Gates that work**: `diff_snap.py <pre> <post>` (zero-tolerance `np.array_equal`) is the
  whole-run bit-identity gate; the in-binary `FESOM_KK_VERIFY` C-twin-vs-Kokkos diff (M2) is the
  per-kernel gate. Capture a **golden** from the unmodified C build first (with provenance:
  source SHA + exact run cmd).

- **L8 — `-fpermissive` was a broader amnesty than "just the void* casts" (D7's blind spot).**
  Removing it surfaced not only the 305 `void*→T*` sites but **12 implicit `int→enum`
  conversions** (`int → fesom_halo_kind` in `fesom_mesh.cpp` — raw `2 /* ELEM2D */` / `4 /*
  ELEM2D_FULL */` where *every other caller in the tree* used the enum name). C allows int→enum
  implicitly; C++ forbids it; `-fpermissive` had hidden it. Fix = the named constants (same
  integer value ⇒ codegen-neutral ⇒ still bit-identical). **Lesson: when you drop `-fpermissive`,
  budget for several conversion classes surfacing, not just the one you expected** — and let the
  compiler enumerate them (`cmake --build … -- -k 0` to collect every straggler in one pass).
- **L9 — decltype-cast codemod must bound the LHS to the `=`'s own line.** A naive "scan left to
  the previous `;`/`{`/`}` for the assignment LHS" silently swallows a preceding `/* comment */`
  line (a comment has no statement terminator), emitting `(decltype(/* comment */ lhs))`. Stop the
  backward LHS scan at the **newline**. (The lone multi-line `realloc` keeps its `type *var =` on a
  single line, so confining the LHS to that line is safe.) This bug mis-classified 23 declarations
  as member-assignments before it was caught by auditing the diff (every changed line must contain
  an alloc keyword; nothing else may change).
- **L10 — subscript / deref LHS needs `remove_reference_t`, not bare `decltype`.**
  `decltype(arr[i])` and `decltype(*p)` are *lvalue references* (`T&`), and a C-style cast to a
  reference can't bind the `void*` **rvalue** the allocator returns → hard error. Use
  `(std::remove_reference_t<decltype(lhs)>)` (and `#include <type_traits>`). Bare `decltype` is
  correct **only** for id-expressions (`x`) and class-member access (`m->z`), which yield the
  declared, non-reference type. Here that was 4 sites (`s->accum[v]`, `io->owned_vars[p]`, two
  `*buf`); the other 301 used plain `decltype`/declaration casts. The `decltype`/`remove_reference`
  approach is **type-exact** — the cast is *by construction* the LHS's own type, so it can never
  introduce a wrong-type conversion (unlike a hand-written or inferred type).
- **L11 — nvcc's front-end (EDG) accepted the host code with no *extra* fixes beyond g++'s.** Once
  the casts (L1) and the int→enum fix (L8) made the Serial (g++, no `-fpermissive`) build clean,
  the CUDA build compiled all 36 model TUs through nvcc with only benign `#177-D`/`#550-D`
  unused-variable warnings. At M0 there are no device kernels, so nvcc compiles all-host code fast
  (~17 s, `-j16`); EDG strictness ⊇ g++-no-permissive turned out to add nothing here. (Device-code
  divergences are expected to begin at M2.1, not M0.)

- **L12 — a stray `*/` in a C block comment closes it early → a cascade of *misleading* STL
  errors.** Writing `sync_*/modify_*` inside a `/* … */` doc-comment in `fesom_field.hpp` contained
  the token `*/`, which terminated the comment; the trailing prose then parsed as code and the
  compiler vomited ~12 errors deep inside `<bits/stl_*.h>` / Kokkos-Tools (`_M_max_size`,
  `max_size` not a member, placement-`operator new` not matching) — nothing pointed at the comment.
  Kokkos' own `sync_*` / `modify_*` naming convention makes this easy to hit. **Use `//` line
  comments for doc-blocks that mention `sync_*`/`modify_*`/pointer-glob patterns**, or space them
  (`sync_* / modify_*`). When STL/standard-library errors appear in a TU you didn't change
  structurally, suspect a comment/macro lexing problem first, and compile the TU alone to see the
  *first* diagnostic (the real one).

- **L13 — `memset(struct,0,sizeof)` is undefined behaviour once the struct holds a `Field`
  (DualView) member.** `fesom_mesh_init`/`_free` used `memset(m,0,sizeof(*m))`; a DualView is a
  non-trivial type (refcounted handle), so memset corrupts its internal state and skips the
  refcount bookkeeping (→ leak or, on re-alloc over a memset-zeroed handle, mismatched frees). Fix
  = `*m = fesom_mesh{};` (D13): well-defined, zeros the PODs, and assigns an empty DualView to each
  Field (releasing any prior allocation). Watch for this in **every** struct that gains a `Field` —
  `memset`/`calloc`/`memcpy` of a now-non-POD aggregate is the trap. (The struct must also be
  constructed, not `malloc`'d: `fesom_mesh mesh;` default-constructs the Fields — verified the only
  instance in `fesom_main.cpp` is a stack object, and nothing `malloc`s/`memcpy`s the struct.)
- **L14 — host writes through the raw alias bypass the DualView modify-flags, so `sync_device()`
  alone is a silent no-op.** After `field.alloc()`, legacy host code fills the array via the raw
  pointer (`m->area[i]=…`), which the DualView cannot see. `DualView::sync_device()` only deep-copies
  when the host-modified flag is set, so without a preceding `modify_host()` the device copy stays at
  the alloc-time zeros — a latent stale-device bug that would first bite an M2 device kernel, not the
  M1 gate (no device reads yet). **Rule: after a legacy host kernel writes a `Field` via `.h()`, call
  `modify_host()` before `sync_device()`.** The M1.2 set-once geometry does this once at the end of
  `compute_metrics` (`mesh_sync_geometry_device`: `modify_host(); sync_device();` per field). On
  Serial/OpenMP host==device so it's a no-op; on CUDA it is the only (bitwise-exact) device op at M1.
- **L15 — a missing output directory surfaces as a NetCDF "Permission denied", not "No such file".**
  The pi-smoke recipe runs `fesom_port … /tmp/pi_check …`; if `/tmp/pi_check` doesn't exist (fresh
  login, ephemeral `/tmp`), the model gets through every setup/sanity line and only dies at the first
  snapshot write with `NetCDF error: Permission denied` (`fesom_io.cpp:394`) — a misleading errno for
  a missing dir. **Recipe fix: `mkdir -p <out_dir>` before the run** (now folded into the gate).

- **L16 — an array with a free+realloc lifecycle (MPI scatter) must re-`alloc()` its Field at EVERY
  reassignment, not just the final one.** `scatter_mesh` reassigns `m->coord_nod2D` three times:
  `read_*` (rank 0 / npes==1), the rank!=0 broadcast receive buffer, and the per-rank local-slice
  swap. To keep the `Field` the sole owner across all of it, each `free(m->X); m->X = malloc(...)`
  became `m->X_fld.alloc(...); m->X = m->X_fld.h();` (`.alloc` releases the prior allocation via
  refcounting -- no `free` needed). Two traps: (a) the local-slice swap fills a *raw* `new_*` buffer
  from the global Field, so you can't just alias `.h()` to it -- re-`alloc()` the Field to the local
  extent (releasing the global storage the slice was read from, already copied out),
  `memcpy(field.h(), new_X, ...)`, then `free(new_X)`. (b) a helper taking `T** buf` that does
  `free(*buf); *buf = malloc(...)` (here `bcast_real_array`) cannot operate on a `Field` -- inline it
  as `if (rank!=0) fld.alloc(); MPI_Bcast(fld.h(),...)`. Only the np=2 `dist_2` gate (D14) exercises
  any of this -- np=1 never enters `scatter_mesh`.

- **L17 — `nvcc_wrapper` rejects `.c`-extension sources even with CMake `LANGUAGE CXX`.** The three
  C unit tests (`tests/test_*.c`, compiled as C++ via `set_source_files_properties(... LANGUAGE CXX)`,
  per D9) build fine under g++ but fail under the CUDA build's `nvcc_wrapper` with `nvcc fatal: No
  input files specified` — the wrapper keys off the `.c` file extension, not the CMake LANGUAGE
  property, and drops the input. This is **pre-existing** (independent of M1.2 — the test sources and
  their CMake targets were untouched) and only surfaced because a *full* `cmake --build build-cuda`
  (vs `--target fesom_port`) tries to build them. Fix: guard those three targets with
  `if(NOT Kokkos_ENABLE_CUDA)` — they're pure host logic (no Kokkos, backend-independent), fully
  covered by the Serial/OpenMP `ctest`; the model (`fesom_port`) and `test_field` build on every
  backend. **The model itself compiled cleanly under nvcc** — Wave-1's `*m = fesom_mesh{}` and the
  `Field`-in-struct/`DualView` patterns gave nvcc no trouble (consistent with L11). Also: in a bash
  one-liner, `echo "...exit=$?"` after a `$(date)` reports **date's** exit, not the build's — verify
  build success from the log/`Built target`, not such an `echo`.

- **L18 — the login-node `vader` CMA single-copy path makes `MPI_Gatherv` buffer-address-dependent: identical sends → different `recv`.** M1.3 (dyn/aux/tracers → Field) passed np=1 bit-identical but the np=2 pi gate diverged (Av/Kv ~0.3, widespread, from step 1). It was **NOT** an M1.3 bug: the per-step OWNED state is byte-identical (proven by dumping `aux->Av/Kv[0..myDim*nl)` per rank — `cmp` identical on both ranks), and the snapshot I/O gather (`fesom_io.cpp`) is untouched. Dumping the gather internals showed the per-rank SEND buffers (`local`) byte-identical AND the gather plan (`all_e/displ_e/gathered_myList_e`) identical, yet `MPI_Gatherv`'s `recv` differed between the calloc and Field builds. The trigger: the bigger Field structs shift heap/buffer addresses, and **OpenMPI's `vader` shared-memory BTL CMA path (`process_vm_readv`) returns address-dependent garbage** on this login node. **Fix/avoid: `export OMPI_MCA_btl_vader_single_copy_mechanism=none`** — with it, calloc(M1.2) and Field(M1.3) np=2 are **byte-identical**. Consequences: (1) the np=2 gate (D14) and its oracle MUST set this; the old `pi_np2_ref_m12` was CMA-tainted (regenerated as `…_m13_nocma`). (2) It's deterministic *within* a build (run-to-run identical) but not *across* builds — so it masquerades as a code regression. (3) **Debugging method that nailed it: bisect compute-vs-I/O by dumping the OWNED state right after the producing kernel; if owned is byte-identical the divergence is in the gather/transport, not the port.** (4) The model COMPUTE never uses `vader` CMA for results (halo exchange is pack→Isend/Irecv→unpack, and that path was fine); only the diagnostic `MPI_Gatherv` snapshot gather hit it.
- **L19 — `scripts/diff_snap.py` takes two DIRECTORIES, not files; given a file it silently prints `No snap_*.nc in <path>` and exits "clean".** A per-step bisect loop that called `diff_snap.py dirA/snap_000001.nc dirB/snap_000001.nc` reported every step "identical" — a **false negative** that sent the L18 hunt down a multi-hour wrong path (chasing an "I/O-mean" red herring). **Always diff DIRECTORIES** (`diff_snap.py dirA dirB`, which iterates matching `snap_*.nc`); to compare one step, put the two `snap_NNNNNN.nc` in their own dirs. When a comparator reports "identical" unexpectedly, verify it actually compared something (it prints `Comparing N file(s)`).
- **L20 — embedded-by-value sub-structs and arrays-of-structs migrate transparently under the Field-alias pattern; zero special handling.** `fesom_ice` embeds `fesom_ice_data data[3]` + `fesom_ice_work work` + `fesom_ice_thermo thermo` **by value**. Adding `Field` members inside those nested types needed no extra ceremony: value-initialising the enclosing stack object (`*ice = fesom_ice{}`) recurses through the C array and the nested aggregates — every nested DualView default-ctor runs (alloc-reset) and every nested DualView is released (free) — so the single top-level assignment is BOTH the reset (in `_init`) AND the full free (in `_free`) for all 49 nested Fields. M1.4 was **bit-identical on the FIRST gate run** (Serial np=1 == golden, np=2 == `…m13_nocma` oracle, ctest 4/4, CUDA np=1 (A100) == golden), confirming the D15 "audit pointer-swaps / stack-vs-malloc BEFORE editing" discipline keeps these mechanical struct migrations a clean single pass. The one subtlety: a persistent array can be allocated in a **different TU** and **lazily** — `work.fct_massmatrix` (`fesom_ice_fct.cpp:fesom_ice_mass_matrix_fill`, guarded `if (== NULL)`, sized `stiff->nnz`, called once from `main`); migrating it = declare the Field in the substruct + convert the `calloc`→`.alloc`+alias at that foreign call site, and the `== NULL` guard survives because the raw alias is NULL until `.alloc` re-points it. (Also: env-gated *array-data* memsets through the raw alias — e.g. `FESOM_NO_WIND`'s `memset(forcing.stress_surf,0,…)` in `fesom_main.cpp` — stay UNCHANGED; they zero `double`s via the host pointer, never touching Field internals, unlike the whole-struct `memset` which is the D13 trap.)

- **L21 — `assert()` is neutered by `NDEBUG`, which CMake's Release/RelWithDebInfo configs define — so an assert-based guard silently vanishes in exactly the build that matters.** The `FESOM_KK_SYNCCHECK` coherence guard in `h_checked()` originally used `assert(auth_ != Device)`. But the SYNCCHECK build *must* be `-O3` Release to compare bit-for-bit against the fma=fast golden (a Debug `-O0` build would diverge on its own, defeating the gate); Release adds `-DNDEBUG`, which makes `assert()` a no-op → the guard would compile to nothing and "pass" vacuously. Fix: a self-contained `if (cond) { fprintf(stderr,…); abort(); }` gated **only** on `FESOM_KK_SYNCCHECK`, never on `NDEBUG`. **Rule: a diagnostic that has to fire in a Release/optimised build must not be built on `assert()`** — use an explicit abort (or `Kokkos::abort`) behind its own macro. (Zero-cost when the macro is off: the block is `#ifdef`-elided.)
- **L22 — the M1.5 sync rails are bit-identical-to-production *by construction*, which is what makes them safe to add before any device compute exists.** Two independent reasons, both worth internalising for future "instrument-without-perturbing" work: (1) **The per-step round-trip is a bitwise no-op on the host bytes.** `modify_host(); sync_device(); modify_device(); sync_host();` leaves host memory at exactly its pre-call value — on Serial/OpenMP host==device so all four are no-ops, and on CUDA each leg is a byte `deep_copy` (H→D then D→H), so the host round-trips to itself. It is `#ifdef FESOM_KK_SYNCCHECK`-only, so the **production build is byte-for-byte the non-SYNCCHECK build** (verified: pi smoke np=1+np=2 ALL FIELDS BIT-IDENTICAL with the macro on). (2) **`h_checked()` routing is pointer-identical today**: the migrated raw alias *is* `field.h()`, and `h_checked()` returns `h()` after the (compiled-out-by-default) check, so `fesom_exchange_nod3D(aux->bvfreq_fld.h_checked(),…)` passes the identical pointer `aux->bvfreq` did. **Why no guard fires at M1 — and why that is the proof, not a dud:** `modify_device()` is the *only* call that sets a Field `Device`-authoritative, and it appears solely inside the round-trip, immediately followed by `sync_host()` (→ `Synced`). So outside that 2-line window every Field is `Host`/`Synced`, and host writes through the raw alias never change the tag (L14). Therefore every halo/I/O `h_checked()` necessarily sees a non-`Device` field → the assert *cannot* fire at M1, which **is** the executable proof that M1 is uniformly host-authoritative (the SYNC_MAP's central claim). From M2, a real device kernel's `modify_device` + a forgotten `sync_host` flips one field to `Device` at a halo/I/O read and the guard bites. (Also: bouncing an *unallocated* Field is safe — a default-constructed `DualView` has 0-extent views, so `modify_*`/`sync_*` are harmless no-ops; this lets the forcing round-trip cover optional fields like `virtual_salt` without a presence check.)

- **L23 — `-ffp-contract=off` was a codegen *no-op* on Levante's baseline x86-64 host: the golden didn't change.** Adopting it for M2.1 (D18), the expectation was a tiny ULP shift in the pi golden (fma removed), requiring a golden re-capture. Instead the `-ffp-contract=off` Serial build was **byte-identical** to the fma=fast golden (np=1 *and* np=2, all fields, zero-tolerance `diff_snap.py`; `ctest` 4/4). Reason: the build targets **baseline x86-64 with no `-march=native`/`-mfma`**, and without an FMA instruction in the target ISA the compiler has nothing to contract into — so `off` vs the default emits the same code. Two takeaways: (1) **the host Serial path was already fma-free**, which retroactively explains why every M0/M1 Serial==C gate held at the default; the real fma/transcendental divergence at M2.1 will come from the **device** (CUDA `--fmad=true` default + libdevice `sqrt`/poly), not the host. (2) **Set the flag anyway** — it costs nothing, makes the determinism guarantee explicit and portable, and is the one knob that would bite silently if a future build added `-march=native` (then the host *would* start fusing and a kernel gate could break with no source change). Don't skip a correctness knob just because it is currently inert on one toolchain.

- **L24 — M2.1 (first device kernels) landed bit-identical on the FIRST Serial gate run; the M1 groundwork paid off.** All 20 pi steps `FESOM_KK_VERIFY=eos max|Δ|==0` for every output (density/hpressure/bvfreq/dbsfc/sw_α/sw_β + MLD1), the full-run `diff_snap.py` was `ALL FIELDS BIT-IDENTICAL` to the golden (np=1 **and** np=2-CMA-off), `ctest` 4/4, and the **SYNCCHECK build ran clean** (no `h_checked()` abort) and bit-identical. Four things made the device port safe and worth internalising:
  - **The `*/`-in-comment trap (L12) struck a THIRD time** — a doc comment line `temporaries (bulk_*/rhopot/rho[64])` contains `*/`, which closed the block comment early; the prose after it parsed as code → a cascade of *misleading* `error: extended character → not valid in an identifier` / `stray '\`'` diagnostics pointing at the comment text, not the real cause. **The glob/pointer patterns `X_*/Y` in a `/* */` block are the recurring culprit** (here a list of scratch arrays). Reword to drop the literal `*/` (`bulk_0, bulk_pz, …` instead of `bulk_*/…`). When weird unicode/stray-token errors erupt from inside a comment, grep the new comments for `*/` before anything else.
  - **The SYNCCHECK guard did REAL work for the first time.** Through M1 it could never fire (L22: nothing set a Field `Device`-authoritative outside the 2-line round-trip). Now the EOS kernel's `modify_device()` genuinely flips the 7 outputs to `Device`, and the driver's `sync_host()` flips them back to `Synced` before the halo `h_checked()` reads — so on Serial-SYNCCHECK the tag actually transitions `Device→Synced` every step and the guard *would* abort if the output sync were dropped. It ran clean → the rail is correct. This is the executable payoff of building the guard at M1.5 before any compute moved to the device.
  - **`hnode` is a time-evolving mesh INPUT, not set-once geometry — easy to miss.** `pressure_bv` reads `mesh->hnode` (the hpressure accumulation), and `hnode` is *not* in the one-shot `mesh_sync_geometry_device` push (it changes each step via host ALE). So the EOS input rail must `modify_host()+sync_device()` it every step, alongside T/S — whereas `Z`/`ulevels`/`nlevels` are set-once and already device-current. The SYNC_MAP §1 "reads … mesh geometry" wording hid this; the actual read set (verified in the kernel body) is the source of truth. **For each kernel, derive the device-input set from the kernel body, and classify each mesh input as set-once (one-shot push) vs evolving (per-step sync).**
  - **The expected CUDA divergence is DEVICE-only.** On the host backend `Kokkos::sqrt`==libm `sqrt` (IEEE correctly-rounded) and `-ffp-contract=off` removes host fma, so Serial/OpenMP are bit-identical; the fma + libdevice `sqrt`/polynomial ULPs that break bit-identity live only in the CUDA device code path (→ climate-close, by design D5). So a kernel can be *fully* validated on Serial (`max|Δ|==0`) and the GPU acceptance is a separate climate-close budget — the two gates test different things.

- **L25 — a `FESOM_KK_VERIFY` key match must guard against substring-superset collisions: `"pp"` ⊂ `"kpp"`.** M2.1's EOS gate detects its key with a plain `strstr(e,"eos")`, which is fine because no other key contains "eos". But the M2.2 PP key `"pp"` is a substring of the M2.3 KPP key `"kpp"` — a plain `strstr(e,"pp")` would make `FESOM_KK_VERIFY=kpp` *also* fire the PP gate (a latent cross-trigger that would only surface at M2.3). Fix: scan for `"pp"` **not immediately preceded by `'k'`** (`for (q=strstr(e,"pp"); q; q=strstr(q+2,"pp")) if (q==e||q[-1]!='k') {found}`), so `kpp`→no, `pp`/`eos,pp`/`kpp,pp`→yes. **Rule: when verify keys can be substrings of one another, match the token, not a bare substring** — and fix it when you ADD the colliding key, not when it bites. (Captured now because the next task literally adds `kpp`.)
- **L26 — a read-modify-write kernel needs its gate to capture the PRE-kernel input; the EOS "recompute from intact inputs" pattern doesn't apply.** `fesom_eos_verify` (D19) works by running the C twin into `aux` and diffing, because the EOS kernel's inputs (T/S) are *never modified by the kernel* — the C twin recomputes the same outputs from intact state. `mo_convect` breaks that assumption: it RAISES `Kv`/`Av` **in place** (`Kv=max(Kv,imix) where N²<0`), so after the device kernel + `sync_host` the host no longer holds the kernel's input — the C-twin oracle has nothing to recompute from. Fix: the **driver snapshots the pre-kernel `Kv`/`Av`** (host-current, just before the IN rail) and passes them to `fesom_mo_convect_verify`, which restores the input, runs the C twin on it, diffs vs the saved KK output, then restores the KK output (non-intrusive, as EOS). `compute_vel_nodes`/`pp_mixing` stayed EOS-style (their outputs are full overwrites from un-modified inputs — `pp_mixing` Loops 1&2 are assignments, so the result is independent of prior `Kv`/`Av`). **Rule: classify each ported kernel's outputs as full-overwrite (EOS-style gate) vs read-modify-write (capture-before gate).**
- **L27 — a field can hand off device→HOST→device WITHIN one step; the second device read still needs `modify_host()+sync_device()`.** `aux->bvfreq` is *produced by a device kernel* (`pressure_bv_kk`, substep 1), then `sync_host()`'d, halo-exchanged, and **overwritten on the host by `smooth_nod3D`** (the N² horizontal smoother, raw alias), then read by *another device kernel* (`mo_convect_kk`, substep 3). Because a device kernel produced it first, it is tempting to assume `bvfreq` is already device-current at substep 3 — but the host `smooth_nod3D` write is invisible to the DualView (L14), so `mo_convect`'s rail MUST `modify_host()+sync_device()` it, not a bare `sync_device()` (which would feed the kernel the *pre-smooth* device bytes). This is the second concrete instance of L24's "derive the device-input set and classify host-vs-device authority **from the kernel body**, not from a one-line description" — the first was `hnode`; here the subtlety is that the producer is itself a device kernel. M2.2 (`compute_vel_nodes_kk`, `pp_mixing_kk`, `mo_convect_kk`) then landed **bit-identical on the first Serial gate run** — all 20 pi steps `FESOM_KK_VERIFY=pp max|Δ|==0` for every output (uvnode/Kv/Av) on both the default KPP path (compute_vel_nodes+mo_convect) and the PP path (+pp_mixing), full-run `diff_snap.py` ALL FIELDS BIT-IDENTICAL (np=1 & np=2-CMA-off), `ctest` 4/4, SYNCCHECK clean on BOTH branches, and **OpenMP bit-identical** — confirming D19 extends from pure maps to a node **gather** (`compute_vel_nodes`: the per-node accumulation is a *private* reduction over the CSR range, not a cross-thread reduce, so it keeps the C accumulation order on every backend) and to a 3-loop dependent kernel.

- **L28 — the Serial bit-identity gate CANNOT catch a missing `sync_device()` of a kernel INPUT, so sync all device-kernel inputs explicitly — never rely on "it's already device-current from an earlier substep".** On the Serial (and OpenMP) backend the Field host and device views *alias the same memory*, so every `modify_host/sync_device/sync_host` is a no-op and the kernel reads the live host data regardless of whether you synced it. A forgotten input `sync_device()` therefore passes the Serial `max|Δ|==0` gate **and** SYNCCHECK (whose `h_checked()` only traps a stale-*host* read, not a stale-*device* read) — it only manifests on CUDA, as a hard-to-spot climate-ish divergence (the kernel reads alloc-zeros or last-step's device bytes). KPP reads 11 input fields; rather than reason about which are "already device-current from the substep-1 EOS rail" (`sw_alpha/sw_beta/dbsfc/hnode/S`) vs genuinely host-dirty (`bvfreq` via the L27 smooth, `uvnode` via its halo, the 4 `forcing` fields), the KPP IN rail `modify_host()+sync_device()`s **all** of them. The redundant copies (Synced fields re-copied) are cheap host→device moves and the price of robustness; correctness-first (D6). **Rule: derive a kernel's full device-input set from its body and sync every one in its IN rail; the Serial gate validates arithmetic, NOT the sync graph.**
- **L29 — KPP (1046 LoC, the largest kernel; M2.3) landed bit-identical on the first COMPLETE Serial gate run, proving the D19/D20 template scales to a multi-stage pipeline with internal exchanges, a table-lookup inline function, and per-column scratch.** What the EOS/PP template absorbed without change: (1) **6 sub-stages as separate `parallel_for`s** (`prestep`×2, `ri_iwmix`×2, `bldepth`×3, `blmix`, `enhance`, `combine`, `viscAE`) flowing device→device on owned nodes — the launch barriers give the inter-stage ordering for free (D20), and only the 2 halo points break the device residency (D21). (2) **`kpp_wscale` as a templated `KOKKOS_INLINE_FUNCTION`** taking the `wmt/wst` lookup tables as device Views (set-once, one-shot-pushed in `fesom_kpp_init` like the mesh geometry) — the GPU analogue of a `static` host helper. (3) **per-column stack scratch in the lambda** (`blmix`'s `dthick[64]`, `dcol[64][3]` ≈ 2 KB/thread — slow GPU local memory, accepted slow-first) exactly as EOS's `bulk_*[64]`. (4) C `continue` in a per-entity loop → `return` in the per-entity lambda; C `break` out of an inner level loop → unchanged (each lambda runs its own inner loop). (5) the verify saves/restores the file-static `s_kpp_call` so the C-twin oracle's `++` doesn't double-count the dump-step counter. CUDA (A100) ran clean and climate-close at the **unchanged** M2.1 budget (Av/Kv ≈0.095 at the *same* threshold-flip nodes — KPP's own device-fma ULPs are below the resolution of the bvfreq-seeded mixing-threshold flips). ⚠️ **The `*/`-in-comment trap (L12/L24) struck a FOURTH time** — a header doc-comment `sync_device on bvfreq/sw_*/dbsfc/...` contains `*/` (the `sw_*/dbsfc` glob), closing the block early → `'dbsfc' does not name a type` + `extended character § is not valid` cascades pointing at the prose, not the comment. The `X_*/Y` glob in a `/* */` block is now a confirmed serial offender; **grep new `/* */` comments for `*/` before building** (reword `sw_alpha/sw_beta` instead of `sw_*`).

- **L30 — M2.4 `pressure_force` (substep 2) landed bit-identical first run; its INPUT rail is a third concrete L27 instance — a device output re-read on the device at a LATER substep needs `modify_host()+sync_device()`, not a bare `sync_device()`.** `pgf_x/pgf_y` is a clean per-element map (each element writes only its own slots → EOS-style D19, **Serial AND OpenMP bit-identical**, no scatter). The only subtlety is the input rail for `aux->hpressure`: it is *produced on the device* by `pressure_bv_kk` (substep 1), then `sync_host()`'d and **halo-exchanged on the host** (substep 1). The PGF kernel reads `hpressure` at each element's 3 **vertices**, which can be HALO nodes (`n0/n1/n2 ≥ myDim_nod2D`) — so the kernel genuinely needs the halo-exchanged values, and the host halo write is invisible to the DualView (L14). A bare `sync_device()` (assuming "it's device-current from substep 1") would feed the kernel the *pre-halo* device bytes → a stale-halo divergence the Serial gate can't catch (L28: host==device aliases). So the rail does `hpressure.modify_host(); hpressure.sync_device()`. This is the same shape as `bvfreq`→`mo_convect` (L27, where the host writer was `smooth_nod3D`); here the host "writer" is the halo exchange itself. **Rule confirmed: a cross-substep input that any host op (smoother OR halo) touched after its device production must be re-pushed with `modify_host()+sync_device()` — derive this from the kernel's actual read footprint (does it read halo entities?), per L24/L28.** Verify key `pgf` (no substring collision with `eos`/`pp`/`kpp`).

- **L31 — M2.4 `impl_vert_visc` (substep 6) — the per-element TDMA shape lands bit-identical (Serial AND OpenMP) on the first run; the EOS `bulk_*[64]` scratch pattern carries a full tridiagonal solve unchanged.** The whole TDMA — coefficient build (interior/bottom/surface rows), the "solve-for-`du`" conversion, and the Thomas forward-elim + back-sub — runs **sequentially in level INSIDE the per-element lambda** over lambda-local `[64]` stack scratch (10 columns: `zbar_n/Z_n/a/b/c/ur/vr/cp/up/vp` ≈ 5 KB/thread of slow device-local memory, slow-first accepted). Each element solves only its own `(u,v)` columns → no cross-element race → Serial == the C twin **and OpenMP is bit-identical too** (no scatter/reduction — unlike the substep-4/5 scatter kernels). Three rail details worth recording: (1) **the body reads MORE than the SYNC_MAP row's guess** — the row listed `uv_rhs/Av/stress_surf`, but the kernel also reads `uv` (bottom drag + the `du` conversion), `w_i` (the vertical-advection coupling — at the 3 element vertices, which can be **halo** nodes), and the **evolving** `mesh->helem` (rebuilt each step by `ale_commit_thickness`, so NOT in the one-shot geometry push — must be `modify_host()+sync_device()` per step). L24/L28 again: **derive the device-input set from the body, sync every one** (the IN rail syncs all 6). (2) `forcing` is `const` in the driver → localized `const_cast` for `stress_surf`'s coherence sync (the KPP/D21 pattern). (3) `uv_rhs` is **read-modify-write** (read to build the RHS, overwritten with the solution) → the verify is the L26 capture-before (driver snapshots pre-kernel `uv_rhs`); the levels the kernel leaves untouched (outside `[nzmin,nzmax)`) match by construction (both KK and C twin keep the captured value there). `sqrt`→`Kokkos::sqrt` (host == libm IEEE → Serial bit-identical). Verify key `ivisc` (distinct token from substep-5's `vfilt` so neither is a substring of the other, L25).

- **L32 — M2.4 `visc_filt_bidiff` (substep 5) — the first SCATTER kernel landed Serial bit-identical on the first run, empirically confirming D22.** The biharmonic ∇⁴ is two edge→element scatter stages around an internal halo. Ported with `Kokkos::atomic_add` (D22): `FESOM_KK_VERIFY=vfilt` was `max|Δ|==0` for all 20 pi steps on Serial AND the full-run pi smoke stayed `ALL FIELDS BIT-IDENTICAL` to the golden — so the single-thread `atomic_add` really does reproduce the C `+=` byte-for-byte (the `-ffp-contract=off` build keeps the add unfused). Five things worth recording: (1) **`Uc[el1]-=du` → `Kokkos::atomic_add(&Uc(i), -du)`**; for the `/area` cases write `-(du / a1)` (parenthesised) so it is unambiguously the negation of the value the C subtracts — `a - b == a + (-b)` and `(-x)/y == -(x/y)` are both IEEE-exact, so it is bit-identical. (2) **The internal Uc/Vc elem3D halo is a D21 bracket** (`modify_device → sync_host → exchange via h_checked → modify_host → sync_device`), the exact KPP `smooth_blmc` idiom; the exchange is gated `npes>1` (no-op at np=1) but the sync round-trip is a no-op on Serial regardless. (3) **The stage-1 scratch `u_b/v_b` is zeroed on the device** with a `parallel_for` fill (matching the C `memset` over `E_alloc*nl`) — not synced from host (pure scratch, fully overwritten). (4) **The verify captures the FULL local `uv_rhs`** (not just owned) because the stage-2 scatter writes halo elements too (via the eDim edges) — restoring only owned would leave the C twin double-applying the biharmonic to halo entries; the subsequent halo exchange would overwrite that garbage in production, but capturing full keeps the verify clean (and it is a no-op distinction at np=1 where eDim=0). (5) **Keep `FESOM_VISC_MULT`** (a real physics knob scaling γ0/1/2, default 1.0) read on the host and captured into the lambda; the one-time `FESOM_DIAG_VISCEDGE` host diagnostic stays verbatim in the preamble. Verify key `vfilt` (distinct token from `ivisc`, no substring collision, L25). **OpenMP is now climate-close (≲1e-12), not bit-identical, for this kernel — the expected D22/D5 regime change, the first in the port.**

- **L33 — M2.4 `compute_vel_rhs` (substep 4) landed Serial bit-identical first run; it is the most COMPOSITE kernel so far — it combines every M2 pattern at once — and the handoff under-described it as merely "element-parallel".** The reality: `compute_vel_rhs` calls `momentum_adv_scalar` (momadv_opt=2) **unconditionally**, so a faithful device port is two race-free per-element maps (the Coriolis/SSH-grad/PGF body + the AB2 assembly, D19) bracketing a **5-stage** `momentum_adv_scalar_kk` that itself contains a per-node private gather (vertical advection, D19), an **edge→node SCATTER** (horizontal advection, `atomic_add`, D22), a per-node in-place scale, an **INTERNAL `uvnode_rhs` nod3D halo bracket** (D21), and a per-element gather (vertex→element) — 7 `parallel_for`s + 1 halo bracket in one substep. Recording what made it land clean: (1) **⚠️ the AB2 `eps=0.1`** stabilization offset (the dt=1800 trap, PORTING_LESSONS §1) is captured verbatim into the lambda (`ab1=-(0.5+eps)`, `ab2=(1.5+eps)`) — a `const real_t` read on the host (where `FESOM_PHASE1_DT`/`eps` live) and captured by value, never referenced as a global inside the device lambda. (2) **The scatter's `un[n2] -= X`** becomes `atomic_add(&un(n2), -(expr))` — the negation of the *same* expression the `+=` to `n1` uses; `a-X == a+(-X)` IEEE-exact, and re-evaluating `expr` (as the C does in its separate n1/n2 loops) is deterministic, so Serial stays bit-identical. (3) **`uv_rhs`/`uv_rhsAB` stay device-resident across all 7 launches** (part iii writes `uv_rhsAB`, momentum_adv stage 5 `+=`s it, the assembly reads it) — the launch barriers order them; only `uvnode_rhs` breaks residency for its halo. (4) **Verify capture-before on `uv_rhsAB` ONLY** — part i reads `uv_rhsAB` (→ restore it for the C twin), but `uv_rhs` is fully recomputed from `uv_rhsAB`+inputs (never read as a pre-existing input → no restore needed); the untouched levels match by construction (L31). **M2.4 milestone integration check:** all **7** M2 device kernels (eos, pgf, vrhs, vfilt, ivisc, pp, kpp) ran with `FESOM_KK_VERIFY` set simultaneously — **160 verify lines, 0 non-zero** on Serial, and the pi smoke stayed `ALL FIELDS BIT-IDENTICAL` to the golden. Substeps 1–6 of the ocean step are now device-resident (modulo the internal/boundary halos and the mid-step CG round-trip). Verify key `vrhs` (no substring collision).
- **L34 — M2.5 ALE (substeps 12/14) landed Serial bit-identical first run; its headline finding is that the handoff/SYNC_MAP claim "GM off by default → `fer_w` dead on the golden path" is FALSE — the pi smoke runs GM ON, so `vert_vel`'s `fer_w` accumulator is LIVE, and porting its `gm_on` branch verbatim + wiring its `fer_uv`/`fer_w` rails is exactly what kept pi == golden.** The five kernels split cleanly by shape (all derived from the C BODY, L33): (a) **`thickness`/`commit`/`cflz`/`wvel_split` = race-free maps → bit-identical on Serial AND OpenMP** (verified: their `FESOM_KK_VERIFY=ale` max|Δ|=0 on *both* backends). `thickness`/`commit`'s `hnode:=hnode_new` is a flat copy `parallel_for` (the device twin of `memcpy`); `commit`'s `helem` mean is a second launch (the barrier orders the read of the just-copied `hnode`, D20). (b) **`vert_vel` = an edge→node SCATTER (`atomic_add`, D22) + a per-node sequential level cumsum** → Serial bit-identical, OpenMP/CUDA climate-close (measured OpenMP `w`≈3.4e-21, `fer_w`≈7e-22 — the D22 regime, ≈FP floor, ≪ the ≲1e-12 budget; consistent with M2.4). The cumsum is a per-node *private* scan (each node owns its column → no cross-thread dep → race-free), so only the scatter step breaks OpenMP bit-identity, not the scan. Four rail/correctness points worth recording: (1) **`cflz`'s `+=` is NOT a scatter** — each node accumulates into its OWN `[0,nl)` column slice (disjoint across nodes), so it is a private accumulation like `compute_vel_nodes` (L27), bit-identical on OpenMP; the device kernel zeros the node's full column in-lambda (the per-node twin of the C `memset`) before accumulating, so the full array matches the C `memset`+loop. The gather-vs-scatter line (D22/SCATTER_STRATEGY §) is per-*receiver-ownership*, not per-`+=`. (2) **`hnode_new` stays device-resident across 12a→12c→14** — `thickness` (12a) writes it on the device then `sync_host()`s it (REQUIRED: the host tracer advect/diff in substeps 13/13b READ it via the raw alias), leaving it `Synced`; nothing writes it again until next step, so `cflz` (12c) and `commit` (14) read the device copy current with a no-op `sync_device()` (never `modify_host()` it — that would falsely mark host-dirty). (3) **No ALE kernel has an internal halo** — every `fesom_exchange_nod3D(w/cfl_z/w_e/w_i)` and `exchange_elem3D(helem)` is a DRIVER halo between kernels, so the data bounces device→host(halo)→device and each kernel's IN rail re-pushes the just-halo'd input (`modify_host()+sync_device()` on `w` before `cflz`, on `cfl_z` before `wvel_split`) — no D21 bracket needed. (4) **`use_wsplit=.false.` preserved verbatim** (the dt=1800 trap, PORTING_LESSONS §0 #3): the full `use_wsplit` branch is in the lambda; with the compile-const flag 0 every interface takes the `else` (w_e=w, w_i=0). No kernel needed an L26 capture-before — every output is a full overwrite from inputs the kernel doesn't modify, so the C-twin oracle recomputes from intact state (EOS-style gate). **The GM-on surprise is the L24/L33 rule biting again: don't trust "dead on the golden path" — derive liveness from what the run actually does; the verify (Serial `fer_w`=0 AND OpenMP `fer_w`≠0) is what disambiguated that the `gm_on` branch IS exercised.** Verify key `ale` (one key gates all five; no substring collision with eos/pp/kpp/pgf/ivisc/vfilt/vrhs).
- **L35 — M2.5b-b: the substep-1b GM chain (5 kernels) landed Serial AND OpenMP bit-identical on the first complete run; it is the first DEVICE→DEVICE chain whose internal halos all live in the DRIVER (the ALE pattern, NOT KPP's D21), and it confirms the L34 "GM is ON in pi" finding feeds a fully device-resident streamfunction solve.** The chain is `compute_sigma_xy`→`compute_neutral_slope`→`init_redi_gm`→`fer_solve_gamma`→`fer_gamma2vel`, and every kernel is a race-free map / private gather / per-node TDMA (no scatter, no cross-thread reduce) → `FESOM_KK_VERIFY=gm` was `max|Δ|==0` on **both** Serial and OpenMP for all 5 kernels × 20 steps, pi stayed `ALL FIELDS BIT-IDENTICAL` (np=1 + np=2 CMA-off), SYNCCHECK clean, CUDA built. What this added to the record: (1) **the C twins each do their OWN internal halo, but unlike KPP they are NOT D21 brackets — they move to the driver** (the ALE/M2.5 shape): the 5 kernels flow device→device reading each upstream's OWNED output on the device (sync_host leaves the device view intact, so the next kernel reads the producer's bytes without a re-push), and the driver does `sync_host`+halo between them. The crux for deciding re-push-or-not: **does the next device kernel read the just-halo'd field at HALO entries?** Only `fer_gamma2vel` does (it gathers `fer_gamma` at the 3 element vertices, which can be halo nodes) → `fer_gamma` is the ONE re-push (`modify_host()+sync_device()` after its halo, the L30 bvfreq/hpressure cross-op shape). `sigma_xy`/`ns`/`st`/`fer_C`/`fer_K`/`Ki` are halo'd (for the verify bit-identity + downstream substep-13 Redi host readers) but read ONLY at owned entries by the intra-chain kernels → no re-push. (2) **must preserve EVERY halo the C does, even ones no in-chain kernel consumes** (`sigma_xy` is GM-internal yet halo'd): dropping it would diverge the halo entries from the C twin → the verify (which runs the C twin WITH its halo) would FAIL at np>1, and it could feed an I/O/downstream halo read. No-simplification applies to halos too. (3) **`fer_solve_gamma` is the L31 per-node TDMA** (10 × `[64]` lambda-local scratch ≈ 5 KB/thread) — the `impl_vert_visc` shape carries a second tridiagonal solve unchanged, bit-identical on OpenMP (race-free). (4) **ODM95 tanh-taper + `scaling_GMzexp` depth-exp + the Redi `sqrt(tapfac)` taper copied verbatim** into the lambdas (`tanh`/`exp`/`fabs`/`sqrt`→`Kokkos::`), every namelist constant a captured `const real_t` — the GM/Redi numerics knobs (PORTING_LESSONS) are exactly the "module-default scalar carries physics" trap, so a single mistyped constant would fail the Serial gate. (5) **the gm scratch Field-wrap (M2.5b-a) used `*g = fesom_gm{}` not `memset`** (D13/L13 — raw memset over DualView members is UB), the M2.3a KPP precedent. Verify key `gm` (2 chars, no substring collision with any existing key — plain `strstr`).
- **L36 — M2.5b-c: the substep-13 GM/Redi diffusion landed Serial bit-identical (and, surprisingly, OpenMP bit-identical too) on the first run; its headline is a SCOPE decision derived from the data flow — the bolus add/sub STAY ON HOST because they have no device consumer, while the Redi (real compute) goes to device as an island between the host FCT calls.** (1) **The bolus-host decision (L24/L33 in action).** The task said "port the bolus add/sub", but tracing the actual readers showed the bolus-augmented `uv`/`w`/`w_e` are consumed ONLY by the host FCT advect + host tracer-diff (grep-verified: both only READ, never write them; the device Redi doesn't read `uv`/`w`/`w_e` at all). Porting the trivial `uv+=fer_uv` maps to device would force a device→host round-trip serving only host code — pure overhead, zero device consumer. So they stay host (already bit-identical — M2.5 had them host on the golden path; M2.5b-b kept `fer_uv`/`fer_w` sync_host'd so the host bolus reads them current). They move to device in M2.6 alongside the FCT, when a device consumer of `uv`/`w`/`w_e` first exists. **Rule: a kernel goes to device when it has a device producer AND a device consumer (or is substantial offloadable compute); a trivial map whose only consumer is still host is a round-trip with negative value — defer it to its consumer's task.** Contrast KPP/the Redi (substantial compute → worth the host bounce) vs the bolus (a `+=` shuffle → not). (2) **The Redi as a device island with a host `values` round-trip** — `diff_ver`+`diff_hor` run between the host FCT(T)/FCT(S) calls: host FCT writes `values` → IN rail `modify_host()+sync_device(values)` → device Redi `+=` → OUT `sync_host(values)` → host halo, the exact M2.2-KPP-between-host / §5-CG bounce, accepted until M2.6. (3) **Two D21 internal halos** — `diff_ver` exchanges `tr_xy` (read by its own per-node gather at HALO elements), `diff_hor` exchanges `tr_z` (read at HALO edge-endpoints); `tr_xy` then flows diff_ver→diff_hor device-current (the bracket's final `sync_device` left it owned+halo current) — no re-push between the two kernels, but `slope_tapered`/`Ki` ARE re-pushed in the driver IN rail because `diff_hor` reads them at halo edge-endpoints (L30). (4) **`diff_hor` is the 4th scatter (D22)** — edge→node `atomic_add` into `values` with 5 partial-cell branches + the `COMPUTE_KH_TZ_S` macro carried verbatim (macro indexing `[...]`→`(...)` for the device views). Serial bit-identical; **OpenMP also bit-identical on the pi mesh** (unlike vert_vel, the Redi scatter didn't cross an associativity boundary that diverged — the whole-run OpenMP floor stayed the M2.5 vert_vel `w`≈3.4e-21, no NEW class). (5) **`values` is read-modify-write → L26 capture-before**: the driver snapshots the post-FCT `values` (pre-Redi) into a `std::vector` and passes it to `fesom_gm_redi_verify`, which restores it, runs BOTH C twins in order (diff_ver rebuilds `tr_xy`, diff_hor reads it), diffs, restores KK — the combined-Redi gate (one verify per tracer, key `gm`). This completes the GM/Redi PHYSICS on device (1b chain + Redi); the GM scratch wrap (M2.5b-a) + the streamfunction chain (M2.5b-b) + this close M2.5b.

- **L37 — M2.6-b: the FCT tracer advection (the largest pipeline — `fesom_tracer_advect_one_fct`, ~24 `parallel_for` launches + 3 internal-exchange D21 brackets + 3 edge→node scatters in ONE function) landed Serial bit-identical on the FIRST complete run; it is the M2 capstone for the ocean step.** `FESOM_KK_VERIFY=tradv` was `max|Δ|==0` for all 20 pi steps × 2 tracers (40 lines, 0 non-zero), the production pi smoke stayed `ALL FIELDS BIT-IDENTICAL` (np=1 AND np=2 CMA-off — exercising all 3 D21 brackets + 3 scatters under MPI), `ctest` 4/4, SYNCCHECK clean. What carried the D19–D22/L26/L33 template through the most composite kernel yet:
  - **⚠️ Both dt=1800 tracer traps preserved verbatim (PORTING_LESSONS §1).** (1) **AB2 `eps=0.1`** in `init_tracers_AB` (`c_old=-(0.5+eps)`, `c_new=(1.5+eps)`), a `const real_t` captured into the lambda. (2) **`feedback_mfct_gradient_from_values`**: the MFCT element gradient (`tracer_gradient_elements`) is built from **`values`** while the MFCT *flux* (`adv_tra_hor_mfct`) uses **`valuesAB`** — two DIFFERENT tracer fields in the same HO step (the Fortran `oce_tracer_mod.F90:127` uses `values`, :126-the-`valuesAB`-variant is commented out). A single field swap here would pass dt=500 and drift at dt=1800. Both kernels read exactly the C twin's field.
  - **The a3+a4 FUSION removed the C's malloc'd `[N*nl]` `tvert_max/min` entirely.** The Zalesak `a3` (cluster max/min over surrounding cells) writes `tvert[n,nz]` and `a4` (admissible increment vs LO with a vertical 3-layer cluster) reads `tvert[n, nz-1..nz+1]` — all within node `n`'s OWN column. So a3+a4 fuse into one per-OWNED-node lambda with **column-local `tvmax/tvmin[NL_MAX]` scratch** (the EOS/`impl_vert_visc` per-column-scratch shape), bit-identical and avoiding a per-call device allocation. **Rule: when a 2-stage node algorithm's second stage reads the first's output only within the same node's column, fuse to lambda-local scratch (don't replicate the C's global temp array).**
  - **AUX (per-element Zalesak max/min) and the scatter receivers replicate the C's "halo stays at its prior value" behaviour, NOT a fresh zero.** `a2` writes AUX for OWNED elements only (incl. the `±bignumber` deep-level padding); `a3` gathers AUX over a node's surrounding cells, **which can be HALO elements `a2` never wrote** — on BOTH device and host those read the alloc-zero (`Field.alloc` zero-inits; the C `calloc` + never-write), so they match with **no per-call AUX zeroing** (matching the C, which also never zeros AUX per call). Likewise `compute_fct_LO`/`fct_plus`/`fct_minus` scatter into HALO node slots (the edge endpoints) that the owned-only zero loops never cleared — but **nothing reads those halo slots before the D21 exchange overwrites them** (`a1`/`b2`/the finalise all loop owned-only), so the halo scatter garbage is irrelevant and the device need only match the C's owned region + let the exchange fix the halo. **Rule (L24/L33 again): derive the zero/halo extent from the C body; a device kernel that "helpfully" zeros the full extent can DIVERGE from a C twin that intentionally leaves halo stale — verify whether any consumer reads the halo before its exchange.**
  - **`adv_tra_ver_qr4c`'s same-slot double write is preserved by keeping the per-node body ONE lambda (sequential steps), not separate launches.** When `nzmax-nzmin==2` the "2nd layer" (`nzmin+1`) and "bottom-1" (`nzmax-1`) are the SAME interface, written twice in sequence (the second `= … - flux` consumes the first's result). A per-node lambda runs the surface/2nd/bottom-1/bottom/interior steps sequentially exactly as the C → the double-application is reproduced; splitting into per-step launches would also work (same node owns the slot) but the single lambda is the faithful, obvious port. (D20's "one launch per dependent loop" is for CROSS-entity deps; an intra-column sequential dependence stays inside the lambda.)
  - **3 SCATTERS (D22 `atomic_add`):** `compute_fct_LO` divergence (`fct_LO[n1]+=f; [n2]-=f`), the Zalesak `fct_plus/fct_minus` antidiffusive-sum assembly (b1 horizontal), and `flux2dtracer`'s horizontal `del_ttf_advhoriz` accumulation. `[n2]-=X` → `atomic_add(&v(n2), -(X))` with the negation parenthesised (L32). **Serial bit-identical; OpenMP climate-close BUT added NO new divergence class on the pi mesh** — the whole-run OpenMP floor stayed the M2.5 `vert_vel` `w`≈3.4e-21 / advected `u`≈2.2e-19, exactly as before M2.6, i.e. the 3 FCT scatters didn't cross an associativity boundary that diverged (same finding as the M2.5b-c Redi scatter, L36).
  - **3 D21 internal-exchange brackets, ALL owned by the function** (the C twin does the exchanges *inside* the pipeline, not in the step driver): `fct_LO` (nod3D — Zalesak `a1` reads LO at halo nodes), `tr_xy` (`FESOM_HALO_ELEM2D_FULL`, **`nl` not `nl-1`** — the FCT `tr_xy` is `[E*nl*2]` stride-nl, unlike the GM Redi `tr_xy` which is `[E*(nl-1)*2]`; derive the exchange `nl`/ncomp from the array's own stride, L33), and `fct_plus`+`fct_minus` (two nod3D calls — `b3` horizontal reads them at edge endpoints). Each is the KPP/Redi idiom `modify_device → sync_host → exchange (h_checked) → modify_host → sync_device`. The DRIVER owns only the IN rail (push `values`/`valuesold`/`uv`/`w_e`/`hnode`/`hnode_new`/`helem` — the bolus-augmented `uv`/`w_e` from the still-host 13a, L28) and the OUT rail (`sync_host(values, valuesold)` — the Redi rail + halo + next step read both; valuesold because `init_AB` set it `=values` on the device, L27-style).
  - **Verify = L26 capture-before with TWO inputs (`values` AND `valuesold`).** `init_AB` reads the ORIGINAL `valuesold` (last step's values) to form `valuesAB`, THEN overwrites `valuesold=values`. So the C-twin oracle needs BOTH the pre-FCT `values` and the pre-FCT `valuesold` restored (restoring only `values` would feed the C twin the post-FCT `valuesold` and diverge). Snapshot both pre-FCT, restore both, run the C twin, diff the final `values`, restore KK `values`+`valuesold`. `del_ttf`/`valuesAB` are FCT-internal (no host reader after the FCT; C-overwritten == KK on Serial).
  - **The upwind path stays untouched C** (`fesom_tracer_advect_one`, `flux2dtracer_upwind`, `adv_tra_hor_central`) — dead on the golden timestep (only the `do_sanity` probe in `fesom_main` calls `advect_one`); only the FCT path (`_fct`) is the default tracer scheme, so only it is ported. Verify key `tradv` (distinct token, collision-free both directions, L25).

- **L38 — M2.6-c: the GM bolus add/sub moved to device (`fesom_gm_bolus_apply_kk`, the L36 deferral resolved) — landed Serial AND OpenMP bit-identical; the subtle part is the verify interaction, NOT the kernel.** The bolus is the simplest possible kernel: two elementwise maps (`uv += sgn*fer_uv` over `(myDim+eDim)` elements; `w`,`w_e += sgn*fer_w` over `(myDim+eDim)` nodes), `sgn=±1.0`. `1.0*x==x`, `-1.0*x==-x`, and `a+(-b)==a-b` are all IEEE-exact, so the sign-parameterised map is bit-identical to the C `+=`/`-=` and one function serves both 13a (add) and 13c (sub). No scatter/reduce → bit-identical on Serial AND OpenMP. **L36's rule fired exactly as written:** the bolus moved to device the moment a device CONSUMER of `uv`/`w_e` existed (the M2.6-b device FCT) — not before. Three things worth recording:
  - **The device bolus-add must `sync_host(uv/w/w_e)` even though the FCT consumes them on the device** — for two reasons, neither obvious. (1) The next step's substep-3 (`update_vel`/`compute_vel_nodes`) reads `uv` on the HOST, so after 13c restores it the host must be current; symmetrically the host must mirror the augmented velocity across the FCT region for any host reader. (2) **The `FESOM_KK_VERIFY=tradv` C twin reads `dyn->uv` on the HOST** — if 13a left host `uv` pre-bolus (device-only augmentation), the C twin would advect with un-augmented velocity and the verify would FALSELY fail. So 13a does `modify_device → sync_host(uv,w,w_e)`: the host mirrors the device-augmented velocity. Then `uv`/`w`/`w_e` stay `Synced` (augmented) on the device through the whole FCT region (the FCT only READS them; Redi/tracer-diff/sfloor never touch them), so 13c needs NO input push — it reads the device-augmented `uv` + the still-device-current `fer_uv`/`fer_w` (pushed at 13a) and subtracts. The FCT IN rail's `uv`/`w_e` push is kept (it is a no-op when `gm` — they're already `Synced` — and is REQUIRED when `!gm`, where there is no bolus to make them device-current).
  - **The tradv verify CANNOT independently catch a wrong device bolus** — and that is fine. Because both the device FCT and the verify's C twin read the SAME device-bolus-augmented `uv` (the C twin via the 13a `sync_host`), a bug in the bolus would corrupt BOTH identically → the verify would still report `max|Δ|=0`. The bolus's correctness therefore rests on the **Serial `diff_snap` pi==golden gate**: the golden uses the C HOST bolus, and on Serial the device bolus kernel runs on the same memory (host==device) as the identical C loop, so pi==golden directly proves device-bolus == C-bolus. **Rule: a read-modify-write kernel whose output feeds a verified downstream kernel (which reads the same buffer) is NOT covered by that downstream's verify — its own gate is the end-to-end bit-identity run.** (Net-restoring `add` then `sub` is also not the identity in FP — `(uv+fer)-fer != uv` in general — but the device reproduces the C's exact two-step arithmetic, so it matches the golden bit-for-bit whatever the C gets.)
  - **This completes the ocean step's device residency through substep 13** (the only remaining host compute is the M2.7 `impl_vert_diff_tracers` + the trivial salinity floor + the mid-step §5 CG). Verified all backends at the unchanged M2.5/M2.6-b budget: Serial pi==golden (np=1 + np=2 CMA-off), `tradv` 40×`max|Δ|=0`, `ctest` 4/4, SYNCCHECK clean, `FESOM_NO_GMREDI=1` clean (bolus gm-gated → skipped), OpenMP climate-close (same floor as M2.6-b — the bolus map added nothing), CUDA climate-close.

- **L39 — M2.7: the implicit vertical tracer diffusion (`impl_vert_diff_tracers`, substep 13b — the LAST host ocean compute) landed Serial AND OpenMP bit-identical on the first complete run; it closes the ocean step's device residency, and its one judgement call — keeping the salinity floor on the host — is L36 applied verbatim.** `FESOM_KK_VERIFY=trdiff` was `max|Δ|==0` for all 20 pi steps × 2 tracers (40 lines, 0 non-zero) on Serial; the pi smoke stayed `ALL FIELDS BIT-IDENTICAL` (np=1 AND np=2 CMA-off — exercising the new IN/OUT rails + the two `h_checked()` `values` halos under MPI), `ctest` 4/4, SYNCCHECK clean (no abort, == golden). **OpenMP is bit-identical** for this kernel: the `trdiff` verify prints `0.000e+00` on OpenMP too, and the whole-run OpenMP diff shows NO `temp`/`salt` divergence class — only the pre-existing M2.5 vert_vel `w`≈3.4e-21 / momentum-scatter `u`≈2.2e-19 floor (the per-node TDMA has no scatter, unlike the Redi/FCT). The all-keys integration check (eos/pp/kpp/pgf/vrhs/vfilt/ivisc/ale/gm/tradv/trdiff simultaneously) was **460 verify lines, 0 non-zero** on Serial. Five things worth recording:
  - **The per-node TDMA is the THIRD instance of the L31 shape** (`impl_vert_visc_kk` per-element, `fer_solve_gamma_kk` per-node, now `diff_ver_part_impl_ale_kk` per-node). The whole Thomas solve — coefficient build (surface / interior / bottom rows + the Redi K33 `Ty/Ty1` augmentation) + RHS + surface BC + shortwave penetration + forward-elim + back-sub — runs sequentially in level INSIDE the per-node lambda over 8 × `[NL_MAX]` lambda-local stack columns (`a/b/c/tr/cp/tp/zbar_n/Z_n`). Each node solves only its own column of `values` → no cross-node race → bit-identical on Serial AND OpenMP. The `[0,NL_MAX)` scratch zero-init is copied verbatim from the C (only `[nzmin,nzmax]` is ever read, so the result is independent of the zeroed range, but verbatim removes all doubt).
  - **`bc_surface` → a templated `KOKKOS_INLINE_FUNCTION`** over the four forcing Views (the `kpp_wscale_kk` idiom). `id` is a per-launch constant (T=1, S=2) but both branches compile, so all four forcing arrays (heat_flux/water_flux for T, virtual_salt/relax_salt for S) are captured into the lambda and synced in the IN rail regardless of the launched tracer — the dead branch's reads are compiled, never executed. `x*0.0==0.0` for finite `x`, so the `is_nonlinfs=0` water_flux term is exactly 0 whatever the value, but the read is preserved verbatim.
  - **The GM K33 augmentation needs the optional `slope_tapered`/`Ki` Views captured even when `gm==NULL`** — declared as default-constructed empty `fesom::Field::dev_view_t`, assigned only `if (gm)`, indexed only under the captured `gm_on` int. An empty Kokkos View is safe to capture by value into a `KOKKOS_LAMBDA` (a null device handle with zero extents, never dereferenced when `gm_on=0`) — the device twin of the C `st=NULL`/`Ki=NULL` guard. This is the first kernel whose gm-dependence lives INSIDE the lambda (the FCT/Redi were `if(gm)`-gated in the driver), because tracer diffusion runs unconditionally while the K33 term is gm-only.
  - **The salinity floor STAYS HOST (L36 applied).** Tracing the readers: after the device trdiff + OUT `sync_host(values)` + the two host nod3D halos, the host S-floor clamps S over myDim+eDim (idempotent, deterministic, no halo needed — the PORTING_LESSONS intent). The ONLY device consumer of the clamped S is NEXT step's substep-1 EOS, which already re-syncs T/S with `modify_host()+sync_device()` at its own IN rail — so the silent host halo+clamp writes propagate there for free. Moving the floor to device would be a pure round-trip serving only the host halo, with zero device consumer this step. Exactly L36/L38's "a kernel goes to device when it has a device consumer." The device trdiff's OUT-rail `sync_host` then the host halo+clamp leaving `values` Synced-but-silently-host-modified is the SAME shape the host trdiff had (the next IN rail re-syncs); the only change is trdiff itself moving to the device.
  - **The IN rail is the L28 derive-from-the-body discipline once more.** The body reads `Kv` (KPP/PP device output — re-pushed `modify_host()+sync_device()` so the device owned Kv matches the host post-halo, the ivisc `Av` pattern), the 5 forcing arrays (`forcing` is `const` in the driver → localized `const_cast`, the KPP/ivisc pattern), per-tracer `values` (post-Redi `sync_host` + the FCT/Redi halos), `slope_tapered`/`Ki` re-pushed if gm. `hnode_new` is Synced since 12a (no-op `sync_device`, documents the dep) and `area`/`areasvol`/`zbar`/levels are set-once device-current (no push — the FCT/ALE convention; re-pushing the `[N*nl]` geometry every step would be wasteful and is unnecessary). `values` is read-modify-write → the verify is the L26 capture-before for BOTH T and S (snapshot KK, restore pre, run the C-twin DRIVER which does T then S, diff both, restore KK). Verify key `trdiff` — collision-free: "tradv"⊄"trdiff" and "trdiff"⊄"tradv", plain `strstr` (L25). **M2 milestone: the full ocean step minus the §5 mid-step CG round-trip (M4.2) + the M4.1 reductions + the ice step is now device-resident.**

- **L40 — M3.1: multi-GPU run configuration (rank→GPU device mapping). This is RUN-CONFIG, not a kernel port — there is NO `FESOM_KK_VERIFY` gate; the change is one block in `src/fesom_main.cpp` + the GPU job scripts + `docs/RUN_GPU.md`. Four findings worth recording:**
  - **The device-id mapping is the WHOLE fix, and correctness follows from it alone.** `Kokkos::initialize(argc,argv)` set no device id → every rank grabbed GPU 0 (the multi-GPU bug). Replaced with the NODE-LOCAL rank (`MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, …)` → `MPI_Comm_rank`; `getenv("SLURM_LOCALID")` fallback) fed to `Kokkos::InitializationSettings::set_device_id`. **Node-local, NOT global** — every node exposes GPUs 0..(g-1), so global rank 5 on node 1 (4 GPUs/node) must drive device 1. **No kernel or halo changes:** every `fesom_exchange_*` already runs on HOST pointers (`h_checked()`) and the M1.5 rails `sync_host` before / `sync_device` after each halo, so a multi-GPU halo stages device→host→MPI→host→device with REGULAR (non-GPU-aware) MPI — the binding is the only change (GPU-aware device-pointer halos = M5/perf). Verified: pi `dist_2` (rank0→dev0, rank1→dev1, distinct A100 UUIDs) and CORE2 `dist_8` across 2 `gpu` nodes (the split correctly RESETS the node-local rank to 0..3 on each node). **No-op on Serial/OpenMP:** Kokkos reads `device_id` only from a GPU backend's `Impl::get_gpu()` (confirmed in `Kokkos_Core.cpp` — never called for host-only builds), so the CPU bit-identity oracle is untouched — re-ran ALL gates after the change (460 `FESOM_KK_VERIFY` lines `max|Δ|=0`, pi==golden np=1 AND np=2 CMA-off, `ctest` 4/4, SYNCCHECK clean; OpenMP at the unchanged M2.5 floor `u`≈2.2e-19/`w`≈3.4e-21, `device_id()`=-1). For `npes==1` the node-local rank is 0 ⇒ device 0 = the old default.
  - **SLURM binding contract: every GPU on a node visible to every rank on it** — `--gres=gpu:<g> --ntasks-per-node=<g> --gpu-bind=none`. Then `set_device_id(local_rank)` picks `visible_devices[local_rank]` (distinct per rank). If a per-task bind (`--gpus-per-task=1`/`--gpu-bind=single`) made each rank see only ITS GPU as device 0, rank>0 would request an id ≥ the visible count and **Kokkos ABORTS** (`Requested GPU with id 'N' but only M GPU(s) available`) — a loud signal, not a silent double-bind. Kokkos' built-in `set_map_device_id_by("mpi_rank")` (env-var local rank + `% num_visible` modulo) is the alternative that tolerates per-task binding/oversubscription; the explicit `MPI_COMM_TYPE_SHARED` split was chosen because it is launcher-independent and the printed mapping is under our control (the route for >1 rank/GPU via MPS is the modulo form, M5). gpu-devel = 2 A100/node (30 min); gpu = 4 A100/node (12 h) — so `#nodes = ceil(N/GPUs-per-node)`.
  - **Validating a multi-rank GPU run needs the SAME-rank-count Serial oracle, NOT the np=1 golden** (the M1_ACCEPTANCE same-rank rule, now for CUDA). The decomposition perturbs scatter/halo order at partition boundaries, so np=1 and np=2 are NOT bit-identical even on Serial (pi step 20: pure np=1↔np=2 **Serial** diff = density 3.9e-5 / T 1.7e-4 / u 5.1e-3 — exactly why a separate np=2 oracle `…_m13_nocma` exists, L18). Diffing **np=2-CUDA vs the np=1 golden folds that partition effect (3.9e-5) together with the tiny CUDA device-fma budget (2.75e-11)** and the partition effect swamps it by ~6 orders → looks alarming but is a PASS (D5). Diff vs the **np=2 Serial oracle** to isolate the CUDA budget (here density 2.75e-11 ≈ the single-GPU 3.18e-12 — no new divergence class).
  - **The honest perf finding (recorded so no inflated claim leaks downstream): at M3.1 the CPU is ~17× FASTER node-for-node.** CORE2 dt=1800 per-step (loop = wall(105 steps) − wall(5 steps), no I/O): GPU 8×A100 / 2 `gpu` nodes `dist_8` = **0.86 s/step**; CPU 256 cores / 2 `compute` nodes `dist_256` = **0.051 s/step**. (The 8-cores `dist_8` run = 1.96 s/step "shows" the GPU 2.3× faster, but that is a PER-RANK comparison against 1/16 of a CPU node — misleading as hardware.) The GPU is not slow: the **CG solver + sea ice are STILL ON HOST** and, pinned to `dist_8` (1 rank/A100, the M3.1 contract), run only **8-way parallel vs the CPU's 256-way (~32× less)**, dominating the step (~0.7 of the 0.86 s = host CG ~127 iters/step + ice + the SYNC_MAP §5 PCIe round-trip); CORE2 also under-fills an A100 (~500k wet pts/GPU). `build-cuda` host backend is **Serial** = 1 host thread/rank. **The fix is M4.2 (CG→device) + M4.3 (ice→device) + higher-res meshes — a GPU↔CPU benchmark is only meaningful post-M4.2.** The §5 mid-step round-trip bit exactly as predicted. ALSO (run hygiene): model OUTPUT goes to `/work/ab0995/a270088/port2`, never `$HOME` (60 GB home quota; one CORE2 GPU run = 3.5 GB of snapshots) — see [[feedback-hpc-run-hygiene]].

- **L41 — M4.2: the §5 SSH block (compute_ssh_rhs + the preconditioned CG + update_vel + compute_hbar, substeps 7-10) moved to the device, closing the SYNC_MAP §5 mid-step host round-trip — the last per-step host ocean compute besides the ice step. Landed Serial bit-identical on the first complete run (`FESOM_KK_VERIFY=ssh` = `max|Δ|==0` for all 20 pi steps × 7 read-modify-write fields, 140 lines, 0 non-zero); pi==golden (np=1 AND np=2 CMA-off); `ctest` 4/4; SYNCCHECK np=1+np=2 clean + bit-identical; all 12 keys together (eos…trdiff,ssh) = 0 on Serial. It is the perf unlock (the M3.1 §5 PCIe round-trip + serial host CG was ~0.7 of the GPU step, L40). Six things worth recording:**
  - **The CG is HOST loop control + DEVICE vector kernels — you do NOT move the iteration loop or the convergence logic to the device.** The `for iter` loop, the `α`/`β`/`residual` scalars, the `residual < rtol` break, and the per-iteration `MPI_Allreduce` all stay on the host (the Allreduce result IS a host scalar, and the loop must branch on it). Each vector op becomes one launch: the SpMV `App = A·p`, the dot products, the AXPYs. The dots' `Kokkos::parallel_reduce` returns the partial sum to a host scalar, then the **unchanged** scalar `MPI_Allreduce` finishes it. ~5 launches × ~127 iters/step = ~640 launches/step — far cheaper than round-tripping the whole state to a serial host CG (the L40 bottleneck); kernel-fusion / graph capture is an M5 optimisation.
  - **The SpMV is a per-ROW CSR GATHER, NOT a scatter** (`y(row)=Σ_n vals(n)·v(colind(n))` over the row's own `[rowptr(row),rowptr(row+1))`). Each output row writes only `y(row)` and reads its own matrix row + gathers `v` at `colind` (which reaches into the halo → `v` must be halo-current first). Race-free, the inner sum sequential per row → **bit-identical on Serial AND OpenMP** (the verify confirmed: the SpMV-fed maps like `hbar_old=hbar` show `max|Δ|=0` on OpenMP). This is the L27 `compute_vel_nodes` gather shape, not the D22 scatter.
  - **The dot products are the FIRST `Kokkos::parallel_reduce` in the port — Serial bit-identical, OpenMP/CUDA climate-close (FP reduction associativity).** Kokkos Serial reduces sequentially in index order, exactly the C `for(row) s+=…` accumulation, so `max|Δ|==0`; OpenMP/CUDA use a tree/partial-sums reduction → the last ULPs differ → `α`/`β`/`residual` differ → the CG can even take a different iteration count → `d_eta` drifts at the FP floor. This is the documented GPU non-determinism source for the CG (D22 ladder), distinct from the scatter atomics. Fused the C's single `sp0/sp1` loop into one 2-accumulator `parallel_reduce` (`(int i, double& l0, double& l1)`) — one pass, same per-accumulator order, still Serial-bit-identical.
  - **⚠️ `compute_ssh_rhs` and `compute_hbar` are EDGE→NODE SCATTERS, not "node maps"** (the handoff mis-described `compute_ssh_rhs` as a node map). Both accumulate a per-edge transport flux into `ssh_rhs(n1)+=`/`(n2)-=` (resp. `ssh_rhs_old`) where the row endpoints can be halo nodes → `Kokkos::atomic_add` in natural edge order (D22, the `-(c1+c2)` negation parenthesised, L32). So the M4.2 OpenMP/CUDA divergence has TWO new D22 sources: the 2 SSH scatters AND the dot-product reduce. The verify isolated them — the scatter dominates (`ssh_rhs` `max|Δ|≈4e-11` absolute on OpenMP, ≈1e-17 relative on the large transport-divergence field) over the reduce (`d_eta`≈5e-19). **Whole-run OpenMP floor rose** from the pre-M4.2 `u`≈2.2e-19 (M2.5 vert_vel) to `T`≈1.8e-15 / `Av/Kv`≈2e-17 / `u/v`/`eta`≈1e-18 at step 20 — still ≪ the ≲1e-12 budget, NO blow-up (the climate-close PASS, not a regression). Derive the gather-vs-scatter call from the C BODY (L33): `compute_vel_nodes`/the SpMV accumulate into the receiver's OWN slot (gather, bit-identical); `compute_ssh_rhs`/`compute_hbar`/the M2 edge loops accumulate a shared edge flux into TWO different nodes (scatter, atomic).
  - **The CG owns its per-iteration `pp`/`rr`/`X` halo brackets (D21, the KPP/FCT pattern), host-staged.** Before each SpMV the gathered vector's halo must be current, so the device CG brackets each exchange `modify_device → sync_host → fesom_halo_exchange(h_checked) → modify_host → sync_device`, guarded `npes>1` (no-op at np==1 → Serial/CUDA-1GPU read the device view directly, bit-identical). On multi-GPU this stages device→host→MPI→host→device each iteration (regular MPI; GPU-aware device-pointer halos = M5). The generic bracket's leading `modify_device()` is essential for `rr`/`pp` (the kernel wrote them on device) and a harmless redundant device→host copy for the first `EXCH(X)` (X just pushed, Synced). **The C twin's exit `EXCH(X)` (solver.F90:507) is DROPPED** — the driver does `exchange_nod2D(d_eta)` immediately after the CG returns (the same unchanged X), so the two are idempotent → bit-identical; the CG ends with `modify_device(d_eta)` so the driver `sync_host`s the owned X before that halo.
  - **Two L30 cross-op re-pushes + one judgement call (eta_n stays HOST).** `update_vel` reads `d_eta` at the 3 element vertices (incl. HALO) → the driver re-pushes `d_eta` (`modify_host()+sync_device()`) after its nod2D halo. `compute_hbar` reads `uv` at `edge_tri` (interior elements only — so its owned device copy from `update_vel_kk` is already current — but the line-472 `uv` halo was written on the HOST, so re-push `uv` to keep the device coherent; cheap and robust vs relying on the partition guarantee). `ssh_rhs` needs NO re-push before the CG (the CG reads it at OWNED rows only — derive from the body). **eta_n (substep 11) stays HOST** (SYNC_MAP row 11 sanctions it): it is a trivial nod2D map reading `hbar`/`hbar_old` (which are `sync_host`'d for their halos + the verify anyway) and writing `eta_n` (which next step's substep-4 IN rail pushes to device regardless) — porting it would add a kernel + an `hbar` re-push to save a tiny nod2D copy, with no perf value. The CG round-trip (not eta_n) is what M4.2 closes; this is L36/L39 applied (a kernel goes to device when it has a device producer AND consumer or is substantial offloadable compute). The block verify replicates the host eta_n loop so it is still gated. **Verify = capture-before (L26) over all 7 outputs** (`ssh_rhs`/`d_eta`/`uv`/`ssh_rhs_old`/`hbar`/`hbar_old`/`eta_n`) — the driver snapshots them PRE-block, the verify restores them, runs the 4 C twins + eta_n in order, diffs, restores KK. The CG-as-deterministic-warm-start means the C CG re-runs the identical iterations from the restored `d_eta`. Verify key `ssh` (collision-free: `ssh` ⊄ any of eos/pp/kpp/pgf/vrhs/vfilt/ivisc/ale/gm/tradv/trdiff and none ⊄ `ssh`, plain `strstr`, L25). **The whole ocean step (substeps 1-14) now flows on the device** except the trivial host eta_n map, the salinity floor (L39), and the ice step (M4.3). **M4.2 acceptance PASSED** (job `25146822`): the 1-yr CORE2 Serial run (17280 steps × ~90 CG iters/step, real JRA55 forcing) reproduced all **13** monthly snapshots **ALL FIELDS BIT-IDENTICAL** to `/scratch/a/a270088/m1_accept/cref` — the device CG touches every step, so the per-step Serial bit-identity holds over a full year (the M2-acceptance discipline; the C twins are UNCHANGED, only `_kk` twins added).

- **L42 — M4.3a: the simple sea-ice coupling/maps (`ocean2ice` gather + `cut_off` clamp + the h_ice/h_snow diagnostic) moved to the device — the FIRST sea-ice kernels, establishing the ice device rails + the CORE2-based verify harness. Its headline is the VALIDATION-PATH finding, not the kernels (which are trivial).**
  - **⚠️ The sea-ice physics is FORCED-ONLY — the pi smoke does NOT exercise it.** The pi mesh is a warm idealized basin (T∈[10,15]°C, no freezing) → the ice tracers stay zero, so EVP/FCT/`cut_off`/diag run on TRIVIAL (zero) state, and thermo is gated off entirely (`forcing && jra && sr` is false without JRA55). So `FESOM_KK_VERIFY=<icekey>` on pi is only MEANINGFUL for `ocean2ice` (which reads the non-trivial ocean surface); for everything else the per-kernel gate needs a short **CORE2** run (SLURM, real JRA55 forcing → polar freezing → ice). `fesom_ice_initial_state` cold-starts ice at step 0 in polar regions, so even a ~120-step CORE2 run exercises the kernels non-trivially. This is the defining constraint for ALL of M4.3 (b/c/d): the verify is CORE2-SLURM-based, not pi-based — a slower cadence than M2. (`jobs/job_ice_verify_core2`: dist_16, 1 node, `FESOM_KK_VERIFY=icemap`, output → `/scratch`.)
  - **NO data-layer step for M4.3.** M1.4 already `Field`-wrapped EVERY persistent ice array (the `T_ICE`/`_DATA`/`_WORK`/`_THERMO` structs — uice/vice/srfoce_*/stress_*/sigma*/eps*/fct_*/inv_*/the thermo per-node arrays/etc.), explicitly "for M4.3". So unlike M2.5b-a/M2.6-a/M4.2-a, M4.3 goes straight to kernels — the device views are `ice->X_fld.d()` with no wrap.
  - **`ocean2ice` is the `compute_vel_nodes` (L27) shape + a map** → bit-identical Serial AND OpenMP (verified on pi, both backends `max|Δ|=0`, non-trivially — the ocean surface is active). Two race-free node kernels: (1) `srfoce_temp/salt/ssh` = surface tracer + `hbar` (per-node map, skip cavity → keep prior like the C `continue`); (2) `srfoce_u/v` = the area-weighted PRIVATE gather of surface `uv` over the node's surrounding elements. The C zeros `u_w/v_w` over `[0,N)` then gathers over `[0,myDim)`, so halo+cavity stay 0 — folded into ONE launch (`u_w(n) = (n<myDim && !cavity) ? gather : 0`). `cut_off`/diag are race-free per-node maps. **All bit-identical on OpenMP** (no scatter/reduce among the M4.3a kernels).
  - **The device-island-in-host pattern (the M2.2/§5 round-trip shape):** the M4.3a kernels are device islands inside the still-HOST ice step (EVP/FCT/thermo are host until M4.3b-d), so each round-trips device→host (IN rail push of the ocean state / ice values it reads, L28/L14 — the host EVP/FCT wrote the ice tracers via the raw alias; OUT `sync_host` for the host consumers). The round-trips compose away as the rest of the ice step moves to device. `dyn`/`tracers` are `const` in `fesom_ice_step` → localized `const_cast` for the IN-rail `modify_host()+sync_device()` (the forcing pattern).
  - **np>1 verify subtlety — halo-exchanged outputs must be diffed AFTER the driver halo.** `ocean2ice` halo-exchanges `srfoce_u/v` (the device kernel leaves their halo = 0 pre-exchange; the C twin halos internally). Diffing `[0,N)` BEFORE the driver halo would false-positive on the halo nodes at np>1 (KK 0 vs C owner-values). Fix: run the verify AFTER the driver `exchange_nod2D(srfoce_u/v)` so both KK and C have owner-halo values → `[0,N)` diff is clean. `cut_off`/diag have NO exchange (compute `[0,N)` directly) → diff-anywhere. Verify shapes: `ocean2ice`/diag are EOS-style (full overwrite from intact inputs → no capture-before); `cut_off` is L26 capture-before (RMW clamp). One key `icemap` gates all 3 (the `ale` pattern; collision-free token, L25).

- **L43 — M4.3b: the EVP sea-ice dynamics (the 120-subcycle rheology island) moved to the device — the CG/M4.2 pattern applied to sea ice, and the FIRST device kernel verified ONLY on CORE2 (the pi smoke can't exercise it; L42).** `FESOM_KK_VERIFY=evp` was `max|Δ|==0` for uice/vice/sigma11/12/22 across all 60 steps × 16 ranks on a CORE2 dist_16 run with ACTIVE ice (3840 verify lines, 0 nonzero; real ice drift — uice max 0.95 m/s, 33416 ice nodes; 120 subcycles/step). pi shows 0 trivially (zero ice). Five things worth recording:
  - **The 120-subcycle loop is HOST loop control + DEVICE per-subcycle kernels + a per-subcycle uice/vice halo bracket — structurally identical to the M4.2 CG** (and SIMPLER: the subcycle count is FIXED at `evp_rheol_steps=120`, no convergence scalar, so no `parallel_reduce`/`MPI_Allreduce` in the loop). Each subcycle: `stress_tensor_kk` (per-element, sigma RMW, race-free) → `stress2rhs_kk` (zero-map + element→node SCATTER + finalise-map) → save-old (map) → velocity-update (per-node implicit drag/Coriolis solve, map) → the coastal BC + halo bracket. The setup (Steps 1-4) is 4 device kernels (zero / mass / `ice_strength`+elevation-gradient-rhs SCATTER / area-divide).
  - **Two element→node SCATTERS (D22, `atomic_add`):** Step 3's elevation-gradient `rhs_a/rhs_m` and `stress2rhs` Loop 2's stress-divergence `u_rhs/v_rhs` (`[n] -= X` → `atomic_add(&v(n), -(X))`, L32). **Serial bit-identical** (ordered atomics); OpenMP/CUDA climate-close — and the EVP COMPOUNDS the scatter perturbation over 120 subcycles, so the OpenMP/CUDA divergence on ACTIVE ice is larger than a single scatter (the EVP relaxation should damp it; to be confirmed at the M4 CORE2 OpenMP/CUDA acceptance — the pi smoke can't show it since ice is zero). Everything else (stress_tensor, velocity-update, save-old) is a race-free per-element/per-node map.
  - **The coastal BC stays the verbatim C edge loop on the HOST, folded into the halo bracket's host phase.** It zeroes uice/vice at open-boundary edge endpoints via `partit->myList_edge2D[ed] > mesh->edge2D_in` — and `myList_edge2D` is NOT `Field`-backed (it is MPI partition data, not mesh/ice state). Rather than wrap it or build a device boundary mask, the BC runs on the host right after `sync_host(uice/vice)` and before the halo (uice/vice are host-resident there for the halo anyway): `velocity-update_kk → sync_host(uice/vice) → [host coastal BC] → halo (npes>1) → modify_host+sync_device`. On Serial the syncs are no-ops (the BC writes host==device) but they clear the Device auth flag so SYNCCHECK's `h_checked` doesn't fire. **At np>1 (production multi-GPU) the per-subcycle uice/vice round-trip is needed for the halo regardless, so the BC folding is free; only np=1 single-GPU pays an extra round-trip → a device-boundary-mask BC is the M5 optimisation.** (`bc_index_nod2D` can't be reused — it's built with the serial `edge_tri[*+1]>=0` detection, which the EVP comment flags as wrong at multi-rank.)
  - **`stress2rhs` Loop 1 zeros `u_rhs/v_rhs` over OWNED only (the C bound) — the halo `u_rhs` accumulates scatter garbage across subcycles but is NEVER read** (Loop 3 + the velocity update are owned-only). Ported faithfully (zero `[0,myDim)`, scatter `[0,N)`, finalise `[0,myDim)`); the unused halo garbage matches the C bit-for-bit on Serial (same stale history + same atomic order) and is irrelevant to uice/vice. (The L37 FCT "halo stays at its prior value" rule again — derive the zero/read extent from the C body.)
  - **Verify = capture-before (L26) over uice/vice + sigma11/12/22** — the rheology state is RMW *across subcycles AND across ocean steps* (each subcycle's `stress_tensor` reads the prior sigma; the next ocean step's EVP continues from this step's uice/vice/sigma). The driver snapshots those 5 PRE-EVP, the verify restores them and runs the whole C `fesom_ice_evp_dynamics` (setup + 120 subcycles), diffs, restores KK. The inputs (a/m/ms, srfoce_*, stress_atmice_*) are intact (EVP doesn't modify them). The IN rail must `sync_host(uice/vice/sigma)` on OUT so next step's IN-rail push isn't stale (sigma has no other consumer — it would otherwise drift device-only). Verify key `evp` (collision-free, L25).

- **L44 — M4.3c: the sea-ice FCT advection (`ice_TG_rhs` + `ice_fct_solve`) moved to the device — the M2.6 ocean-FCT analogue but 2-D (single surface layer), reusing the M4.2 SSH-CSR machinery for the mass-matrix solves.** `FESOM_KK_VERIFY=icefct` was `max|Δ|==0` for a_ice/m_ice/m_snow across all 60 steps × 16 ranks on a CORE2 dist_16 run with ACTIVE ice (`job 25158693`, 960 verify lines, 0 nonzero — alongside evp 960 + icemap 2880, all 0; ~116 CG iters/step → real ocean dynamics); pi shows 0 trivially (zero ice). Five things worth recording:
  - **The FCT is THREE structural patterns already in the toolkit, composed.** (1) `tg_rhs` = the EVP `stress2rhs` shape: zero-map `[0,myDim)` + element→node SCATTER (`atomic_add`) into `values_rhs`, no halo (the C never exchanges rhs; consumers read OWNED). (2) `ice_solve_low/high_order` = the M4.2 **CG `cg_spmv` per-row CSR gather** — `sum += mm(k)·v(colind(k))` over the row, race-free (each row writes its own output) → **bit-identical Serial AND OpenMP** (not a scatter); `ice_solve_high_order`'s consistent-mass iteration is the **CG/EVP host-loop**: a host `for it<num_iter_solve-1` over device sweeps (gather-correct + map + copy-back) with a per-iter `dvalues` halo. (3) `ice_fem_fct` (×3 logical tracers) = the Zalesak limiter = the closest M2.6 ocean-FCT shape but flat (no `[N*nl]`): init maps, per-element antidiffusive fluxes (each elem writes its OWN 3 `fct_fluxes` slots → race-free), cluster min/max (CSR gather), `icepplus/icepminus` element→node SCATTER, correction map, limit map, apply (overwrite `vals=valuesl` + element→node SCATTER `vals += fluxes`).
  - **`fct_massmatrix` shares the `ssh_stiff` CSR sparsity → reuse the M4.2-a device CSR verbatim.** The Fortran `nn_pos/nn_num` neighbour lookup is the `ssh_stiff` `rowptr/colind` (already pushed once by `fesom_ssh_preconditioner`); only the matrix *values* (`fct_massmatrix`) need a new one-shot push, added at the end of `fesom_ice_mass_matrix_fill` (set-once, never updated → valid all run, the M4.2-a pattern). No new CSR build, no data-layer step (M1.4 wrapped `fct_*` already, L42). The ice step runs BEFORE the ocean CG each iteration, but the CSR is read-only in both, so the shared device copy is fine.
  - **THREE element→node SCATTERS (D22, `atomic_add` in natural element order):** `tg_rhs` assemble, `fem_fct`'s `icepplus/icepminus` +/- sum (a *conditional* atomic — `flux>0 ? atomic_add(&icepp) : atomic_add(&icepm)`), and `fem_fct`'s final `vals += limited_flux` apply. **Serial bit-identical** (single-thread ordered atomics == the C sequential `+=`); OpenMP/CUDA climate-close on ACTIVE ice (pi shows 0 — zero ice). Everything else — the antidiffusive-flux build, the limit step, the CSR-gather solves — is a race-free per-element/per-row map (each entity writes only its own slot/row).
  - **Capture-before needs ONLY the 3 `data[*].values`, not the FCT-internal scratch.** Each `fem_fct` *overwrites* its tracer's `values` (`=valuesl` then `+=fluxes`), but the upstream `tg_rhs`/LO/HO/its-own-`fem_fct`-antidiff all read the ORIGINAL `values` first, and `fem_fct(A)` doesn't touch `data[MICE].values` — so the 3 values are each "read-original then overwrite" within their own `fem_fct`, and snapshotting them PRE-FCT is sufficient (L26). `values_rhs/valuesl/dvalues/fct_*` are FCT-internal: fully recomputed by the C twin (tg_rhs zeroes rhs, LO/HO overwrite valuesl/dvalues, fem_fct re-inits tmax/tmin/icepplus/icepminus) → **not captured AND not pushed** in the IN rail (only `uice/vice` + the 3 `values`). The `tg_rhs` halo-`rhs` "stale-summed but unconsumed" rule (L37/L43) means the verify's C twin scatters onto the device-tg_rhs halo garbage — harmless, `rhs[halo]` feeds nothing, `values` is unaffected. Verify runs the C twin on ALL ranks (its LO/HO/fem_fct halos are collective).
  - **`INFINITY` as the cluster min/max sentinel compiles under nvcc and is bit-identical.** The C inits `lo=+INFINITY, hi=-INFINITY`; every `ssh_stiff` CSR row is non-empty (the diagonal `colind[rowptr[row]]==row` is always present), so the sentinel is always replaced by a finite neighbour value, and min/max *select* an actual value (no arithmetic) → ±INFINITY, ±1e30, or first-element init all give the identical finite result. Kept `INFINITY` for fidelity to the C. The internal-halo count is high (21 D21 brackets per FCT call: 3 `valuesl` + 3 first-approx `dvalues` + 3×2 iter `dvalues` + 3×(2 `icepplus/icepminus` + 1 `values`)) — all OWNED by the FCT, `if(npes>1)`-guarded host round-trips (eDim=0 at np=1 → no halo), matching the C `exchange_nod` count one-for-one. GPU-aware MPI to fuse them is M5.

- **L45 — M4.3d-a: the sea-ice thermodynamics (the per-node column physics) moved to the device — a single race-free map over [0,N), and the FIRST device kernel to need a NON-ice input Field-wrapped (the 8 JRA55 atmospheric arrays).** `FESOM_KK_VERIFY=icethermo` was `max|Δ|==0` for m_ice/a_ice/m_snow/t_skin/flx_h/flx_fw/thdgr across all 60 steps × 16 ranks on a CORE2 dist_16 run with ACTIVE ice (`job 25159173`, 960 lines, 0 nonzero; the 960 printed lines *prove* the thermo block is entered — on pi it is 0 lines since `forcing && jra && sr` is false). Five things worth recording:
  - **The column physics is per-node and race-free → a single map over [0,N), bit-identical Serial AND OpenMP (NO scatter, NO reduction)** — the M2.7 tracer-diff-TDMA shape. Each node reads its own a/m/ms/t_skin/thdgr + the atmospheric forcing and writes only its own outputs; no node touches another. Loop 1 (the `ustar` friction velocity, myDim) owns its `ustar` nod2D halo bracket (Loop 2 reads `ustar` at halo n); Loop 2 (the `therm_ice` call, [0,N)) has no halo (the C bound is myDim+eDim, halo-inclusive, and downstream reads it directly).
  - **`therm_ice`/`obudget`/`budget`/`flooding`/`tfrez` became `KOKKOS_INLINE_FUNCTION` device twins (the `bc_surface_kk`/`kpp_wscale_kk` precedent), DUPLICATED from the host C (which stays the D19 oracle).** They are pure arithmetic (scalars + pointer-to-local outputs — the pointers become device-stack addresses inside the lambda, fine), so the port is mechanical: add the qualifier, `sqrt/exp/pow → Kokkos::`, drop `FESOM_CHECK` (the config invariants snowdist==1/open_water_albedo==0 are CORE2-fixed), keep `pow(tk,4.0)` vs `tk*tk*tk*tk` EXACTLY as each host site wrote it (Serial `Kokkos::pow`==libm `pow` → bit-identical). The verify (max|Δ|==0) is what catches a transcription slip; sharing code with the host would make the verify trivial, so duplication is the point.
  - **The `fesom_ice_thermo` struct can't cross to the device (it holds `fesom::Field` members) → copy its ~30 scalar physics constants into a POD `IceThermC`, fill it on the host, capture it BY VALUE in the lambda, pass it `const&` to the device helpers.** This is the general recipe for "a kernel needs a struct-ful of constants whose struct also owns Views": mirror the scalars in a trivially-copyable POD. (A host pointer to the Field-bearing struct would be a host address — unusable on the GPU.)
  - **JRA55 was the ONLY non-Field thermo input → Field-wrap its 8 physics arrays (the M1.4 pattern, deferred from M1 because jra-on-device wasn't needed until now).** forcing/ice/mesh were already Field-backed. The 8 arrays are `calloc`-own-storage filled in-place by the per-step time interpolation (via the raw alias), so the IN rail does `modify_host()+sync_device()` each step (the forcing-producer pattern). ⚠️ **jra is CORE2-only — pi never allocates it (`use_jra=0`), so the wrap AND the thermo kernel are exercised ONLY on the CORE2 SLURM gate** (pi just proves no build/ocean regression). Adding Fields to the struct forced `fesom_jra55_{init,free}`'s `memset(jra,0,…)` → `*jra = fesom_jra55{}` (memset on a Field-bearing struct is UB, L13/D13).
  - **The `*/` comment trap: `srfoce_*/values` inside a `/* … */` block closes the comment early** (the `*/`), turning the rest into code → a wall of "X not declared" errors pointing at a *comment* line. Bit me twice. When a comment lists glob-ish array names, don't write `prefix_*/suffix`. (Verify is the gate, but this is a compile-time porting gotcha worth the muscle-memory.) Verify = capture-before (L26) the 5 RMW inputs (m_ice/m_snow/a_ice/t_skin/thdgr — read as `therm_ice` inputs, then overwritten); the C twin's `ustar` exchange is collective → run it on ALL ranks. Key `icethermo`.

- **L46 — M4.3d-b: `oce_fluxes` (the ice→ocean flux coupling) moved to the device — the LAST sea-ice kernel, completing M4.3 (the whole sea-ice step is now device-resident).** `FESOM_KK_VERIFY=iceflux` was `max|Δ|==0` for heat_flux/water_flux/virtual_salt/relax_salt across all 60 steps × 16 ranks on a CORE2 dist_16 ACTIVE-ice run (`job 25159374`, 960 lines, 0 nonzero; all five ice keys evp/icemap/icefct/icethermo/iceflux green together). Four things worth recording:
  - **It is maps + the FIRST sea-ice GLOBAL reductions.** `heat_flux/water_flux = -flx_h/-flx_fw`, `virtual_salt = rsss·water_flux`, `relax_salt = surf_relax_S·(Ssurf−S)` are per-node maps over [0,N); the salt-conservation correction is two `integrate_nod_2D` calls (Σ data(n)·areasvol(node3d(n,ulev−1)) over OWNED + an `MPI_Allreduce`) — **the M4.2 `cg_dot` shape: `Kokkos::parallel_reduce` + the scalar Allreduce**. Serial reduces in index order == the C loop → **bit-identical**; OpenMP/CUDA use a tree reduce → climate-close (D22). The net-subtract is an owned-node map (⚠️ literal-port subtlety: the virtual_salt subtract skips cavity `ulevels_nod2D>1`, the relax_salt subtract does NOT — match each C loop exactly, L33).
  - **The forcing OUT is the "→host(forcing)" device-island handoff.** oce_fluxes OVERWRITES forcing heat_flux/water_flux/virtual_salt/relax_salt (device), then the kernel `sync_host`s them and does its 4 halos on the HOST; the OCEAN step's existing tracer-diff IN rail re-pushes the halo'd host forcing → device. So the ice step writes forcing on the host, the ocean step reads it on the device — the host is the handoff buffer (an M5 fusion target). The `flx_h/flx_fw` inputs, by contrast, are **device-current from the M4.3d-a thermo (device→device, no push)** — the first ice device→device handoff (thermo kernel → oce_fluxes kernel, both in the same step).
  - **EOS-style verify (no capture-before): oce_fluxes is a full overwrite from intact inputs** (flx_h/flx_fw/S/Ssurf — it doesn't read its own outputs except water_flux, written-then-read within the same call). So snapshot the 4 KK forcing fields, run the C twin (recomputes from the intact inputs; its `integrate_nod_2D` Allreduce + the 4 halos are collective → run on ALL ranks), diff, restore KK. Key `iceflux`.
  - **M4.3 is COMPLETE — the whole sea-ice step (ocean2ice / EVP / FCT / thermo / oce_fluxes) is device-resident, and with it the WHOLE MODEL** (ocean + ice) save the trivial host `eta_n` map + the salinity floor (L39). The five-kernel CORE2 verify harness (`jobs/job_ice_verify_core2`, keys evp,icemap,icefct,icethermo,iceflux) is the standing sea-ice gate; the M4 acceptance (1-yr CORE2 Serial bit-identical to cref) is the milestone close → tag `m4-full-device`.

- **L47 — M5.1a: GPU-aware-MPI on-device halo exchange (`src/fesom_halo_device.{hpp,cpp}`) — the gate, the data-path validation insight, and the *measured* payoff (which is modest, and that itself is the finding).** Replaces the D21 host-staged bracket (full-field device→host PCIe sync, host pack, MPI, host unpack, full-field host→device sync) with: device pack (`slist` gather) → GPU-aware MPI on **device** send/recv buffers → device unpack (`rlist` scatter). Six things to record:
  - **THE GATE was a MPI-module swap, not code.** Levante's `openmpi/4.1.2-gcc-11.2.0` (env.sh) is built `--without-cuda` and its UCX 1.12 has **no cuda transports**, so a device pointer into `MPI_Sendrecv` **SEGFAULTS** (proven, `jobs/job_mpi_cuda_smoke`, job 25166379). The NVIDIA-built **`openmpi/4.1.5-nvhpc-24.7`** (+ `netcdf-c/main-openmpi-4.1.5-nvhpc-24.7`) IS CUDA-aware (UCX `cuda_copy`/`cuda_ipc`, `smcuda` BTL, `--with-cuda`) → device-ptr MPI **WORKS** (job 25166531). build-cuda now sources **`env_cuda.sh`** (clean reconfigure: `find_package(MPI)` + `nc-config` pick up the loaded modules; struct layout unchanged so no `touch`). Serial/OpenMP keep env.sh's 4.1.2 — **Approach B: the bit-identity oracle's toolchain is untouched** (device path is `#ifdef KOKKOS_ENABLE_CUDA`). **§4.1's "prove it first" smoke saved the session** — without it I'd have built the whole halo layer on a SEGFAULTing MPI. ⚠️ `gdr_copy` (GPUDirect-RDMA kernel mod) is ABSENT on Levante GPU nodes → inter-node GPU↔GPU stages the (small) PACKED halo through pinned host via `cuda_copy` (still a win; not true GPUDirect). Device-ptr MPI works inter-node anyway (CORE2 dist_8, 2 nodes — verified).
  - **The index math collapses to a flat gather/scatter.** The send buffer is packed in `slist` order and the file convention gives `sptr[0]==rptr[0]==1`, so the per-PE MPI offset `(sptr[k]-sptr[0])*stride` equals the running list index `g*stride`. ⇒ PACK is `send[g*stride+c]=field[(slist[g]-1)*stride+c]` over the whole `slist`, no per-PE bookkeeping in the kernel; UNPACK is the `rlist` analogue and is **race-free** (each halo node appears once in `rlist`) → no atomics. The MPI loop keeps the host path's exact tag / PE order / `MPI_DOUBLE` counts (the bytes moved are identical). Kind-generic via `stride = nl*nc` (one function does NOD2D/NOD3D/ELEM2D/ELEM3D/ELEM2D_FULL); device `slist`/`rlist` are a one-shot copy; send/recv Views grow-on-demand and are **freed before `Kokkos::finalize()`** (`fesom_halo_device_free()` in main — `fesom_halo_free_buffers()` is never called, so don't piggyback). `Kokkos::view_alloc(WithoutInitializing,"label")` reads the `const char*` as a wrap-ptr → use `View("label",n)`; a `const char*` *variable* needs `std::string(label)`.
  - **The dispatch helper `fesom_halo_field(Field&,kind,nl,nc,p)`** replaces the boilerplate bracket: device path under CUDA (no PCIe sync), the EXACT legacy host-staged bracket on Serial/OpenMP. **`FESOM_HOST_HALO=1` forces the legacy path inside the CUDA build** — the A/B regression toggle (same binary, diff only the data path). It also drops the legacy path's pointless npes==1 round-trip (device already holds the data).
  - **VALIDATION is NOT byte-identity — it's "at the run-to-run noise floor".** I predicted device==host byte-for-byte (pure data move) and was WRONG: CUDA is non-deterministic **run-to-run** because the upstream atomic-scatter kernels (`momentum_adv`/`vert_vel`/…) reassociate, perturbing `ssh_rhs` → the CG → everything by ~1e-13. So the gate is **host-vs-device ≈ host-vs-host** (run host-staged twice for the floor). Result (pi dist_2, full-flip build): host-vs-dev Av/Kv/T/density **≤** host-vs-host (no NEW divergence; data path, not arithmetic) → PASS. `diff_snap.py`'s "DIVERGENCE" verdict is its CUDA-blind heuristic (CUDA is never bit-identical run-to-run). ⚠️ Av/Kv can show isolated ~1e-11 excursions in *either* comparison — a KPP mixing **threshold-flip** off sub-ULP u/v noise (the known divergence class, [[reference-cuda-eos-divergence]]), not a stride bug; the back-to-back repro disambiguates.
  - **Measure with a startup-free INTERNAL timer, not wall-subtraction.** `(wall@105−wall@5)/100` was swamped by ~2.7 s of variable CUDA-context/JIT/file-cache startup (it hid the signal: it made CG-only look 2% *slower*). Added a `MPI_Wtime` loop timer in `fesom_main.cpp` (fence+barrier at both ends, excludes 5 warmup steps, prints `s/step`): two reps now agree to <0.2 %.
  - **THE PAYOFF IS MODEST (~8%), AND THAT IS THE FINDING.** CORE2 dist_8 (internal timer, 2 reps): host-staged **0.780** → device-halo **0.716 s/step** = **8.2% faster**, from flipping CG (nod2D) + momentum (`uvnode_rhs` nod3D, `u_b/v_b` elem3D) + gm (`tr_xy` elem2D_full, `tr_z` nod3D) + ice FCT. The "halo-bound" premise was only *partly* right: the win is the **large nod3D fields** (~48 MB, the PCIe sync is real); the **small nod2D hot loops (CG ~200 exch/step) barely benefit** (1 MB field — PCIe-cheap; the device path still fences). And the *other* big nod3D halos (**kpp `diffK`/`blmc`, eos pressure**) are **host-bound by their `fesom_smooth_nod3D`** (a host op forcing the round-trip regardless) → NOT device-halo candidates. So ~8% is near the **ceiling** for this optimization on this config. The real GPU lever is elsewhere: port the kpp/eos **smoothing** to device (kills those round-trips), batch/fuse the CG's ~1000 kernel launches/step, or a **bigger mesh** to fill the A100s (M3.1 note). EVP (120×uice/vice, nod2D + a device coastal-BC) is left host-staged (nod2D ⇒ modest, + verify risk) — flippable next but low payoff. Don't oversell GPU-aware MPI as "the" unlock — it's a correct, free-now ~8%.
  - **Bigger-mesh follow-up (farc, ~638k nodes × 48, ~5× CORE2) settles the "fill the A100" question — and the answer is the CG, not mesh size.** farc on GPU (dt=900, ~200 CG iters/step): device-halo gain **grew to ~12%** (dist_4 7.28→6.38; dist_8 3.73→3.30 — bigger nod3D fields ⇒ more PCIe eliminated, confirming the mechanism), and the GPU **strong-scales near-linearly** (dist_4→dist_8 = 1.93× on 2× GPUs ⇒ A100s not saturated). **BUT node-for-node the CPU is still ~7.4× faster** (farc dist_256 = 2 compute nodes = 0.445 s/step vs dist_8 = 2 gpu nodes = 3.30). ⚠️ I FIRST BLAMED THE CG — and that was WRONG (corrected in M5.2). A CG-share timer (`FESOM_CG_PROFILE=1`) shows the **CG is only 1–5 % of the step** (host 1.75 ms/iter, dev 0.305; ~219 iters/step) — the "16.5 ms/iter" conflated the CG iterations with the *whole* step (the CG runs once/step, ~0.38 s host / 0.067 dev). So a bigger mesh grows the device-halo payoff AND the GPU strong-scales, but does NOT flip GPU↔CPU — **and the cause is NOT the CG**. The real bottleneck is the **bulk ocean+sea-ice kernels (~95–99 % of the step)**, un-profiled (likely the per-column TDMA solves [KPP/impl_vert_visc/tracer-diff/GM-gamma], the many FCT launches, and the host-bound kpp/eos smoothing + S-floor + ice host parts). **L48 = the lesson: measure the share before optimizing — I "optimized" the CG (two bit-identity-safe fusions: SpMV+dot, sp0/sp1→one Allreduce; Serial np1+np2 still byte-identical) and it was immaterial because the CG is tiny.** The device-halo's CG win (5.7× faster CG) is real but ~⅓ of its 0.89 s/step gain (the other ⅔ = ocean nod3D halos). NEXT (the actual path to a GPU win): a per-substep timer in `fesom_step.cpp` → find the dominant kernels, optimize THOSE. Jobs `job_farc_gpu_np1`, `job_gpuaware_time_farc_dist{4,8}`, `job_farc_cpu_dist256`; data GPU_FIDELITY §M5.1b/§M5.2.

- **L48 — M5.3/M5.4/M5.5: profile the step, then flip the host-staged ocean halos to device-halo (the real PCIe win).** M5.3 (per-phase timer `FESOM_STEP_PROFILE=1` + `nsys`): the GPU step is **ocean 82%, sea-ice only 7.4%** (the Fortran "it's the ice" intuition does NOT carry to the GPU — ice is 2-D/cheap), and the ocean's 82% is ~half **3-D kernels** (FCT ~30%, momentum ~15%, diffusion ~14%, GM ~10%) + ~half **full-field PCIe** (`cudaMemcpy` ~83% of API time) = the host-staged ocean halos + EOS/KPP `smooth_nod3D` round-trips. M5.4/M5.5 flipped them to `fesom_halo_field` (+ the device smoother for B): **CORE2 dist_8 host-staged 0.772 → device-halo 0.577 = ~25%** (uv_rhs +8%, FCT +8%, pgf/uvnode +2%, bvfreq +2.5%). Three flip patterns + their gotchas:
  - **Self-contained bracket** (CG `exch`, FCT internal halos): `modify_device; sync_host; exchange; modify_host; sync_device` all in one place → replace wholesale with `fesom_halo_field`. Trivial.
  - **Split-rail** (the M1.5 ocean halos — uv_rhs/pgf/uvnode): the OUT-rail `sync_host` sits at the producer, the `modify_host()+sync_device()` IN re-push at the consumer (sometimes substeps later). Replace the OUT-rail+halo with `fesom_halo_field` AND **remove the downstream IN re-push** (else it clobbers the device halo with stale host). Trace each field's consumer(s).
  - **Smoothing (lever B)**: the field's halo is host-bound because a host `smooth_nod3D` runs between sweeps → port the smoother to a device kernel (`fesom_smooth_nod3D_kk`: per-node patch gather, 2 race-free kernels [gather then scale — separate to avoid the arr read-write race] + device-halo between sweeps) + remove the consumer re-pushes. For a multi-channel field (KPP `blmc`[3]) add a `base` element-offset to the device-halo + smoother so a slab is exchanged/smoothed on its own. **Surprise: blmc was +13%** (vs bvfreq's 2.5%) — its 9 sweeps ran **single-threaded on the host** (the GPU build's Serial host), a large hidden cost **`nsys` never showed** (it profiles GPU kernels only; nsys ranked KPP ~1%) but the phase profiler counted inside the ocean's 82%. **Lesson: a host kernel's COMPUTE cost is invisible to a GPU-kernel profiler — use the host+device phase timer to find it.** Lever B ≈ 15% total; cumulative device-halo+smoother **0.716 → 0.503 s/step (~30%)**.
  - **Safety**: the per-kernel verifies read the RAW alias (not `h_checked()`) → no SYNCCHECK assert; on Serial every sync is a no-op (host==device) so removing them is a no-op there. Validate EACH flip: Serial np1+np2 bit-identical + SYNCCHECK-clean + **CUDA A/B (device-halo vs `FESOM_HOST_HALO=1`) at the run-to-run noise floor**.
  - ⚠️ **The I/O-staleness trap (cost me a silent M5.4b regression).** Flipping a field that is ALSO a snapshot output (pgf, bvfreq, density, Kv, Av) removes the OUT-rail `sync_host` the I/O gather relied on → the field is DEVICE-authoritative at I/O and `h_checked()` reads the STALE host copy on CUDA (pgf_x snapshot was 1.4e-3 vs the correct 2.0e-6; the MODEL was fine — device-resident, consumed on-device — only the diagnostic OUTPUT was stale). Fix: a pre-I/O `sync_host` of the device-resident I/O-output fields, gated on the snapshot step (`fesom_main.cpp`). **Neither the Serial gate nor SYNCCHECK catches it** (on Serial `fesom_halo_field` leaves the field synced), and I missed it because the A/B check grepped a SUBSET of fields → **always diff ALL output fields.**

## C. Process lessons

- Validate at **every** step on Serial bit-identity; commit per milestone-step; **tag** milestones.
- Record **provenance** (C source SHA, exact build/run recipe) so any reference is regenerable.
- nvcc full-model compiles are slow → run them **in the background**; iterate config on the login
  node, run device smokes via `srun -p gpu-devel -A ab0995 --gres=gpu:1`.
- **A phantom multi-rank divergence debugging ladder (from the M1.3 vader-CMA saga — follow it
  BEFORE suspecting the port; it would have saved hours).** When np=1 is bit-identical but np>1
  diverges after a change that *shouldn't* affect results (a storage/struct refactor):
  1. **Confirm it's reproducible and your change caused it**: rebuild the pre-change HEAD, run np>1,
     diff vs the oracle (it should match). Then your change vs HEAD. (Don't trust a possibly-stale
     scratch oracle — regenerate it.) Beware **incremental-build staleness** after a struct-layout
     change: `touch src/*` to force a full recompile, or the divergence may be a stale `.o`.
  2. **Bisect compute-vs-I/O**: dump the OWNED state (`field[0..myDim*stride)` per rank) right after
     the producing kernel, both builds, and `cmp`. **Byte-identical owned ⇒ the port is correct and
     the divergence is in the gather/transport/output path, not the physics.** (This single check
     reframes the whole hunt.) A coarse sum+max checksum can match by luck — `cmp` the raw bytes.
  3. **ASan clean ≠ no memory problem.** ASan catches OOB / use-after-free but **not uninitialized
     reads** and **not transport-layer artifacts**. A clean ASan run does not exonerate the gather.
  4. **Suspect the MPI transport.** If per-rank SENDS + the gather plan (counts/displs/lists) are all
     byte-identical yet `MPI_Gatherv`'s `recv` differs between builds, it's the transport, not your
     code — see L18 (login-node `vader` CMA). Try `OMPI_MCA_btl_vader_single_copy_mechanism=none`.
  5. **Don't be fooled by "data-dependent" divergence**: a transport/gather bug only *shows* in
     non-zero fields, so it looks like it singles out specific variables (Av/Kv/uv diverged, the
     ~0 `pgf` didn't) — that's a visibility artifact, not a clue about which kernel is "wrong".
- **np=1 bit-identity is necessary but NOT sufficient.** np=1 skips `scatter_mesh`, every halo
  exchange, and the cross-rank gather, and has no halo/eXDim storage — so it cannot test those code
  paths or that region's init. The np≥2 gate (D14, CMA-off) is a *separate* obligation for any change
  that touches storage size/layout or MPI.

- **L49 — M5.6/M5.7/M5.8: a per-substep profiler finds the HIDDEN host-staged sinks the GPU-kernel-only profiler (nsys) can't see, then flip them.** A fence-bounded host+device per-substep timer (`src/fesom_profile.{hpp,cpp}`, `FESOM_STEP_PROFILE=1`) marking every ocean substep + ice phase gave the full-model CORE2 dist_8 breakdown (FCT 17%, KPP-mixing 11%, GM 10%, SSH 9%, EVP 9%) and surfaced two `blmc`-class sinks nsys missed because the cost was host-side `smooth_nod3D`/halo staging, not a GPU kernel. M5.7 (`5b67666`) kept KPP `diffK`/`viscA`/`Kv`/`Av` device-resident through KPP→mo_convect→ivisc→trdiff (3_mixing 11%→6%); M5.8 (`c2fa25e`) moved the EVP per-subcycle coastal-BC to a device kernel over a precomputed per-node `evp_coastal_mask` and kept `uice`/`vice` device-resident across the 120 subcycles (ice_dyn 9.7%→7.9%). **0.503 → 0.476 → ~0.464 s/step.** ⚠️ A cached `static Kokkos::View` (the EVP mask) destructs AFTER `Kokkos::finalize()` → SIGABRT at exit; FIX = file-static + an explicit free-function (`fesom_ice_evp_free()`) called before finalize — the same rule as `fesom_halo_device_free`. Full detail in `docs/GPU_FIDELITY.md` §M5.6–§M5.8.

- **L50 — M5.9-pin: to tell a REAL host reader of a device-resident field from mere chaos-sensitivity to a sync FENCE, poison the host copy with NaN but KEEP the sync.** The M5.9 fix `sync_host`'d four device-halo'd fields (`bvfreq`, `pgf_x/y`, `uvnode`, `uv_rhs`) after a leave-one-out (toggle each sync on/off) said all four were needed (each, left un-synced, decorrelated the CORE2 CUDA run to ~0.4 vs the Serial oracle). **That leave-one-out was confounded:** removing a `sync_host` removes a device FENCE, which on a chaotic CUDA run reshuffles atomic-scatter/launch ordering enough to slide the trajectory to the same ~0.4 attractor — WITHOUT any stale read. The clean discriminator: at each site **keep the `sync_host` (so device scheduling is byte-identical to the fixed run) and overwrite the host copy with NaN AFTER it, without `modify_host`** (device stays correct → every DEVICE consumer is unaffected). Only a genuine HOST read then sees the NaN, which propagates to the snapshot. Result (CORE2 dist_8 CUDA, ICE active): poisoning `{bvfreq,pgf,uv_rhs}` → model **byte-for-byte the clean run** (the netCDF was even healed by the snapshot-gated pre-I/O `sync_host`, L48; only the read-only min/max *print* saw the NaN); poisoning `uvnode` → NaN wind stress → NaN momentum → `CG_kk: pp·App = nan` abort. **So only `uvnode` has a real host reader** — `fesom_bulk_compute` (the JRA55 bulk wind-stress formula, surface row, every CORE2 step), which is exactly why the whole class was invisible on pi (pi uses analytical forcing, never calls bulk; `bvfreq`/`pgf`/`uv_rhs` are read only by device kernels — KPP/mo_convect/GM, compute_vel_rhs, impl_vert_visc/compute_ssh_rhs). Six things worth recording:
  - **A documented "all N required" from toggling syncs is suspect on a chaotic backend** — toggling a sync changes BOTH data freshness AND the fence/scheduling. Separate them: corrupt the data while holding the fence constant.
  - **NaN is the right poison**: sticky + obvious (a clear crash/NaN if read), inert if unread. Distinguishes "read → NaN out" from "fence-perturbation → 0.4 with finite output."
  - **A device-resident field's host min/max PRINT is allowed to be stale** (cosmetic, like `Kv`/`Av` already are) — only the saved netCDF must be fresh, via the snapshot-gated pre-I/O `sync_host` (L48), which runs AFTER the print. Don't add a per-step sync just to keep a log line current.
  - **Raw aliases (`aux->bvfreq`, `dyn->uvnode`) are cached `h()` pointers bound once at alloc**, so instrumenting `Field::h()` does NOT catch alias reads — but poisoning the underlying host buffer (via that same pointer) does.
  - **The fix is to DELETE the 3 placebo syncs and keep `uvnode`'s** — 3.1% recovered (CORE2 dist_8 clean, same-node, 2 runs: all-syncs 0.4931 → 0.4777 s/step), the measured cost of the 3 placebo PCIe copies (4 fields). That is near the ceiling: `uvnode` genuinely needs a sync, so the full M5.9 +6.2% was never fully recoverable.
  - **A surface-only `uvnode` refresh (bulk reads only nz=0) bought only +0.5%** (0.4752 = 3.6%) over the full sync (0.4777 = 3.1%): its per-step `Kokkos::View` scratch alloc + host unpack loop nearly cancel the ~94 MB PCIe saving. Not worth the custom kernel; a persistent-buffer version (+ finalize-free, L49) is a future micro-opt.

- **L51 — 2026-05-28: the GPU fidelity gate's cached Serial oracle has a freshness budget.** During M5.11 profile pass, `scripts/gpu_fidelity_gate.sh` FAILED with worst-case T=1.28 vs the cached `/work/.../kokkos_gpu_runs/serref_core2/snap_000020.nc` (built 2026-05-27 20:46 from `master @ c2fa25e`, pre-M5.9 FIX). The same binary that passed M3.2 1-yr CUDA + 2-yr OMP earlier *the same day* was "failing" the 20-step gate. **Diagnosis**: rebuilding build-serial against current `master @ 466ea3e` (post-M5.9 FIX `6ba27e9` + M5.9-pin `05182aa`) produced a Serial output that differs from the cached oracle at **5 cells** (node 125225, levels 17–21 — all at the bathymetry transition). All other 5962321 cells were bit-identical. The differing cells have value 0 in the cached oracle and ULP-sized non-zero in the fresh oracle. **Cause**: M5.9 FIX / M5.9-pin added/dropped `sync_host()` calls in `fesom_step.cpp`. These are no-ops on Serial at runtime, but the surrounding source-line changes were enough for the compiler to reorder host loads/stores at one bathymetry-edge codepath → fp ULP drift. The CUDA path was never broken; the oracle was. Bisect ladder that pinpointed it:
  1. Today's CUDA vs cached oracle: max\|Δ\|=1.28 (FAIL)
  2. Today's OMP vs today's Serial: 0.000 (bit-identical — OMP healthy)
  3. Today's CUDA vs today's Serial: 6.2e-4 (climate-close, PASS)
  4. Today's Serial vs cached oracle: 1.28 (the oracle is stale, not CUDA)

  **Fix**: promote `today_serial/snap_000020.nc` to be the new oracle; rename the
  stale one with a `.STALE_<reason>` suffix. **Rule going forward**: rebuild the
  oracle whenever (a) any commit touches `src/fesom_step.cpp` (even via no-op
  sync_host calls), (b) the build env changes (compiler / openmpi / netcdf
  module update), or (c) the gate fails unexpectedly on a binary that should
  be inert. `scripts/gpu_fidelity_gate.sh --fresh-oracle` does this automatically.
  See `docs/STATE_TODAY.md` for the full 2026-05-28 cleanup record. ALSO: the
  unexpected gate fail was the user's clue that "the development tree is a mess" —
  no single doc told them which oracle is canonical. `STATE_TODAY.md` fills that
  gap; treat such state-snapshot docs as a first-class deliverable alongside
  the lessons log.

- **L52 — M5.11: with all halos and host-staging flips already shipped, the GPU wall on dist_8 is launch density, not single-kernel cost or bandwidth.** The profile pass (Kokkos `set_begin/end_parallel_{for,reduce,scan}_callback` auto-instrumentation in `fesom_profile.cpp` — no source-level wrapping needed) gave the per-kernel-time table across np=1 / dist_4 / dist_8. **Key finding**: the **biggest individual kernel at dist_8 is only 1.51% of step** (`fesom_halo_unpack` at 526 calls/step × 14.8 µs). Top-12 kernels combined are < 8% of step. **There is no fat compute kernel to optimize.** The cost is in **2450+ launches/step** from 3 high-frequency regions: halo pack+unpack (1052), CG iters (~560), EVP per-subcycle (~840). Per-call costs are 13–30 µs — near the bare CUDA launch overhead floor (~5–10 µs). The M5.12 lever is therefore **fusion** (collapse adjacent kernels in EVP, CG, halo brackets), not memory coalescing (Lever C M5.10b already failed for this very reason at FCT, the dominant phase). The auto-instrumentation pattern: register Kokkos profiling-tools callbacks, fence-bound them in begin/end → every parallel_for gets a bucket by its Kokkos label; no per-call source change. Adds ~12% overhead in profile mode (zero with env unset; the install_callbacks early-return makes the model byte-identical when `FESOM_STEP_PROFILE` is unset). PCIe is no longer dominant at dist_8 (1.07 GB/step vs the M5.3 stale 13 GB/step — the M5.1+M5.4+M5.7 halo flip campaign reduced PCIe by 12×). See `docs/PROFILE_M59.md` for the full table + bound-label decision tree, and `docs/M512_PLAN.md` for the lever plan.

- **L53 — M5.12f (2026-05-28): deferring the in-loop CG-dot reduction fence (host-scalar → device-`View` reducer) is PERF-NEUTRAL, and the CG is too small a fraction of the step for the lever to ever matter. REVERTED.** The change converted the two in-loop `parallel_reduce` sites in `fesom_ssh.cpp` (`fesom_cg_spmv_dot`, `fesom_cg_psolve_dot2`) from host-scalar reducers to device `Kokkos::View` reducers (the 2-output `psolve_dot2` via two rank-0 subviews of one `View<real_t[2]>` → one `deep_copy`), so the implicit post-kernel fence defers to an explicit `deep_copy` placed right before the `MPI_Allreduce`. **The mechanism is REAL and confirmed from the Kokkos 4.4.1 source** (`core/src/Cuda/Kokkos_Cuda_Parallel_Range.hpp:373-378`: a host-scalar result is not device-accessible → `execute()` does a synchronous post-kernel `DeepCopy` bracketed by `Kokkos::fence`; a device-`View` result is device-accessible → that automatic fence is skipped, leaving the value on device until the explicit `deep_copy` fences). The combined reducer over two subviews lands wholly in device memory — correct, race-free (distinct 8-byte slots), and Serial bit-identical. **But it bought nothing**: same-day CORE2 dist_8 (interleaved before/after on ONE 2-GPU-node allocation, 195 timed steps × 2 reps) measured **0.4836 → 0.4874 s/step = +0.78% SLOWER** (within the ~5% noise band but consistently ordered → a real tiny regression). **Two reasons, both general**: (1) the deferred fence has **no async window to fill** — the host needs each dot *immediately* for the Allreduce / `al=s_old/s_aux` / `be=sp0/s_old` / residual, so the `deep_copy` fence lands right after the kernel anyway, while the View-reducer path adds a hair of dispatch overhead; (2) the CG is only **~1–5% of the step** (M5.2), so the whole `f` lever's ceiling is sub-1%. Even the f-2 follow-up (device-pointer `MPI_Allreduce` on CUDA-aware MPI, which WOULD eliminate the D2H/H2D round-trips) has a **sub-0.2% ceiling** (~few µs × ~224 reduce→Allreduce round-trips/step on a ~484 ms step) and costs a 3.5 h M3.2 re-validation + a CUDA reduction-order numerics shift. **Decision: f lever deprioritized/abandoned; the device-View-defers-the-fence pattern is filed for future use in a site that DOES have an async window (a reduction whose result isn't needed until after more device kernels), but is not a win for the CG dots.** Gates that passed (proving the change was *sound*, just useless): `FESOM_KK_VERIFY=ssh` Serial max|Δ|==0; `gpu_fidelity_gate.sh` PASS (worst 4.971e-3 h_ice, 27 fields); 3-agent adversarial review all-PASS. See `docs/plans/20260528-m512-fusion.md` §M5.12f §RESULT.

- **L54 — M5.12d (2026-05-28): collapsing FEW BIG (bandwidth-bound) kernels saves only their launch/alloc OVERHEAD, not their compute — so the launch-density thesis pays off on MANY SMALL kernels, not big ones. The blmc smoother collapse landed a real but small ~0.7% win (COMMITTED); the strategic read is to reprioritize toward EVP fusion (b) and halo aggregation (g).** The M5.5 device smoother smoothed blmc as 3 channels × a per-channel call, each call = 3 sweeps × (gather+scale) + its own `vol`/`work` `Kokkos::View` pair. Per step that's **18 kernel launches + 6 View alloc/zero-init cycles**. M5.12d generalized `fesom_smooth_nod3D_kk` with `(nslab, slab_stride)` and a flat `RangePolicy(0, nslab*Nmy)` (slab `s=idx/Nmy`, node `n=idx%Nmy`, offset `base+s*slab_stride`) so all 3 channels smooth in ONE call → **6 launches + 2 alloc cycles**. Channels are independent (each reads/writes only its own slab) so it's **Serial bit-identical** (verified: `FESOM_KK_VERIFY=kpp` max|Δ|==0 AND full-model CORE2 `diff_snap.py` ALL FIELDS BIT-IDENTICAL → M3.2 skipped). `nslab=1` default keeps the bvfreq caller + signature byte-identical, so **`fesom_step.cpp` is untouched** → cached gate oracle stays valid (no `--fresh-oracle`; L51). **The key quantitative lesson**: the smoother gather is ~350µs/call and **memory-bandwidth-bound** (R1c) — collapsing 9→3 calls does NOT remove that compute (the same total nod3D-gather traffic just runs in 3 bigger kernels). The ONLY saving is **12 fewer launches + 4 fewer `cudaMallocAsync` (~344µs each, nsys §5) + their zero-init memsets** ≈ ~0.7% measured. **Corollary for the rest of M5.12**: a launch-fusion lever's payoff ≈ (launches removed) × (per-launch overhead, NOT per-launch wall-time). It's large only when the fused kernels are SMALL and overhead-bound (near the ~13µs floor) — i.e. **EVP per-subcycle (b): ~480 launches of ~13µs kernels → the real ~1–3% prize**; halo aggregation (g). It's small for few big bandwidth/compute-bound kernels (smoother d, and CG f where the fraction is tiny too — L53). Plan §M5.12d §RESULT; reprioritize b ahead of g.

- **L55 — M5.12b (2026-05-28) + the M5.12 fusion-campaign retrospective: kernel-launch fusion caps at <1% per lever, because the removable per-launch OVERHEAD is only ~6-7µs (not the ~13µs per-call time, most of which is compute the fusion does NOT remove). The real throughput lever is mesh size, not launch count.** M5.12b fused the 7 EVP per-subcycle kernels into **3** (zero / stress+scatter / node-update), saving **480 launches/step** — the single biggest launch cut available — yet measured only **+0.61%** (same-allocation CORE2 dist_8). The arithmetic: 480 × ~6.5µs ≈ 3ms on a ~480ms step ≈ 0.6%, exactly as measured. **The whole fusion campaign confirms the cap**: f (CG, ~110 launches in a 1-5%-of-step solver) **−0.78% → reverted** (L53); d (smoother, 12 big bandwidth-bound launches) **+0.73%** (L54); b (EVP, 480 small launches) **+0.61%**. The launch-density wall (L52, 2450 launches/step) is real, but *reducing* it yields diminishing returns because per-launch overhead is small and the kernels' compute/bandwidth dominates their wall-time. The remaining M5.12 levers (g halo-aggregation, e Kokkos::Graph, h FCT) target the same overhead → also sub-1% each; **M5.12 cumulative is ~3-4%, not the planned 10-15%.** Per `docs/SCALING_DARS.md` the dominant lever is **mesh size** (CORE2 7× → dars 4.1× GPU/CPU gap, zero code) — production at dars/NG5-class meshes already recovers ~half the gap. **The b implementation is also a reusable pattern**: fuse adjacent SAME-INDEX-SPACE kernels (element+element, node+node) keeping each entity's read set local and the scatter in entity order → Serial bit-identical with NO team scratch and NO atomic reorder (verified: evp + all 5 ice keys max|Δ|==0, full-model CORE2 ALL FIELDS BIT-IDENTICAL, M3.2 skipped because the math is unchanged). This is strictly safer than the team-scratch-partial-sum approach the original M512_PLAN proposed (which reorders the scatter → breaks bit-identity → forces M3.2). Plan §M5.12b §RESULT.

- **L56 — M5.12/NG5 (2026-05-29, CORRECTED): at production mesh size NG5 is PCIe-DATA-MOVEMENT-bound (GPU computes only ~7 % of the step, ~75 % is full-field host↔device `cudaMemcpy`) — so the production lever is DEVICE RESIDENCY (continue the M5.1/M5.4/M5.7 halo-flip campaign into the 3-D regime), NOT Lever C, NOT launch-fusion.** ⚠️ This corrects an earlier draft of L56 that — using only the `FESOM_STEP_PROFILE=1` per-phase *wall* timer — concluded "NG5 is compute/bandwidth-bound → Lever C." That was a **methodology error**: the per-phase timer measures phase *wall* = kernel + PCIe-sync + MPI together and cannot separate compute from data-movement inside a phase. The decisive instrument is an `nsys` CUDA trace (kernels **vs** memcpy). nsys on NG5 dist_16 rank 0 (8 steps, job `25227869`, snapshots off, 16.94 s/step): **GPU kernels = 1.19 s/step = 7 %; PCIe `cudaMemcpy` = 12.74 s/step = 75 %** (H2D 8.13 + D2H 4.61; `cudaMemcpy` = 90.6 % of all CUDA API time, ~4 575 transfers/step, max single 327–355 ms full-field, H2D-dominant ⇒ halo unpack/re-push not I/O); MPI/sync/host = 18 %. The "FCT scaled 41× linearly" claim was NOT proof of compute-bound — PCIe full-field syncs scale with volume identically; nsys settles it: **the FCT *phase wall* is 3.708 s/step but FCT *kernels* are only 0.338 s/step → 11× more wall than compute, the gap being full-field PCIe halo syncs.** Within the 7 % compute: smoother 30 %, FCT 28 %, momentum/ALE/KPP 16 %, GM/EOS/SSH ~6 % — all memory-leaning per `ncu --set basic` (SOL Memory ~50–58 %), but irrelevant at 7 % of the step (Lever C touches the 7 %, not the 75 %). The regime shift CORE2→NG5 is real but it is **nod2D-halo-latency-bound → nod3D-PCIe-bandwidth-bound** (two flavors of the *same* data-movement wall), not "launch-bound → compute-bound." **Lever:** flip the remaining host-staged nod3D halos (FCT internals, GM chain, ALE, tracer-diff) to `fesom_halo_field`, eliminate host-op syncs (L39 salinity floor, L50 uvnode) where a device twin exists — each un-flipped halo cheap at CORE2 (16k/rank) is a full-field 259 MB sync at NG5 (462k/rank). **The 3.8× GPU/CPU gap is mostly this reducible PCIe overhead, NOT an intrinsic compute floor** — re-measure after the flips. **Two sub-lessons: (1) the per-phase WALL timer can't tell compute from PCIe inside a phase — use nsys kernel-vs-memcpy to attribute a bottleneck; linear scaling-with-volume is consistent with BOTH compute- and PCIe-bound and proves neither. (2) profile at the PRODUCTION mesh, not the fast dev mesh — the dominant phase changes (here from nod2D to nod3D), though the underlying wall — data movement — does not.** ⚠️ Earlier ncu read reported the FCT/GM kernel durations as µs; that was a unit misread — they are ms-scale per the in-situ nsys trace (the SOL bound-labels were correct; only durations wrong). ⚠️ dist_8 GPU (926k nod2D/rank) is the A100-80GB device-memory ceiling (profile-mode run OOM'd on `kpp.viscA`; scaling run barely fit). See `docs/SCALING_NG5.md` § nsys decomposition + `docs/figures/nsys_ng5_breakdown.png`.

- **L57 — M5.13 (2026-05-29/30): the NG5 device-residency campaign (a–f) cut the NG5 step 34 % by flipping the remaining host-staged nod3D/elem3D halos to `fesom_halo_field`; three method/trap lessons matter more than the individual flips.** Executed L56's mandate: flipped `cfl_z`(a), EOS `hpressure`+`sw_alpha`/`sw_beta`(b), the **GM quartet** `fer_gamma`/`slope_tapered`/`Ki`/`fer_uv`(c, the biggest single win), `uv_rhsAB`(d), ALE `w`/`w_e`+bolus(e), ALE commit `hnode`/`helem`(f) — each = the L48 split-rail recipe (OUT-rail+halo → one `fesom_halo_field`, REMOVE every downstream IN re-push). **Result (NG5 dist_16, clean timing): step 16.27 → 10.88 (a–f) → 6.97 s/step (a–f+g1-uv) = −57 %; node-for-node GPU/CPU 3.76× → 2.51× → 1.61× — BELOW the ~2× target; PCIe `cudaMemcpy` (nsys a–f) 12.74 → 7.48 s/step = −41 %** (CPU unchanged — the flips are `#ifdef KOKKOS_ENABLE_CUDA`, Serial byte-identical). **g1-uv (full `uv` device-residency) was the single biggest win** — `uv` (ELEM3D×nl×2, the largest field) is read ~11×/step, so removing all its re-pushes cut ~3.9 s/step at NG5 and halved the CORE2 deep_copy MB (641→342). (**FINAL after +g1-T: NG5 16.27→6.12 s/step = −62 %, GPU/CPU 3.76×→1.41×, PCIe `cudaMemcpy` 12.74→2.83 s/step = 75 %→44 %; climate-validated; see L58 for the outcome/scaling/validation lessons.**) Three lessons:
  1. **The deep_copy-count proxy is the right per-flip signal when CORE2 wall-time is flat.** Run the mandatory CORE2-active-ice fidelity gate's CUDA leg with `FESOM_STEP_PROFILE=1` → it prints `deep_copy: N calls/step, M MB/step` AND the field-fidelity verdict from ONE GPU job. The **count of full-field transfers removed per flip is mesh-independent and deterministic** (a:207.7→b:199.7→c:188.7→d:186.7→e:175.8→f:163.8 calls/step; 1067→641 MB/step) even though CORE2 s/step barely moved — it's the faithful proxy for the NG5 win (each removed transfer is a ~259 MB nod3D sync at NG5, ×~30 the CORE2 MB). Profiling adds only fences → never changes the fidelity verdict.
  2. **A field made device-resident ACROSS the step boundary (produced at end-of-step, read at start-of-next) with NON-ZERO initial values needs a one-time init push for step 1 — there is no prior end-of-step producer.** f flipped the substep-14 commit of the EVOLVING `hnode`/`helem` (NOT in the set-once `mesh_sync_geometry_device` push, SYNC_MAP §8) and removed the 10 next-step IN re-pushes — but step 1 then read stale/zero device `hnode`/`helem` → `CG_kk pp·App=nan` abort at iter 1. FIX = one `mesh.hnode/helem.modify_host()+sync_device()` before the time loop (`fesom_main.cpp`); steps 2+ get them from the commit. **The CORE2-active-ice gate CAUGHT this; Serial/pi could NOT (host==device → the init push is a no-op, the stale-read can't happen) — exactly the [[feedback-gpu-fidelity-gate]] / L50 reason pi is insufficient.** (Contrast `uv_rhsAB`(d): also cross-step but zero-init + the `is_first_step` AB2 path → no crash.)
  3. **A PARTIAL flip of a high-fan-out field clobbers — full residency is all-or-nothing (L48 restated, sharpened).** A downstream `modify_host()+sync_device()` left in place pushes STALE host onto the correct device value. So flipping the producing halo of a field read at N later sites requires removing ALL N re-pushes (until the next host-write or step end). For `uv` (read across substeps 3–13 + next step + the ice-step ocean2ice) and tracer `values` (EOS/GM/FCT/Redi/trdiff + ocean2ice) this spans **into the ice step** (`fesom_ice_coupling.cpp`) — high blast radius, crash-prone. **g1-uv (full `uv` residency) was then DONE** (commit `e2ad90e`) — it PROVED the cross-file full-residency method: removing all 11 uv re-pushes spans into the ice step (the ocean2ice uv push, `fesom_ice.cpp`) + needs a step-1 init push (uv is zero-init so technically optional, added for robustness), and it passed the gate clean. **g1-T (tracer `T` values + valuesold) DONE — but only after the gate caught 2 wrong attempts; the debugging path is the lesson.** Attempt 1 (T values device-resident, valuesold host-staged) FAILED: deterministic T 5.0e-2 → Kv 5.4e-1 / S / u / v / h_ice. I hypothesised the values↔valuesold MFCT-coupling asymmetry and flipped valuesold too (attempt 2) → **bit-identical FAIL**, disproving it. The REAL cause: **`fesom_bulk_compute` reads `SST = T[surface]` on the HOST every step** (`fesom_bulk.cpp:259`, the JRA55 air-sea heat flux) — a device-resident T fed bulk a STALE SST → wrong flux → T drift. **This is L50 verbatim (the uvnode→bulk wind-stress case), now for SST.** FIX = one targeted T `sync_host` after trdiff (T device-resident *with* a bulk SST sync, like uvnode); gate PASS worst 1.4e-2 (T 5e-2→1.6e-3). **Sub-lessons: (1) a bit-identical, deterministic CUDA divergence is a fixed stale HOST read, NOT atomic-scatter chaos — hunt the host reader (grep the field in the forcing/bulk/coupling paths) before theorising about kernel coupling. (2) Test the hypothesis cheaply: flipping valuesold and getting an IDENTICAL result disproved it in one gate cycle. (3) Every prognostic field that bulk/forcing reads on its surface row (`uvnode` wind, `SST`=T, and `SSS`=S for sss_runoff) is an L50 host reader — full device-residency must keep a targeted surface sync for it.** **g2 (S-floor) deferred** (conditional gate NOT met; and S would hit the same L50 SSS host reader). Tags/commits: a `6b54df3` · b `150996d` · c `73c79fd` · d `db5ef5d` · e `74d3c89` · f `da41f64` · g1-uv `e2ad90e` · g1-T `eaac63b`. See `docs/SCALING_NG5.md` § M5.13 + `docs/GPU_FIDELITY.md` § M5.13.

- **L58 — M5.13 outcome (2026-05-30): once device residency removes the PCIe wall, the GPU/CPU gap is set by per-rank WORK and its Levante-A100 floor is ~1.4× (per-step launch/MPI overhead), NOT 1× — plus the climate-validation rigor that proved the campaign changed nothing.** Final: NG5 dist_16 step **16.27→6.12 s/step (−62 %)**, node-for-node GPU/CPU **3.76×→1.41×**, PCIe `cudaMemcpy` **12.74→2.83 s/step (75 %→44 %)**. Cross-mesh: the SAME flips cut **dars 4.10×→1.60×** (mesh-general, not an NG5 artifact). Three reusable lessons:
  1. **Post-PCIe, the ratio is a data/compute-balance function — more per-rank work → nearer parity, MONOTONE, but it asymptotes ~1.4×.** dars dist_16 (197k nod2D/rank) = 1.60×, dars dist_8 (395k/rank, 2× the work) = **1.52× (closer to 1)**, NG5 dist_16 (462k/rank) = **1.41×**: denser packing feeds the A100 more and closes the gap (it is feed-rate-limited at low per-rank work). BUT none reach 1× — the residual ~1.4× is the irreducible per-step overhead (~720 kernel launches/step + MPI), i.e. the launch-fusion / Lever-C frontier (**NOT** PCIe; that's solved). So the packing/mesh-size lever and the launch-overhead lever are **separate**: device residency took the gap 3.8×→1.4× ("curiosity"→"within 1.5× and energy-competitive"); closing the last ~0.4× needs the *other* lever, and approaching 1× needs BOTH more per-rank work (denser — but the CPU OOMs on bigger meshes first) AND launch fusion. `docs/figures/scaling_meshsize_trend.png`.
  2. **To prove an Approach-B perf refactor (binary changes but `#ifdef CUDA` / Serial byte-identical) is climate-neutral, re-run the PRE-refactor binary with the SAME CURRENT compare script (apples-to-apples) and show backend-vs-C is statistically identical pre/post — do NOT diff a fresh run against the doc's recorded old numbers.** The 1-yr CORE2 CUDA campaign-binary climate (run `25236304`) was statistically identical to the pre-campaign M5.9-pin run on EVERY field (sst corr 1.00000/bias 1.2e-4; sss/ssh/a_ice/m_ice/uice match to 3–4 sig figs) — pre vs post differ only in the 4th–5th sig fig = run-to-run GPU non-determinism between two CUDA runs (D22) → zero climate-level change. The naive comparison would have conflated the binary change with a **compare-SCRIPT** change: the doc's pre-campaign `a_ice`/`m_ice`-vs-Fortran ~0.91/0.98 were a stale **pre-`466ea3e`** ice-mask-averaging artifact ([[feedback-ice-mask-averaging]]); the current script gives 0.99997 for BOTH runs. (The one genuine C↔Fortran ice budget is `uice`=0.85, identical pre/post, GPU-independent since uice-vs-C=0.99978.) **Recorded validation numbers age when the analysis script changes, not just the model — always re-baseline with the current script before claiming (no-)regression.**
  3. **A per-step staleness-gate's worst-cell |Δ| is a tripwire CEILING, not a climate metric — it does not accumulate.** g1-T's worst T |Δ|=1.6e-3 (vs the 1e-2 ceiling) prompted "since when is 1.6e-3 acceptable?"; the 1-yr climate test answers definitively — over a full year the campaign binary tracks C-port to corr 1.00000 / bias O(1e-4), exactly as the pre-campaign binary. The 20-step gate detects STALENESS (a missing sync reads ~1e-1, orders above the ~1e-3 GPU-noise floor); it is **not** a fidelity bound. Settle a fidelity question with the multi-year climate compare ([[feedback-gpu-fidelity-gate]]), never the per-step gate. See `docs/GPU_FIDELITY.md` § Climate validation.

- **L59 — M5.14 (2026-05-30): finishing device residency on the LAST host-staged nod3D fields (S, density, fer_w, w_i) broke through L58's supposed "~1.4× launch/MPI asymptote" and reached node-for-node PARITY — salinity was a huge hidden PCIe chunk that L58's nsys (a–f only) had not yet removed.** Four flips (all the L48 split-rail recipe), each gated: **S** (the g2 mirror-T flip — biggest blast radius: values+valuesold device-resident, ~13 re-push sites + 2 halos + the host salinity-floor→device-kernel + 2 cross-file ice re-pushes, one atomic commit `491ccb8`); **density** (`84c1d8d` — no on-device consumer, halo+residency exist only for I/O); **fer_w**+**w_i** (`0bc7da9` — clean 12_ale wins). **Result (NG5 dist_16, SAME-DAY baseline — rebuilt the pre-Lever-A `build-cuda` and ran the identical job today, per [[feedback-perf-same-day-baseline]]): step 6.14→3.80 s/step (−38 %); deep_copy 11.02→6.57 GB/step (−40 %), 139→121 calls.** Node-for-node **GPU 3.80 / CPU 4.33 = ~0.88× → the GPU is now ~14 % FASTER than the CPU** (the same-day GPU baseline reproduced L58's 6.12 to 0.4 %, so today's conditions match and the CPU 4.33 ref is valid; a same-day CPU re-measure confirmed it). **This overturns L58's headline conclusion**: L58 read the residual 1.41× as an irreducible ~720-launch/MPI floor, but that nsys was taken on the a–f binary BEFORE S/density were flipped — S alone (biggest field `13_fct`, the biggest phase) was still doing ~13 full-field 259 MB D2H/step at NG5. Removing it dropped the step below the CPU. **The regime genuinely flipped to compute-bound**: the post-M5.14 nsys kernel summary has the `fesom_smooth_nod3D` smoother as the #1 GPU kernel (25.7 % of kernel time), FCT/momentum/GM/trdiff filling the rest — `cudaMemcpy` no longer dominates. Four reusable lessons:
  1. **A "launch/MPI asymptote" claim is only valid for the residency state it was measured at — re-measure after each residency increment.** L58's 1.4× floor was real for a–f but not fundamental; one more big field (S) crossed parity. Don't extrapolate an asymptote from a partial campaign.
  2. **The host salinity floor → a race-free per-node device clamp `S(i)=max(S(i),0.5)` is bit-identical (Serial AND CUDA) and unblocks full S residency.** It's an idempotent elementwise op over myDim+eDim×column (no scatter/reduction), placed AFTER the post-trdiff device halo so the owned+halo clamp reproduces the prior host exchange-then-floor order. This retires the L36/L39 "salinity-floor-stays-host" pin — the floor was only host because nothing else forced S onto the device; now S is device-resident, the floor follows. (`fesom_salinity_floor_kk` in `fesom_tracer_diff.cpp`; 0.2 ms/step, nsys ~0.0 %.)
  3. **Output-field residency requires moving the MONTHLY-MEAN accumulation to the device (Task 4, `2611936`) FIRST, else the mean reads a stale host alias.** S/density/temp are monthly-mean outputs; their per-step host copy existed only to feed the mean. Device-side per-element accumulation (`resolve_{salt,sss,density}_dev`, bit-identical time-sum) removes that D2H AND fixes a latent bug — the already-device-resident u/v/w/Kv/Av/bvfreq means had been accumulating STALE host values between snapshots since the M5.x campaigns (gate-blind: snapshots were fine, means were frozen). Validate the mean path separately from the snapshot gate: Serial dev-vs-host `.monthly.nc` bit-identical + CUDA device-accum mean vs Serial ground-truth ≤1e-3 (the snapshot gate cannot see a stale mean).
  4. **T-trim (full→surface-only sync for the bulk-SST host reader) was EVALUATED and SKIPPED — the NG5 re-time decided it.** T's full sync is only ~4 % of the 6.57 GB/step deep_copy (259 MB, in `13b_trdiff`=5.25 %); a surface-only refresh would save ≲1–2 % of the step but needs a persistent buffer + a manual host-surface write that sidesteps the DualView (the exact staleness-prone pattern the gate catches; M5.9-pin already found surface trims hit per-step alloc overhead "not worth +0.5 %"). **Measure before implementing a marginal optimization — the data killed it cleanly.** Commits: t4 `2611936` · S `491ccb8` · density `84c1d8d` · fer_w/w_i `0bc7da9`. See `docs/SCALING_NG5.md` § M5.14 + `docs/GPU_FIDELITY.md` § M5.14.

- **L60 — M5.15 (2026-05-30): a clean re-profile OVERTURNED M5.14's "compute-bound" verdict — the step is data-movement/halo-latency bound (GPU util ~30 %), so the lever was FINISHING RESIDENCY on the GM chain, not kernel coalescing (C) or CG-reduction.** T1+T2 (`81b8947`) flipped the GM host-bracket halos `sigma_xy`/`neutral_slope`/`fer_K` to `fesom_halo_field` + guarded the verify-only `fer_tapfac`/`fer_scal`; T3 (`a39ac70`) removed the verify-only `bvfreq`+`dbsfc` syncs (+ the redundant dbsfc re-push). **Cumulative NG5 dist_16: 3.80→3.456 s/step (−9 %), deep_copy 6.57→4.10 GB/step (−38 %), node-for-node 0.879→~0.80× (GPU ~25 % faster than CPU).** All bit-identical Serial + gate PASS (climate-safe; 1-yr climate launched to close, job 25245115). Six reusable lessons:
  1. **GPU UTILIZATION (kernel-active % of wall) is the bottleneck metric — NOT the kernel mix or the API-time summary.** M5.14 called it "compute-bound" from "the smoother is the #1 *kernel* (25.7 %)" + "cudaMemcpy no longer dominates the *API summary*". But the windowed kernel-busy fraction of wall was only ~30 % → the GPU idles ~70 % on PCIe + halos. A kernel being the biggest slice of GPU-*busy* time ≠ the GPU being the wall. Always compute kernel-time ÷ wall over a steady-state window, and **split memcpy by `copyKind`** — the raw API summary lumps the one-time startup transfers + DtoD with the per-step HtoD/DtoH, so it reads ~55 % "cudaMemcpy" when steady-state PCIe is ~34 %.
  2. **To attribute residual PCIe to specific fields, instrument the single DualView sync chokepoint** (`Field::sync_host/sync_device`, `need_sync`-gated, compile-guarded `FESOM_SYNC_LOG` in `fesom_field.hpp`) → per-field DtoH/HtoD names+bytes. `nsys` cannot name Kokkos deep_copies (everything is the generic `cuda_parallel_launch_local_memory` symbol) and `Kokkos::Profiling::pushRegion` reaches nsys only with `kp_nvtx_connector.so` (not built). The logger is the definitive tool; run it on the small CORE2 config (the field SET + per-step COUNT are mesh-invariant; bytes scale).
  3. **A `sync_host` whose only host reader is the `FESOM_KK_VERIFY` C-twin is PURE WASTE in production** — the twin runs only under verify, where Serial host==device makes the sync a no-op anyway. Guard behind `if (s_verify_<key>)` or remove. **⚠️ raw `.h()` reads BYPASS SYNCCHECK** (only `h_checked()` asserts) → you must trace the reader's verify-gating IN THE SOURCE, not trust a clean SYNCCHECK. Discriminator: host C-twin raw-pointer reads (`aux->field[idx]`) are verify-only; device `.d()` reads are production. (dbsfc: host `kpp_bldepth` reading `aux->dbsfc[...]` is inside the `s_verify_kpp`-gated `fesom_kpp_verify`; production is `kpp_bldepth_kk` `.d()` → removable.)
  4. **A residency flip kills TWO costs at once.** A host-bracket halo is `sync_host` (DtoH) + host `fesom_halo_exchange` (the `MPI_Waitall`) + re-import (HtoD); flipping it to `fesom_halo_field` removes the PCIe AND moves the exchange to the device GPU-aware path (shrinks the Waitall). Fingerprint: the `1b_gm` phase collapsed 11–16 %→4.6 % of loop.
  5. **Residency EXHAUSTS.** After the GM halos, the remaining per-step DtoH is genuinely-required host readers — `T` and `uvnode` for the JRA55 **bulk forcing** (`fesom_bulk.cpp:259` reads SST, the wind-stress reads `uvnode`, both on host next step; the L50 class), `S` already device, `ghats` a deliberate host-keep, `MLD1_ind` tiny. The remaining ~47 % `MPI_Waitall` is then load-imbalance + many-small-message latency → the **Lever B interior/boundary OVERLAP** target, NOT more residency. The CG-reduction flavor of B stays dead (`Allreduce` 0.2 %, re-confirms M5.2/L?).
  6. **Validation-run gotcha:** the per-kernel/bit-identity ladder must run on a **compute node** (`-p compute`, `srun`) with `source /sw/etc/profile.levante` (init the spack module env *before* `env.sh`, else `module load` silently picks the wrong HDF5 → `nc_create` EACCES) and the output dir **`mkdir -p`'d** (`fesom_port` does not create it; login-node HDF5-on-Lustre also EACCES). Reusable: `jobs/job_m515_serial_val` (eos/kpp/gm verify + pi np1/np2 bit-id + SYNCCHECK in one). See `docs/GPU_FIDELITY.md` § M5.15, `docs/plans/20260530-m515-gm-residency.md`, [[project-m515-gm-residency]].

- **L61 — M5.16 (2026-05-30): port a HOST forcing routine to a device per-node map and it pays TWICE — the compute moves off the single-threaded host AND the DtoH that the host read forced disappears.** `fesom_bulk_compute` (the L&Y09 air-sea bulk formulae) was 16 % of the NG5 step running single-threaded on the GPU build's Kokkos-Serial host (the blmc/L49 trap). Ported it to a `KOKKOS_LAMBDA` per-surface-node MAP (`fesom_bulk_compute_kk`, `fesom_bulk.cpp`): `ncar_ocean_fluxes_mode`+`obudget` → `KOKKOS_INLINE_FUNCTION` device twins (the IceThermC/L45 recipe — scalar `#define`s are device-safe, runtime `z_*` captured by value; std math → `Kokkos::`), 8 JRA55 surface fields HtoD'd at bulk, **SST=`T`[surface] + `uvnode` read on `.d()`**. **NG5 dist_16: 3.4524→2.6766 s/step (−22.5 %, rigorous same-day SAME-NODE baseline), `force:bulk_compute` 16.0 %→0.95 %, deep_copy 4.10→3.41 GB/step.** Six lessons:
  1. **A device port of a host READER dissolves the DtoH that reader forced.** `T`/`uvnode` were the LAST "genuinely-required host reader" DtoH (L50/L60 — the M5.9-pin NaN-poison proved bulk was their SOLE host reader). Porting bulk to read them on-device let the per-step `T` `sync_host` (`fesom_step.cpp:900`) + `uvnode` `sync_host` (`:334`) be DELETED. So the −22.5 % = the compute port (−0.53 s, the `force:bulk_compute` collapse) PLUS the residency unlock (−0.25 s, deep_copy −0.69 GB). **A 16 %-measured lever returned 22.5 % because the two wins compound** — when a host routine is also the last DtoH's consumer, porting it is simultaneously a compute lever AND the residency finish. Re-profile after to catch the bonus.
  2. **DROP-IN first (Phase A), fully-device-resident later (Phase B).** The device kernel writes all outputs on device, halos the 3 the C halos via `fesom_halo_field`, then **`sync_host`s the FULL output set** → the downstream (`oce_fluxes_mom` [still host], the ice-step IN rails, the ocean re-pushes) sees byte-for-byte the host-authoritative state the C twin left → ZERO downstream edits → the validation surface is just the new kernel + the 2 deleted syncs. Making `forcing` *fully* device-resident (drop the output `sync_host` + every downstream re-push, port `oce_fluxes_mom`) is a separate, riskier, smaller follow-on (it reclaims the small nod2D round-trip Phase A re-adds — deep_copy −16.7 % not more). Capture the big win cheaply; measure before paying for the tail.
  3. **The consumer map is the design.** Before flipping forcing outputs to device-authoritative, trace every consumer: `Ch_atm_oce`/`Ce_atm_oce`→ice-thermo `.d()`, `stress_atmice`→EVP `.d()` (device), but **`stress_node_surf`→`oce_fluxes_mom` is HOST** (reads+blends+overwrites) and **`heat_flux`/`water_flux` are OVERWRITTEN by `oce_fluxes` (reads `flx_h`/`flx_fw`, not bulk's value)**. The lone host consumer (`oce_fluxes_mom`) is why Phase A `sync_host`s `stress_node_surf`; the overwrite is why bulk's `heat_flux` only matters for the verify. Map it or you'll push stale host over a device write (the re-push trap).
  4. **FORCED-ONLY physics → the verify is CORE2, NOT pi (L42 restated).** pi uses analytical wind stress and never calls `fesom_bulk_compute` (`jra55_year≤0`) → the `FESOM_KK_VERIFY=bulk` Serial max|Δ|==0 gate is meaningful only on CORE2-SLURM with JRA55 active (`job_bulk_verify_core2`, the `job_ice_verify_core2` clone). Ride the ice keys along to confirm the forcing→ice chain is unchanged. The verify is the EOS/iceflux style (full overwrite from intact inputs → snapshot KK, run C twin, diff, restore — no capture-before); the caller `sync_host`s `T`+`uvnode` first so the C twin's host SST/current reads are current (no-op on Serial).
  5. **Same-day SAME-NODE baseline (the perf-baseline lesson, executed).** To attribute the −22.5 % cleanly, rebuilt the prior commit (revert the 4 files via a saved `git diff` patch — NO commit; build the binary, `cp` it to a stable path; restore + rebuild) and ran `job_ng5_prof` — it landed on the SAME 4 GPU nodes and reproduced M5.15's 3.456 (3.4524) exactly → the win is real, not node-mix. A running SLURM job is unaffected by clobbering its on-disk binary (the linker's rename gives a new inode; the process holds the original) — verified the climate/gate/profile all exec'd the right build via the `nm fesom_bulk_compute_kk` symbol count + the exec-time-vs-clobber timeline.
  6. **The win exceeded the estimate, so the next-lever ranking shifts.** Forcing is now 8.7 % of the step (`force:jra55_read` 3.6 % [stays host — NetCDF read+interp] + bulk 0.95 %). Residency is fully exhausted. The remaining levers are **B** (the ~47 % `MPI_Waitall` interior/boundary overlap — re-measure the overlappable-vs-load-imbalance split first) and **C** (kernel coalescing, still last — GPU util is higher now but the wall is halo-latency, not kernel compute). Memory [[project-m516-bulk-port]]; `docs/GPU_FIDELITY.md` §M5.16.

- **L62 — M5.17 (2026-05-30): MEASURE the `MPI_Waitall` before building comm-overlap — a barrier-isolation experiment proved 79 % of it is LOAD-IMBALANCE idle, not recoverable comm.** The roadmap nominated Lever B (interior/boundary overlap + halo aggregation) off "`MPI_Waitall` ~47 % of the step." The discipline: a chunk of that is fast-ranks-idling-for-slow-ranks, which overlap CANNOT recover — measure the split first. **The cheap decisive instrument: an env-gated `MPI_Barrier(MPI_COMM_FESOM)` BEFORE every halo exchange (`FESOM_HALO_BARRIER=1`) + a per-rank Barrier/Waitall accountant (`FESOM_HALO_MPI_PROF=1`, `MPI_Reduce` min/mean/max at loop end), in `fesom_halo_device.cpp` + `fesom_halo.cpp` (BOTH halo paths).** The barrier absorbs the per-rank arrival skew, so the following Waitall measures PURE comm. Five lessons:
  1. **The split (NG5 dist_16 M5.16 binary, `job_ng5_halo_split`):** halo `MPI_Waitall` = 0.288 s/step = **10.8 % of the 2.677 s step** (NOT 47 % — that old nsys number was rank-0 wall lumping imbalance idle + setup + the I/O Gatherv). Barrier split = **imbalance 79 % (0.246 s) / comm 21 % (0.065 s)**; **overlappable-comm CEILING = 0.065 s/step = 2.4 %** → Lever B not worth it.
  2. **Three cross-checks (don't trust one number):** (a) the barrier SPLIT (79/21); (b) base-run across-rank Waitall SPREAD huge (0.147→0.565 = 0.42 s skew = imbalance signature) vs barrier-run Waitall TIGHT (0.041→0.082 = clean uniform comm); (c) the WALL test — +638 barriers/step moved the step only +0.7 %, proving the imbalance idle was ALREADY paid as Waitall (the barrier relabels it, not an artifact). The per-exchange barrier upper-bounds imbalance (blocks cross-kernel skew amortization); the +0.7 % wall bounds the over-count to noise.
  3. **Deadlock trap:** the barrier MUST precede the per-rank empty-comm early-return (`if (cs->rPEnum==0 && cs->sPEnum==0) return;`) — `fesom_halo_field` is rank-uniform, so a rank with empty comm for a kind would skip the collective while others block → hang. Balanced dist_16 never triggers it, but fix structurally.
  4. **Measurement-only → byte-clean:** gated (the Waitall wrapper does the same `MPI_Waitall`; barrier off in prod), validated **pi np1+np2 BIT-IDENTICAL** (np2 exercises the host edit) + the `clean` run reproduced 2.6766 s/step EXACTLY. CG/EVP `Allreduce` DEAD at 0.4 % (re-confirms M5.2/L60).
  5. **The walls after Lever B dies:** (1) load imbalance ~9 % → **Lever D** (better/work-weighted partition, deployment-side, no port-code risk) IF static; (2) GPU-compute + residual PCIe ~89 % → **M5.18**: nsys #1 kernel `fesom_smooth_nod3D_kk` = **25.7 % of GPU compute**, slow because uncoalesced node-major gather (`arr[node*NL+nz]`, one-thread-per-node → 32 cache lines/warp) + depth divergence; `deep_copy` still 3.41 GB/step. Fix = LOCAL bit-identical re-parallelize to one-thread-per-(node,LEVEL) (level = contiguous → coalesced), no global layout refactor. Memory [[project-m517-mpi-comms]]; `docs/GPU_FIDELITY.md` §M5.17; prompt `docs/plans/20260531-m518-smoother-compute-PROMPT.md`.

- **L63 — M5.18 (2026-05-30): the coalescing lever — re-parallelize a node-major kernel over the CONTIGUOUS inner dim (level), not the node, for a bit-identical 10–15× kernel speedup.** `fesom_smooth_nod3D_kk` (the #1 GPU kernel, 25.7 % of GPU compute) mapped one thread per (slab,owned-node) with an internal level loop → consecutive threads = consecutive nodes → every store strided by `NL=70` → uncoalesced (ncu: STORE 29.5 sectors/req, SM util **2.2 %** = ALUs 98 % idle on memory latency, occ 52 %, duration 48–145 ms with 3× depth-divergence spread). **Re-parallelized to one thread per (slab,node,LEVEL)** — flat `RangePolicy(0, nslab*Nmy*NL)` decoded to `(s,n,nz)`, mask `nz∉[uln,nlnz]`. Lessons:
  1. **WHY it works without a layout refactor:** the field is ALREADY node-major (`arr[node*NL+nz]`), so the LEVEL is the contiguous inner dim. A warp of 32 consecutive flat idx = 32 consecutive levels of the SAME node → reads `arr(sb+v*NL+nz0..+31)` and stores at `idx` are contiguous = **coalesced**; depth divergence vanishes (one level/thread). The heavyweight Lever C (`rank-1 → View<double**>`) is NOT needed when the existing layout already makes some axis contiguous — exploit it. This is the local, low-risk cousin.
  2. **Bit-identity holds because the per-`(n,nz)` accumulation is unchanged:** the element loop runs the SAME `k=o0..o1` order; a **register accumulator** (`w += a*(...)`) is IEEE op-for-op identical to the old in-memory `work[...] += ...` (FP add sequence is what matters, not where the accumulator lives). Serial AND OpenMP `max|Δ|=0` (pure map, no scatter). Masked threads (shallow-node tail, `nz>nlnz`) early-return → cheap; acceptable thread-waste for a memory-bound kernel (coalescing + occupancy dominate).
  3. **Result (ncu A/B + same-node s/step A/B):** gather 64→**4.1 ms (15.5×)**, STORE sec/req 29.5→6.9, LOAD 23.8→2.9, occ 52→**91 %**, SM 2.2→**59 %**; scale 13.9→1.4 ms. **Whole step (NG5 dist_16, clean, same allocation): 2.4823→2.1296 s/step = −14.2 %**, all in the ocean phase (1.799→1.449, −19.5 %), other phases byte-stable. ~2× the 5–8 % estimate because the kernel beat the assumed 2–4× by going 10–15×.
  4. **A node-major kernel needs its OWN isolated verify — the upstream gate may not cover it.** `eos` runs BEFORE the bvfreq smoother (sees unsmoothed bvfreq); `kpp` sees blmc only through `max(viscA/diffK, blmc)` (a small-blmc bug is masked). Added `FESOM_KK_VERIFY=smooth` (`fesom_smooth_nod3D_kk_verify`): capture-before input → run kernel → run host C twin per channel → diff owned. CUDA gate PASS at the SAME floor as M5.16 (worst 7.7e-3, zero new divergence). **Generalize the lever:** re-rank kernels (nsys) and apply per-(node,level) to the other node-major ones (FCT, `compute_vel_rhs`, GM `compute_sigma`, `diff_ver_part_impl`); then attack the 3.41 GB/step residual PCIe. Memory [[project-m518-smoother-compute]]; `docs/GPU_FIDELITY.md` §M5.18.

- **L64 — M5.19 (2026-05-30): the coalescing lever GENERALIZES (−5.35 % more from 6 kernels), and "column scratch" is often a false bucket-B blocker — read its data flow.** Applied the L63 flat lever to the next memory-bound node/element-major kernels: `compute_vel_rhs` (`fesom_vel_rhs_elem`+`assembly`), `momadv_area`/`v2e`, and GM `compute_sigma_xy`+`compute_neutral_slope`. Same recipe — `RangePolicy(0, N*nl)` decode `(n,nz)`, mask `nz∉[lo,hi)`, register-write; per-element/node scalars recompute per thread (warp-broadcast, cheap). Lessons:
  1. **The bucket dictates the technique (classify by code, not profile):** A pure-per-level-map → flat lever (bit-identical); B per-column → TeamPolicy OR flat-lever-if-eligible (below); C TDMA → Lever C layout only; D scatter → different axis. M5.18 didn't touch these so the pre-M5.18 nsys ranking minus the smoother is the valid target list — step-0 re-profile is a sanity check, not a planning input.
  2. **"Column scratch" is only a TRUE bucket-B blocker if it carries a CROSS-LEVEL REDUCTION or a recurrence on WRITTEN values.** GM `sigma_xy`'s `tx/sx/vol[NL_MAX]` were per-level accumulators (no reduction) → each `(n,nz)` thread re-walks the surrounding elements in the SAME `k=o0..o1` order and accumulates only its level in registers; the element/level loops just swap order, total arithmetic unchanged → byte-identical. GM `neutral_slope`'s 3 passes + `c1[NL_MAX]` had no cross-level WRITE dep (Pass-2 `c1[nz]` reads only INPUT `bvfreq[nz/nz+1]`; Pass-3 re-reads `ns[nz]` = the bits Pass-1 wrote) → the passes FUSE into one per-`(n,nz)` map, `c1` register-local. **The prompt guessed GM→TeamPolicy; the data flow said flat-lever (low-risk, bit-identical). Always trace the scratch before reaching for TeamPolicy.** Input-only vertical-neighbor reads (`bvfreq[nz+1]`, FCT `qr4c_v`'s `valsAB[nz±2]`) do NOT block the flat lever.
  3. **Result:** ncu A/B — `vel_rhs_elem` 13.4×, `sigma_xy` 12.6×, `vel_rhs_assembly` 12.0×, `neutral_slope` 10.6× (LD 22–30→3–8 sectors/req, SM 2–3 %→18–44 %); whole step (NG5 dist_16, same-node) **2.1278→2.0139 = −5.35 %**, the 4 kernels' ~102 ms/step ncu savings ≈ the 114 ms clean-step delta → fully attributed. CUDA gate PASS (worst 1.7e-3, zero new divergence); `vrhs`/`gm` Serial=0; pi np1+np2 BIT-IDENTICAL.
  4. **The shared verify key covers the whole chain — exploit it.** `vrhs` diffs the FINAL `uv_rhs`/`uv_rhsAB` vs the C twin (incl. `momadv_*`); `gm` covers `sigma_xy`+`neutral_slope`+all downstream → both `momadv_area`/`v2e` and the GM flips validated for free under existing keys. Generalize next (M5.20): bucket-A FCT sub-kernels (`zal_a1`/`a2`/`qr4c_v`/`f2d_v`/`ale_recon`, each <0.9 %, `tradv` verify) + `update_vel`; `13_fct` (22.8 %) is ~24 mixed-bucket kernels sharing scratch → focused pass, higher risk/lower ROI than these were. Memory [[project-m519-kernel-coalescing]]; `docs/GPU_FIDELITY.md` §M5.19.

- **L65 — M5.20 (2026-05-30): the PCIe track — MEASURE per-field first (it refuted the planned lever), then device-residency on the real drivers (`hnode_new` + `sw_3d`) = −36 % deep_copy, −17.6 % step.** The residual 3.41 GB/step `deep_copy` is ~100 % PCIe (nsys: D2D 0.04 %). `FESOM_SYNC_LOG` (a scoped CMake option → one `SYNCLOG D2H/H2D <label> <bytes>` line per actual Field sync; `jobs/job_ng5_synclog`) attributed it: `hnode_new` 778, `sw_3d` 519, `ghats` 256, `S` 249 MB/step — **NOT the 8 JRA55 forcing fields (~7 MB/step each), so the prompt's Phase-B-forcing plan would have gained ~0.** Lessons:
  1. **`deep_copy` BYTES ≠ WALL-CLOCK — a per-step host↔device round-trip of a big field costs FAR more than its bandwidth (the headline insight).** `hnode_new`'s 742 MB/step at ~25 GB/s PCIe = ~0.030 s/step of pure bandwidth, yet removing it cut the step by **0.23 s/step (−11.4 %)** — ~8× the bandwidth estimate. Why: each `sync_host`/`sync_device` is a BLOCKING fence — the host stalls waiting for the copy and the GPU pipeline drains/refills at the substep boundary; `hnode_new`'s 3× round-trip serialized 3 substep transitions per step. **Consequence for prioritizing PCIe work:** value a field by how often it round-trips per step and whether it crosses substep boundaries (pipeline stalls), NOT by its byte count. The deep_copy GB/step metric UNDER-states the opportunity for per-step round-trippers and OVER-states it for one-shot/startup transfers. (This is why the estimate "~2.6 % from 38 % of PCIe" was 4× too low.)
  2. **Measure per-field before building — it refuted the planned lever (3rd time).** Phase-B forcing was the prompt's §6 plan; the `FESOM_SYNC_LOG` attribution showed the 8 JRA55 forcing fields are only ~7 MB/step each → ~0 gain. Same discipline that killed Lever B (M5.17 barrier-isolation) and re-aimed M5.15 (re-profile). BOTH a-priori guesses were wrong (mine "host-staged halos", the prompt's "forcing") — it was residency-gap round-trips. **The reusable METHOD:** (a) nsys `cuda_gpu_mem_time_sum` first to confirm the deep_copy is PCIe (HtoD+DtoH) vs device↔device (here D2D was 0.04 % — so genuinely PCIe; if D2D dominated, the "PCIe" framing would be wrong); (b) `FESOM_SYNC_LOG` (compile-time `#ifdef` in `fesom_field.hpp`, exposed as a scoped CMake `option` so only `fesom_port` rebuilds, Kokkos cached) emits `SYNCLOG D2H/H2D <label> <bytes>` per ACTUAL Field sync; (c) run multi-rank (halos need ≥2 ranks; np=1 misses them), per-rank stderr (`--error=err.%t`) to isolate rank 0; (d) aggregate per (label,dir) and read **calls/step** — ≈1+ = a per-step driver, ≈1/NSTEPS = a one-time startup transfer (mesh/IC) that the byte-total inflates.
  3. **A "placebo" sync can still be load-bearing at STEP 1 — the first-step IC seed.** `hnode_new`'s 3 syncs looked like pure M5.13 "self-containment" defensive re-pushes (the M5.9-pin pattern); removing all 3 CRASHED (`CG_kk pp·App=-nan`, step 1). Root cause: substep-1b GM reads `hnode_new` on device BEFORE the substep-12a thickness kernel writes it that step, so at step 1 the value must be the host IC — the removed substep-1 re-push had been silently providing it. Fix = device-resident (for step ≥2 the device holds last step's 12a value, IDENTICAL to what the old re-push copied via the host) + **seed the device copy from the IC at step 1 only** (`if(step_n==1) modify_host()+sync_device()`). **Reusable checklist before deleting any re-push:** (i) list every DEVICE reader and its substep index; (ii) find the producing kernel's substep — if any reader precedes it, that field needs a prior value; (iii) at step 1 the "prior value" is the IC (host) → keep a one-time `step_n==1` seed; (iv) the CUDA gate (CORE2-ice) catches the miss, pi CANNOT (forcing/ice not exercised — the M5.9-pin blind spot). The crash signature (10 s run, only `snap_000000`) = NaN abort at step 1.
  4. **Residency optimizations can be CONFIG-specific — guard them for the configs you don't run yet.** `hnode_new` is "not really used in linfs" (it ≡ `hnode`, a trivial device copy — `fesom_ale_thickness_linfs_kk` is one assignment) but becomes a genuinely-evolving thickness under **zstar** (a future vertical-coordinate option; no zstar code/flag exists yet — the model is linfs-hardcoded). The device-residency is a LINFS optimization, NOT a coordinate-agnostic invariant → ⚠️-guarded at all 3 sites with the explicit instruction that zstar must restore `hnode_new`'s host rail if it host-computes or host-reads it. **General rule:** an optimization that exploits "field X is trivial/unused in the current config" must carry a guard naming the config and the future config that breaks it. (The user flagged this — honor domain constraints the code doesn't yet encode; a passing gate on the current config does NOT prove future-config safety.)
  5. **Host-computed forcing → device kernel (the M5.16 pattern) for the 3-D ones; split off small 2-D side effects.** `sw_3d` (the Jerlov shortwave-penetration profile, `fesom_cal_shortwave_rad`) ported to a per-surface-node device kernel `fesom_cal_shortwave_rad_kk`; crucially the `heat_flux[n2] += swsurf` SIDE EFFECT was LEFT on the host (it is a cheap nod2D op whose own PCIe is ~7 MB/step, and moving it would tangle with the device bulk's heat_flux ownership) — only the big 3-D `sw_3d` profile moved to device. `chl` pushed to device on update (const-once / monthly, not per-step). Both `sw_3d` pushes removed → zero sw_3d sync sites. `exp`/`log10` on device → **Serial bit-identical** (Kokkos Serial uses the host libm) but a NEW CUDA divergence class (last-ULP transcendentals, the EOS class) — negligible (climate corr=1.00000), but note it ticks the gate floor.
  6. **Validating a forcing-path device port: pi is useless (analytical forcing skips it) — diff a fresh CORE2 Serial oracle against a SAVED prior oracle.** Before the gate (which overwrites `serref_core2`), `cp -r` the prior oracle; the gate `--fresh-oracle` rebuilds it with the new Serial code; `diff_snap.py` the two → ALL-FIELDS-BIT-IDENTICAL proves the new device kernel equals the host C-twin op-for-op on Serial (the device kernel overwrites the host value with an identical one on Serial, where host==device). Result for M5.20: gate PASS (worst 3.87e-3), Serial fresh-oracle==M5.19 bit-identical, deep_copy 3413→2176 (−36 %), step −17.6 %, 1-yr climate corr=1.00000 vs M5.19 (zero cost). Remaining drivers (M5.21): `ghats` 256 + `S` 249 MB/step D2H — both DELIBERATE host-stays (M5.7 "ghats stays host"; L36/L39 host salinity floor), so each needs its host consumer ported or the decision reconsidered (harder than a placebo removal or a forcing port). Memory [[project-m520-pcie-residency]]; `docs/GPU_FIDELITY.md` §M5.20.

- **L66 — M5.21 (2026-05-30): the flat lever FINISHES bucket-A (8 FCT sub-kernels + `update_vel`) = −8.1 % step, bit-identical, fully attributed.** The L63→L64 lever applied to the last classified bucket-A maps in `fesom_tracer_adv.cpp` (`upw1v`,`LO_final`,`grad_elem`,`zal_a1`,`zal_a2`,`f2d_v`,`ale_recon`,`qr4c_v`) + `fesom_update_vel`. Re-profile (the discipline) **confirmed** the plan this time (`13_fct` still #1 at 25 %; the bucket-A kernels were a *bigger* share than the M5.19 doc estimated because the step shrank under them → ~7.9 % of loop). Same recipe verbatim: `RangePolicy(0,N*nl)`, decode `(n,nz)`, mask, register-write; the field's LEVEL is the contiguous inner dim so the warp coalesces. Lessons:
  1. **A same-level read-modify-write whose result depends on the original loop's *processing order* IS a within-column write-dep — but can still be flat-levered if you reproduce its NET effect.** `fct_qr4c_v` does `aflux_v[k] = HO(k) − aflux_v[k]` per level (fine — vertical reads `valsAB[nz±2]`/`Zc` are INPUTS), EXCEPT for a 2-level-deep column (`nzmax−nzmin==2`) where its "2nd-layer" and "bottom-1" cases write the SAME level twice, the 2nd reading the 1st's result with the identical term → they **cancel to a net no-op** (the level keeps its upw1v value). A naive one-formula-per-level flat lever applies it once (wrong); the fix recognises the cancellation and leaves that level untouched (`if(nzmax-nzmin==2 && nz==nzmin+1) return`), with an explicit surface/2nd/bottom-1/bottom/interior branch dispatch for the rest. Moral: re-derive the original's final value *per level for every column depth* (D=1,2,3,≥4) before flipping — the cancellation exists only at D=2.
  2. **pi cannot exercise shallow-column edge cases — CORE2-Serial fresh-vs-saved is the load-bearing proof.** pi's idealised mesh has no 2-level columns, so pi `tradv=0` did NOT cover qr4c_v's D=2 branch. The decisive check = `cp -r serref_core2 serref_saved`; gate `--fresh-oracle` rebuilds it from the new Serial; `diff_snap.py` saved-vs-fresh = ALL-FIELDS-BIT-IDENTICAL → proves bit-identical on real bathymetry incl. the shallow nodes. (Same harness as the M5.20 forcing-port proof L65.6 — reused for a *compute* edge case here.)
  3. **A pure-compute lever leaves `deep_copy` untouched — use that as the A/B sanity cross-check.** The same-node A/B showed `deep_copy` identical at 2175.84 MB/step on both legs → confirms no accidental PCIe change (and re-confirms M5.20's 2176). The win was −8.07 % clean (1.6359→1.5039 s/step); each flipped kernel dropped **−78…−92 %** (the M5.18 smoother signature), and the 9 per-kernel s/step reductions summed to ~0.131 ≈ the −0.132 A/B delta → fully attributed to coalescing, no hidden cost. Gate PASS (worst h_ice 1.19e-2 = the unchanged scatter floor — bit-identical Serial ⇒ the flipped no-atomic kernels add zero CUDA divergence). **The clean flat lever is now exhausted; M5.22's compute wins (FCT `*_h` scatters, the "b" cluster, the TDMAs) need edge-coloring / TeamPolicy / Lever C.** Memory [[project-m521-coalescing-finish]]; `docs/GPU_FIDELITY.md` §M5.21.

- **L67 — M5.21 Lever 2 (2026-05-30): a doc's "deliberate host-stay" is a HYPOTHESIS, not a fact — re-derive PCIe drivers from CURRENT code before optimizing (the user caught a 2-for-2 stale plan).** The prompt's Lever-2 named `ghats` 256 + `S` 249 MB/step D2H as host-stays to "port the consumer". A user question ("didn't we already do the salinity floor on device?") triggered a re-trace; BOTH attributions were wrong. Lessons:
  1. **`ghats` was a PLACEBO (computed + synced + halo'd, consumed by NOTHING).** `use_kpp_nonlclflx=.false.` ⇒ the KPP non-local flux is skipped, so the per-step `ghats` `sync_host`+halo served only an off-by-default debug dump. Guarding it on the dump flag (`fesom_kpp.cpp:1597`) → **−3.08 % NG5, bit-identical, deep_copy −244 MB/step (calls 121→120 = exactly the one D2H)**. The **decisive proof = SYNCCHECK with the producing feature ACTIVE** (pi runs KPP, confirmed via the `kpp` verify firing): a removed sync that left `ghats` device-authoritative would abort SYNCCHECK on ANY host read — 0 aborts ⇒ empirically no consumer. (Same −Nx-bandwidth fence-stall shape as M5.20/L65: −3.08 % ≈ 4–5× ghats's 256-MB bandwidth.)
  2. **`S` 249 D2H was REDUNDANT host work, NOT a port target — the thing was ALREADY on device.** The salinity floor went to device at M5.14 (L57.2); the SSS restoring + virtual-salt + runoff are ALL on device too (`fesom_ice_oce_fluxes_kk` reads `S` device-side, computes `relax_salt`/`virtual_salt` with device global integrals `integrate_nod_2D_kk`; runoff folds into the ice freshwater flux). The 249 D2H is the host `fesom_sss_runoff_step_cal` (`fesom_main.cpp:1073`) recomputing those fluxes on host (reading `S`) only to be **overwritten** by the device ice oce_fluxes one substep later — pure waste. So "move SSS restoring to device" = a control-flow trim (split off the monthly `Ssurf` read, re-gate the per-step half on the ice-OFF fallback `s_no_ice_thermo`), NOT a kernel port. **Generalizable trap:** when a forcing/coupling field round-trips, check whether a LATER device step overwrites it — a redundant pre-compute looks like a "host-stay" in the synclog but has no real host consumer.
  3. **The synclog binary can be STALE.** `fesom_port_synclog` was the pre-M5.20 build → it still showed `hnode_new`/`sw_3d` (gone since M5.20) and 3413 MB/step. Rebuild the `-DFESOM_SYNC_LOG=ON` binary on current source (a fresh `build-cuda-synclog` dir, ~20 min) before trusting per-field PCIe. Memory [[project-m521-coalescing-finish]]; `docs/GPU_FIDELITY.md` §M5.21 Lever-2.

- **L68 — M5.22 (2026-05-31): re-establish the WHOLE step budget before picking a lever — at 4 nodes the step is compute-bound, at 16 it's comm-bound, so the optimization frontier SPLITS.** The campaign had only ever re-profiled *one kernel at a time*; M5.22 re-derived the whole step budget from scratch (nsys CUDA+MPI + `FESOM_STEP_PROFILE` + `FESOM_SYNC_LOG` + halo barrier-isolation, at BOTH node counts — `docs/PROFILE_M522.md`). The L56 "75 % PCIe" is dead. Lessons:
  1. **Budget at MORE THAN ONE node count.** NG5 **4 nodes (dist_16, 1.34 s/step) = COMPUTE-bound** (Σ GPU-kernel ≈46 % of wall; PCIe RETIRED — synclog 659 MB/step all forcing/nod2D, the nod3D round-trippers `hnode_new`/`sw_3d`/`ghats`/`S` confirmed gone; halo comm 4.6 % / load-imbalance 9.7 %). NG5 **16 nodes (dist_64, 0.56 s/step) = COMM-bound** (per-rank GPU work ÷~4 → GPU-active 28 %; `MPI_Waitall` 82.6 % of MPI time). The levers that pay at 4 nodes (compute: bucket C/D) buy little at 16, where the wall is the halo + the ~9.7 %-and-growing imbalance. A single-node-count profile mis-ranks the levers. This IS the SCALING_M522 shrink mechanism (NG5 2.95×@4N → 2.38×@16N). **The frontier split:** single-node compute (Lever C / edge-coloring) vs comm/scaling (Lever D re-partition) are now co-equal; PCIe and Lever-B (comm overlap, ceiling 4.6 %) are retired.
  2. **A "finished" flat-lever sweep can still have missed a clean bucket-A subset — re-read the code, don't trust the prior sweep's coverage.** The §M5.19/§M5.21 sweep had declared bucket-A done, but the budget + a fresh code-read found 3 FCT Zalesak kernels still one-thread-per-NODE that are clean bucket A: `fct_zal_b1v` (`fesom_tracer_adv.cpp:1759`, reads input-only `aflux_v[nz+1]`), `fct_zal_b2` (`:1785`, pure per-level map), `fct_zal_b3v` (`:1801`). **The discriminator for b3v** (which a first read mis-called "marginal-B"): trace what the `nz±1` index *actually references* — b3v's `nz-1` reads are of `fplus/fminus` (the b2 *output*, an INPUT here), NOT a recurrence on its own just-written `aflux_v`, so it's bucket A (the surface case `nz==nu1` folds into the mask). Only a recurrence on the SAME field's just-written values (the TDMAs' `cp[nz]←cp[nz-1]`) is bucket C. Flipped all 3 (M5.18 flat lever): **−3.10 % NG5 (1.3387→1.2972 s/step, job 25254299), BIT-IDENTICAL** (Serial `tradv` `max|Δ|=0`; CORE2-Serial fresh-vs-saved ALL FIELDS BIT-IDENTICAL; CUDA gate PASS worst 7.5e-3 = unchanged floor; deep_copy unchanged = pure-compute).
  3. **Lever C is feasible to PROTOTYPE on one TDMA without the full refactor.** The full rank-1→`View<double**>` LEVEL-outer change touches 82 fields + 896 index-macro sites (`FESOM_NODE3D`/`ELEM3D`/`ELEMVEC`, `fesom_types.h:33-35`) — too big to commit blind. But ONE TDMA (`impl_vert_diff_tracers`, `fesom_tracer_diff.cpp:438`) can be de-risked with a dedicated `View<double**,LayoutLeft>` scratch + transpose-in/out around the one kernel, ncu-measuring the coalescing win, touching ZERO of the 82 fields. That's the recommended M5.23 first step for the compute frontier (M5.22 mapped it; did not build it). Memory [[project-m522-deep-profile]]; `docs/GPU_FIDELITY.md` §M5.22.

- **L69 — M5.23 (2026-05-31): the first COMM-regime win (EVP two-field fused halo, −9.1%) + measure a comm lever in the COMM regime + never double-submit to a shared output dir.** L1 of the §8.4 comm menu: the EVP subcycle exchanged `uice` then `vice` as two adjacent NOD2D nc=1 halos ×120 subcycles = 240 msgs/step (the #1 message-count contributor). Fused → ONE msg/neighbour via a new two-field device exchange `fesom_halo_exchange_device2` + dispatch `fesom_halo_field2` (`fesom_halo_device.{cpp,hpp}`), co-packing `[f0(stride) f1(stride)]` per halo node (stride 2×; flat buffer-offset collapse + race-free unpack carry over). Call site `fesom_ice_evp.cpp:716`. Binary `fesom_port_m522c`. Sub-lessons:
  1. **A comm lever shows NOTHING in the compute regime — A/B it at the right per-rank size.** NG5 dist_16 (463k 2D-pts/rank, 4 nodes, COMPUTE-bound) was ~flat for L1; the win only appears at the comm-bound per-rank size, reproduced cheaply via [[feedback-per-rank-proxy]]: **dars 8N (99k/rank = NG5@16N proxy) 0.3836→0.3487 = −9.1%**, farc 2N (80k/rank) −7.4%; calls/step 584→444 / 970→830 (exactly 140 fewer = 120 subcyc×(2→1)+tally), halo MB/step UNCHANGED → pure message-count cut.
  2. **Bit-identity is FREE by construction (M5.1 Approach-B): the device fused path is `#ifdef KOKKOS_ENABLE_CUDA`; Serial/OpenMP/`FESOM_HOST_HALO=1` fall back to the EXACT two legacy host brackets** → the Serial oracle can't change. Proven anyway: fresh CORE2-Serial ALL FIELDS BIT-IDENTICAL + an independent per-variable `np.array_equal` (81/81) + CUDA gate PASS (worst 4.5e-3, unchanged floor; the fused exchange adds zero atomics).
  3. **⚠️ NEVER double-submit two jobs that `rm -rf`+write the SAME output dir — it produces a FALSE failure.** I re-issued an already-sent submit batch after a transient channel stall; the duplicate's `rm -rf "$OUT"` deleted `snap_000010.nc` mid-compare → `diff_snap` reported "DIVERGENCE" (MISSING file), which looked like a real bit-identity FAIL. The clean re-run alone confirmed bit-identical. **A collision only ever BREAKS a result (missing/partial output), never fakes a PASS** — so on a "DIVERGENCE", first check all expected snapshots exist before suspecting a code bug, and verify what actually reached SLURM before re-submitting. The two-field entry point now makes the L3 adjacent pairs (FCT plus/minus, PGF, visc, GM, hnode-helem) cheap follow-ons. Memory [[project-m522-deep-profile]]; `docs/PROFILE_M522.md` §8.6, `docs/GPU_FIDELITY.md` §M5.22-comm.

- **L70 — M5.23 (2026-05-31): L3 — stack the cheap comm pairs on the L1 entry point (10 same-kind halo fusions, −2.4 % at the NG5@16N proxy, bit-identical); the deterministic msg-count drop is the proof, not the timing; the win is bounded by how comm-bound the regime is.** Dropped `fesom_halo_field2` onto the 10 VERIFIED adjacent same-kind/same-stride pairs (FCT plus/minus ×2/step, EOS density+hpressure & sw_alpha+sw_beta, PGF, GM neutral_slope+slope_tapered & fer_K+Ki, vert-vel w+fer_w & w_e+w_i, visc u_b+v_b, bulk heat+water). Binary `fesom_port_m523L3`. Sub-lessons:
  1. **A pure drop-in on a proven entry point needs NO new machinery — the win is just finding the pairs.** L1 built `fesom_halo_field2`; L3 was 10 one-line edits (2 `fesom_halo_field` → 1 `fesom_halo_field2`). The whole effort was the code-read + per-pair verification (same kind/n_levels/n_components/base_off + literally adjacent + both fields owned-written before + nothing reads/writes between). Verified-not-fused: `Kv`+`Av` & `hnode`+`helem` (DIFFERENT kinds NOD3D vs ELEM3D — the prompt over-listed them; the over-list cost nothing because I re-read), `uv_rhs`+`uv_rhsAB` (reserved for L5 poison-test — if dead, removing beats fusing), `diffK` two-slab (one field → needs a slab-fuser).
  2. **The DETERMINISTIC message-count drop is the lever's proof; the timing % is the payoff (and is regime-dependent).** `FESOM_HALO_MPI_PROF` + the `[fesom_prof]` halo_pack/pack2 counts gave an EXACT, mesh-INVARIANT confirmation the fusion fired: single-field exchanges −21.8 (farc) / −21.9 (dars), two-field +11.0 (both) = the 11 fused-pair invocations/step (FCT×2 + 9 others incl. 3 GM → GM active on dars), MPI calls/step −12 (both), halo MB/step UNCHANGED. Trust that over the timing jitter. The timing: dars 8N (CG ~13 %) −2.38 %, farc 2N (CG 37 %, L3 doesn't touch CG) −1.06 % → the L3 payoff scales with the non-CG comm fraction, so the dars/NG5-class number is the one that matters.
  3. **The Serial bit-id can't catch an adjacency error; the CUDA gate must.** field2 on Serial = two sequential brackets REGARDLESS of adjacency → the Serial diff is trivially bit-identical (it only catches kind/nl/nc/field typos). A genuine adjacency error (co-packing STALE owned data because a kernel between the two calls writes f1) only bites the CUDA co-pack. The gate caught nothing because all 10 ARE adjacent — confirmed by every fused-halo-downstream field (density/bvfreq/T/S/u/v/w/Av/Kv) sitting at the climate floor (worst h_ice 7.29e-3). Always run the gate for a fusion, not just the Serial diff.
  4. **Why −2.4 % not −9 %: there is no second EVP.** L1's pair ran 120×/step (per subcycle); every L3 pair is structural/once-per-step (FCT 2×). The high-frequency halo surface was a single hotspot (EVP); the rest is the long once-per-step tail. Stacking all 10 = −2.4 %. Honest takeaway for the SYPD goal: the cheap comm pairs are now exhausted; the remaining climate-safe comm levers (L2 persistent requests, L5 poison-test, a fieldN for the EOS 5-block) are each ≤~1–3 %, and **mixed precision (≈×2) is the only lever that reaches 2 SYPD**. Memory [[project-m522-deep-profile]]; `docs/PROFILE_M522.md` §8.7, `docs/GPU_FIDELITY.md` §M5.23.

- **L71 — M5.23 (2026-05-31): finish the cheap comm grind — L5 dead-halo poison-test (1 free removal), fieldN EOS 5→1 (−0.6 %), L2 persistent requests = MEASURED DEAD END; the grind has PLATEAUED → mixed precision is next.** Three levers off the §8.4 menu, each measured in the comm regime (dars 8N = NG5@16N proxy), each held to Serial-bit-id + the CUDA gate. Sub-lessons:
  1. **L5 poison-test (the L67 method, now on a HALO): NaN the halo right after the exchange, run the gate — PASS-while-poisoned ⇒ no downstream reader ⇒ DEAD ⇒ remove.** A gated device kernel NaNs the field's halo tail `[myDim·stride, size)` after the exchange (`FESOM_POISON_<F>=1`). **`bvfreq` (step.cpp:216) = NEEDED**: the device smoother `fesom_smooth_nod3D_kk` gathers it at element vertices INCL. halo nodes (eos.cpp:560) → the poison CRASHED the run = the POSITIVE CONTROL (proves the NaN propagates). **`uv_rhsAB` (step.cpp:467) = DEAD**: `compute_vel_rhs_kk` reads it only at OWNED elements (`E=myDim_elem2D`); momadv scatters INTO it but its flux reads `uvnode_rhs`, not `uv_rhsAB` → nothing reads its halo. Poison PASSED the gate (worst 8.4e-3, NaN-free) → **removed the exchange = a whole ELEM3D nl=2 halo + 2 fences/step, bit-identical** (Serial all-fields). **ALWAYS run a KNOWN-LIVE field as the positive control in the SAME job** — its crash proves a clean PASS elsewhere isn't a broken-poison false-negative. (NaN is safe here: bvfreq's smoother is a SUM, uv_rhsAB's consumers are LINEAR → no fmax/min absorption; if a consumer used fmax, switch to a huge finite poison.) The Serial bit-id ALSO confirms the dead-halo removal is bit-identical on the host path (dead is backend-independent — it's about whether a reader exists).
  2. **fieldN (EOS 5-block 3→1): generalize field2 to N fields — N pack kernels → 1 fence → 1 msg/neighbour → N unpack kernels.** New `fesom_halo_exchange_deviceN(Field *const*, nf, …)` + dispatch `fesom_halo_fieldN({&f0,…})`. The device-array-of-Views problem (a KOKKOS_LAMBDA can't deref a HOST `Field**`) is sidestepped by looping the N launches on the HOST, each capturing ONE `fi=fields[i]->d()` that writes the DISJOINT buffer slot `g·stride + i·fs` (no fence between the N packs — same CUDA stream serializes them + the slots are disjoint anyway). Folded density+hpressure+bvfreq+sw_alpha+sw_beta (all NOD3D nl 1; L3 had taken them 5→3) into ONE exchange. **−0.56 % dars-8N / −0.78 % farc-2N, bit-identical** (Serial ALL-FIELDS + gate worst 5.7e-3; deterministic **−3 MPI calls/step** = −2 EOS + −1 uv_rhsAB; halo −1…2 MB/step/rank = the uv_rhsAB removal, fieldN co-pack keeps bytes constant). `nf==2` reproduces device2's layout exactly, so device2 could later be retired onto deviceN.
  3. **L2 persistent MPI requests = MEASURED DEAD END (reverted).** Implemented `MPI_Recv_init`/`Send_init` once per (kind,stride) + `Startall`/`Waitall` per call (one shared helper for device/device2/deviceN; rebuild-on-`grow()` via base-pointer staleness; env toggle for a same-binary A/B). **Gate PASSED** — device-ptr persistent requests DO work in OpenMPI-4.1.5-nvhpc/UCX and move byte-identical data. But the same-binary A/B was **flat-to-SLOWER: farc-2N fresh 0.3015 → persist 0.3023 (+0.27 %), dars-8N 0.3681→0.3683 (+0.05 %)**. WHY: UCX Irecv/Isend POSTING is already cheap; the real per-exchange cost is the 2 `Kokkos::fence()`s + the transfer + the load-imbalance Waitall — NONE of which persistent requests touch. And L2 doesn't change the calls/step (still 1 Waitall/exchange) → no deterministic proof, only noisy timing. **Reverted** to the inline fresh path. **Worth retrying on LUMI/AMD** (Cray-MPICH has a different request-setup cost) — the implementation is sketched in this lesson. (Same shape as L62/M5.17: a comm idea the regime's real wall — fences/imbalance — makes a no-op.)
  4. **The cheap comm grind has PLATEAUED — decision returns to mixed precision.** L1 (EVP, fired 120×/step) −9.1 % → L3 (10 structural pairs) −2.4 % → fieldN+L5 −0.6 % → L2 dead. The once-per-step structural halo surface is EXHAUSTED. The remaining climate-safe comm levers: **L4** (CG 2→1 Allreduce, Chronopoulos/Eijkhout — CG is 14 % dars / 38 % farc of the step, the higher-potential one, but NOT bit-identical → reassociates the reduction → gate-class + needs the 1-yr climate; parked) and **mixed precision (≈×2 — the ONLY lever that reaches the 2-SYPD target, halving BOTH compute and comm bytes).** Per the user's M5.23 deferral, mixed precision re-surfaces now. Binary `fesom_port_m523fN` (= live `fesom_port`, fieldN+L5); BEFORE `fesom_port_m523L3`. Memory [[project-m522-deep-profile]]; `docs/PROFILE_M522.md` §8.8, `docs/GPU_FIDELITY.md` §M5.23.

- **L72 — M5.24 (2026-05-31): the first COMPUTE-frontier lever — in-place Thomas aliasing shrinks the 3 ocean vertical-solver per-thread local frames; bit-identical, GPU kernels −6.5…−19.5 %, step ~−2 % CORE2. The generalizable rule: storage-reuse only pays on genuinely MEMORY-bound column kernels (pressure_bv fusion = +28 % dead-end, reverted).** Diagnosis (static `cuobjdump --dump-resource-usage` + ncu CORE2 single-GPU): `impl_vert_visc`, `diff_ver_part_impl_ale` (tracer), `fer_solve_gamma` (GM) are local-memory-traffic bound (~3–7 % compute SM throughput, ~50 % memory); each spills 8–10 `real_t[FESOM_MAX_LEVELS=128]` per-thread scratch arrays (STACK 8192–10240 B/thread). The spill is **mesh-INDEPENDENT** (128 is compile-time; real nl=47/70) and the traffic is nl-bounded, so the lever is **fewer ARRAYS, not a smaller ceiling** — and you CAN'T size to exact nl (GPU device lambda-locals can't be VLAs → would force per-mesh compile). Sub-lessons:
  1. **In-place Thomas (the bit-identical storage reuse):** `cp` reuses `c` (modified-c), the forward-eliminated RHS reuses the RHS array (`tp→tr`; `up,vp→ur,vr`; `tp_x,tp_y→tr_x,tr_y`). Each `c[nz]`/RHS`[nz]` is read THEN overwritten in level order → provably identical FP. Coded minimally as `real_t *cp = c, *tp = tr;` (pointers into the existing arrays, declared AFTER them). visc/gm 10→7 (10240→7168), tracer 8→6 (8192→6144 + trim the redundant `0..NL_MAX` init zeroing of the aliased pair). **Serial ALL-FIELDS bit-id + gate PASS**; kernels impl_vert_visc −19.5 %, tracer −10.2 % (×2/step), GM −6.5 %; step CORE2 single-GPU ~−2.2 %.
  2. **Mechanism = CACHE LOCALITY, not occupancy or instruction count.** The aliasing does NOT cut the store-instruction COUNT (cp/c are the same writes) — it cuts the reserved frame + the init zeroing, so the smaller per-thread working set stays resident in L1. Proof: `impl_vert_visc` got 19.5 % FASTER even though its achieved occupancy DROPPED (32.6→27.3 %, REG 80→82). ⇒ on a spill-bound column kernel, shrinking the live array set beats chasing occupancy.
  3. **pressure_bv = MEASURED DEAD END (reverted) — the lever's BOUNDARY.** Fused the hpressure cumulative loop into Pass 2 to drop `rho[]` (5120→4096, **bit-id PASS**) but it ran **+28.3 % SLOWER** (1.69→2.17 ms). pressure_bv is NOT memory-bound — it runs the JM-EOS polynomial evals (real compute), so the frame-shrink bought nothing while the fusion ADDED a branch (`if nz==nzmin`) + serialized the hpressure top-down dependency into the hot loop. `git restore`. **ALWAYS read ncu compute % BEFORE applying this lever** — it's a memory-bound-only tool; on compute-bound kernels the restructure costs more than the frame saves. `redi_ver_node` (5120) + `kpp_blmix` (4096) have NO clean reuse (txn/tyn/zbar_n/z_n consumed together; dcol[128][3]/dthick live whole-kernel). ⇒ the fewer-arrays lever is **EXHAUSTED at the 3 TDMAs**.
  4. **Orthogonal next lever: mixed precision** would halve the bytes of every one of these spill arrays (the same kernels) → the compute-frontier and the mixed-precision lever hit the same hot columns. Binary `fesom_port_m524tdma`; jobs `job_ncu_tdma`/`job_ncu_m524_ab` (profile), `job_tdma_ab` (step A/B), `job_core2_serial_m524` (Serial bit-id). Memory [[project-m524-tdma-footprint]]; `docs/GPU_FIDELITY.md` §M5.24.

- **L73 — M6 Phase 0 (2026-07-12): a bit-gate that has stood for months can be broken by an INPUT you don't version — the shared mesh changed under us. Fingerprint your inputs, and keep a regression rung that compares against the STANDING oracle, not just against the run you launched five minutes ago.** The M6 oracle-certification job ran two legs (new C oracle vs Kokkos-Serial) and I added a third, unasked-for gate: Kokkos-Serial vs the standing `serref_m522_saved` oracle from 5 weeks earlier. The paired gate PASSED; **the regression gate FAILED** — at step 0, on `nlevels`/`nlevels_nod2D`. Those are static mesh-topology fields: they cannot change unless the *mesh* changes. Root cause: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/nlvls.out`+`elvls.out` were **replaced on 2026-07-03** (2 nodes / 4 elements on the Ross Sea shelf, all shallower; `sum(nlevels_nod2D)` 3 832 750 → 3 832 745). The port reads those files directly (`fesom_mesh.c:340-361`), so it is a real bathymetry change — and *every* archived C/Fortran/Kokkos reference predates it. Sub-lessons:
  - **The paired gate alone would have shipped a lie.** Both of today's legs read today's mesh, so they agreed perfectly with each other and the certification "passed". Only the comparison against a months-old artifact could see that the *world* had moved. **A same-day paired gate proves A==B; it cannot prove A==the-thing-you-validated-last-month.** Keep at least one rung anchored to a frozen artifact.
  - **Read the failure's SHAPE, not its size.** The diff also showed `S` off by 34.6 and `T` by 1.3 — panic numbers. But `lat`/`lon`/`elem_nodes` were *bit-identical* while `nlevels*` were not, which is a combination no code change can produce. The huge tracer diffs were the *consequence* (levels that are wet in one run and dry in the other), not the cause. **A static-topology field in the diff list means the mesh moved; look there first, and ignore the scary-looking physics fields.**
  - **Fix at the source, not in the comparator.** Options were: accept and caveat every future climate compare, or re-baseline everything, or take a **private copy of the mesh** with the reference bathymetry restored (`/work/ab0995/a270088/port2/mesh/core2`, the old levels recovered from the in-place `*.20260528_regenerated` backups, `/pool` untouched). The copy is 1.1 GB and made the problem *disappear* — with it, a fresh Kokkos-Serial run is bit-identical to `serref_m522_saved` again, so the entire reference ladder stays valid and no verdict carries an asterisk. **Never mutate a shared `/pool`-class directory to fix your own reproducibility; copy it and pin your copy.**
  - **Record the input fingerprint next to every archived reference.** `m6_baseline_serial/PROVENANCE.md` now carries the mesh path, the level-sums, and the `nlvls`/`elvls` mtimes. Cheap; would have turned a half-hour investigation into a one-line diff.

- **L74 — M6 Phase 0 (2026-07-12): the port writes vectors in the ROTATED frame and Fortran writes GEOGRAPHIC — and because the rotation is an ISOMETRY, the mismatch hides behind every magnitude-based metric. It cost this project a fictitious "known F↔C ice-edge budget" of uice corr ≈0.92 that stood for the whole M5 campaign.** FESOM computes on a rotated grid (CORE2 Euler 50/15/−90); `(u,v)`/`(uice,vice)` are rotated-frame internally. Fortran's `io_meandata` rotates them to geographic at output (`do_rotation`); the Kokkos port writes them as stored. `m32_climate_compare.py` compared them raw. Measured on the M5.23 CUDA 1-yr run vs the Fortran linfs+KPP reference, everything else held fixed: **`uice` 0.9187 → 0.9997 and `vice` 0.4266 → 0.9998** once the port's vectors are rotated r2g. 0.9187 reproduces the recorded 0.919 to four digits. Cross-checked purely C-side: `c_tke_2yr` (rotated) vs `fortran_linfs_tke` (geo) goes 0.9187 → **1.0000**. The port was always right; the *comparator* was wrong. Sub-lessons:
  - **An isometry is the most dangerous class of comparison bug, because every sanity check you'd naturally reach for still passes.** `|speed|` is invariant to 1e-16. Ice extent, ice volume, drift speed — all perfect. Only the *components* decorrelate. If a vector field's magnitude agrees but its components don't, suspect a frame before you suspect physics.
  - **It hid because only HALF the vector pair was ever compared.** `FIELDS` had `uice` but not `vice`. `uice` degraded to a plausible 0.92 — low enough to invent a story about ("marginal ice edge"), high enough not to alarm. `vice` was at **0.43** and nobody was looking. **Never compare one component of a vector pair; the partner is where the frame bug announces itself.**
  - **A plausible story is how a measurement error becomes doctrine.** The 0.92 got an explanation ("a real, modest C-vs-Fortran ice-velocity/EVP difference"), the explanation got written into `REFERENCE_RUNS.md`, `GPU_FIDELITY.md` §M5.13–15 and the project memory, and thereafter it was a *known budget* that nobody re-derived. The tell was available all along: `uice`-vs-C-port was 0.9998 while `uice`-vs-Fortran was 0.92, and the C port is bit-identical to Serial Kokkos — so the gap had to live in something that differed between the two *comparisons*, not in the port.
  - **The fix is a shared transform, not a per-script hack**: `scripts/fesom_frame.py` holds the `vector_r2g` port (`gen_modules_rotate_grid.F90:164-202`) plus a table of which archived C output is in which frame (the C tree flipped its default to geographic at `75406d3`, so its own outputs are split). Verify a rotation two-sidedly before trusting it: it must *improve* a known rotated-vs-geo pair AND *degrade* a known geo-vs-geo pair.

- **L75 — M6.1 (2026-07-12): a CUDA fidelity gate that thresholds on `max|Δ|` cannot tell ONE flipped node from a domain-wide corruption. Gate on COHERENCE (how many entries are over ceiling), not on the maximum. The TKE port failed the gate on 1 entry out of 5,962,326 and was completely correct.** The knob-ON gate FAILed on `h_ice` (1.48e-01 vs ceil 1e-01) and `vice` (6.13e-02 vs 5e-02) even though Kokkos-Serial was ALL-FIELDS BIT-IDENTICAL to the C oracle at the same config and the column core was bit-identical over 4000 randomised columns. Sub-lessons — the diagnostic ladder is the reusable part:
  - **Discriminator 1 — determinism.** A staleness/correctness bug is DETERMINISTIC: every CUDA run makes the same mistake, so two CUDA runs agree with each other and both differ from Serial. Atomic-scatter nondeterminism is the opposite. Run CUDA **twice** and compare: for the ice fields `|CUDA_A − CUDA_B|` was **10–135× LARGER** than `|CUDA − Serial|` → the FAIL was just a run-to-run draw from the EVP scatter spread. (Restricted to real ice pack `a_ice > 0.15`, the diffs were 1.8e-04 / 4.6e-04 — the headline numbers were entirely the `h_ice = m_ice/a_ice` blow-up where `a_ice ≈ 0.005`. **A ratio diagnostic with a vanishing denominator is not a fidelity signal.**)
  - **Discriminator 2 — growth shape.** `Kv`/`Av` came out the *other* way (two CUDA runs agreed to 1e-5 while both differed from Serial by 1e-1) — the one signature that really would mean a bug, so it needed a real answer, not a story. Snapshot **every step** and plot the relative error: chaotic amplification of a rounding seed grows MONOTONICALLY. Kv instead sat at ~1e-6 at every step **except step 5**, where it spiked to 2.4e-2 and fell straight back. **A spike that comes and goes is not chaos and is not a bug — it is a branch flip.**
  - **Discriminator 3 — spatial coherence, the decisive one.** Count the entries over ceiling. The step-5 Kv spike was **1 entry out of 5,962,326**; steps 4 and 6 had zero above 1e-4. The gate's h_ice/vice failures were **1 entry out of 126,858 each**. A stale-halo bug corrupts whole halo regions and amplifies domain-wide — thousands to millions of entries. **One entry is never that.**
  - **The mechanism, and why it will recur.** Schemes built on COMPARE-SELECT CLAMPS produce isolated *finite* differences under any floating-point divergence. TKE is full of them: `prandtl = max(1, min(10, 6.6·Ri))`, `KappaM = min(KappaM_max, c_k·mxl·√e)`, the mxl min-chain. A node within **1 ULP** of a clamp boundary lands on OPPOSITE SIDES under CUDA's libdevice math and the host's glibc; the branch flips; that node's Kv jumps by a finite amount. It is deterministic (same device math every run) — which is exactly why Kv/Av looked "deterministic vs Serial" and nearly sent the hunt after a phantom sync bug. **Every clamped scheme (mEVP's `delta_min`, zstar's thickness floors) will do this. Check the count before believing a FAIL.**
  - **The fix belongs in the GATE, not in the ceilings.** `gpu_fidelity_check.py` now fails a field only when it is over ceiling at **> `OUTLIER_TOL` (32) entries**, and always prints the count. Raising the ceilings would have blinded the KPP path; adding a coherence criterion made the gate *sharper* for every config, because coherence is the property that actually distinguishes the M5.9-class bug the gate exists to catch.

- **L76 — M6.1 (2026-07-12): two nvcc/Kokkos mechanics worth knowing before you hit them.**
  - **An extended `__host__ __device__` lambda cannot FIRST-capture a variable inside an `if constexpr` block** ("An extended __host__ __device__ lambda cannot first-capture variable in constexpr-if context"). The TKE column lambda used `if constexpr (WITH_DIAG)` around the 13 diag views, which appear nowhere else in the lambda → 13 hard errors. Fix: a plain `if (WITH_DIAG)` on the compile-time-constant bool. The branch still folds away identically (same codegen, same dead-code elimination of the unused `.data()` reads) but the capture is legal. `if constexpr` inside a *normal* `KOKKOS_INLINE_FUNCTION` is fine — the restriction is specific to extended lambdas.
  - **Templating a device kernel on a compile-time `WITH_DIAG` bool is a free way to shrink a fat per-thread frame.** The C's `integrate_tke` computes 13 budget slabs unconditionally into `[129]`-double locals. They are pure OUTPUTS (nothing in `tke_new`/`KappaM`/`KappaH` reads them back), so with `WITH_DIAG=false` the compiler proves all 13 dead and eliminates them: **~33 KB → ~20 KB per thread**. The computation is still *written* unconditionally, exactly as the C writes it — this is DCE of unused outputs, not a reordering, so numerics and evaluation order are untouched (and the twin gate runs with `WITH_DIAG=true`, so the diag path is verified too). Faithfulness and frame size are not in conflict here.

- **L77 — M6.1 (2026-07-12): NEVER source `env.sh` and `env_cuda.sh` in the same script.** A CUDA leg that had worked all day segfaulted in `ucp_dt_pack` ("invalid permissions for mapped object") the moment I put a Serial leg in front of it. `env_cuda.sh` does `module --force purge` and loads the CUDA-aware `openmpi/4.1.5-nvhpc-24.7` — but it does NOT set `UCX_TLS`, and **`env.sh`'s exported `UCX_TLS=mm,knem,cma,dc_mlx5,dc_x,self` (a host-only transport list, no `cuda_copy`/`cuda_ipc`) survives the module swap**. UCX then tries to pack DEVICE pointers through a host transport → segfault. The failure looks like MPI/hardware flakiness, not like an env bug. **Rule: one env per script.** Split paired Serial/CUDA comparisons into two SLURM jobs (or `unset` every `UCX_*`/`OMPI_MCA_*` before switching).

- **L78 — M6.3 (2026-07-12): when a new mode makes a previously-CONSTANT array time-varying, every read of it becomes a bug site — and at COLD START the new array is BITWISE EQUAL to the old one, so step 1 passes and step 2 fails. Grep for the reads; do not trust that you re-pointed them.** zstar makes `Z_3d_n`/`zbar_3d_n` (per-node depths) live, where linfs holds them equal to the static 1-D `Z`/`zbar` forever. Task 3.6 was *called* "geometry re-points audit" and I still shipped **8 wrong sites**. The bit-id failed with snapshots 0 and 1 EXACTLY clean and step 2 diverging — the signature is diagnostic, not incidental: **at cold start `hbar == 0`, so `Z_3d_n == Z` bit-for-bit**; a missed re-point is invisible until the surface has actually moved.
  - **The live bug was ONE line**: `fesom_kpp.cpp:937`, the device kernel's `delhat = |Z(kn)| - hbl` reading the static `Z` where the C reads `Z_3d_n` (`fesom_kpp.c:516`). `delhat` drives the KPP boundary-layer shape function → `Kv`/`Av` shift ~9e-5 → `u`/`v` via the implicit vertical viscosity → `ssh_rhs` → `d_eta` → `eta_n`. The *sibling* kernel 60 lines up (line 365) HAD been re-pointed correctly. **A partially-applied sweep is more dangerous than an un-started one**: the correct siblings make the file look done.
  - **Host/device twins double every site.** The port keeps a host twin per device kernel (the `FESOM_KK_VERIFY` reference). I re-pointed the device kernels and left 7 host twins on the static arrays (`eos` ×6, `pp` ×1, `tracer_adv` ×1, `kpp` ×3). Same class as the M6.3 hpressure bug (gated the device twin, not the host one — the startup PGF then ran through the un-gated host path and broke snapshot 0). **A verify reference that disagrees with the C is not a harmless copy — it is a lying oracle.** Rule: `grep` for the symbol across *both* twins and diff the count against the C's, per file.
  - **Re-pointing too EAGERLY is also a deviation.** In `fesom_phc.cpp` I had swapped in `Z_3d_n` where the C reads the static `Z` — because I read the C's *comment* ("Fortran: -Z_3d_n[k,ii]") as if it were the C's *code* (the next line is `mesh->Z[k]`). Inert here (PHC runs once at IC, where the two are bitwise equal), but it is exactly the "fixing a C asymmetry" the golden rule forbids. **The C keeps static depths deliberately in several places** (`eos.c:174` hpressure surface term, `ale.c:75-87`, `gm.c:517/744`, `ic.c`, `ssh.c:144`, `momentum.c:326`, `tracer_diff.c:148`) — it is a site-by-site match, never a blanket swap.
  - **The bisect that found it** (reusable): a 6-tag gid-keyed dump at identical sites in both trees (`FESOM_ALE_DUMP_DIR`, `scripts/ale_dump_diff.py`) — forcing / pgf / sshsolve / hbar / Wvel / thickness. It killed my prime suspect outright (`s2 forcing` EXACTLY 0 on all 4 components ⇒ the freshwater balancing, `evaporation` and `ice_sublimation` were all bit-perfect) and re-ordered the causal chain: I had read a 1.9e-13 shift at 97.76% of nodes as "a global scalar off by one ULP", but **`Δu`=3.5e-4 is far too large to come from `Δd_eta`=2.5e-16** — `ssh_rhs` was DOWNSTREAM of `u`, not the seed. **Order your suspects by the magnitude they can actually produce, and let the elliptic solve explain the "global" smear** (a CG spreads a 4-node ULP seed to every node through its Allreduce dot products; "it's everywhere" is NOT evidence of a global scalar).

- **L79 — M6.2/M6.3 (2026-07-12): a climate correlation is meaningless until you have measured the C-vs-Fortran BASELINE *for that same scheme*. Every scheme has its own reproducibility floor, and reading one scheme's floor onto another is how a fake "known budget" is born.** The mEVP 1-yr close came back with `uice` 0.933 / `vice` 0.913 — alarming next to the TKE leg's 0.9997, and exactly the shape of a port bug. It is not one. Feeding the *C oracle itself* in as the backend (`m32_climate_compare.py --backend-frame geo`) and diffing it against the Fortran mEVP reference gives the floor:

  | scheme | what it changes | C-vs-Fortran `uice` | C-vs-Fortran `vice` |
  |---|---|---|---|
  | TKE | ocean vertical mixing only | ~1.0 | ~1.0 |
  | zstar | vertical coordinate only | **0.99999** | **0.99999** |
  | mEVP | **the ice rheology itself** | **0.95438** | **0.93908** |

  mEVP is a fixed-point iteration (α=β=250 pseudo-steps) that is only *approximately* converged, so an ice-velocity field is genuinely not reproducible to 1e-5 across two independent implementations — while TKE and zstar leave std EVP alone and reproduce `uice` at ~1.0. Our CUDA-mEVP (0.933/0.913) sits ~0.02 under its own scheme's floor, which is the D22 atomic-scatter noise amplified by that same iteration; every mass/scalar field stays ≥0.9999 (`a_ice` 0.99997, `m_ice` 0.99998, `sst`/`ssh` 1.00000) and **Kokkos-Serial mEVP is BIT-IDENTICAL to the C oracle**, so the port is provably exact. **Rule: before you judge a backend-vs-C number, spend the 30 seconds to measure C-vs-Fortran on the same two reference dirs.** It is free (the refs already exist), it needs no HPC allocation, and it is the difference between "the port is broken" and "this scheme has a 0.95 floor". This is L74's lesson in a new costume: do not accept a suspicious correlation as a *property of the port* until you have measured what the property actually is. Baselines are tabulated in `docs/REFERENCE_RUNS.md`.

### L80 — A knob that does nothing passes EVERY gate. Trust the arithmetic before the gates. (M7, 2026-07-14)

`src/fesom_speed.hpp` enforces "Serial stays legacy" with

```c
#ifndef KOKKOS_ENABLE_CUDA
    if (on && !fesom_speed_force_serial()) on = 0;
#endif
```

`KOKKOS_ENABLE_CUDA` comes from Kokkos' **generated config**. Include `fesom_speed.hpp` in a TU
*before* anything that pulls that config in, and the macro is not yet defined — the guard fires **even
on a CUDA build**, and **every knob in that TU silently resolves to OFF**. It killed
`FESOM_SPEED_SWSKIP` (`fesom_bulk.cpp`) and `FESOM_SPEED_IOACC` (`fesom_io.cpp`) while the same knobs
worked in `fesom_ice_coupling.cpp` and `fesom_halo_device.cpp` — a **per-TU, invisible** failure.

**Every correctness gate went green on the dead knob:**
- knob-OFF byte gate — passes, because knob-OFF is exactly what a dead knob gives you.
- CUDA fidelity gate — passes, because the output *is* the legacy output.
- **FORCE_SERIAL byte proof — passes, because `FORCE_SERIAL` bypasses the very guard that was killing
  the knob.** The strongest proof in the arsenal was structurally blind to this.

**What caught it was physics.** The lever removes a **261 MB/rank/step `memset`** and the A/B said it
cost **0.01%**. That is not a disappointing lever, it is *arithmetically impossible* — a 261 MB
single-threaded memset cannot cost under ~10 ms of a 1280 ms step.

**Rules:**
1. **Before believing a null A/B, verify the knob FIRED.** 0.00% is a claim about the code path.
2. **Sanity-check any payoff against a physical floor** (bytes moved ÷ bandwidth, flops ÷ rate). A
   measurement below the floor means the plumbing is wrong, not the hardware.
3. **Any header whose behaviour depends on a config macro must include that config itself.**
   `fesom_speed.hpp` now includes `<Kokkos_Macros.hpp>` → include-order-independent.
4. **Verify with the preprocessor, not by reading includes:** `nvcc -E <real flags> | grep <guard>`
   settles in seconds what an include-chain audit only guesses at.
5. **Make knobs announce themselves.** `fesom_speed_resolve` now prints one line on rank 0 when a
   lever turns ON, and shouts if it was requested and resolved to OFF.

### L81 — A profiler tells you WHERE the time is, never WHICH SOURCE LINE. (M7, 2026-07-14)

The M7 stall budget correctly found that **32–34% of the NG5@4N GPU step is host time** with no traced
CUDA call and no traced MPI call (confirmed two independent ways; uniform, stdev 1.5%), and bounded it
to the microsecond: a window starting after the last sea-ice kernel and ending exactly at
`fesom_cal_shortwave_rad_kk`. **`-t cuda,mpi` cannot see inside host code by construction** — it can
localise a cost but never name it.

I then *guessed* which function in that window was the cost, twice, and shipped a lever for each:
`fesom_ice_oce_fluxes_mom` (**−0.72%**) and the dead host `sw_3d` (whose knob turned out to be dead —
see L80). Both were real bugs; both levers are correct and bit-identical. The model's own phase timer
"confirmed" 24.6% — but **two agreeing measurements of the same *window* are not independent
confirmation of an *attribution*.**

**Rules:** localising ≠ attributing. To name host time, use **CPU sampling with call stacks**
(`nsys profile --sample=process-tree --backtrace=dwarf -t osrt`; note the values are
`process-tree|system-wide|none` — *not* `cpu`). And **only a same-allocation A/B turns an attribution
into a fact — run it before writing the number down.**

---

### L82 — CPU sampling names a function too. It can name the WRONG one — an inlined callee eats its caller's loop. (M7 A.2, 2026-07-14)

L81's fix for "the profiler can't name host time" was **CPU sampling with call stacks**. It worked: the
hostprof run pointed straight at the JRA55 forcing. But its top frame was a **lie**, and the lie was
plausible enough to have funded a whole package on the wrong function.

The sampler said `getcoeffld` (the JRA time-slice reader + bilinear interpolator) cost **38.3 ms/step**,
uniformly across all 24 steps of the window — with a clean stack, `getcoeffld ← fesom_jra55_step ← main`.

**It cannot cost anything.** `getcoeffld` only runs when `rdate` leaves the current 3-hourly JRA
interval — at dt=180 that is **once per 60 steps**, and the year-start prefetch covers step 1, so in a
25-step run **its body executes ZERO times.** (Verified by transcribing the C control flow to Python and
running it against the real `uas.1958.nc` time axis: refreshes land at steps 61 / 121 / 181.)

**Mechanism:** `getcoeffld` is `static` → inlined into its caller. `nsys --backtrace=dwarf` expands
**DWARF inline frames**, so PCs in the caller's own per-node loop — which sits right next to the inlined
body — get reported under the inlined callee's name. The stack looks perfect. It is still wrong.

**What actually settled it:** the model's **own host timer** (`FPROF` bracket `force:jra55_read` =
**75.2 ms/step at 1.0 calls/step**) plus **node arithmetic** (463 k nodes/rank × 4 `sincos` per node ≈
44 ms at 2.5 GHz). A timer around a call cannot misattribute; a sampler's symbol always can.

**Rules:**
- **A sampler localises to a REGION, and only *claims* to name a function.** Treat the function name as a
  hypothesis, not a result — especially for `static`/inlined/header code.
- **Before believing a per-step cost, ask how many times the thing is CALLED per step.** A function that
  cannot run every step cannot cost you every step. Control flow beats any profile.
- **Bracket the suspect region with a host timer.** It is minutes of work and it is the only host
  measurement that cannot lie about *which call* it is timing.
- L80's rule, restated: **the arithmetic is the arbiter.** It killed a fake 0.00% (L80) and now a fake
  38 ms/step.

*(The payoff for being suspicious: the real cost is the wind-rotation `sincos` — and its four values per
node depend only on the node's fixed coordinates, i.e. it is a **mesh constant recomputed 1.85 M times
per rank per step**. Task D.0 `ROTCACHE` caches it: ~20 lines, bit-identical, ~−4.4% of step. Chasing
`getcoeffld` would have optimised a function that never runs.)*

### L83 — A gate that never reads the lever's output is not evidence. (M7 A.1, 2026-07-14)

`FESOM_SPEED_FLAT` flattens four kernels; two of them (`io_acc_u`/`io_acc_v`) are **time-mean
accumulators**. Their output goes only to the monthly stream (`u/v.fesom.1958.monthly.nc`) — it never
reaches `snap_*.nc`, and `snap_*.nc` is the only thing `diff_snap.py` globbed.

So the FORCE_SERIAL byte proof — the campaign's strongest correctness statement — would have **run those
two kernels and compared nothing they wrote**. It would have passed. It would have meant nothing for half
the lever. This is L80's dead-knob trap wearing the other mask: there, the gate passed because the code
*didn't run*; here, the gate passes because the *output isn't looked at*.

**Fixed:** `diff_snap.py` takes `--pattern` (default `snap_*.nc`, so every existing gate is byte-for-byte
unchanged) and the proof now runs twice — `snap_*.nc` **and** `*.monthly.nc` (17 files, all bit-identical).

**Rule:** for every lever, name the file the lever's output actually lands in, and check the gate opens
that file. "The gate passed" is only evidence if the gate *read the bytes the lever wrote*.

---

### L84 — The GPU config runs 4 ranks/node; the CPU config runs 128. Every un-ported host loop therefore costs the GPU run ~32× more. (M7 D.0, 2026-07-14)

This one explains the whole shape of the M7 campaign, and I only saw it while checking a claim I had
already written down.

I flagged `ROTCACHE` (a host-side fix) as a threat to the GPU-vs-CPU ratio: "it speeds the CPU
reference up too, so enable it on both sides before quoting a ratio." Right in principle. **Wrong by
160×**, because I assumed the two configs pay the same host cost. They do not:

| | ranks/node | ranks @ NG5 4 nodes | **nodes/rank** | rotation trig |
|---|--:|--:|--:|--:|
| GPU run | **4** (one per GPU) | 16 | **463 k** | 37 ms of a 913 ms step = **4.05%** |
| CPU run | **128** (one per core) | 512 | **14.5 k** | 1.2 ms of a 4599 ms step = **0.025%** |

The forcing loop is **per-rank serial host work**. The CPU run spreads the mesh over 128 processes per
node; the GPU run concentrates it into 4. **So the GPU config carries 32× more host work per rank —
for exactly the same code.**

**Consequences, and they are not small:**
1. **A host-side cost that is invisible in a CPU profile can dominate the GPU step.** 0.025% vs 4%. Any
   "we profiled it on CPU and that loop is nothing" reasoning is void for the GPU config.
2. **This is *why* host code keeps being the answer in M7** — `SWSKIP` (−26%), `IOACC`, `ICEFLUXDEV`,
   and now the forcing (75 ms/step). It is not coincidence and not sloppiness in one routine: it is
   structural. Expect the next bottleneck to be host code too.
3. **The fix for host code in this port is "move it to the device", never "make the host loop
   faster".** Optimising the host loop (D.0, −4%) buys the amplification factor once; porting it (D.1,
   −8%) removes it. D.0 is a down payment on D.1, not a substitute.
4. **A host-side lever barely moves the CPU denominator**, so it does not inflate the ratio — but only
   because of the rank-count asymmetry, *not* because the lever is GPU-specific. Check the arithmetic
   before either claiming or dismissing a ratio distortion.

**Measured, not just derived** (job 26244994, `jobs/job_m7_ab_cpu` — the compute-partition twin of the
GPU A/B; note a CPU leg must set `FESOM_SPEED_FORCE_SERIAL=1` as well, since knobs resolve OFF on a
non-CUDA build): NG5@4N CPU, `base` **4.5853 s/step** vs `ROTCACHE` **4.6089** = **+0.51%**. The lever
that is worth **−4% on the GPU is worth NOTHING on the CPU.** (Read honestly: within-leg rep spread is
0.24–0.37%, so 0.51% is at the edge of resolution, and no mechanism supports a *real* cost — the table
is ~123 MB/node/step ≈ 0.3 ms at 400 GB/s, not the ~23 ms a genuine +0.51% needs. Call it *no
measurable effect*, not a regression.)

**So the "levers act on the CUDA path only" rule turns out to be load-bearing for PERFORMANCE, not just
a guard for the Serial debug oracle.** A bit-identical transformation can be a large win on one backend
and worth nothing on the other, purely because of how the mesh is divided among ranks.

*(Corollary for anyone reading a "GPU is N× the CPU node" number: part of that N is the GPU config
being forced to run the un-ported serial remnant 32× more concentrated. That is a real cost of the
port's current state, not an artefact — but it also means the ratio will keep improving as host code
moves to the device, independently of any kernel getting faster.)*

---

### L85 — An attribution by COUNT and an attribution by TIME can name opposite culprits. (M7 package H, 2026-07-14)

A re-analysis of the M7 nsys trace asked: *of all the PCIe copies in a timestep, how many happen
inside an MPI call?* The answer came back clean and unambiguous: **94.5 %** (57.7 % inside
`MPI_Waitall`, 35.3 % inside `MPI_Isend`, only 5.5 % outside). The obvious reading — *"CUDA-aware MPI
is host-staging every halo message; that is the biggest lever in the campaign"* — is **FALSE**, and I
was one step from shipping it.

**The 94.5 % is a COUNT. The MPI copies are the tiny ones.** Re-weighting the *same rows* by
duration inverts the conclusion completely:

| | calls/step | **ms/step** | MB/step |
|---|--:|--:|--:|
| inside an MPI call (UCX staging) | 4 157 | **18.2** | 184 |
| NOT in MPI — the model's own DualView rails | 244 | **77.6** | 455 |

**244 copies out of 4 401 (5.5 % by count) are 81 % of the time.** They are full nod2D fields
(3.71 MB each, 114/step) that the model deep-copies host↔device itself. The real lever was never
MPI at all — and the plan's budget table had been carrying that pool under the label *"MPI staging +
forcing HtoD"* for the whole campaign, which is exactly why it never became a package.

**Rule: weight an attribution by the quantity you actually intend to spend.** If the goal is
milliseconds, a histogram of *events* is not evidence — it is a different question that happens to
return a confident number. Ask "what fraction of the TIME", never "what fraction of the calls", and
if a distribution is heavy-tailed in size (it usually is: halo messages span 400 B to 3.7 MB here),
the two answers will not merely differ in degree — **they will point at different code.**

This is L81/L82 in new clothing: a profiler (or any aggregate) will *localise* honestly and *attribute*
misleadingly. The defence is the same one that has now worked four times in M7 — **do the arithmetic
before you believe the summary.**

*(Same session, same trace: the step-profiler's own `Kokkos::fence()` around every labelled kernel
(`fesom_profile.cpp:61,:73`) inflates the phase it measures — 5 labelled kernels × 120 EVP subcycles =
~1 200 extra fences/step, reporting `ice_dyn` at 73.5 ms when nsys says the loop is **38.3 ms**. An
instrument that costs what it measures will size your next package for you, wrongly. Cross-check any
phase number that brackets many small kernels against a trace that does not fence.)*

---

### L86 — A missing `modify_host()` is invisible to every gate you have, because `d()` does not sync. (M7, 2026-07-14)

`fesom_step.cpp:773` writes `dyn->eta_n[]` through the **raw host alias** and never calls
`modify_host()`. Its only rail (`:511`) fires *earlier* in the step. So on CUDA the DEVICE `eta_n`
held the **previous step's** value for the whole back half of every step — while the DualView still
reported `Synced`. `Field::d()` returns the device view **without syncing** (`fesom_field.hpp:74`), so
`resolve_ssh_dev` (`fesom_io.cpp:868` — the `FESOM_SPEED_IOACC` accumulator, **in the blessed set**)
silently accumulated `ssh` one step stale.

**It shipped, and it passed everything:**
- the **FORCE_SERIAL byte proof cannot see it** — on Serial `.d()` and `.h()` are the SAME MEMORY, so a
  missing rail is a no-op there. (This is the D.1 trap stated as a general law: **no Serial gate can
  ever validate a coherence invariant.**)
- the **knob-OFF byte gate reads `snap_*.nc`**, written from the HOST path (`fesom_io.cpp:370`,
  `.h_checked()`) — which was always correct.
- one stale step in ~14 000 is **far below the climate gate's floor**.

**Two rules fall out:**
1. **A host write through a raw alias is a coherence event.** Grep for raw-pointer writes
   (`x->field[n] = …`) in the per-step path and check each one is followed by `modify_host()`. The
   DualView cannot detect what it never saw.
2. **`modify_host()` alone is not a fix** — it only sets a dirty bit that a *later* `sync_device()`
   would act on. If a device reader takes `.d()` directly (as every `_kk` kernel and every device I/O
   resolver does), the fix must be an actual **`modify_host(); sync_device();`** push, or the loop must
   move to the device.

**Generalised:** *the strongest gate in this project (the FORCE_SERIAL byte proof) is structurally
blind to the single most common CUDA-only bug class.* Any change that touches a rail, a `modify_*`, or
a `sync_*` is validated by the **CUDA fidelity gate or by nothing.**

### 🔴 L86 — THE DEMONSTRATION. Same binary, same hour, four gates. (M7 H.2, 2026-07-14)

The first cut of `FESOM_SPEED_ICERAILS` (package H.2) deleted 55 full-field rails from the ice step.
It was submitted to every gate at once:

| gate | verdict |
|---|---|
| knob-OFF byte gate (26255200) | **PASS** |
| **FORCE_SERIAL byte proof, ICERAILS ON (26255201)** | **`diff_snap rc=0` — BIT-IDENTICAL to the certified baseline** |
| ICERAILS announce | **1** — the knob fired (not L80) |
| **CUDA fidelity gate, ICERAILS alone (26255202)** | **CATASTROPHIC FAIL: 15 fields over ceiling COHERENTLY. `a_ice max|Δ| = 9.83e-01` — i.e. the ENTIRE ice concentration. `m_ice` 2.15, `h_ice` 2.22. THERE WAS NO SEA ICE.** |

**The project's strongest gate certified, as bit-for-bit identical to ground truth, a lever that
deletes the sea ice.** Because on Serial `.d()` and `.h()` are the same memory, so all 55 deleted
rails are no-ops there — and the bug was *precisely* a rail.

**The bug (and it is a general trap):** `Field::alloc()` zero-inits BOTH spaces and marks them
`Synced` (`fesom_field.hpp:64`). The ice INITIAL CONDITION is then written **on the host** through the
raw alias — the DualView never learns. **The only thing that had ever carried that IC to the device
was the per-step IN rail.** Delete the rail as "a pure round trip between two device kernels" — which
it demonstrably was, on every step but the first — and the device ice state stays at **zero**.

**The rule: before deleting a rail because "both sides are device kernels", ask WHO PUT THE INITIAL
VALUE THERE.** A per-step rail is also, silently, the IC push. The Z7 trap (a field that is bitwise
equal at cold start) wearing the opposite face: here the field is *not* equal at cold start, and only
step 1 is wrong — but in a coupled model step 1 being wrong is the whole run.

Fixed with an explicit one-shot IC push under the knob. **Cost of the CUDA gate: 42 seconds. Cost of
trusting the byte proof: a silently ice-free ocean.**

---

### L87 — A host timer tells you how long the HOST SAT THERE. It does NOT tell you what is REMOVABLE. (M7 D.1, 2026-07-14)

L82 concluded — correctly, against a lying sampler — that *"a **host timer** (`FPROF`) around the call is
the only host measurement that cannot misattribute."* **That is true of WHICH call, and false of HOW
MUCH.** D.1 was pre-registered at **−6.6 %** off the FPROF number and delivered **−2.16 %**. The knob
fired (announce checked — not L80). The lever was fine; **the sizing was wrong at the source.**

| | ms/step |
|---|--:|
| FPROF host timer, `force:jra55_read` | **75.2** |
| nsys **GPU-idle gap** for the same region (the honest exposed cost) | **60.0** (4.9 of it PCIe) |
| ROTCACHE (D.0) had already removed | −19.7 |
| ⇒ real removable pool at the D.1 baseline | **~35–40**, not the 57.3 that was pre-registered |
| D.1 delivered | **18.7** |

Two separate errors, and they compound:
1. **The timer over-read the region by ~25 %** (75.2 vs 60.0). A host timer brackets everything inside
   it — including any fence, `deep_copy`, or `sync_*` that is really *waiting for the GPU to drain work
   it owes anyway*. That wait does not disappear when you port the loop.
2. **Roughly half the pool survived the port.** A host timer says nothing about what the *device*
   version will cost, nor what stays behind (the residual halo, the new D2H at the surviving host
   reader, launch/fence overhead).

**The honest instrument for "how much host time is EXPOSED" is the GPU-IDLE GAP in a trace** — the
wall time between the end of one kernel and the start of the next, where the GPU is doing nothing. That
is time a port can actually recover. Compute it straight from the nsys sqlite; it needs no new run:

```sql
-- per step: gap = start(kernel i) - end(kernel i-1), grouped by the kernel the gap ENDS at
```

**Do this before pre-registering any host-to-device port.** It would have sized D.1 at ~35–40 ms, not
57.3, and it costs nothing.

*(The same query paid for itself twice over: it showed **222.6 ms/step — 24 % of the step — is GPU-idle
gaps >1 ms**, and ranked them. The three gaps before the sea-ice kernels (`ice_thermodynamics` 22.7,
`ice_evp_dynamics` 19.5, `ocean2ice` 16.5) sum to **58.7 ms of pure GPU idle spent waiting on the ice
step's PCIe rails** — a third independent confirmation of package H, from a different angle than either
the memcpy accounting or the source audit. **When a lever misses, profile the miss: the diagnostic that
explains it usually finds the next lever.**)*

---

### L88 — Do NOT decompose a measurement by SUBTRACTING A MODEL and then name the remainder. (M7, 2026-07-15)

A 25-step run was **72.6 ms/step slower** than a 300-step run. We knew `getcoeffld` was the culprit, so
we *modelled* it (49.3 ms/step for the first K steps, then 7.7) and subtracted: 41.9 ms. The CG spin-up
explained 8.7 more. **The 22 ms left over was then given a physical story** — *"it sits in the halos /
host / MPI remainder that the PHASE profiler structurally cannot see. Only a trace reaches it."* It went
into the handoff as an open question with a job attached to it.

**A trace reached it. There was nothing there.** Two matched nsys traces (25-step and 300-step, same
binary, same flags) have **identical GPU-idle gap budgets** — 93.6 vs 94.1 ms/step — and steady-state
steps within 8 ms. The per-step wall series is a smooth 745 → 729 ms decay over ~200 steps that tracks
the CG iteration ramp (86 → 72) exactly. **The whole residual is the CG spin-up. The 22 ms was MODEL
ERROR in the subtraction, wearing a physical story.**

**The rule:** `measured_total − modelled_part` is **not** a measurement of anything. Its error bar is the
model's error bar, and you do not know that. If you want to know what is left, **measure what is left** —
here, one `--diff` of two censuses that were already on disk, and one per-step time series, both free.

*(Corollary — the honest version of this decomposition is a **CONTROLLED DIFFERENTIAL**: change ONE
thing, re-measure, and let the *difference of two measurements* name the part. That is what the eta_n fix
and the mEVP fix below both did, and neither needed a model.)*

---

### L89 — Name a stall by the PAIR it sits between, not by the kernel it delays. (M7 H.3, 2026-07-15)

The gap census (L87) attributes GPU-idle time to *the kernel the gap ENDS at*. That found package H. But
"`ocean2ice` waits 16.8 ms" does not tell you **what to delete** — `ocean2ice` is innocent; it is the
victim. Keying every gap by **`(predecessor → victim)`** and itemising the PCIe inside it **by count, MB
and direction** turns the same free query into an *address*:

```
fesom_halo_exchange_device2 -> fesom_ocean2ice_kk   16.8 ms   6.7 PCIe   8 copies   28.3 MB  DtoH
```

7.07 MB then six of 3.54 MB, **all DtoH, in source order** — which is `2N` doubles followed by six `N`
doubles, which is *exactly* the seven `sync_host()` calls at `fesom_bulk.cpp:629-635`, in the order they
are written. The census did not just size the lever; **it pointed at the lines.**

**And it caught the half a byte-count would have missed.** Between the last copy and the next kernel sat
**7.8 ms of untraced host time** — a dead 930k-element interpolation loop in the same gap. The plan had
sized this lever from the *rails* alone (30 MB ⇒ ~1 %). The gap census, which counts **time the GPU is
idle** rather than **bytes moved**, found both halves and doubled the estimate to 2.2 %.

⇒ **Size a rail lever by the GAP, not by the BYTES.** Bytes miss the host code standing next to them.

---

### L90 — A source audit is an ARGUMENT. A trace is a MEASUREMENT. When they disagree, the trace wins — and the bug is usually one `#ifdef`. (M7 H.3, 2026-07-15)

A careful, thorough source audit concluded that 3 of bulk's 7 `sync_host()` calls were **already no-ops at
`npes>1`**, because their halo exchange (`fesom_halo_field`) ends with `modify_host(); sync_device()` and
therefore leaves the field `Synced`. It quoted the lines. It was **wrong**.

```c
inline void fesom_halo_field(fesom::Field &f, ...) {
    f.modify_device();
    if (!p || p->npes <= 1) return;
#ifdef KOKKOS_ENABLE_CUDA
    if (fesom_halo_device_active()) { fesom_halo_exchange_device(...); return; }   // <-- CUDA STOPS HERE
#endif
    f.sync_host();  fesom_halo_exchange(...);  f.modify_host();  f.sync_device();  // <-- the FALLBACK
}
```

On CUDA it **returns early** and leaves the field `Auth::Device`. The code the audit read is the
**host-staged fallback for Serial/OpenMP**. The trace showed all seven copies firing, in source order,
with the right sizes. **The measurement was right; the argument lost on a single early return.**

This is L85's twin. L85 was *"a COUNT and a TIME name opposite culprits."* L90 is *"SOURCE and a TRACE
name opposite culprits"* — and the failure mode is the same: a reading of the code that is locally
correct and globally wrong, because the path you are on is selected somewhere you did not look.
**Before believing that a rail is dead, look for it in a trace. If it is there, it is not dead.**

---

### L91 — THE OPTIONS MATRIX IS NOT A ONE-TIME GATE. Re-run it for EVERY new coherence lever. (M7, 2026-07-15)

`FESOM_SPEED_ICERAILS` shipped (`a96e299`, gated, CUDA-fidelity-green) with a **live correctness bug**:
under `FESOM_WHICH_EVP=1` (mEVP) it **clobbered `srfoce_u/v/ssh` and `stress_atmice` with a stale host
mirror every step**, and with `alloc()`'s **zeros at step 1**. Measured against mEVP's own Serial oracle:
**13 fields coherently over ceiling — `h_ice` off by 1.94 m, `T` by 1.93 °C, `a_ice` by 0.58.**

The knob correctly gated the **standard-EVP** branch's IN/OUT rails on `!icerails`. **Forty lines below,
the mEVP branch does the same pushes and was not gated.** mEVP reads *exactly the same state* — its own
comment says so — but it is a different `else if`, and no default gate ever enters it.

**Here is the part that matters, and it is not the part you expect.**

**The project ALREADY HAD an options-matrix gate.** M7 Task 5.1 ran `FESOM_SPEED=1` × {TKE, mEVP, zstar},
each against *that knob's own M6 Serial oracle*, and **all three passed** (jobs 26238785-87). The
infrastructure was right, the reference was right, someone had already thought of this.

**It was run once, in session 3. `ICERAILS` landed in session 5. Nobody re-ran it.** The matrix was
green for the knob set that existed *when it was run*, and it silently stopped covering the knob set that
existed later. **A gate that is not in the ladder is a gate that has already expired.**

So the lesson is NOT "add an options row" — that row exists. It is:

> **Any lever that changes WHO OWNS A FIELD invalidates every previously-passed options gate.**
> The options matrix belongs in the per-lever ladder, next to the CUDA fidelity gate — not in a
> one-off task that gets ticked off and never re-run.

**And the two structural reasons it stayed invisible are the usual two:**
- The **default is `whichEVP=0`**, so no default-config gate executes that branch — ever.
- mEVP's own M6 certification was a **Serial bit-identity test**, and on Serial `.d()` **is** `.h()`, so
  every one of these rails is a no-op and every clobber is a self-assignment. **L86, exactly: NO SERIAL
  GATE CAN VALIDATE A COHERENCE INVARIANT.** It could not have caught this, and it did not.

**Mechanically:** when a lever flips a field from host- to device-authoritative, `grep` **every**
`modify_host()` on that field in the whole file — not the ones on the default path, *all* of them — and
ask of each: *does this still have a producer?* Option knobs (`FESOM_WHICH_EVP`, `FESOM_MIX_SCHEME`,
`FESOM_ALE`) select code your gates do not run.

*(Proven by a controlled differential, not asserted: the **pre-fix** binary with `mEVP + ICERAILS`
against the mEVP Serial oracle — **predicted FAIL, and it FAILED** — versus the **post-fix** binary —
**predicted PASS, and it PASSED**. One code change; the bug appears and disappears. See L88: this is
what "measure what is left" looks like when the thing you are measuring is a bug.)*

---

### L92 — An instrument's RESIDUAL bucket is where its blind spots hide. Name every column, and make them SUM. (M7 H.7, 2026-07-15)

The GPU-idle gap census (L87/L89) attributes each stall to PCIe / MPI / **host**. `host` was defined as
the residual: `gap − (memcpy ∪ MPI)`. That is where it lied.

It reported **`kpp_mixing`: 8.4 ms, 0.0 PCIe, 0.0 MPI, 8.4 host** and **`smooth_nod3D`: 13.4 ms, 1.1
PCIe, 12.4 host**. I wrote *"the KPP host chain — 21.8 ms of gap, 16.3 ms of it PURE HOST COMPUTE with
ZERO PCIe. Not a rail at all. Now the largest host pool"* into the handoff, and queued it as the next
lever: **port the KPP host loops to the device.**

**There are no KPP host loops.** Adding a **FENCE** column and itemising the CUDA runtime API inside
those gaps by name took four minutes and said:

| gap → | ms/step | what is ACTUALLY in it |
|---|--:|---|
| `smooth_nod3D` | 13.2 | **`cudaMallocAsync` 11.93 ms/step, 4.0 calls/step** |
| `kpp_mixing` | 9.0 | **`cudaDeviceSynchronize` 8.91 ms/step, 4.0 calls/step** |
| `compute_sigma_xy` | 3.4 | **`cudaDeviceSynchronize` 3.36 ms/step** |

**It was never host code. It was the ALLOCATOR.** `fesom_smooth_nod3D_kk` (`fesom_eos.cpp:555`)
constructed two `Kokkos::View`s of `nslab × myDim_nod2D × nl` doubles — hundreds of MB — **inside the
hot loop, every call, every step.** Two Views × two call sites = **exactly the 4.0 `cudaMallocAsync`/step
measured.** And the fences are the *same* bug: **Kokkos MUST fence when a View is destroyed**, to
guarantee no in-flight kernel still references the memory it is about to free. *Allocate → use →
destructor → FENCE → free.*

**A fence has no memcpy and no MPI. So it fell into the residual and wore host compute's clothes.**
A device port would have burned a session and recovered nothing.

**Two rules, and the second is the general one:**

1. **A stall with zero PCIe and zero MPI is not automatically host compute.** Four causes look
   identical from the outside and want four *different* levers:
   | cause | the lever |
   |---|---|
   | a memcpy | **delete the rail** |
   | an MPI call | **overlap the comm** |
   | `cudaDeviceSynchronize` | **remove the fence** |
   | `cudaMallocAsync` / `cudaFree` | **HOIST THE ALLOCATION** |
   | nothing traced at all | **port the host code** |
   The census now carries an explicit **FENCE** column so this class cannot masquerade again.

2. 🔴 **The general rule: NEVER let a bucket called "everything else" carry a lever.** A residual is
   whatever your instrument could not name — its size is the sum of the real thing *and every blind
   spot you have*. **Make the columns SUM to the total.** When `smooth_nod3D` showed `gap 13.2` against
   `PCIe 1.1 + MPI 0.0 + FENCE 0.0 + host 0.1 = 1.2`, the missing **12 ms was the bug announcing
   itself** — and it only announced itself because the columns were forced to add up.
   *(This is L88's twin. L88: do not name the remainder of `measured − modelled`. L92: do not name the
   remainder of `total − everything_I_thought_to_measure`. Both are "the residual is not evidence".)*

**And a standing Kokkos rule, worth its own line:** a `Kokkos::View` constructed in a per-step function
costs you **TWICE** — the allocation *and* the mandatory deallocation fence. **Hoist scratch buffers to
static/persistent storage.** It is usually bit-identical (check: are the buffers written by ASSIGNMENT
from register accumulators, not `+=`? do the producer and consumer kernels share a mask, so unwritten
entries are never read? then nothing depends on the zero-init) and it is the cheapest lever there is.
*(H.7 was **−4.21 %** — the second-largest lever in the campaign — and bit-identical: `diff_snap rc=0`.)*

---

### L93 — CALIBRATION: a HOST TIMER over-predicts. The GAP CENSUS under-predicts. Size from the census and treat it as a FLOOR. (M7, 2026-07-15)

Five levers have now been pre-registered off a *measured* pool and then scored against the A/B. The
pattern is not noise — it is a property of the two instruments:

| lever | sized from | pre-registered | measured | |
|---|---|--:|--:|---|
| D.1 `FORCEDEV` | **a host timer** (FPROF) | −6.6 % | **−2.16 %** | ❌ **missed by 3× (OVER)** |
| H.1 `FLUXDEV` | measured per-copy PCIe | −1.1 % | −1.45 % | ✅ beat |
| H.2 `ICERAILS` | the gap census | −4.1 % | −6.03 % | ✅ beat |
| H.3 `BULKTAIL` | the gap census | −2.2 % | −2.43 % | ✅ beat |
| H.7 `SMOOTHSCRATCH` | the gap census | −2.8 % *(ceiling −3.3)* | **−4.21 %** | ✅ **beat, PAST THE CEILING** |

**A HOST TIMER OVER-PREDICTS** because it brackets everything inside it — including time the host spends
waiting for the GPU to drain work it owes *anyway*. That wait does not disappear when you port the loop
(L87).

**THE GAP CENSUS UNDER-PREDICTS**, for three structural reasons — know them, because they are the size of
the error:
1. **It thresholds.** `--min-gap-ms 1.0` throws away every sub-millisecond stall, and there are *many*
   (every small fence, every small free).
2. **You will under-attribute at the margins.** For H.7 I *saw* a 3.36 ms fence before `compute_sigma_xy`,
   judged it "probably the same class — check it", and **excluded it from the pre-registration.** It was
   the same class. That alone was a third of my error.
3. 🔴 **ENTANGLEMENT — and the stall budget says so in its own footer.** Removing a stall does not only
   return *its own* time; it lets the **host run ahead**, so the launch queue stays full and **downstream
   launch-gap time is recovered too.** *"Spin alone is the FLOOR of the payoff, spin+gap the CEILING."*

⇒ **Size from the gap census, quote it as a FLOOR, and put the ceiling well above it.** A pre-registration
is a commitment, not a prediction you get to be proud of — and **being wrong LOW four times running is
still being wrong.** Fixing the range is cheaper than fixing the credibility.

---

### L94 — THE LEVANTE `gpu` PARTITION IS HETEROGENEOUS. Pin `-C a100_80`, or your anchor is not a measurement. (M7, 2026-07-15)

| node range | feature | GPU | HBM bandwidth |
|---|---|---|---|
| `l40xxx` (cell09) | `a100_40` | A100 **40 GB** | HBM2, **1555 GB/s** |
| `l50xxx` (cell13) | `a100_80` | A100 **80 GB** | HBM2e, **1935 GB/s** (~**25 %** more) |

FESOM's step is **74.6 % memory-bound kernels**, and in an MPI job **the slowest rank sets the pace**
(every step ends in a halo exchange). So **ONE `a100_40` node in a 4-node allocation drags the WHOLE
JOB.** Measured, *same binary, same config, same day, 300 steps*:

| allocation | s/step |
|---|--:|
| pure `a100_80` | **0.7058** |
| 3× `a100_80` + **1× `a100_40`** | **0.7298** |
| | **+3.4 %, from the hardware alone** |

**That is larger than most levers in this campaign.** An unconstrained GPU anchor is not a measurement.

**How it nearly cost a landed lever.** H.7's 300-step anchor came back at 0.7298 against the previous
anchor's 0.7058 — i.e. **"H.7 makes the model 3.4 % SLOWER"** — while its own same-allocation A/B said
**−4.21 %**. I did not publish the regression, because:

> 🔴 **WHEN AN ABSOLUTE ANCHOR CONTRADICTS A CONTROLLED A/B, SUSPECT THE ANCHOR.**
> An **A/B is IMMUNE** to node heterogeneity — both legs run on the **same nodes**, so the hardware
> cancels. An **ABSOLUTE ANCHOR HAS NO SUCH PROTECTION.** The A/B is the robust instrument; the anchor
> is the fragile one. That asymmetry is *why* this campaign compares levers by same-allocation A/B.

The chain that caught it: the **L80 announce check passed** (`FESOM_SPEED_SMOOTHSCRATCH = ON`, binary md5
correct) — so the lever *had* fired, which meant the contradiction could not be a dead knob, which forced
me to look at **`nodes:`** in the job output. `l40369`. `sinfo -N -n l40369 -o "%f"` → `a100_40`.
**Two minutes.**

**Fixed at source:** `jobs/job_m7_ab_env` now carries `#SBATCH --constraint=a100_80`. **Audit your
existing numbers before trusting them** — `grep "^nodes:" abenv.*.out` and check every one. *(Ours: the
`6.49×` ratio and both levers' A/Bs were all on pure `a100_80`. Exactly one number — the h9 anchor — was
contaminated, and it was never published.)*

**The general rule, beyond Levante:** on any heterogeneous partition, a **cross-allocation absolute
comparison is a hardware lottery**. Constrain the hardware, or measure only *differences within one
allocation*. `feedback-perf-same-day-baseline` said "~5 % cluster noise, re-measure same-day" — **it was
right, and this is the mechanism behind it.**

---

### L95 — A PROTOCOL IS VALIDATED PER MESH, NOT PER CAMPAIGN. A longer window can walk into a known instability the short window never reached. (M7, 2026-07-15)

Session 6 replaced the contaminated 35-step protocol with the clean 300-step one, and the whole NG5
standard set re-ran on it without incident. Session 7 then queued **dars@8N** on the same protocol
— and **both CPU reps died at exactly step 204** (`CG_kk abort: pp·App = nan`; deterministic,
because Serial is bit-reproducible, so identical reps die at the identical step).

Nothing was newly wrong. **SCALING_M524 had documented the dars cold-start instability class all
along** ("dt=240 is CFL-unstable from the *cold* PHC start on BOTH dars and NG5"), and the M5.24
partition probe showed blowups arrive **earlier the finer the decomposition** — but that probe was
**NG5-only**, and every dars number in two campaigns was a **35-step window** that ended 169 steps
before the cliff. The protocol upgrade was validated exactly once, on the mesh it was designed on.

- **The rule:** when a measurement protocol changes (length, dt, IC, cadence), re-ask *"does this
  configuration even complete?"* **per mesh × per partition** — an instability boundary is a
  property of the configuration, not of the protocol.
- **The tell in the wreckage:** both reps dying at the SAME step = deterministic physics, not a
  flaky node or an OOM. (A hardware failure would not reproduce to the step; an OOM would show in
  the job accounting.)
- **The honest fallback:** measure the longest window that completes with margin (dars@8N: 150 of
  the 204 available steps — past the CG ramp that settles by ~step 30–50) and **annotate the ledger
  row with its protocol** instead of silently mixing windows (the top of GPU_SPEED_M7.md exists
  because mixed protocols once faked a ratio).

---

### L96 — TWO WAYS A KERNEL-RESOURCE TABLE LIES: the spill column isn't called "spill", and a demangled name's first familiar token isn't the kernel. Both returned CONFIDENT wrong answers before the right one. (M7 session 8, 2026-07-15)

Re-deriving package C's spill pool (cuobjdump × kernel-busy) hit two independent tool traps in one
hour; each produced a *complete, plausible, wrong* table, and only disagreement with a
pre-registered expectation (14 spilling kernels) forced the re-look — L92's rule ("a residual is
not evidence; make the columns sum") generalized to *any* derived table.

1. **In `cuobjdump --dump-resource-usage`, spills live in `STACK`, not `LOCAL`.** `LOCAL` is the
   static `.local` section — **0 for every kernel in this binary** — while the ptxas stack frame
   (spill slots + ABI stack) is `STACK`. Keying on LOCAL yields an empty pool that reads as
   "nothing spills". (nsys's `localMemoryPerThread` is a third trap — a non-collection artifact,
   always 0, from the same family as L82/L83.) The tell: the pre-registration said 14 spillers;
   grep said `STACK:[1-9]` on exactly 14 functions.
2. **Never name a kernel by "the first `fesom_*` token" in its demangled symbol.** For static
   (internal-linkage) kernels the first familiar token is a PARAMETER TYPE (`fesom_mesh`,
   `fesom_kpp`…) or the internal-linkage module hash (`fesom_tke_cpp_a1eea344`). The TDMA kernel
   `diff_ver_part_impl_ale_kk` — **32.6 ms/step, the pool's #2 entry** — hid inside a fake
   "fesom_mesh" row for exactly this reason, invisible to every previous per-kernel table. Parse
   the enclosing function after `ParallelFor</ParallelReduce<` instead
   (`scripts/m7_kernel_busy.py`, `scripts/m7_spill_pool.py`). The tell: a "kernel" named after a
   struct, with a launch count that matches no call site.

**Cross-check that certifies the fix:** the corrected busy table sums to the census's kernel-busy
total (543.2 ms/step) and the corrected pool reproduces the session-7 audit's per-kernel numbers
(42.3 ms / 5,120 B / 58 reg for `redi_expl`) to the digit.

### L97 — A SPILL POOL IS AN UPPER BOUND WHOSE REALIZATION FACTOR CAN BE NEGATIVE. The spills can be the CHEAPEST bytes a kernel moves — and the structure that spills can be what keeps the caches hot. (M7 session 9, 2026-07-15)

Package C's premise was "188.8 ms/step of busy time sits in 10 spilling kernels ⇒ removing the
spills recovers a slice of it". C.1 rebuilt the pool's #1 kernel (`fesom_gm_redi_ver_node`, five
NL_MAX column arrays, 5,120 B/thread STACK, 36.8 ms/step) as a single bottom-up sweep with O(1)
carried scalars: **REG 54 / STACK 0, FORCE_SERIAL byte-proof rc=0 — a perfect lever on every
static metric. The A/B measured +1.88 %.** The ncu pair (26268785/87) explains it:

| | dur/launch | DRAM | L2 | local | occupancy |
|---|--:|--:|--:|--:|--:|
| spilling original | 15.76 ms | 17.8 GB | 42.9 GB | 5.1 GB | 46.5 % |
| STACK-0 sweep | 22.17 ms | **+45 %** | **+27 %** | 0 | 53.1 % |

Removing 5.1 GB of local traffic ADDED ~8 GB of DRAM traffic, and time tracks DRAM. Two
mechanisms, both structural:
1. **Local memory is thread-interleaved ⇒ spill ld/st are perfectly coalesced.** A "GB of spill
   traffic" and a "GB of irregular gather traffic" are not the same currency — the spill GB was
   ~free.
2. **The three-pass original was cache-shaped; the fused sweep is not.** The old gather pass ran
   top-anchored (all threads of a warp at the SAME level ⇒ adjacent nodes share element-lines)
   with a tiny working set per phase (lines reused within a few iterations). The fused sweep
   anchors each thread at its own bottom depth (no cross-thread line sharing) and stretches every
   line's required lifetime across the whole gather+flux+apply body (evicted before reuse). The
   local arrays were BUYING the phase separation.

Rules this adds:
- **"STACK bytes × busy ms" RANKS candidates; it never PRICES one.** Price = a per-kernel ncu of
  the restructured kernel, or a strict-reduction probe (delete stores, change nothing else) —
  something that cannot be slower.
- **A lever that changes loop STRUCTURE (fusion, anchor direction, phase count) must be priced as
  a cache-locality change, not a byte-count change.** Byte deltas that ignore reuse-distance are
  models, and today the model had the wrong SIGN.
- The byte proof and the perf claim are INDEPENDENT: 8/8 correctness gates green and a wrong-sign
  A/B coexisted happily. Gates certify values, never speed.
- Ops note: an opt-in `_exp` knob is where a byte-correct-but-slower lever goes to be studied —
  never let it ride the master. And pre-register the A/B with a FLOOR OF ZERO when the mechanism
  is unpriced: a null result on a strict-reduction probe is a DECISION, not a failure.

---

### L98 — A THRESHOLD IS PART OF THE MEASUREMENT. A pool that is "18 ms" at one gap threshold was 97 ms at another — and a campaign decision was made on the small number. (M7 session 10, 2026-07-15)

The gap census counted GPU-idle gaps **> 1 ms** and reported "package E (halo MPI-wait) =
18 ms/step at 4N"; the session-9 handoff then did honest-looking arithmetic on it ("E 18 + C tail
+ H.10 ≈ −4 % ⇒ the 8× stretch is NOT reachable") and the user set the campaign endpoint on that
basis. The 16N census broke the story: the step was only 49.7 % kernel-busy yet the >1 ms census
could name 14 ms of the 139 ms idle. Histogramming the residual (L92: a residual is not
evidence) found **81 % of all idle in ~350 gaps/step of 100–1000 µs** — individually under the
bar. Re-running BOTH censuses at `--min-gap-ms 0.1`:

| halo-wait pool | 4N | 16N |
|---|--:|--:|
| @ >1 ms threshold | 18.0 ms (2.7 %) | 13.2 ms (4.8 %) |
| **@ >0.1 ms threshold** | **97.3 ms (14.8 %)** | **124.8 ms (45.0 %)** |

The event count is IDENTICAL at both scales (358/step — per-field exchange calls, structural),
per-event wait ~278→386 µs. Kernels scale ÷3.94; event latency doesn't ⇒ this IS the whole
4N→16N ratio decay.

- **The rule: a derived pool inherits every parameter of the tool that derived it.** Quote the
  threshold with the number ("18 ms at >1 ms") or the number will outlive the caveat — this one
  did, for two sessions, into a user decision.
- **The tell: a pool that SHRINKS while its share of idle EXPLODES.** 18→13 ms against
  83→50 % busy cannot both be true of the same mechanism; the contradiction is the alarm.
- **The consequence-handling rule: when a re-measurement invalidates the basis of a USER
  decision, SURFACE it — do not silently re-litigate the decision, and do not silently keep
  honoring the stale basis either.** (Here: "~7× is the 4N endpoint" was decided on the 18 ms
  figure; the honest 97 ms pool puts 8× back in range at 50 % conversion.)
- Ops note: `--min-gap-ms 0.1` is the census setting of record for all package-E work; the
  >1 ms tables remain valid for the classes they named (SSH, ice-thermo, jra55).

### L99 — AN INSTABILITY BOUNDARY IS PARTITION-MARGINAL, NOT JUST MESH-MARGINAL; and two small audit traps that nearly mislabeled it. (M7 session 10, 2026-07-15)

dars@2N (dist_8) at dt180 from cold PHC died with `CG_kk: pp·App = -nan` at step ~10 — while
dist_16 and dist_32 complete 150 steps at the same dt, same binary, same IC. Both binaries
(row0 AND h14), all reps, same step ⇒ deterministic configuration failure. dt180-from-cold is
MARGINAL on dars and **partition-dependent reduction/rounding order decides which decomposition
trips** — and it was the COARSEST that died, INVERTING the L95/NG5 "finer dies earlier"
observation. L95 refined: re-ask "does this configuration complete" per mesh × per partition ×
per dt, and expect no monotone rule in the partition axis. (Rescue that preserved the survey
point: dt120, annotated as its own protocol — a same-day pair's Δ% is per-step work and stays
comparable.)

Two audit traps from the same session, both caught before they cost anything:
1. **The "Killed" tasks were MPI_Abort collateral, not OOM.** First hypothesis (default memory
   too small) had a fix ready to submit; READING THE LOG found the FATAL. A signal that
   pattern-matches a known failure class can have a different cause — check before acting.
2. **`job_m7_gate_serial` SPLITS streams: the `[fesom_speed]` announces land in `run.err`,
   not `run.log`** (job_m7_ab_env merges them). An L80 announce audit that greps the wrong
   stream reports a dead knob on a live lever. Grep the right stream before crying L80.

### L100 — A PAIR-KEYED CENSUS COLLAPSES SAME-TAG CHAINS: the biggest site in the halo pool (the CG solver, 28-31 %) was INVISIBLE because both ends of its gaps carried the same kernel tag. (M7 session 11, 2026-07-15)

The gap census names a gap by (predecessor → victim) kernel pair, and the halo pack/unpack
lambdas all demangle to their ENCLOSING exchange function — so every same-class MPI wait,
whatever the call site, lands in ONE `halo → halo` row. Filtering for `ssh_solve_cg` showed
~0.2 ms, and session 10 concluded (flagged "verify") that the CG's ~146 exchanges/step were not
in the pool. The E.0 timeline walker (`scripts/m7_halo_sites.py`: a maximal run of halo-class
kernels = one block; the BRACKETING compute kernels name the site) found the truth: **CG 27/38
ms (4N/16N) — the pool's single largest site — and the ledger reconciles exactly (349
exchanges/step = the 349 >0.1 ms events, one wait each, at both scales).**

- **The rule: when all members of a class share one tag, a pair-keyed table can only tell you
  the class exists, not where it is called from.** Attribution needs the timeline (bracketing
  neighbors), not the pair histogram.
- **The tell:** a phase that "contains no pool events" while the pool's event count (~350/step)
  is far larger than any static call census of the remaining phases can explain (~66 singles).
  Reconcile counts BEFORE believing an absence.
- Corollary of the same walk: the ~70 KB "staging copies" are UCX pipeline CHUNKS, not
  per-message payloads (2D halo msgs are ~10-20 KB, 3D exchanges MB-scale) — the device-pointer
  MPI path bounces every byte D2H+H2D through pinned host (181 MB/step at 4N, symmetric).

---

### L101 — MATCHED PHYSICS BEFORE ROBUSTNESS VERDICTS: three sessions of "the port rides CFLz less robustly than Fortran" dissolved the day the two models ran the SAME viscosity scheme. (M7 session 16, 2026-07-19)

The port hardwires `opt_visc=7` (bidiff — the work_core reference scheme); the Fortran scale
job pinned its template's `opt_visc=5` with a comment claiming that "exactly matches the
Kokkos/C port" — wrong, and never re-checked. Every Fortran stability/timing reference
(F0/F1, all m524 refs) therefore ran scheme 5 against the port's scheme 7. On the NG5
dist_4096 dt180 cold start (the M5.24 instability class):

| wsplit-ON | Fortran | port |
|---|---|---|
| opt_visc=5 | F1: rides the CFLz arc (peak 6.1@248) to rc=0 | dbg6: **identical arc** (peak 6.02@255, same cell, same max-migration, final 5.03 vs 5.06), rc=0 |
| opt_visc=7 | F2: **eta-NaN @203**, rank 2813, CFLz plateau ~4.16 | dbg5: **detonates @200**, rank 2813, plateau 4.09 |

Same event, same cell (−5.46/35.94), same endgame, both directions. The port was never less
robust — it was running different physics than its comparator.

- **The rule: before ANY cross-model robustness/stability verdict, diff the FULL physics
  namelist of the actual comparator runs against the port's compiled constants.** A scheme
  choice that is climatically invisible on the certified mesh (CORE2: v5-vs-v7 final-state
  stats differ in the 2nd digit only) can be the entire story at an extreme transient.
- **The trap that hid it:** `visc_filt_bcksct` was fully ported, audited, allocated — and DEAD
  CODE (never called). A grep for the Fortran routine name "finds" the port and passes a
  structural audit; only a call-site check reveals the scheme is unreachable.
- **Cost of the confusion was near-zero once measured** (the reassuring counter): scheme
  5-vs-7 timing delta = 0.0 % on CORE2 c1, +0.8 % on NG5 c32 (matched knob-only pairs, frozen
  bin `visc0`) — the cross-model SYPD comparisons carry <1 % scheme bias.
- Knobs shipped: `FESOM_VISC_OPT=5|7` (default 7 = certified byte path; 5 host-only),
  `FESOM_UV_GUARD` (the port-only uv>5 abort masked the endgame — Fortran has no such guard),
  `FESOM_VISC_EASYBSRETURN`. User policy: scheme 5 = low-res (CORE2) physics only; high-res
  meshes run 7 — matching their production practice.
- Corollary (rule 0.41 extension): the NG5/dars cold-start CFLz wall at production dt is a
  BENCHMARK-PROTOCOL artifact present in BOTH models under scheme 7. Production runs start
  from spun-up states and never see it. Protocol dt-ladders (dars dt120, NG5 32N dt60) stand.

---

## L102 — dolpung (GH200) first light: two x86 env leaks + the gdr_copy pin failure (2026-07-22)

The port compiled and ran on DKRZ's `dolpung` partition (42 nodes × 4 GH200: 72-core Grace
ARM + H100-class GPU, CUDA 12.9, RHEL 9.5 aarch64) the same day it was attempted. CORE2
dist_4, dt1800, 35 steps: knob-off 1.857 s/step, `FESOM_SPEED=1` 0.779 s/step (min-of-2,
rc=0, T/S sane, it= diag zeros match A100 logs) — the speed pack is worth **2.4×** on GH200
vs ~10 % on A100-class runs. Serial oracle + np2 pi smokes pass on Grace. Recipes:
`env_dolpung.sh` (toolchain) + `jobs/job_dolpung_gpu` (run template).

- **Compile ON the node** (`salloc -t 360 -A mh1571 -p dolpung -n 288`): logins are x86,
  nodes aarch64. Toolchain has NO usable modules for MPI/netCDF — the matched set is spack
  dirs by absolute path (gcc 14.2 `am53qrz`, nvhpc 25.7 CUDA 12.9 `dsupkck`, CUDA-aware
  `openmpi/4.1.8_cuda-12.9_nvhpc-25.7_gcc-14` under `/sw/mpi/...`, `netcdf-c-4.9.3-6xe2mx2`).
  CMake: `-DKokkos_ARCH_HOPPER90=ON -DKokkos_ARCH_NATIVE=ON` + nvcc_wrapper, exactly as the
  JUPITER plan predicted. Build ~7 min on 64 Grace cores.
- **⚠️ srun exports the x86 login environment onto the ARM node.** Two configure failures
  before the first clean one: (1) the login modules' `CMAKE_PREFIX_PATH` steered cmake to the
  **x86 netCDF** even with the ARM `nc-config` first in PATH (find_program consults
  CMAKE_PREFIX_PATH before PATH) — pin `-DNC_CONFIG=` and sanitize; (2) the x86 spack `git`
  ("cannot execute binary file") broke Kokkos' build_env_info. `env_dolpung.sh` strips every
  `/sw/spack-levante/` (x86) PATH/LD_LIBRARY_PATH entry — reusable pattern for any
  cross-arch partition.
- **⚠️ The fabric has no working GPUDirect.** First device-halo (step 1) dies:
  `gdr_copy_md.c gdr_pin_buffer failed ret:22` → `Fatal: failed to register buffer with mem
  type domain cuda`; multi-node adds `ibv_reg_mr(... cuda ...) Bad address`. SUPERSEDED
  workarounds, in the order tried: blanket `^gdr_copy` runs but is 3.6× slow (UCX picks
  knem/xpmem/IB-loopback for intra-node GPU traffic); the explicit
  `UCX_TLS=self,sm,cuda_copy,cuda_ipc[,dc_mlx5]` list recovers that; **the real answer is
  `FESOM_HALO_STAGE=1` (see the s17 act-2 entry / dolpung campaign doc): device-packed
  halos with the MPI leg on pinned-host mirrors — full speed, no GPUDirect, CG levers
  live.** Levante's `UCX_NET_DEVICES=mlx5_0:1` pin does NOT transfer (as the JUPITER plan
  warned); plain srun + pmix works with zero MPI flags.
- **Scheduler quirk:** fresh `mh1571_gpu` association ⇒ priority 1 (all sprio factors 0);
  jobs sit PENDING ~40 min beside 39 idle nodes, with a misleading "Nodes DOWN/DRAINED/
  reserved" reason. Not an access problem (you must be in Unix group `dolpunguser` +
  an allowed account; both held). Budget the lag or hold `--no-shell` allocations.

*Keep appending. Date entries when the context (versions, paths) might age.*

## L103 — the debug fallback that ate a factor of two (dolpung, 2026-07-22)

The dolpung fleet's first 21 points ran with `FESOM_HOST_HALO=1`, chosen because it was
the only setting that survived the fabric's missing GPUDirect. It produced a coherent,
publishable-looking campaign that said "GH200 ≈ A100" — and it was wrong by ~2×. The
user's calibration ("JUPITER is 3× A100 and dolpung ≥ JUPITER") is what re-opened it.

- **What the knob actually does:** `FESOM_HOST_HALO` is a *debug* toggle (M5.1-era A/B
  against the device path). It reverts every exchange to the legacy FULL-FIELD
  device↔host sync bracket **and**, because `fesom_halo_device_active()` gates them, it
  silently deactivates CGPIPE and CGPOLY. Both printed `!! requested but INACTIVE` in
  every single v1 log. Nobody read them. **A perf config must be audited for what it
  turns OFF, not only for what it turns on** (L80's dead-knob rule, inverted).
- **The fix that recovered the hardware:** `FESOM_HALO_STAGE=1` — keep the device
  pack/unpack and the per-partner packed buffers; stage only the PACKED bytes through
  pinned host for the MPI leg. 10–100× less traffic than full-field, no device pointers,
  CG levers alive. CORE2 g1: 0.0788 → 0.0460. Fleet-wide: 1.35–1.9× over A100-best
  instead of parity.
- **Two hypotheses falsified on the way** (both plausible, both cost a job each):
  1-core-per-rank SLURM binding (`-c1` 0.0789 vs `-c72` 0.0791) and UCX
  memtype-cache/CUDA-layer overhead. Neither mattered. The negative results are what
  made the third hypothesis (transport bytes) the only survivor.
- **Certification corollary:** we tried to gate STAGE with a trajectory bitwise diff and
  got "DIVERGENCE" — then measured the CUDA build against *itself*: dev-vs-dev = 3.7e-4
  at 10 steps, dev-vs-stage = 3.8e-4. **Trajectory bitwise gates are meaningless for the
  CUDA build** (atomics ⇒ run-to-run nondeterminism). The valid instruments are the
  per-exchange halo selfcheck and `FESOM_CGPOLY_SELFCHECK` (both clean under STAGE), plus
  the untouched Serial byte gates. A "divergence" from an invalid gate nearly buried a
  correct 2× speedup.

### L102 — A THRESHOLD-GATED CODE PATH CAN BE DEAD WHILE THE KNOB IS ALIVE: `FESOM_WSPLIT=1` announced itself, ran, and produced BIT-IDENTICAL output — because the CFLz never reached the trigger. (M7 wsplit certification, 2026-08-04)

L80 says: verify the knob FIRED before believing a null A/B. Its usual failure mode is a knob
that never got read (a `getenv` behind the wrong `#ifdef`, a value cached before the export).
This is the *next* one out: the knob is read, the announce prints, the resolve is correct — and
the guarded arithmetic still never executes, because the branch is `if (knob && quantity >
threshold)` and the test configuration never crosses the threshold.

| CORE2 dist_8, 20 steps, dt1800 | max CFLz | trigger | splitter | diff vs knob-off |
|---|---|---|---|---|
| `FESOM_WSPLIT=1` (default maxcfl 1.0) | 0.822 | 1.0 | never fires | **rc=0 BIT-IDENTICAL** (26695054) |
| `FESOM_WSPLIT=1 FESOM_WSPLIT_MAXCFL=0.3` | 0.822 | 0.3 | `w_i/w` ≈ 0.63 | differs (26695170) |

Both runs print `[wsplit] FESOM_WSPLIT = ON`. Read as a gate, the first says "wsplit is
byte-safe"; it actually says nothing about wsplit at all. Had the certification stopped there,
the ladder would have been 10/10 green on a device kernel that was never launched.

**The rule: a lever whose code is behind a numerical threshold needs the threshold as a knob
too, and the certification config must be shown to CROSS it — with an in-run measurement, not
an argument.** Here that measurement is the `[cflzmax]` line printing `w` and `w_i` at the
global-max cell: `w_i` ≠ 0 IS the proof the branch ran. Making `wsplit_maxcfl` settable also
closed a Fortran-parity gap (it is a namelist parameter there), which is the usual shape of
this fix — the constant was hardwired precisely because the reference config never varied it.

**Corollary for the inert case:** the bit-identical result is not worthless — it is the *null
rung* of the ladder, proving the path is inert below threshold, which is a real requirement.
Record it as that, and never as evidence the machinery works.

### L103 — THE OCEAN-PHASE LOAD IMBALANCE IS BATHYMETRIC, AND VERTEX WEIGHTING CANNOT BUY IT BACK: balancing the vertical costs a ~90× worse edgecut, which GPUs punish 30 %. Partition track CLOSED. (M10, 2026-08-06)

**The imbalance is real and large.** Per-rank `ocean` busy spans 4.8× (farc 2048 ranks:
8.8 / 26.6 / 42.5 ms; CORE2 864: 3.8 / 13.0 / 18.2). A bulk-synchronous phase costs its
maximum, so the tax is **11 % of the CORE2 step and 19 % of the fArc step** — comparable to
the entire barotropic solve.

**1. It is BATHYMETRY, not ice, and not the horizontal partition.** Correlating per-rank ocean
compute against per-rank mesh content over all 2048 fArc ranks:

| correlation with per-rank ocean compute | r |
|---|--:|
| **3D nodes owned** (Σ `nlvls` over owned columns) | **0.967** |
| 2D nodes owned | 0.003 |
| solver **wait** vs 3D nodes owned | **−0.771** |

The partition balances *surface* nodes to 1.01× while the work is per water column, and 3D
nodes span **9.4×**. Fit: `ocean_busy = 2.24 ms per 1000 3D nodes + 10.2 ms`. The negative
correlation for solver wait is the causal closure: **deep-column ranks wait LEAST because they
are the stragglers** — the ranks everyone waits for are the deep ones, not the icy ones, which
is why an ice-based hypothesis tests negative.

**2. A null result on this lever is worthless unless the partition actually CHANGED.** The
M7-era conclusion rested on regenerating NG5 with 3D weighting — but /pool NG5 was *already*
dual-weighted (3D spread 1.04×), so the regenerated dists came out **byte-identical** and the
A/B compared a partition against itself. That is L80 at partition scale. **Check the 3D
balance of the partition you are "fixing" before believing any repartitioning null.**

**3. Measured properly on a mesh that IS imbalanced, it still loses — and the reason is not
the halo node count.** CORE2, one binary with runtime-selectable weighting (the knob matters:
it is a compile-time `#ifdef PART_WEIGHTED` in `fort_part.c`, so 2D-only and dual otherwise
require two binaries), all arms byte-identical on the mesh definition:

| backend / rung | vertices/core | dual vs 2D-only |
|---|--:|--:|
| CPU 256 r | 495 | **−4.62 %** ✅ |
| CPU 512 r | 248 | 0.00 % (crossover) |
| CPU 864 r | 146 | +8.71 % |
| **GPU 8 r (2 nodes)** | 15857 | **+29.74 %** ❌ |

**4. Halo NODES understate the damage by two orders of magnitude — use EDGECUT.** Dual
weighting grows halo nodes ~40 %, which predicts a ~1 % compute penalty. Measured penalty:
**+13 % CPU, +20 % GPU** on ocean compute *mean*. The reason is in the cut, not the surface:

| CORE2 ranks | edgecut 2D-only | edgecut dual | ratio |
|--:|--:|--:|--:|
| 8 | 1 335 | 120 883 | **91×** |
| 16 | 2 549 | 217 791 | 85× |
| 32 | 4 307 | 375 211 | 87× |

Balancing the vertical forces METIS to fragment subdomains. That is a locality collapse, and
**the GPU punishes it far harder than the CPU** (memory-access sensitivity): even the
regenerated *2D-only* partition is +4.18 % against the shipped one on GPU, versus +0.68 % on
CPU. **Any regenerated decomposition must be measured on GPU before use, not merely checked
for balance.**

**5. Some partitions simply do not work — this is known FESOM behaviour, not a port defect
(user, 2026-08-06; L99 again).** A regenerated 2D-only `dist_128` made baseline CG diverge at
iteration 1 (`residual=5.49698e+45`) where the shipped and dual-weighted partitions at the
same rank count both ran clean. **Do not debug this.** A freshly generated decomposition can
be unusable for reasons that have nothing to do with the port, so the rule is operational, not
diagnostic: **smoke-test every new partition for a few steps before it is used for anything,
and discard the ones that fail.** Budget for a fraction of generated partitions being thrown
away.

**VERDICT — the partition track is CLOSED.** The imbalance is genuine and worth ~6 ms of a
45 ms step, but vertex weighting trades a 1.2–1.5× compute spread for a ~90× worse cut and
loses everywhere that matters: it wins only on CPU below ~250 vertices/core, has no GPU
counterpart, and the production point sits on the crossover. Do not re-derive this. If the
imbalance is ever attacked again it must be by a method that does **not** cost edgecut —
which vertex weighting inherently does.

### L104 — ON A SERIAL BUILD EVERY `FESOM_SPEED_*` LEVER IS INERT WITHOUT `FESOM_SPEED_FORCE_SERIAL=1`. Setting the knob is not the same as the knob acting. (M10, 2026-08-06)

`fesom_speed.hpp:111-113`:

```c
#ifndef KOKKOS_ENABLE_CUDA
  if (on && !fesom_speed_force_serial()) on = 0;   /* Serial stays legacy */
#endif
```

The lever resolves to **OFF** on a Serial/CPU build unless `FESOM_SPEED_FORCE_SERIAL=1` is
also set. This is deliberate (the Serial backend is the bit-identical oracle and must stay
legacy), and `job_m10_ab_cpu` sets it correctly. **It bit twice in one session in hand-written
jobs:**

1. `FESOM_SPEED_PHASESTATS=1` produced no phase table at all in two 7-node runs.
2. A NG5 diagnosis job set `FESOM_SPEED=1` without it, so the "speed" and "legacy" legs were
   **the same run twice** — they returned byte-identical fields (`uv=2.56e+00`, `eta=1.35e+00`)
   and reproduced none of the failure being investigated.

**Both were silent.** The resolver does warn (`!! … WAS REQUESTED BUT RESOLVES TO OFF`) but on
*stderr*, which a job that redirects per-leg output will not surface, and case-sensitive greps
for "phasestats" miss a message naming `FESOM_SPEED_PHASESTATS`.

**Rules.** (i) Any hand-written CPU job exercising a `FESOM_SPEED_*` lever must export
`FESOM_SPEED_FORCE_SERIAL=1`, or copy the knob block from `job_m10_ab_cpu`. (ii) **Assert the
lever ANNOUNCED itself** before trusting any A/B — the announce is the only sound test.

🔴 **CORRECTION (same day): the tempting third rule — "identical field diagnostics prove the
knob did not fire" — is WRONG, and I used it to reach a false conclusion.** Most of the
`FESOM_SPEED=1` set is *certified bit-identical* (CGPIPE et al.), so identical output across
that boundary is exactly what a correctly-firing knob produces. In the run that provoked this
lesson the knob HAD fired (`FESOM_SPEED_FORCEDEV/ROTCACHE/IOACC/BULKTAIL = ON` in the log)
while the fields matched to the last digit. Conversely **silence is not proof of absence**:
`FESOM_VISC_OPT` (`fesom_step.cpp:658`) changes the viscosity scheme and announces NOTHING, so
grepping for it and finding nothing says only that it is a quiet knob. Judge a knob by its
announce, or by an observable it is *designed* to move — never by field equality.

### L105 — A MESH SHIPS PARTITIONS OF DIFFERENT QUALITY, AND A BAD ONE BLOWS THE MODEL UP. Check the partition before blaming the timestep. (M10, 2026-08-06)

**Measured (26749650):** NG5, 4096 ranks, dt180, cold start, one allocation, one binary, only
the decomposition differing:

| partition at 4096 ranks | 3-D balance | 300 steps |
|---|--:|---|
| /pool, as shipped | **13.89×** | **BLOWUP at step 175** (`uv` 5.76 and climbing) |
| freshly generated, our partitioner | **1.05×** | **completed**, `uv` 3.25, 0.6205 s/step |

**The /pool NG5 partitions are not homogeneous, and the split predicts the failure exactly:**

| shipped | 3-D balance | outcome |
|---|--:|---|
| `dist_1024` | 1.03× | completes |
| `dist_2048` | 1.04× | completes |
| `dist_4096` | **13.89×** | blows up |
| `dist_8192` | **13.98×** | blows up |

The small counts were built with dual 2-D+3-D weighting; the large ones were not. Four for
four, balance predicts survival.

🔴 **This RETRACTS the M10 reading that "NG5 at dt180 cold is a physics limit shared with
Fortran", and it undermines the M7 evidence it rested on.** M7 recorded the same death at step
200 (port) / 203 (Fortran) — but only ever on `dist_4096`, i.e. on one of the two anomalous
partitions. Fortran dying there too proves only that Fortran also dislikes that decomposition.
**Neither campaign varied the partition, so both read a property of one bad decomposition as a
property of the configuration.**

**Rules.** (i) When a high-resolution mesh blows up, **measure the partition's 3-D balance
before touching dt, viscosity or the solver** — it is minutes of analysis against hours of
node time chasing physics. (ii) Do not assume the partitions shipped with a mesh were all
built the same way; check, per rank count. (iii) A regenerated decomposition is not
automatically worse — here it was strictly better, and it unblocked measurements that had
looked physically impossible. (iv) Balance is a *marker*, not the mechanism: skewed work costs
time, not accuracy, so what actually breaks the model is whatever else is wrong with a
badly-formed partition (elongated or fragmented subdomains). The mechanism is untested.

*(Consistent with the standing user observation that some FESOM partitions simply do not work
— L103 §5. The difference here is that the failure is not random: it tracks a measurable
property of the partition, so it can be screened for in advance.)*

### L106 — A GUARD THAT IS BLIND TO NaN TURNS A BROKEN RUN INTO A WIN. The failure mode is not a crash, it is a *fast* number. (M10/M13, 2026-08-14)

The three M10 solver variants entered their iteration loop under `if (resid >= rtol)`.
**`NaN >= rtol` is false**, so a solve entered with a non-finite state skipped the loop
entirely and returned "converged, 0 iterations" — at zero cost. A run whose state had gone
NaN therefore did not crash: it completed all 300 steps, printed a state row of `it=0`,
`uv=0` and inverted ±1e30 T/S sentinels, and reported a *timing*.

Measured, on the one partition where both behaviours could be produced side by side
(`core2_wgt0/dist_128`, 1 node, job 26961492):

| | s/step |
|---|--:|
| zombie (NaN state, campaign binary) | **0.1838** |
| healthy run, same config, fixed IC | **0.2060** |

**The zombie is 10.8 % faster** — indistinguishable in magnitude from the wins the campaign
was measuring. Four of 77 archived A/B runs contained such legs (all already void for other
reasons; no quoted row was affected, but that was luck, not process).

Three transferable rules:

1. **Every convergence/stall test must have an explicit non-finite branch.** A comparison
   with NaN is false, so *any* `if (bad) fail;` guard silently passes NaN. Write the criterion
   the way the reference implementation does — baseline `cg` here has always died on
   `residual != residual || residual > 1e30`, which is exactly why stock CG was the only
   scheme that failed loudly and why "SE and oati run clean" was an exit-code illusion.
2. **A completed run is not a valid measurement.** The harness already refused to quote a leg
   with `fallbacks>0`; it had no test for a leg that finished at zero cost. Terminal-state
   sanity (finite `uv`, physical T/S range) belongs in the harvest, next to the fallback count.
3. **The cheapest reproduction wins.** The zombie was first seen on NG5 at 64 nodes. The same
   phenomenon lives on a 1-node CORE2 partition that the load-balance study had already flagged
   as unusable — before/after in three minutes instead of an hour of 64-node time. When a
   failure is a property of the *initial condition*, mesh size is not part of the mechanism.
