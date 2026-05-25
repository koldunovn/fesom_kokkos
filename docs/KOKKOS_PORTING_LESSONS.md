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
| D14 | **np=2 pi run on `dist_2` (login-node, `vader` shmem) is the cheap gate for `scatter_mesh`.** Captured a pre-change np=2 snapshot oracle (`/scratch/a/a270088/pi_np2_ref_m12`); a correct scatter refactor must reproduce it byte-for-byte | The np=1 pi smoke does **not** call `scatter_mesh` (single rank → identity), so it can't catch a bug in the MPI mesh redistribution that Wave 2 edits. The global gathered snapshot is rank-count-deterministic, so np2-after == np2-before is a zero-tolerance gate for the data shuffle, runnable in seconds without a compute node |

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

## C. Process lessons

- Validate at **every** step on Serial bit-identity; commit per milestone-step; **tag** milestones.
- Record **provenance** (C source SHA, exact build/run recipe) so any reference is regenerable.
- nvcc full-model compiles are slow → run them **in the background**; iterate config on the login
  node, run device smokes via `srun -p gpu-devel -A ab0995 --gres=gpu:1`.

---

*Keep appending. Date entries when the context (versions, paths) might age.*
