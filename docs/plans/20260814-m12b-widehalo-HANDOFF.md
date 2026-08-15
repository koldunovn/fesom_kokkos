# M12b — wide-halo infrastructure for SE subcycling (+ future higher-order advection) — HANDOFF

**You are a fresh session in the worktree `~/port_kokkos_wh`, branch `m12b-widehalo`
(off `m12-split-explicit` @ ab25f6c).**
🔴 FIRST: your auto-memory index is worktree-scoped — read
`/home/a/a270088/.claude/projects/-home-a-a270088-port-kokkos/memory/MEMORY.md` (the MAIN
index) before anything else. Then read `docs/SSH_SE_M12.md` (the SE track reference) and
`docs/plans/20260813-m12-split-explicit.md` (full gate record; the M12b GO decision block).

## The task

Build a **K-ring wide halo as GENERAL ocean-mesh infrastructure** — not an SE-local hack —
and make the SE subcycling its first customer (exchange every K substeps instead of every
substep). Nikolay's explicit requirement (2026-08-14): design it so it can later serve
**higher-order advection schemes** as a second customer.

## Why (measured motivation, 2026-08-14 boards)

- The SE block (FESOM_SSH_MODE=se; module `src/fesom_ssh_se.{h,cpp}`) does 2 tiny exchanges
  per substep (η NOD2D scalar + Ū ELEM2D pair), M=20–90 substeps/step. Phasestats: at
  **CORE2 16N GPU the bt phase is 7.8 ms busy + 11.8 ms MPI WAIT (60% wait)** of an 84 ms
  step — the wait is the remaining removable cost; NG5 64 GPU: 32% wait. CPU at scale:
  farc-2048 bt = 6.6 busy + 5.2 wait ms.
- Sergey (2026-08-14): "На GPU переход на wide halo должен принести преимущества, а на CPU
  это просто лучшее скалирование" — GPU-first; the CPU variant is a scaling play (his notes:
  CPU only doubled halo, <300 verts/core regime).
- M9 precedent (the mEVP EVPWIDE lever, worktree `~/port_kokkos_ice`): K=8 exact wide halo
  with LEAN rim writing is certified for ICE nodes — the pattern (K-ring comm lists, exact
  equivalence to per-substep exchange, lean writing = only the rim depth actually needed,
  L109 raw-pointer captures) is the design template. Read the M9 memory file + its handoff.

## Design sketch (from the M12 session — brainstorm it properly, don't take it as gospel)

1. **Mesh infra:** at startup (knob-gated), build K-ring comm lists for nod2D and elem2D
   (ring 1 = the existing eDim/com_nod2D/com_elem2D; rings 2..K grown by adjacency), device
   Fields for the ring buffers, a generic `fesom_halo_field_wide(f, kind, nl, nc, K, p)`
   entry over the existing device-halo brackets. This is the piece higher-order advection
   reuses later (it needs ≥2-ring element halos for wider stencils).
2. **SE consumer:** with K-ring-valid state, run K substeps locally (each substep shrinks
   the valid rim by 1 ring for η and Ū alternately), exchange once per K substeps with
   K-deep messages. EXACT variant first (redundant rim compute, bitwise-equivalent result —
   the M9 "exact wide" certification), LEAN writing second (M9: lean took wide from −10…−28%
   to −37…−58% of the ice). K=8 start per M9/Sergey.
3. **Gates:** (a) knob-off null (byte gate vs the m12 branch tip); (b) K=1 ≡ current path;
   (c) EXACT-equivalence: K-wide run bitwise/floor-equal to the per-substep-exchange run on
   Serial (the strongest gate M9 used); (d) the SE invariant suite (FESOM_SE_CHECK) must stay
   machine-class; (e) perf pairs at CORE2 16N GPU + NG5 64 GPU (the 60%/32% wait points),
   then CPU doubled-halo (K=2) at the big-mesh scale points if the phasestats say so.

## Constraints and house rules (binding)

- The M12 SE module is CERTIFIED — do not change its numerics; wide halo must be a pure
  communication/scheduling transformation (the exact-equivalence gate enforces this).
- Knobs are proper options (abort on unrecognized; the fesom_wsplit_on template).
- L109: capture raw pointers in kernels (closures >512 B go to constant-memory ghost path).
- No atomics in the SE path (SASS-audited today — keep it that way; gather forms only).
- BIN= pinning, same-day pinned pairs, `-C a100_80` on GPU absolutes, cheap gates get
  `-t 00:06:00`-class walltimes, ≤16 GPU nodes without asking, binaries → /work + sha only.
- Build dirs are NOT in the worktree (gitignored): create your own `build-m7serial`/
  `build-m7cuda` here (cmake -B, `env.sh`/`env_cuda.sh`; see docs/BUILD.md). ctest must pass.
- 🔴 CPU scaling questions live on farc/dars/NG5 — CORE2 is GPU points + ≤128r gates ONLY
  (user rule 2026-08-14, memory `feedback-no-core2-cpu-scaling.md`).

## Coordination with the M12 session (main checkout, branch m12-split-explicit)

The M12 session is finishing harvest/docs on `m12-split-explicit` — expected to add only
docs/plan/report commits from here on. If it ever has to touch `src/` again it will say so
loudly in its reports; rebase onto the updated m12 tip in that case. Merge plan: M12 merges
to main first; m12b rebases (docs-vs-src should be disjoint → trivial).

## First actions for this session

1. Read the main MEMORY.md + `docs/SSH_SE_M12.md` + the M12 plan + the M9 memory file.
2. `/brainstorm` the infrastructure design (ring construction, storage layout, the
   advection-reuse API surface, exact-vs-lean phasing), then `/planning:make` the M12b plan.
3. Implement gate-by-gate per the house ladder.
