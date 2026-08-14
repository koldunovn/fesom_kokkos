# M13 det-IC rollout — user decisions + re-run plan (2026-08-14/15)

**User decisions (Nikolay, 2026-08-14 evening):**
1. **Default stays as close to vanilla FESOM as possible** — the legacy (vanilla-equivalent,
   partition-dependent) fill remains the default; the fix is the opt-in knob
   `FESOM_IC_EXTRAP=det`. (Already implemented this way; knob-off byte gate passed ×2.)
2. **The knob is used for all campaign experiments** — SSH solvers (M10), split-explicit (M12),
   sea ice (M9) — and everything that makes sense is re-run under det → clean, cross-partition-
   comparable scaling curves.
3. **The partition-exploration code (M11) is patched to see the change**, with a plan to re-run
   its failed experiments.
4. **A PR with the fix goes to main FESOM** (Fortran, gen_support.F90 — same defect upstream).

Root-cause + fix + acceptance evidence: `docs/CG_BLOWUPS_M13.md`. All statements below assume
that document.

## 0. Where the knob now lives

| tree | branch | state |
|---|---|---|
| `~/port_kokkos` (main checkout) | `m13-cg-robustness` | native (d1861c2 + a61c047 + docs) |
| `~/port_kokkos_part` (M11) | `m11-partition` | cherry-picked (4685cce + 7a084e9) + handoff addendum (bbcb934) |
| `~/port_kokkos_ssh` (M10) | `m10-ssh-solvers` | cherry-picked (ddae127 + e7ff7ae) |
| `~/port_kokkos_ice` (M9) | `m9-mevp-double` | cherry-picked (0f87e4d + 41da460) |
| `~/port_kokkos_wh` (M12b) | `m12b-widehalo` | **NOT touched (active session)** — cherry-pick `d1861c2 a61c047` from the main checkout when convenient; byte-inert with the knob off, so no gate impact |
| `~/port_kokkos_mp` (M8) | `m8-precision` | not patched (storm hunt is a mid-run year-7 event, unrelated); cherry-pick if SP scaling curves are refreshed |
| `m12-split-explicit` branch | — | m13 is a fast-forward of it; the M12 session can `git merge m13-cg-robustness` (FF) or keep committing docs on m13 |

⚠️ **Every re-run needs a re-pinned bin** — the existing frozen bins (se0 pair, m10
`stallknob_*`, M9/M11 pins) do NOT have the knob. Available now:
`/work/ab0995/a270088/port2/m13/bin/fesom_port_serial_det1` (sha 95a4d6a4…, = m12-tip + probe +
det, byte-inert extras when knobs are off — validated 8×) and `fesom_port_cuda_det1`
(d20ba293…, first GPU leg bg64det 26960226 pending in queue). Other tracks build from their
patched branch tips.

## 1. M12 split-explicit — rebuild the NG5 board + suspect points (HIGH)

The G4 NG5 rows are the contaminated ones: on dist_4096/8192/32768 the SE and oati legs were
NaN zombies (oati's 0-iteration steps are artificially fast), and every legacy NG5 cold start
integrates a different IC. Under det all six partitions run clean and produce identical state
rows → a full, honest NG5 scaling curve becomes possible for the first time.

- **Re-run**: NG5 CPU ladder 32N/64N/128N/160N/192N/256N × {si, se} × reps a,b (+ one
  phasestats rep at the board points), dt 180, wsplit=1, `FESOM_IC_EXTRAP=det`. Driver:
  `m13/submit_g4_det_ng5.sh` (below). oati arms additionally need an m10-branch rebuild
  (the m10 bin lacks the knob) — decide whether oati stays on the re-board first.
- **Re-run**: dars 2048-CPU G4 pair (the +1.1% verdict is suspect — dars fills are
  demonstrably dangerous: F45 = Marmara) and the dars GPU point.
- **Refresh for curve consistency (cheaper, same-day rule applies anyway)**: farc 2048 CPU,
  CORE2 4N/16N GPU (needs cuda_det1-class bin).
- House rules unchanged: same-day pinned pairs, min-of-2, ladder dt, `BIN=` pinned, cheap
  walltimes, ≤16 GPU nodes.

## 2. M11 partitioning — voided verdicts + gate methodology (HIGH)

All M11 stability "diverged/blowup" verdicts measured the IC artifact, not the partition:

- **Re-screen under det** (3,000-step, ladder dt): NG5 2048-CPU alternates (F34 legs a4m,
  a4u30 — a4m_seedb's "clean" run is equally void as evidence), NG5 64-GPU alternates, dars
  2048 `MINCONN` (F39 — if clean, re-race dars CPU: the KaMinPar recommendation could revert),
  dars 64 u30's dist_2048 sibling. dars seed660013 dist_2048: already flipped (f45det
  26960206, 300 steps; extend to 3,000 for the formal screen).
- **Re-race where the verdict could change**: NG5 2048 CPU (was a NULL with 3 of 4 arms dead —
  now 4 live arms), dars 2048 CPU.
- **Accuracy-gate re-derivation**: the seed-control envelopes included IC spread between
  partitions; under det all controls share bitwise-identical ICs → envelopes shrink → the
  disturbance tiers (dars 64 u30 −19.7 % tier 4, dars a4m −14.3 % tier 2, NG5 a4m −9.8 %
  tier 3, fArc 16 GPU, CORE2 864) should be re-derived before any of them is acted on.
  Expect sharper discrimination, in both directions.
- The four PROMOTED certified meshes are partition FILES — unaffected as artifacts; their
  legacy-IC screen evidence stands as-run. Future `m11_promote` evidence should be produced
  with det (record the knob in the evidence line).

## 3. M9 sea ice (MEDIUM)

The EVPWIDE/lean headline table (CORE2/farc/dars/NG5 GPU points, both currencies) used legacy
ICs; the ice-relevant contamination is indirect (ocean state under the ice) but the NG5/dars
points inherit the general caveat. When the track resumes: rebuild a pinned bin from
`m9-mevp-double` tip (patched), re-run the NG5 + dars points under det first, CORE2/farc as
convenient.

## 4. M10 SSH solvers (MEDIUM-LOW)

No known zombies (the farc "divergences" were stall-guard false positives — a different,
already-understood story), but the solver-comparison tables should eventually be det-uniform
with the M12 re-board (shared farc/dars points). Branch patched; rebuild `stallknob_*`-class
bins from `m10-ssh-solvers` tip when re-running. Note oati/pcsi/cg2's stall-guard NaN-blindness
(`0 iters forever`) — worth a guard fix (isfinite check) in that worktree before its bins are
reused for anything load-bearing.

## 5. Upstream PR (in progress)

Fortran translation of the two-phase deterministic fill into `gen_support.F90` behind a
namelist flag (default `.false.` = vanilla behavior), PR onto FESOM/fesom2 main following the
M11 upstream-PR pattern (`docs/upstream/PR_meshpart_*.md` precedent). Status + body:
`docs/upstream/PR_ic_extrap.md` (this session).

## 6. Explicitly NOT re-run

- M7 certified board + 63-yr hindcast (CORE2, legacy fill on a mesh with minimal dummy
  coverage; the certification is vs the C reference under identical ICs — internally
  consistent).
- M8 1964-storm hunt (mid-run event, restart-class — IC-independent).
- M12b W-gates (bitwise SE-vs-wide comparisons — IC-invariant by construction).
