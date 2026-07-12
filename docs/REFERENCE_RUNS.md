# Reference runs for FESOM2 Kokkos-port climate validation

Catalog of the canonical CORE2 reference runs used to validate Kokkos-backend
climate output (M3.2 and beyond). For each ref: path, physics knobs, build
provenance, and what it's good for.

**Default policy (2026-05-28)** — KPP is the default mix scheme for everything in
this repo going forward (CORE2 production, comparison scripts, ref selection).
Only use PP references when the user explicitly opts in. The Kokkos port HEAD
defaults to KPP (`src/fesom_step.cpp:113`: `s_use_kpp=1`).

---

## ⚠️ CORE2 mesh: use the private copy, not `/pool` (2026-07-12)

**`MESH=/work/ab0995/a270088/port2/mesh/core2`** — every job from M6 on must point here.

On **2026-07-03** the shared `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2` mesh had
`nlvls.out` and `elvls.out` (the per-node / per-element level counts = bathymetry)
replaced: 2 nodes and 4 elements changed on the **Ross Sea shelf** (node 125225: 22→18
levels, node 125227: 20→19; ≈154°W/77°S), all shallower. The port reads those files
directly (`fesom_mesh.c:340-361`), so it is a real bathymetry change — and **every
reference in this document was produced before it**.

Caught in M6 Task 0.1 by a regression gate: a fresh Kokkos-Serial run diverged from the
standing `serref_m522_saved` oracle at **step 0**, in `nlevels`/`nlevels_nod2D` — fields
that cannot change unless the mesh does. Signature to recognise: `lat`/`lon`/`elem_nodes`
match, `nlevels*` do not.

| | sum(nlevels_nod2D) | sum(nlevels_elem) |
|---|---|---|
| `/pool` today (2026-07-03 files) | 3 832 745 | 7 366 741 |
| **private copy = every archived ref** | **3 832 750** | **7 366 752** |

The private copy is a full `cp -a` of the /pool tree with the pre-2026-07-03 levels
restored from the in-place backups (`*.20260528_regenerated`); the July-3 /pool version is
kept alongside as `*.20260703_pool`, so neither bathymetry is lost and `/pool` is untouched.
Proof it is the right one: with it, a fresh Kokkos-Serial run is **bit-identical to
`serref_m522_saved`** (gate 2 of `jobs/job_m6_oracle_cert`). Full detail:
`$MESH/MESH_PROVENANCE.md`.

**Rule going forward: record the mesh level-sums with any reference you archive.** A
bathymetry swap under a shared mesh path is invisible until it silently fails a bit-gate.

---

## The M6 C oracle (2026-07-12)

**`/home/a/a270088/port2/fesom2_port_zstar` @ `df8b9a8`**, fresh CMake Release build at
`build-m6oracle/fesom_port`. This replaces the old C oracle (`fesom2_port`) for the M6
options campaign, because it is the C tree that carries TKE + mEVP + zstar.

**Certified** by `jobs/job_m6_oracle_cert` (SLURM 26210028, 2026-07-12), CORE2 dist_8 /
8 ranks / dt=1800 / 20 steps / snap_every=10, all M6 knobs unset:

| gate | comparison | result |
|---|---|---|
| 1 (the certification) | C oracle `df8b9a8` vs Kokkos-Serial `b678974` | **ALL FIELDS BIT-IDENTICAL** |
| 2 (lineage) | Kokkos-Serial `b678974` vs `serref_m522_saved` | **ALL FIELDS BIT-IDENTICAL** |

Gate 1 says: with every option knob off, the zstar tree is the same model the Kokkos port
was validated against — so it is a sound oracle to port the three options from. Gate 2 says
the private mesh reproduces the archived reference lineage exactly.

**M6 knob-OFF baseline:** `/work/ab0995/a270088/port2/m6_baseline_serial/` (the gate-1
Kokkos-Serial snapshots + `PROVENANCE.md`). Every M6 task that touches shared code must
reproduce it byte-for-byte with the knobs off.

C-oracle knob semantics (verified against the Kokkos port's — they agree at default):
`FESOM_MIX_SCHEME` unset→KPP / `P*`→PP / `TKE`|`cvmix_TKE`→TKE; `FESOM_WHICH_EVP`
unset|`0`→std EVP, `1`→mEVP, else abort; `FESOM_ALE` unset|`linfs`→linfs, `zstar`→zstar,
else abort.

---

## Canonical references (DEFAULT for KPP runs)

### Fortran-KPP — `/scratch/a/a270088/fortran_kpp_5yr_fix`
- **Length:** 5 yr (1958–1962), instantaneous (`<var>.fesom.<yr>.nc`)
- **Source workdir:** `/home/a/a270088/port2/fesom2/work_kpp_5yr_d1800/`
- **Physics:** `mix_scheme='KPP'`, `ice_gamma_fct=0.5`, `linfs`, dt=1800, CORE2 namelist
- **Use:** absolute "does the GPU reproduce the science?" budget. Combines
  Fortran↔C divergence + C↔CUDA divergence. CUDA-vs-F = the full skill gap.

### C-port-KPP — `/work/ab0995/a270088/port/kpp_5yr_fix`
- **Length:** 5 yr (1958–1962), monthly means (`<var>.fesom.<yr>.monthly.nc`)
- **Build:** C-port `6ecabe8` ("KPP K11: wrap-up — KPP complete & default"), May 25 2026
- **Job:** `jobs/job_kpp_5yr_fix` (sets `FESOM_MIX_SCHEME=KPP`, dt=1800, 864r); same SHA as `kpp_2yr_rebase` (its 2-yr cousin), but matches the C-port's published `kpp_5yr_fix_figures` validation set.
- **Physics:** KPP, `ice_gamma_fct=0.5` (post-fix `7c6663b`), linfs, dt=1800
- **Use:** **the canonical backend-vs-C isolation reference for KPP Kokkos runs.**
  C twin == Serial Kokkos == bit-identical, so CUDA-vs-this isolates the GPU
  scatter/reduce drift (D22). Validated against Fortran-KPP at C-port commit
  `375f3eb` ("C+KPP reproduces Fortran-KPP").

---

## Secondary references (PP only — use only when explicitly running PP)

### Fortran-PP — `/scratch/a/a270088/fortran_pp_2yr`
- 2 yr (1958–1959), instantaneous. Source: `work_linfs_pp_2yr/` (`mix_scheme='PP'`).
- Only use if the Kokkos backend was run with `FESOM_MIX_SCHEME=PP`.

### C-port-PP (post-fixes) — `/work/ab0995/a270088/port/pp_2yr_rebase`
- 2 yr (1958–1959), monthly. C-port post-bvfreq + post-gamma fixes.
- Job: `jobs/job_pp_2yr_dt1800` (sets `FESOM_MIX_SCHEME=PP`).
- Use only paired with a PP Kokkos run.

---

## Deprecated references (do NOT use for new Kokkos-port comparisons)

These were the original handout choices (`docs/NEXT_SESSION_BIAS_INVESTIGATION.md`).
They predate the KPP default flip and the `ice_gamma_fct=0.25→0.5` fix; using
them against a KPP Kokkos run conflates port drift with **2 physics-config deltas**
and produced the spurious "M3.2 paradox" (CUDA-vs-C ≈ −0.236 °C SST bias).

| Path                                            | Build       | Why deprecated                                                                   |
|-------------------------------------------------|-------------|-----------------------------------------------------------------------------------|
| `/work/ab0995/a270088/port/eps_2yr_dt1800`      | C-port `8dac9975` (May 23) | Default mix at that commit was PP (KPP flip came @ `8d0cdbc`); `ice_gamma_fct=0.25` (pre-fix `7c6663b`). |
| Fortran-PP refs used for KPP backends           | —           | Scheme mismatch on its own ~+0.2 °C SST.                                          |

---

## ⚠️ Vector frame: the port writes ROTATED, Fortran writes GEOGRAPHIC (2026-07-12)

FESOM computes on a **rotated grid** (CORE2 Euler angles 50/15/−90 — the pole is moved off
Greenland), so `(u,v)` and `(uice,vice)` live in the rotated frame internally. Who writes what:

| source | vector frame |
|---|---|
| Fortran | **geographic** — always (`io_meandata` rotates, `do_rotation`) |
| C port **< `75406d3`** (before 2026-06-11 20:37) | rotated |
| C port **≥ `75406d3`** | **geographic** (default; `FESOM_IO_VECTOR_FRAME=rotated` opts out) |
| **Kokkos port** | **rotated** — it has no frame knob (porting one is a Post-Completion item) |

Comparing across frames is an **isometry**: `|speed|`, ice extent and ice volume all look
perfect while the *components* decorrelate. That is a plausible-looking wrong answer, and it
had one on record here:

**The "known F↔C ice-edge budget" (uice corr ≈0.92) was this frame mismatch, not physics.**
Measured on the M5.23 CUDA 1-yr run vs the Fortran linfs+KPP reference, everything else held
fixed: `uice` 0.9187 → **0.9997** and `vice` 0.4266 → **0.9998** once rotated. Cross-checked
purely on the C side: `c_tke_2yr` (rotated) vs `fortran_linfs_tke` (geo) goes 0.9187 →
**1.0000**. It hid for the whole M5 campaign because `m32_climate_compare.py` compared `uice`
but **not** `vice` — a 0.43 would have been noticed at once.

Fixed: `scripts/fesom_frame.py` (the r2g transform + a per-output frame table) is now wired
into `scripts/m32_climate_compare.py`, which rotates every rotated-frame source to geographic
before comparing and now includes `vice`. Use `--cref-frame {geo,rotated}` to declare the C
reference's frame. Scalars (sst/sss/ssh/a_ice/m_ice) are frame-free and unaffected.

This is an equivalence, not an approximation: the C campaign gated its in-model rotation
against this same offline transform at 7e-15 (job 25524763), and the transform is verified
here as an isometry to 1e-16.

---

## Comparison verdict (CUDA 1-yr 1958) — REGENERATED 2026-07-12

Fresh run of `scripts/m32_climate_compare.py` on the **M5.23 CUDA 1-yr** output, with the
vector frame corrected and against the purge-safe Fortran anchor:

- backend: `/work/ab0995/a270088/port2/kokkos_gpu_runs/m32_cuda_m523fN_1yr`
- Fortran: `/work/ab0995/a270088/port/zstar/fortran_linfs_2yr_b` (linfs+KPP; **use this**, see below)
- C-port:  `/work/ab0995/a270088/port/kpp_5yr_fix` (`--cref-frame rotated`)

| field | CUDA-vs-Fortran | CUDA-vs-C-port |
|-------|--------------------------------------------|--------------------------------------------|
| sst   | corr 1.00000, bias +4.6e-5 °C, RMS 1.45e-2 | corr 1.00000, bias +1.1e-4 °C, RMS 1.41e-2 |
| sss   | corr 0.99996, bias −5.3e-4, RMS 2.62e-2    | corr 0.99996, bias −1.8e-4, RMS 2.61e-2    |
| ssh   | corr 1.00000, bias +2.1e-5, RMS 1.01e-3    | corr 1.00000, bias −1.4e-5, RMS 9.05e-4    |
| a_ice | corr 0.99997, bias +1.3e-4, RMS 2.89e-3    | corr 0.99997, bias +1.6e-4, RMS 2.86e-3    |
| m_ice | corr 0.99997, bias −3.9e-4, RMS 5.11e-3    | corr 0.99998, bias −1.5e-4, RMS 3.57e-3    |
| uice  | corr 0.99973, bias −8.8e-5, RMS 5.88e-4    | corr 0.99974, bias −8.1e-5, RMS 5.78e-4    |
| vice  | corr 0.99976, bias +8.4e-6, RMS 3.84e-4    | corr 0.99976, bias +1.1e-5, RMS 3.78e-4    |

**The port reproduces Fortran at corr ≥ 0.9997 on every field, ice velocity included** — and
CUDA-vs-Fortran is now indistinguishable from CUDA-vs-C-port, which is exactly what a faithful
port should show. There is no residual "F↔C ice budget" to explain.

The **previous version of this table** (a_ice 0.9066 / m_ice 0.98244 / uice 0.8502 vs Fortran)
is superseded. Three things were wrong with it, in increasing order of subtlety: it predates
the per-month NaN→0 ice-mask fix (2026-05-30, `feedback-ice-mask-averaging`); it used a
Fortran anchor whose binary predates the `2682a9fb` sbc cold-start wind-rotation fix (a known
C-vs-Fortran transient the port never had); and its vector fields were frame-mismatched (above).
The old CUDA-vs-C column was always sound — same frame, same convention — which is why the
scatter-drift story it told (O(1e-4) ocean, O(1e-3) ice) held up.

The historical "M3.2 paradox" note stands: that one was CUDA-KPP compared against C-port-**PP**.
See `docs/m32_bias_investigation.md` for that evidence trail.

---

## M6 campaign references (2026-07-12)

All under `/work/ab0995/a270088/port/` (purge-safe). Every one was produced on the
**2026-05-28 bathymetry** — i.e. the private mesh above, not today's `/pool`.

| feature | Fortran anchor (geo) | C comparator | C frame | dumps / bisect rails |
|---|---|---|---|---|
| **baseline** (linfs+KPP) | `zstar/fortran_linfs_2yr_b` | `kpp_5yr_fix` | rotated | — |
| **TKE** (M6.1) | `tke/fortran_linfs_tke` (2 yr) | `tke/c_tke_2yr` | rotated | `tke/fdump`, `cdump`, `cdump_v2`, `replay`, `t0_byteident`, `t4_xrank` |
| **mEVP** (M6.2) | `mevp/fortran_mevp_2yr` | `mevp/c_mevp_2yr` (+`c_evp_2yr` std-EVP control) | **geo** | `mevp/cdump_16r`, `fdump_16r`, `m1_byteident`, `m4_xrank` |
| **zstar** (M6.3) | `zstar/fortran_zstar_2yr` | `zstar/c_zstar_2yr` | rotated | `zstar/fdump`, `fdump_k2`, `z0_byteident`, `z2_cdump`, `z7_xrank` |
| **all-3** (M6.4) | `mevp/fortran_all3` | `mevp/c_all3_1yr` | **geo** | `iovec_gate` |

Pass `--cref-frame geo` for the mEVP and all-3 legs, `--cref-frame rotated` for TKE/zstar/KPP.
Reference namelists (all 10 per feature) + upstream provenance are vendored in
`jobs/m6_namelists/{tke,mevp,zstar}/`; every feature config is a verified **single-knob** clone
of the linfs+KPP baseline.

⚠️ `/scratch/a/a270088/fortran_kpp_5yr_fix` — the Fortran-KPP anchor named at the top of this
document — has been **purged down to restart files**; its output `.nc` are gone. Use
`/work/.../zstar/fortran_linfs_2yr_b` instead (same linfs+KPP config, purge-safe, and its
binary post-dates the sbc cold-start fix). Same for `/scratch/a/a270088/fortran_pp_2yr` —
verify before relying on it.

---

## Quick-pick by use case

- **Validating any Kokkos-port CORE2 KPP run (the default):**
  `--fref /scratch/a/a270088/fortran_kpp_5yr_fix --cref /work/ab0995/a270088/port/kpp_5yr_fix`
- **Validating a Kokkos-port PP run (only if explicit):**
  `--fref /scratch/a/a270088/fortran_pp_2yr --cref /work/ab0995/a270088/port/pp_2yr_rebase`

`scripts/m32_climate_compare.py` defaults to the KPP pair; override via the
`--fref` / `--cref` flags if needed.
