# M6 reference namelists

Vendored 2026-07-12 (M6 Task 0.2) from the C oracle tree
`/home/a/a270088/port2/fesom2_port_zstar/docs/{tke,mevp,zstar}_reference_namelists/`.

These are the **Fortran** namelists of the runs that produced the archived reference output
on `/work/ab0995/a270088/port/`. The Kokkos port takes its config from CLI args + env knobs,
not namelists — so nothing here is read at runtime. They are here to (a) document exactly
what "the TKE config" / "the mEVP config" / "the zstar config" means, and (b) be the source
of truth for the constant tables transcribed into the port.

Each subdirectory keeps its upstream `PROVENANCE.md` (SLURM job ids, work dirs, binary
sha256s, and the echo-verified namelist values). Do not edit these files — they are a record.

## The single-knob guarantee

Every feature config is a one-line diff from the same linfs+KPP baseline
(`work_linfs_2yr_b` → `/work/.../zstar/fortran_linfs_2yr_b`), apart from `ResultPath`:

| dir | the one knob | file |
|---|---|---|
| `tke/` | `mix_scheme = 'KPP'` → `'cvmix_TKE'` | `namelist.oce` |
| `mevp/` | `whichEVP = 0` → `1` | `namelist.ice` |
| `zstar/` | `which_ALE = 'linfs'` → `'zstar'` | `namelist.config` |

Re-verified 2026-07-12. This is what makes each archived reference a clean attribution of
the feature — and what the port's env knobs (`FESOM_MIX_SCHEME` / `FESOM_WHICH_EVP` /
`FESOM_ALE`) must reproduce exactly.

## Verified constant tables

Transcribed and cross-checked against the C oracle's source in
`docs/plans/20260712-m6-options-tke-mevp-zstar.md` (Task 0.2). Two worth flagging here:

- **`tke_cd = 3.75`** — the NAMELIST value. The Fortran module default is 1.0, and it loses.
  A port that picks up the module default would run subtly different surface TKE forcing.
- **`ice_ave_steps = 1`** ⇒ `ice_dt` = ocean dt ⇒ mEVP's `rdt` is the **full** step (no 0.5).
