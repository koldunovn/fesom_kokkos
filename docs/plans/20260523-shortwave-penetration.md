# Shortwave penetration port (use_sw_pene)

**Why:** the C lumps all surface heat (incl. shortwave) into the top layer; CORE2/Fortran
penetrates the visible shortwave through the upper column (`use_sw_pene=.true.`). This
causes the C's warm-surface / cold-subsurface bias (ΔT +0.23°C surface → −0.72°C @35m →
~0 @175m; salt unaffected). Diagnosed 2026-05-23 (see DT1800_HANDOFF / vertical_profiles.png).

**Mandate: faithful port of the Fortran, no invention/simplification/shortcut.**

## Fortran sources to port (verbatim behavior)
- `oce_shortwave_pene.F90` `cal_shortwave_rad` — builds `sw_3d` from chl (Sweeney 2005
  2-band exponential), removes the visible 0.54 fraction from `heat_flux`, skips cavity +
  under-ice nodes.
- `oce_ale_tracer.F90:990` — applies `tr(nz) += (sw_3d(nz,n) − sw_3d(nz+1,n)·area(nz+1)/
  areasvol(nz))·dt` for the temperature tracer (ID==1), linfs zinv=dt.
- chl read: `read_other_NetCDF(chl_file,'chl',i,chl,.true.,.true.)` monthly
  (gen_surface_forcing.F90:1585) — ALREADY ported in C as `fesom_read_other_NetCDF`
  (fesom_sss_runoff.c). const fallback: `chl = chl_const`.

## C plumbing (all already present/matching)
- `qsr = (1−albw)·shortwave` from the bulk (albw=0.066=Fortran LY2004) → `swsurf = qsr·0.54`.
- `FESOM_VCPW=4.2e6`, `mesh->zbar_3d_n`, `mesh->area`, `mesh->areasvol` — all match Fortran.
- a_ice = `ice->data[FESOM_ICE_AICE].values`; heat_flux = `forcing->heat_flux`.
- temperature RHS in `diff_ver_part_impl_ale` (fesom_tracer_diff.c), surface flux via `bc_surface`.

## Steps
1. **Arrays**: add `chl[nod2D+eDim]`, `sw_3d[(nod2D+eDim)*nl]` to forcing (alloc/free).
2. **`fesom_cal_shortwave_rad`** (new, in fesom_bulk.c or fesom_shortwave.c): port
   oce_shortwave_pene.F90 line-for-line — zero sw_3d; per surface node skip cavity/ice;
   swsurf=qsr·0.54; heat_flux+=swsurf; chl≥0.02; Sweeney v1/v2/sc1/sc2 polynomials;
   swsurf/=vcpw; sw_3d[nzmin]=swsurf; exp profile down the column with the aux<1e-5/bottom cutoff.
3. **chl source**: const (`chl=chl_const`, default 0.1) OR Sweeney via `fesom_read_other_NetCDF`
   with the monthly-update + month-lookahead logic (gen_surface_forcing.F90:1585). Default Sweeney.
4. **Apply** in `diff_ver_part_impl_ale` for id==1: add the sw_3d divergence to `tr[nz]` per layer.
5. **Config**: `use_sw_pene` (default on), `chl_data_source` (Sweeney/None), chl file path, `chl_const`.
6. **Wiring**: call `fesom_cal_shortwave_rad` after ice→ocean coupling, before the tracer
   diffusion that consumes heat_flux (so the surface BC sees the reduced heat_flux). Chl
   updated monthly in the forcing/calendar step.

## Validation
- Fortran no-pene run (job 25088587, use_sw_pene=.false.): C-current profile should match it.
- After the port, the C profile should match the with-pene reference (fortran_pp_2yr, Sweeney):
  warm-surface/cold-subsurface ΔT dipole should collapse toward 0. Re-run
  scripts/eps_vertical_profiles.py + the climate compare.
