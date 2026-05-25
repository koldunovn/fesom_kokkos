# MFCT 3rd-order horizontal tracer advection port

**Why:** the C uses 2nd-order **central** as the high-order horizontal tracer flux
(`fesom_tracer_adv.c adv_tra_hor_central`), a documented interim simplification. CORE2
uses **MFCT** (`namelist.tra`: `MFCT, QR4C, FCT, 0., 1.`). Central drops the entire
`edge_up_dn_grad` 1/6 third-order reconstruction term → more diffusive at sharp tracer
fronts → the river-mouth SSS bias and the residual equatorial-cold-tongue SST bias
(diagnosed 2026-05-23; runoff/albedo/sw_pene already ruled out/fixed). User: port exactly,
no "equivalent" simplification.

**The MFCT flux (num_ord=0 for CORE2; oce_adv_tra_hor.F90:546-834):**
```
Tmean1 = ttf(n1) + (2(ttf(n2)-ttf(n1)) + edge_dxdy(1)*a*up_dn(1) + edge_dxdy(2)*r*up_dn(3))/6
Tmean2 = ttf(n2) - (2(ttf(n2)-ttf(n1)) + edge_dxdy(1)*a*up_dn(2) + edge_dxdy(2)*r*up_dn(4))/6
vflux  = (-VEL(2,el1)*dX1 + VEL(1,el1)*dY1)*helem(el1)   [el2: +VEL(2)*dX2 - VEL(1)*dY2]
cHO    = (vflux+|vflux|)*Tmean1 + (vflux-|vflux|)*Tmean2
flux  -= 0.5*(1-num_ord)*cHO + vflux*num_ord*0.5*(Tmean1+Tmean2)
```
a = r_earth*elem_cos(el1) (mean of both for interior edges). num_ord=0 → pure upwind-biased
3rd-order. Three depth bands (nu1..nu12-1 via el1; nu2..nu12-1 via el2; nu12..nl12 both),
matching the bidiff/upw1 level structure already in the C.

## Components (4)
1. **`edge_up_dn_tri[2*edge]`** (mesh, one-time): for each edge, the upwind & downwind
   triangle indices. Port `oce_muscl_adv.F90:185-343` — uses edge geometry to pick which of
   the surrounding triangles (beyond el1/el2) is up/down. Store in mesh; 0 = none (boundary).
2. **`tr_xy[2*nl*elem]`** (per step, per tracer): elemental tracer gradient.
   `tr_xy(1,nz,e)=Σ_k gradient_sca[0..2][k]*T(node_k,nz)`, `tr_xy(2,..)=Σ gradient_sca[3..5][k]*T`.
   (= the Fortran tracer_gradient_elements; gradient_sca already in the C mesh.)
3. **`edge_up_dn_grad[4*nl*edge]`** (per step, per tracer): `fill_up_dn_grad`
   (oce_muscl_adv.F90:356-525). Shared levels: copy tr_xy at the up/dn triangles. Non-shared
   (cavity/bathy) levels: area-weighted mean of tr_xy over the node's surrounding triangles.
4. **`adv_tra_hor_mfct`** (per step, per tracer): the flux above — REPLACES adv_tra_hor_central
   in the FCT high-order step (fesom_tracer_adv.c:1054). Keep the LO upwind + Zalesak limiter.

## Wiring
- Mesh setup: build edge_up_dn_tri once (after the edge/elem topology is ready).
- Per tracer per step (inside fesom_tracer_advect_one_fct): compute tr_xy → fill_up_dn_grad →
  adv_tra_hor_mfct (HO flux), exchange tr_xy/edge_up_dn_grad as Fortran does.
- Config: tracer hor adv = MFCT, num_ord=0 (CORE2 namelist.tra).

## Validation
- The river-mouth SSS hotspots + the residual equatorial SST should shrink toward Fortran.
- Re-run dt=1800 1yr + compare maps (surf_diff_maps) and profiles vs fortran_pp_2yr.
- Cross-check: FCT still monotone (no new over/undershoots); rest-state (UV=0) unchanged.
