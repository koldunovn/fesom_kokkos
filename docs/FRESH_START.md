# FESOM2 C Port — Fresh Start Bootstrap Document

This document contains **everything** needed to start the FESOM2 → C port from
scratch. Read it fully before writing any code.

## Golden Rule

**DO NOT simplify, approximate, or "improve" the Fortran code when porting.**
Copy it line-by-line. Every shortcut attempted has caused bugs. If the Fortran
has `area/areasvol` ratios, include them. If the Fortran uses a lookup table,
port the lookup table. If the Fortran has tests for `nzmin` (cavity), include
them even if we don't use cavities. Trust the Fortran.

---

## 1. What FESOM2 Is

FESOM2 is an **unstructured-grid finite-volume ocean model**. It uses:
- Triangular horizontal mesh (nodes + triangles/elements)
- Z-level vertical discretization with **ALE** (Arbitrary Lagrangian-Eulerian)
- Split between **baroclinic** (3D, slow) and **barotropic** (2D, fast) modes
- **MPI parallelization** from the start (serial mode is MPI with 1 rank)

### Source code layout
- Fortran source: `/home/a/a270088/port2/fesom2/src/`  (~73K lines total)
- C port target: `/home/a/a270088/port2/fesom2_port/src/`

### Key Fortran files (read in this order)
1. `fesom_main.F90` — program entry point
2. `fesom_module.F90` — main driver, init + timestep loop
3. `oce_setup_step.F90` — model setup, namelist reading
4. `gen_modules_read_mesh.F90` — mesh I/O
5. `oce_ale.F90` — ALE timestep driver (`oce_timestep_ale`), ALE thickness updates
6. `oce_ale_tracer.F90` — tracer advection/diffusion driver (`solve_tracers_ale`)
7. `oce_ale_pressure_bv.F90` — EOS, pressure, Brunt-Väisälä frequency
8. `oce_ale_vel_rhs.F90` — momentum RHS (Coriolis, PGF, advection)
9. `oce_dyn.F90` — velocity update after SSH solve
10. `oce_ale_mixing_pp.F90` — PP vertical mixing
11. `oce_ale_mixing_kpp.F90` — FESOM1.4 KPP (what CORE2 actually uses)
12. `oce_adv_tra_driver.F90` + `oce_adv_tra_fct.F90` — FCT tracer advection

---

## 2. Mesh Structure

### Horizontal grid
- **Nodes**: where scalars live (T, S, SSH, pressure)
- **Elements** (triangles): where velocities live (u, v as 2D horizontal)
- **Edges**: between pairs of nodes; each edge has 1-2 adjacent elements

**CRITICAL**: Tracers at NODES. Velocities at ELEMENTS. This is unusual and
catches people out. Interpolation is needed everywhere they interact.

### Vertical grid
- **Mid-layer depths** (Z): where T, S, horizontal velocity u/v are defined
- **Interfaces** (zbar): where vertical velocity w, diffusivity Kv, N² are defined
- Layer `nz` extends from zbar[nz] (upper interface) to zbar[nz+1] (lower interface)
- Z[nz] is the midpoint of layer nz
- Layers are numbered 1-based in Fortran (nz=1 is surface), we need 0-based in C

### Mesh topology arrays (Fortran names, 1-based)
- `coord_nod2D(2, nod2D)`: (x, y) node positions in radians on the rotated grid
- `elem2D_nodes(3, elem2D)`: 3 nodes of each triangle (CCW in Fortran, but set to CW in orientation check)
- `edges(2, edge2D)`: 2 endpoints of each edge
- `edge_tri(2, edge2D)`: 2 elements adjacent to each edge (-1 if boundary)
- `elem_edges(3, elem2D)`: 3 edges of each triangle
- `nod_in_elem2D(num, nod2D)`: elements surrounding each node (variable count)

### Vertical topology
- `nl`: max number of levels globally (CORE2: 48, pi: ~23)
- `nlevels_nod2D(nod)`: number of levels at each node (**K_v^+**, MAX over surrounding elements)
- `nlevels_nod2D_min(nod)`: MIN over surrounding elements (**K_v^-**, used in ALE)
- `nlevels(elem)`: number of levels at each element
- `ulevels_nod2D(nod)`: upper level (=1 without cavities, >1 with cavities)
- `ulevels(elem)`: same for elements

**GOTCHA**: nlevels_nod2D is the MAX over surrounding elements for tracers/W,
but the MIN (nlevels_nod2D_min) is used as the ALE deformation limit. Without
K_v^-, thin layers can form under bathymetry changes and cause blowups.

### Mesh files (in mesh_dir)
- `nod2d.out`: `nod2D / lon_deg lat_deg coast_flag` (first line: count)
- `elem2d.out`: `elem2D / n1 n2 n3` (1-based node IDs)
- `aux3d.out`: `nl / zbar(1) / zbar(2) / ... / zbar(nl)` then for each node: `depth` then `nlevels_nod2D`
- `nlvls.out`: `nlevels_nod2D` per node (one per line)
- `elvls.out`: `nlevels(elem)` per element
- `edges.out`: `n1 n2` per edge
- `edge_tri.out`: `el1 el2` per edge (possibly negative for boundary)
- `edgenum.out`: global edge ID per local edge (for MPI)

**IMPORTANT**: Some meshes are **rotated** (grid north pole moved out of geographic
coordinates), some are **unrotated**. Detect by checking if any Arctic nodes (lat>80°)
exist: if yes, mesh is unrotated and needs rotation applied (to use in computations). FESOM default grid is
rotated with pole at (lon=50, lat=15) — `alpha=50, beta=15, gamma=-90` in the namelist.
Almost all computations are performed in rotated coordinates!

### Mesh partitioning (for MPI)
In `mesh_dir/dist_{N}/`:
- `rpart.out`: offsets (N+1 ints, N = number of ranks)
- `my_list{RRRRR}.out`: global IDs owned by rank R (5-digit)
- `com_info{RRRRR}.out`: halo communication info (neighbors, packing lists)

Available partitions for CORE2: `dist_16, 32, 144, 256, 288, 432, 512, ...`
Available for pi: `dist_2, dist_8`.

---

## 3. Data Paths on Levante

- **Fortran source**: `/home/a/a270088/port/fesom2/src/`
- **CORE2 mesh with partitions**: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`
- **Pi mesh (test)**: `/home/a/a270088/port/fesom2/test/meshes/pi`
- **PHC initial conditions**: `/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc`
- **JRA55 forcing**: `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0`
- **Fortran reference run setup**: `/home/a/a270088/fesom27/fesom2/work_core/` (namelist.oce, namelist.dyn, namelist.forcing)

---

## 4. Mesh Size Comparison: Pi vs CORE2

| Property         | pi     | CORE2   |
|------------------|--------|---------|
| nod2D            | 3,140  | 126,858 |
| elem2D           | 5,839  | 244,659 |
| nl (levels)      | ~23    | 48      |
| Resolution       | ~500km | ~100km  |
| Rotated?         | Yes    | Yes     |
| dt (dynamics)    | 100s   | 500s    |
| Partitions       | 2, 8   | 16, 32, 144, 256, 288, 432, 512 |

### Differences that trip up a porter
1. **Rotation**: Pi and CORE2 fully rotated. 
3. **CORE2 element orientation** needs to be corrected: Fortran swaps elements to
   CW (test_tri check); out of 244659 CORE2 elements, ~244654 get swapped.
4. **CORE2 bathymetry** has very deep trenches (Kuril, Aleutian) where velocity
   blowup has historically occurred. Watch Aleutian Trench (lat 51.77°N, lon 172.78°E,
   depth 6000m, element 194724 in global numbering) — every blowup we had hit there.
5. **CORE2 time step**: practical dt ≤ 500-600s until sea ice is on. Fortran uses
   dt=2700s with ice. Shelf gravity-wave CFL limits dt at ~100m depth to sqrt(gh)=31m/s.

---

## 5. Timestep Sequence (Baroclinic, CG SSH Solver)

This is the order of operations in `oce_timestep_ale` (oce_ale.F90:~2800). **Copy
this order exactly.**

```
 1. compute_pressure_bv        — EOS, density, pressure, N²,  dbsfc (for KPP)
 1b. compute_sw_alpha_beta     — α, β from JM-EOS (needed by KPP + GM)
 1c. compute_sigma_xy + neutral_slope   (only if GM/Redi on)
 2. compute_pressure_force     — PGF at elements
 3. compute_mixing             — Kv, Av (PP) or full KPP (Kv[T], Kv[S], Av, ghats, blmc)
 4. compute_vel_rhs            — Coriolis(AB2) + PGF → uv_rhs
 4b. momentum_advection        — scalar control volume advection (after Coriolis)
 5. viscosity_filter           — horizontal viscosity (backscatter)
 6. impl_vert_visc             — implicit vertical viscosity + wind/drag (TDMA)
 7. compute_ssh_rhs            — -α * div(∫(U+dU)h dz) + (1-α) * ssh_rhs_old
 8. ssh_solve (CG)             — solve for d_eta
 9. update_vel                 — uv += uv_rhs + F(d_eta)
10. compute_hbar               — hbar from transport divergence, save ssh_rhs_old
11. eta_n update               — eta_n = α*hbar + (1-α)*hbar_old
12. ALE step                   — compute hnode_new, update_thickness, compute w
13. GM/Redi bolus velocity     — add bolus to UV, W before advection
14. solve_tracers_ale          — advection (FCT) → diff (with ALE reconstruction) → impl TDMA
    — Remove bolus velocity after advection (before diffusion)
15. update_thickness_ale       — hnode = hnode_new (commit)
```

### Inside solve_tracers_ale / diff_tracers_ale
Fortran order (lines 386-482 in oce_ale_tracer.F90):
```
a. diff_part_hor_redi(tracers)         → writes to del_ttf (NOT to tr directly)
b. diff_ver_part_expl_ale              → writes to del_ttf (only when i_vert_diff=false)
c. diff_ver_part_redi_expl             → writes to del_ttf (only if Redi)
d. ALE reconstruction:
     del_ttf += T * (hnode - hnode_new)
     T       = T + del_ttf / hnode_new
   (equivalent to: T_new = (T_old*hnode + del_ttf) / hnode_new)
e. diff_ver_part_impl_ale              → TDMA on reconstructed T (only when i_vert_diff=true)
```

**CRITICAL**:
- With PP mixing: `i_vert_diff = .false.` → explicit path, surface flux ÷ areasvol ≈ 0 (essentially not applied)
- With KPP mixing: `i_vert_diff = .true.`  → implicit TDMA, surface flux applied via `bc_surface`, KPP nonlocal redistributes
- `del_ttf` is **zeroed in `init_tracers_AB`** at the start of each timestep

---

## 6. ALE Framework

FESOM2 supports three ALE modes (`which_ALE` in namelist):

### zlevel (CORE2 default)
- Only surface layer thickness changes: `hnode_new[0] = hnode[0] + dhbar`
- Interior layers: `hnode_new[nz] = hnode[nz]`
- Simple, fast, standard for ocean modeling

### zstar
- All layers scale proportionally with SSH: `hnode_new[nz] = hnode[nz] * (1 + eta/H)`
- Better for large tide ranges, but more expensive

### linfs (linear free surface)
- `hnode_new = hnode` everywhere (SSH doesn't change layer thicknesses)
- The simplest case — **start here for debugging, everything else should be implemented much later**
- Tracers don't see ALE mass correction (it's zero)

### Local-zstar fallback
Even in zlevel, if dhbar would make the top layer too thin, fall back to zstar
for the top few layers (`lzs` layers). This matters on shallow shelves. But again - use linfs in the beggining, as it is most stable and simples option.

### Key ALE arrays
- `hnode(nl, nod)`: current layer thickness at nodes
- `hnode_new(nl, nod)`: thickness after SSH update
- `helem(nl, elem)`: layer thickness at elements (avg of node thicknesses)
- `zbar_3d_n(nl, nod)`: 3D interface depths (negative, from surface=0 down)
- `Z_3d_n(nl, nod)`: 3D mid-layer depths

**GOTCHA**: `helem` is the ELEMENT layer thickness (from the 3 node thicknesses).
`hnode_new` and `hnode` are the NODE layer thicknesses. Don't mix them.

---

## 7. Key Data Structures (fesom_aux equivalents)

### Scalar fields at NODES [nod2D * nl]
- `T, S` (tracers)
- `density` (in-situ - ρ_ref)
- `pressure`
- `bvfreq` (N²)
- `Kv` (vertical diffusivity — **per tracer in KPP**, so Kv_T and Kv_S)
- `w` (vertical velocity)
- `hnode, hnode_new`
- `areasvol` (scalar control volume area, per level — but same as 2D area for zlevel)

### Scalar fields at ELEMENTS [elem2D * nl]
- `Av` (vertical viscosity)
- `helem`
- `pgf_x, pgf_y` (PGF components)

### Vector fields at ELEMENTS [elem2D * nl * 2]
- `uv` (u, v velocity)
- `uv_rhs`, `uv_rhsAB`, `uv_rhsAB2` (AB2 momentum RHS history)

### SSH fields at NODES [nod2D]
- `eta_n, d_eta`
- `ssh_rhs, ssh_rhs_old`
- `hbar, hbar_old`

### Edge-based [edge2D]
- `edge_dxdy[4]`: edge dx/dy components
- `edge_cross_dxdy[4]`: cross-edge components (dx1, dy1, dx2, dy2) for the
  two adjacent elements — used in edge-based flux computations

---

## 8. Equation of State

FESOM2 uses the **Jackett-McDougall (JM, UNESCO/EOS80)** equation of state:

```
rho_insitu(T, S, z) = bulk_modulus * rhopot / (bulk_modulus + 0.1*z)
```

where `bulk_modulus = bulk_0(T,S) + z*(bulk_pz(T,S) + z*bulk_pz2(T,S))`.

- `rhopot(T, S)`: potential density (no pressure dependence)
- `bulk_0, bulk_pz, bulk_pz2`: 3 polynomial coefficients in T, S

The JM polynomial is ~20 coefficients. See `oce_ale_pressure_bv.F90`
subroutine `densityJM_components`.

**Use this, not linearized α/β.** 

For buoyancy differences (KPP), evaluate BOTH parcels at the SAME depth using
the JM components — this cancels compressibility:
```
rho_surf = eos_jm_at_depth(T_surf, S_surf, z)
rho_nz   = eos_jm_at_depth(T_nz, S_nz, z)
dbsfc    = -g * (rho_surf - rho_nz) / rho_nz  [FESOM1.4 formula]
```

---

## 9. Surface Forcing (CORE2 / JRA55)

Required forcing inputs (at nodes):
- `heat_flux[nod]`: net surface heat flux (W/m², **positive = ocean loses heat**)
- `water_flux[nod]`: E-P freshwater flux (m/s, **positive = ocean loses freshwater**)
- `stress_surf[elem*2]` or `stress_node_surf[nod*2]`: wind stress (N/m²)

### JRA55-do forcing path
`/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/{uas,vas,tas,huss,rsds,rlds,prra,prsn}.{YEAR}.nc`
at 320 lat × 640 lon, 3-hourly. Reader does bilinear interp to mesh nodes,
time-interpolates to model time, then applies L&Y09 bulk formulae to get heat/water fluxes.

### SSS restoring (CORE2 convention)
`/pool/data/AWICM/FESOM2/FORCING/PHC2_salx.nc` — monthly SSS climatology.
Apply as ADDITIVE virtual freshwater flux: `water_flux += (S - S_clim) * piston_vel`.
CORE2 piston velocity: `1.929e-6 m/s` (≈ 50m / year).

### Runoff (CORE2)
`/pool/data/AWICM/FESOM2/FORCING/CORE2_runoff.nc` — single-record annual climatology.
Apply as freshwater source: `water_flux -= runoff`.

---

## 10. Vertical Mixing Schemes

### PP (Pacanowski-Philander) — simplest
```
factor = shear² / (shear² + 5*max(N², 0) + eps)
Kv = mix_coeff * factor³ + K_ver_background
Av = mix_coeff * factor² + A_ver_background    (averaged to elements)
```
Plus convective adjustment: where N² < 0, set Kv = max(Kv, 0.1 m²/s).

### KPP (mix_scheme='KPP', mix_scheme_nmb=1) - but it's complicated, don't use for now

**SOURCE FILE**: `oce_ale_mixing_kpp.F90` (1185 lines, self-contained, NO CVMix).
NOT `gen_modules_cvmix_kpp.F90` (that's the CVMix wrapper, mix_scheme_nmb=3).

Structure:
1. `oce_mixing_kpp_init` — build wm/ws lookup tables (892×482), constants Vtc, cg
2. `oce_mixing_kpp` (main driver):
   - Compute dVsq (velocity shear² w.r.t. surface)
   - Compute ustar, Bo using **sw_alpha, sw_beta from JM-EOS**
   - Call ri_iwmix (interior Ri-dependent mixing, formula (1-(Ri/0.8)²)³)
   - Call bldepth (OBL depth from bulk Ri with Ekman/MO limits)
   - Call blmix_kpp (shape-matched OBL mixing + ghats nonlocal)
   - Call enhance (diffusivity boost at kbl-1)
   - Smooth blmc (3 passes of horizontal smoothing)
   - Combine: within OBL, Kv = max(interior, blmc)
   - Halo exchange Kv, ghats
   - Average viscA node → element with minmix=3e-3 in surface layer

### KPP nonlocal transport (tracer equation)
In `oce_ale_tracer.F90` lines 900-942 (mix_scheme_nmb==1 path), for temperature:
```
Surface: tr(1) += -MIN(ghats(2)*blmc(2,T), 1.0) * (area(2)/areasvol(1)) * heat_flux/vcpw * dt
Middle:  tr(nz)+= (MIN(ghats(nz)*blmc(nz,T), 1) * area/areasvol
                - MIN(ghats(nz+1)*blmc(nz+1,T), 1) * area/areasvol) * heat_flux/vcpw * dt
Bottom:  tr(N) += MIN(ghats(N)*blmc(N,T), 1.0) * (area/areasvol) * heat_flux/vcpw * dt
```
- `ghats(nz, nod)`: nonlocal transport coefficient, units s/m²
- `blmc(nz, nod, 3)`: BL mixing coeffs, [momentum, T, S] at index [1, 2, 3]. For the
  tracer equation, use blmc(:,:,2) for T and blmc(:,:,3) for S.
- Product ghats*blmc is dimensionless, capped at 1.0
- For zlevel without cavities, area/areasvol = 1.0 but **include the ratio anyway**

For salinity: same structure with blmc(:,:,3) and `-rsss*water_flux*dt` (minus sign).

---

## 11. SSH Solver (CG path)

The SSH equation is a 2D Helmholtz-like system solved every step via
**Conjugate Gradient (CG)**:

```
M * d_eta = ssh_rhs
```

where M = `area * (1 + g*dt²*α² * Δ_discrete)` (a stiffness matrix).

### Stiffness matrix construction (element-based Galerkin)
In `oce_ale.F90` subroutine `update_stiff_mat_ale` (computed EACH STEP because
`hbar` changes). Element loop:
- For each element, compute local 3×3 stiffness from element gradients
- Factor: **NEGATIVE** (-g*dt*α*hbar). Got this wrong in an earlier attempt → blowup.
- Scatter to CSR matrix at nodes

**Orientation matters**: the element nodes must be in a consistent handedness (Fortran
normalizes to CW via `test_tri` check). Without this, ~half the stiffness terms have
wrong sign.

### CG preconditioner
Diagonal (Jacobi): `pr = 1 / diag(M)`. Halo must be exchanged before building
(so diagonals on halo nodes are correct).

### Convergence
Typical 10-50 iterations with relative residual tolerance 1e-5.

---

## 12. FCT Tracer Advection

Flux-Corrected Transport (Zalesak). In `oce_adv_tra_fct.F90` and `oce_adv_tra_driver.F90`.

Steps per tracer:
1. Compute low-order (upwind) solution `fct_LO` using `adv_flux_hor`, `adv_flux_ver`.
2. Compute antidiffusive flux = high-order flux - low-order flux.
3. Compute Zalesak limiters (fct_plus, fct_minus) using local min/max bounds.
4. Apply limited antidiffusive flux.
5. Divide flux divergence by areasvol to get tendency, then by hnode_new for concentration.

**Sign convention gotcha**: `edge_vflux` (vertical advective flux) — had a sign error
in a previous C port that made tracer advection reversed. Took 80 steps before blowup.
**Verify the sign carefully** by running a constant advection test.

---

## 13. Partition File Layout (MPI)

For N ranks, `mesh_dir/dist_N/` contains:
- `rpart.out`: N+1 integers, rank offsets into node list
- `my_list{RRRRR}.out` (one per rank, 5-digit zero-padded):
  - First line: `myDim_nod2D eDim_nod2D` (owned + halo nodes)
  - Then myDim+eDim global node IDs (1-based)
- `com_info{RRRRR}.out`: halo exchange schedule
  - `com_nod2D.rPEnum`: number of remote PEs we receive from
  - `com_nod2D.rPE(:)`: their rank IDs
  - `com_nod2D.rptr(:)`: offsets into rlist
  - `com_nod2D.rlist(:)`: local node IDs to receive from each rPE
  - `com_nod2D.sPE(:), sptr, slist`: same for sending
  - Then same 6 sections for elements

**Halo exchange pattern**: `fesom_exchange_nod(field, nlev, mesh)` packs `slist`
into send buffers, does MPI_Isend/Irecv, unpacks to positions in `rlist`.

**GOTCHA — allocate for full local size, not owned**: Arrays must be allocated
for `myDim + eDim` (owned + halo), not just `myDim`. Scatter loops iterate over
myDim only, but reads (and halo exchanges) use indices up to myDim+eDim-1.

---

## 14. Critical Gotchas Discovered During Previous Ports

### 14.1 Tracer diffusion
- The Fortran `diff_tracers_ale` accumulates `del_ttf` from horizontal + explicit
  vertical diffusion, then does ALE reconstruction `T_new = (T*hnode + del_ttf)/hnode_new`,
  THEN runs the implicit TDMA. Do NOT apply horizontal diffusion directly to T.
- For zlevel, `hnode == hnode_new` except surface layer, so the mass correction is
  only nonzero at the surface.

### 14.2 TDMA coefficients
- Include `area/areasvol` ratios even for zlevel (where they = 1.0). Matches Fortran.
- For the w_i (implicit vertical advection) terms: `zinv = dt * area/areasvol = dt`
  for zlevel, NOT `dt/areasvol`. Dividing by areasvol imprints mesh geometry on tracer.

### 14.3 EOS
- Use full JM-EOS always. Linearized α/β breaks OBL depth (stuck at 5m).
- For KPP dbsfc, evaluate both parcels at SAME depth (cancels compressibility).

### 14.4 AB2 Coriolis
- Use single-slot AB history for `AB_order=2`. A previous port used AB_order=3 shift
  but AB_order=2 coefficients → Coriolis weight 2.5 instead of 1.0 → SSH runaway.

### 14.5 Edge vertical advective flux
- Sign of `edge_vflux` in FCT was reversed in a previous port. Check against Fortran.

### 14.6 No artificial bounds
- Don't add arbitrary caps (e.g., "OBL depth ≤ 500m") if the Fortran doesn't have them.
- The Fortran's OBL bounds come from Ekman/Monin-Obukhov limits inside the CVMix
  library for mix_scheme_nmb=3, or inline Ekman/MO in the FESOM1.4 version (mix_scheme_nmb=1).

### 14.7 Default parameters (CORE2 from namelists)
From `work_core/namelist.oce`, `namelist.dyn`, `namelist.tra`:
```
mix_scheme         = 'KPP'     (→ mix_scheme_nmb = 1, FESOM1.4 KPP, but we will use PP)
state_equation     = 1         (full JM-EOS)
which_ALE          = 'zlevel', but we will use linfs
use_density_ref    = .false.   (density_ref = density_0 = 1030 everywhere)
pressure_force     = 'Shchepetkin'  (equivalent to linfs_fullcell in zlevel+full cells)
i_vert_diff        = .true.    (implicit TDMA)  — BUT only effective for KPP
K_hor              = 0.0       (no explicit horizontal diffusion; Redi Ki is the smoother)
K_ver              = 1e-5
A_ver              = 1e-4
opt_visc           = 5         (harmonic + backscatter)
visc_gamma0        = 0.003     (not 0.03!)
use_wsplit         = .true.    wsplit_maxcfl = 1.0
momadv_opt         = 2         (scalar control volume)
AB_order           = 2
Fer_GM             = .true.    K_GM_max = 1000, K_GM_min = 2
Redi               = .true.    Redi_Kmax = 1000
C_d                = 0.0025
```

### 14.8 CORE2 blowup locations (historical)
- **Aleutian Trench**: lat 51.77°N, lon 172.78°E, depth 6000m, global element 194724.
  Every CORE2 blowup we've had hits this element.
- Caused by mix of: edge_vflux sign error, bad IC at deep layers, or unbounded PGF.

### 14.9 Forcing smoothness
- JRA55 heat_flux on mesh has neighbor jumps up to ~290 W/m². It LOOKS noisy but
  the pattern is smooth when averaged over a day. This is NOT the source of the mosaic.

### 14.10 The mosaic pattern
- Known to appear in C runs without sea ice, even with KPP. Root cause never fully
  resolved in earlier attempts. Suspected: KPP OBL depth wrong → flux concentrated
  in top 5m → node-scale differences in TDMA response → areasvol-like pattern.
- Fortran doesn't show it.
---

## 15. Validation Strategy

### Pi mesh quick test (2PE, ~1 min)
```bash
./test_mpi_integration $PI_MESH $PHC 100 100 0 "" /tmp/pi_out
```
Should run 100 steps at dt=100, stable, max_vel ~ 0.3 m/s.

### CORE2 1-day test (72PE, ~3 min)
```bash
mpirun -np 72 ./test_mpi_integration $CORE2 $PHC 172 500 0 $JRA55 /tmp/core2_1day
```
Should run 172 steps ≈ 1 day at dt=500, stable.

### CORE2 10-day test (72PE, ~8 min) — integration test
```bash
mpirun -np 72 ./test_mpi_integration $CORE2 $PHC 1728 500 500 $JRA55 /tmp/core2_10d
```
1728 steps at dt=500 = 10 days. Output every 500 steps for visualization.

### Sanity checks after each major change
1. Pi regression passes (20 steps, dt=100, no blowup)
2. CORE2 1-day doesn't blowup
3. SST range is physical (no < -2°C, no > 35°C without ice)
4. SSH range < 5m
5. Max velocity < 3 m/s in geostrophic adjustment

---

## 16. Minimum Viable Implementation Order (suggested)

Start simple. **Get it working end-to-end first, then add complexity.** ALLWAYS TRUST FORTRAN CODE - IT WORKS AND STABLE, YOUR TASK IS TO REPRODUCE IT!

### Phase 1: Minimum model (target: pi mesh stable for 100 steps)
1. **Mesh I/O**: read nod2d, elem2d, edges, nlvls. Compute areas, edge dx/dy, gradient_sca.
2. **State allocation**: fesom_mesh, fesom_dyn, fesom_aux, fesom_tracers, fesom_forcing
3. **Initial conditions**: constant T=10, S=35 for starters
4. **EOS + pressure**: JM-EOS, hydrostatic pressure (zero SSH)
5. **PGF**: basic fullcell at elements
6. **Linear free surface (linfs)**: `hnode_new = hnode` always. SKIP all ALE complications.
7. **CG SSH solver**: element-based Galerkin, diagonal preconditioner
8. **Momentum**: Coriolis (AB2), PGF, explicit vertical viscosity, bottom drag
9. **Barotropic mode**: use the CG path, no split-explicit yet
10. **Tracer advection**: simple upwind (no FCT yet)
11. **PP mixing**: simple, well-tested
12. **Timestep driver**: wire it all together
13. **Zero forcing**: test with no wind, no heat flux — should stay at rest

### Phase 2: Make pi work (target: pi 1000 steps stable)
14. **FCT advection**: proper high-order + Zalesak limiter. Watch `edge_vflux` sign.
15. **Horizontal viscosity**: opt_visc=5 with backscatter
16. **wsplit**: implicit/explicit vertical velocity splitting
17. **Analytical forcing**: constant wind stress, SST restoring. Should see gyre circulation.
18. **Pi regression**: 1000 steps, dt=100 must not blow up

### Phase 3: Make CORE2 work (target: CORE2 1-day stable with JRA55)
19. **MPI partitioning**: read my_list, com_info, halo exchange
20. **Parallel CG solver**: halo exchange for diagonal, matrix-vector, residual
21. **IC from PHC**: bilinear interp + extrap_nod + vertical fill
22. **CORE2 mesh specifics**: rotation auto-detect, CW orientation, partial cells
23. **zlevel ALE**: hnode_new = hnode + dhbar at surface. Track K_v^- for stability.
24. **JRA55 reader**: bilinear interp to mesh, time interp, L&Y09 bulk formulae
25. **SSS restoring + runoff**: additive on top of bulk fluxes
26. **CORE2 1-day**: 172 steps at dt=500 must be stable

### Phase 4: Long-term stability (target: CORE2 10-day with KPP + ice)
27. **GM/Redi**: neutral slopes, tapering, bolus velocity
28. **KPP mixing**: full FESOM1.4 KPP with lookup tables, ghats, blmc
29. **Nonlocal transport**: in tracer TDMA, using ghats*blmc
30. **Sea ice** (EVP dynamics + FCT + thermo): ice_ocean_drag, salt/heat fluxes
31. **CORE2 10-day**: 1728 steps at dt=500, no blowup, reasonable SST

---

## 17. Key Constants

```c
#define FESOM_PI        3.14159265358979
#define FESOM_RAD       (FESOM_PI / 180.0)
#define FESOM_DENSITY_0 1030.0       /* reference density, kg/m³ */
#define FESOM_G         9.81         /* m/s² */
#define FESOM_R_EARTH   6367500.0    /* m */
#define FESOM_OMEGA     (2*FESOM_PI / 86400.0)  /* earth rotation, rad/s */
#define FESOM_VCPW      4.2e6        /* volumetric heat capacity of seawater, J/(m³·K) */
```

---

## 18. Suggested Array Layout (C, SOA)

Array indexing convention (0-based in C, 1-based in Fortran):
- 2D fields: `array[node]` or `array[elem]`
- 3D fields at nodes: `array[node * nl + level]`  (vertical columns contiguous for cache)
- 3D fields at elements: `array[elem * nl + level]`
- Element-local: `elem_nodes[3 * elem + local_node]`
- Edge-local: `edges[2 * edge + local_node]`
- Vector components on elements: `uv[elem * nl * 2 + level * 2 + component]`

---

## 19. Fortran→C Mapping Cheat Sheet

| Fortran | C | Notes |
|---------|---|-------|
| `1:nl` | `0..nl-1` | 1-based → 0-based |
| `nlevels_nod2D(n)` | `mesh->nlevels_nod[n]` | number of levels |
| `hnode(nz, n)` | `mesh->hnode[n * nl + nz]` | column-major → row-major |
| `UVnode(1, nz, n)` | `dyn->uvnode[n*nl*2 + nz*2 + 0]` | vector components |
| `tracers%data(1)%values(nz,n)` | `tra->data[0].values[n*nl + nz]` | tracer 0 = T |
| `mype` | `fesom_mpi_rank()` | |
| `myDim_nod2D` | `mesh->myDim_nod2D` | owned nodes |
| `eDim_nod2D` | `mesh->eDim_nod2D` | halo nodes |
| `exchange_nod(f)` | `fesom_exchange_nod(f, nlev, mesh)` | halo exchange |
| `call par_ex` | `MPI_Abort` | |

---

## 20. Key Things To Test Immediately

After writing the mesh reader:
1. Print nod2D, elem2D, nl → check against expected (pi: 3140/5839/~23; CORE2: 126858/244659/48).
2. Print first 3 nodes' coords → should be lat/lon in degrees.
3. Print first 3 elements' nodes → 1-based indexing in file, convert to 0-based in memory.
4. Check that `elem_nodes` are all in `[0, nod2D)`.
5. Check that `edge_tri` has at most 2 nonzero values (elements 1 and 2), can be negative (boundary).
6. Compute `elem_area` for all elements, check min/max/mean are reasonable (meters²).

After writing PGF + dynamics:
1. Rest-state test: T=const, S=const, eta_n=0, uv=0 → should stay at rest indefinitely (within machine precision).
2. SSH initial bump → should propagate as gravity wave at sqrt(gH).

After writing the full tracer step:
1. Constant tracer field → should remain constant under advection.
2. Pure diffusion (no velocity) → should smooth Gaussian blob without distortion.

---

## 21. Things That Were Tried And Failed

- **Simplifying ghats**: using a shape function G(σ)=(1-σ)² instead of the full
  lookup-table-based wm/ws → wrong magnitudes, didn't fix mosaic.
- **Adding 500m cap on OBL depth**: not in Fortran, caused wrong physics.
- **Linearizing α/β**: caused OBL stuck at 5m because compressibility wasn't cancelled.
- **Applying horizontal diffusion directly to T, then multiplying by hnode/hnode_new**:
  double-scales the diffusion tendency. Must accumulate into del_ttf first.
- **Using dt/areasvol instead of dt*area/areasvol in w_i TDMA terms**:
  imprinted mesh geometry on tracer evolution. For zlevel, ratio = 1 so coefficient is dt.
- **Starting from KPP**: too complex to debug alongside other physics. Start with PP.
- **CVMix KPP wrapper (gen_modules_cvmix_kpp.F90)**: CORE2 uses the ORIGINAL
  FESOM KPP (oce_ale_mixing_kpp.F90, mix_scheme_nmb=1). These are different.

---

## 22. Reference Fortran Run

To generate reference output for comparison:
```
cd /home/a/a270088/fesom27/fesom2/work_core
# Edit namelist.config: run_length = 1 (days), nstep_per_day = 32 (dt=2700)
# Submit via SLURM
sbatch job_fesom.sh
# Output in results/
```

Compare output fields (SST, SSS, SSH, T/S profiles at selected nodes) between
C run and Fortran run. Target: < 1% difference after 1 day.

---

## 23. Useful Diagnostics to Print Every Step

- `max |uv|` at elements (m/s)
- `max |eta_n|` at nodes (m)
- `max |w|` at nodes (m/s)
- `SST min/max`
- `SSS min/max`
- CG iterations
- If KPP: OBL depth min/max/mean, number of unstable nodes
- If anything > physical threshold, print location (node/elem ID, lat/lon)

---

## End Notes

This port has been attempted multiple times. The pattern: get excited about a
new approach, take shortcuts, hit bugs that took weeks to debug. **Don't do
that again.**

Read the Fortran. Port line-by-line. When in doubt, print Fortran intermediate
values at a specific node and compare against your C port. That's faster than
debugging a weeks-later crash.

Everything here is verified against the Fortran source as of April 2026.
