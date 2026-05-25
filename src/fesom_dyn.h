#ifndef FESOM_DYN_H
#define FESOM_DYN_H

#include "fesom_types.h"
#include "fesom_field.hpp"   // M1.3: DualView-backed storage for the persistent dynamics arrays

struct fesom_mesh;

/*
 * Phase 1 subset of Fortran t_dyn (MOD_DYN.F90:44+).
 *
 *   uv         [elem2D * nl * 2]   instant horizontal velocity (u, v)
 *   uv_rhs     [elem2D * nl * 2]   current momentum RHS
 *   uv_rhsAB   [elem2D * nl * 2]   AB2 history (single slot — see §14.4)
 *   w          [nod2D  * nl]       vertical velocity at scalar locations
 *   eta_n      [nod2D]             SSH at this step
 *   d_eta      [nod2D]             SSH increment
 *   ssh_rhs    [nod2D]             current SSH RHS
 *   ssh_rhs_old[nod2D]             previous SSH RHS (for AM2 blending)
 *
 * Deferred to later phases: fer_uv, fer_w, w_e, w_i, w_old, cfl_z, uvnode,
 * uvnode_rhs, all se_* split-explicit arrays, iceberg arrays.
 */
typedef struct fesom_dyn {
    real_t *uv;
    real_t *uv_rhs;
    real_t *uv_rhsAB;
    real_t *w;
    real_t *w_e;        /* explicit part of w (for tracer advection)         */
    real_t *w_i;        /* implicit part of w (for impl_vert_visc)
                           When CFL_z[nz,n] ≤ wsplit_maxcfl: w_e = w, w_i = 0
                           (compute_Wvel_split, oce_ale.F90:3026-3074).       */
    real_t *cfl_z;      /* [nod2D * nl] vertical CFL at interfaces            */
    real_t *eta_n;
    real_t *d_eta;
    real_t *ssh_rhs;
    real_t *ssh_rhs_old;

    real_t *uvnode;     /* [nod2D * nl * 2] u,v at NODES — interpolated from
                           element-center UV via area-weighted mean. Required
                           by PP mixing (Fortran tracers%data%uvnode is held
                           in dynamics%uvnode). */
    real_t *uvnode_rhs; /* [nod2D * nl * 2] momentum-advection RHS at NODES.
                           Mirror of dynamics%work%uvnode_rhs; written by
                           fesom_momentum_adv_scalar (momadv_opt=2). */

    /* GM bolus velocity. Allocated by fesom_dyn_alloc; written by
     * fesom_fer_gamma2vel (G4) for fer_uv and inside vert_vel_linfs (G6a)
     * for fer_w. Zero when GM is off. Mirror of dynamics%fer_uv / fer_w
     * (MOD_DYN.F90). */
    real_t *fer_uv;     /* [(myDim+eDim+eXDim) * nl * 2]   element-centered */
    real_t *fer_w;      /* [(myDim+eDim) * nl]             node-centered    */

    /* Work arrays for visc_filt_bcksct (opt_visc=5). Mirror of
       dynamics%work%u_b/v_b/u_c/v_c (MOD_DYN.F90 t_dyn_work). */
    real_t *u_b;        /* [elem2D * nl] per-element raw harmonic-visc update */
    real_t *v_b;
    real_t *u_c;        /* [nod2D  * nl] node-smoothed visc-update            */
    real_t *v_c;

    int AB_order;   /* fixed at 2 in Phase 1 */

    /* ===================== M1.3 · DualView data layer =========================
     * Each persistent array above is OWNED by the matching fesom::Field below; the
     * raw pointer is a NON-OWNING alias re-pointed at field.h() right after
     * field.alloc() in fesom_dyn_alloc (the same pattern proven on the 28 mesh
     * arrays in M1.2, D12). These arrays are time-evolving state (written every
     * step through the raw alias); all updates are in-place value writes/memcpy —
     * the alias is NEVER re-pointed by a buffer swap, so it stays valid for the
     * whole run. LayoutRight + host mirror == the C flat layout, so legacy
     * dyn->X[...] and the FESOM_*3D/ELEMVEC macros keep working unchanged → Serial
     * stays bit-identical. No device sync at M1: nothing reads these on device yet
     * (cf. the M1.2 mesh STATE arrays hnode/hbar, likewise not synced); the
     * step-driver sync discipline for this evolving state lands in M1.5/M2 (D15). */
    fesom::Field uv_fld, uv_rhs_fld, uv_rhsAB_fld;
    fesom::Field w_fld, w_e_fld, w_i_fld, cfl_z_fld;
    fesom::Field uvnode_fld, uvnode_rhs_fld;
    fesom::Field u_b_fld, v_b_fld, u_c_fld, v_c_fld;
    fesom::Field eta_n_fld, d_eta_fld, ssh_rhs_fld, ssh_rhs_old_fld;
    fesom::Field fer_uv_fld, fer_w_fld;
} fesom_dyn;

void fesom_dyn_alloc(fesom_dyn *d, const struct fesom_mesh *mesh);
void fesom_dyn_free (fesom_dyn *d);

#endif /* FESOM_DYN_H */
