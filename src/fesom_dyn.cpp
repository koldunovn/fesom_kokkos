#include "fesom_dyn.h"
#include "fesom_mesh.h"
#include "fesom_constants.h"

#include <stdlib.h>
#include <string.h>

void fesom_dyn_alloc(fesom_dyn *d, const struct fesom_mesh *mesh)
{
    // The struct holds fesom::Field members (DualView, non-trivial): a raw memset is UB (L13).
    // Value-initialise instead — zeros every POD and leaves each Field an empty DualView (D13).
    *d = fesom_dyn{};
    d->AB_order = FESOM_PHASE1_AB_ORDER;

    /* MPI: size for full local extent (myDim+eDim for nodes,
     * myDim+eDim+eXDim for elements). */
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int E = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    size_t e_nl_2 = (size_t)E * (size_t)mesh->nl * 2;
    size_t n_nl   = (size_t)N * (size_t)mesh->nl;
    size_t n      = (size_t)N;
    size_t e_nl   = (size_t)E * (size_t)mesh->nl;

    // M1.3: each Field owns its storage; the raw pointer is a non-owning alias = field.h()
    // (D12). .alloc(label, count) takes the count in ELEMENTS (== the old calloc count) and
    // zero-inits like calloc. Element counts are byte-for-byte the C layout → Serial bit-identical.
    d->uv_fld.alloc("dyn.uv", e_nl_2);                 d->uv          = d->uv_fld.h();
    d->uv_rhs_fld.alloc("dyn.uv_rhs", e_nl_2);         d->uv_rhs      = d->uv_rhs_fld.h();
    d->uv_rhsAB_fld.alloc("dyn.uv_rhsAB", e_nl_2);     d->uv_rhsAB    = d->uv_rhsAB_fld.h();
    d->w_fld.alloc("dyn.w", n_nl);                     d->w           = d->w_fld.h();
    d->w_e_fld.alloc("dyn.w_e", n_nl);                 d->w_e         = d->w_e_fld.h();
    d->w_i_fld.alloc("dyn.w_i", n_nl);                 d->w_i         = d->w_i_fld.h();
    d->cfl_z_fld.alloc("dyn.cfl_z", n_nl);             d->cfl_z       = d->cfl_z_fld.h();
    d->uvnode_fld.alloc("dyn.uvnode", n_nl * 2);       d->uvnode      = d->uvnode_fld.h();
    d->uvnode_rhs_fld.alloc("dyn.uvnode_rhs", n_nl * 2); d->uvnode_rhs = d->uvnode_rhs_fld.h();
    d->u_b_fld.alloc("dyn.u_b", e_nl);                 d->u_b         = d->u_b_fld.h();
    d->v_b_fld.alloc("dyn.v_b", e_nl);                 d->v_b         = d->v_b_fld.h();
    d->u_c_fld.alloc("dyn.u_c", n_nl);                 d->u_c         = d->u_c_fld.h();
    d->v_c_fld.alloc("dyn.v_c", n_nl);                 d->v_c         = d->v_c_fld.h();
    d->eta_n_fld.alloc("dyn.eta_n", n);                d->eta_n       = d->eta_n_fld.h();
    d->d_eta_fld.alloc("dyn.d_eta", n);                d->d_eta       = d->d_eta_fld.h();
    d->ssh_rhs_fld.alloc("dyn.ssh_rhs", n);            d->ssh_rhs     = d->ssh_rhs_fld.h();
    d->ssh_rhs_old_fld.alloc("dyn.ssh_rhs_old", n);    d->ssh_rhs_old = d->ssh_rhs_old_fld.h();
    /* GM bolus velocity (Phase G1) — zero until G4/G6a write to them. */
    d->fer_uv_fld.alloc("dyn.fer_uv", e_nl_2);         d->fer_uv      = d->fer_uv_fld.h();
    d->fer_w_fld.alloc("dyn.fer_w", n_nl);             d->fer_w       = d->fer_w_fld.h();
    FESOM_CHECK(d->uv && d->uv_rhs && d->uv_rhsAB && d->w && d->w_e && d->w_i
             && d->cfl_z && d->uvnode && d->uvnode_rhs
             && d->u_b && d->v_b && d->u_c && d->v_c
             && d->eta_n && d->d_eta && d->ssh_rhs && d->ssh_rhs_old
             && d->fer_uv && d->fer_w,
             "fesom_dyn alloc: out of memory");
}

void fesom_dyn_free(fesom_dyn *d)
{
    // Every array is now OWNED by a fesom::Field; the raw pointers are non-owning aliases, so they
    // must NOT be free()d. `*d = fesom_dyn{}` releases every DualView (Kokkos refcounting) and zeros
    // the PODs — the assignment IS the release (D13). Mirrors fesom_mesh_free.
    *d = fesom_dyn{};
}
