#include "fesom_aux.h"
#include "fesom_mesh.h"

#include <stdlib.h>
#include <string.h>

void fesom_aux_alloc(fesom_aux *a, const struct fesom_mesh *mesh)
{
    // Field members (DualView) make a raw memset UB (L13); value-initialise instead (D13).
    *a = fesom_aux{};
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int E = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    size_t n_nl = (size_t)N * (size_t)mesh->nl;
    size_t e_nl = (size_t)E * (size_t)mesh->nl;

    // M1.3: Field owns storage; raw pointer = non-owning alias = field.h() (D12). .alloc count
    // is in ELEMENTS (== the old calloc count) and zero-inits like calloc → Serial bit-identical.
    a->density_m_rho0_fld.alloc("aux.density_m_rho0", n_nl); a->density_m_rho0 = a->density_m_rho0_fld.h();
    a->hpressure_fld.alloc("aux.hpressure", n_nl);           a->hpressure      = a->hpressure_fld.h();
    a->bvfreq_fld.alloc("aux.bvfreq", n_nl);                 a->bvfreq         = a->bvfreq_fld.h();
    a->sw_alpha_fld.alloc("aux.sw_alpha", n_nl);             a->sw_alpha       = a->sw_alpha_fld.h();
    a->sw_beta_fld.alloc("aux.sw_beta", n_nl);               a->sw_beta        = a->sw_beta_fld.h();
    a->Kv_fld.alloc("aux.Kv", n_nl);                         a->Kv             = a->Kv_fld.h();
    a->dbsfc_fld.alloc("aux.dbsfc", n_nl);                   a->dbsfc          = a->dbsfc_fld.h();

    a->Av_fld.alloc("aux.Av", e_nl);                         a->Av             = a->Av_fld.h();
    a->pgf_x_fld.alloc("aux.pgf_x", e_nl);                   a->pgf_x          = a->pgf_x_fld.h();
    a->pgf_y_fld.alloc("aux.pgf_y", e_nl);                   a->pgf_y          = a->pgf_y_fld.h();

    /* Phase G2a: MLD1_ind (filled by fesom_pressure_bv). */
    a->MLD1_ind_fld.alloc("aux.MLD1_ind", (size_t)N);        a->MLD1_ind       = a->MLD1_ind_fld.h();

    FESOM_CHECK(a->density_m_rho0 && a->hpressure && a->bvfreq
             && a->sw_alpha && a->sw_beta && a->Kv && a->dbsfc
             && a->Av && a->pgf_x && a->pgf_y
             && a->MLD1_ind,
             "fesom_aux alloc: out of memory");
}

void fesom_aux_free(fesom_aux *a)
{
    // Raw pointers are non-owning aliases; `*a = fesom_aux{}` releases every Field (refcount) and
    // zeros the PODs — do NOT free() the aliases (D13). Mirrors fesom_mesh_free.
    *a = fesom_aux{};
}
