#include "fesom_forcing.h"
#include "fesom_mesh.h"

#include <stdlib.h>
#include <string.h>

void fesom_forcing_alloc(fesom_forcing *f, const struct fesom_mesh *mesh)
{
    // Field members (DualView) make a raw memset UB (L13); value-initialise instead (D13).
    *f = fesom_forcing{};
    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    int E = mesh->myDim_elem2D + mesh->eDim_elem2D + mesh->eXDim_elem2D;
    size_t n  = (size_t)N;
    size_t e2 = (size_t)E * 2;

    // M1.4: each Field owns its storage; the raw pointer is a non-owning alias = field.h()
    // (D12). .alloc(label, count) takes the count in ELEMENTS (== the old calloc count) and
    // zero-inits like calloc → Serial bit-identical. Halo-sized extents kept verbatim.
    f->heat_flux_fld.alloc("forcing.heat_flux", n);                 f->heat_flux        = f->heat_flux_fld.h();
    f->water_flux_fld.alloc("forcing.water_flux", n);               f->water_flux       = f->water_flux_fld.h();
    f->stress_node_surf_fld.alloc("forcing.stress_node_surf", n*2); f->stress_node_surf = f->stress_node_surf_fld.h();
    f->stress_surf_fld.alloc("forcing.stress_surf", e2);            f->stress_surf      = f->stress_surf_fld.h();
    f->runoff_fld.alloc("forcing.runoff", n);                       f->runoff           = f->runoff_fld.h();
    f->Ssurf_fld.alloc("forcing.Ssurf", n);                         f->Ssurf            = f->Ssurf_fld.h();
    f->virtual_salt_fld.alloc("forcing.virtual_salt", n);           f->virtual_salt     = f->virtual_salt_fld.h();
    f->relax_salt_fld.alloc("forcing.relax_salt", n);               f->relax_salt       = f->relax_salt_fld.h();
    f->Ch_atm_oce_fld.alloc("forcing.Ch_atm_oce", n);               f->Ch_atm_oce       = f->Ch_atm_oce_fld.h();
    f->Ce_atm_oce_fld.alloc("forcing.Ce_atm_oce", n);               f->Ce_atm_oce       = f->Ce_atm_oce_fld.h();
    f->chl_fld.alloc("forcing.chl", n);                             f->chl              = f->chl_fld.h();
    f->sw_3d_fld.alloc("forcing.sw_3d", n * (size_t)mesh->nl);      f->sw_3d            = f->sw_3d_fld.h();
    FESOM_CHECK(f->heat_flux && f->water_flux
             && f->stress_node_surf && f->stress_surf
             && f->runoff && f->Ssurf
             && f->virtual_salt && f->relax_salt
             && f->Ch_atm_oce && f->Ce_atm_oce
             && f->chl && f->sw_3d,
             "fesom_forcing alloc: out of memory");
}

void fesom_forcing_free(fesom_forcing *f)
{
    // Raw pointers are non-owning aliases; `*f = fesom_forcing{}` releases every DualView
    // (Kokkos refcounting) and zeros the PODs — the assignment IS the release (D13). Do NOT
    // free() the aliases. Mirrors fesom_mesh_free / fesom_dyn_free.
    *f = fesom_forcing{};
}
