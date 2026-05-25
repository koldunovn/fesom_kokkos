#include "fesom_tracers.h"
#include "fesom_mesh.h"

#include <stdio.h>    // snprintf — per-tracer Field labels
#include <stdlib.h>
#include <string.h>

void fesom_tracers_alloc(fesom_tracers *t, const struct fesom_mesh *mesh)
{
    // Field members (DualView) make a raw memset UB (L13); value-initialise instead (D13).
    *t = fesom_tracers{};
    t->num_tracers = FESOM_NUM_TRACERS;

    int N = mesh->myDim_nod2D + mesh->eDim_nod2D;
    size_t n_nl = (size_t)N * (size_t)mesh->nl;   // stride nl: [N*nl], NOT N*(nl-1) (PORTING_LESSONS §5)
    for (int k = 0; k < FESOM_NUM_TRACERS; ++k) {
        // M1.3: Field owns storage; raw ptr = non-owning alias = field.h() (D12). Count in ELEMENTS
        // (== the old calloc count), zero-init like calloc → Serial bit-identical.
        char lbl[40];
        snprintf(lbl, sizeof lbl, "tracers.data%d.values", k);
        t->data[k].values_fld.alloc(lbl, n_nl);    t->data[k].values    = t->data[k].values_fld.h();
        snprintf(lbl, sizeof lbl, "tracers.data%d.valuesAB", k);
        t->data[k].valuesAB_fld.alloc(lbl, n_nl);  t->data[k].valuesAB  = t->data[k].valuesAB_fld.h();
        snprintf(lbl, sizeof lbl, "tracers.data%d.valuesold", k);
        t->data[k].valuesold_fld.alloc(lbl, n_nl); t->data[k].valuesold = t->data[k].valuesold_fld.h();
        FESOM_CHECK(t->data[k].values && t->data[k].valuesAB && t->data[k].valuesold,
                    "tracer %d alloc: out of memory", k);
    }
    t->del_ttf_fld.alloc("tracers.del_ttf", n_nl);  t->del_ttf = t->del_ttf_fld.h();
    FESOM_CHECK(t->del_ttf, "del_ttf alloc: out of memory");
}

void fesom_tracers_free(fesom_tracers *t)
{
    // Raw pointers are non-owning aliases; `*t = fesom_tracers{}` resets the data[] array
    // element-wise (releasing each fesom_tracer_data's Fields) and del_ttf_fld, then zeros the
    // PODs — do NOT free() the aliases (D13). Mirrors fesom_mesh_free.
    *t = fesom_tracers{};
}
