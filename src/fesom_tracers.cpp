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

/* ---- M16 Phase D: use_salt_anomaly (upstream #986 oce_setup_step.F90:255-282) ------------ */
real_t fesom_S_ref_anomaly = 0.0;
static int s_salt_anomaly = 0;
int fesom_salt_anomaly_on(void) { return s_salt_anomaly; }

void fesom_salt_anomaly_setup(fesom_tracers *t, const struct fesom_mesh *mesh, int mype)
{
    const char *e = getenv("FESOM_SALT_ANOMALY");
    real_t sref = 0.0;
    if (!e || !e[0] || strcmp(e, "0") == 0) {
        s_salt_anomaly = 0;
    } else if (strcmp(e, "1") == 0) {
        s_salt_anomaly = 1; sref = 35.0;                     /* upstream: S_ref_anomaly = 35.0_WP */
    } else {
        char *end = NULL;
        const double v = strtod(e, &end);
        if (!end || *end || !(v > 0.0)) {
            fprintf(stderr, "FESOM_SALT_ANOMALY=%s not supported (0 | 1 | <positive S_ref, "
                            "measurement only>) — refusing to guess\n", e);
            exit(1);
        }
        s_salt_anomaly = 1; sref = (real_t)v;
    }
    fesom_S_ref_anomaly = sref;
    if (!s_salt_anomaly) return;
    /* Initial conditions arrive absolute -> convert ONCE here, after the PHC load and
     * insitu2pot (which need absolute S), before the AB copies (init_tracers_AB at step 1).
     * Whole array, as upstream (`tracers%data(2)%values = ... - S_ref_anomaly`); every output
     * path adds S_ref back to the whole array, so below-bottom slots read 0 on disk either way.
     * Restart reads convert (or not) by detection in fesom_restart_read. */
    const size_t n = (size_t)(mesh->myDim_nod2D + mesh->eDim_nod2D) * (size_t)mesh->nl;
    real_t *S = t->data[FESOM_TRACER_S].values;
    for (size_t i = 0; i < n; ++i) S[i] -= sref;
    t->data[FESOM_TRACER_S].values_fld.modify_host();
    t->data[FESOM_TRACER_S].values_fld.sync_device();
    if (mype == 0)
        printf("[fesom_port] use_salt_anomaly: salinity state = S - %g%s\n", (double)sref,
               (sref == 35.0) ? "" : "  (non-standard S_ref: measurement only)");
}
