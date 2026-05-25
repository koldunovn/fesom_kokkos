#ifndef FESOM_TRACERS_H
#define FESOM_TRACERS_H

#include "fesom_types.h"
#include "fesom_field.hpp"   // M1.3: DualView-backed storage for the persistent tracer arrays

struct fesom_mesh;

/*
 * Mirror of Fortran t_tracer_data + t_tracer (MOD_TRACER.F90).
 *
 * Phase 1 carries 2 tracers: T (index 0), S (index 1). Adams-Bashforth
 * order 2 → values + valuesAB only (no valuesold).
 *
 *   data[i].values   [nod2D * nl]   instant T or S
 *   data[i].valuesAB [nod2D * nl]   AB2 history for tracer i
 *   del_ttf          [nod2D * nl]   diffusion + ALE-reconstruction accumulator,
 *                                   zeroed at the start of each timestep
 *                                   (FRESH_START.md §5)
 *
 * Note: per FRESH_START.md §14.1, horizontal/explicit-vertical diffusion
 * accumulate into del_ttf; the implicit TDMA later operates on the
 * ALE-reconstructed tracer. Do NOT apply diffusion directly to values.
 */
typedef struct fesom_tracer_data {
    real_t *values;
    real_t *valuesAB;
    real_t *valuesold;   /* previous-timestep values; AB2 single slot
                            (Fortran: tracers%data(k)%valuesold(1,:,:) only) */

    /* M1.3: each pointer above is a NON-OWNING alias = field.h(), re-pointed once in
     * fesom_tracers_alloc (D12). All three are [nod2D * nl] — stride nl (FESOM_NODE3D),
     * NOT nl-1 (PORTING_LESSONS §5 / feedback_tracer_stride_nl). The AB2 history update
     * is an element-wise/memcpy VALUE copy (valuesold = values), never a pointer swap,
     * so the aliases stay valid for the whole run. */
    fesom::Field values_fld;
    fesom::Field valuesAB_fld;
    fesom::Field valuesold_fld;
} fesom_tracer_data;

#define FESOM_NUM_TRACERS 2
#define FESOM_TRACER_T 0
#define FESOM_TRACER_S 1

typedef struct fesom_tracers {
    int                num_tracers;
    fesom_tracer_data  data[FESOM_NUM_TRACERS];
    real_t            *del_ttf;

    /* M1.3: del_ttf owner ([nod2D * nl], zeroed each step); raw ptr is its non-owning alias.
     * Assigning `*t = fesom_tracers{}` resets the data[] array element-wise (each
     * fesom_tracer_data's Fields) AND this Field — so it is the full release (D13). */
    fesom::Field del_ttf_fld;
} fesom_tracers;

void fesom_tracers_alloc(fesom_tracers *t, const struct fesom_mesh *mesh);
void fesom_tracers_free (fesom_tracers *t);

#endif /* FESOM_TRACERS_H */
