#ifndef FESOM_NC_REAL_H
#define FESOM_NC_REAL_H
/*
 * M8 mixed precision: write real_t buffers into NC_DOUBLE variables.
 * Policy (docs/PRECISION_ISLANDS.md): files stay FP64 on disk; conversion
 * happens at the I/O boundary. At FP64 these are direct pass-through calls
 * (zero change — Gate 0); under FESOM_SINGLE_PRECISION they stage through a
 * double copy. `n` = total element count being written (product of counts).
 */
#include <netcdf.h>
#include <stdlib.h>
#include "fesom_types.h"

static inline int fesom_nc_put_vara_real(int ncid, int varid, const size_t *startp,
                                         const size_t *countp, const real_t *data, size_t n)
{
#if defined(FESOM_SINGLE_PRECISION)
    double *tmp = (double *)malloc(n * sizeof(double));
    if (!tmp) return NC_ENOMEM;
    for (size_t i = 0; i < n; ++i) tmp[i] = (double)data[i];
    int rc = nc_put_vara_double(ncid, varid, startp, countp, tmp);
    free(tmp);
    return rc;
#else
    (void)n;
    return nc_put_vara_double(ncid, varid, startp, countp, data);
#endif
}

static inline int fesom_nc_put_var_real(int ncid, int varid, const real_t *data, size_t n)
{
#if defined(FESOM_SINGLE_PRECISION)
    double *tmp = (double *)malloc(n * sizeof(double));
    if (!tmp) return NC_ENOMEM;
    for (size_t i = 0; i < n; ++i) tmp[i] = (double)data[i];
    int rc = nc_put_var_double(ncid, varid, tmp);
    free(tmp);
    return rc;
#else
    (void)n;
    return nc_put_var_double(ncid, varid, data);
#endif
}

/* M16: the read direction — NC_DOUBLE on disk into a real_t buffer (restart, IC-as-restart).
 * FP64: pass-through. SP: staged through a double copy, narrowed once (a float written through
 * the put helpers above reads back bit-exactly: double holds every float). */
static inline int fesom_nc_get_var_real(int ncid, int varid, real_t *data, size_t n)
{
#if defined(FESOM_SINGLE_PRECISION)
    double *tmp = (double *)malloc(n * sizeof(double));
    if (!tmp) return NC_ENOMEM;
    int rc = nc_get_var_double(ncid, varid, tmp);
    if (rc == NC_NOERR) for (size_t i = 0; i < n; ++i) data[i] = (real_t)tmp[i];
    free(tmp);
    return rc;
#else
    (void)n;
    return nc_get_var_double(ncid, varid, data);
#endif
}

#endif /* FESOM_NC_REAL_H */
