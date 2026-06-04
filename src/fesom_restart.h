#ifndef FESOM_RESTART_H
#define FESOM_RESTART_H
/*
 * Simple per-rank binary restart (ng5-long-run). Each rank writes/reads its OWN
 * local prognostic arrays — no gather, no rank-0 OOM, scales trivially. The
 * checkpoint must be RESUMED ON THE SAME RANK COUNT (same partition); enforced by
 * an npes field in the header. Enough state to continue the run (T,S + AB2 history,
 * uv + AB2 momentum, SSH solver carry, ice tracers/velocity/EVP stress) + the step
 * number and model calendar (so JRA55 forcing resumes at the right date).
 */
#include "fesom_calendar.h"

struct fesom_mesh;
struct fesom_dyn;
struct fesom_tracers;
struct fesom_ice;

/* Write <dir>/restart_<rank>.bin for this rank (rank == mype, npes ranks total).
 * Returns 0 on success, non-zero on I/O error. `ice` may be NULL (no sea ice). */
int fesom_restart_write(const char *dir, int step, const fesom_calendar_t *cal,
                        struct fesom_mesh *mesh, struct fesom_dyn *dyn,
                        struct fesom_tracers *tr, struct fesom_ice *ice,
                        int mype, int npes);

/* If <dir>/restart_<rank>.bin exists, load it: overwrite the prognostic Fields
 * (host + device), set *step (last completed step) and *cal. Returns 1 if loaded,
 * 0 if no checkpoint present. Aborts (FESOM_DIE) on a corrupt file or npes mismatch. */
int fesom_restart_read(const char *dir, int *step, fesom_calendar_t *cal,
                       struct fesom_mesh *mesh, struct fesom_dyn *dyn,
                       struct fesom_tracers *tr, struct fesom_ice *ice,
                       int mype, int npes);

#endif /* FESOM_RESTART_H */
