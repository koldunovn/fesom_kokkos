#ifndef FESOM_HALO_H
#define FESOM_HALO_H

#include "fesom_partit.h"
#include "fesom_types.h"

/*
 * Generic halo exchange. Mirror of the 50+ shape-specific
 * exchange_nod / exchange_elem variants in gen_comm.F90, collapsed into
 * one parameterised function plus convenience wrappers.
 *
 * Pack/unpack with explicit buffers (NOT direct Irecv into halo) — Fortran
 * builds MPI_TYPE_INDEXED over detected contiguous blocks because rlist
 * entries from a single rPE are NOT generally contiguous in the local index
 * space (gen_modules_partitioning.F90:237-281). For C, manual pack/unpack is
 * simpler and equivalent.
 *
 * Field layout convention:
 *     field[entity * n_levels * n_components + lev * n_components + comp]
 *
 * For 2D scalar: n_levels=1, n_components=1.
 * For 3D scalar (e.g. T at nodes, mesh->nl levels): n_levels=mesh->nl,
 *   n_components=1.
 * For 3D vector (e.g. UV at elements, 2 components per layer): n_levels=mesh->nl,
 *   n_components=2.
 *
 * Index convention: rlist/slist are 1-based (Fortran-style as stored on
 * disk). The pack/unpack code subtracts 1 when reading/writing field[].
 */

typedef enum {
    FESOM_HALO_NOD2D,        /* uses partit->com_nod2D */
    FESOM_HALO_NOD3D,        /* uses partit->com_nod2D, n_levels per node */
    FESOM_HALO_ELEM2D,       /* uses partit->com_elem2D */
    FESOM_HALO_ELEM3D,       /* uses partit->com_elem2D, n_levels per element */
    FESOM_HALO_ELEM2D_FULL,  /* uses partit->com_elem2D_full (extended halo) */
} fesom_halo_kind;

/* Generic single-field exchange. */
void fesom_halo_exchange(real_t          *field,
                         fesom_halo_kind  kind,
                         int              n_levels,
                         int              n_components,
                         fesom_partit    *p);

/* Convenience wrappers (single field). */
static inline void fesom_exchange_nod2D(real_t *field, fesom_partit *p)
    { fesom_halo_exchange(field, FESOM_HALO_NOD2D, 1, 1, p); }

static inline void fesom_exchange_nod3D(real_t *field, int nl, fesom_partit *p)
    { fesom_halo_exchange(field, FESOM_HALO_NOD3D, nl, 1, p); }

static inline void fesom_exchange_elem2D(real_t *field, fesom_partit *p)
    { fesom_halo_exchange(field, FESOM_HALO_ELEM2D, 1, 1, p); }

static inline void fesom_exchange_elem3D(real_t *field, int nl, fesom_partit *p)
    { fesom_halo_exchange(field, FESOM_HALO_ELEM3D, nl, 1, p); }

static inline void fesom_exchange_elem3D_full(real_t *field, int nl, fesom_partit *p)
    { fesom_halo_exchange(field, FESOM_HALO_ELEM2D_FULL, nl, 1, p); }

/* Identity smoke test (single rank → no-op; multi-rank → pos + neg test).
 * Allocates a temporary field of size (myDim_nod2D + eDim_nod2D) and
 * verifies that after halo exchange, halo entries equal the owning rank id.
 * Aborts on mismatch. Then deliberately corrupts a halo entry on rank 0,
 * re-exchanges, and verifies the corruption is overwritten. */
void fesom_halo_identity_test(fesom_partit *p);

/* Free internal scratch buffers (called from fesom_partit_finalize). */
void fesom_halo_free_buffers(void);

/* M5.17 halo MPI profiler (barrier-isolation; env-gated FESOM_HALO_BARRIER /
 * FESOM_HALO_MPI_PROF; zero cost in production). Splits the per-step halo wait
 * into load-imbalance (MPI_Barrier) vs comm (MPI_Waitall). Used by both the
 * host (fesom_halo.cpp) and device (fesom_halo_device.cpp) exchange paths;
 * defined in fesom_halo_device.cpp (always compiled). fesom_halo_mpi_report is
 * a COLLECTIVE — every rank must call it. */
void fesom_halo_prof_barrier(fesom_partit *p);
void fesom_halo_prof_waitall(int n_reqs, MPI_Request *reqs);
void fesom_halo_prof_bytes(double bytes);
void fesom_halo_mpi_report(int timed_steps, fesom_partit *p);

#endif /* FESOM_HALO_H */
