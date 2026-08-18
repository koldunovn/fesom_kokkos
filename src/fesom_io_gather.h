/*
 * IO subsystem - DOCUMENTED EXCEPTION to the literal Fortran->C port rule.
 * See fesom_io.h banner for context.
 *
 * fesom_io_gather.h - the gather_plan abstraction shared between
 * fesom_io.c (snapshot writer) and fesom_io_stream.c (time-mean
 * streams). Lives in its own header so the two modules don't have
 * to round-trip through fesom_io.h.
 *
 * Internal struct fields are documented but not for direct manipulation
 * by callers; use the four public functions only.
 *
 * Important MPI semantics:
 *   - Nodes are partitioned exclusively across ranks → Σ myDim_n == nod2D.
 *   - Elements may be REPLICATED across rank boundaries (Σ myDim_e ≥ elem2D):
 *     gen_comm.F90:264-270 puts an element in myList_elem2D of EVERY rank that
 *     owns one of its nodes, and each of them computes it.
 *
 * ⚠️ This header used to say duplicate element writes are harmless "because
 * each contributing rank produces the same value after halo exchange". That is
 * FALSE, and the C reference's restart round-trip gate is what disproved it. A
 * halo exchange cannot equalise a replicated element — the element is *owned*
 * on both ranks, so it is in neither one's halo list — and the copies do not
 * agree: visc_filt_bcksct sums into U_b over myDim_edge2D+eDim_edge2D, and two
 * ranks reach the same element's three edges in a different order. Measured on
 * CORE2 at 128 ranks, uv agrees on all 15348 replicated slots after one step
 * and disagrees on 3311 of them after twenty (one ulp). Inherited from the
 * Fortran, not introduced by either port.
 *
 * For the snapshot and time-mean writers this remains harmless in practice —
 * they keep the last writer's value, one ulp from any other owner's. It is NOT
 * harmless for a restart, which is why gather_elem carries the duplicate check
 * (FESOM_IO_GATHER_DUPCHECK=1) and why fesom_io_restart pushes the gathered
 * values back. See rst_canonicalise there, and §2c of
 * port2/fesom2_port_zstar/docs/plans/20260818-restart-io-port.md.
 */
#ifndef FESOM_IO_GATHER_H
#define FESOM_IO_GATHER_H

#include "fesom_types.h"

#include <mpi.h>

struct fesom_mesh;
struct fesom_partit;

typedef struct gather_plan {
    int  npes;
    int  mype;
    int  myDim_n;            /* this rank's interior node count        */
    int  myDim_e;            /* this rank's interior element count     */
    int *all_n;              /* [npes] per-rank myDim_nod2D (rank 0)   */
    int *all_e;              /* [npes] per-rank myDim_elem2D (rank 0)  */
    int *displ_n;            /* [npes] node-count displacements        */
    int *displ_e;            /* [npes] element-count displacements     */
    int  total_n;            /* sum of all_n  == mesh->nod2D           */
    int  total_e;            /* sum of all_e  >= mesh->elem2D          */
    int *gathered_myList_n;  /* [total_n] rank-0 view of myList_nod2D  */
    int *gathered_myList_e;  /* [total_e] rank-0 view of myList_elem2D */
} gather_plan;

void gather_plan_init(gather_plan *gp,
                      const struct fesom_mesh *mesh,
                      struct fesom_partit *partit);
void gather_plan_free(gather_plan *gp);

/* Gather a [myDim_n × stride] node-indexed buffer to rank 0's
 * [total_n × stride] global buffer indexed by global node id (0-based).
 * No-op on non-zero ranks (rank 0 is the only writer in this IO model). */
void gather_node(const real_t *local, int stride,
                 const gather_plan *gp,
                 real_t *global,
                 MPI_Comm comm);

/* Same for element-indexed [myDim_e × stride] → [total_e × stride]. */
void gather_elem(const real_t *local, int stride,
                 const gather_plan *gp,
                 real_t *global,
                 MPI_Comm comm);

/* Names the field in gather_elem's duplicate-check line (FESOM_IO_GATHER_DUPCHECK=1).
 * Reset to "?" after each report, so a writer that does not set one cannot
 * inherit another's label. Costs nothing when the knob is unset. */
void gather_set_label(const char *label);

/* The inverse of the two gathers: rank 0 permutes its global buffer into rank
 * order with the same gathered_myList_* the gather reads on the way out, then
 * MPI_Scatterv sends each rank its interior block. Because the permutation is
 * the gather's read side run backwards, a field written on N ranks and read on
 * M lands on the same global ids either way. Only the OWNED range is written;
 * the caller re-exchanges the halo if its readers touch it. */
void scatter_node(const real_t *global, int stride,
                  const gather_plan *gp,
                  real_t *local,
                  MPI_Comm comm);

void scatter_elem(const real_t *global, int stride,
                  const gather_plan *gp,
                  real_t *local,
                  MPI_Comm comm);

#endif /* FESOM_IO_GATHER_H */
