#ifndef FESOM_TYPES_H
#define FESOM_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * The Fortran side uses WP=8 (double precision) everywhere — see
 * oce_modules.F90:8. real_t is the WORKING precision of the model state; under
 * -DFESOM_SINGLE_PRECISION (CMake -DFESOM_PRECISION=single, M8) it flips to
 * float, and FESOM_MPI_REAL flips with it (field/halo traffic only — MPI
 * REDUCTION scalars stay MPI_DOUBLE, the accumulator-island class).
 *
 * dbl_t marks a DELIBERATE FP64 island (docs/PRECISION_ISLANDS.md): after the
 * M8 sweep, every dbl_t in the tree is a registry-documented decision, and raw
 * `double` in state/kernel code is a sweep bug. The double-precision build is
 * REQUIRED to stay bit-identical to the pre-M8 baseline (Gate 0).
 */
#if defined(FESOM_SINGLE_PRECISION)
typedef float  real_t;
#  define FESOM_MPI_REAL MPI_FLOAT
#else
typedef double real_t;
#  define FESOM_MPI_REAL MPI_DOUBLE
#endif
typedef double dbl_t;   /* deliberate FP64 island marker — never flips */

/*
 * Index helpers matching FRESH_START.md §18. Fortran arrays are
 * (nl, nod) column-major; the C representation is row-major
 * [nod][nl] → flat index node*nl + level. Element-vector fields
 * (uv, uv_rhs) are [elem][nl][2].
 *
 * Names are explicit about the array shape so a misuse at the
 * call site reads wrong: FESOM_ELEM3D() on a node array, etc.
 */
/* Compile-time ceiling on the vertical level count `nl`, used to size the per-column
 * stack/lambda-local scratch arrays (host C twins AND device kernels — both need a
 * compile-time bound; device lambda-locals cannot be VLAs). Covers CORE2/farc/dars (47-48),
 * NG5 (70), and the OMEGA 96-level reference, with margin. A mesh with nl > this fails cleanly
 * at load (fesom_mesh.cpp) instead of overflowing the stack. Raise it if a deeper mesh is added. */
#define FESOM_MAX_LEVELS 128

#define FESOM_NODE3D(node, lev, nl)   ((size_t)(node) * (size_t)(nl) + (size_t)(lev))
#define FESOM_ELEM3D(elem, lev, nl)   ((size_t)(elem) * (size_t)(nl) + (size_t)(lev))
#define FESOM_ELEMVEC(elem, lev, nl)  ((size_t)(elem) * (size_t)(nl) * 2 + (size_t)(lev) * 2)
#define FESOM_ELEM_NODE(elem, k)      ((size_t)(elem) * 3 + (size_t)(k))
#define FESOM_EDGE_NODE(edge, k)      ((size_t)(edge) * 2 + (size_t)(k))
#define FESOM_EDGE_TRI(edge, k)       ((size_t)(edge) * 2 + (size_t)(k))

/*
 * Hard-fail on programmer errors and unrecoverable I/O. Phase 1
 * doesn't need a richer error model; we'll layer one in if the port
 * survives.
 */
/* Kill ALL ranks. abort() only kills this process — other MPI ranks
 * would block forever in collectives. MPI_Abort takes the whole job
 * down. If MPI isn't initialised yet, fall back to abort(). */
#define FESOM_DIE(...) do {                                  \
    fprintf(stderr, "[fesom_port FATAL] " __VA_ARGS__);      \
    fprintf(stderr, "\n  at %s:%d\n", __FILE__, __LINE__);   \
    fflush(stderr);                                          \
    fflush(stdout);                                          \
    int _mpi_inited = 0;                                     \
    MPI_Initialized(&_mpi_inited);                           \
    if (_mpi_inited) MPI_Abort(MPI_COMM_WORLD, 1);           \
    abort();                                                 \
} while (0)

#define FESOM_CHECK(cond, ...) do { if (!(cond)) FESOM_DIE(__VA_ARGS__); } while (0)

#endif /* FESOM_TYPES_H */
