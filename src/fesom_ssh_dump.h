#ifndef FESOM_SSH_DUMP_H
#define FESOM_SSH_DUMP_H
/* M10 T3 — SSH solver-lab dump format (plan docs/plans/20260805-m10-ssh-solvers.md).
 *
 * One raw binary file PER RANK PER DUMPED SOLVE:
 *   <FESOM_SSH_DUMP_DIR>/step<NNNN>/rank<NNNNN>.bin
 *
 * Layout (little-endian, x86-64/aarch64 — written field-by-field, no struct
 * padding on the wire):
 *   u64  magic   0x4D3130444D505631 ("M10DMPV1")
 *   i32  version (1)
 *   i32  step    (1-based solve number == model step for the 1-solve/step loop)
 *   i32  npes, mype, myDim, eDim, nnz, nod2D_global
 *   f64  dt, soltol
 *   i32  maxiter
 *   u64  meshhash  (FNV-1a over myList_nod2D[0 .. myDim+eDim) — partition fingerprint)
 *   then 7 arrays, each as { u64 fnv1a-of-bytes, payload }:
 *     rowptr    [myDim+1] i32
 *     colind    [nnz]     i32
 *     values    [nnz]     f64   (the LIVE per-step matrix — zstar drift is captured)
 *     pr_values [nnz]     f64
 *     b         [myDim]   f64   (ssh_rhs, owned rows)
 *     x0        [myDim]   f64   (warm-start d_eta, owned; halos are owner-derived)
 *     x_final   [myDim]   f64   (the converged solution this binary produced)
 *   footer:
 *     i32  iters      f64 final_res (recurrence)    f64 rtol
 *     u64  magic (again — truncation guard)
 *
 * The reader (tools/fesom_ssh_lab.cpp) re-derives rowptr/colind from the mesh via the
 * NORMAL model build and asserts them BITWISE equal to the dumped ones — that proves the
 * lab reconstructed the exact partitioning + CSR the model ran (zero serialization of the
 * com structures, per the plan's dump-format decision). */

#include <stdint.h>
#include <stdio.h>

#define FESOM_SSHDUMP_MAGIC   0x4D3130444D505631ULL
#define FESOM_SSHDUMP_VERSION 1

static inline uint64_t fesom_sshdump_fnv1a(const void *data, size_t nbytes)
{
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < nbytes; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static inline int fesom_sshdump_wr(FILE *f, const void *p, size_t n)
{ return fwrite(p, 1, n, f) == n ? 0 : -1; }
static inline int fesom_sshdump_rd(FILE *f, void *p, size_t n)
{ return fread(p, 1, n, f) == n ? 0 : -1; }

static inline int fesom_sshdump_wr_arr(FILE *f, const void *p, size_t nbytes)
{
    uint64_t h = fesom_sshdump_fnv1a(p, nbytes);
    if (fesom_sshdump_wr(f, &h, 8)) return -1;
    return fesom_sshdump_wr(f, p, nbytes);
}
/* Reads an array written by wr_arr; verifies the checksum. Returns 0 ok, -1 io, -2 fnv. */
static inline int fesom_sshdump_rd_arr(FILE *f, void *p, size_t nbytes)
{
    uint64_t h = 0;
    if (fesom_sshdump_rd(f, &h, 8)) return -1;
    if (fesom_sshdump_rd(f, p, nbytes)) return -1;
    return (h == fesom_sshdump_fnv1a(p, nbytes)) ? 0 : -2;
}

#endif /* FESOM_SSH_DUMP_H */
