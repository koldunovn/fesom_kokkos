#ifndef FESOM_ALE_DUMP_H
#define FESOM_ALE_DUMP_H

/*
 * M6.3 bisect rail — byte-for-byte mirror of the C oracle's fesom_ale_dump.c
 * (port2/fesom2_port_zstar). Same filenames, same header, same %.17g rows, so
 * the SAME scripts/ale_dump_diff.py reads both codes and diffs them gid-keyed:
 *
 *   <dir>/ale_dump_s<step>_<tag>_rank<R>.txt
 *   # step=.. tag=.. rank=R N=.. ncomp=C
 *   <gid> <v0> ... <v(C-1)>          (1-based gid, one row per OWNED entity)
 *
 * Gated on FESOM_ALE_DUMP_DIR; fires for steps 1..FESOM_ALE_DUMP_STEPS (default 3).
 * Every getter reads the HOST alias, so each entry point sync_host()s the Fields it
 * touches first (a no-op on Serial, a bitwise deep_copy on CUDA). Zero effect on the
 * run when the env var is unset.
 */

struct fesom_aux;
struct fesom_dyn;
struct fesom_forcing;
struct fesom_mesh;
struct fesom_partit;

/* 1 if FESOM_ALE_DUMP_DIR is set and step <= FESOM_ALE_DUMP_STEPS. */
int fesom_ale_dump_active(int step);

void fesom_ale_dump_forcing(int step, const struct fesom_forcing *forcing,
                            const struct fesom_mesh *mesh,
                            struct fesom_partit *partit);

void fesom_ale_dump_pgf(int step, const struct fesom_aux *aux,
                        const struct fesom_mesh *mesh,
                        struct fesom_partit *partit);

void fesom_ale_dump_sshsolve(int step, const struct fesom_dyn *dyn,
                             const struct fesom_mesh *mesh,
                             struct fesom_partit *partit);

void fesom_ale_dump_hbar(int step, const struct fesom_dyn *dyn,
                         const struct fesom_mesh *mesh,
                         struct fesom_partit *partit);

void fesom_ale_dump_vertvel(int step, const struct fesom_dyn *dyn,
                            const struct fesom_mesh *mesh,
                            struct fesom_partit *partit);

void fesom_ale_dump_thickness(int step, const struct fesom_mesh *mesh,
                              struct fesom_partit *partit);

#endif /* FESOM_ALE_DUMP_H */
