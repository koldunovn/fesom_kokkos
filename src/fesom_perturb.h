#ifndef FESOM_PERTURB_H
#define FESOM_PERTURB_H

/*
 * Initial-condition perturbation — port of `do_perturb` / `gen_perturbation`
 * (upstream gen_ic3d.F90:801-915, 975-1004, namelist group &oce_perturb).
 *
 * WHY THE PORT NEEDS IT. The M16 faithfulness matrix judges an SP-vs-DP departure against the
 * spread of FP64 runs that differ only by a rounding-sized nudge of the initial temperature. Only
 * the Fortran had that facility, so the port's SP departure had to be compared against the
 * *Fortran's* envelope — and since the two codes' DP trajectories have themselves diverged, that
 * compares sensitivities measured about different trajectories. With this, each code gets its own
 * envelope and the comparison becomes apples-to-apples (board §3b-year).
 *
 * KNOBS (all unset = OFF = byte-identical to before, which gate G0 checks):
 *   FESOM_PERTURB          1 to enable (upstream `lperturb`)
 *   FESOM_PERTURB_MODE     initial_only (default) | first_step
 *   FESOM_PERTURB_METHOD   gaussian (default)     | uniform
 *   FESOM_PERTURB_SEED     integer; REQUIRED when enabled. -1 = clock (non-reproducible)
 *   FESOM_PERTURB_TEMP     "a,b"  gaussian: "mean,sigma" [K];   uniform: "min,max" [K]
 *   FESOM_PERTURB_SALT     "a,b"  same, in PSU
 *
 * TWO DELIBERATE DIVERGENCES FROM UPSTREAM, both documented rather than hidden:
 *
 * 1. 🔴 THE PORT'S PERTURBATION IS PARTITION-INDEPENDENT; UPSTREAM'S IS NOT. Upstream seeds the
 *    Fortran intrinsic RNG with `perturb_seed + mype + 37*i` and then draws one number per LOCAL
 *    node, so both the seed and the draw order depend on the decomposition: the same
 *    `perturb_seed` gives a different perturbation field at a different rank count. The port
 *    instead derives each node's draw from a hash of (seed, GLOBAL node id, tracer), so the field
 *    is a property of the seed alone. This is the M13 lesson (`ic_extrap_det`, upstream PR #979):
 *    a partition-dependent initial state is a reproducibility hazard, and an ensemble whose
 *    members cannot be reproduced at another rank count is not much of an ensemble.
 *
 * 2. The port CANNOT be bit-identical to the Fortran here, and neither can the Fortran to itself.
 *    `random_number` is not specified by the Fortran standard — ifort and gfortran produce
 *    different sequences from the same seed — so there is no sequence to match. What is matched is
 *    the STATISTICS and the transform: one draw per node applied to every wet level of that column
 *    (i.e. a 2-D field, constant in the vertical, exactly as upstream does it), and the same
 *    Box-Muller/uniform formulae, including upstream's `u1 < 1e-10` clamp.
 *
 * NOT IMPLEMENTED: `first_step` on a RESTART. Upstream applies the perturbation to the restarted
 * state at mstep<=1; the port's hook sits where upstream's does, after the IC load and before the
 * salt anomaly and the ice IC, which on a restart is overwritten by the restart read. Rather than
 * silently perturb nothing, that combination is refused (see fesom_perturb_apply).
 */

#include "fesom_types.h"

struct fesom_mesh;
struct fesom_tracers;
struct fesom_partit;

/* Apply the configured perturbation, if any. No-op (and no output) when FESOM_PERTURB is unset.
 * `is_restart` selects upstream's mode semantics. Announces itself on rank 0 when it fires — an
 * ensemble member that silently did not perturb would produce a zero-width envelope, which reads
 * as "SP is above the envelope" for every variable (the day-4 L80 trap, paid in job 27355632). */
void fesom_perturb_apply(struct fesom_tracers *tracers,
                         const struct fesom_mesh *mesh,
                         const struct fesom_partit *partit,
                         int is_restart, int mype);

#endif /* FESOM_PERTURB_H */
