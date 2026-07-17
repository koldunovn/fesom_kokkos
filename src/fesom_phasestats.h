#ifndef FESOM_PHASESTATS_H
#define FESOM_PHASESTATS_H

/* ======================================================================= *
 * M7 E.IMB.0 — FESOM_SPEED_PHASESTATS: per-rank per-phase busy/wait wall.
 *
 * The 36.1/53.0 ms/step rank-imbalance pool (session-11 E.split) is runtime-
 * real but phase-blind: the barrier split says HOW MUCH skew the exchanges
 * absorb, not WHICH phase generates it. This diagnostic attributes it:
 * per-rank wall per phase {force, ice, coupl, ocean, cg, other} over the
 * timed window, split into busy (own work) and wait (inside MPI), reduced
 * to rank-min/mean/max (+argmax rank) and dumped as a full per-rank table.
 *
 * Design constraints (all load-bearing):
 *   - NO fences. FESOM_STEP_PROFILE's TP_ brackets fence at every phase
 *     boundary (2-3 % and it perturbs what the fence census measures); here
 *     a phase mark is one MPI_Wtime. Phase walls are honest because every
 *     phase is dense with pre-MPI pack fences already (the mandatory ones).
 *   - Wait is measured INSIDE MPI, via PMPI interposition (fesom_phasestats.cpp
 *     wraps Waitall/Allreduce/... and attributes to the CURRENT phase), so
 *     busy = wall − wait is structurally complete — an unhooked collective
 *     would smear victim wait into busy and fake compute imbalance (L92
 *     class). A straggler shows as: ITS busy large in the guilty phase;
 *     everyone else's wait large at their next collective.
 *   - Opt-in ONLY (fesom_speed_on_exp): a diagnostic never rides the
 *     FESOM_SPEED master switch (SYNCSTATS precedent, L91/0.19).
 *   - Window-aligned with the loop timer: open() at n==time_warmup+1,
 *     close() before the end-of-loop barrier. Outside the window the PMPI
 *     wrappers are pass-through (one bool load).
 * ======================================================================= */

#include "fesom_partit.h"

enum fesom_ps_phase {
    FESOM_PH_FORCE = 0,   /* jra55 read + bulk + sss/runoff                */
    FESOM_PH_ICE,         /* fesom_ice_step (dyn EVP + advect + thermo)    */
    FESOM_PH_COUPL,       /* oce_fluxes_mom + shortwave penetration        */
    FESOM_PH_OCEAN,       /* fesom_timestep minus the CG solve             */
    FESOM_PH_CG,          /* fesom_ssh_solve_cg_kk (carved out of ocean)   */
    FESOM_PH_OTHER,       /* io streams / calendar / snapshots / loop tail */
    FESOM_PH_N
};

bool fesom_phasestats_enabled(void);            /* cached knob (announces once) */
void fesom_phasestats_open(void);               /* timed-window start: zero + arm */
void fesom_phasestats_mark(int phase);          /* transition: elapsed -> current phase */
void fesom_phasestats_close(void);              /* flush last segment + disarm */
void fesom_phasestats_report(int timed_steps, fesom_partit *p);  /* COLLECTIVE */

#endif /* FESOM_PHASESTATS_H */
