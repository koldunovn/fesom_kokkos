/*
 * fesom_perturb.cpp — initial-condition perturbation. See fesom_perturb.h for the contract,
 * the knobs, and the two documented divergences from upstream's do_perturb.
 */
#include "fesom_perturb.h"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_tracers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------- RNG ---
 * splitmix64: a counter-based mixer. Chosen because it needs no state carried across nodes —
 * node n's draws are a pure function of (seed, global id, tracer), which is what makes the field
 * partition-independent (divergence 1 in the header). */
static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* uniform in [0,1), 53 significant bits — computed in double regardless of real_t, so the draws
 * themselves are identical in a DP and an SP build and only the STORED perturbation differs. */
static inline double u01(uint64_t z) { return (double)(z >> 11) * (1.0 / 9007199254740992.0); }

/* one draw for (global node id, tracer index) */
static double draw_u(int64_t seed, int gid, int tracer, int which)
{
    uint64_t h = splitmix64((uint64_t)seed * 0x9E3779B97F4A7C15ULL
                            ^ splitmix64((uint64_t)gid * 3ULL + (uint64_t)tracer));
    return u01(splitmix64(h + (uint64_t)which));
}

/* gen_perturbation (gen_ic3d.F90:975-1004), transform-for-transform. */
static double gen_perturbation(int gaussian, double p1, double p2,
                               int64_t seed, int gid, int tracer)
{
    if (!gaussian) {                               /* uniform: (max-min)*U + min */
        return (p2 - p1) * draw_u(seed, gid, tracer, 0) + p1;
    }
    double u1 = draw_u(seed, gid, tracer, 0);
    double u2 = draw_u(seed, gid, tracer, 1);
    if (u1 < 1.0e-10) u1 = 1.0e-10;                /* upstream's log(0) guard, kept exactly */
    return p2 * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);   /* p1 (mean) unused, as upstream */
}

static int parse_pair(const char *s, double *a, double *b, const char *what)
{
    if (!s || !s[0]) { *a = 0.0; *b = 0.0; return 1; }
    char *end = NULL;
    *a = strtod(s, &end);
    if (!end || *end != ',') {
        fprintf(stderr, "[fesom_perturb] %s=\"%s\" is not \"a,b\"\n", what, s);
        return 0;
    }
    const char *second = end + 1;
    *b = strtod(second, &end);
    if (!end || *end) {
        fprintf(stderr, "[fesom_perturb] %s=\"%s\" is not \"a,b\"\n", what, s);
        return 0;
    }
    return 1;
}

void fesom_perturb_apply(struct fesom_tracers *tracers,
                         const struct fesom_mesh *mesh,
                         const struct fesom_partit *partit,
                         int is_restart, int mype)
{
    const char *on = getenv("FESOM_PERTURB");
    if (!on || !on[0] || strcmp(on, "0") == 0) return;      /* OFF: byte-identical (gate G0) */
    if (strcmp(on, "1") != 0) {
        fprintf(stderr, "[fesom_perturb] FESOM_PERTURB=%s not recognised (want 0 | 1)\n", on);
        exit(1);
    }

    const char *mode   = getenv("FESOM_PERTURB_MODE");
    const char *method = getenv("FESOM_PERTURB_METHOD");
    if (!mode   || !mode[0])   mode   = "initial_only";
    if (!method || !method[0]) method = "gaussian";

    const int first_step = (strcmp(mode, "first_step") == 0);
    if (!first_step && strcmp(mode, "initial_only") != 0) {
        fprintf(stderr, "[fesom_perturb] FESOM_PERTURB_MODE=%s not recognised "
                        "(want initial_only | first_step)\n", mode);
        exit(1);
    }
    const int gaussian = (strcmp(method, "gaussian") == 0);
    if (!gaussian && strcmp(method, "uniform") != 0) {
        /* upstream falls back to uniform with a warning; refuse instead — a typo here changes the
         * distribution of an ensemble member and nothing downstream would ever reveal it */
        fprintf(stderr, "[fesom_perturb] FESOM_PERTURB_METHOD=%s not recognised "
                        "(want gaussian | uniform)\n", method);
        exit(1);
    }

    if (is_restart) {
        if (!first_step) {
            if (mype == 0)
                printf("[fesom_perturb] mode=initial_only and this is a RESTART — not perturbing "
                       "(upstream do_perturb, gen_ic3d.F90:735)\n");
            return;
        }
        /* see "NOT IMPLEMENTED" in fesom_perturb.h */
        fprintf(stderr, "[fesom_perturb] FESOM_PERTURB_MODE=first_step on a RESTART is not "
                        "implemented in the port: the hook sits before the restart read, so the "
                        "perturbation would be overwritten. Refusing rather than perturbing "
                        "nothing.\n");
        exit(1);
    }

    const char *seed_s = getenv("FESOM_PERTURB_SEED");
    if (!seed_s || !seed_s[0]) {
        fprintf(stderr, "[fesom_perturb] FESOM_PERTURB=1 needs FESOM_PERTURB_SEED "
                        "(an integer, or -1 for a clock seed). Refusing to pick one: an ensemble "
                        "member with an unrecorded seed cannot be reproduced.\n");
        exit(1);
    }
    int64_t seed = (int64_t)strtoll(seed_s, NULL, 10);
    int clock_seed = 0;
    if (seed == -1) { seed = (int64_t)time(NULL); clock_seed = 1; }

    double t1, t2, s1, s2;
    if (!parse_pair(getenv("FESOM_PERTURB_TEMP"), &t1, &t2, "FESOM_PERTURB_TEMP")) exit(1);
    if (!parse_pair(getenv("FESOM_PERTURB_SALT"), &s1, &s2, "FESOM_PERTURB_SALT")) exit(1);

    if (mype == 0) {
        printf("\n[fesom_perturb] *** INITIAL CONDITION PERTURBATION ***\n");
        printf("[fesom_perturb]   mode   : %s\n", mode);
        printf("[fesom_perturb]   method : %s\n", method);
        if (clock_seed) printf("[fesom_perturb]   seed   : %lld (from the clock — NOT reproducible)\n",
                               (long long)seed);
        else            printf("[fesom_perturb]   seed   : %lld (reproducible, and "
                               "partition-independent — unlike upstream's)\n", (long long)seed);
        if (gaussian) {
            printf("[fesom_perturb]   T      : mean=%.6g sigma=%.6g K\n", t1, t2);
            printf("[fesom_perturb]   S      : mean=%.6g sigma=%.6g PSU\n", s1, s2);
        } else {
            printf("[fesom_perturb]   T      : uniform [%.6g, %.6g] K\n", t1, t2);
            printf("[fesom_perturb]   S      : uniform [%.6g, %.6g] PSU\n", s1, s2);
        }
    }

    /* One draw per node, added to EVERY wet level of that column — a 2-D field constant in the
     * vertical, exactly as upstream's
     *     values(ulevels_nod2D(n):nlevels_nod2D(n)-1, n) = ... + rnum
     * (1-based there; the port's levels are 0-based). Owned nodes only; the halo is refreshed by
     * the first exchange, as upstream (whose halo sync is commented out for the same reason). */
    const int   nl     = mesh->nl;
    const int   nnodes = mesh->myDim_nod2D;
    double sum[2] = {0.0, 0.0}, amax[2] = {0.0, 0.0};
    long   count  = 0;

    for (int tr = 0; tr < 2; ++tr) {
        const int idx = (tr == 0) ? FESOM_TRACER_T : FESOM_TRACER_S;
        const double p1 = (tr == 0) ? t1 : s1;
        const double p2 = (tr == 0) ? t2 : s2;
        if (p1 == 0.0 && p2 == 0.0) continue;                  /* nothing asked for this tracer */
        real_t *v = tracers->data[idx].values;
        for (int n = 0; n < nnodes; ++n) {
            const int gid = partit->myList_nod2D ? partit->myList_nod2D[n] : n + 1;
            const double rnum = gen_perturbation(gaussian, p1, p2, seed, gid, tr);
            const int lo = mesh->ulevels_nod2D ? mesh->ulevels_nod2D[n] - 1 : 0;
            const int hi = mesh->nlevels_nod2D[n] - 1;         /* exclusive: wet mid-levels */
            for (int lev = lo; lev < hi; ++lev)
                v[FESOM_NODE3D(n, lev, nl)] += (real_t)rnum;
            sum[tr]  += rnum;
            if (fabs(rnum) > amax[tr]) amax[tr] = fabs(rnum);
            if (tr == 0) ++count;
        }
        tracers->data[idx].values_fld.modify_host();
        tracers->data[idx].values_fld.sync_device();
    }

    if (mype == 0) {
        printf("[fesom_perturb]   applied to %ld owned nodes on rank 0; "
               "rank-0 mean dT=%.3e (|max| %.3e), mean dS=%.3e (|max| %.3e)\n",
               count, count ? sum[0] / (double)count : 0.0, amax[0],
               count ? sum[1] / (double)count : 0.0, amax[1]);
        printf("[fesom_perturb] *** PERTURBATION APPLIED ***\n\n");
    }
}
