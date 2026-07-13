#pragma once
/* ======================================================================= *
 * M7 speed-campaign knobs (docs/plans/20260714-m7-speed-fp64.md).
 *
 * Every M7 perf lever is gated by an env knob FESOM_SPEED_<NAME>, default
 * OFF. FESOM_SPEED=1 is the master switch (turns every blessed lever ON);
 * a per-lever FESOM_SPEED_<NAME>=0|1 always overrides the master.
 *
 * Rules (the M6 rule, extended):
 *   - knob-OFF is byte-identical to certified HEAD at every commit
 *     (gate: jobs/job_m7_gate_serial vs port2/m6_baseline_serial).
 *   - levers act on the CUDA path only: on a Serial-only build the knobs
 *     resolve to OFF, so the Serial backend keeps the legacy
 *     bit-identical-to-C path forever (the debug oracle).
 *   - FESOM_SPEED_FORCE_SERIAL=1 is the dev-only exception: it lets a
 *     lever run on the Serial backend for the FORCE_SERIAL byte proof
 *     (levered Serial run must diff_snap==baseline for any lever that
 *     claims bit-identity). Never set it in production runs.
 *
 * Call-site idiom (one branch in the hot path, the fesom_halo_device.cpp
 * static-lazy-init pattern):
 *
 *     static int s_knob = -1;
 *     if (fesom_speed_on("NOFENCE2", &s_knob)) { ... levered path ... }
 *
 * Value knobs (e.g. FESOM_SPEED_EVPWIDE=2, FESOM_SPEED_SCATTER=1|2):
 *
 *     static int s_k = -2;
 *     int k = fesom_speed_int("EVPWIDE", 0, &s_k);   // 0 = lever off
 *
 * Unrecognised values abort loudly (M6 idiom).
 * ======================================================================= */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

inline void fesom_speed_bad_value(const char *var, const char *val)
{
    fprintf(stderr, "[fesom_speed] unrecognised %s='%s' (expected an integer)\n", var, val);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* Parse an env var as a full-string integer; abort loudly on junk. */
inline bool fesom_speed_parse_int(const char *var, long *out)
{
    const char *e = getenv(var);
    if (!e || !e[0]) return false;
    char *end = nullptr;
    long v = strtol(e, &end, 10);
    if (!end || *end != '\0') fesom_speed_bad_value(var, e);
    *out = v;
    return true;
}

/* FESOM_SPEED_FORCE_SERIAL=1 — dev-only, enables levers on the Serial
 * backend for the byte-proof runs. */
inline bool fesom_speed_force_serial()
{
    static int cached = -1;
    if (cached < 0) {
        long v = 0;
        cached = (fesom_speed_parse_int("FESOM_SPEED_FORCE_SERIAL", &v) && v != 0) ? 1 : 0;
    }
    return cached != 0;
}

/* Resolve a lever, uncached: per-lever knob > master FESOM_SPEED > OFF.
 * On a non-CUDA build the result is forced OFF unless FORCE_SERIAL. */
inline int fesom_speed_resolve(const char *lever)
{
    char var[80];
    snprintf(var, sizeof var, "FESOM_SPEED_%s", lever);
    long v = 0;
    int on;
    if (fesom_speed_parse_int(var, &v))                on = (v != 0);
    else if (fesom_speed_parse_int("FESOM_SPEED", &v)) on = (v != 0);
    else                                               on = 0;
#ifndef KOKKOS_ENABLE_CUDA
    if (on && !fesom_speed_force_serial()) on = 0;   /* Serial stays legacy */
#endif
    return on;
}

/* Boolean lever with caller-held cache (init the static to -1). */
inline bool fesom_speed_on(const char *lever, int *cache)
{
    if (*cache < 0) *cache = fesom_speed_resolve(lever);
    return *cache != 0;
}

/* Integer-valued lever with caller-held cache (init the static to -2).
 * Returns deflt when the lever is off/unset; the master switch does NOT
 * imply a value knob (value knobs must be set explicitly). */
inline int fesom_speed_int(const char *lever, int deflt, int *cache)
{
    if (*cache < -1) {
        char var[80];
        snprintf(var, sizeof var, "FESOM_SPEED_%s", lever);
        long v = 0;
        int r = fesom_speed_parse_int(var, &v) ? (int)v : deflt;
#ifndef KOKKOS_ENABLE_CUDA
        if (!fesom_speed_force_serial()) r = deflt;   /* Serial stays legacy */
#endif
        *cache = r;
    }
    return *cache;
}
