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

/* ⚠️⚠️ THIS INCLUDE IS LOad-BEARING. DO NOT REMOVE IT. ⚠️⚠️
 *
 * fesom_speed_resolve() below is guarded by `#ifndef KOKKOS_ENABLE_CUDA` (the "Serial stays
 * legacy" rule). That macro comes from Kokkos' generated config. If this header is included
 * in a TU BEFORE anything that pulls the config in, the macro is not yet defined, the guard
 * fires *even on a CUDA build*, and EVERY KNOB IN THAT TU SILENTLY RESOLVES TO OFF.
 *
 * That is not hypothetical: it is exactly what happened to FESOM_SPEED_SWSKIP in
 * fesom_bulk.cpp and FESOM_SPEED_IOACC in fesom_io.cpp. The levers were correct, every
 * correctness gate passed (the FORCE_SERIAL byte proof passes *because* FORCE_SERIAL bypasses
 * this very guard), and the CUDA A/B quietly measured the LEGACY path and reported "0.00%".
 * A knob that does nothing is indistinguishable from a lever that does not pay.
 *
 * Kokkos_Macros.hpp is the cheap header whose only job is to define the KOKKOS_ENABLE_*
 * macros. Including it here makes this header INCLUDE-ORDER-INDEPENDENT. */
#include <Kokkos_Macros.hpp>

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
 * On a non-CUDA build the result is forced OFF unless FORCE_SERIAL.
 *
 * ANNOUNCES ITSELF. Every lever prints one line, once, on rank 0, saying whether it is ON —
 * and shouts if it was REQUESTED but resolved to OFF.
 *
 * This is not cosmetic. FESOM_SPEED_SWSKIP and FESOM_SPEED_IOACC once shipped SILENTLY DEAD on
 * the CUDA build (an include-order bug hid KOKKOS_ENABLE_CUDA from the guard above) and passed
 * EVERY correctness gate while doing so: the knob-OFF byte gate passes because knob-OFF is what
 * a dead knob gives you; the fidelity gate passes because the output is the legacy output; and
 * the FORCE_SERIAL byte proof passes *because FORCE_SERIAL bypasses this very guard*. The only
 * symptom was an A/B reporting 0.00%. **A knob that does nothing is indistinguishable from a
 * lever that does not pay.** So: make the model TELL YOU what it is actually running. */
inline int fesom_speed_resolve(const char *lever)
{
    char var[80];
    snprintf(var, sizeof var, "FESOM_SPEED_%s", lever);
    long v = 0;
    int  asked = 0, on = 0;
    if (fesom_speed_parse_int(var, &v))                { asked = 1; on = (v != 0); }
    else if (fesom_speed_parse_int("FESOM_SPEED", &v)) { asked = 1; on = (v != 0); }
#ifndef KOKKOS_ENABLE_CUDA
    if (on && !fesom_speed_force_serial()) on = 0;   /* Serial stays legacy */
#endif
    int rank = 0, inited = 0;
    MPI_Initialized(&inited);
    if (inited) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        if (on)
            fprintf(stderr, "[fesom_speed] %s = ON\n", var);
        else if (asked && v != 0)
            fprintf(stderr, "[fesom_speed] !! %s WAS REQUESTED BUT RESOLVES TO OFF — the lever is "
                            "NOT running. (Serial backend without FESOM_SPEED_FORCE_SERIAL=1?)\n", var);
        fflush(stderr);
    }
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
