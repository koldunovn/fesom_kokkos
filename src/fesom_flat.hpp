#pragma once
/* fesom_flat.hpp — FESOM_FLAT: run selected edge/element kernels in the flat (X,LEVEL) form.
 *
 * The affected kernels run one thread per EDGE with an inner level loop, so a wavefront reads
 * its per-edge arrays at stride nl*8 B (uncoalesced) and fires its fp64 atomics at one level of
 * 64 different nodes. The flat form gives one thread per (edge,LEVEL): the per-edge read becomes
 * contiguous and each wavefront's atomics land on consecutive levels of ONE node. Measured on
 * fct_zal_b1h at NG5/16: LUMI 7017.9 -> 514.9 ms (13.6x), MN5 594.4 -> 179.2 ms (3.3x).
 *
 * fct_zal_b2 / fct_zal_b3v already ship in this form ("M5.22 flat lever ... coalesced"); this is
 * the same idiom extended to the remaining scatter kernels.
 *
 *   FESOM_FLAT=all | none | comma-list of tags: b1h,lo,f2d,visc1,visc2,momadv
 *
 * Every variant covers the same (edge,level) set with the same values. These kernels already
 * used atomics, so summation order was already non-deterministic: flattening does not make them
 * any less reproducible than they were.
 */
#include <cstdlib>
#include <cstring>

inline bool fesom_flat_on(const char *tag)
{
    static const char *spec = []() { const char *v = getenv("FESOM_FLAT"); return v ? v : ""; }();
    if (!spec[0] || strcmp(spec, "none") == 0) return false;
    if (strcmp(spec, "all") == 0) return true;
    const size_t n = strlen(tag);
    for (const char *p = spec; *p;) {
        const char *c = strchr(p, ',');
        const size_t len = c ? (size_t)(c - p) : strlen(p);
        if (len == n && strncmp(p, tag, n) == 0) return true;
        if (!c) break;
        p = c + 1;
    }
    return false;
}
