#!/usr/bin/env python3
"""
m7_stall_budget.py — decompose the FESOM step's GPU-idle time into NAMED causes.

M7 Task 0.3. Input: the sqlite export of an nsys report (`nsys export --type sqlite`)
taken WITHOUT FESOM_STEP_PROFILE (that wrapper adds its own Kokkos::fence pairs —
profiling the fence budget with a profiler that adds fences measures the profiler).

What it answers: PROFILE_M522 attributed ~46% of the NG5@4N step to GPU kernels, ~10%
to halo load-imbalance, 4.6% to true comm, ~3% to PCIe — and left a ~25-30% REMAINDER
described only as "launch gaps + ~880 device fences/step + blocking-sync stalls". This
script splits that remainder.

Method
------
1. Recover step boundaries from the kernel trace: in an N-step run most kernels launch
   exactly once per step, so the MODE of the per-kernel launch counts is N. Any kernel
   with exactly N launches is a once-per-step anchor; its launch times are the step
   boundaries. (No NVTX needed — and NVTX would have cost us added fences.)
2. Take a steady-state window (skip the first third of steps: at step 1-3 uv~0 and the
   FCT limiters short-circuit, so those steps are not representative).
3. GPU-busy = the merged union of kernel and memcpy intervals. GPU-idle = the complement
   inside the window. Every idle nanosecond is then attributed by asking what the HOST
   was doing at that instant:
       MPI wait     — host inside an MPI call (Waitall / Allreduce / Isend / ...)
       fence spin   — host inside cudaDeviceSynchronize / cudaStreamSynchronize /
                      cudaEventSynchronize while the GPU has ALREADY drained. This is
                      the pure overhead of the fence itself.
       launch gap   — host inside cudaLaunchKernel / cudaMemcpyAsync / other CUDA API,
                      GPU starved: the launch queue ran dry.
       host segment — host in no traced call at all: host-side code between phases.
4. Each sync call is then classified by the kernel that immediately precedes it, which
   separates the pre-MPI pack fence (fesom_halo_device.cpp:251, NOT removable — MPI reads
   the device send buffer from the host) from the post-unpack fence (:286 and its device2/
   deviceN twins, the Task-1.1 removal candidate).

⚠️ Honest caveat, printed with the results: "fence spin" and "launch gap" are ENTANGLED.
A fence's true cost is not only the time the host spins after the GPU drains; it is also
that the fence stops the host running ahead, so the launch queue empties and the GPU
starves at the NEXT kernel. Removing a fence therefore recovers its own spin time PLUS
some share of the launch-gap time. The two buckets bound the payoff: spin alone is the
floor, spin+gap the ceiling.

Usage: m7_stall_budget.py <trace.sqlite> [--steps N] [--json out.json]
"""
import argparse
import bisect
import collections
import json
import os
import re
import sqlite3
import sys

# ⚠️ CUPTI records the runtime API with a VERSION SUFFIX — "cudaStreamSynchronize_v3020",
# "cudaLaunchKernel_v7000". Matching these by equality silently finds NOTHING and reports a
# tidy "0 fences/step", which is how the first run of this script produced an all-zero sync
# budget. Always match on the base name.
#
# ⚠️ AND: not every device sync is OURS. Measured on NG5@4N (rank 0, 23-step window):
#     cudaDeviceSynchronize    996/step  <- Kokkos::fence(). ~782 = 2 x 391 halo exchanges,
#                                           ~178 = the CG's 2 parallel_reduce/iter, rest misc.
#     cudaStreamSynchronize   3513/step  <- NOT ours. Tracks the MESSAGE count (3911/step =
#                                           1955 Isend + 1955 Irecv): it is UCX/CUDA-aware
#                                           MPI's internal per-message stream sync.
#     cudaEventSynchronize     123/step  <- also MPI's (never preceded by a halo kernel).
# Counting all three as "fences" reports ~4600 fences/step and credits Task 1.1 with 6x the
# fences it can actually remove. OUR_SYNC is the Kokkos fence; the rest is MPI's cost and is
# attributed to MPI (it is nested inside the MPI intervals anyway).
OUR_SYNC = ("cudaDeviceSynchronize",)
MPI_SYNC = ("cudaStreamSynchronize", "cudaEventSynchronize")
SYNC_APIS = OUR_SYNC
LAUNCH_APIS = ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaMemcpy", "cudaMemsetAsync",
               "cudaMemcpyToSymbolAsync")


def base_api(name):
    """cudaStreamSynchronize_v3020 -> cudaStreamSynchronize"""
    return re.sub(r"_v\d+$", "", name)


def kernel_tag(demangled):
    """The Kokkos launch wrapper buries our functor inside the template args. What ncu and
    nsys show is the C++ SYMBOL (fesom_impl_vert_visc_kk), never the Kokkos runtime label
    (fct_eud_fill) — the two namespaces are different, which is why the plan's label-based
    ncu regex matched almost nothing."""
    for pat in (r"ParallelFor<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"ParallelReduce<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"ParallelScan<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\("):
        m = re.search(pat, demangled)
        if m:
            return m.group(1)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*_kk)", demangled)
    return m.group(1) if m else demangled[:46]


# ---------------------------------------------------------------- sqlite helpers
def tables(con):
    return {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def pick(candidates, available):
    for c in candidates:
        if c in available:
            return c
    return None


def strings(con):
    if "StringIds" not in tables(con):
        return {}
    return {i: v for i, v in con.execute("SELECT id, value FROM StringIds")}


# ---------------------------------------------------------------- interval algebra
def merge(ivals):
    """Merge overlapping [start,end) intervals. Returns a sorted disjoint list."""
    if not ivals:
        return []
    ivals = sorted(ivals)
    out = [list(ivals[0])]
    for s, e in ivals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def clip(ivals, lo, hi):
    out = []
    for s, e in ivals:
        s, e = max(s, lo), min(e, hi)
        if e > s:
            out.append((s, e))
    return out


def total(ivals):
    return sum(e - s for s, e in ivals)


def complement(ivals, lo, hi):
    """Gaps between merged intervals inside [lo,hi)."""
    out, cur = [], lo
    for s, e in ivals:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        out.append((cur, hi))
    return out


def intersect(a, b):
    """Intersection of two sorted disjoint interval lists (linear sweep)."""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap(a, b):
    return total(intersect(a, b))


def subtract(a, b):
    """a \\ b, both sorted disjoint lists."""
    out = []
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append((cur, min(b[k][0], e)))
            cur = max(cur, b[k][1])
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


# ---------------------------------------------------------------- loaders
def load_kernels(con):
    t = "CUPTI_ACTIVITY_KIND_KERNEL"
    if t not in tables(con):
        return []
    c = cols(con, t)
    name_col = pick(["demangledName", "shortName", "nameId"], c)
    sid = strings(con)
    rows = con.execute(f"SELECT start, end, {name_col} FROM {t} ORDER BY start")
    return [(s, e, kernel_tag(sid.get(n, f"<{n}>"))) for s, e, n in rows]


def load_memcpy(con):
    out = []
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tables(con):
            out += list(con.execute(f"SELECT start, end FROM {t}"))
    return sorted(out)


def load_runtime(con):
    t = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if t not in tables(con):
        return []
    sid = strings(con)
    name_col = pick(["nameId", "demangledName", "shortName"], cols(con, t))
    rows = con.execute(f"SELECT start, end, {name_col} FROM {t} ORDER BY start")
    return [(s, e, base_api(sid.get(n, f"<{n}>"))) for s, e, n in rows]


def load_mpi(con):
    """MPI intervals, DEDUPLICATED.

    ⚠️ MPI_START_WAIT_EVENTS holds one row per REQUEST, not per call: a Waitall(nreq) emits
    nreq rows all carrying the SAME (start,end). Summing their durations multiplies one wall
    interval by the request count — which is how the first run of this script reported 804
    ms/step of Waitall inside a 1275 ms step. Dedup on (start,end,tid) before doing anything
    with counts or durations. (The interval-union attribution was immune, since merge()
    collapses identical intervals — but the per-call table was not.)"""
    sid = strings(con)
    seen = set()
    out = []
    for t in sorted(x for x in tables(con) if x.startswith("MPI_")):
        c = cols(con, t)
        if "start" not in c or "end" not in c:
            continue
        name_col = pick(["textId", "nameId", "text"], c)
        if name_col is None:
            continue
        tid = "globalTid" if "globalTid" in c else None
        sel = f"start, end, {name_col}" + (f", {tid}" if tid else "")
        for row in con.execute(f"SELECT {sel} FROM {t}"):
            s, e, n = row[0], row[1], row[2]
            if s is None or e is None:
                continue
            key = (s, e, row[3] if tid else 0, n)
            if key in seen:
                continue
            seen.add(key)
            nm = n if isinstance(n, str) else sid.get(n, f"<{n}>")
            out.append((s, e, nm))
    return sorted(out)


# ---------------------------------------------------------------- step recovery
def step_boundaries(kernels, forced=None):
    """Return (boundaries, nsteps, anchor_name). Most kernels launch once per step, so
    the mode of the launch counts IS the step count."""
    counts = collections.Counter(k[2] for k in kernels)
    if forced:
        nsteps = forced
    else:
        hist = collections.Counter(c for c in counts.values() if c >= 5)
        if not hist:
            return [], 0, None
        nsteps = hist.most_common(1)[0][0]
    exact = [n for n, c in counts.items() if c == nsteps]
    if not exact:
        near = sorted(counts.items(), key=lambda kv: abs(kv[1] - nsteps))
        if not near:
            return [], 0, None
        exact = [near[0][0]]
        nsteps = counts[exact[0]]
    anchor = sorted(exact)[0]
    b = sorted(k[0] for k in kernels if k[2] == anchor)
    return b, nsteps, anchor


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--steps", type=int, default=None,
                    help="force the step count instead of inferring it from the mode")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if not os.path.exists(a.sqlite):
        print(f"FATAL: {a.sqlite} not found (did `nsys export --type sqlite` fail?)")
        return 2
    con = sqlite3.connect(a.sqlite)

    kern = load_kernels(con)
    if not kern:
        print("FATAL: no CUPTI_ACTIVITY_KIND_KERNEL rows — the trace has no CUDA kernels.")
        print("       tables present:", ", ".join(sorted(tables(con))))
        return 2
    mcpy = load_memcpy(con)
    rt = load_runtime(con)
    mpi = load_mpi(con)

    bnds, nsteps, anchor = step_boundaries(kern, a.steps)
    if len(bnds) < 6:
        print(f"FATAL: could not recover step boundaries (found {len(bnds)}).")
        return 2

    # Steady state: drop the first third of the steps (uv~0 spin-up) and the last one.
    k0 = max(1, nsteps // 3)
    k1 = len(bnds) - 1
    lo, hi = bnds[k0], bnds[k1]
    win_steps = k1 - k0
    win = hi - lo
    if win <= 0 or win_steps <= 0:
        print("FATAL: empty steady-state window.")
        return 2

    ns_ms = 1e-6

    # --- GPU busy / idle -----------------------------------------------------
    # Split kernels from memcpy: most of the memcpy rows are the CUDA-aware MPI stack's OWN
    # device staging copies, not ours, so "kernels only" is the honest measure of useful
    # compute while "kernels ∪ memcpy" is the honest measure of when the GPU is unavailable.
    kern_iv = clip(merge([(s, e) for s, e, _ in kern]), lo, hi)
    mcpy_iv = clip(merge(list(mcpy)), lo, hi)
    gpu = clip(merge([(s, e) for s, e, _ in kern] + list(mcpy)), lo, hi)
    busy = total(gpu)
    busy_k = total(kern_iv)
    busy_m = total(subtract(mcpy_iv, kern_iv))   # memcpy time not already covered by a kernel
    idle = complement(gpu, lo, hi)
    idle_t = total(idle)

    # --- host-side interval sets --------------------------------------------
    sync_iv = merge(clip([(s, e) for s, e, n in rt if n in OUR_SYNC], lo, hi))
    launch_iv = merge(clip([(s, e) for s, e, n in rt if n in LAUNCH_APIS], lo, hi))
    # MPI's own device syncs belong to the MPI bucket, not to ours.
    mpi_iv = merge(clip([(s, e) for s, e, _ in mpi]
                        + [(s, e) for s, e, n in rt if n in MPI_SYNC], lo, hi))
    other_rt = merge(clip([(s, e) for s, e, n in rt
                           if n not in OUR_SYNC and n not in MPI_SYNC
                           and n not in LAUNCH_APIS], lo, hi))

    # Attribute the GPU-idle time. The host-side sets overlap (an MPI call sits inside
    # no CUDA call, but a sync can nest under an MPI wrapper region), so make them
    # DISJOINT by priority before intersecting with idle — otherwise a nanosecond gets
    # counted twice and the buckets do not sum to the idle total.
    # Priority: MPI > sync > launch > other CUDA API > (host).
    idle_m = merge(idle)
    p_mpi = mpi_iv
    p_sync = subtract(sync_iv, p_mpi)
    p_launch = subtract(launch_iv, merge(p_mpi + p_sync))
    p_other = subtract(other_rt, merge(p_mpi + p_sync + p_launch))

    a_mpi = overlap(idle_m, p_mpi)
    a_sync = overlap(idle_m, p_sync)
    a_launch = overlap(idle_m, p_launch)
    a_other = overlap(idle_m, p_other)
    a_host = idle_t - (a_mpi + a_sync + a_launch + a_other)

    # --- classify every sync by WHAT FOLLOWS it ------------------------------
    # The halo's two fences are told apart by their role, not their line number:
    #   pre-MPI pack fence  (:251/:344/:439) — immediately followed by MPI_Irecv/Isend.
    #     MUST STAY: MPI reads the device send buffer from a host-posted call, which is not
    #     ordered against the Kokkos stream.
    #   post-unpack fence   (:286/:377/:475) — followed by a kernel, not by MPI. THE Task-1.1
    #     removal candidate: every consumer is a kernel on the same stream, so stream order
    #     already serialises it.
    # "followed by MPI" is a semantically exact test; matching on the preceding kernel's name
    # is not (pack and unpack are two lambdas inside the SAME C++ function, so they demangle
    # to the same symbol and differ only by an "(instance N)" tag).
    kstarts = [k[0] for k in kern]
    mpi_starts = sorted(s for s, _e, n in mpi if n in ("MPI_Irecv", "MPI_Isend"))
    syncs = [(s, e, n) for s, e, n in rt if n in SYNC_APIS and lo <= s < hi]
    klass = collections.Counter()
    kspin = collections.Counter()
    GAP = 50_000   # ns: an MPI post this soon after a fence belongs to that fence

    # Per-sync spin lookup MUST be indexed, not a linear scan of the idle list: there are
    # ~110k syncs and ~100k idle intervals in the window, so a full overlap() per sync is
    # 10^10 operations (the first version of this hung for minutes). Bisect in instead.
    idle_starts = [x[0] for x in idle_m]

    def spin(s, e):
        i = max(0, bisect.bisect_right(idle_starts, s) - 1)
        acc = 0
        while i < len(idle_m) and idle_m[i][0] < e:
            acc += max(0, min(e, idle_m[i][1]) - max(s, idle_m[i][0]))
            i += 1
        return acc

    for s, e, _n in syncs:
        j = bisect.bisect_left(mpi_starts, e)
        mpi_next = (mpi_starts[j] - e) if j < len(mpi_starts) else None
        i = bisect.bisect_right(kstarts, s) - 1
        prev = kern[i][2] if i >= 0 else "<none>"
        if mpi_next is not None and mpi_next < GAP:
            c = "pre-MPI pack fence  (:251/:344/:439 — MUST STAY)"
        elif "halo_exchange_device" in prev or "halo" in prev.lower():
            c = "post-unpack halo fence (:286/:377/:475 — T1.1 TARGET)"
        else:
            c = f"other sync (after {prev[:34]})"
        klass[c] += 1
        kspin[c] += spin(s, e)   # spin = the part of this sync overlapping GPU-idle

    # --- MPI breakdown (already deduplicated in load_mpi) --------------------
    mpi_by = collections.Counter()
    mpi_n = collections.Counter()
    for s, e, n in mpi:
        if lo <= s < hi:
            mpi_by[n] += e - s
            mpi_n[n] += 1

    # --- top kernels ---------------------------------------------------------
    ktime = collections.Counter()
    kn = collections.Counter()
    for s, e, n in kern:
        if lo <= s < hi:
            ktime[n] += e - s
            kn[n] += 1

    step_ms = win * ns_ms / win_steps

    def row(label, ns, extra=""):
        ms = ns * ns_ms / win_steps
        print(f"  {label:<46} {ms:8.3f} ms/step  {100.0*ns/win:6.2f}%  {extra}")

    print("=" * 96)
    print(f"M7 STALL BUDGET — {os.path.basename(a.sqlite)}")
    print("=" * 96)
    print(f"steps inferred: {nsteps}   anchor kernel: {anchor}")
    print(f"steady-state window: steps {k0}..{k1}  ({win_steps} steps, {win*ns_ms:.1f} ms)")
    print(f"STEP TIME (rank 0, from the trace): {step_ms:.3f} ms/step")
    print()
    print("--- top-level split -------------------------------------------------------")
    row("GPU busy (kernels + memcpy, merged)", busy)
    row("  of which: kernels", busy_k)
    row("  of which: memcpy only (mostly MPI staging)", busy_m)
    row("GPU IDLE", idle_t)
    print()
    print("--- the GPU-idle time, attributed by what the HOST was doing --------------")
    row("MPI wait (halo Waitall / CG Allreduce / ...)", a_mpi)
    row("fence spin (GPU already drained)", a_sync)
    row("launch gap (host in cudaLaunchKernel/Memcpy)", a_launch)
    row("other CUDA API", a_other)
    row("host segment (no traced call)", a_host)
    tot = a_mpi + a_sync + a_launch + a_other + a_host
    row("[sum]", tot, f"(idle {idle_t*ns_ms/win_steps:.3f} ms/step)")
    print()
    print("--- OUR device fences (Kokkos::fence -> cudaDeviceSynchronize) ------------")
    print(f"  total: {len(syncs)}  =  {len(syncs)/win_steps:.1f} per step "
          f"(a fence is classified pre-MPI if an MPI post follows within 50us)")
    for c, n in klass.most_common(12):
        print(f"    {c:<50} {n/win_steps:7.1f}/step   spin {kspin[c]*ns_ms/win_steps:7.3f} ms/step")
    n_mpi_sync = sum(1 for s, _e, n in rt if n in MPI_SYNC and lo <= s < hi)
    print(f"  [MPI's OWN device syncs, not ours: {n_mpi_sync/win_steps:.0f}/step "
          f"(cudaStreamSynchronize/EventSynchronize) — counted in the MPI bucket]")
    print()
    print("--- MPI, by call ----------------------------------------------------------")
    for n, t in mpi_by.most_common(10):
        print(f"    {n:<40} {mpi_n[n]/win_steps:7.1f}/step   {t*ns_ms/win_steps:8.3f} ms/step")
    print()
    print("--- top kernels (cross-check vs PROFILE_M522) -----------------------------")
    for n, t in ktime.most_common(12):
        print(f"    {n[:56]:<58} {kn[n]/win_steps:6.1f}/step {t*ns_ms/win_steps:8.3f} ms/step  {100.0*t/win:5.2f}%")
    print()
    print("NOTE: 'fence spin' and 'launch gap' are ENTANGLED. A fence costs not only the")
    print("      time the host spins after the GPU drains, but also that it stops the host")
    print("      running ahead, so the launch queue empties and the GPU starves at the next")
    print("      kernel. Removing a fence recovers its spin PLUS a share of the launch gap:")
    print("      spin alone is the FLOOR of the payoff, spin+gap the CEILING.")
    print("=" * 96)

    if a.json:
        with open(a.json, "w") as f:
            json.dump({
                "sqlite": a.sqlite, "nsteps": nsteps, "window_steps": win_steps,
                "step_ms": step_ms,
                "gpu_busy_ms": busy * ns_ms / win_steps,
                "gpu_idle_ms": idle_t * ns_ms / win_steps,
                "mpi_ms": a_mpi * ns_ms / win_steps,
                "fence_spin_ms": a_sync * ns_ms / win_steps,
                "launch_gap_ms": a_launch * ns_ms / win_steps,
                "other_api_ms": a_other * ns_ms / win_steps,
                "host_seg_ms": a_host * ns_ms / win_steps,
                "syncs_per_step": len(syncs) / win_steps,
                "sync_classes": {c: {"per_step": n / win_steps,
                                     "spin_ms": kspin[c] * ns_ms / win_steps}
                                 for c, n in klass.items()},
                "mpi_by_call": {n: {"per_step": mpi_n[n] / win_steps,
                                    "ms": t * ns_ms / win_steps}
                                for n, t in mpi_by.items()},
                "top_kernels": {n: {"per_step": kn[n] / win_steps,
                                    "ms": t * ns_ms / win_steps,
                                    "pct": 100.0 * t / win}
                                for n, t in ktime.most_common(20)},
            }, f, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
