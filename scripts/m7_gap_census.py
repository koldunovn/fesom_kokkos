#!/usr/bin/env python3
"""
m7_gap_census.py — the GPU-IDLE GAP CENSUS, and a DIFF of two of them.

THE CHEAPEST HONEST DIAGNOSTIC IN THIS PROJECT (L87). It needs no new run: any nsys sqlite
already on disk will do. It beat the memcpy arithmetic on package H (predicted 58.7 ms of ice
rail stall, actual 50.4; the arithmetic had said 34).

WHAT IT MEASURES
----------------
The wall time between the END of one kernel and the START of the next, i.e. the time the SMs
are doing NOTHING. That is the only time a host->device port can actually recover. A HOST TIMER
cannot tell you this: it says how long the host SAT THERE, which includes waiting for the GPU to
drain work it owes anyway (L87 — that error over-sized D.1 by 3x).

Every gap is attributed to THE KERNEL THE GAP ENDS AT — the kernel that was kept waiting. That
names the stall by its victim, which is the actionable end: "ice_thermodynamics waits 22.7 ms"
points straight at the rails feeding it.

Each gap is further split into the part covered by a MEMCPY (PCIe traffic — the copy engine is
busy even though the SMs are idle; removable by deleting the rail) and the part covered by
nothing at all (HOST/MPI — removable only by porting or overlapping the host code).

TWO NSYS TRAPS THIS SCRIPT EXISTS TO AVOID
------------------------------------------
1. Kernel `shortName` collapses every Kokkos launch to `cuda_parallel_launch_local_memory`.
   You MUST use `demangledName` and regex the functor out of `ParallelFor<...>`.
2. `MPI_START_WAIT_EVENTS` emits ONE ROW PER REQUEST, so a Waitall on 10 requests appears 10x.
   Dedupe by (start,end,tid) or you over-count its time 10x (raw 22,460 ms vs deduped 66 ms/step).

WINDOWING — READ THIS BEFORE DIFFING TWO RUNS
---------------------------------------------
The whole point of diffing a 300-step census against a 25-step one is that the 25-step run is
STILL COLD. So the window matters and must be stated, not defaulted silently. `--from`/`--to`
take STEP INDICES (0-based into the recovered boundary list). The default skips the first third,
which is the steady-state convention used by m7_stall_budget.py.

Usage:
  m7_gap_census.py <trace.sqlite> [--min-gap-ms 1.0] [--from N] [--to N] [--json out.json]
  m7_gap_census.py <a.sqlite> --diff <b.sqlite>      # census(a) - census(b), per kernel
"""
import argparse
import bisect
import collections
import json
import os
import re
import sqlite3
import sys

NS_MS = 1e-6


# ---------------------------------------------------------------- sqlite helpers
def tables(con):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


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


def kernel_tag(demangled):
    """TRAP 1. The Kokkos launch wrapper buries the functor inside the template args, and
    nsys's `shortName` throws it away entirely (everything becomes
    `cuda_parallel_launch_local_memory`). Dig the functor out of the DEMANGLED name."""
    for pat in (r"ParallelFor<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"ParallelReduce<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"ParallelScan<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\("):
        m = re.search(pat, demangled)
        if m:
            return m.group(1)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*_kk)", demangled)
    return m.group(1) if m else demangled[:46]


def load_kernels(con):
    t = "CUPTI_ACTIVITY_KIND_KERNEL"
    if t not in tables(con):
        return []
    c = cols(con, t)
    name_col = pick(["demangledName", "shortName", "nameId"], c)   # TRAP 1: demangled FIRST
    if name_col != "demangledName":
        print(f"WARNING: no demangledName column; falling back to {name_col}. "
              f"Kernel names will collapse (TRAP 1).", file=sys.stderr)
    sid = strings(con)
    rows = con.execute(f"SELECT start, end, {name_col} FROM {t} ORDER BY start")
    return [(s, e, kernel_tag(sid.get(n, f"<{n}>"))) for s, e, n in rows]


def load_memcpy(con):
    out = []
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tables(con):
            out += list(con.execute(f"SELECT start, end FROM {t}"))
    return sorted(out)


# copyKind is CUPTI's enum: 1=HtoD 2=DtoH 3=HtoH 4=DtoD 8=PtoP ... We only care about the
# direction, because a rail's DIRECTION tells you which side is authoritative and therefore
# WHICH sync_*/modify_* call emitted it.
COPY_KIND = {1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "PtoP", 10: "DtoH"}


def load_memcpy_detail(con):
    """(start, end, bytes, kind) — for itemising the PCIe traffic inside a gap."""
    t = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if t not in tables(con):
        return []
    c = cols(con, t)
    if "bytes" not in c or "copyKind" not in c:
        return []
    rows = con.execute(f"SELECT start, end, bytes, copyKind FROM {t} ORDER BY start")
    return [(s, e, b, COPY_KIND.get(k, f"k{k}")) for s, e, b, k in rows]


def load_cuda_api(con):
    """TRAP 3 (added 2026-07-15, and it was a flaw in THIS script).

    The `host` column used to be `gap - (memcpy U MPI)`. That silently lumps together two
    completely different things:
      - the host spinning inside cudaDeviceSynchronize (a FENCE — the GPU has already drained;
        you fix it by removing the fence), and
      - the host running its own code (a PORT — you fix it by moving the loop to the device).
    A fence spin has no memcpy and no MPI, so it landed in `host` and looked exactly like host
    compute. Sizing a port off that number would repeat L87's mistake in a new costume.

    So: subtract the CUDA RUNTIME API too, and report it in its own column. `host` then means
    what it says — the host is in NO traced call at all.

    ⚠️ CUPTI records the runtime API with a VERSION SUFFIX (cudaLaunchKernel_v7000), so match on
    the base name or you find nothing and cheerfully report 0.0."""
    t = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if t not in tables(con):
        return []
    sid = strings(con)
    name_col = pick(["nameId", "demangledName", "shortName"], cols(con, t))
    rows = con.execute(f"SELECT start, end, {name_col} FROM {t} ORDER BY start")
    return [(s, e, re.sub(r"_v\d+$", "", sid.get(n, f"<{n}>"))) for s, e, n in rows]


def load_mpi(con):
    """TRAP 2. MPI_START_WAIT_EVENTS holds one row PER REQUEST: a Waitall(n) emits n rows all
    carrying the SAME (start,end). Dedupe on (start,end,tid) before touching counts or times."""
    sid = strings(con)
    seen, out = set(), []
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
            out.append((s, e, n if isinstance(n, str) else sid.get(n, f"<{n}>")))
    return sorted(out)


# ---------------------------------------------------------------- interval algebra
def merge(ivals):
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


def covered(ivals, starts, s, e):
    """ns of [s,e) covered by the merged disjoint list `ivals` (bisect, not a linear scan)."""
    i = max(0, bisect.bisect_right(starts, s) - 1)
    acc = 0
    while i < len(ivals) and ivals[i][0] < e:
        acc += max(0, min(e, ivals[i][1]) - max(s, ivals[i][0]))
        i += 1
    return acc


# ---------------------------------------------------------------- step recovery
def step_boundaries(kernels, forced=None):
    """In an N-step run most kernels launch exactly once per step, so the MODE of the per-kernel
    launch counts IS N. Any kernel with exactly N launches is a once-per-step anchor."""
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
    return sorted(k[0] for k in kernels if k[2] == anchor), nsteps, anchor


# ---------------------------------------------------------------- the census
def census(path, min_gap_ns, frm=None, to=None, steps=None):
    con = sqlite3.connect(path)
    kern = load_kernels(con)
    if not kern:
        raise SystemExit(f"FATAL: {path} has no CUPTI_ACTIVITY_KIND_KERNEL rows.")
    mcpy = merge(load_memcpy(con))
    mpi = merge([(s, e) for s, e, _ in load_mpi(con)])          # TRAP 2 handled in load_mpi
    bnds, nsteps, anchor = step_boundaries(kern, steps)
    if len(bnds) < 6:
        raise SystemExit(f"FATAL: {path}: could not recover step boundaries ({len(bnds)} found).")

    k0 = frm if frm is not None else max(1, nsteps // 3)
    k1 = to if to is not None else len(bnds) - 1
    k0, k1 = max(0, k0), min(len(bnds) - 1, k1)
    lo, hi = bnds[k0], bnds[k1]
    win_steps = k1 - k0
    win = hi - lo
    if win <= 0 or win_steps <= 0:
        raise SystemExit(f"FATAL: {path}: empty window (steps {k0}..{k1}).")

    # A memcpy and an MPI call OVERLAP each other constantly (CUDA-aware MPI stages the halo
    # through the copy engine from inside MPI_Waitall). So `host` CANNOT be computed as
    # gap - pcie - mpi: that subtracts the overlap twice and prints a NEGATIVE host time.
    # Take the UNION for the host residual, and let the pcie/mpi columns overlap.
    #
    # TRAP 3: the union must include the CUDA RUNTIME API, or a FENCE SPIN (no memcpy, no MPI)
    # is reported as HOST COMPUTE — and you size a device port off a number that a fence removal
    # would have got for free.
    api = load_cuda_api(con)
    fence_iv = merge([(s, e) for s, e, n in api if n == "cudaDeviceSynchronize"])
    api_iv = merge([(s, e) for s, e, _ in api])
    busy_host = merge(mcpy + mpi + api_iv)
    mcpy_s = [x[0] for x in mcpy]
    mpi_s = [x[0] for x in mpi]
    fence_s = [x[0] for x in fence_iv]
    api_s = [x[0] for x in api_iv]
    busy_s = [x[0] for x in busy_host]
    mdet = load_memcpy_detail(con)
    mdet_s = [x[0] for x in mdet]

    # Gaps between CONSECUTIVE KERNELS, attributed to the kernel the gap ENDS at.
    # (Kernels can overlap on different streams, so track the running max end, not the prev end.)
    #
    # ALSO key every gap by the PAIR (predecessor -> victim). "ocean2ice waits 16.8 ms" does not
    # tell you WHICH host code to delete; "bulk_compute -> ocean2ice waits 16.8 ms, 6.7 of it in
    # 7 DtoH copies of 3.71 MB" points straight at a specific sync_host() block.
    g_tot = collections.Counter()   # ns of gap, by victim kernel
    g_n = collections.Counter()     # how many such gaps
    g_cpy = collections.Counter()   # ns of that gap covered by a memcpy (PCIe)
    g_mpi = collections.Counter()   # ns covered by an MPI call
    g_fen = collections.Counter()   # ns covered by cudaDeviceSynchronize (a FENCE — remove it)
    g_hst = collections.Counter()   # ns covered by NOTHING TRACED (real host code — PORT it)
    p_tot = collections.Counter()   # ns of gap, by (predecessor, victim)
    p_cpy = collections.Counter()   # ns of PCIe, by pair
    p_by = collections.Counter()    # copy BYTES, by pair
    p_nc = collections.Counter()    # copy COUNT, by pair
    p_dir = collections.defaultdict(collections.Counter)   # bytes by direction, by pair
    prev_end = None
    prev_tag = "<start>"
    n_gaps = 0
    gap_sum = 0
    ci = 0
    for s, e, tag in kern:
        if prev_end is not None and s > prev_end and lo <= prev_end and s <= hi:
            gap = s - prev_end
            if gap >= min_gap_ns:
                pair = (prev_tag, tag)
                g_tot[tag] += gap
                g_n[tag] += 1
                g_cpy[tag] += covered(mcpy, mcpy_s, prev_end, s)
                g_mpi[tag] += covered(mpi, mpi_s, prev_end, s)
                g_fen[tag] += covered(fence_iv, fence_s, prev_end, s)
                g_hst[tag] += gap - covered(busy_host, busy_s, prev_end, s)
                p_tot[pair] += gap
                p_cpy[pair] += covered(mcpy, mcpy_s, prev_end, s)
                # itemise the copies that START inside this gap
                j = bisect.bisect_left(mdet_s, prev_end)
                while j < len(mdet) and mdet[j][0] < s:
                    p_by[pair] += mdet[j][2]
                    p_nc[pair] += 1
                    p_dir[pair][mdet[j][3]] += mdet[j][2]
                    j += 1
                n_gaps += 1
                gap_sum += gap
        prev_end = e if prev_end is None else max(prev_end, e)
        prev_tag = tag

    kern_busy = sum(e - s for s, e in merge([(x[0], x[1]) for x in kern
                                             if lo <= x[0] < hi]))
    con.close()
    return {
        "path": path, "nsteps": nsteps, "anchor": anchor,
        "win_steps": win_steps, "k0": k0, "k1": k1,
        "step_ms": win * NS_MS / win_steps,
        "kern_busy_ms": kern_busy * NS_MS / win_steps,
        "gap_ms": gap_sum * NS_MS / win_steps,
        "gaps_per_step": n_gaps / win_steps,
        "by_kernel": {t: {"ms": g_tot[t] * NS_MS / win_steps,
                          "n": g_n[t] / win_steps,
                          "pcie_ms": g_cpy[t] * NS_MS / win_steps,
                          "mpi_ms": g_mpi[t] * NS_MS / win_steps,
                          "fence_ms": g_fen[t] * NS_MS / win_steps,
                          "host_ms": g_hst[t] * NS_MS / win_steps}
                      for t in g_tot},
        "by_pair": {f"{p[0]} -> {p[1]}": {
                        "ms": p_tot[p] * NS_MS / win_steps,
                        "pcie_ms": p_cpy[p] * NS_MS / win_steps,
                        "copies": p_nc[p] / win_steps,
                        "MB": p_by[p] / 1048576.0 / win_steps,
                        "dir": {d: v / 1048576.0 / win_steps
                                for d, v in p_dir[p].items()}}
                    for p in p_tot},
    }


def show(c, min_gap_ms):
    print("=" * 100)
    print(f"GPU-IDLE GAP CENSUS — {os.path.basename(c['path'])}")
    print("=" * 100)
    print(f"steps in trace: {c['nsteps']}   anchor kernel: {c['anchor']}")
    print(f"window: steps {c['k0']}..{c['k1']}  ({c['win_steps']} steps)")
    print(f"STEP TIME (rank 0, from the trace): {c['step_ms']:.1f} ms/step")
    print(f"kernels busy:                       {c['kern_busy_ms']:.1f} ms/step "
          f"({100*c['kern_busy_ms']/c['step_ms']:.1f}%)")
    print(f"GAPS > {min_gap_ms} ms between kernels:   {c['gap_ms']:.1f} ms/step "
          f"({100*c['gap_ms']/c['step_ms']:.1f}%)  in {c['gaps_per_step']:.1f} gaps/step")
    print()
    print(f"  {'the kernel KEPT WAITING':<40} {'gap':>7} {'n':>5} {'PCIe':>7} {'MPI':>7} "
          f"{'FENCE':>7} {'host':>7}")
    print(f"  {'-'*40} {'-'*7} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for t, d in sorted(c["by_kernel"].items(), key=lambda kv: -kv[1]["ms"])[:22]:
        print(f"  {t[:40]:<40} {d['ms']:7.1f} {d['n']:5.1f} {d['pcie_ms']:7.1f} "
              f"{d['mpi_ms']:7.1f} {d['fence_ms']:7.1f} {d['host_ms']:7.1f}")
    print()
    print("  gap   = ms/step the SMs sat idle immediately before this kernel")
    print("  PCIe  = of that, ms covered by a memcpy      -> a RAIL.  DELETE IT.")
    print("  MPI   = of that, ms covered by an MPI call   -> COMM.    OVERLAP IT.")
    print("  FENCE = of that, ms inside cudaDeviceSynchronize -> a FENCE. REMOVE IT.")
    print("  host  = covered by NOTHING TRACED           -> HOST CODE. PORT IT.")
    print("  🔴 FENCE vs host is the distinction that decides WHICH LEVER you reach for, and the")
    print("     first version of this script did not make it — a fence spin has no memcpy and no")
    print("     MPI, so it landed in `host` and looked exactly like host compute you must port.")
    print("  NB PCIe/MPI/FENCE overlap each other, so they do NOT sum to `gap`. `host` is the")
    print("     residual against their UNION.")
    print()
    print("--- the same gaps, named by the PAIR they sit between (this is the actionable form) ---")
    print(f"  {'predecessor -> the kernel kept waiting':<58} {'gap':>7} {'PCIe':>7} "
          f"{'copies':>7} {'MB':>8}  direction")
    print(f"  {'-'*58} {'-'*7} {'-'*7} {'-'*7} {'-'*8}  {'-'*22}")
    for p, d in sorted(c["by_pair"].items(), key=lambda kv: -kv[1]["ms"])[:16]:
        dirs = " ".join(f"{k}:{v:.1f}MB" for k, v in
                        sorted(d["dir"].items(), key=lambda kv: -kv[1]))
        print(f"  {p[:58]:<58} {d['ms']:7.1f} {d['pcie_ms']:7.1f} {d['copies']:7.1f} "
              f"{d['MB']:8.1f}  {dirs}")
    print("=" * 100)


def show_diff(a, b, min_gap_ms):
    """a - b, per kernel. Convention: a is the LONG run, b is the SHORT (cold) run, so a
    NEGATIVE delta means the cold run paid MORE there — that is the cold-start artifact."""
    print("=" * 100)
    print(f"GAP-CENSUS DIFF   A = {os.path.basename(a['path'])}  ({a['nsteps']} steps, "
          f"window {a['k0']}..{a['k1']})")
    print(f"                  B = {os.path.basename(b['path'])}  ({b['nsteps']} steps, "
          f"window {b['k0']}..{b['k1']})")
    print("=" * 100)
    print(f"  step time      A {a['step_ms']:8.1f}   B {b['step_ms']:8.1f}   "
          f"A-B {a['step_ms']-b['step_ms']:+8.1f} ms/step")
    print(f"  kernels busy   A {a['kern_busy_ms']:8.1f}   B {b['kern_busy_ms']:8.1f}   "
          f"A-B {a['kern_busy_ms']-b['kern_busy_ms']:+8.1f} ms/step")
    print(f"  gaps > {min_gap_ms}ms    A {a['gap_ms']:8.1f}   B {b['gap_ms']:8.1f}   "
          f"A-B {a['gap_ms']-b['gap_ms']:+8.1f} ms/step")
    print()
    keys = set(a["by_kernel"]) | set(b["by_kernel"])
    Z = {"ms": 0, "n": 0, "pcie_ms": 0, "mpi_ms": 0, "host_ms": 0}
    rows = []
    for t in keys:
        da = a["by_kernel"].get(t, Z)
        db = b["by_kernel"].get(t, Z)
        rows.append((t, da["ms"], db["ms"], da["ms"] - db["ms"],
                     da["pcie_ms"] - db["pcie_ms"], da["mpi_ms"] - db["mpi_ms"],
                     da["host_ms"] - db["host_ms"]))
    rows.sort(key=lambda r: r[3])   # most-improved (most negative) first
    print(f"  {'kernel KEPT WAITING':<38} {'A gap':>8} {'B gap':>8} {'A-B':>8} "
          f"{'dPCIe':>8} {'dMPI':>8} {'dhost':>8}")
    print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for t, ga, gb, d, dp, dm, dh in rows:
        if abs(d) < 0.5:
            continue
        print(f"  {t[:38]:<38} {ga:8.1f} {gb:8.1f} {d:+8.1f} {dp:+8.1f} {dm:+8.1f} {dh:+8.1f}")
    print()
    print("  A-B < 0  =>  the SHORT run paid MORE there. That is the COLD-START ARTIFACT.")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--diff", default=None, help="second sqlite; prints census(A) - census(B)")
    ap.add_argument("--min-gap-ms", type=float, default=1.0)
    ap.add_argument("--from", dest="frm", type=int, default=None, help="first step index")
    ap.add_argument("--to", dest="to", type=int, default=None, help="last step index")
    ap.add_argument("--from-b", type=int, default=None)
    ap.add_argument("--to-b", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    min_gap_ns = int(a.min_gap_ms * 1e6)
    for p in [a.sqlite] + ([a.diff] if a.diff else []):
        if not os.path.exists(p):
            print(f"FATAL: {p} not found (did `nsys export --type sqlite` fail?)")
            return 2

    ca = census(a.sqlite, min_gap_ns, a.frm, a.to)
    show(ca, a.min_gap_ms)
    out = {"A": ca}
    if a.diff:
        cb = census(a.diff, min_gap_ns, a.from_b, a.to_b)
        print()
        show(cb, a.min_gap_ms)
        print()
        show_diff(ca, cb, a.min_gap_ms)
        out["B"] = cb
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
