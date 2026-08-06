#!/usr/bin/env python3
"""M10 open-item-3 — replay the fallback STALL guard against a real residual history.

The guard (src/fesom_ssh.cpp, every variant):

    if (resid < best * 0.999) { best = resid; stall = 0; }
    else if (++stall >= STALL_WINDOW || resid > 1e3 * best) -> SSH_FB_STALL

STALL_WINDOW is 20 for cg2/pipecg and 10 for pcsi/oati (in CHECK events / PAIRS, not
iterations). An ill-conditioned CG plateaus by nature, so "20 consecutive iterations
without a 0.1 % improvement" can be reached by a solve that is converging perfectly
well. This script measures the longest such plateau in a trace, and reports whether
the guard WOULD have fired -- including on baseline `cg`, whose body carries no guard
at all (so its "0 fallbacks" is vacuous, not evidence).

Input: any file containing `[ssh-trace] it=<n> ... res=<r>` lines. Solves are segmented
on `it=` going backwards.

Usage: m10_stall_analysis.py <tracefile> [--window N]
"""
import re
import sys

TRACE = re.compile(rb"\[ssh-trace\]\s+it=(\d+).*?res=([0-9eE+.\-naif]+)")


def solves_from(path):
    """Yield [(iter, resid), ...] per solve. Files carry NUL bytes -- read binary."""
    cur, last = [], None
    with open(path, "rb") as fh:
        for line in fh:
            m = TRACE.search(line)
            if not m:
                continue
            it = int(m.group(1))
            try:
                res = float(m.group(2))
            except ValueError:
                continue
            if last is not None and it <= last and cur:
                yield cur
                cur = []
            cur.append((it, res))
            last = it
    if cur:
        yield cur


def guard(hist, window):
    """Replay the guard. Returns (fired_at_iter or None, longest_plateau)."""
    best = hist[0][1]
    stall = worst = 0
    fired = None
    for it, res in hist:
        if res < best * 0.999:
            best, stall = res, 0
        else:
            stall += 1
            worst = max(worst, stall)
            if fired is None and (stall >= window or res > 1e3 * best):
                fired = it
    return fired, worst


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    window = 20
    if "--window" in sys.argv:
        window = int(sys.argv[sys.argv.index("--window") + 1])

    rows = []
    for n, hist in enumerate(solves_from(path), 1):
        fired, worst = guard(hist, window)
        rows.append((n, len(hist), hist[-1][1], worst, fired))

    if not rows:
        sys.exit(f"no [ssh-trace] lines in {path}")

    print(f"{path}   STALL_WINDOW={window}   solves={len(rows)}")
    print(f"{'solve':>6} {'iters':>7} {'final res':>13} {'max plateau':>12} {'guard fires':>12}")
    for n, iters, res, worst, fired in rows:
        print(f"{n:6d} {iters:7d} {res:13.4e} {worst:12d} "
              f"{('it ' + str(fired)) if fired else '-':>12}")

    worst_all = max(r[3] for r in rows)
    nfire = sum(1 for r in rows if r[4] is not None)
    print(f"\nlongest plateau over all solves: {worst_all} iterations "
          f"(guard threshold {window})")
    print(f"solves where the guard would fire: {nfire}/{len(rows)} "
          f"({100.0 * nfire / len(rows):.1f} %)")


if __name__ == "__main__":
    main()
