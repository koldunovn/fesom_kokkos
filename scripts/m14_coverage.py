#!/usr/bin/env python3
"""
M14 COVERAGE AUDIT — what the plan asks for vs what has actually been measured.

Why this exists: over 2026-08-15/16 every gap in this campaign was found by the USER asking, not
by me checking. GPU ladders for 3 of 4 meshes were never submitted; the partition lever was
documented as "elsewhere absent" instead of being generated; the interaction hunt waited for a
prompt. Each was depth-first absorption (a CUDA build bug) with nothing auditing breadth.

Run this before every status report. It is mechanical, so it does not depend on my remembering.
"""
import glob
import re
import collections

M14 = "/work/ab0995/a270088/port2/m14"

# The plan's matrix (docs/plans/20260815-m14-integration-campaign.md, Tasks C2/C3/D3/D4).
WANT = {
    ("cpu", "core2"): [128, 256, 512, 864, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
    ("cpu", "farc"):  [512, 1024, 2048, 3072, 4096, 8192],
    ("cpu", "dars"):  [512, 1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384],
    ("cpu", "ng5"):   [2048, 4096, 8192, 16384, 20480, 24576, 32768],
    ("a100", "core2"): [4, 8, 16, 32, 64],
    ("a100", "farc"):  [4, 8, 16, 32, 64],
    ("a100", "dars"):  [8, 16, 32, 64],
    ("a100", "ng5"):   [16, 32, 64],
    ("gh200", "core2"): [4, 8, 16, 32],
    ("gh200", "farc"):  [4, 8, 16, 32],
    ("gh200", "dars"):  [4, 8, 16, 32],
    ("gh200", "ng5"):   [16, 32],
}


def have():
    got = collections.defaultdict(lambda: collections.defaultdict(set))
    for f, pat, plat in ((f"{M14}/ladder.*.out", r"M14 CPU ladder — (\w+) ranks=(\d+)", "cpu"),
                         (f"{M14}/gladder.*.out", r"M14 CPU ladder — (\w+) ranks=(\d+)", "a100"),
                         (f"{M14}/dlad.*.out", r"M14 dolpung ladder — (\w+) gpus=(\d+)", "gh200")):
        for p in glob.glob(f):
            txt = open(p, errors="replace").read()
            m = re.search(pat, txt)
            if not m:
                continue
            mesh, n = m.group(1), int(m.group(2))
            n = max(1, round(n / 128)) * 128 if plat == "cpu" else n   # cpu: ranks
            n = int(m.group(2))
            for arm in ("base", "best"):
                if re.search(rf"\n\s+{arm}\s+legs.*?min=([\d.]+)", txt) or \
                   re.search(rf"\n\s+{arm}\s+min=([\d.]+)", txt):
                    got[(plat, mesh)][arm].add(n)
    return got


if __name__ == "__main__":
    got = have()
    print(f"{'platform/mesh':18s} {'base':>26s} {'best':>26s}")
    tb = tx = wb = 0
    for key in sorted(WANT):
        want = set(WANT[key])
        b = got[key]["base"] & want
        x = got[key]["best"] & want
        tb += len(b); tx += len(x); wb += len(want)
        mb = sorted(want - b)
        print(f"  {key[0]+'/'+key[1]:16s} {len(b):2d}/{len(want):2d}"
              f"{('  missing ' + ','.join(str(v) for v in mb[:6])) if mb else '  COMPLETE':>24s}"
              f" {len(x):2d}/{len(want):2d}")
    print(f"\n  TOTAL base {tb}/{wb} rungs, best-arm {tx}/{wb}")
    print("  (best-arm coverage is expected to lag base; base coverage below 100% means "
          "a ladder was never submitted, not that it failed)")
