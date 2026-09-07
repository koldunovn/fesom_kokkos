#!/usr/bin/env python3
"""M16 Task A2 — the accumulation ledger, GREP-GENERATED (plan: "the ledger is grep-generated,
not hand-written"). Lists every `x[...] += ...` / `-=` site in src/ whose left side is an indexed
array element or Kokkos view access (model state or a running sum). Excludes comments, integer
counters/offsets, and timing/profile/wire-counter code.

    scripts/m16_accum_ledger.py            markdown table on stdout
    scripts/m16_accum_ledger.py --count    the site count only (the A2 gate compares it with the
                                           ledger row count in docs/PRECISION_ISLANDS.md)
"""
import re, sys, glob, os, datetime
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SKIP_FILES = re.compile(r'fesom_profile|fesom_phasestats|fesom_io_config|fesom_io_stream_dispatch')
LHS = re.compile(r'([A-Za-z_][A-Za-z0-9_]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)*)\s*[\[(][^=;]*[\])]\s*[-+]=')
INTISH = re.compile(r'\b(n|k|i|j|cnt|count|nnz|off|offset|pos|idx|used|len|bytes|nbytes|iters?|steps?|ncalls|nfail|nsamp|nsub|launches|events|allreduces|iallreduces|halo_events|fallbacks|rowptr|colind|ptr|disp|displs|recvcounts|sendcounts)\s*[\[(]')
NOISE = re.compile(r'stats\.|prof\.|wire\.|timer|\bt_[a-z_]*\s*[-+]=|wall|clock|\b(size_t|int|long|unsigned)\b[^;]*[-+]=')
rows = []
for f in sorted(glob.glob('src/*.cpp') + glob.glob('src/*.hpp') + glob.glob('src/*.h')):
    if SKIP_FILES.search(f): continue
    in_block = False
    for ln, line in enumerate(open(f, errors='replace'), 1):
        s = line.strip()
        if in_block:
            if '*/' in s: in_block = False
            continue
        if s.startswith('/*') and '*/' not in s: in_block = True; continue
        if s.startswith('//') or s.startswith('*') or s.startswith('/*'): continue
        code = s.split('//')[0]
        m = LHS.search(code)
        if not m: continue
        if INTISH.search(code) or NOISE.search(code): continue
        rows.append((f, ln, code.strip()))
if '--count' in sys.argv:
    print(len(rows)); sys.exit(0)
print('| site | statement |'); print('|---|---|')
for f, ln, code in rows:
    print(f'| `{f}:{ln}` | `{code.replace("|", chr(92)+"|")}` |')
print(f'\nsites: {len(rows)}  (generated {datetime.date.today()} by scripts/m16_accum_ledger.py)')
