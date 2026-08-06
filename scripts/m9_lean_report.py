#!/usr/bin/env python3
"""M9 P5 — the LEAN wide halo against the EXACT wide halo, in BOTH currencies.

Quoting a sea-ice scheme as a percentage of the MODEL STEP is what made session 4 report the
divergence form as a null result when, measured against the ice, it removes 2-8.5 % on CPU at
>= 512 ranks. The ice is 5-13 % of the step, so the step divides an ice-only change by about ten.
Every row here therefore carries both:

  step      -- from the CLEAN legs (no PHASESTATS; the instrument perturbs)
  ice cost  -- from the INSTRUMENTED legs, ice + icedyn + iceadv, busy AND wait, which is the
               2019 paper's definition (icedyn busy alone undercounts the ice by about 3x)

and the ROW THAT MATTERS is lean-vs-exact at matched K, because the ring computation is common to
both: the difference between them is the launches and the ordering machinery, which is exactly
the question S. Danilov asked.

usage: m9_lean_report.py [--root /work/ab0995/a270088/port2/m9] [--prefix p5_]
"""
import argparse, glob, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/ab0995/a270088/port2/m9")
ap.add_argument("--prefix", default="p5_")
args = ap.parse_args()

HDR = re.compile(r"^=== M9 (?:CPU|GPU) A/B\s+TAG=(\S+)\s+mesh=(\S+)\s+nodes=(\d+)\s+ntasks=(\d+)"
                 r"\s+dt=(\d+)\s+steps=(\d+)")
LEG = re.compile(r"^LEG\s+(\S+)\s+min\s+([0-9.]+)\s+s/step")
MODE = re.compile(r"^\s+mode\s+= (\S+)")
PHASE = re.compile(r"\[phasestats\]\s+(\w+)\s+\|\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|"
                   r"\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|\s+([0-9.]+)")
ICE = ("ice", "icedyn", "iceadv")


def ice_cost(legdir):
    """busy+wait over the three ice phases, in ms/step. None if no rep log has phasestats.

    The A/B job writes one log per rep (`run.<rep>.log`); the reported s/step is the MIN over
    reps, so the phase split is taken from the same place — the rep with the smallest ice cost.
    Mixing the min step from one rep with the phases of another is the kind of quiet
    inconsistency that turns a 2 % effect into a 5 % one."""
    best = None
    for logpath in sorted(glob.glob(os.path.join(legdir, "run*.log"))):
        tot, seen = 0.0, False
        try:
            with open(logpath, errors="replace") as f:
                for ln in f:
                    m = PHASE.search(ln)
                    if not m or m.group(1) not in ICE:
                        continue
                    seen = True
                    tot += float(m.group(3)) + float(m.group(6))   # busy mean + wait mean
        except OSError:
            continue
        if seen and (best is None or tot < best):
            best = tot
    return best


runs = {}          # tag -> {"hdr":..., "clean":{leg:s}, "phst":{leg:ms}}
for out in sorted(glob.glob(os.path.join(args.root, "ab*.out"))):
    tag = hdr = mode = None
    legs = {}
    with open(out, errors="replace") as f:
        for ln in f:
            m = HDR.match(ln)
            if m:
                tag, hdr = m.group(1), m.groups()
            m = MODE.match(ln)
            if m:
                mode = m.group(1)
            m = LEG.match(ln)
            if m:
                legs[m.group(1)] = float(m.group(2))
    if not tag or not tag.startswith(args.prefix):
        continue
    base = tag[:-5] if tag.endswith("_phst") else tag
    r = runs.setdefault(base, {"hdr": hdr, "clean": {}, "phst": {}})
    if hdr:
        r["hdr"] = hdr
    if mode and mode.startswith("INSTRUMENTED"):
        for lg in legs:
            c = ice_cost(os.path.join(args.root, tag, lg))
            if c is not None:
                r["phst"][lg] = c
    else:
        r["clean"].update(legs)

PAIRS = [("wide_k8", "wide_k8_lean", "classic form, K=8"),
         ("widediv_k8", "widediv_k8_lean", "divergence form, K=8"),
         # The delayed exchange is the speed bar the wide halo was reported to fall short of by
         # about 3x. Quoting that comparison across days would be exactly the drift the campaign
         # banned, so the `p6_` fleet carries a lag8 leg in the SAME allocation and this pair is
         # what may be quoted. Sign convention: negative = the lean wide halo is FASTER.
         ("lag8", "wide_k8_lean", "vs the DELAYED exchange (same allocation), classic lean"),
         ("lag8", "widediv_k8_lean", "vs the DELAYED exchange (same allocation), div lean")]
REF = "standard"

for base in sorted(runs):
    r = runs[base]
    h = r["hdr"]
    if not h:
        continue
    print(f"\n=== {base}   mesh={h[1]}  {h[2]} nodes / {h[3]} ranks  dt={h[4]}  {h[5]} steps ===")
    ref_s = r["clean"].get(REF) or r["clean"].get("classic")
    ref_i = r["phst"].get(REF) or r["phst"].get("classic")
    print(f"    reference: step {ref_s if ref_s else '--'} s   ice cost "
          f"{f'{ref_i:.3f} ms' if ref_i else '--'}")
    print(f"    {'leg':<20} {'step s':>9} {'% step':>9} {'ice ms':>9} {'% ice':>9}")
    for lg in sorted(set(list(r['clean']) + list(r['phst']))):
        s, i = r["clean"].get(lg), r["phst"].get(lg)
        ds = 100.0 * (s - ref_s) / ref_s if (s and ref_s) else None
        di = 100.0 * (i - ref_i) / ref_i if (i and ref_i) else None
        print(f"    {lg:<20} {s if s else float('nan'):>9.4f} "
              f"{ds if ds is not None else float('nan'):>+9.2f} "
              f"{i if i else float('nan'):>9.3f} "
              f"{di if di is not None else float('nan'):>+9.2f}")
    print("    -- LEAN vs EXACT at matched K (the ring computation is common to both) --")
    for ex, ln, lab in PAIRS:
        se, sl = r["clean"].get(ex), r["clean"].get(ln)
        ie, il = r["phst"].get(ex), r["phst"].get(ln)
        ds = 100.0 * (sl - se) / se if (se and sl) else float("nan")
        di = 100.0 * (il - ie) / ie if (ie and il) else float("nan")
        print(f"    {lab:<26} step {ds:+7.2f} %   ice cost {di:+7.2f} %")
