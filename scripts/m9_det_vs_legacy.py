#!/usr/bin/env python3
"""Did the deterministic cold-start hole fill move any M9 conclusion?

Pairs each det run with its legacy twin and reports the two things that matter, in the two
currencies the campaign quotes:

  ABSOLUTE   s/model step and s/step on the ice, det vs legacy. These are EXPECTED to move.
             det integrates a different (better-posed) ocean, so CG iteration counts and the
             ice state differ. A large shift here is information, not a problem.

  VERDICT    each scheme's % against the `standard` leg IN THE SAME RUN -- the number the
             report actually publishes. This is what has to survive. Both legs of an A/B share
             one partition and one IC, so a verdict that moves is telling you the scheme's
             payoff genuinely depends on the ocean state it runs on.

🔴 The scaling figures are the exposed result, not these ratios: rank count is the x-axis, so
   under the legacy fill every point on a curve started from a DIFFERENT ocean. Read the
   per-point verdict shifts here, then look at whether the CURVE SHAPE claim survives.

Tag pairing: scd_* <-> sc_* (scaling fleet), op6_* <-> p6_* (operating points).

Conventions kept from docs/report/make_scaling_figs.py, which is the house parser:
  - the model step comes from the CLEAN legs (PHASESTATS perturbs)
  - the ice cost comes from the INSTRUMENTED twin, rep 1 (the paired rep-1 rule)

usage: m9_det_vs_legacy.py [--root /work/ab0995/a270088/port2/m9] [--currency step|ice|both]
"""
import argparse, glob, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/ab0995/a270088/port2/m9")
ap.add_argument("--currency", default="both", choices=("step", "ice", "both"))
args = ap.parse_args()

HDR = re.compile(r"^=== M9 (?:CPU|GPU) A/B\s+TAG=(\S+)\s+mesh=(\S+)\s+nodes=(\d+)\s+ntasks=(\d+)")
LEG = re.compile(r"^LEG\s+(\S+)\s+min\s+([0-9.]+)\s+s/step")
PHASE = re.compile(r"\[phasestats\]\s+(\w+)\s+\|\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|"
                   r"\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|\s+([0-9.]+)")
ICE = ("ice", "icedyn", "iceadv")

# det tag -> legacy tag
def legacy_of(tag):
    if tag.startswith("scd_"):
        return "sc_" + tag[4:]
    if tag.startswith("op6_"):
        return "p6_" + tag[4:]
    return None


def ice_cost(legdir):
    """ms/step over the three ice phases, busy + wait, from rep 1 (the paired rule)."""
    tot, seen = 0.0, False
    try:
        with open(os.path.join(legdir, "run.1.log"), errors="replace") as f:
            for ln in f:
                m = PHASE.search(ln)
                if m and m.group(1) in ICE:
                    seen = True
                    tot += float(m.group(3)) + float(m.group(6))
    except OSError:
        return None
    return tot if seen else None


# tag -> {"step": {leg: s}, "ice": {leg: s}, "det": bool}
runs = {}
for out in sorted(glob.glob(os.path.join(args.root, "ab*.out"))):
    tag, legs, ic = None, {}, "legacy"
    with open(out, errors="replace") as f:
        for ln in f:
            m = HDR.match(ln)
            if m:
                tag = m.group(1)
            if ln.startswith("    ic     = FESOM_IC_EXTRAP="):
                ic = ln.split("=", 2)[2].strip()
            m = LEG.match(ln)
            if m:
                legs[m.group(1)] = float(m.group(2))
    if not tag:
        continue
    base = tag[:-5] if tag.endswith("_phst") else tag
    r = runs.setdefault(base, {"step": {}, "ice": {}, "ic": ic})
    if tag.endswith("_phst"):
        for lg in legs:
            c = ice_cost(os.path.join(args.root, tag, lg))
            if c is not None:
                r["ice"][lg] = c
    else:
        r["step"].update(legs)
        r["ic"] = ic          # the clean leg's header is the authoritative one


def pct(new, old):
    return 100.0 * (new - old) / old


def verdict(d, ref="standard"):
    """{leg: % against the standard leg of the same run}."""
    if ref not in d:
        return {}
    return {k: pct(v, d[ref]) for k, v in d.items() if k != ref}


rows, shifts = [], []
for dtag in sorted(runs):
    ltag = legacy_of(dtag)
    if not ltag or ltag not in runs:
        continue
    D, L = runs[dtag], runs[ltag]
    if D["ic"] != "det":
        print(f"!! {dtag} header says ic={D['ic']} — the knob did not reach the run", file=sys.stderr)
    for cur in (("step", "ice") if args.currency == "both" else (args.currency,)):
        if not D[cur] or not L[cur]:
            continue
        vd, vl = verdict(D[cur]), verdict(L[cur])
        std = (D[cur].get("standard"), L[cur].get("standard"))
        rows.append((dtag, cur, std, sorted(set(vd) & set(vl)), vd, vl))
        for leg in set(vd) & set(vl):
            shifts.append((abs(vd[leg] - vl[leg]), dtag, cur, leg, vl[leg], vd[leg]))

if not rows:
    sys.exit("no det/legacy pairs found — has the det fleet landed?")

unit = {"step": "s/step", "ice": "ms/step"}
for dtag, cur, std, legs, vd, vl in rows:
    d0, l0 = std
    head = f"{dtag:<26} {cur:<4}  standard {l0:.4g} -> {d0:.4g} {unit[cur]} ({pct(d0, l0):+.1f} %)" \
        if (d0 and l0) else f"{dtag:<26} {cur:<4}"
    print(head)
    for leg in legs:
        flip = "  SIGN FLIP" if vl[leg] * vd[leg] < 0 else ""
        print(f"       {leg:<20} legacy {vl[leg]:+7.2f} %   det {vd[leg]:+7.2f} %"
              f"   shift {vd[leg]-vl[leg]:+6.2f} pp{flip}")

print()
shifts.sort(reverse=True)
print("=== largest verdict shifts (percentage points) ===")
for s, dtag, cur, leg, a, b in shifts[:12]:
    print(f"  {s:6.2f} pp   {dtag} [{cur}] {leg}: {a:+.2f} -> {b:+.2f}")
flips = [x for x in shifts if x[4] * x[5] < 0]
print(f"\n{len(rows)} paired points · {len(shifts)} verdicts compared · "
      f"{len(flips)} changed sign · max shift {shifts[0][0]:.2f} pp")
