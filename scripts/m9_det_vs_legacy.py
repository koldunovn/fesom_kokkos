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


SWEEP = re.compile(r"det extrap: (\d+) fill sweeps \+ (\d+) relax sweeps")


def fill_signature_check(root):
    """⭐ The cheapest possible proof that the fleet's ICs really are partition-independent.

    The deterministic fill iterates to a tolerance, so the number of sweeps it needs is a
    property of the MESH and of nothing else. Two points on the same curve run different rank
    counts; if they report different sweep counts, the fill is still seeing the decomposition
    and the whole re-run is void. Under the legacy fill there is no such invariant to check --
    which is exactly why the defect survived so long.

    Returns True if every mesh has exactly one signature.
    """
    # ⚠️ The fill runs TWICE per model start -- once for temperature, once for salinity -- so a
    # run emits two lines and they differ from each other. The invariant is over the PAIR.
    sigs = {}
    for log in glob.glob(os.path.join(root, "scd_*", "*", "run.1.log")) + \
               glob.glob(os.path.join(root, "op6_*", "*", "run.1.log")):
        tag = log.split(os.sep)[-3]
        mesh = re.sub(r"^(scd_(?:gpu|cpu)|op6)_", "", tag)
        mesh = re.sub(r"(_\d+n)?(_phst)?$", "", mesh)
        try:
            with open(log, errors="replace") as f:
                found = [m.group(0) for m in (SWEEP.search(ln) for ln in f) if m]
        except OSError:
            continue
        if found:
            key = " | ".join(found)          # order is T then S, fixed by the loader
            sigs.setdefault(mesh, {}).setdefault(key, set()).add(f"{tag}/{log.split(os.sep)[-2]}")
    ok = True
    print("=== deterministic-fill signature (one per mesh, or the ICs are not shared) ===")
    for mesh in sorted(sigs):
        variants = sigs[mesh]
        n_runs = len(set().union(*variants.values()))
        if len(variants) == 1:
            print(f"  {mesh:<8} OK   {next(iter(variants))}   ({n_runs} runs)")
        else:
            ok = False
            print(f"  {mesh:<8} !! {len(variants)} DIFFERENT signatures across {n_runs} runs:")
            for s, tags in sorted(variants.items()):
                print(f"           {s}   <- {', '.join(sorted(tags)[:4])}"
                      f"{' …' if len(tags) > 4 else ''}")
    print()
    return ok


STEP1 = re.compile(r"^\s+1\s+it=\s*(\d+)\s+uv=(\S+)\s+eta=(\S+)\s+w=(\S+)\s+\|\s+"
                   r"T\[([^\]]*)\] S\[([^\]]*)\].*?hp=(\S+)\s+pgf=(\S+)\s+rho=(\S+)")


def step1_spread(root, prefixes=("sc", "scd")):
    """⭐⭐ The direct before/after test of the premise this whole re-run rests on.

    Every point of a scaling curve runs a different partition. If the cold-start fill is
    partition-dependent, the points do not start from the same ocean -- so compare the step-1
    diagnostic row ACROSS node counts within one mesh and one backend. Every field on that row
    is a global reduction, so under identical initial conditions it must be identical.

    🔴🔴 TWO traps in choosing the field, and both of them produce a confident wrong answer.

    1. The global T and S EXTREMES are useless: the fill differs in the marginal seas, whose
       values sit inside the global range, so T[] and S[] agree to the printed precision across
       every partition even when the ICs differ. Using them says "no problem anywhere".

    2. ⚠️ ON THE CUDA ARM MOST OF THIS ROW IS STALE. The diagnostics are assembled on the host,
       and the device-resident fields are never mirrored back for it: `uv`, `w`, `vs`, `rs`,
       `bv`, `Kv`, `Av` all print as exactly 0.00e+00, and `pgf`, `rho` and `hp` carry
       un-mirrored values -- the same DARS configuration reads pgf=3.00e-04 on Serial and
       8.20e-02 on CUDA. So **pgf is a Serial-only fingerprint.** The fields that do agree
       across backends, and are therefore usable on both, are the CG iteration count and `eta`.

    `pgf` is the most sensitive field where it is valid, because M13's mechanism IS a density
    front across a single element and pgf is what that front produces.
    """
    rows = {}
    for pfx in prefixes:
        for d in sorted(glob.glob(os.path.join(root, f"{pfx}_gpu_*n", "standard")) +
                        glob.glob(os.path.join(root, f"{pfx}_cpu_*n", "standard"))):
            tag = d.split(os.sep)[-2]
            if tag.endswith("_phst") or not re.match(rf"^{pfx}_(gpu|cpu)_\w+_\d+n$", tag):
                continue
            _, backend, mesh, nodes = tag.split("_")
            try:
                with open(os.path.join(d, "run.1.log"), errors="replace") as f:
                    m = next((STEP1.match(ln) for ln in f if STEP1.match(ln)), None)
            except OSError:
                continue
            if m:
                rows.setdefault((pfx, backend, mesh), []).append(
                    (int(nodes[:-1]), float(m.group(8)), float(m.group(3)), int(m.group(1))))

    print("=== step-1 state across the node counts of one curve "
          "(identical ICs => zero spread) ===")
    print("  pgf is Serial-only (stale host mirror on CUDA); eta and CG iters are valid on both.")
    print(f"  {'fleet':<5} {'back':<4} {'mesh':<6} {'pts':>3}  "
          f"{'max|pgf| (SERIAL ONLY)':>26}  {'eta':>7}  {'CG iters':>9}  verdict")
    for key in sorted(rows):
        pfx, backend, mesh = key
        pts = sorted(rows[key])
        if len(pts) < 2:
            continue
        pg, et, it = [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
        efac = max(et) / min(et) if min(et) > 0 else float("inf")
        if backend == "cpu":
            fac = max(pg) / min(pg) if min(pg) > 0 else float("inf")
            cell = f"{min(pg):.2e}-{max(pg):.2e} ({fac:.2f}x)"
        else:
            fac, cell = 1.0, "n/a (CUDA host mirror)"
        # a curve is partition-dependent if ANY field that is valid on this backend varies
        worst = max(fac, efac, (max(it) / min(it)) if min(it) else 1.0)
        flag = "PARTITION-DEPENDENT" if worst > 1.005 else "flat"
        print(f"  {pfx:<5} {backend:<4} {mesh:<6} {len(pts):>3}  {cell:>26}  "
              f"{efac:>6.2f}x  {min(it):>4}-{max(it):<4}  {flag}")
    print()


def pct(new, old):
    return 100.0 * (new - old) / old


def verdict(d, ref="standard"):
    """{leg: % against the standard leg of the same run}."""
    if ref not in d:
        return {}
    return {k: pct(v, d[ref]) for k, v in d.items() if k != ref}


fill_ok = fill_signature_check(args.root)
step1_spread(args.root)

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
if not fill_ok:
    sys.exit("\n!! fill signatures disagree within a mesh — the det fleet does NOT share one IC")
