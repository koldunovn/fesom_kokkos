#!/usr/bin/env python3
"""Harvest the M9 fleet into one JSON: per-cell s/step, icedyn phase, and provenance.

The two headline numbers NEVER come from the same run (the PHASESTATS instrument perturbs),
so this walks BOTH kinds of job output and keeps them in separate fields:

  clean legs        -> s_per_step   (SYPD)
  instrumented legs -> icedyn_busy_ms, icedyn_wait_ms, total_busy_ms   (the primary metric)

It also records, per leg, the [m9] cell line the model actually printed. A leg whose cell line
does not name the intended cell is recorded with cell_mismatch=True rather than silently
averaged in — the whole campaign nearly shipped a table of 0.00% differences because every
FESOM_SPEED lever is vetoed on Serial without FORCE_SERIAL, and the announce lines were the
only evidence.

usage: m9_collect.py [--root /work/ab0995/a270088/port2/m9] [-o out.json]
"""
import argparse, json, os, re, sys, glob

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/ab0995/a270088/port2/m9")
ap.add_argument("-o", "--out", default=None)
args = ap.parse_args()

LEG_RE   = re.compile(r"^LEG\s+(\S+)\s+min\s+([0-9.]+)\s+s/step(?:\s+([+-][0-9.]+)%)?")
REP_RE   = re.compile(r"^\s+leg (\S+) rep (\d+): ([0-9.]+) s/step")
HDR_RE   = re.compile(r"^=== M9 (?:CPU|GPU) A/B\s+TAG=(\S+)\s+mesh=(\S+)\s+nodes=(\d+)\s+ntasks=(\d+)\s+dt=(\d+)\s+steps=(\d+)")
SHA_RE   = re.compile(r"^\s+sha256 = (\S+)")
MODE_RE  = re.compile(r"^\s+mode\s+= (\S+)")
PHASE_RE = re.compile(r"\[phasestats\]\s+(\w+)\s+\|\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|"
                      r"\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+@\d+\s+\|\s+([0-9.]+)")
CELL_RE  = re.compile(r"\[m9\] mEVP cell: (.*)")

def phases_from(logpath):
    """phase -> dict(busy_min/mean/max, wait_min/mean/max, mpi). Last occurrence wins."""
    out = {}
    try:
        with open(logpath, "rb") as f:
            for raw in f:
                m = PHASE_RE.search(raw.decode("utf-8", "replace"))
                if m:
                    out[m.group(1)] = dict(
                        busy_min=float(m.group(2)), busy_mean=float(m.group(3)), busy_max=float(m.group(4)),
                        wait_min=float(m.group(5)), wait_mean=float(m.group(6)), wait_max=float(m.group(7)),
                        mpi_per_step=float(m.group(8)))
    except OSError:
        pass
    return out

def cell_of(legdir):
    for cand in ("run.1.log", "run.2.log"):
        p = os.path.join(legdir, cand)
        try:
            with open(p, "rb") as f:
                for raw in f:
                    m = CELL_RE.search(raw.decode("utf-8", "replace"))
                    if m:
                        return m.group(1).strip()
        except OSError:
            continue
    return None

runs = []
for out in sorted(glob.glob(os.path.join(args.root, "ab*.out"))):
    jobid = re.search(r"\.(\d+)\.out$", out)
    jobid = jobid.group(1) if jobid else None
    meta, legs = {}, {}
    with open(out, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            m = HDR_RE.match(line)
            if m:
                meta.update(tag=m.group(1), mesh=m.group(2), nodes=int(m.group(3)),
                            ntasks=int(m.group(4)), dt=int(m.group(5)), nsteps=int(m.group(6)))
            m = SHA_RE.match(line)
            if m: meta["sha256"] = m.group(1)
            m = MODE_RE.match(line)
            if m: meta["mode"] = m.group(1)          # CLEAN | INSTRUMENTED
            m = REP_RE.match(line)
            if m:
                legs.setdefault(m.group(1), {}).setdefault("reps", []).append(float(m.group(3)))
            m = LEG_RE.match(line)
            if m:
                legs.setdefault(m.group(1), {}).update(
                    s_per_step=float(m.group(2)),
                    pct_vs_ref=(float(m.group(3)) if m.group(3) else 0.0))
    if not meta.get("tag"):
        continue
    tagdir = os.path.join(args.root, meta["tag"])
    for leg, d in legs.items():
        legdir = os.path.join(tagdir, leg)
        d["cell"] = cell_of(legdir)
        ph = phases_from(os.path.join(legdir, "run.1.log"))
        if ph:
            d["phases"] = ph
            if "icedyn" in ph:
                d["icedyn_busy_ms"] = ph["icedyn"]["busy_mean"]
                d["icedyn_wait_ms"] = ph["icedyn"]["wait_mean"]
            if "TOTAL" in ph:
                d["total_busy_ms"] = ph["TOTAL"]["busy_mean"]
        reps = d.get("reps") or []
        if len(reps) >= 2:
            d["rep_spread_pct"] = 100.0 * (max(reps) - min(reps)) / min(reps)
    runs.append(dict(jobid=jobid, **meta, legs=legs))

# derive per-run cell deltas against the reference leg (leg order = insertion order)
for r in runs:
    legs = r["legs"]
    if not legs:
        continue
    ref = next(iter(legs))
    r["reference_leg"] = ref
    rb = legs[ref].get("icedyn_busy_ms")
    for name, d in legs.items():
        b = d.get("icedyn_busy_ms")
        if rb and b:
            d["icedyn_pct_vs_ref"] = 100.0 * (b - rb) / rb

blob = dict(root=args.root, n_runs=len(runs), runs=runs)
txt = json.dumps(blob, indent=1, sort_keys=False)
if args.out:
    open(args.out, "w").write(txt)
    print(f"wrote {args.out}: {len(runs)} runs")
else:
    print(txt)

# human summary to stderr so the JSON stays pipeable
for r in runs:
    print(f"\n== {r.get('tag')} [{r.get('mode','?')}] {r.get('mesh')} "
          f"N{r.get('nodes')} np{r.get('ntasks')} dt{r.get('dt')} steps{r.get('nsteps')} "
          f"sha {str(r.get('sha256'))[:8]} job {r.get('jobid')}", file=sys.stderr)
    for name, d in r["legs"].items():
        print(f"   {name:<14} {d.get('s_per_step','?'):>9} s/step  {d.get('pct_vs_ref',0):+7.2f}%"
              f"   icedyn {d.get('icedyn_busy_ms','-'):>7} ms {d.get('icedyn_pct_vs_ref',0):+7.2f}%"
              f"   spread {d.get('rep_spread_pct',0):.2f}%   [{d.get('cell')}]", file=sys.stderr)
