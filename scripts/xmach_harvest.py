#!/usr/bin/env python3
"""xmach_harvest.py — collect the XCSV result lines of the cross-machine ladder into ONE csv.

    python3 scripts/xmach_harvest.py <RUNBASE>/logs [more dirs or files...] -o xmach_<machine>.csv

Every job_xmach job prints exactly one line

    XCSV machine,mesh,backend,nodes,units,unit_kind,dt_s,s_per_step,std_s_per_step,reps,steps,gmredi,source,cfg,binary_md5,job,transport

whose first 13 fields ARE the paper's data/jupiter_vs_lumi.csv schema (the figure scripts read
them by name; the four provenance columns after `source` are extra and harmless). A rung that was
run twice keeps the row with the smaller s_per_step (min over admitted legs, as on Levante) and
records the other job id in `also`. Rows with different `cfg` (wsplit state, transport) are
DIFFERENT configurations and are never merged.

No dependencies beyond the standard library.
"""
import argparse, csv, glob, os, sys

COLS = ["machine", "mesh", "backend", "nodes", "units", "unit_kind", "dt_s", "s_per_step",
        "std_s_per_step", "reps", "steps", "gmredi", "source", "cfg", "binary_md5", "job", "transport"]
MESH_ORDER = {"core2": 0, "farc": 1, "dars": 2, "ng5": 3}

def files(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob.glob(os.path.join(p, "*.out")))
        else:
            yield from sorted(glob.glob(p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="xmach.csv")
    a = ap.parse_args()
    rows = {}
    for f in files(a.paths):
        with open(f, errors="replace") as fh:
            for line in fh:
                if not line.startswith("XCSV "):
                    continue
                vals = line[5:].strip().split(",")
                if len(vals) != len(COLS):
                    print(f"!! malformed XCSV in {f}: {line.strip()}", file=sys.stderr); continue
                r = dict(zip(COLS, vals)); r["also"] = ""
                key = (r["machine"], r["mesh"], r["backend"], r["units"], r["cfg"])
                if key in rows:
                    old = rows[key]
                    keep, drop = (r, old) if float(r["s_per_step"]) < float(old["s_per_step"]) else (old, r)
                    keep["also"] = (keep["also"] + " " + drop["job"] + (" " + drop["also"] if drop["also"] else "")).strip()
                    rows[key] = keep
                else:
                    rows[key] = r
    out = sorted(rows.values(), key=lambda r: (r["machine"], MESH_ORDER.get(r["mesh"], 9), r["backend"], int(r["units"]), r["cfg"]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS + ["also"]); w.writeheader(); w.writerows(out)
    print(f"{len(out)} rows -> {a.out}\n")
    print(f"{'machine':14s} {'mesh':6s} {'be':4s} {'nodes':>5s} {'units':>5s} {'dt':>5s} {'s/step':>8s} {'std':>7s} {'SYPD@dt':>8s}  cfg")
    for r in out:
        s = float(r["s_per_step"]); dt = int(r["dt_s"])
        print(f"{r['machine']:14s} {r['mesh']:6s} {r['backend']:4s} {r['nodes']:>5s} {r['units']:>5s} {dt:5d} {s:8.4f} "
              f"{r['std_s_per_step']:>7s} {dt/(365.0*s):8.2f}  {r['cfg']}")
    print("\nSYPD above is at the MEASURED dt. The paper rescales NG5/dars to the production dt (240 s) "
          "with its own CG corrections — leave that to the paper side; deliver s_per_step.")

if __name__ == "__main__":
    main()
