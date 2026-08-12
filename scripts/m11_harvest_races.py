#!/usr/bin/env python3
"""M11: harvest every partition race into one tidy table.

The campaign accumulated its numbers one job at a time, and by the end the verdict has to be
assembled from ~30 job outputs spread over several days. Reading them by hand is how a retracted
headline happens, so this parses them instead.

One row per (job, arm): the mesh, the backend, the rank count, the timestep, the step count, the
min-of-N s/step and the delta against that job's OWN baseline. Cross-job comparison is left to
the caller and should be resisted — the project's standing rule is that a ratio is only earned on
a matched same-day pair (L88), and the `job` column is what lets a reader check that.

    m11_harvest_races.py [--dir /work/.../m11] [--csv out.csv] [--mesh core2] [--backend gpu]

Arms that failed are kept with s_step = NaN and a `status` of FAILED, because a partition that
does not run is a result — three of this campaign's findings are about exactly that.
"""
import argparse
import csv
import glob
import os
import re
import sys

HDR = re.compile(r"=== M11 partition race\s+ranks=(\d+)\s+nodes=(\d+)\s+dt=(\d+)\s+steps=(\d+)\s+reps=(\d+)")
MESHLINE = re.compile(r"^\s{4}(\S+)\s+(/work/\S+)\s*$")
SUMROW = re.compile(r"^\s{2}(\S+)\s+([\d.]+)\s+([\d.]+)%\s*([-+][\d.]+ %)?\s*$")
FAILROW = re.compile(r"^\s{2}(\S+)\s+FAILED\s*$")
REPROW = re.compile(r"^\s{2}(\S+)\s+rep(\d+)\s+rc=(\d+)\s+s/step=(\S+)")

# The sandbox directory name identifies the mesh; the zoo path carries it one level deeper.
MESHNAME = re.compile(r"/mesh_m11/(?:zoo/)?([a-z0-9]+?)(?:_m11|_base|_seed\d*|_hil|_rcm)?(?:/|$)")


def mesh_of(path):
    m = MESHNAME.search(path)
    return m.group(1) if m else "?"


def parse(path):
    txt = open(path, errors="ignore").read().splitlines()
    h = None
    for line in txt:
        m = HDR.search(line)
        if m:
            h = dict(zip(("ranks", "nodes", "dt", "steps", "reps"), (int(x) for x in m.groups())))
            break
    if h is None:
        return []
    job = re.search(r"\.(\d+)\.out$", path)
    job = job.group(1) if job else os.path.basename(path)
    backend = "gpu" if "racepartgpu" in os.path.basename(path) else "cpu"

    meshes, order = {}, []
    for line in txt:
        m = MESHLINE.match(line)
        if m and m.group(1) not in meshes:
            meshes[m.group(1)] = m.group(2)
            order.append(m.group(1))

    # Prefer the summary block; fall back to the per-rep lines when a job died before it.
    rows, in_sum = {}, False
    for line in txt:
        if line.startswith("=== min-of-"):
            in_sum = True
            continue
        if in_sum:
            if line.startswith("  spread =") or line.startswith("==="):
                in_sum = False
                continue
            m = FAILROW.match(line)
            if m:
                rows[m.group(1)] = (float("nan"), "FAILED")
                continue
            m = SUMROW.match(line)
            if m and m.group(1) != "arm":
                rows[m.group(1)] = (float(m.group(2)), "ok")
    if not rows:
        for line in txt:
            m = REPROW.match(line)
            if m:
                a, t = m.group(1), m.group(4)
                v = float(t) if t != "FAILED" else float("nan")
                if a not in rows or (v == v and v < rows[a][0]):
                    rows[a] = (v, "ok" if v == v else "FAILED")

    base = order[0] if order else (list(rows) or [None])[0]
    b = rows.get(base, (float("nan"), ""))[0]
    out = []
    for a in order or rows:
        if a not in rows:
            continue
        t, st = rows[a]
        d = 100 * (t / b - 1) if (t == t and b == b and b) else float("nan")
        out.append(dict(job=job, backend=backend, mesh=mesh_of(meshes.get(a, "")), ranks=h["ranks"],
                        nodes=h["nodes"], dt=h["dt"], steps=h["steps"], reps=h["reps"],
                        arm=a, is_base=(a == base), s_step=t, delta_pct=d, status=st,
                        meshdir=meshes.get(a, "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/ab0995/a270088/port2/m11")
    ap.add_argument("--csv")
    ap.add_argument("--mesh")
    ap.add_argument("--backend", choices=["cpu", "gpu"])
    ap.add_argument("--min-steps", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{a.dir}/racepart*.out")):
        rows += parse(p)
    if a.mesh:
        rows = [r for r in rows if r["mesh"] == a.mesh]
    if a.backend:
        rows = [r for r in rows if r["backend"] == a.backend]
    if a.min_steps:
        rows = [r for r in rows if r["steps"] >= a.min_steps]
    if not rows:
        sys.exit("no race rows matched")

    key = lambda r: (r["mesh"], r["backend"], r["ranks"], r["job"], not r["is_base"])
    rows.sort(key=key)
    cur = None
    for r in rows:
        k = (r["mesh"], r["backend"], r["ranks"])
        if k != cur:
            cur = k
            print(f"\n--- {r['mesh']}  {r['backend'].upper()}  {r['ranks']} ranks")
            print(f"  {'job':>9} {'dt':>5} {'steps':>6} {'reps':>5}  {'arm':<18}{'s/step':>10}{'vs base':>10}")
        s = f"{r['s_step']:.4f}" if r["s_step"] == r["s_step"] else "FAILED"
        d = f"{r['delta_pct']:+.2f} %" if r["delta_pct"] == r["delta_pct"] and not r["is_base"] else ""
        print(f"  {r['job']:>9} {r['dt']:>5} {r['steps']:>6} {r['reps']:>5}  {r['arm']:<18}{s:>10}{d:>10}")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
