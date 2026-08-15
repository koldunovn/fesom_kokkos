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

# M13: which climatology hole-fill the run used. Jobs older than 2026-08-14 have no such line and
# are legacy by construction. This column must exist, because under the legacy fill two partitions
# of one mesh start from DIFFERENT initial conditions (measured: 27 PSU at CORE2 512), so a legacy
# row and a det row are not comparable even at the same point on the same day.
ICLINE = re.compile(r"FESOM_IC_EXTRAP=(\w+)")

# The sandbox directory name identifies the mesh; the zoo path carries it one level deeper. A few
# session-2 arms point at the PRIVATE mesh tree (/port2/mesh/...) rather than the M11 sandbox, so
# match that too and flag it — an arm whose mesh files come from a different tree is not covered
# by the race job's "only the partition differs" md5 check.
MESHNAME = re.compile(
    r"/(mesh_m11|mesh)/(?:zoo/)?([a-z0-9]+?)(?:_m11|_base|_seed\d*|_hil|_rcm|_wgt\d*)?(?:/|$)")


def mesh_and_tree(path):
    m = MESHNAME.search(path)
    if not m:
        return "?", "?"
    return m.group(2), ("sandbox" if m.group(1) == "mesh_m11" else "private")


# The cold-start ladder timestep per mesh. A race at any other dt is not a protocol run: three of
# this campaign's "failures" were a production dt applied to a cold start, and one of them cost
# the baseline itself (Finding 33). Rows at the wrong dt are kept but flagged, never silently
# folded into a verdict.
LADDER_DT = {"core2": 1800, "farc": 900, "dars": 120, "ng5": 180}


def base_mesh(name):
    """core2hil and core2 share a ladder; the suffix marks the numbering, not the mesh."""
    for k in LADDER_DT:
        if name.startswith(k):
            return k
    return name


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

    ic = "legacy"
    for line in txt[:12]:
        m = ICLINE.search(line)
        if m:
            ic = m.group(1)
            break

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
        mname, tree = mesh_and_tree(meshes.get(a, ""))
        out.append(dict(job=job, backend=backend, mesh=mname, tree=tree, ic=ic, ranks=h["ranks"],
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
    ap.add_argument("--best", action="store_true",
                    help="per (mesh, backend, ranks): the best arm among PROTOCOL runs only")
    ap.add_argument("--ic", choices=["det", "legacy"],
                    help="keep only rows run with this climatology hole-fill (M13). Without it, "
                         "--best reports det and legacy separately rather than mixing them.")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{a.dir}/racepart*.out")):
        rows += parse(p)
    if a.mesh:
        rows = [r for r in rows if r["mesh"] == a.mesh]
    if a.backend:
        rows = [r for r in rows if r["backend"] == a.backend]
    if a.ic:
        rows = [r for r in rows if r["ic"] == a.ic]
    if a.min_steps:
        rows = [r for r in rows if r["steps"] >= a.min_steps]
    if not rows:
        sys.exit("no race rows matched")

    for r in rows:
        r["ladder_dt"] = LADDER_DT.get(base_mesh(r["mesh"]))
        r["protocol"] = (r["ladder_dt"] is None or r["dt"] == r["ladder_dt"])

    if a.best:
        pts, off = {}, 0
        for r in rows:
            if not r["protocol"]:
                off += 1
                continue
            if r["is_base"] or r["status"] != "ok":
                continue
            k = (base_mesh(r["mesh"]), r["backend"], r["ranks"], r["ic"])
            if k not in pts or r["delta_pct"] < pts[k]["delta_pct"]:
                pts[k] = r
        print(f"  {'mesh':<8}{'backend':<9}{'ranks':>7} {'IC':<8} {'best arm':<20}"
              f"{'gain':>9}{'steps':>7}  job")
        for k in sorted(pts, key=lambda k: (k[3], k[1], k[0], k[2])):
            r = pts[k]
            num = "" if r["mesh"] == k[0] else f"  [{r['mesh']}]"
            if r["tree"] != "sandbox":
                num += f"  !! mesh files from the {r['tree']} tree"
            print(f"  {k[0]:<8}{k[1]:<9}{k[2]:>7} {k[3]:<8} {r['arm']:<20}{r['delta_pct']:>8.2f}%"
                  f"{r['steps']:>7}  {r['job']}{num}")
        print(f"\n  {off} row(s) excluded: raced at a dt other than the mesh's cold-start ladder dt "
              f"({', '.join(f'{m}={d}' for m, d in LADDER_DT.items())}).")
        return 0

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
