#!/usr/bin/env python3
"""M9 — cross-mesh report for the EXACT wide halo (cells ②/④) against cell ⑤ and classic.

Answers the question the session-2 CORE2 numbers raised: cells ②/④ are exact but cost ghost
work, so *where* does the trade come out positive? The explanatory variable is not rank count
and not mesh size — it is the **ghost/owned ratio**, i.e. how big the K-ring zone is relative to
the rank's own subdomain. That ratio is ~ K·perimeter/area, so it blows up exactly where strong
scaling has made the subdomain small, which is also where the exchange saving is largest. The
two effects fight, and this table is how you see which one wins.

It reads the A/B job outputs directly (not `m9_results.json`) because the number that explains
everything — the wide zone's size — is printed by `evpw_build` into the per-leg run log and is
not part of the timing record.

usage: m9_wide_report.py [--root /work/ab0995/a270088/port2/m9] [--tags f3_]
"""
import argparse
import glob
import os
import re

# 2-D node counts, for owned-nodes-per-rank. Keep in step with the mesh table in the docs.
NOD2D = {"core2": 126858, "farc": 638387, "fArc_sorted": 638387,
         "dars": 3160340, "ng5_w3d": 7402886, "ng5": 7402886}

HDR = re.compile(r"^=== M9 (?:CPU|GPU) A/B\s+TAG=(\S+)\s+mesh=(\S+)\s+nodes=(\d+)\s+ntasks=(\d+)"
                 r"\s+dt=(\d+)\s+steps=(\d+)")
LEG = re.compile(r"^LEG\s+(\S+)\s+min\s+([0-9.]+)\s+s/step(?:\s+([+-][0-9.]+)%)?")
MODE = re.compile(r"^\s+mode\s+= (\S+)")
ZONE = re.compile(r"upd-slots\(max\)=(\d+)\s+ghost-elems\(max\)=(\d+)")
PART = re.compile(r"partners\(max\)=(\d+)")


def zone_of(root, tag):
    """{K: (upd_slots, ghost_elems, partners)} from the wide legs' logs under this tag.

    ⚠️ Keyed by K on purpose. The zone is K rings deep, so its size — and therefore the whole
    cost of exactness — scales with K. Reporting one 'ratio' per run would silently quote
    whichever K happened to be found first, which is exactly the kind of number that ends up in
    a table meaning something different from what the reader assumes."""
    out = {}
    for leg in ("wide2_k2", "wide4_k2", "wide2_k4", "wide4_k4", "wide2_k8", "wide4_k8"):
        K = int(leg.split("_k")[1])
        if K in out:
            continue
        for p in sorted(glob.glob(os.path.join(root, tag, leg, "run.*.log"))):
            try:
                txt = open(p, "rb").read().decode("utf-8", "replace")
            except OSError:
                continue
            z, q = ZONE.search(txt), PART.search(txt)
            if z:
                out[K] = (int(z.group(1)), int(z.group(2)), int(q.group(1)) if q else 0)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/ab0995/a270088/port2/m9")
    # A PREFIX is not selective enough: session 1's farc ladder is also tagged f3_* (f3_farc512_phst
    # etc.), and mixing two campaigns' rows into one table is how a stale number gets quoted as new.
    # Session 2's tags carry the backend in them (f3_<mesh>_[cg]<ranks>), so match on that.
    ap.add_argument("--tags", default=r"^f[345]_(farc|dars|ng5)_[cg]\d+$",
                    help="regex the TAG must fully match")
    args = ap.parse_args()
    tag_re = re.compile(args.tags)

    runs = []
    for path in sorted(glob.glob(os.path.join(args.root, "ab*.out"))):
        txt = open(path, "rb").read().decode("utf-8", "replace")
        h = HDR.search(txt)
        if not h or not tag_re.match(h.group(1)):
            continue
        tag, mesh, nodes, ntasks, dt, steps = (h.group(1), h.group(2), int(h.group(3)),
                                               int(h.group(4)), int(h.group(5)), int(h.group(6)))
        m = MODE.search(txt)
        instrumented = bool(m and "INSTRUMENTED" in m.group(1))
        legs = {}
        for line in txt.splitlines():
            g = LEG.match(line)
            if g:
                legs[g.group(1)] = (float(g.group(2)),
                                    float(g.group(3)) if g.group(3) else 0.0)
        if not legs:
            continue
        runs.append(dict(tag=tag, mesh=mesh, nodes=nodes, ntasks=ntasks, dt=dt, steps=steps,
                         instrumented=instrumented, legs=legs, zone=zone_of(args.root, tag),
                         jobid=os.path.basename(path).split(".")[1]))

    if not runs:
        print(f"no runs matching tag regex '{args.tags}' under {args.root}")
        return

    order = ["lag2", "lag4", "lag8", "wide2_k2", "wide4_k2", "wide2_k4", "wide4_k4",
             "wide2_k8", "wide4_k8"]
    print("M9 — the exact wide halo (② classic form / ④ divergence form) vs cell ⑤ (lagged)")
    print("Δ of the MODEL STEP vs the classic leg, same allocation, min-of-reps.\n")
    print("  The ghost-zone table below carries the explanatory variable: ghost updatable slots")
    print("  per owned node per rank. A ratio near 1 means the K-ring zone is as big as the")
    print("  rank's own subdomain, i.e. exactness is being paid for in duplicated compute.\n")

    hdr = (f"{'tag':<15}{'mesh':<8}{'bk':<4}{'N':>4}{'ranks':>7}{'nod/rank':>10}"
           f"{'classic':>9}" + "".join(f"{c:>11}" for c in order))
    print(hdr)
    print("-" * len(hdr))
    clean = [r for r in runs if not r["instrumented"]]   # s/step comes from CLEAN legs (protocol)
    for r in sorted(clean, key=lambda x: (x["mesh"],
                                          "GPU" if x["ntasks"] <= 4 * x["nodes"] else "CPU",
                                          x["ntasks"], x["tag"])):
        npr = NOD2D.get(r["mesh"], 0) / r["ntasks"] if r["mesh"] in NOD2D else 0
        bk = "GPU" if r["ntasks"] <= 4 * r["nodes"] else "CPU"
        base = r["legs"].get("classic", (float("nan"), 0))[0]
        row = (f"{r['tag']:<15}{r['mesh']:<8}{bk:<4}{r['nodes']:>4}{r['ntasks']:>7}"
               f"{npr:>10.0f}{base:>9.4f}")
        for c in order:
            row += f"{r['legs'][c][1]:>+10.2f}%" if c in r["legs"] else f"{'-':>11}"
        print(row)

    print("\nGHOST ZONE — the cost of exactness, per K (upd-slots(max) / owned nodes per rank):")
    print(f"{'mesh':<8}{'bk':<4}{'ranks':>7}{'nod/rank':>10}   " +
          "  ".join(f"K={K}: ratio (ghost-elems)" for K in (2, 4, 8)))
    for r in sorted(clean, key=lambda x: (x["mesh"],
                                          "GPU" if x["ntasks"] <= 4 * x["nodes"] else "CPU",
                                          x["ntasks"])):
        if not r["zone"]:
            continue
        npr = NOD2D.get(r["mesh"], 0) / r["ntasks"] if r["mesh"] in NOD2D else 0
        bk = "GPU" if r["ntasks"] <= 4 * r["nodes"] else "CPU"
        cells = []
        for K in (2, 4, 8):
            if K in r["zone"] and npr:
                upd, ge, _ = r["zone"][K]
                cells.append(f"{upd/npr:>8.2f} ({ge:>6d})")
            else:
                cells.append(f"{'-':>8}  {'':>8}")
        print(f"{r['mesh']:<8}{bk:<4}{r['ntasks']:>7}{npr:>10.0f}   " + "  ".join(cells))

    print("\nINSTRUMENTED runs (icedyn phase split — do NOT quote s/step from these):")
    for r in runs:
        if r["instrumented"]:
            print(f"  {r['tag']}  job {r['jobid']}  "
                  + "  ".join(f"{k} {v[1]:+.2f}%" for k, v in r["legs"].items() if k != "classic"))


if __name__ == "__main__":
    main()
