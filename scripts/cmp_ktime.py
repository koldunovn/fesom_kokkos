#!/usr/bin/env python3
"""Side-by-side kernel comparison of two kp_ktime CSVs (LUMI HIP vs MN5 CUDA).

    python3 cmp_ktime.py <lumi.csv> <mn5.csv>

Both files come from the same connector, same mesh, same rank count and step count, so the
shares are directly comparable. The point of the diff is to separate two possibilities:

  * a few named kernels are disproportionately expensive on one backend  -> targeted work
  * the profiles are proportionally alike                                -> broad architectural
                                                                            fit, a much bigger job

Kernel times are fenced (see kp_ktime.cpp), so treat the percentages as the signal and the
absolute milliseconds as instrument-inflated.
"""
import csv
import re
import sys
from collections import defaultdict

# Coarse functional groups, first match wins. Tuned to the FESOM Kokkos kernel names.
GROUPS = [
    ("tracer FCT advection", r"tracer.*(fct|advect)|advect.*tracer|fct_"),
    ("halo pack/unpack",     r"halo"),
    ("SSH CG solver",        r"ssh_solve_cg|cg_kk|cgpipe|ssh.*cg"),
    ("horizontal diffusion", r"diff_part_hor|redi|visc_filt|biharm|bidiff"),
    ("vertical velocity",    r"vert_vel|ale_vert"),
    ("pressure / density",   r"pressure|pgf|rho|dens|hydrostat"),
    ("vertical mixing/KPP",  r"kpp|vert_mix|_kv|_av\b"),
    ("sea ice",              r"ice|evp"),
    ("forcing / bulk",       r"forcing|bulk|flux"),
    ("momentum",             r"mom|uv_|velocity"),
]


def shorten(name):
    n = name.strip().strip('"')
    n = re.sub(r"Kokkos::", "", n)
    n = re.sub(r"<[^<>]*>", "", n)
    while re.search(r"<[^<>]*>", n):
        n = re.sub(r"<[^<>]*>", "", n)
    n = re.sub(r"\s+", " ", n)
    return n[:58]


def group_of(name):
    low = name.lower()
    for label, pat in GROUPS:
        if re.search(pat, low):
            return label
    return "other"


def load(path):
    rows, meta = [], ""
    with open(path) as fh:
        lines = [l for l in fh]
    for l in lines:
        if l.startswith("#") and "backend=" in l:
            meta = l.strip("# \n")
    body = [l for l in lines if not l.startswith("#")]
    for r in csv.DictReader(body):
        try:
            rows.append(
                {
                    "name": shorten(r["kernel"]),
                    "pct": float(r["pct"]),
                    "calls": int(r["calls"]),
                    "ms": float(r["total_ms"]),
                    "avg_us": float(r["avg_us"]),
                }
            )
        except (KeyError, ValueError):
            continue
    return rows, meta


def agg(rows):
    g = defaultdict(lambda: {"pct": 0.0, "ms": 0.0, "calls": 0})
    for r in rows:
        k = group_of(r["name"])
        g[k]["pct"] += r["pct"]
        g[k]["ms"] += r["ms"]
        g[k]["calls"] += r["calls"]
    return g


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    ra, ma = load(a)
    rb, mb = load(b)
    print(f"LUMI : {ma}\n       {a}")
    print(f"MN5  : {mb}\n       {b}\n")

    ga, gb = agg(ra), agg(rb)
    keys = sorted(set(ga) | set(gb), key=lambda k: -(ga.get(k, {}).get("pct", 0)))
    print("=" * 78)
    print("BY FUNCTIONAL GROUP (share of fenced device time)")
    print("=" * 78)
    print(f"{'group':<24}{'LUMI %':>9}{'MN5 %':>9}{'LUMI ms':>11}{'MN5 ms':>11}{'ms ratio':>10}")
    for k in keys:
        pa = ga.get(k, {}).get("pct", 0.0)
        pb = gb.get(k, {}).get("pct", 0.0)
        sa = ga.get(k, {}).get("ms", 0.0)
        sb = gb.get(k, {}).get("ms", 0.0)
        ratio = f"{sa / sb:.2f}x" if sb > 0 else "-"
        print(f"{k:<24}{pa:>9.2f}{pb:>9.2f}{sa:>11.1f}{sb:>11.1f}{ratio:>10}")

    ta = sum(v["ms"] for v in ga.values())
    tb = sum(v["ms"] for v in gb.values())
    print(f"{'TOTAL':<24}{100.0:>9.2f}{100.0:>9.2f}{ta:>11.1f}{tb:>11.1f}"
          f"{(ta / tb if tb else 0):>9.2f}x")

    print("\n" + "=" * 78)
    print("TOP KERNELS, PAIRED BY NAME (sorted by LUMI cost)")
    print("=" * 78)
    bmap = defaultdict(lambda: {"ms": 0.0, "pct": 0.0, "avg_us": 0.0, "calls": 0})
    for r in rb:
        e = bmap[r["name"]]
        e["ms"] += r["ms"]
        e["pct"] += r["pct"]
        e["calls"] += r["calls"]
        e["avg_us"] = max(e["avg_us"], r["avg_us"])
    amap = defaultdict(lambda: {"ms": 0.0, "pct": 0.0, "avg_us": 0.0, "calls": 0})
    for r in ra:
        e = amap[r["name"]]
        e["ms"] += r["ms"]
        e["pct"] += r["pct"]
        e["calls"] += r["calls"]
        e["avg_us"] = max(e["avg_us"], r["avg_us"])

    print(f"{'LUMI%':>7}{'MN5%':>7}{'LUMIms':>9}{'MN5ms':>9}{'ratio':>8}  kernel")
    for name, e in sorted(amap.items(), key=lambda kv: -kv[1]["ms"])[:25]:
        f = bmap.get(name)
        if f:
            ratio = f"{e['ms'] / f['ms']:.2f}x" if f["ms"] > 0 else "-"
            print(f"{e['pct']:>7.2f}{f['pct']:>7.2f}{e['ms']:>9.1f}{f['ms']:>9.1f}"
                  f"{ratio:>8}  {name}")
        else:
            print(f"{e['pct']:>7.2f}{'--':>7}{e['ms']:>9.1f}{'--':>9}{'--':>8}  {name}  "
                  f"[not in MN5 profile]")

    only_b = [(n, e) for n, e in bmap.items() if n not in amap]
    if only_b:
        print("\nkernels present only in the MN5 profile:")
        for n, e in sorted(only_b, key=lambda kv: -kv[1]["ms"])[:8]:
            print(f"{'--':>7}{e['pct']:>7.2f}{'--':>9}{e['ms']:>9.1f}{'--':>8}  {n}")

    print("\nSmall-kernel tail (launch-overhead indicator):")
    for tag, rows in (("LUMI", ra), ("MN5", rb)):
        small = [r for r in rows if r["avg_us"] < 20]
        tot = sum(r["ms"] for r in rows)
        sms = sum(r["ms"] for r in small)
        scl = sum(r["calls"] for r in small)
        acl = sum(r["calls"] for r in rows)
        print(f"  {tag}: {len(small)}/{len(rows)} kernels avg<20us — "
              f"{(100 * sms / tot if tot else 0):.1f}% of time, "
              f"{(100 * scl / acl if acl else 0):.1f}% of launches")


if __name__ == "__main__":
    main()
