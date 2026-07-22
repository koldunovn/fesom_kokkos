#!/usr/bin/env python3
"""Consolidated JUPITER fleet table: s25 twin / s26 device / s26 stage / clang.

Parses RESULT lines from the campaign scratch dirs, takes min-of-2 per
(series, point, leg), computes parallel efficiency (dp, vs the series' smallest
point per mesh) and SYPD at production dt per the campaign reporting rule
(rule 0.41 + measured CG corrections):
  core2: dt 1800, corr 1.0        farc: quoted at dt 1200, corr 1.0
  dars:  dt 240, corr 1.0222      ng5:  dt 240, corr 1.0110
SYPD = dt_prod / (365 * s_step * corr), on the dp_cgp and sp_cgp legs.
Emits GitHub markdown, one table per mesh."""
import glob, re, sys
from collections import defaultdict

B = "/e/scratch/e-sta-destine/koldunov1/port2"
LEGS = ["dp", "dp_cgp", "sp", "sp_cgp"]
PROD = {"core2": (1800, 1.0), "farc": (1200, 1.0), "dars": (240, 1.0222), "ng5": (240, 1.0110)}
SERIES = ["s25", "s26dev", "s26stg", "clang"]
LABEL = {"s25": "twin s25+STAGE", "s26dev": "s26+device", "s26stg": "s26+STAGE", "clang": "s26+device clang"}

std = re.compile(r"RESULT (\w+_g\d+)(_stg)? (dp|dp_cgp|sp|sp_cgp) ([ab]) rc=0 s_step=([\d.]+)")
clg = re.compile(r"RESULT clang (dp|dp_cgp|sp|sp_cgp)_([ab]) rc=0 s_step=([\d.]+)")

acc = defaultdict(list)  # (series, tag, leg) -> [vals]
def feed(path_glob, series, stg_series=None):
    for f in glob.glob(path_glob):
        for line in open(f, errors="ignore"):
            m = std.search(line)
            if m:
                tag, stg, leg, _, v = m.groups()
                s = stg_series if (stg and stg_series) else series
                if stg and not stg_series:
                    continue
                acc[(s, tag, leg)].append(float(v))
                continue
            m = clg.search(line)
            if m and series == "clang":
                leg, _, v = m.groups()
                acc[("clang", "core2_g1", leg)].append(float(v))

feed(B + "/scale/slurm.10*.out", "s25")
feed(B + "/scale26/slurm.*.out", "s26dev", stg_series="s26stg")
feed(B + "/clang_ab/slurm.*.out", "clang")

pts = defaultdict(set)  # mesh -> set of g
for (s, tag, leg) in acc:
    mesh, g = tag.rsplit("_g", 1)
    pts[mesh].add(int(g))

def mn(s, tag, leg):
    v = acc.get((s, tag, leg))
    return min(v) if v else None

for mesh in ["core2", "farc", "dars", "ng5"]:
    if mesh not in pts:
        continue
    dt, corr = PROD[mesh]
    gs = sorted(pts[mesh])
    base = {}  # series -> (g0, dp0) for efficiency
    for s in SERIES:
        for g in gs:
            v = mn(s, f"{mesh}_g{g}", "dp")
            if v:
                base[s] = (g, v)
                break
    print(f"\n### {mesh}  (SYPD quoted at dt={dt}{', CG corr x%.4f' % corr if corr != 1.0 else ''})\n")
    print("| point | series | dp | dp_cgp | sp | sp_cgp | eff(dp) | SYPD dp_cgp | SYPD sp_cgp |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in gs:
        tag = f"{mesh}_g{g}"
        for s in SERIES:
            vals = [mn(s, tag, leg) for leg in LEGS]
            if not any(vals):
                continue
            dp, dpc, sp, spc = vals
            eff = ""
            if dp and s in base:
                g0, v0 = base[s]
                eff = f"{100.0 * (v0 / dp) / (g / g0):.0f}%"
            sy = lambda v: f"{dt / (365.0 * v * corr):.1f}" if v else "—"
            fm = lambda v: f"{v:.4f}" if v else "—"
            print(f"| {tag} | {LABEL[s]} | {fm(dp)} | {fm(dpc)} | {fm(sp)} | {fm(spc)} | {eff} | {sy(dpc)} | {sy(spc)} |")
