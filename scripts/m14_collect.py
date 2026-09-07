#!/usr/bin/env python3
"""M14/M16 ladder collector — ONE implementation for both ladders (extracted verbatim from the inline
python at jobs/job_m14_ladder_cpu:171 / _gpu, Task E1). Prints the per-arm leg list, min, spread,
the GAIN line for a base/best pair, and the self-describing CSV row.

  m14_collect.py <legs.txt> <POINT> <N> <BASE_KNOBS> <WSPLIT_TAG> [CSV|GCSV] [PREC]

PREC (M16): when given, `prec=<PREC>` is appended to cfg= so a dp row can never be compared against
an sp row; absent ⇒ the M14 output is reproduced byte-for-byte. A dp/sp arm pair also prints
`SP/DP = <ratio>` (sp min over dp min)."""
import sys, collections
d = collections.OrderedDict()
for ln in open(sys.argv[1]):
    k, v = ln.split(); d.setdefault(k, []).append(float(v))
if not d: print("NO ADMITTED LEGS"); raise SystemExit(1)
for k, v in d.items():
    sp = (max(v) - min(v)) / min(v) * 100
    print(f"  {k:5s} legs {v}  min={min(v):.4f}  spread={sp:.2f}%")
if 'base' in d and 'best' in d:
    b, x = min(d['base']), min(d['best'])
    print(f"  GAIN = {(x-b)/b*100:+.2f}%  (best vs base, min over legs)")
if 'dp' in d and 'sp' in d:
    b, x = min(d['dp']), min(d['sp'])
    print(f"  SP/DP = {x/b:.4f}  ({(x-b)/b*100:+.2f}%, sp vs dp, min over legs, equal leg counts: {len(d['dp'])}/{len(d['sp'])})")
# The CSV must be SELF-DESCRIBING. Two rows for the same mesh+ranks are not duplicates if the
# shared configuration differs: farc 4096 measured 0.0686 on the paper config (linfs+std EVP) and
# 0.0626 on the production config (zstar+mEVP) -- 8.7% apart, and indistinguishable without this.
cfg = (sys.argv[4] or "paper").replace(" ", "+") if len(sys.argv) > 4 else "paper"
cfg += "+" + sys.argv[5] if len(sys.argv) > 5 else "+wsplit?"
tag = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else "CSV"
if len(sys.argv) > 7 and sys.argv[7]: cfg += "+prec=" + sys.argv[7]
print(f"{tag} {sys.argv[2]},{sys.argv[3]},cfg={cfg}," + ",".join(f"{k}={min(v):.4f}" for k, v in d.items()))
