#!/usr/bin/env python3
"""
m7_roofline.py — turn an ncu CSV into the M7 Task-0.4 roofline table.

Input: `ncu -i <rep> --csv --page details` (LONG: one row per (launch, metric), WITH a
"Metric Unit" column) or `--page raw` (WIDE: one column per metric, NO units). Prefer
details — see the unit trap below.

  m7_roofline.py <ncu.csv> [--spill resource_usage.txt] [--peak-dram 1.935e12]

⚠️ THE UNIT TRAP. ncu NORMALISES metric values: dram__bytes.sum.per_second comes back in
"Gbyte/second" (e.g. 700.5), NOT in byte/second. Dividing that by 1e9 to "get GB/s" yields
0.0000007 and prints as 0.0 — a plausible-looking zero next to a perfectly correct
"49.9 %peak". So: scale by the declared unit when it is available, and derive the headline
GB/s from the UNIT-FREE percent-of-peak metric, which cannot be misread.

⚠️ SPILL comes from the BINARY, not from ncu. `cuobjdump --dump-resource-usage` gives the
exact per-thread stack frame of every kernel — free, no GPU, no replay, no unit ambiguity.
Pass it with --spill and it is joined onto the table.

Columns:
  GB/s / %peak   achieved DRAM bandwidth vs the A100's ~1.94 TB/s
  SM% / mem%     compute vs memory speed-of-light
  occ%           achieved occupancy (sm__warps_active)
  regs           registers/thread
  stackB         per-thread stack frame from cuobjdump = register spill / local arrays
  sec/req        sectors per global request. FP64 IDEAL = 8.0 (a 32-thread warp x 8 B =
                 256 B = 8 x 32 B sectors); 32.0 = one sector per thread = fully scattered.
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

PEAK_DRAM = 1.935e12          # A100-80GB HBM2e, byte/s

UNIT = {
    "": 1.0, "%": 1.0, "ratio": 1.0, "register/thread": 1.0,
    "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
    "byte/second": 1.0, "Kbyte/second": 1e3, "Mbyte/second": 1e6,
    "Gbyte/second": 1e9, "Tbyte/second": 1e12,
    "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0,
}

WANT = {
    "dur":    "gpu__time_duration.sum",
    "dram_bs": "dram__bytes.sum.per_second",
    "dram_p": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_p":   "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_p":  "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "occ":    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "regs":   "launch__registers_per_thread",
    "sp_ld":  "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "sp_st":  "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "sec_ld": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "sec_st": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
}


def num(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if not s or s in ("N/A", "n/a", "-", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def tag(name):
    """C++ symbol of the enclosing FESOM function (what ncu actually shows)."""
    for pat in (r"ParallelFor<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"ParallelReduce<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"ParallelScan<(?:.*?::)?([A-Za-z_][A-Za-z0-9_]*)\s*\("):
        m = re.search(pat, name)
        if m:
            return m.group(1)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*_kk)", name)
    return m.group(1) if m else name[:40]


def load_ncu(path):
    """-> {kernel: {metric: [scaled values]}}   (handles both LONG and WIDE)"""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    hdr_i = next((i for i, r in enumerate(rows)
                  if any(c.strip() in ("Kernel Name", "Demangled Name", "Function Name")
                         for c in r)), None)
    if hdr_i is None:
        sys.exit(f"FATAL: no kernel-name column in {path}")
    hdr = [c.strip() for c in rows[hdr_i]]
    body = [r for r in rows[hdr_i + 1:] if len(r) == len(hdr)]

    def col(*names):
        return next((hdr.index(n) for n in names if n in hdr), None)

    kcol = col("Kernel Name", "Demangled Name", "Function Name")
    mn, mv = col("Metric Name"), col("Metric Value")
    mu = col("Metric Unit")

    data = defaultdict(lambda: defaultdict(list))
    if mn is not None and mv is not None:                       # LONG (details): has units
        for r in body:
            v = num(r[mv])
            if v is None or not r[kcol].strip():
                continue
            unit = r[mu].strip() if mu is not None else ""
            data[tag(r[kcol])][r[mn].strip()].append(v * UNIT.get(unit, 1.0))
    else:                                                      # WIDE (raw): unit-blind
        for r in body:
            if not r[kcol].strip():
                continue                                       # ncu's empty units row
            k = tag(r[kcol])
            for j, h in enumerate(hdr):
                v = num(r[j])
                if v is not None:
                    data[k][h].append(v)
    return data


def load_spill(path):
    """cuobjdump --dump-resource-usage -> {kernel: (regs, stack_bytes)}"""
    if not path or not os.path.exists(path):
        return {}
    txt = open(path).read()
    out = {}
    pat = re.compile(r"Function ([^\n:]+):\s*\n\s*REG:(\d+)\s+STACK:(\d+)")
    names = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in pat.finditer(txt)]
    if not names:
        return {}
    dem = subprocess.run(["c++filt"], input="\n".join(n for n, _, _ in names),
                         capture_output=True, text=True).stdout.splitlines()
    for (_, reg, stack), d in zip(names, dem):
        k = tag(d)
        prev = out.get(k, (0, 0))
        out[k] = (max(reg, prev[0]), max(stack, prev[1]))      # worst case per symbol
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--spill", default=None, help="cuobjdump --dump-resource-usage output")
    ap.add_argument("--peak-dram", type=float, default=PEAK_DRAM)
    a = ap.parse_args()

    if not os.path.exists(a.csv) or os.path.getsize(a.csv) == 0:
        sys.exit(f"FATAL: {a.csv} missing or empty — ncu produced no report.")
    data = load_ncu(a.csv)
    if not data:
        sys.exit(f"FATAL: no kernel rows parsed from {a.csv}")
    spill = load_spill(a.spill)

    def g(mm, key):
        vals = mm.get(WANT[key])
        return statistics.mean(vals) if vals else None

    rows = []
    for k, mm in data.items():
        dram_p = g(mm, "dram_p")
        gbs = g(mm, "dram_bs")
        # The percent-of-peak metric is unit-free and cannot be misread — make it the
        # primary and derive GB/s from it. Fall back to the raw rate only if it is absent.
        if dram_p is not None:
            gbs_r = dram_p / 100.0 * a.peak_dram / 1e9
        elif gbs is not None:
            gbs_r = gbs / 1e9
            dram_p = 100.0 * gbs / a.peak_dram
        else:
            gbs_r = None
        sp = spill.get(k)
        rows.append(dict(
            k=k, n=len(next(iter(mm.values()))), dur=g(mm, "dur"),
            gbs=gbs_r, dram_p=dram_p, sm_p=g(mm, "sm_p"), mem_p=g(mm, "mem_p"),
            occ=g(mm, "occ"), regs=(sp[0] if sp else g(mm, "regs")),
            stack=(sp[1] if sp else None), sec=g(mm, "sec_ld"), sec_st=g(mm, "sec_st"),
        ))
    rows.sort(key=lambda r: -(r["dur"] or 0))

    def verdict(r):
        v = []
        if (r["dram_p"] or 0) > 60:
            v.append("DRAM-roofline-bound")
        if (r["stack"] or 0) > 0 and (r["sm_p"] or 0) < 25:
            v.append("SPILL-bound -> T2.3")
        if (r["sec"] or 0) > 16:
            v.append("uncoalesced -> T2.1/2.2")
        if (r["sm_p"] or 0) < 20 and (r["dram_p"] or 0) < 30:
            v.append("latency-bound")
        return ", ".join(v) if v else "balanced"

    def f(x, p=1):
        return "—" if x is None else f"{x:.{p}f}"

    print("=" * 134)
    print(f"M7 ROOFLINE — {os.path.basename(a.csv)}   (A100 peak DRAM {a.peak_dram/1e12:.2f} TB/s; "
          f"ideal sectors/req for FP64 = 8.0)")
    print("=" * 134)
    print(f"{'kernel':<34} {'n':>3} {'GB/s':>7} {'%peak':>6} {'SM%':>6} {'mem%':>6} "
          f"{'occ%':>6} {'regs':>5} {'stackB':>7} {'sec/req':>7}  verdict")
    print("-" * 134)
    for r in rows:
        print(f"{r['k'][:34]:<34} {r['n']:>3} {f(r['gbs'],0):>7} {f(r['dram_p']):>6} "
              f"{f(r['sm_p']):>6} {f(r['mem_p']):>6} {f(r['occ']):>6} {f(r['regs'],0):>5} "
              f"{f(r['stack'],0):>7} {f(r['sec'],1):>7}  {verdict(r)}")
    print("-" * 134)
    print("stackB = per-thread stack frame from cuobjdump (register spill / local arrays).")
    print("sec/req: 8.0 = perfectly coalesced FP64; >16 = the scatter/gather lever (T2.1/2.2).")
    print("=" * 134)


if __name__ == "__main__":
    main()
