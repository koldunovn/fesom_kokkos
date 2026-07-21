#!/usr/bin/env python3
"""M8 scaling figures: SP vs FP64 on dars (dt120, 300 steps, min-of-2, knobs-off).

Parses /work/.../mp/gate2/scal_{c<N>,g<N>}_{dp,sp}/log_rep_{a,b}.txt, takes min-of-2
s/step per leg, asserts the precision banner per leg (L80), and draws:
  (1) s/step vs nodes (log-log), CPU + GPU, both dtypes;
  (2) SP speedup (dp/sp) vs nodes.
Continuity anchor (annotated, CPU only): the M7 s15 knob-free FP64 dt120 curve
5.947/3.031/1.583/0.839/0.413/0.201 (c1..c32) — same physics commit, different day
(cross-campaign class, not a same-day pair; the in-figure ratios ARE same-day).

Usage: mp_scaling_fig.py [--base /work/ab0995/a270088/port2/mp/gate2] [--out DIR]
"""
import argparse
import pathlib
import re

M7_CPU = {1: 5.947, 2: 3.031, 4: 1.583, 8: 0.839, 16: 0.413, 32: 0.201}


def leg(base: pathlib.Path, tag: str, want_banner: str):
    d = base / tag
    vals = []
    banner_ok = False
    for rep in ("a", "b"):
        p = d / f"log_rep_{rep}.txt"
        if not p.exists():
            continue
        t = p.read_text(errors="ignore")
        if f"PRECISION: {want_banner}" in t:
            banner_ok = True
        m = re.findall(r"loop timing:.*?->\s+([0-9.]+)\s+s/step", t)
        if m:
            vals.append(float(m[-1]))
    if not vals:
        return None
    if not banner_ok:
        raise SystemExit(f"L80 FAIL: no '{want_banner}' banner in {d}")
    return min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=pathlib.Path,
                    default=pathlib.Path("/work/ab0995/a270088/port2/mp/gate2"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("/work/ab0995/a270088/port2/mp/gate3/scaling_figs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fam, sizes in (("c", [1, 2, 4, 8, 16, 32]), ("g", [2, 4, 8])):
        for n in sizes:
            dp = leg(args.base, f"scal_{fam}{n}_dp", "DOUBLE")
            sp = leg(args.base, f"scal_{fam}{n}_sp", "SINGLE")
            if dp is None or sp is None:
                print(f"scal_{fam}{n}: incomplete (dp={dp} sp={sp}) — skipped")
                continue
            rows.append((fam, n, dp, sp))
            anchor = f"  [m7 {M7_CPU[n]:.3f}]" if fam == "c" and n in M7_CPU else ""
            print(f"{fam}{n:<3d} dp={dp:.4f} sp={sp:.4f} speedup={dp/sp:.3f}x{anchor}")

    if not rows:
        raise SystemExit("no complete pairs yet")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for fam, label, marker in (("c", "CPU (128 r/node)", "o"), ("g", "GPU (4/node)", "s")):
        ns = [r[1] for r in rows if r[0] == fam]
        if not ns:
            continue
        dps = [r[2] for r in rows if r[0] == fam]
        sps = [r[3] for r in rows if r[0] == fam]
        ax1.loglog(ns, dps, f"-{marker}", label=f"{label} FP64")
        ax1.loglog(ns, sps, f"--{marker}", label=f"{label} FP32")
        ax2.semilogx(ns, [d / s for d, s in zip(dps, sps)], f"-{marker}", label=label)
    m7n = sorted(M7_CPU)
    ax1.loglog(m7n, [M7_CPU[n] for n in m7n], ":x", color="gray",
               label="m7-s15 FP64 anchor (CPU)")
    ax1.set_xlabel("nodes")
    ax1.set_ylabel("s/step (min-of-2)")
    ax1.set_title("dars dt120, 300 steps, knobs-off")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.axhline(1.0, color="gray", lw=0.5)
    ax2.set_xlabel("nodes")
    ax2.set_ylabel("FP32 speedup (×)")
    ax2.set_title("SP speedup vs scale (same-day pinned pairs)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"mp_scaling.{ext}", dpi=150)
    print(f"wrote {args.out}/mp_scaling.png+pdf")


if __name__ == "__main__":
    main()
