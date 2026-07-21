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
M7_NG5 = {4: 4.58, 8: 2.35, 16: 1.21, 32: 0.618}   # s15 curve; c32n = dt60 adopted point

# SYPD at PRODUCTION dt — the m7_scaling_figs.py convention exactly:
# SYPD = dt_prod/(365*sstep); DT_PROD core2 1800 / farc 900 / dars 240 / NG5 240.
# s/step is dt-independent, so projecting measurement runs (dars dt120, NG5 dt180/60)
# to production dt is legitimate; CG dt-correction not applied (same footnote as m7).
FAM_DT = {"c": 240.0, "g": 240.0, "n": 240.0, "N": 240.0, "k": 1800.0, "f": 900.0}


def sypd(dt_s: float, s_step: float) -> float:
    return dt_s / (365.0 * s_step)


def node_axis(ax, counts, label="nodes  (4×A100  /  128-core CPU)"):
    """m7 house style: log-base-2 x with PLAIN node-count labels, no minors."""
    import matplotlib.ticker as mtick
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.set_xlabel(label)


def decimal_log_yaxis(ax, lo, hi):
    """m7 house style: log y-scale with PLAIN decimal tick labels (no powers of 10)."""
    import matplotlib.ticker as mtick
    ticks = [t for t in (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10)
             if lo * 0.95 <= t <= hi * 1.05]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_ylim(lo * 0.9, hi * 1.1)


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
    for fam, tagfmt, sizes in (("c", "scal_c{n}", [1, 2, 4, 8, 16, 32]),
                               ("g", "scal_g{n}", [2, 4, 8]),
                               ("n", "scal_ng5_c{n}", [4, 8, 16, 32]),
                               ("N", "scal_ng5_g{n}", [4, 8, 16, 32]),
                               ("k", "scal_core2_c{n}", [1, 2, 4]),
                               ("f", "scal_farc_c{n}", [1, 2, 4, 8])):
        for n in sizes:
            tag = tagfmt.format(n=n)
            dp = leg(args.base, f"{tag}_dp", "DOUBLE")
            sp = leg(args.base, f"{tag}_sp", "SINGLE")
            if dp is None or sp is None:
                print(f"{tag}: incomplete (dp={dp} sp={sp}) — skipped")
                continue
            rows.append((fam, n, dp, sp))
            anch = {"c": M7_CPU, "n": M7_NG5}.get(fam, {}).get(n)
            anchor = f"  [m7 {anch:.3f}]" if anch else ""
            print(f"{tag:<14s} dp={dp:.4f} sp={sp:.4f} speedup={dp/sp:.3f}x{anchor}")

    if not rows:
        raise SystemExit("no complete pairs yet")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(16, 4.5))
    all_ns, y1, y3 = [], [], []
    for fam, label, marker in (("c", "dars CPU (128 r/node)", "o"), ("g", "dars GPU (4/node)", "s"),
                               ("n", "NG5 CPU (128 r/node)", "^"), ("N", "NG5 GPU (4/node)", "D"),
                               ("k", "CORE2 CPU (128 r/node)", "v"), ("f", "farc CPU (128 r/node)", "P")):
        ns = [r[1] for r in rows if r[0] == fam]
        if not ns:
            continue
        dps = [r[2] for r in rows if r[0] == fam]
        sps = [r[3] for r in rows if r[0] == fam]
        ax1.loglog(ns, dps, f"-{marker}", label=f"{label} FP64")
        ax1.loglog(ns, sps, f"--{marker}", label=f"{label} FP32")
        dt = FAM_DT[fam]
        sy = [sypd(dt, d) for d in dps] + [sypd(dt, s) for s in sps]
        ax3.loglog(ns, sy[:len(ns)], f"-{marker}", label=f"{label} FP64")
        ax3.loglog(ns, sy[len(ns):], f"--{marker}", label=f"{label} FP32")
        ax2.semilogx(ns, [d / s for d, s in zip(dps, sps)], f"-{marker}", label=label)
        all_ns += ns
        y1 += dps + sps
        y3 += sy
    m7n = sorted(M7_CPU)
    ax1.loglog(m7n, [M7_CPU[n] for n in m7n], ":x", color="gray",
               label="m7-s15 FP64 anchor (dars)")
    m7g = sorted(M7_NG5)
    ax1.loglog(m7g, [M7_NG5[n] for n in m7g], ":+", color="darkgray",
               label="m7-s15 FP64 anchor (NG5)")
    y1 += list(M7_CPU.values()) + list(M7_NG5.values())
    all_ns += m7n + m7g
    ax1.set_ylabel("s/step (min-of-2)")
    ax1.set_title("300 steps, knobs-off, per-mesh dt")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax3.set_ylabel("SYPD  (simulated yr / wall day)")
    ax3.set_title("SYPD at production dt (CORE2 1800 · farc 900 · dars/NG5 240)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=7)
    for ax in (ax1, ax3, ax2):
        node_axis(ax, all_ns)
    decimal_log_yaxis(ax1, min(y1), max(y1))
    decimal_log_yaxis(ax3, min(y3), max(y3))
    fig.text(0.995, 0.005,
             "dars/NG5 SYPD at dt240 from dt120/dt180 (c32n dt60) runs — s/step dt-independent; "
             "CG dt-correction not applied",
             ha="right", va="bottom", fontsize=6, color="gray")
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
