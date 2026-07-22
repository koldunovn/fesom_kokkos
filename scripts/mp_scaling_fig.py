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
# SYPD = dt_prod/(365*sstep*CORR); DT_PROD core2 1800 / farc 1200 (user 2026-07-22:
# 20 min, not 15) / dars 240 / NG5 240.
# s/step is near-dt-independent; projecting measurement runs (dars dt120, NG5 dt180/60)
# to production dt uses the m7-s16b MEASURED CG dt-correction (jobs/job_m7_dtpair,
# median-of-3: dars x1.0222, NG5 x1.0110); CORE2/farc measure AT production dt (1.0).
FAM_DT = {"c": 240.0, "g": 240.0, "n": 240.0, "N": 240.0, "k": 1800.0, "f": 1200.0}
FAM_CORR = {"c": 1.0222, "g": 1.0222, "n": 1.0110, "N": 1.0110, "k": 1.0, "f": 1.0}


def sypd(dt_s: float, s_step: float, corr: float = 1.0) -> float:
    return dt_s / (365.0 * s_step * corr)


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
    ticks = [t for t in (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50)
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
                               ("g", "scal_g{n}", [2, 4, 8, 16, 32]),
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
    from matplotlib.lines import Line2D
    fig, (ax1, axB, axC, ax2) = plt.subplots(1, 4, figsize=(20, 4.5))
    all_ns, y1 = [], []
    # SYPD split by mesh size, LINEAR y — the m7_scaling_figs.py throughput layout
    # ((b) CORE2 & farc, (c) multi-million-node meshes)
    nsB, nsC, yB, yC = [], [], [], []
    # one color per mesh family, solid=FP64 / dashed=FP32 — same key in all
    # panels, so the shared legend needs one entry per family + two style proxies
    for fam, label, marker, col in (("c", "dars CPU", "o", "C0"), ("g", "dars GPU", "s", "C1"),
                                    ("n", "NG5 CPU", "^", "C2"), ("N", "NG5 GPU", "D", "C3"),
                                    ("k", "CORE2 CPU", "v", "C4"), ("f", "farc CPU", "P", "C5")):
        ns = [r[1] for r in rows if r[0] == fam]
        if not ns:
            continue
        dps = [r[2] for r in rows if r[0] == fam]
        sps = [r[3] for r in rows if r[0] == fam]
        ax1.loglog(ns, dps, ls="-", marker=marker, color=col, label=label)
        ax1.loglog(ns, sps, ls="--", marker=marker, color=col)
        dt = FAM_DT[fam]
        corr = FAM_CORR[fam]
        sy = [sypd(dt, d, corr) for d in dps] + [sypd(dt, s, corr) for s in sps]
        small = fam in ("k", "f")
        axS = axB if small else axC
        axS.plot(ns, sy[:len(ns)], ls="-", marker=marker, color=col)
        axS.plot(ns, sy[len(ns):], ls="--", marker=marker, color=col)
        if small:
            nsB += ns
            yB += sy
        else:
            nsC += ns
            yC += sy
        ax2.semilogx(ns, [d / s for d, s in zip(dps, sps)], ls="-", marker=marker, color=col)
        all_ns += ns
        y1 += dps + sps
    m7n = sorted(M7_CPU)
    ax1.loglog(m7n, [M7_CPU[n] for n in m7n], ":x", color="gray",
               label="m7 anchor (dars)")
    m7g = sorted(M7_NG5)
    ax1.loglog(m7g, [M7_NG5[n] for n in m7g], ":+", color="darkgray",
               label="m7 anchor (NG5)")
    y1 += list(M7_CPU.values()) + list(M7_NG5.values())
    all_ns += m7n + m7g
    ax1.set_ylabel("s/step (min-of-2)")
    ax1.set_title("(a) time per step (300 steps, knobs-off, per-mesh dt)")
    ax1.grid(True, which="both", alpha=0.3)
    axB.set_ylabel("SYPD  (simulated yr / wall day)")
    axB.set_title("(b) throughput, CORE2 & farc (prod dt 1800 / 1200)")
    axB.set_ylim(0, max(yB) * 1.1)
    axB.grid(True, which="both", alpha=0.3)
    axC.set_ylabel("SYPD  (simulated yr / wall day)")
    axC.set_title("(c) throughput, multi-million-node meshes (prod dt 240)")
    axC.set_ylim(0, max(yC) * 1.1)
    axC.grid(True, which="both", alpha=0.3)
    # one shared horizontal legend BELOW the panels — in-axes it covered the data
    # (user 2026-07-22; same fix as m7_scaling_figs.py fig_scaling)
    handles = ax1.get_legend_handles_labels()[0] + [
        Line2D([], [], color="0.35", ls="-", label="FP64"),
        Line2D([], [], color="0.35", ls="--", label="FP32")]
    # legend row sits ABOVE the footnote line (they collided at the same y)
    fig.legend(handles=handles, ncol=len(handles), fontsize=7.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.022))
    node_axis(ax1, all_ns)
    node_axis(axB, nsB)   # per-panel counts, the m7 gpu_axis(axB, cb) pattern
    node_axis(axC, nsC)
    node_axis(ax2, all_ns)
    decimal_log_yaxis(ax1, min(y1), max(y1))
    fig.text(0.995, 0.005,
             "dars/NG5 SYPD at dt240 from dt120/dt180 (c32n dt60) runs; CG dt-correction "
             "applied (m7-s16b measured: dars x1.0222, NG5 x1.0110)",
             ha="right", va="bottom", fontsize=6, color="gray")
    ax2.axhline(1.0, color="gray", lw=0.5)
    ax2.set_xlabel("nodes")
    ax2.set_ylabel("FP32 speedup (×)")
    ax2.set_title("(d) SP speedup vs scale (same-day pinned pairs)")
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"mp_scaling.{ext}", dpi=150)
    print(f"wrote {args.out}/mp_scaling.png+pdf")

    # ---- class-Bp companion (s4, user-commissioned): SP x full speed stack, GPU ----
    # Tags scal_bp_<mesh>_g<n>_{dp,sp}; posture = m7 Bp (SPEED=1+EVPWIDE=8+CGPOLY=3+
    # unbind+proto pkg). dp-Bp anchors read live from the m7 CSV class-Bp rows.
    m7csv = pathlib.Path("/work/ab0995/a270088/port2/m7/scaling_figs/m7_scaling.csv")
    m7bp = {}
    if m7csv.exists():
        for line in m7csv.read_text().splitlines():
            cc = line.split(",")
            if len(cc) >= 6 and cc[1] == "gpu" and cc[2] == "Bp":
                m7bp[(cc[0], int(cc[3]))] = float(cc[5])
    off_gpu = {("dars", r[1]): (r[2], r[3]) for r in rows if r[0] == "g"}
    off_gpu.update({("ng5", r[1]): (r[2], r[3]) for r in rows if r[0] == "N"})
    BP_FAMS = (("dars", "s", "C1", 240.0, 1.0222), ("ng5", "D", "C3", 240.0, 1.0110),
               ("core2", "v", "C4", 1800.0, 1.0), ("farc", "P", "C5", 1200.0, 1.0))
    BP_SIZES = {"dars": [2, 4, 8, 16, 32], "ng5": [2, 4, 8, 16, 32],
                "core2": [1, 2, 4, 8], "farc": [1, 2, 4, 8, 16, 32]}
    bprows = []
    for mesh, _, _, _, _ in BP_FAMS:
        for n in BP_SIZES[mesh]:
            dp = leg(args.base, f"scal_bp_{mesh}_g{n}_dp", "DOUBLE")
            sp = leg(args.base, f"scal_bp_{mesh}_g{n}_sp", "SINGLE")
            if dp is None or sp is None:
                print(f"bp_{mesh}_g{n}: incomplete (dp={dp} sp={sp}) — skipped")
                continue
            bprows.append((mesh, n, dp, sp))
            an = m7bp.get((mesh, n))
            extra = f"  [m7-Bp {an:.4f}]" if an else ""
            offp = off_gpu.get((mesh, n))
            if offp:
                extra += f"  [off {offp[0]/offp[1]:.2f}x -> Bp {dp/sp:.2f}x; stack dp {offp[0]/dp:.2f}x]"
            print(f"bp_{mesh}_g{n:<3d} dp={dp:.4f} sp={sp:.4f} speedup={dp/sp:.3f}x{extra}")
    if not bprows:
        print("no complete Bp pairs yet — Bp figure skipped")
        return
    fig2, (b1, bB, bC, b2) = plt.subplots(1, 4, figsize=(20, 4.5))
    bp_ns, bp_y1, bp_nsB, bp_nsC, bp_yB, bp_yC = [], [], [], [], [], []
    for mesh, marker, col, dtp, corr in BP_FAMS:
        ns = [r[1] for r in bprows if r[0] == mesh]
        if not ns:
            continue
        dps = [r[2] for r in bprows if r[0] == mesh]
        sps = [r[3] for r in bprows if r[0] == mesh]
        b1.loglog(ns, dps, ls="-", marker=marker, color=col, label=f"{mesh} GPU")
        b1.loglog(ns, sps, ls="--", marker=marker, color=col)
        an = [(n, m7bp[(mesh, n)]) for n in ns if (mesh, n) in m7bp]
        if an:
            b1.loglog([a[0] for a in an], [a[1] for a in an], ls=":", marker="x",
                      color=col, alpha=0.45)
        sy = [sypd(dtp, d, corr) for d in dps] + [sypd(dtp, s, corr) for s in sps]
        small = mesh in ("core2", "farc")
        axS = bB if small else bC
        axS.plot(ns, sy[:len(ns)], ls="-", marker=marker, color=col)
        axS.plot(ns, sy[len(ns):], ls="--", marker=marker, color=col)
        (bp_nsB if small else bp_nsC).extend(ns)
        (bp_yB if small else bp_yC).extend(sy)
        b2.semilogx(ns, [d / s for d, s in zip(dps, sps)], ls="-", marker=marker, color=col)
        offs = [(n, off_gpu[(mesh, n)][0] / off_gpu[(mesh, n)][1])
                for n in ns if (mesh, n) in off_gpu]
        if offs:
            b2.semilogx([o[0] for o in offs], [o[1] for o in offs], ls=":", marker=marker,
                        color=col, alpha=0.45)
        bp_ns += ns
        bp_y1 += dps + sps + [a[1] for a in an]
    b1.set_ylabel("s/step (min-of-2)")
    b1.set_title("(a) time per step, class Bp (dotted x = m7 FP64-Bp anchor)")
    b1.grid(True, which="both", alpha=0.3)
    bB.set_ylabel("SYPD  (simulated yr / wall day)")
    bB.set_title("(b) throughput, CORE2 & farc (prod dt 1800 / 1200)")
    bB.set_ylim(0, max(bp_yB) * 1.1 if bp_yB else 1)
    bB.grid(True, which="both", alpha=0.3)
    bC.set_ylabel("SYPD  (simulated yr / wall day)")
    bC.set_title("(c) throughput, multi-million-node meshes (prod dt 240)")
    bC.set_ylim(0, max(bp_yC) * 1.1 if bp_yC else 1)
    bC.grid(True, which="both", alpha=0.3)
    b2.axhline(1.0, color="gray", lw=0.5)
    b2.set_ylabel("FP32 speedup (×)")
    b2.set_title("(d) SP speedup: Bp (solid) vs knobs-off (dotted)")
    b2.grid(True, which="both", alpha=0.3)
    handles2 = b1.get_legend_handles_labels()[0] + [
        Line2D([], [], color="0.35", ls="-", label="FP64"),
        Line2D([], [], color="0.35", ls="--", label="FP32")]
    fig2.legend(handles=handles2, ncol=len(handles2), fontsize=7.5, frameon=False,
                loc="lower center", bbox_to_anchor=(0.5, 0.022))
    node_axis(b1, bp_ns)
    if bp_nsB:
        node_axis(bB, bp_nsB)
    if bp_nsC:
        node_axis(bC, bp_nsC)
    node_axis(b2, bp_ns)
    decimal_log_yaxis(b1, min(bp_y1), max(bp_y1))
    fig2.text(0.995, 0.005,
              "class Bp = FESOM_SPEED=1 + EVPWIDE=8 + CGPOLY=3 + unbind + proto pkg "
              "(farc g32 = class B: proto hangs farc at 128 ranks, m7-s16b); knobs fired "
              "asserted per leg (L80); CG dt-corr applied (dars x1.0222, NG5 x1.0110)",
              ha="right", va="bottom", fontsize=6, color="gray")
    fig2.tight_layout(rect=[0, 0.075, 1, 1])
    for ext in ("png", "pdf"):
        fig2.savefig(args.out / f"mp_scaling_bp.{ext}", dpi=150)
    print(f"wrote {args.out}/mp_scaling_bp.png+pdf")


if __name__ == "__main__":
    main()
