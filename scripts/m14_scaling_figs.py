#!/usr/bin/env python3
"""
M14 strong-scaling figures: the state BEFORE these campaigns vs the best combination AFTER.

Three figures, all harvested from the job logs under $M14 — no manual editing:
  m14_scaling_sstep.png    wall time per step, one panel per platform (log y, decimal ticks)
  m14_scaling_sypd.png     SYPD at PRODUCTION dt — 2 rows (small meshes / large meshes)
                           x 3 platforms, because a shared linear SYPD axis hides
                           dars/NG5 entirely behind CORE2 (the same reason m7's fig10
                           split (b) from (c))
  m14_speedup.png          best/base speedup vs scale — the campaign's mechanism: every
                           lever attacks COMMUNICATION, so the payoff grows with how far
                           past the knee the point is, and is a LOSS below it

Conventions are taken from scripts/m7_scaling_figs.py (standing rule: read it before plotting):
log2 node axis with explicit tick labels, plain-decimal log y, paper mesh colours via
paper_jax/scripts/common, before = open marker + dotted, after = filled + solid, and ONE shared
horizontal legend BELOW the panels (in-axes it sat on the CORE2 curves — user, 2026-07-21).

SYPD = DT_PROD / (365 · s_per_step).

⚠️ NO CG dt-CORRECTION APPLIED (user decision, 2026-08-16: "for now we can live without a
factor"). dars is measured at dt120 and NG5 at dt180 but both are reported at dt240, and s/step is
NOT dt-independent for the implicit solve — a larger dt needs more CG iterations. m7_scaling_figs.py
carries measured factors (dars ×1.0222, NG5 ×1.0110) for exactly this. Omitting them OVERSTATES
dars and NG5 SYPD by ~2 % and ~1 %. Stated on the figure so it cannot be read as exact.
⚠️ The SE arm's cost is genuinely dt-dependent (its barotropic subcycle count M is set by the
CFL at the baroclinic dt), so an SE point rescaled to production dt is an estimate, not a
measurement — flagged in the caption.

⚠️ TWO ESTIMATOR RULES, both learned the hard way in this project:
  · The ABSOLUTE curves take the min over every admitted leg of every job at a point (M12b's
    min-over-all-legs lesson: min-of-2 on a bimodal arm is not reproducible).
  · A SPEEDUP is only ever computed WITHIN one job — both arms shared an allocation there.
    Never base_min/best_min across jobs. Where several jobs tested the SAME lever at a point the
    gains are MEDIANed (M11's F43 rule); across DIFFERENT levers the best one wins, because the
    curve is "best configuration", not "best repeat".
"""
import glob
import os
import re
import sys
import collections
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/a/a270088/paper_jax/scripts")
try:
    import common
    MESH_COLOR, MESH_LABEL = common.MESH_COLOR, common.MESH_LABEL
    MESH_ORDER = common.MESH_ORDER
    common.set_style()
except Exception:                                   # keep the figure buildable off-paper
    MESH_COLOR = {"core2": "#4C72B0", "farc": "#DD8452", "dars": "#55A868", "ng5": "#C44E52"}
    MESH_LABEL = {"core2": "CORE2", "farc": "fArc", "dars": "DARS", "ng5": "NG5"}
    MESH_ORDER = ["core2", "farc", "dars", "ng5"]

M14 = "/work/ab0995/a270088/port2/m14"
DT_PROD = {"core2": 1800.0, "farc": 900.0, "dars": 240.0, "ng5": 240.0}
SYPD = lambda mesh, s: DT_PROD[mesh] / (365.0 * s)

PLATS = [("cpu",   "Levante CPU  (128-core nodes)", "nodes"),
         ("a100",  "Levante A100  (4 GPU/node)",    "number of GPUs"),
         ("jupiter", "JUPITER GH200  (4 GPU/node)", "number of GPUs")]

SMALL = ("core2", "farc")          # SYPD O(10-200)
LARGE = ("dars", "ng5")            # SYPD O(1-10) — a shared linear axis hides these entirely


def _lever(txt):
    """The best arm's recipe, as a short label. None => this job's best arm is not admissible."""
    bk = re.search(r"best knobs: (.*)", txt)
    bm = re.search(r"best mesh : \S*/(\w+)/dist_", txt)
    lev = (("part+" if bm else "") + (bk.group(1) if bk else "")) \
        .replace("FESOM_SSH_SOLVER=", "").replace("FESOM_SSH_MODE=", "") \
        .replace("FESOM_SPEED_EVPWIDE_LEAN=1", "lean").replace("FESOM_SPEED_EVPWIDE=", "evpwide K=") \
        .replace("FESOM_SE_M=", "M=").replace("FESOM_", "").strip()
    return re.sub(r"\s+", " ", lev)


# The three lever families the campaign can isolate, because each was measured on its own against
# the same baseline in its own job. A job that carries TWO levers (part+oati) is a `combo` and
# belongs to none of them; a job with no lever at all is a `null` cost control, and admitting it
# as a "best" arm would put a noise-level pair on a lever's curve.
FAMILIES = {"ssh":  "SSH solve",
            "ice":  "sea-ice wide halo",
            "part": "partitioning"}


def classify(txt):
    """-> (family, detail). detail distinguishes curves WITHIN a family (oati vs SE, K, engine)."""
    bk = re.search(r"best knobs: (.*)", txt)
    bm = re.search(r"^best mesh : \S*/(\w+)/dist_", txt, re.M)
    knobs = re.sub(r"\s+", " ", bk.group(1).strip()) if bk else ""
    part = bm.group(1) if bm else None
    ssh = "SSH_SOLVER" in knobs or "SSH_MODE" in knobs
    ice = "EVPWIDE" in knobs
    if part and not knobs:
        return "part", part
    if part or (ssh and ice):
        return "combo", (f"part={part} " if part else "") + knobs
    if ssh:
        if "SSH_MODE=se" in knobs:
            # M is a PER-MESH tuning (fArc 90, dars 20); the default 50 is a different
            # configuration, not a repeat of the same one, so it stays its own curve.
            return "ssh", "SE (tuned M)" if re.search(r"SE_M=\d+", knobs) else "SE (default M=50)"
        m = re.search(r"SSH_SOLVER=(\w+)", knobs)
        return "ssh", m.group(1) if m else "solver"
    if ice:
        m = re.search(r"EVPWIDE=(\d+)", knobs)
        return "ice", "K=" + (m.group(1) if m else "?") + (" lean" if "LEAN" in knobs else "")
    return "null", ""


def harvest():
    """-> list of per-job records {plat, mesh, units, job, lever, base, best}.

    units = nodes for CPU, GPUs for the two GPU platforms. A record may carry only one arm:
    the baseline ladders ran ARMS='base' alone, and a job whose best legs were all rejected
    (the open GPU CG NaN) keeps its base."""
    recs = []

    def add(plat, mesh, units, job, lever, txt, pat):
        fam, detail = classify(txt)
        # The wsplit state is part of the CONFIGURATION, not a lever, so two rows measured with
        # it on and off are not repeats of each other. Runs predating the 2026-08-16 harness guard
        # carry no line at all; they are all wsplit-off, tagged `wsplit?` to keep that visible.
        w = re.search(r"^wsplit\s*: (\w+)", txt, re.M)
        r = dict(plat=plat, mesh=mesh, units=units, job=job, lever=lever, fam=fam, detail=detail,
                 wsplit=w.group(1) if w else "wsplit?")
        for arm in ("base", "best"):
            m = re.search(pat.format(arm=arm), txt)
            if m and float(m.group(1)) > 0:
                r[arm] = float(m.group(1))
        if "base" in r or "best" in r:
            recs.append(r)

    RESULT = r"\n\s+{arm}\s+(?:legs.*?)?min=([\d.]+)"

    # --- Levante CPU -------------------------------------------------------
    for f in glob.glob(f"{M14}/ladder.*.out"):
        txt = open(f, errors="replace").read()
        m = re.search(r"M14 CPU ladder — (\w+) ranks=(\d+)", txt)
        if not m:
            continue
        add("cpu", m.group(1), max(1, round(int(m.group(2)) / 128)),
            os.path.basename(f), _lever(txt), txt, RESULT)

    # --- Levante A100 ------------------------------------------------------
    for f in glob.glob(f"{M14}/gladder.*.out"):
        txt = open(f, errors="replace").read()
        m = re.search(r"M14 CPU ladder — (\w+) ranks=(\d+)", txt)   # derived job keeps the header
        if not m:
            continue
        add("a100", m.group(1), int(m.group(2)), os.path.basename(f), _lever(txt), txt, RESULT)

    # --- JUPITER GH200 (JSC booster) ---------------------------------------
    # Replaces the dolpung column (user, 2026-08-16). dolpung was a 42-node proxy; JUPITER is
    # the real GH200 result and reaches 256 nodes = 1024 GPUs. Its data arrives as the tidy CSV
    # the JUPITER session harvested, not as job logs — those stayed on that machine.
    # 🔴 The dolpung EVPWIDE rows are deliberately NOT carried over: every one of them ran the
    # pre-fix code path, which posted MPI on device pointers and crashed on a fabric without
    # GPUDirect. The "GH200 loses on the ice wide halo" verdict was an artifact of that bug
    # (commit 94877a9); post-fix JUPITER has it winning on every mesh at every rung.
    csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs/plans/20260816-m14-scaling.csv")
    if os.path.exists(csv):
        import csv as _csv
        FAM = {"oati": ("ssh", "oati"), "se": ("ssh", "SE (tuned M)"),
               "part": ("part", "a5_u30/a4u30"), "evpwide": ("ice", "K=8 lean"),
               "part+oati": ("combo", "part+oati"), "base": ("null", "")}
        with open(csv) as fh:
            for r in _csv.DictReader(fh):
                if r.get("void_mixture", "").strip().lower() == "true":
                    continue                      # variant/baseline mixture — not a measurement
                # 🔴 The JUPITER campaign spans two binaries. `fab4919c` predates the EVPWIDE
                # staging fix and posts MPI on device pointers, which on this fabric crashes.
                # 23 of the 48 wide-halo rows are on it: they measure the broken path, not the
                # lever, so they are dropped. Every other lever is untouched by that fix and
                # both binaries are valid for them.
                if r["lever"] == "evpwide" and r.get("md5", "").startswith("fab4919c"):
                    continue
                fam, detail = FAM.get(r["lever"], ("combo", r["lever"]))
                rec = dict(plat="jupiter", mesh=r["mesh"], units=int(r["ranks"]),
                           job=r["job"], lever=r["lever"], fam=fam, detail=detail,
                           wsplit=r.get("wsplit", "wsplit?"))
                for arm in ("base", "best"):
                    v = (r.get(arm) or "").strip()
                    if v:
                        rec[arm] = float(v)
                if "base" in rec or "best" in rec:
                    recs.append(rec)
    return recs


def check_wsplit(recs):
    """A curve must not silently merge a wsplit-on measurement with a wsplit-off one."""
    seen = collections.defaultdict(set)
    for r in recs:
        seen[(r["plat"], r["mesh"], r["units"])].add(r["wsplit"])
    mixed = {k: v for k, v in seen.items() if len(v) > 1}
    if mixed:
        print("\n  \U0001F534 MIXED wsplit STATE — these points merge different configurations:")
        for (plat, mesh, units), tags in sorted(mixed.items()):
            print(f"     {plat:5s} {mesh:6s} {units:5d}  {sorted(tags)}")
        print("     Split the figure by wsplit state before quoting any of these.\n")
    else:
        tags = sorted({r["wsplit"] for r in recs})
        print(f"  wsplit state, uniform per point: {tags}")


def in_best_combination(r):
    """Which measured levers the "best combination" curve is allowed to take.

    A null control carries no lever. The sea-ice wide halo is a measured LOSS on CPU at every K
    and on GH200, so the recommended combination excludes it there — but the records are still
    harvested, because the ice-only figure has to show exactly that loss."""
    if r["fam"] == "null":
        return False
    # The ice wide halo is a win on A100 and, since the staging fix (94877a9), on JUPITER too.
    # It remains a measured loss on CPU at every K and every rank count.
    if r["fam"] == "ice" and r["plat"] not in ("a100", "jupiter"):
        return False
    return True


def curves(recs):
    """-> {plat: {mesh: {units: {'base': s, 'best': s, 'lever': str}}}}, min over every job."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in recs:
        d = out[r["plat"]][r["mesh"]].setdefault(r["units"], {})
        for arm in ("base", "best"):
            if arm == "best" and not in_best_combination(r):
                continue
            if arm in r and (arm not in d or r[arm] < d[arm]):
                d[arm] = r[arm]
                if arm == "best":
                    d["lever"] = r["lever"]
    return out


def speedups(recs):
    """-> {plat: {mesh: {units: (speedup, lever, n_jobs)}}}, WITHIN-job only.

    Same lever measured several times => median gain (F43). Different levers => the best."""
    per_lever = collections.defaultdict(list)
    for r in recs:
        if "base" in r and "best" in r and in_best_combination(r):
            per_lever[(r["plat"], r["mesh"], r["units"], r["lever"])].append(r["base"] / r["best"])
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    best = {}
    for (plat, mesh, units, lev), vals in per_lever.items():
        s = statistics.median(vals)
        k = (plat, mesh, units)
        if k not in best or s > best[k][0]:
            best[k] = (s, lev, len(vals))
    for (plat, mesh, units), v in best.items():
        out[plat][mesh][units] = v
    return out


# ---------------------------------------------------------------- axes helpers
def decimal_log_yaxis(ax, lo, hi):
    ticks = [t for t in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3)
             if lo * 0.95 <= t <= hi * 1.05]
    if ticks:
        ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(lo * 0.85, hi * 1.15)


def log2_axis(ax, counts, label):
    """Powers of two only. Labelling every measured rung collided (128/160/192 and 256/320
    overprinted); the markers still sit at their true x, the ticks are just a readable ruler."""
    ax.set_xscale("log", base=2)
    counts = sorted(set(int(c) for c in counts))
    lo, hi = counts[0], counts[-1]
    ticks, t = [], 1
    while t <= hi * 2:
        if lo <= t <= hi:
            ticks.append(t)
        t *= 2
    if hi not in ticks:
        if ticks and hi / ticks[-1] < 1.4:      # 256 and 320 overprinted each other
            ticks.pop()
        ticks.append(hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(c) for c in ticks], fontsize=8)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(label)


def shared_legend(fig, data, extra):
    handles = [Line2D([], [], color=MESH_COLOR.get(m, "k"), marker="o", ms=4.5, ls="-",
                      label=MESH_LABEL.get(m, m))
               for m in MESH_ORDER if any(m in data.get(p, {}) for p, _, _ in PLATS)] + extra
    fig.legend(handles=handles, ncol=len(handles), fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.075))


NOTE_LEVERS = ("Best = the winning lever at each point, measured against the pre-campaign baseline "
               "in the SAME allocation (ABBA, warmup discarded, min over legs).  Levers: SSH solver "
               "`oati` or split-explicit SE (per-mesh FESOM_SE_M: fArc 90, dars 20), the M11 "
               "partition where one exists, and on both GPU machines the sea-ice wide halo — a win "
               "there, a measured loss on CPU at every K.")
NOTE_SYPD = ("SYPD = dt_prod/(365·s_step), dt_prod = CORE2 1800 · fArc 900 · dars 240 · NG5 240.  "
             "dars is measured at dt120 and NG5 at dt180 with NO CG dt-correction applied "
             "(it would be ×1.022 / ×1.011), so dars/NG5 SYPD is ~2 %/1 % optimistic.")
NOTE_GAP = ("JUPITER points are single-allocation on a production-loaded fabric.  NG5 on A100 is "
            "baseline-only: a cold-start CFL blow-up killed 30 of its 37 legs in BOTH arms.")


def annotate_gaps(ax, plat, data):
    """A curve that is baseline-only must SAY why, or a reader reads it as 'no gain'."""
    if plat != "a100":
        return
    missing = [MESH_LABEL.get(m, m) for m in MESH_ORDER
               if m in data.get(plat, {})
               and not any("best" in v for v in data[plat][m].values())]
    if missing:
        ax.text(0.02, 0.03, "  ".join(missing) + ": baseline only — GPU CG NaN\n"
                "killed 30/37 legs, in BOTH arms",
                transform=ax.transAxes, fontsize=6.6, color="0.35", va="bottom")


# ---------------------------------------------------------------- figures
def fig_sstep(data, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    lo, hi = 1e9, 0
    for ax, (plat, pname, xlab) in zip(axes, PLATS):
        allx = []
        for mesh in [m for m in MESH_ORDER if m in data.get(plat, {})]:
            col, pts = MESH_COLOR.get(mesh, "k"), data[plat][mesh]
            for arm, ls, mfc in (("base", ":", "none"), ("best", "-", col)):
                xs = sorted(u for u in pts if arm in pts[u])
                if not xs:
                    continue
                ys = [pts[u][arm] for u in xs]
                allx += xs
                lo, hi = min(lo, min(ys)), max(hi, max(ys))
                ax.plot(xs, ys, marker="o", ms=4.5, ls=ls, color=col, markerfacecolor=mfc,
                        lw=1.7 if arm == "best" else 1.2)
        if allx:
            log2_axis(ax, allx, xlab)
        ax.set_title(pname, fontsize=10)
        ax.grid(alpha=0.25, which="major")
        ax.set_yscale("log")
        ax.set_ylabel("wall time per step  (s)" if plat == "cpu" else "")
        annotate_gaps(ax, plat, data)
    for ax in axes:
        decimal_log_yaxis(ax, lo, hi)
    shared_legend(fig, data, [
        Line2D([], [], color="0.35", ls=":", marker="o", ms=4.5, markerfacecolor="none",
               label="before campaigns (baseline)"),
        Line2D([], [], color="0.35", ls="-", marker="o", ms=4.5, label="best combination")])
    fig.suptitle("M14 — strong scaling: wall time per step", fontsize=11, y=0.99)
    fig.text(0.5, 0.008, NOTE_LEVERS + "  " + NOTE_GAP, ha="center", fontsize=6.6, color="0.35",
             wrap=True)
    fig.tight_layout(rect=[0, 0.145, 1, 0.965])
    fig.savefig(fname, dpi=140)
    print("wrote", fname)


def fig_sypd(data, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6))
    for row, group, gname in ((0, SMALL, "CORE2 / fArc"), (1, LARGE, "DARS / NG5")):
        for col, (plat, pname, xlab) in enumerate(PLATS):
            ax = axes[row][col]
            allx, top = [], 0
            for mesh in [m for m in MESH_ORDER if m in data.get(plat, {}) and m in group]:
                c, pts = MESH_COLOR.get(mesh, "k"), data[plat][mesh]
                for arm, ls, mfc in (("base", ":", "none"), ("best", "-", c)):
                    xs = sorted(u for u in pts if arm in pts[u])
                    if not xs:
                        continue
                    ys = [SYPD(mesh, pts[u][arm]) for u in xs]
                    allx += xs
                    top = max(top, max(ys))
                    ax.plot(xs, ys, marker="o", ms=4.5, ls=ls, color=c, markerfacecolor=mfc,
                            lw=1.7 if arm == "best" else 1.2)
            if allx:
                log2_axis(ax, allx, xlab if row else "")
            ax.set_ylim(0, top * 1.12 if top else 1)
            ax.grid(alpha=0.25)
            ax.set_title(f"{pname} — {gname}", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("SYPD  (simulated years / wall day)")
            annotate_gaps(ax, plat, data)
    shared_legend(fig, data, [
        Line2D([], [], color="0.35", ls=":", marker="o", ms=4.5, markerfacecolor="none",
               label="before campaigns (baseline)"),
        Line2D([], [], color="0.35", ls="-", marker="o", ms=4.5, label="best combination")])
    fig.suptitle("M14 — strong scaling: throughput at production dt", fontsize=11, y=0.99)
    fig.text(0.5, 0.008, NOTE_SYPD + "  " + NOTE_GAP, ha="center", fontsize=6.6, color="0.35",
             wrap=True)
    fig.tight_layout(rect=[0, 0.075, 1, 0.975])
    fig.savefig(fname, dpi=140)
    print("wrote", fname)


def fig_speedup(sp, data, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    hi = max([v[0] for p in sp for m in sp[p] for v in sp[p][m].values()] + [1.1])
    for ax, (plat, pname, xlab) in zip(axes, PLATS):
        allx = []
        for mesh in [m for m in MESH_ORDER if m in sp.get(plat, {})]:
            c, pts = MESH_COLOR.get(mesh, "k"), sp[plat][mesh]
            xs = sorted(pts)
            allx += xs
            ax.plot(xs, [pts[u][0] for u in xs], marker="o", ms=4.5, ls="-", color=c, lw=1.7)
        ax.axhline(1.0, color="0.5", lw=1.0, ls="--")
        if allx:
            log2_axis(ax, allx, xlab)
        ax.set_ylim(0.93, hi * 1.06)
        ax.grid(alpha=0.25)
        ax.set_title(pname, fontsize=10)
        if plat == "cpu":
            ax.set_ylabel("speedup  (baseline / best)")
        annotate_gaps(ax, plat, data)
    shared_legend(fig, data, [
        Line2D([], [], color="0.5", ls="--", lw=1.0, label="no gain (1.0)")])
    fig.suptitle("M14 — payoff of the combined levers vs scale "
                 "(every lever attacks communication, so the gain grows past the knee)",
                 fontsize=11, y=0.99)
    fig.text(0.5, 0.008,
             "Each point is a WITHIN-allocation A/B: baseline and best ran in the same job, ABBA, "
             "warmup discarded, min over legs — never a ratio of two jobs' minima.  Repeats of one "
             "lever are MEDIANed; where several levers were tested at a point the best one is "
             "plotted.  JUPITER points are single-allocation and the machine was under real "
             "production load — the campaign\'s own ruling is that a loaded fabric is the "
             "deployment-relevant condition, but the largest JUPITER gains carry an "
             "allocation-to-allocation spread the CSV does not record.  "
             "Levers: CPU/GH200 = SSH solver `oati` or split-explicit SE (per-mesh "
             "FESOM_SE_M: fArc 90, dars 20) + the M11 certified partition where one exists; A100 "
             "ALSO takes the sea-ice wide halo (evpwide K=8, lean), a win there and a measured "
             "loss on CPU. On JUPITER it also wins, once the staging fix stopped it posting MPI "
             "on device pointers.  A100 dars has no point at 32 GPUs: that job's baseline legs "
             "were rejected, so the pair does not exist even though both arms were measured "
             "somewhere.",
             ha="center", fontsize=6.6, color="0.35", wrap=True)
    fig.tight_layout(rect=[0, 0.145, 1, 0.965])
    fig.savefig(fname, dpi=140)
    print("wrote", fname)


DETAIL_LS = ["-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]


def fig_lever(recs, data, fam, fname):
    """One lever, on its own: absolute curves (row 0) and its within-job speedup (row 1).

    The baseline (dotted) is the SAME curve in every lever figure — min over every admitted base
    leg at that rung, from any job. The lever curve is min over that family's best legs only.
    Speedups are within-job, so a rung where the lever ran but its own baseline legs died has an
    absolute point and NO speedup point — that asymmetry is real and is left visible."""
    mine = [r for r in recs if r["fam"] == fam]
    # Most-measured detail gets the solid line: within a family it is the one the campaign
    # actually settled on (oati, K=8, a5_u30), and the eye reads solid as the main curve.
    freq = collections.Counter(r["detail"] for r in mine)
    details = sorted(freq, key=lambda d: (-freq[d], d))
    ls_of = {d: DETAIL_LS[i % len(DETAIL_LS)] for i, d in enumerate(details)}

    lev_abs = collections.defaultdict(dict)          # (plat,mesh,detail) -> {units: s}
    lev_sp = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in mine:
        k = (r["plat"], r["mesh"], r["detail"])
        if "best" in r:
            d = lev_abs[k]
            if r["units"] not in d or r["best"] < d[r["units"]]:
                d[r["units"]] = r["best"]
        if "base" in r and "best" in r:
            lev_sp[k][r["units"]].append(r["base"] / r["best"])

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.8))
    lo, hi, shi, slo = 1e9, 0, 1.05, 1.0
    for col, (plat, pname, xlab) in enumerate(PLATS):
        a0, a1 = axes[0][col], axes[1][col]
        allx = []
        for mesh in MESH_ORDER:
            c = MESH_COLOR.get(mesh, "k")
            base = {u: v["base"] for u, v in data.get(plat, {}).get(mesh, {}).items()
                    if "base" in v}
            has = [d for d in details if lev_abs.get((plat, mesh, d))]
            if not has:
                continue
            xs = sorted(base)
            if xs:                                   # baseline, identical in all three figures
                a0.plot(xs, [base[u] for u in xs], marker="o", ms=4, ls=":", color=c, lw=1.2,
                        markerfacecolor="none")
                allx += xs
                lo, hi = min(lo, min(base.values())), max(hi, max(base.values()))
            for d in has:
                pts = lev_abs[(plat, mesh, d)]
                xs = sorted(pts)
                allx += xs
                lo, hi = min(lo, min(pts.values())), max(hi, max(pts.values()))
                a0.plot(xs, [pts[u] for u in xs], marker="o", ms=4.5, ls=ls_of[d], color=c, lw=1.7)
                sp = lev_sp.get((plat, mesh, d), {})
                xs = sorted(sp)
                if xs:
                    ys = [statistics.median(sp[u]) for u in xs]
                    shi, slo = max(shi, max(ys)), min(slo, min(ys))
                    a1.plot(xs, ys, marker="o", ms=4.5, ls=ls_of[d], color=c, lw=1.7)
        for ax in (a0, a1):
            if allx:
                log2_axis(ax, allx, xlab if ax is a1 else "")
            ax.grid(alpha=0.25)
            if not allx:
                ax.text(0.5, 0.5, "not measured", transform=ax.transAxes, ha="center",
                        fontsize=9, color="0.6")
                ax.set_xticks([])
                ax.set_yticks([])
        a1.axhline(1.0, color="0.5", lw=1.0, ls="--")
        a0.set_title(pname, fontsize=10)
        if allx:
            a0.set_yscale("log")
        if col == 0:
            a0.set_ylabel("wall time per step  (s)")
            a1.set_ylabel("speedup  (baseline / lever)")
    for col in range(3):
        if axes[0][col].get_xticks().size:
            decimal_log_yaxis(axes[0][col], lo, hi)
        axes[1][col].set_ylim(min(0.97, slo * 0.99), max(1.03, shi * 1.03))

    handles = [Line2D([], [], color=MESH_COLOR.get(m, "k"), marker="o", ms=4.5, ls="-",
                      label=MESH_LABEL.get(m, m)) for m in MESH_ORDER
               if any((p, m, d) in lev_abs for p, _, _ in PLATS for d in details)]
    handles += [Line2D([], [], color="0.35", ls=":", marker="o", ms=4, markerfacecolor="none",
                       label="baseline")]
    handles += [Line2D([], [], color="0.35", ls=ls_of[d], label=d) for d in details]
    ncol = min(len(handles), 6)
    rows = -(-len(handles) // ncol)                  # the legend wrapped onto the x-label
    bottom = 0.075 + 0.022 * rows
    fig.legend(handles=handles, ncol=ncol, fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, bottom - 0.022 * rows))
    fig.suptitle(f"M14 — {FAMILIES[fam]} alone: strong scaling and speedup", fontsize=11, y=0.99)
    fig.text(0.5, 0.008, LEVER_NOTE[fam], ha="center", fontsize=6.6, color="0.35", wrap=True)
    fig.tight_layout(rect=[0, bottom, 1, 0.965])
    fig.savefig(fname, dpi=140)
    print("wrote", fname)


LEVER_NOTE = {
    "ssh": ("Two distinct SSH levers, both measured alone against the same baseline: the `oati` "
            "Krylov solver (M10) and split-explicit barotropic subcycling (M12, `FESOM_SSH_MODE=se`, "
            "requires zstar).  SE needs a PER-MESH subcycle count: the default M=50 ABORTS on fArc "
            "and inverts the verdict on dars, so fArc is run at M=90 and dars at M=20.  An SE point "
            "rescaled to production dt would be an estimate — these are s/step at the measured dt."),
    "ice": ("The M9/M12b wide ice halo (FESOM_SPEED_EVPWIDE=K with the lean writing), which trades "
            "halo EXCHANGES for replicated ghost work, so it pays only where a flat per-message "
            "staging cost dominates.  On A100 that holds and it is the campaign's best GPU lever; "
            "on CPU it is a LOSS at every K and every rank count, which is why the CPU panel sits "
            "BELOW 1.0 — that is the measurement, not a plotting error."),
    "part": ("Partitioning alone: the same binary and knobs, only the dist_N tree swapped for an "
             "M11 partition (a5_u30 = slack UFACTOR=30, plus the KaHyPar/KaMinPar external engines "
             "and the certified *_v1 promotions).  Measured on CPU only — the M11 GPU points were "
             "re-measured in that campaign, not re-run here."),
}


if __name__ == "__main__":
    recs = harvest()
    check_wsplit(recs)
    data, sp = curves(recs), speedups(recs)
    for plat, pname, _ in PLATS:
        for mesh in MESH_ORDER:
            if mesh not in data.get(plat, {}):
                continue
            pts = data[plat][mesh]
            nb = sum(1 for u in pts if "best" in pts[u])
            best = max(((v[0], u, v[1]) for u, v in sp.get(plat, {}).get(mesh, {}).items()),
                       default=None)
            tag = f"  best A/B {best[0]:.3f}x @{best[1]} [{best[2]}]" if best else "  no A/B pair"
            print(f"  {pname:32s} {mesh:6s}: {len(pts):2d} rungs, {nb:2d} with a best arm{tag}")
    print("\n  per-lever job counts (jobs with a usable within-job pair / jobs with a best arm):")
    for fam in FAMILIES:
        n = [r for r in recs if r["fam"] == fam]
        print(f"    {fam:5s} {FAMILIES[fam]:20s} "
              f"{sum(1 for r in n if 'base' in r and 'best' in r):3d} / {sum(1 for r in n if 'best' in r):3d}"
              f"   details: {sorted({r['detail'] for r in n})}")
    for fam in ("combo", "null"):
        n = [r for r in recs if r["fam"] == fam]
        print(f"    {fam:5s} {'(excluded from lever figures)':20s} {len(n):3d} jobs")

    fig_sstep(data, f"{M14}/m14_scaling_sstep.png")
    fig_sypd(data, f"{M14}/m14_scaling_sypd.png")
    fig_speedup(sp, data, f"{M14}/m14_speedup.png")
    for fam in FAMILIES:
        fig_lever(recs, data, fam, f"{M14}/m14_lever_{fam}.png")
