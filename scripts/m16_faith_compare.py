#!/usr/bin/env python3
"""M16 Gate 4 — the faithfulness statement, computed from the arms' NetCDF output.

    m16_faith_compare.py <runroot> [--vars sst,a_ice,temp,salt] [--rec -1]

The question this answers is NOT "is SP close to DP" — that alone has no bar. It is:

    does the PORT depart from double precision the same way the FORTRAN does,
    and is either departure larger than the spread of FP64 runs that differ only
    by a 2e-4 K perturbation of the initial temperature?

So for each variable it reports, over the points valid in every arm:

    relL2(sp, dp)  =  ||X_sp - X_dp||_2 / ||X_dp||_2      per code
    mean, and the mean shift from that code's own DP arm

and then the two lines that carry the verdict:

    RATIO      relL2_port / relL2_fortran   -- 1.0 means the port's SP behaves
                                               exactly as faithfully as upstream's
    ENVELOPE   relL2 of each FP64 noise twin vs the unperturbed FP64 run -- an
               SP-DP departure below this is not a precision finding, it is the
               model's own sensitivity to a rounding-sized nudge (Suvarchal's design)

Arms are directories under <runroot>: fdp fsp pdp psp fdp_s<seed>...  Missing arms are
skipped with a note rather than failing, so the script is useful before the matrix is complete.

--where adds a spatial breakdown of the port-vs-Fortran difference: how concentrated it is, which
latitudes carry it, and whether the ice edge is over-represented. Chaotic growth is diffuse and
sits where the flow is energetic; a mechanism is concentrated and sits somewhere nameable.
"""
import sys, os, glob, math
import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    sys.exit("needs netCDF4: use /work/ab0995/a270088/mambaforge/envs/nereus/bin/python")

VARS_DEFAULT = ["sst", "a_ice", "temp", "salt"]


def load(arm_dir, var, rec):
    """Return the record `rec` of `var` from whichever file in arm_dir/output holds it."""
    pats = [os.path.join(arm_dir, "output", f"{var}.fesom.*.nc"),
            os.path.join(arm_dir, "output", f"{var}.*.nc"),
            os.path.join(arm_dir, "output", "*.nc")]
    for p in pats:
        for f in sorted(glob.glob(p)):
            try:
                ds = Dataset(f)
            except OSError:
                continue
            if var in ds.variables:
                v = ds.variables[var]
                a = np.array(v[rec] if v.ndim > 1 else v[:], dtype=np.float64)
                ds.close()
                return a, os.path.basename(f)
            ds.close()
    return None, None


def stats(arms, var, rec):
    """Load `var` for every arm; build one shared validity mask; return dict arm -> array."""
    got, src = {}, {}
    for a, d in arms.items():
        x, f = load(d, var, rec)
        if x is not None:
            got[a], src[a] = x, f
    if not got:
        return None, None, None
    shapes = {a: x.shape for a, x in got.items()}
    if len(set(shapes.values())) > 1:
        return None, None, f"shape mismatch across arms: {shapes}"
    mask = np.ones(next(iter(got.values())).shape, dtype=bool)
    for x in got.values():
        mask &= np.isfinite(x)
        mask &= (np.abs(x) < 1e30)
    # below-bottom padding: a point that is exactly zero in EVERY arm carries no signal
    allzero = np.ones_like(mask)
    for x in got.values():
        allzero &= (x == 0.0)
    mask &= ~allzero
    return got, mask, src


def relL2(a, b, mask):
    d = a[mask] - b[mask]
    n = np.linalg.norm(b[mask])
    return float(np.linalg.norm(d) / n) if n > 0 else float("nan")


def where(runroot, arms, var, rec):
    """Say WHERE the two codes differ, so a gap can be told from a bug.

    A difference spread thinly over the whole ocean is chaotic growth; one concentrated at a
    handful of nodes, at the ice edge, or at a bathymetry feature is a mechanism worth naming.
    Prints concentration, latitude band, and ice-edge over/under-representation."""
    md_path = os.path.join(runroot, "fdp", "output", "fesom.mesh.diag.nc")
    if not os.path.exists(md_path):
        print(f"   (no mesh diag at {md_path} — skipping the spatial breakdown)")
        return
    md = Dataset(md_path)
    lon = np.array(md.variables["lon"][:]); lat = np.array(md.variables["lat"][:])
    md.close()
    if np.abs(lat).max() < 3.2:            # radians, as some builds write them
        lon, lat = np.degrees(lon), np.degrees(lat)

    f_dp, _ = load(arms.get("fdp", ""), var, rec)
    p_dp, _ = load(arms.get("pdp", ""), var, rec)
    f_sp, _ = load(arms.get("fsp", ""), var, rec)
    if f_dp is None or p_dp is None:
        return
    ok = np.isfinite(f_dp) & np.isfinite(p_dp) & (np.abs(f_dp) < 1e30)
    d = np.where(ok, np.abs(p_dp - f_dp), 0.0)
    ref = np.where(ok, np.abs(f_sp - f_dp), 0.0) if f_sp is not None else None
    if d.ndim == 2:                        # (nz, nod2) -> worst level per node
        d = d.max(axis=0); ok = ok.any(axis=0)
        if ref is not None:
            ref = ref.max(axis=0)
    c = d[ok]
    o = np.sort(c)[::-1]; n1 = max(1, len(o) // 100)
    print(f"   |port-DP - fortran-DP| per node: max {c.max():.4g}  median {np.median(c):.4g}"
          f"  p99 {np.percentile(c, 99):.4g}")
    print(f"   concentration: {100 * (o[:n1] ** 2).sum() / (o ** 2).sum():.1f}%"
          f" of the sum of squares lives in the top 1% of nodes")
    if ref is not None:
        print(f"   for scale, max |fortran-SP - fortran-DP| = {ref[ok].max():.4g}")
    top = np.argsort(d)[::-1][:400]
    print(f"   top-400 nodes: median lat {np.median(lat[top]):+.1f}"
          f"   poleward of 60: {100 * np.mean(np.abs(lat[top]) > 60):.0f}%")
    a, _ = load(arms.get("fdp", ""), "a_ice", rec)
    if a is not None:
        a = a if a.ndim == 1 else a[0]
        edge = lambda m: float(np.mean((m > 0.05) & (m < 0.95)))
        print(f"   ice edge (0.05<a_ice<0.95): {edge(a[top]):.2f} of the top nodes"
              f" vs {edge(a):.2f} of all nodes"
              f" -> {'OVER' if edge(a[top]) > edge(a) else 'under'}-represented")


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    runroot = args[0]
    varlist = VARS_DEFAULT
    rec = -1
    if "--vars" in args:
        varlist = args[args.index("--vars") + 1].split(",")
    if "--rec" in args:
        rec = int(args[args.index("--rec") + 1])
    do_where = "--where" in args

    names = sorted(d for d in os.listdir(runroot)
                   if os.path.isdir(os.path.join(runroot, d)))
    arms = {n: os.path.join(runroot, n) for n in names}
    # Two noise families, two questions (see m16_faith_setup.sh AMP):
    #   *_s<seed>  sigma 2e-4 K  -- Suvarchal's amplitude, the climate-variability bar
    #   *_r<seed>  sigma 1e-6 K  -- rounding-scale, the bar that SP should actually match
    # Each CODE has its own ensemble (the port's since 2026-09-10, src/fesom_perturb.*). Normalising
    # a code's SP-DP departure by ITS OWN envelope is the apples-to-apples comparison: the two DP
    # trajectories have themselves diverged, so a shared envelope would measure sensitivities about
    # different trajectories (board §3b-year).
    noise_s = [n for n in names if n.startswith("fdp_s")]
    noise_r = [n for n in names if n.startswith("fdp_r")]
    pnoise_s = [n for n in names if n.startswith("pdp_s")]
    pnoise_r = [n for n in names if n.startswith("pdp_r")]
    # the port's noise arms must be judged against the baseline built from the SAME binary
    pbase = "pdp_f2" if "pdp_f2" in names else "pdp"
    noise = noise_s + noise_r + pnoise_s + pnoise_r

    print(f"runroot : {runroot}")
    print(f"arms    : {' '.join(names)}")
    print(f"record  : {rec}   variables: {','.join(varlist)}")
    print()

    verdict = {}
    for var in varlist:
        got, mask, src = stats(arms, var, rec)
        if got is None:
            print(f"## {var}: not found in any arm" + (f" ({src})" if src else ""))
            print()
            continue
        npts = int(mask.sum())
        print(f"## {var}   valid points {npts} of {mask.size}"
              f"   files {sorted(set(src.values()))}")
        for a in sorted(got):
            m = float(got[a][mask].mean())
            base = "fdp" if a.startswith("f") else "pdp"
            shift = ""
            if base in got and a != base:
                shift = f"   mean-shift vs {base} {m - float(got[base][mask].mean()):+.6e}"
            print(f"   {a:<12} mean {m:.10g}{shift}")

        pairs = {}
        if "fdp" in got and "fsp" in got:
            pairs["fortran"] = relL2(got["fsp"], got["fdp"], mask)
        if "pdp" in got and "psp" in got:
            pairs["port"] = relL2(got["psp"], got["pdp"], mask)
        for k, v in pairs.items():
            print(f"   relL2 SP vs DP  [{k:<7}] {v:.6e}")

        env  = {"s": [], "r": []}          # fortran
        penv = {"s": [], "r": []}          # port
        for n in noise:
            base = "fdp" if n.startswith("fdp") else pbase
            if n in got and base in got:
                e = relL2(got[n], got[base], mask)
                is_s = n in noise_s or n in pnoise_s
                (env if n.startswith("fdp") else penv)["s" if is_s else "r"].append(e)
                amp = "2e-4 K" if is_s else "1e-6 K"
                print(f"   relL2 noise vs own DP [{n}, sigma {amp}] {e:.6e}")
        if pbase in got and "pdp" in got and pbase != "pdp":
            d = relL2(got[pbase], got["pdp"], mask)
            print(f"   [check] {pbase} vs pdp (different binaries, both DP): {d:.6e}"
                  f"  {'BITWISE EQUAL' if d == 0.0 else '<-- NOT byte-neutral'}")

        # the two code paths compared against each other, at equal precision. This is the number
        # that decides whether "the port and upstream are the same model" in the only sense a
        # chaotic system allows: not bit equality, but a difference no larger than the model's own
        # response to a nudge it cannot help making.
        cross = {}
        for p, label in (("dp", "port-DP vs fortran-DP"), ("sp", "port-SP vs fortran-SP")):
            fa, pa = f"f{p}", f"p{p}"
            if fa in got and pa in got:
                cross[p] = relL2(got[pa], got[fa], mask)
                print(f"   relL2 {label} {cross[p]:.6e}")

        if do_where and "pdp" in got and "fdp" in got:
            where(runroot, arms, var, rec)

        if "fortran" in pairs and "port" in pairs:
            r = pairs["port"] / pairs["fortran"] if pairs["fortran"] > 0 else float("nan")
            print(f"   RATIO port/fortran SP-DP departure : {r:.4f}")
            verdict[var] = (pairs, env, r, cross, penv)
        elif pairs:
            verdict[var] = (pairs, env, None, cross, penv)
        print()

    if verdict:
        print("=" * 72)
        print("SUMMARY")
        for var, (pairs, env, r, cross, penv) in verdict.items():
            line = f"  {var:<6}"
            for k in ("fortran", "port"):
                if k in pairs:
                    line += f"  {k}-SP/DP {pairs[k]:.3e}"
            if r is not None:
                line += f"  |  ratio {r:.3f}"
            print(line)
            if "dp" in cross:
                print(f"         port-vs-fortran at equal precision (DP): {cross['dp']:.3e}")
            for fam, amp in (("r", "1e-6 K"), ("s", "2e-4 K")):
                if not env[fam] and not penv[fam]:
                    continue
                bits = []
                if env[fam] and "fortran" in pairs:
                    e = max(env[fam])
                    bits.append(f"fortran {pairs['fortran'] / e:5.2f}x own env")
                if penv[fam] and "port" in pairs:
                    pe = max(penv[fam])
                    bits.append(f"port {pairs['port'] / pe:5.2f}x own env")
                elif env[fam] and "port" in pairs:
                    e = max(env[fam])
                    bits.append(f"port {pairs['port'] / e:5.2f}x FORTRAN's env (no port ensemble)")
                if env[fam] and "dp" in cross:
                    e = max(env[fam])
                    bits.append(f"code-vs-code {cross['dp'] / e:5.2f}x")
                print(f"         sigma {amp}:  " + " | ".join(bits))
        print()
        print("  Read. RATIO near 1 = the port loses as much to single precision as upstream does;")
        print("  that is the G4 statement, and it does not depend on either departure being small.")
        print("  The ENVELOPES say whether the departures matter at all. The 1e-6 K family is the")
        print("  honest comparator: it is a nudge the size of float32 rounding itself, so an SP-DP")
        print("  departure that sits at or below it is behaving exactly like rounding. The 2e-4 K")
        print("  family is Suvarchal's climate amplitude -- ~200x larger than SP rounding on a 10 K")
        print("  field, so 'below the 2e-4 envelope' is a generous statement, not a strong one, on")
        print("  a run this short. Over decades the two families converge as both saturate.")
        print()
        print("  Each code is now normalised by ITS OWN envelope where one exists. That is the")
        print("  apples-to-apples form: the two DP trajectories have diverged, so a shared envelope")
        print("  would compare sensitivities measured about different trajectories. A ratio near or")
        print("  below 1 means that code's single precision is doing no more than a rounding-sized")
        print("  nudge does to it.")
        print()
        print("  The code-vs-code line is the strongest claim available: if the port and upstream")
        print("  differ, at the SAME precision, by no more than the model's response to a nudge it")
        print("  cannot avoid making, then they are the same model in the only sense a chaotic")
        print("  system permits. If it sits ABOVE the envelope, the gap is a real port difference")
        print("  and belongs in the registry, not in the precision story.")


if __name__ == "__main__":
    main()
