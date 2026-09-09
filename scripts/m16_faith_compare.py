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

    names = sorted(d for d in os.listdir(runroot)
                   if os.path.isdir(os.path.join(runroot, d)))
    arms = {n: os.path.join(runroot, n) for n in names}
    noise = [n for n in names if n.startswith("fdp_s")]

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

        env = []
        for n in noise:
            if n in got and "fdp" in got:
                e = relL2(got[n], got["fdp"], mask)
                env.append(e)
                print(f"   relL2 noise vs DP [{n}] {e:.6e}")

        # the two code paths compared against each other, at equal precision:
        for p, label in (("dp", "port-DP vs fortran-DP"), ("sp", "port-SP vs fortran-SP")):
            fa, pa = f"f{p}", f"p{p}"
            if fa in got and pa in got:
                print(f"   relL2 {label} {relL2(got[pa], got[fa], mask):.6e}")

        if "fortran" in pairs and "port" in pairs:
            r = pairs["port"] / pairs["fortran"] if pairs["fortran"] > 0 else float("nan")
            print(f"   RATIO port/fortran SP-DP departure : {r:.4f}")
            verdict[var] = (pairs, env, r)
        elif pairs:
            verdict[var] = (pairs, env, None)
        print()

    if verdict:
        print("=" * 72)
        print("SUMMARY")
        for var, (pairs, env, r) in verdict.items():
            emax = max(env) if env else None
            line = f"  {var:<6}"
            for k in ("fortran", "port"):
                if k in pairs:
                    line += f"  {k}-SP/DP {pairs[k]:.3e}"
            if emax is not None:
                line += f"  |  FP64 noise envelope {emax:.3e}"
                for k in ("fortran", "port"):
                    if k in pairs:
                        tag = "ABOVE" if pairs[k] > emax else "below"
                        line += f"  [{k} {tag} envelope]"
            if r is not None:
                line += f"  |  ratio {r:.3f}"
            print(line)
        print()
        print("  Read: a departure BELOW the envelope is indistinguishable from the model's own")
        print("  sensitivity to a 2e-4 K nudge, so it is not evidence that SP hurts. A ratio near")
        print("  1 means the port loses exactly as much to single precision as upstream does.")


if __name__ == "__main__":
    main()
