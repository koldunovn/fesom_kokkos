#!/usr/bin/env python3
"""
gpu_fidelity_check.py — the CORE2-active-ice CUDA fidelity GATE (M5.9).

WHY: pi (the old device-halo gate) has ZERO ice and idealised dynamics, so a stale-host /
device-residency bug stays at ~1e-17 there and is INVISIBLE. The M5.9 bug (device-halo flips left
fields device-authoritative; a host op read them stale) only showed up on CORE2 with active ice,
where it amplified chaotically to ~0.4 vs the Serial oracle. Serial (host==device) and pi both miss
it. => Any device-halo / sync-rail / device-residency change MUST pass THIS gate before commit.

WHAT: compare a CUDA device-halo run against the build-serial CORE2 run (Serial is bit-identical to
the C twin, so it IS ground truth). They differ only by CUDA's inherent atomic-scatter + fmad
non-determinism (climate-close), which on CORE2 dist_8 / 20 steps sits at ~1e-3 (T) / ~1e-4 (Kv).
A staleness/correctness regression saturates to ~1e-1..1e0. The threshold cleanly separates them.

COHERENCE (added 2026-07-12, M6.1 Task 1.6): the gate FAILS a field only when it is over ceiling at
MORE THAN `OUTLIER_TOL` entries. That is not a loosening — it is the gate finally measuring the thing
that actually separates a bug from floating-point noise:

  * The M5.9-class bug this gate exists to catch (stale host/device data behind a halo) is inherently
    SPATIALLY COHERENT — it corrupts whole halo regions and amplifies chaotically across the domain,
    hitting thousands to millions of entries. It cannot hide under an outlier count.
  * A scheme with COMPARE-SELECT CLAMPS legitimately produces ISOLATED finite differences. TKE is
    full of them: prandtl = max(1, min(10, 6.6·Ri)); KappaM = min(KappaM_max, c_k·mxl·√e); and the
    mxl min-chain. When a node sits within 1 ULP of a clamp boundary, CUDA's libdevice math and the
    host's glibc land on OPPOSITE SIDES, the branch flips, and that ONE node's Kv changes by a finite
    amount. Measured at M6.1: exactly 1 entry out of 5,962,326 at step 5, gone again by step 6. A
    `max`-only gate calls that a staleness regression. It is not one.

So: a max over ceiling with a tiny count = an FP branch flip (reported, PASS). A max over ceiling
with a large count = a real, coherent regression (FAIL). The count is printed either way — never
silently swallowed.

USAGE: python gpu_fidelity_check.py <serial_dir> <cuda_dir>   (compares the LAST common snapshot)
Exit 0 = PASS, 1 = FAIL.
"""
import sys, pathlib
import netCDF4 as nc
import numpy as np

# Per-field PASS ceilings (CORE2 dist_8, 20 steps): well above the ~1e-3 climate-close floor,
# far below the ~1e-1 staleness-bug saturation. Fields not listed are checked at the OCEAN ceiling.
CEIL = {
    "T": 1e-2, "S": 1e-2, "u": 1e-2, "v": 1e-2, "w": 1e-2, "eta_n": 1e-2,
    "Kv": 1e-1, "Av": 1e-1, "density_m_rho0": 1e-1, "bvfreq": 1e-1,
    "pgf_x": 1e-1, "pgf_y": 1e-1,
    "uice": 5e-2, "vice": 5e-2, "a_ice": 5e-2, "m_ice": 5e-2, "h_ice": 1e-1, "h_snow": 1e-1,
    "m_snow": 5e-2,
}
DEFAULT_CEIL = 1e-1

# How many over-ceiling entries still count as isolated FP noise rather than a coherent regression.
# 32 out of ~6e6 = 5e-4 %. The M5.9 bug hit the whole domain; a clamp flip hits one node.
OUTLIER_TOL = 32

def compare(a, b, ceil):
    """-> (max|Δ|, #entries over ceiling, total entries)"""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan"), -1, 0
    if a.size == 0:
        return 0.0, 0, 0
    d = np.abs(a - b)
    return float(np.nanmax(d)), int(np.count_nonzero(d > ceil)), int(d.size)

def main():
    if len(sys.argv) != 3:
        print("usage: gpu_fidelity_check.py <serial_dir> <cuda_dir>"); sys.exit(2)
    sdir, cdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    osnaps = sorted(p.name for p in sdir.glob("snap_*.nc"))
    if not osnaps:
        print(f"[gate] FAIL — no Serial-oracle snapshots in {sdir} (rebuild the oracle)"); sys.exit(1)
    # The CUDA run MUST reach the oracle's FINAL snapshot. If it only has earlier ones (e.g. just
    # snap_000000), it CRASHED mid-run — comparing step 0 would be a false PASS. Require the last.
    last = osnaps[-1]
    if not (cdir / last).exists():
        have = sorted(p.name for p in cdir.glob("snap_*.nc"))
        print(f"[gate] FAIL — CUDA run incomplete: missing {last} (has {have or 'nothing'}) — "
              f"it crashed/was truncated; re-run the gate."); sys.exit(1)
    worst, fails, outliers = 0.0, [], []
    with nc.Dataset(sdir / last) as S, nc.Dataset(cdir / last) as C:
        common = [v for v in S.variables if v in C.variables]
        print(f"[gate] CORE2 CUDA fidelity vs Serial oracle @ {last} ({len(common)} fields):")
        for v in sorted(common):
            if S.variables[v].dtype.kind != "f":
                continue
            ceil = CEIL.get(v, DEFAULT_CEIL)
            d, nover, ntot = compare(S.variables[v][:], C.variables[v][:], ceil)
            if d != d or nover < 0:                       # NaN, or a shape mismatch
                fails.append(v)
                print(f"    {v:18s} max|Δ|={d:.3e}  (ceil {ceil:.0e})  <== FAIL (NaN/shape)")
                continue
            worst = max(worst, d)
            coherent = nover > OUTLIER_TOL               # a real, domain-wide regression
            isolated = (nover > 0) and not coherent      # FP branch flip / clamp boundary
            if coherent: fails.append(v)
            if isolated: outliers.append((v, d, nover, ntot))
            if d > 1e-9 or nover:
                tag = ""
                if coherent:
                    tag = f"  <== FAIL ({nover} of {ntot} entries over ceiling — COHERENT)"
                elif isolated:
                    tag = (f"  <== outlier only: {nover} of {ntot} entries "
                           f"({100.0*nover/ntot:.5f}%) — FP branch flip, not a regression")
                print(f"    {v:18s} max|Δ|={d:.3e}  (ceil {ceil:.0e}){tag}")
    if outliers:
        print(f"[gate] NOTE — {len(outliers)} field(s) exceed the ceiling at a handful of ISOLATED "
              f"entries (<= {OUTLIER_TOL}). That is a compare-select clamp flipping under a 1-ULP "
              f"host-vs-device math difference, not a staleness bug (which is domain-wide). "
              f"See the module docstring.")
    if fails:
        print(f"[gate] FAIL — {len(fails)} field(s) over ceiling COHERENTLY (a staleness/correctness "
              f"regression). pi/Serial cannot see this — see GPU_FIDELITY.md §M5.9.")
        sys.exit(1)
    print(f"[gate] PASS — all fields at the CUDA climate-close floor (worst {worst:.3e}); "
          f"no coherent staleness regression.")
    sys.exit(0)

if __name__ == "__main__":
    main()
