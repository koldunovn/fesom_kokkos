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

def maxabs(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    return float(np.nanmax(np.abs(a - b))) if a.size else 0.0

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
    worst, fails = 0.0, []
    with nc.Dataset(sdir / last) as S, nc.Dataset(cdir / last) as C:
        common = [v for v in S.variables if v in C.variables]
        print(f"[gate] CORE2 CUDA fidelity vs Serial oracle @ {last} ({len(common)} fields):")
        for v in sorted(common):
            if S.variables[v].dtype.kind != "f":
                continue
            d = maxabs(S.variables[v][:], C.variables[v][:])
            ceil = CEIL.get(v, DEFAULT_CEIL)
            bad = (d != d) or (d > ceil)     # NaN or over ceiling
            if bad: fails.append((v, d, ceil))
            worst = max(worst, 0.0 if d != d else d)
            if d > 1e-9 or bad:
                print(f"    {v:18s} max|Δ|={d:.3e}  (ceil {ceil:.0e}){'  <== FAIL' if bad else ''}")
    if fails:
        print(f"[gate] FAIL — {len(fails)} field(s) over ceiling (a staleness/correctness regression). "
              f"pi/Serial cannot see this — see GPU_FIDELITY.md §M5.9.")
        sys.exit(1)
    print(f"[gate] PASS — all fields at the CUDA climate-close floor (worst {worst:.3e}); no staleness regression.")
    sys.exit(0)

if __name__ == "__main__":
    main()
