#!/usr/bin/env python3
"""m7_halo_sites.py — E.0 SITE ATTRIBUTION of the halo-wait pool (M7 session 11).

The gap census (m7_gap_census.py) keys gaps by (predecessor -> victim) kernel pair, but ALL
pack/unpack kernels of one exchange class share one tag (the lambda's enclosing function), so
every same-class MPI wait collapses into one 'halo -> halo' row and the SITES are invisible —
that is exactly how session 10's "the CG's exchanges are apparently NOT in this pool" misread
happened (they are the pool's largest component; their gaps were keyed halo->halo, not ->cg).

This walker recovers the sites: a MAXIMAL RUN of halo-class kernels is one exchange BLOCK; the
compute kernels bracketing the run name the call site; kernel count inside the run gives the
exchange count (device/device2: n/2 per exchange; deviceN: 2*nf per exchange, one block = one
call). Internal gaps (pack->unpack = the MPI wait) are summed per site and split into
MPI-covered / PCIe-covered, with the >0.1ms event count (L98: quote the threshold).

Usage: m7_halo_sites.py <trace.sqlite> [<trace2.sqlite> ...]
Window: same convention as the census (skip the first nsteps//3, steady state).
"""
import bisect
import collections
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "m7_gap_census", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "m7_gap_census.py"))
gc = importlib.util.module_from_spec(_spec)
sys.modules["m7_gap_census"] = gc
_spec.loader.exec_module(gc)          # main() is __main__-guarded; import is side-effect free

HALO = {"fesom_halo_exchange_device": "d1", "fesom_halo_exchange_device2": "d2",
        "fesom_halo_exchange_deviceN": "dN"}
MIN_GAP = 100_000   # ns — 0.1 ms, the census setting of record for package E (L98)


def walk(path):
    import sqlite3
    con = sqlite3.connect(path)
    kern = gc.load_kernels(con)
    mpi = gc.merge([(s, e) for s, e, _ in gc.load_mpi(con)])
    mcpy = gc.merge(gc.load_memcpy(con))
    mcpy_s = [x[0] for x in mcpy]
    mpi_s = [x[0] for x in mpi]
    bnds, nsteps, anchor = gc.step_boundaries(kern)
    k0 = max(1, nsteps // 3)
    lo, hi = bnds[k0], bnds[-1]
    win = len(bnds) - 1 - k0
    kern = [k for k in kern if k[0] >= lo and k[1] <= hi]

    sites = collections.defaultdict(lambda: dict(blocks=0, kern=0, gap=0, mpi=0,
                                                 cpy=0, ev01=0, cls=collections.Counter()))
    i, n = 0, len(kern)
    prev_tag = "<start>"
    while i < n:
        if kern[i][2] not in HALO:
            prev_tag = kern[i][2]
            i += 1
            continue
        j = i
        while j < n and kern[j][2] in HALO:
            j += 1
        run = kern[i:j]
        succ = kern[j][2] if j < n else "<end>"
        gap = gmpi = gcpy = ev = 0
        pe = run[0][1]
        for ks, ke, _ in run[1:]:
            if ks > pe:
                g = ks - pe
                gap += g
                gmpi += gc.covered(mpi, mpi_s, pe, ks)
                gcpy += gc.covered(mcpy, mcpy_s, pe, ks)
                if g >= MIN_GAP:
                    ev += 1
            pe = max(pe, ke)
        clsc = collections.Counter(HALO[k[2]] for k in run)
        key = (prev_tag, succ, tuple(sorted(clsc.items())))
        st = sites[key]
        st["blocks"] += 1; st["kern"] += len(run); st["gap"] += gap
        st["mpi"] += gmpi; st["cpy"] += gcpy; st["ev01"] += ev
        st["cls"].update(clsc)
        prev_tag = run[-1][2]
        i = j
    return sites, win, anchor


def report(path):
    sites, win, anchor = walk(path)
    NS = 1e-6
    print(f"\n===== {path}\n      window={win} steps (steady), anchor={anchor}, "
          f"threshold >0.1 ms (L98)")
    cls_ex = collections.Counter(); cls_gap = collections.Counter()
    for (pred, succ, cls), s in sites.items():
        d = dict(cls)
        main = max(d, key=d.get)
        cls_gap[main] += s["gap"]
        if "dN" in d:
            cls_ex["dN"] += s["blocks"]
        for c, kn in d.items():
            if c != "dN":
                cls_ex[c] += s["blocks"] * kn / 2
    for c in ("d1", "d2", "dN"):
        print(f"  class {c}: {cls_ex[c]/win:7.1f} exch/step  "
              f"{cls_gap[c]/win*NS:7.2f} ms/step wait")
    print(f"  TOTAL  : {sum(cls_ex.values())/win:7.1f} exch/step  "
          f"{sum(s['gap'] for s in sites.values())/win*NS:7.2f} ms/step wait  "
          f"({sum(s['ev01'] for s in sites.values())/win:.0f} events >0.1ms)")
    print(f"  {'site: predecessor -> successor [class x kernels]':<84} "
          f"{'blk/st':>6} {'ex/st':>6} {'wait':>7} {'mpi':>6} {'pcie':>5} {'us/ex':>6}")
    for (pred, succ, cls), s in sorted(sites.items(), key=lambda kv: -kv[1]["gap"]):
        g = s["gap"] / win * NS
        if g < 0.02:
            continue
        d = dict(cls)
        ex = s["blocks"] if "dN" in d else s["kern"] / 2
        cl = ",".join(f"{k}x{v}" for k, v in cls)
        print(f"  {pred+' -> '+succ+' ['+cl+']':<84} {s['blocks']/win:6.2f} "
              f"{ex/win:6.1f} {g:7.2f} {s['mpi']/win*NS:6.2f} {s['cpy']/win*NS:5.2f} "
              f"{s['gap']/max(1,ex)*1e-3:6.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
