#!/usr/bin/env python3
"""Collect the M5.24 GPU+CPU strong-scaling sweep into one JSON + printed table.
Scans /work/.../kokkos_gpu_runs/m524_<mesh>_<gpu|cpu>_n<N>/ (+ m524ref_ng5180_* refs),
extracts best s/step (min over rep_a/rep_b), CG iters, T/S sanity; computes nod2D/rank and SYPD.
   s/step is ~dt-independent (same kernels/work per step); dars measured at the CFL-stable dt=180,
   SYPD reported at the production dt (dars/NG5 240) via the formula SYPD = dt/(365*s_step).
Usage: python3 scripts/collect_m524_scaling.py [--print]   ->  docs/m524_scaling_data.json
"""
import os, re, json, glob, sys

RUNDIR = "/work/ab0995/a270088/port2/kokkos_gpu_runs"
OUT    = "/home/a/a270088/port_kokkos/docs/m524_scaling_data.json"

NOD2D = {"core2":126858, "farc":638387, "dars":3160340, "ng5":7402886}
NL    = {"core2":48, "farc":48, "dars":57, "ng5":70}
DT_MEAS = {"core2":1800, "farc":900, "dars":180, "ng5":180}   # stable dt the m524_<mesh>_ s/step was measured at
                                                              # (dars+NG5 CFL-unstable at dt=240 from cold IC -> measured at 180)
DT_PROD = {"core2":1800, "farc":900, "dars":240, "ng5":240}   # production dt for SYPD headline
RANKS_PER_NODE = {"gpu":4, "cpu":128}

def sane(tmin,tmax,smin,smax):
    if None in (tmin,tmax,smin,smax): return None
    return (-3.0 <= tmin) and (tmax <= 32.0) and (3.0 <= smin) and (smax <= 43.0)

def parse_run(d):
    """min s/step over reps + CG + T/S from a run dir."""
    ssteps=[]; cg=None; tmin=tmax=smin=smax=None
    logs = sorted(glob.glob(os.path.join(d,"log_rep_*.txt")))
    if not logs:
        lg = os.path.join(d,"run.log")
        if os.path.exists(lg): logs=[lg]
    for lg in logs:
        try: txt=open(lg,errors="ignore").read()
        except OSError: continue
        for m in re.finditer(r"->\s*([0-9.]+)\s*s/step", txt):
            ssteps.append(float(m.group(1)))
        cgm = re.findall(r"\bit=(\d+)", txt)
        if cgm: cg=int(cgm[-1])
        tsm = re.findall(r"T\[\s*([-0-9.]+),\s*([-0-9.]+)\]\s*S\[\s*([-0-9.]+),\s*([-0-9.]+)\]", txt)
        if tsm:
            tmin,tmax,smin,smax = map(float, tsm[-1])
    best = min(ssteps) if ssteps else None
    return best, sorted(ssteps), cg, (tmin,tmax,smin,smax)

rows=[]
for d in sorted(glob.glob(os.path.join(RUNDIR,"m524*_n*"))):
    base=os.path.basename(d)
    if "prof" in base: continue
    mref = re.match(r"m524ref_(core2|farc|dars|ng5)(\d+)_(gpu|cpu)_n(\d+)$", base)  # e.g. m524ref_ng5180_gpu_n4
    mstd = re.match(r"m524_(core2|farc|dars|ng5)_(gpu|cpu)_n(\d+)$", base)
    if mref:
        mesh,dt_meas,back,nodes = mref.group(1),int(mref.group(2)),mref.group(3),int(mref.group(4)); ref=True
    elif mstd:
        mesh,back,nodes = mstd.group(1),mstd.group(2),int(mstd.group(3)); ref=False; dt_meas=DT_MEAS[mesh]
    else:
        continue
    sstep,reps,cg,(tmin,tmax,smin,smax) = parse_run(d)
    if sstep is None: continue
    ntasks = RANKS_PER_NODE[back]*nodes
    per_rank = NOD2D[mesh]/ntasks
    rows.append(dict(tag=base, ref=bool(ref), mesh=mesh, backend=back, nodes=nodes,
                     ntasks=ntasks, dist=ntasks, sstep=sstep, reps=reps, cg=cg,
                     dt_meas=dt_meas, nod2d_per_rank=round(per_rank,1),
                     tmin=tmin,tmax=tmax,smin=smin,smax=smax,
                     sane=sane(tmin,tmax,smin,smax)))

data = dict(meta=dict(nod2d=NOD2D, nl=NL, dt_meas=DT_MEAS, dt_prod=DT_PROD,
                      ranks_per_node=RANKS_PER_NODE), rows=rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(data, open(OUT,"w"), indent=1)

if "--print" in sys.argv or True:
    print(f"# {len(rows)} runs -> {OUT}\n")
    hdr=f"{'mesh':6} {'bk':3} {'N':>3} {'ntask':>5} {'nod2D/rk':>9} {'s/step':>8} {'dtM':>4} {'CG':>4} {'sane':>4}  T/S"
    print(hdr); print("-"*len(hdr))
    for r in sorted(rows, key=lambda x:(x["mesh"],x["backend"],x["ref"],x["nodes"])):
        ts=f"T[{r['tmin']},{r['tmax']}] S[{r['smin']},{r['smax']}]" if r['tmin'] is not None else ""
        flag = "" if r["sane"] else " <<BLOWN" if r["sane"] is False else ""
        tag = (r["mesh"]+"*") if r["ref"] else r["mesh"]
        print(f"{tag:6} {r['backend']:3} {r['nodes']:>3} {r['ntasks']:>5} {r['nod2d_per_rank']:>9.0f} "
              f"{r['sstep']:>8.4f} {r['dt_meas']:>4} {str(r['cg']):>4} {str(r['sane']):>4}  {ts}{flag}")
