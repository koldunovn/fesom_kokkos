#!/usr/bin/env python3
"""Collect the 4-way CPU/GPU implementation comparison into one JSON:
   kokkos_gpu (CUDA) | kokkos_cpu (Serial-128) | cport (pure C) | fortran (FESOM2 linfs+KPP).
Kokkos from docs/m524_scaling_data.json; C-port from m524cport.*.out (warm-cache long205 wall-diff);
Fortran from m524fort.*.out (Runtime-for-all-timesteps/nsteps). -> docs/m524_compare_data.json
"""
import json, re, glob, os
RUN="/work/ab0995/a270088/port2/kokkos_gpu_runs"
KJSON="/home/a/a270088/port_kokkos/docs/m524_scaling_data.json"
OUT="/home/a/a270088/port_kokkos/docs/m524_compare_data.json"
MESHES=["core2","farc","dars","ng5"]
data={m:{} for m in MESHES}

# --- Kokkos GPU + CPU(Serial) from the main scaling JSON (sane, non-ref) ---
K=json.load(open(KJSON))
for r in K["rows"]:
    if r["ref"] or not r["sane"]: continue
    impl = "kokkos_gpu" if r["backend"]=="gpu" else "kokkos_cpu"
    data[r["mesh"]].setdefault(impl,{})[str(r["nodes"])]=r["sstep"]

# --- C-port: warm-cache wall-diff. Per (mesh,nodes) keep the POSITIVE result from the LONGEST span
#     (long205 > long105 > the early init-dominated long45); glob order is non-deterministic so we
#     dedup explicitly by span rather than last-file-wins. ng5-32N long205 attempts node-Killed -> use long105 retry.
cport_cand={}   # (mesh,nodes) -> list of (span, sstep)
for f in glob.glob(f"{RUN}/m524cport.*.out"):
    t=open(f,errors="ignore").read()
    mt=re.search(r"CPORT m524_(core2|farc|dars|ng5)_cport_n(\d+)",t)
    if not mt: continue
    for sp,v in re.findall(r"\[cport\][^\n]*long(\d+)[^\n]*->\s*(-?[0-9.]+)\s*s/step",t):
        cport_cand.setdefault((mt.group(1),mt.group(2)),[]).append((int(sp),float(v)))
for (mesh,n),cands in cport_cand.items():
    kc=data[mesh].get("kokkos_cpu",{})                 # plausible vs the Kokkos-Serial CPU twin (drops crashed
    ok=[(sp,v) for sp,v in cands                        # 64-level-cap runs + node-Killed negatives at selection time,
        if v>0 and (n not in kc or 0.4*kc[n] <= v <= 1.6*kc[n])]   # so a bad span doesn't shadow the good one)
    if ok:                                             # longest plausible span = most init-amortized estimate
        data[mesh].setdefault("cport",{})[n]=max(ok,key=lambda x:x[0])[1]

# --- Fortran: the [fortran] line (Runtime/nsteps) ---
for f in glob.glob(f"{RUN}/m524fort.*.out"):
    t=open(f,errors="ignore").read()
    mt=re.search(r"FORTRAN m524_(core2|farc|dars|ng5)_fortran_n(\d+)",t)
    st=re.search(r"\[fortran\][^\n]*->\s*([0-9.]+)\s*s/step",t)
    if mt and st: data[mt.group(1)].setdefault("fortran",{})[mt.group(2)]=float(st.group(1))

# --- Kokkos-OpenMP (16 ranks x 8 threads): from the run dirs, internal loop timer, min over reps ---
for d in glob.glob(f"{RUN}/m524_*_omp_n*"):
    mt=re.match(r".*/m524_(core2|farc|dars|ng5)_omp_n(\d+)$",d)
    if not mt: continue
    ss=[]
    for lg in glob.glob(f"{d}/log_rep_*.txt"):
        for m2 in re.finditer(r"->\s*([0-9.]+)\s*s/step",open(lg,errors="ignore").read()): ss.append(float(m2.group(1)))
    if ss: data[mt.group(1)].setdefault("omp",{})[mt.group(2)]=min(ss)

# drop implausible cport/fortran points (crashed runs -> bogus wall-diff): a CPU twin must be
# within 0.4-1.6x the Kokkos-Serial CPU at the same node (e.g. NG5 C-port crashed on its 64-level cap).
for m in MESHES:
    kc=data[m].get("kokkos_cpu",{})
    for impl in ("cport","fortran","omp"):
        if impl in data[m]:
            data[m][impl]={n:v for n,v in data[m][impl].items() if n not in kc or 0.4*kc[n] <= v <= 1.6*kc[n]}
            if not data[m][impl]: del data[m][impl]

json.dump(data,open(OUT,"w"),indent=1)
ORD=["kokkos_gpu","kokkos_cpu","omp","cport","fortran"]
for m in MESHES:
    print(f"=== {m} ===")
    for i in ORD:
        if i in data[m]:
            print(f"  {i:11}", {int(n):round(v,4) for n,v in sorted(data[m][i].items(),key=lambda x:int(x[0]))})
print(f"\n-> {OUT}")
