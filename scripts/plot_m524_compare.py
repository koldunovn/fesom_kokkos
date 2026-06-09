#!/usr/bin/env python3
"""4-way implementation comparison from docs/m524_compare_data.json:
   Kokkos-GPU (CUDA) vs Kokkos-CPU (Serial-128) vs C-port (pure C) vs Fortran (FESOM2 linfs+KPP).
  docs/figures/m524_compare_4way.png  -- 2x2 strong scaling (s/step vs nodes), one panel per mesh
  docs/figures/m524_compare_cpu_rel.png -- CPU implementations relative to Fortran (the Kokkos overhead)
All same workload (linfs+KPP, JRA55 1958, PHC IC); dars/NG5 at the CFL-stable dt=180.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D=json.load(open("/home/a/a270088/port_kokkos/docs/m524_compare_data.json"))
MESHES=["core2","farc","dars","ng5"]
MLAB={"core2":"CORE2 (0.13M, dt1800)","farc":"farc (0.64M, dt900)","dars":"dars (3.16M, dt180)","ng5":"NG5 (7.40M, dt180)"}
STY={"kokkos_gpu":("#d62728","-","o","Kokkos-GPU (A100x4/node)"),
     "kokkos_cpu":("#1f77b4","--","s","Kokkos-CPU (Serial 128c)"),
     "omp":       ("#9467bd","--","v","Kokkos-OpenMP (16x8)"),
     "cport":     ("#2ca02c",":","^","C-port (pure C, 128c)"),
     "fortran":   ("#000000","-.","D","Fortran FESOM2 (128c)")}
def srt(d): return sorted((int(n),v) for n,v in d.items())

plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":130,"savefig.dpi":150})
os.makedirs("/home/a/a270088/port_kokkos/docs/figures",exist_ok=True)

# ===== Figure 1: 2x2 strong scaling, one panel per mesh =====
fig,ax=plt.subplots(2,2,figsize=(13,10))
for k,m in enumerate(MESHES):
    a=ax[k//2,k%2]
    for impl,(c,ls,mk,lab) in STY.items():
        if impl in D[m] and D[m][impl]:
            pts=srt(D[m][impl]); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            a.plot(xs,ys,ls=ls,marker=mk,color=c,label=lab,lw=1.8,ms=6)
    a.set_xscale("log",base=2); a.set_yscale("log")
    a.set_xticks([1,2,4,8,16,32]); a.set_xticklabels([1,2,4,8,16,32])
    a.set_xlabel("nodes"); a.set_ylabel("s/step"); a.set_title(MLAB[m])
    a.legend(fontsize=8)
fig.suptitle("FESOM2 M5.24 — 4-way scaling: Kokkos-GPU vs Kokkos-CPU vs C-port vs Fortran (same linfs+KPP workload)",
             fontsize=12.5,y=0.995)
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig("/home/a/a270088/port_kokkos/docs/figures/m524_compare_4way.png",bbox_inches="tight"); plt.close(fig)

# ===== Figure 2: CPU implementations relative to Fortran (=1.0) — the Kokkos-CPU overhead =====
fig2,g=plt.subplots(figsize=(9.5,5.8))
CPU=["fortran","cport","omp","kokkos_cpu"]
for impl in CPU:
    c,ls,mk,lab=STY[impl]
    xs=[]; ys=[]
    for m in MESHES:
        if impl in D[m] and "fortran" in D[m]:
            for n,v in D[m][impl].items():
                if n in D[m].get("fortran",{}):
                    xs.append(f"{m}\n{n}N"); ys.append(v/D[m]["fortran"][n])
    # plot per mesh as grouped — simpler: scatter vs index
for j,m in enumerate(MESHES):
    if "fortran" not in D[m]: continue
    base=D[m]["fortran"]
    common=sorted(set(int(n) for n in base), key=int)
    for impl in CPU:
        if impl not in D[m]: continue
        c,ls,mk,lab=STY[impl]
        xs=[n for n in common if str(n) in D[m][impl]]
        ys=[D[m][impl][str(n)]/base[str(n)] for n in xs]
        g.plot([f"{m[:4]}{n}" for n in xs],ys,marker=mk,ls="none",color=c,ms=8,
                label=lab if j==0 else None)
g.axhline(1.0,color="k",lw=1); g.text(0,1.01,"Fortran = 1.0",fontsize=9)
g.set_ylabel("s/step relative to Fortran  (>1 = slower than Fortran)")
g.set_xlabel("mesh + nodes"); g.set_title("CPU implementations vs Fortran (linfs+KPP): the Kokkos-CPU abstraction overhead")
g.legend(fontsize=9); g.tick_params(axis='x',labelrotation=90,labelsize=7)
fig2.tight_layout(); fig2.savefig("/home/a/a270088/port_kokkos/docs/figures/m524_compare_cpu_rel.png",bbox_inches="tight"); plt.close(fig2)
print("wrote m524_compare_4way.png + m524_compare_cpu_rel.png")
for m in MESHES:
    fr=D[m].get("fortran",{}); kc=D[m].get("kokkos_cpu",{}); cp=D[m].get("cport",{})
    common=[n for n in fr if n in kc]
    if common:
        ov=np.mean([kc[n]/fr[n] for n in common])
        print(f"  {m}: kokkos-cpu/fortran avg = {ov:.2f}x  (n={common})")
