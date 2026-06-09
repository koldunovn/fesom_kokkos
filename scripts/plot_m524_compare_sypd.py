#!/usr/bin/env python3
"""SYPD version of the 4/5-way comparison (mirror of plot_m524_compare.py fig 1, but SYPD not s/step):
   Kokkos-GPU | Kokkos-CPU (Serial) | Kokkos-OpenMP (16x8) | C-port | Fortran.
SYPD = dt_prod/(365*s_step); dars/NG5 measured at the CFL-stable dt=180, reported at production dt=240
(x1.03 CG-corr). core2 dt1800, farc dt900. -> docs/figures/m524_compare_4way_sypd.png
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
YT=[0.1,0.2,0.3,0.5,1,2,3,5,10,20,30,50,100]   # plain-number SYPD ticks (no 10^x)

D=json.load(open("/home/a/a270088/port_kokkos/docs/m524_compare_data.json"))
MESHES=["core2","farc","dars","ng5"]
MLAB={"core2":"CORE2 (0.13M, dt1800)","farc":"farc (0.64M, dt900)","dars":"dars (3.16M, dt240)","ng5":"NG5 (7.40M, dt240)"}
DT_PROD={"core2":1800,"farc":900,"dars":240,"ng5":240}; CORR=1.03
STY={"kokkos_gpu":("#d62728","-","o","Kokkos-GPU (A100x4/node)"),
     "kokkos_cpu":("#1f77b4","--","s","Kokkos-CPU (Serial 128c)"),
     "omp":       ("#9467bd","--","v","Kokkos-OpenMP (16x8)"),
     "cport":     ("#2ca02c",":","^","C-port (pure C, 128c)"),
     "fortran":   ("#000000","-.","D","Fortran FESOM2 (128c)")}
def sypd(m,s): return DT_PROD[m]/(365.0*(s*CORR if m in("dars","ng5") else s))
def srt(d): return sorted((int(n),v) for n,v in d.items())

plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":130,"savefig.dpi":150})
os.makedirs("/home/a/a270088/port_kokkos/docs/figures",exist_ok=True)
fig,ax=plt.subplots(2,2,figsize=(13,10))
for k,m in enumerate(MESHES):
    a=ax[k//2,k%2]
    for impl,(c,ls,mk,lab) in STY.items():
        if impl in D[m] and D[m][impl]:
            pts=srt(D[m][impl]); xs=[p[0] for p in pts]; ys=[sypd(m,p[1]) for p in pts]
            a.plot(xs,ys,ls=ls,marker=mk,color=c,label=lab,lw=1.8,ms=6)
    if m in ("dars","ng5"):  # the production 1-2 SYPD band only meaningful for the big meshes
        a.axhspan(1,2,color="green",alpha=0.08)
        a.axhline(1,color="grey",lw=0.7,ls=":"); a.axhline(2,color="grey",lw=0.7,ls=":")
    a.set_xscale("log",base=2); a.set_yscale("log")
    a.set_xticks([1,2,4,8,16,32]); a.set_xticklabels([1,2,4,8,16,32])
    a.yaxis.set_major_locator(FixedLocator(YT))
    a.yaxis.set_major_formatter(FixedFormatter([("%g"%t) for t in YT]))
    a.yaxis.set_minor_locator(FixedLocator([])); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("nodes"); a.set_ylabel("SYPD"); a.set_title(MLAB[m]); a.legend(fontsize=8)
fig.suptitle("FESOM2 M5.24 — 4/5-way SYPD: Kokkos-GPU vs Kokkos-CPU vs Kokkos-OpenMP vs C-port vs Fortran\n"
             "(SYPD at production timestep; dars/NG5 green band = 1-2 SYPD)",fontsize=12,y=0.997)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig("/home/a/a270088/port_kokkos/docs/figures/m524_compare_4way_sypd.png",bbox_inches="tight")
print("wrote m524_compare_4way_sypd.png")
# quick SYPD summary at 4N
for m in MESHES:
    print(f"  {m} 4N SYPD:", {i:round(sypd(m,D[m][i]['4']),2) for i in STY if i in D[m] and '4' in D[m][i]})
