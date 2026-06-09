#!/usr/bin/env python3
"""M5.24 scaling figures from docs/m524_scaling_data.json (collect_m524_scaling.py).
Renders whatever has landed (robust to partial data):
  docs/figures/m524_scaling_overview.png  -- 2x2: strong-scaling / SYPD / node-for-node ratio / mesh trend
  docs/figures/m524_sypd.png              -- standalone SYPD vs nodes (GPU+CPU, all meshes)
s/step is ~dt-independent; dars measured at the CFL-stable dt=180, SYPD at production dt=240 via
SYPD=dt/(365*s_step) with a small CG correction measured from NG5 (dt=240 main vs dt=180 ref).
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("/home/a/a270088/port_kokkos/docs/m524_scaling_data.json"))
rows, meta = D["rows"], D["meta"]
DT_PROD, NOD2D = meta["dt_prod"], meta["nod2d"]
MESHES = ["core2","farc","dars","ng5"]
MLAB   = {"core2":"CORE2 (0.13M)","farc":"farc (0.64M)","dars":"dars (3.16M)","ng5":"NG5 (7.40M)"}
MSIZE  = {"core2":0.127,"farc":0.638,"dars":3.16,"ng5":7.40}
COL    = {"core2":"#1f77b4","farc":"#2ca02c","dars":"#ff7f0e","ng5":"#d62728"}
POWER  = 2160/560.0  # GPU node W / CPU node W

def series(mesh, back, ref=False):
    return {r["nodes"]: r["sstep"] for r in rows
            if r["mesh"]==mesh and r["backend"]==back and r["ref"]==ref and r["sane"]}

# dars sstep measured at dt=180; production dt=240. Correction = NG5 sstep(240)/sstep(180).
# dars+NG5 s/step measured at the CFL-stable dt=180; production dt=240. The step cost at 240 is
# ~3% higher than at 180 (CG iters 89->115, CG ~12% of step) -- measured on NG5's clean 16N 240/180 pair.
CORR = 1.03
def sypd(mesh, sstep):
    s = sstep*CORR if mesh in ("dars","ng5") else sstep
    return DT_PROD[mesh]/(365.0*s)

plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":130,"savefig.dpi":150})
os.makedirs("/home/a/a270088/port_kokkos/docs/figures", exist_ok=True)

# ============================ Figure 1: 2x2 overview ============================
fig, ax = plt.subplots(2,2,figsize=(14,10.5))

# (a) strong scaling -----------------------------------------------------------
a=ax[0,0]
for m in MESHES:
    g,c = series(m,"gpu"), series(m,"cpu")
    if g:
        x=sorted(g); a.plot(x,[g[n] for n in x],"-o",color=COL[m],label=f"{m} GPU")
        x0=x[0]; a.plot([x0,max(x)],[g[x0]*x0/x0, g[x0]*x0/max(x)],":",color=COL[m],lw=0.8,alpha=0.4)
    if c:
        x=sorted(c); a.plot(x,[c[n] for n in x],"--s",color=COL[m],mfc="white",label=f"{m} CPU")
a.set_xscale("log",base=2); a.set_yscale("log")
a.set_xticks([1,2,4,8,16,32]); a.set_xticklabels([1,2,4,8,16,32])
a.set_xlabel("nodes"); a.set_ylabel("s/step")
a.set_title("(a) Strong scaling — M5.24 (dotted = ideal 1/N per mesh)")
a.legend(fontsize=7.5,ncol=2)

# (b) SYPD vs nodes ------------------------------------------------------------
b=ax[0,1]
for m in MESHES:
    g,c = series(m,"gpu"), series(m,"cpu")
    if g:
        x=sorted(g); b.plot(x,[sypd(m,g[n]) for n in x],"-o",color=COL[m],label=f"{m} GPU")
    if c:
        x=sorted(c); b.plot(x,[sypd(m,c[n]) for n in x],"--s",color=COL[m],mfc="white")
b.axhspan(1,2,color="green",alpha=0.07); b.axhline(1,color="grey",lw=0.7,ls=":"); b.axhline(2,color="grey",lw=0.7,ls=":")
b.text(1.05,1.02,"1 SYPD",fontsize=8,color="grey"); b.text(1.05,2.02,"2 SYPD",fontsize=8,color="grey")
b.set_xscale("log",base=2); b.set_yscale("log"); b.set_xticks([1,2,4,8,16,32]); b.set_xticklabels([1,2,4,8,16,32])
b.set_xlabel("nodes"); b.set_ylabel("SYPD  (production dt: core2 1800, farc 900, dars/NG5 240)")
b.set_title("(b) SYPD vs nodes  (solid GPU, dashed CPU)"); b.legend(fontsize=7.5,ncol=2)

# (c) node-for-node GPU/CPU ratio ----------------------------------------------
c2=ax[1,0]
for m in MESHES:
    g,c = series(m,"gpu"), series(m,"cpu")
    common=sorted(set(g)&set(c))
    if common:
        c2.plot(common,[g[n]/c[n] for n in common],"-o",color=COL[m],label=m)
c2.axhline(1.0,color="k",lw=0.8); c2.text(1.05,1.03,"parity",fontsize=8)
c2.set_xscale("log",base=2); c2.set_xticks([1,2,4,8,16,32]); c2.set_xticklabels([1,2,4,8,16,32])
c2.set_xlabel("nodes"); c2.set_ylabel("GPU / CPU  (<1 = GPU faster)")
c2.set_title("(c) Node-for-node ratio (Kokkos CUDA / Serial-128)"); c2.set_ylim(0,2.6); c2.legend(fontsize=9)

# (d) mesh-size trend: 4-node ratio + per-watt ---------------------------------
d=ax[1,1]
xs=[]; rk=[]
for m in MESHES:
    g,c=series(m,"gpu"),series(m,"cpu")
    if 4 in g and 4 in c:
        xs.append(MSIZE[m]); rk.append(g[4]/c[4])
if xs:
    d.plot(xs,rk,"-o",color="#1a73e8",lw=2.4,ms=9,label="GPU/CPU per node (4N)")
    for m in MESHES:
        g,c=series(m,"gpu"),series(m,"cpu")
        if 4 in g and 4 in c:
            r=g[4]/c[4]; lab=f"{1/r:.1f}x fast" if r<1 else f"{r:.1f}x slow"
            d.annotate(f"{m}\n{lab}",(MSIZE[m],r),textcoords="offset points",xytext=(6,7),fontsize=8.5,fontweight="bold")
    d.plot(xs,[r*POWER for r in rk],"--D",color="#9aa0a6",lw=1.8,ms=7,label=f"x per-Wh (GPU {2160}W/CPU {560}W)")
d.axhline(1.0,color="k",ls=":",lw=0.8)
d.set_xscale("log"); d.set_xlabel("mesh size (M nodes)"); d.set_ylabel("GPU/CPU per node @4N")
d.set_xticks([0.127,0.638,3.16,7.40]); d.set_xticklabels(["0.13","0.64","3.16","7.4"])
d.set_title("(d) Mesh-size trend @4 nodes (throughput + per-watt)"); d.legend(fontsize=8)

fig.suptitle(f"FESOM2 C++/Kokkos M5.24 — GPU (A100x4/node) vs CPU (EPYC 128c/node) strong scaling   "
             f"[dars dt-corr x{CORR:.3f}]", fontsize=13, y=0.995)
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig("/home/a/a270088/port_kokkos/docs/figures/m524_scaling_overview.png", bbox_inches="tight")
plt.close(fig)

# ============================ Figure 2: standalone SYPD ============================
fig2,g=plt.subplots(figsize=(9.5,6.2))
for m in MESHES:
    gp,cp = series(m,"gpu"), series(m,"cpu")
    if gp:
        x=sorted(gp); g.plot(x,[sypd(m,gp[n]) for n in x],"-o",color=COL[m],lw=2.2,ms=7,label=f"{MLAB[m]} GPU")
    if cp:
        x=sorted(cp); g.plot(x,[sypd(m,cp[n]) for n in x],"--s",color=COL[m],mfc="white",lw=1.6,ms=6,label=f"{MLAB[m]} CPU")
g.axhspan(1,2,color="green",alpha=0.08)
g.axhline(1,color="grey",lw=0.8,ls=":"); g.axhline(2,color="grey",lw=0.8,ls=":")
g.text(33,1.0,"1 SYPD",fontsize=9,color="#2a7",va="center"); g.text(33,2.0,"2 SYPD",fontsize=9,color="#2a7",va="center")
# headline NG5 GPU annotations
gp=series("ng5","gpu")
for n in sorted(gp):
    s=sypd("ng5",gp[n])
    if n>=16: g.annotate(f"{s:.2f}",(n,s),textcoords="offset points",xytext=(4,6),fontsize=8.5,fontweight="bold",color=COL["ng5"])
g.set_xscale("log",base=2); g.set_yscale("log"); g.set_xticks([1,2,4,8,16,32]); g.set_xticklabels([1,2,4,8,16,32])
g.set_yticks([0.1,0.2,0.5,1,2,5,10,20,50]); g.set_yticklabels(["0.1","0.2","0.5","1","2","5","10","20","50"])
g.set_xlabel("nodes"); g.set_ylabel("SYPD (Simulated Years Per Day)")
g.set_title("FESOM2 Kokkos M5.24 — SYPD at production timestep\n"
            "(CORE2 30min, farc 15min, dars/NG5 4min step; green = 1-2 SYPD production band)", fontsize=11)
g.legend(fontsize=8.5,ncol=2,loc="upper left")
fig2.tight_layout(); fig2.savefig("/home/a/a270088/port_kokkos/docs/figures/m524_sypd.png", bbox_inches="tight")
plt.close(fig2)
print(f"wrote m524_scaling_overview.png + m524_sypd.png   (dars+NG5 measured @dt180, SYPD@dt240 via x{CORR:.2f} CG-corr)")
