#!/usr/bin/env python3
"""M5.24 'where does step time go' across 3 regimes:
   CORE2@4N CPU (512c sweet spot) | farc@4N GPU (near parity) | NG5@16N GPU (comm-bound).
Left  : coarse composition (ocean / sea-ice / coupling-halo / forcing / other) -- the regime shift.
Right : ocean-phase detail (FCT, SSH(CG+halo), KPP, GM, ALE, ...) -- where the ocean time goes.
Parses FESOM_STEP_PROFILE from each run.log. -> docs/figures/m524_profile_phases.png
"""
import os, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN="/work/ab0995/a270088/port2/kokkos_gpu_runs"
CONFIGS=[
 ("CORE2 4N CPU\n(512c sweet spot)", f"{RUN}/m524_prof_core2_cpu_n4"),
 ("farc 4N GPU\n(near parity)",      f"{RUN}/m524_prof_parity_gpu"),
 ("NG5 16N GPU\n(comm-bound)",       f"{RUN}/m524_prof_ng5_gpu_n16"),
]
# coarse buckets (from the STEP PROFILE line)
CAT=["ocean","sea-ice","coupling (o2i/halo)","forcing","other (halo/host/MPI)"]
CATC={"ocean":"#2a9d8f","sea-ice":"#7b2cbf","coupling (o2i/halo)":"#e63946","forcing":"#f4a261","other (halo/host/MPI)":"#adb5bd"}
# ocean-phase detail (numbered phases)
PH={"13_fct":"FCT advection","7_ssh":"SSH solve (CG+halo)","3_mixing":"KPP mixing","1b_gm":"GM/Redi",
    "12_ale":"ALE","13b_trdiff":"tracer diff","6_ivisc":"impl vert visc","1_eos":"EOS/density",
    "4_velrhs":"momentum RHS","5_viscfilt":"visc filter","2_pgf":"pressure grad"}
PHORDER=["FCT advection","SSH solve (CG+halo)","KPP mixing","GM/Redi","ALE","tracer diff",
         "impl vert visc","EOS/density","momentum RHS","visc filter","pressure grad"]

def parse(d):
    lg=os.path.join(d,"run.log")
    if not os.path.exists(lg): return None
    t=open(lg,errors="ignore").read()
    ms=re.search(r"->\s*([0-9.]+)\s*s/step",t)
    mc=re.search(r"STEP PROFILE.*?forcing\s*([0-9.]+)%.*?sea-ice\s*([0-9.]+)%.*?coupling\s*([0-9.]+)%.*?ocean\s*([0-9.]+)%",t)
    if not mc: return None
    f,si,cp,oc=map(float,mc.groups())
    coarse={"ocean":oc,"sea-ice":si,"coupling (o2i/halo)":cp,"forcing":f,
            "other (halo/host/MPI)":max(0.0,100-(oc+si+cp+f))}
    ph={}
    for nm,pct in re.findall(r"\[fesom_prof\]\s+(\S+)\s+([0-9.]+)%",t):
        if nm in PH: ph[PH[nm]]=ph.get(PH[nm],0)+float(pct)
    return dict(sstep=float(ms.group(1)) if ms else None, coarse=coarse, ph=ph)

data=[(lab,parse(d)) for lab,d in CONFIGS]; data=[(l,p) for l,p in data if p]
if not data: print("no profile data yet"); raise SystemExit

fig,(axA,axB)=plt.subplots(1,2,figsize=(15.5,6.8),gridspec_kw=dict(width_ratios=[1,1.25]))

# --- Panel A: coarse stacked composition (one bar per config) ---
labels=[l for l,_ in data]; yA=np.arange(len(data))[::-1]
left=np.zeros(len(data))
for cat in CAT:
    vals=np.array([p["coarse"][cat] for _,p in data])
    axA.barh(yA,vals,left=left,color=CATC[cat],edgecolor="white",label=cat,height=0.62)
    for yi,v,l0 in zip(yA,vals,left):
        if v>=3: axA.text(l0+v/2,yi,f"{v:.0f}",va="center",ha="center",fontsize=8.5,color="white",fontweight="bold")
    left+=vals
axA.set_yticks(yA); axA.set_yticklabels([f"{l}\n{p['sstep']:.3f} s/step" for l,p in data],fontsize=9)
axA.set_xlabel("% of the timed step"); axA.set_xlim(0,100)
axA.set_title("(a) Coarse composition — the regime shift\ncoupling/halo: 0.7% (CPU) → 1.5% (parity) → 14% (comm-bound)",fontsize=10.5)
axA.legend(fontsize=8,loc="lower center",bbox_to_anchor=(0.5,-0.30),ncol=3)
axA.grid(axis="x",alpha=0.3)

# --- Panel B: ocean-phase detail (grouped bars) ---
n=len(data); yB=np.arange(len(PHORDER)); h=0.8/n
for i,(lab,p) in enumerate(data):
    vals=[p["ph"].get(k,0) for k in PHORDER]
    off=(i-(n-1)/2)*h
    axB.barh(yB-off,vals,height=h,color=["#1f77b4","#2ca02c","#d62728"][i%3],
             edgecolor="black",lw=0.3,label=lab.replace("\n"," "))
    for yi,v in zip(yB-off,vals):
        if v>=1: axB.text(v+0.1,yi,f"{v:.0f}",va="center",fontsize=6.8)
axB.set_yticks(yB); axB.set_yticklabels(PHORDER); axB.invert_yaxis()
axB.set_xlabel("% of the timed step"); axB.set_title("(b) Ocean-phase detail (rank0)\nNG5@16N: SSH(CG+halo) is #1 — the comm wall",fontsize=10.5)
axB.legend(fontsize=8,loc="lower right"); axB.grid(axis="x",alpha=0.3)

fig.suptitle("FESOM2 Kokkos M5.24 — where step time goes, across 3 regimes",fontsize=13,y=1.0)
fig.tight_layout(); fig.savefig("/home/a/a270088/port_kokkos/docs/figures/m524_profile_phases.png",bbox_inches="tight")
print(f"wrote m524_profile_phases.png  ({len(data)} configs)")
