#!/usr/bin/env python3
"""Vertical-profile diagnostic of the C-vs-Fortran SST/upper-ocean bias.
C dt=1800 (eps fix, eps_2yr_dt1800) vs Fortran+PP dt=1800 (fortran_pp_2yr), 1959.

Hypothesis (code analysis): the C lacks shortwave penetration (use_sw_pene=.true.
in CORE2; Fortran deposits sw_3d through the column, C dumps all heat_flux in the
surface layer) -> warm surface / cold just below = vertical REDISTRIBUTION, not
net heat. Test: global + regional T(z)/S(z) and the difference profiles; the ΔT(z)
crossover should sit at the shortwave absorption depth, and upper-ocean heat
content should ~match (redistribution).

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python eps_vertical_profiles.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

C = "/work/ab0995/a270088/port/eps_2yr_dt1800"
F = "/scratch/a/a270088/fortran_pp_2yr"
DIAG = f"{F}/fesom.mesh.diag.nc"
OUT = pathlib.Path("/home/a/a270088/port2/fesom2_port/docs/validation_eps_2yr_dt1800")
FILL = 1e29

d = nc.Dataset(f"{C}/temp.fesom.1959.monthly.nc")
Z = np.asarray(d.variables["Z"][:]); lat = np.asarray(d.variables["lat"][:])
d.close()
# node area from elem_area + face_nodes
dd = nc.Dataset(DIAG); ea = np.asarray(dd.variables["elem_area"][:])
fn = np.asarray(dd.variables["face_nodes"][:]); dd.close()
fn = fn.T if fn.shape[0] == 3 else fn; fn = fn-1 if fn.min()==1 else fn
narea = np.zeros(len(lat))
for k in range(3): np.add.at(narea, fn[:,k], ea/3.0)

def annual3d(path, var, ext):
    a = np.asarray(nc.Dataset(f"{path}/{var}.fesom.1959{ext}.nc").variables[var][:])  # (12,nz,n)
    a = np.where(np.abs(a) < FILL, a, np.nan)
    return np.nanmean(a, axis=0)   # (nz, n)

cT = annual3d(C,"temp",".monthly"); fT = annual3d(F,"temp","")
cS = annual3d(C,"salt",".monthly"); fS = annual3d(F,"salt","")
# COMMON wet mask per level = where Fortran is valid (correct bathymetry). The C
# writes S=0/garbage below the seafloor, so we must restrict BOTH codes to the
# Fortran-wet nodes for a fair comparison (else the C dry-node zeros fake a deep
# cold+fresh bias).
WET = np.isfinite(fT) & (np.abs(fT) < FILL)
allm = np.ones(len(lat), bool)

def prof(field3d, mask):
    """area-weighted mean per level over (Fortran-wet ∧ region) nodes."""
    nz = field3d.shape[0]; out = np.full(nz, np.nan)
    for k in range(nz):
        v = field3d[k]; ok = WET[k] & mask & np.isfinite(v)
        if ok.sum() > 50: out[k] = np.sum(v[ok]*narea[ok]) / np.sum(narea[ok])
    return out

print("=== global area-weighted mean profiles (annual 1959) ===")
print(" depth   T_C    T_F   dT(C-F)  | S_C    S_F    dS(C-F)")
gcT,gfT,gcS,gfS = prof(cT,allm),prof(fT,allm),prof(cS,allm),prof(fS,allm)
for k in range(20):
    print(f" {Z[k]:6.0f}  {gcT[k]:5.2f}  {gfT[k]:5.2f}  {gcT[k]-gfT[k]:+6.3f}  | {gcS[k]:5.2f} {gfS[k]:5.2f} {gcS[k]-gfS[k]:+6.3f}")

# sanity: deep-level node count + S range (fill contamination?)
for k in (10, 19):  # ~95m, ~450m
    vC = cS[k]; vF = fS[k]; okC = np.isfinite(vC); okF = np.isfinite(vF)
    print(f"  [sanity z={Z[k]:.0f}m] C: nwet={okC.sum()} S∈[{np.nanmin(vC):.2f},{np.nanmax(vC):.2f}] | "
          f"F: nwet={okF.sum()} S∈[{np.nanmin(vF):.2f},{np.nanmax(vF):.2f}]")

# upper-ocean heat content (proxy: depth-integral of T) C vs F, global mean
dz = np.abs(np.gradient(Z))  # layer thickness approx, length nz
def colint(prof_, top_m):
    sel = np.abs(Z) <= top_m
    return np.nansum(prof_[sel]*dz[sel])
for top in (50,100,300):
    print(f"  heat-content proxy (∫T dz, 0-{top}m): C={colint(gcT,top):.1f}  F={colint(gfT,top):.1f}  "
          f"Δ={colint(gcT,top)-colint(gfT,top):+.2f}  degC·m")

# ---- figure ----
bands = [("Tropics |lat|<20", np.abs(lat)<20), ("Subtropics 20-40", (np.abs(lat)>=20)&(np.abs(lat)<40)),
         ("Midlat 40-60", (np.abs(lat)>=40)&(np.abs(lat)<60)), ("High-lat >60", np.abs(lat)>=60)]
fig, ax = plt.subplots(2,3, figsize=(16,10))
top = np.abs(Z) <= 300
ax[0,0].plot(gcT[top],Z[top],'-s',ms=3,label="C (dt1800,eps)",color="tab:red")
ax[0,0].plot(gfT[top],Z[top],'-o',ms=3,label="Fortran",color="k"); ax[0,0].set_title("global T(z)"); ax[0,0].set_xlabel("degC"); ax[0,0].legend()
ax[0,1].plot((gcT-gfT)[top],Z[top],'-s',ms=3,color="tab:purple"); ax[0,1].axvline(0,color="grey",lw=.7); ax[0,1].set_title("ΔT = C − Fortran"); ax[0,1].set_xlabel("degC")
ax[0,2].plot(gcS[top],Z[top],'-s',ms=3,label="C",color="tab:red"); ax[0,2].plot(gfS[top],Z[top],'-o',ms=3,label="Fortran",color="k"); ax[0,2].set_title("global S(z)"); ax[0,2].set_xlabel("PSU"); ax[0,2].legend()
ax[1,0].plot((gcS-gfS)[top],Z[top],'-s',ms=3,color="tab:purple"); ax[1,0].axvline(0,color="grey",lw=.7); ax[1,0].set_title("ΔS = C − Fortran"); ax[1,0].set_xlabel("PSU")
for nm,m in bands:
    ax[1,1].plot((prof(cT,m)-prof(fT,m))[top], Z[top], '-', label=nm)
ax[1,1].axvline(0,color="grey",lw=.7); ax[1,1].set_title("ΔT(z) by latitude band"); ax[1,1].set_xlabel("degC"); ax[1,1].legend(fontsize=8)
for nm,m in bands:
    ax[1,2].plot((prof(cS,m)-prof(fS,m))[top], Z[top], '-', label=nm)
ax[1,2].axvline(0,color="grey",lw=.7); ax[1,2].set_title("ΔS(z) by latitude band"); ax[1,2].set_xlabel("PSU"); ax[1,2].legend(fontsize=8)
for a in ax.flat: a.set_ylabel("depth [m]"); a.grid(alpha=.3)
fig.suptitle("Vertical profiles: C (dt1800, eps fix) vs Fortran+PP — annual 1959\n"
             "warm-surface/cold-below ΔT = missing shortwave penetration (use_sw_pene)?", fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"vertical_profiles.png", dpi=120, bbox_inches="tight")
print("wrote", OUT/"vertical_profiles.png")
