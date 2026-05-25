#!/usr/bin/env python3
"""Progression of fixes — March 1959 Arctic ice/ocean drift vs Fortran PP.
 BUGGY    : core2_864_2yr_dt1200   (no 0.5, geographic winds)
 +0.5     : ice_fix_2yr            (ice_strength 0.5 restored)
 +windrot : ice_fix_windrot_2yr    (0.5 + vector_g2r wind rotation)
 Fortran  : fortran_pp_2yr         (target)

Run: PYTHONPATH=/home/a/a270088/PYTHON \
  /work/ab0995/a270088/mambaforge/envs/nereus/bin/python ice_progression.py
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc
F=pathlib.Path("/scratch/a/a270088/fortran_pp_2yr"); DIAG=F/"fesom.mesh.diag.nc"
SNAP="/work/ab0995/a270088/port/dt1800_snap/snap_001340.nc"
lat=np.asarray(nc.Dataset(DIAG)["lat"][:])
en=np.asarray(nc.Dataset(SNAP)["elem_nodes"][:]); en=en-1 if en.min()==1 else en
latE=lat[en].mean(1); MON=2; YR=1959; FILL=1e20; arcN=lat>70; arcE=latE>70
def clean(a): return np.where(np.abs(a)>FILL,np.nan,a)
def metrics(d,suf):
    try:
        u=clean(np.asarray(nc.Dataset(f"{d}/uice.fesom.{YR}{suf}.nc")["uice"][MON]))
        v=clean(np.asarray(nc.Dataset(f"{d}/vice.fesom.{YR}{suf}.nc")["vice"][MON]))
        si=np.hypot(u,v)
        uo=clean(np.asarray(nc.Dataset(f"{d}/u.fesom.{YR}{suf}.nc")["u"][MON,0,:]))
        vo=clean(np.asarray(nc.Dataset(f"{d}/v.fesom.{YR}{suf}.nc")["v"][MON,0,:]))
        so=np.hypot(uo,vo)
        mi=clean(np.asarray(nc.Dataset(f"{d}/m_ice.fesom.{YR}{suf}.nc")["m_ice"][MON]))
        return np.nanmean(si[arcN]),np.nanmean(so[arcE]),np.nanmean(mi[arcN]),np.nanmax(mi[arcN])
    except Exception as e:
        return None
runs=[("BUGGY (no fix)","/work/ab0995/a270088/port/core2_864_2yr_dt1200",".monthly"),
      ("+0.5 ice_strength","/work/ab0995/a270088/port/ice_fix_2yr",".monthly"),
      ("+windrot (g2r)","/work/ab0995/a270088/port/ice_fix_windrot_2yr",".monthly"),
      ("Fortran PP (target)","/scratch/a/a270088/fortran_pp_2yr","")]
print(f"=== March {YR}, Arctic >70N ===")
print(f"  {'run':22s} {'ice spd':>8s} {'oce spd':>8s} {'ice/oce':>8s} {'oceC/F':>7s} {'m_ice mn':>8s} {'m_ice mx':>8s}")
fref=metrics(runs[-1][1],runs[-1][2]); ocef=fref[1]
for nm,d,suf in runs:
    m=metrics(d,suf)
    if m is None: print(f"  {nm:22s}  NOT READY"); continue
    si,so,mn,mx=m
    print(f"  {nm:22s} {si:8.4f} {so:8.4f} {si/so:8.2f} {so/ocef:7.2f} {mn:8.2f} {mx:8.2f}")
