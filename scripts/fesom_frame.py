#!/usr/bin/env python3
"""Vector-frame handling for FESOM output comparisons.

FESOM runs on a ROTATED grid (Euler alpha/beta/gamma = 50/15/-90 for CORE2: the pole is
moved off Greenland). Its velocity components live in that rotated frame internally.

Who writes what:
  * Fortran            -- ALWAYS geographic. io_meandata rotates vector pairs at output
                          (`do_rotation`, io_meandata.F90:2972-2979).
  * C port <  75406d3  -- ROTATED (native). Every C output archived before 2026-06-11 20:37.
  * C port >= 75406d3  -- GEOGRAPHIC by default (`FESOM_IO_VECTOR_FRAME=geo|rotated`).
  * Kokkos port        -- ROTATED (native). It has no frame knob (porting one is a
                          Post-Completion item on the M6 plan).

So comparing the Kokkos port's (u,v)/(uice,vice) against a Fortran or a modern-C reference
WITHOUT rotating is a frame mismatch. It is a nasty one: the rotation is an isometry, so
|speed|, ice extent and volume all look perfect while the COMPONENTS decorrelate. Measured
on the M5.23 CUDA 1-yr run vs the Fortran linfs+KPP reference (2026-07-12):

    uice corr   as-written 0.9187  ->  after r2g  0.9997
    vice corr   as-written 0.4266  ->  after r2g  0.9998

That 0.919 was on record in this repo as the "known F<->C ice-edge budget". It was never a
physics budget -- it was this frame mismatch. (It hid because m32_climate_compare compared
`uice` but not `vice`; a 0.43 would have been noticed immediately.)

Rule: rotate every rotated-frame source to geographic, then compare. Scalars are frame-free.
The C campaign gated its in-model rotation against this same offline transform at 7e-15
(job 25524763), so this is an equivalence, not an approximation.
"""
import numpy as np

# CORE2 rotated-grid Euler angles (namelist.config / gen_modules_rotate_grid.F90).
ALPHA_EULER, BETA_EULER, GAMMA_EULER = 50.0, 15.0, -90.0

# The M6 private mesh -- NOT /pool (whose nlvls/elvls were swapped 2026-07-03).
DEFAULT_MESH = "/work/ab0995/a270088/port2/mesh/core2"


class Rotator:
    """r2g for FESOM vector pairs on a given mesh. Port of vector_r2g
    (gen_modules_rotate_grid.F90:164-202)."""

    def __init__(self, mesh_dir=DEFAULT_MESH):
        raw = np.loadtxt(f"{mesh_dir}/nod2d.out", skiprows=1)
        lon, lat = raw[:, 1], raw[:, 2]        # nod2d.out is GEOGRAPHIC; the model rotates internally
        a, b, g = np.radians([ALPHA_EULER, BETA_EULER, GAMMA_EULER])
        self.RM = np.array([
            [np.cos(g)*np.cos(a) - np.sin(g)*np.cos(b)*np.sin(a),
             np.cos(g)*np.sin(a) + np.sin(g)*np.cos(b)*np.cos(a),
             np.sin(g)*np.sin(b)],
            [-np.sin(g)*np.cos(a) - np.cos(g)*np.cos(b)*np.sin(a),
             -np.sin(g)*np.sin(a) + np.cos(g)*np.cos(b)*np.cos(a),
             np.cos(g)*np.sin(b)],
            [np.sin(b)*np.sin(a), -np.sin(b)*np.cos(a), np.cos(b)]])
        self.glon, self.glat = np.radians(lon), np.radians(lat)
        xg = np.cos(self.glat)*np.cos(self.glon)
        yg = np.cos(self.glat)*np.sin(self.glon)
        zg = np.sin(self.glat)
        xr = self.RM[0, 0]*xg + self.RM[0, 1]*yg + self.RM[0, 2]*zg
        yr = self.RM[1, 0]*xg + self.RM[1, 1]*yg + self.RM[1, 2]*zg
        zr = self.RM[2, 0]*xg + self.RM[2, 1]*yg + self.RM[2, 2]*zg
        self.rlat = np.arcsin(np.clip(zr, -1.0, 1.0))
        self.rlon = np.arctan2(yr, xr)
        self.n = lon.size

    def r2g(self, u, v):
        """rotated components -> geographic components. Broadcasts over leading axes."""
        RM, rlat, rlon, glat, glon = self.RM, self.rlat, self.rlon, self.glat, self.glon
        txg = -v*np.sin(rlat)*np.cos(rlon) - u*np.sin(rlon)
        tyg = -v*np.sin(rlat)*np.sin(rlon) + u*np.cos(rlon)
        tzg = v*np.cos(rlat)
        txr = RM[0, 0]*txg + RM[1, 0]*tyg + RM[2, 0]*tzg      # M^T (vector_r2g:195-197)
        tyr = RM[0, 1]*txg + RM[1, 1]*tyg + RM[2, 1]*tzg
        tzr = RM[0, 2]*txg + RM[1, 2]*tyg + RM[2, 2]*tzg
        vlat = -np.sin(glat)*np.cos(glon)*txr - np.sin(glat)*np.sin(glon)*tyr + np.cos(glat)*tzr
        vlon = -np.sin(glon)*txr + np.cos(glon)*tyr
        return vlon, vlat

    def to_geo(self, u, v, frame):
        """No-op if `frame` is already 'geo'."""
        if frame == "geo":
            return u, v
        if frame == "rotated":
            return self.r2g(u, v)
        raise ValueError(f"frame must be 'geo' or 'rotated', got {frame!r}")


# Vector pairs FESOM writes. Comparing either member alone across frames is meaningless.
VECTOR_PAIRS = {"uice": "vice", "u": "v"}


def frame_of_c_output(path_or_sha_date=None):
    """Reference table for the archived C-port outputs (M6 Task 0.2, verified empirically
    by rotating each against its Fortran twin -- not merely inferred from commit dates)."""
    return {
        "c_tke_2yr":       "rotated",   # SHA 8260deae, 2026-06-11 03:04  (pre-75406d3)
        "c_zstar_tke_2yr": "rotated",   # SHA 45afc012, 2026-06-11 03:34
        "c_zstar_tke_5yr": "rotated",   # SHA 45afc012, 2026-06-11 04:09
        "c_zstar_2yr":     "rotated",   # SHA b4340fd9, 2026-06-10 21:02
        "kpp_5yr_fix":     "rotated",   # C-port KPP ref, 2026-05-25
        "c_mevp_2yr":      "geo",       # SHA 75406d3, 2026-06-11 21:04  (>= the geo-default commit)
        "c_mevp_5yr":      "geo",       # SHA 75406d3, 2026-06-11 21:43
        "c_evp_2yr":       "geo",       # SHA 75406d3, 2026-06-11 21:05
        "c_all3_1yr":      "geo",       # SHA df8b9a8,  2026-06-13 23:31
    }
