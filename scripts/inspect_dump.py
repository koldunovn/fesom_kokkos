#!/usr/bin/env python3
"""
Read and pretty-print one or more FESOM2 reference-dump files produced by
the fesom_dump_shim Fortran module.

Binary record layout (little-endian, stream, no header):
    int32   step
    int32   substep_id
    int32   probe_global_id     (Fortran 1-based)
    int32   nlevels
    char[24] field_name         (blank-padded ASCII)
    real64  values[nlevels]

Usage:
    python3 inspect_dump.py <prefix>.<rrrrr> [more files ...]
    python3 inspect_dump.py --summary <files...>     # one line per record
    python3 inspect_dump.py --filter step=1 substep=1 field=density <files...>
"""
from __future__ import annotations
import struct
import sys

SUBSTEP_NAMES = {
    0:  "init",
    1:  "pressure_bv",
    2:  "sw_alpha_beta",
    3:  "pressure_force",
    4:  "mixing",
    5:  "vel_rhs",
    6:  "viscosity_filter",
    7:  "impl_vert_visc",
    8:  "ssh_rhs",
    9:  "ssh_solve",
    10: "update_vel",
    11: "compute_hbar",
    12: "eta_n",
    13: "ale_step",
    14: "gm_bolus",
    15: "solve_tracers",
    16: "update_thickness",
}

HEADER_FMT = "<iiii24s"        # 4 int32 + 24 char  ==> 40 bytes
HEADER_LEN = struct.calcsize(HEADER_FMT)


def read_records(path: str):
    with open(path, "rb") as fh:
        while True:
            hdr = fh.read(HEADER_LEN)
            if not hdr:
                return
            if len(hdr) != HEADER_LEN:
                raise IOError(f"truncated header in {path}: got {len(hdr)} bytes")
            step, sub, gid, nlev, name = struct.unpack(HEADER_FMT, hdr)
            name = name.decode("ascii", "replace").rstrip()
            payload = fh.read(8 * nlev)
            if len(payload) != 8 * nlev:
                raise IOError(f"truncated payload in {path}: want {8*nlev} got {len(payload)}")
            values = struct.unpack(f"<{nlev}d", payload)
            yield step, sub, gid, nlev, name, values


def parse_filters(args):
    filters = {}
    rest = []
    for a in args:
        if "=" in a and a.split("=", 1)[0] in ("step", "substep", "field", "probe"):
            k, v = a.split("=", 1)
            filters[k] = v
        else:
            rest.append(a)
    return filters, rest


def matches(filters, step, sub, gid, name):
    if "step" in filters and str(step) != filters["step"]:
        return False
    if "substep" in filters:
        want = filters["substep"]
        if want.isdigit():
            if str(sub) != want:
                return False
        else:
            if SUBSTEP_NAMES.get(sub, "?") != want:
                return False
    if "field" in filters and name != filters["field"]:
        return False
    if "probe" in filters and str(gid) != filters["probe"]:
        return False
    return True


def main():
    args = sys.argv[1:]
    summary = "--summary" in args
    args = [a for a in args if a != "--summary"]

    filters, files = parse_filters(args)
    if not files:
        print(__doc__)
        sys.exit(2)

    n_total = n_match = 0
    for path in files:
        for step, sub, gid, nlev, name, vals in read_records(path):
            n_total += 1
            if not matches(filters, step, sub, gid, name):
                continue
            n_match += 1
            sub_name = SUBSTEP_NAMES.get(sub, f"?({sub})")
            tag = f"step={step:5d}  sub={sub:2d}/{sub_name:<18s}  probe={gid:6d}  {name:<10s}  nlev={nlev:3d}"
            if summary:
                print(tag, " ", " ".join(f"{v:11.4e}" for v in vals[:3]),
                      "..." if nlev > 3 else "")
            else:
                print(tag)
                for k, v in enumerate(vals, start=1):
                    print(f"    nz={k:3d}  {v: .12e}")

    print(f"\n[{n_match}/{n_total} records matched filter]")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # head/less closed the pipe early — silently swallow.
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
