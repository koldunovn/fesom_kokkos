#!/usr/bin/env python3
"""
The two ports carry INDEPENDENT copies of the restart layer, and they have to
write interchangeable files: the paper's bit-identity tier compares a
Kokkos-Serial restart against a C one field by field, which is only meaningful
if the field tables agree. A field added to one tree and not the other does not
fail to compile — it fails at the gate, hours later, on a node.

This compares the two rst_build tables and the two format versions directly,
and takes seconds. Run it after touching either tree's fesom_io_restart.

Exit 0 when the tables agree, 1 when they do not.
"""
import re
import sys

C_SRC  = "/home/a/a270088/port2/fesom2_port_zstar/src/fesom_io_restart.c"
KK_SRC = "/home/a/a270088/port_kokkos_int/src/fesom_io_restart.cpp"
C_HDR  = "/home/a/a270088/port2/fesom2_port_zstar/src/fesom_io_restart.h"
KK_HDR = "/home/a/a270088/port_kokkos_int/src/fesom_io_restart.h"

# Each entry is `v[n++] = rst_var{ {"name", "name2"}, base, [fld,] is_elem, levels, ncomp, halo };`
# The Kokkos table carries the extra Field pointer, so compare on the parts that
# describe the FILE: the variable names and the four shape/halo integers.
ENTRY = re.compile(r'v\[n\+\+\]\s*=\s*\(?rst_var\)?\{\s*\{([^}]*)\}\s*,(.*?)\};', re.S)


def table(path, has_fld):
    out = []
    for names, rest in ENTRY.findall(open(path).read()):
        nm = tuple(n.strip() for n in names.split(","))
        # trailing fields: [base,] [fld,] is_elem, levels, ncomp, halo
        tail = [t.strip() for t in rest.split(",")]
        shape = tail[-4:]
        out.append((nm, tuple(shape)))
    return out


def version(path):
    m = re.search(r'#define\s+FESOM_RESTART_FORMAT_VERSION\s+(\d+)', open(path).read())
    return m.group(1) if m else None


# In the Kokkos table each entry names a host pointer AND the Field behind it.
# Pairing the wrong two is invisible on Serial — the host and device views alias
# there and every sync is a no-op — and wrong on CUDA, where the write would
# pull one field and gather another. Nothing but this check sees it without a
# GPU node.
PAIR = re.compile(r'v\[n\+\+\]\s*=\s*rst_var\{\s*\{[^}]*\}\s*,\s*([^,]+),\s*([^,]+),')


def check_field_pairing():
    bad = 0
    entries = PAIR.findall(open(KK_SRC).read())
    for base, fld in entries:
        b, f = base.strip(), fld.strip().lstrip("&")
        if "?" in b:                      # the tke entry is a ternary
            b = b.split("?")[1].split(":")[0].strip()
            f = f.split("?")[1].split(":")[0].strip().lstrip("&")
        if f != b + "_fld":
            print(f"FIELD MISMATCH: base={b} paired with {f}, expected {b}_fld")
            bad = 1
    print(f"{len(entries)} Kokkos entries checked for base/Field pairing")
    return bad


def main():
    c, kk = table(C_SRC, False), table(KK_SRC, True)
    bad = check_field_pairing()
    if version(C_HDR) != version(KK_HDR):
        print(f"FORMAT VERSION differs: C={version(C_HDR)} Kokkos={version(KK_HDR)}")
        bad = 1
    cn = {e[0][0]: e for e in c}
    kn = {e[0][0]: e for e in kk}
    for name in sorted(set(cn) | set(kn)):
        if name not in kn:
            print(f"MISSING in Kokkos: {name}"); bad = 1
        elif name not in cn:
            print(f"MISSING in C:      {name}"); bad = 1
        elif cn[name] != kn[name]:
            print(f"DIFFERS {name}: C={cn[name]}  Kokkos={kn[name]}"); bad = 1
    print(f"{len(cn)} fields in C, {len(kn)} in Kokkos, format version {version(C_HDR)}")
    print("FAIL: the two restart tables have drifted" if bad
          else "PASS: the two restart tables agree")
    return bad


if __name__ == "__main__":
    sys.exit(main())
