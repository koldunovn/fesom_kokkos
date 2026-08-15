#!/usr/bin/env python3
"""M11 negative control: corrupt one com_info halo entry so the model-side gate must fire.

The halo/dist correctness gate we rely on for every injected/generated partition is
`fesom_halo_identity_test` (src/fesom_halo.cpp:217, called unconditionally from
fesom_main.cpp:368 on npes>1). A gate is only worth quoting if it has been shown to
FAIL on a broken input, so this tool manufactures exactly one broken input.

Corruption model (`--mode swap-rlist`, the default): swap two entries of the nod2D
`rlist` in one rank's com_info file. rlist holds the LOCAL indices into which the
received halo values are unpacked, and in a healthy dist it is the identity map over
the halo tail (myDim+1 … myDim+eDim). Swapping two of its entries:

  * changes no count, no pointer, no message size and no wire traffic — the run does
    not crash, it simply puts two neighbours' values in each other's slots;
  * is the silent-wrongness failure mode a partitioning campaign actually risks;
  * is caught by the identity test as exactly two mismatched global ids.

Only the two 12-character integer fields are rewritten, so `diff` against the pristine
file shows one line pair and nothing else.

com_info<NNNNN>.out layout (verified against src/fesom_partit.cpp:154-207):
    rank
    then three com_struct blocks (nod2D, elem2D, elem2D_full), each:
        rPEnum, rPE[rPEnum], rptr[rPEnum+1], rlist[rlist_size],
        sPEnum, sPE[sPEnum], sptr[sPEnum+1], slist[sptr[-1]-sptr[0]]
    rlist_size comes from my_list<NNNNN>.out: eDim_nod2D for the nod2D block.

usage:
  m11_corrupt_com_info.py <dist_dir> [--rank N] [--mode swap-rlist] [--dry-run]
"""
import argparse
import re
import sys

TOKEN = re.compile(rb"\S+")


def tokens_with_spans(path):
    raw = open(path, "rb").read()
    spans = [(m.start(), m.end()) for m in TOKEN.finditer(raw)]
    vals = [int(raw[s:e]) for s, e in spans]
    return raw, spans, vals


def read_my_list_dims(dist_dir, rank):
    """-> (myDim_nod2D, eDim_nod2D)"""
    path = f"{dist_dir}/my_list{rank:05d}.out"
    _, _, v = tokens_with_spans(path)
    if v[0] != rank:
        sys.exit(f"{path}: says rank={v[0]}, expected {rank}")
    return v[1], v[2]


def locate_nod2d_rlist(vals, edim):
    """-> (first_index, count) of the nod2D rlist inside the com_info token stream."""
    i = 1                      # [0] = rank
    r_pe_num = vals[i]; i += 1
    i += r_pe_num              # rPE
    rptr = vals[i:i + r_pe_num + 1]
    i += r_pe_num + 1
    n = rptr[-1] - rptr[0]
    if n != edim:
        sys.exit(f"rlist length from rptr ({n}) != eDim_nod2D from my_list ({edim}) "
                 "— com_info and my_list disagree, refusing to touch the file")
    return i, n, rptr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dist_dir")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--mode", default="swap-rlist", choices=["swap-rlist"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    path = f"{a.dist_dir}/com_info{a.rank:05d}.out"
    mydim, edim = read_my_list_dims(a.dist_dir, a.rank)
    raw, spans, vals = tokens_with_spans(path)
    if vals[0] != a.rank:
        sys.exit(f"{path}: says rank={vals[0]}, expected {a.rank}")

    base, n, rptr = locate_nod2d_rlist(vals, edim)
    print(f"{path}: myDim_nod2D={mydim} eDim_nod2D={edim} "
          f"rPEnum={vals[1]} rptr={rptr}")

    # Pick two entries served by DIFFERENT sender PEs when the rank has >1
    # neighbour, so the corruption crosses a message boundary (the realistic case).
    if len(rptr) >= 3:
        i0 = 0
        i1 = rptr[1] - rptr[0]
    else:
        if n < 2:
            sys.exit("halo too small to corrupt (eDim < 2)")
        i0, i1 = 0, n - 1
    if i1 >= n:
        sys.exit(f"computed rlist offset {i1} out of range (n={n})")

    t0, t1 = base + i0, base + i1
    v0, v1 = vals[t0], vals[t1]
    print(f"swap nod2D rlist[{i0}]={v0} <-> rlist[{i1}]={v1} "
          f"(local halo slots, myDim+1..myDim+eDim = {mydim+1}..{mydim+edim})")
    if v0 == v1:
        sys.exit("the two entries are equal — swapping would be a no-op")

    if a.dry_run:
        print("dry-run: file untouched")
        return

    def field(idx, value):
        s, e = spans[idx]
        return raw[s:e], f"{value:>{e - s}d}".encode()

    old0, new0 = field(t0, v1)
    old1, new1 = field(t1, v0)
    if len(new0) != len(old0) or len(new1) != len(old1):
        sys.exit("replacement width mismatch — refusing to reflow the file")

    out = bytearray(raw)
    out[spans[t0][0]:spans[t0][1]] = new0
    out[spans[t1][0]:spans[t1][1]] = new1
    open(path, "wb").write(bytes(out))
    print(f"CORRUPTED {path} (2 fields rewritten, file size unchanged: "
          f"{len(raw)} -> {len(out)} bytes)")
    print("expected model behaviour: fesom_halo_identity_test aborts with 2 mismatches")


if __name__ == "__main__":
    main()
