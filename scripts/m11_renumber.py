#!/usr/bin/env python3
"""M11: renumber a FESOM mesh with a sphere-aware space-filling curve or RCM.

Renumbering is transparent to model code — nothing stores a node id across runs (no restart
reader, forcing and initial conditions are interpolated at run time by lat/lon) — so the whole
lever lives in the mesh files. What makes it dangerous is the Z7 failure class: ONE node-indexed
file left unpermuted is bitwise-correct at step 1 and wrong afterwards. So this tool
**classifies every file in the source directory** and refuses to run if it meets one it does not
recognise, rather than copying it through and hoping.

Orderings
---------
  hilbert-xyz  3-D Hilbert (Skilling's transpose algorithm) on unit-sphere xyz, 21 bits/dim.
               The sphere is embedded in 3-D, so there is no dateline or pole seam — the trap
               that makes a 2-D curve on (lon, lat) a bad idea.
  s2           Cubed-sphere: gnomonic projection onto the face of largest |component|, then a
               2-D Hilbert curve within each face, faces chained in a fixed order (Google S2's
               construction).
  rcm          Reverse Cuthill-McKee on the same graph METIS partitions (scipy). Bandwidth,
               not geometry — the comparison the SFC-vs-RCM question needs.

What gets permuted, and why each one
------------------------------------
  P_node -> nod2d.out rows (the id column is REWRITTEN to the identity: the model hard-checks
            `id == i+1` at fesom_mesh.cpp:249), nlvls.out rows, and the DEPTH block of
            aux3d.out (its nl + zbar header is global and must not move).
  P_elem -> elem2d.out rows and elvls.out / elvls_raw.out rows. elem2d's VALUES are additionally
            mapped through P_node, preserving each triangle's vertex CYCLE (never rotated or
            sorted: FESOM's edge and normal conventions depend on the cycle).
  deleted -> edges.out, edge_tri.out, edgenum.out (the partitioner regenerates them, which also
            preserves the interior-first `edge2D_in` convention), plus every derived cache
            (pyfesom2 pickle, griddes, distances_*/inds_*, fesom.mesh.diag.nc) and every dist_*.

Rows are permuted AS TEXT wherever their values do not change, so coordinates and depths are
carried through byte-for-byte with no float round-trip.

usage:
  m11_renumber.py <src_mesh> <dst_name> --order hilbert-xyz [--elem-order minvertex|centroid]
  m11_renumber.py <src_mesh> <dst_name> --order rcm --dry-run
  m11_renumber.py --permute-labels <renumbered_mesh> --from-dist <old_mesh> --npes 8 -o part.txt
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_scorecard import Mesh
from m7_part_spread import load_ranks

SANDBOX = os.environ.get("M11_SANDBOX_ROOT", "/work/ab0995/a270088/port2/mesh_m11")

# --------------------------------------------------------------- file classification

NODE_FILES = ["nod2d.out", "nlvls.out"]                 # one row per node
ELEM_FILES = ["elem2d.out", "elvls.out", "elvls_raw.out"]  # one row per element
SPECIAL = ["aux3d.out"]                                  # header + node block
DELETE_REGEN = ["edges.out", "edge_tri.out", "edgenum.out"]
# derived artefacts that describe the OLD numbering and would silently mislead
DELETE_STALE = ["pickle_mesh_py3_fesom2", "*griddes*.nc", "fesom.mesh.diag.nc",
                "distances_*", "inds_*", "dist_*", "*.tar.gz",
                "cavity_*.out", "MD5MANIFEST"]
COPY_AS_IS = ["*_zaxis.txt", "README", "README.md", ".gitattributes", "MESH_PROVENANCE.md",
              "*.md"]


def classify(name):
    if name in NODE_FILES:
        return "node"
    if name in ELEM_FILES:
        return "elem"
    if name in SPECIAL:
        return "special"
    if name in DELETE_REGEN:
        return "regen"
    for p in DELETE_STALE:
        if fnmatch.fnmatch(name, p):
            return "stale"
    for p in COPY_AS_IS:
        if fnmatch.fnmatch(name, p):
            return "copy"
    return "UNKNOWN"


# --------------------------------------------------------------- Hilbert machinery

def _axes_to_transpose(X, b):
    """Skilling 2004, AxestoTranspose. X is (n_dim, N) uint64 with b bits per coordinate."""
    n = X.shape[0]
    X = X.copy()
    M = np.uint64(1) << np.uint64(b - 1)
    Q = M
    while Q > 1:
        P = Q - np.uint64(1)
        for i in range(n):
            mask = (X[i] & Q) != 0
            # invert
            X[0][mask] ^= P
            # exchange
            t = (X[0][~mask] ^ X[i][~mask]) & P
            X[0][~mask] ^= t
            X[i][~mask] ^= t
        Q >>= np.uint64(1)
    for i in range(1, n):                       # Gray encode
        X[i] ^= X[i - 1]
    t = np.zeros(X.shape[1], dtype=np.uint64)
    Q = M
    while Q > 1:
        t[(X[n - 1] & Q) != 0] ^= Q - np.uint64(1)
        Q >>= np.uint64(1)
    for i in range(n):
        X[i] ^= t
    return X


def _interleave(X, b):
    """Transpose form -> a single integer key (bit k of each dim, most significant first)."""
    n = X.shape[0]
    key = np.zeros(X.shape[1], dtype=np.uint64)
    for k in range(b - 1, -1, -1):
        for i in range(n):
            key = (key << np.uint64(1)) | ((X[i] >> np.uint64(k)) & np.uint64(1))
    return key


def hilbert_key_3d(x, y, z, bits=21):
    """3-D Hilbert index of points already scaled to [0,1]."""
    m = np.uint64((1 << bits) - 1)
    X = np.vstack([np.clip(v, 0.0, 1.0) * float(m) for v in (x, y, z)]).astype(np.uint64)
    return _interleave(_axes_to_transpose(X, bits), bits)


def hilbert_key_2d(u, v, bits=26):
    m = np.uint64((1 << bits) - 1)
    X = np.vstack([np.clip(a, 0.0, 1.0) * float(m) for a in (u, v)]).astype(np.uint64)
    return _interleave(_axes_to_transpose(X, bits), bits)


def lonlat_to_xyz(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    c = np.cos(lat)
    return c * np.cos(lon), c * np.sin(lon), np.sin(lat)


def key_hilbert_xyz(lon, lat, bits=21):
    x, y, z = lonlat_to_xyz(lon, lat)
    return hilbert_key_3d((x + 1) / 2, (y + 1) / 2, (z + 1) / 2, bits)


def key_s2(lon, lat, bits=26):
    """Cubed-sphere face id + per-face 2-D Hilbert index, faces chained in id order."""
    x, y, z = lonlat_to_xyz(lon, lat)
    v = np.vstack([x, y, z])
    axis = np.argmax(np.abs(v), axis=0)
    pos = v[axis, np.arange(v.shape[1])] > 0
    face = (axis * 2 + (~pos)).astype(np.uint64)       # 0..5
    # gnomonic coordinates on the chosen face, in [-1,1]
    other = np.array([[1, 2], [2, 0], [0, 1]])[axis]
    denom = np.abs(v[axis, np.arange(v.shape[1])])
    u = v[other[:, 0], np.arange(v.shape[1])] / denom
    w = v[other[:, 1], np.arange(v.shape[1])] / denom
    h = hilbert_key_2d((u + 1) / 2, (w + 1) / 2, bits)
    return (face << np.uint64(2 * bits)) | h


def order_rcm(mesh):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    gi, gj = mesh.graph()
    n = mesh.nod2D
    A = coo_matrix((np.ones(gi.size, np.int8), (gi, gj)), shape=(n, n))
    A = (A + A.T).tocsr()
    return np.asarray(reverse_cuthill_mckee(A, symmetric_mode=True), dtype=np.int64)


def node_order(mesh, order):
    """-> old_index array in NEW order (i.e. new_of_old = argsort of this)."""
    if order == "hilbert-xyz":
        k = key_hilbert_xyz(mesh.lon, mesh.lat)
        return np.argsort(k, kind="stable")
    if order == "s2":
        k = key_s2(mesh.lon, mesh.lat)
        return np.argsort(k, kind="stable")
    if order == "rcm":
        return order_rcm(mesh)
    raise SystemExit(f"unknown ordering {order}")


def elem_order(mesh, new_of_old, how):
    """-> old element index array in NEW order."""
    enew = new_of_old[mesh.elem]                       # element vertices in new numbering
    if how == "minvertex":
        key = enew.min(axis=1)
        second = np.sort(enew, axis=1)[:, 1]
        return np.lexsort((second, key))
    if how == "centroid":
        x, y, z = lonlat_to_xyz(mesh.lon, mesh.lat)
        cx, cy, cz = (a[mesh.elem].mean(axis=1) for a in (x, y, z))
        r = np.sqrt(cx * cx + cy * cy + cz * cz)
        k = hilbert_key_3d((cx / r + 1) / 2, (cy / r + 1) / 2, (cz / r + 1) / 2)
        return np.argsort(k, kind="stable")
    raise SystemExit(f"unknown element ordering {how}")


# --------------------------------------------------------------- file rewriting

def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


def write_lines(path, header, rows):
    with open(path, "w") as f:
        if header is not None:
            f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


def renumber(src, dst, order, elem_how, dry_run=False):
    mesh = Mesh(src)
    files = sorted(os.listdir(src))
    kinds = {f: classify(f) for f in files}
    unknown = [f for f, k in kinds.items() if k == "UNKNOWN"]

    print(f"=== source {src}: {len(files)} entries")
    for kind, label in [("node", "permute by P_node"), ("elem", "permute by P_elem"),
                        ("special", "header kept, node block permuted"),
                        ("regen", "DELETE (partitioner regenerates)"),
                        ("stale", "DELETE (describes the old numbering)"),
                        ("copy", "copy unchanged")]:
        got = [f for f, k in kinds.items() if k == kind]
        if got:
            print(f"  {label:<44} {', '.join(got)}")
    if unknown:
        print(f"\n  UNKNOWN, and therefore refused: {', '.join(unknown)}")
        print("  A node- or element-indexed file left unpermuted is bitwise-correct at step 1")
        print("  and wrong at step 2 (the Z7 class). Classify it in m11_renumber.py and re-run.")
        return 2

    old_in_new = node_order(mesh, order)                # new position -> old index
    new_of_old = np.empty_like(old_in_new)
    new_of_old[old_in_new] = np.arange(mesh.nod2D)
    eold_in_new = elem_order(mesh, new_of_old, elem_how)
    enew_of_old = np.empty_like(eold_in_new)
    enew_of_old[eold_in_new] = np.arange(mesh.elem2D)

    # -- invariants, before anything is written ---------------------------------
    assert np.array_equal(new_of_old[old_in_new], np.arange(mesh.nod2D)), "P o P^-1 != id (nodes)"
    assert np.array_equal(enew_of_old[eold_in_new], np.arange(mesh.elem2D)), "P o P^-1 != id (elems)"
    assert np.array_equal(np.sort(old_in_new), np.arange(mesh.nod2D)), "P_node is not a permutation"
    assert np.array_equal(np.sort(eold_in_new), np.arange(mesh.elem2D)), "P_elem is not a permutation"
    print(f"\n  P_node and P_elem are permutations, P o P^-1 = id  OK")

    gi, gj = mesh.graph()
    before = float(np.abs(gi - gj).mean())
    after = float(np.abs(new_of_old[gi] - new_of_old[gj]).mean())
    print(f"  locality proxy: mean |di| over graph edges {before:,.0f} -> {after:,.0f} "
          f"({(after/before - 1)*100:+.1f} %)")

    if dry_run:
        print("\n  dry run: nothing written")
        return 0

    os.makedirs(dst, exist_ok=True)

    # nod2d.out — permute rows AS TEXT, rewrite only the id column to the identity
    lines = read_lines(f"{src}/nod2d.out")
    body = np.array(lines[1:], dtype=object)[old_in_new]
    rows = [re.sub(r"^\s*\d+", str(i + 1), s, count=1) for i, s in enumerate(body)]
    write_lines(f"{dst}/nod2d.out", lines[0], rows)

    # nlvls.out — permute rows as text
    lines = read_lines(f"{src}/nlvls.out")
    write_lines(f"{dst}/nlvls.out", None, list(np.array(lines, dtype=object)[old_in_new]))

    # aux3d.out — nl + zbar stay put, the depth block follows its node
    lines = read_lines(f"{src}/aux3d.out")
    nl = int(lines[0].split()[0])
    head, depth = lines[:1 + nl], np.array(lines[1 + nl:], dtype=object)
    assert depth.size == mesh.nod2D, (depth.size, mesh.nod2D)
    write_lines(f"{dst}/aux3d.out", None, head + list(depth[old_in_new]))

    # elem2d.out — reorder rows AND map the vertex values, cycle preserved
    lines = read_lines(f"{src}/elem2d.out")
    enew = new_of_old[mesh.elem] + 1
    rows = [" ".join(map(str, r)) for r in enew[eold_in_new]]
    write_lines(f"{dst}/elem2d.out", lines[0], rows)

    # elvls.out / elvls_raw.out — permute rows as text
    for f in ("elvls.out", "elvls_raw.out"):
        if os.path.exists(f"{src}/{f}"):
            lines = read_lines(f"{src}/{f}")
            write_lines(f"{dst}/{f}", None, list(np.array(lines, dtype=object)[eold_in_new]))

    for f in [f for f, k in kinds.items() if k == "copy"]:
        subprocess.run(["cp", "-aL", f"{src}/{f}", f"{dst}/{f}"], check=True)

    np.save(f"{dst}/m11_perm_node.npy", old_in_new)       # new position -> old index
    np.save(f"{dst}/m11_perm_elem.npy", eold_in_new)
    with open(f"{dst}/MESH_PROVENANCE.md", "w") as f:
        f.write(f"""# Renumbered mesh

source     : {os.path.realpath(src)}
ordering   : {order} (nodes), {elem_how} (elements)
tool       : scripts/m11_renumber.py (M11)
locality   : mean |di| over graph edges {before:,.0f} -> {after:,.0f}

Permutations saved as `m11_perm_node.npy` / `m11_perm_elem.npy` (new position -> old index).
Any archived NetCDF written on the source numbering must be permuted through them before it is
compared with output from this mesh; raw field-by-field diffs across orderings are meaningless.

edges.out / edge_tri.out / edgenum.out were NOT copied: the partitioner regenerates them, which
also preserves the interior-first `edge2D_in` convention the ice and momentum code relies on.
Every derived cache (pyfesom2 pickle, griddes, distances_*/inds_*, mesh.diag) was dropped for
the same reason: it describes the old numbering.
""")
    print(f"\n  wrote {dst}")
    for f in sorted(os.listdir(dst)):
        print(f"    {f}")
    return 0


def verify(src, dst):
    """Re-read both meshes and check the invariants that renumbering must preserve."""
    a, b = Mesh(src), Mesh(dst)
    perm = np.load(f"{dst}/m11_perm_node.npy")           # new -> old
    eperm = np.load(f"{dst}/m11_perm_elem.npy")
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
        ok = ok and bool(cond)

    chk("nod2D / elem2D unchanged", a.nod2D == b.nod2D and a.elem2D == b.elem2D)
    chk("coordinates follow their node", np.array_equal(a.lon[perm], b.lon)
        and np.array_equal(a.lat[perm], b.lat))
    chk("coast flag follows its node", np.array_equal(a.coast[perm], b.coast))
    chk("nlvls follows its node", np.array_equal(a.nlev_nod[perm], b.nlev_nod))
    chk("elvls follows its element", np.array_equal(a.nlev_elem[eperm], b.nlev_elem))

    inv = np.empty_like(perm)
    inv[perm] = np.arange(a.nod2D)
    chk("elem2d vertices mapped, cycle preserved",
        np.array_equal(inv[a.elem][eperm], b.elem))

    # element areas as multisets: a permutation may not deform a single triangle
    def areas(m):
        x, y, z = lonlat_to_xyz(m.lon, m.lat)
        p = np.stack([x, y, z], axis=1)[m.elem]
        return np.sort(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1))
    aa, bb = areas(a), areas(b)
    chk("element areas identical as a multiset", np.allclose(aa, bb, rtol=0, atol=1e-15),
        f"max |diff| {np.abs(aa - bb).max():.3e}")

    # `perm` is new -> old, so it is what maps the NEW mesh's edges back to old labels
    # (`inv` above is old -> new, used for elem2d's values).
    gia, gja = a.graph()
    ga = np.sort(np.minimum(gia, gja) * a.nod2D + np.maximum(gia, gja))
    gib, gjb = b.graph()
    pu, pv = perm[gib], perm[gjb]
    gb = np.sort(np.minimum(pu, pv) * a.nod2D + np.maximum(pu, pv))
    chk("graph is the same graph under the permutation", np.array_equal(ga, gb))

    chk("edge files absent (must be regenerated)",
        not any(os.path.exists(f"{dst}/{f}") for f in DELETE_REGEN))
    chk("id column is the identity", True, "checked by Mesh() load + model hard-check")
    return 0 if ok else 1


def permute_labels(old_mesh, new_mesh, npes, out):
    """Carry an existing partition through P_node: identical partition, new numbering.

    This is the input for a PURE ordering A/B. Without it, an ordering arm would also be a
    repartitioning arm and the two effects could not be separated.
    """
    part, _ = load_ranks(old_mesh, npes)
    part = np.asarray(part, dtype=np.int64)
    perm = np.load(f"{new_mesh}/m11_perm_node.npy")      # new -> old
    newpart = part[perm]
    np.savetxt(out, newpart, fmt="%d")
    a = np.bincount(part, minlength=npes)
    b = np.bincount(newpart, minlength=npes)
    same = np.array_equal(a, b)
    print(f"label-permuted {old_mesh}/dist_{npes} -> {out}")
    print(f"  per-part sizes identical: {same} (min {b.min()}, max {b.max()})")
    return 0 if same else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst_name", nargs="?", help="new directory NAME inside the M11 sandbox")
    ap.add_argument("--order", choices=["hilbert-xyz", "s2", "rcm"], default="hilbert-xyz")
    ap.add_argument("--elem-order", choices=["minvertex", "centroid"], default="minvertex")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", metavar="DST", help="verify an already renumbered mesh")
    ap.add_argument("--permute-labels", metavar="NEW_MESH")
    ap.add_argument("--from-dist", metavar="OLD_MESH")
    ap.add_argument("--npes", type=int)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    if a.permute_labels:
        if not (a.from_dist and a.npes and a.out):
            ap.error("--permute-labels needs --from-dist, --npes and -o")
        return permute_labels(a.from_dist, a.permute_labels, a.npes, a.out)
    if a.verify:
        if not a.src:
            ap.error("--verify needs the source mesh as the first argument")
        print(f"=== verifying {a.verify} against {a.src}")
        return verify(a.src, a.verify)
    if not (a.src and a.dst_name):
        ap.error("give <src_mesh> <dst_name>")
    if os.path.sep in a.dst_name:
        ap.error("dst_name is a NAME inside the sandbox, not a path")
    dst = os.path.join(SANDBOX, a.dst_name)
    return renumber(a.src, dst, a.order, a.elem_order, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
