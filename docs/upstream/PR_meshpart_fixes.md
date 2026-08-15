# PR A — meshpart: two robustness fixes

**Branch:** `fix_meshpart_reshape_stack_overflow` (exists locally in
`/work/ab0995/a270088/port2/partm11/fesom2`, 2 commits `9f35654b` + `ed3c53b9` on main from
2026-08; not yet pushed to the fork).

**Title:** `meshpart: fix large-mesh segfault; never overwrite existing level/edge files`

---

The mesh partitioner (`fesom_meshpart`) has two long-standing issues that surface on large
meshes, found while generating partitions for GPU scaling work.

**1. Segfault on large meshes (stack overflow in `read_mesh_ini`).**
`read_mesh_ini` fills `elem2D_nodes` with `reshape()`. With ifort the array temporary that
`reshape()` creates is placed on the stack; for a ~30 M-element mesh the temporary is
`3*elem2D*4` bytes ≈ 177 MB, which overflows any normal stack limit and segfaults inside
`read_mesh_ini`. Small meshes (e.g. CORE2) fit, which is why this survived since 2019. The fix
replaces the two `reshape()` calls with element-wise copy loops that use no temporary, so the
tool works for any mesh size regardless of stack limit or `-heap-arrays`.

**2. The partitioner silently overwrites mesh level/edge files.**
`fvom_init` unconditionally rewrote `nlvls.out`, `elvls.out`, `elvls_raw.out`, `edgenum.out`,
`edges.out`, `edge_tri.out` (and the cavity variants) in the mesh directory on every
partitioner run. If the level-finding algorithm has changed since the files were first
created, a *partitioning* run silently redefines the vertical level structure under a running
experiment: restarts written earlier are then read with zero-thickness layers at nodes that
got deeper, and the model dies with a division by zero in `pressure_bv` at the first timestep
(this really happened: a partitioner run on a pool mesh invalidated a full 63-year JRA55
spinup's restarts). The fix: each of these files is written only if it does not exist yet,
with a warning otherwise; the partitioning itself still uses the freshly computed in-memory
edges/levels, and deleting the files by hand forces regeneration.

Both changes affect only the standalone partitioner (`mesh_part/`), not the model.
