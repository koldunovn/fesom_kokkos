# `partm11` — the M11 patched partitioner

The partitioner trees themselves live off-repo (they are 220 MB apiece and contain a vendored
METIS); this directory holds the **patches**, so the builds are reproducible from the upstream
tree alone.

## Trees

| tree | METIS | source | role |
|---|---|---|---|
| `/work/ab0995/a270088/port2/partm11/fesom2_ref` | bundled 5.1.0 | pristine `~/fesom_part/fesom2` | the reference for null-1 |
| `/work/ab0995/a270088/port2/partm11/fesom2` | bundled 5.1.0 | + `fort_part.c.m11.patch` | **partm11-a**, carries both nulls |
| `/work/ab0995/a270088/port2/partm11/fesom2_b` | **5.2.1** (prebuilt) | + both patches | **partm11-b**, arm A0 |
| `/work/ab0995/a270088/port2/partm11/inst` | — | GKlib + METIS v5.2.1 install prefix | |

Build (all three): source `/sw/etc/profile.levante` then `<tree>/env/levante.dkrz.de/shell`
(Intel oneAPI 2022.0.1 + OpenMPI 4.1.2), then `bash mesh_part/configure.sh`. For `fesom2_b`,
configure manually with `-DMETIS_PREBUILT_ROOT=<partm11>/inst` instead.

METIS 5.2.1 + GKlib (`git`, tags below), both built into `inst`:

```
cd lib/GKlib      && CFLAGS=-fPIC make config prefix=$INST cc=$(which mpicc) && make -j8 install
cd lib/metis-5.2.1&& CFLAGS=-fPIC make config prefix=$INST gklib_path=$INST cc=$(which mpicc) \
                  && make -j8 install
```

Pinned upstream commits: METIS `f5ae915` (tag `v5.2.1`), GKlib `3b7d61b`.

## Three traps this build hit, all silent

1. **`i64=0` gives you 64-bit indices.** METIS 5.2.1's Makefile tests
   `ifneq ($(i64),not-set)`, so *any* value — including `0` — selects `IDXTYPEWIDTH 64`. The
   Fortran side passes int32 arrays, so that would corrupt everything. **Omit `i64` and `r64`
   entirely**, then check `head -2 $INST/include/metis.h` says 32/32.
2. **METIS 5.2.1's Makefile silently drops `cflags=`.** It forwards `prefix`, `gklib_path`,
   `cc`, `shared`, `i64`, … and nothing else. Its `-fPIC` is added only on the
   `CMAKE_COMPILER_IS_GNUCC` branch of `conf/gkbuild.cmake`, so an Intel build has no `-fPIC`
   and linking it into `libfesom_meshpart_C.so` fails with a relocation error. Pass `CFLAGS`
   through the environment.
3. **GKlib installs into `lib64`, METIS looks in `lib`.** `ln -sfn lib64 $INST/lib`.

## Provenance: the md5 of the executable is NOT enough

The C partitioning code compiles into **`libfesom_meshpart_C.so`**, and `fesom_meshpart` links
against it. Editing `fort_part.c` therefore leaves the executable's md5 *unchanged*. Record
both — `scripts/m11_partgen.sh` prints both on every run. (M10 lost two runs to a stale
partitioner binary; this is the mechanism that makes that easy.)

Related: the executable finds the library through an rpath of `$ORIGIN/../lib64`, resolved
from the *real* path of the running image. **Copying the executable into a work directory
breaks it** ("cannot open shared object file", job 26851114). Run it in place; only the
namelists need to be in the working directory.

## Knobs

See the header comment of `fort_part.c.m11.patch`. All are optional, all announce themselves,
all abort on a value they do not recognise (L80), and with none of them set the binary
reproduces the unpatched output byte-for-byte (null-1).
