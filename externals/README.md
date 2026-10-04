# externals

`kokkos/` is a git submodule pinned to Kokkos 4.4.01 (commit 15dc143e5), which the model builds
in-tree. In a git checkout, fetch it with

    git submodule update --init --recursive

A source archive (for example the one on Zenodo) does not contain it. Either unpack the
`kokkos-4.4.01` archive published next to the model source into `externals/kokkos`, or clone it:

    git clone --branch 4.4.01 --depth 1 https://github.com/kokkos/kokkos.git externals/kokkos

Kokkos is distributed under the Apache License 2.0 with LLVM exceptions.
