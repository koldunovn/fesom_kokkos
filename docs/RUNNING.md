# Running fesom_kokkos outside Levante

This page shows how to build the C++/Kokkos port for CPUs or GPUs and run the CORE2
configuration used in the paper for one model year, using the input data archived on Zenodo.

## 1. Build

Requirements: a C++17 compiler, MPI, netCDF-C (with `nc-config` on the `PATH`) and CMake 3.18 or
newer. Kokkos 4.4.01 is a git submodule and is built together with the model:

```bash
git submodule update --init --recursive        # from a source archive: see externals/README.md
```

One build directory per back-end:

```bash
# CPU, Serial back-end (reproduces the C port byte for byte)
cmake -S . -B build-serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-serial -j

# NVIDIA GPU (needs a CUDA-aware MPI); use the architecture flag of your GPU,
# e.g. -DKokkos_ARCH_AMPERE80=ON for A100, -DKokkos_ARCH_HOPPER90=ON for H100/GH200
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER=$PWD/externals/kokkos/bin/nvcc_wrapper
cmake --build build-cuda -j

# AMD GPU (e.g. MI250X)
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_HIP=ON \
      -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_CXX_COMPILER=hipcc
cmake --build build-hip -j
```

Each build produces `fesom_port` in its build directory. The CUDA build always disables Kokkos'
stream-ordered memory pool (`CMakeLists.txt`), which corrupted GPU-aware MPI halo exchanges.
The paper's builds used GCC 11.2 with Open MPI 4.1.2 (Serial) and NVHPC 24.7 with the CUDA-aware
Open MPI 4.1.5 (CUDA) on Levante; `env.sh`, `env_cuda.sh`, `env_lumi.sh` and `env_mn5.sh` load the
module sets used on Levante, LUMI and MareNostrum 5.

## 2. Input data

The Kokkos port reads exactly the same files as the C port. Everything the CORE2 runs of the paper
read for 1958 is archived at [doi:10.5281/zenodo.21324318](https://doi.org/10.5281/zenodo.21324318)
(version 1.2, [doi:10.5281/zenodo.21770779](https://doi.org/10.5281/zenodo.21770779)), byte-identical
to the files the paper's runs read:

```bash
mkdir -p ~/fesom-data && cd ~/fesom-data
for f in core2_mesh_ic.zip core2_partitions.zip core2_forcing_1958.zip; do
    wget "https://zenodo.org/records/21770779/files/$f?download=1" -O "$f"
    unzip -q "$f"
done
cd core2_mesh_ic/mesh_core2_raw                # the model expects <mesh_dir>/dist_<ranks>
for d in ~/fesom-data/core2_partitions/dist_*; do ln -s "$d" .; done
```

The archives hold the CORE2 mesh in FESOM2 text format, decompositions for 2 to 864 ranks, the
PHC3.0 initial state, the salinity-restoring, runoff and chlorophyll files, and the eight JRA55-do
v1.4.0 fields for 1958. For later years add the JRA55-do files of those years to the forcing
directory, named `<var>.<year>.nc`; the paper used v1.4.0 for 1958 to 2018
([doi:10.22033/ESGF/input4MIPs.10842](https://doi.org/10.22033/ESGF/input4MIPs.10842)) and v1.5.0 for
2019 and 2020 ([doi:10.22033/ESGF/input4MIPs.15017](https://doi.org/10.22033/ESGF/input4MIPs.15017)).

## 3. Run one year of CORE2

```bash
D=~/fesom-data
export FESOM_ALE=zstar FESOM_MIX_SCHEME=TKE FESOM_WHICH_EVP=1 FESOM_IC_EXTRAP=det
export FESOM_FORCING_DIR=$D/core2_forcing_1958
export FESOM_SSS_PATH=$D/core2_mesh_ic/PHC2_salx.nc
export FESOM_RUNOFF_PATH=$D/core2_mesh_ic/CORE2_runoff.nc
export FESOM_CHL_FILE=$D/core2_mesh_ic/Sweeney_2005.nc
mkdir -p out
mpirun -np 4 build-cuda/fesom_port $D/core2_mesh_ic/mesh_core2_raw out 1800 17520 -1 \
       $D/core2_mesh_ic/phc3.0_winter.nc 1958
```

The positional arguments are `<mesh_dir> <output_dir> <dt_s> <nsteps> <snapshot_every> <phc_file>
<first_forcing_year>`; 17520 steps of 1800 s are the 365 days of 1958. Run one MPI rank per GPU.
The output is a set of monthly-mean files, `out/<variable>.fesom.1958.monthly.nc`.

The paper's CUDA hindcast ran on one node with four A100 GPUs (`dist_4`) and these speed settings,
all of which keep the Serial build byte-identical to the C port (they have no effect on CPU builds):

```bash
export FESOM_SPEED_SWSKIP=1 FESOM_SPEED_ICEFLUXDEV=1 FESOM_SPEED_NOFENCE2=1 FESOM_SPEED_IOACC=1 \
       FESOM_SPEED_FLAT=1 FESOM_SPEED_TDMANOINIT=1 FESOM_SPEED_ROTCACHE=1 FESOM_SPEED_FORCEDEV=1 \
       FESOM_SPEED_FLUXDEV=1 FESOM_SPEED_ICERAILS=1 FESOM_SPEED_SSHRAILS=1 FESOM_SPEED_LAZYSNAP=1 \
       FESOM_SPEED_BULKTAIL=1 FESOM_SPEED_SMOOTHSCRATCH=1 FESOM_SPEED_FERNOINIT=1 FESOM_SPEED_VISCNOINIT=1
export FESOM_SPEED_CG1R=0 FESOM_SPEED_CGPIPE=0 FESOM_SPEED_CGPOLY=0 FESOM_SPEED_ICELAG=0 \
       FESOM_SPEED_EVPWIDE=0 FESOM_SPEED_EVPWIDE_LEAN=0 FESOM_SPEED_EVPWIDE_FUSE=0
```

GPU runs are not bit-reproducible from one run to the next, because atomic additions and parallel
reductions change the order of floating-point sums. The Serial build on a fixed number of ranks is.

## 4. Checking the Serial build against the C port

Build the C port (`fesom_port`, version 1.0.0) and the Serial back-end of this port, run both with
the same configuration, rank count and restart schedule (`FESOM_RESTART_OUT`, `FESOM_RESTART_AT`),
and compare the restart files: every variable must agree exactly. The paper did this for one model
year on 128 ranks.

## 5. Restarts and other options

The restart variables (`FESOM_RESTART_OUT`, `FESOM_RESTART_AT`, `FESOM_RESTART_EVERY`,
`FESOM_RESTART_IN`) and the scheme switches (`FESOM_ALE`, `FESOM_MIX_SCHEME`, `FESOM_WHICH_EVP`,
`FESOM_IC_EXTRAP`) are the same as in the C port. Section "Environment knobs" of the README lists
the speed and diagnostic switches; `FESOM_FLAT` selects the flattened edge-scatter kernels of
pull request 2, which are off by default and were not used for the paper.
