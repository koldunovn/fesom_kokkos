# Building (Kokkos backends) on Levante

The C++/Kokkos port builds for three backends from one source. Kokkos is vendored as a
git submodule at `externals/kokkos` (**4.4.01**, matching ICON) and built in-tree via
`add_subdirectory`. There is **no Kokkos module on Levante**; a spack package exists as an
alternative (`spack install kokkos +cuda cuda_arch=80 +openmp` / `+rocm amdgpu_target=gfx90a`).

First checkout must init the submodule:
```bash
git submodule update --init --recursive
```

## Module environments

| Backend | modules |
|---|---|
| Serial / OpenMP | `gcc/11.2.0-gcc-11.2.0` |
| CUDA (A100) | `gcc/11.2.0-gcc-11.2.0` + `nvhpc/24.7-gcc-11.2.0` (nvcc 12.5), `NVCC_WRAPPER_DEFAULT_COMPILER=g++` |
| MPI + netCDF (full model, M0.3+) | + `openmpi/4.1.2-gcc-11.2.0` `netcdf-c/4.8.1-gcc-11.2.0` (see `env.sh`) |

`cmake` is the system `/usr/bin/cmake` (3.26.5) — survives `module purge`.

## Kokkos backend smoke (`kokkos_smoke/`, validated 2026-05-25)

Exercises `parallel_for` + `parallel_reduce` + a `DualView` host↔device round-trip.

```bash
# --- Serial (login node) --- SMOKE PASS
module --force purge && module load gcc/11.2.0-gcc-11.2.0
cmake -S kokkos_smoke -B build-smoke-serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-smoke-serial -j 16 && ./build-smoke-serial/kokkos_smoke

# --- OpenMP (login node) --- SMOKE PASS
cmake -S kokkos_smoke -B build-smoke-omp -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON
cmake --build build-smoke-omp -j 16 && OMP_NUM_THREADS=4 OMP_PROC_BIND=spread ./build-smoke-omp/kokkos_smoke

# --- CUDA (compile anywhere; RUN on a GPU node) ---
module --force purge && module load gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake -S kokkos_smoke -B build-smoke-cuda -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
      -DCMAKE_CXX_COMPILER=$PWD/externals/kokkos/bin/nvcc_wrapper
cmake --build build-smoke-cuda -j 16
# run on a GPU (the login node has no GPU):
srun -p gpu-devel -A ab0995 --gres=gpu:1 --time=00:05:00 ./build-smoke-cuda/kokkos_smoke
# or: sbatch jobs/job_kokkos_smoke_gpu
```

## LUMI (AMD/HIP) — later (M6)

Same submodule; reconfigure with `-DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON` and ROCm's
`hipcc` as `CMAKE_CXX_COMPILER`. No source changes (the no-vendor-lock contract).

## Login-node MPI caveat

UCX/IB transports are unavailable off the compute nodes. For a single-rank model smoke on a
login node, override the `env.sh` OpenMPI knobs — see `docs/reference/PROVENANCE.md`.
