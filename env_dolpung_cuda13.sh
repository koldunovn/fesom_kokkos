# env_dolpung_cuda13.sh — CUDA-13.x toolkit variants for the dolpung GH200 partition.
#
# WHY: the fleet builds with CUDA 12.9 (nvhpc 25.7) because that is what the
# CUDA-aware OpenMPI on /sw/mpi was built against. But nvcc compiles the code
# that dominates our runtime, and the ARM tree also ships CUDA 13.0 (nvhpc 25.9,
# WITH a matched openmpi) and 13.2 (nvhpc 26.5, no matched MPI). Newer nvcc =
# newer Hopper codegen; this file makes that a one-variable A/B.
#
# Select with CUDAVAR=130 (default) or 132:
#   130 = fully coherent stack: nvcc 13.0 + openmpi 4.1.8_cuda-13.0_nvhpc-25.9
#         + netcdf-c-4.9.3-zjiux4i (verified linked against THAT libmpi).
#   132 = nvcc 13.2 (nvhpc 26.5) on the same 13.0 MPI/netCDF — a DEVICE-CODEGEN
#         probe. Safe here because FESOM_HALO_STAGE=1 means MPI only ever sees
#         pinned HOST pointers, so the MPI's own CUDA coupling is not exercised.
#         (Do not use the 132 build with device-pointer MPI.)
_SWARM=/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2
CUDAVAR=${CUDAVAR:-130}
case "$CUDAVAR" in
  130) export DOLPUNG_CUDA=$_SWARM/nvhpc-25.9-t4lua7j/Linux_aarch64/25.9/cuda/13.0 ;;
  # ⚠️ nvhpc-26.5 exposes its tree under BOTH Linux_aarch64/26.5 and .../2026 —
  # a glob matches both and silently breaks PATH ("nvcc: command not found").
  # Always name the version directory explicitly.
  132) export DOLPUNG_CUDA=$_SWARM/nvhpc-26.5-soicsjn/Linux_aarch64/26.5/cuda/13.2 ;;
  *)   echo "env_dolpung_cuda13.sh: CUDAVAR must be 130 or 132 (got $CUDAVAR)" >&2; return 1 ;;
esac
[ -x "$DOLPUNG_CUDA/bin/nvcc" ] || { echo "env_dolpung_cuda13.sh: no nvcc at $DOLPUNG_CUDA/bin" >&2; return 1; }
export DOLPUNG_GCC=$_SWARM/gcc-14.2.0-am53qrz
export DOLPUNG_MPI=/sw/mpi/linux-rhel9-neoverse_v2/openmpi/4.1.8_cuda-13.0_nvhpc-25.9_gcc-14
export DOLPUNG_NETCDF=$_SWARM/netcdf-c-4.9.3-zjiux4i
# Same x86-login-env hygiene as env_dolpung.sh (srun exports the submitting shell).
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | paste -sd:)
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | paste -sd:)
_ARM_GIT=$_SWARM/git-2.47.0-6tacdsl/bin
export PATH=$DOLPUNG_MPI/bin:$DOLPUNG_NETCDF/bin:$DOLPUNG_CUDA/bin:$DOLPUNG_GCC/bin:$_ARM_GIT:$PATH
export LD_LIBRARY_PATH=$DOLPUNG_GCC/lib64:$DOLPUNG_MPI/lib:$DOLPUNG_NETCDF/lib:$DOLPUNG_CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_WRAPPER_DEFAULT_COMPILER=$DOLPUNG_GCC/bin/g++
# ⚠️ CUDA 13 DROPPED Volta (sm_70) — and Kokkos 4.4's nvcc_wrapper hardcodes
# default_arch="sm_70", appending it whenever no -arch is on the command line
# (exactly what CMake's compiler try-compile does) => "nvcc fatal: Unsupported
# gpu architecture 'sm_70'" before a single real source is touched. Use the
# sm_90-default copy as CMAKE_CXX_COMPILER; real TUs still get Kokkos' own
# HOPPER90 arch flag, so this only fixes the arch-less probe invocations.
export DOLPUNG_NVCC_WRAPPER=/home/a/a270088/port_kokkos/scripts/nvcc_wrapper_sm90
export CMAKE_PREFIX_PATH=$DOLPUNG_NETCDF:$DOLPUNG_MPI
unset HDF5_PLUGIN_PATH PKG_CONFIG_PATH
