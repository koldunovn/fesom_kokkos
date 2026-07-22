# env_dolpung.sh — build/run environment for the DKRZ dolpung GH200 partition
# (42 nodes, 4x Grace-Hopper each: 72-core Grace ARM + H100-class GPU, CUDA 12.9).
#
# Nodes are aarch64, logins are x86 -> COMPILE ON A DOLPUNG NODE
# (salloc -t 360 -A mh1571 -p dolpung -n 288), never on the login.
#
# NO `module load` on purpose: (1) the ARM tree's module names carry spack
# hashes and differ from the x86 names, (2) a login shell (`bash -l`) auto-loads
# x86 modules from the user profile which just error out on ARM. Absolute paths
# are deterministic in both salloc and sbatch. Matched set (each built against
# the others — the MPI's own configure line is the source of these paths):
#   gcc 14.2.0            (spack am53qrz)                      host C++
#   nvhpc 25.7 CUDA 12.9  (spack dsupkck)                      nvcc, device
#   openmpi 4.1.8_cuda-12.9_nvhpc-25.7_gcc-14                  CUDA-aware (UCX 1.19 cuda)
#   netcdf-c 4.9.3-6xe2mx2 (+ hdf5 1.14.3 via rpath)           linked vs THAT openmpi
_SWARM=/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2
export DOLPUNG_GCC=$_SWARM/gcc-14.2.0-am53qrz
export DOLPUNG_CUDA=$_SWARM/nvhpc-25.7-dsupkck/Linux_aarch64/25.7/cuda/12.9
export DOLPUNG_MPI=/sw/mpi/linux-rhel9-neoverse_v2/openmpi/4.1.8_cuda-12.9_nvhpc-25.7_gcc-14
export DOLPUNG_NETCDF=$_SWARM/netcdf-c-4.9.3-6xe2mx2
# Strip x86 spack entries leaked from the login environment (srun exports the
# submitting shell's env; the x86 tree is /sw/spack-levante/<pkg>, the ARM tree
# /sw/spack-levante-0.23.1/... — an x86 `git` first in PATH broke the Kokkos
# configure with "cannot execute binary file").
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | paste -sd:)
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | paste -sd:)
_ARM_GIT=$_SWARM/git-2.47.0-6tacdsl/bin
export PATH=$DOLPUNG_MPI/bin:$DOLPUNG_NETCDF/bin:$DOLPUNG_CUDA/bin:$DOLPUNG_GCC/bin:$_ARM_GIT:$PATH
export LD_LIBRARY_PATH=$DOLPUNG_GCC/lib64:$DOLPUNG_MPI/lib:$DOLPUNG_NETCDF/lib:$DOLPUNG_CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_WRAPPER_DEFAULT_COMPILER=$DOLPUNG_GCC/bin/g++
# x86 login-env leakage: srun exports the submitting shell's environment, and the
# login profile's x86 modules set CMAKE_PREFIX_PATH to the x86 netcdf — cmake
# prefers that over PATH and finds the wrong-arch library (bit us on build #1).
export CMAKE_PREFIX_PATH=$DOLPUNG_NETCDF:$DOLPUNG_MPI
unset HDF5_PLUGIN_PATH PKG_CONFIG_PATH
