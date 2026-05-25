#!/bin/bash
# Kokkos-purity guard: the port must stay vendor-neutral (CUDA now, AMD/HIP later).
# Fail if any raw CUDA/HIP-ism appears in src/ — use only Kokkos abstractions
# (KOKKOS_LAMBDA/KOKKOS_FUNCTION, Kokkos::deep_copy, Kokkos:: math).
set -u
SRC_DIR="${1:-src}"
# Patterns that betray a vendor lock-in. (Word-ish boundaries to avoid false hits.)
PAT='cudaMalloc|cudaMemcpy|cudaFree|cudaSetDevice|<cuda_runtime|<cuda\.h>|__CUDA_ARCH__|__device__|__global__|__host__|hipMalloc|hipMemcpy|<hip/|__HIP_'
hits=$(grep -rInE "$PAT" "$SRC_DIR" 2>/dev/null \
        | grep -vE '^\s*\*|//|/\*')   # ignore comments
if [ -n "$hits" ]; then
  echo "KOKKOS-PURITY FAIL — raw vendor API found (use Kokkos abstractions):"
  echo "$hits"
  exit 1
fi
echo "KOKKOS-PURITY OK ($SRC_DIR is vendor-neutral)"
exit 0
