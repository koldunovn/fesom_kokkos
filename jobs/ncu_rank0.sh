#!/bin/bash
# Rank 0: ncu profiles FCT+GM kernels but SKIPS the first $NCU_SKIP matches (the spin-up steps,
# where uv~0 makes the kernels short-circuit) so the profiled ones are steady-state-representative.
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  exec ncu --set basic --launch-skip "${NCU_SKIP:-650}" --launch-count "${NCU_COUNT:-90}" \
           --kernel-name-base demangled \
           --kernel-name "regex:${NCU_REGEX:-tracer_advect_one_fct|redi|fer_|sigma_xy|neutral_slope|gm_bolus}" \
           --target-processes application-only \
           -o "$NCU_OUT" -f "$FESOM" "$@"
else
  exec "$FESOM" "$@"
fi
