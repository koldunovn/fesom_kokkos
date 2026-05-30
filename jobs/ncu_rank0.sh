#!/bin/bash
# Rank 0: ncu profiles FCT+GM kernels but SKIPS the first $NCU_SKIP matches (the spin-up steps,
# where uv~0 makes the kernels short-circuit) so the profiled ones are steady-state-representative.
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  ARGS=(--launch-skip "${NCU_SKIP:-650}" --launch-count "${NCU_COUNT:-90}"
        --kernel-name-base demangled
        --kernel-name "regex:${NCU_REGEX:-tracer_advect_one_fct|redi|fer_|sigma_xy|neutral_slope|gm_bolus}"
        --target-processes application-only)
  # NCU_METRICS (comma list) → targeted --metrics (few passes, captures sectors/req etc. that
  # --set basic omits); else the default --set basic (occupancy/duration/SOL). M5.18.
  if [ -n "${NCU_METRICS:-}" ]; then ARGS+=(--metrics "$NCU_METRICS"); else ARGS+=(--set "${NCU_SET:-basic}"); fi
  exec ncu "${ARGS[@]}" -o "$NCU_OUT" -f "$FESOM" "$@"
else
  exec "$FESOM" "$@"
fi
