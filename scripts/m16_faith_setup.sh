#!/usr/bin/env bash
# M16 Gate 4 — build one Fortran run directory for the FAITHFULNESS matrix.
#
#   scripts/m16_faith_setup.sh <rundir> [run_length] [run_length_unit] [perturb_seed] [step_per_day] [amp]
#
# The matrix compares four arms — Fortran-DP, Fortran-SP, port-DP, port-SP — on the ONE setup both
# codes can share, plus FP64 noise twins that set the significance bar (Suvarchal's design: an SP-DP
# difference only counts if it exceeds the spread of FP64 runs seeded with a tiny IC perturbation).
#
# The setup is upstream's `setups/test_core2` (CORE2 mesh + JRA55 + PHC) with FOUR DELIBERATE
# DEVIATIONS, each forced by the requirement that both codes run the SAME physics:
#
#  1. MESH = /work/ab0995/a270088/port2/mesh/core2 (our private copy), NOT /pool.
#     They are not the same mesh: /pool's nlvls.out/elvls.out were rewritten 2026-07-03 and differ
#     from our copy at 2 nodes and 4 elements (made shallower, 22->18 / 20->19). Six points is tiny
#     but it is real bathymetry, and a bathymetry difference is not a precision difference. Both
#     codes therefore read the SAME file. (Reading /pool is allowed; writing to it never is.)
#  2. use_ocean_only_forcing = .false. — upstream's namelist.forcing.JRA turns the sftof land/sea
#     mask ON; the port has no sftof support at all (no mention in src/). .false. is the FORTRAN
#     SOURCE DEFAULT (gen_surface_forcing.F90:135), so this is a supported configuration, not a
#     hack: it is the behaviour the port's ancestor had.
#  3. Output precision 8, not upstream's 4 for temp/sst/a_ice. A study of single vs double must not
#     push its own diagnostic through a float write — prec 4 erases every difference below ~1e-7
#     relative before it can be measured.
#  4. Upstream's `fcheck:` truth values in setups/test_core2/setup.yml are NOT the anchor. They were
#     produced by a tool that ships in the CI docker image and is absent from the repo, on /pool's
#     mesh, at prec 4. Our own Fortran-DP arm is the reference; the fcheck values are at best a
#     sanity cross-check on it.
#
# Everything else is upstream's default: which_ALE='zstar', mix_scheme='KPP', whichEVP=0 (standard
# EVP, 120 subcycles), start 1958, JRA55-do-v1.4.0, PHC3.0 winter. step_per_day is an argument.
# 64 ranks on 1 node is upstream's own PR-940 benchmark posture (oracle PROVENANCE.txt).
set -eu

RUNDIR=${1:?usage: m16_faith_setup.sh <rundir> [run_length] [run_length_unit] [perturb_seed]}
RLEN=${2:-1}
RUNIT=${3:-m}
SEED=${4:-}
# step_per_day 32 (dt 2700) is upstream's test_core2 CI value; 48 (dt 1800) is this project's CORE2
# protocol dt. 2700 is measurably marginal on a cold start -- the 1-month pilot logged CFLz_max up
# to 2.46 against the 1.75 warning threshold -- and a configuration that sits near its stability
# limit is the wrong place to ask a precision question: an SP arm that fails there would be
# reporting the dt, not the precision. Production runs use 48.
SPD=${5:-32}
# Perturbation amplitude, K (gaussian sigma on temperature at the first step). TWO amplitudes are
# meaningful and they answer different questions:
#   2e-4  Suvarchal's value, used for his 60-yr dpnoise ensembles. Over decades a nudge this size
#         saturates and the spread measures the model's internal variability -- the right bar for a
#         CLIMATE comparison. Over a month or a year it has not saturated, and it is ~200x larger
#         than single-precision rounding on a ~10 K field, so it is a GENEROUS bar for SP.
#   1e-6  rounding-scale: comparable to what float32 itself does to a ~10 K temperature. The sharp
#         test is whether the SP-DP departure matches the spread this produces -- that is the claim
#         "SP behaves like a rounding-level perturbation, nothing more".
AMP=${6:-2.D-4}
# DET=1 turns on upstream's deterministic IC hole fill (namelist.tra ic_extrap_det, our own M13
# contribution, upstream PR #979, present in the oracle's a62f180). The port's counterpart is
# FESOM_IC_EXTRAP=det. Both codes default to the LEGACY fill, which is partition-dependent by
# construction -- so with DET off the two codes start from initial conditions that need not agree,
# and every later difference inherits that seed. Turning det on in BOTH is the only way to ask
# whether the code-to-code gap is dynamics or hole-filling.
DET=${7:-0}

CFG=/home/a/a270088/fesom2_sp/config
MESH=/work/ab0995/a270088/port2/mesh/core2
PHCDIR=/pool/data/AWICM/FESOM2/INITIAL/phc3.0
FORC=/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0

[ -d "$CFG" ]  || { echo "no upstream config dir: $CFG"; exit 2; }
[ -d "$MESH" ] || { echo "no mesh: $MESH"; exit 2; }

OUT="$RUNDIR/output"
mkdir -p "$OUT"

# ---------------------------------------------------------------- namelist.config
cat > "$RUNDIR/namelist.config" <<EOF
&modelname
runid = 'fesom'
/
&timestep
step_per_day      = $SPD
run_length        = $RLEN
run_length_unit   = '$RUNIT'
/
&clockinit
timenew = 0.0
daynew  = 1
yearnew = 1958
/
&paths
MeshPath         = '$MESH/'
ClimateDataPath  = '$PHCDIR/'
ResultPath       = '$OUT/'
RestartInPath    = '$OUT/'
RestartOutPath   = '$OUT/'
/
&restart_log
restart_length          = 1
restart_length_unit     = 'y'
raw_restart_length      = 1
raw_restart_length_unit = 'off'
bin_restart_length      = 1
bin_restart_length_unit = 'off'
logfile_outfreq         = 320
/
&ale_def
which_ALE          = 'zstar'
use_partial_cell   = .false.
/
&geometry
cartesian       = .false.
fplane          = .false.
cyclic_length   = 360
rotated_grid    = .true.
force_rotation  = .true.
alphaEuler      = 50.
betaEuler       = 15.
gammaEuler      = -90.
/
&calendar
include_fleapyear = .true.
/
&run_config
use_ice                  = .true.
use_cavity               = .false.
use_cavity_partial_cell  = .false.
use_floatice             = .false.
use_sw_pene              = .true.
flag_debug               = .false.
use_transit              = .false.
use_hosing               = .false.
hosing_mode              = 'surf'
hosing_hSv               = 0.0
/
&machine
n_levels = 2
n_part   = 2, 128
/
&icebergs
use_icesheet_coupling = .false.
ib_num                = 1
use_icebergs          = .false.
steps_per_ib_step     = 8
ib_async_mode         = 0
l_allowgrounding      = 2
/
&io_parallel
parallel_write    = .false.
n_writers         = 0
n_writers_restart = -1
n_readers_restart = -1
chunk_levels      = 8
/
EOF

# ---------------------------------------------------------------- namelist.io
# Only the four scalars the comparison needs, monthly means, DOUBLE (deviation 3 above).
cat > "$RUNDIR/namelist.io" <<'EOF'
&diag_list
ldiag_solver      = .false.
lcurt_stress_surf = .false.
ldiag_curl_vel3   = .false.
ldiag_Ri          = .false.
ldiag_turbflux    = .false.
ldiag_salt3D      = .false.
ldiag_dMOC        = .false.
ldiag_diapmix     = .false.
ldiag_DVD         = .false.
ldiag_forc        = .false.
ldiag_extflds     = .false.
ldiag_destine     = .false.
ldiag_trflx       = .false.
ldiag_uvw_sqr     = .false.
ldiag_trgrd_xyz   = .false.
ldiag_cmor        = .false.
/
&nml_general
io_listsize       = 120
vec_autorotate    = .false.
compression_level = 1
/
&nml_list
io_list =  'sst       ',1, 'm', 8,
           'sss       ',1, 'm', 8,
           'ssh       ',1, 'm', 8,
           'a_ice     ',1, 'm', 8,
           'm_ice     ',1, 'm', 8,
           'temp      ',1, 'm', 8,
           'salt      ',1, 'm', 8,
/
EOF

# ---------------------------------------------------------------- physics namelists
# Straight from upstream, with mix_scheme='KPP' (namelist.oce.core2 == the CORE2 production choice)
# and whichEVP=0 (namelist.ice default = standard EVP, 120 subcycles).
cp "$CFG/namelist.oce.core2" "$RUNDIR/namelist.oce"
cp "$CFG/namelist.ice"       "$RUNDIR/namelist.ice"
cp "$CFG/namelist.tra"       "$RUNDIR/namelist.tra"
if [ "$DET" = 1 ]; then
    sed -i "s/^ *ic_extrap_det *= *\.false\./ic_extrap_det = .true./" "$RUNDIR/namelist.tra"
    grep -q "ic_extrap_det = .true." "$RUNDIR/namelist.tra" \
      || { echo "FATAL: DET=1 requested but the ic_extrap_det edit did not apply"; exit 3; }
fi
cp "$CFG/namelist.dyn"       "$RUNDIR/namelist.dyn"
cp "$CFG/namelist.cvmix"     "$RUNDIR/namelist.cvmix"
cp "$CFG/namelist.icepack"   "$RUNDIR/namelist.icepack" 2>/dev/null || true

# IC perturbation (the FP64 noise twins). Upstream ships &oce_perturb in gen_ic3d.F90; a seed
# argument appends the block. Gaussian on temperature at the first step; see AMP above for why two
# amplitudes are run. Read sequentially from namelist.oce AFTER &oce_dyn (gen_model_setup.F90:140),
# so appending at the end of the file is where it belongs.
if [ -n "$SEED" ]; then
cat >> "$RUNDIR/namelist.oce" <<EOF

&oce_perturb
lperturb       = .true.
perturb_mode   = 'first_step'
perturb_method = 'gaussian'
perturb_seed   = $SEED
temp_perturb   = 0.0, $AMP
salt_perturb   = 0.0, 0.0
/
EOF
fi

# ---------------------------------------------------------------- namelist.forcing
# Upstream's JRA template is already Levante-pathed; the one edit is deviation 2.
sed -e "s|use_ocean_only_forcing *= *\.true\.|use_ocean_only_forcing = .false.|" \
    "$CFG/namelist.forcing.JRA" > "$RUNDIR/namelist.forcing"
grep -q "use_ocean_only_forcing = .false." "$RUNDIR/namelist.forcing" \
  || { echo "FATAL: the ocean-mask edit did not apply — check namelist.forcing.JRA"; exit 3; }

# ---------------------------------------------------------------- clock (cold start)
# Two identical lines == initial run (gen_modules_clock.F90 clock_init).
printf ' 0.0 1 1958\n 0.0 1 1958\n' > "$OUT/fesom.clock"

echo "run dir ready: $RUNDIR   ($RLEN$RUNIT, step_per_day=$SPD -> dt=$((86400/SPD))s, seed=${SEED:-none}${SEED:+, amp=$AMP K}, det=$DET)"
