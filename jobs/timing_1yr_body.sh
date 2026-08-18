# timing_1yr_body.sh — shared body of job_timing_1yr_{cpu,gpu}.
#
# Task 5's last item: the leg that SIZES Task 7. One simulated year in the
# production configuration, at one node count, so the 63-year hindcasts are
# planned against a measured number instead of an extrapolated one.
#
# Why a year and not the 300 steps a scaling campaign uses:
#   - the number Task 7 actually needs is "simulated years per 12 h job", and
#     that includes the monthly output, which a 300-step leg never writes;
#   - the SSH solver's iteration count is state-dependent and drifts through the
#     seasonal cycle, so a cold-start average is not the production average;
#   - the restart write is part of the production cadence and costs 1.2 GB of
#     gather-and-write per year.
#
# Two legs per point. The short one is not a warm-up: startup (mesh read, PHC
# interpolation, forcing open) is paid once per JOB and the year cost is paid
# once per year, so sizing a 12 h chained job needs both, separately.
#
# Expects: BIN TAG NSTEPS_SHORT NSTEPS_YEAR ENVSTR NTASKS
set -u
MESH=/work/ab0995/a270088/port2/mesh/core2
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
ROOT=/work/ab0995/a270088/port_paper_v2/runs/timing/$TAG
PY=/work/ab0995/a270088/mambaforge/envs/nereus/bin/python

[ -d "$MESH/dist_$NTASKS" ] || { echo "ERROR: $MESH/dist_$NTASKS missing — no partition for $NTASKS ranks"; exit 2; }
rm -rf "$ROOT"; mkdir -p "$ROOT"
{ echo "tag       $TAG"
  echo "binary    $BIN  md5 $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "ntasks    $NTASKS  nodes=$SLURM_NNODES  $SLURM_JOB_NODELIST"
  echo "steps     short=$NSTEPS_SHORT year=$NSTEPS_YEAR  dt=1800"
  echo "env       $ENVSTR"
  echo "jobid     $SLURM_JOB_ID  $(date -Is)"; } > "$ROOT/PROVENANCE.txt"
cat "$ROOT/PROVENANCE.txt"

# The restart cadence is part of what is being timed. One write per simulated
# year: the year is the natural chunk (the monthly streams are per-year files,
# the drift watcher runs per year), and 1.2 GB x 63 is affordable. ⚠️ Whatever
# this ends up being, plan Task 7 requires the SAME value in c_bit and gpu_bit —
# writing a restart canonicalises the replicated element slots, so two runs with
# different cadences are different realisations and their byte gate would fail on
# the cadence alone.
leg () {   # $1 = dir, $2 = nsteps, $3 = restart_every (0 = none)
    local out="$ROOT/$1" steps=$2 every=$3 t0 t1
    mkdir -p "$out"
    t0=$(date +%s.%N)
    env $ENVSTR FESOM_PRINT_EVERY=2000 FESOM_RESTART_OUT="$out" FESOM_RESTART_EVERY=$every \
        srun -l --ntasks=$NTASKS "$BIN" "$MESH" "$out" 1800 $steps -1 "$PHC" 1958 \
        > "$out/run.log" 2> "$out/run.err"
    local rc=$?
    t1=$(date +%s.%N)
    echo "$rc $(echo "$t1 - $t0" | bc)" > "$out/WALL"
    printf "  %-6s rc=%-3s wall=%8.1f s   %s\n" "$1" "$rc" "$(cut -d' ' -f2 "$out/WALL")" \
           "$(grep -h 'loop timing' "$out/run.log" | tail -1)"
    return $rc
}

echo; echo "=== $TAG : short leg ($NSTEPS_SHORT steps) then one simulated year ($NSTEPS_YEAR) ==="
leg short "$NSTEPS_SHORT" 0        || { echo "ERROR: the short leg failed — nothing to time"; tail -5 "$ROOT/short/run.err"; exit 2; }
leg year  "$NSTEPS_YEAR"  "$NSTEPS_YEAR" || { echo "ERROR: the year leg failed"; tail -5 "$ROOT/year/run.err"; exit 2; }

echo; echo "=== sizing ==="
"$PY" - "$ROOT" "$NSTEPS_SHORT" "$NSTEPS_YEAR" "$NTASKS" "$SLURM_NNODES" "$TAG" <<'PYEOF'
import json, sys, pathlib
root, ns, ny, ntasks, nodes, tag = pathlib.Path(sys.argv[1]), *map(int, sys.argv[2:6]), sys.argv[6]

def wall(name):
    rc, w = (root / name / "WALL").read_text().split()
    return int(rc), float(w)

_, ts = wall("short")
_, ty = wall("year")
s_step = (ty - ts) / (ny - ns)          # the short leg cancels the fixed startup
startup = ts - ns * s_step
year_s = ny * s_step
sypd = 86400.0 / year_s
out_gb = sum(f.stat().st_size for f in (root / "year").glob("*.nc")) / 2**30
rst_gb = sum(f.stat().st_size for f in (root / "year").glob("*.restart.nc")) / 2**30
years12 = (12 * 3600 - startup) / year_s
years8 = (8 * 3600 - startup) / year_s

print(f"  per-step         {s_step:.4f} s        (from the difference of the two legs)")
print(f"  startup          {startup:8.1f} s      (paid once per job, not once per year)")
print(f"  one model year   {year_s:8.1f} s = {year_s/3600:.2f} h")
print(f"  SYPD             {sypd:8.2f}")
print(f"  output/year      {out_gb:8.2f} GiB     (of which restart {rst_gb:.2f})")
print(f"  63 years         {63*year_s/3600:8.1f} h of compute, {63*out_gb/1024:.2f} TiB of output")
print(f"  years per job     8 h wall: {years8:5.1f}     12 h wall: {years12:5.1f}")
print(f"  jobs for 63 yr    8 h wall: {-(-63//max(1,int(years8))):5d}     "
      f"12 h wall: {-(-63//max(1,int(years12))):5d}")

json.dump({"tag": tag, "ntasks": ntasks, "nodes": nodes, "s_step": s_step,
           "startup_s": startup, "year_s": year_s, "sypd": sypd,
           "out_gib_per_year": out_gb, "restart_gib_per_year": rst_gb,
           "years_per_8h": years8, "years_per_12h": years12},
          open(root / "timing.json", "w"), indent=2)
print(f"\n  written: {root}/timing.json")
PYEOF
echo "timing_1yr exit 0"
