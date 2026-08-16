#!/usr/bin/env bash
# jupiter_fetch.sh — RUN THIS ON JUPITER. Pulls everything M14 needs from Levante.
#
#   bash scripts/jupiter_fetch.sh            # fetch what is missing
#   DRYRUN=1 bash scripts/jupiter_fetch.sh   # show what would move, transfer nothing
#   ONLY=partitions bash scripts/jupiter_fetch.sh
#
# Direction: JUPITER pulls FROM Levante over ssh. You need to be able to
# `ssh a270088@levante.dkrz.de` from the JUPITER login node — set that up first
# (ssh-copy-id, or an agent-forwarded key). Nothing here writes to Levante.
#
# 🔴 WHAT IS AND IS NOT ALREADY THERE (from the July M7 campaign, findings doc
# 20260723-m7-JUPITER-scaling-FINDINGS.md):
#   ALREADY ON JUPITER — do not re-fetch:
#     meshes + STOCK dist_N   /e/scratch/e-sta-destine/koldunov1/meshes  (ng5 to 8192, dars to 4096)
#     forcing 1958            /e/scratch/e-sta-destine/koldunov1/forcing_1958
#     PHC IC + Sweeney chl    /e/scratch/hclimrep/koldunov1/meshes/{phc3.0,Sweeney}
#   NEW FOR M14 — this script's real job:
#     the M11 OPTIMISED partitions (mesh_m11 zoo + the certified promotions)
# The default run moves ~6.3 GB (NG5's optimised set is ~4.3 GB of that: 7 rungs x ~0.6 GB —
# an optimised partition is the same size as the stock one it replaces). The mesh/forcing
# stanzas are kept, guarded by an existence check, so a purged scratch still works.
set -u

LEV=${LEV:-a270088@levante.dkrz.de}
SCRATCH=${SCRATCH:-/e/scratch/e-sta-destine/koldunov1}
MESHES=$SCRATCH/meshes
MESHOPT=$SCRATCH/meshes_m11
FORCING=$SCRATCH/forcing_1958
SHARED=${SHARED:-/e/scratch/hclimrep/koldunov1/meshes}
ONLY=${ONLY:-all}
RSYNC="rsync -avP --partial"
[ -n "${DRYRUN:-}" ] && RSYNC="$RSYNC --dry-run"

# The seven static files the port actually opens. NG5's directory is 27 GB; these are 2.0 GB.
# The rest is fesom.mesh.diag.nc, the IFS griddes files and pyfesom caches — never read here.
STATICS=(nod2d.out elem2d.out aux3d.out nlvls.out elvls.out edges.out edge_tri.out)

say () { printf '\n=== %s\n' "$*"; }
want () { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

# --- 1. the M11 optimised partitions — THE NEW THING -------------------------
# Engine per mesh = M11's GPU winner: core2/farc/ng5 = a5_u30 (slack UFACTOR=30),
# dars = a4u30 (MINCONN+CONTIG+u30). These are CANDIDATES to race on JUPITER, not
# certified winners there: M11's own finding is that the winner is point-specific,
# and every M11 number was earned on Levante hardware.
# 🔴 Copy ONLY the rungs this machine can use. The engine directories on Levante also hold
# CPU-scale partitions the JUPITER ladder can never reach — ng5/a5_u30 goes to dist_40960, and
# core2/farc/dars to 2048. Measured, those unusable rungs are ~5.8 GB (ng5 alone ~4.3: dist_2048
# 0.67, 4096 0.71, ... 40960 1.20), so a whole-directory rsync would roughly DOUBLE the transfer
# — 12 GB instead of 6.3. Worth avoiding; not the order-of-magnitude I first claimed.
if want partitions; then
    say "M11 optimised partitions -> $MESHOPT"
    mkdir -p "$MESHOPT"
    opt_rungs () { case $1 in
        core2) echo 4 8 16 32 64 ;;
        farc)  echo 4 8 16 32 64 128 ;;
        dars)  echo 16 32 64 128 256 512 ;;
        ng5)   echo 16 32 64 128 256 512 1024 ;;
    esac; }
    for spec in core2:a5_u30 farc:a5_u30 dars:a4u30 ng5:a5_u30; do
        m=${spec%%:*}; e=${spec##*:}
        src=/work/ab0995/a270088/port2/mesh_m11/zoo/$m/$e
        mkdir -p "$MESHOPT/$m/$e"
        for n in $(opt_rungs "$m"); do
            exp=$((2*n+1))
            [ "$(ls "$MESHOPT/$m/$e/dist_$n" 2>/dev/null | wc -l)" -eq "$exp" ] && \
                { echo "  $m/$e/dist_$n present ($exp files)"; continue; }
            # 🔴 Check completeness on the SOURCE first. A partitioner writing dist_1024 right
            # now has a directory that exists and is half-full; rsync would copy it happily and
            # the model would then fail at rank 1670 with a missing my_list. Verified real:
            # ng5/a5_u30/dist_1024 was at 1669/2049 files while this script was being written.
            got=$(ssh "$LEV" "ls $src/dist_$n 2>/dev/null | wc -l" 2>/dev/null || echo 0)
            if [ "${got:-0}" -ne "$exp" ]; then
                echo "  !! SKIP $m/$e/dist_$n — Levante has $got/$exp files (still generating?)"
                continue
            fi
            $RSYNC "$LEV:$src/dist_$n/" "$MESHOPT/$m/$e/dist_$n/"
            got=$(ls "$MESHOPT/$m/$e/dist_$n" 2>/dev/null | wc -l)
            [ "$got" -eq "$exp" ] || echo "  !! $m/$e/dist_$n INCOMPLETE after copy: $got/$exp"
        done
    done
    say "M11 certified promotions -> $MESHOPT/certified"
    mkdir -p "$MESHOPT/certified"
    $RSYNC "$LEV:/work/ab0995/a270088/port2/mesh_m11_certified/" "$MESHOPT/certified/"
fi

# --- 2. mesh statics + stock dists (only if absent) ---------------------------
if want meshes; then
    for m in core2_private farc dars ng5; do
        src=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/${m/core2_private/core2}
        # 🔴 CORE2 comes from the PRIVATE copy, never /pool: /pool's nlvls/elvls are
        # swapped (L73). This is the gate mesh — copy it, never regenerate it.
        [ "$m" = core2_private ] && src=/work/ab0995/a270088/port2/mesh/core2
        if [ -f "$MESHES/$m/nod2d.out" ]; then
            echo "  $m statics present, skipping"
        else
            say "$m statics -> $MESHES/$m"
            mkdir -p "$MESHES/$m"
            for f in "${STATICS[@]}"; do $RSYNC "$LEV:$src/$f" "$MESHES/$m/" || true; done
        fi
    done
fi

# --- 3. stock dist_N for the JUPITER rungs (only the missing ones) ------------
# 4 ranks per node. 256 nodes = 1024 ranks is the declared ceiling.
if want dists; then
    fetch_dists () {   # fetch_dists <local mesh dir> <remote roots...> -- <ranks...>
        local dst=$1; shift; local roots=(); local ranks=()
        while [ "$1" != -- ]; do roots+=("$1"); shift; done; shift; ranks=("$@")
        for n in "${ranks[@]}"; do
            [ -d "$dst/dist_$n" ] && { echo "  dist_$n present"; continue; }
            for r in "${roots[@]}"; do
                if ssh "$LEV" "test -d $r/dist_$n" 2>/dev/null; then
                    $RSYNC "$LEV:$r/dist_$n/" "$dst/dist_$n/" && break
                fi
            done
        done
    }
    P=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
    W=/work/ab0995/a270088/port2/mesh
    say "stock dists"
    fetch_dists "$MESHES/core2_private" "$W/core2" "$W/core2_bigpart" -- 4 8 16 32 64
    fetch_dists "$MESHES/farc"  "$P/farc"  "$W/farc_bigpart"  -- 4 8 16 32 64 128
    fetch_dists "$MESHES/dars"  "$P/dars"  "$W/dars_bigpart"  -- 16 32 64 128 256 512
    fetch_dists "$MESHES/ng5"   "$P/ng5"   "$W/ng5_bigpart"   -- 16 32 64 128 256 512 1024
fi

# --- 4. forcing + initial condition (only if absent) --------------------------
if want forcing; then
    if [ -f "$FORCING/tas.1958.nc" ]; then
        echo "  forcing present, skipping"
    else
        say "JRA55 1958 forcing (~11 GB) -> $FORCING"
        mkdir -p "$FORCING"
        F=/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0
        $RSYNC "$LEV:$F/{uas,vas,huss,rsds,rlds,tas,prra,prsn}.1958.nc" "$FORCING/"
        $RSYNC "$LEV:$F/PHC2_salx.nc" "$LEV:$F/CORE2_runoff.nc" "$FORCING/"
    fi
    [ -f "$SHARED/phc3.0/phc3.0_winter.nc" ] || {
        say "PHC initial condition -> $SHARED/phc3.0"
        mkdir -p "$SHARED/phc3.0"
        $RSYNC "$LEV:/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc" \
               "$SHARED/phc3.0/"; }
    [ -f "$SHARED/Sweeney/Sweeney_2005.nc" ] || {
        say "chlorophyll climatology -> $SHARED/Sweeney"
        mkdir -p "$SHARED/Sweeney"
        $RSYNC "$LEV:/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc" "$SHARED/Sweeney/"; }
fi

# --- 5. verify ----------------------------------------------------------------
say "inventory"
for m in core2_private farc dars ng5; do
    s=0; for f in "${STATICS[@]}"; do [ -f "$MESHES/$m/$f" ] && s=$((s+1)); done
    printf "  %-14s statics %d/7   stock dists: %s\n" "$m" "$s" \
        "$(ls -d "$MESHES/$m"/dist_* 2>/dev/null | sed 's#.*/dist_##' | sort -n | tr '\n' ' ')"
done
for d in "$MESHOPT"/*/*/; do
    [ -d "$d" ] || continue
    printf "  opt %-22s %s\n" "${d#$MESHOPT/}" \
        "$(ls -d "$d"dist_* 2>/dev/null | sed 's#.*/dist_##' | sort -n | tr '\n' ' ')"
done
echo
echo "A dist_N directory holds rpart.out + one com_info<rank>.out and one my_list<rank>.out per"
echo "rank, so a complete one has exactly 2N+1 files (verified: ng5/dist_16 = 33)."
echo "Spot-check:   for d in \$MESHOPT/*/*/dist_*; do echo \"\$(ls \$d|wc -l) \$d\"; done"
