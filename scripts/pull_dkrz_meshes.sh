#!/usr/bin/env bash
# Pull the farc + dars meshes (+ gate extras) from DKRZ/Levante to JUPITER.
# Run ON a JUPITER login node — outbound direction (JUPITER→Levante) since
# pushing from Levante into JUPITER is impractical (JSC key management).
#
#   bash scripts/pull_dkrz_meshes.sh [dkrz_user]      # default a270088
#
# One ssh ControlMaster connection is opened first, so you type your DKRZ
# password ONCE; every rsync below reuses it. Safe to re-run: rsync resumes.
# Several GB over WAN — run it inside tmux/screen, not a bare terminal.
set -u
U=${1:-a270088}
H=levante.dkrz.de
DEST=/e/scratch/e-sta-destine/koldunov1/meshes
GREF=/e/scratch/e-sta-destine/koldunov1/gate_refs
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
WORK=/work/ab0995/a270088/port2
CM=$HOME/.ssh/cm-dkrz
mkdir -p "$DEST" "$GREF" "$HOME/.ssh"

echo "== opening master connection to $U@$H (password prompt) =="
ssh -o ControlMaster=auto -o ControlPath="$CM" -o ControlPersist=8h "$U@$H" true || exit 1
rs(){ rsync -avP -e "ssh -o ControlPath=$CM" "$@"; }

# Fleet needs dist_{4,8,16,32,64,128} (ranks = 4 x nodes, g1..g32); the larger
# dist rules cover the beyond-g64 scalability probe on dars — if a dist doesn't
# exist on /pool the rule simply matches nothing. Tarballs and foreign dists
# are excluded; everything else (statics + caches) transfers.
INC=(--include='dist_4/***'   --include='dist_8/***'    --include='dist_16/***'
     --include='dist_32/***'  --include='dist_64/***'   --include='dist_128/***'
     --include='dist_256/***' --include='dist_512/***'  --include='dist_1024/***'
     --include='dist_2048/***' --include='dist_4096/***'
     --exclude='dist_*' --exclude='*.tar.gz')

echo "== farc =="
rs "${INC[@]}" "$U@$H:$POOL/farc/" "$DEST/farc/"
echo "== dars =="
rs "${INC[@]}" "$U@$H:$POOL/dars/" "$DEST/dars/"

# Gate extras (small): the private CORE2 gate mesh (dist_1,2,8 — L73: never the
# /pool core2 for gates) + the Levante serial baseline snapshots for the J-G3
# cross-architecture drift documentation.
echo "== private core2 (gate mesh) =="
rs --exclude='*.tar.gz' "$U@$H:$WORK/mesh/core2/" "$DEST/core2_private/"
echo "== m6_baseline_serial (J-G3 refs) =="
rs "$U@$H:$WORK/m6_baseline_serial/" "$GREF/m6_baseline_serial/"

echo "=== transfer complete ==="
du -sh "$DEST/farc" "$DEST/dars" "$DEST/core2_private" "$GREF/m6_baseline_serial" 2>/dev/null
echo "dists present:"
for m in farc dars; do echo "  $m: $(ls -d $DEST/$m/dist_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"; done
