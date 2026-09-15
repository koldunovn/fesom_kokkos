#!/usr/bin/env bash
# scripts/xmach_pull_from_levante.sh — pull the paper's input bundle from DKRZ Levante and verify it.
#
#   DEST=<INPUTS> [DKRZ_USER=a270088] [ONLY=core2,ng5,forcing_1958,ic] bash scripts/xmach_pull_from_levante.sh
#
# Run ON the target machine (LUMI / MN5) inside tmux; needs an ssh login on levante.dkrz.de.
# rsync resumes, so re-running after an interruption is safe. ~25 GB in total.
set -u
DEST=${DEST:?DEST=<local inputs dir>}
U=${DKRZ_USER:-a270088}
SRC=/work/ab0995/a270088/port2/xmach/inputs
mkdir -p "$DEST"
if [ -n "${ONLY:-}" ]; then
  for d in ${ONLY//,/ }; do rsync -avP --partial "$U@levante.dkrz.de:$SRC/$d/" "$DEST/$d/"; done
  rsync -avP "$U@levante.dkrz.de:$SRC/MANIFEST.sha256" "$DEST/"
else
  rsync -avP --partial "$U@levante.dkrz.de:$SRC/" "$DEST/"
fi
cd "$DEST" && grep -E "^[0-9a-f]{64}  \./(${ONLY:-.*}//,/|)" MANIFEST.sha256 > /dev/null 2>&1
echo "verifying (files not yet transferred are reported as missing — expected with ONLY=)"
sha256sum -c MANIFEST.sha256 --quiet --ignore-missing && echo "BUNDLE VERIFIED: every transferred file matches Levante"
