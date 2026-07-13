# Sourced by the M7 job scripts. Writes $OUT/SHA.txt with commit + BINARY MD5.
#
# Why this exists: env_cuda.sh's module set drops git from PATH, so `git rev-parse`
# inside a GPU job fails and SHA.txt lands EMPTY — silently. Every M7 GPU run before
# 2026-07-14 (baselines, gate) has a 0-byte SHA.txt for exactly this reason.
#
# The md5 of the binary is the REAL provenance: a commit hash does not pin the
# numbers (a build can lag or lead the tree), but the binary that produced them does.
m7_provenance() {   # m7_provenance <outdir> <binary>
    local out=$1 bin=$2 g sha md5
    g=$(command -v git || ls -d /sw/spack-levante/git-*/bin/git 2>/dev/null | head -1)
    sha=$( [ -n "$g" ] && (cd "${ROOT:-.}" && "$g" rev-parse HEAD 2>/dev/null) || true )
    : "${sha:=UNKNOWN}"
    md5=$(md5sum "$bin" 2>/dev/null | cut -d' ' -f1)
    { echo "commit $sha"
      echo "binary $bin"
      echo "md5    $md5"
      echo "job    ${SLURM_JOB_ID:-none}"
      echo "date   $(date -Is)"
    } > "$out/SHA.txt"
    echo "provenance: commit=${sha:0:12} binary_md5=$md5"
}
