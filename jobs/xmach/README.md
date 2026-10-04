# jobs/xmach — the LUMI / MareNostrum 5 re-run of the paper's scaling ladders

Read `docs/plans/20260915-XMACH-LUMI-MN5-PACKAGE.md` first; it is the complete instruction set.

| file | role |
|---|---|
| `machine_lumi.sh`, `machine_mn5.sh` | the only place a site path, account, partition, launcher or ranks-per-node lives — fill in four paths |
| `submit_xmach.sh` | `<lumi|mn5> <gpu|cpu> <gates|path|fleet|deep>` — idempotent matrix submitter |
| `job_xmach` | one rung: warm-up + 2 legs of 300 steps, the paper's base configuration, liveness checks, one `XCSV` line |
| `job_xmach_gates` | day-0 gates: Serial determinism, GPU fidelity, halo selfcheck, transport A/B |
| `../../scripts/xmach_harvest.py` | XCSV lines → the paper's CSV schema |
| `../../scripts/xmach_pull_from_levante.sh` | rsync the input bundle from Levante + verify its manifest |
| `levante_partgen_112.sh` | Levante-side only: generated the 112-multiple partitions for MN5 GPP nodes |
