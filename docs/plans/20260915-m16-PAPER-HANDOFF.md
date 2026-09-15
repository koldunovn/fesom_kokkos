# M16 → paper handoff (2026-09-15)

Read this first in the paper session (`~/Suvi/paper_sp`, GMD draft; its Kokkos "shadow" data are
**M8-era** (`data/kokkos_m8.csv`, `kokkos63_*.nc` from `port2/mp`, which is GONE) and are superseded
by everything below). The full record is `docs/MIXED_PRECISION_M16.md` (§3r–§3v fixes, §4 the
instrument ladder, **§5a/§6a/§7a results**); the registry is `docs/PRECISION_ISLANDS.md`.

## The binary
`bin/final` = `/work/ab0995/a270088/port2/m16/bin/final/{dp,sp}/{fesom_port_serial,fesom_port_cuda}`,
commit `dfab63c`, `PROVENANCE.txt` per dir (md5s, `USE_SINGLE_PRECISION`). Gate 0 (FP64 byte-identity
vs `ref0`) PASS; ctest 5/5 both precisions; CUDA with `Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF`.
Oracle: upstream FESOM/fesom2 main `a62f180` (#940 + #984 #986 #995 #997), Intel,
`/work/ab0995/a270088/port2/m16/oracle/{dp,sp}` (`_instr` twins carry the env-gated instruments).

## Faithfulness (CORE2 mesh, JRA55, PHC, zstar, KPP, EVP; `opt_visc=5`, `ice_diff=0` both codes)
Roots under `/work/ab0995/a270088/port2/m16/faith/`; reader `scripts/m16_faith_compare.py`
(`--year YYYY|all`), ensemble/band reader `scripts/m16_analysis/final_matrix.py`, GPU reader
`scripts/m16_analysis/gpu_matrix.py`. Metric: relL2 over the shared valid mask; each code's SP vs its
own DP; FP64 σ=1e-6 K / 2e-4 K twins = the code's own noise envelope.

| claim | root | arms | number (Dec 1958 unless said) |
|---|---|---|---|
| port SP is as faithful as upstream SP, globally | `final_year` | pdp psp fdp fsp + 5 seeds × 2 σ per code + 3 SP members per code | ratio sst 0.76±0.08, temp 0.65±0.06, salt 0.83±0.09, ssh 1.18±0.05, a_ice 1.95±0.06 |
| the one residual: Antarctic sea-ice band | same | same | 60–70°S sst 2.63±0.02, temp 2.06±0.01, salt 1.91±0.03; tropics 0.52–0.62 |
| the residual is absolute-salinity float rounding | `final_year_anom` (#986 ON both codes) | fdp fsp pdp psp | band sst 2.63→1.21, salt 1.91→1.11; global ratio 0.32–0.38 (port 3× better) |
| multi-year | `final_3yr` | fdp fsp pdp psp, 1958–1960 | T/S ratio 0.5–0.75 all three years; DP-vs-DP ceiling 1.0e-2 sst by yr 2; F SP−DP half of it by yr 3, port a third |
| GPU = CPU | `final_year_gpu` | gdp gdp_2 gsp (CUDA 2×4 A100) + pdp psp (Serial, opt 7) | CUDA SP−DP = Serial SP−DP to 2–6 %; 5× (global) / 17× (band) above CUDA self-noise |
| the fix that made this true | §3u/§3v: Redi/GM composed onto absolute S instead of `del_ttf` | `year_g1`…`year_g3` | month-1 salt ratio 1.95 → 1.02 |
| conformance defects fixed on the way (nulls for the gap) | §3r CG in dbl_t; §3s opt_visc pin; §4b ice_diff pin; §4d EVP_STEPS | — | — |

Caveats the paper must carry: (1) the per-step site where the port rounds absolute S more than
upstream in the band is NOT identified (same expressions in both codes, no FMA in either binary) —
state it as measured + class-identified + removed by upstream's own #986; (2) "SP is inside the
noise" is a multi-year claim — in 12 months neither code's SP−DP is inside its FP64 envelope
(F 1.5×, port 2.4×), and the port's envelope is 2× tighter than upstream's (§3b-year-b); (3) GPU
arms run `opt_visc=7` (bcksct is host-only); (4) the code-to-code DP gap is chaos (§3b), never
compare port-SP to Fortran-DP directly.

## Speed (knobs-off: `FESOM_SPEED=1` + `FESOM_IC_EXTRAP=det`, plain CG, #984 precond, no M9/M10/M11/M12;
## port defaults linfs · opt_visc 7 · `WSPLIT=1` off CORE2; ABBA dp sp sp dp, 300 steps, min over legs, own timer)
- GPU ladder (`port2/m14/gladder.<job>.out`, §6a table): SP/DP CORE2 0.825→0.94 (4→64 GPUs; stops
  scaling at 8), fArc 0.816→0.93 (8→64), dars 0.823/0.768 (8/16), NG5 0.837 (16); device memory −43…−46 %.
  PENDING at 12:00: dars 32/64/128 GPUs, NG5 32/64/128 GPUs (27459862/863/887, 865/866/888).
- CPU port ladder (`port2/m14/ladder.<job>.out`, §7a): CORE2 0.658→0.839 (64→864 r); fArc
  0.670→0.886 (512→4096); dars 0.642→0.745 (1024→8192); NG5 0.662/0.642/0.683 (2048/4096/8192).
- CPU Fortran ladder, matched physics (`port2/m16/fladder/fladder.<job>.out`, `FLADDER` line):
  CORE2 0.556→0.848; fArc 0.620/0.672/0.789/0.916. PENDING: dars ×4, NG5 ×3 (27468359–365).
  ⚠️ the first Fortran ladder (27460035–056) ran zstar+opt5 — CORE2 only, do not use off CORE2.
- Absolute: port CPU is 20–30 % slower than the Intel Fortran at DP (both scalar SSE); CUDA CORE2 1×4
  DP 0.068 s/step vs CPU 1×64 0.435.
- Suvarchal's pre-merge CORE2 SP/DP (0.55/0.56/0.61 at 64/128/256 r) agrees with ours to 0.01–0.05.

## What is NOT in hand
- The 60–70°S per-step site (see caveat 1). Not to be hunted further (memory: 26 instruments at parity).
- Energy/cost numbers; the 60-yr climate arms are Suvarchal's Fortran runs, not ours.
- Nothing pushed: `m16-precision` is 35 commits ahead of origin (ask before push; no session lines).
