# Next session — M5.23 (cont.): grind the remaining climate-safe comm levers (fieldN → L5 → L2)

*Paste this whole file to start. Self-contained. Written 2026-05-31 at the close of the M5.23-L3 pair-fusion session. The user's explicit choice for next session: **keep grinding the cheap comm levers** (mixed-precision deferred — see §1). A continuation of the IMPLEMENT+VALIDATE campaign: the entry point is built and proven across 11 sites, the budget is established, the remaining lever menu is mapped + code-read. Pick a lever off §3, build, validate the full ladder (§5), measure in the COMM regime (§6).*

---

## 0. State — where we are (read first)

The FESOM2 **C→C++/Kokkos** port (ocean + sea-ice) is device-resident + validated. **`master` @ tag `m5.21-coalescing-ghats-sss` (`4f9ea70`; NO git remote → local-only).** Everything below is **UNCOMMITTED working-tree** (the user commits only when asked — do NOT commit unless asked).

**Binary chain (all in `build-cuda/`, same node-mix, A/B-comparable):**
- `fesom_port_m522`  = pre-M5.22 (= the old m521 tip rebuilt).
- `fesom_port_m522b` = + M5.22 b-cluster flat lever, −3.10 % NG5, bit-identical.
- `fesom_port_m522c` = + M5.23 **L1** EVP two-field fused halo, **−9.1 % at dars-8N**, bit-identical.
- `fesom_port_m523L3` = + M5.23 **L3** 10 adjacent same-kind PAIR fusions, **−2.38 % at dars-8N / −1.06 % farc-2N**, bit-identical. **← `build-cuda/fesom_port` == this (the live binary).**
- `build-serial/fesom_port` also carries L1+L3 (the bit-id proof). ⚠️ `build-synccheck`/`build-omp` are STALE (m521-era) — rebuild only if you touch device-sync/host code.

**What L1 + L3 established (full detail: `docs/PROFILE_M522.md` §8.6/§8.7, `docs/GPU_FIDELITY.md` §M5.23, lessons L69/L70, memory [[project-m522-deep-profile]]):**

1. **The two-field fused-halo entry point `fesom_halo_field2` is DONE, validated, and live across 11 sites** (1 EVP + 10 L3 pairs). It co-packs two same-kind/same-stride fields into ONE message/neighbour (`[f0(stride) f1(stride)]`), byte-identical to two separate exchanges; Serial/OpenMP/`FESOM_HOST_HALO=1` fall back to the EXACT two legacy brackets (M5.1 Approach-B → the oracle is untouched by construction). `src/fesom_halo_device.{cpp,hpp}`.
2. **The cheap same-kind PAIR surface is EXHAUSTED.** L3 fused all 10 verified adjacent same-kind pairs (FCT/EOS/PGF/GM/vert-vel/visc/bulk) → only −2.4 %, because they are structural (once-per-step; FCT 2×), unlike L1's EVP pair which fired 120×/step. **There is no second EVP.** The remaining comm levers (§3) are each ≤~1–3 %.
3. **A comm lever shows NOTHING in the compute regime — measure at the right per-rank size.** NG5 dist_16 (4 nodes) is COMPUTE-bound (a comm lever reads ~flat). The win only appears at the comm-bound per-rank size: **dars dist_32 (8 nodes, 99k 2D-pts/rank) = the faithful NG5@16N proxy**; farc dist_8 (2 nodes, 80k/rank) brackets it. The DETERMINISTIC proof a halo lever fired is the `[halo-mpi-prof] calls/step` + `[fesom_prof] halo_pack/pack2` count drop (mesh-invariant), NOT the timing %.
4. **The SYPD reality (the real goal):** SYPD = 0.493/(s/step) at dt=180. NG5 GPU = 1.0 SYPD@16N, ~1.4@32N, ~1.8@64N. Production needs **1–2 SYPD**. The cheap comm levers stack toward only **~1.1–1.2 SYPD** at 16N; **mixed precision (≈×2) is the only lever that reaches 2** (it halves compute AND comm bytes). ⚠️ **The user CHOSE to grind the cheap comm levers first this session (mixed precision deferred, NOT rejected).** When this grind plateaus, the mixed-precision-campaign decision returns — flag it again then. (Repartitioning, the biggest single climate-safe lever ~23 %, stays SHELVED by user choice — keep in memory, don't pursue.)

---

## 1. The mission

**Harvest the remaining climate-safe comm levers (§3) — fieldN, L5 dead-exchange removal, L2 persistent requests — measuring each in the COMM regime (dars-8N), holding the full validation ladder (§5).** Each is bit-identical (L5 if-dead) and ≤~1–3 %; they stack on top of L1+L3. Honest expectation: this gets NG5@16N from ~1.0 toward ~1.1–1.2 SYPD — a real but bounded gain. **Do NOT over-engineer; one lever at a time, A/B same-allocation, report the delta + bit-identity/gate.** After the grind, re-surface the mixed-precision question (the path to 2 SYPD).

---

## 2. The proven machinery (build on it)

`src/fesom_halo_device.{cpp,hpp}`:
- `fesom_halo_exchange_device(f, kind, nl, nc, p, base_off)` — single-field device exchange: pack (`:244`) → `Kokkos::fence()` (`:251`) → Irecv/Isend (`:265`/`:271`) → `MPI_Waitall` (`:274`) → unpack (`:279`) → fence (`:286`). Requests **NOT persistent**; comm-lists `g_dev[5]` built-once (`build_lists`); buffers `s.send_d`/`s.recv_d` grown on demand (`grow(...)` `:231`).
- `fesom_halo_exchange_device2(f0, f1, kind, nl, nc, p, base_off)` (`:296`) — two-field co-pack: per-node block `[f0(fs) f1(fs)]`, stride `2*fs`; PACK gathers both into `send` (`:340-341`), UNPACK scatters both back (`:373-374`). Race-free, no atomics.
- `fesom_halo_field` / `fesom_halo_field2` (`.hpp`) — the dispatch wrappers (CUDA → device path; else the legacy host brackets).

---

## 3. The remaining comm lever menu — RECOMMENDED ORDER (the user's grind path)

**Do them in this order — L5 first (it's free measurement and it INFORMS fieldN), then fieldN, then L2.**

### L5 — dead/redundant exchange poison-test (do FIRST; LOW; bit-identical IF dead; may bank a FREE removal)
The L67 method: keep the exchange, then NaN-poison its result on the device right after, run CORE2 (gate) — if the model is byte-for-byte the clean run, the exchange has NO downstream reader → DROP it (removes a whole exchange + 2 fences, beats any fusion). Candidates (verify, don't assume dead):
- **`bvfreq`** halo `src/fesom_step.cpp:216` (NOD3D nl 1). The M5.9 comment already argues its only host readers are the diagnostic print + netCDF snap (covered by the pre-I/O sync); a prior NaN-poison "proved no model-feedback host reader" — but that was for a *sync*, not the halo. If the halo itself is dead (the device smoother re-dirties bvfreq right after at `:229`+, and GM/KPP/mo_convect read it on device AFTER the smoother), the `:216` halo may be pre-smoother-redundant → free removal. **This is why L5 precedes fieldN: if bvfreq's halo is dead, you DELETE it instead of folding it into the EOS fieldN.**
- **`uv_rhsAB`** halo `src/fesom_step.cpp:467` (ELEM3D nl 2). AB2 history; flagged a poison candidate since M5.22. If dead → free removal (also retires the §L3 decision to leave it un-fused).
- Method: add a gated NaN-poison kernel after the exchange (env `FESOM_POISON_<field>=1`), run `gpu_fidelity_gate.sh`; PASS-while-poisoned ⇒ dead ⇒ remove the exchange (the host-halo Serial path too) ⇒ re-run the full ladder.

### fieldN — generalize field2 to 3+ fields; collapse the EOS 5-block (LOW–MED; bit-identical; the concrete ≤~1 % win)
The EOS outputs are **FIVE consecutive NOD3D nl=1 same-stride halos** — `density_m_rho0` (`step.cpp:214`, currently field2'd with `hpressure`), `bvfreq` (`:216`, single), `sw_alpha`+`sw_beta` (`:226`, field2'd). L3 already took them 5→3; a fieldN takes the **3→1** (and folds `bvfreq` IN, unless L5 deleted it → then 4→1). Saves ~2 more exchanges + 4 fences/step in the every-step path.
- **Implementation (the §8.4 design, sidesteps the device-array-of-Views problem):** `fesom_halo_exchange_deviceN` with per-node block `[f0(fs) f1(fs) … f(N-1)(fs)]`, stride `N*fs`. Do **N separate pack kernels** (field i writes buffer slot `g*stride + i*fs`; disjoint regions, no fence between them) → ONE fence → ONE Irecv/Isend/neighbour → **N unpack kernels** → ONE fence. (Can't pass `Field**` into a KOKKOS_LAMBDA — device can't deref host pointers — so loop the launches on the HOST, each capturing one `f_i.d()`.) Dispatch `fesom_halo_fieldN({&f0,…}, kind, nl, nc, p)`; Serial = N sequential legacy brackets (bit-identical by construction).
- Bit-identity is the SAME argument as field2 (co-pack only). The CUDA gate catches adjacency errors (a stale co-pack). Scan for OTHER 3+ same-kind clusters while you're there (the EOS block is the obvious one; most else are pairs already fused).

### L2 — persistent MPI requests (MED; bit-identical; BROADEST reach — all ~970 calls/step)
`MPI_Send_init`/`Recv_init` once (keyed on kind/stride/buffer) + `MPI_Startall`/`Waitall` per call, replacing the fresh `Irecv`/`Isend` at `fesom_halo_device.cpp:265-274` (single) + `:356-365` (two-field). Cuts per-call setup latency on every exchange.
- ⚠️ **The gotcha:** the persistent request binds a fixed buffer address + count. `grow(s.send_d/recv_d)` (`:231`/`:321`) REALLOCATES the device buffers when a bigger exchange appears → stale request handles. Must re-`MPI_*_init` whenever the buffer grows (or pre-grow to max once at init). Also re-init when count/PE-list changes (it doesn't after warm-up — `g_dev[kind]` is built-once).
- ⚠️ There's a **host twin** for the Serial/OpenMP path (`src/fesom_halo.cpp` `fesom_halo_exchange`) — if you persist the device path, the host path is unchanged (still fresh requests) → Serial stays bit-identical. Bit-identity is by construction (same bytes, same tag/PE order, same MPI_DOUBLE counts; only the request lifecycle changes). Prove with the gate anyway.
- Effect estimate: modest but broad (latency on ~970 calls/step). The §8.2 measurement put genuine comm-proper at ~14 % of step (the rest of Waitall is load imbalance) — L2 chips the comm-proper, not the imbalance.

### (parked, not this grind) L4 CG 2→1 Allreduce — `fesom_ssh.cpp` (Allreduce `:719` scalar, `:805` already-fused 2-elem rz/rr). Chronopoulos/Eijkhout reorder to halve Allreduce count (~134→67/step). MED–HIGH, **CUDA-gate-class NOT Serial-bit-id** (reassociates the reduction). CG is ~12.5 % of step at dars-8N; this is the one comm lever that's NOT bit-identical, so it needs the gate as its acceptance bar. Pick up only if fieldN/L5/L2 underwhelm.

---

## 4. The comm code-read (verified file:line, current tree)
- Halo: `src/fesom_halo_device.cpp:204` `fesom_halo_exchange_device` (single), `:296` `fesom_halo_exchange_device2` (two-field). Irecv/Isend `:265/:271` (single), `:356/:362` (two-field); Waitall via `fesom_halo_prof_waitall` `:274/:365`; buffer grow `:231/:321`; reqs vector `:257/:348`.
- EOS halos: `src/fesom_step.cpp:214` (density+hpressure, field2), `:216` (bvfreq, single — L5/fieldN target), `:226` (sw_alpha+sw_beta, field2).
- L5 other candidate: `src/fesom_step.cpp:467` (uv_rhsAB).
- CG (L4): `src/fesom_ssh.cpp` — `MPI_Allreduce` `:719` (per-iter scalar macro), `:805` (2-elem rz/rr already fused); ~65 iters/step at dars-8N (89 at NG5).
- The 11 live `fesom_halo_field2` sites: `grep -rn fesom_halo_field2 src/`.

---

## 5. Validation ladder (every code change — the discipline that has held all campaign)
1. **Per-kernel `FESOM_KK_VERIFY=<key>` Serial `max|Δ|==0`** where there's an arithmetic change (none for a pure halo lever → step 2 is the proof).
2. **CORE2-Serial fresh-vs-saved ALL-FIELDS-BIT-IDENTICAL** — clone `jobs/job_core2_serial_m523L3` (runs build-serial CORE2 dist_8, `diff_snap.py` vs the preserved oracle **`serref_m522_saved`**). ⚠️ For fieldN/L2 this is bit-identical by construction (M5.1 Approach-B / same-bytes) — PROVE it. It catches kind/nl/nc/field typos. **It does NOT catch a CUDA adjacency/co-pack error** (Serial does the exchanges sequentially regardless) — that's the gate's job (L70).
3. **CUDA fidelity gate** `scripts/gpu_fidelity_gate.sh --fresh-oracle` (CORE2 dist_8 ice-active). PASS = the climate-close floor (worst h_ice ~7e-3; L3 was 7.29e-3). For a fusion/persistent-req: zero new atomics → floor UNCHANGED, and every fused-halo-DOWNSTREAM field at floor ⇒ no co-pack/adjacency bug. For L5: PASS-while-poisoned ⇒ the exchange is dead.
4. **A/B in the COMM regime** — clone `jobs/job_dars_l3_ab` (BEFORE=`fesom_port_m523L3`, AFTER=your new build; dars dist_32 = 8N = 99k/rank = the NG5@16N proxy) + `jobs/job_farc_l3_ab` (farc dist_8 = 2N = 80k/rank bracket). Read `loop timing` + the `[halo-mpi-prof] calls/step` + `[fesom_prof] halo_pack/pack2` count drop (the deterministic proof). dt=180 dars / 900 farc, 50 steps. (NG5 dist_16 4N = compute regime = ~flat — DON'T measure a comm lever there.)
5. **1-yr CORE2 CUDA climate** to close the milestone (corr=1.00000 vs m523L3 expected for bit-identical changes; gate + Serial bit-id make this a formality for halo levers).

---

## 6. The per-rank-size proxy method ([[feedback-per-rank-proxy]])
The comm/strong-scaling regime is set by **2D-vertices per rank**, NOT mesh size. NG5 dist_64 (16N) = 115.7k/rank; reproduce cheaply: **dars dist_32 (8N) = 98.8k/rank** (best proxy, same CG-conditioning class); farc dist_8 (2N) = 80k/rank brackets it. ⚠️ faithful for HALO geometry, NOT CG iters (conditioning: farc 229 vs dars 67 vs NG5 89 — use dars for anything CG, e.g. L4). nod2D: farc 638387, dars 3160340, ng5 7402886. Jobs: `job_dars_l3_ab`/`job_farc_l3_ab` (A/B), `job_{farc,dars}_halo_split` (barrier-isolation imbalance-vs-comm), `job_farc_nsys` (MPI message profile).

## 7. Hard constraints (carry every session)
- **Build GPU with `source ./env_cuda.sh`** (`openmpi/4.1.5-nvhpc-24.7`, CUDA-aware; env.sh's 4.1.2 SEGFAULTs on device ptrs). ⚠️ `env_cuda.sh` PURGES `git` — git ops in a separate shell. CPU/Serial builds use `env.sh`. (Builds are login-node incremental `make -C build-cuda -j8`; nvcc doesn't need a GPU. The `/usr/bin/time -v` wrapper can swallow the per-file "Building CXX" lines — check the binary size/mtime changed.)
- **Output → `/work/ab0995/a270088/port2/kokkos_gpu_runs/…`, NEVER `$HOME`** (60 GB quota). Big/NG5/CORE2/dars runs via SLURM, never login. dars/NG5 jobs `rm` the ~50 GB `*.monthly.nc`.
- **⚠️⚠️ NEVER double-submit two jobs that `rm -rf`+write the SAME output dir** (L69): a deleted snapshot mid-compare gives a FALSE diff_snap DIVERGENCE. `squeue` before re-issuing any submit. A collision only ever BREAKS a result, never fakes a PASS → on a surprise DIVERGENCE, first check all expected snapshots exist.
- **Same-day same-node perf baselines only** ([[feedback-perf-same-day-baseline]]); A/B both binaries back-to-back in ONE allocation. The absolute s/step of a baseline binary drifts run-to-run (CG-iter window + node mix) — only the WITHIN-allocation A/B delta is clean. **Device/halo/sync changes MUST pass `gpu_fidelity_gate.sh` before commit** ([[feedback-gpu-fidelity-gate]]); pi is insufficient (no ice). **Commit/push only when the user asks.** KPP is the default mix_scheme.
- ⚠️ `hnode_new` device-residency is LINFS-only (zstar would need its host rail restored) — carry the caution for any new residency work.

## 8. Pointers
- **Docs:** `docs/PROFILE_M522.md` (§1 budget 4N/16N, §8 comm deep-dive, §8.6 L1, **§8.7 L3 + the lever-surface map**); `docs/GPU_FIDELITY.md` §M5.22/§M5.22-comm/**§M5.23 (L1+L3)**; `docs/KOKKOS_PORTING_LESSONS.md` **L70** (L3 + the "Serial can't catch co-pack, gate must" + "deterministic msg-drop is the proof" lessons), **L69** (L1 + double-submit trap), L68 (budget/frontier-split), L67 (poison-test method = the L5 recipe), L63–L66 (flat-lever/PCIe history).
- **Memory:** [[project-m522-deep-profile]] (the full M5.22+L1+L3 record), [[feedback-per-rank-proxy]], [[feedback-perf-same-day-baseline]], [[feedback-gpu-fidelity-gate]], [[reference-build-run]], [[reference-cuda-aware-mpi]].
- **Code (build on):** `src/fesom_halo_device.{cpp,hpp}` (`fesom_halo_exchange_device2` + `fesom_halo_field2` = the fieldN/L2 surface), `src/fesom_step.cpp:214-226` (EOS block = fieldN target), `:216`/`:467` (L5 candidates), `src/fesom_ssh.cpp` (CG = L4).
- **Jobs (clone the L3 set):** `job_dars_l3_ab` / `job_farc_l3_ab` (comm-regime A/B), `job_core2_serial_m523L3` (Serial bit-id), `scripts/gpu_fidelity_gate.sh`, `scripts/diff_snap.py`. Oracle: `serref_m522_saved`. Live binary `build-cuda/fesom_port` == `fesom_port_m523L3`; baselines `_m522c` (L1, the L3 BEFORE), `_m522b`, `_m522`.

## 9. Bottom line
The two-field fused-halo entry point is built and proven (L1 −9.1 %, L3 −2.4 %, both bit-identical, 11 live sites), and the cheap same-kind PAIR surface is exhausted. **This session (user's choice): grind the remaining climate-safe comm levers — L5 poison-test (free removals, informs fieldN) → fieldN (EOS 5-block 3→1) → L2 persistent requests (broadest) — each measured in the COMM regime (dars-8N proxy) with the full ladder.** Expect ≤~1–3 % each, stacking toward ~1.1–1.2 SYPD at 16N. Measure before building, A/B in the right regime, hold the Serial-bit-id + gate on every change (the gate is the ONLY catcher of a CUDA co-pack/adjacency bug), trust the deterministic msg-count drop over timing jitter, never double-submit to a shared dir. **When this grind plateaus, re-surface the mixed-precision (≈×2) campaign — it remains the only path to the 2-SYPD target.** Repartitioning stays shelved (user's call).
