# FESOM2 Kokkos port — host↔device SYNC MAP

**What this is.** The per-substep map of which memory space (host / device) is *authoritative*
for each persistent field as a timestep runs, and where a host↔device `deep_copy` (a `Field`
`sync_host`/`sync_device`) must sit. It is the GPU analogue of, and runs in lock-step with, the
**MPI halo-exchange map** that is already inlined as comments in `src/fesom_step.cpp`. Every place
the halo map says "exchange field X here" is also a place the sync map must say "X must be
host-current here" (halo pack/unpack is a host operation until M5).

**Status at M1 (this milestone).** *All compute is still on the host.* No `parallel_for` runs on
the device; the only device operation anywhere is `Kokkos::deep_copy` of `double`/`int`, which is
bitwise-exact. Therefore **every field is host-authoritative at every substep**, and the map below
is, today, uniformly "host". The value of writing it now is that it (a) fixes the *contract* the
M2/M4 kernel tasks implement field-by-field, and (b) is *executable* — the `-DFESOM_KK_SYNCCHECK`
build (see §7) walks these rails every step and aborts if the contract is ever violated.

Read with: `docs/plans/20260525-kokkos-port.md` (§M1.5, §M2 cross-cutting notes), the halo cheat
sheet in `src/fesom_step.cpp`, and `docs/KOKKOS_PORTING_LESSONS.md` (D17, L14, L21, L22).

---

## 0. The cadence decision (D17) — host-authoritative + lazy device sync

The reviewable question for M1.5 was: *how often does the evolving state (dyn / tracers / forcing /
ice) round-trip host↔device?* The decision, for the correctness phase (M1–M4):

> **Host-authoritative, synced lazily by the first device kernel that needs each field — never
> eager per-step blanket copies.** A field moves to the device only when the *specific* kernel that
> consumes it is ported (M2/M4); that kernel's task adds the `sync_device(inputs)` before it and
> `modify_device(outputs)` after it. Until then the field is never copied, so the data layer adds
> **zero** host↔device traffic and the run stays bit-identical on every backend.

Why not eager copies now: a per-step `sync_device` of all 126 fields would be (a) pure dead weight
at M1 (nothing reads them on device), (b) a maintenance trap — the copies would have to be deleted
and re-placed at the true kernel boundaries in M2 anyway, and (c) it would mask, not expose, the
real M2 sync work. The **set-once mesh geometry is the one exception**: it is pushed to the device
*once* after `compute_metrics` (`mesh_sync_geometry_device`, §8) because it never changes again.

Consequence for the M2/M4 kernel author: **the sync points are owned by the kernel task, not by a
central per-step copy.** This map tells you, per substep, exactly which inputs must be device-current
before your kernel and which outputs you must mark `modify_device` after it.

---

## 1. How to read this map — currency states & the contract

Each `Field` carries an authoritative-space tag (`Synced` / `Host` / `Device`), set by
`modify_host()` / `modify_device()` and cleared to `Synced` by a matching `sync_*` (see
`src/fesom_field.hpp`).

The contract every ported substep must honour (the "rails"):

| Situation | Required call(s) | Why |
|---|---|---|
| Host kernel wrote field via the raw alias `f.h()` | `f.modify_host()` afterwards | The DualView cannot see writes through the raw pointer (**L14**); without this, a later `sync_device` is a silent no-op and the device keeps stale data |
| About to run a **device** kernel that reads field | `f.sync_device()` before it | Pull host writes onto the device (bitwise `deep_copy`; no-op if already current) |
| A **device** kernel wrote field | `f.modify_device()` after it | Mark device newer so a later `sync_host` actually copies |
| About to **halo-exchange / I/O-gather / run a legacy host kernel** that reads field | `f.sync_host()` before it (and read via `f.h_checked()`) | Halo pack/unpack and the snapshot gather are host code; reading host while the device is authoritative is the stale-read bug the guard traps |

`h_checked()` is `h()` plus, under `-DFESOM_KK_SYNCCHECK`, an abort if the field is
device-authoritative — install it at host entry points so a *missing* `sync_host()` aborts at the
read instead of silently corrupting output. At M1 it never fires (everything is host/synced); it
earns its keep from M2 on.

**Notation below:** `[H]` = host-authoritative at M1 (the whole map today). `→dev(X)` /
`mod_dev(X)` / `→host(X)` are the calls the *future* device port of that substep must add. "halo X"
reproduces the existing MPI exchange (host) — it implies `→host(X)` once X lives on the device.

---

## 2. The ocean step — `fesom_timestep` (`src/fesom_step.cpp`)

Substeps follow the code 1:1; ported in M2.1–M2.7. Element/node/level layout is `[entity*nl+lev]`.

| # | Kernel (M2 task) | Reads | Writes | Halo after | M2/M4 sync to add |
|---|---|---|---|---|---|
| 1 | ✅ **`pressure_bv_kk`, `sw_alpha_beta_kk` (M2.1 — DONE, first device kernels)** | tracers T/S `values`; mesh `hnode`, `Z`, set-once `ulevels/nlevels_nod2D` | aux `density_m_rho0`, `hpressure`, `bvfreq`, `dbsfc`, `MLD1_ind`, `sw_alpha`, `sw_beta` | nod3D ×5 (density/hpressure/bvfreq/sw_α/sw_β), then `smooth_nod3D(bvfreq)` | **implemented:** driver IN rail `modify_host()+sync_device()` on `T,S,hnode` (host-written via raw alias, L14; `Z/ulevels/nlevels` already device-current from the one-shot push); kernels `mod_dev` their outputs; driver `sync_host()` all 7 outputs (incl. non-halo `dbsfc`→KPP, `MLD1_ind`→GM) before the halos; all 5 halo reads + `smooth_nod3D` routed through `h_checked()` |
| 1b| ✅ **GM/Redi prereq chain (M2.5b-b — DONE)** `sigma_xy`/`neutral_slope`/`init_redi_gm`/`fer_solve_gamma`/`fer_gamma2vel` | aux `bvfreq`,`sw_alpha`,`sw_beta`; tracers `T`/`S`; mesh `hnode_new`,`helem` (evolving) + set-once `gradient_sca`/`elem_*`/`nod_in_elem2D`/`zbar(_3d_n)`/`area`/levels | gm `sigma_xy`/`neutral_slope`/`slope_tapered`/`fer_tapfac`/`fer_scal`/`fer_K`/`fer_C`/`Ki`/`fer_gamma`; dyn `fer_uv` | `sigma_xy` nod2D(×2); `neutral_slope`,`slope_tapered` nod2D(nl-1,×3); `fer_C` nod2D, `fer_K`,`Ki` nod2D; `fer_gamma` nod2D(×2); `fer_uv` elem2D-full(×2) — ALL **driver halos** (no D21) | **implemented (5 device kernels, ⚠️ GM ON in pi L34 → golden path):** driver IN rail `modify_host()+sync_device()` on `bvfreq`(L27 smooth)/`sw_alpha`/`sw_beta`/`T`/`S`/`hnode_new`/`helem` (L28); kernels flow DEVICE→DEVICE (each reads its upstream's OWNED output on device); per-kernel `mod_dev` outputs + driver `sync_host` before each halo (`h_checked`); the ONLY re-push is `fer_gamma` (`modify_host()+sync_device()` after its halo — `fer_gamma2vel` reads it at HALO vertices, L30). All 5 = race-free maps/gathers/per-node TDMA → **Serial AND OpenMP bit-identical** (`FESOM_KK_VERIFY=gm` max\|Δ\|=0 on both). `fer_uv` feeds device `vert_vel` (12b) + bolus (13a); `slope_tapered`/`Ki` feed Redi (13). Wrapped gm scratch in `Field` (M2.5b-a). |
| 2 | ✅ **`pressure_force_linfs_fullcell_kk` (M2.4 — DONE)** | aux `hpressure` (EOS device output, sync_host'd + halo'd on host in substep 1); set-once mesh `gradient_sca`/`elem_nodes`/`ulevels`/`nlevels` | aux `pgf_x`,`pgf_y` (elem) | elem3D ×2 | **implemented:** driver IN rail `modify_host()+sync_device()` on **`hpressure`** — the L27 device→host(halo)→device hand-off: the kernel reads `hpressure` at the element's 3 **vertices** (which can be HALO nodes), and the substep-1 host halo write is invisible to the DualView, so a bare `sync_device()` would feed the kernel pre-halo device bytes; kernel `mod_dev(pgf_x,pgf_y)`; driver `sync_host(pgf_x,pgf_y)` before the elem3D halos (via `h_checked`). EOS-style full overwrite (no read of `density_m_rho0` — the original row's guess; the body reads only `hpressure`). |
| 3 | ✅ **`compute_vel_nodes_kk` (M2.2 — DONE)** | dyn `uv` (per-step); mesh `nod_in_elem2D` CSR / `elem_area` / levels (set-once) | dyn `uvnode` | nod3D `uvnode` | **implemented:** driver IN `modify_host()+sync_device()` on `uv` (host-written by update_vel+bolus via raw alias, L14); kernel `mod_dev(uvnode)`; driver `sync_host(uvnode)` before the halo (via `h_checked`) + the host KPP/PP readers |
| 3 | ✅ **KPP `kpp_mixing_kk` (M2.3 — DONE)** *or* ✅ **PP `pp_mixing_kk` (M2.2 — DONE)** | aux `bvfreq`, `sw_alpha/beta`, `dbsfc`; dyn `uvnode`; tracers `S`; forcing `stress_node_surf/heat_flux/water_flux/sw_3d`; mesh `hnode` | aux `Av` (elem), `Kv` (node) | KPP: **2 internal exchange points** (see §6): `smooth_blmc` (blmc ×3 + 3-sweep smoother) + the pre-elem-average (diffK ×2, ghats, viscA); PP: `Kv` nod3D, `Av` elem3D | **KPP implemented:** the driver (substep-3, non-const structs) owns the IN rail `modify_host()+sync_device()` on all the inputs at left (D21) + the OUT rail `sync_host(Av,Kv)`; `fesom_kpp_mixing_kk` owns the two INTERNAL brackets (`mod_dev`→`sync_host`→halo+smooth via `h_checked`→`mod_host`→`sync_device`) and `mod_dev(Av,Kv)`. **PP implemented:** driver IN `modify_host()+sync_device()` on `uvnode`+`bvfreq`; kernel `mod_dev(Kv,Av)` (3 launches; loop-2-before-loop-3); driver `sync_host(Kv,Av)` before halos |
| 3 | ✅ **`mo_convect_kk` (M2.2 — DONE)** | aux `bvfreq`,`Kv`,`Av` | aux `Kv`,`Av` | `Kv` nod3D, `Av` elem3D | **implemented:** driver IN `modify_host()+sync_device()` on **`bvfreq`** (first device reader AFTER the host smooth_nod3D — a bare `sync_device` would be a silent no-op, L14/L24) + `Kv`,`Av` (host-authoritative: KPP-written on the default path, or PP-halo'd); kernel `mod_dev(Kv,Av)`; driver `sync_host(Kv,Av)` before the halos (`h_checked`) |
| 4 | ✅ **`compute_vel_rhs_kk` (M2.4 — DONE) — ⚠️AB2 eps=0.1; embeds `momentum_adv_scalar_kk` (scatter + internal halo)** | dyn `uv`, `uv_rhsAB` (read by part i; read-modify-write), `eta_n` (read at 3 vertices incl HALO), `w_e`; aux `pgf_x`/`pgf_y` (substep-2 device output, sync_host'd+halo'd); mesh `hnode` (evolving), set-once `gradient_sca`/`coriolis`/`elem_area`/`areasvol`/`edges`/`edge_tri`/`edge_cross_dxdy`/`nod_in_elem2D(_offsets)`/`elem_nodes`/levels | dyn `uv_rhs`, `uv_rhsAB`; scratch `uvnode_rhs` | **internal** nod3D `uvnode_rhs` (§6) then elem3D `uv_rhs`,`uv_rhsAB` | **implemented:** driver IN rail `modify_host()+sync_device()` on `uv`/`uv_rhsAB`/`eta_n`/`w_e`/`pgf_x`/`pgf_y`/`hnode` (all of them, L28); the AB2 `eps=0.1` is in the kernel; `momentum_adv_scalar_kk` scatters edge→node into `uvnode_rhs` via **`atomic_add`** (D22) and owns the **INTERNAL `uvnode_rhs` nod3D halo bracket** (D21, §6); kernel `mod_dev(uv_rhs,uv_rhsAB)`; driver `sync_host` both before the elem3D halos. Verify `vrhs` = L26 capture-before on `uv_rhsAB` (part i reads it; `uv_rhs` fully recomputed → not captured). |
| 5 | ✅ **`visc_filt_bidiff_kk` (M2.4 — DONE; first SCATTER + internal halo)** | dyn `uv`, `uv_rhs` (read-modify-write); dyn `u_b`/`v_b` scratch (zeroed on device); set-once mesh `edge_tri`/`elem_area`/`ulevels`/`nlevels` | dyn `uv_rhs` | **internal** elem3D `u_b`/`v_b` (§6) then elem3D `uv_rhs` | **implemented:** driver IN rail `modify_host()+sync_device()` on `uv` + `uv_rhs` (L28); two edge→element stages scatter via **`Kokkos::atomic_add`** (D22 — Serial bit-identical, OpenMP/CUDA climate-close); the function owns the **INTERNAL Uc/Vc(=`u_b`/`v_b`) elem3D halo bracket** (D21, §6); kernel `mod_dev(uv_rhs)`; driver `sync_host(uv_rhs)` before the elem3D halo (via `h_checked`). Verify `vfilt` = L26 capture-before, FULL extent (scatter writes halo elems). |
| 6 | ✅ **`impl_vert_visc_kk` (M2.4 — DONE, per-elem TDMA)** | dyn `uv_rhs` (read-modify-write), `uv`, `w_i` (read at the 3 vertices incl. HALO); aux `Av`; mesh `helem` (evolving), set-once `zbar`/`elem_nodes`/`ulevels`/`nlevels`; forcing `stress_surf` | dyn `uv_rhs` | elem3D `uv_rhs` | **implemented:** driver IN rail `modify_host()+sync_device()` on **all** of `uv_rhs`/`uv`/`w_i`/`Av`/`helem`/`stress_surf` (L28 — sync every input the body reads; the SYNC_MAP guess listed only `uv_rhs/Av/stress_surf`, but the body also reads `uv`, `w_i`, and the evolving `helem`); `forcing` const → localized `const_cast` for `stress_surf`. Kernel `mod_dev(uv_rhs)`; driver `sync_host(uv_rhs)` before the elem3D halo (via `h_checked`). Per-element TDMA → race-free, **Serial AND OpenMP bit-identical**. Verify `ivisc` = L26 capture-before (read-modify-write). |
| 7 | `compute_ssh_rhs_linfs` (M4.2) | dyn `uv_rhs`,`d_eta`; mesh | dyn `ssh_rhs` | nod2D `ssh_rhs` | **mid-step host bracket — see §5** |
| 8 | `ssh_solve_cg` (M4.2) — CG w/ `MPI_Allreduce` dots | mesh stiffness CSR; dyn `ssh_rhs` | dyn `d_eta` | nod2D `d_eta` | stays HOST through M2/M3 (§5); on device only at M4.2 |
| 9 | `update_vel` (M2.4) | dyn `uv_rhs`,`d_eta`; mesh | dyn `uv` | elem3D `uv` | `→dev(uv_rhs,d_eta)`; `mod_dev(uv)`; `→host(uv)` before halo |
| 10| `compute_hbar` (M4.2) | dyn `uv`; mesh `hbar`,`hbar_old` | dyn `ssh_rhs_old`; mesh `hbar` | nod2D `ssh_rhs_old`,`hbar` | part of the §5 bracket |
| 11| `eta_n` inline | mesh `hbar`,`hbar_old`,`ulevels_nod2D` | dyn `eta_n` | (covered) | trivial; fold into M2.4 device region or keep host |
| 12| ✅ **ALE `thickness`/`vert_vel`/`cflz`/`wvel_split` (M2.5 — DONE)** | mesh `hnode` (12a), `helem` + dyn `uv` (+`fer_uv` gm) (12b), set-once `area`/`edges`/`edge_tri`/`edge_cross`/levels; `w`,`cfl_z` (re-read after each driver halo) | mesh `hnode_new` (12a); dyn `w`[,`fer_w` gm] (12b), `cfl_z` (12c), `w_e`,`w_i` (12d) | `w` nod3D[,`fer_w` gm], `cfl_z`, `w_e`, `w_i` (all DRIVER halos — no internal exchange) | **implemented (4 device kernels, each its own rail — NO internal halo, no D21):** 12a `thickness_linfs_kk` IN `modify_host()+sync_device(hnode)`, `mod_dev(hnode_new)`, OUT `sync_host(hnode_new)` (read on HOST by tracer adv/diff substeps 13/13b — see row note); 12b `vert_vel_linfs_kk` IN `uv`(+`fer_uv` if gm), `helem`, **EDGE→NODE SCATTER (`atomic_add`, D22) + per-node level cumsum**, `mod_dev(w[,fer_w])`, OUT `sync_host` before the nod3D halo (`h_checked`); 12c `compute_cflz_kk` IN re-push `w` (just halo'd) + no-op `sync_device(hnode_new)` (Synced from 12a), per-node OWN-column accumulation (NOT a scatter → bit-identical OpenMP), `mod_dev(cfl_z)`, OUT `sync_host`; 12d `compute_wvel_split_kk` (⚠️`use_wsplit=.false.`→`w_e=w,w_i=0`) IN re-push `cfl_z`, pure map, `mod_dev(w_e,w_i)`, OUT `sync_host` before the halos. ⚠️ **GM is ON in pi** → the `fer_*` branch is LIVE (Serial `fer_w` bit-identical, OpenMP climate-close), L34. |
| 13a| bolus add (M2.5b, gm) | dyn `fer_uv`,`fer_w` | dyn `uv`,`w`,`w_e` (in place) | — | `→dev(fer_uv,fer_w)`; `mod_dev(uv,w,w_e)` — honour `feedback_bolus_divergence_balance` |
| 13| FCT tracer advection T,S (M2.6) — **AB2 eps=0.1** | dyn `uv`,`w*`; tracers; mesh | tracers `values` (T then S) | nod3D `values` per tracer | **FCT pipeline has internal exchanges — see §6**; wrap FCT scratch in `Field`; `mod_dev(values)` |
| 13b| `impl_vert_diff_tracers` (M2.7, per-node TDMA) + S-floor | tracers; aux `Kv`; forcing fluxes | tracers `values` | nod3D `values` ×2 | `→dev(values,Kv,fluxes)`; `mod_dev(values)`; `→host(values)` before halos |
| 13c| bolus sub (M2.5b, gm) | dyn `fer_uv`,`fer_w` | dyn `uv`,`w`,`w_e` (restore) | — | mirror of 13a |
| 14| ✅ **`ale_commit_thickness` (M2.5 — DONE)** | mesh `hnode_new` (Synced since 12a; substep-13 host tracer adv/diff only READ it), set-once `elem_nodes`/levels | mesh `hnode`,`helem` | `hnode` nod3D, `helem` elem3D | **implemented:** IN no-op `sync_device(hnode_new)` (device-current since 12a — never `modify_host()`, host didn't write it); `commit_thickness_kk` = `hnode:=hnode_new` flat copy then `helem` vertex-mean (2 launches, barrier orders the helem read of the copied hnode, D20); race-free maps → bit-identical all backends; `mod_dev(hnode,helem)`; OUT `sync_host(hnode,helem)` before the nod3D/elem3D halos (`h_checked`). Both EVOLVING → feed next step's substep-1 EOS + substep-6 TDMA. |

At M1 the whole column was `[H]`. **M2.1 landed substep 1 (EOS) — the first LIVE rail:** its
inputs are pushed to the device and its outputs round-trip back to the host before the halos, so
substep 1 is the worked example of the §9 checklist in action (the rest of the column is still
`[H]` until its kernel lands). The **`h_checked()` guards** at the §1 halos (all 5 EOS outputs +
`smooth_nod3D`) now actually transition `Device→Synced` each step under `-DFESOM_KK_SYNCCHECK` and
would abort if the output `sync_host()` were dropped — verified: the SYNCCHECK pi smoke runs clean
(no abort) and bit-identical. The §13-T (values) guards remain installed as forward
worked examples until M2.6 wires its rails.

**M2.2 landed substep 3 (PP mixing) — the second LIVE rail group:** `compute_vel_nodes_kk` (always-on),
`pp_mixing_kk` (PP branch), and `mo_convect_kk` (always-on) are device kernels, each with the §9
checklist wired in the driver. Two subtleties this substep added to the worked record: (a) **a mid-step
device→host→device hand-off** — `bvfreq` is produced on the device in substep 1, overwritten on the
**host** by `smooth_nod3D`, then read on the device by `mo_convect_kk`, so that rail uses
`modify_host()+sync_device()` (the host smooth is invisible to the DualView, L14/L27), not a bare
`sync_device()`; (b) on the **default KPP path** `compute_vel_nodes_kk`+`mo_convect_kk` run on the device
while KPP itself is still a HOST kernel between them, so `uvnode` round-trips device→host (kernel→halo+KPP)
and `Kv`/`Av` round-trip host→device→host (KPP→`mo_convect`→halos) — an expected within-step bounce, like
the §5 CG round-trip. The §9 (`uvnode`) and the Kv/Av halo guards now transition `Device→Synced` each step
under SYNCCHECK and ran clean on **both** the KPP and PP branches.

**M2.3 put the rest of substep 3 (KPP) on the device** — so on the default path substep 3 is now
*fully* device-resident except its internal halo round-trips: `compute_vel_nodes_kk` → halo →
`kpp_mixing_kk` (device, with its two internal exchange brackets) → `mo_convect_kk` → halos. The
within-step `Kv/Av` host↔device bounce noted for M2.2 (b) is now just the KPP OUT rail (`sync_host`)
feeding `mo_convect`'s IN rail (`modify_host+sync_device`) — kept for uniformity with the PP branch
and the verify. The KPP IN rail (driver) syncs **all** inputs KPP reads (D21/L28: the Serial gate
can't catch a missing `sync_device`, so they are synced explicitly, not assumed device-current from
substep 1). The two KPP internal brackets transitioned their `h_checked()` guards `Device→Synced`
cleanly under SYNCCHECK on the default path.

**M2.5 put the ALE block (substeps 12 + 14) on the device** — five `_kk` twins, each with its own
IN/OUT rail (rows 12/14). Unlike KPP, **no ALE kernel has an internal exchange** — every
`fesom_exchange_*` sits in the driver *between* kernels, so there is no D21 bracket; instead the data
bounces device→host(halo)→device and each kernel's IN rail re-pushes the just-halo'd input
(`modify_host()+sync_device()` on `w` before `cflz`, on `cfl_z` before `wvel_split`). Two rail
subtleties: (a) **`hnode_new` is device-resident across 12a→12c→14** — `thickness` writes it on the
device and `sync_host()`s it (the substep-13 host tracer advect/diff READ it via the raw alias, so it
MUST be host-current), leaving it `Synced`; `cflz` and `commit` then read the device copy current with a
no-op `sync_device()` and never `modify_host()` it (the host never writes it). (b) **GM is ON in the pi
smoke** (contrary to the earlier "GM off by default" note), so `vert_vel`'s `fer_w` accumulator is LIVE:
the `if(gm)` IN rail syncs `fer_uv`→device and OUT `sync_host`s `fer_w`, and the verify proved the
`gm_on` branch bit-identical on Serial (`fer_w` max|Δ|=0) and climate-close on OpenMP (≈7e-22) — L34.
Only `vert_vel` is a SCATTER (edge→node `atomic_add`, D22); the other four are race-free maps (`cflz`'s
`+=` accumulates into each node's OWN column, not a cross-thread scatter) → bit-identical on Serial AND
OpenMP. The new nod3D/elem3D halo guards (`w`/`cfl_z`/`w_e`/`w_i`/`hnode`/`helem`, all via `h_checked()`)
transition `Device→Synced` each step under SYNCCHECK and ran clean.

**M2.5b-b put the substep-1b GM chain on the device** — five `_kk` twins (row 1b), the GM
streamfunction/bolus prerequisite that **runs every step on the golden path** (GM is ON in pi, L34).
Unlike KPP (D21 internal brackets), the C twins' six internal halos move to the **driver** (the ALE
pattern), so the kernels flow DEVICE→DEVICE reading each upstream's OWNED output on the device, and the
driver does `sync_host`+halo between them. The ONE re-push is `fer_gamma` (`modify_host()+sync_device()`
after its halo) because `fer_gamma2vel` reads it at HALO vertices (L30 — the bvfreq/hpressure cross-op
shape). The IN rail syncs every input the chain reads (L28): `bvfreq` is the L27 device→host(smooth)→
device hand-off, `sw_alpha`/`sw_beta` were halo'd on the host (substep 1), `T`/`S` were pushed by the EOS
rail (re-pushed for self-containment), `hnode_new`/`helem` are evolving mesh. All five are race-free
maps/gathers/per-node TDMAs (no scatter) → **Serial AND OpenMP bit-identical** (`FESOM_KK_VERIFY=gm`
max|Δ|=0 on both, all 5 kernels × 20 steps). `fer_solve_gamma` is the per-node TDMA (the L31/`impl_vert_visc`
shape) and `compute_sigma_xy`/`fer_gamma2vel` are gathers (the L27/`compute_vel_nodes` shape). The
substep-1b chain produces `fer_uv` (consumed by the device `vert_vel`, 12b, + the bolus add, 13a) and
`slope_tapered`/`Ki` (consumed by the substep-13 Redi). The 8 new halo guards transition `Device→Synced`
each step under SYNCCHECK and ran clean (np=1 **and** np=2 CMA-off vs `…m13_nocma`).

---

## 3. The sea-ice step — `fesom_ice_step` (`src/fesom_ice.cpp`; ported in M4.3)

Runs *before* the ocean step each iteration (it overwrites `heat_flux`/`water_flux` the ocean step
consumes). Currency at M1: `[H]` throughout.

| Sub | Kernel | Reads | Writes | M4.3 sync to add |
|---|---|---|---|---|
| ocean2ice | `fesom_ocean2ice` | dyn surface `uv`, tracers SST/SSS, `eta` | ice `srfoce_u/v/temp/salt/ssh` | `→dev` ocean surface; `mod_dev(srfoce_*)` |
| EVP | `fesom_ice_evp_dynamics` (whichEVP=0) | ice `srfoce_*`,`stress_atmice_*`, work, mesh | ice `uice/vice`, work `sigma*/eps*/inv_*` | internal exchanges → bracket each; `mod_dev(uice,vice)` |
| FCT | `ice_tg_rhs` + `ice_fct_solve` | ice `uice/vice`, `data[*].values`, work `fct_*`, stiffness | ice `data[*].values*` | internal exchanges (§6 pattern); wrap `fct_massmatrix`+work; `mod_dev(values)` |
| cut_off | `fesom_ice_cut_off` | ice `data[*].values` | ice `data[*].values` (clamp) | `→dev`/`mod_dev(values)` |
| thermo | `fesom_ice_thermodynamics` + `oce_fluxes` | ice state, forcing, jra, sr | ice `thermo.*`, `flx_*`; forcing `heat_flux`,`water_flux`,`virtual_salt`,`relax_salt` | `→dev`; `mod_dev(flx_*, forcing fluxes)`; `→host(forcing)` before the ocean step reads them |
| diag | `h_ice`/`h_snow` | ice `data[AICE/MICE/MSNOW].values` | ice `h_ice`,`h_snow` | `mod_dev(h_ice,h_snow)` |

---

## 4. The main loop — forcing producers + I/O (`src/fesom_main.cpp`)

Per iteration, *before* `fesom_timestep`, these host producers finalise the surface forcing (an
**input** to the ocean step):

1. `jra55_step_cal` + `bulk_compute` → forcing `stress_*`, `heat_flux`, `water_flux`, `Ch/Ce_atm_oce`.
2. `sss_runoff_step_cal` → forcing `runoff`, `relax_salt`, `Ssurf`.
3. env knobs (`NO_WIND`/`NO_HFLUX`/`FREEZE_TS`) — host memsets of *array data* via the raw alias
   (these stay unchanged; they zero `double`s, never touch `Field` internals — L20).
4. `fesom_ice_step` (§3) — overwrites `heat_flux`/`water_flux`/`virtual_salt`/`relax_salt`.
5. `ice_oce_fluxes_mom` → forcing `stress_surf` (ice-ocean drag).
6. shortwave `cal_shortwave_rad` → forcing `chl`, `sw_3d`.

→ **Rail:** all six are host producers, so before the (future device) ocean step the forcing must
be `→dev`. That is exactly where the SYNCCHECK forcing round-trip sits (§7). Currency: `[H]`.

**I/O is a host read** of the device-backed state:
- `fesom_io_step` (monthly-mean accumulation) and `fesom_io_write_snapshot` (gather → rank-0
  serial-netcdf) read tracers/dyn/aux/ice on the host. The snapshot gather block in
  `src/fesom_io.cpp` is fully routed through `h_checked()` as the worked I/O example.
- **Rail:** from M2 on, `sync_host()` every gathered field before I/O. I/O cadence (monthly /
  snapshot) is far coarser than the step, so this is cheap; do it lazily at the I/O call, not
  per step.

---

## 5. The mid-step host round-trip (the CG SSH solver) — explicit

The SSH solve (substeps 7–10) is a **parallel CG with `MPI_Allreduce` dot products** and **stays on
the host through M2 and M3** (ported only at M4.2). So while the ocean kernels around it are on the
device (after M2), each step does a deliberate host round-trip *in the middle*:

```
... momentum on device ... → sync_host(uv_rhs, d_eta)  → [host] compute_ssh_rhs → CG → update inputs
                            → sync_device(uv, d_eta)    → ... device kernels resume (update_vel) ...
```

This is **expected, not a regression** (plan §M2 cross-cutting note). State it in the M2 acceptance.
It disappears at M4.2 when the CG itself moves to device (the dot-product reduction order becomes the
GPU non-determinism source there).

---

## 6. Intra-kernel exchanges — the map is per-*substep*, not per-top-level-kernel

Several ported routines do `fesom_exchange_*` **inside** themselves; each internal exchange is a host
operation and needs its own `sync_host → halo → sync_device` bracket (these are explicit checkboxes
in the M2.3 / M2.4 / M2.6 / M4.3 tasks):

- ✅ **`momentum_adv_scalar_kk`** (inside `compute_vel_rhs_kk`, substep 4, M2.4 — DONE) has **1
  internal exchange point**: `exchange_nod3D(uvnode_rhs, 2-comp)` between the per-node compute
  (vertical + horizontal advection + /areasvol) and the vertex→element average, bracketed
  `modify_device → sync_host → halo (h_checked) → modify_host → sync_device` (D21). The exchange is
  unconditional (no-op at np=1, like the C twin). The horizontal-advection stage is an edge→node
  **scatter** (`atomic_add`, D22).
- ✅ **`visc_filt_bidiff_kk`** (substep 5, M2.4 — DONE) has **1 internal exchange point**: the
  `exchange_elem(U_c,V_c)` between its two Laplacian stages (`u_b`/`v_b` ×1 each), bracketed
  `modify_device → sync_host → halo (h_checked) → modify_host → sync_device` inside the function
  (D21, the KPP `smooth_blmc` analogue). Exchange gated `npes>1` (no-op at np=1); the round-trip is a
  no-op on Serial. The two stages themselves are edge→element **scatters** (`atomic_add`, D22 / §`SCATTER`).
- ✅ **KPP** (`fesom_kpp_mixing_kk`, M2.3 — DONE) has **2 internal exchange points** wrapping 7
  `fesom_exchange_nod3D` calls: (1) **smooth_blmc** after `enhance` — exchange the 3 `blmc` channels,
  then a 3-sweep `fesom_smooth_nod3D` per channel (the smoother does its own internal exchanges, host);
  (2) the **pre-elem-average** exchanges after `combine` — `diffK` ch0/ch1, `ghats` (`nl-1`), `viscA`.
  Each point is bracketed `modify_device → sync_host → exchange+smooth (h_checked) → modify_host →
  sync_device` inside `fesom_kpp_mixing_kk` (the function owns these because it owns the exchanges;
  the IN/OUT rails are the driver's, D21). `ghats` is not re-synced to device after its exchange
  (gated off in CORE2, not read on device after). Candidates for on-device pack/unpack in M5.
- **FCT pipeline** (`tracer_advect_one_fct`, M2.6; and the ice FCT, M4.3) interleaves low/high-order
  flux assembly with exchanges. Same bracket per internal exchange. The edge→node flux scatter is a
  separate decision (`docs/SCATTER_STRATEGY.md`, M2.6): Serial keeps natural edge order
  (`max|Δ|==0`); GPU uses `atomic_add`; edge-coloring (if ever) is GPU-only.

---

## 7. Proving the rails — `-DFESOM_KK_SYNCCHECK` (M1.5 deliverable test)

Built green & gated this milestone (`build-synccheck`, Serial Release + `-DFESOM_KK_SYNCCHECK=ON`).
Two mechanisms, both **compiled out** in the default/production build (so the production run is
byte-for-byte the non-SYNCCHECK run — verified bit-identical to the golden on the pi smoke, np=1 and
np=2):

1. **Per-step coherence round-trip.** At the end of `fesom_timestep` (ocean), the end of
   `fesom_ice_step` (ice), and before `fesom_timestep` (forcing), a representative set of evolving
   Fields is bounced **host → device → host**:
   `modify_host(); sync_device(); modify_device(); sync_host();`. `modify_host()` first because the
   host wrote them via the raw alias this step (L14); `modify_device()` models the future device
   kernel that will write them. On Serial/OpenMP host==device so it is a no-op; on CUDA each leg is a
   bitwise-exact `deep_copy`, so **the host bytes are unchanged and the run stays bit-identical**.
   This exercises the exact `deep_copy` path and flag-transition logic the M2 rails will use, every
   step, on the real model state — before any compute moves to the device.
2. **`h_checked()` at host entry points.** A few representative halo exchanges in `fesom_step.cpp`
   (density, bvfreq, uv, T-values) and the **entire snapshot gather** in `fesom_io.cpp` read through
   `Field::h_checked()` instead of the raw alias. Pointer-identical today; under SYNCCHECK each
   aborts if the field is device-authoritative when the host reads it. At M1 they never fire (the
   round-trip leaves every field `Synced`, and nothing else sets `Device`), which **is** the proof
   that M1 is uniformly host-authoritative — exactly what this map asserts. From M2, a forgotten
   `sync_host()` before a halo/I/O read aborts at the read site.

Result this milestone: SYNCCHECK build runs the pi smoke to completion (np=1 **and** np=2),
**no guard fires**, output **ALL FIELDS BIT-IDENTICAL** to the golden.

> Note: the SYNCCHECK guard uses `fprintf`+`abort`, **not** `assert()`, because the diagnostic build
> must be `-O3` Release (to match the golden bit-for-bit) and Release defines `NDEBUG`, which would
> silently disable an `assert()` exactly in the build that matters (L21).

---

## 8. Already done — the set-once mesh geometry

`mesh_sync_geometry_device` (`src/fesom_mesh.cpp`) pushes the 33 set-once geometry/connectivity
Fields to the device **once**, right after `compute_metrics`, with `modify_host(); sync_device();`
per field (L14). These never change again, so they need no per-step sync — the M2 kernels read them
device-current for free. The time-evolving mesh state (`hnode/hnode_new/helem/hbar/hbar_old`) is
**not** in that one-shot push; it follows the per-step lazy rails above (substeps 10, 12, 14).

---

## 9. M2/M4 kernel-author checklist (what your task adds to this map)

When you port substep *K* to a device `parallel_for`:

1. **Inputs:** `sync_device()` each field *K* reads that a *host* kernel last wrote (skip ones
   already device-current from an earlier device kernel this step — that's the point of lazy sync).
2. **Outputs:** `modify_device()` each field *K* writes.
3. **Before its halo / I/O / a downstream host kernel:** `sync_host()` the field and read it through
   `h_checked()`.
4. **Intra-kernel exchanges** (KPP ×6, FCT): bracket each (§6).
5. **Scratch:** wrap *K*'s own scratch arrays in `Field` (deferred from M1 per the plan scope note).
6. Update this file's row for *K* (drop `[H]`, record the actual calls) **in the same commit**, and
   keep the SYNCCHECK build green.
