#ifndef FESOM_SSH_SE_H
#define FESOM_SSH_SE_H

#include "fesom_types.h"

struct fesom_mesh;
struct fesom_dyn;
struct fesom_forcing;
struct fesom_partit;

/*
 * M12 — split-explicit barotropic solver (FESOM_SSH_MODE=se).
 *
 * Replaces the §5 SSH block (substeps 6b-10: stiffness update, SSH RHS, CG,
 * update_vel, hbar) with SM2005 AB3-AM4 barotropic subcycling per Sergey's
 * notes (ssh_sergey/subcycling.tex) + Banerjee et al. 2024. Substep 11
 * (eta_n = α·hbar + (1-α)·hbar_old, α=1) is SHARED: the SE finalize writes
 * hbar=η^{n+1}, hbar_old=ηⁿ on device and syncs them to host, so the existing
 * substep-11 host loop + push reproduces eta_n=η^{n+1} on both spaces through
 * the already-validated rail (and the two out-of-block SSHRAILS-gated pushes
 * become identity copies instead of clobbers).
 *
 * Knob contract (proper option — unrecognised value aborts):
 *   FESOM_SSH_MODE = si|unset -> the CG path, byte-identical;
 *                    se       -> this module; REQUIRES FESOM_ALE=zstar.
 *   FESOM_SE_M       substeps per baroclinic step (default 50); checked
 *                    against the mesh gravity-wave limit at startup.
 *   FESOM_SE_M_FORCE=1 downgrade the CFL abort to a warning.
 * Incompatible under se (abort at startup): FESOM_DIAG_SSHSLV, FESOM_DIAG_SPREAD,
 * FESOM_KK_VERIFY=ssh (they introspect SI-block fields). FESOM_SPEED_SSHRAILS is
 * force-resolved OFF with a notice (SI-block lever).
 *
 * Plan + gates G0-G4: docs/plans/20260813-m12-split-explicit.md.
 */

/* Cached knob reads (abort on unrecognised values). */
int    fesom_se_on(void);
int    fesom_se_m(void);
double fesom_se_chi(void);
int    fesom_se_check_on(void);   /* FESOM_SE_CHECK diagnostic family (unset/0 = off) */

/* M12b wide halo. FESOM_SE_WIDE=K: 0/unset = off (the certified per-substep
 * path); 1 = the K=1 rung — η is computed on the ring-1 (eDim) nodes from the
 * OWNER's div-CSR rows instead of being exchanged, and Ū is exchanged over
 * com_elem2D_full (element rings 1-2) once per substep instead of twice.
 * Messages/substep 2 -> 1 (measured ×0.500-0.513: com_elem2D_full reaches the
 * SAME partner ranks as com_elem2D), doubles/substep ×1.34-1.49, result BITWISE
 * identical. K >= 2 aborts: the deep-K rungs need the extended-mesh layer
 * (docs/plans/20260814-m12b-widehalo.md, Technical Details §D). */
int    fesom_se_wide(void);

/* Parse-time guards (zstar requirement, incompatible diagnostics). Call once,
 * right after fesom_ale_mode_init() — needs no mesh. */
void fesom_se_mode_init(void);

/* Startup: barotropic-CFL report (M vs the mesh gravity-wave limit; one
 * MPI_Allreduce) + SE state allocation. No-op when the knob is off (then it
 * only prints the L80 dead-knob note if FESOM_SE_* is set anyway). */
void fesom_se_startup(const struct fesom_mesh *mesh, struct fesom_partit *partit);

/* The SE block: forcing assembly + M-substep SM2005 AB3-AM4 subcycling +
 * finalize (hbar_old:=hbar, hbar:=η^{n+1}, host-synced — the coherence
 * contract; substep 11 derives eta_n from hbar as under SI). Replaces
 * substeps 6b-10. Returns M (the driver's "iteration count").
 * T5 adds the corrector trim of uv after this call. */
int fesom_se_step(int step_n, struct fesom_mesh *mesh,
                  struct fesom_dyn *dyn,
                  const struct fesom_forcing *forcing,
                  struct fesom_partit *partit);

/* Release the SE device Views. MUST be called before Kokkos::finalize()
 * (the fesom_ssh_cgpipe_free pattern). No-op when the knob never fired. */
void fesom_se_free(void);

#endif /* FESOM_SSH_SE_H */
