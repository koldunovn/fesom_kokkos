#ifndef FESOM_EOS_H
#define FESOM_EOS_H

#include "fesom_types.h"
#ifdef __cplusplus
#include <cstddef>   /* std::size_t for the *_kk device-twin signatures */
#endif

struct fesom_mesh;
struct fesom_aux;
struct fesom_tracers;
struct fesom_partit;

/*
 * Jackett-McDougall (JM, EOS80) equation of state — split form. Returns the
 * three secant-bulk-modulus coefficients (bulk_0, bulk_pz, bulk_pz2) and the
 * potential density rhopot. In-situ density is then assembled by the caller as
 *
 *   bulk = bulk_0 + z*(bulk_pz + z*bulk_pz2)
 *   rho  = bulk * rhopot / (bulk + 0.1 * z * state_eq_int)
 *
 * Mirror of densityJM_components (oce_ale_pressure_bv.F90:2605-2669).
 */
void fesom_eos_jm_components(real_t t, real_t s,
                             real_t *bulk_0, real_t *bulk_pz, real_t *bulk_pz2,
                             real_t *rhopot);

/*
 * Phase 1 subset of pressure_bv (oce_ale_pressure_bv.F90:194-501):
 *   - JM-EOS at every wet level
 *   - density_m_rho0  = rho_in_situ - density_0  (use_density_ref=.false.)
 *   - hpressure       (linfs branch, full cells, no cavity)
 *   - bvfreq          (N², padded; horizontal smoothing done by fesom_smooth_nod3D,
 *                      called from fesom_step after the bvfreq halo exchange)
 *   - dbsfc           (buoyancy re surface for KPP bldepth; stored unconditionally,
 *                      PP never reads it — Fortran oce_ale_pressure_bv.F90:332,337,339)
 *   - MLD1_ind        (Large et al. 1997 / FESOM 1.4 — Phase G2a;
 *                      stored 0-based, used by GM init_Redi_GM)
 *
 * Deferred:
 *   - vertical N² smoothing (N2smth_v=.false. in CORE2)
 *   - MLD2/MLD3 (Levitus, Griffies)
 *   - density_dmoc (diagnostic)
 *   - cavity / partial-cell branches
 */
void fesom_pressure_bv(const struct fesom_tracers *tracers,
                       const struct fesom_mesh    *mesh,
                       struct fesom_aux           *aux);

/*
 * M2.1 DEVICE kernel twin of fesom_pressure_bv (Kokkos parallel_for over owned
 * nodes; level loops inside the lambda; verbatim arithmetic). Production path
 * from M2.1. Requires its inputs (tracers T/S, mesh hnode) device-current — the
 * step driver's EOS input rail (sync_device) supplies that; marks its aux
 * outputs modify_device(). See docs/SYNC_MAP.md §1. The host C twin above stays
 * in-tree as the FESOM_KK_VERIFY=eos oracle until M2 closes.
 */
void fesom_pressure_bv_kk(const struct fesom_tracers *tracers,
                          const struct fesom_mesh    *mesh,
                          struct fesom_aux           *aux);

/*
 * Area-weighted node-patch horizontal smoother — port of smooth_nod3D
 * (gen_support.F90:99-198). Smooths `arr` [(myDim+eDim)*nl] in place with
 * `n_smooth` sweeps, each an element-patch area-weighted average followed by a
 * halo exchange. The caller must supply a valid halo `arr` on entry. Used for N²
 * (N2smth_h=.true., N2smth_hidx=1).
 */
void fesom_smooth_nod3D(real_t *arr, int nl, int n_smooth,
                        const struct fesom_mesh *mesh, struct fesom_partit *p);

/* DEVICE twin (M5.5, lever B): same smoother on-device (no host round-trip).
 * arr_fld DEVICE-current with a valid halo in; DEVICE-authoritative + halo'd out. */
#ifdef __cplusplus
namespace fesom { template <class> class FieldT; using Field = FieldT<double>; }
void fesom_smooth_nod3D_kk(fesom::Field &arr_fld, int n_smooth,
                           const struct fesom_mesh *mesh, struct fesom_partit *p,
                           std::size_t base = 0,           /* slab offset for multi-channel fields */
                           int nslab = 1,                  /* M5.12d: # contiguous channels to smooth in one call */
                           std::size_t slab_stride = 0);   /* byte... element offset between channels (n_nod*nl) */

/* FESOM_KK_VERIFY=smooth (M5.18): isolated per-kernel gate for the device smoother — runs
 * the production kernel then the host C twin on a capture-before snapshot and asserts Serial
 * max|Δ|==0 over the owned region (the eos gate runs before the bvfreq smoother, kpp covers
 * blmc only transitively). Same arg shape as fesom_smooth_nod3D_kk + a label + step. */
void fesom_smooth_nod3D_kk_verify(fesom::Field &arr_fld, int n_smooth,
                                  const struct fesom_mesh *mesh, struct fesom_partit *p,
                                  std::size_t base, int nslab, std::size_t slab_stride,
                                  const char *label, int step_n);
#endif

/*
 * Pressure gradient force at elements (Phase 1: linfs + full cells).
 *   pgf_x[e][nz] = Σ_i ∇N_i^x * hpressure[nz][V_i(e)] / density_0
 *   pgf_y[e][nz] = Σ_i ∇N_i^y * hpressure[nz][V_i(e)] / density_0
 *
 * Mirror of pressure_force_4_linfs_fullcell (oce_ale_pressure_bv.F90:575-614).
 * Other variants (Shchepetkin / cubic-spline / partial cells / cavity)
 * deferred to later phases — namelist default is 'shchepetkin' but the
 * dispatcher picks linfs_fullcell when use_partial_cell=.false. AND
 * use_cavity_partial_cell=.false. (both true in Phase 1).
 */
void fesom_pressure_force_linfs_fullcell(const struct fesom_mesh *mesh,
                                         struct fesom_aux        *aux);

/*
 * M2.4 DEVICE kernel twin of fesom_pressure_force_linfs_fullcell (Kokkos
 * parallel_for over owned elements; the level loop inside the lambda; each
 * element writes only its own pgf_x/pgf_y → race-free, EOS-style — Serial AND
 * OpenMP bit-identical). INPUT hpressure must be device-current (the driver's
 * substep-2 rail does modify_host()+sync_device() on it — it was produced on the
 * device in substep 1, then sync_host'd + halo-exchanged on the host, L27); the
 * set-once mesh gradient_sca/elem_nodes/ulevels/nlevels are already device-current.
 * Marks pgf_x/pgf_y modify_device(). See docs/SYNC_MAP.md §2 row 2.
 */
/* M6.3 (Z6) — the zstar/zlevel PGF (which_pgf='shchepetkin', the module default the reference
 * uses). Self-contained on density_m_rho0 + the LIVE Z_3d_n/helem; makes ZERO hpressure
 * references (which is why hpressure is skipped entirely under zstar). One thread per element. */
void fesom_pressure_force_zxxxx_shchepetkin_kk(const struct fesom_mesh *mesh,
                                               struct fesom_aux        *aux);

void fesom_pressure_force_linfs_fullcell_kk(const struct fesom_mesh *mesh,
                                            struct fesom_aux        *aux);

/*
 * FESOM_KK_VERIFY=pgf in-binary per-kernel gate (the fesom_eos_verify shape):
 * snapshots the Kokkos pgf_x/pgf_y already in aux, runs the host C twin (full
 * overwrite from the intact hpressure → EOS-style, no capture-before needed),
 * diffs, asserts max|Δ|==0 on Serial, restores the Kokkos result. Non-intrusive.
 */
void fesom_pressure_force_verify(const struct fesom_mesh *mesh,
                                 struct fesom_aux        *aux,
                                 int step_n);

/*
 * Compute thermal expansion (sw_alpha) and saline contraction (sw_beta)
 * coefficients per node per level, from McDougall (1987). Mirror of
 * oce_ale_pressure_bv.F90:2751-2846 sw_alpha_beta. Outputs into
 * aux->sw_alpha and aux->sw_beta.
 *
 * Halo-exchanged at end (the caller may add an explicit exchange too).
 */
void fesom_compute_sw_alpha_beta(const struct fesom_tracers *tracers,
                                 const struct fesom_mesh    *mesh,
                                 struct fesom_aux           *aux);

/* M2.1 DEVICE kernel twin of fesom_compute_sw_alpha_beta (Kokkos parallel_for).
 * Production path; marks sw_alpha/sw_beta modify_device(). See docs/SYNC_MAP.md §1. */
void fesom_compute_sw_alpha_beta_kk(const struct fesom_tracers *tracers,
                                    const struct fesom_mesh    *mesh,
                                    struct fesom_aux           *aux);

/*
 * FESOM_KK_VERIFY=eos in-binary per-kernel gate: runs the host C twins and diffs
 * them against the Kokkos production result already in aux, reporting max|Δ| per
 * field and asserting max|Δ|==0 on the Serial backend. Diagnostic only (env-gated),
 * and non-intrusive — it restores the Kokkos result into aux before returning.
 * Call AFTER the *_kk kernels + their output sync_host(). step_n is for the log.
 */
void fesom_eos_verify(const struct fesom_tracers *tracers,
                      const struct fesom_mesh    *mesh,
                      struct fesom_aux           *aux,
                      int step_n);

#endif /* FESOM_EOS_H */
