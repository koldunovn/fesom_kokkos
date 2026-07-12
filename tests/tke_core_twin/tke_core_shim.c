/* C-side shim: expose the ORACLE's column core under names that cannot collide with the
 * Kokkos port's template. Compiled as C against the C oracle's own headers. */
#include "fesom_cvmix_tke.h"
#include <stdlib.h>

void c_tke_init(void)
{
    /* the C's own call-site values (fesom_tke.c:227-241) */
    fesom_cvmix_init_tke(/*c_k*/0.1, /*c_eps*/0.7, /*cd*/3.75, /*alpha_tke*/30.0,
                         /*mxl_min*/1.0e-8, /*kappaM_min*/0.0, /*kappaM_max*/100.0,
                         /*tke_mxl_choice*/2, /*use_ubound_dirichlet*/0,
                         /*use_lbound_dirichlet*/0, /*only_tke*/1, /*l_lc*/0, /*clc*/0.3,
                         /*tke_min*/1.0e-6, /*tke_surf_min*/1.0e-4);
}

void c_tke_integrate(int nlev, double dtime, double rho_ref, double grav,
                     const double *dzw, const double *dzt, const double *tke_old,
                     const double *Ssqr, const double *Nsqr, const double *zerocol,
                     double forc_tke_surf, double forc_rho_surf,
                     double *tke_new, double *KappaM_out, double *KappaH_out,
                     double *diag_out /* 13*(nlev+1) or NULL */)
{
    if (!diag_out) {
        fesom_cvmix_integrate_tke(nlev, dtime, rho_ref, grav, dzw, dzt, tke_old, Ssqr, Nsqr,
                                  zerocol, zerocol, zerocol, zerocol,
                                  forc_tke_surf, forc_rho_surf, 0.0,
                                  tke_new, KappaM_out, KappaH_out, NULL);
        return;
    }
    int n = nlev + 1;
    fesom_cvmix_tke_diag d;
    d.Tbpr = diag_out + 0*n;  d.Tspr = diag_out + 1*n;  d.Tdif = diag_out + 2*n;
    d.Tdis = diag_out + 3*n;  d.Twin = diag_out + 4*n;  d.Tiwf = diag_out + 5*n;
    d.Tbck = diag_out + 6*n;  d.Ttot = diag_out + 7*n;  d.Lmix = diag_out + 8*n;
    d.Pr   = diag_out + 9*n;  d.int1 = diag_out + 10*n; d.int2 = diag_out + 11*n;
    d.int3 = diag_out + 12*n;
    fesom_cvmix_integrate_tke(nlev, dtime, rho_ref, grav, dzw, dzt, tke_old, Ssqr, Nsqr,
                              zerocol, zerocol, zerocol, zerocol,
                              forc_tke_surf, forc_rho_surf, 0.0,
                              tke_new, KappaM_out, KappaH_out, &d);
}
