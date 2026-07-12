/*
 * M6.1 Task 1.2 — COLUMN-CORE TWIN GATE.
 *
 * Runs the C oracle's integrate_tke and the Kokkos port's transcription on identical
 * synthetic columns and requires BIT-IDENTICAL outputs (tke_new, KappaM, KappaH, and all 13
 * diag slabs). This isolates the column core from the driver: if it passes, any later bit-id
 * failure is provably in the DRIVER (column assembly / halos / Av-Kv wiring), not the math.
 *
 * Columns are randomised over physically plausible ranges plus deliberate edge cases
 * (near-zero and negative Nsqr, zero shear, zero surface forcing, thin bottom layers,
 * nlev at both ends of the range).
 */
#include "fesom_cvmix_tke.hpp"    // the Kokkos port's transcription

#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

extern "C" {
void c_tke_init(void);
void c_tke_integrate(int nlev, double dtime, double rho_ref, double grav,
                     const double *dzw, const double *dzt, const double *tke_old,
                     const double *Ssqr, const double *Nsqr, const double *zerocol,
                     double forc_tke_surf, double forc_rho_surf,
                     double *tke_new, double *KappaM_out, double *KappaH_out,
                     double *diag_out);
}

static int bitcmp(const char *what, const double *a, const double *b, int n, int col)
{
    for (int k = 0; k < n; ++k) {
        if (std::memcmp(&a[k], &b[k], sizeof(double)) != 0) {
            std::printf("  MISMATCH col=%d %s[%d]: C=%.17g  KK=%.17g  (|d|=%.3g)\n",
                        col, what, k, a[k], b[k], std::fabs(a[k] - b[k]));
            return 1;
        }
    }
    return 0;
}

int main()
{
    c_tke_init();

    std::mt19937_64 rng(20260712);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const double dtime = 1800.0, rho_ref = 1030.0, grav = 9.81;
    const int NCOL = 4000;
    int bad = 0, checked = 0;

    for (int col = 0; col < NCOL; ++col) {
        /* nlev sweeps the whole plausible range incl. the degenerate small end */
        int nlev = 2 + (int)(U(rng) * 68.0);          /* 2..69  (CORE2 max 46, NG5 69) */
        if (col < 8) nlev = 2 + col % 4;              /* hammer the tiny-column edge cases */
        const int n = nlev + 1;

        std::vector<double> dzw(n, 0.0), dzt(n, 0.0), tke_old(n, 0.0);
        std::vector<double> Ssqr(n, 0.0), Nsqr(n, 0.0), zerocol(n, 0.0);

        for (int k = 0; k < nlev; ++k) dzw[k] = 1.0 + U(rng) * 300.0;      /* layer thickness */
        for (int k = 0; k < n; ++k)    dzt[k] = 0.5 + U(rng) * 300.0;      /* tracer spacing  */
        for (int k = 0; k < n; ++k)    tke_old[k] = U(rng) * 1e-3;
        for (int k = 0; k < n; ++k)    Ssqr[k] = U(rng) * 1e-4;
        for (int k = 0; k < n; ++k)    Nsqr[k] = (U(rng) - 0.25) * 1e-3;   /* incl. NEGATIVE */

        /* edge cases the ocean really produces */
        if (col % 7 == 0) for (int k = 0; k < n; ++k) Nsqr[k] = 0.0;       /* fully mixed     */
        if (col % 11 == 0) for (int k = 0; k < n; ++k) Ssqr[k] = 0.0;      /* no shear        */
        if (col % 13 == 0) for (int k = 0; k < n; ++k) tke_old[k] = 0.0;   /* cold start      */
        if (col % 17 == 0) for (int k = 0; k < n; ++k) Nsqr[k] = -1e-6;    /* unstable column */

        double forc_tke_surf = (col % 5 == 0) ? 0.0 : U(rng) * 1e-3;
        double forc_rho_surf = (U(rng) - 0.5) * 1e-6;

        std::vector<double> tkeC(n), kmC(n), khC(n), dC(13 * n, 0.0);
        std::vector<double> tkeK(n), kmK(n), khK(n), dK(13 * n, 0.0);

        c_tke_integrate(nlev, dtime, rho_ref, grav, dzw.data(), dzt.data(), tke_old.data(),
                        Ssqr.data(), Nsqr.data(), zerocol.data(),
                        forc_tke_surf, forc_rho_surf,
                        tkeC.data(), kmC.data(), khC.data(), dC.data());

        fesom_cvmix_tke_diag dg{};
        dg.Tbpr = dK.data() + 0*n;  dg.Tspr = dK.data() + 1*n;  dg.Tdif = dK.data() + 2*n;
        dg.Tdis = dK.data() + 3*n;  dg.Twin = dK.data() + 4*n;  dg.Tiwf = dK.data() + 5*n;
        dg.Tbck = dK.data() + 6*n;  dg.Ttot = dK.data() + 7*n;  dg.Lmix = dK.data() + 8*n;
        dg.Pr   = dK.data() + 9*n;  dg.int1 = dK.data() + 10*n; dg.int2 = dK.data() + 11*n;
        dg.int3 = dK.data() + 12*n;

        fesom_cvmix_integrate_tke<true>(nlev, dtime, rho_ref, grav, dzw.data(), dzt.data(),
                                        tke_old.data(), Ssqr.data(), Nsqr.data(),
                                        zerocol.data(), zerocol.data(), zerocol.data(),
                                        zerocol.data(), forc_tke_surf, forc_rho_surf, 0.0,
                                        tkeK.data(), kmK.data(), khK.data(), &dg);

        int e = 0;
        e += bitcmp("tke_new", tkeC.data(), tkeK.data(), n, col);
        e += bitcmp("KappaM",  kmC.data(),  kmK.data(),  n, col);
        e += bitcmp("KappaH",  khC.data(),  khK.data(),  n, col);
        static const char *dn[13] = {"Tbpr","Tspr","Tdif","Tdis","Twin","Tiwf","Tbck",
                                     "Ttot","Lmix","Pr","int1","int2","int3"};
        for (int j = 0; j < 13; ++j)
            e += bitcmp(dn[j], dC.data() + j*n, dK.data() + j*n, n, col);

        checked += (3 + 13) * n;
        if (e) { ++bad; if (bad > 3) break; }
    }

    std::printf("\ncolumns: %d   values compared: %d   columns with a mismatch: %d\n",
                NCOL, checked, bad);
    if (bad == 0) {
        std::printf("=== COLUMN-CORE TWIN GATE PASS — Kokkos integrate_tke is BIT-IDENTICAL "
                    "to the C oracle ===\n");
        return 0;
    }
    std::printf("=== COLUMN-CORE TWIN GATE FAIL ===\n");
    return 1;
}
