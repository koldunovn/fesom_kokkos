/* M10 T4 — host-side solver testbed (plan docs/plans/20260805-m10-ssh-solvers.md, Layer 0.3).
 *
 * A serial (npes==1) SPD fixture with KNOWN spectrum on which every M10 solver can be checked
 * against theory before it ever meets the ocean: a 2-D 5-point Laplacian on an n×n grid, built
 * directly in the model's own `fesom_ssh_stiff` CSR shape so the production kernels can run on
 * it unchanged as each solver lands (T5b–T8b add their assertions here).
 *
 * Why a Laplacian: its eigenvalues are exactly
 *     λ(p,q) = 4 − 2cos(pπ/(n+1)) − 2cos(qπ/(n+1)),   p,q = 1..n
 * so κ, the Chebyshev convergence bound, and the Lanczos targets are all analytic — the
 * checker compares against arithmetic, not against another implementation.
 *
 * Scaffold contents (this file, T4):
 *   - the fixture + its analytic spectrum
 *   - a reference PCG (the comparator's ground truth)
 *   - the α/β-sequence comparator, with an injected-perturbation self-test
 *   - the Chebyshev-rate checker
 *   - the symmetry-defect metric + the D^{-1/2}CD^{-1/2} symmetriser (derivations §0.5)
 *   - a direct test of the σ-recurrence identity that cg2/pipecg/oati inherit (§0.4)
 *
 * Runs on the login node in <1 s. `ctest -R ssh_solvers` or ./test_ssh_solvers.
 */
#include "fesom_ssh.h"
#include "fesom_types.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

static int g_fail = 0;
static void check(bool ok, const char *what, double got = 0.0, double want = 0.0)
{
    if (ok) { printf("  ok    %s\n", what); return; }
    printf("  FAIL  %s   (got %.6e, want %.6e)\n", what, got, want);
    ++g_fail;
}

/* ---------------------------------------------------------------- fixture */

/* 2-D 5-point Laplacian on an n×n interior grid, Dirichlet: N = n² unknowns.
 * Row layout matches the model's convention: the DIAGONAL IS AT OFFSET 0 of each row
 * (fesom_ssh.h documents this; the preconditioner and every SpMV rely on it). */
struct Fixture {
    int n = 0, N = 0;
    std::vector<int>    rowptr, colind;
    std::vector<real_t> values, pr_values;

    /* `mass_contrast` > 0 adds a per-node positive mass term m_i to the diagonal, spanning
     * [1, 1+mass_contrast] across the grid. It mimics the SSH stiffness matrix's
     * `areasvol/dt` diagonal, which varies by orders of magnitude across an ocean mesh.
     * With mass_contrast == 0 the diagonal is CONSTANT and the MITgcm preconditioner is
     * accidentally symmetric — that variant cannot exercise derivations §0.4 at all, which
     * is exactly why the variable-diagonal fixture exists. A stays symmetric either way. */
    void build(int n_, double mass_contrast = 0.0)
    {
        n = n_; N = n * n;
        rowptr.assign((size_t)N + 1, 0);
        colind.clear(); values.clear();
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const int row = j * n + i;
                const double fx = (double)i / (double)(n - 1), fy = (double)j / (double)(n - 1);
                const double mass = mass_contrast > 0.0
                    ? 1.0 + mass_contrast * (0.5 * (fx + fy)) : 0.0;
                colind.push_back(row);  values.push_back(4.0 + mass);   /* diagonal FIRST */
                if (i > 0)     { colind.push_back(row - 1); values.push_back(-1.0); }
                if (i < n - 1) { colind.push_back(row + 1); values.push_back(-1.0); }
                if (j > 0)     { colind.push_back(row - n); values.push_back(-1.0); }
                if (j < n - 1) { colind.push_back(row + n); values.push_back(-1.0); }
                rowptr[(size_t)row + 1] = (int)colind.size();
            }
    }

    /* the model's MITgcm-class preconditioner, built by the SAME formula as
     * fesom_ssh_preconditioner (src/fesom_ssh.cpp:265-275) — non-symmetric by construction */
    void build_pr()
    {
        pr_values.assign(colind.size(), 0.0);
        std::vector<real_t> d((size_t)N);
        for (int r = 0; r < N; ++r) d[(size_t)r] = values[(size_t)rowptr[r]];
        for (int r = 0; r < N; ++r) {
            const real_t dr = d[(size_t)r];
            pr_values[(size_t)rowptr[r]] = 1.0 / dr;
            for (int k = rowptr[r] + 1; k < rowptr[r + 1]; ++k)
                pr_values[(size_t)k] = -0.5 * (values[(size_t)k] / dr)
                                     / (dr + d[(size_t)colind[(size_t)k]]);
        }
    }

    /* derivations §0.5: p̃r[i,j] = pr[i,j]·sqrt(d_i/d_j) — symmetric, same spectrum */
    std::vector<real_t> symmetrised_pr() const
    {
        std::vector<real_t> s(pr_values);
        std::vector<real_t> d((size_t)N);
        for (int r = 0; r < N; ++r) d[(size_t)r] = values[(size_t)rowptr[r]];
        for (int r = 0; r < N; ++r)
            for (int k = rowptr[r] + 1; k < rowptr[r + 1]; ++k)
                s[(size_t)k] = pr_values[(size_t)k]
                             * sqrt(d[(size_t)r] / d[(size_t)colind[(size_t)k]]);
        return s;
    }

    void spmv(const std::vector<real_t> &vals,
              const std::vector<real_t> &v, std::vector<real_t> &y) const
    {
        for (int r = 0; r < N; ++r) {
            real_t s = 0.0;
            for (int k = rowptr[r]; k < rowptr[r + 1]; ++k) s += vals[(size_t)k] * v[(size_t)colind[(size_t)k]];
            y[(size_t)r] = s;
        }
    }

    /* max |X_ij − X_ji| / max |X_ij| over off-diagonals — the lab's --sym-check metric */
    double sym_defect(const std::vector<real_t> &vals) const
    {
        double mxd = 0.0, mx = 0.0;
        for (int r = 0; r < N; ++r)
            for (int k = rowptr[r]; k < rowptr[r + 1]; ++k) {
                const int c = colind[(size_t)k];
                if (c == r) continue;
                mx = std::max(mx, fabs((double)vals[(size_t)k]));
                double rev = 0.0;
                for (int m = rowptr[c]; m < rowptr[c + 1]; ++m)
                    if (colind[(size_t)m] == r) { rev = vals[(size_t)m]; break; }
                mxd = std::max(mxd, fabs((double)vals[(size_t)k] - rev));
            }
        return mx > 0.0 ? mxd / mx : 0.0;
    }

    /* analytic extreme eigenvalues of the 5-point Laplacian */
    double lam_min() const { return 4.0 - 4.0 * cos(M_PI / (n + 1)); }
    double lam_max() const { return 4.0 + 4.0 * cos(M_PI / (n + 1)); }
};

static double dot(const std::vector<real_t> &a, const std::vector<real_t> &b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += (double)a[i] * (double)b[i];
    return s;
}

/* -------------------------------------------------- reference PCG + trace */

struct Trace { std::vector<double> alpha, beta, sigma_true, sigma_rec, resid; };

/* Reference preconditioned CG — the ground truth every variant is compared against.
 * Additionally records the CG-CG σ recurrence alongside the true (p,Ap) so the
 * derivations §0.4 identity can be asserted directly. */
static Trace ref_pcg(const Fixture &F, const std::vector<real_t> &PR,
                     const std::vector<real_t> &b, std::vector<real_t> &x,
                     int maxiter, double tol)
{
    Trace T;
    const int N = F.N;
    std::vector<real_t> r(N), u(N), p(N), ap(N), w(N);
    F.spmv(F.values, x, ap);
    for (int i = 0; i < N; ++i) r[(size_t)i] = b[(size_t)i] - ap[(size_t)i];
    F.spmv(PR, r, u);
    p = u;
    double gamma = dot(r, u), sigma_rec = 0.0, beta = 0.0;
    const double bnorm = sqrt(dot(b, b));
    for (int it = 1; it <= maxiter; ++it) {
        F.spmv(F.values, p, ap);
        const double sigma_true = dot(p, ap);
        if (it == 1) sigma_rec = sigma_true;          /* p₀ = u₀ ⇒ σ₀ = δ₀ exactly */
        const double alpha = gamma / sigma_true;
        T.alpha.push_back(alpha);
        T.sigma_true.push_back(sigma_true);
        T.sigma_rec.push_back(sigma_rec);
        for (int i = 0; i < N; ++i) {
            x[(size_t)i] += alpha * p[(size_t)i];
            r[(size_t)i] -= alpha * ap[(size_t)i];
        }
        const double rn = sqrt(dot(r, r));
        T.resid.push_back(rn);
        F.spmv(PR, r, u);
        const double gamma_new = dot(r, u);
        beta = gamma_new / gamma;
        T.beta.push_back(beta);
        gamma = gamma_new;
        F.spmv(F.values, u, w);                       /* w = A u  ⇒ δ = (w,u) */
        sigma_rec = dot(w, u) - beta * beta * sigma_rec;
        for (int i = 0; i < N; ++i) p[(size_t)i] = u[(size_t)i] + beta * p[(size_t)i];
        if (rn <= tol * bnorm) break;
    }
    return T;
}

/* α/β-sequence comparator: max relative difference over the first `k` iterations.
 * This is the instrument T5b–T7 use to certify "reproduces reference PCG to rounding". */
static double seq_maxreldiff(const std::vector<double> &a, const std::vector<double> &b, int k)
{
    double m = 0.0;
    const int n = std::min((int)std::min(a.size(), b.size()), k);
    for (int i = 0; i < n; ++i) {
        const double den = std::max(fabs(a[(size_t)i]), 1e-300);
        m = std::max(m, fabs(a[(size_t)i] - b[(size_t)i]) / den);
    }
    return m;
}

/* Chebyshev-rate checker: after k P-CSI-class iterations on a spectrum in [ν,µ], the
 * error-reduction factor must not exceed 2·((√κ−1)/(√κ+1))^k (Golub–Varga). */
static double chebyshev_bound(double nu, double mu, int k)
{
    const double kappa = mu / nu;
    const double q = (sqrt(kappa) - 1.0) / (sqrt(kappa) + 1.0);
    return 2.0 * pow(q, (double)k);
}

/* A bare P-CSI on the fixture (derivations §4.3) — exercises the recurrence that T8b will
 * implement in the model, so the ω resolution (T-6) is checked by convergence, not by eye.
 *
 * Returns the A-NORM ERROR reduction ‖x_k − x*‖_A / ‖x_0 − x*‖_A. That is the quantity the
 * Chebyshev theory bounds; a residual ratio is NOT (it differs by up to κ(M⁻¹A), which on
 * this fixture is enough to make a correct implementation look like it misses the bound).
 * `omega_variant`: 0 = the derived 1/(4α²) coefficient, 1 = the misread (1/4)α² (T-6 guard). */
static double pcsi_energy_rate(const Fixture &F, const std::vector<real_t> &PR,
                               const std::vector<real_t> &b, const std::vector<real_t> &xstar,
                               double nu, double mu, int iters, int omega_variant = 0)
{
    const int N = F.N;
    const double gamma = 0.5 * (mu + nu);
    const double alpha = 2.0 / (mu - nu);
    const double coef  = omega_variant == 0 ? 1.0 / (4.0 * alpha * alpha)
                                            : 0.25 * alpha * alpha;
    std::vector<real_t> x(N, 0.0), r(N), rp(N), dx(N), ax(N), e(N), ae(N);
    auto energy = [&](const std::vector<real_t> &xx) {
        for (int i = 0; i < N; ++i) e[(size_t)i] = xx[(size_t)i] - xstar[(size_t)i];
        F.spmv(F.values, e, ae);
        return sqrt(fabs(dot(e, ae)));
    };
    const double e0 = energy(x);
    F.spmv(F.values, x, ax);
    for (int i = 0; i < N; ++i) r[(size_t)i] = b[(size_t)i] - ax[(size_t)i];
    F.spmv(PR, r, rp);
    for (int i = 0; i < N; ++i) { dx[(size_t)i] = rp[(size_t)i] / gamma; x[(size_t)i] += dx[(size_t)i]; }
    F.spmv(F.values, x, ax);
    for (int i = 0; i < N; ++i) r[(size_t)i] = b[(size_t)i] - ax[(size_t)i];
    double omega = 2.0 / gamma;
    for (int k = 1; k <= iters; ++k) {
        omega = 1.0 / (gamma - coef * omega);                      /* T-6 resolution */
        F.spmv(PR, r, rp);
        for (int i = 0; i < N; ++i)
            dx[(size_t)i] = omega * rp[(size_t)i] + (gamma * omega - 1.0) * dx[(size_t)i];
        for (int i = 0; i < N; ++i) x[(size_t)i] += dx[(size_t)i];
        F.spmv(F.values, x, ax);
        for (int i = 0; i < N; ++i) r[(size_t)i] = b[(size_t)i] - ax[(size_t)i];
    }
    return energy(x) / e0;
}

/* power iteration for λmax of PR·A (used to bracket the preconditioned spectrum) */
static double lam_max_prec(const Fixture &F, const std::vector<real_t> &PR, int iters)
{
    const int N = F.N;
    std::vector<real_t> v(N), t(N), w(N);
    for (int i = 0; i < N; ++i) v[(size_t)i] = 1.0 + 0.001 * (i % 17);
    double lam = 0.0;
    for (int k = 0; k < iters; ++k) {
        const double nrm = sqrt(dot(v, v));
        for (int i = 0; i < N; ++i) v[(size_t)i] /= nrm;
        F.spmv(F.values, v, t);
        F.spmv(PR, t, w);
        lam = dot(v, w);
        v = w;
    }
    return lam;
}

/* power iteration for λmax of the PRECONDITIONER ALONE (the similarity claim of §0.5 is
 * about M⁻¹, not about M⁻¹A — see the note in the derivations doc). */
static double lam_max_op(const Fixture &F, const std::vector<real_t> &OP, int iters)
{
    const int N = F.N;
    std::vector<real_t> v(N), w(N);
    for (int i = 0; i < N; ++i) v[(size_t)i] = 1.0 + 0.001 * (i % 17);
    double lam = 0.0;
    for (int k = 0; k < iters; ++k) {
        const double nrm = sqrt(dot(v, v));
        for (int i = 0; i < N; ++i) v[(size_t)i] /= nrm;
        F.spmv(OP, v, w);
        lam = dot(v, w);
        v = w;
    }
    return lam;
}

/* --- Lanczos on M⁻¹A in the M⁻¹ inner product + Sturm bisection on T_m ---------------
 * This is the serial prototype of the T8a in-model estimator, and it doubles as the T-5
 * guard: `sqrt_start=false` reproduces [P] App. C's `q₁ = r₀/(r₀ᵀs₀)` (no square root),
 * which must visibly fail to recover the known spectrum.
 * Full reorthogonalisation — this is a 1024-unknown fixture, not the hot path. */
struct Lanczos { std::vector<double> alpha, beta; };

static Lanczos lanczos_prec(const Fixture &F, const std::vector<real_t> &PR,
                            int m, bool sqrt_start = true)
{
    const int N = F.N;
    Lanczos L;
    std::vector<real_t> q(N), qprev(N, 0.0), r(N), p(N), s(N);
    std::vector<std::vector<real_t>> Q;
    for (int i = 0; i < N; ++i) r[(size_t)i] = 1.0 + 0.37 * sin(0.7 * i);
    F.spmv(PR, r, s);
    const double n0 = dot(r, s);
    const double scale = sqrt_start ? sqrt(n0) : n0;         /* T-5: the square root */
    for (int i = 0; i < N; ++i) q[(size_t)i] = r[(size_t)i] / scale;
    double beta_prev = 0.0;
    for (int j = 0; j < m; ++j) {
        F.spmv(PR, q, p);                                    /* p = M⁻¹ q */
        F.spmv(F.values, p, r);                              /* r = A M⁻¹ q */
        for (int i = 0; i < N; ++i) r[(size_t)i] -= beta_prev * qprev[(size_t)i];
        const double a = dot(p, r);
        L.alpha.push_back(a);
        for (int i = 0; i < N; ++i) r[(size_t)i] -= a * q[(size_t)i];
        /* full reorthogonalisation in the M⁻¹ inner product */
        for (const auto &qq : Q) {
            F.spmv(PR, r, s);
            const double c = dot(qq, s);
            for (int i = 0; i < N; ++i) r[(size_t)i] -= c * qq[(size_t)i];
        }
        F.spmv(PR, r, s);
        const double b = sqrt(fabs(dot(r, s)));
        if (b < 1e-14) break;
        L.beta.push_back(b);
        Q.push_back(q);
        qprev = q;
        for (int i = 0; i < N; ++i) q[(size_t)i] = r[(size_t)i] / b;
        beta_prev = b;
    }
    return L;
}

/* number of eigenvalues of the symmetric tridiagonal T strictly below x (Sturm sequence) */
static int sturm_count(const Lanczos &L, double x)
{
    const int m = (int)L.alpha.size();
    int cnt = 0;
    double d = L.alpha[0] - x;
    if (d < 0.0) ++cnt;
    for (int i = 1; i < m; ++i) {
        if (fabs(d) < 1e-300) d = 1e-300;
        d = (L.alpha[(size_t)i] - x) - L.beta[(size_t)i - 1] * L.beta[(size_t)i - 1] / d;
        if (d < 0.0) ++cnt;
    }
    return cnt;
}

static void tridiag_extremes(const Lanczos &L, double *lo, double *hi)
{
    const int m = (int)L.alpha.size();
    double a = L.alpha[0], b = L.alpha[0];
    for (int i = 0; i < m; ++i) {                            /* Gershgorin bracket */
        const double bl = (i > 0)     ? L.beta[(size_t)i - 1] : 0.0;
        const double br = (i < m - 1) ? L.beta[(size_t)i]     : 0.0;
        a = std::min(a, L.alpha[(size_t)i] - bl - br);
        b = std::max(b, L.alpha[(size_t)i] + bl + br);
    }
    auto bisect = [&](int want) {                            /* smallest x with count >= want */
        double x0 = a, x1 = b;
        for (int it = 0; it < 200; ++it) {
            const double xm = 0.5 * (x0 + x1);
            if (sturm_count(L, xm) >= want) x1 = xm; else x0 = xm;
        }
        return 0.5 * (x0 + x1);
    };
    *lo = bisect(1);
    *hi = bisect(m);
}

int main(void)
{
    printf("=== M10 solver testbed (T4 scaffold) ===\n");

    Fixture F;
    F.build(32);                       /* 1024 unknowns */
    F.build_pr();
    printf("fixture: %d x %d Laplacian, N=%d, nnz=%d\n", F.n, F.n, F.N, (int)F.colind.size());

    /* ---- 1. fixture sanity: analytic spectrum, symmetry of A ---- */
    check(F.sym_defect(F.values) < 1e-15, "A is symmetric (defect ~ 0)", F.sym_defect(F.values), 0.0);
    {
        const double lo = F.lam_min(), hi = F.lam_max();
        check(lo > 0.0 && hi < 8.0 + 1e-12, "analytic Laplacian spectrum in (0,8)", lo, hi);
    }

    /* ---- 2. reference PCG converges ---- */
    std::vector<real_t> b((size_t)F.N), x((size_t)F.N, 0.0);
    for (int i = 0; i < F.N; ++i) b[(size_t)i] = sin(0.1 * i) + 0.5 * cos(0.03 * i);
    Trace T0 = ref_pcg(F, F.pr_values, b, x, 500, 1e-10);
    check(T0.resid.back() <= 1e-10 * sqrt(dot(b, b)) * 1.0001,
          "reference PCG converges to 1e-10", T0.resid.back(), 1e-10 * sqrt(dot(b, b)));
    printf("        (reference PCG: %d iterations)\n", (int)T0.resid.size());

    /* ---- 3. comparator self-test: it must DETECT an injected 1e-9 perturbation ---- */
    {
        std::vector<double> perturbed = T0.alpha;
        perturbed[3] *= (1.0 + 1e-9);
        const double d_same = seq_maxreldiff(T0.alpha, T0.alpha, 20);
        const double d_pert = seq_maxreldiff(T0.alpha, perturbed, 20);
        check(d_same == 0.0, "comparator: identical sequences give 0", d_same, 0.0);
        check(d_pert > 5e-10, "comparator: DETECTS an injected 1e-9 perturbation", d_pert, 1e-9);
    }

    /* ---- 4. derivations §0.4/§0.5: the σ recurrence needs a symmetric M ----
     * Run on the VARIABLE-diagonal fixture: the uniform Laplacian has a constant diagonal,
     * which makes the MITgcm preconditioner accidentally symmetric and hides the effect. */
    {
        Fixture V;
        V.build(32, 200.0);        /* diagonal spans 4 → 204: an ocean-mesh-like contrast */
        V.build_pr();
        std::vector<real_t> bv((size_t)V.N);
        for (int i = 0; i < V.N; ++i) bv[(size_t)i] = sin(0.1 * i) + 0.5 * cos(0.03 * i);

        const double defect_orig = V.sym_defect(V.pr_values);
        std::vector<real_t> prs = V.symmetrised_pr();
        const double defect_sym = V.sym_defect(prs);
        printf("        [variable-diagonal fixture, d in [4,204]]\n");
        printf("        pr symmetry defect: as-built %.3e, symmetrised %.3e\n",
               defect_orig, defect_sym);
        check(defect_orig > 1e-3,
              "variable-diagonal fixture DOES expose the asymmetry (else the test is vacuous)",
              defect_orig, 1e-3);
        /* M16: rounding-level bound scales with the working precision (SP measured 7.9e-8) */
        check(defect_sym < (sizeof(real_t) == 4 ? 1e-6 : 1e-14), "symmetrised preconditioner IS symmetric (rounding level)", defect_sym, 0.0);

        /* §0.5 similarity: D^{1/2}(D⁻¹C)D^{-1/2} = D^{-1/2}CD^{-1/2}, so M̃⁻¹ and M⁻¹ have
         * IDENTICAL spectra. That statement is about the PRECONDITIONER; the preconditioned
         * operator M⁻¹A is NOT similar to M̃⁻¹A, and its extremes shift slightly (measured). */
        const double lm_op_orig = lam_max_op(V, V.pr_values, 500);
        const double lm_op_sym  = lam_max_op(V, prs,         500);
        printf("        lambda_max(M^-1)  : as-built %.9e, symmetrised %.9e  (similarity claim)\n",
               lm_op_orig, lm_op_sym);
        check(fabs(lm_op_orig - lm_op_sym) / fabs(lm_op_orig) < 1e-6,
              "symmetrisation preserves the PRECONDITIONER spectrum (similar matrices)",
              fabs(lm_op_orig - lm_op_sym) / fabs(lm_op_orig), 0.0);

        const double lmax_orig = lam_max_prec(V, V.pr_values, 400);
        const double lmax_sym  = lam_max_prec(V, prs,         400);
        printf("        lambda_max(M^-1 A): as-built %.9f, symmetrised %.9f  (shift %.2f %%)\n",
               lmax_orig, lmax_sym, 100.0 * fabs(lmax_orig - lmax_sym) / lmax_orig);
        check(fabs(lmax_orig - lmax_sym) / lmax_orig < 0.05,
              "the preconditioned operator's lambda_max shifts by < 5 % (same convergence class)",
              fabs(lmax_orig - lmax_sym) / lmax_orig, 0.05);

        /* σ_true vs σ_recurred, both preconditioners — THE §0.4 assertion */
        auto worst_drift = [](const Trace &T) {
            double m = 0.0;
            for (size_t i = 1; i < T.sigma_true.size(); ++i)
                m = std::max(m, fabs(T.sigma_true[i] - T.sigma_rec[i]) / fabs(T.sigma_true[i]));
            return m;
        };
        std::vector<real_t> x1((size_t)V.N, 0.0), x2((size_t)V.N, 0.0);
        Trace Ta = ref_pcg(V, V.pr_values, bv, x1, 300, 1e-10);
        Trace Tb = ref_pcg(V, prs,          bv, x2, 300, 1e-10);
        const double da = worst_drift(Ta), db = worst_drift(Tb);
        printf("        sigma drift: as-built %.3e (%d it), symmetrised %.3e (%d it)\n",
               da, (int)Ta.resid.size(), db, (int)Tb.resid.size());
        check(db < (sizeof(real_t) == 4 ? 1e-6 : 1e-9), "symmetric M: the sigma recurrence is EXACT (rounding level)", db, 0.0);
        check(da > 1e3 * std::max(db, 1e-15),
              "NON-symmetric M: the sigma recurrence DRIFTS (derivations sec 0.4 confirmed)",
              da, db);
    }

    /* ---- 5. Chebyshev rate: P-CSI must meet the Golub-Varga bound (T-6 ω resolution) ----
     * Measured in the A-norm of the ERROR, which is what the theory bounds. x* comes from a
     * reference PCG driven to 1e-14. */
    {
        std::vector<real_t> prs = F.symmetrised_pr();
        std::vector<real_t> xstar((size_t)F.N, 0.0);
        ref_pcg(F, prs, b, xstar, 2000, 1e-14);

        /* [ν,µ] from Lanczos on M̃⁻¹A, then the SAFE margins (deflate ν, inflate µ) — the
         * spectrum MUST lie inside, otherwise the Chebyshev polynomial is not small where
         * the error lives and the bound simply does not apply. */
        Lanczos L = lanczos_prec(F, prs, 120);
        double th_lo = 0.0, th_hi = 0.0;
        tridiag_extremes(L, &th_lo, &th_hi);
        const double nu = th_lo * 0.90, mu = th_hi * 1.10;
        printf("        Lanczos(120) on M^-1 A: theta in [%.6e, %.6e] -> [nu,mu] = [%.6e, %.6e], kappa=%.1f\n",
               th_lo, th_hi, nu, mu, mu / nu);
        check(th_lo > 0.0 && th_hi > th_lo, "Lanczos returns a sane positive interval", th_lo, th_hi);
        for (int k : {10, 20, 40}) {
            const double got  = pcsi_energy_rate(F, prs, b, xstar, nu, mu, k);
            const double want = chebyshev_bound(nu, mu, k);
            printf("        pcsi k=%2d: A-norm error reduction %.4e, Chebyshev bound %.4e\n",
                   k, got, want);
            char nm[72]; snprintf(nm, sizeof nm, "pcsi meets the Chebyshev bound at k=%d", k);
            check(got <= want, nm, got, want);
        }
        /* T-6 guard: the MISREAD coefficient must visibly fail to meet the bound. If both
         * readings converged at the theoretical rate, the resolution would not matter — this
         * assertion is what makes §4.2 a decision rather than a preference. */
        const double good = pcsi_energy_rate(F, prs, b, xstar, nu, mu, 40, 0);
        const double bad  = pcsi_energy_rate(F, prs, b, xstar, nu, mu, 40, 1);
        const double want = chebyshev_bound(nu, mu, 40);
        printf("        T-6 guard @k=40: 1/(4a^2) -> %.4e (bound %.4e) | (1/4)a^2 -> %.4e\n",
               good, want, bad);
        check(good <= want, "T-6: the DERIVED omega meets the bound", good, want);
        check(bad > want, "T-6: the MISREAD omega does NOT meet the bound", bad, want);
    }

    /* ---- 6. Ritz values converge OUTWARD with m — this is what justifies the safe margin
     * directions (deflate ν, inflate µ) that T8a will use. Tested against ITSELF at growing
     * m (interlacing), not against a power iteration: the Euclidean Rayleigh quotient of a
     * non-self-adjoint-in-that-inner-product operator is the WORSE instrument here, and
     * checking a good estimator against a bad one only measures the bad one. */
    {
        std::vector<real_t> prs = F.symmetrised_pr();
        double lo[3], hi[3];
        const int ms[3] = { 20, 60, 120 };
        for (int t = 0; t < 3; ++t) {
            Lanczos L = lanczos_prec(F, prs, ms[t], true);
            tridiag_extremes(L, &lo[t], &hi[t]);
            printf("        Lanczos m=%3d: theta_min %.6e  theta_max %.6f\n", ms[t], lo[t], hi[t]);
        }
        check(hi[0] <= hi[1] * (1 + 1e-9) && hi[1] <= hi[2] * (1 + 1e-9),
              "theta_max is non-decreasing in m (converges to lambda_max from BELOW -> inflate mu)",
              hi[0], hi[2]);
        check(lo[0] >= lo[1] * (1 - 1e-9) && lo[1] >= lo[2] * (1 - 1e-9),
              "theta_min is non-increasing in m (converges to lambda_min from ABOVE -> deflate nu)",
              lo[0], lo[2]);
    }

    /* ---- 7. T-5: the Lanczos start vector needs the SQUARE ROOT of the inner product ----
     * [P] App. C prints `q1 = r0/(r0^T s0)`. The missing square root corrupts exactly one
     * entry of T — α₁ — because every later vector is renormalised properly. So the damage
     * is largest at SMALL m, which is precisely the regime [P] recommends and the regime
     * FESOM_PCSI_LANCZOS (default 30) will run in. Measured at both ends here. */
    {
        std::vector<real_t> prs = F.symmetrised_pr();
        Lanczos ref = lanczos_prec(F, prs, 200, true);
        double lo_ref = 0, hi_ref = 0;
        tridiag_extremes(ref, &lo_ref, &hi_ref);
        printf("        T-5 reference (m=200, rooted): theta in [%.6e, %.6f]\n", lo_ref, hi_ref);
        for (int m : {8, 30, 120}) {
            Lanczos a = lanczos_prec(F, prs, m, true);
            Lanczos c = lanczos_prec(F, prs, m, false);
            double la, ha, lc, hc;
            tridiag_extremes(a, &la, &ha);
            tridiag_extremes(c, &lc, &hc);
            printf("        T-5 m=%3d: rooted theta_min %.6e | un-rooted %.6e  (ratio %.3g)\n",
                   m, la, lc, lc / la);
            if (m == 8) {
                check(fabs(lc - la) / la > 0.05,
                      "T-5: at small m the un-rooted start VISIBLY corrupts theta_min",
                      fabs(lc - la) / la, 0.05);
                check(fabs(la - lo_ref) / lo_ref < fabs(lc - lo_ref) / lo_ref,
                      "T-5: the rooted start is closer to the converged answer at small m",
                      fabs(la - lo_ref) / lo_ref, fabs(lc - lo_ref) / lo_ref);
            }
        }
    }

    printf("=== %s (%d failure%s) ===\n", g_fail ? "FAIL" : "PASS", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
