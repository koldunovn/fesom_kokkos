// M0.2 Kokkos backend smoke: exercises the three primitives the port relies on —
// parallel_for, parallel_reduce, and a DualView host<->device round-trip.
// Prints the active execution space so we can confirm Serial / OpenMP / Cuda.
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <cmath>
#include <cstdio>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        std::printf("ExecSpace: %s\n", Kokkos::DefaultExecutionSpace::name());

        const int N = 1000000;
        Kokkos::View<double*> a("a", N);
        Kokkos::parallel_for("fill", N, KOKKOS_LAMBDA(const int i) { a(i) = 1.0 * i; });

        double sum = 0.0;
        Kokkos::parallel_reduce(
            "sum", N, KOKKOS_LAMBDA(const int i, double& s) { s += a(i); }, sum);
        const double expect = (double)(N - 1) * (double)N / 2.0;
        // tolerance: reduction order differs across backends (expected); ULP at ~5e11 is ~1e-4
        const bool sum_ok = std::fabs(sum - expect) < 1.0;
        std::printf("parallel_reduce sum=%.1f expect=%.1f %s\n", sum, expect,
                    sum_ok ? "OK" : "MISMATCH");

        Kokkos::DualView<double*> dv("dv", 5);
        auto h = dv.view_host();
        for (int i = 0; i < 5; ++i) h(i) = i + 0.5;
        dv.modify_host();
        dv.sync_device();
        auto d = dv.view_device();
        Kokkos::parallel_for("scale", 5, KOKKOS_LAMBDA(const int i) { d(i) *= 2.0; });
        dv.modify_device();
        dv.sync_host();
        const double got = dv.view_host()(2);  // (2 + 0.5) * 2 = 5.0
        const bool dv_ok = std::fabs(got - 5.0) < 1e-12;
        std::printf("DualView round-trip dv[2]=%.3f expect=5.000 %s\n", got,
                    dv_ok ? "OK" : "MISMATCH");

        rc = (sum_ok && dv_ok) ? 0 : 1;
    }
    Kokkos::finalize();
    std::printf("%s\n", rc == 0 ? "SMOKE PASS" : "SMOKE FAIL");
    return rc;
}
